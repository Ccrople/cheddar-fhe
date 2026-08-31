#pragma once

#include <complex>
#include <iosfwd>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include <rmm/device_uvector.hpp>

#include "core/Container.h"
#include "core/DeviceVector.h"
#include "core/NTT.h"
#include "core/NPInfo.h"
#include "core/Parameter.h"
#include "core/Pcmm.h"

namespace cheddar {

/**
 * @brief The encoding unit, on the device.
 *
 * `Encoder` states outright that it "is not very optimized as it is not of our
 * highest priority to optimize offline operations", and it is not: every one
 * of its three encodings is a single host thread walking the message, and the
 * RNS decomposition inside it is a GMP `mpz_mod` per (value, prime) pair. At
 * the shapes this tree actually runs that is not an offline cost any more --
 * a layer's `pcmm: convert weights` row is 86.88 s of exactly this, against
 * 41.5 s for the whole encrypted layer -- so this class is the same three
 * encodings with every stage on the GPU.
 *
 * ## The three stages, and why they are public
 *
 * An encoding is a pipeline, and the question this class was built to answer
 * is which stage costs what. So the stages are public and take device
 * pointers, and `Encode`/`EncodeCoeff`/`EncodeMatrix` are the compositions:
 *
 *   1. `SpecialIFFT`     -- the iDFT of the slot encoding, in double
 *                           precision, two shared-memory passes.
 *   2. `FftToCoeff`      -- the bit reversal, the 1/S normalisation and the
 *                           fold that places a slot value at its coefficient.
 *   3. `RnsDecompose`    -- round(value * scale) into RNS limbs, optionally in
 *                           Montgomery form. This is the stage the host spends
 *                           GMP on.
 *
 * and then the existing device NTT closes it, unchanged.
 *
 * ## What replaces BigInt
 *
 * `RealVectorToPlaintext` builds a `BigInt` per coefficient and reduces it
 * modulo every prime. Nothing here needs arbitrary precision: `round(v)` of a
 * double IS a double, so its exact value is `mantissa * 2^e` with a 53-bit
 * mantissa, and
 *
 *     (m * 2^e) mod p  =  ((m mod p) * (2^e mod p)) mod p
 *
 * is exact in 64-bit arithmetic for any p below the word width. The common
 * case -- every value this pipeline encodes -- has |round(v)| < 2^64 and takes
 * a single Barrett reduction; the general case costs one modular exponentiation
 * of a small exponent. So the whole of `BigInt` disappears rather than being
 * ported, and the result is bit-identical to the host's, which is what
 * `EncodeGpuTest` checks limb by limb.
 *
 * ## Shared memory
 *
 * Both kernel families use it, for the two different reasons a kernel does.
 * The FFT stages stage a chunk of the transform in shared memory and run every
 * stage that stays inside the chunk without touching global memory again --
 * the same split the NTT makes between its two phases, at the same place, for
 * the same reason. `RnsDecompose` uses it as a broadcast table: the prime, its
 * Barrett constant and `2^word_bits mod p` are the same for all 256 threads of
 * a block and the Barrett constant costs a 64-bit division to make, so the
 * first `num_primes` threads make them once per block and everyone reads them
 * out of shared memory.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class GpuEncoder {
 private:
  const Parameter<word> &param_;
  const NTTHandler<word> &ntt_handler_;
  const int degree_;
  const int max_slots_;
  const int cyclotomic_index_;

  // The special-FFT twiddles, real and imaginary interleaved, in the layout
  // the transform indexes them by: entry `stride + j` is
  //
  //     exp(-2 pi i * (5^j mod 8*stride) / (8*stride))
  //
  // which is `Encoder::SpecialIFFT`'s `twiddle_factors_[(st8 - 5^j) * gap]`
  // written without the detour through the length-M table. The index runs over
  // `stride = 1, 2, ..., max_slots/2` and `j < stride`, so it is a bijection
  // onto [1, max_slots) -- one table serves every slot count, exactly as the
  // host's does, because the entry depends on (stride, j) and not on how many
  // slots the message happens to fill.
  rmm::device_uvector<double> twiddle_;

  // Scratch, grown on demand and reused: the transform's working buffer
  // (complex, so 2 doubles a slot) and the coefficient vector it lands in.
  mutable rmm::device_uvector<double> fft_;
  mutable rmm::device_uvector<double> coeff_;

  // Host staging for the message, so the H2D copy is one contiguous transfer
  // rather than one per slot, and device staging for a matrix handed over in
  // host memory.
  mutable HostVector<double> host_stage_;
  mutable rmm::device_uvector<double> value_stage_;

  // Per-prime constants the RNS stage reads: the prime, its Barrett constant
  // and R^2 mod p, three uint64 a prime, in the order `GetPrimesPtr` lists
  // them. Cached per NPInfo because building one costs a host division per
  // prime and a transfer, and a layer encodes thousands of operands against
  // the same handful of levels. The Montgomery constant is not here: the
  // Parameter already holds it on the device, indexed identically.
  using PrimeKey = std::tuple<int, int, int, int>;
  mutable std::map<PrimeKey, rmm::device_uvector<uint64_t>> prime_constants_;

  const uint64_t *PrimeConstants(const NPInfo &np) const;

  static constexpr int kRnsBlockDim = 256;
  // The largest chunk one block may hold in shared memory: 2^11 complex
  // doubles is 32 KiB, which is the last power of two that leaves two blocks
  // resident per SM on every architecture this tree targets.
  static constexpr int kMaxLogChunk = 11;

  void EnsureScratch(int num_slots) const;
  void StageMessage(double *device_dst, const std::vector<Complex> &message,
                    int num_slots) const;

 public:
  GpuEncoder(const Parameter<word> &param, const NTTHandler<word> &ntt_handler);

  // Owns device state, like every other handler here.
  GpuEncoder(const GpuEncoder &) = delete;
  GpuEncoder &operator=(const GpuEncoder &) = delete;

  /**
   * @brief Stage 1. The special inverse FFT, in place on `data`, which holds
   * `num_slots` complex values as interleaved (re, im) doubles.
   *
   * Decimation in frequency, the same butterflies in the same order as
   * `Encoder::SpecialIFFT`, split into two shared-memory passes: the stages
   * whose stride is at least `2^s2` connect elements that differ only in the
   * high bits of the index, so a block that gathers with stride `2^s2` runs
   * all of them locally; the rest connect elements inside a contiguous chunk.
   * The output is left in bit-reversed order and un-normalised -- `FftToCoeff`
   * finishes it -- because that permutation is free when it is fused into the
   * read of the next stage and a full pass when it is not.
   *
   * @param data device buffer, 2 * num_slots doubles, in place
   * @param num_slots power of two, at most MaxNumSlots()
   */
  void SpecialIFFT(double *data, int num_slots) const;

  /**
   * @brief Stage 2. Bit-reverse, normalise by 1/num_slots, and place each slot
   * value at the coefficient the encoding puts it at: `i * gap` for the real
   * part, plus `degree/2` for the imaginary part off the conjugate-invariant
   * ring, where there is no imaginary axis and the real part is the whole of
   * it. Coefficients no slot reaches are zeroed.
   *
   * @param coeff output device buffer, `degree` doubles
   * @param fft input device buffer, 2 * num_slots doubles, bit-reversed
   * @param num_slots power of two
   */
  void FftToCoeff(double *coeff, const double *fft, int num_slots) const;

  /**
   * @brief Stage 3. `dst[j * n + i] = round(src[i] * scale) mod p_j`, over the
   * `np`'s primes in the order `Parameter::GetPrimesPtr` lists them, so the
   * result is the limb layout every container here uses.
   *
   * @param dst device buffer, num_total_primes * n words
   * @param src device buffer, n doubles
   * @param n number of values
   * @param np which primes
   * @param scale the scale to round against
   * @param montgomery leave each limb in Montgomery form, which is what a
   *        plaintext matrix wants and what a plaintext about to be NTT'd does
   *        not (the NTT converts on the way in)
   */
  void RnsDecompose(word *dst, const double *src, int n, const NPInfo &np,
                    double scale, bool montgomery) const;

  /**
   * @brief The slot encoding, all four stages on the device. Same contract as
   * `Encoder::Encode`, and the same output up to the half-ulp `Encoder`
   * concedes in `EncodeCoeff`'s comment: this rounds to nearest where
   * `ComplexVectorToPlaintext` truncates through `BigInt(double)`.
   */
  void Encode(Plaintext<word> &ptxt, int level, double scale,
              const std::vector<Complex> &message, int num_aux = 0) const;

  /**
   * @brief The coefficient encoding. Bit-identical to `Encoder::EncodeCoeff`.
   */
  void EncodeCoeff(Plaintext<word> &ptxt, int level, double scale,
                   const std::vector<double> &coeffs, int num_aux = 0) const;

  /**
   * @brief The plaintext matrix encoding. Bit-identical to
   * `PcmmHandler::EncodeMatrix`, which is the `pcmm: convert weights` row.
   */
  void EncodeMatrix(PlainMatrix<word> &res, int level, double scale,
                    const std::vector<double> &values, int rows, int cols,
                    int num_aux = 0) const;

  /**
   * @brief The same, from values already on the device -- which is what a
   * caller that gathers its weight rows on the GPU has, and what separates the
   * PCIe transfer from the encoding in a measurement.
   */
  void EncodeMatrixFromDevice(PlainMatrix<word> &res, int level, double scale,
                              const double *values, int rows, int cols,
                              int num_aux = 0) const;

  // Scratch the composed calls reuse, exposed so a benchmark can time a stage
  // without the allocation in front of it.
  double *FftScratch(int num_slots) const;
  double *CoeffScratch() const;

  /**
   * @brief What the machine makes of these kernels: registers, static and
   * dynamic shared memory, and the resident blocks per SM the occupancy
   * calculator gives for the launch configuration each stage actually uses.
   *
   * It lives here rather than in the benchmark because the kernels are file
   * static -- a caller cannot take their address -- and because the numbers
   * are a property of the build, not of the run.
   *
   * @param os where to print
   * @param num_primes the RNS stage's prime count, which sets its dynamic
   *        shared memory
   * @param log_chunk the FFT stage's chunk, which sets its own
   */
  static void ReportKernelAttributes(std::ostream &os, int num_primes,
                                     int log_chunk);
};

}  // namespace cheddar
