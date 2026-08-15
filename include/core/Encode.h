#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#include "core/Container.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Encoder class for encoding/decoding messages into plaintexts and
 * preparing constants. The implementation is not very optimized as it is not of
 * our highest priority to optimize offline operations. Nevertheless, we intend
 * to optimize this class in the future for faster testing experience.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class Encoder {
 private:
  const Parameter<word> &param_;
  const NTTHandler<word> &ntt_handler_;
  const int M_;

  std::vector<Complex> twiddle_factors_;

  void EncodeWorker(Plaintext<word> &ptxt, int level, double scale,
                    std::vector<Complex> &message, int num_aux = 0) const;
  void ComplexVectorToPlaintext(Plaintext<word> &ptxt, int level, double scale,
                                const std::vector<Complex> &data,
                                int num_aux = 0) const;
  void PlaintextToComplexVector(std::vector<Complex> &data,
                                const Plaintext<word> &ptxt) const;
  void RealVectorToPlaintext(Plaintext<word> &ptxt, int level, double scale,
                             const std::vector<double> &coeffs,
                             int num_aux = 0) const;
  void PlaintextToRealVector(std::vector<double> &coeffs,
                             const Plaintext<word> &ptxt) const;
  DeviceVector<word> MakeICRTConstants(const NPInfo &np) const;
  void SpecialIFFT(std::vector<Complex> &data) const;
  void SpecialFFT(std::vector<Complex> &data) const;

 public:
  /**
   * @brief Construct a new Encoder object.
   *
   * @param param CKKS parameter
   * @param ntt_handler NTTHandler object
   */
  Encoder(const Parameter<word> &param, const NTTHandler<word> &ntt_handler);

  /**
   * @brief Encode a message into a plaintext for a given level and scale.
   * The message will be padded to the nearest power of 2.
   *
   * @param ptxt output plaintext (NTT-applied)
   * @param level level of the plaintext
   * @param scale scale to apply
   * @param message complex message to encode.
   * @param num_aux number of auxiliary primes
   */
  void Encode(Plaintext<word> &ptxt, int level, double scale,
              const std::vector<Complex> &message, int num_aux = 0) const;

  /**
   * @brief Decode a plaintext into a complex message.
   *
   * @param message output complex message
   * @param ptxt input plaintext (NTT-applied)
   */
  void Decode(std::vector<Complex> &message, const Plaintext<word> &ptxt) const;

  /**
   * @brief Encode a real vector directly into the COEFFICIENTS of a plaintext,
   * bypassing the special FFT:
   *
   *     ptxt  =  sum_i  round(scale * coeffs[i]) * X^i    in Z[X]/(X^N + 1)
   *
   * This is the `Ecd_coeff` of Bae et al. (CRYPTO 2024) and is the encoding the
   * reduction-based matrix-multiplication algorithms operate in. Unlike the
   * slot encoding, multiplication of two coefficient-encoded plaintexts is the
   * negacyclic convolution of the encoded vectors, not their pointwise product.
   *
   * Coefficients past `coeffs.size()` are set to zero. Note that the value is
   * rounded to nearest here, whereas Encode() truncates towards zero via
   * BigInt(double); the difference is half a unit in the last place of `scale`.
   *
   * @param ptxt output plaintext (NTT-applied, as for Encode)
   * @param level level of the plaintext
   * @param scale scale to apply
   * @param coeffs real coefficient vector, at most `degree` entries
   * @param num_aux number of auxiliary primes
   */
  void EncodeCoeff(Plaintext<word> &ptxt, int level, double scale,
                   const std::vector<double> &coeffs, int num_aux = 0) const;

  /**
   * @brief Decode a coefficient-encoded plaintext back into a real vector of
   * length `degree`. Inverse of EncodeCoeff.
   *
   * @param coeffs output real coefficient vector (resized to `degree`)
   * @param ptxt input plaintext (NTT-applied)
   */
  void DecodeCoeff(std::vector<double> &coeffs,
                   const Plaintext<word> &ptxt) const;

  /**
   * @brief Encode into the **Slots-in-Coefficients (SinC)** encoding of Cheon,
   * Kang and Lee (ePrint 2025/1957), in the form [SYLPH] states as its eq. (1).
   * This is the encoding the batch CC-MM operates in.
   *
   * ## The definition
   *
   * Let `k = sub_degree` divide N, and identify the subring
   * `R_k = Z[Y]/(Y^k + 1)` with a subring of `R_N` by `Y -> X^(N/k)`. Splitting
   * the message into N/k consecutive blocks of k/2 slots each,
   *
   *     Ecd_SinC(z) = sum_{i<N/k}  round( scale * iDFT_k( z^(i) ) ) * X^i
   *
   * Written out on coefficients, which is what the implementation does:
   *
   *     coefficient at position  i + t*(N/k)   =   iDFT_k( z^(i) )_t
   *
   * for `0 <= i < N/k` and `0 <= t < k`. The map `(i, t) -> i + t*(N/k)` is a
   * bijection onto [0, N), so every coefficient is written exactly once.
   *
   * ## Why it is "intermediate", and how that is checked
   *
   * The two endpoints are the encodings Cheddar already has, and they are not
   * an analogy -- they fall out of the formula:
   *
   *   * `sub_degree == 2`: iDFT_2 is the identity on one complex number, so the
   *     coefficients become `Re(z_i)` at position `i` and `Im(z_i)` at position
   *     `i + N/2`. That is exactly EncodeCoeff on the interleaved parts.
   *   * `sub_degree == degree`: one block, positions `t`, so it is exactly
   *     Encode.
   *
   * SinCEncodeTest asserts both, against those two functions rather than
   * against a reimplementation. The k=2 case is bit-exact (both routes round
   * through RealVectorToPlaintext); the k=N case is compared after decoding,
   * because Encode truncates via BigInt(double) where this rounds.
   *
   * ## No new transform was needed
   *
   * `iDFT_k` is `SpecialIFFT` called on a vector of `k/2` slots. Cheddar's
   * special FFT already derives its twiddle stride from the slot count rather
   * than from the ring degree (`gap = M_ / st8`), which is what makes sparsely
   * packed messages work -- and a sparse message *is* a subring element. The
   * two facts are the same fact, so the existing transform is reused verbatim
   * and the slot encoding keeps routing through the identical code.
   *
   * ## Level budget for what this feeds (ringdegree12_30, max_level = 1)
   *
   * | stage                                  | levels |
   * |----------------------------------------|--------|
   * | SinC encode / decode                   | **0**  |
   * | CMT (Adjust, ScrambleAuto, invAdjust)  | 0      |
   * | the tensor product, [KANG] Alg. 4 st.2 | **1**  |
   * | relinearize, step 5                    | 0      |
   * | rescale, step 7                        | -      |
   * | **total**                              | **1**  |
   *
   * That is the entire budget of this ring, with zero slack: the input must sit
   * at level 1 and the output lands at level 0. A second ciphertext-ciphertext
   * multiplication anywhere in the chain does not fit, and raising the ring to
   * two levels puts log2 QP at 130.75 and the security at 101.3 gate bits. The
   * encoding itself is free, which is why it is the first thing built.
   *
   * @param ptxt output plaintext (NTT-applied, as for Encode)
   * @param level level of the plaintext
   * @param scale scale to apply
   * @param message complex message, exactly `degree / 2` entries
   * @param sub_degree k, a power of two dividing the ring degree, >= 2
   * @param num_aux number of auxiliary primes
   */
  void EncodeSinC(Plaintext<word> &ptxt, int level, double scale,
                  const std::vector<Complex> &message, int sub_degree,
                  int num_aux = 0) const;

  /**
   * @brief Decode a SinC-encoded plaintext. Inverse of EncodeSinC, and it must
   * be given the same `sub_degree` -- the encoding does not record it, exactly
   * as the coefficient encoding does not record that it is one.
   *
   * @param message output complex message (resized to `degree / 2`)
   * @param ptxt input plaintext (NTT-applied)
   * @param sub_degree k, the same value used to encode
   */
  void DecodeSinC(std::vector<Complex> &message, const Plaintext<word> &ptxt,
                  int sub_degree) const;

  /**
   * @brief Encode a real number (double) into an RNS constant for a given level
   * and scale.
   *
   * @param constant output RNS constant
   * @param level level of the constant
   * @param scale scale to apply
   * @param number real number to encode
   * @param num_aux number of auxiliary primes (default: 0 --> none)
   */
  void EncodeConstant(Constant<word> &constant, int level, double scale,
                      double number, int num_aux = 0) const;

  /**
   * @brief Get the twiddle factor for a given index, which is used for special
   * FFT.
   *
   * @param index twiddle factor index
   * @return Complex twiddle factor
   */
  Complex GetTwiddleFactor(int index) const;
};

}  // namespace cheddar
