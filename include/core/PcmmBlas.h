#pragma once

#ifdef USE_CUBLAS

#include <cublas_v2.h>

#include <vector>

#include "core/Container.h"
#include "core/Parameter.h"
#include "core/Pcmm.h"

namespace cheddar {

/**
 * @brief PCMM through cuBLAS, which is where [SYLPH]'s speed comes from.
 *
 * ## Why this is a GEMM at all
 *
 * [BAE] section 1.1: `(U.A) . Toep(sk) + (U.B) = U.M`, so PCMM *is* two
 * plaintext matrix products. `PcmmAccum` already computes, per RNS prime,
 * `DST[rows x degree] = U[rows x cols] * SRC[cols x degree]` -- a GEMM written
 * by hand. [SYLPH] section 2.2 credits its performance to "offloading most
 * computations to fast BLAS libraries", and we were not.
 *
 * ## Why int8 and not fp64
 *
 * cuBLAS does no modular arithmetic, so an exact product needs operands split
 * into pieces small enough that nothing overflows the accumulator. Measured on
 * this A6000 for one prime slice (128 x 4096 by 4096 x 4096, 4.29 GFLOP):
 *
 *     fp64           7.315 ms    0.6 TFLOP/s   3 splits -> 131.7 ms
 *     fp32           0.205 ms   20.9 TFLOP/s   not exact for 30-bit residues
 *     int8 -> int32  0.086 ms   49.9 TOP/s     -> a few ms
 *
 * against `PcmmAccum`'s 36.7 ms for the same work. **fp64 is a dead end here**:
 * the A6000 runs it at 1/32 of fp32, so `cublasDgemm` would be slower than the
 * kernel it replaces. Measuring this before building the machinery is the only
 * reason it did not get built the wrong way.
 *
 * ## The splitting
 *
 * Residues are below a prime of up to 30 bits. Signed int8 holds
 * `[0, 127]` safely, so each value becomes `S = ceil(bits / 7)` pieces of 7
 * bits. A piece product is at most `127 * 127 < 2^14`, and summing `cols`
 * of them (4096 = 2^12) reaches `2^26`, inside int32 with room to spare.
 *
 * Pieces `k` of U and `l` of the source contribute at shift `7 * (k + l)`, so
 * the `S^2` products fall into `2S - 1` shift groups. cuBLAS accumulates each
 * group in place with `beta = 1`, and one kernel recombines them modulo the
 * prime. That keeps the group sums inside int32 while the recombination, which
 * needs 64-bit intermediates, happens once per output element rather than once
 * per product.
 *
 * ## Correctness
 *
 * `PcmmHandler::Multiply` stays untouched and is the reference: this path must
 * agree with it exactly, not approximately. `U` is taken as plain residues here
 * rather than in Montgomery form, and both produce plain residues, so equality
 * is bit-for-bit and a test can assert it.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class PcmmBlasHandler {
 private:
  using Ct = Ciphertext<word>;

  const Parameter<word> &param_;
  cublasHandle_t handle_ = nullptr;

  // 7 bits per int8 piece: signed int8 reaches 127, so this is the largest
  // width that never needs an offset or a sign fixup.
  static constexpr int kPieceBits = 7;

 public:
  /** @brief U split into int8 pieces, one set per RNS prime. Setup-time. */
  struct SplitMatrix {
    int rows = 0;
    int cols = 0;
    int pieces = 0;   // S
    double scale = 1.0;
    NPInfo np;
    // pieces * primes * rows * cols int8 values, piece-major
    DeviceVector<int8_t> data;
  };

  explicit PcmmBlasHandler(const Parameter<word> &param);
  ~PcmmBlasHandler();

  PcmmBlasHandler(const PcmmBlasHandler &) = delete;
  PcmmBlasHandler &operator=(const PcmmBlasHandler &) = delete;

  /**
   * @brief Split the plaintext matrix into int8 pieces, once, offline.
   *
   * This is [SYLPH] section 5.3's model conversion: the weights are put into
   * the form the product consumes and kept on the GPU (section 5.1). It takes
   * the same `values` `PcmmHandler::EncodeMatrix` does rather than an encoded
   * matrix, because the GEMM wants plain residues and `PlainMatrix` holds
   * Montgomery form.
   */
  void SplitMatrixFrom(SplitMatrix &res, int level, double scale,
                       const std::vector<double> &values, int rows, int cols,
                       int num_aux = 0) const;

  /** @brief Device bytes the split matrix holds. */
  static size_t SplitBytes(const SplitMatrix &m) {
    return m.data.size() * sizeof(int8_t);
  }

  /**
   * @brief res[i] = sum_j u[i][j] * cts[j], through cuBLAS.
   *
   * Same contract as `PcmmHandler::Multiply`: no rescaling, result at
   * `u.scale * cts[0].scale`.
   */
  void Multiply(std::vector<Ct> &res, const SplitMatrix &u,
                const std::vector<Ct> &cts) const;

  /**
   * @brief The same product on MLWE ciphertexts, which is the format the Llama
   * projections actually run in.
   *
   * `CoeffLinearLeg` reaches the product by `ModDecomp`, so its operands are
   * `MlweCiphertext`, not `Ciphertext`, and the RLWE overload above was never
   * on the layer's path. `PcmmHandler` carries the same pair of overloads for
   * the same reason, and the second one is the one the block calls.
   *
   * No new machinery is involved: an MLWE ciphertext is a rank-k `a_` plus a
   * single `b_`, both laid out with all limb data contiguous, so each is one
   * more plaintext product of the same shape -- the b-part over vectors of
   * `degree` words per limb and the a-part over `rank * degree`. That is
   * exactly the substitution `PcmmHandler::Multiply(MLWE)` makes when it
   * passes `a_stride` to `PcmmAccum` in place of the degree.
   *
   * @param res output, resized to u.rows
   * @param u split plaintext matrix, u.cols must equal cts.size()
   * @param cts input MLWE ciphertexts, sharing one NP, rank and degree
   */
  void Multiply(std::vector<MlweCiphertext<word>> &res, const SplitMatrix &u,
                const std::vector<MlweCiphertext<word>> &cts) const;

 private:
  /**
   * @brief One plaintext product against one ciphertext component.
   *
   * `vec_len` is the number of words a single RNS limb of the component holds:
   * the ring degree for an RLWE component, the small degree for an MLWE b-part
   * and `rank * small_degree` for an MLWE a-part. Everything else about the
   * product -- the gather, the `pieces^2` GEMMs and the recombination -- is
   * identical, which is why both public overloads are two calls to this.
   */
  void MultiplyComponent(word *const *dst_ptrs, const word *const *src_ptrs,
                         const SplitMatrix &u, const NPInfo &np,
                         int vec_len) const;
};

}  // namespace cheddar

#endif  // USE_CUBLAS
