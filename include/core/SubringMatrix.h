#pragma once

#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/DeviceVector.h"
#include "core/Encode.h"
#include "core/NPInfo.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Batch plaintext-ciphertext matrix multiplication of Cheon, Kang and
 * Lee, "Fast Batch Matrix Multiplication in Ciphertexts" (ePrint 2025/1957),
 * Algorithm 1 -- the subring matrix encryption and the product on it.
 *
 * ## The matrix encryption ([KANG] section 3.1)
 *
 * With `k | N`, `d = N/k` and `Y = X^d`, so that `R_k = Z[Y]/(Y^k + 1)` sits
 * inside `R_N`:
 *
 *     Vec^d_k(m)_i = the coefficients m[i + t*d], 0 <= t < k
 *
 * Stacking d' ciphertexts column-wise gives `B + Toep^d_k(sk) * A = M` over
 * `R_k`, where the row index is the Vec component and the column index is
 * which ciphertext.
 *
 * Two things about that are worth stating plainly, because both save work:
 *
 *   * **`Vec^d_k` is the SinC block index.** `Ecd_SinC` puts `iDFT_k(z^(i))`
 *     at exactly the coefficients `i + t*d`, so component i of the Vec view is
 *     the subring element carrying message block i. No new layout, no new
 *     kernel, and the index map is the one SinCIndexMapIsABijection pins.
 *   * **`Toep^d_k` is [BAE]'s Toeplitz-with-Y.** Entry `[l][m]` is `s_{l-m}`
 *     for `m <= l` and `Y * s_{l-m+d}` for `m > l`, which is the arrangement
 *     `MlweHandler::ModDecomp` already materialises -- built there from `a`
 *     rather than from `sk`. [BAE] appendix A notes the two play symmetric
 *     roles; this is that symmetry, and it means Theorem 1 of [KANG] and the
 *     MLWE identity of [BAE] are one fact.
 *
 * ## Why the product needs no matrix machinery at all
 *
 * `U` has entries in `R_k`, and `Vec` is `R_k`-linear: for `u = u(Y)` there is
 * no `X^i` factor to move, so `Vec(u*m)_i = u * Vec(m)_i`. Therefore
 *
 *     (A*U)_{i,l} = sum_j Vec(a_j)_i * u_{j,l} = Vec( sum_j u_{j,l} * a_j )_i
 *
 * and the whole d x d' by d' x d'' product collapses, at the ciphertext level,
 * to
 *
 *     res_l = sum_j u_{j,l} * ct_j
 *
 * a plaintext-ciphertext linear combination in `R_N` whose plaintexts happen
 * to be supported on multiples of `X^d`. **The Vec dimension is a passenger:
 * the product contracts the ciphertext index and never touches the row index.**
 *
 * So there is no inverse transform, no subring NTT and no reindexing. In the
 * NTT domain a plaintext multiplication is pointwise, so one fused
 * multiply-accumulate over both ciphertext components is the entire operation.
 *
 * ## What makes it a *batch* product
 *
 * Under SinC the subring element `u` acts on every block through its `DFT_k`,
 * `uhat` in `C^{k/2}` -- that is the ring homomorphism
 * SinCSubringProductIsSlotwiseWithinABlock checks. Writing slots as
 * `[block i][lane t]`,
 *
 *     res_l[i][t] = sum_j uhat_{j,l}[t] * z_j[i][t]
 *
 * which for each fixed lane `t` is an independent `(d x d') * (d' x d'')`
 * matrix product. One pass therefore evaluates **k/2 matrix products at once**,
 * with the `d` rows living inside a single ciphertext and the `d'` contraction
 * running across ciphertexts.
 *
 * ## Level budget
 *
 * One plaintext multiplication and one rescale: **depth 1**, no rotation,
 * automorphism or relinearization key of any kind. That is the entire budget
 * of `ringdegree12_30` and it is what [KANG] Table 1 claims for Algorithm 1.
 * The batch CC-MM (Algorithm 4) spends the same single level on its tensor
 * product, so nothing else in either chain may consume one.
 *
 * @tparam word uint32_t or uint64_t
 */

/**
 * @brief The plaintext matrix `U` over `R_k`, encoded for the product above:
 * `cols_in * cols_out` subring elements, held as one device buffer laid out
 * `[j][l][limb][coefficient]` so the accumulator can index it directly.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SubringWeights {
 public:
  SubringWeights() = default;

  // Movable, but not copyable (it owns device memory).
  SubringWeights(SubringWeights &&) = default;
  SubringWeights &operator=(SubringWeights &&) = default;

  int GetColsIn() const;
  int GetColsOut() const;
  int GetSubDegree() const;
  double GetScale() const;
  NPInfo GetNP() const;

  // member variables (public, as elsewhere in Cheddar)
  int cols_in_ = 0;      // d', the number of input ciphertexts (contracted)
  int cols_out_ = 0;     // d'', the number of output ciphertexts
  int sub_degree_ = 0;   // k
  double scale_ = 1.0;
  NPInfo np_;
  DeviceVector<word> data_;
};

template <typename word>
class SubringMatrixHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

  const Parameter<word> &param_;
  const Encoder<word> &encoder_;

  static constexpr int kernel_block_dim_ = 256;

 public:
  SubringMatrixHandler(const Parameter<word> &param,
                       const Encoder<word> &encoder);

  // disable copying (or moving also)
  SubringMatrixHandler(const SubringMatrixHandler &) = delete;
  SubringMatrixHandler &operator=(const SubringMatrixHandler &) = delete;

  /**
   * @brief Encode the batch of plaintext matrices into `U`.
   *
   * Entry `(j, l)` of `U` is the subring element whose `DFT_k` is the vector
   * of that entry across the batch: `uhat_{j,l}[t] = values[t][j][l]`. It is
   * built by SinC-encoding the message that repeats `uhat_{j,l}` in every one
   * of the `d` blocks -- a subring element is exactly a message with that
   * period, which is the same fact that makes sparsely packed ciphertexts
   * work.
   *
   * @param res output weights
   * @param level level to encode at
   * @param scale scale to apply
   * @param values the batch, indexed `values[t][j * cols_out + l]` with
   *        `t < sub_degree / 2` lanes and each inner vector of length
   *        `cols_in * cols_out`
   * @param cols_in d', the number of input ciphertexts
   * @param cols_out d'', the number of output ciphertexts
   * @param sub_degree k, a power of two dividing the ring degree
   * @param num_aux number of auxiliary primes (default: 0 --> none)
   */
  void EncodeWeights(SubringWeights<word> &res, int level, double scale,
                     const std::vector<std::vector<Complex>> &values,
                     int cols_in, int cols_out, int sub_degree,
                     int num_aux = 0) const;

  /**
   * @brief [KANG] Algorithm 1: `(B, A) <- (B*U, A*U)`, then rescale.
   *
   * `res[l] = sum_j u[j][l] * cts[j]`, evaluated pointwise in the NTT domain
   * on both ciphertext components, followed by one rescale. The result is one
   * level below the input and carries the canonical scale of that level when
   * both operands carried the canonical scale of theirs.
   *
   * @param context the CKKS context, for the rescale
   * @param res output ciphertexts, resized to `u.cols_out_`
   * @param u the encoded weights
   * @param cts input ciphertexts, `u.cols_in_` of them, all at one level, at
   *        the weights' NP, and without an rx_ part
   */
  void Multiply(ConstContextPtr<word> context, std::vector<Ct> &res,
                const SubringWeights<word> &u,
                const std::vector<Ct> &cts) const;
};

}  // namespace cheddar
