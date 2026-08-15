#pragma once

#include <vector>

#include "core/Container.h"
#include "core/DeviceVector.h"
#include "core/Mlwe.h"
#include "core/NPInfo.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Plaintext-ciphertext matrix multiplication (PC-MM) of Bae, Cheon,
 * Hanrot, Park and Stehle, "Plaintext-Ciphertext Matrix Multiplication and FHE
 * Bootstrapping: Fast and Fused", CRYPTO 2024.
 *
 * ## The identity
 *
 * With coefficient encoding, d ciphertexts (a_i, b_i) whose messages m_i are
 * the rows of a matrix M satisfy, over Z_q,
 *
 *     A * Toep(sk) + B  =  M
 *
 * where row i of A (resp. B, M) is the coefficient vector of a_i (resp. b_i,
 * m_i) and Toep(sk) has X^i * sk as its i-th row. Left-multiplying by a
 * plaintext matrix U preserves the shape:
 *
 *     (U*A) * Toep(sk) + (U*B)  =  U*M
 *
 * so (U*A, U*B) is an encryption of U*M. The whole product is therefore two
 * plaintext matrix products, one multiplicative level, and -- the property
 * that matters most on a GPU whose memory is dominated by keys -- **no
 * rotation, automorphism or relinearization key at all**.
 *
 * Both products are linear in the rows of A and B, and the NTT is linear, so
 * they may be evaluated directly on NTT-domain ciphertext components. No
 * inverse NTT is needed anywhere in this file.
 *
 * ## Which ciphertext format, and which conversions
 *
 * The paper gives three formats. The choice is made by the number of columns
 * of the *encrypted* matrix relative to the ring degree (paper, section 4),
 * and every format additionally requires d >= N^(1/2):
 *
 *   columns >= N : shared-a RLWE   -- PP-MM_{d,d,d} + PP-MM_{d,d,N}
 *   columns <= N : MLWE            -- PP-MM_{d,d,d} + PP-MM_{d,d,N}
 *   either, with precomputed key-switching keys for the rows of U*S : one PP-MM
 *
 * For Llama-3-8B at 128 tokens, Sylph (Park et al., arXiv 2601.18511, Table 4)
 * runs PCMM at **ring degree 256 in coefficient encoding**, reached by
 * "ring-switching followed by an MLWE decomposition" (Sylph section 3.2). The
 * encrypted activation has 128 token-columns, which is below that degree, so
 * the format is **MLWE**, and the module rank k carries the security dimension:
 * MLWE of rank k over degree d is as hard as RLWE over degree kd, which is why
 * dropping to degree 256 costs nothing in security.
 *
 * Sylph's weight memory (its Table 5) shows only the ~4x RNS expansion of the
 * fp16 weights and no key material for U*S, so Sylph uses the plain MLWE
 * algorithm (paper, Algorithm 2) and *not* the precomputation variant. We
 * follow that.
 *
 * The conversion chain that implies, from the non-linear layers down and back:
 *
 *   slot @ 2^16
 *     -> SlotToCoeff (fused into bootstrapping)      coefficient @ 2^16
 *     -> RingSwitch                                  coefficient @ N1
 *        (needs a switching key into a subring secret, so it is only secure at
 *         a small modulus -- this is why the product is scheduled at a low
 *         level, not merely why it is faster there)
 *     -> ModDecomp                                   MLWE^k @ d
 *        (**free**: no key, no security cost, since k*d is preserved)
 *     -> the two PP-MMs                              <- this file
 *     -> ModPack                                     coefficient @ N1
 *        (needs switching keys)
 *     -> RingSwitch                                  coefficient @ 2^16
 *     -> bootstrap, which restores slot encoding
 *
 * ## What is implemented here
 *
 * Only the product itself, in the **RLWE format at a single ring degree**:
 * `rows` output ciphertexts, each a U-combination of `cols` input ciphertexts.
 * That needs none of the conversions above and is directly useful, and it is
 * the same two PP-MMs the MLWE path performs -- the format changes only how
 * the rows of A and B are gathered, not the product. The kernel therefore
 * takes row pointers rather than whole ciphertexts, so the MLWE path can reuse
 * it once ModDecomp and an MLWE container exist.
 *
 * Rescaling is not performed, matching Context::Mult(Ct, Ct, Pt).
 *
 * @tparam word uint32_t or uint64_t
 */

/**
 * @brief A real matrix encoded into RNS limbs for use as the plaintext operand
 * of a PC-MM. Entries are stored in Montgomery form, laid out as
 * [prime][row][col].
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class PlainMatrix {
 public:
  PlainMatrix() = default;

  // Movable, but not copyable (it owns device memory).
  PlainMatrix(PlainMatrix &&) = default;
  PlainMatrix &operator=(PlainMatrix &&) = default;

  int GetRows() const;
  int GetCols() const;
  double GetScale() const;
  NPInfo GetNP() const;

  // member variables (public, as elsewhere in Cheddar)
  int rows_ = 0;
  int cols_ = 0;
  double scale_ = 1.0;
  NPInfo np_;
  DeviceVector<word> data_;
};

template <typename word>
class PcmmHandler {
 private:
  using Ct = Ciphertext<word>;

  const Parameter<word> &param_;

  static constexpr int kernel_block_dim_ = 256;

 public:
  explicit PcmmHandler(const Parameter<word> &param);

  // disable copying (or moving also)
  PcmmHandler(const PcmmHandler &) = delete;
  PcmmHandler &operator=(const PcmmHandler &) = delete;

  /**
   * @brief Encode a row-major real matrix into RNS limbs for use as the
   * plaintext operand.
   *
   * @param res output encoded matrix
   * @param level level to encode at
   * @param scale scale to apply to every entry
   * @param values row-major entries, exactly rows * cols of them
   * @param rows number of rows of U
   * @param cols number of columns of U
   * @param num_aux number of auxiliary primes (default: 0 --> none)
   */
  void EncodeMatrix(PlainMatrix<word> &res, int level, double scale,
                    const std::vector<double> &values, int rows, int cols,
                    int num_aux = 0) const;

  /**
   * @brief res[i] = sum_j u[i][j] * cts[j], for coefficient-encoded
   * ciphertexts sharing one level. Rescaling is not performed; the result is
   * at scale u.scale * cts[0].scale.
   *
   * @param res output ciphertexts, resized to u.rows_
   * @param u encoded plaintext matrix, u.cols_ must equal cts.size()
   * @param cts input ciphertexts, all at the same NP and without an rx_ part
   */
  void Multiply(std::vector<Ct> &res, const PlainMatrix<word> &u,
                const std::vector<Ct> &cts) const;

  /**
   * @brief The same product on MLWE ciphertexts: res[i] = sum_j u[i][j] *
   * cts[j].
   *
   * This is the format [SYLPH] table 4 actually uses for the PC-MM, which runs
   * at ring degree 256 -- reached from the degree-4096 ring by ModDecomp, and
   * so narrower than the ring the RLWE overload above assumes (the header's
   * format rule: columns <= N calls for MLWE).
   *
   * No new kernel is involved. An MLWE ciphertext is a rank-k vector `a_` plus
   * a single `b_`, and the product is a scalar linear combination of both, so
   * it is the RLWE case with k components instead of one. The layout
   * `a_[(limb * rank + j) * degree + s]` keeps all k * degree words of a limb
   * contiguous, which means the existing accumulator handles the a-part
   * verbatim once it is told the per-limb stride is `rank * degree` rather
   * than `degree`.
   *
   * Being a scalar combination, this is also domain-agnostic: it is correct on
   * the coefficient-domain output of ModDecomp exactly as it is on NTT-domain
   * RLWE, because scaling and addition commute with the transform.
   *
   * As with the RLWE overload, no rescaling is performed; the result carries
   * scale `u.scale_ * cts[0].scale_`.
   *
   * @param res output, resized to u.rows_
   * @param u encoded plaintext matrix, u.cols_ must equal cts.size()
   * @param cts input MLWE ciphertexts, all sharing one NP, rank and degree
   */
  void Multiply(std::vector<MlweCiphertext<word>> &res,
                const PlainMatrix<word> &u,
                const std::vector<MlweCiphertext<word>> &cts) const;
};

}  // namespace cheddar
