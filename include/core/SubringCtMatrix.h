#pragma once

#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/NPInfo.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Matrix products over the subring `R_k` on ciphertext-derived
 * matrices -- the one primitive [KANG] (ePrint 2025/1957) Algorithm 4 needs
 * that the plaintext path does not supply.
 *
 * ## Why the plaintext engine does not extend to this
 *
 * `SubringMatrixHandler::Multiply` gets away with a pointwise product in the
 * NTT domain because its right operand is a *subring element* and `Vec` is
 * `R_k`-linear: `Vec(u*m)_i = u * Vec(m)_i`, so the Vec dimension is a
 * passenger and the contraction runs over the ciphertext index alone.
 *
 * Algorithm 4 step 2 is the opposite case. After the CMT of step 1 the second
 * operand is *row-wise*, so
 *
 *     C[i][l] = sum_j Vec(b_j)_i * Vec(b_bar_j)_l
 *
 * -- the contraction runs over the ciphertext index of **both** operands and
 * the free indices are the two Vec indices. That is an outer product in the
 * Vec dimensions, and an `R_N` product cannot produce it: multiplying
 * `b_j * b_bar_j` in `R_N` collapses `(i, l)` into a convolution over `i + l`.
 * Separating the Vec components is therefore unavoidable, and it is the whole
 * content of this file.
 *
 * ## The representation
 *
 * [KANG] assumes the matrices are NTT-transformed over `R_{q,k}`, which makes
 * the `R_k` product pointwise and hands the rest to BLAS.
 *
 * The entries are held **in the coefficient domain**, and the product runs
 * through a length-k negacyclic transform of its own so that it becomes k
 * independent scalar matrix products -- the form [KANG] assumes and the form
 * BLAS wants.
 *
 * The transform does **not** have to agree with Cheddar's degree-N one. Both
 * directions are built here from the same primitive 2k-th root, so they invert
 * each other whichever root is chosen, and the coefficients that come back out
 * are the same either way. That removes the dependence on an unverified
 * convention that `Doing.md` section 1.5b would otherwise have made a risk.
 *
 * What makes it safe to add a transform at all is that the `O(k^2)` direct
 * convolution is kept alongside it (`MultiplyMatricesReference`). The two
 * share no arithmetic beyond the modular multiply, so the fast path is
 * validated against a genuinely independent method rather than against
 * nothing.
 *
 * ## Montgomery
 *
 * The components are inverse-transformed with `montgomery_conversion = false`,
 * which leaves Montgomery residues rather than plain ones (the opposite of the
 * default, and the opposite of what `ModDecomp` wants). That is deliberate:
 * `MultMontgomery(aR, bR) = abR` keeps the convolution in the same
 * representation, and the forward transform back is likewise taken with
 * `montgomery_conversion = false`, so the ciphertext components come out in
 * the form every other kernel expects.
 *
 * @tparam word uint32_t or uint64_t
 */

/**
 * @brief A `rows x cols` matrix over `R_k`, entries in the coefficient domain,
 * laid out `[row][col][limb][coefficient of Y]`.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SubringCoeffMatrix {
 public:
  SubringCoeffMatrix() = default;

  // Movable, but not copyable (it owns device memory).
  SubringCoeffMatrix(SubringCoeffMatrix &&) = default;
  SubringCoeffMatrix &operator=(SubringCoeffMatrix &&) = default;

  int GetRows() const { return rows_; }
  int GetCols() const { return cols_; }
  int GetSubDegree() const { return sub_degree_; }
  NPInfo GetNP() const { return np_; }
  double GetScale() const { return scale_; }

  // member variables (public, as elsewhere in Cheddar)
  int rows_ = 0;
  int cols_ = 0;
  int sub_degree_ = 0;  // k
  NPInfo np_;
  double scale_ = 1.0;
  DeviceVector<word> data_;
};

template <typename word>
class SubringCtMatrixHandler {
 private:
  using Ct = Ciphertext<word>;

  const Parameter<word> &param_;
  const NTTHandler<word> &ntt_handler_;

  static constexpr int kernel_block_dim_ = 128;

  // Twiddles for the length-k negacyclic transform, per prime of `np`:
  // psi_rev[j] = psi^bitrev(j), its inverse, and k^-1, all in Montgomery form.
  //
  // These are built here rather than lifted from NTTHandler, whose tables are
  // private and indexed for the full degree. Nothing is lost by that: the two
  // directions below are each other's inverse for *any* primitive 2k-th root,
  // so this transform never has to agree with Cheddar's choice of psi -- it
  // only has to agree with itself, which the reference product checks.
  struct SubringTwiddles {
    DeviceVector<word> forward;
    DeviceVector<word> inverse;
    DeviceVector<word> scale;
  };

  SubringTwiddles BuildTwiddles(const NPInfo &np, int sub_degree) const;
  void TransformEntries(SubringCoeffMatrix<word> &mat,
                        const SubringTwiddles &tw, bool inverse) const;

 public:
  SubringCtMatrixHandler(const Parameter<word> &param,
                         const NTTHandler<word> &ntt_handler);

  // disable copying (or moving also)
  SubringCtMatrixHandler(const SubringCtMatrixHandler &) = delete;
  SubringCtMatrixHandler &operator=(const SubringCtMatrixHandler &) = delete;

  /**
   * @brief Read `cts` as the two matrices `(B, A)` of the subring matrix
   * encryption, `Vec^d_k` applied to each component.
   *
   * With `row_wise = false` this is [KANG] Definition 2: ciphertext j supplies
   * **column** j, so entry `(i, j)` is `Vec(b_j)_i` and the matrix is `d x d'`.
   * With `row_wise = true` ciphertext j supplies **row** j instead, which is
   * the shape Algorithm 4 needs for the operand that has been through CMT --
   * `CMT` followed by a matrix transpose is a row-wise encryption, and this
   * flag is that transpose.
   *
   * @param b_mat output, the b-part matrix
   * @param a_mat output, the a-part matrix
   * @param cts the ciphertexts, all at one level and without an rx_ part
   * @param sub_degree k
   * @param row_wise whether each ciphertext is a row rather than a column
   */
  void ToMatrices(SubringCoeffMatrix<word> &b_mat,
                  SubringCoeffMatrix<word> &a_mat, const std::vector<Ct> &cts,
                  int sub_degree, bool row_wise) const;

  /**
   * @brief `res = lhs * rhs` over `R_k`, through the length-k transform: each
   * entry is transformed, the product becomes k independent scalar matrix
   * products, and the result is transformed back.
   *
   * @param res output, resized; may not alias either input
   * @param lhs left operand, `rows x inner`
   * @param rhs right operand, `inner x cols`
   */
  void MultiplyMatrices(SubringCoeffMatrix<word> &res,
                        const SubringCoeffMatrix<word> &lhs,
                        const SubringCoeffMatrix<word> &rhs) const;

  /**
   * @brief The same product by direct negacyclic convolution, without any
   * transform.
   *
   * Kept as the reference the transform-based path is checked against. The two
   * share no arithmetic beyond the modular multiply, so agreement is evidence
   * rather than a restatement -- and a length-k transform is exactly the kind
   * of code where a twiddle convention can be self-consistently wrong.
   *
   * `O(k^2)` per entry pair against the transform path `O(k log k)`, so it is
   * for tests and small shapes, not for the pipeline.
   */
  void MultiplyMatricesReference(SubringCoeffMatrix<word> &res,
                                 const SubringCoeffMatrix<word> &lhs,
                                 const SubringCoeffMatrix<word> &rhs) const;

  /**
   * @brief The inverse of `ToMatrices` with `row_wise = false`: rebuild one
   * ciphertext per column from a b-part and an a-part matrix.
   *
   * @param res output ciphertexts, resized to the matrices' column count
   * @param b_mat the b-part matrix
   * @param a_mat the a-part matrix
   * @param scale the scale to record on the outputs
   * @param num_slots the slot count to record on the outputs
   */
  void ToCiphertexts(std::vector<Ct> &res,
                     const SubringCoeffMatrix<word> &b_mat,
                     const SubringCoeffMatrix<word> &a_mat, double scale,
                     int num_slots) const;
};

}  // namespace cheddar
