#pragma once

#include <vector>

#include "core/Cmt.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/NTT.h"
#include "core/Parameter.h"
#include "core/SubringCtMatrix.h"

namespace cheddar {

/**
 * @brief Batch ciphertext-ciphertext matrix multiplication, [KANG] (ePrint
 * 2025/1957) Algorithm 4. One call evaluates `k/2` independent `d x d`
 * products.
 *
 * ## The algorithm, and where each piece already lived
 *
 *     1  (B_bar, A_bar) <- CMT((B', A'))^T          CmtHandler
 *     2  [C00 C01; C10 C11] <- [B; A] * [B_bar | A_bar]
 *                                                   SubringCtMatrixHandler
 *     3  (D0, D1) <- CMT((C00, C01))^T              CmtHandler
 *     4  (D2, D3) <- CMT((C10, C11))^T              CmtHandler
 *     5  (E0, E1) <- KeySwitch_{sk^2 -> sk}((0, D3))  Context::Relinearize
 *     6  (Bres, Ares) <- (E0, E1) + (D0, D1 + D2)
 *     7  rescale by Delta
 *
 * There is no new kernel here: every step is one of the pieces already built
 * and tested. What the file supplies is the wiring, and the wiring is where
 * the content is -- specifically, which of the four matrices is column-wise
 * and which is row-wise at each step.
 *
 * ## The one thing worth stating twice
 *
 * Step 2 contracts over the **ciphertext index of both operands**. That is
 * what the CMT of step 1 buys: it turns the second operand from column-wise
 * into row-wise, so that `C[i][l] = sum_j Vec(b_j)_i * Vec(b_bar_j)_l` has the
 * two Vec indices free. Without it the two operands would contract along
 * mismatched axes and no amount of ring arithmetic would fix it -- see
 * `SubringCtMatrix.h`.
 *
 * Steps 3 and 4 run the conversion the other way. `(C00, C01)` satisfies
 * `C00 + C01 * Toep(sk)^T`, a row-wise encryption; reading its **rows** as
 * ciphertexts presents it to CMT as a column-wise encryption of the transpose,
 * and CMT returns the column-wise form the last steps need. That reading is
 * the `transpose` flag of `SubringCtMatrixHandler::ToCiphertexts`.
 *
 * ## Level budget
 *
 * **Exactly one**, spent on step 2 and returned by the rescale of step 7. The
 * three CMTs are level-free (monomials, key switches, and a `d^-1` that is an
 * exact modular inverse), and the relinearization of step 5 is a key switch.
 * That is the whole budget of `ringdegree12_30`, which is why every one of
 * those had to be free: input at level 1, output at level 0, no slack.
 *
 * ## Keys
 *
 * The `d` automorphism keys the CMTs need, plus one relinearization key. The
 * automorphism keys are shared across all three CMT calls.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class BatchCcmmHandler {
 private:
  using Ct = Ciphertext<word>;

  const Parameter<word> &param_;
  CmtHandler<word> cmt_;
  SubringCtMatrixHandler<word> matrix_;

 public:
  BatchCcmmHandler(const Parameter<word> &param,
                   const NTTHandler<word> &ntt_handler);

  // disable copying (or moving also)
  BatchCcmmHandler(const BatchCcmmHandler &) = delete;
  BatchCcmmHandler &operator=(const BatchCcmmHandler &) = delete;

  /**
   * @brief The rotation indices the three CMT calls will ask for, so a caller
   * can generate exactly those keys.
   */
  std::vector<int> RotationIndices(int sub_degree) const;

  /**
   * @brief [KANG] Algorithm 4. `res` is the matrix encryption of
   * `{M_l * M'_l}`, one level below the inputs.
   *
   * @param context the CKKS context
   * @param res output, resized to d = degree / sub_degree ciphertexts
   * @param lhs the d ciphertexts of the first matrix encryption
   * @param rhs the d ciphertexts of the second
   * @param sub_degree k
   * @param evk_map the automorphism keys of `RotationIndices` and the
   *        multiplication key
   */
  void Multiply(ConstContextPtr<word> context, std::vector<Ct> &res,
                const std::vector<Ct> &lhs, const std::vector<Ct> &rhs,
                int sub_degree, const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
