#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/MultiLevelCiphertext.h"
#include "extension/EvalPoly.h"
#include "extension/SlimPolyMath.h"

namespace cheddar {


/**
 * @brief [SYLPH] Algorithm 1 on a ciphertext: the encrypted half of the above.
 *
 * The input must be **slim**: its message repeats with period `block_slots`,
 * so that every one of the `2^j` blocks sees the same argument and evaluates a
 * different leaf of the tree. `Compile` turns a `SlimPlan` into the plaintexts
 * `v^(delta)` and `m^(l)` that Algorithm 1 reads, and `Evaluate` runs the
 * loop.
 *
 * The leaf evaluation is `EvalPoly`'s Paterson-Stockmeyer tree with
 * **plaintext** coefficients rather than scalars -- that is what lets the
 * blocks carry different polynomials -- which is why the basis powers are
 * built here and multiplied by encoded vectors instead of `Constant`s.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SlimPolyHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  ConstContextPtr<word> context_;
  slim::SlimPlan plan_;
  int block_slots_ = 0;
  int input_level_ = 0;
  double input_scale_ = 0.0;
  int output_level_ = 0;
  double output_scale_ = 0.0;
  bool fold_leading_ = false;
  bool compiled_ = false;
  //! `v^(1)` in the block layout, for the caller to fold upstream.
  std::vector<Complex> leading_msg_;

  // v^(delta) for 0 <= delta <= 2^(k-j), one full-width plaintext each: slot
  // `i * block_slots + s` carries leaf `i mod 2^j`'s degree-delta coefficient.
  std::vector<std::vector<Complex>> leaf_msg_;
  // m^(l) for 0 <= l < j, in the same block layout.
  std::vector<std::vector<Complex>> m_msg_;

  // The compiled per-level pieces: the leaf evaluation's plaintexts at the
  // level they are consumed, and the m^(l) plaintexts at theirs.
  std::vector<Pt> leaf_pt_;
  std::vector<Pt> m_pt_;
  // The scale `T_delta` arrives at, which each `v^(delta)` has to divide out
  // so that every product reaches the accumulation at one scale.
  std::vector<double> basis_scale_;

 public:
  /**
   * @param context the evaluation context
   * @param plan the decomposition, from `slim::BuildPlan`
   * @param block_slots `2^t`, the period of the slim message. Must satisfy
   *        `block_slots * plan.NumBlocks() <= context->GetNumSlots()`.
   * @param input_level level the argument arrives at
   * @param input_scale its scale
   * @param output_scale the scale the result is wanted at
   * @param fold_leading appendix D's last paragraph: "by obtaining on input a
   *        ciphertext containing `v^(2^(k-j))_i^(2^(j-k)) . ct_i` by fusing a
   *        plaintext-ciphertext multiplication in a previous operation,
   *        Algorithm 1 can be made to use only `k` levels".
   *
   *        **This is the whole difference between slim costing a level and
   *        costing nothing.** It requires `j == k`, where the leaf is degree
   *        one: the leaf is then `v^(1)_i x + v^(0)_i`, and if the caller has
   *        already multiplied `x` by `v^(1)_i` then what is left is a
   *        plaintext ADD, which spends no level at all. Algorithm 1 becomes
   *        `k` levels for degree `2^k`, which is exactly what
   *        Paterson-Stockmeyer spends on degree `2^k - 1` -- so at equal
   *        levels the degrees are equal and slim wins on multiplications
   *        outright.
   *
   *        The caller gets `v^(1)` from `GetLeadingMessage()` and is
   *        responsible for folding it into whatever plaintext multiply
   *        already precedes the evaluation. In the softmax's auxiliary track
   *        that is the affine map onto the fit domain, whose two constants
   *        simply become plaintexts -- the same level either way.
   */
  SlimPolyHandler(ConstContextPtr<word> context, slim::SlimPlan plan,
                  int block_slots, int input_level, double input_scale,
                  double output_scale, bool fold_leading = false);

  /**
   * @brief `v^(1)`, one value per slot, for a caller folding it upstream.
   *
   * Empty unless `fold_leading` is on. Slot `i * block_slots + s` carries leaf
   * `i mod 2^j`'s degree-one Chebyshev coefficient, which is what the input
   * must already have been multiplied by.
   */
  const std::vector<Complex> &GetLeadingMessage() const {
    return leading_msg_;
  }

  /** @brief Whether appendix D's fold is on, hence `k` levels not `k + 1`. */
  bool GetFoldLeading() const { return fold_leading_; }

  SlimPolyHandler(const SlimPolyHandler &) = delete;
  SlimPolyHandler &operator=(const SlimPolyHandler &) = delete;

  /** @brief Encode the plan's plaintexts; must run before `Evaluate`. */
  void Compile();

  /** @brief The level the result lands at. */
  int GetOutputLevel() const { return output_level_; }

  /** @brief Rotation distances the fold needs: `block_slots * 2^(l-1)`. */
  std::vector<int> GetRotationDistances() const;

  /**
   * @brief `res = P(input)`, Algorithm 1.
   *
   * `res` comes back with the answer in **every** block, so a caller may read
   * whichever one its layout prefers and does not have to mask.
   */
  void Evaluate(Ct &res, const Ct &input, const EvkMap<word> &evk) const;

  /** @brief Device bytes the compiled plaintexts hold, 0 before `Compile`. */
  size_t GetPlaintextBytes() const;
};

}  // namespace cheddar
