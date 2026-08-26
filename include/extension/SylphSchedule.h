#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/Container.h"
#include "core/EvkMap.h"
#include "extension/BootContext.h"

namespace cheddar {

/**
 * @brief [SYLPH] figure 2's Private Prefill cycle, as one schedulable object.
 *
 *     slots @ EvalModEnd --non-linear--> --StC--> coeff --descend + PC-MM-->
 *     --HalfBoot--> slots @ EvalModEnd
 *
 * A Llama-3 decoder block is four turns of this loop and nothing else --
 * (RMSNorm, QKV), (SoftMax, AV and O), (RMSNorm, gate/up), (SiLU, down) -- so
 * the loop is worth building once rather than four times. What it owns is the
 * part that is not any operator's business: which level each leg starts and
 * ends on, and what the scale is when it gets there.
 *
 * ## The two joints, and why neither is free bookkeeping
 *
 * **HalfBoot's output is not at a canonical scale.** It lands carrying
 * `eval_mod_->end_scale_` -- around 2^58 against level 19's canonical 2^35 on
 * `bootparam_35`. Inside `Boot` that is harmless, because the only thing that
 * ever sees it is StC, and a linear transform tracks any input scale
 * multiplicatively (`Hoist.cu:510`). [SYLPH] puts non-linear work there
 * instead, and `EvalPoly` is *not* scale-agnostic: it asserts its compiled
 * input scale (`EvalPoly.cpp:112`), and every handler here compiles against
 * `param_.GetScale(input_level)` (`SiLu.cu:43`, `SoftMax.cu:88`). SiLu.h
 * records what a mismatched scale did before it was an assert -- RMSNorm
 * "exact in the clear, wrong by up to 29% encrypted".
 *
 * So a canonicalisation step exists, it costs one level, and it is the reason
 * the non-linear budget is `slack - 1` rather than `slack`. See `Canonicalise`.
 *
 * It is not avoidable by teaching the handlers an input scale. `BasisEval`'s
 * `final_scale_` is `working_scale / GetRescalePrimeProd(working_level)`
 * (`EvalPoly.cpp:151`), so squaring a basis element maps `s` to `s^2 / prod`
 * and the canonical scale is that map's fixed point. Started at 2^58 the chain
 * runs 2^81, 2^127 and overruns the modulus in three squarings. The shrink is
 * what keeps the polynomial's own scale bounded, not a convention.
 *
 * Measured against a host reference -- `DecodeCoeff` of the plaintext HalfBoot
 * was handed, placed by `AttentionPacking::SlotOfCoeff` -- it costs **0.4
 * bits**: 11.68 in, 11.27 out, against a scale-preserving rescale that is
 * bit-identical to its input at 11.68.
 *
 * **StC is not scale-free, and a stage arrives at the wrong scale.**
 * `stc_const_` is folded into the phase matrices' *values*, not their encoding
 * scale (`EvalSpecialFFT.cpp:282` multiplies the striped matrix), and it is
 * calibrated so that an input at EvalMod's end scale lands on the canonical
 * scale of `GetEndLevel()`. A stage that has been through `Canonicalise`
 * arrives at 2^35 instead of 2^58, and the output scale shrinks by that same
 * 2^-23. The declared scale tracks it exactly, so this is not an arithmetic
 * error -- it is a numerical one, and it is total: measured on the first run
 * of `TheTransportPreservesTheMessage`, the message came back at **-6.25 bits**
 * against Boot, buried under a noise floor that had not moved with it.
 *
 * `ToCoeff` therefore scales back up to `GetStCInputScale()` before calling
 * StC. That is a multiply by an exact 1.0 encoded at the ratio: it moves the
 * ciphertext's integers and its declared scale together, and costs no level
 * because nothing is rescaled. EvalMod's inflated end scale is exactly the
 * headroom StC was built to consume, so arriving with it makes a stage
 * indistinguishable from `Boot` at that point -- and `Boot` is verified, to a
 * max absolute difference of 0 against HalfBoot + StC. `ToCoeff` can then
 * declare Boot's own constant, `GetScale(GetEndLevel())`.
 *
 * The scale-up also disposes of `LevelDown`'s drift. StC is pinned to
 * `GetStCStartLevel()`, so a stage shallower than the slack has to descend to
 * meet it, and `LevelDown`'s rescales divide by the actual prime products
 * rather than the nominal 2^35 -- a part in 800 per level, worth up to 10 bits
 * of `Boot`'s precision until it was carried explicitly
 * (`BootContext.cpp:405`). That drift was never a bookkeeping error: the
 * declared scale after a descent is exact, it is simply not the nominal value.
 * Dividing by that same declared scale to compute the scale-up factor lands on
 * `GetStCInputScale()` however far the ciphertext fell, so no separate drift
 * term survives.
 *
 * ## What this does not do
 *
 * It does not run the operators and it does not run the product. Those have
 * their own handlers and their own tests; a schedule that also owned them
 * could not be tested apart from them. It carries the ciphertext between the
 * two domains and tells a caller, before anything is encrypted, whether the
 * plan closes.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SylphSchedule {
 private:
  using Ct = Ciphertext<word>;

  std::shared_ptr<const BootContext<word>> boot_;
  int num_slots_;

 public:
  /**
   * @brief One turn of the loop, as a level cost.
   *
   * `nonlinear_depth` is what the slot-domain operators spend, *not* counting
   * the canonicalisation this class inserts. `linear_depth` is what the
   * coefficient-domain leg spends: the descent to the product's ring plus the
   * product itself.
   */
  struct Stage {
    std::string name;
    int nonlinear_depth;
    int linear_depth;
  };

  /**
   * @param boot the BootContext, whose BootParameter carries the slack
   * @param num_slots slots per ciphertext; StC and HalfBoot must be prepared
   * for it
   */
  SylphSchedule(std::shared_ptr<const BootContext<word>> boot, int num_slots);

  // disable copying (or moving also)
  SylphSchedule(const SylphSchedule &) = delete;
  SylphSchedule &operator=(const SylphSchedule &) = delete;

  /** @brief Level `ToSlot` lands on, where the non-linear leg begins. */
  int GetSlotLevel() const;
  /** @brief Level StC is compiled at, where the non-linear leg must end. */
  int GetStCLevel() const;
  /** @brief Level StC leaves, where the linear leg begins. */
  int GetCoeffLevel() const;
  /** @brief The gap between EvalMod's end and StC's start. */
  int GetSlack() const;

  /** @brief Levels a stage's operators may spend, after canonicalisation. */
  int GetNonlinearBudget() const { return GetSlack() - 1; }
  /** @brief Levels the coefficient-domain leg may spend before level 0. */
  int GetLinearBudget() const { return GetCoeffLevel(); }

  /** @brief The scale `ToSlot` actually hands back, which is not canonical. */
  double GetSlotScale() const;
  /** @brief The scale the operators at `GetSlotLevel()` are compiled for. */
  double GetCanonicalSlotScale() const;

  /**
   * @brief The slack a set of stages needs: the deepest, plus the one level
   * of canonicalisation every stage pays.
   */
  static int RequiredSlack(const std::vector<Stage> &stages);

  /**
   * @brief Whether a stage fits this schedule.
   *
   * @param stage the stage to check
   * @param why if non-null, receives a description of the first thing that
   * does not fit, and is left untouched when everything does
   */
  bool Fits(const Stage &stage, std::string *why = nullptr) const;

  /** @brief The level plan as a table, for reports. */
  std::string DescribePlan(const std::vector<Stage> &stages) const;

  /**
   * @brief Bring a ciphertext to the canonical scale of the level below, for
   * one level.
   *
   * A multiply by an exact 1.0 and a rescale. The constant is encoded at
   * whatever scale makes the product land canonical, which for a HalfBoot
   * output on `bootparam_35` is 2^35 * 2^35 / 2^58 = 2^12 -- small, but 1.0 is
   * exact at any scale, so nothing is lost that an ordinary rescale would not
   * also lose. Measured exact: `canon.GetScale() / canonical - 1.0` came back
   * at 0, because the multiply and the rescale use the same actual prime
   * product.
   *
   * `magnitude` multiplies the message on the way through, and it is the
   * other half of the crossing constant. The two legs of the cycle want
   * opposite magnitudes -- ModRaise needs a small message, an operator's
   * polynomial wants exactly the interval it was fitted on -- and this
   * multiply is already being paid for, so growing the message back here
   * costs nothing. Without it the only ways to hand `SiLuHandler` an argument
   * on [-1, 1] while the bootstrap before it sees at most 0.5 are a constant
   * multiply (a level) or a fit interval twice as wide (a doubled degree,
   * which is also a level).
   *
   * The constant encoded is `restore * magnitude` at scale `factor`, and
   * `EncodeConstant` stores `round(number * scale)`, so a magnitude small
   * enough to round that product to zero is rejected rather than silently
   * annihilating the ciphertext -- the failure mode `ToCoeff` already guards
   * against.
   *
   * @param res output, at `level - 1` and `param_.GetScale(level - 1)`
   * @param x input at any scale
   * @param magnitude an extra factor on the message; 1.0 leaves it alone
   */
  void Canonicalise(Ct &res, const Ct &x, double magnitude = 1.0) const;

  /**
   * @brief The slot to coefficient leg: descend to StC's level, then StC.
   *
   * @param res output, coefficient-encoded at `GetCoeffLevel()`
   * @param x slot-encoded, at or above `GetStCLevel()`
   * @param evk_map supplies the rotation keys StC needs
   * @param min_ks whether to use minimum key-switching
   */
  void ToCoeff(Ct &res, const Ct &x, const EvkMap<word> &evk_map,
               bool min_ks = false) const;

  /**
   * @brief The coefficient to slot leg: HalfBoot.
   *
   * HalfBoot descends its input to level 0 itself and then forces EvalMod's
   * start scale (`BootContext.cpp:439, 470`), so an input above level 0 has
   * its scale drift silently absorbed. The descent is done here instead, and
   * the drift returned, so a caller that arrives above level 0 can correct its
   * own bookkeeping rather than lose the factor.
   *
   * @param res output, slot-encoded at `GetSlotLevel()` and `GetSlotScale()`
   * @param x coefficient-encoded, at any level
   * @param evk_map supplies the bootstrapping keys
   * @param min_ks whether to use minimum key-switching
   * @return the ratio by which the descent to level 0 moved the scale; 1.0
   * when the input was already there
   */
  double ToSlot(Ct &res, const Ct &x, const EvkMap<word> &evk_map,
                bool min_ks = false) const;

  /**
   * @brief Two ciphertexts up the same leg, on one HalfBoot.
   *
   * `BootContext::HalfBootPair` fills the imaginary axis that this pipeline
   * leaves empty, so a pair costs what one crossing costs. The contract it
   * needs -- payload in coefficients `0 .. N/2-1` only -- is exactly what
   * `ToCoeff` produces from real-axis slots, which is every ciphertext this
   * schedule ever hands back.
   *
   * @param res_lo slots of `lo`; identical to what `ToSlot(lo)` would give
   * @param res_hi slots of `hi`
   * @param lo coefficient-encoded, at any level
   * @param hi coefficient-encoded, at the same level and scale as `lo`
   * @param evk_map supplies the bootstrapping keys
   * @param min_ks whether to use minimum key-switching
   * @return the descent drift, as `ToSlot` returns it; it is common to the pair
   */
  double ToSlotPair(Ct &res_lo, Ct &res_hi, const Ct &lo, const Ct &hi,
                    const EvkMap<word> &evk_map, bool min_ks = false) const;
};

}  // namespace cheddar
