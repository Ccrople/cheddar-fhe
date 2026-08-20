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
 * **StC is compiled at one level and one input scale, and a stage arrives at
 * neither.** `EvalSpecialFFT` pins the three StC phases to
 * `GetStCStartLevel()` downward, so a stage that spends fewer levels than the
 * slack has to descend to meet it. `LevelDown`'s rescales divide by the actual
 * prime products, which miss the nominal 2^35 by up to a part in 800, so the
 * scale that arrives is not the scale that was declared -- and that drift is
 * not cosmetic. It was worth up to 10 bits of the bootstrap's precision until
 * `Boot` was taught to carry it (`BootContext.cpp:405`). `ToCoeff` does the
 * descent and the arithmetic in one place so no stage has to remember.
 *
 * ## What ToCoeff declares, and where that comes from
 *
 * StC folds `stc_const_` into the phase matrices' *values*, not their encoding
 * scale (`EvalSpecialFFT.cpp:282` multiplies the striped matrix), so the
 * message comes out scaled by it and `Hoist`'s scale tracking cannot see it.
 * `Boot` compensates by declaring the nominal end scale outright. Every step
 * of the chain is linear in the input scale, so the one anchor point that is
 * verified -- `Boot`, whose result `half_boot_test` matches to a max absolute
 * difference of 0 -- extends to any other input scale by its ratio:
 *
 *     declared = GetScale(GetEndLevel()) * (arriving scale / GetStCInputScale())
 *
 * At `arriving == GetStCInputScale()` that is `Boot`'s own formula, drift
 * included, because the drift is already inside the arriving scale.
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
   * output is around 2^6 -- small, but 1.0 is exact at any scale, so nothing
   * is lost that an ordinary rescale would not also lose.
   *
   * @param res output, at `level - 1` and `param_.GetScale(level - 1)`
   * @param x input at any scale
   */
  void Canonicalise(Ct &res, const Ct &x) const;

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
};

}  // namespace cheddar
