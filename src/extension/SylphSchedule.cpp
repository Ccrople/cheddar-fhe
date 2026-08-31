#include "extension/SylphSchedule.h"

#include <algorithm>
#include <sstream>

#include "common/Assert.h"
#include "extension/CiModuleBasis.h"

namespace cheddar {

template <typename word>
SylphSchedule<word>::SylphSchedule(
    std::shared_ptr<const BootContext<word>> boot, int num_slots)
    : boot_{std::move(boot)}, num_slots_{num_slots} {
  AssertTrue(boot_ != nullptr, "SylphSchedule: no BootContext");
  // MaxNumSlots(), not degree/2: the same number on the ordinary ring and
  // `degree` on the conjugate-invariant one, where a full-slot cycle is
  // 65536 real slots. Nothing else in this class reads the ring -- `ToSlot`
  // and `ToCoeff` pass `num_slots_` straight to HalfBoot and SlotToCoeff,
  // which take it as an argument -- so this assertion was the whole of what
  // kept the cycle off R+.
  AssertTrue(num_slots_ > 0 && num_slots_ <= boot_->param_.MaxNumSlots(),
             "SylphSchedule: invalid slot count");
  // Without slack, StC starts exactly where EvalMod ends and the non-linear
  // leg has nowhere to go. That is the shipped presets' configuration and it
  // is correct for Boot; it just cannot run this cycle, and saying so here is
  // better than a level assert several hundred milliseconds later.
  AssertTrue(GetSlack() >= 1,
             "SylphSchedule: the BootParameter has no slack, so there is no "
             "room between EvalMod and StC for the non-linear leg. Construct "
             "the BootContext with num_slack_levels >= RequiredSlack(stages).");
}

template <typename word>
int SylphSchedule<word>::GetSlotLevel() const {
  return boot_->GetBootParameter().GetEvalModEndLevel();
}

template <typename word>
int SylphSchedule<word>::GetStCLevel() const {
  return boot_->GetBootParameter().GetStCStartLevel();
}

template <typename word>
int SylphSchedule<word>::GetCoeffLevel() const {
  if (basis_ != nullptr) return GetStCLevel() - basis_->GetStCNumLevels();
  return boot_->GetBootParameter().GetEndLevel();
}

template <typename word>
void SylphSchedule<word>::SetModuleBasis(const CiModuleBasis<word> *basis) {
  if (basis != nullptr) {
    AssertTrue(boot_->param_.conjugate_invariant_,
               "SylphSchedule: the module basis is a conjugate-invariant "
               "object");
    AssertTrue(basis->GetStCLevel() == GetStCLevel(),
               "SylphSchedule: the module StC is compiled at level " +
                   std::to_string(basis->GetStCLevel()) +
                   " but this schedule's StC level is " +
                   std::to_string(GetStCLevel()));
    AssertTrue(basis->GetCtSLevel() ==
                   boot_->GetBootParameter().GetCtSStartLevel(),
               "SylphSchedule: the module CtS must be compiled at the "
               "BootParameter's CtS start level for HalfBootModule");
    AssertTrue(basis->GetStCNumLevels() <= GetStCLevel(),
               "SylphSchedule: the module StC spends more levels than the "
               "schedule has below its StC level");
  }
  basis_ = basis;
}

template <typename word>
double SylphSchedule<word>::ModuleStCConst(int num_levels) const {
  // `stc_const_` is calibrated so that an input at EvalMod's end scale lands
  // on `GetScale(GetEndLevel())` once `ToCoeff` re-declares the scale; a StC
  // that lands elsewhere wants that level's canonical scale in its place.
  AssertTrue(num_levels >= 1 && num_levels <= GetStCLevel(),
             "ModuleStCConst: the module StC's level count is not within the "
             "schedule's StC level");
  const auto &p = boot_->param_;
  return boot_->GetStCConst() * p.GetScale(GetStCLevel() - num_levels) /
         p.GetScale(boot_->GetBootParameter().GetEndLevel());
}

template <typename word>
int SylphSchedule<word>::GetSlack() const {
  return boot_->GetBootParameter().GetNumSlackLevels();
}

template <typename word>
double SylphSchedule<word>::GetSlotScale() const {
  return boot_->GetStCInputScale();
}

template <typename word>
double SylphSchedule<word>::GetCanonicalSlotScale() const {
  return boot_->param_.GetScale(GetSlotLevel());
}

template <typename word>
int SylphSchedule<word>::RequiredSlack(const std::vector<Stage> &stages) {
  int deepest = 0;
  for (const auto &stage : stages) {
    deepest = std::max(deepest, stage.nonlinear_depth);
  }
  // Every stage pays one level to canonicalise HalfBoot's output scale, and
  // one slack serves all four stages -- the shallow ones descend to meet StC,
  // which the descent measurement showed costs no precision (19.7101 to
  // 19.7099 bits over four levels).
  return deepest + 1;
}

template <typename word>
bool SylphSchedule<word>::Fits(const Stage &stage, std::string *why) const {
  if (stage.nonlinear_depth > GetNonlinearBudget()) {
    if (why != nullptr) {
      std::ostringstream os;
      os << stage.name << ": the non-linear leg needs "
         << stage.nonlinear_depth << " levels but the slack of " << GetSlack()
         << " leaves " << GetNonlinearBudget()
         << " after canonicalisation. Raise num_slack_levels to "
         << (stage.nonlinear_depth + 1) << ".";
      *why = os.str();
    }
    return false;
  }
  if (stage.linear_depth > GetLinearBudget()) {
    if (why != nullptr) {
      std::ostringstream os;
      os << stage.name << ": the linear leg needs " << stage.linear_depth
         << " levels but StC leaves it at level " << GetCoeffLevel()
         << ". Lowering the slack raises the non-linear budget and lowers "
            "this one by the same amount.";
      *why = os.str();
    }
    return false;
  }
  return true;
}

template <typename word>
std::string SylphSchedule<word>::DescribePlan(
    const std::vector<Stage> &stages) const {
  std::ostringstream os;
  os << "[SYLPH] figure 2 cycle on a max_level " << boot_->param_.max_level_
     << " parameter set, slack " << GetSlack() << "\n";
  os << "  ModRaise/CtS/EvalMod land slots at level " << GetSlotLevel();
  if (boot_->IsEvalModPrepared()) {
    os << ", scale " << GetSlotScale() << " (canonical there is "
       << GetCanonicalSlotScale() << ")";
  } else {
    // The plan is level arithmetic and must not need a prepared context. Only
    // the landing scale does, because it comes out of EvalMod's compiled chain.
    os << ", at a scale only PrepareEvalMod() can report";
  }
  os << "\n";
  os << "  canonicalise: " << GetSlotLevel() << " -> " << (GetSlotLevel() - 1)
     << ", so the non-linear budget is " << GetNonlinearBudget() << "\n";
  os << "  StC spans " << GetStCLevel() << " -> " << GetCoeffLevel()
     << ", so the linear budget is " << GetLinearBudget() << "\n";
  for (const auto &stage : stages) {
    std::string why;
    const bool ok = Fits(stage, &why);
    os << "  " << (ok ? "fits" : "DOES NOT FIT") << "  " << stage.name
       << ": non-linear " << stage.nonlinear_depth << "/"
       << GetNonlinearBudget() << ", linear " << stage.linear_depth << "/"
       << GetLinearBudget();
    if (!ok) os << "  <-- " << why;
    os << "\n";
  }
  return os.str();
}

template <typename word>
void SylphSchedule<word>::Canonicalise(Ct &res, const Ct &x,
                                       double magnitude) const {
  const int level = boot_->param_.NPToLevel(x.GetNP());
  AssertTrue(level >= 1,
             "Canonicalise: nothing below level " + std::to_string(level) +
                 " to rescale into");
  const double target = boot_->param_.GetScale(level - 1);
  // Restore the magnitude HalfBoot divided out. The two legs of the cycle want
  // opposite things: ModRaise needs the message a factor
  // 2^-log_message_ratio below q0, which is what that parameter reserves, and
  // the operators need it as large as possible because CKKS precision is
  // absolute and a value only gets the bits it occupies below the scale.
  //
  // Measured, sizing the input so the bootstrap could carry it at all: the
  // message arrived at max 0.0094 and RMSNorm returned 5.22 bits, against 9.64
  // bits for the same operator fed the same relative precision at max 0.48 in
  // a no-bootstrap sweep. The 51x is the whole difference.
  //
  // So the slot leg runs at operator magnitude and the coefficient leg at
  // bootstrap magnitude, and the two constants are exact inverses that ride
  // along on multiplies already being paid for -- this one, and ToCoeff's
  // scale-up. Neither costs a level.
  const double restore =
      std::pow(2.0, boot_->GetBootParameter().GetLogMessageRatio());
  // Mult multiplies the scales and Rescale divides by the level's actual
  // prime product, so this factor is exactly what lands on `target`. Both
  // halves use the same `GetRescalePrimeProd`, which is the *actual* product
  // rather than the nominal 2^35, so unlike LevelDown this leaves no drift.
  const double factor =
      target * boot_->param_.GetRescalePrimeProd(level) / x.GetScale();
  // The caller's own half of the crossing constant rides on the same
  // multiply. See the header: this is what lets an operator receive its
  // argument on exactly the interval its polynomial owns while the bootstrap
  // that precedes it still carries half of it.
  const double number = restore * magnitude;
  AssertTrue(number * factor >= 1.0,
             "Canonicalise: the constant would encode as round(" +
                 std::to_string(number * factor) +
                 "), which rounds to zero and annihilates the ciphertext. The "
                 "magnitude asked for is too small for this level's scale.");
  Constant<word> one;
  boot_->encoder_.EncodeConstant(one, level, factor, number);
  Ct tmp;
  boot_->Mult(tmp, x, one);
  boot_->Rescale(res, tmp);
}

template <typename word>
void SylphSchedule<word>::ToCoeff(Ct &res, const Ct &x,
                                  const EvkMap<word> &evk_map,
                                  bool min_ks /*= false*/,
                                  bool native_basis /*= false*/) const {
  const int level = boot_->param_.NPToLevel(x.GetNP());
  const int stc_level = GetStCLevel();
  AssertTrue(level >= stc_level,
             "ToCoeff: the input is at level " + std::to_string(level) +
                 " but StC is compiled at " + std::to_string(stc_level) +
                 ", so the non-linear leg spent " +
                 std::to_string(stc_level - level) +
                 " levels too many. StC cannot be moved after the fact; the "
                 "slack is a BootParameter setting.");

  // SETTLE A PENDING RESCALE FIRST.
  //
  // `RmsNormHandler::Apply` ends with `Mult(res, res, weight_pt_)` and no
  // Rescale (`RmsNorm.cu`), so its output carries scale^2 -- 2^70 where the
  // level's canonical scale is 2^35. Cheddar's convention is that Mult does
  // not rescale and the caller does, so this is the operator's contract and
  // not a defect, but it means an operator's stated depth is levels *consumed*
  // and there is one more owed.
  //
  // Left alone it is silent and total. The scale-up below would ask for a
  // factor of 2^58 / 2^70 = 2^-12, and with the message-ratio factor the
  // constant handed to EncodeConstant is 2^-17, which rounds to **zero** --
  // annihilating the ciphertext. Measured: every coefficient came back 0, and
  // the level assertions all still passed, which is exactly how it hid.
  Ct settled;
  const Ct *src = &x;
  int cur = level;
  if (x.GetScale() > boot_->GetStCInputScale()) {
    AssertTrue(cur > stc_level,
               "ToCoeff: the input owes a rescale -- its scale is " +
                   std::to_string(x.GetScale()) + " against StC's " +
                   std::to_string(boot_->GetStCInputScale()) +
                   " -- but it is already at StC's level " +
                   std::to_string(stc_level) +
                   ", so there is nowhere to spend it. Budget the operator at "
                   "one level more than it consumes.");
    boot_->Rescale(settled, x);
    src = &settled;
    cur = boot_->param_.NPToLevel(settled.GetNP());
  }

  Ct descended;
  if (cur > stc_level) {
    boot_->LevelDown(descended, *src, stc_level);
    src = &descended;
  }

  // THE SCALE-UP, and why it is not optional. Measured on the first run of
  // TheTransportPreservesTheMessage: a canonical 2^35 input produced a result
  // at scale 2^12 -- 2^-23 of Boot's -- and the message came back buried, at
  // -6.25 bits against Boot's answer.
  //
  // StC is not scale-free. `stc_const_` is folded into the phase matrices'
  // values (`EvalSpecialFFT.cpp:282`) and is calibrated so that an input at
  // EvalMod's end scale lands on the canonical scale of `GetEndLevel()`. Feed
  // it anything smaller and the output scale shrinks with it, which is
  // arithmetically correct -- the declared scale tracks it exactly -- and
  // numerically ruinous, because the message then occupies twelve bits above a
  // noise floor that did not move.
  //
  // EvalMod's inflated end scale is what buys StC that headroom, so the fix is
  // to arrive with it. A multiply by an exact 1.0 encoded at the ratio moves
  // the ciphertext's integers up and its declared scale with them, and costs
  // no level because nothing is rescaled. At level 11 on `bootparam_35` this
  // puts a message of order one at 2^58 against a modulus near 2^435.
  //
  // It also removes the need to carry LevelDown's drift. The drift was never a
  // bookkeeping error -- the declared scale after a descent is exact, it is
  // just not the nominal 2^35 -- so dividing by that same declared scale here
  // lands on `GetStCInputScale()` however far the ciphertext fell.
  // The value carried here is `r`, not 1.0, which undoes the magnitude
  // Canonicalise restored. StC supplies the 1/r that makes `Boot` message
  // preserving, so without this the loop would gain a factor of
  // 2^log_message_ratio every turn and the state would leave the range
  // ModRaise can carry after a couple of them. Riding on the scale-up's own
  // multiply, it is free.
  const double up_factor = boot_->GetStCInputScale() / src->GetScale();
  const double undo_restore =
      std::pow(2.0, -boot_->GetBootParameter().GetLogMessageRatio());
  // Never encode a constant that rounds to zero. EncodeConstant stores
  // round(number * scale), so a product below 1/2 is a silent annihilation --
  // the failure this guard exists to name, which cost a day precisely because
  // every level assertion downstream of it still passed.
  AssertTrue(up_factor * undo_restore >= 1.0,
             "ToCoeff: the scale-up constant would encode as " +
                 std::to_string(up_factor * undo_restore) +
                 ", which rounds to zero and annihilates the ciphertext. The "
                 "input's scale is " + std::to_string(src->GetScale()) +
                 " against StC's " + std::to_string(boot_->GetStCInputScale()) +
                 "; an input above StC's scale owes a rescale.");
  Constant<word> up;
  boot_->encoder_.EncodeConstant(up, stc_level, up_factor, undo_restore);
  boot_->Mult(res, *src, up);

  const bool module = basis_ != nullptr && !native_basis;
  if (module) {
    // The module StC: the slots become the element's MODULE coordinates,
    // with `ModuleStCConst()` folded so the bookkeeping below is the native
    // one's. It takes `basis_->GetStCNumLevels()` levels, which is what
    // `GetCoeffLevel()` reports.
    basis_->EvaluateStC(boot_, res, res, evk_map);
  } else {
    boot_->SlotToCoeff(res, num_slots_, res, evk_map, min_ks);
  }
  res.SetNumSlots(num_slots_);
  // Now exactly Boot's situation, so exactly Boot's constant.
  res.SetScale(boot_->param_.GetScale(
      module ? GetCoeffLevel() : boot_->GetBootParameter().GetEndLevel()));
}

template <typename word>
double SylphSchedule<word>::ToSlot(Ct &res, const Ct &x,
                                   const EvkMap<word> &evk_map,
                                   bool min_ks /*= false*/) const {
  const int level = boot_->param_.NPToLevel(x.GetNP());
  double drift = 1.0;
  Ct low;
  const Ct *src = &x;
  if (level > 0) {
    const double before = x.GetScale();
    boot_->LevelDown(low, x, 0);
    drift = low.GetScale() / before;
    src = &low;
  }
  if (basis_ != nullptr) {
    boot_->HalfBootModule(res, *src, evk_map, *basis_);
  } else {
    boot_->HalfBoot(res, *src, evk_map, min_ks);
  }
  return drift;
}

template <typename word>
double SylphSchedule<word>::ToSlotPair(Ct &res_lo, Ct &res_hi, const Ct &lo,
                                       const Ct &hi,
                                       const EvkMap<word> &evk_map,
                                       bool min_ks /*= false*/) const {
  AssertTrue(basis_ == nullptr,
             "ToSlotPair: the pair form reads the native basis; on the "
             "module basis use ToSlot");
  const int level = boot_->param_.NPToLevel(lo.GetNP());
  AssertTrue(level == boot_->param_.NPToLevel(hi.GetNP()),
             "ToSlotPair: the two ciphertexts must be at the same level");
  double drift = 1.0;
  if (level > 0) {
    // The same descent both sides, taken here rather than inside HalfBootPair,
    // for the reason `ToSlot` takes it: the drift has to be visible.
    const double before = lo.GetScale();
    Ct low_lo, low_hi;
    boot_->LevelDown(low_lo, lo, 0);
    boot_->LevelDown(low_hi, hi, 0);
    drift = low_lo.GetScale() / before;
    boot_->HalfBootPair(res_lo, res_hi, low_lo, low_hi, evk_map, min_ks);
  } else {
    boot_->HalfBootPair(res_lo, res_hi, lo, hi, evk_map, min_ks);
  }
  return drift;
}

template <typename word>
double SylphSchedule<word>::ToSlotSplit(Ct &res_lo, Ct &res_hi,
                                        const Ct &merged,
                                        const EvkMap<word> &evk_map,
                                        bool min_ks /*= false*/) const {
  AssertTrue(basis_ == nullptr,
             "ToSlotSplit: the split form reads the native basis; on the "
             "module basis use ToSlot");
  const int level = boot_->param_.NPToLevel(merged.GetNP());
  double drift = 1.0;
  if (level > 0) {
    const double before = merged.GetScale();
    Ct low;
    boot_->LevelDown(low, merged, 0);
    drift = low.GetScale() / before;
    boot_->HalfBootSplit(res_lo, res_hi, low, evk_map, min_ks);
  } else {
    boot_->HalfBootSplit(res_lo, res_hi, merged, evk_map, min_ks);
  }
  return drift;
}

template class SylphSchedule<uint32_t>;
template class SylphSchedule<uint64_t>;

}  // namespace cheddar
