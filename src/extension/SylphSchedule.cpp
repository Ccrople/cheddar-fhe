#include "extension/SylphSchedule.h"

#include <algorithm>
#include <sstream>

#include "common/Assert.h"

namespace cheddar {

template <typename word>
SylphSchedule<word>::SylphSchedule(
    std::shared_ptr<const BootContext<word>> boot, int num_slots)
    : boot_{std::move(boot)}, num_slots_{num_slots} {
  AssertTrue(boot_ != nullptr, "SylphSchedule: no BootContext");
  AssertTrue(num_slots_ > 0 && num_slots_ <= boot_->param_.degree_ / 2,
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
  return boot_->GetBootParameter().GetEndLevel();
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
  os << "  ModRaise/CtS/EvalMod land slots at level " << GetSlotLevel()
     << ", scale " << GetSlotScale() << " (canonical there is "
     << GetCanonicalSlotScale() << ")\n";
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
void SylphSchedule<word>::Canonicalise(Ct &res, const Ct &x) const {
  const int level = boot_->param_.NPToLevel(x.GetNP());
  AssertTrue(level >= 1,
             "Canonicalise: nothing below level " + std::to_string(level) +
                 " to rescale into");
  const double target = boot_->param_.GetScale(level - 1);
  // Mult multiplies the scales and Rescale divides by the level's actual
  // prime product, so this factor is exactly what lands on `target`. Both
  // halves use the same `GetRescalePrimeProd`, which is the *actual* product
  // rather than the nominal 2^35, so unlike LevelDown this leaves no drift.
  const double factor =
      target * boot_->param_.GetRescalePrimeProd(level) / x.GetScale();
  Constant<word> one;
  boot_->encoder_.EncodeConstant(one, level, factor, 1.0);
  Ct tmp;
  boot_->Mult(tmp, x, one);
  boot_->Rescale(res, tmp);
}

template <typename word>
void SylphSchedule<word>::ToCoeff(Ct &res, const Ct &x,
                                  const EvkMap<word> &evk_map,
                                  bool min_ks /*= false*/) const {
  const int level = boot_->param_.NPToLevel(x.GetNP());
  const int stc_level = GetStCLevel();
  AssertTrue(level >= stc_level,
             "ToCoeff: the input is at level " + std::to_string(level) +
                 " but StC is compiled at " + std::to_string(stc_level) +
                 ", so the non-linear leg spent " +
                 std::to_string(stc_level - level) +
                 " levels too many. StC cannot be moved after the fact; the "
                 "slack is a BootParameter setting.");

  double stc_input_scale = 0.0;
  if (level > stc_level) {
    boot_->LevelDown(res, x, stc_level);
    stc_input_scale = res.GetScale();
    boot_->SlotToCoeff(res, num_slots_, res, evk_map, min_ks);
  } else {
    stc_input_scale = x.GetScale();
    boot_->SlotToCoeff(res, num_slots_, x, evk_map, min_ks);
  }
  res.SetNumSlots(num_slots_);

  // See the header: linear in the input scale, anchored on Boot. The descent's
  // drift needs no separate term because it is already inside
  // `stc_input_scale`, which is read after LevelDown rather than before.
  res.SetScale(boot_->param_.GetScale(GetCoeffLevel()) * stc_input_scale /
               boot_->GetStCInputScale());
}

template <typename word>
double SylphSchedule<word>::ToSlot(Ct &res, const Ct &x,
                                   const EvkMap<word> &evk_map,
                                   bool min_ks /*= false*/) const {
  const int level = boot_->param_.NPToLevel(x.GetNP());
  double drift = 1.0;
  if (level > 0) {
    const double before = x.GetScale();
    Ct low;
    boot_->LevelDown(low, x, 0);
    drift = low.GetScale() / before;
    boot_->HalfBoot(res, low, evk_map, min_ks);
  } else {
    boot_->HalfBoot(res, x, evk_map, min_ks);
  }
  return drift;
}

template class SylphSchedule<uint32_t>;
template class SylphSchedule<uint64_t>;

}  // namespace cheddar
