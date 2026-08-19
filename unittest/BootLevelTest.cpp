// Level-targeted bootstrapping.
//
// WHY. [SYLPH] figure 2 marks where bootstrapping happens, and the private
// prefill path runs non-linear work at a high level, linear work at a low one,
// then bootstraps. Reproducing that schedule needs a bootstrap that lands where
// the schedule wants it, not where the parameter set happens to put it. Two
// things depend on it:
//
//   * SoftMax's 8-level main track. [SYLPH] figure 2 bootstraps the narrow
//     auxiliary track separately, which is what keeps the norm square, the
//     affine map and the inverse square root off the main track. Measured
//     without it, ours is 13 levels.
//   * Sizing the parameter set at all. If the deepest bootstrap-to-bootstrap
//     segment needs only 10 levels then nothing needs 19, and every operation
//     gets shorter: HRot is 159 us at level 0 against 1403 us at max level.
//
// WHAT CHANGED. `BootParameter` already parameterised CoeffToSlot, EvalMod and
// SlotToCoeff by its own max_level -- GetCtSStartLevel, GetEvalModStartLevel and
// GetStCStartLevel all derive from it -- and AddRequiredRotations already
// requested keys at GetMaxLevel(). ModUp was the one step still hard-wired to
// `param_.max_level_`, so a BootContext always climbed to the top of the
// parameter set whatever its BootParameter said. `ModUpToLevel` fixes that: the
// basis-lift kernel iterates over the destination's primes, so no target basis
// is special, and `mod_max_intt_const_` depends only on the level-zero base and
// the ring degree, so it is reused unchanged.
//
// A BootContext lands at `max_level - (CtS + EvalMod + StC)` = `max_level - 15`
// for these presets, so asking for a lower landing level now means less work
// rather than a LevelDown afterwards.
//
// WHAT IS MEASURED. Correctness first -- the message must survive a bootstrap
// that climbs less far -- then the time, because a correctness-only result would
// prove the mechanism and none of the value.

#include <algorithm>
#include <string>
#include <vector>

#include "Testbed.h"

namespace {
constexpr int kNumSlots = 1 << 15;
constexpr int kWarmUp = 3;
// CtS 4 + EvalMod 8 (Log2Ceil(27) coefficients + 3 double-angle) + StC 3.
constexpr int kBootDepth = 15;
constexpr int kBootCtsEvalMod = 12;
constexpr int kBootStcLevels = 3;
}  // namespace

// The shared body. It has to be a fixture method: the message helpers and the
// context are protected members of Testbed.
class BootLevelBase : public Testbed32 {
 protected:
  void RunBoot(int expected_landing, const char *label) {
    using word = uint32_t;
    auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
    ASSERT_NE(boot_context, nullptr);

    boot_context->PrepareEvalMod();
    boot_context->PrepareEvalSpecialFFT(kNumSlots);
    EvkRequest req;
    boot_context->AddRequiredRotations(req, kNumSlots);
    interface_->PrepareRotationKey(req);

    std::vector<Complex> msg, res;
    GenerateRandomMessage(msg, kNumSlots);
    Ciphertext<word> ct, ct_res;

    __ProfileStart(label, kWarmUp, EncodeAndEncrypt(ct, msg, 0));
    boot_context->Boot(ct_res, ct, interface_->GetEvkMap());
    __ProfileEnd(label);

    const int landed = param_->NPToLevel(ct_res.GetNP());
    std::cout << label << ": climbed to " << BootMaxLevel() << ", landed at "
              << landed << " (expected " << expected_landing << ")"
              << std::endl;
    EXPECT_EQ(landed, expected_landing)
        << "the bootstrap did not land where its BootParameter asked, so ModUp "
           "is still climbing to the parameter set's maximum";

    DecryptAndDecode(res, ct_res);
    CompareMessages(msg, res);
  }
};

// WHAT THIS MEASURED, AND THE ANSWER IS NOT WHAT I EXPECTED.
//
// Removing ModUp's hard-wiring is necessary but not sufficient. BootContext's
// constructor requires, at BootContext.cpp:42,
//
//     param.max_level_ == boot_param.max_level_  &&
//     param.default_encryption_level_ == boot_param.GetStCStartLevel()
//
// and `bootparam_30`'s grafting schedule explains why. Its level_config adds
// **one** prime per level up to 22 and **two** per level from 23 to 34 -- twelve
// double-prime levels, which is exactly CtS(4) + EvalMod(8). EvalMod needs about
// 58 bits per level and two 30-bit primes supply it; one does not. The preset
// reserves its double-prime block for the bootstrap by construction.
//
// So a bootstrap that lands lower needs a purpose-built parameter set:
// default_encryption_level = L + 3, max_level = L + 15, level_config with the
// double-prime block wherever CtS and EvalMod will sit, and the resulting QP
// still inside the security bound. Relaxing the assertion instead would put
// EvalMod on 30-bit levels and destroy its precision silently, which is the
// worst available outcome.
//
// The tests below therefore record two things: that the default landing level
// works through ModUpToLevel, and that an inconsistent BootParameter is
// rejected rather than quietly mis-evaluated.

class BootToDefault : public BootLevelBase {};

INSTANTIATE_TEST_SUITE_P(
    Cheddar, BootToDefault, testing::Values("bootparam_30.json"),
    [](const testing::TestParamInfo<BootToDefault::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });

TEST_P(BootToDefault, LandsWhereItsBootParameterAsks) {
  // Still 19 here, because that is what this preset's prime schedule allows --
  // but it now gets there through ModUpToLevel(boot_param_.GetMaxLevel())
  // rather than a hard-wired param_.max_level_.
  RunBoot(19, "Boot-19");
}

TEST_P(BootToDefault, ALowerTargetNeedsItsOwnParameterSet) {
  // Documented as a checked property, not left as a crash for the next person
  // to rediscover. bootparam_30 lands at 19 and nowhere else.
  const int stc_start = param_->default_encryption_level_;
  EXPECT_EQ(stc_start + kBootStcLevels, param_->max_level_ - kBootCtsEvalMod)
      << "the preset's default encryption level is pinned to the StC start";

  // The double-prime block sits exactly on CtS + EvalMod. If that ever stops
  // being true, a lower landing level may have become reachable in this preset
  // and this test should be revisited.
  int single = 0, doubled = 0;
  for (int l = 1; l <= param_->max_level_; l++) {
    const int added = param_->LevelToNP(l).num_main_ -
                      param_->LevelToNP(l - 1).num_main_;
    if (l > param_->max_level_ - kBootCtsEvalMod) {
      doubled += (added == 2) ? 1 : 0;
    } else {
      single += (added == 1) ? 1 : 0;
    }
  }
  std::cout << "levels above the StC start with two primes: " << doubled
            << " of " << kBootCtsEvalMod << "; levels below with one: "
            << single << std::endl;
  EXPECT_EQ(doubled, kBootCtsEvalMod)
      << "EvalMod needs ~58 bits per level, so the bootstrap region must be "
         "the double-prime region; a landing level below that needs a "
         "parameter set built for it";
}
