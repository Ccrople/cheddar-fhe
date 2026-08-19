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

// One fixture per landing level. The BootParameter is fixed at construction, so
// one context cannot land in two places -- that is a property of the design, not
// of the test: a schedule wanting several landing levels needs several prepared
// contexts, and their keys and transforms are what it pays for.
#define BOOT_TO(name, landing)                                             \
  class name : public BootLevelBase {                                      \
   protected:                                                              \
    int BootMaxLevel() const override { return (landing) + kBootDepth; }   \
  };                                                                       \
  INSTANTIATE_TEST_SUITE_P(                                                \
      Cheddar, name, testing::Values("bootparam_30.json"),                 \
      [](const testing::TestParamInfo<name::ParamType> &info) {            \
        std::string p = info.param;                                        \
        std::replace(p.begin(), p.end(), '.', '_');                        \
        return p;                                                          \
      })

BOOT_TO(BootToDefault, 19);
BOOT_TO(BootTo14, 14);
BOOT_TO(BootTo9, 9);

TEST_P(BootToDefault, LandsWhereTheParameterSetPutsIt) { RunBoot(19, "Boot-19"); }
TEST_P(BootTo14, LandsFiveLevelsLower) { RunBoot(14, "Boot-14"); }
TEST_P(BootTo9, LandsTenLevelsLower) { RunBoot(9, "Boot-9"); }
