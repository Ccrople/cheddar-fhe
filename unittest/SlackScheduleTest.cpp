// Slack between EvalMod and StC: the gap [SYLPH] figure 2's schedule needs.
//
// WHAT 1.5w GOT WRONG, AND WHY THIS EXISTS. BootParameter derives
// GetStCStartLevel() as max_level - num_cts - EvalMod, and HalfBoot is Boot
// minus StC, so HalfBoot lands on precisely StC's compiled level. The gap is
// zero and no non-linear operator fits in it -- not even RoPE at one level. I
// read that as "the parameter set is six levels short" and said a taller set
// was needed. It is not: Boot's input has to be at level 0 regardless, so on
// bootparam_35 the sixteen levels below StC's output are the linear phase's
// working range and almost none of them are used. Moving the conversion down
// by D takes the slack from there and costs no max_level at all.
//
// ONE UNKNOWN AT A TIME. The two things that could go wrong are separable, and
// 1.5t's five dead hypotheses came from testing them together:
//
//   * the LEVEL. StC's phases are LinearTransforms pinned to
//     GetStCStartLevel() - i with plaintexts encoded there. Compiling them
//     lower is the whole change, and either it works or it does not.
//   * the SCALE. StC receives EvalMod's end scale inside Boot. A caller who
//     puts real work in the gap changes what arrives.
//
// SlackIsInvisibleToBoot fixes the level and leaves the scale alone: Boot with
// slack D is the same operation as Boot without it, so it is checked against
// the no-slack answer through its own known-good path. WorkInTheGapReachesStC
// then spends the D levels on real kernels and asks whether StC still accepts
// what comes out.
//
// Boot is the reference in both, because that is what broke the deadlock last
// time: comparing against a known-good result one step away, rather than
// against an expectation about what the result should be.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Testbed.h"

using word = uint32_t;

namespace {
// One level more than RMSNorm at the degree its measured spread allows (3),
// so the gap is big enough to be a real schedule and small enough to leave the
// linear phase ten levels on bootparam_35.
constexpr int kSlack = 4;
}  // namespace

class SlackTestbed : public Testbed32 {
 protected:
  int BootSlackLevels() const override { return kSlack; }
};

// The arithmetic the knob is supposed to produce, before any ciphertext moves.
TEST_P(SlackTestbed, TheGapOpensWhereItShould) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const BootParameter &bp = boot->GetBootParameter();

  EXPECT_EQ(bp.GetNumSlackLevels(), kSlack);
  EXPECT_EQ(bp.GetStCStartLevel(), bp.GetEvalModEndLevel() - kSlack);
  EXPECT_EQ(bp.GetEndLevel(), bp.GetStCStartLevel() - bp.num_stc_levels_);
  std::cout << "CtS " << bp.GetCtSStartLevel() << " -> EvalMod "
            << bp.GetEvalModStartLevel() << " -> ends " << bp.GetEvalModEndLevel()
            << " -> [gap " << kSlack << "] -> StC " << bp.GetStCStartLevel()
            << " -> " << bp.GetEndLevel() << std::endl;

  // The levels come from below, not from a taller parameter set.
  EXPECT_EQ(bp.GetMaxLevel(), param_->max_level_)
      << "the slack was taken out of max_level, which is the thing it is "
         "supposed not to need";
  EXPECT_GE(bp.GetEndLevel(), 0)
      << "the gap ate the linear phase's working range";
  std::cout << "levels left under StC's output for the linear phase: "
            << bp.GetEndLevel() << std::endl;
}

// LEVEL ONLY. Boot puts nothing in the gap, so with slack it crosses with a
// LevelDown -- a multiply by a level-down constant and a rescale, which leaves
// the declared scale alone. So StC still receives EvalMod's end scale, and the
// only thing that changed is which level its phases were compiled at.
//
// If this passes, compiling StC lower is sound and the rest is about scale.
TEST_P(SlackTestbed, SlackIsInvisibleToBoot) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, 0);

  Ciphertext<word> got;
  boot->Boot(got, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const int level = param_->NPToLevel(got.GetNP());
  std::cout << "Boot with slack " << kSlack << " landed at level " << level
            << " (no slack lands at " << level + kSlack << ")" << std::endl;
  EXPECT_EQ(level, boot->GetBootParameter().GetEndLevel());

  std::vector<Complex> out;
  DecryptAndDecode(out, got);
  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_slots; i++) {
    worst = std::max({worst, std::abs(out[i].real() - msg[i].real()),
                      std::abs(out[i].imag() - msg[i].imag())});
    absmax = std::max(absmax, std::abs(msg[i]));
  }
  std::cout << "Boot through the gap: max abs err " << worst << " ("
            << -std::log2(worst / absmax) << " bits)" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 12.0)
      << "compiling StC below where EvalMod ends broke bootstrapping itself, "
         "so the gap is not usable and nothing downstream matters";
}

// SCALE TOO. Now spend the gap on real kernels instead of a LevelDown, and ask
// whether StC still accepts the result.
//
// The work is a multiply by the constant one followed by a rescale, repeated
// kSlack times. That is deliberately the cheapest thing that is still a real
// level: it runs the same Mult and Rescale every operator runs, it consumes
// exactly one level per iteration, and it leaves the message untouched -- so
// the answer is comparable to Boot's directly, with no reference to write.
//
// The constant is encoded at GetRescalePrimeProd(level), which is the scale
// that makes multiply-then-rescale scale-preserving. Whether a real operator's
// canonical-scale plaintexts also survive is the next question and not this
// one; getting this far first is the point.
TEST_P(SlackTestbed, WorkInTheGapReachesStC) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const BootParameter &bp = boot->GetBootParameter();
  const int num_slots = param_->degree_ / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, 0);

  // The reference: Boot, which crosses the gap with a LevelDown.
  Ciphertext<word> want;
  boot->Boot(want, ct, interface_->GetEvkMap());

  // The schedule: HalfBoot into the slot domain, work, then StC.
  Ciphertext<word> half;
  boot->HalfBoot(half, ct, interface_->GetEvkMap());
  const int landed = param_->NPToLevel(half.GetNP());
  std::cout << "HalfBoot landed at " << landed << ", StC is compiled at "
            << bp.GetStCStartLevel() << ", so the gap is "
            << landed - bp.GetStCStartLevel() << " levels" << std::endl;
  ASSERT_EQ(landed, bp.GetEvalModEndLevel());
  ASSERT_EQ(landed - bp.GetStCStartLevel(), kSlack)
      << "there is no gap to put work in";

  for (int i = 0; i < kSlack; i++) {
    const int level = param_->NPToLevel(half.GetNP());
    Constant<word> one;
    context_->encoder_.EncodeConstant(
        one, level, param_->GetRescalePrimeProd(level), 1.0);
    Ciphertext<word> tmp;
    context_->Mult(tmp, half, one);
    context_->Rescale(half, tmp);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::cout << "  gap step " << i << ": level " << level << " -> "
              << param_->NPToLevel(half.GetNP()) << ", scale "
              << half.GetScale() << std::endl;
  }
  ASSERT_EQ(param_->NPToLevel(half.GetNP()), bp.GetStCStartLevel())
      << "the work did not land on StC's compiled level";
  std::cout << "scale arriving at StC " << half.GetScale()
            << ", scale it was compiled for " << boot->GetStCInputScale()
            << " (ratio " << half.GetScale() / boot->GetStCInputScale() << ")"
            << std::endl;

  Ciphertext<word> got;
  boot->SlotToCoeff(got, num_slots, half, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  got.SetNumSlots(num_slots);
  got.SetScale(want.GetScale());
  ASSERT_EQ(param_->NPToLevel(got.GetNP()), param_->NPToLevel(want.GetNP()));

  std::vector<Complex> a, b;
  DecryptAndDecode(a, want);
  DecryptAndDecode(b, got);
  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_slots; i++) {
    worst = std::max({worst, std::abs(a[i].real() - b[i].real()),
                      std::abs(a[i].imag() - b[i].imag())});
    absmax = std::max(absmax, std::abs(a[i]));
  }
  std::cout << "work in the gap vs Boot: max abs diff " << worst << " ("
            << -std::log2(worst / absmax) << " bits relative to " << absmax
            << ")" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 12.0)
      << "StC did not accept what the gap produced; the printed scale ratio "
         "above is the first thing to look at";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, SlackTestbed, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<SlackTestbed::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
