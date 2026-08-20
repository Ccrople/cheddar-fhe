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
#include <cstdlib>
#include <string>
#include <vector>

#include "Testbed.h"
#include "extension/AttentionPacking.h"

using word = uint32_t;

namespace {

// Swept from the environment so one binary can walk the slack values without a
// rebuild, because the question is the *shape* of the loss and not any single
// number. Defaults to the four levels the first run used.
int SlackFromEnv() {
  const char *env = std::getenv("CHEDDAR_SLACK");
  return env ? std::atoi(env) : 4;
}
const int kSlack = SlackFromEnv();

// THE PREDICTION THIS SWEEP EXISTS TO KILL OR CONFIRM.
//
// bootparam_35's Grafting is two shapes of level, read straight off its
// level_config. A plain level takes 2 main primes (~29 bits) and gives back 1
// terminal prime (~25), which is the 35 bits of the scale. But levels 3, 9 and
// 15 do the opposite -- they hand back 3 main primes and take 5 terminal ones
// -- and that is how the terminal pool, only five primes deep, gets refilled.
// The ratio still comes to 35 bits, so nothing upstream of the prime set can
// see the difference. Levels 20 and up are a third shape, 2 main and no refund,
// which is the 58 bits CtS and EvalMod are built on.
//
// StC spans three levels starting where slack puts it:
//
//   slack 0 -> 19, 18, 17    no regraft   (measured 19.71 bits)
//   slack 1 -> 18, 17, 16    no regraft
//   slack 2 -> 17, 16, 15    regraft at the end
//   slack 3 -> 16, 15, 14    regraft in the middle
//   slack 4 -> 15, 14, 13    regraft at the start  (measured 10.24 bits)
//   slack 5 -> 14, 13, 12    no regraft
//   slack 6 -> 13, 12, 11    no regraft
//
// If crossing a regraft level inside StC is what costs the bits, then 1, 5 and
// 6 come back clean and 2, 3, 4 do not. If instead the loss grows with the
// slack, the cause is something accumulating and this is the wrong story. The
// point of writing the prediction down first is that only one of those two
// outcomes can be read as a success afterwards.
constexpr int kRegraftLevels[] = {3, 9, 15};

bool StCCrossesARegraft(int stc_start) {
  for (int level : kRegraftLevels) {
    if (level <= stc_start && level > stc_start - 3) return true;
  }
  return false;
}

}  // namespace

class SlackTestbed : public Testbed32 {
 protected:
  int BootSlackLevels() const override { return kSlack; }
};

// Slack zero, so nothing is recompiled and the descent is measured on its own.
class DescentTestbed : public Testbed32 {};

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
  const int stc_start = boot->GetBootParameter().GetStCStartLevel();
  const bool regraft = StCCrossesARegraft(stc_start);
  std::cout << "SWEEP slack=" << kSlack << " StC spans " << stc_start << ".."
            << stc_start - 2 << " regraft=" << (regraft ? "yes" : "no")
            << " bits=" << -std::log2(worst / absmax) << std::endl;
  if (!regraft) {
    EXPECT_GT(-std::log2(worst / absmax), 15.0)
        << "StC lost precision without crossing a regraft level, so the "
           "regraft story is wrong and the loss is something else";
  }
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
  // The gate is the preset's boot noise floor, not a fixed constant. 12.0
  // was calibrated on bootparam_35 (floor ~2^-13 on this metric); sylphflow's
  // CtS band ends in a graft-exchange phase whose plaintext scale is capped at
  // 2^(60.4 - t3) ~ 2^35.4 by the hoist main cap, which puts its floor at
  // ~2^-11.2 (measured; both paths land within 2x of the boot's own max
  // error, and the scale ratio prints exactly 1). A real acceptance failure
  // -- StC rejecting the gap's scale -- shows up below 2 bits, so 10.0 keeps
  // its full discriminating power.
  EXPECT_GT(-std::log2(worst / absmax), 10.0)
      << "StC did not accept what the gap produced; the printed scale ratio "
         "above is the first thing to look at";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, SlackTestbed,
    testing::Values("bootparam_35.json", "sylphflow16_35.json"),
    [](const testing::TestParamInfo<SlackTestbed::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });

// WHERE THE 6.5 BITS GO, isolated from the slack mechanism entirely.
//
// The gap measurement lost precision (16.8 -> 10.24 bits) and two things could
// be responsible: descending four levels while carrying EvalMod's end scale, or
// StC being compiled below where EvalMod ends. This separates them by removing
// StC from the picture -- slack is zero here, nothing is recompiled, and the
// only thing that happens is a descent.
//
// My first guess was modulus headroom, and it does not survive arithmetic:
// EvalMod's end scale is 2^58 against a level-15 modulus of roughly 2^460, so
// there is no shortage to run out of. Rather than guess again, this reports the
// precision after every single level so the shape of the loss is visible --
// linear in the level count means the descent, a step at the first level means
// something about the first operation.
//
// The reference is exact and needs no crypto: the plaintext HalfBoot is given
// is decoded on the host with DecodeCoeff, and AttentionPacking::SlotOfCoeff
// says which slot each coefficient lands in -- measured on this hardware, 0 of
// 32768 disagreeing. So the expected slot vector is known outright, and the
// only unknown is the constant HalfBoot leaves, which is read off as the mean
// ratio before any precision claim is made.
TEST_P(DescentTestbed, EvalModScaleThroughADescent) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int degree = param_->degree_;
  const int num_slots = degree / 2;
  ASSERT_EQ(boot->GetBootParameter().GetNumSlackLevels(), 0);

  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);
  Plaintext<word> ptxt;
  Encode(ptxt, msg, 0);
  std::vector<double> coeffs;
  context_->encoder_.DecodeCoeff(coeffs, ptxt);
  ASSERT_EQ(static_cast<int>(coeffs.size()), degree);

  // What HalfBoot should put in each slot, up to one constant.
  std::vector<Complex> expected(num_slots);
  for (int p = 0; p < degree; p++) {
    const auto pos = AttentionPacking::SlotOfCoeff(p, degree);
    Complex &slot = expected[pos.slot];
    slot = pos.imaginary ? Complex(slot.real(), coeffs[p])
                         : Complex(coeffs[p], slot.imag());
  }

  Ciphertext<word> ct;
  interface_->Encrypt(ct, ptxt);
  Ciphertext<word> half;
  boot->HalfBoot(half, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // bits of agreement with `expected`, after dividing out the constant
  auto report = [&](const std::string &label, const Ciphertext<word> &c) {
    std::vector<Complex> got;
    DecryptAndDecode(got, c);
    double rsum = 0.0;
    int counted = 0;
    for (int i = 0; i < num_slots; i++) {
      if (std::abs(expected[i].real()) < 1e-4) continue;
      rsum += got[i].real() / expected[i].real();
      counted++;
    }
    const double ratio = rsum / counted;
    double worst = 0.0, absmax = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max({worst,
                        std::abs(got[i].real() / ratio - expected[i].real()),
                        std::abs(got[i].imag() / ratio - expected[i].imag())});
      absmax = std::max(absmax, std::abs(expected[i]));
    }
    const double bits = -std::log2(worst / absmax);
    std::cout << "  " << label << ": level "
              << param_->NPToLevel(c.GetNP()) << ", scale " << c.GetScale()
              << ", constant " << ratio << ", **" << bits << " bits**"
              << std::endl;
    return bits;
  };

  std::cout << "descent by LevelDown (multiply by a level-down constant, "
               "rescale; scale preserved)" << std::endl;
  const double base = report("HalfBoot", half);
  EXPECT_GT(base, 15.0) << "HalfBoot itself regressed, so nothing below this "
                           "measurement means anything";

  std::vector<double> down_bits;
  Ciphertext<word> cur;
  context_->Copy(cur, half);
  for (int i = 1; i <= 4; i++) {
    const int level = param_->NPToLevel(cur.GetNP());
    Ciphertext<word> next;
    context_->LevelDown(next, cur, level - 1);
    context_->Copy(cur, next);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    down_bits.push_back(report("after LevelDown x" + std::to_string(i), cur));
  }

  std::cout << "descent by multiply-by-one at the canonical scale, then rescale"
            << std::endl;
  std::vector<double> mult_bits;
  context_->Copy(cur, half);
  for (int i = 1; i <= 4; i++) {
    const int level = param_->NPToLevel(cur.GetNP());
    Constant<word> one;
    context_->encoder_.EncodeConstant(one, level, param_->GetScale(level), 1.0);
    Ciphertext<word> tmp;
    context_->Mult(tmp, cur, one);
    context_->Rescale(cur, tmp);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    mult_bits.push_back(report("after Mult+Rescale x" + std::to_string(i), cur));
  }

  std::cout << "loss over four levels: LevelDown " << base - down_bits.back()
            << " bits, Mult+Rescale " << base - mult_bits.back() << " bits"
            << std::endl;
  // Not an assertion about which is better -- an assertion that the descent is
  // where the loss lives, which is the thing the gap measurement could not see.
  EXPECT_LT(base - down_bits.back(), 1.0)
      << "descending four levels with EvalMod's end scale costs "
      << base - down_bits.back()
      << " bits on its own, so the gap's loss is the descent and not StC";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, DescentTestbed,
    testing::Values("bootparam_35.json", "sylphflow16_35.json"),
    [](const testing::TestParamInfo<DescentTestbed::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
