// The [SYLPH] figure 2 cycle, assembled.
//
// Everything up to here proved the pieces separately: the operators against
// host references, HalfBoot against Boot, the slot/coefficient permutation
// against a GPU measurement, the slack gap against Boot again. None of them
// answers whether the pieces *join*, and the joins are where the level and
// scale bookkeeping lives.
//
// Three tests, escalating, each one step from something already known good.
//
//   1. ThePlanClosesForAWholeBlock  -- arithmetic only. Does a Llama-3-8B
//      decoder block's four stages fit the level budget, and with how much
//      room? Nothing is encrypted, so a wrong answer here costs seconds
//      instead of the twenty minutes the other two take.
//
//   2. TheTransportPreservesTheMessage -- the cycle with no operator in it.
//      Boot is the reference, because Boot performs exactly the same descent
//      and the same StC and is verified to a max absolute difference of 0
//      against HalfBoot + StC. What this adds is Canonicalise in the middle,
//      which changes the scale StC is fed, and so exercises ToCoeff's scale
//      formula rather than reproducing Boot's constant.
//
//   3. RmsNormRunsInTheGapAndReachesStC -- one real stage, on real weights.
//      RmsNormTest already validates RMSNorm at an ordinary encryption level;
//      the only new thing is that its input now comes out of HalfBoot, which
//      lands at a non-canonical scale. That is precisely the failure SiLu.h
//      records ("exact in the clear, wrong by up to 29% encrypted"), so the
//      test asserts the operator's measured depth matches the plan and then
//      hands the result to ToCoeff.
//
// THE PREDICTION, written down before the run.
//
// Canonicalise is exact -- both its multiply and its rescale go through the
// same GetRescalePrimeProd, which is the actual prime product and not the
// nominal 2^35 -- so test 2 should match Boot to within ordinary bootstrap
// precision (the slack sweep measured 17.1-17.6 bits across slack 0..6) and
// not to some ratio near 2^23, which is what a dropped canonicalisation factor
// would look like. If test 2 comes back off by a clean power of two, the
// factor is the bug and not the arithmetic.
//
// Test 3's risk is different and is about magnitude, not bookkeeping: EvalPoly
// asserts its compiled input scale, so a scale error aborts loudly rather than
// returning a wrong number. The thing that can silently go wrong is the
// *value* range -- RMSNorm's polynomial window is calibrated on the clear
// input, and a bootstrap's output carries its own error into the mean square.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Testbed.h"
#include "extension/AttentionPacking.h"
#include "extension/RmsNorm.h"
#include "extension/SylphSchedule.h"

using word = uint32_t;

namespace {

// Slack is a BootParameter setting, so it is fixed when the context is built
// and cannot be varied inside a test. The env override exists so the whole
// file can be re-run at another value without a rebuild, the same way
// SlackScheduleTest sweeps.
int SlackFromEnv() {
  const char *env = std::getenv("CHEDDAR_SLACK");
  return env ? std::atoi(env) : 8;
}
const int kSlack = SlackFromEnv();

constexpr int kAllTokens = 128;
constexpr int kChannels = 4096;
constexpr int kTokens = 64;      // encrypted user segment, clear of the sinks
constexpr int kFirstToken = 64;
constexpr double kEps = 1e-5;

// RMSNorm at Chebyshev degree 9: one level for the square, ceil(log2(10)) = 4
// for the polynomial, two to apply the result and the weight. RmsNorm.h states
// the rule; test 3 measures the number rather than trusting it.
constexpr int kRmsNormDepth = 7;

// The four turns of figure 2's loop that make one Llama-3-8B decoder block.
// Depths are the handlers' own documented costs, not estimates:
// RmsNorm.h's 1 + ceil(log2(d+1)) + 2, SoftMax.h's GetMainTrackDepth of 7 with
// the normalisation bootstrapped off the main track, SiLu.h's degree-31 fit.
// The linear leg is the descent to the product's ring plus the product, which
// PipelineChainTest runs at 2.
std::vector<SylphSchedule<word>::Stage> BlockStages() {
  return {
      {"attn RMSNorm -> QKV", kRmsNormDepth, 2},
      {"SoftMax -> AV and O", 7, 2},
      {"ffn RMSNorm -> gate/up", kRmsNormDepth, 2},
      {"SiLU -> down", 5, 2},
  };
}

std::string DataDir() {
  const char *env = std::getenv("LLAMA3_REAL_DIR");
  return env ? std::string(env) : std::string();
}

bool ReadF32(const std::string &path, size_t count, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<float> raw(count);
  f.read(reinterpret_cast<char *>(raw.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  if (static_cast<size_t>(f.gcount()) != count * sizeof(float)) return false;
  out.assign(raw.begin(), raw.end());
  return true;
}

double MeanSquare(const std::vector<double> &x, int token) {
  double s = 0.0;
  for (int c = 0; c < kChannels; c++) {
    const double v = x[static_cast<size_t>(token) * kChannels + c];
    s += v * v;
  }
  return s / kChannels;
}

}  // namespace

class CycleTestbed : public Testbed32 {
 protected:
  int BootSlackLevels() const override { return kSlack; }
};

// ---------------------------------------------------------------------------

TEST_P(CycleTestbed, ThePlanClosesForAWholeBlock) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  SylphSchedule<word> sched(boot, param_->degree_ / 2);

  // The levels need no preparation, but the landing scale is EvalMod's own
  // and this test asserts on it. PrepareEvalMod compiles the polynomial and
  // touches no rotation key, so it is not what makes bootstrapping slow.
  boot->PrepareEvalMod();

  const auto stages = BlockStages();
  std::cout << sched.DescribePlan(stages);

  // The slack the block needs, computed from the stages rather than assumed.
  const int required = SylphSchedule<word>::RequiredSlack(stages);
  std::cout << "the block needs slack " << required << ", this context has "
            << sched.GetSlack() << std::endl;
  EXPECT_GE(sched.GetSlack(), required)
      << "rebuild the context with num_slack_levels >= " << required;

  // The level arithmetic, spelled out so a parameter change that breaks it
  // says which relation went.
  const BootParameter &bp = boot->GetBootParameter();
  EXPECT_EQ(sched.GetSlotLevel(), bp.GetEvalModEndLevel());
  EXPECT_EQ(sched.GetStCLevel(), sched.GetSlotLevel() - kSlack);
  EXPECT_EQ(sched.GetCoeffLevel(), sched.GetStCLevel() - bp.num_stc_levels_);
  EXPECT_EQ(sched.GetNonlinearBudget(), kSlack - 1);

  // Both budgets have to be positive at once, and they trade against each
  // other one for one -- that trade is the whole content of the schedule.
  EXPECT_GT(sched.GetNonlinearBudget(), 0);
  EXPECT_GT(sched.GetLinearBudget(), 0);
  for (const auto &stage : stages) {
    std::string why;
    EXPECT_TRUE(sched.Fits(stage, &why)) << why;
  }

  // HalfBoot's landing scale is the reason a canonicalisation level exists.
  // If these ever coincide, the level is free and GetNonlinearBudget should
  // go back to being the slack itself.
  std::cout << "HalfBoot lands at scale " << sched.GetSlotScale()
            << " against a canonical " << sched.GetCanonicalSlotScale()
            << " (ratio 2^"
            << std::log2(sched.GetSlotScale() / sched.GetCanonicalSlotScale())
            << ")" << std::endl;
  EXPECT_NE(sched.GetSlotScale(), sched.GetCanonicalSlotScale());
}

// ---------------------------------------------------------------------------

TEST_P(CycleTestbed, TheTransportPreservesTheMessage) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);
  SylphSchedule<word> sched(boot, num_slots);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, 0);

  // The reference: Boot, which makes the same trip with its own LevelDown.
  Ciphertext<word> want;
  boot->Boot(want, ct, interface_->GetEvkMap());

  // The cycle: coefficients in, slots out, canonicalise, coefficients again.
  Ciphertext<word> slots;
  const double drift = sched.ToSlot(slots, ct, interface_->GetEvkMap());
  EXPECT_EQ(drift, 1.0) << "the input was already at level 0";
  ASSERT_EQ(param_->NPToLevel(slots.GetNP()), sched.GetSlotLevel());
  EXPECT_EQ(slots.GetScale(), sched.GetSlotScale());

  Ciphertext<word> canon;
  sched.Canonicalise(canon, slots);
  ASSERT_EQ(param_->NPToLevel(canon.GetNP()), sched.GetSlotLevel() - 1);
  // The claim the header makes: this leaves no drift, because the multiply and
  // the rescale divide by the same actual prime product.
  const double canonical = param_->GetScale(sched.GetSlotLevel() - 1);
  std::cout << "canonicalised to " << canon.GetScale() << ", canonical is "
            << canonical << ", ratio - 1 = "
            << (canon.GetScale() / canonical - 1.0) << std::endl;
  EXPECT_LT(std::abs(canon.GetScale() / canonical - 1.0), 1e-12)
      << "Canonicalise is supposed to be exact";

  // Spend two levels the way an operator would, then let ToCoeff descend the
  // rest -- so both the caller's rescales and ToCoeff's LevelDown are on the
  // path, which is the arrangement every real stage will have.
  for (int i = 0; i < 2; i++) {
    const int level = param_->NPToLevel(canon.GetNP());
    Constant<word> one;
    context_->encoder_.EncodeConstant(
        one, level, param_->GetRescalePrimeProd(level), 1.0);
    Ciphertext<word> tmp;
    context_->Mult(tmp, canon, one);
    context_->Rescale(canon, tmp);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  Ciphertext<word> got;
  sched.ToCoeff(got, canon, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(param_->NPToLevel(got.GetNP()), sched.GetCoeffLevel());
  ASSERT_EQ(param_->NPToLevel(got.GetNP()), param_->NPToLevel(want.GetNP()));

  // No SetScale here. ToCoeff scales up to StC's compiled input scale, so the
  // two answers should now agree on the scale as well as the message; the
  // first run of this test had them 2^-23 apart, which is what the scale-up
  // exists to remove. Boot's own value carries its LevelDown drift and the
  // cycle's does not, so they are close rather than equal.
  std::cout << "Boot declares scale " << want.GetScale() << ", the cycle "
            << got.GetScale() << " (ratio " << got.GetScale() / want.GetScale()
            << ")" << std::endl;
  EXPECT_LT(std::abs(got.GetScale() / want.GetScale() - 1.0), 1e-2)
      << "the two are no longer at comparable scales, so ToCoeff's scale-up "
         "did not land where StC was calibrated";

  std::vector<Complex> a, b;
  DecryptAndDecode(a, want);
  DecryptAndDecode(b, got);
  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_slots; i++) {
    worst = std::max({worst, std::abs(a[i].real() - b[i].real()),
                      std::abs(a[i].imag() - b[i].imag())});
    absmax = std::max(absmax, std::abs(a[i]));
  }
  const double bits = -std::log2(worst / absmax);
  std::cout << "cycle vs Boot: max abs diff " << worst << " (" << bits
            << " bits relative to " << absmax << ")" << std::endl;
  EXPECT_GT(bits, 12.0)
      << "the transport lost the message. A ratio near a clean power of two "
         "in the scales printed above means a dropped canonicalisation "
         "factor; anything else means the descent.";
}

// ---------------------------------------------------------------------------

// THE SCALE DROP, WITH NOTHING ELSE IN THE PICTURE.
//
// WhereCanonicalisationCosts localised 9.6 bits to Canonicalise, but every
// number in it came out of a bootstrap, so "the rescale is at fault" and "the
// bootstrap's noise interacts badly with the rescale" are still the same
// measurement. This removes the bootstrap: a freshly encrypted ciphertext,
// scaled up by a known amount and then rescaled back down to canonical, which
// is exactly the shape of Canonicalise and nothing else.
//
// THE PREDICTION. A rescale's added error is a rounding term -- tau_b + tau_a*s
// with tau in [-1/2, 1/2] -- so it is *absolute*, a few hundred integer units
// for a ternary secret of weight 32768, call it 2^9. Every variant lands its
// message on the same canonical 2^35, and each one's incoming noise is divided
// by exactly the factor it was multiplied by, so **every row should read the
// same**. If the residual instead falls off with the shrink, the error
// introduced at this transition is not a rounding term and is far larger than
// one -- and level 19 -> 18 is a Grafting regraft, two main primes dropped and
// one terminal added, which is the part of it that is not an ordinary rescale.
//
// A flat table means Canonicalise is innocent and the interaction is with the
// bootstrap's noise; a falling table means the operation itself is.
TEST_P(CycleTestbed, TheScaleDropIsWhereTheBitsGo) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  const int level = boot->GetBootParameter().GetEvalModEndLevel();

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);

  std::cout << "level " << level << " -> " << (level - 1) << ", canonical "
            << param_->GetScale(level) << " -> " << param_->GetScale(level - 1)
            << ", rescale product " << param_->GetRescalePrimeProd(level)
            << std::endl;

  for (int shrink : {0, 6, 12, 18, 23}) {
    Ciphertext<word> ct;
    EncodeAndEncrypt(ct, msg, level);

    // Up by 2^shrink, no rescale: integers and declared scale move together,
    // so the message is untouched and so is its precision.
    if (shrink > 0) {
      Constant<word> up;
      context_->encoder_.EncodeConstant(up, level, std::pow(2.0, shrink), 1.0);
      context_->Mult(ct, ct, up);
    }

    // Down to canonical, which is Canonicalise's own arithmetic.
    const double factor = param_->GetScale(level - 1) *
                          param_->GetRescalePrimeProd(level) / ct.GetScale();
    Constant<word> down;
    context_->encoder_.EncodeConstant(down, level, factor, 1.0);
    Ciphertext<word> tmp;
    context_->Mult(tmp, ct, down);
    Ciphertext<word> out;
    context_->Rescale(out, tmp);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<Complex> got;
    DecryptAndDecode(got, out);
    double num = 0.0, den = 0.0, absmax = 0.0;
    for (int i = 0; i < num_slots; i++) {
      num += got[i].real() * msg[i].real() + got[i].imag() * msg[i].imag();
      den += msg[i].real() * msg[i].real() + msg[i].imag() * msg[i].imag();
      absmax = std::max(absmax, std::abs(msg[i]));
    }
    const double fit = num / den;
    double worst = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max({worst, std::abs(fit * msg[i].real() - got[i].real()),
                        std::abs(fit * msg[i].imag() - got[i].imag())});
    }
    std::cout << "  up 2^" << shrink << " then back down: constant " << factor
              << ", factor " << fit << ", residual "
              << -std::log2(worst / (absmax * std::abs(fit))) << " bits"
              << std::endl;
  }
}

// ---------------------------------------------------------------------------

// WHERE CANONICALISATION'S TEN BITS GO.
//
// Measured, all at slack 8 on bootparam_35: Boot alone reaches 16.94 bits and
// SlackScheduleTest's HalfBoot + gap + StC -- which never leaves EvalMod's end
// scale -- reaches 15.86. The cycle, whose only additions are Canonicalise and
// the scale-up that undoes it, reaches 5.63. So the slack is not the problem
// and StC is not the problem; carrying the ciphertext at 2^35 is.
//
// Two candidates, and they are separated by *when* the scale goes back up:
//
//   B: canonicalise and immediately scale back up, then descend at 2^58.
//      Isolates the down-and-up round trip from everything else.
//   C: canonicalise, descend seven levels at 2^35, scale up at StC's level.
//      This is what ToCoeff does today.
//
// If B is clean and C is not, the cost is the descent at a low scale -- every
// LevelDown rescale adds a rounding error that is absolute, so it is 2^23
// larger relative to a message at 2^35 than to one at 2^58. If B is also dirty,
// the round trip itself is lossy and Canonicalise is the wrong instrument.
//
// A is the control: no canonicalisation at all, which should reproduce
// SlackScheduleTest's 15.86 through this file's own code path.
TEST_P(CycleTestbed, WhereCanonicalisationCosts) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);
  SylphSchedule<word> sched(boot, num_slots);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, 0);

  Ciphertext<word> want;
  boot->Boot(want, ct, interface_->GetEvkMap());
  std::vector<Complex> a;
  DecryptAndDecode(a, want);

  // Multiply by an exact 1.0 at whatever scale is asked for. At `factor`
  // equal to the level's rescale product this is scale-preserving; at any
  // other value it moves the declared scale and the integers together.
  auto scale_by = [&](Ciphertext<word> &res, const Ciphertext<word> &x,
                      double factor) {
    const int level = param_->NPToLevel(x.GetNP());
    Constant<word> c;
    context_->encoder_.EncodeConstant(c, level, factor, 1.0);
    context_->Mult(res, x, c);
  };

  // TWO NUMBERS, NOT ONE, and the distinction is the whole point.
  //
  // A wrong declared scale and a lost bit are both "the answer differs from
  // Boot", and the first bisection could not tell them apart -- which is why
  // its control came back at 11.82 against SlackScheduleTest's 15.86 for what
  // should be the same computation. The difference is that SlackScheduleTest
  // ends with `got.SetScale(want.GetScale())`, copying Boot's declared scale
  // outright and so erasing any factor before it measures anything.
  //
  // So: fit the best single scalar first (least squares over every slot), then
  // report the residual around it. The fitted factor is bookkeeping and costs
  // nothing that a SetScale cannot repair; the residual is precision and is
  // gone for good.
  auto report = [&](const char *name, const Ciphertext<word> &got) {
    std::vector<Complex> b;
    DecryptAndDecode(b, got);
    double num = 0.0, den = 0.0, absmax = 0.0;
    for (int i = 0; i < num_slots; i++) {
      num += b[i].real() * a[i].real() + b[i].imag() * a[i].imag();
      den += a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
      absmax = std::max(absmax, std::abs(a[i]));
    }
    const double fit = num / den;
    double worst = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max({worst, std::abs(fit * a[i].real() - b[i].real()),
                        std::abs(fit * a[i].imag() - b[i].imag())});
    }
    const double bits = -std::log2(worst / (absmax * std::abs(fit)));
    std::cout << "  " << name << ": factor " << fit << " (1 - factor = "
              << (1.0 - fit) << "), residual " << bits << " bits" << std::endl;
    return bits;
  };

  Ciphertext<word> landed;
  sched.ToSlot(landed, ct, interface_->GetEvkMap());

  // Before any variant: what did HalfBoot deliver, and what does Canonicalise
  // do to it on its own? These are one operation apart, so if the round trip
  // is lossy it shows here with nothing else in the picture. The comparison is
  // against Boot's slot values, which differ from HalfBoot's by StC's
  // compensation -- a constant, which is exactly what the fitted factor
  // absorbs.
  report("HalfBoot output", landed);
  {
    Ciphertext<word> canon_only;
    sched.Canonicalise(canon_only, landed);
    report("after Canonicalise alone", canon_only);
  }

  // A -- control. Never leaves EvalMod's end scale.
  double bits_a = 0.0;
  {
    Ciphertext<word> v;
    scale_by(v, landed, param_->GetRescalePrimeProd(sched.GetSlotLevel()));
    context_->Rescale(v, v);
    Ciphertext<word> out;
    sched.ToCoeff(out, v, interface_->GetEvkMap());
    bits_a = report("A no canonicalisation", out);
  }

  // B -- down to 2^35 and straight back to 2^58, then descend.
  double bits_b = 0.0;
  {
    Ciphertext<word> canon;
    sched.Canonicalise(canon, landed);
    Ciphertext<word> up;
    scale_by(up, canon, sched.GetSlotScale() / canon.GetScale());
    Ciphertext<word> out;
    sched.ToCoeff(out, up, interface_->GetEvkMap());
    bits_b = report("B canonicalise then undo it at once", out);
  }

  // C -- what ToCoeff does today: descend at the canonical scale.
  double bits_c = 0.0;
  {
    Ciphertext<word> canon;
    sched.Canonicalise(canon, landed);
    Ciphertext<word> out;
    sched.ToCoeff(out, canon, interface_->GetEvkMap());
    bits_c = report("C descend at 2^35, scale up at StC", out);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "verdict: "
            << (bits_b - bits_c > 4.0
                    ? "the descent at a low scale is the cost"
                    : (bits_a - bits_b > 4.0
                           ? "the down-and-up round trip is itself lossy"
                           : "neither variant explains it"))
            << std::endl;
  EXPECT_GT(bits_a, 12.0) << "the control should reproduce SlackScheduleTest";
}

// ---------------------------------------------------------------------------

TEST_P(CycleTestbed, RmsNormRunsInTheGapAndReachesStC) {
  const std::string dir = DataDir();
  if (dir.empty()) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  std::vector<double> x, w;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kAllTokens * kChannels, x));
  ASSERT_TRUE(ReadF32(dir + "/attn_norm.f32", kChannels, w));

  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);
  SylphSchedule<word> sched(boot, num_slots);

  // The stage has to fit before anything is encrypted.
  const SylphSchedule<word>::Stage stage{"attn RMSNorm -> QKV", kRmsNormDepth,
                                         2};
  std::string why;
  ASSERT_TRUE(sched.Fits(stage, &why)) << why;

  // Same calibration as RmsNormTest: alpha_L from the geometric mean, and beta
  // to put |x| near one, which is free because RMSNorm is scale invariant.
  double lo = 1e300, hi = 0.0, log_sum = 0.0;
  for (int t = kFirstToken; t < kFirstToken + kTokens; t++) {
    const double ms = MeanSquare(x, t);
    lo = std::min(lo, ms);
    hi = std::max(hi, ms);
    log_sum += std::log(ms);
  }
  ASSERT_LT(hi / lo, 30.0) << "the segment is outside [SYLPH]'s window";
  const double alpha = 1.0 / std::exp(log_sum / kTokens);
  const double beta = std::sqrt(alpha);
  const double alpha_scaled = alpha / (beta * beta);
  const double eps_scaled = kEps * beta * beta;

  // The operator's input level is where Canonicalise leaves HalfBoot's output,
  // one below the landing level. This is the joint the whole class exists for.
  const int op_level = sched.GetSlotLevel() - 1;
  RmsNormHandler<word> rms(context_, kTokens, kChannels, alpha_scaled, op_level,
                           eps_scaled, 6.0, 9);
  const int num_ct = rms.GetNumCiphertexts();
  std::cout << "T=" << kTokens << " H=" << kChannels << " -> " << num_ct
            << " ciphertexts; RMSNorm compiled at level " << op_level
            << ", StC waits at " << sched.GetStCLevel() << std::endl;
  for (int d : rms.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, op_level);
  }

  // Encrypt in the COEFFICIENT domain at level 0, which is where a real cycle
  // hands the data over: the previous stage's product ends there. The first
  // run of this test used the slot encoder here and RMSNorm returned noise --
  // ToSlot is HalfBoot, coefficients in and slots out, so slot-encoded input
  // arrives at the operator permuted by the bit reversal and its per-token
  // reduction sums over the wrong grouping.
  //
  // AttentionPacking::CoeffOfSlot is the inverse of that permutation, measured
  // on this hardware with 0 of 32768 slots disagreeing, so writing the value
  // for slot s into coefficient CoeffOfSlot({s, false}) makes HalfBoot deliver
  // the token-fastest layout RmsNormHandler documents.
  const int channels_per_ct = num_slots / kTokens;
  const int degree = param_->degree_;
  std::vector<Ciphertext<word>> coeff_cts(num_ct);
  std::vector<std::vector<Complex>> wts(num_ct);
  const double root_alpha = std::sqrt(alpha_scaled);
  for (int i = 0; i < num_ct; i++) {
    std::vector<double> coeffs(degree, 0.0);
    wts[i].assign(num_slots, Complex(0.0, 0.0));
    for (int s = 0; s < num_slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = kFirstToken + (s % kTokens);
      const int p = AttentionPacking::CoeffOfSlot({s, false}, degree);
      coeffs[p] = beta * x[static_cast<size_t>(t) * kChannels + c];
      wts[i][s] = Complex(w[c] * root_alpha, 0.0);
    }
    Plaintext<word> ptxt;
    context_->encoder_.EncodeCoeff(ptxt, 0, DetermineScale(0), coeffs);
    interface_->Encrypt(coeff_cts[i], ptxt);
  }

  // Leg one: up into the slot domain, then onto the operator's scale.
  std::vector<Ciphertext<word>> slot_cts(num_ct);
  for (int i = 0; i < num_ct; i++) {
    Ciphertext<word> landed;
    sched.ToSlot(landed, coeff_cts[i], interface_->GetEvkMap());
    sched.Canonicalise(slot_cts[i], landed);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(param_->NPToLevel(slot_cts[0].GetNP()), op_level);

  // WHAT ACTUALLY ARRIVED, before the operator gets a chance to hide it.
  //
  // RMSNorm is calibrated on the clear input's magnitude -- alpha_L puts the
  // mean square inside the polynomial's window -- so a bootstrap that returns
  // the message at some fraction of its input silently moves the argument out
  // of that window. BootParameter::GetLogMessageRatio documents exactly this
  // for a round trip through the encoding boundary. Rather than assume the
  // factor is 1, or assume it is 2^-5, measure it: the ratio is reported and
  // asserted near 1, so a mismatch names itself instead of appearing as a
  // wrong RMSNorm.
  {
    std::vector<Complex> got0;
    DecryptAndDecode(got0, slot_cts[0]);
    double num = 0.0, den = 0.0;
    for (int s = 0; s < num_slots; s++) {
      const int c = s / kTokens;  // ciphertext 0, so no channel offset
      const int t = kFirstToken + (s % kTokens);
      const double want_v = beta * x[static_cast<size_t>(t) * kChannels + c];
      num += got0[s].real() * want_v;
      den += want_v * want_v;
    }
    const double ratio = num / den;
    std::cout << "HalfBoot round trip: the message came back at " << ratio
              << "x of what went in (log2 " << std::log2(std::abs(ratio))
              << ")" << std::endl;
    EXPECT_LT(std::abs(ratio - 1.0), 0.05)
        << "the bootstrap moved the magnitude, so RMSNorm's window no longer "
           "matches its calibration and its output cannot be read as a "
           "failure of the operator";
  }

  // NOTE. ToSlot is a bootstrap, so what reaches RMSNorm is the *bootstrapped*
  // message, not the plaintext one. The reference below is still the clear
  // RMSNorm of the original input, so the error reported includes the
  // bootstrap's own -- which is the honest number for this stage and is why
  // the threshold is looser than RmsNormTest's.
  std::vector<Ciphertext<word>> res;
  rms.Apply(res, slot_cts, wts, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const int out_level = param_->NPToLevel(res[0].GetNP());
  std::cout << "RMSNorm measured depth " << (op_level - out_level)
            << ", the plan says " << kRmsNormDepth << std::endl;
  EXPECT_EQ(op_level - out_level, kRmsNormDepth)
      << "the stage table in this file disagrees with the circuit; the plan "
         "is what sized the slack";
  EXPECT_EQ(out_level, sched.GetStCLevel())
      << "the stage did not land on StC's level, so ToCoeff will have to "
         "descend or will refuse";

  // Reference, in double.
  std::vector<double> want(static_cast<size_t>(kTokens) * kChannels);
  double want_absmax = 0.0;
  for (int t = 0; t < kTokens; t++) {
    const double inv = 1.0 / std::sqrt(MeanSquare(x, kFirstToken + t) + kEps);
    for (int c = 0; c < kChannels; c++) {
      const double v =
          x[static_cast<size_t>(kFirstToken + t) * kChannels + c] * inv * w[c];
      want[static_cast<size_t>(t) * kChannels + c] = v;
      want_absmax = std::max(want_absmax, std::abs(v));
    }
  }

  double max_abs = 0.0;
  for (int i = 0; i < num_ct; i++) {
    std::vector<Complex> got;
    DecryptAndDecode(got, res[i]);
    for (int s = 0; s < num_slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = s % kTokens;
      max_abs = std::max(
          max_abs, std::abs(got[s].real() -
                            want[static_cast<size_t>(t) * kChannels + c]));
    }
  }
  std::cout << "RMSNorm after a bootstrap: max abs err " << max_abs
            << " against |y| <= " << want_absmax << " ("
            << -std::log2(max_abs / want_absmax) << " bits)" << std::endl;
  EXPECT_GT(-std::log2(max_abs / want_absmax), 8.0);

  // Leg two: the operator's output has to be something StC will take.
  Ciphertext<word> coeff;
  sched.ToCoeff(coeff, res[0], interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(param_->NPToLevel(coeff.GetNP()), sched.GetCoeffLevel());
  std::cout << "the cycle closed: coefficients at level "
            << param_->NPToLevel(coeff.GetNP()) << ", scale "
            << coeff.GetScale() << ", with " << sched.GetLinearBudget()
            << " levels left for the product" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, CycleTestbed, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<CycleTestbed::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
