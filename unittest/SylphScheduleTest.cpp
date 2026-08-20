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

// RMSNorm at Chebyshev degree 7: one level for the square, ceil(log2(8)) = 3
// for the polynomial, two to apply the result and the weight -- six consumed --
// **plus one owed**. Apply ends with Mult(res, res, weight_pt_) and no Rescale
// (RmsNorm.cu), which is Cheddar's convention but means the output carries
// scale^2 and the caller has to spend a level settling it. A schedule that
// budgets only what the operator consumes puts the ciphertext on StC's level
// still owing a rescale, and ToCoeff then has nowhere to spend it.
//
// Degree 9 was measured landing exactly there, and the failure was silent: the
// scale-up constant rounded to zero and every coefficient came back 0 while
// every level assertion passed. Degree 7 costs one level less and fits, at the
// price of margin -- RmsNorm.h puts its 12-bit reach at a window ratio of
// 4.18, so the segment's measured spread is asserted below.
constexpr int kRmsNormDegree = 7;
constexpr double kRmsNormWindow = 4.18;
constexpr int kRmsNormDepth = 7;  // 6 consumed + 1 owed

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
  // BOOT IS A SAME-SHAPE REFERENCE HERE, NOT A PRECISION ONE.
  //
  // Boot runs its own ModRaise/CtS/EvalMod on this ciphertext with these keys,
  // so it is deterministic and identical to the one inside ToSlot -- the
  // bootstrap's error cancels to the extent the two paths share it, and a
  // number read off this comparison is not the cycle's precision. That number
  // lives in WhatTheScaleDropDoesToABootstrappedCiphertext, which measures
  // against DecodeCoeff of the plaintext and reads 11.27 bits, and in the
  // stage test, which reads 9.60 against the clear RMSNorm.
  //
  // What this test is for is that the transport does not *lose* the message:
  // a dropped constant shows up as a clean power of two in the printed scale
  // ratio, and a broken descent as a collapse. It read 5.63 bits before the
  // magnitude restoration and 10.50 after.
  EXPECT_GT(bits, 8.0)
      << "the transport lost the message. A ratio near a clean power of two "
         "in the scales printed above means a dropped constant; anything else "
         "means the descent.";
}

// ---------------------------------------------------------------------------

// THE SCALE DROP, WITH NOTHING ELSE IN THE PICTURE.
//
// A bisection against Boot appeared to localise 9.6 bits to Canonicalise. It
// was wrong -- Boot shares ToSlot's bootstrap exactly, so its error cancels
// for whichever variant shares Boot's own path and the gaps were correlation,
// not precision. That test has been deleted. What survives is the question it
// could not answer, because every number in it came out of a bootstrap: is
// the rescale at fault, or the noise it is handed? This removes the bootstrap: a freshly encrypted ciphertext,
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

// THE SAME DROP, ON BOOTSTRAP OUTPUT, AGAINST AN EXACT REFERENCE.
//
// TheScaleDropIsWhereTheBitsGo came back flat -- 18.59 to 19.01 bits across a
// 2^0 to 2^23 shrink -- so the rescale, the regraft and the small constant are
// all clean and the operation floors at about 18.7 bits. Yet the same
// arithmetic applied to HalfBoot's output cost 9.6. Those two facts are only
// compatible if something about the bootstrapped ciphertext, and not the
// operation, is what the shrink exposes.
//
// So measure it directly, one operation at a time, against a reference that
// owes nothing to another ciphertext: the plaintext handed to HalfBoot is
// decoded on the host with DecodeCoeff, and AttentionPacking::SlotOfCoeff says
// which slot each coefficient lands in -- measured on this hardware with 0 of
// 32768 disagreeing. Three readings, all in the slot domain, all against that:
//
//   1. HalfBoot's output as it lands
//   2. after a scale-preserving rescale   (what variant A did)
//   3. after Canonicalise                 (what variant B did)
//
// The fitted factor absorbs the log_message_ratio, so only the residual is
// compared. If 2 and 3 are equal, the drop is not the cause and the next
// suspect is downstream of it. If 3 is nine bits below 2, then the bootstrap
// leaves something in the ciphertext that a scale-preserving rescale carries
// harmlessly and a shrinking one does not.
TEST_P(CycleTestbed, WhatTheScaleDropDoesToABootstrappedCiphertext) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  const int degree = param_->degree_;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);
  SylphSchedule<word> sched(boot, num_slots);

  // Coefficient domain in, because that is what HalfBoot takes.
  std::vector<double> coeffs(degree);
  for (int p = 0; p < degree; p++) {
    coeffs[p] = 0.5 * std::cos(0.7 * p + 1.0);
  }
  Plaintext<word> ptxt;
  context_->encoder_.EncodeCoeff(ptxt, 0, DetermineScale(0), coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, ptxt);

  // The exact reference: what the host says those coefficients become.
  std::vector<double> round_trip;
  context_->encoder_.DecodeCoeff(round_trip, ptxt);
  std::vector<Complex> want(num_slots);
  for (int p = 0; p < degree; p++) {
    const auto pos = AttentionPacking::SlotOfCoeff(p, degree);
    if (pos.imaginary) {
      want[pos.slot] = Complex(want[pos.slot].real(), round_trip[p]);
    } else {
      want[pos.slot] = Complex(round_trip[p], want[pos.slot].imag());
    }
  }

  auto report = [&](const char *name, const Ciphertext<word> &got_ct) {
    std::vector<Complex> got;
    DecryptAndDecode(got, got_ct);
    double num = 0.0, den = 0.0, absmax = 0.0;
    for (int i = 0; i < num_slots; i++) {
      num += got[i].real() * want[i].real() + got[i].imag() * want[i].imag();
      den += want[i].real() * want[i].real() + want[i].imag() * want[i].imag();
      absmax = std::max(absmax, std::abs(want[i]));
    }
    const double fit = num / den;
    double worst = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max({worst, std::abs(fit * want[i].real() - got[i].real()),
                        std::abs(fit * want[i].imag() - got[i].imag())});
    }
    std::cout << "  " << name << ": factor " << fit << ", residual "
              << -std::log2(worst / (absmax * std::abs(fit))) << " bits"
              << std::endl;
  };

  Ciphertext<word> landed;
  sched.ToSlot(landed, ct, interface_->GetEvkMap());
  const int level = param_->NPToLevel(landed.GetNP());
  report("1 HalfBoot as it lands", landed);

  {
    Constant<word> keep;
    context_->encoder_.EncodeConstant(
        keep, level, param_->GetRescalePrimeProd(level), 1.0);
    Ciphertext<word> tmp, out;
    context_->Mult(tmp, landed, keep);
    context_->Rescale(out, tmp);
    report("2 scale-preserving rescale", out);
  }
  {
    Ciphertext<word> out;
    sched.Canonicalise(out, landed);
    report("3 Canonicalise", out);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
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
  double alpha_scaled = alpha / (beta * beta);
  double eps_scaled = kEps * beta * beta;

  // CALIBRATE FOR WHAT THE BOOTSTRAP DELIVERS, NOT FOR WHAT WENT IN.
  //
  // HalfBoot returns the message at 2^-log_message_ratio of its input --
  // measured 0.0309131, against the documented 2^-5 = 0.03125, so the constant
  // is the right one to design against. RMSNorm is scale *invariant*, so that
  // factor does not cancel: the output is unchanged, but the polynomial's
  // argument alpha_L * mean(x^2) moves by r^2, which is 1024x and straight out
  // of the approximation window. The first run of this test measured its
  // output at 0.04 bits, which is the window being blown, not the circuit.
  //
  // The correction is exact and free. Feeding x' = r*x with alpha' = alpha/r^2
  // and eps' = eps*r^2 leaves the argument bit-for-bit what it would have
  // been, and the output identical:
  //
  //     x' / sqrt(mean(x'^2) + eps') = r*x / sqrt(r^2 (mean(x^2) + eps))
  //
  // which is the same scale invariance that justified beta in the first place.
  // AND SIZE THE INPUT FOR THE BOOTSTRAP, NOT FOR THE OPERATOR.
  //
  // beta was chosen to put the *RMS* near one, which is what RmsNormTest wants
  // because CKKS precision is absolute. The bootstrap wants something else: its
  // message has to stay inside the range EvalMod approximates, which is what
  // log_message_ratio reserves. On this segment max|beta*x| is about 9.4, and
  // feeding that through HalfBoot returned an arriving residual of **1.20
  // bits** -- the input to RMSNorm was already destroyed, before the operator
  // ran, which is why calibrating the window and matching the magnitude both
  // changed nothing.
  //
  // The characterisation run that reached 11.27 bits used a cosine of
  // amplitude 0.5, so 0.5 is a magnitude this bootstrap is known to carry.
  // Size the encoded input to that and let RMSNorm's scale invariance absorb
  // it: with v = B*x arriving as r*B*x, alpha_L = alpha / (r*B)^2 and
  // eps = kEps * (r*B)^2 leave the argument and the output unchanged.
  double max_abs_x = 0.0;
  for (int t = kFirstToken; t < kFirstToken + kTokens; t++) {
    for (int c = 0; c < kChannels; c++) {
      max_abs_x = std::max(
          max_abs_x, std::abs(x[static_cast<size_t>(t) * kChannels + c]));
    }
  }
  // Canonicalise now restores the factor r that HalfBoot divides out, so the
  // operator sees the same magnitude that was encoded and the calibration is
  // in terms of B alone. The prediction, from the no-bootstrap sweep: an
  // arriving max near 0.5 with about 11 bits of relative precision gave 9.64
  // bits out, so that is what this should read, against 5.22 before.
  const double r = std::pow(2.0, -boot->GetBootParameter().GetLogMessageRatio());
  const double kBootSafeMax = 0.5;
  const double B = kBootSafeMax / max_abs_x;
  alpha_scaled = alpha / (B * B);
  eps_scaled = kEps * B * B;
  std::cout << "input scaled by B = " << B << " so max|B*x| = "
            << kBootSafeMax << " (beta would have given "
            << (beta * max_abs_x) << "); r = " << r
            << " restored by Canonicalise, alpha_L = " << alpha_scaled
            << std::endl;

  // The operator's input level is where Canonicalise leaves HalfBoot's output,
  // one below the landing level. This is the joint the whole class exists for.
  const int op_level = sched.GetSlotLevel() - 1;
  RmsNormHandler<word> rms(context_, kTokens, kChannels, alpha_scaled, op_level,
                           eps_scaled, kRmsNormWindow, kRmsNormDegree);
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
      coeffs[p] = B * x[static_cast<size_t>(t) * kChannels + c];
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
      const double want_v = B * x[static_cast<size_t>(t) * kChannels + c];
      num += got0[s].real() * want_v;
      den += want_v * want_v;
    }
    const double ratio = num / den;
    std::cout << "HalfBoot round trip: the message came back at " << ratio
              << "x of what went in (log2 " << std::log2(std::abs(ratio))
              << "), designed against r = " << r << std::endl;

    // THE RESIDUAL, which is the number I should have taken first.
    //
    // The ratio alone says the layout and the magnitude are right and says
    // nothing about whether the values are. Three hypotheses died today from
    // being argued instead of measured -- the window, the magnitude, the input
    // precision -- and each time the missing measurement was this one. A
    // no-bootstrap sweep established that RMSNorm turns an 11.3-bit input into
    // 9.64 bits out, so if this reads near 11 then the operator is being handed
    // something reasonable and the fault is downstream; if it reads far worse,
    // the bootstrap on real Llama data is worse than on the synthetic input it
    // was characterised with, and that is the answer.
    double resid = 0.0, want_absmax = 0.0;
    for (int s = 0; s < num_slots; s++) {
      const int c = s / kTokens;
      const int t = kFirstToken + (s % kTokens);
      // Canonicalise restored r, so what arrives is B*x. `ratio` is the
      // fitted factor and is now expected to be one.
      const double want_v =
          ratio * B * x[static_cast<size_t>(t) * kChannels + c];
      resid = std::max(resid, std::abs(got0[s].real() - want_v));
      want_absmax = std::max(want_absmax, std::abs(want_v));
    }
    std::cout << "  arriving residual " << -std::log2(resid / want_absmax)
              << " bits relative to a max of " << want_absmax
              << "; imaginary leakage ";
    double imag_max = 0.0;
    for (int s = 0; s < num_slots; s++) {
      imag_max = std::max(imag_max, std::abs(got0[s].imag()));
    }
    std::cout << imag_max << " (" << (imag_max / want_absmax) << " of max)"
              << std::endl;
    // The calibration above is built on the documented constant, so what has
    // to hold is that the measured factor matches it -- not that it is one.
    // The window is 6x wide and the argument moves as the square, so a few
    // percent here is harmless and a factor of two is not.
    // Canonicalise restores the ratio, so the round trip is now expected to
    // be magnitude preserving. Measured 0.990322, and the 1% shortfall is
    // the bootstrap's own -- the documented constant is exactly 2^-5 and
    // HalfBoot delivers 2^-5.014.
    EXPECT_LT(std::abs(ratio - 1.0), 0.1)
        << "the cycle is not magnitude preserving, so RMSNorm's window is "
           "calibrated against the wrong magnitude and its output cannot be "
           "read as a failure of the operator";
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
  std::cout << "RMSNorm consumed " << (op_level - out_level)
            << " levels and owes a rescale (scale " << res[0].GetScale()
            << " against a canonical " << param_->GetScale(out_level)
            << "), so the plan budgets " << kRmsNormDepth << std::endl;
  EXPECT_EQ(op_level - out_level, kRmsNormDepth - 1)
      << "the stage table in this file disagrees with the circuit; the plan "
         "is what sized the slack";
  EXPECT_EQ(out_level, sched.GetStCLevel() + 1)
      << "the stage must land one above StC so ToCoeff can settle the rescale "
         "the operator owes";

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

// ---------------------------------------------------------------------------

// IS IT THE OPERATOR OR THE BOOTSTRAP?
//
// Calibrating for the message ratio moved the stage from 0.04 bits to 0.45 --
// still garbage, so the approximation window was not the whole story. Two
// things remain, and no bootstrap is needed to separate them: RMSNorm might be
// failing at level 18 on a message whose magnitude is r = 2^-5 of what it was
// calibrated on, or it might be failing on the 11.27-bit input a bootstrap
// hands it.
//
// This encrypts the same values directly at the operator's level, so the input
// is a fresh ciphertext at roughly 30 bits instead of a bootstrapped one at 11.
// Everything else -- level, magnitude, calibration, weights, packing -- is what
// the stage uses.
//
//   clean and correct  -> the operator is fine and 11.27 bits is not enough
//   clean and wrong    -> the operator cannot work at this magnitude, and the
//                         fix is to restore it, which Canonicalise can do for
//                         free by multiplying by 1/r instead of by 1
TEST_P(CycleTestbed, RmsNormAtTheOperatorLevelWithoutABootstrap) {
  const std::string dir = DataDir();
  if (dir.empty()) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  std::vector<double> x, w;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kAllTokens * kChannels, x));
  ASSERT_TRUE(ReadF32(dir + "/attn_norm.f32", kChannels, w));

  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  const int op_level = boot->GetBootParameter().GetEvalModEndLevel() - 1;
  const int channels_per_ct = num_slots / kTokens;

  double log_sum = 0.0;
  for (int t = kFirstToken; t < kFirstToken + kTokens; t++) {
    log_sum += std::log(MeanSquare(x, t));
  }
  const double alpha = 1.0 / std::exp(log_sum / kTokens);
  const double beta = std::sqrt(alpha);

  // HOW MUCH INPUT PRECISION DOES RMSNORM NEED?
  //
  // Magnitude turned out not to matter -- 13.51 bits at m = 1 and 13.48 at
  // m = r -- so the only difference left between this and the stage is that
  // the stage's input comes out of a bootstrap at 11.27 bits. Rather than
  // assume that is the cause, inject noise of a known size into a clean input
  // and find the threshold. `noise_bits` is the error added relative to the
  // largest value in the ciphertext, which is how the bootstrap's own residual
  // was measured, so the numbers are comparable.
  //
  // If 11.3 reproduces the stage's 0.45 bits, the story is settled and the
  // question becomes where to find the missing precision. If it stays clean,
  // then input precision is not the cause and something specific to
  // bootstrapped ciphertexts is.
  const double r_ratio =
      std::pow(2.0, -boot->GetBootParameter().GetLogMessageRatio());
  struct Case {
    double m;
    double noise_bits;
  };
  for (const Case &cs : std::vector<Case>{{1.0, 0.0},
                                          {r_ratio, 0.0},
                                          {r_ratio, 16.0},
                                          {r_ratio, 13.0},
                                          {r_ratio, 11.3}}) {
    const double m = cs.m;
    const double a_used = 1.0 / (m * m);
    const double eps_used = kEps * beta * beta * m * m;
    RmsNormHandler<word> rms(context_, kTokens, kChannels, a_used, op_level,
                             eps_used, kRmsNormWindow, kRmsNormDegree);
    const int num_ct = rms.GetNumCiphertexts();
    for (int d : rms.GetRotationDistances()) {
      interface_->PrepareRotationKey(d, op_level);
    }

    const double root = std::sqrt(a_used);
    // The largest packed value, which is what noise_bits is measured against.
    double packed_max = 0.0;
    for (int t = kFirstToken; t < kFirstToken + kTokens; t++) {
      for (int c = 0; c < kChannels; c++) {
        packed_max = std::max(
            packed_max,
            std::abs(m * beta * x[static_cast<size_t>(t) * kChannels + c]));
      }
    }
    const double noise =
        cs.noise_bits > 0.0 ? packed_max * std::pow(2.0, -cs.noise_bits) : 0.0;
    unsigned seed = 12345u;
    auto next = [&seed]() {
      seed = seed * 1103515245u + 12345u;
      return (static_cast<double>((seed >> 16) & 0x7fff) / 16383.5) - 1.0;
    };

    std::vector<Ciphertext<word>> cts(num_ct);
    std::vector<std::vector<Complex>> wts(num_ct);
    for (int i = 0; i < num_ct; i++) {
      std::vector<Complex> msg(num_slots);
      wts[i].assign(num_slots, Complex(0.0, 0.0));
      for (int s = 0; s < num_slots; s++) {
        const int c = i * channels_per_ct + s / kTokens;
        const int t = kFirstToken + (s % kTokens);
        msg[s] = Complex(m * beta * x[static_cast<size_t>(t) * kChannels + c] +
                             noise * next(),
                         0.0);
        wts[i][s] = Complex(w[c] * root, 0.0);
      }
      EncodeAndEncrypt(cts[i], msg, op_level);
    }

    std::vector<Ciphertext<word>> res;
    rms.Apply(res, cts, wts, interface_->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    double worst = 0.0, absmax = 0.0;
    for (int i = 0; i < num_ct; i++) {
      std::vector<Complex> got;
      DecryptAndDecode(got, res[i]);
      for (int s = 0; s < num_slots; s++) {
        const int c = i * channels_per_ct + s / kTokens;
        const int t = s % kTokens;
        const double inv =
            1.0 / std::sqrt(MeanSquare(x, kFirstToken + t) + kEps);
        const double want =
            x[static_cast<size_t>(kFirstToken + t) * kChannels + c] * inv * w[c];
        worst = std::max(worst, std::abs(got[s].real() - want));
        absmax = std::max(absmax, std::abs(want));
      }
    }
    std::cout << "  magnitude " << m << ", input noise "
              << (cs.noise_bits > 0.0 ? std::to_string(cs.noise_bits) + " bits"
                                      : std::string("none"))
              << ": max abs err " << worst << " against " << absmax << " ("
              << -std::log2(worst / absmax) << " bits out)" << std::endl;
  }
}

// ---------------------------------------------------------------------------

// THE LOOP CLOSES, AND CLOSES AGAIN.
//
// A decoder block is four turns of figure 2's cycle and the model is 32 of
// those, so the question that matters is not whether one leg works but whether
// the state coming out of a turn is a legal input to the next one. Every level
// and every scale has to arrive back where it started, or the second turn is a
// different computation from the first and nothing composes.
//
// This runs two full turns with a real operator in each -- coefficients at
// level 0 in, ToSlot, Canonicalise, RMSNorm, ToCoeff, descend, and round again
// -- and checks the state after each. RMSNorm is idempotent on its own output
// up to the weights, which is what makes a two-turn check meaningful without a
// product in between: turn two normalises what turn one produced, and the host
// can say exactly what that is.
//
// The linear leg is deliberately absent, and its absence is the point of the
// descent that stands in for it: [SYLPH] puts the product between ToCoeff and
// ToSlot, and what this establishes is that the slot the product would occupy
// is reachable and that the loop closes around it. Attaching the product needs
// the degree-4096 ring, which needs the ring-switch parameter pair that
// bootparam_35 does not have -- Doing.md 1.5y.
TEST_P(CycleTestbed, TheLoopClosesTwice) {
  const std::string dir = DataDir();
  if (dir.empty()) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  std::vector<double> x, w;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kAllTokens * kChannels, x));
  ASSERT_TRUE(ReadF32(dir + "/attn_norm.f32", kChannels, w));

  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  const int degree = param_->degree_;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);
  SylphSchedule<word> sched(boot, num_slots);

  const int op_level = sched.GetSlotLevel() - 1;
  const int channels_per_ct = num_slots / kTokens;

  // EVERY TURN'S OUTPUT HAS TO FIT THE NEXT TURN'S BOOTSTRAP.
  //
  // The single-stage test established that ModRaise cannot carry a message
  // much past 0.5: at max 9.4 the arriving residual was 1.20 bits, and at 0.5
  // it was 10.95. A stage's output is the next stage's input, so its magnitude
  // is a scheduling constraint rather than a detail of the operator -- and it
  // bites immediately, because RMSNorm's output reaches 6.43 while its input
  // was sized to 0.5.
  //
  // Sizing it costs nothing. RMSNorm already multiplies by a per-channel
  // plaintext, so a scalar folds into that weight for free, exactly as
  // sqrt(alpha_L) already does. What the host has to do is track it, which is
  // what s1 and s2 are below.
  auto host_rmsnorm = [&](const std::vector<double> &in, int first_token,
                          std::vector<double> &out) {
    out.assign(static_cast<size_t>(kTokens) * kChannels, 0.0);
    for (int t = 0; t < kTokens; t++) {
      double sq = 0.0;
      for (int c = 0; c < kChannels; c++) {
        const double v =
            in[static_cast<size_t>(first_token + t) * kChannels + c];
        sq += v * v;
      }
      const double inv = 1.0 / std::sqrt(sq / kChannels + kEps);
      for (int c = 0; c < kChannels; c++) {
        out[static_cast<size_t>(t) * kChannels + c] =
            in[static_cast<size_t>(first_token + t) * kChannels + c] * inv *
            w[c];
      }
    }
  };
  auto max_abs = [](const std::vector<double> &v, size_t off, size_t n) {
    double m = 0.0;
    for (size_t k = 0; k < n; k++) m = std::max(m, std::abs(v[off + k]));
    return m;
  };
  auto geomean_ms = [&](const std::vector<double> &v, int first_token) {
    double ls = 0.0;
    for (int t = 0; t < kTokens; t++) {
      double sq = 0.0;
      for (int c = 0; c < kChannels; c++) {
        const double u =
            v[static_cast<size_t>(first_token + t) * kChannels + c];
        sq += u * u;
      }
      ls += std::log(sq / kChannels);
    }
    return std::exp(ls / kTokens);
  };

  const double kBootSafeMax = 0.5;

  // Turn one, on the real residual stream.
  std::vector<double> y1;
  host_rmsnorm(x, kFirstToken, y1);
  const double b1 =
      kBootSafeMax /
      max_abs(x, static_cast<size_t>(kFirstToken) * kChannels,
              static_cast<size_t>(kTokens) * kChannels);
  const double s1 = kBootSafeMax / max_abs(y1, 0, y1.size());
  const double a1 = 1.0 / geomean_ms(x, kFirstToken) / (b1 * b1);

  // Turn two, on turn one's scaled output.
  std::vector<double> y1s(y1.size());
  for (size_t k = 0; k < y1.size(); k++) y1s[k] = s1 * y1[k];
  std::vector<double> y2;
  host_rmsnorm(y1s, 0, y2);
  const double s2 = kBootSafeMax / max_abs(y2, 0, y2.size());
  const double a2 = 1.0 / geomean_ms(y1s, 0);

  std::cout << "input sizing b1 " << b1 << ", output sizing s1 " << s1
            << " and s2 " << s2 << "; alpha_L " << a1 << " then " << a2
            << std::endl;

  RmsNormHandler<word> rms1(context_, kTokens, kChannels, a1, op_level,
                            kEps * b1 * b1, kRmsNormWindow, kRmsNormDegree);
  RmsNormHandler<word> rms2(context_, kTokens, kChannels, a2, op_level,
                            kEps * s1 * s1, kRmsNormWindow, kRmsNormDegree);
  const int num_ct = rms1.GetNumCiphertexts();
  for (int d : rms1.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, op_level);
  }

  // Turn one's input, coefficient-encoded at level 0 -- where the previous
  // turn's product would have left it.
  std::vector<Ciphertext<word>> state(num_ct);
  for (int i = 0; i < num_ct; i++) {
    std::vector<double> coeffs(degree, 0.0);
    for (int s = 0; s < num_slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = kFirstToken + (s % kTokens);
      coeffs[AttentionPacking::CoeffOfSlot({s, false}, degree)] =
          b1 * x[static_cast<size_t>(t) * kChannels + c];
    }
    Plaintext<word> ptxt;
    context_->encoder_.EncodeCoeff(ptxt, 0, DetermineScale(0), coeffs);
    interface_->Encrypt(state[i], ptxt);
  }

  for (int turn = 1; turn <= 2; turn++) {
    RmsNormHandler<word> &rms = (turn == 1) ? rms1 : rms2;
    // sqrt(alpha_L) is what the handler expects folded in; the output sizing
    // rides along on the same plaintext.
    const double root = std::sqrt(turn == 1 ? a1 : a2);
    const double out_scale = (turn == 1) ? s1 : s2;
    std::vector<std::vector<Complex>> wts(
        num_ct, std::vector<Complex>(num_slots, Complex(0.0, 0.0)));
    for (int i = 0; i < num_ct; i++) {
      for (int s = 0; s < num_slots; s++) {
        const int c = i * channels_per_ct + s / kTokens;
        wts[i][s] = Complex(w[c] * root * out_scale, 0.0);
      }
    }

    std::vector<Ciphertext<word>> slots(num_ct);
    for (int i = 0; i < num_ct; i++) {
      Ciphertext<word> landed;
      const double drift =
          sched.ToSlot(landed, state[i], interface_->GetEvkMap());
      if (i == 0) {
        std::cout << "turn " << turn << ": ToSlot landed at level "
                  << param_->NPToLevel(landed.GetNP()) << ", drift " << drift
                  << std::endl;
        EXPECT_EQ(param_->NPToLevel(landed.GetNP()), sched.GetSlotLevel());
      }
      sched.Canonicalise(slots[i], landed);
    }

    std::vector<Ciphertext<word>> normed;
    rms.Apply(normed, slots, wts, interface_->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(param_->NPToLevel(normed[0].GetNP()), sched.GetStCLevel() + 1)
        << "turn " << turn
        << " must land one above StC, so ToCoeff can settle the rescale the "
           "operator owes";

    // Read the turn's output on both sides of the descent. Nothing else in
    // this file goes below level 8, and on bootparam_35 level 0 is [0, 2] --
    // no main primes and two terminal -- so reaching it from level 8's [11, 0]
    // drops eleven and adds two, which is a regraft nothing here has
    // exercised. Measuring both sides says whether a zero output came out of
    // the cycle or out of the descent standing in for the product.
    // ToCoeff builds its scale-up constant as
    // EncodeConstant(1.0 * r, stc_level, GetStCInputScale() / src.GetScale()),
    // and EncodeConstant rounds `number * scale` to an integer -- so if that
    // product falls below 1/2 the constant is zero and the ciphertext is
    // annihilated, which is exactly what an all-zero coefficient vector looks
    // like. These are the three numbers that decide it.
    {
      const double got_scale = normed[0].GetScale();
      const double up = boot->GetStCInputScale() / got_scale;
      const double rr =
          std::pow(2.0, -boot->GetBootParameter().GetLogMessageRatio());
      std::cout << "    turn " << turn << " RMSNorm out scale " << got_scale
                << ", canonical at " << sched.GetStCLevel() << " is "
                << param_->GetScale(sched.GetStCLevel())
                << ", StC wants " << boot->GetStCInputScale()
                << " -> up_factor " << up << ", constant integer " << (rr * up)
                << std::endl;
    }
    const std::vector<double> &ref = (turn == 1) ? y1s : y2;
    const double ref_scale = (turn == 1) ? 1.0 : s2;
    for (int i = 0; i < num_ct; i++) {
      Ciphertext<word> coeff;
      sched.ToCoeff(coeff, normed[i], interface_->GetEvkMap());
      if (i == 0) {
        auto read = [&](const Ciphertext<word> &c, const char *where) {
          Plaintext<word> pt;
          interface_->Decrypt(pt, c);
          std::vector<double> cf;
          context_->encoder_.DecodeCoeff(cf, pt);
          double e = 0.0, m = 0.0, num = 0.0, den = 0.0;
          for (int sl = 0; sl < num_slots; sl++) {
            const int ch = sl / kTokens;
            const int tk = sl % kTokens;
            const double want =
                ref_scale * ref[static_cast<size_t>(tk) * kChannels + ch];
            const double got =
                cf[AttentionPacking::CoeffOfSlot({sl, false}, degree)];
            e = std::max(e, std::abs(got - want));
            m = std::max(m, std::abs(want));
            num += got * want;
            den += want * want;
          }
          // Where did it go? If the whole coefficient vector is empty the
          // ciphertext is, and if it is not then the data is somewhere this
          // index map does not look.
          double all = 0.0;
          for (int q = 0; q < degree; q++) all = std::max(all, std::abs(cf[q]));
          std::cout << "    turn " << turn << " " << where << ": "
                    << -std::log2(e / m) << " bits (want max " << m
                    << ", fit " << (num / den) << ", largest coefficient "
                    << all << ")" << std::endl;
        };
        read(coeff, "after ToCoeff, level 8");
        Ciphertext<word> low;
        context_->LevelDown(low, coeff, 0);
        read(low, "after LevelDown to 0");
      }
      // Stand-in for the linear leg: descend to level 0, which is where the
      // product would have left the ciphertext for the next ToSlot.
      context_->LevelDown(state[i], coeff, 0);
    }
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::cout << "turn " << turn << ": closed at level "
              << param_->NPToLevel(state[0].GetNP()) << ", scale "
              << state[0].GetScale() << std::endl;
    EXPECT_EQ(param_->NPToLevel(state[0].GetNP()), 0)
        << "turn " << turn << " did not return to the level it started at";
  }

  // The state is coefficient-encoded at level 0, carrying s2 * y2.
  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_ct; i++) {
    Plaintext<word> ptxt;
    interface_->Decrypt(ptxt, state[i]);
    std::vector<double> coeffs;
    context_->encoder_.DecodeCoeff(coeffs, ptxt);
    for (int s = 0; s < num_slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = s % kTokens;
      const double want = y2[static_cast<size_t>(t) * kChannels + c];
      const double got =
          coeffs[AttentionPacking::CoeffOfSlot({s, false}, degree)] / s2;
      worst = std::max(worst, std::abs(got - want));
      absmax = std::max(absmax, std::abs(want));
    }
  }
  std::cout << "two full turns vs the host: max abs err " << worst
            << " against |y2| <= " << absmax << " ("
            << -std::log2(worst / absmax) << " bits)" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 6.0);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, CycleTestbed, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<CycleTestbed::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
