// The robust-parameter acceptance test: does a preset HONOR the
// scale/landing contract, measured on the device against the library's own
// nominal numbers?
//
// WHY. The recurring parameter problems all trace to one root: the
// scale/landing contract is neither written down nor checked. EvalMod's
// scale recursion leaves a DIFFERENT landing scale per ladder (2^51.9 on
// land18c4e10 against the 2^58 fixed point, Doing.md 7.36), the nominal
// StCInputScale missed the measured tower-boot scale by 0.28% (7.37's
// re-encode hack), EvalPoly past used degree 7 landed 2^400 garbage on the
// decode path (7.45), and ladder tests were polluted by scale luck until
// their entry was pinned. `reference/scripts/param_audit.py` predicts the
// NOMINAL side of all of this on the host; this test measures the ACTUAL
// side on the device and pins the two together, per preset:
//
//   (1) TheEvalModScaleRecursionMatchesTheHost -- the audit's model of
//       EvalMod's recursion equals the library's (GetEvalModStartScale /
//       GetStCInputScale), so the host deviation map stays honest.
//   (2) TheBootLandsAtItsNominalScale -- full Boot's carried factor over
//       1.0 in ppm (the measured-vs-nominal landing-scale miss; 7.37's
//       0.28% class), plus the HalfBoot -> SlotToCoeff cycle (HalfBoot's
//       own carried constant is measured, not derived -- HalfBootTest's
//       lesson) and the K-edge tail count.
//   (3) TrimmedPolynomialsLandAtEveryDegree (run LAST; a library Fail()
//       aborts the process) -- EvalPoly through the used-degree probe +
//       level-trimming logic (CiDecodeLayer's EvalAtDepth pattern) at
//       every degree 2..15, against a host Chebyshev evaluation. The
//       deg > 7 bug is watched permanently.
//   (4) TheCanonicalDescentIsExactAndTheBareDescentDrifts -- CanonicalTo's
//       walk lands the canonical scale at every level; LevelDown keeps the
//       message but leaves the declared scale off canonical by exactly the
//       ladder's per-level drift (printed).
//   (5) TheRideProbe -- HalfBoot residual against message amplitude; the
//       cubic EvalMod tail (-0.00258 m^3 on ci16_35) fitted from the
//       measurements, so a preset's usable ride is a printed number, not
//       folklore.
//
// One preset per process (the project's minimal-regression habit):
//   CHEDDAR_ROBUST_PARAM=ci16_35_land17c3e10.json ./param_robust_test
// Default preset: ci16_35.json. CHEDDAR_ROBUST_STRICT=1 tightens the gates
// to what gen_landing v3's output must satisfy.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "common/CommonUtils.h"
#include "extension/EvalPoly.h"

namespace {

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::Log2Ceil;
using cheddar::Plaintext;

const char *RobustParam() {
  const char *e = std::getenv("CHEDDAR_ROBUST_PARAM");
  return (e && e[0]) ? e : "ci16_35.json";
}
bool Strict() {
  const char *e = std::getenv("CHEDDAR_ROBUST_STRICT");
  return e != nullptr && e[0] == '1';
}

std::vector<Complex> RandomReal(int num_slots, double amp, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> d(-amp, amp);
  std::vector<Complex> m(num_slots);
  for (auto &c : m) c = Complex(d(gen), 0.0);
  return m;
}

void EncryptAt(const Ring &r, Ciphertext<word> &ct,
               const std::vector<Complex> &msg, int level) {
  Plaintext<word> pt;
  r.context->encoder_.Encode(pt, level, r.param->GetScale(level), msg, 0);
  r.ui->Encrypt(ct, pt);
  ct.SetNumSlots(r.param->MaxNumSlots());
}

std::vector<Complex> Decrypt(const Ring &r, const Ciphertext<word> &ct) {
  Plaintext<word> pt;
  r.ui->Decrypt(pt, ct);
  std::vector<Complex> out;
  r.context->encoder_.Decode(out, pt);
  return out;
}

// The scalar the output carries, fitted, and the residual against it.
struct Fit {
  double carried, residual;
};
Fit FitResidual(const std::vector<Complex> &input,
                const std::vector<Complex> &got) {
  double num = 0.0, den = 0.0;
  const size_t n = std::min(input.size(), got.size());
  for (size_t i = 0; i < n; i++) {
    num += got[i].real() * input[i].real();
    den += input[i].real() * input[i].real();
  }
  const double c = (den > 0) ? num / den : 0.0;
  double res = 0.0;
  for (size_t i = 0; i < n; i++)
    res = std::max(res, std::abs(got[i] - c * input[i]));
  return {c, res};
}

std::shared_ptr<BootContext<word>> PrepareBoot(const Ring &r) {
  auto b = std::dynamic_pointer_cast<BootContext<word>>(r.context);
  EXPECT_NE(b, nullptr) << "ring is not a BootContext -- is boot:true set?";
  const int num_slots = r.param->MaxNumSlots();
  b->PrepareEvalMod();
  b->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  b->AddRequiredRotations(req, num_slots, /*min_ks=*/false);
  r.ui->PrepareRotationKey(req);
  return b;
}

}  // namespace

// (1) The host auditor's model of EvalMod's scale recursion IS the
// library's. `param_audit.py` predicts every ladder's landing scale from
// the preset JSON alone; this pins that prediction to
// GetEvalModStartScale/GetStCInputScale, so a change to EvalMod.cpp that
// moves the recursion flags the auditor as stale instead of silently
// diverging from it.
TEST(ParamRobust, TheEvalModScaleRecursionMatchesTheHost) {
  Ring r(RobustParam());
  auto b = std::dynamic_pointer_cast<BootContext<word>>(r.context);
  ASSERT_NE(b, nullptr);
  b->PrepareEvalMod();

  const auto &bp = b->GetBootParameter();
  const int start_level = bp.GetEvalModStartLevel();
  const int nem = bp.GetNumEvalModLevels();

  // EvalMod.cpp: start = 2^round(log2 prod(start_level)), then
  // s <- s^2 / prod(level) down the band (the polynomial's levels and the
  // double angles walk the same descending sequence).
  const int first_log_scale = static_cast<int>(
      std::log2(r.param->GetRescalePrimeProd(start_level)) + 0.5);
  double s = static_cast<double>(UINT64_C(1) << first_log_scale);
  const double start_expect = s;
  for (int i = 0; i < nem; i++) {
    s = s * s / r.param->GetRescalePrimeProd(start_level - i);
  }
  std::cout << "[recursion] " << RobustParam() << ": EvalMod " << start_level
            << " -> " << (start_level - nem) << ", start 2^"
            << first_log_scale << ", predicted end 2^" << std::log2(s)
            << " (wander " << (std::log2(s) - first_log_scale)
            << " bits); library start 2^" << std::log2(b->GetEvalModStartScale())
            << ", end 2^" << std::log2(b->GetStCInputScale()) << std::endl;
  EXPECT_DOUBLE_EQ(b->GetEvalModStartScale(), start_expect)
      << "the auditor's start-scale rule no longer matches EvalMod.cpp";
  EXPECT_NEAR(b->GetStCInputScale() / s, 1.0, 1e-12)
      << "the auditor's recursion no longer matches EvalMod.cpp";
}

// (2) The landing scale, measured. HalfBoot's output declares
// eval_mod_->end_scale_ and its message rides the DERIVED crossing
// constant; full Boot declares GetScale(EndLevel) * leveldown_drift and is
// message preserving. Any deviation of the fitted carried factor from the
// derived one is the measured-vs-nominal landing-scale miss -- the 0.28%
// that forced 7.37's measured-scale re-encode. Reported in ppm so the map
// gets a number per preset.
TEST(ParamRobust, TheBootLandsAtItsNominalScale) {
  Ring r(RobustParam());
  auto b = PrepareBoot(r);
  const int num_slots = r.param->MaxNumSlots();
  const auto &bp = b->GetBootParameter();

  const auto msg = RandomReal(num_slots, 0.5, 0xA11CE);
  Ciphertext<word> ct;
  EncryptAt(r, ct, msg, 0);

  // The half route, measured in the shape the pipeline actually runs
  // (HalfBootTest's lesson): HalfBoot leaves the input's COEFFICIENTS
  // bit-reversed in the slots -- not comparable to the message directly --
  // so the honest check is the HalfBoot -> SlotToCoeff cycle, whose
  // permutations cancel and whose composite constant is measured, not
  // derived (BootContext.cpp says so at HalfBoot's SetScale).
  Ciphertext<word> half;
  b->HalfBoot(half, ct, r.ui->GetEvkMap());
  ASSERT_EQ(r.param->NPToLevel(half.GetNP()), bp.GetEvalModEndLevel());
  Ciphertext<word> cyc;
  b->SlotToCoeff(cyc, num_slots, half, r.ui->GetEvkMap());
  const Fit h = FitResidual(msg, Decrypt(r, cyc));
  std::cout << "[landing] HalfBoot+StC cycle carried " << h.carried << " = 2^"
            << std::log2(std::abs(h.carried)) << " (derived crossing ratio 2^"
            << std::log2(b->GetMessageRatio())
            << "); relative residual 2^"
            << std::log2(h.residual / std::abs(h.carried)) << std::endl;
  EXPECT_LT(h.residual / std::abs(h.carried), 1e-2)
      << "the message did not survive the HalfBoot -> SlotToCoeff cycle";

  // The K-edge tail: slots whose ModRaise wrap-around fell past EvalMod's
  // range land far out, and one such slot is a rank-one error the layer's
  // projections spread (the K = 16 lesson, Doing.md 3.9/CLAUDE.md).
  {
    const auto got = Decrypt(r, cyc);
    int far = 0;
    for (int i = 0; i < num_slots; i++) {
      if (std::abs(got[i].real() - h.carried * msg[i].real()) >
          1e-3 * std::abs(h.carried)) {
        far++;
      }
    }
    std::cout << "[landing] slots past 1e-3 relative residual: " << far
              << " of " << num_slots << std::endl;
    if (Strict()) {
      EXPECT_EQ(far, 0) << "K-edge tail: slots past EvalMod's range";
    }
  }

  if (bp.GetEndLevel() >= 1) {
    Ciphertext<word> full;
    EncryptAt(r, ct, msg, 0);
    b->Boot(full, ct, r.ui->GetEvkMap());
    ASSERT_EQ(r.param->NPToLevel(full.GetNP()), bp.GetEndLevel());
    const Fit f = FitResidual(msg, Decrypt(r, full));
    const double full_miss = f.carried - 1.0;
    std::cout << "[landing] Boot carried " << f.carried << ": miss "
              << full_miss * 1e6 << " ppm; residual " << f.residual << " = 2^"
              << std::log2(f.residual) << "; declared landing scale 2^"
              << std::log2(full.GetScale()) << " vs canonical 2^"
              << std::log2(r.param->GetScale(bp.GetEndLevel())) << std::endl;
    EXPECT_LT(f.residual, 0.01) << "Boot did not preserve the message";
    EXPECT_LT(std::abs(full_miss), Strict() ? 1e-3 : 2e-2)
        << "full Boot is not message preserving at the nominal scale";
  } else {
    std::cout << "[landing] full Boot skipped (lands at " << bp.GetEndLevel()
              << ")" << std::endl;
  }
}

// (4) The canonical descent (the decode layer's CanonicalTo: a 1.0
// plaintext multiply and a rescale per level) lands the canonical scale
// AND the message at every level; the bare LevelDown keeps the message but
// leaves the declared scale where it was, off canonical by the ladder's
// accumulated per-level drift -- printed, because that offset SQUARES
// through any polynomial walk that follows it (the decode lesson).
TEST(ParamRobust, TheCanonicalDescentIsExactAndTheBareDescentDrifts) {
  Ring r(RobustParam());
  auto ctx = r.context;
  const int num_slots = r.param->MaxNumSlots();
  const int dec = r.param->default_encryption_level_;
  const int lo = std::max(0, dec - 4);
  ASSERT_GT(dec, lo) << "no levels to descend";

  const auto msg = RandomReal(num_slots, 0.5, 0xDE5C);
  const std::vector<Complex> ones(num_slots, Complex(1.0, 0.0));

  // CanonicalTo's walk.
  Ciphertext<word> ct;
  EncryptAt(r, ct, msg, dec);
  for (int l = dec; l > lo; l--) {
    Plaintext<word> one;
    ctx->encoder_.Encode(one, l, r.param->GetScale(l), ones, 0);
    Ciphertext<word> tmp;
    ctx->Mult(tmp, ct, one);
    ctx->Rescale(ct, tmp);
    const double declared = ct.GetScale();
    const double canonical = r.param->GetScale(l - 1);
    EXPECT_NEAR(declared / canonical, 1.0, 1e-9)
        << "the canonical descent left level " << (l - 1) << " at 2^"
        << std::log2(declared) << " instead of 2^" << std::log2(canonical);
  }
  {
    const Fit f = FitResidual(msg, Decrypt(r, ct));
    std::cout << "[descent] CanonicalTo " << dec << " -> " << lo
              << ": carried " << f.carried << ", residual 2^"
              << std::log2(f.residual) << std::endl;
    EXPECT_NEAR(f.carried, 1.0, 1e-6);
    EXPECT_LT(f.residual, 1e-3);
  }

  // The bare LevelDown: message intact, declared scale carried unchanged.
  Ciphertext<word> ld;
  EncryptAt(r, ct, msg, dec);
  ctx->LevelDown(ld, ct, lo);
  const double off_bits =
      std::log2(ld.GetScale() / r.param->GetScale(lo));
  const Fit f = FitResidual(msg, Decrypt(r, ld));
  std::cout << "[descent] LevelDown " << dec << " -> " << lo << ": declared 2^"
            << std::log2(ld.GetScale()) << " vs canonical 2^"
            << std::log2(r.param->GetScale(lo)) << " (off " << off_bits
            << " bits); carried " << f.carried << ", residual 2^"
            << std::log2(f.residual) << std::endl;
  EXPECT_NEAR(f.carried, 1.0, 1e-6)
      << "LevelDown's declared scale stopped tracking the message";
  EXPECT_LT(f.residual, 1e-3);
}

// (5) The ride probe: HalfBoot residual against message amplitude. The
// EvalMod tail is a CUBIC in the ride (-0.00258 m^3 on ci16_35, CLAUDE.md);
// fitting it from measurements makes a preset's usable ride a printed
// number. Sanity-gated only -- the fit is the deliverable.
TEST(ParamRobust, TheRideProbe) {
  Ring r(RobustParam());
  auto b = PrepareBoot(r);
  const int num_slots = r.param->MaxNumSlots();

  const double amps[] = {0.05, 0.1, 0.2, 0.3, 0.4};
  std::vector<double> rms_at;
  for (double amp : amps) {
    const auto msg = RandomReal(num_slots, amp, 0x71DE);
    Ciphertext<word> ct;
    EncryptAt(r, ct, msg, 0);
    // The HalfBoot -> SlotToCoeff cycle (as in the landing test): the
    // direct HalfBoot output holds bit-reversed coefficients, whose
    // residual measures the encoding, not EvalMod's tail.
    Ciphertext<word> half, res;
    b->HalfBoot(half, ct, r.ui->GetEvkMap());
    b->SlotToCoeff(res, num_slots, half, r.ui->GetEvkMap());
    const auto got = Decrypt(r, res);
    const Fit f = FitResidual(msg, got);
    double rms = 0.0;
    for (int i = 0; i < num_slots; i++) {
      const double e = got[i].real() - f.carried * msg[i].real();
      rms += e * e;
    }
    rms = std::sqrt(rms / num_slots) / std::abs(f.carried);
    rms_at.push_back(rms);
    std::cout << "[ride] amp " << amp << ": rms residual 2^"
              << std::log2(rms) << " (message units)" << std::endl;
  }
  // Fit rms(m) ~ a * m + c * m^3 from the smallest and largest amplitude.
  const double m0 = amps[0], m1 = amps[4];
  const double c3 = (rms_at[4] / m1 - rms_at[0] / m0) / (m1 * m1 - m0 * m0);
  std::cout << "[ride] fitted cubic coefficient ~" << c3
            << " (ci16_35's fitted EvalMod tail was -0.00258 m^3 on the "
            << "mean; this is the rms view)" << std::endl;
  EXPECT_LT(rms_at[2], 1e-3)
      << "the residual at ride 0.2 is out of family for a working preset";
}

// (3, run LAST: a library Fail() aborts the process) EvalPoly at every
// used degree 2..15 (EvalPoly rejects degree < 2) through the trimming logic --
// CiDecodeLayer::EvalAtDepth's exact pattern: probe the used degree,
// trim the landing to lv - Log2Ceil(used + 1), compile against the
// canonical target scale there, evaluate. Doing.md 7.45 measured 2^400
// garbage past used degree 7 on the decode path; this is the permanent
// minimal watch. A pure Chebyshev T_d input pins the used degree to
// exactly d (no silent 1e-9 trimming).
TEST(ParamRobust, TrimmedPolynomialsLandAtEveryDegree) {
  Ring r(RobustParam());
  auto ctx = r.context;
  const auto &mult_key = r.ui->GetEvkMap().GetMultiplicationKey();
  const int num_slots = r.param->MaxNumSlots();

  const int dec = r.param->default_encryption_level_;
  const int lv = std::min(dec, 6);
  const int max_fit_degree = (1 << std::max(lv, 0)) - 1;
  ASSERT_GE(lv, 1) << "no levels to evaluate a polynomial";

  const auto msg = RandomReal(num_slots, 0.95, 0x9017);

  for (int d = 2; d <= std::min(15, max_fit_degree); d++) {
    std::vector<double> coeffs(static_cast<size_t>(d) + 1, 0.0);
    coeffs[d] = 0.5;  // 0.5 * T_d, well inside [-1, 1] on the domain

    Ciphertext<word> in;
    EncryptAt(r, in, msg, lv);
    const double in_scale = in.GetScale();
    const int used = cheddar::EvalPoly<word>(coeffs, lv, in_scale, in_scale,
                                             /*chebyshev=*/true)
                         .GetPolyDegree();
    ASSERT_EQ(used, d) << "the probe trimmed a pure T_" << d;
    const int lr = lv - Log2Ceil(used + 1);
    cheddar::EvalPoly<word> poly(coeffs, lv, in_scale,
                                 r.param->GetScale(lr), /*chebyshev=*/true);
    poly.Compile(ctx);
    Ciphertext<word> out;
    poly.Evaluate(ctx, out, in, mult_key);

    const auto got = Decrypt(r, out);
    double err = 0.0, absmax = 0.0;
    for (int i = 0; i < num_slots; i++) {
      const double x = msg[i].real();
      const double want = 0.5 * std::cos(d * std::acos(std::min(
                                    1.0, std::max(-1.0, x))));
      err = std::max(err, std::abs(got[i].real() - want));
      absmax = std::max(absmax, std::abs(want));
    }
    std::cout << "[poly] degree " << d << ": lands level "
              << r.param->NPToLevel(out.GetNP()) << " (asked " << lr
              << "), declared scale 2^" << std::log2(out.GetScale())
              << ", max err " << err << " = 2^" << std::log2(err)
              << std::endl;
    EXPECT_EQ(r.param->NPToLevel(out.GetNP()), lr)
        << "degree " << d << " did not land where the trim asked";
    EXPECT_LT(err / std::max(absmax, 0.5), 1e-2)
        << "degree " << d
        << " did not evaluate (the used-degree > 7 class, Doing.md 7.45)";
  }
}
