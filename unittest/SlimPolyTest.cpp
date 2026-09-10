// [SYLPH] section 3.4 and Algorithm 1: slim polynomial evaluation.
//
// The `SlimMath.*` cases are pure host arithmetic and need NO GPU --
// `slim_poly_test --gtest_filter='SlimMath.*'` runs anywhere, which is the
// point: the decomposition is where this algorithm can be wrong, and it can be
// checked before any card is asked for.
//
// What the paper claims and what is checked here:
//
//   lemma 1     P = U^2 + V^2 - m with deg U, deg V <= deg P / 2
//   eq. (2)     the combination, with the 1/sqrt 2 that keeps both halves at
//               full degree -- without it V loses its leading term and the
//               recursion stops
//   eq. (3)     the tree, children of `i` at `i` and `i + 2^(l-1)`
//   Algorithm 1 evaluate, then j squarings each followed by a rotation by
//               `2^t 2^(l-1)` and a subtraction of `m^(l-1)`
//   theorem 1   k + 1 levels, O(2^((k-j)/2)) + j multiplications, j rotations
//   appendix D  the search: a decomposition whose U and V are small on the
//               interval AND whose children need a small m
//
// The last of those is not decoration. Without the lookahead half of appendix
// D's objective, `m` on the softmax's own inverse-square-root window reaches
// 8.8e5 at degree 64 -- 16 bits of dynamic range spent on a cancellation --
// and with it `m` stays near 1 at every depth. That measurement is the reason
// `DecomposeOptions::lookahead` defaults to true, and the
// `SlimMath.AppendixDSearch` case pins it.

#include <algorithm>
#include <cmath>
#include <vector>

#include "Testbed.h"
#include "extension/ChebyshevFit.h"
#include "extension/SlimPoly.h"

using word = uint32_t;

namespace {

using cheddar::slim::ChebPoly;

// The Cho iteration's LATER window at T = 128: `sq` is a collision probability
// so it lives in [1/live, 1], a theorem rather than a statistic, and its ratio
// of 128 is what no calibration can narrow.
constexpr double kLaterLo = 1.0 / 128.0;
constexpr double kLaterHi = 1.0;

ChebPoly InvSqrtFit(double lo, double hi, int degree) {
  const double a = 0.5 * (hi - lo), b = 0.5 * (hi + lo);
  return cheddar::chebfit::Interpolate(
      [a, b](double v) { return 1.0 / std::sqrt(a * v + b); }, degree);
}

double WorstOnInterval(const cheddar::slim::SlimPlan &plan, const ChebPoly &p) {
  double worst = 0.0;
  for (int i = 0; i < 2049; i++) {
    const double x = -1.0 + 2.0 * i / 2048.0;
    worst = std::max(
        worst, std::abs(plan.PlainEvaluate(x) - cheddar::slim::Eval(p, x)));
  }
  return worst;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Lemma 1 on answers that are known by hand.
// ---------------------------------------------------------------------------
TEST(SlimMath, LemmaOneOnKnownPolynomials) {
  cheddar::slim::DecomposeOptions opt;

  // 2 T_2 = 4x^2 - 2. Its minimum over R is -2 at x = 0, so m = 2 exactly and
  // U^2 + V^2 = 4x^2.
  const ChebPoly p{0.0, 0.0, 2.0};
  const cheddar::slim::Decomposition d = cheddar::slim::Decompose(p, opt);
  ASSERT_TRUE(d.ok) << d.why;
  EXPECT_NEAR(d.m, 2.0, 1e-6);
  EXPECT_LT(d.residual, 1e-12);
  // Both halves must carry the full degree with a positive leading
  // coefficient, or the recursion cannot continue -- that is what eq. (2)'s
  // 1/sqrt 2 is for.
  EXPECT_EQ(cheddar::slim::Degree(d.u), 1);
  EXPECT_EQ(cheddar::slim::Degree(d.v), 1);
  EXPECT_GT(d.u.back(), 0.0);
  EXPECT_GT(d.v.back(), 0.0);

  // A polynomial already positive on R: m comes out negative and the identity
  // must still hold.
  const ChebPoly q{3.0, 0.0, 1.0, 0.0, 0.5};
  const cheddar::slim::Decomposition e = cheddar::slim::Decompose(q, opt);
  ASSERT_TRUE(e.ok) << e.why;
  EXPECT_LT(e.residual, 1e-12);
}

// ---------------------------------------------------------------------------
// 2. Eq. (3)'s tree, and Algorithm 1 in the clear.
// ---------------------------------------------------------------------------
TEST(SlimMath, TheTreeReproducesThePolynomial) {
  cheddar::slim::DecomposeOptions opt;
  for (int degree : {16, 32, 64}) {
    const ChebPoly p = InvSqrtFit(kLaterLo, kLaterHi, degree);
    int k = 0;
    while ((1 << k) < degree) k++;
    for (int j = 1; j <= k; j++) {
      const cheddar::slim::SlimPlan plan = cheddar::slim::BuildPlan(p, j, opt);
      ASSERT_TRUE(plan.ok) << "degree " << degree << " j " << j << ": "
                           << plan.why;
      EXPECT_EQ(plan.k, k);
      EXPECT_EQ(plan.NumBlocks(), 1 << j);
      EXPECT_EQ(static_cast<int>(plan.leaf.size()), 1 << j);
      // Theorem 1's level count, which is the whole trade: `k + 1` levels for
      // degree `2^k`, where Paterson-Stockmeyer spends `k + 1` on `2^(k+1) - 1`.
      EXPECT_EQ(plan.NumLevels(), k + 1);
      // The plan's own arithmetic must sit far below the fit it is evaluating,
      // or slim would be paying for itself twice.
      const double fit_error = std::pow(2.0, degree == 16   ? -1.0
                                             : degree == 32 ? -5.5
                                                            : -14.0);
      EXPECT_LT(WorstOnInterval(plan, p), 0.05 * fit_error)
          << "degree " << degree << " j " << j;
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Appendix D's search, which is the difference between usable and not.
// ---------------------------------------------------------------------------
TEST(SlimMath, AppendixDSearch) {
  const ChebPoly p = InvSqrtFit(kLaterLo, kLaterHi, 64);

  cheddar::slim::DecomposeOptions blind;
  blind.lookahead = false;
  const cheddar::slim::SlimPlan a = cheddar::slim::BuildPlan(p, 3, blind);
  ASSERT_TRUE(a.ok) << a.why;

  cheddar::slim::DecomposeOptions full;  // lookahead defaults on
  const cheddar::slim::SlimPlan b = cheddar::slim::BuildPlan(p, 3, full);
  ASSERT_TRUE(b.ok) << b.why;

  std::cout << "appendix D, degree 64 on [1/128, 1] at j = 3:" << std::endl
            << "  sup-norm objective only: m " << a.worst_m << ", range +"
            << a.range_bits << " bits" << std::endl
            << "  with the m lookahead   : m " << b.worst_m << ", range +"
            << b.range_bits << " bits" << std::endl;

  // Both are correct as polynomial identities; only one is affordable in a
  // ciphertext, and the difference is more than ten bits of dynamic range.
  EXPECT_LT(b.worst_m, a.worst_m);
  EXPECT_LT(b.range_bits, 1.0);
  EXPECT_GT(a.range_bits, 10.0);
}

// ---------------------------------------------------------------------------
// 4. What a level budget buys: slim against Paterson-Stockmeyer.
// ---------------------------------------------------------------------------
TEST(SlimMath, TheBudgetIsStatedNotInferred) {
  for (int levels = 4; levels <= 8; levels++) {
    const cheddar::slim::SlimBudget nofold =
        cheddar::slim::Budget(levels, levels / 2, false);
    const cheddar::slim::SlimBudget fold =
        cheddar::slim::Budget(levels, levels / 2, true);
    // Without the fold slim buys HALF the degree at the same levels, which is
    // why it is a speed technique and never an accuracy one; with appendix D's
    // fold it buys 2^levels against Paterson-Stockmeyer's 2^levels - 1, and
    // then it dominates on both axes.
    EXPECT_EQ(nofold.slim_degree * 2, nofold.paterson_degree + 1);
    EXPECT_EQ(fold.slim_degree, fold.paterson_degree + 1);
    EXPECT_LT(nofold.slim_mults, nofold.paterson_mults);
    std::cout << levels << " levels: PS deg " << nofold.paterson_degree << " ("
              << nofold.paterson_mults << " mults) | slim deg "
              << nofold.slim_degree << " (" << nofold.slim_mults << " mults, "
              << nofold.slim_rotations << " rots) | slim+fold deg "
              << fold.slim_degree << std::endl;
  }
}

// ---------------------------------------------------------------------------
// 5. A negative leading coefficient has no lemma-1 decomposition, and the
//    plan must say so rather than produce a wrong sign.
// ---------------------------------------------------------------------------
TEST(SlimMath, NegativeLeadingCoefficientIsNegatedNotPadded) {
  const ChebPoly r{0.0, 0.0, -2.0};
  cheddar::slim::DecomposeOptions opt;
  const cheddar::slim::SlimPlan plan = cheddar::slim::BuildPlan(r, 1, opt);
  ASSERT_TRUE(plan.ok) << plan.why;
  EXPECT_TRUE(plan.negated);
  EXPECT_NEAR(plan.PlainEvaluate(0.3), -cheddar::slim::Eval(r, 0.3), 1e-9);
}

// ---------------------------------------------------------------------------
// 6. Algorithm 1 on a ciphertext, against its own plan and against the fit.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, SlimAlgorithmOne) {
  constexpr int kBlockSlots = 128;  // 2^t, the slim message's period
  constexpr int kDegree = 16;       // 2^k
  constexpr int kJ = 4;             // full recursion: leaf degree 1

  const int level = default_encryption_level_;
  const int slots = param_->degree_ / 2;
  ASSERT_GE(slots, kBlockSlots * (1 << kJ))
      << "[SYLPH] 3.4 needs 2^(t+j) <= N";

  const ChebPoly p = InvSqrtFit(kLaterLo, kLaterHi, kDegree);
  cheddar::slim::DecomposeOptions opt;
  const cheddar::slim::SlimPlan plan = cheddar::slim::BuildPlan(p, kJ, opt);
  ASSERT_TRUE(plan.ok) << plan.why;
  ASSERT_FALSE(plan.negated);

  const double in_scale = param_->GetScale(level);
  const int out_level = level - plan.NumLevels();
  ASSERT_GE(out_level, 0);
  cheddar::SlimPolyHandler<word> h(context_, plan, kBlockSlots, level,
                                   in_scale, param_->GetScale(out_level));
  h.Compile();
  for (int d : h.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, level);
  }

  // A SLIM message: periodic with period kBlockSlots, so every block sees the
  // same argument and evaluates a different leaf.
  std::vector<double> arg(kBlockSlots);
  std::vector<cheddar::Complex> msg(slots, cheddar::Complex(0.0, 0.0));
  for (int s = 0; s < kBlockSlots; s++) {
    arg[s] = -1.0 + 2.0 * s / (kBlockSlots - 1.0);
  }
  for (int s = 0; s < slots; s++) {
    msg[s] = cheddar::Complex(arg[s % kBlockSlots], 0.0);
  }

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  Ciphertext<word> res;
  h.Evaluate(res, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  EXPECT_EQ(param_->NPToLevel(res.GetNP()), h.GetOutputLevel());
  EXPECT_EQ(level - h.GetOutputLevel(), plan.NumLevels())
      << "theorem 1 says k + 1 levels for degree 2^k";

  std::vector<cheddar::Complex> got;
  DecryptAndDecode(got, res);

  // The answer must be in EVERY block: after the last iteration each block
  // holds U^(0)_0 = P, which is what makes the caller free of a mask.
  double worst_plan = 0.0, worst_fit = 0.0, worst_block_spread = 0.0;
  for (int s = 0; s < kBlockSlots; s++) {
    const double want_plan = plan.PlainEvaluate(arg[s]);
    const double want_fit = cheddar::slim::Eval(p, arg[s]);
    double lo = 1e300, hi = -1e300;
    for (int b = 0; b < (1 << kJ); b++) {
      const double g = got[b * kBlockSlots + s].real();
      lo = std::min(lo, g);
      hi = std::max(hi, g);
      worst_plan = std::max(worst_plan, std::abs(g - want_plan));
      worst_fit = std::max(worst_fit, std::abs(g - want_fit));
    }
    worst_block_spread = std::max(worst_block_spread, hi - lo);
  }
  std::cout << "slim degree " << kDegree << " j " << kJ << ": vs its own plan "
            << worst_plan << ", vs the Chebyshev fit " << worst_fit
            << ", block-to-block spread " << worst_block_spread << std::endl;
  EXPECT_LT(worst_block_spread, 1e-2)
      << "every block must hold P after the last iteration";
  EXPECT_LT(worst_plan, 1e-2);
}

// ---------------------------------------------------------------------------
// 7. Appendix D's fold: the same answer, one level cheaper.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, SlimAppendixDFold) {
  constexpr int kBlockSlots = 128;
  constexpr int kDegree = 16;  // 2^k with k = j, which is what the fold needs
  constexpr int kJ = 4;

  const int level = default_encryption_level_;
  const int slots = param_->degree_ / 2;
  ASSERT_GE(slots, kBlockSlots * (1 << kJ));

  const ChebPoly p = InvSqrtFit(kLaterLo, kLaterHi, kDegree);
  cheddar::slim::DecomposeOptions opt;
  const cheddar::slim::SlimPlan plan = cheddar::slim::BuildPlan(p, kJ, opt);
  ASSERT_TRUE(plan.ok) << plan.why;
  ASSERT_EQ(plan.k, kJ) << "the fold needs a leaf of degree one";

  const double in_scale = param_->GetScale(level);
  // NumLevels(true) is `k`, one less than the unfolded `k + 1`.
  const int out_level = level - plan.NumLevels(/*fold_leading=*/true);
  ASSERT_GE(out_level, 0);
  cheddar::SlimPolyHandler<word> h(context_, plan, kBlockSlots, level, in_scale,
                                   param_->GetScale(out_level),
                                   /*fold_leading=*/true);
  h.Compile();
  for (int d : h.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, level);
  }

  // The caller's half of appendix D: the input arrives ALREADY multiplied by
  // `v^(1)`. In the softmax walk that multiply is the affine map onto the fit
  // domain, whose scalars simply become plaintexts; here it is done in the
  // clear, which is the same thing one operation earlier.
  const std::vector<cheddar::Complex> &v1 = h.GetLeadingMessage();
  ASSERT_EQ(static_cast<int>(v1.size()), param_->MaxNumSlots());
  std::vector<double> arg(kBlockSlots);
  for (int s = 0; s < kBlockSlots; s++) {
    arg[s] = -1.0 + 2.0 * s / (kBlockSlots - 1.0);
  }
  std::vector<cheddar::Complex> msg(slots, cheddar::Complex(0.0, 0.0));
  for (int s = 0; s < slots; s++) {
    msg[s] = cheddar::Complex(v1[s].real() * arg[s % kBlockSlots], 0.0);
  }

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);
  Ciphertext<word> res;
  h.Evaluate(res, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  EXPECT_EQ(level - param_->NPToLevel(res.GetNP()), plan.k)
      << "appendix D's fold makes Algorithm 1 k levels, not k + 1";

  std::vector<cheddar::Complex> got;
  DecryptAndDecode(got, res);
  double worst = 0.0;
  for (int s = 0; s < kBlockSlots; s++) {
    const double want = plan.PlainEvaluate(arg[s]);
    for (int b = 0; b < (1 << kJ); b++) {
      worst = std::max(worst,
                       std::abs(got[b * kBlockSlots + s].real() - want));
    }
  }
  std::cout << "appendix D fold: " << plan.k << " levels (unfolded would be "
            << plan.k + 1 << "), worst " << worst << std::endl;
  EXPECT_LT(worst, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
