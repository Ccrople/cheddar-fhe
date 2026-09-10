// [SYLPH] section 4.2 (eq. 5), appendix E (lemma 2) and section 4.3.
//
// The `SylphPcmmMath.*` cases are index algebra and need NO GPU --
// `sylph_pcmm_test --gtest_filter='SylphPcmmMath.*'` runs anywhere. That is
// where this algorithm can be wrong: the plaintext's index,
// `rot_R^(-l(i + j b) - j b) . rot_C^(i + j b)`, carries one shift from
// lemma 2 and another from hoisting the giant step, and getting either wrong
// gives a product that is a permutation of the right answer rather than an
// obviously broken one.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "Testbed.h"
#include "extension/SylphPcmm.h"

using word = uint32_t;

namespace {

using cheddar::sylph_pcmm::Mat;
using cheddar::sylph_pcmm::MatMul;
using cheddar::sylph_pcmm::RotC;
using cheddar::sylph_pcmm::RotR;
using cheddar::sylph_pcmm::Sigma;
using cheddar::sylph_pcmm::Tau;
using cheddar::sylph_pcmm::TauPow;

double MaxDiff(const Mat &a, const Mat &b) {
  double w = 0.0;
  for (size_t i = 0; i < a.size(); i++) w = std::max(w, std::abs(a[i] - b[i]));
  return w;
}

Mat Rand(int d, unsigned seed, double sigma = 1.0) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> n(0.0, sigma);
  Mat m(static_cast<size_t>(d) * d);
  for (double &v : m) v = n(rng);
  return m;
}

// SoftMax down each COLUMN of a d x d matrix, which is the axis PC-attention
// normalises over.
Mat SoftMaxColumns(const Mat &m, int d) {
  Mat r(m.size(), 0.0);
  for (int j = 0; j < d; j++) {
    double mx = -1e300;
    for (int i = 0; i < d; i++) mx = std::max(mx, m[i * d + j]);
    double sum = 0.0;
    for (int i = 0; i < d; i++) {
      r[i * d + j] = std::exp(m[i * d + j] - mx);
      sum += r[i * d + j];
    }
    for (int i = 0; i < d; i++) r[i * d + j] /= sum;
  }
  return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Appendix E's lemma 2, which is what buys the level.
// ---------------------------------------------------------------------------
TEST(SylphPcmmMath, LemmaTwo) {
  for (int d : {4, 8, 16}) {
    const Mat m = Rand(d, 11u + d);
    for (int k = 0; k < d; k++) {
      EXPECT_LT(MaxDiff(Tau(RotR(m, d, k), d), RotR(Tau(m, d), d, k)), 1e-12)
          << "tau(rot_R^k(M)) != rot_R^k(tau(M)) at d " << d << " k " << k;
      EXPECT_LT(MaxDiff(Tau(RotC(m, d, k), d),
                        RotR(RotC(Tau(m, d), d, k), d, -k)),
                1e-12)
          << "tau(rot_C^k(M)) != rot_R^-k rot_C^k tau(M) at d " << d << " k "
          << k;
    }
  }
}

// ---------------------------------------------------------------------------
// 2. JKLS eq. (4), the identity eq. (5) rearranges.
// ---------------------------------------------------------------------------
TEST(SylphPcmmMath, JklsEquationFour) {
  for (int d : {4, 8, 16}) {
    const Mat a = Rand(d, 3u + d), b = Rand(d, 5u + d);
    Mat c(static_cast<size_t>(d) * d, 0.0);
    const Mat sa = Sigma(a, d), tb = Tau(b, d);
    for (int k = 0; k < d; k++) {
      const Mat l = RotC(sa, d, k), r = RotR(tb, d, k);
      for (size_t s = 0; s < c.size(); s++) c[s] += l[s] * r[s];
    }
    EXPECT_LT(MaxDiff(c, MatMul(a, b, d)), 1e-10) << "d " << d;
  }
}

// ---------------------------------------------------------------------------
// 3. Eq. (5) itself, at every tau power and every legal baby-step split.
// ---------------------------------------------------------------------------
TEST(SylphPcmmMath, EquationFive) {
  for (int d : {4, 8, 16, 32}) {
    const Mat a = Rand(d, 17u + d), b = Rand(d, 23u + d);
    const Mat c = MatMul(a, b, d);
    for (int l = 0; l <= 3; l++) {
      const Mat tau_b = TauPow(b, d, l + 1);  // what section 4.2 feeds
      const Mat want = TauPow(c, d, l);       // what it promises
      for (int bb = 1; bb <= d; bb++) {
        if (d % bb != 0) continue;
        const cheddar::sylph_pcmm::SylphPcmmPlan plan =
            cheddar::sylph_pcmm::BuildPlan(a, d, l, bb);
        ASSERT_TRUE(plan.ok) << plan.why;
        EXPECT_LT(MaxDiff(cheddar::sylph_pcmm::PlainApply(plan, tau_b), want),
                  1e-10)
            << "d " << d << " l " << l << " b " << bb;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 4. The cost: O(sqrt d) rotations and ONE level, which is the section's claim.
// ---------------------------------------------------------------------------
TEST(SylphPcmmMath, CostIsSquareRootAndOneLevel) {
  for (int d : {16, 64, 128, 256}) {
    const cheddar::sylph_pcmm::SylphPcmmPlan plan =
        cheddar::sylph_pcmm::BuildPlan(Rand(d, 29u + d), d, 1);
    ASSERT_TRUE(plan.ok) << plan.why;
    EXPECT_EQ(plan.b * plan.g, d);
    EXPECT_LE(plan.NumRotations(), 2 * static_cast<int>(std::sqrt(d)) + 1)
        << "the BSGS split is not a square root at d " << d;
    EXPECT_EQ(static_cast<int>(plan.RotationDistances().size()),
              plan.NumRotations());
    // Every rotation is a `rot_R`, so every distance is a whole number of rows.
    // A column rotation on a ciphertext is the level JKLS spends and eq. (5)
    // does not.
    for (int dist : plan.RotationDistances()) EXPECT_EQ(dist % d, 0);
    std::cout << "d " << d << ": b " << plan.b << " g " << plan.g << ", "
              << plan.NumRotations() << " rotations (naive " << (d - 1)
              << "), 1 level" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// 5. Section 4.3: SoftMax does not mind being handed tau(M).
// ---------------------------------------------------------------------------
TEST(SylphPcmmMath, SoftMaxSeesColumnsOfTau) {
  // "Each vector (tau(M)_{j,i})_j is the i-th column of M rotated by i ... and
  // SoftMax((tau(M)_{j,i})_j) is SoftMax((M_{j,i})_j) rotated by i, so that the
  // output is tau(SoftMax(...))". SoftMax is permutation equivariant, so the
  // operator between the two PCMMs is unchanged -- no mask, no re-layout, no
  // level. This states that as an identity rather than as a remark.
  for (int d : {4, 8, 16}) {
    const Mat m = Rand(d, 41u + d, 3.0);
    const Mat got = SoftMaxColumns(Tau(m, d), d);
    const Mat want = Tau(SoftMaxColumns(m, d), d);
    EXPECT_LT(MaxDiff(got, want), 1e-12) << "d " << d;
  }
}

// ---------------------------------------------------------------------------
// 6. Eq. (5) on a ciphertext, against the host reference and against A B.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, EquationFiveOnEncrypted) {
  constexpr int kD = 32;   // 1024 slots, so it fits every fixture
  constexpr int kTau = 1;  // input tau^2(B), output tau^1(C)

  const int level = default_encryption_level_;
  const int slots = param_->degree_ / 2;
  ASSERT_GE(slots, kD * kD);

  // Weights the size a projection's actually are, so the product's magnitude
  // is representative rather than unit.
  const Mat a = Rand(kD, 101u, 0.05);
  const Mat b = Rand(kD, 202u, 1.0);
  const cheddar::sylph_pcmm::SylphPcmmPlan plan =
      cheddar::sylph_pcmm::BuildPlan(a, kD, kTau);
  ASSERT_TRUE(plan.ok) << plan.why;

  const double in_scale = param_->GetScale(level);
  cheddar::SylphPcmm<word> pcmm(context_, plan, level, in_scale);
  pcmm.Compile();
  for (int dist : pcmm.GetRotationDistances()) {
    interface_->PrepareRotationKey(dist, level);
  }

  const Mat tau_b = TauPow(b, kD, kTau + 1);
  std::vector<cheddar::Complex> msg(slots, cheddar::Complex(0.0, 0.0));
  for (size_t s = 0; s < tau_b.size(); s++) {
    msg[s] = cheddar::Complex(tau_b[s], 0.0);
  }

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  Ciphertext<word> res;
  pcmm.Apply(res, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  EXPECT_EQ(param_->NPToLevel(res.GetNP()), pcmm.GetOutputLevel());
  EXPECT_EQ(level - pcmm.GetOutputLevel(), 1)
      << "section 4.2's whole claim is ONE multiplicative level";

  std::vector<cheddar::Complex> got;
  DecryptAndDecode(got, res);

  const Mat want = TauPow(MatMul(a, b, kD), kD, kTau);
  double worst = 0.0, ref = 0.0;
  for (size_t s = 0; s < want.size(); s++) {
    worst = std::max(worst, std::abs(got[s].real() - want[s]));
    ref = std::max(ref, std::abs(want[s]));
  }
  std::cout << "eq. (5) encrypted, d " << kD << ", tau^" << kTau << ": worst "
            << worst << " against |C| " << ref << " ("
            << -std::log2(worst / ref) << " bits), " << pcmm.GetNumRotations()
            << " rotations" << std::endl;
  EXPECT_LT(worst / ref, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
