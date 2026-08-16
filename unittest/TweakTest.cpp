// TWEAK, [KANG] (ePrint 2025/1957) Algorithm 2, and the monomial arithmetic it
// runs on.
//
// THIRD STEP OF THE BATCH CC-MM, and the key-free half of CMT. CMT is
//
//     X^i * ct -> TWEAK(+1) -> d^-1 and Auto(.; 2kt+1) -> TWEAK(-1) -> X^-i*ct
//
// and everything here is the first, second, fourth and fifth of those. The
// third -- ScrambleAuto, where the d key switchings live -- is not implemented
// yet, so nothing in this file needs a key or a multiplicative level.
//
// WHY THE MONOMIALS ARE EXACT. X^e is encoded as one coefficient of +-1 at
// scale 1, so Mult(Ct, Pt) returns the product at the input's own scale and
// level. Nothing is rounded, which means the reference below can be an exact
// coefficient permutation and any disagreement is a real error rather than a
// tolerance question.
//
// SEPARATE BINARY, degree 4096; see SmallRingNttTest.cpp for the trap.

#undef ENABLE_EXTENSION

#include <cmath>

#include "Testbed.h"
#include "core/Cmt.h"

using word = uint32_t;

namespace {

// [SYLPH] section 3.3 runs the batch CC-MM in SinC_{2^7, 2^12}, so d = 32.
constexpr int kSubDegree = 128;

// c' = X^e * c in R[X]/(X^N + 1), on real coefficients. The wrap past X^N
// carries a sign, which is the whole content of the negacyclic ring.
std::vector<double> MonomialShift(const std::vector<double> &c, long long e,
                                  int degree) {
  std::vector<double> res(degree, 0.0);
  const long long two_n = 2LL * degree;
  for (int p = 0; p < degree; p++) {
    long long q = ((p + e) % two_n + two_n) % two_n;
    double sign = 1.0;
    if (q >= degree) {
      q -= degree;
      sign = -1.0;
    }
    res[q] += sign * c[p];
  }
  return res;
}

double MaxAbsDiff(const std::vector<double> &got,
                  const std::vector<double> &want) {
  double worst = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    worst = std::max(worst, std::abs(got[i] - want[i]));
  }
  return worst;
}

}  // namespace

// Monomial multiplication on its own, including the two cases that carry the
// sign: an exponent past the ring degree, and a negative one.
TEST_P(Testbed32, MonomialMultiplicationMatchesCoefficientShift) {
  const int degree = param_->degree_;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  CmtHandler<word> cmt(*param_, context_->ntt_handler_);

  std::vector<double> coeffs(degree);
  Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  for (long long e : {0LL, 1LL, 255LL, 4095LL, 4096LL, 5000LL, 8191LL, -1LL,
                      -4096LL, -5000LL}) {
    Plaintext<word> monomial;
    cmt.EncodeMonomial(monomial, level, static_cast<int>(e));

    Ciphertext<word> shifted;
    context_->Mult(shifted, ct, monomial);

    ASSERT_EQ(param_->NPToLevel(shifted.GetNP()), level)
        << "a monomial multiplication must not consume a level";
    EXPECT_NEAR(shifted.GetScale() / scale, 1.0, 1e-9)
        << "a scale-1 plaintext must leave the scale alone";

    Plaintext<word> out;
    interface_->Decrypt(out, shifted);
    std::vector<double> got;
    context_->encoder_.DecodeCoeff(got, out);

    const std::vector<double> want = MonomialShift(coeffs, e, degree);
    const double err = MaxAbsDiff(got, want);
    std::cout << "X^" << e << ": max error " << err << std::endl;
    ASSERT_LT(err, 1e-5) << "monomial multiplication is wrong at e = " << e;
  }
}

// TWEAK against its definition, res[i] = sum_j X^(2*k*i*j*sign) * m_j,
// evaluated on the host as exact coefficient shifts.
TEST_P(Testbed32, TweakMatchesItsDefinition) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);
  ASSERT_EQ(d, 32);

  CmtHandler<word> cmt(*param_, context_->ntt_handler_);

  // Small inputs: TWEAK sums d of them, so the encoded result reaches d * 2^30
  // against a level-1 modulus of 2^69.8.
  std::vector<std::vector<double>> m(d, std::vector<double>(degree));
  std::vector<Ciphertext<word>> cts(d);
  for (int j = 0; j < d; j++) {
    Random::SampleUniformReal(m[j].data(), degree, -0.5, 0.5);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, m[j]);
    interface_->Encrypt(cts[j], pt);
  }

  for (int sign : {1, -1}) {
    std::vector<Ciphertext<word>> res;
    cmt.Tweak(context_, res, cts, kSubDegree, sign);
    ASSERT_EQ(static_cast<int>(res.size()), d);

    double worst = 0.0;
    for (int i = 0; i < d; i++) {
      ASSERT_EQ(param_->NPToLevel(res[i].GetNP()), level)
          << "TWEAK must consume no level";

      Plaintext<word> out;
      interface_->Decrypt(out, res[i]);
      std::vector<double> got;
      context_->encoder_.DecodeCoeff(got, out);

      std::vector<double> want(degree, 0.0);
      for (int j = 0; j < d; j++) {
        const long long e =
            2LL * kSubDegree * i * j * sign;
        const std::vector<double> term = MonomialShift(m[j], e, degree);
        for (int p = 0; p < degree; p++) want[p] += term[p];
      }
      worst = std::max(worst, MaxAbsDiff(got, want));
    }
    std::cout << "TWEAK sign " << sign << ", d = " << d << ", k = "
              << kSubDegree << ": max error " << worst << std::endl;
    ASSERT_LT(worst, 1e-4);
  }
}

// X^(2k) has order exactly d in the ring, so the two directions compose to
// multiplication by d -- which is why Algorithm 3 carries a d^-1. Checking it
// separately is worth the run: it is the property Algorithm 3 depends on, and
// it fails for a different reason than the definition test would (a twiddle
// that is right up to conjugation still passes one of them).
TEST_P(Testbed32, TweakRoundTripsToTheInputTimesD) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  CmtHandler<word> cmt(*param_, context_->ntt_handler_);

  // d^2 = 1024 is the worst intermediate growth, so keep the inputs small.
  std::vector<std::vector<double>> m(d, std::vector<double>(degree));
  std::vector<Ciphertext<word>> cts(d);
  for (int j = 0; j < d; j++) {
    Random::SampleUniformReal(m[j].data(), degree, -0.25, 0.25);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, m[j]);
    interface_->Encrypt(cts[j], pt);
  }

  std::vector<Ciphertext<word>> forward, back;
  cmt.Tweak(context_, forward, cts, kSubDegree, 1);
  cmt.Tweak(context_, back, forward, kSubDegree, -1);
  ASSERT_EQ(static_cast<int>(back.size()), d);

  double worst = 0.0;
  for (int j = 0; j < d; j++) {
    Plaintext<word> out;
    interface_->Decrypt(out, back[j]);
    std::vector<double> got;
    context_->encoder_.DecodeCoeff(got, out);

    std::vector<double> want(degree);
    for (int p = 0; p < degree; p++) want[p] = d * m[j][p];
    worst = std::max(worst, MaxAbsDiff(got, want));
  }

  std::cout << "TWEAK(-1) . TWEAK(+1) against " << d << " * input: max error "
            << worst << std::endl;
  ASSERT_LT(worst, 1e-4);
}

INSTANTIATE_TEST_SUITE_P(
    Tweak, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
