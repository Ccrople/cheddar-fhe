// ScrambleAuto and the whole CMT, [KANG] (ePrint 2025/1957) Algorithm 3.
//
// FOURTH STEP OF THE BATCH CC-MM. TweakTest covered the key-free stages; this
// adds the one that needs key material and then assembles Algorithm 3:
//
//     X^i*ct -> TWEAK(+1) -> d^-1, ScrambleAuto -> TWEAK(-1) -> X^-j*ct
//
// THE LEVEL CLAIM IS THE POINT. Every stage above is level-free -- monomials
// and additions, key switches and permutations, and a d^-1 that is a change to
// the recorded scale rather than an operation. The batch CC-MM spends the one
// multiplicative level of this ring on its tensor product, so a CMT that cost
// one would leave Algorithm 4 with nowhere to go. Both tests assert the level
// is unchanged rather than trusting it.
//
// THE INVOLUTION IS THE STRONG TEST. CMT is a transpose, so running it twice
// must return the input. That check needs no commitment to how the matrix is
// laid out inside the ciphertexts -- which is exactly the part the digest does
// not pin down -- and a wrong permutation t*, a wrong automorphism index or a
// misplaced d^-1 all break it.
//
// SEPARATE BINARY, degree 4096; see SmallRingNttTest.cpp for the trap.

#undef ENABLE_EXTENSION

#include <cmath>
#include <cstdlib>

#include "Testbed.h"
#include "core/Cmt.h"

using word = uint32_t;

namespace {
// The small ring is selected by environment so the same suite runs against
// either scale: ringdegree12_30 by default, ringdegree12_35 -- the 2^35 pair
// [SYLPH]'s ladder needs -- when CHEDDAR_SMALL_PARAM says so.
std::vector<const char *> SmallRingParams() {
  const char *env = std::getenv("CHEDDAR_SMALL_PARAM");
  if (env != nullptr && env[0] != 0) return {env};
  return {"ringdegree12_30.json"};
}
}  // namespace

namespace {

constexpr int kSubDegree = 128;  // [SYLPH] 3.3 uses SinC_{2^7, 2^12}, d = 32

// m(X) -> m(X^g) on real coefficients: coefficient p moves to p*g mod 2N, and
// the wrap past X^N carries a sign.
std::vector<double> AutomorphismCoeffs(const std::vector<double> &c,
                                       long long g, int degree) {
  std::vector<double> res(degree, 0.0);
  const long long two_n = 2LL * degree;
  for (int p = 0; p < degree; p++) {
    long long q = (static_cast<long long>(p) * g) % two_n;
    double sign = 1.0;
    if (q >= degree) {
      q -= degree;
      sign = -1.0;
    }
    res[q] += sign * c[p];
  }
  return res;
}

// Deliberately brute force, where the implementation uses a Newton iteration:
// the two agreeing is then evidence rather than a restatement.
long long InvMod2N(long long g, long long two_n) {
  for (long long x = 1; x < two_n; x += 2) {
    if ((g * x) % two_n == 1) return x;
  }
  return -1;
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

// The group theory, before any ciphertext: {2kt+1} must be a subgroup of order
// d inside <5>, or none of this has a rotation key to stand on.
TEST_P(Testbed32, ScrambleAutoIndicesFormTheSubgroup) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const long long two_n = 2LL * degree;

  CmtHandler<word> cmt(*param_, context_->ntt_handler_);
  const std::vector<int> indices = cmt.ScrambleAutoRotationIndices(kSubDegree);
  ASSERT_EQ(static_cast<int>(indices.size()), d - 1);

  for (int t = 1; t < d; t++) {
    const long long g = (2LL * kSubDegree * t + 1) % two_n;
    EXPECT_EQ(param_->GetGaloisFactor(indices[t - 1]), g)
        << "index " << indices[t - 1] << " is not 2kt+1 at t = " << t;
    // Every index is a multiple of k/2, which is what H = <5^(k/2)> means.
    EXPECT_EQ(indices[t - 1] % (kSubDegree / 2), 0)
        << "index " << indices[t - 1] << " is outside the subgroup";
    EXPECT_NE(InvMod2N(g, two_n), -1) << "2kt+1 not invertible at t = " << t;
  }
  std::cout << "d = " << d << ", " << indices.size()
            << " non-identity automorphisms, all in the subgroup" << std::endl;
}

// ScrambleAuto against its definition, res[t] = cts[t*](X^(2kt+1)).
TEST_P(Testbed32, ScrambleAutoAppliesTheStatedAutomorphism) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);
  const long long two_n = 2LL * degree;

  CmtHandler<word> cmt(*param_, context_->ntt_handler_);
  for (int index : cmt.ScrambleAutoRotationIndices(kSubDegree)) {
    interface_->PrepareRotationKey(index, level);
  }

  std::vector<std::vector<double>> m(d, std::vector<double>(degree));
  std::vector<Ciphertext<word>> cts(d);
  for (int t = 0; t < d; t++) {
    Random::SampleUniformReal(m[t].data(), degree, -1.0, 1.0);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, m[t]);
    interface_->Encrypt(cts[t], pt);
  }

  std::vector<Ciphertext<word>> res;
  cmt.ScrambleAuto(context_, res, cts, kSubDegree, interface_->GetEvkMap());
  ASSERT_EQ(static_cast<int>(res.size()), d);

  double worst = 0.0;
  for (int t = 0; t < d; t++) {
    ASSERT_EQ(param_->NPToLevel(res[t].GetNP()), level)
        << "a key switch and a permutation cost no level";

    const long long g = (2LL * kSubDegree * t + 1) % two_n;
    const long long inverse = InvMod2N(g, two_n);
    ASSERT_NE(inverse, -1);
    ASSERT_EQ((inverse - 1) % (2LL * kSubDegree), 0)
        << "the inverse left the subgroup at t = " << t;
    const int t_star =
        static_cast<int>(((inverse - 1) / (2LL * kSubDegree)) % d);

    Plaintext<word> out;
    interface_->Decrypt(out, res[t]);
    std::vector<double> got;
    context_->encoder_.DecodeCoeff(got, out);

    const std::vector<double> want = AutomorphismCoeffs(m[t_star], g, degree);
    worst = std::max(worst, MaxAbsDiff(got, want));
  }
  std::cout << "ScrambleAuto, d = " << d << ": max error " << worst
            << std::endl;
  ASSERT_LT(worst, 1e-4);
}

// CMT is a transpose, so twice is the identity. This is the check that needs
// no commitment to the matrix layout, and the one that a wrong t*, a wrong
// automorphism index or a misplaced d^-1 all fail.
TEST_P(Testbed32, CmtIsAnInvolution) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  CmtHandler<word> cmt(*param_, context_->ntt_handler_);
  for (int index : cmt.ScrambleAutoRotationIndices(kSubDegree)) {
    interface_->PrepareRotationKey(index, level);
  }

  // TWEAK sums d of these before the d^-1 relabelling brings it back, so the
  // intermediate reaches d * 0.5 * 2^30 against a level-1 modulus of 2^69.8.
  std::vector<std::vector<double>> m(d, std::vector<double>(degree));
  std::vector<Ciphertext<word>> cts(d);
  for (int j = 0; j < d; j++) {
    Random::SampleUniformReal(m[j].data(), degree, -0.5, 0.5);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, m[j]);
    interface_->Encrypt(cts[j], pt);
  }

  std::vector<Ciphertext<word>> once, twice;
  cmt.Cmt(context_, once, cts, kSubDegree, interface_->GetEvkMap());
  cmt.Cmt(context_, twice, once, kSubDegree, interface_->GetEvkMap());
  ASSERT_EQ(static_cast<int>(twice.size()), d);

  double worst = 0.0;
  for (int j = 0; j < d; j++) {
    ASSERT_EQ(param_->NPToLevel(twice[j].GetNP()), level)
        << "CMT must consume no level, twice over";

    Plaintext<word> out;
    interface_->Decrypt(out, twice[j]);
    std::vector<double> got;
    context_->encoder_.DecodeCoeff(got, out);
    worst = std::max(worst, MaxAbsDiff(got, m[j]));
  }

  std::cout << "CMT twice, d = " << d << ", scale 2^"
            << std::log2(twice[0].GetScale()) << ": max error " << worst
            << std::endl;
  ASSERT_LT(worst, 1e-3);
}

// The batched path (one buffer, the ciphertext index on blockIdx.z, ~40
// launches) against the serial one (Context's per-ciphertext operations,
// ~6.5k launches at d = 128) on the same inputs: every stage is modular
// arithmetic through the same kernels or the same per-element operations, so
// the two must agree WORD FOR WORD, not to a tolerance. Checked at the top
// level and at level 1 -- the lifted ring's Cmt runs at level 1, where the
// key switch has two digits and a terminal-prime offset.
TEST_P(Testbed32, TheBatchedCmtIsTheSerialOneWordForWord) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const int top = param_->max_level_;
  const double scale = DetermineScale(top);

  CmtHandler<word> cmt(*param_, context_->ntt_handler_);
  for (int index : cmt.ScrambleAutoRotationIndices(kSubDegree)) {
    interface_->PrepareRotationKey(index, top);
  }

  std::vector<Ciphertext<word>> at_top(d);
  for (int j = 0; j < d; j++) {
    std::vector<double> m(degree);
    Random::SampleUniformReal(m.data(), degree, -0.5, 0.5);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, top, scale, m);
    interface_->Encrypt(at_top[j], pt);
  }

  for (int level : {top, 1}) {
    if (level > top) continue;
    std::vector<Ciphertext<word>> cts(d);
    for (int j = 0; j < d; j++) context_->LevelDown(cts[j], at_top[j], level);

    ASSERT_TRUE(CmtHandler<word>::BatchEnabled())
        << "CHEDDAR_CMT_SERIAL=1 would compare the serial path with itself";
    std::vector<Ciphertext<word>> serial, batched;
    cmt.CmtSerial(context_, serial, cts, kSubDegree, interface_->GetEvkMap());
    cmt.Cmt(context_, batched, cts, kSubDegree, interface_->GetEvkMap());
    ASSERT_EQ(static_cast<int>(batched.size()), d);

    size_t differing = 0, total = 0;
    for (int j = 0; j < d; j++) {
      ASSERT_EQ(batched[j].GetNP(), serial[j].GetNP());
      ASSERT_EQ(batched[j].GetScale(), serial[j].GetScale());
      ASSERT_EQ(batched[j].GetNumSlots(), serial[j].GetNumSlots());
      ASSERT_FALSE(batched[j].HasRx());
      const DeviceVector<word> *got[2] = {&batched[j].bx_, &batched[j].ax_};
      const DeviceVector<word> *want[2] = {&serial[j].bx_, &serial[j].ax_};
      for (int p = 0; p < 2; p++) {
        HostVector<word> a, b;
        CopyDeviceToHost(a, *got[p]);
        CopyDeviceToHost(b, *want[p]);
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++) differing += (a[i] != b[i]);
        total += a.size();
      }
    }
    std::cout << "level " << level << ", d = " << d << ": " << differing
              << " of " << total << " words differ" << std::endl;
    EXPECT_EQ(differing, 0u) << "at level " << level;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cmt, Testbed32, testing::ValuesIn(SmallRingParams()),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
