// The keyless lift of a conjugate-invariant ciphertext into the ordinary
// ring of the same conductor, and the keyless descent back (Doing.md 1.5bk).
//
// WHY THESE MAPS EXIST. 1.5bh forced the batch CC-MM onto R+_4096, and
// [KANG]'s machinery for it -- TWEAK's monomial twiddles, the CMT built on
// them -- has no conjugate-invariant form: inside R+ the subring-fixing
// automorphisms act on the module pairs (i, d-i) jointly, and separating a
// pair needs the odd part X^a - X^-a, which the totally real ring quotients
// away. R+_4096 is the real subring of the ordinary conductor-16384 ring of
// degree 8192, so the CC-MM stretch lifts there -- where monomials exist and
// the ordinary Algorithm 4 already runs -- and descends afterwards. Both
// crossings are keyless: the inclusion is a ring homomorphism, and the
// embedded secret is its own conjugate, so the trace T = id + conj projects
// a ciphertext back without a key switch.
//
// THE TRACE, NOT THE AVERAGE. T/2 is the same projection of the element,
// but 2^-1 mod q of an odd small integer is a residue near q/2, so the
// averaged descent destroys the noise: about half the ciphertext noise's
// antisymmetric parts are odd, E puts ~q/2 there, and the result decrypts
// to garbage -- invisibly so on lifted ciphertexts, whose exactly
// mirror-antisymmetric components make the halving collapse to the
// identity. That failure mode was found by measurement here, which is why
// DescendReadsTheEvenPart exists: it is the general-component case no other
// test covers. The factor 2 rides the recorded scale.
//
// WHAT IS CHECKED, AND AGAINST WHAT.
//
//   TheOrdinaryRingWorksAtDegreeEightK -- the target ring alone: encrypt,
//       multiply, decrypt against host slot products. NTT.h declares logN
//       12-16 but nothing had ever run 13; this isolates "the ring works"
//       from "the lift works".
//
//   LiftDecryptsUnderTheEmbeddedSecret -- encrypt on R+, lift, decrypt with
//       the big ring's interface holding LiftSecret of the CI secret. The
//       coefficients must be the exact index map c_j -> X^j - X^{N-j}, and
//       the big ring's slots must be real and, as a multiset, exactly the
//       R+ slots -- the two rings read the same canonical embedding.
//
//   DescendInvertsLift -- down and back on the lifted image.
//
//   DescendReadsTheEvenPart -- the general-component case: a ciphertext
//       encrypted NATIVELY on the big ring (components fresh and general,
//       message arbitrary) must descend to the even part of its message,
//       computed on the host from the plaintext alone.
//
//   AProductOnTheLiftedRingDescendsToTheCiProduct -- the round trip the
//       CC-MM will ride: lift two ciphertexts, HMult on the ordinary ring
//       under the embedded secret (whose multiplication key the interface
//       built from the adopted secret like any other), descend the product,
//       decrypt on R+: slotwise z * w. The product's message is even because
//       the real subring is closed under multiplication, so the descent
//       loses nothing.
//
//   ABatchCcmmOnTheLiftedRingComputesTheCiProducts -- [KANG] Algorithm 4 run
//       UNMODIFIED on the lifted ring at sub-degree 2k over lifted CI SinC
//       operands. The lift is not block-clean: per lane the lifted bundle is
//       Lambda = I + w^-1 * P E of the CI batch, with P the block flip
//       i -> d-i, E killing block 0 and w the lane's primitive (4k)-th root
//       -- c_i = X^i - X^(N-i) splits across the ordinary blocks i and d-i.
//       A full contraction therefore computes A (I + cos(theta) P E) B, not
//       A B. The contract that removes the twist exactly: the lhs bundle
//       uses only ciphertexts x < d/2, and the rhs confines its data to CI
//       blocks x < d/2. Then every term of A P E B pairs a live lhs column
//       with a dead rhs row, the contamination is zero identically --
//       Algorithm 4 unchanged, no extra key, no noise price -- and the
//       descent reads the exact per-lane real products (d x d/2)(d/2 x d)
//       against a host reference. Over these operands the product is even
//       BEFORE the descent (D0 P conj(Lambda AB) = Lambda AB), and that is
//       checked on the big ring's own coefficients.
//
//   TheFullContractionCarriesTheCosineTwistedFlip -- the same run without
//       the contract, as the measured form of the twist: per CI lane the
//       excess over A B must be exactly lambda_t * A P E B for a fitted
//       scalar lambda_t, and the multiset of the lambda_t must be the k
//       cosines cos(pi (2j+1) / 2k) of the primitive (4k)-th roots. That
//       pins the whole Lambda algebra the contract is derived from, and
//       doubles as the negative control: without the contract the naive
//       read is wrong by O(1).

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/BatchCcmm.h"
#include "core/CiLift.h"

using word = uint32_t;
using cheddar::BatchCcmmHandler;
using cheddar::Ciphertext;
using cheddar::CiLiftHandler;
using cheddar::Complex;
using cheddar::Plaintext;
using Ring = ringfixture::Ring<word>;

namespace {
constexpr const char *kCiParam = "ci12_35.json";
constexpr const char *kBigParam = "ringdegree13_35.json";
}  // namespace

TEST(CiLift, TheOrdinaryRingWorksAtDegreeEightK) {
  Ring big(kBigParam);
  const int level = big.param->max_level_;
  const int num_slots = big.param->MaxNumSlots();
  const double scale = big.param->GetScale(level);

  std::mt19937_64 gen(0x11F7);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<Complex> z(num_slots), w(num_slots);
  for (auto &v : z) v = Complex(dist(gen), dist(gen));
  for (auto &v : w) v = Complex(dist(gen), dist(gen));

  Plaintext<word> pz, pw;
  big.context->encoder_.Encode(pz, level, scale, z);
  big.context->encoder_.Encode(pw, level, scale, w);
  Ciphertext<word> cz, cw, cres;
  big.ui->Encrypt(cz, pz);
  big.ui->Encrypt(cw, pw);
  big.context->HMult(cres, cz, cw, big.ui->GetMultiplicationKey(), true);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  Plaintext<word> out;
  big.ui->Decrypt(out, cres);
  std::vector<Complex> got;
  big.context->encoder_.Decode(got, out);
  ASSERT_EQ(static_cast<int>(got.size()), num_slots);

  double worst = 0.0;
  for (int s = 0; s < num_slots; s++) {
    worst = std::max(worst, std::abs(got[s] - z[s] * w[s]));
  }
  std::cout << "degree-8192 HMult at level " << level << ": max error "
            << worst << std::endl;
  ASSERT_LT(worst, 1e-3);
}

TEST(CiLift, LiftDecryptsUnderTheEmbeddedSecret) {
  Ring ci(kCiParam);
  Ring big(kBigParam,
           CiLiftHandler<word>::LiftSecret(ci.ui->GetSecretCoeffs()));

  const int n = ci.Degree();
  const int N = big.Degree();
  const int level = ci.param->max_level_;
  const double scale = ci.param->GetScale(level);

  CiLiftHandler<word> lift(ci.context, big.context);

  std::mt19937_64 gen(0xE18ED);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> coeffs(n);
  for (auto &c : coeffs) c = dist(gen);

  Plaintext<word> pt;
  ci.context->encoder_.EncodeCoeff(pt, level, scale, coeffs);
  Ciphertext<word> ct;
  ci.ui->Encrypt(ct, pt);

  Ciphertext<word> lifted;
  lift.Lift(lifted, ct);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // The coefficients: the exact index map.
  Plaintext<word> back;
  big.ui->Decrypt(back, lifted);
  std::vector<double> got;
  big.context->encoder_.DecodeCoeff(got, back);
  ASSERT_EQ(static_cast<int>(got.size()), N);

  double worst = std::abs(got[n]);  // c_n = 0: nothing may land there
  for (int j = 0; j < n; j++) {
    worst = std::max(worst, std::abs(got[j] - coeffs[j]));
    if (j != 0) worst = std::max(worst, std::abs(got[N - j] + coeffs[j]));
  }
  std::cout << "lift, coefficients: max |diff| = " << worst << std::endl;
  EXPECT_LT(worst, 1e-3);

  // The slots: a lifted element is real in every big-ring slot, and the two
  // rings evaluate the same field embeddings, so the value multisets agree
  // exactly -- only the slot bookkeeping differs. Compared sorted.
  std::vector<Complex> ci_slots, big_slots;
  ci.context->encoder_.Decode(ci_slots, pt);
  big.context->encoder_.Decode(big_slots, back);
  ASSERT_EQ(static_cast<int>(ci_slots.size()), n);
  ASSERT_EQ(static_cast<int>(big_slots.size()), n);

  double worst_imag = 0.0;
  std::vector<double> a(n), b(n);
  for (int s = 0; s < n; s++) {
    worst_imag = std::max(worst_imag, std::abs(big_slots[s].imag()));
    a[s] = big_slots[s].real();
    b[s] = ci_slots[s].real();
  }
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  double worst_slots = 0.0;
  for (int s = 0; s < n; s++) {
    worst_slots = std::max(worst_slots, std::abs(a[s] - b[s]));
  }
  std::cout << "lift, slots: max imag " << worst_imag
            << ", sorted multisets: max |diff| = " << worst_slots
            << std::endl;
  EXPECT_LT(worst_imag, 1e-4);
  EXPECT_LT(worst_slots, 1e-4);
}

TEST(CiLift, DescendInvertsLift) {
  Ring ci(kCiParam);
  Ring big(kBigParam,
           CiLiftHandler<word>::LiftSecret(ci.ui->GetSecretCoeffs()));

  const int n = ci.Degree();
  const int level = ci.param->max_level_;
  const double scale = ci.param->GetScale(level);

  CiLiftHandler<word> lift(ci.context, big.context);

  std::mt19937_64 gen(0xD0DEC1);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> coeffs(n);
  for (auto &c : coeffs) c = dist(gen);

  Plaintext<word> pt;
  ci.context->encoder_.EncodeCoeff(pt, level, scale, coeffs);
  Ciphertext<word> ct;
  ci.ui->Encrypt(ct, pt);

  Ciphertext<word> lifted, back;
  lift.Lift(lifted, ct);
  lift.Descend(back, lifted);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_NEAR(back.GetScale() / ct.GetScale(), 2.0, 1e-9)
      << "the trace doubles the scale";

  Plaintext<word> out;
  ci.ui->Decrypt(out, back);
  std::vector<double> got;
  ci.context->encoder_.DecodeCoeff(got, out);
  ASSERT_EQ(static_cast<int>(got.size()), n);

  double worst = 0.0;
  for (int j = 0; j < n; j++) {
    worst = std::max(worst, std::abs(got[j] - coeffs[j]));
  }
  std::cout << "lift -> descend round trip: max |diff| = " << worst
            << std::endl;
  ASSERT_LT(worst, 1e-3);
}

// The case the round trip cannot see: a natively encrypted big-ring
// ciphertext has general components, not mirror-antisymmetric ones, and the
// halved variant of the descent silently destroyed exactly these. The even
// part of the message is the host reference.
TEST(CiLift, DescendReadsTheEvenPart) {
  Ring ci(kCiParam);
  Ring big(kBigParam,
           CiLiftHandler<word>::LiftSecret(ci.ui->GetSecretCoeffs()));

  const int n = ci.Degree();
  const int N = big.Degree();
  const int level = big.param->max_level_;
  const double scale = big.param->GetScale(level);

  CiLiftHandler<word> lift(ci.context, big.context);

  std::mt19937_64 gen(0xF00D);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> mcoeffs(N);
  for (auto &c : mcoeffs) c = dist(gen);

  Plaintext<word> pt;
  big.context->encoder_.EncodeCoeff(pt, level, scale, mcoeffs);
  Ciphertext<word> ct, down;
  big.ui->Encrypt(ct, pt);
  lift.Descend(down, ct);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  Plaintext<word> out;
  ci.ui->Decrypt(out, down);
  std::vector<double> got;
  ci.context->encoder_.DecodeCoeff(got, out);
  ASSERT_EQ(static_cast<int>(got.size()), n);

  double worst = 0.0;
  for (int j = 0; j < n; j++) {
    const double want =
        (j == 0) ? mcoeffs[0] : 0.5 * (mcoeffs[j] - mcoeffs[N - j]);
    worst = std::max(worst, std::abs(got[j] - want));
  }
  std::cout << "descend of a native ciphertext: |got - even part| max "
            << worst << std::endl;
  ASSERT_LT(worst, 1e-3);
}

TEST(CiLift, AProductOnTheLiftedRingDescendsToTheCiProduct) {
  Ring ci(kCiParam);
  Ring big(kBigParam,
           CiLiftHandler<word>::LiftSecret(ci.ui->GetSecretCoeffs()));

  const int level = ci.param->max_level_;
  const double scale = ci.param->GetScale(level);
  const int num_slots = ci.param->MaxNumSlots();

  CiLiftHandler<word> lift(ci.context, big.context);

  std::mt19937_64 gen(0x9B0D);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<Complex> z(num_slots), w(num_slots);
  for (auto &v : z) v = Complex(dist(gen), 0.0);
  for (auto &v : w) v = Complex(dist(gen), 0.0);

  Plaintext<word> pz, pw;
  ci.context->encoder_.Encode(pz, level, scale, z);
  ci.context->encoder_.Encode(pw, level, scale, w);
  Ciphertext<word> cz, cw;
  ci.ui->Encrypt(cz, pz);
  ci.ui->Encrypt(cw, pw);

  Ciphertext<word> lz, lw, prod, down;
  lift.Lift(lz, cz);
  lift.Lift(lw, cw);
  big.context->HMult(prod, lz, lw, big.ui->GetMultiplicationKey(), true);
  lift.Descend(down, prod);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // The halfway point: the product decrypted on the big ring itself must
  // already carry the slotwise products (as a multiset -- the slot orders
  // differ), or the descent would be blamed for an HMult failure.
  {
    Plaintext<word> mid;
    big.ui->Decrypt(mid, prod);
    std::vector<Complex> mid_slots;
    big.context->encoder_.Decode(mid_slots, mid);
    std::vector<double> a(mid_slots.size()), b(num_slots);
    double mid_imag = 0.0;
    for (size_t s = 0; s < mid_slots.size(); s++) {
      mid_imag = std::max(mid_imag, std::abs(mid_slots[s].imag()));
      a[s] = mid_slots[s].real();
    }
    for (int s = 0; s < num_slots; s++) b[s] = z[s].real() * w[s].real();
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    double mid_worst = 0.0;
    for (int s = 0; s < num_slots; s++) {
      mid_worst = std::max(mid_worst, std::abs(a[s] - b[s]));
    }
    std::cout << "midpoint on the big ring: max imag " << mid_imag
              << ", sorted multisets max |diff| = " << mid_worst << std::endl;
    EXPECT_LT(mid_worst, 1e-4);
  }

  Plaintext<word> out;
  ci.ui->Decrypt(out, down);
  std::vector<Complex> got;
  ci.context->encoder_.Decode(got, out);
  ASSERT_EQ(static_cast<int>(got.size()), num_slots);

  double worst = 0.0;
  for (int s = 0; s < num_slots; s++) {
    worst = std::max(worst,
                     std::abs(got[s].real() - z[s].real() * w[s].real()));
    worst = std::max(worst, std::abs(got[s].imag()));
  }
  std::cout << "lift -> HMult(8192) -> descend: max error " << worst
            << std::endl;
  ASSERT_LT(worst, 1e-3);
}

// ---------------------------------------------------------------------------
// The batch CC-MM on the lifted ring (Doing.md 1.5bl). Helpers shared by the
// two tests below; k is the CI sub-degree, the lifted ring runs [KANG]
// Algorithm 4 at 2k, and d = n/k = N/2k is the block and bundle size on both
// rings at once.
// ---------------------------------------------------------------------------

namespace {

constexpr int kCiSubDegree = 128;

// [lane][row][col] at the CI level, k lanes of d x d.
using RealBatch = std::vector<std::vector<std::vector<double>>>;

RealBatch SampleBatch(int lanes, int d, int live_rows, int live_cols,
                      double bound, std::mt19937_64 &gen) {
  std::uniform_real_distribution<double> dist(-bound, bound);
  RealBatch m(lanes,
              std::vector<std::vector<double>>(d, std::vector<double>(d, 0.0)));
  for (int t = 0; t < lanes; t++) {
    for (int i = 0; i < live_rows; i++) {
      for (int x = 0; x < live_cols; x++) m[t][i][x] = dist(gen);
    }
  }
  return m;
}

// CI MatEcd composed with the lift: ciphertext x carries column x -- block i,
// lane t of its SinC message holds m[t][i][x] -- encrypted on R+, lifted.
void EncryptLiftedColumns(ringfixture::Ring<word> &ci,
                          const CiLiftHandler<word> &lift, const RealBatch &m,
                          int level, double scale,
                          std::vector<Ciphertext<word>> &out) {
  const int k = kCiSubDegree;
  const int d = static_cast<int>(m[0].size());
  const int degree = ci.Degree();
  out.resize(d);
  std::vector<Complex> message(degree);
  Plaintext<word> pt;
  Ciphertext<word> ct;
  for (int x = 0; x < d; x++) {
    for (int i = 0; i < d; i++) {
      for (int t = 0; t < k; t++) {
        message[static_cast<size_t>(i) * k + t] = Complex(m[t][i][x], 0.0);
      }
    }
    ci.context->encoder_.EncodeSinC(pt, level, scale, message, k);
    ci.ui->Encrypt(ct, pt);
    lift.Lift(out[x], ct);
  }
}

void DescendAndDecode(ringfixture::Ring<word> &ci,
                      const CiLiftHandler<word> &lift,
                      const std::vector<Ciphertext<word>> &res,
                      std::vector<std::vector<Complex>> &got) {
  got.resize(res.size());
  Ciphertext<word> down;
  Plaintext<word> out;
  for (size_t j = 0; j < res.size(); j++) {
    lift.Descend(down, res[j]);
    ci.ui->Decrypt(out, down);
    ci.context->encoder_.DecodeSinC(got[j], out, kCiSubDegree);
  }
}

}  // namespace

// The clean contract: lhs ciphertexts confined to x < d/2, rhs data confined
// to CI blocks x < d/2, and Algorithm 4 -- unmodified, at sub-degree 2k --
// descends to the exact per-lane products.
TEST(CiLift, ABatchCcmmOnTheLiftedRingComputesTheCiProducts) {
  Ring ci(kCiParam);
  Ring big(kBigParam,
           CiLiftHandler<word>::LiftSecret(ci.ui->GetSecretCoeffs()));

  const int n = ci.Degree();
  const int N = big.Degree();
  const int k = kCiSubDegree;
  const int d = n / k;
  const int c = d / 2;
  const int level = ci.param->max_level_;
  const double scale = ci.param->GetScale(level);
  ASSERT_EQ(level, 1) << "Algorithm 4 needs exactly the one level this "
                         "chain has above its floor";

  CiLiftHandler<word> lift(ci.context, big.context);
  BatchCcmmHandler<word> ccmm(*big.param, big.context->ntt_handler_);
  for (int index : ccmm.RotationIndices(2 * k)) {
    big.ui->PrepareRotationKey(index, level);
  }

  std::mt19937_64 gen(0xCC33);
  const RealBatch a = SampleBatch(k, d, d, c, 0.15, gen);
  const RealBatch b = SampleBatch(k, d, c, d, 0.15, gen);

  std::vector<Ciphertext<word>> lhs, rhs, res;
  EncryptLiftedColumns(ci, lift, a, level, scale, lhs);
  EncryptLiftedColumns(ci, lift, b, level, scale, rhs);
  ccmm.Multiply(big.context, res, lhs, rhs, 2 * k, big.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), d);
  for (const auto &r : res) {
    ASSERT_EQ(big.param->NPToLevel(r.GetNP()), level - 1)
        << "Algorithm 4 spends exactly one level";
    EXPECT_NEAR(r.GetScale() / big.param->GetScale(level - 1), 1.0, 1e-6);
  }

  // Under the contract the product is even before the descent: its big-ring
  // coefficients must already be mirror-antisymmetric, so the trace loses
  // nothing. This is the sigma-free midpoint probe.
  {
    Plaintext<word> mid;
    big.ui->Decrypt(mid, res[0]);
    std::vector<double> mc;
    big.context->encoder_.DecodeCoeff(mc, mid);
    double worst_odd = std::abs(mc[n]);
    for (int j = 1; j < n; j++) {
      worst_odd = std::max(worst_odd, std::abs(mc[j] + mc[N - j]));
    }
    std::cout << "midpoint evenness on the big ring: max |m_j + m_(N-j)| = "
              << worst_odd << std::endl;
    EXPECT_LT(worst_odd, 1e-2);
  }

  std::vector<std::vector<Complex>> got;
  DescendAndDecode(ci, lift, res, got);

  double worst = 0.0;
  double worst_imag = 0.0;
  for (int j = 0; j < d; j++) {
    ASSERT_EQ(static_cast<int>(got[j].size()), n);
    for (int i = 0; i < d; i++) {
      for (int t = 0; t < k; t++) {
        double want = 0.0;
        for (int x = 0; x < d; x++) want += a[t][i][x] * b[t][x][j];
        const Complex &g = got[j][static_cast<size_t>(i) * k + t];
        worst = std::max(worst, std::abs(g.real() - want));
        worst_imag = std::max(worst_imag, std::abs(g.imag()));
      }
    }
  }
  std::cout << "lifted batch CCMM: " << k << " real lanes of (" << d << "x"
            << c << ")(" << c << "x" << d << "), max error " << worst
            << ", max imag " << worst_imag << std::endl;
  ASSERT_LT(worst, 1e-2);
  ASSERT_LT(worst_imag, 1e-2);
}

// Without the contract the naive read is wrong by design, and wrong in
// exactly the derived way: per lane the excess over A*B is a single scalar
// times the block-flipped product A*(P E B), and the scalars are the
// cosines of the primitive (4k)-th roots. Fitting the scalar per lane and
// comparing the multiset against the cosines pins the Lambda algebra with
// no reference to the slot correspondence between the two rings.
TEST(CiLift, TheFullContractionCarriesTheCosineTwistedFlip) {
  Ring ci(kCiParam);
  Ring big(kBigParam,
           CiLiftHandler<word>::LiftSecret(ci.ui->GetSecretCoeffs()));

  const int n = ci.Degree();
  const int k = kCiSubDegree;
  const int d = n / k;
  const int level = ci.param->max_level_;
  const double scale = ci.param->GetScale(level);

  CiLiftHandler<word> lift(ci.context, big.context);
  BatchCcmmHandler<word> ccmm(*big.param, big.context->ntt_handler_);
  for (int index : ccmm.RotationIndices(2 * k)) {
    big.ui->PrepareRotationKey(index, level);
  }

  std::mt19937_64 gen(0xC0517);
  const RealBatch a = SampleBatch(k, d, d, d, 0.15, gen);
  const RealBatch b = SampleBatch(k, d, d, d, 0.15, gen);

  std::vector<Ciphertext<word>> lhs, rhs, res;
  EncryptLiftedColumns(ci, lift, a, level, scale, lhs);
  EncryptLiftedColumns(ci, lift, b, level, scale, rhs);
  ccmm.Multiply(big.context, res, lhs, rhs, 2 * k, big.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<std::vector<Complex>> got;
  DescendAndDecode(ci, lift, res, got);

  double worst_naive = 0.0;
  double worst_resid = 0.0;
  double worst_imag = 0.0;
  std::vector<double> lambda(k);
  std::vector<std::vector<double>> excess(d, std::vector<double>(d));
  std::vector<std::vector<double>> flip(d, std::vector<double>(d));
  for (int t = 0; t < k; t++) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        double ab = 0.0;
        double fl = 0.0;
        for (int x = 0; x < d; x++) ab += a[t][i][x] * b[t][x][j];
        for (int x = 1; x < d; x++) fl += a[t][i][x] * b[t][d - x][j];
        const Complex &g = got[j][static_cast<size_t>(i) * k + t];
        excess[i][j] = g.real() - ab;
        flip[i][j] = fl;
        worst_naive = std::max(worst_naive, std::abs(excess[i][j]));
        worst_imag = std::max(worst_imag, std::abs(g.imag()));
        num += excess[i][j] * fl;
        den += fl * fl;
      }
    }
    lambda[t] = num / den;
    for (int i = 0; i < d; i++) {
      for (int j = 0; j < d; j++) {
        worst_resid = std::max(
            worst_resid, std::abs(excess[i][j] - lambda[t] * flip[i][j]));
      }
    }
  }

  std::sort(lambda.begin(), lambda.end());
  const double pi = std::acos(-1.0);
  std::vector<double> want_cos(k);
  for (int j = 0; j < k; j++) {
    want_cos[j] = std::cos(pi * (2 * j + 1) / (2 * k));
  }
  std::sort(want_cos.begin(), want_cos.end());
  double worst_cos = 0.0;
  for (int t = 0; t < k; t++) {
    worst_cos = std::max(worst_cos, std::abs(lambda[t] - want_cos[t]));
  }

  std::cout << "full contraction: naive |got - AB| max " << worst_naive
            << ", rank-one residual max " << worst_resid
            << ", |sorted lambda - sorted cos| max " << worst_cos
            << ", max imag " << worst_imag << std::endl;
  ASSERT_GT(worst_naive, 0.05) << "the twist should be O(1), not noise";
  ASSERT_LT(worst_resid, 1e-2)
      << "the excess must be exactly lambda * A(PE B) per lane";
  ASSERT_LT(worst_cos, 5e-3)
      << "the lambdas must be the primitive (4k)-th root cosines";
  ASSERT_LT(worst_imag, 1e-2);
}
