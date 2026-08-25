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

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/CiLift.h"

using word = uint32_t;
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
