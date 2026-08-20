// Ring switching, degree 65536 -> sixteen ciphertexts of degree 4096.
//
// [BAE] appendix A:
//
//     1. (a~, b~) <- KeySwitch( (a,b), swk_{sk -> sk'} )   sk' in the subring
//     2. (a_i, b_i) = ( e*_i(a~), e*_i(b~) )               0 <= i < k
//
// The check is end to end and needs no host reference for the crypto: encrypt
// a known coefficient vector in the big ring, switch, and decrypt each output
// with the *small* ring's own secret. If the construction is right, output i
// decodes to the stride-k slice m[i], m[i+k], m[i+2k], ... and nothing else
// lines up by accident -- a wrong stride, a wrong secret embedding, or a
// missed Montgomery conversion all produce noise rather than a shifted answer.
//
// WHY THREE PARAMETER FILES EXIST. The switching key is published at modulus
// P*Q and an attacker can apply e*_i to it, reading RLWE samples at degree
// 4096, so log2(PQ) must fit *that* degree's budget (~104 bits). The shipped
// presets carry alpha = 12, putting P alone at 2^372. ringswitch16_30.json is
// therefore the same ring degree and the same primes as bootparam_30's level
// 1, but with one auxiliary prime instead of twelve -- log2 PQ = 100.75, which
// the S1 estimator scores at 132.8 gate bits for degree 4096.
//
// The existing short base was the obvious alternative and it fails for a
// different reason: it is secure (log2 PQ = 79.5) but Context routes it only
// at level 0, where Q sits about ten bits above the scale. After the message
// and the ciphertext scale roughly five bits remain for a plaintext matrix,
// and a typical Llama-3 weight of 0.0175 rounds to nothing there.

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/RingSwitch.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::Plaintext;
using cheddar::RingSwitchHandler;
using Ring = ringfixture::Ring<word>;

namespace {

// The pair is selected by environment so the same suite can be run against
// either scale. `ringswitch16_30`/`ringdegree12_30` is the pair this file was
// written for; `ringswitch16_35`/`ringdegree12_35` is the 2^35 pair, whose
// terminal primes sit near 2^25 rather than 2^30 so that log2(PQ) clears the
// degree-4096 budget -- which is exactly what bootparam_35's own bottom levels
// could not do.
const char *ParamFromEnv(const char *var, const char *fallback) {
  const char *env = std::getenv(var);
  return (env != nullptr && env[0] != 0) ? env : fallback;
}
const char *kSwitchParam = ParamFromEnv("CHEDDAR_SWITCH_PARAM",
                                        "ringswitch16_30.json");
const char *kSmallParam = ParamFromEnv("CHEDDAR_SMALL_PARAM",
                                       "ringdegree12_30.json");

}  // namespace

// The two parameter files have to describe the same modulus chain, or step 2
// would hand the small ring limbs it has no primes for. The handler asserts
// this, but stating it here makes a bad edit to either JSON fail with a
// readable message rather than inside a constructor.
TEST(RingSwitch, ParameterSetsShareTheirPrimes) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);

  ASSERT_EQ(big.Degree(), 65536);
  ASSERT_EQ(small.Degree(), 4096);
  EXPECT_EQ(big.param->max_level_, small.param->max_level_);
  EXPECT_EQ(big.param->alpha_, 1);
  EXPECT_EQ(small.param->alpha_, 1);

  for (int level = 0; level <= big.param->max_level_; level++) {
    const auto bp = big.param->GetPrimeVector(big.param->LevelToNP(level));
    const auto sp = small.param->GetPrimeVector(small.param->LevelToNP(level));
    EXPECT_EQ(bp, sp) << "primes differ at level " << level;
  }
  std::cout << "both rings carry " << big.param->L_ << " q primes and alpha "
            << big.param->alpha_ << std::endl;
}

TEST(RingSwitch, SwitchesAndPreservesTheStridedSlices) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);

  const int degree = big.Degree();
  const int small_degree = small.Degree();
  const int rank = degree / small_degree;
  ASSERT_EQ(rank, 16);

  // The key is built in the big ring but targets the small ring's own secret,
  // which only the small ring's UserInterface knows.
  const int level = big.param->max_level_;
  big.ui->PrepareRingSwitchKey(small_degree, small.ui->GetSecretCoeffs(),
                               level);
  std::cout << "ring-switch key prepared for rank " << rank << " at level "
            << level << std::endl;

  std::mt19937_64 gen(0x515C0DE);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> coeffs(degree);
  for (auto &c : coeffs) c = dist(gen);

  Plaintext<word> pt;
  big.context->encoder_.EncodeCoeff(pt, level, big.param->GetScale(level),
                                    coeffs);
  Ciphertext<word> ct;
  big.ui->Encrypt(ct, pt);

  RingSwitchHandler<word> rs(big.context, small.context);
  ASSERT_EQ(rs.GetRank(), rank);

  std::vector<Ciphertext<word>> parts;
  rs.Switch(parts, ct, big.ui->GetRingSwitchKey(rank));
  ASSERT_EQ(static_cast<int>(parts.size()), rank);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  double worst = 0.0;
  int worst_i = -1, worst_s = -1;
  for (int i = 0; i < rank; i++) {
    // Decrypted with the SMALL ring's secret. That is the whole claim: after
    // the switch these ciphertexts belong to the other ring outright.
    Plaintext<word> back;
    small.ui->Decrypt(back, parts[i]);
    std::vector<double> got;
    small.context->encoder_.DecodeCoeff(got, back);
    ASSERT_EQ(static_cast<int>(got.size()), small_degree);

    for (int s = 0; s < small_degree; s++) {
      const double d = std::abs(got[s] - coeffs[i + rank * s]);
      if (d > worst) {
        worst = d;
        worst_i = i;
        worst_s = s;
      }
    }
  }

  std::cout << "ring switch " << degree << " -> " << rank << " x "
            << small_degree << ": max |diff| = " << worst << " (i=" << worst_i
            << ", s=" << worst_s << ")" << std::endl;

  // A key switch adds noise on top of the encryption noise, and this one uses
  // dnum = 3, so the bar is looser than a plain round trip -- but orders of
  // magnitude tighter than any indexing or convention error would leave it.
  EXPECT_LT(worst, 1e-3);
}

// Down and back up. This is what a Llama block actually needs: without the
// return trip the pipeline can descend, multiply, and then has nowhere to go.
TEST(RingSwitch, RoundTripsThroughTheSmallRing) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);

  const int degree = big.Degree();
  const int small_degree = small.Degree();
  const int rank = degree / small_degree;
  const int level = big.param->max_level_;

  const auto &small_secret = small.ui->GetSecretCoeffs();
  big.ui->PrepareRingSwitchKey(small_degree, small_secret, level);
  big.ui->PrepareInverseRingSwitchKey(small_degree, small_secret, level);

  std::mt19937_64 gen(0xD0DEC0DE);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> coeffs(degree);
  for (auto &c : coeffs) c = dist(gen);

  Plaintext<word> pt;
  big.context->encoder_.EncodeCoeff(pt, level, big.param->GetScale(level),
                                    coeffs);
  Ciphertext<word> ct;
  big.ui->Encrypt(ct, pt);

  RingSwitchHandler<word> rs(big.context, small.context);

  std::vector<Ciphertext<word>> parts;
  rs.Switch(parts, ct, big.ui->GetRingSwitchKey(rank));

  Ciphertext<word> back;
  rs.SwitchBack(back, parts, big.ui->GetInverseRingSwitchKey(rank));
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Back in the big ring, under the big ring's own secret again.
  Plaintext<word> back_pt;
  big.ui->Decrypt(back_pt, back);
  std::vector<double> got;
  big.context->encoder_.DecodeCoeff(got, back_pt);
  ASSERT_EQ(static_cast<int>(got.size()), degree);

  double worst = 0.0;
  int worst_c = -1;
  for (int c = 0; c < degree; c++) {
    const double d = std::abs(got[c] - coeffs[c]);
    if (d > worst) {
      worst = d;
      worst_c = c;
    }
  }
  std::cout << "round trip " << degree << " -> " << rank << " x "
            << small_degree << " -> " << degree << ": max |diff| = " << worst
            << " (coefficient " << worst_c << ")" << std::endl;

  // Two key switches now, so a little looser than the one-way bound.
  EXPECT_LT(worst, 2e-3);
}
