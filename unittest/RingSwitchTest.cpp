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
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#include "core/RingSwitch.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::EvaluationKey;
using cheddar::MlweCiphertext;
using cheddar::MlweHandler;
using cheddar::PcmmHandler;
using cheddar::PlainMatrix;
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

// On the conjugate-invariant pair (`ci_ringswitch16_35` / `ci12_35`) the
// switched components are not stride slices: the subring secret still
// collapses the module rank to one, but the components come out of the
// alternating-sign suffix-sum scan down each class pair (i, k-i) -- the same
// map ModDecomp uses there (Doing.md 1.5ba). The host expectation below
// computes it in exact real arithmetic; the ordinary ring is the stride.
std::vector<std::vector<double>> HostComponents(
    const std::vector<double> &coeffs, int rank, int small_degree,
    bool conjugate_invariant) {
  std::vector<std::vector<double>> comp(rank,
                                        std::vector<double>(small_degree));
  if (!conjugate_invariant) {
    for (int i = 0; i < rank; i++) {
      for (int t = 0; t < small_degree; t++) {
        comp[i][t] = coeffs[t * rank + i];
      }
    }
    return comp;
  }
  for (int t = 0; t < small_degree; t++) comp[0][t] = coeffs[t * rank];
  for (int i = 1; i <= rank / 2; i++) {
    const int mi = rank - i;
    double acc_i = 0.0;
    double acc_m = 0.0;
    for (int t = small_degree - 1; t >= 0; t--) {
      const double new_i = coeffs[t * rank + i] - acc_m;
      const double new_m = coeffs[t * rank + mi] - acc_i;
      comp[i][t] = new_i;
      comp[mi][t] = new_m;
      acc_i = new_i;
      acc_m = new_m;
    }
  }
  return comp;
}

// The inverse of HostComponents: the banded two-term recomposition on the
// conjugate-invariant ring, the plain interleave on the ordinary one -- what
// ModPack and SwitchBack both implement over their own containers.
std::vector<double> HostRecompose(const std::vector<std::vector<double>> &comp,
                                  int rank, int small_degree,
                                  bool conjugate_invariant) {
  std::vector<double> out(static_cast<size_t>(rank) * small_degree, 0.0);
  for (int t = 0; t < small_degree; t++) {
    for (int i = 0; i < rank; i++) {
      double v = comp[i][t];
      if (conjugate_invariant && i != 0 && t + 1 < small_degree) {
        v += comp[rank - i][t + 1];
      }
      out[static_cast<size_t>(t) * rank + i] = v;
    }
  }
  return out;
}

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

TEST(RingSwitch, SwitchesAndPreservesTheComponents) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);

  const int degree = big.Degree();
  const int small_degree = small.Degree();
  const int rank = degree / small_degree;
  ASSERT_EQ(rank, 16);
  const bool ci = big.param->conjugate_invariant_;

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

  const auto expected = HostComponents(coeffs, rank, small_degree, ci);

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
      const double d = std::abs(got[s] - expected[i][s]);
      if (d > worst) {
        worst = d;
        worst_i = i;
        worst_s = s;
      }
    }
  }

  std::cout << (ci ? "CI " : "ordinary ") << "ring switch " << degree
            << " -> " << rank << " x " << small_degree
            << ": max |diff| = " << worst << " (i=" << worst_i
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

// ---------------------------------------------------------------------------
// The switched descent at the layer's own shape (Doing.md 1.5bh): switch to
// the small ring, ModDecomp every part to the projection degree of 128, mix
// all 512 module components with one 512 x 512 plaintext matrix -- the shape
// of one weight tile -- then ModPack each part with the SMALL ring's 32
// embedded-secret keys and switch back up. Against the same composition of
// maps in exact real arithmetic on the host.
//
// This is the whole switched leg of the PC-MM: every key the route needs
// exists here (one switch key, one inverse key, and 32 small-ring ModPack
// keys serving all 16 parts -- against the direct route's 512 keys at the
// big degree), and the reported error is the route's noise datum at the real
// shape, the other half of the measurement the descent choice of 1.5bg/1.5bh
// waits on. On the conjugate-invariant pair both the switch and the
// decomposition are scans, and the mixing destroys the coherence that lets a
// pure round trip contract them back out -- which is exactly why this is a
// measurement and not an estimate.
TEST(RingSwitch, DescendsToTheProjectionShapeAndReturns) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);

  const int degree = big.Degree();
  const int small_ring_degree = small.Degree();
  const int rank = degree / small_ring_degree;
  const int proj_degree = 128;
  const int mlwe_rank = small_ring_degree / proj_degree;
  const int cols = rank * mlwe_rank;
  const bool ci = big.param->conjugate_invariant_;
  constexpr double kWeightScale = 268435456.0;  // 2^28

  const int level = big.param->max_level_;
  const auto &small_secret = small.ui->GetSecretCoeffs();
  big.ui->PrepareRingSwitchKey(small_ring_degree, small_secret, level);
  big.ui->PrepareInverseRingSwitchKey(small_ring_degree, small_secret, level);
  small.ui->PrepareModPackKeys(proj_degree);
  std::vector<const EvaluationKey<word> *> keys;
  for (int j = 0; j < mlwe_rank; j++) {
    keys.push_back(&small.ui->GetModPackKey(mlwe_rank, j));
  }

  // On the conjugate-invariant pair the components grow twice over -- the
  // switch's scan of chain 4096 and the decomposition's of chain 128 -- and
  // the product's scale has to fit what remains of a terminal-prime modulus,
  // so the inputs start at a quarter of the usual range.
  std::mt19937_64 gen(0x5CA1AB1E);
  std::uniform_real_distribution<double> dist(-0.25, 0.25);
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
  ASSERT_EQ(static_cast<int>(parts.size()), rank);

  // Decompose every part; column j of the product is module component
  // j % mlwe_rank of part j / mlwe_rank.
  MlweHandler<word> mlwe(*small.param, small.context->ntt_handler_);
  std::vector<MlweCiphertext<word>> columns;
  columns.reserve(cols);
  for (int p = 0; p < rank; p++) {
    std::vector<MlweCiphertext<word>> decomposed;
    mlwe.ModDecomp(decomposed, parts[p], proj_degree);
    ASSERT_EQ(static_cast<int>(decomposed.size()), mlwe_rank);
    for (auto &c : decomposed) columns.push_back(std::move(c));
  }

  // One tile-shaped mixing matrix across all parts, normalized by
  // 1/sqrt(cols) so the mixed magnitude stays at the components' own.
  std::vector<double> values(static_cast<size_t>(cols) * cols);
  {
    std::uniform_real_distribution<double> wdist(-1.0, 1.0);
    const double norm = 1.0 / std::sqrt(static_cast<double>(cols));
    for (auto &v : values) v = wdist(gen) * norm;
  }

  const int part_level = small.param->NPToLevel(columns[0].np_);
  ASSERT_GE(part_level, 0);
  PcmmHandler<word> pcmm(*small.param);
  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, part_level, kWeightScale, values, cols, cols);

  std::vector<MlweCiphertext<word>> mixed;
  pcmm.Multiply(mixed, u, columns);
  ASSERT_EQ(static_cast<int>(mixed.size()), cols);

  // Pack rows [mlwe_rank*p, mlwe_rank*(p+1)) back into small-ring ciphertext
  // p -- the same 32 keys serve every part, which is the switched route's
  // key-count argument.
  std::vector<Ciphertext<word>> packed(rank);
  for (int p = 0; p < rank; p++) {
    std::vector<MlweCiphertext<word>> group;
    group.reserve(mlwe_rank);
    for (int j = 0; j < mlwe_rank; j++) {
      group.push_back(std::move(mixed[static_cast<size_t>(p) * mlwe_rank + j]));
    }
    mlwe.ModPack(small.context, packed[p], group, keys);
  }

  Ciphertext<word> back;
  rs.SwitchBack(back, packed, big.ui->GetInverseRingSwitchKey(rank));
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  Plaintext<word> back_pt;
  big.ui->Decrypt(back_pt, back);
  std::vector<double> got;
  big.context->encoder_.DecodeCoeff(got, back_pt);
  ASSERT_EQ(static_cast<int>(got.size()), degree);

  // The same composition on the host, in exact real arithmetic.
  const auto comp16 = HostComponents(coeffs, rank, small_ring_degree, ci);
  std::vector<std::vector<double>> columns_host;
  columns_host.reserve(cols);
  for (int p = 0; p < rank; p++) {
    auto sub = HostComponents(comp16[p], mlwe_rank, proj_degree, ci);
    for (auto &s : sub) columns_host.push_back(std::move(s));
  }
  std::vector<std::vector<double>> parts_host(rank);
  for (int p = 0; p < rank; p++) {
    std::vector<std::vector<double>> mixed_host(
        mlwe_rank, std::vector<double>(proj_degree, 0.0));
    for (int l = 0; l < mlwe_rank; l++) {
      const double *row =
          values.data() + (static_cast<size_t>(p) * mlwe_rank + l) * cols;
      for (int j = 0; j < cols; j++) {
        const double w = row[j];
        for (int t = 0; t < proj_degree; t++) {
          mixed_host[l][t] += w * columns_host[j][t];
        }
      }
    }
    parts_host[p] = HostRecompose(mixed_host, mlwe_rank, proj_degree, ci);
  }
  const auto expected = HostRecompose(parts_host, rank, small_ring_degree, ci);

  double worst = 0.0;
  int worst_c = -1;
  for (int c = 0; c < degree; c++) {
    const double d = std::abs(got[c] - expected[c]);
    if (d > worst) {
      worst = d;
      worst_c = c;
    }
  }
  std::cout << (ci ? "CI " : "ordinary ") << "switched descent " << degree
            << " -> " << rank << " x " << small_ring_degree << " -> " << cols
            << " x " << proj_degree << " -> back: max |diff| = " << worst
            << " (coefficient " << worst_c << ")" << std::endl;
  EXPECT_LT(worst, 2e-3);
}

// ---------------------------------------------------------------------------
// [SYLPH] section 3.3: ring switching works on SinC-encoded ciphertexts too.
//
// THIS IS THE JOINT THE ATTENTION PATH TURNS ON, and nothing in this repo has
// ever checked it. [SYLPH] Table 4 puts the batch CC-MM at ring degree 4096 in
// the SinC encoding, and the only way a tensor that lives at 65536 gets there
// is: partial SlotToCoeff (BootContext::SlotToSinC, measured in
// sinc_transform_test) and then this switch. The two halves are each verified
// on their own; this is the composition.
//
// THE CLAIM, STATED AS AN INDEX IDENTITY. Ecd_SinC_{k,N} puts block i at the
// coefficients congruent to i mod d, d = N/k. The switch of rank r sends
// coefficient p to position p/r of ciphertext p mod r. So with the SAME k on
// both sides -- d' = N'/k = d/r -- big block i splits as
//
//     i = j + r * i'      j = which small ciphertext, i' = its own block
//
// and the lane index is untouched. That is [SYLPH]'s "k' | k" condition at
// k' = k, and it is what makes the switch free of any re-encoding.
//
// WHAT WOULD SLIP THROUGH A NORM CHECK. Every value survives a switch that
// scrambled the blocks, so the comparison is entrywise against the exact
// prediction above, and a control that omits the `j + r * i'` interleave has
// to fail.
//
// ON THE CONJUGATE-INVARIANT PAIR the index identity does not survive: the
// switch is the scan, and at the block level a part's blocks are a
// triangular mix of the big blocks with seam terms that cross into the
// neighbouring subring position -- not a redistribution. The comparison
// there is against the exact composition of the validated component map
// with the small ring's own DecodeSinC, and the ordinary interleave becomes
// the control that has to FAIL.
TEST(RingSwitch, CarriesTheSinCEncodingToTheSmallRing) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);

  const bool ci = big.param->conjugate_invariant_;
  const int degree = big.Degree();
  const int small_degree = small.Degree();
  const int rank = degree / small_degree;
  const int level = big.param->max_level_;

  big.ui->PrepareRingSwitchKey(small_degree, small.ui->GetSecretCoeffs(),
                               level);
  RingSwitchHandler<word> rs(big.context, small.context);

  // k = 32 makes the small ring's matrix 4096/32 = 128 wide, which is exactly
  // Llama-3's per-head 128x128 attention product with 16 lanes for 16 heads.
  // k = 128 is [SYLPH]'s own instance: 32 blocks, 64 lanes.
  for (int k : {32, 128}) {
    const int d_big = degree / k;
    const int d_small = small_degree / k;
    const int lanes = ci ? k : k / 2;
    ASSERT_EQ(d_big, rank * d_small);

    std::mt19937_64 gen(0x51C0DEULL ^ static_cast<unsigned long long>(k));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<cheddar::Complex> msg(ci ? degree : degree / 2);
    for (auto &z : msg) {
      z = ci ? cheddar::Complex(dist(gen), 0.0)
             : cheddar::Complex(dist(gen), dist(gen));
    }

    Plaintext<word> pt;
    big.context->encoder_.EncodeSinC(pt, level, big.param->GetScale(level),
                                     msg, k);
    Ciphertext<word> ct;
    big.ui->Encrypt(ct, pt);

    std::vector<Ciphertext<word>> parts;
    rs.Switch(parts, ct, big.ui->GetRingSwitchKey(rank));
    ASSERT_EQ(static_cast<int>(parts.size()), rank);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    double worst = 0.0, control = 0.0;

    // On the conjugate-invariant pair there is no index formula to check:
    // the parts carry the block-level SCAN of the big blocks, seam terms
    // included, not a redistribution. Each part is still a well-formed SinC
    // ciphertext, and its exact message is predicted by composing the
    // validated component map with the small ring's own DecodeSinC. The
    // ordinary interleave is kept as the control that has to fail there.
    std::vector<std::vector<double>> comp;
    if (ci) {
      std::vector<double> big_coeffs;
      big.context->encoder_.DecodeCoeff(big_coeffs, pt);
      comp = HostComponents(big_coeffs, rank, small_degree, true);
    }

    for (int j = 0; j < rank; j++) {
      Plaintext<word> back;
      small.ui->Decrypt(back, parts[j]);
      std::vector<cheddar::Complex> got;
      small.context->encoder_.DecodeSinC(got, back, k);
      ASSERT_EQ(static_cast<int>(got.size()),
                ci ? small_degree : small_degree / 2);

      if (ci) {
        Plaintext<word> pred_pt;
        small.context->encoder_.EncodeCoeff(pred_pt, small.param->max_level_,
                                            big.param->GetScale(level),
                                            comp[j]);
        std::vector<cheddar::Complex> want;
        small.context->encoder_.DecodeSinC(want, pred_pt, k);

        for (int ip = 0; ip < d_small; ip++) {
          for (int t = 0; t < lanes; t++) {
            const auto v = got[ip * lanes + t];
            worst = std::max(worst, std::abs(v - want[ip * lanes + t]));
            control = std::max(
                control, std::abs(v - msg[(j + rank * ip) * lanes + t]));
          }
        }
        continue;
      }

      for (int ip = 0; ip < d_small; ip++) {
        for (int t = 0; t < lanes; t++) {
          const auto v = got[ip * lanes + t];
          worst = std::max(
              worst, std::abs(v - msg[(j + rank * ip) * lanes + t]));
          // The same index without the interleave: blocks laid out
          // contiguously per ciphertext instead of strided.
          control = std::max(
              control,
              std::abs(v - msg[(j * d_small + ip) * lanes + t]));
        }
      }
    }

    std::cout << (ci ? "CI " : "ordinary ") << "SinC k=" << k << ": "
              << degree << " (" << d_big << " blocks) -> " << rank << " x "
              << small_degree << " (" << d_small << " blocks, " << lanes
              << " lanes): max |diff| = " << worst << ", control ("
              << (ci ? "ordinary interleave" : "no interleave")
              << ") = " << control << std::endl;

    EXPECT_LT(worst, ci ? 5e-3 : 2e-3)
        << "the switch did not carry SinC(" << k << ")";
    EXPECT_GT(control, 1e-2)
        << (ci ? "the parts read as an ordinary block interleave, which the "
                 "banded module structure says cannot happen"
               : "the blocks are NOT interleaved, so this test would pass "
                 "against a switch that laid them out contiguously");
  }
}
