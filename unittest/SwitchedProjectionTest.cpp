// [SYLPH] section 3.2's own descent for the PC-MM: ring-switch first, then
// decompose. The projection is the same projection either way, so this test
// runs it BOTH ways on the same ciphertexts and requires the two to agree with
// each other and with a host product.
//
//     direct      65536 --ModDecomp(rank 256)--------------------> 256
//     switched    65536 --RingSwitch(16)--> 4096 --ModDecomp(16)--> 256
//
// WHY THE TWO ROUTES ARE THE SAME PRODUCT. Both reach the same 256 module
// components of a parent -- the direct one by a single stride, the switched
// one by two -- and all 256 are MLWE ciphertexts under the *same* rank-16
// module secret of the product ring, so a scalar combination may mix them
// freely. What differs is only which component is called which:
//
//     direct     component index  =  c mod 256
//     switched   small ciphertext =  c mod 16,  component = (c / 16) mod 16
//
// and composing the second pair gives back the first. `CoeffLinearLeg::
// Component` is that one line, folded into the plaintext operand where it is
// free. **This test exists because that line is the entire difference**, it is
// an index identity rather than an approximation, and every way of getting it
// wrong -- swapping the two strides, enumerating the components before the
// small ciphertexts, reversing eight bits instead of two nibbles -- leaves a
// result with exactly the right magnitude that still decrypts cleanly.
//
// WHY IT IS WORTH THE SWITCH. The descent costs one key switch per parent at
// the block's degree. It saves, per output group, 256 ModPack key switches at
// degree 65536 in exchange for 256 at degree 4096 plus one inverse switch at
// 65536 -- about fifteen times less -- and it holds a sixteenth of the module
// components, which is what removes the reason `parents_per_tile` exists.
//
// THE RINGS. `ringswitch16_35` stands in for the block here as well as for the
// switching context: it is the block's ring degree with the block's primes and
// alpha 1, and in the pipeline the two share a secret and a ciphertext crosses
// between them without a word changing (`RingSwitch.h`). Using one object for
// both is that special case. `ringdegree12_35` is the product ring and has its
// own secret and its own ModPack keys.
//
// SCALE. Unchanged from `PipelineChainTest` and `LlamaProjectionTest`: the
// weight scale is `GetScale(1)` and nothing else lands the rescaled product on
// the ring's canonical level-0 scale. It is read here from the **product**
// ring's parameter, because that is the ring the operand is encoded against.

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/AttentionPacking.h"
#include "extension/LlamaLinear.h"

using word = uint32_t;
using cheddar::AttentionPacking;
using cheddar::Ciphertext;
using cheddar::EvaluationKey;
using cheddar::Plaintext;
using Ring = ringfixture::Ring<word>;

namespace {

constexpr int kTokens = 128;       // the only count the packing admits
constexpr int kInChannels = 1024;  // four parents, so the parent order shows
constexpr int kOutChannels = 512;  // two ModPack groups, so the reversals show
constexpr int kSmallDegree = 256;  // the PC-MM ring
constexpr int kLevel = 1;  // the product's input level; it rescales to 0

// The block's packing: slot s carries token s % T of channel s / T, and the
// conversion to coefficients is the bit reversal the homomorphic transform
// omits.
int CoeffOfTokenChannel(int token, int channel_in_ct, int num_slots) {
  const int slot = token + kTokens * channel_in_ct;
  return AttentionPacking::CoeffOfSlot({slot, false}, num_slots * 2);
}

// `CoeffLinearLeg` implements only `Project`; the two ciphertext-ciphertext
// products are pure virtual on purpose so nothing falls back to a stand-in.
class ProjectOnlyLeg : public cheddar::CoeffLinearLeg<word> {
 public:
  using cheddar::CoeffLinearLeg<word>::CoeffLinearLeg;

  void Scores(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double,
              const std::vector<double> &) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: Scores is not in this test");
  }
  void Values(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: Values is not in this test");
  }
  void LocateScore(int, int, int, int &, int &) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: there is no score layout here");
  }
};

// Decrypt a projection's output groups back into [token][out channel].
void ReadBack(std::vector<double> &out,
              const std::vector<Ciphertext<word>> &res, const Ring &ring,
              int rank, int num_slots) {
  out.assign(static_cast<size_t>(kTokens) * kOutChannels, 0.0);
  for (size_t g = 0; g < res.size(); g++) {
    Plaintext<word> pt;
    ring.ui->Decrypt(pt, res[g]);
    std::vector<double> coeffs;
    ring.context->encoder_.DecodeCoeff(coeffs, pt);
    for (int c = 0; c < rank; c++) {
      for (int t = 0; t < kTokens; t++) {
        out[static_cast<size_t>(t) * kOutChannels + g * rank + c] =
            coeffs[CoeffOfTokenChannel(t, c, num_slots)];
      }
    }
  }
}

}  // namespace

TEST(SwitchedProjection, TheDescentAgreesWithTheDirectRouteAndWithTheHost) {
  const char *big_param = std::getenv("CHEDDAR_SWITCH_PARAM");
  const char *small_param = std::getenv("CHEDDAR_SMALL_PARAM");
  Ring block(big_param && big_param[0] ? big_param : "ringswitch16_35.json");
  Ring small(small_param && small_param[0] ? small_param
                                           : "ringdegree12_35.json");

  const int degree = block.Degree();
  const int num_slots = degree / 2;
  const int mid_degree = small.Degree();
  const int rank = degree / kSmallDegree;          // 256, the channels per ct
  const int ring_rank = degree / mid_degree;       // 16, the ring-switch rank
  const int sub_rank = mid_degree / kSmallDegree;  // 16, the ModDecomp rank
  const int num_parents = kInChannels / rank;
  // Fixed here; only the conjugate-invariant test below takes a width knob.
  const int in_channels = kInChannels;

  ASSERT_EQ(rank, 256);
  ASSERT_EQ(ring_rank * sub_rank, rank)
      << "the two strides must compose to the rank the channel map is stated "
         "against";
  ASSERT_EQ(num_slots / kTokens, rank);
  ASSERT_EQ(num_parents, 4);
  ASSERT_GE(block.param->max_level_, kLevel);
  ASSERT_EQ(small.param->max_level_, kLevel)
      << "the product ring is supposed to have exactly one multiplicative "
         "level";

  const double ct_scale = block.param->GetScale(kLevel);
  const double u_scale = small.param->GetScale(kLevel);
  EXPECT_NEAR(u_scale / block.param->GetScale(kLevel), 1.0, 1e-9)
      << "the two rings share their primes at this level, so they must agree "
         "on the scale as well";

  // ---- the tensors, shaped like Llama-3-8B ------------------------------
  std::mt19937_64 gen(0x5117C4);
  std::normal_distribution<double> xd(0.0, 1.0);
  std::normal_distribution<double> wd(0.0, 0.0175);  // Llama-3's W_Q RMS
  std::vector<double> x(static_cast<size_t>(kTokens) * in_channels);
  std::vector<double> w(static_cast<size_t>(in_channels) * kOutChannels);
  for (auto &v : x) v = xd(gen);
  for (auto &v : w) v = wd(gen);
  // RMSNorm the activation, as the block does before this product.
  for (int t = 0; t < kTokens; t++) {
    double sq = 0.0;
    for (int c = 0; c < in_channels; c++) {
      const double v = x[static_cast<size_t>(t) * in_channels + c];
      sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq / in_channels + 1e-5);
    for (int c = 0; c < in_channels; c++) {
      x[static_cast<size_t>(t) * in_channels + c] *= inv;
    }
  }

  std::vector<double> want(static_cast<size_t>(kTokens) * kOutChannels, 0.0);
  double want_max = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int o = 0; o < kOutChannels; o++) {
      double acc = 0.0;
      for (int c = 0; c < in_channels; c++) {
        acc += x[static_cast<size_t>(t) * in_channels + c] *
               w[static_cast<size_t>(c) * kOutChannels + o];
      }
      want[static_cast<size_t>(t) * kOutChannels + o] = acc;
      want_max = std::max(want_max, std::abs(acc));
    }
  }

  // ---- keys -------------------------------------------------------------
  block.ui->PrepareRingSwitchKey(mid_degree, small.ui->GetSecretCoeffs(),
                                 kLevel);
  block.ui->PrepareInverseRingSwitchKey(mid_degree,
                                        small.ui->GetSecretCoeffs(), kLevel);
  small.ui->PrepareModPackKeys(kSmallDegree, kLevel);
  std::vector<const EvaluationKey<word> *> small_keys(sub_rank);
  for (int j = 0; j < sub_rank; j++) {
    small_keys[j] = &small.ui->GetModPackKey(sub_rank, j);
  }

  // ---- encrypt in the block's packing -----------------------------------
  std::vector<Ciphertext<word>> parents(num_parents);
  for (int p = 0; p < num_parents; p++) {
    std::vector<double> coeffs(degree, 0.0);
    for (int c = 0; c < rank; c++) {
      for (int t = 0; t < kTokens; t++) {
        coeffs[CoeffOfTokenChannel(t, c, num_slots)] =
            x[static_cast<size_t>(t) * kInChannels + p * rank + c];
      }
    }
    Plaintext<word> pt;
    block.context->encoder_.EncodeCoeff(pt, kLevel, ct_scale, coeffs);
    block.ui->Encrypt(parents[p], pt);
  }

  // ---- [SYLPH]'s route --------------------------------------------------
  cheddar::CoeffLinearLeg<word>::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.product_level = kLevel;
  cfg.parents_per_tile = 0;  // the descent holds a sixteenth; nothing to tile

  cheddar::CoeffLinearLeg<word>::Descent descent;
  descent.switch_context = block.context;
  descent.small_context = small.context;
  descent.forward = &block.ui->GetRingSwitchKey(ring_rank);
  descent.inverse = &block.ui->GetInverseRingSwitchKey(ring_rank);
  descent.modpack_keys = small_keys;

  std::vector<double> got_switched;
  {
    ProjectOnlyLeg leg(block.context, cfg, {}, descent);
    EXPECT_TRUE(leg.IsRingSwitched());
    EXPECT_EQ(leg.GetRingRank(), ring_rank);
    EXPECT_EQ(leg.GetSubRank(), sub_rank);

    std::vector<Ciphertext<word>> res;
    leg.Project(res, parents, in_channels, kOutChannels, w, 1.0, "switched");
    ASSERT_EQ(static_cast<int>(res.size()), kOutChannels / rank);
    EXPECT_EQ(block.param->NPToLevel(res[0].GetNP()), 0)
        << "the product spends exactly one level however it descends";
    EXPECT_NEAR(res[0].GetScale() / block.param->GetScale(0), 1.0, 1e-6)
        << "the ciphertext returns to the block ring off its canonical scale, "
           "which EvalPoly would later abort on";
    EXPECT_EQ(res[0].GetNumSlots(), num_slots)
        << "SwitchBack has to restore the block's slot count, or the next "
           "bootstrap will not take it";
    ReadBack(got_switched, res, block, rank, num_slots);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  double sw_max = 0.0, sw_mean = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    const double e = std::abs(got_switched[i] - want[i]);
    sw_max = std::max(sw_max, e);
    sw_mean += e;
  }
  sw_mean /= static_cast<double>(want.size());
  std::cout << "  ring-switched: |Y| max " << want_max << ", max abs err "
            << sw_max << ", mean " << sw_mean << " ("
            << -std::log2(sw_max / want_max) << " bits)" << std::endl;
  EXPECT_LT(sw_max / want_max, 1e-3)
      << "the ring-switched projection disagrees with the host product";

  // ---- the direct route, for comparison ---------------------------------
  //
  // 256 ModPack keys at the block's degree, which is what the descent exists
  // to avoid; they are generated here only so the two routes can be compared
  // on the same ciphertexts in the same process.
  block.ui->PrepareModPackKeys(kSmallDegree, kLevel);
  std::vector<const EvaluationKey<word> *> big_keys(rank);
  for (int j = 0; j < rank; j++) {
    big_keys[j] = &block.ui->GetModPackKey(rank, j);
  }

  std::vector<double> got_direct;
  {
    cheddar::CoeffLinearLeg<word>::Config direct_cfg = cfg;
    direct_cfg.parents_per_tile = 16;
    ProjectOnlyLeg leg(block.context, direct_cfg, big_keys);
    EXPECT_FALSE(leg.IsRingSwitched());
    std::vector<Ciphertext<word>> res;
    leg.Project(res, parents, in_channels, kOutChannels, w, 1.0, "direct");
    ASSERT_EQ(static_cast<int>(res.size()), kOutChannels / rank);
    ReadBack(got_direct, res, block, rank, num_slots);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  double dir_max = 0.0, gap = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    dir_max = std::max(dir_max, std::abs(got_direct[i] - want[i]));
    gap = std::max(gap, std::abs(got_direct[i] - got_switched[i]));
  }
  std::cout << "  direct:        max abs err " << dir_max << " ("
            << -std::log2(dir_max / want_max) << " bits)" << std::endl;
  std::cout << "  the two routes differ by at most " << gap << " ("
            << gap / want_max << " relative)" << std::endl;
  EXPECT_LT(dir_max / want_max, 1e-3)
      << "the direct projection disagrees with the host product";

  // Both routes carry their own independent noise -- different keys, a
  // different number of key switches, a different ring for the accumulation --
  // so they are required to agree to the accuracy of the product and not
  // bit-for-bit. An index mistake would put this at order |Y|.
  EXPECT_LT(gap / want_max, 1e-3)
      << "the ring-switched and direct descents computed different products, "
         "which means the component reindexing is wrong rather than the "
         "arithmetic";

  // The control: the reindexing is NOT the identity, so reading the switched
  // result as if it were would have to fail. Without this a run in which
  // `Component` silently returned `flat` would pass everything above.
  ASSERT_NE(ring_rank, 1);
  int moved = 0;
  for (int n = 0; n < rank; n++) {
    if ((n / sub_rank) + ring_rank * (n % sub_rank) != n) moved++;
  }
  EXPECT_GT(moved, rank / 2)
      << "the two descents would be indistinguishable at this rank";
}

// ===========================================================================
// The same projection on the conjugate-invariant ring, both descents.
//
// [SYLPH] section 3.2's descent DOES port to R+ -- but not as the reindexing
// it is on the ordinary ring, and the difference is what this test pins.
//
//     direct      65536 --ModDecomp(rank 512)--------------------> 128
//     switched    65536 --RingSwitch(16)--> 4096 --ModDecomp(32)--> 128
//
// ON THE ORDINARY RING both decompositions split coefficients by residue, so
// the two strides compose as an index identity and `CoeffLinearLeg::
// Component` -- one permutation of the plaintext -- is the whole difference
// between the routes.
//
// ON R+ NEITHER IS A SPLIT. The module basis is not the monomials but the
// Chebyshev elements `c_m = Y^m + Y^-m`, and `c_a c_b = c_{a+b} + c_{a-b}`,
// so the composed basis element for the two-stage index `(j, n)` is
//
//     c_{ring_rank * n + j}  +  c_{|ring_rank * n - j|}
//
// -- a SUM of two one-stage components, not one of them. Enumerated on a
// miniature ring (D = 32 at 2 x 4) and confirmed at D = 256 at 4 x 8: the
// ordinary formula returns exactly the first of the two and the second is
// missing. Measured on the device before the fix, the switched route
// disagreed with the host product by 3.8 relative at half density and 5.8 at
// full, while the direct route was exact to 24.3 bits on the same
// ciphertexts. No permutation repairs that; a two-term sum is not a
// relabelling. (Half density does not repair it either -- it repairs the
// READ, 1.5by, where the pairing is at the big rank and the dead half kills
// the partner; here the pairing happens twice, at ranks 16 and 32, and a
// big-rank half-density set is scattered in both.)
//
// WHAT REPAIRS IT is refusing the question. Nothing forces the channel to be
// a one-stage module component -- the block chooses its packing. Declaring
// the channel to be the TWO-STAGE index makes the switched descent exact:
// the decomposition hands back what the recomposition put in, the mix is a
// scalar combination of those, and one projection's output is the next
// one's input under the same convention, so it composes across the whole
// coefficient-domain leg. The price is paid only where the payload is read
// as slots: each channel then appears at two coefficient addresses,
// `ring_rank*n + j` and `|ring_rank*n - j|`, ADDED -- exactly the structure
// Doing.md 1.5bp found for the CC-MM chain's nested operand, resolved the
// same way, by the layout rather than by a transform.
//
// So the two routes are read in DIFFERENT packings here and compared on the
// channels, which is the only thing a projection promises.
//
// 1.5bi's `RingSwitch.DescendsToTheProjectionShapeAndReturns` validated the
// switched chain against "the same composition of maps on the host" -- i.e.
// against whatever the composition gives, not against a channel indexing.
// That is why this went unnoticed until a projection asked for it.
//
// CHEDDAR_CI_IN_CHANNELS sets the contraction width, which is the only knob
// that moves the trade: ModPack is width-independent and the PP-MM is not.
// ===========================================================================

namespace {

// The banded two-term recomposition (Doing.md 1.5ba / 1.5bp): what ModPack
// writes, and therefore what a parent has to carry for ModDecomp to hand back
// the components asked for.
std::vector<double> CiRecompose(const std::vector<std::vector<double>> &comp,
                                int rank, int small_degree) {
  std::vector<double> out(static_cast<size_t>(rank) * small_degree, 0.0);
  for (int t = 0; t < small_degree; t++) {
    for (int i = 0; i < rank; i++) {
      double v = comp[i][t];
      if (i != 0 && t + 1 < small_degree) v += comp[rank - i][t + 1];
      out[static_cast<size_t>(t) * rank + i] = v;
    }
  }
  return out;
}

// Its inverse, an alternating suffix scan down the position axis, coupling
// `i` with `rank - i`.
std::vector<std::vector<double>> CiComponents(const std::vector<double> &coeffs,
                                              int rank, int small_degree) {
  std::vector<std::vector<double>> comp(rank,
                                        std::vector<double>(small_degree));
  for (int t = 0; t < small_degree; t++) {
    comp[0][t] = coeffs[static_cast<size_t>(t) * rank];
  }
  for (int i = 1; i <= rank / 2; i++) {
    const int mi = rank - i;
    double acc_i = 0.0, acc_m = 0.0;
    for (int t = small_degree - 1; t >= 0; t--) {
      const double new_i = coeffs[static_cast<size_t>(t) * rank + i] - acc_m;
      const double new_m = coeffs[static_cast<size_t>(t) * rank + mi] - acc_i;
      comp[i][t] = new_i;
      comp[mi][t] = new_m;
      acc_i = new_i;
      acc_m = new_m;
    }
  }
  return comp;
}

// The SAME two scans run in two stages, which is what the ring-switched
// descent physically does: `ring_rank` parts of the big ring, each split at
// `sub_rank`. Component `flat = j * sub_rank + n`.
std::vector<double> CiRecomposeTwoStage(
    const std::vector<std::vector<double>> &comp, int ring_rank, int sub_rank,
    int small_degree) {
  std::vector<std::vector<double>> parts(ring_rank);
  for (int j = 0; j < ring_rank; j++) {
    std::vector<std::vector<double>> sub(sub_rank);
    for (int n = 0; n < sub_rank; n++) sub[n] = comp[j * sub_rank + n];
    parts[j] = CiRecompose(sub, sub_rank, small_degree);
  }
  return CiRecompose(parts, ring_rank, sub_rank * small_degree);
}

std::vector<std::vector<double>> CiComponentsTwoStage(
    const std::vector<double> &coeffs, int ring_rank, int sub_rank,
    int small_degree) {
  const auto parts =
      CiComponents(coeffs, ring_rank, sub_rank * small_degree);
  std::vector<std::vector<double>> comp(ring_rank * sub_rank);
  for (int j = 0; j < ring_rank; j++) {
    const auto sub = CiComponents(parts[j], sub_rank, small_degree);
    for (int n = 0; n < sub_rank; n++) comp[j * sub_rank + n] = sub[n];
  }
  return comp;
}

double SecondsSince(const std::chrono::steady_clock::time_point &t0) {
  cudaDeviceSynchronize();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

}  // namespace

TEST(SwitchedProjection, TheRingSwitchedDescentPortsToTheRealSubring) {
  const char *big_param = std::getenv("CHEDDAR_CI_SWITCH_PARAM");
  const char *small_param = std::getenv("CHEDDAR_CI_SMALL_PARAM");
  Ring block(big_param && big_param[0] ? big_param
                                       : "ci_ringswitch16_35.json");
  Ring small(small_param && small_param[0] ? small_param : "ci12_35.json");
  ASSERT_TRUE(block.param->conjugate_invariant_);
  ASSERT_TRUE(small.param->conjugate_invariant_);

  using Leg = cheddar::CoeffLinearLeg<word>;
  const int degree = block.Degree();
  const int num_slots = block.param->MaxNumSlots();
  const int mid_degree = small.Degree();
  const int ci_small_degree = Leg::SmallDegreeFor(kTokens, *block.param);
  const int rank = degree / ci_small_degree;          // 512 channels per ct
  const int ring_rank = degree / mid_degree;          // 16
  const int sub_rank = mid_degree / ci_small_degree;  // 32
  const char *width = std::getenv("CHEDDAR_CI_IN_CHANNELS");
  const int in_channels = width ? std::atoi(width) : kInChannels;
  // The output width matters for a reason the input one does not: `Decompose`
  // is hoisted OUT of the group loop, so the whole decomposition is shared by
  // every output ciphertext and a one-group measurement charges all of it to
  // one. The layer's projections are 8 groups (4096 channels) or 56
  // (14336), never one.
  const char *ow = std::getenv("CHEDDAR_CI_OUT_CHANNELS");
  const int out_channels = ow ? std::atoi(ow) : kOutChannels;
  const int num_parents = in_channels / rank;

  ASSERT_EQ(num_slots, degree) << "R+ gives `degree` real slots";
  ASSERT_EQ(ci_small_degree, kTokens)
      << "on R+ a T-token channel occupies T coefficients, not 2T";
  ASSERT_EQ(rank, 512);
  ASSERT_EQ(ring_rank * sub_rank, rank);
  ASSERT_EQ(num_slots / kTokens, rank);
  ASSERT_GE(num_parents, 1);

  const double ct_scale = block.param->GetScale(kLevel);

  // ---- the tensors ------------------------------------------------------
  std::mt19937_64 gen(0xC15117);
  std::normal_distribution<double> xd(0.0, 1.0);
  std::normal_distribution<double> wd(0.0, 0.0175);  // Llama-3's W_Q RMS
  std::vector<double> x(static_cast<size_t>(kTokens) * in_channels);
  std::vector<double> w(static_cast<size_t>(in_channels) * out_channels);
  for (auto &v : x) v = xd(gen);
  for (auto &v : w) v = wd(gen);

  // HALF DENSITY, WHICH IS THE LEG'S CONTRACT ON R+ AND NOT A CONVENIENCE.
  //
  // 1.5by: ModPack's banded recomposition writes `comp_I[t] + comp_{rank-I}
  // [t+1]`, so a full-density projection output is a MIXTURE of two channels
  // and nothing downstream can read it. What a projection on R+ emits is 256
  // live components of 512, with the partner of every live one dead -- and
  // then every entry lands clean at its primary coefficient and the upper
  // half holds shifted duplicates rather than mixtures.
  //
  // A component `i` carries channel `BitReverse(i, 9)`, so the live half
  // `i < 256` is exactly the EVEN channels. Zeroing the odd ones on both
  // axes is that contract, stated in the units this test speaks.
  //
  // Whether the ring-switched descent needs it is the question this test was
  // rewritten to ask; the run below (`CHEDDAR_CI_FULL_DENSITY=1`) is the
  // control that says what happens without it.
  const bool half_density = std::getenv("CHEDDAR_CI_FULL_DENSITY") == nullptr;
  if (half_density) {
    for (int t = 0; t < kTokens; t++) {
      for (int c = 1; c < in_channels; c += 2) {
        x[static_cast<size_t>(t) * in_channels + c] = 0.0;
      }
    }
    for (int c = 0; c < in_channels; c++) {
      for (int o = 0; o < out_channels; o++) {
        if ((c % 2) || (o % 2)) w[static_cast<size_t>(c) * out_channels + o] = 0.0;
      }
    }
  }
  std::cout << "  operands are " << (half_density ? "HALF" : "FULL")
            << " density" << std::endl;
  for (int t = 0; t < kTokens; t++) {
    double sq = 0.0;
    for (int c = 0; c < in_channels; c++) {
      const double v = x[static_cast<size_t>(t) * in_channels + c];
      sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq / in_channels + 1e-5);
    for (int c = 0; c < in_channels; c++) {
      x[static_cast<size_t>(t) * in_channels + c] *= inv;
    }
  }

  std::vector<double> want(static_cast<size_t>(kTokens) * out_channels, 0.0);
  double want_max = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int o = 0; o < out_channels; o++) {
      double acc = 0.0;
      for (int c = 0; c < in_channels; c++) {
        acc += x[static_cast<size_t>(t) * in_channels + c] *
               w[static_cast<size_t>(c) * out_channels + o];
      }
      want[static_cast<size_t>(t) * out_channels + o] = acc;
      want_max = std::max(want_max, std::abs(acc));
    }
  }

  // ---- encrypt: the block's packing, delivered as a banded image --------
  //
  // The plain packing says where a (token, channel) sits in the coefficient
  // index; splitting that index into (component, position) says which module
  // component has to carry it; and the banded recomposition is what a parent
  // must hold for ModDecomp to hand those components back.
  auto pack_components = [&](int parent) {
    std::vector<std::vector<double>> comp(
        rank, std::vector<double>(ci_small_degree, 0.0));
    for (int c = 0; c < rank; c++) {
      for (int t = 0; t < kTokens; t++) {
        const int k = AttentionPacking::CoeffOfSlot(
            {t + kTokens * c, false}, degree, /*conjugate_invariant=*/true);
        comp[k % rank][k / rank] =
            x[static_cast<size_t>(t) * in_channels + parent * rank + c];
      }
    }
    return comp;
  };

  // THE PACKING IS THE ROUTE'S OWN, AND THAT IS THE FIX.
  //
  // Nothing forces the channel to be a ONE-stage module component -- the
  // block chooses its packing. Declaring the channel to be the two-stage
  // index makes the switched descent exact: the decomposition hands back
  // what the recomposition put in, the mix is a scalar combination of
  // those, and one projection's output is the next one's input under the
  // same convention. Enumerated on miniature rings first (D = 32 at 2 x 4
  // and D = 256 at 4 x 8, both exact to 1e-14) rather than derived on the
  // device, which is 1.5bx's lesson.
  auto encode_parents = [&](bool two_stage) {
    std::vector<Ciphertext<word>> res(num_parents);
    for (int p = 0; p < num_parents; p++) {
      const auto comp = pack_components(p);
      Plaintext<word> pt;
      block.context->encoder_.EncodeCoeff(
          pt, kLevel, ct_scale,
          two_stage ? CiRecomposeTwoStage(comp, ring_rank, sub_rank,
                                          ci_small_degree)
                    : CiRecompose(comp, rank, ci_small_degree));
      block.ui->Encrypt(res[p], pt);
    }
    return res;
  };

  auto read_back = [&](std::vector<double> &out,
                       const std::vector<Ciphertext<word>> &res,
                       bool two_stage) {
    out.assign(static_cast<size_t>(kTokens) * out_channels, 0.0);
    for (size_t g = 0; g < res.size(); g++) {
      Plaintext<word> pt;
      block.ui->Decrypt(pt, res[g]);
      std::vector<double> coeffs;
      block.context->encoder_.DecodeCoeff(coeffs, pt);
      const auto comp =
          two_stage ? CiComponentsTwoStage(coeffs, ring_rank, sub_rank,
                                           ci_small_degree)
                    : CiComponents(coeffs, rank, ci_small_degree);
      for (int c = 0; c < rank; c++) {
        for (int t = 0; t < kTokens; t++) {
          const int k = AttentionPacking::CoeffOfSlot(
              {t + kTokens * c, false}, degree, /*conjugate_invariant=*/true);
          out[static_cast<size_t>(t) * out_channels + g * rank + c] =
              comp[k % rank][k / rank];
        }
      }
    }
  };

  auto report = [&](const char *name, const std::vector<double> &got,
                    double seconds, int emissions) {
    double mx = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
      mx = std::max(mx, std::abs(got[i] - want[i]));
    }
    std::cout << "  " << name << ": max abs err " << mx << " ("
              << -std::log2(mx / want_max) << " bits), " << (seconds * 1e3)
              << " ms for " << emissions << " output ciphertext(s) = "
              << (seconds * 1e3 / emissions) << " ms/ct" << std::endl;
    return mx;
  };

  Leg::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.product_level = kLevel;

  // ---- [SYLPH] section 3.2's route --------------------------------------
  block.ui->PrepareRingSwitchKey(mid_degree, small.ui->GetSecretCoeffs(),
                                 kLevel);
  block.ui->PrepareInverseRingSwitchKey(mid_degree,
                                        small.ui->GetSecretCoeffs(), kLevel);
  small.ui->PrepareModPackKeys(ci_small_degree, kLevel);
  std::vector<const EvaluationKey<word> *> small_keys(sub_rank);
  for (int j = 0; j < sub_rank; j++) {
    small_keys[j] = &small.ui->GetModPackKey(sub_rank, j);
  }

  Leg::Descent descent;
  descent.switch_context = block.context;
  descent.small_context = small.context;
  descent.forward = &block.ui->GetRingSwitchKey(ring_rank);
  descent.inverse = &block.ui->GetInverseRingSwitchKey(ring_rank);
  descent.modpack_keys = small_keys;

  std::vector<double> got_switched;
  double sw_seconds = 0.0;
  {
    Leg::Config sw_cfg = cfg;
    sw_cfg.parents_per_tile = 0;  // the descent holds a sixteenth
    ProjectOnlyLeg leg(block.context, sw_cfg, {}, descent);
    EXPECT_TRUE(leg.IsRingSwitched());
    EXPECT_EQ(leg.GetRank(), rank);
    EXPECT_EQ(leg.GetSmallDegree(), ci_small_degree);
    EXPECT_EQ(leg.GetRingRank(), ring_rank);
    EXPECT_EQ(leg.GetSubRank(), sub_rank);

    const auto sw_parents = encode_parents(/*two_stage=*/true);
    std::vector<Ciphertext<word>> res;
    leg.Project(res, sw_parents, in_channels, out_channels, w, 1.0, "switched");
    const auto t0 = std::chrono::steady_clock::now();
    leg.Project(res, sw_parents, in_channels, out_channels, w, 1.0, "switched");
    sw_seconds = SecondsSince(t0);
    ASSERT_EQ(static_cast<int>(res.size()), out_channels / rank);
    EXPECT_EQ(block.param->NPToLevel(res[0].GetNP()), kLevel - 1);
    EXPECT_NEAR(res[0].GetScale() / block.param->GetScale(kLevel - 1), 1.0,
                1e-6);
    EXPECT_EQ(res[0].GetNumSlots(), num_slots);
    read_back(got_switched, res, /*two_stage=*/true);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const double sw_max =
      report("ring-switched", got_switched, sw_seconds, out_channels / rank);
  // THE FINDING, PINNED AS A REGRESSION.
  //
  // On the ordinary ring both decompositions split coefficients by residue,
  // so composing the two strides is an index identity and `Component()` --
  // one permutation of the plaintext -- is the whole difference between the
  // routes. On R+ neither is a split: both are banded scans, pairing
  // component `i` with `rank - i` at their OWN rank. Composing a rank-16
  // scan with a rank-32 one is not a rank-512 scan, it is the rank-512 scan
  // of a two-term sum (Doing.md 1.5bp's `g`), and no permutation of the
  // plaintext expresses that. Measured: the switched route disagrees with
  // the host product by 3.8 relative at half density and 5.8 at full, while
  // the direct route is exact to 24.3 bits on the same ciphertexts.
  //
  // Half density does NOT rescue it, which is the part worth keeping. It
  // rescues the READ (1.5by) because there the pairing is at the big rank
  // and the dead half kills the partner. Here the pairing happens twice, at
  // ranks 16 and 32, and a half-density set in big-rank terms is scattered
  // in both -- neither partner is dead.
  //
  // So [SYLPH] section 3.2's descent does not port to R+ as a reindexing,
  // and the CI leg takes the direct route. This assertion is the wrong way
  // round on purpose: if someone derives the composition and fixes
  // `Component()`, this fails and they delete it.
  EXPECT_LT(sw_max / want_max, 1e-3)
      << "the ring-switched descent disagrees with the host product on R+";

  // ---- the direct route, 512 ModPack keys at the block's degree ----------
  block.ui->PrepareModPackKeys(ci_small_degree, kLevel);
  std::vector<const EvaluationKey<word> *> big_keys(rank);
  for (int j = 0; j < rank; j++) {
    big_keys[j] = &block.ui->GetModPackKey(rank, j);
  }

  std::vector<double> got_direct;
  double dir_seconds = 0.0;
  {
    Leg::Config direct_cfg = cfg;
    direct_cfg.parents_per_tile = 0;
    ProjectOnlyLeg leg(block.context, direct_cfg, big_keys);
    EXPECT_FALSE(leg.IsRingSwitched());
    const auto dir_parents = encode_parents(/*two_stage=*/false);
    std::vector<Ciphertext<word>> res;
    leg.Project(res, dir_parents, in_channels, out_channels, w, 1.0, "direct");
    const auto t0 = std::chrono::steady_clock::now();
    leg.Project(res, dir_parents, in_channels, out_channels, w, 1.0, "direct");
    dir_seconds = SecondsSince(t0);
    ASSERT_EQ(static_cast<int>(res.size()), out_channels / rank);
    read_back(got_direct, res, /*two_stage=*/false);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const double dir_max =
      report("direct       ", got_direct, dir_seconds, out_channels / rank);
  EXPECT_LT(dir_max / want_max, 1e-3)
      << "the direct projection disagrees with the host product";

  double gap = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    gap = std::max(gap, std::abs(got_direct[i] - got_switched[i]));
  }
  std::cout << "  the two routes differ by at most " << gap << " ("
            << (gap / want_max) << " relative); |Y| max " << want_max
            << std::endl;
  std::cout << "  THE TRADE (Doing.md 1.5bi): switched " << (sw_seconds * 1e3)
            << " ms and 1 + 1 + " << sub_rank << " keys against direct "
            << (dir_seconds * 1e3) << " ms and " << rank
            << " keys at degree " << degree << "; noise "
            << -std::log2(sw_max / want_max) << " vs "
            << -std::log2(dir_max / want_max) << " bits" << std::endl;
  EXPECT_LT(gap / want_max, 1e-3)
      << "the two R+ descents computed different products -- each is read in "
         "its own packing, so this compares the CHANNELS and not the "
         "coefficients";

  // What DOES have to hold: the direct route is the leg's route on R+, so
  // its agreement with the host product is a real assertion and not a pin.
  EXPECT_LT(dir_max / want_max, 1e-5)
      << "the direct conjugate-invariant projection is what the leg runs";
}
