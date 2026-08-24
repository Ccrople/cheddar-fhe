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
  std::vector<double> x(static_cast<size_t>(kTokens) * kInChannels);
  std::vector<double> w(static_cast<size_t>(kInChannels) * kOutChannels);
  for (auto &v : x) v = xd(gen);
  for (auto &v : w) v = wd(gen);
  // RMSNorm the activation, as the block does before this product.
  for (int t = 0; t < kTokens; t++) {
    double sq = 0.0;
    for (int c = 0; c < kInChannels; c++) {
      const double v = x[static_cast<size_t>(t) * kInChannels + c];
      sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq / kInChannels + 1e-5);
    for (int c = 0; c < kInChannels; c++) {
      x[static_cast<size_t>(t) * kInChannels + c] *= inv;
    }
  }

  std::vector<double> want(static_cast<size_t>(kTokens) * kOutChannels, 0.0);
  double want_max = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int o = 0; o < kOutChannels; o++) {
      double acc = 0.0;
      for (int c = 0; c < kInChannels; c++) {
        acc += x[static_cast<size_t>(t) * kInChannels + c] *
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
    leg.Project(res, parents, kInChannels, kOutChannels, w, 1.0, "switched");
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
    leg.Project(res, parents, kInChannels, kOutChannels, w, 1.0, "direct");
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
