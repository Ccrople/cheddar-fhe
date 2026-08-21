// A Llama-3 projection, encrypted, in the block's own packing, on the block's
// own parameter set, with no ring switch anywhere.
//
// WHY THIS TEST EXISTS. `LlamaBlockTest` runs a whole decoder block but hands
// its seven matrix products to `HostLinearLeg`, which decrypts, multiplies in
// the clear and re-encrypts. The reason recorded in Doing.md 1.5y was that the
// coefficient-domain product needs a ring-switching parameter pair and
// `bootparam_35` has no securely usable switching level. That reason does not
// survive reading 1.5y's own conclusion:
//
//     "ring switching is an optimisation of the linear leg, not a correctness
//      requirement"
//
// Ring switching buys a *smaller ring*. What the product actually needs is one
// channel per ciphertext, so that the Bae PC-MM's scalar combination
// `res[i] = sum_j u[i][j] * cts[j]` contracts the channel axis. **ModDecomp
// alone delivers that**, and ModDecomp is free: no key, no security spent,
// because rank k over degree N' is exactly RLWE over degree kN' (Mlwe.h:26).
//
// THE ALIGNMENT, WHICH IS THE WHOLE CONTENT OF THIS FILE.
//
// ModDecomp sends coefficient `i + rank * s` of the parent to position `s` of
// MLWE ciphertext `i` (Mlwe.cu:21). So it groups by residue mod rank, and the
// axis it isolates is whichever one owns the **low coefficient bits**.
//
// The block's slot layout is `slot s -> token s % T, channel s / T`, so the
// token owns the low *slot* bits. `CoeffOfSlot` is exactly `BitReverse(s, 15)`
// (AttentionPacking.cpp:102-110), which sends low slot bits to high
// coefficient bits. In coefficients, therefore, the **channel is low** -- and
// that is precisely what ModDecomp isolates. Concretely, for T = 128,
//
//     coeff(token t, channel-in-ct c) = BitReverse8(c) + 256 * BitReverse7(t)
//
// so `coeff mod 256` is `BitReverse8(c)`: one channel per MLWE ciphertext,
// with the channel index bit-reversed. That reversal is a fixed permutation of
// the rows of the weight matrix and is absorbed into the plaintext operand for
// nothing.
//
// T = 128 IS THE ONLY TOKEN COUNT THAT WORKS, which is worth stating because
// `LlamaBlockTest` currently uses 64. Enumerating all 32768 slots:
//
//     T = 64    rank 128 -> 4 channels/ct   rank 256 -> 2   rank 512 illegal
//     T = 128   rank 128 -> 2 channels/ct   rank 256 -> 1   rank 512 illegal
//
// rank 512 is illegal because `Mlwe.cu:158` requires `small_degree % 256 == 0`
// and 65536/512 = 128. So the truncated-attention hole and the encrypted-
// product hole have the same fix.
//
// THE PRODUCT IS LAYOUT-PRESERVING, which is what makes it a drop-in for
// `LlamaBlock::LinearLeg`. Composing the maps, an output channel `o` at token
// `t` lands at coefficient `BitReverse8(o) + 256 * BitReverse7(t)`, i.e. at
// `CoeffOfSlot({t + 128*o, false})` -- the same layout the input arrived in.
// This test asserts that rather than assuming it.
//
// THE SCALE BUDGET HAS EXACTLY ONE ANSWER, as it did for the ring-switched
// chain in PipelineChainTest. Rescaling the product from level 1 lands on
// `GetScale(0)` only when the weight scale is `GetScale(1)`:
//
//     GetScale(1) * w / GetRescalePrimeProd(1) = GetScale(0)  =>  w = GetScale(1)
//
// For `bootparam_35` that is 2^35.002, putting the product at 2^70.004 against
// log2 Q1 = 85.018 -- about fifteen bits of headroom for the message
// magnitude, against block crossings that all sit near 0.5.
//
// WHAT THIS COSTS, STATED PLAINLY. ModDecomp to rank k multiplies the a-part
// by k: one parent's 3 x 65536 words become 256 MLWE ciphertexts of 3 x 65536
// words each. At 4096 input channels that is ~3.2 GB, and it is the reason
// [SYLPH] ring-switches to degree 4096 first (k = 16 rather than 256). This
// file deliberately pays that cost to isolate the product from the switching:
// a failure here means the *product* is wrong.

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "Testbed.h"
#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#include "extension/AttentionPacking.h"
#include "extension/LlamaLinear.h"

using word = uint32_t;
using cheddar::AttentionPacking;
using cheddar::Ciphertext;
using cheddar::EvaluationKey;
using cheddar::MlweCiphertext;
using cheddar::MlweHandler;
using cheddar::PcmmHandler;
using cheddar::PlainMatrix;
using cheddar::Plaintext;

namespace {

constexpr int kTokens = 128;       // the only count the packing admits
constexpr int kInChannels = 4096;  // Llama-3-8B's hidden width, in full
constexpr int kOutChannels = 256;  // exactly one ModPack group
constexpr int kSmallDegree = 256;  // so rank = 65536 / 256 = 256
constexpr int kLevel = 1;          // the product's input level; it rescales to 0

int BitRev(int x, int bits) {
  return static_cast<int>(cheddar::BitReverseInt(x, bits));
}

// The block's packing, as LlamaBlockTest builds it: slot s carries token s % T
// of channel s / T, and the conversion to coefficients is the bit reversal.
int CoeffOfTokenChannel(int token, int channel_in_ct, int num_slots) {
  const int slot = token + kTokens * channel_in_ct;
  return AttentionPacking::CoeffOfSlot({slot, false}, num_slots * 2);
}

std::string DataDir() {
  const char *env = std::getenv("LLAMA3_REAL_DIR");
  return env ? std::string(env) : std::string();
}

bool ReadF32(const std::string &path, size_t count, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<float> raw(count);
  f.read(reinterpret_cast<char *>(raw.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  if (static_cast<size_t>(f.gcount()) != count * sizeof(float)) return false;
  out.assign(raw.begin(), raw.end());
  return true;
}

// `CoeffLinearLeg` implements only `Project`; `Scores` and `Values` are
// ciphertext-ciphertext products that need `BatchCcmmHandler` and are left
// pure virtual on purpose, so that nothing silently falls back to a stand-in.
// This makes the class concrete for a test that exercises the projection only.
class ProjectOnlyLeg : public cheddar::CoeffLinearLeg<word> {
 public:
  using cheddar::CoeffLinearLeg<word>::CoeffLinearLeg;

  void Scores(std::vector<Ciphertext<word>> &, const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double,
              const std::vector<double> &) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: Scores is not part of this test");
  }
  void Values(std::vector<Ciphertext<word>> &, const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: Values is not part of this test");
  }
  void LocateScore(int, int, int, int &, int &) const override {
    cheddar::AssertTrue(false,
                        "ProjectOnlyLeg: there is no score layout in this test");
  }
};

}  // namespace

TEST_P(Testbed32, ProjectionInTheBlockPackingWithoutARingSwitch) {
  const int degree = 1 << log_degree_;
  const int num_slots = degree / 2;
  const int rank = degree / kSmallDegree;
  const int num_parents = kInChannels / (num_slots / kTokens);

  ASSERT_EQ(rank, 256);
  ASSERT_EQ(num_slots / kTokens, rank)
      << "the block's channels-per-ciphertext must equal the ModDecomp rank, "
         "which is what puts exactly one channel in each MLWE ciphertext";
  ASSERT_EQ(num_parents, 16);
  ASSERT_GE(param_->max_level_, kLevel);

  const double ct_scale = param_->GetScale(kLevel);
  const double u_scale = param_->GetScale(kLevel);  // the one answer; see above
  EXPECT_NEAR(ct_scale * u_scale / param_->GetRescalePrimeProd(kLevel) /
                  param_->GetScale(0),
              1.0, 1e-9)
      << "the weight scale must be GetScale(1) for the rescaled product to "
         "land on the ring's own level-0 scale";

  // ---- the tensors ------------------------------------------------------
  //
  // Real Llama-3-8B layer-2 weights when the bundle is present, and weights
  // shaped like them when it is not, so the test always runs. The activation
  // is RMSNorm'd in either case, because that is what reaches the projection
  // in the block and it is what bounds |Y|.
  std::vector<double> x(static_cast<size_t>(kTokens) * kInChannels);
  std::vector<double> w(static_cast<size_t>(kInChannels) * kOutChannels);
  const std::string dir = DataDir();
  bool real_weights = false;
  if (!dir.empty()) {
    std::vector<double> all_x, all_w;
    if (ReadF32(dir + "/input.f32", static_cast<size_t>(128) * kInChannels,
                all_x) &&
        ReadF32(dir + "/wq.f32",
                static_cast<size_t>(kInChannels) * kInChannels, all_w)) {
      x.assign(all_x.begin(), all_x.begin() + x.size());
      for (int i = 0; i < kInChannels; i++) {
        for (int o = 0; o < kOutChannels; o++) {
          w[static_cast<size_t>(i) * kOutChannels + o] =
              all_w[static_cast<size_t>(i) * kInChannels + o];
        }
      }
      real_weights = true;
    }
  }
  if (!real_weights) {
    std::mt19937_64 gen(0x11A3);
    std::normal_distribution<double> xd(0.0, 1.0);
    std::normal_distribution<double> wd(0.0, 0.0175);  // Llama-3's W_Q RMS
    for (auto &v : x) v = xd(gen);
    for (auto &v : w) v = wd(gen);
  }
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

  // ---- the host reference, computed before any crypto -------------------
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
  ASSERT_LT(want_max * ct_scale * u_scale,
            std::pow(2.0, 84.0))  // Q1/2, with room to spare
      << "the product would exceed the level-1 modulus";

  interface_->PrepareModPackKeys(kSmallDegree, kLevel);

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
    context_->encoder_.EncodeCoeff(pt, kLevel, ct_scale, coeffs);
    interface_->Encrypt(parents[p], pt);
  }

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);
  PcmmHandler<word> pcmm(*param_);

  // ---- ModDecomp: one channel per MLWE ciphertext ------------------------
  //
  // MLWE ciphertext `i` of parent `p` carries channel `p * rank +
  // BitReverse8(i)`. Rather than permute the ciphertexts, the permutation is
  // folded into the weight matrix below, where it is free.
  std::vector<MlweCiphertext<word>> columns;
  columns.reserve(static_cast<size_t>(num_parents) * rank);
  for (int p = 0; p < num_parents; p++) {
    std::vector<MlweCiphertext<word>> decomposed;
    mlwe.ModDecomp(decomposed, parents[p], kSmallDegree);
    ASSERT_EQ(static_cast<int>(decomposed.size()), rank);
    for (auto &d : decomposed) columns.push_back(std::move(d));
  }
  ASSERT_EQ(static_cast<int>(columns.size()), kInChannels);

  // ---- the plaintext operand --------------------------------------------
  //
  // Row `r` of U becomes MLWE ciphertext `r`, which ModPack will place at
  // decomposition index `r`, which decodes as output channel BitReverse8(r).
  // So row r must carry output channel BitReverse8(r); and column
  // `p * rank + i` carries input channel `p * rank + BitReverse8(i)`. Both
  // reversals are their own inverse.
  const int log_rank = cheddar::Log2Ceil(rank);
  std::vector<double> u_values(static_cast<size_t>(kOutChannels) * kInChannels);
  for (int r = 0; r < kOutChannels; r++) {
    const int out_channel = BitRev(r, log_rank);
    for (int p = 0; p < num_parents; p++) {
      for (int i = 0; i < rank; i++) {
        const int in_channel = p * rank + BitRev(i, log_rank);
        u_values[static_cast<size_t>(r) * kInChannels + p * rank + i] =
            w[static_cast<size_t>(in_channel) * kOutChannels + out_channel];
      }
    }
  }
  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, kLevel, u_scale, u_values, kOutChannels, kInChannels);

  // ---- the product, then back ------------------------------------------
  std::vector<MlweCiphertext<word>> product;
  pcmm.Multiply(product, u, columns);
  ASSERT_EQ(static_cast<int>(product.size()), kOutChannels);

  std::vector<const EvaluationKey<word> *> modpack_keys(rank);
  for (int j = 0; j < rank; j++) {
    modpack_keys[j] = &interface_->GetModPackKey(rank, j);
  }

  Ciphertext<word> repacked;
  mlwe.ModPack(context_, repacked, product, modpack_keys);
  EXPECT_EQ(param_->NPToLevel(repacked.GetNP()), kLevel)
      << "ModPack is a key switch and must not consume a level";

  Ciphertext<word> out;
  context_->Rescale(out, repacked);
  EXPECT_EQ(param_->NPToLevel(out.GetNP()), 0);
  EXPECT_NEAR(out.GetScale() / param_->GetScale(0), 1.0, 1e-9)
      << "the rescaled product must be canonical at level 0, or EvalPoly will "
         "abort inside the next bootstrap three layers from here";

  // ---- read it back in the block's packing ------------------------------
  Plaintext<word> pt_out;
  interface_->Decrypt(pt_out, out);
  std::vector<double> got;
  context_->encoder_.DecodeCoeff(got, pt_out);

  double max_err = 0.0, mean_err = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int o = 0; o < kOutChannels; o++) {
      // The layout-preserving claim: output channel o sits exactly where an
      // input channel o would have.
      const double v = got[CoeffOfTokenChannel(t, o, num_slots)];
      const double e =
          std::abs(v - want[static_cast<size_t>(t) * kOutChannels + o]);
      max_err = std::max(max_err, e);
      mean_err += e;
    }
  }
  mean_err /= static_cast<double>(kTokens) * kOutChannels;

  std::cout << "  weights: " << (real_weights ? "real Llama-3-8B layer 2"
                                              : "shaped like Llama-3-8B")
            << std::endl;
  std::cout << "  |Y| max " << want_max << ", max abs err " << max_err
            << ", mean " << mean_err << std::endl;
  std::cout << "  relative " << (max_err / want_max) << "  ("
            << -std::log2(max_err / want_max) << " bits)" << std::endl;

  EXPECT_LT(max_err / want_max, 1e-3)
      << "the encrypted projection disagrees with the host product";
}

// The same product through `CoeffLinearLeg` -- the class the block will call --
// rather than through the handlers directly. What this adds over the test above
// is the leg's own bookkeeping: the two bit reversals across *several* ModPack
// groups (one group cannot distinguish them from the identity), the level and
// scale contract it owes `LinearLeg`, and its decision to skip the descent when
// the input already sits at the product level.
TEST_P(Testbed32, TheLegProjectsThroughSeveralModPackGroups) {
  constexpr int kLegOut = 512;  // two ModPack groups, so the reversals show
  const int degree = 1 << log_degree_;
  const int num_slots = degree / 2;
  const int rank = degree / kSmallDegree;
  const int num_parents = kInChannels / rank;
  const double ct_scale = param_->GetScale(kLevel);
  // A crossing constant, folded into the weights the way the block does it.
  const double w_scale = 0.5;

  std::mt19937_64 gen(0x1EA5);
  std::normal_distribution<double> xd(0.0, 1.0);
  std::normal_distribution<double> wd(0.0, 0.0175);
  std::vector<double> x(static_cast<size_t>(kTokens) * kInChannels);
  std::vector<double> w(static_cast<size_t>(kInChannels) * kLegOut);
  for (auto &v : x) v = xd(gen);
  for (auto &v : w) v = wd(gen);

  std::vector<double> want(static_cast<size_t>(kTokens) * kLegOut, 0.0);
  double want_max = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int o = 0; o < kLegOut; o++) {
      double acc = 0.0;
      for (int c = 0; c < kInChannels; c++) {
        acc += x[static_cast<size_t>(t) * kInChannels + c] *
               w[static_cast<size_t>(c) * kLegOut + o];
      }
      acc *= w_scale;
      want[static_cast<size_t>(t) * kLegOut + o] = acc;
      want_max = std::max(want_max, std::abs(acc));
    }
  }

  interface_->PrepareModPackKeys(kSmallDegree, kLevel);
  std::vector<const EvaluationKey<word> *> keys(rank);
  for (int j = 0; j < rank; j++) keys[j] = &interface_->GetModPackKey(rank, j);

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
    context_->encoder_.EncodeCoeff(pt, kLevel, ct_scale, coeffs);
    interface_->Encrypt(parents[p], pt);
  }

  // ONCE WITHOUT TILING AND ONCE WITH, and the two must agree.
  //
  // Tiling is what keeps the down projection's 14336-channel contraction off
  // the 11.3 GB of module components it would otherwise hold at once, and it
  // is the one part of the leg that changes the *order* of the arithmetic: the
  // partial products are ModPacked and added as ordinary ciphertexts before a
  // single rescale. 3 parents against 16 gives six tiles with a remainder of
  // one, so the short last tile is exercised too. `num_parents` is 16 here, so
  // the untiled run really is a single tile.
  std::vector<std::vector<double>> decoded;
  for (int tile : {0, 3}) {
    cheddar::CoeffLinearLeg<word>::Config lcfg;
    lcfg.num_tokens = kTokens;
    lcfg.product_level = kLevel;
    lcfg.parents_per_tile = tile;
    ProjectOnlyLeg leg(context_, lcfg, keys);
    EXPECT_EQ(leg.GetRank(), rank);
    EXPECT_EQ(leg.GetSmallDegree(), kSmallDegree);

    std::vector<Ciphertext<word>> out;
    leg.Project(out, parents, kInChannels, kLegOut, w, w_scale, "legQ");
    ASSERT_EQ(static_cast<int>(out.size()), kLegOut / rank);

    std::vector<double> flat(static_cast<size_t>(kTokens) * kLegOut, 0.0);
    double max_err = 0.0;
    for (int g = 0; g < kLegOut / rank; g++) {
      EXPECT_EQ(param_->NPToLevel(out[g].GetNP()), kLevel - 1)
          << "the leg owes LinearLeg a result one level below the product";
      EXPECT_NEAR(out[g].GetScale() / param_->GetScale(kLevel - 1), 1.0, 1e-9)
          << "and it owes it canonically scaled";
      Plaintext<word> pt;
      interface_->Decrypt(pt, out[g]);
      std::vector<double> got;
      context_->encoder_.DecodeCoeff(got, pt);
      for (int c = 0; c < rank; c++) {
        for (int t = 0; t < kTokens; t++) {
          const double v = got[CoeffOfTokenChannel(t, c, num_slots)];
          flat[static_cast<size_t>(t) * kLegOut + g * rank + c] = v;
          max_err = std::max(
              max_err, std::abs(v - want[static_cast<size_t>(t) * kLegOut +
                                         g * rank + c]));
        }
      }
    }
    std::cout << "  leg: " << (kLegOut / rank) << " ModPack groups, "
              << (tile == 0 ? num_parents : tile) << " parents per tile, |Y| max "
              << want_max << ", max abs err " << max_err << "  ("
              << -std::log2(max_err / want_max) << " bits)" << std::endl;
    EXPECT_LT(max_err / want_max, 1e-3);
    decoded.push_back(std::move(flat));
  }

  double drift = 0.0;
  for (size_t i = 0; i < decoded[0].size(); i++) {
    drift = std::max(drift, std::abs(decoded[0][i] - decoded[1][i]));
  }
  std::cout << "  tiled vs untiled: max abs difference " << drift << std::endl;
  EXPECT_LT(drift, 1e-6 * want_max)
      << "tiling reordered the contraction into a different answer";
}

INSTANTIATE_TEST_SUITE_P(
    BlockPacking, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string name = info.param;
      for (auto &c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
      }
      return name;
    });
