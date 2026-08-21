// Both attention products as the block will call them: **slots in, slots
// out**, with everything in between -- the permutations, the exchange,
// SlotToSinC, the ring switch, [KANG] Algorithm 4, HalfBoot and the StC
// prefix -- inside `SinCAttention`.
//
// WHAT IS NEW HERE AND NOT IN AttentionTransportTest. That test ends at the
// product's SinC output and reads it with `DecodeSinC`. This one carries it
// the rest of the way: **through the bootstrap and back into slots**, which is
// the half of the round trip the pipeline cannot avoid, because the batch
// CC-MM leaves its result at level 0 and nothing but a bootstrap gets out of
// level 0. It is also the first time the prefix runs on a product's output
// rather than on a transform's, and the first time it carries
// `Canonicalise`'s constant.
//
// WHAT THE TEST ASSERTS, AND WHY IT FITS A CONSTANT FIRST. The chain is a
// linear map times a constant that the parameter set fixes: StC bakes in
// `stc_const_`, CtS bakes in `cts_const_`, EvalMod has its own, and the prefix
// carries Canonicalise's. With `halfboot_scale` supplied, `SinCAttention`
// puts the result on the canonical scale of its level and the constant should
// come out **1** -- so the test identifies it, prints it, and requires it to
// be 1 as well as requiring the residual to be small. Getting the residual
// right with a constant of 17 would mean the level bookkeeping is wrong in a
// way that a later `EvalPoly` would abort on.
//
// THE CONTROLS. Both results have exactly the right magnitude under a wrong
// layout, so each product is read a second way that would be right if the
// layout were: the transpose for `Q K^T`, and the un-replicated KV head for
// GQA. Both have to fail.
//
// LEVELS. Q, K and V enter together at `swap_level`; P enters at
// `sinc_level`, because in the block P comes off SoftMax at the bottom of the
// slot leg while Q, K and V come off RoPE at the top. That is why
// `sinc_level` is its own parameter and not `swap_level - 3`.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "extension/SinCAttention.h"

using word = uint32_t;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::Plaintext;
using cheddar::SinCAttention;
using Ring = ringfixture::Ring<word>;

namespace {

constexpr int kTokens = 128;
constexpr int kHeadDim = 128;
constexpr int kLanes = 16;
constexpr int kKvHeads = 4;
constexpr int kGqaGroup = kLanes / kKvHeads;
constexpr int kSubDegree = 32;
constexpr int kPhases = 3;

int BitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? std::atoi(e) : fallback;
}

const char *Env(const char *name, const char *fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? e : fallback;
}

// Q's packing, and V's, and K's -- the three the block will use. See
// AttentionTransportTest for why each is what it is; the short version is that
// Q spends its ciphertext axis on the low three bits of the channel so RoPE's
// pair stays inside one ciphertext, K spends nothing on it because the
// exchange takes it for the key, and V spends nothing on it either and puts
// c's low three bits where the swap will drop them into the exchange.
void LocateQuery(int head, int c, int token, int &ct, int &slot) {
  ct = (head / kLanes) * (kHeadDim / 8) + BitRev(c & 7, 3);
  slot = ((c >> 3) * kLanes + (head % kLanes)) * kTokens + token;
}
void LocateKey(int kv, int c, int key, int &ct, int &slot) {
  ct = kv & 1;
  slot = (c * 2 + (kv >> 1)) * kTokens + key;
}
void LocateValue(int kv, int c, int key, int &ct, int &slot) {
  ct = kv & 1;
  slot = ((c >> 3) * kLanes + ((kv >> 1) & 1) * 8 + (c & 7)) * kTokens + key;
}

// The score layout, which is where SoftMax works and therefore where P is
// handed back: ciphertext BitRev3(key & 7), slot [key>>3 | query | head].
void LocateScore(int head, int query, int key, int &ct, int &slot) {
  ct = BitRev(key & 7, 3);
  slot = ((key >> 3) << 11) | (query << 4) | head;
}
// The output layout, which is Q's own after its [4|7] swap.
void LocateOutput(int head, int query, int c, int &ct, int &slot) {
  ct = BitRev(c & 7, 3);
  slot = ((c >> 3) << 11) | (query << 4) | head;
}

// Fit the constant the chain multiplies by, then report what is left. The
// constant is a property of the parameter set, not of the data.
struct Fit {
  Complex constant;
  double residual;
  double biggest;
};

Fit Compare(const std::vector<std::vector<Complex>> &got,
            const std::vector<double> &want,
            void (*locate)(int, int, int, int &, int &), int dim_b) {
  Complex sum(0.0, 0.0);
  int counted = 0;
  double biggest = 0.0;
  for (int head = 0; head < kLanes; head++) {
    for (int a = 0; a < kTokens; a++) {
      for (int b = 0; b < dim_b; b++) {
        const double w =
            want[(static_cast<size_t>(head) * kTokens + a) * dim_b + b];
        biggest = std::max(biggest, std::abs(w));
        if (std::abs(w) < 1e-3) continue;
        int ct, slot;
        locate(head, a, b, ct, slot);
        sum += got[ct][slot] / w;
        counted++;
      }
    }
  }
  const Complex constant = sum / static_cast<double>(counted);
  double worst = 0.0;
  for (int head = 0; head < kLanes; head++) {
    for (int a = 0; a < kTokens; a++) {
      for (int b = 0; b < dim_b; b++) {
        const double w =
            want[(static_cast<size_t>(head) * kTokens + a) * dim_b + b];
        int ct, slot;
        locate(head, a, b, ct, slot);
        worst = std::max(worst, std::abs(got[ct][slot] / constant - w));
      }
    }
  }
  return {constant, worst, biggest};
}

}  // namespace

TEST(SinCAttention, TheTwoProductsRunSlotsToSlots) {
  const int kSwapLevel = EnvInt("CHEDDAR_SWAP_LEVEL", 15);
  const int kSinCLevel = EnvInt("CHEDDAR_SINC_LEVEL", 12);
  const int kProductLevel = 1;

  Ring big(Env("CHEDDAR_BLOCK_PARAM", "sylphflow16_35.json"), {}, 8);
  Ring sw(Env("CHEDDAR_SWITCH_PARAM", "ringswitch16_35.json"),
          big.ui->GetSecretCoeffs());
  Ring small(Env("CHEDDAR_SMALL_PARAM", "ringdegree12_35.json"));
  const int num_slots = big.Degree() / 2;

  auto boot = std::dynamic_pointer_cast<BootContext<word>>(big.context);
  ASSERT_NE(boot, nullptr);
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  boot->PrepareSinC(num_slots, kSubDegree, kSinCLevel, kSinCLevel, kPhases);

  SinCAttention<word>::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.head_dim = kHeadDim;
  cfg.lanes = kLanes;
  cfg.gqa_group = kGqaGroup;
  cfg.sub_degree = kSubDegree;
  cfg.sinc_phases = kPhases;
  cfg.product_level = kProductLevel;
  cfg.swap_level = kSwapLevel;
  cfg.sinc_level = kSinCLevel;
  cfg.prefix_level = boot->GetBootParameter().GetEvalModEndLevel();
  cfg.halfboot_scale = boot->GetStCInputScale();
  cfg.verbose = true;
  std::cout << "swap at " << cfg.swap_level << "/" << cfg.swap_level - 1
            << ", exchange at " << cfg.swap_level - 2 << ", SinC at "
            << kSinCLevel << ".." << kSinCLevel - kPhases + 1 << ", product at "
            << kProductLevel << ", HalfBoot lands at " << cfg.prefix_level
            << ", prefix leaves " << cfg.prefix_level - 1 << std::endl;

  SinCAttention<word> attn(boot, sw.context, small.context, cfg);
  const int num_cts = attn.GetNumCiphertexts();
  ASSERT_EQ(num_cts, 8);

  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  attn.AddRequiredRotations(req);
  size_t free_before = 0, total = 0;
  cudaMemGetInfo(&free_before, &total);
  big.ui->PrepareRotationKey(req);
  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total);
  std::cout << "big-ring keys: " << (free_before - free_after) / (1 << 20)
            << " MiB, " << free_after / (1 << 20) << " MiB free after"
            << std::endl;

  std::cout << "step: ring switch keys" << std::endl;
  sw.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                              kProductLevel);
  sw.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                     small.ui->GetSecretCoeffs(),
                                     kProductLevel);
  std::cout << "step: small-ring keys" << std::endl;
  for (int idx : attn.SmallRotationIndices()) {
    small.ui->PrepareRotationKey(idx, kProductLevel);
  }
  typename SinCAttention<word>::Keys keys;
  keys.big = &big.ui->GetEvkMap();
  keys.small = &small.ui->GetEvkMap();
  keys.ring_switch = &sw.ui->GetRingSwitchKey(attn.GetLayout().rank);
  keys.inverse_ring_switch =
      &sw.ui->GetInverseRingSwitchKey(attn.GetLayout().rank);

  // ---- the tensors ------------------------------------------------------
  std::mt19937_64 gen(0x5C1AA);
  std::uniform_real_distribution<double> d(-0.08, 0.08);
  std::vector<double> q(static_cast<size_t>(kLanes) * kTokens * kHeadDim);
  std::vector<double> k(static_cast<size_t>(kKvHeads) * kTokens * kHeadDim);
  std::vector<double> v(k.size());
  for (auto &z : q) z = d(gen);
  for (auto &z : k) z = d(gen);
  for (auto &z : v) z = d(gen);
  auto at = [](const std::vector<double> &t, int a, int b, int c) {
    return t[(static_cast<size_t>(a) * kTokens + b) * kHeadDim + c];
  };

  auto encrypt = [&](const std::vector<std::vector<Complex>> &msg, int level,
                     std::vector<Ciphertext<word>> &out) {
    out.resize(msg.size());
    for (size_t i = 0; i < msg.size(); i++) {
      Plaintext<word> pt;
      big.context->encoder_.Encode(pt, level, big.param->GetScale(level),
                                   msg[i]);
      big.ui->Encrypt(out[i], pt);
    }
  };

  std::vector<std::vector<Complex>> q_msg(
      num_cts, std::vector<Complex>(num_slots, Complex(0)));
  std::vector<std::vector<Complex>> k_msg(
      2, std::vector<Complex>(num_slots, Complex(0)));
  std::vector<std::vector<Complex>> v_msg(
      2, std::vector<Complex>(num_slots, Complex(0)));
  for (int head = 0; head < kLanes; head++) {
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kHeadDim; c++) {
        int ct, slot;
        LocateQuery(head, c, t, ct, slot);
        q_msg[ct][slot] = Complex(at(q, head, t, c), 0.0);
      }
    }
  }
  for (int kv = 0; kv < kKvHeads; kv++) {
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kHeadDim; c++) {
        int ct, slot;
        LocateKey(kv, c, t, ct, slot);
        k_msg[ct][slot] = Complex(at(k, kv, t, c), 0.0);
        LocateValue(kv, c, t, ct, slot);
        v_msg[ct][slot] = Complex(at(v, kv, t, c), 0.0);
      }
    }
  }

  std::cout << "step: encrypt the operands" << std::endl;
  std::vector<Ciphertext<word>> q_ct, k_ct, v_ct;
  encrypt(q_msg, kSwapLevel, q_ct);
  encrypt(k_msg, kSwapLevel, k_ct);
  encrypt(v_msg, kSwapLevel, v_ct);

  // ---- Q K^T, slots to slots -------------------------------------------
  std::cout << "step: Scores" << std::endl;
  std::vector<Ciphertext<word>> scores;
  attn.Scores(scores, q_ct, k_ct, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(scores.size()), num_cts);
  for (const auto &c : scores) {
    EXPECT_EQ(big.param->NPToLevel(c.GetNP()), attn.GetOutputLevel());
    EXPECT_NEAR(c.GetScale() / big.param->GetScale(attn.GetOutputLevel()), 1.0,
                1e-6)
        << "the prefix did not put the result on its level's canonical scale, "
           "which EvalPoly aborts on a few operators later";
  }

  std::vector<std::vector<Complex>> got(num_cts);
  for (int i = 0; i < num_cts; i++) {
    Plaintext<word> pt;
    big.ui->Decrypt(pt, scores[i]);
    big.context->encoder_.Decode(got[i], pt);
  }
  scores.clear();

  std::vector<double> want_s(static_cast<size_t>(kLanes) * kTokens * kTokens);
  std::vector<double> want_t(want_s.size());
  for (int head = 0; head < kLanes; head++) {
    for (int query = 0; query < kTokens; query++) {
      for (int key = 0; key < kTokens; key++) {
        double s = 0.0, t = 0.0;
        for (int c = 0; c < kHeadDim; c++) {
          s += at(q, head, query, c) * at(k, head / kGqaGroup, key, c);
          t += at(q, head, key, c) * at(k, head / kGqaGroup, query, c);
        }
        const size_t o =
            (static_cast<size_t>(head) * kTokens + query) * kTokens + key;
        want_s[o] = s;
        want_t[o] = t;
      }
    }
  }
  const Fit fs = Compare(got, want_s, LocateScore, kTokens);
  const Fit ft = Compare(got, want_t, LocateScore, kTokens);
  std::cout << "Q K^T slots to slots: constant " << fs.constant
            << ", residual " << fs.residual << " against |want| <= "
            << fs.biggest << std::endl;
  std::cout << "  control (transposed): residual " << ft.residual << std::endl;
  EXPECT_LT(fs.residual, 5e-3)
      << "the score product did not come back where SoftMax reads it";
  EXPECT_GT(ft.residual, 1e-2)
      << "the scores are symmetric enough that a transpose would pass";
  EXPECT_NEAR(fs.constant.real(), 1.0, 0.05)
      << "the prefix carries Canonicalise, so the chain should be the identity "
         "and not merely proportional to it";
  EXPECT_LT(std::abs(fs.constant.imag()), 0.05);

  // ---- P V, slots to slots ----------------------------------------------
  //
  // P is encoded where SoftMax leaves it, which is the whole claim: the Values
  // product reads it with no permutation of any kind. Causal and
  // row-stochastic, with at least half the row valid so no entry dominates --
  // the batch CC-MM accumulates d terms inside the polynomial product and the
  // level-1 ring has no headroom for a spike.
  std::uniform_real_distribution<double> weight(-1.0, 1.0);
  std::vector<double> p(static_cast<size_t>(kLanes) * kTokens * kTokens, 0.0);
  for (int head = 0; head < kLanes; head++) {
    for (int query = 0; query < kTokens; query++) {
      const int valid = kTokens / 2 + (query % (kTokens / 2)) + 1;
      const size_t base =
          (static_cast<size_t>(head) * kTokens + query) * kTokens;
      double sum = 0.0;
      for (int key = 0; key < valid; key++) {
        const double w = std::exp(weight(gen));
        p[base + key] = w;
        sum += w;
      }
      for (int key = 0; key < valid; key++) p[base + key] /= sum;
    }
  }
  std::vector<std::vector<Complex>> p_msg(
      num_cts, std::vector<Complex>(num_slots, Complex(0)));
  for (int head = 0; head < kLanes; head++) {
    for (int query = 0; query < kTokens; query++) {
      for (int key = 0; key < kTokens; key++) {
        int ct, slot;
        LocateScore(head, query, key, ct, slot);
        p_msg[ct][slot] = Complex(
            p[(static_cast<size_t>(head) * kTokens + query) * kTokens + key],
            0.0);
      }
    }
  }
  std::vector<Ciphertext<word>> p_ct;
  encrypt(p_msg, kSinCLevel, p_ct);

  std::cout << "step: Values" << std::endl;
  std::vector<Ciphertext<word>> out;
  attn.Values(out, p_ct, v_ct, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(out.size()), num_cts);

  for (int i = 0; i < num_cts; i++) {
    EXPECT_EQ(big.param->NPToLevel(out[i].GetNP()), attn.GetOutputLevel());
    Plaintext<word> pt;
    big.ui->Decrypt(pt, out[i]);
    big.context->encoder_.Decode(got[i], pt);
  }
  out.clear();

  std::vector<double> want_o(static_cast<size_t>(kLanes) * kTokens * kHeadDim);
  std::vector<double> want_g(want_o.size());  // the un-replicated control
  for (int head = 0; head < kLanes; head++) {
    for (int query = 0; query < kTokens; query++) {
      for (int c = 0; c < kHeadDim; c++) {
        double o = 0.0, g = 0.0;
        for (int key = 0; key < kTokens; key++) {
          const double pr =
              p[(static_cast<size_t>(head) * kTokens + query) * kTokens + key];
          o += pr * at(v, head / kGqaGroup, key, c);
          g += pr * at(v, head % kKvHeads, key, c);
        }
        const size_t idx =
            (static_cast<size_t>(head) * kTokens + query) * kHeadDim + c;
        want_o[idx] = o;
        want_g[idx] = g;
      }
    }
  }
  const Fit fo = Compare(got, want_o, LocateOutput, kHeadDim);
  const Fit fg = Compare(got, want_g, LocateOutput, kHeadDim);
  std::cout << "P V slots to slots: constant " << fo.constant << ", residual "
            << fo.residual << " against |want| <= " << fo.biggest << std::endl;
  std::cout << "  control (lanes not replicated by KV head): residual "
            << fg.residual << std::endl;
  EXPECT_LT(fo.residual, 5e-3)
      << "the attention output did not come back in the Q operand's own slot "
         "layout, so the block does not close on itself";
  EXPECT_GT(fg.residual, 1e-3)
      << "every lane holds the same values, so this is not checking GQA";
  EXPECT_NEAR(fo.constant.real(), 1.0, 0.05);
  EXPECT_LT(std::abs(fo.constant.imag()), 0.05);
}
