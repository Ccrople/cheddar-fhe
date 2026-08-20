// The attention path's transport, from the block's own packing all the way
// into a batch CC-MM and back out as scores:
//
//   slots @ 65536, block packing
//     --SlotPermute [4|7], as [4|4]@3 then [4|3]-->   2 levels
//     --SlotToSinC(32), 11 stages in 3 phases-->      3 levels
//     --LevelDown to the product level-->             free
//     --RingSwitch + [KANG] Alg. 4 + SwitchBack-->    1 level
//   = Q K^T, per head
//
// WHAT THIS IS FOR, AND WHY THE PERMUTATION IS THE ONLY ONE LEFT. [SYLPH] 3.2
// picks its layout so that the bit reversal on the way to coefficients and the
// partial one on the way to SinC land the data where the next algorithm wants
// it. It does not remove every conversion -- the transpose survives -- and this
// is where the surviving one is paid for.
//
// The block holds a tensor as `slot = channel_in_ct * T + token`, so the token
// owns the LOW slot bits. `SwitchedCcmmLayout` wants
// `[BitRev(column mod rank) | BitRev(row) | lane]`, so the row -- which for Q
// is the token -- sits in the MIDDLE and the lane in the low bits. Exchanging a
// 4-bit field with a 7-bit one is exactly `SwapAdjacentFields`, and
// `SlotPermuteTest` already showed the wide swap equals the two narrow ones it
// decomposes into. That is the whole permutation.
//
// AND THE TWO BIT REVERSALS CANCEL RATHER THAN BEING PAID FOR. The layout asks
// for `BitRev(row)` in the slot index and the block delivers the token
// unreversed, which looks like a second permutation to buy. It is not: the row
// index is ours to name, and naming it `row = BitRev(query)` makes
// `BitRev(row) = query` and the reversal disappear. What comes back is the
// score matrix with its query axis permuted, which SoftMax -- a per-row
// operator that reduces over the whole key axis -- cannot tell from the
// unpermuted one. The same trick names the contraction index `BitRev(key)` so
// that V, the other tensor whose token axis is a matrix row, needs no reversal
// either. This test uses the naming, so if it were wrong the product would come
// out permuted and the entrywise comparison would say so.
//
// THE CHANNEL ASSIGNMENT IS THE OTHER HALF, AND IT IS FREE. Which output
// channel of the QKV projection lands in which ciphertext and which slot is a
// permutation of the weight matrix's columns, so the block gets to choose it.
// This test uses the choice the block will use: the ciphertext index carries
// the LOW three bits of the channel-within-head and `channel_in_ct` is
// `[c >> 3 | head]`, so that RoPE's pair (c, c + 64) -- which differs only in
// `c >> 3` -- stays inside one ciphertext.
//
// WHERE IT IS COMPILED. The transforms run at level 11 and below rather than
// straight under the bootstrap. Nothing needs them higher -- `LevelDown` to 11
// is free -- and every plaintext and every rotation key is proportional to the
// limb count, so this is several gigabytes of key material rather than a dozen.
// The floor is level 7: `SlotPermuteTest` and `SinCTransformTest` both measured
// a standalone LinearTransform going wrong below it.
//
// THREE CONTEXTS, ONE SECRET. sylphflow16_35 holds the block, ringswitch16_35
// hosts the switching key switch because its alpha is small enough for the
// small ring's budget, and ringdegree12_35 is the product ring. The first two
// share their primes at levels 0 and 1, so a ciphertext crosses between them
// with no words changing -- but only if they also share a SECRET, which is what
// `UserInterface(context, secret_coeffs)` is for. That crossing has never been
// exercised before; the test asserts the NPInfo matches before relying on it.
//
// AND THE SCALE TAKES CARE OF ITSELF, WHICH IS NOT OBVIOUS. A grafted ladder
// does not hold one scale at every level -- sylphflow16_35 is 2^35 from level 3
// up and 2^29.88 at level 1, where the product runs -- so an operand walking
// down five levels looks like it should arrive five bits off the value that
// ring defines. It does not: `Context::LevelDown` is a chain of multiply-by-one
// and rescale whose constants are encoded at each level's own scale, so it
// lands on the canonical value of wherever it stops. Correcting the transform's
// output scale instead is not merely unnecessary, it breaks that chain, because
// the constants assume a canonical input. Measured, at a cost of one run.
//
// K COSTS ONE MOVE MORE THAN Q, AND IT IS NOT A PERMUTATION. Q's operand wants
// the channel on the column axis, which is where the projection can put it;
// K's wants the KEY there, three of whose bits therefore land on the
// ciphertext axis. No LinearTransform can move them: it is a map inside one
// ciphertext. `CtAxisExchange` is that move, and K's chain is
//
//   2 cts, slot = [c | kv1 | key]
//     --[4|4]@3 then [4|4]@7, i.e. the [8|4]@3 swap-->      2 levels
//   2 cts, slot = [key>>3 | c | kv1 | key&7]
//     --CtAxisExchange, three bits at offset 0-->           1 level
//   8 cts, ct = key&7, slot = [key>>3 | c | kv1 | kv0 | rep]
//     --SlotToSinC(32)-->                                   3 levels
//
// GQA RIDES ALONG FREE. Llama-3-8B has 8 KV heads against 32 query heads, so
// this call group's 16 lanes need 4 KV heads four times each -- 2 source
// ciphertexts becoming 8. The exchange takes a source array shorter than its
// output, and the copy index lands in the LOW bits of the exchanged field,
// which after the exchange is where the lane index wants it: the four lanes
// sharing a KV head are adjacent. No rotation is added for it.
//
// AND THE NAMING OF THE KEY AXIS IS WHAT MAKES BOTH ENDS WORK. The column
// index of the right operand is ours to name, and the choice is
// `column = BitRev7(key)` -- the same reversal Q's channel axis uses. It puts
// `BitRev4(column mod 16) = key >> 3` in the high slot bits, which is where
// the swap leaves it, and `column / 16 = BitRev3(key & 7)` on the ciphertext
// axis, which is a relabelling of the exchange's output array and therefore
// free. The same choice is what will let the Values product read the scores
// with no permutation at all: `BitRev7(column)` is then `key` itself.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/CtAxisExchange.h"
#include "core/SwitchedCcmm.h"
#include "extension/SlotPermute.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::CtAxisExchange;
using cheddar::EvkRequest;
using cheddar::Plaintext;
using cheddar::SlotPermute;
using cheddar::SwapAdjacentFields;
using cheddar::SwitchedCcmmHandler;
using cheddar::SwitchedCcmmLayout;
using Ring = ringfixture::Ring<word>;

namespace {

// Llama-3-8B, one call group.
constexpr int kTokens = 128;    // T
constexpr int kHeadDim = 128;   // per-head channels
constexpr int kLanes = 16;      // heads per CC-MM call
constexpr int kSubDegree = 32;  // k: 4096/32 = 128 wide, 16 lanes
constexpr int kKvHeads = 4;     // KV heads this call group draws on
constexpr int kGqaGroup = kLanes / kKvHeads;  // query heads per KV head

// Where the transforms are compiled. The permutation takes two levels and the
// SinC transform three, and every standalone LinearTransform has a floor
// somewhere -- SlotPermuteTest put it at 7 on bootparam_35. Both are settable
// from the environment because finding that floor on a new ladder is a sweep,
// not a derivation.
constexpr int kSinCPhases = 3;
constexpr int kProductLevel = 1;

int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? std::atoi(e) : fallback;
}

int BitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

const char *Env(const char *name, const char *fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? e : fallback;
}

// THE BLOCK'S PACKING, and the channel assignment the QKV projection folds in.
//
// A tensor of `heads * head_dim` channels over T tokens is held as
// `slot = channel_in_ct * T + token` in `channels / (slots/T)` ciphertexts.
// Which (head, c) pair is which channel is the weight matrix's business, and
// this is the choice:
//
//     ciphertext    = head/lanes * (head_dim/8) + BitRev3(c & 7)
//     channel_in_ct = (c >> 3) * 16 + (head mod lanes)
//
// so that after the [4|7] swap the slot index reads `[c >> 3 | token | head]`,
// which is `[BitRev4(column mod 16) | BitRev(row) | lane]` for
// `column = BitRev7(c)` -- the contraction index -- and so that c and c + 64,
// RoPE's pair, differ only in `c >> 3` and therefore stay inside one
// ciphertext. Choosing the contraction index to be BitRev7(c) rather than c is
// what buys that last property: with `column = c` the pair would straddle two
// ciphertexts and RoPE could not run.
struct BlockPacking {
  int tokens = kTokens;
  int slots = 0;

  int ChannelsPerCt() const { return slots / tokens; }

  void Locate(int head, int c, int token, int &ct, int &slot) const {
    const int channel_in_ct = (c >> 3) * kLanes + (head % kLanes);
    ct = (head / kLanes) * (kHeadDim / 8) + BitRev(c & 7, 3);
    slot = channel_in_ct * tokens + token;
  }
};

// K'S PACKING, and why it is not Q's.
//
//     ciphertext    = kv mod 2
//     channel_in_ct = c * 2 + kv / 2
//
// so the slot index reads `[c (8..14) | kv1 (7) | key (0..6)]` and this call
// group's four KV heads fill exactly two ciphertexts. K spends NOTHING on the
// ciphertext axis, because after the exchange that axis belongs to the key --
// which also means the whole channel index stays in the slots, so RoPE's pair
// (c, c + 64) is inside one ciphertext for free rather than by construction.
struct KeyPacking {
  int tokens = kTokens;

  void Locate(int kv, int c, int key, int &ct, int &slot) const {
    ct = kv & 1;
    slot = (c * 2 + (kv >> 1)) * tokens + key;
  }
};

}  // namespace

TEST(AttentionTransport, CarriesTheBlockPackingIntoAScoreProduct) {
  const int kSinCLevel = EnvInt("CHEDDAR_SINC_LEVEL", 9);
  const int kPermuteLevel = EnvInt("CHEDDAR_PERMUTE_LEVEL", kSinCLevel + 2);
  // K's leg is one move longer than Q's and has to arrive at the same place,
  // so it starts a level higher: two swaps and then the exchange.
  const int kKeyLevel = kPermuteLevel + 1;
  std::cout << "permute at " << kPermuteLevel << "/" << kPermuteLevel - 1
            << ", K at " << kKeyLevel << "/" << kKeyLevel - 1 << "/"
            << kKeyLevel - 2 << ", SinC at " << kSinCLevel << ".."
            << kSinCLevel - kSinCPhases + 1 << ", product at " << kProductLevel
            << std::endl;

  // ---- the three rings, and one secret between the two big ones ----------
  Ring big(Env("CHEDDAR_BLOCK_PARAM", "sylphflow16_35.json"), {}, 8);
  Ring sw(Env("CHEDDAR_SWITCH_PARAM", "ringswitch16_35.json"),
          big.ui->GetSecretCoeffs());
  Ring small(Env("CHEDDAR_SMALL_PARAM", "ringdegree12_35.json"));

  const int degree = big.Degree();
  const int num_slots = degree / 2;
  ASSERT_EQ(sw.Degree(), degree);

  SwitchedCcmmLayout layout(degree, small.Degree(), kSubDegree);
  ASSERT_EQ(layout.dim, kTokens);
  ASSERT_EQ(layout.dim, kHeadDim);
  ASSERT_EQ(layout.lanes, kLanes);
  ASSERT_EQ(layout.num_cts, 8);

  BlockPacking pack;
  pack.slots = num_slots;
  ASSERT_EQ(pack.ChannelsPerCt(), 256);

  // The two Contexts must agree on the primes at the product level, or the
  // crossing below is meaningless. This is the assertion the whole three-ring
  // arrangement rests on and nothing had ever checked it.
  ASSERT_EQ(big.param->LevelToNP(kProductLevel),
            sw.param->LevelToNP(kProductLevel))
      << "the block's ring and the switching ring disagree at the product "
         "level, so a ciphertext cannot cross";

  // ...and they must agree on the SECRET, which is the half that has no
  // structural check anywhere. One ciphertext encrypted on one and decrypted on
  // the other says so in a millisecond; without it every failure downstream
  // looks like a layout mistake.
  {
    std::vector<Complex> probe(num_slots);
    for (int i = 0; i < num_slots; i++) {
      probe[i] = Complex(0.5 - 1.0 * ((i * 7919) % 1000) / 1000.0, 0.0);
    }
    Plaintext<word> pt;
    big.context->encoder_.Encode(pt, kProductLevel,
                                 big.param->GetScale(kProductLevel), probe);
    Ciphertext<word> ct;
    big.ui->Encrypt(ct, pt);
    Plaintext<word> back;
    sw.ui->Decrypt(back, ct);
    std::vector<Complex> got;
    sw.context->encoder_.Decode(got, back);
    double worst = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max(worst, std::abs(got[i] - probe[i]));
    }
    std::cout << "crossing block -> switching ring: max |diff| = " << worst
              << std::endl;
    ASSERT_LT(worst, 1e-3)
        << "the two big Contexts do not share a secret, so nothing below this "
           "means anything";
  }

  auto boot = std::dynamic_pointer_cast<cheddar::BootContext<word>>(big.context);
  ASSERT_NE(boot, nullptr);
  boot->PrepareEvalSpecialFFT(num_slots);
  boot->PrepareSinC(num_slots, kSubDegree, kSinCLevel, kSinCLevel,
                    kSinCPhases);

  // ---- the permutation, as the two narrow swaps ------------------------
  //
  // `[4 | 7]` in one transform is 2048 diagonals and therefore 2048 plaintexts;
  // `[4|4]` at offset 3 followed by `[4|3]` is the same permutation for 31 and
  // 127, at the cost of one more level. SlotPermuteTest checks the equality
  // exactly.
  const std::vector<int> step_a = SwapAdjacentFields(num_slots, 4, 4, 3);
  const std::vector<int> step_b = SwapAdjacentFields(num_slots, 3, 4);
  std::vector<int> wide(num_slots);
  for (int t = 0; t < num_slots; t++) wide[t] = step_b[step_a[t]];
  {
    const std::vector<int> direct = SwapAdjacentFields(num_slots, 7, 4);
    int mismatch = 0;
    for (int t = 0; t < num_slots; t++) {
      if (direct[t] != wide[t]) mismatch++;
    }
    ASSERT_EQ(mismatch, 0) << "the two narrow swaps are not the wide one";
  }
  SlotPermute<word> swap_a(big.context, step_a, kPermuteLevel);
  SlotPermute<word> swap_b(big.context, step_b, kPermuteLevel - 1);
  std::cout << "permute step 1: " << swap_a.GetNumDiagonals()
            << " diagonals, stride " << swap_a.GetStride() << ", BSGS "
            << swap_a.GetBS() << "x" << swap_a.GetGS() << ", shift "
            << swap_a.GetShift() << std::endl;
  std::cout << "permute step 2: " << swap_b.GetNumDiagonals()
            << " diagonals, stride " << swap_b.GetStride() << ", BSGS "
            << swap_b.GetBS() << "x" << swap_b.GetGS() << ", shift "
            << swap_b.GetShift() << std::endl;

  // ---- K's leg: the [8|4]@3 swap, then the cross-ciphertext exchange ----
  //
  // `[8|4]@3` in one transform is 511 diagonals; as two `[4|4]` swaps it is
  // 31 and 31, for one more level -- the same trade `[4|7]` makes above, and
  // the level is there because K's leg starts one higher anyway.
  const std::vector<int> k_step_a = SwapAdjacentFields(num_slots, 4, 4, 3);
  const std::vector<int> k_step_b = SwapAdjacentFields(num_slots, 4, 4, 7);
  {
    std::vector<int> composed(num_slots);
    for (int t = 0; t < num_slots; t++) composed[t] = k_step_b[k_step_a[t]];
    const std::vector<int> direct = SwapAdjacentFields(num_slots, 4, 8, 3);
    int mismatch = 0;
    for (int t = 0; t < num_slots; t++) {
      if (direct[t] != composed[t]) mismatch++;
    }
    ASSERT_EQ(mismatch, 0)
        << "the two [4|4] swaps do not compose to the [8|4] one";
  }
  SlotPermute<word> k_swap_a(big.context, k_step_a, kKeyLevel);
  SlotPermute<word> k_swap_b(big.context, k_step_b, kKeyLevel - 1);
  CtAxisExchange<word> exchange(big.context, /*log_num_cts=*/3,
                                /*field_offset=*/0, kKeyLevel - 2,
                                /*log_num_src_cts=*/1);
  std::cout << "K swap 1: " << k_swap_a.GetNumDiagonals() << " diagonals, K "
            << "swap 2: " << k_swap_b.GetNumDiagonals()
            << " diagonals; exchange: " << exchange.GetNumSrcCts() << " -> "
            << exchange.GetNumCts() << " cts, "
            << exchange.RotationIndices().size() << " rotations, "
            << exchange.GetNumMults() << " plaintext mults" << std::endl;

  EvkRequest req;
  swap_a.AddRequiredRotations(req);
  swap_b.AddRequiredRotations(req);
  k_swap_a.AddRequiredRotations(req);
  k_swap_b.AddRequiredRotations(req);
  exchange.AddRequiredRotations(req);
  boot->AddRequiredSinCRotations(req, num_slots);
  size_t free_before = 0, total = 0;
  cudaMemGetInfo(&free_before, &total);
  big.ui->PrepareRotationKey(req);
  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total);
  std::cout << "rotation keys: " << (free_before - free_after) / (1 << 20)
            << " MiB, " << free_after / (1 << 20) << " MiB free after"
            << std::endl;

  // ---- the tensors ------------------------------------------------------
  std::mt19937_64 gen(0xA77E4);
  std::uniform_real_distribution<double> dist(-0.08, 0.08);
  // q[head][token][c] over 16 query heads and k[kv][token][c] over the 4 KV
  // heads they share -- Llama-3-8B's grouped-query attention, at the real
  // ratio. Lane `head` reads KV head `head / 4`, and that fourfold replication
  // is carried by the exchange rather than by the projection.
  std::vector<double> q(static_cast<size_t>(kLanes) * kTokens * kHeadDim);
  std::vector<double> k(static_cast<size_t>(kKvHeads) * kTokens * kHeadDim);
  for (auto &v : q) v = dist(gen);
  for (auto &v : k) v = dist(gen);
  auto at = [](const std::vector<double> &t, int head, int token, int c) {
    return t[(static_cast<size_t>(head) * kTokens + token) * kHeadDim + c];
  };

  // ---- Q, through the whole transport -----------------------------------
  const int num_q_cts = kLanes * kHeadDim / pack.ChannelsPerCt();
  ASSERT_EQ(num_q_cts, 8);
  std::vector<std::vector<Complex>> q_slots(
      num_q_cts, std::vector<Complex>(num_slots, Complex(0)));
  for (int head = 0; head < kLanes; head++) {
    for (int token = 0; token < kTokens; token++) {
      for (int c = 0; c < kHeadDim; c++) {
        int ct, slot;
        pack.Locate(head, c, token, ct, slot);
        ASSERT_LT(ct, num_q_cts);
        q_slots[ct][slot] = Complex(at(q, head, token, c), 0.0);
      }
    }
  }

  std::vector<Ciphertext<word>> q_operand(num_q_cts);
  for (int i = 0; i < num_q_cts; i++) {
    Plaintext<word> pt;
    big.context->encoder_.Encode(pt, kPermuteLevel,
                                 big.param->GetScale(kPermuteLevel),
                                 q_slots[i]);
    Ciphertext<word> ct;
    big.ui->Encrypt(ct, pt);

    Ciphertext<word> a, b, sinc;
    swap_a.Evaluate(big.context, a, ct, big.ui->GetEvkMap());
    swap_b.Evaluate(big.context, b, a, big.ui->GetEvkMap());
    ASSERT_EQ(big.param->NPToLevel(b.GetNP()), kPermuteLevel - 2);

    // Stage by stage on the first ciphertext only: three transforms compose
    // here and each fails the same way, so the ledger is worth one decryption.
    if (i == 0) {
      auto slot_check = [&](const Ciphertext<word> &c,
                            const std::vector<int> &perm, const char *name) {
        Plaintext<word> pt;
        big.ui->Decrypt(pt, c);
        std::vector<Complex> got;
        big.context->encoder_.Decode(got, pt);
        double worst = 0.0, biggest = 0.0;
        for (int t = 0; t < num_slots; t++) {
          biggest = std::max(biggest, std::abs(got[t]));
          worst = std::max(worst, std::abs(got[perm[t]] - q_slots[0][t]));
        }
        std::cout << "  " << name << ": max |diff| = " << worst
                  << ", |got| <= " << biggest << std::endl;
      };
      slot_check(a, step_a, "after swap [4|4]@3");
      slot_check(b, wide, "after swap [4|3] (composed)");
    }

    boot->SlotToSinC(sinc, num_slots, b, big.ui->GetEvkMap());
    ASSERT_EQ(big.param->NPToLevel(sinc.GetNP()), kSinCLevel - kSinCPhases);
    if (i == 0) {
      Plaintext<word> pt;
      big.ui->Decrypt(pt, sinc);
      std::vector<Complex> got;
      big.context->encoder_.DecodeSinC(got, pt, kSubDegree);
      double worst = 0.0, biggest = 0.0;
      for (int t = 0; t < num_slots; t++) {
        biggest = std::max(biggest, std::abs(got[t]));
      }
      for (int head = 0; head < kLanes; head++) {
        for (int query = 0; query < kTokens; query++) {
          for (int c = 0; c < kHeadDim; c++) {
            int ct_i, index;
            layout.LocateSinC(BitRev(query, 7), BitRev(c, 7), head, ct_i,
                              index);
            if (ct_i != 0) continue;
            worst = std::max(
                worst, std::abs(got[index].real() - at(q, head, query, c)));
          }
        }
      }
      std::cout << "  after SlotToSinC: max |diff| = " << worst
                << ", |got| <= " << biggest << std::endl;
    }
    big.context->LevelDown(q_operand[i], sinc, kProductLevel);
    // LevelDown is five multiply-and-rescale steps and each one rounds, so
    // the canonical value is met to about 3e-4 rather than exactly.
    EXPECT_NEAR(q_operand[i].GetScale() / big.param->GetScale(kProductLevel),
                1.0, 1e-3)
        << "the operand arrives at the product off the canonical scale";
  }
  ASSERT_EQ(big.param->LevelToNP(kProductLevel), q_operand[0].GetNP())
      << "the operand is not at the level the product expects";

  // The transport, on its own, before the product runs on it. Splitting the
  // two is worth a decryption: a layout mistake and a crossing mistake produce
  // the same garbage at the end and have nothing in common as bugs.
  {
    double worst = 0.0, biggest = 0.0;
    for (int i = 0; i < num_q_cts; i++) {
      Plaintext<word> pt;
      big.ui->Decrypt(pt, q_operand[i]);
      std::vector<Complex> got;
      big.context->encoder_.DecodeSinC(got, pt, kSubDegree);
      for (const auto &z : got) biggest = std::max(biggest, std::abs(z));
      for (int head = 0; head < kLanes; head++) {
        for (int query = 0; query < kTokens; query++) {
          for (int c = 0; c < kHeadDim; c++) {
            int ct, index;
            layout.LocateSinC(BitRev(query, 7), BitRev(c, 7), head, ct, index);
            if (ct != i) continue;
            worst = std::max(worst,
                             std::abs(got[index].real() -
                                      at(q, head, query, c)));
          }
        }
      }
    }
    std::cout << "Q operand after permute + SinC + LevelDown: max |diff| = "
              << worst << ", |got| <= " << biggest << std::endl;
    EXPECT_LT(worst, 1e-3)
        << "the transport did not put Q where SwitchedCcmmLayout says it goes";
  }

  // ---- K, through the swap, the exchange and the SinC transform ---------
  const double product_scale = sw.param->GetScale(kProductLevel);
  KeyPacking kpack;
  const int num_k_src = kKvHeads * kHeadDim / pack.ChannelsPerCt();
  ASSERT_EQ(num_k_src, 2);
  std::vector<std::vector<Complex>> k_slots(
      num_k_src, std::vector<Complex>(num_slots, Complex(0)));
  for (int kv = 0; kv < kKvHeads; kv++) {
    for (int key = 0; key < kTokens; key++) {
      for (int c = 0; c < kHeadDim; c++) {
        int ct, slot;
        kpack.Locate(kv, c, key, ct, slot);
        ASSERT_LT(ct, num_k_src);
        k_slots[ct][slot] = Complex(at(k, kv, key, c), 0.0);
      }
    }
  }

  std::vector<Ciphertext<word>> k_swapped(num_k_src);
  for (int i = 0; i < num_k_src; i++) {
    Plaintext<word> pt;
    big.context->encoder_.Encode(pt, kKeyLevel, big.param->GetScale(kKeyLevel),
                                 k_slots[i]);
    Ciphertext<word> ct, a;
    big.ui->Encrypt(ct, pt);
    k_swap_a.Evaluate(big.context, a, ct, big.ui->GetEvkMap());
    k_swap_b.Evaluate(big.context, k_swapped[i], a, big.ui->GetEvkMap());
    ASSERT_EQ(big.param->NPToLevel(k_swapped[i].GetNP()), kKeyLevel - 2);
  }

  std::vector<Ciphertext<word>> exchanged;
  exchange.Evaluate(exchanged, k_swapped, big.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(exchanged.size()), layout.num_cts);
  ASSERT_EQ(big.param->NPToLevel(exchanged[0].GetNP()), kSinCLevel);

  // The exchange delivers the key's low three bits as its ARRAY index; the
  // layout wants `column / rank = BitRev3(key & 7)` there. That is a
  // relabelling of eight pointers, which is why the naming above is free.
  std::vector<Ciphertext<word>> k_operand(layout.num_cts);
  for (int y = 0; y < layout.num_cts; y++) {
    Ciphertext<word> sinc;
    boot->SlotToSinC(sinc, num_slots, exchanged[y], big.ui->GetEvkMap());
    big.context->LevelDown(k_operand[BitRev(y, 3)], sinc, kProductLevel);
  }
  exchanged.clear();

  // The same split as for Q: check the transport before the product runs on
  // it, because a layout mistake and a product mistake look identical at the
  // end and have nothing in common as bugs.
  {
    double worst = 0.0, biggest = 0.0, unreplicated = 0.0;
    for (int i = 0; i < layout.num_cts; i++) {
      Plaintext<word> pt;
      big.ui->Decrypt(pt, k_operand[i]);
      std::vector<Complex> got;
      big.context->encoder_.DecodeSinC(got, pt, kSubDegree);
      for (const auto &z : got) biggest = std::max(biggest, std::abs(z));
      for (int head = 0; head < kLanes; head++) {
        for (int key = 0; key < kTokens; key++) {
          for (int c = 0; c < kHeadDim; c++) {
            int ct, index;
            layout.LocateSinC(BitRev(c, 7), BitRev(key, 7), head, ct, index);
            if (ct != i) continue;
            const double want = at(k, head / kGqaGroup, key, c);
            worst = std::max(worst, std::abs(got[index].real() - want));
            // The control: what a lane would hold if the four copies of a KV
            // head had not been made, i.e. if lane `head` read KV head
            // `head mod 4` instead.
            unreplicated = std::max(
                unreplicated,
                std::abs(got[index].real() - at(k, head % kKvHeads, key, c)));
          }
        }
      }
    }
    std::cout << "K operand after swap + exchange + SinC + LevelDown: max "
                 "|diff| = "
              << worst << ", |got| <= " << biggest << std::endl;
    std::cout << "  control (lanes not replicated by KV head): "
              << unreplicated << std::endl;
    EXPECT_LT(worst, 1e-3)
        << "the transport did not put K where SwitchedCcmmLayout says it goes";
    EXPECT_GT(unreplicated, 1e-2)
        << "every lane holds the same keys, so this test is not checking that "
           "GQA replication happened";
    EXPECT_NEAR(k_operand[0].GetScale() / product_scale, 1.0, 1e-3);
  }

  // Q must have landed on the same scale K was encoded at, or the product's
  // rescale does not return to the canonical level-0 value.
  EXPECT_NEAR(q_operand[0].GetScale() / product_scale, 1.0, 1e-3);

  // ---- the product ------------------------------------------------------
  // The switching key has to live on the switching Context, whose alpha is 1;
  // the block's own alpha is 12 and a key published at that PQ would not fit
  // the small ring's budget at all. Same secret, different ring.
  sw.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                              kProductLevel);
  sw.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                     small.ui->GetSecretCoeffs(),
                                     kProductLevel);

  SwitchedCcmmHandler<word> product(sw.context, small.context, kSubDegree);
  for (int idx : product.SmallRotationIndices()) {
    small.ui->PrepareRotationKey(idx, kProductLevel);
  }

  std::vector<Ciphertext<word>> scores;
  product.Multiply(scores, q_operand, k_operand,
                   sw.ui->GetRingSwitchKey(layout.rank),
                   sw.ui->GetInverseRingSwitchKey(layout.rank),
                   small.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(scores.size()), layout.num_cts);

  // ---- against the host, entrywise --------------------------------------
  std::vector<std::vector<Complex>> got(layout.num_cts);
  for (int i = 0; i < layout.num_cts; i++) {
    EXPECT_EQ(sw.param->NPToLevel(scores[i].GetNP()), kProductLevel - 1);
    Plaintext<word> pt;
    sw.ui->Decrypt(pt, scores[i]);
    sw.context->encoder_.DecodeSinC(got[i], pt, kSubDegree);
  }

  double worst = 0.0, sum = 0.0, biggest = 0.0, unpermuted = 0.0;
  for (int head = 0; head < kLanes; head++) {
    for (int query = 0; query < kTokens; query++) {
      for (int key = 0; key < kTokens; key++) {
        double want = 0.0;
        for (int c = 0; c < kHeadDim; c++) {
          want += at(q, head, query, c) * at(k, head / kGqaGroup, key, c);
        }
        biggest = std::max(biggest, std::abs(want));
        int ct, index;
        layout.LocateSinC(BitRev(query, 7), BitRev(key, 7), head, ct, index);
        const double d = std::abs(got[ct][index].real() - want);
        worst = std::max(worst, d);
        sum += d;
        // The control: the same score read where it would sit if the row and
        // column indices were NOT the bit-reversed ones. If the two reversals
        // did not really cancel, this is what would have been right.
        int ct2, index2;
        layout.LocateSinC(query, key, head, ct2, index2);
        unpermuted =
            std::max(unpermuted, std::abs(got[ct2][index2].real() - want));
      }
    }
  }
  const double mean = sum / (static_cast<double>(kLanes) * kTokens * kTokens);

  std::cout << "Q K^T through the transport: max |diff| = " << worst
            << ", mean " << mean << ", |want| <= " << biggest << std::endl;
  std::cout << "  control (row/column not bit-reversed): " << unpermuted
            << std::endl;

  EXPECT_LT(worst, 5e-3)
      << "the block's packing did not arrive where the product looks for it";
  EXPECT_GT(unpermuted, 1e-2)
      << "the naming of the row and column indices makes no difference here, "
         "so this test is not checking that the two bit reversals cancel";
}
