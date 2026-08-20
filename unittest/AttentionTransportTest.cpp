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
// K IS SUPPLIED IN ITS OWN LAYOUT, NOT TRANSPORTED. Q's operand wants the
// channel on the column axis, which is where the projection can put it; K's
// wants the KEY there, three of whose bits therefore land on the ciphertext
// axis. That is a cross-ciphertext move, it is a separate piece of machinery,
// and it is not built yet. So K is encoded on the host in the layout the
// product needs and the test pins the Q half of the chain, which is the half
// the permutation and the SinC transform live in.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/SwitchedCcmm.h"
#include "extension/SlotPermute.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::Complex;
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

// Where the transforms are compiled. The permutation takes two levels and the
// SinC transform three, and the floor for any of them is 7.
constexpr int kPermuteLevel = 11;
constexpr int kSinCLevel = 9;
constexpr int kSinCPhases = 3;
constexpr int kProductLevel = 1;

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

}  // namespace

TEST(AttentionTransport, CarriesTheBlockPackingIntoAScoreProduct) {
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

  auto boot = std::dynamic_pointer_cast<cheddar::BootContext<word>>(big.context);
  ASSERT_NE(boot, nullptr);
  boot->PrepareEvalSpecialFFT(num_slots);
  // The last phase's plaintexts are encoded so the operand lands on the
  // product ring's own level-1 scale. On a grafted ladder that is not the
  // scale the transform started at -- sylphflow16_35 is 2^35 from level 3 up
  // and 2^29.88 at level 1 -- and `LevelDown` does not move a scale, so
  // without this the operand reaches the product five bits off and the error
  // does not surface until a bootstrap two turns later.
  boot->PrepareSinC(num_slots, kSubDegree, kSinCLevel, kSinCLevel, kSinCPhases,
                    big.param->GetScale(kProductLevel));

  // ---- the permutation, as the two narrow swaps ------------------------
  //
  // `[4 | 7]` in one transform is 2048 diagonals and therefore 2048 plaintexts;
  // `[4|4]` at offset 3 followed by `[4|3]` is the same permutation for 31 and
  // 127, at the cost of one more level. SlotPermuteTest checks the equality
  // exactly.
  SlotPermute<word> swap_a(big.context,
                           SwapAdjacentFields(num_slots, 4, 4, 3),
                           kPermuteLevel);
  SlotPermute<word> swap_b(big.context, SwapAdjacentFields(num_slots, 3, 4),
                           kPermuteLevel - 1);
  std::cout << "permute step 1: " << swap_a.GetNumDiagonals()
            << " diagonals, stride " << swap_a.GetStride() << ", BSGS "
            << swap_a.GetBS() << "x" << swap_a.GetGS() << ", shift "
            << swap_a.GetShift() << std::endl;
  std::cout << "permute step 2: " << swap_b.GetNumDiagonals()
            << " diagonals, stride " << swap_b.GetStride() << ", BSGS "
            << swap_b.GetBS() << "x" << swap_b.GetGS() << ", shift "
            << swap_b.GetShift() << std::endl;

  EvkRequest req;
  swap_a.AddRequiredRotations(req);
  swap_b.AddRequiredRotations(req);
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
  // q[head][token][c] and k[head][token][c], one call group of 16 heads. GQA
  // repetition is the projection's business and is not modelled here; each
  // head simply gets its own keys.
  std::vector<double> q(static_cast<size_t>(kLanes) * kTokens * kHeadDim);
  std::vector<double> k(q.size());
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
    boot->SlotToSinC(sinc, num_slots, b, big.ui->GetEvkMap());
    ASSERT_EQ(big.param->NPToLevel(sinc.GetNP()), kSinCLevel - kSinCPhases);
    big.context->LevelDown(q_operand[i], sinc, kProductLevel);
    EXPECT_NEAR(q_operand[i].GetScale() / big.param->GetScale(kProductLevel),
                1.0, 1e-6)
        << "the operand arrives at the product off the canonical scale";
  }
  ASSERT_EQ(big.param->LevelToNP(kProductLevel), q_operand[0].GetNP())
      << "the operand is not at the level the product expects";

  // ---- K, encoded on the host in the layout the product wants -----------
  //
  // row = the contraction index BitRev(c), column = BitRev(key), lane = head.
  const double product_scale = sw.param->GetScale(kProductLevel);
  std::vector<Ciphertext<word>> k_operand(layout.num_cts);
  {
    std::vector<std::vector<Complex>> msg(
        layout.num_cts, std::vector<Complex>(num_slots, Complex(0)));
    for (int head = 0; head < kLanes; head++) {
      for (int key = 0; key < kTokens; key++) {
        for (int c = 0; c < kHeadDim; c++) {
          int ct, index;
          layout.LocateSinC(BitRev(c, 7), BitRev(key, 7), head, ct, index);
          msg[ct][index] = Complex(at(k, head, key, c), 0.0);
        }
      }
    }
    for (int i = 0; i < layout.num_cts; i++) {
      Plaintext<word> pt;
      sw.context->encoder_.EncodeSinC(pt, kProductLevel, product_scale, msg[i],
                                      kSubDegree);
      sw.ui->Encrypt(k_operand[i], pt);
    }
  }

  // Q must have landed on the same scale K was encoded at, or the product's
  // rescale does not return to the canonical level-0 value.
  EXPECT_NEAR(q_operand[0].GetScale() / product_scale, 1.0, 1e-6);

  // ---- the product ------------------------------------------------------
  big.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                               kProductLevel);
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
          want += at(q, head, query, c) * at(k, head, key, c);
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
