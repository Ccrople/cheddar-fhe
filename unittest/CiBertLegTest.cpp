// The attention leg at BERT-Base's shape: 12 heads of 64, no RoPE, every key
// live.
//
// `CiBootSet.TheLibraryLegReproducesTheReference` is the same walk at Llama's
// shape and is this file's sibling; what is measured here is that the three
// things BERT changes are the only three things that change.
//
//   * **12 heads of the layout's 32 lanes.** The transport's addresses are the
//     RING's -- a 5-bit head field, a 4-bit column field, a 7-bit token field
//     -- so a model with fewer heads leaves lanes empty and nothing about the
//     doorstep moves. The dead lanes are killed by the transport's own mask
//     (`Config::num_heads`), and the softmax is told to put their argument at
//     exactly one, because a dead lane's row norm is otherwise zero and the
//     inverse square root of zero is outside every window.
//   * **A 64-wide head, so the scores are ONE chain call.** The chain
//     contracts `contraction = 64` columns per call; Llama's 128 channels are
//     two calls summed, BERT's 64 are one. The VALUES contraction is over the
//     128 key tokens either way, so that stays two.
//   * **No RoPE and no causal mask.** BERT adds learned absolute positions
//     before layer 0, so the transport keeps its restore multiply and drops
//     the angles; and every key is live, which is `bidirectional` -- the
//     per-row shift and the row_norm fold still do their work, and only the
//     0/1 mask becomes all ones.
//
// WHAT THIS TEST DOES NOT DO. The projections are not run: Q, K and V are
// encrypted straight into the doorstep the emissions would have written, at
// the scale a `HalfBoot` would have declared. That is the same simplification
// `TheRopedScoresReturnToSlotsOnTheRealLadder` makes and for the same reason
// -- what is under test here is the leg, and a wrong projection would be
// indistinguishable from a wrong transport.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/CiLift.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/CiSinCAttention.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::Plaintext;

namespace {

constexpr const char *kBootParam = "ci16_35.json";
constexpr const char *kSwitchParam = "ci_ringswitch16_35_boot.json";
constexpr const char *kSmallParam = "ci12_35_boot.json";
constexpr const char *kLiftedParam = "ringdegree13_35_boot.json";

// BERT-Base.
constexpr int kHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kT = 128;
// The chain rides at this: the scores' magnitude times `carried` has to stay
// inside EvalMod's range, which is what the leg's own assert checks.
constexpr double kScoreTarget = 0.30;

int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

// The doorstep (Doing.md 1.5bx), which is the LAYOUT's and not the model's:
// entry (token t, channel c, head i) sits at slot rev4(c % 16) << 12 |
// rev5(i) << 7 | rev7(t) of ciphertext c / 16.
int Door0(int t, int c, int i) {
  return (Rev(c % 16, 4) << 12) | (Rev(i, 5) << 7) | Rev(t, 7);
}

using Batch = std::vector<std::vector<std::vector<double>>>;  // [head][t][c]

Batch MakeBatch(std::mt19937_64 &gen, double a, int rows, int cols) {
  std::uniform_real_distribution<double> d(-a, a);
  Batch b(kHeads,
          std::vector<std::vector<double>>(rows, std::vector<double>(cols)));
  for (auto &h : b) {
    for (auto &r : h) {
      for (auto &v : r) v = d(gen);
    }
  }
  return b;
}

}  // namespace

TEST(CiBertLeg, TheLegRunsAtTheBertShape) {
  Ring boot(kBootParam);
  Ring swtch(kSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kSmallParam);
  Ring lifted(kLiftedParam,
              cheddar::CiLiftHandler<word>::LiftSecret(
                  small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const int num_slots = boot.param->MaxNumSlots();
  const int chain_level = 2;

  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest req;
    bctx->AddRequiredRotations(req, num_slots);
    boot.ui->PrepareRotationKey(req);
  }

  // ---- the clear model ---------------------------------------------------
  std::mt19937_64 gen(0xBE47);
  Batch q = MakeBatch(gen, 0.20, kT, kHeadDim);
  Batch k = MakeBatch(gen, 0.20, kT, kHeadDim);
  const Batch v = MakeBatch(gen, 0.30, kT, kHeadDim);

  // The scores, and the common factor that puts them where the chain wants
  // them. A projection would carry this in its weights; here it is applied to
  // Q and K directly, which is the same thing one step later.
  auto scores = [&](const Batch &qq, const Batch &kk) {
    Batch s(kHeads, std::vector<std::vector<double>>(
                        kT, std::vector<double>(kT, 0.0)));
    for (int h = 0; h < kHeads; h++) {
      for (int t = 0; t < kT; t++) {
        for (int u = 0; u < kT; u++) {
          double acc = 0.0;
          for (int c = 0; c < kHeadDim; c++) acc += qq[h][t][c] * kk[h][u][c];
          s[h][t][u] = acc;
        }
      }
    }
    return s;
  };
  {
    Batch s = scores(q, k);
    double mx = 0.0;
    for (const auto &h : s) {
      for (const auto &r : h) {
        for (double x : r) mx = std::max(mx, std::abs(x));
      }
    }
    const double f = std::sqrt(kScoreTarget / mx);
    for (auto &h : q) {
      for (auto &r : h) {
        for (double &x : r) x *= f;
      }
    }
    for (auto &h : k) {
      for (auto &r : h) {
        for (double &x : r) x *= f;
      }
    }
  }
  const Batch S = scores(q, k);
  double smin = 1e300, smax = -1e300;
  for (const auto &h : S) {
    for (const auto &r : h) {
      for (double x : r) {
        smin = std::min(smin, x);
        smax = std::max(smax, x);
      }
    }
  }
  const double span = smax - smin;
  // `m_eff` is the MODEL's exponent scale, not this synthetic draw's. The leg
  // evaluates `exp(m_eff (S - shift) / span)` and the ratio is scale free, so
  // m_eff alone decides how peaked the softmax is and how hard the exp fit
  // has to work -- and it is what the leg derives its exp degree from.
  // Measured on the real checkpoint (`reference_forward_bert.py`), BERT-Base's
  // twelve layers span 22.7 to 61.0; the low end is used here.
  const double m_eff = 24.0;

  // ---- the images, encrypted straight into the doorstep -------------------
  const auto &enc = boot.context->encoder_;
  typename cheddar::CiSinCAttention<word>::Config acfg;
  acfg.dense_images = true;
  acfg.num_heads = kHeads;
  acfg.head_dim = kHeadDim;
  acfg.rope = false;
  acfg.restore = 1.0;  // no HalfBoot in front of these, so nothing to undo
  const auto t0 = std::chrono::steady_clock::now();
  cheddar::CiSinCAttention<word> attn(bctx, swtch.context, small.context,
                                      lifted.context, acfg);
  const auto t1 = std::chrono::steady_clock::now();
  const auto &layout = attn.GetLayout();
  ASSERT_EQ(attn.GetNumImages(), kHeadDim / layout.rank);
  ASSERT_EQ(attn.GetNumHeads(), kHeads);

  auto encrypt_images = [&](const Batch &x, std::vector<Ciphertext<word>> &out) {
    out.resize(attn.GetNumImages());
    for (int g = 0; g < attn.GetNumImages(); g++) {
      std::vector<Complex> msg(boot.Degree(), Complex(0.0, 0.0));
      for (int t = 0; t < kT; t++) {
        for (int cp = 0; cp < layout.rank; cp++) {
          for (int h = 0; h < kHeads; h++) {
            msg[Door0(t, g * layout.rank + cp, h)] =
                Complex(x[h][t][g * layout.rank + cp], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      // At the scale a HalfBoot declares its output at, which is what the
      // transport's plaintexts are stated against (`Config::landing_scale`).
      enc.Encode(pt, acfg.land_level, bctx->GetStCInputScale(), msg);
      boot.ui->Encrypt(out[g], pt);
      out[g].SetNumSlots(num_slots);
    }
  };
  std::vector<Ciphertext<word>> q_ct, k_ct, v_ct;
  encrypt_images(q, q_ct);
  encrypt_images(k, k_ct);
  encrypt_images(v, v_ct);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // ---- keys ---------------------------------------------------------------
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : attn.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  {
    EvkRequest req;
    attn.AddSwitchRotations(req);
    swtch.ui->PrepareRotationKey(req);
  }
  {
    EvkRequest req;
    attn.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  typename cheddar::CiSinCAttention<word>::Keys keys;
  keys.boot = &boot.ui->GetEvkMap();
  keys.swtch = &swtch.ui->GetEvkMap();
  keys.lifted = &lifted.ui->GetEvkMap();
  keys.ring_switch = &swtch.ui->GetRingSwitchKey(layout.rank);
  keys.inverse_ring_switch = &swtch.ui->GetInverseRingSwitchKey(layout.rank);

  // ---- the softmax calibration, per row, every key live -------------------
  //
  // THE DEAD LANES ARE PART OF THE CALIBRATION. A lane the model does not use
  // carries zero scores, and the walk would hand its inverse square root a row
  // norm of zero -- outside any window, where a Chebyshev grows like
  // cosh(d arccosh v) and takes the whole ciphertext with it. Shift zero and
  // norm `dim` put those rows at exactly one instead: y = exp(0) = 1 at every
  // key, pre-divided by sqrt(dim), so the row's square sums to one.
  std::vector<std::vector<double>> row_shift(
      layout.lanes, std::vector<double>(layout.dim, 0.0));
  std::vector<std::vector<double>> row_norm(
      layout.lanes,
      std::vector<double>(layout.dim, static_cast<double>(layout.dim)));
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = Rev(lane, 5);
    if (head >= kHeads) continue;  // dead: the defaults above are the answer
    for (int row = 0; row < layout.dim; row++) {
      double mx = -1e300;
      for (int col = 0; col < layout.dim; col++) {
        mx = std::max(mx, S[head][row][col]);
      }
      row_shift[lane][row] = mx;
      double sq = 0.0;
      for (int col = 0; col < layout.dim; col++) {
        sq += std::exp(m_eff * (S[head][row][col] - mx) / span);
      }
      row_norm[lane][row] = sq;
    }
  }
  typename cheddar::CiSinCAttention<word>::SoftMaxCalibration calib;
  calib.m_eff = m_eff;
  calib.span = span;
  calib.shift = smax;
  calib.norm_lo = 0.9;
  calib.norm_hi = 1.1;
  calib.causal = true;          // the per-row walk
  calib.bidirectional = true;   // with every key live
  calib.row_shift = row_shift;
  calib.row_norm = row_norm;
  attn.PrepareSoftMax(calib);

  // ---- the leg ------------------------------------------------------------
  const auto t2 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> s0;
  double carried = 0.0;
  attn.Scores(s0, q_ct, k_ct, keys, &carried);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t3 = std::chrono::steady_clock::now();
  ASSERT_LT(carried * std::max(std::abs(smax), std::abs(smin)), 0.95)
      << "the transport's canonicalising fold did not land carried in "
         "EvalMod's range";

  std::vector<Ciphertext<word>> booted(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    s0[bi].SetNumSlots(num_slots);
    bctx->Boot(booted[bi], s0[bi], boot.ui->GetEvkMap());
  }
  ASSERT_EQ(boot.param->NPToLevel(booted[0].GetNP()), attn.GetTopLevel());

  const auto t4 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> P;
  attn.SoftMax(P, booted, carried, boot.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t5 = std::chrono::steady_clock::now();

  // P, read back: the row sums say whether the bidirectional mask is really
  // all ones, and the dead lanes say whether they stayed in range.
  double rowsum_dev = 0.0, dead_worst = 0.0;
  {
    std::vector<std::vector<double>> rowsum(
        layout.lanes, std::vector<double>(layout.dim, 0.0));
    for (int bi = 0; bi < layout.num_cts; bi++) {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, P[bi]);
      std::vector<Complex> slots;
      boot.context->encoder_.Decode(slots, pt);
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            rowsum[lane][row] += slots[slot].real();
          }
        }
      }
    }
    for (int lane = 0; lane < layout.lanes; lane++) {
      for (int row = 0; row < layout.dim; row++) {
        if (Rev(lane, 5) < kHeads) {
          rowsum_dev = std::max(rowsum_dev, std::abs(rowsum[lane][row] - 1.0));
        } else {
          dead_worst = std::max(dead_worst, std::abs(rowsum[lane][row]));
        }
      }
    }
  }

  const auto t6 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> out;
  attn.Values(out, P, v_ct, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t7 = std::chrono::steady_clock::now();

  // ---- against the bidirectional leg in the clear -------------------------
  double worst = 0.0, biggest = 0.0, transposed = 0.0;
  for (int bi = 0; bi < attn.GetNumImages(); bi++) {
    ASSERT_EQ(boot.param->NPToLevel(out[bi].GetNP()), 0);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, out[bi]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int lane = 0; lane < layout.lanes; lane++) {
      const int head = Rev(lane, 5);
      if (head >= kHeads) continue;
      for (int row = 0; row < layout.dim; row++) {
        std::vector<double> p(layout.dim, 0.0);
        double z = 0.0;
        for (int col = 0; col < layout.dim; col++) {
          p[col] = std::exp(m_eff * (S[head][row][col] - row_shift[lane][row]) /
                            span);
          z += p[col];
        }
        for (double &x : p) x /= z;
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          double want = 0.0, want_t = 0.0;
          for (int col = 0; col < layout.dim; col++) {
            want += p[col] * v[head][col][column];
            want_t += p[col] * v[head][column][col % kHeadDim];
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          biggest = std::max(biggest, std::abs(want));
          worst = std::max(worst, std::abs(got - want));
          transposed = std::max(transposed, std::abs(got - want_t));
        }
      }
    }
  }

  auto secs = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << "the BERT leg (12 heads of 64, no RoPE, bidirectional): vs the "
            << "clear " << worst << " (|output| <= " << biggest
            << ", carried " << carried << ", m_eff " << m_eff << ")"
            << std::endl;
  std::cout << "  live row sums off one by " << rowsum_dev
            << ", dead lanes' row sums <= " << dead_worst << std::endl;
  std::cout << "  control: transposed " << transposed << std::endl;
  std::cout << "  cost: construct " << secs(t0, t1) << " s, Scores "
            << secs(t2, t3) << " s, SoftMax " << secs(t4, t5) << " s, Values "
            << secs(t6, t7) << " s" << std::endl;

  EXPECT_LT(worst, 2e-2) << "the leg did not reproduce BERT's attention";
  EXPECT_LT(rowsum_dev, 1e-2)
      << "the bidirectional mask is not all ones, or the row_norm fold is off";
  EXPECT_GT(transposed, 5e-3) << "the transposed control agrees, so the "
                                 "addressing proves nothing";
}
