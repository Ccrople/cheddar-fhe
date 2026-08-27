// The conjugate-invariant FFN, one stage at a time.
//
// The attention leg runs on real weights (Doing.md 1.5ch). The FFN has never
// run on R+ at all, and it is the half of the layer that needs no CC-MM, no
// SinC and no transport: RMSNorm, two projections, SiLU, a pointwise multiply,
// one projection, one residual add. Doing.md 1.5cs is its specification and
// this file is that specification checked one stage at a time, because the
// only hard part is the layout and a single end-to-end number cannot say
// which stage moved it.
//
// THE LAYOUT, WHICH IS THE WHOLE QUESTION. A projection's output on R+ is the
// banded image at rank 512: coefficient `k = p * 512 + I` carries component
// `I` at position `p`. `CoeffOfSlot` on R+ is `BitReverse(k, 16)` (1.5bh), and
// k's bits are `I` in 0..8 and `p` in 9..15, so
//
//     slot = rev9(I) * 128 + rev7(p)
//
// -- the channel in the high nine bits and the token in the low seven, which
// is the block's own packing and [SYLPH] 3.2's "token index varies fastest".
// The channel a component carries is `rev9(I)` (`CoeffLinearLeg`'s
// `BitReverse(Component(r))`), so slot = channel * 128 + rev7(token).
//
// The token comes out BIT-REVERSED and that costs nothing: RMSNorm reduces
// over the channel field at a fixed low field, and rev7 is a bijection on the
// low field, so the reduction sums the right token whatever order they sit in.
//
// HALF DENSITY. Live components are `I < 256`, i.e. bit 8 of I clear, i.e.
// `rev9(I)` EVEN, i.e. **slot bit 7 = 0** -- 1.5by's statement re-derived from
// the coefficient end. The odd slots hold `comp_{512-I}[p+1]`, which is the
// NEIGHBOURING TOKEN's data and not a copy: a reduction over the channel field
// would return `(channels at token p) + (channels at token p+1)`, which is not
// twice anything. So the duplicates must be masked, and the mask is free
// because it rides the multiply that canonicalises HalfBoot's output scale
// anyway.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <utility>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/BootContext.h"
#include "extension/RmsNorm.h"

using word = uint32_t;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::Plaintext;
using Ring = ringfixture::Ring<word>;

namespace {

constexpr const char *kParam = "ci16_35.json";
constexpr int kTokens = 128;      // T, and the small degree on R+
constexpr int kRank = 512;        // degree / T, the channels a ciphertext holds
constexpr int kLive = kRank / 2;  // 1.5by: 256 live components of 512
constexpr double kEps = 1e-5;

int Rev(int v, int bits) {
  int r = 0;
  for (int b = 0; b < bits; b++) r |= ((v >> b) & 1) << (bits - 1 - b);
  return r;
}

// The banded two-term recomposition ModPack writes (1.5ba / 1.5bp), so a
// parent built with it decomposes back to exactly these components -- and so
// that a test operand is the shape a projection actually emits.
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

}  // namespace

// ---------------------------------------------------------------------------
// Stage 1: the crossing and RMSNorm, on the image a projection emits.
//
// Encrypt a half-density banded image at level 0, HalfBoot it into slots,
// spend one multiply on the mask-and-canonicalise, run RMSNorm over a DECLARED
// width of twice the live one, and compare against the same formula in double
// at the live addresses.
//
// What this pins, and none of it has been checked before:
//   - that a projection's coefficient image lands in slots at
//     `channel * 128 + rev7(token)`;
//   - that the live half is exactly slot bit 7 = 0;
//   - that the odd half carries the neighbouring token rather than a copy,
//     which is why the mask is not optional;
//   - that `RmsNormHandler` needs nothing but a doubled declared width.
// ---------------------------------------------------------------------------
TEST(CiFfn, TheCrossingAndRmsNormRunOnTheHalfDensityImage) {
  Ring boot(kParam);
  ASSERT_TRUE(boot.param->conjugate_invariant_);
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr) << kParam << " did not come up as a BootContext";

  const int degree = boot.Degree();
  const int num_slots = boot.param->MaxNumSlots();
  ASSERT_EQ(num_slots, degree);
  ASSERT_EQ(degree / kTokens, kRank);
  const int declared = kRank;  // 512 declared channels, 256 of them live

  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  const int land_level = boot.param->default_encryption_level_;

  // ---- the tensor, one ciphertext wide ---------------------------------
  //
  // Live channel `c` is EVEN and sits at component `rev9(c)`, which is the
  // half `I < 256`. The magnitudes are order one, which is where [SYLPH]
  // section 3.1's calibration puts RMSNorm's input (its table 2 says 7.65)
  // and where the scale is well spent -- RmsNormTest's own lesson.
  std::mt19937_64 gen(0xF7A11);
  std::normal_distribution<double> xd(0.0, 1.0);
  std::vector<double> x(static_cast<size_t>(kTokens) * declared, 0.0);
  std::vector<double> wn(declared, 0.0);
  for (int t = 0; t < kTokens; t++) {
    for (int c = 0; c < declared; c += 2) {
      x[static_cast<size_t>(t) * declared + c] = xd(gen);
    }
  }
  for (int c = 0; c < declared; c += 2) wn[c] = 0.5 + 0.5 * xd(gen);

  // ---- the components, and the parent that decomposes to them ----------
  std::vector<std::vector<double>> comp(kRank,
                                        std::vector<double>(kTokens, 0.0));
  for (int t = 0; t < kTokens; t++) {
    for (int c = 0; c < declared; c += 2) {
      const int I = Rev(c, 9);
      ASSERT_LT(I, kLive) << "an even channel must land in the live half";
      comp[I][Rev(t, 7)] = x[static_cast<size_t>(t) * declared + c];
    }
  }
  Plaintext<word> pt;
  boot.context->encoder_.EncodeCoeff(pt, 0, boot.param->GetScale(0),
                                     CiRecompose(comp, kRank, kTokens));
  Ciphertext<word> ct;
  boot.ui->Encrypt(ct, pt);
  ct.SetNumSlots(num_slots);

  // ---- the crossing ----------------------------------------------------
  Ciphertext<word> lifted;
  bctx->HalfBoot(lifted, ct, boot.ui->GetEvkMap());
  ASSERT_EQ(boot.param->NPToLevel(lifted.GetNP()), land_level);

  // What HalfBoot landed, before anything is done to it. The claim is
  // `slot = channel * 128 + rev7(token)` with the live half at bit 7 = 0, and
  // it is checked rather than assumed -- 1.5bx's lesson is that these maps are
  // never to be hand-derived and believed.
  {
    Plaintext<word> raw_pt;
    boot.ui->Decrypt(raw_pt, lifted);
    std::vector<Complex> raw;
    boot.context->encoder_.Decode(raw, raw_pt);
    double num = 0.0, den = 0.0;
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < declared; c += 2) {
        const double want = x[static_cast<size_t>(t) * declared + c];
        num += raw[c * kTokens + Rev(t, 7)].real() * want;
        den += want * want;
      }
    }
    const double landed = num / den;  // HalfBoot's boundary constant
    double live_err = 0.0, dead_max = 0.0, dead_neighbour = 0.0;
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < declared; c++) {
        const double got = raw[c * kTokens + Rev(t, 7)].real();
        if (c % 2 == 0) {
          const double want =
              landed * x[static_cast<size_t>(t) * declared + c];
          live_err = std::max(live_err, std::abs(got - want));
        } else {
          dead_max = std::max(dead_max, std::abs(got));
          // The claim about WHAT sits there: comp_{512-I}[p+1], i.e. the
          // partner channel at the next position.
          const int I = Rev(c, 9);
          const int p = Rev(t, 7);
          const double neighbour =
              (p + 1 < kTokens) ? comp[kRank - I][p + 1] : 0.0;
          dead_neighbour =
              std::max(dead_neighbour, std::abs(got - landed * neighbour));
        }
      }
    }
    std::cout << "HalfBoot boundary constant " << landed << " (2^"
              << std::log2(landed) << ")" << std::endl;
    std::cout << "  live slots (bit 7 = 0) match to " << live_err << std::endl;
    std::cout << "  dead slots reach " << dead_max << " and match the "
              << "neighbouring-token partner to " << dead_neighbour
              << std::endl;
    const double span = std::abs(landed);
    EXPECT_LT(live_err, 1e-3 * span)
        << "the coefficient image did not land at channel * 128 + rev7(token)";
    EXPECT_GT(dead_max, 1e-2 * span)
        << "the odd slots are empty, so half density is not what 1.5by says "
           "it is and the mask below would be measuring nothing";
    EXPECT_LT(dead_neighbour, 1e-3 * span)
        << "the odd slots hold something other than comp_{512-I}[p+1]";
  }

  // ---- mask and canonicalise, one multiply -----------------------------
  //
  // The mask is 1 on slot bit 7 = 0 and 0 elsewhere. Its SCALE is what
  // canonicalises: HalfBoot declares its output at GetStCInputScale(), and
  // one Mult + Rescale takes that to the canonical scale of the level below
  // for exactly one plaintext scale. Reading it off the ciphertext rather
  // than deriving it is deliberate -- a merely plausible scale survives every
  // plain operation and stops being silent three layers later, inside
  // EvalPoly.
  const int op_level = land_level - 1;
  const double restore = 1.0;  // no magnitude to put back in this stage
  const double pt_scale = boot.param->GetScale(op_level) *
                          boot.param->GetRescalePrimeProd(land_level) /
                          lifted.GetScale();
  std::vector<Complex> mask(num_slots, Complex(0.0, 0.0));
  for (int c = 0; c < declared; c += 2) {
    for (int t = 0; t < kTokens; t++) {
      mask[c * kTokens + t] = Complex(restore, 0.0);
    }
  }
  Plaintext<word> mask_pt;
  boot.context->encoder_.Encode(mask_pt, land_level, pt_scale, mask);
  Ciphertext<word> masked;
  boot.context->Mult(masked, lifted, mask_pt);
  boot.context->Rescale(masked, masked);
  ASSERT_EQ(boot.param->NPToLevel(masked.GetNP()), op_level);
  EXPECT_NEAR(masked.GetScale() / boot.param->GetScale(op_level), 1.0, 1e-6)
      << "the mask's scale has to leave the ciphertext canonical, or EvalPoly "
         "aborts inside RMSNorm";
  const double masked_scale = masked.GetScale();
  (void)masked_scale;

  // ---- RMSNorm, at a DECLARED width of twice the live one --------------
  //
  // The handler's num_ct is T * H / num_slots, so declaring H = 512 gives the
  // one ciphertext a 256-channel half-density tensor occupies. The reduction
  // then sums 256 live values and 256 zeros, and the factor of two in the mean
  // is absorbed by the layer constant -- which is calibrated anyway.
  double log_sum = 0.0, lo = 1e300, hi = 0.0;
  std::vector<double> ms(kTokens, 0.0);
  for (int t = 0; t < kTokens; t++) {
    double s = 0.0;
    for (int c = 0; c < declared; c++) {
      const double v = x[static_cast<size_t>(t) * declared + c];
      s += v * v;
    }
    ms[t] = s / declared;  // the DECLARED width, which is what the circuit sums
    log_sum += std::log(ms[t]);
    lo = std::min(lo, ms[t]);
    hi = std::max(hi, ms[t]);
  }
  const double alpha = 1.0 / std::exp(log_sum / kTokens);
  std::cout << "mean-square spread " << (hi / lo) << "x, alpha_L = " << alpha
            << std::endl;

  const double window = 6.0;
  const int degree_poly = 9;
  cheddar::RmsNormHandler<word> rms(boot.context, kTokens, declared, alpha,
                                    op_level, kEps, window, degree_poly);
  ASSERT_EQ(rms.GetNumCiphertexts(), 1)
      << "a 512-channel declared tensor at T = 128 is one R+ ciphertext";
  for (int d : rms.GetRotationDistances()) {
    boot.ui->PrepareRotationKey(d, op_level);
  }

  const double root_alpha = std::sqrt(alpha);
  std::vector<std::vector<Complex>> wts(1);
  wts[0].assign(num_slots, Complex(0.0, 0.0));
  for (int c = 0; c < declared; c++) {
    for (int t = 0; t < kTokens; t++) {
      wts[0][c * kTokens + t] = Complex(wn[c] * root_alpha, 0.0);
    }
  }
  // Ciphertext is move-only, so the operand vector is built by move.
  std::vector<Ciphertext<word>> in(1), out;
  in[0] = std::move(masked);
  rms.Apply(out, in, wts, boot.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(out.size(), 1u);

  // ---- against the same formula in double ------------------------------
  Plaintext<word> out_pt;
  boot.ui->Decrypt(out_pt, out[0]);
  std::vector<Complex> got;
  boot.context->encoder_.Decode(got, out_pt);

  double max_abs = 0.0, want_absmax = 0.0, dead_after = 0.0;
  for (int t = 0; t < kTokens; t++) {
    const double inv = 1.0 / std::sqrt(ms[t] + kEps);
    for (int c = 0; c < declared; c++) {
      const double v = got[c * kTokens + Rev(t, 7)].real();
      if (c % 2) {
        dead_after = std::max(dead_after, std::abs(v));
        continue;
      }
      const double want =
          x[static_cast<size_t>(t) * declared + c] * inv * wn[c];
      want_absmax = std::max(want_absmax, std::abs(want));
      max_abs = std::max(max_abs, std::abs(v - want));
    }
  }
  const double rel = max_abs / want_absmax;
  std::cout << "RMSNorm on the half-density image: |y| max " << want_absmax
            << ", max abs err " << max_abs << ", relative " << rel << " = 2^"
            << std::log2(rel) << std::endl;
  std::cout << "  dead slots after the mask reach " << dead_after << std::endl;
  std::cout << "[SYLPH] 3.1.2's target is 12 bits, i.e. " << std::pow(2.0, -12)
            << std::endl;
  EXPECT_LT(rel, std::pow(2.0, -12))
      << "below [SYLPH]'s 12-bit precision target";
  EXPECT_LT(dead_after, 1e-3 * want_absmax)
      << "the mask did not kill the duplicates, so the reduction summed two "
         "tokens";
}
