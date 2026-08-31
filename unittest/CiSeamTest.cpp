// The seam as a library class, checked the way its consumer reads it.
//
// `CiLlamaSeam` is the promotion of the seam that `CiSwitchedCcmmTest`'s layer
// and three separate cases in `CiFfnTest` each wrote out inline: three slot
// transforms carrying the CC-MM chain's layout to the banded half-density
// coefficient image the next projection reads. The map is intricate -- a
// staged bit permutation, a bit-reversed token decrement, and a duplicate copy
// that must skip component zero -- and every one of those three has cost a
// wrong layer at least once.
//
// ## What is checked, and why it is the consumer's read
//
// NOT "is every duplicate where the formula says". That is the formula marking
// its own work, and it is exactly how a `row - 1` token shift survived for a
// whole increment while the layer's O projection read the same image at
// err/mx 14.48. What runs here is the read `CoeffLinearLeg` performs:
// coefficients out, then the banded scan `ModDecomp` inverts. A duplicate
// misplaced by one position makes that scan walk the error the length of the
// ring, so it cannot be missed.
//
// Two bands, separately. The LIVE components must equal the chain entries the
// seam was handed; the DEAD ones must come out at zero, which is what half
// density means to everything downstream.
//
// About two and a half minutes, against a quarter of an hour for the layer.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "extension/CiLlamaSeam.h"
#include "extension/SylphSchedule.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using Complex = std::complex<double>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::EvkRequest;
using cheddar::Plaintext;

namespace {

constexpr const char *kParam = "ci16_35.json";
// The layer's own shape: 128 tokens, 16 columns of the chain layout, 32 lanes,
// rank 512 with 256 declared per half.
constexpr int kCols = 16, kRows = 128, kLanes = 32, kRank = 512;
// A half-density ciphertext carries at most `rank/2 - 1` live channels --
// component zero has no partner, and `I -> rank - I` has exactly two fixed
// points -- so everything at or above this must read back as zero.
constexpr int kLive = kRank / 2;

int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

// The inverse of the banded recomposition `rec[p*rank + I] = comp_I[p] +
// [I != 0] comp_{rank-I}[p+1]`: an alternating suffix scan, run from the far
// end where both terms are known.
std::vector<std::vector<double>> BandedComponents(
    const std::vector<double> &coeffs, int rank, int rows) {
  std::vector<std::vector<double>> comp(rank, std::vector<double>(rows, 0.0));
  for (int t = 0; t < rows; t++) {
    comp[0][t] = coeffs[static_cast<size_t>(t) * rank];
  }
  for (int i = 1; i <= rank / 2; i++) {
    const int mi = rank - i;
    double ai = 0.0, am = 0.0;
    for (int t = rows - 1; t >= 0; t--) {
      const double ni = coeffs[static_cast<size_t>(t) * rank + i] - am;
      const double nm = coeffs[static_cast<size_t>(t) * rank + mi] - ai;
      comp[i][t] = ni;
      comp[mi][t] = nm;
      ai = ni;
      am = nm;
    }
  }
  return comp;
}

}  // namespace

TEST(CiSeam, TheLibrarySeamHandsTheProjectionAReadableImage) {
  // Slack nine: `SlotToCoeff` is compiled at `GetStCStartLevel()`, and at
  // slack twelve that is 7, so its phases run at 7, 6 and 5 -- two of them
  // inside ci16_35's `num_accum == 1` zone -- and the coefficients come back
  // at 4.99e+47. The seam derives its own levels from wherever StC lands, so
  // this is the one number the caller still owns.
  const char *sl = std::getenv("CHEDDAR_JOIN_SLACK");
  const int kSlack = sl != nullptr ? std::atoi(sl) : 9;
  Ring boot(kParam, {}, kSlack);
  ASSERT_TRUE(boot.param->conjugate_invariant_);
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const int degree = boot.Degree();
  const int num_slots = boot.param->MaxNumSlots();
  ASSERT_EQ(degree, num_slots);

  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest req;
    bctx->AddRequiredRotations(req, num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  cheddar::SylphSchedule<word> sched(bctx, num_slots);

  cheddar::CiSwitchedCcmmLayout layout;
  layout.dim = kRows;
  layout.lanes = kLanes;
  layout.num_cts = 8;
  layout.rank = kRank / kLanes;  // 16 columns per ciphertext

  typename cheddar::CiLlamaSeam<word>::Config scfg;
  scfg.verbose = true;
  cheddar::CiLlamaSeam<word> seam(boot.context, layout, sched.GetStCLevel(),
                                  scfg);
  std::cout << "slot " << sched.GetSlotLevel() << ", StC "
            << sched.GetStCLevel() << ", seam input at "
            << seam.GetInputLevel() << std::endl;

  const int half = 0;
  seam.PrepareHalf(half);
  ASSERT_EQ(seam.GetPreparedHalf(), half);
  {
    EvkRequest req;
    seam.AddRequiredRotations(req);
    seam.AddHalfRotations(req);
    boot.ui->PrepareRotationKey(req);
  }

  // ---- the data, in the chain's own addresses ---------------------------
  auto slot_chain = [](int row, int col, int lane) {
    return Rev(col, 4) * 4096 + Rev(row, 7) * 32 + lane;
  };
  auto chan_of = [](int col, int lh) { return Rev(col, 4) * 32 + Rev(lh, 5); };

  // THE AMPLITUDE IS PART OF THE MEASUREMENT, because the seam is three
  // LinearTransforms and a LinearTransform's added error is ABSOLUTE. Encrypted
  // at sigma = 1 this test has always reported ~2^-14.4, and the LAYER hands
  // the seam the booted attention output, whose coefficients reach 4.2e-04 --
  // slot rms of order 0.02, forty times colder. A component validated hot and
  // run cold is the fault this tree has now found four times (1.5ec's
  // reduction, 1.5ea's SiLU range, 1.5ec's invsqrt window, and this), so the
  // amplitude is a knob and the default is unchanged.
  const double amp = [] {
    const char *e = std::getenv("CHEDDAR_SEAM_AMP");
    return (e && e[0]) ? std::atof(e) : 1.0;
  }();
  std::mt19937_64 gen(0x5EA3);
  std::normal_distribution<double> xd(0.0, amp);
  std::vector<Complex> msg(num_slots, Complex(0.0, 0.0));
  std::vector<std::vector<std::vector<double>>> v(
      kRows, std::vector<std::vector<double>>(
                 kCols, std::vector<double>(kLanes, 0.0)));
  for (int row = 0; row < kRows; row++) {
    for (int col = 0; col < kCols; col++) {
      for (int lane = 0; lane < kLanes; lane++) {
        const double x = xd(gen);
        v[row][col][lane] = x;
        msg[slot_chain(row, col, lane)] = Complex(x, 0.0);
      }
    }
  }

  Plaintext<word> pt;
  const int in_level = seam.GetInputLevel();
  boot.context->encoder_.Encode(pt, in_level, boot.param->GetScale(in_level),
                                msg);
  Ciphertext<word> ct;
  boot.ui->Encrypt(ct, pt);
  ct.SetNumSlots(num_slots);

  Ciphertext<word> out;
  seam.Apply(out, ct, sched, boot.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::cout << "  the seam gave a coefficient ciphertext at level "
            << boot.param->NPToLevel(out.GetNP()) << std::endl;

  // ---- the consumer's read ----------------------------------------------
  Plaintext<word> out_pt;
  boot.ui->Decrypt(out_pt, out);
  std::vector<double> coeffs;
  boot.context->encoder_.DecodeCoeff(coeffs, out_pt);
  const auto comp = BandedComponents(coeffs, kRank, kRows);

  // The seam and `ToCoeff` between them carry a constant that is a property of
  // the BootParameter, not of the data, so it is fitted once over the whole
  // live band and then divided out -- exactly as the layer's O projection does.
  double num = 0.0, den = 0.0, absmax = 0.0;
  for (int col = 0; col < kCols; col++) {
    for (int lh = 0; lh < 16; lh++) {
      const int lane = half * 16 + lh;
      const int I = Rev(chan_of(col, lh), 9);
      for (int row = 0; row < kRows; row++) {
        const double want = v[row][col][lane];
        num += comp[I][row] * want;
        den += want * want;
        absmax = std::max(absmax, std::abs(want));
      }
    }
  }
  ASSERT_GT(den, 1e-6) << "the reference is zero, so nothing here can fail";
  const double carried = num / den;

  double live_err = 0.0;
  for (int col = 0; col < kCols; col++) {
    for (int lh = 0; lh < 16; lh++) {
      const int lane = half * 16 + lh;
      const int I = Rev(chan_of(col, lh), 9);
      for (int row = 0; row < kRows; row++) {
        live_err = std::max(live_err,
                            std::abs(comp[I][row] / carried -
                                     v[row][col][lane]));
      }
    }
  }
  // The dead half has to come out of the scan at zero: that is what half
  // density means to everything downstream, and the scan would walk any
  // misplaced duplicate the length of the ring rather than hide it.
  double dead_err = 0.0;
  for (int I = kLive; I < kRank; I++) {
    for (int p = 0; p < kRows; p++) {
      dead_err = std::max(dead_err, std::abs(comp[I][p] / carried));
    }
  }

  std::cout << "THE LIBRARY SEAM, read as components: live " << live_err
            << ", dead " << dead_err << " against |v| <= " << absmax
            << ", carried " << carried << " (amplitude " << amp
            << ", live 2^" << std::log2(live_err / absmax) << ")" << std::endl;
  EXPECT_LT(live_err, 5e-2 * absmax)
      << "the seam did not hand the projection a readable banded image";
  EXPECT_LT(dead_err, 5e-2 * absmax)
      << "the dead half is not dead, so the image is not half density";

  // A CONTROL THAT MUST FAIL. Reading the live band one token position out is
  // the mistake the tree actually made twice: once as a duplicate written at
  // `row - 1` when the image sat at position `rev7(row)`, and once as a whole
  // image at position `rev7(row)` when the leg's doorstep needs position
  // `row`. The seam now reverses the token field last, so the position IS the
  // row. If this control passes too, the check above is not testing what it
  // claims.
  double shifted_err = 0.0;
  for (int col = 0; col < kCols; col++) {
    for (int lh = 0; lh < 16; lh++) {
      const int lane = half * 16 + lh;
      const int I = Rev(chan_of(col, lh), 9);
      for (int row = 1; row < kRows; row++) {
        shifted_err = std::max(
            shifted_err, std::abs(comp[I][row] / carried -
                                  v[row - 1][col][lane]));
      }
    }
  }
  std::cout << "  control, the live band read one token out: " << shifted_err
            << std::endl;
  EXPECT_GT(shifted_err, 0.5 * absmax)
      << "the wrong reading fits as well as the right one, so this test "
         "cannot tell them apart";
}
