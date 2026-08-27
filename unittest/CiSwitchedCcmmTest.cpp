// The conjugate-invariant CC-MM chain, end to end (Doing.md 1.5bl):
//
//   ci16(65536) --Switch--> 16 x ci12(4096) --CiLift--> 8192
//               --[KANG] Alg.4 at 2k--> --CiDescend--> --SwitchBack--> ci16
//
// WHY THIS TEST AND NOT THE PIECES IT COMPOSES. RingSwitchTest shows the CI
// switch's parts are the banded block scan, CiLiftTest shows Algorithm 4 on
// the lifted ring computes exact products under the half-contraction
// contract. Neither says where in a BIG ciphertext an entry has to sit for
// the product to find it -- and on R+ that address is not a slot
// permutation: a big operand is, in coefficients, the banded RECOMPOSITION
// of its parts, the exact inverse of what the switch computes. This test
// assembles operands that way on the host, drives the whole device chain,
// and reads the result back through the scan -- entrywise against per-lane
// host products, with the transposed result as the control that has to
// fail.
//
// THE CONTRACT RIDES THROUGH THE SWITCH. 1.5bl's exactness needs the dead
// contraction range to be zero. The lhs half is exact by construction (the
// handler pads with device zeros and never switches the dead columns). The
// rhs half is data: its parts' upper blocks hold only encode and switch
// noise, so the flip term is bounded by the switch's own lane noise
// (Doing.md 1.5bj: ~sqrt(k) twice over 1.5e-05) -- the same floor the live
// data already carries. The chain's accuracy is switch-limited either way,
// and this test measures exactly that.
//
// TWO SHAPES. sub_degree 32 is the Llama shape: d = 128 (T x head_dim
// products), 32 real lanes, 8 rhs big ciphertexts, contraction 64 -- one
// half of a 128-deep contraction per call. sub_degree 128 is the shape
// CiLiftTest validated the primitive at (d = 32), so a failure here at 32
// but not at 128 indicts the shape, not the chain.
//
// SEPARATE BINARY: three rings alive at once; see SmallRingNttTest.cpp.

// ENABLE_EXTENSION stays DEFINED here, unlike the sibling multi-ring
// binaries: CiBootSet's consumption test needs ci16_35 to come up as its
// own BootContext, and RingFixture only builds one when the extension is
// visible. Every other preset in this file says boot: false and still gets
// a plain Context, so nothing else in the binary changes.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/CiSwitchedCcmm.h"
#include "core/EvkRequest.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#include "extension/BootContext.h"
#include "extension/ChebyshevFit.h"
#include "extension/CiSinCAttention.h"
#include "extension/EvalPoly.h"
#include "extension/EvalSpecialFFT.h"
#include "extension/LinearTransform.h"
#include "extension/LlamaLinear.h"
#include "extension/SylphSchedule.h"
#include "extension/StripedMatrix.h"

using word = uint32_t;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::CiLiftHandler;
using cheddar::Constant;
using cheddar::EvalPoly;
using cheddar::CiSwitchedCcmmHandler;
using cheddar::CiSinCConverter;
using cheddar::CiSwitchedCcmmLayout;
using cheddar::EvkRequest;
using cheddar::Complex;
using cheddar::Plaintext;
using Ring = ringfixture::Ring<word>;

namespace {

constexpr const char *kSwitchParam = "ci_ringswitch16_35.json";
constexpr const char *kSmallParam = "ci12_35.json";
constexpr const char *kLiftedParam = "ringdegree13_35.json";

// [lane][row][col] real batches at the CI level, with a live sub-rectangle.
using RealBatch = std::vector<std::vector<std::vector<double>>>;

RealBatch SampleBatch(int lanes, int d, int live_rows, int live_cols,
                      double bound, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-bound, bound);
  RealBatch m(lanes,
              std::vector<std::vector<double>>(d, std::vector<double>(d, 0.0)));
  for (int t = 0; t < lanes; t++) {
    for (int i = 0; i < live_rows; i++) {
      for (int x = 0; x < live_cols; x++) m[t][i][x] = dist(gen);
    }
  }
  return m;
}

// The banded two-term recomposition of RingSwitchTest -- what SwitchBack
// implements on the device, used here to assemble a big operand whose
// switch yields exactly the parts the layout prescribes.
std::vector<double> HostRecompose(const std::vector<std::vector<double>> &comp,
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

// The inverse scan, to read the result's parts back out of a big
// ciphertext's coefficients.
std::vector<std::vector<double>> HostComponents(
    const std::vector<double> &coeffs, int rank, int small_degree) {
  std::vector<std::vector<double>> comp(rank,
                                        std::vector<double>(small_degree));
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

void RunChain(int sub_degree, uint64_t seed) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);
  Ring lifted(kLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int level = big.param->max_level_;
  ASSERT_EQ(level, 1) << "the chain is supposed to hold exactly one level";
  const double scale = big.param->GetScale(level);

  CiSwitchedCcmmHandler<word> handler(big.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.rank, 16);
  ASSERT_EQ(layout.lanes, sub_degree);

  big.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                               level);
  big.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                      small.ui->GetSecretCoeffs(), level);
  const std::vector<int> rot = handler.LiftedRotationIndices();
  for (int idx : rot) lifted.ui->PrepareRotationKey(idx, level);

  // The contraction sums d/2 terms; 0.08 keeps the products two decades
  // under the level-1 modulus at both shapes.
  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, seed);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  seed + 1);

  // A big operand ciphertext: each of its parts' CI SinC messages is filled
  // per LocatePart, host-encoded to coefficients on the small ring, and the
  // rank parts recomposed into the big ring's coefficients.
  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    std::vector<Complex> part_msg(small.Degree());
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<std::vector<double>> part_coeffs(layout.rank);
      for (int j = 0; j < layout.rank; j++) {
        const int column = bi * layout.rank + j;
        for (int row = 0; row < layout.dim; row++) {
          for (int lane = 0; lane < layout.lanes; lane++) {
            int part, index;
            layout.LocatePart(row, column, lane, part, index);
            ASSERT_EQ(part, column);
            part_msg[index] = Complex(m[lane][row][column], 0.0);
          }
        }
        Plaintext<word> pt;
        small.context->encoder_.EncodeSinC(pt, level, scale, part_msg,
                                           sub_degree);
        small.context->encoder_.DecodeCoeff(part_coeffs[j], pt);
      }
      Plaintext<word> big_pt;
      big.context->encoder_.EncodeCoeff(
          big_pt, level, scale,
          HostRecompose(part_coeffs, layout.rank, small.Degree()));
      big.ui->Encrypt(out[bi], big_pt);
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);

  handler.Multiply(res, lhs, rhs, big.ui->GetRingSwitchKey(layout.rank),
                   big.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), layout.num_cts);

  for (const auto &ct : res) {
    ASSERT_EQ(big.param->NPToLevel(ct.GetNP()), level - 1)
        << "the batch CC-MM spends exactly one level";
    EXPECT_NEAR(ct.GetScale() / big.param->GetScale(level - 1), 2.0, 1e-6)
        << "the descent's trace doubles the recorded scale, exactly";
  }

  // Read the parts back out through the scan, and their lanes through the
  // small ring's own encoder.
  std::vector<std::vector<Complex>> got(layout.dim);
  const double bridge_scale = small.param->GetScale(level);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Plaintext<word> pt;
    big.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    big.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(bridge, level, bridge_scale,
                                          comp[j]);
      small.context->encoder_.DecodeSinC(got[bi * layout.rank + j], bridge,
                                         sub_degree);
    }
  }

  double worst = 0.0, sum = 0.0, transposed = 0.0, worst_imag = 0.0;
  double biggest = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0;
        for (int x = 0; x < layout.contraction; x++) {
          want += a[lane][row][x] * b[lane][x][column];
          want_t += a[lane][column][x] * b[lane][x][row];
        }
        biggest = std::max(biggest, std::abs(want));
        int part, index;
        layout.LocatePart(row, column, lane, part, index);
        const Complex &g = got[part][index];
        const double d = std::abs(g.real() - want);
        worst = std::max(worst, d);
        sum += d;
        worst_imag = std::max(worst_imag, std::abs(g.imag()));
        transposed = std::max(transposed, std::abs(g.real() - want_t));
      }
    }
  }
  const double mean = sum / (static_cast<double>(layout.lanes) * layout.dim *
                             layout.dim);

  std::cout << "CiSwitchedCcmm " << layout.num_cts << " x " << big.Degree()
            << " -> " << layout.dim << " x " << small.Degree() << " -> "
            << lifted.Degree() << " (" << layout.dim << "x"
            << layout.contraction << ")(" << layout.contraction << "x"
            << layout.dim << "), " << layout.lanes
            << " real lanes: max |diff| = " << worst << ", mean " << mean
            << ", max imag " << worst_imag << ", |want| <= " << biggest
            << std::endl;
  std::cout << "  control: transposed result " << transposed << std::endl;

  EXPECT_LT(worst, 2e-2) << "the chain did not land where the layout says";
  EXPECT_LT(worst_imag, 2e-2);
  EXPECT_GT(transposed, 1e-2)
      << "the result is symmetric enough that a transpose would pass, so "
         "this run is not checking the orientation";
}

}  // namespace

// The part addressing is total and involutive -- host-only.
TEST(CiSwitchedCcmm, ThePartAddressingTilesExactly) {
  CiSwitchedCcmmLayout layout(65536, 4096, 32);
  ASSERT_EQ(layout.rank, 16);
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  ASSERT_EQ(layout.contraction, 64);
  long long checked = 0;
  for (int row = 0; row < layout.dim; row++) {
    for (int column = 0; column < layout.dim; column++) {
      for (int lane = 0; lane < layout.lanes; lane++) {
        int part, index, r2, c2, l2;
        layout.LocatePart(row, column, lane, part, index);
        ASSERT_TRUE(layout.PositionPart(part, index, r2, c2, l2));
        ASSERT_EQ(r2, row);
        ASSERT_EQ(c2, column);
        ASSERT_EQ(l2, lane);
        checked++;
      }
    }
  }
  EXPECT_EQ(checked, static_cast<long long>(layout.dim) * layout.dim *
                         layout.lanes);
  EXPECT_EQ(checked, static_cast<long long>(layout.dim) *
                         static_cast<long long>(4096))
      << "the operand should fill its parts exactly, with no waste";
}

// The Llama shape: d = 128 = T = head_dim, 32 real lanes, contraction 64 --
// one half of a 128-deep contraction per call, two calls summed for a full
// attention product.
TEST(CiSwitchedCcmm, ProducesPerLaneProductsAtLlamaShape) {
  RunChain(32, 0x5C1CC1);
}

// The shape CiLiftTest validated the lifted primitive at, driven through
// the switch: d = 32, 128 real lanes, contraction 16.
TEST(CiSwitchedCcmm, ProducesPerLaneProductsAtTheLiftShape) {
  RunChain(128, 0x5C1CC2);
}
// ---------------------------------------------------------------------------
// The `_l2` chain and what it measured (Doing.md 1.5bo).
//
// The trio ci12_35_l2 / ci_ringswitch16_35_l2 / ringdegree13_35_l2 is the
// 1.5bm chain with one more terminal prime (a third level, so a conversion
// can precede Algorithm 4) and a TWO-prime auxiliary basis. These are
// correctness-lane parameters: the wider basis changes the key-switch
// modulus PQ, and [SYLPH]'s alpha/security analysis has not been redone --
// nothing here is a security claim.
//
// `CiSinCConverter` is the one-phase, one-level form of 1.5bn's conversions
// for rings that cannot host EvalSpecialFFT; its identity is validated below
// on fresh small-ring content.
//
// THE ROUTE 1.5bn PROPOSED IS MEASURED DEAD, and the second test pins why.
// "Parts arrive in slot form" is true of the SIGNAL -- the host-scan and the
// bare key switch are both clean -- but the switch's key-switch noise then
// rides the banded scan as an alternating SUFFIX SUM: a random walk,
// ~sqrt(ns/2) amplification with a 1/f spectrum, whose energy the full slot
// read concentrates onto the theta ~ 0 and pi slot family (0, 1, ns/2, ...)
// at ~n times the mode amplitude. Band-limited reads only ever see ~2k/pi of
// it, which is exactly the "switch noise floor" 1.5bg and 1.5bj measured; a
// slot read sees spikes three orders larger. Widening the auxiliary basis
// bought 4-5x (the ModDown floor, not 1/P, dominates) and no more. So the
// per-part conversion cannot follow the switch, and the CC-MM operands must
// be NESTED before it -- which is a layer-packing question, not a transform:
// the flat->nested basis change measures d diagonals wide, and neither a
// mid-chain diagonal nor diagonals at every stage boundary close it.
// ---------------------------------------------------------------------------

namespace {

constexpr const char *kSwitchL2Param = "ci_ringswitch16_35_l2.json";
constexpr const char *kSmallL2Param = "ci12_35_l2.json";

int BitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

int Log2(int v) {
  int r = 0;
  while ((1 << r) < v) r++;
  return r;
}

}  // namespace

// The converter alone, against the host encoder: 1.5bn's identity at the
// small ring, in one phase and one level -- possible where PrepareSinC needs
// two phases because a single phase carries no complex intermediate
// (Re(M x) = Re(M) x for real x).
TEST(CiSlotChain, TheOnePhaseConverterMatchesTheEncoder) {
  Ring small(kSmallL2Param);
  const int level = small.param->max_level_;
  ASSERT_EQ(level, 2) << "the _l2 chain holds two levels above its floor";
  const int degree = small.Degree();
  const int k = 128;
  const int d = degree / k;
  const int p = Log2(d);
  const double scale = small.param->GetScale(level);

  CiSinCConverter<word> conv(small.context, k, level, level);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  small.ui->PrepareRotationKey(req);

  std::mt19937_64 gen(0x51C0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Complex> msg(degree);
  for (auto &v : msg) v = Complex(dist(gen), 0.0);

  Plaintext<word> pt;
  small.context->encoder_.Encode(pt, level, scale, msg);
  Ciphertext<word> ct, sinc;
  small.ui->Encrypt(ct, pt);
  conv.SlotToSinC(small.context, sinc, ct, small.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(small.param->NPToLevel(sinc.GetNP()), level - 1);
  EXPECT_NEAR(sinc.GetScale() / ct.GetScale(), 1.0, 1e-6);

  Plaintext<word> out;
  small.ui->Decrypt(out, sinc);
  std::vector<Complex> got;
  small.context->encoder_.DecodeSinC(got, out, k);

  std::vector<Complex> want(degree);
  double fwd = 0.0, control = 0.0;
  for (int i = 0; i < d; i++) {
    for (int r = 0; r < k; r++) {
      const size_t at = static_cast<size_t>(i) * k + r;
      want[at] = msg[static_cast<size_t>(BitRev(i, p)) * k + r];
      fwd = std::max(fwd, std::abs(got[at].real() - want[at].real()));
      control = std::max(control, std::abs(got[at].real() - msg[at].real()));
    }
  }

  // The inverse, from a freshly encoded SinC ciphertext back to slots.
  Plaintext<word> inv_pt;
  small.context->encoder_.EncodeSinC(inv_pt, level, scale, want, k);
  Ciphertext<word> inv_ct, back;
  small.ui->Encrypt(inv_ct, inv_pt);
  conv.SinCToSlot(small.context, back, inv_ct, small.ui->GetEvkMap());
  Plaintext<word> back_pt;
  small.ui->Decrypt(back_pt, back);
  std::vector<Complex> undone;
  small.context->encoder_.Decode(undone, back_pt);
  double inv = 0.0;
  for (int i = 0; i < degree; i++) {
    inv = std::max(inv, std::abs(undone[i] - msg[i]));
  }

  std::cout << "one-phase converter, k = " << k << ": forward " << fwd
            << ", inverse " << inv << ", control (no bit reversal) "
            << control << std::endl;
  EXPECT_LT(fwd, 2e-5 * k) << "the read is the scan; the bound scales with "
                              "its ~2k/pi";
  EXPECT_LT(inv, 2e-2);
  EXPECT_GT(control, 1e-2);
}

// The finding, pinned as a regression: the switch's key-switch noise is
// clean until the scan walks it, and slot reads then concentrate the walk.
// Three clean stages and one dirty read, each asserted. If the LAST
// assertion ever fails -- the slot read coming back clean -- the noise floor
// moved and 1.5bo's route decision should be revisited, not the test.
TEST(CiSlotChain, TheScanWalksTheSwitchNoiseAndSlotReadsConcentrateIt) {
  Ring big(kSwitchL2Param);
  Ring small(kSmallL2Param);
  const int level = big.param->max_level_;
  const double scale = big.param->GetScale(level);
  const int ns = small.Degree();
  const int rank = big.Degree() / ns;

  big.ui->PrepareRingSwitchKey(ns, small.ui->GetSecretCoeffs(), level);
  cheddar::RingSwitchHandler<word> switcher(big.context, small.context);

  std::mt19937_64 gen(0x9A47);
  std::uniform_real_distribution<double> dist(-0.08, 0.08);
  std::vector<Complex> msg(big.Degree());
  for (auto &v : msg) v = Complex(dist(gen), 0.0);

  Plaintext<word> pt;
  big.context->encoder_.EncodeSinC(pt, level, scale, msg, ns);
  Ciphertext<word> ct;
  big.ui->Encrypt(ct, pt);

  // Stage 1: the big decrypt read through the HOST scan -- exact math, no
  // device switch. Clean.
  std::vector<double> big_c;
  {
    Plaintext<word> big_out;
    big.ui->Decrypt(big_out, ct);
    big.context->encoder_.DecodeCoeff(big_c, big_out);
  }
  const auto host_comp = HostComponents(big_c, rank, ns);
  std::vector<double> want_c;
  {
    std::vector<Complex> block(ns);
    for (int t = 0; t < ns; t++) block[t] = msg[static_cast<size_t>(1) * ns + t];
    Plaintext<word> ref;
    small.context->encoder_.Encode(ref, level, scale, block);
    small.context->encoder_.DecodeCoeff(want_c, ref);
  }
  double host_worst = 0.0;
  for (int t = 0; t < ns; t++) {
    host_worst = std::max(host_worst, std::abs(host_comp[1][t] - want_c[t]));
  }

  // Stage 2: the key switch alone, decrypted under the embedded subring
  // secret on the big ring. Clean -- the noise floor before the scan.
  double ks_worst = 0.0;
  {
    std::vector<int> emb(big.Degree(), 0);
    const auto &ssec = small.ui->GetSecretCoeffs();
    for (int j = 0; j < ns; j++) emb[static_cast<size_t>(j) * rank] = ssec[j];
    cheddar::UserInterface<word> emb_ui(big.context, emb);
    Ciphertext<word> switched;
    big.context->MultKey(switched, ct, big.ui->GetRingSwitchKey(rank));
    Plaintext<word> sw_pt;
    emb_ui.Decrypt(sw_pt, switched);
    std::vector<double> sw_c;
    big.context->encoder_.DecodeCoeff(sw_c, sw_pt);
    for (int t = 0; t < big.Degree(); t++) {
      ks_worst = std::max(ks_worst, std::abs(sw_c[t] - big_c[t]));
    }
  }

  // Stage 3: the full switch. The parts' COEFFICIENTS carry the scan's walk
  // of the key-switch noise -- ~sqrt(ns/2) over the floor, still small.
  std::vector<Ciphertext<word>> parts;
  switcher.Switch(parts, ct, big.ui->GetRingSwitchKey(rank));
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  double coeff_worst = 0.0;
  double slot_worst = 0.0;
  {
    Plaintext<word> out;
    small.ui->Decrypt(out, parts[1]);
    std::vector<double> dev_c;
    small.context->encoder_.DecodeCoeff(dev_c, out);
    for (int t = 0; t < ns; t++) {
      coeff_worst = std::max(coeff_worst, std::abs(dev_c[t] - want_c[t]));
    }
    // Stage 4: the slot read of the same part -- the walk's 1/f energy
    // concentrated onto the theta ~ 0 / pi slot family. Dirty, by physics.
    std::vector<Complex> slots;
    small.context->encoder_.Decode(slots, out);
    for (int s2 = 0; s2 < ns; s2++) {
      slot_worst = std::max(
          slot_worst,
          std::abs(slots[s2].real() - msg[static_cast<size_t>(1) * ns + s2].real()));
    }
  }

  std::cout << "switch noise, staged: host scan " << host_worst
            << ", key switch alone " << ks_worst << ", part coefficients "
            << coeff_worst << ", part slot read " << slot_worst << std::endl;
  EXPECT_LT(host_worst, 1e-4) << "the exact-math path must be clean";
  EXPECT_LT(ks_worst, 1e-4) << "the key switch itself must be clean";
  EXPECT_LT(coeff_worst, 3e-4)
      << "the walk should stay ~sqrt(ns/2) over the key-switch floor";
  EXPECT_GT(slot_worst, 5e-3)
      << "the slot read came back CLEAN: the noise floor moved, and the "
         "route decision of Doing.md 1.5bo should be revisited";
}

// ---------------------------------------------------------------------------
// The mixed-radix identity of the banded recompose, and the packing it buys
// (Doing.md 1.5bp).
//
// 1.5bo closed the transform routes to the nested arrangement: no
// diagonal-decorated map carries a FLAT big SinC ciphertext to the chain's
// part-level operand, and no conversion may follow the switch. What remained
// was "a layer-packing decision" -- and this is it. The banded recompose is
// mixed-radix associative:
//
//     rec_rank( { rec_perpart(pcomp_I) } )  =  rec_big( g(pcomp) ),
//     g(pcomp)[row * rank + I] = pcomp_I[row] + [I != 0] pcomp_{rank-I}[row+1]
//
// so the nested operand IS a flat big-ring SinC ciphertext at the same
// sub-degree, provided its message carries each entry at two block
// addresses (LocateSlot). The route to the chain from big-ring slots is
// then: pack the two-term sums in slot form (clean data, one
// block-permutation add), run the flat CI SlotToSinC -- here the one-phase
// CiSinCConverter, in its first big-ring use -- and hand the result to
// Multiply unchanged. Nothing converts at the switch boundary, which is
// exactly what 1.5bo's noise physics demands.
// ---------------------------------------------------------------------------

namespace {

constexpr const char *kLiftedL2Param = "ringdegree13_35_l2.json";

// The block-level scan: g's exact inverse, applied to a flat SinC message
// read off a result ciphertext to recover the per-part (nested) content.
// Same recursion as the coefficient scan, at block granularity, vectorised
// over the lanes.
std::vector<std::vector<std::vector<double>>> HostBlockScan(
    const std::vector<Complex> &fm, int rank, int dim, int lanes) {
  std::vector<std::vector<std::vector<double>>> comp(
      rank, std::vector<std::vector<double>>(dim,
                                             std::vector<double>(lanes, 0.0)));
  auto at = [&](int block, int r) {
    return fm[static_cast<size_t>(block) * lanes + r].real();
  };
  for (int bp = 0; bp < dim; bp++) {
    for (int r = 0; r < lanes; r++) comp[0][bp][r] = at(bp * rank, r);
  }
  for (int i = 1; i <= rank / 2; i++) {
    const int mi = rank - i;
    std::vector<double> acc_i(lanes, 0.0), acc_m(lanes, 0.0);
    for (int bp = dim - 1; bp >= 0; bp--) {
      for (int r = 0; r < lanes; r++) {
        const double new_i = at(bp * rank + i, r) - acc_m[r];
        const double new_m = at(bp * rank + mi, r) - acc_i[r];
        comp[i][bp][r] = new_i;
        comp[mi][bp][r] = new_m;  // mi == i rewrites the same value
        acc_i[r] = new_i;
        acc_m[r] = new_m;
      }
    }
  }
  return comp;
}

}  // namespace

// The identity itself, against the real encoders and exactly: one big
// ciphertext's worth of per-part SinC messages, encoded nested (per-part
// EncodeSinC + the switch recompose) and flat (one big EncodeSinC of the
// two-term block sums), must produce the SAME coefficients -- and
// LocateSlot's two addresses must build the same flat message.
TEST(CiNestedPacking, TheFlatEncodingOfTheBlockSumsIsTheNestedOperand) {
  Ring big(kSwitchParam);
  Ring small(kSmallParam);
  const int level = big.param->max_level_;
  const double scale = big.param->GetScale(level);

  for (const int sub_degree : {32, 128}) {
    CiSwitchedCcmmLayout layout(big.Degree(), small.Degree(), sub_degree);
    const int k = layout.lanes;
    const int ds = layout.dim;
    const int rank = layout.rank;
    const int log_blocks = Log2(rank * ds);

    std::mt19937_64 gen(0xB9 + sub_degree);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<Complex>> part_msg(
        rank, std::vector<Complex>(small.Degree()));
    for (auto &m : part_msg) {
      for (auto &v : m) v = Complex(dist(gen), 0.0);
    }

    // Nested: per-part encode, then the switch recompose.
    std::vector<std::vector<double>> part_coeffs(rank);
    for (int j = 0; j < rank; j++) {
      Plaintext<word> pt;
      small.context->encoder_.EncodeSinC(pt, level, scale, part_msg[j],
                                         sub_degree);
      small.context->encoder_.DecodeCoeff(part_coeffs[j], pt);
    }
    const auto nested = HostRecompose(part_coeffs, rank, small.Degree());

    // Flat: the two-term block sums, one big EncodeSinC.
    std::vector<Complex> fmsg(big.Degree(), Complex(0.0, 0.0));
    for (int bp = 0; bp < ds; bp++) {
      for (int i = 0; i < rank; i++) {
        for (int r = 0; r < k; r++) {
          Complex v = part_msg[i][static_cast<size_t>(bp) * k + r];
          if (i != 0 && bp + 1 < ds) {
            v += part_msg[rank - i][static_cast<size_t>(bp + 1) * k + r];
          }
          fmsg[(static_cast<size_t>(bp) * rank + i) * k + r] = v;
        }
      }
    }
    Plaintext<word> flat_pt;
    big.context->encoder_.EncodeSinC(flat_pt, level, scale, fmsg, sub_degree);
    std::vector<double> flat;
    big.context->encoder_.DecodeCoeff(flat, flat_pt);

    double worst = 0.0;
    for (int t = 0; t < big.Degree(); t++) {
      worst = std::max(worst, std::abs(nested[t] - flat[t]));
    }
    std::cout << "mixed-radix identity, sub_degree " << sub_degree
              << ": max |nested - flat(g)| = " << worst << std::endl;
    EXPECT_LT(worst, 1e-7) << "the identity must hold to encode rounding";

    // LocateSlot builds the same flat message through the slot map.
    std::vector<Complex> slot_msg(big.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < ds; row++) {
      for (int i = 0; i < rank; i++) {
        for (int r = 0; r < k; r++) {
          const Complex v = part_msg[i][static_cast<size_t>(row) * k + r];
          int ct_idx, slot, copy_slot;
          const int n = layout.LocateSlot(row, i, r, ct_idx, slot, copy_slot);
          ASSERT_EQ(ct_idx, 0);
          slot_msg[slot] += v;
          if (n == 2) slot_msg[copy_slot] += v;
        }
      }
    }
    double slot_worst = 0.0;
    for (int b = 0; b < rank * ds; b++) {
      for (int r = 0; r < k; r++) {
        const Complex direct = fmsg[static_cast<size_t>(b) * k + r];
        const Complex located =
            slot_msg[static_cast<size_t>(BitRev(b, log_blocks)) * k + r];
        slot_worst = std::max(slot_worst, std::abs(direct - located));
      }
    }
    EXPECT_LT(slot_worst, 1e-12)
        << "LocateSlot disagrees with the identity's own construction";
  }
}

// The route, end to end on the device: slot-form big ciphertexts carrying
// the two-term sums, the one-phase converter's first big-ring outing (512
// diagonals at sub_degree 128), the unchanged chain, and both reads of the
// result -- the part-level read of 1.5bm and the flat read undone by the
// block scan. The _l2 trio's third level is exactly this transform's budget.
TEST(CiNestedPacking, ASlotCiphertextReachesTheChainThroughTheConverter) {
  Ring big(kSwitchL2Param);
  Ring small(kSmallL2Param);
  Ring lifted(kLiftedL2Param,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int top = big.param->max_level_;
  ASSERT_EQ(top, 2) << "the _l2 chain holds the conversion level";
  const int chain_level = top - 1;
  const int sub_degree = 128;

  CiSwitchedCcmmHandler<word> handler(big.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.num_cts, 2);

  big.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                               chain_level);
  big.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                      small.ui->GetSecretCoeffs(),
                                      chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(big.context, sub_degree, /*forward_level=*/top,
                             /*inverse_level=*/-1);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  big.ui->PrepareRotationKey(req);

  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, 0x1B9A);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  0x1B9B);

  // A big operand from SLOT form: every entry added at both LocateSlot
  // addresses, encoded on the slots, and descended by the converter.
  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<Complex> slot_msg(big.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            const double v = m[lane][row][column];
            if (v == 0.0) continue;
            int ct_idx, slot, copy_slot;
            const int n =
                layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            ASSERT_EQ(ct_idx, bi);
            slot_msg[slot] += v;
            if (n == 2) slot_msg[copy_slot] += v;
          }
        }
      }
      Plaintext<word> pt;
      big.context->encoder_.Encode(pt, top, big.param->GetScale(top),
                                   slot_msg);
      Ciphertext<word> enc;
      big.ui->Encrypt(enc, pt);
      conv.SlotToSinC(big.context, out[bi], enc, big.ui->GetEvkMap());
      ASSERT_EQ(big.param->NPToLevel(out[bi].GetNP()), chain_level);
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // The converter alone, on the big ring, measured in both bases: the
  // COEFFICIENT error against the host flat encode is the transform's own
  // noise; the lane read then pays the scan's conditioning on top of it.
  double conv_err = 0.0;
  double conv_coeff_err = 0.0;
  {
    std::vector<Complex> fmsg(big.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int i = 0; i < layout.rank; i++) {
        for (int r = 0; r < layout.lanes; r++) {
          double want = b[r][row][i];
          if (i != 0 && row + 1 < layout.dim) {
            want += b[r][row + 1][layout.rank - i];
          }
          fmsg[(static_cast<size_t>(row) * layout.rank + i) * layout.lanes +
               r] = Complex(want, 0.0);
        }
      }
    }
    Plaintext<word> probe;
    big.ui->Decrypt(probe, rhs[0]);
    std::vector<Complex> got;
    big.context->encoder_.DecodeSinC(got, probe, sub_degree);
    for (size_t at = 0; at < fmsg.size(); at++) {
      conv_err = std::max(conv_err, std::abs(got[at].real() - fmsg[at].real()));
    }
    std::vector<double> got_c, want_c;
    big.context->encoder_.DecodeCoeff(got_c, probe);
    Plaintext<word> host_pt;
    big.context->encoder_.EncodeSinC(host_pt, chain_level,
                                     big.param->GetScale(top), fmsg,
                                     sub_degree);
    big.context->encoder_.DecodeCoeff(want_c, host_pt);
    for (int t = 0; t < big.Degree(); t++) {
      conv_coeff_err =
          std::max(conv_coeff_err, std::abs(got_c[t] - want_c[t]));
    }
  }

  handler.Multiply(res, lhs, rhs, big.ui->GetRingSwitchKey(layout.rank),
                   big.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), layout.num_cts);
  // The converter preserves its INPUT scale -- GetScale(top), not the
  // canonical scale of chain_level -- so the chain's exact output scale is
  // stated from there: Algorithm 4 squares it and rescales by chain_level's
  // prime product, and the descent doubles it.
  const double operand_scale = big.param->GetScale(top);
  const double product_scale = 2.0 * operand_scale * operand_scale /
                               big.param->GetRescalePrimeProd(chain_level);
  for (const auto &ct : res) {
    ASSERT_EQ(big.param->NPToLevel(ct.GetNP()), chain_level - 1);
    EXPECT_NEAR(ct.GetScale() / product_scale, 1.0, 1e-6);
  }

  // Read 1, part-level (1.5bm's): the coefficient scan, each part's lanes
  // through the small ring's encoder.
  std::vector<std::vector<Complex>> got_parts(layout.dim);
  // Read 2, flat: DecodeSinC on the big ring, then the block scan undoes g.
  std::vector<std::vector<std::vector<std::vector<double>>>> got_flat;
  const double bridge_scale = small.param->GetScale(chain_level);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Plaintext<word> pt;
    big.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    big.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(bridge, chain_level, bridge_scale,
                                          comp[j]);
      small.context->encoder_.DecodeSinC(got_parts[bi * layout.rank + j],
                                         bridge, sub_degree);
    }
    std::vector<Complex> fm;
    big.context->encoder_.DecodeSinC(fm, pt, sub_degree);
    got_flat.push_back(
        HostBlockScan(fm, layout.rank, layout.dim, layout.lanes));
  }

  double worst_part = 0.0, worst_flat = 0.0, transposed = 0.0;
  double no_unscan = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0;
        for (int x = 0; x < layout.contraction; x++) {
          want += a[lane][row][x] * b[lane][x][column];
          want_t += a[lane][column][x] * b[lane][x][row];
        }
        int part, index;
        layout.LocatePart(row, column, lane, part, index);
        const double got_p = got_parts[part][index].real();
        worst_part = std::max(worst_part, std::abs(got_p - want));
        transposed = std::max(transposed, std::abs(got_p - want_t));
        const int bi = column / layout.rank;
        const int cls = column % layout.rank;
        worst_flat = std::max(
            worst_flat, std::abs(got_flat[bi][cls][row][lane] - want));
      }
    }
  }
  // The control that pins g as load-bearing on the read side: the flat
  // message itself, NOT unscanned, is the block sums and must miss the
  // products wherever a partner is live.
  {
    Plaintext<word> pt;
    big.ui->Decrypt(pt, res[0]);
    std::vector<Complex> fm;
    big.context->encoder_.DecodeSinC(fm, pt, sub_degree);
    for (int row = 0; row + 1 < layout.dim; row++) {
      for (int i = 1; i < layout.rank; i++) {
        for (int r = 0; r < layout.lanes; r++) {
          double want = 0.0;
          for (int x = 0; x < layout.contraction; x++) {
            want += a[r][row][x] * b[r][x][i];
          }
          const size_t at =
              (static_cast<size_t>(row) * layout.rank + i) * layout.lanes + r;
          no_unscan = std::max(no_unscan, std::abs(fm[at].real() - want));
        }
      }
    }
  }

  std::cout << "slot -> converter -> chain, sub_degree " << sub_degree
            << ": converter coeff " << conv_coeff_err << " / lane read "
            << conv_err << ", products (part read) " << worst_part
            << ", (flat read) " << worst_flat << std::endl;
  std::cout << "  controls: transposed " << transposed << ", flat without "
            << "the block scan " << no_unscan << std::endl;

  EXPECT_LT(conv_coeff_err, 2e-3)
      << "the one-phase big-ring transform's own noise moved";
  EXPECT_LT(conv_err, 4e-2)
      << "the lane read pays the scan's conditioning over the coefficient "
         "floor";
  EXPECT_LT(worst_part, 2e-2)
      << "the converted operands did not land where LocatePart says";
  EXPECT_LT(worst_flat, 2e-2)
      << "the flat read plus the block scan must recover the same products";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(no_unscan, 3e-2)
      << "the raw flat message already matches the products, so the block "
         "sums are not present and this run is not testing the identity";
}

// ---------------------------------------------------------------------------
// The folds, and the loop they close (Doing.md 1.5bq).
//
// 1.5bp packed the two-term sums on the HOST and read the result back
// through a HOST block scan; the leg cannot. Both maps live on the same
// stride-k diagonal lattice as the composed conversion itself, so they FOLD
// into it -- the copy-add as a column relabelling of the forward matrix
// (M -> M (I + pi2)), the block scan as a row recombination of the inverse
// (M -> G M) -- and the diagonal count cannot pass its ceiling degree / k.
// No extra level, no extra rotations: the converter consumes slots holding
// each entry ONCE (its primary LocateSlot address, a bijective layout) and
// returns them the same way.
//
// The _l3 trio (one more terminal prime, 38535169) holds the whole loop:
// encrypt at 3, nested forward -> 2, the chain -> 1, nested inverse -> 0.
// Slots to slots, every arrow homomorphic.
// ---------------------------------------------------------------------------

namespace {

constexpr const char *kSwitchL3Param = "ci_ringswitch16_35_l3.json";
constexpr const char *kSmallL3Param = "ci12_35_l3.json";
constexpr const char *kLiftedL3Param = "ringdegree13_35_l3.json";

}  // namespace

TEST(CiNestedPacking, TheNestedConverterClosesTheLoop) {
  Ring big(kSwitchL3Param);
  Ring small(kSmallL3Param);
  Ring lifted(kLiftedL3Param,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int top = big.param->max_level_;
  ASSERT_EQ(top, 3) << "the _l3 chain holds both conversion levels";
  const int chain_level = top - 1;
  const int inverse_level = chain_level - 1;
  const int sub_degree = 128;

  CiSwitchedCcmmHandler<word> handler(big.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.num_cts, 2);

  big.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                               chain_level);
  big.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                      small.ui->GetSecretCoeffs(),
                                      chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(big.context, sub_degree, /*forward_level=*/top,
                             /*inverse_level=*/inverse_level, &layout);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  big.ui->PrepareRotationKey(req);

  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, 0x1B9C);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  0x1B9D);

  // Primary addresses only, by ASSIGNMENT: the nested forward owns the
  // copy-add now, so the slot layout is bijective -- what a leg would
  // actually hold.
  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<Complex> slot_msg(big.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            slot_msg[slot] = Complex(m[lane][row][column], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      big.context->encoder_.Encode(pt, top, big.param->GetScale(top),
                                   slot_msg);
      Ciphertext<word> enc;
      big.ui->Encrypt(enc, pt);
      conv.SlotToSinC(big.context, out[bi], enc, big.ui->GetEvkMap());
      ASSERT_EQ(big.param->NPToLevel(out[bi].GetNP()), chain_level);
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Stage 1: the folded forward produced the two-address flat encoding --
  // the coefficient probe against the host encode of the SUMMED message.
  double fwd_coeff_err = 0.0;
  {
    std::vector<Complex> fmsg(big.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int i = 0; i < layout.rank; i++) {
        for (int r = 0; r < layout.lanes; r++) {
          double want = b[r][row][i];
          if (i != 0 && row + 1 < layout.dim) {
            want += b[r][row + 1][layout.rank - i];
          }
          fmsg[(static_cast<size_t>(row) * layout.rank + i) * layout.lanes +
               r] = Complex(want, 0.0);
        }
      }
    }
    Plaintext<word> probe;
    big.ui->Decrypt(probe, rhs[0]);
    std::vector<double> got_c, want_c;
    big.context->encoder_.DecodeCoeff(got_c, probe);
    Plaintext<word> host_pt;
    big.context->encoder_.EncodeSinC(host_pt, chain_level,
                                     big.param->GetScale(top), fmsg,
                                     sub_degree);
    big.context->encoder_.DecodeCoeff(want_c, host_pt);
    for (int t = 0; t < big.Degree(); t++) {
      fwd_coeff_err =
          std::max(fwd_coeff_err, std::abs(got_c[t] - want_c[t]));
    }
  }

  handler.Multiply(res, lhs, rhs, big.ui->GetRingSwitchKey(layout.rank),
                   big.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Stage 2: the folded inverse returns the loop to slots, at level 0.
  std::vector<std::vector<Complex>> slots(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(big.context, back, res[bi], big.ui->GetEvkMap());
    ASSERT_EQ(big.param->NPToLevel(back.GetNP()), inverse_level - 1);
    Plaintext<word> pt;
    big.ui->Decrypt(pt, back);
    big.context->encoder_.Decode(slots[bi], pt);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  double worst_loop = 0.0, transposed = 0.0, unsummed = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0, want_sum = 0.0;
        for (int x = 0; x < layout.contraction; x++) {
          want += a[lane][row][x] * b[lane][x][column];
          want_t += a[lane][column][x] * b[lane][x][row];
        }
        const int cls = column % layout.rank;
        if (cls != 0 && row + 1 < layout.dim) {
          const int partner =
              (column / layout.rank) * layout.rank + (layout.rank - cls);
          for (int x = 0; x < layout.contraction; x++) {
            want_sum += a[lane][row + 1][x] * b[lane][x][partner];
          }
        }
        int ct_idx, slot, copy_slot;
        layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
        const double got = slots[ct_idx][slot].real();
        worst_loop = std::max(worst_loop, std::abs(got - want));
        transposed = std::max(transposed, std::abs(got - want_t));
        // The scan-undone value differs from the raw block sum by the
        // partner product; if they agree everywhere the fold did nothing.
        unsummed = std::max(unsummed, std::abs(want_sum));
      }
    }
  }

  // Stage 3: the inverse alone, isolated from the chain -- a freshly
  // encoded flat ciphertext of KNOWN block sums must come back unsummed.
  double inv_err = 0.0;
  {
    std::mt19937_64 gen(0x1B9E);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> s(
        layout.dim * layout.rank, std::vector<double>(layout.lanes));
    for (auto &blk : s) {
      for (auto &v : blk) v = dist(gen);
    }
    std::vector<Complex> gs(big.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int i = 0; i < layout.rank; i++) {
        for (int r = 0; r < layout.lanes; r++) {
          double v = s[row * layout.rank + i][r];
          if (i != 0 && row + 1 < layout.dim) {
            v += s[(row + 1) * layout.rank + (layout.rank - i)][r];
          }
          gs[(static_cast<size_t>(row) * layout.rank + i) * layout.lanes +
             r] = Complex(v, 0.0);
        }
      }
    }
    Plaintext<word> pt;
    big.context->encoder_.EncodeSinC(pt, inverse_level,
                                     big.param->GetScale(inverse_level), gs,
                                     sub_degree);
    Ciphertext<word> enc, back;
    big.ui->Encrypt(enc, pt);
    conv.SinCToSlot(big.context, back, enc, big.ui->GetEvkMap());
    Plaintext<word> out;
    big.ui->Decrypt(out, back);
    std::vector<Complex> got;
    big.context->encoder_.Decode(got, out);
    for (int row = 0; row < layout.dim; row++) {
      for (int i = 0; i < layout.rank; i++) {
        for (int r = 0; r < layout.lanes; r++) {
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, i, r, ct_idx, slot, copy_slot);
          inv_err = std::max(
              inv_err,
              std::abs(got[slot].real() - s[row * layout.rank + i][r]));
        }
      }
    }
  }

  std::cout << "nested converter loop, sub_degree " << sub_degree
            << ": forward coeff " << fwd_coeff_err << ", loop products "
            << worst_loop << ", inverse alone " << inv_err << std::endl;
  std::cout << "  controls: transposed " << transposed
            << ", largest partner product the scan removed " << unsummed
            << std::endl;

  EXPECT_LT(fwd_coeff_err, 2e-3)
      << "the folded forward does not produce the two-address encoding";
  EXPECT_LT(worst_loop, 5e-2)
      << "the homomorphic loop did not return the products to their "
         "primary slots";
  EXPECT_LT(inv_err, 2e-2) << "the folded inverse does not undo the sums";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(unsummed, 1e-2)
      << "no partner product was ever live, so the fold went unexercised";
}

// ---------------------------------------------------------------------------
// The Llama alignment, and the two-call contraction sum (Doing.md 1.5br).
//
// sub_degree 32 is the layer's own shape: d = 128 = T = head_dim, 32 real
// lanes = 32 heads, one 128 x 128 product per lane. The half-contraction
// contract caps one call at depth d/2 = 64, so the full 128-deep product is
// TWO calls summed -- and the packing makes the split nearly free on the
// lhs: its 128 columns are big ciphertexts 0..3 and 4..7 of ONE packing
// (the copy-add mixes only within a big ciphertext), so call 2 simply hands
// the upper four over as its bundle. Only the rhs is packed per call, its
// contraction rows placed into blocks 0..63 both times.
//
// This is also the sub-32 nested converter's first outing -- 2048 diagonals
// against the lift shape's 512 -- and the timings printed here are the cost
// side of the route question 1.5bq left open (one-phase converter vs the
// multi-phase transform with the folds in its boundary phases).
// ---------------------------------------------------------------------------

TEST(CiNestedPacking, TheTwoCallSumClosesTheLlamaContraction) {
  Ring big(kSwitchL3Param);
  Ring small(kSmallL3Param);
  Ring lifted(kLiftedL3Param,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int top = big.param->max_level_;
  ASSERT_EQ(top, 3);
  const int chain_level = top - 1;
  const int inverse_level = chain_level - 1;
  const int sub_degree = 32;

  CiSwitchedCcmmHandler<word> handler(big.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  ASSERT_EQ(layout.contraction, 64);

  big.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                               chain_level);
  big.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                      small.ui->GetSecretCoeffs(),
                                      chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }

  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> conv(big.context, sub_degree, /*forward_level=*/top,
                             /*inverse_level=*/inverse_level, &layout);
  const auto t1 = std::chrono::steady_clock::now();
  EvkRequest req;
  conv.AddRequiredRotations(req);
  big.ui->PrepareRotationKey(req);

  // A full 128 x 128 per lane on both sides: the contraction is 128 deep
  // and no single call may take it.
  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0x11A3);
  const RealBatch full_b = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                       layout.dim, 0.08, 0x11A4);
  // The rhs of call c holds rows 64c .. 64c+63 of B in blocks 0..63.
  auto rhs_half = [&](int call) {
    RealBatch h(layout.lanes,
                std::vector<std::vector<double>>(
                    layout.dim, std::vector<double>(layout.dim, 0.0)));
    for (int t = 0; t < layout.lanes; t++) {
      for (int x = 0; x < layout.contraction; x++) {
        h[t][x] = full_b[t][call * layout.contraction + x];
      }
    }
    return h;
  };

  // `first_ct` picks which big ciphertexts of the operand's own packing are
  // built: the lhs of call 2 is big ciphertexts 4..7 of A's single packing.
  double fwd_seconds = 0.0;
  int fwd_count = 0;
  auto build = [&](const RealBatch &m, int first_ct, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int i = 0; i < num_big; i++) {
      const int bi = first_ct + i;
      std::vector<Complex> slot_msg(big.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            ASSERT_EQ(ct_idx, bi);
            slot_msg[slot] = Complex(m[lane][row][column], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      big.context->encoder_.Encode(pt, top, big.param->GetScale(top),
                                   slot_msg);
      Ciphertext<word> enc;
      big.ui->Encrypt(enc, pt);
      const auto f0 = std::chrono::steady_clock::now();
      conv.SlotToSinC(big.context, out[i], enc, big.ui->GetEvkMap());
      cudaDeviceSynchronize();
      fwd_seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - f0)
                         .count();
      fwd_count++;
    }
  };

  std::vector<Ciphertext<word>> res;
  double mult_seconds = 0.0;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> lhs, rhs, part;
    build(a, call * layout.num_cts / 2, layout.num_cts / 2, lhs);
    build(rhs_half(call), 0, layout.num_cts, rhs);
    const auto m0 = std::chrono::steady_clock::now();
    handler.Multiply(part, lhs, rhs, big.ui->GetRingSwitchKey(layout.rank),
                     big.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    mult_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - m0)
            .count();
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        big.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  // The loop back to slots, and the full-depth reference.
  double worst = 0.0, transposed = 0.0, deepest_upper = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(big.context, back, res[bi], big.ui->GetEvkMap());
    Plaintext<word> pt;
    big.ui->Decrypt(pt, back);
    std::vector<Complex> slots;
    big.context->encoder_.Decode(slots, pt);
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        const int column = bi * layout.rank + j;
        for (int lane = 0; lane < layout.lanes; lane++) {
          double want = 0.0, want_t = 0.0, upper = 0.0;
          for (int x = 0; x < layout.dim; x++) {
            want += a[lane][row][x] * full_b[lane][x][column];
            want_t += a[lane][column][x] * full_b[lane][x][row];
          }
          for (int x = layout.contraction; x < layout.dim; x++) {
            upper += a[lane][row][x] * full_b[lane][x][column];
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          worst = std::max(worst, std::abs(got - want));
          transposed = std::max(transposed, std::abs(got - want_t));
          deepest_upper = std::max(deepest_upper, std::abs(upper));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "two-call sum at the Llama alignment (128x128, depth 64+64, "
            << layout.lanes << " lanes): products " << worst << std::endl;
  std::cout << "  controls: transposed " << transposed
            << ", largest upper-half contribution " << deepest_upper
            << std::endl;
  std::cout << "  cost of the sub-32 nested converter: build "
            << std::chrono::duration<double>(t1 - t0).count()
            << " s, forward "
            << fwd_seconds / std::max(fwd_count, 1) << " s/ct over "
            << fwd_count << " cts, one chain call "
            << mult_seconds / 2.0 << " s" << std::endl;

  EXPECT_LT(worst, 5e-2)
      << "the two half-contractions do not sum to the 128-deep product";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(deepest_upper, 3e-2)
      << "the upper half of the contraction never mattered, so this run "
         "is not testing the sum";
}

// ---------------------------------------------------------------------------
// RoPE in the operand layouts, rotation-free (Doing.md 1.5bs).
//
// Llama's rotate_half pairs channel m with channel m + 64 -- and the
// half-contraction split of 1.5br cuts the CONTRACTION axis at exactly 64,
// which for the scores product is the channel axis on BOTH operands. So the
// pairing never crosses a ciphertext's own slots:
//
//   * Q (lhs): channel = column; channels m and m + 64 are big ciphertexts
//     k and k + 4 of the one nested packing, same slot. RoPE is plaintext
//     multiplies and ciphertext adds ACROSS the ct pair -- zero rotations,
//     and the cos/sin masks of ct k serve ct k + 4 unchanged.
//   * K (rhs): channel = row, and the per-call packing puts channels m and
//     m + 64 at the SAME block of the two calls' bundles. RoPE is the same
//     ct-pair arithmetic across the bundles -- zero rotations, and the two
//     calls share their masks.
//
// One plaintext-multiply level each, nothing else. The angles ride the
// masks: theta depends on (channel pair, token) = (a slot field, a slot
// field), so a per-ciphertext plaintext encodes them exactly.
//
// The scores also settle SoftMax's placement without running it: the output
// lands at (row = query, column = key, lane = head) primary addresses, so
// the key axis SoftMax reduces over is the column axis -- four slot-field
// rotations plus ciphertext adds -- and P then IS the next product's lhs
// with its contraction split already on the big-ciphertext boundary.
// ---------------------------------------------------------------------------

TEST(CiNestedPacking, RopeRidesTheHalfContractionSplitWithoutRotations) {
  Ring big(kSwitchL3Param);
  Ring small(kSmallL3Param);
  Ring lifted(kLiftedL3Param,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int top = big.param->max_level_;
  ASSERT_EQ(top, 3);
  const int conv_level = top - 1;      // RoPE spends the top level
  const int chain_level = conv_level - 1;
  const int sub_degree = 32;

  CiSwitchedCcmmHandler<word> handler(big.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.num_cts, 8);
  const int half = layout.contraction;  // 64: the split, and RoPE's pair gap

  big.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                               chain_level);
  big.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                      small.ui->GetSecretCoeffs(),
                                      chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(big.context, sub_degree,
                             /*forward_level=*/conv_level,
                             /*inverse_level=*/-1, &layout);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  big.ui->PrepareRotationKey(req);

  // Q[lane][token][channel], K[lane][token][channel]; theta_m = base^(-2m/d).
  const RealBatch q = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0x40E1);
  const RealBatch kk = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                   layout.dim, 0.08, 0x40E2);
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), s = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * s;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * s;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  const double pt_scale = big.param->GetRescalePrimeProd(top);
  // One slot-form ciphertext of an operand packing, entries at primary
  // addresses, NOT yet converted.
  auto encrypt_slots = [&](const RealBatch &m, int bi, Ciphertext<word> &out) {
    std::vector<Complex> slot_msg(big.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        for (int lane = 0; lane < layout.lanes; lane++) {
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, bi * layout.rank + j, lane, ct_idx, slot,
                            copy_slot);
          slot_msg[slot] = Complex(m[lane][row][bi * layout.rank + j], 0.0);
        }
      }
    }
    Plaintext<word> pt;
    big.context->encoder_.Encode(pt, top, big.param->GetScale(top), slot_msg);
    big.ui->Encrypt(out, pt);
  };
  // A mask over one ciphertext's slots, from (row, cls, lane) -> value.
  auto mask = [&](int bi, auto value, Plaintext<word> &pt) {
    std::vector<Complex> msg(big.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        for (int lane = 0; lane < layout.lanes; lane++) {
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, bi * layout.rank + j, lane, ct_idx, slot,
                            copy_slot);
          msg[slot] = Complex(value(row, j), 0.0);
        }
      }
    }
    big.context->encoder_.Encode(pt, top, pt_scale, msg);
  };
  // The rotation-free RoPE pair: lo' = lo cos - hi sin, hi' = hi cos +
  // lo sin, rescaled back to conv_level.
  auto rope_pair = [&](Ciphertext<word> &lo, Ciphertext<word> &hi,
                       const Plaintext<word> &cos_pt,
                       const Plaintext<word> &sin_pt,
                       const Plaintext<word> &neg_sin_pt) {
    Ciphertext<word> a, b;
    big.context->Mult(a, lo, cos_pt);
    big.context->Mult(b, hi, neg_sin_pt);
    big.context->Add(a, a, b);
    big.context->Mult(b, hi, cos_pt);
    Ciphertext<word> d;
    big.context->Mult(d, lo, sin_pt);
    big.context->Add(b, b, d);
    big.context->Rescale(lo, a);
    big.context->Rescale(hi, b);
  };

  // Q: one packing of eight; RoPE pairs ct k with ct k + 4, masks indexed by
  // the pair's shared m = k * rank + cls, angle by the QUERY token = row.
  std::vector<Ciphertext<word>> q_cts(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) encrypt_slots(q, bi, q_cts[bi]);
  for (int k = 0; k < layout.num_cts / 2; k++) {
    Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
    auto ang = [&](int row, int cls) {
      return static_cast<double>(row) * theta[k * layout.rank + cls];
    };
    mask(k, [&](int r, int c) { return std::cos(ang(r, c)); }, cos_pt);
    mask(k, [&](int r, int c) { return std::sin(ang(r, c)); }, sin_pt);
    mask(k, [&](int r, int c) { return -std::sin(ang(r, c)); }, neg_sin_pt);
    rope_pair(q_cts[k], q_cts[k + layout.num_cts / 2], cos_pt, sin_pt,
              neg_sin_pt);
  }

  // K: the two calls' bundles, channels m and m + 64 at the same block;
  // RoPE pairs bundle 1 with bundle 2 ciphertext by ciphertext, masks
  // indexed by the channel = row block, angle by the KEY token = column.
  std::vector<Ciphertext<word>> k_lo(layout.num_cts), k_hi(layout.num_cts);
  {
    auto half_batch = [&](int call) {
      RealBatch h(layout.lanes,
                  std::vector<std::vector<double>>(
                      layout.dim, std::vector<double>(layout.dim, 0.0)));
      for (int t = 0; t < layout.lanes; t++) {
        for (int x = 0; x < half; x++) {
          for (int j = 0; j < layout.dim; j++) {
            h[t][x][j] = kk[t][j][call * half + x];
          }
        }
      }
      return h;
    };
    const RealBatch b1 = half_batch(0), b2 = half_batch(1);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      encrypt_slots(b1, bi, k_lo[bi]);
      encrypt_slots(b2, bi, k_hi[bi]);
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      auto ang = [&](int row, int cls) {
        return static_cast<double>(bi * layout.rank + cls) *
               theta[row % half];
      };
      mask(bi, [&](int r, int c) { return std::cos(ang(r, c)); }, cos_pt);
      mask(bi, [&](int r, int c) { return std::sin(ang(r, c)); }, sin_pt);
      mask(bi, [&](int r, int c) { return -std::sin(ang(r, c)); },
           neg_sin_pt);
      rope_pair(k_lo[bi], k_hi[bi], cos_pt, sin_pt, neg_sin_pt);
    }
  }

  // Descend everything and run the two calls.
  auto convert = [&](std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      ASSERT_EQ(big.param->NPToLevel(ct.GetNP()), conv_level);
      Ciphertext<word> sinc;
      conv.SlotToSinC(big.context, sinc, ct, big.ui->GetEvkMap());
      ct = std::move(sinc);
    }
  };
  convert(q_cts);
  convert(k_lo);
  convert(k_hi);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Ciphertext<word>> res;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_cts[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> &rhs = (call == 0) ? k_lo : k_hi;
    std::vector<Ciphertext<word>> part;
    handler.Multiply(part, lhs, rhs, big.ui->GetRingSwitchKey(layout.rank),
                     big.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        big.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  // The part-level read, against the RoPE'd host product; the un-RoPE'd
  // product is the control that has to fail.
  double worst = 0.0, transposed = 0.0, norope = 0.0;
  const double bridge_scale = small.param->GetScale(chain_level);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Plaintext<word> pt;
    big.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    big.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(bridge, chain_level, bridge_scale,
                                          comp[j]);
      std::vector<Complex> got;
      small.context->encoder_.DecodeSinC(got, bridge, sub_degree);
      const int column = bi * layout.rank + j;
      for (int row = 0; row < layout.dim; row++) {
        for (int lane = 0; lane < layout.lanes; lane++) {
          double want = 0.0, want_t = 0.0, want_raw = 0.0;
          for (int c = 0; c < layout.dim; c++) {
            want += q_ref[lane][row][c] * k_ref[lane][column][c];
            want_t += q_ref[lane][column][c] * k_ref[lane][row][c];
            want_raw += q[lane][row][c] * kk[lane][column][c];
          }
          int part_idx, index;
          layout.LocatePart(row, column, lane, part_idx, index);
          const double g = got[index].real();
          worst = std::max(worst, std::abs(g - want));
          transposed = std::max(transposed, std::abs(g - want_t));
          norope = std::max(norope, std::abs(g - want_raw));
        }
      }
    }
  }

  std::cout << "RoPE'd scores at the Llama alignment: products " << worst
            << std::endl;
  std::cout << "  controls: transposed " << transposed << ", un-RoPE'd "
            << norope << std::endl;

  EXPECT_LT(worst, 5e-2)
      << "RoPE across the ct pairs did not land the rotated scores";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(norope, 3e-2)
      << "the RoPE'd and raw products agree, so the rotation never "
         "happened";
}

// ---------------------------------------------------------------------------
// The real bootstrap set (Doing.md 1.5bt).
//
// 1.5bs closed with "these want the real bootstrap set -- a ci16 with the
// switch pair's bottom primes -- not the correctness trio". Backwards: the
// boot parameter already owns a chain-shaped bottom. ci16_35's levels
//
//   L0 (0,2) {29884417, 37224449}   q0 = 2^49.98
//   L1 (2,1) {m0, m1, t0}           rescale m0 m1 / t1 = 2^35.008
//   L2 (4,0) {m0 .. m3}             rescale m2 m3 / t0 = 2^35.002
//   L3 (1,5) {m0, t0 .. t4}         rescale 2^35.149 (the graft exchange)
//   L4 (3,4) {m0..m2, t0..t3}       rescale m1 m2 / t4 = 2^34.91
//
// are a valid switch-trio ladder in themselves, so the trio COMES TO THE
// BOOT PARAMETER: ci_ringswitch16_35_boot / ci12_35_boot /
// ringdegree13_35_boot carry those five configs and their primes verbatim
// -- the first trio with MAIN primes -- plus two synthetic top levels
// (3,5) and (4,5) that only satisfy Parameter's all-primes-at-the-last-
// level rule (split in two because a rescale may drop mains or terminals,
// not both) and hold no ciphertext. A ciphertext of ci16_35 at level <= 4
// IS a ciphertext of
// the switching Context, limb for limb: no transport, no new bootstrap
// parameter, and ci16_35's measured bootstrap is untouched. The switching
// Context holds the block's secret (RingFixture's adopted-secret
// constructor, the [SYLPH] ladder pattern).
//
// WHY THE CONVERSIONS RUN ON THE TRIO AND NOT ON ci16_35. On ci16_35's
// twelve-prime aux basis (alpha 12, max_num_ter 5), every level with
// num_main + 5 <= 12 -- levels 0..6 -- sits in 1.5x's num_accum == 1
// hoisted-accumulation zone, and a hoisted LinearTransform there returns
// uniform mod-Q noise. Measured here, and pinned below as a regression:
// the same converter that is exact on the trio is mod-Q garbage at
// ci16_35's level 3. The narrow trio has no zone at all (alpha 2 <
// max_num_ter), which lands the leg exactly where [SYLPH] wanted its
// switch keys anyway. Plain ops (RoPE's plaintext multiplies, Rescale,
// encrypt/decrypt) are zone-free and stay on the boot Context.
//
// The two tests walk the layer's own bottom,
//
//   L4 --RoPE--> L3 --SlotToSinC--> L2 --chain--> L1 --SinCToSlot--> L0
//
// with every rung canonical (2^35 within millibits, against the
// correctness trios' 2^25-ish rungs) -- the first time the scores pipeline
// runs at the ladder the layer will actually stand on. The third test
// (1.5bu) then hands the level-0 output to ci16_35's own Boot and reads
// the scores back at dec level. NOT here: any security claim for the trio
// (Q * P = 2^182+ at degree 4096: correctness-lane, exactly like the
// _l2/_l3 trios), and any bootstrap performance claim (Boot's timing is
// measured by Bootstrapping.cpp, warm; here it runs cold, once).
// ---------------------------------------------------------------------------

constexpr const char *kBootParam = "ci16_35.json";
constexpr const char *kBootSwitchParam = "ci_ringswitch16_35_boot.json";
constexpr const char *kBootSmallParam = "ci12_35_boot.json";
constexpr const char *kBootLiftedParam = "ringdegree13_35_boot.json";

TEST(CiBootSet, TheLoopRunsOnTheRealBootstrapLadder) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int fwd_level = 3;      // (1,5), on ci16_35 and on the trio alike
  const int chain_level = 2;    // (4,0)
  const int inverse_level = 1;  // (2,1)
  ASSERT_EQ(swtch.param->max_level_, 6) << "L0..L4 plus the synthetic tops";

  // The matched-set precondition, in numbers: the rescale ladders agree at
  // every shared level because the primes do. Editing either file alone
  // breaks this first.
  for (int l = 1; l <= 4; l++) {
    ASSERT_NEAR(boot.param->GetRescalePrimeProd(l) /
                    swtch.param->GetRescalePrimeProd(l),
                1.0, 1e-12)
        << "ci16_35 and the _boot trio disagree at level " << l;
  }

  const int sub_degree = 128;
  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.num_cts, 2);

  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(swtch.context, sub_degree,
                             /*forward_level=*/fwd_level,
                             /*inverse_level=*/inverse_level, &layout);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const double s = boot.param->GetScale(fwd_level);

  // Zone-free plain ops on the boot Context: the ct x pt ladder at the
  // very levels whose HOISTED transforms die below.
  {
    std::mt19937_64 gen(0xB009);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int level = fwd_level; level >= 1; level--) {
      std::vector<Complex> av(boot.Degree()), bv(boot.Degree());
      for (int i = 0; i < boot.Degree(); i++) {
        av[i] = Complex(dist(gen), 0.0);
        bv[i] = Complex(dist(gen), 0.0);
      }
      Plaintext<word> pa, pb;
      boot.context->encoder_.Encode(pa, level, boot.param->GetScale(level),
                                    av);
      boot.context->encoder_.Encode(
          pb, level, boot.param->GetRescalePrimeProd(level), bv);
      Ciphertext<word> ct, prod, resc;
      boot.ui->Encrypt(ct, pa);
      boot.context->Mult(prod, ct, pb);
      boot.context->Rescale(resc, prod);
      Plaintext<word> back;
      boot.ui->Decrypt(back, resc);
      std::vector<Complex> got;
      boot.context->encoder_.Decode(got, back);
      double err = 0.0;
      for (int i = 0; i < boot.Degree(); i++) {
        err = std::max(
            err, std::abs(got[i].real() - av[i].real() * bv[i].real()));
      }
      EXPECT_LT(err, 1e-2)
          << "plain ct x pt + rescale broke at ci16_35 level " << level
          << " -- the graft exchange itself, not the hoist zone";
    }
  }

  // The zone, pinned as a regression: the SAME conversion, hoisted on
  // ci16_35's alpha-12 basis at level 3 (num_main 1 + max_num_ter 5 <=
  // alpha 12, 1.5x's num_accum == 1 criterion), returns mod-Q noise. If
  // this assertion ever fails -- the boot-side conversion coming back
  // clean -- the underlying Hoist defect got fixed and the leg's
  // conversions may move onto the boot Context; revisit 1.5bt.
  {
    CiSinCConverter<word> zone(boot.context, sub_degree,
                               /*forward_level=*/fwd_level,
                               /*inverse_level=*/-1, &layout);
    EvkRequest zone_req;
    zone.AddRequiredRotations(zone_req);
    boot.ui->PrepareRotationKey(zone_req);
    std::vector<Complex> msg(boot.Degree(), Complex(0.0, 0.0));
    for (int i = 0; i < boot.Degree(); i++) {
      msg[i] = Complex(((i * 2654435761ULL) % 1000) / 1000.0 - 0.5, 0.0);
    }
    Plaintext<word> pt;
    boot.context->encoder_.Encode(pt, fwd_level, s, msg);
    Ciphertext<word> enc, sinc;
    boot.ui->Encrypt(enc, pt);
    zone.SlotToSinC(boot.context, sinc, enc, boot.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    Plaintext<word> out;
    boot.ui->Decrypt(out, sinc);
    std::vector<double> coeffs;
    boot.context->encoder_.DecodeCoeff(coeffs, out);
    double magnitude = 0.0;
    for (double c : coeffs) magnitude = std::max(magnitude, std::abs(c));
    std::cout << "  the hoist zone on ci16_35 at level 3: |output| "
              << magnitude << " (mod-Q noise; clean would be O(1))"
              << std::endl;
    EXPECT_GT(magnitude, 1e3)
        << "the hoisted conversion on ci16_35's alpha-12 basis came back "
           "clean -- the 1.5x zone moved; revisit where the conversions "
           "live";
  }

  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, 0xB007);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  0xB008);

  // Primary addresses by assignment; the nested forward owns the copy-add.
  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<Complex> slot_msg(boot.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            slot_msg[slot] = Complex(m[lane][row][column], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      boot.context->encoder_.Encode(pt, fwd_level, s, slot_msg);
      Ciphertext<word> enc;
      boot.ui->Encrypt(enc, pt);
      conv.SlotToSinC(swtch.context, out[bi], enc, swtch.ui->GetEvkMap());
      ASSERT_EQ(boot.param->NPToLevel(out[bi].GetNP()), chain_level);
      ASSERT_EQ(swtch.param->NPToLevel(out[bi].GetNP()), chain_level);
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // The crossing, measured as an identity: the same ciphertext decrypts to
  // the same coefficients through either Context.
  double crossing = 0.0;
  {
    Plaintext<word> pb, ps;
    boot.ui->Decrypt(pb, rhs[0]);
    swtch.ui->Decrypt(ps, rhs[0]);
    std::vector<double> cb, cs;
    boot.context->encoder_.DecodeCoeff(cb, pb);
    swtch.context->encoder_.DecodeCoeff(cs, ps);
    for (int t = 0; t < boot.Degree(); t++) {
      crossing = std::max(crossing, std::abs(cb[t] - cs[t]));
    }
  }

  // The folded forward's coefficient probe, against the host encode of the
  // two-address summed message -- on the boot ladder this time.
  double fwd_coeff_err = 0.0;
  {
    std::vector<Complex> fmsg(boot.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int i = 0; i < layout.rank; i++) {
        for (int r = 0; r < layout.lanes; r++) {
          double want = b[r][row][i];
          if (i != 0 && row + 1 < layout.dim) {
            want += b[r][row + 1][layout.rank - i];
          }
          fmsg[(static_cast<size_t>(row) * layout.rank + i) * layout.lanes +
               r] = Complex(want, 0.0);
        }
      }
    }
    Plaintext<word> probe;
    boot.ui->Decrypt(probe, rhs[0]);
    std::vector<double> got_c, want_c;
    boot.context->encoder_.DecodeCoeff(got_c, probe);
    Plaintext<word> host_pt;
    boot.context->encoder_.EncodeSinC(host_pt, chain_level, s, fmsg,
                                      sub_degree);
    boot.context->encoder_.DecodeCoeff(want_c, host_pt);
    for (int t = 0; t < boot.Degree(); t++) {
      fwd_coeff_err =
          std::max(fwd_coeff_err, std::abs(got_c[t] - want_c[t]));
    }
  }

  handler.Multiply(res, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                   swtch.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // On the canonical ladder the 1.5bm factor-of-two law holds in its exact
  // form: 2 s^2 over the chain level's rescale product.
  const double product_scale =
      2.0 * s * s / boot.param->GetRescalePrimeProd(chain_level);
  for (const auto &ct : res) {
    ASSERT_EQ(boot.param->NPToLevel(ct.GetNP()), inverse_level);
    EXPECT_NEAR(ct.GetScale() / product_scale, 1.0, 1e-6)
        << "the chain's scale law does not close on the boot ladder";
  }

  // Back to slots at level 0 through the trio, decrypted through the boot
  // Context -- the crossing in the other direction.
  double worst_loop = 0.0, transposed = 0.0, unsummed = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, res[bi], swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(back.GetNP()), inverse_level - 1);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, back);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        const int column = bi * layout.rank + j;
        for (int lane = 0; lane < layout.lanes; lane++) {
          double want = 0.0, want_t = 0.0, want_sum = 0.0;
          for (int x = 0; x < layout.contraction; x++) {
            want += a[lane][row][x] * b[lane][x][column];
            want_t += a[lane][column][x] * b[lane][x][row];
          }
          const int cls = column % layout.rank;
          if (cls != 0 && row + 1 < layout.dim) {
            const int partner =
                bi * layout.rank + (layout.rank - cls);
            for (int x = 0; x < layout.contraction; x++) {
              want_sum += a[lane][row + 1][x] * b[lane][x][partner];
            }
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          worst_loop = std::max(worst_loop, std::abs(got - want));
          transposed = std::max(transposed, std::abs(got - want_t));
          unsummed = std::max(unsummed, std::abs(want_sum));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "the loop on the real bootstrap ladder (ci16_35 L3..L0), "
            << "sub_degree " << sub_degree << ": crossing " << crossing
            << ", forward coeff " << fwd_coeff_err << ", loop products "
            << worst_loop << std::endl;
  std::cout << "  controls: transposed " << transposed
            << ", largest partner product the scan removed " << unsummed
            << std::endl;

  EXPECT_LT(crossing, 1e-9)
      << "a level-2 ciphertext of ci16_35 is not a ciphertext of the "
         "switching Context -- the limb layouts diverged";
  EXPECT_LT(fwd_coeff_err, 2e-3);
  EXPECT_LT(worst_loop, 5e-2)
      << "the loop did not return the products to their primary slots on "
         "the boot ladder";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(unsummed, 1e-2);
}

// The Llama scores pipeline on the same rungs: RoPE at L4 (rotation-free,
// 1.5bs), the sub-32 nested descent at L3, the two-call contraction sum
// through the chain at L2, and -- what the _l3 trio never had the levels
// for -- the RoPE'd scores RETURNED TO SLOTS at L0, primary addresses,
// ready for SoftMax's four slot-field rotations. Four levels of the
// nineteen below dec.
TEST(CiBootSet, TheRopedScoresReturnToSlotsOnTheRealLadder) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int rope_level = 4;     // (3,4): RoPE's plaintext-multiply level
  const int conv_level = 3;     // the nested descent, on the trio
  const int chain_level = 2;    // (4,0)
  const int inverse_level = 1;  // the return to slots
  const int sub_degree = 32;
  ASSERT_EQ(swtch.param->max_level_, 6);

  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  const int half = layout.contraction;  // 64

  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> conv(swtch.context, sub_degree,
                             /*forward_level=*/conv_level,
                             /*inverse_level=*/inverse_level, &layout);
  const auto t1 = std::chrono::steady_clock::now();
  EvkRequest req;
  conv.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const RealBatch q = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0xB0E1);
  const RealBatch kk = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                   layout.dim, 0.08, 0xB0E2);
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  const double pt_scale = boot.param->GetRescalePrimeProd(rope_level);
  auto encrypt_slots = [&](const RealBatch &m, int bi, Ciphertext<word> &out) {
    std::vector<Complex> slot_msg(boot.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        for (int lane = 0; lane < layout.lanes; lane++) {
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, bi * layout.rank + j, lane, ct_idx, slot,
                            copy_slot);
          slot_msg[slot] = Complex(m[lane][row][bi * layout.rank + j], 0.0);
        }
      }
    }
    Plaintext<word> pt;
    boot.context->encoder_.Encode(pt, rope_level,
                                  boot.param->GetScale(rope_level), slot_msg);
    boot.ui->Encrypt(out, pt);
  };
  auto mask = [&](int bi, auto value, Plaintext<word> &pt) {
    std::vector<Complex> msg(boot.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        for (int lane = 0; lane < layout.lanes; lane++) {
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, bi * layout.rank + j, lane, ct_idx, slot,
                            copy_slot);
          msg[slot] = Complex(value(row, j), 0.0);
        }
      }
    }
    boot.context->encoder_.Encode(pt, rope_level, pt_scale, msg);
  };
  auto rope_pair = [&](Ciphertext<word> &lo, Ciphertext<word> &hi,
                       const Plaintext<word> &cos_pt,
                       const Plaintext<word> &sin_pt,
                       const Plaintext<word> &neg_sin_pt) {
    Ciphertext<word> a, b;
    boot.context->Mult(a, lo, cos_pt);
    boot.context->Mult(b, hi, neg_sin_pt);
    boot.context->Add(a, a, b);
    boot.context->Mult(b, hi, cos_pt);
    Ciphertext<word> d;
    boot.context->Mult(d, lo, sin_pt);
    boot.context->Add(b, b, d);
    boot.context->Rescale(lo, a);
    boot.context->Rescale(hi, b);
  };

  std::vector<Ciphertext<word>> q_cts(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) encrypt_slots(q, bi, q_cts[bi]);
  for (int k = 0; k < layout.num_cts / 2; k++) {
    Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
    auto ang = [&](int row, int cls) {
      return static_cast<double>(row) * theta[k * layout.rank + cls];
    };
    mask(k, [&](int r, int c) { return std::cos(ang(r, c)); }, cos_pt);
    mask(k, [&](int r, int c) { return std::sin(ang(r, c)); }, sin_pt);
    mask(k, [&](int r, int c) { return -std::sin(ang(r, c)); }, neg_sin_pt);
    rope_pair(q_cts[k], q_cts[k + layout.num_cts / 2], cos_pt, sin_pt,
              neg_sin_pt);
  }

  std::vector<Ciphertext<word>> k_lo(layout.num_cts), k_hi(layout.num_cts);
  {
    auto half_batch = [&](int call) {
      RealBatch h(layout.lanes,
                  std::vector<std::vector<double>>(
                      layout.dim, std::vector<double>(layout.dim, 0.0)));
      for (int t = 0; t < layout.lanes; t++) {
        for (int x = 0; x < half; x++) {
          for (int j = 0; j < layout.dim; j++) {
            h[t][x][j] = kk[t][j][call * half + x];
          }
        }
      }
      return h;
    };
    const RealBatch b1 = half_batch(0), b2 = half_batch(1);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      encrypt_slots(b1, bi, k_lo[bi]);
      encrypt_slots(b2, bi, k_hi[bi]);
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      auto ang = [&](int row, int cls) {
        return static_cast<double>(bi * layout.rank + cls) *
               theta[row % half];
      };
      mask(bi, [&](int r, int c) { return std::cos(ang(r, c)); }, cos_pt);
      mask(bi, [&](int r, int c) { return std::sin(ang(r, c)); }, sin_pt);
      mask(bi, [&](int r, int c) { return -std::sin(ang(r, c)); },
           neg_sin_pt);
      rope_pair(k_lo[bi], k_hi[bi], cos_pt, sin_pt, neg_sin_pt);
    }
  }

  double fwd_seconds = 0.0;
  int fwd_count = 0;
  auto convert = [&](std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      ASSERT_EQ(boot.param->NPToLevel(ct.GetNP()), conv_level);
      Ciphertext<word> sinc;
      const auto f0 = std::chrono::steady_clock::now();
      conv.SlotToSinC(swtch.context, sinc, ct, swtch.ui->GetEvkMap());
      cudaDeviceSynchronize();
      fwd_seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - f0)
                         .count();
      fwd_count++;
      ct = std::move(sinc);
    }
  };
  convert(q_cts);
  convert(k_lo);
  convert(k_hi);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Ciphertext<word>> res;
  double mult_seconds = 0.0;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_cts[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> &rhs = (call == 0) ? k_lo : k_hi;
    std::vector<Ciphertext<word>> part;
    const auto m0 = std::chrono::steady_clock::now();
    handler.Multiply(part, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    mult_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - m0)
            .count();
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  // The return to slots the _l3 trio never had the levels for.
  double worst = 0.0, transposed = 0.0, norope = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, res[bi], swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(back.GetNP()), inverse_level - 1);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, back);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        const int column = bi * layout.rank + j;
        for (int lane = 0; lane < layout.lanes; lane++) {
          double want = 0.0, want_t = 0.0, want_raw = 0.0;
          for (int c = 0; c < layout.dim; c++) {
            want += q_ref[lane][row][c] * k_ref[lane][column][c];
            want_t += q_ref[lane][column][c] * k_ref[lane][row][c];
            want_raw += q[lane][row][c] * kk[lane][column][c];
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          worst = std::max(worst, std::abs(got - want));
          transposed = std::max(transposed, std::abs(got - want_t));
          norope = std::max(norope, std::abs(got - want_raw));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "RoPE'd scores, returned to slots on the real ladder "
            << "(L4 RoPE, L3 descent, L2 chain x2, L1 return): products "
            << worst << std::endl;
  std::cout << "  controls: transposed " << transposed << ", un-RoPE'd "
            << norope << std::endl;
  std::cout << "  cost at the boot levels: converter build "
            << std::chrono::duration<double>(t1 - t0).count()
            << " s, forward " << fwd_seconds / std::max(fwd_count, 1)
            << " s/ct over " << fwd_count << " cts, one chain call "
            << mult_seconds / 2.0 << " s" << std::endl;

  EXPECT_LT(worst, 5e-2)
      << "the scores did not survive the full L4..L0 walk";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(norope, 3e-2);
}

// The bootstrap consumes the level-0 scores (Doing.md 1.5bu).
//
// 1.5bt left the scores in slots at level 0 -- SoftMax's slot addresses,
// but the wrong level: nothing evaluates at level 0, and the only way out
// is ci16_35's own bootstrap. This test closes that edge: the sub-128
// loop's products, landed at their primary addresses at level 0, go
// through Boot -- the measured CI bootstrap, keys and all, prepared
// exactly as Bootstrapping.cpp prepares it -- and come out still at their
// primary addresses, against the same host products, with the transposed
// read as the control. Boot lands at GetEndLevel() = dec - num_stc = 16,
// not at dec 19: EvalMod ends at 19 and full Boot's own StC spends three
// more below it. Landing AT dec is HalfBoot's property, which is one more
// reason the fused route below is the eventual one. The bootstrap's transforms run at
// levels 19..31, far above the level-7 hoist-zone ceiling that forces the
// leg's own conversions onto the trio.
//
// THE SCALE CONTRACT AT THE BOOT BOUNDARY. Boot never reads the input's
// declared scale: it multiplies the DATA by 2^log_scaleup and assumes the
// data sits at base_scale with the message in (-1, 1) (BootContext's
// constructor). The chain's output arrives at 2 s^2 / q_rescale(chain) =
// carried * base_scale -- carried ~ 2^1.15 on these rungs, the 1.5bm
// factor of two times the rungs' millibit drift -- so what Boot returns
// is carried * m at the canonical dec scale. The factor rides through
// EvalMod inside the message (|carried * m| stays far inside (-1, 1)) and
// is divided out here; the layer will fold it into SoftMax's first
// plaintext multiply instead.
//
// WHAT THIS DOES NOT BUY: the fused route. The ordinary leg feeds
// HalfBoot with the product's SinC coefficients directly and finishes
// with the StC prefix at dec level (SinCAttention), spending no bottom
// level on the return conversion and no StC inside the bootstrap. The CI
// analogue needs the prefix's nested form -- the composed map from what
// HalfBoot leaves in slots to the layout's primary addresses -- which
// does not exist yet. Until it does, the leg pays inverse_level for
// SinCToSlot and Boot pays its StC; this test is the contract that route
// has to beat.
TEST(CiBootSet, TheBootstrapConsumesTheLevelZeroScores) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr)
      << "ci16_35 did not come up as a BootContext -- RingFixture's boot "
         "branch is compiled out";

  const int fwd_level = 3;      // (1,5)
  const int chain_level = 2;    // (4,0)
  const int inverse_level = 1;  // (2,1)
  const int sub_degree = 128;
  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.num_cts, 2);

  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(swtch.context, sub_degree,
                             /*forward_level=*/fwd_level,
                             /*inverse_level=*/inverse_level, &layout);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  // The bootstrap, prepared on the boot Context. boot.ui's constructor
  // already made the multiplication, conjugation and sparse-encapsulation
  // keys; only the CtS/StC rotations are added here.
  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  EvkRequest boot_req;
  bctx->AddRequiredRotations(boot_req, num_slots);
  boot.ui->PrepareRotationKey(boot_req);

  const double s = boot.param->GetScale(fwd_level);
  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, 0xB010);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  0xB011);

  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<Complex> slot_msg(boot.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            slot_msg[slot] = Complex(m[lane][row][column], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      boot.context->encoder_.Encode(pt, fwd_level, s, slot_msg);
      Ciphertext<word> enc;
      boot.ui->Encrypt(enc, pt);
      conv.SlotToSinC(swtch.context, out[bi], enc, swtch.ui->GetEvkMap());
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);
  handler.Multiply(res, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                   swtch.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // What arrives at the boundary: the chain's exact output scale, as a
  // multiple of the base scale Boot assumes.
  const double carried_want =
      2.0 * s * s /
      (boot.param->GetRescalePrimeProd(chain_level) *
       boot.param->base_scale_);

  double pre_worst = 0.0, post_worst = 0.0, transposed = 0.0;
  double carried_seen = 0.0, boot_seconds = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, res[bi], swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(back.GetNP()), 0);

    const double carried = back.GetScale() / boot.param->base_scale_;
    carried_seen = carried;

    // The pre-boot read, so the bootstrap's own contribution is visible
    // as the difference of the two numbers printed below.
    std::vector<Complex> before;
    {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, back);
      boot.context->encoder_.Decode(before, pt);
    }

    back.SetNumSlots(num_slots);
    Ciphertext<word> dec;
    const auto b0 = std::chrono::steady_clock::now();
    bctx->Boot(dec, back, boot.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    boot_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - b0)
            .count();
    // Full Boot lands at GetEndLevel() = dec - num_stc (16 here): EvalMod
    // ends AT dec 19 -- that is HalfBoot's landing, the one the schedule
    // calls dec -- and Boot's own StC spends its three levels below it.
    ASSERT_EQ(boot.param->NPToLevel(dec.GetNP()),
              bctx->GetBootParameter().GetEndLevel())
        << "the bootstrap did not land where its parameter says";

    std::vector<Complex> after;
    {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, dec);
      boot.context->encoder_.Decode(after, pt);
    }

    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        const int column = bi * layout.rank + j;
        for (int lane = 0; lane < layout.lanes; lane++) {
          double want = 0.0, want_t = 0.0;
          for (int x = 0; x < layout.contraction; x++) {
            want += a[lane][row][x] * b[lane][x][column];
            want_t += a[lane][column][x] * b[lane][x][row];
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          pre_worst =
              std::max(pre_worst, std::abs(before[slot].real() - want));
          const double got = after[slot].real() / carried;
          post_worst = std::max(post_worst, std::abs(got - want));
          transposed = std::max(transposed, std::abs(got - want_t));
        }
      }
    }
  }

  std::cout << "the bootstrap consumes the level-0 scores (sub_degree "
            << sub_degree << "): products before Boot " << pre_worst
            << ", after Boot at level "
            << bctx->GetBootParameter().GetEndLevel() << " " << post_worst
            << std::endl;
  std::cout << "  carried scale at the boundary " << carried_seen
            << " x base (want " << carried_want << "), Boot "
            << boot_seconds / layout.num_cts
            << " s/ct cold (correctness lane, not a performance claim)"
            << std::endl;
  std::cout << "  control: transposed " << transposed << std::endl;

  EXPECT_NEAR(carried_seen / carried_want, 1.0, 1e-6)
      << "the chain's 2 s^2 / q_rescale law did not arrive at the boot "
         "boundary intact";
  EXPECT_LT(post_worst, 1e-2)
      << "the bootstrap did not return the scores to their primary slots";
  EXPECT_GT(transposed, 3e-2);
}

// SoftMax normalizes the booted scores (Doing.md 1.5bv).
//
// 1.5bs settled the placement without running the arithmetic; this runs it,
// on the layout's own slot fields, at the levels the layer will have.
// [SYLPH] section 2.3 / Cho et al., k = 1: y = exp, then one Euclidean
// normalisation (y / ||y||_2)^2 whose square already sums to one -- exactly
// SoftMaxHandler's algorithm, but that handler is compiled to the ordinary
// ring (num_slots = degree / 2) and to a key axis on the top slot field.
//
// THE BIT REVERSAL PUTS THE CLS FIELD ON TOP, AND THE REDUCTION IS FREE.
// LocateSlot's primary address is block BitRev(row * rank + cls), lane in
// the low bits -- so cls, the LOW four bits of the block index, lands on
// the TOP four bits of the slot index, at stride num_slots / rank. The
// rotate-and-add tree over a top field wraps modulo the field exactly
// (the carry falls off the vector's end), so
//
//   1 ct add  (the ciphertext half of the key axis)
//   4 rotate-adds by (num_slots / rank) * 2^t
//
// leaves the full row norm at EVERY slot: no mask, no mirror broadcast,
// every slot exact, the same shape as the handler's own convention with
// the keys merely visited in bit-reversed order -- which a sum cannot
// see. 1.5bs's "broadcast the mirror" turns out to be unnecessary.
//
// SHARPNESS LIVES IN THE POLYNOMIAL, NOT THE CIPHERTEXT. The bootstrap
// carries u = 2 (S - c) / M + 1 in [-1, 1]; the effective score span
// M_eff is baked into exp's Chebyshev fit (exp(M_eff (u - 1) / 4) for
// k = 1), exactly as the layer folds 1 / sqrt(d_head) and the calibrated
// shift upstream of HalfBoot. So a chain whose raw products span ~0.3 can
// still exercise a sharp softmax -- M_eff = 8 here -- and the affine's
// plaintext multiply is also where 1.5bu's carried factor divides out, as
// promised there. NOT here: the causal mask (one more plaintext multiply,
// mechanically identical to the merged mask above; which norm interval it
// leaves per row is a calibration question for real data), and any claim
// about calibrated ranges -- lo/hi are read off this test's own data, as
// [SYLPH] reads them off the real layer's.
TEST(CiBootSet, SoftMaxNormalizesTheBootedScores) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int fwd_level = 3, chain_level = 2, inverse_level = 1;
  const int sub_degree = 128;
  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 32);
  ASSERT_EQ(layout.rank, 16);
  ASSERT_EQ(layout.lanes, 128);
  ASSERT_EQ(layout.num_cts, 2);

  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(swtch.context, sub_degree, fwd_level,
                             inverse_level, &layout);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  EvkRequest boot_req;
  bctx->AddRequiredRotations(boot_req, num_slots);
  boot.ui->PrepareRotationKey(boot_req);

  // ---- the scores, through the measured pipeline to level 16 ------------
  const double s = boot.param->GetScale(fwd_level);
  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, 0xB020);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  0xB021);

  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<Complex> slot_msg(boot.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            slot_msg[slot] = Complex(m[lane][row][column], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      boot.context->encoder_.Encode(pt, fwd_level, s, slot_msg);
      Ciphertext<word> enc;
      boot.ui->Encrypt(enc, pt);
      conv.SlotToSinC(swtch.context, out[bi], enc, swtch.ui->GetEvkMap());
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);
  handler.Multiply(res, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                   swtch.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());

  double carried = 0.0;
  std::vector<Ciphertext<word>> scores(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, res[bi], swtch.ui->GetEvkMap());
    carried = back.GetScale() / boot.param->base_scale_;
    back.SetNumSlots(num_slots);
    bctx->Boot(scores[bi], back, boot.ui->GetEvkMap());
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int top = boot.param->NPToLevel(scores[0].GetNP());
  ASSERT_EQ(top, bctx->GetBootParameter().GetEndLevel());

  // ---- host calibration, public exactly as the layer's is ---------------
  // S[lane][row][col], the true products; c and M off the data; u in [-1,1].
  const double m_eff = 8.0;
  std::vector<std::vector<std::vector<double>>> S(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
  double smin = 1e300, smax = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int col = 0; col < layout.dim; col++) {
        double v = 0.0;
        for (int x = 0; x < layout.contraction; x++) {
          v += a[lane][row][x] * b[lane][x][col];
        }
        S[lane][row][col] = v;
        smin = std::min(smin, v);
        smax = std::max(smax, v);
      }
    }
  }
  const double span = smax - smin;
  auto u_of = [&](double v) { return 2.0 * (v - smax) / span + 1.0; };

  // ---- the polynomials, SoftMax.cu's own construction -------------------
  const double half = m_eff / 4.0;  // k = 1: exp on [-M_eff / 2, 0]
  auto exp_coeffs = cheddar::chebfit::Interpolate(
      [half](double v) { return std::exp(half * (v - 1.0)); }, 9);
  const int exp_in = top - 1;
  auto log2ceil = [](int n) {
    int r = 0;
    while ((1 << r) < n) r++;
    return r;
  };
  const int exp_used =
      EvalPoly<word>(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                     boot.param->GetScale(exp_in), true)
          .GetPolyDegree();
  const int exp_out = exp_in - log2ceil(exp_used + 1);
  EvalPoly<word> exp_poly(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                          boot.param->GetScale(exp_out), true);
  exp_poly.Compile(boot.context);

  // The norm interval, read off this data with a margin -- rows are
  // (row, lane) pairs, keys are the column axis across both ciphertexts.
  double lo = 1e300, hi = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      double sq = 0.0;
      for (int col = 0; col < layout.dim; col++) {
        const double y = std::exp(half * (u_of(S[lane][row][col]) - 1.0));
        sq += y * y;
      }
      lo = std::min(lo, sq);
      hi = std::max(hi, sq);
    }
  }
  lo *= 0.9;
  hi *= 1.1;
  const double aff_a = 0.5 * (hi - lo);
  const double aff_b = 0.5 * (hi + lo);
  auto inv_coeffs = cheddar::chebfit::Interpolate(
      [aff_a, aff_b](double v) { return 1.0 / std::sqrt(aff_a * v + aff_b); },
      15);
  const int sq_level = exp_out - 1;    // HMult(y, y)
  const int poly_in = sq_level - 1;    // the merged mask / affine multiply
  const int inv_used =
      EvalPoly<word>(inv_coeffs, poly_in, boot.param->GetScale(poly_in),
                     boot.param->GetScale(poly_in), true)
          .GetPolyDegree();
  const int inv_out = poly_in - log2ceil(inv_used + 1);
  ASSERT_GE(inv_out - 2, 0) << "the softmax does not fit below the boot";
  EvalPoly<word> inv_poly(inv_coeffs, poly_in,
                          boot.param->GetScale(poly_in),
                          boot.param->GetScale(inv_out), true);
  inv_poly.Compile(boot.context);

  // ---- rotation keys for the reduce tree --------------------------------
  // cls sits on the TOP four slot bits (see the header comment), so the
  // stride is num_slots / rank and the tree is exact at every slot.
  std::vector<int> reduce_dist;
  for (int t = 0; t < 4; t++) {
    reduce_dist.push_back((num_slots / layout.rank) << t);
  }
  {
    EvkRequest rot_req;
    for (int d : reduce_dist) rot_req.AddRequest(d, sq_level);
    boot.ui->PrepareRotationKey(rot_req);
  }
  const auto &evk = boot.ui->GetEvkMap();
  const auto &mult_key = evk.GetMultiplicationKey();

  // ---- the circuit ------------------------------------------------------
  // Affine at `top`: u = a1 * (carried * S) + a0, carried dividing out in
  // a1 exactly as 1.5bu promised.
  const double a1 = 2.0 / (span * carried);
  const double a0 = 1.0 - 2.0 * smax / span;
  std::vector<Ciphertext<word>> y(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Constant<word> c1;
    boot.context->encoder_.EncodeConstant(c1, top,
                                          boot.param->GetScale(top), a1);
    Ciphertext<word> t1, u_ct;
    boot.context->Mult(t1, scores[bi], c1);
    boot.context->Rescale(u_ct, t1);
    Constant<word> c0;
    boot.context->encoder_.EncodeConstant(c0, exp_in, u_ct.GetScale(), a0);
    boot.context->Add(u_ct, u_ct, c0);
    // y = exp(M_eff (u - 1) / 4)
    exp_poly.Evaluate(boot.context, y[bi], u_ct, mult_key);
  }

  // The Euclidean norm over the split key axis: one ct add, then the
  // top-field reduce tree -- exact at every slot, no mask, no broadcast.
  Ciphertext<word> sq, term, rotated;
  boot.context->HMult(sq, y[0], y[0], mult_key);
  boot.context->HMult(term, y[1], y[1], mult_key);
  boot.context->Add(sq, sq, term);
  for (int d : reduce_dist) {
    boot.context->HRotAdd(rotated, sq, sq, evk.GetRotationKey(d), d);
    boot.context->Copy(sq, rotated);
  }
  // The affine map onto the polynomial's domain, SoftMax.cu's own pattern.
  {
    Constant<word> inv_a;
    boot.context->encoder_.EncodeConstant(
        inv_a, sq_level, boot.param->GetScale(sq_level), 1.0 / aff_a);
    Ciphertext<word> scaled;
    boot.context->Mult(scaled, sq, inv_a);
    boot.context->Rescale(sq, scaled);
    Constant<word> shift;
    boot.context->encoder_.EncodeConstant(shift, poly_in, sq.GetScale(),
                                          -aff_b / aff_a);
    boot.context->Add(sq, sq, shift);
  }

  // r = 1 / ||y||_2, then P = (y r)^2 -- which already sums to one.
  Ciphertext<word> r;
  inv_poly.Evaluate(boot.context, r, sq, mult_key);
  const int meet = boot.param->NPToLevel(r.GetNP());
  std::vector<Ciphertext<word>> P(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> levelled, prod;
    boot.context->LevelDown(levelled, y[bi], meet);
    boot.context->HMult(prod, levelled, r, mult_key);
    boot.context->HMult(P[bi], prod, prod, mult_key);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int out_level = boot.param->NPToLevel(P[0].GetNP());

  // ---- the reads --------------------------------------------------------
  // True softmax; the same circuit in the clear (separating fit error from
  // crypto error, SoftMaxHandler::PlainSoftMax's pattern); the transposed
  // control; and the row sums P * V depends on.
  double worst_true = 0.0, worst_plain = 0.0, transposed = 0.0;
  double worst_rowsum = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, P[bi]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int lane = 0; lane < layout.lanes; lane++) {
      for (int row = 0; row < layout.dim; row++) {
        // Host references for this (row, lane) row.
        std::vector<double> yp(layout.dim);
        double zsum = 0.0, ysq = 0.0;
        for (int col = 0; col < layout.dim; col++) {
          zsum += std::exp(m_eff * (u_of(S[lane][row][col]) - 1.0) / 2.0);
          yp[col] = exp_poly.PlainEvaluate(u_of(S[lane][row][col]));
          ysq += yp[col] * yp[col];
        }
        const double rp = inv_poly.PlainEvaluate((ysq - aff_b) / aff_a);
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          const double want_true =
              std::exp(m_eff * (u_of(S[lane][row][column]) - 1.0) / 2.0) /
              zsum;
          const double want_plain =
              (yp[column] * rp) * (yp[column] * rp);
          const double want_t =
              std::exp(m_eff * (u_of(S[lane][column][row]) - 1.0) / 2.0);
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          worst_true = std::max(worst_true, std::abs(got - want_true));
          worst_plain = std::max(worst_plain, std::abs(got - want_plain));
          transposed =
              std::max(transposed, std::abs(got - want_t / zsum));
        }
      }
    }
  }
  // Row sums need both ciphertexts together.
  {
    std::vector<std::vector<Complex>> all(layout.num_cts);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, P[bi]);
      boot.context->encoder_.Decode(all[bi], pt);
    }
    for (int lane = 0; lane < layout.lanes; lane++) {
      for (int row = 0; row < layout.dim; row++) {
        double rowsum = 0.0;
        for (int col = 0; col < layout.dim; col++) {
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, col, lane, ct_idx, slot, copy_slot);
          rowsum += all[col / layout.rank][slot].real();
        }
        worst_rowsum = std::max(worst_rowsum, std::abs(rowsum - 1.0));
      }
    }
  }

  std::cout << "softmax on the booted scores (M_eff " << m_eff
            << ", exp deg " << exp_used << " @" << exp_in << ".." << exp_out
            << ", invsqrt deg " << inv_used << " @" << poly_in << ".."
            << inv_out << ", P @" << out_level << "): vs true softmax "
            << worst_true << ", vs the same circuit in the clear "
            << worst_plain << std::endl;
  std::cout << "  row sums off one by " << worst_rowsum
            << ", norm interval [" << lo << ", " << hi
            << "] read off the data" << std::endl;
  std::cout << "  control: transposed " << transposed << std::endl;

  EXPECT_LT(worst_true, 3e-2)
      << "the softmax did not land on the true row distribution";
  EXPECT_LT(worst_plain, 1e-2)
      << "the encrypted circuit disagrees with its own plaintext twin, so "
         "the error is crypto, not fit";
  EXPECT_LT(worst_rowsum, 5e-2)
      << "the rows do not sum to one, which is the property P V consumes";
  EXPECT_GT(transposed, 5e-2);
}

// The normalized scores contract with values (Doing.md 1.5bw).
//
// 1.5bs's closing claim, now run: "the normalized P IS the next product's
// lhs verbatim -- same layout object, contraction = key = column, its
// two-call split already on the big-ciphertext boundary." The softmax of
// 1.5bv leaves P at level 3 at the primary addresses, and level 3 is
// fwd_level: the SAME converter and the SAME chain that descended Q take P
// down against V, split 16 + 16 over two calls whose lhs halves are P's
// own two big ciphertexts -- no repacking, no rotation, no new key. V, the
// rhs, packs per call exactly as K did (rows call * 16 + x at rhs rows x).
// The output lands at the primary addresses at level 0: attention output
// rows, one full leg cycle
//
//   scores @0 -> Boot -> @16 -> softmax -> P @3 -> descent @2 -> chain
//   -> @1 -> slots @0
//
// ready for the next bootstrap -- the leg's steady state on the real
// ladder.
//
// P is a convex combination per row (1.5bv's row sums), so the output is
// bounded by max |V| and the second product's noise is read against that:
// the test compares the output against the true softmax times V, against
// the DECRYPTED P times V (separating what the second contraction adds
// from what P arrived with), with the transposed read and the
// single-call half sum as the controls that must fail.
TEST(CiBootSet, TheNormalizedScoresContractWithValues) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int fwd_level = 3, chain_level = 2, inverse_level = 1;
  const int sub_degree = 128;
  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.num_cts, 2);
  const int half_keys = layout.contraction;  // 16

  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(swtch.context, sub_degree, fwd_level,
                             inverse_level, &layout);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  EvkRequest boot_req;
  bctx->AddRequiredRotations(boot_req, num_slots);
  boot.ui->PrepareRotationKey(boot_req);

  // ---- scores through the pipeline, softmax on top: 1.5bu + 1.5bv ------
  const double s = boot.param->GetScale(fwd_level);
  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, 0xB030);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  0xB031);
  const RealBatch v = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0xB032);

  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<Complex> slot_msg(boot.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            slot_msg[slot] = Complex(m[lane][row][column], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      boot.context->encoder_.Encode(pt, fwd_level, s, slot_msg);
      Ciphertext<word> enc;
      boot.ui->Encrypt(enc, pt);
      conv.SlotToSinC(swtch.context, out[bi], enc, swtch.ui->GetEvkMap());
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);
  handler.Multiply(res, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                   swtch.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());

  double carried = 0.0;
  std::vector<Ciphertext<word>> scores(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, res[bi], swtch.ui->GetEvkMap());
    carried = back.GetScale() / boot.param->base_scale_;
    back.SetNumSlots(num_slots);
    bctx->Boot(scores[bi], back, boot.ui->GetEvkMap());
  }
  const int top = boot.param->NPToLevel(scores[0].GetNP());

  const double m_eff = 8.0;
  std::vector<std::vector<std::vector<double>>> S(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
  double smin = 1e300, smax = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int col = 0; col < layout.dim; col++) {
        double val = 0.0;
        for (int x = 0; x < layout.contraction; x++) {
          val += a[lane][row][x] * b[lane][x][col];
        }
        S[lane][row][col] = val;
        smin = std::min(smin, val);
        smax = std::max(smax, val);
      }
    }
  }
  const double span = smax - smin;
  auto u_of = [&](double val) { return 2.0 * (val - smax) / span + 1.0; };

  const double half = m_eff / 4.0;
  auto exp_coeffs = cheddar::chebfit::Interpolate(
      [half](double val) { return std::exp(half * (val - 1.0)); }, 9);
  const int exp_in = top - 1;
  auto log2ceil = [](int n) {
    int r = 0;
    while ((1 << r) < n) r++;
    return r;
  };
  const int exp_used =
      EvalPoly<word>(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                     boot.param->GetScale(exp_in), true)
          .GetPolyDegree();
  const int exp_out = exp_in - log2ceil(exp_used + 1);
  EvalPoly<word> exp_poly(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                          boot.param->GetScale(exp_out), true);
  exp_poly.Compile(boot.context);

  double lo = 1e300, hi = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      double sqv = 0.0;
      for (int col = 0; col < layout.dim; col++) {
        const double yv = std::exp(half * (u_of(S[lane][row][col]) - 1.0));
        sqv += yv * yv;
      }
      lo = std::min(lo, sqv);
      hi = std::max(hi, sqv);
    }
  }
  lo *= 0.9;
  hi *= 1.1;
  const double aff_a = 0.5 * (hi - lo);
  const double aff_b = 0.5 * (hi + lo);
  auto inv_coeffs = cheddar::chebfit::Interpolate(
      [aff_a, aff_b](double val) {
        return 1.0 / std::sqrt(aff_a * val + aff_b);
      },
      15);
  const int sq_level = exp_out - 1;
  const int poly_in = sq_level - 1;
  const int inv_used =
      EvalPoly<word>(inv_coeffs, poly_in, boot.param->GetScale(poly_in),
                     boot.param->GetScale(poly_in), true)
          .GetPolyDegree();
  const int inv_out = poly_in - log2ceil(inv_used + 1);
  EvalPoly<word> inv_poly(inv_coeffs, poly_in,
                          boot.param->GetScale(poly_in),
                          boot.param->GetScale(inv_out), true);
  inv_poly.Compile(boot.context);

  std::vector<int> reduce_dist;
  for (int t = 0; t < 4; t++) {
    reduce_dist.push_back((num_slots / layout.rank) << t);
  }
  {
    EvkRequest rot_req;
    for (int d : reduce_dist) rot_req.AddRequest(d, sq_level);
    boot.ui->PrepareRotationKey(rot_req);
  }
  const auto &evk = boot.ui->GetEvkMap();
  const auto &mult_key = evk.GetMultiplicationKey();

  const double a1 = 2.0 / (span * carried);
  const double a0 = 1.0 - 2.0 * smax / span;
  std::vector<Ciphertext<word>> y(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Constant<word> c1;
    boot.context->encoder_.EncodeConstant(c1, top,
                                          boot.param->GetScale(top), a1);
    Ciphertext<word> t1, u_ct;
    boot.context->Mult(t1, scores[bi], c1);
    boot.context->Rescale(u_ct, t1);
    Constant<word> c0;
    boot.context->encoder_.EncodeConstant(c0, exp_in, u_ct.GetScale(), a0);
    boot.context->Add(u_ct, u_ct, c0);
    exp_poly.Evaluate(boot.context, y[bi], u_ct, mult_key);
  }
  Ciphertext<word> sq, term, rotated;
  boot.context->HMult(sq, y[0], y[0], mult_key);
  boot.context->HMult(term, y[1], y[1], mult_key);
  boot.context->Add(sq, sq, term);
  for (int d : reduce_dist) {
    boot.context->HRotAdd(rotated, sq, sq, evk.GetRotationKey(d), d);
    boot.context->Copy(sq, rotated);
  }
  {
    Constant<word> inv_a;
    boot.context->encoder_.EncodeConstant(
        inv_a, sq_level, boot.param->GetScale(sq_level), 1.0 / aff_a);
    Ciphertext<word> scaled;
    boot.context->Mult(scaled, sq, inv_a);
    boot.context->Rescale(sq, scaled);
    Constant<word> shift;
    boot.context->encoder_.EncodeConstant(shift, poly_in, sq.GetScale(),
                                          -aff_b / aff_a);
    boot.context->Add(sq, sq, shift);
  }
  Ciphertext<word> r;
  inv_poly.Evaluate(boot.context, r, sq, mult_key);
  const int meet = boot.param->NPToLevel(r.GetNP());
  std::vector<Ciphertext<word>> P(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> levelled, prod;
    boot.context->LevelDown(levelled, y[bi], meet);
    boot.context->HMult(prod, levelled, r, mult_key);
    boot.context->HMult(P[bi], prod, prod, mult_key);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(boot.param->NPToLevel(P[0].GetNP()), fwd_level)
      << "the softmax did not leave P at fwd_level, so the second descent "
         "has no rung to stand on";

  // What P actually arrived as, for separating the second product's own
  // error from P's incoming one.
  std::vector<std::vector<std::vector<double>>> P_dec(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
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
          P_dec[lane][row][column] = slots[slot].real();
        }
      }
    }
  }

  // ---- the second contraction: P's own ciphertexts are the lhs halves --
  std::vector<Ciphertext<word>> p_sinc(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    conv.SlotToSinC(swtch.context, p_sinc[bi], P[bi],
                    swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(p_sinc[bi].GetNP()), chain_level);
  }

  std::vector<Ciphertext<word>> out;
  for (int call = 0; call < 2; call++) {
    // V packs per call, exactly as K did: rows call * 16 + x at rhs rows x.
    RealBatch h(layout.lanes,
                std::vector<std::vector<double>>(
                    layout.dim, std::vector<double>(layout.dim, 0.0)));
    for (int lane = 0; lane < layout.lanes; lane++) {
      for (int x = 0; x < half_keys; x++) {
        for (int col = 0; col < layout.dim; col++) {
          h[lane][x][col] = v[lane][call * half_keys + x][col];
        }
      }
    }
    std::vector<Ciphertext<word>> v_rhs, p_lhs, part;
    build(h, layout.num_cts, v_rhs);
    p_lhs.push_back(std::move(p_sinc[call]));
    handler.Multiply(part, p_lhs, v_rhs,
                     swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    if (call == 0) {
      out = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(out[bi], out[bi], part[bi]);
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // ---- back to slots at level 0, read against the references -----------
  double worst_true = 0.0, worst_incoming = 0.0;
  double transposed = 0.0, half_sum = 0.0, biggest = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, out[bi], swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(back.GetNP()), 0);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, back);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int lane = 0; lane < layout.lanes; lane++) {
      for (int row = 0; row < layout.dim; row++) {
        // The true softmax row, once per (row, lane).
        std::vector<double> p_true(layout.dim);
        double zsum = 0.0;
        for (int k = 0; k < layout.dim; k++) {
          p_true[k] = std::exp(m_eff * (u_of(S[lane][row][k]) - 1.0) / 2.0);
          zsum += p_true[k];
        }
        for (double &pv : p_true) pv /= zsum;
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          double want = 0.0, want_in = 0.0, want_t = 0.0, want_half = 0.0;
          for (int k = 0; k < layout.dim; k++) {
            want += p_true[k] * v[lane][k][column];
            want_in += P_dec[lane][row][k] * v[lane][k][column];
            want_t += P_dec[lane][column][k] * v[lane][k][row];
            if (k < half_keys) {
              want_half += P_dec[lane][row][k] * v[lane][k][column];
            }
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          biggest = std::max(biggest, std::abs(want));
          worst_true = std::max(worst_true, std::abs(got - want));
          worst_incoming =
              std::max(worst_incoming, std::abs(got - want_in));
          transposed = std::max(transposed, std::abs(got - want_t));
          half_sum = std::max(half_sum, std::abs(got - want_half));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "P x V off the softmax's own P (sub_degree " << sub_degree
            << ", P @" << fwd_level << " -> chain -> slots @0): vs true "
            << "softmax x V " << worst_true << ", vs decrypted P x V "
            << worst_incoming << " (|output| up to " << biggest << ")"
            << std::endl;
  std::cout << "  controls: transposed " << transposed
            << ", single-call half sum " << half_sum << std::endl;

  EXPECT_LT(worst_true, 2e-3)
      << "the attention output did not land at the primary addresses";
  EXPECT_LT(worst_incoming, 1e-3)
      << "the second contraction added more than its own floor";
  EXPECT_GT(transposed, 1e-2);
  EXPECT_GT(half_sum, 1e-2)
      << "one call alone matches, so the two-call sum never happened";
}

// The projection transport, from the doorstep to the chain (Doing.md 1.5bx).
//
// Every operand so far was BUILT at the chain's primary addresses. The real
// leg's Q/K/V arrive from the projections, and on R+ their post-HalfBoot
// slots are forced by physics into [rev9(component) | rev7(token)] -- token
// in the LOW seven bits, channel/head in the component field -- g-mixed by
// the banded ModPack. Undoing that (the rank-512 unmix and the token/head
// field exchange, both hoisted transforms ABOVE the level-7 zone) is the
// doorstep transform, the next increment. What this test pins is everything
// from the doorstep to the chain, and the DESIGN of the doorstep target:
//
//   THE DOORSTEP LAYOUT (one choice for Q, K and V): ciphertext l holds
//   channels [16l, 16l+16), and entry (token t, channel c, head i) sits at
//
//       slot = rev7(t) * 512 + rev4(c mod 16) * 32 + i
//
//   (the head field arrives un-reversed because the weight-row packing
//   w = [rev5(i) | c mod 16] absorbs rev9's reversal -- a free relabelling
//   of W_Q/W_K/W_V rows).
//
// Three claims, all measured here:
//
// 1. ROPE COSTS ONE MASK SET, SHARED BY Q AND K. Channel c pairs with
//    c +- 64 = ciphertext l +- 4 (ct pairs, zero rotations, 1.5bs's shape),
//    and the angle token * theta[c mod 64] reads the SAME slot fields for
//    both tensors -- in the chain layout Q and K sit transposed and needed
//    two mask sets; here they need one.
//
// 2. Q'S TRANSPORT IS A FREE FOLD. Its chain block [rev4(c') | rev7(q)] is
//    a block permutation of its doorstep block [rev7(q) | rev4(c')], lane
//    (head) untouched -- so it rides the nested forward as a column
//    relabelling (CiSinCConverter's forward_premap), zero diagonal growth
//    (the sub-32 forward already holds all 2048 lattice diagonals), zero
//    extra rotations, zero levels. V's transport is the SAME permutation
//    (its chain block is [rev4(c') | rev7(t)]), so it is covered by
//    construction and not re-run.
//
// 3. K MUST CROSS CIPHERTEXTS, AND 32 MASKED ROTATIONS PER CALL DO IT. As
//    the rhs of the scores product K's column is the TOKEN, so its top
//    three token bits become the ciphertext index -- data from every
//    doorstep ciphertext lands in every chain ciphertext, which no per-ct
//    transform can do. The exchange: piece (l -> t_hi) masks source ct l
//    where slot bits [11..9] == rev3(t_hi) (a 0/1 mask, one level, shared
//    across l), rotates by (rev3(t_hi) - l mod 4) * 512 -- at most 10
//    distinct indices -- and adds into dest ct t_hi. That leaves the
//    intermediate block [rev4(t') | l mod 4 | rev4(c')], and the remaining
//    interior swap to the chain's [rev4(t') | rev4(c') | rev2(l mod 4)*2]
//    is again a lane-preserving block permutation: K's own free fold. The
//    per-call channel split (rows x < 64) falls out for free: call c's
//    pieces draw only from source cts 4c..4c+3, and every other slot of
//    the masked sum is exactly zero, which IS the rhs half-contraction
//    contract.
//
// The walk: Q doorstep @4 --RoPE--> @3 --premapped fwd--> @2; K doorstep
// @5 --RoPE--> @4 --exchange--> @3 --premapped fwd--> @2; two chain calls
// --> @1; parts read through the scan against host RoPE'd scores, with the
// transposed and un-RoPE'd reads as controls. RoPE, the exchange and its
// rotations run on ci16_35 (plain ops and single key switches, zone-free);
// the hoisted conversions stay on the trio.
TEST(CiBootSet, TheProjectionTransportReachesTheChain) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int k_door_level = 5;   // K spends RoPE and the exchange
  const int q_door_level = 4;   // Q spends only RoPE
  const int exchange_level = 4;
  const int conv_level = 3;
  const int chain_level = 2;
  const int sub_degree = 32;

  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  const int half = layout.contraction;  // 64

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  // The doorstep address of entry (token, channel, head).
  auto door_slot = [&](int t, int c, int i) {
    return (rev(t, 7) << 9) | (rev(c % 16, 4) << 5) | i;
  };

  // The two block premaps of the class comment. K's interior swap also
  // REVERSES the moved field: the chain's low bits hold rev7(row) and the
  // row's ct-index part arrives plain from the exchange, so the field
  // content is rev3'd on the way down, not just repositioned.
  const int num_blocks = boot.Degree() / sub_degree;  // 2048
  std::vector<int> pre_q(num_blocks), pre_k(num_blocks);
  for (int b = 0; b < num_blocks; b++) {
    pre_q[b] = ((b & 15) << 7) | (b >> 4);
    pre_k[b] = (b & (15 << 7)) | ((b & 15) << 3) | rev((b >> 4) & 7, 3);
  }

  // The host identity behind claims 2 and 3: the composed maps land every
  // entry at exactly its primary LocateSlot address.
  {
    int bad = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int c = 0; c < layout.dim; c++) {
        for (int i = 0; i < layout.lanes; i++) {
          int ct_idx, slot, copy_slot;
          // Q: (row = query token, column = channel, lane = head).
          layout.LocateSlot(t, c, i, ct_idx, slot, copy_slot);
          const int ds = door_slot(t, c, i);
          if (ct_idx != c / 16 ||
              slot != ((pre_q[ds >> 5] << 5) | (ds & 31))) {
            bad++;
          }
          // K: (row = channel within the call's half, column = token). The
          // exchange leaves block [rev4(t') | l mod 4 | rev4(c')] in ct
          // t/16; pre_k must take it home.
          layout.LocateSlot(c % half, t, i, ct_idx, slot, copy_slot);
          const int ib = (rev(t % 16, 4) << 7) | (((c / 16) % 4) << 4) |
                         rev(c % 16, 4);
          if (ct_idx != t / 16 || slot != ((pre_k[ib] << 5) | i)) bad++;
        }
      }
    }
    ASSERT_EQ(bad, 0) << "the composed transport maps do not reach the "
                         "chain's primary addresses";
  }

  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> conv_q(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_q);
  CiSinCConverter<word> conv_k(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_k);
  const auto t1 = std::chrono::steady_clock::now();
  EvkRequest req;
  conv_q.AddRequiredRotations(req);
  conv_k.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  // The exchange's rotation keys: (rev3(t_hi) - l mod 4) * 512, at most 10
  // distinct nonzero indices, on ci16_35.
  {
    std::set<int> idxs;
    for (int u = 0; u < 4; u++) {
      for (int v = 0; v < 8; v++) {
        const int rot = (v - u) * 512;
        if (rot != 0) {
          idxs.insert((rot % boot.Degree() + boot.Degree()) % boot.Degree());
        }
      }
    }
    for (int idx : idxs) boot.ui->PrepareRotationKey(idx, exchange_level);
  }

  const RealBatch q = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0xB0F1);
  const RealBatch kk = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                   layout.dim, 0.08, 0xB0F2);
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  // Doorstep encrypt: ciphertext l of tensor m[head][token][channel].
  auto encrypt_door = [&](const RealBatch &m, int l, int level,
                          Ciphertext<word> &out) {
    std::vector<Complex> slot_msg(boot.Degree(), Complex(0.0, 0.0));
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < layout.rank; cp++) {
        for (int i = 0; i < layout.lanes; i++) {
          slot_msg[door_slot(t, l * 16 + cp, i)] =
              Complex(m[i][t][l * 16 + cp], 0.0);
        }
      }
    }
    Plaintext<word> pt;
    boot.context->encoder_.Encode(pt, level, boot.param->GetScale(level),
                                  slot_msg);
    boot.ui->Encrypt(out, pt);
  };

  // RoPE on the doorstep layout: claim 1's one mask set (per level).
  auto rope_door = [&](std::vector<Ciphertext<word>> &cts, int level) {
    const double pt_scale = boot.param->GetRescalePrimeProd(level);
    for (int lo = 0; lo < layout.num_cts / 2; lo++) {
      std::vector<Complex> cm(boot.Degree(), Complex(0.0, 0.0));
      std::vector<Complex> sm(boot.Degree(), Complex(0.0, 0.0));
      std::vector<Complex> nm(boot.Degree(), Complex(0.0, 0.0));
      for (int t = 0; t < layout.dim; t++) {
        for (int cp = 0; cp < layout.rank; cp++) {
          const double ang = t * theta[lo * 16 + cp];
          for (int i = 0; i < layout.lanes; i++) {
            const int slot = door_slot(t, lo * 16 + cp, i);
            cm[slot] = Complex(std::cos(ang), 0.0);
            sm[slot] = Complex(std::sin(ang), 0.0);
            nm[slot] = Complex(-std::sin(ang), 0.0);
          }
        }
      }
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      boot.context->encoder_.Encode(cos_pt, level, pt_scale, cm);
      boot.context->encoder_.Encode(sin_pt, level, pt_scale, sm);
      boot.context->encoder_.Encode(neg_sin_pt, level, pt_scale, nm);
      Ciphertext<word> &lo_ct = cts[lo];
      Ciphertext<word> &hi_ct = cts[lo + layout.num_cts / 2];
      Ciphertext<word> a, b, d;
      boot.context->Mult(a, lo_ct, cos_pt);
      boot.context->Mult(b, hi_ct, neg_sin_pt);
      boot.context->Add(a, a, b);
      boot.context->Mult(b, hi_ct, cos_pt);
      boot.context->Mult(d, lo_ct, sin_pt);
      boot.context->Add(b, b, d);
      boot.context->Rescale(lo_ct, a);
      boot.context->Rescale(hi_ct, b);
    }
  };

  std::vector<Ciphertext<word>> q_cts(layout.num_cts), k_cts(layout.num_cts);
  for (int l = 0; l < layout.num_cts; l++) {
    encrypt_door(q, l, q_door_level, q_cts[l]);
    encrypt_door(kk, l, k_door_level, k_cts[l]);
  }
  rope_door(q_cts, q_door_level);
  rope_door(k_cts, k_door_level);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Claim 3's exchange: per call, 8 dest cts out of 4 source cts.
  const double ex_pt_scale = boot.param->GetRescalePrimeProd(exchange_level);
  std::vector<Plaintext<word>> sel(8);
  for (int v = 0; v < 8; v++) {
    std::vector<Complex> msg(boot.Degree(), Complex(0.0, 0.0));
    for (int s = 0; s < boot.Degree(); s++) {
      if (((s >> 9) & 7) == v) msg[s] = Complex(1.0, 0.0);
    }
    boot.context->encoder_.Encode(sel[v], exchange_level, ex_pt_scale, msg);
  }
  double exchange_seconds = 0.0;
  auto exchange = [&](int call) {
    const auto e0 = std::chrono::steady_clock::now();
    std::vector<Ciphertext<word>> out(layout.num_cts);
    for (int t_hi = 0; t_hi < layout.num_cts; t_hi++) {
      const int v = rev(t_hi, 3);
      Ciphertext<word> acc;
      bool first = true;
      for (int l = call * 4; l < call * 4 + 4; l++) {
        Ciphertext<word> piece;
        boot.context->Mult(piece, k_cts[l], sel[v]);
        const int rot = (v - l % 4) * 512;
        if (rot != 0) {
          const int idx =
              (rot % boot.Degree() + boot.Degree()) % boot.Degree();
          Ciphertext<word> moved;
          boot.context->HRot(moved, piece,
                             boot.ui->GetEvkMap().GetRotationKey(idx), idx);
          piece = std::move(moved);
        }
        if (first) {
          acc = std::move(piece);
          first = false;
        } else {
          boot.context->Add(acc, acc, piece);
        }
      }
      boot.context->Rescale(out[t_hi], acc);
    }
    cudaDeviceSynchronize();
    exchange_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - e0)
            .count();
    return out;
  };

  double fwd_seconds = 0.0;
  int fwd_count = 0;
  auto convert = [&](CiSinCConverter<word> &conv,
                     std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      ASSERT_EQ(boot.param->NPToLevel(ct.GetNP()), conv_level);
      Ciphertext<word> sinc;
      const auto f0 = std::chrono::steady_clock::now();
      conv.SlotToSinC(swtch.context, sinc, ct, swtch.ui->GetEvkMap());
      cudaDeviceSynchronize();
      fwd_seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - f0)
                         .count();
      fwd_count++;
      ct = std::move(sinc);
    }
  };
  convert(conv_q, q_cts);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Ciphertext<word>> res;
  double mult_seconds = 0.0;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> rhs = exchange(call);
    convert(conv_k, rhs);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_cts[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> part;
    const auto m0 = std::chrono::steady_clock::now();
    handler.Multiply(part, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    mult_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - m0)
            .count();
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  // Parts read through the scan (the return to slots is 1.5bt's, already
  // measured; the claim here is the transport).
  std::vector<std::vector<Complex>> got(layout.dim);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    ASSERT_EQ(boot.param->NPToLevel(res[bi].GetNP()), chain_level - 1);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    boot.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(
          bridge, chain_level - 1,
          small.param->GetScale(chain_level - 1), comp[j]);
      small.context->encoder_.DecodeSinC(got[bi * layout.rank + j], bridge,
                                         sub_degree);
    }
  }

  double worst = 0.0, transposed = 0.0, norope = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0, want_raw = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          want += q_ref[lane][row][c] * k_ref[lane][column][c];
          want_t += q_ref[lane][column][c] * k_ref[lane][row][c];
          want_raw += q[lane][row][c] * kk[lane][column][c];
        }
        int part, index;
        layout.LocatePart(row, column, lane, part, index);
        const double g = got[part][index].real();
        worst = std::max(worst, std::abs(g - want));
        transposed = std::max(transposed, std::abs(g - want_t));
        norope = std::max(norope, std::abs(g - want_raw));
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "the projection transport (doorstep @" << k_door_level
            << "/@" << q_door_level << " -> RoPE -> exchange -> premapped "
            << "descent @" << conv_level << " -> chain x2): products "
            << worst << std::endl;
  std::cout << "  controls: transposed " << transposed << ", un-RoPE'd "
            << norope << std::endl;
  std::cout << "  cost: converter builds "
            << std::chrono::duration<double>(t1 - t0).count()
            << " s (both, forward-only), forward "
            << fwd_seconds / std::max(fwd_count, 1) << " s/ct over "
            << fwd_count << " cts, exchange " << exchange_seconds / 2.0
            << " s/call (32 masked rotations), one chain call "
            << mult_seconds / 2.0 << " s" << std::endl;

  EXPECT_LT(worst, 5e-2)
      << "the transported operands did not land the scores at the primary "
         "parts";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(norope, 3e-2)
      << "RoPE on the doorstep layout did nothing the product can see";
}

// The doorstep transform dissolves (Doing.md 1.5by).
//
// 1.5bx left one piece between HalfBoot's landing and its doorstep: the
// rank-512 unmix of the banded ModPack plus the token/head field exchange.
// The unmix, as a slot transform, is DEAD in every ordering: the scan's
// triangular fanout walks the position axis, bit reversals scatter every
// step to its own offset, and the count approaches the full 65536 -- no
// level budget holds it. This test measures the design that makes the
// transform unnecessary:
//
// 1. HALF-DENSITY MODPACK IS CLEAN BY CONSTRUCTION. The banded partner of
//    component I is 512 - I, so a projection ciphertext carrying only 256
//    live components (I < 256: heads 0..15 of its channel group, packing
//    I = [head(5) | channel'(4)]) recomposes with every partner term ZERO:
//    each entry lands clean at its primary coefficient, and coefficients
//    I >= 256 hold shifted DUPLICATES (comp_{512-I}[t+1]), not mixtures --
//    asserted here exactly against HostRecompose at rank 512. The cost is
//    ciphertext count: 16 half ciphertexts per tensor instead of 8 (the
//    ModPack key-switch count is unchanged -- every weight row switches
//    once either way -- but each half needs its own HalfBoot).
//
// 2. THE KILL IS FREE AND THE MERGE IS ONE ROTATION. Post-transform, live
//    entries sit at slot [rev4(c') | rev5(head) | rev7(t)] with slot bit 7
//    = 0 and the duplicates at bit 7 = 1 -- so RoPE's masks, built over
//    live addresses only, kill the duplicates as a side effect (RoPE stays
//    ONE mask set, shared by Q, K and both halves). The halves then merge
//    by a single rotation: rev5(16 + v) = rev5(v) + 1, so shifting the
//    upper-head half by 128 slots lands its lanes exactly in the killed
//    duplicate addresses. Eight rotations and adds per tensor, one key.
//
// 3. THE EXCHANGE IS 63 DIAGONALS. Head must reach the low five slot bits
//    (the chain's lanes) from the component field, and token must leave
//    them -- forced by ModPack (position high) vs the SinC lanes. With the
//    packing above the exchange collapses to ONE lane-count field swap,
//    slot bits [11..7] <-> [4..0]: offsets (B - A) * 127, exactly 63
//    diagonals, one level, one hoisted LinearTransform -- run on ci16_35
//    at level 9, ABOVE the level-7 hoist zone. Every remaining block
//    tangle (the doubly-split token field, rev5'd lanes) is absorbed for
//    free by 1.5bx's converter premap and a lane relabelling in the read.
//
// K's cross-ciphertext exchange re-derives on the new fields (mask bits
// [9..7] = rev3(t_hi), stride-128 rotations, same 32 pieces per call); the
// premaps are built by ENUMERATION against LocateSlot rather than by hand
// -- 1.5bx's lesson -- and asserted bijective before any device time.
//
// The walk: banded half-density slot images @10 (the measured BitRev of
// 1.5bh stands in for HalfBoot) -> RoPE + kill @10 -> merge -> exchange
// @9 -> [K: cross-ct @8] -> premapped descent @3 -> chain x2 -> parts,
// against host RoPE'd scores; transposed and un-RoPE'd reads as controls.
TEST(CiBootSet, HalfDensityProjectionsDissolveTheDoorstepTransform) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  const int enc_level = 10;       // the stand-in for HalfBoot's landing
  const int exchange_level = 9;   // the 63-diagonal field swap
  const int cross_level = 8;      // K's cross-ciphertext masks
  const int conv_level = 3;
  const int chain_level = 2;
  const int sub_degree = 32;
  const int n = boot.Degree();

  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  const int half = layout.contraction;  // 64

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  // The field swap: slot bits [11..7] <-> [4..0].
  auto exch = [](int s) {
    const int a = (s >> 7) & 31, b = s & 31;
    return (s & ~((31 << 7) | 31)) | (b << 7) | a;
  };
  // Post-merge, pre-exchange address of entry (token t, channel c, head i)
  // in ciphertext c / 16: rev16 of the banded primary coefficient.
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };
  // Post-exchange (the transformed doorstep).
  auto door1 = [&](int t, int c, int i) { return exch(door0(t, c, i)); };
  // K's post-cross intermediate: the [9..7] field carries l mod 4, the
  // ciphertext index becomes t / 16.
  auto door1k = [&](int t, int c, int i) {
    return (door1(t, c, i) & ~(7 << 7)) | (((c / 16) % 4) << 7);
  };

  // ---- the half-density claim, exactly, on the host -------------------
  {
    std::mt19937_64 gen(0xB101);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> comp(
        512, std::vector<double>(layout.dim, 0.0));
    for (int I = 0; I < 256; I++) {
      for (int p = 0; p < layout.dim; p++) comp[I][p] = dist(gen);
    }
    const auto rec = HostRecompose(comp, 512, layout.dim);
    int bad = 0;
    for (int p = 0; p < layout.dim; p++) {
      for (int I = 0; I < 512; I++) {
        const double got = rec[static_cast<size_t>(p) * 512 + I];
        double want = 0.0;
        if (I < 256) {
          want = comp[I][p];  // clean at the primary, no partner term
        } else if (p + 1 < layout.dim) {
          want = comp[512 - I][p + 1];  // a duplicate, not a mixture
        }
        if (got != want) bad++;
      }
    }
    ASSERT_EQ(bad, 0) << "half-density recomposition is not clean";
  }

  // ---- premaps by enumeration, asserted before device time ------------
  const int num_blocks = n / sub_degree;
  std::vector<int> pre_q(num_blocks, -1), pre_k(num_blocks, -1);
  {
    int bad = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int c = 0; c < layout.dim; c++) {
        int ct_idx, slot, copy_slot;
        // Q: (row = query token, column = channel); the ct index must
        // already match, and door1's block must map to the chain's.
        layout.LocateSlot(t, c, 0, ct_idx, slot, copy_slot);
        if (ct_idx != c / 16) bad++;
        int db = door1(t, c, 0) >> 5;
        if (pre_q[db] == -1) pre_q[db] = slot >> 5;
        if (pre_q[db] != (slot >> 5)) bad++;
        // K: (row = channel within the call's half, column = token),
        // through the cross-ct intermediate.
        layout.LocateSlot(c % half, t, 0, ct_idx, slot, copy_slot);
        if (ct_idx != t / 16) bad++;
        db = door1k(t, c, 0) >> 5;
        if (pre_k[db] == -1) pre_k[db] = slot >> 5;
        if (pre_k[db] != (slot >> 5)) bad++;
        // Lanes: the low five bits must be rev5(head), untouched by both
        // maps (spot-check one head).
        if ((door1(t, c, 21) & 31) != rev(21, 5)) bad++;
      }
    }
    ASSERT_EQ(bad, 0) << "the enumerated maps are inconsistent";
    for (int b = 0; b < num_blocks; b++) {
      ASSERT_NE(pre_q[b], -1) << "door1 does not cover block " << b;
    }
    // door1k only reaches deposit fields 0..3; the uncovered input blocks
    // hold EXACT zeros (no cross piece deposits there), so completing the
    // bijection by sending them, in order, to the unused chain blocks --
    // which are exactly the rows >= 64 the half-contraction contract wants
    // zero -- is not a convention but the contract itself.
    {
      std::vector<char> used(num_blocks, 0);
      for (int b = 0; b < num_blocks; b++) {
        if (pre_k[b] != -1) used[pre_k[b]] = 1;
      }
      std::vector<int> free_out;
      for (int b = 0; b < num_blocks; b++) {
        if (!used[b]) free_out.push_back(b);
      }
      size_t fo = 0;
      for (int b = 0; b < num_blocks; b++) {
        if (pre_k[b] == -1) {
          ASSERT_LT(fo, free_out.size());
          pre_k[b] = free_out[fo++];
        }
      }
      ASSERT_EQ(fo, free_out.size())
          << "the dead-block completion did not balance";
    }
  }

  // ---- keys and converters --------------------------------------------
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> conv_q(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_q);
  CiSinCConverter<word> conv_k(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_k);
  const auto t1 = std::chrono::steady_clock::now();
  EvkRequest req;
  conv_q.AddRequiredRotations(req);
  conv_k.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  // The exchange, as a striped permutation matrix: 63 diagonals at
  // offsets 127 * [-31, 31]. The straddle around zero is PrepareSinC's
  // window wall: `DetermineStride` would see the wrapped negatives, find
  // gcd 1 and demand bs * gs >= num_slots. Same fix as the SinC prefix:
  // pre_rotation = -window and additional_pt_rot = window align the grid
  // to 127 * [0, 62], the result comes back rotated by the window, and
  // one HRot finishes it.
  const int window = 31 * 127;
  const auto e0 = std::chrono::steady_clock::now();
  cheddar::StripedMatrix em(n, n);
  for (int r = 0; r < n; r++) {
    const int in = exch(r);  // an involution: E^-1 = E
    const int off = ((in - r) % n + n) % n;
    em.try_emplace(off, n, Complex(0.0, 0.0));
    em[off][r] = Complex(1.0, 0.0);
  }
  ASSERT_EQ(em.GetNumDiag(), 63)
      << "the field swap left its 63-diagonal budget";
  cheddar::LinearTransform<word> lt_exch(
      boot.context, em, exchange_level,
      boot.param->GetRescalePrimeProd(exchange_level), 8, 8,
      /*pre_rotation=*/-window, /*additional_pt_rot=*/window);
  const auto t2 = std::chrono::steady_clock::now();
  const int window_back = n - window;
  {
    EvkRequest ereq;
    lt_exch.AddRequiredRotations(ereq);
    ereq.AddRequest(window_back, exchange_level - 1);
    boot.ui->PrepareRotationKey(ereq);
  }
  // The merge rotation and K's cross-ct rotations, on ci16_35.
  const int merge_idx = n - 128;  // dest[s + 128] = src[s]
  boot.ui->PrepareRotationKey(merge_idx, exchange_level);
  {
    std::set<int> idxs;
    for (int u = 0; u < 4; u++) {
      for (int v = 0; v < 8; v++) {
        const int rot = (v - u) * 128;
        if (rot != 0) idxs.insert((rot % n + n) % n);
      }
    }
    for (int idx : idxs) boot.ui->PrepareRotationKey(idx, cross_level);
  }

  // ---- data, and the banded half-density slot images ------------------
  const RealBatch q = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0xB102);
  const RealBatch kk = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                   layout.dim, 0.08, 0xB103);
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  // Family fam of channel group l: heads [fam*16, fam*16+16), components
  // I = (i - fam*16) * 16 + c', I < 256 live, banded-recomposed and sent
  // through the measured coefficient -> slot bit reversal.
  auto encrypt_half = [&](const RealBatch &m, int l, int fam,
                          Ciphertext<word> &out) {
    std::vector<std::vector<double>> comp(
        512, std::vector<double>(layout.dim, 0.0));
    for (int hh = 0; hh < 16; hh++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int p = 0; p < layout.dim; p++) {
          comp[hh * 16 + cp][p] = m[fam * 16 + hh][p][l * 16 + cp];
        }
      }
    }
    const auto rec = HostRecompose(comp, 512, layout.dim);
    std::vector<Complex> slot_msg(n, Complex(0.0, 0.0));
    for (int coeff = 0; coeff < n; coeff++) {
      slot_msg[rev(coeff, 16)] = Complex(rec[coeff], 0.0);
    }
    Plaintext<word> pt;
    boot.context->encoder_.Encode(pt, enc_level,
                                  boot.param->GetScale(enc_level), slot_msg);
    boot.ui->Encrypt(out, pt);
  };

  // RoPE + kill on the half-density layout: masks over LIVE addresses
  // only (bit 7 clear), zero on the duplicates -- one mask set for both
  // families and both tensors.
  auto rope_and_kill = [&](std::vector<Ciphertext<word>> &a_cts,
                           std::vector<Ciphertext<word>> &b_cts) {
    const double pt_scale = boot.param->GetRescalePrimeProd(enc_level);
    for (int lo = 0; lo < 4; lo++) {
      std::vector<Complex> cm(n, Complex(0.0, 0.0));
      std::vector<Complex> sm(n, Complex(0.0, 0.0));
      std::vector<Complex> nm(n, Complex(0.0, 0.0));
      for (int t = 0; t < layout.dim; t++) {
        for (int cp = 0; cp < 16; cp++) {
          const double ang = t * theta[lo * 16 + cp];
          for (int hh = 0; hh < 16; hh++) {
            const int slot = door0(t, lo * 16 + cp, hh);  // bit 7 clear
            cm[slot] = Complex(std::cos(ang), 0.0);
            sm[slot] = Complex(std::sin(ang), 0.0);
            nm[slot] = Complex(-std::sin(ang), 0.0);
          }
        }
      }
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      boot.context->encoder_.Encode(cos_pt, enc_level, pt_scale, cm);
      boot.context->encoder_.Encode(sin_pt, enc_level, pt_scale, sm);
      boot.context->encoder_.Encode(neg_sin_pt, enc_level, pt_scale, nm);
      for (int fam = 0; fam < 2; fam++) {
        std::vector<Ciphertext<word>> &cts = (fam == 0) ? a_cts : b_cts;
        Ciphertext<word> &lo_ct = cts[lo];
        Ciphertext<word> &hi_ct = cts[lo + 4];
        Ciphertext<word> aa, bb, dd;
        boot.context->Mult(aa, lo_ct, cos_pt);
        boot.context->Mult(bb, hi_ct, neg_sin_pt);
        boot.context->Add(aa, aa, bb);
        boot.context->Mult(bb, hi_ct, cos_pt);
        boot.context->Mult(dd, lo_ct, sin_pt);
        boot.context->Add(bb, bb, dd);
        boot.context->Rescale(lo_ct, aa);
        boot.context->Rescale(hi_ct, bb);
      }
    }
  };

  // Merge: the upper-head family shifts by 128 into the killed duplicate
  // addresses (rev5(16 + v) = rev5(v) + 1), one rotation per pair.
  auto merge = [&](std::vector<Ciphertext<word>> &a_cts,
                   std::vector<Ciphertext<word>> &b_cts) {
    for (int l = 0; l < layout.num_cts; l++) {
      Ciphertext<word> moved;
      boot.context->HRot(moved, b_cts[l],
                         boot.ui->GetEvkMap().GetRotationKey(merge_idx),
                         merge_idx);
      boot.context->Add(a_cts[l], a_cts[l], moved);
    }
  };

  auto exchange_all = [&](std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      ASSERT_EQ(boot.param->NPToLevel(ct.GetNP()), exchange_level);
      Ciphertext<word> shifted, swapped;
      lt_exch.Evaluate(boot.context, shifted, ct, boot.ui->GetEvkMap());
      boot.context->HRot(swapped, shifted,
                         boot.ui->GetEvkMap().GetRotationKey(window_back),
                         window_back);
      ct = std::move(swapped);
    }
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8);
  for (int l = 0; l < 8; l++) {
    encrypt_half(q, l, 0, q_a[l]);
    encrypt_half(q, l, 1, q_b[l]);
    encrypt_half(kk, l, 0, k_a[l]);
    encrypt_half(kk, l, 1, k_b[l]);
  }
  rope_and_kill(q_a, q_b);
  rope_and_kill(k_a, k_b);
  merge(q_a, q_b);
  merge(k_a, k_b);
  exchange_all(q_a);
  exchange_all(k_a);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // K's cross-ct exchange on the post-swap fields: mask bits [9..7] ==
  // rev3(t_hi), deposit l mod 4, stride-128 rotations.
  const double cr_pt_scale = boot.param->GetRescalePrimeProd(cross_level);
  std::vector<Plaintext<word>> sel(8);
  for (int v = 0; v < 8; v++) {
    std::vector<Complex> msg(n, Complex(0.0, 0.0));
    for (int s = 0; s < n; s++) {
      if (((s >> 7) & 7) == v) msg[s] = Complex(1.0, 0.0);
    }
    boot.context->encoder_.Encode(sel[v], cross_level, cr_pt_scale, msg);
  }
  auto cross = [&](int call) {
    std::vector<Ciphertext<word>> out(layout.num_cts);
    for (int t_hi = 0; t_hi < layout.num_cts; t_hi++) {
      const int v = rev(t_hi, 3);
      Ciphertext<word> acc;
      bool first = true;
      for (int l = call * 4; l < call * 4 + 4; l++) {
        Ciphertext<word> piece;
        boot.context->Mult(piece, k_a[l], sel[v]);
        const int rot = (v - l % 4) * 128;
        if (rot != 0) {
          const int idx = (rot % n + n) % n;
          Ciphertext<word> moved;
          boot.context->HRot(moved, piece,
                             boot.ui->GetEvkMap().GetRotationKey(idx), idx);
          piece = std::move(moved);
        }
        if (first) {
          acc = std::move(piece);
          first = false;
        } else {
          boot.context->Add(acc, acc, piece);
        }
      }
      boot.context->Rescale(out[t_hi], acc);
    }
    return out;
  };

  double fwd_seconds = 0.0;
  int fwd_count = 0;
  auto convert = [&](CiSinCConverter<word> &conv,
                     std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      Ciphertext<word> at_level;
      boot.context->LevelDown(at_level, ct, conv_level);
      Ciphertext<word> sinc;
      const auto f0 = std::chrono::steady_clock::now();
      conv.SlotToSinC(swtch.context, sinc, at_level, swtch.ui->GetEvkMap());
      cudaDeviceSynchronize();
      fwd_seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - f0)
                         .count();
      fwd_count++;
      ct = std::move(sinc);
    }
  };
  convert(conv_q, q_a);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Ciphertext<word>> res;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> rhs = cross(call);
    convert(conv_k, rhs);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_a[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> part;
    handler.Multiply(part, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  // Parts read through the scan; the chain's lane holds head rev5(lane).
  std::vector<std::vector<Complex>> got(layout.dim);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    ASSERT_EQ(boot.param->NPToLevel(res[bi].GetNP()), chain_level - 1);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    boot.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(
          bridge, chain_level - 1,
          small.param->GetScale(chain_level - 1), comp[j]);
      small.context->encoder_.DecodeSinC(got[bi * layout.rank + j], bridge,
                                         sub_degree);
    }
  }

  double worst = 0.0, transposed = 0.0, norope = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0, want_raw = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          want += q_ref[head][row][c] * k_ref[head][column][c];
          want_t += q_ref[head][column][c] * k_ref[head][row][c];
          want_raw += q[head][row][c] * kk[head][column][c];
        }
        int part, index;
        layout.LocatePart(row, column, lane, part, index);
        const double g = got[part][index].real();
        worst = std::max(worst, std::abs(g - want));
        transposed = std::max(transposed, std::abs(g - want_t));
        norope = std::max(norope, std::abs(g - want_raw));
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "half-density projections through the dissolved doorstep "
            << "(banded images @" << enc_level << " -> RoPE+kill -> merge "
            << "-> 63-diagonal exchange @" << exchange_level
            << " -> [K: cross @" << cross_level
            << "] -> premapped descent @" << conv_level
            << " -> chain x2): products " << worst << std::endl;
  std::cout << "  controls: transposed " << transposed << ", un-RoPE'd "
            << norope << std::endl;
  std::cout << "  cost: exchange build "
            << std::chrono::duration<double>(t2 - e0).count()
            << " s (63 diagonals, level " << exchange_level
            << ", on ci16_35 above the zone), converter builds "
            << std::chrono::duration<double>(t1 - t0).count()
            << " s (both, forward-only), forward "
            << fwd_seconds / std::max(fwd_count, 1) << " s/ct over "
            << fwd_count << " cts" << std::endl;

  EXPECT_LT(worst, 5e-2)
      << "the dissolved doorstep did not land the scores at the primary "
         "parts";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(norope, 3e-2);
}

// The real HalfBoot delivers the banded images (Doing.md 1.5bz).
//
// 1.5by measured the transport from the banded ModPack image to the chain
// with the coefficient -> slot edge STOOD IN by a host encode of the
// bit-reversed image (legitimate -- 1.5bh measured that edge exactly --
// but composed, not run). This test runs it: the half-density banded
// coefficient images are encrypted at level 0, exactly where the PC-MM
// leaves its output, and cross into slots through ci16_35's own HalfBoot
// -- ModRaise -> CtS -> EvalMod, landing AT dec 19 (full Boot's StC is
// what lands lower; 1.5bu) -- before the 1.5by pipeline takes over.
//
// THE SCALE CONTRACT AT THE HALFBOOT BOUNDARY. HalfBoot's output decodes
// to a CONSTANT multiple of the input coefficients -- 1.5bh measured the
// map as a permutation times one constant, and the constant is the
// message ratio 2^-log_message_ratio (~1/32) times the rungs' drift. The
// ordinary leg restores those five bits inside the SinC prefix's folded
// Canonicalise multiply; here the SAME fold rides RoPE's masks (one
// plaintext multiply at the landing level, already being spent), so the
// factor never reaches the products. The constant is measured off ONE
// decrypted ciphertext, asserted flat across entries (< 2% spread) and
// within half a bit of 1/32, then folded -- a parameter calibration, not
// a data-dependent step.
//
// Everything downstream is 1.5by verbatim, three levels higher: RoPE +
// kill @19 (one mask set, duplicates die), merge @18 (one +128
// rotation), the 63-diagonal exchange @18 (rebuilt at this level, same
// window convention), K's cross @17, LevelDown to the trio, premapped
// descent @3, two chain calls, parts against host RoPE'd scores. What
// this buys: the projection-to-scores path now runs with NO stand-ins
// between the PC-MM's own output encoding and the product -- and the 16
// HalfBoots per tensor that half-density costs are timed on the real
// ladder.
TEST(CiBootSet, TheRealHalfBootDeliversTheBandedImages) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int land_level = 19;      // dec: HalfBoot's landing (1.5bu)
  const int exchange_level = 18;  // RoPE spends 19 -> 18
  const int cross_level = 17;     // the exchange spends 18 -> 17
  const int conv_level = 3;
  const int chain_level = 2;
  const int sub_degree = 32;
  const int n = boot.Degree();

  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  const int half = layout.contraction;  // 64

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  auto exch = [](int s) {
    const int a = (s >> 7) & 31, b = s & 31;
    return (s & ~((31 << 7) | 31)) | (b << 7) | a;
  };
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };
  auto door1 = [&](int t, int c, int i) { return exch(door0(t, c, i)); };
  auto door1k = [&](int t, int c, int i) {
    return (door1(t, c, i) & ~(7 << 7)) | (((c / 16) % 4) << 7);
  };

  // Premaps by enumeration, as in 1.5by.
  const int num_blocks = n / sub_degree;
  std::vector<int> pre_q(num_blocks, -1), pre_k(num_blocks, -1);
  {
    int bad = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int c = 0; c < layout.dim; c++) {
        int ct_idx, slot, copy_slot;
        layout.LocateSlot(t, c, 0, ct_idx, slot, copy_slot);
        if (ct_idx != c / 16) bad++;
        int db = door1(t, c, 0) >> 5;
        if (pre_q[db] == -1) pre_q[db] = slot >> 5;
        if (pre_q[db] != (slot >> 5)) bad++;
        layout.LocateSlot(c % half, t, 0, ct_idx, slot, copy_slot);
        if (ct_idx != t / 16) bad++;
        db = door1k(t, c, 0) >> 5;
        if (pre_k[db] == -1) pre_k[db] = slot >> 5;
        if (pre_k[db] != (slot >> 5)) bad++;
      }
    }
    ASSERT_EQ(bad, 0);
    std::vector<char> used(num_blocks, 0);
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] != -1) used[pre_k[b]] = 1;
    }
    std::vector<int> free_out;
    for (int b = 0; b < num_blocks; b++) {
      if (!used[b]) free_out.push_back(b);
    }
    size_t fo = 0;
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] == -1) pre_k[b] = free_out[fo++];
    }
    ASSERT_EQ(fo, free_out.size());
  }

  // ---- keys, converters, the exchange at the landing rungs -------------
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv_q(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_q);
  CiSinCConverter<word> conv_k(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_k);
  EvkRequest req;
  conv_q.AddRequiredRotations(req);
  conv_k.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const int window = 31 * 127;
  cheddar::StripedMatrix em(n, n);
  for (int r = 0; r < n; r++) {
    const int in = exch(r);
    const int off = ((in - r) % n + n) % n;
    em.try_emplace(off, n, Complex(0.0, 0.0));
    em[off][r] = Complex(1.0, 0.0);
  }
  ASSERT_EQ(em.GetNumDiag(), 63);
  cheddar::LinearTransform<word> lt_exch(
      boot.context, em, exchange_level,
      boot.param->GetRescalePrimeProd(exchange_level), 8, 8,
      /*pre_rotation=*/-window, /*additional_pt_rot=*/window);
  const int window_back = n - window;
  {
    EvkRequest ereq;
    lt_exch.AddRequiredRotations(ereq);
    ereq.AddRequest(window_back, exchange_level - 1);
    boot.ui->PrepareRotationKey(ereq);
  }
  const int merge_idx = n - 128;
  boot.ui->PrepareRotationKey(merge_idx, exchange_level);
  {
    std::set<int> idxs;
    for (int u = 0; u < 4; u++) {
      for (int v = 0; v < 8; v++) {
        const int rot = (v - u) * 128;
        if (rot != 0) idxs.insert((rot % n + n) % n);
      }
    }
    for (int idx : idxs) boot.ui->PrepareRotationKey(idx, cross_level);
  }

  // The bootstrap itself, prepared exactly as Bootstrapping.cpp does.
  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest boot_req;
    bctx->AddRequiredRotations(boot_req, num_slots);
    boot.ui->PrepareRotationKey(boot_req);
  }

  // ---- data ------------------------------------------------------------
  const RealBatch q = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0xB111);
  const RealBatch kk = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                   layout.dim, 0.08, 0xB112);
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  // The banded half-density coefficient image of family fam of channel
  // group l -- what the PC-MM's ModPack emits -- encrypted at level 0 and
  // taken through the real HalfBoot.
  double halfboot_seconds = 0.0;
  int halfboot_count = 0;
  auto boot_half = [&](const RealBatch &m, int l, int fam,
                       Ciphertext<word> &out) {
    std::vector<std::vector<double>> comp(
        512, std::vector<double>(layout.dim, 0.0));
    for (int hh = 0; hh < 16; hh++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int p = 0; p < layout.dim; p++) {
          comp[hh * 16 + cp][p] = m[fam * 16 + hh][p][l * 16 + cp];
        }
      }
    }
    const auto rec = HostRecompose(comp, 512, layout.dim);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, 0, boot.param->GetScale(0), rec);
    Ciphertext<word> enc;
    boot.ui->Encrypt(enc, pt);
    enc.SetNumSlots(num_slots);
    const auto h0 = std::chrono::steady_clock::now();
    bctx->HalfBoot(out, enc, boot.ui->GetEvkMap());
    cudaDeviceSynchronize();
    halfboot_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - h0)
            .count();
    halfboot_count++;
    ASSERT_EQ(boot.param->NPToLevel(out.GetNP()), land_level)
        << "HalfBoot did not land at dec";
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8);
  for (int l = 0; l < 8; l++) {
    boot_half(q, l, 0, q_a[l]);
    boot_half(q, l, 1, q_b[l]);
    boot_half(kk, l, 0, k_a[l]);
    boot_half(kk, l, 1, k_b[l]);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // ---- the boundary constant, measured off one ciphertext --------------
  // Decode-by-declared returns constant * coefficient; the constant is the
  // message ratio times the rungs' drift, flat across entries.
  double hb_const = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, k_a[0]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    double rlo = 1e300, rhi = -1e300, rsum = 0.0;
    int counted = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const double want = kk[hh][t][cp];
          if (std::abs(want) < 0.02) continue;
          const double r = slots[door0(t, cp, hh)].real() / want;
          rlo = std::min(rlo, r);
          rhi = std::max(rhi, r);
          rsum += r;
          counted++;
        }
      }
    }
    hb_const = rsum / counted;
    std::cout << "  HalfBoot boundary constant " << hb_const << " (2^"
              << std::log2(std::abs(hb_const)) << ", want ~2^-5) over "
              << counted << " entries, spread [" << rlo << ", " << rhi << "]"
              << std::endl;
    ASSERT_LT((rhi - rlo) / std::abs(hb_const), 2e-2)
        << "the landing is not a permutation times one constant";
    ASSERT_LT(std::abs(std::log2(std::abs(hb_const) * 32.0)), 0.5)
        << "the constant is not the message ratio";
  }

  // ---- RoPE + kill, with the boundary constant folded into the masks ---
  const double restore = 1.0 / hb_const;
  auto rope_and_kill = [&](std::vector<Ciphertext<word>> &a_cts,
                           std::vector<Ciphertext<word>> &b_cts) {
    const double pt_scale = boot.param->GetRescalePrimeProd(land_level);
    for (int lo = 0; lo < 4; lo++) {
      std::vector<Complex> cm(n, Complex(0.0, 0.0));
      std::vector<Complex> sm(n, Complex(0.0, 0.0));
      std::vector<Complex> nm(n, Complex(0.0, 0.0));
      for (int t = 0; t < layout.dim; t++) {
        for (int cp = 0; cp < 16; cp++) {
          const double ang = t * theta[lo * 16 + cp];
          for (int hh = 0; hh < 16; hh++) {
            const int slot = door0(t, lo * 16 + cp, hh);
            cm[slot] = Complex(std::cos(ang) * restore, 0.0);
            sm[slot] = Complex(std::sin(ang) * restore, 0.0);
            nm[slot] = Complex(-std::sin(ang) * restore, 0.0);
          }
        }
      }
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      boot.context->encoder_.Encode(cos_pt, land_level, pt_scale, cm);
      boot.context->encoder_.Encode(sin_pt, land_level, pt_scale, sm);
      boot.context->encoder_.Encode(neg_sin_pt, land_level, pt_scale, nm);
      for (int fam = 0; fam < 2; fam++) {
        std::vector<Ciphertext<word>> &cts = (fam == 0) ? a_cts : b_cts;
        Ciphertext<word> &lo_ct = cts[lo];
        Ciphertext<word> &hi_ct = cts[lo + 4];
        Ciphertext<word> aa, bb, dd;
        boot.context->Mult(aa, lo_ct, cos_pt);
        boot.context->Mult(bb, hi_ct, neg_sin_pt);
        boot.context->Add(aa, aa, bb);
        boot.context->Mult(bb, hi_ct, cos_pt);
        boot.context->Mult(dd, lo_ct, sin_pt);
        boot.context->Add(bb, bb, dd);
        boot.context->Rescale(lo_ct, aa);
        boot.context->Rescale(hi_ct, bb);
      }
    }
  };
  auto merge = [&](std::vector<Ciphertext<word>> &a_cts,
                   std::vector<Ciphertext<word>> &b_cts) {
    for (int l = 0; l < layout.num_cts; l++) {
      Ciphertext<word> moved;
      boot.context->HRot(moved, b_cts[l],
                         boot.ui->GetEvkMap().GetRotationKey(merge_idx),
                         merge_idx);
      boot.context->Add(a_cts[l], a_cts[l], moved);
    }
  };
  auto exchange_all = [&](std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      ASSERT_EQ(boot.param->NPToLevel(ct.GetNP()), exchange_level);
      Ciphertext<word> shifted, swapped;
      lt_exch.Evaluate(boot.context, shifted, ct, boot.ui->GetEvkMap());
      boot.context->HRot(swapped, shifted,
                         boot.ui->GetEvkMap().GetRotationKey(window_back),
                         window_back);
      ct = std::move(swapped);
    }
  };

  rope_and_kill(q_a, q_b);
  rope_and_kill(k_a, k_b);
  merge(q_a, q_b);
  merge(k_a, k_b);
  exchange_all(q_a);
  exchange_all(k_a);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const double cr_pt_scale = boot.param->GetRescalePrimeProd(cross_level);
  std::vector<Plaintext<word>> sel(8);
  for (int v = 0; v < 8; v++) {
    std::vector<Complex> msg(n, Complex(0.0, 0.0));
    for (int s = 0; s < n; s++) {
      if (((s >> 7) & 7) == v) msg[s] = Complex(1.0, 0.0);
    }
    boot.context->encoder_.Encode(sel[v], cross_level, cr_pt_scale, msg);
  }
  auto cross = [&](int call) {
    std::vector<Ciphertext<word>> out(layout.num_cts);
    for (int t_hi = 0; t_hi < layout.num_cts; t_hi++) {
      const int v = rev(t_hi, 3);
      Ciphertext<word> acc;
      bool first = true;
      for (int l = call * 4; l < call * 4 + 4; l++) {
        Ciphertext<word> piece;
        boot.context->Mult(piece, k_a[l], sel[v]);
        const int rot = (v - l % 4) * 128;
        if (rot != 0) {
          const int idx = (rot % n + n) % n;
          Ciphertext<word> moved;
          boot.context->HRot(moved, piece,
                             boot.ui->GetEvkMap().GetRotationKey(idx), idx);
          piece = std::move(moved);
        }
        if (first) {
          acc = std::move(piece);
          first = false;
        } else {
          boot.context->Add(acc, acc, piece);
        }
      }
      boot.context->Rescale(out[t_hi], acc);
    }
    return out;
  };

  auto convert = [&](CiSinCConverter<word> &conv,
                     std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      Ciphertext<word> at_level;
      boot.context->LevelDown(at_level, ct, conv_level);
      Ciphertext<word> sinc;
      conv.SlotToSinC(swtch.context, sinc, at_level, swtch.ui->GetEvkMap());
      ct = std::move(sinc);
    }
  };
  convert(conv_q, q_a);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Ciphertext<word>> res;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> rhs = cross(call);
    convert(conv_k, rhs);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_a[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> part;
    handler.Multiply(part, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  std::vector<std::vector<Complex>> got(layout.dim);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    ASSERT_EQ(boot.param->NPToLevel(res[bi].GetNP()), chain_level - 1);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    boot.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(
          bridge, chain_level - 1,
          small.param->GetScale(chain_level - 1), comp[j]);
      small.context->encoder_.DecodeSinC(got[bi * layout.rank + j], bridge,
                                         sub_degree);
    }
  }

  double worst = 0.0, transposed = 0.0, norope = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0, want_raw = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          want += q_ref[head][row][c] * k_ref[head][column][c];
          want_t += q_ref[head][column][c] * k_ref[head][row][c];
          want_raw += q[head][row][c] * kk[head][column][c];
        }
        int part, index;
        layout.LocatePart(row, column, lane, part, index);
        const double g = got[part][index].real();
        worst = std::max(worst, std::abs(g - want));
        transposed = std::max(transposed, std::abs(g - want_t));
        norope = std::max(norope, std::abs(g - want_raw));
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "the real HalfBoot delivers the banded images (coeffs @0 -> "
            << "HalfBoot -> @" << land_level << " -> RoPE+restore+kill -> "
            << "merge -> exchange @" << exchange_level << " -> [K: cross @"
            << cross_level << "] -> premapped descent @" << conv_level
            << " -> chain x2): products " << worst << std::endl;
  std::cout << "  controls: transposed " << transposed << ", un-RoPE'd "
            << norope << std::endl;
  std::cout << "  cost: HalfBoot "
            << halfboot_seconds / std::max(halfboot_count, 1) << " s/ct over "
            << halfboot_count
            << " cts (the half-density price, correctness lane)" << std::endl;

  EXPECT_LT(worst, 2e-2)
      << "the HalfBoot'd operands did not land the scores at the primary "
         "parts";
  EXPECT_GT(transposed, 3e-2);
  EXPECT_GT(norope, 3e-2);
}

// The real PC-MM emits the half-density images (Doing.md 1.5ca).
//
// 1.5bz ran the real HalfBoot under the transport but still BUILT the
// banded coefficient images on the host. This computes them: one input
// ciphertext X (rank 512: 512 input channels, each a degree-128 token
// polynomial) is ModDecomp'd ONCE on the device, and every projection
// half-ciphertext is a real PC-MM over those parts -- a 512 x 512
// weight matrix whose rows I >= 256 are ZERO, mixed by PcmmHandler and
// recomposed by the 512-key ModPack. Half-density survives the real
// machinery EXACTLY: a zero weight row gives integer-zero mixed
// components, and the key switch of an exact zero is exact zero, so
// every banded partner term is still identically nothing. The weight
// scale is chosen as GetRescalePrimeProd(1) so one Rescale drops the
// packed output to level 0 at canonical scale -- exactly where HalfBoot
// wants it -- and from there 1.5bz runs verbatim: HalfBoot to dec 19,
// the boundary constant measured and folded into RoPE (restore), merge,
// the 63-diagonal exchange, K's cross, the premapped descent, two chain
// calls.
//
// Q and K project from the SAME X with different weights, as in the
// layer; the host reference is the projection computed in the clear
// (q = W_Q X), so from X to the scores there is NO stand-in left: real
// ModDecomp, real PC-MM, real ModPack, real HalfBoot, real transport,
// real chain. Toy width: 512 input channels (one X ciphertext) against
// the layer's 4096 -- the full width is the same machinery summed over
// eight input ciphertexts and adds nothing structural.
TEST(CiBootSet, TheRealPcmmEmitsTheHalfDensityImages) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int pcmm_level = 1;       // the product level; one Rescale to 0
  const int land_level = 19;
  const int exchange_level = 18;
  const int cross_level = 17;
  const int conv_level = 3;
  const int chain_level = 2;
  const int sub_degree = 32;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;  // 128 = T
  const int in_ch = proj_rank;           // toy width: one X ciphertext

  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  const int half = layout.contraction;  // 64

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  auto exch = [](int s) {
    const int a = (s >> 7) & 31, b = s & 31;
    return (s & ~((31 << 7) | 31)) | (b << 7) | a;
  };
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };
  auto door1 = [&](int t, int c, int i) { return exch(door0(t, c, i)); };
  auto door1k = [&](int t, int c, int i) {
    return (door1(t, c, i) & ~(7 << 7)) | (((c / 16) % 4) << 7);
  };

  const int num_blocks = n / sub_degree;
  std::vector<int> pre_q(num_blocks, -1), pre_k(num_blocks, -1);
  {
    int bad = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int c = 0; c < layout.dim; c++) {
        int ct_idx, slot, copy_slot;
        layout.LocateSlot(t, c, 0, ct_idx, slot, copy_slot);
        if (ct_idx != c / 16) bad++;
        int db = door1(t, c, 0) >> 5;
        if (pre_q[db] == -1) pre_q[db] = slot >> 5;
        if (pre_q[db] != (slot >> 5)) bad++;
        layout.LocateSlot(c % half, t, 0, ct_idx, slot, copy_slot);
        if (ct_idx != t / 16) bad++;
        db = door1k(t, c, 0) >> 5;
        if (pre_k[db] == -1) pre_k[db] = slot >> 5;
        if (pre_k[db] != (slot >> 5)) bad++;
      }
    }
    ASSERT_EQ(bad, 0);
    std::vector<char> used(num_blocks, 0);
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] != -1) used[pre_k[b]] = 1;
    }
    std::vector<int> free_out;
    for (int b = 0; b < num_blocks; b++) {
      if (!used[b]) free_out.push_back(b);
    }
    size_t fo = 0;
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] == -1) pre_k[b] = free_out[fo++];
    }
    ASSERT_EQ(fo, free_out.size());
  }

  // ---- keys, converters, the exchange --------------------------------
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv_q(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_q);
  CiSinCConverter<word> conv_k(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_k);
  EvkRequest req;
  conv_q.AddRequiredRotations(req);
  conv_k.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const int window = 31 * 127;
  cheddar::StripedMatrix em(n, n);
  for (int r = 0; r < n; r++) {
    const int in = exch(r);
    const int off = ((in - r) % n + n) % n;
    em.try_emplace(off, n, Complex(0.0, 0.0));
    em[off][r] = Complex(1.0, 0.0);
  }
  ASSERT_EQ(em.GetNumDiag(), 63);
  cheddar::LinearTransform<word> lt_exch(
      boot.context, em, exchange_level,
      boot.param->GetRescalePrimeProd(exchange_level), 8, 8,
      /*pre_rotation=*/-window, /*additional_pt_rot=*/window);
  const int window_back = n - window;
  {
    EvkRequest ereq;
    lt_exch.AddRequiredRotations(ereq);
    ereq.AddRequest(window_back, exchange_level - 1);
    boot.ui->PrepareRotationKey(ereq);
  }
  const int merge_idx = n - 128;
  boot.ui->PrepareRotationKey(merge_idx, exchange_level);
  {
    std::set<int> idxs;
    for (int u = 0; u < 4; u++) {
      for (int v = 0; v < 8; v++) {
        const int rot = (v - u) * 128;
        if (rot != 0) idxs.insert((rot % n + n) % n);
      }
    }
    for (int idx : idxs) boot.ui->PrepareRotationKey(idx, cross_level);
  }

  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest boot_req;
    bctx->AddRequiredRotations(boot_req, num_slots);
    boot.ui->PrepareRotationKey(boot_req);
  }

  // ---- the projections: X, weights, and the real PC-MM ----------------
  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);
  boot.ui->PrepareModPackKeys(proj_small, pcmm_level);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < proj_rank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
  }

  // X: 512 input channels, each a token polynomial; the residual's own
  // provenance is upstream of the leg and out of scope here.
  std::mt19937_64 gen(0xB121);
  std::uniform_real_distribution<double> xd(-1.0, 1.0);
  std::vector<std::vector<double>> x_comp(
      in_ch, std::vector<double>(proj_small, 0.0));
  for (auto &ch : x_comp) {
    for (auto &v : ch) v = xd(gen);
  }
  // Weights sized so the projected values stay ~0.08.
  const double wa = 0.24 / std::sqrt(static_cast<double>(in_ch));
  std::uniform_real_distribution<double> wd(-wa, wa);
  // wq/wk[head][channel][o]
  std::vector<std::vector<std::vector<double>>> wq(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(in_ch, 0.0)));
  auto wk = wq;
  for (int i = 0; i < layout.lanes; i++) {
    for (int c = 0; c < layout.dim; c++) {
      for (int o = 0; o < in_ch; o++) {
        wq[i][c][o] = wd(gen);
        wk[i][c][o] = wd(gen);
      }
    }
  }
  // The projections in the clear -- the reference the ciphertext path
  // must reproduce.
  auto project = [&](const std::vector<std::vector<std::vector<double>>> &w) {
    RealBatch r(layout.lanes,
                std::vector<std::vector<double>>(
                    layout.dim, std::vector<double>(layout.dim, 0.0)));
    for (int i = 0; i < layout.lanes; i++) {
      for (int t = 0; t < layout.dim; t++) {
        for (int c = 0; c < layout.dim; c++) {
          double s = 0.0;
          for (int o = 0; o < in_ch; o++) s += w[i][c][o] * x_comp[o][t];
          r[i][t][c] = s;
        }
      }
    }
    return r;
  };
  const RealBatch q = project(wq);
  const RealBatch kk = project(wk);

  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  // Encrypt X as the banded recomposition of its channel components and
  // decompose it ONCE; both tensors' projections mix the same parts.
  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  {
    const auto x_rec = HostRecompose(x_comp, proj_rank, proj_small);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                       boot.param->GetScale(pcmm_level),
                                       x_rec);
    Ciphertext<word> x_ct;
    boot.ui->Encrypt(x_ct, pt);
    mlwe.ModDecomp(x_parts, x_ct, proj_small);
    ASSERT_EQ(static_cast<int>(x_parts.size()), proj_rank);
  }

  // One projection half-ciphertext: rows I = h*16 + cp of the 512 x 512
  // matrix carry W[fam*16 + h][l*16 + cp][:]; rows I >= 256 are zero and
  // stay exactly zero through Multiply and ModPack. The weight scale is
  // the level's rescale prime product, so one Rescale lands the packed
  // output at level 0, canonical scale, ready for HalfBoot.
  const double w_scale = boot.param->GetRescalePrimeProd(pcmm_level);
  double pcmm_seconds = 0.0, halfboot_seconds = 0.0;
  int emit_count = 0;
  auto emit_half = [&](const std::vector<std::vector<std::vector<double>>> &w,
                       int l, int fam, Ciphertext<word> &out) {
    std::vector<double> vals(static_cast<size_t>(proj_rank) * in_ch, 0.0);
    for (int hh = 0; hh < 16; hh++) {
      for (int cp = 0; cp < 16; cp++) {
        const int row = hh * 16 + cp;
        for (int o = 0; o < in_ch; o++) {
          vals[static_cast<size_t>(row) * in_ch + o] =
              w[fam * 16 + hh][l * 16 + cp][o];
        }
      }
    }
    const auto p0 = std::chrono::steady_clock::now();
    cheddar::PlainMatrix<word> u;
    pcmm.EncodeMatrix(u, pcmm_level, w_scale, vals, proj_rank, in_ch);
    std::vector<cheddar::MlweCiphertext<word>> mixed;
    pcmm.Multiply(mixed, u, x_parts);
    ASSERT_EQ(static_cast<int>(mixed.size()), proj_rank);
    Ciphertext<word> packed, dropped;
    mlwe.ModPack(boot.context, packed, mixed, pack_keys);
    ASSERT_EQ(boot.param->NPToLevel(packed.GetNP()), pcmm_level);
    boot.context->Rescale(dropped, packed);
    ASSERT_EQ(boot.param->NPToLevel(dropped.GetNP()), 0);
    cudaDeviceSynchronize();
    pcmm_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - p0)
            .count();
    dropped.SetNumSlots(num_slots);
    const auto h0 = std::chrono::steady_clock::now();
    bctx->HalfBoot(out, dropped, boot.ui->GetEvkMap());
    cudaDeviceSynchronize();
    halfboot_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - h0)
            .count();
    emit_count++;
    ASSERT_EQ(boot.param->NPToLevel(out.GetNP()), land_level);
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8);
  for (int l = 0; l < 8; l++) {
    emit_half(wq, l, 0, q_a[l]);
    emit_half(wq, l, 1, q_b[l]);
    emit_half(wk, l, 0, k_a[l]);
    emit_half(wk, l, 1, k_b[l]);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // The boundary constant, measured off one ciphertext as in 1.5bz --
  // now over PC-MM-computed values.
  double hb_const = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, k_a[0]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    double rlo = 1e300, rhi = -1e300, rsum = 0.0;
    int counted = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const double want = kk[hh][t][cp];
          if (std::abs(want) < 0.02) continue;
          const double r = slots[door0(t, cp, hh)].real() / want;
          rlo = std::min(rlo, r);
          rhi = std::max(rhi, r);
          rsum += r;
          counted++;
        }
      }
    }
    hb_const = rsum / counted;
    std::cout << "  boundary constant over the PC-MM'd values " << hb_const
              << " (2^" << std::log2(std::abs(hb_const)) << "), spread ["
              << rlo << ", " << rhi << "] over " << counted << std::endl;
    ASSERT_LT((rhi - rlo) / std::abs(hb_const), 5e-2);
    ASSERT_LT(std::abs(std::log2(std::abs(hb_const) * 32.0)), 0.5);
  }

  const double restore = 1.0 / hb_const;
  auto rope_and_kill = [&](std::vector<Ciphertext<word>> &a_cts,
                           std::vector<Ciphertext<word>> &b_cts) {
    const double pt_scale = boot.param->GetRescalePrimeProd(land_level);
    for (int lo = 0; lo < 4; lo++) {
      std::vector<Complex> cm(n, Complex(0.0, 0.0));
      std::vector<Complex> sm(n, Complex(0.0, 0.0));
      std::vector<Complex> nm(n, Complex(0.0, 0.0));
      for (int t = 0; t < layout.dim; t++) {
        for (int cp = 0; cp < 16; cp++) {
          const double ang = t * theta[lo * 16 + cp];
          for (int hh = 0; hh < 16; hh++) {
            const int slot = door0(t, lo * 16 + cp, hh);
            cm[slot] = Complex(std::cos(ang) * restore, 0.0);
            sm[slot] = Complex(std::sin(ang) * restore, 0.0);
            nm[slot] = Complex(-std::sin(ang) * restore, 0.0);
          }
        }
      }
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      boot.context->encoder_.Encode(cos_pt, land_level, pt_scale, cm);
      boot.context->encoder_.Encode(sin_pt, land_level, pt_scale, sm);
      boot.context->encoder_.Encode(neg_sin_pt, land_level, pt_scale, nm);
      for (int fam = 0; fam < 2; fam++) {
        std::vector<Ciphertext<word>> &cts = (fam == 0) ? a_cts : b_cts;
        Ciphertext<word> &lo_ct = cts[lo];
        Ciphertext<word> &hi_ct = cts[lo + 4];
        Ciphertext<word> aa, bb, dd;
        boot.context->Mult(aa, lo_ct, cos_pt);
        boot.context->Mult(bb, hi_ct, neg_sin_pt);
        boot.context->Add(aa, aa, bb);
        boot.context->Mult(bb, hi_ct, cos_pt);
        boot.context->Mult(dd, lo_ct, sin_pt);
        boot.context->Add(bb, bb, dd);
        boot.context->Rescale(lo_ct, aa);
        boot.context->Rescale(hi_ct, bb);
      }
    }
  };
  auto merge = [&](std::vector<Ciphertext<word>> &a_cts,
                   std::vector<Ciphertext<word>> &b_cts) {
    for (int l = 0; l < layout.num_cts; l++) {
      Ciphertext<word> moved;
      boot.context->HRot(moved, b_cts[l],
                         boot.ui->GetEvkMap().GetRotationKey(merge_idx),
                         merge_idx);
      boot.context->Add(a_cts[l], a_cts[l], moved);
    }
  };
  auto exchange_all = [&](std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      ASSERT_EQ(boot.param->NPToLevel(ct.GetNP()), exchange_level);
      Ciphertext<word> shifted, swapped;
      lt_exch.Evaluate(boot.context, shifted, ct, boot.ui->GetEvkMap());
      boot.context->HRot(swapped, shifted,
                         boot.ui->GetEvkMap().GetRotationKey(window_back),
                         window_back);
      ct = std::move(swapped);
    }
  };

  rope_and_kill(q_a, q_b);
  rope_and_kill(k_a, k_b);
  merge(q_a, q_b);
  merge(k_a, k_b);
  exchange_all(q_a);
  exchange_all(k_a);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const double cr_pt_scale = boot.param->GetRescalePrimeProd(cross_level);
  std::vector<Plaintext<word>> sel(8);
  for (int v = 0; v < 8; v++) {
    std::vector<Complex> msg(n, Complex(0.0, 0.0));
    for (int s = 0; s < n; s++) {
      if (((s >> 7) & 7) == v) msg[s] = Complex(1.0, 0.0);
    }
    boot.context->encoder_.Encode(sel[v], cross_level, cr_pt_scale, msg);
  }
  auto cross = [&](int call) {
    std::vector<Ciphertext<word>> out(layout.num_cts);
    for (int t_hi = 0; t_hi < layout.num_cts; t_hi++) {
      const int v = rev(t_hi, 3);
      Ciphertext<word> acc;
      bool first = true;
      for (int l = call * 4; l < call * 4 + 4; l++) {
        Ciphertext<word> piece;
        boot.context->Mult(piece, k_a[l], sel[v]);
        const int rot = (v - l % 4) * 128;
        if (rot != 0) {
          const int idx = (rot % n + n) % n;
          Ciphertext<word> moved;
          boot.context->HRot(moved, piece,
                             boot.ui->GetEvkMap().GetRotationKey(idx), idx);
          piece = std::move(moved);
        }
        if (first) {
          acc = std::move(piece);
          first = false;
        } else {
          boot.context->Add(acc, acc, piece);
        }
      }
      boot.context->Rescale(out[t_hi], acc);
    }
    return out;
  };

  auto convert = [&](CiSinCConverter<word> &conv,
                     std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      Ciphertext<word> at_level;
      boot.context->LevelDown(at_level, ct, conv_level);
      Ciphertext<word> sinc;
      conv.SlotToSinC(swtch.context, sinc, at_level, swtch.ui->GetEvkMap());
      ct = std::move(sinc);
    }
  };
  convert(conv_q, q_a);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Ciphertext<word>> res;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> rhs = cross(call);
    convert(conv_k, rhs);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_a[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> part;
    handler.Multiply(part, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  std::vector<std::vector<Complex>> got(layout.dim);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    ASSERT_EQ(boot.param->NPToLevel(res[bi].GetNP()), chain_level - 1);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    boot.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(
          bridge, chain_level - 1,
          small.param->GetScale(chain_level - 1), comp[j]);
      small.context->encoder_.DecodeSinC(got[bi * layout.rank + j], bridge,
                                         sub_degree);
    }
  }

  double worst = 0.0, transposed = 0.0, norope = 0.0, biggest = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0, want_raw = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          want += q_ref[head][row][c] * k_ref[head][column][c];
          want_t += q_ref[head][column][c] * k_ref[head][row][c];
          want_raw += q[head][row][c] * kk[head][column][c];
        }
        biggest = std::max(biggest, std::abs(want));
        int part, index;
        layout.LocatePart(row, column, lane, part, index);
        const double g = got[part][index].real();
        worst = std::max(worst, std::abs(g - want));
        transposed = std::max(transposed, std::abs(g - want_t));
        norope = std::max(norope, std::abs(g - want_raw));
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "the real PC-MM emits the half-density images (X -> "
            << "ModDecomp -> PC-MM x32 -> ModPack -> @0 -> HalfBoot -> @"
            << land_level << " -> transport -> descent @" << conv_level
            << " -> chain x2): products " << worst << " (|scores| <= "
            << biggest << ")" << std::endl;
  std::cout << "  controls: transposed " << transposed << ", un-RoPE'd "
            << norope << std::endl;
  std::cout << "  cost: PC-MM emit (encode+mix+pack+rescale) "
            << pcmm_seconds / std::max(emit_count, 1) << " s/ct, HalfBoot "
            << halfboot_seconds / std::max(emit_count, 1) << " s/ct over "
            << emit_count << " cts (toy width " << in_ch
            << ", correctness lane)" << std::endl;

  EXPECT_LT(worst, 2e-2)
      << "the PC-MM'd projections did not land the scores at the primary "
         "parts";
  EXPECT_GT(transposed, 1e-2);
  EXPECT_GT(norope, 1e-2)
      << "RoPE did nothing the product can see";
}

// The attention leg closes end to end (Doing.md 1.5cb).
//
// Every piece is measured; this runs them as ONE leg, at the Llama
// alignment, from the residual to the attention output:
//
//   X --ModDecomp--> parts --PC-MM x48--> Q,K,V half-images @0
//     --HalfBoot--> @19 --RoPE+restore+kill / restore+kill--> merge
//     --exchange @18--> [K: cross @17] [V: per-call align @17]
//     --premapped descents @3--> scores chain x2 --> @1
//     --SinCToSlot--> @0 --Boot--> @16 --softmax--> P @3
//     --nested descent--> P x V chain x2 --> @1 --SinCToSlot--> @0
//
// Three things are new against the pieces:
//
// 1. V RIDES Q'S CONVERTER. V's chain role is (row = key token, column
//    = channel), and its post-exchange block is the SAME function of
//    (token, channel) as Q's -- so pre_q takes V home too. The per-call
//    key split needs only a 0/1 mask on the call bit (door1's slot bit
//    7 = the token's top bit) and, for the odd call, ONE rotation by
//    128 (rev7(x) = rev7(t) - call: clearing the call bit IS the row
//    relabelling). V has no angles, so its kill mask is the restore
//    constant alone.
//
// 2. THE SOFTMAX REDUCTION AT SUB 32 SPANS THE CIPHERTEXTS. The key
//    axis is 16 classes (top four slot bits, the same stride-4096 tree
//    as sub 128) times EIGHT ciphertexts, so the Euclidean norm is
//    eight squared ciphertexts summed, then the four-rotation tree --
//    still no mask, still CtS-held indices.
//
// 3. THE FIVE-BIT HEADROOM AND THE BOOT CONTRACT COMPETE, AND THE
//    TRANSPORT ARBITRATES. HalfBoot's declared scale is EvalMod's end
//    scale (~2^58 on these rungs), and 1.5bz's restore fold rode it
//    undiminished through the descent and the chain -- which is where
//    the five bits of floor came from, and which the scores-only tests
//    could afford because they never left level 1. The Boot boundary
//    cannot: carried = declared / base must satisfy carried * |S| < 1,
//    so the operands must arrive CANONICAL -- the ordinary leg's
//    Canonicalise, whose CI home is the transport's own multiplies. A
//    factor gamma = sqrt(canonical / declared) folds into BOTH the
//    RoPE/restore masks and the exchange's plaintexts (integers stay
//    ~2^28, no precision cliff; a single-multiply fold would be a
//    2^-12 cliff), landing Q at canonical @17 and K/V beside it. The
//    first run of this test tripped the guard at carried ~ 2^47 and is
//    why the fold exists; the guard stays.
//
// The host reference is the whole leg in the clear: projections from
// the same X and weights, RoPE, true softmax, times V. Controls: the
// transposed P read and the single-call half sum. Causal mask and
// real-data calibration stay open (data, not mechanism); this is the
// leg the assembly will wrap.
TEST(CiBootSet, TheAttentionLegClosesEndToEnd) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int pcmm_level = 1;
  const int land_level = 19;
  const int exchange_level = 18;
  const int cross_level = 17;
  const int conv_level = 3;
  const int chain_level = 2;
  const int sub_degree = 32;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;
  const int in_ch = proj_rank;

  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 32);
  ASSERT_EQ(layout.num_cts, 8);
  const int half = layout.contraction;  // 64

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  auto exch = [](int s) {
    const int a = (s >> 7) & 31, b = s & 31;
    return (s & ~((31 << 7) | 31)) | (b << 7) | a;
  };
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };
  auto door1 = [&](int t, int c, int i) { return exch(door0(t, c, i)); };
  auto door1k = [&](int t, int c, int i) {
    return (door1(t, c, i) & ~(7 << 7)) | (((c / 16) % 4) << 7);
  };

  const int num_blocks = n / sub_degree;
  std::vector<int> pre_q(num_blocks, -1), pre_k(num_blocks, -1);
  {
    int bad = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int c = 0; c < layout.dim; c++) {
        int ct_idx, slot, copy_slot;
        layout.LocateSlot(t, c, 0, ct_idx, slot, copy_slot);
        if (ct_idx != c / 16) bad++;
        int db = door1(t, c, 0) >> 5;
        if (pre_q[db] == -1) pre_q[db] = slot >> 5;
        if (pre_q[db] != (slot >> 5)) bad++;
        layout.LocateSlot(c % half, t, 0, ct_idx, slot, copy_slot);
        if (ct_idx != t / 16) bad++;
        db = door1k(t, c, 0) >> 5;
        if (pre_k[db] == -1) pre_k[db] = slot >> 5;
        if (pre_k[db] != (slot >> 5)) bad++;
      }
    }
    ASSERT_EQ(bad, 0);
    std::vector<char> used(num_blocks, 0);
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] != -1) used[pre_k[b]] = 1;
    }
    std::vector<int> free_out;
    for (int b = 0; b < num_blocks; b++) {
      if (!used[b]) free_out.push_back(b);
    }
    size_t fo = 0;
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] == -1) pre_k[b] = free_out[fo++];
    }
    ASSERT_EQ(fo, free_out.size());
  }

  // ---- keys and the three converters ----------------------------------
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> conv_q(swtch.context, sub_degree, conv_level,
                               /*inverse_level=*/-1, &layout, &pre_q);
  CiSinCConverter<word> conv_k(swtch.context, sub_degree, conv_level,
                               /*inverse_level=*/-1, &layout, &pre_k);
  // The plain nested pair: P's own descent, and both returns to slots.
  CiSinCConverter<word> conv_pv(swtch.context, sub_degree, conv_level,
                                /*inverse_level=*/1, &layout);
  const auto t1 = std::chrono::steady_clock::now();
  EvkRequest req;
  conv_q.AddRequiredRotations(req);
  conv_k.AddRequiredRotations(req);
  conv_pv.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const int window = 31 * 127;
  cheddar::StripedMatrix em(n, n);
  for (int r = 0; r < n; r++) {
    const int in = exch(r);
    const int off = ((in - r) % n + n) % n;
    em.try_emplace(off, n, Complex(0.0, 0.0));
    em[off][r] = Complex(1.0, 0.0);
  }
  // The canonicalising fold of comment point 3. GetStCInputScale --
  // HalfBoot's declared output scale -- is only known once EvalMod is
  // prepared, so the boot set comes up before the exchange compiles.
  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest boot_req;
    bctx->AddRequiredRotations(boot_req, num_slots);
    boot.ui->PrepareRotationKey(boot_req);
  }
  const double gamma = std::sqrt(boot.param->GetScale(cross_level) /
                                 bctx->GetStCInputScale());
  cheddar::LinearTransform<word> lt_exch(
      boot.context, em, exchange_level,
      boot.param->GetRescalePrimeProd(exchange_level) * gamma, 8, 8,
      /*pre_rotation=*/-window, /*additional_pt_rot=*/window);
  const int window_back = n - window;
  {
    EvkRequest ereq;
    lt_exch.AddRequiredRotations(ereq);
    ereq.AddRequest(window_back, exchange_level - 1);
    boot.ui->PrepareRotationKey(ereq);
  }
  const int merge_idx = n - 128;
  boot.ui->PrepareRotationKey(merge_idx, exchange_level);
  {
    std::set<int> idxs;
    for (int u = 0; u < 4; u++) {
      for (int v = 0; v < 8; v++) {
        const int rot = (v - u) * 128;
        if (rot != 0) idxs.insert((rot % n + n) % n);
      }
    }
    idxs.insert(128);  // V's odd-call alignment
    for (int idx : idxs) boot.ui->PrepareRotationKey(idx, cross_level);
  }

  // ---- the projections, real PC-MM as in 1.5ca ------------------------
  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);
  boot.ui->PrepareModPackKeys(proj_small, pcmm_level);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < proj_rank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
  }

  std::mt19937_64 gen(0xB131);
  std::uniform_real_distribution<double> xd(-1.0, 1.0);
  std::vector<std::vector<double>> x_comp(
      in_ch, std::vector<double>(proj_small, 0.0));
  for (auto &ch : x_comp) {
    for (auto &v : ch) v = xd(gen);
  }
  const double wa = 0.24 / std::sqrt(static_cast<double>(in_ch));
  std::uniform_real_distribution<double> wd(-wa, wa);
  using W = std::vector<std::vector<std::vector<double>>>;
  W wq(layout.lanes, std::vector<std::vector<double>>(
                         layout.dim, std::vector<double>(in_ch, 0.0)));
  W wk = wq, wv = wq;
  for (int i = 0; i < layout.lanes; i++) {
    for (int c = 0; c < layout.dim; c++) {
      for (int o = 0; o < in_ch; o++) {
        wq[i][c][o] = wd(gen);
        wk[i][c][o] = wd(gen);
        wv[i][c][o] = wd(gen);
      }
    }
  }
  auto project = [&](const W &w) {
    RealBatch r(layout.lanes,
                std::vector<std::vector<double>>(
                    layout.dim, std::vector<double>(layout.dim, 0.0)));
    for (int i = 0; i < layout.lanes; i++) {
      for (int t = 0; t < layout.dim; t++) {
        for (int c = 0; c < layout.dim; c++) {
          double s = 0.0;
          for (int o = 0; o < in_ch; o++) s += w[i][c][o] * x_comp[o][t];
          r[i][t][c] = s;
        }
      }
    }
    return r;
  };
  const RealBatch q = project(wq);
  const RealBatch kk = project(wk);
  const RealBatch vv = project(wv);

  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  {
    const auto x_rec = HostRecompose(x_comp, proj_rank, proj_small);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                       boot.param->GetScale(pcmm_level),
                                       x_rec);
    Ciphertext<word> x_ct;
    boot.ui->Encrypt(x_ct, pt);
    mlwe.ModDecomp(x_parts, x_ct, proj_small);
  }

  const double w_scale = boot.param->GetRescalePrimeProd(pcmm_level);
  auto emit_half = [&](const W &w, int l, int fam, Ciphertext<word> &out) {
    std::vector<double> vals(static_cast<size_t>(proj_rank) * in_ch, 0.0);
    for (int hh = 0; hh < 16; hh++) {
      for (int cp = 0; cp < 16; cp++) {
        const int row = hh * 16 + cp;
        for (int o = 0; o < in_ch; o++) {
          vals[static_cast<size_t>(row) * in_ch + o] =
              w[fam * 16 + hh][l * 16 + cp][o];
        }
      }
    }
    cheddar::PlainMatrix<word> u;
    pcmm.EncodeMatrix(u, pcmm_level, w_scale, vals, proj_rank, in_ch);
    std::vector<cheddar::MlweCiphertext<word>> mixed;
    pcmm.Multiply(mixed, u, x_parts);
    Ciphertext<word> packed, dropped;
    mlwe.ModPack(boot.context, packed, mixed, pack_keys);
    boot.context->Rescale(dropped, packed);
    dropped.SetNumSlots(num_slots);
    bctx->HalfBoot(out, dropped, boot.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(out.GetNP()), land_level);
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8), v_a(8),
      v_b(8);
  for (int l = 0; l < 8; l++) {
    emit_half(wq, l, 0, q_a[l]);
    emit_half(wq, l, 1, q_b[l]);
    emit_half(wk, l, 0, k_a[l]);
    emit_half(wk, l, 1, k_b[l]);
    emit_half(wv, l, 0, v_a[l]);
    emit_half(wv, l, 1, v_b[l]);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_NEAR(q_a[0].GetScale() / bctx->GetStCInputScale(), 1.0, 1e-9)
      << "HalfBoot's declared scale is not GetStCInputScale, so gamma "
         "canonicalised against the wrong number";

  double hb_const = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, k_a[0]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    double rsum = 0.0;
    int counted = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const double want = kk[hh][t][cp];
          if (std::abs(want) < 0.02) continue;
          rsum += slots[door0(t, cp, hh)].real() / want;
          counted++;
        }
      }
    }
    hb_const = rsum / counted;
    ASSERT_LT(std::abs(std::log2(std::abs(hb_const) * 32.0)), 0.5);
  }
  const double restore = 1.0 / hb_const;

  // RoPE + restore + kill; with_angles = false is V's plain restore.
  auto rope_and_kill = [&](std::vector<Ciphertext<word>> &a_cts,
                           std::vector<Ciphertext<word>> &b_cts,
                           bool with_angles) {
    const double pt_scale =
        boot.param->GetRescalePrimeProd(land_level) * gamma;
    for (int lo = 0; lo < 4; lo++) {
      std::vector<Complex> cm(n, Complex(0.0, 0.0));
      std::vector<Complex> sm(n, Complex(0.0, 0.0));
      std::vector<Complex> nm(n, Complex(0.0, 0.0));
      for (int t = 0; t < layout.dim; t++) {
        for (int cp = 0; cp < 16; cp++) {
          const double ang =
              with_angles ? t * theta[lo * 16 + cp] : 0.0;
          for (int hh = 0; hh < 16; hh++) {
            const int slot = door0(t, lo * 16 + cp, hh);
            cm[slot] = Complex(std::cos(ang) * restore, 0.0);
            sm[slot] = Complex(std::sin(ang) * restore, 0.0);
            nm[slot] = Complex(-std::sin(ang) * restore, 0.0);
          }
        }
      }
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      boot.context->encoder_.Encode(cos_pt, land_level, pt_scale, cm);
      boot.context->encoder_.Encode(sin_pt, land_level, pt_scale, sm);
      boot.context->encoder_.Encode(neg_sin_pt, land_level, pt_scale, nm);
      for (int fam = 0; fam < 2; fam++) {
        std::vector<Ciphertext<word>> &cts = (fam == 0) ? a_cts : b_cts;
        Ciphertext<word> &lo_ct = cts[lo];
        Ciphertext<word> &hi_ct = cts[lo + 4];
        Ciphertext<word> aa, bb, dd;
        boot.context->Mult(aa, lo_ct, cos_pt);
        boot.context->Mult(bb, hi_ct, neg_sin_pt);
        boot.context->Add(aa, aa, bb);
        boot.context->Mult(bb, hi_ct, cos_pt);
        boot.context->Mult(dd, lo_ct, sin_pt);
        boot.context->Add(bb, bb, dd);
        boot.context->Rescale(lo_ct, aa);
        boot.context->Rescale(hi_ct, bb);
      }
    }
  };
  auto merge = [&](std::vector<Ciphertext<word>> &a_cts,
                   std::vector<Ciphertext<word>> &b_cts) {
    for (int l = 0; l < layout.num_cts; l++) {
      Ciphertext<word> moved;
      boot.context->HRot(moved, b_cts[l],
                         boot.ui->GetEvkMap().GetRotationKey(merge_idx),
                         merge_idx);
      boot.context->Add(a_cts[l], a_cts[l], moved);
    }
  };
  auto exchange_all = [&](std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      Ciphertext<word> shifted, swapped;
      lt_exch.Evaluate(boot.context, shifted, ct, boot.ui->GetEvkMap());
      boot.context->HRot(swapped, shifted,
                         boot.ui->GetEvkMap().GetRotationKey(window_back),
                         window_back);
      ct = std::move(swapped);
    }
  };

  rope_and_kill(q_a, q_b, true);
  rope_and_kill(k_a, k_b, true);
  rope_and_kill(v_a, v_b, false);
  merge(q_a, q_b);
  merge(k_a, k_b);
  merge(v_a, v_b);
  exchange_all(q_a);
  exchange_all(k_a);
  exchange_all(v_a);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // K's cross, as before.
  const double cr_pt_scale = boot.param->GetRescalePrimeProd(cross_level);
  std::vector<Plaintext<word>> sel(8);
  for (int v = 0; v < 8; v++) {
    std::vector<Complex> msg(n, Complex(0.0, 0.0));
    for (int s = 0; s < n; s++) {
      if (((s >> 7) & 7) == v) msg[s] = Complex(1.0, 0.0);
    }
    boot.context->encoder_.Encode(sel[v], cross_level, cr_pt_scale, msg);
  }
  auto cross = [&](int call) {
    std::vector<Ciphertext<word>> out(layout.num_cts);
    for (int t_hi = 0; t_hi < layout.num_cts; t_hi++) {
      const int v = rev(t_hi, 3);
      Ciphertext<word> acc;
      bool first = true;
      for (int l = call * 4; l < call * 4 + 4; l++) {
        Ciphertext<word> piece;
        boot.context->Mult(piece, k_a[l], sel[v]);
        const int rot = (v - l % 4) * 128;
        if (rot != 0) {
          const int idx = (rot % n + n) % n;
          Ciphertext<word> moved;
          boot.context->HRot(moved, piece,
                             boot.ui->GetEvkMap().GetRotationKey(idx), idx);
          piece = std::move(moved);
        }
        if (first) {
          acc = std::move(piece);
          first = false;
        } else {
          boot.context->Add(acc, acc, piece);
        }
      }
      boot.context->Rescale(out[t_hi], acc);
    }
    return out;
  };
  // V's per-call alignment: mask the call bit, odd call shifts by 128.
  std::vector<Plaintext<word>> vsel(2);
  for (int call = 0; call < 2; call++) {
    std::vector<Complex> msg(n, Complex(0.0, 0.0));
    for (int s = 0; s < n; s++) {
      if (((s >> 7) & 1) == call) msg[s] = Complex(1.0, 0.0);
    }
    boot.context->encoder_.Encode(vsel[call], cross_level, cr_pt_scale, msg);
  }
  auto v_call = [&](int call) {
    std::vector<Ciphertext<word>> out(layout.num_cts);
    for (int l = 0; l < layout.num_cts; l++) {
      Ciphertext<word> piece;
      boot.context->Mult(piece, v_a[l], vsel[call]);
      if (call == 1) {
        Ciphertext<word> moved;
        boot.context->HRot(moved, piece,
                           boot.ui->GetEvkMap().GetRotationKey(128), 128);
        piece = std::move(moved);
      }
      boot.context->Rescale(out[l], piece);
    }
    return out;
  };

  auto convert = [&](CiSinCConverter<word> &conv,
                     std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      Ciphertext<word> at_level;
      boot.context->LevelDown(at_level, ct, conv_level);
      Ciphertext<word> sinc;
      conv.SlotToSinC(swtch.context, sinc, at_level, swtch.ui->GetEvkMap());
      ct = std::move(sinc);
    }
  };
  convert(conv_q, q_a);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // ---- scores ----------------------------------------------------------
  std::vector<Ciphertext<word>> res;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> rhs = cross(call);
    convert(conv_k, rhs);
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_a[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> part;
    handler.Multiply(part, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  // ---- host calibration off the leg's own clear twin -------------------
  const double m_eff = 8.0;
  std::vector<std::vector<std::vector<double>>> S(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
  double smin = 1e300, smax = -1e300;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int col = 0; col < layout.dim; col++) {
        double v = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          v += q_ref[head][row][c] * k_ref[head][col][c];
        }
        S[head][row][col] = v;
        smin = std::min(smin, v);
        smax = std::max(smax, v);
      }
    }
  }
  const double span = smax - smin;
  auto u_of = [&](double v) { return 2.0 * (v - smax) / span + 1.0; };

  // ---- scores to slots, Boot, softmax ---------------------------------
  double carried = 0.0;
  std::vector<Ciphertext<word>> scores(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv_pv.SinCToSlot(swtch.context, back, res[bi], swtch.ui->GetEvkMap());
    carried = back.GetScale() / boot.param->base_scale_;
    back.SetNumSlots(num_slots);
    bctx->Boot(scores[bi], back, boot.ui->GetEvkMap());
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int top = boot.param->NPToLevel(scores[0].GetNP());
  ASSERT_EQ(top, bctx->GetBootParameter().GetEndLevel());
  ASSERT_LT(carried * std::max(std::abs(smax), std::abs(smin)), 0.95)
      << "the walk's declared drift pushed the booted message outside "
         "EvalMod's range; fold a constant into restore";

  const double hb = m_eff / 4.0;
  auto exp_coeffs = cheddar::chebfit::Interpolate(
      [hb](double v) { return std::exp(hb * (v - 1.0)); }, 9);
  const int exp_in = top - 1;
  auto log2ceil = [](int x) {
    int r = 0;
    while ((1 << r) < x) r++;
    return r;
  };
  const int exp_used =
      EvalPoly<word>(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                     boot.param->GetScale(exp_in), true)
          .GetPolyDegree();
  const int exp_out = exp_in - log2ceil(exp_used + 1);
  EvalPoly<word> exp_poly(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                          boot.param->GetScale(exp_out), true);
  exp_poly.Compile(boot.context);

  double lo = 1e300, hi = -1e300;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      double sq = 0.0;
      for (int col = 0; col < layout.dim; col++) {
        const double y = std::exp(hb * (u_of(S[head][row][col]) - 1.0));
        sq += y * y;
      }
      lo = std::min(lo, sq);
      hi = std::max(hi, sq);
    }
  }
  lo *= 0.9;
  hi *= 1.1;
  const double aff_a = 0.5 * (hi - lo);
  const double aff_b = 0.5 * (hi + lo);
  auto inv_coeffs = cheddar::chebfit::Interpolate(
      [aff_a, aff_b](double v) { return 1.0 / std::sqrt(aff_a * v + aff_b); },
      15);
  const int sq_level = exp_out - 1;
  const int poly_in = sq_level - 1;
  const int inv_used =
      EvalPoly<word>(inv_coeffs, poly_in, boot.param->GetScale(poly_in),
                     boot.param->GetScale(poly_in), true)
          .GetPolyDegree();
  const int inv_out = poly_in - log2ceil(inv_used + 1);
  EvalPoly<word> inv_poly(inv_coeffs, poly_in,
                          boot.param->GetScale(poly_in),
                          boot.param->GetScale(inv_out), true);
  inv_poly.Compile(boot.context);

  std::vector<int> reduce_dist;
  for (int t = 0; t < 4; t++) {
    reduce_dist.push_back((num_slots / layout.rank) << t);
  }
  {
    EvkRequest rot_req;
    for (int d : reduce_dist) rot_req.AddRequest(d, sq_level);
    boot.ui->PrepareRotationKey(rot_req);
  }
  const auto &evk = boot.ui->GetEvkMap();
  const auto &mult_key = evk.GetMultiplicationKey();

  const double a1 = 2.0 / (span * carried);
  const double a0 = 1.0 - 2.0 * smax / span;
  std::vector<Ciphertext<word>> y(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Constant<word> c1;
    boot.context->encoder_.EncodeConstant(c1, top,
                                          boot.param->GetScale(top), a1);
    Ciphertext<word> t1, u_ct;
    boot.context->Mult(t1, scores[bi], c1);
    boot.context->Rescale(u_ct, t1);
    Constant<word> c0;
    boot.context->encoder_.EncodeConstant(c0, exp_in, u_ct.GetScale(), a0);
    boot.context->Add(u_ct, u_ct, c0);
    exp_poly.Evaluate(boot.context, y[bi], u_ct, mult_key);
  }
  // The norm: eight squared ciphertexts summed, then the top-field tree.
  Ciphertext<word> sq, term, rotated;
  boot.context->HMult(sq, y[0], y[0], mult_key);
  for (int bi = 1; bi < layout.num_cts; bi++) {
    boot.context->HMult(term, y[bi], y[bi], mult_key);
    boot.context->Add(sq, sq, term);
  }
  for (int d : reduce_dist) {
    boot.context->HRotAdd(rotated, sq, sq, evk.GetRotationKey(d), d);
    boot.context->Copy(sq, rotated);
  }
  {
    Constant<word> inv_a;
    boot.context->encoder_.EncodeConstant(
        inv_a, sq_level, boot.param->GetScale(sq_level), 1.0 / aff_a);
    Ciphertext<word> scaled;
    boot.context->Mult(scaled, sq, inv_a);
    boot.context->Rescale(sq, scaled);
    Constant<word> shift;
    boot.context->encoder_.EncodeConstant(shift, poly_in, sq.GetScale(),
                                          -aff_b / aff_a);
    boot.context->Add(sq, sq, shift);
  }
  Ciphertext<word> r;
  inv_poly.Evaluate(boot.context, r, sq, mult_key);
  const int meet = boot.param->NPToLevel(r.GetNP());
  std::vector<Ciphertext<word>> P(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> levelled, prod;
    boot.context->LevelDown(levelled, y[bi], meet);
    boot.context->HMult(prod, levelled, r, mult_key);
    boot.context->HMult(P[bi], prod, prod, mult_key);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(boot.param->NPToLevel(P[0].GetNP()), conv_level)
      << "the softmax did not leave P at fwd_level";

  // What P arrived as, indexed by head for the host contraction.
  std::vector<std::vector<std::vector<double>>> P_dec(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
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
          P_dec[rev(lane, 5)][row][column] = slots[slot].real();
        }
      }
    }
  }

  // ---- P x V through the same chain ------------------------------------
  std::vector<Ciphertext<word>> p_sinc(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    conv_pv.SlotToSinC(swtch.context, p_sinc[bi], P[bi],
                       swtch.ui->GetEvkMap());
  }
  std::vector<Ciphertext<word>> out;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> v_rhs = v_call(call);
    convert(conv_q, v_rhs);  // V rides Q's converter
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    std::vector<Ciphertext<word>> p_lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      p_lhs.push_back(std::move(p_sinc[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> part;
    handler.Multiply(part, p_lhs, v_rhs,
                     swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    if (call == 0) {
      out = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(out[bi], out[bi], part[bi]);
      }
    }
  }

  // ---- the attention output, against the leg in the clear --------------
  double worst_true = 0.0, worst_incoming = 0.0;
  double transposed = 0.0, half_sum = 0.0, biggest = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv_pv.SinCToSlot(swtch.context, back, out[bi], swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(back.GetNP()), 0);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, back);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int lane = 0; lane < layout.lanes; lane++) {
      const int head = rev(lane, 5);
      for (int row = 0; row < layout.dim; row++) {
        std::vector<double> p_true(layout.dim);
        double zsum = 0.0;
        for (int k = 0; k < layout.dim; k++) {
          p_true[k] =
              std::exp(m_eff * (u_of(S[head][row][k]) - 1.0) / 2.0);
          zsum += p_true[k];
        }
        for (double &pv : p_true) pv /= zsum;
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          double want = 0.0, want_in = 0.0, want_t = 0.0, want_half = 0.0;
          for (int k = 0; k < layout.dim; k++) {
            want += p_true[k] * vv[head][k][column];
            want_in += P_dec[head][row][k] * vv[head][k][column];
            want_t += P_dec[head][column][k] * vv[head][k][row];
            if (k < half) {
              want_half += P_dec[head][row][k] * vv[head][k][column];
            }
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          biggest = std::max(biggest, std::abs(want));
          worst_true = std::max(worst_true, std::abs(got - want));
          worst_incoming =
              std::max(worst_incoming, std::abs(got - want_in));
          transposed = std::max(transposed, std::abs(got - want_t));
          half_sum = std::max(half_sum, std::abs(got - want_half));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "the attention leg end to end (X -> PC-MM x48 -> HalfBoot "
            << "-> transport -> scores -> Boot -> softmax -> P x V -> "
            << "slots @0): vs the leg in the clear " << worst_true
            << ", vs decrypted P x V " << worst_incoming << " (|output| <= "
            << biggest << ", carried " << carried << ")" << std::endl;
  std::cout << "  controls: transposed " << transposed
            << ", single-call half sum " << half_sum << std::endl;
  std::cout << "  cost: converter builds "
            << std::chrono::duration<double>(t1 - t0).count()
            << " s (three, one with the inverse)" << std::endl;

  EXPECT_LT(worst_true, 2e-2)
      << "the leg did not land the attention output at the primary "
         "addresses";
  EXPECT_LT(worst_incoming, 1e-2)
      << "the second contraction added more than its own floor";
  EXPECT_GT(transposed, 5e-3);
  EXPECT_GT(half_sum, 5e-3)
      << "one call alone matches, so the two-call sum never happened";
}

// The causal mask folds into the softmax walk (Doing.md 1.5cc).
//
// 1.5bv left two data questions open: the causal mask, and which norm
// interval each row then lands in (a row with few live keys has a small
// norm). Both close together, and the mechanism is one plaintext the
// walk does not have plus one it already does:
//
// 1. THE MASK IS ONE 0/1 PLAINTEXT MULTIPLY ON y = exp(...), and the
//    level it needs comes out of exp's own fit: deg 9 -> deg 7 is a
//    log2ceil of 4 -> 3, and the deg-7 Chebyshev truncation of
//    exp(2(u - 1)) on [-1, 1] is ~7e-6 -- two orders below the 4e-4
//    boot-floor the softmax already carries. Every level downstream is
//    UNCHANGED (sq at 10, invsqrt at 9..5), so the causal walk lands P
//    at fwd_level exactly like the plain one. Masking y rather than P
//    is load-bearing: the Euclidean norm then sums live keys only,
//    which is what makes the masked softmax causal instead of clipped.
//
// 2. THE AFFINE'S SHIFT GOES PER-ROW, AND THAT IS THE INTERVAL ANSWER.
//    a0 was the global constant 1 - 2 smax / span; softmax is
//    shift-invariant within a row, so a0 -- already one Add -- can be a
//    plaintext carrying each (lane, row)'s own live-key max instead
//    (masked slots keep the global shift so their u stays inside the
//    fit domain; the mask kills their y anyway). Then every row's
//    largest live y is exactly 1 and the norm interval is [1, live
//    row sum] BY CONSTRUCTION, against the global shift under which a
//    row whose live keys sit far below smax collapses toward
//    e^-m_eff and the interval spans orders of magnitude (both
//    intervals printed below; the global one is what 1.5bv warned
//    about). What remains data-dependent is exactly what span already
//    was: rowmax and span are calibrated estimates with margins, not
//    exact reads -- here, as in 1.5bv, they are read off this test's
//    own data.
//
// Measured at sub_degree 128 through the real pipeline (encrypt @3 ->
// chain -> slots @0 -> Boot -> @16 -> causal softmax -> P @3 -> chain
// x2 -> slots @0), against the causal softmax times V in the clear.
// Controls: the UN-masked softmax reference must fail, the transposed
// read must fail, P at masked addresses must be numerically zero, and
// live row sums must be one.
TEST(CiBootSet, TheCausalMaskFoldsIntoTheSoftmaxWalk) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int fwd_level = 3, chain_level = 2, inverse_level = 1;
  const int sub_degree = 128;
  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  ASSERT_EQ(layout.num_cts, 2);
  const int half_keys = layout.contraction;  // 16

  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv(swtch.context, sub_degree, fwd_level,
                             inverse_level, &layout);
  EvkRequest req;
  conv.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  EvkRequest boot_req;
  bctx->AddRequiredRotations(boot_req, num_slots);
  boot.ui->PrepareRotationKey(boot_req);

  // ---- scores through the pipeline: 1.5bw's front half, verbatim -------
  const double s = boot.param->GetScale(fwd_level);
  const RealBatch a = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.contraction, 0.08, 0xCA05);
  const RealBatch b = SampleBatch(layout.lanes, layout.dim,
                                  layout.contraction, layout.dim, 0.08,
                                  0xCA06);
  const RealBatch v = SampleBatch(layout.lanes, layout.dim, layout.dim,
                                  layout.dim, 0.08, 0xCA07);

  auto build = [&](const RealBatch &m, int num_big,
                   std::vector<Ciphertext<word>> &out) {
    out.resize(num_big);
    for (int bi = 0; bi < num_big; bi++) {
      std::vector<Complex> slot_msg(boot.Degree(), Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            slot_msg[slot] = Complex(m[lane][row][column], 0.0);
          }
        }
      }
      Plaintext<word> pt;
      boot.context->encoder_.Encode(pt, fwd_level, s, slot_msg);
      Ciphertext<word> enc;
      boot.ui->Encrypt(enc, pt);
      conv.SlotToSinC(swtch.context, out[bi], enc, swtch.ui->GetEvkMap());
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  build(a, layout.num_cts / 2, lhs);
  build(b, layout.num_cts, rhs);
  handler.Multiply(res, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                   swtch.ui->GetInverseRingSwitchKey(layout.rank),
                   lifted.ui->GetEvkMap());

  double carried = 0.0;
  std::vector<Ciphertext<word>> scores(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, res[bi], swtch.ui->GetEvkMap());
    carried = back.GetScale() / boot.param->base_scale_;
    back.SetNumSlots(num_slots);
    bctx->Boot(scores[bi], back, boot.ui->GetEvkMap());
  }
  const int top = boot.param->NPToLevel(scores[0].GetNP());

  // ---- host calibration: per-row live maxima, both norm intervals ------
  const double m_eff = 8.0;
  std::vector<std::vector<std::vector<double>>> S(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
  double smin = 1e300, smax = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int col = 0; col < layout.dim; col++) {
        double val = 0.0;
        for (int x = 0; x < layout.contraction; x++) {
          val += a[lane][row][x] * b[lane][x][col];
        }
        S[lane][row][col] = val;
        smin = std::min(smin, val);
        smax = std::max(smax, val);
      }
    }
  }
  const double span = smax - smin;
  auto u_of = [&](double val) { return 2.0 * (val - smax) / span + 1.0; };

  std::vector<std::vector<double>> rowmax(
      layout.lanes, std::vector<double>(layout.dim, -1e300));
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int k = 0; k <= row; k++) {
        rowmax[lane][row] = std::max(rowmax[lane][row], S[lane][row][k]);
      }
    }
  }
  double raw_lo = 1e300, raw_hi = -1e300, glo = 1e300, ghi = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      double sqv = 0.0, gsqv = 0.0;
      for (int k = 0; k <= row; k++) {
        sqv += std::exp(m_eff * (S[lane][row][k] - rowmax[lane][row]) / span);
        gsqv += std::exp(m_eff * (S[lane][row][k] - smax) / span);
      }
      raw_lo = std::min(raw_lo, sqv);
      raw_hi = std::max(raw_hi, sqv);
      glo = std::min(glo, gsqv);
      ghi = std::max(ghi, gsqv);
    }
  }
  const double lo = raw_lo * 0.9, hi = raw_hi * 1.1;

  const double hb = m_eff / 4.0;
  auto exp_coeffs = cheddar::chebfit::Interpolate(
      [hb](double val) { return std::exp(hb * (val - 1.0)); }, 7);
  const int exp_in = top - 1;
  auto log2ceil = [](int n) {
    int r = 0;
    while ((1 << r) < n) r++;
    return r;
  };
  const int exp_used =
      EvalPoly<word>(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                     boot.param->GetScale(exp_in), true)
          .GetPolyDegree();
  const int exp_out = exp_in - log2ceil(exp_used + 1);
  EvalPoly<word> exp_poly(exp_coeffs, exp_in, boot.param->GetScale(exp_in),
                          boot.param->GetScale(exp_out), true);
  exp_poly.Compile(boot.context);

  const int mask_level = exp_out - 1;
  const int sq_level = mask_level - 1;
  const int poly_in = sq_level - 1;
  const double aff_a = 0.5 * (hi - lo);
  const double aff_b = 0.5 * (hi + lo);
  auto inv_coeffs = cheddar::chebfit::Interpolate(
      [aff_a, aff_b](double val) {
        return 1.0 / std::sqrt(aff_a * val + aff_b);
      },
      15);
  const int inv_used =
      EvalPoly<word>(inv_coeffs, poly_in, boot.param->GetScale(poly_in),
                     boot.param->GetScale(poly_in), true)
          .GetPolyDegree();
  const int inv_out = poly_in - log2ceil(inv_used + 1);
  EvalPoly<word> inv_poly(inv_coeffs, poly_in,
                          boot.param->GetScale(poly_in),
                          boot.param->GetScale(inv_out), true);
  inv_poly.Compile(boot.context);

  std::vector<int> reduce_dist;
  for (int t = 0; t < 4; t++) {
    reduce_dist.push_back((num_slots / layout.rank) << t);
  }
  {
    EvkRequest rot_req;
    for (int d : reduce_dist) rot_req.AddRequest(d, sq_level);
    boot.ui->PrepareRotationKey(rot_req);
  }
  const auto &evk = boot.ui->GetEvkMap();
  const auto &mult_key = evk.GetMultiplicationKey();

  // ---- the affine with per-row shifts, exp, THE MASK -------------------
  const double a1 = 2.0 / (span * carried);
  std::vector<Ciphertext<word>> y(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Constant<word> c1;
    boot.context->encoder_.EncodeConstant(c1, top,
                                          boot.param->GetScale(top), a1);
    Ciphertext<word> t1, u_ct;
    boot.context->Mult(t1, scores[bi], c1);
    boot.context->Rescale(u_ct, t1);
    std::vector<Complex> a0_msg(boot.Degree(), Complex(0.0, 0.0));
    std::vector<Complex> mask_msg(boot.Degree(), Complex(0.0, 0.0));
    for (int row = 0; row < layout.dim; row++) {
      for (int j = 0; j < layout.rank; j++) {
        const int column = bi * layout.rank + j;
        for (int lane = 0; lane < layout.lanes; lane++) {
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const bool live = column <= row;
          const double shift = live ? rowmax[lane][row] : smax;
          a0_msg[slot] = Complex(1.0 - 2.0 * shift / span, 0.0);
          mask_msg[slot] = Complex(live ? 1.0 : 0.0, 0.0);
        }
      }
    }
    Plaintext<word> a0_pt;
    boot.context->encoder_.Encode(a0_pt, exp_in, u_ct.GetScale(), a0_msg);
    boot.context->Add(u_ct, u_ct, a0_pt);
    Ciphertext<word> y_full;
    exp_poly.Evaluate(boot.context, y_full, u_ct, mult_key);
    Plaintext<word> mask_pt;
    boot.context->encoder_.Encode(mask_pt, exp_out,
                                  boot.param->GetScale(exp_out), mask_msg);
    Ciphertext<word> t2;
    boot.context->Mult(t2, y_full, mask_pt);
    boot.context->Rescale(y[bi], t2);
  }
  // The norm now sums live keys only; the tree is 1.5bv's, untouched.
  Ciphertext<word> sq, term, rotated;
  boot.context->HMult(sq, y[0], y[0], mult_key);
  boot.context->HMult(term, y[1], y[1], mult_key);
  boot.context->Add(sq, sq, term);
  for (int d : reduce_dist) {
    boot.context->HRotAdd(rotated, sq, sq, evk.GetRotationKey(d), d);
    boot.context->Copy(sq, rotated);
  }
  {
    Constant<word> inv_a;
    boot.context->encoder_.EncodeConstant(
        inv_a, sq_level, boot.param->GetScale(sq_level), 1.0 / aff_a);
    Ciphertext<word> scaled;
    boot.context->Mult(scaled, sq, inv_a);
    boot.context->Rescale(sq, scaled);
    Constant<word> shift;
    boot.context->encoder_.EncodeConstant(shift, poly_in, sq.GetScale(),
                                          -aff_b / aff_a);
    boot.context->Add(sq, sq, shift);
  }
  Ciphertext<word> r;
  inv_poly.Evaluate(boot.context, r, sq, mult_key);
  const int meet = boot.param->NPToLevel(r.GetNP());
  std::vector<Ciphertext<word>> P(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> levelled, prod;
    boot.context->LevelDown(levelled, y[bi], meet);
    boot.context->HMult(prod, levelled, r, mult_key);
    boot.context->HMult(P[bi], prod, prod, mult_key);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int p_level = boot.param->NPToLevel(P[0].GetNP());
  ASSERT_GE(p_level, fwd_level)
      << "the masked walk overspent its levels; P fell below fwd_level";
  if (p_level > fwd_level) {
    for (int bi = 0; bi < layout.num_cts; bi++) {
      Ciphertext<word> down;
      boot.context->LevelDown(down, P[bi], fwd_level);
      P[bi] = std::move(down);
    }
  }
  ASSERT_EQ(boot.param->NPToLevel(P[0].GetNP()), fwd_level);

  // What P arrived as: the causal shape is checked HERE, before V.
  std::vector<std::vector<std::vector<double>>> P_dec(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
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
          P_dec[lane][row][column] = slots[slot].real();
        }
      }
    }
  }
  double masked_resid = 0.0, rowsum_dev = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      double rs = 0.0;
      for (int k = 0; k < layout.dim; k++) {
        if (k <= row) {
          rs += P_dec[lane][row][k];
        } else {
          masked_resid =
              std::max(masked_resid, std::abs(P_dec[lane][row][k]));
        }
      }
      rowsum_dev = std::max(rowsum_dev, std::abs(rs - 1.0));
    }
  }

  // ---- P x V, exactly 1.5bw's second contraction -----------------------
  std::vector<Ciphertext<word>> p_sinc(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    conv.SlotToSinC(swtch.context, p_sinc[bi], P[bi],
                    swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(p_sinc[bi].GetNP()), chain_level);
  }
  std::vector<Ciphertext<word>> out;
  for (int call = 0; call < 2; call++) {
    RealBatch h(layout.lanes,
                std::vector<std::vector<double>>(
                    layout.dim, std::vector<double>(layout.dim, 0.0)));
    for (int lane = 0; lane < layout.lanes; lane++) {
      for (int x = 0; x < half_keys; x++) {
        for (int col = 0; col < layout.dim; col++) {
          h[lane][x][col] = v[lane][call * half_keys + x][col];
        }
      }
    }
    std::vector<Ciphertext<word>> v_rhs, p_lhs, part;
    build(h, layout.num_cts, v_rhs);
    p_lhs.push_back(std::move(p_sinc[call]));
    handler.Multiply(part, p_lhs, v_rhs,
                     swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    if (call == 0) {
      out = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(out[bi], out[bi], part[bi]);
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // ---- back to slots, against the causal leg in the clear --------------
  double worst_true = 0.0, worst_incoming = 0.0, worst_plain = 0.0;
  double transposed = 0.0, half_sum = 0.0, biggest = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ciphertext<word> back;
    conv.SinCToSlot(swtch.context, back, out[bi], swtch.ui->GetEvkMap());
    ASSERT_EQ(boot.param->NPToLevel(back.GetNP()), 0);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, back);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int lane = 0; lane < layout.lanes; lane++) {
      for (int row = 0; row < layout.dim; row++) {
        std::vector<double> p_true(layout.dim, 0.0);
        std::vector<double> p_plain(layout.dim, 0.0);
        double zsum = 0.0, gzsum = 0.0;
        for (int k = 0; k < layout.dim; k++) {
          p_plain[k] = std::exp(m_eff * (u_of(S[lane][row][k]) - 1.0) / 2.0);
          gzsum += p_plain[k];
          if (k <= row) {
            p_true[k] = std::exp(
                m_eff * (S[lane][row][k] - rowmax[lane][row]) / span);
            zsum += p_true[k];
          }
        }
        for (double &pv : p_true) pv /= zsum;
        for (double &pv : p_plain) pv /= gzsum;
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          double want = 0.0, want_in = 0.0, want_plain = 0.0;
          double want_t = 0.0, want_half = 0.0;
          for (int k = 0; k < layout.dim; k++) {
            want += p_true[k] * v[lane][k][column];
            want_in += P_dec[lane][row][k] * v[lane][k][column];
            want_plain += p_plain[k] * v[lane][k][column];
            want_t += P_dec[lane][column][k] * v[lane][k][row];
            if (k < half_keys) {
              want_half += P_dec[lane][row][k] * v[lane][k][column];
            }
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          biggest = std::max(biggest, std::abs(want));
          worst_true = std::max(worst_true, std::abs(got - want));
          worst_incoming =
              std::max(worst_incoming, std::abs(got - want_in));
          worst_plain = std::max(worst_plain, std::abs(got - want_plain));
          transposed = std::max(transposed, std::abs(got - want_t));
          half_sum = std::max(half_sum, std::abs(got - want_half));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "the causal softmax through the walk (exp deg " << exp_used
            << " @" << exp_in << ".." << exp_out << ", mask @" << mask_level
            << ", invsqrt deg " << inv_used << " @" << poly_in << ".."
            << inv_out << "): vs causal softmax x V " << worst_true
            << ", vs decrypted P x V " << worst_incoming
            << " (|output| up to " << biggest << ")" << std::endl;
  std::cout << "  norm interval: per-row shifts [" << raw_lo << ", "
            << raw_hi << "]; the global shift would need [" << glo << ", "
            << ghi << "]" << std::endl;
  std::cout << "  P at masked addresses <= " << masked_resid
            << ", live row sums off one by " << rowsum_dev << std::endl;
  std::cout << "  controls: un-masked softmax " << worst_plain
            << ", transposed " << transposed << ", single-call half sum "
            << half_sum << std::endl;

  EXPECT_LT(worst_true, 2e-3)
      << "the causal attention output did not land at the primary "
         "addresses";
  EXPECT_LT(worst_incoming, 1e-3)
      << "the second contraction added more than its own floor";
  EXPECT_LT(masked_resid, 1e-3) << "the mask leaked into P";
  EXPECT_LT(rowsum_dev, 1e-2)
      << "the live rows are not stochastic, so the norm summed the wrong "
         "keys";
  EXPECT_GT(worst_plain, 5e-3)
      << "the causal and plain softmax agree, so the mask did nothing";
  EXPECT_GT(transposed, 1e-2);
  EXPECT_GT(half_sum, 1e-2)
      << "one call alone matches, so the two-call sum never happened";
}

// The library leg reproduces the reference (Doing.md 1.5cd).
//
// TheAttentionLegClosesEndToEnd is the reference implementation;
// `CiSinCAttention` is its promotion into the library, and this test is the
// promotion's proof: the SAME projections (real PC-MM emissions, real
// HalfBoots, as 1.5ca/1.5cb), driven through the handler's Scores ->
// caller's Boot -> PrepareSoftMax/SoftMax -> Values, against the SAME
// host reference. Two things differ from the reference on purpose:
//
// 1. THE SOFTMAX RUNS CAUSAL -- 1.5cc's mechanism at the Llama alignment,
//    which the reference has not run: exp deg 7 paying for the mask, the
//    per-row shifts carried by `SoftMaxCalibration::row_shift` (indexed by
//    LAYOUT lane), and -- new against 1.5cc, forced by the 128-key rows --
//    the per-row norm estimates folded into the mask as est^-1/2
//    (`row_norm`): at 128 live keys the raw interval is wide enough that
//    invsqrt deg 15 costs ~1.4e-2 on the row sums (this test's first run
//    measured it), and the fold collapses the interval to the actual /
//    estimate ratio at zero levels, est cancelling identically in
//    P = (y r)^2. The un-masked softmax reference must FAIL as a control,
//    and P at masked addresses must be numerically zero.
//
// 2. V'S RESTORE IS ONE MASK MULTIPLY per ciphertext instead of the
//    reference's zero-angle RoPE pair arithmetic -- the same values (the
//    sin terms were exact plaintext zeros), a quarter of the multiplies.
//
// Everything else -- premaps, gamma's split fold, the exchange, K's cross,
// V riding Q's converter, the chain calls, the walk's levels -- is the
// reference transcribed, and the layout constants are asserted inside the
// handler at construction.
TEST(CiBootSet, TheLibraryLegReproducesTheReference) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int pcmm_level = 1;
  const int chain_level = 2;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;
  const int in_ch = proj_rank;
  const int num_slots = boot.param->MaxNumSlots();

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };

  // The boot set first: the handler's constructor reads GetStCInputScale.
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest boot_req;
    bctx->AddRequiredRotations(boot_req, num_slots);
    boot.ui->PrepareRotationKey(boot_req);
  }

  // ---- the projections: real PC-MM emissions, as in 1.5ca/1.5cb ---------
  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);
  boot.ui->PrepareModPackKeys(proj_small, pcmm_level);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < proj_rank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
  }

  std::mt19937_64 gen(0xB141);
  std::uniform_real_distribution<double> xd(-1.0, 1.0);
  std::vector<std::vector<double>> x_comp(
      in_ch, std::vector<double>(proj_small, 0.0));
  for (auto &ch : x_comp) {
    for (auto &v : ch) v = xd(gen);
  }
  const int kDim = 128, kLanes = 32;
  const double wa = 0.24 / std::sqrt(static_cast<double>(in_ch));
  std::uniform_real_distribution<double> wd(-wa, wa);
  using W = std::vector<std::vector<std::vector<double>>>;
  W wq(kLanes, std::vector<std::vector<double>>(
                   kDim, std::vector<double>(in_ch, 0.0)));
  W wk = wq, wv = wq;
  for (int i = 0; i < kLanes; i++) {
    for (int c = 0; c < kDim; c++) {
      for (int o = 0; o < in_ch; o++) {
        wq[i][c][o] = wd(gen);
        wk[i][c][o] = wd(gen);
        wv[i][c][o] = wd(gen);
      }
    }
  }
  auto project = [&](const W &w) {
    RealBatch r(kLanes, std::vector<std::vector<double>>(
                            kDim, std::vector<double>(kDim, 0.0)));
    for (int i = 0; i < kLanes; i++) {
      for (int t = 0; t < kDim; t++) {
        for (int c = 0; c < kDim; c++) {
          double s = 0.0;
          for (int o = 0; o < in_ch; o++) s += w[i][c][o] * x_comp[o][t];
          r[i][t][c] = s;
        }
      }
    }
    return r;
  };
  const RealBatch q = project(wq);
  const RealBatch kk = project(wk);
  const RealBatch vv = project(wv);

  const int half = kDim / 2;
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / kDim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < kLanes; t++) {
      for (int i = 0; i < kDim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  {
    const auto x_rec = HostRecompose(x_comp, proj_rank, proj_small);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                       boot.param->GetScale(pcmm_level),
                                       x_rec);
    Ciphertext<word> x_ct;
    boot.ui->Encrypt(x_ct, pt);
    mlwe.ModDecomp(x_parts, x_ct, proj_small);
  }

  const double w_scale = boot.param->GetRescalePrimeProd(pcmm_level);
  auto emit_half = [&](const W &w, int l, int fam, Ciphertext<word> &out) {
    std::vector<double> vals(static_cast<size_t>(proj_rank) * in_ch, 0.0);
    for (int hh = 0; hh < 16; hh++) {
      for (int cp = 0; cp < 16; cp++) {
        const int row = hh * 16 + cp;
        for (int o = 0; o < in_ch; o++) {
          vals[static_cast<size_t>(row) * in_ch + o] =
              w[fam * 16 + hh][l * 16 + cp][o];
        }
      }
    }
    cheddar::PlainMatrix<word> u;
    pcmm.EncodeMatrix(u, pcmm_level, w_scale, vals, proj_rank, in_ch);
    std::vector<cheddar::MlweCiphertext<word>> mixed;
    pcmm.Multiply(mixed, u, x_parts);
    Ciphertext<word> packed, dropped;
    mlwe.ModPack(boot.context, packed, mixed, pack_keys);
    boot.context->Rescale(dropped, packed);
    dropped.SetNumSlots(num_slots);
    bctx->HalfBoot(out, dropped, boot.ui->GetEvkMap());
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8), v_a(8),
      v_b(8);
  for (int l = 0; l < 8; l++) {
    emit_half(wq, l, 0, q_a[l]);
    emit_half(wq, l, 1, q_b[l]);
    emit_half(wk, l, 0, k_a[l]);
    emit_half(wk, l, 1, k_b[l]);
    emit_half(wv, l, 0, v_a[l]);
    emit_half(wv, l, 1, v_b[l]);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_NEAR(q_a[0].GetScale() / bctx->GetStCInputScale(), 1.0, 1e-9);

  double hb_const = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, k_a[0]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    double rsum = 0.0;
    int counted = 0;
    for (int t = 0; t < kDim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const double want = kk[hh][t][cp];
          if (std::abs(want) < 0.02) continue;
          rsum += slots[door0(t, cp, hh)].real() / want;
          counted++;
        }
      }
    }
    hb_const = rsum / counted;
    ASSERT_LT(std::abs(std::log2(std::abs(hb_const) * 32.0)), 0.5);
  }

  // ---- the handler, its keys ------------------------------------------
  typename cheddar::CiSinCAttention<word>::Config acfg;
  acfg.restore = 1.0 / hb_const;
  const auto t0 = std::chrono::steady_clock::now();
  cheddar::CiSinCAttention<word> attn(bctx, swtch.context, small.context,
                                      lifted.context, acfg);
  const auto t1 = std::chrono::steady_clock::now();
  const auto &layout = attn.GetLayout();

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

  // ---- host calibration off the clear twin ----------------------------
  const double m_eff = 8.0;
  std::vector<std::vector<std::vector<double>>> S(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
  double smin = 1e300, smax = -1e300;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int col = 0; col < layout.dim; col++) {
        double v = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          v += q_ref[head][row][c] * k_ref[head][col][c];
        }
        S[head][row][col] = v;
        smin = std::min(smin, v);
        smax = std::max(smax, v);
      }
    }
  }
  const double span = smax - smin;
  // row_shift is indexed by the LAYOUT lane; the physical head there is
  // rev5(lane), exactly as every slot read in this suite.
  std::vector<std::vector<double>> row_shift(
      layout.lanes, std::vector<double>(layout.dim, -1e300));
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      for (int k = 0; k <= row; k++) {
        row_shift[lane][row] =
            std::max(row_shift[lane][row], S[head][row][k]);
      }
    }
  }
  // The live-norm estimates, folded into the mask as est^-1/2. At 128
  // live keys the raw interval is wide enough that invsqrt deg 15 costs
  // ~1.4e-2 on the row sums (the first run of this test measured it);
  // with the fold the interval is the actual / estimate ratio, here 1 by
  // construction, and only the margins remain.
  std::vector<std::vector<double>> row_norm(
      layout.lanes, std::vector<double>(layout.dim, 0.0));
  double raw_lo = 1e300, raw_hi = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      double sqv = 0.0;
      for (int k = 0; k <= row; k++) {
        sqv += std::exp(m_eff * (S[head][row][k] - row_shift[lane][row]) /
                        span);
      }
      row_norm[lane][row] = sqv;
      raw_lo = std::min(raw_lo, sqv);
      raw_hi = std::max(raw_hi, sqv);
    }
  }

  typename cheddar::CiSinCAttention<word>::SoftMaxCalibration calib;
  calib.m_eff = m_eff;
  calib.span = span;
  calib.shift = smax;
  calib.norm_lo = 0.9;
  calib.norm_hi = 1.1;
  calib.causal = true;
  calib.row_shift = row_shift;
  calib.row_norm = row_norm;
  attn.PrepareSoftMax(calib);

  // ---- the leg through the handler ------------------------------------
  const auto t2 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> s0;
  attn.Scores(s0, q_a, q_b, k_a, k_b, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t3 = std::chrono::steady_clock::now();
  const double carried = s0[0].GetScale() / boot.param->base_scale_;
  ASSERT_LT(carried * std::max(std::abs(smax), std::abs(smin)), 0.95)
      << "the handler's canonicalising fold did not land carried in "
         "EvalMod's range";

  std::vector<Ciphertext<word>> scores(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    s0[bi].SetNumSlots(num_slots);
    bctx->Boot(scores[bi], s0[bi], boot.ui->GetEvkMap());
  }
  ASSERT_EQ(boot.param->NPToLevel(scores[0].GetNP()), attn.GetTopLevel());

  const auto t4 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> P;
  attn.SoftMax(P, scores, carried, boot.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t5 = std::chrono::steady_clock::now();

  // What P arrived as: the causal shape checked before V.
  std::vector<std::vector<std::vector<double>>> P_dec(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
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
          P_dec[rev(lane, 5)][row][column] = slots[slot].real();
        }
      }
    }
  }
  double masked_resid = 0.0, rowsum_dev = 0.0;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      double rs = 0.0;
      for (int k = 0; k < layout.dim; k++) {
        if (k <= row) {
          rs += P_dec[head][row][k];
        } else {
          masked_resid =
              std::max(masked_resid, std::abs(P_dec[head][row][k]));
        }
      }
      rowsum_dev = std::max(rowsum_dev, std::abs(rs - 1.0));
    }
  }

  const auto t6 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> out;
  attn.Values(out, P, v_a, v_b, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t7 = std::chrono::steady_clock::now();

  // ---- against the causal leg in the clear ----------------------------
  double worst_true = 0.0, worst_incoming = 0.0, worst_plain = 0.0;
  double transposed = 0.0, biggest = 0.0;
  double copy_gap = 0.0;
  long long copy_seen = 0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    ASSERT_EQ(boot.param->NPToLevel(out[bi].GetNP()), 0);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, out[bi]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int lane = 0; lane < layout.lanes; lane++) {
      const int head = rev(lane, 5);
      for (int row = 0; row < layout.dim; row++) {
        std::vector<double> p_true(layout.dim, 0.0);
        std::vector<double> p_plain(layout.dim, 0.0);
        double zsum = 0.0, gzsum = 0.0;
        for (int k = 0; k < layout.dim; k++) {
          p_plain[k] =
              std::exp(m_eff * (S[head][row][k] - smax) / span);
          gzsum += p_plain[k];
          if (k <= row) {
            p_true[k] = std::exp(
                m_eff * (S[head][row][k] - row_shift[lane][row]) / span);
            zsum += p_true[k];
          }
        }
        for (double &pv : p_true) pv /= zsum;
        for (double &pv : p_plain) pv /= gzsum;
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          double want = 0.0, want_in = 0.0, want_plain = 0.0, want_t = 0.0;
          for (int k = 0; k < layout.dim; k++) {
            want += p_true[k] * vv[head][k][column];
            want_in += P_dec[head][row][k] * vv[head][k][column];
            want_plain += p_plain[k] * vv[head][k][column];
            want_t += P_dec[head][column][k] * vv[head][k][row];
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          // THE SEAM TO THE O PROJECTION, asked here because this is the
          // only place the leg's output is already in hand. A
          // coefficient-domain projection on R+ needs the BANDED
          // convention -- each entry at its own address AND at its
          // partner's (Doing.md 1.5cs) -- which is the pair LocateSlot
          // names. If the OUTPUT carries it, Boot + StC hands the O
          // projection what it wants with no transform and the CI block's
          // last seam is free.
          if (copy_slot >= 0) {
            copy_gap = std::max(copy_gap,
                                std::abs(slots[copy_slot].real() - got));
            copy_seen++;
          }
          biggest = std::max(biggest, std::abs(want));
          worst_true = std::max(worst_true, std::abs(got - want));
          worst_incoming =
              std::max(worst_incoming, std::abs(got - want_in));
          worst_plain = std::max(worst_plain, std::abs(got - want_plain));
          transposed = std::max(transposed, std::abs(got - want_t));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  auto secs = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << "the library leg (CiSinCAttention, causal): vs the causal "
            << "leg in the clear " << worst_true << ", vs decrypted P x V "
            << worst_incoming << " (|output| <= " << biggest << ", carried "
            << carried << ")" << std::endl;
  std::cout << "  P at masked addresses <= " << masked_resid
            << ", live row sums off one by " << rowsum_dev << std::endl;
  std::cout << "  raw live-norm interval [" << raw_lo << ", " << raw_hi
            << "] folded to [0.9, 1.1] by the row_norm mask" << std::endl;
  std::cout << "  controls: un-masked softmax " << worst_plain
            << ", transposed " << transposed << std::endl;
  std::cout << "  THE SEAM: the output's copy addresses differ from their "
            << "primaries by at most " << copy_gap << " over " << copy_seen
            << " pairs (|output| <= " << biggest << "). Below the leg's own "
            << "floor means the banded convention is already there and the "
            << "O projection needs no transform." << std::endl;
  std::cout << "  cost: construct " << secs(t0, t1) << " s, Scores "
            << secs(t2, t3) << " s, SoftMax " << secs(t4, t5)
            << " s, Values " << secs(t6, t7) << " s" << std::endl;

  EXPECT_LT(worst_true, 2e-2)
      << "the handler did not land the attention output at the primary "
         "addresses";
  EXPECT_LT(worst_incoming, 1e-2)
      << "the second contraction added more than its own floor";
  EXPECT_LT(masked_resid, 1e-3) << "the causal mask leaked into P";
  EXPECT_LT(rowsum_dev, 1e-2);
  EXPECT_GT(worst_plain, 5e-3)
      << "the causal and plain softmax agree, so the mask did nothing";
  EXPECT_GT(transposed, 5e-3);
}

// The projections run at full width (Doing.md 1.5ce).
//
// 1.5ca emitted the half-density images from a toy 512-channel X -- one
// ciphertext, one ModDecomp, a 512 x 512 weight slice. The layer's real
// residual is 4096 channels = EIGHT such ciphertexts, and the promised
// "same machinery summed over eight input cts" turns out to need no sum
// at all: `PcmmHandler::Multiply` computes res[i] = sum_j u[i][j] cts[j]
// over however many columns U has, so the eight ciphertexts' ModDecomp
// parts CONCATENATE into one 4096-part vector and U widens to 512 x 4096
// -- the cross-ciphertext accumulation lives inside the PP-MM's own inner
// sum, zero new mechanism, zero library code.
//
// What changes against 1.5ca, exactly:
//   - X is 4096 channels over eight ciphertexts, each holding channels
//     [512 l, 512 l + 512) as the banded recomposition of its components;
//     eight ModDecomps run ONCE and all 32 emissions mix the same parts.
//   - The weight slice per half-image is 512 x 4096 (rows I >= 256 still
//     zero), encoded at the same rp(1) scale.
//   - ModPack, Rescale, HalfBoot, the transport and the chain are
//     untouched -- width never reaches them.
//
// Measured to the scores through the real transport (HalfBoot @19 ->
// RoPE + restore + kill -> merge -> exchange -> [K: cross] -> premapped
// descents @3 -> chain x2), against the full-width projections in the
// clear, with the transposed and un-RoPE'd controls. The weights scale as
// 1/sqrt(width), so |S| stays put and the noise datum is directly
// comparable with 1.5ca's 1.42e-04.
TEST(CiBootSet, TheProjectionsRunAtFullWidth) {
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int pcmm_level = 1;
  const int land_level = 19;
  const int exchange_level = 18;
  const int cross_level = 17;
  const int conv_level = 3;
  const int chain_level = 2;
  const int sub_degree = 32;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;  // 128 = T
  const int num_x = 8;                   // eight X ciphertexts
  const int in_ch = num_x * proj_rank;   // 4096: the layer's real width

  CiSwitchedCcmmHandler<word> handler(swtch.context, small.context,
                                      lifted.context, sub_degree);
  const CiSwitchedCcmmLayout &layout = handler.GetLayout();
  const int half = layout.contraction;  // 64

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  auto exch = [](int s) {
    const int a = (s >> 7) & 31, b = s & 31;
    return (s & ~((31 << 7) | 31)) | (b << 7) | a;
  };
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };
  auto door1 = [&](int t, int c, int i) { return exch(door0(t, c, i)); };
  auto door1k = [&](int t, int c, int i) {
    return (door1(t, c, i) & ~(7 << 7)) | (((c / 16) % 4) << 7);
  };

  const int num_blocks = n / sub_degree;
  std::vector<int> pre_q(num_blocks, -1), pre_k(num_blocks, -1);
  {
    int bad = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int c = 0; c < layout.dim; c++) {
        int ct_idx, slot, copy_slot;
        layout.LocateSlot(t, c, 0, ct_idx, slot, copy_slot);
        if (ct_idx != c / 16) bad++;
        int db = door1(t, c, 0) >> 5;
        if (pre_q[db] == -1) pre_q[db] = slot >> 5;
        if (pre_q[db] != (slot >> 5)) bad++;
        layout.LocateSlot(c % half, t, 0, ct_idx, slot, copy_slot);
        if (ct_idx != t / 16) bad++;
        db = door1k(t, c, 0) >> 5;
        if (pre_k[db] == -1) pre_k[db] = slot >> 5;
        if (pre_k[db] != (slot >> 5)) bad++;
      }
    }
    ASSERT_EQ(bad, 0);
    std::vector<char> used(num_blocks, 0);
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] != -1) used[pre_k[b]] = 1;
    }
    std::vector<int> free_out;
    for (int b = 0; b < num_blocks; b++) {
      if (!used[b]) free_out.push_back(b);
    }
    size_t fo = 0;
    for (int b = 0; b < num_blocks; b++) {
      if (pre_k[b] == -1) pre_k[b] = free_out[fo++];
    }
    ASSERT_EQ(fo, free_out.size());
  }

  // ---- keys, converters, the exchange --------------------------------
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : handler.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  CiSinCConverter<word> conv_q(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_q);
  CiSinCConverter<word> conv_k(swtch.context, sub_degree,
                               /*forward_level=*/conv_level,
                               /*inverse_level=*/-1, &layout, &pre_k);
  EvkRequest req;
  conv_q.AddRequiredRotations(req);
  conv_k.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  const int window = 31 * 127;
  cheddar::StripedMatrix em(n, n);
  for (int r = 0; r < n; r++) {
    const int in = exch(r);
    const int off = ((in - r) % n + n) % n;
    em.try_emplace(off, n, Complex(0.0, 0.0));
    em[off][r] = Complex(1.0, 0.0);
  }
  cheddar::LinearTransform<word> lt_exch(
      boot.context, em, exchange_level,
      boot.param->GetRescalePrimeProd(exchange_level), 8, 8,
      /*pre_rotation=*/-window, /*additional_pt_rot=*/window);
  const int window_back = n - window;
  {
    EvkRequest ereq;
    lt_exch.AddRequiredRotations(ereq);
    ereq.AddRequest(window_back, exchange_level - 1);
    boot.ui->PrepareRotationKey(ereq);
  }
  const int merge_idx = n - 128;
  boot.ui->PrepareRotationKey(merge_idx, exchange_level);
  {
    std::set<int> idxs;
    for (int u = 0; u < 4; u++) {
      for (int v = 0; v < 8; v++) {
        const int rot = (v - u) * 128;
        if (rot != 0) idxs.insert((rot % n + n) % n);
      }
    }
    for (int idx : idxs) boot.ui->PrepareRotationKey(idx, cross_level);
  }

  const int num_slots = boot.param->MaxNumSlots();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest boot_req;
    bctx->AddRequiredRotations(boot_req, num_slots);
    boot.ui->PrepareRotationKey(boot_req);
  }

  // ---- the projections: eight X ciphertexts, one part vector ----------
  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);
  boot.ui->PrepareModPackKeys(proj_small, pcmm_level);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < proj_rank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
  }

  std::mt19937_64 gen(0xB151);
  std::uniform_real_distribution<double> xd(-1.0, 1.0);
  std::vector<std::vector<double>> x_comp(
      in_ch, std::vector<double>(proj_small, 0.0));
  for (auto &ch : x_comp) {
    for (auto &v : ch) v = xd(gen);
  }
  const double wa = 0.24 / std::sqrt(static_cast<double>(in_ch));
  std::uniform_real_distribution<double> wd(-wa, wa);
  std::vector<std::vector<std::vector<double>>> wq(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(in_ch, 0.0)));
  auto wk = wq;
  for (int i = 0; i < layout.lanes; i++) {
    for (int c = 0; c < layout.dim; c++) {
      for (int o = 0; o < in_ch; o++) {
        wq[i][c][o] = wd(gen);
        wk[i][c][o] = wd(gen);
      }
    }
  }
  auto project = [&](const std::vector<std::vector<std::vector<double>>> &w) {
    RealBatch r(layout.lanes,
                std::vector<std::vector<double>>(
                    layout.dim, std::vector<double>(layout.dim, 0.0)));
    for (int i = 0; i < layout.lanes; i++) {
      for (int t = 0; t < layout.dim; t++) {
        for (int c = 0; c < layout.dim; c++) {
          double s = 0.0;
          for (int o = 0; o < in_ch; o++) s += w[i][c][o] * x_comp[o][t];
          r[i][t][c] = s;
        }
      }
    }
    return r;
  };
  const RealBatch q = project(wq);
  const RealBatch kk = project(wk);

  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / layout.dim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < layout.lanes; t++) {
      for (int i = 0; i < layout.dim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  // Eight encryptions, eight ModDecomps, ONE concatenated part vector:
  // parts [512 l, 512 l + 512) come from the ciphertext holding channels
  // [512 l, 512 l + 512), so part index == input-channel index and the
  // wider U below reads it directly.
  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  double decomp_seconds = 0.0;
  {
    const auto d0 = std::chrono::steady_clock::now();
    for (int l = 0; l < num_x; l++) {
      std::vector<std::vector<double>> slice(
          x_comp.begin() + l * proj_rank,
          x_comp.begin() + (l + 1) * proj_rank);
      const auto x_rec = HostRecompose(slice, proj_rank, proj_small);
      Plaintext<word> pt;
      boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                         boot.param->GetScale(pcmm_level),
                                         x_rec);
      Ciphertext<word> x_ct;
      boot.ui->Encrypt(x_ct, pt);
      std::vector<cheddar::MlweCiphertext<word>> parts;
      mlwe.ModDecomp(parts, x_ct, proj_small);
      ASSERT_EQ(static_cast<int>(parts.size()), proj_rank);
      for (auto &p : parts) x_parts.push_back(std::move(p));
    }
    cudaDeviceSynchronize();
    decomp_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - d0)
            .count();
  }
  ASSERT_EQ(static_cast<int>(x_parts.size()), in_ch);

  const double w_scale = boot.param->GetRescalePrimeProd(pcmm_level);
  double pcmm_seconds = 0.0, halfboot_seconds = 0.0;
  int emit_count = 0;
  auto emit_half = [&](const std::vector<std::vector<std::vector<double>>> &w,
                       int l, int fam, Ciphertext<word> &out) {
    std::vector<double> vals(static_cast<size_t>(proj_rank) * in_ch, 0.0);
    for (int hh = 0; hh < 16; hh++) {
      for (int cp = 0; cp < 16; cp++) {
        const int row = hh * 16 + cp;
        for (int o = 0; o < in_ch; o++) {
          vals[static_cast<size_t>(row) * in_ch + o] =
              w[fam * 16 + hh][l * 16 + cp][o];
        }
      }
    }
    const auto p0 = std::chrono::steady_clock::now();
    cheddar::PlainMatrix<word> u;
    pcmm.EncodeMatrix(u, pcmm_level, w_scale, vals, proj_rank, in_ch);
    std::vector<cheddar::MlweCiphertext<word>> mixed;
    pcmm.Multiply(mixed, u, x_parts);
    ASSERT_EQ(static_cast<int>(mixed.size()), proj_rank);
    Ciphertext<word> packed, dropped;
    mlwe.ModPack(boot.context, packed, mixed, pack_keys);
    boot.context->Rescale(dropped, packed);
    ASSERT_EQ(boot.param->NPToLevel(dropped.GetNP()), 0);
    cudaDeviceSynchronize();
    pcmm_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - p0)
            .count();
    dropped.SetNumSlots(num_slots);
    const auto h0 = std::chrono::steady_clock::now();
    bctx->HalfBoot(out, dropped, boot.ui->GetEvkMap());
    cudaDeviceSynchronize();
    halfboot_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - h0)
            .count();
    emit_count++;
    ASSERT_EQ(boot.param->NPToLevel(out.GetNP()), land_level);
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8);
  for (int l = 0; l < 8; l++) {
    emit_half(wq, l, 0, q_a[l]);
    emit_half(wq, l, 1, q_b[l]);
    emit_half(wk, l, 0, k_a[l]);
    emit_half(wk, l, 1, k_b[l]);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  double hb_const = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, k_a[0]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    double rlo = 1e300, rhi = -1e300, rsum = 0.0;
    int counted = 0;
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const double want = kk[hh][t][cp];
          if (std::abs(want) < 0.02) continue;
          const double r = slots[door0(t, cp, hh)].real() / want;
          rlo = std::min(rlo, r);
          rhi = std::max(rhi, r);
          rsum += r;
          counted++;
        }
      }
    }
    hb_const = rsum / counted;
    std::cout << "  boundary constant at full width " << hb_const << " (2^"
              << std::log2(std::abs(hb_const)) << "), spread [" << rlo
              << ", " << rhi << "] over " << counted << std::endl;
    ASSERT_LT(std::abs(std::log2(std::abs(hb_const) * 32.0)), 0.5);
  }

  const double restore = 1.0 / hb_const;
  auto rope_and_kill = [&](std::vector<Ciphertext<word>> &a_cts,
                           std::vector<Ciphertext<word>> &b_cts) {
    const double pt_scale = boot.param->GetRescalePrimeProd(land_level);
    for (int lo = 0; lo < 4; lo++) {
      std::vector<Complex> cm(n, Complex(0.0, 0.0));
      std::vector<Complex> sm(n, Complex(0.0, 0.0));
      std::vector<Complex> nm(n, Complex(0.0, 0.0));
      for (int t = 0; t < layout.dim; t++) {
        for (int cp = 0; cp < 16; cp++) {
          const double ang = t * theta[lo * 16 + cp];
          for (int hh = 0; hh < 16; hh++) {
            const int slot = door0(t, lo * 16 + cp, hh);
            cm[slot] = Complex(std::cos(ang) * restore, 0.0);
            sm[slot] = Complex(std::sin(ang) * restore, 0.0);
            nm[slot] = Complex(-std::sin(ang) * restore, 0.0);
          }
        }
      }
      Plaintext<word> cos_pt, sin_pt, neg_sin_pt;
      boot.context->encoder_.Encode(cos_pt, land_level, pt_scale, cm);
      boot.context->encoder_.Encode(sin_pt, land_level, pt_scale, sm);
      boot.context->encoder_.Encode(neg_sin_pt, land_level, pt_scale, nm);
      for (int fam = 0; fam < 2; fam++) {
        std::vector<Ciphertext<word>> &cts = (fam == 0) ? a_cts : b_cts;
        Ciphertext<word> &lo_ct = cts[lo];
        Ciphertext<word> &hi_ct = cts[lo + 4];
        Ciphertext<word> aa, bb, dd;
        boot.context->Mult(aa, lo_ct, cos_pt);
        boot.context->Mult(bb, hi_ct, neg_sin_pt);
        boot.context->Add(aa, aa, bb);
        boot.context->Mult(bb, hi_ct, cos_pt);
        boot.context->Mult(dd, lo_ct, sin_pt);
        boot.context->Add(bb, bb, dd);
        boot.context->Rescale(lo_ct, aa);
        boot.context->Rescale(hi_ct, bb);
      }
    }
  };
  auto merge = [&](std::vector<Ciphertext<word>> &a_cts,
                   std::vector<Ciphertext<word>> &b_cts) {
    for (int l = 0; l < layout.num_cts; l++) {
      Ciphertext<word> moved;
      boot.context->HRot(moved, b_cts[l],
                         boot.ui->GetEvkMap().GetRotationKey(merge_idx),
                         merge_idx);
      boot.context->Add(a_cts[l], a_cts[l], moved);
    }
  };
  auto exchange_all = [&](std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      Ciphertext<word> shifted, swapped;
      lt_exch.Evaluate(boot.context, shifted, ct, boot.ui->GetEvkMap());
      boot.context->HRot(swapped, shifted,
                         boot.ui->GetEvkMap().GetRotationKey(window_back),
                         window_back);
      ct = std::move(swapped);
    }
  };

  rope_and_kill(q_a, q_b);
  rope_and_kill(k_a, k_b);
  merge(q_a, q_b);
  merge(k_a, k_b);
  exchange_all(q_a);
  exchange_all(k_a);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const double cr_pt_scale = boot.param->GetRescalePrimeProd(cross_level);
  std::vector<Plaintext<word>> sel(8);
  for (int v = 0; v < 8; v++) {
    std::vector<Complex> msg(n, Complex(0.0, 0.0));
    for (int s = 0; s < n; s++) {
      if (((s >> 7) & 7) == v) msg[s] = Complex(1.0, 0.0);
    }
    boot.context->encoder_.Encode(sel[v], cross_level, cr_pt_scale, msg);
  }
  auto cross = [&](int call) {
    std::vector<Ciphertext<word>> out(layout.num_cts);
    for (int t_hi = 0; t_hi < layout.num_cts; t_hi++) {
      const int v = rev(t_hi, 3);
      Ciphertext<word> acc;
      bool first = true;
      for (int l = call * 4; l < call * 4 + 4; l++) {
        Ciphertext<word> piece;
        boot.context->Mult(piece, k_a[l], sel[v]);
        const int rot = (v - l % 4) * 128;
        if (rot != 0) {
          const int idx = (rot % n + n) % n;
          Ciphertext<word> moved;
          boot.context->HRot(moved, piece,
                             boot.ui->GetEvkMap().GetRotationKey(idx), idx);
          piece = std::move(moved);
        }
        if (first) {
          acc = std::move(piece);
          first = false;
        } else {
          boot.context->Add(acc, acc, piece);
        }
      }
      boot.context->Rescale(out[t_hi], acc);
    }
    return out;
  };

  auto convert = [&](CiSinCConverter<word> &conv,
                     std::vector<Ciphertext<word>> &cts) {
    for (auto &ct : cts) {
      Ciphertext<word> at_level;
      boot.context->LevelDown(at_level, ct, conv_level);
      Ciphertext<word> sinc;
      conv.SlotToSinC(swtch.context, sinc, at_level, swtch.ui->GetEvkMap());
      ct = std::move(sinc);
    }
  };
  convert(conv_q, q_a);

  std::vector<Ciphertext<word>> res;
  for (int call = 0; call < 2; call++) {
    std::vector<Ciphertext<word>> rhs = cross(call);
    convert(conv_k, rhs);
    std::vector<Ciphertext<word>> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(q_a[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ciphertext<word>> part;
    handler.Multiply(part, lhs, rhs, swtch.ui->GetRingSwitchKey(layout.rank),
                     swtch.ui->GetInverseRingSwitchKey(layout.rank),
                     lifted.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    if (call == 0) {
      res = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        boot.context->Add(res[bi], res[bi], part[bi]);
      }
    }
  }

  std::vector<std::vector<Complex>> got(layout.dim);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    ASSERT_EQ(boot.param->NPToLevel(res[bi].GetNP()), chain_level - 1);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, res[bi]);
    std::vector<double> coeffs;
    boot.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, layout.rank, small.Degree());
    for (int j = 0; j < layout.rank; j++) {
      Plaintext<word> bridge;
      small.context->encoder_.EncodeCoeff(
          bridge, chain_level - 1,
          small.param->GetScale(chain_level - 1), comp[j]);
      small.context->encoder_.DecodeSinC(got[bi * layout.rank + j], bridge,
                                         sub_degree);
    }
  }

  double worst = 0.0, transposed = 0.0, norope = 0.0, biggest = 0.0;
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0, want_raw = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          want += q_ref[head][row][c] * k_ref[head][column][c];
          want_t += q_ref[head][column][c] * k_ref[head][row][c];
          want_raw += q[head][row][c] * kk[head][column][c];
        }
        biggest = std::max(biggest, std::abs(want));
        int part, index;
        layout.LocatePart(row, column, lane, part, index);
        const double g = got[part][index].real();
        worst = std::max(worst, std::abs(g - want));
        transposed = std::max(transposed, std::abs(g - want_t));
        norope = std::max(norope, std::abs(g - want_raw));
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "full width (" << in_ch << " channels over " << num_x
            << " X cts, one " << proj_rank << " x " << in_ch
            << " U per half-image): products " << worst << " (|scores| <= "
            << biggest << ")" << std::endl;
  std::cout << "  controls: transposed " << transposed << ", un-RoPE'd "
            << norope << std::endl;
  std::cout << "  cost: ModDecomp x" << num_x << " " << decomp_seconds
            << " s once, PC-MM emit "
            << pcmm_seconds / std::max(emit_count, 1) << " s/ct, HalfBoot "
            << halfboot_seconds / std::max(emit_count, 1) << " s/ct over "
            << emit_count << " cts" << std::endl;

  EXPECT_LT(worst, 2e-2)
      << "the full-width projections did not land the scores at the "
         "primary parts";
  EXPECT_GT(transposed, 1e-2);
  EXPECT_GT(norope, 1e-2);
}

// The leg runs on the real weights (Doing.md 1.5ch).
//
// Everything so far projected synthetic tensors; this test walks the
// promoted leg on Llama-3-8B's OWN layer-2 numbers: the 128-token
// residual and the q/k/v projection weights from `LLAMA3_REAL_DIR`
// (LlamaBlockTest's convention -- raw f32, weights [input][output]
// row-major, wk/wv 4096 x 1024 GQA), through the full-width PC-MM
// (1.5ce: eight X ciphertexts, one 4096-part vector, 512 x 4096 U),
// the transport, the causal softmax and P x V, driven by
// `CiSinCAttention` end to end. Skipped when the blobs are absent.
//
// What the real data changes, all pinned by the 1.5cf pre-study:
//
// 1. TEMPERATURE IS NOT A KNOB. The circuit computes
//    exp(m_eff S / span); the model computes exp(S); so m_eff = the
//    TRUE span (24.4 here) and hb = span / 4 = 6.1 -- where exp deg 7
//    fits at 3.2e-03 (would dominate the floor) and deg 15 at 2.5e-08
//    for the SAME four levels deg 9 took. The level for the causal
//    mask comes from the OTHER end: with the row_norm estimate folded
//    (1.5cd) the invsqrt interval is a ratio around 1, where deg 7 is
//    below 1e-9 -- so the walk is exp 15 + mask + invsqrt 7, and P
//    lands at forward_level exactly.
//
// 2. THE SCORES MUST BE SCALED INTO EVALMOD'S RANGE. |S| reaches 19.8
//    and the chain computes sqrt(128) S (the 1/sqrt(d) fold is
//    upstream); carried ~2.2 demands |message| < 0.43. The ordinary
//    leg's size_q/size_k answer ports directly: one constant folds
//    into each weight encode (the PC-MM's own w_scale), and the
//    softmax affine divides it back out -- softmax shift-and-scale
//    invariance makes it exact, and the handler's calibration keeps
//    the two unit systems apart (m_eff in TRUE units, span/shift/
//    row_shift in chain units, row_norm unitless).
//
// 3. GQA IS A WEIGHT SLICE. K/V head hh serves query heads 4 hh .. 4
//    hh + 3, so the half-image emission reads wk/wv column
//    ((fam * 16 + hh) / 4) * 128 + c -- replication in the encode,
//    zero mechanism.
//
// The sinks stay upstream on purpose: 1.5cf measured |scores against
// sink keys| <= 2.7 versus 19.8 for the live rest, so the leg's own
// calibration never sees them; the RMSNorm window they do break is
// the block's business (public sinks, 1.5cd recon), and this test
// consumes the host-normed residual as the leg's contract input.
TEST(CiBootSet, TheLegRunsOnTheRealWeights) {
  const char *env = std::getenv("LLAMA3_REAL_DIR");
  if (env == nullptr) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";
  const std::string dir(env);
  auto read_f32 = [&](const std::string &name, size_t count,
                      std::vector<double> &out) {
    std::ifstream f(dir + "/" + name, std::ios::binary);
    if (!f) return false;
    std::vector<float> raw(count);
    f.read(reinterpret_cast<char *>(raw.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != count * sizeof(float))
      return false;
    out.assign(raw.begin(), raw.end());
    return true;
  };
  const int kT = 128, kH = 4096, kKv = 1024, kD = 128;
  std::vector<double> resid, gnorm, wq_f, wk_f, wv_f;
  ASSERT_TRUE(read_f32("input_nosink.f32",
                       static_cast<size_t>(kT) * kH, resid));
  ASSERT_TRUE(read_f32("attn_norm.f32", kH, gnorm));
  ASSERT_TRUE(read_f32("wq.f32", static_cast<size_t>(kH) * kH, wq_f));
  ASSERT_TRUE(read_f32("wk.f32", static_cast<size_t>(kH) * kKv, wk_f));
  ASSERT_TRUE(read_f32("wv.f32", static_cast<size_t>(kH) * kKv, wv_f));

  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  Ring lifted(kBootLiftedParam,
              CiLiftHandler<word>::LiftSecret(small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int pcmm_level = 1;
  const int chain_level = 2;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;  // 128 = T
  const int num_x = 8;
  const int num_slots = boot.param->MaxNumSlots();

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };
  (void)door0;

  // ---- the host pipeline down to the leg's doorstep -------------------
  // RMSNorm is upstream of the leg; the normed residual IS the leg's X.
  std::vector<double> normed(static_cast<size_t>(kT) * kH, 0.0);
  for (int t = 0; t < kT; t++) {
    double sq = 0.0;
    for (int c = 0; c < kH; c++) {
      const double u = resid[static_cast<size_t>(t) * kH + c];
      sq += u * u;
    }
    const double inv = 1.0 / std::sqrt(sq / kH + 1e-5);
    for (int c = 0; c < kH; c++) {
      normed[static_cast<size_t>(t) * kH + c] =
          resid[static_cast<size_t>(t) * kH + c] * inv * gnorm[c];
    }
  }
  // Projections in the clear, RoPE'd, and the true scores -- the 1.5cf
  // analysis inline, for the calibration and the reference.
  const double rope_theta = 500000.0;
  auto project = [&](const std::vector<double> &w, int width, double scale,
                     RealBatch &out) {
    // out[head][t][c], head over width / kD heads at the GQA width.
    const int heads_w = width / kD;
    out.assign(heads_w, std::vector<std::vector<double>>(
                            kT, std::vector<double>(kD, 0.0)));
    for (int t = 0; t < kT; t++) {
      for (int o = 0; o < width; o++) {
        double s = 0.0;
        for (int c = 0; c < kH; c++) {
          s += normed[static_cast<size_t>(t) * kH + c] *
               w[static_cast<size_t>(c) * width + o];
        }
        out[o / kD][t][o % kD] = s * scale;
      }
    }
  };
  auto rope_batch = [&](RealBatch &m) {
    const int half = kD / 2;
    for (auto &head : m) {
      for (int t = 0; t < kT; t++) {
        const std::vector<double> src = head[t];
        for (int j = 0; j < kD; j++) {
          const double f = std::pow(rope_theta, -2.0 * (j % half) / kD);
          const double cs = std::cos(t * f), sn = std::sin(t * f);
          const int partner = (j < half) ? j + half : j - half;
          head[t][j] = src[j] * cs + (j < half ? -1.0 : 1.0) *
                                         src[partner] * sn;
        }
      }
    }
  };

  // Project ONCE unscaled; S is bilinear, so every scaled quantity is a
  // scalar multiple. The image constraint applies PRE-RoPE (that is what
  // HalfBoot carries); the span comes from the RoPE'd scores.
  RealBatch q_true, k_true, v_true;
  project(wq_f, kH, 1.0, q_true);
  project(wk_f, kKv, 1.0, k_true);
  project(wv_f, kKv, 1.0, v_true);
  auto max_abs = [](const RealBatch &m) {
    double r = 0.0;
    for (const auto &h : m) {
      for (const auto &row : h) {
        for (double v : row) r = std::max(r, std::abs(v));
      }
    }
    return r;
  };
  const double qmax = max_abs(q_true), kmax = max_abs(k_true),
               vmax = max_abs(v_true);
  RealBatch q_rope = q_true, k_rope = k_true;
  rope_batch(q_rope);
  rope_batch(k_rope);
  double s_true_min = 1e300, s_true_max = -1e300;
  std::vector<std::vector<std::vector<double>>> S_true(
      32, std::vector<std::vector<double>>(kT, std::vector<double>(kT, 0.0)));
  for (int hd = 0; hd < 32; hd++) {
    for (int t = 0; t < kT; t++) {
      for (int u = 0; u < kT; u++) {
        double dot = 0.0;
        for (int d = 0; d < kD; d++) {
          dot += q_rope[hd][t][d] * k_rope[hd / 4][u][d];
        }
        S_true[hd][t][u] = dot;  // chain units per unit cq ck: sqrt(kD)
                                 // already inside (no 1/sqrt fold)
        s_true_min = std::min(s_true_min, dot);
        s_true_max = std::max(s_true_max, dot);
      }
    }
  }
  // span_true in MODEL units (the temperature): the model's S carries
  // 1/sqrt(kD), the chain's does not.
  const double span_true =
      (s_true_max - s_true_min) / std::sqrt(static_cast<double>(kD));
  // Sizing: the HalfBoot image bound first, then the score-message cap.
  const double img_max = 0.45;
  double cq = img_max / qmax, ck = img_max / kmax;
  const double s_abs_true =
      std::max(std::abs(s_true_max), std::abs(s_true_min));
  const double prod_cap = 0.36 / s_abs_true;  // chain-unit score <= 0.36
  if (cq * ck > prod_cap) {
    const double sh = std::sqrt(prod_cap / (cq * ck));
    cq *= sh;
    ck *= sh;
  }
  const double cv = std::min(1.0, img_max / vmax);
  const double cqk = cq * ck;

  // The scaled clear twins (RoPE commutes with scaling).
  RealBatch q_ref = q_rope, k_ref = k_rope, vv = v_true;
  for (auto &h : q_ref) {
    for (auto &row : h) {
      for (double &v : row) v *= cq;
    }
  }
  for (auto &h : k_ref) {
    for (auto &row : h) {
      for (double &v : row) v *= ck;
    }
  }
  for (auto &h : vv) {
    for (auto &row : h) {
      for (double &v : row) v *= cv;
    }
  }

  // ---- boot set, then the handler -------------------------------------
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest boot_req;
    bctx->AddRequiredRotations(boot_req, num_slots);
    boot.ui->PrepareRotationKey(boot_req);
  }

  // ---- full-width PC-MM emissions from the real numbers (1.5ce) -------
  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);
  boot.ui->PrepareModPackKeys(proj_small, pcmm_level);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < proj_rank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
  }
  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  for (int l = 0; l < num_x; l++) {
    std::vector<std::vector<double>> slice(
        proj_rank, std::vector<double>(proj_small, 0.0));
    for (int o = 0; o < proj_rank; o++) {
      for (int t = 0; t < kT; t++) {
        slice[o][t] = normed[static_cast<size_t>(t) * kH + l * proj_rank + o];
      }
    }
    const auto x_rec = HostRecompose(slice, proj_rank, proj_small);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                       boot.param->GetScale(pcmm_level),
                                       x_rec);
    Ciphertext<word> x_ct;
    boot.ui->Encrypt(x_ct, pt);
    std::vector<cheddar::MlweCiphertext<word>> parts;
    mlwe.ModDecomp(parts, x_ct, proj_small);
    for (auto &p : parts) x_parts.push_back(std::move(p));
  }
  ASSERT_EQ(static_cast<int>(x_parts.size()), kH);

  const double w_scale = boot.param->GetRescalePrimeProd(pcmm_level);
  // family = 0/1 (heads 0..15 / 16..31), l = channel group. kv selects
  // the GQA column block; scale folds cq/ck/cv into the encode.
  //
  // THE ENCODE IS NOT ONLINE WORK. A PlainMatrix is a weight slice in RNS
  // limbs -- it depends on nothing but the model, so a served layer
  // encodes it once and keeps it, exactly as it keeps its keys. 1.5ca and
  // 1.5ce timed it inside the emission and so reported a per-ciphertext
  // cost that is really a setup cost; here the two are separated, which
  // is what CLAUDE.md's reporting rule asks for and what tells the
  // schedule where the leg's time actually is.
  double encode_seconds = 0.0, online_seconds = 0.0, halfboot_seconds = 0.0;
  int emit_count = 0;
  auto encode_half = [&](const std::vector<double> &w, int width, double scale,
                         int l, int fam, cheddar::PlainMatrix<word> &u) {
    std::vector<double> vals(static_cast<size_t>(proj_rank) * kH, 0.0);
    for (int hh = 0; hh < 16; hh++) {
      const int head = fam * 16 + hh;
      const int col_head = (width == kKv) ? head / 4 : head;
      for (int cp = 0; cp < 16; cp++) {
        const int row = hh * 16 + cp;
        const int col = col_head * kD + l * 16 + cp;
        for (int o = 0; o < kH; o++) {
          vals[static_cast<size_t>(row) * kH + o] =
              w[static_cast<size_t>(o) * width + col] * scale;
        }
      }
    }
    const auto c0 = std::chrono::steady_clock::now();
    pcmm.EncodeMatrix(u, pcmm_level, w_scale, vals, proj_rank, kH);
    cudaDeviceSynchronize();
    encode_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - c0)
            .count();
  };
  auto emit_half = [&](const cheddar::PlainMatrix<word> &u,
                       Ciphertext<word> &out) {
    const auto p0 = std::chrono::steady_clock::now();
    std::vector<cheddar::MlweCiphertext<word>> mixed;
    pcmm.Multiply(mixed, u, x_parts);
    Ciphertext<word> packed, dropped;
    mlwe.ModPack(boot.context, packed, mixed, pack_keys);
    boot.context->Rescale(dropped, packed);
    cudaDeviceSynchronize();
    online_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - p0)
            .count();
    dropped.SetNumSlots(num_slots);
    const auto h0 = std::chrono::steady_clock::now();
    bctx->HalfBoot(out, dropped, boot.ui->GetEvkMap());
    cudaDeviceSynchronize();
    halfboot_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - h0)
            .count();
    emit_count++;
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8), v_a(8),
      v_b(8);
  for (int l = 0; l < 8; l++) {
    cheddar::PlainMatrix<word> u;
    encode_half(wq_f, kH, cq, l, 0, u);
    emit_half(u, q_a[l]);
    encode_half(wq_f, kH, cq, l, 1, u);
    emit_half(u, q_b[l]);
    encode_half(wk_f, kKv, ck, l, 0, u);
    emit_half(u, k_a[l]);
    encode_half(wk_f, kKv, ck, l, 1, u);
    emit_half(u, k_b[l]);
    encode_half(wv_f, kKv, cv, l, 0, u);
    emit_half(u, v_a[l]);
    encode_half(wv_f, kKv, cv, l, 1, u);
    emit_half(u, v_b[l]);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::cout << "the real-weight projections: weight encode "
            << encode_seconds / std::max(emit_count, 1)
            << " s/ct (ONE-TIME, model-only), online mix+pack+rescale "
            << online_seconds / std::max(emit_count, 1) << " s/ct, HalfBoot "
            << halfboot_seconds / std::max(emit_count, 1) << " s/ct over "
            << emit_count << " emissions" << std::endl;

  // The boundary constant off one ciphertext, as always -- against the
  // clear K projection (un-RoPE'd, scaled).
  double hb_const = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, k_a[0]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    double rsum = 0.0;
    int counted = 0;
    for (int t = 0; t < kT; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const double want = ck * k_true[hh / 4][t][cp];
          if (std::abs(want) < 0.02) continue;
          rsum += slots[door0(t, cp, hh)].real() / want;
          counted++;
        }
      }
    }
    ASSERT_GT(counted, 1000);
    hb_const = rsum / counted;
    ASSERT_LT(std::abs(std::log2(std::abs(hb_const) * 32.0)), 0.5);
  }

  typename cheddar::CiSinCAttention<word>::Config acfg;
  acfg.restore = 1.0 / hb_const;
  acfg.rope_base = rope_theta;
  cheddar::CiSinCAttention<word> attn(bctx, swtch.context, small.context,
                                      lifted.context, acfg);
  const auto &layout = attn.GetLayout();

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

  // ---- calibration: chain units are cqk x S_true --------------------
  std::vector<std::vector<std::vector<double>>> S(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
  double smin = 1e300, smax = -1e300;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int col = 0; col < layout.dim; col++) {
        const double v = cqk * S_true[head][row][col];
        S[head][row][col] = v;
        smin = std::min(smin, v);
        smax = std::max(smax, v);
      }
    }
  }
  const double span = smax - smin;
  std::vector<std::vector<double>> row_shift(
      layout.lanes, std::vector<double>(layout.dim, -1e300));
  std::vector<std::vector<double>> row_norm(
      layout.lanes, std::vector<double>(layout.dim, 0.0));
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      for (int k = 0; k <= row; k++) {
        row_shift[lane][row] =
            std::max(row_shift[lane][row], S[head][row][k]);
      }
      for (int k = 0; k <= row; k++) {
        row_norm[lane][row] += std::exp(
            span_true * (S[head][row][k] - row_shift[lane][row]) / span);
      }
    }
  }

  typename cheddar::CiSinCAttention<word>::SoftMaxCalibration calib;
  calib.m_eff = span_true;  // fidelity: exp(m_eff S_chain / span) = exp(S)
  calib.span = span;
  calib.shift = smax;
  calib.norm_lo = 0.9;
  calib.norm_hi = 1.1;
  calib.exp_degree = 15;  // hb = span_true / 4 ~ 6.1: deg 7 is 3.2e-03
  calib.inv_degree = 7;   // the ratio interval affords it; funds exp
  calib.causal = true;
  calib.row_shift = row_shift;
  calib.row_norm = row_norm;
  attn.PrepareSoftMax(calib);

  // ---- the leg ---------------------------------------------------------
  std::vector<Ciphertext<word>> s0;
  attn.Scores(s0, q_a, q_b, k_a, k_b, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const double carried = s0[0].GetScale() / boot.param->base_scale_;
  ASSERT_LT(carried * std::max(std::abs(smax), std::abs(smin)), 0.95)
      << "the size_q/size_k fold missed EvalMod's range";

  std::vector<Ciphertext<word>> scores(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    s0[bi].SetNumSlots(num_slots);
    bctx->Boot(scores[bi], s0[bi], boot.ui->GetEvkMap());
  }
  std::vector<Ciphertext<word>> P;
  attn.SoftMax(P, scores, carried, boot.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<std::vector<std::vector<double>>> P_dec(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
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
          P_dec[rev(lane, 5)][row][column] = slots[slot].real();
        }
      }
    }
  }
  double masked_resid = 0.0, rowsum_dev = 0.0;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      double rs = 0.0;
      for (int k = 0; k < layout.dim; k++) {
        if (k <= row) {
          rs += P_dec[head][row][k];
        } else {
          masked_resid =
              std::max(masked_resid, std::abs(P_dec[head][row][k]));
        }
      }
      rowsum_dev = std::max(rowsum_dev, std::abs(rs - 1.0));
    }
  }

  std::vector<Ciphertext<word>> out;
  attn.Values(out, P, v_a, v_b, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // ---- against the true causal attention ------------------------------
  double worst_true = 0.0, worst_incoming = 0.0, worst_plain = 0.0;
  double transposed = 0.0, biggest = 0.0;
  for (int bi = 0; bi < layout.num_cts; bi++) {
    ASSERT_EQ(boot.param->NPToLevel(out[bi].GetNP()), 0);
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, out[bi]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int lane = 0; lane < layout.lanes; lane++) {
      const int head = rev(lane, 5);
      for (int row = 0; row < layout.dim; row++) {
        std::vector<double> p_true(layout.dim, 0.0);
        std::vector<double> p_plain(layout.dim, 0.0);
        double zsum = 0.0, gzsum = 0.0;
        for (int k = 0; k < layout.dim; k++) {
          // TRUE temperature: span_true / span converts chain units back.
          p_plain[k] = std::exp(span_true * (S[head][row][k] - smax) / span);
          gzsum += p_plain[k];
          if (k <= row) {
            p_true[k] = std::exp(
                span_true * (S[head][row][k] - row_shift[lane][row]) /
                span);
            zsum += p_true[k];
          }
        }
        for (double &pv : p_true) pv /= zsum;
        for (double &pv : p_plain) pv /= gzsum;
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          double want = 0.0, want_in = 0.0, want_plain = 0.0, want_t = 0.0;
          for (int k = 0; k < layout.dim; k++) {
            want += p_true[k] * vv[head / 4][k][column];
            want_in += P_dec[head][row][k] * vv[head / 4][k][column];
            want_plain += p_plain[k] * vv[head / 4][k][column];
            want_t += P_dec[head][column][k] * vv[head / 4][k][row];
          }
          int ct_idx, slot, copy_slot;
          layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
          const double got = slots[slot].real();
          biggest = std::max(biggest, std::abs(want));
          worst_true = std::max(worst_true, std::abs(got - want));
          worst_incoming =
              std::max(worst_incoming, std::abs(got - want_in));
          worst_plain = std::max(worst_plain, std::abs(got - want_plain));
          transposed = std::max(transposed, std::abs(got - want_t));
        }
      }
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "the leg on the REAL weights (span_true " << span_true
            << ", cq = ck = " << cq << ", exp deg 15 / invsqrt deg 7): vs "
            << "the true causal attention " << worst_true
            << ", vs decrypted P x V " << worst_incoming
            << " (|output| <= " << biggest << ", carried " << carried << ")"
            << std::endl;
  std::cout << "  P at masked addresses <= " << masked_resid
            << ", live row sums off one by " << rowsum_dev << std::endl;
  std::cout << "  controls: un-masked softmax " << worst_plain
            << ", transposed " << transposed << std::endl;

  EXPECT_LT(worst_true, 5e-2 * std::max(biggest, 1.0))
      << "the real-weight leg did not land the attention output";
  EXPECT_LT(worst_incoming, 2e-2 * std::max(biggest, 1.0))
      << "the second contraction added more than its own floor";
  EXPECT_LT(masked_resid, 1e-3) << "the causal mask leaked into P";
  // The real-data noise datum: the projection floor (1.42e-04, 1.5ca)
  // rides the affine's 2/span amplification into exp at hb ~ 6, so the
  // row sums carry ~2 hb u-noise ~ 1e-2. A tighter number needs a lower
  // PC-MM noise floor, not a better walk.
  EXPECT_LT(rowsum_dev, 5e-2);
  EXPECT_GT(worst_plain, 1e-2 * std::max(biggest, 1.0))
      << "the causal and plain softmax agree, so the mask did nothing";
  EXPECT_GT(transposed, 1e-2 * std::max(biggest, 1.0));
}

// What the transport's height costs (Doing.md 1.5ci).
//
// The leg's transport -- the merge rotation, the 63-diagonal exchange's
// whole BSGS, and K's 32 masked rotations -- ran at levels 18/17 for one
// reason only: that is where RoPE happened to leave the half-images. But
// a key switch pays for the limbs it carries, and nothing between the
// landing and the descent needs the height, so `Config::exchange_level`
// is a dial (`Merge` now drops the halves onto it) bounded below only by
// ci16_35's num_accum == 1 zone -- levels 0..6, which a hoisted
// transform may not enter (Doing.md 1.5bt).
//
// This measures the dial WITHOUT paying for the leg's converters (which
// are 730 s of host build and dominate every leg test): the exchange is
// compiled at each candidate level and evaluated over the leg's own
// ciphertext count, beside a bare HRot and the cross pattern at the same
// levels. It is a cost measurement, not a correctness one -- the walk's
// arithmetic is level-independent by construction (LevelDown preserves
// the scale, and gamma is stated against cross_level) and
// TheLibraryLegReproducesTheReference is what proves that.
TEST(CiBootSet, TheTransportHeightIsADial) {
  Ring boot(kBootParam);
  const int n = boot.Degree();
  const int num_cts = 8;      // one tensor's merged half-images
  const int tensors = 3;      // Q, K, V
  const int window = 31 * 127;

  auto exch = [](int s) {
    const int a = (s >> 7) & 31, b = s & 31;
    return (s & ~((31 << 7) | 31)) | (b << 7) | a;
  };
  cheddar::StripedMatrix em(n, n);
  for (int r = 0; r < n; r++) {
    const int in = exch(r);
    const int off = ((in - r) % n + n) % n;
    em.try_emplace(off, n, Complex(0.0, 0.0));
    em[off][r] = Complex(1.0, 0.0);
  }
  ASSERT_EQ(em.GetNumDiag(), 63);

  const std::vector<int> levels = {18, 12, 8, 7};
  std::mt19937_64 gen(0xD1A1);
  std::uniform_real_distribution<double> dist(-0.3, 0.3);
  std::vector<Complex> msg(n, Complex(0.0, 0.0));
  for (auto &v : msg) v = Complex(dist(gen), 0.0);

  std::cout << "the transport's height, on this box:" << std::endl;
  for (int level : levels) {
    const auto b0 = std::chrono::steady_clock::now();
    cheddar::LinearTransform<word> lt(
        boot.context, em, level,
        boot.param->GetRescalePrimeProd(level), 8, 8,
        /*pre_rotation=*/-window, /*additional_pt_rot=*/window);
    const auto b1 = std::chrono::steady_clock::now();
    {
      EvkRequest req;
      lt.AddRequiredRotations(req);
      req.AddRequest(n - window, level - 1);
      req.AddRequest(n - 128, level);
      std::set<int> idxs;
      for (int u = 0; u < 4; u++) {
        for (int v = 0; v < 8; v++) {
          const int rot = (v - u) * 128;
          if (rot != 0) idxs.insert((rot % n + n) % n);
        }
      }
      for (int idx : idxs) req.AddRequest(idx, level - 1);
      boot.ui->PrepareRotationKey(req);
    }
    const auto &evk = boot.ui->GetEvkMap();

    Plaintext<word> pt;
    boot.context->encoder_.Encode(pt, level, boot.param->GetScale(level), msg);
    std::vector<Ciphertext<word>> cts(num_cts);
    for (auto &ct : cts) boot.ui->Encrypt(ct, pt);

    // warm-up, then the timed pass: one tensor's exchange plus its merge.
    Ciphertext<word> warm;
    lt.Evaluate(boot.context, warm, cts[0], evk);
    cudaDeviceSynchronize();

    const auto e0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < tensors; rep++) {
      for (auto &ct : cts) {
        Ciphertext<word> shifted, swapped;
        lt.Evaluate(boot.context, shifted, ct, evk);
        boot.context->HRot(swapped, shifted,
                           evk.GetRotationKey(n - window), n - window);
      }
    }
    cudaDeviceSynchronize();
    const auto e1 = std::chrono::steady_clock::now();
    // the merge, one rotation per ciphertext per tensor
    for (int rep = 0; rep < tensors; rep++) {
      for (auto &ct : cts) {
        Ciphertext<word> moved;
        boot.context->HRot(moved, ct, evk.GetRotationKey(n - 128), n - 128);
      }
    }
    cudaDeviceSynchronize();
    const auto e2 = std::chrono::steady_clock::now();
    // one bare key switch, for the per-limb curve
    const int reps = 32;
    for (int rep = 0; rep < reps; rep++) {
      Ciphertext<word> moved;
      boot.context->HRot(moved, cts[0], evk.GetRotationKey(n - 128), n - 128);
    }
    cudaDeviceSynchronize();
    const auto e3 = std::chrono::steady_clock::now();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cout << "  level " << level << ": exchange+close "
              << ms(e0, e1) / (tensors * num_cts) << " ms/ct, merge "
              << ms(e1, e2) / (tensors * num_cts) << " ms/ct, bare HRot "
              << ms(e2, e3) / reps << " ms, transform build "
              << ms(b0, b1) / 1000.0 << " s" << std::endl;
  }
  std::cout << "  (the leg spends " << tensors * num_cts
            << " exchanges + " << tensors * num_cts
            << " merges + 64 cross rotations per cycle)" << std::endl;
}

// What the converter's BSGS split costs (Doing.md 1.5cj).
//
// The leg's dominant ONLINE cost is its conversions -- 48 forwards and 16
// inverses per cycle at ~29 ms each, against ~0.3 s of chain and ~6 ms of
// cross -- and every one of them is a 2048-diagonal BSGS whose split
// `CiSinCConverter` picks as sqrt(num_diag) = 64 x 32. That split balances
// a baby step against a giant step, and on R+ they do not cost the same:
// 1.5be measured ~7:1 (a baby step rides the shared ModUp as one fused key
// multiply; a giant step pays its own ModDown + ModUp), which is why
// `BSGSSplit` compiles the bootstrap's CI phases at sqrt(7 num_diag). It
// caps those at 16 baby steps for GSFusedComplexKernel's registers -- and
// that cap does NOT reach here, because these are single-ciphertext
// LinearTransforms and the fused complex giant step lives in
// ComplexLinearTransform. So the converter has been paying a balanced
// split for an unbalanced cost with nothing stopping it.
//
// Whether sqrt(7 D) actually wins at the converter's own level (3, where
// the limb counts and therefore the ModDown share are nothing like the
// bootstrap's) is a measurement. This is it, at the leg's own shape
// (sub_degree 32, chain layout, forward): the same conversion compiled at
// several baby-step counts, timed over the leg's own ciphertext count --
// and CHECKED AGAINST EACH OTHER, because a BSGS split may reschedule the
// arithmetic but must not change it.
TEST(CiBootSet, TheConverterSplitIsAMeasurement) {
  // The leg's own arrangement: the operand is a BOOT-ring ciphertext and
  // crosses keylessly into the switching ring (Doing.md 1.5bt), which is
  // why the switching ring is built on the boot secret and why the scale
  // comes from the boot parameter -- the switching ring has no scale of
  // its own at this level.
  Ring boot(kBootParam);
  Ring swtch(kBootSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kBootSmallParam);
  const int sub_degree = 32;
  const int fwd_level = 3;
  const int n = boot.Degree();
  const CiSwitchedCcmmLayout layout(n, small.Degree(), sub_degree);
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.num_cts, 8);

  std::mt19937_64 gen(0xB5B5);
  std::uniform_real_distribution<double> dist(-0.2, 0.2);
  std::vector<Complex> msg(n, Complex(0.0, 0.0));
  for (int row = 0; row < layout.dim; row++) {
    for (int column = 0; column < layout.dim; column++) {
      for (int lane = 0; lane < layout.lanes; lane++) {
        int ct_idx, slot, copy_slot;
        layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
        if (ct_idx == 0) msg[slot] = Complex(dist(gen), 0.0);
      }
    }
  }
  Plaintext<word> pt;
  boot.context->encoder_.Encode(pt, fwd_level,
                                boot.param->GetScale(fwd_level), msg);

  const std::vector<int> splits = {64, 256, 512, 1024};
  std::vector<double> first_out;
  std::cout << "the converter's split, at the leg's own shape (sub_degree "
            << sub_degree << ", forward, level " << fwd_level << "):"
            << std::endl;
  for (int bs : splits) {
    const auto b0 = std::chrono::steady_clock::now();
    CiSinCConverter<word> conv(swtch.context, sub_degree, fwd_level,
                               /*inverse_level=*/-1, &layout,
                               /*forward_premap=*/nullptr, bs);
    const auto b1 = std::chrono::steady_clock::now();
    {
      EvkRequest req;
      conv.AddRequiredRotations(req);
      swtch.ui->PrepareRotationKey(req);
    }
    Ciphertext<word> in;
    boot.ui->Encrypt(in, pt);

    Ciphertext<word> warm;
    conv.SlotToSinC(swtch.context, warm, in, swtch.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    const int reps = layout.num_cts;  // one tensor's worth
    const auto e0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
      Ciphertext<word> out;
      conv.SlotToSinC(swtch.context, out, in, swtch.ui->GetEvkMap());
    }
    cudaDeviceSynchronize();
    const auto e1 = std::chrono::steady_clock::now();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // The split reschedules; it must not compute anything else.
    Plaintext<word> back;
    boot.ui->Decrypt(back, warm);
    std::vector<double> coeffs;
    boot.context->encoder_.DecodeCoeff(coeffs, back);
    if (first_out.empty()) {
      first_out = coeffs;
    } else {
      double worst = 0.0, biggest = 0.0;
      for (size_t i = 0; i < coeffs.size(); i++) {
        worst = std::max(worst, std::abs(coeffs[i] - first_out[i]));
        biggest = std::max(biggest, std::abs(first_out[i]));
      }
      EXPECT_LT(worst, 1e-5 * std::max(biggest, 1.0))
          << "baby steps " << bs << " computed something else";
    }

    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    EvkRequest keys;
    conv.AddRequiredRotations(keys);
    std::cout << "  baby steps " << bs << " (giant "
              << ((2048 + bs - 1) / bs) << "): " << ms(e0, e1) / reps
              << " ms/ct, build " << ms(b0, b1) / 1000.0 << " s, "
              << keys.size() << " rotation keys" << std::endl;
  }
  std::cout << "  (the leg runs 48 forwards + 16 inverses per cycle)"
            << std::endl;
}

// What ModPack's auxiliary basis costs, and what it buys (Doing.md 1.5ck).
//
// A projection emission is 512 key switches -- ModPack's, one per module
// component -- and the leg runs 48 emissions, so this is the single
// largest population of key switches in the layer by an order of
// magnitude. Every one of them carries `alpha_` auxiliary primes, a width
// sized for the DEEPEST switch in the parameter set, while ModPack runs at
// the shallowest level there is. `PrepareModPackKeys` takes a `num_aux`
// for exactly this reason (it calls `Context::PrepareNarrowKeySwitch` for
// you), and on the ordinary leg the narrow basis was worth 579 ms of a
// 15 s block.
//
// It is opt-in because it changes the key-switch noise: P must still
// exceed the digit, and the caller owns that. So this measures BOTH sides
// on the CI leg's own shape -- the ModPack time and the emission's error
// against the host projection -- at every basis width the parameter set
// allows. The projection error is the leg's noise floor (1.42e-04 through
// the transport, 1.5ca), so a width that doubles it is not a saving.
//
// No converters, no chain, no bootstrap: one X ciphertext, one weight
// slice, and the pack loop.
TEST(CiBootSet, TheModPackBasisIsAChoice) {
  Ring boot(kBootParam);
  const int pcmm_level = 1;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;  // 128
  const int in_ch = proj_rank;
  const int alpha = boot.param->alpha_;
  ASSERT_GE(alpha, 1);

  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);

  std::mt19937_64 gen(0xA0A0);
  std::uniform_real_distribution<double> xd(-1.0, 1.0);
  std::vector<std::vector<double>> x_comp(
      in_ch, std::vector<double>(proj_small, 0.0));
  for (auto &ch : x_comp) {
    for (auto &v : ch) v = xd(gen);
  }
  const double wa = 0.24 / std::sqrt(static_cast<double>(in_ch));
  std::uniform_real_distribution<double> wd(-wa, wa);
  std::vector<double> vals(static_cast<size_t>(proj_rank) * in_ch, 0.0);
  for (int row = 0; row < 256; row++) {  // the half-density contract
    for (int o = 0; o < in_ch; o++) {
      vals[static_cast<size_t>(row) * in_ch + o] = wd(gen);
    }
  }
  // The host reference: row I of the product, as a token polynomial, then
  // the banded recomposition ModPack is supposed to produce.
  std::vector<std::vector<double>> p(proj_rank,
                                     std::vector<double>(proj_small, 0.0));
  for (int row = 0; row < proj_rank; row++) {
    for (int t = 0; t < proj_small; t++) {
      double s = 0.0;
      for (int o = 0; o < in_ch; o++) {
        s += vals[static_cast<size_t>(row) * in_ch + o] * x_comp[o][t];
      }
      p[row][t] = s;
    }
  }
  const auto want = HostRecompose(p, proj_rank, proj_small);

  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  {
    const auto x_rec = HostRecompose(x_comp, proj_rank, proj_small);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                       boot.param->GetScale(pcmm_level),
                                       x_rec);
    Ciphertext<word> x_ct;
    boot.ui->Encrypt(x_ct, pt);
    mlwe.ModDecomp(x_parts, x_ct, proj_small);
  }
  cheddar::PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, pcmm_level,
                    boot.param->GetRescalePrimeProd(pcmm_level), vals,
                    proj_rank, in_ch);
  std::vector<cheddar::MlweCiphertext<word>> mixed;
  pcmm.Multiply(mixed, u, x_parts);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // beta = ceil(num_q / num_aux), and beta > 1 drops ModPack off the
  // grouped mod-up -- so the useful floor is num_aux = num_q, and this
  // prints the number the rule needs.
  std::cout << "ModPack's auxiliary basis (rank " << proj_rank
            << " key switches per emission, alpha " << alpha << ", num_q at "
            << "level " << pcmm_level << " = "
            << boot.param->LevelToNP(pcmm_level).GetNumQ() << "):"
            << std::endl;
  for (int num_aux = alpha; num_aux >= 1; num_aux--) {
    boot.ui->PrepareModPackKeys(proj_small, pcmm_level, num_aux);
    std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
    for (int j = 0; j < proj_rank; j++) {
      pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
    }
    Ciphertext<word> packed;
    mlwe.ModPack(boot.context, packed, mixed, pack_keys);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    const int reps = 4;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
      Ciphertext<word> tmp;
      mlwe.ModPack(boot.context, tmp, mixed, pack_keys);
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    Ciphertext<word> dropped;
    boot.context->Rescale(dropped, packed);
    Plaintext<word> back;
    boot.ui->Decrypt(back, dropped);
    std::vector<double> got;
    boot.context->encoder_.DecodeCoeff(got, back);
    double worst = 0.0, biggest = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
      worst = std::max(worst, std::abs(got[i] - want[i]));
      biggest = std::max(biggest, std::abs(want[i]));
    }
    std::cout << "  num_aux " << num_aux << ": ModPack "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() /
                     reps
              << " ms, emission error " << worst << " (|value| <= " << biggest
              << ")" << std::endl;
    EXPECT_LT(worst, 1e-2 * std::max(biggest, 1.0))
        << "num_aux " << num_aux << " broke the emission -- P no longer "
           "exceeds the digit";
  }
}

// What the PC-MM's row tile buys (Doing.md 1.5cl).
//
// 1.5ch put the layer's cost where it actually is: 48 projections at
// 751 ms of ONLINE work each -- ~36 s against ~2 s for all of the
// attention arithmetic -- and inside a projection ModPack is 22 ms, so
// what a projection costs is the PP-MM. `PcmmAccum` gave one thread one
// output row, which means the ciphertext operand is read once PER OUTPUT
// ROW: 512 passes over the same cols * degree words at the layer's shape,
// on a product that is memory-bound long before it is arithmetic-bound.
// `PcmmAccumTiled` carries TILE rows per thread, so one src load serves
// TILE multiply-adds; the arithmetic and the accumulation order within a
// row are untouched, so it is bit-identical, and the zero test survives
// (a tile skips the load only when every row of it has a zero, which is
// what the half-density weight slices produce in bulk).
//
// The tile is `CHEDDAR_PCMM_ROW_TILE`, read once per process, so this
// test measures ONE tile per run and the sweep is a loop over processes.
// Correctness is checked every run against the host product, not against
// another tile, so a run stands on its own.
TEST(CiBootSet, ThePcmmRowTileIsAMeasurement) {
  Ring boot(kBootParam);
  const int pcmm_level = 1;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;  // 128
  const int num_x = 8;
  const int in_ch = num_x * proj_rank;   // 4096, the layer's real width

  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);

  std::mt19937_64 gen(0xB171);
  std::uniform_real_distribution<double> xd(-1.0, 1.0);
  std::vector<std::vector<double>> x_comp(
      in_ch, std::vector<double>(proj_small, 0.0));
  for (auto &ch : x_comp) {
    for (auto &v : ch) v = xd(gen);
  }
  const double wa = 0.24 / std::sqrt(static_cast<double>(in_ch));
  std::uniform_real_distribution<double> wd(-wa, wa);
  std::vector<double> vals(static_cast<size_t>(proj_rank) * in_ch, 0.0);
  for (int row = 0; row < 256; row++) {  // the half-density contract
    for (int o = 0; o < in_ch; o++) {
      vals[static_cast<size_t>(row) * in_ch + o] = wd(gen);
    }
  }
  std::vector<std::vector<double>> pref(proj_rank,
                                        std::vector<double>(proj_small, 0.0));
  for (int row = 0; row < proj_rank; row++) {
    for (int t = 0; t < proj_small; t++) {
      double s = 0.0;
      for (int o = 0; o < in_ch; o++) {
        s += vals[static_cast<size_t>(row) * in_ch + o] * x_comp[o][t];
      }
      pref[row][t] = s;
    }
  }
  const auto want = HostRecompose(pref, proj_rank, proj_small);

  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  for (int l = 0; l < num_x; l++) {
    std::vector<std::vector<double>> slice(
        x_comp.begin() + l * proj_rank, x_comp.begin() + (l + 1) * proj_rank);
    const auto x_rec = HostRecompose(slice, proj_rank, proj_small);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                       boot.param->GetScale(pcmm_level),
                                       x_rec);
    Ciphertext<word> x_ct;
    boot.ui->Encrypt(x_ct, pt);
    std::vector<cheddar::MlweCiphertext<word>> parts;
    mlwe.ModDecomp(parts, x_ct, proj_small);
    for (auto &p : parts) x_parts.push_back(std::move(p));
  }
  ASSERT_EQ(static_cast<int>(x_parts.size()), in_ch);

  cheddar::PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, pcmm_level,
                    boot.param->GetRescalePrimeProd(pcmm_level), vals,
                    proj_rank, in_ch);

  std::vector<cheddar::MlweCiphertext<word>> mixed;
  pcmm.Multiply(mixed, u, x_parts);  // warm-up
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const int reps = 3;
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) {
    std::vector<cheddar::MlweCiphertext<word>> tmp;
    pcmm.Multiply(tmp, u, x_parts);
  }
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Correctness on its own terms: pack and read against the host product.
  boot.ui->PrepareModPackKeys(proj_small, pcmm_level);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < proj_rank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
  }
  Ciphertext<word> packed, dropped;
  mlwe.ModPack(boot.context, packed, mixed, pack_keys);
  boot.context->Rescale(dropped, packed);
  Plaintext<word> back;
  boot.ui->Decrypt(back, dropped);
  std::vector<double> got;
  boot.context->encoder_.DecodeCoeff(got, back);
  double worst = 0.0, biggest = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    worst = std::max(worst, std::abs(got[i] - want[i]));
    biggest = std::max(biggest, std::abs(want[i]));
  }

  const char *env = std::getenv("CHEDDAR_PCMM_ROW_TILE");
  std::cout << "PC-MM at the layer's shape (" << proj_rank << " x " << in_ch
            << " U over " << in_ch << " MLWE parts, rank " << proj_rank
            << "), CHEDDAR_PCMM_ROW_TILE=" << (env ? env : "unset (8)")
            << ": Multiply "
            << std::chrono::duration<double, std::milli>(t1 - t0).count() /
                   reps
            << " ms, emission error " << worst << " (|value| <= " << biggest
            << ")" << std::endl;
  EXPECT_LT(worst, 1e-2 * std::max(biggest, 1.0))
      << "the tiled product does not agree with the host";
}

// `CoeffLinearLeg` implements only `Project`; the two ciphertext-ciphertext
// products are pure virtual on purpose, so nothing falls back to a stand-in.
// The whole-layer test below uses it for the O projection, whose operands
// come out of the seam.
class ProjectOnlyLegCi : public cheddar::CoeffLinearLeg<word> {
 public:
  using cheddar::CoeffLinearLeg<word>::CoeffLinearLeg;
  void Scores(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double,
              const std::vector<double> &) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLegCi: no Scores here");
  }
  void Values(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLegCi: no Values here");
  }
  void LocateScore(int, int, int, int &, int &) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLegCi: no score layout here");
  }
};

TEST(CiBootSet, TheWholeLayerRunsOnTheRealSubring) {
  Ring boot(kBootParam);
  // HELD BY POINTER SO THEY CAN BE RELEASED. A whole layer needs two
  // bootstrap key sets alive -- the leg's, at slack zero, and the FFN's, at
  // slack nine -- and two of them plus the leg's three converters, its 512
  // big-ring ModPack keys and the lifted ring's automorphisms do not fit in
  // 80 GB: the run dies with cudaErrorMemoryAllocation right after the seam's
  // transforms are built. Nothing after `Values` needs the switching, product
  // or lifted rings, or the converters, so they go before the second
  // BootContext is made.
  auto swtch = std::make_unique<Ring>(kBootSwitchParam,
                                      boot.ui->GetSecretCoeffs());
  auto small = std::make_unique<Ring>(kBootSmallParam);
  auto lifted = std::make_unique<Ring>(
      kBootLiftedParam,
      CiLiftHandler<word>::LiftSecret(small->ui->GetSecretCoeffs()));

  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  const int pcmm_level = 1;
  const int chain_level = 2;
  const int n = boot.Degree();
  const int proj_rank = 512;
  const int proj_small = n / proj_rank;
  const int in_ch = proj_rank;
  const int num_slots = boot.param->MaxNumSlots();

  auto rev = [](int v, int bits) {
    int r = 0;
    for (int j = 0; j < bits; j++) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    return r;
  };
  auto door0 = [&](int t, int c, int i) {
    return (rev(c % 16, 4) << 12) | (rev(i, 5) << 7) | rev(t, 7);
  };

  // The boot set first: the handler's constructor reads GetStCInputScale.
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest boot_req;
    bctx->AddRequiredRotations(boot_req, num_slots);
    boot.ui->PrepareRotationKey(boot_req);
  }

  // ---- the projections: real PC-MM emissions, as in 1.5ca/1.5cb ---------
  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);
  boot.ui->PrepareModPackKeys(proj_small, pcmm_level);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < proj_rank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(proj_rank, j));
  }

  std::mt19937_64 gen(0xB141);
  std::uniform_real_distribution<double> xd(-1.0, 1.0);
  std::vector<std::vector<double>> x_comp(
      in_ch, std::vector<double>(proj_small, 0.0));
  // HALF DENSITY ON THE RESIDUAL STREAM. Components >= rank/2 are the dead
  // half of a banded image (1.5by), and the O projection's output is
  // half-density by construction, so the stream it adds back to has to be
  // too or the residual would mix two channels.
  for (int o = 0; o < proj_rank / 2; o++) {
    for (auto &v : x_comp[o]) v = xd(gen);
  }
  const int kDim = 128, kLanes = 32;
  const double wa = 0.24 / std::sqrt(static_cast<double>(in_ch));
  std::uniform_real_distribution<double> wd(-wa, wa);
  using W = std::vector<std::vector<std::vector<double>>>;
  W wq(kLanes, std::vector<std::vector<double>>(
                   kDim, std::vector<double>(in_ch, 0.0)));
  W wk = wq, wv = wq;
  for (int i = 0; i < kLanes; i++) {
    for (int c = 0; c < kDim; c++) {
      for (int o = 0; o < in_ch; o++) {
        wq[i][c][o] = wd(gen);
        wk[i][c][o] = wd(gen);
        wv[i][c][o] = wd(gen);
      }
    }
  }
  auto project = [&](const W &w) {
    RealBatch r(kLanes, std::vector<std::vector<double>>(
                            kDim, std::vector<double>(kDim, 0.0)));
    for (int i = 0; i < kLanes; i++) {
      for (int t = 0; t < kDim; t++) {
        for (int c = 0; c < kDim; c++) {
          double s = 0.0;
          for (int o = 0; o < in_ch; o++) s += w[i][c][o] * x_comp[o][t];
          r[i][t][c] = s;
        }
      }
    }
    return r;
  };
  const RealBatch q = project(wq);
  const RealBatch kk = project(wk);
  const RealBatch vv = project(wv);

  const int half = kDim / 2;
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(10000.0, -2.0 * m / kDim);
  }
  auto rope_host = [&](const RealBatch &x) {
    RealBatch r = x;
    for (int t = 0; t < kLanes; t++) {
      for (int i = 0; i < kDim; i++) {
        for (int m = 0; m < half; m++) {
          const double c = std::cos(i * theta[m]), sn = std::sin(i * theta[m]);
          r[t][i][m] = x[t][i][m] * c - x[t][i][m + half] * sn;
          r[t][i][m + half] = x[t][i][m + half] * c + x[t][i][m] * sn;
        }
      }
    }
    return r;
  };
  const RealBatch q_ref = rope_host(q);
  const RealBatch k_ref = rope_host(kk);

  std::vector<cheddar::MlweCiphertext<word>> x_parts;
  {
    const auto x_rec = HostRecompose(x_comp, proj_rank, proj_small);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, pcmm_level,
                                       boot.param->GetScale(pcmm_level),
                                       x_rec);
    Ciphertext<word> x_ct;
    boot.ui->Encrypt(x_ct, pt);
    mlwe.ModDecomp(x_parts, x_ct, proj_small);
  }

  const double w_scale = boot.param->GetRescalePrimeProd(pcmm_level);
  auto emit_half = [&](const W &w, int l, int fam, Ciphertext<word> &out) {
    std::vector<double> vals(static_cast<size_t>(proj_rank) * in_ch, 0.0);
    for (int hh = 0; hh < 16; hh++) {
      for (int cp = 0; cp < 16; cp++) {
        const int row = hh * 16 + cp;
        for (int o = 0; o < in_ch; o++) {
          vals[static_cast<size_t>(row) * in_ch + o] =
              w[fam * 16 + hh][l * 16 + cp][o];
        }
      }
    }
    cheddar::PlainMatrix<word> u;
    pcmm.EncodeMatrix(u, pcmm_level, w_scale, vals, proj_rank, in_ch);
    std::vector<cheddar::MlweCiphertext<word>> mixed;
    pcmm.Multiply(mixed, u, x_parts);
    Ciphertext<word> packed, dropped;
    mlwe.ModPack(boot.context, packed, mixed, pack_keys);
    boot.context->Rescale(dropped, packed);
    dropped.SetNumSlots(num_slots);
    bctx->HalfBoot(out, dropped, boot.ui->GetEvkMap());
  };

  std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8), v_a(8),
      v_b(8);
  for (int l = 0; l < 8; l++) {
    emit_half(wq, l, 0, q_a[l]);
    emit_half(wq, l, 1, q_b[l]);
    emit_half(wk, l, 0, k_a[l]);
    emit_half(wk, l, 1, k_b[l]);
    emit_half(wv, l, 0, v_a[l]);
    emit_half(wv, l, 1, v_b[l]);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_NEAR(q_a[0].GetScale() / bctx->GetStCInputScale(), 1.0, 1e-9);

  double hb_const = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, k_a[0]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    double rsum = 0.0;
    int counted = 0;
    for (int t = 0; t < kDim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const double want = kk[hh][t][cp];
          if (std::abs(want) < 0.02) continue;
          rsum += slots[door0(t, cp, hh)].real() / want;
          counted++;
        }
      }
    }
    hb_const = rsum / counted;
    ASSERT_LT(std::abs(std::log2(std::abs(hb_const) * 32.0)), 0.5);
  }

  // ---- the handler, its keys ------------------------------------------
  typename cheddar::CiSinCAttention<word>::Config acfg;
  acfg.restore = 1.0 / hb_const;
  const auto t0 = std::chrono::steady_clock::now();
  auto attn_p = std::make_unique<cheddar::CiSinCAttention<word>>(
      bctx, swtch->context, small->context, lifted->context, acfg);
  const auto t1 = std::chrono::steady_clock::now();
  const auto layout = attn_p->GetLayout();  // by value: attn_p dies below

  swtch->ui->PrepareRingSwitchKey(small->Degree(), small->ui->GetSecretCoeffs(),
                                 chain_level);
  swtch->ui->PrepareInverseRingSwitchKey(small->Degree(),
                                        small->ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : attn_p->LiftedRotationIndices()) {
    lifted->ui->PrepareRotationKey(idx, chain_level);
  }
  {
    EvkRequest req;
    attn_p->AddSwitchRotations(req);
    swtch->ui->PrepareRotationKey(req);
  }
  {
    EvkRequest req;
    attn_p->AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  typename cheddar::CiSinCAttention<word>::Keys keys;
  keys.boot = &boot.ui->GetEvkMap();
  keys.swtch = &swtch->ui->GetEvkMap();
  keys.lifted = &lifted->ui->GetEvkMap();
  keys.ring_switch = &swtch->ui->GetRingSwitchKey(layout.rank);
  keys.inverse_ring_switch = &swtch->ui->GetInverseRingSwitchKey(layout.rank);

  // ---- host calibration off the clear twin ----------------------------
  const double m_eff = 8.0;
  std::vector<std::vector<std::vector<double>>> S(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
  double smin = 1e300, smax = -1e300;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int col = 0; col < layout.dim; col++) {
        double v = 0.0;
        for (int c = 0; c < layout.dim; c++) {
          v += q_ref[head][row][c] * k_ref[head][col][c];
        }
        S[head][row][col] = v;
        smin = std::min(smin, v);
        smax = std::max(smax, v);
      }
    }
  }
  const double span = smax - smin;
  // row_shift is indexed by the LAYOUT lane; the physical head there is
  // rev5(lane), exactly as every slot read in this suite.
  std::vector<std::vector<double>> row_shift(
      layout.lanes, std::vector<double>(layout.dim, -1e300));
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      for (int k = 0; k <= row; k++) {
        row_shift[lane][row] =
            std::max(row_shift[lane][row], S[head][row][k]);
      }
    }
  }
  // The live-norm estimates, folded into the mask as est^-1/2. At 128
  // live keys the raw interval is wide enough that invsqrt deg 15 costs
  // ~1.4e-2 on the row sums (the first run of this test measured it);
  // with the fold the interval is the actual / estimate ratio, here 1 by
  // construction, and only the margins remain.
  std::vector<std::vector<double>> row_norm(
      layout.lanes, std::vector<double>(layout.dim, 0.0));
  double raw_lo = 1e300, raw_hi = -1e300;
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = rev(lane, 5);
    for (int row = 0; row < layout.dim; row++) {
      double sqv = 0.0;
      for (int k = 0; k <= row; k++) {
        sqv += std::exp(m_eff * (S[head][row][k] - row_shift[lane][row]) /
                        span);
      }
      row_norm[lane][row] = sqv;
      raw_lo = std::min(raw_lo, sqv);
      raw_hi = std::max(raw_hi, sqv);
    }
  }

  typename cheddar::CiSinCAttention<word>::SoftMaxCalibration calib;
  calib.m_eff = m_eff;
  calib.span = span;
  calib.shift = smax;
  calib.norm_lo = 0.9;
  calib.norm_hi = 1.1;
  calib.causal = true;
  calib.row_shift = row_shift;
  calib.row_norm = row_norm;
  attn_p->PrepareSoftMax(calib);

  // ---- the leg through the handler ------------------------------------
  const auto t2 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> s0;
  attn_p->Scores(s0, q_a, q_b, k_a, k_b, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t3 = std::chrono::steady_clock::now();
  const double carried = s0[0].GetScale() / boot.param->base_scale_;
  ASSERT_LT(carried * std::max(std::abs(smax), std::abs(smin)), 0.95)
      << "the handler's canonicalising fold did not land carried in "
         "EvalMod's range";

  std::vector<Ciphertext<word>> scores(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    s0[bi].SetNumSlots(num_slots);
    bctx->Boot(scores[bi], s0[bi], boot.ui->GetEvkMap());
  }
  ASSERT_EQ(boot.param->NPToLevel(scores[0].GetNP()), attn_p->GetTopLevel());

  const auto t4 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> P;
  attn_p->SoftMax(P, scores, carried, boot.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t5 = std::chrono::steady_clock::now();

  // What P arrived as: the causal shape checked before V.
  std::vector<std::vector<std::vector<double>>> P_dec(
      layout.lanes, std::vector<std::vector<double>>(
                        layout.dim, std::vector<double>(layout.dim, 0.0)));
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
          P_dec[rev(lane, 5)][row][column] = slots[slot].real();
        }
      }
    }
  }
  double masked_resid = 0.0, rowsum_dev = 0.0;
  for (int head = 0; head < layout.lanes; head++) {
    for (int row = 0; row < layout.dim; row++) {
      double rs = 0.0;
      for (int k = 0; k < layout.dim; k++) {
        if (k <= row) {
          rs += P_dec[head][row][k];
        } else {
          masked_resid =
              std::max(masked_resid, std::abs(P_dec[head][row][k]));
        }
      }
      rowsum_dev = std::max(rowsum_dev, std::abs(rs - 1.0));
    }
  }

  const auto t6 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> out;
  attn_p->Values(out, P, v_a, v_b, keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t7 = std::chrono::steady_clock::now();

  // =====================================================================
  // THE REST OF THE LAYER. Everything above is the attention leg, verbatim
  // from `TheLibraryLegReproducesTheReference`; what follows is the seam,
  // the O projection, the residual, the FFN and the second residual, so a
  // whole Llama-3 decoder layer runs on the conjugate-invariant ring.
  //
  // The attention output is decrypted once, here, and the host reference for
  // everything after it is computed from THAT -- so this measures the rest
  // of the layer exactly, and the attention half is measured by the
  // assertions the leg test already carries and which are kept above.
  // =====================================================================
  // ---- release the leg -------------------------------------------------
  //
  // Everything after this point needs `boot`, `out` and the layout, and
  // nothing else the leg built: not the three converters, not the switching,
  // product or lifted rings, not their keys. Freeing them here is what makes
  // room for the second BootContext.
  {
    size_t before = 0, total = 0, after = 0;
    cudaMemGetInfo(&before, &total);
    attn_p.reset();
    lifted.reset();
    small.reset();
    swtch.reset();
    cudaMemGetInfo(&after, &total);
    std::cout << "  released the leg: " << ((after - before) >> 20)
              << " MiB back, " << (after >> 20) << " MiB free" << std::endl;
  }

  // A MEMORY LEDGER, because three runs died of memory and each guess about
  // which item was to blame cost seventeen minutes. `cudaMemGetInfo` cannot
  // see inside RMM's pool -- that is why the release above reads 0 MiB back --
  // but it does see the pool RESERVE, and reservation growth is what actually
  // runs the card out. Printing it at every stage boundary says which stage
  // grows it.
  auto ledger = [](const char *tag) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::cout << "  [mem] " << tag << ": " << ((total_b - free_b) >> 20)
              << " MiB reserved, " << (free_b >> 20) << " MiB free"
              << std::endl;
  };
  ledger("leg released");

  // A TIME LEDGER BESIDE THE MEMORY ONE, and for the same reason: the target
  // is 7 s a layer and there is no way to aim at it without knowing which
  // rows are the seconds. Everything one-time -- key generation, the
  // converters, the seam's plaintext diagonals -- is reported apart from the
  // online work, because a layer pays those once for all 32 of them.
  auto tmark = std::chrono::steady_clock::now();
  auto stage = [&tmark](const char *tag) {
    cudaDeviceSynchronize();
    const auto now = std::chrono::steady_clock::now();
    std::cout << "  [time] " << tag << ": "
              << std::chrono::duration<double, std::milli>(now - tmark).count()
              << " ms" << std::endl;
    tmark = now;
  };
  {
    auto ms = [](const std::chrono::steady_clock::time_point &a,
                 const std::chrono::steady_clock::time_point &b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cout << "  [time] ONE-TIME leg handler (three converters): "
              << ms(t0, t1) << " ms" << std::endl;
    std::cout << "  [time] leg scores: " << ms(t2, t3) << " ms, 8 Boots: "
              << ms(t3, t4) << " ms, softmax: " << ms(t4, t5)
              << " ms, values: " << ms(t6, t7) << " ms" << std::endl;
    // t5 -> t6 is the host's decryption of P and its reference, not circuit.
  }

  // The attention output, decrypted once and laid out the way the O
  // projection reads it. The seam sends chain entry (row, col, lane) to
  // block channel `chan_of(col, lane % 16)` of half ciphertext
  // `2 * bi + lane / 16`, and `CoeffLinearLeg` numbers a parent's channels
  // `parent * rank + channel`, so that is the flat index. Reading it as
  // "live channel 2j" instead -- the obvious guess -- puts every entry under
  // the wrong weight column.
  const int attn_channels = 2 * layout.num_cts * proj_rank;
  std::vector<double> attn_flat(
      static_cast<size_t>(layout.dim) * attn_channels, 0.0);
  auto chan_of0 = [&](int col, int lh) { return rev(col, 4) * 32 + rev(lh, 5); };
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, out[bi]);
    std::vector<Complex> slots;
    boot.context->encoder_.Decode(slots, pt);
    for (int col = 0; col < layout.rank; col++) {
      for (int lane = 0; lane < layout.lanes; lane++) {
        const int k = 2 * bi + lane / 16;
        const int c = chan_of0(col, lane % 16);
        for (int row = 0; row < layout.dim; row++) {
          int ci, sl, cs;
          layout.LocateSlot(row, bi * layout.rank + col, lane, ci, sl, cs);
          attn_flat[static_cast<size_t>(row) * attn_channels +
                    k * proj_rank + c] = slots[sl].real();
        }
      }
    }
  }

  // ---- a second BootContext, because the two halves want opposite things
  //
  // The leg's softmax walk needs `GetEndLevel()` at 16 -- affine, exp,
  // norm, invsqrt and P at 3 (1.5bv) -- and that is `dec - num_stc - slack`,
  // so the leg needs slack ZERO. The FFN needs slack, because `SlotToCoeff`
  // is compiled at `GetStCStartLevel()` and an operator eight levels deep
  // cannot arrive there without it. Run with slack 9 the leg refuses, in as
  // many words: "the softmax walk overspends its levels".
  //
  // The conflict is real and the tree already knows the answer -- it is what
  // the switching ring is: a second Context over the SAME primes holding the
  // SAME secret, so a ciphertext crosses between them without a word
  // changing. Here the two differ only in slack.
  // SLACK TWELVE, NOT NINE, AND THE REASON IS MEMORY. The seam's 192
  // rotation keys are the layer's largest single key demand, and a key's
  // size is its level's limb count; more slack puts StC lower, which lets
  // the seam sit lower, which makes those keys smaller. Nine put the seam at
  // 13/12 and the run died in the pool; twelve puts it at 10/9. The FFN's
  // own stages still fit -- RMSNorm leaves 11 against StC's 7 -- and
  // `GetCoeffLevel()` is 4, still above the product level.
  Ring boot_ffn(kBootParam, boot.ui->GetSecretCoeffs(),
                /*boot_slack_levels=*/12);
  auto fctx = std::dynamic_pointer_cast<BootContext<word>>(boot_ffn.context);
  ASSERT_NE(fctx, nullptr);
ledger("before the FFN context");
  fctx->PrepareEvalMod();
  fctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest req;
    fctx->AddRequiredRotations(req, num_slots);
    boot.ui->PrepareRotationKey(req);
  }

  ledger("FFN context up");
  stage("ONE-TIME the FFN BootContext and its keys");

  // ---- the seam: chain layout -> the block's banded half-density image --
  // T1 IS A BIT PERMUTATION OF THE SLOT INDEX, and running it as one
  // transform was the whole of the seam's memory. Written out, the chain
  // address `rev4(col) * 4096 + rev7(row) * 32 + lane` and the block address
  // `row + 128 * (rev4(col) * 32 + rev5(lh))` agree on the column field and
  // differ by six TRANSPOSITIONS of index bits -- (11,0) (10,1) (9,2) (8,3)
  // (6,5) (7,4) -- plus a shift of 128 for the upper half, because the
  // transposition puts `half` at destination bit 7 where the packing wants a
  // zero. That is exactly why the one-shot count was 486: a transposition
  // contributes offsets {0, +-(2^i - 2^j)}, so six of them give 3^6 = 729
  // combinations, and the fixed source bit 4 collapses one factor to two:
  // 2 * 3^5 = 486, measured to the digit.
  //
  // Splitting the six across stages multiplies out far less. Searched on the
  // host over every ordered partition, minimising rotation keys:
  //
  //     one stage    486 diagonals, 192 keys   (BSGS 128x64)
  //     two stages   165 diagonals, 100 keys
  //     three        60 diagonals,   56 keys   <-- this
  //
  // The stages are (11,0) | (10,1) | (9,2) (8,3) (6,5) (7,4), at 3, 3 and 54
  // diagonals. Each costs a level, and the levels are there: 12/11/10 for T1
  // and 9 for T2 leaves the output at 8, one above `SlotToCoeff`'s 7 under
  // slack 12 -- and every one of them is above 7, which is where ci16_35's
  // hoisted transforms stop working (1.5bt).
  const std::vector<std::vector<std::pair<int, int>>> t1_stages = {
      {{11, 0}}, {{10, 1}}, {{9, 2}, {8, 3}, {6, 5}, {7, 4}}};
  // THE TOKEN SHIFT IS NOT A SLOT ROTATION, which is what the O projection
  // was reading through. The banded convention is a statement about
  // COEFFICIENT positions -- coefficient `p * rank + I` carries
  // `comp_I[p] + comp_{rank-I}[p+1]` -- and `SlotToCoeff` sends slot
  // `t + 128 * c` to coefficient `rev7(t) * 512 + rev9(c)`, so a step of one
  // in `p` is a step of one in `rev7(t)`, NOT in `t`. The seam shifted by one
  // slot; on the host, reading the resulting image back through
  // `BandedComponents` gives max error 38.56 against |v| <= 4.16, while the
  // corrected token `rev7(rev7(t) - 1)` gives EXACTLY ZERO.
  //
  // That map costs almost nothing: a decrement in bit-reversed order is a
  // carry, so it has just 7 distinct slot offsets, BSGS 8x8. It replaces the
  // rotation and takes one level, so T1 starts one higher.
  const int t1_top = 13, t2_level = 9, tokmap_level = 10;
  const int t1_level = t1_top;  // the level the seam's input is brought to
  auto swap_bits = [](int x, int i, int j) {
    if (((x >> i) & 1) != ((x >> j) & 1)) x ^= (1 << i) | (1 << j);
    return x;
  };
  cheddar::SylphSchedule<word> sched(fctx, num_slots);
  std::cout << "layer: slot " << sched.GetSlotLevel() << ", StC "
            << sched.GetStCLevel() << ", coeff " << sched.GetCoeffLevel()
            << ", seam at " << t1_level << "/" << t2_level << std::endl;
  ASSERT_GT(t2_level - 1, sched.GetStCLevel())
      << "the seam has to leave the ciphertext above StC's level with a "
         "rescale to spare";

  auto slot_chain = [&](int row, int col, int lane) {
    return rev(col, 4) * 4096 + rev(row, 7) * 32 + lane;
  };
  auto slot_block = [&](int token, int chan) { return token + 128 * chan; };
  auto chan_of = [&](int col, int lh) { return rev(col, 4) * 32 + rev(lh, 5); };
  auto best_window = [&](const cheddar::StripedMatrix &m, int *need) {
    std::vector<int> offs;
    for (const auto &kv : m) offs.push_back(kv.first);
    int bw = 0;
    long long bn = -1;
    for (int w : offs) {
      long long g = 0, mx = 0;
      for (int o : offs) {
        const long long r = ((o - w) % n + n) % n;
        mx = std::max(mx, r);
        long long a2 = g, b2 = r;
        while (b2) { const long long t = a2 % b2; a2 = b2; b2 = t; }
        g = a2;
      }
      if (g == 0) continue;
      const long long q = mx / g + 1;
      if (bn < 0 || q < bn) { bn = q; bw = w; }
    }
    *need = static_cast<int>(bn);
    return bw;
  };
  auto split = [](int need) {
    int bs = 1;
    while (bs * bs < need) bs *= 2;
    int gs = 1;
    while (bs * gs < need) gs *= 2;
    return std::make_pair(bs, gs);
  };
  auto pt_scale = [&](int l) {
    return boot_ffn.param->GetScale(l - 1) * boot_ffn.param->GetRescalePrimeProd(l) /
           boot_ffn.param->GetScale(l);
  };

  // ONE T1 AT A TIME. Each is 486 plaintext diagonals at level 10, which is
  // 2.9 GB, and the two halves are never used together; building both up
  // front only kept 2.9 GB pinned for the whole run. `make_t1` is called
  // inside the half loop below and its result dies at the end of it.
  std::unique_ptr<cheddar::LinearTransform<word>> seam_t2;
  int back2 = 0;
  std::vector<std::unique_ptr<cheddar::LinearTransform<word>>> seam_t1_cur;
  std::vector<int> back1_stage;
  auto make_t1 = [&](int half) {
    seam_t1_cur.clear();
    back1_stage.assign(t1_stages.size(), 0);
    // The live source addresses, walked through the stages. A stage's matrix
    // is built from where its inputs are and where it sends them, so the
    // composition is checked by construction rather than re-derived.
    std::vector<int> cur;
    cur.reserve(static_cast<size_t>(layout.rank) * 16 * layout.dim);
    for (int col = 0; col < layout.rank; col++) {
      for (int lh = 0; lh < 16; lh++) {
        for (int row = 0; row < layout.dim; row++) {
          cur.push_back(slot_chain(row, col, half * 16 + lh));
        }
      }
    }
    for (size_t st = 0; st < t1_stages.size(); st++) {
      const bool last = (st + 1 == t1_stages.size());
      std::vector<int> mid(cur.size());
      cheddar::StripedMatrix ms(n, n);
      for (size_t e = 0; e < cur.size(); e++) {
        int y = cur[e];
        for (const auto &sw : t1_stages[st]) y = swap_bits(y, sw.first, sw.second);
        if (last) y -= 128 * half;  // destination bit 7 is a zero, not `half`
        mid[e] = y;
        const int off = ((cur[e] - y) % n + n) % n;
        ms.try_emplace(off, n, Complex(0.0, 0.0));
        ms[off][y] = Complex(1.0, 0.0);
      }
      int need = 0;
      const int w = best_window(ms, &need);
      const auto sp = split(need);
      const int lvl = t1_top - static_cast<int>(st);
      seam_t1_cur.push_back(std::make_unique<cheddar::LinearTransform<word>>(
          boot_ffn.context, ms, lvl, pt_scale(lvl), sp.first, sp.second, w, -w));
      back1_stage[st] = ((w % n) + n) % n;
      std::cout << "  seam T1[" << half << "] stage " << st << ": "
                << ms.GetNumDiag() << " diagonals, " << sp.first << "x"
                << sp.second << " at level " << lvl << std::endl;
      cheddar::EvkRequest req;
      seam_t1_cur.back()->AddRequiredRotations(req);
      req.AddRequest(back1_stage[st], lvl - 1);
      boot.ui->PrepareRotationKey(req);
      cur.swap(mid);
    }
    // The last stage must land exactly on the block's live addresses.
    for (int col = 0; col < layout.rank; col++) {
      for (int lh = 0; lh < 16; lh++) {
        for (int row = 0; row < layout.dim; row++) {
          const size_t e = (static_cast<size_t>(col) * 16 + lh) * layout.dim + row;
          ASSERT_EQ(cur[e], slot_block(row, chan_of(col, lh)))
              << "the staged bit permutation is not T1";
        }
      }
    }
  };
  {
    cheddar::StripedMatrix m2(n, n);
    for (int col = 0; col < layout.rank; col++) {
      for (int lh = 0; lh < 16; lh++) {
        const int c = chan_of(col, lh);
        const int I = rev(c, 9);
        if (I == 0) continue;  // component zero has no partner
        const int cd = rev(proj_rank - I, 9);
        const int off = ((128 * (c - cd)) % n + n) % n;
        m2.try_emplace(off, n, Complex(0.0, 0.0));
        for (int row = 1; row < layout.dim; row++) {
          m2[off][slot_block(row - 1, cd)] = Complex(1.0, 0.0);
        }
      }
    }
    int need = 0;
    const int w = best_window(m2, &need);
    const auto sp = split(need);
    seam_t2 = std::make_unique<cheddar::LinearTransform<word>>(
        boot_ffn.context, m2, t2_level, pt_scale(t2_level), sp.first, sp.second,
        w, -w);
    back2 = ((w % n) + n) % n;
    std::cout << "  seam T2: " << m2.GetNumDiag() << " diagonals, "
              << sp.first << "x" << sp.second << std::endl;
  }
  // The bit-reversed decrement, as a transform on the live image.
  std::unique_ptr<cheddar::LinearTransform<word>> seam_tok;
  int back_tok = 0;
  {
    cheddar::StripedMatrix mt(n, n);
    for (int col = 0; col < layout.rank; col++) {
      for (int lh = 0; lh < 16; lh++) {
        const int c = chan_of(col, lh);
        for (int t = 0; t < layout.dim; t++) {
          const int pos = rev(t, 7);
          if (pos == 0) continue;  // nothing sits one position below it
          const int td = rev(pos - 1, 7);
          const int dst = slot_block(td, c);
          const int off = ((slot_block(t, c) - dst) % n + n) % n;
          mt.try_emplace(off, n, Complex(0.0, 0.0));
          mt[off][dst] = Complex(1.0, 0.0);
        }
      }
    }
    int need = 0;
    const int w = best_window(mt, &need);
    const auto sp = split(need);
    seam_tok = std::make_unique<cheddar::LinearTransform<word>>(
        boot_ffn.context, mt, tokmap_level, pt_scale(tokmap_level), sp.first,
        sp.second, w, -w);
    back_tok = ((w % n) + n) % n;
    std::cout << "  seam token map: " << mt.GetNumDiag() << " diagonals, "
              << sp.first << "x" << sp.second << " at level " << tokmap_level
              << std::endl;
  }
  {
    EvkRequest req;
    seam_tok->AddRequiredRotations(req);
    req.AddRequest(back_tok, tokmap_level - 1);
    seam_t2->AddRequiredRotations(req);
    req.AddRequest(back2, t2_level - 1);
    req.AddRequest(1, t2_level);
    boot.ui->PrepareRotationKey(req);
  }

  stage("ONE-TIME the seam's T2, its token map and their keys");

  // Boot the attention output, run the seam, come back to coefficients.
  std::vector<Ciphertext<word>> booted(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    bctx->Boot(booted[bi], out[bi], boot.ui->GetEvkMap());
  }
  ledger("the eight Boots done");
  stage("the 8 Boots before the seam");
  out.clear();
  out.shrink_to_fit();
  std::vector<Ciphertext<word>> h_cts(2 * layout.num_cts);
  for (int half = 0; half < 2; half++) {
    make_t1(half);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      Ciphertext<word> low, a2, sh, dup, live, sum;
      boot_ffn.context->LevelDown(low, booted[bi], t1_top);
      a2 = std::move(low);
      for (size_t st = 0; st < seam_t1_cur.size(); st++) {
        Ciphertext<word> next;
        seam_t1_cur[st]->Evaluate(boot_ffn.context, next, a2,
                                  boot.ui->GetEvkMap());
        if (back1_stage[st]) {
          Ciphertext<word> r;
          boot_ffn.context->HRot(
              r, next, boot.ui->GetEvkMap().GetRotationKey(back1_stage[st]),
              back1_stage[st]);
          next = std::move(r);
        }
        a2 = std::move(next);
      }
      seam_tok->Evaluate(boot_ffn.context, sh, a2, boot.ui->GetEvkMap());
      if (back_tok) {
        Ciphertext<word> r;
        boot_ffn.context->HRot(
            r, sh, boot.ui->GetEvkMap().GetRotationKey(back_tok), back_tok);
        sh = std::move(r);
      }
      seam_t2->Evaluate(boot_ffn.context, dup, sh, boot.ui->GetEvkMap());
      if (back2) {
        Ciphertext<word> r;
        boot_ffn.context->HRot(r, dup, boot.ui->GetEvkMap().GetRotationKey(back2),
                           back2);
        dup = std::move(r);
      }
      const int dl = boot_ffn.param->NPToLevel(dup.GetNP());
      // The live half now sits two levels above `dup`, because the token map
      // took one; bring it to `dl + 1` so the constant multiply below lands
      // both at the same level AND the same scale.
      if (boot_ffn.param->NPToLevel(a2.GetNP()) != dl + 1) {
        Ciphertext<word> d;
        boot_ffn.context->LevelDown(d, a2, dl + 1);
        a2 = std::move(d);
      }
      cheddar::Constant<word> one;
      boot_ffn.context->encoder_.EncodeConstant(
          one, dl + 1,
          boot_ffn.param->GetScale(dl) * boot_ffn.param->GetRescalePrimeProd(dl + 1) /
              a2.GetScale(),
          1.0);
      boot_ffn.context->Mult(live, a2, one);
      boot_ffn.context->Rescale(live, live);
      boot_ffn.context->Add(sum, live, dup);
      sched.ToCoeff(h_cts[bi * 2 + half], sum, boot.ui->GetEvkMap());
    }
    seam_t1_cur.clear();
  }
  booted.clear();
  booted.shrink_to_fit();
  // The seam is over, and nothing downstream reads it. Its three transforms
  // are 486 + 486 + 129 plaintext diagonals -- 6.7 GB -- and the run that
  // reached here died in the O projection with all of them still resident.
  seam_t2.reset();
  ledger("seam done and released");
  // The T1 stages are rebuilt inside the half loop, so this row carries one
  // one-time cost per half that a real layer would hoist out.
  stage("the seam: T1 x 3 stages, token map, T2, ToCoeff, over 16 halves");
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::cout << "  the seam gave " << h_cts.size()
            << " half-density coefficient ciphertexts at level "
            << boot_ffn.param->NPToLevel(h_cts[0].GetNP()) << std::endl;

  // ---- the O projection, the residual, and the FFN ----------------------
  //
  // Widths: the attention output is `num_cts * rank * lanes` = 4096 channels
  // in 16 half-density ciphertexts (declared 8192); the model dimension is
  // the X the leg projected from, 512 declared with 256 live; the FFN's
  // inner dimension is twice that.
  const int model_declared = proj_rank;            // 512, 256 live
  const int hidden_declared = 2 * proj_rank;       // 1024, 512 live
  ASSERT_EQ(static_cast<int>(h_cts.size()) * proj_rank, attn_channels);

  std::normal_distribution<double> lw(0.0, 0.02);
  auto half_weight = [&](int in_dec, int out_dec) {
    std::vector<double> w(static_cast<size_t>(in_dec) * out_dec, 0.0);
    for (int i = 0; i < in_dec; i += 2) {
      for (int o = 0; o < out_dec; o += 2) {
        w[static_cast<size_t>(i) * out_dec + o] = lw(gen);
      }
    }
    return w;
  };
  const auto wo = half_weight(attn_channels, model_declared);
  const auto wgate = half_weight(model_declared, hidden_declared);
  const auto wup = half_weight(model_declared, hidden_declared);
  const auto wdown = half_weight(hidden_declared, model_declared);
  std::vector<double> wnorm(model_declared, 0.0);
  for (int c = 0; c < model_declared; c += 2) {
    wnorm[c] = 0.5 + 0.5 * std::abs(lw(gen));
  }

  typename cheddar::CoeffLinearLeg<word>::Config lcfg;
  lcfg.num_tokens = proj_small;
  lcfg.product_level = pcmm_level;
  // TILE THE DESCENT. On the direct route a parent's `rank` module components
  // are each as large as the parent's own a-part -- `Decompose`'s own comment
  // says so, and says the tile is what bounds it -- so sixteen parents at rank
  // 512 and five limbs at the product level is 10.7 GB standing at once. The
  // ledger says the layer arrives at the O projection with 15.4 GB, and that
  // is where four runs died. Four parents a tile is 2.7 GB. It is paid for in
  // ModPacks: a tile costs one per output group, so the projections run four
  // times as many. That is time, and the alternative was not running.
  lcfg.parents_per_tile = 4;
  ledger("before the leg object");
  ProjectOnlyLegCi leg(boot_ffn.context, lcfg, pack_keys);
  ledger("leg object built");
  stage("ONE-TIME the projection leg object");

  auto host_mm2 = [&](const std::vector<double> &in, int in_w,
                     const std::vector<double> &w, int out_w) {
    std::vector<double> r(static_cast<size_t>(layout.dim) * out_w, 0.0);
    for (int t = 0; t < layout.dim; t++) {
      for (int o = 0; o < out_w; o++) {
        double acc = 0.0;
        for (int c = 0; c < in_w; c++) {
          acc += in[static_cast<size_t>(t) * in_w + c] *
                 w[static_cast<size_t>(c) * out_w + o];
        }
        r[static_cast<size_t>(t) * out_w + o] = acc;
      }
    }
    return r;
  };

  // SIZE THE RESIDUAL FOR THE CROSSING. The FFN's first stage HalfBoots the
  // residual stream, and ModRaise cannot carry a message much past 0.4
  // (1.5ca's "pre-RoPE |projection| <= 0.45", and the FFN's own first run
  // came back as garbage that still decrypted). |x| is order one and the O
  // product is its own size, so the bound is bought where it is free: in the
  // O projection's weight scale, with the same factor folded into the
  // stream's encode so the residual still adds.
  const auto o_unit = host_mm2(attn_flat, attn_channels, wo, model_declared);
  double res_max = 0.0;
  for (int t = 0; t < proj_small; t++) {
    for (int c = 0; c < model_declared; c += 2) {
      res_max = std::max(res_max,
                         std::abs(x_comp[rev(c, 9)][rev(t, 7)]) +
                             std::abs(o_unit[static_cast<size_t>(t) *
                                             model_declared + c]));
    }
  }
  const double res_scale = 0.35 / std::max(res_max, 1e-12);
  std::cout << "  residual would reach " << res_max << ", so the O weight "
            << "carries " << res_scale << std::endl;

  Ciphertext<word> o_out;
  {
    std::vector<Ciphertext<word>> ins(h_cts.size());
    for (size_t k = 0; k < h_cts.size(); k++) {
      boot_ffn.context->LevelDown(ins[k], h_cts[k], pcmm_level);
    }
    std::vector<Ciphertext<word>> res;
    leg.Project(res, ins, attn_channels, model_declared, wo, res_scale,
                "o");
    ASSERT_EQ(res.size(), 1u);
    o_out = std::move(res[0]);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::cout << "  O projection: one half-density ciphertext at level "
            << boot_ffn.param->NPToLevel(o_out.GetNP()) << std::endl;
  ledger("O projection done");
  // The weight ENCODE is inside `Project` and CLAUDE.md separates it at
  // 0.366 s/ct one-time against 0.751 s/ct online; this row is both.
  stage("the O projection, weight encode included");

  std::vector<double> o_host(o_unit.size(), 0.0);
  for (size_t i = 0; i < o_unit.size(); i++) o_host[i] = o_unit[i] * res_scale;
  std::cout << "THE CI LAYER RAN: attention -> seam -> O projection, all "
            << "encrypted, " << h_cts.size() << " half ciphertexts through "
            << "the seam" << std::endl;

  double o_fit = 1.0;
  // The O projection's output read back through the banded scan, against the
  // same product in double from the decrypted attention output.
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, o_out);
    std::vector<double> coeffs;
    boot_ffn.context->encoder_.DecodeCoeff(coeffs, pt);
    // comp[i][p], the inverse of the banded recomposition
    std::vector<std::vector<double>> comp(
        proj_rank, std::vector<double>(proj_small, 0.0));
    for (int t = 0; t < proj_small; t++) {
      comp[0][t] = coeffs[static_cast<size_t>(t) * proj_rank];
    }
    for (int i = 1; i <= proj_rank / 2; i++) {
      const int mi = proj_rank - i;
      double ai = 0.0, am = 0.0;
      for (int t = proj_small - 1; t >= 0; t--) {
        const double ni = coeffs[static_cast<size_t>(t) * proj_rank + i] - am;
        const double nm = coeffs[static_cast<size_t>(t) * proj_rank + mi] - ai;
        comp[i][t] = ni;
        comp[mi][t] = nm;
        ai = ni;
        am = nm;
      }
    }
    double num = 0.0, den = 0.0, mx = 0.0;
    for (int c = 0; c < model_declared; c += 2) {
      for (int t = 0; t < proj_small; t++) {
        const double want =
            o_host[static_cast<size_t>(t) * model_declared + c];
        num += comp[rev(c, 9)][rev(t, 7)] * want;
        den += want * want;
        mx = std::max(mx, std::abs(want));
      }
    }
    o_fit = num / den;
    const double fit = o_fit;
    double err = 0.0;
    for (int c = 0; c < model_declared; c += 2) {
      for (int t = 0; t < proj_small; t++) {
        const double got = comp[rev(c, 9)][rev(t, 7)] / fit;
        err = std::max(err, std::abs(got - o_host[static_cast<size_t>(t) *
                                                  model_declared + c]));
      }
    }
    std::cout << "  O against the host product from the decrypted attention "
              << "output: " << err << " (|o| <= " << mx << ", carried " << fit
              << ")" << std::endl;
    EXPECT_LT(err / mx, 0.05)
        << "the seam did not hand the O projection a readable banded image";
  }
  // ---- the residual, and the FFN ----------------------------------------
  //
  // THE RESIDUAL'S CONSTANT IS A CALIBRATION NUMBER, and measuring it here
  // is what a block does offline. `h = x + O(attn)`, and the two operands
  // must carry the same constant: the O projection's output carries whatever
  // the crossings and `w_scale` left it with, which is the `carried` fitted
  // just above, so the stream it is added to is encrypted with that factor
  // folded in. [SYLPH] section 3.1.3 calls this calibration and 1.5ch folds
  // cq / ck into the weight encodes the same way; the only difference is
  // that this test measures it in the same process rather than offline.
  Ciphertext<word> stream;
  {
    std::vector<std::vector<double>> scaled(x_comp.size(),
                                            std::vector<double>(proj_small));
    for (size_t i = 0; i < x_comp.size(); i++) {
      for (int t = 0; t < proj_small; t++) {
        scaled[i][t] = x_comp[i][t] * o_fit * res_scale;
      }
    }
    Plaintext<word> pt;
    boot_ffn.context->encoder_.EncodeCoeff(pt, 0, boot_ffn.param->GetScale(0),
                                       HostRecompose(scaled, proj_rank,
                                                     proj_small));
    boot.ui->Encrypt(stream, pt);
    stream.SetNumSlots(num_slots);
  }
  Ciphertext<word> h_ct;
  boot_ffn.context->Add(h_ct, stream, o_out);
  std::cout << "  residual added at level "
            << boot_ffn.param->NPToLevel(h_ct.GetNP()) << std::endl;
  stage("the attention residual");

  // The host's residual stream, in the same units.
  std::vector<double> h_host(
      static_cast<size_t>(proj_small) * model_declared, 0.0);
  for (int c = 0; c < model_declared; c += 2) {
    for (int t = 0; t < proj_small; t++) {
      // The ciphertext carries `o_fit * (x + O)`: the stream was encrypted
      // with o_fit folded in and the O output already carries it. So the
      // host residual is `x + O`, and the factor rides through to the
      // boundary constant the next crossing measures.
      h_host[static_cast<size_t>(t) * model_declared + c] =
          x_comp[rev(c, 9)][rev(t, 7)] * res_scale +
          o_host[static_cast<size_t>(t) * model_declared + c];
    }
  }

  // ---- RMSNorm, gate and up, SiLU, down, and the second residual --------
  //
  // Every slot-domain stage here is DUPLICATE-PRESERVING (1.5cs): the
  // canonicalise is a uniform constant rather than a mask, RMSNorm reduces
  // with `channel_stride = 2` so the two parities stay apart, and its weight
  // carries the PARTNER channel's value at the duplicate slots. SiLU and the
  // gate multiply are pointwise and so are duplicate-safe for nothing.
  const int slot_level = sched.GetSlotLevel();
  const int op_level = slot_level - 1;
  auto canonicalise = [&](Ciphertext<word> &ct, double restore) {
    cheddar::Constant<word> k;
    boot_ffn.context->encoder_.EncodeConstant(
        k, slot_level,
        boot_ffn.param->GetScale(op_level) *
            boot_ffn.param->GetRescalePrimeProd(slot_level) / ct.GetScale(),
        restore);
    boot_ffn.context->Mult(ct, ct, k);
    boot_ffn.context->Rescale(ct, ct);
  };

  double ms_lo = 1e300, ms_hi = 0.0, log_sum = 0.0;
  std::vector<double> ms(proj_small, 0.0);
  for (int t = 0; t < proj_small; t++) {
    double s2 = 0.0;
    for (int c = 0; c < model_declared; c++) {
      const double v = h_host[static_cast<size_t>(t) * model_declared + c];
      s2 += v * v;
    }
    ms[t] = s2 / model_declared;
    log_sum += std::log(ms[t]);
    ms_lo = std::min(ms_lo, ms[t]);
    ms_hi = std::max(ms_hi, ms[t]);
  }
  const double alpha = 1.0 / std::exp(log_sum / proj_small);
  std::cout << "  residual mean-square spread " << (ms_hi / ms_lo) << "x"
            << std::endl;

  double boundary = 0.0;
  Ciphertext<word> normed;
  {
    Ciphertext<word> up;
    sched.ToSlot(up, h_ct, boot.ui->GetEvkMap());
    {
      Plaintext<word> rp;
      boot.ui->Decrypt(rp, up);
      std::vector<Complex> raw;
      boot_ffn.context->encoder_.Decode(raw, rp);
      double num = 0.0, den = 0.0;
      for (int c = 0; c < model_declared; c += 2) {
        for (int t = 0; t < proj_small; t++) {
          const double w = h_host[static_cast<size_t>(t) * model_declared + c];
          num += raw[c * proj_small + t].real() * w;
          den += w * w;
        }
      }
      boundary = num / den;
      std::cout << "  HalfBoot boundary constant " << boundary << std::endl;
      ASSERT_GT(std::abs(boundary), 0.0);
    }
    canonicalise(up, 1.0 / boundary);

    cheddar::RmsNormHandler<word> rms(boot_ffn.context, proj_small,
                                      model_declared, alpha, op_level, 1e-5,
                                      6.0, 9, /*channel_stride=*/2);
    ASSERT_EQ(rms.GetNumCiphertexts(), 1);
    for (int d : rms.GetRotationDistances()) {
      boot.ui->PrepareRotationKey(d, op_level);
    }
    const double root_alpha = std::sqrt(alpha);
    std::vector<std::vector<Complex>> wts(1);
    wts[0].assign(num_slots, Complex(0.0, 0.0));
    for (int c = 0; c < model_declared; c++) {
      const int I = rev(c, 9);
      const int src = (c % 2 == 0) ? c : rev(proj_rank - I, 9);
      for (int t = 0; t < proj_small; t++) {
        wts[0][c * proj_small + t] = Complex(wnorm[src] * root_alpha, 0.0);
      }
    }
    std::vector<Ciphertext<word>> in(1), outv;
    in[0] = std::move(up);
    rms.Apply(outv, in, wts, boot.ui->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    sched.ToCoeff(normed, outv[0], boot.ui->GetEvkMap());
  }
  std::cout << "  RMSNorm(ffn) done, coefficients at level "
            << boot_ffn.param->NPToLevel(normed.GetNP()) << std::endl;
  // This row carries a host decryption -- the boundary constant is measured
  // in-pipeline rather than assumed -- so it is an upper bound on the
  // circuit's own time, not the circuit's time.
  stage("HalfBoot, RMSNorm and back to coefficients (+ one host decrypt)");

  std::vector<double> n_host(h_host.size(), 0.0);
  for (int t = 0; t < proj_small; t++) {
    const double inv = 1.0 / std::sqrt(ms[t] + 1e-5);
    for (int c = 0; c < model_declared; c++) {
      n_host[static_cast<size_t>(t) * model_declared + c] =
          h_host[static_cast<size_t>(t) * model_declared + c] * inv *
          wnorm[c];
    }
  }

  const auto g_host = host_mm2(n_host, model_declared, wgate, hidden_declared);
  const auto u_host = host_mm2(n_host, model_declared, wup, hidden_declared);
  double gmax = 0.0;
  for (double v : g_host) gmax = std::max(gmax, std::abs(v));
  const double proj_size = 0.4 / std::max(gmax, 1e-12);
  const double silu_range = 12.0;

  std::vector<Ciphertext<word>> gate, upv;
  {
    Ciphertext<word> low;
    boot_ffn.context->LevelDown(low, normed, pcmm_level);
    std::vector<Ciphertext<word>> ins(1);
    ins[0] = std::move(low);
    leg.Project(gate, ins, model_declared, hidden_declared, wgate, proj_size,
                "gate");
    leg.Project(upv, ins, model_declared, hidden_declared, wup, proj_size,
                "up");
  }
  ASSERT_EQ(gate.size(), 2u);
  stage("the gate and up projections");

  std::vector<Ciphertext<word>> prod(2);
  {
    cheddar::SiLuHandler<word> silu(boot_ffn.context, silu_range, op_level, 31);
    for (int i = 0; i < 2; i++) {
      Ciphertext<word> g_up, u_up, sv, u_low;
      sched.ToSlot(g_up, gate[i], boot.ui->GetEvkMap());
      sched.ToSlot(u_up, upv[i], boot.ui->GetEvkMap());
      canonicalise(g_up, 1.0 / (boundary * proj_size * silu_range));
      canonicalise(u_up, 1.0 / (boundary * proj_size));
      silu.Apply(sv, g_up, boot.ui->GetEvkMap());
      boot_ffn.context->LevelDown(u_low, u_up,
                              boot_ffn.param->NPToLevel(sv.GetNP()));
      boot_ffn.context->HMult(prod[i], sv, u_low,
                          boot.ui->GetEvkMap().GetMultiplicationKey());
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  stage("HalfBoot x4, SiLU, the gate multiply");

  std::vector<double> gu(g_host.size(), 0.0);
  for (size_t i = 0; i < gu.size(); i++) {
    gu[i] = (g_host[i] / (1.0 + std::exp(-g_host[i]))) * u_host[i];
  }
  const auto y_host = host_mm2(gu, hidden_declared, wdown, model_declared);

  Ciphertext<word> down;
  {
    std::vector<Ciphertext<word>> ins(2);
    for (int i = 0; i < 2; i++) {
      Ciphertext<word> c2;
      sched.ToCoeff(c2, prod[i], boot.ui->GetEvkMap());
      boot_ffn.context->LevelDown(ins[i], c2, pcmm_level);
    }
    std::vector<Ciphertext<word>> res;
    leg.Project(res, ins, hidden_declared, model_declared, wdown, 1.0, "down");
    ASSERT_EQ(res.size(), 1u);
    down = std::move(res[0]);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  stage("ToCoeff x2 and the down projection");

  // ---- the layer's output, against the same layer in double -------------
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, down);
    std::vector<double> coeffs;
    boot_ffn.context->encoder_.DecodeCoeff(coeffs, pt);
    const auto comp = HostComponents(coeffs, proj_rank, proj_small);
    double num = 0.0, den = 0.0, mx = 0.0;
    for (int c = 0; c < model_declared; c += 2) {
      for (int t = 0; t < proj_small; t++) {
        const double want = y_host[static_cast<size_t>(t) * model_declared + c];
        num += comp[rev(c, 9)][rev(t, 7)] * want;
        den += want * want;
        mx = std::max(mx, std::abs(want));
      }
    }
    const double fit = num / den;
    double err = 0.0;
    for (int c = 0; c < model_declared; c += 2) {
      for (int t = 0; t < proj_small; t++) {
        const double got = comp[rev(c, 9)][rev(t, 7)] / fit;
        err = std::max(err, std::abs(
            got - y_host[static_cast<size_t>(t) * model_declared + c]));
      }
    }
    std::cout << "THE WHOLE CI LAYER: down-projection output " << err
              << " against |y| <= " << mx << " (relative " << (err / mx)
              << "), carried " << fit << std::endl;
    EXPECT_LT(err / mx, 0.10)
        << "the conjugate-invariant layer disagrees with the same layer in "
           "double by more than the circuit can explain";
  }
}

