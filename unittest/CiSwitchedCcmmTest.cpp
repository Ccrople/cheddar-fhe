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
#include <iostream>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/CiSwitchedCcmm.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/ChebyshevFit.h"
#include "extension/EvalPoly.h"
#include "extension/EvalSpecialFFT.h"

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
