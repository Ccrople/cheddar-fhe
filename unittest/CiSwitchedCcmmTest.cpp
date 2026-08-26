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

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/CiSwitchedCcmm.h"
#include "core/EvkRequest.h"
#include "extension/EvalSpecialFFT.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::CiLiftHandler;
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
