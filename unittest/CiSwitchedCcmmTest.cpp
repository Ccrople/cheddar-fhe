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

#include <chrono>
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
