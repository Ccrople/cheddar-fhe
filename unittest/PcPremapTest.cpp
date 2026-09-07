// Can the batched layer run on the PLAIN slot map, with the chain
// addressing entering the CC-MM as a converter PREMAP instead?
//
// Why the question. `CiBatchLayout` is chain-addressed today: the entry
// (token t, instance b) sits at `BlockOf(t, g) * lanes + lane` with
// `BlockOf = BitRev(t * rank + g)`, so that `CiSinCConverter`'s forward can
// consume the slots as the CC-MM chain wants them. The addressing is free
// for the CC-MM and costs nothing for the projections, the norms or the
// feed-forward -- none of those read the map. It DOES cost something for a
// plaintext that is constant over the token axis and varies only over the
// instance axis, which is exactly the shape of a PC-attention operand
// (Sylph 4.2: the public KV cache against encrypted queries, one public
// token per output ciphertext). Under the plain map `Slot(t, b) = t * B + b`
// such a vector is periodic with period B in the slot index -- a sparsely
// packed message, `B` native coefficients out of `degree`. Under the chain
// map it is not periodic at all: the token occupies the MIDDLE slot bits
// (5..11 at T = 128, rank 16, lanes 32), and the encode is dense.
// Measured off the device, at the shipped shape: 512 coefficients of 65536
// against ~62000. That is the difference between a 128x compression and
// none, on 4.06M plaintexts a layer.
//
// What this test asks. `CiSinCConverter`'s constructor already takes a
// `forward_premap`: "a lane-preserving slot permutation folded into the
// FORWARD on its input side, given at block granularity ... This is how a
// transport whose composed map is a block permutation rides the conversion
// for free: a column relabelling of the composed matrix, on the same
// lattice, same ceiling" (EvalSpecialFFT.h). The plain -> chain map is
// exactly that: it leaves the low `log2(sub_degree)` slot bits alone and
// permutes the block index by an 11-bit reversal. So the claim to check on
// the device is that the fold is free -- same BSGS split, same diagonal
// count, same plaintext bytes -- and that it computes the same thing.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "RingFixture.h"
#include "core/CiSwitchedCcmm.h"
#include "core/EvkRequest.h"
#include "core/SubringMatrix.h"
#include "extension/CiBatch.h"
#include "extension/EvalSpecialFFT.h"
#include "extension/LinearTransform.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::CiBatchLayout;
using cheddar::CiSinCConverter;
using cheddar::CiSwitchedCcmmLayout;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::Plaintext;

namespace {

// The shipped batched shape (`CiBatchAttention::Config`, `ci16_35_stc2`).
constexpr int kTokens = 128;
constexpr int kRank = 16;
constexpr int kLanes = 32;  // = sub_degree
constexpr int kInstances = kRank * kLanes;
constexpr int kForwardLevel = 4;
constexpr int kInverseLevel = 2;
constexpr int kBabySteps = 256;

int BitRev(int x, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((x >> i) & 1) << (bits - 1 - i);
  return r;
}

double Seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// The premap the plain map needs, built from the two `CiBatchLayout`s
// themselves rather than from a transcription of them, and checked against
// the contract `forward_premap` states: lanes untouched, a bijection over
// `degree / sub_degree` blocks.
std::vector<int> BuildPremap(int degree) {
  const CiBatchLayout plain(degree, kTokens);
  const CiBatchLayout chain(degree, kTokens, kLanes, kRank);
  EXPECT_EQ(plain.num_instances, kInstances);
  EXPECT_EQ(chain.num_instances, kInstances);

  const int num_blocks = degree / kLanes;
  std::vector<int> premap(num_blocks, -1);
  std::vector<int> hit(num_blocks, 0);
  for (int t = 0; t < kTokens; t++) {
    for (int b = 0; b < kInstances; b++) {
      const int ps = plain.Slot(t, b), cs = chain.Slot(t, b);
      // Lane-preserving: the low log2(sub_degree) bits agree.
      EXPECT_EQ(ps % kLanes, cs % kLanes) << "t " << t << " b " << b;
      const int pb = ps / kLanes, cb = cs / kLanes;
      if (premap[pb] >= 0) {
        EXPECT_EQ(premap[pb], cb) << "block " << pb << " is not well defined";
      }
      premap[pb] = cb;
      hit[pb]++;
    }
  }
  // A bijection over every block.
  std::vector<int> seen(num_blocks, 0);
  for (int b = 0; b < num_blocks; b++) {
    EXPECT_GE(premap[b], 0) << "block " << b << " unmapped";
    if (premap[b] >= 0) seen[premap[b]]++;
  }
  for (int b = 0; b < num_blocks; b++) {
    EXPECT_EQ(seen[b], 1) << "block " << b << " is hit " << seen[b] << " times";
  }
  return premap;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The premap is a lane-preserving block bijection, and it is the block
//    index's bit reversal -- no device needed, but stated here so that a
//    layout change fails in the same binary as the cost claim.
// ---------------------------------------------------------------------------
TEST(PcPremap, ThePlainToChainMapIsABlockBitReversal) {
  constexpr int kDegree = 65536;
  const auto premap = BuildPremap(kDegree);
  const int num_blocks = kDegree / kLanes;
  const int bits = 31 - __builtin_clz(kTokens * kRank);  // log2(2048) = 11
  ASSERT_EQ(1 << bits, kTokens * kRank);
  ASSERT_EQ(num_blocks, kTokens * kRank);
  for (int b = 0; b < num_blocks; b++) {
    ASSERT_EQ(premap[b], BitRev(b, bits)) << "block " << b;
  }
  std::cout << "  premap[b] = BitRev_" << bits << "(b) over " << num_blocks
            << " blocks; lanes (" << kLanes << " slots) untouched"
            << std::endl;
}

// ---------------------------------------------------------------------------
// 2. THE COST CLAIM: folding the premap into the forward changes neither the
//    BSGS split, nor the diagonal count it implies, nor the plaintext bytes.
//    EvalSpecialFFT.h's argument is that every map here lives on the
//    stride-`sub_degree` diagonal lattice, so the composed transform cannot
//    pass `degree / sub_degree` diagonals -- and the converter already
//    compiles that many.
// ---------------------------------------------------------------------------
TEST(PcPremap, ThePremapDoesNotGrowTheForward) {
  Ring swtch("ci_ringswitch16_35_boot.json", {}, 0,
             /*build_user_interface=*/false);
  const int degree = swtch.Degree();
  const CiSwitchedCcmmLayout layout(degree, degree / kRank, kLanes);
  ASSERT_EQ(layout.dim, kTokens);
  ASSERT_EQ(layout.rank, kRank);
  ASSERT_EQ(layout.lanes, kLanes);

  const auto premap = BuildPremap(degree);

  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> a(swtch.context, kLanes, kForwardLevel,
                          /*inverse_level=*/-1, &layout,
                          /*forward_premap=*/nullptr, kBabySteps);
  const auto t1 = std::chrono::steady_clock::now();
  CiSinCConverter<word> b(swtch.context, kLanes, kForwardLevel,
                          /*inverse_level=*/-1, &layout, &premap, kBabySteps);
  const auto t2 = std::chrono::steady_clock::now();

  ASSERT_NE(a.GetForward(), nullptr);
  ASSERT_NE(b.GetForward(), nullptr);
  const auto *fa = a.GetForward();
  const auto *fb = b.GetForward();

  std::cout << std::fixed << std::setprecision(2)
            << "  chain-native forward : bs " << fa->GetBS() << " x gs "
            << fa->GetGS() << " = " << (fa->GetBS() * fa->GetGS())
            << " diagonals, " << (fa->PlaintextBytes() >> 20) << " MiB, built in "
            << Seconds(t0, t1) << " s" << std::endl;
  std::cout << "  plain + premap       : bs " << fb->GetBS() << " x gs "
            << fb->GetGS() << " = " << (fb->GetBS() * fb->GetGS())
            << " diagonals, " << (fb->PlaintextBytes() >> 20) << " MiB, built in "
            << Seconds(t1, t2) << " s" << std::endl;
  std::cout << "  ceiling (degree / sub_degree) = " << (degree / kLanes)
            << std::endl;

  EXPECT_EQ(fa->GetBS(), fb->GetBS());
  EXPECT_EQ(fa->GetGS(), fb->GetGS());
  EXPECT_EQ(fa->GetPreRotationAmount(), fb->GetPreRotationAmount());
  EXPECT_EQ(fa->PlaintextBytes(), fb->PlaintextBytes());
  EXPECT_LE(fb->GetBS() * fb->GetGS(), degree / kLanes);
}

// ---------------------------------------------------------------------------
// 3. THE CORRECTNESS CLAIM: the same channel, packed in the PLAIN map and
//    converted through the premap'd forward, is the same SinC operand as the
//    chain-packed channel through the forward the branch builds today.
// ---------------------------------------------------------------------------
TEST(PcPremap, ThePlainMapThroughThePremapIsTheChainMap) {
  // The switching ring's own ladder is four main primes; the DATA's scale is
  // the layer ring's, and the two share their bottom primes, so a ciphertext
  // crosses keylessly (Doing.md 1.5bt). This is `CiBootSet`'s own recipe:
  // encode and encrypt on ci16_35, convert on the switching Context.
  Ring boot("ci16_35.json");
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  const int degree = swtch.Degree();
  ASSERT_EQ(boot.Degree(), degree);
  const CiSwitchedCcmmLayout layout(degree, degree / kRank, kLanes);
  const auto premap = BuildPremap(degree);

  CiSinCConverter<word> a(swtch.context, kLanes, kForwardLevel, -1, &layout,
                          nullptr, kBabySteps);
  CiSinCConverter<word> b(swtch.context, kLanes, kForwardLevel, -1, &layout,
                          &premap, kBabySteps);
  EvkRequest req;
  a.AddRequiredRotations(req);
  b.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  // One channel of the batched layout: a value per (token, instance).
  std::mt19937_64 gen(0x9C0FFEE);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(static_cast<size_t>(kTokens) * kInstances);
  for (auto &v : x) v = dist(gen);

  const CiBatchLayout plain(degree, kTokens);
  const CiBatchLayout chain(degree, kTokens, kLanes, kRank);
  std::vector<Complex> mc(degree, Complex(0.0, 0.0));
  std::vector<Complex> mp(degree, Complex(0.0, 0.0));
  for (int t = 0; t < kTokens; t++) {
    for (int bi = 0; bi < kInstances; bi++) {
      const double v = x[static_cast<size_t>(t) * kInstances + bi];
      mc[chain.Slot(t, bi)] = Complex(v, 0.0);
      mp[plain.Slot(t, bi)] = Complex(v, 0.0);
    }
  }

  const double s = boot.param->GetScale(kForwardLevel);
  Plaintext<word> pc, pp;
  boot.context->encoder_.Encode(pc, kForwardLevel, s, mc);
  boot.context->encoder_.Encode(pp, kForwardLevel, s, mp);
  // `cc2` is the CONTROL: the same chain message encrypted a second time and
  // put through the SAME converter. Its gap to `cc` is the floor two
  // independent encryptions leave -- everything the comparison below must
  // not be read as a layout disagreement.
  Ciphertext<word> cc, cc2, cp;
  boot.ui->Encrypt(cc, pc);
  boot.ui->Encrypt(cc2, pc);
  boot.ui->Encrypt(cp, pp);

  Ciphertext<word> oc, oc2, op;
  a.SlotToSinC(swtch.context, oc, cc, swtch.ui->GetEvkMap());
  a.SlotToSinC(swtch.context, oc2, cc2, swtch.ui->GetEvkMap());
  b.SlotToSinC(swtch.context, op, cp, swtch.ui->GetEvkMap());

  const auto read = [&](const Ciphertext<word> &ct) {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, ct);
    std::vector<double> v;
    boot.context->encoder_.DecodeCoeff(v, pt);
    return v;
  };
  const std::vector<double> vc = read(oc), vc2 = read(oc2), vp = read(op);
  ASSERT_EQ(vc.size(), vp.size());
  ASSERT_EQ(vc.size(), vc2.size());

  double mag = 0.0, floor_gap = 0.0, worst = 0.0;
  for (size_t i = 0; i < vc.size(); i++) {
    mag = std::max(mag, std::abs(vc[i]));
    floor_gap = std::max(floor_gap, std::abs(vc[i] - vc2[i]));
    worst = std::max(worst, std::abs(vc[i] - vp[i]));
  }
  std::cout << std::scientific << std::setprecision(3)
            << "  |chain forward| max            " << mag << std::endl
            << "  CONTROL |chain(enc1) - chain(enc2)| " << floor_gap
            << "   (relative 2^" << std::fixed << std::setprecision(2)
            << std::log2(floor_gap / mag) << ")" << std::endl
            << std::scientific << std::setprecision(3)
            << "  |plain+premap - chain|             " << worst
            << "   (relative 2^" << std::fixed << std::setprecision(2)
            << std::log2(worst / mag) << ")" << std::endl;
  // The premap route must not be further from the chain route than a second
  // encryption of the same message is -- within a small factor, since the two
  // are different COMPILATIONS of the composed matrix and each rounds its own
  // plaintexts (ConverterCompareTest's subject).
  EXPECT_LT(worst, 8.0 * floor_gap)
      << "the premap route disagrees by more than the encryption floor";
  EXPECT_LT(worst, 1e-4 * mag);
}

// ---------------------------------------------------------------------------
// 4. The INVERSE side. `forward_premap` relabels the transform's columns;
//    `inverse_premap` (added with this test) relabels its ROWS, which is what
//    a plain-native layer needs so that `SinCToSlot` writes the plain map
//    rather than the chain one. Same lattice, so the same ceiling argument
//    must hold -- and the SAME vector serves both directions, since a block
//    bit reversal is its own inverse.
// ---------------------------------------------------------------------------
TEST(PcPremap, TheInversePremapDoesNotGrowTheInverse) {
  Ring swtch("ci_ringswitch16_35_boot.json", {}, 0,
             /*build_user_interface=*/false);
  const int degree = swtch.Degree();
  const CiSwitchedCcmmLayout layout(degree, degree / kRank, kLanes);
  const auto premap = BuildPremap(degree);

  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> a(swtch.context, kLanes, /*forward_level=*/-1,
                          kInverseLevel, &layout, /*forward_premap=*/nullptr,
                          kBabySteps);
  const auto t1 = std::chrono::steady_clock::now();
  CiSinCConverter<word> b(swtch.context, kLanes, /*forward_level=*/-1,
                          kInverseLevel, &layout, /*forward_premap=*/nullptr,
                          kBabySteps, &premap);
  const auto t2 = std::chrono::steady_clock::now();

  ASSERT_NE(a.GetInverse(), nullptr);
  ASSERT_NE(b.GetInverse(), nullptr);
  const auto *ia = a.GetInverse();
  const auto *ib = b.GetInverse();

  std::cout << std::fixed << std::setprecision(2)
            << "  chain-native inverse : bs " << ia->GetBS() << " x gs "
            << ia->GetGS() << " = " << (ia->GetBS() * ia->GetGS())
            << " diagonals, " << (ia->PlaintextBytes() >> 20)
            << " MiB, built in " << Seconds(t0, t1) << " s" << std::endl;
  std::cout << "  plain + premap       : bs " << ib->GetBS() << " x gs "
            << ib->GetGS() << " = " << (ib->GetBS() * ib->GetGS())
            << " diagonals, " << (ib->PlaintextBytes() >> 20)
            << " MiB, built in " << Seconds(t1, t2) << " s" << std::endl;

  EXPECT_EQ(ia->GetBS(), ib->GetBS());
  EXPECT_EQ(ia->GetGS(), ib->GetGS());
  EXPECT_EQ(ia->GetPreRotationAmount(), ib->GetPreRotationAmount());
  EXPECT_EQ(ia->PlaintextBytes(), ib->PlaintextBytes());
  EXPECT_LE(ib->GetBS() * ib->GetGS(), degree / kLanes);
}

// ---------------------------------------------------------------------------
// 5. And it is the same values at the caller's addresses: ONE ciphertext
//    through both inverses, read at the chain map's slots out of the first
//    and at the plain map's out of the second.
// ---------------------------------------------------------------------------
TEST(PcPremap, ThePlainMapInverseIsTheChainMapRelabelled) {
  Ring boot("ci16_35.json");
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  const int degree = swtch.Degree();
  const CiSwitchedCcmmLayout layout(degree, degree / kRank, kLanes);
  const auto premap = BuildPremap(degree);

  CiSinCConverter<word> a(swtch.context, kLanes, -1, kInverseLevel, &layout,
                          nullptr, kBabySteps);
  CiSinCConverter<word> b(swtch.context, kLanes, -1, kInverseLevel, &layout,
                          nullptr, kBabySteps, &premap);
  EvkRequest req;
  a.AddRequiredRotations(req);
  b.AddRequiredRotations(req);
  swtch.ui->PrepareRotationKey(req);

  // Any ciphertext at the inverse's level: the claim is about where the
  // OUTPUT rows are written, and the input convention is untouched.
  std::mt19937_64 gen(0x1E5E9A17ULL);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Complex> msg(degree);
  for (auto &v : msg) v = Complex(dist(gen), 0.0);
  const double s = boot.param->GetScale(kInverseLevel);
  Plaintext<word> pt;
  boot.context->encoder_.Encode(pt, kInverseLevel, s, msg);
  Ciphertext<word> in, in2;
  boot.ui->Encrypt(in, pt);
  boot.ui->Encrypt(in2, pt);  // the control's second encryption

  Ciphertext<word> oa, oa2, ob;
  a.SinCToSlot(swtch.context, oa, in, swtch.ui->GetEvkMap());
  a.SinCToSlot(swtch.context, oa2, in2, swtch.ui->GetEvkMap());
  b.SinCToSlot(swtch.context, ob, in, swtch.ui->GetEvkMap());

  const auto read = [&](const Ciphertext<word> &ct) {
    Plaintext<word> p;
    boot.ui->Decrypt(p, ct);
    std::vector<Complex> v;
    boot.context->encoder_.Decode(v, p);
    return v;
  };
  const std::vector<Complex> va = read(oa), va2 = read(oa2), vb = read(ob);

  // Read each at ITS OWN map's primary addresses -- the only slots the
  // inverse gives meaning to (EvalSpecialFFT.h: it "returns the true
  // (unsummed) values at their primary addresses").
  const CiBatchLayout plain(degree, kTokens);
  const CiBatchLayout chain(degree, kTokens, kLanes, kRank);
  double mag = 0.0, floor_gap = 0.0, worst = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int bi = 0; bi < kInstances; bi++) {
      const double ca = va[chain.Slot(t, bi)].real();
      const double ca2 = va2[chain.Slot(t, bi)].real();
      const double pb = vb[plain.Slot(t, bi)].real();
      mag = std::max(mag, std::abs(ca));
      floor_gap = std::max(floor_gap, std::abs(ca - ca2));
      worst = std::max(worst, std::abs(ca - pb));
    }
  }
  std::cout << std::scientific << std::setprecision(3)
            << "  |chain inverse| max                 " << mag << std::endl
            << "  CONTROL |chain(enc1) - chain(enc2)| " << floor_gap
            << "   (relative 2^" << std::fixed << std::setprecision(2)
            << std::log2(floor_gap / mag) << ")" << std::endl
            << std::scientific << std::setprecision(3)
            << "  |plain+premap - chain| at own slots " << worst
            << "   (relative 2^" << std::fixed << std::setprecision(2)
            << std::log2(worst / mag) << ")" << std::endl;
  EXPECT_LT(worst, 8.0 * floor_gap)
      << "the inverse premap disagrees by more than the encryption floor";
  EXPECT_LT(worst, 1e-4 * mag);
}

namespace {

int Gcd(int a, int b) {
  while (b != 0) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a < 0 ? -a : a;
}

// `CiSinCBasis::Split`, so that what this test prices is what a phase of the
// real chain would be given: at most 32 baby steps, never a gs == 1 layout.
std::pair<int, int> Split(int num_diag) {
  int lg = 0;
  while ((1 << lg) < num_diag) lg++;
  int bs = 1 << ((lg + 1) / 2);
  if (bs > 32) bs = 32;
  if (bs > num_diag) bs = 1 << lg;
  if (bs < 1) bs = 1;
  int gs = (num_diag + bs - 1) / bs;
  if (gs < 2) {
    bs = (num_diag + 1) / 2;
    gs = 2;
  }
  return {bs, gs};
}

}  // namespace

// ---------------------------------------------------------------------------
// 6. WHERE THE PLAIN MAP IS NOT FREE -- and what it costs there.
//
//    The converters fold the premap at no cost because every diagonal they
//    carry already sits on the stride-`sub_degree` lattice: a block
//    relabelling only moves a diagonal to another lattice point, and the
//    compiled matrix is already at that lattice's `degree / sub_degree`
//    ceiling. The tower's LANE PREFIX is the opposite. It is
//    `CiButterflyStages(0 .. log2 T_l - 1)` -- it mixes only WITHIN a lane
//    group, so its `2 T_l - 1` offsets are lane offsets, off the lattice
//    entirely -- and composing a block premap on its output side multiplies
//    the two offset sets with nothing to collide. That is checked below in
//    the same binary as the rest, and it is why the fused score return
//    (`CiBatchAttention::BootScoresFused`, one prefix call per score
//    ciphertext, 128 a head) must keep the scores CHAIN-addressed.
//
//    Which is affordable only because of the softmax's shape: the one thing
//    in the score branch that a plain-mapped PC-attention has to meet is the
//    Euclidean-norm accumulator, and that is ONE ciphertext a head
//    (`Config::score_top`: "the softmax's Euclidean-norm accumulator (ONE
//    ciphertext a head) bootstraps on its own short ring"). So the join is
//    32 standalone permutations a layer, and this test prices one.
//
//    The price is small for a reason worth writing down. An 11-bit reversal
//    pairs bit i with bit 10 - i and fixes the middle bit, so the
//    displacement `rev(b) - b` is a sum of five independent terms
//    `2^(10-i) - 2^i`, each taken with a sign or dropped: 3^5 = 243 distinct
//    displacements out of 2048 blocks, and 3^floor(bits/2) at any width.
// ---------------------------------------------------------------------------
TEST(PcPremap, TheStandaloneBlockPermutationIsThreeToTheFive) {
  Ring boot("ci16_35.json");
  const int degree = boot.Degree();
  const auto premap = BuildPremap(degree);
  const int num_blocks = degree / kLanes;
  const int bits = 31 - __builtin_clz(num_blocks);
  ASSERT_EQ(1 << bits, num_blocks);

  // chain -> plain as a matrix: the output slot `p` is a PLAIN address, and
  // it takes whatever the CHAIN map put the same (token, instance) at. The
  // StripedMatrix convention is `m[off][row]` = the entry at (row, row+off).
  cheddar::StripedMatrix perm(degree, degree);
  for (int p = 0; p < degree; p++) {
    const int c = premap[p / kLanes] * kLanes + (p % kLanes);
    const int off = ((c - p) % degree + degree) % degree;
    perm.try_emplace(off, degree, Complex(0.0, 0.0));
    perm[off][p] = Complex(1.0, 0.0);
  }
  const int nd = perm.GetNumDiag();
  int three_to_the = 1;
  for (int i = 0; i < bits / 2; i++) three_to_the *= 3;
  EXPECT_EQ(nd, three_to_the)
      << "the displacement set is not a bit reversal's " << bits / 2
      << " independent signed terms";

  // The BSGS the code's own rule gives it. `pre_rotation` is left at 0: the
  // window would shrink the span, but it also leaves the output rotated by
  // the window (`CiSinCBasis::Compile` carries it to a closing HRot), and
  // what is being priced here is a transform that stands alone.
  int gcd = 0, max_rot = 0;
  for (const auto &[idx, unused] : perm) {
    const int rot = ((idx % degree) + degree) % degree;
    gcd = Gcd(gcd, rot);
    max_rot = std::max(max_rot, rot);
  }
  const int span = (gcd > 0) ? max_rot / gcd + 1 : nd;
  const auto [bs, gs] = Split(std::max(nd, span));

  constexpr int kLevel = 4;
  const auto t0 = std::chrono::steady_clock::now();
  cheddar::LinearTransform<word> lt(boot.context, perm, kLevel,
                                    boot.param->GetRescalePrimeProd(kLevel), bs,
                                    gs, /*pre_rotation=*/0,
                                    /*additional_pt_rot=*/0);
  const auto t1 = std::chrono::steady_clock::now();
  EvkRequest req;
  lt.AddRequiredRotations(req);
  boot.ui->PrepareRotationKey(req);

  std::cout << std::fixed << std::setprecision(2)
            << "  displacements (diagonals)        : " << nd << " of "
            << num_blocks << "   (3^" << (bits / 2) << ")" << std::endl
            << "  on stride " << gcd << ", span " << span << ", BSGS " << bs
            << "x" << gs << std::endl
            << "  KEY SWITCHES a call              : " << req.size()
            << std::endl
            << "  plaintexts                       : "
            << (lt.PlaintextBytes() >> 20) << " MiB, built in "
            << Seconds(t0, t1) << " s" << std::endl;

  // The negative half of the claim, in the same binary.
  std::set<int> composed;
  for (int q = 0; q < num_blocks; q++) {
    const int shift = ((premap[q] - q) % num_blocks + num_blocks) % num_blocks;
    for (int d = -(kLanes - 1); d < kLanes; d++) {
      composed.insert(((shift * kLanes + d) % degree + degree) % degree);
    }
  }
  std::cout << "  the tower's lane prefix alone    : " << (2 * kLanes - 1)
            << " diagonals" << std::endl
            << "  the lane prefix o this premap    : " << composed.size()
            << " diagonals  (= " << nd << " x " << (2 * kLanes - 1)
            << ", nothing collides)" << std::endl;
  EXPECT_EQ(static_cast<int>(composed.size()), nd * (2 * kLanes - 1));

  // And it computes the permutation.
  std::mt19937_64 gen(0xB10CBEEDULL);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(static_cast<size_t>(kTokens) * kInstances);
  for (auto &v : x) v = dist(gen);

  const CiBatchLayout plain(degree, kTokens);
  const CiBatchLayout chain(degree, kTokens, kLanes, kRank);
  std::vector<Complex> mc(degree, Complex(0.0, 0.0));
  for (int t = 0; t < kTokens; t++) {
    for (int bi = 0; bi < kInstances; bi++) {
      mc[chain.Slot(t, bi)] =
          Complex(x[static_cast<size_t>(t) * kInstances + bi], 0.0);
    }
  }
  Plaintext<word> pt;
  boot.context->encoder_.Encode(pt, kLevel, boot.param->GetScale(kLevel), mc);
  Ciphertext<word> in;
  boot.ui->Encrypt(in, pt);
  Ciphertext<word> out;
  lt.Evaluate(boot.context, out, in, boot.ui->GetEvkMap());

  Plaintext<word> got;
  boot.ui->Decrypt(got, out);
  std::vector<Complex> v;
  boot.context->encoder_.Decode(v, got);

  double mag = 0.0, worst = 0.0, at_chain = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int bi = 0; bi < kInstances; bi++) {
      const double want = x[static_cast<size_t>(t) * kInstances + bi];
      mag = std::max(mag, std::abs(want));
      worst = std::max(worst, std::abs(v[plain.Slot(t, bi)].real() - want));
      // A guard against a silently-identity transform: read the OUTPUT at
      // the addresses the input used and the values must have moved.
      at_chain = std::max(at_chain, std::abs(v[chain.Slot(t, bi)].real() - want));
    }
  }
  std::cout << std::scientific << std::setprecision(3)
            << "  |permuted - exact| at plain slots: " << worst
            << "   (relative 2^" << std::fixed << std::setprecision(2)
            << std::log2(worst / mag) << ")" << std::endl
            << std::scientific << std::setprecision(3)
            << "  the same read at the CHAIN slots : " << at_chain
            << "   (must be O(1): the map is not the identity)" << std::endl;
  EXPECT_LT(worst, 1e-4 * mag);
  EXPECT_GT(at_chain, 0.1 * mag);
}

// ---------------------------------------------------------------------------
// 7. WHAT THE PLAIN MAP IS FOR. A PC-attention plaintext is constant over the
//    token axis and varies over the instance axis, and under the plain map
//    that is a SUBRING element -- so it is not a new operator at all, it is
//    `SubringMatrixHandler` ([KANG] Algorithm 1: depth 1, no rotation, no
//    automorphism, no relinearization key), which is already in the tree.
//    `SubringMatrix.cu` reads
//
//        lanes      = conjugate_invariant ? sub_degree : sub_degree / 2
//        num_blocks = degree / sub_degree
//
//    so on this ring at `sub_degree = 512` that is 512 REAL lanes and 128
//    blocks -- numerically the plain map's own two axes, 512 instances and
//    128 tokens.
//
//    The arithmetic does NOT settle the ORDER, and the order is the whole
//    question. `EncodeWeights` hands lane `t` to `EncodeSinC`'s block-0
//    transform; a subring element then multiplies full slot `s` by its
//    k-point spectrum read at SOME index, and whether that index is `s % 512`
//    -- so that lane `t` is `CiBatchLayout::Slot`'s instance `t`, at every
//    token -- is a fact about the encoder's slot convention, not about the
//    algebra. If it is a permutation instead, a plain-native PC-attention
//    has to pre-permute every weight; if the lanes leak, the whole approach
//    is gone. So this measures three things in one pass: that the scaling is
//    PERIODIC in the slot index (the lanes are the structure), that the 512
//    lanes are INDEPENDENT (no blend), and which lane lands where.
//
//    The negative control is in the same read: under the CHAIN map one
//    instance's 128 tokens do not share a lane, which is exactly why the map
//    matters.
// ---------------------------------------------------------------------------
TEST(PcPremap, TheSubringLaneIsThePlainMapsInstance) {
  Ring boot("ci16_35.json");
  const int degree = boot.Degree();
  ASSERT_TRUE(boot.param->conjugate_invariant_)
      << "the lane count is sub_degree only on the conjugate-invariant ring";
  constexpr int kSubDegree = kInstances;  // 512, the batch
  ASSERT_EQ(degree / kSubDegree, kTokens) << "blocks should be the tokens";

  constexpr int kLevel = 4;
  const double scale = boot.param->GetScale(kLevel);

  // The weight: one distinct value a lane, laid on a ramp so that the lane
  // index can be read straight back off the value. `cols_in = cols_out = 1`
  // -- this is a scaling, and the only question is which slots each lane
  // scales.
  const auto lane_value = [](int t) { return (t + 1) / 1024.0; };
  std::vector<std::vector<Complex>> u(kSubDegree,
                                      std::vector<Complex>(1, Complex()));
  for (int t = 0; t < kSubDegree; t++) u[t][0] = Complex(lane_value(t), 0.0);

  cheddar::SubringMatrixHandler<word> subring(*boot.param,
                                              boot.context->encoder_);
  cheddar::SubringWeights<word> weights;
  subring.EncodeWeights(weights, kLevel, scale, u, /*cols_in=*/1,
                        /*cols_out=*/1, kSubDegree);

  // The store, while it is in hand: this is the one line that stands between
  // the algebra and a per-user PC-attention, so price it here rather than
  // assert it from the source.
  const cheddar::NPInfo np = boot.param->LevelToNP(kLevel, /*num_aux=*/0);
  const size_t words = static_cast<size_t>(np.GetNumTotal()) * degree;
  // What the same element needs if it is stored as what it is: the RNS limbs
  // are needed either way, the 128 blocks are not.
  const size_t compact_words = static_cast<size_t>(np.GetNumTotal()) * kSubDegree;
  EXPECT_EQ(weights.data_.size(), words)
      << "EncodeWeights expands a subring element to the full ring degree";
  EXPECT_EQ(words / compact_words, static_cast<size_t>(degree / kSubDegree));

  // The ciphertext: an ORDINARY slot message, as the batched layer's channels
  // are. A random one, not a constant -- a constant would not notice a
  // transform that mixed the blocks.
  std::mt19937_64 gen(0xD1F7C0DEULL);
  std::uniform_real_distribution<double> dist(0.5, 1.0);
  std::vector<Complex> z(degree);
  for (int s = 0; s < degree; s++) z[s] = Complex(dist(gen), 0.0);

  Plaintext<word> pt;
  boot.context->encoder_.Encode(pt, kLevel, scale, z);
  std::vector<Ciphertext<word>> in(1);
  boot.ui->Encrypt(in[0], pt);

  std::vector<Ciphertext<word>> res;
  subring.Multiply(boot.context, res, weights, in);
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(boot.param->NPToLevel(res[0].GetNP()), kLevel - 1)
      << "Algorithm 1 rescales, so the output is one level down";

  Plaintext<word> out;
  boot.ui->Decrypt(out, res[0]);
  std::vector<Complex> v;
  boot.context->encoder_.Decode(v, out);
  ASSERT_EQ(static_cast<int>(v.size()), degree);

  // Read the lane index off every slot. A blended lane -- the leak this is
  // here to catch -- would land between two lane values, half a spacing away
  // (4.9e-4), so the residual is the independence check.
  std::vector<int> lane_of(degree, -1);
  double worst = 0.0;
  int worst_slot = -1;
  for (int s = 0; s < degree; s++) {
    const double ratio = v[s].real() / z[s].real();
    int t = static_cast<int>(std::lround(ratio * 1024.0)) - 1;
    t = std::max(0, std::min(kSubDegree - 1, t));
    const double gap = std::abs(ratio - lane_value(t));
    if (gap > worst) {
      worst = gap;
      worst_slot = s;
    }
    lane_of[s] = t;
  }

  // (a) periodic with the sub-degree: the lanes ARE the structure.
  int off_period = 0;
  for (int s = 0; s < degree; s++) {
    if (lane_of[s] != lane_of[s % kSubDegree]) off_period++;
  }
  // (b) a bijection on one period: 512 lanes, none lost, none doubled. Checked
  //     separately from (c) so that a mere reordering reads differently from
  //     a leak.
  std::vector<int> seen(kSubDegree, 0);
  for (int t = 0; t < kSubDegree; t++) seen[lane_of[t]]++;
  int not_once = 0;
  for (int t = 0; t < kSubDegree; t++) {
    if (seen[t] != 1) not_once++;
  }
  // (c) and the order itself.
  int off_identity = 0;
  for (int s = 0; s < degree; s++) {
    if (lane_of[s] != s % kSubDegree) off_identity++;
  }

  // In the layer's own terms: under the plain map, instance b is scaled by
  // lane b at every one of its tokens.
  const CiBatchLayout plain(degree, kTokens);
  const CiBatchLayout chain(degree, kTokens, kLanes, kRank);
  int plain_bad = 0;
  for (int t = 0; t < kTokens; t++) {
    for (int b = 0; b < kInstances; b++) {
      if (lane_of[plain.Slot(t, b)] != b) plain_bad++;
    }
  }
  // The control: the same weight read through the CHAIN map, one instance
  // across its tokens.
  std::set<int> chain_lanes;
  for (int t = 0; t < kTokens; t++) {
    chain_lanes.insert(lane_of[chain.Slot(t, 0)]);
  }

  std::cout << "  sub_degree " << kSubDegree << " on R+ : " << kSubDegree
            << " real lanes x " << (degree / kSubDegree) << " blocks"
            << std::endl
            << "  periodic in the slot index       : "
            << (off_period == 0 ? "YES" : "NO") << "  (" << off_period
            << " slots off)" << std::endl
            << "  512 distinct lanes on one period : "
            << (not_once == 0 ? "YES" : "NO") << "  (" << not_once
            << " lanes not seen exactly once)" << std::endl
            << "  lane t scales the slots s = t (mod k): "
            << (off_identity == 0 ? "YES" : "NO") << "  (" << off_identity
            << " slots off)" << std::endl
            << "  => under the PLAIN map lane t IS instance t, every token: "
            << (plain_bad == 0 ? "YES" : "NO") << std::endl
            << "  CONTROL: under the CHAIN map instance 0 sees "
            << chain_lanes.size() << " different lanes across its " << kTokens
            << " tokens" << std::endl
            << std::scientific << std::setprecision(3)
            << "  |ratio - lane value| worst       : " << worst << " at slot "
            << worst_slot << "   (lane spacing " << (1.0 / 1024.0) << ")"
            << std::endl
            << std::fixed << std::setprecision(1)
            << "  EncodeWeights stored             : "
            << (weights.data_.size() * sizeof(word) / 1024) << " KiB ("
            << np.GetNumTotal() << " limbs x " << degree << ")" << std::endl
            << "  a COMPACT subring store would be : "
            << (compact_words * sizeof(word) / 1024) << " KiB ("
            << np.GetNumTotal() << " limbs x " << kSubDegree << ") -- "
            << (words / compact_words) << "x, the block count" << std::endl;

  EXPECT_LT(worst, 2e-4) << "a lane blended into its neighbours";
  EXPECT_EQ(off_period, 0) << "the scaling is not periodic in the slot index";
  EXPECT_EQ(not_once, 0) << "one period does not see the 512 lanes exactly once";
  EXPECT_EQ(off_identity, 0) << "lane t does not scale slot t (mod k)";
  EXPECT_EQ(plain_bad, 0)
      << "a subring weight is not the plain map's per-instance plaintext";
  EXPECT_GT(chain_lanes.size(), 1u)
      << "the chain map should NOT give one instance a single lane";
}

// ---------------------------------------------------------------------------
// 8. THE BLOCKER, looked at directly. Test 7 priced it from the outside (128x,
//    the block count); this opens the buffer. A subring element has `k`
//    degrees of freedom, so its NTT-domain image can only take `k` distinct
//    values -- `q(X^d)` evaluates at `psi^(d * e)` and `psi^d` has order `2k`.
//    WHERE those k values sit in the `degree`-long buffer is the one thing a
//    compact store has to know, and it is a fact about the NTT's output
//    ordering, not about the subring: a natural-order NTT makes the buffer
//    periodic with period k, a Cooley-Tukey one (natural in, BIT-REVERSED
//    out, which is what `SubringCtMatrix.cu` says of its own) makes it
//    constant on runs of `d` instead. Both compress to k words; the gather
//    index differs (`x & (k-1)` against `x >> log2 d`), so measure it rather
//    than assume it.
// ---------------------------------------------------------------------------
TEST(PcPremap, TheExpandedSubringStoreIsOneBlockRepeated) {
  Ring boot("ci16_35.json");
  const int degree = boot.Degree();
  constexpr int kSubDegree = kInstances;       // 512
  const int num_blocks = degree / kSubDegree;  // d = 128
  const int log_blocks = 31 - __builtin_clz(num_blocks);
  ASSERT_EQ(1 << log_blocks, num_blocks);

  constexpr int kLevel = 4;
  const double scale = boot.param->GetScale(kLevel);
  const cheddar::NPInfo np = boot.param->LevelToNP(kLevel, /*num_aux=*/0);
  const int limbs = np.GetNumTotal();

  std::mt19937_64 gen(0xC0FFEE77ULL);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<std::vector<Complex>> u(kSubDegree,
                                      std::vector<Complex>(1, Complex()));
  for (int t = 0; t < kSubDegree; t++) u[t][0] = Complex(dist(gen), 0.0);

  cheddar::SubringMatrixHandler<word> subring(*boot.param,
                                              boot.context->encoder_);
  cheddar::SubringWeights<word> weights;
  subring.EncodeWeights(weights, kLevel, scale, u, 1, 1, kSubDegree);

  cheddar::HostVector<word> h;
  cheddar::CopyDeviceToHost(h, weights.data_);
  ASSERT_EQ(static_cast<int>(h.size()), limbs * degree);

  int period_off = 0, run_off = 0;
  for (int p = 0; p < limbs; p++) {
    const size_t base = static_cast<size_t>(p) * degree;
    for (int x = 0; x < degree; x++) {
      if (h[base + x] != h[base + (x & (kSubDegree - 1))]) period_off++;
      if (h[base + x] != h[base + ((x >> log_blocks) << log_blocks)]) run_off++;
    }
  }
  std::set<word> distinct;
  for (int x = 0; x < degree; x++) distinct.insert(h[x]);

  std::cout << "  limbs x degree                   : " << limbs << " x "
            << degree << std::endl
            << "  distinct words in limb 0         : " << distinct.size()
            << " (a subring element has " << kSubDegree << " freedoms)"
            << std::endl
            << "  H1  w[x] == w[x & (k-1)]         : "
            << (period_off == 0 ? "HOLDS" : "no") << "  (" << period_off
            << " off)" << std::endl
            << "  H2  w[x] == w[(x >> log d) << log d] : "
            << (run_off == 0 ? "HOLDS" : "no") << "  (" << run_off << " off)"
            << std::endl;

  EXPECT_LE(distinct.size(), static_cast<size_t>(kSubDegree))
      << "the NTT image of a subring element takes more than k values";
  EXPECT_EQ(distinct.size(), static_cast<size_t>(kSubDegree))
      << "a random subring element should use all k of its freedoms";
  EXPECT_EQ(run_off, 0)
      << "Cheddar's forward NTT leaves the image constant on runs of d; if "
         "this moved, the compact store's gather index moved with it";
}

// ---------------------------------------------------------------------------
// 9. THE COMPACT STORE. Test 8 says the `degree`-long entry is one `k`-word
//    block repeated `d` times, so `EncodeWeights(..., compact = true)` keeps
//    the k words and `Multiply` gathers them with `x >> log2 d`. The claim is
//    not that it is close: it is that it is the SAME PRODUCT, word for word,
//    because the gather hands the kernel the identical operand it read
//    before. Anything less would be a new approximation in a chain that has
//    none.
//
//    Then the point of the exercise, at the shape PC-attention has: the
//    contraction runs over `head_dim` ciphertexts, so the weights are
//    `cols_in = 128` entries and the store is what has to be streamed.
// ---------------------------------------------------------------------------
TEST(PcPremap, TheCompactSubringStoreIsWordForWord) {
  Ring boot("ci16_35.json");
  const int degree = boot.Degree();
  constexpr int kSubDegree = kInstances;       // 512
  const int num_blocks = degree / kSubDegree;  // d = 128
  constexpr int kLevel = 4;
  constexpr int kFanIn = 128;  // head_dim ciphertexts contracted
  constexpr int kReps = 20;
  const double scale = boot.param->GetScale(kLevel);

  std::mt19937_64 gen(0xC1EA9DA7ULL);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  cheddar::SubringMatrixHandler<word> subring(*boot.param,
                                              boot.context->encoder_);

  const auto same_words = [](const cheddar::DeviceVector<word> &x,
                             const cheddar::DeviceVector<word> &y) {
    cheddar::HostVector<word> a, b;
    cheddar::CopyDeviceToHost(a, x);
    cheddar::CopyDeviceToHost(b, y);
    if (a.size() != b.size()) return static_cast<size_t>(-1);
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) diff++;
    }
    return diff;
  };

  // One shared ciphertext message; the fan-in copies differ only in their
  // encryption noise, which the word-for-word comparison does not care about
  // because both products read the SAME ciphertexts.
  std::vector<Complex> z(degree);
  for (int s = 0; s < degree; s++) z[s] = Complex(dist(gen), 0.0);
  Plaintext<word> pt;
  boot.context->encoder_.Encode(pt, kLevel, scale, z);

  // --- A. one entry ------------------------------------------------------
  std::vector<std::vector<Complex>> u1(kSubDegree,
                                       std::vector<Complex>(1, Complex()));
  for (int t = 0; t < kSubDegree; t++) u1[t][0] = Complex(dist(gen), 0.0);

  cheddar::SubringWeights<word> wide, thin;
  subring.EncodeWeights(wide, kLevel, scale, u1, 1, 1, kSubDegree, 0, false);
  subring.EncodeWeights(thin, kLevel, scale, u1, 1, 1, kSubDegree, 0, true);
  EXPECT_EQ(wide.GetEntryDegree(), degree);
  EXPECT_EQ(thin.GetEntryDegree(), kSubDegree);
  ASSERT_EQ(thin.data_.size() * num_blocks, wide.data_.size());

  std::vector<Ciphertext<word>> one(1);
  boot.ui->Encrypt(one[0], pt);
  std::vector<Ciphertext<word>> rw, rt;
  subring.Multiply(boot.context, rw, wide, one);
  subring.Multiply(boot.context, rt, thin, one);
  ASSERT_EQ(rw.size(), 1u);
  ASSERT_EQ(rt.size(), 1u);
  const size_t diff_one =
      same_words(rw[0].bx_, rt[0].bx_) + same_words(rw[0].ax_, rt[0].ax_);
  EXPECT_EQ(diff_one, 0u) << "cols_in = 1: the gather is not the expansion";

  // --- B. the PC-attention shape -----------------------------------------
  std::vector<std::vector<Complex>> uf(kSubDegree,
                                       std::vector<Complex>(kFanIn));
  for (int t = 0; t < kSubDegree; t++) {
    for (int j = 0; j < kFanIn; j++) uf[t][j] = Complex(dist(gen), 0.0);
  }
  cheddar::SubringWeights<word> fat, lean;
  subring.EncodeWeights(fat, kLevel, scale, uf, kFanIn, 1, kSubDegree, 0,
                        false);
  subring.EncodeWeights(lean, kLevel, scale, uf, kFanIn, 1, kSubDegree, 0,
                        true);

  std::vector<Ciphertext<word>> many(kFanIn);
  for (int j = 0; j < kFanIn; j++) boot.ui->Encrypt(many[j], pt);

  subring.Multiply(boot.context, rw, fat, many);
  subring.Multiply(boot.context, rt, lean, many);
  const size_t diff_many =
      same_words(rw[0].bx_, rt[0].bx_) + same_words(rw[0].ax_, rt[0].ax_);
  EXPECT_EQ(diff_many, 0u) << "cols_in = 128: the gather is not the expansion";

  cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < kReps; r++) subring.Multiply(boot.context, rw, fat, many);
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  for (int r = 0; r < kReps; r++) {
    subring.Multiply(boot.context, rt, lean, many);
  }
  cudaDeviceSynchronize();
  const auto t2 = std::chrono::steady_clock::now();

  const double fat_mb = fat.data_.size() * sizeof(word) / 1048576.0;
  const double lean_mb = lean.data_.size() * sizeof(word) / 1048576.0;
  std::cout << std::fixed << std::setprecision(2)
            << "  cols_in " << kFanIn << " (head_dim), cols_out 1, k "
            << kSubDegree << std::endl
            << "  weight store expanded            : " << fat_mb << " MiB"
            << std::endl
            << "  weight store compact             : " << lean_mb << " MiB   ("
            << (fat_mb / lean_mb) << "x smaller)" << std::endl
            << "  Multiply expanded                : "
            << (1000.0 * Seconds(t0, t1) / kReps) << " ms" << std::endl
            << "  Multiply compact                 : "
            << (1000.0 * Seconds(t1, t2) / kReps) << " ms" << std::endl
            << "  word-for-word differences        : " << diff_one << " + "
            << diff_many << std::endl;

  EXPECT_EQ(lean.data_.size() * num_blocks, fat.data_.size());
}

// ---------------------------------------------------------------------------
// 10. THE KEY-TOKEN SHIFT, under the plain map. `CiBatchAttention` splits the
//     score product into two calls over the key tokens, and the second call's
//     tokens `T/2 .. T-1` have to arrive where the first call's sat. Under the
//     chain addressing that is ONE slot rotation by `lanes` (32), because bit
//     `T/2` of the token is bit 0 of the block index -- the constructor
//     asserts it so a layout change fails loudly.
//
//     Under the plain map the token is the SLOW axis, so the same shift is
//     `T/2 * B` slots. That is a much bigger index (32768 against 32) and the
//     question a plain-native attention has to answer is whether it is still
//     ONE rotation, with a key that exists. It is: a rotation index is an
//     index, its cost does not grow with its size, and the map is still
//     lane-preserving at the CC-MM's sub-degree. Checked on the device rather
//     than argued, because "the key exists" is not something arithmetic
//     settles.
// ---------------------------------------------------------------------------
TEST(PcPremap, ThePlainMapsKeyTokenShiftIsStillOneRotation) {
  Ring boot("ci16_35.json");
  const int degree = boot.Degree();
  const CiBatchLayout plain(degree, kTokens);
  const CiBatchLayout chain(degree, kTokens, kLanes, kRank);
  const int half = kTokens / 2;

  // The chain map's claim, as `CiBatchAttention`'s own constructor states it.
  for (int t = 0; t < half; t++) {
    for (int g = 0; g < kRank; g++) {
      ASSERT_EQ(chain.BlockOf(t + half, g), chain.BlockOf(t, g) + 1);
      ASSERT_EQ(chain.Slot(t + half, g * kLanes) - chain.Slot(t, g * kLanes),
                kLanes);
    }
  }
  // The plain map's.
  const int r = half * kInstances;
  for (int t = 0; t < half; t++) {
    for (int b = 0; b < kInstances; b++) {
      ASSERT_EQ(plain.Slot(t + half, b) - plain.Slot(t, b), r);
    }
  }
  ASSERT_EQ(r, 32768);
  ASSERT_EQ(r % kLanes, 0) << "the shift must stay lane-preserving";

  constexpr int kLevel = 4;
  EvkRequest req;
  req.AddRequest(r, kLevel);
  boot.ui->PrepareRotationKey(req);

  std::mt19937_64 gen(0x5A17EDULL);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(static_cast<size_t>(kTokens) * kInstances);
  for (auto &v : x) v = dist(gen);
  std::vector<Complex> m(degree, Complex(0.0, 0.0));
  for (int t = 0; t < kTokens; t++) {
    for (int b = 0; b < kInstances; b++) {
      m[plain.Slot(t, b)] =
          Complex(x[static_cast<size_t>(t) * kInstances + b], 0.0);
    }
  }
  Plaintext<word> pt;
  boot.context->encoder_.Encode(pt, kLevel, boot.param->GetScale(kLevel), m);
  Ciphertext<word> in, out;
  boot.ui->Encrypt(in, pt);
  boot.context->HRot(out, in, boot.ui->GetRotationKey(r), r);

  Plaintext<word> got;
  boot.ui->Decrypt(got, out);
  std::vector<Complex> v;
  boot.context->encoder_.Decode(v, got);

  double mag = 0.0, worst = 0.0, unmoved = 0.0;
  for (int t = 0; t < half; t++) {
    for (int b = 0; b < kInstances; b++) {
      const double want = x[static_cast<size_t>(t + half) * kInstances + b];
      mag = std::max(mag, std::abs(want));
      worst = std::max(worst, std::abs(v[plain.Slot(t, b)].real() - want));
      // The guard: the values must actually have travelled.
      unmoved = std::max(
          unmoved, std::abs(v[plain.Slot(t, b)].real() -
                            x[static_cast<size_t>(t) * kInstances + b]));
    }
  }
  std::cout << "  chain-map key-token shift        : " << kLanes
            << " slots (one block)" << std::endl
            << "  plain-map key-token shift        : " << r << " slots (T/2 x B"
            << "), ONE rotation either way" << std::endl
            << std::scientific << std::setprecision(3)
            << "  |shifted - exact| at plain slots : " << worst
            << "   (relative 2^" << std::fixed << std::setprecision(2)
            << std::log2(worst / mag) << ")" << std::endl;
  EXPECT_LT(worst, 1e-4 * mag);
  EXPECT_GT(unmoved, 0.1 * mag) << "the rotation did nothing";
}

// ---------------------------------------------------------------------------
// 11. AND THE SHIFT COMPOSES WITH THE PREMAP FOR FREE. The shift above is a
//     rotation because `CiBatchAttention` chose it over a second forward
//     converter (3.8 GiB of plaintexts) -- but it is ALSO a block-granularity
//     lane-preserving permutation, so it could instead ride the conversion as
//     part of the premap and cost no key switch at all. The ceiling argument
//     says the composition is free: the converter's diagonals are already on
//     the stride-`sub_degree` lattice and already at `degree / sub_degree`, so
//     ANY block permutation folds, and composing two of them is still one.
//
//     Measured here so the trade is a number rather than a hope: 1024 HRots a
//     layer (one a channel a call a kv head) against one more converter's
//     plaintexts. This test does not make the choice; it prices the option.
// ---------------------------------------------------------------------------
TEST(PcPremap, TheShiftComposedWithThePremapStillFoldsAtTheCeiling) {
  Ring swtch("ci_ringswitch16_35_boot.json", {}, 0,
             /*build_user_interface=*/false);
  const int degree = swtch.Degree();
  const CiSwitchedCcmmLayout layout(degree, degree / kRank, kLanes);
  const auto premap = BuildPremap(degree);
  const int num_blocks = degree / kLanes;
  const int shift_blocks = (kTokens / 2) * kInstances / kLanes;
  ASSERT_EQ(shift_blocks, 1024);

  // The composed map: take the shifted source block, then relabel it the way
  // the plain -> chain premap does.
  std::vector<int> composed(num_blocks);
  for (int p = 0; p < num_blocks; p++) {
    composed[p] = premap[(p + shift_blocks) % num_blocks];
  }
  std::vector<int> seen(num_blocks, 0);
  for (int p = 0; p < num_blocks; p++) seen[composed[p]]++;
  for (int b = 0; b < num_blocks; b++) {
    ASSERT_EQ(seen[b], 1) << "the composition is not a bijection at block " << b;
  }

  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> a(swtch.context, kLanes, kForwardLevel, -1, &layout,
                          nullptr, kBabySteps);
  const auto t1 = std::chrono::steady_clock::now();
  CiSinCConverter<word> b(swtch.context, kLanes, kForwardLevel, -1, &layout,
                          &premap, kBabySteps);
  const auto t2 = std::chrono::steady_clock::now();
  CiSinCConverter<word> c(swtch.context, kLanes, kForwardLevel, -1, &layout,
                          &composed, kBabySteps);
  const auto t3 = std::chrono::steady_clock::now();

  const auto *fa = a.GetForward();
  const auto *fb = b.GetForward();
  const auto *fc = c.GetForward();
  ASSERT_NE(fa, nullptr);
  ASSERT_NE(fb, nullptr);
  ASSERT_NE(fc, nullptr);

  std::cout << std::fixed << std::setprecision(2)
            << "  chain-native            : bs " << fa->GetBS() << " x gs "
            << fa->GetGS() << " = " << (fa->GetBS() * fa->GetGS())
            << " diagonals, " << (fa->PlaintextBytes() >> 20) << " MiB, "
            << Seconds(t0, t1) << " s" << std::endl
            << "  premap                  : bs " << fb->GetBS() << " x gs "
            << fb->GetGS() << " = " << (fb->GetBS() * fb->GetGS())
            << " diagonals, " << (fb->PlaintextBytes() >> 20) << " MiB, "
            << Seconds(t1, t2) << " s" << std::endl
            << "  premap o the shift      : bs " << fc->GetBS() << " x gs "
            << fc->GetGS() << " = " << (fc->GetBS() * fc->GetGS())
            << " diagonals, " << (fc->PlaintextBytes() >> 20) << " MiB, "
            << Seconds(t2, t3) << " s" << std::endl
            << "  ceiling (degree / sub_degree) = " << num_blocks << std::endl;

  EXPECT_EQ(fa->GetBS(), fc->GetBS());
  EXPECT_EQ(fa->GetGS(), fc->GetGS());
  EXPECT_EQ(fa->GetPreRotationAmount(), fc->GetPreRotationAmount());
  EXPECT_EQ(fa->PlaintextBytes(), fc->PlaintextBytes());
  EXPECT_LE(fc->GetBS() * fc->GetGS(), num_blocks);
}
