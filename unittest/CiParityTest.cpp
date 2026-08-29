// [32] -- Kim and Song, "Approximate Homomorphic Encryption over the
// Conjugate-invariant Ring", ICISC 2018 -- is the paper [SYLPH] section 2.1.1
// hands its reader when it says the conjugate-invariant variant "induces a
// number of technicalities". This file checks that Cheddar's R+ is that ring
// and that it keeps the three promises [32] section 5.1 makes for it.
//
// The ring is the same one, term for term:
//
//   [32] 4.3   Rq = {a(X) in Zq[X]/(X^n + 1) : a(X) = a(X^-1)}, stored as the
//              coefficients (a_0, ..., a_{n/2-1}) of a_0 + sum a_i (X^i+X^-i)
//   [32] 4.2   that basis and NOT {1, Y, Y^2, ...} for Y = X + X^-1, because
//              the power basis is not orthogonal under the canonical embedding
//   [32] 4.3   a -> a^ is NTT_{n/2} preceded by the per-index twist
//              a_j zeta_m^j + a_{n/2-j} zeta_m^{j-n/2}, whose inverse carries
//              a factor 2^-1
//
// which is `CiFoldKernel` / `CiUnfoldKernel` in NTT.cu and the c_j basis
// `CiRingTest` already pins. What is not pinned anywhere is section 5.1, and
// section 5.1 is the reason this branch exists:
//
//   claim 1  "it requires n log q bits to express an element ... [so] both
//            schemes essentially have the same key size and ciphertext size"
//   claim 2  "the maximum number of plaintexts packed in a single ciphertext
//            in our scheme is n, while that of HEAAN is (n/2)"
//   claim 3  "both schemes exploit the NTT of dimension n for a ring
//            multiplication, so they have almost same arithmetic complexity"
//
// `ci16_35` and `bootparam_35` are the pair that isolates them. They carry the
// SAME level_config, the same main/terminal/auxiliary counts, the same scale,
// the same encryption level and the same Hamming weights; they differ only in
// the ring, and so in the primes, which on R+ must be 1 mod 4N rather than
// 1 mod 2N. Claims 1 and 2 are therefore exact identities. Claim 3 is a cost
// statement and is MEASURED -- in ONE process, so that the two rings see the
// same card, the same RMM pool and the same warm-up, and alternating, so that
// a card drifting over the run drifts through both sides of every ratio.
//
// Claim 3 is the one Cheddar could fail. R+ multiplication is the ordinary
// negacyclic transform with the fold wrapped around it, and a fold is a full
// extra pass over every limb; section 1.5be moved most of it into the base
// conversion's registers, and what is left is the unfold pass, ModRaise and
// the plain-NTT encoder paths. Whether "most" is enough is exactly this
// measurement -- and it bounds how much of the CI bootstrap's 58.2 ms against
// the ordinary ring's 38.2 (1.5bd) can be charged to ring arithmetic rather
// than to the linear transforms, which is where that record puts it.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::Ciphertext;
using cheddar::Plaintext;

namespace {

constexpr const char *kCiParam = "ci16_35.json";
constexpr const char *kOrdParam = "bootparam_35.json";

// The levels the comparison is taken at. The dnum boundary makes a per-op cost
// piecewise in the limb count -- CLAUDE.md's A6000 table steps at level 23 --
// so one level could not say whether a ratio belongs to the ring or to the
// ladder.
const std::vector<int> kLevels = {0, 8, 19};

constexpr int kWarmUp = 10;
constexpr int kIters = 40;

// Median rather than mean: one hiccup in forty moves a mean by more than the
// effect being measured.
double MedianMs(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  const int n = static_cast<int>(v.size());
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

template <typename F>
double TimeMs(F &&op) {
  std::vector<double> samples;
  samples.reserve(kIters);
  for (int i = 0; i < kWarmUp; i++) op();
  cudaDeviceSynchronize();
  for (int i = 0; i < kIters; i++) {
    const auto t0 = std::chrono::steady_clock::now();
    op();
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return MedianMs(samples);
}

// A coefficient-encoded ciphertext at `level`. Coefficient encoding is the one
// encoding both rings take the same number of inputs for -- `degree` reals --
// so it is what makes the operands the same object on both sides. The slot
// counts differ by construction, and that difference is claim 2.
void MakeCt(Ciphertext<word> &ct, const Ring &r, int level, uint64_t seed) {
  const int degree = r.Degree();
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<double> coeffs(degree);
  for (auto &c : coeffs) c = dist(gen);
  Plaintext<word> pt;
  r.context->encoder_.EncodeCoeff(pt, level, r.param->GetScale(level), coeffs);
  r.ui->Encrypt(ct, pt);
}

struct Row {
  std::string name;
  double ci = 0.0;
  double ord = 0.0;
};

}  // namespace

TEST(CiParity, TheConjugateInvariantRingKeepsKimSongsThreeClaims) {
  Ring ci(kCiParam);
  Ring ord(kOrdParam);

  ASSERT_EQ(ci.Degree(), ord.Degree())
      << "the pair is only an A/B while the two rings have the same rank";
  const int degree = ci.Degree();

  // ---- claim 2: twice the slots ----------------------------------------
  //
  // [32] 4.1's packing sends a(Y) to (a(theta_j)) for 0 <= j < n/2, one REAL
  // slot per coefficient, against HEAAN's n/2 complex slots for n
  // coefficients. Cheddar spells it
  // `MaxNumSlots() = conjugate_invariant ? degree : degree / 2`.
  EXPECT_EQ(ci.param->MaxNumSlots(), degree);
  EXPECT_EQ(ord.param->MaxNumSlots(), degree / 2);
  EXPECT_EQ(ci.param->MaxNumSlots(), 2 * ord.param->MaxNumSlots())
      << "[32] 5.1 claim 2: R+ packs n reals where the cyclotomic ring packs "
         "n/2 complex";

  // ---- claim 1: the same ciphertext ------------------------------------
  //
  // "n log q bits to express an element" -- a CI element is stored by its n
  // free coefficients and NOT by the 2n of the ambient Z[X]/(X^2n + 1) it sits
  // inside, so a limb is the same buffer on both rings and a ciphertext is the
  // same number of words. This is the claim a NON-COMPACT representation would
  // fail, and it is what makes the branch comparable with the ordinary block
  // at all.
  for (int level : kLevels) {
    Ciphertext<word> a, b;
    MakeCt(a, ci, level, 0x51A + level);
    MakeCt(b, ord, level, 0x51A + level);
    const auto npa = a.GetNP(), npb = b.GetNP();
    EXPECT_EQ(npa.num_main_, npb.num_main_) << "level " << level;
    EXPECT_EQ(npa.num_ter_, npb.num_ter_) << "level " << level;
    EXPECT_EQ(npa.num_aux_, npb.num_aux_) << "level " << level;
    EXPECT_EQ(npa.GetNumTotal() * degree, npb.GetNumTotal() * degree)
        << "[32] 5.1 claim 1: a ciphertext must be the same size at level "
        << level;
  }

  // And the operands must still decrypt to what went in, or the timings below
  // are the cost of computing something else.
  EXPECT_LT(ci.CoeffRoundTrip(kLevels.back(), 0xC0FFEE), 1e-6);
  EXPECT_LT(ord.CoeffRoundTrip(kLevels.back(), 0xC0FFEE), 1e-6);

  // ---- claim 3: the same arithmetic -------------------------------------
  constexpr word kRotAmount = 1234;
  ci.ui->PrepareRotationKey(kRotAmount, ci.param->max_level_);
  ord.ui->PrepareRotationKey(kRotAmount, ord.param->max_level_);

  std::vector<Row> rows;
  double worst = 0.0;
  std::string worst_name;

  for (int level : kLevels) {
    Ciphertext<word> ci_a, ci_b, ord_a, ord_b;
    MakeCt(ci_a, ci, level, 0xA1 + level);
    MakeCt(ci_b, ci, level, 0xB2 + level);
    MakeCt(ord_a, ord, level, 0xA1 + level);
    MakeCt(ord_b, ord, level, 0xB2 + level);

    Plaintext<word> ci_pt, ord_pt;
    {
      std::mt19937_64 gen(0xD00D);
      std::uniform_real_distribution<double> dist(-0.5, 0.5);
      std::vector<double> c(degree);
      for (auto &v : c) v = dist(gen);
      ci.context->encoder_.EncodeCoeff(ci_pt, level, ci.param->GetScale(level),
                                       c);
      ord.context->encoder_.EncodeCoeff(ord_pt, level,
                                        ord.param->GetScale(level), c);
    }

    Ciphertext<word> res;
    auto row = [&](const std::string &name, auto &&ci_op, auto &&ord_op) {
      Row r;
      r.name = name + " @" + std::to_string(level);
      r.ci = TimeMs(ci_op);
      r.ord = TimeMs(ord_op);
      const double ratio = r.ci / r.ord;
      if (ratio > worst) {
        worst = ratio;
        worst_name = r.name;
      }
      rows.push_back(r);
    };

    row("Add", [&] { ci.context->Add(res, ci_a, ci_b); },
        [&] { ord.context->Add(res, ord_a, ord_b); });
    row("MultPt", [&] { ci.context->Mult(res, ci_a, ci_pt); },
        [&] { ord.context->Mult(res, ord_a, ord_pt); });
    row("HRot",
        [&] {
          ci.context->HRot(res, ci_a, ci.ui->GetRotationKey(kRotAmount),
                           kRotAmount);
        },
        [&] {
          ord.context->HRot(res, ord_a, ord.ui->GetRotationKey(kRotAmount),
                            kRotAmount);
        });
    if (level >= 1) {
      row("HMult",
          [&] {
            ci.context->HMult(res, ci_a, ci_b, ci.ui->GetMultiplicationKey(),
                              true);
          },
          [&] {
            ord.context->HMult(res, ord_a, ord_b,
                               ord.ui->GetMultiplicationKey(), true);
          });
      Ciphertext<word> ci_prod, ord_prod;
      ci.context->HMult(ci_prod, ci_a, ci_b, ci.ui->GetMultiplicationKey(),
                        false);
      ord.context->HMult(ord_prod, ord_a, ord_b,
                         ord.ui->GetMultiplicationKey(), false);
      row("Rescale", [&] { ci.context->Rescale(res, ci_prod); },
          [&] { ord.context->Rescale(res, ord_prod); });
    }
  }

  std::cout << "\n  [32] 5.1 claim 3, " << kCiParam << " against " << kOrdParam
            << ", median of " << kIters << "\n";
  std::cout << "    " << std::left << std::setw(14) << "op" << std::right
            << std::setw(11) << "R+ (ms)" << std::setw(11) << "ord (ms)"
            << std::setw(9) << "ratio" << std::endl;
  for (const auto &r : rows) {
    std::cout << "    " << std::left << std::setw(14) << r.name << std::right
              << std::fixed << std::setprecision(4) << std::setw(11) << r.ci
              << std::setw(11) << r.ord << std::setprecision(3) << std::setw(9)
              << (r.ci / r.ord) << std::endl;
  }
  std::cout << "    worst " << worst_name << " at " << std::setprecision(3)
            << worst << "x" << std::endl;

  // MEASURED ON AN A100, CUDA 13.0: element-wise operations are at 1.00-1.09x
  // and every path through a key switch or a ModSwitch is at 1.17-1.20x. nsys
  // over this binary attributes the second number to exactly two kernels, and
  // to nothing else:
  //
  //     ModSwitchMatrixMultCi   1259 calls   20.99 ms
  //     ModSwitchMatrixMult     1259 calls   13.89 ms   (the ordinary twin)
  //     CiUnfoldKernel           956 calls    6.08 ms   (R+ only)
  //     CiFoldKernel              38 calls    0.65 ms   (R+ only)
  //
  // Same call count, so the base conversion itself is 1.51x -- that is 1.5be's
  // refold in registers, and it is not free -- and the unfold rides beside it
  // as a separate full pass. The two sum to 13.83 ms against a CI side of
  // ~87.0 ms of the profile's 160.1, i.e. 18.9%, which is the ratio the table
  // above prints. So "almost same" holds at 1.00x for element-wise work and at
  // 1.18x for key switching, and the fold is the whole of the difference.
  //
  // The corollary is about the bootstrap, not about arithmetic: the CI Boot is
  // 58.2 ms against the ordinary ring's 38.2 (1.5bd), which is 1.52x, and ring
  // arithmetic accounts for 1.18 of it. The remaining 1.29x is the linear
  // transforms -- CoeffToSlot and SlotToCoeff over 65536 real slots against
  // 32768 complex ones -- which is where 1.5bd put it without a number.
  //
  // "Almost same" is not a number, so this bound is deliberately loose: it is
  // a regression guard, not a reading of the paper. A row near 1.0 is the fold
  // where 1.5be left it; a row that doubles means the fold has come back out
  // of the base conversion onto the main path, which is a real defect and the
  // only thing this assertion is for.
  EXPECT_LT(worst, 1.60) << "[32] 5.1 claim 3: R+ arithmetic has stopped being "
                            "comparable with the cyclotomic ring";
}
