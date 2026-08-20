// [SYLPH] table 4's CC-MM row, end to end, at Llama-3's own shape:
//
//   8 x 65536  --RingSwitch-->  128 x 4096  --[KANG] Alg. 4-->  --SwitchBack-->
//
// WHY THIS TEST AND NOT THE TWO IT COMPOSES. `RingSwitchTest` shows the switch
// carries a SinC encoding and `BatchCcmmTest` shows Algorithm 4 computes
// per-lane products. Neither says which big ciphertext and which slot a matrix
// entry has to occupy for the second to find what the first delivered, and
// that composition is the whole content of the attention path -- three index
// maps, each a bit-permutation of a different field. Getting one of them wrong
// produces a result of the right magnitude that is a transpose, or
// lane-swapped, or block-contiguous instead of interleaved.
//
// So the comparison is ENTRYWISE against per-lane host products, with two
// controls that have to fail: the transpose of the result, and the layout with
// each ciphertext's blocks contiguous rather than interleaved.
//
// THE SHAPE IS NOT ARBITRARY. sub_degree 32 puts the small ring's matrix at
// 4096/32 = 128 wide, which is exactly Llama-3's per-head attention product --
// T tokens by head_dim -- with 16 lanes, hence 16 heads per call and two calls
// for 32 heads. Per call each operand is 8 big ciphertexts and the arithmetic
// is 128*128*16 = 8*32768: the real half of eight big ciphertexts exactly.
//
// THE HOST HALF OF THE LAYOUT IS CHECKED WITHOUT CRYPTO. SwitchedCcmmLayout
// states an entry's home twice -- once as a slot index, which is what the
// pipeline needs upstream of EvaluateSlotToSinC, and once as a SinC message
// index, which is what the host encoder reads. Only the second is exercised by
// the product; the first is exercised by the transform, which needs a
// BootContext. So the two are tied here by the transform's own documented
// identity, entry by entry, and that is what makes the slot form usable
// upstream without a second GPU test.
//
// SEPARATE BINARY, two rings alive at once; see SmallRingNttTest.cpp.

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/SwitchedCcmm.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::Plaintext;
using cheddar::SwitchedCcmmHandler;
using cheddar::SwitchedCcmmLayout;
using Ring = ringfixture::Ring<word>;

namespace {

// The 2^35 pair is the one [SYLPH]'s ladder runs on and the one the block is
// built against; the 2^30 pair stays selectable so the regression that came
// first can still be run.
const char *SwitchParam() {
  const char *e = std::getenv("CHEDDAR_SWITCH_PARAM");
  return (e != nullptr && e[0] != 0) ? e : "ringswitch16_35.json";
}
const char *SmallParam() {
  const char *e = std::getenv("CHEDDAR_SMALL_PARAM");
  return (e != nullptr && e[0] != 0) ? e : "ringdegree12_35.json";
}

constexpr int kSubDegree = 32;

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

using Batch = std::vector<std::vector<std::vector<double>>>;  // [lane][i][j]

Batch RandomBatch(int lanes, int d, double bound, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(-bound, bound);
  Batch m(lanes, std::vector<std::vector<double>>(d, std::vector<double>(d)));
  for (auto &lane : m) {
    for (auto &row : lane) {
      for (auto &v : row) v = dist(gen);
    }
  }
  return m;
}

}  // namespace

// The layout is a claim about three composed index maps. This half of it needs
// no GPU: it says the slot form and the SinC form of the same entry differ by
// exactly the permutation EvalSpecialFFT documents.
TEST(SwitchedCcmm, TheSlotAndSinCFormsAgreeWithTheTransform) {
  const int big_degree = 65536, small_degree = 4096;
  SwitchedCcmmLayout layout(big_degree, small_degree, kSubDegree);
  ASSERT_EQ(layout.rank, 16);
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 16);
  ASSERT_EQ(layout.num_cts, 8);

  // EvalSpecialFFT.h: slot s = [A | r] with r the low log2(sub_degree/2) bits
  // goes to SinC index [BitRev_p(A) | r], p = log2(degree / sub_degree).
  const int p = Log2(big_degree / kSubDegree);
  const int log_lanes = Log2(layout.lanes);
  ASSERT_EQ(p, 11);

  long long checked = 0;
  for (int row = 0; row < layout.dim; row++) {
    for (int column = 0; column < layout.dim; column++) {
      for (int lane = 0; lane < layout.lanes; lane++) {
        int ct_slot, slot, ct_sinc, index;
        layout.Locate(row, column, lane, ct_slot, slot);
        layout.LocateSinC(row, column, lane, ct_sinc, index);
        ASSERT_EQ(ct_slot, ct_sinc);
        const int want = (BitRev(slot >> log_lanes, p) << log_lanes) |
                         (slot & (layout.lanes - 1));
        ASSERT_EQ(index, want)
            << "row " << row << " column " << column << " lane " << lane;
        int r2, c2, l2;
        ASSERT_TRUE(layout.Position(ct_slot, slot, r2, c2, l2));
        ASSERT_EQ(r2, row);
        ASSERT_EQ(c2, column);
        ASSERT_EQ(l2, lane);
        checked++;
      }
    }
  }
  EXPECT_EQ(checked, static_cast<long long>(layout.dim) * layout.dim *
                         layout.lanes);
  EXPECT_EQ(checked,
            static_cast<long long>(layout.num_cts) * (big_degree / 2))
      << "the operand should fill its ciphertexts exactly, with no waste";
}

TEST(SwitchedCcmm, ProducesPerLaneProductsAtLlamaShape) {
  Ring big(SwitchParam());
  Ring small(SmallParam());

  const int level = big.param->max_level_;
  ASSERT_EQ(level, 1) << "the product ring is supposed to hold one level";
  const double scale = big.param->GetScale(level);

  SwitchedCcmmLayout layout(big.Degree(), small.Degree(), kSubDegree);
  ASSERT_EQ(layout.dim, 128);
  ASSERT_EQ(layout.lanes, 16);
  ASSERT_EQ(layout.num_cts, 8);

  big.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                               level);
  big.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                      small.ui->GetSecretCoeffs(), level);

  SwitchedCcmmHandler<word> handler(big.context, small.context, kSubDegree);
  const std::vector<int> rot = handler.SmallRotationIndices();
  std::cout << "small-ring automorphism keys: " << rot.size() << std::endl;
  for (int idx : rot) small.ui->PrepareRotationKey(idx, level);

  // 128 terms at 0.08 leaves the sum near 1 and the encoded product two
  // decades under the level-1 modulus; the ring has no slack for more.
  const Batch a = RandomBatch(layout.lanes, layout.dim, 0.08, 0x5117A1);
  const Batch b = RandomBatch(layout.lanes, layout.dim, 0.08, 0x5117A2);

  auto encrypt = [&](const Batch &m, std::vector<Ciphertext<word>> &out) {
    out.resize(layout.num_cts);
    std::vector<std::vector<Complex>> msg(
        layout.num_cts, std::vector<Complex>(big.Degree() / 2, Complex(0)));
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        for (int lane = 0; lane < layout.lanes; lane++) {
          int ct, index;
          layout.LocateSinC(row, column, lane, ct, index);
          msg[ct][index] = Complex(m[lane][row][column], 0.0);
        }
      }
    }
    for (int c = 0; c < layout.num_cts; c++) {
      Plaintext<word> pt;
      big.context->encoder_.EncodeSinC(pt, level, scale, msg[c], kSubDegree);
      big.ui->Encrypt(out[c], pt);
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs;
  encrypt(a, lhs);
  encrypt(b, rhs);

  std::vector<Ciphertext<word>> res;
  handler.Multiply(res, lhs, rhs, big.ui->GetRingSwitchKey(layout.rank),
                   big.ui->GetInverseRingSwitchKey(layout.rank),
                   small.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), layout.num_cts);

  for (const auto &ct : res) {
    EXPECT_EQ(big.param->NPToLevel(ct.GetNP()), level - 1)
        << "the batch CC-MM spends exactly one level";
    EXPECT_NEAR(ct.GetScale() / big.param->GetScale(level - 1), 1.0, 1e-6)
        << "the result comes back off the canonical scale of its level, which "
           "EvalPoly would abort on several layers above here";
  }

  std::vector<std::vector<Complex>> got(layout.num_cts);
  for (int c = 0; c < layout.num_cts; c++) {
    Plaintext<word> pt;
    big.ui->Decrypt(pt, res[c]);
    big.context->encoder_.DecodeSinC(got[c], pt, kSubDegree);
    ASSERT_EQ(static_cast<int>(got[c].size()), big.Degree() / 2);
  }

  double worst = 0.0, sum = 0.0, transposed = 0.0, contiguous = 0.0;
  double biggest = 0.0;
  const int blocks_per_ct = layout.dim;  // small blocks each part carries
  for (int lane = 0; lane < layout.lanes; lane++) {
    for (int row = 0; row < layout.dim; row++) {
      for (int column = 0; column < layout.dim; column++) {
        double want = 0.0, want_t = 0.0;
        for (int x = 0; x < layout.dim; x++) {
          want += a[lane][row][x] * b[lane][x][column];
          want_t += a[lane][column][x] * b[lane][x][row];
        }
        biggest = std::max(biggest, std::abs(want));
        int ct, index;
        layout.LocateSinC(row, column, lane, ct, index);
        const double v = got[ct][index].real();
        const double d = std::abs(v - want);
        worst = std::max(worst, d);
        sum += d;
        transposed = std::max(transposed, std::abs(v - want_t));

        // The control layout: within a big ciphertext, put small ciphertext
        // j's blocks contiguously -- big block j * blocks + row -- instead of
        // interleaved as j + rank * row.
        const int j = column % layout.rank;
        const int contig = (j * blocks_per_ct + row) * layout.lanes + lane;
        if (contig < static_cast<int>(got[ct].size())) {
          contiguous =
              std::max(contiguous, std::abs(got[ct][contig].real() - want));
        }
      }
    }
  }
  const double mean =
      sum / (static_cast<double>(layout.lanes) * layout.dim * layout.dim);

  std::cout << "SwitchedCcmm " << layout.num_cts << " x " << big.Degree()
            << " -> " << layout.dim << " x " << small.Degree() << " ("
            << layout.dim << "x" << layout.dim << ", " << layout.lanes
            << " lanes): max |diff| = " << worst << ", mean " << mean
            << ", |want| <= " << biggest << std::endl;
  std::cout << "  controls: transposed result " << transposed
            << ", contiguous block layout " << contiguous << std::endl;

  EXPECT_LT(worst, 2e-3) << "the batch product did not land where the layout "
                            "says it should";
  EXPECT_GT(transposed, 1e-2)
      << "the result is symmetric enough that a transpose would pass, so this "
         "test is not checking the orientation";
  EXPECT_GT(contiguous, 1e-2)
      << "the blocks are NOT contiguous per ciphertext, so this test would "
         "pass against a layout that laid them out that way";
}
