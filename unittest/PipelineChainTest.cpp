// The whole descent and return, in one process.
//
//   65536  --RingSwitch-->  16 x 4096  --ModDecomp-->  16 x 16 x 256 (MLWE)
//          --PC-MM-->  --ModPack-->  16 x 4096  --Rescale-->  --SwitchBack-->  65536
//
// Every piece of this has its own passing test. That is not the same as the
// chain working, and the gap is specifically about **level and scale**, which
// no single-operator test can see:
//
//   * `PcmmHandler::Multiply` deliberately does not rescale, following
//     `Context::Mult`. So the product leaves at u.scale * ct.scale and
//     something downstream has to bring it back.
//   * The degree-4096 ring has exactly **one** multiplicative level. There is
//     no slack to absorb a mistake -- two levels would put log2 QP at 132 and
//     the ring under 128-bit security, so this is a wall, not a budget.
//   * The chain crosses three Contexts, and a scale that is merely *plausible*
//     survives every plain operation silently. It stops being silent at
//     `EvalPoly`, which aborts the process on a scale mismatch -- i.e. inside
//     bootstrapping, three layers above wherever the mistake was made.
//
// Working the budget out on paper first showed the trap is real and narrow:
//
//     u.scale   product   after rescale   canonical at level 0?
//     2^20      2^50      2^20            no, 10 bits low
//     2^30      2^60      2^30            YES
//     2^32      2^62      2^32            no, 2 bits high
//
// Only 2^30 lands on the ring's own level-0 scale. LlamaPcmmTest happens to use
// 2^20, which is fine in isolation and would not be fine here. That is exactly
// the class of thing this test exists to pin down.
//
// WHAT THE CHAIN COMPUTES. Composing the index maps -- RingSwitch takes
// m[i + 16s], ModDecomp then takes stride 16 again, ModPack and SwitchBack
// invert both -- the whole thing applies U to the middle digit:
//
//     final[i + 16r + 256t] = sum_j U[r][j] * m[i + 16j + 256t]
//
// independently for every (i, t). So the host reference is exact and needs no
// crypto, which is what makes this a real check rather than a smoke test.

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#include "core/RingSwitch.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::EvaluationKey;
using cheddar::MlweCiphertext;
using cheddar::MlweHandler;
using cheddar::PcmmHandler;
using cheddar::PlainMatrix;
using cheddar::Plaintext;
using cheddar::RingSwitchHandler;
using Ring = ringfixture::Ring<word>;

TEST(PipelineChain, DescendMultiplyAndReturn) {
  Ring big("ringswitch16_30.json");
  Ring small("ringdegree12_30.json");

  const int degree = big.Degree();          // 65536
  const int mid_degree = small.Degree();    // 4096
  const int small_degree = 256;             // the PC-MM ring
  const int rank = degree / mid_degree;     // 16, the ring-switch rank
  const int sub_rank = mid_degree / small_degree;  // 16, the ModDecomp rank
  const int inner = 256;                    // coefficients per MLWE part
  ASSERT_EQ(rank, 16);
  ASSERT_EQ(sub_rank, 16);

  // The ring's own scale at both ends, which is the only choice that keeps
  // level 0 canonical. See the table above.
  const int level = big.param->max_level_;
  ASSERT_EQ(level, 1) << "the small ring is supposed to have exactly one "
                         "multiplicative level";
  const double ct_scale = big.param->GetScale(level);
  const double u_scale = small.param->GetScale(level);

  big.ui->PrepareRingSwitchKey(mid_degree, small.ui->GetSecretCoeffs(), level);
  big.ui->PrepareInverseRingSwitchKey(mid_degree, small.ui->GetSecretCoeffs(),
                                      level);
  small.ui->PrepareModPackKeys(small_degree, level);

  // Input.
  std::mt19937_64 gen(0xC4A1);
  std::uniform_real_distribution<double> md(-1.0, 1.0);
  // U is kept small on purpose: sixteen terms at u.scale * ct.scale = 2^60 has
  // to stay under Q1/2 = 2^68.76, and |U| <= 0.25 leaves about seven bits.
  std::uniform_real_distribution<double> ud(-0.25, 0.25);

  std::vector<double> m(degree);
  for (auto &v : m) v = md(gen);
  std::vector<double> u_values(sub_rank * sub_rank);
  for (auto &v : u_values) v = ud(gen);

  Plaintext<word> pt;
  big.context->encoder_.EncodeCoeff(pt, level, ct_scale, m);
  Ciphertext<word> ct;
  big.ui->Encrypt(ct, pt);

  RingSwitchHandler<word> rs(big.context, small.context);
  MlweHandler<word> mlwe(*small.param, small.context->ntt_handler_);
  PcmmHandler<word> pcmm(*small.param);

  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, level, u_scale, u_values, sub_rank, sub_rank);

  // --- descend -------------------------------------------------------------
  std::vector<Ciphertext<word>> parts;
  rs.Switch(parts, ct, big.ui->GetRingSwitchKey(rank));
  ASSERT_EQ(static_cast<int>(parts.size()), rank);
  EXPECT_EQ(small.param->NPToLevel(parts[0].GetNP()), level);
  EXPECT_NEAR(parts[0].GetScale() / ct_scale, 1.0, 1e-9)
      << "a ring switch must not move the scale";

  std::vector<const EvaluationKey<word> *> modpack_keys(sub_rank);
  for (int j = 0; j < sub_rank; j++) {
    modpack_keys[j] = &small.ui->GetModPackKey(sub_rank, j);
  }

  std::vector<Ciphertext<word>> packed(rank);
  for (int i = 0; i < rank; i++) {
    std::vector<MlweCiphertext<word>> decomposed;
    mlwe.ModDecomp(decomposed, parts[i], small_degree);
    ASSERT_EQ(static_cast<int>(decomposed.size()), sub_rank);

    std::vector<MlweCiphertext<word>> product;
    pcmm.Multiply(product, u, decomposed);
    ASSERT_EQ(static_cast<int>(product.size()), sub_rank);
    if (i == 0) {
      EXPECT_NEAR(product[0].scale_ / (u_scale * ct_scale), 1.0, 1e-9)
          << "the PC-MM does not rescale, so the product carries the product "
             "of the scales";
    }

    Ciphertext<word> repacked;
    mlwe.ModPack(small.context, repacked, product, modpack_keys);
    if (i == 0) {
      EXPECT_EQ(small.param->NPToLevel(repacked.GetNP()), level)
          << "ModPack is a key switch and must not consume a level";
    }

    small.context->Rescale(packed[i], repacked);
    if (i == 0) {
      EXPECT_EQ(small.param->NPToLevel(packed[i].GetNP()), 0);
      // The whole point: after the single rescale this ring has to offer, the
      // scale is back on the value level 0 is defined at.
      EXPECT_NEAR(packed[i].GetScale() / small.param->GetScale(0), 1.0, 1e-6)
          << "scale drifted off the canonical level-0 value";
    }
  }

  // --- return --------------------------------------------------------------
  Ciphertext<word> final_ct;
  rs.SwitchBack(final_ct, packed, big.ui->GetInverseRingSwitchKey(rank));
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  EXPECT_EQ(big.param->NPToLevel(final_ct.GetNP()), 0);
  EXPECT_NEAR(final_ct.GetScale() / big.param->GetScale(0), 1.0, 1e-6)
      << "the ciphertext returns to the big ring off its canonical scale, "
         "which EvalPoly would later abort on";

  Plaintext<word> back;
  big.ui->Decrypt(back, final_ct);
  std::vector<double> got;
  big.context->encoder_.DecodeCoeff(got, back);
  ASSERT_EQ(static_cast<int>(got.size()), degree);

  // --- host reference ------------------------------------------------------
  double worst = 0.0, sum = 0.0;
  int worst_idx = -1;
  for (int i = 0; i < rank; i++) {
    for (int t = 0; t < inner; t++) {
      for (int r = 0; r < sub_rank; r++) {
        double want = 0.0;
        for (int j = 0; j < sub_rank; j++) {
          want += u_values[r * sub_rank + j] * m[i + rank * j + 256 * t];
        }
        const int idx = i + rank * r + 256 * t;
        const double d = std::abs(got[idx] - want);
        sum += d;
        if (d > worst) {
          worst = d;
          worst_idx = idx;
        }
      }
    }
  }

  std::cout << "chain 65536 -> 16 x 4096 -> 16 x 16 x 256 -> back: max |diff| "
            << worst << " (coefficient " << worst_idx << "), mean "
            << (sum / degree) << std::endl;

  // Four key switches and one rescale, all at dnum 3, on top of the encryption
  // noise. Still orders of magnitude below anything an index or scale error
  // would produce.
  EXPECT_LT(worst, 5e-3);
}
