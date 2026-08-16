// [KANG] (ePrint 2025/1957) Algorithm 1: batch plaintext-ciphertext matrix
// multiplication, `(B, A) <- (B*U, A*U)` over the subring R_k.
//
// SECOND STEP OF THE BATCH CC-MM. SinCEncodeTest established the encoding;
// this is the first algorithm that stands on it, and the one that shares its
// engine with Algorithm 4 -- the CC-MM's step 2 is the same contraction with a
// ciphertext-derived right operand instead of a plaintext one.
//
// WHAT THE TEST HAS TO SHOW. The claim is not "the product is correct" in the
// scalar sense; it is that ONE pass evaluates k/2 INDEPENDENT matrix products,
// with the d rows living inside a single ciphertext and the d' contraction
// running across ciphertexts. So the reference is a host loop over lanes, each
// doing its own (d x d') * (d' x d'') product, and the assertion is that every
// lane comes back with its own answer. A scalar-correct implementation that
// leaked between lanes -- the obvious failure, since all lanes share the same
// ciphertext words -- would pass a norm check and fail this one.
//
// LEVEL BUDGET. One plaintext multiplication and one rescale: depth 1, which
// is the entire budget of this ring. Inputs at level 1, output at level 0. No
// rotation, automorphism or relinearization key is involved, which is what
// [KANG] Table 1 claims for Algorithm 1 and what makes it affordable here.
//
// SEPARATE BINARY, degree 4096; see SmallRingNttTest.cpp for the trap.

#undef ENABLE_EXTENSION

#include <cmath>

#include "Testbed.h"
#include "core/SubringMatrix.h"

using word = uint32_t;

namespace {

// [SYLPH] section 3.3 runs the batch CC-MM in SinC_{2^7, 2^12}.
constexpr int kSubDegree = 128;
constexpr int kColsIn = 4;   // d', contracted across ciphertexts
constexpr int kColsOut = 3;  // d'', output ciphertexts

}  // namespace

// The product, against a host reference that is a loop over lanes.
TEST_P(Testbed32, BatchCpmmMatchesPerLaneMatrixProducts) {
  const int degree = param_->degree_;
  const int lanes = kSubDegree / 2;             // k/2 = 64 matrices at once
  const int num_blocks = degree / kSubDegree;   // d = 32 rows per ciphertext
  const int level = param_->max_level_;
  ASSERT_EQ(level, 1) << "this ring must have exactly one multiplicative level";

  const double scale = DetermineScale(level);

  SubringMatrixHandler<word> subring(*param_, context_->encoder_);

  // Inputs. z[j] is the message of ciphertext j, read as [block][lane].
  std::vector<std::vector<Complex>> z(kColsIn);
  std::vector<Ciphertext<word>> cts(kColsIn);
  for (int j = 0; j < kColsIn; j++) {
    GenerateRandomMessage(z[j], -1, -0.5, 0.5);
    ASSERT_EQ(static_cast<int>(z[j].size()), degree / 2);
    Plaintext<word> pt;
    context_->encoder_.EncodeSinC(pt, level, scale, z[j], kSubDegree);
    interface_->Encrypt(cts[j], pt);
  }

  // Weights. One (cols_in x cols_out) matrix per lane, so the lanes really do
  // carry different matrices -- identical ones would hide any leakage.
  std::vector<std::vector<Complex>> u(lanes);
  for (int t = 0; t < lanes; t++) {
    GenerateRandomMessage(u[t], kColsIn * kColsOut, -0.5, 0.5);
  }

  SubringWeights<word> weights;
  subring.EncodeWeights(weights, level, scale, u, kColsIn, kColsOut,
                        kSubDegree);

  std::vector<Ciphertext<word>> res;
  subring.Multiply(context_, res, weights, cts);
  ASSERT_EQ(static_cast<int>(res.size()), kColsOut);

  // The reference: for every lane t and every row (block) i,
  //     want[l][i][t] = sum_j z[j][i][t] * u[t][j][l]
  double worst = 0.0;
  for (int l = 0; l < kColsOut; l++) {
    ASSERT_EQ(param_->NPToLevel(res[l].GetNP()), level - 1)
        << "Algorithm 1 rescales, so the output is one level down";
    EXPECT_NEAR(res[l].GetScale() / param_->GetScale(level - 1), 1.0, 1e-6)
        << "the product should land on the canonical scale of its level";

    Plaintext<word> out;
    interface_->Decrypt(out, res[l]);
    std::vector<Complex> got;
    context_->encoder_.DecodeSinC(got, out, kSubDegree);
    ASSERT_EQ(static_cast<int>(got.size()), degree / 2);

    for (int i = 0; i < num_blocks; i++) {
      for (int t = 0; t < lanes; t++) {
        Complex want(0.0, 0.0);
        for (int j = 0; j < kColsIn; j++) {
          want += z[j][i * lanes + t] * u[t][j * kColsOut + l];
        }
        worst = std::max(worst, std::abs(got[i * lanes + t] - want));
      }
    }
  }

  std::cout << "batch CPMM: " << lanes << " lanes of (" << num_blocks << " x "
            << kColsIn << ")(" << kColsIn << " x " << kColsOut
            << "), max error " << worst << std::endl;
  ASSERT_LT(worst, 1e-4);
}

// The lanes must be independent. Driving one lane's weights to zero has to
// blank exactly that lane and leave every other one untouched -- if the
// contraction mixed lanes, or if EncodeWeights broadcast a lane across the
// batch, this is where it shows.
//
// Worth its own test because the previous one would still pass with all lanes
// carrying the same matrix, and that is the shape a broadcasting bug takes.
TEST_P(Testbed32, BatchCpmmLanesAreIndependent) {
  const int degree = param_->degree_;
  const int lanes = kSubDegree / 2;
  const int num_blocks = degree / kSubDegree;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);
  const int silenced = 7;

  SubringMatrixHandler<word> subring(*param_, context_->encoder_);

  std::vector<std::vector<Complex>> z(kColsIn);
  std::vector<Ciphertext<word>> cts(kColsIn);
  for (int j = 0; j < kColsIn; j++) {
    GenerateRandomMessage(z[j], -1, -0.5, 0.5);
    Plaintext<word> pt;
    context_->encoder_.EncodeSinC(pt, level, scale, z[j], kSubDegree);
    interface_->Encrypt(cts[j], pt);
  }

  std::vector<std::vector<Complex>> u(lanes);
  for (int t = 0; t < lanes; t++) {
    GenerateRandomMessage(u[t], kColsIn * kColsOut, -0.5, 0.5);
  }
  std::fill(u[silenced].begin(), u[silenced].end(), Complex(0.0, 0.0));

  SubringWeights<word> weights;
  subring.EncodeWeights(weights, level, scale, u, kColsIn, kColsOut,
                        kSubDegree);
  std::vector<Ciphertext<word>> res;
  subring.Multiply(context_, res, weights, cts);

  double worst_silenced = 0.0;
  double live_sum = 0.0;
  long long live_count = 0;
  for (int l = 0; l < kColsOut; l++) {
    Plaintext<word> out;
    interface_->Decrypt(out, res[l]);
    std::vector<Complex> got;
    context_->encoder_.DecodeSinC(got, out, kSubDegree);

    for (int i = 0; i < num_blocks; i++) {
      worst_silenced =
          std::max(worst_silenced, std::abs(got[i * lanes + silenced]));
      for (int t = 0; t < lanes; t++) {
        if (t == silenced) continue;
        live_sum += std::abs(got[i * lanes + t]);
        live_count++;
      }
    }
  }
  const double live_mean = live_sum / live_count;

  std::cout << "lane " << silenced << " silenced: max |output| there "
            << worst_silenced << ", mean |output| elsewhere " << live_mean
            << std::endl;
  ASSERT_LT(worst_silenced, 1e-4) << "a zeroed lane still produced output";
  // Without this the test would pass on an implementation that produced
  // nothing at all.
  ASSERT_GT(live_mean, 1e-2) << "no lane produced output, so the silence "
                                "above proves nothing";
}

INSTANTIATE_TEST_SUITE_P(
    BatchCpmm, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
