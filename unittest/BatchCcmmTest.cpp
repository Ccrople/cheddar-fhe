// Batch CCMM, [KANG] (ePrint 2025/1957) Algorithm 4 -- the last piece.
//
// THE REFERENCE IS THE PAPER'S OWN STATEMENT OF THE OUTPUT. Section 4.3 says
// Algorithm 4 returns a matrix encryption of MatEcd({M_l * M'_l}), so the
// check is a host loop over lanes, each doing its own d x d product. No
// commitment to the internal layout is needed, which is what made the earlier
// pieces awkward to test and makes this one straightforward.
//
// WHAT WOULD SLIP THROUGH A NORM CHECK. All k/2 lanes share the same
// ciphertext words, and the three CMTs move data across the Vec index, so the
// plausible failures are lanes leaking into each other or a transposed result
// -- both of which keep the magnitude about right. So the comparison is
// entrywise against per-lane products, which catches a transpose, and a second
// test silences one lane, which catches leakage between them.
//
// LEVEL. One multiplicative level, spent on step 2 and returned by the rescale
// of step 7; the three CMTs and the relinearization are free. Inputs at
// level 1, output at level 0, which is the entire budget of this ring.
//
// SEPARATE BINARY, degree 4096; see SmallRingNttTest.cpp for the trap.

#undef ENABLE_EXTENSION

#include <cmath>

#include "Testbed.h"
#include "core/BatchCcmm.h"

using word = uint32_t;

namespace {

constexpr int kSubDegree = 128;  // [SYLPH] 3.3: d = 32, k/2 = 64 lanes

using Batch = std::vector<std::vector<std::vector<Complex>>>;  // [lane][i][j]

Batch RandomBatch(int lanes, int d, double bound) {
  Batch m(lanes,
          std::vector<std::vector<Complex>>(d, std::vector<Complex>(d)));
  for (int t = 0; t < lanes; t++) {
    for (int i = 0; i < d; i++) {
      Random::SampleUniformComplex(m[t][i].data(), d, -bound, bound);
    }
  }
  return m;
}

}  // namespace

// Algorithm 4 against per-lane host products.
TEST_P(Testbed32, BatchCcmmMatchesPerLaneProducts) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const int lanes = kSubDegree / 2;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);
  ASSERT_EQ(d, 32);

  BatchCcmmHandler<word> ccmm(*param_, context_->ntt_handler_);
  for (int index : ccmm.RotationIndices(kSubDegree)) {
    interface_->PrepareRotationKey(index, level);
  }

  // Small entries: the product sums d = 32 of them, and the encoded result
  // sits at scale^2 = 2^60 against a level-1 modulus of 2^69.8.
  const Batch m = RandomBatch(lanes, d, 0.15);
  const Batch mp = RandomBatch(lanes, d, 0.15);

  // Ciphertext j carries column j: block i of its SinC message holds the
  // batch of entry (i, j), which is exactly MatEcd composed with Vec.
  auto encrypt = [&](const Batch &src, std::vector<Ciphertext<word>> &out) {
    out.resize(d);
    for (int j = 0; j < d; j++) {
      std::vector<Complex> message(degree / 2);
      for (int i = 0; i < d; i++) {
        for (int t = 0; t < lanes; t++) message[i * lanes + t] = src[t][i][j];
      }
      Plaintext<word> pt;
      context_->encoder_.EncodeSinC(pt, level, scale, message, kSubDegree);
      interface_->Encrypt(out[j], pt);
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs;
  encrypt(m, lhs);
  encrypt(mp, rhs);

  std::vector<Ciphertext<word>> res;
  ccmm.Multiply(context_, res, lhs, rhs, kSubDegree,
                interface_->GetEvkMap());
  ASSERT_EQ(static_cast<int>(res.size()), d);

  double worst = 0.0;
  for (int j = 0; j < d; j++) {
    ASSERT_EQ(param_->NPToLevel(res[j].GetNP()), level - 1)
        << "Algorithm 4 spends exactly one level";
    EXPECT_NEAR(res[j].GetScale() / param_->GetScale(level - 1), 1.0, 1e-6)
        << "the result should land on the canonical scale of its level";

    Plaintext<word> out;
    interface_->Decrypt(out, res[j]);
    std::vector<Complex> got;
    context_->encoder_.DecodeSinC(got, out, kSubDegree);

    for (int i = 0; i < d; i++) {
      for (int t = 0; t < lanes; t++) {
        Complex want(0.0, 0.0);
        for (int x = 0; x < d; x++) want += m[t][i][x] * mp[t][x][j];
        worst = std::max(worst, std::abs(got[i * lanes + t] - want));
      }
    }
  }

  std::cout << "batch CCMM: " << lanes << " lanes of " << d << "x" << d
            << ", max error " << worst << std::endl;
  ASSERT_LT(worst, 1e-2);
}

// One lane silenced. All k/2 lanes share the same ciphertext words and the
// CMTs move data across the Vec index, so leakage between lanes is the
// failure a norm check would miss.
TEST_P(Testbed32, BatchCcmmLanesStayIndependent) {
  const int degree = param_->degree_;
  const int d = degree / kSubDegree;
  const int lanes = kSubDegree / 2;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);
  const int silenced = 11;

  BatchCcmmHandler<word> ccmm(*param_, context_->ntt_handler_);
  for (int index : ccmm.RotationIndices(kSubDegree)) {
    interface_->PrepareRotationKey(index, level);
  }

  Batch m = RandomBatch(lanes, d, 0.15);
  const Batch mp = RandomBatch(lanes, d, 0.15);
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) m[silenced][i][j] = Complex(0.0, 0.0);
  }

  auto encrypt = [&](const Batch &src, std::vector<Ciphertext<word>> &out) {
    out.resize(d);
    for (int j = 0; j < d; j++) {
      std::vector<Complex> message(degree / 2);
      for (int i = 0; i < d; i++) {
        for (int t = 0; t < lanes; t++) message[i * lanes + t] = src[t][i][j];
      }
      Plaintext<word> pt;
      context_->encoder_.EncodeSinC(pt, level, scale, message, kSubDegree);
      interface_->Encrypt(out[j], pt);
    }
  };

  std::vector<Ciphertext<word>> lhs, rhs, res;
  encrypt(m, lhs);
  encrypt(mp, rhs);
  ccmm.Multiply(context_, res, lhs, rhs, kSubDegree,
                interface_->GetEvkMap());

  double worst_silenced = 0.0, live_sum = 0.0;
  long long live_count = 0;
  for (int j = 0; j < d; j++) {
    Plaintext<word> out;
    interface_->Decrypt(out, res[j]);
    std::vector<Complex> got;
    context_->encoder_.DecodeSinC(got, out, kSubDegree);
    for (int i = 0; i < d; i++) {
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
  ASSERT_LT(worst_silenced, 1e-2) << "a zeroed lane still produced output";
  // Otherwise the silence above would be satisfied by producing nothing.
  ASSERT_GT(live_mean, 1e-3) << "no lane produced output";
}

INSTANTIATE_TEST_SUITE_P(
    BatchCcmm, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
