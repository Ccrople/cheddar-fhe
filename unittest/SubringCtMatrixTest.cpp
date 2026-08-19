// The subring matrix product, [KANG] (ePrint 2025/1957) Algorithm 4 step 2.
//
// TWO RISKS, ONE TEST EACH.
//
// The index maps. ToMatrices reads Vec^d_k out of the ciphertext components
// and ToCiphertexts writes it back, column-wise or row-wise. A transposed or
// mis-strided gather is the classic failure, and it is caught by requiring the
// round trip to return the input ciphertexts unchanged.
//
// The transform. MultiplyMatrices runs a length-k negacyclic NTT built here
// rather than taken from Cheddar, so its twiddle order and normalisation are
// new code -- and a transform can be self-consistently wrong, inverting
// cleanly while computing the wrong convolution. That is why the O(k^2)
// direct product is kept: the two share no arithmetic beyond the modular
// multiply, so agreement is evidence. A round-trip test of the transform
// alone would NOT catch a wrong twiddle order; this does.
//
// SHAPES. d = 32 and k = 128 at degree 4096, which is [SYLPH] section 3.3.
// The product is (32 x 4) by (4 x 32), the shape Algorithm 4 step 2 has after
// its CMT: column-wise on the left, row-wise on the right, contracting over
// the ciphertext index of both.
//
// SEPARATE BINARY, degree 4096; see SmallRingNttTest.cpp for the trap.

#undef ENABLE_EXTENSION

#include <cmath>

#include "Testbed.h"
#include "core/SubringCtMatrix.h"

using word = uint32_t;

namespace {

constexpr int kSubDegree = 128;
constexpr int kNumCts = 4;

// Counted rather than asserted per element: the buffers hold hundreds of
// thousands of words and a gtest assertion each would dominate the run.
long long CountMismatches(const HostVector<word> &got,
                          const HostVector<word> &want,
                          const std::vector<word> &primes, int sub_degree,
                          int num_total_primes, std::string *first) {
  long long mismatches = 0;
  const size_t entries = got.size() / (num_total_primes * sub_degree);
  for (size_t e = 0; e < entries; e++) {
    for (int limb = 0; limb < num_total_primes; limb++) {
      const uint64_t p = primes[limb];
      for (int t = 0; t < sub_degree; t++) {
        const size_t idx = (e * num_total_primes + limb) * sub_degree + t;
        if (got[idx] % p != want[idx] % p) {
          if (mismatches == 0 && first != nullptr) {
            *first = "entry " + std::to_string(e) + " limb " +
                     std::to_string(limb) + " t " + std::to_string(t) +
                     ": got " + std::to_string(got[idx]) + " want " +
                     std::to_string(want[idx]);
          }
          mismatches++;
        }
      }
    }
  }
  return mismatches;
}

}  // namespace

// Vec^d_k out and back must be the identity on the ciphertexts. This is where
// a transposed or mis-strided gather shows up, and it needs no arithmetic at
// all -- only the index maps and the Montgomery convention of the transforms
// on the way in and out.
TEST_P(Testbed32, VecRoundTripsThroughCiphertexts) {
  const int degree = param_->degree_;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  SubringCtMatrixHandler<word> handler(*param_, context_->ntt_handler_);

  std::vector<Ciphertext<word>> cts(kNumCts);
  for (int j = 0; j < kNumCts; j++) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);
    interface_->Encrypt(cts[j], pt);
  }

  SubringCoeffMatrix<word> b_mat, a_mat;
  handler.ToMatrices(b_mat, a_mat, cts, kSubDegree, false);
  ASSERT_EQ(b_mat.rows_, degree / kSubDegree);
  ASSERT_EQ(b_mat.cols_, kNumCts);

  std::vector<Ciphertext<word>> back;
  handler.ToCiphertexts(back, b_mat, a_mat, scale, cts[0].GetNumSlots());
  ASSERT_EQ(static_cast<int>(back.size()), kNumCts);

  const NPInfo np = cts[0].GetNP();
  const auto primes = param_->GetPrimeVector(np);
  const int num_total = np.GetNumTotal();

  long long mismatches = 0;
  for (int j = 0; j < kNumCts; j++) {
    HostVector<word> gb, ga, wb, wa;
    CopyDeviceToHost(gb, back[j].bx_);
    CopyDeviceToHost(ga, back[j].ax_);
    CopyDeviceToHost(wb, cts[j].bx_);
    CopyDeviceToHost(wa, cts[j].ax_);
    for (int limb = 0; limb < num_total; limb++) {
      const uint64_t p = primes[limb];
      for (int x = 0; x < degree; x++) {
        const int idx = limb * degree + x;
        if (gb[idx] % p != wb[idx] % p) mismatches++;
        if (ga[idx] % p != wa[idx] % p) mismatches++;
      }
    }
  }
  std::cout << "Vec round trip, d = " << degree / kSubDegree << ", k = "
            << kSubDegree << ": " << mismatches << " mismatches" << std::endl;
  ASSERT_EQ(mismatches, 0);
}

// The transform against the direct convolution. This is the test the
// length-k NTT exists to pass, and the only one that would catch a twiddle
// order that inverts cleanly while convolving wrongly.
TEST_P(Testbed32, TransformProductMatchesDirectConvolution) {
  const int degree = param_->degree_;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);
  const int vec_dim = degree / kSubDegree;

  SubringCtMatrixHandler<word> handler(*param_, context_->ntt_handler_);

  std::vector<Ciphertext<word>> cts(kNumCts);
  for (int j = 0; j < kNumCts; j++) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);
    interface_->Encrypt(cts[j], pt);
  }

  // The Algorithm 4 step 2 shape: column-wise on the left, row-wise on the
  // right, so the contraction runs over the ciphertext index of both.
  SubringCoeffMatrix<word> lhs_b, lhs_a, rhs_b, rhs_a;
  handler.ToMatrices(lhs_b, lhs_a, cts, kSubDegree, false);
  handler.ToMatrices(rhs_b, rhs_a, cts, kSubDegree, true);
  ASSERT_EQ(lhs_b.cols_, rhs_b.rows_);

  SubringCoeffMatrix<word> fast, slow;
  handler.MultiplyMatrices(fast, lhs_b, rhs_b);
  handler.MultiplyMatricesReference(slow, lhs_b, rhs_b);

  ASSERT_EQ(fast.rows_, vec_dim);
  ASSERT_EQ(fast.cols_, vec_dim);
  ASSERT_EQ(fast.data_.size(), slow.data_.size());

  const NPInfo np = lhs_b.np_;
  const auto primes = param_->GetPrimeVector(np);
  const int num_total = np.GetNumTotal();

  HostVector<word> got, want;
  CopyDeviceToHost(got, fast.data_);
  CopyDeviceToHost(want, slow.data_);

  std::string first;
  const long long mismatches =
      CountMismatches(got, want, primes, kSubDegree, num_total, &first);
  std::cout << "transform vs convolution, " << fast.rows_ << "x" << fast.cols_
            << " over R_" << kSubDegree << ": " << mismatches << " mismatches"
            << std::endl;
  ASSERT_EQ(mismatches, 0) << "first at " << first;
}

// The same on the a-parts, and with the operands crossed, so that a bug that
// happened to cancel on one particular pair of inputs does not slip through.
TEST_P(Testbed32, TransformProductMatchesOnCrossedOperands) {
  const int degree = param_->degree_;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  SubringCtMatrixHandler<word> handler(*param_, context_->ntt_handler_);

  std::vector<Ciphertext<word>> cts(kNumCts);
  for (int j = 0; j < kNumCts; j++) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -0.5, 0.5);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);
    interface_->Encrypt(cts[j], pt);
  }

  SubringCoeffMatrix<word> lhs_b, lhs_a, rhs_b, rhs_a;
  handler.ToMatrices(lhs_b, lhs_a, cts, kSubDegree, false);
  handler.ToMatrices(rhs_b, rhs_a, cts, kSubDegree, true);

  const NPInfo np = lhs_b.np_;
  const auto primes = param_->GetPrimeVector(np);
  const int num_total = np.GetNumTotal();

  // The four cross products are exactly C00, C01, C10, C11 of step 2.
  const std::vector<std::pair<const SubringCoeffMatrix<word> *,
                              const SubringCoeffMatrix<word> *>>
      pairs = {{&lhs_b, &rhs_b}, {&lhs_b, &rhs_a},
               {&lhs_a, &rhs_b}, {&lhs_a, &rhs_a}};

  long long total = 0;
  for (size_t c = 0; c < pairs.size(); c++) {
    SubringCoeffMatrix<word> fast, slow;
    handler.MultiplyMatrices(fast, *pairs[c].first, *pairs[c].second);
    handler.MultiplyMatricesReference(slow, *pairs[c].first, *pairs[c].second);

    HostVector<word> got, want;
    CopyDeviceToHost(got, fast.data_);
    CopyDeviceToHost(want, slow.data_);
    std::string first;
    const long long mismatches =
        CountMismatches(got, want, primes, kSubDegree, num_total, &first);
    std::cout << "block C" << c / 2 << c % 2 << ": " << mismatches
              << " mismatches" << std::endl;
    if (mismatches > 0) std::cout << "  first at " << first << std::endl;
    total += mismatches;
  }
  ASSERT_EQ(total, 0);
}

INSTANTIATE_TEST_SUITE_P(
    SubringCtMatrix, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
