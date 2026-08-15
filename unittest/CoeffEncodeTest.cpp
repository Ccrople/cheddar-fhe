// Tests for the coefficient-domain encoding added for the reduction-based
// matrix-multiplication algorithms (Bae et al., CRYPTO 2024).
//
// The decisive test here is CoeffNegacyclicConvolution: under the SLOT encoding
// a ciphertext-plaintext multiplication is the pointwise product of the encoded
// vectors, whereas under the COEFFICIENT encoding it is their negacyclic
// convolution in Z[X]/(X^N + 1). Checking the convolution against a host
// reference is what proves the new encoding really is coefficient-domain, and
// not merely a round-trip that happens to be self-consistent.

#undef ENABLE_EXTENSION

#include "Testbed.h"

using word = uint32_t;

namespace {

// Number of nonzero coefficients used by the convolution test. Kept small so
// the host reference is O(nnz^2) rather than O(degree^2), and so the product
// magnitude stays well inside the level-0 message range.
constexpr int kNumNonzero = 8;

// Negacyclic convolution in Z[X]/(X^degree + 1), on the host: X^degree = -1.
std::vector<double> NegacyclicConvolution(const std::vector<double> &a,
                                          const std::vector<double> &b,
                                          int degree) {
  std::vector<double> res(degree, 0.0);
  for (int i = 0; i < degree; i++) {
    if (a[i] == 0.0) continue;
    for (int j = 0; j < degree; j++) {
      if (b[j] == 0.0) continue;
      int k = i + j;
      if (k < degree) {
        res[k] += a[i] * b[j];
      } else {
        res[k - degree] -= a[i] * b[j];
      }
    }
  }
  return res;
}

// A handful of representative levels: the base, the first rescalable level,
// the middle, and the top. Encoding is a host-side BigInt CRT over every
// coefficient and every prime, so sweeping all levels would dominate runtime
// without testing anything new.
std::vector<int> RepresentativeLevels(int max_level, int min_level) {
  std::vector<int> levels;
  for (int candidate : {min_level, min_level + 1, max_level / 2, max_level}) {
    if (candidate < min_level || candidate > max_level) continue;
    if (std::find(levels.begin(), levels.end(), candidate) == levels.end()) {
      levels.push_back(candidate);
    }
  }
  return levels;
}

void CompareCoeffs(const std::vector<double> &expected,
                   const std::vector<double> &obtained, double max_error,
                   bool print) {
  ASSERT_EQ(expected.size(), obtained.size()) << "Different coefficient counts";

  double max_abs_diff = 0.0;
  double sum_sq_diff = 0.0;
  double sum_sq_expected = 0.0;
  int worst_index = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    double diff = expected[i] - obtained[i];
    if (std::abs(diff) > max_abs_diff) {
      max_abs_diff = std::abs(diff);
      worst_index = static_cast<int>(i);
    }
    sum_sq_diff += diff * diff;
    sum_sq_expected += expected[i] * expected[i];
  }

  if (print) {
    std::cout << std::scientific << std::setprecision(5);
    std::cout << "  max |diff| = " << max_abs_diff << " at coefficient "
              << worst_index << " (expected " << expected[worst_index]
              << ", obtained " << obtained[worst_index] << ")" << std::endl;
    std::cout << "  SNR = " << sum_sq_expected / sum_sq_diff << std::endl;
    std::cout << std::fixed;
  }

  ASSERT_LT(max_abs_diff, max_error)
      << "Coefficient " << worst_index << " differs by " << max_abs_diff;
}

}  // namespace

// Encode a full-width random coefficient vector and read it back.
TEST_P(Testbed32, CoeffEncodeDecode) {
  const int degree = 1 << log_degree_;
  for (int level : RepresentativeLevels(param_->max_level_, 0)) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

    const double scale = DetermineScale(level);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);

    std::vector<double> res;
    context_->encoder_.DecodeCoeff(res, pt);

    std::cout << "CoeffEncodeDecode at level " << level << std::endl;
    CompareCoeffs(coeffs, res, max_error_, true);
  }
}

// The same, through encryption and decryption.
TEST_P(Testbed32, CoeffEncodeEncryptDecryptDecode) {
  const int degree = 1 << log_degree_;
  for (int level : RepresentativeLevels(param_->max_level_, 0)) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

    const double scale = DetermineScale(level);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);

    Ciphertext<word> ct;
    interface_->Encrypt(ct, pt);

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct);

    std::vector<double> res;
    context_->encoder_.DecodeCoeff(res, pt_out);

    std::cout << "CoeffEncodeEncryptDecryptDecode at level " << level
              << std::endl;
    CompareCoeffs(coeffs, res, max_error_, true);
  }
}

// ct(coeff a) * pt(coeff b) must decode to the negacyclic convolution of a and
// b. This is the property the reduction-based PCMM relies on.
TEST_P(Testbed32, CoeffNegacyclicConvolution) {
  const int degree = 1 << log_degree_;
  // level >= 1 so that the product can be rescaled back down.
  for (int level : RepresentativeLevels(param_->max_level_, 1)) {
    std::vector<double> a(degree, 0.0);
    std::vector<double> b(degree, 0.0);
    {
      std::vector<int> idx_a(kNumNonzero), idx_b(kNumNonzero);
      std::vector<double> val_a(kNumNonzero), val_b(kNumNonzero);
      Random::SampleWithoutReplacement(idx_a.data(), kNumNonzero, 0, degree - 1);
      Random::SampleWithoutReplacement(idx_b.data(), kNumNonzero, 0, degree - 1);
      Random::SampleUniformReal(val_a.data(), kNumNonzero, -1.0, 1.0);
      Random::SampleUniformReal(val_b.data(), kNumNonzero, -1.0, 1.0);
      for (int i = 0; i < kNumNonzero; i++) {
        a[idx_a[i]] = val_a[i];
        b[idx_b[i]] = val_b[i];
      }
    }

    const std::vector<double> expected = NegacyclicConvolution(a, b, degree);

    const double scale = DetermineScale(level);
    Plaintext<word> pt_a, pt_b;
    context_->encoder_.EncodeCoeff(pt_a, level, scale, a);
    context_->encoder_.EncodeCoeff(pt_b, level, scale, b);

    Ciphertext<word> ct_a;
    interface_->Encrypt(ct_a, pt_a);

    Ciphertext<word> ct_prod, ct_res;
    context_->Mult(ct_prod, ct_a, pt_b);  // scale^2, not rescaled
    context_->Rescale(ct_res, ct_prod);   // back to ~scale, level - 1

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct_res);

    std::vector<double> res;
    context_->encoder_.DecodeCoeff(res, pt_out);

    std::cout << "CoeffNegacyclicConvolution at level " << level << " -> "
              << level - 1 << std::endl;
    CompareCoeffs(expected, res, max_error_, true);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    // bootparam_30 has no terminal primes; bootparam_40 does, so the two
    // together exercise both shapes of Parameter::GetPrimeVector ordering that
    // the host-side CRT in DecodeCoeff has to follow.
    testing::Values("bootparam_30.json", "bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
