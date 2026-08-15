// Tests for the Bae et al. plaintext-ciphertext matrix product, in the RLWE
// format at a single ring degree.
//
// The product is res[i] = sum_j U[i][j] * cts[j] over coefficient-encoded
// ciphertexts. Correctness is checked against a host reference computed on the
// plaintext coefficient vectors, never against another GPU run.
//
// Two structural properties from the paper are asserted as well, because they
// are what makes this product worth using: it spends exactly one level, and it
// needs no rotation, automorphism or relinearization key.

#undef ENABLE_EXTENSION

#include "Testbed.h"
#include "core/Pcmm.h"

using word = uint32_t;

namespace {

// Small so the host reference and the test itself stay cheap; the product is
// O(rows * cols * degree * num_primes) on the device either way.
constexpr int kRows = 6;
constexpr int kCols = 8;

// Only the leading `kActive` coefficients of each message are exercised, so the
// host reference stays small. The device still computes all `degree` of them,
// and the trailing coefficients are checked to be zero.
constexpr int kActive = 64;

std::vector<std::vector<double>> HostReference(
    const std::vector<double> &u, const std::vector<std::vector<double>> &m,
    int rows, int cols, int degree) {
  std::vector<std::vector<double>> res(rows, std::vector<double>(degree, 0.0));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      const double weight = u[i * cols + j];
      if (weight == 0.0) continue;
      for (int c = 0; c < degree; c++) {
        res[i][c] += weight * m[j][c];
      }
    }
  }
  return res;
}

void CompareCoeffs(const std::vector<double> &expected,
                   const std::vector<double> &obtained, double max_error,
                   const std::string &label, bool print) {
  ASSERT_EQ(expected.size(), obtained.size()) << "Different coefficient counts";
  double max_abs_diff = 0.0;
  int worst_index = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    const double diff = std::abs(expected[i] - obtained[i]);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
      worst_index = static_cast<int>(i);
    }
  }
  if (print) {
    std::cout << std::scientific << std::setprecision(5) << "  " << label
              << ": max |diff| = " << max_abs_diff << " at coefficient "
              << worst_index << " (expected " << expected[worst_index]
              << ", obtained " << obtained[worst_index] << ")" << std::endl;
    std::cout << std::fixed;
  }
  ASSERT_LT(max_abs_diff, max_error)
      << label << ": coefficient " << worst_index << " differs by "
      << max_abs_diff;
}

}  // namespace

TEST_P(Testbed32, PcmmMatchesHostReference) {
  const int degree = 1 << log_degree_;
  PcmmHandler<word> pcmm(*param_);

  // level >= 1 so the product can be rescaled back down.
  for (int level : {1, param_->max_level_ / 2, param_->max_level_}) {
    if (level < 1) continue;

    // Messages: only the first kActive coefficients are nonzero.
    std::vector<std::vector<double>> msgs(kCols,
                                          std::vector<double>(degree, 0.0));
    for (int j = 0; j < kCols; j++) {
      Random::SampleUniformReal(msgs[j].data(), kActive, -1.0, 1.0);
    }
    std::vector<double> u(kRows * kCols);
    Random::SampleUniformReal(u.data(), kRows * kCols, -1.0, 1.0);

    const std::vector<std::vector<double>> expected =
        HostReference(u, msgs, kRows, kCols, degree);

    const double scale = DetermineScale(level);

    std::vector<Ciphertext<word>> cts(kCols);
    for (int j = 0; j < kCols; j++) {
      Plaintext<word> pt;
      context_->encoder_.EncodeCoeff(pt, level, scale, msgs[j]);
      interface_->Encrypt(cts[j], pt);
    }

    PlainMatrix<word> u_enc;
    pcmm.EncodeMatrix(u_enc, level, scale, u, kRows, kCols);

    std::vector<Ciphertext<word>> prod;
    pcmm.Multiply(prod, u_enc, cts);
    ASSERT_EQ(static_cast<int>(prod.size()), kRows);

    std::cout << "PcmmMatchesHostReference at level " << level << " -> "
              << level - 1 << std::endl;
    for (int i = 0; i < kRows; i++) {
      Ciphertext<word> rescaled;
      context_->Rescale(rescaled, prod[i]);

      Plaintext<word> pt_out;
      interface_->Decrypt(pt_out, rescaled);
      std::vector<double> res;
      context_->encoder_.DecodeCoeff(res, pt_out);

      CompareCoeffs(expected[i], res, max_error_, "row " + std::to_string(i),
                    i == 0);
    }
  }
}

// The product must consume exactly one level: it leaves the ciphertext at the
// input level with a squared scale, and a single rescale returns it to the
// input scale one level down.
TEST_P(Testbed32, PcmmSpendsExactlyOneLevel) {
  const int degree = 1 << log_degree_;
  PcmmHandler<word> pcmm(*param_);

  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  std::vector<std::vector<double>> msgs(kCols,
                                        std::vector<double>(degree, 0.0));
  for (int j = 0; j < kCols; j++) {
    Random::SampleUniformReal(msgs[j].data(), kActive, -1.0, 1.0);
  }
  std::vector<double> u(kRows * kCols);
  Random::SampleUniformReal(u.data(), kRows * kCols, -1.0, 1.0);

  std::vector<Ciphertext<word>> cts(kCols);
  for (int j = 0; j < kCols; j++) {
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, msgs[j]);
    interface_->Encrypt(cts[j], pt);
  }

  PlainMatrix<word> u_enc;
  pcmm.EncodeMatrix(u_enc, level, scale, u, kRows, kCols);

  std::vector<Ciphertext<word>> prod;
  pcmm.Multiply(prod, u_enc, cts);

  // Before rescaling: same level, scale multiplied.
  ASSERT_EQ(param_->NPToLevel(prod[0].GetNP()), level);
  ASSERT_NEAR(prod[0].GetScale(), scale * scale, scale * scale * 1e-9);

  Ciphertext<word> rescaled;
  context_->Rescale(rescaled, prod[0]);
  ASSERT_EQ(param_->NPToLevel(rescaled.GetNP()), level - 1);
}

// The product touches no evaluation key. Running it with an EvkMap that holds
// nothing but the keys UserInterface makes at construction must succeed, and
// the key set must be unchanged afterwards.
TEST_P(Testbed32, PcmmNeedsNoRotationOrRelinearizationKey) {
  const int degree = 1 << log_degree_;
  PcmmHandler<word> pcmm(*param_);

  const size_t keys_before = interface_->GetEvkMap().size();

  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  std::vector<std::vector<double>> msgs(kCols,
                                        std::vector<double>(degree, 0.0));
  for (int j = 0; j < kCols; j++) {
    Random::SampleUniformReal(msgs[j].data(), kActive, -1.0, 1.0);
  }
  std::vector<double> u(kRows * kCols);
  Random::SampleUniformReal(u.data(), kRows * kCols, -1.0, 1.0);

  std::vector<Ciphertext<word>> cts(kCols);
  for (int j = 0; j < kCols; j++) {
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, scale, msgs[j]);
    interface_->Encrypt(cts[j], pt);
  }

  PlainMatrix<word> u_enc;
  pcmm.EncodeMatrix(u_enc, level, scale, u, kRows, kCols);

  std::vector<Ciphertext<word>> prod;
  pcmm.Multiply(prod, u_enc, cts);

  ASSERT_EQ(interface_->GetEvkMap().size(), keys_before)
      << "the product allocated an evaluation key";

  // No rotation key was ever requested, so asking for one must still fail.
  // (EvkMap::GetEvk asserts on a missing index, which is why this is checked
  // by absence from the map rather than by calling it.)
  ASSERT_EQ(interface_->GetEvkMap().find(1), interface_->GetEvkMap().end());
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
