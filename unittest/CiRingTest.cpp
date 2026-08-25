// The conjugate-invariant ring itself: does Cheddar's arithmetic, run with
// `conjugate_invariant` set, multiply in Z[Y + Y^-1] rather than in
// Z[X]/(X^N + 1)?
//
// [SYLPH] takes its N real slots from the conjugate-invariant variant of CKKS
// (Kim and Song, ISISC 2018), where Cheddar's encoder gives N/2 complex ones.
// The two rings have the same rank and so the same security, but a different
// multiplication, and this file is what pins that difference down. The
// decisive test is CiRingConvolution: a coefficient-encoded product must come
// out as the ring's own convolution
//
//     c_j * c_k = c_{j+k} + c_{|j-k|},   c_j = Y^j + Y^-j,  Y^(2N) = -1
//
// which is a Toeplitz-plus-Hankel form, and *not* the negacyclic convolution
// the ordinary ring produces. The two disagree in every coefficient, so a
// build that had merely changed the twiddle root without folding -- or folded
// without changing the root -- fails here rather than passing quietly.
//
// The host reference is written in the c_j basis directly and was checked
// against multiplication of the negacyclic lifts, which is a different
// derivation of the same product.

#undef ENABLE_EXTENSION

#include "Testbed.h"

using word = uint32_t;

namespace {

// Nonzeros per operand. Small enough that the reference is O(nnz^2) and the
// product's coefficients stay well inside the level's message range.
constexpr int kNumNonzero = 8;

// Scale of the plaintext operand. The product is never rescaled -- Rescale
// goes through ModDown, which does not carry the fold yet -- so the two scales
// have to multiply to something the modulus at this level holds comfortably.
//
// It also has to be large enough that EncodeCoeff rounding does not dominate.
// At 2^10 it did: round(b_i * 1024) carries up to half a unit, so every
// coefficient of b came in with ~5e-4 of absolute error and the product
// inherited it. Three runs then landed at 2.5e-04, 3.8e-04 and 4.7e-04 against
// a 1e-3 tolerance -- passing, but on a distribution whose tail crosses it,
// which is how this test failed on a fresh box having passed on another. At
// 2^20 the rounding contributes ~5e-7 and the margin is three orders of
// magnitude. The modulus is not the binding constraint: the product sits at
// 2^30 * 2^20 = 2^50 against a level-1 modulus of 2^69.8.
constexpr double kOperandScale = 1048576.0;  // 2^20

// Multiplication in the conjugate-invariant ring R+ = Z[Y + Y^-1] of rank
// `degree`, in the basis {1, c_1, ..., c_{degree-1}}. Index 0 is the
// coefficient of 1, not of c_0 = 2, which is why a term landing at index 0
// contributes twice.
std::vector<double> ConjugateInvariantConvolution(const std::vector<double> &a,
                                                  const std::vector<double> &b,
                                                  int degree) {
  std::vector<double> res(degree, 0.0);
  auto add = [&res, degree](int m, double v) {
    if (m == 0) {
      res[0] += 2.0 * v;  // c_0 = 2
      return;
    }
    if (m == degree) return;  // c_N = 0, since Y^N is a square root of -1
    if (m > degree) {         // c_m = -c_{2N-m}
      m = 2 * degree - m;
      v = -v;
    }
    res[m] += v;
  };

  res[0] += a[0] * b[0];
  for (int k = 1; k < degree; k++) res[k] += a[0] * b[k];
  for (int j = 1; j < degree; j++) res[j] += a[j] * b[0];
  for (int j = 1; j < degree; j++) {
    if (a[j] == 0.0) continue;
    for (int k = 1; k < degree; k++) {
      if (b[k] == 0.0) continue;
      const double v = a[j] * b[k];
      add(j + k, v);
      add(j > k ? j - k : k - j, v);
    }
  }
  return res;
}

// The ordinary ring's product, for the contrast assertion below.
std::vector<double> NegacyclicConvolution(const std::vector<double> &a,
                                          const std::vector<double> &b,
                                          int degree) {
  std::vector<double> res(degree, 0.0);
  for (int i = 0; i < degree; i++) {
    if (a[i] == 0.0) continue;
    for (int j = 0; j < degree; j++) {
      if (b[j] == 0.0) continue;
      const int k = i + j;
      if (k < degree) {
        res[k] += a[i] * b[j];
      } else {
        res[k - degree] -= a[i] * b[j];
      }
    }
  }
  return res;
}

double MaxAbsDiff(const std::vector<double> &x, const std::vector<double> &y) {
  double worst = 0.0;
  for (size_t i = 0; i < x.size(); i++) {
    worst = std::max(worst, std::abs(x[i] - y[i]));
  }
  return worst;
}

void CompareCoeffs(const std::vector<double> &expected,
                   const std::vector<double> &obtained, double max_error,
                   const std::string &what) {
  ASSERT_EQ(expected.size(), obtained.size()) << "Different coefficient counts";

  double max_abs_diff = 0.0;
  double sum_sq_diff = 0.0;
  double sum_sq_expected = 0.0;
  int worst_index = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    const double diff = expected[i] - obtained[i];
    if (std::abs(diff) > max_abs_diff) {
      max_abs_diff = std::abs(diff);
      worst_index = static_cast<int>(i);
    }
    sum_sq_diff += diff * diff;
    sum_sq_expected += expected[i] * expected[i];
  }

  std::cout << std::scientific << std::setprecision(5);
  std::cout << "  " << what << ": max |diff| = " << max_abs_diff
            << " at coefficient " << worst_index << " (expected "
            << expected[worst_index] << ", obtained " << obtained[worst_index]
            << "), SNR = " << sum_sq_expected / sum_sq_diff << std::endl;
  std::cout << std::fixed;

  ASSERT_LT(max_abs_diff, max_error)
      << what << ": coefficient " << worst_index << " differs by "
      << max_abs_diff;
}

void SampleSparse(std::vector<double> &v, int degree) {
  v.assign(degree, 0.0);
  std::vector<int> idx(kNumNonzero);
  std::vector<double> val(kNumNonzero);
  Random::SampleWithoutReplacement(idx.data(), kNumNonzero, 0, degree - 1);
  Random::SampleUniformReal(val.data(), kNumNonzero, -1.0, 1.0);
  for (int i = 0; i < kNumNonzero; i++) v[idx[i]] = val[i];
}

}  // namespace

// The parameter set really is the conjugate-invariant one, and the derived
// quantities that hang off it moved with it. If this fails nothing below it
// means anything.
TEST_P(Testbed32, CiParameterShape) {
  const int degree = 1 << log_degree_;
  const bool ci = param_->conjugate_invariant_;
  EXPECT_EQ(param_->MaxNumSlots(), ci ? degree : degree / 2)
      << "the real embedding gives N slots, the complex one N/2";
  EXPECT_EQ(param_->CyclotomicIndex(), ci ? 4 * degree : 2 * degree);
  if (!ci) return;

  // The Galois group acting on the slots is (Z/4N)^* / {+-1}, of order N and
  // generated by 5. Both ends of the cycle are asserted: 5^N = 1, and no
  // smaller power of 5 is, or the rotations would not reach every slot.
  EXPECT_EQ(param_->GetGaloisFactor(0), 1);
  EXPECT_EQ(param_->GetGaloisFactor(degree), 1);
  for (int i = 1; i < degree; i++) {
    ASSERT_NE(param_->GetGaloisFactor(i), 1)
        << "5 has order " << i << " mod 4N, so the slot orbit is short";
  }
}

// Encode a full-width coefficient vector and read it back. This alone
// exercises the fold and the unfold against each other, but not against
// anything independent -- the next test does that.
TEST_P(Testbed32, CiCoeffRoundTrip) {
  const int degree = 1 << log_degree_;
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), coeffs);

    std::vector<double> res;
    context_->encoder_.DecodeCoeff(res, pt);

    CompareCoeffs(coeffs, res, max_error_,
                  "CiCoeffRoundTrip at level " + std::to_string(level));
  }
}

// The same through encryption, which puts the secret -- a ternary element of
// R+ -- and the error sampling on the same fold.
TEST_P(Testbed32, CiEncryptDecrypt) {
  const int degree = 1 << log_degree_;
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), coeffs);

    Ciphertext<word> ct;
    interface_->Encrypt(ct, pt);

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct);

    std::vector<double> res;
    context_->encoder_.DecodeCoeff(res, pt_out);

    CompareCoeffs(coeffs, res, max_error_,
                  "CiEncryptDecrypt at level " + std::to_string(level));
  }
}

// The decisive one. ct(coeff a) * pt(coeff b) must decode to the product in
// R+, and must NOT be the negacyclic convolution -- the second half of that
// sentence is asserted, not assumed, because the two are the same shape of
// object and only differ in the fold.
TEST_P(Testbed32, CiRingConvolution) {
  const int degree = 1 << log_degree_;
  // From level 1 up. The product is carried at the two scales multiplied and
  // never rescaled, and level 0's modulus is not wide enough to hold that.
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<double> a, b;
    SampleSparse(a, degree);
    SampleSparse(b, degree);

    const bool ci = param_->conjugate_invariant_;
    // The control run on an ordinary parameter set expects the other product,
    // which is what separates "the ring changed" from "the test is wrong".
    const std::vector<double> expected =
        ci ? ConjugateInvariantConvolution(a, b, degree)
           : NegacyclicConvolution(a, b, degree);
    const std::vector<double> other =
        ci ? NegacyclicConvolution(a, b, degree)
           : ConjugateInvariantConvolution(a, b, degree);

    Plaintext<word> pt_a, pt_b;
    context_->encoder_.EncodeCoeff(pt_a, level, DetermineScale(level), a);
    context_->encoder_.EncodeCoeff(pt_b, level, kOperandScale, b);

    Ciphertext<word> ct_a, ct_prod;
    interface_->Encrypt(ct_a, pt_a);
    // No rescale: the result is read back at the product of the two scales,
    // which Mult records for us.
    context_->Mult(ct_prod, ct_a, pt_b);

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct_prod);

    std::vector<double> res;
    context_->encoder_.DecodeCoeff(res, pt_out);

    CompareCoeffs(expected, res, max_error_,
                  "CiRingConvolution at level " + std::to_string(level));

    // The other ring's product is order 1 away, four orders of magnitude above
    // max_error_, so landing on it is distinguishable from landing here.
    const double separation = MaxAbsDiff(expected, other);
    ASSERT_GT(separation, 1e-2)
        << "the two rings' products are too close for this sample to "
           "distinguish them";
    std::cout << "  the other ring's product is " << separation
              << " away, and the result is " << MaxAbsDiff(res, other)
              << " from it" << std::endl;
    ASSERT_GT(MaxAbsDiff(res, other), separation / 2.0)
        << "the result looks like the other ring's product";
  }
}

// The same product, but rescaled -- which is the only thing here that goes
// through ModDown, and so the only thing that exercises the unfold on the
// centred representative MultConstNormalize leaves behind. Both operands are
// encoded at the level scale, so the result lands one level down at the
// canonical scale, exactly as CoeffNegacyclicConvolution does for the
// ordinary ring.
TEST_P(Testbed32, CiRingConvolutionRescaled) {
  const int degree = 1 << log_degree_;
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<double> a, b;
    SampleSparse(a, degree);
    SampleSparse(b, degree);

    const bool ci = param_->conjugate_invariant_;
    const std::vector<double> expected =
        ci ? ConjugateInvariantConvolution(a, b, degree)
           : NegacyclicConvolution(a, b, degree);
    const std::vector<double> other =
        ci ? NegacyclicConvolution(a, b, degree)
           : ConjugateInvariantConvolution(a, b, degree);

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

    CompareCoeffs(expected, res, max_error_,
                  "CiRingConvolutionRescaled at level " +
                      std::to_string(level) + " -> " +
                      std::to_string(level - 1));

    const double separation = MaxAbsDiff(expected, other);
    ASSERT_GT(separation, 1e-2)
        << "the two rings products are too close for this sample to "
           "distinguish them";
    ASSERT_GT(MaxAbsDiff(res, other), separation / 2.0)
        << "the result looks like the other ring product";
  }
}

// Both operands encrypted, so the product needs relinearizing -- and that is
// a key switch, which is ModUp followed by ModDown. It is the one thing here
// that drives the whole mod-switch machinery, and it needs no Galois key, so
// it can be checked before the automorphisms move to the real subring.
TEST_P(Testbed32, CiRingCiphertextProduct) {
  const int degree = 1 << log_degree_;
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<double> a, b;
    SampleSparse(a, degree);
    SampleSparse(b, degree);

    const bool ci = param_->conjugate_invariant_;
    const std::vector<double> expected =
        ci ? ConjugateInvariantConvolution(a, b, degree)
           : NegacyclicConvolution(a, b, degree);
    const std::vector<double> other =
        ci ? NegacyclicConvolution(a, b, degree)
           : ConjugateInvariantConvolution(a, b, degree);

    const double scale = DetermineScale(level);
    Plaintext<word> pt_a, pt_b;
    context_->encoder_.EncodeCoeff(pt_a, level, scale, a);
    context_->encoder_.EncodeCoeff(pt_b, level, scale, b);

    Ciphertext<word> ct_a, ct_b;
    interface_->Encrypt(ct_a, pt_a);
    interface_->Encrypt(ct_b, pt_b);

    Ciphertext<word> ct_res;
    context_->HMult(ct_res, ct_a, ct_b, interface_->GetMultiplicationKey(),
                    true);

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct_res);

    std::vector<double> res;
    context_->encoder_.DecodeCoeff(res, pt_out);

    CompareCoeffs(expected, res, max_error_,
                  "CiRingCiphertextProduct at level " + std::to_string(level) +
                      " -> " + std::to_string(level - 1));

    const double separation = MaxAbsDiff(expected, other);
    ASSERT_GT(separation, 1e-2)
        << "the two rings products are too close for this sample to "
           "distinguish them";
    ASSERT_GT(MaxAbsDiff(res, other), separation / 2.0)
        << "the result looks like the other ring product";
  }
}

INSTANTIATE_TEST_SUITE_P(
    // ringdegree12_30 is the same primes, the same levels and the same shape
    // with the conjugate-invariant flag off -- the control.
    Cheddar, Testbed32,
    testing::Values("ci12_30.json", "ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
