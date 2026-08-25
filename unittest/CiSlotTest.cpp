// The conjugate-invariant slot encoding: does Encode/Decode carry N real
// slots, and is slot t the canonical embedding it claims to be?
//
// This is the half of the conjugate-invariant work that [SYLPH] is actually
// after. CiRingTest settled the multiplication; what makes it worth having is
// that R+ is *totally real*, so its canonical embedding lands in R^N and a
// message is N real numbers where the ordinary ring holds N/2 complex ones.
// Halving the ciphertext count halves the bootstrap with it.
//
// The decisive test is CiSlotEmbedding, and it deliberately does not reuse the
// encoder's own transform. It encodes a coefficient vector, decodes it as
// slots, and compares against the embedding summed directly on the host:
//
//     ordinary ring   slot t  =  sum_j m_j w^(k j),          w = e^(2 pi i/2N)
//     R+              slot t  =  a_0 + sum_j a_j (w^(kj) + w^(-kj)),  w over 4N
//
// with k = 5^t the Galois factor. That pins down three things at once -- the
// embedding, the slot *ordering*, and the fold -- against arithmetic that
// shares no code with the thing under test. Every test here runs unchanged on
// ringdegree12_30 with the flag off, where MaxNumSlots() is N/2 and the
// embedding is the complex one, so a half-applied change cannot pass quietly.

#undef ENABLE_EXTENSION

#include "Testbed.h"

using word = uint32_t;

namespace {

// Nonzeros in the coefficient vector CiSlotEmbedding starts from. The host
// reference is O(degree * nnz), so keeping it sparse makes an exact reference
// over every slot cheap; the embedding is linear, so sparsity costs no
// generality.
constexpr int kNumNonzero = 8;

// The canonical embedding, summed term by term.
//
// One formula covers both rings because the only differences are which
// cyclotomic index the root of unity comes from and which basis the
// coefficients are written in: {X^j} for the ordinary ring, and
// {1, c_1, ..., c_(N-1)} with c_j = Y^j + Y^-j for the real subring -- so
// index 0 stands for 1 rather than for c_0 = 2, and every other index
// contributes its conjugate pair and lands on the real axis.
Complex CanonicalEmbedding(const std::vector<double> &coeffs, int factor,
                           int cyclotomic_index, bool ci) {
  const int degree = static_cast<int>(coeffs.size());
  Complex acc = ci ? Complex(coeffs[0], 0.0) : Complex(0.0, 0.0);
  for (int j = ci ? 1 : 0; j < degree; j++) {
    if (coeffs[j] == 0.0) continue;
    const int64_t exponent = (static_cast<int64_t>(factor) * j) %
                             static_cast<int64_t>(cyclotomic_index);
    const double angle =
        2.0 * M_PI * static_cast<double>(exponent) / cyclotomic_index;
    if (ci) {
      acc += Complex(2.0 * coeffs[j] * std::cos(angle), 0.0);
    } else {
      acc += coeffs[j] * std::polar(1.0, angle);
    }
  }
  return acc;
}

void CompareSlots(const std::vector<Complex> &expected,
                  const std::vector<Complex> &obtained, double max_error,
                  const std::string &what) {
  ASSERT_EQ(expected.size(), obtained.size()) << "Different slot counts";

  double max_abs_diff = 0.0;
  double max_abs_imag = 0.0;
  double sum_sq_diff = 0.0;
  double sum_sq_expected = 0.0;
  int worst_index = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    const double diff = std::abs(expected[i] - obtained[i]);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
      worst_index = static_cast<int>(i);
    }
    max_abs_imag = std::max(max_abs_imag, std::abs(obtained[i].imag()));
    sum_sq_diff += diff * diff;
    sum_sq_expected += std::norm(expected[i]);
  }

  std::cout << std::scientific << std::setprecision(5);
  std::cout << "  " << what << ": " << expected.size()
            << " slots, max |diff| = " << max_abs_diff << " at slot "
            << worst_index << ", SNR = " << sum_sq_expected / sum_sq_diff
            << ", max |Im| = " << max_abs_imag << std::endl;
  std::cout << std::fixed;

  ASSERT_LT(max_abs_diff, max_error)
      << what << ": slot " << worst_index << " differs by " << max_abs_diff;
}

void SampleSparse(std::vector<double> &v, int degree) {
  v.assign(degree, 0.0);
  std::vector<int> idx(kNumNonzero);
  std::vector<double> val(kNumNonzero);
  Random::SampleWithoutReplacement(idx.data(), kNumNonzero, 0, degree - 1);
  Random::SampleUniformReal(val.data(), kNumNonzero, -1.0, 1.0);
  for (int i = 0; i < kNumNonzero; i++) v[idx[i]] = val[i];
}

std::vector<Complex> Pointwise(const std::vector<Complex> &a,
                               const std::vector<Complex> &b) {
  std::vector<Complex> res(a.size());
  for (size_t i = 0; i < a.size(); i++) res[i] = a[i] * b[i];
  return res;
}

}  // namespace

// The decisive one: slot t is the canonical embedding at the t-th Galois
// factor, and the reference shares no code with the encoder.
TEST_P(Testbed32, CiSlotEmbedding) {
  const int degree = 1 << log_degree_;
  const bool ci = param_->conjugate_invariant_;
  const int num_slots = param_->MaxNumSlots();
  const int m = param_->CyclotomicIndex();

  for (int level : LevelsToSweep()) {
    std::vector<double> coeffs;
    SampleSparse(coeffs, degree);

    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), coeffs);

    std::vector<Complex> res;
    context_->encoder_.Decode(res, pt);

    std::vector<Complex> expected(num_slots);
    for (int t = 0; t < num_slots; t++) {
      expected[t] =
          CanonicalEmbedding(coeffs, param_->GetGaloisFactor(t), m, ci);
    }

    CompareSlots(expected, res, max_error_,
                 "CiSlotEmbedding at level " + std::to_string(level));

    // On the real subring the embedding lands on the real axis, and the
    // imaginary part that comes back is the residue of the mirrored
    // coefficient pair against itself -- it measures how far the decoded
    // message drifted off R+ and has to be rounding-sized.
    if (ci) {
      double max_imag = 0.0;
      for (const auto &z : res) max_imag = std::max(max_imag, std::abs(z.imag()));
      ASSERT_LT(max_imag, max_error_)
          << "the decoded message is not on the real subring";
    }
  }
}

// Encode a full-width message and read it back. On R+ that is N real slots,
// which is the whole point; on the ordinary ring it is N/2 complex ones and
// the same body runs unchanged.
TEST_P(Testbed32, CiSlotRoundTrip) {
  const bool ci = param_->conjugate_invariant_;
  const int num_slots = param_->MaxNumSlots();
  ASSERT_EQ(num_slots, ci ? (1 << log_degree_) : (1 << (log_degree_ - 1)));

  for (int level : LevelsToSweep()) {
    std::vector<Complex> msg;
    // A complex message has nowhere to go on the real subring, and Encode
    // rejects one rather than halving it silently.
    GenerateRandomMessage(msg, num_slots, -1.0, 1.0, /*complex=*/!ci);

    Plaintext<word> pt;
    context_->encoder_.Encode(pt, level, DetermineScale(level), msg);
    ASSERT_EQ(pt.GetNumSlots(), num_slots);

    std::vector<Complex> res;
    context_->encoder_.Decode(res, pt);

    CompareSlots(msg, res, max_error_,
                 "CiSlotRoundTrip at level " + std::to_string(level));
  }
}

// Sparse packing. A message of S slots is an element of the rank-S subring,
// and the fold has the same shape there -- f[r] = a_(r*gap) - i a_((S-r)*gap)
// -- so the transform is the same one at a smaller size. Every divisor down
// to 2 is checked, because that is where the coefficient stride is widest.
TEST_P(Testbed32, CiSlotSparsePacking) {
  const bool ci = param_->conjugate_invariant_;
  const int max_num_slots = param_->MaxNumSlots();
  const int level = param_->max_level_;

  // Powers of four, so the stride sweep stays a handful of points at logN 16
  // as well as covering every one of them at logN 12.
  for (int num_slots = 2; num_slots <= max_num_slots; num_slots *= 4) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg, num_slots, -1.0, 1.0, /*complex=*/!ci);

    Plaintext<word> pt;
    context_->encoder_.Encode(pt, level, DetermineScale(level), msg);
    ASSERT_EQ(pt.GetNumSlots(), num_slots);

    std::vector<Complex> res;
    context_->encoder_.Decode(res, pt);

    CompareSlots(msg, res, max_error_,
                 "CiSlotSparsePacking at " + std::to_string(num_slots) +
                     " slots");
  }
}

// Slot-wise multiplication. This is what the embedding is for: it says the
// host's slot index is a coordinate in which the ring product is pointwise,
// which is a statement about the fold, the transform and the slot ordering all
// at once. Encrypted operand times plaintext operand, then rescaled, so it
// also runs the product through ModDown.
TEST_P(Testbed32, CiSlotProduct) {
  const bool ci = param_->conjugate_invariant_;
  const int num_slots = param_->MaxNumSlots();

  for (int level : LevelsToSweep(1)) {
    std::vector<Complex> a, b;
    GenerateRandomMessage(a, num_slots, -1.0, 1.0, /*complex=*/!ci);
    GenerateRandomMessage(b, num_slots, -1.0, 1.0, /*complex=*/!ci);

    const double scale = DetermineScale(level);
    Plaintext<word> pt_a, pt_b;
    context_->encoder_.Encode(pt_a, level, scale, a);
    context_->encoder_.Encode(pt_b, level, scale, b);

    Ciphertext<word> ct_a;
    interface_->Encrypt(ct_a, pt_a);

    Ciphertext<word> ct_prod, ct_res;
    context_->Mult(ct_prod, ct_a, pt_b);
    context_->Rescale(ct_res, ct_prod);

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct_res);

    std::vector<Complex> res;
    context_->encoder_.Decode(res, pt_out);

    CompareSlots(Pointwise(a, b), res, max_error_,
                 "CiSlotProduct at level " + std::to_string(level) + " -> " +
                     std::to_string(level - 1));
  }
}

// The same in the slot domain but with both operands encrypted, so the
// product is relinearized -- a full key switch under the slot encoding, which
// is the path the layer actually takes.
TEST_P(Testbed32, CiSlotCiphertextProduct) {
  const bool ci = param_->conjugate_invariant_;
  const int num_slots = param_->MaxNumSlots();

  for (int level : LevelsToSweep(1)) {
    std::vector<Complex> a, b;
    GenerateRandomMessage(a, num_slots, -1.0, 1.0, /*complex=*/!ci);
    GenerateRandomMessage(b, num_slots, -1.0, 1.0, /*complex=*/!ci);

    const double scale = DetermineScale(level);
    Plaintext<word> pt_a, pt_b;
    context_->encoder_.Encode(pt_a, level, scale, a);
    context_->encoder_.Encode(pt_b, level, scale, b);

    Ciphertext<word> ct_a, ct_b;
    interface_->Encrypt(ct_a, pt_a);
    interface_->Encrypt(ct_b, pt_b);

    Ciphertext<word> ct_res;
    context_->HMult(ct_res, ct_a, ct_b, interface_->GetMultiplicationKey(),
                    true);

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct_res);

    std::vector<Complex> res;
    context_->encoder_.Decode(res, pt_out);

    CompareSlots(Pointwise(a, b), res, max_error_,
                 "CiSlotCiphertextProduct at level " + std::to_string(level) +
                     " -> " + std::to_string(level - 1));
  }
}

// The same automorphisms, read in the slot domain, which is where they are
// supposed to be a cyclic shift. That they are is the payoff of the affine
// index map: the transform's own positions carry (Z/2N)^*, the slots carry
// (Z/4N)^* / {+-1}, and only in the slot coordinate does the action come out
// as "move everything along by r".
TEST_P(Testbed32, CiSlotRotation) {
  const bool ci = param_->conjugate_invariant_;
  const int num_slots = param_->MaxNumSlots();
  const int level = param_->max_level_;

  const std::vector<int> distances = {1, 3, 1234 % num_slots,
                                      num_slots / 2 + 1, num_slots - 1};

  for (int rot : distances) {
    interface_->PrepareRotationKey(rot, level);

    std::vector<Complex> msg;
    GenerateRandomMessage(msg, num_slots, -1.0, 1.0, /*complex=*/!ci);
    std::vector<Complex> expected(num_slots);
    for (int i = 0; i < num_slots; i++) {
      expected[i] = msg[(i + rot) % num_slots];
    }

    Plaintext<word> pt;
    context_->encoder_.Encode(pt, level, DetermineScale(level), msg);

    Ciphertext<word> ct, ct_res;
    interface_->Encrypt(ct, pt);
    context_->HRot(ct_res, ct, interface_->GetRotationKey(rot), rot);

    Plaintext<word> pt_out;
    interface_->Decrypt(pt_out, ct_res);

    std::vector<Complex> res;
    context_->encoder_.Decode(res, pt_out);

    CompareSlots(expected, res, max_error_,
                 "CiSlotRotation by " + std::to_string(rot));
  }
}

INSTANTIATE_TEST_SUITE_P(
    // ringdegree12_30 is the same primes, the same levels and the same shape
    // with the conjugate-invariant flag off -- the control.
    Cheddar, Testbed32,
    testing::Values("ci12_30.json", "ci16_35.json", "ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
