// Slots-in-Coefficients (SinC), the encoding [KANG] (ePrint 2025/1957) batch
// CC-MM operates in, as stated in [SYLPH] eq. (1).
//
// FIRST STEP OF THE BATCH CC-MM, and deliberately the one that costs nothing.
// The degree-4096 ring has exactly one multiplicative level, and [KANG]'s
// Algorithm 4 spends all of it on its single tensor product. Every other stage
// -- CMT, the automorphisms, the relinearization -- has to be level-free, and
// so does the encoding. This file establishes the encoding and spends zero
// levels doing it, which is why it comes before anything else.
//
// WHAT IS CHECKED, AND AGAINST WHAT. The risk in an encoding is that it is
// self-consistent and wrong: encode and decode can agree with each other while
// both disagree with the paper. So the two endpoints are pinned against code
// that already existed and was validated independently --
//
//   sub_degree == 2       must equal EncodeCoeff  (bit-exact, same rounding)
//   sub_degree == degree  must equal Encode       (the plaintext is
//                                                  interchangeable with a
//                                                  slot-encoded one)
//
// -- and those are not analogies, they are what the definition collapses to.
// A round trip alone would pass with a transposed or mis-strided layout; these
// two would not.
//
// SEPARATE BINARY. Testbed builds one Context per process from one parameter
// file, and this needs ringdegree12_30.json at degree 4096. Appending it to a
// suite that instantiates bootparam_* first would silently run at 65536; see
// SmallRingNttTest.cpp for the trap.

#undef ENABLE_EXTENSION

#include <cmath>

#include "Testbed.h"

using word = uint32_t;

namespace {

// Largest absolute deviation over a decoded message.
double MaxAbsDiff(const std::vector<Complex> &got,
                  const std::vector<Complex> &want) {
  double worst = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    worst = std::max(worst, std::abs(got[i] - want[i]));
  }
  return worst;
}

// Negacyclic product in R[Y]/(Y^n + 1), on real coefficients.
std::vector<double> NegacyclicMul(const std::vector<double> &a,
                                  const std::vector<double> &b) {
  const int n = static_cast<int>(a.size());
  std::vector<double> res(n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      const int k = i + j;
      if (k < n) {
        res[k] += a[i] * b[j];
      } else {
        res[k - n] -= a[i] * b[j];
      }
    }
  }
  return res;
}

}  // namespace

// The geometry, before anything numerical: every coefficient must be written
// exactly once by (i, t) -> i + t*(N/k), for every admissible k. This is pure
// arithmetic on the index map and it is where an off-by-one would live.
TEST_P(Testbed32, SinCIndexMapIsABijection) {
  const int degree = param_->degree_;
  for (int sub_degree = 2; sub_degree <= degree; sub_degree *= 2) {
    const int num_blocks = degree / sub_degree;
    const int slots_per_block = sub_degree / 2;

    std::vector<int> hits(degree, 0);
    for (int i = 0; i < num_blocks; i++) {
      for (int t = 0; t < slots_per_block; t++) {
        hits[i + t * num_blocks]++;
        hits[i + (t + slots_per_block) * num_blocks]++;
      }
    }
    for (int c = 0; c < degree; c++) {
      ASSERT_EQ(hits[c], 1) << "sub_degree " << sub_degree << ", coefficient "
                            << c << " written " << hits[c] << " times";
    }
  }
}

// Round trip at every sub-degree the ring admits, at every level.
TEST_P(Testbed32, SinCRoundTrips) {
  const int degree = param_->degree_;
  for (int level = 0; level <= param_->max_level_; level++) {
    const double scale = DetermineScale(level);
    for (int sub_degree : {2, 4, 16, 128, degree}) {
      std::vector<Complex> msg;
      GenerateRandomMessage(msg);
      ASSERT_EQ(static_cast<int>(msg.size()), degree / 2);

      Plaintext<word> ptxt;
      context_->encoder_.EncodeSinC(ptxt, level, scale, msg, sub_degree);
      std::vector<Complex> got;
      context_->encoder_.DecodeSinC(got, ptxt, sub_degree);

      ASSERT_EQ(got.size(), msg.size());
      const double err = MaxAbsDiff(got, msg);
      std::cout << "level " << level << " SinC k=" << sub_degree
                << " round trip: max " << err << std::endl;
      ASSERT_LT(err, 1e-5) << "SinC is lossy at sub_degree " << sub_degree;
    }
  }
}

// sub_degree = 2 is the coefficient encoding. iDFT_2 is the identity on one
// complex number, so the definition collapses to Re at position i and Im at
// position i + N/2 -- which is exactly [SYLPH]'s Ecd_coeff. Both routes reach
// the limbs through RealVectorToPlaintext, so this is bit-exact, not merely
// close.
TEST_P(Testbed32, SinCAtTwoIsCoefficientEncoding) {
  const int degree = param_->degree_;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg);

  std::vector<double> interleaved(degree);
  for (int i = 0; i < degree / 2; i++) {
    interleaved[i] = msg[i].real();
    interleaved[i + degree / 2] = msg[i].imag();
  }

  Plaintext<word> from_sinc, from_coeff;
  context_->encoder_.EncodeSinC(from_sinc, level, scale, msg, 2);
  context_->encoder_.EncodeCoeff(from_coeff, level, scale, interleaved);

  HostVector<word> a, b;
  CopyDeviceToHost(a, from_sinc.mx_);
  CopyDeviceToHost(b, from_coeff.mx_);
  ASSERT_EQ(a.size(), b.size());

  long long mismatches = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) mismatches++;
  }
  std::cout << "SinC k=2 vs EncodeCoeff: " << a.size() << " limb words, "
            << mismatches << " mismatches" << std::endl;
  ASSERT_EQ(mismatches, 0) << "SinC at sub_degree 2 is not the coefficient "
                              "encoding";
}

// sub_degree = degree is the slot encoding: one block, positions t, so the
// plaintext must be interchangeable with a slot-encoded one. Checked by
// decoding it with the *ordinary* Decode rather than DecodeSinC -- if the
// claim holds, Decode cannot tell the difference. Compared after decoding
// because Encode truncates through BigInt(double) where this rounds, which is
// half a unit in the last place of the scale.
TEST_P(Testbed32, SinCAtFullDegreeIsSlotEncoding) {
  const int degree = param_->degree_;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg);

  Plaintext<word> ptxt;
  context_->encoder_.EncodeSinC(ptxt, level, scale, msg, degree);

  std::vector<Complex> got;
  context_->encoder_.Decode(got, ptxt);

  const double err = MaxAbsDiff(got, msg);
  std::cout << "SinC k=degree decoded by ordinary Decode: max " << err
            << std::endl;
  ASSERT_LT(err, 1e-5) << "SinC at sub_degree = degree is not the slot "
                          "encoding";
}

// The property batch CC-MM is actually built on. Within one block the SinC
// coefficients form an element of R_k, and because iDFT_k is a ring
// homomorphism, multiplication *there* is slotwise on that block's message.
// [KANG] section 2.2 is exactly this observation, and it is what turns a batch
// of matrix products into one matrix product over R_k.
//
// NOT the same as multiplying the two plaintexts in R_N. A full-ring product
// mixes the X^i components, since X^i * X^j = X^(i+j) lands in block i+j. That
// is why [KANG] realises the subring products as plaintext matrix products on
// the Vec^d_k components instead of just multiplying the ciphertexts. Pinning
// the property that IS true keeps the one that is not from being assumed.
//
// Host-side throughout: the claim is about the encoding, not about any kernel.
TEST_P(Testbed32, SinCSubringProductIsSlotwiseWithinABlock) {
  const int degree = param_->degree_;
  const int sub_degree = 128;  // [SYLPH] section 3.3 uses SinC_{2^7, 2^12}
  const int num_blocks = degree / sub_degree;
  const int slots_per_block = sub_degree / 2;
  const int level = param_->max_level_;
  const double scale = DetermineScale(level);

  // Half-range inputs: the product of two blocks is compared directly, and
  // keeping |z| <= 0.5 keeps the encoded product inside the modulus.
  std::vector<Complex> z, w;
  GenerateRandomMessage(z, -1, -0.5, 0.5);
  GenerateRandomMessage(w, -1, -0.5, 0.5);

  Plaintext<word> pz, pw;
  context_->encoder_.EncodeSinC(pz, level, scale, z, sub_degree);
  context_->encoder_.EncodeSinC(pw, level, scale, w, sub_degree);

  // Back to the real coefficients the two subring elements are made of.
  std::vector<double> cz, cw;
  context_->encoder_.DecodeCoeff(cz, pz);
  context_->encoder_.DecodeCoeff(cw, pw);

  // Multiply the two subring elements of each block, on the host, in R_k.
  std::vector<double> product_coeffs(degree, 0.0);
  for (int i = 0; i < num_blocks; i++) {
    std::vector<double> mi(sub_degree), ni(sub_degree);
    for (int t = 0; t < sub_degree; t++) {
      mi[t] = cz[i + t * num_blocks];
      ni[t] = cw[i + t * num_blocks];
    }
    const std::vector<double> prod = NegacyclicMul(mi, ni);
    for (int t = 0; t < sub_degree; t++) {
      product_coeffs[i + t * num_blocks] = prod[t];
    }
  }

  // Read those coefficients back as a SinC message. If iDFT_k is the ring
  // homomorphism the algorithm assumes, this is the slotwise product.
  Plaintext<word> pprod;
  context_->encoder_.EncodeCoeff(pprod, level, scale, product_coeffs);
  std::vector<Complex> got;
  context_->encoder_.DecodeSinC(got, pprod, sub_degree);

  std::vector<Complex> want(degree / 2);
  for (int s = 0; s < degree / 2; s++) want[s] = z[s] * w[s];

  const double worst = MaxAbsDiff(got, want);
  std::cout << "SinC subring product, k=" << sub_degree << ", " << num_blocks
            << " blocks: max " << worst << std::endl;
  ASSERT_LT(worst, 1e-4);
}

INSTANTIATE_TEST_SUITE_P(
    SinC, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
