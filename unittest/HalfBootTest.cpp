// HalfBoot: coefficients in, slots out.
//
// WHY THIS IS THE PIPELINE'S KEYSTONE. EvalSpecialFFT compiles SlotToCoeff at
// GetStCStartLevel(), which is exactly default_encryption_level, so an ordinary
// ciphertext can feed it; it compiles CoeffToSlot at GetCtSStartLevel() =
// max_level, which nothing but a ModUp'd ciphertext reaches. So slot ->
// coefficient is a plain call and coefficient -> slot exists *only* inside
// bootstrapping. [SYLPH] section 3.2 fuses encoding conversions into
// bootstrapping; that is not an optimisation we may skip, it is the only door.
//
// Boot is domain-preserving because its CoeffToSlot and SlotToCoeff cancel.
// Stopping before the second one leaves the input's coefficients sitting in the
// output's slots -- which is [BAE]'s HalfBTS and what the digest section 4.3(b)
// already named as "the right structure for us".
//
// THE TEST. Encode a vector into *coefficients*, encrypt, HalfBoot, decode as
// *slots*. Getting the same vector back is the whole claim. Encoding one way and
// decoding the other is the point, not an oversight.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Testbed.h"

using word = uint32_t;

TEST_P(Testbed32, HalfBootTurnsCoefficientsIntoSlots) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);

  const int degree = param_->degree_;
  const int num_slots = degree / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  // Coefficients 0..N/2-1 carry the data; N/2..N-1 stay zero, so the imaginary
  // half of every slot is zero and the comparison is against a real vector.
  // Bootstrapping needs the message inside (-1, 1) (Doing.md 1.3).
  std::vector<double> coeffs(degree, 0.0);
  for (int i = 0; i < num_slots; i++) {
    coeffs[i] = 0.4 * std::sin(0.001 * i) + 0.2 * std::cos(0.017 * i);
  }

  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, 0, param_->GetScale(0), coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  Ciphertext<word> res;
  boot->HalfBoot(res, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const int out_level = param_->NPToLevel(res.GetNP());
  std::cout << "landed at level " << out_level << " (StC start is "
            << (param_->max_level_ - 12) << ", default encryption level "
            << default_encryption_level_ << ")" << std::endl;
  EXPECT_EQ(out_level, default_encryption_level_)
      << "the cycle must return to the level ordinary ciphertexts live at, or "
         "the flow does not close";

  std::vector<Complex> got;
  DecryptAndDecode(got, res);
  ASSERT_EQ(static_cast<int>(got.size()), num_slots);

  double worst = 0.0, absmax = 0.0, worst_imag = 0.0;
  for (int i = 0; i < num_slots; i++) {
    worst = std::max(worst, std::abs(got[i].real() - coeffs[i]));
    worst_imag = std::max(worst_imag, std::abs(got[i].imag()));
    absmax = std::max(absmax, std::abs(coeffs[i]));
  }
  std::cout << "coefficient i -> slot i: max abs err " << worst << " ("
            << -std::log2(worst / absmax) << " bits relative to |v| max "
            << absmax << ")" << std::endl;
  std::cout << "imaginary leakage (coefficients N/2..N-1 were zero): "
            << worst_imag << std::endl;
  // Bootstrapping's own precision is the bar; [SYLPH] 3.1.3 quotes 13 effective
  // bits for a 20-bit procedure, and Cheddar's Boot measures ~2.4e8 SNR.
  EXPECT_GT(-std::log2(worst / absmax), 12.0)
      << "HalfBoot lost more than a bootstrap should";
  EXPECT_LT(worst_imag, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
