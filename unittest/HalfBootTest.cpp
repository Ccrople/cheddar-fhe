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

// THE TEST IS A ROUND TRIP, ON PURPOSE.
//
// A first attempt encoded into coefficient i and expected slot i, and the
// slot/coefficient ratio came back spread over [-0.36, 0.355] instead of being
// one number -- so the gap was never a missing scale factor. CtS carries a
// bit-reversal, which [SYLPH] section 3.3 spells out ("designed to accommodate
// the bit-reversal operation that occurs when moving to coefficient encoding
// ... the roles of dimensions i and k are swapped"), and the identity map was
// simply wrong.
//
// Rather than reconstruct that permutation, this goes slot -> SlotToCoeff ->
// HalfBoot -> slot. The permutation cancels because the two transforms are
// inverse, and this is the shape the pipeline actually uses: the linear algebra
// happens in between, in coefficient encoding at a small ring.
//
// The constants do not cancel -- StC's transform bakes in stc_const_ and CtS's
// bakes in cts_const_, which only compose correctly inside Boot alongside
// scaleup_const_ and EvalMod. So the round trip is expected to return a
// *constant multiple* of the input, and the test identifies it and asserts it
// is constant before trusting it.
TEST_P(Testbed32, HalfBootClosesTheRoundTrip) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);

  const int degree = param_->degree_;
  const int num_slots = degree / 2;
  const int level = default_encryption_level_;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg(num_slots);
  for (int i = 0; i < num_slots; i++) {
    msg[i] = Complex(0.4 * std::sin(0.001 * i) + 0.2 * std::cos(0.017 * i), 0.0);
  }
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  Ciphertext<word> coeff_ct;
  boot->SlotToCoeff(coeff_ct, num_slots, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int coeff_level = param_->NPToLevel(coeff_ct.GetNP());
  std::cout << "SlotToCoeff: level " << level << " -> " << coeff_level
            << " (this direction is a plain call; the transform is compiled at "
            << "GetStCStartLevel(), which is the default encryption level)"
            << std::endl;

  Ciphertext<word> res;
  boot->HalfBoot(res, coeff_ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int out_level = param_->NPToLevel(res.GetNP());
  std::cout << "HalfBoot: -> level " << out_level << std::endl;
  EXPECT_EQ(out_level, level) << "the cycle must return to where it started";

  std::vector<Complex> got;
  DecryptAndDecode(got, res);
  ASSERT_EQ(static_cast<int>(got.size()), num_slots);

  double rlo = 1e300, rhi = -1e300, rsum = 0.0;
  int counted = 0;
  for (int i = 0; i < num_slots; i++) {
    if (std::abs(msg[i].real()) < 0.05) continue;
    const double r = got[i].real() / msg[i].real();
    rlo = std::min(rlo, r);
    rhi = std::max(rhi, r);
    rsum += r;
    counted++;
  }
  const double ratio = rsum / counted;
  std::cout << "round-trip ratio over " << counted << " slots: mean " << ratio
            << ", spread [" << rlo << ", " << rhi << "]" << std::endl;
  ASSERT_LT((rhi - rlo) / std::abs(ratio), 1e-2)
      << "the round trip is not a constant multiple of the input, so the "
         "conversions are not inverse and this is not a constants problem";

  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_slots; i++) {
    worst = std::max(worst, std::abs(got[i].real() / ratio - msg[i].real()));
    absmax = std::max(absmax, std::abs(msg[i].real()));
  }
  std::cout << "after dividing by the ratio: max abs err " << worst << " ("
            << -std::log2(worst / absmax) << " bits)" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 12.0);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
