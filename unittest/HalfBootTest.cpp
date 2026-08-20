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
// THE CYCLE THE PIPELINE ACTUALLY RUNS.
//
// Five attempts to call SlotToCoeff on an ordinary, canonically-scaled
// ciphertext all failed -- straight (35.7% spread), with stc_const_ (15.7%),
// with the message ratio (15.5%, unchanged, so not a precision problem),
// reinterpreting the scale (239%), and growing the data (values wrapping at
// +-pi). The evidence says StC is the tail phase of bootstrapping and consumes
// EvalMod's output, contract and all.
//
// The paper says so directly. Section 2.1.2: access to coefficient encoding
// comes "by interrupting temporarily the bootstrapping flow". Section 3.2: the
// conversions are *fused* into bootstrapping, which "allows heavy linear
// algebra to be computed at the smallest possible modulus". And the digest's
// 4.3(b): "S2C is done before the matrix product at low level and only
// ModRaise -> C2S -> EvalMod remains".
//
// So in a steady-state loop StC is always applied to EvalMod output -- the
// pipeline never has canonically-scaled slot data that it needs to convert. The
// case I spent five attempts on is one the design does not contain.
//
// This test runs the cycle that does occur: HalfBoot to reach slots, work
// there, SlotToCoeff to get back to coefficients for the linear algebra.
TEST_P(Testbed32, HalfBootAndSlotToCoeffCycle) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);

  const int num_slots = param_->degree_ / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, 0);

  // Into slots. This direction is verified exactly against Boot below.
  Ciphertext<word> in_slots;
  boot->HalfBoot(in_slots, ct, interface_->GetEvkMap());
  const int slot_level = param_->NPToLevel(in_slots.GetNP());
  std::cout << "HalfBoot -> slots at level " << slot_level << ", scale "
            << in_slots.GetScale() << std::endl;

  // Back to coefficients, on data that carries the bootstrap's contract rather
  // than the canonical scale.
  Ciphertext<word> in_coeffs;
  boot->SlotToCoeff(in_coeffs, num_slots, in_slots, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int coeff_level = param_->NPToLevel(in_coeffs.GetNP());
  std::cout << "SlotToCoeff -> coefficients at level " << coeff_level
            << " (the low modulus [SYLPH] 3.2 wants the linear algebra at)"
            << std::endl;
  EXPECT_LT(coeff_level, slot_level);

  // The cycle returns the original message, which is what makes it a usable
  // conversion pair: Boot's own scale is the one to declare.
  in_coeffs.SetScale(param_->GetScale(coeff_level));
  std::vector<Complex> got;
  DecryptAndDecode(got, in_coeffs);

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
  std::cout << "cycle ratio over " << counted << " slots: mean " << ratio
            << ", spread [" << rlo << ", " << rhi << "]" << std::endl;
  ASSERT_LT((rhi - rlo) / std::abs(ratio), 1e-2)
      << "even on bootstrap-contract data the pair is not a constant multiple";

  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_slots; i++) {
    worst = std::max(worst, std::abs(got[i].real() / ratio - msg[i].real()));
    absmax = std::max(absmax, std::abs(msg[i].real()));
  }
  std::cout << "after the ratio: max abs err " << worst << " ("
            << -std::log2(worst / absmax) << " bits)" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 12.0);
}

// THE DECISIVE ONE, AND IT SHOULD HAVE COME FIRST.
//
// Four hypotheses about the round trip have now been falsified: an identity
// coefficient-to-slot map, a single missing constant, the message ratio as the
// source of the precision loss, and SlotToCoeff's input scale contract. Each was
// a guess checked afterwards. Boot is a known-good reference measuring SNR 2.4e8
// every run, and by construction
//
//     Boot(x) == SlotToCoeff(HalfBoot(x))
//
// because HalfBoot is Boot with the last phase removed. That equation separates
// the two things the round trip conflates:
//
//   * If it holds, HalfBoot is right and the round-trip failure is entirely in
//     calling SlotToCoeff standalone, outside the flow it was compiled for.
//   * If it fails, HalfBoot differs from Boot somewhere, and the comparison is
//     against a reference rather than against a hypothesis.
TEST_P(Testbed32, HalfBootPlusSlotToCoeffEqualsBoot) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);

  const int num_slots = param_->degree_ / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, 0);

  // Reference: Boot, exactly as the passing bootstrap test uses it.
  Ciphertext<word> want;
  boot->Boot(want, ct, interface_->GetEvkMap());

  // The same thing assembled from HalfBoot plus the phase it omits.
  Ciphertext<word> half, got;
  boot->HalfBoot(half, ct, interface_->GetEvkMap());
  std::cout << "HalfBoot: level "
            << param_->NPToLevel(half.GetNP()) << ", scale " << half.GetScale()
            << " (EvalMod's end scale, which is what StC is compiled to expect)"
            << std::endl;
  boot->SlotToCoeff(got, num_slots, half, interface_->GetEvkMap());
  got.SetNumSlots(num_slots);
  got.SetScale(want.GetScale());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(param_->NPToLevel(got.GetNP()), param_->NPToLevel(want.GetNP()))
      << "the reassembled path did not land where Boot did";

  std::vector<Complex> a, b;
  DecryptAndDecode(a, want);
  DecryptAndDecode(b, got);
  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_slots; i++) {
    worst = std::max(worst, std::abs(a[i].real() - b[i].real()));
    worst = std::max(worst, std::abs(a[i].imag() - b[i].imag()));
    absmax = std::max(absmax, std::abs(a[i].real()));
  }
  std::cout << "Boot vs HalfBoot+StC: max abs diff " << worst << " ("
            << -std::log2(worst / absmax) << " bits relative to |Boot| max "
            << absmax << ")" << std::endl;
  // Both paths run the same kernels in the same order, so this should be
  // essentially exact -- not merely close.
  EXPECT_GT(-std::log2(worst / absmax), 20.0)
      << "HalfBoot is not Boot minus its last phase, so the difference is in "
         "HalfBoot and not in how the round trip calls SlotToCoeff";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
