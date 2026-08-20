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

  // SlotToCoeff's three phases are LinearTransforms pinned to levels with
  // per-phase scales, and stc_const_ is split across them as its cube root. So
  // a wrong input scale does not shift the answer by a constant -- it stops the
  // phases composing, which is why three successive constant compensations each
  // reduced the error and none removed it. Inside Boot the input carries
  // EvalMod's end scale, not the canonical scale of its level.
  //
  // Reinterpreting the scale is free: no level, no kernel, only the declared
  // value changes. That is the same technique that broke RMSNorm by handing
  // EvalPoly a non-canonical scale -- here it is the opposite, matching the
  // scale the library asks for rather than inventing one.
  // StC's phases each rescale by about 2^30. Inside Boot its input carries
  // EvalMod's end scale; an ordinary ciphertext at the same level carries the
  // canonical one, which is 8.4e6 (~2^23) smaller in actual integers. The first
  // rescale then throws away 23 bits and three phases cannot recover them --
  // that is the 15% floor, and it is why *reinterpreting* the scale made things
  // worse rather than better: reinterpreting changes the declared value and
  // leaves the integers alone, so it fixed nothing and broke the bookkeeping.
  //
  // The data has to actually grow. Multiplying by an integer constant encoded
  // at scale 1 does that without a rescale, so it costs no level: the product
  // keeps the input's declared scale while the integers scale up, and declaring
  // the result at StC's expected scale then decodes to the same message.
  const double have = ct.GetScale();
  const double want_scale = boot->GetStCInputScale();
  const double grow = want_scale / have;
  std::cout << "growing the data by " << grow << " so StC's rescales have the "
            << "bits they were compiled for (" << have << " -> " << want_scale
            << ")" << std::endl;
  {
    const int l = param_->NPToLevel(ct.GetNP());
    Constant<word> c;
    context_->encoder_.EncodeConstant(c, l, 1.0, grow);
    Ciphertext<word> grown;
    context_->Mult(grown, ct, c);
    grown.SetScale(want_scale);
    ct = std::move(grown);
    std::cout << "  now at level " << param_->NPToLevel(ct.GetNP())
              << " (no rescale, so no level spent)" << std::endl;
  }

  Ciphertext<word> coeff_ct;
  boot->SlotToCoeff(coeff_ct, num_slots, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int coeff_level = param_->NPToLevel(coeff_ct.GetNP());
  std::cout << "SlotToCoeff: level " << level << " -> " << coeff_level
            << " (this direction is a plain call; the transform is compiled at "
            << "GetStCStartLevel(), which is the default encryption level)"
            << std::endl;

  // SlotToCoeff bakes stc_const_ into its transform, and inside Boot that is
  // undone by scaleup_const_ and EvalMod. Standalone there is nothing in
  // between, so the coefficients come out ~1e-7 of their intended size -- and
  // EvalMod's precision is absolute, so shrinking the argument by 1e-7 is what
  // turned a constant ratio into one with 18% spread. Restore the magnitude
  // before the bootstrap rather than guess which constant it was.
  {
    const int cl = param_->NPToLevel(coeff_ct.GetNP());
    Constant<word> inv_stc;
    // Two factors, both measured rather than derived. stc_const_ is what the
    // StC transform bakes in; the second is 2^log_message_ratio, which showed
    // up as the round trip coming back exactly 1/32.06 of its input. Feeding
    // EvalMod an argument 32x below what it is built for also costs five bits,
    // which is what the residual 15.7% ratio spread was.
    // The previous run divided by this where it should have multiplied, which
    // is why the ratio moved 32x the wrong way.
    const double msg_ratio =
        static_cast<double>(1 << boot->GetBootParameter().GetLogMessageRatio());
    std::cout << "compensating stc_const_ " << boot->GetStCConst()
              << " and message ratio " << msg_ratio << std::endl;
    context_->encoder_.EncodeConstant(inv_stc, cl, param_->GetScale(cl),
                                      msg_ratio / boot->GetStCConst());
    Ciphertext<word> scaled;
    context_->Mult(scaled, coeff_ct, inv_stc);
    context_->Rescale(coeff_ct, scaled);
    std::cout << "compensated stc_const_, now at level "
              << param_->NPToLevel(coeff_ct.GetNP()) << std::endl;
  }

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
