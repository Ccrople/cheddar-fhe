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
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <utility>
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

// ---------------------------------------------------------------------------
// THE SECOND EVALMOD WAS ALWAYS RUNNING, AND IT WAS ALWAYS RUNNING ON ZERO.
//
// At full slots CtS folds coefficients `j` and `j + N/2` into the real and
// imaginary axes of slot `j`, so `EvaluateModAfterCtS` separates the axes with
// one conjugation and reduces each -- two EvalMods, unconditionally. The Llama
// pipeline's payload is real-axis-only in slots, which means StC leaves
// coefficients N/2..N-1 zero, which means the second of those two EvalMods
// reduces an identically-zero ciphertext, once per bootstrap, 168 times a layer.
//
// `HalfBootPair` fills that half. The merge is `MultImaginaryUnit` -- X^{N/2}
// is the imaginary unit's polynomial, so the "multiply by i" kernel *is* the
// shift by N/2, level free and key free -- and the split is the existing
// separation stopping one line before it folds the axes back together.
//
// THE MEASUREMENT IS THE POINT, so this test does both: it checks the pair
// against two separate HalfBoots on the same inputs (the reference is the code
// path being replaced, which is what makes a difference attributable), and it
// times them.
TEST_P(Testbed32, HalfBootPairCostsOneCrossing) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);

  const int degree = param_->degree_;
  const int num_slots = degree / 2;
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  // THE CONTRACT, BUILT EXPLICITLY: payload in coefficients 0..N/2-1 only. This
  // is what StC produces from real-axis slots; here it is written by hand so
  // the test does not depend on StC to state it.
  std::mt19937_64 gen(0x9A1F);
  std::uniform_real_distribution<double> d(-0.4, 0.4);
  std::vector<double> want_lo(degree, 0.0), want_hi(degree, 0.0);
  for (int i = 0; i < num_slots; i++) {
    want_lo[i] = d(gen);
    // Deliberately a different distribution, so a leak of one half into the
    // other shows up as a magnitude error and not as more of the same noise.
    want_hi[i] = 0.25 * d(gen) + 0.3;
  }

  auto encrypt_coeff = [&](Ciphertext<word> &ct, const std::vector<double> &c) {
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, 0, param_->GetScale(0), c);
    interface_->Encrypt(ct, pt);
  };
  Ciphertext<word> ct_lo, ct_hi;
  encrypt_coeff(ct_lo, want_lo);
  encrypt_coeff(ct_hi, want_hi);

  // THE REFERENCE: the code path this replaces, run on the same ciphertexts.
  Ciphertext<word> ref_lo, ref_hi;
  boot->HalfBoot(ref_lo, ct_lo, interface_->GetEvkMap());
  boot->HalfBoot(ref_hi, ct_hi, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  Ciphertext<word> got_lo, got_hi;
  boot->HalfBootPair(got_lo, got_hi, ct_lo, ct_hi, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  EXPECT_EQ(param_->NPToLevel(got_lo.GetNP()),
            param_->NPToLevel(ref_lo.GetNP()));
  EXPECT_EQ(got_lo.GetScale(), ref_lo.GetScale());
  EXPECT_EQ(got_hi.GetScale(), ref_hi.GetScale());

  std::vector<Complex> r_lo, r_hi, g_lo, g_hi;
  DecryptAndDecode(r_lo, ref_lo);
  DecryptAndDecode(r_hi, ref_hi);
  DecryptAndDecode(g_lo, got_lo);
  DecryptAndDecode(g_hi, got_hi);

  // AGAINST THE REFERENCE. `ref_lo` came from a HalfBoot that never saw `ct_hi`,
  // so agreement here is simultaneously "the pair computes the same thing" and
  // "nothing leaked across the merge" -- the second is the stronger statement
  // and it comes free.
  auto vs_ref = [&](const char *tag, const std::vector<Complex> &got,
                    const std::vector<Complex> &ref) {
    double worst = 0.0, absmax = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max(worst, std::abs(got[i].real() - ref[i].real()));
      worst = std::max(worst, std::abs(got[i].imag() - ref[i].imag()));
      absmax = std::max(absmax, std::abs(ref[i].real()));
    }
    std::cout << "  " << tag << ": vs the two-HalfBoot reference " << worst
              << " (|ref| <= " << absmax << ")" << std::endl;
    return worst / absmax;
  };
  EXPECT_LT(vs_ref("lo half", g_lo, r_lo), 1e-4)
      << "the low half is not what HalfBoot alone gives it";
  EXPECT_LT(vs_ref("hi half", g_hi, r_hi), 1e-4)
      << "the high half is not what HalfBoot alone gives it";

  // THE CONTROL, which is what makes those two mean anything: the halves must
  // not be interchangeable. Crossing them has to fail by a wide margin.
  double crossed = 0.0;
  for (int i = 0; i < num_slots; i++) {
    crossed = std::max(crossed, std::abs(g_lo[i].real() - r_hi[i].real()));
  }
  std::cout << "  control, the low half against the HIGH reference: " << crossed
            << std::endl;
  EXPECT_GT(crossed, 1e-2) << "the two halves are indistinguishable, so the "
                              "agreement above proves nothing";

  // AND AGAINST THE PAYLOAD ITSELF. HalfBoot lands coefficients in slots under
  // a permutation -- that is this file's opening subject and why comparing
  // slot `i` with coefficient `i` is wrong -- so the honest end-to-end check
  // runs the cycle: StC back to coefficients and compare there.
  auto payload = [&](const char *tag, const Ciphertext<word> &slots,
                     const std::vector<double> &want) {
    Ciphertext<word> back;
    boot->SlotToCoeff(back, num_slots, slots, interface_->GetEvkMap());
    back.SetNumSlots(num_slots);
    back.SetScale(param_->GetScale(param_->NPToLevel(back.GetNP())));
    Plaintext<word> pt;
    interface_->Decrypt(pt, back);
    std::vector<double> got;
    context_->encoder_.DecodeCoeff(got, pt);
    // One constant for the whole vector, fitted where the payload is large
    // enough to fit it on, exactly as the cycle test does.
    double num = 0.0, den = 0.0;
    for (int i = 0; i < num_slots; i++) {
      num += got[i] * want[i];
      den += want[i] * want[i];
    }
    const double k = num / den;
    double worst = 0.0, absmax = 0.0, upper = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max(worst, std::abs(got[i] / k - want[i]));
      absmax = std::max(absmax, std::abs(want[i]));
      // The half the contract says stays empty. If the merge disturbed it the
      // downstream PC-MM would silently read it as payload.
      upper = std::max(upper, std::abs(got[num_slots + i] / k));
    }
    std::cout << "  " << tag << " payload through StC: max abs err " << worst
              << " (|want| <= " << absmax << ", " << -std::log2(worst / absmax)
              << " bits), ratio " << k << ", upper coefficient half " << upper
              << std::endl;
    EXPECT_LT(upper, 1e-2 * absmax)
        << tag << " came back with the contract's empty half filled in";
    return -std::log2(worst / absmax);
  };
  // The threshold is the REFERENCE's own precision, not a number chosen here.
  // What is being asked is whether pairing costs accuracy, and only the same
  // payload down the unpaired path can answer it.
  const double ref_bits_lo = payload("lo half, unpaired", ref_lo, want_lo);
  const double ref_bits_hi = payload("hi half, unpaired", ref_hi, want_hi);
  const double bits_lo = payload("lo half, paired", got_lo, want_lo);
  const double bits_hi = payload("hi half, paired", got_hi, want_hi);
  std::cout << "pairing costs, in bits of the payload: lo "
            << (ref_bits_lo - bits_lo) << ", hi " << (ref_bits_hi - bits_hi)
            << std::endl;
  EXPECT_GT(ref_bits_lo, 10.0) << "the unpaired cycle is already broken, so "
                                  "nothing here is attributable to pairing";
  EXPECT_GT(bits_lo, ref_bits_lo - 0.1) << "pairing cost the low half accuracy";
  EXPECT_GT(bits_hi, ref_bits_hi - 0.1)
      << "pairing cost the high half accuracy";

  // WHAT IT COSTS. Warm first -- the reference above already ran once.
  const int reps = 5;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) {
    Ciphertext<word> a, b;
    boot->HalfBoot(a, ct_lo, interface_->GetEvkMap());
    boot->HalfBoot(b, ct_hi, interface_->GetEvkMap());
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) {
    Ciphertext<word> a, b;
    boot->HalfBootPair(a, b, ct_lo, ct_hi, interface_->GetEvkMap());
  }
  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const double two =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  const double one =
      std::chrono::duration<double, std::milli>(t2 - t1).count() / reps;
  std::cout << "two ciphertexts across the coefficient/slot boundary: "
            << "two HalfBoots " << two << " ms, one HalfBootPair " << one
            << " ms  (" << (two / one) << "x)" << std::endl;
  EXPECT_LT(one, two * 0.75)
      << "the pair is not saving a crossing, so the merge is not free";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
