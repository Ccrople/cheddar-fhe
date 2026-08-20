// The attention packing layout of [SYLPH] section 3.2, and the measurement it
// stands on.
//
// WHY THE MEASUREMENT IS THE POINT. HalfBootTest established that the
// conversion pair is a transport -- the values come back unchanged -- but it
// established that by running a *round trip*, in which any permutation cancels
// against its own inverse. Its own comment says so: "Rather than reconstruct
// that permutation, this goes slot -> SlotToCoeff -> HalfBoot -> slot." So
// after that test the values were known and the addresses were not, and the
// layout is entirely a statement about addresses.
//
// [SYLPH] 3.2 says the layout is "designed to accommodate the bit-reversal
// operation that occurs when moving to coefficient encoding". Reading
// EvalSpecialFFT.cpp says where that bit reversal comes from in Cheddar: the
// host encoder's SpecialFFT/SpecialIFFT each carry a BitReverseVector
// (Encode.cpp:252, 263) and PopulatePlainMatrices builds only butterfly
// stages, so the homomorphic transform differs from the host one by exactly one
// bit reversal. That is a derivation. Doing.md 1.5t's method note is that five
// hypotheses died in one afternoon because they were reasoned about rather than
// measured, so this file measures it.
//
// THE PROBE. Encode +-A into the *coefficients*, HalfBoot, decode as *slots*.
// Probe b sets coefficient p to -A when bit b of p is set and +A otherwise, so
// the sign of each decoded slot reads off one bit of the coefficient index that
// landed there. Sixteen probes plus one all-positive reference recover the
// entire map with no assumption about its shape -- not "check the guess", but
// "read the answer, then check the guess against it".

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "Testbed.h"
#include "common/CommonUtils.h"
#include "extension/AttentionPacking.h"

using word = uint32_t;
using Axis = AttentionPacking::Axis;
using SlotPosition = AttentionPacking::SlotPosition;
using TensorIndex = AttentionPacking::TensorIndex;

namespace {

// Llama-3-8B attention at 128 tokens, which is the shape [SYLPH] table 4 and
// section 3.2 are written for.
constexpr int kNumHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kNumTokens = 128;
constexpr int kDegree = 1 << 16;

AttentionPacking SylphShape() {
  return AttentionPacking(kNumHeads, kHeadDim, kNumTokens, kDegree);
}

std::string AxisOrderString(const std::array<Axis, 3> &order) {
  std::string s;
  for (int i = 0; i < 3; i++) {
    if (i) s += " < ";
    s += AttentionPacking::AxisName(order[i]);
  }
  return s;
}

}  // namespace

// The shape arithmetic, against the paper's own numbers. [SYLPH] states
// "nheads/4 = 8 ciphertexts"; this class derives 8 from the ring degree instead,
// so the two agreeing is a check and not a tautology.
TEST(AttentionPackingHost, ShapeMatchesTheStatedCiphertextCount) {
  const AttentionPacking layout = SylphShape();
  EXPECT_EQ(layout.GetChannelsPerCiphertext(), 16);
  EXPECT_EQ(layout.GetNumCiphertexts(), 8);
  EXPECT_EQ(layout.GetNumCiphertexts(), kNumHeads / 4);
  EXPECT_EQ(layout.GetTensorSize(), layout.GetNumCiphertexts() * kDegree);
  EXPECT_EQ(layout.GetTensorSize(), 1 << 19);
}

// The literal formula from the paper, transcribed with its constants intact,
// against the generalised one the class implements. If the generalisation is
// wrong anywhere it is wrong here.
TEST(AttentionPackingHost, MatchesTheLiteralFormula) {
  const AttentionPacking layout = SylphShape();
  for (int l = 0; l < layout.GetNumCiphertexts(); l++) {
    for (int s = 0; s < kDegree; s++) {
      const TensorIndex want{/*head=*/(s / 128) % kNumHeads,
                             /*channel=*/l * 512 / kNumHeads +
                                 s / (128 * kNumHeads),
                             /*token=*/s % 128};
      const TensorIndex got = layout.CoeffPosition(l, s);
      ASSERT_EQ(got.head, want.head) << "l=" << l << " s=" << s;
      ASSERT_EQ(got.channel, want.channel) << "l=" << l << " s=" << s;
      ASSERT_EQ(got.token, want.token) << "l=" << l << " s=" << s;
    }
  }
}

// Every tensor entry sits in exactly one place, and CoeffIndexOf inverts
// CoeffPosition. A layout that loses or duplicates an entry is worse than a
// wrong one, because the matrix product would still run.
TEST(AttentionPackingHost, TheCoefficientMapIsABijection) {
  const AttentionPacking layout = SylphShape();
  std::vector<int> hits(layout.GetTensorSize(), 0);
  for (int ct = 0; ct < layout.GetNumCiphertexts(); ct++) {
    for (int coeff = 0; coeff < kDegree; coeff++) {
      const TensorIndex t = layout.CoeffPosition(ct, coeff);
      hits[layout.TensorOffset(t)]++;
      int back_ct = -1, back_coeff = -1;
      layout.CoeffIndexOf(t, back_ct, back_coeff);
      ASSERT_EQ(back_ct, ct);
      ASSERT_EQ(back_coeff, coeff);
    }
  }
  EXPECT_EQ(*std::min_element(hits.begin(), hits.end()), 1);
  EXPECT_EQ(*std::max_element(hits.begin(), hits.end()), 1);
}

TEST(AttentionPackingHost, TheSlotMapIsABijection) {
  std::vector<int> hits(kDegree, 0);
  for (int coeff = 0; coeff < kDegree; coeff++) {
    const SlotPosition pos = AttentionPacking::SlotOfCoeff(coeff, kDegree);
    ASSERT_GE(pos.slot, 0);
    ASSERT_LT(pos.slot, kDegree / 2);
    hits[pos.slot + (pos.imaginary ? kDegree / 2 : 0)]++;
    ASSERT_EQ(AttentionPacking::CoeffOfSlot(pos, kDegree), coeff);
  }
  EXPECT_EQ(*std::min_element(hits.begin(), hits.end()), 1);
  EXPECT_EQ(*std::max_element(hits.begin(), hits.end()), 1);
}

// The claim [SYLPH] 3.2 makes about what the conversion does to the axes,
// stated as arithmetic. The class computes the orders by toggling index bits,
// so this is asserting the outcome of a computation, not restating a constant.
TEST(AttentionPackingHost, TheConversionExchangesTokenAndChannel) {
  const AttentionPacking layout = SylphShape();
  const auto coeff_order = layout.CoeffAxisOrder();
  const auto slot_order = layout.SlotAxisOrder();
  std::cout << "fastest-varying first, coefficient index: "
            << AxisOrderString(coeff_order) << std::endl;
  std::cout << "fastest-varying first, slot index:        "
            << AxisOrderString(slot_order) << std::endl;

  EXPECT_EQ(coeff_order[0], Axis::kToken);
  EXPECT_EQ(coeff_order[1], Axis::kHead);
  EXPECT_EQ(coeff_order[2], Axis::kChannel);

  // Reversing the bits reverses the fields as well as the bits inside them, so
  // the outer two axes trade places and the head axis stays in the middle.
  EXPECT_EQ(slot_order[0], Axis::kChannel);
  EXPECT_EQ(slot_order[1], Axis::kHead);
  EXPECT_EQ(slot_order[2], Axis::kToken);

  // The coefficient index's top bit is the real/imaginary selector and it is
  // the channel field's own top bit, so the channel axis -- and only it -- is
  // split across the two parts of a slot. Getting this wrong is how an
  // operator silently mixes two channels into one complex number.
  std::cout << "the real/imaginary flag splits the "
            << AttentionPacking::AxisName(layout.ImaginaryFlagAxis())
            << " axis" << std::endl;
  EXPECT_EQ(layout.ImaginaryFlagAxis(), Axis::kChannel);
  for (int slot = 0; slot < 4; slot++) {
    const TensorIndex re = layout.SlotToTensor(0, {slot, false});
    const TensorIndex im = layout.SlotToTensor(0, {slot, true});
    EXPECT_EQ(re.head, im.head);
    EXPECT_EQ(re.token, im.token);
    EXPECT_EQ(im.channel - re.channel, layout.GetChannelsPerCiphertext() / 2);
  }
}

TEST(AttentionPackingHost, PackingRoundTrips) {
  const AttentionPacking layout = SylphShape();
  std::vector<double> tensor(layout.GetTensorSize());
  for (int i = 0; i < layout.GetTensorSize(); i++) {
    tensor[i] = std::sin(0.001 * i) + 0.25 * std::cos(0.017 * i);
  }

  std::vector<std::vector<double>> coeff_cts;
  layout.PackCoeff(coeff_cts, tensor);
  ASSERT_EQ(static_cast<int>(coeff_cts.size()), layout.GetNumCiphertexts());
  std::vector<double> back;
  layout.UnpackCoeff(back, coeff_cts);
  ASSERT_EQ(back.size(), tensor.size());
  for (size_t i = 0; i < tensor.size(); i++) ASSERT_EQ(back[i], tensor[i]);

  std::vector<std::vector<Complex>> slot_cts;
  layout.PackSlot(slot_cts, tensor);
  ASSERT_EQ(static_cast<int>(slot_cts.size()), layout.GetNumCiphertexts());
  ASSERT_EQ(static_cast<int>(slot_cts[0].size()), kDegree / 2);
  layout.UnpackSlot(back, slot_cts);
  for (size_t i = 0; i < tensor.size(); i++) ASSERT_EQ(back[i], tensor[i]);

  // And the two packings are the same data, related by the conversion. This is
  // the property the pipeline depends on: hold the tensor in the slot layout,
  // convert, and the coefficient layout is already correct.
  for (int ct = 0; ct < layout.GetNumCiphertexts(); ct++) {
    for (int coeff = 0; coeff < kDegree; coeff++) {
      const SlotPosition pos = AttentionPacking::SlotOfCoeff(coeff, kDegree);
      const Complex &slot = slot_cts[ct][pos.slot];
      ASSERT_EQ(pos.imaginary ? slot.imag() : slot.real(), coeff_cts[ct][coeff]);
    }
  }
}

// A second shape, so the generalisation is exercised away from the constants it
// was read off. Half the tokens means twice the channels per ciphertext and
// half the ciphertexts.
TEST(AttentionPackingHost, GeneralisesAwayFromTheSylphShape) {
  const AttentionPacking layout(kNumHeads, kHeadDim, 64, kDegree);
  EXPECT_EQ(layout.GetChannelsPerCiphertext(), 32);
  EXPECT_EQ(layout.GetNumCiphertexts(), 4);
  EXPECT_EQ(layout.GetTensorSize(), layout.GetNumCiphertexts() * kDegree);
  EXPECT_EQ(layout.CoeffAxisOrder()[0], Axis::kToken);
  EXPECT_EQ(layout.SlotAxisOrder()[0], Axis::kChannel);
  EXPECT_EQ(layout.ImaginaryFlagAxis(), Axis::kChannel);
}

// WHAT THE LAYOUT IS FOR, checked against the two operators that consume it.
//
// The route from the coefficient encoding down to the matrix product's ring is
// RingSwitch then ModDecomp, and both end in the same operation: the X^k-adic
// split e*_i(m)[s] = m[i + k*s]. RingSwitch.h calls its step 2 "a plain
// stride-k gather on both components"; Mlwe.h states the same formula for
// ModDecomp. So composing them is repeated stride splitting of the coefficient
// index, which is pure arithmetic on the layout and needs no GPU.
//
// [SYLPH] table 4 runs PCMM at ring degree 256 with a "Row-split" layout, and
// [BAE]'s identity needs "d ciphertexts whose messages m_i are the rows of a
// matrix M". So the test of the layout is whether the two gathers leave one
// row -- one token -- per degree-256 element. They do, and that is not an
// accident: it is *because* the token field occupies the lowest coefficient
// bits that a stride gather splits rows instead of shredding channels. Put the
// token field anywhere else and this test fails.
TEST(AttentionPackingHost, TheRingSwitchChainLeavesOneTokenPerSmallRingElement) {
  const AttentionPacking layout = SylphShape();
  constexpr int kRingSwitchStride = kDegree / 4096;  // 2^16 -> 2^12
  constexpr int kModDecompStride = 4096 / 256;       // 2^12 -> 2^8
  const int num_elements = kRingSwitchStride * kModDecompStride;

  // element -> the token it holds, -1 until seen
  std::vector<int> token_of(num_elements, -1);
  std::vector<int> entries(num_elements, 0);
  std::vector<uint32_t> heads_seen(num_elements, 0);
  std::vector<uint32_t> channels_seen(num_elements, 0);

  for (int coeff = 0; coeff < kDegree; coeff++) {
    const TensorIndex t = layout.CoeffPosition(0, coeff);
    const int i1 = coeff % kRingSwitchStride;
    const int s1 = coeff / kRingSwitchStride;
    const int i2 = s1 % kModDecompStride;
    const int s2 = s1 / kModDecompStride;
    ASSERT_GE(s2, 0);
    ASSERT_LT(s2, 256) << "the chain does not land inside the degree-256 ring";

    const int element = i1 * kModDecompStride + i2;
    if (token_of[element] == -1) token_of[element] = t.token;
    ASSERT_EQ(token_of[element], t.token)
        << "element " << element << " holds more than one token, so it is not "
           "a row and [BAE]'s identity does not apply to it";
    entries[element]++;
    heads_seen[element] |= 1u << t.head;
    channels_seen[element] |= 1u << t.channel;
  }

  // Each element is one token by 16 heads by 16 channels. The heads are not
  // 0..15: the second gather's stride runs past the token field and takes the
  // head field's own low bit with it, so an element holds one parity class of
  // heads. That is still one row per element -- the head axis splits across
  // elements exactly as the token axis does -- but asserting heads 0..15 was
  // wrong, and this is what the run said instead of what the shape suggested.
  for (int e = 0; e < num_elements; e++) {
    ASSERT_EQ(entries[e], 256) << "element " << e << " is not full";
    ASSERT_EQ(__builtin_popcount(heads_seen[e]), 16)
        << "element " << e << " does not hold 16 heads";
    ASSERT_EQ(__builtin_popcount(channels_seen[e]), 16)
        << "element " << e << " does not hold 16 channels";
    int parity = -1;
    for (int h = 0; h < kNumHeads; h++) {
      if (!(heads_seen[e] & (1u << h))) continue;
      if (parity == -1) parity = h % 2;
      ASSERT_EQ(h % 2, parity)
          << "element " << e << " mixes head parities, so the head axis is "
             "not split cleanly by the gather";
    }
  }

  std::vector<int> distinct(token_of);
  std::sort(distinct.begin(), distinct.end());
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
  std::cout << num_elements
            << " degree-256 elements, each one token by 16 heads by 16 "
               "channels, "
            << distinct.size() << " distinct tokens over them" << std::endl;
  EXPECT_EQ(static_cast<int>(distinct.size()), kNumTokens);
}

// ---------------------------------------------------------------------------
// The measurement.
// ---------------------------------------------------------------------------

// Read the coefficient-to-slot permutation off the hardware, bit by bit, and
// only then compare it with AttentionPacking::SlotOfCoeff.
//
// The amplitude is 0.005 because that is the size of a coefficient of an
// ordinary boot input: a slot message uniform on (-1, 1) has coefficients of
// about 1/sqrt(num_slots). Filling every coefficient with +-0.005 keeps EvalMod
// in the range it is compiled for, and uniformly so, which is the easy case for
// it. Only the signs are read, so the constant HalfBoot applies is irrelevant
// beyond its own sign, which the reference probe measures.
TEST_P(Testbed32, TheConversionPermutationIsABitReversal) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);

  const int degree = param_->degree_;
  const int num_slots = degree / 2;
  const int log_degree = Log2Ceil(degree);
  ASSERT_EQ(degree, kDegree) << "this measurement is written for degree 2^16";

  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  constexpr double kAmplitude = 0.005;

  // Run one probe: coefficients +-kAmplitude, out come slots.
  auto run_probe = [&](const std::vector<double> &coeffs,
                       std::vector<Complex> &slots) {
    Plaintext<word> ptxt;
    context_->encoder_.EncodeCoeff(ptxt, 0, DetermineScale(0), coeffs);
    Ciphertext<word> ct;
    interface_->Encrypt(ct, ptxt);
    Ciphertext<word> in_slots;
    boot->HalfBoot(in_slots, ct, interface_->GetEvkMap());
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    DecryptAndDecode(slots, in_slots);
    ASSERT_GE(static_cast<int>(slots.size()), num_slots);
  };

  // Reference probe: every coefficient positive. It fixes the sign of the
  // constant HalfBoot leaves behind, separately for the real and the imaginary
  // part, and it is also the first evidence that the map is a permutation --
  // every slot should come back with the same magnitude. On its own that is
  // weak, since an average of two equal inputs also has the right magnitude;
  // the bit probes close the gap, because averaging two coefficients whose
  // index bits differ produces a value near zero and the margin check below
  // rejects it.
  std::vector<double> coeffs(degree, kAmplitude);
  std::vector<Complex> reference;
  run_probe(coeffs, reference);

  double lo = 1e300, hi = -1e300;
  for (int s = 0; s < num_slots; s++) {
    lo = std::min({lo, std::abs(reference[s].real()),
                   std::abs(reference[s].imag())});
    hi = std::max({hi, std::abs(reference[s].real()),
                   std::abs(reference[s].imag())});
  }
  std::cout << "reference probe: |value| in [" << lo << ", " << hi << "], "
            << "ratio " << hi / lo << std::endl;
  ASSERT_LT(hi / lo - 1.0, 1e-2)
      << "the reference probe does not come back with one magnitude, so the "
         "conversion mixes coefficients instead of permuting them and no "
         "permutation exists to measure";

  const double sign_re = reference[0].real() > 0 ? 1.0 : -1.0;
  const double sign_im = reference[0].imag() > 0 ? 1.0 : -1.0;
  std::cout << "constant sign: real " << sign_re << ", imag " << sign_im
            << " (magnitude " << hi << " for input " << kAmplitude << ")"
            << std::endl;

  // One probe per bit of the coefficient index.
  std::vector<int> measured_re(num_slots, 0), measured_im(num_slots, 0);
  for (int bit = 0; bit < log_degree; bit++) {
    for (int p = 0; p < degree; p++) {
      coeffs[p] = ((p >> bit) & 1) ? -kAmplitude : kAmplitude;
    }
    std::vector<Complex> slots;
    run_probe(coeffs, slots);
    double worst_margin = 1e300;
    for (int s = 0; s < num_slots; s++) {
      const double re = slots[s].real() * sign_re;
      const double im = slots[s].imag() * sign_im;
      worst_margin = std::min({worst_margin, std::abs(re), std::abs(im)});
      if (re < 0) measured_re[s] |= 1 << bit;
      if (im < 0) measured_im[s] |= 1 << bit;
    }
    std::cout << "probe bit " << bit << ": smallest |value| " << worst_margin
              << " (signs are read at " << (worst_margin / hi)
              << " of the reference magnitude)" << std::endl;
    ASSERT_GT(worst_margin, 0.25 * hi)
        << "a decoded value came back near zero, so its sign is noise and this "
           "probe cannot be believed";
  }

  // What was measured is a permutation.
  std::vector<int> hits(degree, 0);
  for (int s = 0; s < num_slots; s++) {
    ASSERT_GE(measured_re[s], 0);
    ASSERT_LT(measured_re[s], degree);
    ASSERT_GE(measured_im[s], 0);
    ASSERT_LT(measured_im[s], degree);
    hits[measured_re[s]]++;
    hits[measured_im[s]]++;
  }
  EXPECT_EQ(*std::min_element(hits.begin(), hits.end()), 1);
  EXPECT_EQ(*std::max_element(hits.begin(), hits.end()), 1)
      << "the measured map sends two coefficients to the same place";

  // And only now, the comparison with the derivation.
  int mismatches = 0;
  for (int s = 0; s < num_slots; s++) {
    const int want_re = AttentionPacking::CoeffOfSlot({s, false}, degree);
    const int want_im = AttentionPacking::CoeffOfSlot({s, true}, degree);
    if (measured_re[s] != want_re || measured_im[s] != want_im) {
      if (mismatches < 8) {
        std::cout << "  slot " << s << ": measured (re " << measured_re[s]
                  << ", im " << measured_im[s] << "), derived (re " << want_re
                  << ", im " << want_im << ")" << std::endl;
      }
      mismatches++;
    }
  }
  std::cout << "derived map vs measured map: " << mismatches << " of "
            << num_slots << " slots disagree" << std::endl;
  EXPECT_EQ(mismatches, 0)
      << "SlotOfCoeff does not describe what the hardware does; the printed "
         "pairs are the truth and the derivation is what has to change";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
