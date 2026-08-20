// Slots <-> SinC, homomorphically: the "partial bit-reversal operation which
// occurs when moving to SinC encoding" of [SYLPH] section 3.2.
//
// WHAT THIS IS FOR. The batch CC-MM ([KANG] Algorithm 4) reads its operands as
// subring matrix encryptions, whose entries are the `Vec^d_k` components of a
// SinC-encoded plaintext. Every use of it in this repo so far has encoded the
// operand on the *host* and encrypted it -- fine for a unit test of the
// product, useless for attention, where Q, K and V arrive as ciphertexts. The
// missing piece is a homomorphic conversion, and this is it.
//
// THE IDENTITY, AND WHY IT IS CHEAP. Writing n for the slot count, L = log2 n,
// d = degree / sub_degree and p = log2 d, the slot -> SinC(sub_degree) map is
// exactly the LAST p butterfly stages of SlotToCoeff, in SlotToCoeff's own
// ascending-stride order, with no scalar and with the message permuted by a
// bit reversal on the BLOCK index only. So it is one LinearTransform and one
// level, against the full SlotToCoeff's three. `EvalSpecialFFT.h` states it in
// full; here it is measured.
//
// The prefix of StC is a *different* map -- the low-stride stages act inside a
// lane block, which is the wrong axis -- so "run some of the stages" is not
// enough of a specification and this test would fail against it.
//
// WHAT WOULD SLIP THROUGH A ROUND TRIP. A round trip alone passes for any
// invertible map, including one that permutes lanes or drops the bit reversal.
// So the forward direction is checked ENTRYWISE against `DecodeSinC` -- the
// host encoder that `BatchCcmmTest` and `BatchCpmmTest` already build their
// operands with -- and against the exact permutation, not merely against
// "something the right size". The round trip is then a second, independent
// check that the inverse is the inverse.

#include <cmath>
#include <string>
#include <vector>

#include "Testbed.h"
#include "common/CommonUtils.h"

using word = uint32_t;

namespace {

// The configurations that matter, and how many levels each is allowed to
// spend. `sub_degree` is k in Ecd_SinC; the small ring's matrix is
// small_degree/k wide and there are k/2 lanes.
//
//   32   the attention product's own setting: 4096/32 = 128, which is exactly
//        Llama-3's per-head T x head_dim product, with 16 lanes for 16 heads.
//        p = 11 stages, so ONE phase would be 2048 plaintexts; three phases of
//        4 + 4 + 3 are 40, and three levels is what SlotToCoeff costs anyway.
//   512  d = 128 on the *big* ring, one phase, the first case measured here.
//   2048 d = 32, what [SYLPH] table 4 itself runs.
struct Case {
  int sub_degree;
  int phases;
};
constexpr Case kCases[] = {{32, 3}, {512, 1}, {2048, 1}};

int BitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

double MaxDiff(const std::vector<Complex> &a, const std::vector<Complex> &b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

}  // namespace

class SinCTransformFixture : public Testbed32 {
 protected:
  // [SYLPH]'s schedule, so StC starts where the block's does and the levels
  // quoted here are the levels the block would actually be converting at.
  int BootSlackLevels() const override { return 8; }
};

INSTANTIATE_TEST_SUITE_P(Cheddar, SinCTransformFixture,
                         testing::Values("bootparam_35.json"),
                         [](const testing::TestParamInfo<const char *> &info) {
                           std::string name = info.param;
                           return name.substr(0, name.find('.')) + "_json";
                         });

TEST_P(SinCTransformFixture, TheSuffixOfStCIsTheSinCEncoding) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int degree = param_->degree_;
  const int num_slots = degree / 2;

  boot->PrepareEvalSpecialFFT(num_slots);

  for (const Case &c : kCases) {
    const int sub_degree = c.sub_degree;
    const int d = degree / sub_degree;
    const int lanes = sub_degree / 2;
    const int p = cheddar::Log2Ceil(d);

    // WHERE EACH DIRECTION IS COMPILED, AND WHY IT IS NOT THE SAME PLACE.
    //
    // The forward starts where the block's SlotToCoeff starts, because in the
    // pipeline it *replaces* SlotToCoeff: a tensor bound for the product pays
    // this instead of that, not on top of it.
    //
    // The inverse does not sit under the forward's output. In the pipeline it
    // runs on the far side of a bootstrap -- the product returns at level 0 and
    // nothing can follow it there -- so it is compiled high, at the same levels.
    // That is also the only place it *works*: chained straight onto the
    // forward's output the last phase lands at level 6, and a standalone
    // LinearTransform below level 7 does not merely lose accuracy, it returns
    // magnitudes near the modulus. `SlotPermuteTest` found the same floor from
    // the other side. The round trip below therefore runs only when the two
    // agree on a level, and the inverse is checked entrywise against the host
    // encoder either way, which is the stronger statement in any case.
    const int stc_level = boot->GetBootParameter().GetStCStartLevel();
    const int cts_level = stc_level;
    ASSERT_GE(cts_level - c.phases, 1);

    boot->PrepareSinC(num_slots, sub_degree, stc_level, cts_level, c.phases);
    ASSERT_EQ(boot->GetSinCNumPhases(num_slots), c.phases);
    EvkRequest req;
    boot->AddRequiredSinCRotations(req, num_slots);
    interface_->PrepareRotationKey(req);

    std::vector<Complex> message(num_slots);
    Random::SampleUniformComplex(message.data(), num_slots, -1.0, 1.0);

    Ciphertext<word> ct;
    EncodeAndEncrypt(ct, message, stc_level);

    // ---- forward: slots -> SinC ----------------------------------------
    Ciphertext<word> sinc;
    boot->SlotToSinC(sinc, num_slots, ct, interface_->GetEvkMap());
    EXPECT_EQ(param_->NPToLevel(sinc.GetNP()), stc_level - c.phases)
        << "the conversion is one LinearTransform per phase and must cost "
           "exactly that many levels";
    EXPECT_NEAR(sinc.GetScale() / ct.GetScale(), 1.0, 1e-6)
        << "and it must leave the scale where it found it";

    Plaintext<word> pt;
    interface_->Decrypt(pt, sinc);
    std::vector<Complex> got;
    context_->encoder_.DecodeSinC(got, pt, sub_degree);
    ASSERT_EQ(static_cast<int>(got.size()), num_slots);

    // The permutation, entrywise. Block index bit-reversed, lanes untouched.
    std::vector<Complex> want(num_slots);
    for (int i = 0; i < d; i++) {
      for (int r = 0; r < lanes; r++) {
        want[i * lanes + r] = message[BitRev(i, p) * lanes + r];
      }
    }
    const double forward_err = MaxDiff(got, want);

    // A control: the same comparison without the bit reversal. It has to
    // fail, or the test is not testing the permutation at all.
    std::vector<Complex> unreversed(num_slots);
    for (int i = 0; i < d; i++) {
      for (int r = 0; r < lanes; r++) {
        unreversed[i * lanes + r] = message[i * lanes + r];
      }
    }
    const double control_err = MaxDiff(got, unreversed);

    // ---- inverse: SinC -> slots, against the host encoder ---------------
    //
    // A round trip alone passes for any invertible map. This encodes `want`
    // -- the permuted message, as `Encoder::EncodeSinC` writes it -- and
    // requires SinCToSlot to hand back `message` in slots, entrywise. So the
    // inverse is pinned to the same permutation the forward is, independently.
    Plaintext<word> inv_pt;
    context_->encoder_.EncodeSinC(inv_pt, cts_level,
                                  param_->GetScale(cts_level), want,
                                  sub_degree);
    Ciphertext<word> inv_ct;
    interface_->Encrypt(inv_ct, inv_pt);

    Ciphertext<word> back;
    boot->SinCToSlot(back, num_slots, inv_ct, interface_->GetEvkMap());
    EXPECT_EQ(param_->NPToLevel(back.GetNP()), cts_level - c.phases);
    EXPECT_NEAR(back.GetScale() / inv_ct.GetScale(), 1.0, 1e-6);

    std::vector<Complex> undone;
    DecryptAndDecode(undone, back);
    const double inverse_err = MaxDiff(undone, message);

    // And the round trip itself, when the forward leaves its output at the
    // level the inverse was compiled at. It does when there is one phase; with
    // three the forward lands three levels lower, and the note above is why
    // that is not a defect to be chased.
    double round_err = -1.0;
    if (param_->NPToLevel(sinc.GetNP()) == cts_level) {
      Ciphertext<word> chained;
      boot->SinCToSlot(chained, num_slots, sinc, interface_->GetEvkMap());
      std::vector<Complex> round;
      DecryptAndDecode(round, chained);
      round_err = MaxDiff(round, message);
    }

    std::cout << "  sub_degree " << sub_degree << " (d = " << d << ", " << lanes
              << " lanes, " << p << " stages in " << c.phases
              << " phase(s)): forward " << forward_err << ", inverse "
              << inverse_err << ", round trip " << round_err
              << ", control (no bit reversal) " << control_err << std::endl;

    EXPECT_LT(forward_err, 1e-4)
        << "the suffix of StC is not the SinC encoding at sub_degree "
        << sub_degree;
    EXPECT_LT(inverse_err, 1e-4)
        << "the prefix of CtS times 1/d does not undo it at sub_degree "
        << sub_degree;
    if (round_err >= 0.0) {
      EXPECT_LT(round_err, 1e-4)
          << "the two do not compose to the identity at sub_degree "
          << sub_degree;
    }
    if (p > 0) {
      EXPECT_GT(control_err, 1e-2)
          << "the block index is NOT bit-reversed, so this test would pass "
             "against a map that skipped it";
    }
  }
}

// ---------------------------------------------------------------------------
// The other half of the trip: what HalfBoot leaves undone.
// ---------------------------------------------------------------------------
//
// The test above converts slots -> SinC and back inside ONE ring at ONE level,
// which is what a unit test of the identity needs and is not what the pipeline
// does. The pipeline's return trip is forced: the batch CC-MM leaves its result
// at level 0, and the only route out of level 0 is a bootstrap.
//
// `HalfBoot` inverts the WHOLE of SlotToCoeff. A SinC ciphertext is
// `StC(P^-1(s))` with `P` the prefix StC applies before the SinC suffix, so
// what lands in slots is `P^-1(s)` -- and `P` is butterfly stages, not a
// permutation, so `P^-1(s)` is a twiddle-weighted MIXTURE of the values. No
// relabelling of the layout can absorb that; `SinCPrefix` applies `P` and
// finishes the trip.
//
// WHY THIS NEEDS NO HOST ENCODER AND NO GUESS ABOUT THE BIT REVERSAL. The
// reversal cancels. HalfBootTest already shows StC and HalfBoot are inverse up
// to a constant, and StC = SlotToSinC . P, so
//
//     SinCPrefix(HalfBoot(SlotToSinC(s)))  =  const . s
//
// end to end. The constant is real: StC bakes in stc_const_ and CtS bakes in
// cts_const_, and outside a full Boot they do not cancel -- so the test fits
// the best constant first and then asks whether what is left is the input,
// exactly as HalfBootTest does.
//
// THE CONTROL is the same quantity without the prefix. It has the right
// magnitude, which is why a norm-only test would pass against it.
TEST_P(SinCTransformFixture, TheStCPrefixIsWhatHalfBootLeavesUndone) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int degree = param_->degree_;
  const int num_slots = degree / 2;
  // The attention product's own setting, so the prefix here is the 4 stages
  // the score path will actually pay for.
  constexpr int kSubDegree = 32;
  constexpr int kPhases = 3;

  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  const int stc_level = boot->GetBootParameter().GetStCStartLevel();
  // Where HalfBoot lands, which is where the prefix has to be compiled -- and
  // in a [SYLPH] schedule it is also where Canonicalise's multiply sits, so
  // the level the prefix spends is one the cycle was spending anyway.
  const int landing = boot->GetBootParameter().GetEvalModEndLevel();
  std::cout << "StC starts at " << stc_level << ", HalfBoot lands at "
            << landing << std::endl;

  boot->PrepareSinC(num_slots, kSubDegree, stc_level, stc_level, kPhases);
  boot->PrepareSinCPrefix(num_slots, kSubDegree, landing);
  ASSERT_EQ(boot->GetSinCPrefixNumPhases(num_slots), 1)
      << "4 stages is 31 diagonals, which is one transform";

  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  boot->AddRequiredSinCRotations(req, num_slots);
  boot->AddRequiredSinCPrefixRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> message(num_slots);
  Random::SampleUniformComplex(message.data(), num_slots, -1.0, 1.0);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, message, stc_level);

  Ciphertext<word> sinc;
  boot->SlotToSinC(sinc, num_slots, ct, interface_->GetEvkMap());
  ASSERT_EQ(param_->NPToLevel(sinc.GetNP()), stc_level - kPhases);

  // HalfBoot brings its input to level 0 itself, which is where the product
  // would have left it.
  Ciphertext<word> half;
  boot->HalfBoot(half, sinc, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(param_->NPToLevel(half.GetNP()), landing)
      << "the prefix was compiled where HalfBoot was expected to land, and it "
         "landed somewhere else";

  Ciphertext<word> back;
  boot->SinCPrefix(back, num_slots, half, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(param_->NPToLevel(back.GetNP()), landing - 1)
      << "the prefix is one LinearTransform and must cost exactly one level";

  // Fit the best constant, then ask whether what is left is the input. The
  // constant is the uncancelled stc_const_ / cts_const_ pair, not a bug.
  auto residual = [&](const Ciphertext<word> &c, const char *name) {
    std::vector<Complex> got;
    DecryptAndDecode(got, c);
    Complex sum(0.0, 0.0);
    int counted = 0;
    for (int i = 0; i < num_slots; i++) {
      if (std::abs(message[i]) < 0.3) continue;
      sum += got[i] / message[i];
      counted++;
    }
    const Complex ratio = sum / static_cast<double>(counted);
    double worst = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max(worst, std::abs(got[i] / ratio - message[i]));
    }
    std::cout << "  " << name << ": best constant " << ratio << " over "
              << counted << " slots, residual " << worst << " ("
              << -std::log2(worst) << " bits)" << std::endl;
    return worst;
  };

  const double with_prefix = residual(back, "HalfBoot then the prefix");
  const double without = residual(half, "HalfBoot alone (control)");

  EXPECT_LT(with_prefix, 1e-2)
      << "the prefix of SlotToCoeff is not what HalfBoot leaves undone";
  EXPECT_GT(without, 1e-1)
      << "HalfBoot alone already returns the slot vector, so this test is not "
         "checking that the prefix does anything";
}
