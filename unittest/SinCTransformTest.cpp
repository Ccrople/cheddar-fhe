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

    // One level per phase, and the inverse runs on the forward's output, so
    // they are compiled that far apart. Starting where the block's StC starts.
    const int stc_level = boot->GetBootParameter().GetStCStartLevel();
    const int cts_level = stc_level - c.phases;
    ASSERT_GE(cts_level - c.phases, 0);

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

    // ---- inverse: SinC -> slots ----------------------------------------
    Ciphertext<word> back;
    boot->SinCToSlot(back, num_slots, sinc, interface_->GetEvkMap());
    EXPECT_EQ(param_->NPToLevel(back.GetNP()), cts_level - c.phases);
    EXPECT_NEAR(back.GetScale() / sinc.GetScale(), 1.0, 1e-6);

    std::vector<Complex> round;
    DecryptAndDecode(round, back);
    const double round_err = MaxDiff(round, message);

    std::cout << "  sub_degree " << sub_degree << " (d = " << d << ", " << lanes
              << " lanes, " << p << " stages in " << c.phases
              << " phase(s)): forward " << forward_err
              << ", round trip " << round_err << ", control (no bit reversal) "
              << control_err << std::endl;

    EXPECT_LT(forward_err, 1e-4)
        << "the suffix of StC is not the SinC encoding at sub_degree "
        << sub_degree;
    EXPECT_LT(round_err, 1e-4)
        << "the prefix of CtS times 1/d is not its inverse at sub_degree "
        << sub_degree;
    if (p > 0) {
      EXPECT_GT(control_err, 1e-2)
          << "the block index is NOT bit-reversed, so this test would pass "
             "against a map that skipped it";
    }
  }
}
