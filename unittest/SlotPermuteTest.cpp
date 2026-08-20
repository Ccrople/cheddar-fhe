// An arbitrary slot permutation as one LinearTransform -- the transpose
// [SYLPH] section 3.2 says survives its layout choice ("the only survivors are
// transpose and tau^2, done with Halevi-Shoup + BSGS").
//
// WHY THE PIPELINE NEEDS IT. The Bae PC-MM ladder writes coefficient
// p = j + 16c + 256s, so the token is the coordinate of the smallest ring and
// owns the high coefficient bits; after the bootstrap's bit reversal it owns
// the LOW slot bits. The batch CC-MM reads Vec^d_k, whose block index is
// p mod d and whose lane index is p / d, so it wants the token in the MIDDLE:
// [lane | row | column]. No assignment of the channel bits fixes that, because
// the token's position is not an assignment. Something has to move it.
//
// THE TWO SHAPES THAT MATTER, and they cost very different amounts:
//
//   a square n x n transpose        2n-1 diagonals, ALL multiples of n-1
//   swapping an a-bit and a b-bit   2^(a+b) diagonals at stride 1
//   adjacent field
//
// The first is cheap because the common stride does the work of the window.
// The second is only affordable with a pre-rotation, because its raw offsets
// straddle zero and mod the slot count they then span nearly the whole ring.
//
// WHAT THIS TEST PINS. The permutation entrywise -- a norm check passes for
// any permutation at all -- plus the level and scale contract, plus the
// diagonal/stride/BSGS numbers the header claims. It also prints the error
// against the answer shifted by +-shift, so a convention mismatch in
// LinearTransform's window handling reports itself as a number rather than as
// a mysterious failure.

#include <cmath>
#include <string>
#include <vector>

#include "Testbed.h"
#include "common/CommonUtils.h"
#include "extension/SlotPermute.h"

using word = uint32_t;

namespace {

double MaxDiff(const std::vector<Complex> &got, const std::vector<Complex> &msg,
               const std::vector<int> &perm, int shift) {
  const int n = static_cast<int>(msg.size());
  double m = 0.0;
  for (int s = 0; s < n; s++) {
    int dst = perm[s] + shift;
    dst %= n;
    if (dst < 0) dst += n;
    m = std::max(m, std::abs(got[dst] - msg[s]));
  }
  return m;
}

// The square transpose of a 128 x 128 block, repeated across the slot vector.
std::vector<int> SquareTranspose(int num_slots, int n) {
  const int window = n * n;
  std::vector<int> perm(num_slots);
  for (int s = 0; s < num_slots; s++) {
    const int rest = s / window;
    const int in = s % window;
    perm[s] = rest * window + (in % n) * n + (in / n);
  }
  return perm;
}

}  // namespace

class SlotPermuteFixture : public Testbed32 {
 protected:
  int BootSlackLevels() const override { return 8; }
};

INSTANTIATE_TEST_SUITE_P(Cheddar, SlotPermuteFixture,
                         testing::Values("bootparam_35.json"),
                         [](const testing::TestParamInfo<const char *> &info) {
                           std::string name = info.param;
                           return name.substr(0, name.find('.')) + "_json";
                         });

TEST_P(SlotPermuteFixture, PermutesTheSlotsExactly) {
  const int num_slots = param_->degree_ / 2;
  // Low enough that the plaintexts are small -- 2048 diagonals is 2048
  // plaintexts, and they scale with the limb count -- and high enough to leave
  // the output somewhere usable.
  const int level = 4;

  struct Case {
    const char *name;
    std::vector<int> perm;
  };
  std::vector<Case> cases;
  cases.push_back({"square transpose 128x128", SquareTranspose(num_slots, 128)});
  // [channel nibble | token] -> [token | channel nibble], which is the move
  // the CC-MM's operands need.
  cases.push_back({"field swap [4 | 7] -> [7 | 4]",
                   cheddar::SwapAdjacentFields(num_slots, 7, 4)});

  for (auto &c : cases) {
    cheddar::SlotPermute<word> perm(context_, c.perm, level);
    std::cout << "  " << c.name << ": " << perm.GetNumDiagonals()
              << " diagonals, stride " << perm.GetStride() << ", BSGS "
              << perm.GetBS() << "x" << perm.GetGS() << ", shift "
              << perm.GetShift() << std::endl;

    EvkRequest req;
    perm.AddRequiredRotations(req);
    interface_->PrepareRotationKey(req);

    std::vector<Complex> msg(num_slots);
    Random::SampleUniformComplex(msg.data(), num_slots, -1.0, 1.0);
    Ciphertext<word> ct;
    EncodeAndEncrypt(ct, msg, level);

    Ciphertext<word> out;
    perm.Evaluate(context_, out, ct, interface_->GetEvkMap());
    EXPECT_EQ(param_->NPToLevel(out.GetNP()), level - 1)
        << "one LinearTransform, one level";
    EXPECT_NEAR(out.GetScale() / ct.GetScale(), 1.0, 1e-6)
        << "and the scale is preserved";

    std::vector<Complex> got;
    DecryptAndDecode(got, out);

    const double exact = MaxDiff(got, msg, c.perm, 0);
    const double plus = MaxDiff(got, msg, c.perm, perm.GetShift());
    const double minus = MaxDiff(got, msg, c.perm, -perm.GetShift());
    std::cout << "     max |diff|: exact " << exact << ", shifted +s "
              << plus << ", shifted -s " << minus << std::endl;

    EXPECT_LT(exact, 1e-4)
        << c.name << " did not land where it was asked to (shifted +s "
        << plus << ", -s " << minus << ")";
  }
}
