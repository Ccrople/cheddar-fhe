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
// THE BUDGET IS PRINTED BEFORE ANYTHING RUNS. A LinearTransform encodes its
// diagonals at pt_scale = GetRescalePrimeProd(level) and the product it forms
// carries pt_scale * ct_scale, which has to fit Q_level. EvalSpecialFFT never
// has to think about that because it is pinned at the StC levels where the
// modulus is long; a standalone transform at a low level does, and a run that
// silently overflows comes back as values of magnitude 1e38. So the level is
// chosen from the budget rather than picked, and the budget is in the log.
//
// WHAT THIS TEST PINS. The permutation entrywise -- a norm check passes for any
// permutation at all -- the level and scale contract, the diagonal / stride /
// BSGS numbers the header claims, and that a wide field swap equals the two
// narrow ones it decomposes into, which is the form the pipeline will use
// because 2048 plaintexts do not fit where 256 + 128 do.

#include <cmath>
#include <string>
#include <vector>

#include "Testbed.h"
#include "common/CommonUtils.h"
#include "extension/SlotPermute.h"

using word = uint32_t;

namespace {

double MaxDiff(const std::vector<Complex> &got, const std::vector<Complex> &msg,
               const std::vector<int> &perm) {
  double m = 0.0;
  for (size_t s = 0; s < msg.size(); s++) {
    m = std::max(m, std::abs(got[perm[s]] - msg[s]));
  }
  return m;
}

// The square transpose of an n x n block, repeated across the slot vector.
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

std::vector<int> Compose(const std::vector<int> &second,
                         const std::vector<int> &first) {
  std::vector<int> out(first.size());
  for (size_t s = 0; s < first.size(); s++) out[s] = second[first[s]];
  return out;
}

}  // namespace

class SlotPermuteFixture : public Testbed32 {
 protected:
  int BootSlackLevels() const override { return 8; }

  double Log2Modulus(int level) const {
    double bits = 0.0;
    for (auto p : param_->GetPrimeVector(param_->LevelToNP(level))) {
      bits += std::log2(static_cast<double>(p));
    }
    return bits;
  }
};

INSTANTIATE_TEST_SUITE_P(Cheddar, SlotPermuteFixture,
                         testing::Values("bootparam_35.json"),
                         [](const testing::TestParamInfo<const char *> &info) {
                           std::string name = info.param;
                           return name.substr(0, name.find('.')) + "_json";
                         });

TEST_P(SlotPermuteFixture, PermutesTheSlotsExactly) {
  const int num_slots = param_->degree_ / 2;

  // The budget, and the level chosen from it.
  int level = -1;
  std::cout << "level  log2 Q   log2 scale  log2 rescale-prod  headroom"
            << std::endl;
  for (int L = 1; L <= 14; L++) {
    const double q = Log2Modulus(L);
    const double sc = std::log2(param_->GetScale(L));
    const double rp = std::log2(param_->GetRescalePrimeProd(L));
    const double head = q - sc - rp;
    std::cout << "  " << L << "     " << q << "    " << sc << "      " << rp
              << "        " << head << std::endl;
    if (level < 0 && head >= 20.0) level = L;
  }
  ASSERT_GE(level, 1) << "no level leaves room for a plaintext matrix";
  std::cout << "running at level " << level << std::endl;

  struct Case {
    std::string name;
    std::vector<int> perm;
  };
  std::vector<Case> cases;
  // Smallest possible: two adjacent bits, three diagonals. If this fails the
  // machinery is wrong, not the size.
  cases.push_back({"swap [1 | 1]", cheddar::SwapAdjacentFields(num_slots, 1, 1)});
  cases.push_back({"swap [4 | 4] at offset 3",
                   cheddar::SwapAdjacentFields(num_slots, 4, 4, 3)});
  cases.push_back({"swap [4 | 3]", cheddar::SwapAdjacentFields(num_slots, 3, 4)});
  cases.push_back({"square transpose 128x128", SquareTranspose(num_slots, 128)});

  std::vector<Complex> msg(num_slots);
  Random::SampleUniformComplex(msg.data(), num_slots, -1.0, 1.0);

  for (auto &c : cases) {
    cheddar::SlotPermute<word> perm(context_, c.perm, level);
    std::cout << "  " << c.name << ": " << perm.GetNumDiagonals()
              << " diagonals, stride " << perm.GetStride() << ", BSGS "
              << perm.GetBS() << "x" << perm.GetGS() << ", shift "
              << perm.GetShift() << std::endl;

    EvkRequest req;
    perm.AddRequiredRotations(req);
    interface_->PrepareRotationKey(req);

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
    const double err = MaxDiff(got, msg, c.perm);
    std::cout << "     max |diff| = " << err << std::endl;
    EXPECT_LT(err, 1e-4) << c.name << " did not land where it was asked to";
  }

  // THE DECOMPOSITION THE PIPELINE WILL ACTUALLY USE.
  //
  // [A(4) | B(7)] -> [B(7) | A(4)] is a 16 x 128 transpose: 2048 diagonals and
  // therefore 2048 plaintexts, which do not fit where 256 + 128 do. Splitting B
  // as B1(4) | B0(3) turns it into two adjacent swaps -- A with B1, then A with
  // B0 -- and this checks that the composition is the same permutation.
  const auto wide = cheddar::SwapAdjacentFields(num_slots, 7, 4);
  const auto step1 = cheddar::SwapAdjacentFields(num_slots, 4, 4, 3);
  const auto step2 = cheddar::SwapAdjacentFields(num_slots, 3, 4);
  const auto composed = Compose(step2, step1);
  int mismatch = 0;
  for (int s = 0; s < num_slots; s++) {
    if (composed[s] != wide[s]) mismatch++;
  }
  std::cout << "  [4|7] as [4|4]@3 then [4|3]: " << mismatch
            << " slots differ from the wide swap" << std::endl;
  EXPECT_EQ(mismatch, 0)
      << "the two narrow swaps are not the wide one, so the cheap form is not "
         "the same permutation";
}
