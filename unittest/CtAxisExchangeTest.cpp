// Exchanging a bit field of the slot index with the index of an array of
// ciphertexts -- the one move in the attention path that a LinearTransform
// cannot make, because it crosses between ciphertexts.
//
// WHY IT IS NEEDED. SwitchedCcmmLayout puts entry (row, column, lane) in big
// ciphertext `column / rank`, so three bits of the column index sit on the
// ciphertext axis. For Q the column is the channel and the projection's weight
// columns place it for free. For K the column is the KEY TOKEN, which the
// block delivers in the low slot bits and which no weight matrix can move. So
// three slot bits have to become ciphertext bits.
//
// WHAT IS ACTUALLY CHECKED. Entrywise, against a host reference, with a
// control that reads the result where it would sit had the two axes NOT traded
// places -- because the exchange preserves every value and only moves it, so a
// no-op has exactly the right magnitude. The control is three decades above
// the tolerance, which is what makes this a test of the layout rather than of
// the noise.
//
// TWO CASES, ON TWO RINGS, AND THE SECOND IS THE ONE THE MODEL RUNS.
//
//  1. ringswitch16_35 at its only level, a square exchange at a nonzero field
//     offset. It pins the offset arithmetic on its own and it is the cheapest
//     ring in the ladder. Its accuracy is the floor of a big-ring key switch
//     at the bottom of a grafted ladder -- about 1e-04, which is simply
//     `noise / 2^29.877`, the scale that level holds. Rescaling does not
//     improve it: the mask multiply and its rescale move the message and the
//     noise by the same factor.
//  2. sylphflow16_35 at level 11, the exact shape the score product's right
//     operand needs -- 2 ciphertexts to 8, a three-bit field at offset 0,
//     GQA's fourfold replication carried in the low two bits. This is where
//     the attention path actually compiles it, and the scale there is 2^35.
//
// The exchange spends one level, whichever ring it is on.

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/CtAxisExchange.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::CtAxisExchange;
using cheddar::Plaintext;
using Ring = ringfixture::Ring<word>;

namespace {

const char *Env(const char *name, const char *fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? e : fallback;
}

int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? std::atoi(e) : fallback;
}

// A layout mistake lands at the control's magnitude, which is the size of the
// data itself. This separates the two by three decades and prints what was
// actually measured, so a precision regression is visible without pinning the
// test to a particular ring's noise floor.
constexpr double kTolerance = 1e-3;

// `level < 0` means the ring's top level.
void RunCase(const std::string &file, int level_in, int log_cts,
             int field_offset, int log_src) {
  Ring ring(file);
  const int num_slots = ring.Degree() / 2;
  const int level = (level_in < 0) ? ring.param->max_level_ : level_in;
  ASSERT_GE(level, 1);
  ASSERT_LE(level, ring.param->max_level_);
  const double scale = ring.param->GetScale(level);

  const int num_cts = 1 << log_cts;
  const int num_src = 1 << log_src;
  const int step = 1 << field_offset;
  const int replication = log_cts - log_src;

  CtAxisExchange<word> exchange(ring.context, log_cts, field_offset, level,
                                log_src);
  ASSERT_EQ(exchange.GetNumCts(), num_cts);
  ASSERT_EQ(exchange.GetNumSrcCts(), num_src);
  const std::vector<int> rot = exchange.RotationIndices();
  ASSERT_EQ(static_cast<int>(rot.size()), 2 * (num_cts - 1));
  for (int idx : rot) ring.ui->PrepareRotationKey(idx, level);

  // The host tensor, one value per (source ciphertext, slot).
  std::mt19937_64 gen(0xC7A1E ^ (log_cts << 8) ^ (field_offset << 4) ^ log_src);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<std::vector<Complex>> host(
      num_src, std::vector<Complex>(num_slots, Complex(0)));
  for (auto &v : host) {
    for (auto &z : v) z = Complex(dist(gen), 0.0);
  }

  std::vector<Ciphertext<word>> src(num_src);
  for (int i = 0; i < num_src; i++) {
    Plaintext<word> pt;
    ring.context->encoder_.Encode(pt, level, scale, host[i]);
    ring.ui->Encrypt(src[i], pt);
  }

  std::vector<Ciphertext<word>> res;
  exchange.Evaluate(res, src, ring.ui->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), num_cts);

  std::vector<std::vector<Complex>> got(num_cts);
  for (int y = 0; y < num_cts; y++) {
    EXPECT_EQ(ring.param->NPToLevel(res[y].GetNP()), level - 1)
        << "the exchange spends exactly one level";
    EXPECT_NEAR(res[y].GetScale() / ring.param->GetScale(level - 1), 1.0, 1e-6)
        << "the masks were not encoded at the level's own scale, so a "
           "canonical input did not come back canonical";
    Plaintext<word> pt;
    ring.ui->Decrypt(pt, res[y]);
    ring.context->encoder_.Decode(got[y], pt);
  }

  double worst = 0.0, control = 0.0;
  for (int y = 0; y < num_cts; y++) {
    for (int p = 0; p < num_slots; p++) {
      const int x = (p >> field_offset) & (num_cts - 1);
      const int q = p + (y - x) * step;  // the same slot with field := y
      const double want = host[x >> replication][q].real();
      worst = std::max(worst, std::abs(got[y][p].real() - want));
      // The control: what would be there if nothing had moved. It is a real
      // control only where the two axes actually disagree.
      if (x != y) {
        control =
            std::max(control, std::abs(host[y >> replication][p].real() - want));
      }
    }
  }
  std::cout << file << " level " << level << ": exchange w=" << log_cts
            << " offset=" << field_offset << " " << num_src << " -> "
            << num_cts << " cts: max |diff| = " << worst
            << ", control (nothing moved) " << control << ", " << rot.size()
            << " rotations, " << exchange.GetNumMults() << " plaintext mults"
            << std::endl;

  EXPECT_LT(worst, kTolerance)
      << "the exchange did not put the values where its contract says";
  EXPECT_GT(control, 1e-2)
      << "the source is flat enough that no exchange would pass this";
}

}  // namespace

// The square case at a nonzero offset: the field is slot bits 5..7, so this
// fails if the step arithmetic ignores the offset anywhere. On the smallest
// ring in the ladder, at the only level it has.
TEST(CtAxisExchange, TradesASlotFieldForTheCiphertextIndex) {
  RunCase(Env("CHEDDAR_SWITCH_PARAM", "ringswitch16_35.json"), /*level=*/-1,
          /*log_cts=*/3, /*field_offset=*/5, /*log_src=*/3);
}

// The shape the score product's right operand needs, on the ring and at the
// level the attention path compiles it: 2 ciphertexts to 8, GQA's fourfold
// replication in the low two bits of the exchanged field.
TEST(CtAxisExchange, ReplicatesAShortSourceAcrossTheField) {
  RunCase(Env("CHEDDAR_BLOCK_PARAM", "sylphflow16_35.json"),
          EnvInt("CHEDDAR_EXCHANGE_LEVEL", 11), /*log_cts=*/3,
          /*field_offset=*/0, /*log_src=*/1);
}
