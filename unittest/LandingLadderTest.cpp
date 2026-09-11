// ONE PARAMETER SET, ANY LANDING, THE SHORT CLIMB -- measured.
//
// `extension/LandingLadder` cuts, at run time and from ONE preset, the ladder
// whose bootstrap climbs only as far as a chosen landing needs: the pool's
// levels 0..L verbatim, its stationary 2^60 band pairs on top, CoeffToSlot
// from what is left. This file establishes three things about it:
//
//   (1) THE RULE. For every landing of the three ci16_42 pools the cut is a
//       valid Parameter, its band is the band (stationary to the second
//       decimal), its CtS levels are transform levels, and -- the crossing's
//       whole condition -- the primes at every level up to the landing are the
//       pool's primes in the pool's order. The pool's own landing comes back
//       byte for byte. Every ladder is dumped (CHEDDAR_LANDING_DUMP_DIR) for
//       `param_audit.py --strict` and for the diff against the host mirror
//       `landing_ladder.py`, which is a second implementation of the rule.
//   (2) THE CROSSING. A ciphertext ENCRYPTED ON THE POOL goes through the
//       landing ring's bootstrap, lands at L, and is decrypted BY THE POOL,
//       which then multiplies and rescales it as its own; and the other way
//       round. No key moves and no word moves: it is the same ring below L.
//   (3) THE COST. The same landing reached by slack on the pool (the full
//       climb, StC lowered) and by the ladder (the short climb): ms per Boot,
//       synced, on the same box in the same process -- and the ladder's
//       precision, which must be the pool's.
//
// CHEDDAR_LADDER_POOL (default ci16_42_k16_w60.json), CHEDDAR_LADDER_LANDING
// (default 5), CHEDDAR_BOOT_LANDING_JUNCTION=1 fills junctions (see the
// header) instead of taking slack.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "LadderJson.h"
#include "RingFixture.h"
#include "extension/LandingLadder.h"

namespace {

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::JunctionPolicy;
using cheddar::LadderSpec;
using cheddar::LandingLadder;
using cheddar::Plaintext;

const char *kPools[] = {"ci16_42_k16_w60.json", "ci16_42_k32_w60.json",
                        "ci16_42_k64_w60.json"};

const char *PoolParam() {
  const char *e = std::getenv("CHEDDAR_LADDER_POOL");
  return (e && e[0]) ? e : "ci16_42_k16_w60.json";
}
int LandingLevel() {
  const char *e = std::getenv("CHEDDAR_LADDER_LANDING");
  return (e && e[0]) ? std::atoi(e) : 5;
}
JunctionPolicy Policy() {
  const char *e = std::getenv("CHEDDAR_BOOT_LANDING_JUNCTION");
  return (e && e[0] == '1') ? JunctionPolicy::kFillJunction
                            : JunctionPolicy::kStationaryOnly;
}
std::string Stem(const std::string &file) {
  return file.substr(0, file.rfind(".json"));
}
LadderSpec<word> Load(const char *file) {
  return ladderjson::Parse<word>(std::string(PARAM_DIR) + "/" + file);
}

std::vector<Complex> RandomReal(int num_slots, double amp, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> d(-amp, amp);
  std::vector<Complex> m(num_slots);
  for (auto &c : m) c = Complex(d(gen), 0.0);
  return m;
}

void EncryptAt(const Ring &r, Ciphertext<word> &ct,
               const std::vector<Complex> &msg, int level) {
  Plaintext<word> pt;
  r.context->encoder_.Encode(pt, level, r.param->GetScale(level), msg, 0);
  r.ui->Encrypt(ct, pt);
  ct.SetNumSlots(r.param->MaxNumSlots());
}

std::vector<Complex> Decrypt(const Ring &r, const Ciphertext<word> &ct) {
  Plaintext<word> pt;
  r.ui->Decrypt(pt, ct);
  std::vector<Complex> out;
  r.context->encoder_.Decode(out, pt);
  return out;
}

// The scalar the output carries, fitted, and the residual against it: a
// correct bootstrap is a clean scalar multiple of its input.
struct Fit {
  double carried, residual;
};
Fit FitResidual(const std::vector<Complex> &input,
                const std::vector<Complex> &got) {
  double num = 0.0, den = 0.0;
  const size_t n = std::min(input.size(), got.size());
  for (size_t i = 0; i < n; i++) {
    num += got[i].real() * input[i].real();
    den += input[i].real() * input[i].real();
  }
  const double c = (den > 0) ? num / den : 0.0;
  double res = 0.0;
  for (size_t i = 0; i < n; i++)
    res = std::max(res, std::abs(got[i] - c * input[i]));
  return {c, res};
}

std::shared_ptr<BootContext<word>> PrepareBoot(const Ring &r) {
  auto b = std::dynamic_pointer_cast<BootContext<word>>(r.context);
  EXPECT_NE(b, nullptr) << "ring is not a BootContext -- is boot:true set?";
  const int num_slots = r.param->MaxNumSlots();
  b->PrepareEvalMod();
  b->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  b->AddRequiredRotations(req, num_slots, false);
  r.ui->PrepareRotationKey(req);
  return b;
}

// ms per Boot, the device drained on both sides of every repetition.
double TimeBoots(const std::shared_ptr<BootContext<word>> &b, const Ring &r,
                 const Ciphertext<word> &ct, Ciphertext<word> &res, int reps) {
  for (int i = 0; i < 3; i++) b->Boot(res, ct, r.ui->GetEvkMap());
  cudaDeviceSynchronize();
  std::vector<double> t;
  for (int i = 0; i < reps; i++) {
    cudaDeviceSynchronize();
    const auto s = std::chrono::steady_clock::now();
    b->Boot(res, ct, r.ui->GetEvkMap());
    cudaDeviceSynchronize();
    t.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - s)
                    .count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

}  // namespace

// (1) THE RULE, on every landing of every pool.
TEST(LandingLadder, EveryLandingCutsFromThePoolAndKeepsItsPrefix) {
  const char *dump = std::getenv("CHEDDAR_LANDING_DUMP_DIR");
  for (const char *file : kPools) {
    const auto pool = Load(file);
    const LandingLadder<word> lad(pool);
    const auto pool_param = pool.BuildParameter();
    const int ns = pool.num_stc_levels;
    std::cout << "== " << file << ": lands at " << lad.PoolLanding()
              << ", band " << lad.NumEvalModLevels() << " x 2^"
              << lad.BandWidthBits() << ", clean landings";
    for (int l : lad.CleanLandings()) std::cout << " " << l;
    std::cout << std::endl;

    // The pool is its own ladder.
    {
      const auto id = lad.ForLanding(lad.PoolLanding());
      EXPECT_EQ(id.slack, 0);
      EXPECT_EQ(id.spec.level_config, pool.level_config);
      EXPECT_EQ(id.spec.main_primes, pool.main_primes);
      EXPECT_EQ(id.spec.ter_primes, pool.ter_primes);
      EXPECT_EQ(id.spec.default_encryption_level,
                pool.default_encryption_level);
    }

    for (int L = 0; L <= lad.PoolLanding(); L++) {
      for (int fill = 0; fill <= (lad.IsJunction(L) ? 1 : 0); fill++) {
        const auto r = lad.ForLanding(
            L, fill ? JunctionPolicy::kFillJunction
                    : JunctionPolicy::kStationaryOnly);
        std::cout << "  " << LandingLadder<word>::Describe(r) << std::endl;
        // Slack lowers the landing: the ladder lands at Landing(), StC is
        // compiled `slack` levels lower, and the boot ends at L.
        EXPECT_EQ(r.spec.Landing() - r.slack, L);
        EXPECT_EQ(r.landing, L);
        if (!fill && lad.IsClean(L)) EXPECT_EQ(r.slack, 0);
        const auto &s = r.spec;
        const auto p = s.BuildParameter();  // Parameter.cu's own asserts

        // THE CROSSING'S CONDITION, stated on the Parameter: the primes at
        // every level up to the landing are the pool's, in the pool's order.
        for (int l = 0; l <= L; l++) {
          EXPECT_EQ(p->LevelToNP(l), pool_param->LevelToNP(l)) << "level " << l;
          EXPECT_EQ(p->GetPrimeVector(p->LevelToNP(l)),
                    pool_param->GetPrimeVector(pool_param->LevelToNP(l)))
              << "level " << l;
          EXPECT_DOUBLE_EQ(p->GetScale(l), pool_param->GetScale(l))
              << "level " << l;
        }
        // The ladder's own StC levels are the pool's graft steps too.
        for (int l = L + 1; l <= r.ladder_landing + ns; l++) {
          EXPECT_EQ(s.level_config[l], pool.level_config[l]) << "level " << l;
        }
        // The band is the band.
        const int dec = s.default_encryption_level;
        for (int k = 1; k <= lad.NumEvalModLevels(); k++) {
          EXPECT_NEAR(s.RescaleBits(dec + k), lad.BandWidthBits(),
                      fill ? 0.15 : 0.05)
              << "band level " << dec + k;
        }
        EXPECT_LE(r.band_deviation_bits, fill ? 0.15 : 0.05);
        // CoeffToSlot's levels carry a transform each (EvalSpecialFFT's
        // thin rule is < 2^30; param_audit's starved band ends at 2^49).
        for (int l = dec + lad.NumEvalModLevels() + 1; l <= s.MaxLevel(); l++) {
          EXPECT_GE(s.RescaleBits(l), 49.0) << "CtS level " << l;
        }
        EXPECT_EQ(s.MaxLevel(), dec + lad.NumEvalModLevels() + s.num_cts_levels);
        // Every declared prime is used (Parameter.cu's last-level assert
        // holds because BuildParameter passed) and none is duplicated.
        {
          std::vector<word> all(s.main_primes);
          all.insert(all.end(), s.ter_primes.begin(), s.ter_primes.end());
          all.insert(all.end(), s.aux_primes.begin(), s.aux_primes.end());
          std::sort(all.begin(), all.end());
          EXPECT_EQ(std::adjacent_find(all.begin(), all.end()), all.end())
              << "a prime is declared twice";
        }
        if (dump != nullptr && dump[0] != 0 && (r.slack == 0 || fill)) {
          const std::string out = std::string(dump) + "/" + Stem(file) +
                                  "_land" + std::to_string(L) +
                                  (fill ? "fill" : "") + ".json";
          std::ofstream f(out);
          f << s.ToJson();
        }
      }
    }
  }
}

// (2) THE CROSSING.
TEST(LandingLadder, ACiphertextCrossesBetweenThePoolAndTheLandingRing) {
  Ring pool(PoolParam());
  const LandingLadder<word> lad(Load(PoolParam()));
  const int L = LandingLevel();
  const auto r = lad.ForLanding(L, Policy());
  std::cout << "[ladder] " << PoolParam() << ": "
            << LandingLadder<word>::Describe(r) << std::endl;
  Ring land(r.spec, pool.ui->GetSecretCoeffs(), r.slack);
  auto b = PrepareBoot(land);
  ASSERT_EQ(b->GetBootParameter().GetEndLevel(), L);
  const int num_slots = pool.param->MaxNumSlots();
  const auto msg = RandomReal(num_slots, 1.0, 0xA11CE);

  // Encrypted on the pool, bootstrapped by the ladder, read by the pool.
  Ciphertext<word> ct, res;
  EncryptAt(pool, ct, msg, 0);
  b->Boot(res, ct, land.ui->GetEvkMap());
  EXPECT_EQ(land.param->NPToLevel(res.GetNP()), L);
  EXPECT_EQ(pool.param->NPToLevel(res.GetNP()), L)
      << "the landing's NP is not a level of the pool";
  // With slack, BootBack folds the LevelDown's own drift into the scale it
  // declares (the message is untouched and the scale field tracks it), so the
  // landing is canonical only at slack 0; the drift is printed, not judged.
  if (r.slack == 0) {
    EXPECT_DOUBLE_EQ(res.GetScale(), pool.param->GetScale(L))
        << "the ladder's canonical scale at L is not the pool's";
  } else {
    std::cout << "[crossing] declared scale off canonical by 2^"
              << std::log2(res.GetScale() / pool.param->GetScale(L))
              << " (slack " << r.slack << ", BootBack's folded drift)"
              << std::endl;
  }
  const Fit f = FitResidual(msg, Decrypt(pool, res));
  std::cout << "[crossing] pool -> ladder Boot -> pool: carried " << f.carried
            << ", max abs err " << f.residual << " = p " << -std::log2(f.residual)
            << " bits at level " << L << std::endl;
  EXPECT_NEAR(f.carried, 1.0, 0.02) << "Boot should be message preserving";
  EXPECT_LT(f.residual, 0.01) << "the message did not cross";
  {
    const Fit g = FitResidual(msg, Decrypt(land, res));
    EXPECT_NEAR(g.residual, f.residual, 1e-9)
        << "the two rings decrypt the same words differently";
  }

  // The pool computes on it as its own: a plaintext multiply and a rescale.
  if (L >= 1) {
    const auto w = RandomReal(num_slots, 1.0, 0xBEEF);
    Plaintext<word> pt;
    pool.context->encoder_.Encode(pt, L, pool.param->GetScale(L), w, 0);
    Ciphertext<word> prod, down;
    pool.context->Mult(prod, res, pt);
    pool.context->Rescale(down, prod);
    EXPECT_EQ(pool.param->NPToLevel(down.GetNP()), L - 1);
    const auto got = Decrypt(pool, down);
    double worst = 0.0;
    for (int i = 0; i < num_slots; i++) {
      worst = std::max(worst,
                       std::abs(got[i].real() - msg[i].real() * w[i].real()));
    }
    std::cout << "[crossing] then Mult + Rescale on the pool: max abs err "
              << worst << std::endl;
    EXPECT_LT(worst, 0.01) << "the pool could not compute on the landing";
  }

  // HalfBoot lands on the ladder's encryption level, which is a pool level:
  // both rings read the same words to the same values there. Its slots hold
  // the input's COEFFICIENTS bit-reversed (ParamRobustTest), so the message
  // is only checked after the SlotToCoeff cycle -- here read by the pool.
  {
    Ciphertext<word> half;
    EncryptAt(pool, ct, msg, 0);
    b->HalfBoot(half, ct, land.ui->GetEvkMap());
    const int dec = r.spec.default_encryption_level;
    EXPECT_EQ(pool.param->NPToLevel(half.GetNP()), dec);
    const auto a = Decrypt(pool, half), c = Decrypt(land, half);
    double diff = 0.0;
    for (int i = 0; i < num_slots; i++)
      diff = std::max(diff, std::abs(a[i] - c[i]));
    EXPECT_LT(diff, 1e-9) << "the two rings read HalfBoot's words differently";
    const auto &bp = b->GetBootParameter();
    if (bp.GetNumSlackLevels() > 0)
      b->LevelDown(half, half, bp.GetStCStartLevel());
    Ciphertext<word> cyc;
    b->SlotToCoeff(cyc, num_slots, half, land.ui->GetEvkMap());
    EXPECT_EQ(pool.param->NPToLevel(cyc.GetNP()), L);
    const Fit h = FitResidual(msg, Decrypt(pool, cyc));
    std::cout << "[crossing] HalfBoot at " << dec << " + StC, read by the pool: "
              << "carried " << h.carried << " = 2^"
              << std::log2(std::abs(h.carried)) << " (derived ratio 2^"
              << std::log2(b->GetMessageRatio()) << "), relative residual 2^"
              << std::log2(h.residual / std::abs(h.carried)) << std::endl;
    EXPECT_LT(h.residual / std::abs(h.carried), 1e-2);
  }

  // The other way: encrypted on the ladder at L, read by the pool.
  {
    Ciphertext<word> up;
    EncryptAt(land, up, msg, L);
    const Fit u = FitResidual(msg, Decrypt(pool, up));
    EXPECT_NEAR(u.carried, 1.0, 1e-6);
    EXPECT_LT(u.residual, 1e-6) << "the pool cannot read the ladder's ciphertext";
  }
}

// (3) THE COST, and the precision that goes with it.
TEST(LandingLadder, TheShortClimbIsCheaperThanSlackAtThePoolsPrecision) {
  const LandingLadder<word> lad(Load(PoolParam()));
  const int L = LandingLevel();
  if (L == lad.PoolLanding()) {
    GTEST_SKIP() << "the pool's own landing: nothing to compare";
  }
  const int num_slots = lad.Pool().log_degree > 0
                            ? (lad.Pool().conjugate_invariant
                                   ? (1 << lad.Pool().log_degree)
                                   : (1 << (lad.Pool().log_degree - 1)))
                            : 0;
  const auto msg = RandomReal(num_slots, 1.0, 0x5EED);
  const int reps = 10;

  double ms_pool = 0.0, p_pool = 0.0;
  {
    // The full climb, StC lowered by slack, lands at L.
    Ring pool(PoolParam(), {}, lad.PoolLanding() - L);
    auto b = PrepareBoot(pool);
    ASSERT_EQ(b->GetBootParameter().GetEndLevel(), L);
    Ciphertext<word> ct, res;
    EncryptAt(pool, ct, msg, 0);
    ms_pool = TimeBoots(b, pool, ct, res, reps);
    EXPECT_EQ(pool.param->NPToLevel(res.GetNP()), L);
    p_pool = -std::log2(FitResidual(msg, Decrypt(pool, res)).residual);
    std::cout << "[cost] pool + slack " << (lad.PoolLanding() - L)
              << ": climb " << b->GetBootParameter().GetMaxLevel() << " ("
              << pool.param->LevelToNP(b->GetBootParameter().GetMaxLevel())
                     .GetNumTotal()
              << " primes), " << ms_pool << " ms/boot, p " << p_pool
              << std::endl;
  }
  double ms_ladder = 0.0, p_ladder = 0.0;
  const auto r = lad.ForLanding(L, Policy());
  {
    Ring land(r.spec, {}, r.slack);
    auto b = PrepareBoot(land);
    ASSERT_EQ(b->GetBootParameter().GetEndLevel(), L);
    Ciphertext<word> ct, res;
    EncryptAt(land, ct, msg, 0);
    ms_ladder = TimeBoots(b, land, ct, res, reps);
    EXPECT_EQ(land.param->NPToLevel(res.GetNP()), L);
    p_ladder = -std::log2(FitResidual(msg, Decrypt(land, res)).residual);
    std::cout << "[cost] ladder (" << LandingLadder<word>::Describe(r) << "): "
              << ms_ladder << " ms/boot, p " << p_ladder << std::endl;
  }
  std::cout << "[cost] landing " << L << ": ladder / pool+slack = "
            << ms_ladder / ms_pool << std::endl;
  EXPECT_LT(ms_ladder, ms_pool) << "the short climb is not cheaper";
  // The precision is the pool's: the band is the same pairs.
  EXPECT_GT(p_ladder, p_pool - 0.5) << "the ladder lost precision";
}
