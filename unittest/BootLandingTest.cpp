// Level-targeted ("landing") bootstrapping on the conjugate-invariant ring, in
// the form a Llama-3 layer can consume.
//
// WHY. [Grafting] appendix D.2 ("Bootstrapping with Flexible Output Modulus")
// observes that when the work between two bootstraps is shallow -- exactly the
// [SYLPH] private-prefill regime, where the non-linear operators sit high and
// the matrix product at the lowest level -- the bootstrap need not raise the
// modulus all the way to the top. Raising only to the level the schedule needs
// makes ModUp, CoeffToSlot and EvalMod act on fewer RNS limbs.
//
// The mechanism already exists in this library: `BootContext::Boot` climbs
// through `ModUpToLevel(boot_param_.GetMaxLevel())`, and CoeffToSlot, EvalMod
// and SlotToCoeff all derive their levels from the BootParameter, so a
// BootParameter with a smaller max_level lands lower and does every limb
// operation in between on fewer primes. `ci16_40` already ships this (climbs
// to 28, lands 16). What was missing for the LAYER is a scale-35 sub-ladder
// that SHARES ci16_35's bottom primes, so a ci16_35 ciphertext crosses into it
// with no key -- `gen_landing.py` builds it.
//
// HALFBOOT LANDS AT EVERY L in [5, 19]; FULL BOOT AT ODD L >= 11. HalfBoot --
// the layer's crossing op -- is CtS + EvalMod and lands at ANY L (odd residual
// ~1.1e-04; even ~3.8e-04, the ~2-bit cost of the single 25-bit CtS level an
// even landing forces, 5 - t_L being odd). Full Boot additionally runs StC,
// which (a) needs its landing above ci16_35's num_accum==1 zone and (b) at an
// even landing reads the scale the 25-bit CtS left slightly off and corrupts --
// so Boot is clean only at ODD L >= 11 (residual ~3e-05). Junctions (L 9, 15,
// where grafting swapped mains out below the running peak) are fine: EvalMod's
// bottom two levels ride the swapped-out 30-bit mains at 59/60-bit, tolerated.
// The 12-level budget above the landing (8 EvalMod + 4 CtS, pinned by
// default_enc == max - 12) is why there is no room for a bridge. gen_landing.py
// emits any L in [5, 19]; odd L is preferred (full precision, full Boot).
//
// WHAT IS MEASURED. (1) The preset bootstraps, lands where its BootParameter
// asks, and preserves the message -- HalfBoot at every L, Boot where valid.
// (2) A ciphertext ENCRYPTED ON ci16_35 crosses into this ladder with no key
// and comes back readable by ci16_35 -- the layer-usable form. The preset and
// its level are env-driven (CHEDDAR_LAND_PARAM / CHEDDAR_LAND_LEVEL), so any
// generated ladder validates by dropping its JSON into the binary dir -- no
// rebuild. Correctness only.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include "RingFixture.h"
#include "extension/CiModuleBasis.h"

namespace {

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::Plaintext;

constexpr const char *kFull = "ci16_35.json";  // the leg's ring

const char *LandParam() {
  const char *e = std::getenv("CHEDDAR_LAND_PARAM");
  return (e && e[0]) ? e : "ci16_35_land13.json";
}
int LandLevel() {
  const char *e = std::getenv("CHEDDAR_LAND_LEVEL");
  return (e && e[0]) ? std::atoi(e) : 13;
}
bool MinKs() {  // memory lever on this shared A6000; correctness is unaffected
  const char *e = std::getenv("CHEDDAR_CI_MINKS");
  return e != nullptr && e[0] == '1';
}

std::vector<Complex> RandomReal(int num_slots, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> d(-0.5, 0.5);
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

// Fit the scalar the output carries and return the residual. `Boot` is message
// preserving (fit ~ 1); `HalfBoot` leaves the crossing constant on the message
// (fit ~ 2^-log_message_ratio ~ 1/32), which the layer folds into the next
// plaintext multiply. Either way, a correct bootstrap is a clean scalar
// multiple of the input, and the residual is the precision.
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
  b->AddRequiredRotations(req, num_slots, MinKs());
  r.ui->PrepareRotationKey(req);
  return b;
}

int LimbsAt(const Ring &r, int level) {
  return r.param->LevelToNP(level).GetNumTotal();
}

}  // namespace

// (1) The landing preset bootstraps, lands where its BootParameter asks, and
// preserves the message, via both HalfBoot and (where the level budget allows)
// Boot -- with a control that the climb really is shorter.
TEST(BootLanding, LandsLowerAndPreservesTheMessage) {
  Ring land(LandParam());
  auto b = PrepareBoot(land);
  const int num_slots = land.param->MaxNumSlots();

  const int climb = b->GetBootParameter().GetMaxLevel();
  const int halfboot_land = b->GetBootParameter().GetEvalModEndLevel();
  const int boot_land = b->GetBootParameter().GetEndLevel();
  std::cout << "[landing] climb to level " << climb << " (" << LimbsAt(land, climb)
            << " limbs), HalfBoot lands " << halfboot_land << ", Boot lands "
            << boot_land << std::endl;
  EXPECT_EQ(halfboot_land, LandLevel());

  const auto msg = RandomReal(num_slots, 0xA11CE);
  Ciphertext<word> ct;
  EncryptAt(land, ct, msg, 0);

  // HalfBoot -- the layer's crossing op, valid for ANY landing L >= 1. Message
  // preserving up to the crossing constant, so the residual (not the raw error)
  // is the precision.
  Ciphertext<word> hres;
  b->HalfBoot(hres, ct, land.ui->GetEvkMap());
  EXPECT_EQ(land.param->NPToLevel(hres.GetNP()), halfboot_land)
      << "HalfBoot did not land where its BootParameter asks";
  const Fit h = FitResidual(msg, Decrypt(land, hres));
  std::cout << "[landing] HalfBoot carried " << h.carried << ", residual "
            << h.residual << " (a clean scalar multiple; the crossing constant "
            << "the layer folds is absorbed by the fit)" << std::endl;
  EXPECT_LT(h.residual, 0.01) << "HalfBoot did not preserve the message";

  // Full Boot additionally runs SlotToCoeff. Two things restrict it, both
  // measured, neither touching HalfBoot (the layer's crossing op, which has no
  // StC and lands at ANY L >= 5). (1) StC is a hoisted transform, and on
  // ci16_35's alpha-12 basis a hoisted transform in the num_accum==1 zone
  // (levels 0..6) produces mod-Q noise (Doing.md 1.5bt), so Boot needs its
  // landing >= 7. (2) At an EVEN landing the CtS carries a single-terminal
  // 25-bit level (5 - t_L odd); HalfBoot absorbs the ~2-bit precision cost in
  // its fit, but StC then reads a scale the 25-bit phase left slightly off and
  // corrupts (residual ~4.6). So full Boot is valid at ODD landings >= 11;
  // HalfBoot is what an even landing offers, ~2 bits lossier.
  // (1) is gone: Hoist.cu's baby-step dispatch now has its num_accum == 1
  // branch, so StC runs at any level and full Boot is limited by (2) alone --
  // odd landings, with StC's three levels below them.
  const bool boot_ok = boot_land >= 1 && (LandLevel() % 2 == 1);
  if (boot_ok) {
    Ciphertext<word> res;
    EncryptAt(land, ct, msg, 0);
    b->Boot(res, ct, land.ui->GetEvkMap());
    EXPECT_EQ(land.param->NPToLevel(res.GetNP()), boot_land)
        << "Boot did not land where its BootParameter asks";
    const Fit bf = FitResidual(msg, Decrypt(land, res));
    std::cout << "[landing] Boot carried " << bf.carried << ", residual "
              << bf.residual << std::endl;
    EXPECT_LT(bf.residual, 0.01) << "Boot did not preserve the message";
    EXPECT_NEAR(bf.carried, 1.0, 0.02) << "Boot should be message preserving";
  } else {
    std::cout << "[landing] Boot skipped (landing " << LandLevel() << ", lands at "
              << boot_land << "; full Boot needs an odd landing >= 11. HalfBoot "
                 "is the layer op here)" << std::endl;
  }
}

// (2) The layer-usable property: a ciphertext ENCRYPTED ON ci16_35 crosses into
// the landing ladder with no key, bootstraps there (climbing less), and comes
// back readable by ci16_35 -- because the two share a secret and their levels
// 0..L are byte-identical.
TEST(BootLanding, Ci16CiphertextCrossesInAndBackKeylessly) {
  Ring full(kFull);
  Ring land(LandParam(), full.ui->GetSecretCoeffs());  // SAME secret -> keyless
  auto b = PrepareBoot(land);
  const int num_slots = full.param->MaxNumSlots();
  ASSERT_EQ(num_slots, land.param->MaxNumSlots());

  const int land_climb = b->GetBootParameter().GetMaxLevel();
  std::cout << "[cross] ci16_35 climbs to " << full.param->max_level_ << " ("
            << LimbsAt(full, full.param->max_level_) << " limbs); this crossing "
            << "climbs to " << land_climb << " (" << LimbsAt(land, land_climb)
            << " limbs) -- " << (LimbsAt(full, full.param->max_level_) -
                                 LimbsAt(land, land_climb))
            << " fewer, on every ModUp/CtS/EvalMod" << std::endl;

  // The shared bottom is byte-identical -- what "keyless" rests on.
  for (int L = 0; L <= LandLevel(); L++) {
    const auto pf = full.param->GetPrimeVector(full.param->LevelToNP(L));
    const auto pl = land.param->GetPrimeVector(land.param->LevelToNP(L));
    ASSERT_EQ(pf, pl) << "levels 0.." << LandLevel()
                      << " must be byte-identical for a keyless crossing; "
                      << "they differ at level " << L;
  }

  const auto msg = RandomReal(num_slots, 0x50FA);
  Ciphertext<word> ct;
  EncryptAt(full, ct, msg, 0);  // encrypted on ci16_35

  // Bootstrap it with the LANDING ring's BootContext and keys -- no key switch
  // moves it between rings; same secret, same level-0 primes. Boot where its StC
  // clears ci16_35's num_accum==1 zone (message preserving, lands in the shared
  // range), else the layer's HalfBoot.
  const bool use_boot =
      b->GetBootParameter().GetEndLevel() >= 1 && (LandLevel() % 2 == 1);
  Ciphertext<word> res;
  if (use_boot)
    b->Boot(res, ct, land.ui->GetEvkMap());
  else
    b->HalfBoot(res, ct, land.ui->GetEvkMap());
  const int expect_level = use_boot ? b->GetBootParameter().GetEndLevel()
                                    : b->GetBootParameter().GetEvalModEndLevel();
  EXPECT_EQ(land.param->NPToLevel(res.GetNP()), expect_level);

  // Read it back with ci16_35's own UserInterface -- same secret, landed in the
  // shared range, so ci16_35 owns this ciphertext.
  const Fit f = FitResidual(msg, Decrypt(full, res));
  std::cout << "[cross] ci16_35 reads the landed ciphertext via "
            << (use_boot ? "Boot" : "HalfBoot") << ", carried " << f.carried
            << ", residual " << f.residual << std::endl;
  EXPECT_LT(f.residual, 0.01)
      << "the crossed-and-bootstrapped message did not survive; either the "
         "crossing was not keyless or the landing ladder is wrong";
}

// (3) The layer's OTHER crossing op on the landing ladder: `HalfBootModule`
// (Doing.md 3.5-3.7) reads the MODULE coordinates of the coefficient image
// through `CiModuleBasis`'s CoeffToSlot -- in the ladder's own CtS level count,
// which for a `*c2` preset (gen_landing.py's 4th argument) is the two-level
// real form. Coefficients in at level 0, the module coordinates out in the
// slots at the landing, times the crossing constant. Needs the SSE secret
// sampled in the module basis, which this test sets unless given.
TEST(BootLanding, HalfBootModuleLandsTheModuleCoordinates) {
  setenv("CHEDDAR_MODULE_SPARSE_SECRET", "128,16", /*overwrite=*/0);
  Ring land(LandParam());
  auto b = PrepareBoot(land);
  const int n = land.param->MaxNumSlots();
  const int T = 128;
  const int k = n / T;
  const int cts_levels = b->GetBootParameter().num_cts_levels_;
  typename cheddar::CiModuleBasis<word>::Phases ph;
  if (cts_levels == 2) {
    ph.cts_twist = {9};
    ph.cts_small = {7};
  } else if (cts_levels == 3) {
    ph.cts_twist = {9};
    ph.cts_small = {4, 3};
  } else {
    ASSERT_EQ(cts_levels, 4);
    ph.cts_twist = {5, 4};
    ph.cts_small = {4, 3};
  }
  cheddar::CiModuleBasis<word> basis(land.context, T, /*stc_level=*/-1,
                                     b->GetBootParameter().GetCtSStartLevel(),
                                     ph, 1.0, n * b->GetCtSConst());
  {
    EvkRequest req;
    basis.AddRequiredRotations(req, MinKs());
    land.ui->PrepareRotationKey(req);
  }
  std::cout << "[module] " << LandParam() << ": climb to "
            << b->GetBootParameter().GetMaxLevel() << " ("
            << LimbsAt(land, b->GetBootParameter().GetMaxLevel())
            << " limbs), module CtS " << basis.GetCtSNumLevels()
            << " levels from " << basis.GetCtSLevel() << ", HalfBoot lands "
            << b->GetBootParameter().GetEvalModEndLevel() << std::endl;

  // Random module coordinates at the FFN's ride, and their native image
  // `r[t k + i] = x_i[t] + [i != 0] x_{k-i}[t+1]` (what ModPack emits).
  std::mt19937_64 gen(0x3D01);
  std::uniform_real_distribution<double> d(-0.2, 0.2);
  std::vector<double> x(static_cast<size_t>(n));  // flat = t * k + i
  for (auto &v : x) v = d(gen);
  std::vector<double> image(static_cast<size_t>(n), 0.0);
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < k; i++) {
      double v = x[static_cast<size_t>(t) * k + i];
      if (i != 0 && t + 1 < T) v += x[static_cast<size_t>(t + 1) * k + (k - i)];
      image[static_cast<size_t>(t) * k + i] = v;
    }
  }
  Plaintext<word> pt;
  land.context->encoder_.EncodeCoeff(pt, 0, land.param->GetScale(0), image);
  Ciphertext<word> ct;
  land.ui->Encrypt(ct, pt);
  ct.SetNumSlots(n);

  Ciphertext<word> res;
  b->HalfBootModule(res, ct, land.ui->GetEvkMap(), basis);
  EXPECT_EQ(land.param->NPToLevel(res.GetNP()),
            b->GetBootParameter().GetEvalModEndLevel());
  const auto got = Decrypt(land, res);
  // Slot s carries module coordinate BitReverse(s); the message rides the
  // crossing constant, which `GetMessageRatio` states exactly.
  std::vector<Complex> want(static_cast<size_t>(n));
  double absmax = 0.0;
  for (int s = 0; s < n; s++) {
    want[s] = Complex(x[basis.ModuleIndexOfSlot(s)], 0.0);
    absmax = std::max(absmax, std::abs(want[s].real()));
  }
  const Fit f = FitResidual(want, got);
  // Where the residual lives: a few slots far out is EvalMod past its range
  // (the SSE secret's wrap-around, Doing.md 3.6 -- a property of the DRAW at
  // h = 16), every slot a little out is the transform. Both are reported.
  int far = 0, near = 0;
  double rms = 0.0;
  const double ratio = b->GetMessageRatio();
  for (int s = 0; s < n; s++) {
    const double e = std::abs(got[s].real() - ratio * want[s].real());
    rms += e * e;
    if (e > 1e-4 * absmax) far++;
    if (e > 1e-5 * absmax) near++;
  }
  rms = std::sqrt(rms / n);
  std::cout << "[module] HalfBootModule carried " << f.carried
            << " against the derived " << ratio << " (fit/derived "
            << f.carried / ratio << "), residual " << f.residual
            << " on |x| <= " << absmax << " = 2^"
            << std::log2(f.residual / absmax) << ", rms 2^"
            << std::log2(rms / absmax) << " against the derived ratio; slots "
            << "past 1e-5: " << near << ", past 1e-4: " << far << " of " << n
            << std::endl;
  EXPECT_NEAR(f.carried / b->GetMessageRatio(), 1.0, 5e-3)
      << "the module HalfBoot does not carry the crossing constant";
  EXPECT_LT(f.residual / absmax, 2e-3)
      << "the module coordinates did not survive the landing ladder's "
         "HalfBootModule";
}
