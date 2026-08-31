// Two compiled transforms, coefficient by coefficient.
//
// A compiled `LinearTransform` IS its plaintexts, and a plaintext is stored
// NTT-applied: one coefficient off by one changes every limb of its prime. So
// two compilations of the same matrix cannot be compared by their bytes -- the
// leg's converter cache files are 4-7 GB each and `cmp` on them can only say
// "different". This file compares them the way that means something: every
// plaintext of both back through the inverse NTT to plain residues, and each
// coefficient classified by the difference of its INTEGER, read across all of
// its primes at once.
//
// What the classification can find. `HoistHandler::CompilePlaintexts` encodes
// on the device (Doing.md 1.5ez); the host encoder is what wrote the cache
// files this pipeline carried before that. On the conjugate-invariant ring
// both routes reach the integer through `std::round` -- `RealVectorToPlaintext`
// on the host, `RnsDecompose` on the device -- behind the same special FFT in
// the same butterfly order, so two integers can differ only where a value sits
// within a few double ulps of a rounding boundary: by exactly one unit, at a
// rate the ulp predicts. Off the conjugate-invariant ring the host TRUNCATES
// (`ComplexVectorToPlaintext`, through `BigInt(double)`) and the device rounds,
// so a unit's difference is the common case there, not the rare one -- still
// one unit, never two.
//
// The one unit is the unit of a single-prime scale. At a scale wider than a
// double's mantissa -- a two-prime rescale product, 2^70 -- the integer both
// routes round to is itself a double, and the same few ulps of jitter in the
// transformed value are 2^18 units of it. So the bound is stated in the
// double's own terms, `max(1, 8 ulp(scale))`, which is exactly one unit at
// 2^35 and the jitter's size above 2^45. A difference past that, or a residue
// shift that is not the same integer on every prime, is a real disagreement
// about the transform, and that is what is asserted against.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/CiSwitchedCcmm.h"
#include "core/Serialization.h"
#include "extension/EvalSpecialFFT.h"
#include "extension/LinearTransform.h"
#include "extension/StripedMatrix.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::ArchiveReader;
using cheddar::CiSinCConverter;
using cheddar::Complex;
using cheddar::LinearTransform;
using cheddar::NPInfo;
using cheddar::Plaintext;
using cheddar::StripedMatrix;

namespace {

// How far apart two roundings of the same double may land: one unit of the
// scale, or eight ulps of a double at the scale's magnitude when those are
// bigger than a unit.
int64_t JitterBound(double scale) {
  const int exponent = static_cast<int>(std::floor(std::log2(scale)));
  const double ulp = std::ldexp(1.0, exponent - 52);
  return std::max<int64_t>(1, static_cast<int64_t>(std::ceil(8.0 * ulp)));
}

// Per-coefficient verdicts over a set of plaintext pairs.
struct Tally {
  size_t plaintexts = 0;
  size_t coefficients = 0;
  size_t identical = 0;
  size_t plus_one = 0;   // b = a + 1, on every prime
  size_t minus_one = 0;  // b = a - 1, on every prime
  size_t within = 0;     // |b - a| <= the jitter bound, other than the above
  size_t other = 0;      // past the bound, or not the same integer on every
                         // prime
  int64_t max_abs = 0;   // the largest consistent |b - a| seen

  void Add(const Tally &t) {
    plaintexts += t.plaintexts;
    coefficients += t.coefficients;
    identical += t.identical;
    plus_one += t.plus_one;
    minus_one += t.minus_one;
    within += t.within;
    other += t.other;
    max_abs = std::max(max_abs, t.max_abs);
  }
  size_t Units() const { return plus_one + minus_one; }
  void Print(const std::string &what, double log2_scale) const {
    std::cout << "  " << what << ": " << plaintexts << " plaintexts, "
              << coefficients << " coefficients at scale 2^" << std::fixed
              << std::setprecision(2) << log2_scale << " (bound "
              << JitterBound(std::exp2(log2_scale)) << ") -- identical "
              << identical << ", +1 " << plus_one << ", -1 " << minus_one
              << ", within " << within << ", other " << other << ", max |d| "
              << max_abs << "; one unit = 2^" << -log2_scale
              << " of the message, units at rate " << std::scientific
              << std::setprecision(2)
              << (coefficients == 0
                      ? 0.0
                      : static_cast<double>(Units()) / coefficients)
              << std::fixed << std::endl;
  }
};

// A plaintext back in the coefficient domain, plain residues, on the host, in
// the limb layout `[prime][coefficient]`. `INTT`'s default leaves Montgomery
// form on the way out, which `TheReaderSeesPlainResidues` pins.
std::vector<word> PlainResidues(const Ring &ring, const Plaintext<word> &pt) {
  const NPInfo np = pt.GetNP();
  Plaintext<word> tmp;
  tmp.ModifyNP(np);
  auto dst = tmp.View();
  ring.context->ntt_handler_.INTT(dst, np, pt.ConstView());
  cheddar::HostVector<word> host;
  CopyDeviceToHost(host, tmp.mx_);
  cudaDeviceSynchronize();
  return std::vector<word>(host.begin(), host.end());
}

// `b - a` as the signed integer both residues agree on, or nullopt when the
// primes do not tell the same story. The difference is read centred, so a
// small negative shift is not mistaken for a huge positive one.
bool ConsistentDifference(const std::vector<word> &ra,
                          const std::vector<word> &rb,
                          const std::vector<word> &primes, int degree, int i,
                          int64_t &d) {
  for (size_t j = 0; j < primes.size(); j++) {
    const int64_t p = static_cast<int64_t>(primes[j]);
    const size_t at = j * degree + i;
    int64_t dj = (static_cast<int64_t>(rb[at]) - static_cast<int64_t>(ra[at]));
    dj = ((dj % p) + p) % p;
    if (dj > p / 2) dj -= p;
    if (j == 0) {
      d = dj;
    } else if (dj != d) {
      return false;
    }
  }
  return true;
}

Tally Compare(const Ring &ring, const Plaintext<word> &a,
              const Plaintext<word> &b) {
  Tally t;
  t.plaintexts = 1;
  const NPInfo np = a.GetNP();
  if (!(np == b.GetNP()) || a.GetScale() != b.GetScale()) {
    // Not the same plaintext shape at all: every coefficient disagrees.
    t.coefficients = t.other = static_cast<size_t>(ring.Degree());
    return t;
  }
  const auto primes = ring.param->GetPrimeVector(np);
  const int degree = ring.Degree();
  const int64_t bound = JitterBound(a.GetScale());
  const auto ra = PlainResidues(ring, a);
  const auto rb = PlainResidues(ring, b);
  t.coefficients = static_cast<size_t>(degree);
  for (int i = 0; i < degree; i++) {
    int64_t d = 0;
    if (!ConsistentDifference(ra, rb, primes, degree, i, d)) {
      t.other++;
      continue;
    }
    const int64_t magnitude = d < 0 ? -d : d;
    if (d == 0) {
      t.identical++;
    } else if (d == 1) {
      t.plus_one++;
    } else if (d == -1) {
      t.minus_one++;
    } else if (magnitude <= bound) {
      t.within++;
    } else {
      t.other++;
    }
    if (magnitude <= bound) t.max_abs = std::max(t.max_abs, magnitude);
  }
  return t;
}

// Every plaintext pair of two compiled transforms, in lockstep over the
// `[giant][baby]` maps. A key present on one side only is a structural
// disagreement and fails outright.
Tally CompareTransforms(const Ring &ring, const LinearTransform<word> &a,
                        const LinearTransform<word> &b, const char *what) {
  const auto &pa = a.GetHoist().GetPlaintexts();
  const auto &pb = b.GetHoist().GetPlaintexts();
  EXPECT_EQ(a.GetBS(), b.GetBS()) << what;
  EXPECT_EQ(a.GetGS(), b.GetGS()) << what;
  EXPECT_EQ(a.GetDiagonalOffsets(), b.GetDiagonalOffsets()) << what;
  EXPECT_EQ(pa.size(), pb.size()) << what << ": giant-step count";
  Tally total;
  const auto t0 = std::chrono::steady_clock::now();
  size_t done = 0;
  for (const auto &[gs, inner_a] : pa) {
    const auto it = pb.find(gs);
    if (it == pb.end()) {
      ADD_FAILURE() << what << ": giant step " << gs << " only on one side";
      continue;
    }
    const auto &inner_b = it->second;
    EXPECT_EQ(inner_a.size(), inner_b.size())
        << what << ": baby-step count at giant step " << gs;
    for (const auto &[bs, pt_a] : inner_a) {
      const auto jt = inner_b.find(bs);
      if (jt == inner_b.end()) {
        ADD_FAILURE() << what << ": baby step " << bs << " of giant step "
                      << gs << " only on one side";
        continue;
      }
      total.Add(Compare(ring, pt_a, jt->second));
      if (++done % 512 == 0) {
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        std::cout << "    " << what << ": " << done << " plaintexts, "
                  << std::fixed << std::setprecision(1) << s
                  << " s, other so far " << total.other << std::endl;
      }
    }
  }
  return total;
}

double Log2Scale(const LinearTransform<word> &t) {
  const auto &pts = t.GetHoist().GetPlaintexts();
  return std::log2(pts.begin()->second.begin()->second.GetScale());
}

// The lowest level from 1 whose rescale product is one prime, so that the
// unit of the comparison is the unit of the scale.
int SinglePrimeLevel(const Ring &ring) {
  for (int level = 1; level <= ring.param->max_level_; level++) {
    if (std::log2(ring.param->GetRescalePrimeProd(level)) < 45.0) return level;
  }
  return -1;
}

}  // namespace

// ---------------------------------------------------------------------------
// The reader: `INTT` at its default recovers the plain residues of the integer
// the encoder rounded to, in `[prime][coefficient]` order. Pinned against
// `EncodeCoeff`, whose integer is stated in its own comment.
// ---------------------------------------------------------------------------

class CompiledTransform : public testing::TestWithParam<const char *> {};

TEST_P(CompiledTransform, TheReaderSeesPlainResidues) {
  Ring ring(GetParam(), {}, 0, /*build_user_interface=*/false);
  const int level = ring.enc_level;
  const double scale = ring.param->GetScale(level);
  const int degree = ring.Degree();

  std::vector<double> coeffs(degree);
  for (int i = 0; i < degree; i++) {
    coeffs[i] = std::cos(0.731 * i + 0.2) * 0.9;
  }
  Plaintext<word> pt;
  ring.context->encoder_.EncodeCoeff(pt, level, scale, coeffs);
  const auto residues = PlainResidues(ring, pt);
  const NPInfo np = pt.GetNP();
  const auto primes = ring.param->GetPrimeVector(np);
  ASSERT_EQ(residues.size(), primes.size() * degree);

  size_t wrong = 0;
  for (int i = 0; i < degree; i++) {
    const int64_t v = std::llround(coeffs[i] * scale);
    for (size_t j = 0; j < primes.size(); j++) {
      const int64_t p = static_cast<int64_t>(primes[j]);
      const uint64_t want = static_cast<uint64_t>(((v % p) + p) % p);
      if (residues[j * degree + i] != want) wrong++;
    }
  }
  EXPECT_EQ(wrong, 0u) << wrong << " residues are not the rounded integer's";
}

// ---------------------------------------------------------------------------
// A transform compiled on the device against the host encoder on the same
// hoisted messages. The hoist map is rebuilt here from `LinearTransform`'s
// stated convention -- `bs = 2`, `gs = 2`, offsets 0..3, no pre-rotation --
// so the comparison reads the compiled plaintexts and not the compiler.
// ---------------------------------------------------------------------------

TEST_P(CompiledTransform, MatchesTheHostEncoderToAUnit) {
  Ring ring(GetParam(), {}, 0, /*build_user_interface=*/false);
  const bool ci = ring.param->conjugate_invariant_;
  const int level = SinglePrimeLevel(ring);
  ASSERT_GE(level, 1) << "no level with a one-prime rescale product";
  const int height = ring.param->MaxNumSlots();
  const int bs = 2;
  const int gs = 2;
  const double pt_scale = ring.param->GetRescalePrimeProd(level);

  // Four real diagonals of transform-like entries: cosines, so nothing is a
  // dyadic rational that both routes would hit exactly.
  StripedMatrix m(height, height);
  for (int off = 0; off < bs * gs; off++) {
    m.try_emplace(off, height, Complex(0.0, 0.0));
    for (int j = 0; j < height; j++) {
      m[off][j] = Complex(0.7 * std::cos(0.0137 * j + 1.3 * off), 0.0);
    }
  }
  LinearTransform<word> t(ring.context, m, level, pt_scale, bs, gs);

  // `ConstructPlainHoistMap` with stride 1 and gs_stride = bs: offset `off`
  // is baby step `off % bs` of giant step `off - off % bs`, and the message
  // is the diagonal rotated back by the giant step.
  const auto &pts = t.GetHoist().GetPlaintexts();
  ASSERT_EQ(static_cast<int>(pts.size()), gs);
  Tally total;
  for (int off = 0; off < bs * gs; off++) {
    const int bs_rot = off % bs;
    const int gs_rot = off - bs_rot;
    std::vector<Complex> message(height);
    for (int j = 0; j < height; j++) {
      message[(j + gs_rot) % height] = m[off][j];
    }
    Plaintext<word> host_pt;
    ring.context->encoder_.Encode(host_pt, level, pt_scale, message,
                                  ring.param->alpha_);
    ASSERT_TRUE(pts.count(gs_rot) && pts.at(gs_rot).count(bs_rot))
        << "offset " << off << " is not at [" << gs_rot << "][" << bs_rot
        << "]";
    total.Add(Compare(ring, host_pt, pts.at(gs_rot).at(bs_rot)));
  }
  total.Print(std::string(GetParam()) + (ci ? " (R+)" : " (ordinary)") +
                  " level " + std::to_string(level),
              std::log2(pt_scale));

  EXPECT_EQ(total.other, 0u)
      << "a coefficient differs by more than one unit, or by a shift that is "
         "not the same integer on every prime";
  if (ci) {
    // Both routes round; only ulp jitter at a rounding boundary separates
    // them, and at 2^35 that is a handful of coefficients in 2^14.
    EXPECT_LE(total.Units(), total.coefficients / 500);
  } else {
    // The host truncates and the device rounds: about half the coefficients
    // are a unit apart, and it is the device that is right.
    EXPECT_GT(total.Units(), 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(Cheddar, CompiledTransform,
                         testing::Values("ci12_35.json",
                                         "ringdegree12_35.json"),
                         [](const testing::TestParamInfo<const char *> &info) {
                           std::string name(info.param);
                           std::replace(name.begin(), name.end(), '.', '_');
                           return name;
                         });

// ---------------------------------------------------------------------------
// The leg's P/V converter recipe (sub-degree 32, forward at 3, inverse at 1,
// the chain layout at rank 16, 256 baby steps, no premap) built here on the
// switch ring and written where `CHEDDAR_CONVERTER_BUILD_OUT` says, so that
// the test below can hold it against the cache file an earlier build wrote.
// The constructor prints its own split (stages / folds / compile per
// direction); this is the harness for the cold build's cost. Skipped when
// no output path is given: it is minutes and 7 GB.
// ---------------------------------------------------------------------------

TEST(ConverterBuild, ThePvRecipeIsBuiltAndWritten) {
  const char *out = std::getenv("CHEDDAR_CONVERTER_BUILD_OUT");
  if (out == nullptr || out[0] == 0) {
    GTEST_SKIP() << "set CHEDDAR_CONVERTER_BUILD_OUT to the file to write";
  }
  const char *param_env = std::getenv("CHEDDAR_CONVERTER_PARAM");
  const std::string param = (param_env != nullptr && param_env[0] != 0)
                                ? param_env
                                : "ci_ringswitch16_35_boot.json";
  Ring ring(param, {}, 0, /*build_user_interface=*/false);
  const int degree = ring.Degree();
  constexpr int kSubDegree = 32;
  constexpr int kRank = 16;
  constexpr int kForwardLevel = 3;
  constexpr int kInverseLevel = 1;
  constexpr int kBabySteps = 256;
  const cheddar::CiSwitchedCcmmLayout layout(degree, degree / kRank,
                                             kSubDegree);

  const auto t0 = std::chrono::steady_clock::now();
  CiSinCConverter<word> conv(ring.context, kSubDegree, kForwardLevel,
                             kInverseLevel, &layout, /*forward_premap=*/nullptr,
                             kBabySteps);
  const auto t1 = std::chrono::steady_clock::now();
  ASSERT_NE(conv.GetForward(), nullptr);
  ASSERT_NE(conv.GetInverse(), nullptr);

  int64_t written = 0;
  {
    cheddar::ArchiveWriter ar(out, cheddar::IdentityOf(*ring.param));
    conv.Save(ar);
    ar.Close();
    written = ar.Written();
  }
  const auto t2 = std::chrono::steady_clock::now();
  std::cout << "  built both directions in " << std::fixed
            << std::setprecision(1)
            << std::chrono::duration<double>(t1 - t0).count()
            << " s, wrote " << (written >> 20) << " MiB to " << out << " in "
            << std::chrono::duration<double>(t2 - t1).count() << " s"
            << std::endl;
}

// ---------------------------------------------------------------------------
// Two converter cache files -- one written by the host encoder, one by the
// device -- coefficient by coefficient. Driven by the environment because the
// files are the leg's real converters: `CHEDDAR_CONVERTER_A` and `_B` name
// them and `CHEDDAR_CONVERTER_PARAM` the ring they were compiled on (the
// leg's switch ring by default). Skipped when they are not given.
// ---------------------------------------------------------------------------

TEST(ConverterCache, TwoCompilationsAgreeToAUnit) {
  const char *a_path = std::getenv("CHEDDAR_CONVERTER_A");
  const char *b_path = std::getenv("CHEDDAR_CONVERTER_B");
  if (a_path == nullptr || b_path == nullptr) {
    GTEST_SKIP() << "set CHEDDAR_CONVERTER_A and CHEDDAR_CONVERTER_B to two "
                    "converter cache files";
  }
  const char *param_env = std::getenv("CHEDDAR_CONVERTER_PARAM");
  const std::string param = (param_env != nullptr && param_env[0] != 0)
                                ? param_env
                                : "ci_ringswitch16_35_boot.json";

  Ring ring(param, {}, 0, /*build_user_interface=*/false);
  const auto id = cheddar::IdentityOf(*ring.param);
  ASSERT_EQ(ArchiveReader::PeekIdentity(a_path), id)
      << a_path << " was not compiled on " << param;
  ASSERT_EQ(ArchiveReader::PeekIdentity(b_path), id)
      << b_path << " was not compiled on " << param;

  auto load = [&](const char *path) {
    const auto t0 = std::chrono::steady_clock::now();
    ArchiveReader ar(path, id);
    auto conv = CiSinCConverter<word>::Load(ar);
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::cout << "  read " << (ArchiveReader::FileSize(path) >> 20)
              << " MiB from " << path << " in " << std::fixed
              << std::setprecision(1) << s << " s" << std::endl;
    return conv;
  };
  auto a = load(a_path);
  auto b = load(b_path);
  ASSERT_EQ(a->GetSubDegree(), b->GetSubDegree());

  auto direction = [&](const LinearTransform<word> *ta,
                       const LinearTransform<word> *tb, const char *what) {
    ASSERT_EQ(ta == nullptr, tb == nullptr)
        << what << " was built on one side only";
    if (ta == nullptr) return;
    const Tally t = CompareTransforms(ring, *ta, *tb, what);
    t.Print(what, Log2Scale(*ta));
    EXPECT_EQ(t.other, 0u)
        << what << ": a coefficient differs past the jitter bound, or by a "
           "shift that is not the same integer on every prime";
    EXPECT_GT(t.coefficients, 0u);
  };
  direction(a->GetForward(), b->GetForward(), "forward");
  direction(a->GetInverse(), b->GetInverse(), "inverse");
}
