// The SinC conversions of the conjugate-invariant CC-MM chain fused into the
// bootstrap (Doing.md 3.16, [SYLPH] section 2.1.2 / 3.3): the TOWER basis of
// `CiSinCBasis` on the device, at the leg's shape -- degree 65536, 16 parts,
// 128 blocks a part, 32 lanes -- against the host encoder.
//
// What the host script (reference/scripts/ci_nested_sinc.py) established is
// that the chain's nested operand is the element whose tower coordinates are
// the per-part SinC coefficients, that the forward is the tower StC' without
// its lane group (consuming the message at LocateSlot's PRIMARY addresses),
// that the tower CtS' is three real transforms, and that HalfBoot then leaves
// only the lane prefix undone. What only a device can say is what they cost
// in noise and time at full size, and whether EvalMod's range holds under the
// tower-sparse secret. Three tests:
//
//   ForwardLandsTheNestedOperand -- a message through the forward; the
//       decrypted NATIVE coefficients must be the tower recomposition of the
//       per-part CI block encodes (`Encoder::CiBlockEncode`, the banded map
//       twice), which is exactly what `CiSwitchedCcmmHandler::Multiply`'s
//       ring switch takes apart into parts. The flat SinC encode of the same
//       message is the control that must NOT match (it differs by g, 1.5bp).
//
//   CtSReadsTheTowerCoordinates -- an element built from random tower
//       coordinates through the tower CtS' must land them, bit-reversed.
//
//   HalfBootTowerReturnsTheMessage -- the fused return end to end on the
//       K = 64 landing ring: message -> forward -> level 0 -> HalfBootTower
//       -> prefix -> the message at the primary addresses, the fitted
//       constant against the derived message ratio, slots past EvalMod's
//       range counted. Needs the SSE secret sampled sparse in the tower,
//       which this test sets unless given.
//
// The reference is the host, never another run of a transform.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>

#include "Testbed.h"
#include "core/Mlwe.h"
#include "extension/BootContext.h"
#include "extension/CiSinCBasis.h"
#include "extension/Profile.h"

using word = uint32_t;

namespace {

constexpr int kSmallDegree = 4096;  // the ring-switch target: 16 parts
constexpr int kSubDegree = 32;      // the SinC lane degree

// The tower-sparse secret unless the caller chose one: the fused return's
// wrap-around is bounded by the secret's norm in TOWER coordinates
// (ci_nested_sinc.py check 5: h = 16 -> max 32 / std 5.05 against K = 64;
// a native- or module-sparse secret puts it at 780 / 270).
struct TowerSecretEnv {
  TowerSecretEnv() {
    setenv("CHEDDAR_MODULE_SPARSE_SECRET", "4096:128,16", /*overwrite=*/0);
  }
} tower_secret_env;

struct ErrorStats {
  double max_abs = 0.0;
  double rms_err = 0.0;
  double rms_ref = 0.0;
  double Bits() const { return std::log2(rms_ref / rms_err); }
};

ErrorStats Compare(const std::vector<double> &expected,
                   const std::vector<double> &obtained) {
  ErrorStats s;
  double err2 = 0.0, ref2 = 0.0;
  for (size_t i = 0; i < expected.size(); i++) {
    const double d = expected[i] - obtained[i];
    s.max_abs = std::max(s.max_abs, std::abs(d));
    err2 += d * d;
    ref2 += expected[i] * expected[i];
  }
  s.rms_err = std::sqrt(err2 / expected.size());
  s.rms_ref = std::sqrt(ref2 / expected.size());
  return s;
}

void Report(const std::string &name, const ErrorStats &s) {
  std::cout << std::scientific << std::setprecision(3) << "[" << name
            << "] max |err| " << s.max_abs << ", rms err " << s.rms_err
            << ", rms ref " << s.rms_ref << std::fixed << std::setprecision(2)
            << ", " << s.Bits() << " bits" << std::endl;
}

// The tower's native image: tower coordinate flat = (t k_i + j) k_o + i ->
// the per-part banded map (Encode.cpp's CiSinCRecompose on each part), then
// the outer one on the parts.
std::vector<double> TowerNative(const std::vector<double> &x, int ko, int ki,
                                int Tl) {
  const int n_s = ki * Tl;
  const int n = ko * n_s;
  std::vector<double> parts(n, 0.0);  // part i's coefficient t' at t' ko + i
  for (int i = 0; i < ko; i++) {
    for (int j = 0; j < ki; j++) {
      for (int t = 0; t < Tl; t++) {
        const double v = x[(t * ki + j) * ko + i];
        parts[(t * ki + j) * ko + i] += v;
        if (j != 0 && t >= 1) parts[((t - 1) * ki + (ki - j)) * ko + i] += v;
      }
    }
  }
  std::vector<double> native(n, 0.0);
  for (int tp = 0; tp < n_s; tp++) {
    for (int i = 0; i < ko; i++) {
      double v = parts[tp * ko + i];
      if (i != 0 && tp + 1 < n_s) v += parts[(tp + 1) * ko + (ko - i)];
      native[tp * ko + i] = v;
    }
  }
  return native;
}

}  // namespace

class CiSinCBasisTest : public Testbed<word> {};

TEST_P(CiSinCBasisTest, ForwardLandsTheNestedOperand) {
  if (!param_->conjugate_invariant_) GTEST_SKIP() << "R+ only";
  const int n = param_->MaxNumSlots();
  const int ko = n / kSmallDegree;
  const int ki = kSmallDegree / kSubDegree;
  const int Tl = kSubDegree;
  CiSinCBasis<word> basis(n, kSmallDegree, kSubDegree);
  const int level = 3;  // where the leg's descents run
  basis.PrepareForward(context_, "p", level);
  EvkRequest req;
  basis.AddForwardRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> draw;
  GenerateRandomMessage(draw, n, -1.0, 1.0, /*complex=*/false);
  std::vector<double> z(n);
  for (int s = 0; s < n; s++) z[s] = draw[s].real();

  // The host reference: per (part, block) the CI block encode of the lanes
  // read at the primary addresses, then the tower's native image.
  std::vector<double> x(n, 0.0);
  std::vector<double> lanes(Tl), comp;
  for (int i = 0; i < ko; i++) {
    for (int j = 0; j < ki; j++) {
      for (int lane = 0; lane < Tl; lane++) {
        lanes[lane] = z[basis.PrimarySlot(i, j, lane)];
      }
      context_->encoder_.CiBlockEncode(comp, lanes);
      for (int t = 0; t < Tl; t++) x[(t * ki + j) * ko + i] = comp[t];
    }
  }
  const auto want = TowerNative(x, ko, ki, Tl);

  Plaintext<word> pt;
  context_->encoder_.Encode(pt, level, param_->GetScale(level), draw);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  Ciphertext<word> out;
  Profile::Reset();
  __ProfileStart("tower forward", 1, );
  basis.Forward("p", out, ct, interface_->GetEvkMap());
  __ProfileEnd("tower forward");
  Profile::Report("tower forward, warm-up included");
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(param_->NPToLevel(out.GetNP()),
            level - basis.GetForwardNumLevels("p"));

  Plaintext<word> back;
  interface_->Decrypt(back, out);
  std::vector<double> got;
  context_->encoder_.DecodeCoeff(got, back);
  ASSERT_EQ(static_cast<int>(got.size()), n);
  const auto stats = Compare(want, got);
  Report("forward: native coefficients vs the tower encode", stats);

  // The control: the FLAT SinC encode of the same message (one module over
  // the lane ring, no inner structure) is a different element.
  Plaintext<word> flat_pt;
  context_->encoder_.EncodeSinC(flat_pt, level, param_->GetScale(level), draw,
                                kSubDegree);
  std::vector<double> flat;
  context_->encoder_.DecodeCoeff(flat, flat_pt);
  const auto control = Compare(flat, got);
  Report("control: against the flat SinC encode (must NOT match)", control);

  std::cout << "forward diagonals:";
  for (int d : basis.GetForwardDiagonals("p")) std::cout << " " << d;
  std::cout << std::endl;
  EXPECT_GT(stats.Bits(), 12.0) << "the forward did not land the nested "
                                   "operand";
  EXPECT_LT(control.Bits(), 3.0) << "the flat encode should not match";
}

TEST_P(CiSinCBasisTest, CtSReadsTheTowerCoordinates) {
  if (!param_->conjugate_invariant_) GTEST_SKIP() << "R+ only";
  const int n = param_->MaxNumSlots();
  const int ko = n / kSmallDegree;
  const int ki = kSmallDegree / kSubDegree;
  const int Tl = kSubDegree;
  CiSinCBasis<word> basis(n, kSmallDegree, kSubDegree);
  const int level = param_->max_level_;
  basis.PrepareCtS(context_, level, 1.0);
  EvkRequest req;
  basis.AddCtSRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> draw;
  GenerateRandomMessage(draw, n, -1.0, 1.0, /*complex=*/false);
  std::vector<double> x(n);
  for (int i = 0; i < n; i++) x[i] = draw[i].real();
  const auto native = TowerNative(x, ko, ki, Tl);

  // The top level has no canonical scale; the rescale product is what a
  // CoeffToSlot input carries there.
  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), native);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  Ciphertext<word> out;
  Profile::Reset();
  __ProfileStart("tower CtS", 1, );
  basis.EvaluateCtS(out, ct, interface_->GetEvkMap());
  __ProfileEnd("tower CtS");
  Profile::Report("tower CtS, warm-up included");
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(param_->NPToLevel(out.GetNP()), level - basis.GetCtSNumLevels());

  std::vector<Complex> got;
  DecryptAndDecode(got, out);
  std::vector<double> expected(n), obtained(n);
  for (int s = 0; s < n; s++) {
    expected[s] = x[basis.TowerIndexOfSlot(s)];
    obtained[s] = got[s].real();
  }
  const auto stats = Compare(expected, obtained);
  Report("tower CtS: slots vs the coordinates", stats);
  std::cout << "CtS diagonals:";
  for (int d : basis.GetCtSDiagonals()) std::cout << " " << d;
  std::cout << std::endl;
  EXPECT_GT(stats.Bits(), 20.0) << "the tower CtS did not read the "
                                   "coordinates";
}

TEST_P(CiSinCBasisTest, HalfBootTowerReturnsTheMessage) {
  if (!param_->conjugate_invariant_) GTEST_SKIP() << "R+ only";
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr) << "the preset must carry a bootstrap";
  const int n = param_->MaxNumSlots();
  const auto &bp = boot->GetBootParameter();
  const char *secret_env = std::getenv("CHEDDAR_MODULE_SPARSE_SECRET");
  std::cout << "sparse secret: " << (secret_env ? secret_env : "off")
            << "; boot top " << bp.GetMaxLevel() << ", CtS levels "
            << bp.num_cts_levels_ << ", EvalMod " << bp.GetNumEvalModLevels()
            << " levels ending at " << bp.GetEvalModEndLevel() << std::endl;

  typename CiSinCBasis<word>::Phases ph;
  if (bp.num_cts_levels_ == 4) {
    ph.cts_inner = {4, 3};  // the inner twist as a pair chain
  } else if (bp.num_cts_levels_ != 3) {
    GTEST_SKIP() << "the tower CtS' spends three (or four) levels";
  }
  CiSinCBasis<word> basis(n, kSmallDegree, kSubDegree);
  boot->PrepareEvalMod();
  const int forward_level = 3;
  basis.PrepareForward(context_, "p", forward_level);
  basis.PrepareCtS(context_, bp.GetCtSStartLevel(), n * boot->GetCtSConst(),
                   ph);
  basis.PreparePrefix(context_, bp.GetEvalModEndLevel());
  EvkRequest req;
  basis.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  // The ride: EvalMod's error is a cubic in the message (CLAUDE.md section
  // 3), and the leg's scores ride at 0.35, the FFN at 0.2.
  constexpr double kRide = 0.2;
  std::vector<Complex> draw;
  GenerateRandomMessage(draw, n, -kRide, kRide, /*complex=*/false);
  std::vector<double> z(n);
  for (int s = 0; s < n; s++) z[s] = draw[s].real();
  Plaintext<word> pt;
  context_->encoder_.Encode(pt, forward_level, param_->GetScale(forward_level),
                            draw);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  Ciphertext<word> sinc, down, half, out;
  basis.Forward("p", sinc, ct, interface_->GetEvkMap());
  boot->LevelDown(down, sinc, 0);
  Profile::Reset();
  __ProfileStart("HalfBootTower", 1, );
  boot->HalfBootTower(half, down, interface_->GetEvkMap(), basis);
  __ProfileEnd("HalfBootTower");
  __ProfileStart("prefix", 1, );
  basis.Prefix(out, half, interface_->GetEvkMap());
  __ProfileEnd("prefix");
  Profile::Report("the fused return, warm-up included");
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::cout << "landed at level " << param_->NPToLevel(out.GetNP())
            << " (HalfBootTower at " << param_->NPToLevel(half.GetNP())
            << "), scale 2^" << std::log2(out.GetScale()) << std::endl;

  std::vector<Complex> got;
  DecryptAndDecode(got, out);
  std::vector<double> obtained(n);
  for (int s = 0; s < n; s++) obtained[s] = got[s].real();

  // The boundary constant fitted off the slots against the derived message
  // ratio; slots whose wrap-around left EvalMod's range counted apart, the
  // rest refitted and reported. The bootstrap reads its input against the
  // base scale, and the forward preserved the level-3 scale it was handed,
  // so the message arrives times `carried` (1.5bu's factor, which the leg's
  // softmax divides out): the derived constant is `ratio * carried`.
  const double carried = down.GetScale() / param_->base_scale_;
  const double ratio = boot->GetMessageRatio() * carried;
  std::cout << "carried " << carried << " (the input's scale over the base "
            << "scale), derived ratio x carried " << ratio << std::endl;
  double num = 0.0, den = 0.0;
  for (int s = 0; s < n; s++) {
    num += z[s] * obtained[s];
    den += z[s] * z[s];
  }
  const double c = num / den;
  int outliers = 0;
  std::vector<double> e2, o2;
  int over[3] = {0, 0, 0};
  for (int s = 0; s < n; s++) {
    const double e = std::abs(obtained[s] / ratio - z[s]);
    if (e > 0.05) over[0]++;
    if (e > 1.0) over[1]++;
    if (e > 1e3) over[2]++;
    if (e > 0.05) {
      outliers++;
      continue;
    }
    e2.push_back(z[s]);
    o2.push_back(obtained[s] / ratio);
  }
  std::cout << "|err| against the derived ratio: > 0.05: " << over[0]
            << ", > 1: " << over[1] << ", > 1e3: " << over[2] << " of " << n
            << std::endl;
  std::cout << std::scientific << std::setprecision(4) << "fitted constant "
            << c << ", derived message ratio " << ratio << ", fit/derived "
            << std::fixed << std::setprecision(5) << c / ratio << std::endl;
  std::vector<double> scaled(n);
  for (int s = 0; s < n; s++) scaled[s] = obtained[s] / c;
  const auto all = Compare(z, scaled);
  Report("fused return: slots vs the message (all slots, fitted)", all);
  double in_range_bits = all.Bits();
  if (outliers > 0 && outliers < n / 2) {
    const auto in_range = Compare(e2, o2);
    Report("fused return: in-range slots against the derived ratio", in_range);
    in_range_bits = in_range.Bits();
  }
  std::cout << "prefix diagonals:";
  for (int d : basis.GetPrefixDiagonals()) std::cout << " " << d;
  std::cout << "; CtS diagonals:";
  for (int d : basis.GetCtSDiagonals()) std::cout << " " << d;
  std::cout << std::endl;

  EXPECT_EQ(outliers, 0) << "a slot left EvalMod's range";
  EXPECT_GT(in_range_bits, 10.0) << "the fused return is not at precision";
  EXPECT_NEAR(c / ratio, 1.0, 5e-3) << "the fitted constant is not the "
                                       "derived message ratio";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, CiSinCBasisTest, testing::Values("ci16_35_land17c3e10.json"),
    [](const testing::TestParamInfo<CiSinCBasisTest::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
