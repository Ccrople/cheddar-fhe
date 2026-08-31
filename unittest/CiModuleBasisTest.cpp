// The slot <-> coefficient transforms in the MODULE basis of R+ (Doing.md
// 3.5), on the device, at the layer's own shape: degree 65536, T = 128,
// k = 512, ci16_35.
//
// What the host script (reference/scripts/ci_module_basis.py) established is
// that these transforms are the library's own butterfly stages with a
// real-part truncation (StC) and one pair correction (CtS). What only a
// device can say is what they cost in noise and time at full size. Two tests:
//
//   SlotToCoeffLandsTheModuleCoordinates -- random slots through the module
//       StC; the decrypted NATIVE coefficients are scanned on the host
//       (P^-1, the same recurrence as CiModDecompScan) and must be the slots,
//       bit-reversed. Then the device's own ModDecomp -> ModPack round trip
//       on that element, scanned again: the projection's decomposition sees
//       clean full-density components.
//
//   CoeffToSlotReadsTheModuleCoordinates -- an element built from random
//       module coordinates (its native image is the banded P(y)) through the
//       module CtS must land y in the slots. The native CtS on the SAME
//       ciphertext is the control: it lands the banded image, at its own
//       noise, so the two noise figures are comparable.
//
// The reference is the host: EncodeCoeff/DecodeCoeff and the scan, never
// another run of a transform.

#include <cmath>
#include <iomanip>

#include "Testbed.h"
#include "core/Mlwe.h"
#include "extension/CiModuleBasis.h"
#include "extension/EvalSpecialFFT.h"
#include "extension/Profile.h"

using word = uint32_t;

namespace {

constexpr int kSmallDegree = 128;

// P^-1: x[i][t] = r[tk + i] - x[k-i][t+1], the i = 0 class pure. Flat
// index t * k + i both ways.
std::vector<double> HostScan(const std::vector<double> &r, int k, int T) {
  std::vector<double> x(r.size(), 0.0);
  for (int t = 0; t < T; t++) x[t * k] = r[t * k];
  for (int i = 1; i <= k / 2; i++) {
    const int mi = k - i;
    double acc_i = 0.0, acc_m = 0.0;
    for (int t = T - 1; t >= 0; t--) {
      const double ni = r[t * k + i] - acc_m;
      const double nm = r[t * k + mi] - acc_i;
      x[t * k + i] = ni;
      x[t * k + mi] = nm;
      acc_i = ni;
      acc_m = nm;
    }
  }
  return x;
}

// P: r[tk + i] = x[i][t] + [i != 0] x[k-i][t+1].
std::vector<double> HostRecompose(const std::vector<double> &x, int k, int T) {
  std::vector<double> r(x.size(), 0.0);
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < k; i++) {
      double v = x[t * k + i];
      if (i != 0 && t + 1 < T) v += x[(t + 1) * k + (k - i)];
      r[t * k + i] = v;
    }
  }
  return r;
}

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

}  // namespace

class CiModuleBasisTest : public Testbed<word> {
 protected:
  // EvalSpecialFFT and CiModuleBasis only need a Context.
  bool UseBootContext() const override { return false; }
};

TEST_P(CiModuleBasisTest, SlotToCoeffLandsTheModuleCoordinates) {
  if (!param_->conjugate_invariant_) GTEST_SKIP() << "R+ only";
  const int n = param_->MaxNumSlots();
  const int T = kSmallDegree;
  const int k = n / T;
  const int stc_level = default_encryption_level_;

  CiModuleBasis<word> basis(context_, T, stc_level, /*cts_level=*/-1);
  EvkRequest req;
  basis.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> slots;
  GenerateRandomMessage(slots, n, -1.0, 1.0, /*complex=*/false);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, slots, stc_level);

  Ciphertext<word> out;
  Profile::Reset();
  __ProfileStart("module SlotToCoeff", 1, );
  basis.EvaluateStC(context_, out, ct, interface_->GetEvkMap());
  __ProfileEnd("module SlotToCoeff");
  Profile::Report("module SlotToCoeff breakdown, warm-up included");
  ASSERT_EQ(param_->NPToLevel(out.GetNP()), stc_level - basis.GetStCNumLevels());

  // Slot s carries module coordinate BitReverse(s).
  std::vector<double> expected(n, 0.0);
  for (int s = 0; s < n; s++) {
    expected[basis.ModuleIndexOfSlot(s)] = slots[s].real();
  }

  Plaintext<word> pt;
  interface_->Decrypt(pt, out);
  std::vector<double> native;
  context_->encoder_.DecodeCoeff(native, pt);
  ASSERT_EQ(static_cast<int>(native.size()), n);

  // 1. The native image is the banded P(expected) ...
  const auto banded = Compare(HostRecompose(expected, k, T), native);
  Report("StC: native image vs P(slots)", banded);
  // 2. ... and its scan is the slots themselves, full density, no duplicate.
  const auto scanned = Compare(expected, HostScan(native, k, T));
  Report("StC: host scan vs slots", scanned);
  EXPECT_LT(scanned.max_abs, 1e-3);

  // 3. The device's own decomposition and recomposition on that element, at
  //    the level the projection packs at.
  const int pack_level = std::min(2, param_->max_level_);
  Ciphertext<word> low;
  context_->LevelDown(low, out, pack_level);
  interface_->PrepareModPackKeys(T, pack_level);
  std::vector<const EvaluationKey<word> *> keys;
  for (int j = 0; j < k; j++) keys.push_back(&interface_->GetModPackKey(k, j));
  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  std::vector<MlweCiphertext<word>> parts;
  Ciphertext<word> packed;
  __ProfileStart("ModDecomp + ModPack at rank 512", 0, );
  mlwe.ModDecomp(parts, low, T);
  mlwe.ModPack(context_, packed, parts, keys);
  __ProfileEnd("ModDecomp + ModPack at rank 512");
  ASSERT_EQ(static_cast<int>(parts.size()), k);

  interface_->Decrypt(pt, packed);
  std::vector<double> native2;
  context_->encoder_.DecodeCoeff(native2, pt);
  const auto round_trip = Compare(expected, HostScan(native2, k, T));
  Report("StC -> ModDecomp -> ModPack: host scan vs slots", round_trip);
  EXPECT_LT(round_trip.max_abs, 1e-3);
}

TEST_P(CiModuleBasisTest, CoeffToSlotReadsTheModuleCoordinates) {
  if (!param_->conjugate_invariant_) GTEST_SKIP() << "R+ only";
  const int n = param_->MaxNumSlots();
  const int T = kSmallDegree;
  const int k = n / T;
  const int log_n = Log2Ceil(n);
  const int cts_level = param_->max_level_;

  CiModuleBasis<word> basis(context_, T, /*stc_level=*/-1, cts_level);
  // The control: the native CtS, compiled at the same level, answer times 1.
  const BootParameter boot_param(param_->max_level_, num_cts_levels_,
                                 num_stc_levels_, 5, 0);
  ASSERT_EQ(boot_param.GetCtSStartLevel(), cts_level);
  EvalSpecialFFT<word> native_fft(context_, boot_param, n, 1.0 / n, 1.0);

  EvkRequest req;
  basis.AddRequiredRotations(req);
  native_fft.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  // Random module coordinates; the element's native image is P(y).
  std::vector<Complex> draw;
  GenerateRandomMessage(draw, n, -1.0, 1.0, /*complex=*/false);
  std::vector<double> y(n);
  for (int i = 0; i < n; i++) y[i] = draw[i].real();
  const auto image = HostRecompose(y, k, T);

  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, cts_level, DetermineScale(cts_level),
                                 image);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  Ciphertext<word> out;
  Profile::Reset();
  __ProfileStart("module CoeffToSlot", 1, );
  basis.EvaluateCtS(context_, out, ct, interface_->GetEvkMap());
  __ProfileEnd("module CoeffToSlot");
  Profile::Report("module CoeffToSlot breakdown, warm-up included");
  ASSERT_EQ(param_->NPToLevel(out.GetNP()), cts_level - basis.GetCtSNumLevels());

  std::vector<Complex> got;
  DecryptAndDecode(got, out);
  std::vector<double> expected(n), obtained(n);
  for (int s = 0; s < n; s++) {
    expected[s] = y[basis.ModuleIndexOfSlot(s)];
    obtained[s] = got[s].real();
  }
  const auto module = Compare(expected, obtained);
  Report("module CtS: slots vs module coordinates", module);
  EXPECT_LT(module.max_abs, 1e-3);

  Ciphertext<word> out_native;
  Profile::Reset();
  __ProfileStart("native CoeffToSlot (control)", 1, );
  native_fft.EvaluateCtS(context_, out_native, ct, interface_->GetEvkMap());
  __ProfileEnd("native CoeffToSlot (control)");
  Profile::Report("native CoeffToSlot breakdown, warm-up included");

  DecryptAndDecode(got, out_native);
  for (int s = 0; s < n; s++) {
    expected[s] = image[BitReverseInt(s, log_n)];
    obtained[s] = got[s].real();
  }
  const auto native = Compare(expected, obtained);
  Report("native CtS (control): slots vs banded image", native);
  EXPECT_LT(native.max_abs, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, CiModuleBasisTest, testing::Values("ci16_35.json"),
    [](const testing::TestParamInfo<CiModuleBasisTest::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
