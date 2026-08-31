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

#include <cstdlib>
#include <cstring>

#include "Testbed.h"
#include "core/Mlwe.h"
#include "extension/BootContext.h"
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
  // CHEDDAR_TEST_CTS_LEVELS changes its phase count (the preset's is four),
  // which is how a three-phase native CtS is checked outside a bootstrap.
  const char *cts_env = std::getenv("CHEDDAR_TEST_CTS_LEVELS");
  const int native_cts_levels =
      (cts_env != nullptr && cts_env[0] != 0) ? std::atoi(cts_env)
                                              : num_cts_levels_;
  std::cout << "native CtS phases: " << native_cts_levels << std::endl;
  const BootParameter boot_param(param_->max_level_, native_cts_levels,
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

// The half bootstrap with its CoeffToSlot in the module basis: coefficient
// image P(y) at level 0 in, the module coordinates y in the slots out. Two
// knobs decide whether it can work and the test says which it ran with:
// CHEDDAR_MODULE_SPARSE_SECRET=128 samples the SSE secret sparse in the
// module basis (Doing.md 3.5, check 6: the wrap-around std 4.0 against 17.7
// under a native-sparse secret) and CHEDDAR_BOOT_DOUBLE_ANGLE=4 widens
// EvalMod's range to hold that std (K = 32 against 16). The native HalfBoot
// on the same input is the control, with the banded image as its answer.
// The ladder must close on default_encryption_level: with four double angles
// EvalMod takes nine levels, so CoeffToSlot takes three -- one real twist
// phase and a two-phase small chain -- and with the default three it takes
// the preset's four.
namespace {
int DoubleAngles() {
  const char *env = std::getenv("CHEDDAR_BOOT_DOUBLE_ANGLE");
  return (env != nullptr && env[0] != 0) ? std::atoi(env) : 3;
}
}  // namespace

// CHEDDAR_TEST_CTS_LEVELS picks CoeffToSlot's level count (default: the
// preset's), and the boot's top level is placed so that EvalMod still ends
// on default_encryption_level: 19 + (5 + double angles) + CtS levels.
class CiModuleBoot : public Testbed<word> {
 protected:
  int BootCtsLevels() const override {
    const char *env = std::getenv("CHEDDAR_TEST_CTS_LEVELS");
    if (env != nullptr && env[0] != 0) return std::atoi(env);
    return num_cts_levels_;
  }
  int BootMaxLevel() const override {
    const int top = default_encryption_level_ + 5 + DoubleAngles() +
                    BootCtsLevels();
    return std::min(top, param_->max_level_);
  }
  CiModuleBasis<word>::Phases BootPhases() const {
    CiModuleBasis<word>::Phases phases;
    if (BootCtsLevels() == 3) {
      phases.cts_twist = {9};
      phases.cts_small = {4, 3};
    } else {
      phases.cts_twist = {5, 4};
      phases.cts_small = {4, 3};
    }
    return phases;
  }
};

TEST_P(CiModuleBoot, HalfBootReadsTheModuleCoordinates) {
  if (!param_->conjugate_invariant_) GTEST_SKIP() << "R+ only";
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr) << "the preset must carry a bootstrap";
  const int n = param_->MaxNumSlots();
  const int T = kSmallDegree;
  const int k = n / T;
  const int log_n = Log2Ceil(n);

  const char *secret_env = std::getenv("CHEDDAR_MODULE_SPARSE_SECRET");
  const bool module_secret = secret_env != nullptr && secret_env[0] != 0;
  const int double_angle = DoubleAngles();
  std::cout << "module-sparse secret: " << (module_secret ? secret_env : "off")
            << ", EvalMod double angles: " << double_angle
            << " (K = " << (2 << double_angle) << ")" << std::endl;

  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(n);
  const BootParameter boot_param(BootMaxLevel(), BootCtsLevels(),
                                 num_stc_levels_, 5, 0);
  std::cout << "boot top level " << BootMaxLevel() << ", CtS levels "
            << BootCtsLevels() << ", EvalMod ends at "
            << boot_param.GetEvalModEndLevel() << std::endl;
  CiModuleBasis<word> basis(context_, T, /*stc_level=*/-1,
                            boot_param.GetCtSStartLevel(), BootPhases(), 1.0,
                            n * boot->GetCtSConst());
  EvkRequest req;
  boot->AddRequiredRotations(req, n);
  basis.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  // The ride: EvalMod's error is a cubic in the message over the wrap
  // (CLAUDE.md section 3, -0.00258 m^3), 2.6e-3 at |y| = 1 and 2e-5 at the
  // FFN's 0.2, which is what the layer's crossings carry.
  constexpr double kRide = 0.2;
  std::vector<Complex> draw;
  GenerateRandomMessage(draw, n, -kRide, kRide, /*complex=*/false);
  std::vector<double> y(n);
  for (int i = 0; i < n; i++) y[i] = draw[i].real();
  const auto image = HostRecompose(y, k, T);
  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, 0, param_->GetScale(0), image);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  // The boundary constant is fitted off the decrypted slots (1.5bz) and
  // compared with the derived message ratio; the error is reported after
  // dividing by the fit.
  struct Landing {
    double max_abs = 0.0;
    int outliers = 0;
    double in_range_bits = 0.0;
  };
  auto fit_and_report = [&](const std::string &name,
                            const std::vector<double> &expected,
                            const std::vector<double> &obtained) {
    double num = 0.0, den = 0.0;
    for (int s = 0; s < n; s++) {
      num += expected[s] * obtained[s];
      den += expected[s] * expected[s];
    }
    const double c = num / den;
    std::vector<double> scaled(n);
    for (int s = 0; s < n; s++) scaled[s] = obtained[s] / c;
    auto stats = Compare(expected, scaled);
    // A slot whose wrap-around left EvalMod's range comes back as a huge
    // number: count those apart -- against the DERIVED ratio, since one of
    // them is enough to wreck the fit -- and refit and report on the rest,
    // so that a range failure reads as "N slots out of range".
    const double ratio = boot->GetMessageRatio();
    auto in_range = [&](int s) {
      return std::abs(obtained[s] / ratio - expected[s]) <= 0.05;
    };
    int outliers = 0;
    double num2 = 0.0, den2 = 0.0;
    for (int s = 0; s < n; s++) {
      if (!in_range(s)) {
        outliers++;
        continue;
      }
      num2 += expected[s] * obtained[s];
      den2 += expected[s] * expected[s];
    }
    {
      int over[4] = {0, 0, 0, 0};
      for (int s = 0; s < n; s++) {
        const double e = std::abs(obtained[s] / ratio - expected[s]);
        if (e > 0.05) over[0]++;
        if (e > 1.0) over[1]++;
        if (e > 1e3) over[2]++;
        if (e > 1e6) over[3]++;
      }
      std::cout << "[" << name << "] |err| against the derived ratio: > 0.05: "
                << over[0] << ", > 1: " << over[1] << ", > 1e3: " << over[2]
                << ", > 1e6: " << over[3] << " of " << n << std::endl;
    }
    Landing landing;
    landing.outliers = outliers;
    if (outliers > 0 && outliers < n / 2) {
      const double c2 = num2 / den2;
      std::vector<double> e2, o2;
      for (int s = 0; s < n; s++) {
        if (!in_range(s)) continue;
        e2.push_back(expected[s]);
        o2.push_back(obtained[s] / c2);
      }
      std::cout << "[" << name << "] " << outliers << " slots out of range; "
                << "the other " << e2.size() << " refit at " << std::scientific
                << c2 << ":" << std::endl;
      const auto in_range_stats = Compare(e2, o2);
      Report(name + ", in-range slots", in_range_stats);
      landing.in_range_bits = in_range_stats.Bits();
    } else if (outliers == 0) {
      landing.in_range_bits = stats.Bits();
    }
    std::cout << std::scientific << std::setprecision(4) << "[" << name
              << "] fitted constant " << c << ", derived message ratio "
              << boot->GetMessageRatio() << ", fit/derived "
              << std::fixed << std::setprecision(5)
              << c / boot->GetMessageRatio() << std::endl;
    Report(name, stats);
    landing.max_abs = stats.max_abs;
    return landing;
  };

  std::vector<Complex> got;
  std::vector<double> expected(n), obtained(n);

  Ciphertext<word> out;
  Profile::Reset();
  __ProfileStart("module HalfBoot", 1, );
  boot->HalfBootModule(out, ct, interface_->GetEvkMap(), basis);
  __ProfileEnd("module HalfBoot");
  Profile::Report("module HalfBoot breakdown, warm-up included");
  DecryptAndDecode(got, out);
  for (int s = 0; s < n; s++) {
    expected[s] = y[basis.ModuleIndexOfSlot(s)];
    obtained[s] = got[s].real();
  }
  const auto module = fit_and_report("module HalfBoot: slots vs y", expected,
                                     obtained);

  Ciphertext<word> out_native;
  Profile::Reset();
  __ProfileStart("native HalfBoot (control)", 1, );
  boot->HalfBoot(out_native, ct, interface_->GetEvkMap());
  __ProfileEnd("native HalfBoot (control)");
  Profile::Report("native HalfBoot breakdown, warm-up included");
  DecryptAndDecode(got, out_native);
  for (int s = 0; s < n; s++) {
    expected[s] = image[BitReverseInt(s, log_n)];
    obtained[s] = got[s].real();
  }
  const auto native = fit_and_report("native HalfBoot (control): slots vs P(y)",
                                     expected, obtained);

  // What is asserted depends on the knobs. Measured on the A100 (Doing.md
  // 3.6): with the module-sparse secret at h = 16 the module HalfBoot lands
  // every slot at 13.8 bits; at h = 32 the module wrap-around reaches 18 and
  // 5-8 slots of 65536 leave EvalMod's K = 16, which only a wider range can
  // fix (r = 4 is not supported by the library's EvalMod). Without the module
  // secret the module route cannot work at all, and is only reported.
  int module_h = param_->GetSparseHammingWeight();
  if (module_secret) {
    const char *comma = std::strchr(secret_env, ',');
    if (comma != nullptr && std::atoi(comma + 1) > 0) module_h = std::atoi(comma + 1);
  }
  if (module_secret && module_h <= 16) {
    EXPECT_EQ(module.outliers, 0) << "a slot left EvalMod's range";
    EXPECT_GT(module.in_range_bits, 12.0)
        << "the module HalfBoot is not at precision";
  } else if (module_secret) {
    std::cout << "(h = " << module_h << ": " << module.outliers
              << " slots past K = 16 this draw -- reported, not asserted; "
              << "the range is the open item)" << std::endl;
  }
  EXPECT_LT(native.max_abs, 1e-2) << "the native control did not land";
}

// The module route taken apart, each piece against the host, so that a
// failure of the half bootstrap names its stage:
//   1. the in-place scan and recomposition are inverse on device residues;
//   2. the module-centred lift decrypts to the same message as the native
//      lift up to q0 times an integer polynomial, and that integer is small
//      in MODULE coordinates (the native lift's is small in native ones);
//   3. the module CtS with the boot constant, on the lifted ciphertext, lands
//      what the host computes from the decrypted coefficients.
TEST_P(CiModuleBoot, ModuleLiftCentresTheRepresentatives) {
  if (!param_->conjugate_invariant_) GTEST_SKIP() << "R+ only";
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int n = param_->MaxNumSlots();
  const int T = kSmallDegree;
  const int k = n / T;
  const int degree = param_->degree_;
  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  const char *secret_env = std::getenv("CHEDDAR_MODULE_SPARSE_SECRET");
  const bool module_secret = secret_env != nullptr && secret_env[0] != 0;
  std::cout << "module-sparse secret: " << (module_secret ? secret_env : "off")
            << std::endl;

  // 1. scan / recompose round trip on random residues, every limb.
  {
    const NPInfo np = param_->LevelToNP(-1);
    const auto primes = param_->GetPrimeVector(np);
    const int limbs = np.GetNumTotal();
    HostVector<word> host(limbs * degree);
    for (int l = 0; l < limbs; l++) {
      for (int c = 0; c < degree; c++) {
        host[l * degree + c] =
            static_cast<word>((static_cast<uint64_t>(c) * 2654435761u + l * 97u) %
                              primes[l]);
      }
    }
    DeviceVector<word> dev(limbs * degree);
    CopyHostToDevice(dev, host);
    auto view = dev.View(0);
    mlwe.ScanInPlace(view, np, T);
    HostVector<word> scanned(limbs * degree);
    CopyDeviceToHost(scanned, dev);
    mlwe.RecomposeInPlace(view, np, T);
    HostVector<word> back(limbs * degree);
    CopyDeviceToHost(back, dev);
    size_t mismatches = 0, moved = 0;
    for (size_t i = 0; i < host.size(); i++) {
      if (back[i] != host[i]) mismatches++;
      if (scanned[i] != host[i]) moved++;
    }
    std::cout << "[scan/recompose] " << mismatches << " mismatches of "
              << host.size() << ", " << moved << " entries changed by the scan"
              << std::endl;
    EXPECT_EQ(mismatches, 0u);
    EXPECT_GT(moved, host.size() / 2);
  }

  // 2. the two lifts of one level-zero ciphertext.
  constexpr double kRide = 0.2;
  std::vector<Complex> draw;
  GenerateRandomMessage(draw, n, -kRide, kRide, /*complex=*/false);
  std::vector<double> y(n);
  for (int i = 0; i < n; i++) y[i] = draw[i].real();
  const auto image = HostRecompose(y, k, T);
  const double scale = param_->GetScale(0);
  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, 0, scale, image);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  const BootParameter boot_param(BootMaxLevel(), BootCtsLevels(),
                                 num_stc_levels_, 5, 0);
  const int top = boot_param.GetMaxLevel();
  Ciphertext<word> up_native, up_module;
  boot->ModUpToLevel(up_native, ct, interface_->GetEvkMap(), top);
  boot->ModUpToLevel(up_module, ct, interface_->GetEvkMap(), top, T);

  double q0 = 1.0;
  for (word p : param_->GetPrimeVector(param_->LevelToNP(-1))) q0 *= p;

  auto lifted = [&](const Ciphertext<word> &up, const std::string &name) {
    Plaintext<word> out;
    interface_->Decrypt(out, up);
    std::vector<double> coeffs;
    context_->encoder_.DecodeCoeff(coeffs, out);
    // (decrypted - message) / q0 must be an integer polynomial; report its
    // size in native and in module coordinates.
    std::vector<double> wrap(n);
    double non_integer = 0.0, max_native = 0.0;
    for (int c = 0; c < n; c++) {
      wrap[c] = (coeffs[c] - image[c]) * scale / q0;
      non_integer = std::max(non_integer,
                             std::abs(wrap[c] - std::round(wrap[c])));
      max_native = std::max(max_native, std::abs(wrap[c]));
    }
    const auto module = HostScan(wrap, k, T);
    double max_module = 0.0;
    for (double v : module) max_module = std::max(max_module, std::abs(v));
    std::cout << std::fixed << std::setprecision(3) << "[" << name
              << "] wrap-around: max |native| " << max_native
              << ", max |module| " << max_module << ", non-integer part "
              << std::scientific << non_integer << std::endl;
    EXPECT_LT(non_integer, 1e-2) << name;
    return coeffs;
  };
  lifted(up_native, "native lift");
  const auto coeffs_module = lifted(up_module, "module lift");

  // The same two lifts of the ciphertext the half bootstrap actually lifts:
  // scaled up by 2^log_scaleup so the message rides at 2^-5 of q0. Encoded
  // at that scale directly, which is what MultUnsafe(scaleup_const_) makes.
  {
    const int log_scaleup = static_cast<int>(std::round(std::log2(q0))) -
                            static_cast<int>(std::round(std::log2(scale))) - 5;
    const double up_scale = scale * std::ldexp(1.0, log_scaleup);
    Plaintext<word> pt_up;
    context_->encoder_.EncodeCoeff(pt_up, 0, up_scale, image);
    Ciphertext<word> ct_up, a, b;
    interface_->Encrypt(ct_up, pt_up);
    boot->ModUpToLevel(a, ct_up, interface_->GetEvkMap(), top);
    boot->ModUpToLevel(b, ct_up, interface_->GetEvkMap(), top, T);
    for (auto pr : {std::make_pair(&a, "native lift, scaled up"),
                    std::make_pair(&b, "module lift, scaled up")}) {
      Plaintext<word> out;
      interface_->Decrypt(out, *pr.first);
      std::vector<double> coeffs;
      context_->encoder_.DecodeCoeff(coeffs, out);
      std::vector<double> wrap(n);
      double max_native = 0.0;
      for (int c = 0; c < n; c++) {
        wrap[c] = (coeffs[c] - image[c]) * up_scale / q0;
        max_native = std::max(max_native, std::abs(wrap[c]));
      }
      const auto module = HostScan(wrap, k, T);
      double max_module = 0.0;
      int over16 = 0;
      for (double v : module) {
        max_module = std::max(max_module, std::abs(v));
        if (std::abs(v) > 16.0) over16++;
      }
      std::cout << std::fixed << std::setprecision(3) << "[" << pr.second
                << "] wrap-around: max |native| " << max_native
                << ", max |module| " << max_module << ", module > 16: "
                << over16 << std::endl;
    }
  }

  // 3. the module CtS on the module-lifted ciphertext, boot constant in.
  CiModuleBasis<word> basis(context_, T, /*stc_level=*/-1, top, BootPhases(),
                            1.0, n * boot->GetCtSConst());
  EvkRequest req;
  basis.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);
  Ciphertext<word> slots_ct;
  basis.EvaluateCtS(context_, slots_ct, up_module, interface_->GetEvkMap());
  std::vector<Complex> got;
  DecryptAndDecode(got, slots_ct);
  const auto module_coords = HostScan(coeffs_module, k, T);
  const double factor = n * boot->GetCtSConst() * scale / slots_ct.GetScale();
  std::vector<double> expected(n), obtained(n);
  for (int s = 0; s < n; s++) {
    expected[s] = module_coords[basis.ModuleIndexOfSlot(s)] * n *
                  boot->GetCtSConst();
    obtained[s] = got[s].real();
  }
  double num = 0.0, den = 0.0;
  for (int s = 0; s < n; s++) {
    num += expected[s] * obtained[s];
    den += expected[s] * expected[s];
  }
  std::cout << "[module CtS on the lifted ciphertext] fitted expected->obtained "
            << std::scientific << num / den << " (declared-scale factor "
            << factor << ")" << std::endl;
  for (int s = 0; s < n; s++) obtained[s] /= (num / den);
  Report("module CtS on the lifted ciphertext, after the fit",
         Compare(expected, obtained));
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, CiModuleBoot, testing::Values("ci16_35.json"),
    [](const testing::TestParamInfo<CiModuleBoot::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });

INSTANTIATE_TEST_SUITE_P(
    Cheddar, CiModuleBasisTest, testing::Values("ci16_35.json"),
    [](const testing::TestParamInfo<CiModuleBasisTest::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
