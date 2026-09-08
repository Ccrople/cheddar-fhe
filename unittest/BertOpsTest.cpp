// The two operators BERT has and Llama does not: LayerNorm and GELU.
//
// Both are measured against a HOST reference computed in double, never against
// another run of the same code, and both are measured on the packing the layer
// will use -- 128 tokens fastest, a declared channel width wider than the live
// one (BERT's 768 in a declared 1024), and the conjugate-invariant ring's
// `degree` real slots rather than `degree / 2` complex ones.
//
// WHAT EACH TEST IS FOR
//
//   1. GeLuSaturationIsExact          the design's premise, in double: GELU is
//                                     its own asymptote to 1.3e-04 past +-4,
//                                     so the outlier slots need no polynomial
//   2. GeLuOnEncrypted                the circuit, on a message that carries
//                                     both bulk and saturated slots, against
//                                     true GELU
//   3. LayerNormOnEncrypted           the circuit against the host formula
//   4. LayerNormMaskedMeanIsNotAnOptimisation
//                                     the contract of `LayerNorm.h`: with the
//                                     dead channels centred by a CONSTANT
//                                     instead of a mask, the variance collects
//                                     `(declared - live) * mu^2` that does not
//                                     exist. Measured in double, because the
//                                     point is the size of the error, not
//                                     whether the ciphertext survives it.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "Testbed.h"
#include "extension/GeLu.h"
#include "extension/LayerNorm.h"

using word = uint32_t;

namespace {

constexpr int kTokens = 128;
constexpr int kLive = 768;       // BERT-Base's model width
constexpr int kDeclared = 1024;  // two rank-512 ciphertexts
constexpr double kEps = 1e-12;

double TrueGeLu(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double Bits(double err, double ref) {
  if (err <= 0.0) return 99.0;
  return -std::log2(err / ref);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The premise, in double.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, GeLuSaturationIsExact) {
  for (double tau : {3.0, 4.0, 5.0, 6.0}) {
    double pos = 0.0, neg = 0.0;
    for (int i = 0; i <= 2000; i++) {
      const double u = tau + 200.0 * i / 2000.0;
      pos = std::max(pos, std::abs(TrueGeLu(u) - u));
      neg = std::max(neg, std::abs(TrueGeLu(-u)));
    }
    std::cout << "  tau " << tau << ": |GELU(u) - u| <= " << pos
              << " for u > tau, |GELU(-u)| <= " << neg << std::endl;
    if (tau >= 4.0) {
      EXPECT_LT(pos, 2e-4) << "the identity answer for the positive saturated "
                              "group is only exact if this is small";
      EXPECT_LT(neg, 2e-4);
    }
  }
  // And the fit that covers the rest: degree 31 over +-8, which is the plan
  // `reference_forward_bert.py` writes.
  const int level = default_encryption_level_;
  std::vector<GeLuHandler<word>::Group> groups = {
      {GeLuHandler<word>::Kind::kFit, 8.0, 31},
      {GeLuHandler<word>::Kind::kIdentity, 8.0, 0},
      {GeLuHandler<word>::Kind::kZero, 8.0, 0}};
  GeLuHandler<word> gelu(context_, groups, level);
  double err = 0.0;
  for (int i = 0; i <= 4000; i++) {
    const double u = -4.0 + 8.0 * i / 4000.0;  // the BULK slots only
    err = std::max(err, std::abs(gelu.PlainGeLu(0, u) - TrueGeLu(u)));
  }
  std::cout << "  bulk fit (+-8, degree 31) max |err| on |u| <= 4: " << err
            << std::endl;
  EXPECT_LT(err, 1e-3);
}

// ---------------------------------------------------------------------------
// 2. The circuit.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, GeLuOnEncrypted) {
  const int level = default_encryption_level_;
  const int slots = param_->MaxNumSlots();
  constexpr double kTau = 4.0;
  constexpr double kRange = 2.0 * kTau;
  constexpr int kDegree = 31;

  std::vector<GeLuHandler<word>::Group> groups = {
      {GeLuHandler<word>::Kind::kFit, kRange, kDegree},
      {GeLuHandler<word>::Kind::kIdentity, kRange, 0},
      {GeLuHandler<word>::Kind::kZero, kRange, 0}};
  GeLuHandler<word> gelu(context_, groups, level);

  // A message shaped like the real thing: most slots inside the transition,
  // a thousandth of them far outside it in both directions -- which is what
  // the measurement on the checkpoint found (0.03 % to 1 % of slots past 4,
  // overwhelmingly negative).
  std::mt19937_64 rng(20260908);
  std::uniform_real_distribution<double> bulk(-kTau, kTau);
  std::uniform_real_distribution<double> big(kTau, 130.0);
  std::vector<double> u(slots);
  std::vector<std::vector<Complex>> mask(3);
  for (auto &m : mask) m.assign(slots, Complex(0.0, 0.0));
  std::vector<Complex> msg(slots);
  int npos = 0, nneg = 0;
  for (int s = 0; s < slots; s++) {
    const int r = static_cast<int>(rng() % 1000);
    if (r == 0) {
      u[s] = big(rng);
      mask[1][s] = Complex(1.0, 0.0);
      npos++;
    } else if (r == 1) {
      u[s] = -big(rng);
      mask[2][s] = Complex(1.0, 0.0);
      nneg++;
    } else {
      u[s] = bulk(rng);
      mask[0][s] = Complex(1.0, 0.0);
    }
    // The input arrives pre-divided by the range, exactly as SiLU's does: the
    // divisor folds into the intermediate projection's weights for free.
    msg[s] = Complex(u[s] / kRange, 0.0);
  }
  std::cout << "  " << npos << " saturated positive, " << nneg
            << " saturated negative of " << slots << " slots" << std::endl;

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);
  Ciphertext<word> res;
  gelu.Apply(res, ct, mask, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const int out_level = param_->NPToLevel(res.GetNP());
  std::cout << "  level " << level << " -> " << out_level << ", depth "
            << (level - out_level) << std::endl;
  EXPECT_EQ(out_level, gelu.GetOutputLevel());

  std::vector<Complex> got;
  DecryptAndDecode(got, res);
  double err_bulk = 0.0, err_sat = 0.0, ref_bulk = 0.0, ref_sat = 0.0;
  for (int s = 0; s < slots; s++) {
    const double want = TrueGeLu(u[s]);
    const double e = std::abs(got[s].real() - want);
    if (mask[0][s].real() != 0.0) {
      err_bulk = std::max(err_bulk, e);
      ref_bulk = std::max(ref_bulk, std::abs(want));
    } else {
      err_sat = std::max(err_sat, e);
      ref_sat = std::max(ref_sat, std::abs(want));
    }
  }
  std::cout << "  bulk max |err| " << err_bulk << " (" << Bits(err_bulk, ref_bulk)
            << " bits), saturated " << err_sat << " ("
            << Bits(err_sat, ref_sat) << " bits)" << std::endl;
  // Both groups are held to the same ABSOLUTE bound, and that is the honest
  // statement. The bulk is fit limited at degree 31 on +-8. The saturated
  // slots carry no approximation of their own -- they are answered by `u`
  // itself, or by nothing -- but they still pick up the fit's value at zero,
  // because the input mask sets them to zero and the polynomial is evaluated
  // there: |P(0) - GELU(0)| is 9.2e-05 for this fit. Relative to their own
  // magnitudes (up to 130) that is over 18 bits.
  EXPECT_LT(err_bulk, 1e-3);
  EXPECT_LT(err_sat, 1e-3);
}

// ---------------------------------------------------------------------------
// 3. LayerNorm against the host formula.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, LayerNormOnEncrypted) {
  const int level = default_encryption_level_;
  const int slots = param_->MaxNumSlots();
  const int channels_per_ct = slots / kTokens;
  const int num_ct = kDeclared / channels_per_ct;
  ASSERT_GE(num_ct, 1);

  // Data with a per-token spread, so the window is doing work rather than
  // being handed a constant.
  std::mt19937_64 rng(20260909);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::vector<double> x(static_cast<size_t>(kTokens) * kLive);
  std::vector<double> gain(kLive), bias(kLive);
  for (int c = 0; c < kLive; c++) {
    gain[c] = 0.5 + 0.5 * std::abs(gauss(rng));
    bias[c] = 0.1 * gauss(rng);
  }
  for (int t = 0; t < kTokens; t++) {
    // A per-token magnitude spread of 2x, and a per-token OFFSET -- which is
    // the whole reason this operator is not RMSNorm.
    const double m = 1.0 + 1.0 * t / kTokens;
    const double off = 0.5 * std::sin(t * 0.7);
    for (int c = 0; c < kLive; c++) {
      x[static_cast<size_t>(t) * kLive + c] = m * gauss(rng) + off;
    }
  }

  // The host reference, and the calibration read off it exactly as
  // `reference_forward_bert.py` does: alpha at the geometric midpoint of the
  // variance range, the window the range itself with the margin squared.
  std::vector<double> want(static_cast<size_t>(kTokens) * kLive);
  std::vector<double> var(kTokens);
  double want_absmax = 0.0;
  for (int t = 0; t < kTokens; t++) {
    double mu = 0.0;
    for (int c = 0; c < kLive; c++) mu += x[static_cast<size_t>(t) * kLive + c];
    mu /= kLive;
    double v = 0.0;
    for (int c = 0; c < kLive; c++) {
      const double d = x[static_cast<size_t>(t) * kLive + c] - mu;
      v += d * d;
    }
    var[t] = v / kLive;
    const double inv = 1.0 / std::sqrt(var[t] + kEps);
    for (int c = 0; c < kLive; c++) {
      const double y =
          (x[static_cast<size_t>(t) * kLive + c] - mu) * inv * gain[c] + bias[c];
      want[static_cast<size_t>(t) * kLive + c] = y;
      want_absmax = std::max(want_absmax, std::abs(y));
    }
  }
  const double lo = *std::min_element(var.begin(), var.end());
  const double hi = *std::max_element(var.begin(), var.end());
  const double alpha = 1.0 / std::sqrt(lo * hi);
  const double window = std::max(1.5, (hi / lo) * 1.3 * 1.3);
  const int degree = 15;
  std::cout << "  variance " << lo << " .. " << hi << ", alpha " << alpha
            << ", window " << window << ", degree " << degree << std::endl;

  LayerNormHandler<word> ln(context_, kTokens, kDeclared, alpha, level, kEps,
                            window, degree, /*channel_stride=*/1,
                            /*live_channels=*/kLive);
  ASSERT_EQ(ln.GetNumCiphertexts(), num_ct);
  for (int d : ln.GetRotationDistances()) {
    // The level argument is not optional: PrepareRotationKey's default -1 is
    // the dense-to-sparse short base and fails deep inside the rotation.
    interface_->PrepareRotationKey(d, level);
  }

  const double root_alpha = std::sqrt(alpha);
  std::vector<Ciphertext<word>> cts(num_ct);
  std::vector<std::vector<Complex>> wts(num_ct), bs(num_ct), mask(num_ct);
  for (int i = 0; i < num_ct; i++) {
    std::vector<Complex> msg(slots);
    wts[i].assign(slots, Complex(0.0, 0.0));
    bs[i].assign(slots, Complex(0.0, 0.0));
    mask[i].assign(slots, Complex(0.0, 0.0));
    for (int s = 0; s < slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = s % kTokens;
      if (c >= kLive) continue;  // the declared tail is dead, and stays zero
      msg[s] = Complex(x[static_cast<size_t>(t) * kLive + c], 0.0);
      // The gain carries sqrt(alpha) by this class's contract.
      wts[i][s] = Complex(gain[c] * root_alpha, 0.0);
      bs[i][s] = Complex(bias[c], 0.0);
      mask[i][s] = Complex(1.0, 0.0);
    }
    EncodeAndEncrypt(cts[i], msg, level);
  }

  std::vector<Ciphertext<word>> res;
  ln.Apply(res, cts, wts, bs, mask, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), num_ct);
  const int out_level = param_->NPToLevel(res[0].GetNP());
  std::cout << "  level " << level << " -> " << out_level << ", depth "
            << (level - out_level) << " (RMSNorm's + 1 for the centring)"
            << std::endl;
  EXPECT_EQ(out_level, ln.GetOutputLevel());

  double max_err = 0.0, sq = 0.0;
  size_t n = 0;
  for (int i = 0; i < num_ct; i++) {
    std::vector<Complex> got;
    DecryptAndDecode(got, res[i]);
    for (int s = 0; s < slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = s % kTokens;
      if (c >= kLive) continue;
      const double e =
          got[s].real() - want[static_cast<size_t>(t) * kLive + c];
      max_err = std::max(max_err, std::abs(e));
      sq += e * e;
      n++;
    }
  }
  const double rms = std::sqrt(sq / n);
  std::cout << "  max |err| " << max_err << " (" << Bits(max_err, want_absmax)
            << " bits), rms " << rms << std::endl;
  EXPECT_GT(Bits(max_err, want_absmax), 11.0)
      << "LayerNorm should reach [SYLPH] 3.1.2's 12-bit bar the way RMSNorm "
         "does; a miss here is the centring, the mask or the window";
}

// ---------------------------------------------------------------------------
// 4. Why the mean is masked. In double: the point is the size of the error.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, LayerNormMaskedMeanIsNotAnOptimisation) {
  std::mt19937_64 rng(7);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::vector<double> x(kLive);
  const double offset = 0.5;  // a per-token mean, which BERT's stream has
  for (int c = 0; c < kLive; c++) x[c] = gauss(rng) + offset;
  double mu = 0.0;
  for (double v : x) mu += v;
  mu /= kLive;
  double var = 0.0;
  for (double v : x) var += (v - mu) * (v - mu);
  var /= kLive;
  // The unmasked centring leaves `-mu` at every dead declared slot, and the
  // sum of squares that follows divides by the LIVE count -- so it collects
  // `(declared - live) * mu^2 / live` that is not in the data.
  const double bogus = var + (kDeclared - kLive) * mu * mu / kLive;
  std::cout << "  variance " << var << ", unmasked " << bogus << " (+"
            << 100.0 * (bogus / var - 1.0) << " %)" << std::endl;
  EXPECT_GT(bogus / var, 1.05)
      << "if the dead slots cost nothing here then the mask in "
         "LayerNormHandler is dead weight and should go";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    // ci16_35 is the layer's ring; bootparam_35 is the ordinary-ring control,
    // where the same operators run on degree/2 complex slots.
    testing::Values("ci16_35.json", "bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
