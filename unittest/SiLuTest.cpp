// SiLU, x * sigmoid(x), as [SYLPH] evaluates it inside the SwiGLU gate.
//
// THE ORDER OF THESE TESTS IS THE POINT. RMSNorm cost a cycle to a failure
// where the Chebyshev coefficients were exact -- the plaintext oracle came out
// at 7.17e-05, matching the predicted 13.8 bits -- while the encrypted path was
// wrong by 29%, because EvalPoly had been handed a non-canonical input scale.
// A single end-to-end number cannot tell those two apart. So the fit is
// measured first with no ciphertext anywhere near it, and only then does the
// encrypted run get to be interpreted.
//
// WHY THE RANGE IS A TEST PARAMETER AND NOT A CONSTANT. SiLU is not scale
// invariant. RMSNorm(beta*x) = RMSNorm(x), so a badly scaled RMSNorm input is
// free to fix; here the magnitude is the answer, and an argument outside the
// approximation interval is evaluated wrongly with no diagnostic. [SYLPH]
// section 3.1.3 reports degree 31 after calibration against 83 before, "gaining
// 2 levels", and its table 2 puts the calibrated SiLU input at 10.82 -- so
// +-12 is the interval with margin, and degree 31 is what that interval costs.
//
// TARGET. [SYLPH] section 3.1.2: 12 bits of precision matches FP16 perplexity.
//
// AND 2^30 CANNOT REACH IT. Measured here across all three presets, the error
// the circuit adds on top of its own polynomial is a fixed integer magnitude
// divided by the scaling factor -- it falls by very nearly five bits for every
// five bits of scale:
//
//     scale   circuit vs its own polynomial   end to end vs true SiLU
//     2^30    2.74e-03   12.10 bits           11.79 bits   under the bar
//     2^35    1.01e-04   16.86 bits           13.47 bits   fit limited
//     2^40    2.89e-06   21.99 bits           13.54 bits   fit limited
//
// So SiLU becomes fit limited at 2^35 and 2^40 buys nothing more while costing
// three levels (max level 19 against 16). The tests below assert 12 bits from
// 2^35 up and assert that 2^30 falls short, because that is the parameter
// guidance and it should fail loudly if it ever stops being true.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Testbed.h"
#include "extension/SiLu.h"

using word = uint32_t;

namespace {

// [SYLPH] table 2 / section 3.1.3.
constexpr double kSylphRange = 12.0;
constexpr int kSylphDegree = 31;
constexpr double kTargetBits = 12.0;
// Below this the circuit, not the approximation, sets the accuracy.
constexpr double kMinUsableScale = 1.5e10;  // between 2^30 and 2^35

double TrueSiLu(double x) { return x / (1.0 + std::exp(-x)); }

// Error in bits relative to the largest |SiLU| on the interval, which is what
// [SYLPH]'s "12 bits of precision" means.
double Bits(double abs_err, double ref_absmax) {
  if (abs_err <= 0.0) return 1e9;
  return -std::log2(abs_err / ref_absmax);
}

std::string DataDir() {
  const char *env = std::getenv("LLAMA3_REAL_DIR");
  return env ? std::string(env) : std::string();
}

bool ReadF32(const std::string &path, size_t count, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<float> raw(count);
  f.read(reinterpret_cast<char *>(raw.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  if (static_cast<size_t>(f.gcount()) != count * sizeof(float)) return false;
  out.assign(raw.begin(), raw.end());
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The fit alone. No ciphertext is created here on purpose.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, SiLuPolynomialFit) {
  const int level = default_encryption_level_;
  constexpr int kSamples = 4001;

  std::cout << "  range  degree     max abs err   bits   levels" << std::endl;
  struct Case {
    double range;
    int degree;
  };
  // The degree follows the range steeply and every doubling is a level, so the
  // interval is not a free choice -- it is the thing calibration buys.
  const Case cases[] = {{8.0, 19},  {12.0, 27}, {12.0, kSylphDegree},
                        {16.0, 31}, {16.0, 39}, {24.0, 63}};

  for (const Case &c : cases) {
    SiLuHandler<word> silu(context_, c.range, level, c.degree);
    const double ref_absmax = TrueSiLu(c.range);
    double max_err = 0.0;
    for (int i = 0; i < kSamples; i++) {
      const double x = -c.range + 2.0 * c.range * i / (kSamples - 1);
      max_err = std::max(max_err, std::abs(silu.PlainSiLu(x) - TrueSiLu(x)));
    }
    // ceil(log2(degree + 1)) is the Chebyshev depth EvalPoly needs.
    int levels = 0;
    while ((1 << levels) < c.degree + 1) levels++;
    std::cout << "  +-" << c.range << "   " << c.degree << "        " << max_err
              << "   " << Bits(max_err, ref_absmax) << "   " << levels
              << std::endl;

    if (c.range == kSylphRange && c.degree == kSylphDegree) {
      EXPECT_GT(Bits(max_err, ref_absmax), kTargetBits)
          << "[SYLPH]'s own configuration must clear its own 12-bit bar";
    }
  }
  // +-16 at degree 31 is the near miss that pins why calibration matters: the
  // same degree that clears 12 bits on +-12 falls under it four units wider.
  {
    SiLuHandler<word> silu(context_, 16.0, level, kSylphDegree);
    double max_err = 0.0;
    for (int i = 0; i < kSamples; i++) {
      const double x = -16.0 + 32.0 * i / (kSamples - 1);
      max_err = std::max(max_err, std::abs(silu.PlainSiLu(x) - TrueSiLu(x)));
    }
    EXPECT_LT(Bits(max_err, TrueSiLu(16.0)), kTargetBits)
        << "if +-16 also cleared 12 bits at degree 31 then the interval would "
           "not be doing any work and this whole parameter would be noise";
  }
}

// ---------------------------------------------------------------------------
// 2. The circuit. Compared against the fit, not only against true SiLU, so a
//    scale bug is separated from an approximation error.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, SiLuOnEncryptedSweep) {
  const int level = default_encryption_level_;
  const int slots = param_->MaxNumSlots();
  SiLuHandler<word> silu(context_, kSylphRange, level, kSylphDegree);

  // A dense sweep of the whole interval, so the worst point cannot be missed.
  std::vector<double> x(slots);
  for (int s = 0; s < slots; s++) {
    x[s] = -kSylphRange + 2.0 * kSylphRange * s / (slots - 1);
  }

  // Apply takes x / range. Folding that divide into the gate projection's
  // plaintext weights is free; doing it here by reinterpreting the scale is
  // what broke RMSNorm.
  std::vector<Complex> msg(slots);
  for (int s = 0; s < slots; s++) msg[s] = Complex(x[s] / kSylphRange, 0.0);

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  Ciphertext<word> res;
  silu.Apply(res, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const int out_level = param_->NPToLevel(res.GetNP());
  std::cout << "output level " << out_level << " from input level " << level
            << ", so depth " << (level - out_level) << std::endl;
  EXPECT_NEAR(res.GetScale() / param_->GetScale(out_level), 1.0, 1e-9)
      << "the output scale must be canonical, which is exactly the invariant "
         "that RMSNorm violated";

  std::vector<Complex> got;
  DecryptAndDecode(got, res);

  const double ref_absmax = TrueSiLu(kSylphRange);
  double err_vs_fit = 0.0, err_vs_true = 0.0;
  double worst_x = 0.0;
  for (int s = 0; s < slots; s++) {
    const double fit = silu.PlainSiLu(x[s]);
    err_vs_fit = std::max(err_vs_fit, std::abs(got[s].real() - fit));
    const double d = std::abs(got[s].real() - TrueSiLu(x[s]));
    if (d > err_vs_true) {
      err_vs_true = d;
      worst_x = x[s];
    }
  }
  std::cout << "circuit vs its own polynomial: " << err_vs_fit << "  ("
            << Bits(err_vs_fit, ref_absmax) << " bits)" << std::endl;
  std::cout << "circuit vs true SiLU:          " << err_vs_true << "  ("
            << Bits(err_vs_true, ref_absmax)
            << " bits), worst at x = " << worst_x << std::endl;

  if (param_->base_scale_ < kMinUsableScale) {
    // 2^30. Recorded as a fact about the parameter set, not tolerated as a
    // near miss: if this ever passes, the scale guidance for the Llama
    // non-linearities has changed and needs revisiting.
    EXPECT_LT(Bits(err_vs_true, ref_absmax), kTargetBits)
        << "2^30 now reaches 12 bits, so the recorded requirement of 2^35 for "
           "the Llama non-linearities no longer holds";
    return;
  }
  // The circuit must not be the limiting factor. If it were, this is the
  // number that would move first.
  EXPECT_GT(Bits(err_vs_fit, ref_absmax), kTargetBits + 2.0)
      << "the homomorphic evaluation is losing more than the fit does, which "
         "means the circuit and not the approximation is the bottleneck";
  EXPECT_GT(Bits(err_vs_true, ref_absmax), kTargetBits);
}

// ---------------------------------------------------------------------------
// 3. Real Llama-3-8B weights.
//
// A PROXY, AND LABELLED AS ONE. SiLU's true argument is
// RMSNorm_ffn(h) @ W_gate with h = layer_input + attn_out, and attn_out needs
// the attention block, which is not built yet. Substituting the layer input
// for h uses the real ffn_norm and the real W_gate but the wrong residual.
//
// What it can still settle is whether the encrypted evaluation is exact on an
// argument distribution that came out of the model rather than out of a
// generator. What it cannot settle is the interval: these are layer-2 numbers,
// while [SYLPH]'s 10.82 is a maximum over all 32 layers.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, SiLuOnRealLlama3Gate) {
  const std::string dir = DataDir();
  if (dir.empty()) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  constexpr int kAllTokens = 128;
  constexpr int kChannels = 4096;
  constexpr int kHidden = 14336;
  constexpr int kTokens = 64;
  constexpr int kFirstToken = 64;  // clear of the BOS sinks
  constexpr int kDims = 512;       // slice of the intermediate dimension
  constexpr double kEps = 1e-5;

  std::vector<double> x, w;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kAllTokens * kChannels, x));
  ASSERT_TRUE(ReadF32(dir + "/ffn_norm.f32", kChannels, w));

  // W_gate is row-major [in, out]; keep only the kDims columns we need instead
  // of turning all 234 MB into doubles.
  std::vector<double> wg(static_cast<size_t>(kChannels) * kDims);
  {
    std::ifstream f(dir + "/wgate.f32", std::ios::binary);
    ASSERT_TRUE(f.good()) << "cannot open " << dir << "/wgate.f32";
    std::vector<float> row(kHidden);
    for (int c = 0; c < kChannels; c++) {
      f.read(reinterpret_cast<char *>(row.data()), kHidden * sizeof(float));
      ASSERT_EQ(static_cast<size_t>(f.gcount()), kHidden * sizeof(float));
      for (int j = 0; j < kDims; j++) {
        wg[static_cast<size_t>(c) * kDims + j] = row[j];
      }
    }
  }

  // RMSNorm_ffn, then the gate projection, both in double.
  std::vector<double> g(static_cast<size_t>(kTokens) * kDims, 0.0);
  for (int t = 0; t < kTokens; t++) {
    const int tt = kFirstToken + t;
    double sq = 0.0;
    for (int c = 0; c < kChannels; c++) {
      const double v = x[static_cast<size_t>(tt) * kChannels + c];
      sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq / kChannels + kEps);
    double *out = g.data() + static_cast<size_t>(t) * kDims;
    for (int c = 0; c < kChannels; c++) {
      const double a = x[static_cast<size_t>(tt) * kChannels + c] * inv * w[c];
      if (a == 0.0) continue;
      const double *wrow = wg.data() + static_cast<size_t>(c) * kDims;
      for (int j = 0; j < kDims; j++) out[j] += a * wrow[j];
    }
  }

  double g_absmax = 0.0, g_sq = 0.0;
  for (double v : g) {
    g_absmax = std::max(g_absmax, std::abs(v));
    g_sq += v * v;
  }
  const double g_rms = std::sqrt(g_sq / g.size());
  std::cout << "PROXY gate input |g| max " << g_absmax << ", rms " << g_rms
            << " over " << kTokens << " tokens x " << kDims << " dims"
            << std::endl;
  std::cout << "[SYLPH] table 2 calibrated SiLU input is 10.82, a maximum over "
               "all 32 layers; this is layer 2 with the wrong residual, so the "
               "gap is expected and the interval must still come from "
               "per-layer calibration"
            << std::endl;

  // Evaluate at [SYLPH]'s interval. It covers this data with room to spare,
  // which is the safe direction -- the unsafe direction is silent.
  ASSERT_LT(g_absmax, kSylphRange)
      << "the proxy exceeds the interval, so the numbers below would be "
         "measuring extrapolation and not SiLU";

  const int level = default_encryption_level_;
  const int slots = param_->MaxNumSlots();
  ASSERT_EQ(kTokens * kDims, slots);
  SiLuHandler<word> silu(context_, kSylphRange, level, kSylphDegree);

  // Token-fastest, matching [SYLPH] section 3.2 and RmsNormTest.
  std::vector<Complex> msg(slots);
  for (int s = 0; s < slots; s++) {
    const int j = s / kTokens;
    const int t = s % kTokens;
    msg[s] = Complex(g[static_cast<size_t>(t) * kDims + j] / kSylphRange, 0.0);
  }
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  Ciphertext<word> res;
  silu.Apply(res, ct, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Complex> got;
  DecryptAndDecode(got, res);

  // Two references. Relative to |SiLU| on the interval is [SYLPH]'s bar;
  // relative to what this data actually reaches is the honest local number,
  // and it is far worse because the data occupies a small part of the
  // interval the polynomial was asked to cover.
  const double ref_interval = TrueSiLu(kSylphRange);
  double local_absmax = 0.0;
  for (double v : g) {
    local_absmax = std::max(local_absmax, std::abs(TrueSiLu(v)));
  }

  double max_err = 0.0, sum_err = 0.0;
  for (int s = 0; s < slots; s++) {
    const int j = s / kTokens;
    const int t = s % kTokens;
    const double want = TrueSiLu(g[static_cast<size_t>(t) * kDims + j]);
    const double d = std::abs(got[s].real() - want);
    max_err = std::max(max_err, d);
    sum_err += d;
  }
  std::cout << "max abs err " << max_err << ", mean " << (sum_err / slots)
            << std::endl;
  std::cout << "  vs |SiLU| on +-" << kSylphRange << " (" << ref_interval
            << "):  " << Bits(max_err, ref_interval) << " bits" << std::endl;
  std::cout << "  vs what this data reaches (" << local_absmax
            << "):  " << Bits(max_err, local_absmax) << " bits" << std::endl;
  std::cout << "the second number is the cost of covering an interval much "
               "wider than the data needs, which is what per-layer calibration "
               "recovers"
            << std::endl;

  if (param_->base_scale_ < kMinUsableScale) {
    EXPECT_LT(Bits(max_err, ref_interval), kTargetBits)
        << "2^30 now reaches 12 bits on real weights, so the recorded "
           "requirement of 2^35 no longer holds";
  } else {
    EXPECT_GT(Bits(max_err, ref_interval), kTargetBits);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    // ci16_35: SiLU is pointwise, so the conjugate-invariant ring should cost
    // it nothing but give it twice the slots ([SYLPH] section 2.1). Measured
    // rather than assumed.
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json", "ci16_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
