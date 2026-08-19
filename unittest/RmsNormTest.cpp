// RMSNorm on the real Llama-3-8B layer-2 residual stream, against [SYLPH].
//
// SINK TOKENS ARE EXCLUDED, AND THAT IS THE POINT. [SYLPH] evaluates the
// inverse square root on a window of only 30x -- [1/sqrt(30), sqrt(30)], from
// the companion Llama-2 paper's table 4. Measured on this bundle, the
// per-token mean square spans 285,946x, and three tokens account for all of it:
//
//     t0 = 33.12,  t1 = 33.12,  t3 = 32.64,   ~0.0005 everywhere else
//     prompt: 128000 128000 382 128000 ...    i.e. BOS, BOS, _, BOS
//
// [SYLPH] section 3.1.1 keeps those out of the encrypted path entirely: their
// Key-Value states are precomputed offline and injected as a static cache,
// which is sound because the sink prefix does not depend on the user's input.
// The remaining tokens span 4.87x and fit the window with room to spare. This
// test therefore takes 64 consecutive non-sink tokens, which is what an
// encrypted user segment actually looks like.
//
// The reference is the same formula in double, so what is measured is the
// homomorphic circuit -- the square, the rotate-and-add reduction, the
// polynomial and the two multiplies -- and not the model.
//
// TARGET. [SYLPH] section 3.1.2 reports that 12 bits of precision matches FP16
// perplexity, so 2^-12 relative is the bar. The Chebyshev fit alone is 13.8
// bits; anything materially worse than that comes from the circuit.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Testbed.h"
#include "extension/RmsNorm.h"

using word = uint32_t;

namespace {

constexpr int kAllTokens = 128;
constexpr int kChannels = 4096;
constexpr int kTokens = 64;      // encrypted user segment, a power of two
constexpr int kFirstToken = 64;  // t64..t127, all clear of the sinks
constexpr double kEps = 1e-5;

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

double MeanSquare(const std::vector<double> &x, int token) {
  double s = 0.0;
  for (int c = 0; c < kChannels; c++) {
    const double v = x[static_cast<size_t>(token) * kChannels + c];
    s += v * v;
  }
  return s / kChannels;
}

}  // namespace

TEST_P(Testbed32, RmsNormOnRealLlama3) {
  const std::string dir = DataDir();
  if (dir.empty()) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  std::vector<double> x, w;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kAllTokens * kChannels, x));
  ASSERT_TRUE(ReadF32(dir + "/attn_norm.f32", kChannels, w));

  // Confirm the sinks are where the analysis says, so a different bundle
  // cannot silently invalidate the token choice.
  for (int t : {0, 1, 3}) {
    EXPECT_GT(MeanSquare(x, t), 1.0) << "token " << t << " should be a sink";
  }
  double lo = 1e300, hi = 0.0, log_sum = 0.0;
  for (int t = kFirstToken; t < kFirstToken + kTokens; t++) {
    const double ms = MeanSquare(x, t);
    lo = std::min(lo, ms);
    hi = std::max(hi, ms);
    log_sum += std::log(ms);
  }
  std::cout << "user segment t" << kFirstToken << "..t"
            << (kFirstToken + kTokens - 1) << ": mean-square spread " << (hi / lo)
            << "x, and Sylph's window allows 30x" << std::endl;
  EXPECT_LT(hi / lo, 30.0);

  // alpha_L = 1 / geometric mean, which centres the argument in the window.
  const double alpha = 1.0 / std::exp(log_sum / kTokens);
  std::cout << "alpha_L = " << alpha << ", so the argument sits in ["
            << (alpha * lo) << ", " << (alpha * hi) << "]" << std::endl;

  // RANGE CONTROL, standing in for [SYLPH]'s calibration.
  //
  // CKKS precision is absolute, not relative: a value only gets as many bits
  // as it occupies below the scaling factor. On this uncalibrated bundle the
  // user tokens have |x| ~ 0.022, so x^2 ~ 4e-4 and the square uses about 19
  // of the 30 bits the scale offers. The eleven wasted bits show up directly:
  // the reduction alone lands at 0.3-0.5% relative, and because the error is
  // correlated across slots, summing 4096 terms does not average it away.
  //
  // [SYLPH] does not meet this, because after calibration its RMSNorm input is
  // 7.65 (table 2) -- order one, where the scale is well spent. Our data is
  // some 400x smaller.
  //
  // RMSNorm is invariant under x -> beta*x, so scaling the input to order one
  // is exactly free mathematically. Feeding beta*x with alpha/beta^2 and
  // eps*beta^2 leaves the polynomial's argument bit-for-bit identical, and the
  // beta cancels in the output. Choosing beta = sqrt(alpha) makes the layer
  // constant exactly 1, which is what a calibrated model would have handed us.
  const double beta = std::sqrt(alpha);
  const double alpha_scaled = alpha / (beta * beta);
  const double eps_scaled = kEps * beta * beta;
  std::cout << "beta = " << beta << " puts |x| near one; alpha_L becomes "
            << alpha_scaled << std::endl;

  const int level = default_encryption_level_;
  RmsNormHandler<word> rms(context_, kTokens, kChannels, alpha_scaled, level,
                           eps_scaled);
  const int num_ct = rms.GetNumCiphertexts();
  std::cout << "T=" << kTokens << " H=" << kChannels << " -> " << num_ct
            << " ciphertexts, " << rms.GetRotationDistances().size()
            << " rotations" << std::endl;
  // The max_level argument is not optional in practice. PrepareRotationKey
  // defaults it to -1, and GetNPForEvk reserves -1 for the dense-to-sparse
  // short base -- two auxiliary primes, not the full range. A key built that
  // way fails deep inside the rotation with "QSize mismatch", nowhere near the
  // call that asked for it. PrepareModPackKeys remaps -1 for exactly this
  // reason; PrepareRotationKey does not.
  for (int d : rms.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, level);
  }

  // Pack token-fastest, matching [SYLPH] section 3.2.
  const int slots = param_->degree_ / 2;
  const int channels_per_ct = slots / kTokens;
  std::vector<Ciphertext<word>> cts(num_ct);
  std::vector<std::vector<Complex>> wts(num_ct);
  const double root_alpha = std::sqrt(alpha_scaled);
  for (int i = 0; i < num_ct; i++) {
    std::vector<Complex> msg(slots);
    wts[i].assign(slots, Complex(0.0, 0.0));
    for (int s = 0; s < slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = kFirstToken + (s % kTokens);
      msg[s] = Complex(beta * x[static_cast<size_t>(t) * kChannels + c], 0.0);
      wts[i][s] = Complex(w[c] * root_alpha, 0.0);
    }
    EncodeAndEncrypt(cts[i], msg, level);
  }

  std::vector<Ciphertext<word>> res;
  rms.Apply(res, cts, wts, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), num_ct);
  const int out_level = param_->NPToLevel(res[0].GetNP());
  std::cout << "output level " << out_level << " from input level " << level
            << ", so depth " << (level - out_level) << std::endl;

  // Reference, in double.
  std::vector<double> want(static_cast<size_t>(kTokens) * kChannels);
  double want_absmax = 0.0;
  for (int t = 0; t < kTokens; t++) {
    const double inv = 1.0 / std::sqrt(MeanSquare(x, kFirstToken + t) + kEps);
    for (int c = 0; c < kChannels; c++) {
      const double v =
          x[static_cast<size_t>(kFirstToken + t) * kChannels + c] * inv * w[c];
      want[static_cast<size_t>(t) * kChannels + c] = v;
      want_absmax = std::max(want_absmax, std::abs(v));
    }
  }

  double max_abs = 0.0, sum_abs = 0.0;
  for (int i = 0; i < num_ct; i++) {
    std::vector<Complex> got;
    DecryptAndDecode(got, res[i]);
    for (int s = 0; s < slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = s % kTokens;
      const double d = std::abs(
          got[s].real() - want[static_cast<size_t>(t) * kChannels + c]);
      max_abs = std::max(max_abs, d);
      sum_abs += d;
    }
  }
  // Separate the polynomial from the application. Since
  //     y[t][c] = x[t][c] * r[t] * sqrt(alpha) * w[c],
  // any channel with a non-tiny x recovers r[t]*sqrt(alpha), and comparing that
  // to the true 1/sqrt(mean + eps) says whether the inverse square root or the
  // surrounding circuit is at fault.
  {
    std::vector<Complex> got0;
    DecryptAndDecode(got0, res[0]);
    std::cout << "  token   recovered r      true r        rel" << std::endl;
    for (int t = 0; t < kTokens; t += 8) {
      // pick the channel in ciphertext 0 with the largest |x| for this token
      int best_c = 0;
      double best = 0.0;
      for (int cc = 0; cc < channels_per_ct; cc++) {
        const double v =
            std::abs(x[static_cast<size_t>(kFirstToken + t) * kChannels + cc]);
        if (v > best) { best = v; best_c = cc; }
      }
      const int s = best_c * kTokens + t;
      const double xv =
          x[static_cast<size_t>(kFirstToken + t) * kChannels + best_c];
      const double recovered = got0[s].real() / (xv * w[best_c]);  // beta cancels
      const double truth =
          1.0 / std::sqrt(MeanSquare(x, kFirstToken + t) + kEps);
      std::cout << "  t" << (kFirstToken + t) << "  recovered " << recovered
                << "  true " << truth << "  rel "
                << (std::abs(recovered - truth) / truth) << std::endl;
    }
  }

  const double rel = max_abs / want_absmax;
  std::cout << "RMSNorm: |y| max " << want_absmax << ", max abs err " << max_abs
            << ", mean " << (sum_abs / (1.0 * kTokens * kChannels))
            << ", relative " << rel << " = 2^" << std::log2(rel) << std::endl;
  std::cout << "Sylph's target is 12 bits, i.e. 2^-12 = " << std::pow(2.0, -12)
            << std::endl;

  EXPECT_LT(rel, std::pow(2.0, -12)) << "below Sylph's 12-bit precision target";
}

// Isolate the reduction. The full test showed the recovered inverse square
// root wrong by 0.4% to 23% with no dependence on the argument -- t80 sits mid
// interval and is the worst, t88 sits near the edge and is nearly right -- so
// the polynomial cannot be what is wrong. Either the square or the
// rotate-and-add is not producing sum_c x[t][c]^2.
//
// This checks exactly that quantity and nothing else.
TEST_P(Testbed32, RmsNormReductionIsExact) {
  const std::string dir = DataDir();
  if (dir.empty()) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  std::vector<double> x;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kAllTokens * kChannels, x));

  const int level = default_encryption_level_;
  const int slots = param_->degree_ / 2;
  const int channels_per_ct = slots / kTokens;
  const int num_ct = kChannels / channels_per_ct;

  std::vector<int> dists;
  for (int d = kTokens; d < slots; d *= 2) dists.push_back(d);
  for (int d : dists) interface_->PrepareRotationKey(d, level);

  std::vector<Ciphertext<word>> cts(num_ct);
  for (int i = 0; i < num_ct; i++) {
    std::vector<Complex> msg(slots);
    for (int s = 0; s < slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int tk = kFirstToken + (s % kTokens);
      msg[s] = Complex(x[static_cast<size_t>(tk) * kChannels + c], 0.0);
    }
    EncodeAndEncrypt(cts[i], msg, level);
  }

  const auto &mult_key = interface_->GetMultiplicationKey();
  Ciphertext<word> acc, sq, rotated;
  for (int i = 0; i < num_ct; i++) {
    context_->HMult(sq, cts[i], cts[i], mult_key);
    if (i == 0) {
      context_->Copy(acc, sq);
    } else {
      context_->Add(acc, acc, sq);
    }
  }
  for (int d : dists) {
    context_->HRotAdd(rotated, acc, acc, interface_->GetRotationKey(d), d);
    context_->Copy(acc, rotated);
  }

  std::vector<Complex> got;
  DecryptAndDecode(got, acc);

  double worst = 0.0;
  int worst_t = -1;
  for (int tk = 0; tk < kTokens; tk++) {
    double want = 0.0;
    for (int c = 0; c < kChannels; c++) {
      const double v = x[static_cast<size_t>(kFirstToken + tk) * kChannels + c];
      want += v * v;
    }
    const double rel = std::abs(got[tk].real() - want) / want;
    if (rel > worst) { worst = rel; worst_t = tk; }
    if (tk % 16 == 0) {
      std::cout << "  t" << (kFirstToken + tk) << "  got " << got[tk].real()
                << "  want " << want << "  rel " << rel << std::endl;
    }
  }
  std::cout << "reduction worst relative error " << worst << " at token "
            << (kFirstToken + worst_t) << std::endl;
  EXPECT_LT(worst, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
