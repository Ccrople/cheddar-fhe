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

  const int level = default_encryption_level_;
  RmsNormHandler<word> rms(context_, kTokens, kChannels, alpha, level, kEps);
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
  std::vector<Plaintext<word>> wts(num_ct);
  const double root_alpha = std::sqrt(alpha);
  for (int i = 0; i < num_ct; i++) {
    std::vector<Complex> msg(slots), wmsg(slots);
    for (int s = 0; s < slots; s++) {
      const int c = i * channels_per_ct + s / kTokens;
      const int t = kFirstToken + (s % kTokens);
      msg[s] = Complex(x[static_cast<size_t>(t) * kChannels + c], 0.0);
      wmsg[s] = Complex(w[c] * root_alpha, 0.0);
    }
    EncodeAndEncrypt(cts[i], msg, level);
    Encode(wts[i], wmsg, level);
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
  const double rel = max_abs / want_absmax;
  std::cout << "RMSNorm: |y| max " << want_absmax << ", max abs err " << max_abs
            << ", mean " << (sum_abs / (1.0 * kTokens * kChannels))
            << ", relative " << rel << " = 2^" << std::log2(rel) << std::endl;
  std::cout << "Sylph's target is 12 bits, i.e. 2^-12 = " << std::pow(2.0, -12)
            << std::endl;

  EXPECT_LT(rel, std::pow(2.0, -12)) << "below Sylph's 12-bit precision target";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
