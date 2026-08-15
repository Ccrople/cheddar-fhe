// The Bae PC-MM against a real Llama-3-8B projection.
//
// Everything before this validated the product against itself: random inputs,
// host references I wrote, and a commutation identity. Those establish that the
// arithmetic is self-consistent. They do not establish that it survives the
// numbers an actual model produces, which is a different question -- Llama-3's
// residual stream carries outliers (this bundle's layer-2 input reaches 225.7
// against an RMS of 0.88) and its weights are small and dense (|W_Q| <= 0.239,
// RMS 0.0175). Precision has to be argued against those, not against U(-1,1).
//
// So this computes one real thing end to end: the query projection of layer 2,
//
//     Y = RMSNorm(x) @ W_Q        x: 128 tokens x 4096 channels
//
// over the *full* 4096-channel inner dimension, for the first 128 output
// channels, and compares against the same product in double on the host. Not a
// tile and not a partial sum -- a complete, real output block.
//
// DATA. /mnt/home2_hawaii/student/JHJun/llama3_real, produced by HEonGPU's
// benchmark/fetch_llama3_weights.py, which runs the true fp32 model prefix
// (RoPE included) so the layer-2 input is the genuine residual stream rather
// than noise shaped to look like one. Projections are stored already
// transposed to row-major [in_channels, out_channels]. Point LLAMA3_REAL_DIR
// at it; the test skips if unset, so the suite still runs without the bundle.
//
// LAYOUT. This uses the RLWE overload at ring degree 4096: one ciphertext per
// input channel, holding that channel's 128 token values in coefficients
// 0..127. That wastes 32x of each polynomial, and fixing that waste is exactly
// what the MLWE format is for -- one degree-4096 parent carries 16 channels
// once ModDecomp splits it to degree 256. The MLWE overload is proven equal to
// this one by commutation (SmallRingContextTest) and PipelineChainTest runs it
// for real, so the wasteful packing is kept here on purpose: it isolates the
// arithmetic from the ring switching and the repacking, which is what makes a
// failure here mean the *product* is wrong.
//
// SCALES. Activations at 2^30 (the ring's own scale) and weights at 2^30. The
// product is therefore at 2^60, and |Y| <= 10.5 puts the encoded result at
// 2^63.4 against log2 Q1 = 69.76 -- about six bits of headroom. Weights at
// 2^30 keep roughly twenty-four bits on a typical entry of RMS 0.0175, which
// is what bounds the error here; the activation quantisation is far below it.
//
// 2^30 is also the only weight scale that would leave a *rescaled* result on
// this ring's canonical level-0 scale, which is what PipelineChainTest needs.
// Nothing is rescaled here -- the result is decoded at the product scale -- but
// matching it keeps the two tests telling the same story.

#undef ENABLE_EXTENSION

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Testbed.h"
#include "core/Pcmm.h"

using word = uint32_t;

namespace {

constexpr int kTokens = 128;
constexpr int kChannels = 4096;
constexpr int kOutChannels = 128;
constexpr double kRmsEps = 1e-5;
constexpr double kWeightScale = 1073741824.0;  // 2^30

std::string DataDir() {
  const char *env = std::getenv("LLAMA3_REAL_DIR");
  return env ? std::string(env) : std::string();
}

// Flat little-endian float32, exactly as fetch_llama3_weights.py writes it.
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

TEST_P(Testbed32, RealLlama3QueryProjection) {
  const std::string dir = DataDir();
  if (dir.empty()) {
    GTEST_SKIP() << "LLAMA3_REAL_DIR is not set; skipping the real-weight "
                    "validation. Point it at the llama3_real bundle.";
  }

  std::vector<double> x, wq, attn_norm;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kTokens * kChannels, x))
      << "cannot read " << dir << "/input.f32";
  ASSERT_TRUE(ReadF32(dir + "/wq.f32", kChannels * kChannels, wq))
      << "cannot read " << dir << "/wq.f32";
  ASSERT_TRUE(ReadF32(dir + "/attn_norm.f32", kChannels, attn_norm))
      << "cannot read " << dir << "/attn_norm.f32";

  // Llama RMSNorm, per token: x / sqrt(mean(x^2) + eps) * w.
  std::vector<double> xn(kTokens * kChannels);
  for (int t = 0; t < kTokens; t++) {
    double sq = 0.0;
    for (int c = 0; c < kChannels; c++) {
      const double v = x[t * kChannels + c];
      sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq / kChannels + kRmsEps);
    for (int c = 0; c < kChannels; c++) {
      xn[t * kChannels + c] = x[t * kChannels + c] * inv * attn_norm[c];
    }
  }

  // Host reference, in double, over the full inner dimension.
  std::vector<double> want(kTokens * kOutChannels, 0.0);
  for (int c = 0; c < kChannels; c++) {
    const double *wrow = wq.data() + static_cast<size_t>(c) * kChannels;
    for (int t = 0; t < kTokens; t++) {
      const double a = xn[t * kChannels + c];
      if (a == 0.0) continue;
      double *wrow_out = want.data() + t * kOutChannels;
      for (int o = 0; o < kOutChannels; o++) wrow_out[o] += a * wrow[o];
    }
  }
  double want_absmax = 0.0;
  for (double v : want) want_absmax = std::max(want_absmax, std::abs(v));
  std::cout << "reference |Y| max " << want_absmax << " over " << kTokens
            << " tokens x " << kOutChannels << " out channels" << std::endl;

  // One ciphertext per input channel: coefficients 0..127 are that channel's
  // token values, the remaining 3968 are zero.
  const int degree = param_->degree_;
  const int level = param_->max_level_;
  const double ct_scale = DetermineScale(level);
  PcmmHandler<word> pcmm(*param_);

  std::vector<Ciphertext<word>> cts(kChannels);
  {
    std::vector<double> coeffs(degree, 0.0);
    Plaintext<word> pt;
    for (int c = 0; c < kChannels; c++) {
      for (int t = 0; t < kTokens; t++) coeffs[t] = xn[t * kChannels + c];
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(cts[c], pt);
    }
  }
  std::cout << "encrypted " << kChannels << " channel ciphertexts at level "
            << level << std::endl;

  // U[o][c] = W_Q[c][o]; the bundle stores W_Q row-major as [in, out].
  std::vector<double> u_values(static_cast<size_t>(kOutChannels) * kChannels);
  for (int o = 0; o < kOutChannels; o++) {
    for (int c = 0; c < kChannels; c++) {
      u_values[static_cast<size_t>(o) * kChannels + c] =
          wq[static_cast<size_t>(c) * kChannels + o];
    }
  }
  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, level, kWeightScale, u_values, kOutChannels, kChannels);

  std::vector<Ciphertext<word>> res;
  pcmm.Multiply(res, u, cts);
  ASSERT_EQ(static_cast<int>(res.size()), kOutChannels);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // The product does not rescale, so the result sits at u.scale * ct.scale and
  // DecodeCoeff divides by exactly that.
  EXPECT_NEAR(res[0].GetScale() / (kWeightScale * ct_scale), 1.0, 1e-9);

  double max_abs = 0.0, sum_abs = 0.0;
  int worst_o = -1, worst_t = -1;
  for (int o = 0; o < kOutChannels; o++) {
    Plaintext<word> pt;
    interface_->Decrypt(pt, res[o]);
    std::vector<double> got;
    context_->encoder_.DecodeCoeff(got, pt);
    ASSERT_EQ(static_cast<int>(got.size()), degree);

    for (int t = 0; t < kTokens; t++) {
      const double d = std::abs(got[t] - want[t * kOutChannels + o]);
      sum_abs += d;
      if (d > max_abs) {
        max_abs = d;
        worst_o = o;
        worst_t = t;
      }
    }
  }
  const double mean_abs = sum_abs / (kTokens * kOutChannels);

  std::cout << "Q projection over " << kChannels << " channels: max abs err "
            << max_abs << " (token " << worst_t << ", out " << worst_o
            << "), mean abs err " << mean_abs << ", relative to |Y| max "
            << (max_abs / want_absmax) << std::endl;

  // Weight quantisation at 2^30 dominates: a half-ulp of 2^-31 against 4096
  // random-signed terms and an activation RMS of 0.39 predicts roughly 1e-8.
  // The bound below is loose enough not to be flaky and tight enough that a
  // real regression -- a wrong transpose, a lost limb, an overflow -- moves it
  // by orders of magnitude rather than percent.
  EXPECT_LT(max_abs / want_absmax, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    SmallRing, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
