// RoPE, [SYLPH] section 3.2: "a position-wise plaintext-ciphertext
// multiplication", one level.
//
// THE REFERENCE IS WRITTEN FROM THE DEFINITION, NOT FROM THE HANDLER. RoPE is
// pure index arithmetic, and the one bug that has cost this project real time
// twice now is a layout mistake: SoftMax's reduction crossed row boundaries
// because a cyclic rotation was treated as a block operation. So the host
// reference below loops over (token, head, j) the textbook way and never
// touches the slot formula the handler uses. If the two agree, the layout
// mapping is right; if only the encrypted path disagrees, the circuit is at
// fault.
//
// Llama-3 uses theta = 500000. Hugging Face's rotate_half pairs j with j + D/2,
// not 2j with 2j+1, and the weights come from a Hugging Face mirror.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Testbed.h"
#include "extension/RoPe.h"

using word = uint32_t;

namespace {

constexpr int kTokens = 128;
constexpr int kHeadDim = 128;
constexpr double kTheta = 500000.0;

// Textbook RoPE over a (channel, token) grid, written from the definition.
// channel c belongs to head c / D with within-head index j = c % D.
std::vector<double> HostRoPe(const std::vector<double> &x, int channels,
                             int first_position) {
  const int half = kHeadDim / 2;
  std::vector<double> out(x.size(), 0.0);
  for (int c = 0; c < channels; c++) {
    const int j = c % kHeadDim;
    const int base = c - j;  // first channel of this head
    for (int t = 0; t < kTokens; t++) {
      const int p = first_position + t;
      const double f = std::pow(kTheta, -2.0 * (j % half) / (double)kHeadDim);
      const double ang = p * f;
      const double cs = std::cos(ang), sn = std::sin(ang);
      const double self = x[(size_t)c * kTokens + t];
      const double partner =
          (j < half) ? x[((size_t)base + j + half) * kTokens + t]
                     : x[((size_t)base + j - half) * kTokens + t];
      out[(size_t)c * kTokens + t] =
          (j < half) ? self * cs - partner * sn : self * cs + partner * sn;
    }
  }
  return out;
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

// Deterministic, so a failure reproduces.
double Sample(int c, int t) {
  return std::sin(0.7 * c + 0.13 * t) * (1.0 + 0.1 * std::cos(0.03 * c));
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The layout mapping, with no ciphertext present.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, RoPeLayoutMatchesTheDefinition) {
  const int slots = param_->degree_ / 2;
  const int channels = slots / kTokens;
  RoPeHandler<word> rope(context_, kTokens, kHeadDim,
                         default_encryption_level_, kTheta);

  for (int first : {0, 64, 1000}) {
    std::vector<double> x(slots);
    for (int c = 0; c < channels; c++) {
      for (int t = 0; t < kTokens; t++) x[(size_t)c * kTokens + t] = Sample(c, t);
    }
    std::vector<double> got;
    rope.PlainApply(got, x, first);
    auto want = HostRoPe(x, channels, first);

    double worst = 0.0;
    for (int i = 0; i < slots; i++) {
      worst = std::max(worst, std::abs(got[i] - want[i]));
    }
    std::cout << "first_position " << first << ": layout vs definition "
              << worst << std::endl;
    // Both are double arithmetic over the same values, so anything above
    // rounding means the slot formula and the definition disagree.
    EXPECT_LT(worst, 1e-12)
        << "the slot mapping and the textbook definition disagree, so the "
           "partner channel is being read from the wrong place";
  }
  std::cout << "rotation distances: ";
  for (int d : rope.GetRotationDistances()) std::cout << d << " ";
  std::cout << "(two, one per half)" << std::endl;
}

// ---------------------------------------------------------------------------
// 2. The circuit. One level is the claim being checked.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, RoPeOnEncrypted) {
  const int level = default_encryption_level_;
  const int slots = param_->degree_ / 2;
  const int channels = slots / kTokens;
  constexpr int kFirst = 64;

  RoPeHandler<word> rope(context_, kTokens, kHeadDim, level, kTheta);
  for (int d : rope.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, level);
  }

  std::vector<double> x(slots);
  std::vector<Complex> msg(slots);
  for (int c = 0; c < channels; c++) {
    for (int t = 0; t < kTokens; t++) {
      const double v = Sample(c, t);
      x[(size_t)c * kTokens + t] = v;
      msg[(size_t)c * kTokens + t] = Complex(v, 0.0);
    }
  }

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);
  Ciphertext<word> res;
  rope.Apply(res, ct, kFirst, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const int out_level = param_->NPToLevel(res.GetNP());
  std::cout << "output level " << out_level << " from " << level << ", depth "
            << (level - out_level) << std::endl;
  EXPECT_EQ(level - out_level, 1)
      << "[SYLPH] 3.2 calls RoPE a position-wise plaintext-ciphertext "
         "multiplication, which is one level; more than that means the three "
         "products are not sharing a single Rescale";
  EXPECT_NEAR(res.GetScale() / param_->GetScale(out_level), 1.0, 1e-9)
      << "the output scale must stay canonical";

  std::vector<Complex> got;
  DecryptAndDecode(got, res);
  auto want = HostRoPe(x, channels, kFirst);

  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < slots; i++) {
    worst = std::max(worst, std::abs(got[i].real() - want[i]));
    absmax = std::max(absmax, std::abs(want[i]));
  }
  std::cout << "circuit vs definition: " << worst << " ("
            << -std::log2(worst / absmax) << " bits relative to |y| max "
            << absmax << ")" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 12.0);
}

// ---------------------------------------------------------------------------
// 3. The real layer-2 query projection.
//
// Unlike SiLU's and SoftMax's real-weight cases, this one needs no proxy: RoPE
// sits directly on the QKV output, and RMSNorm(x) @ W_Q is exactly that -- no
// missing residual, no unbuilt block in between.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, RoPeOnRealLlama3Query) {
  const std::string dir = DataDir();
  if (dir.empty()) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  constexpr int kChannels = 4096;
  constexpr double kEps = 1e-5;
  const int slots = param_->degree_ / 2;
  const int channels = slots / kTokens;  // 256 of the 4096 query channels
  ASSERT_LE(channels, kChannels);
  ASSERT_EQ(channels % kHeadDim, 0);

  std::vector<double> xin, wn;
  ASSERT_TRUE(ReadF32(dir + "/input.f32", kTokens * kChannels, xin));
  ASSERT_TRUE(ReadF32(dir + "/attn_norm.f32", kChannels, wn));

  // W_Q is row-major [in, out]; keep only the `channels` columns needed.
  std::vector<double> wq((size_t)kChannels * channels);
  {
    std::ifstream f(dir + "/wq.f32", std::ios::binary);
    ASSERT_TRUE(f.good());
    std::vector<float> row(kChannels);
    for (int c = 0; c < kChannels; c++) {
      f.read(reinterpret_cast<char *>(row.data()), kChannels * sizeof(float));
      ASSERT_EQ((size_t)f.gcount(), kChannels * sizeof(float));
      for (int o = 0; o < channels; o++) {
        wq[(size_t)c * channels + o] = row[o];
      }
    }
  }

  // pre-attention RMSNorm, then the query projection, in double
  std::vector<double> q((size_t)channels * kTokens, 0.0);
  for (int t = 0; t < kTokens; t++) {
    double sq = 0.0;
    for (int c = 0; c < kChannels; c++) {
      const double v = xin[(size_t)t * kChannels + c];
      sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sq / kChannels + kEps);
    for (int c = 0; c < kChannels; c++) {
      const double a = xin[(size_t)t * kChannels + c] * inv * wn[c];
      if (a == 0.0) continue;
      const double *wrow = wq.data() + (size_t)c * channels;
      for (int o = 0; o < channels; o++) q[(size_t)o * kTokens + t] += a * wrow[o];
    }
  }
  double q_absmax = 0.0;
  for (double v : q) q_absmax = std::max(q_absmax, std::abs(v));
  std::cout << "real |Q| max " << q_absmax << " over " << kTokens
            << " tokens x " << channels << " channels" << std::endl;

  const int level = default_encryption_level_;
  RoPeHandler<word> rope(context_, kTokens, kHeadDim, level, kTheta);
  for (int d : rope.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, level);
  }
  std::vector<Complex> msg(slots);
  for (int i = 0; i < slots; i++) msg[i] = Complex(q[i], 0.0);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  Ciphertext<word> res;
  rope.Apply(res, ct, 0, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<Complex> got;
  DecryptAndDecode(got, res);
  auto want = HostRoPe(q, channels, 0);

  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < slots; i++) {
    worst = std::max(worst, std::abs(got[i].real() - want[i]));
    absmax = std::max(absmax, std::abs(want[i]));
  }
  std::cout << "real weights: max abs err " << worst << " ("
            << -std::log2(worst / absmax) << " bits relative to |y| max "
            << absmax << ")" << std::endl;
  // RoPE is an orthogonal rotation, so it must not change the magnitude.
  EXPECT_NEAR(absmax / q_absmax, 1.0, 0.05)
      << "RoPE rotates within each channel pair, so the largest magnitude "
         "should be essentially unchanged";
  EXPECT_GT(-std::log2(worst / absmax), 12.0);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
