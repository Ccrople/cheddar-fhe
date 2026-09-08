// The PUBLIC PREFILL STAGE, which carries no FHE at all.
//
// [SYLPH] 4.1 splits the prompt: "process the public prompt P in the clear and
// generate a public KV cache (Kpub, Vpub)" -- the public half is a standard
// Llama prefill on the host, and only the PC-attention that reads its output
// is homomorphic. `CiPcAttention` already enforces that on its side (its
// `ChunkSource` hands it plain doubles and there is no ciphertext type on the
// public operand); `PublicPrefill` is what fills them.
//
// So nothing here is encrypted and nothing here needs a GPU. What has to be
// checked is that the CONVENTION matches the encrypted branch, because three
// things have to agree exactly or the two halves cannot be added, and each of
// them is easy to get silently wrong:
//
//   1. the activation  -- rmsnorm WITH `attn_norm`, then the raw wk/wv, which
//      is the encrypted branch's own split (`CiBatchProjection::FoldGain`
//      puts the norm gain on the weights and feeds them the normalised
//      stream), plus [SYLPH] 3.4's sink rescale over the whole context;
//   2. RoPE          -- and this is the one the whole T = 4096 design rests
//      on, so it is measured as an IDENTITY below rather than asserted;
//   3. the carried factor `ck`.
//
// Test 2 is the load-bearing one. `CiBatchAttention::BuildRope` gives query
// token `t` the angle `t * theta` -- it calls the encrypted block position 0 --
// while at T = 4096 the encrypted tokens really sit at `ptok .. ptok + 127`.
// The claim is that the correction can be taken entirely on the PUBLIC side,
// for free, by rotating the public key at `p - ptok`, because RoPE is
// relative. If that claim is wrong the encrypted branch has to change and its
// RoPE plaintexts are shared by Q and K, so it is worth a measurement.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "extension/PublicPrefill.h"

using cheddar::PublicPrefillChunk;
using cheddar::PublicPrefillLayer;
using cheddar::PublicSinkRescale;

namespace {

constexpr int kModel = 4096;
constexpr int kKvHeads = 8;
constexpr int kDim = 128;
constexpr int kCols = kKvHeads * kDim;  // 1024
constexpr double kRopeBase = 500000.0;
constexpr double kEps = 1e-5;

bool ReadF32(const std::string &path, size_t want, std::vector<float> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) return false;
  out.resize(want);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(want * sizeof(float)));
  return static_cast<size_t>(f.gcount()) == want * sizeof(float);
}

// An INDEPENDENT host K/V for one (instance, token): written from the recipe
// rather than from `PublicPrefill`, so that agreeing means something.
void HostKv(std::vector<double> &k, std::vector<double> &v, const double *x,
            const std::vector<float> &wk, const std::vector<float> &wv,
            const std::vector<float> &an, int kv_head, double pos, double ck) {
  double ms = 0.0;
  for (int c = 0; c < kModel; c++) ms += x[c] * x[c];
  const double inv = 1.0 / std::sqrt(ms / kModel + kEps);
  std::vector<double> y(kModel);
  for (int c = 0; c < kModel; c++) y[c] = x[c] * inv * double(an[c]);

  k.assign(kDim, 0.0);
  v.assign(kDim, 0.0);
  const int base = kv_head * kDim;
  for (int c = 0; c < kModel; c++) {
    for (int d = 0; d < kDim; d++) {
      k[d] += y[c] * double(wk[size_t(c) * kCols + base + d]);
      v[d] += y[c] * double(wv[size_t(c) * kCols + base + d]);
    }
  }
  const int half = kDim / 2;
  for (int d = 0; d < half; d++) {
    const double a = pos * std::pow(kRopeBase, -2.0 * d / kDim);
    const double lo = k[d], hi = k[d + half];
    k[d] = lo * std::cos(a) - hi * std::sin(a);
    k[d + half] = hi * std::cos(a) + lo * std::sin(a);
  }
  for (int d = 0; d < kDim; d++) k[d] *= ck;
}

// rotate_half at `pos`, in place -- the same rotation `CiBatchAttention::Rope`
// applies to Q and K.
void Rope(std::vector<double> &m, double pos) {
  const int half = kDim / 2;
  for (int d = 0; d < half; d++) {
    const double a = pos * std::pow(kRopeBase, -2.0 * d / kDim);
    const double lo = m[d], hi = m[d + half];
    m[d] = lo * std::cos(a) - hi * std::sin(a);
    m[d + half] = hi * std::cos(a) + lo * std::sin(a);
  }
}

double Dot(const std::vector<double> &a, const std::vector<double> &b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
  return s;
}

std::string WeightDir() {
  const char *e = std::getenv("LLAMA3_ALL_DIR");
  return (e != nullptr) ? std::string(e) : std::string();
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. ROPE IS RELATIVE, so the encrypted branch does not have to move.
//
//    The design of the T = 4096 join takes the whole absolute-position
//    correction on the public side: the encrypted query keeps the angle
//    `t * theta` that `BuildRope` already gives it, and the public key at
//    absolute `p` is rotated at `p - ptok` instead of `p`. This says the two
//    score matrices are the same one.
//
//    No weights and no GPU: it is a property of the rotation.
// ---------------------------------------------------------------------------
TEST(PublicPrefill, RopeIsRelativeSoTheEncryptedBranchDoesNotMove) {
  constexpr int kPtok = 3968;
  std::mt19937_64 gen(0xA11CE);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> q0(kDim), k0(kDim);
  for (int d = 0; d < kDim; d++) {
    q0[d] = dist(gen);
    k0[d] = dist(gen);
  }

  double worst = 0.0, mag = 0.0;
  for (int t : {0, 1, 63, 127}) {
    for (int p : {0, 1, 1000, 3967}) {
      // What the T = 4096 model really means: absolute positions.
      std::vector<double> qa = q0, ka = k0;
      Rope(qa, kPtok + t);
      Rope(ka, p);
      const double truth = Dot(qa, ka);
      // What the code does: the encrypted branch untouched, the public key
      // rotated at `p - ptok`.
      std::vector<double> qb = q0, kb = k0;
      Rope(qb, t);
      Rope(kb, p - kPtok);
      const double ours = Dot(qb, kb);
      worst = std::max(worst, std::abs(ours - truth));
      mag = std::max(mag, std::abs(truth));
    }
  }
  std::cout << std::scientific << std::setprecision(3)
            << "  |shifted - absolute|             : " << worst << " of "
            << mag << "   (relative 2^" << std::fixed << std::setprecision(1)
            << std::log2(worst / mag) << ")" << std::endl;
  // Float64 rounding of two large angles, not an approximation of anything.
  EXPECT_LT(worst, 1e-9 * mag);
}

// ---------------------------------------------------------------------------
// 2. THE SINK RESCALE is [SYLPH] 3.4's, over the WHOLE context.
//
//    It brings the first rows to the geometric-mean power of the rest BEFORE
//    the norm, so it cannot be done chunk by chunk -- which is exactly why it
//    is a separate call the driver makes once. Checked against the recipe.
// ---------------------------------------------------------------------------
TEST(PublicPrefill, TheSinkRescaleIsTheGeometricMeanOfTheBody) {
  constexpr int kB = 3, kT = 64, kSink = 2;
  std::mt19937_64 gen(0xBEEF);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(size_t(kB) * kT * kModel);
  for (auto &v : x) v = dist(gen);
  // Make the sinks loud, as a beginning-of-sequence row really is.
  for (int b = 0; b < kB; b++) {
    for (int t = 0; t < kSink; t++) {
      double *row = &x[(size_t(b) * kT + t) * kModel];
      for (int c = 0; c < kModel; c++) row[c] *= 30.0;
    }
  }

  std::vector<double> fac;
  PublicSinkRescale(fac, x.data(), kModel, kB, kT, kSink);
  ASSERT_EQ(fac.size(), size_t(kB) * kT);

  double worst = 0.0;
  for (int b = 0; b < kB; b++) {
    std::vector<double> ms(kT);
    for (int t = 0; t < kT; t++) {
      const double *row = &x[(size_t(b) * kT + t) * kModel];
      double s = 0.0;
      for (int c = 0; c < kModel; c++) s += row[c] * row[c];
      ms[t] = s / kModel;
    }
    double logsum = 0.0;
    for (int t = kSink; t < kT; t++) logsum += std::log(ms[t]);
    const double target = std::exp(logsum / (kT - kSink));
    for (int t = 0; t < kT; t++) {
      const double want = (t < kSink) ? std::sqrt(target / ms[t]) : 1.0;
      worst = std::max(worst,
                       std::abs(fac[size_t(b) * kT + t] - want) / want);
      // A rescaled sink really does land on the body's power.
      if (t < kSink) {
        const double got = ms[t] * want * want;
        EXPECT_LT(std::abs(got - target), 1e-9 * target);
      }
    }
  }
  std::cout << std::scientific << std::setprecision(3)
            << "  worst relative factor error      : " << worst << std::endl;
  EXPECT_LT(worst, 1e-12);
}

// ---------------------------------------------------------------------------
// 3. THE K AND V THEMSELVES, on the real layer-0 weights, against a host
//    computation written from the recipe rather than from the code under
//    test -- and laid out the way `CiPcAttention::EncodeKeys` reads them,
//    because a transposed operand is the failure this cannot afford.
//
//    Also that the CHUNKING is seamless: a chunk boundary is not a place
//    where the answer changes.
// ---------------------------------------------------------------------------
TEST(PublicPrefill, TheRealLayerZeroKvMatchesTheHost) {
  const std::string ld = WeightDir();
  if (ld.empty()) GTEST_SKIP() << "LLAMA3_ALL_DIR is not set";

  std::vector<float> wk, wv, an, x0;
  const std::string l0 = ld + "/L00";
  ASSERT_TRUE(ReadF32(l0 + "/wk.f32", size_t(kModel) * kCols, wk)) << l0;
  ASSERT_TRUE(ReadF32(l0 + "/wv.f32", size_t(kModel) * kCols, wv));
  ASSERT_TRUE(ReadF32(l0 + "/attn_norm.f32", kModel, an));

  // The 4096-token embedding: layer 0's input, and the public half of a
  // T = 4096 context is a prefix of it.
  const char *in_env = std::getenv("PC4096_INPUT");
  const std::string in = (in_env != nullptr)
                             ? std::string(in_env)
                             : ld + "/input_nosink_4096.f32";
  const int tokens = 4096;
  if (!ReadF32(in, size_t(tokens) * kModel, x0)) {
    GTEST_SKIP() << "no 4096-token input at " << in
                 << " (PC4096_INPUT overrides)";
  }

  // A modest instance count and context: this is a CONVENTION check, and the
  // convention does not depend on either. Instance b is the prompt at its own
  // factor, which is how every batched test here makes 512 of one prompt.
  constexpr int kB = 4;
  constexpr int kPtok = 96;
  constexpr int kEnc = 128;
  constexpr int kKvHead = 3;
  constexpr double kCk = 0.375;

  std::vector<double> hidden(size_t(kB) * kPtok * kModel);
  for (int b = 0; b < kB; b++) {
    const double f = 0.98 + 0.01 * b;
    for (int p = 0; p < kPtok; p++) {
      for (int c = 0; c < kModel; c++) {
        hidden[(size_t(b) * kPtok + p) * kModel + c] =
            f * double(x0[size_t(p) * kModel + c]);
      }
    }
  }

  PublicPrefillLayer layer;
  layer.wk = wk.data();
  layer.wv = wv.data();
  layer.attn_norm = an.data();
  layer.ck = kCk;

  // The whole context in one call, then the same context in chunks.
  const auto run = [&](int chunk, std::vector<double> &kall,
                       std::vector<double> &vall) {
    kall.assign(size_t(kDim) * kPtok * kB, 0.0);
    vall.assign(size_t(kPtok) * kDim * kB, 0.0);
    for (int start = 0; start < kPtok; start += chunk) {
      const int w = std::min(chunk, kPtok - start);
      std::vector<double> kb(size_t(kDim) * w * kB, 0.0);
      std::vector<double> vb(size_t(w) * kDim * kB, 0.0);
      // The chunk's rows, [instance][p][model].
      std::vector<double> h(size_t(kB) * w * kModel);
      for (int b = 0; b < kB; b++) {
        std::memcpy(&h[size_t(b) * w * kModel],
                    &hidden[(size_t(b) * kPtok + start) * kModel],
                    size_t(w) * kModel * sizeof(double));
      }
      // The RoPE position of the chunk's first token is `start - ptok`: the
      // encrypted block sits at 0..127 as far as `BuildRope` is concerned.
      PublicPrefillChunk(kb, vb, layer, h.data(), kB, w, kKvHead,
                         start - kPtok);
      for (int d = 0; d < kDim; d++) {
        for (int p = 0; p < w; p++) {
          for (int b = 0; b < kB; b++) {
            kall[(size_t(d) * kPtok + start + p) * kB + b] =
                kb[(size_t(d) * w + p) * kB + b];
          }
        }
      }
      for (int p = 0; p < w; p++) {
        for (int d = 0; d < kDim; d++) {
          for (int b = 0; b < kB; b++) {
            vall[(size_t(start + p) * kDim + d) * kB + b] =
                vb[(size_t(p) * kDim + d) * kB + b];
          }
        }
      }
    }
  };

  std::vector<double> k_one, v_one, k_many, v_many;
  run(kPtok, k_one, v_one);
  run(13, k_many, v_many);  // a chunk that divides nothing

  double chunk_diff = 0.0, chunk_mag = 0.0;
  for (size_t i = 0; i < k_one.size(); i++) {
    chunk_diff = std::max(chunk_diff, std::abs(k_one[i] - k_many[i]));
    chunk_mag = std::max(chunk_mag, std::abs(k_one[i]));
  }
  for (size_t i = 0; i < v_one.size(); i++) {
    chunk_diff = std::max(chunk_diff, std::abs(v_one[i] - v_many[i]));
  }

  // Against the recipe, on a sample of (instance, token).
  double worst_k = 0.0, worst_v = 0.0, mag_k = 0.0, mag_v = 0.0;
  std::vector<double> kh, vh;
  for (int b : {0, kB - 1}) {
    for (int p : {0, 1, kPtok / 2, kPtok - 1}) {
      HostKv(kh, vh, &hidden[(size_t(b) * kPtok + p) * kModel], wk, wv, an,
             kKvHead, double(p - kPtok), kCk);
      for (int d = 0; d < kDim; d++) {
        const double gk = k_one[(size_t(d) * kPtok + p) * kB + b];
        const double gv = v_one[(size_t(p) * kDim + d) * kB + b];
        worst_k = std::max(worst_k, std::abs(gk - kh[d]));
        worst_v = std::max(worst_v, std::abs(gv - vh[d]));
        mag_k = std::max(mag_k, std::abs(kh[d]));
        mag_v = std::max(mag_v, std::abs(vh[d]));
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3)
            << "  |K - host|                       : " << worst_k << " of "
            << mag_k << std::endl
            << "  |V - host|                       : " << worst_v << " of "
            << mag_v << std::endl
            << "  one chunk vs 13-token chunks     : " << chunk_diff << " of "
            << chunk_mag << std::endl;

  EXPECT_GT(mag_k, 1e-6) << "the reference K is ~zero";
  EXPECT_GT(mag_v, 1e-6) << "the reference V is ~zero";
  // Both sides are float64 over the same f32 weights in the same order, so
  // the only difference is the order of the channel sum.
  EXPECT_LT(worst_k, 1e-9 * mag_k);
  EXPECT_LT(worst_v, 1e-9 * mag_v);
  EXPECT_EQ(chunk_diff, 0.0) << "a chunk boundary changed the answer";
}
