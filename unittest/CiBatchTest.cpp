// The BATCHED layout on R+, and [KANG] Algorithm 1 as its projection.
//
// One channel is one ciphertext; the ciphertext's `degree` real slots hold
// `num_instances = degree / T` independent prompts of T tokens. A projection
// is then `res[o] = sum_c W[c][o] ct[c]` -- a scalar combination of whole
// ciphertexts through the int8 GEMM -- with no decomposition and no key
// switch, and it commutes with the slot encoding, so it runs on slot-form
// ciphertexts as they stand. See `extension/CiBatch.h`.
//
// TWO CLAIMS, TWO TESTS.
//
//  1. The product is right, per instance, per token, per channel, against a
//     host loop -- at a small width where the whole reference is cheap, at
//     two levels (the product is level-agnostic, which is the point of the
//     slot-form projection), with the output landing canonical one level
//     down. A leak between instances would fail this: every instance carries
//     its own input.
//
//  2. At the MODEL'S width, on the real layer-0 weights (LLAMA3_ALL_DIR):
//     what the seven projections of a layer cost for `num_instances`
//     prompts at once, and what they land at against float64 on a sample
//     of instances. This is the number the branch exists to see -- the
//     per-prompt cost of a projection with the ModPack gone.
//
// SEPARATE BINARY: RingFixture, one ring; it needs the extension only for
// `GpuEncoder` through the Context and the NVTX scopes.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "common/Assert.h"
#include "core/BatchCcmm.h"
#include "core/CiLift.h"
#include "common/ParallelFor.h"
#include "extension/BootContext.h"
#include "extension/CiBatch.h"
#include "extension/CiBatchAttention.h"
#include "extension/CiBatchLayer.h"

using word = uint32_t;
using cheddar::BootContext;
using cheddar::CiBatchLayout;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::Plaintext;
using Ring = ringfixture::Ring<word>;

namespace {

const char *Param() {
  const char *env = std::getenv("CHEDDAR_CI_BATCH_PARAM");
  return (env && env[0]) ? env : "ci16_35.json";
}
int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return (e && e[0]) ? std::atoi(e) : fallback;
}
constexpr int kTokens = 128;

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
Clock::time_point Sync() {
  cudaDeviceSynchronize();
  return Clock::now();
}

// A tensor of the layout, on the host: `x[(b * T + t) * C + c]`.
struct HostTensor {
  int instances = 0, tokens = 0, channels = 0;
  std::vector<double> v;
  double &At(int b, int t, int c) {
    return v[(static_cast<size_t>(b) * tokens + t) * channels + c];
  }
  double At(int b, int t, int c) const {
    return v[(static_cast<size_t>(b) * tokens + t) * channels + c];
  }
};

// Every channel of `x` as one ciphertext at `level`, in the layout.
void EncryptChannels(Ring &ring, const CiBatchLayout &layout,
                     const HostTensor &x, int level,
                     std::vector<Ciphertext<word>> &cts) {
  const double scale = ring.param->GetScale(level);
  cts.clear();
  cts.resize(x.channels);
  std::vector<double> values(static_cast<size_t>(layout.num_instances) *
                             layout.num_tokens);
  std::vector<Complex> msg;
  for (int c = 0; c < x.channels; c++) {
    for (int b = 0; b < layout.num_instances; b++) {
      for (int t = 0; t < layout.num_tokens; t++) {
        values[static_cast<size_t>(b) * layout.num_tokens + t] = x.At(b, t, c);
      }
    }
    layout.Pack(msg, values);
    Plaintext<word> pt;
    ring.context->gpu_encoder_.Encode(pt, level, scale, msg);
    ring.ui->Encrypt(cts[c], pt);
  }
}

// The channels `which` of `cts`, decrypted into `y` (which has every channel
// of the output's width; the ones not asked for stay zero).
void DecryptChannels(Ring &ring, const CiBatchLayout &layout,
                     const std::vector<Ciphertext<word>> &cts,
                     const std::vector<int> &which, HostTensor &y) {
  std::vector<Complex> msg;
  std::vector<double> values;
  for (int c : which) {
    Plaintext<word> pt;
    ring.ui->Decrypt(pt, cts[c]);
    ring.context->encoder_.Decode(msg, pt);
    layout.Unpack(values, msg);
    for (int b = 0; b < y.instances; b++) {
      for (int t = 0; t < y.tokens; t++) {
        y.At(b, t, c) = values[static_cast<size_t>(b) * layout.num_tokens + t];
      }
    }
  }
}

// `y = x W`, W `[in][out]` row-major, over the instances `bs` and the
// channels `os` only.
void HostProject(const HostTensor &x, const std::vector<float> &w, int out,
                 const std::vector<int> &bs, const std::vector<int> &os,
                 HostTensor &y) {
  const int in = x.channels;
  y.instances = x.instances;
  y.tokens = x.tokens;
  y.channels = out;
  y.v.assign(static_cast<size_t>(x.instances) * x.tokens * out, 0.0);
  cheddar::ParallelFor(static_cast<int>(os.size()), [&](int begin, int end) {
    for (int oi = begin; oi < end; oi++) {
      const int o = os[oi];
      for (int b : bs) {
        for (int t = 0; t < x.tokens; t++) {
          double acc = 0.0;
          const double *row =
              &x.v[(static_cast<size_t>(b) * x.tokens + t) * in];
          for (int c = 0; c < in; c++) {
            acc += row[c] *
                   static_cast<double>(w[static_cast<size_t>(c) * out + o]);
          }
          y.At(b, t, o) = acc;
        }
      }
    }
  });
}

struct Err {
  double rms_rel = 0.0, max_abs = 0.0, rms_ref = 0.0;
};
Err Compare(const HostTensor &got, const HostTensor &want,
            const std::vector<int> &bs, const std::vector<int> &os) {
  double se = 0.0, sr = 0.0, mx = 0.0;
  size_t n = 0;
  for (int b : bs) {
    for (int t = 0; t < want.tokens; t++) {
      for (int o : os) {
        const double d = got.At(b, t, o) - want.At(b, t, o);
        se += d * d;
        sr += want.At(b, t, o) * want.At(b, t, o);
        mx = std::max(mx, std::abs(d));
        n++;
      }
    }
  }
  Err e;
  e.rms_rel = std::sqrt(se / sr);
  e.max_abs = mx;
  e.rms_ref = std::sqrt(sr / static_cast<double>(n));
  return e;
}
double Bits(double rel) { return -std::log2(rel); }

bool ReadF32(const std::string &path, size_t count, std::vector<float> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.resize(count);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  return static_cast<size_t>(f.gcount()) == count * sizeof(float);
}

void ToDevice(cheddar::DeviceVector<float> &d, const std::vector<float> &h) {
  cheddar::HostVector<float> hv(h.size());
  std::copy(h.begin(), h.end(), hv.begin());
  d.resize(static_cast<int>(hv.size()));
  cheddar::CopyHostToDevice(d, hv);
}

size_t FreeMiB() {
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  return f >> 20;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The product against the host, at a small width, at two levels.
// ---------------------------------------------------------------------------
TEST(CiBatch, TheKangProjectionMatchesTheHostPerInstance) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring ring(Param());
  const CiBatchLayout layout(ring.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  std::cout << "  layout: " << layout.num_slots << " slots = " << B
            << " instances x " << kTokens << " tokens" << std::endl;

  constexpr int kIn = 48, kOut = 24;
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> ux(-1.0, 1.0), uw(-0.05, 0.05);
  HostTensor x{B, kTokens, kIn, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * kIn);
  for (auto &v : x.v) v = ux(gen);
  std::vector<float> w(static_cast<size_t>(kIn) * kOut);
  for (auto &v : w) v = static_cast<float>(uw(gen));
  cheddar::DeviceVector<float> w_dev;
  ToDevice(w_dev, w);

  std::vector<int> all_b(B), all_o(kOut);
  for (int b = 0; b < B; b++) all_b[b] = b;
  for (int o = 0; o < kOut; o++) all_o[o] = o;
  HostTensor want;
  HostProject(x, w, kOut, all_b, all_o, want);

  typename cheddar::CiBatchProjection<word>::Config cfg;
  cfg.rows_per_tile = 10;  // three tiles of 10, 10, 4: the tiling is on
  cfg.verbose = true;
  cheddar::CiBatchProjection<word> proj(ring.context, cfg);

  for (int level : {1, 8}) {
    if (level > ring.param->max_level_) continue;
    std::vector<Ciphertext<word>> cts;
    EncryptChannels(ring, layout, x, level, cts);
    proj.Prepare("w", w_dev.data(), kIn, kOut, level);

    std::vector<Ciphertext<word>> res;
    proj.Project(res, cts, "w");
    ASSERT_EQ(static_cast<int>(res.size()), kOut);
    for (const auto &r : res) {
      EXPECT_EQ(ring.param->NPToLevel(r.GetNP()), level - 1)
          << "the product rescales once";
      EXPECT_NEAR(r.GetScale() / ring.param->GetScale(level - 1), 1.0, 1e-9)
          << "the output is canonical at its level";
    }

    HostTensor got{B, kTokens, kOut, {}};
    got.v.assign(static_cast<size_t>(B) * kTokens * kOut, 0.0);
    DecryptChannels(ring, layout, res, all_o, got);
    const Err e = Compare(got, want, all_b, all_o);
    std::cout << "  level " << level << " -> " << (level - 1) << ": rms rel "
              << std::scientific << std::setprecision(3) << e.rms_rel << " ("
              << std::fixed << std::setprecision(2) << Bits(e.rms_rel)
              << " bits), max abs " << std::scientific << e.max_abs
              << ", ref rms " << e.rms_ref << std::endl;
    EXPECT_LT(e.rms_rel, std::ldexp(1.0, -15))
        << "the product is exact mod q: what remains is the rescale's rounding "
           "under the dense secret, ~2^-20 absolute per slot at scale 2^35";

    // Per-instance separation: the worst instance is not worse than the
    // whole, which a leak between instances would break.
    double worst = 0.0;
    for (int b = 0; b < B; b++) {
      worst = std::max(worst, Compare(got, want, {b}, all_o).rms_rel);
    }
    EXPECT_LT(worst, std::ldexp(1.0, -14));
  }
#endif
}

// ---------------------------------------------------------------------------
// 2. The model's seven projections at its width, on the real layer-0 weights.
// ---------------------------------------------------------------------------
TEST(CiBatch, TheRealLayerZeroProjectionsAtFullWidth) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  const char *wdir_env = std::getenv("LLAMA3_ALL_DIR");
  if (wdir_env == nullptr) GTEST_SKIP() << "LLAMA3_ALL_DIR is not set";
  const std::string ld = std::string(wdir_env) + "/L00";
  constexpr int kH = 4096, kKv = 1024, kI = 14336;

  Ring ring(Param());
  const CiBatchLayout layout(ring.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  const int level = EnvInt("CHEDDAR_CI_BATCH_LEVEL", 1);
  const int check_instances = EnvInt("CHEDDAR_CI_BATCH_CHECK", 4);
  std::cout << "  " << Param() << ": " << B << " instances x " << kTokens
            << " tokens, product at level " << level << " ("
            << ring.param->LevelToNP(level).GetNumTotal() << " limbs), "
            << FreeMiB() << " MiB free" << std::endl;

  // The layer's input for every instance: the embedding output of the
  // recorded prompt, each instance scaled by its own factor so that no two
  // instances carry the same data.
  std::vector<float> input;
  ASSERT_TRUE(ReadF32(ld + "/../input_nosink.f32",
                      static_cast<size_t>(kTokens) * kH, input))
      << "input_nosink.f32 beside the layer directories";
  // Normalised per token to unit rms, which is what the projections read in
  // the layer (an RMSNorm output), so that the relative error is stated
  // against a representative magnitude: the product's floor is ABSOLUTE
  // (the rescale's rounding, ~2^-20 a slot at scale 2^35).
  HostTensor x{B, kTokens, kH, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * kH);
  for (int b = 0; b < B; b++) {
    const double f = 0.5 + static_cast<double>(b) / B;
    for (int t = 0; t < kTokens; t++) {
      double ms = 0.0;
      for (int c = 0; c < kH; c++) {
        const double v = input[static_cast<size_t>(t) * kH + c];
        ms += v * v;
      }
      const double r = 1.0 / std::sqrt(ms / kH + 1e-12);
      for (int c = 0; c < kH; c++) {
        x.At(b, t, c) = f * r * input[static_cast<size_t>(t) * kH + c];
      }
    }
  }

  auto t0 = Sync();
  std::vector<Ciphertext<word>> stream;
  EncryptChannels(ring, layout, x, level, stream);
  auto t1 = Sync();
  std::cout << "  encrypt " << kH << " channel ciphertexts: " << std::fixed
            << std::setprecision(1) << Ms(t0, t1) << " ms, " << FreeMiB()
            << " MiB free" << std::endl;

  typename cheddar::CiBatchProjection<word>::Config cfg;
  cfg.rows_per_tile = EnvInt("CHEDDAR_CI_BATCH_TILE", 2048);
  cheddar::CiBatchProjection<word> proj(ring.context, cfg);

  std::vector<int> bs;
  for (int b = 0; b < std::min(check_instances, B); b++) {
    bs.push_back((b * B) / std::min(check_instances, B));
  }

  struct Row {
    const char *name;
    int in, out;
    double prepare_ms, project_ms, mib;
    Err err;
  };
  std::vector<Row> rows;

  auto run = [&](const char *name, const char *file, int in, int out,
                 const std::vector<Ciphertext<word>> &src,
                 const HostTensor &src_host, int at_level,
                 std::vector<Ciphertext<word>> &dst, HostTensor *dst_host) {
    std::vector<float> w;
    ASSERT_TRUE(ReadF32(ld + "/" + file, static_cast<size_t>(in) * out, w))
        << file;
    cheddar::DeviceVector<float> w_dev;
    ToDevice(w_dev, w);
    auto p0 = Sync();
    proj.Prepare(name, w_dev.data(), in, out, at_level);
    auto p1 = Sync();
    proj.Project(dst, src, name);
    auto p2 = Sync();
    ASSERT_EQ(static_cast<int>(dst.size()), out);

    // Checked on a sample of output channels spread over the width and on
    // `bs` instances; every token.
    std::vector<int> os;
    const int stride = std::max(1, out / 64);
    for (int o = 0; o < out; o += stride) os.push_back(o);
    HostTensor want;
    HostProject(src_host, w, out, bs, os, want);
    HostTensor got{B, kTokens, out, {}};
    got.v.assign(static_cast<size_t>(B) * kTokens * out, 0.0);
    DecryptChannels(ring, layout, dst, os, got);
    Row r{name, in, out, Ms(p0, p1), Ms(p1, p2),
          static_cast<double>(proj.Get(name).bytes) / (1024.0 * 1024.0),
          Compare(got, want, bs, os)};
    rows.push_back(r);
    std::cout << "  " << std::left << std::setw(6) << name << std::right
              << std::setw(6) << in << " -> " << std::setw(6) << out
              << "  prepare " << std::fixed << std::setprecision(1)
              << std::setw(8) << r.prepare_ms << " ms  project "
              << std::setw(8) << r.project_ms << " ms  (" << std::setprecision(3)
              << r.project_ms / B << " ms/instance)  " << std::setprecision(0)
              << r.mib << " MiB  rms rel 2^-" << std::setprecision(2)
              << Bits(r.err.rms_rel) << "  " << FreeMiB() << " MiB free"
              << std::endl;
    if (dst_host != nullptr) {
      // The full host product for the next stage's input: every instance,
      // every channel -- on the sampled instances only, which is all the
      // next check reads.
      std::vector<int> all_o(out);
      for (int o = 0; o < out; o++) all_o[o] = o;
      HostProject(src_host, w, out, bs, all_o, *dst_host);
    }
    proj.Release(name);
  };

  std::vector<Ciphertext<word>> q, k, v, o, g, u, d;
  run("q", "wq.f32", kH, kH, stream, x, level, q, nullptr);
  q.clear();
  run("k", "wk.f32", kH, kKv, stream, x, level, k, nullptr);
  k.clear();
  run("v", "wv.f32", kH, kKv, stream, x, level, v, nullptr);
  v.clear();
  run("o", "wo.f32", kH, kH, stream, x, level, o, nullptr);
  o.clear();
  // The down projection reads the hidden width: its input is `up`'s output
  // (one level down), which is 14336 ciphertexts and the widest contraction
  // of the layer.
  HostTensor up_host;
  run("gate", "wgate.f32", kH, kI, stream, x, level, g, nullptr);
  g.clear();
  run("up", "wup.f32", kH, kI, stream, x, level, u, &up_host);
  stream.clear();
  if (level - 1 >= 1) {
    run("down", "wdown.f32", kI, kH, u, up_host, level - 1, d, nullptr);
  } else {
    std::cout << "  down: skipped, the up output is at level 0 (run with "
                 "CHEDDAR_CI_BATCH_LEVEL >= 2)"
              << std::endl;
  }

  double total = 0.0;
  for (const auto &r : rows) {
    total += r.project_ms;
    EXPECT_LT(r.err.rms_rel, std::ldexp(1.0, -13)) << r.name;
  }
  std::cout << "  seven projections: " << std::fixed << std::setprecision(1)
            << total << " ms for " << B << " instances = " << std::setprecision(2)
            << total / B << " ms per instance-layer" << std::endl;
#endif
}

// ---------------------------------------------------------------------------
// 3. The feed-forward half of layer 0 on the real weights, B instances at
//    once, against the float64 reference per instance.
// ---------------------------------------------------------------------------
namespace {

bool ReadF64(const std::string &path, size_t count, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.resize(count);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(count * sizeof(double)));
  return static_cast<size_t>(f.gcount()) == count * sizeof(double);
}

// `reference_forward.py`'s feed-forward half in float64 on one instance:
// hr = sink * h, hn = rms_norm(hr, gain), y = (silu(hn Wg) * (hn Wu)) Wd,
// out = h + y. `h` is `[T][H]`; `y` and `out` come back the same shape.
void HostFfn(const std::vector<double> &h, int T, int H, int I,
             const std::vector<double> &sink, const std::vector<float> &gain,
             const std::vector<float> &wg, const std::vector<float> &wu,
             const std::vector<float> &wd, double eps, std::vector<double> &y,
             std::vector<double> &out) {
  std::vector<double> hn(static_cast<size_t>(T) * H);
  for (int t = 0; t < T; t++) {
    const double s = sink.empty() ? 1.0 : sink[t];
    double ms = 0.0;
    for (int c = 0; c < H; c++) {
      const double v = s * h[static_cast<size_t>(t) * H + c];
      ms += v * v;
    }
    ms /= H;
    const double r = 1.0 / std::sqrt(ms + eps);
    for (int c = 0; c < H; c++) {
      hn[static_cast<size_t>(t) * H + c] =
          s * h[static_cast<size_t>(t) * H + c] * r * gain[c];
    }
  }
  std::vector<double> gu(static_cast<size_t>(T) * I);
  cheddar::ParallelFor(I, [&](int begin, int end) {
    for (int j = begin; j < end; j++) {
      for (int t = 0; t < T; t++) {
        double g = 0.0, u = 0.0;
        const double *row = &hn[static_cast<size_t>(t) * H];
        for (int c = 0; c < H; c++) {
          g += row[c] * static_cast<double>(wg[static_cast<size_t>(c) * I + j]);
          u += row[c] * static_cast<double>(wu[static_cast<size_t>(c) * I + j]);
        }
        gu[static_cast<size_t>(t) * I + j] = g / (1.0 + std::exp(-g)) * u;
      }
    }
  });
  y.assign(static_cast<size_t>(T) * H, 0.0);
  out.assign(static_cast<size_t>(T) * H, 0.0);
  cheddar::ParallelFor(H, [&](int begin, int end) {
    for (int c = begin; c < end; c++) {
      for (int t = 0; t < T; t++) {
        double acc = 0.0;
        const double *row = &gu[static_cast<size_t>(t) * I];
        for (int j = 0; j < I; j++) {
          acc += row[j] * static_cast<double>(wd[static_cast<size_t>(j) * H + c]);
        }
        y[static_cast<size_t>(t) * H + c] = acc;
        out[static_cast<size_t>(t) * H + c] = h[static_cast<size_t>(t) * H + c] + acc;
      }
    }
  });
}

// rms relative error of `got` against `want` over tokens `[t0, T)`.
double RmsRel(const std::vector<double> &got, const std::vector<double> &want,
              int T, int H, int t0) {
  double se = 0.0, sr = 0.0;
  for (int t = t0; t < T; t++) {
    for (int c = 0; c < H; c++) {
      const size_t i = static_cast<size_t>(t) * H + c;
      se += (got[i] - want[i]) * (got[i] - want[i]);
      sr += want[i] * want[i];
    }
  }
  return std::sqrt(se / sr);
}

}  // namespace

TEST(CiBatch, TheFeedForwardRunsOnTheRealLayerZero) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  const char *wdir_env = std::getenv("LLAMA3_ALL_DIR");
  const char *rdir_env = std::getenv("LLAMA3_REF_DIR");
  if (wdir_env == nullptr || rdir_env == nullptr) {
    GTEST_SKIP() << "LLAMA3_ALL_DIR and LLAMA3_REF_DIR must both be set";
  }
  const std::string ld = std::string(wdir_env) + "/L00";
  const std::string rd = rdir_env;
  constexpr int kH = 4096, kI = 14336, kSinkTokens = 2;
  const double eps = 1e-5;

  // The layer's inputs and the reference: x0 (the embedding), the clear
  // attention output av, so that h = x0 + av Wo is the post-attention
  // residual, and h_L00 = h + ffn(h) the layer's output in float64.
  std::vector<float> x0, wo, wg, wu, wd, gain;
  std::vector<double> av, h_ref;
  ASSERT_TRUE(ReadF32(ld + "/../input_nosink.f32", static_cast<size_t>(kTokens) * kH, x0));
  ASSERT_TRUE(ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, wo));
  ASSERT_TRUE(ReadF32(ld + "/wgate.f32", static_cast<size_t>(kH) * kI, wg));
  ASSERT_TRUE(ReadF32(ld + "/wup.f32", static_cast<size_t>(kH) * kI, wu));
  ASSERT_TRUE(ReadF32(ld + "/wdown.f32", static_cast<size_t>(kI) * kH, wd));
  ASSERT_TRUE(ReadF32(ld + "/ffn_norm.f32", kH, gain));
  ASSERT_TRUE(ReadF64(rd + "/av_L00.f64", static_cast<size_t>(kTokens) * kH, av));
  ASSERT_TRUE(ReadF64(rd + "/h_L00.f64", static_cast<size_t>(kTokens) * kH, h_ref));
  std::vector<double> h(static_cast<size_t>(kTokens) * kH);
  cheddar::ParallelFor(kH, [&](int begin, int end) {
    for (int c = begin; c < end; c++) {
      for (int t = 0; t < kTokens; t++) {
        double acc = 0.0;
        for (int m = 0; m < kH; m++) {
          acc += av[static_cast<size_t>(t) * kH + m] *
                 static_cast<double>(wo[static_cast<size_t>(m) * kH + c]);
        }
        h[static_cast<size_t>(t) * kH + c] =
            static_cast<double>(x0[static_cast<size_t>(t) * kH + c]) + acc;
      }
    }
  });

  // The calibration, as the reference script fitted it for layer 0.
  cheddar::CiBatchLayer<word>::Calibration cal;
  double resid_absmax = 1.0;
  {
    std::ifstream f(rd + "/calib.json");
    ASSERT_TRUE(f.good()) << rd << "/calib.json";
    nlohmann::json cj = nlohmann::json::parse(f)["layers"][0];
    cal.alpha = cj["alpha"];
    cal.norm_window = cj["norm_window"];
    cal.silu_range = cj["silu_range"];
    resid_absmax = cj["resid_absmax"];
    cal.ffn_sink.assign(kTokens, 1.0);
    if (cj.contains("ffn_sink")) {
      const auto v = cj["ffn_sink"].get<std::vector<double>>();
      for (size_t i = 0; i < v.size() && i < cal.ffn_sink.size(); i++) {
        cal.ffn_sink[i] = v[i];
      }
    }
  }
  const double ride = [] {
    const char *e = std::getenv("CHEDDAR_CI_BATCH_RIDE");
    return (e && e[0]) ? std::atof(e) : 0.35;
  }();
  cal.stream_scale = ride / resid_absmax;
  std::cout << "  calibration: alpha " << cal.alpha << " window "
            << cal.norm_window << " silu_range " << cal.silu_range
            << " resid_absmax " << resid_absmax << " -> stream_scale "
            << cal.stream_scale << " (ride " << ride << ")" << std::endl;

  // The host reference of MY feed-forward on the recorded prompt must be
  // the exporter's layer output, or the reference below is not one.
  {
    std::vector<double> y1, out1;
    HostFfn(h, kTokens, kH, kI, cal.ffn_sink, gain, wg, wu, wd, eps, y1, out1);
    const double d = RmsRel(out1, h_ref, kTokens, kH, 0);
    std::cout << "  host feed-forward vs h_L00.f64: rms rel " << std::scientific
              << d << std::fixed << std::endl;
    ASSERT_LT(d, 1e-9) << "the host reference does not reproduce the exporter";
  }

  Ring ring(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(ring.context);
  ASSERT_NE(bctx, nullptr);
  const CiBatchLayout layout(ring.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;

  cheddar::CiBatchLayer<word>::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.model = kH;
  cfg.hidden = kI;
  cfg.eps = eps;
  cfg.rows_per_tile = EnvInt("CHEDDAR_CI_BATCH_TILE", 512);
  cfg.norm_apply_level = EnvInt("CHEDDAR_CI_BATCH_HOLD", 8);
  cfg.hold_channels = EnvInt("CHEDDAR_CI_BATCH_HOLD_CHANNELS", 1) != 0;
  cfg.verbose = true;
  auto t0 = Sync();
  cheddar::CiBatchLayer<word> layer(bctx, cfg);
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    layer.AddRequiredRotations(req);
    ring.ui->PrepareRotationKey(req);
  }
  auto t1 = Sync();
  std::cout << "  setup (boot tables + keys): " << std::fixed
            << std::setprecision(1) << Ms(t0, t1) / 1000.0 << " s, " << FreeMiB()
            << " MiB free" << std::endl;

  // The instances: the recorded prompt's residual at a per-instance factor,
  // 1.0 at instance B/2 (which the exporter's own output then checks).
  HostTensor x{B, kTokens, kH, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * kH);
  auto factor = [&](int b) { return 0.5 + static_cast<double>(b) / B; };
  for (int b = 0; b < B; b++) {
    const double f = factor(b) * cal.stream_scale;
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kH; c++) {
        x.At(b, t, c) = f * h[static_cast<size_t>(t) * kH + c];
      }
    }
  }
  std::vector<Ciphertext<word>> stream;
  EncryptChannels(ring, layout, x, 0, stream);
  x.v.clear();
  x.v.shrink_to_fit();
  auto t2 = Sync();
  std::cout << "  encrypt the stream (" << kH << " ciphertexts at level 0): "
            << Ms(t1, t2) / 1000.0 << " s, " << FreeMiB() << " MiB free"
            << std::endl;

  cheddar::DeviceVector<float> g_dev, u_dev, d_dev;
  ToDevice(g_dev, wg);
  ToDevice(u_dev, wu);
  ToDevice(d_dev, wd);
  cheddar::CiBatchLayer<word>::Weights w;
  w.gate = g_dev.data();
  w.up = u_dev.data();
  w.down = d_dev.data();
  w.ffn_norm.assign(gain.begin(), gain.end());

  bctx->ResetBootCounts();
  auto t3 = Sync();
  std::vector<Ciphertext<word>> res;
  layer.FeedForward(res, stream, w, cal, ring.ui->GetEvkMap());
  auto t4 = Sync();
  const auto st = layer.GetStages();
  const auto counts = bctx->GetBootCounts();
  std::cout << "  FEED-FORWARD, " << B << " instances: " << Ms(t3, t4) / 1000.0
            << " s wall = " << Ms(t3, t4) / B << " ms per instance; boots "
            << counts.full << "; stages boot " << st.boot << " norm " << st.norm
            << " gate/up " << st.gate_up << " silu " << st.silu << " down "
            << st.down << " s; " << FreeMiB() << " MiB free" << std::endl;
  ASSERT_EQ(static_cast<int>(res.size()), kH);

  // Checked instances: the four spread over the batch, B/2 among them.
  std::vector<int> bs = {0, B / 4, B / 2, B - 1};
  std::vector<int> all_c(kH);
  for (int c = 0; c < kH; c++) all_c[c] = c;
  HostTensor got{B, kTokens, kH, {}};
  got.v.assign(static_cast<size_t>(B) * kTokens * kH, 0.0);
  DecryptChannels(ring, layout, res, all_c, got);
  auto t5 = Sync();
  std::cout << "  decrypt " << kH << " ciphertexts: " << Ms(t4, t5) / 1000.0
            << " s" << std::endl;

  double worst_layer = 0.0;
  for (int b : bs) {
    std::vector<double> hb(static_cast<size_t>(kTokens) * kH);
    for (size_t i = 0; i < hb.size(); i++) hb[i] = factor(b) * h[i];
    std::vector<double> y_ref, out_ref;
    HostFfn(hb, kTokens, kH, kI, cal.ffn_sink, gain, wg, wu, wd, eps, y_ref,
            out_ref);
    std::vector<double> out_got(hb.size()), y_got(hb.size());
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kH; c++) {
        const size_t i = static_cast<size_t>(t) * kH + c;
        out_got[i] = got.At(b, t, c) / cal.stream_scale;
        y_got[i] = out_got[i] - hb[i];
      }
    }
    const double e_layer = RmsRel(out_got, out_ref, kTokens, kH, kSinkTokens);
    const double e_ffn = RmsRel(y_got, y_ref, kTokens, kH, kSinkTokens);
    const double e_all = RmsRel(out_got, out_ref, kTokens, kH, 0);
    std::cout << "  instance " << std::setw(3) << b << " (x" << std::fixed
              << std::setprecision(3) << factor(b) << "): layer output 2^-"
              << std::setprecision(2) << Bits(e_layer)
              << " over user tokens (2^-" << Bits(e_all)
              << " over all), the feed-forward alone 2^-" << Bits(e_ffn)
              << std::endl;
    worst_layer = std::max(worst_layer, e_layer);
  }
  EXPECT_LT(worst_layer, std::ldexp(1.0, -8))
      << "the batched feed-forward is far from the float64 reference";
#endif
}

// ---------------------------------------------------------------------------
// 4. The score product on the batched layout: Q K^T with K as projected.
//
// On the batched layout Q and K both come out of the projection one channel
// per ciphertext with the tokens in the blocks. [KANG] Algorithm 4 contracts
// the CIPHERTEXT index of both operands after its step 1 has made the second
// operand row-wise -- and K as projected IS the row-wise encryption of K^T
// (ciphertext c = row c of K^T = channel c over the tokens). So the score
// product skips step 1 (`rhs_row_wise`) and transposes nothing.
//
// What has to be checked is the lifted-ring twist of Doing.md 1.5bl. The lift
// puts Lambda = I + w^-1 P E on the BLOCK index of every lifted ciphertext,
// and in the elided form the blocks are the FREE indices of both operands, so
// the product comes back as Lambda S Lambda^T per lane: the CI read strips the
// outer Lambda (the lift structure itself) and the descent's trace turns the
// inner one into (I + cos(theta) P E) on the KEY-TOKEN axis -- score column l
// arrives as S[:, l] + cos(theta) S[:, d - l]. Confining K's live tokens to
// blocks x < d/2 kills the partner term identically for every l < d/2. That
// is the derivation; this test is whether the hardware agrees: one call,
// full 128-channel contraction, key tokens 0..63 live, and columns 0..63
// compared against the per-lane host product. The other 64 columns are
// where the flipped partners land and are reported, not asserted.
// ---------------------------------------------------------------------------
namespace {
using RealBatch = std::vector<std::vector<std::vector<double>>>;  // [lane][i][x]
}  // namespace

TEST(CiBatch, TheElidedScoreProductHoldsUnderTheContract) {
  Ring ci("ci12_35_boot.json");
  Ring big("ringdegree13_35_boot.json",
           cheddar::CiLiftHandler<word>::LiftSecret(ci.ui->GetSecretCoeffs()));
  const int n = ci.Degree();
  const int k = 32;  // CI lanes: the instances of one group
  const int d = n / k;  // 128: tokens as blocks, channels as ciphertexts
  const int half = d / 2;
  const int level = 2;
  const double scale = ci.param->GetScale(level);
  ASSERT_LE(level, ci.param->max_level_);

  cheddar::CiLiftHandler<word> lift(ci.context, big.context);
  cheddar::BatchCcmmHandler<word> ccmm(*big.param, big.context->ntt_handler_);
  for (int index : ccmm.RotationIndices(2 * k)) {
    big.ui->PrepareRotationKey(index, level);
  }

  // q[t][i][x]: lane t, token i, channel x -- every token live. k[t][l][x]:
  // key token l, channel x -- live only for l < d/2.
  std::mt19937_64 gen(0xB47C);
  std::uniform_real_distribution<double> dist(-0.15, 0.15);
  RealBatch q(k, std::vector<std::vector<double>>(d, std::vector<double>(d)));
  RealBatch kk(k, std::vector<std::vector<double>>(d, std::vector<double>(d, 0.0)));
  for (int t = 0; t < k; t++) {
    for (int i = 0; i < d; i++) {
      for (int x = 0; x < d; x++) q[t][i][x] = dist(gen);
    }
    for (int l = 0; l < half; l++) {
      for (int x = 0; x < d; x++) kk[t][l][x] = dist(gen);
    }
  }
  // Ciphertext x = channel x: block i (token), lane t of its CI SinC
  // message holds m[t][i][x]. Encrypted on R+, lifted.
  auto encrypt_channels = [&](const RealBatch &m,
                              std::vector<Ciphertext<word>> &out) {
    out.resize(d);
    std::vector<Complex> message(n);
    for (int x = 0; x < d; x++) {
      for (int i = 0; i < d; i++) {
        for (int t = 0; t < k; t++) {
          message[static_cast<size_t>(i) * k + t] = Complex(m[t][i][x], 0.0);
        }
      }
      Plaintext<word> pt;
      ci.context->encoder_.EncodeSinC(pt, level, scale, message, k);
      Ciphertext<word> ct;
      ci.ui->Encrypt(ct, pt);
      lift.Lift(out[x], ct);
    }
  };
  std::vector<Ciphertext<word>> lhs, rhs, res;
  encrypt_channels(q, lhs);
  encrypt_channels(kk, rhs);

  auto t0 = Sync();
  ccmm.Multiply(big.context, res, lhs, rhs, 2 * k, big.ui->GetEvkMap(),
                /*rhs_row_wise=*/true);
  auto t1 = Sync();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(res.size()), d);
  std::cout << "  elided Algorithm 4 at d = " << d << ", " << k
            << " lanes, on the lifted ring: " << Ms(t0, t1) << " ms" << std::endl;

  // Descend, decode: got[l][i * k + t] is score column l (key token), row i
  // (query token), lane t.
  double worst_live = 0.0, worst_dead = 0.0, ref_rms = 0.0;
  size_t n_ref = 0;
  for (int l = 0; l < d; l++) {
    Ciphertext<word> down;
    lift.Descend(down, res[l]);
    Plaintext<word> out;
    ci.ui->Decrypt(out, down);
    std::vector<Complex> got;
    ci.context->encoder_.DecodeSinC(got, out, k);
    // The descent's trace doubles the message and records the factor in the
    // scale; DecodeSinC reads the recorded scale, so `got` is in units.
    for (int i = 0; i < d; i++) {
      for (int t = 0; t < k; t++) {
        double want = 0.0;
        for (int x = 0; x < d; x++) want += q[t][i][x] * kk[t][l][x];
        const double g = got[static_cast<size_t>(i) * k + t].real();
        if (l < half) {
          worst_live = std::max(worst_live, std::abs(g - want));
          ref_rms += want * want;
          n_ref++;
        } else {
          worst_dead = std::max(worst_dead, std::abs(g - want));
        }
      }
    }
  }
  ref_rms = std::sqrt(ref_rms / static_cast<double>(n_ref));
  std::cout << "  key tokens < " << half << ": max |error| " << std::scientific
            << worst_live << " (reference rms " << ref_rms
            << "); key tokens >= " << half
            << " (the flipped partners' addresses): max |error| " << worst_dead
            << std::fixed << std::endl;
  EXPECT_LT(worst_live, 1e-3)
      << "the elided product is not Q K^T on the live key tokens";
}

// ---------------------------------------------------------------------------
// 5. One head's scores on the batched layout, end to end: RoPE, the chain
//    forward, the ring switch, the lift, the elided Algorithm 4 per group,
//    the descent, the switch back and the return to slots -- against the
//    host on a sample of instances.
// ---------------------------------------------------------------------------
TEST(CiBatch, TheScoresOfOneHeadMatchTheHost) {
  Ring boot("ci16_35.json");
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  Ring small("ci12_35_boot.json");
  Ring lifted("ringdegree13_35_boot.json",
              cheddar::CiLiftHandler<word>::LiftSecret(
                  small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  cheddar::CiBatchAttention<word>::Config cfg;
  cfg.verbose = true;
  auto t0 = Sync();
  cheddar::CiBatchAttention<word> attn(bctx, swtch.context, small.context,
                                       lifted.context, cfg);
  const int chain_level = attn.GetChainLevel();
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : attn.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  {
    cheddar::EvkRequest req;
    attn.AddSwitchRotations(req);
    swtch.ui->PrepareRotationKey(req);
  }
  auto t1 = Sync();
  std::cout << "  setup (three converters + keys): " << std::fixed
            << std::setprecision(1) << Ms(t0, t1) / 1000.0 << " s, "
            << FreeMiB() << " MiB free" << std::endl;

  const CiBatchLayout &layout = attn.GetLayout();
  const int B = layout.num_instances;
  const int D = cfg.head_dim;
  const int T = cfg.num_tokens;

  // Random Q and K for one head, every instance its own.
  std::mt19937_64 gen(0x5C0E);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  HostTensor q{B, T, D, {}}, k{B, T, D, {}};
  q.v.resize(static_cast<size_t>(B) * T * D);
  k.v.resize(q.v.size());
  for (auto &v : q.v) v = dist(gen);
  for (auto &v : k.v) v = dist(gen);
  std::vector<Ciphertext<word>> q_cts, k_cts;
  EncryptChannels(boot, layout, q, cfg.rope_level, q_cts);
  EncryptChannels(boot, layout, k, cfg.rope_level, k_cts);

  cheddar::CiBatchAttention<word>::Keys keys;
  keys.swtch = &swtch.ui->GetEvkMap();
  keys.lifted = &lifted.ui->GetEvkMap();
  keys.ring_switch = &swtch.ui->GetRingSwitchKey(attn.GetChain().rank);
  keys.inverse_ring_switch =
      &swtch.ui->GetInverseRingSwitchKey(attn.GetChain().rank);

  auto t2 = Sync();
  std::vector<Ciphertext<word>> scores;
  attn.Scores(scores, q_cts, k_cts, keys);
  auto t3 = Sync();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(scores.size()), T);
  std::cout << "  one head's scores for " << B << " instances: "
            << Ms(t2, t3) / 1000.0 << " s (" << Ms(t2, t3) / B
            << " ms per instance), out at level "
            << boot.param->NPToLevel(scores[0].GetNP()) << ", scale / canonical "
            << scores[0].GetScale() /
                   boot.param->GetScale(boot.param->NPToLevel(scores[0].GetNP()))
            << std::endl;

  // Host: RoPE both, S[b][i][l] = sum_c q'[b][i][c] k'[b][l][c].
  auto rope_host = [&](HostTensor &m) {
    const int half = D / 2;
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int c = 0; c < half; c++) {
          const double theta = std::pow(cfg.rope_base, -2.0 * c / D);
          const double a = t * theta;
          const double lo = m.At(b, t, c), hi = m.At(b, t, c + half);
          m.At(b, t, c) = lo * std::cos(a) - hi * std::sin(a);
          m.At(b, t, c + half) = hi * std::cos(a) + lo * std::sin(a);
        }
      }
    }
  };
  rope_host(q);
  rope_host(k);
  const std::vector<int> bs = {0, B / 3, B / 2, B - 1};
  std::vector<int> all_l(T);
  for (int l = 0; l < T; l++) all_l[l] = l;
  HostTensor got{B, T, T, {}};
  got.v.assign(static_cast<size_t>(B) * T * T, 0.0);
  DecryptChannels(boot, layout, scores, all_l, got);
  HostTensor want{B, T, T, {}};
  want.v.assign(got.v.size(), 0.0);
  for (int b : bs) {
    for (int i = 0; i < T; i++) {
      for (int l = 0; l < T; l++) {
        double acc = 0.0;
        for (int c = 0; c < D; c++) acc += q.At(b, i, c) * k.At(b, l, c);
        want.At(b, i, l) = acc;
      }
    }
  }
  const Err e = Compare(got, want, bs, all_l);
  std::vector<int> lo_l(all_l.begin(), all_l.begin() + T / 2);
  std::vector<int> hi_l(all_l.begin() + T / 2, all_l.end());
  const Err e_lo = Compare(got, want, bs, lo_l), e_hi = Compare(got, want, bs, hi_l);
  std::cout << "  scores vs host: rms rel 2^-" << std::setprecision(2)
            << Bits(e.rms_rel) << " (max abs " << std::scientific << e.max_abs
            << ", ref rms " << e.rms_ref << "); key tokens < " << T / 2
            << ": 2^-" << std::fixed << Bits(e_lo.rms_rel) << ", >= " << T / 2
            << ": 2^-" << Bits(e_hi.rms_rel) << std::endl;
  EXPECT_LT(e.rms_rel, std::ldexp(1.0, -8));
}
