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
#include "common/ParallelFor.h"
#include "extension/CiBatch.h"

using word = uint32_t;
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
    EXPECT_LT(e.rms_rel, std::ldexp(1.0, -18))
        << "the product is exact mod q; only the encoding rounding remains";

    // Per-instance separation: the worst instance is not worse than the
    // whole, which a leak between instances would break.
    double worst = 0.0;
    for (int b = 0; b < B; b++) {
      worst = std::max(worst, Compare(got, want, {b}, all_o).rms_rel);
    }
    EXPECT_LT(worst, std::ldexp(1.0, -16));
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
  HostTensor x{B, kTokens, kH, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * kH);
  for (int b = 0; b < B; b++) {
    const double f = 0.5 + static_cast<double>(b) / B;
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kH; c++) {
        x.At(b, t, c) = f * input[static_cast<size_t>(t) * kH + c];
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
  run("k", "wk.f32", kH, kKv, stream, x, level, k, nullptr);
  run("v", "wv.f32", kH, kKv, stream, x, level, v, nullptr);
  run("o", "wo.f32", kH, kH, stream, x, level, o, nullptr);
  q.clear();
  k.clear();
  v.clear();
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
    EXPECT_LT(r.err.rms_rel, std::ldexp(1.0, -16)) << r.name;
  }
  std::cout << "  seven projections: " << std::fixed << std::setprecision(1)
            << total << " ms for " << B << " instances = " << std::setprecision(2)
            << total / B << " ms per instance-layer" << std::endl;
#endif
}
