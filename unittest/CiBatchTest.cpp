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
#include "extension/ChebyshevFit.h"
#include "extension/CiDecode.h"
#include "extension/CiDecodeLayer.h"
#include "extension/Hoist.h"

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
// The fused scores' tower ring: K = 64 with the prefix landing where the
// layer preset's own Boot does. The default pairs the default layer preset
// `ci16_35` (landing 16, `bdeafd3`'s causal fold): the shipped
// `land17c3e10` (EvalMod ends at 17). ci16_35_stc2 (17) needs
// `land18c4e10`. The landing-15 layer preset `ci16_35_land17c4e8s2`
// rides the SAME land17c3e10 (the prefix enters after a LevelDown; the
// junction L16 ladder's EvalMod measured 2^-7) plus
// `CHEDDAR_CI_BATCH_AFFINE_PREFIX=1`.
const char *TowerParam() {
  const char *env = std::getenv("CHEDDAR_CI_BATCH_TOWER_PARAM");
  return (env && env[0]) ? env : "ci16_35_land17c3e10.json";
}
int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return (e && e[0]) ? std::atoi(e) : fallback;
}
constexpr int kTokens = 128;

// The norm's channel-boot ring (Doing.md 7.38): CHEDDAR_CI_BATCH_CHAN_PARAM
// names a gen_landing sub-ladder of the layer preset (ci16_35_land11c4e8s2,
// landing 9) on the SAME secret; NormTurn then sums the squares before any
// bootstrap, boots the one accumulator on the deep ring, and boots each
// channel once on this ring. Returns the Ring to keep alive, or null.
std::unique_ptr<Ring> WireChannelRing(Ring &boot,
                                      cheddar::CiBatchLayer<word> &layer) {
  const char *p = std::getenv("CHEDDAR_CI_BATCH_CHAN_PARAM");
  if (p == nullptr || p[0] == '\0') return nullptr;
  auto chan = std::make_unique<Ring>(p, boot.ui->GetSecretCoeffs());
  auto cctx =
      std::dynamic_pointer_cast<BootContext<word>>(chan->context);
  cheddar::AssertTrue(cctx != nullptr, "the channel ring's preset must boot");
  cctx->PrepareEvalMod();
  cctx->PrepareEvalSpecialFFT(layer.GetLayout().num_slots);
  {
    cheddar::EvkRequest req;
    cctx->AddRequiredRotations(req, layer.GetLayout().num_slots);
    chan->ui->PrepareRotationKey(req);
  }
  layer.SetChannelBoot(cctx, &chan->ui->GetEvkMap());
  std::cout << "  channel ring " << p << " landing "
            << cctx->GetBootParameter().GetEndLevel() << std::endl;
  return chan;
}

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

// The real-weight tests ride each instance at its own factor f in
// [0.5, 1.5): a norm window fitted on the recorded prompt (f = 1) must
// widen to the population -- the mean square scales by f^2 -- or the
// out-of-window instances' inverse square root (a Chebyshev evaluated
// past [-1, 1]) blows up by ~2^28, and at level 0 those slots wrap the
// modulus and poison every instance, the checked f = 1 one included.
void WidenForInstanceFactors(double &alpha, double &window,
                             double f_lo = 0.5, double f_hi = 1.5) {
  alpha /= f_lo * f_hi;  // 1/sqrt((lo f_lo^2)(hi f_hi^2)) restated
  window *= (f_hi * f_hi) / (f_lo * f_lo);
}

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
  WidenForInstanceFactors(cal.alpha, cal.norm_window);
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
  cfg.hold_channels_ffn = EnvInt("CHEDDAR_CI_BATCH_HOLD_CHANNELS_FFN", 1) != 0;
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
  std::unique_ptr<Ring> chan = WireChannelRing(ring, layer);
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
  EncryptChannels(ring, layout, x, chan ? 1 : 0, stream);
  x.v.clear();
  x.v.shrink_to_fit();
  auto t2 = Sync();
  std::cout << "  encrypt the stream (" << kH << " ciphertexts at level "
            << (chan ? 1 : 0) << "): "
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
  Ring boot(Param());
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
  {
    cheddar::EvkRequest req;
    attn.AddBootRotations(req);
    boot.ui->PrepareRotationKey(req);
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
  keys.boot = &boot.ui->GetEvkMap();
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
  HostTensor q_orig = q;  // pre-RoPE, for the repeat loop's re-encryptions
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

  // The layer calls Scores 32 times a layer against handler state (the
  // Cmt plans, the converters' scratch, the ping-pong buffers) that the
  // single call above never exercises twice, and reuses one K for the four
  // heads of a kv group. CHEDDAR_CI_BATCH_SCORES_REPEAT=N replays the SAME
  // product N more times (fresh Q encryption, the SAME K ciphertexts) and
  // compares every replay: a call-count- or reuse-dependent corruption
  // shows as a deviation that grows with the replay index.
  const int reps = EnvInt("CHEDDAR_CI_BATCH_SCORES_REPEAT", 0);
  for (int rep = 0; rep < reps; rep++) {
    std::vector<Ciphertext<word>> q2, s2;
    EncryptChannels(boot, layout, q_orig, cfg.rope_level, q2);
    attn.Scores(s2, q2, k_cts, keys);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    HostTensor g2{B, T, T, {}};
    g2.v.assign(static_cast<size_t>(B) * T * T, 0.0);
    DecryptChannels(boot, layout, s2, all_l, g2);
    const Err er = Compare(g2, want, bs, all_l);
    double mx = 0.0;
    int mb = 0, mt = 0, mi = 0;
    for (int b : bs) {
      for (int t = 0; t < T; t++) {
        for (int l = 0; l < T; l++) {
          const double d = std::abs(g2.At(b, t, l) - want.At(b, t, l));
          if (d > mx) {
            mx = d;
            mb = b; mt = t; mi = l;
          }
        }
      }
    }
    std::cout << "  [rep " << std::setw(2) << rep << "] rms rel 2^-"
              << std::fixed << std::setprecision(2) << Bits(er.rms_rel)
              << ", worst |dev| " << std::scientific << mx << " at (b "
              << std::fixed << mb << ", t " << mt << ", l " << mi << ")"
              << std::endl;
    EXPECT_LT(er.rms_rel, std::ldexp(1.0, -7)) << "replay " << rep;
  }
}

// ---------------------------------------------------------------------------
// 5a'. The ct-batched descend/return (B512_ccmm_ideas idea [2]) against the
//      serial converter loop, WORD FOR WORD: the same Scores call twice on
//      the same inputs, once with `HoistHandler::EvaluateBatch` forced to
//      its serial loop and once batched. Everything around the converters
//      is deterministic per ciphertext, so any differing word is the
//      batching's (the arena baby steps, the shared-table PAccum batch, the
//      grouped rotate/fold).
TEST(CiBatch, TheBatchedConverterIsWordForWord) {
  Ring boot(Param());
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  Ring small("ci12_35_boot.json");
  Ring lifted("ringdegree13_35_boot.json",
              cheddar::CiLiftHandler<word>::LiftSecret(
                  small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  cheddar::CiBatchAttention<word>::Config cfg;
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
  {
    cheddar::EvkRequest req;
    attn.AddBootRotations(req);
    boot.ui->PrepareRotationKey(req);
  }

  const CiBatchLayout &layout = attn.GetLayout();
  const int B = layout.num_instances;
  const int D = cfg.head_dim;
  const int T = cfg.num_tokens;
  std::mt19937_64 gen(0xB512);
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
  keys.boot = &boot.ui->GetEvkMap();
  keys.swtch = &swtch.ui->GetEvkMap();
  keys.lifted = &lifted.ui->GetEvkMap();
  keys.ring_switch = &swtch.ui->GetRingSwitchKey(attn.GetChain().rank);
  keys.inverse_ring_switch =
      &swtch.ui->GetInverseRingSwitchKey(attn.GetChain().rank);

  // Scores consumes Q: one encryption, two device copies.
  std::vector<Ciphertext<word>> q_a(D), q_b(D);
  for (int c = 0; c < D; c++) {
    boot.context->Copy(q_a[c], q_cts[c]);
    boot.context->Copy(q_b[c], q_cts[c]);
  }

  std::vector<Ciphertext<word>> s_serial, s_batch;
  auto t0 = Sync();
  cheddar::CiBatchAttention<word>::SetConvSerial(true);
  attn.Scores(s_serial, q_a, k_cts, keys);
  auto t1 = Sync();
  const auto ph0 = attn.GetPhaseSeconds();
  cheddar::CiBatchAttention<word>::SetConvSerial(false);
  attn.Scores(s_batch, q_b, k_cts, keys);
  auto t2 = Sync();
  const auto ph1 = attn.GetPhaseSeconds();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(s_serial.size(), s_batch.size());
  std::cout << "  one head's scores, serial " << std::fixed
            << std::setprecision(2) << Ms(t0, t1) / 1000.0 << " s -> batched "
            << Ms(t1, t2) / 1000.0 << " s" << std::endl;
  std::cout << "  batched call's phases (device s): descend "
            << ph1.descend - ph0.descend << " (pre "
            << ph1.desc_pre - ph0.desc_pre << ", convert "
            << ph1.desc_conv - ph0.desc_conv << ", switch "
            << ph1.desc_switch - ph0.desc_switch << ", lift "
            << ph1.desc_lift - ph0.desc_lift << "), multiply "
            << ph1.multiply - ph0.multiply << ", lift.descend "
            << ph1.lift_descend - ph0.lift_descend << ", return "
            << ph1.ret - ph0.ret << std::endl;

  size_t diff = 0, total = 0;
  for (size_t i = 0; i < s_serial.size(); i++) {
    EXPECT_EQ(s_serial[i].GetScale(), s_batch[i].GetScale());
    EXPECT_EQ(s_serial[i].GetNumSlots(), s_batch[i].GetNumSlots());
    for (int part = 0; part < 2; part++) {
      const auto &da = part ? s_serial[i].ax_ : s_serial[i].bx_;
      const auto &db = part ? s_batch[i].ax_ : s_batch[i].bx_;
      ASSERT_EQ(da.size(), db.size());
      cheddar::HostVector<word> ha, hb;
      cheddar::CopyDeviceToHost(ha, da);
      cheddar::CopyDeviceToHost(hb, db);
      total += ha.size();
      for (size_t w = 0; w < ha.size(); w++) diff += (ha[w] != hb[w]);
    }
  }
  std::cout << "  batched vs serial converters: " << diff << " of " << total
            << " words differ" << std::endl;
  EXPECT_EQ(diff, 0u);
}

// ---------------------------------------------------------------------------
// 5a''. The fused score boot (idea [4]) against the serial return + full
//       Boot, slot by slot on random data: both routes produce "the booted
//       scores as a Boot left them" (carried * m at the landing level), so
//       after dividing each by its own carried they must agree to boot
//       precision. A global constant off means a scale/ratio convention; a
//       scattered error means the tower basis does not read the batch
//       chain's output. Needs no weights.
TEST(CiBatch, TheFusedScoreBootMatchesTheSerialBoot) {
  Ring boot(Param());
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  Ring small("ci12_35_boot.json");
  Ring lifted("ringdegree13_35_boot.json",
              cheddar::CiLiftHandler<word>::LiftSecret(
                  small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  std::unique_ptr<Ring> tower;
  {
    const char *prev = std::getenv("CHEDDAR_MODULE_SPARSE_SECRET");
    const std::string saved = prev ? prev : "";
    setenv("CHEDDAR_MODULE_SPARSE_SECRET", "4096:128,16", 1);
    tower = std::make_unique<Ring>(TowerParam(), boot.ui->GetSecretCoeffs(),
                                   /*slack=*/0);
    if (prev) {
      setenv("CHEDDAR_MODULE_SPARSE_SECRET", saved.c_str(), 1);
    } else {
      unsetenv("CHEDDAR_MODULE_SPARSE_SECRET");
    }
  }
  auto tctx = std::dynamic_pointer_cast<BootContext<word>>(tower->context);
  ASSERT_NE(tctx, nullptr);
  tctx->PrepareEvalMod();

  // ONE attention object for both routes (two would double the converters'
  // ~14 GiB); the fused return is flipped at runtime.
  cheddar::CiBatchAttention<word>::Config cfg_s;
  cfg_s.fused_scores = true;
  cfg_s.verbose = true;
  cheddar::CiBatchAttention<word> attn_s(bctx, swtch.context, small.context,
                                         lifted.context, cfg_s, tctx);
  cheddar::CiBatchAttention<word> &attn_f = attn_s;

  const int num_slots = attn_s.GetLayout().num_slots;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  const int chain_level = attn_s.GetChainLevel();
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : attn_s.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  {
    cheddar::EvkRequest req;
    attn_s.AddSwitchRotations(req);
    attn_f.AddSwitchRotations(req);
    swtch.ui->PrepareRotationKey(req);
  }
  {
    cheddar::EvkRequest req;
    attn_s.AddBootRotations(req);
    bctx->AddRequiredRotations(req, num_slots, false);
    boot.ui->PrepareRotationKey(req);
  }
  {
    cheddar::EvkRequest req;
    attn_f.AddTowerRotations(req);
    tower->ui->PrepareRotationKey(req);
  }

  const CiBatchLayout &layout = attn_s.GetLayout();
  const int B = layout.num_instances;
  const int D = cfg_s.head_dim;
  const int T = cfg_s.num_tokens;
  std::mt19937_64 gen(0xF05E);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  HostTensor q{B, T, D, {}}, k{B, T, D, {}};
  q.v.resize(static_cast<size_t>(B) * T * D);
  k.v.resize(q.v.size());
  for (auto &v : q.v) v = dist(gen);
  for (auto &v : k.v) v = dist(gen);
  std::vector<Ciphertext<word>> q_cts, k_cts;
  EncryptChannels(boot, layout, q, cfg_s.rope_level, q_cts);
  EncryptChannels(boot, layout, k, cfg_s.rope_level, k_cts);

  cheddar::CiBatchAttention<word>::Keys keys;
  keys.boot = &boot.ui->GetEvkMap();
  keys.swtch = &swtch.ui->GetEvkMap();
  keys.lifted = &lifted.ui->GetEvkMap();
  keys.tower = &tower->ui->GetEvkMap();
  keys.ring_switch = &swtch.ui->GetRingSwitchKey(attn_s.GetChain().rank);
  keys.inverse_ring_switch =
      &swtch.ui->GetInverseRingSwitchKey(attn_s.GetChain().rank);

  std::vector<Ciphertext<word>> q_a(D), q_b(D);
  for (int c = 0; c < D; c++) {
    boot.context->Copy(q_a[c], q_cts[c]);
    boot.context->Copy(q_b[c], q_cts[c]);
  }

  // The serial route: converter return, then the full Boot (groups of 16;
  // the whole T at once OOMs beside the tower tables).
  attn_s.SetFusedScores(false);
  std::vector<Ciphertext<word>> s_ref;
  attn_s.Scores(s_ref, q_a, k_cts, keys);
  const int ls = boot.param->NPToLevel(s_ref[0].GetNP());
  const double carried_ref = s_ref[0].GetScale() / boot.param->GetScale(ls);
  std::vector<Ciphertext<word>> booted_ref(T);
  for (int l0 = 0; l0 < T; l0 += 16) {
    const int g = std::min(T - l0, 16);
    std::vector<const Ciphertext<word> *> in(g);
    for (int j = 0; j < g; j++) in[j] = &s_ref[l0 + j];
    std::vector<Ciphertext<word>> out;
    bctx->BootBatch(out, in, boot.ui->GetEvkMap());
    for (int j = 0; j < g; j++) {
      booted_ref[l0 + j] = std::move(out[j]);
      s_ref[l0 + j] = Ciphertext<word>();
    }
  }

  // The fused route: the SinC element, HalfBootTower + prefix. The carried
  // reading is the serial one -- recorded over canonical at the
  // ciphertext's level (the descent to 0 preserves the offset).
  attn_f.SetFusedScores(true);
  std::vector<Ciphertext<word>> s_sinc;
  attn_f.Scores(s_sinc, q_b, k_cts, keys);
  const int lf = boot.param->NPToLevel(s_sinc[0].GetNP());
  const double carried_f = s_sinc[0].GetScale() / boot.param->GetScale(lf);
  std::vector<Ciphertext<word>> booted_f;
  attn_f.BootScoresFused(booted_f, s_sinc, keys, 16);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "  carried: serial " << carried_ref << ", fused " << carried_f
            << " (ratio " << carried_f / carried_ref << ")" << std::endl;
  std::cout << "  landings: serial "
            << boot.param->NPToLevel(booted_ref[0].GetNP()) << " scale "
            << booted_ref[0].GetScale() << ", fused "
            << boot.param->NPToLevel(booted_f[0].GetNP()) << " scale "
            << booted_f[0].GetScale() << std::endl;

  // Slot-by-slot: each route's message over its own carried.
  double num = 0.0, den = 0.0, ratio_sum = 0.0;
  int ratio_n = 0;
  for (int l = 0; l < T; l += 17) {  // a sample of key tokens
    std::vector<Complex> mr, mf;
    {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, booted_ref[l]);
      boot.context->encoder_.Decode(mr, pt);
      boot.ui->Decrypt(pt, booted_f[l]);
      boot.context->encoder_.Decode(mf, pt);
    }
    for (size_t s = 0; s < mr.size(); s++) {
      const double a = mr[s].real() / carried_ref;
      const double b = mf[s].real() / carried_f;
      num += (a - b) * (a - b);
      den += a * a;
      if (std::abs(a) > 1e-3 && ratio_n < 200000) {
        ratio_sum += b / a;
        ratio_n++;
      }
    }
  }
  const double rel = std::sqrt(num / std::max(den, 1e-300));
  std::cout << "  fused vs serial booted scores: rms rel 2^-" << std::fixed
            << std::setprecision(2) << Bits(rel) << ", mean ratio "
            << std::setprecision(6) << (ratio_n ? ratio_sum / ratio_n : 0.0)
            << " over " << ratio_n << " large slots" << std::endl;
  EXPECT_LT(rel, std::ldexp(1.0, -8));
}

// ---------------------------------------------------------------------------
// 5b. The softmax alone, on the real head-0 scores, through the exact seam
//     the layer uses: the scores encrypted at the chain's output level with
//     the chain's non-canonical scale (carried), booted, then SoftMax --
//     against the true causal softmax. The host mirror
//     (reference: softmax_mirror.py) puts the algorithm at 2^-15.5 on these
//     scores, so anything worse here is the crypto seam, not the fit.
// ---------------------------------------------------------------------------
TEST(CiBatch, TheSoftMaxOfOneHeadMatchesTheHost) {
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
  constexpr int kH = 4096, kKv = 1024, kHeads = 32;
  const int T = kTokens, head = 0;
  const double eps = 1e-5;

  std::vector<float> x0, wq, wk, wv, wo, gain;
  ASSERT_TRUE(ReadF32(ld + "/../input_nosink.f32",
                      static_cast<size_t>(T) * kH, x0));
  ASSERT_TRUE(ReadF32(ld + "/wq.f32", static_cast<size_t>(kH) * kH, wq));
  ASSERT_TRUE(ReadF32(ld + "/wk.f32", static_cast<size_t>(kH) * kKv, wk));
  ASSERT_TRUE(ReadF32(ld + "/wv.f32", static_cast<size_t>(kH) * kKv, wv));
  ASSERT_TRUE(ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, wo));
  ASSERT_TRUE(ReadF32(ld + "/attn_norm.f32", kH, gain));

  std::ifstream cf(rd + "/calib.json");
  ASSERT_TRUE(cf.good()) << rd << "/calib.json";
  nlohmann::json cj = nlohmann::json::parse(cf)["layers"][0];
  const double s_min = cj["s_raw_min"], s_max = cj["s_raw_max"];
  const double span_raw = s_max - s_min;
  const double m_eff = cj["span"];
  const double score_ride = 0.35;
  const double cqk = score_ride / std::max(std::abs(s_min), std::abs(s_max));
  std::vector<double> sink(T, 1.0);
  if (cj.contains("attn_sink")) {
    const auto v = cj["attn_sink"].get<std::vector<double>>();
    for (size_t i = 0; i < v.size() && i < sink.size(); i++) sink[i] = v[i];
  }

  cheddar::CiBatchAttention<word>::Config acfg;
  acfg.verbose = true;
  // [3]: walk from a lower score landing (the aux boot split's freed top).
  acfg.score_top = EnvInt("CHEDDAR_CI_BATCH_SCORE_TOP", 0);

  // Host: the real head-0 raw scores -- norm, Q/K, RoPE, the contraction.
  const int D = acfg.head_dim;
  std::vector<double> y(static_cast<size_t>(T) * kH);
  for (int t = 0; t < T; t++) {
    double ms = 0.0;
    for (int c = 0; c < kH; c++) {
      const double v = sink[t] * x0[static_cast<size_t>(t) * kH + c];
      ms += v * v;
    }
    const double r = 1.0 / std::sqrt(ms / kH + eps);
    for (int c = 0; c < kH; c++) {
      y[static_cast<size_t>(t) * kH + c] =
          sink[t] * x0[static_cast<size_t>(t) * kH + c] * r * gain[c];
    }
  }
  std::vector<double> q(static_cast<size_t>(T) * D), k(q.size());
  const int kv_col = (head / 4) * D;
  cheddar::ParallelFor(T, [&](int begin, int end) {
    for (int t = begin; t < end; t++) {
      for (int d = 0; d < D; d++) {
        double aq = 0.0, ak = 0.0;
        for (int c = 0; c < kH; c++) {
          const double yc = y[static_cast<size_t>(t) * kH + c];
          aq += yc * wq[static_cast<size_t>(c) * kH + head * D + d];
          ak += yc * wk[static_cast<size_t>(c) * kKv + kv_col + d];
        }
        q[static_cast<size_t>(t) * D + d] = aq;
        k[static_cast<size_t>(t) * D + d] = ak;
      }
    }
  });
  const int half = D / 2;
  for (int t = 0; t < T; t++) {
    for (int c = 0; c < half; c++) {
      const double theta = std::pow(acfg.rope_base, -2.0 * c / D);
      const double a = t * theta;
      for (double *m : {q.data(), k.data()}) {
        const double lo = m[static_cast<size_t>(t) * D + c];
        const double hi = m[static_cast<size_t>(t) * D + c + half];
        m[static_cast<size_t>(t) * D + c] =
            lo * std::cos(a) - hi * std::sin(a);
        m[static_cast<size_t>(t) * D + c + half] =
            hi * std::cos(a) + lo * std::sin(a);
      }
    }
  }
  std::vector<double> s_raw(static_cast<size_t>(T) * T);
  for (int t = 0; t < T; t++) {
    for (int l = 0; l < T; l++) {
      double acc = 0.0;
      for (int d = 0; d < D; d++) {
        acc += q[static_cast<size_t>(t) * D + d] *
               k[static_cast<size_t>(l) * D + d];
      }
      s_raw[static_cast<size_t>(t) * T + l] = acc;
    }
  }

  // The true causal softmax the walk is to reproduce.
  std::vector<double> row_shift(T);
  for (int t = 0; t < T; t++) {
    row_shift[t] = cj["row_shift_raw"][head][t].get<double>();
  }
  std::vector<double> p_true(static_cast<size_t>(T) * T, 0.0);
  for (int t = 0; t < T; t++) {
    double sum = 0.0;
    for (int l = 0; l <= t; l++) {
      const double e = std::exp(
          m_eff * (s_raw[static_cast<size_t>(t) * T + l] - row_shift[t]) /
          span_raw);
      p_true[static_cast<size_t>(t) * T + l] = e;
      sum += e;
    }
    for (int l = 0; l <= t; l++) p_true[static_cast<size_t>(t) * T + l] /= sum;
  }

  Ring boot(Param());
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  Ring small("ci12_35_boot.json");
  Ring lifted("ringdegree13_35_boot.json",
              cheddar::CiLiftHandler<word>::LiftSecret(
                  small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  cheddar::CiBatchLayer<word>::Config lcfg;
  lcfg.num_tokens = T;
  lcfg.model = 32;
  lcfg.hidden = 64;
  lcfg.rows_per_tile = 32;
  cheddar::CiBatchLayer<word> layer(bctx, lcfg);
  cheddar::CiBatchAttention<word> attn(bctx, swtch.context, small.context,
                                       lifted.context, acfg);
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(attn.GetLayout().num_slots);
  {
    cheddar::EvkRequest req;
    layer.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
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
  {
    cheddar::EvkRequest req;
    attn.AddBootRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  cheddar::CiBatchAttention<word>::Keys akeys;
  akeys.boot = &boot.ui->GetEvkMap();
  akeys.swtch = &swtch.ui->GetEvkMap();
  akeys.lifted = &lifted.ui->GetEvkMap();
  akeys.ring_switch = &swtch.ui->GetRingSwitchKey(attn.GetChain().rank);
  akeys.inverse_ring_switch =
      &swtch.ui->GetInverseRingSwitchKey(attn.GetChain().rank);
  // [3]: the aux accumulator's own bootstrap on the channel ring, before
  // PrepareSoftMax (the inverse square root compiles at its landing).
  std::unique_ptr<Ring> chan = WireChannelRing(boot, layer);
  if (chan && EnvInt("CHEDDAR_CI_BATCH_AUX_BOOT", 0) != 0) {
    attn.SetAuxBoot(
        std::dynamic_pointer_cast<BootContext<word>>(chan->context),
        &chan->ui->GetEvkMap());
    std::cout << "  aux boot on the channel ring" << std::endl;
  }
  {
    typename cheddar::CiBatchAttention<word>::SoftMaxCalibration sc;
    sc.m_eff = m_eff;
    sc.span = cqk * span_raw;
    sc.shift = cqk * s_max;
    sc.causal = true;
    // The aux boot split fits the walk only at degree 7 (the layer's own):
    // from the landing-9 aux ring, deg 15's four levels put P below the
    // forward level.
    sc.inv_degree = EnvInt("CHEDDAR_CI_BATCH_INV_DEGREE", 15);
    sc.row_shift.assign(kHeads, std::vector<double>(T, 0.0));
    sc.row_norm.assign(kHeads, std::vector<double>(T, 1.0));
    for (int h = 0; h < kHeads; h++) {
      for (int t = 0; t < T; t++) {
        sc.row_shift[h][t] = cqk * cj["row_shift_raw"][h][t].get<double>();
        sc.row_norm[h][t] = cj["row_norm"][h][t].get<double>();
      }
    }
    attn.PrepareSoftMax(sc);
  }

  const CiBatchLayout &layout = attn.GetLayout();
  const int B = layout.num_instances;
  std::vector<int> bs = {0, B - 1}, all_l(T);
  for (int l = 0; l < T; l++) all_l[l] = l;

  // The chain leaves the scores at a low level with a non-canonical scale
  // (`carried`, 1.8 on the real chain); the layer boots them and hands
  // `carried` to the softmax. Both ends of that seam:
  for (const double carried : {1.0, 1.8}) {
    HostTensor s{B, T, T, {}};
    s.v.resize(static_cast<size_t>(B) * T * T);
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int l = 0; l < T; l++) {
          s.At(b, t, l) =
              carried * cqk * s_raw[static_cast<size_t>(t) * T + l];
        }
      }
    }
    std::vector<Ciphertext<word>> cts;
    EncryptChannels(boot, layout, s, 1, cts);
    // The chain's state: raw carries m * carried at the canonical scale =
    // m at scale carried * canonical.
    for (auto &ct : cts) {
      ct.SetScale(carried * boot.param->GetScale(1));
    }
    auto t0 = Sync();
    std::vector<Ciphertext<word>> booted(T);
    for (int l = 0; l < T; l++) {
      bctx->Boot(booted[l], cts[l], boot.ui->GetEvkMap());
    }
    cts.clear();
    std::vector<Ciphertext<word>> P;
    attn.SoftMax(P, booted, head, carried, boot.ui->GetEvkMap());
    auto t1 = Sync();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    HostTensor got{B, T, T, {}};
    got.v.assign(static_cast<size_t>(B) * T * T, 0.0);
    DecryptChannels(boot, layout, P, all_l, got);
    HostTensor want{B, T, T, {}};
    want.v.assign(got.v.size(), 0.0);
    for (int b : bs) {
      for (int t = 0; t < T; t++) {
        for (int l = 0; l < T; l++) {
          want.At(b, t, l) = p_true[static_cast<size_t>(t) * T + l];
        }
      }
    }
    const Err e = Compare(got, want, bs, all_l);
    std::cout << "  softmax (carried " << carried << "): rms rel 2^-"
              << std::fixed << std::setprecision(2) << Bits(e.rms_rel)
              << " (max abs " << std::scientific << e.max_abs << ", ref rms "
              << e.rms_ref << ") in " << std::fixed << Ms(t0, t1) / 1000.0
              << " s (boot + walk)" << std::endl;
    EXPECT_LT(e.rms_rel, std::ldexp(1.0, -8)) << "carried " << carried;
  }

  // --- Values on the same head: the TRUE P against host P V, then the O
  //     slice with the measured ratio -- the two stages the half test runs
  //     that nothing had verified in isolation.
  const int D2 = acfg.head_dim;
  std::vector<double> v_host(static_cast<size_t>(T) * D2);
  cheddar::ParallelFor(T, [&](int begin, int end) {
    for (int l = begin; l < end; l++) {
      for (int d = 0; d < D2; d++) {
        double acc = 0.0;
        for (int c = 0; c < kH; c++) {
          acc += y[static_cast<size_t>(l) * kH + c] *
                 wv[static_cast<size_t>(c) * kKv + kv_col + d];
        }
        v_host[static_cast<size_t>(l) * D2 + d] = acc;
      }
    }
  });
  std::vector<Ciphertext<word>> p_cts, v_cts;
  {
    HostTensor pt{B, T, T, {}};
    pt.v.resize(static_cast<size_t>(B) * T * T);
    HostTensor vt{B, T, D2, {}};
    vt.v.resize(static_cast<size_t>(B) * T * D2);
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int l = 0; l < T; l++) {
          pt.At(b, t, l) = p_true[static_cast<size_t>(t) * T + l];
        }
        for (int d = 0; d < D2; d++) {
          vt.At(b, t, d) = v_host[static_cast<size_t>(t) * D2 + d];
        }
      }
    }
    EncryptChannels(boot, layout, pt, acfg.forward_level, p_cts);
    EncryptChannels(boot, layout, vt, acfg.forward_level + 1, v_cts);
  }
  auto tv0 = Sync();
  std::vector<Ciphertext<word>> pv;
  attn.Values(pv, p_cts, v_cts, akeys);
  auto tv1 = Sync();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(static_cast<int>(pv.size()), D2);
  {
    std::vector<int> all_d(D2);
    for (int d = 0; d < D2; d++) all_d[d] = d;
    HostTensor got{B, T, D2, {}};
    got.v.assign(static_cast<size_t>(B) * T * D2, 0.0);
    DecryptChannels(boot, layout, pv, all_d, got);
    HostTensor want{B, T, D2, {}};
    want.v.assign(got.v.size(), 0.0);
    for (int b : bs) {
      for (int t = 0; t < T; t++) {
        for (int d = 0; d < D2; d++) {
          double acc = 0.0;
          for (int l = 0; l <= t; l++) {
            acc += p_true[static_cast<size_t>(t) * T + l] *
                   v_host[static_cast<size_t>(l) * D2 + d];
          }
          want.At(b, t, d) = acc;
        }
      }
    }
    const Err ev = Compare(got, want, bs, all_d);
    std::cout << "  values (P V): rms rel 2^-" << std::fixed
              << std::setprecision(2) << Bits(ev.rms_rel) << " (max abs "
              << std::scientific << ev.max_abs << ", ref rms " << ev.rms_ref
              << ") in " << std::fixed << Ms(tv0, tv1) / 1000.0 << " s"
              << std::endl;
    EXPECT_LT(ev.rms_rel, std::ldexp(1.0, -8)) << "the values product";

    // The O slice as the layer runs it: level and ratio read off the
    // attention output, the weight told both.
    const int o_level = boot.param->NPToLevel(pv[0].GetNP());
    const double o_ratio =
        pv[0].GetScale() / boot.param->GetScale(o_level);
    std::cout << "  attention out at level " << o_level << ", ratio "
              << o_ratio << std::endl;
    cheddar::CiBatchProjection<word> &proj = layer.GetProjection();
    std::vector<float> wo_slice(static_cast<size_t>(D2) * kH);
    for (int d = 0; d < D2; d++) {
      for (int c = 0; c < kH; c++) {
        wo_slice[static_cast<size_t>(d) * kH + c] =
            wo[(static_cast<size_t>(head) * D2 + d) * kH + c];
      }
    }
    cheddar::DeviceVector<float> wo_dev;
    ToDevice(wo_dev, wo_slice);
    proj.Prepare("o.head0", wo_dev.data(), D2, kH, o_level, 1.0, o_ratio);
    std::vector<Ciphertext<word>> o_cts;
    proj.Project(o_cts, pv, "o.head0");
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const int keep = 64;
    std::vector<int> ks(keep);
    for (int c = 0; c < keep; c++) ks[c] = c * (kH / keep);
    HostTensor o_got{B, T, kH, {}};
    o_got.v.assign(static_cast<size_t>(B) * T * kH, 0.0);
    DecryptChannels(boot, layout, o_cts, ks, o_got);
    HostTensor o_want{B, T, kH, {}};
    o_want.v.assign(o_got.v.size(), 0.0);
    for (int b : bs) {
      for (int t = 0; t < T; t++) {
        for (int c : ks) {
          double acc = 0.0;
          for (int d = 0; d < D2; d++) {
            acc += want.At(b, t, d) *
                   wo_slice[static_cast<size_t>(d) * kH + c];
          }
          o_want.At(b, t, c) = acc;
        }
      }
    }
    const Err eo = Compare(o_got, o_want, bs, ks);
    std::cout << "  O slice (head 0): rms rel 2^-" << std::fixed
              << Bits(eo.rms_rel) << " (max abs " << std::scientific
              << eo.max_abs << ", ref rms " << eo.rms_ref << ")" << std::fixed
              << std::endl;
    EXPECT_LT(eo.rms_rel, std::ldexp(1.0, -8)) << "the O projection slice";
  }
#endif
}

// ---------------------------------------------------------------------------
// 6. The attention half of layer 0 on the real weights, B instances at once:
//    norm, Q/K/V, per head scores -> Boot -> softmax -> P V, O, residual --
//    against the exporter's clear attention output. RMSNorm is scale
//    invariant per token, so every instance (the recorded prompt at its own
//    factor) has the SAME attention output, and the reference is av_L00.f64
//    times Wo for all of them, plus each instance's own input.
// ---------------------------------------------------------------------------
TEST(CiBatch, TheAttentionHalfRunsOnTheRealLayerZero) {
  const char *wdir_env = std::getenv("LLAMA3_ALL_DIR");
  const char *rdir_env = std::getenv("LLAMA3_REF_DIR");
  if (wdir_env == nullptr || rdir_env == nullptr) {
    GTEST_SKIP() << "LLAMA3_ALL_DIR and LLAMA3_REF_DIR must both be set";
  }
  const std::string ld = std::string(wdir_env) + "/L00";
  const std::string rd = rdir_env;
  constexpr int kH = 4096, kKv = 1024, kHeads = 32, kSinkTokens = 2;

  std::vector<float> x0, wq, wk, wv, wo, gain;
  std::vector<double> av;
  ASSERT_TRUE(ReadF32(ld + "/../input_nosink.f32",
                      static_cast<size_t>(kTokens) * kH, x0));
  ASSERT_TRUE(ReadF32(ld + "/wq.f32", static_cast<size_t>(kH) * kH, wq));
  ASSERT_TRUE(ReadF32(ld + "/wk.f32", static_cast<size_t>(kH) * kKv, wk));
  ASSERT_TRUE(ReadF32(ld + "/wv.f32", static_cast<size_t>(kH) * kKv, wv));
  ASSERT_TRUE(ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, wo));
  ASSERT_TRUE(ReadF32(ld + "/attn_norm.f32", kH, gain));
  ASSERT_TRUE(ReadF64(rd + "/av_L00.f64", static_cast<size_t>(kTokens) * kH, av));
  // o = av Wo, the clear O output; the reference layer input is x0.
  std::vector<double> o_ref(static_cast<size_t>(kTokens) * kH);
  cheddar::ParallelFor(kH, [&](int begin, int end) {
    for (int c = begin; c < end; c++) {
      for (int t = 0; t < kTokens; t++) {
        double acc = 0.0;
        for (int m = 0; m < kH; m++) {
          acc += av[static_cast<size_t>(t) * kH + m] *
                 static_cast<double>(wo[static_cast<size_t>(m) * kH + c]);
        }
        o_ref[static_cast<size_t>(t) * kH + c] = acc;
      }
    }
  });

  cheddar::CiBatchLayer<word>::Calibration cal;
  double in_absmax = 1.0, resid_absmax = 1.0;
  {
    std::ifstream f(rd + "/calib.json");
    ASSERT_TRUE(f.good()) << rd << "/calib.json";
    nlohmann::json cj = nlohmann::json::parse(f)["layers"][0];
    cal.attn_alpha = cj["attn_alpha"];
    cal.attn_norm_window = cj["attn_norm_window"];
    in_absmax = cj["in_absmax"];
    resid_absmax = cj["resid_absmax"];
    cal.attn_sink.assign(kTokens, 1.0);
    if (cj.contains("attn_sink")) {
      const auto v = cj["attn_sink"].get<std::vector<double>>();
      for (size_t i = 0; i < v.size() && i < cal.attn_sink.size(); i++) {
        cal.attn_sink[i] = v[i];
      }
    }
    const double s_min = cj["s_raw_min"], s_max = cj["s_raw_max"];
    cal.span_raw = s_max - s_min;
    cal.s_raw_max = s_max;
    cal.m_eff = cj["span"];
    const auto &rs = cj["row_shift_raw"];
    const auto &rn = cj["row_norm"];
    cal.row_shift_raw.assign(kHeads, std::vector<double>(kTokens, 0.0));
    cal.row_norm.assign(kHeads, std::vector<double>(kTokens, 1.0));
    for (int h = 0; h < kHeads; h++) {
      for (int t = 0; t < kTokens; t++) {
        cal.row_shift_raw[h][t] = rs[h][t].get<double>();
        cal.row_norm[h][t] = rn[h][t].get<double>();
      }
    }
    const double score_ride = [] {
      // (The 2026-09-02 wrap theory that briefly lowered this to 0.25 was
      // refuted -- the corruption was the eps-vs-embedding drift, Doing
      // 7.22 -- so the ride keeps its designed 0.35.)
      const char *e = std::getenv("CHEDDAR_CI_BATCH_SCORE_RIDE");
      return (e && e[0]) ? std::atof(e) : 0.35;
    }();
    const double cqk = score_ride / std::max(std::abs(s_min), std::abs(s_max));
    cal.cq = cal.ck = std::sqrt(cqk);
  }
  const double ride = [] {
    const char *e = std::getenv("CHEDDAR_CI_BATCH_RIDE");
    return (e && e[0]) ? std::atof(e) : 0.35;
  }();
  cal.stream_scale = ride / std::max(in_absmax, resid_absmax);
  WidenForInstanceFactors(cal.attn_alpha, cal.attn_norm_window, 0.98, 1.02);
  std::cout << "  calibration: attn_alpha " << cal.attn_alpha << " window "
            << cal.attn_norm_window << " span_raw " << cal.span_raw
            << " s_raw_max " << cal.s_raw_max << " m_eff " << cal.m_eff
            << " cq = ck = " << cal.cq << " stream_scale " << cal.stream_scale
            << std::endl;

  Ring boot(Param());
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  Ring small("ci12_35_boot.json");
  Ring lifted("ringdegree13_35_boot.json",
              cheddar::CiLiftHandler<word>::LiftSecret(
                  small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);

  // Idea [4]: the scores' return fused into their bootstrap on the TOWER
  // ring (K = 64), its SSE secret sampled tower-sparse around ITS Ring only
  // (the fused return's wrap-around is bounded there; CiModelTest does the
  // same for the A100 leg).
  const bool fused_scores = EnvInt("CHEDDAR_CI_BATCH_FUSED_SCORES", 0) != 0;
  std::unique_ptr<Ring> tower;
  std::shared_ptr<BootContext<word>> tctx;
  if (fused_scores) {
    const char *prev = std::getenv("CHEDDAR_MODULE_SPARSE_SECRET");
    const std::string saved = prev ? prev : "";
    setenv("CHEDDAR_MODULE_SPARSE_SECRET", "4096:128,16", 1);
    tower = std::make_unique<Ring>(TowerParam(),
                                   boot.ui->GetSecretCoeffs(), /*slack=*/0);
    if (prev) {
      setenv("CHEDDAR_MODULE_SPARSE_SECRET", saved.c_str(), 1);
    } else {
      unsetenv("CHEDDAR_MODULE_SPARSE_SECRET");
    }
    tctx = std::dynamic_pointer_cast<BootContext<word>>(tower->context);
    ASSERT_NE(tctx, nullptr);
    tctx->PrepareEvalMod();
  }

  auto t0 = Sync();
  cheddar::CiBatchLayer<word>::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.model = kH;
  // Multiple of the head dim (128). At 512 the first group's K/V/Q tiles
  // and the projection's persistent buffers stand ~5 GiB taller, and
  // attn11 died ~2 GiB short in head 0's softmax (driver 80.8 of 81.9
  // GiB): 256 is the margin, and the configuration cb_run.sh documents.
  cfg.rows_per_tile = EnvInt("CHEDDAR_CI_BATCH_TILE", 256);
  cfg.norm_apply_level = EnvInt("CHEDDAR_CI_BATCH_HOLD", 8);
  cfg.hold_channels = EnvInt("CHEDDAR_CI_BATCH_HOLD_CHANNELS", 1) != 0;
  cfg.verbose = true;
  cfg.lanes = 32;  // the chain: 32 lanes a group, rank 16
  cfg.rank = 16;
  cheddar::CiBatchLayer<word> layer(bctx, cfg);
  cheddar::CiBatchAttention<word>::Config acfg;
  // y at the attention hold - 1, the projections one below.
  acfg.rope_level = cfg.norm_apply_level_attn - 2;
  acfg.fused_scores = fused_scores;
  acfg.affine_in_prefix =
      fused_scores && EnvInt("CHEDDAR_CI_BATCH_AFFINE_PREFIX", 0) != 0;
  // [3]: the aux boot split -- the scores land at score_top (12 with the
  // land13c3e10 tower) and the accumulator boots on the channel ring.
  acfg.score_top = EnvInt("CHEDDAR_CI_BATCH_SCORE_TOP", 0);
  acfg.verbose = true;
  cheddar::CiBatchAttention<word> attn(bctx, swtch.context, small.context,
                                       lifted.context, acfg, tctx);
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(attn.GetLayout().num_slots);
  {
    cheddar::EvkRequest req;
    layer.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  std::unique_ptr<Ring> chan = WireChannelRing(boot, layer);
  if (chan && EnvInt("CHEDDAR_CI_BATCH_AUX_BOOT", 0) != 0) {
    attn.SetAuxBoot(
        std::dynamic_pointer_cast<BootContext<word>>(chan->context),
        &chan->ui->GetEvkMap());
    std::cout << "  aux boot on the channel ring" << std::endl;
  }
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
  {
    cheddar::EvkRequest req;
    attn.AddBootRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  if (fused_scores) {
    cheddar::EvkRequest req;
    attn.AddTowerRotations(req);
    tower->ui->PrepareRotationKey(req);
  }
  auto t1 = Sync();
  std::cout << "  setup (boot tables, three converters, keys): " << std::fixed
            << std::setprecision(1) << Ms(t0, t1) / 1000.0 << " s, "
            << FreeMiB() << " MiB free" << std::endl;

  const CiBatchLayout &layout = attn.GetLayout();
  const int B = layout.num_instances;
  // [0.98, 1.02], NOT [0.5, 1.5): layer 0's input is the EMBEDDING, whose
  // mean square (~4e-5) is only ~4x the norm's eps (1e-5), so the norm
  // leaves a residual factor 1/sqrt(1 + (1/f^2 - 1) eps/(ms + eps)) on the
  // stream -- +-13% on the scores at f = 1.5, which the exp squares
  // amplify by e^{~22 du} into a 15x swing of sq against the row_norm
  // estimate: head 31's invsqrt left its domain at u2 ~ 30 and one head's
  // 1e15 P poisoned every channel through the O sum (Doing 7.22). At
  // +-2% the swing stays inside the widened window. The softmax
  // calibration is a single-prompt fit; covering a real cross-prompt
  // population is a separate calibration item, not this test's job.
  auto factor = [&](int b) { return 0.98 + 0.04 * static_cast<double>(b) / B; };
  HostTensor x{B, kTokens, kH, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * kH);
  for (int b = 0; b < B; b++) {
    const double f = factor(b) * cal.stream_scale;
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kH; c++) {
        x.At(b, t, c) =
            f * static_cast<double>(x0[static_cast<size_t>(t) * kH + c]);
      }
    }
  }
  std::vector<Ciphertext<word>> stream;
  EncryptChannels(boot, layout, x, chan ? 1 : 0, stream);
  x.v.clear();
  x.v.shrink_to_fit();
  auto t2 = Sync();
  std::cout << "  encrypt the stream: " << Ms(t1, t2) / 1000.0 << " s, "
            << FreeMiB() << " MiB free" << std::endl;

  cheddar::DeviceVector<float> q_dev, k_dev, v_dev, o_dev;
  ToDevice(q_dev, wq);
  ToDevice(k_dev, wk);
  ToDevice(v_dev, wv);
  ToDevice(o_dev, wo);
  cheddar::CiBatchLayer<word>::AttnWeights w;
  w.q = q_dev.data();
  w.k = k_dev.data();
  w.v = v_dev.data();
  w.o = o_dev.data();
  w.attn_norm.assign(gain.begin(), gain.end());

  cheddar::CiBatchAttention<word>::Keys akeys;
  akeys.boot = &boot.ui->GetEvkMap();
  akeys.swtch = &swtch.ui->GetEvkMap();
  akeys.lifted = &lifted.ui->GetEvkMap();
  akeys.tower = fused_scores ? &tower->ui->GetEvkMap() : nullptr;
  akeys.ring_switch = &swtch.ui->GetRingSwitchKey(attn.GetChain().rank);
  akeys.inverse_ring_switch =
      &swtch.ui->GetInverseRingSwitchKey(attn.GetChain().rank);

  bctx->ResetBootCounts();
  // With CHEDDAR_CI_BATCH_MAX_KV < 8 the run is a bisection: only those kv
  // groups, head 0's intermediates captured, each stage compared against
  // the host at f = 1, and the full-output comparison skipped.
  const int max_kv = EnvInt("CHEDDAR_CI_BATCH_MAX_KV", 8);
  cheddar::CiBatchLayer<word>::AttnDebug dbg;
  dbg.head = EnvInt("CHEDDAR_CI_BATCH_DBG_HEAD", 0);
  // The debug path opens on a kv subset OR on an explicitly chosen head
  // (a head in the last group needs all 8 groups run to be reached).
  const bool debugging =
      max_kv < 8 || std::getenv("CHEDDAR_CI_BATCH_DBG_HEAD") != nullptr;
  auto t3 = Sync();
  std::vector<Ciphertext<word>> res;
  layer.Attention(res, stream, w, cal, attn, akeys, boot.ui->GetEvkMap(),
                  debugging ? &dbg : nullptr);
  auto t4 = Sync();
  const auto st = layer.GetStages();
  const auto counts = bctx->GetBootCounts();
  std::cout << "  ATTENTION, " << B << " instances: " << Ms(t3, t4) / 1000.0
            << " s wall = " << Ms(t3, t4) / B << " ms per instance; boots "
            << counts.full << " (tables " << layer.GetPrepareSeconds()
            << " s); stages boot " << st.boot << " norm " << st.norm
            << " q/k/v " << st.qkv << " scores " << st.scores << " softmax "
            << st.softmax << " values " << st.values << " o " << st.o
            << " s; " << FreeMiB() << " MiB free" << std::endl;
  ASSERT_EQ(static_cast<int>(res.size()), kH);

  if (debugging) {
    // The bisection path: the captured head's stages against the host, at
    // the f = 1 instance.
    const int D = 128, head = dbg.head, T = kTokens;
    std::vector<double> yn(static_cast<size_t>(T) * kH);
    for (int t = 0; t < T; t++) {
      double ms = 0.0;
      for (int c = 0; c < kH; c++) {
        const double v =
            cal.attn_sink[t] * x0[static_cast<size_t>(t) * kH + c];
        ms += v * v;
      }
      const double r = 1.0 / std::sqrt(ms / kH + 1e-5);
      for (int c = 0; c < kH; c++) {
        yn[static_cast<size_t>(t) * kH + c] =
            cal.attn_sink[t] * x0[static_cast<size_t>(t) * kH + c] * r;
      }
    }
    std::vector<double> qh(static_cast<size_t>(T) * D),
        kh(static_cast<size_t>(T) * D), vh(static_cast<size_t>(T) * D);
    const int kv_col = (head / 4) * D;
    cheddar::ParallelFor(T, [&](int begin, int end) {
      for (int t = begin; t < end; t++) {
        for (int d = 0; d < D; d++) {
          double aq = 0.0, ak = 0.0, av2 = 0.0;
          for (int c = 0; c < kH; c++) {
            const double yg = yn[static_cast<size_t>(t) * kH + c] *
                              static_cast<double>(gain[c]);
            aq += yg * wq[static_cast<size_t>(c) * kH + head * D + d];
            ak += yg * wk[static_cast<size_t>(c) * kKv + kv_col + d];
            av2 += yg * wv[static_cast<size_t>(c) * kKv + kv_col + d];
          }
          qh[static_cast<size_t>(t) * D + d] = cal.cq * aq;
          kh[static_cast<size_t>(t) * D + d] = cal.ck * ak;
          vh[static_cast<size_t>(t) * D + d] = av2;
        }
      }
    });
    std::vector<int> bref = {B / 2};
    auto check = [&](const char *tag, const std::vector<Ciphertext<word>> &cts,
                     const std::vector<double> &host, int width,
                     double divide) {
      if (cts.empty()) {
        std::cout << "  [dbg] " << tag << ": not captured" << std::endl;
        return;
      }
      std::vector<int> all_w(width);
      for (int i = 0; i < width; i++) all_w[i] = i;
      HostTensor g{B, T, width, {}};
      g.v.assign(static_cast<size_t>(B) * T * width, 0.0);
      DecryptChannels(boot, layout, cts, all_w, g);
      HostTensor want{B, T, width, {}};
      want.v.assign(g.v.size(), 0.0);
      for (int b : bref) {
        for (int t = 0; t < T; t++) {
          for (int i2 = 0; i2 < width; i2++) {
            want.At(b, t, i2) =
                host[static_cast<size_t>(t) * width + i2] / divide;
          }
        }
      }
      const Err e = Compare(g, want, bref, all_w);
      // The blow detector across EVERY instance: the largest decrypted
      // value anywhere, and which (instance, token, index) holds it --
      // the f = 1 comparison alone missed garbage confined to other
      // instances.
      double mx_all = 0.0;
      int mb = 0, mt = 0, mi = 0;
      for (int b2 = 0; b2 < B; b2++) {
        for (int t2 = 0; t2 < T; t2++) {
          for (int i2 = 0; i2 < width; i2++) {
            const double v = std::abs(g.At(b2, t2, i2));
            if (v > mx_all) {
              mx_all = v;
              mb = b2; mt = t2; mi = i2;
            }
          }
        }
      }
      std::cout << "  [dbg] " << tag << ": rms rel 2^-" << std::fixed
                << std::setprecision(2) << Bits(e.rms_rel) << " (max abs "
                << std::scientific << e.max_abs << ", ref rms " << e.rms_ref
                << "); ALL-instance max |value| " << mx_all << " at (b "
                << std::fixed << mb << ", t " << mt << ", i " << mi << ")"
                << std::endl;
    };
    check("q (pre-RoPE)", dbg.q, qh, D, 1.0);
    check("k (pre-RoPE)", dbg.k, kh, D, 1.0);
    check("v", dbg.v, vh, D, 1.0);
    // RoPE for the scores.
    const int half2 = D / 2;
    for (int t = 0; t < T; t++) {
      for (int c = 0; c < half2; c++) {
        const double theta = std::pow(500000.0, -2.0 * c / D);
        const double a = t * theta;
        for (double *m : {qh.data(), kh.data()}) {
          const double lo = m[static_cast<size_t>(t) * D + c];
          const double hi = m[static_cast<size_t>(t) * D + c + half2];
          m[static_cast<size_t>(t) * D + c] =
              lo * std::cos(a) - hi * std::sin(a);
          m[static_cast<size_t>(t) * D + c + half2] =
              hi * std::cos(a) + lo * std::sin(a);
        }
      }
    }
    std::vector<double> sh(static_cast<size_t>(T) * T);
    for (int t = 0; t < T; t++) {
      for (int l = 0; l < T; l++) {
        double acc = 0.0;
        for (int d = 0; d < D; d++) {
          acc += qh[static_cast<size_t>(t) * D + d] *
                 kh[static_cast<size_t>(l) * D + d];
        }
        sh[static_cast<size_t>(t) * T + l] = acc;
      }
    }
    check("scores", dbg.scores, sh, T, 1.0);
    // The f-profile of one named slot: CHEDDAR_CI_BATCH_DBG_SLOT="t,l"
    // prints the decoded score at (t, l) for a sweep of instances, against
    // the f-independent host value -- the corruption's dependence on the
    // instance factor (zero? linear? thresholded?) names its mechanism.
    if (const char *sl = std::getenv("CHEDDAR_CI_BATCH_DBG_SLOT")) {
      int st = 0, sll = 0;
      if (std::sscanf(sl, "%d,%d", &st, &sll) == 2 && !dbg.scores.empty()) {
        std::vector<int> one = {sll};
        HostTensor g{B, T, T, {}};
        g.v.assign(static_cast<size_t>(B) * T * T, 0.0);
        DecryptChannels(boot, layout, dbg.scores, one, g);
        std::cout << "  [dbg] slot (t " << st << ", l " << sll
                  << "): host " << std::scientific
                  << sh[static_cast<size_t>(st) * T + sll] << "; decoded by instance:"
                  << std::endl;
        for (int b2 = 0; b2 < B; b2 += 32) {
          std::cout << "    b " << std::setw(3) << b2 << " (f "
                    << std::fixed << std::setprecision(3)
                    << (0.5 + static_cast<double>(b2) / B) << "): "
                    << std::scientific << g.At(b2, st, sll) << std::endl;
        }
        std::cout << "    b 511 (f 1.498): " << std::scientific
                  << g.At(511, st, sll) << std::fixed << std::endl;
      }
    }
    double carried = 1.0;
    if (!dbg.scores.empty()) {
      const int ls = boot.param->NPToLevel(dbg.scores[0].GetNP());
      carried = dbg.scores[0].GetScale() / boot.param->GetScale(ls);
      std::cout << "  [dbg] carried " << carried << std::endl;
    }
    // Booted: the boot leaves carried in the message.
    {
      std::vector<double> shc(sh.size());
      for (size_t i = 0; i < sh.size(); i++) shc[i] = sh[i] * carried;
      check("booted scores", dbg.booted, shc, T, 1.0);
    }
    // P: the true causal softmax (row shift and norm are the fit's).
    std::vector<double> pt2(static_cast<size_t>(T) * T, 0.0);
    for (int t = 0; t < T; t++) {
      const double rs = cal.row_shift_raw[head][t];
      double sum = 0.0;
      for (int l = 0; l <= t; l++) {
        const double sr = sh[static_cast<size_t>(t) * T + l] / (cal.cq * cal.ck);
        const double e2 = std::exp(cal.m_eff * (sr - rs) / cal.span_raw);
        pt2[static_cast<size_t>(t) * T + l] = e2;
        sum += e2;
      }
      for (int l = 0; l <= t; l++) pt2[static_cast<size_t>(t) * T + l] /= sum;
    }
    check("P", dbg.P, pt2, T, 1.0);
    std::vector<double> pv2(static_cast<size_t>(T) * D, 0.0);
    for (int t = 0; t < T; t++) {
      for (int d = 0; d < D; d++) {
        double acc = 0.0;
        for (int l = 0; l <= t; l++) {
          acc += pt2[static_cast<size_t>(t) * T + l] *
                 vh[static_cast<size_t>(l) * D + d];
        }
        pv2[static_cast<size_t>(t) * D + d] = acc;
      }
    }
    check("P V (head out)", dbg.out, pv2, D, 1.0);
    // The blow-up detector, no reference needed: the residual's magnitude
    // over a spread of channels. Sane is ~stream_scale x O(1); the failure
    // under hunt is ~1e5.
    {
      const int probe = 64;
      std::vector<int> ks(probe);
      for (int c = 0; c < probe; c++) ks[c] = c * (kH / probe);
      HostTensor g{B, T, kH, {}};
      g.v.assign(static_cast<size_t>(B) * T * kH, 0.0);
      DecryptChannels(boot, layout, res, ks, g);
      double mx = 0.0, s2 = 0.0;
      size_t n = 0;
      for (int b : bref) {
        for (int t = 0; t < T; t++) {
          for (int c : ks) {
            const double v = g.At(b, t, c);
            mx = std::max(mx, std::abs(v));
            s2 += v * v;
            n++;
          }
        }
      }
      std::cout << "  [dbg] residual magnitude over " << probe
                << " channels: rms " << std::scientific
                << std::sqrt(s2 / static_cast<double>(n)) << ", max " << mx
                << std::fixed << std::endl;
    }
    std::cout << "  [dbg] bisection over " << max_kv
              << " kv group(s), head " << head
              << "; the full-output comparison is skipped" << std::endl;
    return;
  }

  std::vector<int> bs = {0, B / 4, B / 2, B - 1};
  std::vector<int> all_c(kH);
  for (int c = 0; c < kH; c++) all_c[c] = c;
  HostTensor got{B, kTokens, kH, {}};
  got.v.assign(static_cast<size_t>(B) * kTokens * kH, 0.0);
  DecryptChannels(boot, layout, res, all_c, got);
  double worst = 0.0;
  for (int b : bs) {
    std::vector<double> want(static_cast<size_t>(kTokens) * kH),
        o_got(want.size()), out_got(want.size());
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kH; c++) {
        const size_t i = static_cast<size_t>(t) * kH + c;
        const double xb = factor(b) * x0[i];
        want[i] = xb + o_ref[i];
        out_got[i] = got.At(b, t, c) / cal.stream_scale;
        o_got[i] = out_got[i] - xb;
      }
    }
    const double e_layer = RmsRel(out_got, want, kTokens, kH, kSinkTokens);
    const double e_o = RmsRel(o_got, o_ref, kTokens, kH, kSinkTokens);
    std::cout << "  instance " << std::setw(3) << b << " (x" << std::fixed
              << std::setprecision(3) << factor(b)
              << "): post-attention residual 2^-" << std::setprecision(2)
              << Bits(e_layer) << ", the attention output (after O) alone 2^-"
              << Bits(e_o) << " over user tokens" << std::endl;
    worst = std::max(worst, e_layer);
  }
  EXPECT_LT(worst, std::ldexp(1.0, -6));
}

// ---------------------------------------------------------------------------
// 7. Whole layers, chained, on the real weights: layer L reads L-1's
//    ENCRYPTED output. `CHEDDAR_CI_BATCH_LAYERS` layers from layer 0; the
//    instance at factor 1 (B/2) is checked against the exporter's h_L{L}.f64
//    on the user tokens after every layer -- the other instances carry the
//    same prompt at another factor and have no float64 twin past the first
//    attention. The stream carries one factor for the whole run, sized on
//    the run's largest residual.
// ---------------------------------------------------------------------------
TEST(CiBatch, TheLayerChainRunsOnTheRealWeights) {
  const char *wdir_env = std::getenv("LLAMA3_ALL_DIR");
  const char *rdir_env = std::getenv("LLAMA3_REF_DIR");
  if (wdir_env == nullptr || rdir_env == nullptr) {
    GTEST_SKIP() << "LLAMA3_ALL_DIR and LLAMA3_REF_DIR must both be set";
  }
  const std::string wd = wdir_env;
  const std::string rd = rdir_env;
  constexpr int kH = 4096, kKv = 1024, kI = 14336, kHeads = 32,
                kSinkTokens = 2;
  const int num_layers = EnvInt("CHEDDAR_CI_BATCH_LAYERS", 1);
  const double ride = [] {
    const char *e = std::getenv("CHEDDAR_CI_BATCH_RIDE");
    return (e && e[0]) ? std::atof(e) : 0.35;
  }();
  const double score_ride = [] {
    const char *e = std::getenv("CHEDDAR_CI_BATCH_SCORE_RIDE");
    return (e && e[0]) ? std::atof(e) : 0.35;
  }();

  nlohmann::json calib_all;
  {
    std::ifstream f(rd + "/calib.json");
    ASSERT_TRUE(f.good()) << rd << "/calib.json";
    calib_all = nlohmann::json::parse(f);
  }
  // One stream factor for the run: the largest residual any of its layers
  // reaches, sink rows included (as the single-prompt model test sizes it).
  double stream_absmax = calib_all["layers"][0]["in_absmax"].get<double>();
  for (int L = 0; L < num_layers; L++) {
    const auto &cj = calib_all["layers"][L];
    stream_absmax = std::max(stream_absmax, cj["out_absmax"].get<double>());
    stream_absmax = std::max(stream_absmax, cj["resid_absmax"].get<double>());
  }
  const double stream_scale = ride / stream_absmax;
  std::cout << "  " << num_layers << " layer(s); the stream reaches "
            << stream_absmax << ", so it carries " << stream_scale
            << " (ride " << ride << ")" << std::endl;

  std::vector<float> x0;
  ASSERT_TRUE(ReadF32(wd + "/input_nosink.f32",
                      static_cast<size_t>(kTokens) * kH, x0));

  Ring boot(Param());
  Ring swtch("ci_ringswitch16_35_boot.json", boot.ui->GetSecretCoeffs());
  Ring small("ci12_35_boot.json");
  Ring lifted("ringdegree13_35_boot.json",
              cheddar::CiLiftHandler<word>::LiftSecret(
                  small.ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const bool fused_scores = EnvInt("CHEDDAR_CI_BATCH_FUSED_SCORES", 0) != 0;
  std::unique_ptr<Ring> tower;
  std::shared_ptr<BootContext<word>> tctx;
  if (fused_scores) {
    const char *prev = std::getenv("CHEDDAR_MODULE_SPARSE_SECRET");
    const std::string saved = prev ? prev : "";
    setenv("CHEDDAR_MODULE_SPARSE_SECRET", "4096:128,16", 1);
    tower = std::make_unique<Ring>(TowerParam(),
                                   boot.ui->GetSecretCoeffs(), /*slack=*/0);
    if (prev) {
      setenv("CHEDDAR_MODULE_SPARSE_SECRET", saved.c_str(), 1);
    } else {
      unsetenv("CHEDDAR_MODULE_SPARSE_SECRET");
    }
    tctx = std::dynamic_pointer_cast<BootContext<word>>(tower->context);
    ASSERT_NE(tctx, nullptr);
    tctx->PrepareEvalMod();
  }
  auto t0 = Sync();
  cheddar::CiBatchLayer<word>::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.model = kH;
  cfg.hidden = kI;
  cfg.rows_per_tile = EnvInt("CHEDDAR_CI_BATCH_TILE", 512);
  cfg.norm_apply_level = EnvInt("CHEDDAR_CI_BATCH_HOLD", 8);
  cfg.hold_channels = EnvInt("CHEDDAR_CI_BATCH_HOLD_CHANNELS", 0) != 0;
  // The B200 residency switches; every default is the A100 configuration.
  cfg.hold_channels_ffn = EnvInt("CHEDDAR_CI_BATCH_HOLD_CHANNELS_FFN", 1) != 0;
  cfg.release_boot_tables = EnvInt("CHEDDAR_CI_BATCH_RELEASE_TABLES", 1) != 0;
  cfg.park_stream = EnvInt("CHEDDAR_CI_BATCH_PARK", 1) != 0;
  cfg.unstage_converters = EnvInt("CHEDDAR_CI_BATCH_UNSTAGE", 1) != 0;
  cfg.verbose = EnvInt("CHEDDAR_CI_BATCH_VERBOSE", 1) != 0;
  cfg.lanes = 32;  // the chain: 32 lanes a group, rank 16
  cfg.rank = 16;
  cheddar::CiBatchLayer<word> layer(bctx, cfg);
  cheddar::CiBatchAttention<word>::Config acfg;
  acfg.rope_level = cfg.norm_apply_level_attn - 2;
  acfg.fused_scores = fused_scores;
  acfg.affine_in_prefix =
      fused_scores && EnvInt("CHEDDAR_CI_BATCH_AFFINE_PREFIX", 0) != 0;
  acfg.score_top = EnvInt("CHEDDAR_CI_BATCH_SCORE_TOP", 0);
  acfg.verbose = cfg.verbose;
  cheddar::CiBatchAttention<word> attn(bctx, swtch.context, small.context,
                                       lifted.context, acfg, tctx);
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(attn.GetLayout().num_slots);
  {
    cheddar::EvkRequest req;
    layer.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
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
  {
    cheddar::EvkRequest req;
    attn.AddBootRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  if (fused_scores) {
    cheddar::EvkRequest req;
    attn.AddTowerRotations(req);
    tower->ui->PrepareRotationKey(req);
  }
  cheddar::CiBatchAttention<word>::Keys akeys;
  akeys.boot = &boot.ui->GetEvkMap();
  akeys.swtch = &swtch.ui->GetEvkMap();
  akeys.lifted = &lifted.ui->GetEvkMap();
  akeys.tower = fused_scores ? &tower->ui->GetEvkMap() : nullptr;
  akeys.ring_switch = &swtch.ui->GetRingSwitchKey(attn.GetChain().rank);
  akeys.inverse_ring_switch =
      &swtch.ui->GetInverseRingSwitchKey(attn.GetChain().rank);
  // [2]/[3]: the channel-boot ring and the aux boot split.
  std::unique_ptr<Ring> chan = WireChannelRing(boot, layer);
  if (chan && EnvInt("CHEDDAR_CI_BATCH_AUX_BOOT", 0) != 0) {
    attn.SetAuxBoot(
        std::dynamic_pointer_cast<BootContext<word>>(chan->context),
        &chan->ui->GetEvkMap());
    std::cout << "  aux boot on the channel ring" << std::endl;
  }
  auto t1 = Sync();
  std::cout << "  setup: " << std::fixed << std::setprecision(1)
            << Ms(t0, t1) / 1000.0 << " s, " << FreeMiB() << " MiB free"
            << std::endl;

  const CiBatchLayout &layout = attn.GetLayout();
  const int B = layout.num_instances;
  const int b_ref = B / 2;  // factor 1.0
  // Narrowed for the same eps-vs-embedding reason as the attention half
  // (Doing 7.22): the layer-0 norm leaves a residual factor on scaled
  // instances that a single-prompt softmax calibration cannot cover.
  auto factor = [&](int b) { return 0.98 + 0.04 * static_cast<double>(b) / B; };
  std::vector<Ciphertext<word>> stream;
  {
    HostTensor x{B, kTokens, kH, {}};
    x.v.resize(static_cast<size_t>(B) * kTokens * kH);
    for (int b = 0; b < B; b++) {
      const double f = factor(b) * stream_scale;
      for (int t = 0; t < kTokens; t++) {
        for (int c = 0; c < kH; c++) {
          x.At(b, t, c) =
              f * static_cast<double>(x0[static_cast<size_t>(t) * kH + c]);
        }
      }
    }
    EncryptChannels(boot, layout, x, chan ? 1 : 0, stream);
  }
  auto t2 = Sync();
  std::cout << "  encrypt the stream: " << Ms(t1, t2) / 1000.0 << " s"
            << std::endl;

  double total_s = 0.0;
  for (int L = 0; L < num_layers; L++) {
    const std::string ld = wd + "/L" + (L < 10 ? "0" : "") + std::to_string(L);
    const auto &cj = calib_all["layers"][L];
    std::vector<float> wq, wk, wv, wo, wg, wu, wd_, an, fn;
    ASSERT_TRUE(ReadF32(ld + "/wq.f32", static_cast<size_t>(kH) * kH, wq));
    ASSERT_TRUE(ReadF32(ld + "/wk.f32", static_cast<size_t>(kH) * kKv, wk));
    ASSERT_TRUE(ReadF32(ld + "/wv.f32", static_cast<size_t>(kH) * kKv, wv));
    ASSERT_TRUE(ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, wo));
    ASSERT_TRUE(ReadF32(ld + "/wgate.f32", static_cast<size_t>(kH) * kI, wg));
    ASSERT_TRUE(ReadF32(ld + "/wup.f32", static_cast<size_t>(kH) * kI, wu));
    ASSERT_TRUE(ReadF32(ld + "/wdown.f32", static_cast<size_t>(kI) * kH, wd_));
    ASSERT_TRUE(ReadF32(ld + "/attn_norm.f32", kH, an));
    ASSERT_TRUE(ReadF32(ld + "/ffn_norm.f32", kH, fn));
    cheddar::DeviceVector<float> q_dev, k_dev, v_dev, o_dev, g_dev, u_dev,
        d_dev;
    ToDevice(q_dev, wq);
    ToDevice(k_dev, wk);
    ToDevice(v_dev, wv);
    ToDevice(o_dev, wo);
    ToDevice(g_dev, wg);
    ToDevice(u_dev, wu);
    ToDevice(d_dev, wd_);
    cheddar::CiBatchLayer<word>::AttnWeights aw;
    aw.q = q_dev.data();
    aw.k = k_dev.data();
    aw.v = v_dev.data();
    aw.o = o_dev.data();
    aw.attn_norm.assign(an.begin(), an.end());
    cheddar::CiBatchLayer<word>::Weights fw;
    fw.gate = g_dev.data();
    fw.up = u_dev.data();
    fw.down = d_dev.data();
    fw.ffn_norm.assign(fn.begin(), fn.end());

    cheddar::CiBatchLayer<word>::Calibration cal;
    cal.alpha = cj["alpha"];
    cal.norm_window = cj["norm_window"];
    cal.silu_range = cj["silu_range"];
    cal.attn_alpha = cj["attn_alpha"];
    cal.attn_norm_window = cj["attn_norm_window"];
    cal.stream_scale = stream_scale;
    cal.attn_sink.assign(kTokens, 1.0);
    cal.ffn_sink.assign(kTokens, 1.0);
    if (cj.contains("attn_sink")) {
      const auto v = cj["attn_sink"].get<std::vector<double>>();
      for (size_t i = 0; i < v.size() && i < cal.attn_sink.size(); i++) {
        cal.attn_sink[i] = v[i];
      }
    }
    if (cj.contains("ffn_sink")) {
      const auto v = cj["ffn_sink"].get<std::vector<double>>();
      for (size_t i = 0; i < v.size() && i < cal.ffn_sink.size(); i++) {
        cal.ffn_sink[i] = v[i];
      }
    }
    const double s_min = cj["s_raw_min"], s_max = cj["s_raw_max"];
    cal.span_raw = s_max - s_min;
    cal.s_raw_max = s_max;
    cal.m_eff = cj["span"];
    cal.row_shift_raw.assign(kHeads, std::vector<double>(kTokens, 0.0));
    cal.row_norm.assign(kHeads, std::vector<double>(kTokens, 1.0));
    for (int h = 0; h < kHeads; h++) {
      for (int t = 0; t < kTokens; t++) {
        cal.row_shift_raw[h][t] = cj["row_shift_raw"][h][t].get<double>();
        cal.row_norm[h][t] = cj["row_norm"][h][t].get<double>();
      }
    }
    const double cqk = score_ride / std::max(std::abs(s_min), std::abs(s_max));
    cal.cq = cal.ck = std::sqrt(cqk);
    WidenForInstanceFactors(cal.alpha, cal.norm_window, 0.98, 1.02);
    WidenForInstanceFactors(cal.attn_alpha, cal.attn_norm_window, 0.98, 1.02);

    bctx->ResetBootCounts();
    auto tl0 = Sync();
    std::vector<Ciphertext<word>> next;
    layer.Layer(next, stream, aw, fw, cal, attn, akeys, boot.ui->GetEvkMap());
    auto tl1 = Sync();
    stream = std::move(next);
    const auto st = layer.GetStages();
    const auto counts = bctx->GetBootCounts();
    total_s += Ms(tl0, tl1) / 1000.0;
    std::cout << "  LAYER " << L << ", " << B << " instances: "
              << Ms(tl0, tl1) / 1000.0 << " s wall = " << Ms(tl0, tl1) / B
              << " ms per instance-layer; boots " << counts.full
              << " (tables " << layer.GetPrepareSeconds()
              << " s); boot " << st.boot << " norm " << st.norm << " q/k/v "
              << st.qkv << " scores " << st.scores << " softmax "
              << st.softmax << " values " << st.values << " o " << st.o
              << " gate/up " << st.gate_up << " silu " << st.silu << " down "
              << st.down << " s; " << FreeMiB() << " MiB free" << std::endl;

    // The reference instance against the exporter's residual after layer L.
    std::vector<double> h_ref;
    ASSERT_TRUE(ReadF64(rd + "/h_L" + (L < 10 ? "0" : "") + std::to_string(L) +
                            ".f64",
                        static_cast<size_t>(kTokens) * kH, h_ref));
    HostTensor got{B, kTokens, kH, {}};
    got.v.assign(static_cast<size_t>(B) * kTokens * kH, 0.0);
    std::vector<int> all_c(kH);
    for (int c = 0; c < kH; c++) all_c[c] = c;
    DecryptChannels(boot, layout, stream, all_c, got);
    std::vector<double> out(static_cast<size_t>(kTokens) * kH);
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < kH; c++) {
        out[static_cast<size_t>(t) * kH + c] = got.At(b_ref, t, c) / stream_scale;
      }
    }
    const double e_user = RmsRel(out, h_ref, kTokens, kH, kSinkTokens);
    const double e_all = RmsRel(out, h_ref, kTokens, kH, 0);
    std::cout << "  layer " << L << " output, instance " << b_ref
              << " vs h_L" << L << ".f64: 2^-" << std::setprecision(2)
              << Bits(e_user) << " over user tokens (2^-" << Bits(e_all)
              << " over all)" << std::endl;
    EXPECT_LT(e_user, std::ldexp(1.0, -4)) << "layer " << L;
  }
  std::cout << "  " << num_layers << " layer(s): " << total_s << " s = "
            << total_s / num_layers / B * 1000.0 << " ms per instance-layer"
            << std::endl;
}

// ---------------------------------------------------------------------------
// 8. The split built one column at a time is the split built at once: the
//    same projection from both, value for value.
// ---------------------------------------------------------------------------
TEST(CiBatch, TheIncrementalSplitProjectsLikeTheWholeOne) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring ring(Param());
  const CiBatchLayout layout(ring.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  constexpr int kIn = 40, kOut = 24;
  const int level = 7;
  std::mt19937_64 gen(11);
  std::uniform_real_distribution<double> ux(-1.0, 1.0), uw(-0.05, 0.05);
  HostTensor x{B, kTokens, kIn, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * kIn);
  for (auto &v : x.v) v = ux(gen);
  std::vector<float> w(static_cast<size_t>(kIn) * kOut);
  for (auto &v : w) v = static_cast<float>(uw(gen));
  cheddar::DeviceVector<float> w_dev;
  ToDevice(w_dev, w);
  std::vector<Ciphertext<word>> cts;
  EncryptChannels(ring, layout, x, level, cts);

  typename cheddar::CiBatchProjection<word>::Config cfg;
  cfg.rows_per_tile = 16;
  cheddar::CiBatchProjection<word> proj(ring.context, cfg);
  proj.Prepare("w", w_dev.data(), kIn, kOut, level);

  std::vector<Ciphertext<word>> whole, incremental;
  proj.Project(whole, cts, "w");
  {
    typename cheddar::CiBatchProjection<word>::Source src;
    proj.BeginSplit(src, kIn, level, layout.num_slots);
    for (int c = 0; c < kIn; c++) proj.AddColumn(src, c, cts[c]);
    for (int t = 0; t < proj.NumTiles("w"); t++) {
      std::vector<Ciphertext<word>> part;
      proj.Project(part, src, "w", t);
      for (auto &p : part) incremental.push_back(std::move(p));
    }
  }
  ASSERT_EQ(whole.size(), incremental.size());
  std::vector<int> all_b(B), all_o(kOut);
  for (int b = 0; b < B; b++) all_b[b] = b;
  for (int o = 0; o < kOut; o++) all_o[o] = o;
  HostTensor a{B, kTokens, kOut, {}}, c2{B, kTokens, kOut, {}};
  a.v.assign(static_cast<size_t>(B) * kTokens * kOut, 0.0);
  c2.v = a.v;
  DecryptChannels(ring, layout, whole, all_o, a);
  DecryptChannels(ring, layout, incremental, all_o, c2);
  HostTensor want;
  HostProject(x, w, kOut, all_b, all_o, want);
  const Err ew = Compare(a, want, all_b, all_o);
  const Err ei = Compare(c2, want, all_b, all_o);
  const Err d = Compare(c2, a, all_b, all_o);
  std::cout << "  whole split: 2^-" << std::fixed << std::setprecision(2)
            << Bits(ew.rms_rel) << ", incremental: 2^-" << Bits(ei.rms_rel)
            << ", the two apart: " << std::scientific << d.max_abs << std::fixed
            << std::endl;
  EXPECT_LT(ew.rms_rel, std::ldexp(1.0, -15));
  EXPECT_LT(ei.rms_rel, std::ldexp(1.0, -15));
  EXPECT_LT(d.max_abs, 1e-9) << "the two splits must give the same words";
#endif
}

// ---------------------------------------------------------------------------
// 9. The norm alone, at a small width: NormTurn -> an identity projection
//    onto the first channels -> the host's x / sqrt(mean(x^2) + eps).
// ---------------------------------------------------------------------------
// The decode dense-repacking microbenchmark (the two decode design notes,
// 2026-09-04): 128 replica-C-layout channel ciphertexts are PACKED into one
// dense ciphertext (mask + add), the ONE ciphertext bootstraps, and the
// channels are UNPACKED back to replica form (mask + 7 rotate-add
// broadcasts each) -- against the baseline of 128 individual bootstraps.
// Also times the dense-VMM primitive triple (rotate + pt-mult + add) for
// the design-2 projection estimate. Synthetic data; correctness checked to
// boot-family precision on sample channels.
TEST(CiBatch, DecodePackBootUnpackBench) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const CiBatchLayout layout(boot.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  const int K = kTokens;  // 128 channels = one dense ct
  const int lvl_in = 1;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, layout.num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  const int land = bctx->GetBootParameter().GetEndLevel();
  // Broadcast strides over the token axis (slot = t * B + b).
  std::vector<int> strides;
  for (int s = B; s < B * kTokens; s <<= 1) strides.push_back(s);
  for (int s : strides) {
    boot.ui->PrepareRotationKey(s, land);
    boot.ui->PrepareRotationKey(s, 3);
  }
  // Keys for the hoisted unpack's baby strides (the power-of-two ladder
  // strides are above; a key made at `land` serves every level below it).
  const int n_baby = std::max(2, EnvInt("CHEDDAR_DECODE_UNPACK_BABY", 16));
  for (int b = 1; b < n_baby; b++) {
    boot.ui->PrepareRotationKey(b * B, land);
  }
  const auto &mult_key = boot.ui->GetEvkMap().GetMultiplicationKey();
  (void)mult_key;

  // Data: x[c][b], replicated over tokens in the C-layout.
  std::mt19937_64 gen(0x9e1);
  std::uniform_real_distribution<double> ux(-0.3, 0.3);
  std::vector<std::vector<double>> x(K, std::vector<double>(B));
  for (auto &row : x)
    for (auto &v : row) v = ux(gen);
  HostTensor hx{B, kTokens, K, {}};
  hx.v.resize(static_cast<size_t>(B) * kTokens * K);
  for (int b = 0; b < B; b++)
    for (int t = 0; t < kTokens; t++)
      for (int c = 0; c < K; c++) hx.At(b, t, c) = x[c][b];
  std::vector<Ciphertext<word>> cts;
  EncryptChannels(boot, layout, hx, lvl_in, cts);

  // Masks: token == c, one set at the input level, one at the landing.
  auto make_mask = [&](int c, int level, Plaintext<word> &pt) {
    std::vector<double> vals(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) vals[static_cast<size_t>(b) * kTokens + c] = 1.0;
    std::vector<Complex> msg;
    layout.Pack(msg, vals);
    boot.context->gpu_encoder_.Encode(pt, level, boot.param->GetScale(level),
                                      msg);
  };
  std::vector<Plaintext<word>> mask_in(K), mask_land(K);
  for (int c = 0; c < K; c++) {
    make_mask(c, lvl_in, mask_in[c]);
    make_mask(c, land, mask_land[c]);
  }

  // --- Baseline: 128 individual (grouped) bootstraps.
  auto t0 = Sync();
  {
    const int group = EnvInt("CHEDDAR_EVALMOD_BATCH", 32);
    for (int c0 = 0; c0 < K; c0 += group) {
      const int g = std::min(K - c0, group);
      std::vector<const Ciphertext<word> *> in(g);
      for (int j = 0; j < g; j++) in[j] = &cts[c0 + j];
      std::vector<Ciphertext<word>> out_g;
      bctx->BootBatch(out_g, in, boot.ui->GetEvkMap());
    }
  }
  auto t1 = Sync();

  // --- Candidate: pack -> one boot -> unpack.
  Ciphertext<word> dense;
  {
    Ciphertext<word> acc;
    for (int c = 0; c < K; c++) {
      Ciphertext<word> m;
      bctx->Mult(m, cts[c], mask_in[c]);
      if (c == 0) {
        acc = std::move(m);
      } else {
        bctx->Add(acc, acc, m);
      }
    }
    bctx->Rescale(dense, acc);
  }
  auto t2 = Sync();
  Ciphertext<word> dense_up;
  bctx->Boot(dense_up, dense, boot.ui->GetEvkMap());
  auto t3 = Sync();
  // The decode WORKING level: both unpacks may run below the landing (the
  // consumers do not need the full boot budget), which shrinks every rotation
  // and mask. Untimed: this models where decode would already sit.
  const int u_lvl =
      std::min(land, std::max(4, EnvInt("CHEDDAR_DECODE_UNPACK_LEVEL", land)));
  if (u_lvl < land) {
    Ciphertext<word> down;
    bctx->LevelDown(down, dense_up, u_lvl);
    dense_up = std::move(down);
    for (int c = 0; c < K; c++) make_mask(c, u_lvl, mask_land[c]);
  }
  auto t3b = Sync();
  std::vector<Ciphertext<word>> unpacked(K);
  {
    for (int c = 0; c < K; c++) {
      Ciphertext<word> m, mm;
      bctx->Mult(mm, dense_up, mask_land[c]);
      bctx->Rescale(m, mm);
      for (int s : strides) {
        Ciphertext<word> r;
        bctx->HRot(r, m, boot.ui->GetRotationKey(s), s);
        bctx->Add(m, m, r);
      }
      unpacked[c] = std::move(m);
    }
  }
  auto t4 = Sync();

  // --- Hoisted unpack: the babies rot_{bB}(dense_up) are computed ONCE and
  // shared by every channel (the hoisting); channel c is then an indicator
  // SELECT over the babies (n_baby masked babies summed, one rescale) and a
  // log ladder over the remaining strides. Key switches:
  // (n_baby - 1) + K * log2(K / n_baby) against the naive K * log2(K).
  std::vector<Ciphertext<word>> hoisted(K);
  {
    const int dir = EnvInt("CHEDDAR_DECODE_UNPACK_DIR", -1);
    std::vector<Ciphertext<word>> baby_store(n_baby);
    std::vector<const Ciphertext<word> *> babies(n_baby);
    babies[0] = &dense_up;
    for (int b = 1; b < n_baby; b++) {
      bctx->HRot(baby_store[b], dense_up, boot.ui->GetRotationKey(b * B),
                 b * B);
      babies[b] = &baby_store[b];
    }
    for (int c = 0; c < K; c++) {
      Ciphertext<word> acc;
      for (int b = 0; b < n_baby; b++) {
        const int row = ((c + dir * b) % K + K) % K;
        Ciphertext<word> pm;
        bctx->Mult(pm, *babies[b], mask_land[row]);
        if (b == 0) {
          acc = std::move(pm);
        } else {
          bctx->Add(acc, acc, pm);
        }
      }
      Ciphertext<word> sel;
      bctx->Rescale(sel, acc);
      for (int s = n_baby * B; s < B * kTokens; s <<= 1) {
        Ciphertext<word> r;
        bctx->HRot(r, sel, boot.ui->GetRotationKey(s), s);
        bctx->Add(sel, sel, r);
      }
      hoisted[c] = std::move(sel);
    }
  }
  auto t5 = Sync();

  // --- Dense-VMM primitive triple at a decode working level.
  double triple_ms = 0.0;
  {
    Ciphertext<word> w, acc2;
    bctx->LevelDown(w, dense_up, 3);
    Plaintext<word> mask3;
    make_mask(0, 3, mask3);
    auto p0 = Sync();
    const int reps = 128;
    for (int i = 0; i < reps; i++) {
      Ciphertext<word> rr, pm;
      bctx->HRot(rr, w, boot.ui->GetRotationKey(B), B);
      bctx->Mult(pm, rr, mask3);
      if (i == 0) {
        acc2 = std::move(pm);
      } else {
        bctx->Add(acc2, acc2, pm);
      }
    }
    auto p1 = Sync();
    triple_ms = Ms(p0, p1) / reps;
  }

  // Correctness on sample channels (boot-family precision), both unpacks,
  // and the two unpacks against each other on one channel.
  double worst = 0.0, worst_h = 0.0, cross = 0.0;
  auto sample = [&](const Ciphertext<word> &ct, int c, double &w,
                    std::vector<double> *keep) {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, ct);
    std::vector<Complex> msg;
    boot.context->encoder_.Decode(msg, pt);
    std::vector<double> vals;
    layout.Unpack(vals, msg);
    for (int b = 0; b < B; b += 37) {
      for (int t = 0; t < kTokens; t += 17) {
        w = std::max(
            w, std::abs(vals[static_cast<size_t>(b) * kTokens + t] - x[c][b]));
      }
    }
    if (keep != nullptr) *keep = std::move(vals);
  };
  std::vector<double> naive63, hoist63;
  for (int c : {0, 63, 127}) {
    sample(unpacked[c], c, worst, c == 63 ? &naive63 : nullptr);
    sample(hoisted[c], c, worst_h, c == 63 ? &hoist63 : nullptr);
  }
  for (size_t i = 0; i < naive63.size(); i++) {
    cross = std::max(cross, std::abs(naive63[i] - hoist63[i]));
  }

  const double base_ms = Ms(t0, t1);
  const double pack_ms = Ms(t1, t2), boot_ms = Ms(t2, t3),
               unpack_ms = Ms(t3b, t4), hoist_ms = Ms(t4, t5);
  const double cand_ms = pack_ms + boot_ms + unpack_ms;
  std::cout << "  [bench] baseline 128 grouped boots: " << base_ms << " ms ("
            << base_ms / K << " ms/ct)" << std::endl;
  std::cout << "  [bench] pack " << pack_ms << " + boot " << boot_ms
            << " + unpack " << unpack_ms << " = " << cand_ms
            << " ms  -> speedup x" << base_ms / cand_ms << std::endl;
  std::cout << "  [bench] unpack alone: " << unpack_ms / K
            << " ms/channel (7 HRot + mask)" << std::endl;
  int ladder = 0;
  for (int s = n_baby; s < kTokens; s <<= 1) ladder++;
  std::cout << "  [bench] hoisted unpack (" << n_baby << " babies + select + "
            << ladder << "-step ladder, level " << u_lvl << "): " << hoist_ms
            << " ms (" << hoist_ms / K << " ms/channel) -> x"
            << unpack_ms / hoist_ms << " vs naive" << std::endl;
  std::cout << "  [bench] pack+boot+hoisted = "
            << pack_ms + boot_ms + hoist_ms << " ms -> x"
            << base_ms / (pack_ms + boot_ms + hoist_ms) << " vs 128 boots"
            << std::endl;
  std::cout << "  [bench] hoisted worst abs err " << worst_h
            << ", |hoisted - naive| ch63 max " << cross << std::endl;
  // Sum over the four projections of in*out/65536 = 3,322 triples per
  // token-layer (B-independent); naive, no hoisting.
  std::cout << "  [bench] dense-VMM triple (HRot+mask+add @L3): " << triple_ms
            << " ms -> naive dense VMM ~" << triple_ms * 3322.0
            << " ms per token-layer" << std::endl;
  std::cout << "  [bench] worst abs error after pack/boot/unpack: " << worst
            << std::endl;
  EXPECT_LT(worst, 5e-3);
  EXPECT_LT(worst_h, 5e-3);

  // --- The LIBRARY unpack (`CiDecodeUnpack`): all K rotations hoisted from
  // ONE ModUp, the select as ONE PAccumRotBatchCt launch, one mod-down per
  // channel, no ladder. Setup (mask compile, keys) is untimed, as a decode
  // deployment would hold both.
  {
    std::vector<cheddar::Message> row_masks(K);
    for (int j = 0; j < K; j++) {
      std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
      for (int b = 0; b < B; b++) {
        mv[static_cast<size_t>(b) * kTokens + j] = 1.0;
      }
      layout.Pack(row_masks[j], mv);
    }
    cheddar::CiDecodeUnpack<word> fast(
        boot.context, row_masks, B, u_lvl,
        boot.param->GetRescalePrimeProd(u_lvl));
    {
      cheddar::EvkRequest req;
      fast.AddRequiredRotations(req);
      boot.ui->PrepareRotationKey(req);
    }
    std::vector<Ciphertext<word>> fouts;
    auto f0 = Sync();
    fast.Evaluate(boot.context, fouts, dense_up, boot.ui->GetEvkMap());
    auto f1 = Sync();
    double worst_f = 0.0;
    for (int c : {0, 63, 127}) sample(fouts[c], c, worst_f, nullptr);
    const double fast_ms = Ms(f0, f1);
    std::cout << "  [bench] library unpack (hoisted, no ladder): " << fast_ms
              << " ms -> x" << unpack_ms / fast_ms << " vs naive; worst abs err "
              << worst_f << std::endl;
    std::cout << "  [bench] pack+boot+library = "
              << pack_ms + boot_ms + fast_ms << " ms -> x"
              << base_ms / (pack_ms + boot_ms + fast_ms) << " vs 128 boots"
              << std::endl;
    EXPECT_LT(worst_f, 5e-3);
  }
#endif
}

// The library unpack (`CiDecodeUnpack`, Doing.md 7.40 roadmap [1]) in
// isolation, no bootstrap anywhere: the dense ciphertext is encrypted
// directly at the working level. Channel 0 must be WORD-FOR-WORD the
// serial `HoistHandler::Evaluate` of the map the class compiles (same
// babies, same plaintexts, modular sums in a commuting order, same final
// mod-down); sampled channels must carry the broadcast values.
TEST(CiBatch, TheHoistedUnpackMatchesTheHandler) {
  Ring ring(Param());
  const CiBatchLayout layout(ring.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  const int K = kTokens;
  const int lvl = EnvInt("CHEDDAR_DECODE_UNPACK_LEVEL", 8);

  // x[c][b] in token row c of the dense ciphertext.
  std::mt19937_64 gen(0xdec0de);
  std::uniform_real_distribution<double> ux(-0.3, 0.3);
  std::vector<std::vector<double>> x(K, std::vector<double>(B));
  for (auto &row : x)
    for (auto &v : row) v = ux(gen);
  std::vector<double> vals(static_cast<size_t>(B) * kTokens);
  for (int b = 0; b < B; b++)
    for (int t = 0; t < kTokens; t++)
      vals[static_cast<size_t>(b) * kTokens + t] = x[t][b];
  std::vector<Complex> msg;
  layout.Pack(msg, vals);
  Plaintext<word> dpt;
  ring.context->gpu_encoder_.Encode(dpt, lvl, ring.param->GetScale(lvl), msg);
  Ciphertext<word> dense;
  ring.ui->Encrypt(dense, dpt);

  std::vector<cheddar::Message> row_masks(K);
  for (int j = 0; j < K; j++) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) mv[static_cast<size_t>(b) * kTokens + j] = 1.0;
    layout.Pack(row_masks[j], mv);
  }
  cheddar::CiDecodeUnpack<word> unpack(ring.context, row_masks, B, lvl,
                                       ring.param->GetRescalePrimeProd(lvl));
  {
    cheddar::EvkRequest req;
    unpack.AddRequiredRotations(req);
    ring.ui->PrepareRotationKey(req);
  }

  std::vector<Ciphertext<word>> outs;
  auto t0 = Sync();
  unpack.Evaluate(ring.context, outs, dense, ring.ui->GetEvkMap());
  auto t1 = Sync();
  ASSERT_EQ(static_cast<int>(outs.size()), K);

  // The word gate: channel 0 against the serial evaluation of the same map.
  Ciphertext<word> ref0;
  unpack.Handler().Evaluate(ring.context, ref0, dense, ring.ui->GetEvkMap());
  EXPECT_EQ(ref0.GetScale(), outs[0].GetScale());
  size_t diff = 0, total = 0;
  for (int part = 0; part < 2; part++) {
    const auto &da = part ? ref0.ax_ : ref0.bx_;
    const auto &db = part ? outs[0].ax_ : outs[0].bx_;
    ASSERT_EQ(da.size(), db.size());
    cheddar::HostVector<word> ha, hb;
    cheddar::CopyDeviceToHost(ha, da);
    cheddar::CopyDeviceToHost(hb, db);
    total += ha.size();
    for (size_t w = 0; w < ha.size(); w++) diff += (ha[w] != hb[w]);
  }
  std::cout << "  unpack vs handler: " << diff << " of " << total
            << " words differ" << std::endl;
  EXPECT_EQ(diff, 0u);

  // The values: sampled channels carry x[c][b] in every sampled row.
  double worst = 0.0;
  for (int c : {0, 1, 63, 127}) {
    Plaintext<word> pt;
    ring.ui->Decrypt(pt, outs[c]);
    std::vector<Complex> m2;
    ring.context->encoder_.Decode(m2, pt);
    std::vector<double> got;
    layout.Unpack(got, m2);
    for (int b = 0; b < B; b += 37) {
      for (int t = 0; t < kTokens; t += 17) {
        worst = std::max(
            worst,
            std::abs(got[static_cast<size_t>(b) * kTokens + t] - x[c][b]));
      }
    }
  }
  std::cout << "  unpack worst abs err " << worst << " at level " << lvl
            << ", " << Ms(t0, t1) << " ms for " << K << " channels"
            << std::endl;
  EXPECT_LT(worst, 2e-3);
}

// The DECODE NormTurn (Doing.md 7.40 roadmap [2]), one dense group in
// isolation: broadcast channel ciphertexts pack into ONE dense ct, ONE
// bootstrap refreshes all 128 channels, and -- unlike the prefill norm --
// the square-sum is computed AFTER the boot (dense^2, one relinearization,
// a 7-rotation ladder over the token axis), so nothing large is ever
// bootstrapped and the ride lesson never applies. The inverse square root
// runs at depth on the one accumulator (alpha at the geometric midpoint of
// the measured range, window = ratio x 1.3, beta = sqrt(alpha) folded into
// the declared scale), the normalizer multiplies the DENSE ct (one mult
// for 128 channels), and the hoisted unpack returns normalized broadcast
// channels. Host reference: y = x / sqrt(mean_c(x^2) + eps). Norm weights
// are not applied here -- in decode they fold into the next Kang
// projection's scalar weights.
TEST(CiBatch, TheDecodeNormMatchesTheHost) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const CiBatchLayout layout(boot.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  const int K = kTokens;  // the group's channels = one dense ct
  const double eps = 1e-5;
  const int lvl_in = 1;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, layout.num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  const auto &mult_key = boot.ui->GetEvkMap().GetMultiplicationKey();
  const int land = bctx->GetBootParameter().GetEndLevel();
  for (int s = B; s < B * kTokens; s <<= 1) {
    boot.ui->PrepareRotationKey(s, land);
  }

  // x[c][b], layer-like magnitudes, BROADCAST over the token axis (the
  // decode-resident form every op preserves).
  std::mt19937_64 gen(0x9107);
  std::uniform_real_distribution<double> ux(-1.0, 1.0);
  std::vector<std::vector<double>> x(K, std::vector<double>(B));
  for (auto &row : x)
    for (auto &v : row) v = 0.2 * ux(gen);
  HostTensor hx{B, kTokens, K, {}};
  hx.v.resize(static_cast<size_t>(B) * kTokens * K);
  for (int b = 0; b < B; b++)
    for (int t = 0; t < kTokens; t++)
      for (int c = 0; c < K; c++) hx.At(b, t, c) = x[c][b];
  std::vector<Ciphertext<word>> stream;
  EncryptChannels(boot, layout, hx, lvl_in, stream);

  auto make_mask = [&](int c, int level, Plaintext<word> &pt) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) mv[static_cast<size_t>(b) * kTokens + c] = 1.0;
    std::vector<Complex> msg;
    layout.Pack(msg, mv);
    boot.context->gpu_encoder_.Encode(pt, level, boot.param->GetScale(level),
                                      msg);
  };

  // The pack masks, compiled once (setup, untimed -- a deployment holds
  // them for every step).
  std::vector<Plaintext<word>> mask_in(K);
  for (int c = 0; c < K; c++) make_mask(c, lvl_in, mask_in[c]);

  // --- Pack: 128 broadcast channels into one dense ct.
  auto t0 = Sync();
  Ciphertext<word> dense;
  {
    Ciphertext<word> acc;
    for (int c = 0; c < K; c++) {
      Ciphertext<word> m;
      bctx->Mult(m, stream[c], mask_in[c]);
      if (c == 0) {
        acc = std::move(m);
      } else {
        bctx->Add(acc, acc, m);
      }
    }
    bctx->Rescale(dense, acc);
  }
  auto t1 = Sync();

  // --- One boot for the whole group.
  Ciphertext<word> dense_up;
  bctx->Boot(dense_up, dense, boot.ui->GetEvkMap());
  auto t2 = Sync();

  // --- The accumulator AFTER the boot: dense^2, one relinearization, the
  // ladder over the token axis -> sum of the group's squares, broadcast.
  Ciphertext<word> s2;
  {
    Ciphertext<word> sq;
    bctx->Mult(sq, dense_up, dense_up);
    bctx->RelinearizeRescale(s2, sq, mult_key);
    for (int s = B; s < B * kTokens; s <<= 1) {
      Ciphertext<word> r;
      bctx->HRot(r, s2, boot.ui->GetRotationKey(s), s);
      bctx->Add(s2, s2, r);
    }
  }
  auto t3 = Sync();

  // --- The affine onto the window and the inverse square root at depth.
  // Calibration from the data: u = alpha * (S / K + eps) with alpha at the
  // geometric midpoint of u's range and window = ratio x 1.3.
  double lo = 1e30, hi = 0.0;
  for (int b = 0; b < B; b++) {
    double s = 0.0;
    for (int c = 0; c < K; c++) s += x[c][b] * x[c][b];
    const double m = s / K + eps;
    lo = std::min(lo, m);
    hi = std::max(hi, m);
  }
  const double alpha = 1.0 / std::sqrt(lo * hi);
  const double window = (hi / lo) * 1.3;
  const double wl = 1.0 / std::sqrt(window), wh = std::sqrt(window);
  const double aw = 0.5 * (wh - wl), bw = 0.5 * (wh + wl);
  Ciphertext<word> r_inv;
  {
    const int l2 = boot.param->NPToLevel(s2.GetNP());
    Plaintext<word> pk;
    {
      std::vector<double> kv(static_cast<size_t>(B) * kTokens,
                             alpha / (static_cast<double>(K) * aw));
      std::vector<Complex> msg;
      layout.Pack(msg, kv);
      boot.context->gpu_encoder_.Encode(pk, l2, boot.param->GetScale(l2), msg);
    }
    Ciphertext<word> v;
    {
      Ciphertext<word> tmp;
      bctx->Mult(tmp, s2, pk);
      bctx->Rescale(v, tmp);
    }
    const int lv = boot.param->NPToLevel(v.GetNP());
    {
      cheddar::Constant<word> shift;
      boot.context->encoder_.EncodeConstant(
          shift, lv, boot.param->GetScale(lv), (alpha * eps - bw) / aw);
      bctx->Add(v, v, shift);
    }
    const int degree = 7;
    auto coeffs = cheddar::chebfit::Interpolate(
        [aw, bw](double t) { return 1.0 / std::sqrt(aw * t + bw); }, degree);
    const double in_scale = boot.param->GetScale(lv);
    const int used = cheddar::EvalPoly<word>(coeffs, lv, in_scale, in_scale,
                                             true)
                         .GetPolyDegree();
    int lr = lv;
    for (int d = used + 1; d > 1; d = (d + 1) / 2) lr--;
    cheddar::EvalPoly<word> inv(coeffs, lv, in_scale, boot.param->GetScale(lr),
                                true);
    inv.Compile(bctx);
    inv.Evaluate(bctx, r_inv, v, mult_key);
    // beta = sqrt(alpha) rides the declared scale: the polynomial gave
    // 1 / sqrt(alpha (m + eps)).
    r_inv.SetScale(r_inv.GetScale() / std::sqrt(alpha));
  }
  auto t4 = Sync();

  // --- Apply: ONE ciphertext multiply normalizes all 128 channels.
  Ciphertext<word> dense_n;
  {
    const int lr = boot.param->NPToLevel(r_inv.GetNP());
    Ciphertext<word> dd;
    bctx->LevelDown(dd, dense_up, lr);
    Ciphertext<word> prod;
    bctx->Mult(prod, dd, r_inv);
    bctx->RelinearizeRescale(dense_n, prod, mult_key);
  }
  auto t5 = Sync();

  // --- The hoisted unpack back to broadcast channels.
  const int lu = boot.param->NPToLevel(dense_n.GetNP());
  std::vector<cheddar::Message> row_masks(K);
  for (int j = 0; j < K; j++) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) mv[static_cast<size_t>(b) * kTokens + j] = 1.0;
    layout.Pack(row_masks[j], mv);
  }
  cheddar::CiDecodeUnpack<word> unpack(boot.context, row_masks, B, lu,
                                       boot.param->GetRescalePrimeProd(lu));
  {
    cheddar::EvkRequest req;
    unpack.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  auto t5b = Sync();
  std::vector<Ciphertext<word>> outs;
  unpack.Evaluate(boot.context, outs, dense_n, boot.ui->GetEvkMap());
  auto t6 = Sync();

  // --- Host reference: y = x / sqrt(mean_c(x^2) + eps).
  double num = 0.0, den = 0.0, worst = 0.0;
  for (int c : {0, 1, 63, 127}) {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, outs[c]);
    std::vector<Complex> m2;
    boot.context->encoder_.Decode(m2, pt);
    std::vector<double> got;
    layout.Unpack(got, m2);
    for (int b = 0; b < B; b += 23) {
      double s = 0.0;
      for (int cc = 0; cc < K; cc++) s += x[cc][b] * x[cc][b];
      const double ref = x[c][b] / std::sqrt(s / K + eps);
      for (int t = 0; t < kTokens; t += 31) {
        const double g = got[static_cast<size_t>(b) * kTokens + t];
        num += (g - ref) * (g - ref);
        den += ref * ref;
        worst = std::max(worst, std::abs(g - ref));
      }
    }
  }
  const double bits = 0.5 * std::log2(num / den);
  std::cout << "  [decode norm] pack " << Ms(t0, t1) << " + boot "
            << Ms(t1, t2) << " + accum " << Ms(t2, t3) << " + invsqrt "
            << Ms(t3, t4) << " + apply " << Ms(t4, t5) << " + unpack "
            << Ms(t5b, t6) << " = "
            << Ms(t0, t5) + Ms(t5b, t6) << " ms for " << K << " channels"
            << std::endl;
  std::cout << "  [decode norm] rms rel 2^" << bits << ", worst abs "
            << worst << "; window " << window << " alpha " << alpha
            << ", out level " << boot.param->NPToLevel(outs[0].GetNP())
            << std::endl;
  EXPECT_LT(bits, -8.0);
  EXPECT_LT(worst, 3e-2);
#endif
}

// The DECODE attention, one head in isolation (Doing.md 7.40 roadmap [3]):
// the query is a broadcast channel ct per head dim, the K cache is the
// prefill channel layout VERBATIM, and the score product is a pure
// channel reduction -- 128 elementwise ct products summed, one
// relinearization, the scores landing in the token rows of ONE ct. A
// gamma from the data folds the scores inside EvalMod's ride before the
// ONE boot; softmax is exp + a token-axis ladder + a reciprocal, all at
// depth on that one ct; the probabilities FAN OUT through the same
// hoisted unpack (row masks = token rows); and ScoreV is 128 elementwise
// products against V_t[d, b] (token-outside), landing the head's output
// in the d rows of one dense ct. Host reference: softmax(q K / sqrt(d)) V.
// No sinks, no GQA here -- assembly business.
TEST(CiBatch, TheDecodeAttentionHeadMatchesTheHost) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const CiBatchLayout layout(boot.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  const int D = kTokens;  // head dim = token rows = 128
  const int T = kTokens;  // cached history
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, layout.num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  const auto &mult_key = boot.ui->GetEvkMap().GetMultiplicationKey();
  const int land = bctx->GetBootParameter().GetEndLevel();
  for (int s = B; s < B * kTokens; s <<= 1) {
    boot.ui->PrepareRotationKey(s, land);
  }

  // q[c][b] (the decode token's head), K[c][t][b] (the cache, prefill
  // layout), V[d][t][b] (token-outside).
  std::mt19937_64 gen(0xa77e);
  std::uniform_real_distribution<double> ux(-1.0, 1.0);
  std::vector<std::vector<double>> q(D, std::vector<double>(B));
  std::vector<std::vector<std::vector<double>>> Kc(
      D, std::vector<std::vector<double>>(T, std::vector<double>(B)));
  auto Vc = Kc;  // V[d][t][b]
  for (auto &r : q)
    for (auto &v : r) v = 0.2 * ux(gen);
  for (auto &m : Kc)
    for (auto &r : m)
      for (auto &v : r) v = 0.2 * ux(gen);
  for (auto &m : Vc)
    for (auto &r : m)
      for (auto &v : r) v = 0.2 * ux(gen);

  const int lvl_qk = 2;
  std::vector<Ciphertext<word>> q_cts, k_cts;
  {
    HostTensor hq{B, kTokens, D, {}};
    hq.v.resize(static_cast<size_t>(B) * kTokens * D);
    HostTensor hk = hq;
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < kTokens; t++) {
        for (int c = 0; c < D; c++) {
          hq.At(b, t, c) = q[c][b];        // broadcast
          hk.At(b, t, c) = Kc[c][t][b];    // per-token rows
        }
      }
    }
    EncryptChannels(boot, layout, hq, lvl_qk, q_cts);
    EncryptChannels(boot, layout, hk, lvl_qk, k_cts);
  }

  // Host scores and softmax, and the calibration ranges.
  const double sqd = std::sqrt(static_cast<double>(D));
  std::vector<std::vector<double>> s_host(T, std::vector<double>(B));
  double smax = 0.0, s_lo = 1e30, s_hi = -1e30;
  for (int t = 0; t < T; t++) {
    for (int b = 0; b < B; b++) {
      double s = 0.0;
      for (int c = 0; c < D; c++) s += q[c][b] * Kc[c][t][b];
      s_host[t][b] = s;
      smax = std::max(smax, std::abs(s));
      s_lo = std::min(s_lo, s);
      s_hi = std::max(s_hi, s);
    }
  }
  const double gamma = 0.3 / smax;  // the booted score rides at <= 0.3

  // --- The score product: a channel reduction, one relinearization.
  auto t0 = Sync();
  Ciphertext<word> s_ct;
  {
    Ciphertext<word> acc;
    for (int c = 0; c < D; c++) {
      Ciphertext<word> m;
      bctx->Mult(m, q_cts[c], k_cts[c]);
      if (c == 0) {
        acc = std::move(m);
      } else {
        bctx->Add(acc, acc, m);
      }
    }
    Ciphertext<word> rel;
    bctx->RelinearizeRescale(rel, acc, mult_key);
    // gamma folds the scores inside the ride; the exp affine unfolds it.
    const int l = boot.param->NPToLevel(rel.GetNP());
    Plaintext<word> pg;
    {
      std::vector<double> gv(static_cast<size_t>(B) * kTokens, gamma);
      std::vector<Complex> msg;
      layout.Pack(msg, gv);
      boot.context->gpu_encoder_.Encode(pg, l, boot.param->GetScale(l), msg);
    }
    Ciphertext<word> gm;
    bctx->Mult(gm, rel, pg);
    bctx->Rescale(s_ct, gm);
  }
  auto t1 = Sync();
  Ciphertext<word> s_up;
  bctx->Boot(s_up, s_ct, boot.ui->GetEvkMap());
  auto t2 = Sync();

  // A generic "affine onto [-1, 1] then a Chebyshev polynomial" step.
  auto poly_step = [&](const Ciphertext<word> &in, double in_factor,
                       double lo, double hi, auto f, int degree,
                       Ciphertext<word> &out) {
    // in holds in_factor * u for u in [lo, hi]; out = f(u).
    const double pad = 0.02 * (hi - lo) + 1e-12;
    lo -= pad;
    hi += pad;
    const double a = 0.5 * (hi - lo), b = 0.5 * (hi + lo);
    const int l = boot.param->NPToLevel(in.GetNP());
    Plaintext<word> pk;
    {
      std::vector<double> kv(static_cast<size_t>(B) * kTokens,
                             1.0 / (in_factor * a));
      std::vector<Complex> msg;
      layout.Pack(msg, kv);
      boot.context->gpu_encoder_.Encode(pk, l, boot.param->GetScale(l), msg);
    }
    Ciphertext<word> v;
    {
      Ciphertext<word> tmp;
      bctx->Mult(tmp, in, pk);
      bctx->Rescale(v, tmp);
    }
    const int lv = boot.param->NPToLevel(v.GetNP());
    {
      cheddar::Constant<word> shift;
      boot.context->encoder_.EncodeConstant(shift, lv,
                                            boot.param->GetScale(lv), -b / a);
      bctx->Add(v, v, shift);
    }
    auto coeffs = cheddar::chebfit::Interpolate(
        [a, b, f](double t) { return f(a * t + b); }, degree);
    const double in_scale = boot.param->GetScale(lv);
    const int used =
        cheddar::EvalPoly<word>(coeffs, lv, in_scale, in_scale, true)
            .GetPolyDegree();
    int lr = lv;
    for (int d = used + 1; d > 1; d = (d + 1) / 2) lr--;
    cheddar::EvalPoly<word> poly(coeffs, lv, in_scale,
                                 boot.param->GetScale(lr), true);
    poly.Compile(bctx);
    poly.Evaluate(bctx, out, v, mult_key);
  };

  // --- Softmax on the ONE score ct: exp, the ladder, a reciprocal.
  Ciphertext<word> p_ct;
  {
    // e = exp(u), u = s / sqrt(d): the booted message is gamma * s
    // = (gamma * sqrt(d)) * u.
    Ciphertext<word> e;
    poly_step(s_up, gamma * sqd, s_lo / sqd, s_hi / sqd,
              [](double u) { return std::exp(u); }, 15, e);
    // Z = sum_t e, broadcast over the rows.
    Ciphertext<word> z;
    bctx->LevelDown(z, e, boot.param->NPToLevel(e.GetNP()));
    for (int s = B; s < B * kTokens; s <<= 1) {
      Ciphertext<word> r;
      bctx->HRot(r, z, boot.ui->GetRotationKey(s), s);
      bctx->Add(z, z, r);
    }
    // The reciprocal's range from the host.
    double z_lo = 1e30, z_hi = -1e30;
    for (int b = 0; b < B; b++) {
      double zz = 0.0;
      for (int t = 0; t < T; t++) zz += std::exp(s_host[t][b] / sqd);
      z_lo = std::min(z_lo, zz);
      z_hi = std::max(z_hi, zz);
    }
    Ciphertext<word> r_inv;
    poly_step(z, 1.0, z_lo, z_hi, [](double u) { return 1.0 / u; }, 15,
              r_inv);
    // P = e * (1 / Z).
    Ciphertext<word> ed;
    bctx->LevelDown(ed, e, boot.param->NPToLevel(r_inv.GetNP()));
    Ciphertext<word> prod;
    bctx->Mult(prod, ed, r_inv);
    bctx->RelinearizeRescale(p_ct, prod, mult_key);
  }
  auto t3 = Sync();

  // --- The fan-out: the same hoisted unpack, row masks = token rows.
  const int lp = boot.param->NPToLevel(p_ct.GetNP());
  std::vector<cheddar::Message> row_masks(T);
  for (int j = 0; j < T; j++) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) mv[static_cast<size_t>(b) * kTokens + j] = 1.0;
    layout.Pack(row_masks[j], mv);
  }
  cheddar::CiDecodeUnpack<word> fanout(boot.context, row_masks, B, lp,
                                       boot.param->GetRescalePrimeProd(lp));
  {
    cheddar::EvkRequest req;
    fanout.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  auto t3b = Sync();
  std::vector<Ciphertext<word>> p_t;
  fanout.Evaluate(boot.context, p_t, p_ct, boot.ui->GetEvkMap());
  auto t4 = Sync();

  // --- ScoreV: 128 elementwise products against V_t[d, b].
  const int lv_v = boot.param->NPToLevel(p_t[0].GetNP());
  std::vector<Ciphertext<word>> v_cts;
  {
    HostTensor hv{B, kTokens, T, {}};
    hv.v.resize(static_cast<size_t>(B) * kTokens * T);
    for (int b = 0; b < B; b++) {
      for (int d = 0; d < kTokens; d++) {
        for (int t = 0; t < T; t++) hv.At(b, d, t) = Vc[d][t][b];
      }
    }
    EncryptChannels(boot, layout, hv, lv_v, v_cts);
  }
  auto t4b = Sync();
  Ciphertext<word> out_ct;
  {
    Ciphertext<word> acc;
    for (int t = 0; t < T; t++) {
      Ciphertext<word> m;
      bctx->Mult(m, p_t[t], v_cts[t]);
      if (t == 0) {
        acc = std::move(m);
      } else {
        bctx->Add(acc, acc, m);
      }
    }
    bctx->RelinearizeRescale(out_ct, acc, mult_key);
  }
  auto t5 = Sync();

  // --- Host reference: softmax(q K / sqrt(d)) V, sampled.
  double num = 0.0, den = 0.0, worst = 0.0;
  {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, out_ct);
    std::vector<Complex> m2;
    boot.context->encoder_.Decode(m2, pt);
    std::vector<double> got;
    layout.Unpack(got, m2);
    for (int b = 0; b < B; b += 23) {
      std::vector<double> p(T);
      double z = 0.0;
      for (int t = 0; t < T; t++) {
        p[t] = std::exp(s_host[t][b] / sqd);
        z += p[t];
      }
      for (int t = 0; t < T; t++) p[t] /= z;
      for (int d = 0; d < kTokens; d += 7) {
        double ref = 0.0;
        for (int t = 0; t < T; t++) ref += p[t] * Vc[d][t][b];
        const double g = got[static_cast<size_t>(b) * kTokens + d];
        num += (g - ref) * (g - ref);
        den += ref * ref;
        worst = std::max(worst, std::abs(g - ref));
      }
    }
  }
  const double bits = 0.5 * std::log2(num / den);
  std::cout << "  [decode attn] qk+gamma " << Ms(t0, t1) << " + boot "
            << Ms(t1, t2) << " + softmax " << Ms(t2, t3) << " + fanout "
            << Ms(t3b, t4) << " + scorev " << Ms(t4b, t5) << " = "
            << Ms(t0, t3) + Ms(t3b, t4) + Ms(t4b, t5) << " ms for one head"
            << std::endl;
  std::cout << "  [decode attn] rms rel 2^" << bits << ", worst abs " << worst
            << "; gamma " << gamma << ", P level " << lp << ", out level "
            << boot.param->NPToLevel(out_ct.GetNP()) << std::endl;
  EXPECT_LT(bits, -6.0);

  // --- ScoreV route B: V kept in the K LAYOUT (channel cts V_d[t, b], no
  // one-time transpose, no fan-out): out_d = rowsum(P . V_d), one mult and
  // a 7-rotation ladder per head dim, output BROADCAST channel cts -- the
  // form the Kang O projection wants anyway.
  {
    std::vector<Ciphertext<word>> vk_cts;
    {
      HostTensor hv{B, kTokens, kTokens, {}};
      hv.v.resize(static_cast<size_t>(B) * kTokens * kTokens);
      for (int b = 0; b < B; b++) {
        for (int t = 0; t < kTokens; t++) {
          for (int d = 0; d < kTokens; d++) hv.At(b, t, d) = Vc[d][t][b];
        }
      }
      EncryptChannels(boot, layout, hv, lp, vk_cts);
    }
    auto b0 = Sync();
    std::vector<Ciphertext<word>> out_b(kTokens);
    for (int d = 0; d < kTokens; d++) {
      Ciphertext<word> m, y;
      bctx->Mult(m, p_ct, vk_cts[d]);
      bctx->RelinearizeRescale(y, m, mult_key);
      for (int s = B; s < B * kTokens; s <<= 1) {
        Ciphertext<word> r;
        bctx->HRot(r, y, boot.ui->GetRotationKey(s), s);
        bctx->Add(y, y, r);
      }
      out_b[d] = std::move(y);
    }
    auto b1 = Sync();
    double num_b = 0.0, den_b = 0.0;
    for (int d : {0, 63, 127}) {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, out_b[d]);
      std::vector<Complex> m2;
      boot.context->encoder_.Decode(m2, pt);
      std::vector<double> got;
      layout.Unpack(got, m2);
      for (int b = 0; b < B; b += 23) {
        std::vector<double> p(T);
        double z = 0.0;
        for (int t = 0; t < T; t++) {
          p[t] = std::exp(s_host[t][b] / sqd);
          z += p[t];
        }
        double ref = 0.0;
        for (int t = 0; t < T; t++) ref += (p[t] / z) * Vc[d][t][b];
        for (int t = 0; t < kTokens; t += 31) {
          const double g = got[static_cast<size_t>(b) * kTokens + t];
          num_b += (g - ref) * (g - ref);
          den_b += ref * ref;
        }
      }
    }
    std::cout << "  [decode attn] route B (V in the K layout, mult+ladder "
              << "per dim): " << Ms(b0, b1) << " ms, rms rel 2^"
              << 0.5 * std::log2(num_b / den_b)
              << " -- no V transpose, no fan-out" << std::endl;
    EXPECT_LT(0.5 * std::log2(num_b / den_b), -6.0);
  }
#endif
}

// The DECODE FFN middle (Doing.md 7.40 roadmap [5]), one dense group in
// isolation: the gate and up pre-activations (layer-like +-3, far past
// EvalMod's ride) pack into TWO dense cts with the ride gammas FOLDED INTO
// THE PACK MASKS (a mask entry gamma instead of 1 -- free), each boots
// once, SiLU runs at depth on the gate ct (the affine unfolds gamma_g),
// the up ct unfolds ITS gamma through the declared scale (free), one
// product, and the hoisted unpack returns silu(g) * u as broadcast
// channels. Host reference: silu(g) * u. The Kang gate/up/down projections
// around this are prefill-proven and level-agnostic; they are not re-gated
// here.
TEST(CiBatch, TheDecodeFfnMatchesTheHost) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const CiBatchLayout layout(boot.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  const int K = kTokens;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, layout.num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  const auto &mult_key = boot.ui->GetEvkMap().GetMultiplicationKey();

  // Pre-activations, layer-like: |g|, |u| up to ~3.
  std::mt19937_64 gen(0xff17);
  std::uniform_real_distribution<double> ux(-1.0, 1.0);
  std::vector<std::vector<double>> g(K, std::vector<double>(B)), u = g;
  double gmax = 0.0, umax = 0.0;
  for (int c = 0; c < K; c++) {
    for (int b = 0; b < B; b++) {
      g[c][b] = 3.0 * ux(gen);
      u[c][b] = 3.0 * ux(gen);
      gmax = std::max(gmax, std::abs(g[c][b]));
      umax = std::max(umax, std::abs(u[c][b]));
    }
  }
  const double gam_g = 0.3 / gmax, gam_u = 0.3 / umax;
  const int lvl_in = 1;
  std::vector<Ciphertext<word>> g_cts, u_cts;
  {
    HostTensor hg{B, kTokens, K, {}};
    hg.v.resize(static_cast<size_t>(B) * kTokens * K);
    HostTensor hu = hg;
    for (int b = 0; b < B; b++)
      for (int t = 0; t < kTokens; t++)
        for (int c = 0; c < K; c++) {
          hg.At(b, t, c) = g[c][b];
          hu.At(b, t, c) = u[c][b];
        }
    EncryptChannels(boot, layout, hg, lvl_in, g_cts);
    EncryptChannels(boot, layout, hu, lvl_in, u_cts);
  }

  // Pack masks CARRYING the gammas (setup, untimed).
  auto make_gmask = [&](int c, double gamma, Plaintext<word> &pt) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) {
      mv[static_cast<size_t>(b) * kTokens + c] = gamma;
    }
    std::vector<Complex> msg;
    layout.Pack(msg, mv);
    boot.context->gpu_encoder_.Encode(pt, lvl_in,
                                      boot.param->GetScale(lvl_in), msg);
  };
  std::vector<Plaintext<word>> mg(K), mu(K);
  for (int c = 0; c < K; c++) {
    make_gmask(c, gam_g, mg[c]);
    make_gmask(c, gam_u, mu[c]);
  }

  auto pack = [&](const std::vector<Ciphertext<word>> &cts,
                  const std::vector<Plaintext<word>> &masks,
                  Ciphertext<word> &dense) {
    Ciphertext<word> acc;
    for (int c = 0; c < K; c++) {
      Ciphertext<word> m;
      bctx->Mult(m, cts[c], masks[c]);
      if (c == 0) {
        acc = std::move(m);
      } else {
        bctx->Add(acc, acc, m);
      }
    }
    bctx->Rescale(dense, acc);
  };

  auto t0 = Sync();
  Ciphertext<word> dg, du;
  pack(g_cts, mg, dg);
  pack(u_cts, mu, du);
  auto t1 = Sync();
  Ciphertext<word> dg_up, du_up;
  bctx->Boot(dg_up, dg, boot.ui->GetEvkMap());
  bctx->Boot(du_up, du, boot.ui->GetEvkMap());
  auto t2 = Sync();

  // SiLU at depth on the gate ct: the booted message is gam_g * g.
  Ciphertext<word> sg;
  {
    const double lo = -gmax * 1.02, hi = gmax * 1.02;
    const double a = 0.5 * (hi - lo), b = 0.5 * (hi + lo);
    const int l = boot.param->NPToLevel(dg_up.GetNP());
    Plaintext<word> pk;
    {
      std::vector<double> kv(static_cast<size_t>(B) * kTokens,
                             1.0 / (gam_g * a));
      std::vector<Complex> msg;
      layout.Pack(msg, kv);
      boot.context->gpu_encoder_.Encode(pk, l, boot.param->GetScale(l), msg);
    }
    Ciphertext<word> v;
    {
      Ciphertext<word> tmp;
      bctx->Mult(tmp, dg_up, pk);
      bctx->Rescale(v, tmp);
    }
    const int lv = boot.param->NPToLevel(v.GetNP());
    {
      cheddar::Constant<word> shift;
      boot.context->encoder_.EncodeConstant(shift, lv,
                                            boot.param->GetScale(lv), -b / a);
      bctx->Add(v, v, shift);
    }
    auto coeffs = cheddar::chebfit::Interpolate(
        [a, b](double t) {
          const double x = a * t + b;
          return x / (1.0 + std::exp(-x));
        },
        15);
    const double in_scale = boot.param->GetScale(lv);
    const int used =
        cheddar::EvalPoly<word>(coeffs, lv, in_scale, in_scale, true)
            .GetPolyDegree();
    int lr = lv;
    for (int d = used + 1; d > 1; d = (d + 1) / 2) lr--;
    cheddar::EvalPoly<word> silu(coeffs, lv, in_scale,
                                 boot.param->GetScale(lr), true);
    silu.Compile(bctx);
    silu.Evaluate(bctx, sg, v, mult_key);
  }
  // The up ct unfolds its gamma through the declared scale, free.
  du_up.SetScale(du_up.GetScale() * gam_u);
  Ciphertext<word> dm;
  {
    const int lr = boot.param->NPToLevel(sg.GetNP());
    Ciphertext<word> dd;
    bctx->LevelDown(dd, du_up, lr);
    Ciphertext<word> prod;
    bctx->Mult(prod, sg, dd);
    bctx->RelinearizeRescale(dm, prod, mult_key);
  }
  auto t3 = Sync();

  // The hoisted unpack back to broadcast channels.
  const int lu = boot.param->NPToLevel(dm.GetNP());
  std::vector<cheddar::Message> row_masks(K);
  for (int j = 0; j < K; j++) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) mv[static_cast<size_t>(b) * kTokens + j] = 1.0;
    layout.Pack(row_masks[j], mv);
  }
  cheddar::CiDecodeUnpack<word> unpack(boot.context, row_masks, B, lu,
                                       boot.param->GetRescalePrimeProd(lu));
  {
    cheddar::EvkRequest req;
    unpack.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  auto t3b = Sync();
  std::vector<Ciphertext<word>> outs;
  unpack.Evaluate(boot.context, outs, dm, boot.ui->GetEvkMap());
  auto t4 = Sync();

  // Host reference: silu(g) * u.
  double num = 0.0, den = 0.0, worst = 0.0;
  for (int c : {0, 1, 63, 127}) {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, outs[c]);
    std::vector<Complex> m2;
    boot.context->encoder_.Decode(m2, pt);
    std::vector<double> got;
    layout.Unpack(got, m2);
    for (int b = 0; b < B; b += 23) {
      const double gg = g[c][b];
      const double ref = gg / (1.0 + std::exp(-gg)) * u[c][b];
      for (int t = 0; t < kTokens; t += 31) {
        const double got_v = got[static_cast<size_t>(b) * kTokens + t];
        num += (got_v - ref) * (got_v - ref);
        den += ref * ref;
        worst = std::max(worst, std::abs(got_v - ref));
      }
    }
  }
  const double bits = 0.5 * std::log2(num / den);
  std::cout << "  [decode ffn] pack " << Ms(t0, t1) << " + 2 boots "
            << Ms(t1, t2) << " + silu*u " << Ms(t2, t3) << " + unpack "
            << Ms(t3b, t4) << " = " << Ms(t0, t3) + Ms(t3b, t4)
            << " ms for " << K << " channels" << std::endl;
  std::cout << "  [decode ffn] rms rel 2^" << bits << ", worst abs " << worst
            << "; gam_g " << gam_g << " gam_u " << gam_u << ", out level "
            << boot.param->NPToLevel(outs[0].GetNP()) << std::endl;
  EXPECT_LT(bits, -7.0);
#endif
}

// The DECODE attention on the REAL layer-0 weights (roadmap [6]'s first
// assembly increment, [4]'s per-step appends included): 127 prompt tokens'
// K/V cache computed on the host in float64 (RMSNorm, GQA weight slices,
// RoPE at each position -- what the prefill leaves behind; the Sylph norm
// sink cancels exactly and does not reach the cache), the decode token at
// position 127 APPENDED encrypted (K: one masked add per channel; V: one
// pack), and four query heads spanning three KV heads run the [3]
// pipeline against the true float64 softmax(q k / sqrt(d)) V. Real-data
// machinery new here: the max-shift folded into the gamma affine
// (e <= 1), the exp WALK (a deg-15 polynomial on w / 2^k and k squarings,
// k from the measured score range), and the reciprocal applied AFTER
// ScoreV (one multiply on the output, saving a level and a
// relinearization). RoPE at the fixed decode position is a public linear
// map folded into the projection on the host, as the real layer folds it
// into the Kang weights. Instances are replicated (the synthetic gates
// varied them).
TEST(CiBatch, TheDecodeAttentionRunsOnTheRealLayer) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  const char *wdir_env = std::getenv("LLAMA3_ALL_DIR");
  if (wdir_env == nullptr) GTEST_SKIP() << "LLAMA3_ALL_DIR is not set";
  const std::string ld = std::string(wdir_env) + "/L00";
  constexpr int kH = 4096, kKv = 1024;
  const int D = kTokens, half = D / 2, T = kTokens, Tc = T - 1;
  const double eps = 1e-5, rope_base = 500000.0;
  const double sqd = std::sqrt(static_cast<double>(D));
  const std::vector<int> heads = {0, 1, 4, 31};
  std::vector<int> kvs;  // unique kv heads, in order
  for (int h : heads) {
    const int kv = h / 4;
    if (std::find(kvs.begin(), kvs.end(), kv) == kvs.end()) kvs.push_back(kv);
  }
  auto kv_slot = [&](int h) {
    return static_cast<int>(std::find(kvs.begin(), kvs.end(), h / 4) -
                            kvs.begin());
  };

  std::vector<float> x0, wq, wk, wv, gain;
  ASSERT_TRUE(ReadF32(ld + "/../input_nosink.f32",
                      static_cast<size_t>(T) * kH, x0));
  ASSERT_TRUE(ReadF32(ld + "/wq.f32", static_cast<size_t>(kH) * kH, wq));
  ASSERT_TRUE(ReadF32(ld + "/wk.f32", static_cast<size_t>(kH) * kKv, wk));
  ASSERT_TRUE(ReadF32(ld + "/wv.f32", static_cast<size_t>(kH) * kKv, wv));
  ASSERT_TRUE(ReadF32(ld + "/attn_norm.f32", kH, gain));

  // Host float64: the norm, the per-position K/V (RoPE'd), the decode
  // token's queries, the true softmax(q k / sqrt(d)) V.
  std::vector<double> y(static_cast<size_t>(T) * kH);
  for (int t = 0; t < T; t++) {
    double ms = 0.0;
    for (int c = 0; c < kH; c++) {
      const double v = x0[static_cast<size_t>(t) * kH + c];
      ms += v * v;
    }
    const double r = 1.0 / std::sqrt(ms / kH + eps);
    for (int c = 0; c < kH; c++) {
      y[static_cast<size_t>(t) * kH + c] =
          x0[static_cast<size_t>(t) * kH + c] * r * gain[c];
    }
  }
  auto rope = [&](std::vector<double> &m, int pos) {
    for (int c = 0; c < half; c++) {
      const double a = pos * std::pow(rope_base, -2.0 * c / D);
      const double lo = m[c], hi = m[c + half];
      m[c] = lo * std::cos(a) - hi * std::sin(a);
      m[c + half] = hi * std::cos(a) + lo * std::sin(a);
    }
  };
  const int nkv = static_cast<int>(kvs.size());
  // k_host[kv][t*D+d], v_host[kv][t*D+d] for t = 0..T-1 (T-1 is the decode
  // token at position T-1).
  std::vector<std::vector<double>> k_host(nkv,
                                          std::vector<double>(T * D)),
      v_host = k_host;
  cheddar::ParallelFor(T, [&](int begin, int end) {
    for (int t = begin; t < end; t++) {
      for (int s = 0; s < nkv; s++) {
        const int col = kvs[s] * D;
        std::vector<double> kk(D), vv(D);
        for (int d = 0; d < D; d++) {
          double ak = 0.0, av = 0.0;
          for (int c = 0; c < kH; c++) {
            const double yc = y[static_cast<size_t>(t) * kH + c];
            ak += yc * wk[static_cast<size_t>(c) * kKv + col + d];
            av += yc * wv[static_cast<size_t>(c) * kKv + col + d];
          }
          kk[d] = ak;
          vv[d] = av;
        }
        rope(kk, t);
        for (int d = 0; d < D; d++) {
          k_host[s][static_cast<size_t>(t) * D + d] = kk[d];
          v_host[s][static_cast<size_t>(t) * D + d] = vv[d];
        }
      }
    }
  });
  const int nh = static_cast<int>(heads.size());
  std::vector<std::vector<double>> q_host(nh, std::vector<double>(D));
  for (int i = 0; i < nh; i++) {
    for (int d = 0; d < D; d++) {
      double a = 0.0;
      for (int c = 0; c < kH; c++) {
        a += y[static_cast<size_t>(Tc) * kH + c] *
             wq[static_cast<size_t>(c) * kH + heads[i] * D + d];
      }
      q_host[i][d] = a;
    }
    rope(q_host[i], Tc);
  }
  std::vector<std::vector<double>> s_host(nh, std::vector<double>(T)),
      ref(nh, std::vector<double>(D));
  std::vector<double> s_lo(nh, 1e30), s_hi(nh, -1e30);
  for (int i = 0; i < nh; i++) {
    const int s = kv_slot(heads[i]);
    for (int l = 0; l < T; l++) {
      double a = 0.0;
      for (int d = 0; d < D; d++) {
        a += q_host[i][d] * k_host[s][static_cast<size_t>(l) * D + d];
      }
      s_host[i][l] = a;
      s_lo[i] = std::min(s_lo[i], a);
      s_hi[i] = std::max(s_hi[i], a);
    }
    double z = 0.0;
    std::vector<double> p(T);
    for (int l = 0; l < T; l++) {
      p[l] = std::exp((s_host[i][l] - s_hi[i]) / sqd);
      z += p[l];
    }
    for (int d = 0; d < D; d++) {
      double a = 0.0;
      for (int l = 0; l < T; l++) {
        a += (p[l] / z) * v_host[s][static_cast<size_t>(l) * D + d];
      }
      ref[i][d] = a;
    }
  }

  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const CiBatchLayout layout(boot.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, layout.num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  const auto &mult_key = boot.ui->GetEvkMap().GetMultiplicationKey();
  const int land = bctx->GetBootParameter().GetEndLevel();
  for (int s = B; s < B * kTokens; s <<= 1) {
    boot.ui->PrepareRotationKey(s, land);
  }

  // The walk's squaring count from the widest head's range.
  double r_w = 0.0;
  for (int i = 0; i < nh; i++) {
    r_w = std::max(r_w, (s_hi[i] - s_lo[i]) / sqd);
  }
  int k_sq = 0;
  while (r_w / (1 << k_sq) > 2.0) k_sq++;
  const int le = land - 1 - 3;        // exp poly's landing (deg 7)
  const int lf = le - k_sq;           // e after the squarings = fanout level
  const int lv_v = lf - 1;            // the V cache's level

  // Encryptions. The K cache rows 0..T-2 (row T-1 empty), the V cache
  // tokens 0..T-2, the decode token's k/v/q as broadcast channels.
  auto enc_group = [&](int level, auto fill,
                       std::vector<Ciphertext<word>> &cts) {
    HostTensor hx{B, kTokens, D, {}};
    hx.v.resize(static_cast<size_t>(B) * kTokens * D);
    for (int b = 0; b < B; b++)
      for (int t = 0; t < kTokens; t++)
        for (int c = 0; c < D; c++) hx.At(b, t, c) = fill(t, c);
    EncryptChannels(boot, layout, hx, level, cts);
  };
  std::vector<std::vector<Ciphertext<word>>> kc(nkv), vt(nkv), knew(nkv),
      vnew(nkv);
  for (int s = 0; s < nkv; s++) {
    enc_group(2,
              [&](int t, int c) {
                return t < Tc ? k_host[s][static_cast<size_t>(t) * D + c]
                              : 0.0;
              },
              kc[s]);
    enc_group(3,
              [&](int, int c) {
                return k_host[s][static_cast<size_t>(Tc) * D + c];
              },
              knew[s]);
    // V in the token-outside form: channel = token, rows = head dim.
    enc_group(lv_v,
              [&](int d, int t) {
                return t < Tc ? v_host[s][static_cast<size_t>(t) * D + d]
                              : 0.0;
              },
              vt[s]);
    enc_group(lf,
              [&](int, int c) {
                return v_host[s][static_cast<size_t>(Tc) * D + c];
              },
              vnew[s]);
  }
  // The ride gamma folds into q BEFORE the product (in the real layer,
  // into the Kang Q weights -- free): the scores then flow at <= 0.3 from
  // the first partial sum, never near the low-level message headroom (the
  // unscaled route wrapped head 31's +-251 scores at level 1's ~2^8 cap).
  std::vector<double> gammas(nh);
  std::vector<std::vector<Ciphertext<word>>> q_cts(nh);
  for (int i = 0; i < nh; i++) {
    gammas[i] = 0.3 / std::max(s_hi[i] - s_lo[i], 1e-6);
    enc_group(2, [&](int, int c) { return gammas[i] * q_host[i][c]; },
              q_cts[i]);
  }
  auto make_mask = [&](int row, int level, double value,
                       Plaintext<word> &pt) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) {
      mv[static_cast<size_t>(b) * kTokens + row] = value;
    }
    std::vector<Complex> msg;
    layout.Pack(msg, mv);
    boot.context->gpu_encoder_.Encode(pt, level, boot.param->GetScale(level),
                                      msg);
  };

  // --- [4] The per-step APPENDS, encrypted: K row T-1 by one masked add
  // per channel; the V token ct by one pack.
  auto t0 = Sync();
  {
    Plaintext<word> mk;
    make_mask(Tc, 3, 1.0, mk);
    std::vector<Plaintext<word>> mp(D);
    for (int d = 0; d < D; d++) make_mask(d, lf, 1.0, mp[d]);
    for (int s = 0; s < nkv; s++) {
      for (int c = 0; c < D; c++) {
        Ciphertext<word> m, r;
        bctx->Mult(m, knew[s][c], mk);
        bctx->Rescale(r, m);
        bctx->Add(kc[s][c], kc[s][c], r);
      }
      Ciphertext<word> acc;
      for (int d = 0; d < D; d++) {
        Ciphertext<word> m;
        bctx->Mult(m, vnew[s][d], mp[d]);
        if (d == 0) {
          acc = std::move(m);
        } else {
          bctx->Add(acc, acc, m);
        }
      }
      bctx->Rescale(vt[s][Tc], acc);
    }
  }
  auto t1 = Sync();

  // The fan-out, compiled once at the e level (shared by every head).
  std::vector<cheddar::Message> row_masks(T);
  for (int j = 0; j < T; j++) {
    std::vector<double> mv(static_cast<size_t>(B) * kTokens, 0.0);
    for (int b = 0; b < B; b++) mv[static_cast<size_t>(b) * kTokens + j] = 1.0;
    layout.Pack(row_masks[j], mv);
  }
  cheddar::CiDecodeUnpack<word> fanout(boot.context, row_masks, B, lf,
                                       boot.param->GetRescalePrimeProd(lf));
  {
    cheddar::EvkRequest req;
    fanout.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  auto t1b = Sync();

  // --- Per head: scores, gamma + max shift, boot, the exp walk, Z, the
  // fan-out, ScoreV, and the reciprocal LAST.
  double worst_bits = -1e30;
  for (int i = 0; i < nh; i++) {
    const int s = kv_slot(heads[i]);
    const double range = s_hi[i] - s_lo[i];
    const double gamma = gammas[i];
    Ciphertext<word> s_up;
    {
      Ciphertext<word> acc;
      for (int c = 0; c < D; c++) {
        Ciphertext<word> m;
        bctx->Mult(m, q_cts[i][c], kc[s][c]);
        if (c == 0) {
          acc = std::move(m);
        } else {
          bctx->Add(acc, acc, m);
        }
      }
      Ciphertext<word> sc;
      bctx->RelinearizeRescale(sc, acc, mult_key);
      const int l0 = boot.param->NPToLevel(sc.GetNP());
      cheddar::Constant<word> shift;
      boot.context->encoder_.EncodeConstant(
          shift, l0, boot.param->GetScale(l0), -gamma * s_hi[i]);
      bctx->Add(sc, sc, shift);
      bctx->Boot(s_up, sc, boot.ui->GetEvkMap());
    }
    // e = exp((s - s_hi) / sqd): a deg-15 polynomial on w / 2^k, then k
    // squarings. The booted message is gamma (s - s_hi).
    Ciphertext<word> e;
    {
      const double rg = range / sqd / (1 << k_sq);  // g in [-rg, 0]
      const double lo = -rg - 0.02 * rg - 1e-9, hi = 0.02 * rg + 1e-9;
      const double a = 0.5 * (hi - lo), b = 0.5 * (hi + lo);
      const double in_factor = gamma * sqd * (1 << k_sq);
      const int l = boot.param->NPToLevel(s_up.GetNP());
      Plaintext<word> pk;
      {
        std::vector<double> kv2(static_cast<size_t>(B) * kTokens,
                                1.0 / (in_factor * a));
        std::vector<Complex> msg;
        layout.Pack(msg, kv2);
        boot.context->gpu_encoder_.Encode(pk, l, boot.param->GetScale(l),
                                          msg);
      }
      Ciphertext<word> v;
      {
        Ciphertext<word> tmp;
        bctx->Mult(tmp, s_up, pk);
        bctx->Rescale(v, tmp);
      }
      const int lvv = boot.param->NPToLevel(v.GetNP());
      {
        cheddar::Constant<word> shift;
        boot.context->encoder_.EncodeConstant(
            shift, lvv, boot.param->GetScale(lvv), -b / a);
        bctx->Add(v, v, shift);
      }
      auto coeffs = cheddar::chebfit::Interpolate(
          [a, b](double t) { return std::exp(a * t + b); }, 7);
      const double in_scale = boot.param->GetScale(lvv);
      const int used =
          cheddar::EvalPoly<word>(coeffs, lvv, in_scale, in_scale, true)
              .GetPolyDegree();
      // Degree 7, never higher: at a requested degree 15 the narrow heads'
      // Chebyshev tails underflow (used 5-6, fine) but the sink head's a of
      // ~0.7 keeps a used degree past 7, where the compiled tree's landing
      // no longer matches Log2Ceil(used + 1) and the evaluation returns
      // 2^400-scale garbage (the 2026-09-04 bisection). The walk's
      // squarings make high degrees unnecessary anyway.
      int lr = lvv;
      for (int d = used + 1; d > 1; d = (d + 1) / 2) lr--;
      cheddar::EvalPoly<word> pexp(coeffs, lvv, in_scale,
                                   boot.param->GetScale(lr), true);
      pexp.Compile(bctx);
      {
        Ciphertext<word> e_raw;
        pexp.Evaluate(bctx, e_raw, v, mult_key);
        bctx->LevelDown(e, e_raw, le);
      }
      for (int j = 0; j < k_sq; j++) {
        Ciphertext<word> m, e2;
        bctx->Mult(m, e, e);
        bctx->RelinearizeRescale(e2, m, mult_key);
        e = std::move(e2);
      }
    }
    // Z = sum_t e (the token ladder), and 1 / Z compiled on its range.
    Ciphertext<word> r_inv;
    {
      Ciphertext<word> z;
      bctx->LevelDown(z, e, boot.param->NPToLevel(e.GetNP()));
      for (int st = B; st < B * kTokens; st <<= 1) {
        Ciphertext<word> r;
        bctx->HRot(r, z, boot.ui->GetRotationKey(st), st);
        bctx->Add(z, z, r);
      }
      double zh = 0.0;
      for (int l = 0; l < T; l++) {
        zh += std::exp((s_host[i][l] - s_hi[i]) / sqd);
      }
      const double lo = zh * 0.9, hi = zh * 1.1;
      const double a = 0.5 * (hi - lo), b = 0.5 * (hi + lo);
      const int l = boot.param->NPToLevel(z.GetNP());
      Plaintext<word> pk;
      {
        std::vector<double> kv2(static_cast<size_t>(B) * kTokens, 1.0 / a);
        std::vector<Complex> msg;
        layout.Pack(msg, kv2);
        boot.context->gpu_encoder_.Encode(pk, l, boot.param->GetScale(l),
                                          msg);
      }
      Ciphertext<word> v;
      {
        Ciphertext<word> tmp;
        bctx->Mult(tmp, z, pk);
        bctx->Rescale(v, tmp);
      }
      const int lvv = boot.param->NPToLevel(v.GetNP());
      {
        cheddar::Constant<word> shift;
        boot.context->encoder_.EncodeConstant(
            shift, lvv, boot.param->GetScale(lvv), -b / a);
        bctx->Add(v, v, shift);
      }
      auto coeffs = cheddar::chebfit::Interpolate(
          [a, b](double t) { return 1.0 / (a * t + b); }, 7);
      const double in_scale = boot.param->GetScale(lvv);
      const int used =
          cheddar::EvalPoly<word>(coeffs, lvv, in_scale, in_scale, true)
              .GetPolyDegree();
      int lr = lvv;
      for (int d = used + 1; d > 1; d = (d + 1) / 2) lr--;
      cheddar::EvalPoly<word> rec(coeffs, lvv, in_scale,
                                  boot.param->GetScale(lr), true);
      rec.Compile(bctx);
      rec.Evaluate(bctx, r_inv, v, mult_key);
    }
    // The fan-out of the UNNORMALIZED e, ScoreV, then the reciprocal.
    std::vector<Ciphertext<word>> e_t;
    fanout.Evaluate(boot.context, e_t, e, boot.ui->GetEvkMap());
    Ciphertext<word> out_raw;
    {
      Ciphertext<word> acc;
      for (int t = 0; t < T; t++) {
        Ciphertext<word> m;
        bctx->Mult(m, e_t[t], vt[s][t]);
        if (t == 0) {
          acc = std::move(m);
        } else {
          bctx->Add(acc, acc, m);
        }
      }
      bctx->RelinearizeRescale(out_raw, acc, mult_key);
    }
    Ciphertext<word> out_ct;
    {
      const int lr = boot.param->NPToLevel(r_inv.GetNP());
      Ciphertext<word> dd;
      bctx->LevelDown(dd, out_raw, lr);
      Ciphertext<word> prod;
      bctx->Mult(prod, dd, r_inv);
      bctx->RelinearizeRescale(out_ct, prod, mult_key);
    }

    double num = 0.0, den = 0.0;
    {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, out_ct);
      std::vector<Complex> m2;
      boot.context->encoder_.Decode(m2, pt);
      std::vector<double> got;
      layout.Unpack(got, m2);
      for (int b = 0; b < B; b += 129) {
        for (int d = 0; d < D; d++) {
          const double g = got[static_cast<size_t>(b) * kTokens + d];
          num += (g - ref[i][d]) * (g - ref[i][d]);
          den += ref[i][d] * ref[i][d];
        }
      }
    }
    const double bits = 0.5 * std::log2(num / den);
    worst_bits = std::max(worst_bits, bits);
    std::cout << "  [decode real] head " << heads[i] << " (kv "
              << heads[i] / 4 << "): rms rel 2^" << bits << ", score range "
              << s_lo[i] << ".." << s_hi[i] << std::endl;
  }
  auto t2 = Sync();
  std::cout << "  [decode real] appends " << Ms(t0, t1) << " ms, " << nh
            << " heads " << Ms(t1b, t2) << " ms (" << Ms(t1b, t2) / nh
            << " ms/head); walk k=" << k_sq << ", e level " << lf
            << ", V level " << lv_v << std::endl;
  EXPECT_LT(worst_bits, -6.0);
#endif
}

// The WHOLE decode step on the real layer-0 weights (Doing.md 7.40 roadmap
// [6]'s main body): `CiDecodeLayer::Step` chains the five gated mechanisms
// -- the decode NormTurn (7.42), the Kang q/k/v with the norm gain, the
// per-head score gammas AND RoPE at the position folded into the weights,
// the encrypted K/V appends (7.45), all 32 heads through the one-boot
// softmax and the hoisted fan-out (7.43), O + residual, the second norm,
// the gate/up -> SiLU -> down walk with the ride gammas in the pack masks
// (7.44), and the second residual -- 1-step e2e against a float64 mirror
// of the same step (norm -> RoPE'd projections -> true softmax(q k /
// sqrt(d)) V over the 128-position cache -> O -> residual -> norm -> FFN
// -> residual). The cache below `position` is the host prefill's (what a
// prefill run leaves behind); the decode token's k/v enter ENCRYPTED
// through the layer's own projections and appends. Instances are
// replicated; both residual streams sit at level 1, so the step chains.
TEST(CiBatch, TheDecodeLayerRunsOnTheRealLayer) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  const char *wdir_env = std::getenv("LLAMA3_ALL_DIR");
  if (wdir_env == nullptr) GTEST_SKIP() << "LLAMA3_ALL_DIR is not set";
  const std::string ld = std::string(wdir_env) + "/L00";
  constexpr int kH = 4096, kKv = 1024, kI = 14336, kHeads = 32, kKvHeads = 8;
  const int D = kTokens, half = D / 2, T = kTokens, Tc = T - 1;
  const double eps = 1e-5, rope_base = 500000.0;
  const double sqd = std::sqrt(static_cast<double>(D));

  std::vector<float> x0, wq, wk, wv, wo, wg, wu, wdn, an, fn;
  ASSERT_TRUE(ReadF32(ld + "/../input_nosink.f32",
                      static_cast<size_t>(T) * kH, x0));
  ASSERT_TRUE(ReadF32(ld + "/wq.f32", static_cast<size_t>(kH) * kH, wq));
  ASSERT_TRUE(ReadF32(ld + "/wk.f32", static_cast<size_t>(kH) * kKv, wk));
  ASSERT_TRUE(ReadF32(ld + "/wv.f32", static_cast<size_t>(kH) * kKv, wv));
  ASSERT_TRUE(ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, wo));
  ASSERT_TRUE(ReadF32(ld + "/wgate.f32", static_cast<size_t>(kH) * kI, wg));
  ASSERT_TRUE(ReadF32(ld + "/wup.f32", static_cast<size_t>(kH) * kI, wu));
  ASSERT_TRUE(ReadF32(ld + "/wdown.f32", static_cast<size_t>(kI) * kH, wdn));
  ASSERT_TRUE(ReadF32(ld + "/attn_norm.f32", kH, an));
  ASSERT_TRUE(ReadF32(ld + "/ffn_norm.f32", kH, fn));

  // ---- The float64 mirror. Norms of every token (the cache reads them),
  // the per-position RoPE'd K/V, the decode token's whole step.
  auto rmsnorm = [&](const double *x, const float *gain,
                     std::vector<double> &out) {
    double ms = 0.0;
    for (int c = 0; c < kH; c++) ms += x[c] * x[c];
    const double r = 1.0 / std::sqrt(ms / kH + eps);
    out.resize(kH);
    for (int c = 0; c < kH; c++) out[c] = x[c] * r * gain[c];
  };
  auto rope = [&](double *m, int pos) {
    for (int c = 0; c < half; c++) {
      const double a = pos * std::pow(rope_base, -2.0 * c / D);
      const double lo = m[c], hi = m[c + half];
      m[c] = lo * std::cos(a) - hi * std::sin(a);
      m[c + half] = hi * std::cos(a) + lo * std::sin(a);
    }
  };
  std::vector<double> y(static_cast<size_t>(T) * kH);
  for (int t = 0; t < T; t++) {
    std::vector<double> xt(kH), yt;
    for (int c = 0; c < kH; c++) xt[c] = x0[static_cast<size_t>(t) * kH + c];
    rmsnorm(xt.data(), an.data(), yt);
    std::copy(yt.begin(), yt.end(), y.begin() + static_cast<size_t>(t) * kH);
  }
  // k_host/v_host[t * kKv + kv * D + d], every position (Tc = the mirror
  // of the encrypted appends).
  std::vector<double> k_host(static_cast<size_t>(T) * kKv),
      v_host(static_cast<size_t>(T) * kKv);
  cheddar::ParallelFor(T, [&](int begin, int end) {
    for (int t = begin; t < end; t++) {
      for (int col = 0; col < kKv; col++) {
        double ak = 0.0, av2 = 0.0;
        for (int c = 0; c < kH; c++) {
          const double yc = y[static_cast<size_t>(t) * kH + c];
          ak += yc * wk[static_cast<size_t>(c) * kKv + col];
          av2 += yc * wv[static_cast<size_t>(c) * kKv + col];
        }
        k_host[static_cast<size_t>(t) * kKv + col] = ak;
        v_host[static_cast<size_t>(t) * kKv + col] = av2;
      }
      for (int kv = 0; kv < kKvHeads; kv++) {
        rope(&k_host[static_cast<size_t>(t) * kKv + kv * D], t);
      }
    }
  });
  std::vector<double> q_host(kH);
  cheddar::ParallelFor(kH, [&](int begin, int end) {
    for (int o = begin; o < end; o++) {
      double a = 0.0;
      for (int c = 0; c < kH; c++) {
        a += y[static_cast<size_t>(Tc) * kH + c] *
             wq[static_cast<size_t>(c) * kH + o];
      }
      q_host[o] = a;
    }
  });
  for (int h = 0; h < kHeads; h++) rope(&q_host[static_cast<size_t>(h) * D], Tc);
  std::vector<double> s_lo(kHeads, 1e30), s_hi(kHeads, -1e30),
      z_host(kHeads), av(kH);
  for (int h = 0; h < kHeads; h++) {
    const int kv = h / (kHeads / kKvHeads);
    std::vector<double> s(T);
    for (int l = 0; l < T; l++) {
      double a = 0.0;
      for (int d = 0; d < D; d++) {
        a += q_host[static_cast<size_t>(h) * D + d] *
             k_host[static_cast<size_t>(l) * kKv + kv * D + d];
      }
      s[l] = a;
      s_lo[h] = std::min(s_lo[h], a);
      s_hi[h] = std::max(s_hi[h], a);
    }
    double z = 0.0;
    std::vector<double> p(T);
    for (int l = 0; l < T; l++) {
      p[l] = std::exp((s[l] - s_hi[h]) / sqd);
      z += p[l];
    }
    z_host[h] = z;
    for (int d = 0; d < D; d++) {
      double a = 0.0;
      for (int l = 0; l < T; l++) {
        a += (p[l] / z) * v_host[static_cast<size_t>(l) * kKv + kv * D + d];
      }
      av[static_cast<size_t>(h) * D + d] = a;
    }
  }
  std::vector<double> mid(kH), out_ref(kH);
  {
    std::vector<double> o_ref(kH);
    cheddar::ParallelFor(kH, [&](int begin, int end) {
      for (int c = begin; c < end; c++) {
        double a = 0.0;
        for (int i = 0; i < kH; i++) {
          a += av[i] * wo[static_cast<size_t>(i) * kH + c];
        }
        o_ref[c] = a;
      }
    });
    for (int c = 0; c < kH; c++) {
      mid[c] = x0[static_cast<size_t>(Tc) * kH + c] + o_ref[c];
    }
  }
  std::vector<double> y2, gg(kI), uu(kI), dd(kH);
  rmsnorm(mid.data(), fn.data(), y2);
  cheddar::ParallelFor(kI, [&](int begin, int end) {
    for (int j = begin; j < end; j++) {
      double ag = 0.0, au = 0.0;
      for (int c = 0; c < kH; c++) {
        ag += y2[c] * wg[static_cast<size_t>(c) * kI + j];
        au += y2[c] * wu[static_cast<size_t>(c) * kI + j];
      }
      gg[j] = ag;
      uu[j] = au;
    }
  });
  std::vector<double> hh(kI);
  for (int j = 0; j < kI; j++) {
    hh[j] = gg[j] / (1.0 + std::exp(-gg[j])) * uu[j];
  }
  cheddar::ParallelFor(kH, [&](int begin, int end) {
    for (int c = begin; c < end; c++) {
      double a = 0.0;
      for (int j = 0; j < kI; j++) {
        a += hh[j] * wdn[static_cast<size_t>(j) * kH + c];
      }
      dd[c] = a;
    }
  });
  for (int c = 0; c < kH; c++) out_ref[c] = mid[c] + dd[c];

  // ---- Calibration from the mirror (replicated instances: single-point
  // norm windows widened by the standard 1.3 ratio).
  cheddar::CiDecodeLayer<word>::Calibration cal;
  {
    double msA = 0.0, msF = 0.0, absmax = 0.0, gmax = 0.0, umax = 0.0;
    for (int c = 0; c < kH; c++) {
      const double xa = x0[static_cast<size_t>(Tc) * kH + c];
      msA += xa * xa;
      msF += mid[c] * mid[c];
      absmax = std::max({absmax, std::abs(xa), std::abs(mid[c]),
                         std::abs(out_ref[c])});
    }
    for (int j = 0; j < kI; j++) {
      gmax = std::max(gmax, std::abs(gg[j]));
      umax = std::max(umax, std::abs(uu[j]));
    }
    cal.attn_alpha = 1.0 / (msA / kH + eps);
    cal.attn_window = 1.3;
    cal.ffn_alpha = 1.0 / (msF / kH + eps);
    cal.ffn_window = 1.3;
    cal.s_lo = s_lo;
    cal.s_hi = s_hi;
    cal.z_lo = z_host;
    cal.z_hi = z_host;
    cal.silu_gmax = gmax;
    cal.up_umax = umax;
    cal.stream_scale = 0.35 / absmax;
  }

  // ---- The ring, the layer, the keys.
  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  cheddar::CiDecodeLayer<word>::Config cfg;
  cfg.verbose = EnvInt("CHEDDAR_DECODE_VERBOSE", 0) != 0;
  cheddar::CiDecodeLayer<word> layer(bctx, cfg);
  const CiBatchLayout &layout = layer.GetLayout();
  const int B = layout.num_instances;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, layout.num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  {
    cheddar::EvkRequest req;
    layer.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  const int k_sq = layer.ExpSquarings(cal);
  const int lf = layer.FanoutLevel(cal);
  std::cout << "  [decode layer] walk k=" << k_sq << ", fanout level " << lf
            << ", stream carries " << cal.stream_scale << ", "
            << FreeMiB() << " MiB free after setup" << std::endl;

  cheddar::CiDecodeLayer<word>::HostWeights hw;
  hw.q = wq.data();
  hw.k = wk.data();
  hw.v = wv.data();
  hw.o = wo.data();
  hw.gate = wg.data();
  hw.up = wu.data();
  hw.down = wdn.data();
  hw.attn_norm.assign(an.begin(), an.end());
  hw.ffn_norm.assign(fn.begin(), fn.end());

  // ---- Encrypt: the stream (the decode token broadcast, in stream_scale
  // units) and the prefill's K/V cache, replicated over the instances.
  auto encrypt_one = [&](const std::vector<double> &values, int level,
                         Ciphertext<word> &ct) {
    std::vector<Complex> msg;
    layout.Pack(msg, values);
    Plaintext<word> pt;
    boot.context->gpu_encoder_.Encode(pt, level,
                                      boot.param->GetScale(level), msg);
    boot.ui->Encrypt(ct, pt);
  };
  std::vector<Ciphertext<word>> stream(kH);
  {
    std::vector<double> values(static_cast<size_t>(B) * T);
    for (int c = 0; c < kH; c++) {
      const double v = cal.stream_scale * x0[static_cast<size_t>(Tc) * kH + c];
      std::fill(values.begin(), values.end(), v);
      encrypt_one(values, layer.StreamLevel(), stream[c]);
    }
  }
  cheddar::CiDecodeLayer<word>::Cache cache;
  cache.kc.resize(kKvHeads);
  cache.vt.resize(kKvHeads);
  {
    std::vector<double> values(static_cast<size_t>(B) * T);
    for (int kv = 0; kv < kKvHeads; kv++) {
      cache.kc[kv].resize(D);
      for (int d = 0; d < D; d++) {
        for (int b = 0; b < B; b++) {
          for (int t = 0; t < T; t++) {
            values[static_cast<size_t>(b) * T + t] =
                t < Tc ? k_host[static_cast<size_t>(t) * kKv + kv * D + d]
                       : 0.0;
          }
        }
        encrypt_one(values, layer.KCacheLevel(), cache.kc[kv][d]);
      }
      cache.vt[kv].resize(T);
      for (int tc = 0; tc < T; tc++) {
        for (int b = 0; b < B; b++) {
          for (int d = 0; d < T; d++) {
            values[static_cast<size_t>(b) * T + d] =
                tc < Tc ? v_host[static_cast<size_t>(tc) * kKv + kv * D + d]
                        : 0.0;
          }
        }
        encrypt_one(values, layer.VCacheLevel(cal), cache.vt[kv][tc]);
      }
    }
  }
  auto t0 = Sync();

  // ---- The step.
  bctx->ResetBootCounts();
  std::vector<Ciphertext<word>> next;
  cheddar::CiDecodeLayer<word>::Debug dbg;
  layer.Step(next, stream, cache, hw, cal, Tc, boot.ui->GetEvkMap(), &dbg);
  auto t1 = Sync();
  const auto counts = bctx->GetBootCounts();
  const auto &st = layer.GetStages();
  std::cout << "  [decode layer] step " << Ms(t0, t1) / 1000.0
            << " s: norm1 " << st.norm1 << " qkv " << st.qkv << " append "
            << st.append << " heads " << st.heads << " o " << st.o
            << " norm2 " << st.norm2 << " gate/up " << st.gate_up << " mid "
            << st.mid << " down " << st.down << " s; boots " << counts.full
            << "; " << FreeMiB() << " MiB free" << std::endl;

  // ---- The comparison, in model units.
  auto compare = [&](const std::vector<Ciphertext<word>> &cts,
                     const std::vector<double> &ref, int stride,
                     const char *tag) {
    double num = 0.0, den = 0.0, worst = 0.0;
    for (int c = 0; c < kH; c += stride) {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, cts[c]);
      std::vector<Complex> m2;
      boot.context->encoder_.Decode(m2, pt);
      std::vector<double> got;
      layout.Unpack(got, m2);
      for (const int slot : {3, static_cast<int>(B / 2) * T + 77}) {
        const double g = got[slot] / cal.stream_scale;
        num += (g - ref[c]) * (g - ref[c]);
        den += ref[c] * ref[c];
        worst = std::max(worst, std::abs(g - ref[c]));
      }
    }
    const double bits = 0.5 * std::log2(num / den);
    std::cout << "  [decode layer] " << tag << ": rms rel 2^" << bits
              << ", worst abs " << worst << std::endl;
    return bits;
  };
  compare(dbg.mid, mid, 7, "mid (attention half)");
  const double bits = compare(next, out_ref, 1, "layer out vs float64");
  EXPECT_EQ(boot.param->NPToLevel(next[0].GetNP()), layer.StreamLevel());
  EXPECT_LT(bits, -4.0);
#endif
}

// The accumulator's half of the channel-ring NormTurn (Doing.md 7.38), in
// isolation: squares summed at level 1 BEFORE any bootstrap, relinearized
// once, and the ONE sum bootstrapped on the deep ring -- against the same
// sum computed the old way (boot first, square after) and against the host.
TEST(CiBatch, TheAccumulatorBootMatchesTheSquares) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const int model = EnvInt("CHEDDAR_CI_BATCH_NORM_MODEL", 32);
  const CiBatchLayout layout(boot.param->MaxNumSlots(), kTokens);
  const int B = layout.num_instances;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    bctx->AddRequiredRotations(req, layout.num_slots);
    boot.ui->PrepareRotationKey(req);
  }
  const auto &mult_key = boot.ui->GetEvkMap().GetMultiplicationKey();

  // Channel values at the layer's riding height (x s ~ +-0.35 max), so the
  // sum's message magnitude is the real layer's per-channel-count.
  std::mt19937_64 gen(0x517);
  std::uniform_real_distribution<double> ux(-1.0, 1.0);
  HostTensor x{B, kTokens, model, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * model);
  for (auto &v : x.v) v = 0.2 * ux(gen);
  std::vector<Ciphertext<word>> stream;
  EncryptChannels(boot, layout, x, 1, stream);

  // The new path sums ride-sized PARTIALS before their boots: a whole sum
  // past EvalMod's fitted ride extrapolates the polynomial (measured
  // 1.2e-3 rms at message 0.43, 7% at 2.8 -- the 2026-09-04 bisection).
  const int chunk = EnvInt("CHEDDAR_CI_BATCH_ACCUM_CHUNK", 4);
  auto sum_of_squares = [&](bool boot_first, Ciphertext<word> &s2) {
    s2 = Ciphertext<word>();
    for (int c0 = 0; c0 < model; c0 += (boot_first ? model : chunk)) {
      const int g = std::min(model - c0, boot_first ? model : chunk);
      Ciphertext<word> acc;
      for (int j = 0; j < g; j++) {
        const int c = c0 + j;
        Ciphertext<word> sq;
        if (boot_first) {
          Ciphertext<word> up;
          bctx->Boot(up, stream[c], boot.ui->GetEvkMap());
          bctx->Mult(sq, up, up);
        } else {
          bctx->Mult(sq, stream[c], stream[c]);
        }
        if (j == 0) {
          acc = std::move(sq);
        } else {
          bctx->Add(acc, acc, sq);
        }
      }
      Ciphertext<word> rel;
      bctx->RelinearizeRescale(rel, acc, mult_key);
      Ciphertext<word> part;
      if (boot_first) {
        part = std::move(rel);
      } else {
        bctx->Boot(part, rel, boot.ui->GetEvkMap());
      }
      if (c0 == 0) {
        s2 = std::move(part);
      } else {
        bctx->Add(s2, s2, part);
      }
    }
  };
  auto decode_sum = [&](const Ciphertext<word> &s2, std::vector<double> &out) {
    Plaintext<word> pt;
    boot.ui->Decrypt(pt, s2);
    std::vector<Complex> got;
    boot.context->encoder_.Decode(got, pt);
    std::vector<double> vals;
    layout.Unpack(vals, got);
    out = std::move(vals);
  };

  Ciphertext<word> s2_new, s2_old;
  sum_of_squares(false, s2_new);
  sum_of_squares(true, s2_old);
  std::cout << "  new path: level "
            << boot.param->NPToLevel(s2_new.GetNP()) << " scale 2^"
            << std::log2(s2_new.GetScale()) << "; old path: level "
            << boot.param->NPToLevel(s2_old.GetNP()) << " scale 2^"
            << std::log2(s2_old.GetScale()) << std::endl;
  std::vector<double> got_new, got_old;
  decode_sum(s2_new, got_new);
  decode_sum(s2_old, got_old);

  double e_new = 0.0, e_old = 0.0, ref = 0.0, worst = 0.0;
  int worst_i = -1;
  for (int b = 0; b < B; b += 97) {
    for (int t = 0; t < kTokens; t++) {
      double s = 0.0;
      for (int c = 0; c < model; c++) {
        const double v = x.At(b, t, c);
        s += v * v;
      }
      const size_t i = static_cast<size_t>(b) * kTokens + t;
      e_new += (got_new[i] - s) * (got_new[i] - s);
      e_old += (got_old[i] - s) * (got_old[i] - s);
      ref += s * s;
      if (std::abs(got_new[i] - s) > worst) {
        worst = std::abs(got_new[i] - s);
        worst_i = static_cast<int>(i);
      }
    }
  }
  std::cout << "  sum of " << model << " squares vs host: new (square, boot) "
            << std::scientific << std::sqrt(e_new / ref)
            << ", old (boot, square) " << std::sqrt(e_old / ref)
            << ", worst new abs " << worst << " at " << worst_i << std::fixed
            << std::endl;
  EXPECT_LT(std::sqrt(e_old / ref), 1e-3);
  EXPECT_LT(std::sqrt(e_new / ref), 1e-3);
#endif
}

TEST(CiBatch, TheNormTurnMatchesTheHost) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  Ring boot(Param());
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const int model = EnvInt("CHEDDAR_CI_BATCH_NORM_MODEL", 256);
  const int keep = 32;
  cheddar::CiBatchLayer<word>::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.model = model;
  cfg.hidden = 512;
  cfg.rows_per_tile = 32;
  cfg.hold_channels = EnvInt("CHEDDAR_CI_BATCH_HOLD_CHANNELS", 0) != 0;
  cfg.verbose = true;
  cheddar::CiBatchLayer<word> layer(bctx, cfg);
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layer.GetLayout().num_slots);
  {
    cheddar::EvkRequest req;
    layer.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  std::unique_ptr<Ring> chan = WireChannelRing(boot, layer);
  const CiBatchLayout &layout = layer.GetLayout();
  const int B = layout.num_instances;

  // Per-token mean squares spread over ~4x, as a calibrated layer's are;
  // the stream carries `stream_scale` and a per-token sink factor is on.
  std::mt19937_64 gen(0x407);
  std::uniform_real_distribution<double> ux(-1.0, 1.0);
  HostTensor x{B, kTokens, model, {}};
  x.v.resize(static_cast<size_t>(B) * kTokens * model);
  std::vector<double> row_scale(kTokens);
  for (int t = 0; t < kTokens; t++) row_scale[t] = 0.05 * (1.0 + (t % 7) / 3.5);
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < kTokens; t++) {
      for (int c = 0; c < model; c++) x.At(b, t, c) = row_scale[t] * ux(gen);
    }
  }
  std::vector<double> sink(kTokens, 1.0);
  sink[0] = 0.5;
  sink[1] = 2.0;
  const double eps = 1e-5;
  // The calibration off the data: alpha at the geometric midpoint of the
  // rescaled mean squares' range, the window their ratio times 1.3^2.
  double lo = 1e300, hi = 0.0;
  for (int t = 0; t < kTokens; t++) {
    const double ms = sink[t] * sink[t] * row_scale[t] * row_scale[t] / 3.0;
    lo = std::min(lo, ms + eps);
    hi = std::max(hi, ms + eps);
  }
  const double alpha = 1.0 / std::sqrt(lo * hi);
  const double window = std::max(1.5, hi / lo * 1.69);
  const double stream_scale = 2.0;
  std::cout << "  alpha " << alpha << " window " << window << std::endl;

  HostTensor xs{B, kTokens, model, {}};
  xs.v = x.v;
  for (auto &v : xs.v) v *= stream_scale;
  std::vector<Ciphertext<word>> stream;
  // The channel-ring path squares the stream before any boot: level 1.
  EncryptChannels(boot, layout, xs, chan ? 1 : 0, stream);

  auto t0 = Sync();
  typename cheddar::CiBatchProjection<word>::Source src;
  layer.NormTurn(src, stream, alpha, window, sink, stream_scale,
                 boot.ui->GetEvkMap(), cfg.hold_channels, cfg.norm_apply_level,
                 /*release_tables=*/true);
  auto t1 = Sync();
  std::cout << "  NormTurn on " << model << " channels: " << Ms(t0, t1) / 1000.0
            << " s" << std::endl;

  // The identity onto the first `keep` channels.
  std::vector<float> eye(static_cast<size_t>(model) * keep, 0.0f);
  for (int c = 0; c < keep; c++) eye[static_cast<size_t>(c) * keep + c] = 1.0f;
  cheddar::DeviceVector<float> eye_dev;
  ToDevice(eye_dev, eye);
  cheddar::CiBatchProjection<word> &proj = layer.GetProjection();
  proj.Prepare("eye", eye_dev.data(), model, keep, src.level);
  std::vector<Ciphertext<word>> y;
  proj.Project(y, src, "eye", 0);
  ASSERT_EQ(static_cast<int>(y.size()), keep);

  std::vector<int> bs = {0, B / 2, B - 1}, ks(keep);
  for (int c = 0; c < keep; c++) ks[c] = c;
  HostTensor got{B, kTokens, keep, {}};
  got.v.assign(static_cast<size_t>(B) * kTokens * keep, 0.0);
  DecryptChannels(boot, layout, y, ks, got);
  HostTensor want{B, kTokens, keep, {}};
  want.v.assign(got.v.size(), 0.0);
  for (int b : bs) {
    for (int t = 0; t < kTokens; t++) {
      double ms = 0.0;
      for (int c = 0; c < model; c++) {
        const double v = sink[t] * x.At(b, t, c);
        ms += v * v;
      }
      const double r = 1.0 / std::sqrt(ms / model + eps);
      for (int c = 0; c < keep; c++) want.At(b, t, c) = sink[t] * x.At(b, t, c) * r;
    }
  }
  const Err e = Compare(got, want, bs, ks);
  std::cout << "  norm vs host: rms rel 2^-" << std::fixed << std::setprecision(2)
            << Bits(e.rms_rel) << " (max abs " << std::scientific << e.max_abs
            << ", ref rms " << e.rms_ref << ")" << std::fixed << std::endl;
  EXPECT_LT(e.rms_rel, std::ldexp(1.0, -10));
#endif
}
