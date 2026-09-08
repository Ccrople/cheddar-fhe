// BERT-Base layer 0's non-leg half on the real checkpoint: the O projection
// with its bias, the residual, the post-attention LayerNorm, the whole GELU
// feed-forward with its two biases, the second residual and the output
// LayerNorm -- against `h_L00.f64`, the float64 reference.
//
// The attention leg is NOT run here. `ci_bert_leg_test` runs it at BERT's
// shape against its own clear reference; what this test needs from it is its
// OUTPUT, which `reference_forward_bert.py` writes as `av_L00.f64`, encrypted
// straight into the layout the seam would have left it in. Splitting the two
// is what makes a failure legible: everything measured here is the half this
// file's class is responsible for.
//
// WHAT IS MEASURED, in order, each against the same float64 forward:
//
//   stage 0   the stream, read back the way `ModDecomp` reads it
//   stage 1   H = LayerNorm(X + O(av) + b_o)          -- AttentionTurn
//   stage 2   Z = LayerNorm(H + W_out(GELU(W_int H + b_int)) + b_out)
//                                                      -- FeedForward
//
// Run it with BERT_ALL_DIR and BERT_REF_DIR pointing at `export_bert.py`'s
// and `reference_forward_bert.py`'s output; it skips without them.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/CiSwitchedCcmm.h"
#include "core/EvkRequest.h"
#include "core/MemoryPool.h"
#include "extension/BootContext.h"
#include "extension/CiBertLayer.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::Plaintext;

namespace {

constexpr const char *kFfnParam = "ci16_35_land13c2e9.json";

// BERT-Base, and the packing: T * rank is the slot count, 768 model channels
// in two dense rank-512 ciphertexts and 3072 hidden ones in six.
constexpr int kT = 128, kH = 768, kI = 3072, kHeads = 12, kD = 64;
constexpr int kRank = 512, kPcmmLevel = 1;
constexpr int kDeclaredH = 1024, kDeclaredI = 3072;
constexpr double kEps = 1e-12;
constexpr double kRide = 0.2;

int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

// The banded recomposition a coefficient image IS, and its inverse -- what
// `ModDecomp` performs and what every read here uses.
std::vector<double> Recompose(const std::vector<std::vector<double>> &comp) {
  std::vector<double> out(static_cast<size_t>(kRank) * kT, 0.0);
  for (int t = 0; t < kT; t++) {
    for (int i = 0; i < kRank; i++) {
      double v = comp[i][t];
      if (i != 0 && t + 1 < kT) v += comp[kRank - i][t + 1];
      out[static_cast<size_t>(t) * kRank + i] = v;
    }
  }
  return out;
}

std::vector<std::vector<double>> Components(const std::vector<double> &co) {
  std::vector<std::vector<double>> comp(kRank, std::vector<double>(kT, 0.0));
  for (int t = 0; t < kT; t++) comp[0][t] = co[static_cast<size_t>(t) * kRank];
  for (int i = 1; i <= kRank / 2; i++) {
    const int mi = kRank - i;
    double ai = 0.0, am = 0.0;
    for (int t = kT - 1; t >= 0; t--) {
      const double ni = co[static_cast<size_t>(t) * kRank + i] - am;
      const double nm = co[static_cast<size_t>(t) * kRank + mi] - ai;
      comp[i][t] = ni;
      comp[mi][t] = nm;
      ai = ni;
      am = nm;
    }
  }
  return comp;
}

bool ReadF32(const std::string &path, size_t n, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<float> raw(n);
  f.read(reinterpret_cast<char *>(raw.data()),
         static_cast<std::streamsize>(n * sizeof(float)));
  if (static_cast<size_t>(f.gcount()) != n * sizeof(float)) return false;
  out.assign(raw.begin(), raw.end());
  return true;
}

bool ReadF64(const std::string &path, size_t n, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.assign(n, 0.0);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(n * sizeof(double)));
  return static_cast<size_t>(f.gcount()) == n * sizeof(double);
}

bool ReadU8(const std::string &path, size_t n, std::vector<unsigned char> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.assign(n, 0);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(n));
  return static_cast<size_t>(f.gcount()) == n;
}

double GeLuHost(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// LayerNorm in double, and the per-token factors that make its argument one.
void LayerNormHost(const std::vector<double> &x, const std::vector<double> &g,
                   const std::vector<double> &b, std::vector<double> &out,
                   std::vector<double> &token_scale) {
  out.assign(x.size(), 0.0);
  token_scale.assign(kT, 1.0);
  for (int t = 0; t < kT; t++) {
    double mu = 0.0;
    for (int c = 0; c < kH; c++) mu += x[static_cast<size_t>(t) * kH + c];
    mu /= kH;
    double var = 0.0;
    for (int c = 0; c < kH; c++) {
      const double d = x[static_cast<size_t>(t) * kH + c] - mu;
      var += d * d;
    }
    var /= kH;
    token_scale[t] = 1.0 / std::sqrt(var);
    const double inv = 1.0 / std::sqrt(var + kEps);
    for (int c = 0; c < kH; c++) {
      out[static_cast<size_t>(t) * kH + c] =
          (x[static_cast<size_t>(t) * kH + c] - mu) * inv * g[c] + b[c];
    }
  }
}

// One stage's report: relative error against the float64 forward, over the
// live channels, with the message's own factor divided out.
void Report(const char *what, const std::vector<double> &got,
            const std::vector<double> &want) {
  double num = 0.0, den = 0.0, worst = 0.0, mx = 0.0, sq = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    num += got[i] * want[i];
    den += want[i] * want[i];
    mx = std::max(mx, std::abs(want[i]));
  }
  const double f = den > 0.0 ? num / den : 1.0;
  for (size_t i = 0; i < want.size(); i++) {
    const double d = got[i] / f - want[i];
    worst = std::max(worst, std::abs(d));
    sq += d * d;
  }
  std::cout << "  [" << what << "] relative " << (worst / mx) << " = 2^"
            << std::log2(worst / mx) << ", rms 2^"
            << 0.5 * std::log2(sq / den) << ", carried " << f << std::endl;
}

double RelBits(const std::vector<double> &got, const std::vector<double> &want) {
  double num = 0.0, den = 0.0, sq = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    num += got[i] * want[i];
    den += want[i] * want[i];
  }
  const double f = den > 0.0 ? num / den : 1.0;
  for (size_t i = 0; i < want.size(); i++) {
    const double d = got[i] / f - want[i];
    sq += d * d;
  }
  return 0.5 * std::log2(sq / den);
}

}  // namespace

TEST(CiBert, TheTurnsRunOnTheRealWeights) {
  const char *wdir_env = std::getenv("BERT_ALL_DIR");
  const char *rdir_env = std::getenv("BERT_REF_DIR");
  if (wdir_env == nullptr || rdir_env == nullptr) {
    GTEST_SKIP() << "BERT_ALL_DIR / BERT_REF_DIR are not set";
  }
  const std::string wdir(wdir_env), rdir(rdir_env), ld = wdir + "/L00";

  // `HalfBootModule` needs the SSE secret sparse in the MODULE basis
  // (Doing.md 3.6); set before any Ring samples one.
  setenv("CHEDDAR_MODULE_SPARSE_SECRET", "128,16", /*overwrite=*/0);

  // ---- the ring ----------------------------------------------------------
  //
  // ONE ring, not the Llama model's four: without the leg there is no chain,
  // no switching ring and no lifted ring, and the layer's own half runs
  // entirely on the landing ladder the FFN uses there -- slack nine, because
  // `SlotToCoeff` is compiled at `GetStCStartLevel()` and an operator eight
  // levels deep cannot reach it without slack.
  const auto t_setup0 = std::chrono::steady_clock::now();
  Ring ring(kFfnParam, /*secret_coeffs=*/{}, /*boot_slack_levels=*/9,
            /*build_user_interface=*/true);
  auto ctx = std::dynamic_pointer_cast<BootContext<word>>(ring.context);
  ASSERT_NE(ctx, nullptr);
  cheddar::UserInterface<word> &ui = *ring.ui;
  const cheddar::EvkMap<word> &evk = ui.GetEvkMap();
  const int num_slots = ring.param->MaxNumSlots();
  ASSERT_EQ(num_slots, kT * kRank);

  ui.PrepareModPackKeys(kT, kPcmmLevel, /*num_aux=*/-1);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < kRank; j++) pack_keys.push_back(&ui.GetModPackKey(kRank, j));
  ctx->PrepareEvalMod();
  ctx->PrepareEvalSpecialFFT(num_slots, cheddar::BootVariant::kNormal, nullptr);
  {
    EvkRequest req;
    ctx->AddRequiredRotations(req, num_slots, /*min_ks=*/false);
    ui.PrepareRotationKey(req);
  }
  // The module basis is this Context's CoeffToSlot; the native CtS tables are
  // never read and the seam's native StC is all it keeps.
  ctx->ReleaseCtS(num_slots);

  const cheddar::CiSwitchedCcmmLayout layout(ring.Degree(), 4096, 32);
  typename cheddar::CiBertLayer<word>::Config cfg;
  cfg.num_tokens = kT;
  cfg.proj_rank = kRank;
  cfg.model_declared = kDeclaredH;
  cfg.model_live = kH;
  cfg.hidden_declared = kDeclaredI;
  cfg.hidden_live = kI;
  cfg.num_heads = kHeads;
  cfg.head_dim = kD;
  cfg.eps = kEps;
  cfg.product_level = kPcmmLevel;
  cfg.verbose = true;
  cheddar::CiBertLayer<word> layer(ctx, layout, pack_keys, cfg);
  {
    EvkRequest req;
    layer.AddRequiredRotations(req);
    ui.PrepareRotationKey(req);
  }
  const auto t_setup1 = std::chrono::steady_clock::now();
  std::cout << "setup " << std::chrono::duration<double>(t_setup1 - t_setup0).count()
            << " s" << std::endl;
  cheddar::MemoryPool::Report("setup done");

  // ---- the clear model ---------------------------------------------------
  std::vector<double> x, av, href, wo, bo, wint, bint, wout, bout;
  std::vector<double> ag, ab, fg, fb;
  ASSERT_TRUE(ReadF32(wdir + "/input.f32", static_cast<size_t>(kT) * kH, x));
  ASSERT_TRUE(ReadF64(rdir + "/av_L00.f64", static_cast<size_t>(kT) * kH, av));
  ASSERT_TRUE(ReadF64(rdir + "/h_L00.f64", static_cast<size_t>(kT) * kH, href));
  ASSERT_TRUE(ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, wo));
  ASSERT_TRUE(ReadF32(ld + "/bo.f32", kH, bo));
  ASSERT_TRUE(ReadF32(ld + "/wint.f32", static_cast<size_t>(kH) * kI, wint));
  ASSERT_TRUE(ReadF32(ld + "/bint.f32", kI, bint));
  ASSERT_TRUE(ReadF32(ld + "/wout.f32", static_cast<size_t>(kI) * kH, wout));
  ASSERT_TRUE(ReadF32(ld + "/bout.f32", kH, bout));
  ASSERT_TRUE(ReadF32(ld + "/attn_norm.f32", kH, ag));
  ASSERT_TRUE(ReadF32(ld + "/attn_norm_bias.f32", kH, ab));
  ASSERT_TRUE(ReadF32(ld + "/ffn_norm.f32", kH, fg));
  ASSERT_TRUE(ReadF32(ld + "/ffn_norm_bias.f32", kH, fb));
  std::vector<unsigned char> gelu_group;
  ASSERT_TRUE(ReadU8(rdir + "/gelu_group_L00.u8",
                     static_cast<size_t>(kT) * kI, gelu_group))
      << "run reference_forward_bert.py; it writes the GELU plan's groups";

  // h_pre = x + av @ wo + b_o, then the norm; and the whole feed-forward.
  std::vector<double> h_pre(static_cast<size_t>(kT) * kH, 0.0);
  for (int t = 0; t < kT; t++) {
    for (int c = 0; c < kH; c++) {
      double acc = bo[c];
      for (int j = 0; j < kH; j++) {
        acc += av[static_cast<size_t>(t) * kH + j] *
               wo[static_cast<size_t>(j) * kH + c];
      }
      h_pre[static_cast<size_t>(t) * kH + c] =
          x[static_cast<size_t>(t) * kH + c] + acc;
    }
  }
  std::vector<double> h, attn_token_scale;
  LayerNormHost(h_pre, ag, ab, h, attn_token_scale);
  std::vector<double> u(static_cast<size_t>(kT) * kI, 0.0);
  for (int t = 0; t < kT; t++) {
    for (int j = 0; j < kI; j++) {
      double acc = bint[j];
      for (int c = 0; c < kH; c++) {
        acc += h[static_cast<size_t>(t) * kH + c] *
               wint[static_cast<size_t>(c) * kI + j];
      }
      u[static_cast<size_t>(t) * kI + j] = acc;
    }
  }
  std::vector<double> z_pre(static_cast<size_t>(kT) * kH, 0.0);
  for (int t = 0; t < kT; t++) {
    for (int c = 0; c < kH; c++) {
      double acc = bout[c];
      for (int j = 0; j < kI; j++) {
        acc += GeLuHost(u[static_cast<size_t>(t) * kI + j]) *
               wout[static_cast<size_t>(j) * kH + c];
      }
      z_pre[static_cast<size_t>(t) * kH + c] =
          h[static_cast<size_t>(t) * kH + c] + acc;
    }
  }
  std::vector<double> z, ffn_token_scale;
  LayerNormHost(z_pre, fg, fb, z, ffn_token_scale);
  {
    // The host forward must be the file's, or nothing below means anything.
    double worst = 0.0, mx = 0.0;
    for (size_t i = 0; i < z.size(); i++) {
      worst = std::max(worst, std::abs(z[i] - href[i]));
      mx = std::max(mx, std::abs(href[i]));
    }
    std::cout << "host forward vs h_L00.f64: " << (worst / mx) << std::endl;
    ASSERT_LT(worst / mx, 1e-5)
        << "this test's own float64 forward does not reproduce the reference";
  }

  // ---- the calibration ---------------------------------------------------
  double resid_absmax = 0.0, u_absmax = 0.0;
  for (double v : h_pre) resid_absmax = std::max(resid_absmax, std::abs(v));
  for (double v : z_pre) resid_absmax = std::max(resid_absmax, std::abs(v));
  for (double v : u) u_absmax = std::max(u_absmax, std::abs(v));
  const double stream_scale = kRide / resid_absmax;
  const double int_scale = kRide / (stream_scale * u_absmax);
  typename cheddar::CiBertLayer<word>::Calibration cal;
  cal.stream_scale = stream_scale;
  // WITH THE PER-TOKEN RESCALE THE WINDOW IS ONE. LayerNorm is exactly scale
  // invariant, so `1/sqrt(var_t)` in front of it puts every token's argument
  // at exactly one -- and the window then only has to cover what the
  // calibration itself misses, which on this prompt is nothing. A served
  // prompt whose variance moved would need the margin instead; that is the
  // open item in `reference/docs/BERT_BASE_B1.md`.
  cal.attn_alpha = 1.0;
  cal.attn_window = 1.5;
  cal.ffn_alpha = 1.0;
  cal.ffn_window = 1.5;
  cal.attn_scale = attn_token_scale;
  cal.ffn_scale = ffn_token_scale;
  cal.o_scale = stream_scale;   // the seam's images carry the model's own units
  cal.int_scale = int_scale;
  cal.out_scale = stream_scale / layer.GetKappa();
  cal.gelu_range = 8.0;
  cal.gelu_degree = 31;
  cal.gelu_group = gelu_group;
  std::cout << "stream_scale " << stream_scale << " (|resid| <= "
            << resid_absmax << "), int_scale " << int_scale << " (|u| <= "
            << u_absmax << "), kappa " << layer.GetKappa() << ", crossing "
            << layer.GetCrossing() << std::endl;

  // ---- the weights, declared -------------------------------------------
  //
  // The seam's map, which is `CiLlamaSeam`'s own: chain ciphertext `bi`,
  // column `col` and lane `lane` land at declared channel
  // `rev4(col) * 32 + rev5(lane)` of image `bi`, and carry the model's
  // attention channel `rev5(lane) * head_dim + bi * rank + col`. Only the
  // images with `bi * 16 < head_dim` carry anything, which for BERT's 64-wide
  // head is four of the layout's eight.
  const int num_images = kD / layout.rank;
  const int attn_declared = num_images * kRank;
  std::vector<int> attn_map(attn_declared, -1);
  for (int bi = 0; bi < num_images; bi++) {
    for (int col = 0; col < layout.rank; col++) {
      for (int lane = 0; lane < layout.lanes; lane++) {
        const int head = Rev(lane, 5);
        if (head >= kHeads) continue;
        const int cc = Rev(col, 4) * 32 + Rev(lane, 5);
        attn_map[bi * kRank + cc] = head * kD + bi * layout.rank + col;
      }
    }
  }
  std::vector<double> wo_dec(static_cast<size_t>(attn_declared) * kDeclaredH,
                             0.0);
  for (int in_d = 0; in_d < attn_declared; in_d++) {
    const int a = attn_map[in_d];
    if (a < 0) continue;
    for (int c = 0; c < kH; c++) {
      wo_dec[static_cast<size_t>(in_d) * kDeclaredH + c] =
          wo[static_cast<size_t>(a) * kH + c];
    }
  }
  std::vector<double> wint_dec(static_cast<size_t>(kDeclaredH) * kDeclaredI,
                               0.0);
  for (int c = 0; c < kH; c++) {
    for (int j = 0; j < kI; j++) {
      wint_dec[static_cast<size_t>(c) * kDeclaredI + j] =
          wint[static_cast<size_t>(c) * kI + j];
    }
  }
  std::vector<double> wout_dec(static_cast<size_t>(kDeclaredI) * kDeclaredH,
                               0.0);
  for (int j = 0; j < kI; j++) {
    for (int c = 0; c < kH; c++) {
      wout_dec[static_cast<size_t>(j) * kDeclaredH + c] =
          wout[static_cast<size_t>(j) * kH + c];
    }
  }
  auto declare = [](const std::vector<double> &v, int declared) {
    std::vector<double> out(declared, 0.0);
    for (size_t i = 0; i < v.size(); i++) out[i] = v[i];
    return out;
  };
  const std::vector<double> bo_dec = declare(bo, kDeclaredH);
  const std::vector<double> bint_dec = declare(bint, kDeclaredI);
  const std::vector<double> bout_dec = declare(bout, kDeclaredH);
  const std::vector<double> ag_dec = declare(ag, kDeclaredH);
  const std::vector<double> ab_dec = declare(ab, kDeclaredH);
  const std::vector<double> fg_dec = declare(fg, kDeclaredH);
  const std::vector<double> fb_dec = declare(fb, kDeclaredH);

  typename cheddar::CiBertLayer<word>::Weights w;
  w.o.host = &wo_dec;
  w.inter.host = &wint_dec;
  w.out.host = &wout_dec;
  w.bo = &bo_dec;
  w.bint = &bint_dec;
  w.bout = &bout_dec;
  w.attn_gain = &ag_dec;
  w.attn_bias = &ab_dec;
  w.ffn_gain = &fg_dec;
  w.ffn_bias = &fb_dec;
  w.tag = "L00";

  // ---- encrypt: the stream, and the attention output in the seam's layout -
  const int op_level = layer.GetStreamLevel();
  auto encrypt_image = [&](const std::vector<std::vector<double>> &comp,
                           int level) {
    const auto co = Recompose(comp);
    Plaintext<word> pt;
    ring.context->encoder_.EncodeCoeff(pt, level, ring.param->GetScale(level),
                                       co);
    Ciphertext<word> ct;
    ui.Encrypt(ct, pt);
    ct.SetNumSlots(num_slots);
    return ct;
  };
  std::vector<Ciphertext<word>> stream(kDeclaredH / kRank);
  for (int k = 0; k < kDeclaredH / kRank; k++) {
    std::vector<std::vector<double>> comp(kRank, std::vector<double>(kT, 0.0));
    for (int c = k * kRank; c < std::min(kH, (k + 1) * kRank); c++) {
      for (int t = 0; t < kT; t++) {
        comp[Rev(c - k * kRank, 9)][t] =
            stream_scale * x[static_cast<size_t>(t) * kH + c];
      }
    }
    stream[k] = encrypt_image(comp, op_level);
  }
  std::vector<Ciphertext<word>> seamed(num_images);
  for (int bi = 0; bi < num_images; bi++) {
    std::vector<std::vector<double>> comp(kRank, std::vector<double>(kT, 0.0));
    for (int cc = 0; cc < kRank; cc++) {
      const int a = attn_map[bi * kRank + cc];
      if (a < 0) continue;
      for (int t = 0; t < kT; t++) {
        comp[Rev(cc, 9)][t] = av[static_cast<size_t>(t) * kH + a];
      }
    }
    seamed[bi] = encrypt_image(comp, kPcmmLevel);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Stage 0: the stream, read back through `Components`.
  {
    std::vector<double> got(static_cast<size_t>(kT) * kH, 0.0);
    for (int k = 0; k < kDeclaredH / kRank; k++) {
      Plaintext<word> pt;
      ui.Decrypt(pt, stream[k]);
      std::vector<double> co;
      ring.context->encoder_.DecodeCoeff(co, pt);
      const auto comp = Components(co);
      for (int c = k * kRank; c < std::min(kH, (k + 1) * kRank); c++) {
        for (int t = 0; t < kT; t++) {
          got[static_cast<size_t>(t) * kH + c] =
              comp[Rev(c - k * kRank, 9)][t];
        }
      }
    }
    Report("stage 0: the stream", got, x);
  }

  // ---- stage 1: O, the residual and the post-attention LayerNorm ---------
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> h_ct;
  layer.AttentionTurn(h_ct, seamed, stream, w, cal, evk);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t1 = std::chrono::steady_clock::now();
  ASSERT_EQ(static_cast<int>(h_ct.size()), kDeclaredH / kRank);
  {
    std::vector<double> got(static_cast<size_t>(kT) * kH, 0.0);
    for (int k = 0; k < kDeclaredH / kRank; k++) {
      Plaintext<word> pt;
      ui.Decrypt(pt, h_ct[k]);
      std::vector<double> co;
      ring.context->encoder_.DecodeCoeff(co, pt);
      const auto comp = Components(co);
      for (int c = k * kRank; c < std::min(kH, (k + 1) * kRank); c++) {
        for (int t = 0; t < kT; t++) {
          got[static_cast<size_t>(t) * kH + c] =
              comp[Rev(c - k * kRank, 9)][t];
        }
      }
    }
    Report("stage 1: H = LayerNorm(X + O + b_o)", got, h);
    EXPECT_LT(RelBits(got, h), -8.0)
        << "the attention turn is worse than 2^-8 against the clear model";
  }

  // ---- stage 2: the feed-forward, its residual and the output norm -------
  const auto t2 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> z_ct;
  layer.FeedForward(z_ct, h_ct, w, cal, evk);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t3 = std::chrono::steady_clock::now();
  {
    std::vector<double> got(static_cast<size_t>(kT) * kH, 0.0);
    for (int k = 0; k < kDeclaredH / kRank; k++) {
      Plaintext<word> pt;
      ui.Decrypt(pt, z_ct[k]);
      std::vector<double> co;
      ring.context->encoder_.DecodeCoeff(co, pt);
      const auto comp = Components(co);
      for (int c = k * kRank; c < std::min(kH, (k + 1) * kRank); c++) {
        for (int t = 0; t < kT; t++) {
          got[static_cast<size_t>(t) * kH + c] =
              comp[Rev(c - k * kRank, 9)][t];
        }
      }
    }
    Report("stage 2: Z = LayerNorm(H + FFN(H))", got, z);
    EXPECT_LT(RelBits(got, z), -7.0)
        << "the layer's non-leg half is worse than 2^-7 against h_L00.f64";
  }
  auto secs = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << "cost: AttentionTurn " << secs(t0, t1) << " s, FeedForward "
            << secs(t2, t3) << " s" << std::endl;
  cheddar::MemoryPool::Report("done");
}
