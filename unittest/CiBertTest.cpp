// BERT-Base layer 0 on the real checkpoint, against `h_L00.f64`.
//
// TWO TESTS, and the split is deliberate.
//
//   TheTurnsRunOnTheRealWeights   the NON-LEG half alone: the O projection
//                                 with its bias, the residual, the
//                                 post-attention LayerNorm, the whole GELU
//                                 feed-forward with its two biases, the second
//                                 residual and the output LayerNorm. The
//                                 attention output is the reference's own
//                                 `av_L00.f64`, encrypted straight into the
//                                 layout the seam leaves -- so a failure here
//                                 is this half's.
//   TheWholeLayerRunsOnTheRealWeights
//                                 the two halves joined: Emit -> the twelve
//                                 HalfBootModules -> the leg -> Boot ->
//                                 the seam -> the two turns. One ring for the
//                                 leg, one for the layer, and the three the
//                                 chain needs.
//
// `ci_bert_leg_test` measures the leg by itself at the same shape. Between
// the three, every stage has a reference that does not depend on the others.
//
// Run with BERT_ALL_DIR and BERT_REF_DIR pointing at `export_bert.py`'s and
// `reference_forward_bert.py`'s output; both skip without them.

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
#include "core/CiLift.h"
#include "core/CiSwitchedCcmm.h"
#include "core/EvkRequest.h"
#include "core/MemoryPool.h"
#include "extension/BootContext.h"
#include "extension/CiBertLayer.h"
#include "extension/CiSinCAttention.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvkRequest;
using cheddar::Plaintext;
using json = nlohmann::json;

namespace {

constexpr const char *kFfnParamDefault = "ci16_35_land13c2e9.json";
// The layer half's ring, overridable: a CERTIFIED calibration has no
// per-token rescale, so its invsqrt windows are the raw variance's (17 to
// 12,000 across the twenty-four norms) and the degree the Chebyshev rate
// asks of those does not fit the landing at 13. `gen_landing.py ci16_35.json
// 17 parameters/ci16_35_land17c2e9.json 2 9` builds the deeper one.
inline const char *FfnParam() {
  const char *e = std::getenv("BERT_FFN_PARAM");
  return (e && *e) ? e : kFfnParamDefault;
}
// The layer half's boot SLACK, overridable. Only StC depends on it
// (`BootContext.h`): more slack puts StC lower, which is one more level for
// the norm and the GELU between the landing and `ToCoeff` -- and, measured
// there, LESS StC memory, not more. Nine is what the Llama FFN uses.
inline int FfnSlack() {
  const char *e = std::getenv("BERT_FFN_SLACK");
  return (e && *e) ? std::atoi(e) : 9;
}
constexpr const char *kBootParam = "ci16_35.json";
constexpr const char *kSwitchParam = "ci_ringswitch16_35_boot.json";
constexpr const char *kSmallParam = "ci12_35_boot.json";
constexpr const char *kLiftedParam = "ringdegree13_35_boot.json";
// The leg's TOWER ring (Doing.md 3.16): three CtS levels and K = 64, its SSE
// secret sampled in the tower basis. Fused, `HalfBootTower` plus the lane
// prefix IS the return, so there are no converters and no native tables on
// the leg ring -- 15 GiB and 262 switching-ring rotation keys that the
// converter route needs and this one does not.
constexpr const char *kTowerParam = "ci16_35_land17c3e10.json";

// BERT-Base, and the packing: T * rank is the slot count, 768 model channels
// in two dense rank-512 ciphertexts and 3072 hidden ones in six.
constexpr int kT = 128, kH = 768, kI = 3072, kHeads = 12, kD = 64;
constexpr int kRank = 512, kPcmmLevel = 1;
constexpr int kDeclaredH = 1024, kDeclaredI = 3072;
// The tied MLM decoder's vocabulary, and it rounded up to a multiple of
// the module rank: the dead tail is exactly zero.
constexpr int kV = 30522, kVDeclared = 30720;
constexpr double kEps = 1e-12;
// The crossings ride at a fifth of EvalMod's range. `BERT_RIDE` is a
// DIAGNOSTIC: raising the public per-channel suppression raises every
// crossing's ride by the same factor, so lowering this one separates "the
// suppression is too deep" from "the ride is now too high" -- the two move
// together otherwise and cannot be told apart from a single sweep.
double Ride() {
  static const double v = [] {
    const char *e = std::getenv("BERT_RIDE");
    return (e != nullptr && *e != 0) ? std::atof(e) : 0.2;
  }();
  return v;
}

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
  f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(n));
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

/**
 * @brief Layer 0's weights, and the float64 forward this file measures
 * against. Both tests read the same thing; the whole-layer one uses more of
 * it.
 */
struct Layer0 {
  std::vector<double> x, av, href, gelu_dummy;
  std::vector<double> wq, wk, wv, bq, bk, bv, wo, bo;
  std::vector<double> wint, bint, wout, bout, ag, ab, fg, fb;
  std::vector<unsigned char> gelu_group;
  // The forward, stage by stage.
  std::vector<double> q, k, v, av_host, h_pre, h, u, z_pre, z;
  std::vector<double> attn_token_scale, ffn_token_scale;
  // What the leg's calibration is read off.
  std::vector<std::vector<double>> row_shift, row_norm;  // [head][row], RAW
  double s_raw_min = 0.0, s_raw_max = 0.0, span_raw = 0.0, m_eff = 0.0;
  double qmax = 0.0, kmax = 0.0, vmax = 0.0;
  double resid_absmax = 0.0, u_absmax = 0.0;
};

std::string Two(int n) {
  return (n < 10 ? "0" : "") + std::to_string(n);
}

// Layer `n`'s weights and the float64 forward of that layer, from its own
// CLEAR input -- which is the reference's `h_L{n-1}`, never the encrypted
// stream. The calibration has to come from the clear model ([SYLPH] 3.1) and
// this is where it does.
bool LoadLayer(const std::string &wdir, const std::string &rdir, int n,
               Layer0 &L) {
  const std::string ld = wdir + "/L" + Two(n);
  const size_t th = static_cast<size_t>(kT) * kH;
  if (n == 0) {
    if (!ReadF32(wdir + "/input.f32", th, L.x)) return false;
  } else if (!ReadF64(rdir + "/h_L" + Two(n - 1) + ".f64", th, L.x)) {
    return false;
  }
  if (!ReadF64(rdir + "/av_L" + Two(n) + ".f64", th, L.av)) return false;
  if (!ReadF64(rdir + "/h_L" + Two(n) + ".f64", th, L.href)) return false;
  if (!ReadF32(ld + "/wq.f32", static_cast<size_t>(kH) * kH, L.wq)) return false;
  if (!ReadF32(ld + "/wk.f32", static_cast<size_t>(kH) * kH, L.wk)) return false;
  if (!ReadF32(ld + "/wv.f32", static_cast<size_t>(kH) * kH, L.wv)) return false;
  if (!ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, L.wo)) return false;
  if (!ReadF32(ld + "/bq.f32", kH, L.bq)) return false;
  if (!ReadF32(ld + "/bk.f32", kH, L.bk)) return false;
  if (!ReadF32(ld + "/bv.f32", kH, L.bv)) return false;
  if (!ReadF32(ld + "/bo.f32", kH, L.bo)) return false;
  if (!ReadF32(ld + "/wint.f32", static_cast<size_t>(kH) * kI, L.wint)) return false;
  if (!ReadF32(ld + "/bint.f32", kI, L.bint)) return false;
  if (!ReadF32(ld + "/wout.f32", static_cast<size_t>(kI) * kH, L.wout)) return false;
  if (!ReadF32(ld + "/bout.f32", kH, L.bout)) return false;
  if (!ReadF32(ld + "/attn_norm.f32", kH, L.ag)) return false;
  if (!ReadF32(ld + "/attn_norm_bias.f32", kH, L.ab)) return false;
  if (!ReadF32(ld + "/ffn_norm.f32", kH, L.fg)) return false;
  if (!ReadF32(ld + "/ffn_norm_bias.f32", kH, L.fb)) return false;
  if (!ReadU8(rdir + "/gelu_group_L" + Two(n) + ".u8",
              static_cast<size_t>(kT) * kI, L.gelu_group)) {
    return false;
  }

  // ---- the attention, from the raw stream (post-norm: no norm in front) ---
  auto project = [&](const std::vector<double> &w, const std::vector<double> &b,
                     std::vector<double> &out) {
    out.assign(th, 0.0);
    for (int t = 0; t < kT; t++) {
      double *ot = &out[static_cast<size_t>(t) * kH];
      for (int o = 0; o < kH; o++) ot[o] = b[o];
      for (int c = 0; c < kH; c++) {
        const double xv = L.x[static_cast<size_t>(t) * kH + c];
        const double *wr = &w[static_cast<size_t>(c) * kH];
        for (int o = 0; o < kH; o++) ot[o] += xv * wr[o];
      }
    }
  };
  project(L.wq, L.bq, L.q);
  project(L.wk, L.bk, L.k);
  project(L.wv, L.bv, L.v);
  for (double a : L.q) L.qmax = std::max(L.qmax, std::abs(a));
  for (double a : L.k) L.kmax = std::max(L.kmax, std::abs(a));
  for (double a : L.v) L.vmax = std::max(L.vmax, std::abs(a));

  L.s_raw_min = 1e300;
  L.s_raw_max = -1e300;
  L.row_shift.assign(kHeads, std::vector<double>(kT, -1e300));
  L.row_norm.assign(kHeads, std::vector<double>(kT, 0.0));
  std::vector<std::vector<std::vector<double>>> S(
      kHeads, std::vector<std::vector<double>>(kT, std::vector<double>(kT)));
  for (int hd = 0; hd < kHeads; hd++) {
    for (int t = 0; t < kT; t++) {
      for (int s = 0; s < kT; s++) {
        double acc = 0.0;
        for (int c = 0; c < kD; c++) {
          acc += L.q[static_cast<size_t>(t) * kH + hd * kD + c] *
                 L.k[static_cast<size_t>(s) * kH + hd * kD + c];
        }
        S[hd][t][s] = acc;
        L.s_raw_min = std::min(L.s_raw_min, acc);
        L.s_raw_max = std::max(L.s_raw_max, acc);
        L.row_shift[hd][t] = std::max(L.row_shift[hd][t], acc);
      }
    }
  }
  L.span_raw = L.s_raw_max - L.s_raw_min;
  L.m_eff = L.span_raw / std::sqrt(static_cast<double>(kD));
  // The live-norm estimate the softmax folds into its mask; bidirectional, so
  // the sum runs over the whole row.
  L.av_host.assign(th, 0.0);
  for (int hd = 0; hd < kHeads; hd++) {
    for (int t = 0; t < kT; t++) {
      double sum = 0.0;
      std::vector<double> p(kT, 0.0);
      for (int s = 0; s < kT; s++) {
        p[s] = std::exp(L.m_eff * (S[hd][t][s] - L.row_shift[hd][t]) /
                        L.span_raw);
        sum += p[s];
      }
      L.row_norm[hd][t] = sum;
      for (int s = 0; s < kT; s++) {
        const double w = p[s] / sum;
        for (int c = 0; c < kD; c++) {
          L.av_host[static_cast<size_t>(t) * kH + hd * kD + c] +=
              w * L.v[static_cast<size_t>(s) * kH + hd * kD + c];
        }
      }
    }
  }

  // ---- the rest of the layer ---------------------------------------------
  L.h_pre.assign(th, 0.0);
  for (int t = 0; t < kT; t++) {
    double *ht = &L.h_pre[static_cast<size_t>(t) * kH];
    for (int c = 0; c < kH; c++) {
      ht[c] = L.x[static_cast<size_t>(t) * kH + c] + L.bo[c];
    }
    for (int j = 0; j < kH; j++) {
      const double av = L.av[static_cast<size_t>(t) * kH + j];
      const double *wr = &L.wo[static_cast<size_t>(j) * kH];
      for (int c = 0; c < kH; c++) ht[c] += av * wr[c];
    }
  }
  LayerNormHost(L.h_pre, L.ag, L.ab, L.h, L.attn_token_scale);
  // The inner loop runs along the WEIGHT's own stride; the transposed order
  // is a 30 s host forward a layer and this is a second of it.
  L.u.assign(static_cast<size_t>(kT) * kI, 0.0);
  for (int t = 0; t < kT; t++) {
    double *ut = &L.u[static_cast<size_t>(t) * kI];
    for (int j = 0; j < kI; j++) ut[j] = L.bint[j];
    for (int c = 0; c < kH; c++) {
      const double hv = L.h[static_cast<size_t>(t) * kH + c];
      const double *wr = &L.wint[static_cast<size_t>(c) * kI];
      for (int j = 0; j < kI; j++) ut[j] += hv * wr[j];
    }
  }
  L.z_pre.assign(th, 0.0);
  {
    std::vector<double> g(kI);
    for (int t = 0; t < kT; t++) {
      double *zt = &L.z_pre[static_cast<size_t>(t) * kH];
      for (int c = 0; c < kH; c++) {
        zt[c] = L.h[static_cast<size_t>(t) * kH + c] + L.bout[c];
      }
      for (int j = 0; j < kI; j++) {
        g[j] = GeLuHost(L.u[static_cast<size_t>(t) * kI + j]);
      }
      for (int j = 0; j < kI; j++) {
        const double gv = g[j];
        const double *wr = &L.wout[static_cast<size_t>(j) * kH];
        for (int c = 0; c < kH; c++) zt[c] += gv * wr[c];
      }
    }
  }
  LayerNormHost(L.z_pre, L.fg, L.fb, L.z, L.ffn_token_scale);
  for (double a : L.h_pre) L.resid_absmax = std::max(L.resid_absmax, std::abs(a));
  for (double a : L.z_pre) L.resid_absmax = std::max(L.resid_absmax, std::abs(a));
  for (double a : L.u) L.u_absmax = std::max(L.u_absmax, std::abs(a));
  return true;
}

// One stage's report: relative error against the float64 forward, with the
// message's own factor fitted out and printed.
double Report(const char *what, const std::vector<double> &got,
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
            << std::log2(worst / mx) << ", rms 2^" << 0.5 * std::log2(sq / den)
            << ", carried " << f << std::endl;
  return f;
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

// A model-channel image, read back through `Components`.
void ReadStream(const Ring &ring, const std::vector<Ciphertext<word>> &cts,
                std::vector<double> &got) {
  got.assign(static_cast<size_t>(kT) * kH, 0.0);
  for (size_t k = 0; k < cts.size(); k++) {
    Plaintext<word> pt;
    ring.ui->Decrypt(pt, cts[k]);
    std::vector<double> co;
    ring.context->encoder_.DecodeCoeff(co, pt);
    const auto comp = Components(co);
    for (int c = static_cast<int>(k) * kRank;
         c < std::min<int>(kH, (static_cast<int>(k) + 1) * kRank); c++) {
      for (int t = 0; t < kT; t++) {
        got[static_cast<size_t>(t) * kH + c] =
            comp[Rev(c - static_cast<int>(k) * kRank, 9)][t];
      }
    }
  }
}

// The weights, declared. The model's own maps are identities on the module
// basis (`ModelSlot(c) = c`, `HiddenSlot(j) = j`); what is not an identity is
// the ATTENTION side, where the seam's layout decides which declared channel
// carries which (head, channel).
std::vector<double> DeclareSquare(const std::vector<double> &w, int in_live,
                                  int in_declared, int out_live,
                                  int out_declared) {
  std::vector<double> out(static_cast<size_t>(in_declared) * out_declared, 0.0);
  for (int i = 0; i < in_live; i++) {
    for (int o = 0; o < out_live; o++) {
      out[static_cast<size_t>(i) * out_declared + o] =
          w[static_cast<size_t>(i) * out_live + o];
    }
  }
  return out;
}

std::vector<double> Declare(const std::vector<double> &v, int declared) {
  std::vector<double> out(declared, 0.0);
  for (size_t i = 0; i < v.size(); i++) out[i] = v[i];
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The non-leg half, on one ring.
// ---------------------------------------------------------------------------
TEST(CiBert, TheTurnsRunOnTheRealWeights) {
  const char *wdir_env = std::getenv("BERT_ALL_DIR");
  const char *rdir_env = std::getenv("BERT_REF_DIR");
  if (wdir_env == nullptr || rdir_env == nullptr) {
    GTEST_SKIP() << "BERT_ALL_DIR / BERT_REF_DIR are not set";
  }
  Layer0 L;
  ASSERT_TRUE(LoadLayer(wdir_env, rdir_env, 0, L))
      << "could not read layer 0 -- run export_bert.py and "
         "reference_forward_bert.py (REF_DUMP_U is not needed)";
  {
    double worst = 0.0, mx = 0.0;
    for (size_t i = 0; i < L.z.size(); i++) {
      worst = std::max(worst, std::abs(L.z[i] - L.href[i]));
      mx = std::max(mx, std::abs(L.href[i]));
    }
    std::cout << "host forward vs h_L00.f64: " << (worst / mx) << std::endl;
    ASSERT_LT(worst / mx, 1e-5)
        << "this test's own float64 forward does not reproduce the reference";
  }

  // `HalfBootModule` needs the SSE secret sparse in the MODULE basis
  // (Doing.md 3.6); set before any Ring samples one.
  setenv("CHEDDAR_MODULE_SPARSE_SECRET", "128,16", /*overwrite=*/0);

  // ONE ring, not the whole-layer test's five: without the leg there is no
  // chain, no switching ring and no lifted ring, and the layer's own half runs
  // entirely on the landing ladder -- slack nine, because `SlotToCoeff` is
  // compiled at `GetStCStartLevel()` and an operator eight levels deep cannot
  // reach it without slack.
  const auto t_setup0 = std::chrono::steady_clock::now();
  Ring ring(FfnParam(), /*secret_coeffs=*/{}, /*boot_slack_levels=*/FfnSlack(),
            /*build_user_interface=*/true);
  auto ctx = std::dynamic_pointer_cast<BootContext<word>>(ring.context);
  ASSERT_NE(ctx, nullptr);
  cheddar::UserInterface<word> &ui = *ring.ui;
  const cheddar::EvkMap<word> &evk = ui.GetEvkMap();
  const int num_slots = ring.param->MaxNumSlots();
  ASSERT_EQ(num_slots, kT * kRank);

  ui.PrepareModPackKeys(kT, kPcmmLevel, /*num_aux=*/-1);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < kRank; j++) {
    pack_keys.push_back(&ui.GetModPackKey(kRank, j));
  }
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
  std::cout << "setup "
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t_setup0).count()
            << " s" << std::endl;

  // ---- the calibration ---------------------------------------------------
  const double stream_scale = Ride() / L.resid_absmax;
  const double int_scale = Ride() / (stream_scale * L.u_absmax);
  typename cheddar::CiBertLayer<word>::Calibration cal;
  cal.stream_scale = stream_scale;
  // WITH THE PER-TOKEN RESCALE THE WINDOW IS ONE. LayerNorm is exactly scale
  // invariant, so `1/sqrt(var_t)` in front of it puts every token's argument
  // at exactly one, and the window then only has to cover what the
  // calibration itself misses -- on this prompt, nothing. A served prompt
  // whose variance moved would need the margin instead; that is the open item
  // in `reference/docs/BERT_BASE_B1.md`.
  cal.attn_alpha = 1.0;
  cal.attn_window = 1.5;
  cal.ffn_alpha = 1.0;
  cal.ffn_window = 1.5;
  cal.attn_scale = L.attn_token_scale;
  cal.ffn_scale = L.ffn_token_scale;
  cal.o_scale = stream_scale;  // the images below carry the model's own units
  cal.int_scale = int_scale;
  cal.out_scale = stream_scale / layer.GetKappa();
  cal.gelu_range = 8.0;
  cal.gelu_degree = 31;
  cal.gelu_group = L.gelu_group;
  std::cout << "stream_scale " << stream_scale << " (|resid| <= "
            << L.resid_absmax << "), int_scale " << int_scale << " (|u| <= "
            << L.u_absmax << "), kappa " << layer.GetKappa() << std::endl;

  // ---- the weights, declared --------------------------------------------
  const int num_images = kD / layout.rank;
  const int attn_declared = num_images * kRank;
  std::vector<int> attn_map(attn_declared, -1);
  for (int bi = 0; bi < num_images; bi++) {
    for (int col = 0; col < layout.rank; col++) {
      for (int lane = 0; lane < layout.lanes; lane++) {
        const int head = Rev(lane, 5);
        if (head >= kHeads) continue;
        attn_map[bi * kRank + Rev(col, 4) * 32 + Rev(lane, 5)] =
            head * kD + bi * layout.rank + col;
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
          L.wo[static_cast<size_t>(a) * kH + c];
    }
  }
  const std::vector<double> wint_dec =
      DeclareSquare(L.wint, kH, kDeclaredH, kI, kDeclaredI);
  const std::vector<double> wout_dec =
      DeclareSquare(L.wout, kI, kDeclaredI, kH, kDeclaredH);
  const std::vector<double> bo_dec = Declare(L.bo, kDeclaredH);
  const std::vector<double> bint_dec = Declare(L.bint, kDeclaredI);
  const std::vector<double> bout_dec = Declare(L.bout, kDeclaredH);
  const std::vector<double> ag_dec = Declare(L.ag, kDeclaredH);
  const std::vector<double> ab_dec = Declare(L.ab, kDeclaredH);
  const std::vector<double> fg_dec = Declare(L.fg, kDeclaredH);
  const std::vector<double> fb_dec = Declare(L.fb, kDeclaredH);

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

  // ---- encrypt the stream, and the attention output in the seam's layout --
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
            stream_scale * L.x[static_cast<size_t>(t) * kH + c];
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
        comp[Rev(cc, 9)][t] = L.av[static_cast<size_t>(t) * kH + a];
      }
    }
    seamed[bi] = encrypt_image(comp, kPcmmLevel);
  }
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  {
    std::vector<double> got;
    ReadStream(ring, stream, got);
    Report("stage 0: the stream", got, L.x);
  }

  // ---- stage 1: O, the residual and the post-attention LayerNorm ---------
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> h_ct;
  layer.AttentionTurn(h_ct, seamed, stream, w, cal, evk);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t1 = std::chrono::steady_clock::now();
  {
    std::vector<double> got;
    ReadStream(ring, h_ct, got);
    Report("stage 1: H = LayerNorm(X + O + b_o)", got, L.h);
    EXPECT_LT(RelBits(got, L.h), -8.0)
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
    std::vector<double> got;
    ReadStream(ring, z_ct, got);
    Report("stage 2: Z = LayerNorm(H + FFN(H))", got, L.z);
    EXPECT_LT(RelBits(got, L.z), -7.0)
        << "the layer's non-leg half is worse than 2^-7 against h_L00.f64";
  }
  auto secs = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << "cost: AttentionTurn " << secs(t0, t1) << " s, FeedForward "
            << secs(t2, t3) << " s" << std::endl;
  cheddar::MemoryPool::Report("done");
}

// ---------------------------------------------------------------------------
// 2. The whole layer: the leg and the two turns, on five rings.
// ---------------------------------------------------------------------------
TEST(CiBert, TheWholeLayerRunsOnTheRealWeights) {
  const char *wdir_env = std::getenv("BERT_ALL_DIR");
  const char *rdir_env = std::getenv("BERT_REF_DIR");
  if (wdir_env == nullptr || rdir_env == nullptr) {
    GTEST_SKIP() << "BERT_ALL_DIR / BERT_REF_DIR are not set";
  }
  // `BERT_LAYERS` layers, chained: layer L reads L-1's ENCRYPTED output. One
  // is the single-layer run; twelve is the model.
  const char *nl = std::getenv("BERT_LAYERS");
  const int num_layers = (nl != nullptr && *nl != 0) ? std::atoi(nl) : 1;
  ASSERT_GE(num_layers, 1);
  ASSERT_LE(num_layers, 12);

  // THE STREAM'S FACTOR IS ONE NUMBER FOR THE WHOLE CHAIN. Each layer's norm
  // puts it back on the way out, so a per-layer factor would have to be told
  // to the NEXT layer as well; the Llama model test sizes it once off every
  // layer's residual maximum and this does the same. The per-layer numbers
  // come from the clear model, as everything else here does.
  // PER LAYER, not one for the chain. Layer L's factor has to keep both
  // crossings it is read by inside EvalMod's range: the residual L writes
  // (`z_pre`) and the one L+1's attention writes on top of it (`h_pre`).
  // A certified plan (`reference/scripts/bert_plan.py`) has NOTHING per token
  // and nothing per prompt in it: the GELU's bands come from the weights
  // alone and both norms state a window with a margin. Everything the served
  // prompt would otherwise contribute is switched off together, here.
  // THE PUBLIC PER-CHANNEL SUPPRESSION, read once. The model-channel
  // stream carries it end to end, so it is ONE vector for the whole
  // chain and not one a layer: the residual add is `X + O` and both
  // sides must carry the same factor.
  std::vector<double> chan_sup;
  if (const char *cdir0 = std::getenv("BERT_CALIB_DIR")) {
    std::ifstream cf0(std::string(cdir0) + "/corpus.json");
    if (cf0.good()) {
      json p0;
      cf0 >> p0;
      if (p0.contains("channel_suppress")) {
        chan_sup = p0["channel_suppress"].get<std::vector<double>>();
        chan_sup.resize(kDeclaredH, 1.0);
        // `BERT_CHAN_CAP` floors the suppression. The plan's smallest factor
        // is 1/238, and a consumer of the stream divides its weights by it --
        // so sixteen rows of every q/k/v and up projection grow by that
        // factor, which the int8 product has to hold, and the crossing's
        // plaintext grows by it too. A cap trades ride for both.
        if (const char *cp = std::getenv("BERT_CHAN_CAP")) {
          const double floor_v = 1.0 / std::atof(cp);
          for (double &v : chan_sup) v = std::max(v, floor_v);
        }
        double lo = 1.0;
        for (double v : chan_sup) lo = std::min(lo, v);
        std::cout << "PUBLIC channel suppression: " << chan_sup.size()
                  << " channels, smallest factor " << lo << std::endl;
      }
    }
  }
  const bool CS = !chan_sup.empty();
  auto unsup = [&](std::vector<double> &v) {
    if (!CS) return;
    for (int t = 0; t < kT; t++)
      for (int c = 0; c < kH; c++)
        v[static_cast<size_t>(t) * kH + c] /= chan_sup[c];
  };

  const bool certified = std::getenv("BERT_CALIB_DIR") != nullptr;
  const bool gelu_only_env = std::getenv("BERT_CERT_GELU_ONLY") != nullptr;
  std::vector<double> attn_resid(num_layers, 0.0), ffn_resid(num_layers, 0.0);
  std::vector<std::vector<double>> suppress(num_layers);
  for (int n = 0; n < num_layers; n++) {
    Layer0 probe;
    ASSERT_TRUE(LoadLayer(wdir_env, rdir_env, n, probe))
        << "could not read layer " << n;
    // THE RIDE IS SIZED ON WHAT ACTUALLY CROSSES. The stream carries the
    // public per-channel factor `d_c` end to end, so the message the
    // crossing sees is `s * d_c * resid`, and sizing `s` off the RAW
    // maximum would put the message a factor `max|d_c resid| / max|resid|`
    // BELOW `Ride()` -- which is the whole point of the suppression, spent
    // backwards. Measured: with the plan's sixteen channels that factor is
    // 1/88 at layer 10, so the un-suppressed sizing rides 6.5 bits low and
    // the suppression reads as no gain at all.
    for (int t = 0; t < kT; t++) {
      for (int c = 0; c < kH; c++) {
        const double d = CS ? chan_sup[c] : 1.0;
        attn_resid[n] = std::max(
            attn_resid[n],
            std::abs(probe.h_pre[static_cast<size_t>(t) * kH + c]) * d);
      }
    }
    // THE FEED-FORWARD'S RESIDUAL ROWS. A handful of them are a hundred times
    // the rest -- layer 10 reaches 1001.5 at token 48 against a row-maximum
    // median of 9.5 -- and the crossing is peak limited, so they cost every
    // other row its ride. The public per-token factor brings each row to the
    // median and the output LayerNorm takes it straight back out.
    std::vector<double> row(kT, 0.0);
    for (int t = 0; t < kT; t++) {
      for (int c = 0; c < kH; c++) {
        row[t] = std::max(
            row[t], std::abs(probe.z_pre[static_cast<size_t>(t) * kH + c]) *
                        (CS ? chan_sup[c] : 1.0));
      }
    }
    std::vector<double> sorted = row;
    std::sort(sorted.begin(), sorted.end());
    const double target = sorted[kT / 2];
    suppress[n].assign(kT, 1.0);
    for (int t = 0; t < kT; t++) {
      // THE SUPPRESSION IS PER PROMPT -- it is read off THIS prompt's
      // `z_pre` -- so a CERTIFIED plan cannot use it. Without it the ride is
      // set by the true row maximum (1001.5 at layer 10 against a median of
      // 9.5) instead of the median, and the bits that costs the stream are
      // exactly the price of prompt independence. It is the one number the
      // host study could not predict, so it is printed below.
      if ((!certified || gelu_only_env) && row[t] > target) {
        suppress[n][t] = target / row[t];
      }
      ffn_resid[n] = std::max(ffn_resid[n], row[t] * suppress[n][t]);
    }
    if (n == 0) {
      // The attention this test's own forward computes must be the
      // reference's, or the leg is being measured against the wrong thing.
      double worst = 0.0, mx = 0.0;
      for (size_t i = 0; i < probe.av.size(); i++) {
        worst = std::max(worst, std::abs(probe.av_host[i] - probe.av[i]));
        mx = std::max(mx, std::abs(probe.av[i]));
      }
      std::cout << "host attention vs av_L00.f64: " << (worst / mx)
                << std::endl;
      ASSERT_LT(worst / mx, 1e-6);
    }
  }
  std::vector<double> s_in(num_layers, 0.0), s_out(num_layers, 0.0);
  for (int n = 0; n < num_layers; n++) {
    double reach = ffn_resid[n];
    if (n + 1 < num_layers) reach = std::max(reach, attn_resid[n + 1]);
    s_out[n] = Ride() / reach;
  }
  s_in[0] = Ride() / attn_resid[0];
  for (int n = 1; n < num_layers; n++) s_in[n] = s_out[n - 1];
  std::cout << num_layers << " layer(s); residuals";
  for (int n = 0; n < num_layers; n++) {
    std::cout << " " << std::max(attn_resid[n], ffn_resid[n]);
  }
  std::cout << std::endl;
  setenv("CHEDDAR_MODULE_SPARSE_SECRET", "128,16", /*overwrite=*/0);

  // ---- five rings, one secret -------------------------------------------
  //
  // The leg on `ci16_35` at slack ZERO (its softmax walk needs `GetEndLevel()`
  // at 16), the layer on the K = 32 landing ladder at slack NINE, and the
  // three the chain switches through. A ciphertext crosses between the first
  // two only at a SHARED level, which is what `GetSeamInputLevel()` is for.
  const auto t_setup0 = std::chrono::steady_clock::now();
  Ring boot(kBootParam);
  Ring swtch(kSwitchParam, boot.ui->GetSecretCoeffs());
  Ring small(kSmallParam);
  Ring lifted(kLiftedParam, cheddar::CiLiftHandler<word>::LiftSecret(
                                small.ui->GetSecretCoeffs()));
  Ring ffn(FfnParam(), boot.ui->GetSecretCoeffs(), /*boot_slack_levels=*/FfnSlack(),
           /*build_user_interface=*/true);
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  auto fctx = std::dynamic_pointer_cast<BootContext<word>>(ffn.context);
  ASSERT_NE(bctx, nullptr);
  ASSERT_NE(fctx, nullptr);
  cheddar::UserInterface<word> &fui = *ffn.ui;
  const cheddar::EvkMap<word> &fevk = fui.GetEvkMap();
  const cheddar::EvkMap<word> &bevk = boot.ui->GetEvkMap();
  const int num_slots = boot.param->MaxNumSlots();
  const int chain_level = 2;

  // FUSED: ci16_35's native tables serve nothing. The scores and the
  // attention output return through the TOWER ring, q/k/v and the stream
  // cross through `HalfBootModule` (no native table), and the seam runs on
  // the FFN ring -- so `PrepareEvalSpecialFFT` here would be ~6 GiB and a
  // rotation-key set for nobody.
  bctx->PrepareEvalMod();
  fui.PrepareModPackKeys(kT, kPcmmLevel, /*num_aux=*/-1);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < kRank; j++) {
    pack_keys.push_back(&fui.GetModPackKey(kRank, j));
  }
  fctx->PrepareEvalMod();
  fctx->PrepareEvalSpecialFFT(num_slots, cheddar::BootVariant::kNormal, nullptr);
  {
    EvkRequest req;
    fctx->AddRequiredRotations(req, num_slots, /*min_ks=*/false);
    fui.PrepareRotationKey(req);
  }
  fctx->ReleaseCtS(num_slots);
  // The two rings share their bottom primes, so they share the crossing
  // constant; the leg's `restore` and the layer's own bookkeeping both rest
  // on that.
  ASSERT_NEAR(bctx->GetMessageRatio(), fctx->GetMessageRatio(),
              1e-12 * std::abs(bctx->GetMessageRatio()));

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
  const cheddar::CiSwitchedCcmmLayout layout(boot.Degree(), small.Degree(), 32);
  cheddar::CiBertLayer<word> layer(fctx, layout, pack_keys, cfg);
  {
    EvkRequest req;
    layer.AddRequiredRotations(req);
    fui.PrepareRotationKey(req);
  }

  // ---- the leg's tower ring ----------------------------------------------
  //
  // Its SSE secret is sampled in the TOWER basis, because the tower-centred
  // ModRaise's wrap-around is bounded there and nowhere else (Doing.md 3.16:
  // max 32 / std 5 at h = 16, against 780 / 270 for a native or module-sparse
  // one). Every other ring keeps the module setting, so the environment is
  // saved and put back around this one construction.
  std::unique_ptr<Ring> tower;
  {
    // `BERT_TOWER_SPARSE` is the tower weight as a KNOB, because it is a
    // SECURITY parameter and sixteen is not enough. `security_estimate.py`
    // prices the sparse secret the attacker actually sees -- the NATIVE
    // coefficients, which the tower map spreads a nonzero over up to four of
    // -- against the drop-columns hybrid at this ring's logQP of 1771:
    //
    //     tower h=16 -> 57 native -> 121.1 bits   (below the 128 the MAIN
    //                                              secret has at the same QP)
    //     tower h=24 -> 91 native -> 127.8 bits
    //     tower h=32 -> 122 native -> 128.1 bits  = the primal ceiling
    //
    // and `ci_nested_sinc.py`'s wrap-around says it is free in levels: the
    // tower-centred ModRaise reads max 32.1 at h = 16, 42.5 at 24 and 47.1 at
    // 32, all inside the K = 64 this ring already pays for (48 reaches 51.7,
    // which is the first weight that looks tight).
    const char *prev = std::getenv("CHEDDAR_MODULE_SPARSE_SECRET");
    const std::string saved = prev ? prev : "";
    const char *tower_env = std::getenv("BERT_TOWER_SPARSE");
    setenv("CHEDDAR_MODULE_SPARSE_SECRET",
           (tower_env && *tower_env) ? tower_env : "4096:128,16",
           /*overwrite=*/1);
    tower = std::make_unique<Ring>(kTowerParam, boot.ui->GetSecretCoeffs(),
                                   /*boot_slack_levels=*/0);
    if (prev) {
      setenv("CHEDDAR_MODULE_SPARSE_SECRET", saved.c_str(), 1);
    } else {
      unsetenv("CHEDDAR_MODULE_SPARSE_SECRET");
    }
  }
  auto lctx = std::dynamic_pointer_cast<BootContext<word>>(tower->context);
  ASSERT_NE(lctx, nullptr);
  lctx->PrepareEvalMod();  // and no native tables: the tower runs none
  ASSERT_NEAR(lctx->GetMessageRatio(), bctx->GetMessageRatio(),
              1e-9 * std::abs(bctx->GetMessageRatio()));

  // ---- the leg ------------------------------------------------------------
  typename cheddar::CiSinCAttention<word>::Config acfg;
  acfg.dense_images = true;
  acfg.num_heads = kHeads;
  acfg.head_dim = kD;
  acfg.rope = false;  // BERT's positions are learned and already in the input
  acfg.restore = 1.0 / bctx->GetMessageRatio();
  acfg.land_level = fctx->GetBootParameter().GetEvalModEndLevel();
  acfg.landing_scale = fctx->GetStCInputScale();
  // Fused: the forwards spend two levels, so the chain runs at 1 and the
  // return is the tower's HalfBoot plus the prefix rather than an inverse
  // converter.
  acfg.fused = true;
  acfg.chain_level = 1;
  acfg.inverse_level = 0;
  cheddar::CiSinCAttention<word> attn(bctx, swtch.context, small.context,
                                      lifted.context, acfg, lctx);
  ASSERT_EQ(attn.GetNumImages(), kD / layout.rank);
  swtch.ui->PrepareRingSwitchKey(small.Degree(), small.ui->GetSecretCoeffs(),
                                 chain_level);
  swtch.ui->PrepareInverseRingSwitchKey(small.Degree(),
                                        small.ui->GetSecretCoeffs(),
                                        chain_level);
  for (int idx : attn.LiftedRotationIndices()) {
    lifted.ui->PrepareRotationKey(idx, chain_level);
  }
  {
    EvkRequest req;
    attn.AddSwitchRotations(req);
    swtch.ui->PrepareRotationKey(req);
  }
  {
    EvkRequest req;
    attn.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  {
    EvkRequest req;
    attn.AddTowerRotations(req);
    tower->ui->PrepareRotationKey(req);
  }
  typename cheddar::CiSinCAttention<word>::Keys keys;
  keys.boot = &bevk;
  keys.swtch = &swtch.ui->GetEvkMap();
  keys.lifted = &lifted.ui->GetEvkMap();
  keys.tower = &tower->ui->GetEvkMap();
  keys.ring_switch = &swtch.ui->GetRingSwitchKey(layout.rank);
  keys.inverse_ring_switch = &swtch.ui->GetInverseRingSwitchKey(layout.rank);
  layer.Base().PrepareSeamHalf(0);
  {
    EvkRequest req;
    layer.Base().AddSeamHalfRotations(req);
    fui.PrepareRotationKey(req);
  }
  std::cout << "setup "
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t_setup0).count()
            << " s" << std::endl;
  cheddar::MemoryPool::Report("setup done");

  // ---- the stream, encrypted once; every layer after the first reads the
  //      one before it ------------------------------------------------------

  const int op_level = layer.GetStreamLevel();
  std::vector<Ciphertext<word>> stream(kDeclaredH / kRank);
  {
    Layer0 first;
    ASSERT_TRUE(LoadLayer(wdir_env, rdir_env, 0, first));
    for (int k = 0; k < kDeclaredH / kRank; k++) {
      std::vector<std::vector<double>> comp(kRank,
                                            std::vector<double>(kT, 0.0));
      for (int c = k * kRank; c < std::min(kH, (k + 1) * kRank); c++) {
        for (int t = 0; t < kT; t++) {
          comp[Rev(c - k * kRank, 9)][t] =
              s_in[0] * first.x[static_cast<size_t>(t) * kH + c] *
              (chan_sup.empty() ? 1.0 : chan_sup[c]);
        }
      }
      const auto co = Recompose(comp);
      Plaintext<word> pt;
      ffn.context->encoder_.EncodeCoeff(pt, op_level,
                                        ffn.param->GetScale(op_level), co);
      boot.ui->Encrypt(stream[k], pt);
      stream[k].SetNumSlots(num_slots);
    }
  }

  double worst_layer = -1e300;
  for (int LAYER = 0; LAYER < num_layers; LAYER++) {
  Layer0 L;
  ASSERT_TRUE(LoadLayer(wdir_env, rdir_env, LAYER, L));
  const std::string tag = "L" + Two(LAYER);
  std::cout << "==== layer " << LAYER << " ====" << std::endl;
  // ---- the calibration ---------------------------------------------------
  //
  // The q/k/v sizing is the Llama model test's, on BERT's own maxima: the
  // HalfBoot image bound first, then the cap on the chain-unit score message.
  const double img_max = 0.45;
  double cq = img_max / L.qmax, ck = img_max / L.kmax;
  const double s_abs = std::max(std::abs(L.s_raw_max), std::abs(L.s_raw_min));
  const double prod_cap = 0.36 / s_abs;
  if (cq * ck > prod_cap) {
    const double sh = std::sqrt(prod_cap / (cq * ck));
    cq *= sh;
    ck *= sh;
  }
  const double cv = std::min(1.0, img_max / L.vmax);
  const double cqk = cq * ck;
  const double stream_scale = s_in[LAYER];
  const double stream_out = s_out[LAYER];
  // The intermediate crossing reads what the attention turn WROTE.
  const double int_scale = Ride() / (stream_out * L.u_absmax);
  typename cheddar::CiBertLayer<word>::Calibration cal;
  cal.stream_scale = stream_scale;
  // The PUBLIC per-channel suppression the weights above were folded
  // with. Without this the library never takes it back out at the
  // crossing, and the norm reads a stream that still carries `d_c` --
  // measured: the sixteen suppressed channels are most of the row's
  // energy, so the LayerNorm comes back 24 % wrong at a cap of two.
  cal.channel_suppress = chan_sup;
  cal.stream_out = stream_out;
  cal.row_suppress = suppress[LAYER];
  cal.attn_alpha = 1.0;
  cal.attn_window = 1.5;
  cal.ffn_alpha = 1.0;
  cal.ffn_window = 1.5;
  cal.attn_scale = L.attn_token_scale;
  // THE SUPPRESSION IS PART OF THIS NORM'S ARGUMENT. Its input carries the
  // per-token factor, so its variance carries the SQUARE of it, and the
  // rescale that puts the invsqrt's argument at one has to carry the factor
  // itself -- left out, a row suppressed by 1/100 arrives at 1e-4 where the
  // window is [1/sqrt 1.5, sqrt 1.5] and the polynomial is evaluated where it
  // was never fitted. Both are public per-token factors applied at the same
  // multiply, so they simply multiply.
  cal.ffn_scale = L.ffn_token_scale;
  for (int t = 0; t < kT; t++) cal.ffn_scale[t] /= suppress[LAYER][t];
  // `BERT_LN_CRUDE=<degree>` runs the inverse square root in TWO stages with
  // a crossing between them (`LayerNorm.h`'s Mode comment). On the served
  // prompt's own window of 1.5 the split has nothing to buy -- the point of
  // running it here is that it must be a NO-OP in accuracy, which is what
  // says the plumbing is right before a wide window is asked of it.
  // `BERT_LN_REFINE_WINDOW` is the second stage's window; the default of 4
  // covers a crude relative error up to about a third.
  if (const char *cd = std::getenv("BERT_LN_CRUDE")) {
    const int d = std::atoi(cd);
    const char *rw = std::getenv("BERT_LN_REFINE_WINDOW");
    const double w = (rw && *rw) ? std::atof(rw) : 4.0;
    cal.attn_crude_degree = d;
    cal.ffn_crude_degree = d;
    cal.attn_refine_window = w;
    cal.ffn_refine_window = w;
    std::cout << "TWO-STAGE invsqrt: crude degree " << d
              << ", refine window " << w << std::endl;
  }
  cal.q_scale = cq / stream_scale;
  cal.k_scale = ck / stream_scale;
  cal.v_scale = cv / stream_scale;
  cal.int_scale = int_scale;
  cal.out_scale = stream_out / layer.GetKappa();
  cal.gelu_range = 8.0;
  cal.gelu_degree = 31;
  cal.gelu_group = L.gelu_group;
  // `BERT_CALIB_DIR`: the GELU plan from a CORPUS rather than from the served
  // prompt. `bert_corpus.py` writes it -- a slot is bulk only if its whole
  // corpus range fits the fitted interval, anything past it is answered by
  // `u` or by nothing where its sign is stable, and by a wide fit where it is
  // not. Without this the plan is the served prompt's own, which is not a
  // calibration at all.
  if (const char *cdir = std::getenv("BERT_CALIB_DIR")) {
    std::ifstream cf(std::string(cdir) + "/corpus.json");
    ASSERT_TRUE(cf.good()) << "no corpus.json in BERT_CALIB_DIR";
    json cj = json::parse(cf);
    const auto &cl = cj["layers"][LAYER];
    cal.gelu_range = cl["gelu_range"].get<double>();
    cal.gelu_degree = static_cast<int>(cl["gelu_degree"].get<double>());
    cal.gelu_wide_range = cl["gelu_wide_range"].get<double>();
    cal.gelu_wide_degree =
        static_cast<int>(cl["gelu_wide_degree"].get<double>());
    ASSERT_TRUE(ReadU8(std::string(cdir) + "/gelu_group_L" + Two(LAYER) +
                           ".u8",
                       static_cast<size_t>(kT) * kI, cal.gelu_group))
        << "no corpus GELU groups for layer " << LAYER;
    if (cl.contains("gelu_bands")) {
      // ---- THE CERTIFIED PLAN ---------------------------------------------
      //
      // Every band is a fit over an interval `u` cannot leave: with
      // `a_j = gain * W[:,j]` and `c_j = bias . W[:,j] + b_j`, a LayerNorm
      // output lies exactly on the sphere of radius sqrt(768), so
      // `|u_j - c_j| <= sqrt(768) ||a_j - mean(a_j)||` for EVERY input.
      // Checked over 28 prompts x 128 tokens x 3072 channels x 12 layers:
      // never violated, maximum occupancy 0.9527. So there is no identity
      // group and no zero group -- those two were the per-prompt part, and
      // the only ones whose failure is a wrong answer rather than a bounded
      // one (`reference/docs/BERT_BASE_B1.md` 11.1).
      cal.gelu_bands.clear();
      for (const auto &bj : cl["gelu_bands"]) {
        typename cheddar::CiBertLayer<word>::Calibration::GeLuBand band;
        band.range = bj["range"].get<double>();
        band.degree = static_cast<int>(bj["degree"].get<double>());
        band.coeffs = bj["coeffs"].get<std::vector<double>>();
        ASSERT_GT(band.range, 0.0) << "band " << cal.gelu_bands.size()
                                   << " of layer " << LAYER << " states no "
                                      "range; the mask divides by it";
        cal.gelu_bands.push_back(band);
      }
      // ---- the MODE plan, when the file carries one ---------------------
      //
      // A band gives every channel in it the band's WIDEST interval; a mode
      // plan gives each channel its OWN, at R evaluations rather than 3072,
      // because the per-channel coefficient matrix is numerically low rank.
      // Measured at degree 127 on layer 0: bands 1.5e-02, per channel
      // 4.3e-04 (`reference/docs/BERT_BASE_B1.md` 12.6).
      if (cl.contains("gelu_modes") && !cl["gelu_modes"].empty()) {
        // The modes come out of an SVD, so the FIRST R of them are the best
        // rank-R plan there is: `BERT_GELU_MODES` sweeps the cost/accuracy
        // trade without regenerating anything.
        size_t keep = cl["gelu_modes"].size();
        if (const char *k = std::getenv("BERT_GELU_MODES")) {
          keep = std::min<size_t>(keep, std::max(1, std::atoi(k)));
        }
        cal.gelu_modes.clear();
        for (const auto &mj : cl["gelu_modes"]) {
          if (cal.gelu_modes.size() >= keep) break;
          typename cheddar::CiBertLayer<word>::Calibration::GeLuMode m;
          m.coeffs = mj["coeffs"].get<std::vector<double>>();
          m.weight = mj["weight"].get<std::vector<double>>();
          ASSERT_EQ(static_cast<int>(m.weight.size()), kI)
              << "a mode's weight must cover every hidden channel";
          cal.gelu_modes.push_back(m);
        }
        cal.gelu_rad = cl["gelu_rad"].get<std::vector<double>>();
        cal.gelu_centre = cl["gelu_centre"].get<std::vector<double>>();
        cal.gelu_bands.clear();
        std::cout << "MODE plan: " << cal.gelu_modes.size()
                  << " modes of degree " << (cal.gelu_modes[0].coeffs.size() - 1)
                  << ", each channel on its OWN certified interval"
                  << std::endl;
      }
      // The two norms: a window with a MARGIN (the variance has no certified
      // interval), the degree the Chebyshev rate for `x^-1/2` asks of that
      // window, and a vector fitted for RELATIVE error.
      cal.attn_alpha = cl["attn_norm"]["alpha"].get<double>();
      cal.attn_window = cl["attn_norm"]["window"].get<double>();
      cal.attn_degree =
          static_cast<int>(cl["attn_norm"]["degree"].get<double>());
      cal.attn_invsqrt =
          cl["attn_norm"]["coeffs"].get<std::vector<double>>();
      cal.ffn_alpha = cl["ffn_norm"]["alpha"].get<double>();
      cal.ffn_window = cl["ffn_norm"]["window"].get<double>();
      cal.ffn_degree =
          static_cast<int>(cl["ffn_norm"]["degree"].get<double>());
      cal.ffn_invsqrt = cl["ffn_norm"]["coeffs"].get<std::vector<double>>();
      // THE TWO-STAGE INVERSE SQUARE ROOT, when the plan carries one
      // (`LN_BUDGET=<levels a span>` in `bert_plan.py`). A norm whose window
      // no affordable degree covers in ONE span gets a crude stage on the
      // same alpha and window, a crossing, and a refine stage whose window is
      // `((1+eps)/(1-eps))^2` for the crude stage's worst relative error --
      // a bound over its interval, not a statistic over prompts. Without it
      // the feed-forward norms of layers 9 and 10 want degree 511 and no
      // landing affords that; with it they want 31 and 63.
      auto stage = [](const nlohmann::json &j,
                      typename cheddar::CiBertLayer<word>::Calibration &c,
                      bool ffn) {
        if (!j.contains("crude_degree")) return;
        const int d = static_cast<int>(j["crude_degree"].get<double>());
        auto c1 = j["crude_coeffs"].get<std::vector<double>>();
        const double rw = j["refine_window"].get<double>();
        auto c2 = j["refine_coeffs"].get<std::vector<double>>();
        if (ffn) {
          c.ffn_crude_degree = d;
          c.ffn_crude_invsqrt = c1;
          c.ffn_refine_window = rw;
          c.ffn_refine_invsqrt = c2;
        } else {
          c.attn_crude_degree = d;
          c.attn_crude_invsqrt = c1;
          c.attn_refine_window = rw;
          c.attn_refine_invsqrt = c2;
        }
      };
      stage(cl["attn_norm"], cal, /*ffn=*/false);
      stage(cl["ffn_norm"], cal, /*ffn=*/true);
      if (cal.attn_crude_degree > 0 || cal.ffn_crude_degree > 0) {
        std::cout << "TWO-STAGE norms: attn crude "
                  << cal.attn_crude_degree << " refine window "
                  << cal.attn_refine_window << " deg "
                  << (cal.attn_refine_invsqrt.empty()
                          ? 0
                          : int(cal.attn_refine_invsqrt.size()) - 1)
                  << " | ffn crude " << cal.ffn_crude_degree
                  << " refine window " << cal.ffn_refine_window << " deg "
                  << (cal.ffn_refine_invsqrt.empty()
                          ? 0
                          : int(cal.ffn_refine_invsqrt.size()) - 1)
                  << std::endl;
      }
      // AND EVERYTHING PER TOKEN GOES. The plan's windows are stated for the
      // RAW variance, so a per-token rescale left on would move the invsqrt's
      // argument off the window it was fitted for.
      //
      // `BERT_CERT_GELU_ONLY=1` keeps the norms as they were -- their
      // per-token rescale, their window of 1.5, their degree 7 -- and
      // certifies the GELU alone. That half FITS today's landing (a
      // degree-63 band tree is `levels(63) + 1 = 7`, exactly the budget)
      // where the norms' half does not (degree 31 is 9 against 7, and no
      // landing that affords it fits an 80 GB card: 12.5). So it is the
      // only way to measure the certified GELU on the real crypto today.
      const bool gelu_only = std::getenv("BERT_CERT_GELU_ONLY") != nullptr;
      if (!gelu_only) {
        cal.attn_scale.clear();
        cal.ffn_scale.clear();
        cal.row_suppress.clear();
      } else {
        cal.attn_alpha = 1.0;
        cal.attn_window = 1.5;
        cal.attn_degree = 0;
        cal.attn_invsqrt.clear();
        cal.ffn_alpha = 1.0;
        cal.ffn_window = 1.5;
        cal.ffn_degree = 0;
        cal.ffn_invsqrt.clear();
      }
      std::cout << "CERTIFIED plan: " << cal.gelu_bands.size()
                << " GELU bands, ranges";
      for (const auto &b : cal.gelu_bands) std::cout << " " << b.range;
      std::cout << " deg " << cal.gelu_bands[0].degree
                << "; attn window " << cal.attn_window << " deg "
                << cal.attn_degree << ", ffn window " << cal.ffn_window
                << " deg " << cal.ffn_degree
                << (gelu_only ? "; GELU ONLY (the norms keep their "
                                "per-prompt per-token rescale)"
                              : "; nothing per token")
                << std::endl;
    } else {
      std::cout << "corpus GELU plan: bulk +-" << cal.gelu_range << " deg "
                << cal.gelu_degree << ", wide +-" << cal.gelu_wide_range
                << " deg " << cal.gelu_wide_degree << ", "
                << static_cast<int>(cl["gelu_wide_channels"].get<double>())
                << " wide channels"
                << std::endl;
    }
  }
  std::cout << "cq " << cq << ", ck " << ck << ", cv " << cv
            << ", stream in " << stream_scale << " out " << stream_out
            << ", int_scale " << int_scale << ", m_eff " << L.m_eff
            << std::endl;

  // ---- the weights, declared --------------------------------------------
  //
  // The emission's rows, dense (Doing.md 3.12): one image per channel group,
  // rows `hh * 16 + cp` over the model's heads -- which is the leg's doorstep
  // after its merge. Module row `r` of group `g` is declared output
  // `g * rank + rev(r)`, which is `Project`'s own contract.
  const int qkv_groups = kD / layout.rank;
  const int qkv_declared = qkv_groups * kRank;
  auto qkv_entry = [&](int g, int hh, int cp, int &oc, int &o) {
    const int row = hh * layout.rank + cp;
    oc = g * kRank + Rev(row, 9);
    o = hh * kD + g * layout.rank + cp;
  };
  auto declare_qkv = [&](const std::vector<double> &wsrc,
                         std::vector<double> &out) {
    out.assign(static_cast<size_t>(kDeclaredH) * qkv_declared, 0.0);
    for (int g = 0; g < qkv_groups; g++) {
      for (int hh = 0; hh < kHeads; hh++) {
        for (int cp = 0; cp < layout.rank; cp++) {
          int oc = 0, o = 0;
          qkv_entry(g, hh, cp, oc, o);
          for (int c = 0; c < kH; c++) {
            out[static_cast<size_t>(c) * qkv_declared + oc] =
                wsrc[static_cast<size_t>(c) * kH + o];
          }
        }
      }
    }
  };
  auto declare_qkv_bias = [&](const std::vector<double> &bsrc,
                              std::vector<double> &out) {
    out.assign(qkv_declared, 0.0);
    for (int g = 0; g < qkv_groups; g++) {
      for (int hh = 0; hh < kHeads; hh++) {
        for (int cp = 0; cp < layout.rank; cp++) {
          int oc = 0, o = 0;
          qkv_entry(g, hh, cp, oc, o);
          out[oc] = bsrc[o];
        }
      }
    }
  };
  std::vector<double> wq_dec, wk_dec, wv_dec, bq_dec, bk_dec, bv_dec;
  declare_qkv(L.wq, wq_dec);
  declare_qkv(L.wk, wk_dec);
  declare_qkv(L.wv, wv_dec);
  declare_qkv_bias(L.bq, bq_dec);
  declare_qkv_bias(L.bk, bk_dec);
  declare_qkv_bias(L.bv, bv_dec);

  const int num_images = kD / layout.rank;
  const int attn_declared = num_images * kRank;
  std::vector<int> attn_map(attn_declared, -1);
  for (int bi = 0; bi < num_images; bi++) {
    for (int col = 0; col < layout.rank; col++) {
      for (int lane = 0; lane < layout.lanes; lane++) {
        const int head = Rev(lane, 5);
        if (head >= kHeads) continue;
        attn_map[bi * kRank + Rev(col, 4) * 32 + Rev(lane, 5)] =
            head * kD + bi * layout.rank + col;
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
          L.wo[static_cast<size_t>(a) * kH + c];
    }
  }
  const std::vector<double> wint_dec =
      DeclareSquare(L.wint, kH, kDeclaredH, kI, kDeclaredI);
  const std::vector<double> wout_dec =
      DeclareSquare(L.wout, kI, kDeclaredI, kH, kDeclaredH);
  const std::vector<double> bo_dec = Declare(L.bo, kDeclaredH);
  const std::vector<double> bint_dec = Declare(L.bint, kDeclaredI);
  const std::vector<double> bout_dec = Declare(L.bout, kDeclaredH);
  const std::vector<double> ag_dec = Declare(L.ag, kDeclaredH);
  const std::vector<double> ab_dec = Declare(L.ab, kDeclaredH);
  const std::vector<double> fg_dec = Declare(L.fg, kDeclaredH);
  const std::vector<double> fb_dec = Declare(L.fb, kDeclaredH);

  // THE WEIGHT FOLD. The model-channel stream carries `d_c`, so everything
  // that PRODUCES it multiplies by `d_c` (the O and down projections and
  // their biases) and everything that CONSUMES it divides (q/k/v and the up
  // projection). The norms' gain and bias are NOT folded here -- `NormTurn`
  // applies `d_c` to them itself, and the crossing's plaintext takes it back
  // out before the reduction, which is the one place a weight cannot.
  std::vector<double> wq_f(wq_dec), wk_f(wk_dec), wv_f(wv_dec), wo_f(wo_dec),
      wint_f(wint_dec), wout_f(wout_dec), bo_f(bo_dec), bout_f(bout_dec);
  if (CS) {
    const int qkv_declared = static_cast<int>(wq_dec.size()) / kDeclaredH;
    for (int in_d = 0; in_d < kDeclaredH; in_d++) {
      const double inv = 1.0 / chan_sup[in_d];
      for (int o = 0; o < qkv_declared; o++) {
        const size_t x = static_cast<size_t>(in_d) * qkv_declared + o;
        wq_f[x] *= inv;
        wk_f[x] *= inv;
        wv_f[x] *= inv;
      }
      for (int o = 0; o < kDeclaredI; o++) {
        wint_f[static_cast<size_t>(in_d) * kDeclaredI + o] *= inv;
      }
    }
    for (int c = 0; c < kDeclaredH; c++) {
      const double d = chan_sup[c];
      for (int in_d = 0; in_d < attn_declared; in_d++) {
        wo_f[static_cast<size_t>(in_d) * kDeclaredH + c] *= d;
      }
      for (int in_d = 0; in_d < kDeclaredI; in_d++) {
        wout_f[static_cast<size_t>(in_d) * kDeclaredH + c] *= d;
      }
      bo_f[c] *= d;
      bout_f[c] *= d;
    }
  }

  typename cheddar::CiBertLayer<word>::Weights w;
  w.q.host = &wq_f;
  w.k.host = &wk_f;
  w.v.host = &wv_f;
  w.o.host = &wo_f;
  w.inter.host = &wint_f;
  w.out.host = &wout_f;
  w.bq = &bq_dec;
  w.bk = &bk_dec;
  w.bv = &bv_dec;
  w.bo = &bo_f;
  w.bint = &bint_dec;
  w.bout = &bout_f;
  w.attn_gain = &ag_dec;
  w.attn_bias = &ab_dec;
  w.ffn_gain = &fg_dec;
  w.ffn_bias = &fb_dec;
  w.tag = tag;

  // ---- Q, K and V, and their crossings -----------------------------------
  const auto t_emit0 = std::chrono::steady_clock::now();
  std::vector<std::vector<Ciphertext<word>>> qkv;
  layer.Emit(qkv, stream, qkv_declared, w, cal);
  std::vector<std::vector<Ciphertext<word>>> imgs(3);
  for (int j = 0; j < 3; j++) {
    imgs[j].resize(qkv_groups);
    for (int g = 0; g < qkv_groups; g++) {
      qkv[j][g].SetNumSlots(num_slots);
      fctx->HalfBootModule(imgs[j][g], qkv[j][g], fevk,
                           *layer.Base().GetModuleBasis());
    }
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t_emit1 = std::chrono::steady_clock::now();

  // ---- the softmax calibration, in chain units ---------------------------
  typename cheddar::CiSinCAttention<word>::SoftMaxCalibration sc;
  sc.m_eff = L.m_eff;
  sc.span = cqk * L.span_raw;
  sc.shift = cqk * L.s_raw_max;
  sc.norm_lo = 0.9;
  sc.norm_hi = 1.1;
  sc.inv_degree = 7;
  sc.causal = true;         // the per-row walk
  sc.bidirectional = true;  // with every key live
  // A DEAD LANE NEEDS A LIVE-LOOKING ROW. Its scores are zero, so shift zero
  // and norm `dim` put its argument at exactly one; left at zero the inverse
  // square root would be asked for 1/sqrt(0), outside every window.
  sc.row_shift.assign(layout.lanes, std::vector<double>(kT, 0.0));
  sc.row_norm.assign(layout.lanes,
                     std::vector<double>(kT, static_cast<double>(layout.dim)));
  for (int lane = 0; lane < layout.lanes; lane++) {
    const int head = Rev(lane, 5);
    if (head >= kHeads) continue;
    for (int t = 0; t < kT; t++) {
      sc.row_shift[lane][t] = cqk * L.row_shift[head][t];
      sc.row_norm[lane][t] = L.row_norm[head][t];
    }
  }
  attn.PrepareSoftMax(sc);

  // ---- the leg -----------------------------------------------------------
  const auto t_leg0 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> s0;
  double carried = 0.0;
  attn.Scores(s0, imgs[0], imgs[1], keys, &carried);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_LT(carried * cqk * s_abs, 0.95)
      << "the q/k sizing missed EvalMod's range";
  // FUSED: `Scores` hands back SLOTS at `GetTopLevel()`, carrying the chain's
  // factor as a `Boot` would have -- the tower's HalfBoot and the lane prefix
  // are the return, so there is no caller Boot here at all.
  ASSERT_EQ(boot.param->NPToLevel(s0[0].GetNP()), attn.GetTopLevel());
  std::vector<Ciphertext<word>> P;
  attn.SoftMax(P, s0, carried, bevk);
  std::vector<Ciphertext<word>> attn_out;
  attn.Values(attn_out, P, imgs[2], keys);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t_leg1 = std::chrono::steady_clock::now();

  // ---- the Boots and the seam -------------------------------------------
  //
  // Only the images the model fills: the chain writes `num_cts` column
  // ciphertexts and BERT's 64-wide head lives in the first four of them.
  const auto t_seam0 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> h_cts(num_images);
  for (int bi = 0; bi < num_images; bi++) {
    // Fused, `Values` already returned these booted, at the tower's landing
    // less the prefix. They only have to meet the seam, whose level the two
    // rings share.
    Ciphertext<word> &booted = attn_out[bi];
    const int seam_in = layer.Base().GetSeamInputLevel();
    ASSERT_GE(boot.param->NPToLevel(booted.GetNP()), seam_in)
        << "the leg's return landed below the seam's input level";
    if (boot.param->NPToLevel(booted.GetNP()) > seam_in) {
      Ciphertext<word> down;
      bctx->LevelDown(down, booted, seam_in);
      booted = std::move(down);
    }
    layer.Base().Seam(h_cts[bi], booted, fevk);
  }
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t_seam1 = std::chrono::steady_clock::now();

  // What the seam handed over, against the clear attention: this is where the
  // leg's own factor is read, and `o_scale` is stated against it. In a run
  // that does not decrypt, it is a calibration constant like any other.
  double o_carried = 1.0;
  {
    std::vector<double> got(static_cast<size_t>(kT) * kH, 0.0);
    for (int bi = 0; bi < num_images; bi++) {
      Plaintext<word> pt;
      ffn.ui->Decrypt(pt, h_cts[bi]);
      std::vector<double> co;
      ffn.context->encoder_.DecodeCoeff(co, pt);
      const auto comp = Components(co);
      for (int cc = 0; cc < kRank; cc++) {
        const int a = attn_map[bi * kRank + cc];
        if (a < 0) continue;
        for (int t = 0; t < kT; t++) {
          got[static_cast<size_t>(t) * kH + a] = comp[Rev(cc, 9)][t];
        }
      }
    }
      o_carried = Report("stage 1: the seam's attention output", got, L.av);
    EXPECT_LT(RelBits(got, L.av), -7.0)
        << "the leg and the seam did not deliver the attention output";
  }
  cal.o_scale = stream_scale / o_carried;

  // ---- the two turns -----------------------------------------------------
  const auto t_turn0 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> h_ct;
  layer.AttentionTurn(h_ct, h_cts, stream, w, cal, fevk);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  {
    // H CARRIES THE ROW SUPPRESSION, by design: the feed-forward's two halves
    // both do and the output LayerNorm takes it back out. So the reference
    // for this stage is the suppressed one; stage 3 below is the answer.
    std::vector<double> want = L.h;
    for (int t = 0; t < kT; t++) {
      for (int c = 0; c < kH; c++) {
        want[static_cast<size_t>(t) * kH + c] *= suppress[LAYER][t];
      }
    }
    std::vector<double> got;
    ReadStream(ffn, h_ct, got);
    unsup(got);
    Report("stage 2: H = LayerNorm(X + O + b_o), suppressed", got, want);
    EXPECT_LT(RelBits(got, want), -7.0);
  }
  std::vector<Ciphertext<word>> z_ct;
  layer.FeedForward(z_ct, h_ct, w, cal, fevk);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  const auto t_turn1 = std::chrono::steady_clock::now();
  {
    std::vector<double> got;
    ReadStream(ffn, z_ct, got);
    unsup(got);
    const double bits =
        RelBits(got, L.href);
    Report(("stage 3: THE LAYER, against h_L" + Two(LAYER) + ".f64").c_str(),
           got, L.href);
    worst_layer = std::max(worst_layer, bits);
    EXPECT_LT(bits, -6.0)
        << "layer " << LAYER << " is worse than 2^-6 against the float64 "
           "reference";
  }
  auto secs = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << "cost: Emit + 12 HalfBootModules " << secs(t_emit0, t_emit1)
            << " s, the leg " << secs(t_leg0, t_leg1) << " s, the seam "
            << secs(t_seam0, t_seam1) << " s, the two turns "
            << secs(t_turn0, t_turn1) << " s, LAYER "
            << secs(t_emit0, t_turn1) << " s" << std::endl;
  cheddar::MemoryPool::Report(("layer " + Two(LAYER) + " done").c_str());
  // THE CHAIN: the next layer reads this one's ENCRYPTED output. The weights'
  // converted operands go with the layer -- the leg caches by tag, and
  // twelve layers' worth would stand on the card at once otherwise.
  stream = std::move(z_ct);
  layer.Base().ReleaseWeights(tag);
  }
  // ---- THE THREE PREDICTION HEADS ---------------------------------------
  //
  // What a deployment actually serves. They read the encoder's output -- the
  // `stream` the loop above just left -- so they run once at the end of the
  // chain and not once a layer, and `BERT_HEADS=1` is what asks for them.
  // Both of their non-linearities are CERTIFIED by the same sphere argument
  // the feed-forward's GELU is (`bert_plan.py head_plan`): the encoder's
  // output is a LayerNorm output, so nothing here is read off this prompt.
  if (std::getenv("BERT_HEADS") != nullptr && num_layers == 12) {
    json hp;
    {
      const char *cdirh = std::getenv("BERT_CALIB_DIR");
      ASSERT_NE(cdirh, nullptr) << "the heads need a plan (BERT_CALIB_DIR)";
      std::ifstream hf(std::string(cdirh) + "/corpus.json");
      ASSERT_TRUE(hf.good()) << "no corpus.json in BERT_CALIB_DIR";
      json all;
      hf >> all;
      ASSERT_TRUE(all.contains("head"))
          << "the plan has no head section -- rerun bert_plan.py against a "
             "bert_all that has head/ and a corpus built by bert_cache.py";
      hp = all["head"];
    }
    const std::string hdir = std::string(wdir_env) + "/head";
    std::vector<double> pw, pb, cw, cb, mw, mb, mg, mnb, dec, ob;
    const size_t hh = static_cast<size_t>(kH) * kH;
    ASSERT_TRUE(ReadF32(hdir + "/pool_w.f32", hh, pw));
    ASSERT_TRUE(ReadF32(hdir + "/pool_b.f32", kH, pb));
    ASSERT_TRUE(ReadF32(hdir + "/cls_w.f32", static_cast<size_t>(kH) * 2, cw));
    ASSERT_TRUE(ReadF32(hdir + "/cls_b.f32", 2, cb));
    ASSERT_TRUE(ReadF32(hdir + "/mlm_w.f32", hh, mw));
    ASSERT_TRUE(ReadF32(hdir + "/mlm_b.f32", kH, mb));
    ASSERT_TRUE(ReadF32(hdir + "/mlm_norm.f32", kH, mg));
    ASSERT_TRUE(ReadF32(hdir + "/mlm_norm_bias.f32", kH, mnb));
    ASSERT_TRUE(ReadF32(hdir + "/mlm_dec.f32",
                        static_cast<size_t>(kH) * kV, dec));
    ASSERT_TRUE(ReadF32(hdir + "/mlm_out_b.f32", kV, ob));

    std::vector<double> pool_ref, pooled_ref, cls_ref, mlm_ref;
    ASSERT_TRUE(ReadF64(std::string(rdir_env) + "/pooled.f64", kH, pooled_ref));
    ASSERT_TRUE(ReadF64(std::string(rdir_env) + "/cls_logits.f64", 2, cls_ref));
    ASSERT_TRUE(ReadF64(std::string(rdir_env) + "/mlm_logits.f64",
                        static_cast<size_t>(kT) * kV, mlm_ref));

    // The declared forms, and the SAME fold the layers use: the head's two
    // first projections CONSUME the model stream, so they divide by `d_c`;
    // the classifier and the decoder read the head's own intermediate, which
    // never carried it.
    std::vector<double> pool_f =
        DeclareSquare(pw, kH, kDeclaredH, kH, kDeclaredH);
    std::vector<double> mlm_f =
        DeclareSquare(mw, kH, kDeclaredH, kH, kDeclaredH);
    if (CS) {
      for (int in_d = 0; in_d < kDeclaredH; in_d++) {
        const double inv = 1.0 / chan_sup[in_d];
        for (int o = 0; o < kDeclaredH; o++) {
          pool_f[static_cast<size_t>(in_d) * kDeclaredH + o] *= inv;
          mlm_f[static_cast<size_t>(in_d) * kDeclaredH + o] *= inv;
        }
      }
    }
    const std::vector<double> cls_d =
        DeclareSquare(cw, kH, kDeclaredH, 2, kRank);
    const std::vector<double> dec_d =
        DeclareSquare(dec, kH, kDeclaredH, kV, kVDeclared);
    const std::vector<double> pb_d = Declare(pb, kDeclaredH);
    const std::vector<double> cb_d = Declare(cb, kRank);
    const std::vector<double> mb_d = Declare(mb, kDeclaredH);
    const std::vector<double> mg_d = Declare(mg, kDeclaredH);
    const std::vector<double> mnb_d = Declare(mnb, kDeclaredH);
    const std::vector<double> ob_d = Declare(ob, kVDeclared);

    typename cheddar::CiBertLayer<word>::HeadWeights hw;
    hw.pool.host = &pool_f;
    hw.cls.host = &cls_d;
    hw.mlm.host = &mlm_f;
    hw.dec.host = &dec_d;
    hw.pool_bias = &pb_d;
    hw.cls_bias = &cb_d;
    hw.mlm_bias = &mb_d;
    hw.mlm_gain = &mg_d;
    hw.mlm_norm_bias = &mnb_d;
    hw.dec_bias = &ob_d;
    hw.tag = "head";

    typename cheddar::CiBertLayer<word>::HeadCalibration hc;
    hc.stream_scale = s_out[num_layers - 1];
    hc.channel_suppress = chan_sup;
    hc.tanh_range = hp["pool_range"].get<double>();
    hc.tanh_degree = static_cast<int>(hp["pool_degree"].get<double>());
    hc.tanh_coeffs = hp["pool_coeffs"].get<std::vector<double>>();
    hc.gelu_range = hp["mlm_range"].get<double>();
    hc.gelu_degree = static_cast<int>(hp["mlm_degree"].get<double>());
    hc.gelu_coeffs = hp["mlm_coeffs"].get<std::vector<double>>();
    hc.mlm_alpha = hp["mlm_norm"]["alpha"].get<double>();
    hc.mlm_window = hp["mlm_norm"]["window"].get<double>();
    hc.mlm_degree = static_cast<int>(hp["mlm_norm"]["degree"].get<double>());
    hc.mlm_invsqrt = hp["mlm_norm"]["coeffs"].get<std::vector<double>>();
    if (hp["mlm_norm"].contains("crude_degree")) {
      hc.mlm_crude_degree =
          static_cast<int>(hp["mlm_norm"]["crude_degree"].get<double>());
      hc.mlm_crude_invsqrt =
          hp["mlm_norm"]["crude_coeffs"].get<std::vector<double>>();
      hc.mlm_refine_window = hp["mlm_norm"]["refine_window"].get<double>();
      hc.mlm_refine_invsqrt =
          hp["mlm_norm"]["refine_coeffs"].get<std::vector<double>>();
    }
    hc.vocab_live = kV;
    hc.vocab_declared = kVDeclared;
    // The two projections' rides. `u` is bounded by the certified interval,
    // so both are a division and not a fit -- and the two output scales are
    // free, because nothing crosses after them: the answer is decrypted.
    hc.pool_scale = Ride() / (hc.stream_scale * hc.tanh_range);
    hc.mlm_scale = Ride() / (hc.stream_scale * hc.gelu_range);
    // The norm's output has a CERTIFIED bound of its own and needs no
    // prompt: a LayerNorm output lies exactly on the sphere of radius
    // sqrt(kH), so `|gain_j n_j + bias_j| <= sqrt(kH) |gain_j| + |bias_j|`.
    // Riding it there is worth real bits -- the decoder reads it through an
    // int8 product and a message at the stream's own factor would sit two
    // orders under the ride for no reason.
    double nbound = 0.0;
    for (int j = 0; j < kH; j++) {
      nbound = std::max(nbound,
                        std::sqrt(static_cast<double>(kH)) * std::abs(mg[j]) +
                            std::abs(mnb[j]));
    }
    hc.mlm_out_scale = Ride() / nbound;
    // The GELU's answer crosses once more (its LayerNorm's `ToSlot`), so it
    // has to ride like any other crossing input. `|GELU(u)| <= |u| <=
    // gelu_range` is certified, and the factor rides the fit.
    hc.gelu_out_scale = Ride() / (layer.GetKappa() * hc.gelu_range);
    // Nothing crosses after the two answers, so their scales are free.
    hc.cls_scale = 1.0;
    hc.dec_scale = 1.0;
    std::cout << "THE HEADS: tanh over +-" << hc.tanh_range << " at degree "
              << hc.tanh_degree << ", the MLM GELU over +-" << hc.gelu_range
              << " at degree " << hc.gelu_degree
              << ", its LayerNorm window " << hc.mlm_window << " at degree "
              << hc.mlm_degree << "; kappa " << layer.GetKappa()
              << ", the GELU's answer rides " << hc.gelu_out_scale
              << " and the norm's output " << hc.mlm_out_scale << std::endl;

    const auto t_head0 = std::chrono::steady_clock::now();
    {
      std::vector<Ciphertext<word>> lg;
      layer.Pooler(lg, stream, hw, hc, fevk);
      cudaDeviceSynchronize();
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      ASSERT_EQ(lg.size(), 1u);
      Plaintext<word> pt;
      ffn.ui->Decrypt(pt, lg[0]);
      std::vector<double> co;
      ffn.context->encoder_.DecodeCoeff(co, pt);
      const auto comp = Components(co);
      // Two logits, at declared 0 and 1, and only the [CLS] row is the
      // pooler's -- so token 0 is what a deployment reads.
      std::vector<double> got(2, 0.0), want(cls_ref);
      const double f = hc.cls_scale * layer.GetKappa();
      for (int j = 0; j < 2; j++) got[j] = comp[Rev(j, 9)][0] / f;
      std::cout << "  NSP logits " << got[0] << " " << got[1] << " against "
                << want[0] << " " << want[1] << std::endl;
      const double bits = RelBits(got, want);
      std::cout << "  [the NSP classifier] rms 2^" << bits << std::endl;
      EXPECT_LT(bits, -5.0) << "the pooler and the classifier disagree with "
                               "the float64 reference";
    }
    const auto t_head1 = std::chrono::steady_clock::now();
    {
      std::vector<Ciphertext<word>> lg, nrm;
      layer.MlmHead(lg, stream, hw, hc, fevk, &nrm);
      cudaDeviceSynchronize();
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      ASSERT_EQ(static_cast<int>(lg.size()), kVDeclared / kRank);
      std::vector<double> got(static_cast<size_t>(kT) * kV, 0.0);
      const double f = hc.dec_scale * hc.mlm_out_scale;
      for (size_t g = 0; g < lg.size(); g++) {
        Plaintext<word> pt;
        ffn.ui->Decrypt(pt, lg[g]);
        std::vector<double> co;
        ffn.context->encoder_.DecodeCoeff(co, pt);
        const auto comp = Components(co);
        for (int j = 0; j < kRank; j++) {
          const int declared = static_cast<int>(g) * kRank + j;
          if (declared >= kV) continue;
          for (int t = 0; t < kT; t++) {
            got[static_cast<size_t>(t) * kV + declared] =
                comp[Rev(j, 9)][t] / f;
          }
        }
      }
      // THE HEAD CUT IN TWO. The transform, its GELU and its LayerNorm on
      // one side and the tied decoder on the other -- so a number that
      // disagrees says WHICH half, and `mlm_norm_out.f64` is the reference
      // for the join.
      {
        std::vector<double> nref, ngot(static_cast<size_t>(kT) * kH, 0.0);
        if (ReadF64(std::string(rdir_env) + "/mlm_norm_out.f64",
                    static_cast<size_t>(kT) * kH, nref)) {
          for (size_t k = 0; k < nrm.size(); k++) {
            Plaintext<word> pt;
            ffn.ui->Decrypt(pt, nrm[k]);
            std::vector<double> co;
            ffn.context->encoder_.DecodeCoeff(co, pt);
            const auto comp = Components(co);
            for (int j = 0; j < kRank; j++) {
              const int declared = static_cast<int>(k) * kRank + j;
              if (declared >= kH) continue;
              for (int t = 0; t < kT; t++) {
                ngot[static_cast<size_t>(t) * kH + declared] =
                    comp[Rev(j, 9)][t] / hc.mlm_out_scale;
              }
            }
          }
          Report("the MLM transform + GELU + LayerNorm", ngot, nref);
        }
      }
      const double bits = RelBits(got, mlm_ref);
      std::cout << "  [the MLM head] rms 2^" << bits << std::endl;
      // The number a deployment reads is the ARGMAX, so it is reported
      // beside the rms: a head that agrees on every token's top-1 is right
      // whatever the logits' last bits do.
      int agree = 0;
      for (int t = 0; t < kT; t++) {
        int ga = 0, wa = 0;
        for (int j = 1; j < kV; j++) {
          const size_t b = static_cast<size_t>(t) * kV;
          if (got[b + j] > got[b + ga]) ga = j;
          if (mlm_ref[b + j] > mlm_ref[b + wa]) wa = j;
        }
        if (ga == wa) agree++;
      }
      std::cout << "  [the MLM head] top-1 agrees on " << agree << " of " << kT
                << " tokens" << std::endl;
      EXPECT_LT(bits, -5.0)
          << "the MLM head disagrees with the float64 reference";
      EXPECT_GE(agree, kT - 2) << "the MLM head's top-1 moved";
    }
    const auto t_head2 = std::chrono::steady_clock::now();
    std::cout << "cost: the pooler + classifier "
              << std::chrono::duration<double>(t_head1 - t_head0).count()
              << " s, the MLM head "
              << std::chrono::duration<double>(t_head2 - t_head1).count()
              << " s" << std::endl;
  }
  std::cout << "the chain: " << num_layers << " layer(s), worst rms 2^"
            << worst_layer << std::endl;
}
