// Llama-3-8B decoder layers at the MODEL'S OWN WIDTH, encrypted end to end on
// the conjugate-invariant ring, one after another.
//
// ## What is new here
//
// Every earlier CI test ran one half of a layer at full width or the whole
// layer at a correctness width. `CiFfn.TheFullWidthFeedForwardRunsOnTheReal
// Weights` is the model's FFN; `CiBootSet.TheLegRunsOnTheRealWeights` is the
// model's attention leg; `CiBootSet.TheWholeLayerRunsOnTheRealSubring` is the
// whole layer at 512 declared channels of 4096, on synthetic weights. This
// joins them: 4096 live channels in seventeen half-density ciphertexts, 14336
// hidden in fifty-six, the real parameters of a named layer, and the output
// compared against the same layer in float64 -- then repeated, feeding each
// layer's residual stream to the next.
//
// ## What it needs, and where those come from
//
//   LLAMA3_ALL_DIR   `L00 .. L31`, each with the seven projections and two
//                    norms as f32 blobs in [in, out] order, plus
//                    `input_nosink.f32`.  `reference/export_layers.py`.
//   LLAMA3_REF_DIR   `h_L{k}.f64`, the residual stream after layer k in
//                    float64, and `calib.json`.  `reference/reference_forward.py`.
//
// The reference is float64 over the same bf16-widened weights, so the only
// thing between it and the true model is float64 arithmetic -- far below
// anything a ciphertext can see. The CALIBRATION comes from the same script
// because that is what [SYLPH] section 3.1 says calibration is: ranges fitted
// offline on the clear model, not measured in the run.
//
// ## Three things this test refuses to do
//
// **It does not decrypt to find a constant.** The crossing constant is
// `BootContext::GetMessageRatio()` and `kappa` follows from it (Doing.md
// 1.5dk); the leg's `restore` is `1 / crossing` for the same reason. Every
// earlier test fitted at least one of these against a decrypted twin.
//
// **It does not reuse a weight name.** `CoeffLinearLeg` caches converted
// weights by name and asserts the fingerprint, so every layer tags its own --
// a repeated tag with different weights would otherwise be a wrong layer that
// still decrypts.
//
// **It does not report the sink rows.** A prefix of beginning-of-sequence
// tokens has a hidden state whose mean square is ~277,000x the user tokens',
// which no Chebyshev window covers; the tree's standing treatment is a public
// rescaled copy ([SYLPH] 3.1.1's prefix is prompt-independent and therefore
// public), and the reference applies the same rescale at every norm. Accuracy
// is reported on tokens `kSinkTokens .. T-1`, as `LlamaBlockTest` does.
//
// ## Bootstraps are counted, not estimated
//
// `BootContext::GetBootCounts()` counts every crossing the library performs.
// [SYLPH] Table 6 has no bootstrap row because its crossings sit inside the
// projection rows, and the only way to compare is to count.

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/MemoryPool.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#include "extension/CiLlamaLayer.h"
#include "extension/CiSinCAttention.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using Complex = std::complex<double>;
using cheddar::BootContext;
using cheddar::Ciphertext;
using cheddar::EvkRequest;
using cheddar::Plaintext;
using json = nlohmann::json;

namespace {

constexpr const char *kBootParam = "ci16_35.json";
constexpr const char *kSwitchParam = "ci_ringswitch16_35_boot.json";
constexpr const char *kSmallParam = "ci12_35_boot.json";
constexpr const char *kLiftedParam = "ringdegree13_35_boot.json";

// The model.
constexpr int kT = 128, kH = 4096, kKv = 1024, kD = 128, kI = 14336;
constexpr int kHeads = kH / kD, kKvHeads = kKv / kD;   // 32 and 8
constexpr double kRopeTheta = 500000.0;
constexpr int kSinkTokens = 2;

// The packing. A half-density ciphertext carries `rank/2 - 1` live model
// channels -- component zero has no partner -- and `rank/2` hidden ones,
// because nothing reduces over the hidden axis.
constexpr int kRank = 512, kSmall = kT, kPcmmLevel = 1;
constexpr int kPerModel = kRank / 2 - 1;   // 255
constexpr int kPerHidden = kRank / 2;      // 256
constexpr int kNumH = (kH + kPerModel - 1) / kPerModel;       // 17
constexpr int kNumHid = (kI + kPerHidden - 1) / kPerHidden;   // 56
constexpr int kDeclaredH = kNumH * kRank;                     // 8704
constexpr int kDeclaredHid = kNumHid * kRank;                 // 28672

int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

int ModelSlot(int m) { return (m / kPerModel) * kRank + 2 * (m % kPerModel + 1); }
int HiddenSlot(int j) { return (j / kPerHidden) * kRank + 2 * (j % kPerHidden); }

// The banded recomposition `rec[t*rank + I] = comp_I[t] + [I!=0] comp_{rank-I}[t+1]`
// and its inverse, which is what `ModDecomp` and a slot read respectively see.
std::vector<double> Recompose(const std::vector<std::vector<double>> &comp) {
  std::vector<double> out(static_cast<size_t>(kRank) * kSmall, 0.0);
  for (int t = 0; t < kSmall; t++) {
    for (int i = 0; i < kRank; i++) {
      double v = comp[i][t];
      if (i != 0 && t + 1 < kSmall) v += comp[kRank - i][t + 1];
      out[static_cast<size_t>(t) * kRank + i] = v;
    }
  }
  return out;
}

std::vector<std::vector<double>> Components(const std::vector<double> &co) {
  std::vector<std::vector<double>> comp(kRank, std::vector<double>(kSmall, 0.0));
  for (int t = 0; t < kSmall; t++) comp[0][t] = co[static_cast<size_t>(t) * kRank];
  for (int i = 1; i <= kRank / 2; i++) {
    const int mi = kRank - i;
    double ai = 0.0, am = 0.0;
    for (int t = kSmall - 1; t >= 0; t--) {
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

bool ReadF64(const std::string &path, size_t count, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.assign(count, 0.0);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(count * sizeof(double)));
  return static_cast<size_t>(f.gcount()) == count * sizeof(double);
}

double Ms(const std::chrono::steady_clock::time_point &a,
          const std::chrono::steady_clock::time_point &b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::chrono::steady_clock::time_point Tick() {
  cudaDeviceSynchronize();
  return std::chrono::steady_clock::now();
}

int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? std::atoi(e) : fallback;
}

double EnvDouble(const char *name, double fallback) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != 0) ? std::atof(e) : fallback;
}

}  // namespace

TEST(CiModel, TheModelRunsAtTheFullWidth) {
  const char *wdir_env = std::getenv("LLAMA3_ALL_DIR");
  const char *rdir_env = std::getenv("LLAMA3_REF_DIR");
  if (wdir_env == nullptr || rdir_env == nullptr) {
    GTEST_SKIP() << "LLAMA3_ALL_DIR and LLAMA3_REF_DIR must both be set";
  }
  const std::string wdir(wdir_env), rdir(rdir_env);
  const int num_layers = EnvInt("CHEDDAR_CI_LAYERS", 1);
  const double ride = EnvDouble("CHEDDAR_CI_RIDE", 0.2);
  const bool min_ks = EnvInt("CHEDDAR_CI_MINKS", 0) != 0;

  json calib_all;
  {
    std::ifstream f(rdir + "/calib.json");
    ASSERT_TRUE(f.good()) << "cannot open " << rdir << "/calib.json";
    f >> calib_all;
  }
  ASSERT_GE(static_cast<int>(calib_all["layers"].size()), num_layers);

  std::vector<double> x0;
  ASSERT_TRUE(ReadF32(wdir + "/input_nosink.f32",
                      static_cast<size_t>(kT) * kH, x0));

  // ---- the four rings, one secret -----------------------------------------
  Ring boot(kBootParam);  // the leg: slack zero, the softmax walk needs it
  auto swtch = std::make_unique<Ring>(kSwitchParam, boot.ui->GetSecretCoeffs());
  auto small = std::make_unique<Ring>(kSmallParam);
  auto lifted = std::make_unique<Ring>(
      kLiftedParam,
      cheddar::CiLiftHandler<word>::LiftSecret(small->ui->GetSecretCoeffs()));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  const int num_slots = boot.param->MaxNumSlots();
  ASSERT_EQ(num_slots, boot.Degree());

  const auto t_setup0 = Tick();
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(num_slots);
  {
    EvkRequest req;
    bctx->AddRequiredRotations(req, num_slots, min_ks);
    boot.ui->PrepareRotationKey(req);
  }

  cheddar::MlweHandler<word> mlwe(*boot.param, boot.context->ntt_handler_);
  cheddar::PcmmHandler<word> pcmm(*boot.param);
  const int pack_aux = boot.ui->PrepareModPackKeys(kSmall, kPcmmLevel,
                                                   /*num_aux=*/-1);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < kRank; j++) {
    pack_keys.push_back(&boot.ui->GetModPackKey(kRank, j));
  }

  // THE CROSSING CONSTANT, DERIVED. `HalfBoot` multiplies the message by
  // `level_zero_scale / q0`; `BootContext` computes it exactly and the leg's
  // `restore` is its inverse. Every earlier test measured this by decrypting a
  // ciphertext and fitting, which is right on the preset it was taken on.
  const double crossing = bctx->GetMessageRatio();
  std::cout << "crossing constant " << crossing << " = 2^"
            << std::log2(std::abs(crossing)) << " (derived)" << std::endl;

  typename cheddar::CiSinCAttention<word>::Config acfg;
  acfg.restore = 1.0 / crossing;
  acfg.rope_base = kRopeTheta;
  auto attn = std::make_unique<cheddar::CiSinCAttention<word>>(
      bctx, swtch->context, small->context, lifted->context, acfg);
  const auto layout = attn->GetLayout();

  swtch->ui->PrepareRingSwitchKey(small->Degree(),
                                  small->ui->GetSecretCoeffs(), 2);
  swtch->ui->PrepareInverseRingSwitchKey(small->Degree(),
                                         small->ui->GetSecretCoeffs(), 2);
  for (int idx : attn->LiftedRotationIndices()) {
    lifted->ui->PrepareRotationKey(idx, 2);
  }
  {
    EvkRequest req;
    attn->AddSwitchRotations(req);
    swtch->ui->PrepareRotationKey(req);
  }
  {
    EvkRequest req;
    attn->AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  typename cheddar::CiSinCAttention<word>::Keys keys;
  keys.boot = &boot.ui->GetEvkMap();
  keys.swtch = &swtch->ui->GetEvkMap();
  keys.lifted = &lifted->ui->GetEvkMap();
  keys.ring_switch = &swtch->ui->GetRingSwitchKey(layout.rank);
  keys.inverse_ring_switch = &swtch->ui->GetInverseRingSwitchKey(layout.rank);

  // ---- the FFN's Context, and the layer ------------------------------------
  Ring boot_ffn(kBootParam, boot.ui->GetSecretCoeffs(), /*slack=*/9,
                /*build_user_interface=*/false);
  auto fctx = std::dynamic_pointer_cast<BootContext<word>>(boot_ffn.context);
  ASSERT_NE(fctx, nullptr);
  boot_ffn.context->PrepareNarrowKeySwitch(kPcmmLevel, pack_aux);
  fctx->PrepareEvalMod();
  fctx->PrepareEvalSpecialFFT(num_slots, cheddar::BootVariant::kNormal,
                              /*cts_donor=*/bctx.get());
  {
    EvkRequest req;
    fctx->AddRequiredRotations(req, num_slots, min_ks);
    boot.ui->PrepareRotationKey(req);
  }

  typename cheddar::CiLlamaLayer<word>::Config lcfg;
  lcfg.num_tokens = kT;
  lcfg.proj_rank = kRank;
  lcfg.model_declared = kDeclaredH;
  lcfg.hidden_declared = kDeclaredHid;
  lcfg.model_live = kH;
  lcfg.product_level = kPcmmLevel;
  lcfg.ride = ride;
  lcfg.min_ks = min_ks;
  lcfg.verbose = true;
  cheddar::CiLlamaLayer<word> layer(fctx, layout, pack_keys, lcfg);
  {
    EvkRequest req;
    layer.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  // Both seam halves' rotations up front: the stages themselves are built per
  // half inside the loop, but their keys are the same set every layer.
  for (int half = 0; half < 2; half++) {
    layer.PrepareSeamHalf(half);
    EvkRequest req;
    layer.AddSeamHalfRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  layer.DropSeamHalf();
  const auto t_setup1 = Tick();
  std::cout << "[time] ONE-TIME setup (keys, converters, the layer): "
            << Ms(t_setup0, t_setup1) / 1000.0 << " s" << std::endl;

  // ---- the residual stream, encrypted --------------------------------------
  //
  // The stream carries a global factor so that its crossing rides at
  // `Config::ride`; every comparison below divides the fitted factor out, and
  // the factor itself rides through the layer unchanged because RMSNorm is
  // scale invariant and every projection is linear.
  auto rescale_sinks = [](std::vector<double> &x) {
    std::vector<double> ms(kT, 0.0);
    for (int t = 0; t < kT; t++) {
      double s = 0.0;
      for (int c = 0; c < kH; c++) {
        const double v = x[static_cast<size_t>(t) * kH + c];
        s += v * v;
      }
      ms[t] = s / kH;
    }
    double log_sum = 0.0;
    for (int t = kSinkTokens; t < kT; t++) log_sum += std::log(ms[t]);
    const double target = std::exp(log_sum / (kT - kSinkTokens));
    for (int t = 0; t < kSinkTokens; t++) {
      const double f = std::sqrt(target / ms[t]);
      for (int c = 0; c < kH; c++) x[static_cast<size_t>(t) * kH + c] *= f;
    }
  };
  rescale_sinks(x0);

  // ONE FACTOR FOR THE WHOLE RUN, sized on the largest residual any layer
  // reaches. The residual add forces the stream and the two projections that
  // write it to agree, and RMSNorm makes everything downstream scale free, so
  // a per-layer factor would have to be re-derived at every add for nothing.
  // The reference already knows every layer's magnitude.
  double x_absmax = 0.0;
  for (double v : x0) x_absmax = std::max(x_absmax, std::abs(v));
  double stream_absmax = x_absmax;
  for (int L = 0; L < num_layers; L++) {
    stream_absmax = std::max(
        stream_absmax, calib_all["layers"][L]["out_absmax"].get<double>());
    stream_absmax = std::max(
        stream_absmax, calib_all["layers"][L]["resid_absmax"].get<double>());
  }
  const double stream_scale = ride / std::max(stream_absmax, 1e-12);
  std::cout << "input |x| <= " << x_absmax << ", the stream reaches "
            << stream_absmax << " over " << num_layers
            << " layers, so it carries " << stream_scale << std::endl;

  auto encrypt_stream = [&](const std::vector<double> &x, double scale) {
    std::vector<Ciphertext<word>> out(kNumH);
    for (int k = 0; k < kNumH; k++) {
      std::vector<std::vector<double>> comp(kRank,
                                            std::vector<double>(kSmall, 0.0));
      for (int m = k * kPerModel; m < std::min(kH, (k + 1) * kPerModel); m++) {
        const int c = ModelSlot(m) - k * kRank;
        for (int t = 0; t < kT; t++) {
          comp[Rev(c, 9)][Rev(t, 7)] =
              scale * x[static_cast<size_t>(t) * kH + m];
        }
      }
      const auto co = Recompose(comp);
      Plaintext<word> pt;
      boot_ffn.context->encoder_.EncodeCoeff(
          pt, 0, boot_ffn.param->GetScale(0), co);
      boot.ui->Encrypt(out[k], pt);
      out[k].SetNumSlots(num_slots);
    }
    return out;
  };
  std::vector<Ciphertext<word>> stream = encrypt_stream(x0, stream_scale);

  // ---- the layers ----------------------------------------------------------
  bctx->ResetBootCounts();
  fctx->ResetBootCounts();
  const auto t_run0 = Tick();

  for (int L = 0; L < num_layers; L++) {
    const auto t_layer0 = Tick();
    const std::string ld = wdir + "/L" + (L < 10 ? "0" : "") + std::to_string(L);
    const json &cj = calib_all["layers"][L];
    std::cout << "\n================ layer " << L << " ================"
              << std::endl;

    std::vector<double> wq_f, wk_f, wv_f, wo_f, wg_f, wu_f, wd_f, an_f, fn_f;
    ASSERT_TRUE(ReadF32(ld + "/wq.f32", static_cast<size_t>(kH) * kH, wq_f));
    ASSERT_TRUE(ReadF32(ld + "/wk.f32", static_cast<size_t>(kH) * kKv, wk_f));
    ASSERT_TRUE(ReadF32(ld + "/wv.f32", static_cast<size_t>(kH) * kKv, wv_f));
    ASSERT_TRUE(ReadF32(ld + "/wo.f32", static_cast<size_t>(kH) * kH, wo_f));
    ASSERT_TRUE(ReadF32(ld + "/wgate.f32", static_cast<size_t>(kH) * kI, wg_f));
    ASSERT_TRUE(ReadF32(ld + "/wup.f32", static_cast<size_t>(kH) * kI, wu_f));
    ASSERT_TRUE(ReadF32(ld + "/wdown.f32", static_cast<size_t>(kI) * kH, wd_f));
    ASSERT_TRUE(ReadF32(ld + "/attn_norm.f32", kH, an_f));
    ASSERT_TRUE(ReadF32(ld + "/ffn_norm.f32", kH, fn_f));

    // ---- the pre-attention norm, encrypted --------------------------------
    typename cheddar::CiLlamaLayer<word>::Calibration cal;
    cal.attn_alpha = cj["attn_alpha"].get<double>();
    cal.attn_norm_window = cj["attn_norm_window"].get<double>();
    cal.alpha = cj["alpha"].get<double>();
    cal.norm_window = cj["norm_window"].get<double>();
    cal.silu_range = cj["silu_range"].get<double>();

    std::vector<double> an_dec(kDeclaredH, 0.0), fn_dec(kDeclaredH, 0.0);
    for (int c = 0; c < kH; c++) {
      an_dec[ModelSlot(c)] = an_f[c];
      fn_dec[ModelSlot(c)] = fn_f[c];
    }

    std::vector<Ciphertext<word>> normed;
    layer.AttentionNorm(normed, stream, an_dec, cal, boot.ui->GetEvkMap());
    const auto t_norm = Tick();

    // ---- Q, K and V: the real PC-MM at full width -------------------------
    std::vector<cheddar::MlweCiphertext<word>> x_parts;
    for (int k = 0; k < kNumH; k++) {
      Ciphertext<word> low;
      boot_ffn.context->LevelDown(low, normed[k], kPcmmLevel);
      std::vector<cheddar::MlweCiphertext<word>> parts;
      mlwe.ModDecomp(parts, low, kSmall);
      for (auto &p : parts) x_parts.push_back(std::move(p));
    }
    ASSERT_EQ(static_cast<int>(x_parts.size()), kDeclaredH);

    // The sizing, from the clear model's own maxima: the HalfBoot image bound
    // first, then the cap on the chain-unit score message.
    const double img_max = 0.45;
    const double qmax = cj["q_absmax"].get<double>();
    const double kmax = cj["k_absmax"].get<double>();
    const double vmax = cj["v_absmax"].get<double>();
    const double s_raw_min = cj["s_raw_min"].get<double>();
    const double s_raw_max = cj["s_raw_max"].get<double>();
    const double span_raw = s_raw_max - s_raw_min;
    const double m_eff = span_raw / std::sqrt(static_cast<double>(kD));
    double cq = img_max / qmax, ck = img_max / kmax;
    const double s_abs = std::max(std::abs(s_raw_max), std::abs(s_raw_min));
    const double prod_cap = 0.36 / s_abs;
    if (cq * ck > prod_cap) {
      const double sh = std::sqrt(prod_cap / (cq * ck));
      cq *= sh;
      ck *= sh;
    }
    const double cv = std::min(1.0, img_max / vmax);
    const double cqk = cq * ck;

    const double w_scale = boot.param->GetRescalePrimeProd(kPcmmLevel);
    // The projections' declared-index weights. `w[head][chan][declared in]`
    // is what an emission reads: row `hh*16 + cp` of half `(l, fam)` is head
    // `fam*16 + hh`, channel `l*16 + cp`. GQA is a weight slice: key/value
    // head `head / 4`.
    auto emit = [&](const std::vector<double> &w, int width, double scale,
                    int l, int fam, Ciphertext<word> &out) {
      std::vector<double> vals(static_cast<size_t>(kRank) * kDeclaredH, 0.0);
      const int heads_w = width / kD;
      for (int hh = 0; hh < 16; hh++) {
        for (int cp = 0; cp < 16; cp++) {
          const int row = hh * 16 + cp;
          const int head = fam * 16 + hh;
          const int chan = l * 16 + cp;
          // GQA IS `head / 4`, NOT `head % 8`. Llama-3-8B has 32 query heads
          // over 8 key/value heads, and each key/value head serves four
          // CONSECUTIVE query heads. The modulus reads the right number of
          // heads and the wrong ones, which is a layer that decrypts and is
          // wrong -- measured at relative 55.8 with a NEGATIVE fitted factor,
          // which is what an essentially random reindexing looks like.
          const int o = (heads_w == kHeads ? head : head / (kHeads / heads_w)) *
                            kD + chan;
          for (int c = 0; c < kH; c++) {
            vals[static_cast<size_t>(row) * kDeclaredH + ModelSlot(c)] =
                scale * w[static_cast<size_t>(c) * width + o];
          }
        }
      }
      cheddar::PlainMatrix<word> u;
      pcmm.EncodeMatrix(u, kPcmmLevel, w_scale, vals, kRank, kDeclaredH);
      std::vector<cheddar::MlweCiphertext<word>> mixed;
      pcmm.Multiply(mixed, u, x_parts);
      Ciphertext<word> packed, dropped;
      mlwe.ModPack(boot.context, packed, mixed, pack_keys);
      boot.context->Rescale(dropped, packed);
      dropped.SetNumSlots(num_slots);
      bctx->HalfBoot(out, dropped, boot.ui->GetEvkMap(), min_ks);
    };

    std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8), v_a(8), v_b(8);
    for (int l = 0; l < 8; l++) {
      emit(wq_f, kH, cq, l, 0, q_a[l]);
      emit(wq_f, kH, cq, l, 1, q_b[l]);
      emit(wk_f, kKv, ck, l, 0, k_a[l]);
      emit(wk_f, kKv, ck, l, 1, k_b[l]);
      emit(wv_f, kKv, cv, l, 0, v_a[l]);
      emit(wv_f, kKv, cv, l, 1, v_b[l]);
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const auto t_proj = Tick();

    // ---- the softmax calibration, in chain units --------------------------
    typename cheddar::CiSinCAttention<word>::SoftMaxCalibration sc;
    sc.m_eff = m_eff;
    sc.span = cqk * span_raw;
    sc.shift = cqk * s_raw_max;
    sc.norm_lo = 0.9;
    sc.norm_hi = 1.1;
    sc.exp_degree = 15;
    sc.inv_degree = 7;
    sc.causal = true;
    sc.row_shift.assign(layout.lanes, std::vector<double>(kT, 0.0));
    sc.row_norm.assign(layout.lanes, std::vector<double>(kT, 0.0));
    {
      const auto &rs = cj["row_shift_raw"];
      const auto &rn = cj["row_norm"];
      for (int lane = 0; lane < layout.lanes; lane++) {
        const int head = Rev(lane, 5);
        for (int t = 0; t < kT; t++) {
          sc.row_shift[lane][t] = cqk * rs[head][t].get<double>();
          sc.row_norm[lane][t] = rn[head][t].get<double>();
        }
      }
    }
    attn->PrepareSoftMax(sc);

    // ---- the leg ----------------------------------------------------------
    std::vector<Ciphertext<word>> s0;
    attn->Scores(s0, q_a, q_b, k_a, k_b, keys);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const double carried = s0[0].GetScale() / boot.param->base_scale_;
    ASSERT_LT(carried * cqk * s_abs, 0.95)
        << "the size_q/size_k fold missed EvalMod's range";
    std::vector<Ciphertext<word>> scores(layout.num_cts);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      s0[bi].SetNumSlots(num_slots);
      bctx->Boot(scores[bi], s0[bi], boot.ui->GetEvkMap(), min_ks);
    }
    std::vector<Ciphertext<word>> P;
    attn->SoftMax(P, scores, carried, boot.ui->GetEvkMap());
    std::vector<Ciphertext<word>> attn_out;
    attn->Values(attn_out, P, v_a, v_b, keys);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const auto t_leg = Tick();

    // ---- the seam ---------------------------------------------------------
    std::vector<Ciphertext<word>> booted(layout.num_cts);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      bctx->Boot(booted[bi], attn_out[bi], boot.ui->GetEvkMap(), min_ks);
    }
    attn_out.clear();
    std::vector<Ciphertext<word>> h_cts(2 * layout.num_cts);
    for (int half = 0; half < 2; half++) {
      layer.PrepareSeamHalf(half);
      for (int bi = 0; bi < layout.num_cts; bi++) {
        layer.Seam(h_cts[bi * 2 + half], booted[bi], boot.ui->GetEvkMap());
      }
    }
    layer.DropSeamHalf();
    booted.clear();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const auto t_seam = Tick();

    // ---- the O projection, the residual, the FFN --------------------------
    //
    // The O weight takes V's own sizing back out and puts the residual at the
    // ride: the attention output carries `cv`, the stream carries
    // `stream_scale`, and the two must agree before they are added.
    const double o_absmax = cj["o_absmax"].get<double>();
    const double resid_absmax = cj["resid_absmax"].get<double>();
    (void)o_absmax;
    typename cheddar::CiLlamaLayer<word>::Weights lw;
    std::vector<double> wo_dec(static_cast<size_t>(2 * layout.num_cts * kRank) *
                                   kDeclaredH,
                               0.0);
    std::vector<double> wg_dec(static_cast<size_t>(kDeclaredH) * kDeclaredHid,
                               0.0);
    std::vector<double> wu_dec = wg_dec;
    std::vector<double> wd_dec(static_cast<size_t>(kDeclaredHid) * kDeclaredH,
                               0.0);
    // The attention output's declared channel: half ciphertext
    // `2*bi + lane/16`, channel `rev4(col)*32 + rev5(lane%16)`, which is
    // exactly `CiLlamaSeam`'s `chan_of` and the order `CoeffLinearLeg` numbers
    // a parent's channels in.
    {
      const int attn_declared = 2 * layout.num_cts * kRank;
      for (int bi = 0; bi < layout.num_cts; bi++) {
        for (int col = 0; col < layout.rank; col++) {
          for (int lane = 0; lane < layout.lanes; lane++) {
            const int k = 2 * bi + lane / 16;
            const int cc = Rev(col, 4) * 32 + Rev(lane % 16, 5);
            const int in_declared = k * kRank + cc;
            // head = rev5(lane), channel-in-head = bi*16 + col
            const int head = Rev(lane, 5);
            const int chan = bi * layout.rank + col;
            const int o_in = head * kD + chan;
            for (int c = 0; c < kH; c++) {
              wo_dec[static_cast<size_t>(in_declared) * kDeclaredH +
                     ModelSlot(c)] =
                  wo_f[static_cast<size_t>(o_in) * kH + c];
            }
          }
        }
      }
      (void)attn_declared;
    }
    for (int c = 0; c < kH; c++) {
      const size_t dc = ModelSlot(c);
      for (int j = 0; j < kI; j++) {
        const size_t dj = HiddenSlot(j);
        wg_dec[dc * kDeclaredHid + dj] = wg_f[static_cast<size_t>(c) * kI + j];
        wu_dec[dc * kDeclaredHid + dj] = wu_f[static_cast<size_t>(c) * kI + j];
        wd_dec[dj * kDeclaredH + dc] = wd_f[static_cast<size_t>(j) * kH + c];
      }
    }

    // The O projection's factor: divide out V's sizing, then put the residual
    // at the ride. The incoming stream already carries `stream_scale`.
    // The O weight takes V's own sizing back out and puts the result on the
    // stream's factor, so the residual add sees two quantities that agree.
    cal.res_scale = stream_scale / cv;
    // The gate and up are read off a NORMALISED stream, so they are in the
    // model's own units and their ride is set on those.
    cal.gate_scale = ride / std::max(cj["gate_absmax"].get<double>(), 1e-12);
    cal.stream_scale = stream_scale;
    lw.o = &wo_dec;
    lw.gate = &wg_dec;
    lw.up = &wu_dec;
    lw.down = &wd_dec;
    lw.ffn_norm = &fn_dec;
    lw.tag = "L" + std::to_string(L);

    std::vector<Ciphertext<word>> next;
    layer.FeedForward(next, h_cts, stream, lw, cal, boot.ui->GetEvkMap());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const auto t_ffn = Tick();

    // ---- against the same layer in float64 --------------------------------
    std::vector<double> ref;
    const std::string rp = rdir + "/h_L" + (L < 10 ? "0" : "") +
                           std::to_string(L) + ".f64";
    ASSERT_TRUE(ReadF64(rp, static_cast<size_t>(kT) * kH, ref))
        << "cannot read " << rp;

    double num = 0.0, den = 0.0, absmax = 0.0;
    std::vector<std::vector<std::vector<double>>> got(kNumH);
    for (int k = 0; k < kNumH; k++) {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, next[k]);
      std::vector<double> co;
      boot_ffn.context->encoder_.DecodeCoeff(co, pt);
      got[k] = Components(co);
    }
    for (int m = 0; m < kH; m++) {
      const int k = m / kPerModel;
      const int c = ModelSlot(m) - k * kRank;
      for (int t = kSinkTokens; t < kT; t++) {
        const double want = ref[static_cast<size_t>(t) * kH + m];
        num += got[k][Rev(c, 9)][Rev(t, 7)] * want;
        den += want * want;
        absmax = std::max(absmax, std::abs(want));
      }
    }
    ASSERT_GT(den, 1e-12) << "the reference is zero";
    const double fit = num / den;
    double err = 0.0;
    for (int m = 0; m < kH; m++) {
      const int k = m / kPerModel;
      const int c = ModelSlot(m) - k * kRank;
      for (int t = kSinkTokens; t < kT; t++) {
        const double v = got[k][Rev(c, 9)][Rev(t, 7)] / fit;
        err = std::max(err, std::abs(v - ref[static_cast<size_t>(t) * kH + m]));
      }
    }
    const auto bl = bctx->GetBootCounts();
    const auto bf = fctx->GetBootCounts();
    std::cout << "LAYER " << L << ": " << err << " against |h| <= " << absmax
              << " (relative " << (err / absmax) << " = 2^"
              << std::log2(err / absmax) << "), carried " << fit << std::endl;
    std::cout << "  [boot] leg Context full " << bl.full << ", half "
              << bl.half << " | FFN Context full " << bf.full << ", half "
              << bf.half << " | total " << (bl.Total() + bf.Total())
              << std::endl;
    std::cout << "  [time] norm " << Ms(t_layer0, t_norm) << " ms, Q/K/V "
              << Ms(t_norm, t_proj) << ", leg " << Ms(t_proj, t_leg)
              << ", seam " << Ms(t_leg, t_seam) << ", FFN "
              << Ms(t_seam, t_ffn) << ", layer "
              << Ms(t_layer0, t_ffn) / 1000.0 << " s" << std::endl;
    {
      size_t free_b = 0, total_b = 0;
      cudaMemGetInfo(&free_b, &total_b);
      std::cout << "  [mem] " << ((total_b - free_b) >> 20) << " MiB reserved"
                << std::endl;
    }
    EXPECT_LT(err / absmax, 0.25)
        << "layer " << L << " disagrees with the same layer in double by more "
        << "than the circuit can explain";
    (void)resid_absmax;

    stream = std::move(next);
  }

  const auto t_run1 = Tick();
  const auto bl = bctx->GetBootCounts();
  const auto bf = fctx->GetBootCounts();
  std::cout << "\n" << num_layers << " FULL-WIDTH LAYERS in "
            << Ms(t_run0, t_run1) / 1000.0 << " s, "
            << (bl.Total() + bf.Total()) << " bootstraps ("
            << (bl.Total() + bf.Total()) / num_layers << " a layer)"
            << std::endl;
}
