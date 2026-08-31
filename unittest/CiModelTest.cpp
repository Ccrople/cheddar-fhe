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
// `CHEDDAR_CI_MODULE=1`: the layer's non-leg half on the MODULE basis
// (Doing.md 3.5-3.7) -- every component live, 8 residual and 28 hidden
// ciphertexts, the crossings through `CiModuleBasis`/`HalfBootModule`, the
// SSE secret sampled in the module basis. The leg and the seam are unchanged
// (their images stay banded); the Q/K/V emissions read dense and write
// banded.
const bool kModule = [] {
  const char *e = std::getenv("CHEDDAR_CI_MODULE");
  return e != nullptr && e[0] == '1';
}();
const int kDensity = kModule ? 1 : 2;
const int kPerModel = kModule ? kRank : kRank / 2 - 1;   // 512 or 255
const int kPerHidden = kModule ? kRank : kRank / 2;      // 512 or 256
const int kNumH = (kH + kPerModel - 1) / kPerModel;       // 8 or 17
const int kNumHid = (kI + kPerHidden - 1) / kPerHidden;   // 28 or 56
const int kDeclaredH = kNumH * kRank;                     // 4096 or 8704
const int kDeclaredHid = kNumHid * kRank;                 // 14336 or 28672

int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

// THE STREAM'S COEFFICIENT POSITION FOR TOKEN `t`, which is the one thing the
// two halves of this tree had never had to agree on. A PC-MM preserves the
// coefficient position, and the leg's doorstep (`CiSinCAttention`'s `Door`)
// puts `rev7(token)` in the slot's low seven bits, so a projection whose input
// sits at position `p` emits a doorstep whose low seven bits are `rev7(p)`:
// the leg needs `p = t`. The FFN, the seam and every `CiFfnTest` encode were
// built on `p = rev7(t)`, which is equally self-consistent -- RMSNorm, SiLU
// and the projections are all per-token, so nothing there can see a permuted
// token axis -- and the model layer is the first place where the leg's
// convention and the seam's output meet. `CHEDDAR_CI_TOKPOS=0` restores the
// old one, for bisection against the tests that still carry it.
int g_tok_pos = 1;
int Pos(int t) { return g_tok_pos != 0 ? t : Rev(t, 7); }

int ModelSlot(int m) {
  return cheddar::CiLlamaLayer<word>::ModelSlot(m, kRank, kDensity);
}
int HiddenSlot(int j) {
  return cheddar::CiLlamaLayer<word>::HiddenSlot(j, kRank, kDensity);
}

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

// The blob as the exporter wrote it, for `CoeffLinearLeg::DeviceWeights`.
bool ReadF32Host(const std::string &path, size_t count,
                 cheddar::HostVector<float> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.resize(count);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  return static_cast<size_t>(f.gcount()) == count * sizeof(float);
}

// WHICH WEIGHT FORM THE LAYER GETS. Default: the f32 tensors on the device,
// encoded where they are read (Doing.md 3). `CHEDDAR_CI_HOST_WEIGHTS=1` builds
// the declared-width double matrices on the host instead -- the form every
// run before 2026-08-31 used, kept as the A/B for the conversion row.
bool HostWeightsFromEnv() {
  const char *e = std::getenv("CHEDDAR_CI_HOST_WEIGHTS");
  return e != nullptr && !(e[0] == '0' && e[1] == '\0');
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
  g_tok_pos = EnvInt("CHEDDAR_CI_TOKPOS", 1);
  const int stop_after = EnvInt("CHEDDAR_CI_STOP_AFTER", 0);
  const int first_layer = EnvInt("CHEDDAR_CI_FIRST_LAYER", 0);

  json calib_all;
  {
    std::ifstream f(rdir + "/calib.json");
    ASSERT_TRUE(f.good()) << "cannot open " << rdir << "/calib.json";
    f >> calib_all;
  }
  ASSERT_GE(static_cast<int>(calib_all["layers"].size()),
            first_layer + num_layers);

  // WHERE THE RUN STARTS. Normally the model's own input, but a run can be
  // begun at any layer from the reference's CLEAN residual stream, which is
  // the only way to separate a layer's own noise from what it inherited: the
  // three-layer run lands at 2^-6.45 / 2^-3.05 / 2^-2.92, and a float64
  // perturbation study (`reference/layer_gain.py`) puts a real layer's gain
  // on a relative input perturbation at 0.73 to 0.92 -- CONTRACTING -- so the
  // jump at layer 1 is this pipeline's, not the model's, and it has to be
  // asked of layer 1 alone.
  std::vector<double> x0;
  if (first_layer > 0) {
    ASSERT_TRUE(ReadF64(rdir + "/h_L" + (first_layer < 11 ? "0" : "") +
                            std::to_string(first_layer - 1) + ".f64",
                        static_cast<size_t>(kT) * kH, x0));
    std::cout << "STARTING AT LAYER " << first_layer
              << " from the reference's clean stream" << std::endl;
  } else {
    ASSERT_TRUE(ReadF32(wdir + "/input_nosink.f32",
                        static_cast<size_t>(kT) * kH, x0));
  }

  // AN A/B ON THE SINKS, AND STAGES 1, 1.5 AND 2 ARE VALID UNDER IT.
  //
  // CKKS error is absolute, so a ciphertext's relative precision at an entry
  // is set by the LARGEST entry it holds -- and from layer 1 on the residual
  // stream carries 11.07 at its two sink rows against 0.10 to 1.93 at the 126
  // user rows (`reference/streampeak.py`), because layer 1's feed-forward
  // output is 194x bigger there (`reference/whererows.py`). [SYLPH] 3.1.1
  // removes the sink prefix from the encrypted path for exactly this reason;
  // this tree keeps it at full magnitude and only rescales the norm's
  // ARGUMENT, so `stream_scale = ride / |stream|_max` is set by rows that are
  // not the answer and the user tokens ride at `ride / 78`.
  //
  // DIVIDING the sink rows by a public factor -- not replacing them -- leaves
  // every downstream comparison valid without touching the reference, because
  // RMSNorm is scale invariant PER TOKEN: the norm's output at a sink row is
  // the true one whatever the row was scaled by, so the sink keys and values
  // the attention reads are unchanged, and so is every user row. The norm's
  // own sink rescale absorbs the factor (its argument must not move). The
  // stream is then sized on its own peak, which is the point: dropping the
  // sinks without re-riding measures nothing, since every user row keeps
  // exactly the value it had.
  //
  // The LAYER's closing number is NOT valid under this, because `o` and `y`
  // are not suppressed to match and the residual add would mix conventions.
  // That is what the mechanism has to fix; this measures whether it is worth
  // building.
  double sink_suppress = 1.0;
  if (EnvInt("CHEDDAR_CI_SINK_SUPPRESS", 0) != 0) {
    double smax = 0.0, umax = 0.0;
    for (int t = 0; t < kSinkTokens; t++) {
      for (int c = 0; c < kH; c++) {
        smax = std::max(smax, std::abs(x0[static_cast<size_t>(t) * kH + c]));
      }
    }
    for (int t = kSinkTokens; t < kT; t++) {
      for (int c = 0; c < kH; c++) {
        umax = std::max(umax, std::abs(x0[static_cast<size_t>(t) * kH + c]));
      }
    }
    sink_suppress = std::max(1.0, smax / std::max(umax, 1e-30));
    for (int t = 0; t < kSinkTokens; t++) {
      for (int c = 0; c < kH; c++) {
        x0[static_cast<size_t>(t) * kH + c] /= sink_suppress;
      }
    }
    std::cout << "SINKS SUPPRESSED by " << sink_suppress << " (they were "
              << smax << " against the user rows' " << umax << ")" << std::endl;
  }

  // ---- the four rings, one secret -----------------------------------------
  if (kModule) {
    // `HalfBootModule` needs the SSE secret sparse in the module basis
    // (Doing.md 3.6); set before any Ring samples one, unless given.
    setenv("CHEDDAR_MODULE_SPARSE_SECRET", "128,16", /*overwrite=*/0);
    std::cout << "MODULE BASIS: " << kNumH << " residual and " << kNumHid
              << " hidden ciphertexts, CHEDDAR_MODULE_SPARSE_SECRET="
              << std::getenv("CHEDDAR_MODULE_SPARSE_SECRET") << std::endl;
  }
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

  // THE FEED-FORWARD'S RING. `CHEDDAR_CI_FFN_PARAM` names a landing sub-ladder
  // of ci16_35 (gen_landing.py; `ci16_35_land13c2` climbs to 23 and lands
  // HalfBoot at 13 with a two-level CtS, Doing.md 3.9) for the layer's
  // non-leg half; its levels 0..L are ci16_35's own, so ciphertexts cross
  // between the two rings without a key, while the KEYS are per ring -- the
  // key layout follows the ring's main count -- so that ring gets its own
  // UserInterface on the same secret, and every key the layer uses comes from
  // it. Unset, the FFN's Context is a second BootContext on ci16_35 itself,
  // sharing the leg's keys, as before.
  const char *ffn_param_env = std::getenv("CHEDDAR_CI_FFN_PARAM");
  const std::string ffn_param =
      (ffn_param_env && ffn_param_env[0]) ? ffn_param_env : kBootParam;
  const bool ffn_own_ring = ffn_param != kBootParam;
  Ring boot_ffn(ffn_param, boot.ui->GetSecretCoeffs(), /*slack=*/9,
                /*build_user_interface=*/ffn_own_ring);
  cheddar::UserInterface<word> &fui = ffn_own_ring ? *boot_ffn.ui : *boot.ui;
  const cheddar::EvkMap<word> &fevk = fui.GetEvkMap();
  auto fctx = std::dynamic_pointer_cast<BootContext<word>>(boot_ffn.context);
  ASSERT_NE(fctx, nullptr);
  ASSERT_EQ(fctx->GetBootParameter().GetEvalModEndLevel(),
            boot_ffn.param->default_encryption_level_);
  std::cout << "FFN ring " << ffn_param << ": climbs to "
            << fctx->GetBootParameter().GetMaxLevel() << " ("
            << boot_ffn.param->LevelToNP(fctx->GetBootParameter().GetMaxLevel())
                   .GetNumTotal()
            << " limbs), HalfBoot lands "
            << fctx->GetBootParameter().GetEvalModEndLevel() << ", StC at "
            << fctx->GetBootParameter().GetStCStartLevel()
            << (ffn_own_ring ? ", its own keys" : ", the leg's keys")
            << std::endl;
  if (ffn_own_ring) {
    // The shared bottom, checked: what "keyless" rests on.
    for (int L = 0; L <= boot_ffn.param->default_encryption_level_; L++) {
      ASSERT_EQ(boot.param->GetPrimeVector(boot.param->LevelToNP(L)),
                boot_ffn.param->GetPrimeVector(boot_ffn.param->LevelToNP(L)))
          << "the FFN ring's level " << L << " differs from ci16_35's";
    }
  }

  // ModPack's keys live on the ring that runs the projections: the FFN's.
  const int pack_aux = fui.PrepareModPackKeys(kSmall, kPcmmLevel,
                                              /*num_aux=*/-1);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys;
  for (int j = 0; j < kRank; j++) {
    pack_keys.push_back(&fui.GetModPackKey(kRank, j));
  }

  // THE CROSSING CONSTANT, DERIVED. `HalfBoot` multiplies the message by
  // `level_zero_scale / q0`; `BootContext` computes it exactly and the leg's
  // `restore` is its inverse. Every earlier test measured this by decrypting a
  // ciphertext and fitting, which is right on the preset it was taken on.
  const double crossing = bctx->GetMessageRatio();
  std::cout << "crossing constant " << crossing << " = 2^"
            << std::log2(std::abs(crossing)) << " (derived)" << std::endl;

  // THE LEG'S LANDING RING. `CHEDDAR_CI_LEG_PARAM` names a landing sub-ladder
  // (Doing.md 3.9: `ci16_35_land13`, climb to 25) for the 48 q/k/v HalfBoots
  // -- only RoPE spends a level before `Merge` drops to the exchange at 8 --
  // and the 8 chain-output Boots, which feed the seam at its input level.
  // The 8 score Boots stay on ci16_35: the softmax walk needs exactly 16 -> 3.
  // Same secret, its own keys; ciphertexts cross as they are.
  const char *leg_env = std::getenv("CHEDDAR_CI_LEG_PARAM");
  const std::string leg_param = (leg_env && leg_env[0]) ? leg_env : kBootParam;
  const bool leg_own_ring = leg_param != kBootParam;
  std::unique_ptr<Ring> leg_land;
  std::shared_ptr<BootContext<word>> lctx = bctx;
  const cheddar::EvkMap<word> *levk = &boot.ui->GetEvkMap();
  if (leg_own_ring) {
    leg_land = std::make_unique<Ring>(leg_param, boot.ui->GetSecretCoeffs(),
                                      /*slack=*/0);
    lctx = std::dynamic_pointer_cast<BootContext<word>>(leg_land->context);
    ASSERT_NE(lctx, nullptr);
    lctx->PrepareEvalMod();
    lctx->PrepareEvalSpecialFFT(num_slots);
    EvkRequest req;
    lctx->AddRequiredRotations(req, num_slots, min_ks);
    leg_land->ui->PrepareRotationKey(req);
    levk = &leg_land->ui->GetEvkMap();
    ASSERT_NEAR(lctx->GetMessageRatio() / crossing, 1.0, 1e-9)
        << "the landing ring's crossing constant differs from ci16_35's";
    for (int L = 0; L <= leg_land->param->default_encryption_level_; L++) {
      ASSERT_EQ(boot.param->GetPrimeVector(boot.param->LevelToNP(L)),
                leg_land->param->GetPrimeVector(leg_land->param->LevelToNP(L)))
          << "the leg's landing ring differs from ci16_35 at level " << L;
    }
    std::cout << "leg landing ring " << leg_param << ": climbs to "
              << lctx->GetBootParameter().GetMaxLevel() << " ("
              << leg_land->param
                     ->LevelToNP(lctx->GetBootParameter().GetMaxLevel())
                     .GetNumTotal()
              << " limbs), HalfBoot lands "
              << lctx->GetBootParameter().GetEvalModEndLevel()
              << ", Boot lands " << lctx->GetBootParameter().GetEndLevel()
              << std::endl;
  }

  typename cheddar::CiSinCAttention<word>::Config acfg;
  acfg.restore = 1.0 / crossing;
  acfg.rope_base = kRopeTheta;
  // Where the q/k/v HalfBoots land, which is where the RoPE masks are encoded.
  acfg.land_level = lctx->GetBootParameter().GetEvalModEndLevel();
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
  if (!ffn_own_ring) {
    // The same ring as the leg's: the ModPack keys were prepared through the
    // leg's UserInterface, so this Context needs the narrow basis they use.
    boot_ffn.context->PrepareNarrowKeySwitch(kPcmmLevel, pack_aux);
  }
  fctx->PrepareEvalMod();
  // The native CtS/StC tables: donated by the leg's Context when the two
  // share a ring (the tables are encoded against the primes), compiled here
  // otherwise.
  fctx->PrepareEvalSpecialFFT(num_slots, cheddar::BootVariant::kNormal,
                              ffn_own_ring ? nullptr : bctx.get());
  {
    EvkRequest req;
    fctx->AddRequiredRotations(req, num_slots, min_ks);
    fui.PrepareRotationKey(req);
  }
  if (kModule && ffn_own_ring) {
    // On the module basis this Context crosses through HalfBootModule, whose
    // CoeffToSlot is the CiModuleBasis; its native CtS tables are never read
    // and the seam's native StC is all it keeps.
    fctx->ReleaseCtS(num_slots);
  }

  typename cheddar::CiLlamaLayer<word>::Config lcfg;
  lcfg.num_tokens = kT;
  lcfg.proj_rank = kRank;
  lcfg.model_declared = kDeclaredH;
  lcfg.hidden_declared = kDeclaredHid;
  lcfg.model_live = kH;
  lcfg.module_basis = kModule;
  lcfg.product_level = kPcmmLevel;
  // SIXTEEN PARENTS A TILE, which is what the full-width cost model prescribes
  // for the direct route (Doing.md 1.5dc: 228.2 ms per output ciphertext at
  // tile 4 against 130.5 at 16). Four was the correctness-width layer's
  // setting, forced by memory at rank 512 and full density; half density
  // halves exactly that, so the two changes pay for each other. A tile costs
  // one extra ModPack per output group -- `rank` key switches -- and the down
  // projection has 56 parents, so at tile 4 it pays fourteen of them.
  lcfg.parents_per_tile = EnvInt("CHEDDAR_PARENTS_PER_TILE", 16);
  lcfg.ride = ride;
  // The ciphertext's epsilon, not the model's; see the calibration below.
  lcfg.eps = 1e-5;   // the MODEL's: the stream factor is divided out first
  lcfg.min_ks = min_ks;
  // Split the norm from the conversion under it. `[stage 0.5]` puts the
  // crossing at 2^-18 to 2^-21 while the norm's coefficient read is 2^-9.3,
  // so ten bits are made between them -- and `SlotToCoeff` is seven hoisted
  // key-switch layers standing right there.
  lcfg.keep_norm_slots = EnvInt("CHEDDAR_CI_NORM_SLOTS", 0) != 0;
  lcfg.verbose = true;
  // `stream_scale` is derived below from the reference, but the layer needs
  // the ciphertext's epsilon at construction, so it is computed here.
  cheddar::CiLlamaLayer<word> layer(fctx, layout, pack_keys, lcfg);
  {
    EvkRequest req;
    layer.AddRequiredRotations(req);
    fui.PrepareRotationKey(req);
  }
  // Both seam halves' rotations up front: the stages themselves are built per
  // half inside the loop, but their keys are the same set every layer.
  for (int half = 0; half < (kModule ? 1 : 2); half++) {
    layer.PrepareSeamHalf(half);
    EvkRequest req;
    layer.AddSeamHalfRotations(req);
    fui.PrepareRotationKey(req);
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
  // NOT HERE ANY MORE. Layer 0's `attn_sink` does it at the crossing, exactly
  // as every other layer's does, so the stream this test encrypts is the raw
  // model quantity and the rescale happens in one place. Doing both applies
  // the factor twice.
  (void)rescale_sinks;

  // ONE FACTOR FOR THE WHOLE RUN, sized on the largest residual any layer
  // reaches. The residual add forces the stream and the two projections that
  // write it to agree, and RMSNorm makes everything downstream scale free, so
  // a per-layer factor would have to be re-derived at every add for nothing.
  // The reference already knows every layer's magnitude.
  double x_absmax = 0.0;
  for (double v : x0) x_absmax = std::max(x_absmax, std::abs(v));
  double stream_absmax = x_absmax;
  // With the sinks replaced the reference's peaks describe a different
  // stream, and the RIDE is the whole point: dropping them without raising
  // the ride leaves every user row at exactly the value it had, so it would
  // measure nothing. The stream is sized on its own peak instead.
  const bool sinks_suppressed = sink_suppress != 1.0;
  for (int L = first_layer; L < first_layer + num_layers && !sinks_suppressed; L++) {
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
          comp[Rev(c, 9)][Pos(t)] =
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

  // ---- STAGE 0: the stream, read back the way its consumer reads it -------
  //
  // Before anything else runs. `Components` is the banded inverse that
  // `ModDecomp` performs and that every downstream read here uses, so if the
  // encoding and this read do not agree on a freshly encrypted ciphertext
  // nothing measured further along means anything.
  {
    double n0 = 0.0, d0 = 0.0, m0 = 0.0, e0 = 0.0;
    std::vector<std::vector<std::vector<double>>> g0(kNumH);
    for (int k = 0; k < kNumH; k++) {
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, stream[k]);
      std::vector<double> co;
      boot_ffn.context->encoder_.DecodeCoeff(co, pt);
      g0[k] = Components(co);
    }
    for (int m = 0; m < kH; m++) {
      const int k = m / kPerModel;
      const int c = ModelSlot(m) - k * kRank;
      for (int t = 0; t < kT; t++) {
        const double wv = stream_scale * x0[static_cast<size_t>(t) * kH + m];
        n0 += g0[k][Rev(c, 9)][Pos(t)] * wv;
        d0 += wv * wv;
        m0 = std::max(m0, std::abs(wv));
      }
    }
    const double f0 = n0 / d0;
    for (int m = 0; m < kH; m++) {
      const int k = m / kPerModel;
      const int c = ModelSlot(m) - k * kRank;
      for (int t = 0; t < kT; t++) {
        e0 = std::max(e0,
                      std::abs(g0[k][Rev(c, 9)][Pos(t)] / f0 -
                               stream_scale * x0[static_cast<size_t>(t) * kH + m]));
      }
    }
    std::cout << "  [stage 0] the encrypted stream: " << (e0 / m0) << " = 2^"
              << std::log2(e0 / m0) << ", carried " << f0 << std::endl;
    if (stop_after == -1) return;
  }

  // ---- STAGE 0.5: THE CROSSING, ALONE AND AT THE MODEL'S OWN WIDTH ---------
  //
  // Doing.md 1.5cs measured RMSNorm at 2^-10.78 through a crossing against
  // 2^-13.47 with no bootstrap in front of it, showed the gap does not move
  // across 3.85 bits of bootstrap precision (`ci16_40` against `ci16_35`),
  // called it deterministic and left it unattributed. This asks the crossing
  // the question directly: HalfBoot the stream that stage 0 just verified at
  // 2^-21 and read it back in SLOTS, fitting one global factor. Nothing at
  // all is between the two reads, so what this prints is the crossing's own
  // error on this data at this ride -- and the norm's 2^-9.3 is either
  // explained by it or is not.
  //
  // The slot address is the one the whole branch runs on: a coefficient at
  // (position p, component I) is slot `Rev(I, 9) * num_tokens + Rev(p, 7)`,
  // and `Pos(t)` is the position convention of 1.5du.
  if (EnvInt("CHEDDAR_CI_CROSSING_PROBE", 0) != 0) {
    std::vector<std::vector<Complex>> sl(kNumH);
    for (int k = 0; k < kNumH; k++) {
      Ciphertext<word> cx;
      bctx->HalfBoot(cx, stream[k], boot.ui->GetEvkMap(), min_ks);
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, cx);
      boot.context->encoder_.Decode(sl[k], pt);
    }
    double n = 0.0, d = 0.0, mx = 0.0, e = 0.0, q = 0.0;
    for (int m = 0; m < kH; m++) {
      const int k = m / kPerModel;
      const int c = ModelSlot(m) - k * kRank;
      for (int t = 0; t < kT; t++) {
        const size_t sx = static_cast<size_t>(c) * kT + Rev(Pos(t), 7);
        const double wv = stream_scale * x0[static_cast<size_t>(t) * kH + m];
        n += sl[k][sx].real() * wv;
        d += wv * wv;
        mx = std::max(mx, std::abs(wv));
      }
    }
    const double f = n / d;
    for (int m = 0; m < kH; m++) {
      const int k = m / kPerModel;
      const int c = ModelSlot(m) - k * kRank;
      for (int t = 0; t < kT; t++) {
        const size_t sx = static_cast<size_t>(c) * kT + Rev(Pos(t), 7);
        const double dv = sl[k][sx].real() / f -
                          stream_scale * x0[static_cast<size_t>(t) * kH + m];
        e = std::max(e, std::abs(dv));
        q += dv * dv;
      }
    }
    std::cout << "  [stage 0.5] the crossing ALONE, in slots: " << (e / mx)
              << " = 2^" << std::log2(e / mx) << ", rms 2^"
              << 0.5 * std::log2(q / d) << ", carried " << f
              << " against the message ratio " << crossing << std::endl;
    std::cout << "             ride: the stream reaches "
              << (stream_scale * x_absmax) << std::endl;
    return;
  }

  // ---- the layers ----------------------------------------------------------
  bctx->ResetBootCounts();
  fctx->ResetBootCounts();
  const auto t_run0 = Tick();

  for (int L = first_layer; L < first_layer + num_layers; L++) {
    const auto t_layer0 = Tick();
    const std::string ld = wdir + "/L" + (L < 10 ? "0" : "") + std::to_string(L);
    // The weight-cache name. A repeated tag with different weights is a wrong
    // layer that still decrypts, so it carries the layer index.
    const std::string tag = "L" + std::to_string(L);
    // MODEL CONVERSION IS NOT INFERENCE. A layer runs each of its seven
    // weights once, so nothing inside it can amortise the plaintext-operand
    // build; [SYLPH] 5.3 does it offline and 5.1 keeps the result resident.
    // The leg times it for itself so the ledger below can subtract it.
    layer.GetProjectionLeg().ResetProjectionTimers();
    layer.ResetPrepareTimer();
    double seam_prep_ms = 0.0;
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

    // The seven projections as the exporter wrote them, on the device. Under
    // `CHEDDAR_CI_HOST_WEIGHTS` these stay empty and the declared matrices
    // below are built instead.
    using Leg = cheddar::CoeffLinearLeg<word>;
    using DW = typename Leg::DeviceWeights;
    const bool host_weights = HostWeightsFromEnv();
    struct Tensor {
      cheddar::HostVector<float> host;
      cheddar::DeviceVector<float> dev;
      int rows = 0, cols = 0;
    };
    Tensor tq, tk, tv, to, tg, tu, td;
    auto load = [&](Tensor &t, const char *name, int rows, int cols) {
      t.rows = rows;
      t.cols = cols;
      if (host_weights) return true;
      if (!ReadF32Host(ld + "/" + name, static_cast<size_t>(rows) * cols,
                       t.host)) {
        return false;
      }
      t.dev.resize(static_cast<int>(t.host.size()));
      cheddar::CopyHostToDevice(t.dev, t.host);
      return true;
    };
    ASSERT_TRUE(load(tq, "wq.f32", kH, kH));
    ASSERT_TRUE(load(tk, "wk.f32", kH, kKv));
    ASSERT_TRUE(load(tv, "wv.f32", kH, kKv));
    ASSERT_TRUE(load(to, "wo.f32", kH, kH));
    ASSERT_TRUE(load(tg, "wgate.f32", kH, kI));
    ASSERT_TRUE(load(tu, "wup.f32", kH, kI));
    ASSERT_TRUE(load(td, "wdown.f32", kI, kH));
    auto device_weights = [&](const Tensor &t, const std::vector<int> &in_slot,
                              const std::vector<int> &out_slot, double scale) {
      DW dw;
      dw.data = &t.dev;
      dw.in_live = t.rows;
      dw.out_live = t.cols;
      dw.in_slot = &in_slot;
      dw.out_slot = &out_slot;
      dw.fingerprint = Leg::Fingerprint(t.host.data(), t.host.size(), scale);
      return dw;
    };
    // The declared -> live maps the layer states once.
    std::vector<int> model_map, hidden_map;
    cheddar::CiLlamaLayer<word>::ModelMap(model_map, kDeclaredH, kRank, kH,
                                          kDensity);
    cheddar::CiLlamaLayer<word>::HiddenMap(hidden_map, kDeclaredHid, kRank,
                                           kI, kDensity);

    // ---- the pre-attention norm, encrypted --------------------------------
    typename cheddar::CiLlamaLayer<word>::Calibration cal;
    // ALPHA IS ABOUT THE CIPHERTEXT'S MAGNITUDE, NOT THE MODEL'S.
    // `RmsNormHandler` wants `alpha * mean(x^2)` near 1 for the `x` it is
    // handed, and the stream carries `stream_scale`, so its mean square is
    // `stream_scale^2` smaller than the model's. Left alone the invsqrt's
    // argument lands at `stream_scale^2` -- 0.0038 here -- which is far
    // outside any window, and outside its interval a Chebyshev fit grows like
    // cosh(d arccosh(v)): the first run of this test came back at |.| ~ 400
    // against a reference of 0.37, with a NEGATIVE fitted factor. The window
    // is a RATIO and so needs no such correction.
    // ALPHA AND EPS ARE BOTH ABOUT THE CIPHERTEXT, AND THEY MOVE TOGETHER.
    // `RmsNormHandler` evaluates `1/sqrt(alpha * (S/n + eps))` on the `x` it
    // is handed, and the stream carries `stream_scale`, so `S` is `s^2` times
    // the model's. `alpha / s^2` puts the argument back at 1 -- and `eps` is
    // an ADDITIVE term in the same bracket, so it has to be `s^2 * eps_model`
    // or `alpha/s^2` inflates it by `1/s^2` (260x here) and it dominates the
    // bracket outright. Scaling alpha alone measured relative 829; leaving
    // both alone measured 0.246 at the norm, with the invsqrt evaluated at
    // 0.0038 where its window is [0.77, 1.3] and a Chebyshev fit does
    // whatever it likes.
    //
    // The OUTPUT stays in model units either way: the weight carries
    // `sqrt(alpha)` and the bracket `alpha`, so they cancel exactly.
    cal.attn_alpha = cj["attn_alpha"].get<double>();
    cal.attn_norm_window = cj["attn_norm_window"].get<double>();
    cal.alpha = cj["alpha"].get<double>();
    cal.norm_window = cj["norm_window"].get<double>();
    cal.silu_range = cj["silu_range"].get<double>();
    cal.stream_scale = stream_scale;
    // [SYLPH] 3.1.1, at BOTH norms. Without it a 32-layer run dies at layer 1:
    // the sink rows leave layer 1 at 74327x the user rows' mean square and the
    // invsqrt is handed an argument no window contains (measured: relative
    // 94.5, `carried 1693` against a stream scale of 0.019).
    cal.attn_sink.assign(kT, 1.0);
    cal.ffn_sink.assign(kT, 1.0);
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
    // The stream's sink rows were divided by `sink_suppress`, so the norm's
    // rescale has to multiply it back or the polynomial's argument moves --
    // and the whole point is that it does not.
    for (int t = 0; t < kSinkTokens && L == first_layer; t++) {
      cal.attn_sink[t] *= sink_suppress;
    }
    // SUPPRESS THE FEED-FORWARD'S OWN SINK ROWS. `CHEDDAR_CI_SINK_SUPPRESS`
    // rescales what the layer is HANDED and measured nothing (131x of ride,
    // 0.001 bits); this rescales what the layer MAKES, which at layer 1 is
    // 194x bigger at the sinks than at the user rows. The two decide between
    // the only two models left -- noise made per token, or noise sized by the
    // ciphertext's largest entry -- and the layer's closing number is on the
    // user rows either way.
    const double y_sup = EnvDouble("CHEDDAR_CI_Y_SUPPRESS", 1.0);
    if (y_sup != 1.0) {
      cal.up_sink.assign(kT, 1.0);
      for (int t = 0; t < kSinkTokens; t++) cal.up_sink[t] = 1.0 / y_sup;
      if (L == first_layer) {
        std::cout << "THE FFN's SINK ROWS SUPPRESSED by " << y_sup << std::endl;
      }
    }

    std::vector<double> an_dec(kDeclaredH, 0.0), fn_dec(kDeclaredH, 0.0);
    for (int c = 0; c < kH; c++) {
      an_dec[ModelSlot(c)] = an_f[c];
      fn_dec[ModelSlot(c)] = fn_f[c];
    }

    std::vector<Ciphertext<word>> normed;
    layer.AttentionNorm(normed, stream, an_dec, cal, fevk);
    const auto t_norm = Tick();

    // ---- STAGE 1: the pre-attention norm, against the host ---------------
    //
    // A staged check, because guessing at the closing number costs a run of
    // several minutes each time and this tree has paid that bill repeatedly.
    // `CHEDDAR_CI_STOP_AFTER=1` stops here.
    {
      std::vector<double> want(static_cast<size_t>(kT) * kH, 0.0);
      std::vector<double> hin;
      if (L == 0) {
        hin = x0;
      } else {
        ASSERT_TRUE(ReadF64(rdir + "/h_L" + (L < 11 ? "0" : "") +
                                std::to_string(L - 1) + ".f64",
                            static_cast<size_t>(kT) * kH, hin));
      }
      // The norm's input is the SINK-RESCALED stream: the factor rides the
      // crossing's own multiply, so the host reference has to carry it too.
      // `attn_sink` already carries `sink_suppress` where that is in play,
      // and `hin` is the UNSUPPRESSED reference, so this is the same product.
      for (int t = 0; t < kSinkTokens; t++) {
        for (int c = 0; c < kH; c++) {
          hin[static_cast<size_t>(t) * kH + c] *=
              cal.attn_sink[t] / (L == first_layer ? sink_suppress : 1.0);
        }
      }
      for (int t = 0; t < kT; t++) {
        double sq = 0.0;
        for (int c = 0; c < kH; c++) {
          const double v = hin[static_cast<size_t>(t) * kH + c];
          sq += v * v;
        }
        const double inv = 1.0 / std::sqrt(sq / kH + 1e-5);
        for (int c = 0; c < kH; c++) {
          want[static_cast<size_t>(t) * kH + c] =
              hin[static_cast<size_t>(t) * kH + c] * inv * an_f[c];
        }
      }
      double n2 = 0.0, d2 = 0.0, mx2 = 0.0;
      std::vector<std::vector<std::vector<double>>> g2(kNumH);
      for (int k = 0; k < kNumH; k++) {
        Plaintext<word> pt;
        boot.ui->Decrypt(pt, normed[k]);
        std::vector<double> co;
        boot_ffn.context->encoder_.DecodeCoeff(co, pt);
        g2[k] = Components(co);
      }
      for (int m = 0; m < kH; m++) {
        const int k = m / kPerModel;
        const int c = ModelSlot(m) - k * kRank;
        for (int t = kSinkTokens; t < kT; t++) {
          const double wv = want[static_cast<size_t>(t) * kH + m];
          n2 += g2[k][Rev(c, 9)][Pos(t)] * wv;
          d2 += wv * wv;
          mx2 = std::max(mx2, std::abs(wv));
        }
      }
      const double f2 = n2 / d2;
      double e2 = 0.0, q2 = 0.0;
      for (int m = 0; m < kH; m++) {
        const int k = m / kPerModel;
        const int c = ModelSlot(m) - k * kRank;
        for (int t = kSinkTokens; t < kT; t++) {
          const double d = g2[k][Rev(c, 9)][Pos(t)] / f2 -
                           want[static_cast<size_t>(t) * kH + m];
          e2 = std::max(e2, std::abs(d));
          q2 += d * d;
        }
      }
      // BOTH METRICS, because they say different things. A max over
      // 126 x 4096 entries against a max reference is dominated by the one
      // outlier channel at both ends; the rms ratio is what an operator
      // downstream actually receives, and it is what [SYLPH]'s per-operation
      // bar is stated against.
      std::cout << "  [stage 1] RMSNorm(attn): " << e2 << " against |.| <= "
                << mx2 << " (relative " << (e2 / mx2) << " = 2^"
                << std::log2(e2 / mx2) << ", rms 2^"
                << 0.5 * std::log2(q2 / d2) << "), carried " << f2
                << std::endl;
      // AND THE SAME NORM ONE CONVERSION EARLIER, IN SLOTS. Whatever the
      // difference between this and the coefficient reads below is, it was
      // made by `SlotToCoeff` and by nothing else.
      if (!layer.GetNormSlots().empty()) {
        double n = 0.0, dd = 0.0, mm = 0.0, ee = 0.0, qq = 0.0;
        std::vector<std::vector<Complex>> sv(kNumH);
        for (int k = 0; k < kNumH; k++) {
          Plaintext<word> pt;
          boot.ui->Decrypt(pt, layer.GetNormSlots()[k]);
          boot.context->encoder_.Decode(sv[k], pt);
        }
        for (int m = 0; m < kH; m++) {
          const int k = m / kPerModel;
          const int c = ModelSlot(m) - k * kRank;
          for (int t = kSinkTokens; t < kT; t++) {
            const size_t sx = static_cast<size_t>(c) * kT + Rev(Pos(t), 7);
            const double wv = want[static_cast<size_t>(t) * kH + m];
            n += sv[k][sx].real() * wv;
            dd += wv * wv;
            mm = std::max(mm, std::abs(wv));
          }
        }
        const double ff = n / dd;
        for (int m = 0; m < kH; m++) {
          const int k = m / kPerModel;
          const int c = ModelSlot(m) - k * kRank;
          for (int t = kSinkTokens; t < kT; t++) {
            const size_t sx = static_cast<size_t>(c) * kT + Rev(Pos(t), 7);
            const double dv = sv[k][sx].real() / ff -
                              want[static_cast<size_t>(t) * kH + m];
            ee = std::max(ee, std::abs(dv));
            qq += dv * dv;
          }
        }
        std::cout << "    [stage 1] IN SLOTS, before SlotToCoeff: " << (ee / mm)
                  << " = 2^" << std::log2(ee / mm) << ", rms 2^"
                  << 0.5 * std::log2(qq / dd) << ", carried " << ff
                  << std::endl;
      }
      // AND THE REDUCTION ON ITS OWN. Everything after it is a Chebyshev
      // fit whose value is known in double (`the FIT ALONE` below), so if the
      // sum of squares is already at the norm's own figure the polynomial is
      // exonerated and the square-and-rotate is where the bits go. The
      // reduction broadcasts, so every slot of a band carries its token's
      // sum; channel 2 is the first live one.
      if (!layer.GetNormSlots().empty()) {
        Plaintext<word> pt;
        boot.ui->Decrypt(pt, layer.GetNormAcc());
        std::vector<Complex> sv;
        boot.context->encoder_.Decode(sv, pt);
        std::vector<double> host(kT, 0.0);
        for (int t = 0; t < kT; t++) {
          double q = 0.0;
          for (int cc = 0; cc < kH; cc++) {
            const double v = hin[static_cast<size_t>(t) * kH + cc];
            q += v * v;
          }
          host[t] = q;
        }
        double n = 0.0, dd = 0.0, mm = 0.0, ee = 0.0, qq = 0.0;
        for (int t = kSinkTokens; t < kT; t++) {
          const double got = sv[static_cast<size_t>(2) * kT + Rev(Pos(t), 7)].real();
          n += got * host[t];
          dd += host[t] * host[t];
          mm = std::max(mm, host[t]);
        }
        const double ff = n / dd;
        for (int t = kSinkTokens; t < kT; t++) {
          const double got = sv[static_cast<size_t>(2) * kT + Rev(Pos(t), 7)].real();
          const double dv = got / ff - host[t];
          ee = std::max(ee, std::abs(dv));
          qq += dv * dv;
        }
        std::cout << "    [stage 1] THE REDUCTION ALONE (sum of squares): "
                  << (ee / mm) << " = 2^" << std::log2(ee / mm) << ", rms 2^"
                  << 0.5 * std::log2(qq / dd) << ", carried " << ff
                  << std::endl;
      }
      // AND THE SAME READ WITHOUT THE SCAN. `Components` is an alternating
      // suffix sum, so it mixes the live band with the duplicate band and
      // walks any error the length of the ring. Reading the two bands
      // straight out of the coefficients says which one is wrong -- and the
      // duplicates are not decoration here, they are what makes the image
      // correctly banded for the next `ModDecomp` (1.5cs). The module basis
      // has no band, so the read is the scan and this says nothing there.
      if (!kModule) {
        double lo = 0.0, du = 0.0;
        for (int k = 0; k < kNumH; k++) {
          Plaintext<word> pt;
          boot.ui->Decrypt(pt, normed[k]);
          std::vector<double> co;
          boot_ffn.context->encoder_.DecodeCoeff(co, pt);
          for (int m = k * kPerModel;
               m < std::min(kH, (k + 1) * kPerModel); m++) {
            const int c = ModelSlot(m) - k * kRank;
            const int I = Rev(c, 9);
            const int Id = kRank - I;
            for (int t = kSinkTokens; t < kT; t++) {
              const int p = Pos(t);
              const double wv = want[static_cast<size_t>(t) * kH + m];
              lo = std::max(lo, std::abs(co[static_cast<size_t>(p) * kRank + I] /
                                             f2 - wv));
              // THE DUPLICATE IS ONE POSITION BELOW, NOT ABOVE.
              // `rec[p*rank + I] = comp_I[p] + [I!=0] comp_{rank-I}[p+1]`, so
              // `comp_I[p]` appears a second time at `(p - 1, rank - I)` --
              // which is exactly where `CiLlamaSeam`'s T2 writes it.
              if (p >= 1) {
                du = std::max(
                    du, std::abs(co[static_cast<size_t>(p - 1) * kRank + Id] /
                                     f2 - wv));
              }
            }
          }
        }
        std::cout << "    [stage 1] without the scan: live " << (lo / mx2)
                  << " = 2^" << std::log2(lo / mx2) << ", duplicate "
                  << (du / mx2) << " = 2^" << std::log2(du / mx2) << std::endl;
      }
      // THE FIT ALONE, in double, against the library's OWN compiled
      // polynomial -- `CiFfn.TheFitsAloneExplainTheFfnError`'s method (1.5cu),
      // which capped that test before any ciphertext existed. If the encrypted
      // norm sits at this number the answer is the window and the degree; if
      // it sits well above it, the answer is the circuit and no calibration
      // will move it. The layer answers it, because the handler has to be
      // built at the layer's own operator level and a caller that reassembles
      // that from the outside gets it wrong.
      {
        std::vector<double> msq(kT, 0.0);
        for (int t = 0; t < kT; t++) {
          double s2 = 0.0;
          for (int c = 0; c < kH; c++) {
            const double v = hin[static_cast<size_t>(t) * kH + c];
            s2 += v * v;
          }
          msq[t] = s2 / kH;
        }
        // DOES THE WINDOW CONTAIN THE ARGUMENT? A Chebyshev fit outside its
        // interval does not fail, it returns a plausible wrong number -- and
        // the window rule that stood here covered the argument's RATIO, which
        // is only the same as its REACH when the sample is log-symmetric
        // around `alpha`'s geometric mean. Llama's per-token mean squares are
        // not: measured across the 32 real layers, THIRTEEN ran outside the
        // interval, where the fit goes from ~2^-28 to ~2^-5. One line, at the
        // first stage of the run, is what that class of failure costs to
        // catch.
        {
          const double a = 0.5 * (std::sqrt(cal.attn_norm_window) -
                                  1.0 / std::sqrt(cal.attn_norm_window));
          const double b = 0.5 * (std::sqrt(cal.attn_norm_window) +
                                  1.0 / std::sqrt(cal.attn_norm_window));
          double reach = 0.0;
          for (int t = kSinkTokens; t < kT; t++) {
            reach = std::max(
                reach,
                std::abs((cal.attn_alpha * (msq[t] + 1e-5) - b) / a));
          }
          std::cout << "    [stage 1] the invsqrt argument reaches " << reach
                    << " of its window (" << cal.attn_norm_window << ")"
                    << std::endl;
          EXPECT_LE(reach, 1.0)
              << "the inverse square root is evaluated outside the interval "
              << "it was fitted on, where its error is not the fit's";
        }
        const auto invs = layer.PlainNormInvSqrt(
            cal.attn_alpha, cal.attn_norm_window, msq);
        double q = 0.0, d = 0.0;
        for (int t = kSinkTokens; t < kT; t++) {
          const double inv = invs[t];
          for (int c = 0; c < kH; c++) {
            const double got =
                hin[static_cast<size_t>(t) * kH + c] * inv * an_f[c];
            const double w = want[static_cast<size_t>(t) * kH + c];
            q += (got - w) * (got - w);
            d += w * w;
          }
        }
        std::cout << "    [stage 1] the FIT ALONE, in double, against the "
                  << "library's own polynomial: rms 2^"
                  << 0.5 * std::log2(q / d) << " (window "
                  << cal.attn_norm_window << ")" << std::endl;
      }
      if (stop_after == 1) return;
    }

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

    // ---- Q, K and V, through the layer's own projection leg ---------------
    //
    // THE SAME OPERATOR AS O, GATE, UP AND DOWN. A raw `PcmmHandler` here
    // contracts 8704 components against 512 output rows where 4352 and 256 are
    // live and repeats a ModPack per output group -- four times the work
    // `CoeffLinearLeg` does with `input_density = output_density = 2` and one
    // tile (1.5dd: 228.2 ms an output ciphertext against 49.8). It also has to
    // restate the module/declared bit reversal by hand, which is where this
    // test's own first version went wrong.
    //
    // `w[in_declared][out_declared]`, and the leg reverses both axes: module
    // row `r` of group `g` is declared output `g * rank + rev(r)`. Group
    // `g = 2*l + fam` is half-image (channel group `l`, heads `16*fam ..`), so
    // row `hh*16 + cp` carries head `16*fam + hh`, channel `16*l + cp` --
    // exactly the doorstep `CiSinCAttention` reads. GQA is a weight slice:
    // key/value head `head / 4`.
    std::vector<Ciphertext<word>> ins(kNumH);
    for (int k = 0; k < kNumH; k++) {
      boot_ffn.context->LevelDown(ins[k], normed[k], kPcmmLevel);
    }
    const int kQkvOut = 16 * kRank;
    auto declare_qkv = [&](const std::vector<double> &w, int width,
                           std::vector<double> &out) {
      out.assign(static_cast<size_t>(kDeclaredH) * kQkvOut, 0.0);
      const int heads_w = width / kD;
      for (int g = 0; g < 16; g++) {
        const int l = g / 2, fam = g % 2;
        for (int hh = 0; hh < 16; hh++) {
          for (int cp = 0; cp < 16; cp++) {
            const int row = hh * 16 + cp;
            const int head = fam * 16 + hh;
            const int chan = l * 16 + cp;
            const int o =
                (heads_w == kHeads ? head : head / (kHeads / heads_w)) * kD +
                chan;
            const int oc = g * kRank + Rev(row, 9);
            for (int c = 0; c < kH; c++) {
              out[static_cast<size_t>(ModelSlot(c)) * kQkvOut + oc] =
                  w[static_cast<size_t>(c) * width + o];
            }
          }
        }
      }
    };

    // The same map as `declare_qkv`, as a declared-output -> live-column
    // vector: this is the whole of what the device form needs.
    auto qkv_out_map = [&](int width, std::vector<int> &out_slot) {
      out_slot.assign(kQkvOut, -1);
      const int heads_w = width / kD;
      for (int g = 0; g < 16; g++) {
        const int l = g / 2, fam = g % 2;
        for (int hh = 0; hh < 16; hh++) {
          for (int cp = 0; cp < 16; cp++) {
            const int row = hh * 16 + cp;
            const int head = fam * 16 + hh;
            const int chan = l * 16 + cp;
            out_slot[g * kRank + Rev(row, 9)] =
                (heads_w == kHeads ? head : head / (kHeads / heads_w)) * kD +
                chan;
          }
        }
      }
    };

    std::vector<Ciphertext<word>> q_a(8), q_b(8), k_a(8), k_b(8), v_a(8), v_b(8);
    {
      const char *names[3] = {".q", ".k", ".v"};
      const std::vector<double> *srcs[3] = {&wq_f, &wk_f, &wv_f};
      const Tensor *tens[3] = {&tq, &tk, &tv};
      const int widths[3] = {kH, kKv, kKv};
      const double scales[3] = {cq, ck, cv};
      std::vector<Ciphertext<word>> *dst[3][2] = {
          {&q_a, &q_b}, {&k_a, &k_b}, {&v_a, &v_b}};
      for (int j = 0; j < 3; j++) {
        // The emission reads the norm output at the layer's own density and
        // writes the leg's half-density images (the leg is unchanged under
        // the module basis, Doing.md 3.7 step 4).
        std::vector<Ciphertext<word>> raw;
        typename cheddar::CiLlamaLayer<word>::ProjectionWeight pw;
        std::vector<double> wdec;
        std::vector<int> out_slot;
        DW dw;
        if (host_weights) {
          declare_qkv(*srcs[j], widths[j], wdec);
          pw.host = &wdec;
        } else {
          qkv_out_map(widths[j], out_slot);
          dw = device_weights(*tens[j], model_map, out_slot, scales[j]);
          pw.device = &dw;
        }
        layer.Project(raw, ins, kDeclaredH, kQkvOut, pw, scales[j],
                      (tag + names[j]).c_str(), /*in_density=*/kDensity,
                      /*out_density=*/2);
        ASSERT_EQ(static_cast<int>(raw.size()), 16);
        for (int g = 0; g < 16; g++) {
          raw[g].SetNumSlots(num_slots);
          lctx->HalfBoot((*dst[j][g % 2])[g / 2], raw[g], *levk, min_ks);
        }
      }
    }
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const auto t_proj = Tick();

    // ---- STAGE 1.5: one Q half-image, at the doorstep addresses ----------
    //
    // `CiBootSet.TheLegRunsOnTheRealWeights` passes at full width with the
    // same handler, so the leg is not what is wrong; what differs here is
    // where its operands come from -- an ENCRYPTED half-density norm output
    // rather than a host-normed dense one -- and this is the first place that
    // difference is visible. The doorstep (Doing.md 1.5bx): entry
    // (token t, channel c, head i) sits at slot
    // `rev4(c mod 16) << 12 | rev5(i) << 7 | rev7(t)` of half-image `c / 16`.
    {
      std::vector<double> hin;
      if (L == 0) {
        hin = x0;
      } else {
        ASSERT_TRUE(ReadF64(rdir + "/h_L" + (L < 11 ? "0" : "") +
                                std::to_string(L - 1) + ".f64",
                            static_cast<size_t>(kT) * kH, hin));
      }
      // The norm's input is the SINK-RESCALED stream: the factor rides the
      // crossing's own multiply, so the host reference has to carry it too.
      // `attn_sink` already carries `sink_suppress` where that is in play,
      // and `hin` is the UNSUPPRESSED reference, so this is the same product.
      for (int t = 0; t < kSinkTokens; t++) {
        for (int c = 0; c < kH; c++) {
          hin[static_cast<size_t>(t) * kH + c] *=
              cal.attn_sink[t] / (L == first_layer ? sink_suppress : 1.0);
        }
      }
      std::vector<double> nrm(static_cast<size_t>(kT) * kH, 0.0);
      for (int t = 0; t < kT; t++) {
        double sq = 0.0;
        for (int c = 0; c < kH; c++) {
          const double v = hin[static_cast<size_t>(t) * kH + c];
          sq += v * v;
        }
        const double inv = 1.0 / std::sqrt(sq / kH + 1e-5);
        for (int c = 0; c < kH; c++) {
          nrm[static_cast<size_t>(t) * kH + c] =
              hin[static_cast<size_t>(t) * kH + c] * inv * an_f[c];
        }
      }
      // Q, in the clear, pre-RoPE: the emission carries no RoPE.
      const int l = 0;
      Plaintext<word> pt;
      boot.ui->Decrypt(pt, q_a[l]);
      std::vector<Complex> sl;
      boot.context->encoder_.Decode(sl, pt);
      double nq = 0.0, dq = 0.0, mq = 0.0;
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const int head = hh, chan = l * 16 + cp;
          for (int t = 0; t < kT; t++) {
            double q = 0.0;
            for (int c = 0; c < kH; c++) {
              q += nrm[static_cast<size_t>(t) * kH + c] *
                   wq_f[static_cast<size_t>(c) * kH + head * kD + chan];
            }
            const double wv = cq * q;
            const int slot = (Rev(cp, 4) << 12) | (Rev(hh, 5) << 7) | Rev(t, 7);
            nq += sl[slot].real() * wv;
            dq += wv * wv;
            mq = std::max(mq, std::abs(wv));
          }
        }
      }
      ASSERT_GT(dq, 1e-30) << "the Q reference is zero";
      const double fq = nq / dq;
      double eq = 0.0, qq = 0.0;
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          const int head = hh, chan = l * 16 + cp;
          for (int t = 0; t < kT; t++) {
            double q = 0.0;
            for (int c = 0; c < kH; c++) {
              q += nrm[static_cast<size_t>(t) * kH + c] *
                   wq_f[static_cast<size_t>(c) * kH + head * kD + chan];
            }
            const int slot = (Rev(cp, 4) << 12) | (Rev(hh, 5) << 7) | Rev(t, 7);
            const double d = sl[slot].real() / fq - cq * q;
            eq = std::max(eq, std::abs(d));
            qq += d * d;
          }
        }
      }
      std::cout << "  [stage 1.5] Q half-image 0 at the doorstep: "
                << (eq / mq) << " = 2^" << std::log2(eq / mq) << ", rms 2^"
                << 0.5 * std::log2(qq / dq) << ", carried " << fq
                << " (the crossing is " << crossing << ")" << std::endl;
      if (stop_after == 15) return;

      // SUBSTITUTE EXACT OPERANDS, and see whether the leg still misses.
      // 1.5ej brackets a five-bit term: at this very layer and against this
      // very reference tensor, `CiBootSet.TheLegRunsOnTheRealWeights` reads
      // 2^-9.45 while stage 2 below reads 2^-4.14, and the two things between
      // them are measured far too small to carry it (the eight Boots 2^-13.4,
      // the seam's own path 2^-14.4). The difference is what each hands the
      // leg. This replaces the LIVE doorstep entries of Q, K and V with the
      // clear values -- every other slot kept exactly as the pipeline left it,
      // and the per-ciphertext factor FITTED rather than assumed, so no scale
      // convention is being guessed at -- and leaves the rest of the leg
      // untouched. If stage 2 then reads ~2^-9.4 the term is the operands; if
      // it does not, it is the leg's own path in this test.
      if (EnvInt("CHEDDAR_CI_EXACT_QKV", 0) != 0) {
        const std::vector<double> *wsrc[3] = {&wq_f, &wk_f, &wv_f};
        const int wwidth[3] = {kH, kKv, kKv};
        const double wscale[3] = {cq, ck, cv};
        std::vector<Ciphertext<word>> *wdst[3][2] = {
            {&q_a, &q_b}, {&k_a, &k_b}, {&v_a, &v_b}};
        double worst = 0.0;
        for (int j = 0; j < 3; j++) {
          const int heads_w = wwidth[j] / kD;
          for (int fam = 0; fam < 2; fam++) {
            for (int li = 0; li < 8; li++) {
              Ciphertext<word> &ct = (*wdst[j][fam])[li];
              std::vector<Complex> sv;
              {
                Plaintext<word> pt;
                boot.ui->Decrypt(pt, ct);
                boot.context->encoder_.Decode(sv, pt);
              }
              std::vector<double> want(sv.size(), 0.0);
              std::vector<int> at;
              double n = 0.0, d = 0.0;
              for (int cp = 0; cp < 16; cp++) {
                for (int hh = 0; hh < 16; hh++) {
                  const int head = fam * 16 + hh, chan = li * 16 + cp;
                  const int o =
                      (heads_w == kHeads ? head : head / (kHeads / heads_w)) *
                          kD + chan;
                  for (int t = 0; t < kT; t++) {
                    double acc = 0.0;
                    for (int c = 0; c < kH; c++) {
                      acc += nrm[static_cast<size_t>(t) * kH + c] *
                             (*wsrc[j])[static_cast<size_t>(c) * wwidth[j] + o];
                    }
                    const int slot =
                        (Rev(cp, 4) << 12) | (Rev(hh, 5) << 7) | Rev(t, 7);
                    want[slot] = wscale[j] * acc;
                    at.push_back(slot);
                    n += sv[slot].real() * want[slot];
                    d += want[slot] * want[slot];
                  }
                }
              }
              ASSERT_GT(d, 1e-30) << "the exact operand is zero";
              const double f = n / d;
              double e = 0.0, mx = 0.0;
              for (int slot : at) {
                e = std::max(e, std::abs(sv[slot].real() / f - want[slot]));
                mx = std::max(mx, std::abs(want[slot]));
                sv[slot] = Complex(f * want[slot], 0.0);
              }
              worst = std::max(worst, e / mx);
              const int lv = boot.param->NPToLevel(ct.GetNP());
              Plaintext<word> npt;
              boot.context->encoder_.Encode(npt, lv, ct.GetScale(), sv);
              Ciphertext<word> fresh;
              boot.ui->Encrypt(fresh, npt);
              boot.context->Copy(ct, fresh);
            }
          }
        }
        std::cout << "  [stage 1.5] EXACT Q/K/V substituted at the doorstep; "
                  << "what they replaced was off by at most " << worst << " = 2^"
                  << std::log2(worst) << std::endl;
      }
    }

    // ---- the softmax calibration, in chain units --------------------------
    typename cheddar::CiSinCAttention<word>::SoftMaxCalibration sc;
    sc.m_eff = m_eff;
    sc.span = cqk * span_raw;
    sc.shift = cqk * s_raw_max;
    sc.norm_lo = 0.9;
    sc.norm_hi = 1.1;
    // 0 = derive it from `m_eff`, which is per layer and runs 17.1 to 97.3
    // across the real 32: 15 is right for 31 of them and buys 8.5 bits at
    // layer 31.
    sc.exp_degree = 0;
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
    //
    // STAGE 1.75: WHAT THE EIGHT BOOTS COST. 1.5cw named this the one element
    // between the leg and the banded image that nobody had measured -- the
    // leg's own figure is 2^-9.5 at full width and the seam's own path is
    // 2^-14.4, and stage 2 below sits far under both, so the difference is
    // either here or in the composition. It needs no layout knowledge at all:
    // `Boot` is message preserving, so the same ciphertext before and after
    // is the same message and one fitted factor covers the scale. The read is
    // in COEFFICIENTS because that is the form the leg leaves and the seam
    // takes.
    std::vector<Ciphertext<word>> booted(layout.num_cts);
    {
      double n = 0.0, d = 0.0, mx = 0.0, e = 0.0, q = 0.0, in_mx = 0.0;
      for (int bi = 0; bi < layout.num_cts; bi++) {
        std::vector<double> before, after;
        {
          Plaintext<word> pt;
          boot.ui->Decrypt(pt, attn_out[bi]);
          boot.context->encoder_.DecodeCoeff(before, pt);
        }
        // On the leg's landing ring these land at `GetEndLevel()`, which must
        // be at or above where the seam starts.
        lctx->Boot(booted[bi], attn_out[bi], *levk, min_ks);
        if (bi == 0) {
          ASSERT_GE(boot.param->NPToLevel(booted[bi].GetNP()),
                    layer.GetSeamInputLevel())
              << "the chain-output Boots land below the seam's input level";
        }
        {
          Plaintext<word> pt;
          boot.ui->Decrypt(pt, booted[bi]);
          boot.context->encoder_.DecodeCoeff(after, pt);
        }
        ASSERT_EQ(before.size(), after.size());
        for (size_t i = 0; i < before.size(); i++) {
          n += after[i] * before[i];
          d += before[i] * before[i];
          mx = std::max(mx, std::abs(before[i]));
          in_mx = std::max(in_mx, std::abs(after[i]));
        }
      }
      const double f = n / d;
      for (int bi = 0; bi < layout.num_cts; bi++) {
        std::vector<double> before, after;
        {
          Plaintext<word> pt;
          boot.ui->Decrypt(pt, attn_out[bi]);
          boot.context->encoder_.DecodeCoeff(before, pt);
        }
        {
          Plaintext<word> pt;
          boot.ui->Decrypt(pt, booted[bi]);
          boot.context->encoder_.DecodeCoeff(after, pt);
        }
        for (size_t i = 0; i < before.size(); i++) {
          const double dv = after[i] / f - before[i];
          e = std::max(e, std::abs(dv));
          q += dv * dv;
        }
      }
      std::cout << "  [stage 1.75] the eight Boots on the attention output: "
                << (e / mx) << " = 2^" << std::log2(e / mx) << ", rms 2^"
                << 0.5 * std::log2(q / d) << ", carried " << f
                << " -- the leg leaves |.| <= " << mx << ", Boot returns |.| <= "
                << in_mx << std::endl;
    }
    attn_out.clear();
    // THE SEAM'S IMAGES: two half-density ones per booted ciphertext on the
    // banded image, one dense one on the module basis (Doing.md 3.7 step 3,
    // T2 gone). `seam_ct`/`seam_chan` state where (bi, col, lane) lands.
    const int seam_halves = kModule ? 1 : 2;
    auto seam_ct = [&](int bi, int lane) {
      return kModule ? bi : 2 * bi + lane / 16;
    };
    auto seam_chan = [&](int col, int lane) {
      return Rev(col, 4) * 32 + Rev(kModule ? lane : lane % 16, 5);
    };
    std::vector<Ciphertext<word>> h_cts(seam_halves * layout.num_cts);
    for (int half = 0; half < seam_halves; half++) {
      // T1's three stages are compiled per half and dropped, so this is one
      // more piece of per-layer preparation sitting inside an online row --
      // the same stages every layer, held apart only because each is the
      // largest object the seam owns.
      const auto sp0 = Tick();
      layer.PrepareSeamHalf(half);
      seam_prep_ms += Ms(sp0, Tick());
      for (int bi = 0; bi < layout.num_cts; bi++) {
        layer.Seam(h_cts[bi * seam_halves + half], booted[bi], fevk);
      }
    }
    layer.DropSeamHalf();
    booted.clear();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    const auto t_seam = Tick();

    // ---- STAGE 2: the seam's images, against the clear attention output ---
    //
    // The leg hands its result back carrying a factor of its own -- the gamma
    // fold of Doing.md 1.5cb, measured at 2.185 by the correctness-width layer
    // test, which fitted it in run and called it calibration. It is what sizes
    // the residual add, so it is fitted here the same way, on the ONE
    // quantity that cannot be derived from a BootParameter.
    double o_carried = 1.0;
    {
      std::vector<double> av;
      ASSERT_TRUE(ReadF64(rdir + "/av_L" + (L < 10 ? "0" : "") +
                              std::to_string(L) + ".f64",
                          static_cast<size_t>(kT) * kH, av));
      std::vector<std::vector<std::vector<double>>> ga(h_cts.size());
      for (size_t k = 0; k < h_cts.size(); k++) {
        Plaintext<word> pt;
        boot.ui->Decrypt(pt, h_cts[k]);
        std::vector<double> co;
        boot_ffn.context->encoder_.DecodeCoeff(co, pt);
        ga[k] = Components(co);
      }
      double na = 0.0, da = 0.0, ma = 0.0;
      for (int bi = 0; bi < layout.num_cts; bi++) {
        for (int col = 0; col < layout.rank; col++) {
          for (int lane = 0; lane < layout.lanes; lane++) {
            const int k = seam_ct(bi, lane);
            const int cc = seam_chan(col, lane);
            const int head = Rev(lane, 5);
            const int chan = bi * layout.rank + col;
            for (int t = kSinkTokens; t < kT; t++) {
              const double wv =
                  cv * av[static_cast<size_t>(t) * kH + head * kD + chan];
              na += ga[k][Rev(cc, 9)][Pos(t)] * wv;
              da += wv * wv;
              ma = std::max(ma, std::abs(wv));
            }
          }
        }
      }
      ASSERT_GT(da, 1e-20) << "the attention reference is zero";
      o_carried = na / da;
      double ea = 0.0, qa = 0.0;
      for (int bi = 0; bi < layout.num_cts; bi++) {
        for (int col = 0; col < layout.rank; col++) {
          for (int lane = 0; lane < layout.lanes; lane++) {
            const int k = seam_ct(bi, lane);
            const int cc = seam_chan(col, lane);
            const int head = Rev(lane, 5);
            const int chan = bi * layout.rank + col;
            for (int t = kSinkTokens; t < kT; t++) {
              const double d =
                  ga[k][Rev(cc, 9)][Pos(t)] / o_carried -
                  cv * av[static_cast<size_t>(t) * kH + head * kD + chan];
              ea = std::max(ea, std::abs(d));
              qa += d * d;
            }
          }
        }
      }
      std::cout << "  [stage 2] the seam's images vs the clear attention: "
                << (ea / ma) << " = 2^" << std::log2(ea / ma) << ", rms 2^"
                << 0.5 * std::log2(qa / da) << ", carried " << o_carried
                << " (|ref| <= " << ma << ")" << std::endl;
      // AND THE SAME IMAGES WITHOUT THE SCAN, as stage 1 reads the norm.
      // `Components` is an alternating suffix sum over the whole ring, so it
      // mixes the live band with the duplicate band; stage 1 measures 1.3
      // bits between the two reads. If stage 2's gap against the leg's own
      // 2^-9.5 is the scan it shows here, and if it is not, the two bands
      // say whether the image is correctly banded -- two bands failing
      // IDENTICALLY is a read at the wrong address, not a wrong image
      // (1.5du).
      if (!kModule) {
        std::vector<std::vector<double>> raw(h_cts.size());
        for (size_t k = 0; k < h_cts.size(); k++) {
          Plaintext<word> pt;
          boot.ui->Decrypt(pt, h_cts[k]);
          boot_ffn.context->encoder_.DecodeCoeff(raw[k], pt);
        }
        double lo = 0.0, du = 0.0, ql = 0.0, qd = 0.0;
        for (int bi = 0; bi < layout.num_cts; bi++) {
          for (int col = 0; col < layout.rank; col++) {
            for (int lane = 0; lane < layout.lanes; lane++) {
              const int k = seam_ct(bi, lane);
              const int cc = seam_chan(col, lane);
              const int head = Rev(lane, 5);
              const int chan = bi * layout.rank + col;
              const int Id = Rev(cc, 9);
              for (int t = kSinkTokens; t < kT; t++) {
                const int p = Pos(t);
                const double wv =
                    cv * av[static_cast<size_t>(t) * kH + head * kD + chan];
                const double gl =
                    raw[k][static_cast<size_t>(p) * kRank + Id] / o_carried;
                lo = std::max(lo, std::abs(gl - wv));
                ql += (gl - wv) * (gl - wv);
                // The banded convention puts the duplicate one position BACK
                // (1.5du): a dead component at `p - 1` carries the partner
                // whose live copy is at `p`.
                if (p >= 1 && Id != 0) {
                  const double gd =
                      raw[k][static_cast<size_t>(p - 1) * kRank +
                             (kRank - Id)] / o_carried;
                  du = std::max(du, std::abs(gd - wv));
                  qd += (gd - wv) * (gd - wv);
                }
              }
            }
          }
        }
        std::cout << "    [stage 2] without the scan: live " << (lo / ma)
                  << " = 2^" << std::log2(lo / ma) << " (rms 2^"
                  << 0.5 * std::log2(ql / da) << "), duplicate " << (du / ma)
                  << " = 2^" << std::log2(du / ma) << " (rms 2^"
                  << 0.5 * std::log2(qd / da) << ")" << std::endl;
      }
      // IS IT NOISE OR IS IT A CONSTANT? The max and the rms above come out
      // within 0.07 bits of each other, which noise does not do -- a max over
      // 126 x 4096 entries is normally several bits above the rms. So the
      // same residual is refitted with one factor PER HEAD and one PER TOKEN.
      // If either collapses it, the cause is a calibration constant with that
      // index (the softmax's `row_norm` is per (lane, row), its `row_shift`
      // per row), and not the leg's arithmetic.
      {
        const int kHeadsN = kH / kD;
        std::vector<double> nh(kHeadsN, 0.0), dh(kHeadsN, 0.0);
        std::vector<double> nt(kT, 0.0), dt(kT, 0.0);
        for (int bi = 0; bi < layout.num_cts; bi++) {
          for (int col = 0; col < layout.rank; col++) {
            for (int lane = 0; lane < layout.lanes; lane++) {
              const int k = seam_ct(bi, lane);
              const int cc = seam_chan(col, lane);
              const int head = Rev(lane, 5);
              const int chan = bi * layout.rank + col;
              for (int t = kSinkTokens; t < kT; t++) {
                const double got = ga[k][Rev(cc, 9)][Pos(t)];
                const double wv =
                    cv * av[static_cast<size_t>(t) * kH + head * kD + chan];
                nh[head] += got * wv;  dh[head] += wv * wv;
                nt[t] += got * wv;     dt[t] += wv * wv;
              }
            }
          }
        }
        double qh = 0.0, qt = 0.0;
        for (int bi = 0; bi < layout.num_cts; bi++) {
          for (int col = 0; col < layout.rank; col++) {
            for (int lane = 0; lane < layout.lanes; lane++) {
              const int k = seam_ct(bi, lane);
              const int cc = seam_chan(col, lane);
              const int head = Rev(lane, 5);
              const int chan = bi * layout.rank + col;
              for (int t = kSinkTokens; t < kT; t++) {
                const double got = ga[k][Rev(cc, 9)][Pos(t)];
                const double wv =
                    cv * av[static_cast<size_t>(t) * kH + head * kD + chan];
                const double eh = got / (nh[head] / dh[head]) - wv;
                const double et = got / (nt[t] / dt[t]) - wv;
                qh += eh * eh;  qt += et * et;
              }
            }
          }
        }
        double hlo = 1e300, hhi = -1e300, tlo = 1e300, thi = -1e300;
        for (int i = 0; i < kHeadsN; i++) {
          hlo = std::min(hlo, nh[i] / dh[i]);
          hhi = std::max(hhi, nh[i] / dh[i]);
        }
        for (int t = kSinkTokens; t < kT; t++) {
          tlo = std::min(tlo, nt[t] / dt[t]);
          thi = std::max(thi, nt[t] / dt[t]);
        }
        std::cout << "    [stage 2] refitted PER HEAD: rms 2^"
                  << 0.5 * std::log2(qh / da) << " (factors " << hlo << " .. "
                  << hhi << "); PER TOKEN: rms 2^" << 0.5 * std::log2(qt / da)
                  << " (factors " << tlo << " .. " << thi << ")" << std::endl;
      }
      // WHERE IS THE RESIDUAL? 1.5ek excluded nine external differences
      // between this comparison and `CiBootSet.TheLegRunsOnTheRealWeights`,
      // which reads 2^-9.45 on the same weights, the same tensor, the same
      // rings and the same sizing. If what is left is an ADDRESS fault -- and
      // the live and duplicate bands agreeing to four digits is 1.5du's own
      // signature for one -- the residual will concentrate on particular
      // lanes or column blocks. If it is arithmetic it will not.
      {
        std::vector<double> el(layout.lanes, 0.0), dl(layout.lanes, 0.0);
        std::vector<double> ec(layout.rank, 0.0), dc(layout.rank, 0.0);
        for (int bi = 0; bi < layout.num_cts; bi++) {
          for (int col = 0; col < layout.rank; col++) {
            for (int lane = 0; lane < layout.lanes; lane++) {
              const int k = seam_ct(bi, lane);
              const int cc = seam_chan(col, lane);
              const int head = Rev(lane, 5);
              const int chan = bi * layout.rank + col;
              for (int t = kSinkTokens; t < kT; t++) {
                const double wv =
                    cv * av[static_cast<size_t>(t) * kH + head * kD + chan];
                const double d =
                    ga[k][Rev(cc, 9)][Pos(t)] / o_carried - wv;
                el[lane] += d * d;  dl[lane] += wv * wv;
                ec[col] += d * d;   dc[col] += wv * wv;
              }
            }
          }
        }
        auto worst = [](const std::vector<double> &e,
                        const std::vector<double> &d, const char *tag) {
          double lo = 1e300, hi = -1e300;
          int ilo = 0, ihi = 0;
          for (size_t i = 0; i < e.size(); i++) {
            const double r = 0.5 * std::log2(e[i] / d[i]);
            if (r < lo) { lo = r; ilo = static_cast<int>(i); }
            if (r > hi) { hi = r; ihi = static_cast<int>(i); }
          }
          std::cout << "    [stage 2] by " << tag << ": best 2^" << lo
                    << " (" << ilo << "), worst 2^" << hi << " (" << ihi
                    << "), spread " << (hi - lo) << " bits" << std::endl;
        };
        worst(el, dl, "LANE");
        worst(ec, dc, "COLUMN");
      }
      if (stop_after == 2) return;
    }

    // ---- the O projection, the residual, the FFN --------------------------
    //
    // The O weight takes V's own sizing back out and puts the residual at the
    // ride: the attention output carries `cv`, the stream carries
    // `stream_scale`, and the two must agree before they are added.
    const double o_absmax = cj["o_absmax"].get<double>();
    const double resid_absmax = cj["resid_absmax"].get<double>();
    (void)o_absmax;
    typename cheddar::CiLlamaLayer<word>::Weights lw;
    // The attention output's declared channel: half ciphertext
    // `2*bi + lane/16`, channel `rev4(col)*32 + rev5(lane%16)`, which is
    // exactly `CiLlamaSeam`'s `chan_of` and the order `CoeffLinearLeg` numbers
    // a parent's channels in; its live row of `wo` is `head*128 + chan` with
    // head = rev5(lane), chan = bi*16 + col.
    const int attn_declared = seam_halves * layout.num_cts * kRank;
    std::vector<int> attn_map(attn_declared, -1);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      for (int col = 0; col < layout.rank; col++) {
        for (int lane = 0; lane < layout.lanes; lane++) {
          const int k = seam_ct(bi, lane);
          const int cc = seam_chan(col, lane);
          attn_map[k * kRank + cc] = Rev(lane, 5) * kD + bi * layout.rank + col;
        }
      }
    }
    std::vector<double> wo_dec, wg_dec, wu_dec, wd_dec;
    DW dw_o, dw_g, dw_u, dw_d;
    if (host_weights) {
      wo_dec.assign(static_cast<size_t>(attn_declared) * kDeclaredH, 0.0);
      wg_dec.assign(static_cast<size_t>(kDeclaredH) * kDeclaredHid, 0.0);
      wu_dec = wg_dec;
      wd_dec.assign(static_cast<size_t>(kDeclaredHid) * kDeclaredH, 0.0);
      for (int in_declared = 0; in_declared < attn_declared; in_declared++) {
        const int o_in = attn_map[in_declared];
        if (o_in < 0) continue;
        for (int c = 0; c < kH; c++) {
          wo_dec[static_cast<size_t>(in_declared) * kDeclaredH + ModelSlot(c)] =
              wo_f[static_cast<size_t>(o_in) * kH + c];
        }
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
    }

    // The O projection's factor: divide out V's sizing, then put the residual
    // at the ride. The incoming stream already carries `stream_scale`.
    // The O weight takes V's own sizing back out and puts the result on the
    // stream's factor, so the residual add sees two quantities that agree.
    cal.res_scale = stream_scale / (cv * o_carried);
    // The gate and up are read off a NORMALISED stream, so they are in the
    // model's own units and their ride is set on those.
    cal.gate_scale = ride / std::max(cj["gate_absmax"].get<double>(), 1e-12);
    if (host_weights) {
      lw.o.host = &wo_dec;
      lw.gate.host = &wg_dec;
      lw.up.host = &wu_dec;
      lw.down.host = &wd_dec;
    } else {
      dw_o = device_weights(to, attn_map, model_map, cal.res_scale);
      dw_g = device_weights(tg, model_map, hidden_map, cal.gate_scale);
      dw_u = device_weights(tu, model_map, hidden_map, cal.gate_scale);
      dw_d = device_weights(td, hidden_map, model_map, stream_scale);
      lw.o.device = &dw_o;
      lw.gate.device = &dw_g;
      lw.up.device = &dw_u;
      lw.down.device = &dw_d;
    }
    lw.ffn_norm = &fn_dec;
    lw.tag = tag;

    std::vector<Ciphertext<word>> next;
    layer.FeedForward(next, h_cts, stream, lw, cal, fevk);
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
        num += got[k][Rev(c, 9)][Pos(t)] * want;
        den += want * want;
        absmax = std::max(absmax, std::abs(want));
      }
    }
    ASSERT_GT(den, 1e-12) << "the reference is zero";
    const double fit = num / den;
    double err = 0.0, qerr = 0.0;
    for (int m = 0; m < kH; m++) {
      const int k = m / kPerModel;
      const int c = ModelSlot(m) - k * kRank;
      for (int t = kSinkTokens; t < kT; t++) {
        const double v = got[k][Rev(c, 9)][Pos(t)] / fit;
        const double d = v - ref[static_cast<size_t>(t) * kH + m];
        err = std::max(err, std::abs(d));
        qerr += d * d;
      }
    }
    const auto bl = bctx->GetBootCounts();
    const auto bf = fctx->GetBootCounts();
    std::cout << "LAYER " << L << ": " << err << " against |h| <= " << absmax
              << " (relative " << (err / absmax) << " = 2^"
              << std::log2(err / absmax) << ", rms 2^"
              << 0.5 * std::log2(qerr / den) << "), carried " << fit
              << std::endl;
    std::cout << "  [boot] leg Context full " << bl.full << ", half "
              << bl.half << " | FFN Context full " << bf.full << ", half "
              << bf.half << " | total " << (bl.Total() + bf.Total())
              << std::endl;
    const double conv_ms =
        1000.0 * layer.GetProjectionLeg().GetConvertSeconds();
    const double stage_ms =
        1000.0 * layer.GetProjectionLeg().GetStageSeconds();
    std::cout << "  [time] norm " << Ms(t_layer0, t_norm) << " ms, Q/K/V "
              << Ms(t_norm, t_proj) << ", leg " << Ms(t_proj, t_leg)
              << ", seam " << Ms(t_leg, t_seam) << ", FFN "
              << Ms(t_seam, t_ffn) << ", layer "
              << Ms(t_layer0, t_ffn) / 1000.0 << " s" << std::endl;
    const double prep_ms = 1000.0 * layer.GetPrepareSeconds() + seam_prep_ms;
    std::cout << "  [time] of which PER-LAYER PREPARATION: model conversion "
              << conv_ms << " ms, host staging " << stage_ms
              << " ms, the seam's T1 stages " << seam_prep_ms
              << " ms, the two RMSNorm handlers "
              << 1000.0 * layer.GetPrepareSeconds() << " ms -- so the layer's "
              << "own ARITHMETIC is "
              << (Ms(t_layer0, t_ffn) - conv_ms - stage_ms - prep_ms) / 1000.0
              << " s" << std::endl;
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
