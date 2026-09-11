// BERT-Tiny on the batched layout, against bert_tiny/reference.py's float64.
//
//   BERT_TINY_ALL=<export.py dir> BERT_TINY_REF=<reference.py dir with
//   sim.py's calib.json> [BERT_TINY_LAYERS=2] [BERT_TINY_INSTANCES=1|all]
//   [BERT_TINY_PARAM=ci16_35_k16_w58.json] [BERT_TINY_INPUT_LEVEL=<top>]
//   ./ci_bert_tiny_test
//
// THE BATCH IS ALWAYS FULL. An empty instance is an all-zero prompt, whose
// LayerNorm variance is 0 -- outside every fitted window -- and a Chebyshev
// evaluated outside its interval is ~1e8 there; CKKS noise lives in the
// coefficient domain, so that one instance's garbage pollutes EVERY slot of
// the ciphertext (measured: one hidden channel of the live prompt at 2^+18).
// So by default (`BERT_TINY_INSTANCES=all`) the recorded prompt fills every
// instance, which is also the leak check between instances at no extra cost;
// `BERT_TINY_INPUTS=<inputs.f32>` ([N, T, H], export.py --prompts) fills the
// batch with N DISTINCT prompts (cyclically if N < B), each checked against
// its own row of `h_L{k}.f64` ([N, T, H], reference.py --inputs).
// Layer L reads layer L-1's ENCRYPTED output: the chain is what is measured.
//
// More knobs: `BERT_TINY_MASK=<mask.u8>` ([N, T] bytes, 1 = real token;
// default: the `mask.u8` beside BERT_TINY_INPUTS if it exists, `none` to
// ignore it); `BERT_TINY_HEAD=1` runs the pooler + classifier after the last
// layer and prints every instance's logits and label (checked against
// `cls_logits.f64` [N, 2] when the reference has it); `BERT_TINY_CALIB`
// points at the calibration (default `$BERT_TINY_REF/calib.json`), and a
// reference directory WITHOUT `h_L*.f64` is serve mode: no comparison, just
// the labels (`BERT_TINY_LABELS_OUT=<file>` writes "instance logit0 logit1
// label" lines, `BERT_TINY_PRINT_LABELS=1` prints them all).

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "common/Assert.h"
#include "extension/BootContext.h"
#include "extension/CiBatch.h"
#include "extension/CiBertTiny.h"

using word = uint32_t;
using cheddar::BootContext;
using cheddar::CiBatchLayout;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::Plaintext;
using Ring = ringfixture::Ring<word>;
using Layer = cheddar::CiBertTinyLayer<word>;
using json = nlohmann::json;

namespace {

const char *Env(const char *name, const char *fallback) {
  const char *e = std::getenv(name);
  return (e && e[0]) ? e : fallback;
}
int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return (e && e[0]) ? std::atoi(e) : fallback;
}

bool ReadF32(const std::string &path, size_t count, std::vector<float> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.resize(count);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  return static_cast<size_t>(f.gcount()) == count * sizeof(float);
}
bool ReadF64(const std::string &path, size_t count, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.resize(count);
  f.read(reinterpret_cast<char *>(out.data()),
         static_cast<std::streamsize>(count * sizeof(double)));
  return static_cast<size_t>(f.gcount()) == count * sizeof(double);
}
std::vector<double> ReadVec(const std::string &path, size_t n) {
  std::vector<float> f;
  cheddar::AssertTrue(ReadF32(path, n, f), "cannot read " + path);
  return std::vector<double>(f.begin(), f.end());
}
void ToDevice(cheddar::DeviceVector<float> &d, const std::vector<float> &h) {
  cheddar::HostVector<float> hv(h.size());
  std::copy(h.begin(), h.end(), hv.begin());
  d.resize(static_cast<int>(hv.size()));
  cheddar::CopyHostToDevice(d, hv);
}

Layer::PolySpec Spec(const json &j) {
  Layer::PolySpec s;
  s.lo = j["lo"].get<double>();
  s.hi = j["hi"].get<double>();
  s.degree = j["degree"].get<int>();
  return s;
}
Layer::Calibration::Norm Norm(const json &j) {
  Layer::Calibration::Norm n;
  n.inv = Spec(j);
  n.r_max = j["r_max"].get<double>();
  n.out_absmax = j["out_absmax"].get<double>();
  return n;
}
Layer::Calibration ReadCalib(const json &cj) {
  Layer::Calibration c;
  c.in_absmax = cj["in_absmax"].get<double>();
  const json &sm = cj["softmax"];
  c.niter = sm["niter"].get<int>();
  c.shift = sm["shift"].get<std::vector<std::vector<double>>>();
  c.exp = Spec(sm["exp"]);
  for (const auto &p : sm["inv"]) c.inv.push_back(Spec(p));
  c.ln1 = Norm(cj["ln1"]);
  c.ln2 = Norm(cj["ln2"]);
  c.gelu = Spec(cj["gelu"]);
  c.h_pre_absmax = cj["h_pre_absmax"].get<double>();
  c.z_pre_absmax = cj["z_pre_absmax"].get<double>();
  return c;
}

// One layer's tensors on the device and its vectors, from export.py's dir.
struct LayerFiles {
  cheddar::DeviceVector<float> wq, wk, wv, wo, wint, wout;
  Layer::Weights w;
  void Load(const std::string &d, int H, int I) {
    std::vector<float> f;
    auto mat = [&](const char *name, size_t n, cheddar::DeviceVector<float> &dv) {
      cheddar::AssertTrue(ReadF32(d + "/" + name, n, f), std::string("cannot read ") + name);
      ToDevice(dv, f);
    };
    const size_t HH = static_cast<size_t>(H) * H, HI = static_cast<size_t>(H) * I;
    mat("wq.f32", HH, wq);
    mat("wk.f32", HH, wk);
    mat("wv.f32", HH, wv);
    mat("wo.f32", HH, wo);
    mat("wint.f32", HI, wint);
    mat("wout.f32", HI, wout);
    w.wq = wq.data();
    w.wk = wk.data();
    w.wv = wv.data();
    w.wo = wo.data();
    w.wint = wint.data();
    w.wout = wout.data();
    w.bq = ReadVec(d + "/bq.f32", H);
    w.bk = ReadVec(d + "/bk.f32", H);
    w.bv = ReadVec(d + "/bv.f32", H);
    w.bo = ReadVec(d + "/bo.f32", H);
    w.bint = ReadVec(d + "/bint.f32", I);
    w.bout = ReadVec(d + "/bout.f32", H);
    w.attn_norm = ReadVec(d + "/attn_norm.f32", H);
    w.attn_norm_bias = ReadVec(d + "/attn_norm_bias.f32", H);
    w.ffn_norm = ReadVec(d + "/ffn_norm.f32", H);
    w.ffn_norm_bias = ReadVec(d + "/ffn_norm_bias.f32", H);
  }
};

// `x[n][t][c]`, N prompts, into the live instances' channel ciphertexts
// (instance b holds prompt b % N), carrying `carry`, at `level`.
void EncryptPrompts(Ring &ring, const CiBatchLayout &layout,
                    const std::vector<double> &x, int N, int H, int live,
                    double carry, int level, std::vector<Ciphertext<word>> &cts) {
  const int T = layout.num_tokens;
  const double scale = ring.param->GetScale(level);
  cts.clear();
  cts.resize(H);
  std::vector<double> values(static_cast<size_t>(layout.num_instances) * T, 0.0);
  std::vector<Complex> msg;
  for (int c = 0; c < H; c++) {
    for (int b = 0; b < live; b++) {
      const size_t n = static_cast<size_t>(b % N);
      for (int t = 0; t < T; t++) {
        values[static_cast<size_t>(b) * T + t] =
            carry * x[(n * T + t) * H + c];
      }
    }
    layout.Pack(msg, values);
    Plaintext<word> pt;
    ring.context->gpu_encoder_.Encode(pt, level, scale, msg);
    ring.ui->Encrypt(cts[c], pt);
  }
}

// Decrypt every channel; `y[b][t][c]` in model units (divided by `carry`).
void DecryptAll(Ring &ring, const CiBatchLayout &layout,
                const std::vector<Ciphertext<word>> &cts, int H, int live,
                double carry, std::vector<double> &y) {
  const int T = layout.num_tokens;
  y.assign(static_cast<size_t>(live) * T * H, 0.0);
  std::vector<Complex> msg;
  std::vector<double> values;
  for (int c = 0; c < H; c++) {
    Plaintext<word> pt;
    ring.ui->Decrypt(pt, cts[c]);
    ring.context->encoder_.Decode(msg, pt);
    layout.Unpack(values, msg);
    for (int b = 0; b < live; b++) {
      for (int t = 0; t < T; t++) {
        y[(static_cast<size_t>(b) * T + t) * H + c] =
            values[static_cast<size_t>(b) * T + t] / carry;
      }
    }
  }
}

struct Err {
  double rms_rel = 0.0, max_abs = 0.0, worst_instance = 0.0;
};
// `want` is [N][T][H]; instance b is checked against prompt b % N.
Err Compare(const std::vector<double> &got, const std::vector<double> &want,
            int N, int live, int T, int H) {
  Err e;
  double se = 0.0, sr = 0.0;
  for (int b = 0; b < live; b++) {
    double seb = 0.0, srb = 0.0;
    const size_t n = static_cast<size_t>(b % N);
    for (int t = 0; t < T; t++) {
      for (int c = 0; c < H; c++) {
        const size_t i = (static_cast<size_t>(b) * T + t) * H + c;
        const double w = want[(n * T + t) * H + c];
        const double d = got[i] - w;
        seb += d * d;
        srb += w * w;
        e.max_abs = std::max(e.max_abs, std::abs(d));
      }
    }
    se += seb;
    sr += srb;
    e.worst_instance = std::max(e.worst_instance, std::sqrt(seb / srb));
  }
  e.rms_rel = std::sqrt(se / sr);
  return e;
}
double Bits(double rel) { return -std::log2(std::max(rel, 1e-300)); }

}  // namespace

TEST(CiBertTiny, TheChainRunsOnTheRealWeights) {
#ifndef USE_CUBLAS
  GTEST_SKIP() << "built without cuBLAS";
#else
  const char *all_env = std::getenv("BERT_TINY_ALL");
  const char *ref_env = std::getenv("BERT_TINY_REF");
  if (all_env == nullptr || ref_env == nullptr) {
    GTEST_SKIP() << "BERT_TINY_ALL and BERT_TINY_REF must both be set";
  }
  const std::string ad = all_env, rd = ref_env;
  json meta, calib;
  {
    std::ifstream f(ad + "/meta.json");
    ASSERT_TRUE(f.good()) << ad + "/meta.json";
    meta = json::parse(f);
    const std::string cpath = Env("BERT_TINY_CALIB", (rd + "/calib.json").c_str());
    std::ifstream g(cpath);
    ASSERT_TRUE(g.good()) << cpath;
    calib = json::parse(g);
  }
  const bool have_ref = std::ifstream(rd + "/h_L00.f64").good();
  const bool want_head = EnvInt("BERT_TINY_HEAD", 0) != 0;
  Layer::Config cfg;
  cfg.shape.model = meta["channels"].get<int>();
  cfg.shape.hidden = meta["hidden"].get<int>();
  cfg.shape.heads = meta["heads"].get<int>();
  cfg.shape.head_dim = meta["head_dim"].get<int>();
  cfg.shape.tokens = meta["tokens"].get<int>();
  cfg.shape.eps = meta["ln_eps"].get<double>();
  cfg.boot_group = EnvInt("BERT_TINY_BOOT_GROUP", 8);
  cfg.baby_steps = EnvInt("BERT_TINY_BABY", 0);
  cfg.verbose = EnvInt("BERT_TINY_VERBOSE", 1) != 0;
  const int H = cfg.shape.model, I = cfg.shape.hidden, T = cfg.shape.tokens;
  const int num_layers = std::min(EnvInt("BERT_TINY_LAYERS", meta["layers"].get<int>()),
                                  meta["layers"].get<int>());
  ASSERT_EQ(calib["tokens"].get<int>(), T);

  Ring boot(Env("BERT_TINY_PARAM", "ci16_35_k16_w58.json"));
  auto bctx = std::dynamic_pointer_cast<BootContext<word>>(boot.context);
  ASSERT_NE(bctx, nullptr);
  auto t0 = std::chrono::steady_clock::now();
  Layer layer(bctx, cfg);
  const CiBatchLayout &layout = layer.GetLayout();
  const int live = std::string(Env("BERT_TINY_INSTANCES", "all")) == "all"
                       ? layout.num_instances
                       : std::min(EnvInt("BERT_TINY_INSTANCES", 1), layout.num_instances);
  std::cout << "  BERT-Tiny: " << num_layers << " layer(s), H " << H << ", I " << I
            << ", " << cfg.shape.heads << " heads x " << cfg.shape.head_dim
            << "; T " << T << " x B " << layout.num_instances << " (" << live
            << " live) on " << Env("BERT_TINY_PARAM", "ci16_35_k16_w58.json")
            << std::endl;
  bctx->PrepareEvalMod();
  bctx->PrepareEvalSpecialFFT(layout.num_slots);
  {
    cheddar::EvkRequest req;
    layer.AddRequiredRotations(req);
    boot.ui->PrepareRotationKey(req);
  }
  const auto &evk = boot.ui->GetEvkMap();
  cudaDeviceSynchronize();
  std::cout << "  setup " << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0).count()
            << " s" << std::endl;

  // The input set: the recorded prompt, or BERT_TINY_INPUTS' N prompts.
  const std::string inputs = Env("BERT_TINY_INPUTS", "");
  std::vector<double> x0;
  int N = 1;
  if (inputs.empty()) {
    x0 = ReadVec(ad + "/input.f32", static_cast<size_t>(T) * H);
  } else {
    std::ifstream f(inputs, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(f.good()) << inputs;
    const size_t bytes = static_cast<size_t>(f.tellg());
    N = static_cast<int>(bytes / (static_cast<size_t>(T) * H * sizeof(float)));
    ASSERT_GT(N, 0);
    x0 = ReadVec(inputs, static_cast<size_t>(N) * T * H);
  }
  std::cout << "  " << N << " prompt(s) -> " << live << " instance(s)" << std::endl;
  // The attention mask, if the prompts are padded.
  {
    std::string mpath = Env("BERT_TINY_MASK", "");
    if (mpath.empty() && !inputs.empty()) {
      const std::string beside = inputs.substr(0, inputs.find_last_of("/\\") + 1) + "mask.u8";
      if (std::ifstream(beside).good()) mpath = beside;
    }
    if (!mpath.empty() && mpath != "none") {
      std::ifstream f(mpath, std::ios::binary);
      ASSERT_TRUE(f.good()) << mpath;
      std::vector<uint8_t> m(static_cast<size_t>(N) * T);
      f.read(reinterpret_cast<char *>(m.data()), static_cast<std::streamsize>(m.size()));
      ASSERT_EQ(static_cast<size_t>(f.gcount()), m.size()) << mpath;
      std::vector<uint8_t> valid(static_cast<size_t>(layout.num_instances) * T, 1);
      int shortest = T;
      for (int b = 0; b < layout.num_instances; b++) {
        const size_t n = static_cast<size_t>(b % N);
        int real = 0;
        for (int t = 0; t < T; t++) {
          valid[static_cast<size_t>(b) * T + t] = m[n * T + t];
          real += m[n * T + t] != 0;
        }
        shortest = std::min(shortest, real);
      }
      layer.SetMask(valid);
      std::cout << "  mask " << mpath << ": shortest prompt " << shortest
                << " real tokens of " << T << std::endl;
    }
  }
  // The head's tensors.
  cheddar::DeviceVector<float> pool_w, cls_w;
  Layer::HeadWeights hw;
  Layer::HeadCalibration hc;
  const int last = num_layers - 1;
  if (want_head) {
    ASSERT_TRUE(calib.contains("head") && !calib["head"].is_null())
        << "the calibration has no head section (sim.py on an export with head/)";
    std::vector<float> f;
    ASSERT_TRUE(ReadF32(ad + "/head/pool_w.f32", static_cast<size_t>(H) * H, f));
    ToDevice(pool_w, f);
    hw.pool_b = ReadVec(ad + "/head/pool_b.f32", H);
    hw.cls_b = ReadVec(ad + "/head/cls_b.f32", 2);
    ASSERT_TRUE(ReadF32(ad + "/head/cls_w.f32", static_cast<size_t>(H) * 2, f));
    ToDevice(cls_w, f);
    hw.pool_w = pool_w.data();
    hw.cls_w = cls_w.data();
    hc.tanh = Spec(calib["head"]["tanh"]);
  }
  std::vector<LayerFiles> files(num_layers);
  for (int L = 0; L < num_layers; L++) {
    char d[8];
    std::snprintf(d, sizeof d, "L%02d", L);
    files[L].Load(ad + "/" + d, H, I);
  }
  Layer::Stream in;
  {
    // Prepare layer 0 first: its carry is the calibration's.
    layer.Prepare(files[0].w, ReadCalib(calib["layers"][0]));
    in.carry = layer.InputCarry();
    const int top = layer.TopLevel();
    const int level = std::min(EnvInt("BERT_TINY_INPUT_LEVEL", top), boot.enc_level);
    EncryptPrompts(boot, layout, x0, N, H, live, in.carry, level, in.cts);
    std::cout << "  input at level " << level << ", carry " << in.carry << std::endl;
  }
  // BERT_TINY_DUMP=<dir>: every intermediate the layer taps, decrypted into
  // <dir>/L<L>_<name>.f64 as [ct][live][T] doubles in model units, and an
  // index.txt of "name count factor"; bert_tiny/debug.py compares them
  // against the float64 model's own intermediates.
  const std::string dump = Env("BERT_TINY_DUMP", "");
  std::ofstream index;
  int cur_layer = 0;
  if (!dump.empty()) {
    index.open(dump + "/index.txt");
    ASSERT_TRUE(index.good()) << dump;
    layer.SetProbe([&](const std::string &name,
                       const std::vector<Ciphertext<word>> &cts, double factor) {
      std::vector<double> all;
      DecryptAll(boot, layout, cts, static_cast<int>(cts.size()), live, factor, all);
      // DecryptAll is [b][t][c]; write [c][b][t]
      const int n = static_cast<int>(cts.size());
      std::vector<double> out(static_cast<size_t>(n) * live * T);
      for (int c = 0; c < n; c++) {
        for (int b = 0; b < live; b++) {
          for (int t = 0; t < T; t++) {
            out[(static_cast<size_t>(c) * live + b) * T + t] =
                all[(static_cast<size_t>(b) * T + t) * n + c];
          }
        }
      }
      const std::string fname = "L" + std::to_string(cur_layer) + "_" + name;
      std::ofstream f(dump + "/" + fname + ".f64", std::ios::binary);
      f.write(reinterpret_cast<const char *>(out.data()),
              static_cast<std::streamsize>(out.size() * sizeof(double)));
      index << fname << " " << n << " " << std::setprecision(17) << factor
            << std::endl;
    });
  }
  double worst = 0.0;
  for (int L = 0; L < num_layers; L++) {
    cur_layer = L;
    if (L > 0) layer.Prepare(files[L].w, ReadCalib(calib["layers"][L]));
    Layer::Stream out;
    layer.Layer(out, in, evk);
    const auto &s = layer.GetStages();
    Err e;
    if (have_ref) {
      std::vector<double> got, want;
      DecryptAll(boot, layout, out.cts, H, live, out.carry, got);
      char name[16];
      std::snprintf(name, sizeof name, "/h_L%02d.f64", L);
      ASSERT_TRUE(ReadF64(rd + name, static_cast<size_t>(N) * T * H, want)) << rd + name;
      e = Compare(got, want, N, live, T, H);
    }
    std::cout << "LAYER " << L << ": rms 2^" << std::fixed << std::setprecision(2)
              << -Bits(e.rms_rel) << " (worst instance 2^" << -Bits(e.worst_instance)
              << ", max |err| " << std::setprecision(4) << e.max_abs << ")  "
              << std::setprecision(2) << s.total << " s = boot " << s.boot
              << " (" << s.wide_boots << " wide + " << s.narrow_boots
              << " narrow), qkv " << s.qkv << ", scores " << s.scores
              << ", softmax " << s.softmax << ", values " << s.values << ", o "
              << s.o << ", ln " << s.ln << ", ffn " << s.ffn << ", gelu "
              << s.gelu << "; " << s.rotations << " rotations, " << s.relins
              << " relins" << std::endl;
    worst = std::max(worst, e.rms_rel);
    in = std::move(out);
  }
  if (have_ref) {
    std::cout << "CHAIN worst rms 2^" << std::setprecision(2) << -Bits(worst)
              << " over " << num_layers << " layer(s), " << live << " instance(s)"
              << std::endl;
    EXPECT_LT(worst, std::ldexp(1.0, -3));
  }

  // The head: the pooler + classifier on the last layer's output, the
  // answer of instance b at its [CLS] slot (token 0).
  if (want_head) {
    layer.PrepareHead(hw, hc, calib["layers"][last]["ln2"]["out_absmax"].get<double>());
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Ciphertext<word>> logits;
    layer.Head(logits, in, evk);
    cudaDeviceSynchronize();
    const double head_s = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0).count();
    const int C = static_cast<int>(logits.size());
    std::vector<double> got;
    DecryptAll(boot, layout, logits, C, live, 1.0, got);  // [b][t][c]
    std::vector<double> want;
    const bool have_logits =
        ReadF64(rd + "/cls_logits.f64", static_cast<size_t>(N) * C, want);
    std::ofstream lab_out;
    const std::string lab_path = Env("BERT_TINY_LABELS_OUT", "");
    if (!lab_path.empty()) lab_out.open(lab_path);
    const bool print_all = EnvInt("BERT_TINY_PRINT_LABELS", 0) != 0;
    int agree = 0;
    double se = 0.0, sr = 0.0;
    for (int b = 0; b < live; b++) {
      const size_t n = static_cast<size_t>(b % N);
      int lab = 0, ref_lab = -1;
      for (int c = 1; c < C; c++) {
        if (got[(static_cast<size_t>(b) * T) * C + c] > got[(static_cast<size_t>(b) * T) * C + lab]) lab = c;
      }
      if (have_logits) {
        ref_lab = 0;
        for (int c = 0; c < C; c++) {
          const double g = got[(static_cast<size_t>(b) * T) * C + c], w = want[n * C + c];
          se += (g - w) * (g - w);
          sr += w * w;
          if (w > want[n * C + ref_lab]) ref_lab = c;
        }
        agree += (lab == ref_lab);
      }
      if (b < N && (print_all || b < 8)) {
        std::cout << "  prompt " << b << ": logits";
        for (int c = 0; c < C; c++) {
          std::cout << " " << std::setprecision(4) << got[(static_cast<size_t>(b) * T) * C + c];
        }
        std::cout << " -> label " << lab;
        if (have_logits) {
          std::cout << " (float64";
          for (int c = 0; c < C; c++) std::cout << " " << want[n * C + c];
          std::cout << " -> " << ref_lab << ")";
        }
        std::cout << std::endl;
      }
      if (lab_out.is_open() && b < N) {
        lab_out << b;
        for (int c = 0; c < C; c++) lab_out << " " << got[(static_cast<size_t>(b) * T) * C + c];
        lab_out << " " << lab << "\n";
      }
    }
    std::cout << "HEAD: " << head_s << " s";
    if (have_logits) {
      std::cout << ", logits rms 2^" << std::setprecision(2) << -Bits(std::sqrt(se / sr))
                << ", labels agree " << agree << " / " << live;
      EXPECT_GE(agree, live - live / 50);
    }
    std::cout << std::endl;
  }
#endif
}
