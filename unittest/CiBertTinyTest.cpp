// BERT-Tiny on the batched layout, against bert_tiny/reference.py's float64.
//
//   BERT_TINY_ALL=<export.py dir> BERT_TINY_REF=<reference.py dir with
//   sim.py's calib.json> [BERT_TINY_LAYERS=2] [BERT_TINY_INSTANCES=1|all]
//   [BERT_TINY_PARAM=ci16_35_k16_w58.json] [BERT_TINY_INPUT_LEVEL=<top>]
//   ./ci_bert_tiny_test
//
// One prompt (the recorded one) in instance 0; `BERT_TINY_INSTANCES=all`
// puts the same prompt in every instance of the batch, which checks that
// the instances are independent (a leak between prompts fails it) at no
// extra cost -- the layer's work does not depend on how many are live.
// Layer L reads layer L-1's ENCRYPTED output: the chain is what is measured.

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
  AssertTrue(ReadF32(path, n, f), "cannot read " + path);
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
      AssertTrue(ReadF32(d + "/" + name, n, f), std::string("cannot read ") + name);
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

// `x[t][c]` of one prompt into every live instance's channel ciphertexts,
// carrying `carry`, at `level`.
void EncryptPrompt(Ring &ring, const CiBatchLayout &layout,
                   const std::vector<double> &x, int H, int live, double carry,
                   int level, std::vector<Ciphertext<word>> &cts) {
  const int T = layout.num_tokens;
  const double scale = ring.param->GetScale(level);
  cts.clear();
  cts.resize(H);
  std::vector<double> values(static_cast<size_t>(layout.num_instances) * T, 0.0);
  std::vector<Complex> msg;
  for (int c = 0; c < H; c++) {
    for (int b = 0; b < live; b++) {
      for (int t = 0; t < T; t++) {
        values[static_cast<size_t>(b) * T + t] = carry * x[static_cast<size_t>(t) * H + c];
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
Err Compare(const std::vector<double> &got, const std::vector<double> &want,
            int live, int T, int H) {
  Err e;
  double se = 0.0, sr = 0.0;
  for (int b = 0; b < live; b++) {
    double seb = 0.0, srb = 0.0;
    for (int t = 0; t < T; t++) {
      for (int c = 0; c < H; c++) {
        const size_t i = (static_cast<size_t>(b) * T + t) * H + c;
        const double w = want[static_cast<size_t>(t) * H + c];
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
    std::ifstream g(rd + "/calib.json");
    ASSERT_TRUE(g.good()) << rd + "/calib.json";
    calib = json::parse(g);
  }
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
  const int live = std::string(Env("BERT_TINY_INSTANCES", "1")) == "all"
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

  // The input, and the chain.
  std::vector<double> x0 = ReadVec(ad + "/input.f32", static_cast<size_t>(T) * H);
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
    EncryptPrompt(boot, layout, x0, H, live, in.carry, level, in.cts);
    std::cout << "  input at level " << level << ", carry " << in.carry << std::endl;
  }
  double worst = 0.0;
  for (int L = 0; L < num_layers; L++) {
    if (L > 0) layer.Prepare(files[L].w, ReadCalib(calib["layers"][L]));
    Layer::Stream out;
    layer.Layer(out, in, evk);
    std::vector<double> got, want;
    DecryptAll(boot, layout, out.cts, H, live, out.carry, got);
    char name[16];
    std::snprintf(name, sizeof name, "/h_L%02d.f64", L);
    ASSERT_TRUE(ReadF64(rd + name, static_cast<size_t>(T) * H, want)) << rd + name;
    const Err e = Compare(got, want, live, T, H);
    const auto &s = layer.GetStages();
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
  std::cout << "CHAIN worst rms 2^" << std::setprecision(2) << -Bits(worst)
            << " over " << num_layers << " layer(s), " << live << " instance(s)"
            << std::endl;
  EXPECT_LT(worst, std::ldexp(1.0, -3));
#endif
}
