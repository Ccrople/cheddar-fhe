// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Weights.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

#include "oracle/Rng.h"

namespace oracle {
namespace {

double FanInScale(WeightScaling s, int64_t fan_in) {
  if (s == WeightScaling::kUnscaled) return 1.0;
  return 1.0 / std::sqrt(static_cast<double>(fan_in));
}

/// Fills a matrix from its own derived stream so that a matrix's contents
/// depend on its name and the layer index, never on the order of construction.
Tensor MakeMatrix(const SyntheticSpec& spec, int64_t layer, const char* name,
                  int64_t rows, int64_t cols) {
  Rng rng(spec.seed);
  Rng stream = rng.Derive(HashName(name) ^ (static_cast<uint64_t>(layer) * 0x9E3779B1ULL));
  Tensor t({rows, cols});
  const double s = FanInScale(spec.scaling, rows);
  for (int64_t i = 0; i < t.Size(); ++i) t[i] = stream.Normal() * s;
  return t;
}

std::vector<double> MakeNormWeights(const SyntheticSpec& spec, int64_t layer,
                                    const char* name, int64_t n) {
  Rng rng(spec.seed);
  Rng stream = rng.Derive(HashName(name) ^ (static_cast<uint64_t>(layer) * 0x85EBCA6BULL));
  std::vector<double> w(static_cast<size_t>(n));
  for (auto& x : w) x = 1.0 + spec.norm_weight_jitter * stream.Normal();
  return w;
}

bool ReadWholeFile(const std::string& path, std::vector<char>* bytes,
                   std::string* error) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  const std::streamsize n = f.tellg();
  if (n < 0) {
    if (error) *error = "cannot size " + path;
    return false;
  }
  f.seekg(0, std::ios::beg);
  bytes->resize(static_cast<size_t>(n));
  if (n > 0 && !f.read(bytes->data(), n)) {
    if (error) *error = "short read on " + path;
    return false;
  }
  return true;
}

float BytesToFloat(const char* p) {
  // Little-endian by construction, so a memcpy is correct on every platform
  // this project targets (x86-64 Windows, x86-64 Linux).
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}

}  // namespace

LayerWeights MakeSyntheticLayer(const Llama3Config& c, const SyntheticSpec& spec,
                                int64_t layer) {
  LayerWeights w;
  w.attn_norm = MakeNormWeights(spec, layer, "attn_norm", c.hidden_size);
  w.ffn_norm = MakeNormWeights(spec, layer, "ffn_norm", c.hidden_size);
  w.wq = MakeMatrix(spec, layer, "wq", c.hidden_size, c.QChannels());
  w.wk = MakeMatrix(spec, layer, "wk", c.hidden_size, c.KvChannels());
  w.wv = MakeMatrix(spec, layer, "wv", c.hidden_size, c.KvChannels());
  w.wo = MakeMatrix(spec, layer, "wo", c.QChannels(), c.hidden_size);
  w.wgate = MakeMatrix(spec, layer, "wgate", c.hidden_size, c.intermediate_size);
  w.wup = MakeMatrix(spec, layer, "wup", c.hidden_size, c.intermediate_size);
  w.wdown = MakeMatrix(spec, layer, "wdown", c.intermediate_size, c.hidden_size);
  w.Validate(c);
  return w;
}

std::vector<LayerWeights> MakeSyntheticLayers(const Llama3Config& c,
                                              const SyntheticSpec& spec,
                                              int64_t num_layers) {
  std::vector<LayerWeights> out;
  out.reserve(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i)
    out.push_back(MakeSyntheticLayer(c, spec, i));
  return out;
}

Tensor MakeSyntheticActivations(const Llama3Config& c, const SyntheticSpec& spec,
                                int64_t tokens) {
  Rng rng(spec.seed);
  Rng stream = rng.Derive(HashName("activations"));
  Tensor x({tokens, c.hidden_size});
  for (int64_t i = 0; i < x.Size(); ++i) x[i] = stream.Normal() * spec.activation_scale;

  if (spec.outlier_fraction > 0.0 && spec.outlier_gain != 1.0) {
    // Outlier CHANNELS, not outlier elements: the phenomenon this imitates is
    // a channel of the residual stream that is large for every token.
    Rng osel = rng.Derive(HashName("outlier_channels"));
    std::vector<unsigned char> is_outlier(static_cast<size_t>(c.hidden_size), 0);
    for (int64_t ch = 0; ch < c.hidden_size; ++ch)
      is_outlier[static_cast<size_t>(ch)] =
          osel.Uniform() < spec.outlier_fraction ? 1 : 0;
    for (int64_t t = 0; t < tokens; ++t)
      for (int64_t ch = 0; ch < c.hidden_size; ++ch)
        if (is_outlier[static_cast<size_t>(ch)]) x.At(t, ch) *= spec.outlier_gain;
  }
  return x;
}

// ---------------------------------------------------------------------------

bool LoadF32(const std::string& path, const std::vector<int64_t>& shape,
             Tensor* out, std::string* error) {
  std::vector<char> bytes;
  if (!ReadWholeFile(path, &bytes, error)) return false;
  int64_t want = 1;
  for (int64_t d : shape) want *= d;
  const int64_t got = static_cast<int64_t>(bytes.size() / sizeof(float));
  if (bytes.size() % sizeof(float) != 0 || got != want) {
    if (error)
      *error = path + ": expected " + std::to_string(want) +
               " float32 values (" + std::to_string(want * 4) + " bytes), file "
               "holds " + std::to_string(got) + " (" +
               std::to_string(bytes.size()) +
               " bytes). The configuration does not match this checkpoint.";
    return false;
  }
  *out = Tensor(shape);
  for (int64_t i = 0; i < want; ++i)
    (*out)[i] = static_cast<double>(BytesToFloat(bytes.data() + i * 4));
  return true;
}

bool LoadF32Vector(const std::string& path, int64_t length,
                   std::vector<double>* out, std::string* error) {
  Tensor t;
  if (!LoadF32(path, {length}, &t, error)) return false;
  out->resize(static_cast<size_t>(length));
  for (int64_t i = 0; i < length; ++i) (*out)[static_cast<size_t>(i)] = t[i];
  return true;
}

bool LoadLayerWeights(const WeightPaths& p, const Llama3Config& c,
                      LayerWeights* out, std::string* error) {
  const std::string d = p.dir.empty() ? std::string(".") : p.dir;
  auto join = [&d](const std::string& f) { return d + "/" + f; };
  LayerWeights w;
  if (!LoadF32Vector(join(p.attn_norm), c.hidden_size, &w.attn_norm, error)) return false;
  if (!LoadF32Vector(join(p.ffn_norm), c.hidden_size, &w.ffn_norm, error)) return false;
  if (!LoadF32(join(p.q), {c.hidden_size, c.QChannels()}, &w.wq, error)) return false;
  if (!LoadF32(join(p.k), {c.hidden_size, c.KvChannels()}, &w.wk, error)) return false;
  if (!LoadF32(join(p.v), {c.hidden_size, c.KvChannels()}, &w.wv, error)) return false;
  if (!LoadF32(join(p.o), {c.QChannels(), c.hidden_size}, &w.wo, error)) return false;
  if (!LoadF32(join(p.gate), {c.hidden_size, c.intermediate_size}, &w.wgate, error)) return false;
  if (!LoadF32(join(p.up), {c.hidden_size, c.intermediate_size}, &w.wup, error)) return false;
  if (!LoadF32(join(p.down), {c.intermediate_size, c.hidden_size}, &w.wdown, error)) return false;
  try {
    w.Validate(c);
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
  *out = std::move(w);
  return true;
}

bool LoadInputActivations(const WeightPaths& p, const Llama3Config& c,
                          int64_t tokens, Tensor* out, std::string* error) {
  const std::string d = p.dir.empty() ? std::string(".") : p.dir;
  return LoadF32(d + "/" + p.input, {tokens, c.hidden_size}, out, error);
}

std::string WeightSourceNote(bool loaded_real, const std::string& dir) {
  if (loaded_real)
    return "weights: LOADED from " + dir +
           " (locally supplied float32 files; not part of this repository)";
  return "weights: SYNTHETIC (deterministic, seeded; no Meta weights present). "
         "Every range below is a property of the synthetic distribution, not "
         "of Llama-3.";
}

std::string LocalWeightsHowto() {
  return
R"(Connecting locally available Llama-3-8B weights
==============================================
Nothing in this repository downloads, contains, or redistributes model
weights, and this oracle never reaches the network. What follows is what YOU
do on a machine that already has the weights, and what the oracle then reads.

STEP 1 -- produce a bundle for one block.
  `Projects/HEonGPU/benchmark/fetch_llama3_weights.py` already writes exactly
  the format this oracle reads. It is a separate, read-only repository; run it
  from there, and point --dest anywhere you like. It needs numpy, tokenizers,
  and about 17 GiB of cache:

      python3 fetch_llama3_weights.py --cache <cache> --scan
      python3 fetch_llama3_weights.py --cache <cache> --layer 2 \
          --tokens 128 --dest <bundle>

  --scan prints every layer's input bound first, which is how the layer is
  chosen rather than assumed.

  Gated weights: that script points at an ungated mirror. If your copy of the
  weights came from Meta directly under their licence, export the same nine
  files yourself; the oracle does not care where they came from and does not
  check.

STEP 2 -- what the bundle must contain.
  Raw, headerless, little-endian float32, row-major, ALREADY TRANSPOSED to
  [in_features, out_features]:

      wq.f32          hidden x heads*head_dim              4096 x 4096
      wk.f32          hidden x kv_heads*head_dim           4096 x 1024
      wv.f32          hidden x kv_heads*head_dim           4096 x 1024
      wo.f32          heads*head_dim x hidden              4096 x 4096
      wgate.f32       hidden x intermediate                4096 x 14336
      wup.f32         hidden x intermediate                4096 x 14336
      wdown.f32       intermediate x hidden               14336 x 4096
      attn_norm.f32   hidden                                    4096
      ffn_norm.f32    hidden                                    4096
      input.f32       tokens x hidden           (optional)  128 x 4096

  A file whose size does not match is reported by name with both counts. That
  is the intended failure mode: a silent reshape would be far worse.

STEP 3 -- run the oracle against it. Nothing else changes.

      llama3_oracle_cli calibrate --weights-dir <bundle> --tokens 128 \
          --out <report-dir>
      llama3_oracle_cli dump      --weights-dir <bundle> --tokens 128 \
          --out <tensor-dir>

  Every report records whether it used real or synthetic weights, in its first
  line and in its JSON, so a calibration number can never be mistaken for the
  other kind.

STEP 4 -- the thing to check before trusting the result.
  RoPE channel pairing. HuggingFace weights need `--rope half_split` (the
  default); an original Meta `consolidated.00.pth` export needs
  `--rope interleaved`. Both produce correctly-shaped, correctly-normed
  tensors, so no assertion here can tell them apart. If you have a trusted
  forward pass, compare against it once and record which convention matched.

WHAT IS STILL SYNTHETIC EVEN WITH REAL WEIGHTS
  input.f32 is one 128-token window of one prompt. It fixes the *scale* of the
  activations, which is what the calibration needs, but a calibration interval
  measured on a single window is a lower bound on the interval a deployment
  sees. Widen it deliberately before committing a polynomial degree, and say
  by how much.
)";
}

}  // namespace oracle
