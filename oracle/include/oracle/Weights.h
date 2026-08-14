// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic synthetic weights and activations, and the optional hooks for
// weights the user supplies locally.
//
// NO MODEL WEIGHTS ARE EMBEDDED IN OR DOWNLOADED BY THIS PROJECT. The loader
// reads a directory the caller names; if that directory does not exist, every
// tool here still runs, on synthetic data.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "oracle/Config.h"
#include "oracle/Ops.h"

namespace oracle {

/// How the synthetic weights are scaled.
///
/// This matters more than it looks. HEonGPU's record (§23.7 of
/// reference/LLAMA3_8B_LAYER_FLOW.md) reports that unscaled random weights at
/// Llama-3's width put raw attention scores across a span of hundreds, so the
/// SoftMax was asked to fit exp over [-300, 0] -- a dynamic range of e^75 --
/// and the failure surfaced as an int64 overflow at decrypt, which looks like
/// a library fault. A trained model does not present that.
///
/// kUnitVariance therefore scales each projection by 1/sqrt(fan_in) so that a
/// unit-variance input gives a unit-variance output, which puts the synthetic
/// score distribution in the same decade as a trained model's. It is a
/// deliberate modelling choice, NOT a measurement of Llama-3, and every
/// calibration number derived from it is labelled synthetic.
enum class WeightScaling {
  /// std = 1/sqrt(fan_in). Bounded, trained-model-like magnitudes.
  kUnitVariance = 0,
  /// std = 1. Reproduces the unbounded regime the record warns about; used by
  /// the `stress` test vector and by nothing else.
  kUnscaled = 1,
};

struct SyntheticSpec {
  uint64_t seed = 20260814;
  WeightScaling scaling = WeightScaling::kUnitVariance;
  /// RMSNorm weights are drawn as 1 + norm_weight_jitter * N(0,1).
  double norm_weight_jitter = 0.10;
  /// Fraction of residual channels given `outlier_gain` times the magnitude.
  /// Real Llama residual streams have a handful of very large channels; this
  /// reproduces the shape of that without claiming its size.
  double outlier_fraction = 0.0;
  double outlier_gain = 1.0;
  /// Activation magnitude, in units of standard deviation.
  double activation_scale = 1.0;
};

/// Deterministic weights for layer `layer_index`. Depends only on
/// (spec.seed, layer_index, config shape) -- changing the token count never
/// changes a weight.
LayerWeights MakeSyntheticLayer(const Llama3Config& config,
                                const SyntheticSpec& spec, int64_t layer_index);

std::vector<LayerWeights> MakeSyntheticLayers(const Llama3Config& config,
                                              const SyntheticSpec& spec,
                                              int64_t num_layers);

/// Deterministic [tokens, hidden] activations. Independent stream from the
/// weights, so token count and layer count vary independently.
Tensor MakeSyntheticActivations(const Llama3Config& config,
                                const SyntheticSpec& spec, int64_t tokens);

// ---------------------------------------------------------------------------
// Local weight hooks
// ---------------------------------------------------------------------------

/// The on-disk format the loader expects. It is deliberately the format
/// `HEonGPU/benchmark/fetch_llama3_weights.py` already writes, so a user who
/// has run that script once has a directory this oracle reads with no
/// conversion:
///
///   <dir>/wq.f32     float32 little-endian, [hidden, heads*head_dim]
///   <dir>/wk.f32                            [hidden, kv_heads*head_dim]
///   <dir>/wv.f32                            [hidden, kv_heads*head_dim]
///   <dir>/wo.f32                            [heads*head_dim, hidden]
///   <dir>/wgate.f32                         [hidden, intermediate]
///   <dir>/wup.f32                           [hidden, intermediate]
///   <dir>/wdown.f32                         [intermediate, hidden]
///   <dir>/attn_norm.f32                     [hidden]
///   <dir>/ffn_norm.f32                      [hidden]
///   <dir>/input.f32       (optional)        [tokens, hidden]
///
/// All matrices are ALREADY TRANSPOSED to [in_features, out_features]; the
/// fetch script does that transpose on the way out. Files are raw, headerless,
/// row-major.
///
/// The oracle never fetches anything. If the directory is absent every tool
/// falls back to synthetic data and says so in its report.
struct WeightPaths {
  std::string dir;
  std::string q = "wq.f32";
  std::string k = "wk.f32";
  std::string v = "wv.f32";
  std::string o = "wo.f32";
  std::string gate = "wgate.f32";
  std::string up = "wup.f32";
  std::string down = "wdown.f32";
  std::string attn_norm = "attn_norm.f32";
  std::string ffn_norm = "ffn_norm.f32";
  std::string input = "input.f32";
};

/// Reads a headerless little-endian float32 file into a tensor of the given
/// shape, promoting to float64. Returns false and fills `error` if the file is
/// missing or its size does not match the shape exactly (a size mismatch is
/// the usual symptom of a config that does not match the checkpoint).
bool LoadF32(const std::string& path, const std::vector<int64_t>& shape,
             Tensor* out, std::string* error);

/// Same, into a flat vector (for the two norm weight vectors).
bool LoadF32Vector(const std::string& path, int64_t length,
                   std::vector<double>* out, std::string* error);

/// Loads a full layer. Returns false with a specific message on the first
/// file that is missing or mis-sized.
bool LoadLayerWeights(const WeightPaths& paths, const Llama3Config& config,
                      LayerWeights* out, std::string* error);

/// Loads `<dir>/input.f32` as [tokens, hidden]. Optional.
bool LoadInputActivations(const WeightPaths& paths, const Llama3Config& config,
                          int64_t tokens, Tensor* out, std::string* error);

/// Written into every report so a reader always knows which it is looking at.
std::string WeightSourceNote(bool loaded_real, const std::string& dir);

/// Step-by-step instructions for connecting local weights, printed by
/// `llama3_oracle_cli weights-howto`.
std::string LocalWeightsHowto();

}  // namespace oracle
