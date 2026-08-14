// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// The Llama-3-8B model configuration, and where every number in it came from.
//
// Session S2 owns this file. Nothing here selects a cryptographic parameter,
// a packing, or a ciphertext layout: this is the *value* contract only.

#pragma once

#include <cstdint>
#include <string>

namespace oracle {

/// Which channel pairing RoPE rotates.
///
/// The two conventions are NOT interchangeable and produce different tensors
/// from the same weights. Which one is correct is decided by the weight file,
/// not by preference -- see `RopeConventionNote()`.
enum class RopeConvention {
  /// HuggingFace `rotate_half`: channel `c` pairs with `c + head_dim/2`.
  /// This is correct for `*.safetensors` weights from a HF-format repo, which
  /// is what `HEonGPU/benchmark/fetch_llama3_weights.py` downloads.
  kHalfSplit = 0,
  /// Meta reference implementation: channel `2c` pairs with `2c + 1`.
  /// Correct for the original `consolidated.00.pth` checkpoint layout.
  kInterleaved = 1,
};

/// How the SoftMax numerator is shifted before exponentiation.
enum class SoftmaxMode {
  /// Subtract the per-row maximum over unmasked keys. Numerically exact and
  /// the mathematical reference. NOT implementable under FHE -- a max is not
  /// a low-degree polynomial.
  kRowMax = 0,
  /// Subtract a single constant for every row. This is what an encrypted
  /// SoftMax actually does, so the calibration ranges that matter to
  /// Sessions 4-7 are the ones measured in this mode.
  kFixedShift = 1,
  /// No shift at all. Present only so a test can demonstrate that it
  /// overflows where the other two do not.
  kNaive = 2,
};

/// Llama-3-8B, plus the knobs the oracle needs that the model does not have.
struct Llama3Config {
  // ---- verified model shape (see ConfigProvenance()) ----
  int64_t vocab_size = 128256;
  int64_t hidden_size = 4096;        ///< d_model, the residual stream width
  int64_t intermediate_size = 14336; ///< SwiGLU inner width
  int64_t num_layers = 32;
  int64_t num_heads = 32;            ///< query heads
  int64_t num_kv_heads = 8;          ///< key/value heads; GQA group = 32/8 = 4
  int64_t head_dim = 128;
  double rms_norm_eps = 1e-5;
  double rope_theta = 500000.0;
  int64_t max_position_embeddings = 8192;

  // ---- oracle-side conventions, not model parameters ----
  RopeConvention rope_convention = RopeConvention::kHalfSplit;
  SoftmaxMode softmax_mode = SoftmaxMode::kRowMax;
  /// Constant subtracted from every score when `softmax_mode == kFixedShift`.
  double softmax_fixed_shift = 0.0;
  /// Fold `1/head_dim` into the RMSNorm fit instead of the reduction. Changes
  /// which range the `1/sqrt` approximation is calibrated over, nothing else.
  bool rmsnorm_fold_mean_into_fit = false;
  bool causal = true;
  /// Apply rotary embedding to Q and K. Always true for real Llama-3; the
  /// switch exists because an attention sublayer without RoPE is a strictly
  /// simpler circuit that some encrypted milestones may want to hit first,
  /// and because it makes the causal/mask invariants exactly checkable.
  /// HEonGPU's rect path shipped with rope off and documented it as a
  /// simplification of the CIRCUIT, not of the model -- keep that distinction.
  bool rope_enabled = true;

  // ---- derived ----
  int64_t QChannels() const { return num_heads * head_dim; }
  int64_t KvChannels() const { return num_kv_heads * head_dim; }
  int64_t GqaGroup() const { return num_heads / num_kv_heads; }
  double AttentionScale() const;  ///< 1 / sqrt(head_dim)

  /// Throws std::invalid_argument naming the first violated invariant.
  void Validate() const;

  /// Stable one-line identity used in every serialized manifest.
  std::string Identity() const;
};

/// The full 8B configuration. Every field is repository-sourced.
Llama3Config Llama3_8B();

/// A shape-reduced configuration with Llama-3's *structure* preserved:
/// GQA ratio, head_dim divisibility, eps and theta are unchanged. Used by the
/// unit tests, which are testing the circuit and not the width.
///
/// `heads` must be divisible by `kv_heads`; `hidden` is set to
/// `heads * head_dim`.
Llama3Config Llama3Reduced(int64_t heads, int64_t kv_heads, int64_t head_dim,
                           int64_t intermediate);

/// Human-readable provenance for every verified field: the file and line this
/// oracle read it from. Printed by `llama3_oracle_cli config`.
std::string ConfigProvenance();

/// Why the RoPE convention is a property of the weight file.
std::string RopeConventionNote();

const char* ToString(RopeConvention c);
const char* ToString(SoftmaxMode m);

}  // namespace oracle
