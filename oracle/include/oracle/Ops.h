// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// The reference operations. Every one is standalone, pure, and written in
// explicit index order -- no BLAS, no threading, no reassociation. That is
// what makes the oracle an oracle: two runs of it agree exactly, so any
// disagreement with an encrypted module belongs to the encrypted module.
//
// Layout convention for every activation tensor here:
//     rank 2, [tokens, channels], row-major, channel fastest.
// Layout convention for every weight matrix here:
//     rank 2, [in_features, out_features], row-major.
//     NOTE this is the TRANSPOSE of HuggingFace's [out, in]. The transpose is
//     done once, at load time, by LoadLayerWeightsF32 / the fetch script.
// Layout convention for score-shaped tensors:
//     rank 3, [query_head, query_token, key_token].
//
// None of this is a ciphertext packing. S0 owns the mapping from these
// logical shapes onto slots; this file fixes only the values.

#pragma once

#include <string>
#include <vector>

#include "oracle/Config.h"
#include "oracle/Tensor.h"

namespace oracle {

// ---------------------------------------------------------------------------
// Boundary export
// ---------------------------------------------------------------------------

/// Metadata travelling with every exported intermediate tensor.
struct TensorMeta {
  std::string dims;         ///< comma-separated axis names, e.g. "token,channel"
  std::string description;  ///< what the tensor is
  /// Which encrypted module is expected to produce or consume it. Free text;
  /// the HANDOFF fixes the vocabulary.
  std::string consumer;
  int64_t layer = -1;
  /// Set for the tensors that are the *input* of a polynomial approximation.
  /// The calibrator keys its ranges off this.
  std::string fit_role;  // "" | "rsqrt" | "exp" | "reciprocal" | "silu"
};

/// Receives every named intermediate tensor the reference produces.
class TensorSink {
 public:
  virtual ~TensorSink() = default;
  /// Return false to skip materialising an expensive tensor entirely.
  virtual bool Wants(const std::string& name) const {
    (void)name;
    return true;
  }
  virtual void Emit(const std::string& name, const Tensor& value,
                    const TensorMeta& meta) = 0;
};

/// Null-safe emit.
void Emit(TensorSink* sink, const std::string& name, const Tensor& value,
          const TensorMeta& meta);

/// A sink that keeps everything in memory. Used by tests.
class RecordingSink : public TensorSink {
 public:
  void Emit(const std::string& name, const Tensor& value,
            const TensorMeta& meta) override;
  bool Has(const std::string& name) const;
  const Tensor& Get(const std::string& name) const;  ///< throws if absent
  const TensorMeta& MetaOf(const std::string& name) const;
  const std::vector<std::string>& Names() const { return order_; }

 private:
  std::vector<std::string> order_;
  std::vector<std::pair<Tensor, TensorMeta>> values_;
  int64_t IndexOf(const std::string& name) const;
};

/// A sink that forwards to several sinks. `Wants` is the OR of theirs.
class TeeSink : public TensorSink {
 public:
  void Add(TensorSink* s) { sinks_.push_back(s); }
  bool Wants(const std::string& name) const override;
  void Emit(const std::string& name, const Tensor& value,
            const TensorMeta& meta) override;

 private:
  std::vector<TensorSink*> sinks_;
};

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// out[t][c] = x[t][c] * weight[c] / sqrt(mean_c(x[t][c]^2) + eps)
///
/// `sum_sq` (optional, [tokens]) receives sum_c x[t][c]^2 -- the reduction an
/// encrypted RMSNorm actually computes.
/// `mean_sq` (optional, [tokens]) receives sum_sq/channels + eps -- the
/// argument the 1/sqrt approximation is evaluated at.
/// `inv_rms` (optional, [tokens]) receives 1/sqrt(mean_sq).
Tensor RmsNorm(const Tensor& x, const std::vector<double>& weight, double eps,
               Tensor* sum_sq = nullptr, Tensor* mean_sq = nullptr,
               Tensor* inv_rms = nullptr);

/// out[t][o] = sum_i x[t][i] * w[i][o]. No bias -- Llama-3 has none.
/// Summation is in ascending `i` with a single float64 accumulator.
Tensor Linear(const Tensor& x, const Tensor& w);

/// cos/sin tables for positions [offset, offset+tokens), shape
/// [tokens, head_dim/2]. Angle for pair index c is
/// `position * theta^(-2c/head_dim)`.
void RopeTables(int64_t tokens, int64_t head_dim, double theta, int64_t offset,
                Tensor* cos_table, Tensor* sin_table);

/// Rotary embedding applied in place to a [tokens, heads*head_dim] tensor.
/// See RopeConvention -- the pairing is a property of the weight file.
void ApplyRope(Tensor* x, int64_t heads, int64_t head_dim, double theta,
               int64_t position_offset, RopeConvention convention);

/// True when query token `q` may attend to key token `k` under a causal mask.
/// Both are indices into the same window; `q_offset`/`k_offset` place that
/// window in absolute positions (they differ only once a KV cache exists).
bool CausalAllowed(int64_t q, int64_t k, int64_t q_offset, int64_t k_offset);

/// SoftMax of one row over `n` keys, with `allowed` marking the unmasked ones.
///
/// The mask is applied to the EXPONENTIALS, not to the scores: an encrypted
/// SoftMax multiplies exp(s) by a 0/1 plaintext mask, and adding -inf to a
/// score is not something CKKS can do. The two are equivalent in exact
/// arithmetic; this is the form the encrypted circuit matches.
///
/// `exp_input` / `exp_value` (optional, size n) and `denominator` (optional)
/// receive the intermediate quantities the calibrator needs.
void SoftmaxRow(const double* scores, const unsigned char* allowed, int64_t n,
                SoftmaxMode mode, double fixed_shift, double* out,
                double* exp_input = nullptr, double* exp_value = nullptr,
                double* denominator = nullptr);

/// x * sigmoid(x), computed in the branch-stable form.
double Silu(double x);

/// out = silu(gate) * up, elementwise.
Tensor SwiGlu(const Tensor& gate, const Tensor& up, Tensor* silu_out = nullptr);

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

struct LayerWeights {
  std::vector<double> attn_norm;  ///< [hidden]
  std::vector<double> ffn_norm;   ///< [hidden]
  Tensor wq;                      ///< [hidden, num_heads*head_dim]
  Tensor wk;                      ///< [hidden, num_kv_heads*head_dim]
  Tensor wv;                      ///< [hidden, num_kv_heads*head_dim]
  Tensor wo;                      ///< [num_heads*head_dim, hidden]
  Tensor wgate;                   ///< [hidden, intermediate]
  Tensor wup;                     ///< [hidden, intermediate]
  Tensor wdown;                   ///< [intermediate, hidden]

  /// Throws std::invalid_argument naming the first shape that disagrees.
  void Validate(const Llama3Config& config) const;
};

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

/// Grouped-query attention. Query head `h` reads key/value head
/// `h / (num_heads/num_kv_heads)`. This is NOT multi-head attention with a
/// repeated K/V tensor materialised first; the group mapping is explicit, and
/// `GqaGroupsShareKv` in the test suite pins it.
///
/// q: [tokens, num_heads*head_dim]; k, v: [tokens, num_kv_heads*head_dim].
/// Returns [tokens, num_heads*head_dim].
Tensor GqaAttention(const Tensor& q, const Tensor& k, const Tensor& v,
                    const Llama3Config& config, int64_t position_offset,
                    TensorSink* sink, const std::string& prefix);

/// down(silu(gate(x)) * up(x)).
Tensor FeedForward(const Tensor& x, const LayerWeights& w,
                   const Llama3Config& config, TensorSink* sink,
                   const std::string& prefix);

/// The attention half of a decoder block, residual included:
///   x + Wo * GQA(RoPE(Wq n), RoPE(Wk n), Wv n)   where n = RMSNorm(x)
Tensor AttentionSublayer(const Tensor& x, const LayerWeights& w,
                         const Llama3Config& config, int64_t position_offset,
                         TensorSink* sink, const std::string& prefix);

/// One full Llama-3 decoder block.
Tensor DecoderBlock(const Tensor& x, const LayerWeights& w,
                    const Llama3Config& config, int64_t position_offset,
                    TensorSink* sink, const std::string& prefix);

/// `num_layers` decoder blocks in sequence, emitting under prefix "L0.", ... .
Tensor Forward(const Tensor& x, const std::vector<LayerWeights>& layers,
               const Llama3Config& config, int64_t position_offset,
               TensorSink* sink);

/// The exact list of boundary names one layer emits, in emission order, with
/// their metadata. Used by the HANDOFF and by the manifest writer to declare
/// the contract without running the model.
struct BoundaryDecl {
  std::string suffix;  ///< name after the "L<n>." prefix
  std::string dims;
  std::string shape_expr;  ///< e.g. "tokens x hidden"
  std::string fit_role;
  std::string description;
};
const std::vector<BoundaryDecl>& LayerBoundaries();

/// The numerical conventions this oracle commits to, as text. Reproduced in
/// every report so a consumer never has to guess.
std::string NumericalConventions();

}  // namespace oracle
