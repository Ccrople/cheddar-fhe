// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace oracle {
namespace {

int64_t LayerFromPrefix(const std::string& prefix) {
  if (prefix.size() < 2 || prefix[0] != 'L') return -1;
  size_t i = 1;
  int64_t v = 0;
  bool any = false;
  while (i < prefix.size() && prefix[i] >= '0' && prefix[i] <= '9') {
    v = v * 10 + (prefix[i] - '0');
    ++i;
    any = true;
  }
  if (!any || i >= prefix.size() || prefix[i] != '.') return -1;
  return v;
}

}  // namespace

void Emit(TensorSink* sink, const std::string& name, const Tensor& value,
          const TensorMeta& meta) {
  if (sink == nullptr) return;
  if (!sink->Wants(name)) return;
  sink->Emit(name, value, meta);
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

void RecordingSink::Emit(const std::string& name, const Tensor& value,
                         const TensorMeta& meta) {
  const int64_t idx = IndexOf(name);
  if (idx >= 0) {
    values_[static_cast<size_t>(idx)] = {value, meta};
    return;
  }
  order_.push_back(name);
  values_.emplace_back(value, meta);
}

int64_t RecordingSink::IndexOf(const std::string& name) const {
  for (size_t i = 0; i < order_.size(); ++i)
    if (order_[i] == name) return static_cast<int64_t>(i);
  return -1;
}

bool RecordingSink::Has(const std::string& name) const {
  return IndexOf(name) >= 0;
}

const Tensor& RecordingSink::Get(const std::string& name) const {
  const int64_t i = IndexOf(name);
  if (i < 0) throw std::out_of_range("RecordingSink: no tensor named " + name);
  return values_[static_cast<size_t>(i)].first;
}

const TensorMeta& RecordingSink::MetaOf(const std::string& name) const {
  const int64_t i = IndexOf(name);
  if (i < 0) throw std::out_of_range("RecordingSink: no tensor named " + name);
  return values_[static_cast<size_t>(i)].second;
}

bool TeeSink::Wants(const std::string& name) const {
  for (TensorSink* s : sinks_)
    if (s->Wants(name)) return true;
  return false;
}

void TeeSink::Emit(const std::string& name, const Tensor& value,
                   const TensorMeta& meta) {
  for (TensorSink* s : sinks_)
    if (s->Wants(name)) s->Emit(name, value, meta);
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

Tensor RmsNorm(const Tensor& x, const std::vector<double>& weight, double eps,
               Tensor* sum_sq, Tensor* mean_sq, Tensor* inv_rms) {
  if (x.Rank() != 2)
    throw std::invalid_argument("RmsNorm: expected [tokens, channels]");
  const int64_t tokens = x.Dim(0), channels = x.Dim(1);
  if (static_cast<int64_t>(weight.size()) != channels)
    throw std::invalid_argument("RmsNorm: weight length != channels");

  Tensor out({tokens, channels});
  if (sum_sq) *sum_sq = Tensor({tokens});
  if (mean_sq) *mean_sq = Tensor({tokens});
  if (inv_rms) *inv_rms = Tensor({tokens});

  for (int64_t t = 0; t < tokens; ++t) {
    const double* row = x.Row(t);
    double acc = 0.0;
    for (int64_t c = 0; c < channels; ++c) acc += row[c] * row[c];
    const double ms = acc / static_cast<double>(channels) + eps;
    const double inv = 1.0 / std::sqrt(ms);
    if (sum_sq) (*sum_sq)[t] = acc;
    if (mean_sq) (*mean_sq)[t] = ms;
    if (inv_rms) (*inv_rms)[t] = inv;
    double* o = out.Row(t);
    for (int64_t c = 0; c < channels; ++c) o[c] = row[c] * inv * weight[static_cast<size_t>(c)];
  }
  return out;
}

Tensor Linear(const Tensor& x, const Tensor& w) {
  if (x.Rank() != 2 || w.Rank() != 2)
    throw std::invalid_argument("Linear: both operands must be rank 2");
  if (x.Dim(1) != w.Dim(0))
    throw std::invalid_argument("Linear: inner dimension mismatch " +
                                x.ShapeString() + " * " + w.ShapeString());
  const int64_t tokens = x.Dim(0), in_f = x.Dim(1), out_f = w.Dim(1);
  Tensor out({tokens, out_f});  // zero-initialised
  // Loop order t-i-j, not t-j-i. The summation order over the input feature
  // index `i` is IDENTICAL either way -- each output accumulates the terms
  // i = 0, 1, 2, ... in that order, so the result is bit-for-bit the same --
  // but this order walks `w` and `out` sequentially instead of striding
  // through `w` by out_features. At the real 8B width that is the difference
  // between one pass over a 470 MB matrix and 14336 of them.
  for (int64_t t = 0; t < tokens; ++t) {
    const double* xr = x.Row(t);
    double* o = out.Row(t);
    for (int64_t i = 0; i < in_f; ++i) {
      const double xi = xr[i];
      const double* wr = w.Row(i);
      for (int64_t j = 0; j < out_f; ++j) o[j] += xi * wr[j];
    }
  }
  return out;
}

void RopeTables(int64_t tokens, int64_t head_dim, double theta, int64_t offset,
                Tensor* cos_table, Tensor* sin_table) {
  const int64_t half = head_dim / 2;
  *cos_table = Tensor({tokens, half});
  *sin_table = Tensor({tokens, half});
  for (int64_t t = 0; t < tokens; ++t) {
    const double pos = static_cast<double>(offset + t);
    for (int64_t c = 0; c < half; ++c) {
      const double inv_freq =
          std::pow(theta, -2.0 * static_cast<double>(c) /
                              static_cast<double>(head_dim));
      const double angle = pos * inv_freq;
      cos_table->At(t, c) = std::cos(angle);
      sin_table->At(t, c) = std::sin(angle);
    }
  }
}

void ApplyRope(Tensor* x, int64_t heads, int64_t head_dim, double theta,
               int64_t position_offset, RopeConvention convention) {
  if (x->Rank() != 2)
    throw std::invalid_argument("ApplyRope: expected [tokens, heads*head_dim]");
  const int64_t tokens = x->Dim(0);
  if (x->Dim(1) != heads * head_dim)
    throw std::invalid_argument("ApplyRope: channels != heads*head_dim");
  const int64_t half = head_dim / 2;

  Tensor cos_t, sin_t;
  RopeTables(tokens, head_dim, theta, position_offset, &cos_t, &sin_t);

  for (int64_t t = 0; t < tokens; ++t) {
    double* row = x->Row(t);
    for (int64_t h = 0; h < heads; ++h) {
      double* v = row + h * head_dim;
      for (int64_t c = 0; c < half; ++c) {
        const double co = cos_t.At(t, c), si = sin_t.At(t, c);
        int64_t a, b;
        if (convention == RopeConvention::kHalfSplit) {
          a = c;
          b = c + half;
        } else {
          a = 2 * c;
          b = 2 * c + 1;
        }
        const double va = v[a], vb = v[b];
        v[a] = va * co - vb * si;
        v[b] = vb * co + va * si;
      }
    }
  }
}

bool CausalAllowed(int64_t q, int64_t k, int64_t q_offset, int64_t k_offset) {
  return (k_offset + k) <= (q_offset + q);
}

void SoftmaxRow(const double* scores, const unsigned char* allowed, int64_t n,
                SoftmaxMode mode, double fixed_shift, double* out,
                double* exp_input, double* exp_value, double* denominator) {
  double shift = 0.0;
  if (mode == SoftmaxMode::kRowMax) {
    shift = -std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < n; ++j)
      if (allowed[j] && scores[j] > shift) shift = scores[j];
    if (!std::isfinite(shift)) shift = 0.0;  // fully masked row
  } else if (mode == SoftmaxMode::kFixedShift) {
    shift = fixed_shift;
  }

  double denom = 0.0;
  for (int64_t j = 0; j < n; ++j) {
    const double u = scores[j] - shift;
    // The mask multiplies the EXPONENTIAL, matching the encrypted circuit.
    const double e = allowed[j] ? std::exp(u) : 0.0;
    if (exp_input) exp_input[j] = u;
    if (exp_value) exp_value[j] = e;
    out[j] = e;
    denom += e;
  }
  if (denominator) *denominator = denom;
  const double inv = denom > 0.0 ? 1.0 / denom : 0.0;
  for (int64_t j = 0; j < n; ++j) out[j] *= inv;
}

double Silu(double x) {
  // x / (1 + exp(-x)); the branch keeps exp() away from overflow for x << 0,
  // which the naive form reaches at about x = -746.
  if (x >= 0.0) return x / (1.0 + std::exp(-x));
  const double e = std::exp(x);
  return x * e / (1.0 + e);
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up, Tensor* silu_out) {
  if (!gate.SameShape(up))
    throw std::invalid_argument("SwiGlu: gate/up shape mismatch");
  Tensor out(gate.Shape());
  if (silu_out) *silu_out = Tensor(gate.Shape());
  for (int64_t i = 0; i < gate.Size(); ++i) {
    const double s = Silu(gate[i]);
    if (silu_out) (*silu_out)[i] = s;
    out[i] = s * up[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

void LayerWeights::Validate(const Llama3Config& c) const {
  auto want = [](const Tensor& t, int64_t r, int64_t col, const char* name) {
    if (t.Rank() != 2 || t.Dim(0) != r || t.Dim(1) != col)
      throw std::invalid_argument(std::string("LayerWeights: ") + name +
                                  " has shape " + t.ShapeString() +
                                  ", expected " + std::to_string(r) + "x" +
                                  std::to_string(col));
  };
  if (static_cast<int64_t>(attn_norm.size()) != c.hidden_size)
    throw std::invalid_argument("LayerWeights: attn_norm length != hidden_size");
  if (static_cast<int64_t>(ffn_norm.size()) != c.hidden_size)
    throw std::invalid_argument("LayerWeights: ffn_norm length != hidden_size");
  want(wq, c.hidden_size, c.QChannels(), "wq");
  want(wk, c.hidden_size, c.KvChannels(), "wk");
  want(wv, c.hidden_size, c.KvChannels(), "wv");
  want(wo, c.QChannels(), c.hidden_size, "wo");
  want(wgate, c.hidden_size, c.intermediate_size, "wgate");
  want(wup, c.hidden_size, c.intermediate_size, "wup");
  want(wdown, c.intermediate_size, c.hidden_size, "wdown");
}

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

Tensor GqaAttention(const Tensor& q, const Tensor& k, const Tensor& v,
                    const Llama3Config& config, int64_t position_offset,
                    TensorSink* sink, const std::string& prefix) {
  const int64_t tokens = q.Dim(0);
  const int64_t heads = config.num_heads, kv_heads = config.num_kv_heads;
  const int64_t hd = config.head_dim, group = config.GqaGroup();
  if (q.Dim(1) != heads * hd) throw std::invalid_argument("GqaAttention: q width");
  if (k.Dim(1) != kv_heads * hd) throw std::invalid_argument("GqaAttention: k width");
  if (!k.SameShape(v)) throw std::invalid_argument("GqaAttention: k/v shape");
  if (k.Dim(0) != tokens)
    throw std::invalid_argument("GqaAttention: k tokens != q tokens "
                                "(no KV cache in this oracle yet)");
  const int64_t layer = LayerFromPrefix(prefix);
  auto meta = [&](const char* dims, const char* desc, const char* consumer,
                  const char* fit) {
    TensorMeta m;
    m.dims = dims;
    m.description = desc;
    m.consumer = consumer;
    m.fit_role = fit;
    m.layer = layer;
    return m;
  };
  const bool want_exp = sink && (sink->Wants(prefix + "attn.exp_input") ||
                                 sink->Wants(prefix + "attn.exp"));

  Tensor scores({heads, tokens, tokens});
  Tensor probs({heads, tokens, tokens});
  Tensor denom({heads, tokens});
  Tensor recip({heads, tokens});
  Tensor exp_input, exp_value;
  if (want_exp) {
    exp_input = Tensor({heads, tokens, tokens});
    exp_value = Tensor({heads, tokens, tokens});
  }
  Tensor out({tokens, heads * hd});

  const double scale = config.AttentionScale();
  std::vector<unsigned char> allowed(static_cast<size_t>(tokens));

  for (int64_t h = 0; h < heads; ++h) {
    const int64_t kv = h / group;
    for (int64_t i = 0; i < tokens; ++i) {
      const double* qv = q.Row(i) + h * hd;
      double* srow = &scores.At(h, i, 0);
      for (int64_t j = 0; j < tokens; ++j) {
        const double* kvv = k.Row(j) + kv * hd;
        double acc = 0.0;
        for (int64_t c = 0; c < hd; ++c) acc += qv[c] * kvv[c];
        srow[j] = acc * scale;
        allowed[static_cast<size_t>(j)] =
            config.causal
                ? (CausalAllowed(i, j, position_offset, position_offset) ? 1 : 0)
                : 1;
      }
      double d = 0.0;
      SoftmaxRow(srow, allowed.data(), tokens, config.softmax_mode,
                 config.softmax_fixed_shift, &probs.At(h, i, 0),
                 want_exp ? &exp_input.At(h, i, 0) : nullptr,
                 want_exp ? &exp_value.At(h, i, 0) : nullptr, &d);
      denom.At(h, i) = d;
      recip.At(h, i) = d > 0.0 ? 1.0 / d : 0.0;

      double* orow = out.Row(i) + h * hd;
      const double* prow = &probs.At(h, i, 0);
      for (int64_t c = 0; c < hd; ++c) {
        double acc = 0.0;
        for (int64_t j = 0; j < tokens; ++j) acc += prow[j] * v.At(j, kv * hd + c);
        orow[c] = acc;
      }
    }
  }

  Emit(sink, prefix + "attn.scores", scores,
       meta("head,query,key", "Q K^T scaled by 1/sqrt(head_dim), before mask",
            "softmax module", ""));
  if (want_exp) {
    Emit(sink, prefix + "attn.exp_input", exp_input,
         meta("head,query,key", "score minus the SoftMax shift", "exp fit",
              "exp"));
    Emit(sink, prefix + "attn.exp", exp_value,
         meta("head,query,key", "exp(shifted score), causal mask applied "
                                "multiplicatively", "denominator reduction", ""));
  }
  Emit(sink, prefix + "attn.denominator", denom,
       meta("head,query", "row sum of the masked exponentials",
            "reciprocal fit", "reciprocal"));
  Emit(sink, prefix + "attn.reciprocal", recip,
       meta("head,query", "1 / denominator", "normalise product", ""));
  Emit(sink, prefix + "attn.probs", probs,
       meta("head,query,key", "attention probabilities; exactly 0 where masked",
            "PV product", ""));
  Emit(sink, prefix + "attn.context", out,
       meta("token,channel", "attention output before the O projection",
            "O projection", ""));
  return out;
}

Tensor FeedForward(const Tensor& x, const LayerWeights& w,
                   const Llama3Config& config, TensorSink* sink,
                   const std::string& prefix) {
  const int64_t layer = LayerFromPrefix(prefix);
  auto meta = [&](const char* dims, const char* desc, const char* consumer,
                  const char* fit) {
    TensorMeta m;
    m.dims = dims;
    m.description = desc;
    m.consumer = consumer;
    m.fit_role = fit;
    m.layer = layer;
    return m;
  };
  Tensor gate = Linear(x, w.wgate);
  Tensor up = Linear(x, w.wup);
  Tensor silu;
  Tensor act = SwiGlu(gate, up, &silu);
  Tensor down = Linear(act, w.wdown);

  Emit(sink, prefix + "ffn.gate", gate,
       meta("token,channel", "gate projection output", "SiLU fit", "silu"));
  Emit(sink, prefix + "ffn.up", up,
       meta("token,channel", "up projection output", "gate product", ""));
  Emit(sink, prefix + "ffn.silu", silu,
       meta("token,channel", "silu(gate)", "gate product", ""));
  Emit(sink, prefix + "ffn.swiglu", act,
       meta("token,channel", "silu(gate) * up", "down projection", ""));
  Emit(sink, prefix + "ffn.down", down,
       meta("token,channel", "down projection output, before the residual add",
            "residual add", ""));
  (void)config;
  return down;
}

Tensor AttentionSublayer(const Tensor& x, const LayerWeights& w,
                         const Llama3Config& config, int64_t position_offset,
                         TensorSink* sink, const std::string& prefix) {
  const int64_t layer = LayerFromPrefix(prefix);
  auto meta = [&](const char* dims, const char* desc, const char* consumer,
                  const char* fit) {
    TensorMeta m;
    m.dims = dims;
    m.description = desc;
    m.consumer = consumer;
    m.fit_role = fit;
    m.layer = layer;
    return m;
  };

  Tensor sum_sq, mean_sq, inv_rms;
  Tensor n = RmsNorm(x, w.attn_norm, config.rms_norm_eps, &sum_sq, &mean_sq,
                     &inv_rms);
  Emit(sink, prefix + "attn_norm.sum_sq", sum_sq,
       meta("token", "sum of squares over channels", "1/sqrt fit (folded)",
            config.rmsnorm_fold_mean_into_fit ? "rsqrt" : ""));
  Emit(sink, prefix + "attn_norm.mean_sq", mean_sq,
       meta("token", "sum_sq/hidden + eps", "1/sqrt fit",
            config.rmsnorm_fold_mean_into_fit ? "" : "rsqrt"));
  Emit(sink, prefix + "attn_norm.inv_rms", inv_rms,
       meta("token", "1/sqrt(mean_sq)", "broadcast product", ""));
  Emit(sink, prefix + "attn_norm.out", n,
       meta("token,channel", "RMSNorm output, weight applied",
            "Q/K/V projections", ""));

  Tensor q = Linear(n, w.wq);
  Tensor k = Linear(n, w.wk);
  Tensor v = Linear(n, w.wv);
  Emit(sink, prefix + "q_proj", q, meta("token,channel", "Wq n", "RoPE", ""));
  Emit(sink, prefix + "k_proj", k, meta("token,channel", "Wk n", "RoPE", ""));
  Emit(sink, prefix + "v_proj", v,
       meta("token,channel", "Wv n", "PV product", ""));

  if (config.rope_enabled) {
    ApplyRope(&q, config.num_heads, config.head_dim, config.rope_theta,
              position_offset, config.rope_convention);
    ApplyRope(&k, config.num_kv_heads, config.head_dim, config.rope_theta,
              position_offset, config.rope_convention);
  }
  Emit(sink, prefix + "q_rope", q,
       meta("token,channel", config.rope_enabled ? "RoPE(Wq n)"
                                                 : "Wq n (RoPE DISABLED)",
            "QK^T product", ""));
  Emit(sink, prefix + "k_rope", k,
       meta("token,channel", config.rope_enabled ? "RoPE(Wk n)"
                                                 : "Wk n (RoPE DISABLED)",
            "QK^T product", ""));

  Tensor ctx = GqaAttention(q, k, v, config, position_offset, sink, prefix);
  Tensor proj = Linear(ctx, w.wo);
  Emit(sink, prefix + "o_proj", proj,
       meta("token,channel", "Wo applied to the attention context",
            "residual add", ""));

  Tensor out = x;
  AddInPlace(&out, proj);
  return out;
}

Tensor DecoderBlock(const Tensor& x, const LayerWeights& w,
                    const Llama3Config& config, int64_t position_offset,
                    TensorSink* sink, const std::string& prefix) {
  w.Validate(config);
  if (x.Rank() != 2 || x.Dim(1) != config.hidden_size)
    throw std::invalid_argument("DecoderBlock: input must be [tokens, hidden]");
  const int64_t layer = LayerFromPrefix(prefix);
  auto meta = [&](const char* dims, const char* desc, const char* consumer) {
    TensorMeta m;
    m.dims = dims;
    m.description = desc;
    m.consumer = consumer;
    m.layer = layer;
    return m;
  };

  Emit(sink, prefix + "x_in", x,
       meta("token,channel", "residual stream entering the block",
            "attention RMSNorm"));

  Tensor mid = AttentionSublayer(x, w, config, position_offset, sink, prefix);
  Emit(sink, prefix + "x_mid", mid,
       meta("token,channel", "residual stream after the attention half",
            "FFN RMSNorm"));

  Tensor sum_sq, mean_sq, inv_rms;
  Tensor n = RmsNorm(mid, w.ffn_norm, config.rms_norm_eps, &sum_sq, &mean_sq,
                     &inv_rms);
  {
    TensorMeta m = meta("token", "sum of squares over channels", "1/sqrt fit");
    m.fit_role = config.rmsnorm_fold_mean_into_fit ? "rsqrt" : "";
    Emit(sink, prefix + "ffn_norm.sum_sq", sum_sq, m);
    m = meta("token", "sum_sq/hidden + eps", "1/sqrt fit");
    m.fit_role = config.rmsnorm_fold_mean_into_fit ? "" : "rsqrt";
    Emit(sink, prefix + "ffn_norm.mean_sq", mean_sq, m);
  }
  Emit(sink, prefix + "ffn_norm.inv_rms", inv_rms,
       meta("token", "1/sqrt(mean_sq)", "broadcast product"));
  Emit(sink, prefix + "ffn_norm.out", n,
       meta("token,channel", "RMSNorm output, weight applied",
            "gate/up projections"));

  Tensor ffn = FeedForward(n, w, config, sink, prefix);
  Tensor out = mid;
  AddInPlace(&out, ffn);
  Emit(sink, prefix + "x_out", out,
       meta("token,channel", "residual stream leaving the block",
            "next block / final norm"));
  return out;
}

Tensor Forward(const Tensor& x, const std::vector<LayerWeights>& layers,
               const Llama3Config& config, int64_t position_offset,
               TensorSink* sink) {
  Tensor cur = x;
  for (size_t i = 0; i < layers.size(); ++i) {
    const std::string prefix = "L" + std::to_string(i) + ".";
    cur = DecoderBlock(cur, layers[i], config, position_offset, sink, prefix);
  }
  return cur;
}

const std::vector<BoundaryDecl>& LayerBoundaries() {
  static const std::vector<BoundaryDecl> kDecls = {
      {"x_in", "token,channel", "tokens x hidden", "",
       "residual stream entering the block"},
      {"attn_norm.sum_sq", "token", "tokens", "rsqrt (folded variant)",
       "sum_c x^2, the reduction an encrypted RMSNorm computes"},
      {"attn_norm.mean_sq", "token", "tokens", "rsqrt",
       "sum_sq/hidden + eps, the 1/sqrt argument"},
      {"attn_norm.inv_rms", "token", "tokens", "",
       "1/sqrt(mean_sq), broadcast across channels"},
      {"attn_norm.out", "token,channel", "tokens x hidden", "",
       "RMSNorm output with the learned weight applied"},
      {"q_proj", "token,channel", "tokens x heads*head_dim", "", "Wq n"},
      {"k_proj", "token,channel", "tokens x kv_heads*head_dim", "", "Wk n"},
      {"v_proj", "token,channel", "tokens x kv_heads*head_dim", "", "Wv n"},
      {"q_rope", "token,channel", "tokens x heads*head_dim", "", "RoPE(Wq n)"},
      {"k_rope", "token,channel", "tokens x kv_heads*head_dim", "",
       "RoPE(Wk n)"},
      {"attn.scores", "head,query,key", "heads x tokens x tokens", "",
       "Q K^T / sqrt(head_dim), before masking"},
      {"attn.exp_input", "head,query,key", "heads x tokens x tokens", "exp",
       "score minus the SoftMax shift"},
      {"attn.exp", "head,query,key", "heads x tokens x tokens", "",
       "exp(shifted score) with the causal mask applied multiplicatively"},
      {"attn.denominator", "head,query", "heads x tokens", "reciprocal",
       "row sum of the masked exponentials"},
      {"attn.reciprocal", "head,query", "heads x tokens", "", "1/denominator"},
      {"attn.probs", "head,query,key", "heads x tokens x tokens", "",
       "attention probabilities, exactly zero where masked"},
      {"attn.context", "token,channel", "tokens x heads*head_dim", "",
       "P V, before the O projection"},
      {"o_proj", "token,channel", "tokens x hidden", "", "Wo applied"},
      {"x_mid", "token,channel", "tokens x hidden", "",
       "residual stream after the attention half"},
      {"ffn_norm.sum_sq", "token", "tokens", "rsqrt (folded variant)",
       "sum_c x^2"},
      {"ffn_norm.mean_sq", "token", "tokens", "rsqrt", "sum_sq/hidden + eps"},
      {"ffn_norm.inv_rms", "token", "tokens", "", "1/sqrt(mean_sq)"},
      {"ffn_norm.out", "token,channel", "tokens x hidden", "",
       "RMSNorm output with the learned weight applied"},
      {"ffn.gate", "token,channel", "tokens x intermediate", "silu",
       "gate projection output"},
      {"ffn.up", "token,channel", "tokens x intermediate", "",
       "up projection output"},
      {"ffn.silu", "token,channel", "tokens x intermediate", "", "silu(gate)"},
      {"ffn.swiglu", "token,channel", "tokens x intermediate", "",
       "silu(gate) * up"},
      {"ffn.down", "token,channel", "tokens x hidden", "",
       "down projection output, before the residual add"},
      {"x_out", "token,channel", "tokens x hidden", "",
       "residual stream leaving the block"},
  };
  return kDecls;
}

std::string NumericalConventions() {
  return
R"(Numerical conventions of the plaintext oracle
============================================
1.  Arithmetic is IEEE-754 binary64 (double) everywhere. No float32 pass and
    no bfloat16 pass. Rationale: the oracle must be the *value* reference, so
    its own rounding must be far below the CKKS error it is used to measure
    (the measured Cheddar bootstrap error is ~4.5e-05, twelve orders of
    magnitude above float64 rounding at these widths).
    Consequence: comparing the oracle against a float32 HuggingFace run will
    show ~1e-7 relative differences that belong to the float32 run.

2.  Every reduction is a single sequential float64 accumulator in ascending
    index order. No pairwise summation, no Kahan, no BLAS, no OpenMP, no
    reassociation. `Linear` sums over the input feature index; the attention
    score sums over head_dim; the RMSNorm sums over channels; the PV product
    sums over the key index.
    Build flags enforce this: /fp:precise on MSVC, -ffp-contract=off and no
    -ffast-math on GCC/Clang. FMA contraction is disabled because it changes
    the result and is not portable between hosts.

3.  Randomness is SplitMix64 plus an explicit Box-Muller transform written
    inside this project (oracle/include/oracle/Rng.h). Nothing depends on
    std::normal_distribution, whose variate consumption is implementation
    defined. Same seed, same tensors, on MSVC and on GCC.

4.  The causal mask multiplies the EXPONENTIALS by 0/1 rather than adding
    -infinity to the scores, because that is what an encrypted SoftMax does.
    Masked probabilities are therefore exactly +0.0, not a small number.

5.  RMSNorm computes 1/sqrt(sum_c x^2 / hidden + eps) with eps INSIDE the
    square root, matching the Llama reference. `sum_sq` and `mean_sq` are both
    exported so a folded-mean encrypted variant can be calibrated against the
    interval it actually sees.

6.  RoPE is applied to Q and K only, never to V, and its angle for pair index
    c is position * theta^(-2c/head_dim). Position is absolute
    (position_offset + token index).

7.  Attention scores are scaled by 1/sqrt(head_dim) BEFORE the SoftMax shift,
    so every exported score is already scaled.

8.  Determinism claim, stated precisely: two runs of the same binary on the
    same host with the same inputs are bit-identical, and the exported
    checksums prove it. Across hosts and compilers the results agree to the
    last ulp of libm's exp/log/cos/sin/pow, which is NOT guaranteed to be
    bit-identical. Cross-host equality must therefore be asserted with the
    tolerance-based comparison utility, never with checksums.
)";
}

}  // namespace oracle
