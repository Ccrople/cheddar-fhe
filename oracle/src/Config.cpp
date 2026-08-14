// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Config.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace oracle {

double Llama3Config::AttentionScale() const {
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

void Llama3Config::Validate() const {
  auto fail = [](const std::string& m) {
    throw std::invalid_argument("Llama3Config: " + m);
  };
  if (hidden_size <= 0) fail("hidden_size must be positive");
  if (head_dim <= 0) fail("head_dim must be positive");
  if (head_dim % 2 != 0) fail("head_dim must be even (RoPE pairs channels)");
  if (num_heads <= 0) fail("num_heads must be positive");
  if (num_kv_heads <= 0) fail("num_kv_heads must be positive");
  if (num_heads % num_kv_heads != 0)
    fail("num_heads must be divisible by num_kv_heads (GQA group size)");
  if (num_heads * head_dim != hidden_size)
    fail("num_heads * head_dim must equal hidden_size");
  if (intermediate_size <= 0) fail("intermediate_size must be positive");
  if (num_layers <= 0) fail("num_layers must be positive");
  if (!(rms_norm_eps > 0.0)) fail("rms_norm_eps must be positive");
  if (!(rope_theta > 1.0)) fail("rope_theta must be greater than 1");
  if (max_position_embeddings <= 0) fail("max_position_embeddings must be positive");
}

std::string Llama3Config::Identity() const {
  std::ostringstream os;
  os << "llama3"
     << " d=" << hidden_size << " i=" << intermediate_size
     << " L=" << num_layers << " h=" << num_heads << " kv=" << num_kv_heads
     << " hd=" << head_dim << " eps=" << rms_norm_eps
     << " theta=" << rope_theta << " rope=" << ToString(rope_convention)
     << (rope_enabled ? "" : " ROPE_OFF")
     << " softmax=" << ToString(softmax_mode);
  return os.str();
}

Llama3Config Llama3_8B() {
  Llama3Config c;  // defaults are the 8B values
  c.Validate();
  return c;
}

Llama3Config Llama3Reduced(int64_t heads, int64_t kv_heads, int64_t head_dim,
                           int64_t intermediate) {
  Llama3Config c = Llama3_8B();
  c.num_heads = heads;
  c.num_kv_heads = kv_heads;
  c.head_dim = head_dim;
  c.hidden_size = heads * head_dim;
  c.intermediate_size = intermediate;
  c.num_layers = 2;
  c.vocab_size = 0;  // no embedding table in the reduced model
  c.Validate();
  return c;
}

const char* ToString(RopeConvention c) {
  switch (c) {
    case RopeConvention::kHalfSplit: return "half_split";
    case RopeConvention::kInterleaved: return "interleaved";
  }
  return "unknown";
}

const char* ToString(SoftmaxMode m) {
  switch (m) {
    case SoftmaxMode::kRowMax: return "row_max";
    case SoftmaxMode::kFixedShift: return "fixed_shift";
    case SoftmaxMode::kNaive: return "naive";
  }
  return "unknown";
}

std::string ConfigProvenance() {
  return
R"(Llama-3-8B configuration -- provenance
======================================
Every value below was read out of a file in this checkout or in the read-only
HEonGPU reference tree on 2026-08-14. Nothing is quoted from memory.

VERIFIED FROM REPOSITORY EVIDENCE
---------------------------------
  vocab_size             128256   Projects/HEonGPU/benchmark/fetch_llama3_weights.py:60
  hidden_size (d_model)    4096   fetch_llama3_weights.py:61  (D_MODEL)
                                  reference/LLAMA3_8B_LAYER_FLOW.md:987-988
  intermediate_size       14336   fetch_llama3_weights.py:62  (HIDDEN)
                                  reference/LLAMA3_8B_LAYER_FLOW.md:987-988
  num_layers                 32   fetch_llama3_weights.py:63  (LAYERS)
  num_heads                  32   fetch_llama3_weights.py:64  (HEADS)
  num_kv_heads                8   fetch_llama3_weights.py:65  (KV_HEADS)
  head_dim                  128   fetch_llama3_weights.py:66  (HEAD_DIM)
  rope_theta            500000.0  fetch_llama3_weights.py:67  (ROPE_THETA)
                                  HEonGPU src/include/heongpu/host/ckks/
                                    llama3_batch.cuh:477 ("Llama-3's base is
                                    500000; Llama-2's is 10000")
                                    llama3_batch16.cuh:600, llama3_prep.cuh:108,
                                    llama3_rect.cuh:1175
  rms_norm_eps             1e-5   fetch_llama3_weights.py:68  (RMS_EPS)
                                  HEonGPU llama3.cuh:510, llama3_batch.cuh:843,
                                    llama3_batch16.cuh:388, llama3_prep.cuh:236,
                                    llama3_rect.cuh:1007
  max_position_embeddings  8192   reference/LLAMA3_8B_LAYER_FLOW.md:2442
                                  ("Llama-3-8B's context is 8192")

  Cross-check, independent of the constants above:
    hidden_size == num_heads * head_dim   ->  4096 == 32 * 128   OK
    GQA group  == num_heads / num_kv_heads ->  32 / 8 = 4        OK
    kv_channels == num_kv_heads * head_dim ->  8 * 128 = 1024,
      matching LLAMA3_8B_LAYER_FLOW.md:3003 ("kv_channels stays at 1024
      against q_channels 4096")

ASSUMED, NOT VERIFIED HERE -- standard Llama-3 architecture, flagged so a
later session can confirm against a real config.json rather than inherit it:
  - no bias term on any of q/k/v/o/gate/up/down projections
  - normalisation is RMSNorm (not LayerNorm), applied pre-attention and
    pre-FFN, with the residual stream added *after* each sublayer
  - no RoPE frequency scaling (Llama 3.0 has none; Llama 3.1 adds a "llama3"
    rope_scaling block -- if the weights supplied later are 3.1, this oracle
    is wrong until scaling is added)
  - the FFN is SwiGLU: down(silu(gate(x)) * up(x))
  - reference dtype: HuggingFace ships bfloat16; the fetch script promotes to
    float32; this oracle computes in float64 (see NumericalConventions())

THE ONE CONVENTION THAT IS NOT A CONSTANT
-----------------------------------------
RoPE channel pairing. See RopeConventionNote(). Getting it wrong changes every
Q and K tensor while leaving every shape and every norm intact, so no shape
assertion catches it.
)";
}

std::string RopeConventionNote() {
  return
R"(RoPE channel pairing is a property of the weight file, not a preference.

  half_split  (default)  channel c pairs with c + head_dim/2, angle index c.
                         This is HuggingFace's `rotate_half`. It is what
                         HEonGPU/benchmark/fetch_llama3_weights.py:apply_rope
                         implements, and it is correct for weights taken from
                         a HuggingFace-format `model.safetensors` repo.

  interleaved            channel 2c pairs with 2c+1, angle index c. This is
                         Meta's original reference implementation, correct for
                         the `consolidated.00.pth` checkpoint layout.

The two are related by a fixed permutation of the *rows of Wq and Wk*. HF
applies that permutation once at conversion time, which is why both are
"correct Llama-3" and why the choice must follow the file you loaded.

Detection rule for a later session: load real weights, run both conventions,
and compare the resulting attention output against a trusted forward pass. A
shape check will not distinguish them, and neither will any norm: RoPE is
orthogonal in both conventions, so |q| is identical either way. The unit test
`RopeConventionsDiffer` exists to keep this from silently collapsing.
)";
}

}  // namespace oracle
