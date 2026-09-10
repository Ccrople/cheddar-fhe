#pragma once

#include <vector>

namespace cheddar {

/**
 * @brief [SYLPH] 4.1's PUBLIC PREFILL STAGE: the half of the heterogeneous
 * prefill that carries no FHE at all.
 *
 * The paper is explicit about it -- "process the public prompt P in the clear
 * and generate a public KV cache (Kpub, Vpub); it follows the standard Llama
 * prefill", and "the public prefill stage is performed in the clear, allowing
 * us to fully leverage existing LLM implementations" (4.1). `CiPcAttention`'s
 * interface already enforces that on its side: its `ChunkSource` hands it
 * plain doubles and there is no ciphertext type anywhere on the public
 * operand. This is the piece that FILLS them, for one layer, from the layer's
 * own weights.
 *
 * It is ordinary float arithmetic and it is here, beside `CiPcAttention`,
 * only because the CONVENTION is not ordinary: three things have to match the
 * encrypted branch exactly or the two halves cannot be added.
 *
 * ## 1. The activation. The same normalisation, and the sinks
 *
 * The encrypted branch folds the RMSNorm gain into the projection weights
 * (`CiBatchProjection::FoldGain`, `res[c][o] = gain[c] * w[c][o]`, the gain on
 * the INPUT channel) and feeds them the normalised stream. So the public side
 * must apply `rmsnorm` WITH `attn_norm` and then the raw `wk`/`wv`, which is
 * what `PublicPrefillChunk` does -- the same recipe the batched layer's own
 * host reference uses.
 *
 * The SINK rescale ([SYLPH] 3.4, `reference_forward.py`'s `rescale_sinks`)
 * cannot be done chunk by chunk: it puts the first `sink_tokens` rows at the
 * geometric-mean power of the rest, so it needs every token at once.
 * `PublicSinkRescale` computes the factors from the whole public context and
 * the caller applies them to `hidden` before chunking. Skipping it moves the
 * sink rows' k and v by ~1.5% through the norm's eps, and the sink-heavy
 * heads carry that into the last rows.
 *
 * ## 2. RoPE, and why the encrypted branch does not move
 *
 * `CiBatchAttention::BuildRope` gives query token `t` the angle `t * theta` --
 * it calls the encrypted block position 0. At T = 4096 the encrypted tokens
 * really sit at `ptok .. ptok + etok - 1`, so a public key at absolute `p`
 * has to reach the query at relative distance `(ptok + t) - p`.
 *
 * RoPE is RELATIVE: `<R_i q, R_j k>` depends only on `i - j`. So the whole
 * correction can be taken on the PUBLIC side, where it is free because it is
 * host arithmetic -- rotate the public key at position `p - ptok`, a negative
 * index, and the relative angle is exactly right. The encrypted branch is
 * untouched, which matters because its RoPE plaintexts are built once per
 * layer and its Q and K share them.
 *
 * That is what `rope_pos0` is for: pass `start - ptok`, not `start`.
 *
 * ## 3. The carried factor
 *
 * The encrypted branch's K weights carry `ck` (`gk[i] = attn_norm[i] * ck` in
 * `CiBatchLayer::Attention`) so that the ciphertext stays in range. The public
 * scores have to land in the SAME units as the encrypted ones for one span
 * and one row shift to serve both, so `ck` is folded here too and
 * `CiPcAttention::Calibration::carried` is `cq * ck` for both halves. V
 * carries no such factor on either side.
 *
 * ## What this does NOT do
 *
 * It computes ONE layer's K and V from that layer's input. The public hidden
 * state entering layer L is the caller's: at L = 0 it is the embedding (which
 * `export_layers.py` already writes), and beyond it, it is the public prompt
 * carried through layers 0..L-1 of a standard float Llama forward. That
 * forward is deliberately outside this library -- it is exactly the "existing
 * LLM implementation" the paper says to reuse, and the public tokens are
 * causal, so it never depends on the encrypted ones.
 */
struct PublicPrefillLayer {
  //! `[model][kv_heads * head_dim]`, input-channel major -- the layout the
  //! batched layer reads (`w.k[c * kv_heads * head_dim + col]`).
  const float *wk = nullptr;
  const float *wv = nullptr;
  //! `[model]`, the pre-attention RMSNorm gain.
  const float *attn_norm = nullptr;
  int model = 4096;
  int kv_heads = 8;
  int head_dim = 128;
  double eps = 1e-5;
  double rope_base = 500000.0;  //!< Llama-3's theta
  //! The factor the encrypted branch's K weights carry; see 3 above.
  double ck = 1.0;
};

/**
 * @brief [SYLPH] 3.4's sink rescale over the WHOLE public context: the first
 * `sink_tokens` rows brought to the geometric-mean power of the rest, BEFORE
 * the norm.
 *
 * @param factor out, `[tokens]`; 1.0 everywhere past the sinks
 * @param hidden `[instance][token][model]`
 *
 * The factors are per (instance, token) because the mean power is; the result
 * is `[instance][token]` in the same order.
 */
void PublicSinkRescale(std::vector<double> &factor, const double *hidden,
                       int model, int num_instances, int tokens,
                       int sink_tokens);

/**
 * @brief One chunk of one kv head's public K and V, in the layouts
 * `CiPcAttention::EncodeKeys` and `EncodeValues` document:
 *
 *     k[(c * width + p) * num_instances + b] = Kpub[b][start + p][c]
 *     v[(p * head_dim + c) * num_instances + b] = Vpub[b][start + p][c]
 *
 * Both vectors arrive already sized, as `CiPcAttention::ChunkSource` hands
 * them over.
 *
 * @param hidden `[instance][p][model]` for the chunk's `width` tokens, the
 *        sink rescale ALREADY applied
 * @param rope_pos0 the RoPE position of the chunk's first token. For Sylph's
 *        split this is `start - ptok` (negative), never `start` -- see 2 above.
 */
void PublicPrefillChunk(std::vector<double> &k, std::vector<double> &v,
                        const PublicPrefillLayer &layer, const double *hidden,
                        int num_instances, int width, int kv_head,
                        int rope_pos0);

}  // namespace cheddar
