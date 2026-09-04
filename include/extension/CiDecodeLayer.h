#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/CiBatch.h"
#include "extension/CiDecode.h"

namespace cheddar {

#ifdef USE_CUBLAS

/**
 * @brief One DECODE step of a Llama-3 layer on the batched layout
 * (Doing.md 7.40's design [1], assembled from the gated mechanisms of
 * 7.41-7.45): the op-dependent hybrid -- [KANG] scalar projections on
 * BROADCAST channel ciphertexts, dense packing (`num_tokens` channels into
 * one ciphertext's token rows) ONLY around a bootstrap or a slot-wise
 * polynomial, and the asymmetric KV cache (K in the prefill channel layout
 * verbatim; V token-outside).
 *
 * ## The step
 *
 *     stream (model broadcast cts at level 1, carrying stream_scale)
 *       NormTurn: pack 32 groups -> 32 boots -> squares AFTER the boot
 *         (dense^2 + token ladder), invsqrt at depth, apply, unpack  -> y
 *       q/k/v: Algorithm 1 (attn gain, per-head score gamma AND RoPE at
 *         the step's position folded into the weights)
 *       appends: K row `position` by a masked add per channel; the V
 *         token ciphertext by one pack
 *       per head: scores = D elementwise products (a channel reduction),
 *         the max shift in a constant, ONE boot, the exp walk (deg-7
 *         Chebyshev + squarings), Z ladder + deg-3 reciprocal, the
 *         fan-out (CiDecodeUnpack), ScoreV against V_t, reciprocal LAST,
 *         unpack -> the head's broadcast channels
 *       O (stream_scale folded), residual                       -> mid @ 1
 *       NormTurn -> y2; gate/up (ffn gain folded); pack with the ride
 *         gammas IN THE MASKS; 2 boots; SiLU at depth; the up gamma
 *         unfolded through the declared scale; product; unpack; down
 *         (stream_scale folded) per input-row tile; residual      -> @ 1
 *
 * Both residual streams sit at level 1, so the step CHAINS: the next
 * layer's norm can pack (a pack consumes one level).
 *
 * ## The level ledger (land = the boot's end level, 16 on `ci16_35`)
 *
 *   norm: dense 0 -> boot land -> square land-1 -> affine land-2 ->
 *     invsqrt (deg 7, 3 levels) land-5 -> apply land-6 -> a LevelDown to
 *     the unpack level: the attention's y @ 6 (the V append reads it),
 *     the feed-forward's y2 @ 2 ([BAE]'s rule in memory -- 4096 channels
 *     cost 2.1 GiB a limb, so each stands at the lowest level its
 *     consumers allow)
 *   the V APPEND: a per-token plaintext fold of W_v on y @ 6 (no Kang
 *     projection, no split), landing vt[position] canonical @ 5
 *   scores: q @ 1 x K @ 1 -> 0 -> boot land -> exp affine land-1 ->
 *     poly le = land-1-Log2Ceil(deg+1) -> k squarings -> canonical @ 7
 *   fan-out @ 7 -> e_t @ 6, LevelDown 5 = the V cache's level; ScoreV
 *     -> 4; Z's reciprocal (deg 3, 2 levels) @ 4; the product 3;
 *     unpack 2; O (per head, accumulated) -> 1
 *   The budget needs the walk to end at or above 7 (O's output must reach
 *   level 1), so it caps k at le-7 and lets the per-chunk range grow
 *   instead (deg 7 absorbs a range of ~4 at ~5e-5); an end above 7
 *   descends canonically.
 *
 * Where a polynomial's trimmed degree lands it ABOVE its planned level,
 * the descent is a scale-preserving canonical rescale (a 1.0 plaintext
 * multiply per level), never a bare LevelDown -- the exp walk SQUARES its
 * input k times, and a declared-scale offset squares with it.
 *
 * ## What the caller owns
 *
 * The `BootContext` with its EvalMod and FFT tables prepared and every
 * rotation key of `AddRequiredRotations` made; the K/V cache in `Cache`
 * (K at `KCacheLevel`, rows above `position` zero; V at `VCacheLevel`,
 * the slot at `position` left for the step to fill); the HOST weight
 * tensors (`reference/export_layers.py`'s `[in][out]` f32 -- the step
 * folds RoPE at `position` into q/k on the host, so it takes host
 * pointers where the prefill layer takes device ones); the calibration,
 * fitted offline.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiDecodeLayer {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

 public:
  struct Config {
    int num_tokens = 128;  //!< token rows; must equal the head dimension
    int model = 4096;
    int hidden = 14336;
    int num_heads = 32;
    int num_kv_heads = 8;
    double eps = 1e-5;
    double rope_base = 500000.0;
    //! The height a bootstrapped message rides at: the per-head score
    //! gamma, and the gate/up pack-mask gammas, are `ride / range`.
    double ride = 0.3;
    int invsqrt_degree = 7;
    //! Capped at 7: EvalPoly past a used degree 7 on this path returns
    //! 2^400-scale garbage (Doing.md 7.45); the walk's squarings carry
    //! the range instead.
    int exp_degree = 7;
    //! The Z window is +-10%, where degree 3 is ~1e-5 relative -- and its
    //! two levels (not three) are what lets the O output reach level 1.
    int recip_degree = 3;
    int silu_degree = 15;
    //! 1024: the tile GEMM buffers (rows x 2 x limbs) stay under 3 GiB.
    int rows_per_tile = 1024;
    //! Bootstraps grouped through `BootBatch` (word-for-word the serial
    //! loop; ~24.6 ms/boot at 32 vs ~48 serial on the A100). 32 makes
    //! each site one batch: the norm's 32 groups, a feed-forward tile's
    //! 16 gate/up pairs, the 32 heads' score boots. 1 = serial.
    int boot_group = 32;
    bool verbose = false;
  };

  /** @brief Fitted offline on the clear model, never measured in the run. */
  struct Calibration {
    double stream_scale = 1.0;
    //! Each norm's layer constant and invsqrt window (already widened if
    //! the instances ride factors).
    double attn_alpha = 1.0, attn_window = 1.3;
    double ffn_alpha = 1.0, ffn_window = 1.3;
    //! Per-head RAW score windows over the live cache (`num_heads` each).
    std::vector<double> s_lo, s_hi;
    //! Per-head softmax denominator windows (the class widens +-10%).
    std::vector<double> z_lo, z_hi;
    //! max |gate pre-activation| and max |up pre-activation|.
    double silu_gmax = 1.0, up_umax = 1.0;
  };

  /** @brief HOST tensors, `[in][out]` f32 (`export_layers.py`'s layout). */
  struct HostWeights {
    const float *q = nullptr;     //!< `[model][heads * D]`
    const float *k = nullptr;     //!< `[model][kv_heads * D]`
    const float *v = nullptr;     //!< `[model][kv_heads * D]`
    const float *o = nullptr;     //!< `[heads * D][model]`
    const float *gate = nullptr;  //!< `[model][hidden]`
    const float *up = nullptr;    //!< `[model][hidden]`
    const float *down = nullptr;  //!< `[hidden][model]`
    std::vector<double> attn_norm, ffn_norm;  //!< `model` gains
  };

  /**
   * @brief The per-layer KV cache. `kc[kv][d]`: the prefill K layout
   * verbatim -- channel d's token rows are the cache positions, RoPE'd,
   * rows at and above `position` zero. `vt[kv][t]`: token-outside V --
   * ciphertext t's token rows are the head dimensions of position t;
   * sized `num_tokens`, the slot at `position` filled by the step.
   */
  struct Cache {
    std::vector<std::vector<Ct>> kc, vt;
  };

  struct Stages {
    double norm1 = 0.0, qkv = 0.0, append = 0.0, heads = 0.0, o = 0.0;
    double norm2 = 0.0, gate_up = 0.0, mid = 0.0, down = 0.0, total = 0.0;
  };

  /** @brief Copies of the step's midpoints, for a gate's bisection. */
  struct Debug {
    std::vector<Ct> mid;
  };

  CiDecodeLayer(std::shared_ptr<BootContext<word>> boot, const Config &cfg);
  CiDecodeLayer(const CiDecodeLayer &) = delete;
  CiDecodeLayer &operator=(const CiDecodeLayer &) = delete;

  const CiBatchLayout &GetLayout() const { return layout_; }
  //! The token ladder's strides and the hoisted unpack's baby rotations,
  //! all requested at the boot's landing.
  void AddRequiredRotations(EvkRequest &req) const;

  //! The exp walk's squaring count, from the widest head's range, capped
  //! so that the level budget closes (see the ledger above).
  int ExpSquarings(const Calibration &c) const;
  //! Where e stands for the fan-out: 7, always (the walk lands at or
  //! above it and descends canonically).
  int FanoutLevel() const { return 7; }
  int KCacheLevel() const { return 1; }
  //! ScoreV meets the cache at 5: e_t (6 after the fan-out) LevelDowns
  //! onto it, and the product still reaches the reciprocal at 4.
  int VCacheLevel() const { return 5; }
  int StreamLevel() const { return 1; }

  /**
   * @brief One decode step. `stream` is consumed; `next` is the layer's
   * output residual stream, `model` broadcast channels at level 1 in
   * `stream_scale` units. `position` must be `num_tokens - 1` (a shorter
   * live cache would leave exp(shift) in the dead rows of Z; masking
   * them is the open generalisation).
   */
  void Step(std::vector<Ct> &next, std::vector<Ct> &stream, Cache &cache,
            const HostWeights &w, const Calibration &c, int position,
            const EvkMap<word> &evk, Debug *dbg = nullptr);

  const Stages &GetStages() const { return stages_; }

 private:
  //! Every channel brought to `level` IN PLACE (the copies never coexist:
  //! at the model's width a spare limb set is gigabytes).
  void LowerTo(std::vector<Ct> &x, int level) const;
  //! `Config::verbose`: the phase and the card's free MiB.
  void Note(const char *what) const;
  //! Every input bootstrapped, `Config::boot_group` at a time through
  //! `BootBatch` (word-for-word the serial loop).
  void BootMany(std::vector<Ct> &out, const std::vector<Ct> &in,
                const EvkMap<word> &evk) const;
  /**
   * @brief The residual stream parked in host memory while a half runs
   * (`CiBatchLayer`'s pattern): a half reads its stream at the norm and
   * again at the residual add, and between the two the card wants every
   * byte -- 4096 level-1 ciphertexts are 4.3 GiB.
   */
  struct Parked {
    std::vector<HostVector<word>> bx, ax;
    std::vector<NPInfo> np;
    std::vector<double> scale;
    std::vector<int> slots;
  };
  void Park(Parked &parked, std::vector<Ct> &stream) const;
  void Unpark(std::vector<Ct> &stream, Parked &parked) const;
  //! A canonical descent: one 1.0-plaintext multiply and rescale per
  //! level, so a canonical input stays canonical at `target`.
  void CanonicalTo(Ct &ct, int target) const;
  void EncodeFull(Pt &pt, int level, double scale, double value) const;
  //! The row indicators of one pack, encoded at `level` carrying `value`.
  std::vector<Pt> MakeRowMasks(int level, double value) const;
  //! dense = Rescale(sum_r mask_r * chan[c0 + r]).
  void PackGroup(Ct &dense, const std::vector<Ct> &chan, int c0,
                 const std::vector<Pt> &masks) const;
  //! The hoisted unpack compiled at `pt_level` (scale-preserving), cached.
  CiDecodeUnpack<word> &UnpackAt(int pt_level);
  /**
   * @brief `out = fn(pk_value * in + shift_value)` in the Chebyshev basis:
   * one full-slot plaintext multiply, one constant add, one compiled
   * EvalPoly landing canonical at its own trimmed depth. `fn` is fitted
   * on [-1, 1]; the caller folds its window into `pk_value`/`shift_value`.
   */
  void EvalAtDepth(Ct &out, const Ct &in, double pk_value, double shift_value,
                   const std::function<double(double)> &fn, int degree,
                   const EvkMap<word> &evk) const;
  /**
   * @brief pack -> boot -> squares -> invsqrt at depth -> apply -> a
   * LevelDown to `unpack_pt_level` -> unpack. `y` gets `model` broadcast
   * channels at `unpack_pt_level - 1`, in model units (the stream's
   * factor and sqrt(alpha) ride the reciprocal's declared scale). The
   * level is the caller's because it is a MEMORY choice: 4096 channels
   * cost 2.1 GiB a limb, so each half unpacks at the lowest level its
   * consumers allow (7 for the attention's V append, 3 for the
   * feed-forward's gate/up).
   */
  void NormTurn(std::vector<Ct> &y, const std::vector<Ct> &stream,
                double alpha, double window, double stream_scale,
                int unpack_pt_level, const EvkMap<word> &evk);
  //! The q/k weight fold: the norm gain on the rows, a per-head factor
  //! and RoPE at `position` on the columns. `head_scale` empty = 1.
  std::vector<float> FoldQK(const float *w, int out,
                            const std::vector<double> &gain,
                            const std::vector<double> &head_scale,
                            int position) const;

  std::shared_ptr<BootContext<word>> boot_;
  Config cfg_;
  CiBatchLayout layout_;
  std::unique_ptr<CiBatchProjection<word>> proj_;
  //! The token-row indicator messages, shared by every unpack.
  std::vector<Message> row_masks_;
  std::map<int, std::unique_ptr<CiDecodeUnpack<word>>> unpacks_;
  mutable Stages stages_;
};

#endif  // USE_CUBLAS

}  // namespace cheddar
