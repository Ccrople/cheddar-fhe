#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/CiBatch.h"
#include "extension/CiBatchAttention.h"
#include "extension/EvalPoly.h"

namespace cheddar {

#ifdef USE_CUBLAS

/**
 * @brief A Llama-3 decoder layer on the BATCHED layout (`CiBatchLayout`):
 * slot-resident from end to end, the projections [KANG] Algorithm 1, the
 * channel reductions sums of ciphertexts, and every crossing a plain
 * `Boot`.
 *
 * ## The feed-forward half, as it is built here
 *
 *     stream  (model cts at level 0, carrying `stream_scale`)
 *       Boot each               -> level `top` (the boot's end level)
 *       acc = sum_c x_c (x) x_c -> ONE relinearize + rescale     [pass A]
 *       u = alpha (sink^2 S / (H s^2) + eps), v = (u - b) / a    one pt
 *                                                    mult, one const add
 *       r = invsqrt(v)          -> Chebyshev, `Log2Ceil(deg + 1)` levels
 *       r' = r (.) sink         -> one pt mult (the per-token rescale)
 *       y_c = x_c (.) r'        -> at `norm_apply_level - 1`        [pass B]
 *     per hidden chunk (one tile of the gate/up operands):
 *       g = y W_gate' , u = y W_up'     Algorithm 1, gain and 1/range folded
 *       h = SiLU(g) (.) u
 *       d += h W_down'                   Algorithm 1 over the chunk's rows
 *     res = stream + d                   both at level 0
 *
 * The sink rescale ([SYLPH] 3.1.1's public per-token factor) FEEDS THE NORM
 * AND NOTHING ELSE, exactly as `reference_forward.py` applies it: RMSNorm is
 * scale invariant per token, so it changes only where the polynomial's
 * argument lands, and here it costs two plaintext multiplies on two SINGLE
 * ciphertexts -- `sink^2` folded into the affine map's own multiply on the
 * accumulator, and `sink` onto `r` -- against one per ciphertext on the
 * single-prompt layer.
 *
 * ## Levels, and why the norm's output is held LOW
 *
 * The hidden width is 14336 ciphertexts and a ciphertext at level L holds
 * `(L + 2)` limbs of 65536 words: the FFN cannot hold its hidden at the
 * norm's natural output level (12) -- that is 90 GiB. [BAE]'s rule applies
 * here in memory as well as in time: the matrix products run at the LOWEST
 * level the chain allows. So the booted channel is brought down to
 * `norm_apply_level` right after its square is taken, `r'` is brought to the
 * same level, and everything after runs on 7-limb-and-smaller ciphertexts:
 * gate/up at 7 -> 6, SiLU (degree 15, four levels) -> 2, the product -> 1,
 * down -> 0, the residual at 0. The hidden is walked one tile at a time
 * (`Config::rows_per_tile` channels of gate and up, the same rows of down),
 * so at most one tile of the hidden is alive.
 *
 * ## What the caller owns
 *
 * The `BootContext` (its EvalMod and FFT tables prepared, its rotation keys
 * made from `AddRequiredRotations`), the weights on the device in
 * `reference/export_layers.py`'s `[in][out]` f32 layout, and the
 * calibration, which is the reference script's `calib.json` read for the
 * layer -- fitted offline on the clear model, never measured in the run.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiBatchLayer {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

 public:
  struct Config {
    int num_tokens = 128;
    int model = 4096;
    int hidden = 14336;
    double eps = 1e-5;
    //! The CC-MM chain's addressing (`CiBatchLayout` chain-addressed):
    //! the lanes of an instance group and the ring switch's rank, as the
    //! attention's `GetChain()` has them. Both 0 = the plain map. A layer
    //! that feeds `CiBatchAttention` MUST use its map, or the per-token
    //! plaintexts of the norms address the wrong slots.
    int lanes = 0;
    int rank = 0;
    //! Output channels per projection tile, and the hidden chunk the
    //! feed-forward walks; must divide `hidden`. A tile's GEMM output and
    //! its batched rescale are `2 * rows * limbs * degree` words each, so
    //! at level 7 (10 limbs) 512 rows is 2.7 + 2.4 GiB in flight.
    int rows_per_tile = 512;
    //! Where the booted channel and the inverse square root meet; the
    //! normalised stream lands one below. 8 lands the feed-forward at 0,
    //! and it is the LOWEST that does: SiLU's four levels, the product and
    //! the down projection need six below the norm's output.
    int norm_apply_level = 8;
    //! The ATTENTION norm's hold. 7 is tight: y at 6, the q/k/v
    //! projections at 5 (= the attention's rope_level), RoPE at 4 = the
    //! forward level -- no LevelDown wasted, and the normalised split and
    //! the q/k/v tiles are one limb smaller than at the feed-forward's 8.
    int norm_apply_level_attn = 7;
    //! Keep every booted channel (at `norm_apply_level`, 11 limbs = 23.6
    //! GiB at the model's width) between the sum of squares and the apply,
    //! or boot each channel twice and hold nothing. The first is 4096
    //! bootstraps cheaper a norm, the second 24 GiB smaller.
    //! The attention's norm holds nothing by default: the converters and
    //! the split of the normalised stream already stand beside it. The
    //! feed-forward's norm holds (`hold_channels_ffn`): its peak was 55 GiB
    //! with them held, and 4096 bootstraps are four minutes a layer.
    bool hold_channels = false;
    bool hold_channels_ffn = true;
    //! Drop the boot's CoeffToSlot/SlotToCoeff tables (6.4 GiB on
    //! `ci16_35`) once a norm's bootstraps are done and rebuild them at the
    //! next norm: nothing between two norms bootstraps, and the tables are
    //! the largest single object the Context holds.
    bool release_boot_tables = true;
    //! Keep the residual stream ON THE CARD while a half runs, instead of
    //! parking its 4.3 GiB in host memory between the norm and the
    //! residual add. The default is the A100 configuration (parked); a
    //! box with headroom (the B200's 179 GiB) skips the two copies.
    bool park_stream = true;
    //! Unstage the attention's converters around the feed-forward
    //! (`CiBatchAttention::UnstageConverters` at the seam). The default is
    //! the A100 configuration (unstaged); with headroom they stay put.
    bool unstage_converters = true;
    //! SiLU's Chebyshev degree (four levels at 15; the range is folded
    //! into the gate weight, so the polynomial is fitted on [-1, 1]).
    int silu_degree = 15;
    //! 0 derives the invsqrt degree from the window as `CiLlamaLayer`
    //! does: 9 up to 2.5, 15 up to 12, 31 beyond.
    int rms_degree = 0;
    bool verbose = false;
  };

  /** @brief What `reference_forward.py` fits for one layer. */
  struct Calibration {
    //! The feed-forward norm's layer constant and invsqrt window.
    double alpha = 1.0;
    double norm_window = 2.0;
    //! The same two for the pre-attention norm.
    double attn_alpha = 1.0;
    double attn_norm_window = 2.0;
    //! SiLU's fitted range, `margin * max|gate|`.
    double silu_range = 1.0;
    //! The factor the residual stream carries so that it rides into the
    //! bootstrap at the height EvalMod likes; the down (and O) weights put
    //! it back on their output.
    double stream_scale = 1.0;
    //! The public per-token sink rescale at each norm; empty = none.
    std::vector<double> attn_sink, ffn_sink;

    //! THE ATTENTION'S OWN. `cq` and `ck` are the factors the Q and K
    //! weights carry so that the scores ride into their bootstrap at the
    //! height EvalMod likes: `cq * ck = ride / max|s_raw|`. The softmax
    //! then works in CHAIN units (raw scores times `cq * ck`): `span_raw =
    //! s_raw_max - s_raw_min`, the shift `s_raw_max`, the per-(head, row)
    //! live-key maximum `row_shift_raw`, the live row-norm estimate
    //! `row_norm` (invariant), and `m_eff = span_raw / sqrt(D)` -- all
    //! `reference_forward.py`'s.
    double cq = 1.0, ck = 1.0;
    double span_raw = 1.0, s_raw_max = 0.0, m_eff = 8.0;
    std::vector<std::vector<double>> row_shift_raw, row_norm;
  };

  /** @brief The attention half's tensors on the device, `[in][out]` f32. */
  struct AttnWeights {
    const float *q = nullptr;  //!< `[model][heads * D]`
    const float *k = nullptr;  //!< `[model][kv_heads * D]`
    const float *v = nullptr;  //!< `[model][kv_heads * D]`
    const float *o = nullptr;  //!< `[heads * D][model]`
    std::vector<double> attn_norm;  //!< `model` gains
  };

  /** @brief One layer's tensors on the device, `[in][out]` f32. */
  struct Weights {
    const float *gate = nullptr;  //!< `[model][hidden]`
    const float *up = nullptr;    //!< `[model][hidden]`
    const float *down = nullptr;  //!< `[hidden][model]`
    //! The two RMSNorm gains, `model` each; folded into the weights that
    //! read the normalised stream.
    std::vector<double> ffn_norm, attn_norm;
  };

  //! `boot` is not const: the layer prepares and drops its transform
  //! tables around each norm (`Config::release_boot_tables`).
  CiBatchLayer(std::shared_ptr<BootContext<word>> boot, const Config &cfg);

  CiBatchLayer(const CiBatchLayer &) = delete;
  CiBatchLayer &operator=(const CiBatchLayer &) = delete;

  const CiBatchLayout &GetLayout() const { return layout_; }
  CiBatchProjection<word> &GetProjection() { return *proj_; }
  //! The boot's rotations; nothing here rotates on its own.
  void AddRequiredRotations(EvkRequest &req) const;

  /**
   * @brief Put the norm's CHANNEL bootstraps on a short landing ring
   * (Doing.md 7.38): with it set, `NormTurn` sums the squares BEFORE any
   * bootstrap (at the stream's own level, which must be >= 1), boots the
   * ONE accumulator on the deep layer ring, and boots each channel exactly
   * once -- on `chan` -- for the apply. The deep ladder then serves two
   * bootstraps a layer, and the 8,192 channel boots ride `chan`'s shorter
   * climb. `hold`/`hold_channels` become moot (there is no second pass to
   * hold for). `chan` must share the layer's secret and its bottom primes
   * up to its landing (a `gen_landing.py` sub-ladder of the layer preset),
   * and `Config::norm_apply_level` must be <= `chan`'s landing.
   *
   * The caller keeps `chan`'s EvalMod and FFT tables prepared; this class
   * prepares/releases them beside the deep ring's
   * (`Config::release_boot_tables`).
   */
  void SetChannelBoot(std::shared_ptr<BootContext<word>> chan,
                      const EvkMap<word> *chan_evk) {
    chan_boot_ = std::move(chan);
    chan_evk_ = chan_evk;
  }

  /**
   * @brief The residual stream through one RMSNorm: booted, squared and
   * summed over channels, the inverse square root on the ONE accumulator,
   * applied back. The gain is NOT applied -- fold it into the weight that
   * reads `y` (`CiBatchProjection::FoldGain`).
   *
   * The output is not `model` ciphertexts but their SPLIT -- the int8
   * source every projection reading the normalised stream takes
   * (`CiBatchProjection::Source`, at `Config::norm_apply_level - 1`, in
   * the model's own units: the norm is scale invariant, and `stream_scale`
   * and `sink` are told to the affine map so that it is). Each normalised
   * channel is split the moment it exists and dropped, because at the
   * model's width the channels and their split are 21 GiB each and never
   * need to coexist.
   *
   * @param stream `model` ciphertexts, any level, carrying `stream_scale`
   * @param sink the per-token rescale, or empty
   * @param hold keep the booted channels between the two passes
   *        (`Config::hold_channels` / `hold_channels_ffn`)
   * @param release_tables drop the boot tables after the last bootstrap.
   *        Only where nothing boots until the phase's live set has shrunk
   *        (the feed-forward: SiLU and down bootstrap nothing). The
   *        attention boots at its FIRST head, with the split and a tile of
   *        heads live: releasing here just re-prepares 18.6 GiB into a
   *        fragmented pool, which is the OOM of 2026-09-02.
   */
  void NormTurn(typename CiBatchProjection<word>::Source &src,
                const std::vector<Ct> &stream, double alpha, double window,
                const std::vector<double> &sink, double stream_scale,
                const EvkMap<word> &evk, bool hold, int hold_level,
                bool release_tables) const;

  /**
   * @brief The whole feed-forward half: norm, gate/up, SiLU, down, residual.
   *
   * @param res the next residual stream, `model` ciphertexts at level 0,
   *        carrying `stream_scale`
   * @param stream the post-attention residual, `model` ciphertexts at
   *        level 0, carrying `stream_scale`; PARKED in host memory while
   *        the half runs and put back, word for word, for the residual
   */
  void FeedForward(std::vector<Ct> &res, std::vector<Ct> &stream,
                   const Weights &w, const Calibration &c,
                   const EvkMap<word> &evk);

  /**
   * @brief The whole attention half: norm, the Q/K/V projections, per head
   * the scores, their bootstraps, the softmax and P V, the O projection,
   * the residual. One kv group (four heads) is in flight at a time.
   *
   * @param res the post-attention residual, `model` ciphertexts at level 0,
   *        carrying `stream_scale`
   * @param stream the layer's input, `model` ciphertexts at level 0,
   *        carrying `stream_scale`
   * @param attn the attention products, whose layout this layer's is
   * @param akeys the attention's keys on its three rings
   * @param dbg when given, copies of one head's intermediates land here
   *        so a test can compare the REAL path's stages against a host
   *        mirror; `CHEDDAR_CI_BATCH_MAX_KV` limits the kv groups run for
   *        the same purpose (the residual then holds only those heads)
   */
  struct AttnDebug {
    int head = 0;
    std::vector<Ct> q, k, v, scores, booted, P, out;
  };
  void Attention(std::vector<Ct> &res, std::vector<Ct> &stream,
                 const AttnWeights &w, const Calibration &c,
                 CiBatchAttention<word> &attn,
                 const typename CiBatchAttention<word>::Keys &akeys,
                 const EvkMap<word> &evk, AttnDebug *dbg = nullptr);

  /** @brief One whole layer: the attention half, then the feed-forward. */
  void Layer(std::vector<Ct> &res, std::vector<Ct> &stream,
             const AttnWeights &aw, const Weights &fw, const Calibration &c,
             CiBatchAttention<word> &attn,
             const typename CiBatchAttention<word>::Keys &akeys,
             const EvkMap<word> &evk) {
    std::vector<Ct> mid;
    Attention(mid, stream, aw, c, attn, akeys, evk);
    // The seam is memory-bound (chain1m: 42.5 GiB live here, and the
    // feed-forward wants its 20 GiB arena plus its churn on top): the
    // input stream is dead -- `mid` replaced it -- and the converters are
    // unused until the next layer's attention.
    stream.clear();
    if (cfg_.unstage_converters) attn.UnstageConverters();
    const Stages a = stages_;
    FeedForward(res, mid, fw, c, evk);
    if (cfg_.unstage_converters) attn.StageConverters();
    stages_.boot += a.boot;
    stages_.norm += a.norm;
    stages_.qkv = a.qkv;
    stages_.scores = a.scores;
    stages_.softmax = a.softmax;
    stages_.values = a.values;
    stages_.o = a.o;
    stages_.total += a.total;
  }

  //! Device seconds of the last half's stages (host clocks around
  //! synchronised spans).
  struct Stages {
    double boot = 0.0, norm = 0.0, gate_up = 0.0, silu = 0.0, down = 0.0;
    double qkv = 0.0, scores = 0.0, softmax = 0.0, values = 0.0, o = 0.0;
    double total = 0.0;
  };
  const Stages &GetStages() const { return stages_; }

 private:
  int NormDegree(double window) const;

  /**
   * @brief The residual stream parked in host memory while a half runs.
   *
   * A half reads the stream at its norm and again at its residual add, and
   * between the two it needs every byte the card has: 4096 level-0
   * ciphertexts are 4.3 GiB. `Park` copies them out and frees them,
   * `Unpark` puts the same words back.
   */
  struct ParkedStream {
    std::vector<HostVector<word>> bx, ax;
    std::vector<NPInfo> np;
    std::vector<double> scale;
    std::vector<int> slots;
  };
  void Park(ParkedStream &parked, std::vector<Ct> &stream) const;
  void Unpark(std::vector<Ct> &stream, ParkedStream &parked) const;

  std::shared_ptr<BootContext<word>> boot_;
  //! The channel-boot ring (`SetChannelBoot`), or null = every bootstrap
  //! on `boot_` with the squares summed after the channel boots.
  std::shared_ptr<BootContext<word>> chan_boot_;
  const EvkMap<word> *chan_evk_ = nullptr;
  Config cfg_;
  CiBatchLayout layout_;
  std::unique_ptr<CiBatchProjection<word>> proj_;
  mutable Stages stages_;
  //! Seconds spent (re)building the boot tables, apart from the arithmetic.
  mutable double prepare_seconds_ = 0.0;

 public:
  double GetPrepareSeconds() const { return prepare_seconds_; }
};

#endif  // USE_CUBLAS

}  // namespace cheddar
