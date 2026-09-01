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
    //! Keep every booted channel (at `norm_apply_level`, 11 limbs = 23.6
    //! GiB at the model's width) between the sum of squares and the apply,
    //! or boot each channel twice and hold nothing. The first is 4096
    //! bootstraps cheaper a norm, the second 24 GiB smaller.
    bool hold_channels = true;
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

  CiBatchLayer(std::shared_ptr<const BootContext<word>> boot,
               const Config &cfg);

  CiBatchLayer(const CiBatchLayer &) = delete;
  CiBatchLayer &operator=(const CiBatchLayer &) = delete;

  const CiBatchLayout &GetLayout() const { return layout_; }
  CiBatchProjection<word> &GetProjection() { return *proj_; }
  //! The boot's rotations; nothing here rotates on its own.
  void AddRequiredRotations(EvkRequest &req) const;

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
   */
  void NormTurn(typename CiBatchProjection<word>::Source &src,
                const std::vector<Ct> &stream, double alpha, double window,
                const std::vector<double> &sink, double stream_scale,
                const EvkMap<word> &evk) const;

  /**
   * @brief The whole feed-forward half: norm, gate/up, SiLU, down, residual.
   *
   * @param res the next residual stream, `model` ciphertexts at level 0,
   *        carrying `stream_scale`
   * @param stream the post-attention residual, `model` ciphertexts at
   *        level 0, carrying `stream_scale`
   */
  void FeedForward(std::vector<Ct> &res, const std::vector<Ct> &stream,
                   const Weights &w, const Calibration &c,
                   const EvkMap<word> &evk);

  //! Device seconds of the last `FeedForward`'s stages (event pairs).
  struct Stages {
    double boot = 0.0, norm = 0.0, gate_up = 0.0, silu = 0.0, down = 0.0;
    double total = 0.0;
  };
  const Stages &GetStages() const { return stages_; }

 private:
  int NormDegree(double window) const;

  std::shared_ptr<const BootContext<word>> boot_;
  Config cfg_;
  CiBatchLayout layout_;
  std::unique_ptr<CiBatchProjection<word>> proj_;
  mutable Stages stages_;
};

#endif  // USE_CUBLAS

}  // namespace cheddar
