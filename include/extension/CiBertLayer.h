#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/CiSwitchedCcmm.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/CiLlamaLayer.h"
#include "extension/GeLu.h"
#include "extension/LayerNorm.h"
#include "extension/SylphSchedule.h"

namespace cheddar {

/**
 * @brief One BERT-Base encoder layer on R+, everything except the attention
 * leg: the six projections with their biases, both residuals, both
 * LayerNorms, and the GELU feed-forward.
 *
 * `CiLlamaLayer` is the same half of a Llama decoder layer and this class is
 * its sibling, not its replacement -- it OWNS one, because the projection leg,
 * the seam and the module basis are the CI line's and not the model's. What
 * is BERT's is the order, the two extra operators, and the biases.
 *
 * ## The order, which is the difference that touches everything
 *
 * Llama is pre-norm: the stream is normalised, the normalised copy feeds the
 * projections, and the raw stream carries on down the residual. BERT is
 * POST-norm: the residual is added first and the norm is the last thing the
 * layer does.
 *
 *     [caller] X -> Emit -> q/k/v images -> HalfBootModule -> the leg
 *     [caller] Boot, seam
 *     THIS     AttentionTurn:  O + b_o, + X, LayerNorm  -> H @ op_level
 *     THIS     FeedForward:    W_int + b_int, GELU, W_out + b_out, + H,
 *                              LayerNorm                -> the next stream
 *
 * Three consequences, all of them load-bearing:
 *
 *  - **Q, K and V read the stream RAW.** There is no pre-attention norm and
 *    so no crossing in front of the projections. The stream they read is the
 *    PREVIOUS layer's LayerNorm output, which is already in coefficients at
 *    `op_level` -- so a BERT stream never sits at level 0 and the layer has
 *    one fewer bootstrap group than the Llama one.
 *  - **The stream is self-limiting.** A LayerNorm output is unit variance
 *    times the gain, so its peak-to-rms is 10-15 where Llama's is 132
 *    (`quarot_export.py`). The crossings' ride is not the problem it is on
 *    that line, and the orthogonal rotation that fixes it there cannot be
 *    folded through a post-norm model anyway: the gain sits ON the stream, so
 *    `Q^T diag(g) Q` is not diagonal.
 *  - **The norm's own scaling has to carry the stream factor.** LayerNorm is
 *    scale invariant, so whatever the layer's input carried is gone by its
 *    output; `Calibration::stream_scale` therefore rides the norm's GAIN and
 *    BIAS, not a projection weight, and the next crossing takes it out again.
 *
 * ## What each stage carries, exactly
 *
 * `ToSlot` multiplies the message by `crossing = GetMessageRatio()`, and
 * `ToCoeff` by `kappa = 2^-log_message_ratio / crossing`, so a full turn
 * carries the nominal ratio and each half is separately known (Doing.md
 * 1.5cv/1.5dk; both are DERIVED here, never fitted). Writing `s` for
 * `stream_scale` and `m` for a value in the model's own units:
 *
 *     stream (coefficients)        s * m           -- the norms' gain carries
 *                                  `s_out / kappa`, so `ToCoeff`'s own
 *                                  `kappa` leaves exactly what the next layer
 *                                  will read as its own `s`
 *     q/k/v images, at the leg     q_scale * s * m -> q_scale = c_q / s
 *     O's output                   o_scale * (what the seam carried) * m,
 *                                  and it must be `s * m`
 *     the FFN crossing's input     crossing * int_scale * s * m -> the
 *                                  Canonicalise there divides by all three
 *                                  and by the GELU's range
 *     the out projection's input   kappa * (GELU in model units), because a
 *                                  `ToCoeff` sits in front of it -> so
 *                                  `out_scale = s / kappa`
 *
 * `GetKappa()` and `GetCrossing()` hand the caller both constants, because
 * two of the scalings above are stated in terms of them and a caller that
 * recomputes them will eventually recompute one of them wrongly.
 *
 * ## The biases are plaintext adds in the COEFFICIENT domain
 *
 * All six of them, on the projection's own output, at its own level: a bias
 * is constant along the token axis, so on the module basis its coefficient
 * image is `rec[t * rank + i] = b[declared(i)]` for every `t` -- one
 * `EncodeCoeff` and one `Add`, no level and no key. The caller states the
 * bias at DECLARED output indices, the same indices `Project` takes its
 * weight map in, and this class does the bit reversal (`Project`'s contract:
 * module row `r` of group `g` is declared output `g * rank + rev(r)`).
 *
 * ## The two operators
 *
 * `LayerNormHandler` and `GeLuHandler`; their headers carry the measurements.
 * The LayerNorm costs one level more than RMSNorm (the centring) and gets it
 * back from its degree: the per-token public rescale below collapses the
 * invsqrt window to the calibration's own error, where degree 7 or 9 is
 * enough against the 15 to 23 a raw window needs.
 *
 * ## The per-token rescale, and why it is legitimate
 *
 * LayerNorm is EXACTLY scale invariant -- mean and standard deviation scale
 * together -- so a public per-token factor in front of it cancels
 * identically. It is needed: measured on the real checkpoint, the per-token
 * variance spans 2419x at layer 9's feed-forward norm, against a window a
 * degree-23 invsqrt covers at 30x. The factors ride the crossing's own
 * constant multiply, which makes them free, exactly as [SYLPH] 3.1.1's sink
 * rescale does on the Llama line. What is NOT available here is that
 * section's other half: a bidirectional encoder has no prompt-independent
 * token whose state could be precomputed, so there is no injectable prefix
 * and the factors are calibration, with a calibration's exposure.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiBertLayer {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

 public:
  using DeviceWeights = typename CoeffLinearLeg<word>::DeviceWeights;
  using ProjectionWeight = typename CiLlamaLayer<word>::ProjectionWeight;

  struct Config {
    int num_tokens = 128;  //!< T
    int proj_rank = 512;   //!< the module rank; T * rank = the slot count
    //! BERT-Base's 768 model channels in two rank-512 ciphertexts, and its
    //! 3072 hidden ones in six. The declared width is what the ciphertexts
    //! carry and the live width is what the model has; the norms divide by
    //! the live one and the dead tail stays exactly zero.
    int model_declared = 1024;
    int model_live = 768;
    int hidden_declared = 3072;
    int hidden_live = 3072;
    int num_heads = 12;
    int head_dim = 64;
    double eps = 1e-12;  //!< BERT's LayerNorm epsilon
    int product_level = 1;
    int parents_per_tile = 4;
    //! 0 derives the invsqrt degree from the window, as `RmsNorm.h`'s table
    //! does: 7 up to 4.18, 9 up to 6, 15 up to 12, 23 up to 30.
    int norm_degree = 0;
    bool verbose = false;
  };

  /** @brief One layer's plaintext weights, in either of the leg's forms. */
  struct Weights {
    //! `model_declared x qkv_declared`, one per tensor.
    ProjectionWeight q, k, v;
    //! `attn_declared x model_declared`.
    ProjectionWeight o;
    //! `model_declared x hidden_declared` and back.
    ProjectionWeight inter, out;
    //! The six biases, at DECLARED output indices (see the header).
    const std::vector<double> *bq = nullptr;
    const std::vector<double> *bk = nullptr;
    const std::vector<double> *bv = nullptr;
    const std::vector<double> *bo = nullptr;
    const std::vector<double> *bint = nullptr;
    const std::vector<double> *bout = nullptr;
    //! Both LayerNorms' gain and bias, at declared MODEL channels.
    const std::vector<double> *attn_gain = nullptr;
    const std::vector<double> *attn_bias = nullptr;
    const std::vector<double> *ffn_gain = nullptr;
    const std::vector<double> *ffn_bias = nullptr;
    //! The projection leg's cache name; a repeated tag with different
    //! weights is a wrong layer that still decrypts.
    std::string tag;
  };

  /** @brief What [SYLPH] 3.1 fits offline on the clear model. */
  struct Calibration {
    //! Both norms, on the VARIANCE (not the mean square): alpha at the
    //! geometric midpoint of the range, the window the range itself with the
    //! margin squared. `reference_forward_bert.py` writes both.
    double attn_alpha = 1.0;
    double attn_window = 2.0;
    double ffn_alpha = 1.0;
    double ffn_window = 2.0;
    //! Each norm's invsqrt degree, 0 to derive it from the window as
    //! `Config::norm_degree` does. It is PER NORM because the two norms of
    //! one layer do not want the same degree: the Chebyshev rate for
    //! `x^-1/2` on `[eps, 1]` is `1 + 2 sqrt(eps)`, so the degree a window
    //! needs is `~ ln(1/delta) sqrt(window) / 2` and BERT's windows differ by
    //! two orders of magnitude between the attention norm (17 to 366 over the
    //! twelve layers, with a safety margin already in) and the feed-forward
    //! norm at layers 9 and 10 (12,000). Nothing else moves that degree --
    //! composition, repeated squaring and a Newton refine all leave the rate
    //! alone (`reference/docs/BERT_BASE_B1.md` 11.3).
    int attn_degree = 0;
    int ffn_degree = 0;
    //! Fitted Chebyshev coefficients for each norm's `1/sqrt`, empty to
    //! interpolate. Fit them for the RELATIVE error (the weight divided by
    //! the function): the norm multiplies the centred row by this, so a
    //! relative error passes straight through, while `1/sqrt` itself varies
    //! by 54x over a wide window and an absolute-error objective therefore
    //! spends its accuracy in the wrong place -- worth about a level, and at
    //! degree 31 the absolute objective took the chain to 9e+08.
    std::vector<double> attn_invsqrt, ffn_invsqrt;
    //! The public per-token rescale at each norm, one entry per token, empty
    //! for none. See the header: LayerNorm is exactly scale invariant, so
    //! these cancel and cost nothing.
    std::vector<double> attn_scale, ffn_scale;
    //! What the INPUT stream carries, per model unit: `Emit` reads it, the O
    //! projection's output has to match it (it is added to the input), and
    //! the post-attention crossing divides it out.
    double stream_scale = 1.0;
    //! What this layer WRITES, per model unit -- both norms' output, and so
    //! the next layer's `stream_scale`. Zero means the same as the input's.
    //!
    //! ONE FACTOR FOR A WHOLE CHAIN COSTS EVERY LAYER THE WORST LAYER'S
    //! RESIDUAL. Measured over BERT-Base's twelve, the residual reaches
    //! 1001.5 where layer 0's is 62.5; a single ride sized on the first put
    //! layer 0 at rms 2^-6.73 against the 2^-10.68 it reaches on its own,
    //! which is exactly the four bits a 16x colder crossing predicts. The
    //! factor is free either way -- it rides the norms' gain and bias, which
    //! are plaintexts -- so there is no reason to share it.
    double stream_out = 0.0;
    //! The factors the six projections' weights carry (see the header for
    //! what each has to be).
    double q_scale = 1.0, k_scale = 1.0, v_scale = 1.0;
    double o_scale = 1.0, int_scale = 1.0, out_scale = 1.0;
    //! THE FEED-FORWARD'S RESIDUAL ROWS, suppressed by a public per-token
    //! factor. One entry per token, 1.0 for none, empty for no suppression.
    //!
    //! Measured on the real checkpoint, layer 10's `H + FFN(H)` reaches
    //! 1001.5 at token 48 channel 180 -- thirteen slots past 100 in three
    //! columns and five rows -- against a row-maximum median of 9.5. The
    //! crossing is PEAK limited, so those five rows cost every other row a
    //! hundredfold ride: layers 9 and 10 came back at rms 2^-7.08 and 2^-6.44
    //! where their neighbours sit at 2^-8.5.
    //!
    //! It closes inside the layer. `H` carries the factor (the
    //! post-attention norm's gain and bias are plaintexts, so it is free
    //! there), the feed-forward's output carries it too (folded into GELU's
    //! own masks and its output projection's bias), the GELU's argument has
    //! it divided out at the crossing it is already being scaled by, and the
    //! output LayerNorm -- exactly scale invariant per token -- removes it
    //! from the answer. Nothing downstream needs to know.
    std::vector<double> row_suppress;
    //! GELU's plan: the fitted half-interval, its degree, and the per-slot
    //! group assignment (0 bulk, 1 saturated positive, 2 saturated negative)
    //! indexed `[token * hidden_live + channel]`. `GeLu.h` carries the
    //! measurements; `reference_forward_bert.py` writes the plan.
    double gelu_range = 8.0;
    int gelu_degree = 31;
    //! The WIDE fitted group, for the slots a corpus says reach past
    //! `gelu_range` with no stable sign -- 30 of 393,216 at layers 9 and 10,
    //! none elsewhere. Zero for none; group 3 in `gelu_group`.
    double gelu_wide_range = 0.0;
    int gelu_wide_degree = 63;
    std::vector<unsigned char> gelu_group;
    //! One band of the CERTIFIED plan (`reference/scripts/bert_plan.py`).
    struct GeLuBand {
      double range = 0.0;
      int degree = 63;
      std::vector<double> coeffs;
    };
    //! The certified band plan. When it is non-empty it REPLACES the four
    //! groups above: every band is a `kFit` -- there is no identity and no
    //! zero group, so no slot is ever answered by `u` or by nothing -- and
    //! the band a channel is in comes from the WEIGHTS, not from a prompt.
    //!
    //! WHY THAT IS SAFE WITHOUT A CORPUS. A LayerNorm output lies exactly on
    //! the sphere of radius `sqrt(768)`, so with `a_j = gain * W[:,j]` and
    //! `c_j = bias . W[:,j] + b_j`,
    //!
    //!     |u_j - c_j| <= sqrt(768) ||a_j - mean(a_j)||
    //!
    //! for EVERY input -- another prompt, an adversarial one, a random
    //! ciphertext. A band's range is that ceiling maximised over its
    //! channels, so no slot can leave its own interval and the unbounded
    //! failure mode (a Chebyshev evaluated outside, `GeLu.h`) is gone rather
    //! than made unlikely. The corpus is read for the fit's WEIGHTING only,
    //! which can cost accuracy and cannot cost safety. Measured over 28
    //! prompts, 17 of them hostile: the ceiling is 1.5x-3.5x the corpus
    //! maximum per layer and the corpus reaches 0.95 of it at layers 3-7,
    //! so it is not a loose bound (`reference/docs/BERT_BASE_B1.md` 11.1).
    std::vector<GeLuBand> gelu_bands;
  };

  CiBertLayer(std::shared_ptr<const BootContext<word>> boot,
              const CiSwitchedCcmmLayout &layout,
              std::vector<const EvaluationKey<word> *> modpack_keys,
              const Config &cfg);

  CiBertLayer(const CiBertLayer &) = delete;
  CiBertLayer &operator=(const CiBertLayer &) = delete;

  //! The CI line's own half: the projection leg, the seam and the module
  //! basis, which this class holds rather than reimplements.
  CiLlamaLayer<word> &Base() { return base_; }
  const CiLlamaLayer<word> &Base() const { return base_; }

  void AddRequiredRotations(EvkRequest &req) const;

  //! Where the layer's stream lives: coefficients at this level, which is
  //! what both norms' `ToCoeff` leaves and what the projections read.
  int GetStreamLevel() const { return op_level_; }
  int GetNumModelCts() const { return num_model_cts_; }
  int GetNumHiddenCts() const { return num_hidden_cts_; }
  //! Images a Q, K or V tensor occupies: `head_dim / rank` of them.
  int GetNumImages() const { return num_images_; }
  //! The two crossing constants, DERIVED (Doing.md 1.5dk): `ToSlot`
  //! multiplies the message by `GetCrossing()` and `ToCoeff` by
  //! `GetKappa()`. Two of the calibration's scalings are stated in terms of
  //! them; see the class comment's table.
  double GetCrossing() const { return crossing_; }
  double GetKappa() const { return kappa_; }
  //! Where the norms' StC is compiled, which is what their degree has to
  //! land on.
  int GetStCLevel() const { return sched_.GetStCLevel(); }

  /**
   * @brief Q, K and V from the stream, with their biases: `GetNumImages()`
   * coefficient images per tensor, at the level a `HalfBoot` takes.
   *
   * @param qkv receives three vectors, in the order q, k, v
   * @param stream the residual stream in coefficients at `GetStreamLevel()`
   * @param qkv_declared the declared output width the caller's maps state
   */
  void Emit(std::vector<std::vector<Ct>> &qkv, const std::vector<Ct> &stream,
            int qkv_declared, const Weights &w, const Calibration &c);

  /**
   * @brief The O projection with its bias, the residual, and the
   * post-attention LayerNorm.
   *
   * @param res the feed-forward's input, in coefficients at
   *        `GetStreamLevel()`
   * @param seamed the seam's images, `attn_declared / proj_rank` of them
   * @param stream the layer's own input, at `GetStreamLevel()`
   */
  void AttentionTurn(std::vector<Ct> &res, const std::vector<Ct> &seamed,
                     const std::vector<Ct> &stream, const Weights &w,
                     const Calibration &c, const EvkMap<word> &evk);

  /**
   * @brief The whole feed-forward, its residual, and the output LayerNorm:
   * the next layer's stream, in coefficients at `GetStreamLevel()`.
   */
  void FeedForward(std::vector<Ct> &res, const std::vector<Ct> &h,
                   const Weights &w, const Calibration &c,
                   const EvkMap<word> &evk);

  //! The norms' slot output, kept when `keep_norm_slots` is on; diagnostic.
  const std::vector<Ct> &GetNormSlots() const { return norm_slots_; }
  void KeepNormSlots(bool on) { keep_norm_slots_ = on; }

  //! The invsqrt this layer would evaluate, in the clear -- the plaintext
  //! oracle that separates a wrong window from a wrong circuit.
  double PlainNormInvSqrt(bool ffn, double alpha, double window,
                          double var) const;

 private:
  //! One projection's bias, added on its own coefficient images. With
  //! `per_token` non-empty the bias carries that factor as well, which is
  //! what a suppressed residual needs (see `Calibration::row_suppress`).
  void AddBias(std::vector<Ct> &imgs, const std::vector<double> &bias_declared,
               double scale,
               const std::vector<double> &per_token = {}) const;
  //! The crossing, the LayerNorm and the return to coefficients.
  void NormTurn(std::vector<Ct> &res, const std::vector<Ct> &stream,
                const std::vector<double> &gain,
                const std::vector<double> &bias, double alpha, double window,
                int degree, const std::vector<double> &invsqrt,
                const std::vector<double> &token_scale, double in_scale,
                double out_scale, const std::vector<double> &row_suppress,
                const EvkMap<word> &evk);
  //! `x *= factor`, on the crossing's own constant multiply.
  void Canonicalise(Ct &ct, double factor) const;
  void Canonicalise(Ct &ct, const Pt &pt) const;
  //! The per-token plaintext the crossing's multiply carries.
  Pt TokenPlaintext(double factor, const std::vector<double> &per_token,
                    double scale) const;
  int NormDegree(double window) const;

  std::shared_ptr<const BootContext<word>> boot_;
  Config cfg_;
  CiLlamaLayer<word> base_;
  SylphSchedule<word> sched_;
  int num_slots_ = 0;
  int num_model_cts_ = 0;
  int num_hidden_cts_ = 0;
  int num_images_ = 0;
  int slot_level_ = 0;
  int op_level_ = 0;
  double crossing_ = 1.0;
  double kappa_ = 1.0;
  bool keep_norm_slots_ = false;
  std::vector<Ct> norm_slots_;
  //! The GELU crossing's per-token plaintext, built once per feed-forward.
  Pt gelu_pt_;
};

}  // namespace cheddar
