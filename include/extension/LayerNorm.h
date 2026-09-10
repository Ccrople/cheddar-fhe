#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/EvalPoly.h"

namespace cheddar {

/**
 * @brief BERT's LayerNorm on the same packing `RmsNormHandler` works in.
 *
 *     y[t][c] = (x[t][c] - mu[t]) / sqrt(var[t] + eps) * g[c] + b[c]
 *     mu[t]   = mean_c x[t][c],   var[t] = mean_c (x[t][c] - mu[t])^2
 *
 * ## What this is, against `RmsNormHandler`
 *
 * RMSNorm is this operator with `mu` forced to zero and no bias, so the
 * second half here -- the inverse square root on a layer-constant-scaled
 * argument, its Chebyshev window, and the weight multiply -- is that class's
 * circuit unchanged, down to the affine map being folded into one constant
 * multiply and one constant add. Everything in `RmsNorm.h`'s header about the
 * window, the layer constant and `channel_stride` holds here verbatim and is
 * not repeated.
 *
 * The two additions are the CENTRING and the BIAS.
 *
 * ### The centring costs one level, and that is a choice
 *
 * `mu` is the same rotate-and-add tree as the sum of squares, minus the
 * square: free in levels. Dividing it by the channel count is not -- Cheddar's
 * `Mult(Ct, Const)` does not rescale, so `1/H` is a multiply and a rescale,
 * and the subtraction that follows must meet `x` at the level that leaves.
 * So this operator is RMSNorm plus exactly one level.
 *
 * It could be avoided. Writing the answer as
 *
 *     y_i = (x_i r) g_i - (S r) (g_i / H)
 *
 * with `S` the channel SUM and `r` the inverse square root puts `1/H` inside
 * a plaintext, where it is free, at the cost of one extra ciphertext multiply
 * and a second weight plaintext per ciphertext -- but then the variance has
 * to be computed as `mean(x^2) - mu^2`, a difference of two quantities that
 * are equal to within the variance itself. On a LayerNorm output re-entering
 * the next layer that difference is most of the answer, and losing it is a
 * silent accuracy loss of exactly the kind this tree keeps finding. The level
 * is the honest price; the algebra above is written down here so the trade is
 * on the record rather than rediscovered.
 *
 * ### THE MEAN IS MASKED, and that is a contract, not an optimisation
 *
 * The declared width is not the live one: BERT's 768 channels sit in a
 * declared 1024 (two rank-512 ciphertexts), and the reduction sums over
 * every slot. Zeros contribute nothing to a SUM, so `mu` is right if it is
 * divided by the LIVE count -- but `x - mu` is then `-mu` at the dead slots,
 * and the sum of squares that follows collects `(declared - live) * mu^2`
 * that does not exist. The variance would be wrong by exactly that, silently,
 * and the wrongness grows with `mu`.
 *
 * So the `1/live` division is carried by a PLAINTEXT that is `1/live` where
 * the image has data and zero where it does not, instead of by a constant.
 * It is the same multiply and the same single level, and it leaves the
 * centred image exactly zero off the data -- which is also what the banded
 * half-density convention needs, where "has data" includes the duplicate
 * slots (they carry the partner position's value and must be centred by the
 * partner's mean, exactly as `channel_stride = 2` normalises them by the
 * partner's variance).
 *
 * ### The bias is free, and it is not scale invariant
 *
 * `+ b` is a plaintext add at the output level: no level, no key. But unlike
 * everything else in this operator it is NOT scale invariant -- LayerNorm's
 * output is `(x - mu)/sigma * g + b` and a stream carrying a factor `s` needs
 * `s * b`, not `b`. The caller passes the bias already in the stream's units,
 * the same way it passes the gain with `sqrt(alpha)` folded in.
 *
 * ## What the argument of the inverse square root is
 *
 * The VARIANCE, not the mean square: `u = alpha * (var + eps)`. So the
 * caller's `layer_constant` and window come from the clear model's variance
 * spread (`reference_forward_bert.py` writes both). BERT's epsilon is 1e-12
 * against a variance of order 1, which is negligible -- it rides the additive
 * half of the affine map for free anyway.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class LayerNormHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  ConstContextPtr<word> context_;
  int num_tokens_;
  int num_channels_;
  int num_slots_;
  int num_ct_;
  int live_ = 0;
  double layer_constant_;
  double eps_;
  double window_lo_ = 0.0;
  double window_hi_ = 0.0;
  double affine_scale_ = 0.0;
  int input_level_;
  int centre_level_ = -1;
  std::vector<int> rotation_distances_;

  mutable std::vector<std::vector<Complex>> cached_weight_, cached_bias_,
      cached_mask_;
  mutable std::vector<Pt> weight_pt_, bias_pt_, mask_pt_;
  mutable int cached_weight_level_ = -1;
  mutable int cached_bias_level_ = -1;
  mutable int cached_mask_level_ = -1;
  int weight_level_ = -1;
  int bias_level_ = -1;
  int mode_ = 0;  //!< `Mode`, kept as an int so the enum can be declared below
  std::unique_ptr<EvalPoly<word>> inv_sqrt_;

 public:
  /**
   * @brief Which part of the operator this handler is.
   *
   * ## Why the inverse square root is worth splitting
   *
   * Its cost is `degree ~ ln(1/delta) sqrt(R) / 2` in the window ratio `R`,
   * and `R` is the only lever (`BERT_BASE_B1.md` 11.3). Without a per-prompt
   * per-token rescale the feed-forward norms of layers 9 and 10 face
   * `R ~ 10000`, which wants degree 511 -- fourteen levels against the seven
   * or eight a span has, and no landing that affords it.
   *
   * 11.3 rejects a crude-then-refine split on the arithmetic `6 + 2 + 5 = 13`,
   * which adds both stages inside ONE span. They do not have to be in one.
   * The norms already sit between crossings, and a crossing between the
   * stages buys the second one a fresh span for the price of a bootstrap --
   * which is what `SoftMaxCho` does on the batched branch for the same shape
   * of problem.
   *
   * ## Why the second stage needs no margin
   *
   * With `r0 = (1 + e) / sqrt(var)` and `y0 = (x - mu) r0`,
   *
   *     u = mean(y0^2) = var r0^2 = (1 + e)^2   IDENTICALLY,
   *
   * the variance cancelling. So the refine stage's argument lies in
   * `[(1-eps)^2, (1+eps)^2]` where `eps` is the crude stage's worst RELATIVE
   * error over its own interval -- a sup over the interval, not a statistic
   * over prompts. The second window is a THEOREM given the first, and the
   * whole statistical assumption of the operator stays where it already was.
   * `eps < 1` is the only hard condition: at one the next window reaches zero
   * and the singularity is back inside it.
   *
   * And `y0 / sqrt(u) = (x - mu) / sqrt(var)` exactly, so the refine stage's
   * gain carries `sqrt(alpha)` on ITS window exactly as `kWhole`'s does on
   * the layer's -- the caller's gain convention does not move.
   *
   * Measured on the host over 35 English prompts, ten held out
   * (`reference/scripts/bert_sim.py`, design `st10`): one stage at the degree
   * a ten-level span affords leaves the twelve-layer chain at 3.5e-01; this
   * split leaves it at 2.9e-03, for two extra bootstraps a MODEL.
   */
  enum class Mode {
    kWhole,   //!< centre, invsqrt, gain, bias -- the operator in one span
    kCrude,   //!< centre, invsqrt, apply: hands back `y0`, no gain, no bias
    kRefine,  //!< NO centring (`y0` is already centred), then gain and bias
  };

  /**
   * @param context the Context this evaluates in
   * @param num_tokens T, a power of two
   * @param num_channels the DECLARED channel width the mean and the variance
   *        are stated against -- what the ciphertexts carry, dead slots
   *        included, exactly as `RmsNormHandler` takes it
   * @param layer_constant alpha_L, so that `alpha_L * var` lands in the
   *        window; the reciprocal of the geometric midpoint of the variance
   *        range is what `reference_forward_bert.py` writes
   * @param input_level level of the input ciphertexts
   * @param eps LayerNorm's epsilon (BERT: 1e-12)
   * @param window_ratio the ends' ratio of the invsqrt window
   * @param degree the Chebyshev degree for the inverse square root
   * @param channel_stride 1 on the module basis, 2 for a banded half-density
   *        image; see `RmsNorm.h`, the rule is identical and so is the reason
   * @param invsqrt_coeffs the Chebyshev coefficients for `1/sqrt` on the
   *        window, or empty to INTERPOLATE at `degree`. The same argument as
   *        `GeLuHandler::Group::coeffs`: on a WIDE window an interpolant
   *        spends its accuracy uniformly while the tokens are not uniform,
   *        and a data-weighted fit is what makes a window of 2419 -- which is
   *        what layer 9 has without a per-prompt per-token rescale -- cost
   *        2.0e-03 at degree 127 instead of 4.9e-01 at degree 15.
   * @param live_channels the channels the MODEL has, when the declared width
   *        is larger (BERT's 768 in a declared 1024). The mean and the
   *        variance divide by this, not by the declared width; 0 means they
   *        are the same.
   */
  LayerNormHandler(ConstContextPtr<word> context, int num_tokens,
                   int num_channels, double layer_constant, int input_level,
                   double eps = 1e-12, double window_ratio = 4.0,
                   int degree = 15, int channel_stride = 1,
                   int live_channels = 0,
                   const std::vector<double> &invsqrt_coeffs = {},
                   Mode mode = Mode::kWhole);

  LayerNormHandler(const LayerNormHandler &) = delete;
  LayerNormHandler &operator=(const LayerNormHandler &) = delete;

  /** @brief The rotation distances `Apply` needs keys for. */
  const std::vector<int> &GetRotationDistances() const {
    return rotation_distances_;
  }

  int GetNumCiphertexts() const { return num_ct_; }
  //! The level `Apply` leaves its output at.
  int GetOutputLevel() const { return bias_level_; }
  //! The level the centring leaves the stream at (input_level - 1).
  int GetCentreLevel() const { return centre_level_; }

  /**
   * @brief Encode the centring mask, the gain and the bias up front.
   *
   * @param mask one vector per input ciphertext, `1` where the image carries
   *        data and `0` elsewhere -- the header's contract. The `1/live`
   *        division is folded in here, so the caller states WHERE the data
   *        is and this class states what to divide by.
   */
  void PrepareMask(const std::vector<std::vector<Complex>> &mask) const;
  /** @brief Encode the gain and bias plaintexts up front. */
  void Prepare(const std::vector<std::vector<Complex>> &weight,
               const std::vector<std::vector<Complex>> &bias) const;

  /** @brief Device bytes the cached plaintexts hold. */
  size_t GetPlaintextBytes() const;

  /** @brief The compiled inverse square root in the clear, on `u`. */
  double PlainInvSqrt(double u) const;

  /**
   * @brief The channel SUM, broadcast to every slot, at the input level.
   *
   * Exposed for the same reason `RmsNormHandler::SumOfSquares` is: the halves
   * of this operator have never been measured apart, and a read here says
   * whether a wrong answer was made by the reduction or by the fit. Costs no
   * level -- it is rotations and adds.
   */
  void ChannelSum(Ct &acc, const std::vector<Ct> &x,
                  const EvkMap<word> &evk) const;

  /**
   * @brief `xc = x - mu`, at `GetCentreLevel()`.
   *
   * One plaintext multiply per ciphertext, because the mask is per
   * ciphertext: `mu` is one broadcast value but WHERE it may be subtracted
   * is not (the header's contract). The `1/live` division rides that same
   * multiply.
   */
  void Centre(std::vector<Ct> &xc, const std::vector<Ct> &x,
              const std::vector<std::vector<Complex>> &mask,
              const EvkMap<word> &evk) const;

  /** @brief The centred image's channel sum of squares, broadcast. */
  void SumOfSquares(Ct &acc, const std::vector<Ct> &xc,
                    const EvkMap<word> &evk) const;

  void Apply(std::vector<Ct> &res, const std::vector<Ct> &x,
             const std::vector<std::vector<Complex>> &weight,
             const std::vector<std::vector<Complex>> &bias,
             const std::vector<std::vector<Complex>> &mask,
             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
