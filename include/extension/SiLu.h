#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/EvalPoly.h"

namespace cheddar {

/**
 * @brief SiLU, x * sigmoid(x), as [SYLPH] evaluates it in the SwiGLU gate.
 *
 * ## Why the range is a parameter and not a constant
 *
 * SiLU is **not scale invariant**. RMSNorm tolerates any input magnitude
 * because RMSNorm(beta*x) = RMSNorm(x), so a badly scaled input can be fixed
 * for free; here the magnitude is part of the answer, and an input outside the
 * approximation interval is simply evaluated wrongly. This is the operator
 * where [SYLPH]'s calibration stops being an optimisation and becomes a
 * prerequisite.
 *
 * The degree follows the range steeply, and every doubling of the degree is a
 * level. Measured against a 12-bit target, error relative to the largest
 * |SiLU| on the interval:
 *
 *     range    degree   bits   levels
 *     +-8      19       12.1   5
 *     +-12     27       12.0   5
 *     +-16     39       13.3   6
 *     +-24     63       14.6   6
 *
 * [SYLPH] section 3.1.3 reports degree 31 after calibration against 83 before,
 * "gaining 2 levels". Degree 31 lands at 13.5 bits on +-12 and 11.0 on +-16,
 * which fits its table 2: the calibrated SiLU input is 10.82, so +-12 is the
 * interval with margin. The Llama-2 companion's table 4 lists +-16 and +-24 as
 * the layer-wise and dimension-wise choices across the whole model.
 *
 * ## What this bundle cannot tell us
 *
 * The true input is RMSNorm_ffn(h) @ W_gate with h the post-attention hidden
 * state, which the layer-2 bundle does not contain. Substituting the layer
 * input gives |gate| ~ 0.67 with a dimension-wise spread of only 1.7x -- far
 * milder than [SYLPH]'s 10.82, because those numbers are maxima over all 32
 * layers and layer 2 is early. So the interval has to come from per-layer
 * calibration; nothing here can supply it, and a caller that guesses low will
 * be wrong without any diagnostic.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SiLuHandler {
 private:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;

  ConstContextPtr<word> context_;
  double range_;
  int input_level_;
  std::unique_ptr<EvalPoly<word>> poly_;

 public:
  /**
   * @param context the evaluation context
   * @param range the approximation half-interval; inputs must lie in
   * [-range, range] and are otherwise evaluated wrongly and silently
   * @param input_level level of the input ciphertexts
   * @param degree Chebyshev degree; see the table above for what each range
   * costs at a 12-bit target
   */
  SiLuHandler(ConstContextPtr<word> context, double range, int input_level,
              int degree = 31);

  // disable copying (or moving also)
  SiLuHandler(const SiLuHandler &) = delete;
  SiLuHandler &operator=(const SiLuHandler &) = delete;

  double GetRange() const { return range_; }

  /** @brief The polynomial in the clear, for separating fit from circuit. */
  double PlainSiLu(double x) const;

  /**
   * @brief res = SiLU(x), given x / range.
   *
   * The input is taken pre-divided on purpose. Scaling it here by
   * reinterpreting the scale would be free but hands EvalPoly a non-canonical
   * input scale, which silently broke RMSNorm -- exact in the clear, wrong by
   * up to 29% encrypted. A constant multiply would be correct and cost a
   * level. Neither is needed: SiLU always follows the gate projection, and
   * 1 / range folds into that projection's plaintext weights for nothing.
   *
   * @param res output, SiLU(x) at the canonical scale
   * @param normalised_x the input divided by `range`
   * @param evk_map supplies the multiplication key
   */
  void Apply(Ct &res, const Ct &normalised_x,
             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
