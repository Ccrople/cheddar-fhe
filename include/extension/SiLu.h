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
 * ## The scaling factor has to be at least 2^35
 *
 * Measured on bootparam_30/35/40, the error this circuit adds on top of its own
 * polynomial is a fixed integer magnitude divided by the scaling factor, so it
 * falls by very nearly five bits for every five bits of scale:
 *
 *     scale   circuit vs its own polynomial   end to end vs true SiLU
 *     2^30    2.74e-03   12.10 bits           11.79 bits   under the bar
 *     2^35    1.01e-04   16.86 bits           13.47 bits   fit limited
 *     2^40    2.89e-06   21.99 bits           13.54 bits   fit limited
 *
 * At 2^30 it is the circuit and not the approximation that misses [SYLPH]'s
 * 12-bit bar, so raising the degree cannot recover it -- the fit is already
 * 13.54 bits there. From 2^35 the circuit stops mattering and the fit alone
 * sets the accuracy, which is why 2^40 gains nothing while costing three levels
 * (max level 19 against 16). **2^35 is the preset this operator wants.**
 *
 * ## Guessing HIGH is not the safe side
 *
 * A Chebyshev fit's error is uniform over its interval, so a range wider than
 * the data uses throws away exactly that ratio and says so nowhere -- the same
 * silence as guessing low, without the crash. Measured by
 * `CiFfn.TheFitsAloneExplainTheFfnError` on a gate reaching 2.57: the compiled
 * degree-31 fit is **2^-11.2** relative at range +-12 and **2^-33** at +-3.08,
 * and the whole FFN's fit floor moves from 2^-8.95 to 2^-13.15 with it.
 * [SYLPH] 3.1.3's own +-12 goes with a CALIBRATED input of 10.82, a margin of
 * 1.109; carrying the 12 without the calibration keeps its cost and drops its
 * benefit. A caller that has measured its gate should pass a small multiple of
 * that maximum -- and may then also drop the degree, since the table above is
 * a 12-bit target and a matched range clears it with far less.
 *
 * ## What this bundle cannot tell us
 *
 * The true input is RMSNorm_ffn(h) @ W_gate with h the post-attention hidden
 * state, which the layer-2 bundle does not contain. Substituting the layer
 * input gives |gate| ~ 0.67 with a dimension-wise spread of only 1.7x -- far
 * milder than [SYLPH]'s 10.82, because those numbers are maxima over all 32
 * layers and layer 2 is early. Measured in the test on 64 tokens x 512 dims,
 * |g| reaches 1.23 with an rms of 0.16. So the interval has to come from per-layer
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
