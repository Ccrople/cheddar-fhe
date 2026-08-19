#pragma once

#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/EvalPoly.h"

namespace cheddar {

/**
 * @brief Llama RMSNorm, following [SYLPH] section 3.1 and table 4.
 *
 *     y[t][c] = x[t][c] / sqrt( mean_c(x[t][c]^2) + eps ) * w[c]
 *
 * ## The layer constant, and why it is not optional
 *
 * The circuit never evaluates `1/sqrt` on the raw mean square. [SYLPH] scales
 * by a **layer-wise constant** first so the argument lands in a narrow window,
 * and only then approximates. The window is `[1/sqrt(30), sqrt(30)]` -- a
 * spread of 30x, from the companion Llama-2 paper's table 4, whose caption says
 * outright that this is "the global range after multiplication by a layerwise
 * constant".
 *
 * Thirty is a tight budget, and measuring the real model shows why it is
 * achievable at all. On the Llama-3-8B layer-2 residual stream, the per-token
 * mean square spans **285,946x** -- five orders of magnitude past the window.
 * Almost all of it comes from three tokens:
 *
 *     t0 = 33.12,  t1 = 33.12,  t3 = 32.64,   every other token ~ 0.0005
 *     prompt: 128000 128000 382 128000 ...     i.e. BOS, BOS, _, BOS
 *
 * Those are the attention-sink tokens. [SYLPH] section 3.1.1 does not normalise
 * them -- it **removes them from the encrypted path entirely**, precomputing
 * their Key-Value states offline and injecting them as a static cache, which is
 * sound because the sink prefix is independent of the user's input. Excluding
 * them leaves the remaining 125 tokens spanning **4.87x**, which sits inside the
 * 30x window with room to spare.
 *
 * So the sink exclusion is not a detail of the attention stage that RMSNorm can
 * ignore. It is the precondition that makes a single layer constant work, and a
 * caller that feeds sink tokens through here will silently drive the polynomial
 * far outside its approximation interval.
 *
 * ## Depth
 *
 * One level for the square, `ceil(log2(d+1))` for the polynomial, and two to
 * apply the result and the weight. At degree 23 -- the smallest Chebyshev fit
 * on this window reaching [SYLPH] section 3.1.2's 12-bit precision target,
 * measured at 13.8 bits -- that is 1 + 5 + 2 = 8, against the 7 recorded in
 * Doing.md section 2 for a hand-scheduled version that fuses the weight
 * multiplication into a neighbour.
 *
 * ## Packing
 *
 * [SYLPH] table 4 puts the non-linear operators at ring degree 65536 in slot
 * encoding, with the token index varying fastest (section 3.2). Summing over
 * channels is therefore a rotate-and-add over whole token blocks, and because
 * the same reduction broadcasts the result back to every block it costs
 * `log2(slots / tokens)` rotations and no extra work to redistribute.
 *
 * One deliberate divergence: [SYLPH] uses the conjugate-invariant CKKS variant
 * for N real slots, while Cheddar's encoder gives N/2 complex ones. Using only
 * the real parts, a 128 x 4096 tensor occupies 16 ciphertexts here rather than
 * the 8 the paper reports.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class RmsNormHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  ConstContextPtr<word> context_;
  int num_tokens_;
  int num_channels_;
  int num_slots_;
  int num_ct_;
  double layer_constant_;
  int input_level_;
  std::vector<int> rotation_distances_;
  std::unique_ptr<EvalPoly<word>> inv_sqrt_;

 public:
  /**
   * @param context the degree-65536 Context
   * @param num_tokens T, the number of **user** tokens; sink tokens must
   * already have been excluded (see above)
   * @param num_channels H
   * @param layer_constant alpha_L, chosen so that alpha_L * mean(x^2) lands in
   * [1/sqrt(30), sqrt(30)]. The reciprocal of the geometric mean of the
   * per-token mean square is the natural choice and is what the accompanying
   * test uses.
   * @param input_level level of the input ciphertexts
   * @param degree Chebyshev degree for the inverse square root; 23 reaches the
   * 12-bit target on this window
   */
  RmsNormHandler(ConstContextPtr<word> context, int num_tokens,
                 int num_channels, double layer_constant, int input_level,
                 int degree = 23);

  // disable copying (or moving also)
  RmsNormHandler(const RmsNormHandler &) = delete;
  RmsNormHandler &operator=(const RmsNormHandler &) = delete;

  /** @brief The rotation distances Apply needs keys for. */
  const std::vector<int> &GetRotationDistances() const {
    return rotation_distances_;
  }

  int GetNumCiphertexts() const { return num_ct_; }

  /**
   * @brief Evaluate RMSNorm.
   *
   * @param res output, resized to x.size()
   * @param x input, T x H over `num_ct` ciphertexts, token index fastest
   * @param weight the per-channel weights already encoded with sqrt(alpha_L)
   * folded in, one plaintext per input ciphertext
   * @param evk_map supplies the rotation and multiplication keys
   */
  void Apply(std::vector<Ct> &res, const std::vector<Ct> &x,
             const std::vector<Pt> &weight, const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
