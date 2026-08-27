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
 * The affine map onto the polynomial's domain is *not* among those levels. Its
 * multiplicative half is applied by reinterpreting the ciphertext's scale,
 * which is free, and its additive half is a constant addition, which is also
 * free. Doing it the obvious way would have cost a level, and in Cheddar a
 * further one: Mult(Ct, Const) does not rescale, so the scale would have been
 * left at D^2 and needed a Rescale to recover.
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
  double eps_;
  double window_lo_ = 0.0;
  double window_hi_ = 0.0;
  double affine_scale_ = 0.0;
  int input_level_;
  std::vector<int> rotation_distances_;

  // One full-width Encode per input ciphertext, every call, was most of
  // RMSNorm's measured 288 ms for eight ciphertexts. The weights are fixed for
  // a layer and the level the polynomial leaves is deterministic, so the
  // plaintexts are cached and rebuilt only when the weights or that level
  // actually change -- which keeps Apply's signature and stays correct if a
  // caller does change them.
  mutable std::vector<std::vector<Complex>> cached_weight_;
  mutable std::vector<Pt> weight_pt_;
  mutable int cached_weight_level_ = -1;
  int weight_level_ = -1;
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
   * @param eps Llama's RMSNorm epsilon. It is not negligible here: the
   * measured mean square is around 5e-4 against an eps of 1e-5, a two percent
   * effect on the norm. It costs nothing, because alpha_L * eps folds into the
   * additive half of the affine map and constant addition is level-free.
   * @param window_ratio the width of the approximation window, as the ratio
   * between its ends. [SYLPH] uses 30, which is what its calibrated data spans
   * across layers; a caller whose measured spread is narrower should say so,
   * because the degree needed falls quickly with it and every two degrees is a
   * level. Measured on this bundle's user tokens the spread is 4.18.
   * @param degree Chebyshev degree for the inverse square root. Measured
   * against a 12-bit target: ratio 30 needs 23 (5 levels), ratio 6 needs 9
   * (4 levels), ratio 4.18 needs 7 (3 levels). Degree 7 clears 12 bits only up
   * to ratio 4.18 and falls to 11.9 bits at 5, so depth 7 is reachable but has
   * no margin.
   */
  /**
   * @param channel_stride how many channel slots the reduction steps over.
   *
   * 1 is every channel and is the ordinary ring's case. **2 is what the
   * conjugate-invariant coefficient leg needs**, and it is a contract rather
   * than an optimisation. On R+ a projection's output is the banded
   * recomposition of its components (Doing.md 1.5ba), and a half-density
   * emission -- the only kind that is readable at all (1.5by) -- carries its
   * live values at their own addresses and SHIFTED DUPLICATES at the others.
   * That image is simultaneously clean for a slot operator and correct for
   * the next `ModDecomp`, and it stops being either the moment the duplicates
   * are destroyed.
   *
   * So the duplicates have to survive this operator, and a full-stride
   * rotate-and-add would fold them into the live sum. Stepping by two keeps
   * the parities apart: the live slots sum the live channels and the duplicate
   * slots sum the duplicates -- which are the live values of the partner
   * position, so they land the SAME inverse square root the partner needs.
   * Give the weight plaintext the partner's weight at those slots and the
   * output is again a valid banded image, at no extra level and one rotation
   * fewer.
   *
   * ### COMPONENT ZERO MUST BE EMPTY, and that is a contract on the CALLER
   *
   * The two bands do not sum the same components, and the gap is exactly one.
   * The even slots hold components `I = 0 .. rank/2-1` at position P. The odd
   * slots hold `comp_{rank-I}[P+1]` for `I = rank/2 .. rank-1`, i.e.
   * components `J = 1 .. rank/2`. So `J = rank/2` is dead and contributes
   * nothing, and `J = 0` is MISSING -- `rank - 0` wraps to 0 and the banded
   * recomposition excludes `i == 0`, so component zero has no duplicate
   * anywhere in the image.
   *
   * It is not a choice of packing. `I -> rank - I` has exactly two fixed
   * points on `[0, rank)`, namely 0 and `rank/2`, and a live set `S` needs
   * `S` restricted to `[0, rank/2)` to equal `S` restricted to `(0, rank/2]`
   * for the bands to agree -- which forces both 0 and `rank/2` OUT of `S`.
   * **A half-density ciphertext under this
   * operator therefore carries at most `rank/2 - 1` live channels, not
   * `rank/2`.**
   *
   * Carrying data in component zero does not corrupt component zero. It
   * corrupts EVERY duplicate, because they are all normalised by a mean
   * square that is short one channel, and then the next projection's
   * `ModDecomp` -- whose suffix recursion is
   * `comp_i[P] = coeff[P][i] - comp_{rank-i}[P+1]` -- hands the whole of it to
   * the LIVE components. Measured on `ci16_35` at rank 512 by
   * `CiFfn.TheFeedForwardNetworkRunsOnTheRealSubring`: the duplicate band sits
   * at 9.02e-03 against the live band's 7.23e-04, the following projection
   * lands at 2.35e-02 instead of ~2e-04, and the FFN closes at 2^-4.84.
   * Leaving component zero empty puts both bands at 4.0e-04 and the FFN at
   * 2^-7.72 -- **3.4 bits for one channel in 256.**
   *
   * The emptied component simply contributes zero to both bands, so neither
   * the mean nor the caller's `layer_constant` needs adjusting.
   *
   * `num_channels` stays the DECLARED width either way: it is what the
   * ciphertext count and the mean are stated against, half of it is zero on
   * R+, and a caller's `layer_constant` is calibrated on that mean.
   */
  RmsNormHandler(ConstContextPtr<word> context, int num_tokens,
                 int num_channels, double layer_constant, int input_level,
                 double eps = 1e-5, double window_ratio = 30.0,
                 int degree = 23, int channel_stride = 1);

  // disable copying (or moving also)
  RmsNormHandler(const RmsNormHandler &) = delete;
  RmsNormHandler &operator=(const RmsNormHandler &) = delete;

  /** @brief The rotation distances Apply needs keys for. */
  const std::vector<int> &GetRotationDistances() const {
    return rotation_distances_;
  }

  int GetNumCiphertexts() const { return num_ct_; }

  /**
   * @brief Encode the weight plaintexts up front, so their cost sits in setup.
   *
   * [SYLPH] section 5.1 keeps the model's plaintexts resident on the GPU for
   * the whole run and section 5.3 makes that conversion its own stage, at a
   * measured 4x expansion over the raw weights (table 5: 13.0 GiB to 52.0).
   * Apply does this lazily on first call, which is correct but puts one
   * ~50 ms host encode per ciphertext inside what reads as an online
   * measurement -- eight of them were 264 of RMSNorm's 271 ms.
   */
  void Prepare(const std::vector<std::vector<Complex>> &weight) const;

  /** @brief Device bytes the cached weights hold, 0 before Prepare. */
  size_t GetPlaintextBytes() const;

  /**
   * @brief Evaluate the compiled inverse-square-root polynomial in the clear,
   * on an argument u in the window.
   *
   * This is the plaintext oracle for the polynomial alone. Without it, a wrong
   * Chebyshev convention and a wrong circuit look identical from the outside:
   * both give a decrypted answer that is off by some amount.
   */
  double PlainInvSqrt(double u) const;

  /**
   * @brief Evaluate RMSNorm.
   *
   * @param res output, resized to x.size()
   * @param x input, T x H over `num_ct` ciphertexts, token index fastest
   * @param weight the per-channel weights with sqrt(alpha_L) folded in, as
   * slot vectors rather than plaintexts: the level they must be encoded at is
   * whatever the polynomial happens to leave, which is an internal detail of
   * this circuit and not something a caller should have to track.
   * @param evk_map supplies the rotation and multiplication keys
   */
  void Apply(std::vector<Ct> &res, const std::vector<Ct> &x,
             const std::vector<std::vector<Complex>> &weight,
             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
