#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/EvalPoly.h"

namespace cheddar {

/**
 * @brief SoftMax as [SYLPH] section 2.3 specifies it, following Cho et al.
 *
 * ## The algorithm, and the two things that are easy to get wrong
 *
 *     x  -> translate to [-M, 0], scale to [-M/2^k, 0]
 *     y0 = exp(x')
 *     y_j = (y_{j-1} / ||y_{j-1}||_2)^2   for 1 <= j <= k
 *     return y_k
 *
 * **The norm is Euclidean, not the sum.** That is the whole design. With
 * `z = y / ||y||_2` we get `sum(z^2) = sum(y^2) / ||y||_2^2 = 1`, so the
 * squaring leaves a vector that already sums to one. Hence exactly k
 * iterations and **no final normalisation** -- reading the norm as a sum
 * instead produces a different and much more expensive algorithm.
 *
 * **The denominator is an inverse square root**, the same function RMSNorm
 * evaluates, not a reciprocal.
 *
 * ## Only the last inverse square root has to be accurate
 *
 * [SYLPH] section 2.3 states this; measured on the real layer-2 scores it is
 * not merely approximate but exact. An error in an early inverse square root
 * multiplies that row by a constant c; the squaring makes it c^2; the next
 * normalisation divides by `||c^2 y^2||_2` and removes it completely.
 * Perturbing every early inverse square root by 50% moved the output by
 * nothing at all (49.9 bits, against 49.9 unperturbed). So the early ones exist
 * for range control, not for accuracy, and may be very low degree.
 *
 * ## Choosing k
 *
 * k does not trade accuracy -- the iteration is exact in the clear for every k.
 * It trades the exp interval against the number of normalisations. Measured on
 * the real layer-2 attention scores, whose worst per-row span is 20.6 so M = 21:
 *
 *     k   exp on [-M/2^k, 0]   the accurate 1/sqrt        main track
 *     1   degree 9             [1.54, 6.82]   degree  8   7 levels
 *     2   degree 7             [0.0136, 0.50] degree 28   8 levels
 *     3   degree 5             [0.0088, 0.50] degree 28   10 levels
 *
 * [SYLPH] section 4.3 reports two iterations and 8 main-track levels, which the
 * k=2 row reproduces exactly. It picks k=2 because its calibrated SoftMax input
 * is 32.78 (table 2); at our M = 21 **k=1 is strictly better** -- one level
 * cheaper in the main track and degree 8 instead of 28 in the auxiliary track.
 * As section 4.3 puts it, the iteration count follows from calibrated estimates
 * of the inverse-square-root range, and ours are not [SYLPH]'s.
 *
 * ## What is not here
 *
 * [SYLPH] figure 2 bootstraps the auxiliary track separately, which is what
 * keeps its depth off the main track; that needs level-targeted bootstrapping,
 * which Cheddar does not have. Here both tracks run in one parameter set, so
 * `GetDepth()` reports the honest total and it is larger than 8. The main-track
 * count above is what this becomes once that bootstrap exists.
 *
 * Section 3.4's slim polynomial evaluation is an auxiliary-track optimisation.
 * By its own theorem 1 it does not reduce levels (still k+1 for degree 2^k) --
 * it reduces multiplications to `O(2^((k-j)/2)) + j` and key-switchings. So it
 * is a later speedup, not a prerequisite.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SoftMaxHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  ConstContextPtr<word> context_;
  int num_keys_;
  int num_slots_;
  double range_;
  int input_level_;
  int num_iters_;

  // norm^2 interval entering each inverse square root, from calibration
  std::vector<double> norm_lo_, norm_hi_;

  std::unique_ptr<EvalPoly<word>> exp_poly_;
  std::vector<std::unique_ptr<EvalPoly<word>>> inv_sqrt_;
  std::vector<int> rotation_distances_;

  std::unique_ptr<EvalPoly<word>> MakeInvSqrt(double lo, double hi, int degree,
                                              int level);

 public:
  /**
   * @param context the evaluation context
   * @param num_keys d, the number of entries in one SoftMax row; must be a
   * power of two, and the row must occupy d consecutive slots
   * @param range M, the half-open input span after translation: the argument
   * is taken to lie in [-M, 0]
   * @param input_level level of the input ciphertext
   * @param num_iters k
   * @param norm_lo per-iteration lower bound of `||y||_2^2`, from calibration
   * @param norm_hi per-iteration upper bound of `||y||_2^2`, from calibration
   * @param exp_degree degree of the exponential on [-M/2^k, 0]
   * @param inv_sqrt_degree degree of the *last* inverse square root; the
   * earlier ones use `early_inv_sqrt_degree`, which can be far smaller
   * @param early_inv_sqrt_degree degree of every inverse square root but the
   * last, which only has to keep the magnitudes in range
   */
  SoftMaxHandler(ConstContextPtr<word> context, int num_keys, double range,
                 int input_level, int num_iters,
                 const std::vector<double> &norm_lo,
                 const std::vector<double> &norm_hi, int exp_degree,
                 int inv_sqrt_degree, int early_inv_sqrt_degree = 4);

  // disable copying (or moving also)
  SoftMaxHandler(const SoftMaxHandler &) = delete;
  SoftMaxHandler &operator=(const SoftMaxHandler &) = delete;

  /** @brief Rotation distances the row reduction needs. */
  const std::vector<int> &GetRotationDistances() const {
    return rotation_distances_;
  }

  /**
   * @brief The same circuit in the clear, for separating fit from evaluation.
   *
   * Takes the already-translated argument in [-M, 0] and returns the SoftMax
   * over that row, using the same polynomials the encrypted path uses.
   */
  std::vector<double> PlainSoftMax(const std::vector<double> &x) const;

  /**
   * @brief res = SoftMax over each block of `num_keys` slots.
   *
   * @param res output
   * @param x_scaled the argument mapped onto [-1, 1], that is
   * `2 (x - c) / M + 1` with c the calibrated per-layer maximum. Both halves
   * fold upstream for free: the `2/M` into the QK product's `1/sqrt(d)`, the
   * shift into a constant addition. Doing the scaling here instead would hand
   * EvalPoly a non-canonical input scale, which is what silently broke RMSNorm.
   * @param causal_mask 1 on valid (key <= query) positions and 0 elsewhere,
   * in the same slot layout as the input. Causality is public, so this is a
   * plaintext.
   * @param evk_map supplies the multiplication and rotation keys
   */
  void Apply(Ct &res, const Ct &x_scaled,
             const std::vector<Complex> &causal_mask,
             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
