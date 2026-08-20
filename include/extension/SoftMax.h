#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/BootContext.h"
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
 * ## The auxiliary track, and why it needs no new machinery
 *
 * [SYLPH] figure 2 bootstraps the auxiliary track separately -- that is what
 * keeps the norm square, the affine map and the inverse square root off the
 * main track. Fused into one track the total is 13 levels; separated, the main
 * track is **7**, one better than [SYLPH]'s 8 because k=1 needs one
 * normalisation rather than two.
 *
 * An ordinary `Boot()` suffices **only when the parameter set has no slack**,
 * and that turns out to be the binding constraint rather than the range. It is
 * worth being precise about why, because the two placements look
 * interchangeable and are not.
 *
 * `Boot` lands at `GetEndLevel()`, which a [SYLPH] schedule pushes down by
 * exactly the slack it reserves for the slot leg: on `bootparam_35` with slack
 * 8 it is level 8, against 16 with no slack. Bootstrapping `r` -- the output
 * of the inverse square root, as `boot_aux` does -- needs the result to come
 * back **at or above the main track**, and the main track is at 13 there. So
 * `boot_aux` asserts on any set that reserves slack, which is every set this
 * cycle runs on.
 *
 * Moving the bootstrap one step earlier settles it, and the step is forced.
 * The auxiliary value that *can* be bootstrapped by the cycle rather than by
 * `Boot` is the affine map's output, because that is the only point in the
 * iteration that lands on `GetStCLevel()` -- exp (4) + mask (1) + the norm
 * square (1) + the affine multiply (1) is 7 below the operator's level, which
 * is the slack. `SylphSchedule::ToCoeff` needs its input there and nowhere
 * else, so the placement is not a choice.
 *
 * That is what `AuxBoot` is: a hook the caller fills with its own
 * StC-bootstrap-canonicalise, applied to the affine map's output. The
 * normalisation then comes back fresh at the operator's own level, the
 * polynomial is compiled there, and the main track pays only the multiply and
 * the square -- the same 7 levels `boot_aux` was for.
 *
 * The hook is handed a `magnitude` because the two ends want opposite things
 * again. `||y||_2^2 / a` reaches `2 hi / (hi - lo)`, which for the calibrated
 * [1.54, 6.82] is 2.58 and is well outside what a bootstrap carries; the
 * handler therefore divides by an extra constant on the way in -- free, it
 * rides the affine multiply that is already there -- and tells the hook what
 * to multiply back. `SylphSchedule::Canonicalise` takes exactly that argument
 * and is also already being paid for.
 *
 * What remains unimplemented is section 3.4's *slim* evaluation. The
 * normalisation is one value per row, so it belongs in a sparsely-packed
 * ciphertext whose bootstrap is much cheaper than a full one; here it is
 * broadcast across every slot by the reduction and bootstrapped at full width.
 * That is a cost optimisation, not a level one -- theorem 1 keeps k+1 levels
 * for degree 2^k either way.
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
  int num_rows_;
  double range_;
  int input_level_;
  int num_iters_;

  // norm^2 interval entering each inverse square root, from calibration
  std::vector<double> norm_lo_, norm_hi_;

  bool boot_aux_;
  int aux_return_level_;
  int aux_in_level_;
  std::vector<double> aux_shrink_;
  int exp_out_level_;
  std::unique_ptr<EvalPoly<word>> exp_poly_;
  std::vector<std::unique_ptr<EvalPoly<word>>> inv_sqrt_;
  std::vector<int> rotation_distances_;

  // The causal mask is a full-width plaintext, and Encoder::Encode costs tens
  // of milliseconds on the host -- SpecialIFFT over every slot, then
  // num_primes * degree BigInt reductions, single-threaded, before anything
  // reaches the GPU. Caching it keyed on the mask's own values keeps Apply's
  // signature honest: a caller that changes the mask gets a new plaintext, and
  // comparing 32768 values costs about 0.03 ms against a 35 ms encode.
  mutable std::vector<Complex> cached_mask_;
  mutable int cached_mask_level_ = -1;
  mutable Pt mask_pt_;

  std::unique_ptr<EvalPoly<word>> MakeInvSqrt(double lo, double hi, int degree,
                                              int level);

 public:
  /**
   * @param context the evaluation context
   * @param num_keys d, the number of entries in one SoftMax row; must be a
   * power of two. **The key axis is strided, not contiguous**: key i of row r
   * sits at slot `r + i * (num_slots / d)`. Cheddar rotates cyclically over
   * the whole slot vector, so a contiguous row would have the rotate-and-add
   * straddle row boundaries; striding makes the wrap-around exactly right, and
   * it is the axis convention RmsNormHandler already uses.
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
   * @param boot_aux bootstrap the normalisation before it meets the main
   * track, as [SYLPH] figure 2 does, using an ordinary `Boot`. This changes
   * the level each polynomial is compiled at, so it is fixed here rather than
   * per call, and Apply then requires a BootContext. **Only usable on a
   * parameter set with no slack** -- see the header.
   * @param aux_return_level the level an `AuxBoot` hook returns the auxiliary
   * value at, or -1 for no hook. Supplying it selects the hook over
   * `boot_aux`, which it supersedes on any set that reserves slack.
   * @param aux_boot_max the magnitude the hook is expected to be able to
   * carry, which sets the constant the handler divides by before calling it
   */
  SoftMaxHandler(ConstContextPtr<word> context, int num_keys, double range,
                 int input_level, int num_iters,
                 const std::vector<double> &norm_lo,
                 const std::vector<double> &norm_hi, int exp_degree,
                 int inv_sqrt_degree, int early_inv_sqrt_degree = 4,
                 bool boot_aux = false, int aux_return_level = -1,
                 double aux_boot_max = 0.5);

  /**
   * @brief The caller's bootstrap for the auxiliary track.
   *
   * `res` must come back slot-encoded at `aux_return_level`, on that level's
   * canonical scale, carrying `magnitude` times the message it was given.
   */
  using AuxBoot = std::function<void(Ct &res, const Ct &x, double magnitude)>;

  /** @brief The level the hook will be called at, so a caller can check it. */
  int GetAuxCallLevel() const { return aux_in_level_; }

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
   * @param boot_context if given, the normalisation is bootstrapped before it
   * meets the main track, which is what keeps the main track at 7 levels
   * instead of 13. See GetMainTrackDepth().
   */
  void Apply(Ct &res, const Ct &x_scaled,
             const std::vector<Complex> &causal_mask,
             const EvkMap<word> &evk_map,
             const BootContext<word> *boot_context = nullptr,
             const AuxBoot *aux_boot = nullptr) const;

  /**
   * @brief Levels the main track spends, which is what a level budget counts.
   *
   * exp, the causal mask, then k times (multiply by the normalisation, square).
   * The auxiliary work -- the norm square, the affine map and the inverse
   * square root -- only lands on the main track when it is not bootstrapped.
   */
  int GetMainTrackDepth() const;

  /** @brief Levels the auxiliary track spends per iteration. */
  int GetAuxTrackDepth() const;

  /**
   * @brief Encode the causal mask up front, so its cost sits in setup.
   *
   * [SYLPH] section 5.1 keeps the model's plaintexts resident on the GPU for
   * the whole run and section 5.3 makes that conversion its own stage. Apply
   * does the same work lazily on first call, which is correct but hides a
   * ~50 ms host encode inside what reads as an online measurement.
   */
  void Prepare(const std::vector<Complex> &causal_mask) const;

  /** @brief Device bytes the cached mask holds, 0 before Prepare. */
  size_t GetPlaintextBytes() const;
};

}  // namespace cheddar
