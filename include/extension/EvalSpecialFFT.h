#pragma once

#include <utility>
#include <vector>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootParameter.h"
#include "extension/LinearTransform.h"

namespace cheddar {

/**
 * @brief A class for the homomorphic evaluation of special FFT
 * (SlotToCoeff/StC) and IFFT (CoeffToSlot/CtS)
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class EvalSpecialFFT {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Complex = std::complex<double>;

  const int num_slots_;
  BootParameter boot_param_;

  const double cts_const_;
  const double stc_const_;
  const bool full_slot_;

  std::vector<LinearTransform<word>> cts_phases_;
  std::vector<LinearTransform<word>> stc_phases_;

  std::vector<StripedMatrix> plain_fft_stages_;
  std::vector<StripedMatrix> plain_ifft_stages_;

  // The partial conversions of [SYLPH] section 3.2. Zero or one entry each;
  // a vector rather than an optional because LinearTransform has no default
  // constructor and owns device state.
  int sinc_sub_degree_ = 0;
  std::vector<LinearTransform<word>> sinc_stc_;
  std::vector<LinearTransform<word>> sinc_cts_;
  std::vector<LinearTransform<word>> sinc_prefix_;
  int sinc_prefix_shift_ = 0;
  int sinc_prefix_level_ = -1;

  std::pair<int, int> BSGSSplit(int num_diag) const;
  void PopulatePlainMatrices(ConstContextPtr<word> context);
  void PreparePlaintexts(ConstContextPtr<word> context);

 public:
  EvalSpecialFFT(ConstContextPtr<word> context, const BootParameter &boot_param,
                 int num_slots, double cts_const, double stc_const);

  EvalSpecialFFT(const EvalSpecialFFT &) = delete;
  EvalSpecialFFT &operator=(const EvalSpecialFFT &) = delete;
  EvalSpecialFFT(EvalSpecialFFT &&) = default;

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  void EvaluateCtS(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;
  void EvaluateStC(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;

  /**
   * @brief SLOTS <-> SinC, the "partial bit-reversal operation which occurs
   * when moving to SinC encoding" of [SYLPH] section 3.2.
   *
   * ## The identity
   *
   * Writing `n = num_slots`, `L = log2 n`, `d = degree / sub_degree` and
   * `p = log2 d`, the slot -> SinC(sub_degree) map is **exactly the last `p`
   * butterfly stages of StC**, in StC's own ascending-stride order, with no
   * scalar:
   *
   *     U = plain_fft_stages_[L-1] . ... . plain_fft_stages_[L-p]
   *
   *     input slot s  ->  SinC block  BitRev_p( s >> log2(sub_degree/2) )
   *                       SinC lane   s mod (sub_degree/2)
   *
   * The bit reversal is on the **block index only**; the lanes are untouched.
   * Both endpoints are the ones the definition collapses to: `sub_degree == 2`
   * gives `p = L`, the whole of StC and hence `EncodeCoeff`; `sub_degree ==
   * degree` gives `p = 0`, the identity and hence `Encode`.
   *
   * **The prefix is a different map.** The low-stride stages act inside a lane
   * block, which is the wrong axis, and using them gives the wrong answer for
   * every non-degenerate case. This is a suffix.
   *
   * The inverse is the mirror image: the **first** `p` stages of CtS, in CtS's
   * own order (descending stride), times `1/d` -- `ifft_i . fft_i = 2I` stage
   * by stage, so `p` stages compose to `2^p = d`.
   *
   * ## What it costs
   *
   * The suffix product has exactly `d` diagonals at stride `sub_degree/2`. At
   * `sub_degree = 512` that is 128 of them and one `LinearTransform` -- one
   * level, against the full StC's `num_stc_levels_`. At `sub_degree = 32`,
   * which is what Llama-3's per-head `128 x 128` product needs, it is 2048,
   * and 2048 plaintexts at eleven limbs do not fit anywhere. So the stages are
   * splittable across `num_phases` transforms exactly as StC's own are: three
   * phases of 4 + 4 + 3 stages are 16 + 16 + 8 = 40 plaintexts and three
   * levels, which is what StC would have cost anyway.
   *
   * Nothing here is min-KS or pre-rotated: a standalone conversion has no
   * neighbouring phase to share a rotation with.
   *
   * ## Why it is not `EvalSpecialFFT` with a different `num_slots`
   *
   * Because that selects the *low*-stride stages, not the high ones -- stage
   * `i` of an `n'`-slot transform uses the twiddle modulus `2^(i+3)`, and no
   * `n'` makes `{0..log2 n'-1}` coincide with `{L-p..L-1}`. A smaller
   * `num_slots` also switches on the sparse-packing path, which is a different
   * thing again.
   *
   * @param sub_degree k, a power of two dividing the ring degree
   * @param stc_level the level the slots -> SinC transform is compiled at; it
   *        returns `num_phases` below
   * @param cts_level likewise for SinC -> slots
   * @param num_phases how many `LinearTransform`s the `p` stages are split
   *        into, one level each. **This is the whole cost question.** A
   *        product of `q` stages has `2^q` diagonals and a diagonal is a full
   *        plaintext at the transform's limb count, so one phase carrying all
   *        `p` is `2^p` plaintexts -- 2048, i.e. gigabytes, at the
   *        `sub_degree = 32` the attention product wants -- while three phases
   *        of 4 + 4 + 3 are 40. It is the same trade StC itself makes
   *        (`num_stc_levels_`, three on every logN=16 preset), and it is
   *        level-neutral in the pipeline, because a tensor bound for the
   *        product pays this *instead of* SlotToCoeff rather than on top of it.
   */
  void PrepareSinC(ConstContextPtr<word> context, int sub_degree,
                   int stc_level, int cts_level, int num_phases = 1);

  /// How many levels a conversion spends: one per phase.
  int GetSinCNumPhases() const { return static_cast<int>(sinc_stc_.size()); }
  /// The BSGS split of each forward phase, for reports.
  std::pair<int, int> GetSinCPhaseBSGS(int phase) const {
    return {sinc_stc_.at(phase).GetBS(), sinc_stc_.at(phase).GetGS()};
  }

  /// The `sub_degree` PrepareSinC was called with, or 0.
  int GetSinCSubDegree() const { return sinc_sub_degree_; }

  /**
   * @brief The PREFIX of SlotToCoeff -- the part HalfBoot leaves undone.
   *
   * `SlotToCoeff` is `num_stages` butterfly stages and `SlotToSinC` is the
   * last `p = log2(degree/sub_degree)` of them, so `StC = SlotToSinC . P` with
   * `P` the first `num_stages - p`. A ciphertext holding `SlotToSinC(s)` is
   * therefore `StC(P^-1(s))`, and `HalfBoot` -- which inverts the whole of StC
   * -- hands back `P^-1(s)`, not `s`. This is `P`.
   *
   * It is not a relabelling that could be absorbed into the layout: the
   * butterfly stages combine values with twiddle factors, so what comes back
   * without it is a linear MIXTURE of the scores rather than a permutation of
   * them.
   *
   * At the attention product's `sub_degree = 32` it is 4 stages and 31
   * diagonals -- one level, which the schedule spends on `Canonicalise`'s
   * multiply anyway, plus one HRot for the window (see the .cpp).
   *
   * @param level the level it is compiled at, i.e. where HalfBoot lands
   * @param num_phases how many LinearTransforms to split the stages across,
   *        one level each; 1 is right at `sub_degree = 32`
   * @param constant a scalar folded into the transform's own matrix, and
   *        `pt_scale` the scale its plaintexts are encoded at. Together they
   *        are `SylphSchedule::Canonicalise`: that operator is one multiply by
   *        a constant and one rescale, at the level HalfBoot lands on, which
   *        is exactly this level. Folding it in here is what makes the prefix
   *        free -- the schedule was spending the level anyway -- and it is
   *        also what restores the magnitude HalfBoot divided out, which is
   *        worth about five bits. `pt_scale <= 0` means the default,
   *        `GetRescalePrimeProd(level)`, which preserves the input scale and
   *        folds nothing.
   */
  void PrepareSinCPrefix(ConstContextPtr<word> context, int sub_degree,
                         int level, int num_phases = 1,
                         double constant = 1.0, double pt_scale = -1.0);

  /**
   * @brief The composed prefix matrix and the window rotation it needs, for a
   * caller that wants to compile more than one of them.
   *
   * `PrepareSinCPrefix` holds exactly one, which is enough when the constant
   * folded into it is the same every time. The two attention products need
   * different ones -- `Canonicalise`'s magnitude is `2 / size_scores` for the
   * scores and 1 for the values -- so `SinCAttention` builds its own pair from
   * this. The window is the `additional_pt_rot` the transform must be given,
   * with `pre_rotation = -window`; the result then comes back rotated by
   * `window` and one HRot by `-window` finishes it. See the .cpp for why a
   * window is needed at all.
   *
   * @param sub_degree k
   * @param window receives `2^(num_stages - p)`
   */
  StripedMatrix SinCPrefixMatrix(int sub_degree, int &window) const;

  /// Levels the prefix spends, or 0 if it was never prepared.
  int GetSinCPrefixNumPhases() const {
    return static_cast<int>(sinc_prefix_.size());
  }
  /// The residual rotation the window leaves, undone inside EvaluateSinCPrefix.
  int GetSinCPrefixShift() const { return sinc_prefix_shift_; }

  void AddRequiredSinCRotations(EvkRequest &req) const;
  void AddRequiredSinCPrefixRotations(EvkRequest &req) const;

  void EvaluateSlotToSinC(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
  void EvaluateSinCToSlot(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
  void EvaluateSinCPrefix(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar