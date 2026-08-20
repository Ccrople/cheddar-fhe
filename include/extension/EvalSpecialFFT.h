#pragma once

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
   * The suffix product has exactly `d` diagonals at stride `sub_degree/2`, so
   * it compiles to **one** `LinearTransform` and therefore **one level**,
   * against the full StC's `num_stc_levels_` (three on every logN=16 preset).
   * At degree 65536 and `sub_degree = 512` that is 128 diagonals, BSGS 16x8.
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
   *        returns one below
   * @param cts_level likewise for SinC -> slots
   */
  void PrepareSinC(ConstContextPtr<word> context, int sub_degree,
                   int stc_level, int cts_level);

  /// The `sub_degree` PrepareSinC was called with, or 0.
  int GetSinCSubDegree() const { return sinc_sub_degree_; }

  void AddRequiredSinCRotations(EvkRequest &req) const;

  void EvaluateSlotToSinC(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
  void EvaluateSinCToSlot(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar