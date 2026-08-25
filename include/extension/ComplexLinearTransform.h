#pragma once

#include <map>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/LinearTransform.h"
#include "extension/StripedMatrix.h"

namespace cheddar {

/**
 * @brief A linear transform with a COMPLEX matrix, over ciphertexts whose slots
 * are REAL -- which is what the conjugate-invariant ring forces on every stage
 * of CoeffToSlot and SlotToCoeff.
 *
 * ## Why this exists
 *
 * R+ is totally real, so a slot holds one real number and Encode refuses an
 * imaginary part. The encoding map is real too: `E^-1 = Re(SpecialIFFT)`. But
 * the butterfly factorisation that makes CtS cheap is a factorisation of the
 * *complex* transform, and taking real parts does not commute with it --
 * `Re(W) != Re(W_k) ... Re(W_1)`, because a stage's real output depends on its
 * input's imaginary part as much as on its real one. So both ends of CtS are
 * real and everything in between is not.
 *
 * On the ordinary ring the intermediate is redundant -- `f[N-j] = -i conj(f[j])`
 * -- and the bootstrap collapses it with HConj. That redundancy is an index
 * REVERSAL, and the slot group of R+ is `(Z/4N)^* / {+-1}`, cyclic of order N:
 * the only slot permutations it has are rotations. The one operation the real
 * subring makes free (conjugation, which acts trivially there) is the one
 * operation the collapse would need. Both are the same statement about `{+-1}`.
 * Doing.md 1.5bb has the derivation.
 *
 * So the intermediate is carried as a PAIR of real ciphertexts, `(re, im)`, and
 * a complex matrix M acts on it as
 *
 *     re' = Re(M) re - Im(M) im
 *     im' = Im(M) re + Re(M) im
 *
 * which is two LinearTransforms over the same diagonal offsets.
 *
 * ## What it costs, and what it does not
 *
 * The BABY steps -- the rotations of the input -- are shared: Re(M) and Im(M)
 * are compiled from the same offsets, so one baby-step map per input ciphertext
 * feeds both.
 *
 * The GIANT steps are shared too, and that needs a trick. A giant step is
 * `sum_g rot_g( sum_b pt[g][b] * bs[b] )`, and HoistHandler looks its
 * ciphertexts up by baby-step index as a plain map key. So `merged_` holds
 * Re(M) at key `b` and Im(M) at key `b + width`, over one set of giant steps,
 * and the caller hands it a map carrying two different ciphertexts under those
 * two key blocks. One rotation per giant step then produces a combination of
 * both inputs:
 *
 *     re' = Re(M) re - Im(M) im   <-  merged_( {b: re, b+w: -im} )
 *     im' = Im(M) re + Re(M) im   <-  merged_( {b: im, b+w:  re} )
 *
 * The negation is one element-wise pass over the baby-step results, which is
 * nothing beside a key switch. The fused plaintext-accumulation kernel takes
 * the doubled baby-step count in its stride, so the two halves are one launch
 * rather than two. Against one ordinary LinearTransform at `bs + gs` rotations:
 *
 *     EvaluateFromReal   1 * bs + 2 * gs   (one real input, complex output)
 *     EvaluatePair       2 * bs + 2 * gs
 *     EvaluateToReal     2 * bs + 1 * gs   (only the real part is wanted)
 *
 * min_ks does not take this route: its giant step derives a single stride from
 * the baby-step sequence, and the aliased keys are not that sequence. It falls
 * back to four unfused passes, which is what min_ks trades anyway -- key count
 * against time.
 *
 * The plaintexts are compiled four times over -- Re and Im standalone for
 * EvaluateFromReal and for the baby-step index set, and again inside `merged_`.
 * A diagonal whose part vanishes is kept rather than pruned, so that the halves
 * present identical baby/giant structure and can share a map.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class ComplexLinearTransform {
 private:
  using Ct = Ciphertext<word>;

  LinearTransform<word> re_;
  LinearTransform<word> im_;

  // Re(M) at baby-step key b, Im(M) at b + alias_offset_, over one set of giant
  // steps. Never asked for its own baby step or its own rotations: the aliased
  // keys are not rotation amounts, and re_ / im_ answer for both.
  const int alias_offset_;
  HoistHandler<word> merged_;

  static StripedMatrix TakePart(const StripedMatrix &matrix, bool imaginary);
  static PlainHoistMap MergedHoistMap(const StripedMatrix &matrix, int bs,
                                      int gs, int pre_rotation,
                                      int additional_pt_rot, int alias_offset);

 public:
  ComplexLinearTransform(ConstContextPtr<word> context,
                         const StripedMatrix &matrix, int pt_level,
                         double pt_scale, int bs, int gs = 1,
                         int pre_rotation = 0, int additional_pt_rot = 0);

  ComplexLinearTransform(const ComplexLinearTransform &) = delete;
  ComplexLinearTransform &operator=(const ComplexLinearTransform &) = delete;
  ComplexLinearTransform(ComplexLinearTransform &&) = default;

  int GetBS() const { return re_.GetBS(); }
  int GetGS() const { return re_.GetGS(); }
  int GetPreRotationAmount() const { return re_.GetPreRotationAmount(); }

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  /**
   * @brief `(res_re, res_im) = M * input`, with `input` real.
   *
   * The entry point of a conjugate-invariant CtS or StC: the ciphertext coming
   * in holds real slots and the one leaving does not.
   */
  void EvaluateFromReal(ConstContextPtr<word> context, Ct &res_re, Ct &res_im,
                        const Ct &input, const EvkMap<word> &evk_map,
                        bool min_ks = false) const;

  /**
   * @brief `(res_re, res_im) = M * (in_re + i in_im)`. An output may alias an
   * input: every read happens before either output is written.
   */
  void EvaluatePair(ConstContextPtr<word> context, Ct &res_re, Ct &res_im,
                    const Ct &in_re, const Ct &in_im,
                    const EvkMap<word> &evk_map, bool min_ks = false) const;

  /**
   * @brief `res = Re(M * (in_re + i in_im))`.
   *
   * The exit: the last phase of a conjugate-invariant CtS or StC, where the
   * imaginary half of the answer is redundant and is never formed. That is what
   * makes the last phase cost what an ordinary one costs per input ciphertext
   * rather than twice it.
   */
  void EvaluateToReal(ConstContextPtr<word> context, Ct &res, const Ct &in_re,
                      const Ct &in_im, const EvkMap<word> &evk_map,
                      bool min_ks = false) const;
};

}  // namespace cheddar
