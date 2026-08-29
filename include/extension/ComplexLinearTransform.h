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
 * are compiled from the same offsets, so one baby-step map per input
 * ciphertext feeds both.
 *
 * The GIANT step runs once over both halves
 * (HoistHandler::EvaluateGiantStepComplex): one kernel streams every
 * plaintext limb once and feeds all four real products from it, with the
 * minus sign carried as a modular subtraction, and then each output half is
 * rotated and mod-downed as an ordinary giant step. Against one ordinary
 * LinearTransform at bs + gs rotations:
 *
 *     EvaluateFromReal   1 * bs + 2 * gs   (one real input, complex output)
 *     EvaluatePair       2 * bs + 2 * gs
 *     EvaluateToReal     2 * bs + 1 * gs   (only the real part is wanted)
 *
 * and the plaintext STREAM per phase is one pass over Re and Im together --
 * less than the two passes two ordinary transforms make over their own
 * matrices, because here the matrix is shared between the outputs.
 *
 * min_ks does not take this route: its giant step derives a single stride
 * from the baby-step sequence and rotates sequentially, which the fused pass
 * has no analogue of. It falls back to four unfused passes, which is what
 * min_ks trades anyway -- key count against time.
 *
 * The plaintexts are compiled twice -- Re(M) and Im(M), each a full
 * LinearTransform -- and a diagonal whose part vanishes is kept rather than
 * pruned, so that the halves present identical baby/giant structure. That is
 * 2x an ordinary transform's plaintext memory, which is what the doubled
 * matrix content costs and no more.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class ComplexLinearTransform {
 private:
  using Ct = Ciphertext<word>;

  LinearTransform<word> re_;
  LinearTransform<word> im_;


  static StripedMatrix TakePart(const StripedMatrix &matrix, bool imaginary);

  // Deserializing constructor. `LinearTransform` has no default constructor,
  // so `Load` reads the two halves into named locals -- argument evaluation
  // order is unspecified and these are read from a sequential stream -- and
  // hands them over here.
  ComplexLinearTransform(LinearTransform<word> &&re, LinearTransform<word> &&im);

 public:
  ComplexLinearTransform(ConstContextPtr<word> context,
                         const StripedMatrix &matrix, int pt_level,
                         double pt_scale, int bs, int gs = 1,
                         int pre_rotation = 0, int additional_pt_rot = 0);

  ComplexLinearTransform(const ComplexLinearTransform &) = delete;
  ComplexLinearTransform &operator=(const ComplexLinearTransform &) = delete;
  ComplexLinearTransform(ComplexLinearTransform &&) = default;

  /** @brief Write both halves, real first. */
  void Save(ArchiveWriter &ar) const;

  /** @brief Rebuild a transform written by `Save`. */
  static ComplexLinearTransform Load(ArchiveReader &ar);

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
