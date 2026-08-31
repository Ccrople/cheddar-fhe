#pragma once

#include <utility>
#include <vector>

#include "common/CommonUtils.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/ComplexLinearTransform.h"
#include "extension/LinearTransform.h"
#include "extension/StripedMatrix.h"

namespace cheddar {

/**
 * @brief SlotToCoeff and CoeffToSlot on the conjugate-invariant ring that
 * read the MODULE basis `{c_{kt} c_i}` instead of the native basis
 * `{1, c_j}`.
 *
 * ## Why this exists (Doing.md 3.5)
 *
 * R+ is free over its rank-T subring S = Z[c_k] on `{1, c_1, ..., c_{k-1}}`,
 * and `MlweHandler::ModDecomp` extracts exactly those coordinates (the
 * alternating suffix scan). But every transform in the library maps slots to
 * the NATIVE coefficients, and the native image of module coordinate
 * `(i, t)` is banded -- `r[tk + i] = x_i[t] + x_{k-i}[t+1]` -- so a layer
 * that reads native position `tk + i` as (token, channel) can only use half
 * the components, which is the half density of CLAUDE.md section 3.
 *
 * The module coordinates map to slots as
 *
 *     z_s = sum_i c_i(zeta_s) * sum_t x_i[t] c_{kt}(zeta_s)
 *
 * -- the small ring's transform on every component, then a twist across
 * components, cosines only. In terms of the library's own butterfly stages
 * (`EvalSpecialFFT::PopulatePlainMatrices`, composed here by
 * `CiButterflyStages`) that is:
 *
 *   * **StC**: the stride < T stages, a real-part truncation, the stride >= T
 *     stages; the phase-0 column scaling is `2^[i != 0] 2^[t != 0]` on the
 *     coefficient index (the native transform's is `2^[idx != 0]`). No new
 *     stage matrix.
 *   * **CtS**: the stride >= T stages, then ONE pair correction
 *
 *         s = (v - w Flip v) / (1 - w^2),   Flip: (i, t) -> (k - i, t)
 *
 *     with `w` the conjugate small root of the slot's residue mod T, and
 *     `i = 0` (`s = v / 2`) and `i = k/2` (`s = v / (1 + w)`) the two fixed
 *     classes -- the same two special classes `Mlwe.cu`'s scan has; then the
 *     stride < T stages on a real input, whose last phase carries
 *     `1 / 2^[i != 0]` on its rows. The correction's offsets are the odd
 *     multiples of T and zero -- k/2 + 1 of them -- and it is folded into the
 *     last twist phase (which lives on the same stride-T lattice), so CtS
 *     costs the same number of levels as the native one.
 *
 * All of it is checked coefficient by coefficient on the host in
 * `reference/scripts/ci_module_basis.py` before any of this ran on a device.
 * The phases pass their pre-rotations along by the SinC prefix's window
 * rule; StC ends on a whole-lattice phase and comes back clean, CtS ends on
 * the small stages and spends one closing rotation.
 *
 * ## Conventions
 *
 * Both directions keep the library's bit-reversed placement: slot `s` holds
 * module coordinate `flat = BitReverse(s)` with `flat = t * k + i`, exactly
 * where the native transforms put native coefficient `BitReverse(s)`. With
 * `stc_const = cts_const = 1` what comes back is the answer itself: StC
 * produces the element whose module coordinates ARE the input slots, and CtS
 * produces the module coordinates of its input element in its slots.
 *
 * The inverse direction amplifies noise that enters in native coordinates
 * before the correction by `||P^-1||` -- `2T / pi` worst case, `0.68 sqrt(T)`
 * rms for iid noise (`Mlwe.h`) -- which is what `ModDecomp` already pays on
 * the input side of every projection.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiModuleBasis {
 private:
  using Ct = Ciphertext<word>;
  using Transform = ComplexLinearTransform<word>;

  int num_slots_ = 0;
  int small_degree_ = 0;  // T
  int rank_ = 0;          // k
  int log_slots_ = 0;
  int log_small_ = 0;
  int log_rank_ = 0;
  int stc_level_ = -1;
  int cts_level_ = -1;

  // A group is real in and real out (the truncation is what makes the module
  // transform two such groups). One phase: the real part of its matrix as an
  // ordinary LinearTransform, `Re(M x) = Re(M) x`. More: the complex
  // intermediate carried as a pair, lifted by the first and dropped by the
  // last. Exactly one of the two vectors is populated.
  struct Group {
    std::vector<LinearTransform<word>> real;
    std::vector<Transform> pair;
  };
  std::vector<Group> stc_groups_;  // small, then twist
  std::vector<Group> cts_groups_;  // twist, then small
  std::vector<int> stc_diagonals_;
  std::vector<int> cts_diagonals_;
  // The rotation a chain of phases leaves on its output (the SinC prefix's
  // window rule, see the .cpp), undone by one HRot when nonzero.
  int stc_shift_ = 0;
  int cts_shift_ = 0;

  static std::pair<int, int> Split(int num_diag);
  static int NumLevels(const std::vector<Group> &groups);
  StripedMatrix Correction(const Parameter<word> &param,
                           const Encoder<word> &encoder) const;
  // Compile `matrices` as consecutive phases from `start_level` down, cut
  // into groups of `group_sizes`, with the pre-rotations chained; returns
  // the rotation left on the output.
  int Chain(ConstContextPtr<word> context, std::vector<StripedMatrix> &matrices,
            const std::vector<int> &group_sizes, int start_level,
            std::vector<Group> &groups, std::vector<int> &diagonals) const;
  static void RunGroup(ConstContextPtr<word> context, const Group &group,
                       Ct &res, const Ct &input, const EvkMap<word> &evk_map);

 public:
  /**
   * @brief Stages per phase, in application order. The small groups must sum
   * to `log2 T`, the twist groups to `log2 k`. A group of one phase is a
   * single real transform (255 diagonals for seven small stages, the whole
   * stride-T lattice -- k of them -- for the twist); a longer group is a
   * pair chain with fewer diagonals per level. The default is two real
   * phases a direction, which is fewer levels than the native transforms.
   */
  struct Phases {
    std::vector<int> stc_small{7};
    std::vector<int> stc_twist{9};
    std::vector<int> cts_twist{9};
    std::vector<int> cts_small{7};
  };

  /**
   * @param context a conjugate-invariant Context, full slots
   * @param small_degree T, the rank of the subring the module is taken over
   * @param stc_level the level StC is compiled at (it returns
   *        `GetStCNumLevels()` below), or -1 to skip building it
   * @param cts_level likewise for CtS
   * @param phases the stage grouping
   * @param stc_const a constant folded into StC, spread over its phases
   * @param cts_const likewise for CtS
   */
  CiModuleBasis(ConstContextPtr<word> context, int small_degree, int stc_level,
                int cts_level, const Phases &phases = Phases(),
                double stc_const = 1.0, double cts_const = 1.0);

  CiModuleBasis(const CiModuleBasis &) = delete;
  CiModuleBasis &operator=(const CiModuleBasis &) = delete;
  CiModuleBasis(CiModuleBasis &&) = default;

  int GetSmallDegree() const { return small_degree_; }
  int GetRank() const { return rank_; }
  int GetStCNumLevels() const { return NumLevels(stc_groups_); }
  int GetCtSNumLevels() const { return NumLevels(cts_groups_); }
  /// Diagonal counts of the compiled phases, StC then CtS, in evaluation order.
  const std::vector<int> &GetStCDiagonals() const { return stc_diagonals_; }
  const std::vector<int> &GetCtSDiagonals() const { return cts_diagonals_; }

  /// The module coordinate `t * k + i` that slot `s` carries, both directions.
  int ModuleIndexOfSlot(int slot) const {
    return static_cast<int>(BitReverseInt(slot, log_slots_));
  }

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  /**
   * @brief Slots -> the element whose module coordinates are the slots.
   * `res` is coefficient-encoded at `stc_level - GetStCNumLevels()`.
   */
  void EvaluateStC(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map) const;

  /**
   * @brief An element -> its module coordinates in the slots.
   * `res` is slot-encoded at `cts_level - GetCtSNumLevels()`.
   */
  void EvaluateCtS(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
