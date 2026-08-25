#pragma once

#include <iostream>
#include <map>
#include <set>
#include <unordered_map>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/Hoist.h"
#include "extension/StripedMatrix.h"

namespace cheddar {

/**
 * @brief A factory class for EvalHoist.
 *
 * @tparam word
 */
template <typename word>
class LinearTransform {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Dv = DeviceVector<word>;

  int pt_level_;
  double pt_scale_;

  int bs_;
  int gs_;
  int pre_rotation_;
  int additional_pt_rot_;

  // shoule be the last members
  int stride_;
  std::set<int> diag_offsets_;
  HoistHandler<word> hoist_;

  static std::set<int> ExtractDiagOffsets(const StripedMatrix &matrix);

 public:
  /**
   * @brief The two setup steps, as free functions, so that a caller can build a
   * hoist map this class would not: `ComplexLinearTransform` merges the maps of
   * two matrices under one set of giant steps, which needs both halves compiled
   * against the same stride and split.
   */
  static int DetermineStride(const StripedMatrix &matrix, int bs, int gs,
                             int pre_rotation);
  static PlainHoistMap ConstructPlainHoistMap(const StripedMatrix &matrix,
                                              int stride, int bs,
                                              int pre_rotation,
                                              int additional_pt_rot);

  LinearTransform(ConstContextPtr<word> context, const StripedMatrix &matrix,
                  int pt_level, double pt_scale, int bs, int gs = 1,
                  int pre_rotation = 0, int additional_pt_rot = 0);

  bool IsUsingBSGS() const;
  int GetBS() const;
  int GetGS() const;
  int GetPreRotationAmount() const;

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const EvkMap<word> &evk_map, bool min_ks = false) const;

  /**
   * @brief The two halves of Evaluate, separately.
   *
   * `Evaluate` is `EvaluateBabyStep` followed by `EvaluateGiantStep`, and the
   * baby step is where the rotations of the *input* happen -- so a caller with
   * more than one matrix to apply to the same input pays for them once by
   * doing the split itself. That is what `ComplexLinearTransform` needs: on the
   * real subring a complex matrix is a pair of real ones over the same
   * ciphertext, and running them as two `Evaluate` calls would rotate the input
   * twice for no reason.
   *
   * The `bs` map is keyed by baby-step index, and `EvaluateGiantStep` looks its
   * entries up by that key alone. Two transforms compiled from matrices with
   * the same diagonal offsets, `bs`, `gs` and `pre_rotation` therefore share a
   * baby-step map exactly; `ComplexLinearTransform` asserts that they do.
   */
  void EvaluateBabyStep(ConstContextPtr<word> context, std::map<int, Ct> &bs,
                        const Ct &input, const EvkMap<word> &evk_map,
                        bool min_ks = false) const;
  void EvaluateGiantStep(ConstContextPtr<word> context, Ct &res,
                         const std::map<int, Ct> &bs,
                         const EvkMap<word> &evk_map,
                         bool min_ks = false) const;

  /// The diagonal offsets the matrix was compiled from, for a caller pairing
  /// two transforms and needing to know they line up.
  const std::set<int> &GetDiagonalOffsets() const { return diag_offsets_; }
};

}  // namespace cheddar