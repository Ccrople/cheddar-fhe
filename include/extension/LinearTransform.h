#pragma once

#include <iostream>
#include <map>
#include <set>
#include <unordered_map>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/Serialization.h"
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

  int DetermineStride(const StripedMatrix &matrix);
  PlainHoistMap ConstructPlainHoistMap(const StripedMatrix &matrix);
  static std::set<int> ExtractDiagOffsets(const StripedMatrix &matrix);

  // Everything a compiled transform holds except the plaintexts, read before
  // `hoist_` is constructed. `hoist_` has no default constructor and must be
  // the last member, so a deserializing constructor cannot read the scalars in
  // its own body; passing them in makes the archive's read order explicit
  // instead of leaving it to member-initialiser evaluation order.
  struct FromArchive {
    int pt_level = 0;
    double pt_scale = 0.0;
    int bs = 0;
    int gs = 0;
    int pre_rotation = 0;
    int additional_pt_rot = 0;
    int stride = 0;
    std::set<int> diag_offsets;
  };
  LinearTransform(FromArchive &&head, ArchiveReader &ar);

 public:
  LinearTransform(ConstContextPtr<word> context, const StripedMatrix &matrix,
                  int pt_level, double pt_scale, int bs, int gs = 1,
                  int pre_rotation = 0, int additional_pt_rot = 0);

  LinearTransform(const LinearTransform &) = delete;
  LinearTransform &operator=(const LinearTransform &) = delete;
  LinearTransform(LinearTransform &&) = default;

  /**
   * @brief Write the compiled transform: its BSGS structure and every
   * plaintext diagonal.
   *
   * What this caches is the host-side matrix construction that feeds
   * `CompilePlaintexts` -- for the Llama leg's converters, seconds to minutes
   * per transform -- and the plaintexts themselves, which the device encodes
   * in seconds.
   */
  void Save(ArchiveWriter &ar) const;

  /** @brief Rebuild a transform written by `Save`. */
  static LinearTransform Load(ArchiveReader &ar);

  bool IsUsingBSGS() const;
  int GetBS() const;
  int GetGS() const;
  int GetPreRotationAmount() const;

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  /// Residency of the compiled plaintexts; see `HoistHandler::Unstage`.
  void Unstage() const { hoist_.Unstage(); }
  void Stage() const { hoist_.Stage(); }
  bool IsOnDevice() const { return hoist_.IsOnDevice(); }
  size_t PlaintextBytes() const { return hoist_.PlaintextBytes(); }

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

  /**
   * @brief `HoistHandler::EvaluateGiantStepComplex` over the handlers of two
   * transforms holding Re(M) and Im(M); see Hoist.h for the contract. Here
   * because the handlers are private to the transforms that own them.
   */
  static void EvaluateGiantStepComplex(ConstContextPtr<word> context,
                                       Ct &res_re, Ct *res_im,
                                       const LinearTransform &re_t,
                                       const LinearTransform &im_t,
                                       const std::map<int, Ct> &bs_re,
                                       const std::map<int, Ct> *bs_im,
                                       const EvkMap<word> &evk_map);

  /// The diagonal offsets the matrix was compiled from, for a caller pairing
  /// two transforms and needing to know they line up.
  const std::set<int> &GetDiagonalOffsets() const { return diag_offsets_; }

  /// The compiled handler, for a caller reading its plaintexts.
  const HoistHandler<word> &GetHoist() const { return hoist_; }
};

}  // namespace cheddar