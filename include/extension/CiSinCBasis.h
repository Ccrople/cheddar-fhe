#pragma once

#include <string>
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
 * @brief The SinC conversions of the conjugate-invariant CC-MM chain, fused
 * into the bootstrap the way [SYLPH] section 2.1.2 / 3.3 fuses them: the
 * forward is a partial SlotToCoeff, the return is a HalfBoot plus the prefix
 * it leaves undone -- read on the TOWER basis of R+.
 *
 * ## The tower (reference/scripts/ci_nested_sinc.py)
 *
 * The chain's operand is NESTED (Doing.md 1.5bo/1.5bp): the big ciphertext
 * ring-switches into `k_o` parts on the small ring, and each part is its own
 * SinC element at the lane degree `T_l` -- `k_i = small_degree / T_l` blocks
 * of `T_l` lane coefficients. So the operand is the element whose
 * coordinates in the basis
 *
 *     c_i * c_{k_o j} * c_{k_o k_i t},    i < k_o (part), j < k_i (block), t < T_l
 *
 * are the per-part SinC coefficients: the module basis of Doing.md 3.5 taken
 * twice, the small ring's own module structure inside the big one's. Its
 * native image is the banded map applied twice (`MlweHandler::ScanInPlace`
 * with an inner rank). Three consequences, each checked coefficient by
 * coefficient on the host before any of this ran on a device:
 *
 *   * **The forward is the tower StC' with its lane group skipped** -- the
 *     butterfly stages of stride `T_l` and above, real part after each twist
 *     group, with the column scaling `2^[i != 0] 2^[j != 0]` on the input --
 *     and it consumes the message at exactly `CiSwitchedCcmmLayout::
 *     LocateSlot`'s PRIMARY addresses (block `BitRev(j k_o + i)`, lanes
 *     untouched): the SinC identity "the last p stages of StC", read on the
 *     tower. No nested fold, no copy-add: the tower IS the nested layout.
 *   * **The tower CtS'** is the ifft stages in three groups -- the outer
 *     twist with the outer pair correction, the inner twist with the inner
 *     one, the lane stages -- each group real in and real out, so each is
 *     one ordinary `LinearTransform` of `Re(M)`: `k_o`, `2 k_i - 1` and
 *     `2 T_l - 1` diagonals (16 / 255 / 63 at the Llama shape). A
 *     `HalfBoot` whose CoeffToSlot is this (`BootContext::HalfBootTower`)
 *     lands the parts' SinC coefficients in slots.
 *   * **The prefix** is the lane fft stages with the column scaling
 *     `2^[t != 0]`: one real transform of `2 T_l - 1` diagonals that returns
 *     the message to the primary addresses. That is what the archive's 1.5cg
 *     found impossible on the NATIVE basis (the block scan `g^-1` made every
 *     prefix dense): on the tower the scan is the corrections' business, and
 *     the prefix is lane-local.
 *
 * Against the nested `CiSinCConverter` (2048 diagonals a direction, 5 GiB a
 * converter, staged from host memory) this is ~300-450 diagonals a forward
 * across two levels, resident, and the return replaces `SinCToSlot` plus a
 * full `Boot` (its own StC) with `HalfBootTower` plus 63 diagonals, landing
 * two levels higher.
 *
 * ## What the bootstrap needs
 *
 * `HalfBootTower` centres the level-zero representatives in tower
 * coordinates (scan twice, lift, recompose twice) and the wrap-around
 * EvalMod removes is then bounded by the sparse secret's TOWER norm -- so
 * the SSE secret must be sampled sparse in the tower
 * (`CHEDDAR_MODULE_SPARSE_SECRET=<small_degree>:<inner_rank>,<h>`). Measured
 * on the host at the layer's shape (16; 128; 32), h = 16: max 32 / std 5.05
 * against K = 32, so the ring carries K = 64 (`ci16_35_land17c3e10`, ten
 * EvalMod levels); a native- or (128,512)-sparse secret puts it at 780 / 270.
 *
 * ## Conventions
 *
 * Slot `s` of the CtS' output holds tower coordinate `flat = BitRev(s)` with
 * `flat = (t k_i + j) k_o + i`; the forward's input and the prefix's output
 * are the primary addresses `BitRev(j k_o + i) * T_l + lane`. The phases
 * chain their pre-rotations by the SinC prefix's window rule and each
 * direction closes with at most one rotation.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiSinCBasis {
 private:
  using Ct = Ciphertext<word>;
  using Transform = ComplexLinearTransform<word>;

  int num_slots_ = 0;
  int outer_rank_ = 0;   // k_o: parts per switch
  int inner_rank_ = 0;   // k_i: blocks per part
  int lane_degree_ = 0;  // T_l: the SinC sub-degree
  int log_slots_ = 0;
  int log_outer_ = 0;
  int log_inner_ = 0;
  int log_lane_ = 0;

  // A group is real in and real out; one phase compiles as Re(M), more as a
  // pair chain (as CiModuleBasis).
  struct Group {
    std::vector<LinearTransform<word>> real;
    std::vector<Transform> pair;
  };
  struct Chain {
    std::vector<Group> groups;
    std::vector<int> diagonals;
    int level = -1;
    int shift = 0;
    ConstContextPtr<word> context;
  };
  struct NamedForward {
    std::string name;
    Chain chain;
  };
  std::vector<NamedForward> forwards_;
  Chain cts_;
  Chain prefix_;
  // Leading thin single-terminal CtS levels (an even landing ladder's top,
  // e.g. `land18c4e10`'s 25-bit terminal), consumed by a PURE RESCALE before
  // the compiled phases -- EvalSpecialFFT's rule, device-measured there:
  // folding a ~25-bit rescale into a transform phase injects large
  // coefficient noise (the first land18 tower read 2.5 bits).
  std::vector<int> cts_thin_levels_;
  std::vector<Constant<word>> cts_thin_consts_;

  static std::pair<int, int> Split(int num_diag);
  static int NumLevels(const Chain &chain);
  StripedMatrix Correction(const Parameter<word> &param,
                           const Encoder<word> &encoder, bool inner) const;
  void Compile(ConstContextPtr<word> context,
               std::vector<StripedMatrix> &matrices,
               const std::vector<int> &group_sizes, int start_level,
               Chain &chain) const;
  static void RunGroup(ConstContextPtr<word> context, const Group &group,
                       Ct &res, const Ct &input, const EvkMap<word> &evk_map);
  void Run(const Chain &chain, Ct &res, const Ct &input,
           const EvkMap<word> &evk_map) const;
  static void AddChainRotations(const Chain &chain, int num_slots,
                                EvkRequest &req, bool min_ks);
  const Chain &FindForward(const std::string &name) const;

 public:
  /**
   * @brief Stages per phase, in application order; each group must sum to
   * its stage count (`log2 k_i` inner, `log2 k_o` outer, `log2 T_l` lanes).
   * A group of one phase is a single real transform; longer groups are pair
   * chains with fewer diagonals per level. The defaults are one real phase
   * per group: two levels a forward, three a CtS', one a prefix.
   */
  struct Phases {
    std::vector<int> forward_inner;  // empty = {log2 k_i}
    std::vector<int> forward_outer;  // empty = {log2 k_o}
    std::vector<int> cts_outer;      // empty = {log2 k_o}
    std::vector<int> cts_inner;      // empty = {log2 k_i}
    std::vector<int> cts_lane;       // empty = {log2 T_l}
    std::vector<int> prefix;         // empty = {log2 T_l}
  };

  /**
   * @param big_degree the conjugate-invariant big ring's degree (its slot
   *        count too: R+ has one real slot per coefficient)
   * @param small_degree the ring-switch target's degree, `k_o` parts
   * @param sub_degree the SinC lane degree `T_l`
   */
  CiSinCBasis(int big_degree, int small_degree, int sub_degree);

  CiSinCBasis(const CiSinCBasis &) = delete;
  CiSinCBasis &operator=(const CiSinCBasis &) = delete;

  int GetOuterRank() const { return outer_rank_; }
  int GetInnerRank() const { return inner_rank_; }
  int GetLaneDegree() const { return lane_degree_; }
  /// The ring-switch target's degree, `k_i T_l`: what the outer scan reads.
  int GetOuterSmallDegree() const { return inner_rank_ * lane_degree_; }
  int GetNumSlots() const { return num_slots_; }

  /// `LocateSlot`'s primary address of (part, block, lane).
  int PrimarySlot(int part, int block, int lane) const {
    return static_cast<int>(BitReverseInt(block * outer_rank_ + part,
                                          log_outer_ + log_inner_)) *
               lane_degree_ +
           lane;
  }
  /// The tower coordinate `flat = (t k_i + j) k_o + i` slot `s` carries
  /// after the CtS'.
  int TowerIndexOfSlot(int slot) const {
    return static_cast<int>(BitReverseInt(slot, log_slots_));
  }

  /**
   * @brief Compile a forward (slots at the primary addresses -> the nested
   * SinC element) under `name`, at `level` on `context`.
   *
   * @param premap when non-null, a lane-preserving block permutation folded
   *        into the input side (Doing.md 1.5bx): `(*premap)[b]` is the
   *        primary block that the caller's block `b` feeds. It composes
   *        outermost, after the column scaling, and costs whatever
   *        diagonals it adds to the first phase (441 against 255 for the
   *        leg's Q transport at the Llama shape).
   * @param fold_premap fold the premap into the first phase (above), or --
   *        false -- run it as its own leading permutation phase, one more
   *        level. A premap that moves bits between the block field and the
   *        part field cannot fold (the leg's K transport fills the whole
   *        stride-T_l lattice, 2048 diagonals) and stands alone at 479.
   */
  void PrepareForward(ConstContextPtr<word> context, const std::string &name,
                      int level, const std::vector<int> *premap = nullptr,
                      const Phases &phases = Phases(),
                      bool fold_premap = true);
  /// The tower CoeffToSlot at `level` with `cts_const` folded in (a
  /// bootstrap's is `MaxNumSlots() * GetCtSConst()`).
  void PrepareCtS(ConstContextPtr<word> context, int level, double cts_const,
                  const Phases &phases = Phases());
  /// The lane prefix at `level`; `constant` folds a scalar into its matrix
  /// and `pt_scale` (> 0) sets its plaintext scale, else the level's rescale
  /// product, which preserves the input's scale.
  void PreparePrefix(ConstContextPtr<word> context, int level,
                     double constant = 1.0, double pt_scale = -1.0,
                     const Phases &phases = Phases());

  bool HasForward(const std::string &name) const;
  int GetForwardLevel(const std::string &name) const;
  int GetForwardNumLevels(const std::string &name) const;
  const std::vector<int> &GetForwardDiagonals(const std::string &name) const;
  int GetCtSLevel() const { return cts_.level; }
  int GetCtSNumLevels() const {
    return NumLevels(cts_) + static_cast<int>(cts_thin_levels_.size());
  }
  const std::vector<int> &GetCtSDiagonals() const { return cts_.diagonals; }
  int GetPrefixLevel() const { return prefix_.level; }
  int GetPrefixNumLevels() const { return NumLevels(prefix_); }
  const std::vector<int> &GetPrefixDiagonals() const {
    return prefix_.diagonals;
  }

  /// Every compiled chain's rotations, each on its own context's ring.
  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;
  void AddForwardRotations(EvkRequest &req, bool min_ks = false) const;
  void AddCtSRotations(EvkRequest &req, bool min_ks = false) const;
  void AddPrefixRotations(EvkRequest &req, bool min_ks = false) const;

  /// Slots at the primary addresses -> the nested SinC element, at
  /// `GetForwardLevel(name) - GetForwardNumLevels(name)`.
  void Forward(const std::string &name, Ct &res, const Ct &input,
               const EvkMap<word> &evk_map) const;
  /// An element -> its tower coordinates in the slots.
  void EvaluateCtS(Ct &res, const Ct &input, const EvkMap<word> &evk_map) const;
  /// Tower coordinates in slots -> the message at the primary addresses.
  void Prefix(Ct &res, const Ct &input, const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
