#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/CiSwitchedCcmm.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/LinearTransform.h"
#include "extension/StripedMatrix.h"
#include "extension/SylphSchedule.h"

namespace cheddar {

/**
 * @brief The seam: the attention output's chain layout carried to the banded
 * half-density coefficient image the next projection reads.
 *
 * ## Why there is a seam at all
 *
 * `CiSinCAttention` hands back the attention output in the CC-MM chain's own
 * layout, where an entry is addressed by `(row, column, lane)` and every entry
 * appears at ONE slot. `CoeffLinearLeg` reads a banded half-density
 * coefficient image, where a model channel appears at TWO coefficient
 * addresses -- the live one and its shifted duplicate -- because that is the
 * unique fixed point of `ModPack`'s band (Doing.md 1.5dk). The chain's operand
 * form carries the duplicates and its OUTPUT form does not, measured: primary
 * against `copy_slot` differed by 0.359782 over 487680 pairs against
 * `|output| <= 0.289119`, larger than the output itself.
 *
 * So the O projection cannot take the attention output as it stands, and what
 * closes the gap is two slot transforms of known size -- counted on the host
 * in a minute after the device measurement had cost thirteen.
 *
 * ## What the three transforms are
 *
 * **T1 is a BIT PERMUTATION of the slot index.** The chain address
 * `rev4(col) * 4096 + rev7(row) * 32 + lane` and the block address
 * `row + 128 * (rev4(col) * 32 + rev5(lh))` agree on the column field and
 * differ by six transpositions -- (11,0) (10,1) (9,2) (8,3) (6,5) (7,4) --
 * plus a shift of 128 on the upper half, because the transposition puts
 * `half` at destination bit 7 where the packing wants a zero.
 *
 * That is why the one-shot count is 486: a transposition contributes offsets
 * `{0, +-(2^i - 2^j)}`, six give `3^6 = 729`, and the fixed source bit 4
 * collapses one factor to two -- `2 * 3^5 = 486`, measured to the digit. Run
 * as one transform it is 2.9 GB of plaintext diagonals per half and it set the
 * whole layer's memory ceiling. Split across stages the count multiplies out
 * far less; searched on the host over every ordered partition, minimising
 * rotation keys:
 *
 *     one stage    486 diagonals, 192 keys   (BSGS 128x64)
 *     two stages   165 diagonals, 100 keys
 *     three         60 diagonals,  56 keys   <-- what this builds
 *
 * *Never build a bit-permutation slot map as one transform: its diagonal count
 * is a product over the bits it moves, so it falls geometrically per stage
 * while levels rise by one.*
 *
 * **The TOKEN MAP is a bit-reversed decrement, not a slot rotation.** The
 * banded convention is a statement about COEFFICIENT positions -- coefficient
 * `p * rank + I` carries `comp_I[p] + comp_{rank-I}[p+1]` -- and `SlotToCoeff`
 * sends slot `t + 128 c` to coefficient `rev7(t) * 512 + rev9(c)`, so a step of
 * one in `p` is a step of one in `rev7(t)`, NOT in `t`. Shifting by one slot
 * instead reads back through `BandedComponents` at max error 38.56 against
 * `|v| <= 4.16`; `rev7(rev7(t) - 1)` gives exactly zero. A decrement in
 * bit-reversed order is a carry, so it has just 7 distinct offsets.
 *
 * **T2 creates the duplicates**, one block-permutation copy-add on clean
 * slots. **Component zero has no partner**: `rank - 0` wraps to channel 0,
 * which is even and so live, and the banded recomposition excludes `i == 0` --
 * so taking the formula literally writes a duplicate over a live value, and
 * that alone held the live half at 2.54 while T1 was exact at 2.9e-05.
 *
 * ## Two things that are derived, not written down
 *
 * **The levels come from `SylphSchedule::GetStCLevel()`.** A ladder of level
 * constants is only correct for the slack it was typed under: the slack was
 * once raised from 9 to 12 for a memory reason, which moved
 * `GetStCStartLevel()` from 10 to 7 and put `SlotToCoeff`'s phases inside
 * `ci16_35`'s `num_accum == 1` zone (levels 0..6), returning coefficients at
 * 4.99e+47. Stacking the seam on whatever StC reports cannot come apart that
 * way, and the constructor asserts the zone.
 *
 * **The window convention and its closing rotation.** `LinearTransform`'s
 * `DetermineStride` cannot see negative offsets, so each matrix is compiled
 * with `pre_rotation = -w` for the window `w` that minimises the BSGS span,
 * and the caller owes one rotation by `w` afterwards. `Apply` does it.
 *
 * ## Memory: T1 is per half and deliberately not in the constructor
 *
 * The two halves' T1s are never used together and each is the largest object
 * the seam owns, so `PrepareHalf` builds one and `Apply` consumes it. The
 * price is that its rotation keys can only be requested after it exists --
 * hence `AddHalfRotations` between the two calls, rather than one
 * `AddRequiredRotations` in the constructor.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiLlamaSeam {
 private:
  using Ct = Ciphertext<word>;

 public:
  struct Config {
    //! THE PROJECTION'S rank, which is NOT `layout.rank`. The chain layout's
    //! rank is its columns per ciphertext (16 at the Llama alignment); this is
    //! the module rank the banded convention is stated against (512), so it is
    //! what `rank - I` means and what the channel index is bit-reversed over.
    //! Conflating the two is the kind of mistake that produces a layer which
    //! decrypts cleanly and is wrong.
    int proj_rank = 512;
    //! The three stages T1 is split into, as transpositions of slot-index
    //! bits. The default is the searched optimum above; a caller changing it
    //! changes only how many diagonals and levels the same map costs.
    std::vector<std::vector<std::pair<int, int>>> t1_stages = {
        {{11, 0}}, {{10, 1}}, {{9, 2}, {8, 3}, {6, 5}, {7, 4}}};
    bool verbose = false;
  };

  /**
   * @param context the Context the seam evaluates in -- the FFN's, whose
   *        slack is what `SylphSchedule` reports and what the levels derive
   *        from
   * @param layout the chain layout the attention output arrives in
   * @param stc_level `SylphSchedule::GetStCLevel()` on that Context
   * @param cfg the staging
   */
  CiLlamaSeam(ConstContextPtr<word> context, const CiSwitchedCcmmLayout &layout,
              int stc_level, const Config &cfg = Config{});

  CiLlamaSeam(const CiLlamaSeam &) = delete;
  CiLlamaSeam &operator=(const CiLlamaSeam &) = delete;

  //! The rotations T2 and the token map need, and their closing rotations.
  //! T1's are `AddHalfRotations`, because T1 does not exist yet.
  void AddRequiredRotations(EvkRequest &req) const;

  //! Build `half`'s T1 stages, replacing whatever `PrepareHalf` built before.
  void PrepareHalf(int half);

  //! The rotations the currently prepared half's stages need.
  void AddHalfRotations(EvkRequest &req) const;

  //! Drop the prepared half's plaintexts. `PrepareHalf` does it implicitly.
  void DropHalf();

  /**
   * @brief Carry one booted chain-layout ciphertext to its banded coefficient
   * image, for the half `PrepareHalf` last built.
   *
   * @param res the banded half-density coefficient ciphertext
   * @param booted one of the attention output's ciphertexts, bootstrapped
   * @param sched the schedule whose `ToCoeff` closes the turn
   */
  void Apply(Ct &res, const Ct &booted, SylphSchedule<word> &sched,
             const EvkMap<word> &evk, bool min_ks = false) const;

  //! The level the seam brings its input down to before the first stage.
  int GetInputLevel() const { return t1_top_; }

  //! Which half `PrepareHalf` last built, or -1.
  int GetPreparedHalf() const { return prepared_half_; }

 private:
  //! One compiled transform plus the rotation its window convention owes.
  struct Stage {
    std::unique_ptr<LinearTransform<word>> transform;
    int back = 0;
    int level = 0;
  };

  //! The plaintext scale that rescales a transform's output onto `level - 1`.
  double PtScale(int level) const;
  //! The window minimising the BSGS span, and the span it needs.
  int BestWindow(const StripedMatrix &m, int *need) const;
  //! Compile `m` at `level` with the window convention.
  Stage Compile(const StripedMatrix &m, int level) const;
  //! Evaluate one stage and pay its closing rotation.
  void RunStage(Ct &res, const Ct &in, const Stage &st,
                const EvkMap<word> &evk) const;

  ConstContextPtr<word> context_;
  CiSwitchedCcmmLayout layout_;
  Config cfg_;
  int degree_ = 0;
  //! log2 of the chain layout's columns per ciphertext, and of the projection
  //! rank. Kept apart on purpose; see `Config::proj_rank`.
  int log_cols_ = 0;
  int log_proj_rank_ = 0;
  int log_dim_ = 0;

  // The derived ladder. T1's stages sit above the token map, which sits above
  // T2, which sits two levels above SlotToCoeff.
  int t2_level_ = 0;
  int tokmap_level_ = 0;
  int t1_top_ = 0;

  Stage t2_, tokmap_;
  std::vector<Stage> t1_;
  int prepared_half_ = -1;
};

}  // namespace cheddar
