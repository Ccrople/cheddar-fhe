#pragma once

#include <vector>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/LinearTransform.h"

namespace cheddar {

/**
 * @brief An arbitrary permutation of the slot index, as one `LinearTransform`.
 *
 * ## Why this exists
 *
 * [SYLPH] section 3.2 chooses its packing so that the bit reversal on the way
 * to coefficient encoding and the partial bit reversal on the way to SinC land
 * the data where the next algorithm wants it -- "effectively removing the need
 * for most conversions throughout the entire process". It does not remove all
 * of them: **the transpose survives**, and [SYLPH] does it with Halevi-Shoup
 * and BSGS. This is that.
 *
 * It is needed because the two products want opposite things of the same
 * index. The Bae PC-MM ladder -- ring switch, `ModDecomp`, product at ring
 * degree 256, `ModPack`, inverse ring switch -- writes coefficient
 * `p = j + 16c + 256*s`, so the **token** is the coordinate of the smallest
 * ring and therefore owns the *high* coefficient bits, with the channel split
 * across the two low nibbles. The batch CC-MM reads `Vec^d_k`, whose block
 * index is `p mod d` and whose lane index is `p / d`, so it wants the token in
 * the *middle*: `[lane | row | column]`. No assignment of the channel bits
 * fixes that, because the token's position is not an assignment -- it is where
 * the ladder puts the ring it lives in.
 *
 * ## What it costs, and why a permutation is not automatically ruinous
 *
 * A permutation's striped form has one diagonal per distinct value of
 * `s - perm[s]`, so the cost is set by how structured the map is, not by how
 * many slots move. The two that matter here:
 *
 *  - a square `n x n` transpose has `2n - 1` distinct offsets, **all multiples
 *    of `n - 1`** -- so 255 diagonals at stride 127 for `n = 128`, BSGS 16x16,
 *    about 31 rotations;
 *  - swapping an `a`-bit field with an adjacent `b`-bit one -- a `2^a x 2^b`
 *    transpose -- has `2^(a+b)` offsets at stride 1, so 2048 for `[4 | 7]`,
 *    BSGS 64x32, about 95 rotations.
 *
 * Both are one level. What makes the second one affordable is a window shift:
 * the raw offsets straddle zero, so reduced mod the slot count they span
 * almost the whole ring and `LinearTransform::DetermineStride` would demand
 * `bs * gs >= num_slots`. Building the matrix for the permutation composed
 * with a rotation, and undoing that rotation afterwards with one `HRot`,
 * collapses it to the true spread. This class finds the rotation itself, and
 * takes it only when it wins -- a shift can destroy a common stride, and the
 * transpose above would rather keep its 127.
 *
 * It is deliberately **not** `LinearTransform::pre_rotation`. That parameter
 * reduces the offsets and then rotates by the reduced amount, so it assumes
 * the input already arrives rotated -- free inside `EvalSpecialFFT`, where the
 * previous phase produced exactly that rotation, and garbage for a standalone
 * transform. Measured, not assumed.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SlotPermute {
 private:
  using Ct = Ciphertext<word>;

 public:
  /**
   * @brief Build the permutation at `level`; it returns one level below.
   *
   * @param perm a bijection on `[0, perm.size())`: output slot `perm[s]`
   *        receives input slot `s`. The size must be the ciphertext's slot
   *        count and a power of two.
   * @param level the level the transform is compiled at
   */
  SlotPermute(ConstContextPtr<word> context, const std::vector<int> &perm,
              int level);

  // disable copying (or moving also)
  SlotPermute(const SlotPermute &) = delete;
  SlotPermute &operator=(const SlotPermute &) = delete;

  /// Distinct values of `(s - perm[s]) mod n`, which is the plaintext count.
  int GetNumDiagonals() const { return num_diag_; }
  int GetBS() const { return bs_; }
  int GetGS() const { return gs_; }
  /// The window rotation chosen so the offsets do not straddle zero, undone
  /// afterwards with one HRot. Zero when the raw offsets are already tight --
  /// which is the case whenever they share a large stride.
  int GetShift() const { return shift_; }
  /// The common stride of the offsets after `pre_rotation`.
  int GetStride() const { return stride_; }

  void AddRequiredRotations(EvkRequest &req) const;

  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const EvkMap<word> &evk_map) const;

 private:
  int num_slots_;
  int level_;
  int num_diag_ = 0;
  int bs_ = 1;
  int gs_ = 1;
  int shift_ = 0;
  int stride_ = 1;
  std::vector<LinearTransform<word>> lt_;
};

/**
 * @brief The permutation that swaps two adjacent bit fields of the slot index.
 *
 * Slot `s = high * 2^(a+b) + A * 2^b + B` with `A` an `a`-bit field and `B` a
 * `b`-bit one becomes `high * 2^(a+b) + B * 2^a + A`. `high` is untouched, so
 * the map is the same inside every block of `2^(a+b)` slots.
 *
 * This is the transpose the CC-MM's operands need: the PC-MM ladder leaves
 * `[channel-nibble | token]` in the low slot bits and the CC-MM wants
 * `[token | head-nibble]`.
 *
 * @param num_slots the ciphertext's slot count
 * @param low_bits `b`, the width of the field that ends up on top
 * @param high_bits `a`, the width of the field that ends up at the bottom
 */
std::vector<int> SwapAdjacentFields(int num_slots, int low_bits,
                                    int high_bits);

}  // namespace cheddar
