#pragma once

#include <vector>

#include "core/BatchCcmm.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/RingSwitch.h"

namespace cheddar {

/**
 * @brief [SYLPH] table 4's CC-MM row as the block sees it: a batch of `d x d`
 * ciphertext-ciphertext products whose operands arrive as **big-ring**
 * ciphertexts, descend through the ring switch, and come back.
 *
 *     8 x 65536  --RingSwitch-->  128 x 4096  --[KANG] Alg. 4-->
 *                128 x 4096  --SwitchBack-->  8 x 65536
 *
 * `RingSwitchHandler` and `BatchCcmmHandler` each have their own test and are
 * each correct. What this adds is the part that is neither of theirs: **the
 * index bookkeeping**, i.e. which big ciphertext and which slot of it a matrix
 * entry has to occupy for the product to find it. That is stated once, in
 * `SwitchedCcmmLayout`, and everything upstream is built against it.
 *
 * ## The layout, derived rather than chosen
 *
 * Three maps compose, and none of them is free to be different.
 *
 * **1. Slots to SinC** (`EvalSpecialFFT::EvaluateSlotToSinC`). Writing the
 * slot index of the big ring as `s = [A | r]` with `r` the low
 * `log2(sub_degree/2)` bits, the SinC message index is `[BitRev_p(A) | r]`:
 * the block index is bit-reversed and the **lane index is the low bits of the
 * slot index, untouched**.
 *
 * **2. The ring switch** ([SYLPH] section 3.3, measured in
 * `RingSwitchTest.CarriesTheSinCEncodingToTheSmallRing`). Big block
 * `i = j + rank * i'` lands at block `i'` of small ciphertext `j`, lanes
 * untouched. So the big SinC message index reads, most significant first,
 *
 *     [ i' : small block | j : which small ciphertext | t : lane ]
 *
 * **3. [KANG] Algorithm 4** (`BatchCcmm.h`, and `BatchCcmmTest`'s own
 * encoder). Small ciphertext `j` carries **column** `j`; block `i'` of its
 * SinC message carries **row** `i'`; lane `t` selects which of the `k/2`
 * independent matrices. The operand is `d` ciphertexts and the ring switch
 * supplies `rank` of them per big ciphertext, so a column index splits as
 * `column = big_ciphertext * rank + j`.
 *
 * Composing, entry `(row, column, lane)` of the operand sits in big ciphertext
 * `column / rank` at slot
 *
 *     [ BitRev(column mod rank) | BitRev(row) | lane ]
 *
 * with field widths `log2(rank) + log2(d) + log2(sub_degree/2)`, which is
 * exactly `log2(degree/2)`. `SwitchedCcmmLayout::Locate` is that line and the
 * test checks it entrywise rather than by norm, because every plausible
 * mistake here -- a transposed operand, lanes swapped, blocks laid out
 * contiguously instead of interleaved -- keeps the magnitude about right.
 *
 * ## Why `sub_degree = 32` for Llama-3
 *
 * At the small ring `d = small_degree / sub_degree` is the matrix width and
 * `sub_degree / 2` the lane count. Llama-3's per-head attention product is
 * exactly `128 x 128` -- T tokens by head_dim -- so `sub_degree = 32` gives
 * `d = 128` and **16 lanes**, which is 16 heads per call and two calls for 32
 * heads. Per call each operand is 8 big ciphertexts, and the arithmetic is
 * `128 * 128 * 16 = 8 * 32768`: the real half of eight big ciphertexts,
 * exactly, with no waste.
 *
 * `sub_degree = 128` ([SYLPH]'s own instance, 32 blocks and 64 lanes) makes
 * the matrices `32 x 32`, so a `128 x 128` product has to be blocked into
 * `4 x 4` and summed -- 64 block-products per head against one.
 *
 * ## Level, and keys
 *
 * One level, spent inside Algorithm 4: operands at `level`, result at
 * `level - 1`. The ring switches are key switches and cost none. The keys are
 * a forward and an inverse ring-switching key on the switching Context, and
 * `d` automorphism keys plus a multiplication key on the small one -- about
 * 16 MiB for all 128 at degree 4096, against 0.94 GiB for the same `d` at
 * 65536. That ratio is the ring switch paying for itself.
 *
 * ## The three Contexts
 *
 * The operands belong to whichever Context produced them -- in this pipeline
 * the block's bootstrappable one -- and are handed to the *switching* Context,
 * whose `alpha` is small enough that a switching key published at `PQ` still
 * fits the small ring's security budget. That crossing moves no words: the two
 * share their primes at every level, which `RingSwitchHandler`'s constructor
 * verifies. They must also share a **secret**; see
 * `UserInterface(context, secret_coeffs)`.
 *
 * @tparam word uint32_t or uint64_t
 */
struct SwitchedCcmmLayout {
  int big_degree = 0;
  int small_degree = 0;
  int sub_degree = 0;

  int rank = 0;      ///< big_degree / small_degree, ciphertexts per switch
  int dim = 0;       ///< d = small_degree / sub_degree, the matrix width
  int lanes = 0;     ///< sub_degree / 2, independent products per call
  int num_cts = 0;   ///< d / rank, big ciphertexts per operand

  SwitchedCcmmLayout() = default;
  SwitchedCcmmLayout(int big_degree, int small_degree, int sub_degree);

  /// Big-ring slot count.
  int NumSlots() const { return big_degree / 2; }

  /**
   * @brief Where entry `(row, column, lane)` of an operand lives.
   *
   * @param ct receives the big ciphertext index, in `[0, num_cts)`
   * @param slot receives the slot index within it, in `[0, big_degree/2)`
   */
  void Locate(int row, int column, int lane, int &ct, int &slot) const;

  /**
   * @brief The same entry addressed by its **SinC message index** instead.
   *
   * `Locate` gives the slot the value has to occupy *before*
   * `EvaluateSlotToSinC` runs, which is what the pipeline needs. This gives
   * the index it occupies *after*, which is what `Encoder::EncodeSinC` and
   * `DecodeSinC` are indexed by -- so it is what a host-side reference reads
   * and writes. The two are tied by the transform's own identity, and the
   * test checks that rather than trusting it.
   */
  void LocateSinC(int row, int column, int lane, int &ct, int &index) const;

  /// The inverse of Locate. Returns false if the slot carries no entry, which
  /// cannot happen for a power-of-two layout but is checked rather than
  /// assumed.
  bool Position(int ct, int slot, int &row, int &column, int &lane) const;
};

template <typename word>
class SwitchedCcmmHandler {
 private:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;

 public:
  /**
   * @param switch_ctx the big-ring Context that hosts the ring-switching key
   *        switch; its `alpha` must be small enough for the small ring's
   *        budget, and its primes must match those of the Context the
   *        operands come from
   * @param small_ctx the small ring the product runs in
   * @param sub_degree k
   */
  SwitchedCcmmHandler(ConstContextPtr<word> switch_ctx,
                      ConstContextPtr<word> small_ctx, int sub_degree);

  // disable copying (or moving also)
  SwitchedCcmmHandler(const SwitchedCcmmHandler &) = delete;
  SwitchedCcmmHandler &operator=(const SwitchedCcmmHandler &) = delete;

  const SwitchedCcmmLayout &GetLayout() const { return layout_; }

  /// Automorphism indices the three CMT calls need, on the small ring.
  std::vector<int> SmallRotationIndices() const {
    return ccmm_.RotationIndices(layout_.sub_degree);
  }

  /**
   * @brief `res = lhs * rhs`, one `d x d` product per lane, one level.
   *
   * @param res output, `num_cts` big ciphertexts, one level below the inputs
   * @param lhs `num_cts` big ciphertexts in the layout above
   * @param rhs likewise
   * @param swk the forward ring-switching key, from
   *        `UserInterface::PrepareRingSwitchKey` on the switching Context
   * @param inv_swk the inverse one
   * @param small_evk the small ring's automorphism and multiplication keys
   */
  void Multiply(std::vector<Ct> &res, const std::vector<Ct> &lhs,
                const std::vector<Ct> &rhs, const Evk &swk, const Evk &inv_swk,
                const EvkMap<word> &small_evk) const;

 private:
  // One operand: num_cts big ciphertexts -> d small ones, in column order.
  void Descend(std::vector<Ct> &res, const std::vector<Ct> &operand,
               const Evk &swk) const;

  ConstContextPtr<word> switch_ctx_;
  ConstContextPtr<word> small_ctx_;
  SwitchedCcmmLayout layout_;
  RingSwitchHandler<word> switcher_;
  BatchCcmmHandler<word> ccmm_;
};

}  // namespace cheddar
