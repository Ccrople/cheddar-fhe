#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/CtAxisExchange.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "core/SwitchedCcmm.h"
#include "extension/BootContext.h"
#include "extension/LinearTransform.h"
#include "extension/SlotPermute.h"

namespace cheddar {

/**
 * @brief The two attention products as the block sees them: **slots in, slots
 * out**.
 *
 * ## What this is for
 *
 * `LlamaBlock` alternates between the slot domain, where the non-linear
 * operators live, and a low-modulus domain where the matrix products live. For
 * the seven plaintext-weight projections that second domain is *coefficient*
 * encoding and the route is `SylphSchedule`'s `ToCoeff` / `ToSlot`. For the
 * two ciphertext-ciphertext products -- `Q K^T` and `P V` -- it is **SinC**
 * encoding at a small ring, and the route is different at both ends:
 *
 *     slots  --permute--> --SlotToSinC--> --LevelDown-->  SinC @ product level
 *            --RingSwitch--> [KANG] Alg. 4 --SwitchBack-->  SinC @ 0
 *            --HalfBoot--> --StC prefix-->                 slots
 *
 * Every piece of that is built and measured elsewhere. What this class is, is
 * the statement of **which piece each operand needs**, which is not the same
 * for any two of the four:
 *
 * | operand | permutation | cross-ciphertext | why |
 * |---|---|---|---|
 * | Q | `[4\|7]` | none | its column axis is the channel, which the projection places |
 * | K | `[8\|4]@3` | yes | its column axis is the KEY, which no weight matrix can move |
 * | P | **none** | none | SoftMax leaves it exactly where the next product wants it |
 * | V | `[4\|7]` | yes | column is the channel again, but GQA needs 2 ciphertexts to become 8 |
 *
 * P costing nothing is the load-bearing one. The score product leaves S at
 * `Locate(BitRev7(query), BitRev7(key), head)`; `HalfBoot` plus the prefix
 * return exactly that to slots; SoftMax is elementwise on the main track and
 * reduces along the key axis, so it leaves the layout alone; and for the
 * second product P is the left operand with `row = BitRev7(query)` and
 * contraction `BitRev7(key)`, whose `Locate` is the same ciphertext and the
 * same slot. That is what naming K's column axis `BitRev7(key)` bought, and it
 * matters because the leg between SoftMax and the Values product is the one
 * with no levels to spare.
 *
 * ## The transforms, and why there are only four
 *
 * Q, K and V all enter at the same level, so their first swap is the same
 * object: `[4|4]@3` is the first half of both `[4|7]` and `[8|4]@3`. Only the
 * second half differs -- `[4|3]` for Q and V, `[4|4]@7` for K -- and K and V
 * share one `CtAxisExchange`, because 2-ciphertexts-to-8 at offset 0 is the
 * same move for both. So:
 *
 *     swap_a       [4|4]@3       31 diagonals    Q, K, V
 *     swap_qv_b    [4|3]        127 diagonals    Q, V
 *     swap_k_b     [4|4]@7       31 diagonals    K
 *     exchange     3 bits @ 0    8 masks         K, V
 *
 * Q reaches the SinC transform one level above K and V, since it skips the
 * exchange; it is dropped into place with a `LevelDown`, which costs a level
 * that turn B has and no time at all.
 *
 * ## The prefix carries Canonicalise
 *
 * `SylphSchedule::Canonicalise` is one constant multiply and one rescale at
 * the level `HalfBoot` lands on, and the StC prefix is one `LinearTransform`
 * at the same level. Folding the first into the second's diagonals makes the
 * prefix free -- the schedule was spending that level anyway -- and restores
 * the magnitude `HalfBoot` divided out, which `SinCTransformTest` measured as
 * about five bits. The two products need different magnitudes, so there are
 * two prefixes; they are 31 diagonals each and the cheapest thing here.
 *
 * ## What this class does NOT do
 *
 * The attention output comes back at `Locate(BitRev7(query), BitRev7(c),
 * head)`, which in slot terms is `[c>>3 | query | head]` -- the layout the Q
 * operand had **after** its own `[4|7]` swap, so the block closes on itself.
 * Returning it to the block's packing for the O projection is the inverse of
 * that swap, which is an ordinary `SlotPermute` the caller owns; it is not
 * here because it belongs to the same level budget as RoPE and the block
 * should be able to place it.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SinCAttention {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

 public:
  struct Config {
    int num_tokens = 128;   //!< T
    int head_dim = 128;     //!< channels per head
    int lanes = 16;         //!< heads per call group; must be sub_degree / 2
    int gqa_group = 4;      //!< query heads per KV head
    int sub_degree = 32;    //!< k
    int sinc_phases = 3;    //!< levels SlotToSinC spends
    int product_level = 1;  //!< the CC-MM runs here and rescales to 0

    //! Where the first swap is compiled. Q, K and V all enter here; the
    //! second swap is one below and the exchange one below that.
    int swap_level = 15;
    //! Where SlotToSinC is compiled. It is NOT `swap_level - 3`, and the two
    //! cannot be tied together: Q, K and V arrive from RoPE near the top of
    //! the slot leg, while P arrives from SoftMax at the bottom of it, and one
    //! transform has to serve both. So it is compiled where P can reach and
    //! K and V are dropped into it with a `LevelDown`, which is free. The
    //! constraint is `sinc_level <= swap_level - 3`, plus the floor every
    //! standalone LinearTransform has -- 8 on sylphflow16_35, swept in 1.5ai
    //! and a property of the ladder, not of the transform.
    int sinc_level = 12;
    //! Where HalfBoot lands, which is where the prefix is compiled.
    int prefix_level = 19;

    //! `BootContext::GetStCInputScale()`, the scale HalfBoot hands back. Zero
    //! means do not canonicalise in the prefix, and it then preserves whatever
    //! scale it is given -- correct, but it leaves a level and about five bits
    //! for someone else to spend.
    double halfboot_scale = 0.0;
    //! The constant each product's prefix multiplies by, which is the WHOLE of
    //! what `SylphSchedule::Canonicalise` would apply -- not just its
    //! `magnitude` argument.
    //!
    //! Canonicalise's own constant is `2^log_message_ratio * magnitude`,
    //! because on the coefficient leg `ToCoeff` divided the message by
    //! `2^log_message_ratio` on the way down and the two are exact inverses.
    //! **This leg has no `ToCoeff`**: `SlotToSinC` is a plain transform and
    //! scales nothing, so the operands reach the product at operator magnitude
    //! and the product's output is whatever they make of it. Sizing that so
    //! the bootstrap can carry it is the caller's calibration -- the same job
    //! `size_q` and `size_k` already do on the other leg -- and so is deciding
    //! what comes back. Hence the whole constant, not half of it.
    double score_magnitude = 1.0;
    double value_magnitude = 1.0;

    //! Print each stage's level and scale. Three transforms, a ring descent
    //! and a bootstrap compose here and they all fail the same way, so the
    //! ledger is worth having.
    bool verbose = false;
  };

  /** @brief The keys one call needs, on both rings. */
  struct Keys {
    const EvkMap<word> *big = nullptr;    //!< the block ring
    const EvkMap<word> *small = nullptr;  //!< the product ring
    const Evk *ring_switch = nullptr;
    const Evk *inverse_ring_switch = nullptr;
  };

  /**
   * @param boot the block's bootstrappable Context; the transforms and the
   *        bootstrap both live here
   * @param switch_ctx the ring-switching Context, sharing `boot`'s primes at
   *        the product level and `boot`'s SECRET
   * @param small_ctx the product ring
   */
  SinCAttention(std::shared_ptr<const BootContext<word>> boot,
                ConstContextPtr<word> switch_ctx,
                ConstContextPtr<word> small_ctx, const Config &cfg);

  // disable copying (or moving also)
  SinCAttention(const SinCAttention &) = delete;
  SinCAttention &operator=(const SinCAttention &) = delete;

  const SwitchedCcmmLayout &GetLayout() const { return ccmm_.GetLayout(); }
  /// Ciphertexts one operand occupies, and one result.
  int GetNumCiphertexts() const { return ccmm_.GetLayout().num_cts; }
  /// Where the two operands meet SlotToSinC.
  int GetSinCLevel() const { return cfg_.sinc_level; }
  /// Where a product's result comes back in slots.
  int GetOutputLevel() const { return cfg_.prefix_level - 1; }

  /**
   * @brief Recompile the two prefixes with new constants.
   *
   * The constant the whole chain leaves is a property of the parameter set and
   * **has to be measured**, not derived: `HalfBoot` declares its output at
   * `eval_mod_->end_scale_` and says so in its own comment -- "what the
   * remaining constant is gets measured rather than derived through
   * cts_const_, stc_const_ and q0". Measured here on `sylphflow16_35` it is
   * **0.02985**, which is `2^-5.07` against the `2^-log_message_ratio = 2^-5`
   * the design intends; the 5% is the part that does not cancel because this
   * leg has no `ToCoeff` to cancel against.
   *
   * So the sequence is: run once with 1, read the constant off the result,
   * and set its inverse. Only the two prefixes are rebuilt -- 31 diagonals
   * each -- and **no new rotation keys are needed**, because a constant does
   * not change a transform's diagonal structure.
   */
  void SetMagnitudes(double score_magnitude, double value_magnitude);

  /// Rotations on the block ring: the swaps, the exchange, SlotToSinC, the
  /// prefix and its window. The bootstrap's own are the caller's business.
  void AddRequiredRotations(EvkRequest &req) const;
  /// Rotations on the product ring.
  std::vector<int> SmallRotationIndices() const {
    return ccmm_.SmallRotationIndices();
  }

  /**
   * @brief `res = Q K^T`, one `T x T` per lane, slots in and slots out.
   *
   * @param res output, `num_cts` slot-encoded ciphertexts at `GetOutputLevel()`
   * @param q the rotated queries in the block's packing, `num_cts` of them,
   *        at `Config::swap_level`
   * @param k the rotated keys in K's packing, **2** of them, at the same level
   * @param shift added to each result in SinC form at level 0, one value per
   *        ciphertext and slot, or empty for none. It goes in before the
   *        bootstrap because that is the only place with room for its
   *        magnitude, so it is `shift / scale` in the caller's terms.
   */
  void Scores(std::vector<Ct> &res, const std::vector<Ct> &q,
              const std::vector<Ct> &k, const Keys &keys,
              const std::vector<std::vector<Complex>> &shift = {}) const;

  /**
   * @brief The same, over operands that are a slice of a larger tensor.
   *
   * A ciphertext is movable and not copyable, so a caller holding all 32
   * heads cannot hand over one lane group by slicing the vector. It hands
   * over pointers instead; nothing is copied and nothing is moved out of the
   * caller's tensor, which still owns every ciphertext when the call returns.
   */
  void Scores(std::vector<Ct> &res, const std::vector<const Ct *> &q,
              const std::vector<const Ct *> &k, const Keys &keys,
              const std::vector<std::vector<Complex>> &shift = {}) const;

  /**
   * @brief `res = P V`, slots in and slots out.
   *
   * @param res output, `num_cts` slot-encoded ciphertexts at `GetOutputLevel()`
   * @param p SoftMax's output, `num_cts` of them, at or above
   *        `GetSinCLevel()`
   * @param v the values in V's packing, **2** of them, at `Config::swap_level`
   */
  void Values(std::vector<Ct> &res, const std::vector<Ct> &p,
              const std::vector<Ct> &v, const Keys &keys) const;

  /** @brief The same, over operands that are a slice of a larger tensor. */
  void Values(std::vector<Ct> &res, const std::vector<const Ct *> &p,
              const std::vector<const Ct *> &v, const Keys &keys) const;

 private:
  // One operand's descent from slots to a SinC ciphertext at the product
  // level. `permute` picks which second swap, or none.
  enum class Leg { kQuery, kKey, kValue, kProb };
  void Descend(std::vector<Ct> &res, const std::vector<const Ct *> &x, Leg leg,
               const EvkMap<word> &evk) const;
  // The product, then the shift, then HalfBoot and the prefix.
  void Ascend(std::vector<Ct> &res, std::vector<Ct> &product,
              const LinearTransform<word> &prefix,
              const std::vector<std::vector<Complex>> &shift,
              const EvkMap<word> &evk) const;

  std::shared_ptr<const BootContext<word>> boot_;
  ConstContextPtr<word> switch_ctx_;
  Config cfg_;
  int num_slots_;
  int prefix_window_;

  std::unique_ptr<SlotPermute<word>> swap_a_, swap_qv_b_, swap_k_b_;
  std::unique_ptr<CtAxisExchange<word>> exchange_;
  void BuildPrefixes();

  std::vector<LinearTransform<word>> prefix_;  // [0] scores, [1] values
  SwitchedCcmmHandler<word> ccmm_;
  mutable std::vector<Pt> shift_pt_;
  mutable std::vector<std::vector<Complex>> shift_cache_;
};

}  // namespace cheddar
