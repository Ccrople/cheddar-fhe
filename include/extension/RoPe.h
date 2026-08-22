#pragma once

#include <climits>
#include <map>
#include <utility>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"

namespace cheddar {

/**
 * @brief RoPE, the rotary positional embedding, as [SYLPH] section 3.2 places
 * it: "a position-wise plaintext-ciphertext multiplication", one level.
 *
 * ## The convention
 *
 * Llama-3's weights come from Hugging Face, whose `rotate_half` pairs channel
 * `j` with `j + D/2` rather than `2j` with `2j+1`:
 *
 *     q'[j]       = q[j]       * cos(p * f_j) - q[j + D/2] * sin(p * f_j)
 *     q'[j + D/2] = q[j + D/2] * cos(p * f_j) + q[j]       * sin(p * f_j)
 *
 * with `f_j = theta^(-2j/D)` for `0 <= j < D/2` and the frequencies duplicated
 * across the upper half. Llama-3 uses `theta = 500000` (Llama-2 used 10000).
 *
 * ## Why three plaintext multiplies and two rotations
 *
 * Fetching the partner channel is a *permutation*, not a shift: for `j < D/2`
 * the partner is `D/2` channels up, for `j >= D/2` it is `D/2` down, and a
 * single cyclic rotation cannot do both. Cheddar rotates over the whole slot
 * vector, so a one-sided rotation would also drag the second half of one head
 * into the first half of the next.
 *
 * Folding the masks into the sine plaintexts settles it. With
 *
 *     A[s] = -sin(...) where j < D/2, else 0
 *     B[s] = +sin(...) where j >= D/2, else 0
 *
 * the result is
 *
 *     q (*) cos + Rot(q, +D/2 * T) (*) A + Rot(q, -D/2 * T) (*) B
 *
 * and each slot reads its partner from *within its own head*, because the mask
 * is on by exactly the slots for which the shift stays inside the head. Three
 * `Mult(ct, pt)` at one scale, one `Add`, one `Rescale`: **one level**, matching
 * [SYLPH].
 *
 * The alternative -- folding the half-rotation into the projection's plaintext
 * weight matrix, so RoPE needs no rotation at all -- would need *both* `X W`
 * and `X W P`, that is a second PCMM. Two key switches are far cheaper than a
 * second 4096-wide plaintext product, so this is the right trade.
 *
 * ## Layout
 *
 * Tokens vary fastest and the channel index is strided by `num_tokens`:
 * slot `s` holds token `first_position + (s % T)` of channel `s / T`. This is
 * the packing `RmsNormHandler` and `SoftMaxHandler` already use, and it is
 * [SYLPH] section 3.3's "token index k varies fastest".
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class RoPeHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

  ConstContextPtr<word> context_;
  int num_tokens_;
  int head_dim_;
  int input_level_;
  double theta_;
  int num_slots_;
  int up_, down_;  // rotation distances for the two halves
  std::vector<int> rotation_distances_;
  // Which channel of its head each slot carries, one map per ciphertext
  // shape. Empty for the default packing, where it is `(s / T) % D`.
  std::vector<std::vector<int>> head_channel_;

  // The three plaintexts depend only on (first_position, level, variant), so
  // they are built once and reused. Encoder::Encode runs SpecialIFFT over every
  // slot and then num_primes * degree BigInt reductions, single-threaded on the
  // host, before anything reaches the GPU -- three of those per call was the
  // whole of RoPE's measured 171 ms against SiLU's 7.8 ms for a much heavier
  // circuit.
  //
  // ONE SLOT WAS NOT ENOUGH. With a channel map per ciphertext shape the tables
  // are per variant, and `LlamaBlock::ApplyRoPe` already walks the tensor
  // variant-major so a single slot would hit within each group -- but the last
  // variant is the only one that survives to the next call, and Q's variants
  // are re-encoded from scratch every time the block runs. That was 2.27 s of
  // the measured block against K's 3.3 ms, K having exactly one variant and so
  // still holding it from the run before.
  //
  // The cache is keyed by (level, variant) and holds one position at a time,
  // which is what bounds it: prefill fixes first_position across a segment and
  // fills the map once, decode moves it every token and empties the map on the
  // way past. So it is `variants * 3` plaintexts at the worst -- 120 MB for Q
  // at level 18 on sylphflow16_35 -- and never grows with the prompt.
  struct Encoded {
    Pt cos_pt, lower_pt, upper_pt;
  };
  mutable int cached_position_ = INT_MIN;
  mutable std::map<std::pair<int, int>, Encoded> tables_;

  // The three plaintexts for one (position, level, variant), encoding them if
  // this is the first time they have been asked for.
  const Encoded &GetTables(int first_position, int level, int variant) const;

  // The channel map of one variant, or the default packing's when there is
  // none. Returned by value only where it is built; Apply reads it by
  // reference.
  const std::vector<int> *ChannelMap(int variant) const;

 public:
  /**
   * @param context the evaluation context
   * @param num_tokens T, the number of tokens packed along the fast axis
   * @param head_dim D, the per-head channel count; must be even and must
   * divide the channels held by one ciphertext
   * @param input_level level of the input ciphertext
   * @param theta RoPE base; 500000 for Llama-3, 10000 for Llama-2
   */
  RoPeHandler(ConstContextPtr<word> context, int num_tokens, int head_dim,
              int input_level, double theta = 500000.0);

  /**
   * @brief The same handler on a packing that is not the default one.
   *
   * ## Why a second layout exists at all
   *
   * The ciphertext-ciphertext products want the head in the four slot bits
   * immediately above the token, because that field is the batch CC-MM's lane
   * index. Nothing else can be there, and the head cannot get there for free
   * after the projection -- but it can get there for **nothing at all** by
   * permuting the projection weight's columns, which is what
   * `LlamaBlock::LinearLeg::ChannelOrder` does. What that costs is here: the
   * channel a slot carries is no longer `(s / T) % D`, and three of the
   * channel's bits move onto the ciphertext axis, so each ciphertext of the
   * tensor sees a **different** set of frequencies and needs its own tables.
   *
   * The partner is still one rotation away and, in the layout the SinC leg
   * uses, it is the *same* rotation both ways -- the half-of-head bit lands on
   * the top slot bit, so `+step` and `-step` coincide at `num_slots / 2`.
   *
   * @param head_channel one map per ciphertext shape: `head_channel[v][s]` is
   * the channel of its head that slot `s` of variant `v` carries, in [0, D).
   * @param partner_step slot distance from a lower-half channel to its upper-
   * half partner. Verified against every map rather than trusted.
   */
  RoPeHandler(ConstContextPtr<word> context, int num_tokens, int head_dim,
              int input_level, double theta,
              std::vector<std::vector<int>> head_channel, int partner_step);

  // disable copying (or moving also)
  RoPeHandler(const RoPeHandler &) = delete;
  RoPeHandler &operator=(const RoPeHandler &) = delete;

  const std::vector<int> &GetRotationDistances() const {
    return rotation_distances_;
  }

  /**
   * @brief The same map in the clear, on a slot vector, for a host reference.
   *
   * @param res output slot vector
   * @param x input slot vector, `num_slots` entries in the layout above
   * @param first_position position of the token in slot 0
   */
  void PlainApply(std::vector<double> &res, const std::vector<double> &x,
                  int first_position, int variant = 0) const;

  /**
   * @brief Build the three plaintexts up front, so their cost sits in setup.
   *
   * [SYLPH] converts the model's plaintexts offline and keeps them on the GPU
   * for the whole run (section 5.1, "they need to reside in GPU memory
   * throughout the computation"); section 5.3 puts that conversion in its own
   * stage. Apply will do the same work lazily on its first call, which is
   * correct but hides tens of milliseconds of host encoding inside what looks
   * like an online measurement.
   *
   * @param first_position the position Apply will be called with
   * @param level level of the input; -1 uses the constructor's input_level
   */
  void Prepare(int first_position, int level = -1, int variant = 0) const;

  /** @brief How many channel maps this handler was built with; 1 by default. */
  int GetNumVariants() const {
    return head_channel_.empty() ? 1 : static_cast<int>(head_channel_.size());
  }

  /** @brief Device bytes the cached plaintexts hold, 0 before Prepare. */
  size_t GetPlaintextBytes() const;

  /**
   * @brief res = RoPE(x), costing one level.
   *
   * @param res output
   * @param x input in the layout documented above
   * @param first_position position of the token in slot 0, so a segment that
   * does not start at the beginning of the prompt still gets the right angles
   * @param evk_map supplies the two rotation keys
   */
  void Apply(Ct &res, const Ct &x, int first_position,
             const EvkMap<word> &evk_map, int variant = 0) const;
};

}  // namespace cheddar
