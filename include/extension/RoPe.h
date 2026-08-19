#pragma once

#include <climits>
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

  // The three plaintexts depend only on (first_position, level), so they are
  // built once and reused. Encoder::Encode runs SpecialIFFT over every slot and
  // then num_primes * degree BigInt reductions, single-threaded on the host,
  // before anything reaches the GPU -- three of those per call was the whole of
  // RoPE's measured 171 ms against SiLU's 7.8 ms for a much heavier circuit.
  // Prefill holds first_position fixed across a segment, so the cache hits
  // every time after the first; decode moves it per token, so it is a memo
  // rather than a constructor argument.
  mutable int cached_position_ = INT_MIN;
  mutable int cached_level_ = -1;
  mutable Pt cos_pt_, lower_pt_, upper_pt_;

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
                  int first_position) const;

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
             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
