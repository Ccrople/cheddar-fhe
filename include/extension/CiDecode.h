#pragma once

#include <vector>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/Hoist.h"

namespace cheddar {

/**
 * @brief The decode-side UNPACK (Doing.md 7.40 roadmap [1]): one dense
 * ciphertext whose K token rows carry K channels into K channel
 * ciphertexts, each channel's row BROADCAST over every row -- as one
 * hoisted transform instead of a mask + rotate-add ladder per channel.
 *
 *   out_c = sum_b rot_{b * stride}(input) * row_mask_{(c - b) mod K}
 *
 * The K rotations are computed ONCE (`HoistHandler::EvaluateBabyStep`: one
 * ModUp, the fused baby-step kernel), the select is ONE
 * `PAccumRotBatchCt` launch over every (baby, channel) pair, and each
 * channel pays one final mod-down. The transform consumes one level and
 * leaves scale pt_scale * in_scale / rescale_prod, exactly as a
 * HoistHandler evaluation would -- channel 0's words ARE
 * `HoistHandler::Evaluate`'s (the gate in `ci_batch_test`).
 *
 * The same primitive serves the decode score fan-out and the
 * prefill -> decode boundary: only the row masks differ.
 */
template <typename word>
class CiDecodeUnpack {
 private:
  using Ct = Ciphertext<word>;

  int num_channels_;
  int stride_;
  int pt_level_;
  double pt_scale_;
  // [0][b * stride] = row_masks[(K - b) % K]: the map whose single-giant
  // evaluation is channel 0. Compiled with the swap suppressed so the pts
  // stay the row indicators themselves.
  HoistHandler<word> hoist_;

  static PlainHoistMap MakeMap(const std::vector<Message> &row_masks,
                               int stride);

 public:
  /**
   * @brief Compile the unpack at `pt_level`. `row_masks[j]` is the slot
   * message of the indicator of token row j (the caller's layout decides
   * what a row is); `stride` is the slot stride of one token row.
   */
  CiDecodeUnpack(ConstContextPtr<word> context,
                 const std::vector<Message> &row_masks, int stride,
                 int pt_level, double pt_scale);
  CiDecodeUnpack(const CiDecodeUnpack &) = delete;
  CiDecodeUnpack &operator=(const CiDecodeUnpack &) = delete;
  CiDecodeUnpack(CiDecodeUnpack &&) = default;

  void AddRequiredRotations(EvkRequest &req) const;

  /** @brief The handler whose (serial) Evaluate is channel 0 -- for gates. */
  const HoistHandler<word> &Handler() const { return hoist_; }

  /** @brief `out[c]` = channel c broadcast, at `pt_level - 1`. */
  void Evaluate(ConstContextPtr<word> context, std::vector<Ct> &out,
                const Ct &input, const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
