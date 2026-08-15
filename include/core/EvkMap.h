#pragma once

#include <unordered_map>

#include "core/Container.h"

namespace cheddar {

/**
 * @brief Class for storing client-prepared evaluation keys.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class EvkMap : public std::unordered_map<int, EvaluationKey<word>> {
 private:
  using Base = std::unordered_map<int, EvaluationKey<word>>;
  using Evk = EvaluationKey<word>;

  const Evk &GetEvk(int key_idx) const;

 public:
  static inline constexpr int kConjugationKeyIndex = 11111111;
  static inline constexpr int kMultiplicationKeyIndex = -22222222;
  static inline constexpr int kDenseToSparseKeyIndex = -33333333;
  static inline constexpr int kSparseToDenseKeyIndex = -44444444;
  static inline constexpr int kModPackKeyIndexBase = -55555555;

  using Base::Base;
  EvkMap(const EvkMap &) = delete;
  EvkMap &operator=(const EvkMap &) = delete;
  EvkMap(EvkMap &&) = default;

  /**
   * @brief Key index of the j-th ModPack switching key at module rank `rank`.
   *
   * ModPack needs one switching key per module component, and the components
   * are different polynomials for every rank, so the index must carry both.
   * The offsets used by a given rank occupy [rank, 2 * rank), and those
   * intervals are disjoint over powers of two, so no two ranks collide. The
   * largest offset is 2 * degree, which keeps every index well clear of the
   * other special indices above.
   *
   * @param rank module rank k = degree / small_degree
   * @param j module component index, 0 <= j < rank
   * @return int the key index
   */
  static constexpr int ModPackKeyIndex(int rank, int j) {
    return kModPackKeyIndexBase - rank - j;
  }

  const Evk &GetRotationKey(int rot_idx) const;
  const Evk &GetMultiplicationKey() const;
  const Evk &GetConjugationKey() const;
  const Evk &GetDenseToSparseKey() const;
  const Evk &GetSparseToDenseKey() const;
  const Evk &GetModPackKey(int rank, int j) const;
};

}  // namespace cheddar
