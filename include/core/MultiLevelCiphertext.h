#pragma once

#include <map>

#include "core/Container.h"
#include "core/Encode.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief We sometimes need to keep multiple ciphertexts at different levels but
 * with the same scale. This class is used to keep track of those ciphertexts.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class MultiLevelCiphertext {
 private:
  using Ct = Ciphertext<word>;

  std::map<int, Ct> level_map_;

  // WHICH RING THIS BELONGS TO, AND WHY IT IS NOT INFERRED.
  //
  // This was a static map keyed by ring degree, on the reasoning that a
  // ciphertext names its own ring through its NPInfo. It does not. [SYLPH]'s
  // ladder runs TWO Contexts at the same degree on purpose -- the block's ring
  // and the switching ring share every prime and differ only in `alpha`, which
  // is the whole reason `ringswitch16_35` exists -- so the second one's
  // StaticInit silently replaced the first's entry, and the first ring's
  // EvalMod then asked a two-level parameter to place a level-27 NPInfo.
  // Measured, from inside a bootstrap, as
  // `NPInfo not found: 37 main + 4 terminal, against 2 levels`.
  //
  // Every construction site has its Context in hand, so it is passed rather
  // than looked up.
  const Parameter<word> *param_ = nullptr;

  // Keyed by the parameter itself, for the same reason.
  static inline std::map<const Parameter<word> *, std::vector<Constant<word>>>
      level_down_consts_{};

 public:
  static void StaticInit(const Parameter<word> &param,
                         const Encoder<word> &encoder);
  static void StaticDestroy(const Parameter<word> &param);

  MultiLevelCiphertext(const Parameter<word> &param, Ct &&ct);

  // movable, but not copyable
  MultiLevelCiphertext(MultiLevelCiphertext &&) = default;
  MultiLevelCiphertext &operator=(MultiLevelCiphertext &&) = default;

  int GetMaxLevel() const;
  int GetMinLevel() const;

  Ct &AtLevel(int level);
  const Ct &AtLevel(int level) const;
  bool Exists(int level) const;
  void Clear();

  // For the use in Context::AddLowerLevelsUntil

  void AllocateLevel(int level);

  /**
   * @param param the Context's parameter, selecting which constants to use
   * @param level the level being stepped down from
   */
  static const Constant<word> &GetLevelDownConst(const Parameter<word> &param,
                                                 int level);
};

}  // namespace cheddar
