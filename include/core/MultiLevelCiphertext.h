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

  // Which ring this instance belongs to, taken from its first ciphertext.
  int degree_ = 0;

  // Keyed by ring degree rather than held as a single pointer, so that two
  // Contexts at different degrees can be alive at once. With a single static
  // the second Context's StaticInit silently replaced the first's parameter
  // and level-down constants, and every MultiLevelCiphertext of the first ring
  // then resolved its levels against the wrong modulus chain.
  static inline std::map<int, const Parameter<word> *> params_{};

  // different from the one in Context
  static inline std::map<int, std::vector<Constant<word>>> level_down_consts_{};

  static const Parameter<word> &ParamFor(int degree);

 public:
  static void StaticInit(const Parameter<word> &param,
                         const Encoder<word> &encoder);
  static void StaticDestroy(const Parameter<word> &param);

  MultiLevelCiphertext(Ct &&ct);

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
   * @param degree ring degree selecting which Context's constants to use
   * @param level the level being stepped down from
   */
  static const Constant<word> &GetLevelDownConst(int degree, int level);
};

}  // namespace cheddar
