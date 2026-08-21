#include "core/MultiLevelCiphertext.h"

#include "common/Assert.h"

namespace cheddar {

template <typename word>
void MultiLevelCiphertext<word>::StaticInit(const Parameter<word> &param,
                                            const Encoder<word> &encoder) {
  int max_level = param.max_level_;
  auto &consts = level_down_consts_[&param];
  consts.clear();
  consts.resize(max_level + 1);
  for (int i = 1; i < max_level; i++) {
    double scale = param.GetRescalePrimeProd(i);
    encoder.EncodeConstant(consts.at(i), i, scale, 1.0);
  }
}

template <typename word>
void MultiLevelCiphertext<word>::StaticDestroy(const Parameter<word> &param) {
  // Only this Context's entry; another may still be alive, and it may be at
  // the same ring degree.
  level_down_consts_.erase(&param);
}

template <typename word>
MultiLevelCiphertext<word>::MultiLevelCiphertext(const Parameter<word> &param,
                                                 Ct &&ct)
    : param_{&param} {
  NPInfo np = ct.GetNP();
  AssertTrue(!ct.HasRx(), "MultiLevelCiphertext: Rx is not allowed.");
  AssertTrue(np.num_aux_ == 0,
             "MultiLevelCiphertext: Aux primes are not allowed.");
  AssertTrue(np.degree_ == param.degree_,
             "MultiLevelCiphertext: the ciphertext is at ring degree " +
                 std::to_string(np.degree_) + " and the Context at " +
                 std::to_string(param.degree_));
  int level = param.NPToLevel(np);
  level_map_.try_emplace(level, std::move(ct));
}

template <typename word>
void MultiLevelCiphertext<word>::AllocateLevel(int level) {
  AssertTrue(param_ != nullptr, "MultiLevelCiphertext: no parameter");
  NPInfo np = param_->LevelToNP(level, 0);
  AssertTrue(!Exists(level), "AddCiphertextAtLevel: Level " +
                                 std::to_string(level) + " already exists");
  level_map_.try_emplace(level, np);
}

template <typename word>
const Constant<word> &MultiLevelCiphertext<word>::GetLevelDownConst(
    const Parameter<word> &param, int level) {
  auto found = level_down_consts_.find(&param);
  AssertTrue(found != level_down_consts_.end(),
             "MultiLevelCiphertext: this Context did not register its "
             "level-down constants");
  return found->second.at(level);
}

template <typename word>
int MultiLevelCiphertext<word>::GetMaxLevel() const {
  AssertTrue(!level_map_.empty(), "MultiLevelCiphertext: level_map_ is empty.");
  return level_map_.rbegin()->first;
}

template <typename word>
int MultiLevelCiphertext<word>::GetMinLevel() const {
  AssertTrue(!level_map_.empty(), "MultiLevelCiphertext: level_map_ is empty.");
  return level_map_.begin()->first;
}

template <typename word>
Ciphertext<word> &MultiLevelCiphertext<word>::AtLevel(int level) {
  AssertTrue(Exists(level), "Level does not exist");
  return level_map_.at(level);
}

template <typename word>
const Ciphertext<word> &MultiLevelCiphertext<word>::AtLevel(int level) const {
  AssertTrue(Exists(level), "Level does not exist");
  return level_map_.at(level);
}

template <typename word>
bool MultiLevelCiphertext<word>::Exists(int level) const {
  return level_map_.find(level) != level_map_.end();
}

template <typename word>
void MultiLevelCiphertext<word>::Clear() {
  level_map_.clear();
}

template class MultiLevelCiphertext<uint32_t>;
template class MultiLevelCiphertext<uint64_t>;

}  // namespace cheddar
