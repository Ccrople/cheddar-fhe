#include "core/MultiLevelCiphertext.h"

#include "common/Assert.h"

namespace cheddar {

template <typename word>
const Parameter<word> &MultiLevelCiphertext<word>::ParamFor(int degree) {
  auto found = params_.find(degree);
  AssertTrue(found != params_.end(),
             "MultiLevelCiphertext: no Context registered for ring degree " +
                 std::to_string(degree));
  return *found->second;
}

template <typename word>
void MultiLevelCiphertext<word>::StaticInit(const Parameter<word> &param,
                                            const Encoder<word> &encoder) {
  const int degree = param.degree_;
  params_[degree] = &param;
  int max_level = param.max_level_;
  auto &consts = level_down_consts_[degree];
  consts.clear();
  consts.resize(max_level + 1);
  for (int i = 1; i < max_level; i++) {
    double scale = param.GetRescalePrimeProd(i);
    encoder.EncodeConstant(consts.at(i), i, scale, 1.0);
  }
}

template <typename word>
void MultiLevelCiphertext<word>::StaticDestroy(const Parameter<word> &param) {
  // Only this ring's entries; another Context may still be alive.
  params_.erase(param.degree_);
  level_down_consts_.erase(param.degree_);
}

template <typename word>
MultiLevelCiphertext<word>::MultiLevelCiphertext(Ct &&ct) {
  NPInfo np = ct.GetNP();
  AssertTrue(!ct.HasRx(), "MultiLevelCiphertext: Rx is not allowed.");
  AssertTrue(np.num_aux_ == 0,
             "MultiLevelCiphertext: Aux primes are not allowed.");
  // The ciphertext names its own ring through its NPInfo, so no caller has to
  // say which Context this belongs to.
  degree_ = np.degree_;
  int level = ParamFor(degree_).NPToLevel(np);
  level_map_.try_emplace(level, std::move(ct));
}

template <typename word>
void MultiLevelCiphertext<word>::AllocateLevel(int level) {
  NPInfo np = ParamFor(degree_).LevelToNP(level, 0);
  AssertTrue(!Exists(level), "AddCiphertextAtLevel: Level " +
                                 std::to_string(level) + " already exists");
  level_map_.try_emplace(level, np);
}

template <typename word>
const Constant<word> &MultiLevelCiphertext<word>::GetLevelDownConst(int degree,
                                                                    int level) {
  auto found = level_down_consts_.find(degree);
  AssertTrue(found != level_down_consts_.end(),
             "MultiLevelCiphertext: no Context registered for ring degree " +
                 std::to_string(degree));
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
