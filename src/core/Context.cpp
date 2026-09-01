#include "extension/Profile.h"
#include "core/Context.h"

#include <cstdlib>
#include <tuple>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"

namespace cheddar {

template <typename word>
void Context<word>::MatchResultWith(Ct &res, const Ct &a) const {
  // The ordering is important here.
  if (a.HasRx()) {
    res.ModifyNP(a.GetNP());
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(a.GetNP());
  }
}

template <typename word>
void Context<word>::MatchResultWith(Ct &res, const Ct &a, const Ct &b) const {
  if (a.HasRx() || b.HasRx()) {
    res.ModifyNP(a.GetNP());
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(a.GetNP());
  }
}

template <typename word>
DvConstView<word> Context<word>::GetPProd(NPInfo &np) const {
  int prime_offset = param_.GetMaxNumTer() - np.num_ter_;
  int num_q_primes = np.GetNumQ();
  return DvConstView<word>(p_prod_.data() + prime_offset, num_q_primes);
}

template <typename word>
DvConstView<word> Context<word>::GetPProd(NPInfo &np, int num_aux) const {
  if (num_aux == param_.alpha_) return GetPProd(np);
  const int level = param_.NPToLevel(np);
  auto it = narrow_p_prod_.find({level, num_aux});
  AssertTrue(it != narrow_p_prod_.end(),
             "GetPProd: no narrow key-switch basis was prepared for this "
             "(level, num_aux); call PrepareNarrowKeySwitch first");
  return it->second.ConstView(0);
}

template <typename word>
const ModSwitchHandler<word> &Context<word>::GetModSwitchHandler(
    int level, int num_aux) const {
  if (num_aux == param_.alpha_) return mod_switch_handlers_.at(level);
  auto it = narrow_handlers_.find({level, num_aux});
  AssertTrue(it != narrow_handlers_.end(),
             "GetModSwitchHandler: no narrow key-switch basis was prepared "
             "for this (level, num_aux); call PrepareNarrowKeySwitch first");
  return it->second;
}

template <typename word>
void Context<word>::PrepareNarrowKeySwitch(int level, int num_aux) const {
  AssertTrue(level >= 0 && level <= param_.max_level_,
             "PrepareNarrowKeySwitch: level out of range");
  AssertTrue(num_aux >= 1 && num_aux <= param_.alpha_,
             "PrepareNarrowKeySwitch: num_aux must be in [1, alpha]");
  // At level 0 a key with fewer than alpha auxiliary primes is ALSO how the
  // dense-to-sparse key announces itself, and `AdjustLevelForMultKey` tells
  // the two apart by nothing but the count. Where they would coincide there is
  // no way to tell, and the failure would be a wrong answer rather than an
  // error, so the ambiguity is refused rather than resolved. Nothing wants a
  // narrow basis at level 0 today -- the Llama product level is 1.
  if (level == 0 && param_.IsUsingSparseSecretEncapsulation()) {
    AssertTrue(num_aux != param_.LevelToNP(-1).GetNumQ(),
               "PrepareNarrowKeySwitch: at level 0 this auxiliary count is "
               "indistinguishable from the dense-to-sparse key's; pick "
               "another, or move the switch off level 0");
  }
  if (num_aux == param_.alpha_) return;  // the ordinary handler already exists
  const std::pair<int, int> key(level, num_aux);
  if (narrow_handlers_.count(key) != 0) return;

  narrow_handlers_.emplace(
      std::piecewise_construct, std::forward_as_tuple(key),
      std::forward_as_tuple(param_, level, elem_handler_, ntt_handler_,
                            num_aux));

  // The P product this basis carries, mod each q prime of the level. Same
  // shape and same indexing as p_prod_, so GetPProd's callers do not care
  // which one they were handed -- but over `num_aux` auxiliary primes rather
  // than all `alpha_`, which is the entire difference.
  NPInfo np = param_.LevelToNP(level, num_aux);
  const std::vector<word> primes = param_.GetPrimeVector(np);
  const int num_q_primes = np.GetNumQ();
  HostVector<word> h_p_prod(num_q_primes, 1);
  for (int i = 0; i < num_q_primes; i++) {
    const word mod_prime = primes[i];
    for (int j = 0; j < num_aux; j++) {
      h_p_prod[i] =
          primeutil::MultMod(h_p_prod[i], primes[num_q_primes + j], mod_prime);
    }
    h_p_prod[i] = primeutil::ToMontgomery(h_p_prod[i], mod_prime);
  }
  DeviceVector<word> d_p_prod(num_q_primes);
  CopyHostToDevice(d_p_prod, h_p_prod);
  narrow_p_prod_.emplace(key, std::move(d_p_prod));
}

template <typename word>
const ModSwitchHandler<word> &Context<word>::GetDtSModSwitchHandler() const {
  AssertTrue(param_.IsUsingSparseSecretEncapsulation(),
             "Sparse secret encapsulation is not enabled");
  return mod_switch_handlers_.back();
}

template <typename word>
const ModSwitchHandler<word> &Context<word>::GetStDModSwitchHandler() const {
  AssertTrue(param_.IsUsingSparseSecretEncapsulation(),
             "Sparse secret encapsulation is not enabled");
  return mod_switch_handlers_.at(param_.max_level_);
}

template <typename word>
void Context<word>::AssertSameScale(const double &scale1,
                                    const double &scale2) const {
  static constexpr double kScaleErrorMargin = 1e-12;
  double diff = scale1 - scale2;
  diff = diff < 0 ? -diff : diff;
  AssertTrue(diff < kScaleErrorMargin * scale1, "Scale mismatch");
}

template <typename word>
std::shared_ptr<Context<word>> Context<word>::Create(
    const Parameter<word> &param) {
  return std::shared_ptr<Context<word>>(new Context<word>(param));
}

template <typename word>
Context<word>::Context(const Parameter<word> &param)
    : param_{param},
      memory_pool_(param_),
      elem_handler_(param_),
      ntt_handler_(param_),
      encoder_(param_, ntt_handler_),
      gpu_encoder_(param_, ntt_handler_) {
  // 0. Set some static variables
  MultiLevelCiphertext<word>::StaticInit(param_, encoder_);

  // 1. Initialize mod_switch_handlers_
  for (int level = 0; level <= param_.max_level_; level++) {
    mod_switch_handlers_.emplace_back(param_, level, elem_handler_,
                                      ntt_handler_);
  }

  // 2. Initialize p_prod_
  NPInfo np = param_.LevelToNP(param_.max_level_, param_.alpha_);
  std::vector<word> primes = param_.GetPrimeVector(np);
  int num_q_primes = np.GetNumQ();
  HostVector<word> h_p_prod(num_q_primes, 1);
  for (int i = 0; i < num_q_primes; i++) {
    word mod_prime = primes[i];
    for (int j = 0; j < np.num_aux_; j++) {
      h_p_prod[i] =
          primeutil::MultMod(h_p_prod[i], primes[num_q_primes + j], mod_prime);
    }
    h_p_prod[i] = primeutil::ToMontgomery(h_p_prod[i], mod_prime);
  }
  CopyHostToDevice(p_prod_, h_p_prod);

  if (!param_.IsUsingSparseSecretEncapsulation()) return;

  // 3. Initialize level_down_consts_
  level_down_consts_.resize(param_.default_encryption_level_ + 1);
  for (int i = param_.default_encryption_level_; i > 0; i--) {
    double scale = param_.GetScale(i);
    encoder_.EncodeConstant(level_down_consts_[i], i, scale, 1.0);
  }

  // Extra data for SSE
  mod_switch_handlers_.emplace_back(param_, -1, elem_handler_, ntt_handler_);
  np = param_.LevelToNP(-1, param_.GetSSENumAux());
  primes = param_.GetPrimeVector(np);
  num_q_primes = np.GetNumQ();
  HostVector<word> h_p_prod_dts(num_q_primes, 1);
  for (int i = 0; i < num_q_primes; i++) {
    word mod_prime = primes[i];
    for (int j = 0; j < np.num_aux_; j++) {
      h_p_prod_dts[i] = primeutil::MultMod(h_p_prod_dts[i],
                                           primes[num_q_primes + j], mod_prime);
    }
    h_p_prod_dts[i] = primeutil::ToMontgomery(h_p_prod_dts[i], mod_prime);
  }
  CopyHostToDevice(p_prod_dts_, h_p_prod_dts);
}

template <typename word>
Context<word>::~Context() {
  MultiLevelCiphertext<word>::StaticDestroy(param_);
}

template <typename word>
void Context<word>::Copy(Ct &res, const Ct &a) const {
  if (&res == &a) return;
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());

  CopyDeviceToDevice(res.bx_, a.bx_);
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Add(Ct &res, const Ct &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  bool rx_add = a.HasRx() && b.HasRx();
  MatchResultWith(res, a, b);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  if (rx_add) {
    auto res_temp = res.ViewVector();
    elem_handler_.Add(res_temp, np, a.ConstViewVector(), b.ConstViewVector());
  } else {
    auto res_temp = res.ViewVector(0, true);
    elem_handler_.Add(res_temp, np, a.ConstViewVector(0, true),
                      b.ConstViewVector(0, true));
    if (a.HasRx()) {
      CopyDeviceToDevice(res.rx_, a.rx_);
    } else if (b.HasRx()) {
      CopyDeviceToDevice(res.rx_, b.rx_);
    }
  }
}

template <typename word>
void Context<word>::Add(Ct &res, const Ct &a, const Pt &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.Add(res_temp, np, {a.BxConstView()}, {b.ConstView()});
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Add(Ct &res, const Ct &a, const Const &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.AddConst(res_temp, np, {a.BxConstView()}, b.ConstView());
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Ct &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  bool rx_sub = a.HasRx() && b.HasRx();
  MatchResultWith(res, a, b);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  if (rx_sub) {
    auto res_temp = res.ViewVector();
    elem_handler_.Sub(res_temp, np, a.ConstViewVector(), b.ConstViewVector());
  } else {
    auto res_temp = res.ViewVector(0, true);
    elem_handler_.Sub(res_temp, np, a.ConstViewVector(0, true),
                      b.ConstViewVector(0, true));
    if (a.HasRx()) {
      CopyDeviceToDevice(res.rx_, a.rx_);
    } else if (b.HasRx()) {
      auto res_temp = std::vector<DvView<word>>{res.RxView()};
      elem_handler_.Neg(res_temp, np, {b.RxConstView()});
    }
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Ct &a, const Pt &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.Sub(res_temp, np, {a.BxConstView()}, {b.ConstView()});
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Ct &a, const Const &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.SubConst(res_temp, np, {a.BxConstView()}, b.ConstView());
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Pt &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, b);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.Sub(res_temp, np, {a.ConstView()}, {b.BxConstView()});
  if (b.HasRx()) {
    auto res_temp = std::vector<DvView<word>>{res.AxView(), res.RxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView(), b.RxConstView()});
  } else {
    auto res_temp = std::vector<DvView<word>>{res.AxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView()});
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Const &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, b);
  res.SetNumSlots(b.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.SubOppositeConst(res_temp, np, {b.BxConstView()},
                                 a.ConstView());
  if (b.HasRx()) {
    auto res_temp = std::vector<DvView<word>>{res.AxView(), res.RxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView(), b.RxConstView()});
  } else {
    auto res_temp = std::vector<DvView<word>>{res.AxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView()});
  }
}

template <typename word>
void Context<word>::Neg(Ct &res, const Ct &a) const {
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Neg(res_temp, np, a.ConstViewVector());
}

template <typename word>
void Context<word>::Mult(Ct &res, const Ct &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertFalse(a.HasRx() || b.HasRx(),
              "Relinearization required before Mult Ct x Ct");
  res.ModifyNP(a.GetNP());
  res.PrepareRx();
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Tensor(res_temp, np, a.ConstViewVector(0, true),
                       b.ConstViewVector(0, true));
}

template <typename word>
void Context<word>::Mult(Ct &res, const Ct &a, const Pt &b) const {
  AssertSameNP(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.PMult(res_temp, np, a.ConstViewVector(), b.ConstView());
}

template <typename word>
void Context<word>::Mult(Ct &res, const Ct &a, const Const &b) const {
  AssertSameNP(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale() * b.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.MultConst(res_temp, np, a.ConstViewVector(), b.ConstView());
}

template <typename word>
bool Context<word>::IsMultUnsafeCompatible(int level1, int level2) const {
  if (level1 == level2) return true;

  int min_level = Min(level1, level2);
  int max_level = Max(level1, level2);

  NPInfo min_np = param_.LevelToNP(min_level, 0);
  NPInfo max_np = param_.LevelToNP(max_level, 0);

  return min_np.IsSubsetOf(max_np);
}

template <typename word>
void Context<word>::MultUnsafe(Ct &res, const Ct &a, const Ct &b,
                               int level) const {
  AssertFalse(a.HasRx() || b.HasRx(),
              "Relinearization required before MultUnsafe Ct x Ct");
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();
  AssertTrue(a_np.num_aux_ == 0 && b_np.num_aux_ == 0,
             "MultUnsafe Ct x Ct should be only used for ciphertexts without "
             "aux primes");
  int a_level = param_.NPToLevel(a_np);
  int b_level = param_.NPToLevel(b_np);
  if (level == -1) {
    level = Min(a_level, b_level);
  }

  // In-place operation are not possible if the levels are different
  if ((&res == &a && a_level != level) || (&res == &b && b_level != level)) {
    Ct tmp;
    MultUnsafe(tmp, a, b, level);
    res = std::move(tmp);
    return;
  }

  // Target NPInfo
  NPInfo res_np = param_.LevelToNP(level, 0);
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Ct");

  // Actual computation
  res.ModifyNP(res_np);
  res.PrepareRx();
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  auto res_temp = res.ViewVector();
  elem_handler_.Tensor(res_temp, res_np,
                       a.ConstViewVector(a_front_ignore, true),
                       b.ConstViewVector(b_front_ignore, true));
}

template <typename word>
void Context<word>::MultUnsafe(Ct &res, const Ct &a, const Pt &b,
                               int level) const {
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();
  int a_level = param_.NPToLevel(a_np);
  int b_level = param_.NPToLevel(b_np);
  if (level == -1) {
    level = Min(a_level, b_level);
  }

  // In-place operation are not possible if the levels are different
  if (&res == &a && a_level != level) {
    Ct tmp;
    MultUnsafe(tmp, a, b, level);
    res = std::move(tmp);
    return;
  }

  // Target NPInfo
  int num_aux = Min(a_np.num_aux_, b_np.num_aux_);
  NPInfo res_np = param_.LevelToNP(level, num_aux);
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Pt");

  // Actual computation
  if (a.HasRx()) {
    res.ModifyNP(res_np);
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(res_np);
  }
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  auto res_temp = res.ViewVector();
  elem_handler_.PMult(res_temp, res_np, a.ConstViewVector(a_front_ignore),
                      b.ConstView(b_front_ignore));
}

template <typename word>
void Context<word>::MultUnsafe(Ct &res, const Ct &a, const Const &b,
                               int level) const {
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();
  int a_level = param_.NPToLevel(a_np);
  int b_level = param_.NPToLevel(b_np);
  if (level == -1) {
    level = Min(a_level, b_level);
  }

  // In-place operation are not possible if the levels are different
  if (&res == &a && a_level != level) {
    Ct tmp;
    MultUnsafe(tmp, a, b, level);
    res = std::move(tmp);
    return;
  }

  // Target NPInfo
  int num_aux = Min(a_np.num_aux_, b_np.num_aux_);
  NPInfo res_np = param_.LevelToNP(level, num_aux);
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Const");

  // Actual computation
  if (a.HasRx()) {
    res.ModifyNP(res_np);
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(res_np);
  }
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale() * b.GetScale());

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  auto res_temp = res.ViewVector();
  elem_handler_.MultConst(res_temp, res_np, a.ConstViewVector(a_front_ignore),
                          b.ConstView(b_front_ignore));
}

template <typename word>
void Context<word>::Permute(Ct &res, const Ct &a, int rot_dist) const {
  int num_slots = a.GetNumSlots();
  int rot_idx = rot_dist % num_slots;
  if (rot_idx < 0) rot_idx += num_slots;

  if (rot_idx == 0) {
    Copy(res, a);
    return;
  }
  // in-place operation is not supported
  if (&res == &a) {
    Ct tmp;
    Permute(tmp, a, rot_idx);
    res = std::move(tmp);
    return;
  }

  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = a.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Permute(res_temp, np, rot_idx, a.ConstViewVector());
  // elem_handler_.PermuteAccum(res_temp, np, {rot_idx}, {a.ConstViewVector()});
}

template <typename word>
void Context<word>::PermuteConjugate(Ct &res, const Ct &a) const {
  // On the conjugate-invariant ring this is the identity, and correctly so:
  // conjugation is the {+-1} the acting group quotients out. It is left
  // callable rather than refused because the identity is the right answer.
  static constexpr int conj_rot_idx = -1;
  // in-place operation is not supported
  if (&res == &a) {
    Ct tmp;
    PermuteConjugate(tmp, a);
    res = std::move(tmp);
    return;
  }

  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Permute(res_temp, np, conj_rot_idx, a.ConstViewVector());
}

template <typename word>
void Context<word>::MultImaginaryUnit(Ct &res, const Ct &a) const {
  // R+ is totally real: multiplying its slots by i lands outside the ring, and
  // the ordinary ring's imaginary unit is not even an element of it.
  AssertTrue(!param_.conjugate_invariant_,
             "MultImaginaryUnit: the conjugate-invariant ring is totally real "
             "and has no imaginary unit");
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.MultImaginaryUnit(res_temp, np, a.ConstViewVector(),
                                  ntt_handler_.ImaginaryUnitConstView(np));
}

template <typename word>
void Context<word>::Relinearize(Ct &res, const Ct &a, const Evk &key) const {
  AssertTrue(a.HasRx(), "Relinearize requires aux");
  MultKey(res, a, key);
}

template <typename word>
void Context<word>::AdjustLevelForMultKey(int &level, const int num_q,
                                          const int num_aux) const {
  if (level == 0) {
    // This still has some issues if alpha == num_q at level 0
    //
    // A key with fewer than alpha auxiliary primes used to mean exactly one
    // thing here -- the dense-to-sparse key, which is switched on the short
    // base and so at level -1. A narrow basis that was explicitly prepared is
    // the other thing it can mean now, and it must not be rerouted.
    if (num_aux != param_.alpha_ &&
        narrow_handlers_.count({level, num_aux}) == 0) {
      level = -1;  // maybe...
    }
  }
  if (level == -1) {
    AssertTrue(num_aux == num_q, "Invalid setting for DTS");
  } else {
    // A key on a narrow auxiliary basis is legal exactly when the machinery
    // for it was built; anything else is the old mistake of handing a key from
    // one parameter set to another.
    AssertTrue(num_aux == param_.alpha_ ||
                   narrow_handlers_.count({level, num_aux}) != 0,
               "Invalid setting for MultKeyNoModDown");
  }
}

template <typename word>
void Context<word>::MultKey(Ct &res, const Ct &a, const Evk &key) const {
  NvtxScope _nv("ks: MultKey");
  NPInfo np = a.GetNP();
  int level = param_.NPToLevel(np);
  int num_aux = key.GetNP().num_aux_;
  int num_q = np.GetNumQ();
  AdjustLevelForMultKey(level, num_q, num_aux);
  const auto &mod_switcher = level == -1
                                ? GetDtSModSwitchHandler()
                                : GetModSwitchHandler(level, num_aux);

  Ct accum;
  MultKeyNoModDown(accum, a, key);

  // Prepare result
  res.RemoveRx();
  res.ModifyNP(np);
  res.SetScale(a.GetScale());
  res.SetNumSlots(a.GetNumSlots());

  auto res_bx_view = res.BxView();
  auto res_ax_view = res.AxView();
  mod_switcher.ModDown(res_bx_view, accum.BxConstView());
  mod_switcher.ModDown(res_ax_view, accum.AxConstView());
}

template <typename word>
void Context<word>::MultKeyNoModDown(Ct &accum, const std::vector<Dv> &a_modup,
                                     const Ct &a_orig, const Evk &key) const {
  NvtxScope _nv("ks: MultKeyNoModDown");
  NPInfo a_orig_np = a_orig.GetNP();
  int level = param_.NPToLevel(a_orig_np);
  int num_main = a_orig_np.num_main_;
  int num_ter = a_orig_np.num_ter_;
  int num_aux = key.GetNP().num_aux_;
  int num_q = num_main + num_ter;
  AdjustLevelForMultKey(level, num_q, num_aux);
  int prime_offset = ((level == -1) ? 0 : (param_.GetMaxNumTer() - num_ter));

  AssertTrue(&accum != &a_orig,
             "In-place operation is not supported for MultKeyNoModDown");

  int padded_num_q = num_q + prime_offset;
  int beta = DivCeil(padded_num_q, num_aux);

  AssertTrue(key.GetBeta() >= beta && static_cast<int>(a_modup.size()) == beta,
             "Beta mismatch");

  NPInfo np(num_main, num_ter, num_aux, param_.degree_);
  accum.RemoveRx();
  accum.ModifyNP(np);
  accum.SetScale(a_orig.GetScale());
  accum.SetNumSlots(a_orig.GetNumSlots());

  std::vector<DvView<word>> accum_views = accum.ViewVector();
  std::vector<std::vector<DvConstView<word>>> key_views;
  std::vector<DvConstView<word>> modup_view;

  for (int i = 0; i < beta; i++) {
    int prime_index_end = Min((i + 1) * num_aux, padded_num_q);
    if (prime_index_end <= prime_offset) continue;
    key_views.push_back(key.ConstViewVector(i, prime_offset));
    modup_view.push_back(a_modup.at(i).ConstView(num_aux * param_.degree_));
  }
  elem_handler_.PAccum(accum_views, np, key_views, modup_view);
}

// One contiguous buffer for a whole group of switches, and the batched mod-up
// when the level allows it. The per-switch entry point above is what a narrower
// auxiliary basis still takes.
template <typename word>
void Context<word>::ModUpForKeySwitchBatch(
    Dv &buffer, std::vector<std::vector<DvConstView<word>>> &mod_up_views,
    const Ct &a, const Evk &key, const DvConstView<word> &a_coeffs,
    int batch) const {
  NvtxScope _nv("ks: ModUpBatch");
  NPInfo a_np = a.GetNP();
  AssertTrue(a_np.num_aux_ == 0,
             "ModUpForKeySwitchBatch is not supported for ciphertexts with p "
             "primes");
  AssertTrue(batch >= 1, "ModUpForKeySwitchBatch: invalid batch");

  int num_q = a_np.GetNumQ();
  int level = param_.NPToLevel(a_np);
  int num_aux = key.GetNP().num_aux_;
  AdjustLevelForMultKey(level, num_q, num_aux);
  int prime_offset =
      ((level == -1) ? 0 : (param_.GetMaxNumTer() - a_np.num_ter_));
  int padded_num_q = num_q + prime_offset;
  int beta = DivCeil(padded_num_q, num_aux);
  const auto &mod_switcher = level == -1
                                 ? GetDtSModSwitchHandler()
                                 : GetModSwitchHandler(level, num_aux);

  const int degree = param_.degree_;
  const int limb_words = (num_q + num_aux) * degree;
  const int q_words = num_q * degree;
  AssertTrue(a_coeffs.TotalSize() == batch * q_words,
             "ModUpForKeySwitchBatch: coefficient buffer size mismatch");
  buffer.resize(batch * beta * limb_words);

  // Digit-major (`ModUpFromCoeffBatch`'s layout): digit i of switch s at
  // (i * batch + s). DvConstView holds const members, so it is
  // copy-constructible but not assignable: build the rows rather than
  // assigning them.
  mod_up_views.clear();
  mod_up_views.resize(batch);
  for (int s = 0; s < batch; s++) {
    for (int i = 0; i < beta; i++) {
      mod_up_views[s].emplace_back(buffer.data() + (i * batch + s) * limb_words,
                                   limb_words, num_aux * degree);
    }
  }

  if (level >= 0 && !modup_coeff_serial_) {
    DvView<word> dst(buffer.data(), batch * beta * limb_words, 0);
    mod_switcher.ModUpFromCoeffBatch(dst, a_coeffs, batch);
    return;
  }

  for (int s = 0; s < batch; s++) {
    std::vector<DvView<word>> dst_views;
    for (int i = 0; i < beta; i++) {
      dst_views.emplace_back(buffer.data() + (i * batch + s) * limb_words,
                             limb_words, num_aux * degree);
    }
    DvConstView<word> coeff_s(a_coeffs.data() + s * q_words, q_words, 0);
    mod_switcher.ModUpFromCoeff(dst_views, coeff_s);
  }
}

template <typename word>
bool Context<word>::modup_coeff_serial_ =
    (std::getenv("CHEDDAR_MODUP_COEFF_SERIAL") != nullptr &&
     std::getenv("CHEDDAR_MODUP_COEFF_SERIAL")[0] == '1');

template <typename word>
void Context<word>::SetModUpCoeffSerial(bool serial) {
  modup_coeff_serial_ = serial;
}

template <typename word>
void Context<word>::MultKeyAccumNoModDown(
    Ct &accum, const std::vector<std::vector<DvConstView<word>>> &a_modups,
    const Ct &a_orig, const std::vector<const Evk *> &keys,
    bool accumulate) const {
  NvtxScope _nv("ks: MultKeyAccum");
  AssertTrue(!keys.empty(), "MultKeyAccumNoModDown: no keys");
  AssertTrue(a_modups.size() >= keys.size(),
             "MultKeyAccumNoModDown: fewer mod-up results than keys");
  AssertTrue(&accum != &a_orig,
             "In-place operation is not supported for MultKeyAccumNoModDown");

  NPInfo a_orig_np = a_orig.GetNP();
  int level = param_.NPToLevel(a_orig_np);
  int num_main = a_orig_np.num_main_;
  int num_ter = a_orig_np.num_ter_;
  int num_aux = keys.at(0)->GetNP().num_aux_;
  int num_q = num_main + num_ter;
  AdjustLevelForMultKey(level, num_q, num_aux);
  int prime_offset = ((level == -1) ? 0 : (param_.GetMaxNumTer() - num_ter));
  int padded_num_q = num_q + prime_offset;
  int beta = DivCeil(padded_num_q, num_aux);

  NPInfo np(num_main, num_ter, num_aux, param_.degree_);
  if (accumulate) {
    AssertTrue(accum.GetNP() == np,
               "MultKeyAccumNoModDown: the accumulator is on another basis");
  } else {
    accum.RemoveRx();
    accum.ModifyNP(np);
    accum.SetScale(a_orig.GetScale());
    accum.SetNumSlots(a_orig.GetNumSlots());
  }

  std::vector<DvView<word>> accum_views = accum.ViewVector();
  std::vector<std::vector<DvConstView<word>>> key_views;
  std::vector<DvConstView<word>> modup_view;

  for (size_t k = 0; k < keys.size(); k++) {
    const Evk &key = *keys.at(k);
    AssertTrue(key.GetNP().num_aux_ == num_aux,
               "MultKeyAccumNoModDown: keys differ in auxiliary prime count");
    AssertTrue(key.GetBeta() >= beta &&
                   static_cast<int>(a_modups.at(k).size()) == beta,
               "Beta mismatch");
    for (int i = 0; i < beta; i++) {
      int prime_index_end = Min((i + 1) * num_aux, padded_num_q);
      if (prime_index_end <= prime_offset) continue;
      key_views.push_back(key.ConstViewVector(i, prime_offset));
      modup_view.push_back(a_modups.at(k).at(i));
    }
  }

  // The accumulator joins as the extra ciphertext CPAccumAdd folds in, which
  // is what removes the separate addition per switch.
  if (accumulate) {
    key_views.push_back(accum.ConstViewVector());
  }
  elem_handler_.PAccum(accum_views, np, key_views, modup_view);
}

template <typename word>
void Context<word>::MultKeyNoModDown(Ct &accum, const Ct &a,
                                     const Evk &key) const {
  NvtxScope _nv("ks: MultKeyNoModDown");
  NPInfo a_np = a.GetNP();
  AssertTrue(a_np.num_aux_ == 0,
             "MultKeyNoModDown is not supported for ciphertexts with p primes");
  AssertTrue(&accum != &a,
             "In-place operation is not supported for MultKeyNoModDown");

  int num_q = a_np.GetNumQ();
  int level = param_.NPToLevel(a_np);
  int num_aux = key.GetNP().num_aux_;
  AdjustLevelForMultKey(level, num_q, num_aux);
  int prime_offset =
      ((level == -1) ? 0 : (param_.GetMaxNumTer() - a_np.num_ter_));
  int padded_num_q = num_q + prime_offset;
  int beta = DivCeil(padded_num_q, num_aux);
  const auto &mod_switcher = level == -1
                                ? GetDtSModSwitchHandler()
                                : GetModSwitchHandler(level, num_aux);

  // Mod-up result preparation
  std::vector<Dv> mod_up_result;
  std::vector<DvView<word>> mod_up_result_view;
  for (int i = 0; i < beta; i++) {
    int prime_index_end = Min((i + 1) * num_aux, padded_num_q);
    if (prime_index_end <= prime_offset) {
      mod_up_result.emplace_back(0);
      mod_up_result_view.push_back(mod_up_result[i].View(0));
    } else {
      mod_up_result.emplace_back((num_q + num_aux) * param_.degree_);
      mod_up_result_view.push_back(
          mod_up_result[i].View(num_aux * param_.degree_));
    }
  }

  DvConstView<word> p_prod = (level == -1) ? p_prod_dts_.ConstView(0)
                                          : GetPProd(a_np, num_aux);

  // relinearization or simple mult-key
  if (a.HasRx()) {
    mod_switcher.ModUp(mod_up_result_view, a.RxConstView());
    MultKeyNoModDown(accum, mod_up_result, a, key);
    // accum.bx_ += p_prod * a.bx_
    // accum.ax_ += p_prod * a.ax_
    DvView<word> accum_bx_view(accum.bx_.data(), num_q * param_.degree_, 0);
    DvView<word> accum_ax_view(accum.ax_.data(), num_q * param_.degree_, 0);
    std::vector<DvView<word>> caccum_res{accum_bx_view, accum_ax_view};
    std::vector<std::vector<DvConstView<word>>> src_const_views;
    src_const_views.push_back(a.ConstViewVector(0, true));
    src_const_views.push_back(accum.ConstViewVector());
    elem_handler_.CAccum(caccum_res, a_np, src_const_views, {p_prod});
  } else {
    mod_switcher.ModUp(mod_up_result_view, a.AxConstView());
    MultKeyNoModDown(accum, mod_up_result, a, key);
    // accum.bx_ += p_prod * a.bx_
    DvView<word> accum_bx_view(accum.bx_.data(), num_q * param_.degree_, 0);
    std::vector<DvView<word>> caccum_res{accum_bx_view};
    std::vector<std::vector<DvConstView<word>>> src_const_views;
    src_const_views.push_back(std::vector<DvConstView<word>>{a.BxConstView()});
    src_const_views.push_back(
        std::vector<DvConstView<word>>{accum.BxConstView()});
    elem_handler_.CAccum(caccum_res, a_np, src_const_views, {p_prod});
  }
}

template <typename word>
void Context<word>::MultKeyBatch(word *dst, int dst_ct_stride, const word *src,
                                 int src_ct_stride, const NPInfo &np,
                                 const std::vector<const Evk *> &keys,
                                 int batch) const {
  const int q_words = np.GetNumQ() * param_.degree_;
  AssertTrue(src_ct_stride >= 2 * q_words, "MultKeyBatch: the ciphertexts overlap");
  MultKeyBatch(dst, dst_ct_stride, src + q_words, src_ct_stride, src, nullptr,
               src_ct_stride, np, keys, batch);
}

template <typename word>
void Context<word>::MultKeyBatch(word *dst, int dst_ct_stride,
                                 const word *switched, int switched_stride,
                                 const word *add_b, const word *add_a,
                                 int add_stride, const NPInfo &np,
                                 const std::vector<const Evk *> &keys,
                                 int batch) const {
  NvtxScope _nv("ks: MultKeyBatch");
  const int degree = param_.degree_;
  const int num_q = np.GetNumQ();
  const int q_words = num_q * degree;
  AssertTrue(!keys.empty(), "MultKeyBatch: no keys");
  const int num_aux = keys.at(0)->GetNP().num_aux_;
  const int ext_words = (num_q + num_aux) * degree;
  AssertTrue(dst_ct_stride >= 2 * q_words, "MultKeyBatch: the ciphertexts overlap");
  DeviceVector<word> accum(static_cast<size_t>(batch) * 2 * ext_words);
  MultKeyBatchNoModDown(accum.data(), 2 * ext_words, switched, switched_stride,
                        add_b, add_a, add_stride, np, keys, batch);

  // Mod-down of both parts. With the parts contiguous on both sides the
  // 2 * batch polynomials are one strided group.
  int level = param_.NPToLevel(np);
  AdjustLevelForMultKey(level, num_q, num_aux);
  const auto &mod_switcher = GetModSwitchHandler(level, num_aux);
  if (dst_ct_stride == 2 * q_words) {
    mod_switcher.ModDownBatch(dst, q_words, accum.data(), ext_words,
                              2 * batch);
  } else {
    mod_switcher.ModDownBatch(dst, dst_ct_stride, accum.data(), 2 * ext_words,
                              batch);
    mod_switcher.ModDownBatch(dst + q_words, dst_ct_stride,
                              accum.data() + ext_words, 2 * ext_words, batch);
  }
}

template <typename word>
void Context<word>::MultKeyBatchNoModDown(
    word *dst, int dst_ct_stride, const word *switched, int switched_stride,
    const word *add_b, const word *add_a, int add_stride, const NPInfo &np,
    const std::vector<const Evk *> &keys, int batch) const {
  NvtxScope _nv("ks: MultKeyBatchNoModDown");
  AssertTrue(batch >= 1 && static_cast<int>(keys.size()) == batch,
             "MultKeyBatch: one key per ciphertext");
  AssertTrue(np.num_aux_ == 0,
             "MultKeyBatch is not supported for ciphertexts with p primes");
  const int degree = param_.degree_;
  const int num_q = np.GetNumQ();
  const int q_words = num_q * degree;
  int level = param_.NPToLevel(np);
  const int num_aux = keys.at(0)->GetNP().num_aux_;
  AdjustLevelForMultKey(level, num_q, num_aux);
  AssertTrue(level >= 0, "MultKeyBatch: the dense-to-sparse switch is not batched");
  const int prime_offset = param_.GetMaxNumTer() - np.num_ter_;
  const int padded_num_q = num_q + prime_offset;
  const int beta = DivCeil(padded_num_q, num_aux);
  const auto &mod_switcher = GetModSwitchHandler(level, num_aux);
  const int ext_words = (num_q + num_aux) * degree;
  AssertTrue(switched_stride >= q_words && dst_ct_stride >= 2 * ext_words &&
                 (add_b == nullptr || add_stride >= q_words),
             "MultKeyBatchNoModDown: the ciphertexts overlap");

  // 1. Mod-up of every a-part, one buffer per digit.
  NvtxScope *_d = new NvtxScope("mkb: digit buffers");
  std::vector<DeviceVector<word>> digits;
  std::vector<DvView<word>> digit_views;
  std::vector<int> used;
  digits.reserve(beta);
  for (int i = 0; i < beta; i++) {
    const int prime_index_end = Min((i + 1) * num_aux, padded_num_q);
    if (prime_index_end <= prime_offset) {
      digits.emplace_back(0);
      digit_views.emplace_back(nullptr, 0, 0);
      continue;
    }
    digits.emplace_back(batch * ext_words);
    digit_views.emplace_back(digits.back().data(), batch * ext_words,
                             batch * ext_words - q_words);
    used.push_back(i);
  }
  AssertTrue(!used.empty() &&
                 static_cast<int>(used.size()) <=
                     ElementWiseHandler<word>::max_batch_digits_,
             "MultKeyBatch: digit count out of range");
  delete _d;
  {
    NvtxScope _s("mkb: modup");
    mod_switcher.ModUpBatch(digit_views, switched, switched_stride, batch);
  }

  // 2. The key multiply, every switch with its own key.
  NvtxScope _t("mkb: table + keymult");
  const NPInfo ext_np(np.num_main_, np.num_ter_, num_aux, degree);
  // Four pointers per (switch, digit): the q limbs and the auxiliary limbs of
  // the key's b and a parts. A key made at another level carries more q
  // limbs than this switch reads, so its auxiliary part sits at its own
  // offset -- the same view arithmetic `KeyMult` does per key.
  HostVector<uint64_t> key_table(static_cast<size_t>(batch) * used.size() * 4);
  for (int b = 0; b < batch; b++) {
    const Evk &key = *keys.at(b);
    AssertTrue(key.GetNP().num_aux_ == num_aux,
               "MultKeyBatch: keys differ in auxiliary prime count");
    AssertTrue(key.GetBeta() >= beta, "Beta mismatch");
    for (size_t k = 0; k < used.size(); k++) {
      const DvConstView<word> kb = key.BxConstView(used[k], prime_offset);
      const DvConstView<word> ka = key.AxConstView(used[k], prime_offset);
      AssertTrue(kb.QSize() >= q_words && ka.QSize() == kb.QSize() &&
                     kb.AuxSize() == num_aux * degree &&
                     ka.AuxSize() == num_aux * degree,
                 "MultKeyBatch: a key does not cover this switch's limbs");
      uint64_t *row = key_table.data() + (b * used.size() + k) * 4;
      row[0] = reinterpret_cast<uint64_t>(kb.data());
      row[1] = reinterpret_cast<uint64_t>(ka.data());
      row[2] = reinterpret_cast<uint64_t>(kb.data() + kb.QSize());
      row[3] = reinterpret_cast<uint64_t>(ka.data() + ka.QSize());
    }
  }
  DeviceVector<uint64_t> key_table_dev;
  CopyHostToDevice(key_table_dev, key_table);
  std::vector<const word *> modup_ptrs;
  for (int i : used) modup_ptrs.push_back(digits.at(i).data());
  NPInfo p_prod_np = np;
  const DvConstView<word> p_prod = GetPProd(p_prod_np, num_aux);

  elem_handler_.KeyMultBatch(
      dst, dst_ct_stride, ext_np, modup_ptrs, ext_words,
      reinterpret_cast<const word *const *>(key_table_dev.data()), add_b,
      add_a, add_stride, p_prod.data(), batch);
}

template <typename word>
void Context<word>::RelinearizeRescale(Ct &res, const Ct &a,
                                       const Evk &key) const {
  AssertTrue(a.HasRx(), "RelinearizeRescale requires aux");

  int level = param_.NPToLevel(a.GetNP());

  Ct accum;
  MultKeyNoModDown(accum, a, key);

  // Prepare result
  res.RemoveRx();
  res.ModifyNP(param_.LevelToNP(level - 1));

  res.SetScale(a.GetScale() / param_.GetRescalePrimeProd(level));
  res.SetNumSlots(a.GetNumSlots());

  auto res_bx_view = res.BxView();
  auto res_ax_view = res.AxView();
  mod_switch_handlers_.at(level).ModDownAndRescale(res_bx_view,
                                                   accum.BxConstView());
  mod_switch_handlers_.at(level).ModDownAndRescale(res_ax_view,
                                                   accum.AxConstView());
}

template <typename word>
void Context<word>::Rescale(Ct &res, const Ct &a) const {
  NvtxScope _nv("op: Rescale");
  if (&res == &a) {
    Warn("Rescale is not adequate for in-place operations");
    Ct temp;
    Rescale(temp, a);
    res = std::move(temp);
    return;
  }
  AssertTrue(a.GetNP().num_aux_ == 0,
             "Rescale is not supported for ciphertexts "
             "with p primes");

  int level = param_.NPToLevel(a.GetNP());
  AssertTrue(level > 0, "Not enough q primes to rescale");

  if (a.HasRx()) {
    res.ModifyNP(param_.LevelToNP(level - 1));
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(param_.LevelToNP(level - 1));
  }
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale() / param_.GetRescalePrimeProd(level));

  auto res_bx_view = res.BxView();
  auto res_ax_view = res.AxView();

  mod_switch_handlers_.at(level).Rescale(res_bx_view, a.BxConstView());
  mod_switch_handlers_.at(level).Rescale(res_ax_view, a.AxConstView());
  if (a.HasRx()) {
    auto res_rx_view = res.RxView();
    mod_switch_handlers_.at(level).Rescale(res_rx_view, a.RxConstView());
  }
}

template <typename word>
void Context<word>::HRot(Ct &res, const Ct &a, const Evk &rot_key,
                         int rot_dist) const {
  NvtxScope _nv("ks: HRot");
  int num_slots = a.GetNumSlots();
  rot_dist %= num_slots;
  if (rot_dist < 0) rot_dist += num_slots;
  if (rot_dist == 0) {
    Warn("HRot is not necessary");
    Copy(res, a);
    return;
  }

  Ct tmp;
  MultKey(tmp, a, rot_key);
  Permute(res, tmp, rot_dist);
}

template <typename word>
void Context<word>::HConj(Ct &res, const Ct &a, const Evk &conj_key) const {
  AssertTrue(!param_.conjugate_invariant_,
             "HConj: conjugation acts trivially on the real subring, so this "
             "is a key switch that computes the identity -- use the "
             "ciphertext itself");
  Ct tmp;
  MultKey(tmp, a, conj_key);
  PermuteConjugate(res, tmp);
}

template <typename word>
void Context<word>::HRotAdd(Ct &res, const Ct &a, const Ct &b,
                            const Evk &rot_key, int rot_dist) const {
  NvtxScope _nv("ks: HRotAdd");
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  int num_slots = Max(a.GetNumSlots(), b.GetNumSlots());
  rot_dist %= num_slots;
  if (rot_dist < 0) rot_dist += num_slots;
  res.SetNumSlots(num_slots);
  res.SetScale(a.GetScale());
  if (rot_dist == 0) {
    Warn("HRotAdd is not necessary");
    Add(res, a, b);
    return;
  }

  Ct tmp;
  MultKey(tmp, a, rot_key);

  NPInfo np = a.GetNP();
  if (b.HasRx()) {
    res.ModifyNP(np);
    res.PrepareRx();
    CopyDeviceToDevice(res.rx_, b.rx_);
  } else {
    res.RemoveRx();
    res.ModifyNP(np);
  }

  auto res_temp = res.ViewVector();
  elem_handler_.PermuteAccum(
      res_temp, np, {rot_dist},
      {tmp.ConstViewVector(), b.ConstViewVector(0, true)});
}

template <typename word>
void Context<word>::HConjAdd(Ct &res, const Ct &a, const Ct &b,
                             const Evk &conj_key) const {
  AssertTrue(!param_.conjugate_invariant_,
             "HConjAdd: conjugation acts trivially on the real subring -- this "
             "is Add");
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  int num_slots = Max(a.GetNumSlots(), b.GetNumSlots());
  res.SetNumSlots(num_slots);
  res.SetScale(a.GetScale());

  Ct tmp;
  MultKey(tmp, a, conj_key);

  NPInfo np = a.GetNP();
  if (b.HasRx()) {
    res.ModifyNP(np);
    res.PrepareRx();
    CopyDeviceToDevice(res.rx_, b.rx_);
  } else {
    res.RemoveRx();
    res.ModifyNP(np);
  }

  auto res_temp = res.ViewVector();
  elem_handler_.PermuteAccum(
      res_temp, np, {-1}, {tmp.ConstViewVector(), b.ConstViewVector(0, true)});
}

template <typename word>
void Context<word>::HMult(Ct &res, const Ct &a, const Ct &b,
                          const Evk &mult_key, bool rescale) const {
  NvtxScope _nv("ks: HMult");
  Mult(res, a, b);
  if (rescale) {
    RelinearizeRescale(res, res, mult_key);
  } else {
    Relinearize(res, res, mult_key);
  }
}
template <typename word>
void Context<word>::MadUnsafe(Ct &res, const Ct &a, const Const &b) const {
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();

  NPInfo res_np = res.GetNP();
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Const");
  AssertTrue(res.GetNP().num_aux_ == 0 && a.GetNP().num_aux_ == 0,
             "MadUnsafe should be only used for q primes");
  AssertTrue(res.GetNumSlots() == a.GetNumSlots(),
             "MadUnsafe should be only used for the same number of slots");

  AssertSameScale(res, a.GetScale() * b.GetScale());

  if (a.HasRx()) {
    res.PrepareRx();
  }

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  if (a.HasRx()) {
    auto res_view = res.ViewVector();
    auto res_const_view = res.ConstViewVector();
    auto a_view = a.ConstViewVector(a_front_ignore);
    auto b_view = b.ConstView(b_front_ignore);
    elem_handler_.CAccum(res_view, res.GetNP(), {a_view, res_const_view},
                         {b_view});

    // elem_handler_.MadConstThree(res.bx_, res.ax_, res.aux_, a.bx_, a.ax_,
    // a.aux_, b.cx_, num_q_primes);
  } else {
    auto res_view = res.ViewVector(0, true);
    auto res_const_view = res.ConstViewVector(0, true);
    auto a_view = a.ConstViewVector(a_front_ignore, true);
    auto b_view = b.ConstView(b_front_ignore);
    elem_handler_.CAccum(res_view, res.GetNP(), {a_view, res_const_view},
                         {b_view});
    // elem_handler_.MadConstTwo(res.bx_, res.ax_, a.bx_, a.ax_, b.cx_,
    //                           num_q_primes);
  }
}

template <typename word>
void Context<word>::LevelDown(Ct &res, const Ct &a, int target_level) const {
  NvtxScope _nv("op: LevelDown");
  Ct mult_temp, next;
  int level = param_.NPToLevel(a.GetNP());
  const Ct *prev_res = &a;
  AssertTrue(level >= target_level, "Invalid target level for LevelDown");
  for (int i = level; i > target_level; i--) {
    Mult(mult_temp, *prev_res, level_down_consts_[i]);
    Rescale(next, mult_temp);
    prev_res = &next;
  }
  Copy(res, *prev_res);
}

template <typename word>
void Context<word>::AddLowerLevelsUntil(MultiLevelCiphertext<word> &ml_ct,
                                        int min_level) const {
  if (ml_ct.Exists(min_level)) {
    return;
  }
  int max_level = ml_ct.GetMaxLevel();
  int old_min_level = ml_ct.GetMinLevel();

  AssertTrue(min_level <= max_level && min_level >= 0,
             "AddLowerLevelsUntil: Invalid level " + std::to_string(min_level));

  Ct tmp;
  for (int i = old_min_level - 1; i >= min_level; i--) {
    ml_ct.AllocateLevel(i);
    Mult(tmp, ml_ct.AtLevel(i + 1),
         MultiLevelCiphertext<word>::GetLevelDownConst(param_, i + 1));
    Rescale(ml_ct.AtLevel(i), tmp);
  }
}

template class Context<uint32_t>;
template class Context<uint64_t>;

}  // namespace cheddar
