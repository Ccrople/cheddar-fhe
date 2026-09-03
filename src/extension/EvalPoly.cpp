#include "extension/EvalPoly.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace {

using namespace cheddar;

template <typename T>
void RemoveZero(std::vector<T> &coefficients) {
  while (!coefficients.empty() &&
         Abs(coefficients.back()) < kZeroCoeffThreshold) {
    coefficients.pop_back();
  }
  for (auto &c : coefficients) {
    if (Abs(c) < kZeroCoeffThreshold) {
      c = 0;
    }
  }
}

template <typename T>
void AddPolynomial(std::vector<T> res, const std::vector<T> &a,
                   const std::vector<T> &b) {
  res.resize(std::max(a.size(), b.size()));
  std::fill(res.begin(), res.end(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    res[i] += a[i];
  }
  for (size_t i = 0; i < b.size(); ++i) {
    res[i] += b[i];
  }
}

template <typename T, typename C>
void AddPolynomial(std::vector<T> res, const std::vector<T> &a, const C &b) {
  res.resize(a.size());
  std::fill(res.begin(), res.end(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    res[i] += a[i];
  }
  res[0] += b;
}

template <typename T>
void SubPolynomial(std::vector<T> res, const std::vector<T> &a,
                   const std::vector<T> &b) {
  res.resize(std::max(a.size(), b.size()));
  std::fill(res.begin(), res.end(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    res[i] += a[i];
  }
  for (size_t i = 0; i < b.size(); ++i) {
    res[i] -= b[i];
  }
}

template <typename T, typename C>
void SubPolynomial(std::vector<T> res, const std::vector<T> &a, const C &b) {
  res.resize(a.size());
  std::fill(res.begin(), res.end(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    res[i] += a[i];
  }
  res[0] -= b;
}

template <typename T, typename C>
void SubPolynomial(std::vector<T> res, const C &b, const std::vector<T> &a) {
  res.resize(a.size());
  std::fill(res.begin(), res.end(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    res[i] -= a[i];
  }
  res[0] += b;
}

template <typename T>
void MultiplyPolynomial(std::vector<T> res, const std::vector<T> &a,
                        const std::vector<T> &b) {
  res.resize(a.size() + b.size() - 1);
  std::fill(res.begin(), res.end(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < b.size(); ++j) {
      res[i + j] += a[i] * b[j];
    }
  }
}

template <typename T, typename C>
void MultiplyPolynomial(std::vector<T> res, const std::vector<T> &a,
                        const C &b) {
  res.resize(a.size());
  std::fill(res.begin(), res.end(), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    res[i] = a[i] * b;
  }
}

}  // namespace

namespace cheddar {

// ------------------------ AXYPBZ ------------------------

template <typename word>
void AXYPBZ<word>::AssertSameLevelAndScale(ConstContextPtr<word> context,
                                           const Ct &ct, int level,
                                           double scale) const {
  AssertTrue(ct.GetNP() == context->param_.LevelToNP(level), "Level mismatch");
  context->AssertSameScale(ct, scale);
}

template <typename word>
AXYPBZ<word>::AXYPBZ(ConstContextPtr<word> context, int a, double b,
                     int x_level, double x_scale, int y_level, double y_scale,
                     int z_level, double z_scale)
    : has_z_{true},
      x_level_{x_level},
      x_scale_{x_scale},
      y_level_{y_level},
      y_scale_{y_scale},
      z_level_{z_level},
      z_scale_{z_scale} {
  double working_scale = x_scale * y_scale;
  double b_scale = working_scale / z_scale;
  int working_level = Min(x_level, y_level, z_level);
  AssertTrue(working_level >= 1, "AXYPBZ: Invalid levels");
  AssertTrue(context->IsMultUnsafeCompatible(x_level, working_level) &&
                 context->IsMultUnsafeCompatible(y_level, working_level),
             "AXYPBZ: Invalid levels");
  if (b == 0.0) {
    has_b_ = false;
  } else {
    has_b_ = true;
    AssertTrue(context->IsMultUnsafeCompatible(z_level, working_level),
               "AXYPBZ: Invalid levels");
    context->encoder_.EncodeConstant(b_, working_level, b_scale, b);
  }

  AssertTrue(a != 0, "AXYPBZ: a should not be 0");
  if (a == 1) {
    has_a_ = false;
  } else {
    has_a_ = true;
    context->encoder_.EncodeConstant(a_, working_level, 1.0, a);
  }

  final_level_ = working_level - 1;
  final_scale_ =
      working_scale / context->param_.GetRescalePrimeProd(working_level);
}

// a * x * y + b
template <typename word>
AXYPBZ<word>::AXYPBZ(ConstContextPtr<word> context, int a, double b,
                     int x_level, double x_scale, int y_level, double y_scale)
    : has_z_{false},
      x_level_{x_level},
      x_scale_{x_scale},
      y_level_{y_level},
      y_scale_{y_scale},
      z_level_{0},
      z_scale_{0} {
  double working_scale = x_scale * y_scale;
  int working_level = Min(x_level, y_level);
  AssertTrue(working_level >= 1, "AXYPBZ: Invalid levels");
  AssertTrue(context->IsMultUnsafeCompatible(x_level, working_level) &&
                 context->IsMultUnsafeCompatible(y_level, working_level),
             "AXYPBZ: Invalid levels");
  if (b == 0.0) {
    has_b_ = false;
  } else {
    has_b_ = true;
    context->encoder_.EncodeConstant(b_, working_level, working_scale, b);
  }

  AssertTrue(a != 0, "AXYPBZ: a should not be 0");
  if (a == 1) {
    has_a_ = false;
  } else {
    has_a_ = true;
    context->encoder_.EncodeConstant(a_, working_level, 1.0, a);
  }

  final_level_ = working_level - 1;
  final_scale_ =
      working_scale / context->param_.GetRescalePrimeProd(working_level);
}

template <typename word>
void AXYPBZ<word>::Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &x,
                            const Ct &y, const Ct &z,
                            const Evk &mult_key) const {
  AssertTrue(has_z_, "Z should be provided for AXYPBZ with Z");
  AssertTrue(!x.HasRx() && !y.HasRx() && !z.HasRx(),
             "AXYPBZ: Relinearization required");
  AssertSameLevelAndScale(context, x, x_level_, x_scale_);
  AssertSameLevelAndScale(context, y, y_level_, y_scale_);
  AssertSameLevelAndScale(context, z, z_level_, z_scale_);

  Ct tmp1;
  if (has_a_) {  // a != 1
    context->MultUnsafe(tmp1, x, a_, final_level_ + 1);
    context->MultUnsafe(tmp1, tmp1, y, final_level_ + 1);
  } else {  // a == 1
    context->MultUnsafe(tmp1, x, y, final_level_ + 1);
  }

  if (has_b_) {
    std::vector<DvView<word>> dst{tmp1.BxView(), tmp1.AxView()};
    std::vector<DvConstView<word>> tmp1_view = {tmp1.BxConstView(),
                                                tmp1.AxConstView()};
    NPInfo np = tmp1.GetNP();
    NPInfo z_np = z.GetNP();
    int ter_diff = z_np.num_ter_ - np.num_ter_;
    AssertTrue(ter_diff >= 0, "AXYPBZ: Invalid levels");
    std::vector<DvConstView<word>> z_view = z.ConstViewVector(ter_diff);
    context->elem_handler_.CAccum(dst, np, {z_view, tmp1_view},
                                  {b_.ConstView()});
    tmp1.SetNumSlots(Max(tmp1.GetNumSlots(), z.GetNumSlots()));
  }  // else, b == 0.0
  context->RelinearizeRescale(res, tmp1, mult_key);

  res.SetScale(final_scale_);
}

template <typename word>
void AXYPBZ<word>::Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &x,
                            const Ct &y, const Evk &mult_key) const {
  AssertTrue(!has_z_, "Z should not be provided for AXYPBZ without Z");
  AssertTrue(!x.HasRx() && !y.HasRx(), "AXYPBZ: Relinearization required");
  AssertSameLevelAndScale(context, x, x_level_, x_scale_);
  AssertSameLevelAndScale(context, y, y_level_, y_scale_);

  Ct tmp1;
  if (has_a_) {  // a != 1
    context->MultUnsafe(tmp1, x, a_, final_level_ + 1);
    context->MultUnsafe(tmp1, tmp1, y, final_level_ + 1);
  } else {  // a == 0
    context->MultUnsafe(tmp1, x, y, final_level_ + 1);
  }

  if (has_b_) {  // b != 0.0
    context->Add(tmp1, tmp1, b_);
  }  // else, b == 0.0
  context->RelinearizeRescale(res, tmp1, mult_key);

  res.SetScale(final_scale_);
}

// --------------------------------------------------------

// ------------------------ BasisMap ------------------------

template <typename word>
BasisMap<word>::BasisMap(int input_level, double input_scale, EvalPolyType type,
                         bool chebyshev)
    : input_level_(input_level),
      input_scale_(input_scale),
      type_(type),
      chebyshev_(chebyshev) {}

template <typename word>
int BasisMap<word>::SplitBaseDegree(int base_degree) const {
  AssertTrue(base_degree >= 2,
             "Invalid base degree " + std::to_string(base_degree));
  int split = 0;
  if (IsPowOfTwo(base_degree)) {
    split = base_degree / 2;
  } else if (type_ == EvalPolyType::kEven) {
    split = 1 << Log2Floor(base_degree);
  } else {
    split = (1 << (Log2Floor(base_degree))) - 1;
  }
  return Max(split, base_degree - split);
}

template <typename word>
std::pair<int, double> BasisMap<word>::GetBaseLevelAndScale(
    int base_degree) const {
  AssertTrue(Exists(base_degree), "BasisMap: base_degree does not exist.");
  int level;
  double scale;
  if (base_degree == 1) {
    level = input_level_;
    scale = input_scale_;
  } else {
    level = basis_eval_.at(base_degree).final_level_;
    scale = basis_eval_.at(base_degree).final_scale_;
  }

  return {level, scale};
}

template <typename word>
bool BasisMap<word>::Exists(int base_degree) const {
  return (base_degree == 1) ||
         basis_eval_.find(base_degree) != basis_eval_.end();
}

template <typename word>
void BasisMap<word>::AddBase(ConstContextPtr<word> context, int base_degree) {
  AssertTrue(base_degree >= 2,
             "AddBase: base_degree should be at least 2, given " +
                 std::to_string(base_degree));
  int left_degree = SplitBaseDegree(base_degree);
  int right_degree = base_degree - left_degree;
  // left_degree >= right_degree always holds

  if (!Exists(left_degree)) {
    AddBase(context, left_degree);
  }
  if (!Exists(right_degree)) {
    AddBase(context, right_degree);
  }

  auto [left_level, left_scale] = GetBaseLevelAndScale(left_degree);
  auto [right_level, right_scale] = GetBaseLevelAndScale(right_degree);
  int working_level = Min(left_level, right_level);
  // Find the highest level that is compatible with the working level
  while (!context->IsMultUnsafeCompatible(left_level, working_level)) {
    left_level -= 1;
  }
  // Find the highest level that is compatible with the working level
  while (!context->IsMultUnsafeCompatible(right_level, working_level)) {
    right_level -= 1;
  }

  if (chebyshev_) {  // Chebyshev basis
    int sub_degree = left_degree - right_degree;
    if (sub_degree == 0) {
      basis_eval_.try_emplace(base_degree, context, 2, -1, left_level,
                              left_scale, right_level, right_scale);
    } else {
      if (!Exists(sub_degree)) {
        AddBase(context, sub_degree);
      }
      auto [sub_level, sub_scale] = GetBaseLevelAndScale(sub_degree);
      while (!context->IsMultUnsafeCompatible(sub_level, working_level)) {
        sub_level -= 1;
      }
      basis_eval_.try_emplace(base_degree, context, 2, -1, left_level,
                              left_scale, right_level, right_scale, sub_level,
                              sub_scale);
    }
  } else {  // Normal basis
    basis_eval_.try_emplace(base_degree, context, 1, 0, left_level, left_scale,
                            right_level, right_scale);
  }
}

template <typename word>
void BasisMap<word>::Evaluate(ConstContextPtr<word> context,
                              std::map<int, MLCt> &res,
                              const Evk &mult_key) const {
  AssertTrue(!basis_eval_.empty(), "BasisMap: basis_eval_ is empty.");

  for (const auto &[base_degree, eval] : basis_eval_) {
    int left_degree = SplitBaseDegree(base_degree);
    int right_degree = base_degree - left_degree;
    // left_degree >= right_degree always holds
    int sub_degree = left_degree - right_degree;
    Ct new_base;

    int left_level = eval.x_level_;
    int right_level = eval.y_level_;
    context->AddLowerLevelsUntil(res.at(left_degree), left_level);
    context->AddLowerLevelsUntil(res.at(right_degree), right_level);

    const Ct &left = res.at(left_degree).AtLevel(left_level);
    const Ct &right = res.at(right_degree).AtLevel(right_level);
    if (chebyshev_ && sub_degree != 0) {
      int sub_level = eval.z_level_;
      context->AddLowerLevelsUntil(res.at(sub_degree), sub_level);
      const Ct &sub = res.at(sub_degree).AtLevel(sub_level);
      eval.Evaluate(context, new_base, left, right, sub, mult_key);
    } else {
      eval.Evaluate(context, new_base, left, right, mult_key);
    }
    res.try_emplace(base_degree, context->param_, std::move(new_base));
  }
}

template <typename word>
void BasisMap<word>::PlainEvaluate(std::map<int, double> &res) const {
  AssertTrue(!basis_eval_.empty(), "BasisMap: basis_eval_ is empty.");
  for (const auto &[base_degree, eval] : basis_eval_) {
    int left_degree = SplitBaseDegree(base_degree);
    int right_degree = base_degree - left_degree;
    int sub_degree = Abs(left_degree - right_degree);
    res.try_emplace(base_degree, 0);
    double left = res.at(left_degree);
    double right = res.at(right_degree);
    res[base_degree] = left * right * (chebyshev_ ? 2 : 1);
    if (chebyshev_) {
      double sub = res.at(sub_degree);
      res[base_degree] -= sub;
    }
  }
}

// ----------------------------------------------------------

// ------------------------ EvalPolyNode ------------------------

template <typename word>
EvalPolyNode<word>::EvalPolyNode(const std::vector<double> &coefficients,
                                 int level_margin, int baby_threshold,
                                 bool chebyshev)
    : coefficients_{coefficients} {
  int degree = coefficients.size() - 1;

  int level_consumed_for_leaf_evaluation = Log2Ceil(degree) + 1;
  // Leaf creation
  if (degree < baby_threshold &&
      level_consumed_for_leaf_evaluation <= level_margin) {
    // Place holder for leaf constants;
    for (int i = 0; i <= degree; i++) {
      if (Abs(coefficients[i]) > kZeroCoeffThreshold) {
        leaf_constants_.try_emplace(i);
      }
    }
    return;
  }

  // Recursive creation
  split_degree_ = 1 << Log2Floor(degree);

  std::vector<double> low_coefficients(coefficients.begin(),
                                       coefficients.begin() + split_degree_);
  std::vector<double> high_coefficients(coefficients.begin() + split_degree_,
                                        coefficients.end());
  if (chebyshev) {
    int high_size = high_coefficients.size();
    for (int i = 1; i < high_size; i++) {
      low_coefficients[split_degree_ - i] -= high_coefficients[i];
      high_coefficients[i] *= 2;
    }
  }
  RemoveZero(low_coefficients);
  RemoveZero(high_coefficients);
  AssertFalse(high_coefficients.empty(),
              "Something went wrong during EvalPolyNode creation");

  // It is possible that low_coefficients are just a constant or 0
  // It is possible that high_coefficients are just a constant (but not 0)
  is_low_zero_ = low_coefficients.empty();
  is_low_constant_ = (low_coefficients.size() <= 1);
  is_high_constant_ = (high_coefficients.size() == 1);

  if (!is_low_constant_) {
    low_ = std::make_shared<EvalPolyNode<word>>(low_coefficients, level_margin,
                                                baby_threshold, chebyshev);
  }
  if (!is_high_constant_) {
    high_ = std::make_shared<EvalPolyNode<word>>(
        high_coefficients, level_margin - 1, baby_threshold, chebyshev);
  }
}

template <typename word>
bool EvalPolyNode<word>::IsLeaf() const {
  return high_ == nullptr && (!is_high_constant_);
}

template <typename word>
void EvalPolyNode<word>::CheckRequiredBasis(
    std::set<int> &required_bases) const {
  for (auto it = leaf_constants_.begin(); it != leaf_constants_.end(); it++) {
    required_bases.insert(it->first);
  }
  if (low_ != nullptr) low_->CheckRequiredBasis(required_bases);
  if (high_ != nullptr) high_->CheckRequiredBasis(required_bases);
  required_bases.insert(split_degree_);
}

template <typename word>
void EvalPolyNode<word>::Compile(ConstContextPtr<word> context,
                                 const BasisMap<word> &basis_eval,
                                 int target_level, double target_scale,
                                 bool do_rescale /*= true*/) {
  target_level_ = target_level;
  do_rescale_ = do_rescale;
  // The order of compilation
  int working_level = target_level;
  double working_scale = target_scale;
  if (do_rescale) {
    working_level = target_level + 1;
    working_scale =
        target_scale * context->param_.GetRescalePrimeProd(working_level);
  }

  if (is_low_constant_ && (!is_low_zero_)) {
    context->encoder_.EncodeConstant(low_constant_, working_level,
                                     working_scale, coefficients_[0]);
  } else if (low_ != nullptr) {
    // Low parts target higher scale (lazy rescaling)
    low_->Compile(context, basis_eval, working_level, working_scale, false);
  }
  if (is_high_constant_ || high_ != nullptr) {
    auto [_, split_scale] = basis_eval.GetBaseLevelAndScale(split_degree_);
    double high_scale = working_scale / split_scale;
    if (is_high_constant_) {
      // EncodeConstant takes (constant, level, scale, value). The level and
      // the scale were the other way round here, so a double scale was
      // narrowed to int and used as a level -- LevelToNP then indexed
      // level_config_ with INT_MIN. Every other call site in this file has the
      // right order; this branch only runs when a tree node's high part is a
      // constant, which no polynomial in the library or the bootstrap happens
      // to produce, so it had never been reached.
      context->encoder_.EncodeConstant(high_constant_, working_level,
                                       high_scale,
                                       coefficients_[split_degree_]);
    } else {
      high_->Compile(context, basis_eval, working_level, high_scale, true);
    }
  }

  // prepare leaf constants
  for (auto &[base_degree, constant] : leaf_constants_) {
    double value = coefficients_[base_degree];
    double scale = working_scale;
    if (base_degree != 0) {
      auto [_, base_scale] = basis_eval.GetBaseLevelAndScale(base_degree);
      scale /= base_scale;
    }
    context->encoder_.EncodeConstant(constant, working_level, scale, value);
  }
}

template <typename word>
void EvalPolyNode<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                                  std::map<int, MLCt> &basis,
                                  const Evk &mult_key) const {
  if (IsLeaf()) {
    // Evaluate is not for in-place operations
    EvaluateLeaf(context, res, basis, mult_key, false);
  } else {
    EvaluateMiddleNode(context, res, basis, mult_key);
  }
}

template <typename word>
void EvalPolyNode<word>::EvaluateMiddleNode(ConstContextPtr<word> context,
                                            Ct &res, std::map<int, MLCt> &basis,
                                            const Evk &mult_key) const {
  Ct tmp;
  Ct *accum = &res;
  int working_level = target_level_;
  if (do_rescale_) {
    working_level += 1;
    accum = &tmp;
  }
  AssertTrue(high_ != nullptr || is_high_constant_,
             "This is not a middle node");
  MLCt &ml_split = basis.at(split_degree_);
  int ml_split_level = ml_split.GetMaxLevel();
  while (!context->IsMultUnsafeCompatible(ml_split_level, working_level)) {
    ml_split_level -= 1;
  }
  context->AddLowerLevelsUntil(ml_split, ml_split_level);
  const Ct &split = ml_split.AtLevel(ml_split_level);

  if (high_ != nullptr) {
    high_->Evaluate(context, *accum, basis, mult_key);
    context->MultUnsafe(*accum, *accum, split, working_level);
  } else if (is_high_constant_) {
    context->MultUnsafe(*accum, split, high_constant_, working_level);
  } else {
    Fail("Something went wrong during middle node evaluation");
  }

  if (low_ != nullptr) {
    if (low_->IsLeaf()) {
      // This will perform inplace mad addition to accum (optimized)
      low_->EvaluateLeaf(context, *accum, basis, mult_key, true);
    } else {
      Ct tmp2;
      low_->Evaluate(context, tmp2, basis, mult_key);
      context->Add(*accum, *accum, tmp2);
    }
  } else if (is_low_constant_ && (!is_low_zero_)) {
    context->Add(*accum, *accum, low_constant_);
  }
  if (do_rescale_) {
    if (accum->HasRx()) {
      context->RelinearizeRescale(res, *accum, mult_key);
    } else {
      context->Rescale(res, *accum);
    }
  }
}

template <typename word>
void EvalPolyNode<word>::EvaluateLeaf(ConstContextPtr<word> context, Ct &res,
                                      std::map<int, MLCt> &basis,
                                      const Evk &mult_key, bool inplace) const {
  AssertFalse(do_rescale_ && inplace,
              "Rescale and inplace EvaluateLeaf is not compatible");
  AssertTrue(IsLeaf() && !leaf_constants_.empty(),
             "This is not a leaf node or leaf constants are not available.");

  Ct tmp;
  Ct *accum = &res;
  int working_level = target_level_;
  if (do_rescale_) {
    accum = &tmp;
    working_level += 1;
  }

  NPInfo np = context->param_.LevelToNP(working_level);
  std::vector<std::vector<DvConstView<word>>> ct_srcs;
  std::vector<DvConstView<word>> const_srcs;

  double scale = 0;
  int num_slots = 0;
  bool zero_const = false;
  for (const auto &[base_degree, constant] : leaf_constants_) {
    if (base_degree == 0) {
      zero_const = true;
      if (scale == 0) {
        scale = constant.GetScale();
      }
      // do nothing
    } else {
      MLCt &ml_ct = basis.at(base_degree);
      int ml_ct_level = ml_ct.GetMaxLevel();
      while (!context->IsMultUnsafeCompatible(ml_ct_level, working_level)) {
        ml_ct_level -= 1;
      }
      context->AddLowerLevelsUntil(ml_ct, ml_ct_level);
      const Ct &ct = ml_ct.AtLevel(ml_ct_level);
      int ter_diff = ct.GetNP().num_ter_ - np.num_ter_;
      AssertTrue(ter_diff >= 0, "Leaf evaluation level mismatch");
      ct_srcs.push_back(ct.ConstViewVector(ter_diff));
      const_srcs.push_back(constant.ConstView());
      num_slots = Max(num_slots, ct.GetNumSlots());
      if (scale == 0) {
        scale = ct.GetScale() * constant.GetScale();
      } else {
        context->AssertSameScale(scale, ct.GetScale() * constant.GetScale());
      }
    }
  }

  if (inplace) {
    AssertTrue(accum->GetNP() == np, "Leaf evaluation level mismatch");
    context->AssertSameScale(scale, accum->GetScale());
    accum->SetNumSlots(Max(num_slots, accum->GetNumSlots()));
    ct_srcs.push_back(accum->ConstViewVector(0, true));
  } else {
    accum->RemoveRx();
    accum->ModifyNP(np);
    accum->SetScale(scale);
    accum->SetNumSlots(num_slots);
  }
  std::vector<DvView<word>> dst = accum->ViewVector(0, true);
  context->elem_handler_.CAccum(dst, np, ct_srcs, const_srcs);
  if (zero_const) {
    context->Add(*accum, *accum, leaf_constants_.at(0));
  }

  if (do_rescale_) {
    context->Rescale(res, *accum);
  }
}

template <typename word>
double EvalPolyNode<word>::PlainEvaluate(std::map<int, double> &basis) const {
  if (IsLeaf()) {
    double accum = 0;
    for (const auto &[base_degree, constant] : leaf_constants_) {
      accum += basis[base_degree] * coefficients_[base_degree];
    }
    return accum;
  } else {
    double split = basis[split_degree_];
    double high_value = (is_high_constant_) ? coefficients_[split_degree_]
                                            : high_->PlainEvaluate(basis);
    double low_value = 0;
    if (!is_low_zero_)
      low_value =
          (is_low_constant_) ? coefficients_[0] : low_->PlainEvaluate(basis);
    return low_value + high_value * split;
  }
}

// --------------------------------------------------------------

// ------------------------ EvalPoly ------------------------

template <typename word>
EvalPoly<word>::EvalPoly(const std::vector<double> &coefficients,
                         int input_level, double input_scale,
                         double target_scale, bool chebyshev)
    : coefficients_{coefficients},
      type_{DetermineType()},
      chebyshev_{chebyshev},
      input_level_{input_level},
      input_scale_{input_scale},
      target_scale_{target_scale},
      basis_map_(input_level, input_scale, type_, chebyshev) {}

template <typename word>
EvalPolyType EvalPoly<word>::DetermineType() {
  RemoveZero(coefficients_);
  AssertTrue(GetPolyDegree() >= 2,
             "Use EvalPoly for at least >= 2 degree poly.");
  int degree = GetPolyDegree();
  bool odd_flag = true;
  bool even_flag = true;
  for (int i = 0; i <= degree; i += 2) {
    // if there is a non-zero even term, it is not odd
    if (coefficients_[i] != 0) {
      odd_flag = false;
    }
  }
  for (int i = 1; i <= degree; i += 2) {
    // if there is a non-zero odd term, it is not even
    if (coefficients_[i] != 0) {
      even_flag = false;
    }
  }
  AssertTrue(!(even_flag && odd_flag), "Invalid polynomial type.");
  EvalPolyType type;
  if (even_flag) {
    type = EvalPolyType::kEven;
  } else if (odd_flag) {
    type = EvalPolyType::kOdd;
  } else {
    type = EvalPolyType::kNormal;
  }
  return type;
}

template <typename word>
int EvalPoly<word>::GetPolyDegree() const {
  return coefficients_.size() - 1;
}

template <typename word>
void EvalPoly<word>::ConvertToChebyshevBasis() {
  if (chebyshev_) return;
  PreparePlainChebyshevBasis();

  std::vector<double> old_coefficients = std::move(coefficients_);
  std::vector<double> new_coefficients(GetPolyDegree() + 1, 0);
  for (int i = 0; i <= GetPolyDegree(); i++) {
    auto &cheby_base = plain_chebyshev_basis_[i];
    double quotient = old_coefficients[i] / cheby_base.back();
    new_coefficients[i] = quotient;

    std::vector<double> temp;
    MultiplyPolynomial(temp, cheby_base, quotient);
    SubPolynomial(old_coefficients, old_coefficients, temp);
  }
  coefficients_ = std::move(new_coefficients);
  RemoveZero(coefficients_);
  type_ = DetermineType();
  chebyshev_ = true;
}

template <typename word>
void EvalPoly<word>::ConvertToNormalBasis() {
  if (!chebyshev_) return;
  PreparePlainChebyshevBasis();
  std::vector<double> new_coefficients(GetPolyDegree() + 1, 0);
  for (int i = 0; i <= GetPolyDegree(); i++) {
    std::vector<double> temp;
    MultiplyPolynomial(temp, plain_chebyshev_basis_[i], coefficients_[i]);
    AddPolynomial(new_coefficients, new_coefficients, temp);
  }
  coefficients_ = std::move(new_coefficients);
  RemoveZero(coefficients_);
  type_ = DetermineType();
  chebyshev_ = false;
}

template <typename word>
void EvalPoly<word>::PreparePlainChebyshevBasis() {
  while (plain_chebyshev_basis_.size() < coefficients_.size()) {
    int new_degree = plain_chebyshev_basis_.size();
    plain_chebyshev_basis_.emplace_back(new_degree + 1);
    if (new_degree == 0) {
      plain_chebyshev_basis_[new_degree][0] = 1;
    } else if (new_degree == 1) {
      plain_chebyshev_basis_[new_degree][0] = 0;
      plain_chebyshev_basis_[new_degree][1] = 1;
    } else {
      // T_n(x) = 2xT_(n-1)(x) - T_(n-2)(x)
      MultiplyPolynomial(plain_chebyshev_basis_[new_degree],
                         plain_chebyshev_basis_[new_degree - 1],
                         plain_chebyshev_basis_[1]);
      MultiplyPolynomial(plain_chebyshev_basis_[new_degree],
                         plain_chebyshev_basis_[new_degree], 2);
      SubPolynomial(plain_chebyshev_basis_[new_degree],
                    plain_chebyshev_basis_[new_degree],
                    plain_chebyshev_basis_[new_degree - 2]);
    }
  }
}

template <typename word>
void EvalPoly<word>::Compile(ConstContextPtr<word> context) {
  // Construct main evaluation tree
  int level_consumption = Log2Ceil(GetPolyDegree() + 1);
  int baby_threshold = 1 << DivCeil(level_consumption, 2);

  tree_root_ = std::make_shared<EvalPolyNode<word>>(
      coefficients_, level_consumption, baby_threshold, chebyshev_);

  // Construct basis evaluation sequences
  std::set<int> required_base_degrees;
  tree_root_->CheckRequiredBasis(required_base_degrees);

  for (int deg : required_base_degrees) {
    if (deg == 0 || deg == 1) continue;
    basis_map_.AddBase(context, deg);
  }

  // Actual compilation of the tree
  int target_level = input_level_ - level_consumption;
  tree_root_->Compile(context, basis_map_, target_level, target_scale_);
}

template <typename word>
void EvalPoly<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                              const Ct &input, const Evk &mult_key) const {
  AssertTrue(tree_root_ != nullptr, "EvalPoly: not compiled.");
  NPInfo np = input.GetNP();
  AssertTrue(context->param_.NPToLevel(np) == input_level_,
             "EvalPoly: input level does not match the compiled level.");
  AssertTrue(np.num_aux_ == 0, "ModDown required before EvalPoly evaluation");
  AssertFalse(input.HasRx(),
              "Relinearization required before EvalPoly evaluation");
  context->AssertSameScale(input, input_scale_);

  std::map<int, MLCt> basis;
  // To prevent problems in in-place operations and also to simplify the code
  Ct input_tmp;
  context->Copy(input_tmp, input);
  basis.try_emplace(1, context->param_, std::move(input_tmp));

  basis_map_.Evaluate(context, basis, mult_key);
  tree_root_->Evaluate(context, res, basis, mult_key);
  // To avoid double calculation errors, manually set target scale
  context->AssertSameScale(res, target_scale_);
  res.SetScale(target_scale_);
}

template <typename word>
double EvalPoly<word>::PlainEvaluate(double input) const {
  AssertTrue(tree_root_ != nullptr, "EvalPoly: not compiled.");
  std::map<int, double> basis;
  basis.try_emplace(0, 1);
  basis.try_emplace(1, input);
  basis_map_.PlainEvaluate(basis);
  return tree_root_->PlainEvaluate(basis);
}

// ------------------------ The batched evaluation ------------------------
//
// EvaluateBatch walks the SAME compiled tree once over a CtBatch: every
// ciphertext of the batch follows the identical (level, scale) trajectory the
// serial evaluation takes, so the tree's constants, levels and dispatch are
// shared and only the payloads widen. Each helper below is one serial Context
// operation's batch twin -- the same kernels with the batch on gridDim.z (or
// `MultKeyBatch` for the key switches), so the words are the serial loop's.

// ------------------------ CtBatch ------------------------

template <typename word>
void CtBatch<word>::Allocate(const NPInfo &np, int batch, bool has_rx) {
  np_ = np;
  batch_ = batch;
  has_rx_ = has_rx;
  data_ = DeviceVector<word>(static_cast<int>(batch * CtStride()));
}

template <typename word>
std::vector<DvView<word>> CtBatch<word>::ViewVector(
    int np_front_ignore /*= 0*/, bool ignore_rx /*= false*/) {
  const int aux_size = np_.num_aux_ * np_.degree_;
  const int front = np_front_ignore * np_.degree_;
  const int size = static_cast<int>(PolyWords()) - front;
  const int num_poly = (has_rx_ && !ignore_rx) ? 3 : 2;
  std::vector<DvView<word>> views;
  for (int j = 0; j < num_poly; j++) {
    views.emplace_back(data_.data() + j * PolyWords() + front, size, aux_size);
  }
  return views;
}

template <typename word>
std::vector<DvConstView<word>> CtBatch<word>::ConstViewVector(
    int np_front_ignore /*= 0*/, bool ignore_rx /*= false*/) const {
  const int aux_size = np_.num_aux_ * np_.degree_;
  const int front = np_front_ignore * np_.degree_;
  const int size = static_cast<int>(PolyWords()) - front;
  const int num_poly = (has_rx_ && !ignore_rx) ? 3 : 2;
  std::vector<DvConstView<word>> views;
  for (int j = 0; j < num_poly; j++) {
    views.emplace_back(data_.data() + j * PolyWords() + front, size, aux_size);
  }
  return views;
}

template class CtBatch<uint32_t>;
template class CtBatch<uint64_t>;

namespace {

using namespace cheddar;

// Context::MultUnsafe(res, a, b_const, level) over a batch.
template <typename word>
void MultUnsafeConstBatch(ConstContextPtr<word> context, CtBatch<word> &res,
                          const CtBatch<word> &a, const Constant<word> &b,
                          int level) {
  const NPInfo res_np = context->param_.LevelToNP(level, 0);
  AssertTrue(res_np.IsSubsetOf(a.np_),
             "MultUnsafeConstBatch: incompatible levels");
  res.Allocate(res_np, a.batch_, a.has_rx_);
  res.num_slots_ = a.num_slots_;
  res.scale_ = a.scale_ * b.GetScale();

  const int a_front = a.np_.num_ter_ - res_np.num_ter_;
  const int b_front = b.GetNP().num_ter_ - res_np.num_ter_;
  auto dst = res.ViewVector();
  context->elem_handler_.MultConstBatchCt(
      dst, res_np, a.ConstViewVector(a_front), b.ConstView(b_front), a.batch_,
      res.CtStride(), a.CtStride());
}

// Context::MultUnsafe(res, x, y, level) (Ct x Ct) over a batch. The result
// is always a fresh buffer; the serial in-place case runs the same kernel on
// the same words.
template <typename word>
void MultUnsafeCtBatch(ConstContextPtr<word> context, CtBatch<word> &res,
                       const CtBatch<word> &a, const CtBatch<word> &b,
                       int level) {
  AssertFalse(a.has_rx_ || b.has_rx_,
              "MultUnsafeCtBatch: relinearization required");
  AssertTrue(a.np_.num_aux_ == 0 && b.np_.num_aux_ == 0,
             "MultUnsafeCtBatch: no aux primes");
  const NPInfo res_np = context->param_.LevelToNP(level, 0);
  AssertTrue(res_np.IsSubsetOf(a.np_) && res_np.IsSubsetOf(b.np_),
             "MultUnsafeCtBatch: incompatible levels");
  res.Allocate(res_np, a.batch_, /*has_rx=*/true);
  res.num_slots_ = Max(a.num_slots_, b.num_slots_);
  res.scale_ = a.scale_ * b.scale_;

  const int a_front = a.np_.num_ter_ - res_np.num_ter_;
  const int b_front = b.np_.num_ter_ - res_np.num_ter_;
  auto dst = res.ViewVector();
  context->elem_handler_.TensorBatchCt(
      dst, res_np, a.ConstViewVector(a_front, true),
      b.ConstViewVector(b_front, true), a.batch_, res.CtStride(), a.CtStride(),
      b.CtStride());
}

// Context::Add(res, res, b_const) over a batch, in place: the constant lands
// on the b polynomial and everything else stays.
template <typename word>
void AddConstBatchInplace(ConstContextPtr<word> context, CtBatch<word> &res,
                          const Constant<word> &b) {
  AssertTrue(res.np_ == b.GetNP(), "AddConstBatchInplace: NP mismatch");
  context->AssertSameScale(res.scale_, b.GetScale());
  std::vector<DvView<word>> dst{res.ViewVector().at(0)};
  std::vector<DvConstView<word>> src{res.ConstViewVector().at(0)};
  context->elem_handler_.AddConstBatchCt(dst, res.np_, src, b.ConstView(),
                                         res.batch_, res.CtStride(),
                                         res.CtStride());
}

// Context::Rescale(res, a) over a batch (no rx): the two parts of every
// ciphertext are contiguous, so the 2 * batch polynomials go as one strided
// group through ModSwitchHandler::RescaleBatch.
template <typename word>
void RescaleBatchOp(ConstContextPtr<word> context, CtBatch<word> &res,
                    const CtBatch<word> &a) {
  AssertFalse(a.has_rx_, "RescaleBatchOp: written for relinearized batches");
  AssertTrue(a.np_.num_aux_ == 0, "RescaleBatchOp: no aux primes");
  const int level = context->param_.NPToLevel(a.np_);
  AssertTrue(level > 0, "RescaleBatchOp: not enough q primes");
  res.Allocate(context->param_.LevelToNP(level - 1), a.batch_, false);
  res.num_slots_ = a.num_slots_;
  res.scale_ = a.scale_ / context->param_.GetRescalePrimeProd(level);
  context->mod_switch_handlers_.at(level).RescaleBatch(
      res.data_.data(), static_cast<int>(res.PolyWords()), a.data_.data(),
      static_cast<int>(a.PolyWords()), 2 * a.batch_);
}

// Context::RelinearizeRescale(res, a, key) over a batch.
template <typename word>
void RelinearizeRescaleBatchOp(ConstContextPtr<word> context,
                               CtBatch<word> &res, const CtBatch<word> &a,
                               const EvaluationKey<word> &key) {
  AssertTrue(a.has_rx_, "RelinearizeRescaleBatchOp requires rx");
  const int level = context->param_.NPToLevel(a.np_);
  res.Allocate(context->param_.LevelToNP(level - 1), a.batch_, false);
  res.num_slots_ = a.num_slots_;
  res.scale_ = a.scale_ / context->param_.GetRescalePrimeProd(level);
  context->RelinearizeRescaleBatch(
      res.data_.data(), static_cast<int>(res.CtStride()), a.data_.data(),
      static_cast<int>(a.CtStride()), a.np_, key, a.batch_);
}

// Context::AddLowerLevelsUntil over a batch: the level-down constant multiply
// and the rescale, one batched launch each per level step.
template <typename word>
void AddLowerLevelsUntilBatch(ConstContextPtr<word> context,
                              MLCtBatch<word> &ml, int min_level) {
  if (ml.count(min_level) != 0) return;
  const int max_level = ml.rbegin()->first;
  const int old_min_level = ml.begin()->first;
  AssertTrue(min_level <= max_level && min_level >= 0,
             "AddLowerLevelsUntilBatch: Invalid level " +
                 std::to_string(min_level));
  for (int i = old_min_level - 1; i >= min_level; i--) {
    const CtBatch<word> &up = ml.at(i + 1);
    const Constant<word> &c =
        MultiLevelCiphertext<word>::GetLevelDownConst(context->param_, i + 1);
    CtBatch<word> tmp;
    tmp.Allocate(up.np_, up.batch_, false);
    tmp.num_slots_ = up.num_slots_;
    tmp.scale_ = up.scale_ * c.GetScale();
    auto dst = tmp.ViewVector();
    context->elem_handler_.MultConstBatchCt(dst, up.np_, up.ConstViewVector(),
                                            c.ConstView(), up.batch_,
                                            tmp.CtStride(), up.CtStride());
    CtBatch<word> down;
    RescaleBatchOp(context, down, tmp);
    ml.emplace(i, std::move(down));
  }
}

}  // namespace

// ------------------------ AXYPBZ (batched) ------------------------

template <typename word>
void AXYPBZ<word>::EvaluateBatch(ConstContextPtr<word> context,
                                 CtBatch<word> &res, const CtBatch<word> &x,
                                 const CtBatch<word> &y, const CtBatch<word> &z,
                                 const Evk &mult_key) const {
  AssertTrue(has_z_, "Z should be provided for AXYPBZ with Z");
  AssertTrue(!x.has_rx_ && !y.has_rx_ && !z.has_rx_,
             "AXYPBZ: Relinearization required");
  context->AssertSameScale(x.scale_, x_scale_);
  context->AssertSameScale(y.scale_, y_scale_);
  context->AssertSameScale(z.scale_, z_scale_);

  CtBatch<word> tmp1;
  if (has_a_) {  // a != 1
    CtBatch<word> tmp0;
    MultUnsafeConstBatch(context, tmp0, x, a_, final_level_ + 1);
    MultUnsafeCtBatch(context, tmp1, tmp0, y, final_level_ + 1);
  } else {  // a == 1
    MultUnsafeCtBatch(context, tmp1, x, y, final_level_ + 1);
  }

  if (has_b_) {
    const NPInfo np = tmp1.np_;
    const int ter_diff = z.np_.num_ter_ - np.num_ter_;
    AssertTrue(ter_diff >= 0, "AXYPBZ: Invalid levels");
    auto dst = tmp1.ViewVector(0, true);
    context->elem_handler_.CAccumBatchCt(
        dst, np, {z.ConstViewVector(ter_diff, true),
                  tmp1.ConstViewVector(0, true)},
        {b_.ConstView()}, tmp1.batch_, tmp1.CtStride(),
        {z.CtStride(), tmp1.CtStride()});
    tmp1.num_slots_ = Max(tmp1.num_slots_, z.num_slots_);
  }  // else, b == 0.0
  RelinearizeRescaleBatchOp(context, res, tmp1, mult_key);

  res.scale_ = final_scale_;
}

template <typename word>
void AXYPBZ<word>::EvaluateBatch(ConstContextPtr<word> context,
                                 CtBatch<word> &res, const CtBatch<word> &x,
                                 const CtBatch<word> &y,
                                 const Evk &mult_key) const {
  AssertTrue(!has_z_, "Z should not be provided for AXYPBZ without Z");
  AssertTrue(!x.has_rx_ && !y.has_rx_, "AXYPBZ: Relinearization required");
  context->AssertSameScale(x.scale_, x_scale_);
  context->AssertSameScale(y.scale_, y_scale_);

  CtBatch<word> tmp1;
  if (has_a_) {  // a != 1
    CtBatch<word> tmp0;
    MultUnsafeConstBatch(context, tmp0, x, a_, final_level_ + 1);
    MultUnsafeCtBatch(context, tmp1, tmp0, y, final_level_ + 1);
  } else {  // a == 1
    MultUnsafeCtBatch(context, tmp1, x, y, final_level_ + 1);
  }

  if (has_b_) {  // b != 0.0
    AddConstBatchInplace(context, tmp1, b_);
  }  // else, b == 0.0
  RelinearizeRescaleBatchOp(context, res, tmp1, mult_key);

  res.scale_ = final_scale_;
}

template class AXYPBZ<uint32_t>;
template class AXYPBZ<uint64_t>;

template <typename word>
void BasisMap<word>::EvaluateBatch(ConstContextPtr<word> context,
                                   std::map<int, MLCtBatch<word>> &res,
                                   const Evk &mult_key) const {
  AssertTrue(!basis_eval_.empty(), "BasisMap: basis_eval_ is empty.");

  for (const auto &[base_degree, eval] : basis_eval_) {
    int left_degree = SplitBaseDegree(base_degree);
    int right_degree = base_degree - left_degree;
    // left_degree >= right_degree always holds
    int sub_degree = left_degree - right_degree;
    CtBatch<word> new_base;

    int left_level = eval.x_level_;
    int right_level = eval.y_level_;
    AddLowerLevelsUntilBatch(context, res.at(left_degree), left_level);
    AddLowerLevelsUntilBatch(context, res.at(right_degree), right_level);

    const CtBatch<word> &left = res.at(left_degree).at(left_level);
    const CtBatch<word> &right = res.at(right_degree).at(right_level);
    if (chebyshev_ && sub_degree != 0) {
      int sub_level = eval.z_level_;
      AddLowerLevelsUntilBatch(context, res.at(sub_degree), sub_level);
      const CtBatch<word> &sub = res.at(sub_degree).at(sub_level);
      eval.EvaluateBatch(context, new_base, left, right, sub, mult_key);
    } else {
      eval.EvaluateBatch(context, new_base, left, right, mult_key);
    }
    const int new_level = context->param_.NPToLevel(new_base.np_);
    MLCtBatch<word> ml;
    ml.emplace(new_level, std::move(new_base));
    res.try_emplace(base_degree, std::move(ml));
  }
}

template class BasisMap<uint32_t>;
template class BasisMap<uint64_t>;

template <typename word>
void EvalPolyNode<word>::EvaluateBatch(ConstContextPtr<word> context,
                                       CtBatch<word> &res,
                                       std::map<int, MLCtBatch<word>> &basis,
                                       const Evk &mult_key) const {
  if (IsLeaf()) {
    EvaluateLeafBatch(context, res, basis, mult_key, false);
  } else {
    EvaluateMiddleNodeBatch(context, res, basis, mult_key);
  }
}

template <typename word>
void EvalPolyNode<word>::EvaluateMiddleNodeBatch(
    ConstContextPtr<word> context, CtBatch<word> &res,
    std::map<int, MLCtBatch<word>> &basis, const Evk &mult_key) const {
  CtBatch<word> tmp;
  CtBatch<word> *accum = &res;
  int working_level = target_level_;
  if (do_rescale_) {
    working_level += 1;
    accum = &tmp;
  }
  AssertTrue(high_ != nullptr || is_high_constant_,
             "This is not a middle node");
  MLCtBatch<word> &ml_split = basis.at(split_degree_);
  int ml_split_level = ml_split.rbegin()->first;
  while (!context->IsMultUnsafeCompatible(ml_split_level, working_level)) {
    ml_split_level -= 1;
  }
  AddLowerLevelsUntilBatch(context, ml_split, ml_split_level);
  const CtBatch<word> &split = ml_split.at(ml_split_level);

  if (high_ != nullptr) {
    high_->EvaluateBatch(context, *accum, basis, mult_key);
    CtBatch<word> product;
    MultUnsafeCtBatch(context, product, *accum, split, working_level);
    *accum = std::move(product);
  } else if (is_high_constant_) {
    MultUnsafeConstBatch(context, *accum, split, high_constant_,
                         working_level);
  } else {
    Fail("Something went wrong during middle node evaluation");
  }

  if (low_ != nullptr) {
    if (low_->IsLeaf()) {
      // In-place mad addition into accum, as the serial path does.
      low_->EvaluateLeafBatch(context, *accum, basis, mult_key, true);
    } else {
      CtBatch<word> tmp2;
      low_->EvaluateBatch(context, tmp2, basis, mult_key);
      // Context::Add(*accum, *accum, tmp2) over the batch.
      context->AssertSameScale(accum->scale_, tmp2.scale_);
      AssertTrue(accum->np_ == tmp2.np_, "EvaluateMiddleNodeBatch: NP mismatch");
      const bool rx_add = accum->has_rx_ && tmp2.has_rx_;
      accum->num_slots_ = Max(accum->num_slots_, tmp2.num_slots_);
      if (rx_add) {
        auto dst = accum->ViewVector();
        context->elem_handler_.AddBatchCt(
            dst, accum->np_, accum->ConstViewVector(), tmp2.ConstViewVector(),
            accum->batch_, accum->CtStride(), accum->CtStride(),
            tmp2.CtStride());
      } else if (!accum->has_rx_ && tmp2.has_rx_) {
        // The serial Add would hand tmp2's rx to the result.
        CtBatch<word> sum;
        sum.Allocate(accum->np_, accum->batch_, true);
        sum.num_slots_ = accum->num_slots_;
        sum.scale_ = accum->scale_;
        auto dst = sum.ViewVector(0, true);
        context->elem_handler_.AddBatchCt(
            dst, accum->np_, accum->ConstViewVector(0, true),
            tmp2.ConstViewVector(0, true), accum->batch_, sum.CtStride(),
            accum->CtStride(), tmp2.CtStride());
        cudaMemcpy2DAsync(
            sum.CtData(0) + 2 * sum.PolyWords(), sum.CtStride() * sizeof(word),
            tmp2.CtData(0) + 2 * tmp2.PolyWords(),
            tmp2.CtStride() * sizeof(word), sum.PolyWords() * sizeof(word),
            sum.batch_, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
        *accum = std::move(sum);
      } else {
        auto dst = accum->ViewVector(0, true);
        context->elem_handler_.AddBatchCt(
            dst, accum->np_, accum->ConstViewVector(0, true),
            tmp2.ConstViewVector(0, true), accum->batch_, accum->CtStride(),
            accum->CtStride(), tmp2.CtStride());
      }
    }
  } else if (is_low_constant_ && (!is_low_zero_)) {
    AddConstBatchInplace(context, *accum, low_constant_);
  }
  if (do_rescale_) {
    if (accum->has_rx_) {
      RelinearizeRescaleBatchOp(context, res, *accum, mult_key);
    } else {
      CtBatch<word> rescaled;
      RescaleBatchOp(context, rescaled, *accum);
      res = std::move(rescaled);
    }
  }
}

template <typename word>
void EvalPolyNode<word>::EvaluateLeafBatch(ConstContextPtr<word> context,
                                           CtBatch<word> &res,
                                           std::map<int, MLCtBatch<word>> &basis,
                                           const Evk &mult_key,
                                           bool inplace) const {
  AssertFalse(do_rescale_ && inplace,
              "Rescale and inplace EvaluateLeaf is not compatible");
  AssertTrue(IsLeaf() && !leaf_constants_.empty(),
             "This is not a leaf node or leaf constants are not available.");

  CtBatch<word> tmp;
  CtBatch<word> *accum = &res;
  int working_level = target_level_;
  if (do_rescale_) {
    accum = &tmp;
    working_level += 1;
  }

  NPInfo np = context->param_.LevelToNP(working_level);
  std::vector<std::vector<DvConstView<word>>> ct_srcs;
  std::vector<DvConstView<word>> const_srcs;
  std::vector<size_t> src_strides;

  double scale = 0;
  int num_slots = 0;
  int batch = 0;
  bool zero_const = false;
  for (const auto &[base_degree, constant] : leaf_constants_) {
    if (base_degree == 0) {
      zero_const = true;
      if (scale == 0) {
        scale = constant.GetScale();
      }
      // do nothing
    } else {
      MLCtBatch<word> &ml_ct = basis.at(base_degree);
      int ml_ct_level = ml_ct.rbegin()->first;
      while (!context->IsMultUnsafeCompatible(ml_ct_level, working_level)) {
        ml_ct_level -= 1;
      }
      AddLowerLevelsUntilBatch(context, ml_ct, ml_ct_level);
      const CtBatch<word> &ct = ml_ct.at(ml_ct_level);
      int ter_diff = ct.np_.num_ter_ - np.num_ter_;
      AssertTrue(ter_diff >= 0, "Leaf evaluation level mismatch");
      ct_srcs.push_back(ct.ConstViewVector(ter_diff, true));
      src_strides.push_back(ct.CtStride());
      const_srcs.push_back(constant.ConstView());
      num_slots = Max(num_slots, ct.num_slots_);
      batch = ct.batch_;
      if (scale == 0) {
        scale = ct.scale_ * constant.GetScale();
      } else {
        context->AssertSameScale(scale, ct.scale_ * constant.GetScale());
      }
    }
  }

  if (inplace) {
    AssertTrue(accum->np_ == np, "Leaf evaluation level mismatch");
    context->AssertSameScale(scale, accum->scale_);
    accum->num_slots_ = Max(num_slots, accum->num_slots_);
    ct_srcs.push_back(accum->ConstViewVector(0, true));
    src_strides.push_back(accum->CtStride());
  } else {
    accum->Allocate(np, batch, false);
    accum->scale_ = scale;
    accum->num_slots_ = num_slots;
  }
  std::vector<DvView<word>> dst = accum->ViewVector(0, true);
  context->elem_handler_.CAccumBatchCt(dst, np, ct_srcs, const_srcs,
                                       accum->batch_, accum->CtStride(),
                                       src_strides);
  if (zero_const) {
    AddConstBatchInplace(context, *accum, leaf_constants_.at(0));
  }

  if (do_rescale_) {
    RescaleBatchOp(context, res, *accum);
  }
}

template class EvalPolyNode<uint32_t>;
template class EvalPolyNode<uint64_t>;

template <typename word>
void EvalPoly<word>::EvaluateBatch(ConstContextPtr<word> context,
                                   CtBatch<word> &res,
                                   const CtBatch<word> &input,
                                   const Evk &mult_key) const {
  AssertTrue(tree_root_ != nullptr, "EvalPoly: not compiled.");
  AssertTrue(context->param_.NPToLevel(input.np_) == input_level_,
             "EvalPoly: input level does not match the compiled level.");
  AssertTrue(input.np_.num_aux_ == 0,
             "ModDown required before EvalPoly evaluation");
  AssertFalse(input.has_rx_,
              "Relinearization required before EvalPoly evaluation");
  context->AssertSameScale(input.scale_, input_scale_);

  std::map<int, MLCtBatch<word>> basis;
  // To prevent problems in in-place operations and also to simplify the code
  CtBatch<word> input_tmp;
  input_tmp.Allocate(input.np_, input.batch_, false);
  input_tmp.scale_ = input.scale_;
  input_tmp.num_slots_ = input.num_slots_;
  CopyDeviceToDevice(input_tmp.data_, input.data_);
  MLCtBatch<word> ml;
  ml.emplace(input_level_, std::move(input_tmp));
  basis.try_emplace(1, std::move(ml));

  basis_map_.EvaluateBatch(context, basis, mult_key);
  tree_root_->EvaluateBatch(context, res, basis, mult_key);
  // To avoid double calculation errors, manually set target scale
  context->AssertSameScale(res.scale_, target_scale_);
  res.scale_ = target_scale_;
}

template class EvalPoly<uint32_t>;
template class EvalPoly<uint64_t>;

}  // namespace cheddar