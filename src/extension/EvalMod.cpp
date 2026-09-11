#include "extension/EvalMod.h"

#include <cmath>
#include <string>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
EvalMod<word>::EvalMod(ConstContextPtr<word> context,
                       const BootParameter &boot_param) {
  // Do not check here, checking handled by the BootContext
  int start_level = boot_param.GetEvalModStartLevel();
  int num_double_angle = boot_param.num_double_angle_;
  int double_angle_ratio = 1 << num_double_angle;
  int first_log_scale =
      std::log2(context->param_.GetRescalePrimeProd(start_level)) + 0.5;
  AssertTrue(first_log_scale <= 62, "Invalid eval_mod_scale");
  start_scale_ = (UINT64_C(1) << first_log_scale);

  // See the development notes for details
  int actual_K = (1 << boot_param.num_double_angle_) * boot_param.initial_K_;
  context->encoder_.EncodeConstant(initial_const_, start_level, start_scale_,
                                   -0.25 / actual_K);

  const auto &mod_coefficients = boot_param.mod_coefficients_;
  int mod_levels = Log2Ceil(mod_coefficients.size());

  double target_scale = start_scale_;
  for (int i = 0; i < mod_levels; i++) {
    target_scale = target_scale * target_scale /
                   context->param_.GetRescalePrimeProd(start_level - i);
    // The recursion's fixed point is the rescale width; a ladder whose
    // EvalMod levels differ in width runs away from it (2^60 over 58-bit
    // levels: 2^62, 2^66, 2^74, 2^90 -- measured as an output of exactly
    // zero on a landing preset with its two 30-bit spares at the top of
    // EvalMod, Doing.md 3.9). The next level's plaintexts cannot carry it.
    AssertTrue(target_scale <= std::ldexp(1.0, 62),
               "EvalMod: the polynomial's scale ran away to 2^" +
                   std::to_string(std::log2(target_scale)) + " at level " +
                   std::to_string(start_level - i - 1) +
                   "; every EvalMod level must rescale by (about) the "
                   "start scale's width");
  }

  mod_functions_.emplace_back(boot_param.mod_coefficients_, start_level,
                              start_scale_, target_scale, true);
  mod_functions_[0].Compile(context);

  // TODO(jongmin.kim): add support for other evalmod functions.

  // THE DOUBLE ANGLES ASSUME THE FIXED POINT, AND NOTHING WAS CHECKING IT.
  //
  // Each is a `2x^2 - 1` constructed with the SAME scale in and out (see the
  // `emplace_back` below: `target_scale` twice). A squaring at level L takes
  // the scale to `s^2 / prod(L)`, so "in equals out" holds only where
  // `prod(L) == s` -- the recursion's fixed point. The loop above walks the
  // POLYNOMIAL's `mod_levels` and stops, so the `num_double_angle` levels
  // under it were never examined: a ladder, or a climb, whose double-angle
  // levels sit off the band declared a scale it did not have and returned a
  // silently wrong answer. Measured on `ci16_42_k16_w60`: at the full climb
  // every EvalMod level rescales by 2^60 and p = 20.93; one level shorter,
  // the bottom EvalMod level is a 2^42.7 compute level -- a 17.3-bit drift,
  // entirely inside the double-angle range -- the 2^62 test above never sees
  // it, and the boot lands where it was asked to with p = **-518 bits**.
  //
  // The bound is measured, not guessed. Over every shipped boot preset the
  // worst per-level drift here is 3.61 bits (`ci16_35_land9e9v3`; the 2^42
  // family is 0.00 on every level, `ci16_35_land17c3e10v3` 1.93), so eight
  // bits clears them all with margin and still refuses the 17.3 above.
  constexpr double kMaxDoubleAngleDrift = 8.0;
  double sqrt2pi = std::pow(0.5 / M_PI, 1.0 / double_angle_ratio);
  for (int i = 0; i < num_double_angle; i++) {
    // 2 x^2 - 1
    sqrt2pi *= sqrt2pi;
    int double_angle_level = start_level - mod_levels - i;
    const double prod =
        context->param_.GetRescalePrimeProd(double_angle_level);
    const double drift = std::log2(prod / target_scale);
    AssertTrue(std::abs(drift) <= kMaxDoubleAngleDrift,
               "EvalMod: double angle " + std::to_string(i) + " sits at level " +
                   std::to_string(double_angle_level) + ", which rescales by 2^" +
                   std::to_string(std::log2(prod)) + " against the " +
                   "polynomial's output scale 2^" +
                   std::to_string(std::log2(target_scale)) + " -- a drift of " +
                   std::to_string(drift) +
                   " bits. A `2x^2 - 1` is declared scale-preserving, which "
                   "needs the level's rescale product to BE that scale; off it "
                   "the declared scale is not the data's and the bootstrap "
                   "returns a wrong answer without failing. Either the band "
                   "does not reach this level (a climb shorter than the band's "
                   "excess width) or the ladder's EvalMod levels are uneven.");
    double_angle_.emplace_back(context, 2, -sqrt2pi, double_angle_level,
                               target_scale, double_angle_level, target_scale);
    target_scale = target_scale * target_scale /
                   context->param_.GetRescalePrimeProd(double_angle_level);
  }
  end_scale_ = target_scale;
}

template <typename word>
void EvalMod<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                             const Ct &input, const Evk &mult_key) {
  context->Add(res, input, initial_const_);
  mod_functions_[0].Evaluate(context, res, res, mult_key);
  for (const auto &da : double_angle_) {
    da.Evaluate(context, res, res, res, mult_key);
  }
}

template <typename word>
void EvalMod<word>::EvaluateBatch(ConstContextPtr<word> context,
                                  CtBatch<word> &res, CtBatch<word> &input,
                                  const Evk &mult_key) {
  // Context::Add(res, input, initial_const_): the constant on every b part.
  {
    AssertTrue(input.np_ == initial_const_.GetNP(),
               "EvalMod::EvaluateBatch: NP mismatch");
    context->AssertSameScale(input.scale_, initial_const_.GetScale());
    std::vector<DvView<word>> dst{input.ViewVector().at(0)};
    std::vector<DvConstView<word>> src{input.ConstViewVector().at(0)};
    context->elem_handler_.AddConstBatchCt(
        dst, input.np_, src, initial_const_.ConstView(), input.batch_,
        input.CtStride(), input.CtStride());
  }
  mod_functions_[0].EvaluateBatch(context, res, input, mult_key);
  for (auto &da : double_angle_) {
    da.EvaluateBatch(context, res, res, res, mult_key);
  }
}

template <typename word>
int EvalMod<word>::GetEvalModPolyDegree(int poly_index /*= 0*/) const {
  return mod_functions_.at(poly_index).GetPolyDegree();
}

template <typename word>
int EvalMod<word>::GetNumDoubleAngle() const {
  return double_angle_.size();
}

template class EvalMod<uint32_t>;
template class EvalMod<uint64_t>;

}  // namespace cheddar