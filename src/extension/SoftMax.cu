#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/ChebyshevFit.h"
#include "extension/SoftMax.h"

namespace cheddar {

template <typename word>
std::unique_ptr<EvalPoly<word>> SoftMaxHandler<word>::MakeInvSqrt(double lo,
                                                                  double hi,
                                                                  int degree,
                                                                  int level) {
  // The polynomial lives on [-1, 1], so it is fitted to 1/sqrt(a v + b) with
  // a, b the half-width and centre of the calibrated norm interval. The
  // affine map itself is applied in Apply as a constant multiply and a
  // constant add; reinterpreting the scale instead would be free but hands
  // EvalPoly a non-canonical input scale, which is what silently broke
  // RMSNorm.
  const double a = 0.5 * (hi - lo);
  const double b = 0.5 * (hi + lo);
  auto coeffs = chebfit::Interpolate(
      [a, b](double v) { return 1.0 / std::sqrt(a * v + b); }, degree);

  const double in_scale = context_->param_.GetScale(level);
  const int degree_used =
      EvalPoly<word>(coeffs, level, in_scale, in_scale, true).GetPolyDegree();
  const int out_level = level - Log2Ceil(degree_used + 1);
  AssertTrue(out_level >= 0,
             "SoftMax: the inverse square root does not fit below its level");
  auto poly = std::make_unique<EvalPoly<word>>(
      coeffs, level, in_scale, context_->param_.GetScale(out_level), true);
  poly->Compile(context_);
  return poly;
}

template <typename word>
SoftMaxHandler<word>::SoftMaxHandler(ConstContextPtr<word> context,
                                     int num_keys, double range,
                                     int input_level, int num_iters,
                                     const std::vector<double> &norm_lo,
                                     const std::vector<double> &norm_hi,
                                     int exp_degree, int inv_sqrt_degree,
                                     int early_inv_sqrt_degree)
    : context_{std::move(context)},
      num_keys_{num_keys},
      range_{range},
      input_level_{input_level},
      num_iters_{num_iters},
      norm_lo_{norm_lo},
      norm_hi_{norm_hi} {
  AssertTrue(num_keys_ > 1 && (num_keys_ & (num_keys_ - 1)) == 0,
             "SoftMax: num_keys must be a power of two");
  AssertTrue(range_ > 0.0, "SoftMax: range must be positive");
  AssertTrue(num_iters_ >= 1, "SoftMax: at least one iteration");
  AssertTrue(static_cast<int>(norm_lo_.size()) == num_iters_ &&
                 static_cast<int>(norm_hi_.size()) == num_iters_,
             "SoftMax: one calibrated norm interval per iteration");
  num_slots_ = context_->param_.degree_ / 2;
  AssertTrue(num_slots_ % num_keys_ == 0,
             "SoftMax: the row length must divide the slot count");

  // One row occupies num_keys consecutive slots, so summing it is a
  // rotate-and-add over 1, 2, ... num_keys/2. The same sequence reduces and
  // broadcasts, so afterwards every slot already holds its own row's sum.
  for (int d = 1; d < num_keys_; d *= 2) rotation_distances_.push_back(d);

  // The exponential on [-M/2^k, 0], reached from v in [-1, 1] by
  // x' = M (v - 1) / 2^(k+1).
  const double half = range_ / std::pow(2.0, num_iters_ + 1);
  auto exp_coeffs = chebfit::Interpolate(
      [half](double v) { return std::exp(half * (v - 1.0)); }, exp_degree);
  const double in_scale = context_->param_.GetScale(input_level_);
  const int exp_used =
      EvalPoly<word>(exp_coeffs, input_level_, in_scale, in_scale, true)
          .GetPolyDegree();
  const int exp_out = input_level_ - Log2Ceil(exp_used + 1);
  AssertTrue(exp_out >= 0, "SoftMax: the exponential does not fit");
  exp_poly_ = std::make_unique<EvalPoly<word>>(
      exp_coeffs, input_level_, in_scale, context_->param_.GetScale(exp_out),
      true);
  exp_poly_->Compile(context_);

  // The inverse square roots are compiled lazily against the level each
  // iteration actually reaches, which depends on the ones before it. Walk the
  // levels the same way Apply will.
  //
  // Per iteration the main track spends: one square for the norm, one constant
  // multiply for the affine map, the polynomial, one multiply by the
  // normalisation, one square. [SYLPH] figure 2 keeps the first three off the
  // main track by bootstrapping the auxiliary track separately; without that
  // they land here, which is why the depth below exceeds section 4.3's 8.
  int level = exp_out - 1;  // the causal mask multiply
  for (int j = 0; j < num_iters_; j++) {
    const int norm_level = level - 1;   // square for ||y||^2
    const int poly_in = norm_level - 1; // constant multiply for the affine map
    const int degree =
        (j + 1 == num_iters_) ? inv_sqrt_degree : early_inv_sqrt_degree;
    inv_sqrt_.push_back(
        MakeInvSqrt(norm_lo_[j], norm_hi_[j], degree, poly_in));
    // the main track meets the normalisation at the polynomial's output level
    int used = 0;
    while ((1 << used) < degree + 1) used++;
    level = poly_in - used - 2;  // multiply by 1/||y||, then square
    AssertTrue(level >= 0, "SoftMax: does not fit below the input level");
  }
}

template <typename word>
std::vector<double> SoftMaxHandler<word>::PlainSoftMax(
    const std::vector<double> &x) const {
  const int d = static_cast<int>(x.size());
  std::vector<double> y(d);
  for (int i = 0; i < d; i++) {
    // the same argument the encrypted path sees: v in [-1, 1]
    y[i] = exp_poly_->PlainEvaluate(2.0 * x[i] / range_ + 1.0);
  }
  for (int j = 0; j < num_iters_; j++) {
    double sq = 0.0;
    for (double v : y) sq += v * v;
    const double a = 0.5 * (norm_hi_[j] - norm_lo_[j]);
    const double b = 0.5 * (norm_hi_[j] + norm_lo_[j]);
    const double r = inv_sqrt_[j]->PlainEvaluate((sq - b) / a);
    for (double &v : y) {
      v *= r;
      v = v * v;
    }
  }
  return y;
}

template <typename word>
void SoftMaxHandler<word>::Apply(Ct &res, const Ct &x_scaled,
                                 const std::vector<Complex> &causal_mask,
                                 const EvkMap<word> &evk_map) const {
  AssertTrue(static_cast<int>(causal_mask.size()) == num_slots_,
             "SoftMax: the mask must cover every slot");
  const auto &mult_key = evk_map.GetMultiplicationKey();

  // 1. y = exp(x'). The argument arrives already on [-1, 1]; see the header
  //    for why the mapping is the caller's and not this function's.
  Ct y;
  exp_poly_->Evaluate(context_, y, x_scaled, mult_key);

  // 2. Causal mask. The exponential is strictly positive, so masked positions
  //    have to be zeroed explicitly or they would join the norm and the
  //    output. Causality is public, hence a plaintext.
  {
    const int level = context_->param_.NPToLevel(y.GetNP());
    Pt mask_pt;
    context_->encoder_.Encode(mask_pt, level, context_->param_.GetScale(level),
                              causal_mask);
    Ct masked;
    context_->Mult(masked, y, mask_pt);
    context_->Rescale(y, masked);
  }

  Ct sq, rotated, r, scaled;
  for (int j = 0; j < num_iters_; j++) {
    // 3. ||y||_2^2 over the row. Euclidean, not the sum -- that is what makes
    //    the squaring below land on a vector that already sums to one.
    context_->HMult(sq, y, y, mult_key);
    for (int d : rotation_distances_) {
      // HRotAdd is res = (a << dist) + b, so res must not alias its inputs.
      context_->HRotAdd(rotated, sq, sq, evk_map.GetRotationKey(d), d);
      context_->Copy(sq, rotated);
    }

    // 4. Affine map onto the polynomial's domain. Mult(Ct, Const) does not
    //    rescale in Cheddar, so the scale is restored explicitly; that is the
    //    level this costs.
    const double a = 0.5 * (norm_hi_[j] - norm_lo_[j]);
    const double b = 0.5 * (norm_hi_[j] + norm_lo_[j]);
    const int sq_level = context_->param_.NPToLevel(sq.GetNP());
    Constant<word> inv_a;
    context_->encoder_.EncodeConstant(
        inv_a, sq_level, context_->param_.GetScale(sq_level), 1.0 / a);
    context_->Mult(scaled, sq, inv_a);
    context_->Rescale(sq, scaled);

    const int v_level = context_->param_.NPToLevel(sq.GetNP());
    Constant<word> shift;
    context_->encoder_.EncodeConstant(
        shift, v_level, context_->param_.GetScale(v_level), -b / a);
    context_->Add(sq, sq, shift);

    // 5. r = 1 / ||y||_2. Only the last of these has to be accurate: an error
    //    here multiplies the row by a constant, the squaring makes it a square
    //    constant, and the next normalisation divides it straight back out.
    inv_sqrt_[j]->Evaluate(context_, r, sq, mult_key);

    // 6. y <- (y * r)^2. Now sum(y) = 1 for the last iteration, with no
    //    further normalisation needed.
    const int r_level = context_->param_.NPToLevel(r.GetNP());
    Ct levelled;
    context_->LevelDown(levelled, y, r_level);
    context_->HMult(y, levelled, r, mult_key);
    context_->HMult(sq, y, y, mult_key);
    context_->Copy(y, sq);
  }
  context_->Copy(res, y);
}

template class SoftMaxHandler<uint32_t>;
template class SoftMaxHandler<uint64_t>;

}  // namespace cheddar
