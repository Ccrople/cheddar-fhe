#include <cmath>

#include "extension/Profile.h"
#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/ChebyshevFit.h"
#include "extension/SiLu.h"

namespace cheddar {

namespace {

double SiLu(double x) { return x / (1.0 + std::exp(-x)); }

}  // namespace

template <typename word>
SiLuHandler<word>::SiLuHandler(ConstContextPtr<word> context, double range,
                               int input_level, int degree)
    : context_{std::move(context)}, range_{range}, input_level_{input_level} {
  AssertTrue(range_ > 0.0, "SiLu: range must be positive");
  AssertTrue(degree > 0, "SiLu: degree must be positive");

  // The polynomial lives on [-1, 1] and is fitted to SiLU(range * v), so its
  // output is already SiLU(x) with nothing to undo. What it needs is an input
  // of x / range, which Apply takes as given -- see there for why.
  const double r = range_;
  auto coeffs = chebfit::Interpolate([r](double v) { return SiLu(r * v); },
                                     degree);

  // The output scale has to be the canonical scale of the level the polynomial
  // *lands on*, not of the level it starts from. EvalPoly takes target_scale_
  // on trust: EvalPoly.cpp:843 sets the result to it with no check against the
  // parameter set. Under grafting the two are not the same number --
  // scale_[i] = sqrt(scale_[i-1] * prod_i) drifts, and across the five levels a
  // degree-31 Chebyshev tree consumes on bootparam_30 the gap is 0.84%. The
  // result would still decode correctly, because the declared scale matches the
  // content, but it would be non-canonical for its level and the next Add
  // against anything else would fail with "Scale mismatch".
  //
  // Ask EvalPoly for the degree rather than assume it: the tree is built from
  // whatever coefficients survive, and level_consumption is
  // Log2Ceil(GetPolyDegree() + 1) (EvalPoly.cpp:801, 817).
  const double in_scale = context_->param_.GetScale(input_level_);
  const int degree_used =
      EvalPoly<word>(coeffs, input_level_, in_scale, in_scale, true)
          .GetPolyDegree();
  const int out_level = input_level_ - Log2Ceil(degree_used + 1);
  AssertTrue(out_level >= 0, "SiLu: the polynomial does not fit below the "
                             "input level");
  poly_ = std::make_unique<EvalPoly<word>>(
      coeffs, input_level_, in_scale, context_->param_.GetScale(out_level),
      /*chebyshev=*/true);
  poly_->Compile(context_);
}

template <typename word>
double SiLuHandler<word>::PlainSiLu(double x) const {
  return poly_->PlainEvaluate(x / range_);
}

template <typename word>
void SiLuHandler<word>::Apply(Ct &res, const Ct &normalised_x,
                              const EvkMap<word> &evk_map) const {
  NvtxScope _nv("silu: Apply");
  // The input must already be x / range. Reinterpreting the scale here would
  // be free but would hand EvalPoly a non-canonical input scale, which is
  // exactly what silently broke RMSNorm: the same coefficients were exact in
  // the clear and wrong by up to 29% encrypted, with the error growing along
  // the argument. Doing it with a constant multiply instead would be correct
  // but would cost a level.
  //
  // Neither is necessary. SiLU is always preceded by the gate projection, and
  // 1 / range folds into that projection's plaintext weight matrix for free.
  poly_->Evaluate(context_, res, normalised_x,
                  evk_map.GetMultiplicationKey());
}

template class SiLuHandler<uint32_t>;
template class SiLuHandler<uint64_t>;

}  // namespace cheddar
