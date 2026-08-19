#include <cmath>

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

  const double scale = context_->param_.GetScale(input_level_);
  poly_ = std::make_unique<EvalPoly<word>>(coeffs, input_level_, scale, scale,
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
