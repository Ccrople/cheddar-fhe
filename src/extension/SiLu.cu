#include <cmath>
#include <map>

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
                               int input_level, int degree,
                               bool zero_at_origin)
    : context_{std::move(context)}, range_{range}, input_level_{input_level} {
  AssertTrue(range_ > 0.0, "SiLu: range must be positive");
  AssertTrue(degree > 0, "SiLu: degree must be positive");

  // The polynomial lives on [-1, 1] and is fitted to SiLU(range * v), so its
  // output is already SiLU(x) with nothing to undo. What it needs is an input
  // of x / range, which Apply takes as given -- see there for why.
  const double r = range_;
  auto coeffs = chebfit::Interpolate([r](double v) { return SiLu(r * v); },
                                     degree);

  if (zero_at_origin) {
    // A BAND PLAN EVALUATES EVERY FIT ON THE MASKED INPUT, so a channel
    // outside band `b` feeds `b`'s polynomial a zero -- and collects `p_b(0)`
    // for its trouble, once per band it is not in. `SiLU(0)` is 0, but the
    // interpolant's `p(0)` is only 0 to within the fit error, so seven other
    // bands put seven fit errors on every slot. That is the bug the BERT
    // branch paid a GPU run for (Doing.md 8.8: "a band plan collects every
    // OTHER band's p(0) ... +0.29 a slot, 2^+6.7 in the crypto -- fixed by
    // constraining the fit to p(0) = 0").
    //
    // The constraint is one subtraction and costs at most one more fit error:
    // shifting `p` by `-p(0)` moves it by `|p(0)| <= eps` everywhere.
    //
    // It has to be taken on the polynomial the EVALUATOR runs, not on the one
    // that was fitted. `EvalPoly` zeroes every coefficient under
    // `kZeroCoeffThreshold` (1e-9) -- once on the whole vector and again on
    // each half at every split of its tree -- so a `p(0)` made exactly zero on
    // the fitted coefficients comes back at ~1e-9 (measured on the A100: the
    // first `SiLuBandPlanSumsToSiLu` run, 1.30e-09 at a 1e-12 bar). So the
    // tree `Compile` will build is built here, with the same trim, margin and
    // baby threshold, and read at 0, where the Chebyshev basis is exact:
    // `T_k(0)` is 0 for odd `k` and alternates +1, -1 for even `k`.
    //
    // `c_0` enters that tree with weight exactly one -- the Chebyshev fold
    // `low[split - i] -= high[i]` runs over `i >= 1` and never reaches index
    // 0, and the leaf reads it against `T_0 = 1` -- so ONE subtraction of the
    // tree's own `p(0)` makes it zero to a rounding.
    std::vector<double> trimmed = coeffs;
    while (!trimmed.empty() &&
           std::abs(trimmed.back()) < kZeroCoeffThreshold) {
      trimmed.pop_back();
    }
    for (double &c : trimmed) {
      if (std::abs(c) < kZeroCoeffThreshold) c = 0.0;
    }
    const int deg = static_cast<int>(trimmed.size()) - 1;
    AssertTrue(deg >= 2, "SiLu: the fit trims below degree 2");
    const int levels = Log2Ceil(deg + 1);
    EvalPolyNode<word> tree(trimmed, levels, 1 << DivCeil(levels, 2),
                            /*chebyshev=*/true);
    std::map<int, double> at_zero;
    for (int k = 0; k <= deg; k++) {
      at_zero[k] = (k % 2 != 0) ? 0.0 : ((k % 4 == 0) ? 1.0 : -1.0);
    }
    coeffs[0] -= tree.PlainEvaluate(at_zero);
  }

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
