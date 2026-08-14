// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Cheby.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "oracle/Ops.h"

namespace oracle {
namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

double ChebyFit::Eval(double x) const {
  if (coeffs.empty()) return 0.0;
  // Map [lo, hi] -> [-1, 1].
  const double t = (2.0 * x - (hi + lo)) / (hi - lo);
  // Clenshaw.
  double b1 = 0.0, b2 = 0.0;
  for (int k = degree; k >= 1; --k) {
    const double b0 = 2.0 * t * b1 - b2 + coeffs[static_cast<size_t>(k)];
    b2 = b1;
    b1 = b0;
  }
  return t * b1 - b2 + coeffs[0];
}

ChebyFit FitChebyshev(const std::function<double(double)>& f, double lo,
                      double hi, int degree) {
  if (!(hi > lo)) throw std::invalid_argument("FitChebyshev: hi must exceed lo");
  if (degree < 0) throw std::invalid_argument("FitChebyshev: negative degree");
  ChebyFit fit;
  fit.lo = lo;
  fit.hi = hi;
  fit.degree = degree;
  const int n = degree + 1;
  std::vector<double> fx(static_cast<size_t>(n));
  std::vector<double> tnodes(static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    const double t = std::cos(kPi * (static_cast<double>(j) + 0.5) /
                              static_cast<double>(n));
    tnodes[static_cast<size_t>(j)] = t;
    const double x = 0.5 * (hi + lo) + 0.5 * (hi - lo) * t;
    fx[static_cast<size_t>(j)] = f(x);
  }
  fit.coeffs.assign(static_cast<size_t>(n), 0.0);
  for (int k = 0; k < n; ++k) {
    double acc = 0.0;
    for (int j = 0; j < n; ++j) {
      // T_k(t_j) = cos(k * acos(t_j)) = cos(k * pi * (j+0.5)/n)
      acc += fx[static_cast<size_t>(j)] *
             std::cos(kPi * static_cast<double>(k) *
                      (static_cast<double>(j) + 0.5) / static_cast<double>(n));
    }
    fit.coeffs[static_cast<size_t>(k)] = 2.0 * acc / static_cast<double>(n);
  }
  fit.coeffs[0] *= 0.5;
  (void)tnodes;
  return fit;
}

FitQuality MeasureFit(const std::function<double(double)>& f,
                      const ChebyFit& fit, int64_t eval_points) {
  if (eval_points < 3) eval_points = 3;
  FitQuality q;
  q.degree = fit.degree;
  q.eval_points = eval_points;
  q.depth = 0;
  {
    int d = fit.degree + 1;
    while ((1 << q.depth) < d) ++q.depth;
  }

  // First pass: the magnitude envelope of f. The peak sets the relative-error
  // floor; the peak-to-trough ratio is the interval's dynamic range, which is
  // a separate and often more binding constraint than the fit error.
  bool saw_pos = false, saw_neg = false, saw_zero = false;
  for (int64_t i = 0; i < eval_points; ++i) {
    const double x = fit.lo + (fit.hi - fit.lo) * static_cast<double>(i) /
                                  static_cast<double>(eval_points - 1);
    const double fx = f(x);
    const double v = std::fabs(fx);
    if (v > q.f_abs_max) q.f_abs_max = v;
    if (i == 0 || v < q.f_abs_min) q.f_abs_min = v;
    if (fx > 0.0) saw_pos = true;
    else if (fx < 0.0) saw_neg = true;
    else saw_zero = true;
  }
  q.single_signed = !(saw_zero || (saw_pos && saw_neg));
  q.dynamic_range = q.f_abs_min > 0.0
                        ? q.f_abs_max / q.f_abs_min
                        : std::numeric_limits<double>::infinity();
  const double floor_v = q.f_abs_max > 0.0 ? 1e-6 * q.f_abs_max : 1e-300;

  double sum_sq = 0.0;
  for (int64_t i = 0; i < eval_points; ++i) {
    const double x = fit.lo + (fit.hi - fit.lo) * static_cast<double>(i) /
                                  static_cast<double>(eval_points - 1);
    const double truth = f(x);
    const double err = fit.Eval(x) - truth;
    const double ae = std::fabs(err);
    if (ae > q.max_abs_err) {
      q.max_abs_err = ae;
      q.at_x_abs = x;
    }
    const double denom = std::fabs(truth) > floor_v ? std::fabs(truth) : floor_v;
    const double re = ae / denom;
    if (re > q.max_rel_err) {
      q.max_rel_err = re;
      q.at_x_rel = x;
    }
    sum_sq += err * err;
  }
  q.rms_err = std::sqrt(sum_sq / static_cast<double>(eval_points));
  q.err_over_peak = q.f_abs_max > 0.0 ? q.max_abs_err / q.f_abs_max
                                      : q.max_abs_err;
  return q;
}

const char* ToString(FitCriterion c) {
  switch (c) {
    case FitCriterion::kPointwiseRelative: return "pointwise relative";
    case FitCriterion::kRelativeToPeak: return "relative to peak";
  }
  return "unknown";
}

FitCriterion CriterionFor(const std::string& name) {
  // 1/sqrt and 1/x produce a SCALE FACTOR: the fit's relative error becomes
  // the answer's relative error at every magnitude, so the pointwise measure
  // is the honest one.
  if (name == "rsqrt" || name == "reciprocal")
    return FitCriterion::kPointwiseRelative;
  // exp feeds a normalising sum and SiLU feeds a wide contraction, so what
  // propagates is the absolute error at the scale of the largest terms. For
  // SiLU the pointwise measure is additionally meaningless: SiLU crosses zero,
  // and the pointwise ratio there is set by the location of the root rather
  // than by the quality of the fit.
  if (name == "exp" || name == "silu") return FitCriterion::kRelativeToPeak;
  return FitCriterion::kPointwiseRelative;
}

const std::vector<int>& CandidateDegrees() {
  static const std::vector<int> kDegrees = {3, 7, 15, 31, 63, 127};
  return kDegrees;
}

DegreeRecommendation RecommendDegree(const std::string& function_name,
                                     const std::function<double(double)>& f,
                                     double lo, double hi,
                                     double target_rel_err,
                                     bool interval_measured,
                                     const std::string& interval_source,
                                     double max_dynamic_range) {
  DegreeRecommendation r;
  r.function = function_name;
  r.interval_source = interval_source;
  r.lo = lo;
  r.hi = hi;
  r.target_rel_err = target_rel_err;
  r.interval_measured = interval_measured;
  r.criterion = CriterionFor(function_name);

  for (int d : CandidateDegrees()) {
    ChebyFit fit = FitChebyshev(f, lo, hi, d);
    FitQuality q = MeasureFit(f, fit);
    r.sweep.push_back(q);
    const double judged = r.criterion == FitCriterion::kPointwiseRelative
                              ? q.max_rel_err
                              : q.err_over_peak;
    if (!r.met_target && judged <= target_rel_err) {
      r.met_target = true;
      r.recommended_degree = d;
      r.achieved_rel_err = judged;
      r.depth = q.depth;
    }
  }
  // The dynamic-range gate. It is checked AFTER the sweep so the report still
  // shows what the fit error would have been -- otherwise a reader cannot tell
  // an unrepresentable interval from a merely hard one.
  if (!r.sweep.empty()) {
    r.dynamic_range = r.sweep.front().dynamic_range;
    if (r.sweep.front().single_signed && r.dynamic_range > max_dynamic_range) {
      r.dynamic_range_exceeded = true;
      r.met_target = false;
      r.recommended_degree = -1;
      char buf[640];
      std::snprintf(buf, sizeof(buf),
                    "REJECTED ON DYNAMIC RANGE, not on fit error: %s spans a "
                    "factor of %.4g over this interval, against a budget of "
                    "%.4g. No degree fixes this -- the VALUES do not fit the "
                    "ciphertext scale, and the symptom downstream is a decode "
                    "failure or a silently wrong magnitude rather than a "
                    "visible approximation error. Narrow the interval. The "
                    "budget is a placeholder until Cheddar's usable dynamic "
                    "range is measured on hardware.",
                    function_name.c_str(), r.dynamic_range, max_dynamic_range);
      r.note = buf;
      return r;
    }
  }

  if (!r.met_target && !r.sweep.empty()) {
    const FitQuality& best = r.sweep.back();
    r.recommended_degree = -1;
    r.achieved_rel_err = r.criterion == FitCriterion::kPointwiseRelative
                             ? best.max_rel_err
                             : best.err_over_peak;
    r.depth = best.depth;
    r.note =
        "No candidate degree up to " + std::to_string(CandidateDegrees().back()) +
        " reached the target on this interval. Narrow the interval (which is "
        "what calibration is for), split the domain, or accept a larger error "
        "-- do NOT simply raise the degree, because the levels have to come "
        "from somewhere.";
  }
  return r;
}

std::function<double(double)> FunctionByName(const std::string& name) {
  if (name == "rsqrt") return [](double x) { return 1.0 / std::sqrt(x); };
  if (name == "exp") return [](double x) { return std::exp(x); };
  if (name == "reciprocal") return [](double x) { return 1.0 / x; };
  if (name == "silu") return [](double x) { return Silu(x); };
  throw std::invalid_argument("FunctionByName: unknown function '" + name + "'");
}

Json FitQuality::ToJson() const {
  Json j = Json::Object();
  j.Set("degree", degree);
  j.Set("depth", depth);
  j.Set("eval_points", eval_points);
  j.Set("max_abs_err", max_abs_err);
  j.Set("at_x_abs", at_x_abs);
  j.Set("max_rel_err", max_rel_err);
  j.Set("at_x_rel", at_x_rel);
  j.Set("rms_err", rms_err);
  j.Set("err_over_peak", err_over_peak);
  j.Set("f_abs_max", f_abs_max);
  j.Set("f_abs_min", f_abs_min);
  j.Set("dynamic_range", dynamic_range);
  j.Set("single_signed", single_signed);
  return j;
}

Json DegreeRecommendation::ToJson() const {
  Json j = Json::Object();
  j.Set("function", function);
  j.Set("criterion", std::string(ToString(criterion)));
  j.Set("interval_source", interval_source);
  j.Set("interval_measured", interval_measured);
  j.Set("lo", lo);
  j.Set("hi", hi);
  j.Set("target_rel_err", target_rel_err);
  j.Set("recommended_degree", static_cast<int64_t>(recommended_degree));
  j.Set("achieved_rel_err", achieved_rel_err);
  j.Set("depth", depth);
  j.Set("met_target", met_target);
  j.Set("dynamic_range_exceeded", dynamic_range_exceeded);
  j.Set("dynamic_range", dynamic_range);
  j.Set("note", note);
  Json arr = Json::Array();
  for (const FitQuality& q : sweep) arr.Push(q.ToJson());
  j.Set("sweep", arr);
  return j;
}

std::string DegreeReport(const std::vector<DegreeRecommendation>& recs) {
  std::ostringstream os;
  os << "Polynomial degree recommendations\n";
  os << "=================================\n";
  os << "Each row is a MEASURED max relative approximation error of a degree-n\n";
  os << "Chebyshev interpolant, evaluated in float64 on a 20001-point grid over\n";
  os << "the stated interval. It is a lower bound on what the encrypted\n";
  os << "evaluation achieves -- CKKS noise is on top of it and is not here.\n\n";
  os << "The 'interval' column is the part that is measured or guessed, and it\n";
  os << "is labelled per row. The error column is always measured.\n\n";

  char buf[512];
  for (const DegreeRecommendation& r : recs) {
    std::snprintf(buf, sizeof(buf),
                  "%s over [%.6g, %.6g]   interval: %s (%s)\n"
                  "  judged on: %s error, target %.1e\n"
                  "  function dynamic range over the interval: %.4g%s\n",
                  r.function.c_str(), r.lo, r.hi,
                  r.interval_measured ? "MEASURED" : "GUESSED",
                  r.interval_source.c_str(), ToString(r.criterion),
                  r.target_rel_err, r.dynamic_range,
                  (r.sweep.empty() || r.sweep.front().single_signed)
                      ? ""
                      : "   (f changes sign; not used as a gate)");
    os << buf;
    std::snprintf(buf, sizeof(buf), "  %8s %7s %14s %14s %14s %14s\n",
                  "degree", "depth", "pointwise rel", "err/peak", "max abs err",
                  "rms err");
    os << buf;
    for (const FitQuality& q : r.sweep) {
      std::snprintf(buf, sizeof(buf),
                    "  %8d %7d %14.4e %14.4e %14.4e %14.4e%s\n", q.degree,
                    q.depth, q.max_rel_err, q.err_over_peak, q.max_abs_err,
                    q.rms_err,
                    (r.met_target && q.degree == r.recommended_degree)
                        ? "   <-- smallest that meets the target"
                        : "");
      os << buf;
    }
    if (!r.met_target) os << "  " << r.note << "\n";
    if (r.dynamic_range_exceeded)
      os << "  (the degree sweep above is shown for information only; the "
            "interval itself is rejected)\n";
    os << "\n";
  }

  os <<
R"(Which error column a row is judged on, and why
---------------------------------------------
  rsqrt, reciprocal   POINTWISE RELATIVE. Their output is a scale factor --
                      1/sqrt(x) multiplies the normalised row, 1/denominator
                      multiplies the attention numerator -- so a relative error
                      in the fit is a relative error in the answer at every
                      magnitude, including the small ones.

  exp, silu           RELATIVE TO PEAK (err/peak). Their output is summed or
                      weighted, not used as a scale: exp feeds a normalising
                      sum in which the small terms carry almost no mass, and
                      SiLU feeds a 14336-term contraction. Judging these two
                      pointwise reports an error the circuit does not suffer,
                      and for SiLU it is meaningless outright -- SiLU crosses
                      zero, so the pointwise ratio near the root measures where
                      the root is, not how good the fit is.

Both columns are printed for every row, so the choice can be argued with.

Two ways a row can fail, and they need different fixes
------------------------------------------------------
"no candidate degree reached the target" is an APPROXIMATION failure: the
function is hard to fit on that interval. Narrowing the interval or splitting
the domain fixes it; so does a higher degree, at the cost of levels.

"REJECTED ON DYNAMIC RANGE" is not an approximation failure at all. The
function's own values span more decades over that interval than the ciphertext
scale can carry, so the result is unrepresentable however well it is
approximated, and no degree helps. This is the failure the HEonGPU record
describes: an exp over a span of e^75 did not fail loudly, it returned values
that outgrew int64 at decrypt and looked like a library fault.

How to read the depth column
---------------------------
depth = ceil(log2(degree+1)), the multiplicative depth of a baby-step/giant-
step evaluation. It is a structural count, not a measurement of Cheddar's
EvalPoly, and Cheddar's own scale/level bookkeeping may add to it. Session S3's
level tracer is what settles the real number.

Why the smallest degree that meets the target is the right pick
---------------------------------------------------------------
A degree costs levels and levels cost bootstraps. HEonGPU measured accuracy
going UP when fit degrees came DOWN, because the deeper circuit ran out of
chain and had to be refreshed more often. Take the smallest degree that meets
the target on the CALIBRATED interval, and spend any surplus on narrowing the
interval instead -- that is the cheaper lever by a wide margin.
)";
  return os.str();
}

}  // namespace oracle
