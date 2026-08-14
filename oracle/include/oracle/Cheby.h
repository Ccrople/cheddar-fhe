// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// Chebyshev interpolation, used here for one purpose: to turn "what degree
// should the encrypted fit be?" from a guess into a measurement.
//
// What this measures and what it does NOT
// ---------------------------------------
// It measures the APPROXIMATION error of a degree-n Chebyshev interpolant on
// a given interval, evaluated in float64 on a dense grid. That is a real,
// reproducible number and it is a lower bound on what the encrypted evaluation
// can achieve.
//
// It does NOT measure:
//   - the CKKS evaluation error of that polynomial (scale management, rescale
//     rounding, the noise each product adds). Cheddar's EvalPoly has its own
//     behaviour and only a run on the GPU can report it;
//   - the level cost, beyond the standard ceil(log2(degree+1)) depth of a
//     baby-step/giant-step tree, which is a structural count and not a
//     measurement of Cheddar's EvalPoly;
//   - whether a higher degree is actually better under FHE. It is often not:
//     the HEonGPU record measured accuracy going UP with LOWER-degree fits,
//     because the level a degree costs has to come from somewhere.
//
// Everything this file produces is therefore labelled "measured approximation
// error", never "expected accuracy".

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "oracle/Json.h"

namespace oracle {

/// A Chebyshev interpolant of f on [lo, hi].
struct ChebyFit {
  double lo = -1.0;
  double hi = 1.0;
  int degree = 0;
  std::vector<double> coeffs;  ///< size degree+1, coeffs[0] is the T0 term

  double Eval(double x) const;  ///< Clenshaw recurrence
};

/// Interpolation at the degree+1 Chebyshev nodes of the first kind.
/// Near-minimax; the difference from true minimax is well under the margin any
/// of these intervals carries.
ChebyFit FitChebyshev(const std::function<double(double)>& f, double lo,
                      double hi, int degree);

/// Measured error of a fit against its target, on a dense uniform grid plus
/// the interval endpoints.
struct FitQuality {
  int degree = 0;
  int64_t eval_points = 0;
  double max_abs_err = 0.0;
  double at_x_abs = 0.0;
  /// POINTWISE relative error: max |err| / max(|f(x)|, 1e-6*peak). This is the
  /// right criterion when the approximation's output MULTIPLIES the result --
  /// 1/sqrt(x) scales the normalised row, 1/x scales the attention
  /// probability -- because a relative error in the fit becomes a relative
  /// error in the answer, at every magnitude.
  double max_rel_err = 0.0;
  double at_x_rel = 0.0;
  /// Error relative to the function's PEAK on the interval:
  /// max|err| / max|f|. This is the right criterion when the output is summed
  /// or weighted rather than used as a scale factor -- exp() feeds a
  /// normalising sum in which the small terms carry little mass, and SiLU
  /// feeds a 14336-term contraction. Using the pointwise measure for those two
  /// reports an error that the circuit does not actually suffer, and for a
  /// sign-changing function like SiLU it is dominated entirely by the zero
  /// crossing.
  double err_over_peak = 0.0;
  double rms_err = 0.0;
  double f_abs_max = 0.0;
  double f_abs_min = 0.0;
  /// max|f| / min|f| over the interval. This is a property of the FUNCTION AND
  /// THE INTERVAL, not of the fit, and it is the quantity that decides whether
  /// the result is REPRESENTABLE at all -- a fit can be arbitrarily accurate
  /// in relative terms and still be useless because its output spans more
  /// decades than the ciphertext scale carries. HEonGPU's documented failure
  /// was exactly this: an exp over a span of e^75, which did not fail loudly
  /// but outgrew int64 at decrypt.
  /// Infinite when f reaches zero on the interval.
  double dynamic_range = 0.0;
  /// False when f changes sign (or touches zero) on the interval, in which
  /// case dynamic_range is infinite by construction and is reported but not
  /// used as a gate. SiLU is the only one of the four that does this.
  bool single_signed = true;
  /// ceil(log2(degree+1)) -- the multiplicative depth of a baby-step/giant-step
  /// evaluation. Structural, not measured.
  int depth = 0;

  Json ToJson() const;
};

FitQuality MeasureFit(const std::function<double(double)>& f,
                      const ChebyFit& fit, int64_t eval_points = 20001);

/// The degrees worth trying: 2^k - 1, because a baby-step/giant-step tree's
/// depth is ceil(log2(degree+1)) and any other degree pays the same depth as
/// the next 2^k - 1 above it.
const std::vector<int>& CandidateDegrees();

/// Which measured error a degree is selected on. See FitQuality.
enum class FitCriterion {
  /// max |err| / |f(x)| -- for approximations whose output multiplies the
  /// result: 1/sqrt and 1/x.
  kPointwiseRelative = 0,
  /// max |err| / max|f| -- for approximations whose output is summed or
  /// weighted: exp and SiLU.
  kRelativeToPeak = 1,
};
const char* ToString(FitCriterion c);

/// The criterion each of the four Llama-3 functions is judged on, with the
/// reason. Unknown names default to kPointwiseRelative, the stricter of the
/// two.
FitCriterion CriterionFor(const std::string& function_name);

struct DegreeRecommendation {
  std::string function;
  FitCriterion criterion = FitCriterion::kPointwiseRelative;
  std::string interval_source;
  double lo = 0.0;
  double hi = 0.0;
  double target_rel_err = 0.0;
  int recommended_degree = -1;   ///< -1 if no candidate met the target
  double achieved_rel_err = 0.0;
  int depth = 0;
  bool met_target = false;
  /// Set when the interval was rejected on DYNAMIC RANGE rather than on fit
  /// error. No degree helps; the interval itself is unusable.
  bool dynamic_range_exceeded = false;
  double dynamic_range = 0.0;
  /// True when `lo`/`hi` came from a calibration run; false when they were
  /// supplied by hand. This is the measured-vs-guessed flag and it is carried
  /// all the way into the report.
  bool interval_measured = false;
  std::vector<FitQuality> sweep;
  std::string note;

  Json ToJson() const;
};

/// Sweeps CandidateDegrees() and returns the smallest degree whose MEASURED
/// max relative approximation error is at or below `target_rel_err`.
///
/// A candidate interval is ALSO rejected -- at every degree -- when the
/// function's own dynamic range over it exceeds `max_dynamic_range`. That is
/// not an approximation failure; it is the interval being unrepresentable, and
/// reporting it as a degree problem would send the next session to raise the
/// degree when the fix is to narrow the interval. The default 1e12 is roughly
/// 2^40, an order of magnitude inside what a ~2^30-scale CKKS chain carries;
/// it is a PLACEHOLDER until S3 measures Cheddar's real usable dynamic range,
/// and it is deliberately reported rather than silently applied.
DegreeRecommendation RecommendDegree(const std::string& function_name,
                                     const std::function<double(double)>& f,
                                     double lo, double hi,
                                     double target_rel_err,
                                     bool interval_measured,
                                     const std::string& interval_source,
                                     double max_dynamic_range = 1e12);

/// The four functions the Llama-3 circuit needs, by name:
///   "rsqrt"      x -> 1/sqrt(x)
///   "exp"        x -> exp(x)
///   "reciprocal" x -> 1/x
///   "silu"       x -> x*sigmoid(x)
/// Throws std::invalid_argument on an unknown name.
std::function<double(double)> FunctionByName(const std::string& name);

/// Text report for a set of recommendations, with the measured/guessed split
/// stated per row.
std::string DegreeReport(const std::vector<DegreeRecommendation>& recs);

}  // namespace oracle
