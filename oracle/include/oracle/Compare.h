// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// The numerical comparison utility, and the acceptance thresholds.
//
// One lesson from the HEonGPU record is baked in here (§23.7): an ABSOLUTE
// error means nothing across shapes. A SwiGLU that looked broken at 6.9e-01
// absolute was ordinary at 1.8e-02 relative, because a fit's error grows with
// the range it is fitted over and a wider model gives a wider range. Every
// threshold in this file is therefore relative; `max_abs` is reported for
// diagnosis and is never the pass criterion.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "oracle/Json.h"
#include "oracle/Tensor.h"

namespace oracle {

struct ErrorReport {
  int64_t count = 0;

  double max_abs = 0.0;    ///< max |test - ref|
  int64_t max_abs_at = -1; ///< flat index where it occurred
  double max_rel = 0.0;    ///< max |test-ref| / max(|ref|, rel_floor)
  int64_t max_rel_at = -1;

  double mean_abs = 0.0;
  double rmse = 0.0;
  /// rmse / rms(ref). The scale-free version of rmse, and the number to quote
  /// when comparing across shapes.
  double nrmse = 0.0;
  /// ||test - ref||_2 / ||ref||_2. The primary acceptance metric.
  double rel_l2 = 0.0;
  /// <test,ref> / (||test|| ||ref||). Catches a direction change that a small
  /// rel_l2 could hide when the reference is nearly zero.
  double cosine = 1.0;
  /// <test,ref>/<ref,ref>. HEonGPU's "gain": 1.0 means no systematic scaling.
  /// A gain that drifts from 1 is a scale-management bug, not noise.
  double gain = 1.0;
  double snr_db = 0.0;

  double ref_rms = 0.0;
  double ref_abs_max = 0.0;
  double test_abs_max = 0.0;

  bool has_nan = false;   ///< in either operand
  bool has_inf = false;
  /// Element pairs skipped because one side was not finite. Every metric above
  /// is computed over the finite pairs only, so a report with
  /// nonfinite_count > 0 is diagnostic and never a pass.
  int64_t nonfinite_count = 0;

  Json ToJson() const;
  std::string ToString() const;
};

/// Compares two equally-sized buffers.
///
/// `rel_floor` is the denominator floor for the per-element relative error. It
/// exists because a reference value of exactly 0 -- which the causal mask
/// produces on purpose -- has no relative error. Default is derived from the
/// reference magnitude by `Compare(Tensor, Tensor)`; pass it explicitly when
/// calling the raw form.
ErrorReport Compare(const double* test, const double* ref, int64_t n,
                    double rel_floor);

/// Shape-checked form. `rel_floor` defaults to 1e-12 * rms(ref), i.e. relative
/// error is measured against the tensor's own scale where an element is tiny.
ErrorReport Compare(const Tensor& test, const Tensor& ref,
                    double rel_floor = -1.0);

/// Per-slice reports along one axis of a rank-2 or rank-3 tensor.
/// axis 0 of a [tokens, channels] tensor gives one report per token;
/// axis 1 gives one per channel.
std::vector<ErrorReport> CompareAlongAxis(const Tensor& test, const Tensor& ref,
                                          int64_t axis, double rel_floor = -1.0);

/// The worst slice, and where. Returns index -1 for an empty input.
struct WorstSlice {
  int64_t index = -1;
  ErrorReport report;
};
WorstSlice WorstAlongAxis(const std::vector<ErrorReport>& per_slice);

// ---------------------------------------------------------------------------
// Acceptance thresholds
// ---------------------------------------------------------------------------

/// A pass/fail criterion. All fields are relative except `require_exact_zero`.
struct Threshold {
  std::string name;
  double max_rel_l2 = 0.0;
  double max_rel_pointwise = 0.0;
  double min_cosine = 0.0;
  double max_gain_deviation = 0.0;  ///< |gain - 1| must not exceed this
  std::string rationale;
};

/// Tier 1 -- a single primitive against the oracle, no polynomial fit in the
/// path (add, multiply, rotate, mask, a linear projection, a residual).
/// The only error present is CKKS noise.
Threshold ThresholdPrimitive();

/// Tier 2 -- one module whose path contains ONE polynomial approximation
/// (RMSNorm, SiLU/SwiGLU, a SoftMax numerator). Dominated by the fit, not by
/// the noise.
Threshold ThresholdModule();

/// Tier 3 -- one decoder block: two norms, a SoftMax with two fits, a SwiGLU,
/// four projections and at least one refresh.
Threshold ThresholdBlock();

/// Tier 4 -- several blocks in sequence. Errors compose, and the residual
/// stream carries them forward.
Threshold ThresholdMultiBlock();

/// Per-module tightenings of the tier-2 threshold. A module whose path
/// contains no polynomial approximation at all -- RoPE, a projection, a
/// residual add, a causal mask -- returns the PRIMITIVE threshold, because
/// nothing in it can be approximated and a fit-sized tolerance would hide a
/// real bug. Recognised names:
///   "rmsnorm", "rope", "linear", "projection", "residual", "mask",
///   "softmax", "silu", "swiglu", "ffn", "attention"
/// An unknown name returns ThresholdModule() with the name attached.
Threshold ThresholdForModule(const std::string& module);

/// Tier 4 for an explicit block count. Composition model:
///   rel_l2(n) = rel_l2(block) * sqrt(n)
/// which assumes per-block errors are independent. That assumption is a
/// HYPOTHESIS, not a measurement -- see ThresholdRationale().
Threshold ThresholdMultiBlock(int64_t blocks);

/// All four, for reporting.
std::vector<Threshold> AllThresholds();

/// Does this report satisfy the threshold? `failure` receives the first
/// criterion that failed, named.
bool Accept(const ErrorReport& r, const Threshold& t, std::string* failure);

/// The full rationale text for the threshold table, including the measured
/// Cheddar numbers the tiers are anchored on.
std::string ThresholdRationale();

}  // namespace oracle
