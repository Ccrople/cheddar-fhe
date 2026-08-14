// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// Range calibration.
//
// The single most expensive lesson in the HEonGPU record is "calibrate every
// fit from data, never from the worst case" (§19, §25.6 of
// reference/LLAMA3_8B_LAYER_FLOW.md): a SoftMax reciprocal fitted over its
// worst-case range was 98.6% wrong, and calibrating one bound moved a seam
// from 7.29e-04 to 7.94e-06 -- 92x, for free. This file is the machinery for
// doing that measurement instead of guessing it.
//
// It measures. It does not decide. Which interval a module is finally built
// against is S0's call, on this evidence.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "oracle/Compare.h"
#include "oracle/Json.h"
#include "oracle/Ops.h"

namespace oracle {

/// Streaming statistics for one named quantity, with a deterministic
/// reservoir so quantiles are available without holding every sample.
class RangeStat {
 public:
  RangeStat() = default;
  explicit RangeStat(std::string name, uint64_t seed = 12345,
                     int64_t reservoir_capacity = 65536);

  void Add(double v);
  void AddAll(const double* v, int64_t n);

  const std::string& Name() const { return name_; }
  int64_t Count() const { return count_; }
  /// How many ORIGINAL samples this statistic stands for. Equal to Count()
  /// except after MergedAcrossLayers(), which rebuilds a channel from the
  /// per-layer reservoirs: there, Count() is the resample count and
  /// Represented() is the true number of values seen.
  int64_t Represented() const { return represented_ > 0 ? represented_ : count_; }
  void AddRepresented(int64_t n) { represented_ += n; }
  int64_t NonFinite() const { return nonfinite_; }
  double Min() const { return min_; }
  double Max() const { return max_; }
  double AbsMax() const { return abs_max_; }
  double AbsMin() const { return abs_min_; }
  double Mean() const;
  double StdDev() const;
  /// Linear-interpolated quantile of the reservoir, q in [0,1].
  double Quantile(double q) const;
  /// Width ratio max/min of the strictly positive part -- the number that
  /// decides how hard a 1/x or 1/sqrt(x) fit is. Returns infinity if the
  /// quantity reaches zero.
  double DynamicRange() const;

  Json ToJson() const;

 private:
  std::string name_;
  int64_t count_ = 0;
  int64_t represented_ = 0;
  int64_t nonfinite_ = 0;
  int64_t seen_ = 0;  ///< for reservoir replacement
  double min_ = 0.0, max_ = 0.0, abs_max_ = 0.0, abs_min_ = 0.0;
  double mean_ = 0.0, m2_ = 0.0;
  int64_t capacity_ = 0;
  std::vector<double> reservoir_;
  uint64_t rng_state_ = 0;
  mutable std::vector<double> sorted_;
  mutable bool sorted_valid_ = false;
};

/// A TensorSink that accumulates a RangeStat for every tensor it is given.
/// It stores no tensors, so it can be attached to a full-model run.
class Calibrator : public TensorSink {
 public:
  Calibrator() = default;
  void Emit(const std::string& name, const Tensor& value,
            const TensorMeta& meta) override;

  const std::vector<std::string>& Names() const { return order_; }
  const RangeStat& Stat(const std::string& name) const;
  const TensorMeta& MetaOf(const std::string& name) const;
  bool Has(const std::string& name) const;

  /// Merge every per-layer channel ("L0.ffn.gate", "L1.ffn.gate", ...) into
  /// one channel keyed by the suffix ("ffn.gate"). This is the view a fit
  /// interval should be chosen from when the same encrypted code runs at
  /// every layer.
  Calibrator MergedAcrossLayers() const;

  Json ToJson() const;

 private:
  std::vector<std::string> order_;
  std::vector<RangeStat> stats_;
  std::vector<TensorMeta> metas_;
  int64_t IndexOf(const std::string& name) const;
};

// ---------------------------------------------------------------------------
// Fit intervals
// ---------------------------------------------------------------------------

/// A measured interval, plus what it is for.
struct FitInterval {
  std::string function;   ///< "rsqrt" | "exp" | "reciprocal" | "silu"
  std::string source;     ///< the calibration channel it was measured from
  double lo = 0.0;
  double hi = 0.0;
  double observed_lo = 0.0;   ///< before the margin
  double observed_hi = 0.0;
  double margin = 0.0;        ///< fractional widening applied
  double dynamic_range = 0.0; ///< hi/lo where meaningful, else 0
  int64_t samples = 0;
  bool measured = false;      ///< false means "declared, not measured"
  std::string note;
};

/// Derives one interval per fit role from a (merged) calibrator. `margin` is
/// the fractional widening applied to the observed extremes; 0.05 means the
/// interval is 5% wider on each side than anything actually seen.
///
/// Intervals are built from the OBSERVED EXTREMES, not from quantiles.
/// A Chebyshev fit does not degrade gracefully outside its interval -- it
/// diverges -- so the extremes are the requirement and the quantiles are
/// reported alongside only to show how much of the interval is tail.
std::vector<FitInterval> DeriveFitIntervals(const Calibrator& merged,
                                            double margin = 0.05);

/// The full calibration report as text, including the caveats that decide
/// whether a number may be quoted.
std::string CalibrationReport(const Calibrator& merged,
                              const std::vector<FitInterval>& intervals,
                              const std::string& provenance_line);

/// Why an interval measured on one window is a lower bound.
std::string CalibrationCaveats();

}  // namespace oracle
