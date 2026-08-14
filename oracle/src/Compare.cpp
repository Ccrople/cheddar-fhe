// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Compare.h"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace oracle {

ErrorReport Compare(const double* test, const double* ref, int64_t n,
                    double rel_floor) {
  ErrorReport r;
  if (rel_floor <= 0.0) rel_floor = 1e-300;

  double sum_sq_err = 0.0, sum_sq_ref = 0.0, sum_sq_test = 0.0;
  double dot = 0.0, sum_abs_err = 0.0;
  int64_t finite = 0;

  for (int64_t i = 0; i < n; ++i) {
    const double a = test[i], b = ref[i];
    if (std::isnan(a) || std::isnan(b)) {
      r.has_nan = true;
      ++r.nonfinite_count;
      continue;
    }
    if (std::isinf(a) || std::isinf(b)) {
      r.has_inf = true;
      ++r.nonfinite_count;
      continue;
    }
    ++finite;
    const double e = a - b;
    const double ae = std::fabs(e);
    if (ae > r.max_abs) {
      r.max_abs = ae;
      r.max_abs_at = i;
    }
    const double denom = std::fabs(b) > rel_floor ? std::fabs(b) : rel_floor;
    const double re = ae / denom;
    if (re > r.max_rel) {
      r.max_rel = re;
      r.max_rel_at = i;
    }
    sum_abs_err += ae;
    sum_sq_err += e * e;
    sum_sq_ref += b * b;
    sum_sq_test += a * a;
    dot += a * b;
    if (std::fabs(b) > r.ref_abs_max) r.ref_abs_max = std::fabs(b);
    if (std::fabs(a) > r.test_abs_max) r.test_abs_max = std::fabs(a);
  }

  r.count = finite;
  if (finite == 0) return r;

  const double dn = static_cast<double>(finite);
  r.mean_abs = sum_abs_err / dn;
  r.rmse = std::sqrt(sum_sq_err / dn);
  r.ref_rms = std::sqrt(sum_sq_ref / dn);
  r.nrmse = r.ref_rms > 0.0 ? r.rmse / r.ref_rms : (r.rmse > 0.0 ? INFINITY : 0.0);
  r.rel_l2 = sum_sq_ref > 0.0 ? std::sqrt(sum_sq_err / sum_sq_ref)
                              : (sum_sq_err > 0.0 ? INFINITY : 0.0);
  const double na = std::sqrt(sum_sq_test), nb = std::sqrt(sum_sq_ref);
  r.cosine = (na > 0.0 && nb > 0.0) ? dot / (na * nb) : (na == nb ? 1.0 : 0.0);
  r.gain = sum_sq_ref > 0.0 ? dot / sum_sq_ref : 1.0;
  if (sum_sq_err > 0.0 && sum_sq_ref > 0.0)
    r.snr_db = 10.0 * std::log10(sum_sq_ref / sum_sq_err);
  else if (sum_sq_ref > 0.0)
    r.snr_db = INFINITY;
  return r;
}

ErrorReport Compare(const Tensor& test, const Tensor& ref, double rel_floor) {
  if (!test.SameShape(ref))
    throw std::invalid_argument("Compare: shape mismatch " + test.ShapeString() +
                                " vs " + ref.ShapeString());
  if (rel_floor <= 0.0) {
    double s = 0.0;
    for (int64_t i = 0; i < ref.Size(); ++i)
      if (std::isfinite(ref[i])) s += ref[i] * ref[i];
    const double rms =
        ref.Size() > 0 ? std::sqrt(s / static_cast<double>(ref.Size())) : 0.0;
    rel_floor = rms > 0.0 ? 1e-12 * rms : 1e-300;
  }
  return Compare(test.Data(), ref.Data(), test.Size(), rel_floor);
}

std::vector<ErrorReport> CompareAlongAxis(const Tensor& test, const Tensor& ref,
                                          int64_t axis, double rel_floor) {
  if (!test.SameShape(ref))
    throw std::invalid_argument("CompareAlongAxis: shape mismatch");
  if (test.Rank() != 2)
    throw std::invalid_argument("CompareAlongAxis: rank 2 only");
  if (axis != 0 && axis != 1)
    throw std::invalid_argument("CompareAlongAxis: axis must be 0 or 1");

  const int64_t rows = test.Dim(0), cols = test.Dim(1);
  std::vector<ErrorReport> out;
  if (axis == 0) {
    out.reserve(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r)
      out.push_back(Compare(test.Row(r), ref.Row(r), cols,
                            rel_floor > 0.0 ? rel_floor : 1e-300));
  } else {
    std::vector<double> a(static_cast<size_t>(rows)), b(static_cast<size_t>(rows));
    out.reserve(static_cast<size_t>(cols));
    for (int64_t c = 0; c < cols; ++c) {
      for (int64_t r = 0; r < rows; ++r) {
        a[static_cast<size_t>(r)] = test.At(r, c);
        b[static_cast<size_t>(r)] = ref.At(r, c);
      }
      out.push_back(Compare(a.data(), b.data(), rows,
                            rel_floor > 0.0 ? rel_floor : 1e-300));
    }
  }
  return out;
}

WorstSlice WorstAlongAxis(const std::vector<ErrorReport>& per_slice) {
  WorstSlice w;
  for (size_t i = 0; i < per_slice.size(); ++i) {
    if (w.index < 0 || per_slice[i].rel_l2 > w.report.rel_l2) {
      w.index = static_cast<int64_t>(i);
      w.report = per_slice[i];
    }
  }
  return w;
}

Json ErrorReport::ToJson() const {
  Json j = Json::Object();
  j.Set("count", count);
  j.Set("max_abs", max_abs);
  j.Set("max_abs_at", max_abs_at);
  j.Set("max_rel", max_rel);
  j.Set("max_rel_at", max_rel_at);
  j.Set("mean_abs", mean_abs);
  j.Set("rmse", rmse);
  j.Set("nrmse", nrmse);
  j.Set("rel_l2", rel_l2);
  j.Set("cosine", cosine);
  j.Set("gain", gain);
  j.Set("snr_db", snr_db);
  j.Set("ref_rms", ref_rms);
  j.Set("ref_abs_max", ref_abs_max);
  j.Set("test_abs_max", test_abs_max);
  j.Set("has_nan", has_nan);
  j.Set("has_inf", has_inf);
  j.Set("nonfinite_count", nonfinite_count);
  return j;
}

std::string ErrorReport::ToString() const {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "n=%lld rel_l2=%.4e max_rel=%.4e max_abs=%.4e rmse=%.4e "
                "nrmse=%.4e cos=%.12f gain=%.9f snr=%.2fdB%s",
                static_cast<long long>(count), rel_l2, max_rel, max_abs, rmse,
                nrmse, cosine, gain, snr_db,
                (has_nan || has_inf) ? "  [NON-FINITE PRESENT]" : "");
  return std::string(buf);
}

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

Threshold ThresholdPrimitive() {
  Threshold t;
  t.name = "primitive";
  t.max_rel_l2 = 1e-5;
  t.max_rel_pointwise = 1e-4;
  t.min_cosine = 1.0 - 1e-9;
  t.max_gain_deviation = 1e-5;
  t.rationale =
      "No polynomial approximation anywhere in the path, so the only error is "
      "CKKS noise: rescale rounding, key-switch noise, and encode/decode. "
      "Anchored above Cheddar's own measured bootstrap error (4.46e-05 average "
      "absolute, SNR 2.36e+08 on Boot-Basic at 2^15 slots, bootparam_30) by "
      "requiring a path WITHOUT a bootstrap to be an order tighter.";
  return t;
}

Threshold ThresholdModule() {
  Threshold t;
  t.name = "module";
  t.max_rel_l2 = 1e-2;
  t.max_rel_pointwise = 5e-2;
  t.min_cosine = 1.0 - 1e-4;
  t.max_gain_deviation = 1e-2;
  t.rationale =
      "Exactly one polynomial approximation in the path, so the fit dominates "
      "and the noise does not. Anchored on HEonGPU's measured module errors at "
      "the real 8B widths: RMSNorm 1.99e-05 through a fused island, SoftMax "
      "5.98e-03 relative at the real attention shape, SwiGLU 1.8e-02 relative. "
      "The tier bound is the loosest of those; ThresholdForModule() tightens "
      "it per module, which is what a module test should actually use.";
  return t;
}

Threshold ThresholdBlock() {
  Threshold t;
  t.name = "block";
  t.max_rel_l2 = 2e-2;
  t.max_rel_pointwise = 1e-1;
  t.min_cosine = 1.0 - 5e-4;
  t.max_gain_deviation = 2e-2;
  t.rationale =
      "One decoder block: two RMSNorms, a SoftMax with two fits, a SwiGLU, "
      "four projections, at least one refresh. HEonGPU measured a whole block "
      "at 2.4e-03 to 2.8e-03 relative at production fit degrees, and its "
      "attention half alone at 2.45e-03. The bound is set an order above the "
      "measured value so that a REGRESSION is caught while a different fit "
      "degree is not reported as a failure.";
  return t;
}

Threshold ThresholdMultiBlock(int64_t blocks) {
  Threshold t = ThresholdBlock();
  if (blocks < 1) blocks = 1;
  t.name = "multi_block(" + std::to_string(blocks) + ")";
  t.max_rel_l2 = ThresholdBlock().max_rel_l2 * std::sqrt(static_cast<double>(blocks));
  t.max_rel_pointwise = 2e-1;
  t.min_cosine = 1.0 - 2e-3;
  t.max_gain_deviation = 5e-2;
  t.rationale =
      "Blocks in sequence. rel_l2 grows as sqrt(n) under the assumption that "
      "per-block errors are independent -- a HYPOTHESIS, not a measurement. "
      "The residual stream is a shared path, so a systematic (correlated) "
      "error would grow like n instead. The first multi-block run must "
      "MEASURE the growth exponent and this function must then be corrected; "
      "until it is, treat a tier-4 pass as weak evidence.";
  return t;
}

Threshold ThresholdMultiBlock() { return ThresholdMultiBlock(4); }

Threshold ThresholdForModule(const std::string& module) {
  Threshold t = ThresholdModule();
  t.name = "module:" + module;
  if (module == "rope" || module == "linear" || module == "projection" ||
      module == "residual" || module == "mask") {
    t = ThresholdPrimitive();
    t.name = "primitive:" + module;
    t.rationale =
        "This module contains NO polynomial approximation. RoPE is a rotation "
        "by plaintext cos/sin, a projection is a plaintext matrix product, a "
        "residual is an addition, a causal mask is a 0/1 plaintext multiply. "
        "Giving any of them a fit-sized tolerance would hide a real defect, so "
        "they are held to the primitive tier. " +
        ThresholdPrimitive().rationale;
    return t;
  }
  if (module == "rmsnorm") {
    t.max_rel_l2 = 5e-3;
    t.max_rel_pointwise = 2e-2;
    t.rationale =
        "1/sqrt has a branch point at zero, so its interpolation error is set "
        "by how close the fitted interval comes to it. At d_model = 4096 the "
        "summed square concentrates as 1/sqrt(4096) and the interval collapses "
        "towards a 1.1:1 span, which is why 5e-03 is achievable; at a small "
        "test width the same circuit is legitimately worse, and the test must "
        "widen the bound rather than call the circuit broken.";
  } else if (module == "softmax") {
    t.max_rel_l2 = 1e-2;
    t.max_rel_pointwise = 5e-2;
    t.rationale =
        "Two fits in series (exp then reciprocal) and a mask between them. "
        "HEonGPU measured 5.98e-03 relative at the real attention shape and "
        "7.94e-06 on a calibrated blocked seam. Masked entries are held to a "
        "separate absolute criterion -- see MaskedResidue below.";
  } else if (module == "silu" || module == "swiglu" || module == "ffn") {
    t.max_rel_l2 = 3e-2;
    t.max_rel_pointwise = 1e-1;
    t.rationale =
        "SiLU is fitted over a symmetric interval whose width tracks the model "
        "width, and the gate product then multiplies two approximate factors. "
        "HEonGPU measured 1.8e-02 relative and it was ordinary. Quote relative "
        "error only: the same circuit reads as 6.9e-01 in absolute terms.";
  } else if (module == "attention") {
    t.max_rel_l2 = 1.5e-2;
    t.max_rel_pointwise = 8e-2;
    t.rationale =
        "The whole attention sublayer: RoPE, QK^T, SoftMax, PV, O. HEonGPU "
        "measured 2.45e-03 at production degrees.";
  }
  return t;
}

std::vector<Threshold> AllThresholds() {
  return {ThresholdPrimitive(), ThresholdModule(), ThresholdBlock(),
          ThresholdMultiBlock(4)};
}

bool Accept(const ErrorReport& r, const Threshold& t, std::string* failure) {
  auto fail = [&](const std::string& m) {
    if (failure) *failure = m;
    return false;
  };
  char buf[256];
  if (r.has_nan) return fail("non-finite value present (NaN)");
  if (r.has_inf) return fail("non-finite value present (Inf)");
  if (r.count == 0) return fail("no finite elements to compare");
  if (!(r.rel_l2 <= t.max_rel_l2)) {
    std::snprintf(buf, sizeof(buf), "rel_l2 %.4e exceeds %.4e", r.rel_l2,
                  t.max_rel_l2);
    return fail(buf);
  }
  if (!(r.max_rel <= t.max_rel_pointwise)) {
    std::snprintf(buf, sizeof(buf),
                  "max pointwise relative error %.4e at index %lld exceeds %.4e",
                  r.max_rel, static_cast<long long>(r.max_rel_at),
                  t.max_rel_pointwise);
    return fail(buf);
  }
  if (!(r.cosine >= t.min_cosine)) {
    std::snprintf(buf, sizeof(buf), "cosine %.12f below %.12f", r.cosine,
                  t.min_cosine);
    return fail(buf);
  }
  if (!(std::fabs(r.gain - 1.0) <= t.max_gain_deviation)) {
    std::snprintf(buf, sizeof(buf), "gain %.9f deviates from 1 by more than %.4e",
                  r.gain, t.max_gain_deviation);
    return fail(buf);
  }
  if (failure) failure->clear();
  return true;
}

std::string ThresholdRationale() {
  std::ostringstream os;
  os <<
R"(Acceptance thresholds
====================
Every criterion is RELATIVE. `max_abs` is reported for diagnosis and is never
a pass condition, because an absolute bound reads a correct circuit as broken
the moment the model width moves (HEonGPU saw exactly that: a SwiGLU at
6.9e-01 absolute and 1.8e-02 relative).

The primary metric is rel_l2 = ||test - ref||_2 / ||ref||_2. `gain` is checked
separately because a systematic scale drift -- a mismanaged rescale, a missing
factor -- shows up there while rel_l2 is still small, and `cosine` is checked
because a near-zero reference makes rel_l2 uninformative.

)";
  for (const Threshold& t : AllThresholds()) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%-18s rel_l2 <= %.1e   pointwise rel <= %.1e   cosine >= "
                  "1-%.0e   |gain-1| <= %.1e\n",
                  t.name.c_str(), t.max_rel_l2, t.max_rel_pointwise,
                  1.0 - t.min_cosine, t.max_gain_deviation);
    os << buf;
  }
  os << "\nPer-module tightenings of the module tier:\n";
  for (const char* m : {"rmsnorm", "softmax", "swiglu", "attention", "rope",
                        "linear", "residual", "mask"}) {
    Threshold t = ThresholdForModule(m);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "  %-12s rel_l2 <= %.1e   pointwise <= %.1e\n",
                  m, t.max_rel_l2, t.max_rel_pointwise);
    os << buf;
  }
  os <<
R"(
Two criteria that are NOT relative, and why
-------------------------------------------
MaskedResidue. A causally masked attention probability is exactly +0.0 in the
oracle, so it has no relative error at all. Test it as an ABSOLUTE bound
against the unmasked peak: |p_masked| <= 1e-5 * max_unmasked_p. HEonGPU
measured 8.39e-09 for a masked-off key and 5.61e-07 before calibration -- the
number moves with the calibration, which is exactly why it must be its own
criterion and not folded into rel_l2 over the whole score tensor, where 99% of
the mass would drown it.

ProbabilityMass. Each attention row should sum to 1. Test |sum - 1| <= 1e-2 at
module tier. This is the criterion that catches a wrong reciprocal interval,
which rel_l2 on the output can partially absorb.

STATUS OF THESE NUMBERS
-----------------------
The primitive tier is anchored on Cheddar's own two measured numbers (the
bootstrap error) and is otherwise a HYPOTHESIS: no per-operation error trace
for Cheddar exists yet. Session S3 is producing exactly that. When it lands,
ThresholdPrimitive() must be re-anchored on the measured per-op error at the
level the module actually runs at, and this paragraph deleted.

The module, block and multi-block tiers are anchored on HEonGPU measurements
at the real 8B widths. They transfer as ORDERS OF MAGNITUDE, not as digits:
HEonGPU ran 40-60 bit primes and Cheddar's presets are ~30-bit with 32-bit
words, so every fit degree and every interval must be re-derived here.

The multi-block composition exponent (sqrt(n)) is unmeasured. See
ThresholdMultiBlock().
)";
  return os.str();
}

}  // namespace oracle
