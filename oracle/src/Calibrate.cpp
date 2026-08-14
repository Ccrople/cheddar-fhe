// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Calibrate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "oracle/Rng.h"

namespace oracle {
namespace {

std::string SuffixAfterLayer(const std::string& name) {
  if (name.empty() || name[0] != 'L') return name;
  size_t i = 1;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
  if (i == 1 || i >= name.size() || name[i] != '.') return name;
  return name.substr(i + 1);
}

}  // namespace

RangeStat::RangeStat(std::string name, uint64_t seed, int64_t capacity)
    : name_(std::move(name)), capacity_(capacity) {
  rng_state_ = seed ^ HashName(name_.c_str());
  reservoir_.reserve(static_cast<size_t>(std::min<int64_t>(capacity, 4096)));
}

void RangeStat::Add(double v) {
  if (!std::isfinite(v)) {
    ++nonfinite_;
    return;
  }
  sorted_valid_ = false;
  if (count_ == 0) {
    min_ = max_ = v;
    abs_max_ = abs_min_ = std::fabs(v);
  } else {
    if (v < min_) min_ = v;
    if (v > max_) max_ = v;
    const double a = std::fabs(v);
    if (a > abs_max_) abs_max_ = a;
    if (a < abs_min_) abs_min_ = a;
  }
  ++count_;
  const double delta = v - mean_;
  mean_ += delta / static_cast<double>(count_);
  m2_ += delta * (v - mean_);

  // Reservoir sampling (Vitter R), deterministic via SplitMix64.
  ++seen_;
  if (static_cast<int64_t>(reservoir_.size()) < capacity_) {
    reservoir_.push_back(v);
  } else {
    uint64_t z = (rng_state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    const uint64_t j = z % static_cast<uint64_t>(seen_);
    if (j < static_cast<uint64_t>(capacity_)) reservoir_[static_cast<size_t>(j)] = v;
  }
}

void RangeStat::AddAll(const double* v, int64_t n) {
  for (int64_t i = 0; i < n; ++i) Add(v[i]);
}

double RangeStat::Mean() const { return count_ > 0 ? mean_ : 0.0; }

double RangeStat::StdDev() const {
  if (count_ < 2) return 0.0;
  return std::sqrt(m2_ / static_cast<double>(count_ - 1));
}

double RangeStat::Quantile(double q) const {
  if (reservoir_.empty()) return 0.0;
  if (!sorted_valid_) {
    sorted_ = reservoir_;
    std::sort(sorted_.begin(), sorted_.end());
    sorted_valid_ = true;
  }
  if (q <= 0.0) return sorted_.front();
  if (q >= 1.0) return sorted_.back();
  const double pos = q * static_cast<double>(sorted_.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, sorted_.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return sorted_[lo] * (1.0 - frac) + sorted_[hi] * frac;
}

double RangeStat::DynamicRange() const {
  if (count_ == 0) return 0.0;
  if (abs_min_ <= 0.0) return std::numeric_limits<double>::infinity();
  return abs_max_ / abs_min_;
}

Json RangeStat::ToJson() const {
  Json j = Json::Object();
  j.Set("name", name_);
  j.Set("count", count_);
  j.Set("non_finite", nonfinite_);
  j.Set("min", min_);
  j.Set("max", max_);
  j.Set("abs_min", abs_min_);
  j.Set("abs_max", abs_max_);
  j.Set("mean", Mean());
  j.Set("stddev", StdDev());
  j.Set("dynamic_range", DynamicRange());
  Json q = Json::Object();
  q.Set("p0_01", Quantile(0.0001));
  q.Set("p1", Quantile(0.01));
  q.Set("p50", Quantile(0.5));
  q.Set("p99", Quantile(0.99));
  q.Set("p99_99", Quantile(0.9999));
  j.Set("quantiles", q);
  j.Set("reservoir_size", static_cast<int64_t>(reservoir_.size()));
  return j;
}

// ---------------------------------------------------------------------------

int64_t Calibrator::IndexOf(const std::string& name) const {
  for (size_t i = 0; i < order_.size(); ++i)
    if (order_[i] == name) return static_cast<int64_t>(i);
  return -1;
}

bool Calibrator::Has(const std::string& name) const { return IndexOf(name) >= 0; }

void Calibrator::Emit(const std::string& name, const Tensor& value,
                      const TensorMeta& meta) {
  int64_t i = IndexOf(name);
  if (i < 0) {
    order_.push_back(name);
    stats_.emplace_back(name);
    metas_.push_back(meta);
    i = static_cast<int64_t>(order_.size()) - 1;
  }
  stats_[static_cast<size_t>(i)].AddAll(value.Data(), value.Size());
}

const RangeStat& Calibrator::Stat(const std::string& name) const {
  const int64_t i = IndexOf(name);
  if (i < 0) throw std::out_of_range("Calibrator: no channel " + name);
  return stats_[static_cast<size_t>(i)];
}

const TensorMeta& Calibrator::MetaOf(const std::string& name) const {
  const int64_t i = IndexOf(name);
  if (i < 0) throw std::out_of_range("Calibrator: no channel " + name);
  return metas_[static_cast<size_t>(i)];
}

Calibrator Calibrator::MergedAcrossLayers() const {
  Calibrator out;
  for (size_t i = 0; i < order_.size(); ++i) {
    const std::string key = SuffixAfterLayer(order_[i]);
    int64_t j = out.IndexOf(key);
    if (j < 0) {
      out.order_.push_back(key);
      out.stats_.emplace_back(key);
      TensorMeta m = metas_[i];
      m.layer = -1;
      out.metas_.push_back(m);
      j = static_cast<int64_t>(out.order_.size()) - 1;
    }
    // Merge by replaying the reservoir and correcting the extremes exactly.
    // The reservoir is a sample, so the merged quantiles are approximate; the
    // merged extremes are not, and the extremes are what an interval is built
    // from.
    const RangeStat& src = stats_[i];
    if (src.Count() == 0) continue;
    RangeStat& dst = out.stats_[static_cast<size_t>(j)];
    dst.Add(src.Min());
    dst.Add(src.Max());
    for (int64_t q = 0; q <= 1000; ++q)
      dst.Add(src.Quantile(static_cast<double>(q) / 1000.0));
    dst.AddRepresented(src.Represented());
  }
  return out;
}

Json Calibrator::ToJson() const {
  Json arr = Json::Array();
  for (size_t i = 0; i < order_.size(); ++i) {
    Json j = stats_[i].ToJson();
    j.Set("dims", metas_[i].dims);
    j.Set("description", metas_[i].description);
    j.Set("consumer", metas_[i].consumer);
    j.Set("fit_role", metas_[i].fit_role);
    j.Set("layer", metas_[i].layer);
    arr.Push(j);
  }
  return arr;
}

// ---------------------------------------------------------------------------

namespace {

FitInterval MakeInterval(const std::string& fn, const std::string& source,
                         double lo, double hi, double margin, int64_t samples,
                         const std::string& note) {
  FitInterval fi;
  fi.function = fn;
  fi.source = source;
  fi.observed_lo = lo;
  fi.observed_hi = hi;
  fi.margin = margin;
  fi.samples = samples;
  fi.measured = samples > 0;
  const double width = hi - lo;
  fi.lo = lo - margin * width;
  fi.hi = hi + margin * width;
  fi.note = note;
  return fi;
}

}  // namespace

std::vector<FitInterval> DeriveFitIntervals(const Calibrator& c, double margin) {
  std::vector<FitInterval> out;

  // 1/sqrt -- the RMSNorm argument. Both norms share one encrypted circuit, so
  // the interval is the union over both.
  {
    double lo = 0.0, hi = 0.0;
    int64_t n = 0;
    bool any = false;
    for (const char* ch : {"attn_norm.mean_sq", "ffn_norm.mean_sq"}) {
      if (!c.Has(ch)) continue;
      const RangeStat& s = c.Stat(ch);
      if (s.Count() == 0) continue;
      if (!any) { lo = s.Min(); hi = s.Max(); any = true; }
      lo = std::min(lo, s.Min());
      hi = std::max(hi, s.Max());
      n += s.Represented();
    }
    if (any) {
      FitInterval fi = MakeInterval(
          "rsqrt", "attn_norm.mean_sq + ffn_norm.mean_sq (union)", lo, hi,
          margin, n,
          "1/sqrt has a branch point at 0. The lower bound is the number that "
          "sets the fit degree; it is floored by rms_norm_eps and by nothing "
          "else, so a token whose activations are all near zero drives it. The "
          "lower bound must never be allowed to reach 0.");
      // Never let the margin push the lower bound to or below zero.
      if (fi.lo <= 0.0) {
        fi.lo = lo * 0.5;
        fi.note += " Lower bound clamped to observed_lo/2 to stay positive.";
      }
      fi.dynamic_range = fi.lo > 0.0 ? fi.hi / fi.lo : 0.0;
      out.push_back(fi);
    }
  }

  // exp -- the SoftMax numerator argument.
  if (c.Has("attn.exp_input")) {
    const RangeStat& s = c.Stat("attn.exp_input");
    if (s.Count() > 0) {
      FitInterval fi = MakeInterval(
          "exp", "attn.exp_input", s.Min(), s.Max(), margin, s.Represented(),
          "This channel INCLUDES causally masked positions on purpose. The "
          "encrypted circuit applies the mask to exp(u), not to u, so exp is "
          "evaluated at every position including the masked ones and the fit "
          "interval must cover them. Under row-max shifting an unmasked u is "
          "<= 0, but a masked u can be positive -- which is why the observed "
          "upper bound here is not 0.");
      fi.dynamic_range = 0.0;
      out.push_back(fi);
    }
  }

  // reciprocal -- the SoftMax denominator.
  if (c.Has("attn.denominator")) {
    const RangeStat& s = c.Stat("attn.denominator");
    if (s.Count() > 0) {
      FitInterval fi = MakeInterval(
          "reciprocal", "attn.denominator", s.Min(), s.Max(), margin, s.Represented(),
          "The measured denominator range, NOT the worst case. The worst case "
          "for a causal row of length T is [1, T] (row 0 has one key, row T-1 "
          "has T), and fitting 1/x over that was measured at 98.6% wrong in "
          "the HEonGPU record. The ratio hi/lo below is the whole difficulty "
          "of this fit.");
      if (fi.lo <= 0.0) {
        fi.lo = s.Min() * 0.5;
        fi.note += " Lower bound clamped to observed_lo/2 to stay positive.";
      }
      fi.dynamic_range = fi.lo > 0.0 ? fi.hi / fi.lo : 0.0;
      out.push_back(fi);
    }
  }

  // SiLU -- the gate projection output.
  if (c.Has("ffn.gate")) {
    const RangeStat& s = c.Stat("ffn.gate");
    if (s.Count() > 0) {
      const double a = std::max(std::fabs(s.Min()), std::fabs(s.Max()));
      FitInterval fi = MakeInterval(
          "silu", "ffn.gate", -a, a, margin, s.Represented(),
          "Symmetrised: SiLU is fitted over a symmetric interval because the "
          "gate distribution is close to symmetric and an asymmetric interval "
          "buys nothing. The width tracks the model width, so this number is "
          "NOT transferable to a different hidden size.");
      fi.dynamic_range = 0.0;
      out.push_back(fi);
    }
  }

  return out;
}

std::string CalibrationCaveats() {
  return
R"(How far these numbers may be taken
=================================
1.  An interval measured on N token windows is a LOWER BOUND on the interval a
    deployment sees. Widen it deliberately and say by how much; do not quietly
    fit to the measurement.

2.  With synthetic weights every number here is a property of the synthetic
    distribution, not of Llama-3. The weight scaling (1/sqrt(fan_in)) was
    chosen so the score distribution lands in the same decade a trained model
    produces; that is a modelling choice, and it is the whole reason these
    ranges are plausible rather than the e^75 dynamic range HEonGPU measured
    from unscaled random weights.

3.  The exp channel deliberately includes causally masked positions, because
    the encrypted circuit masks exp(u) rather than u. An interval measured
    over unmasked positions only will be too narrow, and the failure will look
    like Chebyshev instability rather than like a range error.

4.  The reciprocal interval is where calibration pays. The worst case for a
    causal row is [1, T]; the measured range is far narrower, and the ratio
    between them is the degree saving. Recheck it whenever the sequence
    length, the fixed SoftMax shift, or the score scale changes -- all three
    move it.

5.  Quantiles come from a bounded reservoir and are approximate. Minima and
    maxima are exact. Build intervals from the extremes; read the quantiles
    only to see how much of the interval is tail.

6.  After merging across layers, quantiles are approximate twice over (a
    resampling of a sample). The merged extremes remain exact.
)";
}

std::string CalibrationReport(const Calibrator& c,
                              const std::vector<FitInterval>& intervals,
                              const std::string& provenance_line) {
  std::ostringstream os;
  os << "Calibration report -- measured input ranges\n";
  os << "==========================================\n";
  os << provenance_line << "\n\n";

  os << "Measured ranges, merged across layers\n";
  os << "-------------------------------------\n";
  os << "(min/max are exact; quantiles come from a bounded reservoir)\n\n";
  char buf[1024];
  std::snprintf(buf, sizeof(buf), "%-26s %12s %13s %13s %13s %11s %11s\n",
                "channel", "samples", "min", "max", "mean", "stddev", "p99.99");
  os << buf;
  for (const std::string& name : c.Names()) {
    const RangeStat& s = c.Stat(name);
    if (s.Count() == 0) continue;
    std::snprintf(buf, sizeof(buf),
                  "%-26s %12lld %13.6g %13.6g %13.6g %11.4g %11.4g\n",
                  name.c_str(), static_cast<long long>(s.Represented()), s.Min(),
                  s.Max(), s.Mean(), s.StdDev(), s.Quantile(0.9999));
    os << buf;
  }

  os << "\nDerived fit intervals\n";
  os << "---------------------\n";
  for (const FitInterval& fi : intervals) {
    std::snprintf(buf, sizeof(buf),
                  "\n%s  from %s\n"
                  "  observed          [%.6g, %.6g]\n"
                  "  with %.0f%% margin  [%.6g, %.6g]\n"
                  "  samples           %lld\n",
                  fi.function.c_str(), fi.source.c_str(), fi.observed_lo,
                  fi.observed_hi, fi.margin * 100.0, fi.lo, fi.hi,
                  static_cast<long long>(fi.samples));
    os << buf;
    if (fi.dynamic_range > 0.0) {
      std::snprintf(buf, sizeof(buf), "  hi/lo             %.4g\n",
                    fi.dynamic_range);
      os << buf;
    }
    os << "  note: " << fi.note << "\n";
  }

  os << "\n" << CalibrationCaveats();
  return os.str();
}

}  // namespace oracle
