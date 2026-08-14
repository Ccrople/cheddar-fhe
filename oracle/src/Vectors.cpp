// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Vectors.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "oracle/Rng.h"

namespace oracle {
namespace {

void ScaleMatrix(Tensor* t, double s) { ScaleInPlace(t, s); }

}  // namespace

const std::vector<std::string>& TestVectorNames() {
  static const std::vector<std::string> kNames = {
      "baseline",
      "single_token",
      "two_tokens",
      "mask_boundary",
      "near_zero_norm",
      "zero_row",
      "repeated_tokens",
      "repeated_tokens_no_rope",
      "large_scores",
      "extreme_magnitudes",
      "outlier_channels",
      "unscaled_weights",
      "long_window",
  };
  return kNames;
}

TestVector BuildTestVector(const std::string& name, const Llama3Config& base,
                           const SyntheticSpec& spec, int64_t tokens) {
  TestVector v;
  v.name = name;
  v.config = base;
  v.position_offset = 0;
  SyntheticSpec s = spec;
  // Every vector gets its own weight stream so that changing one does not move
  // another.
  s.seed = spec.seed ^ HashName(name.c_str());

  int64_t t = tokens;

  if (name == "baseline") {
    v.stresses = "nothing in particular -- the reference point every other "
                 "vector is read against";
    v.expectation = "all module and block tolerances met at their tier";
  } else if (name == "single_token") {
    t = 1;
    v.stresses = "the degenerate causal row: one key, denominator exactly 1 "
                 "under row-max shifting";
    v.expectation =
        "every attention probability is exactly 1.0; the reciprocal fit is "
        "evaluated at exactly its interval's low end, which is the value most "
        "likely to fall outside a badly-chosen interval";
  } else if (name == "two_tokens") {
    t = 2;
    v.stresses = "the smallest non-trivial causal structure";
    v.expectation =
        "row 0 is one-hot on key 0; row 1 is a genuine two-way SoftMax. An "
        "implementation that broadcasts the mask wrongly fails here and "
        "nowhere else";
  } else if (name == "mask_boundary") {
    t = 8;
    v.stresses = "every causal boundary at once: the diagonal, the first row, "
                 "the last row, and the strictly-upper triangle";
    v.expectation =
        "probability is exactly zero at every (q,k) with k>q, and the row sums "
        "are 1 at every q. Under FHE the masked entries are not exactly zero; "
        "hold them to |p| <= 1e-5 * max unmasked p, which is a separate "
        "criterion from rel_l2 (see ThresholdRationale)";
    v.is_control = true;
  } else if (name == "near_zero_norm") {
    v.stresses = "the LOWER bound of the 1/sqrt interval, where the branch "
                 "point at zero sets the fit error";
    v.expectation =
        "the RMSNorm output of the shrunken tokens is still O(1) -- the norm "
        "divides out the scale. If the encrypted result collapses or explodes "
        "here, the 1/sqrt interval's low end is wrong, not the circuit";
    v.is_control = true;
  } else if (name == "zero_row") {
    v.stresses = "mean_sq exactly equal to rms_norm_eps, the hard floor of the "
                 "1/sqrt interval";
    v.expectation =
        "the normalised row is exactly zero (0 * 1/sqrt(eps) = 0) and nothing "
        "is NaN. A circuit that computes 1/sqrt by Newton iteration from a bad "
        "initial guess diverges here; a Chebyshev fit does not, provided eps "
        "is inside its interval";
    v.is_control = true;
  } else if (name == "repeated_tokens") {
    v.stresses = "identical token content, so the only thing distinguishing "
                 "queries is position (RoPE) and the causal mask";
    v.expectation =
        "scores within a row vary only through RoPE. This is the vector that "
        "catches a RoPE that was wired in but never applied -- with RoPE "
        "missing, every unmasked score in a row is identical and the SoftMax "
        "output becomes exactly uniform, which is trivially detectable";
    v.is_control = true;
  } else if (name == "repeated_tokens_no_rope") {
    v.config.rope_enabled = false;
    v.stresses = "the exactly-uniform SoftMax, and a denominator that walks "
                 "1, 2, 3, ... up the causal triangle";
    v.expectation =
        "with identical tokens and RoPE off, row q is uniform over q+1 keys, "
        "so probability = 1/(q+1) EXACTLY and the row-max-shifted denominator "
        "is exactly q+1. This is the closed-form check for the reciprocal "
        "fit: it exercises the full [1, T] worst-case interval, which is "
        "precisely the interval calibration exists to avoid";
    v.is_control = true;
  } else if (name == "large_scores") {
    v.stresses = "the UPPER bound of the exp interval";
    v.expectation =
        "with Q and K projections scaled up, raw scores grow by the square of "
        "the scale. The oracle's row-max SoftMax is unaffected (it is shift "
        "invariant); an encrypted fixed-shift SoftMax is NOT, and its exp "
        "interval must cover the widened range or it returns nonsense that "
        "looks like Chebyshev instability";
    v.is_control = true;
  } else if (name == "extreme_magnitudes") {
    v.stresses = "cancellation and dynamic range inside the projections and "
                 "the norm reduction";
    v.expectation =
        "channels alternate between 1e-3 and 1e+3, so sum_c x^2 is dominated "
        "by half the channels. The norm output is well-scaled; the projection "
        "accumulates across six orders of magnitude, which is where a CKKS "
        "scale that was chosen for the average magnitude loses the small half";
    v.is_control = true;
  } else if (name == "outlier_channels") {
    v.stresses = "a residual stream with a few very large channels, the shape "
                 "real Llama activations have";
    v.expectation =
        "the norm still returns an O(1) row, but the interval the 1/sqrt fit "
        "needs is wider than the baseline's. This vector is the reason the "
        "calibration must be re-run whenever the input distribution changes";
  } else if (name == "unscaled_weights") {
    s.scaling = WeightScaling::kUnscaled;
    v.stresses = "the failure mode the HEonGPU record documents: random "
                 "weights at full width put raw scores across a span of "
                 "hundreds";
    v.expectation =
        "THIS VECTOR IS EXPECTED TO BE UNFITTABLE. The exp interval it "
        "produces is enormous, and an encrypted SoftMax over it does not fail "
        "loudly -- HEonGPU saw values outgrow int64 at decrypt, which surfaces "
        "as a library error rather than a range error. It is here so that the "
        "calibration tooling can be shown to detect the condition instead of "
        "silently fitting it. A trained model does not present this";
    v.is_control = true;
  } else if (name == "long_window") {
    t = tokens * 4;
    v.stresses = "the causal triangle at a longer window: the denominator "
                 "range widens with the row length";
    v.expectation =
        "the reciprocal interval measured at this window is WIDER than at the "
        "baseline window. Any interval committed to at one sequence length is "
        "invalid at another, and this vector makes that visible rather than "
        "leaving it to be discovered later";
  } else {
    throw std::invalid_argument("BuildTestVector: unknown vector '" + name + "'");
  }

  v.config.Validate();
  v.weights = MakeSyntheticLayer(v.config, s, 0);
  v.activations = MakeSyntheticActivations(v.config, s, t);

  // Per-vector activation and weight surgery.
  if (name == "near_zero_norm") {
    // Shrink a quarter of the tokens towards zero, geometrically, so the
    // 1/sqrt argument sweeps from the baseline down to eps.
    for (int64_t i = 0; i < t; i += 4) {
      const double f = std::pow(10.0, -static_cast<double>(2 + (i / 4) % 6));
      for (int64_t c = 0; c < v.config.hidden_size; ++c) v.activations.At(i, c) *= f;
    }
  } else if (name == "zero_row") {
    for (int64_t c = 0; c < v.config.hidden_size; ++c) {
      v.activations.At(0, c) = 0.0;
      if (t > 2) v.activations.At(t / 2, c) = 0.0;
    }
  } else if (name == "repeated_tokens" || name == "repeated_tokens_no_rope") {
    for (int64_t i = 1; i < t; ++i)
      for (int64_t c = 0; c < v.config.hidden_size; ++c)
        v.activations.At(i, c) = v.activations.At(0, c);
  } else if (name == "large_scores") {
    ScaleMatrix(&v.weights.wq, 8.0);
    ScaleMatrix(&v.weights.wk, 8.0);
  } else if (name == "extreme_magnitudes") {
    for (int64_t i = 0; i < t; ++i)
      for (int64_t c = 0; c < v.config.hidden_size; ++c)
        v.activations.At(i, c) *= (c % 2 == 0) ? 1e-3 : 1e3;
  } else if (name == "outlier_channels") {
    Rng rng(s.seed);
    Rng osel = rng.Derive(HashName("vector_outliers"));
    const int64_t n_out = v.config.hidden_size >= 16 ? v.config.hidden_size / 64 + 1 : 1;
    for (int64_t k = 0; k < n_out; ++k) {
      const int64_t ch =
          static_cast<int64_t>(osel.NextU64() % static_cast<uint64_t>(v.config.hidden_size));
      for (int64_t i = 0; i < t; ++i) v.activations.At(i, ch) *= 30.0;
    }
  }
  return v;
}

std::vector<TestVector> BuildAllTestVectors(const Llama3Config& base,
                                            const SyntheticSpec& spec,
                                            int64_t tokens) {
  std::vector<TestVector> out;
  for (const std::string& n : TestVectorNames())
    out.push_back(BuildTestVector(n, base, spec, tokens));
  return out;
}

std::vector<InvariantResult> CheckInvariants(const RecordingSink& rec,
                                             const Llama3Config& config,
                                             const std::string& prefix) {
  std::vector<InvariantResult> out;
  auto add = [&](const char* name, bool ok, double measured, double bound,
                 const std::string& detail) {
    InvariantResult r;
    r.name = name;
    r.ok = ok;
    r.measured = measured;
    r.bound = bound;
    r.detail = detail;
    out.push_back(r);
  };

  if (!rec.Has(prefix + "attn.probs")) {
    add("attention_probs_recorded", false, 0.0, 0.0,
        "attn.probs was not recorded; the sink filtered it out");
    return out;
  }
  const Tensor& p = rec.Get(prefix + "attn.probs");
  const int64_t heads = p.Dim(0), tokens = p.Dim(1);

  // 1. Row sums.
  double worst_row = 0.0;
  int64_t worst_h = -1, worst_q = -1;
  for (int64_t h = 0; h < heads; ++h)
    for (int64_t q = 0; q < tokens; ++q) {
      double sum = 0.0;
      for (int64_t k = 0; k < tokens; ++k) sum += p.At(h, q, k);
      const double d = std::fabs(sum - 1.0);
      if (d > worst_row) {
        worst_row = d;
        worst_h = h;
        worst_q = q;
      }
    }
  {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "worst at head %lld query %lld",
                  static_cast<long long>(worst_h), static_cast<long long>(worst_q));
    add("attention_rows_sum_to_one", worst_row <= 1e-12, worst_row, 1e-12, buf);
  }

  // 2. Masked entries exactly zero.
  if (config.causal) {
    double worst_mask = 0.0;
    int64_t n_masked = 0;
    for (int64_t h = 0; h < heads; ++h)
      for (int64_t q = 0; q < tokens; ++q)
        for (int64_t k = q + 1; k < tokens; ++k) {
          ++n_masked;
          worst_mask = std::max(worst_mask, std::fabs(p.At(h, q, k)));
        }
    add("masked_probabilities_exactly_zero", worst_mask == 0.0, worst_mask, 0.0,
        std::to_string(n_masked) + " masked entries checked");
  }

  // 3. Row 0 is one-hot on key 0.
  {
    double worst = 0.0;
    for (int64_t h = 0; h < heads; ++h) worst = std::max(worst, std::fabs(p.At(h, 0, 0) - 1.0));
    add("first_row_is_one_hot", worst <= 1e-15, worst, 1e-15,
        "query 0 can attend only to key 0, so its probability must be 1");
  }

  // 4. No non-finite value anywhere that was recorded.
  {
    int64_t bad = 0;
    std::string first;
    for (const std::string& n : rec.Names()) {
      const Tensor& t = rec.Get(n);
      for (int64_t i = 0; i < t.Size(); ++i)
        if (!std::isfinite(t[i])) {
          ++bad;
          if (first.empty()) first = n;
          break;
        }
    }
    add("all_recorded_tensors_finite", bad == 0, static_cast<double>(bad), 0.0,
        bad == 0 ? "" : ("first offending tensor: " + first));
  }

  // 5. Probabilities are in [0, 1].
  {
    double lo = 0.0, hi = 0.0;
    for (int64_t i = 0; i < p.Size(); ++i) {
      lo = std::min(lo, p[i]);
      hi = std::max(hi, p[i]);
    }
    add("probabilities_in_unit_interval", lo >= 0.0 && hi <= 1.0 + 1e-15,
        std::max(-lo, hi - 1.0), 1e-15, "min " + std::to_string(lo) +
                                            ", max " + std::to_string(hi));
  }

  return out;
}

std::string TestVectorCatalogue() {
  std::ostringstream os;
  os << "Test vectors\n";
  os << "============\n";
  os << "A vector marked CONTROL is deliberately hard, or is there to break an\n";
  os << "implementation that took a shortcut. A suite in which every vector\n";
  os << "passes easily is testing nothing.\n\n";
  const Llama3Config base = Llama3Reduced(4, 2, 8, 32);
  SyntheticSpec spec;
  for (const std::string& n : TestVectorNames()) {
    TestVector v = BuildTestVector(n, base, spec, 8);
    os << (v.is_control ? "CONTROL  " : "         ") << n << "\n";
    os << "  tokens        " << v.Tokens()
       << (v.config.rope_enabled ? "" : "   (RoPE disabled)") << "\n";
    os << "  stresses      " << v.stresses << "\n";
    os << "  expectation   " << v.expectation << "\n\n";
  }
  return os.str();
}

}  // namespace oracle
