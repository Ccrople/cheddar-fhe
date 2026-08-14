// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for every reference operation.
//
// Testing discipline carried over from the HEonGPU record, because it was
// learned the hard way there:
//   - validate against a closed form or a hand-computed value, never against
//     another run of the same code;
//   - include controls that are SUPPOSED to distinguish, so a test that also
//     passes for the wrong implementation is caught;
//   - test the asymmetric case deliberately -- a property symmetric in two
//     things is untestable when the two are equal.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "Check.h"
#include "oracle/Calibrate.h"
#include "oracle/Cheby.h"
#include "oracle/Compare.h"
#include "oracle/Config.h"
#include "oracle/Json.h"
#include "oracle/Ops.h"
#include "oracle/Rng.h"
#include "oracle/Serialize.h"
#include "oracle/Vectors.h"
#include "oracle/Weights.h"

using namespace oracle;

namespace {

Llama3Config Small() { return Llama3Reduced(4, 2, 8, 24); }

Tensor RandomTensor(int64_t rows, int64_t cols, uint64_t seed) {
  Rng rng(seed);
  Tensor t({rows, cols});
  for (int64_t i = 0; i < t.Size(); ++i) t[i] = rng.Normal();
  return t;
}

std::string TempDir() {
  const std::string d =
      (std::filesystem::temp_directory_path() / "cheddar_oracle_test").string();
  std::string err;
  EnsureDirectory(d, &err);
  return d;
}

}  // namespace

// ===========================================================================
// Configuration
// ===========================================================================

TEST(Config, MatchesRepositoryEvidence) {
  const Llama3Config c = Llama3_8B();
  CHECK_EQ(c.vocab_size, 128256);
  CHECK_EQ(c.hidden_size, 4096);
  CHECK_EQ(c.intermediate_size, 14336);
  CHECK_EQ(c.num_layers, 32);
  CHECK_EQ(c.num_heads, 32);
  CHECK_EQ(c.num_kv_heads, 8);
  CHECK_EQ(c.head_dim, 128);
  CHECK_EQ(c.rms_norm_eps, 1e-5);
  CHECK_EQ(c.rope_theta, 500000.0);
  CHECK_EQ(c.max_position_embeddings, 8192);
  // Derived, and independent of the constants above.
  CHECK_EQ(c.QChannels(), 4096);
  CHECK_EQ(c.KvChannels(), 1024);
  CHECK_EQ(c.GqaGroup(), 4);
  CHECK_NEAR(c.AttentionScale(), 1.0 / std::sqrt(128.0), 0.0);
}

TEST(Config, ValidateRejectsInconsistentShapes) {
  Llama3Config c = Llama3_8B();
  c.num_kv_heads = 7;  // 32 % 7 != 0
  CHECK_THROWS(c.Validate());

  c = Llama3_8B();
  c.head_dim = 127;  // odd: RoPE pairs channels
  CHECK_THROWS(c.Validate());

  c = Llama3_8B();
  c.hidden_size = 4000;  // != heads * head_dim
  CHECK_THROWS(c.Validate());
}

TEST(Config, ReducedPreservesStructure) {
  const Llama3Config c = Llama3Reduced(4, 2, 8, 24);
  CHECK_EQ(c.hidden_size, 32);
  CHECK_EQ(c.GqaGroup(), 2);
  CHECK_EQ(c.rms_norm_eps, Llama3_8B().rms_norm_eps);
  CHECK_EQ(c.rope_theta, Llama3_8B().rope_theta);
}

// ===========================================================================
// RMSNorm
// ===========================================================================

TEST(RmsNorm, MatchesHandComputedValue) {
  Tensor x({1, 2});
  x.At(0, 0) = 3.0;
  x.At(0, 1) = 4.0;
  const double eps = 1e-5;
  const std::vector<double> w = {1.0, 1.0};
  Tensor sum_sq, mean_sq, inv;
  const Tensor y = RmsNorm(x, w, eps, &sum_sq, &mean_sq, &inv);

  const double expect_sumsq = 25.0;
  const double expect_meansq = 25.0 / 2.0 + eps;
  const double expect_inv = 1.0 / std::sqrt(expect_meansq);
  CHECK_NEAR(sum_sq[0], expect_sumsq, 0.0);
  CHECK_NEAR(mean_sq[0], expect_meansq, 0.0);
  CHECK_NEAR(inv[0], expect_inv, 1e-16);
  CHECK_NEAR(y.At(0, 0), 3.0 * expect_inv, 1e-15);
  CHECK_NEAR(y.At(0, 1), 4.0 * expect_inv, 1e-15);
}

TEST(RmsNorm, WeightIsAppliedPerChannel) {
  Tensor x = RandomTensor(3, 4, 7);
  std::vector<double> w = {2.0, 0.5, -1.0, 3.0};
  std::vector<double> ones(4, 1.0);
  const Tensor a = RmsNorm(x, w, 1e-5);
  const Tensor b = RmsNorm(x, ones, 1e-5);
  for (int64_t t = 0; t < 3; ++t)
    for (int64_t c = 0; c < 4; ++c)
      CHECK_NEAR(a.At(t, c), b.At(t, c) * w[static_cast<size_t>(c)], 1e-14);
}

TEST(RmsNorm, IsScaleInvariantWhenEpsilonIsNegligible) {
  Tensor x = RandomTensor(4, 16, 11);
  ScaleInPlace(&x, 100.0);  // large, so eps is far below the mean square
  Tensor x10 = x;
  ScaleInPlace(&x10, 10.0);
  std::vector<double> w(16, 1.0);
  const Tensor a = RmsNorm(x, w, 1e-5);
  const Tensor b = RmsNorm(x10, w, 1e-5);
  const ErrorReport e = Compare(b, a);
  CHECK_LE(e.rel_l2, 1e-8);
}

TEST(RmsNorm, OutputHasUnitRootMeanSquare) {
  Tensor x = RandomTensor(5, 32, 13);
  std::vector<double> w(32, 1.0);
  const Tensor y = RmsNorm(x, w, 1e-12);
  for (int64_t t = 0; t < 5; ++t) {
    double s = 0.0;
    for (int64_t c = 0; c < 32; ++c) s += y.At(t, c) * y.At(t, c);
    CHECK_NEAR(std::sqrt(s / 32.0), 1.0, 1e-9);
  }
}

TEST(RmsNorm, EpsilonIsInsideTheSquareRoot) {
  // A row of exact zeros: mean_sq must be exactly eps and the output exactly
  // zero. This is the edge case a Newton-iteration inverse-sqrt diverges on.
  Tensor x({2, 4});
  x.Fill(0.0);
  std::vector<double> w(4, 1.0);
  Tensor sum_sq, mean_sq, inv;
  const Tensor y = RmsNorm(x, w, 1e-5, &sum_sq, &mean_sq, &inv);
  CHECK_EQ(sum_sq[0], 0.0);
  CHECK_EQ(mean_sq[0], 1e-5);
  CHECK_NEAR(inv[0], 1.0 / std::sqrt(1e-5), 1e-12);
  for (int64_t i = 0; i < y.Size(); ++i) CHECK_EQ(y[i], 0.0);
}

TEST(RmsNorm, RejectsWrongWeightLength) {
  Tensor x({2, 4});
  std::vector<double> w(3, 1.0);
  CHECK_THROWS(RmsNorm(x, w, 1e-5));
}

// ===========================================================================
// Linear
// ===========================================================================

TEST(Linear, MatchesHandComputedProduct) {
  Tensor x({2, 3});
  x.At(0, 0) = 1; x.At(0, 1) = 2; x.At(0, 2) = 3;
  x.At(1, 0) = 4; x.At(1, 1) = 5; x.At(1, 2) = 6;
  Tensor w({3, 2});  // [in, out]
  w.At(0, 0) = 1; w.At(0, 1) = -1;
  w.At(1, 0) = 0; w.At(1, 1) = 2;
  w.At(2, 0) = 3; w.At(2, 1) = 1;
  const Tensor y = Linear(x, w);
  CHECK_EQ(y.Dim(0), 2);
  CHECK_EQ(y.Dim(1), 2);
  CHECK_NEAR(y.At(0, 0), 1 * 1 + 2 * 0 + 3 * 3, 0.0);   // 10
  CHECK_NEAR(y.At(0, 1), 1 * -1 + 2 * 2 + 3 * 1, 0.0);  // 6
  CHECK_NEAR(y.At(1, 0), 4 * 1 + 5 * 0 + 6 * 3, 0.0);   // 22
  CHECK_NEAR(y.At(1, 1), 4 * -1 + 5 * 2 + 6 * 1, 0.0);  // 12
}

TEST(Linear, IsAdditive) {
  const Tensor a = RandomTensor(3, 5, 21);
  const Tensor b = RandomTensor(3, 5, 22);
  const Tensor w = RandomTensor(5, 4, 23);
  Tensor sum = a;
  AddInPlace(&sum, b);
  Tensor lhs = Linear(sum, w);
  Tensor rhs = Linear(a, w);
  AddInPlace(&rhs, Linear(b, w));
  CHECK_LE(Compare(lhs, rhs).rel_l2, 1e-14);
}

TEST(Linear, RejectsInnerDimensionMismatch) {
  const Tensor x = RandomTensor(2, 3, 1);
  const Tensor w = RandomTensor(4, 2, 2);
  CHECK_THROWS(Linear(x, w));
}

TEST(Linear, ZeroWeightsGiveExactlyZero) {
  const Tensor x = RandomTensor(3, 6, 31);
  Tensor w({6, 4});
  w.Fill(0.0);
  const Tensor y = Linear(x, w);
  for (int64_t i = 0; i < y.Size(); ++i) CHECK_EQ(y[i], 0.0);
}

// ===========================================================================
// RoPE
// ===========================================================================

TEST(Rope, PreservesEveryPairNorm) {
  Tensor q = RandomTensor(6, 2 * 8, 41);  // 2 heads of 8
  const Tensor before = q;
  ApplyRope(&q, 2, 8, 500000.0, 0, RopeConvention::kHalfSplit);
  for (int64_t t = 0; t < 6; ++t)
    for (int64_t h = 0; h < 2; ++h)
      for (int64_t c = 0; c < 4; ++c) {
        const int64_t a = h * 8 + c, b = h * 8 + c + 4;
        const double n0 = before.At(t, a) * before.At(t, a) +
                          before.At(t, b) * before.At(t, b);
        const double n1 = q.At(t, a) * q.At(t, a) + q.At(t, b) * q.At(t, b);
        CHECK_REL(n1, n0, 1e-13);
      }
}

TEST(Rope, DependsOnlyOnRelativePosition) {
  // The defining property of rotary embeddings:
  //   <RoPE(q, m), RoPE(k, n)> is a function of (m - n) alone.
  const int64_t hd = 8;
  Tensor q0({1, hd}), k0({1, hd});
  Rng rng(97);
  for (int64_t c = 0; c < hd; ++c) {
    q0.At(0, c) = rng.Normal();
    k0.At(0, c) = rng.Normal();
  }
  auto dot_at = [&](int64_t m, int64_t n) {
    Tensor q = q0, k = k0;
    ApplyRope(&q, 1, hd, 500000.0, m, RopeConvention::kHalfSplit);
    ApplyRope(&k, 1, hd, 500000.0, n, RopeConvention::kHalfSplit);
    double d = 0.0;
    for (int64_t c = 0; c < hd; ++c) d += q.At(0, c) * k.At(0, c);
    return d;
  };
  const double d31 = dot_at(3, 1);
  const double d53 = dot_at(5, 3);
  const double d97 = dot_at(9, 7);
  CHECK_REL(d53, d31, 1e-10);
  CHECK_REL(d97, d31, 1e-10);

  // Control: a DIFFERENT relative offset must give a different value, or the
  // test above would also pass for an implementation that does nothing.
  const double d41 = dot_at(4, 1);
  CHECK(std::fabs(d41 - d31) > 1e-6 * std::fabs(d31));
}

TEST(Rope, ConventionsDiffer) {
  // CONTROL. half_split and interleaved are both "correct Llama-3"; which one
  // applies is decided by the weight file. Both preserve every norm, so no
  // norm check distinguishes them -- this test exists so a silent collapse of
  // the two is caught.
  Tensor a = RandomTensor(4, 8, 51);
  Tensor b = a;
  ApplyRope(&a, 1, 8, 500000.0, 3, RopeConvention::kHalfSplit);
  ApplyRope(&b, 1, 8, 500000.0, 3, RopeConvention::kInterleaved);
  const ErrorReport e = Compare(a, b);
  CHECK_MSG(e.rel_l2 > 1e-3,
            "the two RoPE conventions produced (nearly) the same tensor");

  // ... and both are orthogonal, which is why the norm cannot tell them apart.
  double na = 0.0, nb = 0.0;
  for (int64_t i = 0; i < a.Size(); ++i) {
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  CHECK_REL(na, nb, 1e-12);
}

TEST(Rope, PositionZeroIsTheIdentity) {
  Tensor a = RandomTensor(3, 8, 61);
  const Tensor before = a;
  ApplyRope(&a, 1, 8, 500000.0, 0, RopeConvention::kHalfSplit);
  // Token 0 at offset 0 has angle 0 for every pair, so it is unchanged.
  for (int64_t c = 0; c < 8; ++c) CHECK_NEAR(a.At(0, c), before.At(0, c), 1e-15);
  // Later tokens are NOT unchanged -- the control for the line above.
  double moved = 0.0;
  for (int64_t c = 0; c < 8; ++c) moved += std::fabs(a.At(1, c) - before.At(1, c));
  CHECK(moved > 1e-6);
}

TEST(Rope, TableAnglesMatchTheFormula) {
  Tensor cos_t, sin_t;
  RopeTables(3, 8, 500000.0, 2, &cos_t, &sin_t);
  for (int64_t t = 0; t < 3; ++t)
    for (int64_t c = 0; c < 4; ++c) {
      const double angle = static_cast<double>(2 + t) *
                           std::pow(500000.0, -2.0 * static_cast<double>(c) / 8.0);
      CHECK_NEAR(cos_t.At(t, c), std::cos(angle), 1e-15);
      CHECK_NEAR(sin_t.At(t, c), std::sin(angle), 1e-15);
    }
}

// ===========================================================================
// Causal mask and SoftMax
// ===========================================================================

TEST(CausalMask, BoundariesAreExact) {
  for (int64_t q = 0; q < 8; ++q)
    for (int64_t k = 0; k < 8; ++k)
      CHECK_EQ(CausalAllowed(q, k, 0, 0), k <= q);
  // The diagonal is allowed, which is the boundary an off-by-one gets wrong.
  CHECK(CausalAllowed(5, 5, 0, 0));
  CHECK(!CausalAllowed(5, 6, 0, 0));
}

TEST(Softmax, RowSumsToOneAndMaskedEntriesAreExactlyZero) {
  const int64_t n = 6;
  std::vector<double> s(static_cast<size_t>(n)), out(static_cast<size_t>(n));
  std::vector<unsigned char> allowed(static_cast<size_t>(n));
  Rng rng(71);
  for (int64_t j = 0; j < n; ++j) {
    s[static_cast<size_t>(j)] = rng.Normal() * 3.0;
    allowed[static_cast<size_t>(j)] = j <= 3 ? 1 : 0;
  }
  SoftmaxRow(s.data(), allowed.data(), n, SoftmaxMode::kRowMax, 0.0, out.data());
  double sum = 0.0;
  for (int64_t j = 0; j < n; ++j) sum += out[static_cast<size_t>(j)];
  CHECK_NEAR(sum, 1.0, 1e-15);
  CHECK_EQ(out[4], 0.0);
  CHECK_EQ(out[5], 0.0);
}

TEST(Softmax, RowMaxAndFixedShiftAgree) {
  // SoftMax is shift invariant, so the two implementable variants must give
  // the same probabilities. Only their INTERMEDIATE ranges differ, and that
  // is exactly what the calibrator measures.
  const int64_t n = 8;
  std::vector<double> s(static_cast<size_t>(n)), a(static_cast<size_t>(n)),
      b(static_cast<size_t>(n));
  std::vector<unsigned char> allowed(static_cast<size_t>(n), 1);
  Rng rng(83);
  for (int64_t j = 0; j < n; ++j) s[static_cast<size_t>(j)] = rng.Normal() * 2.0;
  SoftmaxRow(s.data(), allowed.data(), n, SoftmaxMode::kRowMax, 0.0, a.data());
  SoftmaxRow(s.data(), allowed.data(), n, SoftmaxMode::kFixedShift, 1.75, b.data());
  for (int64_t j = 0; j < n; ++j)
    CHECK_REL(b[static_cast<size_t>(j)], a[static_cast<size_t>(j)], 1e-13);
}

TEST(Softmax, NaiveOverflowsWhereShiftedDoesNot) {
  // CONTROL. This is why the reference shifts, and why an encrypted SoftMax
  // must carry a shift even though it cannot compute a row maximum.
  const int64_t n = 4;
  std::vector<double> s = {900.0, 901.0, 902.0, 903.0};
  std::vector<unsigned char> allowed(static_cast<size_t>(n), 1);
  std::vector<double> stable(4), naive(4);
  SoftmaxRow(s.data(), allowed.data(), n, SoftmaxMode::kRowMax, 0.0, stable.data());
  SoftmaxRow(s.data(), allowed.data(), n, SoftmaxMode::kNaive, 0.0, naive.data());
  double sum = 0.0;
  for (double v : stable) {
    CHECK(std::isfinite(v));
    sum += v;
  }
  CHECK_NEAR(sum, 1.0, 1e-15);
  bool naive_broken = false;
  for (double v : naive)
    if (!std::isfinite(v)) naive_broken = true;
  CHECK_MSG(naive_broken, "the naive SoftMax was expected to overflow here");
}

TEST(Softmax, ExponentialsAreMaskedNotTheScores) {
  // The oracle multiplies exp(u) by the 0/1 mask, matching the encrypted
  // circuit. The consequence is that `exp_input` is defined at masked
  // positions too -- and that is what the exp fit interval must cover.
  const int64_t n = 4;
  std::vector<double> s = {0.0, 5.0, -2.0, 9.0};
  std::vector<unsigned char> allowed = {1, 1, 0, 0};
  std::vector<double> out(4), ein(4), ev(4);
  double denom = 0.0;
  SoftmaxRow(s.data(), allowed.data(), n, SoftmaxMode::kRowMax, 0.0, out.data(),
             ein.data(), ev.data(), &denom);
  // Row max over ALLOWED keys is 5.0, so the masked score 9.0 gives u = +4.
  CHECK_NEAR(ein[3], 4.0, 1e-15);
  CHECK_MSG(ein[3] > 0.0,
            "a masked position can have a positive exp argument; the fit "
            "interval must cover it");
  CHECK_EQ(ev[3], 0.0);
  CHECK_NEAR(denom, std::exp(-5.0) + 1.0, 1e-12);
}

// ===========================================================================
// SiLU / SwiGLU
// ===========================================================================

TEST(Silu, MatchesReferenceValues) {
  CHECK_EQ(Silu(0.0), 0.0);
  CHECK_NEAR(Silu(1.0), 1.0 / (1.0 + std::exp(-1.0)), 1e-15);
  CHECK_NEAR(Silu(-1.0), -1.0 / (1.0 + std::exp(1.0)), 1e-15);
  CHECK_NEAR(Silu(2.5), 2.5 / (1.0 + std::exp(-2.5)), 1e-15);
  // Large |x| must not overflow in either direction.
  CHECK(std::isfinite(Silu(-800.0)));
  CHECK_NEAR(Silu(-800.0), 0.0, 1e-300);
  CHECK_NEAR(Silu(800.0), 800.0, 1e-9);
}

TEST(Silu, IsSmoothAndHasTheExpectedMinimum) {
  // SiLU's minimum is near x = -1.2784, value about -0.2785.
  double best_x = 0.0, best = 0.0;
  for (int i = -400; i <= 0; ++i) {
    const double x = static_cast<double>(i) * 0.01;
    const double v = Silu(x);
    if (v < best) {
      best = v;
      best_x = x;
    }
  }
  CHECK_NEAR(best_x, -1.28, 0.02);
  CHECK_NEAR(best, -0.2785, 1e-3);
}

TEST(SwiGlu, MatchesManualProduct) {
  Tensor g({1, 3}), u({1, 3});
  g.At(0, 0) = 1.0; g.At(0, 1) = -2.0; g.At(0, 2) = 0.0;
  u.At(0, 0) = 2.0; u.At(0, 1) = 3.0;  u.At(0, 2) = 5.0;
  Tensor silu;
  const Tensor y = SwiGlu(g, u, &silu);
  CHECK_NEAR(silu.At(0, 0), Silu(1.0), 0.0);
  CHECK_NEAR(y.At(0, 0), Silu(1.0) * 2.0, 1e-15);
  CHECK_NEAR(y.At(0, 1), Silu(-2.0) * 3.0, 1e-15);
  CHECK_EQ(y.At(0, 2), 0.0);
}

// ===========================================================================
// GQA attention -- the constraint that must not be silently violated
// ===========================================================================

TEST(Gqa, HeadsInAGroupShareKeysAndValues) {
  // heads 4, kv_heads 2 -> group 2, so heads {0,1} read kv 0 and {2,3} read
  // kv 1. Giving heads 0 and 1 the SAME query must give them the same scores.
  //
  // This also distinguishes the correct mapping h/group from the plausible-
  // looking h%kv_heads: under h%kv_heads head 1 would read kv 1 and the scores
  // would differ.
  Llama3Config c = Llama3Reduced(4, 2, 4, 8);
  c.causal = false;
  const int64_t T = 3, hd = c.head_dim;
  Tensor q({T, c.QChannels()}), k({T, c.KvChannels()}), v({T, c.KvChannels()});
  Rng rng(1234);
  for (int64_t i = 0; i < k.Size(); ++i) k[i] = rng.Normal();
  for (int64_t i = 0; i < v.Size(); ++i) v[i] = rng.Normal();
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t cc = 0; cc < hd; ++cc) {
      const double val = rng.Normal();
      q.At(t, 0 * hd + cc) = val;  // head 0
      q.At(t, 1 * hd + cc) = val;  // head 1, identical
      q.At(t, 2 * hd + cc) = rng.Normal();
      q.At(t, 3 * hd + cc) = rng.Normal();
    }
  }
  RecordingSink rec;
  GqaAttention(q, k, v, c, 0, &rec, "T.");
  const Tensor& s = rec.Get("T.attn.scores");
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < T; ++j)
      CHECK_NEAR(s.At(1, i, j), s.At(0, i, j), 1e-15);

  // Control: kv 0 and kv 1 must actually differ, or the check above is vacuous.
  double spread = 0.0;
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < T; ++j)
      spread += std::fabs(s.At(2, i, j) - s.At(0, i, j));
  CHECK(spread > 1e-6);
}

TEST(Gqa, ReducesToMultiHeadWhenKvHeadsEqualsHeads) {
  Llama3Config c = Llama3Reduced(2, 2, 4, 8);
  const int64_t T = 4, hd = c.head_dim;
  const Tensor q = RandomTensor(T, c.QChannels(), 300);
  const Tensor k = RandomTensor(T, c.KvChannels(), 301);
  const Tensor v = RandomTensor(T, c.KvChannels(), 302);
  const Tensor got = GqaAttention(q, k, v, c, 0, nullptr, "T.");

  // Independent multi-head reference, written out here.
  Tensor want({T, c.QChannels()});
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
  for (int64_t h = 0; h < c.num_heads; ++h)
    for (int64_t i = 0; i < T; ++i) {
      std::vector<double> sc(static_cast<size_t>(T), 0.0);
      double m = -1e300;
      for (int64_t j = 0; j <= i; ++j) {
        double acc = 0.0;
        for (int64_t cc = 0; cc < hd; ++cc)
          acc += q.At(i, h * hd + cc) * k.At(j, h * hd + cc);
        sc[static_cast<size_t>(j)] = acc * scale;
        if (sc[static_cast<size_t>(j)] > m) m = sc[static_cast<size_t>(j)];
      }
      double denom = 0.0;
      for (int64_t j = 0; j <= i; ++j) denom += std::exp(sc[static_cast<size_t>(j)] - m);
      for (int64_t cc = 0; cc < hd; ++cc) {
        double acc = 0.0;
        for (int64_t j = 0; j <= i; ++j)
          acc += std::exp(sc[static_cast<size_t>(j)] - m) / denom * v.At(j, h * hd + cc);
        want.At(i, h * hd + cc) = acc;
      }
    }
  CHECK_LE(Compare(got, want).rel_l2, 1e-13);
}

TEST(Gqa, GroupMappingIsNotModulo) {
  // CONTROL, stated as a value check rather than an implementation check:
  // with heads 4 and kv_heads 2, head 2 must read kv head 1. Build k so that
  // kv head 0 is zero and kv head 1 is not; then heads 0 and 1 get zero scores
  // and heads 2 and 3 do not. Under a modulo mapping heads 0 and 2 would both
  // read kv 0 and the pattern would be different.
  Llama3Config c = Llama3Reduced(4, 2, 4, 8);
  c.causal = false;
  const int64_t T = 3, hd = c.head_dim;
  const Tensor q = RandomTensor(T, c.QChannels(), 400);
  Tensor k({T, c.KvChannels()});
  k.Fill(0.0);
  Rng rng(401);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t cc = 0; cc < hd; ++cc) k.At(t, 1 * hd + cc) = rng.Normal() + 1.0;
  const Tensor v = RandomTensor(T, c.KvChannels(), 402);

  RecordingSink rec;
  GqaAttention(q, k, v, c, 0, &rec, "T.");
  const Tensor& s = rec.Get("T.attn.scores");
  for (int64_t h : {0, 1})
    for (int64_t i = 0; i < T; ++i)
      for (int64_t j = 0; j < T; ++j) CHECK_EQ(s.At(h, i, j), 0.0);
  double nz = 0.0;
  for (int64_t h : {2, 3})
    for (int64_t i = 0; i < T; ++i)
      for (int64_t j = 0; j < T; ++j) nz += std::fabs(s.At(h, i, j));
  CHECK(nz > 1e-6);
}

TEST(Gqa, CausalMaskZerosTheUpperTriangle) {
  const Llama3Config c = Llama3Reduced(2, 1, 4, 8);
  const int64_t T = 6;
  const Tensor q = RandomTensor(T, c.QChannels(), 500);
  const Tensor k = RandomTensor(T, c.KvChannels(), 501);
  const Tensor v = RandomTensor(T, c.KvChannels(), 502);
  RecordingSink rec;
  GqaAttention(q, k, v, c, 0, &rec, "T.");
  const Tensor& p = rec.Get("T.attn.probs");
  for (int64_t h = 0; h < c.num_heads; ++h) {
    for (int64_t i = 0; i < T; ++i) {
      double sum = 0.0;
      for (int64_t j = 0; j < T; ++j) {
        if (j > i) CHECK_EQ(p.At(h, i, j), 0.0);
        sum += p.At(h, i, j);
      }
      CHECK_NEAR(sum, 1.0, 1e-15);
    }
    CHECK_NEAR(p.At(h, 0, 0), 1.0, 1e-15);
  }
}

TEST(Gqa, DenominatorAndReciprocalAreConsistent) {
  const Llama3Config c = Llama3Reduced(2, 1, 4, 8);
  const int64_t T = 5;
  const Tensor q = RandomTensor(T, c.QChannels(), 600);
  const Tensor k = RandomTensor(T, c.KvChannels(), 601);
  const Tensor v = RandomTensor(T, c.KvChannels(), 602);
  RecordingSink rec;
  GqaAttention(q, k, v, c, 0, &rec, "T.");
  const Tensor& d = rec.Get("T.attn.denominator");
  const Tensor& r = rec.Get("T.attn.reciprocal");
  for (int64_t h = 0; h < c.num_heads; ++h)
    for (int64_t i = 0; i < T; ++i) CHECK_NEAR(d.At(h, i) * r.At(h, i), 1.0, 1e-14);
  // Row 0 has exactly one unmasked key, and row-max shifting makes its
  // exponential exactly 1, so the denominator is exactly 1.
  for (int64_t h = 0; h < c.num_heads; ++h) CHECK_EQ(d.At(h, 0), 1.0);
}

// ===========================================================================
// Block, residuals, determinism
// ===========================================================================

TEST(Block, ResidualIdentityWithZeroOutputProjections) {
  // With Wo = 0 and Wdown = 0 both sublayers contribute nothing, so the block
  // must return its input EXACTLY. This isolates the two residual adds from
  // everything else in the block.
  const Llama3Config c = Small();
  SyntheticSpec spec;
  LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  w.wo.Fill(0.0);
  w.wdown.Fill(0.0);
  const Tensor x = MakeSyntheticActivations(c, spec, 5);
  const Tensor y = DecoderBlock(x, w, c, 0, nullptr, "L0.");
  for (int64_t i = 0; i < x.Size(); ++i) CHECK_EQ(y[i], x[i]);
}

TEST(Block, IsBitwiseDeterministic) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  const Tensor x = MakeSyntheticActivations(c, spec, 6);
  const Tensor a = DecoderBlock(x, w, c, 0, nullptr, "L0.");
  const Tensor b = DecoderBlock(x, w, c, 0, nullptr, "L0.");
  CHECK_EQ(Checksum(a), Checksum(b));
}

TEST(Block, EmitsEveryDeclaredBoundary) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  const Tensor x = MakeSyntheticActivations(c, spec, 4);
  RecordingSink rec;
  DecoderBlock(x, w, c, 0, &rec, "L0.");
  for (const BoundaryDecl& d : LayerBoundaries())
    CHECK_MSG(rec.Has("L0." + d.suffix),
              ("declared boundary not emitted: " + d.suffix).c_str());
  CHECK_EQ(static_cast<int64_t>(rec.Names().size()),
           static_cast<int64_t>(LayerBoundaries().size()));
}

TEST(Block, BoundaryMetadataCarriesTheFitRoles) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  const Tensor x = MakeSyntheticActivations(c, spec, 4);
  RecordingSink rec;
  DecoderBlock(x, w, c, 0, &rec, "L0.");
  CHECK_EQ(rec.MetaOf("L0.attn_norm.mean_sq").fit_role, std::string("rsqrt"));
  CHECK_EQ(rec.MetaOf("L0.ffn_norm.mean_sq").fit_role, std::string("rsqrt"));
  CHECK_EQ(rec.MetaOf("L0.attn.exp_input").fit_role, std::string("exp"));
  CHECK_EQ(rec.MetaOf("L0.attn.denominator").fit_role, std::string("reciprocal"));
  CHECK_EQ(rec.MetaOf("L0.ffn.gate").fit_role, std::string("silu"));
  CHECK_EQ(rec.MetaOf("L0.x_in").layer, 0);
}

TEST(Block, MultipleLayersChainThroughTheResidualStream) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const std::vector<LayerWeights> layers = MakeSyntheticLayers(c, spec, 3);
  const Tensor x = MakeSyntheticActivations(c, spec, 4);
  RecordingSink rec;
  const Tensor y = Forward(x, layers, c, 0, &rec);
  // Layer n's input is layer n-1's output, exactly.
  for (int64_t l = 1; l < 3; ++l) {
    const Tensor& in = rec.Get("L" + std::to_string(l) + ".x_in");
    const Tensor& prev = rec.Get("L" + std::to_string(l - 1) + ".x_out");
    CHECK_EQ(Checksum(in), Checksum(prev));
  }
  CHECK_EQ(Checksum(y), Checksum(rec.Get("L2.x_out")));
}

TEST(Block, RopeDisabledChangesTheResult) {
  // CONTROL: an attention sublayer without RoPE is a different circuit, and
  // the record is explicit that RoPE was once "wired into nothing".
  Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  const Tensor x = MakeSyntheticActivations(c, spec, 6);
  const Tensor with = DecoderBlock(x, w, c, 0, nullptr, "L0.");
  c.rope_enabled = false;
  const Tensor without = DecoderBlock(x, w, c, 0, nullptr, "L0.");
  CHECK(Compare(without, with).rel_l2 > 1e-6);
}

// ===========================================================================
// Synthetic data determinism
// ===========================================================================

TEST(Synthetic, WeightsDoNotDependOnTokenCount) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights a = MakeSyntheticLayer(c, spec, 0);
  const LayerWeights b = MakeSyntheticLayer(c, spec, 0);
  CHECK_EQ(Checksum(a.wq), Checksum(b.wq));
  CHECK_EQ(Checksum(a.wdown), Checksum(b.wdown));
  // A different layer index gives different weights.
  const LayerWeights l1 = MakeSyntheticLayer(c, spec, 1);
  CHECK(Checksum(a.wq) != Checksum(l1.wq));
}

TEST(Synthetic, ActivationsAreAPrefixWhenTokensGrow) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const Tensor small = MakeSyntheticActivations(c, spec, 3);
  const Tensor large = MakeSyntheticActivations(c, spec, 9);
  for (int64_t i = 0; i < small.Size(); ++i) CHECK_EQ(small[i], large[i]);
}

TEST(Synthetic, DifferentSeedsGiveDifferentData) {
  const Llama3Config c = Small();
  SyntheticSpec a, b;
  b.seed = a.seed + 1;
  CHECK(Checksum(MakeSyntheticLayer(c, a, 0).wq) !=
        Checksum(MakeSyntheticLayer(c, b, 0).wq));
}

TEST(Synthetic, UnitVarianceScalingKeepsScoresBounded) {
  // The modelling choice that keeps the synthetic ranges plausible. With
  // 1/sqrt(fan_in) weights a unit-variance input gives unit-variance scores;
  // with unscaled weights it does not, which is the regime the HEonGPU record
  // reports as an e^75 dynamic range.
  const Llama3Config c = Llama3Reduced(4, 2, 16, 64);
  SyntheticSpec scaled;
  SyntheticSpec raw;
  raw.scaling = WeightScaling::kUnscaled;

  auto score_span = [&](const SyntheticSpec& s) {
    const LayerWeights w = MakeSyntheticLayer(c, s, 0);
    const Tensor x = MakeSyntheticActivations(c, s, 8);
    Calibrator cal;
    DecoderBlock(x, w, c, 0, &cal, "L0.");
    const RangeStat& st = cal.Stat("L0.attn.scores");
    return st.Max() - st.Min();
  };
  const double a = score_span(scaled);
  const double b = score_span(raw);
  CHECK_MSG(a < 60.0, "scaled synthetic scores should stay in a fittable range");
  CHECK_MSG(b > a * 3.0,
            "unscaled weights should visibly widen the score span; if they do "
            "not, the scaling knob is not doing anything");
}

// ===========================================================================
// Serialization
// ===========================================================================

TEST(Serialize, RoundTripsExactly) {
  const std::string dir = TempDir();
  const std::string path = dir + "/roundtrip.tensor";
  Tensor t = RandomTensor(5, 7, 900);
  TensorMeta m;
  m.dims = "token,channel";
  m.description = "a test tensor";
  m.consumer = "nobody";
  m.fit_role = "silu";
  m.layer = 3;
  std::string err;
  CHECK_MSG(WriteTensorFile(path, "L3.test", t, m, TensorDType::kFloat64, &err),
            err.c_str());

  Tensor back;
  std::string name, meta_json;
  CHECK_MSG(ReadTensorFile(path, &back, &name, &meta_json, &err), err.c_str());
  CHECK_EQ(name, std::string("L3.test"));
  CHECK_EQ(back.ShapeString(), t.ShapeString());
  CHECK_EQ(Checksum(back), Checksum(t));
  CHECK(meta_json.find("token,channel") != std::string::npos);
  CHECK(meta_json.find("silu") != std::string::npos);
}

TEST(Serialize, Float32PayloadIsWithinFloat32Precision) {
  const std::string dir = TempDir();
  const std::string path = dir + "/f32.tensor";
  const Tensor t = RandomTensor(4, 9, 901);
  TensorMeta m;
  std::string err;
  CHECK(WriteTensorFile(path, "f32", t, m, TensorDType::kFloat32, &err));
  Tensor back;
  CHECK(ReadTensorFile(path, &back, nullptr, nullptr, &err));
  const ErrorReport e = Compare(back, t);
  CHECK_LE(e.max_rel, 1e-6);
  CHECK(e.max_rel > 0.0);  // it really was narrowed to float32
}

TEST(Serialize, DetectsCorruption) {
  const std::string dir = TempDir();
  const std::string path = dir + "/corrupt.tensor";
  const Tensor t = RandomTensor(3, 4, 902);
  TensorMeta m;
  std::string err;
  CHECK(WriteTensorFile(path, "corrupt", t, m, TensorDType::kFloat64, &err));
  // Flip one byte in the payload.
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(n - 16);
    char c;
    f.read(&c, 1);
    c = static_cast<char>(c ^ 0x01);
    f.seekp(n - 16);
    f.write(&c, 1);
  }
  Tensor back;
  CHECK(!ReadTensorFile(path, &back, nullptr, nullptr, &err));
  CHECK(err.find("checksum") != std::string::npos);
}

TEST(Serialize, RejectsForeignFiles) {
  const std::string dir = TempDir();
  const std::string path = dir + "/foreign.bin";
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const char junk[32] = {0};
    f.write(junk, sizeof(junk));
  }
  Tensor back;
  std::string err;
  CHECK(!ReadTensorFile(path, &back, nullptr, nullptr, &err));
  CHECK(err.find("magic") != std::string::npos);
}

TEST(Serialize, FileSinkWritesAManifest) {
  const std::string dir = TempDir() + "/dump";
  std::string err;
  CHECK(EnsureDirectory(dir, &err));
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  const Tensor x = MakeSyntheticActivations(c, spec, 4);
  FileSink sink(dir, TensorDType::kFloat64);
  sink.SetNameFilter(DumpSet("minimal"));
  DecoderBlock(x, w, c, 0, &sink, "L0.");
  CHECK_EQ(static_cast<int64_t>(sink.Records().size()), 3);
  CHECK(sink.Errors().empty());
  CHECK(sink.WriteManifest(Json::Object(), &err));

  std::ifstream f(dir + "/manifest.json");
  CHECK(static_cast<bool>(f));
  const std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  CHECK(text.find("L0.x_out") != std::string::npos);
  CHECK(text.find("checksum_fnv1a64") != std::string::npos);

  // And the written tensor matches the recorded one.
  RecordingSink rec;
  DecoderBlock(x, w, c, 0, &rec, "L0.");
  Tensor back;
  CHECK(ReadTensorFile(dir + "/" + TensorFileName("L0.x_out"), &back, nullptr,
                       nullptr, &err));
  CHECK_EQ(Checksum(back), Checksum(rec.Get("L0.x_out")));
}

TEST(Serialize, DumpSetsSelectDifferentTensorCounts) {
  CHECK_EQ(static_cast<int64_t>(DumpSet("minimal").size()), 3);
  CHECK(DumpSet("full").empty());  // empty filter accepts everything
  CHECK_EQ(static_cast<int64_t>(DumpSet("boundaries").size()),
           static_cast<int64_t>(LayerBoundaries().size()) - 4);
}

// ===========================================================================
// Comparison metrics
// ===========================================================================

TEST(Compare, IdenticalTensorsHaveZeroError) {
  const Tensor a = RandomTensor(6, 8, 1000);
  const ErrorReport e = Compare(a, a);
  CHECK_EQ(e.max_abs, 0.0);
  CHECK_EQ(e.rel_l2, 0.0);
  CHECK_NEAR(e.cosine, 1.0, 1e-15);
  CHECK_NEAR(e.gain, 1.0, 1e-15);
  CHECK(!e.has_nan);
}

TEST(Compare, KnownPerturbationGivesKnownMetrics) {
  Tensor ref = RandomTensor(10, 10, 1001);
  Tensor test = ref;
  const double delta = 1e-3;
  for (int64_t i = 0; i < test.Size(); ++i) test[i] += delta;
  const ErrorReport e = Compare(test, ref);
  CHECK_NEAR(e.rmse, delta, 1e-15);
  CHECK_NEAR(e.mean_abs, delta, 1e-15);
  CHECK_NEAR(e.max_abs, delta, 1e-12);
}

TEST(Compare, GainCatchesASystematicScaleDrift) {
  // rel_l2 stays small for a 1% scale drift, but `gain` names it. This is the
  // failure a mismanaged rescale produces, and it is why gain is a separate
  // acceptance criterion.
  Tensor ref = RandomTensor(20, 20, 1002);
  Tensor test = ref;
  ScaleInPlace(&test, 1.01);
  const ErrorReport e = Compare(test, ref);
  CHECK_NEAR(e.gain, 1.01, 1e-12);
  CHECK_NEAR(e.cosine, 1.0, 1e-12);  // direction unchanged
  std::string why;
  CHECK(!Accept(e, ThresholdPrimitive(), &why));
  CHECK(why.find("gain") != std::string::npos || why.find("rel_l2") != std::string::npos);
}

TEST(Compare, NonFiniteValuesAreReportedAndNeverAccepted) {
  Tensor ref = RandomTensor(4, 4, 1003);
  Tensor test = ref;
  test[5] = std::nan("");
  const ErrorReport e = Compare(test, ref);
  CHECK(e.has_nan);
  CHECK_EQ(e.nonfinite_count, 1);
  std::string why;
  CHECK(!Accept(e, ThresholdBlock(), &why));
}

TEST(Compare, PerAxisFindsTheWorstSlice) {
  Tensor ref = RandomTensor(5, 6, 1004);
  Tensor test = ref;
  for (int64_t c = 0; c < 6; ++c) test.At(3, c) += 1.0;  // token 3 is wrong
  const WorstSlice w = WorstAlongAxis(CompareAlongAxis(test, ref, 0));
  CHECK_EQ(w.index, 3);
  const WorstSlice wc = WorstAlongAxis(CompareAlongAxis(test, ref, 1));
  CHECK(wc.index >= 0);
}

TEST(Compare, ShapeMismatchThrows) {
  const Tensor a = RandomTensor(3, 4, 1);
  const Tensor b = RandomTensor(4, 3, 2);
  CHECK_THROWS(Compare(a, b));
}

TEST(Thresholds, AreOrderedAndRelative) {
  const Threshold p = ThresholdPrimitive(), m = ThresholdModule(),
                  b = ThresholdBlock(), mb = ThresholdMultiBlock(4);
  CHECK_LT(p.max_rel_l2, m.max_rel_l2);
  CHECK_LE(m.max_rel_l2, b.max_rel_l2);
  CHECK_LT(b.max_rel_l2, mb.max_rel_l2);
  CHECK_LT(p.min_cosine, 1.0 + 1e-12);
  // A module with no polynomial in it is held to the primitive tier.
  CHECK_EQ(ThresholdForModule("rope").max_rel_l2, p.max_rel_l2);
  CHECK_EQ(ThresholdForModule("linear").max_rel_l2, p.max_rel_l2);
  CHECK_EQ(ThresholdForModule("residual").max_rel_l2, p.max_rel_l2);
  // And one with a fit is not.
  CHECK(ThresholdForModule("softmax").max_rel_l2 > p.max_rel_l2);
}

// ===========================================================================
// Chebyshev fits and degree search
// ===========================================================================

TEST(Cheby, ReproducesAPolynomialExactly) {
  auto f = [](double x) { return 2.0 * x * x * x - x + 0.5; };
  const ChebyFit fit = FitChebyshev(f, -2.0, 3.0, 3);
  // Absolute tolerance, not relative: f has a root inside the interval and a
  // relative check there measures nothing but the root's position.
  for (double x = -2.0; x <= 3.0; x += 0.05) CHECK_NEAR(fit.Eval(x), f(x), 1e-10);
}

TEST(Cheby, ExponentialConvergesOnANarrowInterval) {
  auto f = [](double x) { return std::exp(x); };
  const FitQuality q7 = MeasureFit(f, FitChebyshev(f, -1.0, 0.0, 7));
  const FitQuality q15 = MeasureFit(f, FitChebyshev(f, -1.0, 0.0, 15));
  CHECK_LT(q15.max_rel_err, q7.max_rel_err);
  CHECK_LT(q15.max_rel_err, 1e-12);
  CHECK_EQ(q15.depth, 4);  // ceil(log2(16))
  CHECK_EQ(q7.depth, 3);
}

TEST(Cheby, WideIntervalIsHarderThanNarrow) {
  // The whole argument for calibration, as a measurement.
  auto f = [](double x) { return 1.0 / x; };
  const FitQuality narrow = MeasureFit(f, FitChebyshev(f, 1.0, 4.0, 15));
  const FitQuality wide = MeasureFit(f, FitChebyshev(f, 1.0, 128.0, 15));
  CHECK_LT(narrow.max_rel_err, wide.max_rel_err);
  CHECK_MSG(wide.max_rel_err > 100.0 * narrow.max_rel_err,
            "widening the reciprocal interval should cost at least two orders "
            "of magnitude of accuracy at a fixed degree");
}

TEST(Cheby, DegreeSearchReturnsTheSmallestThatMeetsTheTarget) {
  const DegreeRecommendation r =
      RecommendDegree("exp", FunctionByName("exp"), -1.0, 0.0, 1e-6,
                      /*interval_measured=*/true, "unit test");
  CHECK(r.met_target);
  CHECK(r.recommended_degree > 0);
  CHECK_LE(r.achieved_rel_err, 1e-6);
  // Everything below the recommendation must have missed the target.
  for (const FitQuality& q : r.sweep)
    if (q.degree < r.recommended_degree) CHECK(q.max_rel_err > 1e-6);
}

TEST(Cheby, DegreeSearchReportsFailureRatherThanPretending) {
  // 1/sqrt over an interval that reaches close to its branch point.
  const DegreeRecommendation r =
      RecommendDegree("rsqrt", FunctionByName("rsqrt"), 1e-8, 100.0, 1e-8,
                      /*interval_measured=*/false, "unit test");
  CHECK(!r.met_target);
  CHECK_EQ(r.recommended_degree, static_cast<int64_t>(-1));
  CHECK(!r.note.empty());
}

TEST(Cheby, FunctionByNameRejectsUnknown) {
  CHECK_THROWS(FunctionByName("tanh"));
}

TEST(Cheby, CriterionFollowsHowTheOutputPropagates) {
  CHECK(CriterionFor("rsqrt") == FitCriterion::kPointwiseRelative);
  CHECK(CriterionFor("reciprocal") == FitCriterion::kPointwiseRelative);
  CHECK(CriterionFor("exp") == FitCriterion::kRelativeToPeak);
  CHECK(CriterionFor("silu") == FitCriterion::kRelativeToPeak);
  // An unknown function gets the stricter of the two.
  CHECK(CriterionFor("whatever") == FitCriterion::kPointwiseRelative);
}

TEST(Cheby, SiluIsJudgedOnPeakBecauseItCrossesZero) {
  // SiLU has a root at x = 0, so the pointwise relative error near it measures
  // where the root is, not how good the fit is. The two metrics must therefore
  // disagree by orders of magnitude at the SAME degree -- if they ever stop
  // disagreeing, the criterion split has become pointless and should go.
  const ChebyFit fit = FitChebyshev(FunctionByName("silu"), -5.0, 5.0, 31);
  const FitQuality q = MeasureFit(FunctionByName("silu"), fit);
  CHECK(!q.single_signed);
  CHECK_MSG(q.max_rel_err > 1000.0 * q.err_over_peak,
            "the pointwise and peak-relative metrics were expected to differ "
            "sharply for a sign-changing function");

  const DegreeRecommendation r = RecommendDegree(
      "silu", FunctionByName("silu"), -5.0, 5.0, 1e-4, true, "unit test");
  CHECK(r.met_target);
  CHECK(r.criterion == FitCriterion::kRelativeToPeak);
  // And the achieved figure really is the peak-relative one.
  for (const FitQuality& s : r.sweep)
    if (s.degree == r.recommended_degree)
      CHECK_NEAR(r.achieved_rel_err, s.err_over_peak, 0.0);
}

TEST(Cheby, ReciprocalStaysOnThePointwiseCriterion) {
  // 1/x scales the answer, so a small value being relatively wrong is a
  // relatively wrong probability. The peak-relative metric would hide that,
  // and this test pins that it is not used here.
  const DegreeRecommendation r =
      RecommendDegree("reciprocal", FunctionByName("reciprocal"), 1.0, 64.0,
                      1e-4, true, "unit test");
  CHECK(r.criterion == FitCriterion::kPointwiseRelative);
  if (r.met_target)
    for (const FitQuality& s : r.sweep)
      if (s.degree == r.recommended_degree)
        CHECK_NEAR(r.achieved_rel_err, s.max_rel_err, 0.0);
}

// ===========================================================================
// Calibration
// ===========================================================================

TEST(RangeStat, ExtremesAreExactAndQuantilesAreOrdered) {
  RangeStat s("t");
  for (int i = 0; i <= 1000; ++i) s.Add(static_cast<double>(i) * 0.001);
  CHECK_EQ(s.Count(), 1001);
  CHECK_EQ(s.Min(), 0.0);
  CHECK_NEAR(s.Max(), 1.0, 1e-15);
  CHECK_NEAR(s.Mean(), 0.5, 1e-12);
  CHECK_LE(s.Quantile(0.25), s.Quantile(0.5));
  CHECK_LE(s.Quantile(0.5), s.Quantile(0.75));
  CHECK_NEAR(s.Quantile(0.5), 0.5, 1e-2);
}

TEST(RangeStat, IgnoresNonFiniteButCountsThem) {
  RangeStat s("t");
  s.Add(1.0);
  s.Add(std::nan(""));
  s.Add(2.0);
  CHECK_EQ(s.Count(), 2);
  CHECK_EQ(s.NonFinite(), 1);
  CHECK_EQ(s.Max(), 2.0);
}

TEST(Calibrate, MeasuresEveryFitRoleAndMergesAcrossLayers) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const std::vector<LayerWeights> layers = MakeSyntheticLayers(c, spec, 2);
  const Tensor x = MakeSyntheticActivations(c, spec, 6);
  Calibrator cal;
  Forward(x, layers, c, 0, &cal);
  CHECK(cal.Has("L0.attn.exp_input"));
  CHECK(cal.Has("L1.attn.exp_input"));

  const Calibrator m = cal.MergedAcrossLayers();
  CHECK(m.Has("attn.exp_input"));
  CHECK(!m.Has("L0.attn.exp_input"));
  // The merged extremes bound both layers' extremes exactly.
  CHECK_LE(m.Stat("attn.exp_input").Max(),
           std::max(cal.Stat("L0.attn.exp_input").Max(),
                    cal.Stat("L1.attn.exp_input").Max()) + 1e-15);
  CHECK(m.Stat("attn.exp_input").Represented() >
        cal.Stat("L0.attn.exp_input").Count());

  const std::vector<FitInterval> iv = DeriveFitIntervals(m, 0.05);
  CHECK_EQ(static_cast<int64_t>(iv.size()), 4);
  for (const FitInterval& f : iv) {
    CHECK(f.measured);
    CHECK_LT(f.lo, f.hi);
    CHECK(f.samples > 0);
    if (f.function == "rsqrt" || f.function == "reciprocal") CHECK(f.lo > 0.0);
  }
}

TEST(Calibrate, ReciprocalIntervalIsNarrowerThanTheWorstCase) {
  // The measured denominator range must be strictly inside [1, T], which is
  // the worst case a naive implementation would fit over. This is the
  // measurement that justifies calibrating at all.
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const int64_t T = 16;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  const Tensor x = MakeSyntheticActivations(c, spec, T);
  Calibrator cal;
  DecoderBlock(x, w, c, 0, &cal, "L0.");
  const RangeStat& d = cal.Stat("L0.attn.denominator");
  CHECK(d.Min() >= 1.0 - 1e-12);
  CHECK(d.Max() <= static_cast<double>(T) + 1e-9);
  CHECK_MSG(d.Max() < static_cast<double>(T),
            "the measured denominator range should be strictly inside the "
            "worst case [1, T]");
}

TEST(Calibrate, LongerWindowWidensTheReciprocalInterval) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  auto denom_max = [&](int64_t T) {
    const Tensor x = MakeSyntheticActivations(c, spec, T);
    Calibrator cal;
    DecoderBlock(x, w, c, 0, &cal, "L0.");
    return cal.Stat("L0.attn.denominator").Max();
  };
  CHECK(denom_max(32) > denom_max(8));
}

// ===========================================================================
// Test vectors and invariants
// ===========================================================================

TEST(Vectors, EveryVectorBuildsAndSatisfiesTheInvariants) {
  const Llama3Config base = Small();
  SyntheticSpec spec;
  for (const std::string& name : TestVectorNames()) {
    const TestVector v = BuildTestVector(name, base, spec, 8);
    RecordingSink rec;
    DecoderBlock(v.activations, v.weights, v.config, v.position_offset, &rec,
                 "L0.");
    for (const InvariantResult& r : CheckInvariants(rec, v.config, "L0."))
      CHECK_MSG(r.ok, (name + ": " + r.name + " -- " + r.detail).c_str());
  }
}

TEST(Vectors, UnknownNameThrows) {
  CHECK_THROWS(BuildTestVector("no_such_vector", Small(), SyntheticSpec(), 4));
}

TEST(Vectors, RepeatedTokensWithoutRopeGiveTheExactUniformSoftmax) {
  // A closed-form check with no reference implementation involved: identical
  // tokens and no RoPE make every unmasked score in a row equal, so row q is
  // uniform over its q+1 keys and the row-max-shifted denominator is exactly
  // q+1. This exercises the FULL worst-case reciprocal interval [1, T].
  const Llama3Config base = Small();
  SyntheticSpec spec;
  const TestVector v = BuildTestVector("repeated_tokens_no_rope", base, spec, 8);
  CHECK(!v.config.rope_enabled);
  RecordingSink rec;
  DecoderBlock(v.activations, v.weights, v.config, 0, &rec, "L0.");
  const Tensor& p = rec.Get("L0.attn.probs");
  const Tensor& d = rec.Get("L0.attn.denominator");
  const int64_t T = p.Dim(1);
  for (int64_t h = 0; h < p.Dim(0); ++h)
    for (int64_t q = 0; q < T; ++q) {
      CHECK_NEAR(d.At(h, q), static_cast<double>(q + 1), 1e-12);
      for (int64_t k = 0; k <= q; ++k)
        CHECK_REL(p.At(h, q, k), 1.0 / static_cast<double>(q + 1), 1e-14);
    }
}

TEST(Vectors, NearZeroNormStillProducesAUnitNormOutput) {
  const Llama3Config base = Small();
  SyntheticSpec spec;
  const TestVector v = BuildTestVector("near_zero_norm", base, spec, 8);
  RecordingSink rec;
  DecoderBlock(v.activations, v.weights, v.config, 0, &rec, "L0.");
  const Tensor& ms = rec.Get("L0.attn_norm.mean_sq");
  const Tensor& n = rec.Get("L0.attn_norm.out");
  // The shrunken tokens really do drive the 1/sqrt argument down.
  double lo = ms[0], hi = ms[0];
  for (int64_t i = 0; i < ms.Size(); ++i) {
    lo = std::min(lo, ms[i]);
    hi = std::max(hi, ms[i]);
  }
  CHECK_MSG(hi / lo > 100.0,
            "the near-zero vector should span at least two decades of the "
            "1/sqrt argument");
  CHECK(lo >= v.config.rms_norm_eps);
  // And the normalised output is still O(1) for every token.
  for (int64_t t = 0; t < n.Dim(0); ++t) {
    double s = 0.0;
    for (int64_t c = 0; c < n.Dim(1); ++c) s += n.At(t, c) * n.At(t, c);
    CHECK(std::sqrt(s / static_cast<double>(n.Dim(1))) < 100.0);
  }
}

TEST(Vectors, ZeroRowNormalisesToExactlyZero) {
  const Llama3Config base = Small();
  SyntheticSpec spec;
  const TestVector v = BuildTestVector("zero_row", base, spec, 8);
  RecordingSink rec;
  DecoderBlock(v.activations, v.weights, v.config, 0, &rec, "L0.");
  const Tensor& ms = rec.Get("L0.attn_norm.mean_sq");
  const Tensor& n = rec.Get("L0.attn_norm.out");
  CHECK_EQ(ms[0], v.config.rms_norm_eps);
  for (int64_t c = 0; c < n.Dim(1); ++c) CHECK_EQ(n.At(0, c), 0.0);
}

TEST(Vectors, LargeScoresWidenTheExponentialInterval) {
  const Llama3Config base = Small();
  SyntheticSpec spec;
  auto exp_span = [&](const std::string& name) {
    const TestVector v = BuildTestVector(name, base, spec, 8);
    Calibrator cal;
    DecoderBlock(v.activations, v.weights, v.config, 0, &cal, "L0.");
    const RangeStat& s = cal.Stat("L0.attn.exp_input");
    return s.Max() - s.Min();
  };
  CHECK(exp_span("large_scores") > exp_span("baseline"));
}

TEST(Vectors, UnscaledWeightsProduceAnUnfittableInterval) {
  // CONTROL. The documented HEonGPU failure, reproduced so that the tooling
  // can be shown to DETECT it rather than silently fit it.
  const Llama3Config base = Llama3Reduced(4, 2, 16, 64);
  SyntheticSpec spec;
  const TestVector v = BuildTestVector("unscaled_weights", base, spec, 8);
  Calibrator cal;
  DecoderBlock(v.activations, v.weights, v.config, 0, &cal, "L0.");
  const Calibrator m = cal.MergedAcrossLayers();
  const std::vector<FitInterval> iv = DeriveFitIntervals(m, 0.05);
  bool found = false;
  for (const FitInterval& f : iv) {
    if (f.function != "exp") continue;
    found = true;
    const DegreeRecommendation r =
        RecommendDegree("exp", FunctionByName("exp"), f.lo, f.hi, 1e-4, true,
                        f.source);
    CHECK_MSG(!r.met_target,
              "the unscaled-weight vector was expected to be unfittable at "
              "every candidate degree; if it now fits, the control has lost "
              "its meaning and the vector must be made harder");
  }
  CHECK(found);
}

TEST(Vectors, MaskBoundaryCoversEveryCausalEdge) {
  const Llama3Config base = Small();
  SyntheticSpec spec;
  const TestVector v = BuildTestVector("mask_boundary", base, spec, 8);
  CHECK_EQ(v.Tokens(), 8);
  RecordingSink rec;
  DecoderBlock(v.activations, v.weights, v.config, 0, &rec, "L0.");
  const Tensor& p = rec.Get("L0.attn.probs");
  const int64_t T = p.Dim(1);
  for (int64_t h = 0; h < p.Dim(0); ++h) {
    // Diagonal is allowed and non-zero.
    for (int64_t q = 0; q < T; ++q) CHECK(p.At(h, q, q) > 0.0);
    // Strictly upper triangle is exactly zero.
    for (int64_t q = 0; q < T; ++q)
      for (int64_t k = q + 1; k < T; ++k) CHECK_EQ(p.At(h, q, k), 0.0);
    // Last row uses every key.
    for (int64_t k = 0; k < T; ++k) CHECK(p.At(h, T - 1, k) > 0.0);
  }
}

// ===========================================================================
// Weight loading hooks
// ===========================================================================

TEST(Weights, LoaderRoundTripsAWrittenBundle) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  const LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  const std::string dir = TempDir() + "/bundle";
  std::string err;
  CHECK(EnsureDirectory(dir, &err));

  auto write_f32 = [&](const std::string& file, const double* p, int64_t n) {
    std::ofstream f(dir + "/" + file, std::ios::binary | std::ios::trunc);
    for (int64_t i = 0; i < n; ++i) {
      const float v = static_cast<float>(p[i]);
      f.write(reinterpret_cast<const char*>(&v), sizeof(float));
    }
    return static_cast<bool>(f);
  };
  CHECK(write_f32("wq.f32", w.wq.Data(), w.wq.Size()));
  CHECK(write_f32("wk.f32", w.wk.Data(), w.wk.Size()));
  CHECK(write_f32("wv.f32", w.wv.Data(), w.wv.Size()));
  CHECK(write_f32("wo.f32", w.wo.Data(), w.wo.Size()));
  CHECK(write_f32("wgate.f32", w.wgate.Data(), w.wgate.Size()));
  CHECK(write_f32("wup.f32", w.wup.Data(), w.wup.Size()));
  CHECK(write_f32("wdown.f32", w.wdown.Data(), w.wdown.Size()));
  CHECK(write_f32("attn_norm.f32", w.attn_norm.data(),
                  static_cast<int64_t>(w.attn_norm.size())));
  CHECK(write_f32("ffn_norm.f32", w.ffn_norm.data(),
                  static_cast<int64_t>(w.ffn_norm.size())));

  WeightPaths p;
  p.dir = dir;
  LayerWeights back;
  CHECK_MSG(LoadLayerWeights(p, c, &back, &err), err.c_str());
  CHECK_LE(Compare(back.wq, w.wq).max_rel, 1e-6);
  CHECK_LE(Compare(back.wdown, w.wdown).max_rel, 1e-6);
  CHECK_EQ(static_cast<int64_t>(back.attn_norm.size()), c.hidden_size);
}

TEST(Weights, LoaderReportsASizeMismatchByName) {
  const std::string dir = TempDir() + "/bad_bundle";
  std::string err;
  CHECK(EnsureDirectory(dir, &err));
  {
    std::ofstream f(dir + "/attn_norm.f32", std::ios::binary | std::ios::trunc);
    const float v = 1.0f;
    f.write(reinterpret_cast<const char*>(&v), sizeof(float));  // 1 value, need 32
  }
  WeightPaths p;
  p.dir = dir;
  LayerWeights out;
  CHECK(!LoadLayerWeights(p, Small(), &out, &err));
  CHECK(err.find("attn_norm.f32") != std::string::npos);
  CHECK(err.find("does not match this checkpoint") != std::string::npos);
}

TEST(Weights, LoaderFailsCleanlyOnAMissingDirectory) {
  WeightPaths p;
  p.dir = TempDir() + "/definitely_not_here";
  LayerWeights out;
  std::string err;
  CHECK(!LoadLayerWeights(p, Small(), &out, &err));
  CHECK(!err.empty());
}

TEST(Weights, ValidateNamesTheOffendingMatrix) {
  const Llama3Config c = Small();
  SyntheticSpec spec;
  LayerWeights w = MakeSyntheticLayer(c, spec, 0);
  w.wgate = Tensor({c.hidden_size, c.intermediate_size + 1});
  bool named = false;
  try {
    w.Validate(c);
  } catch (const std::exception& e) {
    named = std::string(e.what()).find("wgate") != std::string::npos;
  }
  CHECK(named);
}

// ===========================================================================
// JSON writer (used by every report)
// ===========================================================================

TEST(Json, WritesWellFormedOutput) {
  Json o = Json::Object();
  o.Set("name", "a \"quoted\"\nvalue");
  o.Set("n", static_cast<int64_t>(42));
  o.Set("x", 1.5);
  o.Set("nan", std::nan(""));
  o.Set("flag", true);
  Json arr = Json::Array();
  arr.Push(Json::Of(1.0));
  arr.Push(Json::Of("two"));
  o.Set("list", arr);
  const std::string s = o.Dump(2);
  CHECK(s.find("\\\"quoted\\\"") != std::string::npos);
  CHECK(s.find("\\n") != std::string::npos);
  CHECK(s.find("\"nan\": null") != std::string::npos);
  CHECK(s.find("\"flag\": true") != std::string::npos);
  CHECK(s.find("42") != std::string::npos);
}

// ===========================================================================
// Documentation surfaces -- cheap, and they catch an empty report
// ===========================================================================

TEST(Docs, EveryReportSurfaceIsNonEmpty) {
  CHECK(ConfigProvenance().size() > 500);
  CHECK(RopeConventionNote().size() > 200);
  CHECK(NumericalConventions().size() > 500);
  CHECK(SerializationFormatDoc().size() > 500);
  CHECK(PythonReaderSnippet().find("CHDORC1") != std::string::npos);
  CHECK(ThresholdRationale().size() > 500);
  CHECK(CalibrationCaveats().size() > 300);
  CHECK(LocalWeightsHowto().size() > 500);
  CHECK(TestVectorCatalogue().size() > 500);
  CHECK(LayerBoundaries().size() >= 25);
}

TEST(Docs, ProvenanceCitesTheFilesItRead) {
  const std::string p = ConfigProvenance();
  CHECK(p.find("fetch_llama3_weights.py") != std::string::npos);
  CHECK(p.find("LLAMA3_8B_LAYER_FLOW.md") != std::string::npos);
  CHECK(p.find("128256") != std::string::npos);
  CHECK(p.find("14336") != std::string::npos);
  CHECK(p.find("500000") != std::string::npos);
  // And it separates what was verified from what was assumed.
  CHECK(p.find("ASSUMED, NOT VERIFIED HERE") != std::string::npos);
}
