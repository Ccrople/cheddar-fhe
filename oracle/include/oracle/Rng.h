// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// A deterministic random source that does not depend on the standard library
// implementation.
//
// `std::mt19937_64` is portable but `std::normal_distribution` is NOT -- it is
// free to consume a different number of variates on libstdc++, libc++ and
// MSVC, so the same seed gives different tensors on different hosts. Every
// number this oracle produces therefore comes from SplitMix64 plus an
// explicit Box-Muller transform written out here.
//
// What that buys: identical synthetic weights and activations on Windows/MSVC
// (where the oracle is developed) and on Sicily/GCC-10 (where the encrypted
// tests run), to within the last ulp of libm's log/cos.

#pragma once

#include <cmath>
#include <cstdint>

namespace oracle {

class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed + 0x9E3779B97F4A7C15ULL) {}

  /// Raw 64 bits. SplitMix64, Steele-Lea-Flood.
  uint64_t NextU64() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  /// Uniform in [0, 1) from the top 53 bits.
  double Uniform() {
    return static_cast<double>(NextU64() >> 11) * (1.0 / 9007199254740992.0);
  }

  /// Uniform in [lo, hi).
  double Uniform(double lo, double hi) { return lo + (hi - lo) * Uniform(); }

  /// Standard normal, Box-Muller. Both variates of a pair are used, so the
  /// stream advances by exactly one NextU64 pair per two calls.
  double Normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    // Guard the log against exactly zero; Uniform() can return 0.0.
    double u1 = Uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    const double u2 = Uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 6.283185307179586476925286766559 * u2;
    spare_ = r * std::sin(theta);
    has_spare_ = true;
    return r * std::cos(theta);
  }

  double Normal(double mean, double stddev) { return mean + stddev * Normal(); }

  /// Deterministically derive an independent stream. Used so that changing
  /// the number of tokens does not change the weights, and vice versa.
  Rng Derive(uint64_t tag) const {
    uint64_t z = state_ ^ (tag * 0xD6E8FEB86659FD93ULL);
    z = (z ^ (z >> 32)) * 0xD6E8FEB86659FD93ULL;
    return Rng(z ^ (z >> 32));
  }

 private:
  uint64_t state_;
  double spare_ = 0.0;
  bool has_spare_ = false;
};

/// Stable 64-bit hash of a string, for deriving per-tensor streams by name.
inline uint64_t HashName(const char* s) {
  uint64_t h = 1469598103934665603ULL;
  for (; *s; ++s) {
    h ^= static_cast<unsigned char>(*s);
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace oracle
