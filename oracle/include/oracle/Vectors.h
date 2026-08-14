// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// Named test vectors, including the edge cases an encrypted module is most
// likely to get wrong.
//
// Each vector is a complete (config, weights, activations) triple plus a
// statement of what it stresses and what a correct encrypted module must do
// on it. Some of them are CONTROLS -- inputs that are supposed to be hard, or
// supposed to break a naive implementation. A vector whose expectation is
// "this must fail" is as useful as one that must pass; the HEonGPU record is
// explicit that a layout test which also passes for the identity map is
// testing nothing.

#pragma once

#include <string>
#include <vector>

#include "oracle/Config.h"
#include "oracle/Ops.h"
#include "oracle/Weights.h"

namespace oracle {

struct TestVector {
  std::string name;
  std::string stresses;     ///< which approximation or invariant it targets
  std::string expectation;  ///< what a correct encrypted module must produce
  /// Set when the vector is a deliberate control: it is expected to be hard,
  /// or to break an implementation that took a shortcut.
  bool is_control = false;

  Llama3Config config;
  LayerWeights weights;
  Tensor activations;  ///< [tokens, hidden]
  int64_t position_offset = 0;

  int64_t Tokens() const { return activations.Dim(0); }
};

/// All vector names, in a stable order.
const std::vector<std::string>& TestVectorNames();

/// Builds one vector. `base` supplies the shape; `tokens` the window length.
/// Throws std::invalid_argument on an unknown name.
///
/// Every vector is deterministic in (name, base, spec.seed, tokens).
TestVector BuildTestVector(const std::string& name, const Llama3Config& base,
                           const SyntheticSpec& spec, int64_t tokens);

/// All of them.
std::vector<TestVector> BuildAllTestVectors(const Llama3Config& base,
                                            const SyntheticSpec& spec,
                                            int64_t tokens);

/// Invariants that must hold for the oracle's own output on any vector.
/// Checked by the test suite and re-usable by an encrypted test (with a
/// tolerance instead of exactness).
struct InvariantResult {
  std::string name;
  bool ok = false;
  double measured = 0.0;
  double bound = 0.0;
  std::string detail;
};

/// Runs the structural invariants over a recorded layer:
///   - every attention row sums to 1
///   - every causally masked probability is exactly zero
///   - row 0 of every head is one-hot on key 0
///   - no non-finite value anywhere
///   - the residual identity holds where it should
/// `prefix` is the layer prefix used when recording, e.g. "L0.".
std::vector<InvariantResult> CheckInvariants(const RecordingSink& recorded,
                                             const Llama3Config& config,
                                             const std::string& prefix);

/// Prose description of every vector, for the report.
std::string TestVectorCatalogue();

}  // namespace oracle
