#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief A parameter set as DATA: everything `Parameter`'s constructor takes,
 * plus the boot layout a preset JSON carries beside it.
 *
 * `Parameter` is a device object -- its constructor allocates the per-level
 * prime tables -- and it is immutable once built, so anything that wants to
 * derive one parameter set from another (a landing ladder cut from a pool,
 * `extension/LandingLadder.h`) needs the plain vectors first. The test
 * harnesses parse a preset JSON into one of these; the library never reads
 * JSON itself.
 */
template <typename word>
struct LadderSpec {
  int log_degree = 16;
  double base_scale = 0.0;
  int default_encryption_level = 0;
  std::vector<std::pair<int, int>> level_config;
  std::vector<word> main_primes;
  std::vector<word> ter_primes;
  std::vector<word> aux_primes;
  std::pair<int, int> additional_base{0, 0};
  bool conjugate_invariant = false;
  //! 0 leaves `Parameter`'s own default (dense N/2, sparse 32).
  int dense_hamming_weight = 0;
  int sparse_hamming_weight = 0;

  // The boot layout (`boot: true` presets only).
  bool boot = false;
  int num_cts_levels = 0;
  int num_stc_levels = 0;
  int num_double_angle = 0;  //!< 0 = the process default (BootParameter)
  int log_message_ratio = 5;
  int initial_K = 2;

  int MaxLevel() const { return static_cast<int>(level_config.size()) - 1; }
  //! Where a full Boot with no slack lands: `default_encryption_level -
  //! num_stc_levels` (BootParameter::GetEndLevel with the ladder's own top).
  int Landing() const { return default_encryption_level - num_stc_levels; }

  //! log2 of `Parameter`'s rescale_prime_prod_[level], from the primes.
  double RescaleBits(int level) const;

  //! The `Parameter`, hamming weights applied.
  std::unique_ptr<Parameter<word>> BuildParameter() const;

  /**
   * @brief The preset in the `parameters/*.json` format, so a ladder built at
   * run time can be handed to the host auditor (`param_audit.py`) and to any
   * harness that reads a file.
   */
  std::string ToJson() const;
};

}  // namespace cheddar
