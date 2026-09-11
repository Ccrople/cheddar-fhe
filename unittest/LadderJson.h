#pragma once

// A preset JSON as a `LadderSpec`, and THE LANDING KNOB.
//
// `Testbed` and `RingFixture` each parsed the same six-key JSON on their own;
// this is that parser once, into the library's data form of a parameter set,
// so that both harnesses can hand the spec to `LandingLadder` before building
// the `Parameter` from it.

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "core/LadderSpec.h"

#ifdef ENABLE_EXTENSION
#include "extension/LandingLadder.h"
#endif

namespace ladderjson {

template <typename word>
cheddar::LadderSpec<word> Parse(const std::string &path) {
  using json = nlohmann::json;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open parameter JSON " + path);
  json j = json::parse(f);
  auto need = [&](const char *key) {
    if (!j.contains(key))
      throw std::runtime_error("missing " + std::string(key) + " in " + path);
  };
  need("log_degree");
  need("log_default_scale");
  need("default_encryption_level");
  need("main_primes");
  need("auxiliary_primes");
  need("level_config");

  cheddar::LadderSpec<word> s;
  s.log_degree = j["log_degree"];
  s.base_scale = static_cast<double>(UINT64_C(1) << int(j["log_default_scale"]));
  s.default_encryption_level = j["default_encryption_level"];
  for (const auto &p : j["main_primes"]) s.main_primes.push_back(p);
  if (j.contains("terminal_primes"))
    for (const auto &p : j["terminal_primes"]) s.ter_primes.push_back(p);
  for (const auto &p : j["auxiliary_primes"]) s.aux_primes.push_back(p);
  for (const auto &pr : j["level_config"]) {
    if (!pr.is_array() || pr.size() != 2)
      throw std::runtime_error("level_config should be an array of pairs");
    s.level_config.emplace_back(pr[0], pr[1]);
  }
  if (j.contains("additional_base"))
    s.additional_base = {j["additional_base"][0], j["additional_base"][1]};
  // A second parser has to carry the flag too: without it a
  // conjugate-invariant preset loads as a well-formed ORDINARY ring -- 1 mod
  // 4N implies 1 mod 2N, so nothing rejects the primes -- and every test
  // runs, passes, and tests the wrong ring.
  s.conjugate_invariant =
      j.contains("conjugate_invariant") && bool(j["conjugate_invariant"]);
  if (j.contains("dense_hamming_weight"))
    s.dense_hamming_weight = int(j["dense_hamming_weight"]);
  if (j.contains("sparse_hamming_weight"))
    s.sparse_hamming_weight = int(j["sparse_hamming_weight"]);

  s.boot = j.contains("boot") && bool(j["boot"]);
  if (s.boot) {
    need("num_cts_levels");
    need("num_stc_levels");
    s.num_cts_levels = j["num_cts_levels"];
    s.num_stc_levels = j["num_stc_levels"];
    // A preset may pin EvalMod's double-angle count (K = 32 / K = 64
    // ladders); 0 leaves the process default. The message ratio is a
    // precision knob (EvalMod's ride height, a U with an optimum) that a
    // preset states for itself; `initial_k` buys the same K with fewer
    // double angles, each of which costs ~1.2 bits.
    if (j.contains("num_double_angle"))
      s.num_double_angle = int(j["num_double_angle"]);
    if (j.contains("log_message_ratio"))
      s.log_message_ratio = int(j["log_message_ratio"]);
    if (j.contains("initial_k")) s.initial_K = int(j["initial_k"]);
  }
  return s;
}

#ifdef ENABLE_EXTENSION
/**
 * @brief `CHEDDAR_BOOT_LANDING=L`: rebuild a boot preset's ladder so that its
 * bootstrap climbs only as far as landing L needs (`LandingLadder`). One
 * preset file, any landing, the short climb -- the pool's levels 0..L are
 * kept verbatim, so what the tests encrypt and decrypt is the same ring.
 *
 * `CHEDDAR_BOOT_LANDING_JUNCTION=1` fills a junction landing's bottom band
 * pair from the pool's spare mains instead of taking slack from the exact
 * ladder above it.
 *
 * @return the StC slack the ladder needs on top of whatever the harness
 *         adds, so that the boot lands exactly at L
 */
template <typename word>
int ApplyLandingKnob(cheddar::LadderSpec<word> &spec, const std::string &name) {
  const char *e = std::getenv("CHEDDAR_BOOT_LANDING");
  if (e == nullptr || e[0] == 0 || !spec.boot) return 0;
  const int landing = std::atoi(e);
  const char *jf = std::getenv("CHEDDAR_BOOT_LANDING_JUNCTION");
  const auto policy = (jf != nullptr && jf[0] == '1')
                          ? cheddar::JunctionPolicy::kFillJunction
                          : cheddar::JunctionPolicy::kStationaryOnly;
  cheddar::LandingLadder<word> ladder(spec);
  auto r = ladder.ForLanding(landing, policy);
  std::cout << "[landing ladder] " << name << ": "
            << cheddar::LandingLadder<word>::Describe(r) << std::endl;
  spec = r.spec;
  return r.slack;
}
#endif

}  // namespace ladderjson
