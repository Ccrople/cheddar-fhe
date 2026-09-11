#include "core/LadderSpec.h"

#include <cmath>
#include <sstream>

#include "common/Assert.h"

namespace cheddar {

template <typename word>
double LadderSpec<word>::RescaleBits(int level) const {
  AssertTrue(level >= 1 && level <= MaxLevel(),
             "LadderSpec::RescaleBits: level out of range");
  // Parameter.cu step 5, in log2: the primes a level ADDS over the one below
  // it are the numerator, the ones it DROPS the denominator. The terminal
  // part of a prefix is ter[0..t), the main part main[0..m).
  const auto [mu, tu] = level_config[level];
  const auto [ml, tl] = level_config[level - 1];
  double bits = 0.0;
  const int td = tu - tl, md = mu - ml;
  if (td >= 0) {
    for (int j = 0; j < td; j++) bits += std::log2(double(ter_primes.at(tl + j)));
  } else {
    for (int j = 0; j < -td; j++) bits -= std::log2(double(ter_primes.at(tu + j)));
  }
  if (md >= 0) {
    for (int j = 0; j < md; j++) bits += std::log2(double(main_primes.at(ml + j)));
  } else {
    for (int j = 0; j < -md; j++) bits -= std::log2(double(main_primes.at(mu + j)));
  }
  return bits;
}

template <typename word>
std::unique_ptr<Parameter<word>> LadderSpec<word>::BuildParameter() const {
  auto param = std::make_unique<Parameter<word>>(
      log_degree, base_scale, default_encryption_level, level_config,
      main_primes, aux_primes, ter_primes, additional_base,
      conjugate_invariant);
  // Dense first: SetSparse asserts against the dense weight in force.
  if (dense_hamming_weight > 0) param->SetDenseHammingWeight(dense_hamming_weight);
  if (sparse_hamming_weight > 0) param->SetSparseHammingWeight(sparse_hamming_weight);
  return param;
}

template <typename word>
std::string LadderSpec<word>::ToJson() const {
  // Hand-written on purpose: the library does not link a JSON library, and
  // the format is the six-key preset the harnesses already parse.
  std::ostringstream o;
  auto list = [&o](const char *key, const std::vector<word> &v) {
    o << " \"" << key << "\": [";
    for (size_t i = 0; i < v.size(); i++) o << (i ? ", " : "") << v[i];
    o << "],\n";
  };
  const int log_scale = static_cast<int>(std::lround(std::log2(base_scale)));
  o << "{\n \"log_degree\": " << log_degree << ",\n"
    << " \"log_default_scale\": " << log_scale << ",\n"
    << " \"boot\": " << (boot ? "true" : "false") << ",\n";
  if (dense_hamming_weight > 0)
    o << " \"dense_hamming_weight\": " << dense_hamming_weight << ",\n";
  if (sparse_hamming_weight > 0)
    o << " \"sparse_hamming_weight\": " << sparse_hamming_weight << ",\n";
  if (boot) {
    o << " \"num_cts_levels\": " << num_cts_levels << ",\n"
      << " \"num_stc_levels\": " << num_stc_levels << ",\n";
    if (num_double_angle > 0)
      o << " \"num_double_angle\": " << num_double_angle << ",\n";
    o << " \"log_message_ratio\": " << log_message_ratio << ",\n";
    if (initial_K != 2) o << " \"initial_k\": " << initial_K << ",\n";
  }
  list("terminal_primes", ter_primes);
  list("main_primes", main_primes);
  list("auxiliary_primes", aux_primes);
  o << " \"default_encryption_level\": " << default_encryption_level << ",\n"
    << " \"level_config\": [";
  for (size_t i = 0; i < level_config.size(); i++) {
    o << (i ? ", " : "") << "[" << level_config[i].first << ", "
      << level_config[i].second << "]";
  }
  o << "],\n \"additional_base\": [" << additional_base.first << ", "
    << additional_base.second << "],\n"
    << " \"conjugate_invariant\": " << (conjugate_invariant ? "true" : "false")
    << "\n}\n";
  return o.str();
}

template struct LadderSpec<uint32_t>;
template struct LadderSpec<uint64_t>;

}  // namespace cheddar
