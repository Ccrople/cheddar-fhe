#pragma once

// A complete CKKS instance built from a parameter JSON, for tests that need
// more than one ring alive at a time.
//
// Deliberately not Testbed: Testbed builds exactly one Context per process
// from one parameter file, which is the assumption these tests exist to
// violate.

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "UserInterface.h"

namespace ringfixture {

template <typename word>
struct Ring {
  using json = nlohmann::json;

  int log_degree = 0;
  double scale = 0.0;
  int enc_level = 0;
  std::unique_ptr<cheddar::Parameter<word>> param;
  cheddar::ContextPtr<word> context;
  std::unique_ptr<cheddar::UserInterface<word>> ui;

  explicit Ring(const std::string &file) {
    std::ifstream f(std::string(PARAM_DIR) + "/" + file);
    if (!f) throw std::runtime_error("cannot open " + file);
    json j = json::parse(f);

    log_degree = j["log_degree"];
    scale = static_cast<double>(UINT64_C(1) << int(j["log_default_scale"]));
    enc_level = j["default_encryption_level"];

    std::vector<word> main_primes, ter_primes, aux_primes;
    for (const auto &p : j["main_primes"]) main_primes.push_back(p);
    if (j.contains("terminal_primes")) {
      for (const auto &p : j["terminal_primes"]) ter_primes.push_back(p);
    }
    for (const auto &p : j["auxiliary_primes"]) aux_primes.push_back(p);

    std::vector<std::pair<int, int>> level_config;
    for (const auto &pr : j["level_config"]) {
      level_config.emplace_back(pr[0], pr[1]);
    }
    std::pair<int, int> additional_base{0, 0};
    if (j.contains("additional_base")) {
      additional_base = {j["additional_base"][0], j["additional_base"][1]};
    }

    param = std::make_unique<cheddar::Parameter<word>>(
        log_degree, scale, enc_level, level_config, main_primes, aux_primes,
        ter_primes, additional_base);
    if (j.contains("dense_hamming_weight")) {
      param->SetDenseHammingWeight(int(j["dense_hamming_weight"]));
    }
    if (j.contains("sparse_hamming_weight")) {
      param->SetSparseHammingWeight(int(j["sparse_hamming_weight"]));
    }
    context = cheddar::Context<word>::Create(*param);
    ui = std::make_unique<cheddar::UserInterface<word>>(context);
  }

  int Degree() const { return 1 << log_degree; }

  // Round trip through coefficient encoding, which is the encoding both matrix
  // products use and the one whose buffer sizing depends on the degree.
  double CoeffRoundTrip(int level, uint64_t seed) const {
    const int degree = Degree();
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> coeffs(degree);
    for (auto &c : coeffs) c = dist(gen);

    cheddar::Plaintext<word> pt;
    context->encoder_.EncodeCoeff(pt, level, param->GetScale(level), coeffs);
    cheddar::Ciphertext<word> ct;
    ui->Encrypt(ct, pt);

    cheddar::Plaintext<word> back;
    ui->Decrypt(back, ct);
    std::vector<double> got;
    context->encoder_.DecodeCoeff(got, back);

    double max_abs = 0.0;
    for (int i = 0; i < degree; i++) {
      max_abs = std::max(max_abs, std::abs(got[i] - coeffs[i]));
    }
    return max_abs;
  }

  // One multiplication and one rescale, which exercise ModSwitch and the
  // element-wise kernels.
  double MultRescale(uint64_t seed) const {
    const int slots = Degree() / 2;
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<cheddar::Complex> a(slots), b(slots);
    for (int i = 0; i < slots; i++) {
      a[i] = cheddar::Complex(dist(gen), dist(gen));
      b[i] = cheddar::Complex(dist(gen), dist(gen));
    }

    const int level = enc_level;
    cheddar::Plaintext<word> pa, pb;
    context->encoder_.Encode(pa, level, param->GetScale(level), a);
    context->encoder_.Encode(pb, level, param->GetScale(level), b);
    cheddar::Ciphertext<word> ct;
    ui->Encrypt(ct, pa);

    cheddar::Ciphertext<word> prod, res;
    context->Mult(prod, ct, pb);
    context->Rescale(res, prod);

    cheddar::Plaintext<word> back;
    ui->Decrypt(back, res);
    std::vector<cheddar::Complex> got;
    context->encoder_.Decode(got, back);

    double max_abs = 0.0;
    for (int i = 0; i < slots; i++) {
      max_abs = std::max(max_abs, std::abs(got[i] - a[i] * b[i]));
    }
    return max_abs;
  }
};

}  // namespace ringfixture
