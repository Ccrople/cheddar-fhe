// Two ring degrees, one process, interleaved.
//
// This is the test the whole multi-ring change exists to pass, and until now it
// could not even be written. Cheddar kept the ring degree in three pieces of
// per-process global state, each of which let the second Context silently
// overwrite the first:
//
//   * __cm_log_degree, uploaded once behind a class-level `cm_populated_` in
//     five handlers, so every kernel ran at whichever degree got there first;
//   * Container<word>::degree_, which sized every ciphertext and plaintext
//     buffer -- a wrong value here is not wrong metadata, it is a wrong
//     allocation;
//   * MultiLevelCiphertext's static parameter and level-down constants.
//
// None of the three announced itself. Doing.md 1.5 records the first measured:
// run SmallRingNttTest unfiltered and degree 16 passes, then 15, 14, 13 and 12
// all execute at 16.
//
// So this test interleaves deliberately rather than running one ring to
// completion and then the other. Construct A, construct B, then alternate
// encrypt/operate/decrypt between them. Any surviving global would be captured
// by whichever Context was built last, and A -- built first, used after B
// exists -- is the one that breaks.
//
// Ring A is bootparam_30 at degree 65536, ring B is ringdegree12_30 at 4096,
// which is the actual pair Sylph needs: non-linear work upstairs, batch CC-MM
// and the PC-MM's parent ring downstairs.

#undef ENABLE_EXTENSION

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "UserInterface.h"

using namespace cheddar;
using word = uint32_t;

namespace {

// One complete CKKS instance. Deliberately not Testbed: Testbed builds exactly
// one Context per process, which is the assumption under test.
struct Ring {
  int log_degree = 0;
  double scale = 0.0;
  int enc_level = 0;
  std::unique_ptr<Parameter<word>> param;
  ContextPtr<word> context;
  std::unique_ptr<UserInterface<word>> ui;

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

    param = std::make_unique<Parameter<word>>(log_degree, scale, enc_level,
                                              level_config, main_primes,
                                              aux_primes, ter_primes,
                                              additional_base);
    if (j.contains("dense_hamming_weight")) {
      param->SetDenseHammingWeight(int(j["dense_hamming_weight"]));
    }
    if (j.contains("sparse_hamming_weight")) {
      param->SetSparseHammingWeight(int(j["sparse_hamming_weight"]));
    }
    context = Context<word>::Create(*param);
    ui = std::make_unique<UserInterface<word>>(context);
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

    Plaintext<word> pt;
    context->encoder_.EncodeCoeff(pt, level, param->GetScale(level), coeffs);
    Ciphertext<word> ct;
    ui->Encrypt(ct, pt);

    Plaintext<word> back;
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
  // element-wise kernels -- the two places the constant memory was read.
  double MultRescale(uint64_t seed) const {
    const int slots = Degree() / 2;
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> a(slots), b(slots);
    for (int i = 0; i < slots; i++) {
      a[i] = Complex(dist(gen), dist(gen));
      b[i] = Complex(dist(gen), dist(gen));
    }

    const int level = enc_level;
    Plaintext<word> pa, pb;
    context->encoder_.Encode(pa, level, param->GetScale(level), a);
    context->encoder_.Encode(pb, level, param->GetScale(level), b);
    Ciphertext<word> ct;
    ui->Encrypt(ct, pa);

    Ciphertext<word> prod, res;
    context->Mult(prod, ct, pb);
    context->Rescale(res, prod);

    Plaintext<word> back;
    ui->Decrypt(back, res);
    std::vector<Complex> got;
    context->encoder_.Decode(got, back);

    double max_abs = 0.0;
    for (int i = 0; i < slots; i++) {
      max_abs = std::max(max_abs, std::abs(got[i] - a[i] * b[i]));
    }
    return max_abs;
  }
};

}  // namespace

TEST(MultiRing, TwoDegreesInterleaved) {
  // Construction order matters to the bug: A first, then B. Every global would
  // now hold B's degree, and A is what fails.
  Ring a("bootparam_30.json");
  Ring b("ringdegree12_30.json");

  ASSERT_EQ(a.Degree(), 65536);
  ASSERT_EQ(b.Degree(), 4096);
  std::cout << "ring A degree " << a.Degree() << ", ring B degree "
            << b.Degree() << std::endl;

  // The NPInfo now carries the degree, so a container knows its own ring.
  EXPECT_EQ(a.param->LevelToNP(0).degree_, 65536);
  EXPECT_EQ(b.param->LevelToNP(0).degree_, 4096);
  // ... and two rings never compare equal, which turns a cross-ring operand
  // into a failure at the NP assertions the evaluation API already makes.
  EXPECT_FALSE(a.param->LevelToNP(0) == b.param->LevelToNP(0));

  // Interleaved, alternating, and each ring used after the other was touched.
  for (int round = 0; round < 2; round++) {
    const double a_coeff = a.CoeffRoundTrip(0, 100 + round);
    const double b_coeff = b.CoeffRoundTrip(0, 200 + round);
    std::cout << "round " << round << " coeff round trip: A " << a_coeff
              << ", B " << b_coeff << std::endl;
    EXPECT_LT(a_coeff, 1e-5) << "ring A (65536) is wrong at round " << round;
    EXPECT_LT(b_coeff, 1e-5) << "ring B (4096) is wrong at round " << round;

    const double a_mult = a.MultRescale(300 + round);
    const double b_mult = b.MultRescale(400 + round);
    std::cout << "round " << round << " mult+rescale:     A " << a_mult
              << ", B " << b_mult << std::endl;
    EXPECT_LT(a_mult, 1e-3) << "ring A (65536) is wrong at round " << round;
    EXPECT_LT(b_mult, 1e-3) << "ring B (4096) is wrong at round " << round;
  }
}

// Same thing with the construction order reversed, because "the last Context
// wins" and "the first Context wins" are different bugs and only one of them
// is caught by a fixed order.
TEST(MultiRing, TwoDegreesReversedConstructionOrder) {
  Ring b("ringdegree12_30.json");
  Ring a("bootparam_30.json");

  const double b_coeff = b.CoeffRoundTrip(0, 500);
  const double a_coeff = a.CoeffRoundTrip(0, 600);
  std::cout << "reversed order coeff round trip: A " << a_coeff << ", B "
            << b_coeff << std::endl;
  EXPECT_LT(a_coeff, 1e-5);
  EXPECT_LT(b_coeff, 1e-5);
}
