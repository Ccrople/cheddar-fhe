#pragma once

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "LadderJson.h"
#include "UserInterface.h"

#ifdef ENABLE_EXTENSION
#include "extension/BootContext.h"
#endif

using namespace cheddar;

#define __ProfileStart(name, warm_up, init)                 \
  {                                                         \
    std::cout << ">>>>> " << name << " <<<<<" << std::endl; \
    cudaDeviceSynchronize();                                \
    auto start = std::chrono::high_resolution_clock::now(); \
    for (int __w = 0; __w < warm_up + 1; __w++) {           \
      init;                                                 \
      if (__w == warm_up) {                                 \
        cudaDeviceSynchronize();                            \
        start = std::chrono::high_resolution_clock::now();  \
      }

// Actual region of interest (RoI) execution goes here

#define __ProfileEnd(name)                                                  \
  cudaDeviceSynchronize();                                                  \
  }                                                                         \
  auto end = std::chrono::high_resolution_clock::now();                     \
  std::cout << "Wall clock time (+ sync overhead): "                        \
            << std::chrono::duration_cast<std::chrono::microseconds>(end -  \
                                                                     start) \
                   .count()                                                 \
            << "us" << std::endl;                                           \
  }

void Check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "ERROR:" << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

template <typename T>
void PrintVector(const std::vector<T> &vec, int print_num = 5) {
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "[ ";
  int size = vec.size();
  if (size <= 2 * print_num) {
    for (const auto &elem : vec) {
      std::cout << std::setw(10) << elem << ", ";
    }
  } else {
    for (int i = 0; i < print_num; ++i) {
      std::cout << std::setw(10) << vec[i] << ", ";
    }
    std::cout << " ..., ";
    for (int i = size - print_num; i < size; ++i) {
      std::cout << std::setw(10) << vec[i] << ", ";
    }
  }
  std::cout << "] ( size: " << size << " )" << std::endl;
}

template <typename word>
class Testbed : public testing::TestWithParam<const char *> {
 public:
  int log_degree_;
  double default_scale_;
  int default_encryption_level_;
  std::unique_ptr<Parameter<word>> param_ = nullptr;
  ContextPtr<word> context_ = nullptr;
  std::unique_ptr<UserInterface<word>> interface_ = nullptr;
  std::vector<word> main_primes_;
  std::vector<word> ter_primes_;
  std::vector<word> aux_primes_;
  std::vector<std::pair<int, int>> level_config_;
  std::pair<int, int> additional_base_;
  int num_cts_levels_ = 0;
  int num_stc_levels_ = 0;

 protected:
  static inline constexpr double max_error_ = 1e-3;

  // The StC slack a `CHEDDAR_BOOT_LANDING` ladder asks for on top of
  // `BootSlackLevels()`: a junction landing is served by the exact ladder
  // above it (LandingLadder.h), and this is the difference.
  int landing_slack_ = 0;

  void SetUp() override {
    std::string json_path = std::string(PARAM_DIR) + "/" + GetParam();
    LadderSpec<word> spec;
    try {
      spec = ladderjson::Parse<word>(json_path);
    } catch (const std::exception &e) {
      Check(false, e.what());
    }
#ifdef ENABLE_EXTENSION
    // ONE PRESET, ANY LANDING: `CHEDDAR_BOOT_LANDING=L` cuts the ladder that
    // climbs only as far as L needs, keeping levels 0..L verbatim.
    landing_slack_ = ladderjson::ApplyLandingKnob<word>(spec, GetParam());
#endif
    log_degree_ = spec.log_degree;
    default_scale_ = spec.base_scale;
    default_encryption_level_ = spec.default_encryption_level;
    main_primes_ = spec.main_primes;
    ter_primes_ = spec.ter_primes;
    aux_primes_ = spec.aux_primes;
    level_config_ = spec.level_config;
    additional_base_ = spec.additional_base;

    // Initialize Parameter (hamming weights applied by the spec)
    param_ = spec.BuildParameter();

#ifdef ENABLE_EXTENSION
    if (spec.boot) {
      // Parsed whether or not a BootContext is built: a test that only wants
      // the transforms still has to compile them against the same level split.
      num_cts_levels_ = spec.num_cts_levels;
      num_stc_levels_ = spec.num_stc_levels;
    }

    if (spec.boot && UseBootContext()) {
      std::cout << "Bootstrapping enabled" << std::endl;
      // THE MESSAGE RATIO is EvalMod's ride height and therefore the term
      // that caps a 20-bit bootstrap: the sine's cubic leaves a relative
      // error `a * 2^(-2 ratio)` with `a = 2.58e-3` measured (1.5cv), so
      // ratio 5 is 2.5e-6 = 18.6 bits and every extra bit of ratio buys two
      // -- until the additive floor `N * 2^ratio` takes over and it turns
      // into a U. A preset states its own; the env override sweeps it.
      int log_message_ratio = spec.log_message_ratio;
      if (const char *e = std::getenv("CHEDDAR_BOOT_MSG_RATIO");
          e != nullptr && e[0] != 0) {
        log_message_ratio = std::atoi(e);
      }
      std::cout << "  boot: log_message_ratio " << log_message_ratio
                << ", double angle "
                << (spec.num_double_angle > 0
                        ? std::to_string(spec.num_double_angle)
                        : std::string("default"))
                << std::endl;
      context_ = BootContext<word>::Create(
          *param_, BootParameter(BootMaxLevel(), BootCtsLevels(),
                                 BootStcLevels(), log_message_ratio,
                                 BootSlackLevels() + landing_slack_,
                                 spec.num_double_angle, spec.initial_K));
    } else {
      context_ = Context<word>::Create(*param_);
    }
#else
    context_ = Context<word>::Create(*param_);
#endif
    interface_ = std::make_unique<UserInterface<word>>(context_);
  }

  // Level-targeted bootstrapping. A BootContext lands exactly where its
  // BootParameter's max_level puts it -- GetEndLevel() is max_level minus
  // CtS + EvalMod + StC -- so a test that wants a different landing level
  // overrides this. Climbing less far also makes every limb operation in
  // between shorter, which is the point.
  virtual int BootMaxLevel() const {
    // FREE LANDING FROM ONE PRESET. `Boot` climbs to this level and CtS,
    // EvalMod and StC all derive from it, so a shorter climb lands lower --
    // which is the whole of [Grafting] D.2's flexible output modulus. It
    // needs a ladder whose EvalMod band is stationary wherever the band
    // ends up (`ci20_family.py`); on one that is not, `EvalMod` says so by
    // asserting that its scale ran away.
    if (const char *e = std::getenv("CHEDDAR_BOOT_CLIMB");
        e != nullptr && e[0] != 0) {
      return std::atoi(e);
    }
    return param_->max_level_;
  }

  // Levels left free between EvalMod and StC. Zero reproduces every shipped
  // preset exactly; a test wanting [SYLPH]'s schedule -- non-linear work in
  // the slot domain before the conversion -- overrides it.
  //
  // THIS IS THE LANDING KNOB, and it is the one that does not disturb EvalMod.
  // A shorter CLIMB (`CHEDDAR_BOOT_CLIMB`) moves CtS, EvalMod and StC together,
  // so it drags EvalMod's band off the levels the band was mined for and the
  // recursion `s <- s^2/prod` leaves its fixed point. Slack moves only StC:
  // `GetStCStartLevel() = GetEvalModEndLevel() - slack`, so EvalMod still runs
  // top to bottom inside the band, still lands at 2^60, and `Boot` crosses the
  // gap with LevelDown -- a multiply by a level-down constant plus a rescale,
  // which leaves the DECLARED scale alone. So one parameter set lands anywhere
  // from `GetEndLevel()` down to 0 by choosing slack.
  virtual int BootSlackLevels() const {
    if (const char *e = std::getenv("CHEDDAR_BOOT_SLACK");
        e != nullptr && e[0] != 0) {
      return std::atoi(e);
    }
    return 0;
  }
  // Levels CoeffToSlot spends. The preset's count reproduces every shipped
  // bootstrap; a test whose CoeffToSlot is a different transform, or whose
  // EvalMod is wider (CHEDDAR_BOOT_DOUBLE_ANGLE), moves it so that EvalMod
  // still ends on default_encryption_level, which BootContext asserts.
  virtual int BootCtsLevels() const { return num_cts_levels_; }

  // Levels SlotToCoeff spends, the mirror of `BootCtsLevels`. The preset's
  // count reproduces every shipped bootstrap; a bench that wants the
  // level-against-time curve of the two transforms moves it, and the landing
  // level moves with it.
  virtual int BootStcLevels() const { return num_stc_levels_; }

  // Whether SetUp builds a BootContext when the preset asks for one. A test
  // that needs the extension's transforms but not the bootstrap -- the real
  // subring, where BootContext's constructor still refuses -- overrides this
  // to false and gets a plain Context out of a preset carrying boot primes.
  virtual bool UseBootContext() const { return true; }

  void TearDown() override {
    interface_.reset();
    std::cout << "Context use count (should be 1 to prevent memory leak): "
              << context_.use_count() << std::endl;
    context_.reset();
    param_.reset();
  }

 public:
  int GetDnum() const { return param_->dnum_; }
  int GetAlpha() const { return param_->alpha_; }
  int GetNumTotalLevels() const { return param_->max_level_; }

  // The levels a sweep should visit, from `lowest` upward. All of them when
  // there are few, as at logN 12; at logN 16 there are 32 and the host-side
  // CRT inside Encode/Decode is the unoptimised test-only path UserInterface
  // warns about, so visiting every one costs minutes and buys nothing. The
  // ends, the middle and the step below the top stand in for the rest.
  std::vector<int> LevelsToSweep(int lowest = 0) const {
    const int max_level = param_->max_level_;
    std::vector<int> levels;
    if (max_level - lowest <= 4) {
      for (int level = lowest; level <= max_level; level++) {
        levels.push_back(level);
      }
      return levels;
    }
    for (int level : {lowest, lowest + 1, max_level / 2, max_level - 1,
                      max_level}) {
      if (level >= lowest && level <= max_level) levels.push_back(level);
    }
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    return levels;
  }

  double DetermineScale(int level) const {
    if (level <= default_encryption_level_) {
      return param_->GetScale(level);
    } else {
      // We just use rescale prime product as the scale for test purposes.
      return param_->GetRescalePrimeProd(level);
    }
  }

  void GenerateRandomMessage(std::vector<Complex> &res, int num_slots = -1,
                             double range_min = -1.0, double range_max = 1.0,
                             bool complex = true) {
    if (num_slots == -1) num_slots = param_->MaxNumSlots();

    res.resize(num_slots);
    std::fill(res.begin(), res.end(), Complex(0, 0));
    if (complex) {
      Random::SampleUniformComplex(res.data(), num_slots, range_min, range_max);
    } else {
      Random::SampleUniformReal(res.data(), num_slots, range_min, range_max);
    }
  }

  void EncodeConstant(Constant<word> &constant, double number, int level,
                      bool mod_up = false) const {
    int num_q_primes = param_->LevelToNP(level).GetNumQ();
    int num_aux = mod_up ? GetAlpha() : 0;
    double scale = DetermineScale(level);
    context_->encoder_.EncodeConstant(constant, level, scale, number, num_aux);
  }

  void Encode(Plaintext<word> &res, const std::vector<Complex> &msg, int level,
              bool mod_up = false) const {
    int num_q_primes = param_->LevelToNP(level).GetNumQ();
    int num_p_primes = mod_up ? GetAlpha() : 0;
    double scale = DetermineScale(level);
    context_->encoder_.Encode(res, level, scale, msg, num_p_primes);
  }

  void EncodeAndEncrypt(Ciphertext<word> &res, const std::vector<Complex> &msg,
                        int level, bool mod_up = false) const {
    Plaintext<word> ptxt;
    Encode(ptxt, msg, level, mod_up);
    interface_->Encrypt(res, ptxt);
  }

  void Decode(std::vector<Complex> &res, const Plaintext<word> &ptxt) const {
    context_->encoder_.Decode(res, ptxt);
  }

  void DecryptAndDecode(std::vector<Complex> &res,
                        const Ciphertext<word> &ctxt) const {
    Plaintext<word> ptxt;
    interface_->Decrypt(ptxt, ctxt);
    context_->encoder_.Decode(res, ptxt);
  }

  void CompareMessages(const std::vector<Complex> &msg1,
                       const std::vector<Complex> &msg2, bool print = true,
                       double max_error = max_error_) const {
    int degree = 1 << log_degree_;
    if (print) {
      std::cout << std::endl;
      std::cout << "expected: ";
      PrintVector(msg1);
      std::cout << "obtained: ";
      PrintVector(msg2);
    }

    ASSERT_EQ(msg1.size(), msg2.size()) << "Different message sizes";

    int size = msg1.size();

    bool equal = true;
    double real_diff_min = msg1[0].real() - msg2[0].real();
    double real_diff_max = real_diff_min;
    double imag_diff_min = msg1[0].imag() - msg2[0].imag();
    double imag_diff_max = imag_diff_min;
    double abs_diff_min = std::abs(msg1[0] - msg2[0]);
    double abs_diff_max = abs_diff_min;

    double diff_magnitude_sum = 0;
    double diff_magnitude_sq_sum = 0;
    double msg1_magnitude_sq_sum = 0;

    for (int i = 0; i < size; ++i) {
      Complex diff = msg1[i] - msg2[i];
      if (std::abs(diff.real()) > max_error ||
          std::abs(diff.imag()) > max_error) {
        equal = false;
      }
      real_diff_min = std::min(real_diff_min, diff.real());
      real_diff_max = std::max(real_diff_max, diff.real());
      imag_diff_min = std::min(imag_diff_min, diff.imag());
      imag_diff_max = std::max(imag_diff_max, diff.imag());
      auto abs_diff = std::abs(diff);
      abs_diff_min = std::min(abs_diff_min, abs_diff);
      abs_diff_max = std::max(abs_diff_max, abs_diff);
      diff_magnitude_sum += abs_diff;
      diff_magnitude_sq_sum +=
          diff.real() * diff.real() + diff.imag() * diff.imag();
      msg1_magnitude_sq_sum +=
          msg1[i].real() * msg1[i].real() + msg1[i].imag() * msg1[i].imag();
    }

    // printing error stats
    if (print) {
      std::cout << std::scientific << std::setprecision(5);
      std::cout << "------------ Error stats (diff = expected - obtained) "
                   "------------"
                << std::endl;
      std::cout << "Diff real range: [ " << real_diff_min << ", "
                << real_diff_max << " ]" << std::endl;
      std::cout << "Diff imag range: [ " << imag_diff_min << ", "
                << imag_diff_max << " ]" << std::endl;
      std::cout << "Diff magnitude (sqrt(real^2 + imag^2)) range: ["
                << abs_diff_min << ", " << abs_diff_max << " ]" << std::endl;
      std::cout << "Average diff magnitude: " << diff_magnitude_sum / size
                << std::endl;
      std::cout << "SNR (E[(msg1 magnitude)^2] / E[(diff magnitude)^2]) = "
                << msg1_magnitude_sq_sum / diff_magnitude_sq_sum << std::endl;
      std::cout << "-----------------------------------------------------------"
                   "-------"
                << std::endl;
      std::cout << std::fixed << std::endl;
    }

    ASSERT_EQ(equal, true) << "Messages are not equal";
  }
};

using Testbed32 = Testbed<uint32_t>;
using Testbed64 = Testbed<uint64_t>;