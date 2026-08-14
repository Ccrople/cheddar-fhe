#pragma once

/**
 * @brief S3 capability probe: shared support code.
 *
 * Holds the probe fixture (which times context creation, key generation and
 * warm-up separately), a deterministic message generator, a Chebyshev fitter
 * for the Llama non-linearities, and the environment report.
 */

#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "OpTracer.h"
#include "Testbed.h"
#include "common/CommonUtils.h"
#include "extension/EvalPoly.h"
#include "extension/LinearTransform.h"

namespace s3 {

using cheddar::Complex;

// ---------------------------------------------------------------------------
// Deterministic messages
// ---------------------------------------------------------------------------

/**
 * @brief cheddar::Random seeds from std::random_device, so the probes use
 * their own fixed-seed generator to stay reproducible run to run.
 */
class DetRandom {
 public:
  explicit DetRandom(uint64_t seed = 0x5EEDC0FFEEULL) : gen_{seed} {}

  void Real(std::vector<Complex> &out, int num_slots, double lo = -1.0,
            double hi = 1.0) {
    std::uniform_real_distribution<double> dist(lo, hi);
    out.resize(num_slots);
    for (auto &v : out) v = Complex(dist(gen_), 0.0);
  }

  void ComplexMsg(std::vector<Complex> &out, int num_slots, double lo = -1.0,
                  double hi = 1.0) {
    std::uniform_real_distribution<double> dist(lo, hi);
    out.resize(num_slots);
    for (auto &v : out) v = Complex(dist(gen_), dist(gen_));
  }

 private:
  std::mt19937_64 gen_;
};

// ---------------------------------------------------------------------------
// Chebyshev fitting for the Llama non-linearities
// ---------------------------------------------------------------------------

/**
 * @brief An affine map from the Chebyshev domain [-1, 1] onto the calibrated
 * input range [lo, hi] of a Llama non-linearity.
 */
struct DomainMap {
  double lo = -1.0;
  double hi = 1.0;

  double ToRange(double t) const { return lo + (hi - lo) * (t + 1.0) * 0.5; }
  double ToUnit(double x) const { return 2.0 * (x - lo) / (hi - lo) - 1.0; }
};

/**
 * @brief Mirror of EvalPoly.cpp's RemoveZero: trailing coefficients below
 * kZeroCoeffThreshold are dropped, so the degree EvalPoly actually compiles
 * can be lower than the degree that was requested. The probe must know this
 * before it computes a target scale, because EvalPoly::Evaluate aborts the
 * process on a scale mismatch rather than returning an error.
 */
inline void TrimNearZero(std::vector<double> &coefficients) {
  while (!coefficients.empty() &&
         std::abs(coefficients.back()) < cheddar::kZeroCoeffThreshold) {
    coefficients.pop_back();
  }
  for (auto &c : coefficients) {
    if (std::abs(c) < cheddar::kZeroCoeffThreshold) c = 0.0;
  }
}

/**
 * @brief Chebyshev coefficients of `f` on [-1, 1], in the convention
 * cheddar::EvalPoly uses: f(t) = sum_k c_k T_k(t), with c_0 NOT doubled.
 * (Verified against EvalPoly::ConvertToNormalBasis, which sums c_i * T_i.)
 */
inline std::vector<double> ChebyshevFit(const std::function<double(double)> &f,
                                        int degree, int num_nodes = 0) {
  if (num_nodes <= 0) num_nodes = 4 * (degree + 1);
  std::vector<double> nodes(num_nodes), values(num_nodes);
  for (int j = 0; j < num_nodes; ++j) {
    nodes[j] = std::cos(M_PI * (j + 0.5) / num_nodes);
    values[j] = f(nodes[j]);
  }
  std::vector<double> coeffs(degree + 1, 0.0);
  for (int k = 0; k <= degree; ++k) {
    double acc = 0.0;
    for (int j = 0; j < num_nodes; ++j) {
      acc += values[j] * std::cos(M_PI * k * (j + 0.5) / num_nodes);
    }
    coeffs[k] = (k == 0) ? acc / num_nodes : 2.0 * acc / num_nodes;
  }
  return coeffs;
}

/**
 * @brief Worst-case |fit - truth| of a Chebyshev fit over [-1, 1], sampled
 * densely. Reported so that a chain's ciphertext error can be separated from
 * its approximation error.
 */
inline double ChebyshevFitError(const std::vector<double> &coeffs,
                                const std::function<double(double)> &f,
                                int samples = 4001) {
  double worst = 0.0;
  for (int i = 0; i < samples; ++i) {
    const double t = -1.0 + 2.0 * i / (samples - 1);
    // Clenshaw recurrence
    double b1 = 0.0, b2 = 0.0;
    for (int k = static_cast<int>(coeffs.size()) - 1; k >= 1; --k) {
      const double b0 = 2.0 * t * b1 - b2 + coeffs[k];
      b2 = b1;
      b1 = b0;
    }
    const double value = t * b1 - b2 + coeffs[0];
    worst = std::max(worst, std::abs(value - f(t)));
  }
  return worst;
}

/**
 * @brief The four Llama non-linearity fits, each already composed with its
 * domain map so that the ciphertext stays in [-1, 1]. Ranges are placeholders
 * pending S2's calibration report; the probe measures whatever range it is
 * given and reports it, it does not assert that these are the right ranges.
 */
struct NonlinearFit {
  std::string name;
  DomainMap domain;
  int degree;            // as requested
  int effective_degree;  // after EvalPoly's near-zero trimming
  std::function<double(double)> unit_fn;  // already domain-mapped
  std::vector<double> coeffs;
  double fit_error = 0.0;
};

inline NonlinearFit MakeFit(const std::string &name, DomainMap domain,
                            int degree,
                            const std::function<double(double)> &fn_on_range) {
  NonlinearFit fit;
  fit.name = name;
  fit.domain = domain;
  fit.degree = degree;
  fit.unit_fn = [domain, fn_on_range](double t) {
    return fn_on_range(domain.ToRange(t));
  };
  fit.coeffs = ChebyshevFit(fit.unit_fn, degree);
  TrimNearZero(fit.coeffs);
  fit.effective_degree = static_cast<int>(fit.coeffs.size()) - 1;
  fit.fit_error = ChebyshevFitError(fit.coeffs, fit.unit_fn);
  return fit;
}

/** @brief 1/sqrt(x) for RMSNorm, over a strictly positive calibrated range. */
inline NonlinearFit RsqrtFit(double lo = 0.25, double hi = 4.0,
                             int degree = 15) {
  return MakeFit("rsqrt", {lo, hi}, degree,
                 [](double x) { return 1.0 / std::sqrt(x); });
}

/** @brief exp(x) for the SoftMax numerator, over a shifted (<= 0) range. */
inline NonlinearFit ExpFit(double lo = -8.0, double hi = 0.0,
                           int degree = 15) {
  return MakeFit("exp", {lo, hi}, degree,
                 [](double x) { return std::exp(x); });
}

/** @brief 1/x for the SoftMax denominator, over a strictly positive range. */
inline NonlinearFit ReciprocalFit(double lo = 1.0, double hi = 64.0,
                                  int degree = 63) {
  return MakeFit("reciprocal", {lo, hi}, degree,
                 [](double x) { return 1.0 / x; });
}

/** @brief SiLU(x) = x * sigmoid(x) for the FFN gate. */
inline NonlinearFit SiluFit(double lo = -8.0, double hi = 8.0,
                            int degree = 15) {
  return MakeFit("silu", {lo, hi}, degree, [](double x) {
    return x / (1.0 + std::exp(-x));
  });
}

// ---------------------------------------------------------------------------
// Rotation-key inventory
// ---------------------------------------------------------------------------

/**
 * @brief The result of checking a set of desired rotation distances against
 * what Cheddar's key machinery will actually accept.
 *
 * Three library rules, read from the source, drive this:
 *   - EvkMap::GetRotationKey asserts rot_idx > 0, so index 0 has no key at
 *     all; a caller looping over offsets that includes 0 will abort.
 *   - UserInterface::PrepareRotationKey asserts 0 < rot_idx < degree/2.
 *   - Context::HRot normalizes rot_dist into [0, num_slots), so a left
 *     rotation by -k needs the key for num_slots - k, not for k.
 */
struct RotationInventory {
  int num_slots = 0;
  std::set<int> normalized;              // what must actually be generated
  std::vector<int> zero_requests;        // requested rotations that are no-ops
  std::vector<int> out_of_range;         // cannot be keyed at this ring size
  std::map<int, std::vector<int>> aliases;  // normalized -> raw requests
  int num_duplicate_requests = 0;

  int NumKeysRequired() const { return static_cast<int>(normalized.size()); }
};

/**
 * @brief Normalize a list of signed, logical rotation distances into the
 * positive key indices Cheddar requires, reporting collisions and no-ops.
 */
inline RotationInventory BuildRotationInventory(
    const std::vector<int> &requested, int num_slots) {
  RotationInventory inv;
  inv.num_slots = num_slots;
  for (int raw : requested) {
    int norm = raw % num_slots;
    if (norm < 0) norm += num_slots;
    if (norm == 0) {
      inv.zero_requests.push_back(raw);
      continue;
    }
    if (norm <= 0 || norm >= num_slots) {
      inv.out_of_range.push_back(raw);
      continue;
    }
    if (!inv.normalized.insert(norm).second) ++inv.num_duplicate_requests;
    inv.aliases[norm].push_back(raw);
  }
  return inv;
}

// ---------------------------------------------------------------------------
// Probe fixture
// ---------------------------------------------------------------------------

/**
 * @brief Testbed with phase timing.
 *
 * Testbed::SetUp builds the context and then constructs a UserInterface, which
 * samples secrets and generates the four basic evaluation keys in one go. To
 * report key generation separately, the fixture measures the total and then
 * separately times the construction of a throwaway second UserInterface on the
 * same context. Context-creation cost is the difference. The throwaway
 * interface holds different secrets and is destroyed immediately; it is never
 * used for evaluation.
 */
class ProbeBed : public Testbed<uint32_t> {
 public:
  double setup_total_us_ = 0.0;
  double basic_keygen_us_ = 0.0;
  double context_create_us_ = 0.0;

  void SetUp() override {
    setup_total_us_ = TimeOnce([&]() { Testbed<uint32_t>::SetUp(); });
    {
      std::unique_ptr<cheddar::UserInterface<uint32_t>> throwaway;
      basic_keygen_us_ = TimeOnce([&]() {
        throwaway =
            std::make_unique<cheddar::UserInterface<uint32_t>>(context_);
      });
    }
    context_create_us_ = setup_total_us_ - basic_keygen_us_;
  }

  /** @brief Time the generation of one rotation key. */
  double TimeRotationKey(int rot_idx, int max_level) {
    return TimeOnce(
        [&]() { interface_->PrepareRotationKey(rot_idx, max_level); });
  }

  /** @brief Time the generation of a whole EvkRequest. */
  double TimeRotationKeys(const cheddar::EvkRequest &req) {
    return TimeOnce([&]() { interface_->PrepareRotationKey(req); });
  }
};

// ---------------------------------------------------------------------------
// Environment report
// ---------------------------------------------------------------------------

inline std::string GitRevision() {
  const char *rev = std::getenv("S3_GIT_REV");
  return (rev != nullptr) ? rev : "unknown (set S3_GIT_REV)";
}

inline std::string BuildType() {
#ifdef NDEBUG
  return "Release/RelWithDebInfo (NDEBUG defined)";
#else
  return "Debug (NDEBUG not defined)";
#endif
}

inline void PrintEnvironment(std::ostream &os) {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  int runtime_version = 0;
  int driver_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);
  const MemSample mem = SampleMemory();

  os << "==================== S3 environment ====================\n";
  os << "git_revision      : " << GitRevision() << "\n";
  os << "build_type        : " << BuildType() << "\n";
  os << "host_compiler     : GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "."
     << __GNUC_PATCHLEVEL__ << "\n";
  os << "cuda_runtime      : " << runtime_version / 1000 << "."
     << (runtime_version % 1000) / 10 << "\n";
  os << "cuda_driver       : " << driver_version / 1000 << "."
     << (driver_version % 1000) / 10 << "\n";
  os << "gpu_index         : " << device << "\n";
  os << "gpu_name          : " << prop.name << "\n";
  os << "gpu_arch          : sm_" << prop.major << prop.minor << "\n";
  os << "gpu_total_mib     : " << mem.total_mib << "\n";
  os << "gpu_used_at_start : " << mem.used_mib << " (shared card; includes "
     << "other processes)\n";
  os << "gpu_sm_count      : " << prop.multiProcessorCount << "\n";
}

template <typename Fixture>
inline void PrintParameterSet(std::ostream &os, const Fixture &bed,
                              const char *param_name) {
  const auto &param = *bed.param_;
  os << "-------------------- parameter set --------------------\n";
  os << "param_file        : " << param_name << "\n";
  os << "word              : uint32_t\n";
  os << "log_degree (logN) : " << param.log_degree_ << "\n";
  os << "degree            : " << param.degree_ << "\n";
  os << "max_slots         : " << param.degree_ / 2 << "\n";
  os << "base_scale        : 2^" << std::log2(param.base_scale_) << "\n";
  os << "max_level         : " << param.max_level_ << "\n";
  os << "default_enc_level : " << param.default_encryption_level_ << "\n";
  os << "dnum              : " << param.dnum_ << "\n";
  os << "alpha             : " << param.alpha_ << "\n";
  os << "L                 : " << param.L_ << "\n";
  os << "num_main_primes   : " << param.main_primes_.size() << "\n";
  os << "num_ter_primes    : " << param.ter_primes_.size() << "\n";
  os << "num_aux_primes    : " << param.aux_primes_.size() << "\n";
  os << "max_num_q         : " << param.GetMaxNumQ() << "\n";
  os << "max_num_aux       : " << param.GetMaxNumAux() << "\n";
  os << "dense_hamming_wt  : " << param.GetDenseHammingWeight() << "\n";
  os << "sparse_hamming_wt : " << param.GetSparseHammingWeight() << "\n";
  os << "uses_sse          : " << param.IsUsingSparseSecretEncapsulation()
     << "\n";
}

}  // namespace s3

// gtest's TEST_P / INSTANTIATE_TEST_SUITE_P paste the fixture name into an
// identifier, so they cannot take a namespace-qualified name. Expose an
// unqualified alias for the macros to use.
using ProbeBed = s3::ProbeBed;
