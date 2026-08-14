#pragma once

/**
 * @brief S3 capability probe: a level / scale / error tracer for Cheddar
 * operations.
 *
 * The tracer snapshots, after every traced operation, the metadata that a
 * Llama-3 module needs to agree on at a boundary (logN, level, scale, active
 * modulus counts, ciphertext component count, slot count, caller-supplied
 * logical tensor metadata) together with the decryption error against a
 * host-computed reference. One CSV row is appended per event.
 *
 * This header depends only on public Cheddar APIs so that it stays usable
 * after S0 freezes the module interfaces.
 *
 * Three library facts constrain what can be traced, and they are encoded here
 * rather than left to the caller:
 *   - UserInterface::Decrypt asserts num_aux == 0, so a ciphertext that is
 *     still modded up cannot be decrypted. Such an event is recorded with
 *     metadata only (decrypt_path = "skipped_aux").
 *   - UserInterface::Decrypt asserts !HasRx(), so the three-polynomial result
 *     of a bare Mult cannot be decrypted. The tracer relinearizes a throwaway
 *     copy purely to measure error (decrypt_path = "relin_probe"). That
 *     relinearization is instrumentation and is not part of the traced chain.
 *   - Encoder::Decode is host BigInt work that costs O(num_slots *
 *     num_primes^2). Trace with a small slot count; use full slot counts for
 *     timing runs only.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "UserInterface.h"
#include "core/Context.h"

namespace s3 {

using cheddar::Complex;

/**
 * @brief Logical tensor metadata. Cheddar carries no notion of a tensor, so
 * the caller supplies whatever the module contract says the ciphertext means.
 */
struct LogicalMeta {
  std::string layout;  // e.g. "SLOT", "RECT", "BATCH"
  std::string shape;   // e.g. "rows=8,cols=128"
};

struct ErrorStats {
  bool valid = false;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double snr = 0.0;
};

/**
 * @brief Compare two decoded messages elementwise.
 */
inline ErrorStats CompareDecoded(const std::vector<Complex> &expected,
                                 const std::vector<Complex> &obtained) {
  ErrorStats stats;
  const int size = static_cast<int>(std::min(expected.size(), obtained.size()));
  if (size == 0) return stats;

  double diff_sum = 0.0;
  double diff_sq_sum = 0.0;
  double ref_sq_sum = 0.0;
  for (int i = 0; i < size; ++i) {
    const Complex diff = expected[i] - obtained[i];
    const double abs_diff = std::abs(diff);
    stats.max_abs = std::max(stats.max_abs, abs_diff);
    diff_sum += abs_diff;
    diff_sq_sum += abs_diff * abs_diff;
    ref_sq_sum += std::norm(expected[i]);
  }
  stats.mean_abs = diff_sum / size;
  stats.snr = (diff_sq_sum > 0.0) ? ref_sq_sum / diff_sq_sum
                                  : std::numeric_limits<double>::infinity();
  stats.valid = true;
  return stats;
}

/**
 * @brief Device memory sample. RMM pools memory, so this reports device
 * resident reservation (the pool high-water mark) rather than live allocation.
 * On a shared GPU it also includes other processes, so the tracer reports the
 * delta against a baseline captured at construction.
 */
struct MemSample {
  double used_mib = 0.0;
  double free_mib = 0.0;
  double total_mib = 0.0;
};

inline MemSample SampleMemory() {
  MemSample sample;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return sample;
  constexpr double kMiB = 1024.0 * 1024.0;
  sample.free_mib = static_cast<double>(free_bytes) / kMiB;
  sample.total_mib = static_cast<double>(total_bytes) / kMiB;
  sample.used_mib = sample.total_mib - sample.free_mib;
  return sample;
}

/**
 * @brief Time a single callable with device synchronization on both sides.
 *
 * @return elapsed microseconds
 */
template <typename Fn>
double TimeOnce(Fn &&fn) {
  cudaDeviceSynchronize();
  const auto start = std::chrono::high_resolution_clock::now();
  fn();
  cudaDeviceSynchronize();
  const auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count();
}

/**
 * @brief Time a callable after `warm_up` untimed repetitions, then report the
 * median of `iters` timed repetitions. Warm-up and online cost are deliberately
 * kept separate: the first call on a fresh context pays lazy allocation and
 * kernel load that no steady-state Llama layer pays.
 */
struct LatencyStats {
  double warm_up_us = 0.0;  // cost of the very first (cold) call
  double median_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
  int iters = 0;
};

template <typename Fn>
LatencyStats TimeRepeated(Fn &&fn, int warm_up = 3, int iters = 10) {
  LatencyStats stats;
  stats.iters = iters;
  if (warm_up > 0) {
    stats.warm_up_us = TimeOnce(fn);
    for (int i = 1; i < warm_up; ++i) fn();
    cudaDeviceSynchronize();
  }
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) samples.push_back(TimeOnce(fn));
  std::sort(samples.begin(), samples.end());
  stats.min_us = samples.front();
  stats.max_us = samples.back();
  stats.median_us = samples[samples.size() / 2];
  return stats;
}

template <typename word>
class OpTracer {
  using Ct = cheddar::Ciphertext<word>;
  using Pt = cheddar::Plaintext<word>;

 public:
  /**
   * @brief Construct a tracer writing to `<dir>/<run_id>.csv`.
   *
   * @param context evaluation context (used for decode and for the relin probe)
   * @param interface test-only client interface (decryption, mult key)
   * @param run_id identifier written into every row and used as the file name
   */
  OpTracer(cheddar::ConstContextPtr<word> context,
           const cheddar::UserInterface<word> &interface,
           const std::string &run_id)
      : context_{context}, interface_{interface}, run_id_{run_id} {
    const char *env_dir = std::getenv("S3_TRACE_DIR");
    const std::string dir = (env_dir != nullptr) ? env_dir : ".";
    path_ = dir + "/" + run_id + ".csv";
    out_.open(path_);
    if (out_.is_open()) {
      out_ << "run_id,chain,seq,op,notes,logN,num_slots,level,max_level,"
              "num_main,num_ter,num_aux,num_q,num_total,components,scale,"
              "log2_scale,layout,shape,decrypt_path,max_abs_err,mean_abs_err,"
              "snr,log2_max_err,elapsed_us,gpu_used_mib,gpu_delta_mib,"
              "gpu_peak_mib\n";
    }
    baseline_ = SampleMemory();
    peak_used_mib_ = baseline_.used_mib;
  }

  ~OpTracer() {
    if (out_.is_open()) out_.close();
  }

  const std::string &GetPath() const { return path_; }
  void SetChain(const std::string &chain) { chain_ = chain; }
  double GetPeakUsedMiB() const { return peak_used_mib_; }
  double GetBaselineUsedMiB() const { return baseline_.used_mib; }
  double GetPeakDeltaMiB() const { return peak_used_mib_ - baseline_.used_mib; }
  int GetSeq() const { return seq_; }

  /**
   * @brief Record one traced event.
   *
   * @param op operation name
   * @param ct the ciphertext produced by the operation
   * @param expected host-computed reference message, or an empty vector to
   *                 record metadata only
   * @param meta caller-supplied logical tensor metadata
   * @param elapsed_us measured cost of the operation, or 0 if untimed
   * @param notes free-form note column
   * @return the error statistics that were recorded
   */
  ErrorStats Trace(const std::string &op, const Ct &ct,
                   const std::vector<Complex> &expected,
                   const LogicalMeta &meta = {}, double elapsed_us = 0.0,
                   const std::string &notes = "") {
    const cheddar::NPInfo np = ct.GetNP();
    const auto &param = context_->param_;

    std::string decrypt_path = "direct";
    ErrorStats stats;
    if (expected.empty()) {
      decrypt_path = "not_requested";
    } else if (np.num_aux_ != 0) {
      // Decrypt asserts num_aux == 0; a modded-up ciphertext is metadata only.
      decrypt_path = "skipped_aux";
    } else if (ct.HasRx()) {
      decrypt_path = "relin_probe";
      stats = MeasureViaRelinProbe(ct, expected);
    } else {
      stats = MeasureDirect(ct, expected);
    }

    const MemSample mem = SampleMemory();
    peak_used_mib_ = std::max(peak_used_mib_, mem.used_mib);

    // NPToLevel is only meaningful once the aux primes are gone.
    const int level = (np.num_aux_ == 0) ? param.NPToLevel(np) : -2;
    const double scale = ct.GetScale();

    // Operation names and notes contain commas, so every free-text field is
    // quoted. Without this the trace files do not parse into columns.
    auto q = [](const std::string &s) {
      std::string escaped;
      escaped.reserve(s.size() + 2);
      escaped.push_back('"');
      for (char c : s) {
        if (c == '"') escaped.push_back('"');  // RFC 4180 doubling
        escaped.push_back(c);
      }
      escaped.push_back('"');
      return escaped;
    };

    if (out_.is_open()) {
      out_ << q(run_id_) << ',' << q(chain_) << ',' << seq_ << ',' << q(op)
           << ',' << q(notes) << ',' << param.log_degree_ << ','
           << ct.GetNumSlots() << ',' << level << ',' << param.max_level_
           << ',' << np.num_main_ << ',' << np.num_ter_ << ',' << np.num_aux_
           << ',' << np.GetNumQ() << ',' << np.GetNumTotal() << ','
           << (ct.HasRx() ? 3 : 2) << ',' << std::setprecision(17) << scale
           << ',' << std::setprecision(8) << std::log2(scale) << ','
           << q(meta.layout) << ',' << q(meta.shape) << ',' << q(decrypt_path)
           << ',';
      if (stats.valid) {
        out_ << std::scientific << std::setprecision(6) << stats.max_abs << ','
             << stats.mean_abs << ',' << stats.snr << ','
             << ((stats.max_abs > 0.0) ? std::log2(stats.max_abs)
                                       : -std::numeric_limits<double>::infinity())
             << ',';
      } else {
        out_ << ",,,,";
      }
      out_ << std::fixed << std::setprecision(3) << elapsed_us << ','
           << mem.used_mib << ',' << (mem.used_mib - baseline_.used_mib) << ','
           << peak_used_mib_ << '\n';
      out_.flush();
    }

    ++seq_;
    return stats;
  }

  /**
   * @brief Time an operation and trace its result in one step.
   *
   * @param fn callable that performs the operation and writes into `ct`
   */
  template <typename Fn>
  ErrorStats TraceStep(const std::string &op, Ct &ct,
                       const std::vector<Complex> &expected, Fn &&fn,
                       const LogicalMeta &meta = {},
                       const std::string &notes = "") {
    const double elapsed_us = TimeOnce(fn);
    return Trace(op, ct, expected, meta, elapsed_us, notes);
  }

  /**
   * @brief Decrypt and decode without recording a row. Useful when a caller
   * needs the plaintext value to build the next reference.
   */
  std::vector<Complex> Peek(const Ct &ct) const {
    std::vector<Complex> decoded;
    Pt ptxt;
    interface_.Decrypt(ptxt, ct);
    context_->encoder_.Decode(decoded, ptxt);
    return decoded;
  }

 private:
  ErrorStats MeasureDirect(const Ct &ct,
                           const std::vector<Complex> &expected) const {
    return CompareDecoded(expected, Peek(ct));
  }

  // Instrumentation only: a bare Mult leaves three polynomials, which Decrypt
  // refuses. Relinearize a throwaway copy so the error is still observable.
  ErrorStats MeasureViaRelinProbe(const Ct &ct,
                                  const std::vector<Complex> &expected) const {
    Ct relinearized;
    context_->Relinearize(relinearized, ct, interface_.GetMultiplicationKey());
    return CompareDecoded(expected, Peek(relinearized));
  }

  cheddar::ConstContextPtr<word> context_;
  const cheddar::UserInterface<word> &interface_;
  std::string run_id_;
  std::string chain_ = "";
  std::string path_;
  std::ofstream out_;
  int seq_ = 0;
  MemSample baseline_;
  double peak_used_mib_ = 0.0;
};

}  // namespace s3
