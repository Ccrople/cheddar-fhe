#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace cheddar {

/**
 * @brief Device time between pairs of events, resolved when the ledger is
 *        read rather than by a synchronise at the closing bracket.
 *
 * A preparation step that ends in `cudaDeviceSynchronize` so that a host
 * clock can time it drains the whole queue behind it, and the arithmetic
 * that follows starts on an idle card. `Begin`/`End` enqueue two events on
 * the stream instead; `Seconds` waits for the last `End` -- long past by the
 * time a per-layer ledger is printed -- and adds the elapsed times up. The
 * number is the DEVICE time the bracket took, which is what the old
 * host-clock-plus-sync number was measuring too.
 */
class EventSpanTimer {
 public:
  EventSpanTimer() = default;
  EventSpanTimer(const EventSpanTimer &) = delete;
  EventSpanTimer &operator=(const EventSpanTimer &) = delete;
  ~EventSpanTimer() {
    for (auto &s : pending_) Destroy(s);
    if (has_open_) Destroy(open_);
  }

  void Begin(cudaStream_t stream = cudaStreamLegacy) {
    if (has_open_) End(stream);
    cudaEventCreate(&open_.begin);
    cudaEventCreate(&open_.end);
    cudaEventRecord(open_.begin, stream);
    has_open_ = true;
  }

  void End(cudaStream_t stream = cudaStreamLegacy) {
    if (!has_open_) return;
    cudaEventRecord(open_.end, stream);
    pending_.push_back(open_);
    has_open_ = false;
  }

  //! The accumulated device seconds of every closed bracket.
  double Seconds() const {
    Resolve();
    return done_;
  }

  void Reset() {
    Resolve();
    done_ = 0.0;
  }

 private:
  struct Span {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
  };

  static void Destroy(Span &s) {
    if (s.begin != nullptr) cudaEventDestroy(s.begin);
    if (s.end != nullptr) cudaEventDestroy(s.end);
    s.begin = s.end = nullptr;
  }

  void Resolve() const {
    for (auto &s : pending_) {
      cudaEventSynchronize(s.end);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, s.begin, s.end);
      done_ += static_cast<double>(ms) / 1000.0;
      Destroy(s);
    }
    pending_.clear();
  }

  mutable std::vector<Span> pending_;
  mutable double done_ = 0.0;
  Span open_;
  bool has_open_ = false;
};

}  // namespace cheddar
