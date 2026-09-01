#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace cheddar {

/**
 * @brief The streams of the layer pipeline, and the fences between them.
 *
 * Every kernel the library launches goes to the LEGACY default stream, and
 * that is the compute stream: nothing here moves it. What is added is two
 * NON-BLOCKING streams beside it -- a copy stream for the next layer's
 * host-to-device traffic and an encode stream for its device encodes -- and
 * the one property that makes them worth having: a stream created with
 * `cudaStreamNonBlocking` does not synchronise with the legacy stream. A
 * blocking stream would, implicitly and in both directions, and then the
 * "overlap" would be the same serial queue drawn in three colours.
 *
 * Dependencies between the three are events (`Event::Record` on the
 * producer, `Event::WaitOn` the consumer); the host never waits on any of
 * them in steady state. There is no way to raise the legacy stream's
 * priority (it is fixed at the default, which is the LOWEST), so `Encode()`
 * cannot be put below it: the streams are created at the default priority
 * and the scheduling policy lives in `IdleWindow` and in chunking instead. On
 * a build where the library ran on a created high-priority stream this class
 * is where that would change.
 */
class PipelineStreams {
 public:
  static PipelineStreams &Get();

  static cudaStream_t Compute() { return cudaStreamLegacy; }
  cudaStream_t Copy() const { return copy_; }
  cudaStream_t Encode() const { return encode_; }

  /// The priority each stream was created with (0 is the default and the
  /// lowest; more negative is higher).
  int CopyPriority() const { return copy_priority_; }
  int EncodePriority() const { return encode_priority_; }

  PipelineStreams(const PipelineStreams &) = delete;
  PipelineStreams &operator=(const PipelineStreams &) = delete;

 private:
  PipelineStreams();
  cudaStream_t copy_ = nullptr;
  cudaStream_t encode_ = nullptr;
  int copy_priority_ = 0;
  int encode_priority_ = 0;
};

/**
 * @brief One CUDA event, owned. Records on a producer stream, waited on by a
 * consumer stream; the host only asks whether it has passed.
 *
 * Dedicated events rather than a shared ring: a handle that is kept and
 * waited on later (an arena slot's "free" mark) must not be re-recorded by
 * an unrelated fence in between, or the later wait covers the wrong work.
 */
class Event {
 public:
  Event();
  ~Event();
  Event(Event &&other) noexcept;
  Event &operator=(Event &&other) noexcept;
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  void Record(cudaStream_t stream);
  /// Make `stream` wait for the last `Record`; a no-op before any record.
  void WaitOn(cudaStream_t stream) const;
  /// Whether the last `Record` has passed (true before any record).
  bool Ready() const;
  /// Host wait for the last `Record`. The one place the host blocks on the
  /// pipeline, used only where a pinned buffer is about to be rewritten.
  void Synchronize() const;
  bool Recorded() const { return recorded_; }
  cudaEvent_t Raw() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
  bool recorded_ = false;
};

/**
 * @brief A device buffer OUTSIDE the memory pool, for what other streams
 * read and write.
 *
 * The pool is stream-ordered: a block freed on the legacy stream may still
 * be in use by a kernel queued there, and rmm makes that safe only for the
 * next allocation ON THE SAME STREAM. A buffer handed to the copy or the
 * encode stream out of the pool would therefore either race with the compute
 * queue or -- if allocated on the other stream -- make that stream wait for
 * the whole compute queue at the point of the free. So what the pipeline
 * streams touch is allocated once, here, with `cudaMalloc`, and reused for
 * the run. `Reserve` grows only; growth frees the old buffer, and `cudaFree`
 * synchronises the device, which is why callers reserve at setup.
 */
class DeviceArena {
 public:
  DeviceArena() = default;
  ~DeviceArena();
  DeviceArena(DeviceArena &&other) noexcept;
  DeviceArena &operator=(DeviceArena &&other) noexcept;
  DeviceArena(const DeviceArena &) = delete;
  DeviceArena &operator=(const DeviceArena &) = delete;

  /// At least `bytes` of device memory; the previous contents are lost when
  /// it grows.
  void *Reserve(size_t bytes);
  void *data() const { return data_; }
  size_t capacity() const { return bytes_; }
  void Release();

 private:
  void *data_ = nullptr;
  size_t bytes_ = 0;
};

/**
 * @brief The static, phase-aware scheduling hook.
 *
 * The library names its low-utilisation windows by calling `Notify` as it
 * enters them (EvalMod, the CC-MM chain); whoever prepares the next layer
 * installs a hook that issues a bounded amount of its work at each call. No
 * utilisation is queried at run time: the phases are deterministic, and the
 * policy -- how much per window -- is a static table the hook owns.
 */
class IdleWindow {
 public:
  using Hook = std::function<void(const char *phase)>;
  static void Set(Hook hook);
  static void Notify(const char *phase) {
    if (hook_) hook_(phase);
  }

 private:
  static inline Hook hook_{};
};

/// Copy `bytes` of host memory to `dst` on `stream` through the pinned
/// staging ring when it fits a slot (the form that does not synchronise the
/// stream), direct otherwise.
void StagedUpload(void *dst, const void *src, size_t bytes,
                  cudaStream_t stream);

}  // namespace cheddar
