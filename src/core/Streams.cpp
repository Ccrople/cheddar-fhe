#include "core/Streams.h"

#include <string>
#include <utility>

#include "common/Assert.h"

namespace cheddar {

PipelineStreams &PipelineStreams::Get() {
  // Never destroyed: the streams must outlive every context, and the driver
  // is gone before a static destructor would run.
  static PipelineStreams *streams = new PipelineStreams();
  return *streams;
}

PipelineStreams::PipelineStreams() {
  int least = 0, greatest = 0;
  cudaDeviceGetStreamPriorityRange(&least, &greatest);
  // `least` is the default priority (0), which the legacy stream has and
  // cannot be moved from. Both pipeline streams are created AT that
  // priority: created lower than the legacy stream is not possible, and
  // created higher would let the next layer's encode pre-empt this layer's
  // compute, which is the opposite of the intent.
  copy_priority_ = least;
  encode_priority_ = least;
  AssertTrue(cudaStreamCreateWithPriority(&copy_, cudaStreamNonBlocking,
                                          copy_priority_) == cudaSuccess,
             "PipelineStreams: could not create the copy stream");
  AssertTrue(cudaStreamCreateWithPriority(&encode_, cudaStreamNonBlocking,
                                          encode_priority_) == cudaSuccess,
             "PipelineStreams: could not create the encode stream");
  (void)greatest;
}

Event::Event() {
  AssertTrue(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) ==
                 cudaSuccess,
             "Event: cudaEventCreate failed");
}

Event::~Event() {
  if (event_ != nullptr) cudaEventDestroy(event_);
}

Event::Event(Event &&other) noexcept
    : event_(other.event_), recorded_(other.recorded_) {
  other.event_ = nullptr;
  other.recorded_ = false;
}

Event &Event::operator=(Event &&other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) cudaEventDestroy(event_);
    event_ = other.event_;
    recorded_ = other.recorded_;
    other.event_ = nullptr;
    other.recorded_ = false;
  }
  return *this;
}

void Event::Record(cudaStream_t stream) {
  cudaEventRecord(event_, stream);
  recorded_ = true;
}

void Event::WaitOn(cudaStream_t stream) const {
  if (!recorded_) return;
  cudaStreamWaitEvent(stream, event_, 0);
}

bool Event::Ready() const {
  if (!recorded_) return true;
  const cudaError_t e = cudaEventQuery(event_);
  if (e == cudaErrorNotReady) return false;
  cudaGetLastError();
  return true;
}

void Event::Synchronize() const {
  if (!recorded_) return;
  cudaEventSynchronize(event_);
}

DeviceArena::~DeviceArena() { Release(); }

DeviceArena::DeviceArena(DeviceArena &&other) noexcept
    : data_(other.data_), bytes_(other.bytes_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
}

DeviceArena &DeviceArena::operator=(DeviceArena &&other) noexcept {
  if (this != &other) {
    Release();
    data_ = other.data_;
    bytes_ = other.bytes_;
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

void *DeviceArena::Reserve(size_t bytes) {
  if (bytes <= bytes_) return data_;
  Release();
  void *p = nullptr;
  AssertTrue(cudaMalloc(&p, bytes) == cudaSuccess,
             "DeviceArena: cudaMalloc of " + std::to_string(bytes >> 20) +
                 " MiB failed");
  data_ = p;
  bytes_ = bytes;
  return data_;
}

void DeviceArena::Release() {
  if (data_ == nullptr) return;
  // `cudaFree` waits for the device; the arena is sized at setup so this
  // runs at most once per shape and never inside a layer.
  cudaFree(data_);
  data_ = nullptr;
  bytes_ = 0;
}

void IdleWindow::Set(Hook hook) { hook_ = std::move(hook); }

}  // namespace cheddar
