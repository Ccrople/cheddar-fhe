#include "core/DeviceVector.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/Assert.h"

namespace cheddar {

namespace {

// THE PINNED STAGING RING. `cudaMemcpyAsync` out of PAGEABLE host memory is
// not asynchronous: the runtime synchronises the stream before it starts the
// copy (CUDA Runtime API, "API synchronization behavior"), so every small
// upload -- a hoisted kernel's pointer tables, an encoded constant's limbs, a
// twiddle table -- drained the whole queue standing in front of it, and a
// layer makes tens of thousands of them. Copying the bytes into a pinned slot
// first turns the upload into a real DMA that the host walks away from. The
// one dependency is the slot's reuse: an event recorded behind each transfer
// is waited on before the slot is written again. A ring of sixteen made the
// host wait 6.7 s a layer for a card it was otherwise far ahead of (measured
// 2026-09-01, `reference/nsys/2026-09-01_h2d_fused_layer0`), so the ring
// GROWS while its oldest slot is still in flight, up to `kMaxSlots`, and
// only waits at the cap. Uploads larger than a slot (a whole plaintext, a
// weight tile) take the direct path as before; they are per-layer
// preparation, not per-operation traffic. The ring is never freed: it must
// outlive every context, and the driver is gone before a static destructor
// would run.
class PinnedStaging {
 public:
  static constexpr size_t kSlotBytes = static_cast<size_t>(1) << 20;
  static constexpr int kInitialSlots = 64;
  static constexpr int kMaxSlots = 512;

  static PinnedStaging &Get() {
    static PinnedStaging *ring = new PinnedStaging();
    return *ring;
  }

  // False when the copy could not be staged (too large, or no pinned memory
  // could be had); the caller then copies directly.
  bool Upload(void *dst, const void *src, size_t bytes, cudaStream_t stream) {
    if (!ok_ || bytes > kSlotBytes) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (slots_[next_].recorded &&
        cudaEventQuery(slots_[next_].event) == cudaErrorNotReady) {
      if (static_cast<int>(slots_.size()) < kMaxSlots && Grow()) {
        // The new slot went in at `next_`; the pending one moved up one.
      } else {
        cudaEventSynchronize(slots_[next_].event);
      }
    }
    cudaGetLastError();
    Slot &slot = slots_[next_];
    next_ = (next_ + 1) % static_cast<int>(slots_.size());
    std::memcpy(slot.host, src, bytes);
    cudaMemcpyAsync(dst, slot.host, bytes, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(slot.event, stream);
    slot.recorded = true;
    return true;
  }

 private:
  struct Slot {
    void *host = nullptr;
    cudaEvent_t event = nullptr;
    bool recorded = false;
  };

  static bool Make(Slot &slot) {
    if (cudaMallocHost(&slot.host, kSlotBytes) != cudaSuccess ||
        cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming) !=
            cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }

  // A fresh slot inserted at `next_`, so the ring's order -- oldest at
  // `next_`, newest just behind it -- is kept.
  bool Grow() {
    Slot fresh;
    if (!Make(fresh)) return false;
    slots_.insert(slots_.begin() + next_, fresh);
    return true;
  }

  PinnedStaging() {
    slots_.reserve(kMaxSlots);
    for (int i = 0; i < kInitialSlots; i++) {
      Slot slot;
      if (!Make(slot)) {
        ok_ = false;
        return;
      }
      slots_.push_back(slot);
    }
    ok_ = true;
  }

  std::vector<Slot> slots_;
  int next_ = 0;
  bool ok_ = false;
  std::mutex mutex_;
};

}  // namespace

template <typename word>
DvView<word>::DvView(word *data, int size, int aux_size /*= 0*/)
    : data_(data), size_(size), aux_size_(aux_size) {}

template <typename word>
word *DvView<word>::data() {
  return data_;
}

template <typename word>
const word *DvView<word>::data() const {
  return data_;
}

template <typename word>
int DvView<word>::TotalSize() const {
  return size_;
}

template <typename word>
int DvView<word>::AuxSize() const {
  return aux_size_;
}

template <typename word>
int DvView<word>::QSize() const {
  return size_ - aux_size_;
}

template <typename word>
DvConstView<word>::DvConstView(const word *data, int size, int aux_size /*= 0*/)
    : data_(data), size_(size), aux_size_(aux_size) {}

template <typename word>
const word *DvConstView<word>::data() const {
  return data_;
}

template <typename word>
int DvConstView<word>::TotalSize() const {
  return size_;
}

template <typename word>
int DvConstView<word>::AuxSize() const {
  return aux_size_;
}

template <typename word>
int DvConstView<word>::QSize() const {
  return size_ - aux_size_;
}

template <typename word>
DvConstView<word>::DvConstView(const DvView<word> &view)
    : data_(view.data()), size_(view.TotalSize()), aux_size_(view.AuxSize()) {}

template <typename word>
DeviceVector<word>::DeviceVector(int size /*= 0*/,
                                 cudaStream_t stream /*= cudaStreamLegacy*/)
    : Base(size, stream) {}

template <typename word>
void DeviceVector<word>::resize(int size) {
  // rmm's resize carries the old contents into the new buffer with a
  // `cudaMemcpyAsync` even when there are none: every fresh ciphertext,
  // plaintext part and scratch vector -- 389k of the 433k allocations in a
  // layer's run -- paid a runtime call that copied nothing (nsys: 389k
  // `cudaMemcpyAsync` against 71k copies the card ever saw). An EMPTY vector
  // is given its buffer directly instead.
  if (this->size() == 0 && static_cast<size_t>(size) > this->capacity()) {
    Base fresh(static_cast<size_t>(size), this->stream());
    static_cast<Base &>(*this) = std::move(fresh);
    return;
  }
  this->resize(size, this->stream());
}

template <typename word>
void DeviceVector<word>::ZeroExtend(int size) {
  AssertTrue(size >= 0,
             "DeviceVector::ZeroExtend: size must be positive, but got " +
                 std::to_string(size));
  if (size == 0) return;
  auto old_size = this->size();
  resize(old_size + size);
  cudaMemsetAsync(this->data() + old_size, 0, size * sizeof(word),
                  this->stream());
}

template <typename word>
DvView<word> DeviceVector<word>::View(int aux_size /*= 0*/,
                                      int front_offset /*= 0*/) {
  return DvView<word>(this->data() + front_offset, this->size() - front_offset,
                      aux_size);
}

template <typename word>
DvConstView<word> DeviceVector<word>::ConstView(
    int aux_size /*= 0*/, int front_offset /*= 0*/) const {
  return DvConstView<word>(this->data() + front_offset,
                           this->size() - front_offset, aux_size);
}

template <typename word>
void CopyHostToDevice(DeviceVector<word> &dst, const HostVector<word> &src) {
  dst.resize(src.size());
  const size_t bytes = src.size() * sizeof(word);
  if (bytes == 0) return;
  // Through the pinned ring when it fits (see `PinnedStaging`): the source
  // is consumed before this returns either way, so the caller's contract --
  // free `src` whenever -- is the one it always had.
  if (PinnedStaging::Get().Upload(dst.data(), src.data(), bytes,
                                  dst.stream())) {
    return;
  }
  cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyHostToDevice,
                  dst.stream());
}

template <typename word>
void CopyDeviceToHost(HostVector<word> &dst, const DeviceVector<word> &src) {
  dst.resize(src.size());
  cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(word),
                  cudaMemcpyDeviceToHost, src.stream());
}

template <typename word>
void CopyDeviceToDevice(DeviceVector<word> &dst,
                        const DeviceVector<word> &src) {
  dst.resize(src.size());
  CheckTrue(dst.stream() == src.stream(),
            "CopyDeviceToDevice: Copying between different streams...");
  if (src.data() == dst.data()) return;
  cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(word),
                  cudaMemcpyDeviceToDevice, dst.stream());
}

PinnedHostBuffer::PinnedHostBuffer(size_t bytes) : bytes_(bytes) {
  if (bytes == 0) return;
  void *p = nullptr;
  if (cudaMallocHost(&p, bytes) == cudaSuccess) {
    data_ = static_cast<char *>(p);
    pinned_ = true;
    return;
  }
  // Not fatal: the copies still work, only as pageable ones.
  cudaGetLastError();
  data_ = static_cast<char *>(std::malloc(bytes));
  AssertTrue(data_ != nullptr, "PinnedHostBuffer: out of host memory");
}

PinnedHostBuffer::~PinnedHostBuffer() { Release(); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer &&other) noexcept
    : data_(other.data_), bytes_(other.bytes_), pinned_(other.pinned_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
  other.pinned_ = false;
}

PinnedHostBuffer &PinnedHostBuffer::operator=(
    PinnedHostBuffer &&other) noexcept {
  if (this != &other) {
    Release();
    data_ = other.data_;
    bytes_ = other.bytes_;
    pinned_ = other.pinned_;
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.pinned_ = false;
  }
  return *this;
}

void PinnedHostBuffer::Release() {
  if (data_ == nullptr) return;
  if (pinned_) {
    // The one wait this buffer ever makes: a DMA out of it may still be in
    // flight when its owner dies, and page-locked memory cannot be given
    // back under a transfer.
    cudaStreamSynchronize(cudaStreamLegacy);
    cudaFreeHost(data_);
  } else {
    std::free(data_);
  }
  data_ = nullptr;
  bytes_ = 0;
  pinned_ = false;
}

// Explicit instantiation of the template classes
template class DvView<int32_t>;
template class DvView<int64_t>;
template class DvView<uint32_t>;
template class DvView<uint64_t>;
template class DvView<uint32_t *>;
template class DvView<uint64_t *>;
template class DvView<const uint32_t *>;
template class DvView<const uint64_t *>;
template class DvConstView<int32_t>;
template class DvConstView<int64_t>;
template class DvConstView<uint32_t>;
template class DvConstView<uint64_t>;
template class DvConstView<uint32_t *>;
template class DvConstView<uint64_t *>;
template class DvConstView<const uint32_t *>;
template class DvConstView<const uint64_t *>;
// int8_t carries the split plaintext matrices for the cuBLAS PCMM path, which
// feeds int8 tensor cores; the other widths predate it.
template class DeviceVector<int8_t>;
template class DeviceVector<int32_t>;
template class DeviceVector<int64_t>;
template class DeviceVector<uint32_t>;
template class DeviceVector<uint64_t>;
template class DeviceVector<uint32_t *>;
template class DeviceVector<uint64_t *>;
template class DeviceVector<const uint32_t *>;
template class DeviceVector<const uint64_t *>;
// float carries a model tensor as the exporter wrote it, so the projection
// leg's gathered encode reads the f32 blob at its own size instead of a
// declared-width double matrix built on the host (LlamaLinear.h,
// `DeviceWeights`).
template class DeviceVector<float>;

// Explicit instantiation of the template functions
template void CopyHostToDevice(DeviceVector<int8_t> &dst,
                               const HostVector<int8_t> &src);
template void CopyHostToDevice(DeviceVector<float> &dst,
                               const HostVector<float> &src);
template void CopyHostToDevice(DeviceVector<int32_t> &dst,
                               const HostVector<int32_t> &src);
template void CopyHostToDevice(DeviceVector<int64_t> &dst,
                               const HostVector<int64_t> &src);
template void CopyHostToDevice(DeviceVector<uint32_t> &dst,
                               const HostVector<uint32_t> &src);
template void CopyHostToDevice(DeviceVector<uint64_t> &dst,
                               const HostVector<uint64_t> &src);
template void CopyHostToDevice(DeviceVector<uint32_t *> &dst,
                               const HostVector<uint32_t *> &src);
template void CopyHostToDevice(DeviceVector<uint64_t *> &dst,
                               const HostVector<uint64_t *> &src);
template void CopyHostToDevice(DeviceVector<const uint32_t *> &dst,
                               const HostVector<const uint32_t *> &src);
template void CopyHostToDevice(DeviceVector<const uint64_t *> &dst,
                               const HostVector<const uint64_t *> &src);
// int8_t only had the host-to-device direction: the split plaintext matrices
// were built on the host and never read back. Weight streaming reads them back
// -- that is what a host mirror is -- so the return leg is instantiated too.
template void CopyDeviceToHost(HostVector<int8_t> &dst,
                               const DeviceVector<int8_t> &src);
template void CopyDeviceToHost(HostVector<int32_t> &dst,
                               const DeviceVector<int32_t> &src);
template void CopyDeviceToHost(HostVector<int64_t> &dst,
                               const DeviceVector<int64_t> &src);
template void CopyDeviceToHost(HostVector<uint32_t> &dst,
                               const DeviceVector<uint32_t> &src);
template void CopyDeviceToHost(HostVector<uint64_t> &dst,
                               const DeviceVector<uint64_t> &src);
template void CopyDeviceToHost(HostVector<uint32_t *> &dst,
                               const DeviceVector<uint32_t *> &src);
template void CopyDeviceToHost(HostVector<uint64_t *> &dst,
                               const DeviceVector<uint64_t *> &src);
template void CopyDeviceToHost(HostVector<const uint32_t *> &dst,
                               const DeviceVector<const uint32_t *> &src);
template void CopyDeviceToHost(HostVector<const uint64_t *> &dst,
                               const DeviceVector<const uint64_t *> &src);
template void CopyDeviceToDevice(DeviceVector<int32_t> &dst,
                                 const DeviceVector<int32_t> &src);
template void CopyDeviceToDevice(DeviceVector<int64_t> &dst,
                                 const DeviceVector<int64_t> &src);
template void CopyDeviceToDevice(DeviceVector<uint32_t> &dst,
                                 const DeviceVector<uint32_t> &src);
template void CopyDeviceToDevice(DeviceVector<uint64_t> &dst,
                                 const DeviceVector<uint64_t> &src);
template void CopyDeviceToDevice(DeviceVector<uint32_t *> &dst,
                                 const DeviceVector<uint32_t *> &src);
template void CopyDeviceToDevice(DeviceVector<uint64_t *> &dst,
                                 const DeviceVector<uint64_t *> &src);
template void CopyDeviceToDevice(DeviceVector<const uint32_t *> &dst,
                                 const DeviceVector<const uint32_t *> &src);
template void CopyDeviceToDevice(DeviceVector<const uint64_t *> &dst,
                                 const DeviceVector<const uint64_t *> &src);

}  // namespace cheddar