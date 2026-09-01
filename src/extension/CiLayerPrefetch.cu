#include "extension/CiLayerPrefetch.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <utility>

#include "common/Assert.h"
#include "extension/Profile.h"

namespace cheddar {

namespace {
size_t Align256(size_t bytes) {
  return (bytes + 255) & ~static_cast<size_t>(255);
}
}  // namespace

template <typename word>
CiLayerPrefetcher<word>::CiLayerPrefetcher(
    const CoeffLinearLeg<word> &leg, std::vector<TensorSpec> specs,
    std::function<std::string(int)> layer_dir)
    : leg_{leg}, specs_{std::move(specs)}, layer_dir_{std::move(layer_dir)} {
  AssertTrue(!specs_.empty(), "CiLayerPrefetcher: no tensors");
  for (const auto &s : specs_) {
    const size_t bytes = Align256(s.Count() * sizeof(float));
    host_offsets_.push_back(layer_bytes_);
    layer_bytes_ += bytes;
    if (s.late) {
      device_offsets_.push_back(late_bytes_);
      late_bytes_ += bytes;
    } else {
      device_offsets_.push_back(shared_bytes_);
      shared_bytes_ += bytes;
    }
  }
  for (auto &slot : slots_) {
    slot.host = PinnedHostBuffer(layer_bytes_);
    AssertTrue(slot.host.pinned(),
               "CiLayerPrefetcher: could not pin " +
                   std::to_string(layer_bytes_ >> 20) +
                   " MiB for a layer's tensors");
    if (late_bytes_ > 0) slot.late_buffer.Reserve(late_bytes_);
  }
  if (shared_bytes_ > 0) tensor_buffer_.Reserve(shared_bytes_);
  std::cout << "CiLayerPrefetcher: " << specs_.size() << " tensors, "
            << (layer_bytes_ >> 20) << " MiB a layer: 2 pinned host buffers, "
            << "1 shared device tensor buffer (" << (shared_bytes_ >> 20)
            << " MiB) and 2 late buffers (" << (late_bytes_ >> 20)
            << " MiB each), all outside the pool" << std::endl;
}

template <typename word>
CiLayerPrefetcher<word>::~CiLayerPrefetcher() {
  for (auto &slot : slots_) {
    if (slot.pending) slot.read.wait();
    slot.uploaded.Synchronize();
    slot.late_done.Synchronize();
  }
  encode_issued_.Synchronize();
}

template <typename word>
bool CiLayerPrefetcher<word>::ReadInto(const std::string &path, float *dst,
                                       size_t count) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  const size_t got = std::fread(dst, sizeof(float), count, f);
  std::fclose(f);
  return got == count;
}

template <typename word>
void CiLayerPrefetcher<word>::RequestRead(int L) {
  Slot &slot = SlotOf(L);
  if (slot.pending) {
    slot.read.wait();
    slot.pending = false;
  }
  if (slot.tensors.layer >= 0) {
    // The copy stream read this buffer for the previous occupant (two
    // layers back); the wait is the one host wait of the pipeline and in
    // steady state it has long passed.
    NvtxScope _n("prep: wait_for_ready (pinned buffer reuse)");
    slot.uploaded.Synchronize();
  }
  slot.tensors = LayerTensors{};
  slot.tensors.layer = L;
  slot.tensors.host.resize(specs_.size());
  slot.tensors.device.resize(specs_.size(), nullptr);
  char *base = slot.host.data();
  for (size_t i = 0; i < specs_.size(); i++) {
    slot.tensors.host[i] =
        reinterpret_cast<const float *>(base + host_offsets_[i]);
  }
  const std::string dir = layer_dir_(L);
  std::vector<TensorSpec> specs = specs_;
  std::vector<size_t> offsets = host_offsets_;
  LayerTensors *tensors = &slot.tensors;
  // The worker: files into pinned memory, nothing else. No CUDA call, so no
  // context of its own.
  slot.read = std::async(std::launch::async, [dir, specs, offsets, base,
                                              tensors]() {
    NvtxScope _n("prep: CPU read");
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < specs.size(); i++) {
      float *dst = reinterpret_cast<float *>(base + offsets[i]);
      if (!ReadInto(dir + "/" + specs[i].file, dst, specs[i].Count())) {
        std::cerr << "CiLayerPrefetcher: cannot read " << dir << "/"
                  << specs[i].file << std::endl;
        return false;
      }
    }
    tensors->read_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    return true;
  });
  slot.pending = true;
}

template <typename word>
const typename CiLayerPrefetcher<word>::LayerTensors &
CiLayerPrefetcher<word>::Wait(int L) {
  Slot &slot = SlotOf(L);
  AssertTrue(slot.tensors.layer == L,
             "CiLayerPrefetcher::Wait: layer " + std::to_string(L) +
                 " was not requested");
  if (slot.pending) {
    NvtxScope _n("prep: wait_for_ready (cpu read)");
    const bool ok = slot.read.get();
    slot.pending = false;
    AssertTrue(ok, "CiLayerPrefetcher: the read of layer " +
                       std::to_string(L) + " failed");
    last_read_seconds_ = slot.tensors.read_seconds;
  }
  return slot.tensors;
}

template <typename word>
const Event &CiLayerPrefetcher<word>::Upload(int L) {
  NvtxScope _n("prep: H2D (layer tensors)");
  Slot &slot = SlotOf(L);
  (void)Wait(L);
  AssertTrue(!slot.tensors.uploaded,
             "CiLayerPrefetcher::Upload: layer already uploaded");
  cudaStream_t cpy = PipelineStreams::Get().Copy();
  // The shared buffer's previous occupant: every encode that reads it was
  // issued on the encode stream before `MarkEncodeIssued`, and this makes
  // the overwrite wait for them there. The late buffer's: the compute
  // stream's reads, marked by `MarkLateDone`.
  encode_issued_.WaitOn(cpy);
  slot.late_done.WaitOn(cpy);
  char *shared = static_cast<char *>(tensor_buffer_.data());
  char *late = static_cast<char *>(slot.late_buffer.data());
  for (size_t i = 0; i < specs_.size(); i++) {
    char *dst = (specs_[i].late ? late : shared) + device_offsets_[i];
    cudaMemcpyAsync(dst, slot.host.data() + host_offsets_[i],
                    specs_[i].Count() * sizeof(float), cudaMemcpyHostToDevice,
                    cpy);
    slot.tensors.device[i] = reinterpret_cast<const float *>(dst);
  }
  slot.uploaded.Record(cpy);
  slot.tensors.uploaded = true;
  slot.late_layer = L;
  device_layer_ = L;
  return slot.uploaded;
}

template <typename word>
const Event &CiLayerPrefetcher<word>::Uploaded(int L) const {
  const Slot &slot = SlotOf(L);
  AssertTrue(slot.tensors.layer == L && slot.tensors.uploaded,
             "CiLayerPrefetcher::Uploaded: layer " + std::to_string(L) +
                 " is not uploaded");
  return slot.uploaded;
}

template <typename word>
void CiLayerPrefetcher<word>::MarkEncodeIssued() {
  leg_.RecordEncodeStreamDone(encode_issued_);
}

template <typename word>
void CiLayerPrefetcher<word>::MarkLateDone(int L) {
  SlotOf(L).late_done.Record(PipelineStreams::Compute());
}

template <typename word>
typename CoeffLinearLeg<word>::DeviceWeights CiLayerPrefetcher<word>::Weights(
    int L, int which, const std::vector<int> *in_slot,
    const std::vector<int> *out_slot, double w_scale) const {
  const Slot &slot = SlotOf(L);
  AssertTrue(slot.tensors.layer == L && slot.tensors.uploaded,
             "CiLayerPrefetcher::Weights: layer " + std::to_string(L) +
                 " is not on the device");
  const TensorSpec &s = specs_.at(which);
  typename CoeffLinearLeg<word>::DeviceWeights dw;
  if (s.late) {
    AssertTrue(slot.late_layer == L,
               "CiLayerPrefetcher::Weights: the late buffer holds layer " +
                   std::to_string(slot.late_layer) + ", not " +
                   std::to_string(L));
    dw.ptr = slot.tensors.device[which];
  } else {
    // A shared tensor whose layer has left the buffer (the next layer's
    // upload replaced it) is handed over with NO device pointer: its
    // operands were converted while it was there and are answered from the
    // cache by name; anything that tried to read the tensor again would
    // assert on the null rather than read the next layer's weights.
    dw.ptr = (device_layer_ == L) ? slot.tensors.device[which] : nullptr;
  }
  dw.count = s.Count();
  dw.in_live = s.rows;
  dw.out_live = s.cols;
  dw.in_slot = in_slot;
  dw.out_slot = out_slot;
  dw.fingerprint = CoeffLinearLeg<word>::Fingerprint(
      slot.tensors.host[which], s.Count(), w_scale);
  return dw;
}

template <typename word>
const float *CiLayerPrefetcher<word>::Host(int L, int which) const {
  const Slot &slot = SlotOf(L);
  AssertTrue(slot.tensors.layer == L && !slot.pending,
             "CiLayerPrefetcher::Host: layer " + std::to_string(L) +
                 " is not read");
  return slot.tensors.host.at(which);
}

template class CiLayerPrefetcher<uint32_t>;
template class CiLayerPrefetcher<uint64_t>;

}  // namespace cheddar
