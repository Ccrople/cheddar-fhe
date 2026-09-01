#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <string>
#include <vector>

#include "core/DeviceVector.h"
#include "core/Streams.h"
#include "extension/LlamaLinear.h"

namespace cheddar {

/**
 * @brief The next layer's weights, read and moved while this layer runs.
 *
 * Three stages, three actors, no host wait in steady state (Doing.md 3.21):
 *
 *   CPU worker    read L+1's f32 files into a pinned host buffer
 *   copy stream   the buffer to the device tensor buffer (`Upload`)
 *   encode stream `CoeffLinearLeg`'s prefetch jobs read the tensor buffer
 *                 and write each operand to its pinned mirror
 *
 * The host buffers are two, so the worker can read L+2 while L+1's upload
 * may still be in flight; the device tensor buffer is ONE, because a layer's
 * tensors are consumed by its encode jobs before its own compute starts, so
 * L+1's upload only has to wait for L's jobs to have been issued -- the
 * event `CoeffLinearLeg::RecordEncodeStreamDone` records after the drain.
 * A tensor marked `late` is the exception: it is read by the layer's OWN
 * compute (the O projection, whose scale is fitted in-run and which is
 * therefore converted at its use), so it gets a buffer per layer parity
 * that stands until `MarkLateDone`. All of these buffers are outside the
 * memory pool (`DeviceArena`, `PinnedHostBuffer`): they are read and written
 * by streams other than the compute stream.
 *
 * The worker thread touches no CUDA: it reads files into memory the main
 * thread pinned once, and hands back a future. Readiness is a
 * `std::future` (a blocking wait, not a poll), and the only thing that waits
 * on it is `Wait`, which in steady state finds it already set.
 */
template <typename word>
class CiLayerPrefetcher {
 public:
  struct TensorSpec {
    std::string file;  //!< relative to the layer directory
    int rows = 0;
    int cols = 0;
    //! Read by the layer's own compute rather than by the prefetch jobs;
    //! kept in a per-parity buffer until `MarkLateDone`.
    bool late = false;
    size_t Count() const { return static_cast<size_t>(rows) * cols; }
  };

  struct LayerTensors {
    int layer = -1;
    std::vector<const float *> host;    //!< pinned, one per spec
    std::vector<const float *> device;  //!< on the device, one per spec
    double read_seconds = 0.0;
    bool uploaded = false;
  };

  /**
   * @param leg the projection leg whose prefetch jobs will read the tensors
   * @param specs the layer's tensors, in the order the caller will index them
   * @param layer_dir layer index -> directory holding the files
   */
  CiLayerPrefetcher(const CoeffLinearLeg<word> &leg,
                    std::vector<TensorSpec> specs,
                    std::function<std::string(int)> layer_dir);
  ~CiLayerPrefetcher();

  CiLayerPrefetcher(const CiLayerPrefetcher &) = delete;
  CiLayerPrefetcher &operator=(const CiLayerPrefetcher &) = delete;

  /// Start reading layer `L` on the worker thread into the pinned buffer of
  /// its parity, once that buffer's previous upload has passed.
  void RequestRead(int L);

  /// The layer's tensors in pinned memory; blocks only if the read is not
  /// done yet (`prep: wait_for_ready (cpu read)`).
  const LayerTensors &Wait(int L);

  /// Issue the H2D of layer `L`'s tensors on the copy stream: the shared
  /// ones after the encode stream has finished with the buffer's previous
  /// occupant, the late ones after the compute stream has finished with
  /// theirs. Returns the event the leg's jobs wait on.
  const Event &Upload(int L);

  /// The event `Upload(L)` recorded.
  const Event &Uploaded(int L) const;

  /// The shared tensor buffer's release: call after the leg's jobs for the
  /// layer that occupies it have all been ISSUED (`DrainPrefetch`).
  void MarkEncodeIssued();

  /// The late buffer's release: call on the compute stream's behalf once the
  /// layer's own compute has issued every read of its late tensors.
  void MarkLateDone(int L);

  /// A `DeviceWeights` over tensor `which` of layer `L` (uploaded), with the
  /// fingerprint of its host bytes.
  typename CoeffLinearLeg<word>::DeviceWeights Weights(
      int L, int which, const std::vector<int> *in_slot,
      const std::vector<int> *out_slot, double w_scale) const;

  const float *Host(int L, int which) const;

  size_t PinnedBytes() const { return 2 * layer_bytes_; }
  size_t DeviceBytes() const { return shared_bytes_ + 2 * late_bytes_; }
  double LastReadSeconds() const { return last_read_seconds_; }

 private:
  struct Slot {
    PinnedHostBuffer host;
    LayerTensors tensors;
    std::future<bool> read;
    bool pending = false;
    Event uploaded;   // recorded on the copy stream after the H2D
    Event late_done;  // recorded on the compute stream by MarkLateDone
    DeviceArena late_buffer;
    int late_layer = -1;
  };

  const CoeffLinearLeg<word> &leg_;
  std::vector<TensorSpec> specs_;
  std::function<std::string(int)> layer_dir_;
  std::vector<size_t> host_offsets_;    // byte offset of each tensor, pinned
  std::vector<size_t> device_offsets_;  // in the shared or the late buffer
  size_t layer_bytes_ = 0;
  size_t shared_bytes_ = 0;
  size_t late_bytes_ = 0;
  Slot slots_[2];
  DeviceArena tensor_buffer_;
  int device_layer_ = -1;  // whose shared tensors the buffer holds
  Event encode_issued_;    // the leg's encode stream, past the buffer's use
  double last_read_seconds_ = 0.0;

  Slot &SlotOf(int L) { return slots_[L & 1]; }
  const Slot &SlotOf(int L) const { return slots_[L & 1]; }
  static bool ReadInto(const std::string &path, float *dst, size_t count);
};

}  // namespace cheddar
