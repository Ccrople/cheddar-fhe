#pragma once

#include <chrono>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

// NVTX, if the toolkit has it. CUDA 11 and newer put the header-only v3
// implementation under `nvtx3/`; older layouts keep a top-level header that
// wants -lnvToolsExt, which is not worth linking for. If neither is present
// the ranges compile away and only the host timers remain.
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define CHEDDAR_HAS_NVTX 1
#endif
#endif

namespace cheddar {

/**
 * @brief Where the layer's time goes, at two resolutions at once.
 *
 * ## Why one object does both
 *
 * The two questions a profile has to answer are "which step of the layer is
 * slow" and "which kernel inside that step is slow", and they want different
 * tools: a host timer answers the first, and Nsight Systems answers the second
 * but only if the steps are marked. Marking a step twice -- once for the timer
 * and once for NVTX -- is how the two drift apart, so one scope pushes both.
 *
 * ## Why the timer synchronises and the marker does not
 *
 * Cheddar issues everything on the legacy default stream, so the device work
 * of a step is already serialised against its neighbours; what is not
 * serialised is the *host* return from a launch. Timing without a sync would
 * charge a step for the launches it issued rather than the work it did, and
 * the error is not small -- a step that launches a hundred cheap kernels would
 * look slower than one that launches ten expensive ones.
 *
 * So when timing is on, the scope synchronises at both ends. That is a real
 * perturbation and it is why the report says so out loud. It costs little
 * here because the stream is serial anyway, but a step's number is
 * "wall time including its own drain", not "kernel time".
 *
 * NVTX ranges carry no sync: nsys correlates kernels to ranges by timestamp
 * and does that correctly on an unsynchronised stream. So a profiled run
 * should be made with timing OFF, and the kernel attribution comes out of nsys
 * rather than out of this class.
 *
 * ## Enabling
 *
 * `CHEDDAR_PROFILE` turns the apparatus on and `CHEDDAR_PROFILE_NOSYNC` drops
 * the timers from it; see `Timing()` for why the second run is not optional.
 * NVTX ranges are always pushed -- with no profiler attached that is a
 * predicted branch and a function call, against steps that are milliseconds
 * long.
 */
class Profile {
 public:
  // `CHEDDAR_PROFILE` turns the whole apparatus on: the warm-up pass, the
  // capture range and the host timers.
  static bool Enabled() {
    static const bool on = std::getenv("CHEDDAR_PROFILE") != nullptr;
    return on;
  }

  // `CHEDDAR_PROFILE_NOSYNC` keeps the apparatus and drops the per-step
  // timers, and it is not a refinement -- it is the only way to get a true
  // wall time. Synchronising at every step boundary costs nothing in device
  // work but destroys host/device overlap: normally the host runs ahead
  // queueing launches while the GPU works, and the layer has real host work
  // between launches (weight encoding, the causal mask, the score shift). A
  // synchronised total is therefore an upper bound on the layer's wall time,
  // and the gap between the two runs is exactly the overlap that was lost.
  //
  // An nsys run wants this set too: kernel durations do not care, but a
  // timeline full of artificial drains is a worse thing to read.
  static bool Timing() {
    static const bool on =
        Enabled() && std::getenv("CHEDDAR_PROFILE_NOSYNC") == nullptr;
    return on;
  }

  // Accumulate `seconds` against `label`, keeping first-seen order so the
  // report reads in the order the layer runs rather than alphabetically.
  static void Add(const std::string &label, double seconds);

  // Print the table and forget it, so a warm-up pass and a measured pass do
  // not add together.
  static void Report(const char *title);
  static void Reset();
};

class ProfileScope {
 public:
  explicit ProfileScope(const char *label) : label_(label) {
#ifdef CHEDDAR_HAS_NVTX
    nvtxRangePushA(label);
#endif
    if (Profile::Timing()) {
      cudaDeviceSynchronize();
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~ProfileScope() {
    if (Profile::Timing()) {
      cudaDeviceSynchronize();
      const std::chrono::duration<double> d =
          std::chrono::steady_clock::now() - start_;
      Profile::Add(label_, d.count());
    }
#ifdef CHEDDAR_HAS_NVTX
    nvtxRangePop();
#endif
  }

  ProfileScope(const ProfileScope &) = delete;
  ProfileScope &operator=(const ProfileScope &) = delete;

 private:
  const char *label_;
  std::chrono::steady_clock::time_point start_;
};

/**
 * @brief An NVTX range with no host timer, for marks inside a timed step.
 *
 * `ProfileScope` is used only on the layer's leaf steps, so its rows sum to
 * the layer. Anything nested inside one -- the phases of a projection, the
 * legs of the attention product -- gets this instead: nsys still attributes
 * kernels to it, and the host table stays a partition rather than a tree that
 * double-counts.
 */
class NvtxScope {
 public:
  explicit NvtxScope(const char *label) {
#ifdef CHEDDAR_HAS_NVTX
    nvtxRangePushA(label);
#else
    (void)label;
#endif
  }
  ~NvtxScope() {
#ifdef CHEDDAR_HAS_NVTX
    nvtxRangePop();
#endif
  }
  NvtxScope(const NvtxScope &) = delete;
  NvtxScope &operator=(const NvtxScope &) = delete;
};

}  // namespace cheddar
