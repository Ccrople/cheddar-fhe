#pragma once

#include <cstdint>
#include <memory>
#include <set>

#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief The device allocator every Cheddar buffer goes through.
 *
 * ## Why this is shared between Contexts rather than owned by one
 *
 * RMM's `device_uvector` captures `get_current_device_resource()` **at
 * allocation time** and uses that pointer again to deallocate. When each
 * Context owned a pool and installed it as the current resource, two Contexts
 * produced a use-after-free:
 *
 *   1. Context A is built; the current resource becomes A's pool.
 *   2. Context B is built; the current resource becomes B's pool.
 *   3. A allocates something *now* -- preparing a key, say -- and that buffer
 *      records B's pool as its deallocator.
 *   4. B is destroyed, taking its pool with it.
 *   5. A is destroyed and frees that buffer through a pool that is gone.
 *
 * Nothing in steps 1-3 looks wrong at the call site, and the crash lands in
 * step 5 with no connection to the code that caused it. Restoring the previous
 * resource in the destructor instead of clearing it does not help either: the
 * buffer from step 3 still holds B's pointer.
 *
 * So there is one pool per process, reference-counted, and every Context adds
 * whatever bins its parameters call for. Bins are only size buckets, so their
 * union serves every ring; sizes already present are skipped rather than
 * duplicated. The pool is torn down when the last Context goes, which for a
 * single-Context program is exactly the old behaviour.
 *
 * ## Accounting, and why `cudaMemGetInfo` cannot do it
 *
 * The upstream here is `cuda_async_memory_resource`, so what the driver
 * reports as used is a POOL RESERVATION and a high-water mark: a stage that
 * allocates ten gigabytes and frees them again leaves the reservation raised
 * for every stage after it, and a stage that fits inside what is already
 * reserved is invisible. Section 1.5da spent three runs of a quarter-hour test
 * guessing which stage owned which gigabyte for exactly this reason, and a
 * `CHEDDAR_CI_TILE` sweep that moves an operand from 10.7 GB to 2.7 GB
 * measured *identical* driver numbers at every setting.
 *
 * `SetStatisticsEnabled` wraps the pool in RMM's `statistics_resource_adaptor`
 * before any Context exists, which counts the bytes Cheddar actually asks for.
 * `GetUsage().current_bytes` at a stage boundary is the live demand, which is
 * the number the driver cannot give; `peak_bytes` is the true high-water mark
 * of demand rather than of reservation.
 *
 * It is OFF by default and has to be, twice over. The adaptor takes a lock on
 * every allocation and free, which is small but not free at Cheddar's
 * allocation counts; and it must be installed before the first `MemoryPool`
 * exists, because the current device resource is captured per buffer at
 * allocation time -- the same capture the class comment above is about. So it
 * is chosen once per process, by `CHEDDAR_MEM_STATS=1` or by calling
 * `SetStatisticsEnabled(true)` before the first Context.
 *
 * ## The two RMM defaults that made `cudaMemGetInfo` useless
 *
 * `cuda_async_memory_resource`'s constructor does two things that together
 * turn the driver's numbers into noise, and neither is Cheddar's choice:
 *
 *   - `initial_pool_size` defaults to **half of free memory**, allocated and
 *     immediately deallocated to prime the pool. On an 80 GB card that is
 *     40813 MiB reserved before the first ciphertext exists, so nothing under
 *     40 GiB is visible at all -- which is why 1.5da's `CHEDDAR_CI_TILE` sweep
 *     read *identical* driver numbers at every setting while the operand it
 *     was moving went from 10.7 GB to 2.7 GB.
 *   - `release_threshold` defaults to **total** memory, so the pool never
 *     returns anything to the driver and the reserve is a high-water mark: a
 *     stage that spikes and frees raises the ceiling for every stage after it.
 *
 * `SetPoolSizeMiB` and `SetReleaseThresholdMiB` override them, and
 * `CHEDDAR_POOL_MIB` / `CHEDDAR_POOL_RELEASE_MIB` do the same from the
 * environment -- **the environment wins**, so an existing binary can be
 * measured without an edit.
 *
 * ## And a third thing, which is Cheddar's own
 *
 * Zeroing both of those is NOT enough, and the layer measured it: with the
 * pool primed at nothing and released at every synchronization point, the CI
 * layer still finished on a 65748 MiB reservation against 20616 MiB of live
 * demand. What holds the difference is the `binning_memory_resource` above --
 * its buckets are `fixed_size_memory_resource`s, which carve chunks off the
 * upstream and **never give one back**. That is the point of binning, and it
 * is why an allocation that finds a free block in its bucket costs nothing;
 * it also means no configuration of the pool underneath can make the driver's
 * number a level.
 *
 * `SetMaxBinMiB` / `CHEDDAR_POOL_MAX_BIN_MIB` caps the largest bucket, and
 * zero disables binning outright. **All three at zero is the measuring
 * configuration**: on `MemoryLedger` it takes the driver's peak from 32813 MiB
 * to 24877 against a live peak of 24433.8 -- 34% over demand down to 1.8% --
 * with the run time unchanged at 237 s. None of the three is the RUNNING
 * configuration, since priming, holding and binning are what make the pool
 * fast, so all three are off by default and chosen once per process, before
 * the first Context.
 */
class MemoryPool {
  using DefaultUpstream = rmm::mr::cuda_async_memory_resource;
  using MemoryPoolBase = rmm::mr::binning_memory_resource<DefaultUpstream>;

 public:
  /**
   * @brief Bytes and allocation counts, as RMM's adaptor reports them.
   *
   * `current` is live demand, `peak` is the largest `current` ever reached,
   * and `total` is the sum of every allocation ever made -- churn, not
   * residency. All three are zero when statistics are off.
   */
  struct Usage {
    int64_t current_bytes = 0;
    int64_t peak_bytes = 0;
    int64_t total_bytes = 0;
    int64_t current_allocations = 0;
    int64_t peak_allocations = 0;
  };

  /**
   * @brief Turn byte accounting on or off. Must be called before the first
   *        Context is constructed; after that it has no effect and returns
   *        false, because buffers already hold the resource they were
   *        allocated from.
   *
   * `CHEDDAR_MEM_STATS=1` in the environment does the same thing, and is read
   * when the first pool is built so that a run can be measured without an
   * edit.
   *
   * @return whether the request was honoured
   */
  static bool SetStatisticsEnabled(bool enabled);

  //! Whether byte accounting is active for this process.
  static bool StatisticsEnabled();

  //! Live and peak demand. All zero unless statistics are enabled.
  static Usage GetUsage();

  /**
   * @brief Override RMM's initial pool size, in MiB. Zero primes nothing.
   *
   * Must be called before the first Context, and like
   * `SetStatisticsEnabled` returns false afterwards. Unset leaves RMM's own
   * default, which is half of free memory.
   *
   * @return whether the request was honoured
   */
  static bool SetPoolSizeMiB(int64_t mib);

  /**
   * @brief Override RMM's pool release threshold, in MiB. Zero returns memory
   *        to the driver at every synchronization point.
   *
   * Must be called before the first Context. Unset leaves RMM's own default,
   * which is the card's total memory -- i.e. never release.
   *
   * @return whether the request was honoured
   */
  static bool SetReleaseThresholdMiB(int64_t mib);

  /**
   * @brief Cap the largest size bucket, in MiB. Zero adds no bins at all, so
   *        every allocation goes straight to the async pool.
   *
   * The bins are `fixed_size_memory_resource`s: each one carves chunks off the
   * upstream and **never gives them back**, which is why a run's reservation
   * can sit far above its live demand however the pool below is configured.
   * That is the point of binning -- an allocation that finds a free block in
   * its bucket costs nothing -- but it means the two knobs above cannot make
   * the driver's number honest on their own.
   *
   * Must be called before the first Context. Negative leaves the heuristic
   * bin set the constructor builds.
   *
   * @return whether the request was honoured
   */
  static bool SetMaxBinMiB(int64_t mib);

  template <typename word>
  explicit MemoryPool(const Parameter<word> &param);
  ~MemoryPool();

  // Non-copyable and non-movable: the reference count tracks instances.
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;

 private:
  using StatsAdaptor = rmm::mr::statistics_resource_adaptor<MemoryPoolBase>;

  static inline std::unique_ptr<DefaultUpstream> base_{};
  static inline std::unique_ptr<MemoryPoolBase> pool_{};
  //! Null unless statistics are enabled; when set it is the current resource.
  static inline std::unique_ptr<StatsAdaptor> stats_{};
  static inline std::set<int> bins_{};
  static inline int refcount_ = 0;
  static inline bool stats_requested_ = false;
  //! Negative means "leave RMM's default alone".
  static inline int64_t pool_size_mib_ = -1;
  static inline int64_t release_threshold_mib_ = -1;
  static inline int64_t max_bin_mib_ = -1;

  static void AddBin(int size);
};

}  // namespace cheddar
