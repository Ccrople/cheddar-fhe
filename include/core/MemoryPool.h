#pragma once

#include <memory>
#include <set>

#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

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
 */
class MemoryPool {
  using DefaultUpstream = rmm::mr::cuda_async_memory_resource;
  using MemoryPoolBase = rmm::mr::binning_memory_resource<DefaultUpstream>;

 public:
  template <typename word>
  explicit MemoryPool(const Parameter<word> &param);
  ~MemoryPool();

  // Non-copyable and non-movable: the reference count tracks instances.
  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;

 private:
  static inline std::unique_ptr<DefaultUpstream> base_{};
  static inline std::unique_ptr<MemoryPoolBase> pool_{};
  static inline std::set<int> bins_{};
  static inline int refcount_ = 0;

  static void AddBin(int size);
};

}  // namespace cheddar
