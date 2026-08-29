#include "core/MemoryPool.h"

#include <cstdlib>
#include <cstring>

#include <thrust/optional.h>

namespace cheddar {

bool MemoryPool::SetStatisticsEnabled(bool enabled) {
  // After the first pool exists the answer has to be no. Buffers capture the
  // current device resource at allocation time and free through that same
  // pointer, so swapping the resource under a live Context is the
  // use-after-free the class comment opens with, one indirection further out.
  if (refcount_ != 0) return false;
  stats_requested_ = enabled;
  return true;
}

bool MemoryPool::StatisticsEnabled() { return stats_ != nullptr; }

bool MemoryPool::SetPoolSizeMiB(int64_t mib) {
  if (refcount_ != 0) return false;
  pool_size_mib_ = mib;
  return true;
}

bool MemoryPool::SetReleaseThresholdMiB(int64_t mib) {
  if (refcount_ != 0) return false;
  release_threshold_mib_ = mib;
  return true;
}

bool MemoryPool::SetMaxBinMiB(int64_t mib) {
  if (refcount_ != 0) return false;
  max_bin_mib_ = mib;
  return true;
}

namespace {

// -1 keeps RMM's own default for that field.
int64_t EnvMiB(const char *name, int64_t fallback) {
  const char *env = std::getenv(name);
  if (env == nullptr || *env == 0) return fallback;
  char *end = nullptr;
  const long long value = std::strtoll(env, &end, 10);
  if (end == env || value < 0) return fallback;
  return static_cast<int64_t>(value);
}

}  // namespace

MemoryPool::Usage MemoryPool::GetUsage() {
  Usage usage;
  if (stats_ == nullptr) return usage;
  const auto bytes = stats_->get_bytes_counter();
  const auto allocs = stats_->get_allocations_counter();
  usage.current_bytes = bytes.value;
  usage.peak_bytes = bytes.peak;
  usage.total_bytes = bytes.total;
  usage.current_allocations = allocs.value;
  usage.peak_allocations = allocs.peak;
  return usage;
}

void MemoryPool::AddBin(int size) {
  // A bin above the cap is refused outright: its allocations then fall through
  // to the async pool, where the release threshold actually applies. A cap of
  // zero therefore means no binning at all.
  if (max_bin_mib_ >= 0 &&
      static_cast<int64_t>(size) > max_bin_mib_ * (1 << 20)) {
    return;
  }
  // Bins are size buckets, so a size another Context already asked for needs
  // nothing further; adding it twice would only waste a bucket.
  if (bins_.insert(size).second) {
    pool_->add_bin(size);
  }
}

template <typename word>
MemoryPool::MemoryPool(const Parameter<word> &param) {
  if (refcount_ == 0) {
    // The environment is read here rather than at static-init time so that a
    // program can decide either way before its first Context, and so that a
    // run can be measured without an edit.
    const char *env = std::getenv("CHEDDAR_MEM_STATS");
    if (env != nullptr && std::strcmp(env, "0") != 0) stats_requested_ = true;
    pool_size_mib_ = EnvMiB("CHEDDAR_POOL_MIB", pool_size_mib_);
    release_threshold_mib_ =
        EnvMiB("CHEDDAR_POOL_RELEASE_MIB", release_threshold_mib_);
    max_bin_mib_ = EnvMiB("CHEDDAR_POOL_MAX_BIN_MIB", max_bin_mib_);

    // RMM's own defaults are half of free memory primed and a release
    // threshold of the whole card -- see the header. `thrust::optional{}`
    // is what asks for them; anything else overrides.
    constexpr int64_t kMiB = 1 << 20;
    thrust::optional<std::size_t> initial_pool_size{};
    if (pool_size_mib_ >= 0) {
      initial_pool_size =
          static_cast<std::size_t>(pool_size_mib_) * static_cast<std::size_t>(kMiB);
    }
    thrust::optional<std::size_t> release_threshold{};
    if (release_threshold_mib_ >= 0) {
      release_threshold = static_cast<std::size_t>(release_threshold_mib_) *
                          static_cast<std::size_t>(kMiB);
    }

    base_ = std::make_unique<DefaultUpstream>(initial_pool_size,
                                              release_threshold);
    pool_ = std::make_unique<MemoryPoolBase>(base_.get());
    bins_.clear();
    if (stats_requested_) {
      stats_ = std::make_unique<StatsAdaptor>(pool_.get());
      rmm::mr::set_current_device_resource(stats_.get());
    } else {
      rmm::mr::set_current_device_resource(pool_.get());
    }
  }
  refcount_++;

  // Hueristically add bins to save memory and speed-up bootstrapping.
  const int degree = param.degree_;
  const int word_size = param.word_size_;
  int limb_size = word_size * degree;

  // Some general sizes for small allocation sizes;
  int bin_size = 512;
  int next_threshold = limb_size;
  // Should be: 512, 2048, 8192, 32768, 131072
  for (; bin_size < next_threshold; bin_size *= 4) {
    AddBin(bin_size);
  }
  bin_size = next_threshold;
  int chunk_size = param.alpha_ * limb_size;
  for (; bin_size < chunk_size; bin_size *= 2) {
    // Maybe one more bin will be added
    AddBin(bin_size);
  }
  // Finally, about dnum additional bins;
  bin_size = chunk_size;
  int max_size = (param.L_ + param.alpha_) * limb_size;
  for (; bin_size < max_size; bin_size += chunk_size) {
    AddBin(bin_size);
  }
  AddBin(max_size);
}

MemoryPool::~MemoryPool() {
  refcount_--;
  if (refcount_ == 0) {
    // reset to cuda_device_resource
    rmm::mr::set_current_device_resource(nullptr);
    // Outermost first: the adaptor holds a raw pointer to the pool.
    stats_.reset();
    pool_.reset();
    base_.reset();
    bins_.clear();
  }
}

template MemoryPool::MemoryPool(const Parameter<uint32_t> &param);
template MemoryPool::MemoryPool(const Parameter<uint64_t> &param);

}  // namespace cheddar
