#include "core/MemoryPool.h"

namespace cheddar {

void MemoryPool::AddBin(int size) {
  // Bins are size buckets, so a size another Context already asked for needs
  // nothing further; adding it twice would only waste a bucket.
  if (bins_.insert(size).second) {
    pool_->add_bin(size);
  }
}

template <typename word>
MemoryPool::MemoryPool(const Parameter<word> &param) {
  if (refcount_ == 0) {
    base_ = std::make_unique<DefaultUpstream>();
    pool_ = std::make_unique<MemoryPoolBase>(base_.get());
    bins_.clear();
    rmm::mr::set_current_device_resource(pool_.get());
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
    pool_.reset();
    base_.reset();
    bins_.clear();
  }
}

template MemoryPool::MemoryPool(const Parameter<uint32_t> &param);
template MemoryPool::MemoryPool(const Parameter<uint64_t> &param);

}  // namespace cheddar
