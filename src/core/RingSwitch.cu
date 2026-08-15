#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "core/RingSwitch.h"

namespace cheddar {

namespace kernel {

// res_i[limb][s] = src[limb][i + k*s], the same stride-k gather ModDecomp uses
// for its b-part. Applied to *both* components here, because the subring
// secret collapses the module rank to one.
//
// grid: (small_degree / block, k, num_total_primes)
template <typename word>
__global__ void RingSwitchGather(word *const *dst_ptrs, const word *src,
                                 int rank, int small_degree, int degree) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  const int limb = blockIdx.z;

  dst_ptrs[i][limb * small_degree + s] =
      basic::StreamingLoad(src + limb * degree + i + rank * s);
}

}  // namespace kernel

template <typename word>
RingSwitchHandler<word>::RingSwitchHandler(ConstContextPtr<word> big,
                                           ConstContextPtr<word> small)
    : big_{std::move(big)}, small_{std::move(small)} {
  const int degree = big_->param_.degree_;
  const int small_degree = small_->param_.degree_;
  AssertTrue(small_degree < degree && degree % small_degree == 0,
             "RingSwitchHandler: the small degree must properly divide the "
             "big one");
  rank_ = degree / small_degree;
  AssertTrue(IsPowOfTwo(rank_), "RingSwitchHandler: rank must be a power of "
                                "two");
  AssertTrue(small_degree % kernel_block_dim_ == 0,
             "RingSwitchHandler: small degree must be a multiple of the block "
             "dim");

  // The switched ciphertext keeps the RNS limbs it arrived with -- step 2 only
  // re-indexes coefficients -- so the two rings have to agree on the primes at
  // every level, not merely on how many there are. Checking the level
  // configuration matches is the cheap half of that; the prime lists
  // themselves are compared below.
  AssertTrue(big_->param_.max_level_ == small_->param_.max_level_,
             "RingSwitchHandler: the two rings must share a level "
             "configuration");
  for (int level = 0; level <= big_->param_.max_level_; level++) {
    const NPInfo bnp = big_->param_.LevelToNP(level);
    const NPInfo snp = small_->param_.LevelToNP(level);
    AssertTrue(bnp.num_main_ == snp.num_main_ && bnp.num_ter_ == snp.num_ter_,
               "RingSwitchHandler: prime counts differ at level " +
                   std::to_string(level));
    const auto bp = big_->param_.GetPrimeVector(bnp);
    const auto sp = small_->param_.GetPrimeVector(snp);
    AssertTrue(bp == sp,
               "RingSwitchHandler: the two rings hold different primes at "
               "level " +
                   std::to_string(level) +
                   ", so a switched ciphertext would have no home");
  }
}

template <typename word>
void RingSwitchHandler<word>::Switch(std::vector<Ct> &res, const Ct &ct,
                                     const Evk &swk) const {
  const int degree = big_->param_.degree_;
  const int small_degree = small_->param_.degree_;
  const NPInfo np = ct.GetNP();
  AssertTrue(np.num_aux_ == 0, "RingSwitch: aux primes are not supported");
  AssertTrue(!ct.HasRx(), "RingSwitch: input must not carry an rx_ part");
  AssertTrue(np.degree_ == degree,
             "RingSwitch: input does not belong to the big ring");

  const int level = big_->param_.NPToLevel(np);
  AssertTrue(level >= 0, "RingSwitch: input is not at a valid level");
  const int num_total_primes = np.GetNumTotal();

  // 1. Key-switch onto the subring secret. Ordinary MultKey: the switching key
  //    is an ordinary evaluation key of the big Context, whose alpha was
  //    chosen small enough that its modulus fits the *small* ring's budget.
  Ct switched;
  big_->MultKey(switched, ct, swk);

  // 2. The gather is defined on coefficients, so leave the NTT domain. Both
  //    components take the identical treatment -- there is no Toeplitz
  //    arrangement here, because the subring secret zeroes every module
  //    component but the first.
  DeviceVector<word> a_coeffs(num_total_primes * degree);
  DeviceVector<word> b_coeffs(num_total_primes * degree);
  auto a_view = a_coeffs.View(0);
  auto b_view = b_coeffs.View(0);
  big_->ntt_handler_.INTT(a_view, np, switched.AxConstView());
  big_->ntt_handler_.INTT(b_view, np, switched.BxConstView());

  // The outputs are the small ring's ciphertexts, so their NPInfo has to come
  // from the small parameter -- same prime counts, different ring degree. The
  // constructor already verified the primes themselves agree.
  const NPInfo small_np = small_->param_.LevelToNP(level);

  res.clear();
  res.resize(rank_);
  for (auto &r : res) {
    r.RemoveRx();
    r.ModifyNP(small_np);
    r.SetNumSlots(small_degree / 2);
    r.SetScale(switched.GetScale());
  }

  // Gather into scratch rather than straight into the ciphertexts: the NTT
  // below needs distinct source and destination, and going through scratch
  // costs one buffer instead of one extra device-to-device copy per part.
  std::vector<DeviceVector<word>> gathered_a(rank_), gathered_b(rank_);
  HostVector<word *> h_dst_a(rank_), h_dst_b(rank_);
  for (int i = 0; i < rank_; i++) {
    gathered_a[i].resize(num_total_primes * small_degree);
    gathered_b[i].resize(num_total_primes * small_degree);
    h_dst_a[i] = gathered_a[i].data();
    h_dst_b[i] = gathered_b[i].data();
  }
  DeviceVector<word *> d_dst_a(rank_), d_dst_b(rank_);
  CopyHostToDevice(d_dst_a, h_dst_a);
  CopyHostToDevice(d_dst_b, h_dst_b);

  const dim3 grid_dim(small_degree / kernel_block_dim_, rank_,
                      num_total_primes);
  kernel::RingSwitchGather<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_a.data(), a_coeffs.data(), rank_, small_degree, degree);
  kernel::RingSwitchGather<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_b.data(), b_coeffs.data(), rank_, small_degree, degree);

  // 3. Back into the NTT domain, now at the small degree. Cheddar keeps
  //    ciphertexts transformed, and the Montgomery convention matches the one
  //    ModPack uses for the same coefficient-to-NTT direction.
  for (int i = 0; i < rank_; i++) {
    auto ax_view = res[i].AxView();
    auto bx_view = res[i].BxView();
    small_->ntt_handler_.NTT(ax_view, small_np, gathered_a[i].ConstView(0),
                             true);
    small_->ntt_handler_.NTT(bx_view, small_np, gathered_b[i].ConstView(0),
                             true);
  }
}

template class RingSwitchHandler<uint32_t>;
template class RingSwitchHandler<uint64_t>;

}  // namespace cheddar
