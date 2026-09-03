#include <cstdlib>

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

// The inverse interleave: dst[limb][i + k*s] = src_i[limb][s]. Every output
// coefficient is written exactly once, so no clearing is needed.
//
// grid: (small_degree / block, k, num_total_primes)
template <typename word>
__global__ void RingSwitchScatter(word *dst, const word *const *src_ptrs,
                                  int rank, int small_degree, int degree) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  const int limb = blockIdx.z;

  dst[limb * degree + i + rank * s] =
      basic::StreamingLoad(src_ptrs[i] + limb * small_degree + s);
}

// The conjugate-invariant forms. On R+ the subring secret still zeroes every
// module component but the first -- with sigma_j = 0 for j >= 1 the a~
// formula of Mlwe.cu collapses to a~_l[0] = alpha_l -- so the switch stays
// "one key switch, then split the components". What changes is the split
// itself: the components are not stride slices but the alternating-sign
// suffix-sum scan down each class pair (i, k-i),
//
//     alpha_i[t] = a[tk + i] - alpha_{k-i}[t+1]
//
// and the recomposition is the banded two-term inverse. Both kernels mirror
// Mlwe.cu's CiModDecompScan and CiModPackB, the way RingSwitchGather mirrors
// ModDecompB; see the comments there for the derivation and the launch
// reasoning.
//
// grid: 1D over num_limbs * (rank / 2 + 1) threads
template <typename word>
__global__ void CiRingSwitchScan(word *const *dst_ptrs, const word *src,
                                 const word *primes, int rank,
                                 int small_degree, int degree, int num_limbs) {
  const int num_chains = rank / 2 + 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_limbs * num_chains) return;
  const int limb = tid / num_chains;
  const int chain = tid - limb * num_chains;

  const word *src_limb = src + limb * degree;

  if (chain == 0) {
    word *dst = dst_ptrs[0] + limb * small_degree;
    for (int t = 0; t < small_degree; t++) {
      dst[t] = basic::StreamingLoad(src_limb + t * rank);
    }
    return;
  }

  const word prime = basic::StreamingLoadConst(primes + limb);
  const int i = chain;
  const int mi = rank - chain;
  word *dst_i = dst_ptrs[i] + limb * small_degree;
  word *dst_m = dst_ptrs[mi] + limb * small_degree;

  word acc_i = 0;  // alpha_i[t + 1]
  word acc_m = 0;  // alpha_{k-i}[t + 1]
  for (int t = small_degree - 1; t >= 0; t--) {
    const word vi = basic::StreamingLoad(src_limb + t * rank + i);
    const word vm = basic::StreamingLoad(src_limb + t * rank + mi);
    const word new_i = basic::Sub(vi, acc_m, prime);
    const word new_m = basic::Sub(vm, acc_i, prime);
    dst_i[t] = new_i;
    dst_m[t] = new_m;
    acc_i = new_i;
    acc_m = new_m;
  }
}

// dst[limb][tk + i] = src_i[limb][t] + src_{k-i}[limb][t+1], the banded
// inverse of the scan; the i = 0 class is a pure copy.
//
// grid: (degree / block, num_total_primes)
template <typename word>
__global__ void CiRingSwitchRecompose(word *dst, const word *const *src_ptrs,
                                      const word *primes, int log_rank,
                                      int small_degree, int degree) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;

  const int rank = 1 << log_rank;
  const int i = x & (rank - 1);
  const int t = x >> log_rank;

  word value = basic::StreamingLoad(src_ptrs[i] + limb * small_degree + t);
  if (i != 0 && t + 1 < small_degree) {
    const word prime = basic::StreamingLoadConst(primes + limb);
    const word mirror =
        basic::StreamingLoad(src_ptrs[rank - i] + limb * small_degree + t + 1);
    value = basic::Add(value, mirror, prime);
  }
  dst[limb * degree + x] = value;
}

// One polynomial per blockIdx.z gathered into a strided buffer (the batch
// key switch wants b then a back to back per ciphertext).
template <typename word>
__global__ void RingSwitchGatherPoly(word *dst, int words,
                                     const word *const *src_ptrs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int z = blockIdx.z;
  if (i < words) {
    dst[static_cast<size_t>(z) * words + i] =
        basic::StreamingLoad(src_ptrs[z] + i);
  }
}

// `CiRingSwitchScan` over a GROUP of ciphertexts: one thread per
// (ciphertext, limb, chain), the per-(limb, chain) walk exactly the serial
// kernel's. The serial launch is a single block of `limbs * (rank/2 + 1)`
// threads; the group is what fills the card.
template <typename word>
__global__ void CiRingSwitchScanBatch(word *const *dst_ptrs,
                                      const word *const *src_ptrs,
                                      const word *primes, int rank,
                                      int small_degree, int degree,
                                      int num_limbs, int num_cts) {
  const int num_chains = rank / 2 + 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_cts * num_limbs * num_chains) return;
  const int ct = tid / (num_limbs * num_chains);
  const int rem = tid - ct * (num_limbs * num_chains);
  const int limb = rem / num_chains;
  const int chain = rem - limb * num_chains;

  const word *src_limb = src_ptrs[ct] + limb * degree;
  word *const *dst_ct = dst_ptrs + static_cast<size_t>(ct) * rank;

  if (chain == 0) {
    word *dst = dst_ct[0] + limb * small_degree;
    for (int t = 0; t < small_degree; t++) {
      dst[t] = basic::StreamingLoad(src_limb + t * rank);
    }
    return;
  }

  const word prime = basic::StreamingLoadConst(primes + limb);
  const int i = chain;
  const int mi = rank - chain;
  word *dst_i = dst_ct[i] + limb * small_degree;
  word *dst_m = dst_ct[mi] + limb * small_degree;

  word acc_i = 0;
  word acc_m = 0;
  for (int t = small_degree - 1; t >= 0; t--) {
    const word vi = basic::StreamingLoad(src_limb + t * rank + i);
    const word vm = basic::StreamingLoad(src_limb + t * rank + mi);
    const word new_i = basic::Sub(vi, acc_m, prime);
    const word new_m = basic::Sub(vm, acc_i, prime);
    dst_i[t] = new_i;
    dst_m[t] = new_m;
    acc_i = new_i;
    acc_m = new_m;
  }
}

// `CiRingSwitchRecompose` over a GROUP: blockIdx.z picks the ciphertext.
template <typename word>
__global__ void CiRingSwitchRecomposeBatch(word *dst, size_t dst_ct_stride,
                                           const word *const *src_ptrs,
                                           const word *primes, int log_rank,
                                           int small_degree, int degree) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;
  const int ct = blockIdx.z;

  const int rank = 1 << log_rank;
  const int i = x & (rank - 1);
  const int t = x >> log_rank;
  const word *const *src_ct = src_ptrs + static_cast<size_t>(ct) * rank;

  word value = basic::StreamingLoad(src_ct[i] + limb * small_degree + t);
  if (i != 0 && t + 1 < small_degree) {
    const word prime = basic::StreamingLoadConst(primes + limb);
    const word mirror =
        basic::StreamingLoad(src_ct[rank - i] + limb * small_degree + t + 1);
    value = basic::Add(value, mirror, prime);
  }
  dst[ct * dst_ct_stride + limb * degree + x] = value;
}

}  // namespace kernel

namespace {
bool RingSwitchSerial() {
  static const bool serial = [] {
    const char *env = std::getenv("CHEDDAR_RING_SWITCH_SERIAL");
    return env != nullptr && env[0] == '1';
  }();
  return serial;
}
}  // namespace

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
  AssertTrue(big_->param_.conjugate_invariant_ ==
                 small_->param_.conjugate_invariant_,
             "RingSwitchHandler: the two rings must be the same kind -- a "
             "conjugate-invariant ciphertext has no home in a cyclotomic "
             "subring, nor the reverse");

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
    r.SetNumSlots(small_->param_.MaxNumSlots());
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

  if (big_->param_.conjugate_invariant_) {
    const word *primes = big_->param_.GetPrimesPtr(np);
    constexpr int scan_block_dim = 128;
    const int num_chains = rank_ / 2 + 1;
    const int scan_grid_dim =
        DivCeil(num_total_primes * num_chains, scan_block_dim);
    kernel::CiRingSwitchScan<word><<<scan_grid_dim, scan_block_dim>>>(
        d_dst_a.data(), a_coeffs.data(), primes, rank_, small_degree, degree,
        num_total_primes);
    kernel::CiRingSwitchScan<word><<<scan_grid_dim, scan_block_dim>>>(
        d_dst_b.data(), b_coeffs.data(), primes, rank_, small_degree, degree,
        num_total_primes);
  } else {
    const dim3 grid_dim(small_degree / kernel_block_dim_, rank_,
                        num_total_primes);
    kernel::RingSwitchGather<word><<<grid_dim, kernel_block_dim_>>>(
        d_dst_a.data(), a_coeffs.data(), rank_, small_degree, degree);
    kernel::RingSwitchGather<word><<<grid_dim, kernel_block_dim_>>>(
        d_dst_b.data(), b_coeffs.data(), rank_, small_degree, degree);
  }

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

template <typename word>
void RingSwitchHandler<word>::SwitchBatch(std::vector<std::vector<Ct>> &res,
                                          const std::vector<const Ct *> &cts,
                                          const Evk &swk) const {
  const int n = static_cast<int>(cts.size());
  res.clear();
  res.resize(n);
  if (RingSwitchSerial() || n == 1 || !big_->param_.conjugate_invariant_) {
    for (int c = 0; c < n; c++) Switch(res[c], *cts[c], swk);
    return;
  }
  const int degree = big_->param_.degree_;
  const int small_degree = small_->param_.degree_;
  const NPInfo np = cts[0]->GetNP();
  AssertTrue(np.num_aux_ == 0, "RingSwitch: aux primes are not supported");
  AssertTrue(np.degree_ == degree,
             "RingSwitch: input does not belong to the big ring");
  for (int c = 0; c < n; c++) {
    AssertTrue(cts[c]->GetNP() == np && !cts[c]->HasRx(),
               "RingSwitch::SwitchBatch: the group's inputs disagree");
  }
  const int level = big_->param_.NPToLevel(np);
  AssertTrue(level >= 0, "RingSwitch: input is not at a valid level");
  const int num_total_primes = np.GetNumTotal();
  const size_t q_words = static_cast<size_t>(num_total_primes) * degree;
  const size_t ct_words = 2 * q_words;

  // 1. The group's key switches as ONE MultKeyBatch with the one key: the
  // (b, a) parts gathered back to back per ciphertext.
  DeviceVector<word> switched(static_cast<size_t>(n) * ct_words);
  {
    HostVector<const word *> h_ptrs(2 * n);
    for (int c = 0; c < n; c++) {
      h_ptrs[2 * c] = cts[c]->bx_.data();
      h_ptrs[2 * c + 1] = cts[c]->ax_.data();
    }
    DeviceVector<const word *> d_ptrs(2 * n);
    CopyHostToDevice(d_ptrs, h_ptrs);
    DeviceVector<word> src_buf(static_cast<size_t>(n) * ct_words);
    dim3 grid(DivCeil(static_cast<int>(q_words), kernel_block_dim_), 1,
              2 * n);
    kernel::RingSwitchGatherPoly<word><<<grid, kernel_block_dim_>>>(
        src_buf.data(), static_cast<int>(q_words), d_ptrs.data());
    std::vector<const Evk *> keys(n, &swk);
    big_->MultKeyBatch(switched.data(), static_cast<int>(ct_words),
                       src_buf.data(), static_cast<int>(ct_words), np, keys,
                       n);
  }

  // 2. Leave the NTT domain, per ciphertext (the same INTT calls).
  DeviceVector<word> a_coeffs(static_cast<size_t>(n) * q_words);
  DeviceVector<word> b_coeffs(static_cast<size_t>(n) * q_words);
  for (int c = 0; c < n; c++) {
    DvView<word> a_view(a_coeffs.data() + c * q_words,
                        static_cast<int>(q_words), 0);
    DvView<word> b_view(b_coeffs.data() + c * q_words,
                        static_cast<int>(q_words), 0);
    DvConstView<word> bx_src(switched.data() + c * ct_words,
                             static_cast<int>(q_words), 0);
    DvConstView<word> ax_src(switched.data() + c * ct_words + q_words,
                             static_cast<int>(q_words), 0);
    big_->ntt_handler_.INTT(a_view, np, ax_src);
    big_->ntt_handler_.INTT(b_view, np, bx_src);
  }

  const NPInfo small_np = small_->param_.LevelToNP(level);
  const size_t sp_words = static_cast<size_t>(num_total_primes) * small_degree;
  for (int c = 0; c < n; c++) {
    res[c].resize(rank_);
    for (auto &r : res[c]) {
      r.RemoveRx();
      r.ModifyNP(small_np);
      r.SetNumSlots(small_->param_.MaxNumSlots());
      r.SetScale(cts[c]->GetScale());
    }
  }

  // 3. The component scan, ONE kernel over (ciphertext, limb, chain), into
  // scratch (the NTT below needs distinct source and destination).
  DeviceVector<word> gathered_a(static_cast<size_t>(n) * rank_ * sp_words);
  DeviceVector<word> gathered_b(static_cast<size_t>(n) * rank_ * sp_words);
  {
    HostVector<word *> h_dst(2 * static_cast<size_t>(n) * rank_);
    HostVector<const word *> h_src(2 * n);
    for (int c = 0; c < n; c++) {
      for (int i = 0; i < rank_; i++) {
        h_dst[static_cast<size_t>(c) * rank_ + i] =
            gathered_a.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words;
        h_dst[static_cast<size_t>(n) * rank_ +
              static_cast<size_t>(c) * rank_ + i] =
            gathered_b.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words;
      }
      h_src[c] = a_coeffs.data() + c * q_words;
      h_src[n + c] = b_coeffs.data() + c * q_words;
    }
    DeviceVector<word *> d_dst(2 * static_cast<size_t>(n) * rank_);
    DeviceVector<const word *> d_src(2 * n);
    CopyHostToDevice(d_dst, h_dst);
    CopyHostToDevice(d_src, h_src);
    const word *primes = big_->param_.GetPrimesPtr(np);
    constexpr int scan_block_dim = 128;
    const int num_chains = rank_ / 2 + 1;
    const int scan_grid_dim =
        DivCeil(n * num_total_primes * num_chains, scan_block_dim);
    kernel::CiRingSwitchScanBatch<word><<<scan_grid_dim, scan_block_dim>>>(
        d_dst.data(), d_src.data(), primes, rank_, small_degree, degree,
        num_total_primes, n);
    kernel::CiRingSwitchScanBatch<word><<<scan_grid_dim, scan_block_dim>>>(
        d_dst.data() + static_cast<size_t>(n) * rank_,
        d_src.data() + n, primes, rank_, small_degree, degree,
        num_total_primes, n);
  }

  // 4. Back into the NTT domain at the small degree, per part.
  for (int c = 0; c < n; c++) {
    for (int i = 0; i < rank_; i++) {
      auto ax_view = res[c][i].AxView();
      auto bx_view = res[c][i].BxView();
      DvConstView<word> a_src(
          gathered_a.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words,
          static_cast<int>(sp_words), 0);
      DvConstView<word> b_src(
          gathered_b.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words,
          static_cast<int>(sp_words), 0);
      small_->ntt_handler_.NTT(ax_view, small_np, a_src, true);
      small_->ntt_handler_.NTT(bx_view, small_np, b_src, true);
    }
  }
}

template <typename word>
void RingSwitchHandler<word>::SwitchBack(Ct &res, const std::vector<Ct> &parts,
                                         const Evk &swk) const {
  const int degree = big_->param_.degree_;
  const int small_degree = small_->param_.degree_;
  AssertTrue(static_cast<int>(parts.size()) == rank_,
             "RingSwitch::SwitchBack: expected exactly rank ciphertexts");

  const NPInfo small_np = parts.at(0).GetNP();
  AssertTrue(small_np.num_aux_ == 0,
             "RingSwitch::SwitchBack: aux primes are not supported");
  AssertTrue(small_np.degree_ == small_degree,
             "RingSwitch::SwitchBack: inputs do not belong to the small ring");
  for (const auto &p : parts) {
    AssertTrue(p.GetNP() == small_np,
               "RingSwitch::SwitchBack: ciphertexts differ in NP");
    AssertTrue(!p.HasRx(),
               "RingSwitch::SwitchBack: inputs must not carry an rx_ part");
  }

  const int level = small_->param_.NPToLevel(small_np);
  AssertTrue(level >= 0, "RingSwitch::SwitchBack: inputs are not at a valid "
                         "level");
  const int num_total_primes = small_np.GetNumTotal();

  // 1. Leave the NTT domain at the small degree; the interleave is defined on
  //    coefficients.
  std::vector<DeviceVector<word>> a_coeffs(rank_), b_coeffs(rank_);
  HostVector<word *> h_src_a(rank_), h_src_b(rank_);
  for (int i = 0; i < rank_; i++) {
    a_coeffs[i].resize(num_total_primes * small_degree);
    b_coeffs[i].resize(num_total_primes * small_degree);
    auto av = a_coeffs[i].View(0);
    auto bv = b_coeffs[i].View(0);
    small_->ntt_handler_.INTT(av, small_np, parts[i].AxConstView());
    small_->ntt_handler_.INTT(bv, small_np, parts[i].BxConstView());
    h_src_a[i] = a_coeffs[i].data();
    h_src_b[i] = b_coeffs[i].data();
  }
  DeviceVector<word *> d_src_a(rank_), d_src_b(rank_);
  CopyHostToDevice(d_src_a, h_src_a);
  CopyHostToDevice(d_src_b, h_src_b);

  // 2. X^k-adic recomposition. Free, and already a valid ciphertext at degree
  //    N under the subring secret.
  DeviceVector<word> a_full(num_total_primes * degree);
  DeviceVector<word> b_full(num_total_primes * degree);
  if (big_->param_.conjugate_invariant_) {
    const word *primes = small_->param_.GetPrimesPtr(small_np);
    const int log_rank = Log2Floor(rank_);
    const dim3 grid_dim(degree / kernel_block_dim_, num_total_primes);
    kernel::CiRingSwitchRecompose<word><<<grid_dim, kernel_block_dim_>>>(
        a_full.data(), d_src_a.data(), primes, log_rank, small_degree,
        degree);
    kernel::CiRingSwitchRecompose<word><<<grid_dim, kernel_block_dim_>>>(
        b_full.data(), d_src_b.data(), primes, log_rank, small_degree,
        degree);
  } else {
    const dim3 grid_dim(small_degree / kernel_block_dim_, rank_,
                        num_total_primes);
    kernel::RingSwitchScatter<word><<<grid_dim, kernel_block_dim_>>>(
        a_full.data(), d_src_a.data(), rank_, small_degree, degree);
    kernel::RingSwitchScatter<word><<<grid_dim, kernel_block_dim_>>>(
        b_full.data(), d_src_b.data(), rank_, small_degree, degree);
  }

  // 3. Back into the NTT domain at the big degree, as one ciphertext of the
  //    big Context.
  const NPInfo big_np = big_->param_.LevelToNP(level);
  Ct recomposed;
  recomposed.RemoveRx();
  recomposed.ModifyNP(big_np);
  recomposed.SetNumSlots(big_->param_.MaxNumSlots());
  recomposed.SetScale(parts.at(0).GetScale());
  auto ax_view = recomposed.AxView();
  auto bx_view = recomposed.BxView();
  big_->ntt_handler_.NTT(ax_view, big_np, a_full.ConstView(0), true);
  big_->ntt_handler_.NTT(bx_view, big_np, b_full.ConstView(0), true);

  // 4. One key switch off the subring secret.
  big_->MultKey(res, recomposed, swk);
}

template <typename word>
void RingSwitchHandler<word>::SwitchBackBatch(
    const std::vector<Ct *> &res,
    const std::vector<const std::vector<Ct> *> &parts, const Evk &swk) const {
  const int n = static_cast<int>(parts.size());
  AssertTrue(static_cast<int>(res.size()) == n,
             "RingSwitch::SwitchBackBatch: outputs disagree with inputs");
  if (RingSwitchSerial() || n == 1 || !big_->param_.conjugate_invariant_) {
    for (int c = 0; c < n; c++) SwitchBack(*res[c], *parts[c], swk);
    return;
  }
  const int degree = big_->param_.degree_;
  const int small_degree = small_->param_.degree_;
  const NPInfo small_np = parts[0]->at(0).GetNP();
  AssertTrue(small_np.num_aux_ == 0,
             "RingSwitch::SwitchBack: aux primes are not supported");
  AssertTrue(small_np.degree_ == small_degree,
             "RingSwitch::SwitchBack: inputs do not belong to the small ring");
  for (int c = 0; c < n; c++) {
    AssertTrue(static_cast<int>(parts[c]->size()) == rank_,
               "RingSwitch::SwitchBack: expected exactly rank ciphertexts");
    for (const auto &p : *parts[c]) {
      AssertTrue(p.GetNP() == small_np && !p.HasRx(),
                 "RingSwitch::SwitchBackBatch: the group's inputs disagree");
    }
  }
  const int level = small_->param_.NPToLevel(small_np);
  AssertTrue(level >= 0, "RingSwitch::SwitchBack: inputs are not at a valid "
                         "level");
  const int num_total_primes = small_np.GetNumTotal();
  const size_t sp_words = static_cast<size_t>(num_total_primes) * small_degree;
  const size_t q_words = static_cast<size_t>(num_total_primes) * degree;
  const size_t ct_words = 2 * q_words;

  // 1. Leave the NTT domain at the small degree, per part.
  DeviceVector<word> a_coeffs(static_cast<size_t>(n) * rank_ * sp_words);
  DeviceVector<word> b_coeffs(static_cast<size_t>(n) * rank_ * sp_words);
  for (int c = 0; c < n; c++) {
    for (int i = 0; i < rank_; i++) {
      DvView<word> av(
          a_coeffs.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words,
          static_cast<int>(sp_words), 0);
      DvView<word> bv(
          b_coeffs.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words,
          static_cast<int>(sp_words), 0);
      small_->ntt_handler_.INTT(av, small_np, parts[c]->at(i).AxConstView());
      small_->ntt_handler_.INTT(bv, small_np, parts[c]->at(i).BxConstView());
    }
  }

  // 2. The recomposition, ONE kernel over the group per component, into a
  // (b, a)-per-ciphertext buffer laid out for the batch key switch.
  DeviceVector<word> recomposed(static_cast<size_t>(n) * ct_words);
  {
    HostVector<const word *> h_src(2 * static_cast<size_t>(n) * rank_);
    for (int c = 0; c < n; c++) {
      for (int i = 0; i < rank_; i++) {
        h_src[static_cast<size_t>(c) * rank_ + i] =
            a_coeffs.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words;
        h_src[static_cast<size_t>(n) * rank_ +
              static_cast<size_t>(c) * rank_ + i] =
            b_coeffs.data() + (static_cast<size_t>(c) * rank_ + i) * sp_words;
      }
    }
    DeviceVector<const word *> d_src(2 * static_cast<size_t>(n) * rank_);
    CopyHostToDevice(d_src, h_src);
    const word *primes = small_->param_.GetPrimesPtr(small_np);
    const int log_rank = Log2Floor(rank_);
    const dim3 grid_dim(degree / kernel_block_dim_, num_total_primes, n);
    // b lands first in each slice, a after it.
    kernel::CiRingSwitchRecomposeBatch<word><<<grid_dim, kernel_block_dim_>>>(
        recomposed.data() + q_words, ct_words,
        d_src.data(), primes, log_rank, small_degree, degree);
    kernel::CiRingSwitchRecomposeBatch<word><<<grid_dim, kernel_block_dim_>>>(
        recomposed.data(), ct_words,
        d_src.data() + static_cast<size_t>(n) * rank_, primes, log_rank,
        small_degree, degree);
  }
  a_coeffs = DeviceVector<word>();
  b_coeffs = DeviceVector<word>();

  // 3. Back into the NTT domain at the big degree, per ciphertext, in place
  // through scratch (NTT needs distinct source and destination).
  const NPInfo big_np = big_->param_.LevelToNP(level);
  DeviceVector<word> ntted(static_cast<size_t>(n) * ct_words);
  for (int c = 0; c < n; c++) {
    DvView<word> b_view(ntted.data() + c * ct_words,
                        static_cast<int>(q_words), 0);
    DvView<word> a_view(ntted.data() + c * ct_words + q_words,
                        static_cast<int>(q_words), 0);
    DvConstView<word> b_src(recomposed.data() + c * ct_words,
                            static_cast<int>(q_words), 0);
    DvConstView<word> a_src(recomposed.data() + c * ct_words + q_words,
                            static_cast<int>(q_words), 0);
    big_->ntt_handler_.NTT(a_view, big_np, a_src, true);
    big_->ntt_handler_.NTT(b_view, big_np, b_src, true);
  }
  recomposed = DeviceVector<word>();

  // 4. The key switches off the subring secret as ONE MultKeyBatch.
  DeviceVector<word> out(static_cast<size_t>(n) * ct_words);
  {
    std::vector<const Evk *> keys(n, &swk);
    big_->MultKeyBatch(out.data(), static_cast<int>(ct_words), ntted.data(),
                       static_cast<int>(ct_words), big_np, keys, n);
  }
  for (int c = 0; c < n; c++) {
    Ct &r = *res[c];
    r.RemoveRx();
    r.ModifyNP(big_np);
    r.SetNumSlots(big_->param_.MaxNumSlots());
    r.SetScale(parts[c]->at(0).GetScale());
    cudaMemcpyAsync(r.bx_.data(), out.data() + c * ct_words,
                    q_words * sizeof(word), cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy);
    cudaMemcpyAsync(r.ax_.data(), out.data() + c * ct_words + q_words,
                    q_words * sizeof(word), cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy);
  }
}

template class RingSwitchHandler<uint32_t>;
template class RingSwitchHandler<uint64_t>;

}  // namespace cheddar
