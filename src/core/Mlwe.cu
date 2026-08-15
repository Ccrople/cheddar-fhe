#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "core/Mlwe.h"

namespace cheddar {

namespace kernel {

// b-part: res_i[limb][s] = b[limb][i + k*s], a plain stride-k gather.
//
// grid: (small_degree / block, k, num_total_primes)
template <typename word>
__global__ void ModDecompB(word *const *dst_ptrs, const word *src, int rank,
                           int small_degree, int degree) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y;
  const int limb = blockIdx.z;

  const word value =
      basic::StreamingLoad(src + limb * degree + i + rank * s);
  dst_ptrs[i][limb * small_degree + s] = value;
}

// a-part: res_i[limb][j][s] = a~_i[j][s], where
//
//     a~_i[j] = e*_{i-j}(a)          for j <= i
//             = Y * e*_{i-j+k}(a)    for j >  i
//
// and e*_l(a)[s] = a[l + k*s], (Y*q)[0] = -q[N'-1], (Y*q)[s] = q[s-1].
//
// grid: (small_degree / block, k /* j */, num_total_primes), looping over i,
// so that the k*k*N' outputs are produced with one thread per (j, s) pair.
template <typename word>
__global__ void ModDecompA(word *const *dst_ptrs, const word *src,
                           const word *primes, int rank, int small_degree,
                           int degree) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int limb = blockIdx.z;

  const word prime = basic::StreamingLoadConst(primes + limb);
  const word *src_limb = src + limb * degree;

  for (int i = 0; i < rank; i++) {
    word value;
    if (j <= i) {
      // e*_{i-j}(a)[s] = a[(i - j) + k*s]
      value = basic::StreamingLoad(src_limb + (i - j) + rank * s);
    } else {
      // (Y * e*_{i-j+k}(a))[s]
      const int l = i - j + rank;
      if (s == 0) {
        const word tail =
            basic::StreamingLoad(src_limb + l + rank * (small_degree - 1));
        value = basic::Negate(tail, prime);
      } else {
        value = basic::StreamingLoad(src_limb + l + rank * (s - 1));
      }
    }
    dst_ptrs[i][(limb * rank + j) * small_degree + s] = value;
  }
}

// The inverse interleaving, for both parts:
//
//     A_j[i + k*s] = a_i[j][s],     B[i + k*s] = b_i[s]
//
// Writing x = i + k*s for the output coefficient gives i = x mod k and
// s = x / k, both exact shifts since k is a power of two. Indexing by the
// output that way keeps the stores of a warp contiguous, at the cost of
// gathering the loads -- the same trade ModDecomp makes in the other
// direction.
//
// grid: (degree / block, k /* j */, num_total_primes)
template <typename word>
__global__ void ModPackA(word *const *dst_ptrs, const word *const *src_ptrs,
                         int log_rank, int small_degree, int degree) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int limb = blockIdx.z;

  const int rank = 1 << log_rank;
  const int i = x & (rank - 1);
  const int s = x >> log_rank;

  const word value = basic::StreamingLoad(
      src_ptrs[i] + (limb * rank + j) * small_degree + s);
  dst_ptrs[j][limb * degree + x] = value;
}

// grid: (degree / block, num_total_primes)
template <typename word>
__global__ void ModPackB(word *dst, const word *const *src_ptrs, int log_rank,
                         int small_degree, int degree) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;

  const int i = x & ((1 << log_rank) - 1);
  const int s = x >> log_rank;

  const word value =
      basic::StreamingLoad(src_ptrs[i] + limb * small_degree + s);
  dst[limb * degree + x] = value;
}

}  // namespace kernel

template <typename word>
int MlweCiphertext<word>::GetRank() const {
  return rank_;
}

template <typename word>
int MlweCiphertext<word>::GetDegree() const {
  return degree_;
}

template <typename word>
NPInfo MlweCiphertext<word>::GetNP() const {
  return np_;
}

template <typename word>
double MlweCiphertext<word>::GetScale() const {
  return scale_;
}

template <typename word>
MlweHandler<word>::MlweHandler(const Parameter<word> &param,
                               const NTTHandler<word> &ntt_handler)
    : param_{param}, ntt_handler_{ntt_handler} {
  AssertTrue(param_.degree_ % kernel_block_dim_ == 0,
             "MlweHandler: Invalid kernel block dim");
}

template <typename word>
void MlweHandler<word>::ComponentToHostCoeffs(HostVector<word> &res,
                                              const DvConstView<word> &src,
                                              const NPInfo &np) const {
  const int degree = param_.degree_;
  const int num_total_primes = np.GetNumTotal();

  DeviceVector<word> coeffs(num_total_primes * degree);
  auto view = coeffs.View(np.num_aux_ * degree);
  ntt_handler_.INTT(view, np, src);
  CopyDeviceToHost(res, coeffs);
}

template <typename word>
void MlweHandler<word>::ModDecomp(std::vector<MlweCiphertext<word>> &res,
                                  const Ct &ct, int small_degree) const {
  const int degree = param_.degree_;
  AssertTrue(small_degree > 0 && IsPowOfTwo(small_degree) &&
                 degree % small_degree == 0 && small_degree < degree,
             "ModDecomp: small_degree must be a power of two properly "
             "dividing the ring degree");
  AssertTrue(small_degree % kernel_block_dim_ == 0,
             "ModDecomp: small_degree must be a multiple of the block dim");

  const NPInfo np = ct.GetNP();
  AssertTrue(np.num_aux_ == 0, "ModDecomp: aux primes are not supported");
  AssertTrue(!ct.HasRx(), "ModDecomp: input must not carry an rx_ part");

  const int rank = degree / small_degree;
  const int num_total_primes = np.GetNumTotal();

  // The decomposition is defined on coefficients, so undo the NTT first.
  // INTT's default montgomery_conversion = true multiplies by a plain N^{-1}
  // (NTT.cu:1262, and MultMontgomery drops one factor of R), so what comes out
  // is an ordinary residue, not a Montgomery one. The re-indexing below moves
  // words and does not care, and the product downstream does not either --
  // PcmmAccum's plaintext operand carries the Montgomery factor. ModPack has
  // to know, though: it transforms back with montgomery_conversion = true.
  DeviceVector<word> a_coeffs(num_total_primes * degree);
  DeviceVector<word> b_coeffs(num_total_primes * degree);
  auto a_view = a_coeffs.View(0);
  auto b_view = b_coeffs.View(0);
  ntt_handler_.INTT(a_view, np, ct.AxConstView());
  ntt_handler_.INTT(b_view, np, ct.BxConstView());

  res.clear();
  res.resize(rank);
  for (auto &r : res) {
    r.rank_ = rank;
    r.degree_ = small_degree;
    r.np_ = np;
    r.scale_ = ct.GetScale();
    r.a_.resize(num_total_primes * rank * small_degree);
    r.b_.resize(num_total_primes * small_degree);
  }

  HostVector<word *> h_dst_a(rank), h_dst_b(rank);
  for (int i = 0; i < rank; i++) {
    h_dst_a[i] = res[i].a_.data();
    h_dst_b[i] = res[i].b_.data();
  }
  DeviceVector<word *> d_dst_a(rank), d_dst_b(rank);
  CopyHostToDevice(d_dst_a, h_dst_a);
  CopyHostToDevice(d_dst_b, h_dst_b);

  const word *primes = param_.GetPrimesPtr(np);
  const dim3 grid_dim(small_degree / kernel_block_dim_, rank,
                      num_total_primes);

  kernel::ModDecompB<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_b.data(), b_coeffs.data(), rank, small_degree, degree);
  kernel::ModDecompA<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_a.data(), a_coeffs.data(), primes, rank, small_degree, degree);
}

template <typename word>
void MlweHandler<word>::ModPack(ConstContextPtr<word> context, Ct &res,
                                const std::vector<MlweCiphertext<word>> &cts,
                                const std::vector<const Evk *> &keys) const {
  const int degree = param_.degree_;
  AssertTrue(!cts.empty(), "ModPack: no input ciphertexts");

  const int rank = cts.at(0).rank_;
  const int small_degree = cts.at(0).degree_;
  const NPInfo np = cts.at(0).np_;
  const double scale = cts.at(0).scale_;

  AssertTrue(rank > 1 && IsPowOfTwo(rank) && rank * small_degree == degree,
             "ModPack: rank and degree do not decompose the ring degree");
  AssertTrue(static_cast<int>(cts.size()) == rank,
             "ModPack: expected exactly rank input ciphertexts");
  AssertTrue(static_cast<int>(keys.size()) == rank,
             "ModPack: expected exactly rank switching keys");
  AssertTrue(np.num_aux_ == 0, "ModPack: aux primes are not supported");
  for (const auto &ct : cts) {
    AssertTrue(ct.rank_ == rank && ct.degree_ == small_degree,
               "ModPack: ciphertexts differ in rank or degree");
    AssertTrue(ct.np_ == np, "ModPack: ciphertexts differ in NP");
  }

  const int level = param_.NPToLevel(np);
  AssertTrue(level >= 0, "ModPack: inputs are not at a valid level");
  for (const auto *key : keys) {
    AssertTrue(key != nullptr, "ModPack: null switching key");
    // The dense-to-sparse key takes a different mod-switch handler inside
    // Context::MultKey (Context.cpp:568). Excluding it here is what lets the
    // single mod-down below pick its handler by level alone.
    AssertTrue(key->GetNP().num_aux_ == param_.alpha_,
               "ModPack: switching keys must be ordinary evaluation keys");
  }

  const int num_total_primes = np.GetNumTotal();
  const int log_rank = Log2Floor(rank);

  // 1. X^k-adic recomposition, still in the coefficient domain.
  DeviceVector<word> b_coeffs(num_total_primes * degree);
  std::vector<DeviceVector<word>> a_coeffs(rank);
  for (auto &a : a_coeffs) {
    a.resize(num_total_primes * degree);
  }

  HostVector<word *> h_src_a(rank), h_src_b(rank), h_dst_a(rank);
  for (int i = 0; i < rank; i++) {
    h_src_a[i] = const_cast<word *>(cts[i].a_.data());
    h_src_b[i] = const_cast<word *>(cts[i].b_.data());
    h_dst_a[i] = a_coeffs[i].data();
  }
  DeviceVector<word *> d_src_a(rank), d_src_b(rank), d_dst_a(rank);
  CopyHostToDevice(d_src_a, h_src_a);
  CopyHostToDevice(d_src_b, h_src_b);
  CopyHostToDevice(d_dst_a, h_dst_a);

  const dim3 grid_a(degree / kernel_block_dim_, rank, num_total_primes);
  kernel::ModPackA<word><<<grid_a, kernel_block_dim_>>>(
      d_dst_a.data(), d_src_a.data(), log_rank, small_degree, degree);

  const dim3 grid_b(degree / kernel_block_dim_, num_total_primes);
  kernel::ModPackB<word><<<grid_b, kernel_block_dim_>>>(
      b_coeffs.data(), d_src_b.data(), log_rank, small_degree, degree);

  // 2. Key-switch (A_j, 0) from the j-th embedded secret to the ordinary one,
  //    accumulating before the mod-down. The b-part is zero throughout, so
  //    the p_prod * bx term MultKeyNoModDown folds in contributes nothing and
  //    B can be added once at the end, exactly as the paper writes it.
  Ct input;
  input.RemoveRx();
  input.ModifyNP(np);
  input.SetScale(scale);
  input.SetNumSlots(degree / 2);
  cudaMemsetAsync(input.bx_.data(), 0,
                  static_cast<size_t>(num_total_primes) * degree * sizeof(word),
                  cudaStreamLegacy);

  Ct accum, partial;
  for (int j = 0; j < rank; j++) {
    auto ax_view = input.AxView();
    ntt_handler_.NTT(ax_view, np, a_coeffs[j].ConstView(), true);

    if (j == 0) {
      context->MultKeyNoModDown(accum, input, *keys[j]);
    } else {
      context->MultKeyNoModDown(partial, input, *keys[j]);
      context->Add(accum, accum, partial);
    }
  }

  // 3. One mod-down for all k switches, then the recomposed B.
  res.RemoveRx();
  res.ModifyNP(np);
  res.SetScale(scale);
  res.SetNumSlots(degree / 2);

  const auto &mod_switcher = context->mod_switch_handlers_.at(level);
  auto res_bx_view = res.BxView();
  auto res_ax_view = res.AxView();
  mod_switcher.ModDown(res_bx_view, accum.BxConstView());
  mod_switcher.ModDown(res_ax_view, accum.AxConstView());

  DeviceVector<word> b_ntt(num_total_primes * degree);
  auto b_ntt_view = b_ntt.View(0);
  ntt_handler_.NTT(b_ntt_view, np, b_coeffs.ConstView(), true);

  std::vector<DvView<word>> bx_only{res.BxView()};
  context->elem_handler_.Add(bx_only, np,
                             {res.BxConstView()},
                             {b_ntt.ConstView(0)});
}

template class MlweCiphertext<uint32_t>;
template class MlweCiphertext<uint64_t>;
template class MlweHandler<uint32_t>;
template class MlweHandler<uint64_t>;

}  // namespace cheddar
