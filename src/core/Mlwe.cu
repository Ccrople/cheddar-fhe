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

  // The decomposition is defined on coefficients, so undo the NTT first. The
  // Montgomery representation is preserved by the inverse transform and by the
  // pure re-indexing below, which is what the product downstream expects.
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

template class MlweCiphertext<uint32_t>;
template class MlweCiphertext<uint64_t>;
template class MlweHandler<uint32_t>;
template class MlweHandler<uint64_t>;

}  // namespace cheddar
