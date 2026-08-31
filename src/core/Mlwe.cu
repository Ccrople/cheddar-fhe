#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "core/Mlwe.h"

namespace cheddar {

namespace kernel {

// res = lo + Y^(N'/2) * hi, on a module component, in the coefficient domain.
//
// Negacyclic in Y, so entry s of the shift reads entry s - N'/2 of `hi` and
// entry s + N'/2 negated when s is in the lower half. Nothing here assumes the
// operand halves are empty; that assumption belongs to the caller who is using
// this to put two payloads in one ciphertext, and stating it here would only
// hide a violation.
//
// grid: (words / block), where words spans limb, module index and coefficient
// alike, because a shift by N'/2 stays inside the N'-run it started in.
template <typename word>
__global__ void ShiftedAdd(word *dst, const word *lo, const word *hi,
                           const word *primes, int small_degree,
                           int runs_per_limb, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int half = small_degree >> 1;
  const int s = idx & (small_degree - 1);
  const int limb = idx / (runs_per_limb * small_degree);
  const word prime = basic::StreamingLoadConst(primes + limb);
  word shifted;
  if (s < half) {
    // Y^(N'/2) sends entry s + N'/2 here, and Y^N' = -1 makes it a negation.
    shifted = basic::Negate(basic::StreamingLoad(hi + idx + half), prime);
  } else {
    shifted = basic::StreamingLoad(hi + idx - half);
  }
  dst[idx] = basic::Add(basic::StreamingLoad(lo + idx), shifted, prime);
}

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
                         int log_rank, int small_degree, int degree,
                         int num_src) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int limb = blockIdx.z;

  const int rank = 1 << log_rank;
  const int i = x & (rank - 1);
  const int s = x >> log_rank;

  const word value =
      (i < num_src) ? basic::StreamingLoad(
                          src_ptrs[i] + (limb * rank + j) * small_degree + s)
                    : word{0};
  dst_ptrs[j][limb * degree + x] = value;
}

// grid: (degree / block, num_total_primes)
template <typename word>
__global__ void ModPackB(word *dst, const word *const *src_ptrs, int log_rank,
                         int small_degree, int degree, int num_src) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;

  const int i = x & ((1 << log_rank) - 1);
  const int s = x >> log_rank;

  const word value =
      (i < num_src)
          ? basic::StreamingLoad(src_ptrs[i] + limb * small_degree + s)
          : word{0};
  dst[limb * degree + x] = value;
}


// ---- conjugate-invariant forms ---------------------------------------------
//
// On R+ = Z[Y + Y^-1] the module structure over the subring is not a stride
// interleave. The rank-N' subring is spanned by {1, c_k, c_2k, ...} with
// c_j = Y^j + Y^-j and k = N/N', and R+ is free over it on
// {1, c_1, ..., c_{k-1}} -- but c_i c_{tk} = c_{tk+i} + c_{tk-i} hits two
// coefficient classes, so the change of basis is a banded map rather than a
// gather (Doing.md 1.5ba). With alpha_i the i-th module component of a (an
// N'-vector over the subring basis), the coefficient identity is
//
//     a[tk + i] = alpha_i[t] + alpha_{k-i}[t+1],      alpha_.[N'] = 0
//
// (the i = 0 class is pure), so the decomposition is the alternating-sign
// suffix-sum scan running down the class pair (i, k-i),
//
//     alpha_i[t] = a[tk + i] - alpha_{k-i}[t+1]
//
// and the recomposition is the two-term sum above. The i = k/2 class is its
// own mirror and the recurrence closes on itself; the kernels need no special
// case for it, because the two accumulators then track the same value and the
// double write is idempotent.

// One thread per (limb, chain): chain 0 is the i = 0 gather, chain p >= 1 the
// zigzag over the class pair (p, rank - p). The scan is sequential in t by
// construction, but the chains are short where it matters -- the PC-MM
// configuration has N' = 256 -- and a chain's loads do not depend on its
// accumulator, so they pipeline ahead of the subtraction chain.
//
// dst_ptrs[i][limb * limb_stride + t] receives component i's t-th subring
// coefficient: the b-part passes the per-ciphertext buffers (stride N'), the
// a-part one workspace cut into rank windows of N' (stride N).
//
// grid: 1D over num_limbs * (rank / 2 + 1) threads
template <typename word>
__global__ void CiModDecompScan(word *const *dst_ptrs, int limb_stride,
                                const word *src, const word *primes, int rank,
                                int small_degree, int degree, int num_limbs) {
  const int num_chains = rank / 2 + 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_limbs * num_chains) return;
  const int limb = tid / num_chains;
  const int chain = tid - limb * num_chains;

  const word *src_limb = src + limb * degree;

  if (chain == 0) {
    word *dst = dst_ptrs[0] + limb * limb_stride;
    for (int t = 0; t < small_degree; t++) {
      dst[t] = basic::StreamingLoad(src_limb + t * rank);
    }
    return;
  }

  const word prime = basic::StreamingLoadConst(primes + limb);
  const int i = chain;
  const int mi = rank - chain;
  word *dst_i = dst_ptrs[i] + limb * limb_stride;
  word *dst_m = dst_ptrs[mi] + limb * limb_stride;

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

// In-place forms of the scan and its inverse on ONE polynomial in the big
// layout `[limb][t * rank + i]` -- what a module-centred ModRaise applies to
// the level-zero representatives before the CRT lift and undoes after it
// (Doing.md 3.5). One thread per (limb, class pair), as CiModDecompScan; the
// pure class i = 0 is already its own coordinate and is left alone. The scan
// writes row t after reading it, the recomposition writes row t after reading
// row t + 1, so both are safe in place; the self-mirror class i = rank/2
// computes both of its writes from reads taken first and writes one value.
//
// grid: 1D over num_limbs * (rank / 2 + 1) threads
template <typename word>
__global__ void CiScanInPlace(word *data, const word *primes, int rank,
                              int small_degree, int degree, int num_limbs) {
  const int num_chains = rank / 2 + 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_limbs * num_chains) return;
  const int limb = tid / num_chains;
  const int chain = tid - limb * num_chains;
  if (chain == 0) return;
  const word prime = basic::StreamingLoadConst(primes + limb);
  word *d = data + limb * degree;
  const int i = chain;
  const int mi = rank - chain;
  word acc_i = 0;
  word acc_m = 0;
  for (int t = small_degree - 1; t >= 0; t--) {
    const word vi = d[t * rank + i];
    const word vm = d[t * rank + mi];
    const word new_i = basic::Sub(vi, acc_m, prime);
    const word new_m = basic::Sub(vm, acc_i, prime);
    d[t * rank + i] = new_i;
    d[t * rank + mi] = new_m;
    acc_i = new_i;
    acc_m = new_m;
  }
}

template <typename word>
__global__ void CiRecomposeInPlace(word *data, const word *primes, int rank,
                                   int small_degree, int degree,
                                   int num_limbs) {
  const int num_chains = rank / 2 + 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_limbs * num_chains) return;
  const int limb = tid / num_chains;
  const int chain = tid - limb * num_chains;
  if (chain == 0) return;
  const word prime = basic::StreamingLoadConst(primes + limb);
  word *d = data + limb * degree;
  const int i = chain;
  const int mi = rank - chain;
  for (int t = 0; t < small_degree; t++) {
    const word xi = d[t * rank + i];
    const word xm = d[t * rank + mi];
    const word xi_next = (t + 1 < small_degree) ? d[(t + 1) * rank + i] : 0;
    const word xm_next = (t + 1 < small_degree) ? d[(t + 1) * rank + mi] : 0;
    const word ri = basic::Add(xi, xm_next, prime);
    const word rm = basic::Add(xm, xi_next, prime);
    d[t * rank + i] = ri;
    d[t * rank + mi] = rm;
  }
}

// The a-part of the i-th output is not a shifted slice of a's components as
// it is on the power basis. With s = sum_j sigma_j c_j, the module components
// of a * s follow from c_i c_j = c_{i+j} + c_{|i-j|} and, once i + j crosses
// k, the re-decomposition c_{k+l} = c_k c_l - c_{k-l}. Collecting terms, the
// coefficient of sigma_j in component l is
//
//     a~_l[j] = alpha_{|l-j|} + [l+j < k] alpha_{l+j}
//               + [j > l] c'_1 alpha_{k-(j-l)} - [l+j > k] alpha_{2k-l-j}
//
// for l, j >= 1; the j = 0 column is alpha_l and the l = 0 row is
// 2 alpha_j + c'_1 alpha_{k-j}. c'_1 = c_k is the subring generator, whose
// multiplication is the symmetric shift (c'_1 q)[t] = q[t-1] + q[t+1] under
// q[-1] = q[1] (since c'_1 c'_1 = c'_2 + 2) and q[N'] = 0 (since c'_{N'} = 0).
//
// grid: (small_degree / block, rank /* j */, num_total_primes), looping over
// l, mirroring ModDecompA.
template <typename word>
__global__ void CiModDecompCombine(word *const *dst_ptrs, const word *alpha,
                                   const word *primes, int rank,
                                   int small_degree) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int limb = blockIdx.z;

  const word prime = basic::StreamingLoadConst(primes + limb);
  const word *comp = alpha + limb * rank * small_degree;

  // (c'_1 * alpha_x)[t]
  auto shifted = [&](int x) -> word {
    const word lo = comp[x * small_degree + (t == 0 ? 1 : t - 1)];
    const word hi =
        (t + 1 < small_degree) ? comp[x * small_degree + t + 1] : word{0};
    return basic::Add(lo, hi, prime);
  };

  for (int l = 0; l < rank; l++) {
    word v;
    if (j == 0) {
      v = comp[l * small_degree + t];
    } else if (l == 0) {
      const word once = comp[j * small_degree + t];
      v = basic::Add(basic::Add(once, once, prime), shifted(rank - j), prime);
    } else {
      const int d = (l > j) ? (l - j) : (j - l);
      v = comp[d * small_degree + t];
      if (l + j < rank) {
        v = basic::Add(v, comp[(l + j) * small_degree + t], prime);
      }
      if (j > l) {
        v = basic::Add(v, shifted(rank - (j - l)), prime);
      }
      if (l + j > rank) {
        v = basic::Sub(v, comp[(2 * rank - l - j) * small_degree + t], prime);
      }
    }
    dst_ptrs[l][(limb * rank + j) * small_degree + t] = v;
  }
}

// The inverse interleaving on R+, for the a-part: A_j = sum_l a~_l[j] c_l,
// which in coefficients is the banded two-term recomposition
//
//     A_j[tk + i] = q_i[t] + q_{k-i}[t+1],      q_x = a~_x[j], q_.[N'] = 0
//
// with the i = 0 class a pure copy. Indexing by the output coefficient keeps
// a warp's stores contiguous, exactly as ModPackA does.
//
// grid: (degree / block, rank /* j */, num_total_primes)
template <typename word>
__global__ void CiModPackA(word *const *dst_ptrs, const word *const *src_ptrs,
                           const word *primes, int log_rank, int small_degree,
                           int degree, int num_src) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  const int limb = blockIdx.z;

  const int rank = 1 << log_rank;
  const int i = x & (rank - 1);
  const int t = x >> log_rank;

  // `num_src` components were handed in; the rest of the rank are known zero
  // and are not stored at all, so an absent index contributes nothing rather
  // than reading past the pointer array.
  word value = (i < num_src)
                   ? basic::StreamingLoad(
                         src_ptrs[i] + (limb * rank + j) * small_degree + t)
                   : word{0};
  if (i != 0 && rank - i < num_src && t + 1 < small_degree) {
    const word prime = basic::StreamingLoadConst(primes + limb);
    const word mirror = basic::StreamingLoad(
        src_ptrs[rank - i] + (limb * rank + j) * small_degree + t + 1);
    value = basic::Add(value, mirror, prime);
  }
  dst_ptrs[j][limb * degree + x] = value;
}

// grid: (degree / block, num_total_primes)
template <typename word>
__global__ void CiModPackB(word *dst, const word *const *src_ptrs,
                           const word *primes, int log_rank, int small_degree,
                           int degree, int num_src) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;

  const int rank = 1 << log_rank;
  const int i = x & (rank - 1);
  const int t = x >> log_rank;

  word value = (i < num_src) ? basic::StreamingLoad(src_ptrs[i] +
                                                    limb * small_degree + t)
                             : word{0};
  if (i != 0 && rank - i < num_src && t + 1 < small_degree) {
    const word prime = basic::StreamingLoadConst(primes + limb);
    const word mirror =
        basic::StreamingLoad(src_ptrs[rank - i] + limb * small_degree + t + 1);
    value = basic::Add(value, mirror, prime);
  }
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
  // The conjugate-invariant projection runs at small_degree = 128 (Doing.md
  // 1.5bh), below the 256-thread block every per-position launch here used
  // to assume. small_degree is a power of two, so when it is the smaller of
  // the two it divides the block dim exactly and the launches simply shrink
  // their block instead of refusing the shape.
  const int block_dim = Min(small_degree, kernel_block_dim_);

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

  if (param_.conjugate_invariant_) {
    // The banded form (see the kernel comments): scan the b-part straight
    // into the per-ciphertext buffers, scan the a-part into a workspace of
    // module components, and combine those into the a~ arrangement.
    constexpr int scan_block_dim = 128;
    const int num_chains = rank / 2 + 1;
    const int scan_grid_dim =
        DivCeil(num_total_primes * num_chains, scan_block_dim);

    kernel::CiModDecompScan<word><<<scan_grid_dim, scan_block_dim>>>(
        d_dst_b.data(), small_degree, b_coeffs.data(), primes, rank,
        small_degree, degree, num_total_primes);

    DeviceVector<word> alpha(num_total_primes * degree);
    HostVector<word *> h_alpha(rank);
    for (int i = 0; i < rank; i++) {
      h_alpha[i] = alpha.data() + i * small_degree;
    }
    DeviceVector<word *> d_alpha(rank);
    CopyHostToDevice(d_alpha, h_alpha);

    kernel::CiModDecompScan<word><<<scan_grid_dim, scan_block_dim>>>(
        d_alpha.data(), degree, a_coeffs.data(), primes, rank, small_degree,
        degree, num_total_primes);

    const dim3 grid_dim(small_degree / block_dim, rank, num_total_primes);
    kernel::CiModDecompCombine<word><<<grid_dim, block_dim>>>(
        d_dst_a.data(), alpha.data(), primes, rank, small_degree);
    return;
  }

  const dim3 grid_dim(small_degree / block_dim, rank, num_total_primes);

  kernel::ModDecompB<word><<<grid_dim, block_dim>>>(
      d_dst_b.data(), b_coeffs.data(), rank, small_degree, degree);
  kernel::ModDecompA<word><<<grid_dim, block_dim>>>(
      d_dst_a.data(), a_coeffs.data(), primes, rank, small_degree, degree);
}

template <typename word>
void MlweHandler<word>::AddShiftedHalf(MlweCiphertext<word> &res,
                                       const MlweCiphertext<word> &lo,
                                       const MlweCiphertext<word> &hi) const {
  const int rank = lo.rank_;
  const int small_degree = lo.degree_;
  AssertTrue(rank > 0 && small_degree >= 2 && IsPowOfTwo(small_degree),
             "AddShiftedHalf: invalid module component shape");
  AssertTrue(hi.rank_ == rank && hi.degree_ == small_degree,
             "AddShiftedHalf: the two components differ in rank or degree");
  AssertTrue(lo.np_ == hi.np_,
             "AddShiftedHalf: the two components differ in NP");
  AssertTrue(lo.scale_ == hi.scale_,
             "AddShiftedHalf: the two components differ in scale");

  const int num_total_primes = lo.np_.GetNumTotal();
  // IN PLACE WHEN THE CALLER ASKS FOR IT. Each merge otherwise allocates two
  // fresh device buffers per module component, and a projection merges `rank`
  // of them per output group -- 7168 allocations for gate alone. The kernel is
  // safe aliased on `lo`: a thread reads only `lo[idx]` and writes only
  // `dst[idx]`, so there is no cross-thread dependency to break.
  if (&res != &lo) {
    res.rank_ = rank;
    res.degree_ = small_degree;
    res.np_ = lo.np_;
    res.scale_ = lo.scale_;
    res.a_.resize(static_cast<size_t>(num_total_primes) * rank * small_degree);
    res.b_.resize(static_cast<size_t>(num_total_primes) * small_degree);
  }
  AssertTrue(&res != &hi,
             "AddShiftedHalf: the shifted operand cannot be the destination -- "
             "a thread reads hi at a different index than it writes");

  const word *primes = param_.GetPrimesPtr(lo.np_);
  auto launch = [&](word *dst, const word *a, const word *b, int runs_per_limb) {
    const int total = num_total_primes * runs_per_limb * small_degree;
    const int block = Min(small_degree, kernel_block_dim_);
    kernel::ShiftedAdd<word><<<DivCeil(total, block), block>>>(
        dst, a, b, primes, small_degree, runs_per_limb, total);
  };
  launch(res.a_.data(), lo.a_.data(), hi.a_.data(), rank);
  launch(res.b_.data(), lo.b_.data(), hi.b_.data(), 1);
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
  // FEWER COMPONENTS THAN THE RANK IS A DECLARATION, NOT A SHORTFALL. A
  // half-density emission on R+ has live output components only below
  // `rank/2` (Doing.md 1.5db: `GatherWeights` sends row `r` to declared
  // channel `BitReverseInt(r, log_rank)`, which is even exactly when
  // `r < rank/2`), so the product need not compute the dead half and the
  // caller need not store it. The recomposition treats an absent component as
  // zero, which is what it is.
  //
  // The `rank` KEY SWITCHES BELOW DO NOT SHRINK WITH IT, and the reason is
  // worth stating because it looks like they should: `keys[j]` switches the
  // j-th SUB-SECRET of the rank-`rank` MLWE ciphertext, not the j-th channel,
  // and with the live components below `rank/2` the mirror term still fills
  // every one of the `rank` big a-parts. What this saves is the product and
  // the weight operand -- half of each -- not the pack.
  const int num_live = static_cast<int>(cts.size());
  AssertTrue(num_live > 0 && num_live <= rank,
             "ModPack: expected at most rank input ciphertexts");
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
  const int key_num_aux = keys.at(0)->GetNP().num_aux_;
  for (const auto *key : keys) {
    AssertTrue(key != nullptr, "ModPack: null switching key");
    // The dense-to-sparse key takes a different mod-switch handler inside
    // Context::MultKey (Context.cpp:568). Excluding it here is what lets the
    // single mod-down below pick its handler by (level, num_aux) rather than
    // having to know which kind of key it was handed.
    AssertTrue(key->GetNP().num_aux_ == key_num_aux,
               "ModPack: switching keys differ in auxiliary prime count");
    AssertTrue(key_num_aux >= 1 && key_num_aux <= param_.alpha_,
               "ModPack: switching keys must be ordinary evaluation keys, on "
               "the full auxiliary basis or a narrower one");
  }

  const int num_total_primes = np.GetNumTotal();
  const int log_rank = Log2Floor(rank);

  // 1. X^k-adic recomposition, still in the coefficient domain. The rank
  //    a-parts go into one buffer rather than one each: the key switches below
  //    take a group at a time and want the group's inputs back to back.
  DeviceVector<word> b_coeffs(num_total_primes * degree);
  const int a_words = num_total_primes * degree;
  DeviceVector<word> a_coeffs(rank * a_words);

  HostVector<word *> h_src_a(num_live), h_src_b(num_live), h_dst_a(rank);
  for (int i = 0; i < num_live; i++) {
    h_src_a[i] = const_cast<word *>(cts[i].a_.data());
    h_src_b[i] = const_cast<word *>(cts[i].b_.data());
  }
  for (int i = 0; i < rank; i++) h_dst_a[i] = a_coeffs.data() + i * a_words;
  DeviceVector<word *> d_src_a(num_live), d_src_b(num_live), d_dst_a(rank);
  CopyHostToDevice(d_src_a, h_src_a);
  CopyHostToDevice(d_src_b, h_src_b);
  CopyHostToDevice(d_dst_a, h_dst_a);

  const dim3 grid_a(degree / kernel_block_dim_, rank, num_total_primes);
  const dim3 grid_b(degree / kernel_block_dim_, num_total_primes);
  if (param_.conjugate_invariant_) {
    const word *primes = param_.GetPrimesPtr(np);
    kernel::CiModPackA<word><<<grid_a, kernel_block_dim_>>>(
        d_dst_a.data(), d_src_a.data(), primes, log_rank, small_degree,
        degree, num_live);
    kernel::CiModPackB<word><<<grid_b, kernel_block_dim_>>>(
        b_coeffs.data(), d_src_b.data(), primes, log_rank, small_degree,
        degree, num_live);
  } else {
    kernel::ModPackA<word><<<grid_a, kernel_block_dim_>>>(
        d_dst_a.data(), d_src_a.data(), log_rank, small_degree, degree,
        num_live);
    kernel::ModPackB<word><<<grid_b, kernel_block_dim_>>>(
        b_coeffs.data(), d_src_b.data(), log_rank, small_degree, degree,
        num_live);
  }

  // 2. Key-switch (A_j, 0) from the j-th embedded secret to the ordinary one,
  //    accumulating before the mod-down. The b-part is zero throughout, so
  //    the p_prod * bx term MultKeyNoModDown folds in contributes nothing and
  //    B can be added once at the end, exactly as the paper writes it.
  // `input` carries no data any more, only the shape: the switches take their
  // coefficients directly, and the b-part term that used to need a zeroed bx_
  // is not computed at all. It is what tells the key switch which level, scale
  // and slot count it is working at.
  Ct input;
  input.RemoveRx();
  input.ModifyNP(np);
  input.SetScale(scale);
  input.SetNumSlots(param_.MaxNumSlots());

  // The switches are raised in groups and multiplied in one launch per group.
  // One at a time costs `rank` products plus `rank - 1` additions, and at
  // 24 ms a call for a rank of 256 that launch structure, not the arithmetic,
  // is most of what ModPack is. `kSwitchChunk` is
  // ElementWiseHandler<word>::max_num_accum_: the number of (key, mod-up)
  // pairs the accumulating kernel takes at once. A narrower auxiliary basis
  // raises beta above one and pushes the group past that, which PAccum still
  // computes correctly -- it splits and recurses -- just in more than one
  // launch.
  constexpr int kSwitchChunk = 8;
  DeviceVector<word> modup_buffer;
  std::vector<std::vector<DvConstView<word>>> modup_views;

  Ct accum;
  for (int base = 0; base < rank; base += kSwitchChunk) {
    const int num_in_chunk = Min(kSwitchChunk, rank - base);
    std::vector<const Evk *> chunk_keys(keys.begin() + base,
                                        keys.begin() + base + num_in_chunk);

    // The coefficients go in as they are, a group at a time. The mod-up used
    // to be handed a transform of each of them and open by undoing it.
    DvConstView<word> chunk_coeffs(
        a_coeffs.data() + base * a_words, num_in_chunk * a_words, 0);
    context->ModUpForKeySwitchBatch(modup_buffer, modup_views, input,
                                    *keys[base], chunk_coeffs, num_in_chunk);

    context->MultKeyAccumNoModDown(accum, modup_views, input, chunk_keys,
                                   base > 0);
  }

  // 3. One mod-down for all k switches, then the recomposed B.
  res.RemoveRx();
  res.ModifyNP(np);
  res.SetScale(scale);
  res.SetNumSlots(param_.MaxNumSlots());

  // The accumulator is in whatever extended basis the keys carry, so the
  // mod-down has to come from the same basis.
  const auto &mod_switcher = context->GetModSwitchHandler(level, key_num_aux);
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

template <typename word>
void MlweHandler<word>::ScanInPlace(DvView<word> &poly, const NPInfo &np,
                                    int small_degree) const {
  AssertTrue(param_.conjugate_invariant_,
             "ScanInPlace: the scan is a conjugate-invariant object");
  const int degree = param_.degree_;
  AssertTrue(small_degree > 0 && IsPowOfTwo(small_degree) &&
                 degree % small_degree == 0 && small_degree < degree,
             "ScanInPlace: bad small_degree");
  const int rank = degree / small_degree;
  const int num_limbs = np.GetNumTotal();
  AssertTrue(poly.TotalSize() == num_limbs * degree,
             "ScanInPlace: the view does not match np");
  constexpr int block_dim = 128;
  const int num_chains = rank / 2 + 1;
  const int grid_dim = DivCeil(num_limbs * num_chains, block_dim);
  kernel::CiScanInPlace<word><<<grid_dim, block_dim>>>(
      poly.data(), param_.GetPrimesPtr(np), rank, small_degree, degree,
      num_limbs);
}

template <typename word>
void MlweHandler<word>::RecomposeInPlace(DvView<word> &poly, const NPInfo &np,
                                         int small_degree) const {
  AssertTrue(param_.conjugate_invariant_,
             "RecomposeInPlace: the recomposition is a conjugate-invariant "
             "object");
  const int degree = param_.degree_;
  AssertTrue(small_degree > 0 && IsPowOfTwo(small_degree) &&
                 degree % small_degree == 0 && small_degree < degree,
             "RecomposeInPlace: bad small_degree");
  const int rank = degree / small_degree;
  const int num_limbs = np.GetNumTotal();
  AssertTrue(poly.TotalSize() == num_limbs * degree,
             "RecomposeInPlace: the view does not match np");
  constexpr int block_dim = 128;
  const int num_chains = rank / 2 + 1;
  const int grid_dim = DivCeil(num_limbs * num_chains, block_dim);
  kernel::CiRecomposeInPlace<word><<<grid_dim, block_dim>>>(
      poly.data(), param_.GetPrimesPtr(np), rank, small_degree, degree,
      num_limbs);
}

template class MlweCiphertext<uint32_t>;
template class MlweCiphertext<uint64_t>;
template class MlweHandler<uint32_t>;
template class MlweHandler<uint64_t>;

}  // namespace cheddar
