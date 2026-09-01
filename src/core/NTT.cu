#include <string>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"
#include "common/PtrList.h"
#include "core/NTT.h"
#include "core/NTTUtils.cuh"

namespace {
// https://artificial-mind.net/blog/2020/10/31/constexpr-for
template <int Start, int End, int Inc = 1, class Func>
constexpr void constexpr_for(Func &&func) {
  if constexpr (Start < End) {
    func(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(std::forward<Func>(func));
  }
}
}  // namespace

namespace cheddar {
namespace kernel {
template <typename word, int log_degree>
__global__ void INTTPhase1(make_signed_t<word> *dst, const word *primes,
                           const make_signed_t<word> *inv_primes,
                           const word *twiddle_factors,
                           const word *twiddle_factors_msb, int tw_y_extra,
                           int num_q_primes,
                           const InputPtrList<make_signed_t<word>, 1> src,
                           int src_batch_stride = 0,
                           int dst_batch_stride = 0) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::INTT, Phase::Phase1>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();
  int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  // Indexing preparation
  int y_idx = blockIdx.y;
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb = src.ptrs_[0] + blockIdx.z * src_batch_stride + (y_idx << log_degree);
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
  }
  signed_word *dst_limb = dst + blockIdx.z * dst_batch_stride + (y_idx << log_degree);

  const word *w = twiddle_factors + (tw_y_idx << log_degree);
  const word *w_msb = twiddle_factors_msb + (tw_y_idx * kMsbSize);

  // Load first input
  signed_word local[kPerThreadElems];
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const signed_word *load_ptr = src_limb + (x_idx << kStageMerging);
  basic::VectorizedMove<signed_word, kPerThreadElems>(local, load_ptr);

  // INTT main
  int tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
  int sm_log_stride = 0;
  int sm_idx = batch_idx << kStageMerging;

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = 0; i < num_main_iters; i++) {
    if (i == 0) {
      if constexpr (kOFTwiddle) {
        MultiRadixINTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixINTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                             prime, inv_prime);
      }
    } else {
      if constexpr (kOFTwiddle & !kExtendedOT) {
        MultiRadixINTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixINTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                             prime, inv_prime);
      }
    }

    // Store the results in shared memory and synchronize
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();

    // Adjust indices and strides for the next iteration
    if (i == num_main_iters - 1) {
      tw_idx >>= kTailStages;
      sm_log_stride += kTailStages;
    } else {
      tw_idx >>= kStageMerging;
      sm_log_stride += kStageMerging;
    }

    // Reload the data from shared memory
    sm_idx = (batch_idx & ((1 << sm_log_stride) - 1)) +
             ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }
  }
  MultiRadixINTTLast<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);

  int dst_idx = batch_idx + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
                (row_idx << kNumStages);
  for (int j = 0; j < kPerThreadElems; j++) {
    dst_limb[dst_idx + (j << (kNumStages - kStageMerging))] = local[j];
  }
}

template <typename word, int log_degree,
          elem_func_t<word> elem_func = NopFunc<word>>
__global__ void INTTPhase2(
    make_signed_t<word> *dst, const word *primes,
    const make_signed_t<word> *inv_primes, const word *twiddle_factors,
    int tw_y_extra, int num_q_primes,
    const InputPtrList<make_signed_t<word>, 1> src,
    const InputPtrList<word, 1> src_const = InputPtrList<word, 1>(),
    int src_batch_stride = 0, int dst_batch_stride = 0) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::INTT, Phase::Phase2>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  // constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  // We do not use OF-Twiddle in this phase
  // constexpr int kOFTwiddle = false;
  // We use batching
  constexpr int kLogWarpBatching = Config::LogWarpBatching();

  // Indexing preparation
  // int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y_idx = blockIdx.y;
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  int src_const_idx = y_idx;
  const signed_word *src_limb = src.ptrs_[0] + blockIdx.z * src_batch_stride + (y_idx << log_degree);
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
    src_const_idx += src_const.extra_;
  }
  word src_const_value =
      basic::StreamingLoadConst(src_const.ptrs_[0] + src_const_idx);
  signed_word *dst_limb = dst + blockIdx.z * dst_batch_stride + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);

  // Load first input
  signed_word local[kPerThreadElems];
  constexpr int initial_log_stride = (log_degree - kNumStages);
  int stage_group_idx = threadIdx.x >> kLogWarpBatching;
  int batch_idx = threadIdx.x & ((1 << kLogWarpBatching) - 1);
  const signed_word *load_ptr =
      src_limb + (stage_group_idx << (initial_log_stride + kStageMerging)) +
      batch_idx + (blockIdx.x << kLogWarpBatching);
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad(load_ptr + (i << initial_log_stride));
  }

  int tw_idx = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
  int sm_log_stride = kLogWarpBatching;
  int sm_idx =
      (threadIdx.x & ((1 << sm_log_stride) - 1)) +
      ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = 0; i < num_main_iters; i++) {
    MultiRadixINTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                         prime, inv_prime);

    // Store the results in shared memory and synchronize
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();

    // Adjust indices and strides for the next iteration
    if (i == num_main_iters - 1) {
      tw_idx >>= kTailStages;
      sm_log_stride += kTailStages;
    } else {
      tw_idx >>= kStageMerging;
      sm_log_stride += kStageMerging;
    }

    // Reload the data from shared memory
    sm_idx =
        (threadIdx.x & ((1 << sm_log_stride) - 1)) +
        ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }
  }
  MultiRadixINTTLast<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);

  int dst_idx = batch_idx + (stage_group_idx << initial_log_stride) +
                (blockIdx.x << kLogWarpBatching);

  for (int j = 0; j < kPerThreadElems; j++) {
    elem_func(local[j], local[j], src_const_value, prime, inv_prime);
    dst_limb[dst_idx + (j << (log_degree - kStageMerging))] = local[j];
  }
}

// ----- Conjugate-invariant ring: the fold and its inverse ----- //
//
// The conjugate-invariant ring R+ = Z[Y + Y^-1] of rank N sits inside the
// 4N-th cyclotomic Z[Y]/(Y^2N + 1) as the elements fixed by Y -> Y^-1. Its
// basis is {1, c_1, ..., c_{N-1}} with c_j = Y^j + Y^-j, and because
// Y^-j = -Y^(2N-j) the lift of a coefficient vector `a` into the negacyclic
// ring of degree 2N is *antisymmetric*: hat_j = a_j, hat_N = 0,
// hat_(2N-j) = -a_j.
//
// That lift is where the halving comes from. A negacyclic transform of length
// 2N would evaluate at all 2N roots of Y^2N + 1, but an antisymmetric input
// makes the outputs equal in mirrored pairs, so only N of them carry anything.
// Splitting Z[Y]/(Y^2N + 1) = Z[Y]/(Y^N - i) x Z[Y]/(Y^N + i) at the first
// butterfly stage, the antisymmetry sends the second factor to a reindexing of
// the first -- f_-[j] = i * f_+[N - j] -- so all the information is in
//
//     f_+[j] = hat_j - i * hat_(N-j) = (a mod (Y^N - i))[j],   i = psi4^N
//
// with psi4 a primitive 4N-th root of unity. Reduction mod (Y^N - i) is a ring
// homomorphism out of the negacyclic ring and the lift is one into it, so the
// fold is one too: pointwise multiplication downstream is multiplication in
// R+.
//
// What remains is to evaluate f_+ at the N roots of Y^N - i, which are
// psi4^(1+4t). The butterfly network below evaluates at psi^(1+2t) for psi a
// *2N*-th root, and psi = psi4^2 turns one into the other:
//
//     f_+(psi4^(1+4t)) = sum_j (f_+[j] psi4^-j) psi^((1+2t) j)
//
// so the fold carries a per-index twist by psi4^-j and the network is used
// exactly as it is. Changing the network root instead does not work: its table
// is psi^brv(j), which satisfies the butterfly recursion only while
// psi^N = -1. With a 4N-th root only the first stage would be right.
template <typename word, int log_degree>
__global__ void CiFoldKernel(make_signed_t<word> *dst, const word *primes,
                             const make_signed_t<word> *inv_primes,
                             const word *i_units, const word *fwd_twist,
                             int tw_y_extra, int num_q_primes, int skip_start,
                             int skip_end, int batch_stride,
                             const InputPtrList<make_signed_t<word>, 1> src) {
  using signed_word = make_signed_t<word>;
  constexpr int kDegree = 1 << log_degree;

  int y_idx = blockIdx.y;
  if (y_idx >= skip_start) y_idx += (skip_end - skip_start);
  const int batch_offset = blockIdx.z * batch_stride;

  const word prime = basic::StreamingLoadConst(primes + y_idx);
  const signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb =
      src.ptrs_[0] + batch_offset + (y_idx << log_degree);
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
  }
  signed_word *dst_limb = dst + batch_offset + (y_idx << log_degree);
  const word i_unit = basic::StreamingLoadConst(i_units + tw_y_idx);
  const word *twist = fwd_twist + (tw_y_idx << log_degree);

  // One thread owns the mirrored pair (t, kDegree - t) and writes both, which
  // is what makes this safe in place -- and it has to be its own pass for the
  // same reason. It cannot ride NTT phase 1: a phase-1 block writes only the
  // indices it read, which is why Cheddar can run the transform in place,
  // whereas the mirror of an index belongs to a *different* block. Fusing the
  // fold there reintroduces exactly that cross-block read-after-write, and
  // with few enough blocks the schedule is fixed, so it corrupts
  // deterministically rather than flakily.
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t == 0) {
    constexpr int kHalf = kDegree / 2;
    const word head = static_cast<word>(basic::StreamingLoad(src_limb));
    const word mid = static_cast<word>(basic::StreamingLoad(src_limb + kHalf));
    // hat_N is zero and psi4^0 is one, so index 0 passes straight through.
    dst_limb[0] = static_cast<signed_word>(head);
    const word mid_folded = basic::Sub<word>(
        mid, basic::MultMontgomery<word>(i_unit, mid, prime, inv_prime), prime);
    dst_limb[kHalf] = static_cast<signed_word>(basic::MultMontgomery<word>(
        mid_folded, basic::StreamingLoadConst(twist + kHalf), prime,
        inv_prime));
    return;
  }
  const int mirror_idx = kDegree - t;
  const word head = static_cast<word>(basic::StreamingLoad(src_limb + t));
  const word mirror =
      static_cast<word>(basic::StreamingLoad(src_limb + mirror_idx));
  const word head_folded = basic::Sub<word>(
      head, basic::MultMontgomery<word>(i_unit, mirror, prime, inv_prime),
      prime);
  const word mirror_folded = basic::Sub<word>(
      mirror, basic::MultMontgomery<word>(i_unit, head, prime, inv_prime),
      prime);
  dst_limb[t] = static_cast<signed_word>(basic::MultMontgomery<word>(
      head_folded, basic::StreamingLoadConst(twist + t), prime, inv_prime));
  dst_limb[mirror_idx] = static_cast<signed_word>(basic::MultMontgomery<word>(
      mirror_folded, basic::StreamingLoadConst(twist + mirror_idx), prime,
      inv_prime));
}

// Undo the twist, then a_j = (g[j] + i * g[N - j]) / 2 for j > 0 and
// a_0 = g[0]. Mirrored in pairs exactly as the fold is, and for the same
// reason: this one always runs in place, on the INTT own output.
template <typename word, int log_degree, bool normalized>
__global__ void CiUnfoldKernel(make_signed_t<word> *dst, const word *primes,
                               const make_signed_t<word> *inv_primes,
                               const word *i_units, const word *inv2_units,
                               const word *inv_twist, int tw_y_extra,
                               int num_q_primes, int batch_stride) {
  using signed_word = make_signed_t<word>;
  constexpr int kDegree = 1 << log_degree;

  const int y_idx = blockIdx.y;
  const int batch_offset = blockIdx.z * batch_stride;

  const word prime = basic::StreamingLoadConst(primes + y_idx);
  const signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  if (y_idx >= num_q_primes) tw_y_idx += tw_y_extra;

  signed_word *limb = dst + batch_offset + (y_idx << log_degree);
  const word i_unit = basic::StreamingLoadConst(i_units + tw_y_idx);
  const word inv2 = basic::StreamingLoadConst(inv2_units + tw_y_idx);
  const word *twist = inv_twist + (tw_y_idx << log_degree);

  // INTT phase 2 either leaves the limb in [0, prime) or, with
  // MultConstNormalize, centred on zero -- and ModUp and ModDown both take the
  // centred one, because that is what the base conversion downstream reads.
  // The recombination below is modular either way; only the representative
  // changes, so it is lifted on the way in and put back on the way out.
  auto load = [&](int idx) -> word {
    const signed_word raw = basic::StreamingLoad(limb + idx);
    if constexpr (normalized) {
      return static_cast<word>(raw < 0 ? raw + static_cast<signed_word>(prime)
                                       : raw);
    } else {
      return static_cast<word>(raw);
    }
  };
  auto store = [&](int idx, const word value) {
    if constexpr (normalized) {
      limb[idx] = basic::Normalize<word>(value, prime);
    } else {
      limb[idx] = static_cast<signed_word>(value);
    }
  };

  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t == 0) {
    constexpr int kHalf = kDegree / 2;
    const word mid = basic::MultMontgomery<word>(
        load(kHalf), basic::StreamingLoadConst(twist + kHalf), prime,
        inv_prime);
    store(kHalf, basic::MultMontgomery<word>(
                     basic::Add<word>(mid,
                                      basic::MultMontgomery<word>(
                                          i_unit, mid, prime, inv_prime),
                                      prime),
                     inv2, prime, inv_prime));
    return;  // a_0 = g[0] and psi4^0 is one, so index 0 is left alone
  }
  const int mirror_idx = kDegree - t;
  const word head = basic::MultMontgomery<word>(
      load(t), basic::StreamingLoadConst(twist + t), prime, inv_prime);
  const word mirror = basic::MultMontgomery<word>(
      load(mirror_idx), basic::StreamingLoadConst(twist + mirror_idx), prime,
      inv_prime);

  store(t, basic::MultMontgomery<word>(
               basic::Add<word>(
                   head,
                   basic::MultMontgomery<word>(i_unit, mirror, prime,
                                               inv_prime),
                   prime),
               inv2, prime, inv_prime));
  store(mirror_idx,
        basic::MultMontgomery<word>(
            basic::Add<word>(
                mirror,
                basic::MultMontgomery<word>(i_unit, head, prime, inv_prime),
                prime),
            inv2, prime, inv_prime));
}

template <typename word, int log_degree>
__global__ void NTTPhase1(
    make_signed_t<word> *dst, const word *primes,
    const make_signed_t<word> *inv_primes, const word *twiddle_factors,
    int tw_y_extra, int num_q_primes, int skip_start, int skip_end,
    int batch_stride, const InputPtrList<make_signed_t<word>, 1> src,
    const InputPtrList<word, 1> src_const = InputPtrList<word, 1>()) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase1>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  // constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  // We do not use OF-Twiddle in this phase
  // constexpr int kOFTwiddle = false;
  // We use batching
  constexpr int kLogWarpBatching = Config::LogWarpBatching();

  // Indexing preparation
  int y_idx = blockIdx.y;
  if (y_idx >= skip_start) {
    y_idx += (skip_end - skip_start);
  }
  // Several transforms over the same prime set in one launch: blockIdx.z picks
  // which, and everything else -- primes, twiddles, constants -- is shared.
  // batch_stride is 0 for a single transform, which is the whole of the
  // difference between the two cases.
  const int batch_offset = blockIdx.z * batch_stride;
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb =
      src.ptrs_[0] + batch_offset + (y_idx << log_degree);
  int src_const_idx = y_idx;
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
    src_const_idx += src_const.extra_;
  }
  signed_word *dst_limb = dst + batch_offset + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);

  // Load first input
  signed_word local[kPerThreadElems];
  int stage_group_idx = threadIdx.x >> kLogWarpBatching;
  int batch_idx = threadIdx.x & ((1 << kLogWarpBatching) - 1);
  const signed_word *load_ptr = src_limb + batch_idx +
                                (blockIdx.x << kLogWarpBatching) +
                                (stage_group_idx << (log_degree - kNumStages));
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad<signed_word>(
        load_ptr + (i << (log_degree - kStageMerging)));
  }

  if (src_const.ptrs_[0] != nullptr) {
    const word src_const_value =
        basic::StreamingLoadConst(src_const.ptrs_[0] + src_const_idx);
    for (int i = 0; i < kPerThreadElems; i++) {
      MultConstLazy<word>(local[i], local[i], src_const_value, prime,
                          inv_prime);
    }
  }

  int final_tw_idx = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
  int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = kNumStages - kStageMerging + kLogWarpBatching;

  // First stage
  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++) {
    temp[threadIdx.x + (j << sm_log_stride)] = local[j];
  }
  __syncthreads();
  sm_log_stride -= kTailStages;

  // Subsequent stages
  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (threadIdx.x & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }

    int tw_idx = final_tw_idx >> (kStageMerging * i);
    MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w, prime,
                                                        inv_prime);
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  int dst_idx =
      batch_idx +
      (stage_group_idx << ((log_degree - kNumStages) + kStageMerging)) +
      (blockIdx.x << kLogWarpBatching);

  for (int i = 0; i < kPerThreadElems; i++) {
    dst_limb[dst_idx + (i << (log_degree - kNumStages))] = local[i];
  }
}

template <typename word, int log_degree>
__global__ void NTTPhase2(make_signed_t<word> *dst, const word *primes,
                          const make_signed_t<word> *inv_primes,
                          const word *twiddle_factors,
                          const word *twiddle_factors_msb, int tw_y_extra,
                          int num_q_primes, int skip_start, int skip_end,
                          int batch_stride,
                          const InputPtrList<make_signed_t<word>, 1> src) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase2>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();
  int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  // Indexing preparation
  int y_idx = blockIdx.y;
  if (y_idx >= skip_start) {
    y_idx += (skip_end - skip_start);
  }
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  // See NTTPhase1: blockIdx.z picks which of the batched transforms this is.
  const int batch_offset = blockIdx.z * batch_stride;
  const signed_word *src_limb =
      src.ptrs_[0] + batch_offset + (y_idx << log_degree);
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
  }
  signed_word *dst_limb = dst + batch_offset + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);
  const word *w_msb = twiddle_factors_msb + (tw_y_idx * kMsbSize);

  // Load first input
  signed_word local[kPerThreadElems];
  int log_stride = kNumStages - kStageMerging;
  const signed_word *load_ptr =
      src_limb + batch_idx + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
      (row_idx << kNumStages);
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad(load_ptr + (i << log_stride));
  }

  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int final_tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
  int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = log_stride;

  // First stage
  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++) {
    temp[batch_idx + (j << sm_log_stride)] = local[j];
  }
  __syncthreads();
  sm_log_stride -= kTailStages;

  // Subsequent stages
  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (batch_idx & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }

    int tw_idx = final_tw_idx >> (kStageMerging * i);
    if (i == 0) {
      // last phase
      if constexpr (kOFTwiddle) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    } else {
      if constexpr (kOFTwiddle & !kExtendedOT) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    }
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  // Lazy normalization
  for (int i = 0; i < kPerThreadElems; i++) {
    if (local[i] < 0) {
      local[i] += prime;
    }
  }

  signed_word *dst_ptr = dst_limb + (x_idx << kStageMerging);
  basic::VectorizedMove<signed_word, kPerThreadElems>(dst_ptr, local);
}

// We can safely assume that skip_start = 0, skip_end = 0, and all extra = 0
template <typename word, int log_degree>
__global__ void NTTPhase2ForModDown(
    make_signed_t<word> *dst, const word *primes,
    const make_signed_t<word> *inv_primes, const word *twiddle_factors,
    const word *twiddle_factors_msb, int src2_start, int src2_end,
    const make_signed_t<word> *src, const make_signed_t<word> *src2,
    const word *inv_p_prod, const word *src2_padding = nullptr,
    int batch_stride = 0, int src2_batch_stride = 0) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);
  signed_word *temp_orig = temp;

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase2>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();
  int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  // Indexing preparation
  int y_idx = blockIdx.y;
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb = src + blockIdx.z * batch_stride + (y_idx << log_degree);
  signed_word *dst_limb = dst + blockIdx.z * batch_stride + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);
  const word *w_msb = twiddle_factors_msb + (tw_y_idx * kMsbSize);

  // Load first input
  signed_word local[kPerThreadElems];
  int log_stride = kNumStages - kStageMerging;
  const signed_word *load_ptr =
      src_limb + batch_idx + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
      (row_idx << kNumStages);
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad(load_ptr + (i << log_stride));
  }

  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int final_tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
  int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = log_stride;

  // First stage
  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++) {
    temp[batch_idx + (j << sm_log_stride)] = local[j];
  }
  __syncthreads();
  sm_log_stride -= kTailStages;

  // Subsequent stages
  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (batch_idx & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }

    int tw_idx = final_tw_idx >> (kStageMerging * i);
    if (i == 0) {
      if constexpr (kOFTwiddle) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    } else {
      if constexpr (kOFTwiddle && !kExtendedOT) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    }
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  // Lazy normalization
  for (int i = 0; i < kPerThreadElems; i++) {
    if (local[i] < 0) {
      local[i] += prime;
    }
  }

  // SSA steps
  basic::VectorizedMove<signed_word, kPerThreadElems>(
      temp + batch_idx * kPerThreadElems, local);
  __syncthreads();

  int src2_y_index = y_idx - src2_start;
  int offset = (src2_y_index << log_degree) +
               (blockIdx.x << (kNumStages + kLogWarpBatching));
  const signed_word *src2_pos = src2 + blockIdx.z * src2_batch_stride + offset;
  signed_word inv_p_prod_val = basic::StreamingLoadConst(inv_p_prod + y_idx);
  signed_word *dst_pos =
      dst_limb + (blockIdx.x << (kNumStages + kLogWarpBatching));

  signed_word src2_padding_val = 0;
  bool src2_exists = (src2_y_index >= 0 && y_idx < src2_end);
  if (src2_padding != nullptr && src2_exists) {
    src2_padding_val = basic::StreamingLoadConst(src2_padding + src2_y_index);
  }

  for (int i = threadIdx.x; i < blockDim.x * kPerThreadElems; i += blockDim.x) {
    signed_word res = 0;
    if (src2_exists) {
      res = src2_pos[i];
      if (src2_padding != nullptr) {
        res = basic::detail::__mult_montgomery_lazy<word>(res, src2_padding_val,
                                                          prime, inv_prime);
        if (res < 0) {
          res += prime;
        }
      }
    }
    res -= temp_orig[i];
    res = basic::detail::__mult_montgomery_lazy<word>(res, inv_p_prod_val,
                                                      prime, inv_prime);
    if (res < 0) {
      res += prime;
    }
    dst_pos[i] = res;
  }
}

}  // namespace kernel

// ----- template for each functions ------
template <typename word>
void NTTHandler<word>::CiFold(make_signed_t<word> *dst, const word *primes,
                              const make_signed_t<word> *inv_primes,
                              int tw_prime_offset, int tw_y_extra,
                              int num_q_primes, int num_total_primes,
                              int skip_start, int skip_end, int batch_stride,
                              int batch, const make_signed_t<word> *src,
                              int src_extra) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  AssertTrue((param_.degree_ / 2) % ci_block_dim_ == 0,
             "CiFold: degree too small");

  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src;
  src_ptr_list.extra_ = src_extra;

  dim3 grid_dim(param_.degree_ / 2 / ci_block_dim_, num_total_primes, batch);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::CiFoldKernel<word, j><<<grid_dim, ci_block_dim_>>>(
        dst, primes, inv_primes, ci_i_.data() + tw_prime_offset,
        ci_fwd_twist_.data() + tw_prime_offset * param_.degree_, tw_y_extra,
        num_q_primes, skip_start, skip_end, batch_stride, src_ptr_list);
  });
}

template <typename word>
void NTTHandler<word>::CiUnfold(make_signed_t<word> *dst, const word *primes,
                                const make_signed_t<word> *inv_primes,
                                int tw_prime_offset, int tw_y_extra,
                                int num_q_primes, int num_total_primes,
                                int batch_stride, int batch,
                                bool normalized) const {
  int log_degree = param_.log_degree_;
  AssertTrue((param_.degree_ / 2) % ci_block_dim_ == 0,
             "CiUnfold: degree too small");

  dim3 grid_dim(param_.degree_ / 2 / ci_block_dim_, num_total_primes, batch);
  const word *i_ptr = ci_i_.data() + tw_prime_offset;
  const word *inv2_ptr = ci_inv2_.data() + tw_prime_offset;
  const word *twist_ptr =
      ci_inv_twist_.data() + tw_prime_offset * param_.degree_;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    if (normalized) {
      kernel::CiUnfoldKernel<word, j, true><<<grid_dim, ci_block_dim_>>>(
          dst, primes, inv_primes, i_ptr, inv2_ptr, twist_ptr, tw_y_extra,
          num_q_primes, batch_stride);
    } else {
      kernel::CiUnfoldKernel<word, j, false><<<grid_dim, ci_block_dim_>>>(
          dst, primes, inv_primes, i_ptr, inv2_ptr, twist_ptr, tw_y_extra,
          num_q_primes, batch_stride);
    }
  });
}

template <typename word>
void NTTHandler<word>::NTT(DvView<word> &dst, const NPInfo &np,
                           const DvConstView<word> &src,
                           bool montgomery_conversion /*= false*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "NTT: Invalid dst size");

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 0: the conjugate-invariant fold, landing in dst so that phase 1
  // reads a contiguous buffer and the aux offset is spent exactly once.
  if (param_.conjugate_invariant_) {
    CiFold(dst_ptr, primes, inv_primes, ter_left, main_left, num_q_primes,
           num_total_primes, 0, 0, 0, 1, src_ptr, src_ptr_list.extra_);
    src_ptr_list.ptrs_[0] = dst_ptr;
    src_ptr_list.extra_ = 0;
  }

  // Phase 1
  int block_dim = GetBlockDim(NTTType::NTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    if (montgomery_conversion) {
      InputPtrList<word, 1> src_const_ptr_list;
      src_const_ptr_list.ptrs_[0] = montgomery_converter_.data() + ter_left;
      src_const_ptr_list.extra_ = main_left;
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          0, src_ptr_list, src_const_ptr_list);
    } else {
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          0, src_ptr_list);
    }
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Phase 2
  block_dim = GetBlockDim(NTTType::NTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::NTTPhase2<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, 0, 0, 0, src_ptr_list);
  });
}

template <typename word>
void NTTHandler<word>::INTT(DvView<word> &dst, const NPInfo &np,
                            const DvConstView<word> &src,
                            bool montgomery_conversion /*= true*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "INTT: Invalid dst size");

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = inv_twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      inv_twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 1
  int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, src_ptr_list);
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Preparing src_const
  InputPtrList<word, 1> src_const_ptr_list;
  if (montgomery_conversion) {
    src_const_ptr_list.ptrs_[0] = inv_degree_.data() + ter_left;
  } else {
    src_const_ptr_list.ptrs_[0] = inv_degree_mont_.data() + ter_left;
  }
  src_const_ptr_list.extra_ = main_left;

  // Phase 2
  block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::INTTPhase2<word, j, kernel::MultConst<word>>
        <<<grid_dim, block_dim, shared_mem_size>>>(
            dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
            src_ptr_list, src_const_ptr_list);
  });

  // Phase 3: undo the fold. It commutes with the per-limb constant phase 2
  // applies, so it can sit after it.
  if (param_.conjugate_invariant_) {
    CiUnfold(dst_ptr, primes, inv_primes, ter_left, main_left, num_q_primes,
             num_total_primes, 0, 1, /*normalized=*/false);
  }
}

template <typename word>
void NTTHandler<word>::INTTAndMultConst(DvView<word> &dst, const NPInfo &np,
                                        const DvConstView<word> &src,
                                        const DvConstView<word> &src_const,
                                        bool normalize /*= false*/, int batch /*= 1*/,
                                        int src_batch_stride /*= 0*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  AssertTrue(batch >= 1, "INTTAndMultConst: invalid batch");
  const int dst_batch_stride = (batch == 1) ? 0 : num_total_primes * param_.degree_;
  AssertTrue(dst.TotalSize() == batch * num_total_primes * param_.degree_,
             "INTTForModUp: Invalid dst size");

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;
  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = inv_twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      inv_twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 1
  int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes, batch);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, src_ptr_list, src_batch_stride, dst_batch_stride);
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Preparing src_const
  InputPtrList<word, 1> src_const_ptr_list(src_const);
  src_const_ptr_list.extra_ = src_const.QSize() - num_q_primes;

  // Phase 2
  block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes, batch);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    if (normalize) {
      kernel::INTTPhase2<word, j, kernel::MultConstNormalize<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
              src_ptr_list, src_const_ptr_list, dst_batch_stride, dst_batch_stride);
    } else {
      kernel::INTTPhase2<word, j, kernel::MultConst<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
              src_ptr_list, src_const_ptr_list, dst_batch_stride, dst_batch_stride);
    }
  });

  // Phase 3: undo the fold, in whichever representative phase 2 left behind.
  if (param_.conjugate_invariant_) {
    CiUnfold(dst_ptr, primes, inv_primes, ter_left, main_left, num_q_primes,
             num_total_primes, dst_batch_stride, batch, normalize);
  }
}

template <typename word>
void NTTHandler<word>::NTTForModUp(DvView<word> &dst, const NPInfo &np,
                                   int skip_start, int skip_end,
                                   const DvConstView<word> &src,
                                   int batch /*= 1*/,
                                   bool ci_prefolded /*= false*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  // `batch` transforms over the same prime set, laid out back to back in one
  // buffer. blockIdx.z picks between them, so they share one launch, one set of
  // twiddles, and a grid `batch` times larger -- which for the fifteen limbs a
  // key switch raises is the difference between leaving the card idle and not.
  AssertTrue(batch >= 1, "NTTForModUp: invalid batch");
  const int batch_stride = (batch == 1) ? 0 : num_total_primes * param_.degree_;
  AssertTrue(dst.TotalSize() == batch * num_total_primes * param_.degree_,
             "NTTForModUp: Invalid dst size");
  AssertTrue(batch == 1 || (src.AuxSize() == dst.AuxSize() &&
                            src.QSize() == dst.QSize()),
             "NTTForModUp: a batched transform is in place over one buffer");
  AssertTrue(batch == 1 || !param_.conjugate_invariant_ || skip_start == skip_end,
             "NTTForModUp: a batched transform cannot skip limbs");

  // Extra handling for skip primes
  AssertTrue(skip_start >= 0 && skip_start < num_q_primes &&
                 skip_end >= skip_start && skip_end <= num_q_primes,
             "NTTForModUp: Invalid skip primes");
  num_total_primes -= (skip_end - skip_start);

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 0, on the limbs this transform touches: the skipped ones arrived in
  // the evaluation domain already and must not be folded. A caller whose base
  // conversion already folded on the way out says so and skips the pass.
  if (param_.conjugate_invariant_ && !ci_prefolded) {
    CiFold(dst_ptr, primes, inv_primes, ter_left, main_left, num_q_primes,
           num_total_primes, skip_start, skip_end, batch_stride, batch,
           src_ptr, src_ptr_list.extra_);
    src_ptr_list.ptrs_[0] = dst_ptr;
    src_ptr_list.extra_ = 0;
  }

  // Phase 1
  int block_dim = GetBlockDim(NTTType::NTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes, batch);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    // montgomery_conversion is always false
    if constexpr (kFuseMontgomery) {
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
          skip_start, skip_end, batch_stride, src_ptr_list);
    } else {
      InputPtrList<word, 1> src_const_ptr_list;
      src_const_ptr_list.ptrs_[0] = montgomery_converter_.data() + ter_left;
      src_const_ptr_list.extra_ = main_left;
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
          skip_start, skip_end, batch_stride, src_ptr_list, src_const_ptr_list);
    }
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Phase 2
  block_dim = GetBlockDim(NTTType::NTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase2);
  grid_dim = dim3(param_.degree_ / (1 << stage_merging) / block_dim,
                  num_total_primes, batch);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::NTTPhase2<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, skip_start, skip_end, batch_stride, src_ptr_list);
  });
}

template <typename word>
void NTTHandler<word>::NTTForModDown(
    DvView<word> &dst, const NPInfo &np_src1, const NPInfo &np_src2,
    const DvConstView<word> &src1, const DvConstView<word> &src2,
    const DvConstView<word> &inv_p_prod,
    const DvConstView<word> &src2_padding /*= DvConstView<word>(nullptr, 0)*/,
    bool ci_prefolded /*= false*/, int batch /*= 1*/, int batch_stride /*= 0*/,
    int src2_batch_stride /*= 0*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np_src1.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np_src1.GetNumTotal();
  AssertTrue(batch >= 1, "NTTForModDown: invalid batch");
  AssertTrue(dst.TotalSize() == (batch - 1) * batch_stride + num_total_primes * param_.degree_,
             "NTTForModUp: Invalid dst size");

  // Special restrictions for NTTForModDown
  AssertTrue(np_src1.num_aux_ == 0, "NTTForModDown: num_aux should be 0");

  int num_src2_primes = np_src2.GetNumQ();
  AssertTrue(num_src2_primes <= num_total_primes,
             "NTTForModDown: Invalid src2 size");
  AssertTrue(dst.data() != src2.data(),
             "NTTForModDown: dst and src2 should be different");
  int src2_start = np_src1.num_ter_ - np_src2.num_ter_;
  int src2_end = src2_start + num_src2_primes;
  AssertTrue(src2_end <= num_total_primes, "NTTForModDown: Invalid src2 size");

  const word *primes = param_.GetPrimesPtr(np_src1);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np_src1);
  int ter_left = param_.GetMaxNumTer() - np_src1.num_ter_;
  int main_left = param_.GetMaxNumMain() - np_src1.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src1_ptr = reinterpret_cast<const signed_word *>(src1.data());
  auto src2_ptr = reinterpret_cast<const signed_word *>(src2.data());
  // We do in-place NTT of dst first
  InputPtrList<signed_word, 1> ntt_ptr_list;
  ntt_ptr_list.ptrs_[0] = src1_ptr;
  ntt_ptr_list.extra_ = 0;

  const word *tw_ptr = twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 0. src2 is already in the evaluation domain -- phase 2 subtracts it
  // there -- so only src1 is folded, and not even src1 when the base
  // conversion that produced it folded on the way out.
  if (param_.conjugate_invariant_ && !ci_prefolded) {
    CiFold(dst_ptr, primes, inv_primes, ter_left, main_left, num_q_primes,
           num_total_primes, 0, 0, batch_stride, batch, src1_ptr, ntt_ptr_list.extra_);
    ntt_ptr_list.ptrs_[0] = dst_ptr;
    ntt_ptr_list.extra_ = 0;
  }

  // Phase 1
  int block_dim = GetBlockDim(NTTType::NTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes, batch);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    // montgomery_conversion is always false
    if constexpr (kFuseMontgomery) {
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          batch_stride, ntt_ptr_list);
    } else {
      InputPtrList<word, 1> src_const_ptr_list;
      src_const_ptr_list.ptrs_[0] = montgomery_converter_.data() + ter_left;
      src_const_ptr_list.extra_ = main_left;
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          batch_stride, ntt_ptr_list, src_const_ptr_list);
    }
  });

  // Phase 2
  block_dim = GetBlockDim(NTTType::NTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes, batch);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::NTTPhase2ForModDown<word, j>
        <<<grid_dim, block_dim, shared_mem_size>>>(
            dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, src2_start,
            src2_end, dst_ptr, src2_ptr, inv_p_prod.data(),
            src2_padding.data(), batch_stride, src2_batch_stride);
  });
}

// dst = INTT(src) * const_src
template <typename word>
void NTTHandler<word>::INTTForModDown(
    DvView<word> &dst, const NPInfo &np_src, const NPInfo &np_non_intt,
    const DvConstView<word> &src, const DvConstView<word> &src_const,
    int batch /*= 1*/, int src_batch_stride /*= 0*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_total_primes = np_src.GetNumTotal() - np_non_intt.GetNumTotal();
  AssertTrue(batch >= 1, "INTTForModDown: invalid batch");
  const int dst_batch_stride = (batch == 1) ? 0 : num_total_primes * param_.degree_;
  AssertTrue(dst.TotalSize() == batch * num_total_primes * param_.degree_,
             "INTTForModDown: Invalid dst size");

  // Specific check for INTTForModDown
  AssertTrue(np_src.GetNumQ() * param_.degree_ == src.QSize(),
             "INTTForModDown: Invalid src size");
  AssertTrue(np_src.GetNumTotal() * param_.degree_ == src.TotalSize(),
             "INTTForModDown: Invalid src size");
  AssertTrue(np_non_intt.num_aux_ == 0,
             "INTTForModDown: num_aux should be 0 after moddown");
  AssertTrue(np_non_intt.IsSubsetOf(np_src),
             "INTTForModDown: Invalid np combination");
  AssertTrue(src.data() != dst.data(),
             "INTTForModDown: src and dst should be different");
  AssertTrue(src_const.AuxSize() == np_src.num_aux_,
             "INTTForModDown: Invalid src_const size");
  AssertTrue(
      src_const.TotalSize() == np_src.GetNumTotal() - np_non_intt.GetNumTotal(),
      "INTTForModDown: Invalid src_const size");

  // We either perform INTT on main primes or terminal primes (+ aux primes --
  // optional) and not both.
  // Also, it's possible that we don't perform INTT on any q primes.
  bool intt_on_main = np_src.num_main_ > np_non_intt.num_main_;
  bool intt_on_ter = np_src.num_ter_ > np_non_intt.num_ter_;
  AssertTrue(!intt_on_main || !intt_on_ter,
             "INTTForModDown: Invalid np combination");

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());

  // Preparing src_const
  InputPtrList<word, 1> src_const_ptr_list;
  src_const_ptr_list.ptrs_[0] = src_const.data();
  src_const_ptr_list.extra_ = 0;

  // Case 1: We only perform INTT on the upper part of src
  if (!intt_on_ter) {
    int num_q_primes = np_src.num_main_ - np_non_intt.num_main_;

    const word *primes = param_.GetPrimesPtr(np_src);
    const signed_word *inv_primes = param_.GetInvPrimesPtr(np_src);

    // We ignore lower part
    int num_src_offset_primes = np_src.num_ter_ + np_non_intt.num_main_;
    int num_tw_offset_primes = param_.GetMaxNumTer() + np_non_intt.num_main_;
    primes += num_src_offset_primes;
    inv_primes += num_src_offset_primes;

    int main_left = param_.GetMaxNumMain() - np_src.num_main_;

    InputPtrList<signed_word, 1> src_ptr_list;
    src_ptr_list.ptrs_[0] = src_ptr + num_src_offset_primes * param_.degree_;
    src_ptr_list.extra_ = 0;

    const word *tw_ptr =
        inv_twiddle_factors_.data() + num_tw_offset_primes * param_.degree_;
    const word *tw_msb_ptr =
        inv_twiddle_factors_msb_.data() + num_tw_offset_primes * GetMsbSize();

    // Phase 1
    int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
    int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
    dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                  num_total_primes, batch);
    int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
          num_q_primes, src_ptr_list, src_batch_stride, dst_batch_stride);
    });

    src_ptr_list.ptrs_[0] = dst_ptr;
    src_ptr_list.extra_ = 0;

    // Phase 2
    block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
    stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
    grid_dim = dim3(param_.degree_ / (1 << stage_merging) / block_dim,
                    num_total_primes, batch);
    shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase2<word, j, kernel::MultConstNormalize<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
              src_ptr_list, src_const_ptr_list, dst_batch_stride, dst_batch_stride);
    });

    // Phase 3: undo the fold, on the centred representative
    // MultConstNormalize left. The base conversion this feeds reads the
    // coefficient vector as integers, so it has to see the ring basis and not
    // the folded one. The CI constants take the same prime-axis offset the
    // inverse twiddle table above does.
    if (param_.conjugate_invariant_) {
      CiUnfold(dst_ptr, primes, inv_primes, num_tw_offset_primes, main_left,
               num_q_primes, num_total_primes, dst_batch_stride, batch, /*normalized=*/true);
    }
  } else {  // Case 2. We perform INTT on some of the ter primes and all aux
            // primes
    int num_q_primes = np_src.num_ter_ - np_non_intt.num_ter_;
    int q_size = num_q_primes * param_.degree_;

    const word *primes =
        param_.__GetPrimesPtrModDownWithTerPrimes(np_src, np_non_intt);
    const signed_word *inv_primes =
        param_.__GetInvPrimesPtrModDownWithTerPrimes(np_src, np_non_intt);

    int ter_left = param_.GetMaxNumTer() - np_src.num_ter_;
    int main_left = param_.GetMaxNumMain() - np_src.num_main_;
    int tw_y_extra = param_.GetMaxNumMain() + np_non_intt.num_ter_;

    InputPtrList<signed_word, 1> src_ptr_list;
    src_ptr_list.ptrs_[0] = src_ptr;
    src_ptr_list.extra_ = src.QSize() - q_size;

    const word *tw_ptr =
        inv_twiddle_factors_.data() + ter_left * param_.degree_;
    const word *tw_msb_ptr =
        inv_twiddle_factors_msb_.data() + ter_left * GetMsbSize();

    // Phase 1
    int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
    int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
    dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                  num_total_primes, batch);
    int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, tw_y_extra,
          num_q_primes, src_ptr_list, src_batch_stride, dst_batch_stride);
    });

    src_ptr_list.ptrs_[0] = dst_ptr;
    src_ptr_list.extra_ = 0;

    // Phase 2
    block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
    stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
    grid_dim = dim3(param_.degree_ / (1 << stage_merging) / block_dim,
                    num_total_primes, batch);
    shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase2<word, j, kernel::MultConstNormalize<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, tw_y_extra, num_q_primes,
              src_ptr_list, src_const_ptr_list, dst_batch_stride, dst_batch_stride);
    });

    // Phase 3, as in case 1 -- here the offset is ter_left, which is what this
    // branch hands the twiddle table.
    if (param_.conjugate_invariant_) {
      CiUnfold(dst_ptr, primes, inv_primes, ter_left, tw_y_extra, num_q_primes,
               num_total_primes, dst_batch_stride, batch, /*normalized=*/true);
    }
  }
}

template <typename word>
int NTTHandler<word>::GetLsbSize() const {
  int log_degree = param_.log_degree_;
  int lsb_size = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree)
      lsb_size = NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::LsbSize();
    return;
  });
  return lsb_size;
}

template <typename word>
int NTTHandler<word>::GetMsbSize() const {
  int log_degree = param_.log_degree_;
  int lsb_size = GetLsbSize();
  return (1 << log_degree) / lsb_size;
}

template <typename word>
int NTTHandler<word>::GetLogWarpBatching() const {
  int log_degree = param_.log_degree_;
  int log_warp_batching = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree)
      log_warp_batching =
          NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::LogWarpBatching();
    return;
  });
  return log_warp_batching;
}

template <typename word>
int NTTHandler<word>::GetStageMerging(NTTType type, Phase phase) const {
  int log_degree = param_.log_degree_;
  AssertTrue(log_degree >= min_log_degree_ && log_degree <= max_log_degree_,
             "GetStageMerging: Invalid log_degree");
  int stage_merging = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree) {
      if (type == NTTType::NTT) {
        if (phase == Phase::Phase1) {
          stage_merging =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::StageMerging();
        } else {
          stage_merging =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase2>::StageMerging();
        }
      } else {
        if (phase == Phase::Phase1) {
          stage_merging =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase1>::StageMerging();
        } else {
          stage_merging =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase2>::StageMerging();
        }
      }
    }
    return;
  });
  return stage_merging;
}

template <typename word>
int NTTHandler<word>::GetBlockDim(NTTType type, Phase phase) const {
  int log_degree = param_.log_degree_;
  AssertTrue(log_degree >= min_log_degree_ && log_degree <= max_log_degree_,
             "GetBlockDim: Invalid log_degree");
  int block_dim = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree) {
      if (type == NTTType::NTT) {
        if (phase == Phase::Phase1) {
          block_dim =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::BlockDim();
        } else {
          block_dim =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase2>::BlockDim();
        }
      } else {
        if (phase == Phase::Phase1) {
          block_dim =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase1>::BlockDim();
        } else {
          block_dim =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase2>::BlockDim();
        }
      }
    }
    return;
  });
  return block_dim;
}

template <typename word>
NTTHandler<word>::NTTHandler(const Parameter<word> &param) : param_(param) {
  PopulateTwiddleFactors();
}

template <typename word>
void NTTHandler<word>::PopulateTwiddleFactors() {
  int log_degree = param_.log_degree_;
  AssertTrue(log_degree >= min_log_degree_ && log_degree <= max_log_degree_,
             "NTTHandler: Invalid log_degree");
  int degree = (1 << log_degree);
  NPInfo np = param_.LevelToNP(param_.max_level_, param_.alpha_);
  const auto &primes = param_.GetPrimeVector(np);
  int num_total_primes = np.GetNumTotal();
  int lsb_size = GetLsbSize();
  int msb_size = GetMsbSize();

  Hv h_psi_rev_mont(degree * num_total_primes, 0);
  Hv h_psi_inv_rev_mont(degree * num_total_primes, 0);
  Hv h_N_inv(num_total_primes, 0);
  Hv h_N_inv_mont(num_total_primes, 0);
  Hv h_mont_convert(num_total_primes, 0);

  Hv h_psi_rev_mont_msb(msb_size * num_total_primes, 0);
  Hv h_psi_inv_rev_mont_msb(msb_size * num_total_primes, 0);

  const bool ci = param_.conjugate_invariant_;
  Hv h_ci_i(ci ? num_total_primes : 0, 0);
  Hv h_ci_inv2(ci ? num_total_primes : 0, 0);
  Hv h_ci_fwd_twist(ci ? degree * num_total_primes : 0, 0);
  Hv h_ci_inv_twist(ci ? degree * num_total_primes : 0, 0);

  for (int i = 0; i < num_total_primes; i++) {
    std::vector<word> psi_rev(degree);
    std::vector<word> psi_inv_rev(degree);

    word p = primes[i];
    // The network stays negacyclic and so keeps its 2N-th root. The
    // conjugate-invariant ring needs a 4N-th one, and takes it as psi4 with
    // psi = psi4^2: the extra factor rides the fold as a per-index twist,
    // which costs nothing because the fold already touches every coefficient.
    // Putting psi4 in the table instead would break it -- psi^brv(j) satisfies
    // the butterfly recursion only while psi^degree = -1, and with a 4N-th
    // root only the first stage would come out right.
    word psi;
    if (ci) {
      const word psi4 = primeutil::FindPrimitiveMthRoot(4 * degree, p);
      psi = primeutil::MultMod<word>(psi4, psi4, p);

      // psi4^degree, the square root of -1 that Y^degree reduces to, and 2^-1,
      // which the unfold divides the recombined mirror pair by.
      h_ci_i[i] = primeutil::ToMontgomery<word>(
          primeutil::PowMod<word>(psi4, static_cast<int64_t>(degree), p), p);
      h_ci_inv2[i] =
          primeutil::ToMontgomery<word>(primeutil::InvMod<word>(2, p), p);

      // psi4^-j for the fold and psi4^j for the unfold, in natural order.
      // These are per-coefficient rather than per-butterfly, so they are not
      // bit reversed and cannot be read out of the tables above.
      const word psi4_inv = primeutil::InvMod<word>(psi4, p);
      word fwd = 1;
      word inv = 1;
      for (int j = 0; j < degree; j++) {
        h_ci_fwd_twist[i * degree + j] = primeutil::ToMontgomery<word>(fwd, p);
        h_ci_inv_twist[i * degree + j] = primeutil::ToMontgomery<word>(inv, p);
        fwd = primeutil::MultMod<word>(fwd, psi4_inv, p);
        inv = primeutil::MultMod<word>(inv, psi4, p);
      }
    } else {
      psi = primeutil::FindPrimitiveMthRoot(2 * degree, p);
    }
    word psi_inv = primeutil::InvMod<word>(psi, p);

    h_N_inv[i] = primeutil::InvMod<word>(degree, p);
    h_N_inv_mont[i] = primeutil::ToMontgomery<word>(h_N_inv[i], p);
    h_mont_convert[i] =
        primeutil::ToMontgomery(primeutil::ToMontgomery<word>(1, p), p);

    psi_rev[0] = 1;
    psi_inv_rev[0] = 1;
    for (int j = 1; j < degree; j++) {
      psi_rev[j] = primeutil::MultMod(psi_rev[j - 1], psi, p);
      psi_inv_rev[j] = primeutil::MultMod(psi_inv_rev[j - 1], psi_inv, p);
    }
    BitReverseVector(psi_rev);
    BitReverseVector(psi_inv_rev);

    for (int j = 0; j < degree; j++) {
      h_psi_rev_mont[i * degree + j] =
          primeutil::ToMontgomery<word>(psi_rev[j], p);
      h_psi_inv_rev_mont[i * degree + j] =
          primeutil::ToMontgomery<word>(psi_inv_rev[j], p);
    }

    // OFTwiddle computation
    for (int j = 0; j < msb_size; j++) {
      h_psi_rev_mont_msb[i * msb_size + j] =
          h_psi_rev_mont[i * degree + j * lsb_size];
      h_psi_inv_rev_mont_msb[i * msb_size + j] =
          h_psi_inv_rev_mont[i * degree + j * lsb_size];
    }
  }
  // Copy from host to device
  CopyHostToDevice<word>(twiddle_factors_, h_psi_rev_mont);
  CopyHostToDevice<word>(inv_twiddle_factors_, h_psi_inv_rev_mont);
  CopyHostToDevice<word>(inv_degree_, h_N_inv);
  CopyHostToDevice<word>(inv_degree_mont_, h_N_inv_mont);
  CopyHostToDevice<word>(montgomery_converter_, h_mont_convert);
  CopyHostToDevice<word>(twiddle_factors_msb_, h_psi_rev_mont_msb);
  CopyHostToDevice<word>(inv_twiddle_factors_msb_, h_psi_inv_rev_mont_msb);
  if (ci) {
    CopyHostToDevice<word>(ci_i_, h_ci_i);
    CopyHostToDevice<word>(ci_inv2_, h_ci_inv2);
    CopyHostToDevice<word>(ci_fwd_twist_, h_ci_fwd_twist);
    CopyHostToDevice<word>(ci_inv_twist_, h_ci_inv_twist);
  }
}

template <typename word>
DvConstView<word> NTTHandler<word>::ImaginaryUnitConstView(
    const NPInfo &np) const {
  // twiddle_factors_[1] is psi^(degree/2) after the bit reversal, which is a
  // 4th root of unity only while psi is a 2N-th one. The conjugate-invariant
  // ring has real slots and no imaginary unit to multiply by at all.
  AssertTrue(!param_.conjugate_invariant_,
             "ImaginaryUnitConstView: the conjugate-invariant ring has real "
             "slots and no imaginary unit");
  int ter_offset = (param_.GetMaxNumTer() - np.num_ter_) * param_.degree_;
  int q_size = param_.L_ * param_.degree_ - ter_offset;
  int aux_size = param_.alpha_ * param_.degree_;

  return DvConstView<word>(twiddle_factors_.data() + ter_offset + 1,
                           q_size + aux_size, aux_size);
}

template class NTTHandler<uint32_t>;
template class NTTHandler<uint64_t>;

}  // namespace cheddar