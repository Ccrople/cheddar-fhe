#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/DoubleWord.h"
#include "common/PrimeUtils.h"
#include "common/PtrList.h"
#include "core/ElementWise.h"

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

// ----- ElementWise functions ----- //

// The order of arguments is:
// Result PointerList
// Some metadata
// Input CPointerList

// dst = src_1 + src_2 + ... + src_last;
template <typename word, int num_poly, typename... PtrLists>
__global__ void Sum(int log_degree,
                    OutputPtrList<word, num_poly> dst, const word *primes,
                    int num_q_primes, const PtrLists... srcs) {
  static_assert(sizeof...(PtrLists) > 0, "Sum must have at least one source");
  static_assert((std::is_same_v<PtrLists, InputPtrList<word, num_poly>> && ...),
                "Sum must have InputPtrList as the last source");

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);

  word result[num_poly] = {0};
  bool aux_part = (prime_index >= num_q_primes);

  // We can use this lambda fold expression to expand the variadic template for
  // any function
  // int k = 0;
  (
      [&] {
        int src_index = i;
        if (aux_part) {
          src_index += srcs.extra_;
        }
#pragma unroll
        for (int j = 0; j < num_poly; j++) {
          const word src_value =
              basic::StreamingLoad(srcs.ptrs_[j] + src_index);
          result[j] = basic::Add(result[j], src_value, prime);
        }
        // ++k;
      }(),
      ...);

  // Store the result
#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    dst.ptrs_[j][i] = result[j];
  }
}

// dst = src_1 - src_2;
template <typename word, int num_poly>
__global__ void Sub(int log_degree,
                    OutputPtrList<word, num_poly> dst, const word *primes,
                    int num_q_primes, const InputPtrList<word, num_poly> src1,
                    const InputPtrList<word, num_poly> src2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  int src1_index = i;
  int src2_index = i;
  if (prime_index >= num_q_primes) {
    src1_index += src1.extra_;
    src2_index += src2.extra_;
  }

#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    const word src1_value = basic::StreamingLoad(src1.ptrs_[j] + src1_index);
    const word src2_value = basic::StreamingLoad(src2.ptrs_[j] + src2_index);
    dst.ptrs_[j][i] = basic::Sub(src1_value, src2_value, prime);
  }
}

// dst = -src;
template <typename word, int num_poly>
__global__ void Neg(int log_degree,
                    OutputPtrList<word, num_poly> dst, const word *primes,
                    int num_q_primes, const InputPtrList<word, num_poly> src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  int src_index = i;
  if (prime_index >= num_q_primes) {
    src_index += src.extra_;
  }

#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    const word src_value = basic::StreamingLoad(src.ptrs_[j] + src_index);
    dst.ptrs_[j][i] = basic::Negate(src_value, prime);
  }
}

// dst = src_1 * src_2;
template <typename word, int num_poly>
__global__ void Mult(int log_degree,
                     OutputPtrList<word, num_poly> dst, const word *primes,
                     const make_signed_t<word> *inv_primes, int num_q_primes,
                     const InputPtrList<word, num_poly> src1,
                     const InputPtrList<word, num_poly> src2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const make_signed_t<word> inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  int src1_index = i;
  int src2_index = i;
  if (prime_index >= num_q_primes) {
    src1_index += src1.extra_;
    src2_index += src2.extra_;
  }

#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    const word src1_value = basic::StreamingLoad(src1.ptrs_[j] + src1_index);
    const word src2_value = basic::StreamingLoad(src2.ptrs_[j] + src2_index);
    dst.ptrs_[j][i] =
        basic::MultMontgomery(src1_value, src2_value, prime, inv_prime);
  }
}

// dst = src - const_src;
template <typename word, int num_poly>
__global__ void AddConst(int log_degree,
                         OutputPtrList<word, num_poly> dst, const word *primes,
                         int num_q_primes,
                         const InputPtrList<word, num_poly> src,
                         const InputPtrList<word, 1> const_src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  int src_index = i;
  int const_src_index = prime_index;
  if (prime_index >= num_q_primes) {
    src_index += src.extra_;
    const_src_index += const_src.extra_;
  }
  const word const_src_value =
      basic::StreamingLoadConst(const_src.ptrs_[0] + const_src_index);

#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    const word src_value = basic::StreamingLoad(src.ptrs_[j] + src_index);
    dst.ptrs_[j][i] = basic::Add(src_value, const_src_value, prime);
  }
}

// dst = src - const_src;
template <typename word, int num_poly>
__global__ void SubConst(int log_degree,
                         OutputPtrList<word, num_poly> dst, const word *primes,
                         int num_q_primes,
                         const InputPtrList<word, num_poly> src,
                         const InputPtrList<word, 1> const_src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  int src_index = i;
  int const_src_index = prime_index;
  if (prime_index >= num_q_primes) {
    src_index += src.extra_;
    const_src_index += const_src.extra_;
  }
  const word const_src_value =
      basic::StreamingLoadConst(const_src.ptrs_[0] + const_src_index);

#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    const word src_value = basic::StreamingLoad(src.ptrs_[j] + src_index);
    dst.ptrs_[j][i] = basic::Sub(src_value, const_src_value, prime);
  }
}

// dst = const_src - src;
template <typename word, int num_poly>
__global__ void SubOppositeConst(int log_degree,
                                 OutputPtrList<word, num_poly> dst,
                                 const word *primes, int num_q_primes,
                                 const InputPtrList<word, num_poly> src,
                                 const InputPtrList<word, 1> const_src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  int src_index = i;
  int const_src_index = prime_index;
  if (prime_index >= num_q_primes) {
    src_index += src.extra_;
    const_src_index += const_src.extra_;
  }
  const word const_src_value =
      basic::StreamingLoadConst(const_src.ptrs_[0] + const_src_index);

#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    const word src_value = basic::StreamingLoad(src.ptrs_[j] + src_index);
    dst.ptrs_[j][i] = basic::Sub(const_src_value, src_value, prime);
  }
}

// CAccum/PAccum version without src0
// PtrLists: CPAcuumInputPtrList
template <typename word, int num_poly, bool const_accum, typename... PtrLists>
__global__ void CPAccum(int log_degree,
                        OutputPtrList<word, num_poly> dst, const word *primes,
                        const make_signed_t<word> *inv_primes, int num_q_primes,
                        const PtrLists... srcs) {
  static_assert(sizeof...(PtrLists) > 0,
                "CPAccum must have at least one source");
  static_assert(
      (std::is_same_v<PtrLists, CPAccumInputPtrList<word, num_poly>> && ...),
      "CPAccum must have CPAccumInputPtrList as the last source");

  using signed_word = make_signed_t<word>;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);

  word result[num_poly] = {0};
  bool aux_part = (prime_index >= num_q_primes);

  // We can use this lambda fold expression to expand the variadic template
  // for any function
  (
      [&] {
        int src_index = i;
        int common_index;
        // CAccum vs. PAccum
        if constexpr (const_accum) {
          common_index = prime_index;
        } else {
          common_index = i;
        }
        if (aux_part) {
          src_index += srcs.extra_;
          common_index += srcs.common_extra_;
        }
        word common_temp = 0;
        // CAccum vs. PAccum
        if constexpr (const_accum) {
          common_temp =
              basic::StreamingLoadConst(srcs.common_ptr_ + common_index);
        } else {
          common_temp = basic::StreamingLoad(srcs.common_ptr_ + common_index);
        }

#pragma unroll
        for (int j = 0; j < num_poly; j++) {
          const word src_value =
              basic::StreamingLoad(srcs.ptrs_[j] + src_index);
          word mult =
              basic::MultMontgomery(common_temp, src_value, prime, inv_prime);
          result[j] = basic::Add(result[j], mult, prime);
        }
      }(),
      ...);

// Store the result
#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    dst.ptrs_[j][i] = result[j];
  }
}

// CAccum/PAccum version with src0
// PtrLists: CPAccumInputPtrList
template <typename word, int num_poly, bool const_accum, typename... PtrLists>
__global__ void CPAccumAdd(int log_degree, OutputPtrList<word, num_poly> dst,
                           const word *primes,
                           const make_signed_t<word> *inv_primes,
                           int num_q_primes,
                           const InputPtrList<word, num_poly> src0,
                           const PtrLists... srcs) {
  static_assert(sizeof...(PtrLists) > 0,
                "CPAccumAdd must have at least one source");
  static_assert(
      (std::is_same_v<PtrLists, CPAccumInputPtrList<word, num_poly>> && ...),
      "CPAccumAdd must have CPAccumInputPtrList as the last source");

  using signed_word = make_signed_t<word>;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);

  word result[num_poly] = {0};
  bool aux_part = (prime_index >= num_q_primes);

  // Only this part is different from the upper version
  int src0_index = i;
  if (aux_part) {
    src0_index += src0.extra_;
  }
#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    result[j] = basic::StreamingLoad(src0.ptrs_[j] + src0_index);
  }

  // We can use this lambda fold expression to expand the variadic template
  // for any function
  (
      [&] {
        int src_index = i;
        int common_index;
        // CAccum vs. PAccum
        if constexpr (const_accum) {
          common_index = prime_index;
        } else {
          common_index = i;
        }
        if (aux_part) {
          src_index += srcs.extra_;
          common_index += srcs.common_extra_;
        }
        word common_temp = 0;
        // CAccum vs. PAccum
        if constexpr (const_accum) {
          common_temp =
              basic::StreamingLoadConst(srcs.common_ptr_ + common_index);
        } else {
          common_temp = basic::StreamingLoad(srcs.common_ptr_ + common_index);
        }

#pragma unroll
        for (int j = 0; j < num_poly; j++) {
          const word src_value =
              basic::StreamingLoad(srcs.ptrs_[j] + src_index);
          word mult =
              basic::MultMontgomery(common_temp, src_value, prime, inv_prime);
          result[j] = basic::Add(result[j], mult, prime);
        }
      }(),
      ...);

// Store the result
#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    dst.ptrs_[j][i] = result[j];
  }
}

// dst = permute(src1, r1) + permute(src2, r2) + ... + permute(src_last,
// r_last); {src1, r1} embedded in a single PtrList
template <typename word, int num_poly, typename... PtrLists>
__global__ void Permute(int log_degree,
                        OutputPtrList<word, num_poly> dst, int num_q_primes,
                        const PermuteInputPtrList<word, num_poly> src1) {
  int degree = 1 << log_degree;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  uint32_t x_idx = i & (degree - 1);
  uint32_t x_idx_rev = basic::BitReverse(x_idx, log_degree + 1);

  uint32_t gf = src1.galois_factor_;
  int src1_idx =
      (prime_index << log_degree) +
      basic::BitReverse(x_idx_rev * gf + src1.galois_offset_, log_degree + 1);
  if (prime_index >= num_q_primes) {
    src1_idx += src1.extra_;
  }

  for (int j = 0; j < num_poly; j++) {
    const word src1_value = basic::StreamingLoad(src1.ptrs_[j] + src1_idx);
    dst.ptrs_[j][i] = src1_value;
  }
}

// dst = permute(src1, r1) + permute(src2, r2) + ... + permute(src_last,
// r_last); {src1, r1} embedded in a single PtrList
template <typename word, int num_poly, typename... PtrLists>
__global__ void PermuteAccum(int log_degree, OutputPtrList<word, num_poly> dst,
                             const word *primes, int num_q_primes,
                             const PtrLists... srcs) {
  static_assert(sizeof...(PtrLists) > 0,
                "PermuteAccum must have at least two srcs");
  static_assert(
      (std::is_same_v<PtrLists, PermuteInputPtrList<word, num_poly>> && ...),
      "PermuteAccum must have PermuteInputPtrList as the last src");

  int degree = 1 << log_degree;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  uint32_t x_idx = i & (degree - 1);
  uint32_t x_idx_rev = basic::BitReverse(x_idx, log_degree + 1);
  const word prime = basic::StreamingLoadConst(primes + prime_index);

  word result[num_poly] = {0};
  bool aux_part = (prime_index >= num_q_primes);
  int y_offset = prime_index << log_degree;

  // We can use this lambda fold expression to expand the variadic template
  // for any function
  (
      [&] {
        uint32_t gf = srcs.galois_factor_;
        int src_idx = basic::BitReverse(x_idx_rev * gf + srcs.galois_offset_,
                                        log_degree + 1);
        if (aux_part) {
          src_idx += srcs.extra_;
        }
#pragma unroll
        for (int j = 0; j < num_poly; j++) {
          // This is somehow coalesced
          const word src_value =
              basic::StreamingLoad(srcs.ptrs_[j] + src_idx + y_offset);
          result[j] = basic::Add(result[j], src_value, prime);
        }
      }(),
      ...);

// Store the result
#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    dst.ptrs_[j][i] = result[j];
  }
}

// dst = src0 + permute(src1, r1) + permute(src2, r2) + ... + permute(src_last,
// r_last); {src1, r1} embedded in a single PtrList
template <typename word, int num_poly, typename... PtrLists>
__global__ void PermuteAccumAdd(int log_degree,
                                OutputPtrList<word, num_poly> dst,
                                const word *primes, int num_q_primes,
                                const InputPtrList<word, num_poly> src0,
                                const PtrLists... srcs) {
  static_assert(sizeof...(PtrLists) > 0,
                "PermuteAccum must have at least two srcs");
  static_assert(
      (std::is_same_v<PtrLists, PermuteInputPtrList<word, num_poly>> && ...),
      "PermuteAccum must have PermuteInputPtrList as the last src");

  int degree = 1 << log_degree;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  uint32_t x_idx = i & (degree - 1);
  uint32_t x_idx_rev = basic::BitReverse(x_idx, log_degree + 1);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  bool aux_part = (prime_index >= num_q_primes);

  word result[num_poly];
  int src0_index = i;
  if (aux_part) {
    src0_index += src0.extra_;
  }
#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    result[j] = basic::StreamingLoad(src0.ptrs_[j] + src0_index);
  }

  int y_offset = prime_index << log_degree;

  // We can use this lambda fold expression to expand the variadic template
  // for any function
  (
      [&] {
        uint32_t gf = srcs.galois_factor_;
        int src_idx = basic::BitReverse(x_idx_rev * gf + srcs.galois_offset_,
                                        log_degree + 1);
        if (aux_part) {
          src_idx += srcs.extra_;
        }
#pragma unroll
        for (int j = 0; j < num_poly; j++) {
          // This is somehow coalesced
          const word src_value =
              basic::StreamingLoad(srcs.ptrs_[j] + src_idx + y_offset);
          result[j] = basic::Add(result[j], src_value, prime);
        }
      }(),
      ...);

// Store the result
#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    dst.ptrs_[j][i] = result[j];
  }
}

// (dst_bx, dst_ax, dst_rx) = (b1 * b2, b1 * a2 + a1 * b2, a1 * a2);
template <typename word /*, int num_poly = 2*/>
__global__ void Tensor(int log_degree,
                       OutputPtrList<word, 3> dst, const word *primes,
                       const make_signed_t<word> *inv_primes, int num_q_primes,
                       const InputPtrList<word, 2> src1,
                       const InputPtrList<word, 2> src2) {
  using signed_word = make_signed_t<word>;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  int src1_index = i;
  int src2_index = i;
  if (prime_index >= num_q_primes) {
    src1_index += src1.extra_;
    src2_index += src2.extra_;
  }

  // Load everything first
  signed_word b1 = basic::StreamingLoad(src1.ptrs_[0] + src1_index);
  signed_word a1 = basic::StreamingLoad(src1.ptrs_[1] + src1_index);
  signed_word b2 = basic::StreamingLoad(src2.ptrs_[0] + src2_index);
  signed_word a2 = basic::StreamingLoad(src2.ptrs_[1] + src2_index);

  // karatsuba multiplication
  signed_word b1_plus_a1 = (b1 - prime) + a1;  // [-q, q - 2] range
  signed_word b2_plus_a2 = (b2 - prime) + a2;  // [-q, q - 2] range
  // [-q(q-2), q^2] range for the multiplication
  auto a_mult = basic::detail::__mult_wide<signed_word>(b1_plus_a1, b2_plus_a2);
  word new_ax = basic::ReduceMontgomery(a_mult, prime, inv_prime);

  word b1_times_b2 = basic::MultMontgomery<word>(b1, b2, prime, inv_prime);
  word a1_times_a2 = basic::MultMontgomery<word>(a1, a2, prime, inv_prime);
  // [-q^2 + 2, q^2] range
  new_ax = basic::Sub(new_ax, b1_times_b2, prime);
  new_ax = basic::Sub(new_ax, a1_times_a2, prime);

  // bx, ax, rx
  dst.ptrs_[0][i] = b1_times_b2;
  dst.ptrs_[1][i] = new_ax;
  dst.ptrs_[2][i] = a1_times_a2;
}

// dst = src_1 - src_2;
template <typename word /*, int num_poly = 2*/>
__global__ void TensorSquare(int log_degree,
                             OutputPtrList<word, 3> dst, const word *primes,
                             const make_signed_t<word> *inv_primes,
                             int num_q_primes,
                             const InputPtrList<word, 2> src1) {
  using signed_word = make_signed_t<word>;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  int src1_index = i;
  if (prime_index >= num_q_primes) {
    src1_index += src1.extra_;
  }

  // Load everything first
  word b1 = basic::StreamingLoad(src1.ptrs_[0] + src1_index);
  word a1 = basic::StreamingLoad(src1.ptrs_[1] + src1_index);

  // bx, ax, rx
  dst.ptrs_[0][i] = basic::MultMontgomery<word>(b1, b1, prime, inv_prime);
  word b1_times_a1 = basic::MultMontgomery<word>(b1, a1, prime, inv_prime);
  dst.ptrs_[1][i] = basic::Add<word>(b1_times_a1, b1_times_a1, prime);
  dst.ptrs_[2][i] = basic::MultMontgomery<word>(a1, a1, prime, inv_prime);
}

// Special kernels for bootstrapping

template <typename word>
__global__ void ModUpToMax1(int log_degree,
                            word *dst, const word *primes, const word q0,
                            const word *src) {
  int degree = 1 << log_degree;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  int degree_idx = i & (degree - 1);
  const word prime = basic::StreamingLoadConst(primes + prime_index);

  word res = basic::StreamingLoad(src + degree_idx);
  res = res % prime;
  dst[i] = res;
}

template <typename word>
__global__ void ModUpToMax2(int log_degree,
                            word *dst, const word *primes, const word q0,
                            const word q1, const word *src) {
  using signed_word = make_signed_t<word>;
  using signed_d_word = make_signed_double_word_t<word>;

  int degree = 1 << log_degree;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  int degree_idx = i & (degree - 1);
  const word prime = basic::StreamingLoadConst(primes + prime_index);

  signed_d_word q_prod = basic::detail::__mult_wide(q0, q1);
  signed_d_word half_q_prod = (q_prod >> 1);

  word src0 = basic::StreamingLoad(src + degree_idx);
  word src1 = basic::StreamingLoad(src + degree_idx + degree);
  signed_d_word res = basic::detail::__mult_wide(src0, q1) +
                      basic::detail::__mult_wide(src1, q0);

  // res is in range [0, 2 * q0q1 - 1]
  // convert it into (-q0q1 / 2, q0q1 / 2)
  if (res >= q_prod) res -= q_prod;
  if (res > half_q_prod) res -= q_prod;
  auto reduced =
      static_cast<signed_word>(res % static_cast<signed_d_word>(prime));
  if (reduced < 0) reduced += prime;
  dst[i] = reduced;
}

template <typename word, int num_poly>
__global__ void MultImaginaryUnit(int log_degree,
                                  OutputPtrList<word, num_poly> dst,
                                  const word *primes,
                                  const make_signed_t<word> *inv_primes,
                                  int num_q_primes,
                                  const InputPtrList<word, num_poly> src,
                                  const InputPtrList<word, 1> i_unit) {
  int half_degree = 1 << (log_degree - 1);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const word inv_prime = basic::StreamingLoadConst(inv_primes + prime_index);

  int src_index = i;
  int i_unit_index = prime_index << log_degree;
  if (prime_index >= num_q_primes) {
    i_unit_index += i_unit.extra_;
    src_index += src.extra_;
  }

  word psi_n_over_2 =
      basic::StreamingLoadConst(i_unit.ptrs_[0] + (prime_index << log_degree));
  if (i & half_degree) {  // if (degree_idx >= half_degree)
    psi_n_over_2 = prime - psi_n_over_2;
  }

#pragma unroll
  for (int j = 0; j < num_poly; j++) {
    word src_value = basic::StreamingLoad(src.ptrs_[j] + src_index);
    dst.ptrs_[j][i] =
        basic::MultMontgomery(src_value, psi_n_over_2, prime, inv_prime);
  }
}

}  // namespace kernel

template <typename word>
ElementWiseHandler<word>::ElementWiseHandler(const Parameter<word> &param)
    : param_{param} {
  AssertTrue(param_.degree_ % kernel_block_dim_ == 0,
             "Invalid kernel block dim");
}

template <typename word>
void ElementWiseHandler<word>::AssertNPMatch(std::vector<DvView<word>> &dst,
                                             const NPInfo &np) const {
  int num_q_primes = np.num_main_ + np.num_ter_;
  int num_total_primes = np.GetNumTotal();
  for (const auto &dv_view : dst) {
    AssertTrue(dv_view.QSize() == num_q_primes * param_.degree_,
               "QSize mismatch");
    AssertTrue(dv_view.TotalSize() == num_total_primes * param_.degree_,
               "TotalSize mismatch");
  }
}

template <typename word>
void ElementWiseHandler<word>::Add(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const std::vector<DvConstView<word>> &src2) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()) &&
                 num_poly == static_cast<int>(src2.size()),
             "Add: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "Add: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    InputPtrList<word, j> src2_ptr_list(src2);
    src2_ptr_list.extra_ = src2.at(0).QSize() - q_size;

    kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, num_q_primes, src1_ptr_list, src2_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::Sub(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const std::vector<DvConstView<word>> &src2) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()) &&
                 num_poly == static_cast<int>(src2.size()),
             "Sub: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "Sub: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    InputPtrList<word, j> src2_ptr_list(src2);
    src2_ptr_list.extra_ = src2.at(0).QSize() - q_size;

    kernel::Sub<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, num_q_primes, src1_ptr_list, src2_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::Neg(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()),
             "Neg: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "Neg: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;

    kernel::Neg<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, num_q_primes, src1_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::Mult(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const std::vector<DvConstView<word>> &src2) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()) &&
                 num_poly == static_cast<int>(src2.size()),
             "Mult: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "Mult: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    InputPtrList<word, j> src2_ptr_list(src2);
    src2_ptr_list.extra_ = src2.at(0).QSize() - q_size;

    kernel::Mult<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, inv_primes, num_q_primes, src1_ptr_list,
        src2_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::PMult(std::vector<DvView<word>> &dst,
                                     const NPInfo &np,
                                     const std::vector<DvConstView<word>> &src1,
                                     const DvConstView<word> &src2) const {
  CPAccumWorker<false>(dst, np, {src1}, {src2});
}

template <typename word>
void ElementWiseHandler<word>::AddConst(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const DvConstView<word> &src_const) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()),
             "AddConst: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "AddConst: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    InputPtrList<word, 1> src_const_ptr_list(src_const);
    src_const_ptr_list.extra_ = src_const.QSize() - num_q_primes;

    kernel::AddConst<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, num_q_primes, src1_ptr_list, src_const_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::SubConst(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const DvConstView<word> &src_const) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()),
             "SubConst: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "SubConst: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    InputPtrList<word, 1> src_const_ptr_list(src_const);
    src_const_ptr_list.extra_ = src_const.QSize() - num_q_primes;

    kernel::SubConst<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, num_q_primes, src1_ptr_list, src_const_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::SubOppositeConst(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const DvConstView<word> &src_const) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()),
             "SubOppositeConst: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "SubOppositeConst: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    InputPtrList<word, 1> src_const_ptr_list(src_const);
    src_const_ptr_list.extra_ = src_const.QSize() - num_q_primes;

    kernel::SubOppositeConst<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, num_q_primes, src1_ptr_list, src_const_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::MultConst(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const DvConstView<word> &src_const) const {
  CPAccumWorker<true>(dst, np, {src1}, {src_const});
}

template <typename word>
void ElementWiseHandler<word>::Tensor(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const std::vector<DvConstView<word>> &src2) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(dst.size() == 3 && src1.size() == 2 && src2.size() == 2,
             "Tensor: Invalid number of polynomials");
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  OutputPtrList<word, 3> dst_ptr_list(dst);
  InputPtrList<word, 2> src1_ptr_list(src1);
  src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;

  if (src1.at(0).data() == src2.at(0).data() &&
      src1.at(1).data() == src2.at(1).data()) {
    kernel::TensorSquare<word><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, inv_primes, num_q_primes, src1_ptr_list);
  } else {
    InputPtrList<word, 2> src2_ptr_list(src2);
    src2_ptr_list.extra_ = src2.at(0).QSize() - q_size;
    kernel::Tensor<word><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, inv_primes, num_q_primes, src1_ptr_list,
        src2_ptr_list);
  }
}

template <typename word>
uint32_t ElementWiseHandler<word>::PermuteAmountToGaloisFactor(
    int permute_amount) const {
  if (permute_amount == -1) {
    // Conjugation. On the real subring it is the {+-1} that (Z/4N)^* / {+-1}
    // quotients out, so it acts trivially and the automorphism is the identity
    // rather than X -> X^-1.
    if (param_.conjugate_invariant_) return 1;
    return 2 * param_.degree_ - 1;
  }
  AssertTrue(permute_amount >= 0 && permute_amount <= param_.MaxNumSlots(),
             "Permute: Invalid permute amount");
  return param_.GetGaloisFactor(permute_amount);
}

template <typename word>
void ElementWiseHandler<word>::Permute(
    std::vector<DvView<word>> &dst, const NPInfo &np, int permute_amount,
    const std::vector<DvConstView<word>> &src1) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()),
             "Permute: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "Permute: Invalid number of polynomials");
  for (int i = 0; i < num_poly; i++) {
    AssertTrue(dst.at(i).data() != src1.at(i).data(),
               "Permute does not support inplace operation");
  }
  AssertNPMatch(dst, np);

  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;
  int half_degree = param_.degree_ / 2;

  // dim3 grid_dim(param_.degree_ / kernel_block_dim_, np.GetNumTotal());
  auto galois_factor = PermuteAmountToGaloisFactor(permute_amount);
  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    PermuteInputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    src1_ptr_list.galois_factor_ = galois_factor;
    src1_ptr_list.galois_offset_ = param_.GetGaloisOffset(galois_factor);

    kernel::Permute<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, num_q_primes, src1_ptr_list);
  });
}

template <typename word>
void ElementWiseHandler<word>::PermuteAccum(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<int> &permute_amounts,
    const std::vector<std::vector<DvConstView<word>>> &srcs) const {
  if (kOptimizeAutomorphism) {
    PermuteAccumWorker(dst, np, permute_amounts, srcs);
    return;
  }

  // Naive repetition of Permute and add
  int num_poly = dst.size();
  int num_accum = permute_amounts.size();

  bool has_extra_ct = (static_cast<int>(srcs.size()) == num_accum + 1);
  AssertTrue(num_accum == static_cast<int>(srcs.size()) || has_extra_ct,
             "PermuteAccum: Incompatible srcs/permute_amounts size");

  std::vector<DeviceVector<word>> tmp;
  std::vector<DvView<word>> tmp_view;
  std::vector<DvConstView<word>> tmp_const_view;
  std::vector<DvConstView<word>> dst_const_view;
  for (int i = 0; i < num_poly; i++) {
    tmp.emplace_back(dst.at(i).TotalSize());
    tmp_view.push_back(tmp.at(i).View(dst.at(i).AuxSize()));
    tmp_const_view.push_back(tmp.at(i).ConstView(dst.at(i).AuxSize()));
    dst_const_view.emplace_back(dst.at(i));
  }

  // first iteration
  int per = permute_amounts.at(0);
  const std::vector<DvConstView<word>> *prev_accum = &(srcs.at(0));
  if (per != 0) {
    if (has_extra_ct) {
      Permute(tmp_view, np, per, srcs.at(0));
      Add(dst, np, tmp_const_view, srcs.back());
    } else {
      Permute(tmp_view, np, per, srcs.at(0));
    }
    prev_accum = &dst_const_view;
  } else {
    if (has_extra_ct) {
      Add(dst, np, *prev_accum, srcs.back());
      prev_accum = &dst_const_view;
    }
  }
  for (int i = 1; i < num_accum; i++) {
    per = permute_amounts.at(i);
    if (per != 0) {
      Permute(tmp_view, np, per, srcs.at(i));
      Add(dst, np, tmp_const_view, *prev_accum);
    } else {
      Add(dst, np, srcs.at(i), *prev_accum);
    }
    prev_accum = &dst_const_view;
  }
}

template <typename word>
void ElementWiseHandler<word>::PermuteAccumWorker(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<int> &permute_amounts,
    const std::vector<std::vector<DvConstView<word>>> &srcs) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "PermuteAccum: Invalid number of polynomials");

  int num_accum = permute_amounts.size();

  if (num_accum > max_num_accum_) {
    // Prepare temporary result DV
    std::vector<DeviceVector<word>> temp;
    std::vector<DvView<word>> temp_view;
    std::vector<DvConstView<word>> temp_const_view;
    for (int i = 0; i < num_poly; i++) {
      temp.emplace_back(dst.at(i).TotalSize());
      temp_view.push_back(temp.at(i).View(dst.at(i).AuxSize()));
      temp_const_view.push_back(temp.at(i).ConstView(dst.at(i).AuxSize()));
    }

    // Split the srcs/permute_amounts into front and back
    std::vector<std::vector<DvConstView<word>>> srcs_front(
        srcs.begin(), srcs.begin() + max_num_accum_);
    srcs_front.push_back(temp_const_view);
    std::vector<std::vector<DvConstView<word>>> srcs_back(
        srcs.begin() + max_num_accum_, srcs.end());
    std::vector<int> permute_front(permute_amounts.begin(),
                                   permute_amounts.begin() + max_num_accum_);
    std::vector<int> permute_back(permute_amounts.begin() + max_num_accum_,
                                  permute_amounts.end());

    // Recursive call to accumulate the front and back
    PermuteAccum(temp_view, np, permute_back, srcs_back);
    PermuteAccum(dst, np, permute_front, srcs_front);
    return;
  }

  // Some sanity checks
  int num_ct = srcs.size();
  bool has_extra_ct = (num_ct == num_accum + 1);

  AssertTrue(num_accum > 0 && (num_accum == num_ct || has_extra_ct),
             "PermuteAccum: Invalid number of accumulations");
  for (int i = 0; i < num_accum; i++) {
    const auto &src = srcs.at(i);
    AssertTrue(num_poly == static_cast<int>(src.size()),
               "PermuteAccum: Incompatible dst/src size");
    for (int j = 0; j < num_poly; j++) {
      AssertTrue(dst.at(j).data() != src.at(j).data(),
                 "PermuteAccum does not support inplace operation");
    }
  }
  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  int half_degree = param_.degree_ / 2;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;

    // Prepare PtrList objects
    OutputPtrList<word, j> dst_ptr_list(dst);
    std::vector<PermuteInputPtrList<word, j>> src_ptr_list;
    for (int i = 0; i < num_accum; i++) {
      const auto &src_i = srcs.at(i);
      src_ptr_list.emplace_back(src_i);
      src_ptr_list.back().extra_ = src_i.at(0).QSize() - q_size;
      src_ptr_list.back().galois_factor_ =
          PermuteAmountToGaloisFactor(permute_amounts.at(i));
      src_ptr_list.back().galois_offset_ =
          param_.GetGaloisOffset(src_ptr_list.back().galois_factor_);
    }

    if (has_extra_ct) {
      InputPtrList<word, j> extra_ct(srcs.back());
      extra_ct.extra_ = srcs.back().at(0).QSize() - q_size;

      // Hard-coded kernel launch
      switch (num_accum) {
        case 1:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0]);
          break;
        case 2:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0],
              src_ptr_list[1]);
          break;
        case 3:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2]);
          break;
        case 4:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3]);
          break;
        case 5:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4]);
          break;
        case 6:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4], src_ptr_list[5]);
          break;
        case 7:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4], src_ptr_list[5], src_ptr_list[6]);
          break;
        case 8:
          kernel::PermuteAccumAdd<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, extra_ct, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4], src_ptr_list[5], src_ptr_list[6],
              src_ptr_list[7]);
          break;
        default:
          Fail("PermuteAccum: Invalid number of accumulations");
          break;
      }
    } else {
      // Hard-coded kernel launch
      switch (num_accum) {
        case 1:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0]);
          break;
        case 2:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
              src_ptr_list[1]);
          break;
        case 3:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2]);
          break;
        case 4:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3]);
          break;
        case 5:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4]);
          break;
        case 6:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4], src_ptr_list[5]);
          break;
        case 7:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4], src_ptr_list[5], src_ptr_list[6]);
          break;
        case 8:
          kernel::PermuteAccum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
              dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
              src_ptr_list[1], src_ptr_list[2], src_ptr_list[3],
              src_ptr_list[4], src_ptr_list[5], src_ptr_list[6],
              src_ptr_list[7]);
          break;
        default:
          Fail("PermuteAccum: Invalid number of accumulations");
          break;
      }
    }
  });
}

template <typename word>
void ElementWiseHandler<word>::Accum(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<std::vector<DvConstView<word>>> &srcs) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "Accum: Invalid number of polynomials");

  int num_accum = srcs.size();

  if (num_accum > max_num_accum_) {
    // Prepare temporary result DV
    std::vector<DeviceVector<word>> temp;
    std::vector<DvView<word>> temp_view;
    std::vector<DvConstView<word>> temp_const_view;
    for (int i = 0; i < num_poly; i++) {
      temp.emplace_back(dst.at(i).TotalSize());
      temp_view.push_back(temp.at(i).View(dst.at(i).AuxSize()));
      temp_const_view.push_back(temp.at(i).ConstView(dst.at(i).AuxSize()));
    }

    // Split the srcs/permute_amounts into front and back
    std::vector<std::vector<DvConstView<word>>> srcs_front(
        srcs.begin(), srcs.begin() + max_num_accum_ - 1);
    srcs_front.push_back(temp_const_view);
    std::vector<std::vector<DvConstView<word>>> srcs_back(
        srcs.begin() + max_num_accum_ - 1, srcs.end());

    // Recursive call to accumulate the front and back
    Accum(temp_view, np, srcs_back);
    Accum(dst, np, srcs_front);
    return;
  }

  // Some sanity checks
  AssertTrue(num_accum > 1, "Accum: Invalid number of accumulations");
  for (int i = 0; i < num_accum; i++) {
    const auto &src = srcs.at(i);
    AssertTrue(num_poly == static_cast<int>(src.size()),
               "Accum: Incompatible dst/src size");
  }

  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;

    // Preparing PtrList objects
    OutputPtrList<word, j> dst_ptr_list(dst);
    std::vector<InputPtrList<word, j>> src_ptr_list;
    for (int i = 0; i < num_accum; i++) {
      const auto &src_i = srcs.at(i);
      src_ptr_list.emplace_back(src_i);
      src_ptr_list.back().extra_ = src_i.at(0).QSize() - q_size;
    }

    // Hard-coded kernel launch
    switch (num_accum) {
      case 2:
        kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
            dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
            src_ptr_list[1]);
        break;
      case 3:
        kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
            dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
            src_ptr_list[1], src_ptr_list[2]);
        break;
      case 4:
        kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
            dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
            src_ptr_list[1], src_ptr_list[2], src_ptr_list[3]);
        break;
      case 5:
        kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
            dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
            src_ptr_list[1], src_ptr_list[2], src_ptr_list[3], src_ptr_list[4]);
        break;
      case 6:
        kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
            dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
            src_ptr_list[1], src_ptr_list[2], src_ptr_list[3], src_ptr_list[4],
            src_ptr_list[5]);
        break;
      case 7:
        kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
            dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
            src_ptr_list[1], src_ptr_list[2], src_ptr_list[3], src_ptr_list[4],
            src_ptr_list[5], src_ptr_list[6]);
        break;
      case 8:
        kernel::Sum<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
            dst_ptr_list, primes, num_q_primes, src_ptr_list[0],
            src_ptr_list[1], src_ptr_list[2], src_ptr_list[3], src_ptr_list[4],
            src_ptr_list[5], src_ptr_list[6], src_ptr_list[7]);
        break;
      default:
        Fail("Accum: Invalid number of accumulations");
        break;
    }
  });
}

template <typename word>
void ElementWiseHandler<word>::CAccum(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
    const std::vector<DvConstView<word>> &common_srcs) const {
  if (kMergeCMult) {
    CPAccumWorker<true>(dst, np, ct_srcs, common_srcs);
    return;
  }

  // Naive repetition of CMult and add
  int num_poly = dst.size();
  int num_accum = common_srcs.size();
  bool has_extra_ct = (ct_srcs.size() == (common_srcs.size() + 1));
  AssertTrue(num_accum == static_cast<int>(ct_srcs.size()) || has_extra_ct,
             "CPAccum: Incompatible ct_srcs/common_srcs size");

  std::vector<DeviceVector<word>> tmp;
  std::vector<DvView<word>> tmp_view;
  std::vector<DvConstView<word>> tmp_const_view;
  std::vector<DvConstView<word>> dst_const_view;
  for (int i = 0; i < num_poly; i++) {
    tmp.emplace_back(dst.at(i).TotalSize());
    tmp_view.push_back(tmp.at(i).View(dst.at(i).AuxSize()));
    tmp_const_view.push_back(tmp.at(i).ConstView(dst.at(i).AuxSize()));
    dst_const_view.emplace_back(dst.at(i));
  }

  // first iteration
  if (has_extra_ct) {
    MultConst(tmp_view, np, ct_srcs.at(0), common_srcs.at(0));
    Add(dst, np, tmp_const_view, ct_srcs.back());
  } else {
    MultConst(dst, np, ct_srcs.at(0), common_srcs.at(0));
  }
  for (int i = 1; i < num_accum; i++) {
    MultConst(tmp_view, np, ct_srcs.at(i), common_srcs.at(i));
    Add(dst, np, tmp_const_view, dst_const_view);
  }
}

template <typename word>
void ElementWiseHandler<word>::PAccum(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
    const std::vector<DvConstView<word>> &common_srcs) const {
  if (kMergePMult) {
    CPAccumWorker<false>(dst, np, ct_srcs, common_srcs);
    return;
  }

  // Naive repetition of PMult and add
  int num_poly = dst.size();
  int num_accum = common_srcs.size();
  bool has_extra_ct = (ct_srcs.size() == (common_srcs.size() + 1));
  AssertTrue(num_accum == static_cast<int>(ct_srcs.size()) || has_extra_ct,
             "CPAccum: Incompatible ct_srcs/common_srcs size");

  std::vector<DeviceVector<word>> tmp;
  std::vector<DvView<word>> tmp_view;
  std::vector<DvConstView<word>> tmp_const_view;
  std::vector<DvConstView<word>> dst_const_view;
  for (int i = 0; i < num_poly; i++) {
    tmp.emplace_back(dst.at(i).TotalSize());
    tmp_view.push_back(tmp.at(i).View(dst.at(i).AuxSize()));
    tmp_const_view.push_back(tmp.at(i).ConstView(dst.at(i).AuxSize()));
    dst_const_view.emplace_back(dst.at(i));
  }

  // first iteration
  if (has_extra_ct) {
    PMult(tmp_view, np, ct_srcs.at(0), common_srcs.at(0));
    Add(dst, np, tmp_const_view, ct_srcs.back());
  } else {
    PMult(dst, np, ct_srcs.at(0), common_srcs.at(0));
  }
  for (int i = 1; i < num_accum; i++) {
    PMult(tmp_view, np, ct_srcs.at(i), common_srcs.at(i));
    Add(dst, np, tmp_const_view, dst_const_view);
  }
}

template <typename word>
template <bool const_accum>
void ElementWiseHandler<word>::CPAccumWorker(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
    const std::vector<DvConstView<word>> &common_srcs) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "CPAccum: Invalid number of polynomials");

  int num_accum = common_srcs.size();
  bool has_extra_ct = (ct_srcs.size() == (common_srcs.size() + 1));
  AssertTrue(num_accum == static_cast<int>(ct_srcs.size()) || has_extra_ct,
             "CPAccum: Incompatible ct_srcs/common_srcs size");

  if (num_accum > max_num_accum_) {
    // Prepare temporary result DV
    std::vector<DeviceVector<word>> temp;
    std::vector<DvView<word>> temp_view;
    std::vector<DvConstView<word>> temp_const_view;
    for (int i = 0; i < num_poly; i++) {
      int aux_size = dst.at(i).AuxSize();
      temp.emplace_back(dst.at(i).TotalSize());
      temp_view.push_back(temp.at(i).View(aux_size));
      temp_const_view.push_back(temp.at(i).ConstView(aux_size));
    }

    // Split the ct_srcs/common_srcs into front and back
    std::vector<std::vector<DvConstView<word>>> ct_srcs_front(
        ct_srcs.begin(), ct_srcs.begin() + max_num_accum_);
    ct_srcs_front.push_back(temp_const_view);
    std::vector<std::vector<DvConstView<word>>> ct_srcs_back(
        ct_srcs.begin() + max_num_accum_, ct_srcs.end());
    std::vector<DvConstView<word>> common_srcs_front(
        common_srcs.begin(), common_srcs.begin() + max_num_accum_);
    std::vector<DvConstView<word>> common_srcs_back(
        common_srcs.begin() + max_num_accum_, common_srcs.end());

    // Recursive call to accumulate the front and back
    CPAccumWorker<const_accum>(temp_view, np, ct_srcs_back, common_srcs_back);
    CPAccumWorker<const_accum>(dst, np, ct_srcs_front, common_srcs_front);
    return;
  }

  // Some sanity checks
  AssertTrue(num_accum > 0, "CPAccum: Invalid number of accumulations");
  for (const auto &ct_src : ct_srcs) {
    AssertTrue(num_poly == static_cast<int>(ct_src.size()),
               "CPAccum: Incompatible dst/ct_src size");
  }

  AssertNPMatch(dst, np);

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;

    // Preparing PtrList objects
    OutputPtrList<word, j> dst_ptr_list(dst);
    std::vector<CPAccumInputPtrList<word, j>> src_ptr_list;
    for (int i = 0; i < num_accum; i++) {
      const auto &ct_src_i = ct_srcs.at(i);
      const auto &common_src_i = common_srcs.at(i);
      src_ptr_list.emplace_back(ct_src_i, common_src_i);
      src_ptr_list.back().extra_ = ct_src_i.at(0).QSize() - q_size;
      src_ptr_list.back().common_extra_ = common_src_i.QSize();
      if constexpr (const_accum) {  // CAccum
        src_ptr_list.back().common_extra_ -= num_q_primes;
      } else {  // PAccum
        src_ptr_list.back().common_extra_ -= q_size;
      }
    }

    // Hard-coded kernel launch
    if (has_extra_ct) {
      InputPtrList<word, j> src0(ct_srcs.back());
      src0.extra_ = ct_srcs.back().at(0).QSize() - q_size;
      switch (num_accum) {
        case 1:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_,
                                                dst_ptr_list, primes,
                                                inv_primes, num_q_primes, src0,
                                                src_ptr_list[0]);
          break;
        case 2:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes, src0,
                  src_ptr_list[0], src_ptr_list[1]);
          break;
        case 3:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes, src0,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2]);
          break;
        case 4:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes, src0,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3]);
          break;
        case 5:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes, src0,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4]);
          break;
        case 6:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes, src0,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4], src_ptr_list[5]);
          break;
        case 7:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes, src0,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4], src_ptr_list[5],
                  src_ptr_list[6]);
          break;
        case 8:
          kernel::CPAccumAdd<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes, src0,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4], src_ptr_list[5],
                  src_ptr_list[6], src_ptr_list[7]);
          break;
        default:
          Fail("CPAccum: Invalid number of accumulations");
          break;
      }
    } else {
      switch (num_accum) {
        case 1:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_,
                                                dst_ptr_list, primes,
                                                inv_primes, num_q_primes,
                                                src_ptr_list[0]);
          break;
        case 2:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes,
                  src_ptr_list[0], src_ptr_list[1]);
          break;
        case 3:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2]);
          break;
        case 4:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3]);
          break;
        case 5:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4]);
          break;
        case 6:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4], src_ptr_list[5]);
          break;
        case 7:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4], src_ptr_list[5],
                  src_ptr_list[6]);
          break;
        case 8:
          kernel::CPAccum<word, j, const_accum>
              <<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
                  dst_ptr_list, primes, inv_primes, num_q_primes,
                  src_ptr_list[0], src_ptr_list[1], src_ptr_list[2],
                  src_ptr_list[3], src_ptr_list[4], src_ptr_list[5],
                  src_ptr_list[6], src_ptr_list[7]);
          break;
        default:
          Fail("CPAccum: Invalid number of accumulations");
          break;
      }
    }
  });
}

template <typename word>
void ElementWiseHandler<word>::ModUpToLevel(DvView<word> &dst,
                                           const DvConstView<word> &src1,
                                           int target_level) const {
  // Warn("ModUpToLevel kernel is not very optimal.");
  int num_aux = dst.AuxSize() / param_.degree_;
  if (target_level < 0) target_level = param_.max_level_;
  AssertTrue(target_level > 0 && target_level <= param_.max_level_,
             "ModUpToLevel: target level out of range");
  NPInfo max_np = param_.LevelToNP(target_level, num_aux);
  NPInfo min_np = param_.LevelToNP(-1);
  std::vector<word> base_primes = param_.GetPrimeVector(min_np);
  auto dst_temp = std::vector<DvView<word>>{dst};
  AssertNPMatch(dst_temp, max_np);

  const word *primes = param_.GetPrimesPtr(max_np);
  int num_q_primes = max_np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = max_np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  if (base_primes.size() == 2) {
    kernel::ModUpToMax2<word><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst.data(), primes, base_primes.at(0), base_primes.at(1), src1.data());
  } else if (base_primes.size() == 1) {
    kernel::ModUpToMax1<word><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst.data(), primes, base_primes.at(0), src1.data());
  } else {
    Fail("ModUpToLevel: Invalid base primes size");
  }
}

template <typename word>
void ElementWiseHandler<word>::MultImaginaryUnit(
    std::vector<DvView<word>> &dst, const NPInfo &np,
    const std::vector<DvConstView<word>> &src1,
    const DvConstView<word> &src_i_unit) const {
  // Check the size of the vectors
  int num_poly = dst.size();
  AssertTrue(num_poly == static_cast<int>(src1.size()),
             "MultImaginaryUnit: Incompatible dst/src size");
  AssertTrue(num_poly > 0 && num_poly <= max_num_poly_,
             "MultImaginaryUnit: Invalid number of polynomials");
  AssertNPMatch({dst}, np);

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int grid_dim = np.GetNumTotal() * param_.degree_ / kernel_block_dim_;

  constexpr_for<1, max_num_poly_ + 1>([&](auto j) {
    if (num_poly != j) return;
    OutputPtrList<word, j> dst_ptr_list(dst);
    InputPtrList<word, j> src1_ptr_list(src1);
    src1_ptr_list.extra_ = src1.at(0).QSize() - q_size;
    InputPtrList<word, 1> i_unit(src_i_unit);
    i_unit.extra_ = src_i_unit.QSize() - q_size;

    kernel::MultImaginaryUnit<word, j><<<grid_dim, kernel_block_dim_>>>(param_.log_degree_, 
        dst_ptr_list, primes, inv_primes, num_q_primes, src1_ptr_list, i_unit);
  });
}

// ----- Batched key switching ----- //

namespace kernel {

template <typename word>
struct DigitPtrList {
  const word *ptrs_[8];
  int num_digits_ = 0;
  int batch_stride_ = 0;
};

// One switch per blockIdx.z: accum.b = sum_i modup_i * key_b_i + p_prod * bx,
// accum.a = sum_i modup_i * key_a_i -- CPAccum's per-digit products and sums
// and CAccum's p_prod term, in one pass. The modular sum does not care about
// the order the terms arrive in, so the words are those of the two kernels.
template <typename word>
__global__ void KeyMultBatch(int log_degree, word *dst, int dst_batch_stride,
                             int ext_words, const word *primes,
                             const make_signed_t<word> *inv_primes,
                             int num_q_primes, const DigitPtrList<word> modup,
                             const word *const *key_table, int key_extra,
                             const word *bx, int bx_batch_stride,
                             const word *p_prod) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ext_words) return;
  const int z = blockIdx.z;
  const int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const make_signed_t<word> inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  const bool aux_part = (prime_index >= num_q_primes);
  const int key_index = aux_part ? i + key_extra : i;

  word res_b = 0;
  word res_a = 0;
  const word *const *keys = key_table + z * modup.num_digits_ * 2;
  for (int d = 0; d < modup.num_digits_; d++) {
    const word m = basic::StreamingLoad(modup.ptrs_[d] + z * modup.batch_stride_ + i);
    const word kb = basic::StreamingLoad(keys[2 * d] + key_index);
    const word ka = basic::StreamingLoad(keys[2 * d + 1] + key_index);
    res_b = basic::Add(res_b, basic::MultMontgomery(m, kb, prime, inv_prime),
                       prime);
    res_a = basic::Add(res_a, basic::MultMontgomery(m, ka, prime, inv_prime),
                       prime);
  }
  if (!aux_part) {
    const word pp = basic::StreamingLoadConst(p_prod + prime_index);
    const word b = basic::StreamingLoad(bx + z * bx_batch_stride + i);
    res_b = basic::Add(res_b, basic::MultMontgomery(pp, b, prime, inv_prime),
                       prime);
  }
  word *out = dst + z * dst_batch_stride;
  out[i] = res_b;
  out[ext_words + i] = res_a;
}

// Permute's index map with the automorphism read per blockIdx.z.
template <typename word>
__global__ void PermuteBatch(int log_degree, word *dst, const word *src,
                             int batch_stride, int poly_stride, int total_words,
                             const uint32_t *galois_factors,
                             const uint32_t *galois_offsets) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_words) return;
  const int z = blockIdx.z;
  const int degree = 1 << log_degree;
  const int prime_index = (i >> log_degree);
  const uint32_t x_idx = i & (degree - 1);
  const uint32_t x_idx_rev = basic::BitReverse(x_idx, log_degree + 1);
  const uint32_t gf = galois_factors[z];
  const uint32_t go = galois_offsets[z];
  const int src_idx = (prime_index << log_degree) +
                      basic::BitReverse(x_idx_rev * gf + go, log_degree + 1);
  const int base = z * batch_stride + blockIdx.y * poly_stride;
  dst[base + i] = basic::StreamingLoad(src + base + src_idx);
}

}  // namespace kernel

template <typename word>
void ElementWiseHandler<word>::KeyMultBatch(
    word *dst, int dst_batch_stride, const NPInfo &np,
    const std::vector<const word *> &modup, int modup_batch_stride,
    const word *const *key_table, int key_extra, const word *bx,
    int bx_batch_stride, const word *p_prod, int batch) const {
  const int num_digits = static_cast<int>(modup.size());
  AssertTrue(num_digits >= 1 && num_digits <= max_batch_digits_,
             "KeyMultBatch: invalid digit count");
  AssertTrue(batch >= 1, "KeyMultBatch: invalid batch");
  const int ext_words = np.GetNumTotal() * param_.degree_;
  AssertTrue(dst_batch_stride >= 2 * ext_words,
             "KeyMultBatch: the accumulators overlap");
  kernel::DigitPtrList<word> list;
  for (int d = 0; d < num_digits; d++) list.ptrs_[d] = modup.at(d);
  list.num_digits_ = num_digits;
  list.batch_stride_ = modup_batch_stride;

  dim3 grid_dim(DivCeil(ext_words, kernel_block_dim_), 1, batch);
  kernel::KeyMultBatch<word><<<grid_dim, kernel_block_dim_>>>(
      param_.log_degree_, dst, dst_batch_stride, ext_words,
      param_.GetPrimesPtr(np), param_.GetInvPrimesPtr(np), np.GetNumQ(), list,
      key_table, key_extra, bx, bx_batch_stride, p_prod);
}

template <typename word>
void ElementWiseHandler<word>::PermuteBatch(
    word *dst, const word *src, int batch_stride, int poly_stride,
    int num_poly, const NPInfo &np, const uint32_t *galois_factors,
    const uint32_t *galois_offsets, int batch) const {
  AssertTrue(np.num_aux_ == 0, "PermuteBatch: no auxiliary limbs");
  AssertTrue(num_poly >= 1 && batch >= 1, "PermuteBatch: invalid shape");
  AssertTrue(dst != src, "PermuteBatch does not support inplace operation");
  const int total_words = np.GetNumTotal() * param_.degree_;
  AssertTrue(poly_stride >= total_words && batch_stride >= num_poly * poly_stride,
             "PermuteBatch: the polynomials overlap");
  dim3 grid_dim(DivCeil(total_words, kernel_block_dim_), num_poly, batch);
  kernel::PermuteBatch<word><<<grid_dim, kernel_block_dim_>>>(
      param_.log_degree_, dst, src, batch_stride, poly_stride, total_words,
      galois_factors, galois_offsets);
}

// explicit instantiation
template class ElementWiseHandler<uint32_t>;
template class ElementWiseHandler<uint64_t>;

}  // namespace cheddar