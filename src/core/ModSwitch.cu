#include <algorithm>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/DoubleWord.h"
#include "common/PrimeUtils.h"
#include "core/ModSwitch.h"
#include "extension/Profile.h"

// kernel constants
#define kUnrollNumber 4
#define kLimbBatching 3
#define kNumThreadsPerBlock 256

#define kNumThreadsX 64
#define kNumThreadsY 4

#define kMaxNumAccum 4

// Conjugate-invariant pair windows: half the unroll, so that a thread's own
// columns and their mirror columns together cost exactly the ordinary
// kernel's registers, shared tile and grid.
#define kCiUnroll (kUnrollNumber / 2)
namespace cheddar {
namespace kernel {

template <typename word>
__global__ void ModSwitchMatrixMult(word *dst, const word *primes,
                                    const make_signed_t<word> *inv_primes,
                                    const int src_len, const int dst_len,
                                    const int skip_start, const int skip_end,
                                    const make_signed_t<word> *src,
                                    const make_signed_t<word> *bconv_table,
                                    const int log_degree,
                                    const int dst_stride = 0,
                                    const int src_stride = 0) {
  // Several key switches at one level share the conversion table, so they
  // share a launch: blockIdx.z picks which, and the strides are 0 for one.
  dst += blockIdx.z * dst_stride;
  src += blockIdx.z * src_stride;
  // Load bconv_table into shared memory
  using signed_word = make_signed_t<word>;
  using signed_d_word = make_signed_double_word_t<word>;

  extern __shared__ char __smem[];

  // For mod_prime q, we prepare bconv_table in range
  // [- (q - 1) / 2, (q - 1) / 2] -- normalized
  // The src values are in range [0, 2^31), but we can normalize
  // them to (-2^30, 2^30) range
  // Therefore the multiplication result is in range
  // (- (q - 1) * 2^29, (q - 1) * 2^29)
  // We can do lazy reduction here.
  // After accumulating 8 results,
  // which will be in range (- (q - 1) * 2^32, (q - 1) * 2^32) < (-2^63, 2^63)
  // We then reduce this range to (-q * 2^31, q * 2^31) by
  // adding or subtracting q * 2^32
  // Then, we can again accumulate 4 results, and repeat the process

  signed_word *bconv_vector = reinterpret_cast<signed_word *>(__smem);
  int bconv_vector_size = src_len * (kLimbBatching * kNumThreadsY);

  signed_word *poly_frag = bconv_vector + bconv_vector_size;

  // Loading bconv table into shared memory
  int thread_idx_flattened = threadIdx.x + threadIdx.y * kNumThreadsX;
  int block_offset = blockIdx.y * bconv_vector_size;
  int bconv_table_size = src_len * dst_len;
  for (int i = thread_idx_flattened; i < bconv_vector_size;
       i += (kNumThreadsX * kNumThreadsY)) {
    int bconv_table_index = i + block_offset;
    if (bconv_table_index < bconv_table_size) {
      bconv_vector[i] = bconv_table[bconv_table_index];
    }
  }
  // __syncthreads();

  // Precomputation of offset for src & bconv_vector
  int degree_index =
      blockIdx.x * (kNumThreadsX * kUnrollNumber) + threadIdx.x * kUnrollNumber;
  src += degree_index;
  bconv_vector += src_len * threadIdx.y * kLimbBatching;
  poly_frag += threadIdx.x * kUnrollNumber;

  // Actual accumulation
  // Each thread is in charge of computing kLimbBatching x kUnrollNumber
  // submatrix of dst
  signed_d_word accum[kLimbBatching][kUnrollNumber] = {0};
  signed_word reg_bconv[kLimbBatching];
  signed_word reg_poly[kUnrollNumber];
  int dst_y_position =
      blockIdx.y * (kLimbBatching * kNumThreadsY) + threadIdx.y * kLimbBatching;

  word reg_primes[kLimbBatching];
  signed_word reg_inv_primes[kLimbBatching];
  for (int i = 0; i < kLimbBatching; i++) {
    int prime_index = dst_y_position + i;
    if (prime_index >= dst_len) break;

    if (prime_index >= skip_start) {
      prime_index += (skip_end - skip_start);
    }
    reg_primes[i] = basic::StreamingLoadConst(primes + prime_index);
    reg_inv_primes[i] = basic::StreamingLoadConst(inv_primes + prime_index);
  }

  int num_accumulated = 0;
  for (int i = 0; i < src_len; i += kMaxNumAccum) {
    // 1. Load (kNumThreadsX * kUnrollNumber) x kMaxNumAccum chunk from memory
    // and store it in shared memory (poly_frag)
    if (i > 0)
      __syncthreads();  // works like a lock_acquire (we don't need to lock
                        // until reading __smem)
#pragma unroll
    for (int j = 0; j < (kMaxNumAccum / kNumThreadsY); j++) {
      int y_pos = j * kNumThreadsY + threadIdx.y;
      if (i + y_pos >= src_len) break;
      basic::VectorizedMove<signed_word, kUnrollNumber>(
          poly_frag + y_pos * (kNumThreadsX * kUnrollNumber),
          src + (y_pos << log_degree));
    }
    __syncthreads();  // works like a lock_release

    // 2. Perform actual matrix multiplication
    for (int j = 0; j < kMaxNumAccum; j++) {
      if (i + j >= src_len) break;

      basic::VectorizedMove<signed_word, kUnrollNumber>(
          reg_poly, poly_frag + j * (kUnrollNumber * kNumThreadsX));
      for (int k = 0; k < kLimbBatching; k++) {
        if (dst_y_position + k > dst_len) break;  // out of bounds
        signed_word bconv_const = bconv_vector[src_len * k];
#pragma unroll
        for (int l = 0; l < kUnrollNumber; l++) {
          accum[k][l] += basic::detail::__mult_wide(reg_poly[l], bconv_const);
        }
      }
      // BConv vector offset: i * kMaxNumAccum + j
      bconv_vector += 1;
      num_accumulated += 1;
    }

    // 3. Normalize temporary result if necessary
    if constexpr (std::is_same_v<word, uint32_t>) {
      if (num_accumulated >= kMaxNumAccum) {
        for (int k = 0; k < kLimbBatching; k++) {
          if (dst_y_position + k >= dst_len) break;  // out of bound
          signed_d_word prime_th = reg_primes[k];
          prime_th <<= (sizeof(word) * 8);
          signed_d_word prime_th_half = prime_th >> 1;
#pragma unroll
          for (int l = 0; l < kUnrollNumber; l++) {
            if (accum[k][l] < 0) accum[k][l] += prime_th;
            if (accum[k][l] >= prime_th_half) accum[k][l] -= prime_th;
          }
        }
        num_accumulated -= kMaxNumAccum;
      }
    }

    // src offset: i * kMaxNumAccum * degree
    src += (kMaxNumAccum << log_degree);
  }

  for (int k = 0; k < kLimbBatching; k++) {
    int prime_index = dst_y_position + k;
    if (dst_y_position + k >= dst_len) break;  // out of bound
    if (prime_index >= skip_start) {
      prime_index += (skip_end - skip_start);
    }
    word prime = reg_primes[k];
    signed_word inv_prime = reg_inv_primes[k];
    word res_tmp[kUnrollNumber];
#pragma unroll
    for (int l = 0; l < kUnrollNumber; l++) {
      res_tmp[l] = basic::ReduceMontgomery(accum[k][l], prime, inv_prime);
    }
    basic::VectorizedMove<word, kUnrollNumber>(
        dst + (prime_index << log_degree) + degree_index, res_tmp);
  }
}

// The coefficient-domain stand-in for the constant multiply INTTPhase2
// performs at the end of INTTAndMultConst. `consts` is mod_up1_coeff_ rather
// than mod_up1_ -- see ModSwitchHandler::ModUpFromCoeff for why the two differ
// by N. The centred representative is not cosmetic: ModSwitchMatrixMult reads
// this buffer as signed words, exactly as MultConstNormalize leaves it.
template <typename word>
__global__ void ModUpMultConst(make_signed_t<word> *dst, word *dst_mont,
                               const word *src, const word *primes,
                               const make_signed_t<word> *inv_primes,
                               const word *consts, const word *mont_r2,
                               int log_degree, int q_words, int mont_stride) {
  using signed_word = make_signed_t<word>;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_index = (i >> log_degree);
  // blockIdx.y is which key switch this is. The Montgomery output has its own
  // stride because it is written straight into the extended basis, where each
  // switch occupies num_q + num_aux limbs rather than num_q.
  src += blockIdx.y * q_words;
  dst += blockIdx.y * q_words;
  dst_mont += blockIdx.y * mont_stride;
  const word prime = primes[prime_index];
  const signed_word inv_prime = inv_primes[prime_index];
  const word value = src[i];

  const word temp =
      basic::MultMontgomery(value, consts[prime_index], prime, inv_prime);
  signed_word result = static_cast<signed_word>(temp);
  if (temp > (prime >> 1)) result -= static_cast<signed_word>(prime);
  dst[i] = result;

  // The same coefficients for the limbs that pass through, in the Montgomery
  // form the forward transform downstream will not put them in.
  dst_mont[i] =
      basic::MultMontgomery(value, mont_r2[prime_index], prime, inv_prime);
}

// ----- Conjugate-invariant ring: the conversion carries the fold ----- //
//
// A key switch runs INTT -> base conversion -> NTT, and on the
// conjugate-invariant ring each transform carries a fold pass of its own
// (CiUnfold / CiFold in NTT.cu): one full extra read-modify-write of every
// limb it touches. The FOLD side collapses into the conversion: the kernel
// below reapplies the fold in registers on the way out, and the forward
// transform then starts straight at phase 1 (ci_prefolded) -- for ModUp,
// ModDown, Rescale and ModDownAndRescale alike, which is every key switch
// and every rescale in the system. The UNFOLD side stays where it was: the
// INTT keeps its own cheap elementwise pass. Carrying it in here was built
// and measured twice -- undone at the shared-memory load it cost occupancy
// (72 registers, and 67 us against the ordinary kernel's 27), undone at the
// accumulator read it ran once per destination-limb triple instead of once
// (89 us) -- and the fold-only form below, at 40 us, is what actually beats
// the separate passes.
//
// The fold pairs column t with degree - t (and 0 with degree / 2), so where
// the ordinary kernel hands a thread kUnrollNumber consecutive columns, this
// one hands it kCiUnroll consecutive values of t in [0, degree / 2) AND
// their mirrors: the same column count, register budget, shared tile and
// grid. The matrix arithmetic in the middle is the ordinary kernel's,
// unchanged. Mirror-side global accesses run at odd offsets and stay scalar;
// consecutive threads still touch consecutive addresses, descending.
template <typename word>
__global__ void __launch_bounds__(kNumThreadsPerBlock, 4) ModSwitchMatrixMultCi(
    word *dst, const word *primes, const make_signed_t<word> *inv_primes,
    const int src_len, const int dst_len, const int skip_start,
    const int skip_end, const make_signed_t<word> *src,
    const make_signed_t<word> *bconv_table, const int log_degree,
    const word *ci_i, const word *ci_fwd_twist, const int dst_tw_offset,
    const int dst_tw_num_q, const int dst_tw_extra, const int dst_stride = 0,
    const int src_stride = 0) {
  dst += blockIdx.z * dst_stride;
  src += blockIdx.z * src_stride;
  using signed_word = make_signed_t<word>;
  using signed_d_word = make_signed_double_word_t<word>;

  extern __shared__ char __smem[];

  const int degree = 1 << log_degree;
  const int half = degree >> 1;
  constexpr int kWindow = kNumThreadsX * kCiUnroll;

  signed_word *bconv_vector = reinterpret_cast<signed_word *>(__smem);
  int bconv_vector_size = src_len * (kLimbBatching * kNumThreadsY);

  signed_word *poly_frag = bconv_vector + bconv_vector_size;

  // Loading bconv table into shared memory, exactly as the ordinary kernel
  int thread_idx_flattened = threadIdx.x + threadIdx.y * kNumThreadsX;
  int block_offset = blockIdx.y * bconv_vector_size;
  int bconv_table_size = src_len * dst_len;
  for (int i = thread_idx_flattened; i < bconv_vector_size;
       i += (kNumThreadsX * kNumThreadsY)) {
    int bconv_table_index = i + block_offset;
    if (bconv_table_index < bconv_table_size) {
      bconv_vector[i] = bconv_table[bconv_table_index];
    }
  }

  // The block owns kWindow consecutive values of t and their mirror columns.
  // Row r of the shared tile holds the t window at [0, kWindow) and the
  // mirror window at [kWindow, 2 * kWindow), t-indexed both, so the matrix
  // loop below sees plain consecutive slots either way.
  const int t_base = blockIdx.x * kWindow;
  bconv_vector += src_len * threadIdx.y * kLimbBatching;
  poly_frag += threadIdx.x * kCiUnroll;

  signed_d_word accum[kLimbBatching][2 * kCiUnroll] = {0};
  signed_word reg_poly[2 * kCiUnroll];
  int dst_y_position =
      blockIdx.y * (kLimbBatching * kNumThreadsY) + threadIdx.y * kLimbBatching;

  word reg_primes[kLimbBatching];
  signed_word reg_inv_primes[kLimbBatching];
  for (int i = 0; i < kLimbBatching; i++) {
    int prime_index = dst_y_position + i;
    if (prime_index >= dst_len) break;

    if (prime_index >= skip_start) {
      prime_index += (skip_end - skip_start);
    }
    reg_primes[i] = basic::StreamingLoadConst(primes + prime_index);
    reg_inv_primes[i] = basic::StreamingLoadConst(inv_primes + prime_index);
  }

  const int t0 = t_base + threadIdx.x * kCiUnroll;
  int num_accumulated = 0;
  for (int i = 0; i < src_len; i += kMaxNumAccum) {
    // 1. Load the pair windows into shared memory. The own side is a plain
    // vector move; the mirror side descends and starts at an odd offset, so
    // it stays scalar -- consecutive threads still touch consecutive
    // addresses.
    if (i > 0)
      __syncthreads();  // works like a lock_acquire
#pragma unroll
    for (int j = 0; j < (kMaxNumAccum / kNumThreadsY); j++) {
      int y_pos = j * kNumThreadsY + threadIdx.y;
      if (i + y_pos >= src_len) break;
      const signed_word *src_limb = src + (y_pos << log_degree);
      signed_word *frag_row = poly_frag + y_pos * (2 * kWindow);
      basic::VectorizedMove<signed_word, kCiUnroll>(frag_row, src_limb + t0);
#pragma unroll
      for (int e = 0; e < kCiUnroll; e++) {
        const int t = t0 + e;
        const int mirror = (t == 0) ? half : (degree - t);
        frag_row[kWindow + e] = basic::StreamingLoad(src_limb + mirror);
      }
    }
    __syncthreads();  // works like a lock_release

    // 2. Perform actual matrix multiplication -- the ordinary kernel's loop
    // over the same number of columns per thread.
    for (int j = 0; j < kMaxNumAccum; j++) {
      if (i + j >= src_len) break;

      basic::VectorizedMove<signed_word, kCiUnroll>(
          reg_poly, poly_frag + j * (2 * kWindow));
      basic::VectorizedMove<signed_word, kCiUnroll>(
          reg_poly + kCiUnroll, poly_frag + j * (2 * kWindow) + kWindow);
      for (int k = 0; k < kLimbBatching; k++) {
        if (dst_y_position + k > dst_len) break;  // out of bounds
        signed_word bconv_const = bconv_vector[src_len * k];
#pragma unroll
        for (int l = 0; l < 2 * kCiUnroll; l++) {
          accum[k][l] += basic::detail::__mult_wide(reg_poly[l], bconv_const);
        }
      }
      bconv_vector += 1;
      num_accumulated += 1;
    }

    // 3. Normalize temporary result if necessary
    if constexpr (std::is_same_v<word, uint32_t>) {
      if (num_accumulated >= kMaxNumAccum) {
        for (int k = 0; k < kLimbBatching; k++) {
          if (dst_y_position + k >= dst_len) break;  // out of bound
          signed_d_word prime_th = reg_primes[k];
          prime_th <<= (sizeof(word) * 8);
          signed_d_word prime_th_half = prime_th >> 1;
#pragma unroll
          for (int l = 0; l < 2 * kCiUnroll; l++) {
            if (accum[k][l] < 0) accum[k][l] += prime_th;
            if (accum[k][l] >= prime_th_half) accum[k][l] -= prime_th;
          }
        }
        num_accumulated -= kMaxNumAccum;
      }
    }

    // src offset: i * kMaxNumAccum * degree
    src += (kMaxNumAccum << log_degree);
  }

  for (int k = 0; k < kLimbBatching; k++) {
    int prime_index = dst_y_position + k;
    if (dst_y_position + k >= dst_len) break;  // out of bound
    if (prime_index >= skip_start) {
      prime_index += (skip_end - skip_start);
    }
    word prime = reg_primes[k];
    signed_word inv_prime = reg_inv_primes[k];

    // Reapply the fold on the way out, so the transform downstream starts
    // straight at phase 1. Exactly CiFoldKernel, element for element.
    const int tw = dst_tw_offset + prime_index +
                   (prime_index >= dst_tw_num_q ? dst_tw_extra : 0);
    const word i_unit = ci_i[tw];
    const word *ftw = ci_fwd_twist + (tw << log_degree);

    word v[2 * kCiUnroll];
#pragma unroll
    for (int l = 0; l < 2 * kCiUnroll; l++) {
      v[l] = basic::ReduceMontgomery(accum[k][l], prime, inv_prime);
    }
    word out_own[kCiUnroll];
    word out_mir[kCiUnroll];
#pragma unroll
    for (int e = 0; e < kCiUnroll; e++) {
      const int t = t0 + e;
      const word raw_own = v[e];
      const word raw_mir = v[kCiUnroll + e];
      if (t == 0) {
        // hat_N is zero and psi4^0 is one, so column 0 passes straight
        // through; column degree / 2 folds with itself.
        out_own[e] = raw_own;
        const word m_folded = basic::Sub<word>(
            raw_mir,
            basic::MultMontgomery<word>(i_unit, raw_mir, prime, inv_prime),
            prime);
        out_mir[e] = basic::MultMontgomery<word>(m_folded, ftw[half], prime,
                                                 inv_prime);
      } else {
        out_own[e] = basic::MultMontgomery<word>(
            basic::Sub<word>(raw_own,
                             basic::MultMontgomery<word>(i_unit, raw_mir,
                                                         prime, inv_prime),
                             prime),
            ftw[t], prime, inv_prime);
        out_mir[e] = basic::MultMontgomery<word>(
            basic::Sub<word>(raw_mir,
                             basic::MultMontgomery<word>(i_unit, raw_own,
                                                         prime, inv_prime),
                             prime),
            ftw[degree - t], prime, inv_prime);
      }
    }
    word *dst_limb = dst + (prime_index << log_degree);
    basic::VectorizedMove<word, kCiUnroll>(dst_limb + t0, out_own);
    if (t0 == 0) {
      dst_limb[half] = out_mir[0];
#pragma unroll
      for (int e = 1; e < kCiUnroll; e++) {
        dst_limb[degree - (t0 + e)] = out_mir[e];
      }
    } else {
#pragma unroll
      for (int e = 0; e < kCiUnroll; e++) {
        dst_limb[degree - (t0 + e)] = out_mir[e];
      }
    }
  }
}

// The conjugate-invariant ModUpMultConst: the raw centred copy for the
// matrix product is written exactly as the ordinary kernel writes it, and
// the Montgomery copy for the limbs that pass through is written already
// folded, because the transform it feeds starts straight at phase 1. One
// thread owns the mirrored column pair, as everywhere on this ring.
template <typename word>
__global__ void ModUpMultConstCi(make_signed_t<word> *dst, word *dst_mont,
                                 const word *src, const word *primes,
                                 const make_signed_t<word> *inv_primes,
                                 const word *consts, const word *mont_r2,
                                 const word *ci_i, const word *ci_fwd_twist,
                                 int tw_offset, int log_degree, int q_words,
                                 int mont_stride) {
  using signed_word = make_signed_t<word>;
  const int log_half = log_degree - 1;
  const int half = 1 << log_half;
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_index = pair >> log_half;
  const int t = pair & (half - 1);
  const int mirror = (t == 0) ? half : ((half << 1) - t);
  src += blockIdx.y * q_words;
  dst += blockIdx.y * q_words;
  dst_mont += blockIdx.y * mont_stride;
  const word prime = primes[prime_index];
  const signed_word inv_prime = inv_primes[prime_index];
  const word mult_const = consts[prime_index];
  const word r2 = mont_r2[prime_index];
  const int tw = tw_offset + prime_index;
  const word i_unit = basic::StreamingLoadConst(ci_i + tw);
  const word *ftw = ci_fwd_twist + (tw << log_degree);
  const int base = prime_index << log_degree;

  const word value_own = src[base + t];
  const word value_mir = src[base + mirror];

  const word temp_own =
      basic::MultMontgomery(value_own, mult_const, prime, inv_prime);
  signed_word result = static_cast<signed_word>(temp_own);
  if (temp_own > (prime >> 1)) result -= static_cast<signed_word>(prime);
  dst[base + t] = result;
  const word temp_mir =
      basic::MultMontgomery(value_mir, mult_const, prime, inv_prime);
  result = static_cast<signed_word>(temp_mir);
  if (temp_mir > (prime >> 1)) result -= static_cast<signed_word>(prime);
  dst[base + mirror] = result;

  // The fold commutes with the Montgomery conversion -- both are per-limb
  // scalar multiplies -- so folding the converted values is folding the
  // coefficients.
  const word mv_own = basic::MultMontgomery(value_own, r2, prime, inv_prime);
  const word mv_mir = basic::MultMontgomery(value_mir, r2, prime, inv_prime);
  if (t == 0) {
    dst_mont[base] = mv_own;
    const word m_folded = basic::Sub<word>(
        mv_mir, basic::MultMontgomery<word>(i_unit, mv_mir, prime, inv_prime),
        prime);
    dst_mont[base + half] = basic::MultMontgomery<word>(
        m_folded, basic::StreamingLoadConst(ftw + half), prime, inv_prime);
  } else {
    dst_mont[base + t] = basic::MultMontgomery<word>(
        basic::Sub<word>(mv_own,
                         basic::MultMontgomery<word>(i_unit, mv_mir, prime,
                                                     inv_prime),
                         prime),
        basic::StreamingLoadConst(ftw + t), prime, inv_prime);
    dst_mont[base + mirror] = basic::MultMontgomery<word>(
        basic::Sub<word>(mv_mir,
                         basic::MultMontgomery<word>(i_unit, mv_own, prime,
                                                     inv_prime),
                         prime),
        basic::StreamingLoadConst(ftw + mirror), prime, inv_prime);
  }
}

}  // namespace kernel

template <typename word>
ModSwitchHandler<word>::ModSwitchHandler(
    const Parameter<word> &param, int level,
    const ElementWiseHandler<word> &elem_handler,
    const NTTHandler<word> &ntt_handler, int num_aux /*= 0*/)
    : level_{level},
      num_aux_{num_aux > 0
                   ? num_aux
                   : (level == -1 ? param.GetSSENumAux() : param.alpha_)},
      beta_{level == -1 ? 1
                        : DivCeil(param.LevelToNP(level).num_main_ +
                                      param.GetMaxNumTer(),
                                  num_aux_)},
      param_{param},
      elem_handler_{elem_handler},
      ntt_handler_{ntt_handler} {
  static_assert(kMaxNumAccum % kNumThreadsY == 0,
                "kMaxNumAccum must be divisible by kNumThreadsY");
  static_assert(
      kNumThreadsPerBlock == kNumThreadsX * kNumThreadsY,
      "kNumThreadsPerBlock must be equal to kNumThreadsX * kNumThreadsY");


  NPInfo np = param_.LevelToNP(level_, num_aux_);
  int num_q_primes = np.GetNumQ();
  std::vector<word> level_primes = param_.GetPrimeVector(np);

  int padded_num_q_primes = np.num_main_ + param_.GetMaxNumTer();
  // Add pad to the front of the primes
  if (level_ == -1) padded_num_q_primes = num_q_primes;
  int num_pad = padded_num_q_primes - num_q_primes;

  // mod up constants
  mod_up1_.resize(num_q_primes);
  mod_up1_coeff_.resize(num_q_primes);
  mod_up2_.resize(beta_);
  for (int i = 0; i < beta_; i++) {
    // absolute index
    int src_start = i * num_aux_;
    int src_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (src_end <= num_pad) continue;

    // relative index
    src_start = Max(0, src_start - num_pad);
    src_end = src_end - num_pad;

    std::vector<word> src_primes(level_primes.begin() + src_start,
                                 level_primes.begin() + src_end);
    std::vector<word> dst_primes(level_primes.begin(),
                                 level_primes.begin() + src_start);
    dst_primes.insert(dst_primes.end(), level_primes.begin() + src_end,
                      level_primes.end());

    DeviceVector<word> mod_up1_tmp;
    PopulateModSwitchConstants(mod_up1_tmp, mod_up2_[i], src_primes, dst_primes,
                               0, 0);
    cudaMemcpyAsync(mod_up1_.data() + src_start, mod_up1_tmp.data(),
                    mod_up1_tmp.size() * sizeof(word), cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy);

    // The ModUpFromCoeff constant. PopulateModSwitchConstants folds an N^{-1}
    // into mod_up1_ to normalise the INTT that consumes it; that INTT does not
    // run when the caller already holds coefficients, so this one is the same
    // product inverse without it -- and in Montgomery form, because the kernel
    // that applies it multiplies with MultMontgomery.
    HostVector<word> coeff_const(src_primes.size());
    for (size_t s_idx = 0; s_idx < src_primes.size(); s_idx++) {
      const word modulus = src_primes[s_idx];
      word prod = 1;
      for (size_t t_idx = 0; t_idx < src_primes.size(); t_idx++) {
        if (t_idx == s_idx) continue;
        prod = primeutil::MultMod<word>(prod, src_primes[t_idx], modulus);
      }
      coeff_const[s_idx] = primeutil::ToMontgomery<word>(
          primeutil::InvMod<word>(prod, modulus), modulus);
    }
    DeviceVector<word> mod_up1_coeff_tmp;
    CopyHostToDevice(mod_up1_coeff_tmp, coeff_const);
    cudaMemcpyAsync(mod_up1_coeff_.data() + src_start, mod_up1_coeff_tmp.data(),
                    mod_up1_coeff_tmp.size() * sizeof(word),
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  }

  // R^2 per q prime, for the pass-through limbs on the coefficient path.
  HostVector<word> host_mont_r2(num_q_primes);
  for (int i = 0; i < num_q_primes; i++) {
    const word modulus = level_primes[i];
    host_mont_r2[i] =
        primeutil::ToMontgomery<word>(primeutil::ToMontgomery<word>(1, modulus),
                                      modulus);
  }
  CopyHostToDevice(mont_r2_, host_mont_r2);

  std::vector<word> q_primes(level_primes.begin(),
                             level_primes.begin() + num_q_primes);
  std::vector<word> p_primes(level_primes.begin() + num_q_primes,
                             level_primes.end());

  // ModDown constants
  PopulateModSwitchConstants(mod_down1_, mod_down2_, p_primes, q_primes, 0, 0);
  DeviceVector<word> dummy;
  PopulateModDownEpilogueConstants(inv_prime_prod_, dummy, p_primes, q_primes,
                                   0, 0);

  // Rescale not performed at level -1 or 0
  if (level_ == -1 || level_ == 0) return;

  // Rescale constants
  NPInfo next_np = param_.LevelToNP(level_ - 1, num_aux_);
  std::vector<word> next_level_primes = param_.GetPrimeVector(next_np);
  int next_num_q_primes = next_np.GetNumQ();
  int main_diff = np.num_main_ - next_np.num_main_;
  int ter_diff = np.num_ter_ - next_np.num_ter_;

  std::vector<word> rescale_dst_primes(
      next_level_primes.begin(), next_level_primes.begin() + next_num_q_primes);

  std::vector<word> rescale_src_primes;
  std::vector<word> mod_down_rescale_src_primes;
  if (ter_diff > 0) {  // number of ter decreases
    // May restore main primes
    AssertTrue(main_diff <= 0, "Invalid rescale amount");
    rescale_src_primes.insert(rescale_src_primes.end(), level_primes.begin(),
                              level_primes.begin() + ter_diff);
    mod_down_rescale_src_primes = rescale_src_primes;
    mod_down_rescale_src_primes.insert(mod_down_rescale_src_primes.end(),
                                       p_primes.begin(), p_primes.end());
    rescale_pad_start_ = 0;
    rescale_pad_end_ = num_q_primes - ter_diff;
    rescale_restore_start_ = next_num_q_primes + main_diff;
    rescale_restore_end_ = next_num_q_primes;
  } else if (main_diff > 0) {  // number of main decreases
    // May restore terminal primes
    AssertTrue(ter_diff <= 0, "Invalid rescale amount");
    rescale_src_primes.insert(rescale_src_primes.end(),
                              level_primes.begin() + num_q_primes - main_diff,
                              level_primes.begin() + num_q_primes);
    mod_down_rescale_src_primes = rescale_src_primes;
    mod_down_rescale_src_primes.insert(mod_down_rescale_src_primes.end(),
                                       p_primes.begin(), p_primes.end());
    rescale_pad_start_ = -ter_diff;
    rescale_pad_end_ = num_q_primes - main_diff - ter_diff;
    rescale_restore_start_ = 0;
    rescale_restore_end_ = -ter_diff;
  } else {
    Fail("Invalid rescale amount");
  }
  PopulateModSwitchConstants(rescale1_, rescale2_, rescale_src_primes,
                             rescale_dst_primes, rescale_restore_start_,
                             rescale_restore_end_);
  PopulateModDownEpilogueConstants(
      rescale_inv_prime_prod_, rescale_padding_, rescale_src_primes,
      rescale_dst_primes, rescale_restore_start_, rescale_restore_end_);
  PopulateModSwitchConstants(mod_down_rescale1_, mod_down_rescale2_,
                             mod_down_rescale_src_primes, rescale_dst_primes,
                             rescale_restore_start_, rescale_restore_end_);
  PopulateModDownEpilogueConstants(
      mod_down_rescale_inv_prime_prod_, mod_down_rescale_padding_,
      mod_down_rescale_src_primes, rescale_dst_primes, rescale_restore_start_,
      rescale_restore_end_);

  if (!kFuseModDownEpilogue) {
    HostVector<word> host_entire_padding(num_q_primes + num_aux_, 1);
    for (int i = 0; i < num_q_primes + num_aux_; i++) {
      word mod_prime = level_primes[i];
      for (int j = rescale_restore_start_; j < rescale_restore_end_; j++) {
        host_entire_padding[i] = primeutil::MultMod<word>(
            host_entire_padding[i], rescale_dst_primes[j], mod_prime);
      }
      host_entire_padding[i] =
          primeutil::ToMontgomery<word>(host_entire_padding[i], mod_prime);
    }
    CopyHostToDevice(entire_padding_, host_entire_padding);
  }
}

template <typename word>
void ModSwitchHandler<word>::PopulateModSwitchConstants(
    DeviceVector<word> &const1, DeviceVector<make_signed_t<word>> &const2,
    const std::vector<word> &src_primes, const std::vector<word> &dst_primes,
    int restore_start, int restore_end) {
  int src_len = src_primes.size();
  int dst_len = dst_primes.size();

  HostVector<word> mod_switch1(src_len, 1);
  HostVector<make_signed_t<word>> mod_switch2(src_len * dst_len, 1);

  for (int i = 0; i < src_len; i++) {
    word modulus = src_primes[i];

    for (int j = 0; j < src_len; j++) {
      if (j == i) continue;
      mod_switch1[i] =
          primeutil::MultMod<word>(mod_switch1[i], src_primes[j], modulus);
    }
    mod_switch1[i] = primeutil::MultMod<word>(
        mod_switch1[i], static_cast<word>(param_.degree_),
        modulus);  // for inverse N of INTT
    mod_switch1[i] = primeutil::InvMod<word>(mod_switch1[i], modulus);

    if (kFuseModDownEpilogue) {
      // BitPacker extension
      for (int j = restore_start; j < restore_end; j++) {
        mod_switch1[i] =
            primeutil::MultMod<word>(mod_switch1[i], dst_primes[j], modulus);
      }
    }
    // Deliberately not using the Montgomery form here
  }
  for (int m = 0; m < dst_len; m++) {
    word mod_prime = dst_primes[m];
    for (int i = 0; i < src_len; i++) {
      word accum = 1;
      for (int j = 0; j < src_len; j++) {
        if (i == j) continue;
        accum = primeutil::MultMod<word>(accum, src_primes[j], mod_prime);
      }
      // Double Montgomery form
      accum = primeutil::ToMontgomery<word>(accum, mod_prime);
      accum = primeutil::ToMontgomery<word>(accum, mod_prime);
      mod_switch2[m * src_len + i] = basic::Normalize(accum, mod_prime);
    }
  }

  CopyHostToDevice(const1, mod_switch1);
  CopyHostToDevice(const2, mod_switch2);
}

template <typename word>
void ModSwitchHandler<word>::PopulateModDownEpilogueConstants(
    DeviceVector<word> &inv_p_prod, DeviceVector<word> &padding,
    const std::vector<word> &src_primes, const std::vector<word> &dst_primes,
    int restore_start, int restore_end) {
  int src_len = src_primes.size();
  int dst_len = dst_primes.size();
  int restore_len = restore_end - restore_start;

  HostVector<word> host_inv_p_prod(dst_len, 1);
  for (int i = 0; i < dst_len; i++) {
    word mod_prime = dst_primes[i];
    for (int j = 0; j < src_len; j++) {
      host_inv_p_prod[i] = primeutil::MultMod<word>(host_inv_p_prod[i],
                                                    src_primes[j], mod_prime);
    }
    host_inv_p_prod[i] = primeutil::InvMod<word>(host_inv_p_prod[i], mod_prime);
    host_inv_p_prod[i] =
        primeutil::ToMontgomery<word>(host_inv_p_prod[i], mod_prime);
  }
  CopyHostToDevice(inv_p_prod, host_inv_p_prod);

  if (!kFuseModDownEpilogue) return;
  // We treat padding differently

  HostVector<word> host_padding(dst_len - restore_len, 1);
  for (int i = 0; i < dst_len - restore_len; i++) {
    int mod_prime_index = i;
    if (i >= restore_start) {
      mod_prime_index += restore_len;
    }
    word mod_prime = dst_primes[mod_prime_index];
    for (int j = restore_start; j < restore_end; j++) {
      host_padding[i] =
          primeutil::MultMod<word>(host_padding[i], dst_primes[j], mod_prime);
    }
    host_padding[i] = primeutil::ToMontgomery<word>(host_padding[i], mod_prime);
  }
  CopyHostToDevice(padding, host_padding);
}

template <typename word>
void ModSwitchHandler<word>::PseudoModUp(
    DvView<word> &dst, const DvConstView<word> &src,
    const DvConstView<word> &p_prod) const {
  NPInfo np = param_.LevelToNP(level_, 0);
  int num_q_primes = np.GetNumQ();
  AssertTrue(src.TotalSize() == num_q_primes * param_.degree_,
             "PseudoModUp src size mismatch");
  AssertTrue(dst.TotalSize() == num_q_primes * param_.degree_,
             "PseudoModUp dst size mismatch");
  AssertTrue(p_prod.TotalSize() == num_q_primes,
             "PseudoModUp p_prod size mismatch");
  std::vector<DvView<word>> dst_view{dst};
  std::vector<DvConstView<word>> src_view{src};
  elem_handler_.MultConst(dst_view, np, src_view, p_prod);
}
template <typename word>
void ModSwitchHandler<word>::ModUp(std::vector<DvView<word>> &dst,
                                   const DvConstView<word> &src) const {
  ModUpWorker(dst, &src, nullptr);
}

template <typename word>
void ModSwitchHandler<word>::ModUpFromCoeff(
    std::vector<DvView<word>> &dst,
    const DvConstView<word> &src_coeff) const {
  ModUpWorker(dst, nullptr, &src_coeff);
}

// Every key switch at one level raises the same three limbs into the same
// fifteen with the same tables, and each of those launches is a grid of about
// sixty blocks on a card with a hundred and eight multiprocessors. Doing a
// group of them at once is three launches for the group rather than five each,
// and it is the occupancy that matters more than the launch count.
template <typename word>
void ModSwitchHandler<word>::ModUpFromCoeffBatch(
    DvView<word> &dst, const DvConstView<word> &src_coeff, int batch) const {
  using signed_word = make_signed_t<word>;
  AssertTrue(batch >= 1, "ModUpFromCoeffBatch: invalid batch");
  AssertTrue(level_ >= 0,
             "ModUpFromCoeffBatch: the dense-to-sparse level is not batched");

  NPInfo np = param_.LevelToNP(level_, 0);
  const int num_q_primes = np.GetNumQ();
  const int prime_offset = param_.GetMaxNumTer() - np.num_ter_;
  const int degree = param_.degree_;
  const int q_words = num_q_primes * degree;
  const int total_limbs = num_q_primes + num_aux_;
  const int ext_words = total_limbs * degree;

  AssertTrue(src_coeff.TotalSize() == batch * q_words,
             "ModUpFromCoeffBatch: src size mismatch");
  AssertTrue(dst.TotalSize() == batch * beta_ * ext_words,
             "ModUpFromCoeffBatch: dst size mismatch");
  AssertTrue(q_words % block_dim_ == 0,
             "ModUpFromCoeffBatch: degree must be a multiple of the block dim");

  // The product inverse for the matrix product, and the same coefficients in
  // Montgomery form for the limbs that pass through: with one digit written
  // straight into that digit's q limbs, with more digits into a buffer each
  // digit copies its own limbs from. On the conjugate-invariant ring the
  // pass-through copy is written already folded and the conversion output
  // below folds itself, so the transform at the end skips its fold pass --
  // see ModSwitchMatrixMultCi.
  const bool ci = param_.conjugate_invariant_;
  const int tw_ter_left = param_.GetMaxNumTer() - np.num_ter_;
  DeviceVector<word> src_intt(batch * q_words);
  DeviceVector<word> src_mont;
  word *mont = dst.data();
  int mont_stride = ext_words;
  if (beta_ > 1) {
    src_mont.resize(static_cast<size_t>(batch) * q_words);
    mont = src_mont.data();
    mont_stride = q_words;
  }
  if (ci) {
    AssertTrue((q_words / 2) % block_dim_ == 0,
               "ModUpFromCoeffBatch: degree must be a multiple of twice the "
               "block dim");
    auto cic = ntt_handler_.GetCiConstants();
    dim3 mult_grid((q_words / 2) / block_dim_, batch);
    kernel::ModUpMultConstCi<word><<<mult_grid, block_dim_>>>(
        reinterpret_cast<signed_word *>(src_intt.data()), mont,
        src_coeff.data(), param_.GetPrimesPtr(np), param_.GetInvPrimesPtr(np),
        mod_up1_coeff_.data(), mont_r2_.data(), cic.i_units, cic.fwd_twist,
        tw_ter_left, param_.log_degree_, q_words, mont_stride);
  } else {
    dim3 mult_grid(q_words / block_dim_, batch);
    kernel::ModUpMultConst<word><<<mult_grid, block_dim_>>>(
        reinterpret_cast<signed_word *>(src_intt.data()), mont,
        src_coeff.data(), param_.GetPrimesPtr(np), param_.GetInvPrimesPtr(np),
        mod_up1_coeff_.data(), mont_r2_.data(), param_.log_degree_, q_words,
        mont_stride);
  }

  np.num_aux_ = num_aux_;
  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  const int padded_num_q_primes = num_q_primes + prime_offset;

  // Digit-major: digit i of polynomial b is at dst + (i * batch + b) *
  // ext_words. The aux size of each digit's view is chosen so that the
  // limb-offset correction NTTForModUp applies to the auxiliary part is zero:
  // the buffer is contiguous and wants no correction.
  for (int i = 0; i < beta_; i++) {
    int prime_index_start = i * num_aux_;
    int prime_index_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (prime_index_end <= prime_offset) continue;
    prime_index_start = Max(prime_index_start - prime_offset, 0);
    prime_index_end = prime_index_end - prime_offset;
    const int src_len = prime_index_end - prime_index_start;
    const int dst_len = num_q_primes - src_len + num_aux_;
    word *dst_i = dst.data() + static_cast<size_t>(i) * batch * ext_words;

    if (beta_ > 1) {
      cudaMemcpy2DAsync(dst_i + prime_index_start * degree,
                        static_cast<size_t>(ext_words) * sizeof(word),
                        mont + prime_index_start * degree,
                        static_cast<size_t>(q_words) * sizeof(word),
                        static_cast<size_t>(src_len) * degree * sizeof(word),
                        batch, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    }

    dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                  DivCeil(dst_len, kLimbBatching * kNumThreadsY), batch);
    dim3 block_dim(kNumThreadsX, kNumThreadsY);
    int smem_size =
        src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
    smem_size +=
        kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);
    const signed_word *src_ptr = reinterpret_cast<const signed_word *>(
        src_intt.data() + prime_index_start * degree);

    if (ci) {
      auto cic = ntt_handler_.GetCiConstants();
      kernel::ModSwitchMatrixMultCi<word><<<grid_dim, block_dim, smem_size>>>(
          dst_i, primes, inv_primes, src_len, dst_len, prime_index_start,
          prime_index_end, src_ptr, mod_up2_.at(i).data(), param_.log_degree_,
          cic.i_units, cic.fwd_twist, tw_ter_left, num_q_primes,
          param_.GetMaxNumMain() - np.num_main_, ext_words, q_words);
    } else {
      kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
          dst_i, primes, inv_primes, src_len, dst_len, prime_index_start,
          prime_index_end, src_ptr, mod_up2_.at(i).data(), param_.log_degree_,
          ext_words, q_words);
    }

    // The copied limbs are coefficients too, so they take the same forward
    // transform as the rest instead of being skipped.
    const int total_words = batch * ext_words;
    DvView<word> ntt_view(dst_i, total_words, total_words - q_words);
    ntt_handler_.NTTForModUp(ntt_view, np, 0, 0, ntt_view, batch,
                             /*ci_prefolded=*/ci);
  }
}

template <typename word>
void ModSwitchHandler<word>::ModUpWorker(
    std::vector<DvView<word>> &dst, const DvConstView<word> *src,
    const DvConstView<word> *src_coeff) const {
  using signed_word = make_signed_t<word>;
  NPInfo np = param_.LevelToNP(level_, 0);
  int num_q_primes = np.GetNumQ();
  int prime_offset = (level_ == -1 ? 0 : param_.GetMaxNumTer() - np.num_ter_);
  int degree = param_.degree_;
  AssertTrue((src == nullptr) != (src_coeff == nullptr),
             "ModUp: exactly one of the two sources is given");
  const DvConstView<word> &given = (src != nullptr) ? *src : *src_coeff;
  AssertTrue(given.QSize() == num_q_primes * degree, "ModUp src q size mismatch");
  AssertTrue(given.AuxSize() == 0, "ModUp src aux size mismatch");
  AssertTrue(static_cast<int>(dst.size()) == beta_, "ModUp dst size mismatch");

  // Back to the coefficient domain, scaled by the product inverse. A caller
  // that never left it says so and pays one constant multiply instead -- and
  // the limbs that pass through unchanged then come from that same multiply,
  // in Montgomery form, rather than from an NTT the caller had to run.
  const bool ci = param_.conjugate_invariant_;
  const int tw_ter_left = param_.GetMaxNumTer() - np.num_ter_;
  DeviceVector<word> src_intt(num_q_primes * degree);
  DvView<word> src_intt_view = src_intt.View(0, 0);
  DeviceVector<word> src_mont;
  if (src_coeff == nullptr) {
    ntt_handler_.INTTAndMultConst(src_intt_view, np, *src,
                                  mod_up1_.ConstView(0, 0), true);
  } else if (ci) {
    const int num_words = num_q_primes * degree;
    AssertTrue((num_words / 2) % block_dim_ == 0,
               "ModUpFromCoeff: degree must be a multiple of twice the block "
               "dim");
    src_mont.resize(num_words);
    auto cic = ntt_handler_.GetCiConstants();
    kernel::ModUpMultConstCi<word><<<(num_words / 2) / block_dim_,
                                     block_dim_>>>(
        reinterpret_cast<signed_word *>(src_intt.data()), src_mont.data(),
        src_coeff->data(), param_.GetPrimesPtr(np), param_.GetInvPrimesPtr(np),
        mod_up1_coeff_.data(), mont_r2_.data(), cic.i_units, cic.fwd_twist,
        tw_ter_left, param_.log_degree_, num_words, num_words);
  } else {
    const int num_words = num_q_primes * degree;
    AssertTrue(num_words % block_dim_ == 0,
               "ModUpFromCoeff: degree must be a multiple of the block dim");
    src_mont.resize(num_words);
    kernel::ModUpMultConst<word><<<num_words / block_dim_, block_dim_>>>(
        reinterpret_cast<signed_word *>(src_intt.data()), src_mont.data(),
        src_coeff->data(), param_.GetPrimesPtr(np), param_.GetInvPrimesPtr(np),
        mod_up1_coeff_.data(), mont_r2_.data(), param_.log_degree_, num_words,
        num_words);
  }
  const word *pass_through =
      (src_coeff == nullptr) ? src->data() : src_mont.data();

  int padded_num_q_primes = num_q_primes + prime_offset;
  // Do some checks and copy some values from src to dst
  for (int i = 0; i < beta_; i++) {
    // absolute index
    int prime_index_start = i * num_aux_;
    int prime_index_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (prime_index_end <= prime_offset) continue;

    // relative index
    prime_index_start = Max(prime_index_start - prime_offset, 0);
    prime_index_end = prime_index_end - prime_offset;
    int src_len = prime_index_end - prime_index_start;
    DvView<word> &dst_i = dst.at(i);

    AssertTrue(dst_i.AuxSize() == num_aux_ * degree,
               "ModUp dst aux size mismatch");
    AssertTrue(dst_i.QSize() == num_q_primes * degree,
               "ModUp dst q size mismatch");

    // Copy values from src to dst (asynchronously)
    cudaMemcpyAsync(dst_i.data() + prime_index_start * degree,
                    pass_through + prime_index_start * degree,
                    src_len * degree * sizeof(word), cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy);
  }

  np.num_aux_ = num_aux_;
  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);

  for (int i = 0; i < beta_; i++) {
    // absolute index
    int prime_index_start = i * num_aux_;
    int prime_index_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (prime_index_end <= prime_offset) continue;

    // relative index
    prime_index_start = Max(prime_index_start - prime_offset, 0);
    prime_index_end = prime_index_end - prime_offset;
    int src_len = prime_index_end - prime_index_start;
    int dst_len = num_q_primes - src_len + num_aux_;
    DvView<word> &dst_i = dst.at(i);

    dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                  DivCeil(dst_len, kLimbBatching * kNumThreadsY));
    dim3 block_dim(kNumThreadsX, kNumThreadsY);
    // smem_size for bconv_table
    int smem_size =
        src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
    // extra for src
    smem_size +=
        kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);

    const signed_word *src_ptr = reinterpret_cast<const signed_word *>(
        src_intt.data() + prime_index_start * degree);

    if (ci) {
      // The destination rows take the transform's own skip mapping; on the
      // twiddle axis they are the q basis followed by the aux limbs.
      auto cic = ntt_handler_.GetCiConstants();
      kernel::ModSwitchMatrixMultCi<word><<<grid_dim, block_dim, smem_size>>>(
          dst_i.data(), primes, inv_primes, src_len, dst_len,
          prime_index_start, prime_index_end, src_ptr, mod_up2_.at(i).data(),
          param_.log_degree_, cic.i_units, cic.fwd_twist, tw_ter_left,
          num_q_primes, param_.GetMaxNumMain() - np.num_main_);
    } else {
      kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
          dst_i.data(), primes, inv_primes, src_len, dst_len,
          prime_index_start, prime_index_end, src_ptr, mod_up2_.at(i).data(),
          param_.log_degree_);
    }
    // On the coefficient path the copied limbs are coefficients too, so they
    // take the same forward transform as the rest instead of being skipped.
    if (src_coeff == nullptr) {
      ntt_handler_.NTTForModUp(dst_i, np, prime_index_start, prime_index_end,
                               dst_i, 1, /*ci_prefolded=*/ci);
    } else {
      ntt_handler_.NTTForModUp(dst_i, np, 0, 0, dst_i, 1, /*ci_prefolded=*/ci);
    }
  }
}

// The batched ModUp: ModUpWorker's evaluation-domain path with blockIdx.z
// picking the polynomial. The INTT, the base conversion and the forward
// transform all carry a batch dimension already; the pass-through copy of
// the digit's own limbs is one strided memcpy for the whole group.
template <typename word>
void ModSwitchHandler<word>::ModUpBatch(std::vector<DvView<word>> &dst,
                                        const word *src, int src_batch_stride,
                                        int batch) const {
  using signed_word = make_signed_t<word>;
  const bool ci = param_.conjugate_invariant_;
  const int tw_ter_left = param_.GetMaxNumTer() - param_.LevelToNP(level_, 0).num_ter_;
  AssertTrue(batch >= 1, "ModUpBatch: invalid batch");
  AssertTrue(level_ >= 0, "ModUpBatch: the dense-to-sparse level is not batched");
  NPInfo np = param_.LevelToNP(level_, 0);
  const int num_q_primes = np.GetNumQ();
  const int prime_offset = param_.GetMaxNumTer() - np.num_ter_;
  const int degree = param_.degree_;
  const int q_words = num_q_primes * degree;
  const int ext_words = (num_q_primes + num_aux_) * degree;
  AssertTrue(static_cast<int>(dst.size()) == beta_,
             "ModUpBatch: dst size mismatch");
  AssertTrue(src_batch_stride >= q_words || batch == 1,
             "ModUpBatch: the polynomials overlap");

  NvtxScope *_i = new NvtxScope("mub: intt");
  DeviceVector<word> src_intt(batch * q_words);
  {
    DvView<word> intt_view = src_intt.View(0, 0);
    DvConstView<word> first(src, q_words, 0);
    ntt_handler_.INTTAndMultConst(intt_view, np, first, mod_up1_.ConstView(0, 0),
                                  true, batch, src_batch_stride);
  }

  const int padded_num_q_primes = num_q_primes + prime_offset;
  np.num_aux_ = num_aux_;
  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);

  for (int i = 0; i < beta_; i++) {
    int prime_index_start = i * num_aux_;
    int prime_index_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (prime_index_end <= prime_offset) {
      AssertTrue(dst.at(i).TotalSize() == 0,
                 "ModUpBatch: a skipped digit was given a buffer");
      continue;
    }
    prime_index_start = Max(prime_index_start - prime_offset, 0);
    prime_index_end = prime_index_end - prime_offset;
    const int src_len = prime_index_end - prime_index_start;
    const int dst_len = num_q_primes - src_len + num_aux_;
    DvView<word> &dst_i = dst.at(i);
    AssertTrue(dst_i.TotalSize() == batch * ext_words &&
                   dst_i.AuxSize() == dst_i.TotalSize() - q_words,
               "ModUpBatch: a digit buffer has the wrong shape (it is viewed "
               "with aux = total - q so that the transform's limb offset is "
               "zero)");

    if (_i != nullptr) {
      delete _i;
      _i = nullptr;
    }
    NvtxScope _g("mub: digit");
    // The digit's own limbs pass through untouched, one strided copy for the
    // whole group.
    cudaMemcpy2DAsync(dst_i.data() + prime_index_start * degree,
                      static_cast<size_t>(ext_words) * sizeof(word),
                      src + prime_index_start * degree,
                      static_cast<size_t>(src_batch_stride) * sizeof(word),
                      static_cast<size_t>(src_len) * degree * sizeof(word),
                      batch, cudaMemcpyDeviceToDevice, cudaStreamLegacy);

    dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                  DivCeil(dst_len, kLimbBatching * kNumThreadsY), batch);
    dim3 block_dim(kNumThreadsX, kNumThreadsY);
    int smem_size =
        src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
    smem_size +=
        kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);
    const signed_word *src_ptr = reinterpret_cast<const signed_word *>(
        src_intt.data() + prime_index_start * degree);
    if (ci) {
      auto cic = ntt_handler_.GetCiConstants();
      kernel::ModSwitchMatrixMultCi<word><<<grid_dim, block_dim, smem_size>>>(
          dst_i.data(), primes, inv_primes, src_len, dst_len,
          prime_index_start, prime_index_end, src_ptr, mod_up2_.at(i).data(),
          param_.log_degree_, cic.i_units, cic.fwd_twist, tw_ter_left,
          num_q_primes, param_.GetMaxNumMain() - np.num_main_, ext_words,
          q_words);
    } else {
      kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
          dst_i.data(), primes, inv_primes, src_len, dst_len,
          prime_index_start, prime_index_end, src_ptr, mod_up2_.at(i).data(),
          param_.log_degree_, ext_words, q_words);
    }
    ntt_handler_.NTTForModUp(dst_i, np, prime_index_start, prime_index_end,
                             dst_i, batch, /*ci_prefolded=*/ci);
  }
}

// The batched ModDown: ModDownWorker's ModDown case over `batch` polynomials.
template <typename word>
void ModSwitchHandler<word>::ModDownBatch(word *dst, int dst_batch_stride,
                                          const word *src,
                                          int src_batch_stride,
                                          int batch) const {
  using signed_word = make_signed_t<word>;
  const bool ci = param_.conjugate_invariant_;
  AssertTrue(kFuseModDownEpilogue,
             "ModDownBatch: written against the fused epilogue");
  AssertTrue(batch >= 1, "ModDownBatch: invalid batch");
  AssertTrue(level_ >= 0, "ModDownBatch: the dense-to-sparse level is not batched");
  const int degree = param_.degree_;
  const NPInfo np_src = param_.LevelToNP(level_, num_aux_);
  const NPInfo np_dst = param_.LevelToNP(level_, 0);
  const NPInfo np_non_intt(np_dst.num_main_, np_dst.num_ter_, 0,
                           np_src.degree_);
  const int src_len = np_src.GetNumTotal() - np_non_intt.GetNumTotal();
  const int dst_len = np_dst.GetNumTotal();
  const int src_words = np_src.GetNumTotal() * degree;
  const int dst_words = dst_len * degree;
  AssertTrue(batch == 1 || (src_batch_stride >= src_words &&
                            dst_batch_stride >= dst_words),
             "ModDownBatch: the polynomials overlap");

  DeviceVector<word> src_intt(batch * src_len * degree);
  {
    DvView<word> intt_view = src_intt.View(0, 0);
    DvConstView<word> first(src, src_words, num_aux_ * degree);
    ntt_handler_.INTTForModDown(intt_view, np_src, np_non_intt, first,
                                mod_down1_.ConstView(num_aux_), batch,
                                src_batch_stride);
  }

  const word *primes = param_.GetPrimesPtr(np_dst);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np_dst);
  dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                DivCeil(dst_len, kLimbBatching * kNumThreadsY), batch);
  dim3 block_dim(kNumThreadsX, kNumThreadsY);
  int smem_size =
      src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
  smem_size +=
      kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);
  if (ci) {
    auto cic = ntt_handler_.GetCiConstants();
    kernel::ModSwitchMatrixMultCi<word><<<grid_dim, block_dim, smem_size>>>(
        dst, primes, inv_primes, src_len, dst_len, 0, 0,
        reinterpret_cast<const signed_word *>(src_intt.data()),
        mod_down2_.data(), param_.log_degree_, cic.i_units, cic.fwd_twist,
        param_.GetMaxNumTer() - np_dst.num_ter_, dst_len, 0, dst_batch_stride,
        src_len * degree);
  } else {
    kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
        dst, primes, inv_primes, src_len, dst_len, 0, 0,
        reinterpret_cast<const signed_word *>(src_intt.data()),
        mod_down2_.data(), param_.log_degree_, dst_batch_stride,
        src_len * degree);
  }

  // The epilogue: NTT of the converted limbs, then (src - dst) * P^-1 with
  // the q limbs of the source, exactly as ModDownWorker's ModDown case.
  DvView<word> dst_view(dst, (batch - 1) * dst_batch_stride + dst_words, 0);
  DvConstView<word> src2(src, dst_words, 0);
  ntt_handler_.NTTForModDown(dst_view, np_dst, np_non_intt,
                             DvConstView<word>(dst_view), src2,
                             inv_prime_prod_.ConstView(),
                             DvConstView<word>(nullptr, 0),
                             /*ci_prefolded=*/ci, batch, dst_batch_stride,
                             src_batch_stride);
}

template <typename word>
void ModSwitchHandler<word>::RescaleBatch(word *dst, int dst_batch_stride,
                                         const word *src, int src_batch_stride,
                                         int batch) const {
  using signed_word = make_signed_t<word>;
  const bool ci = param_.conjugate_invariant_;
  AssertTrue(kFuseModDownEpilogue,
             "RescaleBatch: written against the fused epilogue");
  AssertTrue(batch >= 1 && level_ >= 1, "RescaleBatch: invalid batch or level");
  const int degree = param_.degree_;
  const NPInfo np_src = param_.LevelToNP(level_, 0);
  const NPInfo np_dst = param_.LevelToNP(level_ - 1, 0);
  const NPInfo np_non_intt(Min(np_src.num_main_, np_dst.num_main_),
                           Min(np_src.num_ter_, np_dst.num_ter_), 0,
                           np_src.degree_);
  const int src_len = np_src.GetNumTotal() - np_non_intt.GetNumTotal();
  const int dst_len = np_dst.GetNumTotal();
  const int src_words = np_src.GetNumTotal() * degree;
  const int dst_words = dst_len * degree;
  AssertTrue(batch == 1 || (src_batch_stride >= src_words &&
                            dst_batch_stride >= dst_words),
             "RescaleBatch: the polynomials overlap");

  // ModDownWorker's Rescale case, every stage over `batch` polynomials.
  DeviceVector<word> src_intt(batch * src_len * degree);
  {
    DvView<word> intt_view = src_intt.View(0, 0);
    DvConstView<word> first(src, src_words, 0);
    ntt_handler_.INTTForModDown(intt_view, np_src, np_non_intt, first,
                                rescale1_.ConstView(), batch,
                                src_batch_stride);
  }

  const word *primes = param_.GetPrimesPtr(np_dst);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np_dst);
  dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                DivCeil(dst_len, kLimbBatching * kNumThreadsY), batch);
  dim3 block_dim(kNumThreadsX, kNumThreadsY);
  int smem_size =
      src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
  smem_size +=
      kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);
  if (ci) {
    auto cic = ntt_handler_.GetCiConstants();
    kernel::ModSwitchMatrixMultCi<word><<<grid_dim, block_dim, smem_size>>>(
        dst, primes, inv_primes, src_len, dst_len, 0, 0,
        reinterpret_cast<const signed_word *>(src_intt.data()),
        rescale2_.data(), param_.log_degree_, cic.i_units, cic.fwd_twist,
        param_.GetMaxNumTer() - np_dst.num_ter_, dst_len, 0, dst_batch_stride,
        src_len * degree);
  } else {
    kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
        dst, primes, inv_primes, src_len, dst_len, 0, 0,
        reinterpret_cast<const signed_word *>(src_intt.data()),
        rescale2_.data(), param_.log_degree_, dst_batch_stride,
        src_len * degree);
  }

  const int pad_len = rescale_pad_end_ - rescale_pad_start_;
  const int src2_offset = Max(0, np_src.num_ter_ - np_dst.num_ter_);
  DvView<word> dst_view(dst, (batch - 1) * dst_batch_stride + dst_words, 0);
  DvConstView<word> src2(src + src2_offset * degree, pad_len * degree, 0);
  ntt_handler_.NTTForModDown(dst_view, np_dst, np_non_intt,
                             DvConstView<word>(dst_view), src2,
                             rescale_inv_prime_prod_.ConstView(),
                             rescale_padding_.ConstView(),
                             /*ci_prefolded=*/ci, batch, dst_batch_stride,
                             src_batch_stride);
}

template <typename word>
void ModSwitchHandler<word>::ModDown(DvView<word> &dst,
                                     const DvConstView<word> &src) const {
  int degree = param_.degree_;
  NPInfo np = param_.LevelToNP(level_, num_aux_);
  int num_q_primes = np.GetNumQ();

  AssertTrue(src.AuxSize() == np.num_aux_ * degree,
             "ModDown: src aux size mismatch");
  AssertTrue(src.QSize() == num_q_primes * degree,
             "ModDown: src q size mismatch");
  AssertTrue(dst.AuxSize() == 0, "ModDown: dst aux size mismatch");
  AssertTrue(dst.QSize() == num_q_primes * degree,
             "ModDown: dst q size mismatch");

  ModDownWorker(dst, src, ModDownType::ModDown);
}

template <typename word>
void ModSwitchHandler<word>::Rescale(DvView<word> &dst,
                                     const DvConstView<word> &src) const {
  int degree = param_.degree_;
  NPInfo np = param_.LevelToNP(level_, num_aux_);
  NPInfo next_np = param_.LevelToNP(level_ - 1);
  int num_q_primes = np.GetNumQ();

  AssertTrue(src.AuxSize() == 0, "ModDown: src aux size mismatch");
  AssertTrue(src.QSize() == num_q_primes * degree,
             "ModDown: src q size mismatch");
  AssertTrue(dst.AuxSize() == 0, "ModDown: dst aux size mismatch");
  AssertTrue(dst.QSize() == next_np.GetNumQ() * degree,
             "ModDown: dst q size mismatch");

  ModDownWorker(dst, src, ModDownType::Rescale);
}

template <typename word>
void ModSwitchHandler<word>::ModDownAndRescale(
    DvView<word> &dst, const DvConstView<word> &src) const {
  int degree = param_.degree_;
  NPInfo np = param_.LevelToNP(level_, num_aux_);
  NPInfo next_np = param_.LevelToNP(level_ - 1);
  int num_q_primes = np.GetNumQ();

  AssertTrue(src.AuxSize() == np.num_aux_ * degree,
             "ModDown: src aux size mismatch");
  AssertTrue(src.QSize() == num_q_primes * degree,
             "ModDown: src q size mismatch");
  AssertTrue(dst.AuxSize() == 0, "ModDown: dst aux size mismatch");
  AssertTrue(dst.QSize() == next_np.GetNumQ() * degree,
             "ModDown: dst q size mismatch");

  ModDownWorker(dst, src, ModDownType::ModDownAndRescale);
}

template <typename word>
void ModSwitchHandler<word>::ModDownWorker(DvView<word> &dst,
                                           const DvConstView<word> &src,
                                           ModDownType type) const {
  using signed_word = make_signed_t<word>;
  int degree = param_.degree_;
  int num_src_aux = (type == ModDownType::Rescale ? 0 : num_aux_);
  NPInfo np_src = param_.LevelToNP(level_, num_src_aux);
  NPInfo np_dst =
      param_.LevelToNP((type == ModDownType::ModDown ? level_ : level_ - 1), 0);
  NPInfo np_non_intt(Min(np_src.num_main_, np_dst.num_main_),
                     Min(np_src.num_ter_, np_dst.num_ter_), 0, np_src.degree_);

  int src_len = np_src.GetNumTotal() - np_non_intt.GetNumTotal();
  int dst_len = np_dst.GetNumTotal();

  DeviceVector<word> src_extend_temp;
  if (!kFuseModDownEpilogue && type != ModDownType::ModDown) {
    src_extend_temp.resize(src.TotalSize());
    std::vector<DvView<word>> src_extend_temp_view{
        src_extend_temp.View(src.AuxSize(), 0)};
    DvConstView<word> padding_view(entire_padding_.data(), np_src.GetNumTotal(),
                                   num_src_aux);
    elem_handler_.MultConst(src_extend_temp_view, np_src, {src}, padding_view);
  }
  DvConstView<word> src_view =
      (!kFuseModDownEpilogue && type != ModDownType::ModDown)
          ? src_extend_temp.ConstView(src.AuxSize())
          : src;

  // Performing INTTForModDown
  DeviceVector<word> src_intt(src_len * degree);

  DvView<word> src_intt_view = src_intt.View(0, 0);
  DvConstView<word> const1 =
      (type == ModDownType::ModDown
           ? mod_down1_.ConstView(num_src_aux)
           : (type == ModDownType::Rescale
                  ? rescale1_.ConstView()
                  : mod_down_rescale1_.ConstView(num_src_aux)));
  // On the conjugate-invariant ring, with the fused epilogue, the conversion
  // below folds its output in registers on the way out and the forward
  // transform starts straight at phase 1; the INTT keeps its own unfold
  // pass. Without the fused epilogue the plain-NTT fallback folds for
  // itself, so everything keeps its own passes.
  const bool ci = param_.conjugate_invariant_ && kFuseModDownEpilogue;
  ntt_handler_.INTTForModDown(src_intt_view, np_src, np_non_intt, src_view,
                              const1);

  const word *primes = param_.GetPrimesPtr(np_dst);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np_dst);

  const signed_word *src_ptr =
      reinterpret_cast<const signed_word *>(src_intt.data());
  const signed_word *bconv_table =
      (type == ModDownType::ModDown
           ? mod_down2_.data()
           : (type == ModDownType::Rescale ? rescale2_.data()
                                           : mod_down_rescale2_.data()));

  // Do matrix multiplication
  dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                DivCeil(dst_len, kLimbBatching * kNumThreadsY));
  dim3 block_dim(kNumThreadsX, kNumThreadsY);
  // smem_size for bconv_table
  int smem_size =
      src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
  // extra for src
  smem_size +=
      kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);

  if (ci) {
    // The destination rows are the plain q basis of np_dst; the source
    // arrives raw from the INTT and only the fold on the way out is carried
    // here.
    auto cic = ntt_handler_.GetCiConstants();
    kernel::ModSwitchMatrixMultCi<word><<<grid_dim, block_dim, smem_size>>>(
        dst.data(), primes, inv_primes, src_len, dst_len, 0, 0, src_ptr,
        bconv_table, param_.log_degree_, cic.i_units, cic.fwd_twist,
        param_.GetMaxNumTer() - np_dst.num_ter_, dst_len, 0);
  } else {
    kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
        dst.data(), primes, inv_primes, src_len, dst_len, 0, 0, src_ptr,
        bconv_table, param_.log_degree_);
  }

  // Prepare the constants for ModDownEpilogue
  int pad_start = 0;
  int pad_end = dst_len;
  // rescale case
  if (type != ModDownType::ModDown) {
    pad_start = rescale_pad_start_;
    pad_end = rescale_pad_end_;
  }
  int pad_len = pad_end - pad_start;

  DvConstView<word> inv_prime_prod =
      (type == ModDownType::ModDown
           ? inv_prime_prod_.ConstView()
           : (type == ModDownType::Rescale
                  ? rescale_inv_prime_prod_.ConstView()
                  : mod_down_rescale_inv_prime_prod_.ConstView()));
  DvConstView<word> src2_padding(
      (type == ModDownType::ModDown
           ? DvConstView<word>(nullptr, 0, 0)
           : (type == ModDownType::Rescale
                  ? rescale_padding_.ConstView()
                  : mod_down_rescale_padding_.ConstView())));

  // Prepared offseted src2
  int src2_offset = Max(0, np_src.num_ter_ - np_dst.num_ter_);
  DvConstView<word> src2(src_view.data() + src2_offset * degree,
                         pad_len * degree, 0);

  if (kFuseModDownEpilogue) {
    ntt_handler_.NTTForModDown(dst, np_dst, np_non_intt, DvConstView<word>(dst),
                               src2, inv_prime_prod, src2_padding,
                               /*ci_prefolded=*/ci);

  } else {
    ntt_handler_.NTT(dst, np_dst, DvConstView<word>(dst), false);

    // (src - dst) * inv_prime_prod;
    std::vector<DvView<word>> dst_offset_view{
        DvView<word>(dst.data() + pad_start * degree, pad_len * degree)};
    std::vector<DvConstView<word>> dst_offset_const_view{
        DvConstView<word>(dst.data() + pad_start * degree, pad_len * degree)};
    std::vector<DvView<word>> dst_view{dst};

    elem_handler_.Neg(dst_view, np_dst, {DvConstView<word>(dst)});
    elem_handler_.Add(dst_offset_view, np_non_intt, {src2},
                      dst_offset_const_view);

    elem_handler_.MultConst(dst_view, np_dst, {DvConstView<word>(dst)},
                            inv_prime_prod);
  }
}

// explicit instantiation
template class ModSwitchHandler<uint32_t>;
template class ModSwitchHandler<uint64_t>;

}  // namespace cheddar