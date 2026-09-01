#ifdef USE_CUBLAS

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"
#include "core/BigInt.h"
#include "core/PcmmBlas.h"

namespace cheddar {

// Tile width of the transposing gather. 32 makes both halves warp-wide, and
// the launch on the host side needs it too.
constexpr int kTile = 32;

namespace kernel {

// Gather one RNS limb of every source ciphertext into a contiguous
// degree x cols matrix and split it into int8 pieces at the same time.
//
// The gather is what lets a GEMM run at all: the sources are separate
// allocations, and cuBLAS wants one matrix. It is a bandwidth-bound copy of
// cols * degree words, against a product that is rows times larger.
//
// It writes **column-index-contiguous**, `[x][j]` rather than `[j][x]`, which
// is what puts the GEMM on TN and so on the tensor core -- see the layout
// section of the class comment; it is a 5.5x difference and it is the reason
// this kernel is a transpose and not a copy. Neither direction is coalesced on
// its own, so the tile goes through shared memory: reads run along x, which is
// contiguous in each source, and writes run along j, which is contiguous in the
// destination.
// Digit `p` of `r` in BALANCED base 2^piece_bits, i.e. in [-2^(b-1), 2^(b-1)):
// add half a digit at every position, take the plain digit, subtract the
// half back. Exact and branch-free; the biased value needs one bit more than
// `r`, which `PiecesFor` allows for. The host split, the residue split and
// the gather must all take this same digit, so it is stated once.
template <typename word>
__host__ __device__ inline int8_t BalancedDigit(word r, int p, int piece_bits) {
  const uint64_t half = static_cast<uint64_t>(1) << (piece_bits - 1);
  uint64_t bias = 0;
  for (int i = 0; i < 8; i++) bias |= half << (piece_bits * i);
  const uint64_t biased = static_cast<uint64_t>(r) + bias;
  const uint64_t mask = (static_cast<uint64_t>(1) << piece_bits) - 1;
  return static_cast<int8_t>(
      static_cast<int>((biased >> (piece_bits * p)) & mask) -
      static_cast<int>(half));
}

template <typename word>
__global__ void SplitGather(int8_t *dst, const word *const *src_ptrs, int cols,
                            int degree, int limb_offset, int pieces,
                            int piece_bits) {
  // +1 so the column-wise read below does not hit one bank.
  __shared__ word tile[kTile][kTile + 1];
  const size_t n = static_cast<size_t>(cols) * degree;
  const int j0 = blockIdx.x * kTile;
  const int x0 = blockIdx.y * kTile;
  const int tx = threadIdx.x;  // 0 .. kTile - 1
  const int ty = threadIdx.y;  // 0 .. blockDim.y - 1

  for (int i = ty; i < kTile; i += blockDim.y) {
    const int j = j0 + i, x = x0 + tx;
    if (j < cols && x < degree) tile[i][tx] = src_ptrs[j][limb_offset + x];
  }
  __syncthreads();

  for (int i = ty; i < kTile; i += blockDim.y) {
    const int x = x0 + i, j = j0 + tx;
    if (x >= degree || j >= cols) continue;
    const word v = tile[tx][i];
    const size_t o = static_cast<size_t>(x) * cols + j;
    for (int p = 0; p < pieces; p++) {
      dst[static_cast<size_t>(p) * n + o] = BalancedDigit(v, p, piece_bits);
    }
  }
}

// One column of `SplitGather`'s layout: `dst[(p * degree + x) * cols + col]`
// for every position x of the limb, from one ciphertext component.
template <typename word>
__global__ void SplitGatherColumn(int8_t *dst, const word *src, int cols,
                                  int col, int degree, int limb_offset,
                                  int pieces, int piece_bits) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= degree) return;
  const word v = src[limb_offset + x];
  const size_t n = static_cast<size_t>(cols) * degree;
  const size_t o = static_cast<size_t>(x) * cols + col;
  for (int p = 0; p < pieces; p++) {
    dst[static_cast<size_t>(p) * n + o] = BalancedDigit(v, p, piece_bits);
  }
}

// Recombine the shift groups modulo the prime.
//
// Group t holds the sum of every piece product whose combined shift is
// 7 * t, accumulated in int32 by cuBLAS. Each group is at most
// 127 * 127 * cols which stays inside int32; the recombination is where the
// modular arithmetic happens, once per output element rather than once per
// product, which is why the groups exist.
//
// `shift_pow` arrives in Montgomery form and the reduction is Montgomery, not
// a 64-bit remainder. That is not a micro-optimisation: the plain version ran
// two `%` on 64-bit operands per group per output word, and this kernel writes
// 100M words per `Multiply` with 16 groups behind each -- 3.2e9 integer
// divisions, against a GEMM that takes 30 ms. `MultMontgomery` needs only
// `a * b` to stay inside `q * 2^31` (Basic.cuh:102), NOT `a < q`: the cuBLAS
// output is bounded by 127 * 127 * cols, so `cols < 2^31 / 16129` is the whole
// condition and the caller asserts it.
template <typename word>
__global__ void CombineGroups(word *const *dst_ptrs, const int32_t *groups,
                              int num_groups, int rows, int degree,
                              int limb_offset, const word *prime_ptr,
                              const make_signed_t<word> *inv_prime_ptr,
                              const word *shift_pow) {
  // Every row at once. Launching this per output row -- 128 rows x 3 primes x
  // 2 components = 768 launches of 4096 elements each -- put about 23 ms of
  // pure launch overhead on a product whose GEMMs take 2.2 ms. The destinations
  // are separate allocations, so they arrive as a pointer array, exactly as
  // PcmmAccum takes them.
  const size_t n = static_cast<size_t>(rows) * degree;
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const word prime = basic::StreamingLoadConst(prime_ptr);
  const make_signed_t<word> inv_prime = basic::StreamingLoadConst(inv_prime_ptr);
  const int row = static_cast<int>(idx / degree);
  const int x = static_cast<int>(idx % degree);
  // Balanced digits make a group sum signed: a negative one is lifted by the
  // multiple of the prime the host put behind the shifts, and the
  // Montgomery product stays in bounds either way (ProductComponent).
  const int64_t lift =
      static_cast<int64_t>(basic::StreamingLoadConst(shift_pow + num_groups));
  word acc = 0;
  for (int t = 0; t < num_groups; t++) {
    const int64_t g = static_cast<int64_t>(groups[t * n + idx]);
    const word part = static_cast<word>(g < 0 ? g + lift : g);
    const word shift = basic::StreamingLoadConst(shift_pow + t);
    acc = basic::Add(acc, basic::MultMontgomery(part, shift, prime, inv_prime),
                     prime);
  }
  dst_ptrs[row][limb_offset + x] = acc;
}

}  // namespace kernel

template <typename word>
PcmmBlasHandler<word>::PcmmBlasHandler(const Parameter<word> &param)
    : param_{param} {
  AssertTrue(cublasCreate(&handle_) == CUBLAS_STATUS_SUCCESS,
             "PcmmBlas: cublasCreate failed");
}

template <typename word>
PcmmBlasHandler<word>::~PcmmBlasHandler() {
  if (handle_ != nullptr) cublasDestroy(handle_);
}

template <typename word>
void PcmmBlasHandler<word>::SplitMatrixFrom(SplitMatrix &res, int level,
                                            double scale,
                                            const std::vector<double> &values,
                                            int rows, int cols,
                                            int num_aux /*= 0*/) const {
  AssertTrue(rows > 0 && cols > 0, "PcmmBlas: invalid matrix shape");
  AssertTrue(values.size() == static_cast<size_t>(rows) * cols,
             "PcmmBlas: values size does not match rows * cols");

  NPInfo np = param_.LevelToNP(level, num_aux);
  auto primes = param_.GetPrimeVector(np);
  const int num_primes = np.GetNumTotal();

  // Enough 7-bit pieces to cover the widest prime in this basis.
  int max_bits = 1;
  for (int j = 0; j < num_primes; j++) {
    int b = 0;
    for (word p = primes[j]; p != 0; p >>= 1) b++;
    max_bits = b > max_bits ? b : max_bits;
  }
  const int pieces = PiecesFor(max_bits);

  const size_t per_prime = static_cast<size_t>(rows) * cols;
  HostVector<int8_t> host(static_cast<size_t>(pieces) * num_primes * per_prime);

  std::vector<BigInt> big_primes;
  for (int j = 0; j < num_primes; j++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[j]));
  }

  // Plain residues, not Montgomery: the GEMM multiplies raw values, and
  // PcmmAccum's Montgomery form exists only so its per-product reduction is
  // cheap. Both end up writing the same plain residues, which is what makes
  // the two paths comparable bit for bit.
  BigInt residue(static_cast<uint64_t>(0));
  for (size_t i = 0; i < per_prime; i++) {
    BigInt value(std::round(values[i] * scale));
    for (int j = 0; j < num_primes; j++) {
      BigInt::Mod(residue, value, big_primes[j]);
      const word r = static_cast<word>(residue.GetUnsigned());
      for (int p = 0; p < pieces; p++) {
        host[(static_cast<size_t>(p) * num_primes + j) * per_prime + i] =
            kernel::BalancedDigit(r, p, kPieceBits);
      }
    }
  }

  res.rows = rows;
  res.cols = cols;
  res.pieces = pieces;
  res.scale = scale;
  res.np = np;
  res.data.resize(static_cast<int>(host.size()));
  CopyHostToDevice(res.data, host);
}

namespace kernel {
// Prime-major residues in, piece-major pieces out: the layout
// `SplitMatrixFrom` writes from the host, produced here from
// `GpuEncoder::EncodeResiduesGathered`'s buffer. Flat and memory bound.
template <typename word>
__global__ void SplitResidues(int8_t *dst, const word *src, size_t per_prime,
                              int num_primes, int pieces, int piece_bits) {
  const size_t n = per_prime * num_primes;
  const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const word r = src[k];
  for (int p = 0; p < pieces; p++) {
    dst[static_cast<size_t>(p) * n + k] = BalancedDigit(r, p, piece_bits);
  }
}
}  // namespace kernel

template <typename word>
void PcmmBlasHandler<word>::DescribeSplit(SplitMatrix &res, int level,
                                          double scale, int rows, int cols,
                                          int num_aux /*= 0*/) const {
  AssertTrue(rows > 0 && cols > 0, "PcmmBlas: invalid matrix shape");
  NPInfo np = param_.LevelToNP(level, num_aux);
  auto primes = param_.GetPrimeVector(np);
  const int num_primes = np.GetNumTotal();
  int max_bits = 1;
  for (int j = 0; j < num_primes; j++) {
    int b = 0;
    for (word p = primes[j]; p != 0; p >>= 1) b++;
    max_bits = b > max_bits ? b : max_bits;
  }
  res.rows = rows;
  res.cols = cols;
  res.pieces = PiecesFor(max_bits);
  res.scale = scale;
  res.np = np;
  res.view = nullptr;
}

template <typename word>
void PcmmBlasHandler<word>::SplitMatrixFromResidues(
    SplitMatrix &res, int level, double scale, const word *residues, int rows,
    int cols, int num_aux /*= 0*/) const {
  DescribeSplit(res, level, scale, rows, cols, num_aux);
  const size_t per_prime = static_cast<size_t>(rows) * cols;
  const size_t n = per_prime * res.np.GetNumTotal();
  res.data.resize(static_cast<int>(static_cast<size_t>(res.pieces) * n));
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  kernel::SplitResidues<word><<<grid, block>>>(res.data.data(), residues,
                                               per_prime, res.np.GetNumTotal(),
                                               res.pieces, kPieceBits);
}

template <typename word>
void PcmmBlasHandler<word>::SplitResiduesInto(SplitMatrix &res, int level,
                                              double scale,
                                              const word *residues, int rows,
                                              int cols, int8_t *dst,
                                              cudaStream_t stream,
                                              int num_aux /*= 0*/) const {
  DescribeSplit(res, level, scale, rows, cols, num_aux);
  const size_t per_prime = static_cast<size_t>(rows) * cols;
  const size_t n = per_prime * res.np.GetNumTotal();
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  kernel::SplitResidues<word><<<grid, block, 0, stream>>>(
      dst, residues, per_prime, res.np.GetNumTotal(), res.pieces, kPieceBits);
}

namespace {
// The gather and the recombination are both flat; 256 is what every other
// elementwise kernel in the library uses.
constexpr int kBlock = 256;
}  // namespace

template <typename word>
int PcmmBlasHandler<word>::ChunkFor(int cols, int rows, int pieces,
                                    int vec_len) const {
  const int num_groups = pieces * pieces;
  // Chunk boundaries stay aligned to the flat kernels' block, except when the
  // whole component is shorter than one block -- the b-part at the conjugate-
  // invariant projection degree of 128 (Doing.md 1.5bh) -- where the single
  // chunk is the component itself. Both lengths are powers of two, so the
  // smaller of the two divides the larger.
  const int align = std::min(kBlock, vec_len);
  AssertTrue(vec_len % align == 0,
             "PcmmBlas: the component length must be a multiple of the block "
             "dim");

  // Both buffers are indexed by int, because DeviceVector is, and the a-part of
  // a wide tile passes 2^31 words long before it runs out of card: 56 parents
  // at rank 256 is 14336 columns of 65536 words per limb, which is 3.8e9 int8
  // pieces. So the product is chunked along the vector axis -- valid without
  // further thought, since the product is a scalar combination and never mixes
  // positions within a limb.
  constexpr size_t kIndexMax = (static_cast<size_t>(1) << 31) - 1;
  size_t chunk =
      std::min(kIndexMax / (static_cast<size_t>(pieces) * cols),
               kIndexMax / (static_cast<size_t>(num_groups) * rows));

  // The split of the source is no longer scratch -- it is held for the whole
  // tile and its total size is `pieces * cols * vec_len` however it is cut --
  // so the budget now bounds only the int32 accumulator, which is memory the
  // product does not otherwise need. At the layer's shape 2^29 leaves the
  // a-part in a single chunk, which is both the fewest launches and the widest
  // `m` the GEMM can be given.
  static const size_t kScratchWords = [] {
    const char *v = std::getenv("CHEDDAR_PCMM_SCRATCH_LOG2");
    int bits = v != nullptr ? std::atoi(v) : 29;
    if (bits < 20) bits = 20;
    if (bits > 34) bits = 34;
    return static_cast<size_t>(1) << bits;
  }();
  chunk = std::min(chunk,
                   kScratchWords / (static_cast<size_t>(num_groups) * rows));
  chunk = std::min(chunk, static_cast<size_t>(vec_len));
  chunk -= chunk % align;
  AssertTrue(chunk >= static_cast<size_t>(align),
             "PcmmBlas: the operand is too wide for the scratch budget");
  return static_cast<int>(chunk);
}

template <typename word>
void PcmmBlasHandler<word>::SplitComponent(
    std::vector<DeviceVector<int8_t>> &res, const word *const *src_ptrs,
    int cols, int pieces, const NPInfo &np, int vec_len, int chunk) const {
  const int num_primes = np.GetNumTotal();
  const int num_chunks = (vec_len + chunk - 1) / chunk;
  res.clear();
  res.resize(static_cast<size_t>(num_primes) * num_chunks);

  // 32 x 32 words per block, eight rows of it in flight at a time.
  const dim3 block(kTile, 8);
  for (int j = 0; j < num_primes; j++) {
    for (int c = 0; c < num_chunks; c++) {
      const size_t off = static_cast<size_t>(c) * chunk;
      const int span = static_cast<int>(
          std::min(static_cast<size_t>(chunk), vec_len - off));
      const int limb_offset =
          static_cast<int>(j * static_cast<size_t>(vec_len) + off);
      const size_t words = static_cast<size_t>(pieces) * cols * span;
      AssertTrue(words < (static_cast<size_t>(1) << 31),
                 "PcmmBlas: the split of one chunk does not fit an int index");
      auto &buf = res[static_cast<size_t>(j) * num_chunks + c];
      buf.resize(static_cast<int>(words));
      const dim3 grid((cols + kTile - 1) / kTile, (span + kTile - 1) / kTile);
      kernel::SplitGather<word><<<grid, block>>>(buf.data(), src_ptrs, cols,
                                                 span, limb_offset, pieces,
                                                 kPieceBits);
    }
  }
}

template <typename word>
void PcmmBlasHandler<word>::ProductComponent(
    word *const *dst_ptrs, const std::vector<DeviceVector<int8_t>> &split,
    const SplitMatrix &u, const NPInfo &np, int vec_len, int chunk) const {
  const int rows = u.rows, cols = u.cols, pieces = u.pieces;
  const int num_primes = np.GetNumTotal();
  const int num_groups = pieces * pieces;
  const int num_chunks = (vec_len + chunk - 1) / chunk;
  AssertTrue(split.size() == static_cast<size_t>(num_primes) * num_chunks,
             "PcmmBlas: the split source was cut for a different shape");
  // A cuBLAS group entry is within +-128 * 128 * cols. It has to fit int32
  // for the GEMM at all, and CombineGroups lifts a negative one by the
  // smallest multiple of the prime above 2^28 (`kLiftBound`), so the lifted
  // value is below 2^28 + p < 2^31 and its Montgomery product with a shift
  // (< p) stays inside q * 2^31 (Basic.cuh:102).
  constexpr uint64_t kLiftBound = static_cast<uint64_t>(1) << 28;
  AssertTrue(static_cast<uint64_t>(16384) * cols <= kLiftBound,
             "PcmmBlas: too many columns for the signed int32 accumulator");

  DeviceVector<int32_t> groups(
      static_cast<int>(static_cast<size_t>(num_groups) * rows * chunk));

  const auto primes = param_.GetPrimeVector(np);
  const word *prime_ptr = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_prime_ptr = param_.GetInvPrimesPtr(np);
  const int one = 1, zero = 0;

  for (int j = 0; j < num_primes; j++) {
    // 2^(8*(k+l)) mod p, laid out to match the (l * pieces + k) buffer order,
    // and behind them the lift: the smallest multiple of p at or above
    // `kLiftBound`, which brings any group sum (>= -2^28) to a non-negative
    // value below 2^28 + p.
    HostVector<word> h_shift(num_groups + 1);
    {
      const uint64_t p = primes[j];
      std::vector<uint64_t> pow(2 * pieces - 1);
      uint64_t acc = 1;
      for (int s = 0; s < 2 * pieces - 1; s++) {
        pow[s] = acc;
        for (int b2 = 0; b2 < kPieceBits; b2++) acc = (acc * 2) % p;
      }
      for (int l = 0; l < pieces; l++) {
        for (int k = 0; k < pieces; k++) {
          h_shift[l * pieces + k] = primeutil::ToMontgomery(
              static_cast<word>(pow[k + l]), primes[j]);
        }
      }
      h_shift[num_groups] = static_cast<word>(((kLiftBound + p - 1) / p) * p);
    }
    DeviceVector<word> d_shift(num_groups + 1);
    CopyHostToDevice(d_shift, h_shift);

    for (int c = 0; c < num_chunks; c++) {
      const size_t off = static_cast<size_t>(c) * chunk;
      const int span = static_cast<int>(
          std::min(static_cast<size_t>(chunk), vec_len - off));
      const int limb_offset =
          static_cast<int>(j * static_cast<size_t>(vec_len) + off);
      const size_t src_span = static_cast<size_t>(cols) * span;
      const size_t dst_span = static_cast<size_t>(rows) * span;
      const int8_t *src =
          split[static_cast<size_t>(j) * num_chunks + c].data();

      for (int l = 0; l < pieces; l++) {
        const int8_t *b = src + static_cast<size_t>(l) * src_span;
        for (int k = 0; k < pieces; k++) {
          const int8_t *a = u.Ptr() +
                            (static_cast<size_t>(k) * num_primes + j) *
                                static_cast<size_t>(rows) * cols;
          int32_t *c_out =
              groups.data() + static_cast<size_t>(l * pieces + k) * dst_span;
          // TN, which is the only one of the four layouts that reaches the
          // tensor core -- see the class comment. `b` is the source split
          // stored column-index-contiguous, so as a column-major `cols x span`
          // matrix it is A for OP_T; `a` is U row-major, so as a column-major
          // `cols x rows` matrix it is B for OP_N. Both have the reduction
          // index contiguous, which is the whole condition.
          //
          // One buffer per (k, l) and beta = 0. Accumulating several GEMMs into
          // one buffer with beta = 1 was the first thing to suspect when 0.46%
          // of the words came out wrong: cuBLAS's integer path does not
          // guarantee read-modify-write on C the way the floating-point one
          // does. Each product now lands in its own slab and the recombination
          // kernel does all of the summing.
          const cublasStatus_t st = cublasGemmEx(
              handle_, CUBLAS_OP_T, CUBLAS_OP_N, span, rows, cols, &one, b,
              CUDA_R_8I, cols, a, CUDA_R_8I, cols, &zero, c_out, CUDA_R_32I,
              span, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
          AssertTrue(st == CUBLAS_STATUS_SUCCESS, "PcmmBlas: GemmEx failed");
        }
      }

      kernel::CombineGroups<word>
          <<<static_cast<int>((dst_span + kBlock - 1) / kBlock), kBlock>>>(
              dst_ptrs, groups.data(), num_groups, rows, span, limb_offset,
              prime_ptr + j, inv_prime_ptr + j, d_shift.data());
    }
  }
}

template <typename word>
void PcmmBlasHandler<word>::PrepareSource(SplitSource &res,
                                          const SplitMatrix &u,
                                          const std::vector<Ct> &cts,
                                          int rows /*= 0*/) const {
  const int cols = static_cast<int>(cts.size());
  AssertTrue(cols > 0, "PcmmBlas::PrepareSource: no inputs");
  AssertTrue(cols == u.cols,
             "PcmmBlas::PrepareSource: cts size must equal u.cols");
  if (rows <= 0) rows = u.rows;
  AssertTrue(rows >= u.rows,
             "PcmmBlas::PrepareSource: the split must be cut for the largest "
             "tile it will meet");

  const NPInfo np = cts[0].GetNP();
  const int degree = param_.degree_;
  AssertTrue(np.num_aux_ == 0,
             "PcmmBlas::PrepareSource: aux primes unsupported");
  AssertTrue(np == u.np,
             "PcmmBlas::PrepareSource: NP mismatch between u and cts");
  for (const auto &c : cts) {
    AssertTrue(c.GetNP() == np, "PcmmBlas::PrepareSource: mixed NP");
    AssertFalse(c.HasRx(), "PcmmBlas::PrepareSource: input has rx");
  }

  res.cols = cols;
  res.pieces = u.pieces;
  res.rank = 1;
  res.degree = degree;
  res.rows = rows;
  res.num_slots = cts[0].GetNumSlots();
  res.scale = cts[0].GetScale();
  res.np = np;
  // Both components are one ring polynomial per limb, so one chunk serves
  // both -- the same cut the single-shot product made.
  res.chunk_b = ChunkFor(cols, rows, u.pieces, degree);
  res.chunk_a = res.chunk_b;

  HostVector<word *> h_src_bx(cols), h_src_ax(cols);
  for (int j = 0; j < cols; j++) {
    h_src_bx[j] = const_cast<word *>(cts[j].bx_.data());
    h_src_ax[j] = const_cast<word *>(cts[j].ax_.data());
  }
  DeviceVector<word *> d_src_bx(cols), d_src_ax(cols);
  CopyHostToDevice(d_src_bx, h_src_bx);
  CopyHostToDevice(d_src_ax, h_src_ax);

  SplitComponent(res.b_data, d_src_bx.data(), cols, u.pieces, np, degree,
                 res.chunk_b);
  SplitComponent(res.a_data, d_src_ax.data(), cols, u.pieces, np, degree,
                 res.chunk_a);
}

template <typename word>
void PcmmBlasHandler<word>::Multiply(std::vector<Ct> &res,
                                     const SplitMatrix &u,
                                     const SplitSource &src) const {
  const int rows = u.rows, cols = u.cols;
  AssertTrue(rows > 0 && cols > 0, "PcmmBlas::Multiply: bad shape");
  AssertTrue(src.rank == 1 && src.degree == param_.degree_,
             "PcmmBlas::Multiply: the split source is not an RLWE one");
  AssertTrue(cols == src.cols && u.pieces == src.pieces && u.np == src.np,
             "PcmmBlas::Multiply: the split source was prepared for a "
             "different matrix");
  AssertTrue(rows <= src.rows,
             "PcmmBlas::Multiply: the split source was cut for fewer rows "
             "than this tile has");
  AssertTrue(ChunkFor(cols, src.rows, u.pieces, src.degree) == src.chunk_b,
             "PcmmBlas::Multiply: the split source was cut for a different "
             "row count");

  const NPInfo np = src.np;
  const int degree = src.degree;
  res.clear();
  res.resize(rows);
  for (auto &r : res) {
    r.ModifyNP(np);
    r.RemoveRx();
    r.SetNumSlots(src.num_slots);
    r.SetScale(u.scale * src.scale);
  }

  HostVector<word *> h_dst_bx(rows), h_dst_ax(rows);
  for (int i = 0; i < rows; i++) {
    h_dst_bx[i] = res[i].bx_.data();
    h_dst_ax[i] = res[i].ax_.data();
  }
  DeviceVector<word *> d_dst_bx(rows), d_dst_ax(rows);
  CopyHostToDevice(d_dst_bx, h_dst_bx);
  CopyHostToDevice(d_dst_ax, h_dst_ax);

  ProductComponent(d_dst_bx.data(), src.b_data, u, np, degree, src.chunk_b);
  ProductComponent(d_dst_ax.data(), src.a_data, u, np, degree, src.chunk_a);
}

template <typename word>
void PcmmBlasHandler<word>::PrepareSourceBegin(SplitSource &res, int level,
                                               int cols, int rows,
                                               double scale,
                                               int num_slots) const {
  AssertTrue(cols > 0 && rows > 0,
             "PcmmBlas::PrepareSourceBegin: bad shape");
  SplitMatrix shape;
  DescribeSplit(shape, level, scale, rows, cols);
  const int degree = param_.degree_;
  res.cols = cols;
  res.pieces = shape.pieces;
  res.rank = 1;
  res.degree = degree;
  res.rows = rows;
  res.num_slots = num_slots;
  res.scale = scale;
  res.np = shape.np;
  res.chunk_b = ChunkFor(cols, rows, res.pieces, degree);
  res.chunk_a = res.chunk_b;
  const int num_primes = res.np.GetNumTotal();
  const int num_chunks = (degree + res.chunk_b - 1) / res.chunk_b;
  auto size = [&](std::vector<DeviceVector<int8_t>> &bufs) {
    bufs.clear();
    bufs.resize(static_cast<size_t>(num_primes) * num_chunks);
    for (int j = 0; j < num_primes; j++) {
      for (int c = 0; c < num_chunks; c++) {
        const size_t off = static_cast<size_t>(c) * res.chunk_b;
        const int span = static_cast<int>(
            std::min(static_cast<size_t>(res.chunk_b), degree - off));
        const size_t words = static_cast<size_t>(res.pieces) * cols * span;
        AssertTrue(words < (static_cast<size_t>(1) << 31),
                   "PcmmBlas: the split of one chunk does not fit an int "
                   "index");
        bufs[static_cast<size_t>(j) * num_chunks + c].resize(
            static_cast<int>(words));
      }
    }
  };
  size(res.b_data);
  size(res.a_data);
}

template <typename word>
void PcmmBlasHandler<word>::SplitSourceColumn(SplitSource &res, int col,
                                              const Ct &ct) const {
  AssertTrue(res.rank == 1 && res.degree == param_.degree_,
             "PcmmBlas::SplitSourceColumn: not an RLWE split");
  AssertTrue(col >= 0 && col < res.cols,
             "PcmmBlas::SplitSourceColumn: column out of range");
  AssertTrue(ct.GetNP() == res.np,
             "PcmmBlas::SplitSourceColumn: the column is not at the split's "
             "level");
  AssertFalse(ct.HasRx(), "PcmmBlas::SplitSourceColumn: input has rx");
  AssertTrue(std::abs(ct.GetScale() / res.scale - 1.0) < 1e-9,
             "PcmmBlas::SplitSourceColumn: the column's scale differs from "
             "the split's");
  const int degree = res.degree;
  const int num_primes = res.np.GetNumTotal();
  const int num_chunks = (degree + res.chunk_b - 1) / res.chunk_b;
  constexpr int block = 256;
  for (int j = 0; j < num_primes; j++) {
    for (int c = 0; c < num_chunks; c++) {
      const size_t off = static_cast<size_t>(c) * res.chunk_b;
      const int span = static_cast<int>(
          std::min(static_cast<size_t>(res.chunk_b), degree - off));
      const int limb_offset =
          static_cast<int>(j * static_cast<size_t>(degree) + off);
      const int grid = (span + block - 1) / block;
      kernel::SplitGatherColumn<word><<<grid, block>>>(
          res.b_data[static_cast<size_t>(j) * num_chunks + c].data(),
          ct.bx_.data(), res.cols, col, span, limb_offset, res.pieces,
          kPieceBits);
      kernel::SplitGatherColumn<word><<<grid, block>>>(
          res.a_data[static_cast<size_t>(j) * num_chunks + c].data(),
          ct.ax_.data(), res.cols, col, span, limb_offset, res.pieces,
          kPieceBits);
    }
  }
}

template <typename word>
void PcmmBlasHandler<word>::MultiplyInto(word *dst, int dst_ct_stride,
                                         const SplitMatrix &u,
                                         const SplitSource &src) const {
  const int rows = u.rows, cols = u.cols;
  AssertTrue(rows > 0 && cols > 0, "PcmmBlas::MultiplyInto: bad shape");
  AssertTrue(src.rank == 1 && src.degree == param_.degree_,
             "PcmmBlas::MultiplyInto: the split source is not an RLWE one");
  AssertTrue(cols == src.cols && u.pieces == src.pieces && u.np == src.np,
             "PcmmBlas::MultiplyInto: the split source was prepared for a "
             "different matrix");
  AssertTrue(rows <= src.rows,
             "PcmmBlas::MultiplyInto: the split source was cut for fewer "
             "rows than this tile has");
  AssertTrue(ChunkFor(cols, src.rows, u.pieces, src.degree) == src.chunk_b,
             "PcmmBlas::MultiplyInto: the split source was cut for a "
             "different row count");
  const int degree = src.degree;
  const int q_words = src.np.GetNumTotal() * degree;
  AssertTrue(dst_ct_stride >= 2 * q_words,
             "PcmmBlas::MultiplyInto: the row stride must hold both parts");

  HostVector<word *> h_dst_bx(rows), h_dst_ax(rows);
  for (int i = 0; i < rows; i++) {
    h_dst_bx[i] = dst + static_cast<size_t>(i) * dst_ct_stride;
    h_dst_ax[i] = h_dst_bx[i] + q_words;
  }
  DeviceVector<word *> d_dst_bx(rows), d_dst_ax(rows);
  CopyHostToDevice(d_dst_bx, h_dst_bx);
  CopyHostToDevice(d_dst_ax, h_dst_ax);

  ProductComponent(d_dst_bx.data(), src.b_data, u, src.np, degree,
                   src.chunk_b);
  ProductComponent(d_dst_ax.data(), src.a_data, u, src.np, degree,
                   src.chunk_a);
}

template <typename word>
void PcmmBlasHandler<word>::Multiply(std::vector<Ct> &res,
                                     const SplitMatrix &u,
                                     const std::vector<Ct> &cts) const {
  // One product per source, so the split is discarded as soon as it is
  // done; a caller with several row tiles against one source keeps it
  // (`PrepareSource` + `Multiply(res, u, src)`), which is the same words.
  SplitSource src;
  PrepareSource(src, u, cts);
  Multiply(res, u, src);
}

template <typename word>
void PcmmBlasHandler<word>::PrepareSource(
    SplitSource &res, const SplitMatrix &u,
    const std::vector<MlweCiphertext<word>> &cts) const {
  const int cols = static_cast<int>(cts.size());
  AssertTrue(cols > 0, "PcmmBlas::PrepareSource: no inputs");
  AssertTrue(cols == u.cols,
             "PcmmBlas::PrepareSource: cts size must equal u.cols");

  const NPInfo np = cts.at(0).np_;
  AssertTrue(np.num_aux_ == 0,
             "PcmmBlas::PrepareSource: aux primes unsupported");
  AssertTrue(np == u.np,
             "PcmmBlas::PrepareSource: NP mismatch between u and cts");
  const int rank = cts.at(0).rank_;
  const int small_degree = cts.at(0).degree_;
  for (const auto &ct : cts) {
    AssertTrue(ct.np_ == np,
               "PcmmBlas::PrepareSource: ciphertexts differ in NP");
    AssertTrue(ct.rank_ == rank && ct.degree_ == small_degree,
               "PcmmBlas::PrepareSource: ciphertexts differ in rank or "
               "degree");
  }

  // All k blocks of a limb are contiguous, so the a-part is one long row and
  // the product needs no notion of rank -- the same substitution
  // PcmmHandler::Multiply(MLWE) makes when it hands PcmmAccum the a-stride in
  // place of the degree.
  const int a_stride = rank * small_degree;

  res.cols = cols;
  res.pieces = u.pieces;
  res.rank = rank;
  res.degree = small_degree;
  res.scale = cts.at(0).scale_;
  res.np = np;
  res.chunk_b = ChunkFor(cols, u.rows, u.pieces, small_degree);
  res.chunk_a = ChunkFor(cols, u.rows, u.pieces, a_stride);

  HostVector<word *> h_src_a(cols), h_src_b(cols);
  for (int j = 0; j < cols; j++) {
    h_src_a[j] = const_cast<word *>(cts[j].a_.data());
    h_src_b[j] = const_cast<word *>(cts[j].b_.data());
  }
  DeviceVector<word *> d_src_a(cols), d_src_b(cols);
  CopyHostToDevice(d_src_a, h_src_a);
  CopyHostToDevice(d_src_b, h_src_b);

  SplitComponent(res.b_data, d_src_b.data(), cols, u.pieces, np, small_degree,
                 res.chunk_b);
  SplitComponent(res.a_data, d_src_a.data(), cols, u.pieces, np, a_stride,
                 res.chunk_a);
}

template <typename word>
void PcmmBlasHandler<word>::Multiply(std::vector<MlweCiphertext<word>> &res,
                                     const SplitMatrix &u,
                                     const SplitSource &src) const {
  const int rows = u.rows, cols = u.cols;
  AssertTrue(rows > 0 && cols > 0, "PcmmBlas::Multiply(MLWE): bad shape");
  AssertTrue(cols == src.cols && u.pieces == src.pieces && u.np == src.np,
             "PcmmBlas::Multiply(MLWE): the split source was prepared for a "
             "different matrix");

  const int a_stride = src.rank * src.degree;
  AssertTrue(ChunkFor(cols, rows, u.pieces, src.degree) == src.chunk_b &&
                 ChunkFor(cols, rows, u.pieces, a_stride) == src.chunk_a,
             "PcmmBlas::Multiply(MLWE): the split source was cut for a "
             "different row count");

  const int num_primes = src.np.GetNumTotal();
  res.clear();
  res.resize(rows);
  for (auto &r : res) {
    r.rank_ = src.rank;
    r.degree_ = src.degree;
    r.np_ = src.np;
    r.scale_ = u.scale * src.scale;
    r.a_.resize(num_primes * a_stride);
    r.b_.resize(num_primes * src.degree);
  }

  HostVector<word *> h_dst_a(rows), h_dst_b(rows);
  for (int i = 0; i < rows; i++) {
    h_dst_a[i] = res[i].a_.data();
    h_dst_b[i] = res[i].b_.data();
  }
  DeviceVector<word *> d_dst_a(rows), d_dst_b(rows);
  CopyHostToDevice(d_dst_a, h_dst_a);
  CopyHostToDevice(d_dst_b, h_dst_b);

  ProductComponent(d_dst_b.data(), src.b_data, u, src.np, src.degree,
                   src.chunk_b);
  ProductComponent(d_dst_a.data(), src.a_data, u, src.np, a_stride,
                   src.chunk_a);
}

template <typename word>
void PcmmBlasHandler<word>::Multiply(
    std::vector<MlweCiphertext<word>> &res, const SplitMatrix &u,
    const std::vector<MlweCiphertext<word>> &cts) const {
  SplitSource src;
  PrepareSource(src, u, cts);
  Multiply(res, u, src);
}

template class PcmmBlasHandler<uint32_t>;
template class PcmmBlasHandler<uint64_t>;

}  // namespace cheddar

#endif  // USE_CUBLAS
