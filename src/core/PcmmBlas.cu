#ifdef USE_CUBLAS

#include <algorithm>
#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/BigInt.h"
#include "core/PcmmBlas.h"

namespace cheddar {

namespace kernel {

// Gather one RNS limb of every source ciphertext into a contiguous
// cols x degree matrix and split it into int8 pieces at the same time.
//
// The gather is what lets a GEMM run at all: the sources are separate
// allocations, and cuBLAS wants one matrix. It is a bandwidth-bound copy of
// cols * degree words, against a product that is rows times larger.
template <typename word>
__global__ void SplitGather(int8_t *dst, const word *const *src_ptrs, int cols,
                            int degree, int limb_offset, int pieces,
                            int piece_bits) {
  const size_t n = static_cast<size_t>(cols) * degree;
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const int j = static_cast<int>(idx / degree);
  const int x = static_cast<int>(idx % degree);
  const word v = src_ptrs[j][limb_offset + x];
  const word mask = (static_cast<word>(1) << piece_bits) - 1;
  for (int p = 0; p < pieces; p++) {
    dst[static_cast<size_t>(p) * n + idx] =
        static_cast<int8_t>((v >> (piece_bits * p)) & mask);
  }
}

// Recombine the shift groups modulo the prime.
//
// Group t holds the sum of every piece product whose combined shift is
// 7 * t, accumulated in int32 by cuBLAS. Each group is at most
// 127 * 127 * cols * (pieces per group) which stays inside int32; the
// recombination needs 64 bits, and doing it once per output element rather
// than once per product is why the groups exist.
template <typename word>
__global__ void CombineGroups(word *const *dst_ptrs, const int32_t *groups,
                              int num_groups, int rows, int degree,
                              int limb_offset, word prime,
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
  const int row = static_cast<int>(idx / degree);
  const int x = static_cast<int>(idx % degree);
  uint64_t acc = 0;
  for (int t = 0; t < num_groups; t++) {
    const uint64_t part =
        static_cast<uint64_t>(static_cast<uint32_t>(groups[t * n + idx]));
    acc = (acc + (part % prime) * static_cast<uint64_t>(shift_pow[t])) % prime;
  }
  dst_ptrs[row][limb_offset + x] = static_cast<word>(acc);
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
  const int pieces = (max_bits + kPieceBits - 1) / kPieceBits;

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
  const word mask = (static_cast<word>(1) << kPieceBits) - 1;
  for (size_t i = 0; i < per_prime; i++) {
    BigInt value(std::round(values[i] * scale));
    for (int j = 0; j < num_primes; j++) {
      BigInt::Mod(residue, value, big_primes[j]);
      const word r = static_cast<word>(residue.GetUnsigned());
      for (int p = 0; p < pieces; p++) {
        host[(static_cast<size_t>(p) * num_primes + j) * per_prime + i] =
            static_cast<int8_t>((r >> (kPieceBits * p)) & mask);
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

namespace {
// The gather and the recombination are both flat; 256 is what every other
// elementwise kernel in the library uses.
constexpr int kBlock = 256;
}  // namespace

template <typename word>
void PcmmBlasHandler<word>::MultiplyComponent(word *const *dst_ptrs,
                                              const word *const *src_ptrs,
                                              const SplitMatrix &u,
                                              const NPInfo &np,
                                              int vec_len) const {
  const int rows = u.rows, cols = u.cols, pieces = u.pieces;
  const int num_primes = np.GetNumTotal();
  const int num_groups = pieces * pieces;
  AssertTrue(vec_len % kBlock == 0,
             "PcmmBlas: the component length must be a multiple of the block "
             "dim");

  // The scratch buffers are indexed by int, because DeviceVector is, and the
  // a-part of a wide tile passes 2^31 words long before it runs out of card:
  // 56 parents at rank 256 is 14336 columns of 65536 words per limb, which is
  // 3.8e9 int8 pieces. So the product is chunked along the vector axis --
  // valid without further thought, since the product is a scalar combination
  // and never mixes positions within a limb. Bounding the peak is a reason to
  // chunk even where the index would have fitted: at 134M words the two
  // scratch buffers are 134 MB and 537 MB, against a ModDecomp that is already
  // holding gigabytes.
  constexpr size_t kScratchWords = 1u << 27;
  size_t chunk =
      std::min(kScratchWords / (static_cast<size_t>(pieces) * cols),
               kScratchWords / (static_cast<size_t>(num_groups) * rows));
  chunk = std::min(chunk, static_cast<size_t>(vec_len));
  chunk -= chunk % kBlock;
  AssertTrue(chunk >= static_cast<size_t>(kBlock),
             "PcmmBlas: the operand is too wide for the scratch budget");

  const size_t src_n = static_cast<size_t>(cols) * chunk;
  const size_t dst_n = static_cast<size_t>(rows) * chunk;
  DeviceVector<int8_t> src_split(static_cast<int>(pieces * src_n));
  DeviceVector<int32_t> groups(static_cast<int>(num_groups * dst_n));

  const auto primes = param_.GetPrimeVector(np);
  const int one = 1, zero = 0;

  for (int j = 0; j < num_primes; j++) {
    // 2^(7*(k+l)) mod p, laid out to match the (l * pieces + k) buffer order.
    HostVector<word> h_shift(num_groups);
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
          h_shift[l * pieces + k] = static_cast<word>(pow[k + l]);
        }
      }
    }
    DeviceVector<word> d_shift(num_groups);
    CopyHostToDevice(d_shift, h_shift);

    for (size_t off = 0; off < static_cast<size_t>(vec_len); off += chunk) {
      const int span =
          static_cast<int>(std::min(chunk, static_cast<size_t>(vec_len) - off));
      const int limb_offset =
          static_cast<int>(j * static_cast<size_t>(vec_len) + off);
      const size_t src_span = static_cast<size_t>(cols) * span;
      const size_t dst_span = static_cast<size_t>(rows) * span;

      kernel::SplitGather<word>
          <<<static_cast<int>((src_span + kBlock - 1) / kBlock), kBlock>>>(
              src_split.data(), src_ptrs, cols, span, limb_offset, pieces,
              kPieceBits);

      for (int l = 0; l < pieces; l++) {
        const int8_t *b = src_split.data() + static_cast<size_t>(l) * src_span;
        for (int k = 0; k < pieces; k++) {
          const int8_t *a = u.data.data() +
                            (static_cast<size_t>(k) * num_primes + j) *
                                static_cast<size_t>(rows) * cols;
          int32_t *c =
              groups.data() + static_cast<size_t>(l * pieces + k) * dst_span;
          // One buffer per (k, l) and beta = 0. Accumulating several GEMMs into
          // one buffer with beta = 1 was the first thing to suspect when 0.46%
          // of the words came out wrong: cuBLAS's integer path does not
          // guarantee read-modify-write on C the way the floating-point one
          // does. Each product now lands in its own slab and the recombination
          // kernel does all of the summing.
          const cublasStatus_t st = cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, span, rows, cols, &one, b,
              CUDA_R_8I, span, a, CUDA_R_8I, cols, &zero, c, CUDA_R_32I, span,
              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
          AssertTrue(st == CUBLAS_STATUS_SUCCESS, "PcmmBlas: GemmEx failed");
        }
      }

      kernel::CombineGroups<word>
          <<<static_cast<int>((dst_span + kBlock - 1) / kBlock), kBlock>>>(
              dst_ptrs, groups.data(), num_groups, rows, span, limb_offset,
              primes[j], d_shift.data());
    }
  }
}

template <typename word>
void PcmmBlasHandler<word>::Multiply(std::vector<Ct> &res,
                                     const SplitMatrix &u,
                                     const std::vector<Ct> &cts) const {
  const int rows = u.rows, cols = u.cols;
  AssertTrue(static_cast<int>(cts.size()) == cols,
             "PcmmBlas::Multiply: cts size must equal u.cols");
  AssertTrue(!cts.empty(), "PcmmBlas::Multiply: no inputs");

  const NPInfo np = cts[0].GetNP();
  const int degree = param_.degree_;
  AssertTrue(np.num_aux_ == 0, "PcmmBlas::Multiply: aux primes unsupported");
  AssertTrue(np == u.np, "PcmmBlas::Multiply: NP mismatch between u and cts");
  for (const auto &c : cts) {
    AssertTrue(c.GetNP() == np, "PcmmBlas::Multiply: mixed NP");
    AssertFalse(c.HasRx(), "PcmmBlas::Multiply: input has rx");
  }

  const double ct_scale = cts[0].GetScale();
  const int num_slots = cts[0].GetNumSlots();
  res.clear();
  res.resize(rows);
  for (auto &r : res) {
    r.ModifyNP(np);
    r.RemoveRx();
    r.SetNumSlots(num_slots);
    r.SetScale(u.scale * ct_scale);
  }

  HostVector<word *> h_dst_bx(rows), h_dst_ax(rows);
  for (int i = 0; i < rows; i++) {
    h_dst_bx[i] = res[i].bx_.data();
    h_dst_ax[i] = res[i].ax_.data();
  }
  DeviceVector<word *> d_dst_bx(rows), d_dst_ax(rows);
  CopyHostToDevice(d_dst_bx, h_dst_bx);
  CopyHostToDevice(d_dst_ax, h_dst_ax);

  HostVector<word *> h_src_bx(cols), h_src_ax(cols);
  for (int j = 0; j < cols; j++) {
    h_src_bx[j] = const_cast<word *>(cts[j].bx_.data());
    h_src_ax[j] = const_cast<word *>(cts[j].ax_.data());
  }
  DeviceVector<word *> d_src_bx(cols), d_src_ax(cols);
  CopyHostToDevice(d_src_bx, h_src_bx);
  CopyHostToDevice(d_src_ax, h_src_ax);

  MultiplyComponent(d_dst_bx.data(), d_src_bx.data(), u, np, degree);
  MultiplyComponent(d_dst_ax.data(), d_src_ax.data(), u, np, degree);
}

template <typename word>
void PcmmBlasHandler<word>::Multiply(
    std::vector<MlweCiphertext<word>> &res, const SplitMatrix &u,
    const std::vector<MlweCiphertext<word>> &cts) const {
  const int rows = u.rows, cols = u.cols;
  AssertTrue(rows > 0 && cols > 0, "PcmmBlas::Multiply(MLWE): bad shape");
  AssertTrue(static_cast<int>(cts.size()) == cols,
             "PcmmBlas::Multiply(MLWE): cts size must equal u.cols");

  const NPInfo np = cts.at(0).np_;
  AssertTrue(np.num_aux_ == 0,
             "PcmmBlas::Multiply(MLWE): aux primes unsupported");
  AssertTrue(np == u.np,
             "PcmmBlas::Multiply(MLWE): NP mismatch between u and cts");
  const int rank = cts.at(0).rank_;
  const int small_degree = cts.at(0).degree_;
  const double ct_scale = cts.at(0).scale_;
  for (const auto &ct : cts) {
    AssertTrue(ct.np_ == np,
               "PcmmBlas::Multiply(MLWE): ciphertexts differ in NP");
    AssertTrue(ct.rank_ == rank && ct.degree_ == small_degree,
               "PcmmBlas::Multiply(MLWE): ciphertexts differ in rank or "
               "degree");
  }

  const int num_primes = np.GetNumTotal();
  // All k blocks of a limb are contiguous, so the a-part is one long row and
  // the product needs no notion of rank -- the same substitution
  // PcmmHandler::Multiply(MLWE) makes when it hands PcmmAccum the a-stride in
  // place of the degree.
  const int a_stride = rank * small_degree;

  res.clear();
  res.resize(rows);
  for (auto &r : res) {
    r.rank_ = rank;
    r.degree_ = small_degree;
    r.np_ = np;
    r.scale_ = u.scale * ct_scale;
    r.a_.resize(num_primes * a_stride);
    r.b_.resize(num_primes * small_degree);
  }

  HostVector<word *> h_dst_a(rows), h_dst_b(rows);
  HostVector<word *> h_src_a(cols), h_src_b(cols);
  for (int i = 0; i < rows; i++) {
    h_dst_a[i] = res[i].a_.data();
    h_dst_b[i] = res[i].b_.data();
  }
  for (int j = 0; j < cols; j++) {
    h_src_a[j] = const_cast<word *>(cts[j].a_.data());
    h_src_b[j] = const_cast<word *>(cts[j].b_.data());
  }
  DeviceVector<word *> d_dst_a(rows), d_dst_b(rows);
  DeviceVector<word *> d_src_a(cols), d_src_b(cols);
  CopyHostToDevice(d_dst_a, h_dst_a);
  CopyHostToDevice(d_dst_b, h_dst_b);
  CopyHostToDevice(d_src_a, h_src_a);
  CopyHostToDevice(d_src_b, h_src_b);

  MultiplyComponent(d_dst_b.data(), d_src_b.data(), u, np, small_degree);
  MultiplyComponent(d_dst_a.data(), d_src_a.data(), u, np, a_stride);
}

template class PcmmBlasHandler<uint32_t>;
template class PcmmBlasHandler<uint64_t>;

}  // namespace cheddar

#endif  // USE_CUBLAS
