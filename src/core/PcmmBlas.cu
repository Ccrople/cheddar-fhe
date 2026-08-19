#ifdef USE_CUBLAS

#include <cmath>

#include "common/Assert.h"
#include "common/BasicOps.cuh"
#include "core/BigInt.h"
#include "core/PcmmBlas.h"
#include "core/PrimeUtils.h"

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
__global__ void CombineGroups(word *dst, const int32_t *groups, int num_groups,
                              size_t n, size_t group_stride, word prime,
                              const word *shift_pow) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t acc = 0;
  for (int t = 0; t < num_groups; t++) {
    // The groups are rows * degree apart, not degree: this kernel handles one
    // output row, so the row offset is folded into `groups` by the caller but
    // the stride between groups is the whole slab.
    const uint64_t part = static_cast<uint64_t>(
        static_cast<uint32_t>(groups[t * group_stride + idx]));
    acc = (acc + (part % prime) * static_cast<uint64_t>(shift_pow[t])) % prime;
  }
  dst[idx] = static_cast<word>(acc);
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
  AssertTrue(static_cast<int>(values.size()) ==
                 static_cast<size_t>(rows) * cols,
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

template <typename word>
void PcmmBlasHandler<word>::Multiply(std::vector<Ct> &res,
                                     const SplitMatrix &u,
                                     const std::vector<Ct> &cts) const {
  const int rows = u.rows, cols = u.cols, pieces = u.pieces;
  AssertTrue(static_cast<int>(cts.size()) == cols,
             "PcmmBlas::Multiply: cts size must equal u.cols");
  AssertTrue(!cts.empty(), "PcmmBlas::Multiply: no inputs");

  const NPInfo np = cts[0].GetNP();
  const int degree = param_.degree_;
  const int num_primes = np.GetNumTotal();
  AssertTrue(np.num_aux_ == 0, "PcmmBlas::Multiply: aux primes unsupported");
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

  HostVector<word *> h_src_bx(cols), h_src_ax(cols);
  for (int j = 0; j < cols; j++) {
    h_src_bx[j] = const_cast<word *>(cts[j].bx_.data());
    h_src_ax[j] = const_cast<word *>(cts[j].ax_.data());
  }
  DeviceVector<word *> d_src_bx(cols), d_src_ax(cols);
  CopyHostToDevice(d_src_bx, h_src_bx);
  CopyHostToDevice(d_src_ax, h_src_ax);

  const size_t src_n = static_cast<size_t>(cols) * degree;
  const size_t dst_n = static_cast<size_t>(rows) * degree;
  const int num_groups = 2 * pieces - 1;
  DeviceVector<int8_t> src_split(static_cast<int>(pieces * src_n));
  DeviceVector<int32_t> groups(static_cast<int>(num_groups * dst_n));

  const auto primes = param_.GetPrimeVector(np);
  const int one = 1;
  constexpr int kBlock = 256;

  for (int comp = 0; comp < 2; comp++) {
    const word *const *src_ptrs =
        comp == 0 ? d_src_bx.data() : d_src_ax.data();
    for (int j = 0; j < num_primes; j++) {
      const int limb_offset = j * degree;
      kernel::SplitGather<word>
          <<<static_cast<int>((src_n + kBlock - 1) / kBlock), kBlock>>>(
              src_split.data(), src_ptrs, cols, degree, limb_offset, pieces,
              kPieceBits);

      cudaMemset(groups.data(), 0, num_groups * dst_n * sizeof(int32_t));
      for (int l = 0; l < pieces; l++) {
        const int8_t *b = src_split.data() + static_cast<size_t>(l) * src_n;
        for (int k = 0; k < pieces; k++) {
          const int8_t *a = u.data.data() +
                            (static_cast<size_t>(k) * num_primes + j) *
                                static_cast<size_t>(rows) * cols;
          int32_t *c = groups.data() + static_cast<size_t>(k + l) * dst_n;
          // beta = 1 accumulates the products that share a shift, which keeps
          // every group inside int32 and leaves one 64-bit recombination for
          // the kernel below.
          const cublasStatus_t st = cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, degree, rows, cols, &one, b,
              CUDA_R_8I, degree, a, CUDA_R_8I, cols, &one, c, CUDA_R_32I,
              degree, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
          AssertTrue(st == CUBLAS_STATUS_SUCCESS, "PcmmBlas: GemmEx failed");
        }
      }

      // 2^(7t) mod p for each shift group.
      HostVector<word> h_shift(num_groups);
      {
        uint64_t acc = 1;
        const uint64_t p = primes[j];
        for (int t = 0; t < num_groups; t++) {
          h_shift[t] = static_cast<word>(acc);
          for (int s = 0; s < kPieceBits; s++) acc = (acc * 2) % p;
        }
      }
      DeviceVector<word> d_shift(num_groups);
      CopyHostToDevice(d_shift, h_shift);

      for (int i = 0; i < rows; i++) {
        word *dst = (comp == 0 ? res[i].bx_.data() : res[i].ax_.data()) +
                    limb_offset;
        kernel::CombineGroups<word>
            <<<static_cast<int>((degree + kBlock - 1) / kBlock), kBlock>>>(
                dst, groups.data() + static_cast<size_t>(i) * degree,
                num_groups, degree, dst_n, primes[j], d_shift.data());
      }
    }
  }
}

template class PcmmBlasHandler<uint32_t>;
template class PcmmBlasHandler<uint64_t>;

}  // namespace cheddar

#endif  // USE_CUBLAS
