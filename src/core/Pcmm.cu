#include <cmath>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"
#include "core/BigInt.h"
#include "core/Pcmm.h"

namespace cheddar {

namespace kernel {

// One output coefficient of one output row, for one RNS limb:
//
//   dst[row][limb][x] = sum_j u[limb][row][j] * src[j][limb][x]   (mod p_limb)
//
// The rows of the ciphertext operand are reached through a device array of
// pointers rather than a single contiguous buffer, so that the same kernel can
// serve the RLWE layout (one row per ciphertext) and, later, the MLWE layout
// (one row per concatenated a-part). `u` is in Montgomery form.
template <typename word>
__global__ void PcmmAccum(word *const *dst_ptrs, const word *const *src_ptrs,
                          const word *u, const word *primes,
                          const make_signed_t<word> *inv_primes, int rows,
                          int cols, int degree) {
  using signed_word = make_signed_t<word>;

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int prime_index = blockIdx.z;

  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);

  const word *u_row = u + (static_cast<size_t>(prime_index) * rows + row) * cols;
  const int limb_offset = prime_index * degree + x;

  word acc = 0;
  for (int j = 0; j < cols; j++) {
    const word u_value = basic::StreamingLoadConst(u_row + j);
    if (u_value == 0) continue;
    const word src_value = basic::StreamingLoad(src_ptrs[j] + limb_offset);
    const word product =
        basic::MultMontgomery(u_value, src_value, prime, inv_prime);
    acc = basic::Add(acc, product, prime);
  }

  dst_ptrs[row][limb_offset] = acc;
}

}  // namespace kernel

template <typename word>
int PlainMatrix<word>::GetRows() const {
  return rows_;
}

template <typename word>
int PlainMatrix<word>::GetCols() const {
  return cols_;
}

template <typename word>
double PlainMatrix<word>::GetScale() const {
  return scale_;
}

template <typename word>
NPInfo PlainMatrix<word>::GetNP() const {
  return np_;
}

template <typename word>
PcmmHandler<word>::PcmmHandler(const Parameter<word> &param) : param_{param} {
  AssertTrue(param_.degree_ % kernel_block_dim_ == 0,
             "PcmmHandler: Invalid kernel block dim");
}

template <typename word>
void PcmmHandler<word>::EncodeMatrix(PlainMatrix<word> &res, int level,
                                     double scale,
                                     const std::vector<double> &values,
                                     int rows, int cols,
                                     int num_aux /*= 0*/) const {
  AssertTrue(rows > 0 && cols > 0, "EncodeMatrix: Invalid matrix shape");
  AssertTrue(static_cast<int>(values.size()) == rows * cols,
             "EncodeMatrix: values size does not match rows * cols");

  NPInfo np = param_.LevelToNP(level, num_aux);
  auto primes = param_.GetPrimeVector(np);
  const int num_total_primes = np.GetNumTotal();

  HostVector<word> host(static_cast<size_t>(num_total_primes) * rows * cols);

  std::vector<BigInt> big_primes;
  for (int j = 0; j < num_total_primes; j++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[j]));
  }

  BigInt residue(static_cast<uint64_t>(0));
  for (int i = 0; i < rows * cols; i++) {
    // BigInt(double) truncates, so round first, as EncodeCoeff does.
    BigInt value(std::round(values[i] * scale));
    for (int j = 0; j < num_total_primes; j++) {
      // BigInt::Mod is always non-negative, which is the RNS limb a negative
      // entry needs.
      BigInt::Mod(residue, value, big_primes[j]);
      host[static_cast<size_t>(j) * rows * cols + i] =
          primeutil::ToMontgomery(static_cast<word>(residue.GetUnsigned()),
                                  primes[j]);
    }
  }

  res.rows_ = rows;
  res.cols_ = cols;
  res.scale_ = scale;
  res.np_ = np;
  res.data_.resize(static_cast<int>(host.size()));
  CopyHostToDevice(res.data_, host);
}

template <typename word>
void PcmmHandler<word>::Multiply(std::vector<Ct> &res,
                                 const PlainMatrix<word> &u,
                                 const std::vector<Ct> &cts) const {
  const int rows = u.rows_;
  const int cols = u.cols_;
  AssertTrue(cols == static_cast<int>(cts.size()),
             "Pcmm::Multiply: u.cols_ does not match the number of "
             "ciphertexts");
  AssertTrue(rows > 0 && cols > 0, "Pcmm::Multiply: Invalid matrix shape");

  const NPInfo np = cts.at(0).GetNP();
  AssertTrue(np == u.np_, "Pcmm::Multiply: NP mismatch between u and cts");
  const double ct_scale = cts.at(0).GetScale();
  const int num_slots = cts.at(0).GetNumSlots();
  for (const auto &ct : cts) {
    AssertTrue(ct.GetNP() == np, "Pcmm::Multiply: ciphertexts differ in NP");
    AssertTrue(!ct.HasRx(),
               "Pcmm::Multiply: ciphertexts must not carry an rx_ part");
  }

  // Aliasing would make the accumulation read partially-written rows.
  for (const auto &r : res) {
    for (const auto &c : cts) {
      AssertTrue(&r != &c, "Pcmm::Multiply: in-place operation is not "
                           "supported");
    }
  }

  const int degree = param_.degree_;
  const int num_total_primes = np.GetNumTotal();

  res.resize(rows);
  for (auto &r : res) {
    r.RemoveRx();
    r.ModifyNP(np);
    r.SetNumSlots(num_slots);
    r.SetScale(u.scale_ * ct_scale);
  }

  // Row pointers for both ciphertext components. The component data is
  // contiguous over all limbs from offset 0; only the *prime* pointer carries
  // the terminal-prime offset (Parameter::GetPrimesPtr).
  HostVector<word *> h_dst_bx(rows), h_dst_ax(rows);
  HostVector<word *> h_src_bx(cols), h_src_ax(cols);
  for (int i = 0; i < rows; i++) {
    h_dst_bx[i] = res[i].bx_.data();
    h_dst_ax[i] = res[i].ax_.data();
  }
  for (int j = 0; j < cols; j++) {
    h_src_bx[j] = const_cast<word *>(cts[j].bx_.data());
    h_src_ax[j] = const_cast<word *>(cts[j].ax_.data());
  }

  DeviceVector<word *> d_dst_bx(rows), d_dst_ax(rows);
  DeviceVector<word *> d_src_bx(cols), d_src_ax(cols);
  CopyHostToDevice(d_dst_bx, h_dst_bx);
  CopyHostToDevice(d_dst_ax, h_dst_ax);
  CopyHostToDevice(d_src_bx, h_src_bx);
  CopyHostToDevice(d_src_ax, h_src_ax);

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);

  const dim3 grid_dim(degree / kernel_block_dim_, rows, num_total_primes);

  // b and a are two independent products against the same U.
  kernel::PcmmAccum<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_bx.data(), d_src_bx.data(), u.data_.data(), primes, inv_primes,
      rows, cols, degree);
  kernel::PcmmAccum<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_ax.data(), d_src_ax.data(), u.data_.data(), primes, inv_primes,
      rows, cols, degree);
}

template <typename word>
void PcmmHandler<word>::Multiply(
    std::vector<MlweCiphertext<word>> &res, const PlainMatrix<word> &u,
    const std::vector<MlweCiphertext<word>> &cts) const {
  const int rows = u.rows_;
  const int cols = u.cols_;
  AssertTrue(rows > 0 && cols > 0, "Pcmm::Multiply(MLWE): Invalid matrix shape");
  AssertTrue(cols == static_cast<int>(cts.size()),
             "Pcmm::Multiply(MLWE): u.cols_ does not match the number of "
             "ciphertexts");

  const NPInfo np = cts.at(0).np_;
  AssertTrue(np == u.np_, "Pcmm::Multiply(MLWE): NP mismatch between u and cts");
  const int rank = cts.at(0).rank_;
  const int small_degree = cts.at(0).degree_;
  const double ct_scale = cts.at(0).scale_;
  for (const auto &ct : cts) {
    AssertTrue(ct.np_ == np, "Pcmm::Multiply(MLWE): ciphertexts differ in NP");
    AssertTrue(ct.rank_ == rank && ct.degree_ == small_degree,
               "Pcmm::Multiply(MLWE): ciphertexts differ in rank or degree");
  }

  // Aliasing would make the accumulation read partially-written rows.
  for (const auto &r : res) {
    for (const auto &c : cts) {
      AssertTrue(&r != &c,
                 "Pcmm::Multiply(MLWE): in-place operation is not supported");
    }
  }

  const int num_total_primes = np.GetNumTotal();
  // Per-limb stride of the a-part. All k blocks of a limb are contiguous, so
  // the accumulator treats them as one long row and needs no notion of rank.
  const int a_stride = rank * small_degree;

  res.clear();
  res.resize(rows);
  for (auto &r : res) {
    r.rank_ = rank;
    r.degree_ = small_degree;
    r.np_ = np;
    r.scale_ = u.scale_ * ct_scale;
    r.a_.resize(num_total_primes * a_stride);
    r.b_.resize(num_total_primes * small_degree);
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

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);

  // The b-part is one small_degree-vector per limb, and the conjugate-
  // invariant projection's small degree of 128 (Doing.md 1.5bh) sits below
  // the 256-thread block: shrink the block rather than refuse the shape.
  // Both lengths are powers of two, so the smaller divides the other.
  const int block_b = Min(small_degree, kernel_block_dim_);
  const dim3 grid_b(small_degree / block_b, rows, num_total_primes);
  kernel::PcmmAccum<word><<<grid_b, block_b>>>(
      d_dst_b.data(), d_src_b.data(), u.data_.data(), primes, inv_primes, rows,
      cols, small_degree);

  const int block_a = Min(a_stride, kernel_block_dim_);
  const dim3 grid_a(a_stride / block_a, rows, num_total_primes);
  kernel::PcmmAccum<word><<<grid_a, block_a>>>(
      d_dst_a.data(), d_src_a.data(), u.data_.data(), primes, inv_primes, rows,
      cols, a_stride);
}

template class PlainMatrix<uint32_t>;
template class PlainMatrix<uint64_t>;
template class PcmmHandler<uint32_t>;
template class PcmmHandler<uint64_t>;

}  // namespace cheddar
