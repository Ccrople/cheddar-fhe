#include "core/EncodeGpu.h"

#include <algorithm>
#include <iostream>
#include <cmath>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"

namespace cheddar {
namespace kernel {

// ---------------------------------------------------------------------------
// Stage 1: the special inverse FFT.
//
// One kernel serves both passes. A block owns `chunk = 2^log_chunk` elements
// whose global indices are `group_base + k * 2^log_elem_stride`, and runs
// every decimation-in-frequency stage whose butterfly stays inside that set:
// at local stage `t` the partner is `2^(log_chunk-1-t)` away in `k`, so the
// global stride is that times the element stride, and the stages the two
// passes run are exactly the stages whose stride is at or above / below the
// split. Both passes therefore differ only in their launch parameters.
// ---------------------------------------------------------------------------
__global__ void SpecialIfftStagesKernel(double2 *data, const double2 *tw,
                                        int log_chunk, int log_elem_stride) {
  extern __shared__ double2 sh[];

  const int chunk = 1 << log_chunk;
  const int elem_stride = 1 << log_elem_stride;
  const int gid = blockIdx.x;
  // The low `log_elem_stride` bits of the block index select the phase-1
  // gather offset; the rest select a contiguous chunk. One expression covers
  // both passes because in each of them one of the two halves is empty.
  const int group_base =
      (gid & (elem_stride - 1)) +
      ((gid >> log_elem_stride) << (log_elem_stride + log_chunk));

  for (int k = threadIdx.x; k < chunk; k += blockDim.x) {
    sh[k] = data[group_base + (k << log_elem_stride)];
  }
  __syncthreads();

  for (int t = 0; t < log_chunk; t++) {
    const int log_half = log_chunk - 1 - t;
    const int half = 1 << log_half;
    const int stride = half << log_elem_stride;
    const int mask = stride - 1;
    for (int q = threadIdx.x; q < (chunk >> 1); q += blockDim.x) {
      const int lo = ((q >> log_half) << (log_half + 1)) | (q & (half - 1));
      const int hi = lo + half;
      // `j` is the position of the butterfly inside its group of `stride`,
      // measured on the GLOBAL index -- which is what selects the twiddle.
      const int j = (group_base + (lo << log_elem_stride)) & mask;
      const double2 w = tw[stride + j];
      const double2 a = sh[lo];
      const double2 b = sh[hi];
      const double yr = a.x - b.x;
      const double yi = a.y - b.y;
      sh[lo] = make_double2(a.x + b.x, a.y + b.y);
      sh[hi] = make_double2(yr * w.x - yi * w.y, yr * w.y + yi * w.x);
    }
    __syncthreads();
  }

  for (int k = threadIdx.x; k < chunk; k += blockDim.x) {
    data[group_base + (k << log_elem_stride)] = sh[k];
  }
}

// ---------------------------------------------------------------------------
// Stage 2: bit reversal, the 1/S normalisation, and the placement of a slot's
// value at the coefficient the encoding puts it at. Reading bit-reversed and
// writing in order keeps the write coalesced, which is the side that carries
// two values per slot off the conjugate-invariant ring.
// ---------------------------------------------------------------------------
__global__ void FftToCoeffKernel(double *coeff, const double2 *fft,
                                 int num_slots, int log_slots, int gap,
                                 int half_degree, bool conjugate_invariant) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_slots) return;
  const int src = static_cast<int>(__brev(static_cast<unsigned>(i)) >>
                                   (32 - log_slots));
  const double2 v = fft[src];
  const double inv = 1.0 / static_cast<double>(num_slots);
  coeff[i * gap] = v.x * inv;
  if (!conjugate_invariant) coeff[i * gap + half_degree] = v.y * inv;
}

// ---------------------------------------------------------------------------
// Stage 3: RNS decomposition.
//
// What `BigInt` is doing on the host, without `BigInt`. `round(v)` of a double
// is an integer whose exact value is `m * 2^e` with a 53-bit `m`, so
// `(m * 2^e) mod p` is exact in machine arithmetic. Everything this pipeline
// encodes lands below 2^64 and takes one Barrett reduction; the general case
// falls back to `m mod p` followed by `e` modular doublings, which needs no
// wide multiply either.
// ---------------------------------------------------------------------------

// floor(u / p) is approximated by the high half of u * mu with mu =
// floor((2^64 - 1)/p) <= floor(2^64/p), so the quotient is never an
// over-estimate and the remainder is never negative. The correction loop runs
// at most three times.
__device__ __forceinline__ uint64_t Barrett64(uint64_t u, uint64_t p,
                                              uint64_t mu) {
  const uint64_t q = __umul64hi(u, mu);
  uint64_t r = u - q * p;
  while (r >= p) r -= p;
  return r;
}

// |round(v)| >= 2^64. Cold: no message or weight in this tree reaches it, but
// the host path it replaces is exact for any magnitude and this one has to be
// too.
__device__ __forceinline__ uint64_t WideResidue(double a, uint64_t p,
                                                uint64_t mu) {
  int exponent = 0;
  const double m = frexp(a, &exponent);              // a = m * 2^exponent
  const uint64_t mant = static_cast<uint64_t>(ldexp(m, 53));
  int shift = exponent - 53;                         // a = mant * 2^shift
  uint64_t r = Barrett64(mant, p, mu);
  for (int k = 0; k < shift; k++) {
    r <<= 1;
    if (r >= p) r -= p;
  }
  return r;
}

template <typename word>
__global__ void RnsDecomposeKernel(word *dst, const double *src, int n,
                                   int num_primes, const uint64_t *consts,
                                   const make_signed_t<word> *inv_primes,
                                   double scale, bool montgomery) {
  using signed_word = make_signed_t<word>;

  // The per-prime table, staged once per block instead of once per thread:
  // it is the same 24 bytes for all 256 threads and it is read `num_primes`
  // times by each of them.
  extern __shared__ uint64_t sh_const[];
  uint64_t *sh_p = sh_const;
  uint64_t *sh_mu = sh_p + num_primes;
  uint64_t *sh_r2 = sh_mu + num_primes;
  signed_word *sh_qinv = reinterpret_cast<signed_word *>(sh_r2 + num_primes);

  for (int j = threadIdx.x; j < num_primes; j += blockDim.x) {
    sh_p[j] = consts[3 * j];
    sh_mu[j] = consts[3 * j + 1];
    sh_r2[j] = consts[3 * j + 2];
    sh_qinv[j] = inv_primes[j];
  }
  __syncthreads();

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double value = round(src[i] * scale);
  const bool negative = value < 0.0;
  const double magnitude = fabs(value);
  // 2^64 exactly; below it the conversion to uint64_t is exact and one
  // Barrett reduction is the whole of the work.
  const bool narrow = magnitude < 18446744073709551616.0;
  const uint64_t narrow_value =
      narrow ? static_cast<uint64_t>(magnitude) : UINT64_C(0);

  for (int j = 0; j < num_primes; j++) {
    const uint64_t p = sh_p[j];
    uint64_t r = narrow ? Barrett64(narrow_value, p, sh_mu[j])
                        : WideResidue(magnitude, p, sh_mu[j]);
    if (negative && r != 0) r = p - r;
    if (montgomery) {
      // r * R mod p, as ToMontgomery does it on the host: one Montgomery
      // multiplication by R^2 mod p, which is the library's own primitive.
      signed_word t = basic::detail::__mult_montgomery_lazy<word>(
          static_cast<signed_word>(r), static_cast<signed_word>(sh_r2[j]),
          static_cast<word>(p), sh_qinv[j]);
      if (t < 0) t += static_cast<signed_word>(p);
      r = static_cast<uint64_t>(t);
    }
    dst[static_cast<int64_t>(j) * n + i] = static_cast<word>(r);
  }
}

}  // namespace kernel

// ---------------------------------------------------------------------------

template <typename word>
GpuEncoder<word>::GpuEncoder(const Parameter<word> &param,
                             const NTTHandler<word> &ntt_handler)
    : param_{param},
      ntt_handler_{ntt_handler},
      degree_{param.degree_},
      max_slots_{param.MaxNumSlots()},
      cyclotomic_index_{param.CyclotomicIndex()},
      twiddle_(static_cast<size_t>(2) * param.MaxNumSlots(), cudaStreamLegacy),
      fft_(0, cudaStreamLegacy),
      coeff_(static_cast<size_t>(param.degree_), cudaStreamLegacy),
      value_stage_(0, cudaStreamLegacy) {
  // The same entry the host reads out of its length-M table, computed the same
  // way so that the two transforms differ in nothing but where they run.
  HostVector<double> host(static_cast<size_t>(2) * max_slots_, 0.0);
  for (int stride = 1; stride < max_slots_; stride <<= 1) {
    const int st8 = stride << 3;
    const int gap = cyclotomic_index_ / st8;
    for (int j = 0; j < stride; j++) {
      const int g = param_.GetGaloisFactor(j) % st8;
      const int index = (st8 - g) * gap;
      const Complex w = std::polar(
          1.0, 2.0 * M_PI * static_cast<double>(index) / cyclotomic_index_);
      host[2 * (stride + j)] = w.real();
      host[2 * (stride + j) + 1] = w.imag();
    }
  }
  cudaMemcpyAsync(twiddle_.data(), host.data(), host.size() * sizeof(double),
                  cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaStreamSynchronize(cudaStreamLegacy);
}

template <typename word>
void GpuEncoder<word>::EnsureScratch(int num_slots) const {
  const size_t need = static_cast<size_t>(2) * num_slots;
  if (fft_.size() < need) fft_.resize(need, cudaStreamLegacy);
  if (host_stage_.size() < need) host_stage_.resize(need);
}

template <typename word>
double *GpuEncoder<word>::FftScratch(int num_slots) const {
  EnsureScratch(num_slots);
  return fft_.data();
}

template <typename word>
double *GpuEncoder<word>::CoeffScratch() const {
  return coeff_.data();
}

template <typename word>
void GpuEncoder<word>::StageMessage(double *device_dst,
                                    const std::vector<Complex> &message,
                                    int num_slots) const {
  EnsureScratch(num_slots);
  const int given = static_cast<int>(message.size());
  for (int i = 0; i < num_slots; i++) {
    host_stage_[2 * i] = i < given ? message[i].real() : 0.0;
    host_stage_[2 * i + 1] = i < given ? message[i].imag() : 0.0;
  }
  cudaMemcpyAsync(device_dst, host_stage_.data(),
                  static_cast<size_t>(2) * num_slots * sizeof(double),
                  cudaMemcpyHostToDevice, cudaStreamLegacy);
}

template <typename word>
void GpuEncoder<word>::SpecialIFFT(double *data, int num_slots) const {
  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "GpuEncoder::SpecialIFFT: Power of 2 num slots only");
  AssertTrue(num_slots <= max_slots_,
             "GpuEncoder::SpecialIFFT: too many slots for this ring");
  if (num_slots <= 1) return;

  const int log_slots = Log2Ceil(num_slots);
  double2 *d = reinterpret_cast<double2 *>(data);
  const double2 *tw = reinterpret_cast<const double2 *>(twiddle_.data());

  auto launch = [&](int log_chunk, int log_elem_stride) {
    const int chunk = 1 << log_chunk;
    const int blocks = num_slots >> log_chunk;
    const int threads = std::min(chunk >> 1, 256);
    const size_t smem = static_cast<size_t>(chunk) * sizeof(double2);
    kernel::SpecialIfftStagesKernel<<<blocks, std::max(threads, 1), smem,
                                      cudaStreamLegacy>>>(d, tw, log_chunk,
                                                          log_elem_stride);
  };

  if (log_slots <= kMaxLogChunk) {
    launch(log_slots, 0);
    return;
  }
  // Split so that both passes stage a chunk that fits, and so that neither
  // ends up with so few blocks that the machine goes idle -- which is the
  // failure mode of the lopsided split, not a shared-memory limit.
  int log_low = (log_slots + 1) / 2;
  if (log_low > kMaxLogChunk) log_low = kMaxLogChunk;
  const int log_high = log_slots - log_low;
  AssertTrue(log_high <= kMaxLogChunk,
             "GpuEncoder::SpecialIFFT: slot count too large for two passes");
  launch(log_high, log_low);  // the stages with stride >= 2^log_low
  launch(log_low, 0);         // the rest, inside a contiguous chunk
}

template <typename word>
void GpuEncoder<word>::FftToCoeff(double *coeff, const double *fft,
                                  int num_slots) const {
  const int log_slots = Log2Ceil(num_slots);
  const bool ci = param_.conjugate_invariant_;
  const int half_degree = degree_ / 2;
  const int gap = (ci ? degree_ : half_degree) / num_slots;
  if (gap > 1) {
    cudaMemsetAsync(coeff, 0, static_cast<size_t>(degree_) * sizeof(double),
                    cudaStreamLegacy);
  }
  const int threads = 256;
  const int blocks = (num_slots + threads - 1) / threads;
  kernel::FftToCoeffKernel<<<blocks, threads, 0, cudaStreamLegacy>>>(
      coeff, reinterpret_cast<const double2 *>(fft), num_slots, log_slots, gap,
      half_degree, ci);
}

template <typename word>
const uint64_t *GpuEncoder<word>::PrimeConstants(const NPInfo &np) const {
  const auto key = std::make_tuple(np.num_main_, np.num_ter_, np.num_aux_,
                                   np.degree_);
  auto it = prime_constants_.find(key);
  if (it != prime_constants_.end()) return it->second.data();

  const auto primes = param_.GetPrimeVector(np);
  const int num = np.GetNumTotal();
  HostVector<uint64_t> host(static_cast<size_t>(3) * num);
  for (int j = 0; j < num; j++) {
    const uint64_t p = static_cast<uint64_t>(primes[j]);
    host[3 * j] = p;
    // floor((2^64 - 1) / p) = floor(2^64 / p) for every odd p, and it is the
    // one of the two that a 64-bit division can produce.
    host[3 * j + 1] = ~UINT64_C(0) / p;
    // R^2 mod p, R = 2^(8 * sizeof(word)): the constant ToMontgomery needs.
    host[3 * j + 2] = static_cast<uint64_t>(primeutil::PowMod<word>(
        static_cast<word>(2), 16 * static_cast<int64_t>(sizeof(word)),
        static_cast<word>(p)));
  }
  rmm::device_uvector<uint64_t> dv(host.size(), cudaStreamLegacy);
  cudaMemcpyAsync(dv.data(), host.data(), host.size() * sizeof(uint64_t),
                  cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaStreamSynchronize(cudaStreamLegacy);
  auto inserted = prime_constants_.emplace(key, std::move(dv));
  return inserted.first->second.data();
}

template <typename word>
void GpuEncoder<word>::RnsDecompose(word *dst, const double *src, int n,
                                    const NPInfo &np, double scale,
                                    bool montgomery) const {
  const int num_primes = np.GetNumTotal();
  const uint64_t *consts = PrimeConstants(np);
  const int threads = kRnsBlockDim;
  const int blocks = (n + threads - 1) / threads;
  const size_t smem = static_cast<size_t>(num_primes) *
                      (3 * sizeof(uint64_t) + sizeof(make_signed_t<word>));
  kernel::RnsDecomposeKernel<word><<<blocks, threads, smem, cudaStreamLegacy>>>(
      dst, src, n, num_primes, consts, param_.GetInvPrimesPtr(np), scale,
      montgomery);
}

template <typename word>
void GpuEncoder<word>::Encode(Plaintext<word> &ptxt, int level, double scale,
                              const std::vector<Complex> &message,
                              int num_aux /*= 0*/) const {
  const int msg_length = static_cast<int>(message.size());
  const int num_slots = 1 << Log2Ceil<int>(msg_length);
  AssertTrue(num_slots <= max_slots_,
             "GpuEncoder::Encode: too many slots for this ring");
  if (param_.conjugate_invariant_) {
    double max_real = 0.0;
    double max_imag = 0.0;
    for (const auto &value : message) {
      max_real = std::max(max_real, std::abs(value.real()));
      max_imag = std::max(max_imag, std::abs(value.imag()));
    }
    AssertTrue(max_imag <= 1e-8 * std::max(max_real, 1.0),
               "GpuEncoder::Encode: the conjugate-invariant ring has real "
               "slots, and this message has an imaginary part");
  }

  StageMessage(fft_.data(), message, num_slots);
  SpecialIFFT(fft_.data(), num_slots);
  FftToCoeff(coeff_.data(), fft_.data(), num_slots);

  const NPInfo np = param_.LevelToNP(level, num_aux);
  ptxt.ModifyNP(np);
  ptxt.SetNumSlots(num_slots);
  ptxt.SetScale(scale);
  RnsDecompose(ptxt.mx_.data(), coeff_.data(), degree_, np, scale, false);
  auto view = ptxt.View();
  ntt_handler_.NTT(view, np, ptxt.ConstView(), true);
}

template <typename word>
void GpuEncoder<word>::EncodeCoeff(Plaintext<word> &ptxt, int level,
                                   double scale,
                                   const std::vector<double> &coeffs,
                                   int num_aux /*= 0*/) const {
  const int num_coeffs = static_cast<int>(coeffs.size());
  AssertTrue(num_coeffs > 0 && num_coeffs <= degree_,
             "GpuEncoder::EncodeCoeff: invalid number of coefficients");

  if (num_coeffs < degree_) {
    cudaMemsetAsync(coeff_.data(), 0,
                    static_cast<size_t>(degree_) * sizeof(double),
                    cudaStreamLegacy);
  }
  cudaMemcpyAsync(coeff_.data(), coeffs.data(),
                  static_cast<size_t>(num_coeffs) * sizeof(double),
                  cudaMemcpyHostToDevice, cudaStreamLegacy);

  const NPInfo np = param_.LevelToNP(level, num_aux);
  ptxt.ModifyNP(np);
  ptxt.SetNumSlots(max_slots_);
  ptxt.SetScale(scale);
  RnsDecompose(ptxt.mx_.data(), coeff_.data(), degree_, np, scale, false);
  auto view = ptxt.View();
  ntt_handler_.NTT(view, np, ptxt.ConstView(), true);
}

template <typename word>
void GpuEncoder<word>::EncodeMatrixFromDevice(PlainMatrix<word> &res, int level,
                                              double scale,
                                              const double *values, int rows,
                                              int cols,
                                              int num_aux /*= 0*/) const {
  AssertTrue(rows > 0 && cols > 0,
             "GpuEncoder::EncodeMatrix: Invalid matrix shape");
  const NPInfo np = param_.LevelToNP(level, num_aux);
  const int n = rows * cols;
  res.rows_ = rows;
  res.cols_ = cols;
  res.scale_ = scale;
  res.np_ = np;
  res.data_.resize(np.GetNumTotal() * n);
  RnsDecompose(res.data_.data(), values, n, np, scale, true);
}

template <typename word>
void GpuEncoder<word>::EncodeMatrix(PlainMatrix<word> &res, int level,
                                    double scale,
                                    const std::vector<double> &values, int rows,
                                    int cols, int num_aux /*= 0*/) const {
  AssertTrue(static_cast<int>(values.size()) == rows * cols,
             "GpuEncoder::EncodeMatrix: values size does not match rows*cols");
  const size_t n = static_cast<size_t>(rows) * cols;
  if (value_stage_.size() < n) value_stage_.resize(n, cudaStreamLegacy);
  cudaMemcpyAsync(value_stage_.data(), values.data(), n * sizeof(double),
                  cudaMemcpyHostToDevice, cudaStreamLegacy);
  EncodeMatrixFromDevice(res, level, scale, value_stage_.data(), rows, cols,
                         num_aux);
}

template <typename word>
void GpuEncoder<word>::ReportKernelAttributes(std::ostream &os, int num_primes,
                                              int log_chunk) {
  cudaDeviceProp prop;
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  auto report = [&](const char *name, const void *func, int threads,
                    size_t dynamic_smem) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, func) != cudaSuccess) return;
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, func, threads,
                                                  dynamic_smem);
    const int warps = blocks_per_sm * ((threads + 31) / 32);
    const int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    os << "  " << name << ": regs " << attr.numRegs << ", local "
       << attr.localSizeBytes << " B, static smem " << attr.sharedSizeBytes
       << " B, dynamic smem " << dynamic_smem << " B, " << threads
       << " threads/block, " << blocks_per_sm << " blocks/SM, occupancy "
       << (100.0 * warps / max_warps) << "%" << std::endl;
  };

  os << "kernel attributes (" << prop.name << ", " << prop.multiProcessorCount
     << " SMs, " << (prop.sharedMemPerMultiprocessor >> 10)
     << " KiB smem/SM):" << std::endl;
  const int chunk = 1 << log_chunk;
  report("SpecialIfftStagesKernel",
         reinterpret_cast<const void *>(&kernel::SpecialIfftStagesKernel),
         std::max(std::min(chunk >> 1, 256), 1),
         static_cast<size_t>(chunk) * sizeof(double2));
  report("FftToCoeffKernel",
         reinterpret_cast<const void *>(&kernel::FftToCoeffKernel), 256, 0);
  report("RnsDecomposeKernel",
         reinterpret_cast<const void *>(&kernel::RnsDecomposeKernel<word>),
         kRnsBlockDim,
         static_cast<size_t>(num_primes) *
             (3 * sizeof(uint64_t) + sizeof(make_signed_t<word>)));
}

template class GpuEncoder<uint32_t>;
template class GpuEncoder<uint64_t>;

}  // namespace cheddar
