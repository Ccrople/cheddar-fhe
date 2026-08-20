#include <string>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"
#include "core/SubringCtMatrix.h"

namespace cheddar {

namespace kernel {

// Gather Vec^d_k out of coefficient-domain components.
//
//   entry(row, col)[limb][t] = coeff[ct][limb * degree + vec + t * d]
//
// where (row, col) is (vec, ct) column-wise and (ct, vec) row-wise. The
// stride-d slice is the same index map SinC and ModDecomp use; only which of
// the two indices names the ciphertext changes.
//
// grid: (sub_degree / block, rows * cols, num_total_primes)
template <typename word>
__global__ void GatherVec(word *dst, const word *const *src_ptrs, int rows,
                          int cols, int sub_degree, int vec_dim, int degree,
                          int num_total_primes, bool row_wise) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int entry = blockIdx.y;
  const int limb = blockIdx.z;

  const int row = entry / cols;
  const int col = entry - row * cols;
  const int ct_index = row_wise ? row : col;
  const int vec_index = row_wise ? col : row;

  const word value = basic::StreamingLoad(src_ptrs[ct_index] + limb * degree +
                                          vec_index + t * vec_dim);
  dst[(static_cast<size_t>(entry) * num_total_primes + limb) * sub_degree + t] =
      value;
}

// The scatter back: coefficient (vec + t * d) of ciphertext `col`.
//
// grid: (sub_degree / block, rows * cols, num_total_primes)
template <typename word>
__global__ void ScatterVec(word *const *dst_ptrs, const word *src, int rows,
                           int cols, int sub_degree, int vec_dim, int degree,
                           int num_total_primes, bool transpose) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int entry = blockIdx.y;
  const int limb = blockIdx.z;

  const int row = entry / cols;
  const int col = entry - row * cols;
  const int ct_index = transpose ? row : col;
  const int vec_index = transpose ? col : row;

  const word value = basic::StreamingLoad(
      src + (static_cast<size_t>(entry) * num_total_primes + limb) *
                sub_degree + t);
  dst_ptrs[ct_index][limb * degree + vec_index + t * vec_dim] = value;
}

// res[i][l][t] = sum_j ( lhs[i][j] * rhs[j][l] )[t], the product taken in
// R_k = Z[Y]/(Y^k + 1). The wrap past Y^k carries a sign, which is the only
// thing that distinguishes this from an ordinary convolution.
//
// Both operands are Montgomery residues, and MultMontgomery drops the extra
// factor, so the result is one too.
//
// grid: (sub_degree / block, rows * cols, num_total_primes)
template <typename word>
__global__ void SubringConvMatMul(word *dst, const word *lhs, const word *rhs,
                              const word *primes,
                              const make_signed_t<word> *inv_primes, int rows,
                              int inner, int cols, int sub_degree,
                              int num_total_primes) {
  using signed_word = make_signed_t<word>;

  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int entry = blockIdx.y;
  const int limb = blockIdx.z;

  const int i = entry / cols;
  const int l = entry - i * cols;

  const word prime = basic::StreamingLoadConst(primes + limb);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + limb);

  word acc = 0;
  for (int j = 0; j < inner; j++) {
    const word *lhs_entry =
        lhs + (static_cast<size_t>(i * inner + j) * num_total_primes + limb) *
                  sub_degree;
    const word *rhs_entry =
        rhs + (static_cast<size_t>(j * cols + l) * num_total_primes + limb) *
                  sub_degree;

    for (int t1 = 0; t1 <= t; t1++) {
      const word product = basic::MultMontgomery(
          lhs_entry[t1], rhs_entry[t - t1], prime, inv_prime);
      acc = basic::Add(acc, product, prime);
    }
    // Y^k = -1, so everything that wrapped comes back negated.
    for (int t1 = t + 1; t1 < sub_degree; t1++) {
      const word product = basic::MultMontgomery(
          lhs_entry[t1], rhs_entry[sub_degree + t - t1], prime, inv_prime);
      acc = basic::Sub(acc, product, prime);
    }
  }

  dst[(static_cast<size_t>(entry) * num_total_primes + limb) * sub_degree + t] =
      acc;
}

// A length-k negacyclic transform, one block per (entry, limb).
//
// Cooley-Tukey forward (natural order in, bit-reversed out) against
// Gentleman-Sande inverse (bit-reversed in, natural out), with psi_rev[j] =
// psi^bitrev(j) folded into the butterflies so the negacyclic twist costs
// nothing extra. The pair inverts for any primitive 2k-th root, which is why
// this never has to match Cheddar's own choice of psi.
//
// grid: (entries, num_total_primes), block: sub_degree / 2
template <typename word>
__global__ void SubringNttForward(word *data, const word *twiddles,
                                  const word *primes,
                                  const make_signed_t<word> *inv_primes,
                                  int sub_degree, int num_total_primes) {
  using signed_word = make_signed_t<word>;
  extern __shared__ char shared_raw[];
  word *shared = reinterpret_cast<word *>(shared_raw);

  const int entry = blockIdx.x;
  const int limb = blockIdx.y;
  const int b = threadIdx.x;
  const int half = sub_degree / 2;

  const word prime = basic::StreamingLoadConst(primes + limb);
  const signed_word inv_prime = basic::StreamingLoadConst(inv_primes + limb);
  const word *tw = twiddles + static_cast<size_t>(limb) * sub_degree;
  word *entry_data =
      data + (static_cast<size_t>(entry) * num_total_primes + limb) *
                 sub_degree;

  shared[b] = entry_data[b];
  shared[b + half] = entry_data[b + half];
  __syncthreads();

  int t = sub_degree;
  for (int m = 1; m < sub_degree; m <<= 1) {
    t >>= 1;
    const int i = b / t;
    const int j = b - i * t;
    const int j1 = 2 * i * t + j;
    const word s = tw[m + i];
    const word u = shared[j1];
    const word v = basic::MultMontgomery(shared[j1 + t], s, prime, inv_prime);
    shared[j1] = basic::Add(u, v, prime);
    shared[j1 + t] = basic::Sub(u, v, prime);
    __syncthreads();
  }

  entry_data[b] = shared[b];
  entry_data[b + half] = shared[b + half];
}

// grid: (entries, num_total_primes), block: sub_degree / 2
template <typename word>
__global__ void SubringNttInverse(word *data, const word *twiddles,
                                  const word *scale, const word *primes,
                                  const make_signed_t<word> *inv_primes,
                                  int sub_degree, int num_total_primes) {
  using signed_word = make_signed_t<word>;
  extern __shared__ char shared_raw[];
  word *shared = reinterpret_cast<word *>(shared_raw);

  const int entry = blockIdx.x;
  const int limb = blockIdx.y;
  const int b = threadIdx.x;
  const int half = sub_degree / 2;

  const word prime = basic::StreamingLoadConst(primes + limb);
  const signed_word inv_prime = basic::StreamingLoadConst(inv_primes + limb);
  const word *tw = twiddles + static_cast<size_t>(limb) * sub_degree;
  const word k_inv = basic::StreamingLoadConst(scale + limb);
  word *entry_data =
      data + (static_cast<size_t>(entry) * num_total_primes + limb) *
                 sub_degree;

  shared[b] = entry_data[b];
  shared[b + half] = entry_data[b + half];
  __syncthreads();

  int t = 1;
  for (int m = sub_degree; m > 1; m >>= 1) {
    const int h = m >> 1;
    const int i = b / t;
    const int j = b - i * t;
    const int j1 = 2 * i * t + j;
    const word s = tw[h + i];
    const word u = shared[j1];
    const word v = shared[j1 + t];
    shared[j1] = basic::Add(u, v, prime);
    shared[j1 + t] =
        basic::MultMontgomery(basic::Sub(u, v, prime), s, prime, inv_prime);
    __syncthreads();
    t <<= 1;
  }

  entry_data[b] = basic::MultMontgomery(shared[b], k_inv, prime, inv_prime);
  entry_data[b + half] =
      basic::MultMontgomery(shared[b + half], k_inv, prime, inv_prime);
}

// In the transform domain the R_k product is pointwise, so the matrix product
// is k independent scalar ones -- the form [KANG] hands to BLAS.
//
//   res[i][l][t] = sum_j lhs[i][j][t] * rhs[j][l][t]
//
// grid: (sub_degree / block, rows * cols, num_total_primes)
template <typename word>
__global__ void SubringPointwiseMatMul(word *dst, const word *lhs,
                                       const word *rhs, const word *primes,
                                       const make_signed_t<word> *inv_primes,
                                       int rows, int inner, int cols,
                                       int sub_degree, int num_total_primes) {
  using signed_word = make_signed_t<word>;

  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int entry = blockIdx.y;
  const int limb = blockIdx.z;

  const int i = entry / cols;
  const int l = entry - i * cols;

  const word prime = basic::StreamingLoadConst(primes + limb);
  const signed_word inv_prime = basic::StreamingLoadConst(inv_primes + limb);

  word acc = 0;
  for (int j = 0; j < inner; j++) {
    const word left = basic::StreamingLoad(
        lhs + (static_cast<size_t>(i * inner + j) * num_total_primes + limb) *
                  sub_degree + t);
    const word right = basic::StreamingLoad(
        rhs + (static_cast<size_t>(j * cols + l) * num_total_primes + limb) *
                  sub_degree + t);
    acc = basic::Add(acc, basic::MultMontgomery(left, right, prime, inv_prime),
                     prime);
  }

  dst[(static_cast<size_t>(entry) * num_total_primes + limb) * sub_degree + t] =
      acc;
}

}  // namespace kernel

namespace {

// gridDim.y is capped at 65535 on every CUDA architecture, and `entries`
// (= rows * cols of the subring matrix) is launched in it by four of the
// kernels below. Nothing in this library checks a launch error, so exceeding
// it does not fail loudly -- the launch is rejected, the kernel never runs and
// the output buffer keeps whatever it held. For [KANG] Algorithm 4 the matrix
// is d x d with d = degree / sub_degree, so the real constraint is d^2 <= 65535,
// i.e. d <= 255 and hence d <= 128 for the power-of-two d the algorithm uses.
// Llama-3's per-head 128 x 128 attention product sits at d = 128, inside it
// with a factor of four to spare; d = 256 would be silently wrong.
void AssertGridYFits(size_t entries, const char *what) {
  AssertTrue(entries <= 65535,
             std::string(what) + ": rows * cols = " + std::to_string(entries) +
                 " exceeds the gridDim.y limit of 65535, and a rejected launch "
                 "here is silent -- the subring matrix is d x d with "
                 "d = degree / sub_degree, so this needs d <= 255");
}

}  // namespace

template <typename word>
SubringCtMatrixHandler<word>::SubringCtMatrixHandler(
    const Parameter<word> &param, const NTTHandler<word> &ntt_handler)
    : param_{param}, ntt_handler_{ntt_handler} {}

template <typename word>
void SubringCtMatrixHandler<word>::ToMatrices(SubringCoeffMatrix<word> &b_mat,
                                              SubringCoeffMatrix<word> &a_mat,
                                              const std::vector<Ct> &cts,
                                              int sub_degree,
                                              bool row_wise) const {
  const int degree = param_.degree_;
  const int num_cts = static_cast<int>(cts.size());
  AssertTrue(num_cts > 0, "ToMatrices: no input ciphertexts");
  AssertTrue(sub_degree >= 1 && sub_degree <= degree && IsPowOfTwo(sub_degree),
             "ToMatrices: invalid sub_degree");
  AssertTrue(sub_degree % kernel_block_dim_ == 0 ||
                 sub_degree < kernel_block_dim_,
             "ToMatrices: sub_degree must suit the block dim");

  const int vec_dim = degree / sub_degree;  // d
  const NPInfo np = cts.at(0).GetNP();
  AssertTrue(np.num_aux_ == 0, "ToMatrices: aux primes are not supported");
  for (const auto &ct : cts) {
    AssertTrue(ct.GetNP() == np, "ToMatrices: ciphertexts differ in NP");
    AssertTrue(!ct.HasRx(),
               "ToMatrices: ciphertexts must not carry an rx_ part");
  }

  const int rows = row_wise ? num_cts : vec_dim;
  const int cols = row_wise ? vec_dim : num_cts;
  const int num_total_primes = np.GetNumTotal();
  const size_t entries = static_cast<size_t>(rows) * cols;
  AssertGridYFits(entries, "SubringCtMatrix");

  // montgomery_conversion = false keeps Montgomery residues, which is what the
  // convolution below needs; the default would hand back plain ones.
  std::vector<DeviceVector<word>> b_coeffs(num_cts), a_coeffs(num_cts);
  HostVector<word *> h_b(num_cts), h_a(num_cts);
  for (int j = 0; j < num_cts; j++) {
    b_coeffs[j].resize(num_total_primes * degree);
    a_coeffs[j].resize(num_total_primes * degree);
    auto b_view = b_coeffs[j].View(0);
    auto a_view = a_coeffs[j].View(0);
    ntt_handler_.INTT(b_view, np, cts[j].BxConstView(), false);
    ntt_handler_.INTT(a_view, np, cts[j].AxConstView(), false);
    h_b[j] = b_coeffs[j].data();
    h_a[j] = a_coeffs[j].data();
  }
  DeviceVector<word *> d_b(num_cts), d_a(num_cts);
  CopyHostToDevice(d_b, h_b);
  CopyHostToDevice(d_a, h_a);

  for (auto *m : {&b_mat, &a_mat}) {
    m->rows_ = rows;
    m->cols_ = cols;
    m->sub_degree_ = sub_degree;
    m->np_ = np;
    m->scale_ = cts.at(0).GetScale();
    m->data_.resize(
        static_cast<int>(entries * num_total_primes * sub_degree));
  }

  const int block = Min(sub_degree, kernel_block_dim_);
  const dim3 grid(sub_degree / block, static_cast<unsigned>(entries),
                  num_total_primes);
  kernel::GatherVec<word><<<grid, block>>>(
      b_mat.data_.data(), d_b.data(), rows, cols, sub_degree, vec_dim, degree,
      num_total_primes, row_wise);
  kernel::GatherVec<word><<<grid, block>>>(
      a_mat.data_.data(), d_a.data(), rows, cols, sub_degree, vec_dim, degree,
      num_total_primes, row_wise);
}

template <typename word>
void SubringCtMatrixHandler<word>::MultiplyMatricesReference(
    SubringCoeffMatrix<word> &res, const SubringCoeffMatrix<word> &lhs,
    const SubringCoeffMatrix<word> &rhs) const {
  AssertTrue(lhs.cols_ == rhs.rows_,
             "MultiplyMatrices: inner dimensions do not match");
  AssertTrue(lhs.sub_degree_ == rhs.sub_degree_,
             "MultiplyMatrices: operands differ in sub_degree");
  AssertTrue(lhs.np_ == rhs.np_, "MultiplyMatrices: operands differ in NP");
  AssertTrue(&res != &lhs && &res != &rhs,
             "MultiplyMatrices: in-place operation is not supported");

  const int rows = lhs.rows_;
  const int inner = lhs.cols_;
  const int cols = rhs.cols_;
  const int sub_degree = lhs.sub_degree_;
  const NPInfo np = lhs.np_;
  const int num_total_primes = np.GetNumTotal();
  const size_t entries = static_cast<size_t>(rows) * cols;
  AssertGridYFits(entries, "SubringCtMatrix");

  res.rows_ = rows;
  res.cols_ = cols;
  res.sub_degree_ = sub_degree;
  res.np_ = np;
  res.scale_ = lhs.scale_ * rhs.scale_;
  res.data_.resize(static_cast<int>(entries * num_total_primes * sub_degree));

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);

  const int block = Min(sub_degree, kernel_block_dim_);
  const dim3 grid(sub_degree / block, static_cast<unsigned>(entries),
                  num_total_primes);
  kernel::SubringConvMatMul<word><<<grid, block>>>(
      res.data_.data(), lhs.data_.data(), rhs.data_.data(), primes, inv_primes,
      rows, inner, cols, sub_degree, num_total_primes);
}

template <typename word>
void SubringCtMatrixHandler<word>::ToCiphertexts(
    std::vector<Ct> &res, const SubringCoeffMatrix<word> &b_mat,
    const SubringCoeffMatrix<word> &a_mat, double scale, int num_slots,
    bool transpose) const {
  const int degree = param_.degree_;
  AssertTrue(b_mat.rows_ == a_mat.rows_ && b_mat.cols_ == a_mat.cols_ &&
                 b_mat.sub_degree_ == a_mat.sub_degree_ &&
                 b_mat.np_ == a_mat.np_,
             "ToCiphertexts: the two matrices do not agree");

  const int rows = b_mat.rows_;
  const int cols = b_mat.cols_;
  const int sub_degree = b_mat.sub_degree_;
  const NPInfo np = b_mat.np_;
  const int num_total_primes = np.GetNumTotal();
  const int vec_dim = degree / sub_degree;
  const int num_cts = transpose ? rows : cols;
  AssertTrue((transpose ? cols : rows) == vec_dim,
             "ToCiphertexts: the Vec side must have the Vec dimension");

  const size_t entries = static_cast<size_t>(rows) * cols;
  AssertGridYFits(entries, "SubringCtMatrix");

  std::vector<DeviceVector<word>> b_coeffs(num_cts), a_coeffs(num_cts);
  HostVector<word *> h_b(num_cts), h_a(num_cts);
  for (int l = 0; l < num_cts; l++) {
    b_coeffs[l].resize(num_total_primes * degree);
    a_coeffs[l].resize(num_total_primes * degree);
    h_b[l] = b_coeffs[l].data();
    h_a[l] = a_coeffs[l].data();
  }
  DeviceVector<word *> d_b(num_cts), d_a(num_cts);
  CopyHostToDevice(d_b, h_b);
  CopyHostToDevice(d_a, h_a);

  const int block = Min(sub_degree, kernel_block_dim_);
  const dim3 grid(sub_degree / block, static_cast<unsigned>(entries),
                  num_total_primes);
  kernel::ScatterVec<word><<<grid, block>>>(
      d_b.data(), b_mat.data_.data(), rows, cols, sub_degree, vec_dim, degree,
      num_total_primes, transpose);
  kernel::ScatterVec<word><<<grid, block>>>(
      d_a.data(), a_mat.data_.data(), rows, cols, sub_degree, vec_dim, degree,
      num_total_primes, transpose);

  res.resize(num_cts);
  for (int l = 0; l < num_cts; l++) {
    res[l].RemoveRx();
    res[l].ModifyNP(np);
    res[l].SetScale(scale);
    res[l].SetNumSlots(num_slots);
    auto b_view = res[l].BxView();
    auto a_view = res[l].AxView();
    ntt_handler_.NTT(b_view, np, b_coeffs[l].ConstView(), false);
    ntt_handler_.NTT(a_view, np, a_coeffs[l].ConstView(), false);
  }
}

template <typename word>
typename SubringCtMatrixHandler<word>::SubringTwiddles
SubringCtMatrixHandler<word>::BuildTwiddles(const NPInfo &np,
                                            int sub_degree) const {
  const auto primes = param_.GetPrimeVector(np);
  const int num_total_primes = np.GetNumTotal();

  HostVector<word> forward(num_total_primes * sub_degree);
  HostVector<word> inverse(num_total_primes * sub_degree);
  HostVector<word> scale(num_total_primes);

  for (int limb = 0; limb < num_total_primes; limb++) {
    const word prime = primes[limb];
    // Any primitive 2k-th root will do: the two directions are built from the
    // same one, so they invert each other whichever it is.
    const word psi = primeutil::FindPrimitiveMthRoot<word>(2 * sub_degree,
                                                           prime);
    const word psi_inv = primeutil::InvMod<word>(psi, prime);

    std::vector<word> powers(sub_degree), inv_powers(sub_degree);
    powers[0] = 1;
    inv_powers[0] = 1;
    for (int j = 1; j < sub_degree; j++) {
      powers[j] = primeutil::MultMod<word>(powers[j - 1], psi, prime);
      inv_powers[j] = primeutil::MultMod<word>(inv_powers[j - 1], psi_inv,
                                               prime);
    }
    BitReverseVector(powers);
    BitReverseVector(inv_powers);

    for (int j = 0; j < sub_degree; j++) {
      forward[limb * sub_degree + j] =
          primeutil::ToMontgomery<word>(powers[j], prime);
      inverse[limb * sub_degree + j] =
          primeutil::ToMontgomery<word>(inv_powers[j], prime);
    }
    scale[limb] = primeutil::ToMontgomery<word>(
        primeutil::InvMod<word>(static_cast<word>(sub_degree), prime), prime);
  }

  SubringTwiddles tw;
  tw.forward.resize(num_total_primes * sub_degree);
  tw.inverse.resize(num_total_primes * sub_degree);
  tw.scale.resize(num_total_primes);
  CopyHostToDevice(tw.forward, forward);
  CopyHostToDevice(tw.inverse, inverse);
  CopyHostToDevice(tw.scale, scale);
  return tw;
}

template <typename word>
void SubringCtMatrixHandler<word>::TransformEntries(
    SubringCoeffMatrix<word> &mat, const SubringTwiddles &tw,
    bool inverse) const {
  const int sub_degree = mat.sub_degree_;
  const int num_total_primes = mat.np_.GetNumTotal();
  const size_t entries = static_cast<size_t>(mat.rows_) * mat.cols_;

  const word *primes = param_.GetPrimesPtr(mat.np_);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(mat.np_);

  const dim3 grid(static_cast<unsigned>(entries), num_total_primes);
  const int block = sub_degree / 2;
  const int shared = sub_degree * sizeof(word);

  if (inverse) {
    kernel::SubringNttInverse<word><<<grid, block, shared>>>(
        mat.data_.data(), tw.inverse.data(), tw.scale.data(), primes,
        inv_primes, sub_degree, num_total_primes);
  } else {
    kernel::SubringNttForward<word><<<grid, block, shared>>>(
        mat.data_.data(), tw.forward.data(), primes, inv_primes, sub_degree,
        num_total_primes);
  }
}

template <typename word>
void SubringCtMatrixHandler<word>::MultiplyMatrices(
    SubringCoeffMatrix<word> &res, const SubringCoeffMatrix<word> &lhs,
    const SubringCoeffMatrix<word> &rhs) const {
  AssertTrue(lhs.cols_ == rhs.rows_,
             "MultiplyMatrices: inner dimensions do not match");
  AssertTrue(lhs.sub_degree_ == rhs.sub_degree_,
             "MultiplyMatrices: operands differ in sub_degree");
  AssertTrue(lhs.np_ == rhs.np_, "MultiplyMatrices: operands differ in NP");
  AssertTrue(&res != &lhs && &res != &rhs,
             "MultiplyMatrices: in-place operation is not supported");
  AssertTrue(lhs.sub_degree_ >= 2,
             "MultiplyMatrices: sub_degree must be at least 2");

  const int rows = lhs.rows_;
  const int inner = lhs.cols_;
  const int cols = rhs.cols_;
  const int sub_degree = lhs.sub_degree_;
  const NPInfo np = lhs.np_;
  const int num_total_primes = np.GetNumTotal();
  const size_t entries = static_cast<size_t>(rows) * cols;
  AssertGridYFits(entries, "SubringCtMatrix");

  const SubringTwiddles tw = BuildTwiddles(np, sub_degree);

  // The transform is in place, and the operands are const, so they are copied.
  SubringCoeffMatrix<word> left, right;
  for (const auto *src : {&lhs, &rhs}) {
    SubringCoeffMatrix<word> &dst = (src == &lhs) ? left : right;
    dst.rows_ = src->rows_;
    dst.cols_ = src->cols_;
    dst.sub_degree_ = sub_degree;
    dst.np_ = np;
    dst.scale_ = src->scale_;
    dst.data_.resize(static_cast<int>(src->data_.size()));
    CopyDeviceToDevice(dst.data_, src->data_);
    TransformEntries(dst, tw, false);
  }

  res.rows_ = rows;
  res.cols_ = cols;
  res.sub_degree_ = sub_degree;
  res.np_ = np;
  res.scale_ = lhs.scale_ * rhs.scale_;
  res.data_.resize(static_cast<int>(entries * num_total_primes * sub_degree));

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);
  const int block = Min(sub_degree, kernel_block_dim_);
  const dim3 grid(sub_degree / block, static_cast<unsigned>(entries),
                  num_total_primes);
  kernel::SubringPointwiseMatMul<word><<<grid, block>>>(
      res.data_.data(), left.data_.data(), right.data_.data(), primes,
      inv_primes, rows, inner, cols, sub_degree, num_total_primes);

  TransformEntries(res, tw, true);
}

template class SubringCoeffMatrix<uint32_t>;
template class SubringCoeffMatrix<uint64_t>;
template class SubringCtMatrixHandler<uint32_t>;
template class SubringCtMatrixHandler<uint64_t>;

}  // namespace cheddar
