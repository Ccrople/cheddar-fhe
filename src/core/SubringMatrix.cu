#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "core/SubringMatrix.h"

namespace cheddar {

namespace kernel {

// One output coefficient of one output ciphertext component, for one limb:
//
//   dst[l][limb][x] = sum_j u[j][l][limb][x] * src[j][limb][x]   (mod p_limb)
//
// Pointwise in x, because in the NTT domain a plaintext multiplication is
// pointwise and the contraction runs over the ciphertext index only. That is
// the whole of [KANG] Algorithm 1 for one component; the caller runs it once
// for bx and once for ax against the same weights.
//
// `u` is in Montgomery form, as every encoded plaintext is, and MultMontgomery
// drops the extra factor so the result stays in the ciphertext's own
// representation.
//
// grid: (degree / block, cols_out, num_total_primes)
template <typename word>
__global__ void SubringPAccum(word *const *dst_ptrs,
                              const word *const *src_ptrs, const word *u,
                              const word *primes,
                              const make_signed_t<word> *inv_primes,
                              int cols_in, int cols_out, int num_total_primes,
                              int degree) {
  using signed_word = make_signed_t<word>;

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int l = blockIdx.y;
  const int prime_index = blockIdx.z;

  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);

  const int limb_offset = prime_index * degree + x;
  const size_t entry_stride = static_cast<size_t>(num_total_primes) * degree;

  word acc = 0;
  for (int j = 0; j < cols_in; j++) {
    const word u_value = basic::StreamingLoad(
        u + (static_cast<size_t>(j) * cols_out + l) * entry_stride +
        limb_offset);
    const word src_value = basic::StreamingLoad(src_ptrs[j] + limb_offset);
    const word product =
        basic::MultMontgomery(u_value, src_value, prime, inv_prime);
    acc = basic::Add(acc, product, prime);
  }

  dst_ptrs[l][limb_offset] = acc;
}

}  // namespace kernel

template <typename word>
int SubringWeights<word>::GetColsIn() const {
  return cols_in_;
}

template <typename word>
int SubringWeights<word>::GetColsOut() const {
  return cols_out_;
}

template <typename word>
int SubringWeights<word>::GetSubDegree() const {
  return sub_degree_;
}

template <typename word>
double SubringWeights<word>::GetScale() const {
  return scale_;
}

template <typename word>
NPInfo SubringWeights<word>::GetNP() const {
  return np_;
}

template <typename word>
SubringMatrixHandler<word>::SubringMatrixHandler(const Parameter<word> &param,
                                                 const Encoder<word> &encoder)
    : param_{param}, encoder_{encoder} {
  AssertTrue(param_.degree_ % kernel_block_dim_ == 0,
             "SubringMatrixHandler: Invalid kernel block dim");
}

template <typename word>
void SubringMatrixHandler<word>::EncodeWeights(
    SubringWeights<word> &res, int level, double scale,
    const std::vector<std::vector<Complex>> &values, int cols_in, int cols_out,
    int sub_degree, int num_aux /*= 0*/) const {
  const int degree = param_.degree_;
  AssertTrue(cols_in > 0 && cols_out > 0,
             "EncodeWeights: Invalid matrix shape");
  AssertTrue(sub_degree >= 2 && sub_degree <= degree &&
                 IsPowOfTwo(sub_degree) && degree % sub_degree == 0,
             "EncodeWeights: sub_degree must be a power of two dividing the "
             "ring degree");

  const int lanes = sub_degree / 2;      // k/2, the batch size
  const int num_blocks = degree / sub_degree;  // d, the Vec dimension
  AssertTrue(static_cast<int>(values.size()) == lanes,
             "EncodeWeights: expected one matrix per lane");
  for (const auto &lane : values) {
    AssertTrue(static_cast<int>(lane.size()) == cols_in * cols_out,
               "EncodeWeights: each lane must hold cols_in * cols_out entries");
  }

  const NPInfo np = param_.LevelToNP(level, num_aux);
  const int num_total_primes = np.GetNumTotal();
  const size_t entry_words = static_cast<size_t>(num_total_primes) * degree;

  res.cols_in_ = cols_in;
  res.cols_out_ = cols_out;
  res.sub_degree_ = sub_degree;
  res.scale_ = scale;
  res.np_ = np;
  res.data_.resize(static_cast<int>(entry_words * cols_in * cols_out));

  // A subring element is a message that repeats with the block period, so it
  // is encoded by handing SinC the lane vector once per block. Doing it this
  // way rather than writing coefficients directly keeps the encoding on the
  // one path that SinCEncodeTest pins.
  std::vector<Complex> message(degree / 2);
  Pt entry;
  for (int j = 0; j < cols_in; j++) {
    for (int l = 0; l < cols_out; l++) {
      for (int block = 0; block < num_blocks; block++) {
        for (int t = 0; t < lanes; t++) {
          message[block * lanes + t] = values[t][j * cols_out + l];
        }
      }
      encoder_.EncodeSinC(entry, level, scale, message, sub_degree, num_aux);

      const size_t offset =
          (static_cast<size_t>(j) * cols_out + l) * entry_words;
      cudaMemcpyAsync(res.data_.data() + offset, entry.mx_.data(),
                      entry_words * sizeof(word), cudaMemcpyDeviceToDevice,
                      cudaStreamLegacy);
    }
  }
}

template <typename word>
void SubringMatrixHandler<word>::Multiply(ConstContextPtr<word> context,
                                          std::vector<Ct> &res,
                                          const SubringWeights<word> &u,
                                          const std::vector<Ct> &cts) const {
  const int cols_in = u.cols_in_;
  const int cols_out = u.cols_out_;
  AssertTrue(cols_in > 0 && cols_out > 0,
             "Subring::Multiply: Invalid matrix shape");
  AssertTrue(static_cast<int>(cts.size()) == cols_in,
             "Subring::Multiply: expected u.cols_in_ input ciphertexts");

  const NPInfo np = cts.at(0).GetNP();
  AssertTrue(np == u.np_, "Subring::Multiply: NP mismatch between u and cts");
  AssertTrue(np.num_aux_ == 0,
             "Subring::Multiply: aux primes are not supported");
  const double ct_scale = cts.at(0).GetScale();
  const int num_slots = cts.at(0).GetNumSlots();
  for (const auto &ct : cts) {
    AssertTrue(ct.GetNP() == np,
               "Subring::Multiply: ciphertexts differ in NP");
    AssertTrue(!ct.HasRx(),
               "Subring::Multiply: ciphertexts must not carry an rx_ part");
  }

  const int level = param_.NPToLevel(np);
  AssertTrue(level >= 1,
             "Subring::Multiply: the product rescales, so the input must be "
             "above level 0");

  const int degree = param_.degree_;
  const int num_total_primes = np.GetNumTotal();

  // The accumulation cannot write into `res`: it is rescaled into `res`
  // afterwards, and mod-switching reads a different number of limbs than it
  // writes.
  std::vector<Ct> prod(cols_out);
  for (auto &p : prod) {
    p.RemoveRx();
    p.ModifyNP(np);
    p.SetNumSlots(num_slots);
    p.SetScale(u.scale_ * ct_scale);
  }

  HostVector<word *> h_dst_bx(cols_out), h_dst_ax(cols_out);
  HostVector<word *> h_src_bx(cols_in), h_src_ax(cols_in);
  for (int l = 0; l < cols_out; l++) {
    h_dst_bx[l] = prod[l].bx_.data();
    h_dst_ax[l] = prod[l].ax_.data();
  }
  for (int j = 0; j < cols_in; j++) {
    h_src_bx[j] = const_cast<word *>(cts[j].bx_.data());
    h_src_ax[j] = const_cast<word *>(cts[j].ax_.data());
  }

  DeviceVector<word *> d_dst_bx(cols_out), d_dst_ax(cols_out);
  DeviceVector<word *> d_src_bx(cols_in), d_src_ax(cols_in);
  CopyHostToDevice(d_dst_bx, h_dst_bx);
  CopyHostToDevice(d_dst_ax, h_dst_ax);
  CopyHostToDevice(d_src_bx, h_src_bx);
  CopyHostToDevice(d_src_ax, h_src_ax);

  const word *primes = param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = param_.GetInvPrimesPtr(np);
  const dim3 grid_dim(degree / kernel_block_dim_, cols_out, num_total_primes);

  // B*U and A*U are the same product against the same weights.
  kernel::SubringPAccum<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_bx.data(), d_src_bx.data(), u.data_.data(), primes, inv_primes,
      cols_in, cols_out, num_total_primes, degree);
  kernel::SubringPAccum<word><<<grid_dim, kernel_block_dim_>>>(
      d_dst_ax.data(), d_src_ax.data(), u.data_.data(), primes, inv_primes,
      cols_in, cols_out, num_total_primes, degree);

  res.resize(cols_out);
  for (int l = 0; l < cols_out; l++) {
    context->Rescale(res[l], prod[l]);
  }
}

template class SubringWeights<uint32_t>;
template class SubringWeights<uint64_t>;
template class SubringMatrixHandler<uint32_t>;
template class SubringMatrixHandler<uint64_t>;

}  // namespace cheddar
