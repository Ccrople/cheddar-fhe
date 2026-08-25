#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "core/CiLift.h"

namespace cheddar {

namespace kernel {

// c_j -> X^j - X^{N-j}: dst[j] = src[j], dst[N-j] = -src[j]. The j = 0
// thread writes positions 0 and N/2 (c_n = 0, so the odd direction has no
// image), which makes every output position written exactly once.
//
// grid: (n / block, num_total_primes)
template <typename word>
__global__ void CiLiftMap(word *dst, const word *src, const word *primes,
                          int small_degree, int degree) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;

  const word *src_limb = src + limb * small_degree;
  word *dst_limb = dst + limb * degree;

  const word value = basic::StreamingLoad(src_limb + j);
  if (j == 0) {
    dst_limb[0] = value;
    dst_limb[small_degree] = 0;
    return;
  }
  const word prime = basic::StreamingLoadConst(primes + limb);
  dst_limb[j] = value;
  dst_limb[degree - j] = basic::Negate(value, prime);
}

// The trace T = id + conj fused with the c-basis read-off:
// dst[j] = src[j] - src[N-j] for j >= 1, dst[0] = 2 * src[0]. Position N/2
// is annihilated (c_n = 0).
//
// The trace, NOT the average. E = T/2 is the same projection as an element
// of R_q, but 2^-1 mod q of an odd small integer is a residue near q/2, so
// E does not preserve coefficient size: the ciphertext noise e has odd
// antisymmetric parts in about half its positions, E(e) puts ~q/2 there,
// and the descended ciphertext decrypts to garbage. T keeps everything
// small -- T(e) = e + conj(e) -- and the factor 2 rides the recorded scale,
// where it is exact.
//
// grid: (n / block, num_total_primes)
template <typename word>
__global__ void CiDescendMap(word *dst, const word *src, const word *primes,
                             int small_degree, int degree) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int limb = blockIdx.y;

  const word *src_limb = src + limb * degree;
  word *dst_limb = dst + limb * small_degree;

  const word prime = basic::StreamingLoadConst(primes + limb);
  if (j == 0) {
    const word v = basic::StreamingLoad(src_limb);
    dst_limb[0] = basic::Add(v, v, prime);
    return;
  }
  dst_limb[j] = basic::Sub(basic::StreamingLoad(src_limb + j),
                           basic::StreamingLoad(src_limb + degree - j),
                           prime);
}

}  // namespace kernel

template <typename word>
CiLiftHandler<word>::CiLiftHandler(ConstContextPtr<word> ci,
                                   ConstContextPtr<word> big)
    : ci_{std::move(ci)}, big_{std::move(big)} {
  const int small_degree = ci_->param_.degree_;
  const int degree = big_->param_.degree_;
  AssertTrue(degree == 2 * small_degree,
             "CiLiftHandler: the ordinary ring of the same conductor has "
             "twice the conjugate-invariant degree");
  AssertTrue(ci_->param_.conjugate_invariant_,
             "CiLiftHandler: the source must be a conjugate-invariant ring");
  AssertTrue(!big_->param_.conjugate_invariant_,
             "CiLiftHandler: the target must be the ordinary ring");
  AssertTrue(small_degree % kernel_block_dim_ == 0,
             "CiLiftHandler: invalid kernel block dim");

  // The maps only reindex coefficients, so the limbs cross unchanged and the
  // two rings must agree on the primes at every level -- the same condition
  // RingSwitchHandler enforces, for the same reason. 1 mod 4n and 1 mod 2N
  // are the same congruence at N = 2n, so one prime list can serve both.
  AssertTrue(ci_->param_.max_level_ == big_->param_.max_level_,
             "CiLiftHandler: the two rings must share a level configuration");
  for (int level = 0; level <= ci_->param_.max_level_; level++) {
    const NPInfo cnp = ci_->param_.LevelToNP(level);
    const NPInfo bnp = big_->param_.LevelToNP(level);
    AssertTrue(cnp.num_main_ == bnp.num_main_ && cnp.num_ter_ == bnp.num_ter_,
               "CiLiftHandler: prime counts differ at level " +
                   std::to_string(level));
    AssertTrue(ci_->param_.GetPrimeVector(cnp) ==
                   big_->param_.GetPrimeVector(bnp),
               "CiLiftHandler: the two rings hold different primes at level " +
                   std::to_string(level) +
                   ", so a lifted ciphertext would have no home");
  }
}

template <typename word>
void CiLiftHandler<word>::Lift(Ct &res, const Ct &ct) const {
  const int small_degree = ci_->param_.degree_;
  const int degree = big_->param_.degree_;
  const NPInfo np = ct.GetNP();
  AssertTrue(np.num_aux_ == 0, "CiLift: aux primes are not supported");
  AssertTrue(!ct.HasRx(), "CiLift: input must not carry an rx_ part");
  AssertTrue(np.degree_ == small_degree,
             "CiLift: input does not belong to the conjugate-invariant ring");
  const int level = ci_->param_.NPToLevel(np);
  AssertTrue(level >= 0, "CiLift: input is not at a valid level");
  const int num_total_primes = np.GetNumTotal();

  // The map is defined on coefficients, so leave the NTT domain at degree n.
  DeviceVector<word> a_coeffs(num_total_primes * small_degree);
  DeviceVector<word> b_coeffs(num_total_primes * small_degree);
  auto a_view = a_coeffs.View(0);
  auto b_view = b_coeffs.View(0);
  ci_->ntt_handler_.INTT(a_view, np, ct.AxConstView());
  ci_->ntt_handler_.INTT(b_view, np, ct.BxConstView());

  DeviceVector<word> a_lift(num_total_primes * degree);
  DeviceVector<word> b_lift(num_total_primes * degree);
  const word *primes = ci_->param_.GetPrimesPtr(np);
  const dim3 grid_dim(small_degree / kernel_block_dim_, num_total_primes);
  kernel::CiLiftMap<word><<<grid_dim, kernel_block_dim_>>>(
      a_lift.data(), a_coeffs.data(), primes, small_degree, degree);
  kernel::CiLiftMap<word><<<grid_dim, kernel_block_dim_>>>(
      b_lift.data(), b_coeffs.data(), primes, small_degree, degree);

  // Back into the NTT domain at degree N, as a ciphertext of the big
  // Context. The constructor verified the limbs are the same primes.
  const NPInfo big_np = big_->param_.LevelToNP(level);
  res.RemoveRx();
  res.ModifyNP(big_np);
  res.SetScale(ct.GetScale());
  res.SetNumSlots(big_->param_.MaxNumSlots());
  auto ax_view = res.AxView();
  auto bx_view = res.BxView();
  big_->ntt_handler_.NTT(ax_view, big_np, a_lift.ConstView(0), true);
  big_->ntt_handler_.NTT(bx_view, big_np, b_lift.ConstView(0), true);
}

template <typename word>
void CiLiftHandler<word>::Descend(Ct &res, const Ct &ct) const {
  const int small_degree = ci_->param_.degree_;
  const int degree = big_->param_.degree_;
  const NPInfo np = ct.GetNP();
  AssertTrue(np.num_aux_ == 0, "CiDescend: aux primes are not supported");
  AssertTrue(!ct.HasRx(), "CiDescend: input must not carry an rx_ part");
  AssertTrue(np.degree_ == degree,
             "CiDescend: input does not belong to the ordinary ring");
  const int level = big_->param_.NPToLevel(np);
  AssertTrue(level >= 0, "CiDescend: input is not at a valid level");
  const int num_total_primes = np.GetNumTotal();

  DeviceVector<word> a_coeffs(num_total_primes * degree);
  DeviceVector<word> b_coeffs(num_total_primes * degree);
  auto a_view = a_coeffs.View(0);
  auto b_view = b_coeffs.View(0);
  big_->ntt_handler_.INTT(a_view, np, ct.AxConstView());
  big_->ntt_handler_.INTT(b_view, np, ct.BxConstView());

  DeviceVector<word> a_down(num_total_primes * small_degree);
  DeviceVector<word> b_down(num_total_primes * small_degree);
  const word *primes = big_->param_.GetPrimesPtr(np);
  const dim3 grid_dim(small_degree / kernel_block_dim_, num_total_primes);
  kernel::CiDescendMap<word><<<grid_dim, kernel_block_dim_>>>(
      a_down.data(), a_coeffs.data(), primes, small_degree, degree);
  kernel::CiDescendMap<word><<<grid_dim, kernel_block_dim_>>>(
      b_down.data(), b_coeffs.data(), primes, small_degree, degree);

  const NPInfo ci_np = ci_->param_.LevelToNP(level);
  res.RemoveRx();
  res.ModifyNP(ci_np);
  // The trace doubles the message; the factor rides the scale, exactly.
  res.SetScale(2.0 * ct.GetScale());
  res.SetNumSlots(ci_->param_.MaxNumSlots());
  auto ax_view = res.AxView();
  auto bx_view = res.BxView();
  ci_->ntt_handler_.NTT(ax_view, ci_np, a_down.ConstView(0), true);
  ci_->ntt_handler_.NTT(bx_view, ci_np, b_down.ConstView(0), true);
}

template <typename word>
std::vector<int> CiLiftHandler<word>::LiftSecret(
    const std::vector<int> &ci_secret) {
  const int n = static_cast<int>(ci_secret.size());
  std::vector<int> lifted(2 * n, 0);
  lifted[0] = ci_secret[0];
  for (int j = 1; j < n; j++) {
    lifted[j] = ci_secret[j];
    lifted[2 * n - j] = -ci_secret[j];
  }
  return lifted;
}

template class CiLiftHandler<uint32_t>;
template class CiLiftHandler<uint64_t>;

}  // namespace cheddar
