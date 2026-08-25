#pragma once

#include "core/DeviceVector.h"
#include "core/Parameter.h"

namespace cheddar {

namespace kernel {

// dst = f(dst, src, const_src, prime, montgormey)
template <typename word>
using elem_func_t = void (*)(make_signed_t<word> &, const make_signed_t<word>,
                             const word, const word, const make_signed_t<word>);

}  // namespace kernel

enum class NTTType { NTT, INTT };
enum class Phase { Phase1, Phase2 };

template <typename word>
class NTTHandler {
 private:
  using Dv = DeviceVector<word>;
  using Hv = HostVector<word>;


  const Parameter<word> &param_;

  Dv twiddle_factors_;
  Dv twiddle_factors_msb_;
  Dv inv_twiddle_factors_;
  Dv inv_twiddle_factors_msb_;
  Dv inv_degree_;
  Dv inv_degree_mont_;
  Dv montgomery_converter_;

  // Conjugate-invariant ring only, all in Montgomery form and all indexed on
  // the prime axis exactly as twiddle_factors_ is. Per prime:
  //   ci_i_    = psi4^degree, the square root of -1 that Y^degree reduces to
  //   ci_inv2_ = 2^-1, which the unfold divides the mirrored pair by
  // and per prime and coefficient, in natural rather than bit-reversed order:
  //   ci_fwd_twist_ = psi4^-j, folded in on the way into the network
  //   ci_inv_twist_ = psi4^j, taken back out on the way from it
  Dv ci_i_;
  Dv ci_inv2_;
  Dv ci_fwd_twist_;
  Dv ci_inv_twist_;

  // dst = fold(src): the ring element reduced mod (Y^degree - i) and twisted,
  // which is the form the butterfly network diagonalises. Runs before NTT
  // phase 1 and cannot be fused into it -- see CiFoldKernel for why.
  void CiFold(make_signed_t<word> *dst, const word *primes,
              const make_signed_t<word> *inv_primes, int tw_prime_offset,
              int tw_y_extra, int num_q_primes, int num_total_primes,
              int skip_start, int skip_end, int batch_stride, int batch,
              const make_signed_t<word> *src, int src_extra) const;

  // The inverse, in place on the INTT output.
  // Mirrored pairs (j, degree - j) are recombined by one thread each, so it
  // needs no scratch buffer. This one cannot ride a phase kernel the way the
  // fold does: it mixes two *outputs*, which live in different blocks.
  void CiUnfold(make_signed_t<word> *dst, const word *primes,
                const make_signed_t<word> *inv_primes, int tw_prime_offset,
                int tw_y_extra, int num_q_primes, int num_total_primes,
                int batch_stride, int batch, bool normalized) const;

  int GetLsbSize() const;
  int GetMsbSize() const;
  int GetLogWarpBatching() const;
  int GetStageMerging(NTTType type, Phase phase) const;
  int GetBlockDim(NTTType type, Phase phase) const;

 public:
  // TODO: allow for different log_degree
  static constexpr int min_log_degree_ = 12;
  static constexpr int max_log_degree_ = 16;

  explicit NTTHandler(const Parameter<word> &param);

  // disable copying (or moving also)
  NTTHandler(const NTTHandler &) = delete;
  NTTHandler &operator=(const NTTHandler &) = delete;

  // dst = NTT(src), montgomery_conversion is false by default
  void NTT(DvView<word> &dst, const NPInfo &np, const DvConstView<word> &src,
           bool montgomery_conversion = false) const;

  // dst = INTT(src), montgomery_conversion is true by default
  void INTT(DvView<word> &dst, const NPInfo &np, const DvConstView<word> &src,
            bool montgomery_conversion = true) const;

  // dst = INTT(src) * src_const
  void INTTAndMultConst(DvView<word> &dst, const NPInfo &np,
                        const DvConstView<word> &src,
                        const DvConstView<word> &src_const,
                        bool normalize = false) const;

  // special variants for ModUp and ModDown/Rescale/ModDownAndRescale

  void NTTForModUp(DvView<word> &dst, const NPInfo &np, int skip_start,
                   int skip_end, const DvConstView<word> &src, int batch = 1) const;
  void NTTForModDown(DvView<word> &dst, const NPInfo &np_src1,
                     const NPInfo &np_src2, const DvConstView<word> &src1,
                     const DvConstView<word> &src2,
                     const DvConstView<word> &inv_p_prod,
                     const DvConstView<word> &src2_padding =
                         DvConstView<word>(nullptr, 0)) const;
  void INTTForModDown(DvView<word> &dst, const NPInfo &np_src,
                      const NPInfo &np_non_intt, const DvConstView<word> &src,
                      const DvConstView<word> &src_const) const;

  DvConstView<word> ImaginaryUnitConstView(const NPInfo &np) const;

 private:
  void PopulateTwiddleFactors();
};

}  // namespace cheddar