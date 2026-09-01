#include <cstdlib>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "core/BatchCcmm.h"
#include "extension/Profile.h"

namespace cheddar {

namespace kernel {

// The relinearization's operands for d columns in two contiguous buffers:
//   switched[z]          = D3[z]           (d23[z].ax, the sk^2 part)
//   addend[z]            = (D0[z], D1[z] + D2[z])
//                        = (d01[z].bx, d01[z].ax + d23[z].bx)
// src_ptrs[4z .. 4z+3] = d01.bx, d01.ax, d23.bx, d23.ax; blockIdx.y picks the
// output polynomial (0: switched, 1: addend b, 2: addend a).
template <typename word>
__global__ void RelinGather(int log_degree, word *switched, word *addend,
                            int poly_words, const word *const *src_ptrs,
                            const word *primes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= poly_words) return;
  const int z = blockIdx.z;
  const word *const *s = src_ptrs + 4 * z;
  if (blockIdx.y == 0) {
    switched[z * poly_words + i] = basic::StreamingLoad(s[3] + i);
  } else if (blockIdx.y == 1) {
    addend[z * 2 * poly_words + i] = basic::StreamingLoad(s[0] + i);
  } else {
    const word prime = basic::StreamingLoadConst(primes + (i >> log_degree));
    addend[z * 2 * poly_words + poly_words + i] = basic::Add(
        basic::StreamingLoad(s[1] + i), basic::StreamingLoad(s[2] + i), prime);
  }
}

// dst_ptrs[2z + blockIdx.y][i] = src[z][poly blockIdx.y][i].
template <typename word>
__global__ void RelinScatter(word *const *dst_ptrs, int poly_words,
                             const word *src) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= poly_words) return;
  const int z = blockIdx.z;
  dst_ptrs[2 * z + blockIdx.y][i] =
      basic::StreamingLoad(src + z * 2 * poly_words + blockIdx.y * poly_words + i);
}

}  // namespace kernel

template <typename word>
BatchCcmmHandler<word>::BatchCcmmHandler(const Parameter<word> &param,
                                         const NTTHandler<word> &ntt_handler)
    : param_{param}, cmt_{param, ntt_handler}, matrix_{param, ntt_handler} {
  const char *env = std::getenv("CHEDDAR_CCMM_RELIN_SERIAL");
  relin_serial_ = (env != nullptr && env[0] == '1');
}

template <typename word>
void BatchCcmmHandler<word>::SetRelinSerial(bool serial) {
  relin_serial_ = serial;
}

template <typename word>
std::vector<int> BatchCcmmHandler<word>::RotationIndices(
    int sub_degree) const {
  return cmt_.ScrambleAutoRotationIndices(sub_degree);
}

template <typename word>
void BatchCcmmHandler<word>::Multiply(ConstContextPtr<word> context,
                                      std::vector<Ct> &res,
                                      const std::vector<Ct> &lhs,
                                      const std::vector<Ct> &rhs,
                                      int sub_degree,
                                      const EvkMap<word> &evk_map) const {
  const int degree = param_.degree_;
  const int d = degree / sub_degree;
  AssertTrue(sub_degree >= 2 && sub_degree <= degree && IsPowOfTwo(sub_degree),
             "BatchCcmm: invalid sub_degree");
  AssertTrue(static_cast<int>(lhs.size()) == d &&
                 static_cast<int>(rhs.size()) == d,
             "BatchCcmm: each side needs degree / sub_degree ciphertexts");

  const NPInfo np = lhs.at(0).GetNP();
  AssertTrue(np == rhs.at(0).GetNP(),
             "BatchCcmm: the two matrix encryptions differ in NP");
  AssertTrue(np.num_aux_ == 0, "BatchCcmm: aux primes are not supported");

  const int level = param_.NPToLevel(np);
  AssertTrue(level >= 1,
             "BatchCcmm: the product rescales, so the inputs must be above "
             "level 0");

  const int num_slots = lhs.at(0).GetNumSlots();
  const double product_scale = lhs.at(0).GetScale() * rhs.at(0).GetScale();
  const int num_total_primes = np.GetNumTotal();
  const size_t component_bytes =
      static_cast<size_t>(num_total_primes) * degree * sizeof(word);

  // 1. The second operand becomes row-wise. This is the step that makes the
  //    contraction of step 2 line up; everything else follows from it.
  NvtxScope _nv("ccmm: Multiply (lifted ring)");
  std::vector<Ct> rhs_transposed;
  {
    NvtxScope _c("ccmm: Cmt (rhs)");
    cmt_.Cmt(context, rhs_transposed, rhs, sub_degree, evk_map);
  }

  SubringCoeffMatrix<word> b_bar, a_bar;
  matrix_.ToMatrices(b_bar, a_bar, rhs_transposed, sub_degree, true);

  SubringCoeffMatrix<word> b_mat, a_mat;
  matrix_.ToMatrices(b_mat, a_mat, lhs, sub_degree, false);

  // 2. [B; A] * [B_bar | A_bar], the only multiplicative level in the whole
  //    algorithm.
  SubringCoeffMatrix<word> c00, c01, c10, c11;
  NvtxScope *_m = new NvtxScope("ccmm: MultiplyMatrices x4");
  matrix_.MultiplyMatrices(c00, b_mat, b_bar);
  matrix_.MultiplyMatrices(c01, b_mat, a_bar);
  matrix_.MultiplyMatrices(c10, a_mat, b_bar);
  matrix_.MultiplyMatrices(c11, a_mat, a_bar);
  delete _m;

  // 3, 4. Back to column-wise. Reading the rows as ciphertexts presents the
  //       row-wise pair to CMT as a column-wise encryption of the transpose.
  std::vector<Ct> c0_cts, c1_cts;
  matrix_.ToCiphertexts(c0_cts, c00, c01, product_scale, num_slots, true);
  matrix_.ToCiphertexts(c1_cts, c10, c11, product_scale, num_slots, true);

  std::vector<Ct> d01, d23;
  NvtxScope *_c2 = new NvtxScope("ccmm: Cmt (products)");
  cmt_.Cmt(context, d01, c0_cts, sub_degree, evk_map);
  cmt_.Cmt(context, d23, c1_cts, sub_degree, evk_map);
  delete _c2;
  NvtxScope _r("ccmm: Relinearize + Rescale per column");

  res.resize(d);
  const bool batched =
      !relin_serial_ && !param_.conjugate_invariant_ && d >= 2;
  if (batched) {
    // Steps 5-7 for every column in one group: the key switch of the d
    // sk^2 parts with the ONE multiplication key, (D0, D1 + D2) folded in as
    // the switch's addend (Context::MultKeyBatch), the rescale over the 2d
    // polynomials (ModSwitchHandler::RescaleBatch). Every word is the serial
    // loop's: the same kernels, and modular sums in either order.
    const int poly_words = num_total_primes * degree;
    const NPInfo next_np = param_.LevelToNP(level - 1);
    const int next_words = next_np.GetNumTotal() * degree;
    constexpr int block = 256;
    DeviceVector<word> switched(static_cast<size_t>(d) * poly_words);
    DeviceVector<word> addend(static_cast<size_t>(d) * 2 * poly_words);
    {
      HostVector<uint64_t> ptrs(4 * d);
      for (int j = 0; j < d; j++) {
        ptrs[4 * j] = reinterpret_cast<uint64_t>(d01[j].bx_.data());
        ptrs[4 * j + 1] = reinterpret_cast<uint64_t>(d01[j].ax_.data());
        ptrs[4 * j + 2] = reinterpret_cast<uint64_t>(d23[j].bx_.data());
        ptrs[4 * j + 3] = reinterpret_cast<uint64_t>(d23[j].ax_.data());
      }
      DeviceVector<uint64_t> ptrs_dev;
      CopyHostToDevice(ptrs_dev, ptrs);
      kernel::RelinGather<word><<<dim3(DivCeil(poly_words, block), 3, d),
                                  block>>>(
          param_.log_degree_, switched.data(), addend.data(), poly_words,
          reinterpret_cast<const word *const *>(ptrs_dev.data()),
          param_.GetPrimesPtr(np));
    }
    DeviceVector<word> combined_all(static_cast<size_t>(d) * 2 * poly_words);
    {
      std::vector<const EvaluationKey<word> *> keys(
          d, &evk_map.GetMultiplicationKey());
      context->MultKeyBatch(combined_all.data(), 2 * poly_words,
                            switched.data(), poly_words, addend.data(),
                            addend.data() + poly_words, 2 * poly_words, np,
                            keys, d);
    }
    DeviceVector<word> rescaled(static_cast<size_t>(d) * 2 * next_words);
    context->mod_switch_handlers_.at(level).RescaleBatch(
        rescaled.data(), next_words, combined_all.data(), poly_words, 2 * d);
    {
      HostVector<uint64_t> ptrs(2 * d);
      for (int j = 0; j < d; j++) {
        Ct &r = res[j];
        r.RemoveRx();
        r.ModifyNP(next_np);
        r.SetScale(product_scale / param_.GetRescalePrimeProd(level));
        r.SetNumSlots(num_slots);
        ptrs[2 * j] = reinterpret_cast<uint64_t>(r.bx_.data());
        ptrs[2 * j + 1] = reinterpret_cast<uint64_t>(r.ax_.data());
      }
      DeviceVector<uint64_t> ptrs_dev;
      CopyHostToDevice(ptrs_dev, ptrs);
      kernel::RelinScatter<word><<<dim3(DivCeil(next_words, block), 2, d),
                                   block>>>(
          reinterpret_cast<word *const *>(ptrs_dev.data()), next_words,
          rescaled.data());
    }
    return;
  }

  Ct relin_input, relinearized, combined;
  for (int j = 0; j < d; j++) {
    // 5. Relinearize (0, D3). D3 is the a-part of the (D2, D3) pair, and it
    //    sits under sk^2, which is what Cheddar's rx_ slot is for.
    relin_input.ModifyNP(np);
    relin_input.PrepareRx();
    relin_input.SetScale(product_scale);
    relin_input.SetNumSlots(num_slots);
    cudaMemsetAsync(relin_input.bx_.data(), 0, component_bytes,
                    cudaStreamLegacy);
    cudaMemsetAsync(relin_input.ax_.data(), 0, component_bytes,
                    cudaStreamLegacy);
    CopyDeviceToDevice(relin_input.rx_, d23[j].ax_);

    context->Relinearize(relinearized, relin_input,
                         evk_map.GetMultiplicationKey());

    // 6. (E0, E1) + (D0, D1 + D2). The D2 term is the *b*-part of the second
    //    pair and has to land on the *a*-part of the result, which is below
    //    the granularity of Add on whole ciphertexts.
    context->Add(combined, relinearized, d01[j]);
    NPInfo combined_np = combined.GetNP();
    std::vector<DvView<word>> ax_only{combined.AxView()};
    context->elem_handler_.Add(ax_only, combined_np,
                               {combined.AxConstView()},
                               {d23[j].BxConstView()});

    // 7. Rescale, which returns the level step 2 spent.
    context->Rescale(res[j], combined);
  }
}

template class BatchCcmmHandler<uint32_t>;
template class BatchCcmmHandler<uint64_t>;

}  // namespace cheddar
