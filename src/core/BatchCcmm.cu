#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/BatchCcmm.h"

namespace cheddar {

template <typename word>
BatchCcmmHandler<word>::BatchCcmmHandler(const Parameter<word> &param,
                                         const NTTHandler<word> &ntt_handler)
    : param_{param}, cmt_{param, ntt_handler}, matrix_{param, ntt_handler} {}

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
  std::vector<Ct> rhs_transposed;
  cmt_.Cmt(context, rhs_transposed, rhs, sub_degree, evk_map);

  SubringCoeffMatrix<word> b_bar, a_bar;
  matrix_.ToMatrices(b_bar, a_bar, rhs_transposed, sub_degree, true);

  SubringCoeffMatrix<word> b_mat, a_mat;
  matrix_.ToMatrices(b_mat, a_mat, lhs, sub_degree, false);

  // 2. [B; A] * [B_bar | A_bar], the only multiplicative level in the whole
  //    algorithm.
  SubringCoeffMatrix<word> c00, c01, c10, c11;
  matrix_.MultiplyMatrices(c00, b_mat, b_bar);
  matrix_.MultiplyMatrices(c01, b_mat, a_bar);
  matrix_.MultiplyMatrices(c10, a_mat, b_bar);
  matrix_.MultiplyMatrices(c11, a_mat, a_bar);

  // 3, 4. Back to column-wise. Reading the rows as ciphertexts presents the
  //       row-wise pair to CMT as a column-wise encryption of the transpose.
  std::vector<Ct> c0_cts, c1_cts;
  matrix_.ToCiphertexts(c0_cts, c00, c01, product_scale, num_slots, true);
  matrix_.ToCiphertexts(c1_cts, c10, c11, product_scale, num_slots, true);

  std::vector<Ct> d01, d23;
  cmt_.Cmt(context, d01, c0_cts, sub_degree, evk_map);
  cmt_.Cmt(context, d23, c1_cts, sub_degree, evk_map);

  res.resize(d);
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
