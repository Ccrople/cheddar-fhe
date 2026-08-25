#include "extension/ComplexLinearTransform.h"

#include "common/Assert.h"

namespace cheddar {

template <typename word>
StripedMatrix ComplexLinearTransform<word>::TakePart(
    const StripedMatrix &matrix, bool imaginary) {
  StripedMatrix res(matrix.GetHeight(), matrix.GetWidth());
  for (const auto &[idx, diag] : matrix) {
    // Every offset is kept, including one whose part is identically zero: the
    // two halves have to present the same baby/giant structure to share a
    // baby-step map, and pruning here would break that silently.
    res.try_emplace(idx, diag.size(), Complex(0));
    auto &out = res.at(idx);
    for (size_t j = 0; j < diag.size(); j++) {
      out[j] = Complex(imaginary ? diag[j].imag() : diag[j].real(), 0.0);
    }
  }
  return res;
}

template <typename word>
ComplexLinearTransform<word>::ComplexLinearTransform(
    ConstContextPtr<word> context, const StripedMatrix &matrix, int pt_level,
    double pt_scale, int bs, int gs /*= 1*/, int pre_rotation /*= 0*/,
    int additional_pt_rot /*= 0*/)
    : re_{context,  TakePart(matrix, false), pt_level,     pt_scale,
          bs,       gs,                      pre_rotation, additional_pt_rot},
      im_{context,  TakePart(matrix, true),  pt_level,     pt_scale,
          bs,       gs,                      pre_rotation, additional_pt_rot} {
  AssertTrue(re_.GetDiagonalOffsets() == im_.GetDiagonalOffsets(),
             "ComplexLinearTransform: the real and imaginary halves must be "
             "compiled from the same diagonal offsets to share a baby step");
}

template <typename word>
void ComplexLinearTransform<word>::AddRequiredRotations(
    EvkRequest &req, bool min_ks /*= false*/) const {
  // The halves request the same distances; asking both is free and keeps this
  // honest if that ever stops being true.
  re_.AddRequiredRotations(req, min_ks);
  im_.AddRequiredRotations(req, min_ks);
}

template <typename word>
void ComplexLinearTransform<word>::EvaluateFromReal(
    ConstContextPtr<word> context, Ct &res_re, Ct &res_im, const Ct &input,
    const EvkMap<word> &evk_map, bool min_ks /*= false*/) const {
  std::map<int, Ct> bs;
  re_.EvaluateBabyStep(context, bs, input, evk_map, min_ks);
  // The baby step materialises its own rotated copies, so `input` is dead from
  // here and either output may alias it.
  re_.EvaluateGiantStep(context, res_re, bs, evk_map, min_ks);
  im_.EvaluateGiantStep(context, res_im, bs, evk_map, min_ks);
}

template <typename word>
void ComplexLinearTransform<word>::EvaluatePair(ConstContextPtr<word> context,
                                                Ct &res_re, Ct &res_im,
                                                const Ct &in_re,
                                                const Ct &in_im,
                                                const EvkMap<word> &evk_map,
                                                bool min_ks /*= false*/) const {
  std::map<int, Ct> bs_re, bs_im;
  re_.EvaluateBabyStep(context, bs_re, in_re, evk_map, min_ks);
  re_.EvaluateBabyStep(context, bs_im, in_im, evk_map, min_ks);

  Ct rr, ir, ri, ii;
  re_.EvaluateGiantStep(context, rr, bs_re, evk_map, min_ks);
  im_.EvaluateGiantStep(context, ir, bs_re, evk_map, min_ks);
  re_.EvaluateGiantStep(context, ii, bs_im, evk_map, min_ks);
  im_.EvaluateGiantStep(context, ri, bs_im, evk_map, min_ks);

  // Written last, so res_re / res_im may be in_re / in_im.
  context->Sub(res_re, rr, ri);
  context->Add(res_im, ir, ii);
}

template <typename word>
void ComplexLinearTransform<word>::EvaluateToReal(
    ConstContextPtr<word> context, Ct &res, const Ct &in_re, const Ct &in_im,
    const EvkMap<word> &evk_map, bool min_ks /*= false*/) const {
  std::map<int, Ct> bs_re, bs_im;
  re_.EvaluateBabyStep(context, bs_re, in_re, evk_map, min_ks);
  re_.EvaluateBabyStep(context, bs_im, in_im, evk_map, min_ks);

  Ct rr, ri;
  re_.EvaluateGiantStep(context, rr, bs_re, evk_map, min_ks);
  im_.EvaluateGiantStep(context, ri, bs_im, evk_map, min_ks);

  context->Sub(res, rr, ri);
}

template class ComplexLinearTransform<uint32_t>;
template class ComplexLinearTransform<uint64_t>;

}  // namespace cheddar
