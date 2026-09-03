#include "extension/ComplexLinearTransform.h"

#include <utility>

#include "common/Assert.h"

namespace cheddar {

template <typename word>
StripedMatrix ComplexLinearTransform<word>::TakePart(
    const StripedMatrix &matrix, bool imaginary) {
  StripedMatrix res(matrix.GetHeight(), matrix.GetWidth());
  for (const auto &[idx, diag] : matrix) {
    // Every offset is kept, including one whose part is identically zero: the
    // two halves have to present the same baby/giant structure to share a
    // baby step and a giant-step pass, and pruning here would break that
    // silently.
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
  if (min_ks) {
    re_.EvaluateGiantStep(context, res_re, bs, evk_map, min_ks);
    im_.EvaluateGiantStep(context, res_im, bs, evk_map, min_ks);
    return;
  }
  LinearTransform<word>::EvaluateGiantStepComplex(
      context, res_re, &res_im, re_, im_, bs, /*bs_im=*/nullptr, evk_map);
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

  if (min_ks) {
    // min_ks derives one stride from the baby-step sequence and rotates
    // sequentially, which the fused pass below has no analogue of. Four
    // unfused giant steps instead, which is the trade min_ks already makes --
    // key count against time.
    Ct rr, ir, ri, ii;
    re_.EvaluateGiantStep(context, rr, bs_re, evk_map, min_ks);
    im_.EvaluateGiantStep(context, ir, bs_re, evk_map, min_ks);
    re_.EvaluateGiantStep(context, ii, bs_im, evk_map, min_ks);
    im_.EvaluateGiantStep(context, ri, bs_im, evk_map, min_ks);
    context->Sub(res_re, rr, ri);
    context->Add(res_im, ir, ii);
    return;
  }

  // Both baby steps are materialised, so the inputs are dead and the outputs
  // may alias them. The fused giant step streams every plaintext once, feeds
  // all four real products from it, and carries the minus sign as a modular
  // subtraction -- no negated copy of anything.
  LinearTransform<word>::EvaluateGiantStepComplex(context, res_re, &res_im,
                                                  re_, im_, bs_re, &bs_im,
                                                  evk_map);
}

template <typename word>
void ComplexLinearTransform<word>::EvaluateToReal(
    ConstContextPtr<word> context, Ct &res, const Ct &in_re, const Ct &in_im,
    const EvkMap<word> &evk_map, bool min_ks /*= false*/) const {
  std::map<int, Ct> bs_re, bs_im;
  re_.EvaluateBabyStep(context, bs_re, in_re, evk_map, min_ks);
  re_.EvaluateBabyStep(context, bs_im, in_im, evk_map, min_ks);

  if (min_ks) {
    Ct rr, ri;
    re_.EvaluateGiantStep(context, rr, bs_re, evk_map, min_ks);
    im_.EvaluateGiantStep(context, ri, bs_im, evk_map, min_ks);
    context->Sub(res, rr, ri);
    return;
  }

  LinearTransform<word>::EvaluateGiantStepComplex(context, res,
                                                  /*res_im=*/nullptr, re_, im_,
                                                  bs_re, &bs_im, evk_map);
}

template <typename word>
void ComplexLinearTransform<word>::EvaluateFromRealBatch(
    ConstContextPtr<word> context, std::vector<Ct> &res_re,
    std::vector<Ct> &res_im, const std::vector<const Ct *> &inputs,
    const EvkMap<word> &evk_map) const {
  const int n = static_cast<int>(inputs.size());
  res_re.resize(n);
  res_im.resize(n);
  std::vector<std::map<int, Ct>> bs(n);
  std::vector<const std::map<int, Ct> *> bs_ptrs(n);
  std::vector<Ct *> re_ptrs(n), im_ptrs(n);
  std::vector<std::map<int, Ct> *> bs_out(n);
  for (int i = 0; i < n; i++) {
    bs_out[i] = &bs[i];
    bs_ptrs[i] = &bs[i];
    re_ptrs[i] = &res_re[i];
    im_ptrs[i] = &res_im[i];
  }
  re_.EvaluateBabyStepBatch(context, bs_out, inputs, evk_map);
  LinearTransform<word>::EvaluateGiantStepComplexBatch(
      context, re_ptrs, &im_ptrs, re_, im_, bs_ptrs, /*bs_im=*/nullptr,
      evk_map);
}

template <typename word>
void ComplexLinearTransform<word>::EvaluatePairBatch(
    ConstContextPtr<word> context, std::vector<Ct> &re, std::vector<Ct> &im,
    const EvkMap<word> &evk_map) const {
  const int n = static_cast<int>(re.size());
  AssertTrue(static_cast<int>(im.size()) == n,
             "ComplexLinearTransform: the pair group's halves disagree");
  std::vector<std::map<int, Ct>> bs_re(n), bs_im(n);
  std::vector<const std::map<int, Ct> *> bs_re_ptrs(n), bs_im_ptrs(n);
  std::vector<Ct *> re_ptrs(n), im_ptrs(n);
  // One baby-step group of both halves: 2n ciphertexts at one level share
  // the transform's keys, so the group is one ModUpBatch + one fused kernel.
  std::vector<std::map<int, Ct> *> bs_out(2 * n);
  std::vector<const Ct *> bs_in(2 * n);
  for (int i = 0; i < n; i++) {
    bs_out[i] = &bs_re[i];
    bs_out[n + i] = &bs_im[i];
    bs_in[i] = &re[i];
    bs_in[n + i] = &im[i];
    bs_re_ptrs[i] = &bs_re[i];
    bs_im_ptrs[i] = &bs_im[i];
    // The baby steps are materialised, so the inputs are dead and the
    // outputs alias them, as the serial EvaluatePair allows.
    re_ptrs[i] = &re[i];
    im_ptrs[i] = &im[i];
  }
  re_.EvaluateBabyStepBatch(context, bs_out, bs_in, evk_map);
  LinearTransform<word>::EvaluateGiantStepComplexBatch(
      context, re_ptrs, &im_ptrs, re_, im_, bs_re_ptrs, &bs_im_ptrs, evk_map);
}

template <typename word>
void ComplexLinearTransform<word>::EvaluateToRealBatch(
    ConstContextPtr<word> context, std::vector<Ct> &res,
    const std::vector<const Ct *> &in_re, const std::vector<const Ct *> &in_im,
    const EvkMap<word> &evk_map) const {
  const int n = static_cast<int>(in_re.size());
  AssertTrue(static_cast<int>(in_im.size()) == n,
             "ComplexLinearTransform: the to-real group's halves disagree");
  res.resize(n);
  std::vector<std::map<int, Ct>> bs_re(n), bs_im(n);
  std::vector<const std::map<int, Ct> *> bs_re_ptrs(n), bs_im_ptrs(n);
  std::vector<Ct *> res_ptrs(n);
  std::vector<std::map<int, Ct> *> bs_out(2 * n);
  std::vector<const Ct *> bs_in(2 * n);
  for (int i = 0; i < n; i++) {
    bs_out[i] = &bs_re[i];
    bs_out[n + i] = &bs_im[i];
    bs_in[i] = in_re[i];
    bs_in[n + i] = in_im[i];
    bs_re_ptrs[i] = &bs_re[i];
    bs_im_ptrs[i] = &bs_im[i];
    res_ptrs[i] = &res[i];
  }
  re_.EvaluateBabyStepBatch(context, bs_out, bs_in, evk_map);
  LinearTransform<word>::EvaluateGiantStepComplexBatch(
      context, res_ptrs, /*res_im=*/nullptr, re_, im_, bs_re_ptrs,
      &bs_im_ptrs, evk_map);
}

template <typename word>
ComplexLinearTransform<word>::ComplexLinearTransform(LinearTransform<word> &&re,
                                                     LinearTransform<word> &&im)
    : re_(std::move(re)), im_(std::move(im)) {}

template <typename word>
void ComplexLinearTransform<word>::Save(ArchiveWriter &ar) const {
  ar.Tag("cxlintrans");
  re_.Save(ar);
  im_.Save(ar);
}

template <typename word>
ComplexLinearTransform<word> ComplexLinearTransform<word>::Load(
    ArchiveReader &ar) {
  ar.Tag("cxlintrans");
  auto re = LinearTransform<word>::Load(ar);
  auto im = LinearTransform<word>::Load(ar);
  return ComplexLinearTransform<word>(std::move(re), std::move(im));
}

template class ComplexLinearTransform<uint32_t>;
template class ComplexLinearTransform<uint64_t>;

}  // namespace cheddar
