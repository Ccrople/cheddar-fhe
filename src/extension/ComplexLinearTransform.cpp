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
PlainHoistMap ComplexLinearTransform<word>::MergedHoistMap(
    const StripedMatrix &matrix, int bs, int gs, int pre_rotation,
    int additional_pt_rot, int alias_offset) {
  const StripedMatrix real = TakePart(matrix, false);
  const StripedMatrix imag = TakePart(matrix, true);
  // Both halves have the same diagonal offsets by construction, so they get the
  // same stride and the same baby/giant split -- which is what lets them share
  // giant steps at all.
  const int stride =
      LinearTransform<word>::DetermineStride(real, bs, gs, pre_rotation);
  PlainHoistMap merged = LinearTransform<word>::ConstructPlainHoistMap(
      real, stride, bs, pre_rotation, additional_pt_rot);
  const PlainHoistMap imag_map = LinearTransform<word>::ConstructPlainHoistMap(
      imag, stride, bs, pre_rotation, additional_pt_rot);

  for (const auto &[gs_idx, bs_map] : imag_map) {
    AssertTrue(merged.find(gs_idx) != merged.end(),
               "ComplexLinearTransform: the halves disagree on giant steps");
    for (const auto &[bs_idx, message] : bs_map) {
      AssertTrue(bs_idx < alias_offset,
                 "ComplexLinearTransform: the alias offset collides with a "
                 "real baby-step index");
      merged.at(gs_idx).try_emplace(bs_idx + alias_offset, message);
    }
  }
  return merged;
}

template <typename word>
ComplexLinearTransform<word>::ComplexLinearTransform(
    ConstContextPtr<word> context, const StripedMatrix &matrix, int pt_level,
    double pt_scale, int bs, int gs /*= 1*/, int pre_rotation /*= 0*/,
    int additional_pt_rot /*= 0*/)
    : re_{context,  TakePart(matrix, false), pt_level,     pt_scale,
          bs,       gs,                      pre_rotation, additional_pt_rot},
      im_{context,  TakePart(matrix, true),  pt_level,     pt_scale,
          bs,       gs,                      pre_rotation, additional_pt_rot},
      alias_offset_{matrix.GetWidth()},
      merged_{context,
              MergedHoistMap(matrix, bs, gs, pre_rotation, additional_pt_rot,
                             matrix.GetWidth()),
              pt_level, pt_scale, /*suppress_bs_swap=*/true} {
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

  if (min_ks) {
    // min_ks derives one stride from the baby-step sequence, and the aliased
    // keys are not a sequence. Four unfused giant steps instead, which is the
    // trade min_ks already makes.
    Ct rr, ir, ri, ii;
    re_.EvaluateGiantStep(context, rr, bs_re, evk_map, min_ks);
    im_.EvaluateGiantStep(context, ir, bs_re, evk_map, min_ks);
    re_.EvaluateGiantStep(context, ii, bs_im, evk_map, min_ks);
    im_.EvaluateGiantStep(context, ri, bs_im, evk_map, min_ks);
    context->Sub(res_re, rr, ri);
    context->Add(res_im, ir, ii);
    return;
  }

  // One feed map, filled twice. Pass one is {b: re, b+w: -im}, which merged_
  // reads as Re(M) re + Im(M) (-im); pass two moves re up into the aliased
  // block and drops im into the plain one, giving Re(M) im + Im(M) re. No
  // ciphertext is copied -- the second fill is two moves per baby step.
  std::map<int, Ct> feed;
  for (auto &[bs_idx, ct] : bs_re) {
    context->Neg(feed[bs_idx + alias_offset_], bs_im.at(bs_idx));
    feed[bs_idx] = std::move(ct);
  }
  // Both baby steps are already materialised, so in_re / in_im are dead and the
  // results can be written straight into them.
  merged_.EvaluateGiantStep(context, res_re, feed, evk_map, false);

  for (auto &[bs_idx, ct] : bs_im) {
    feed[bs_idx + alias_offset_] = std::move(feed[bs_idx]);
    feed[bs_idx] = std::move(ct);
  }
  merged_.EvaluateGiantStep(context, res_im, feed, evk_map, false);
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

  std::map<int, Ct> feed;
  for (auto &[bs_idx, ct] : bs_re) {
    context->Neg(feed[bs_idx + alias_offset_], bs_im.at(bs_idx));
    feed[bs_idx] = std::move(ct);
  }
  merged_.EvaluateGiantStep(context, res, feed, evk_map, false);
}

template class ComplexLinearTransform<uint32_t>;
template class ComplexLinearTransform<uint64_t>;

}  // namespace cheddar
