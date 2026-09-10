#include "extension/SylphPcmm.h"

#include <algorithm>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
SylphPcmm<word>::SylphPcmm(ConstContextPtr<word> context,
                           sylph_pcmm::SylphPcmmPlan plan, int input_level,
                           double input_scale, bool cache_plaintexts)
    : context_{std::move(context)},
      plan_{std::move(plan)},
      input_level_{input_level},
      input_scale_{input_scale},
      cache_{cache_plaintexts} {
  AssertTrue(plan_.ok, "SylphPcmm: the plan is not usable: " + plan_.why);
  const int slots = context_->param_.degree_;
  AssertTrue(plan_.d * plan_.d <= slots,
             "SylphPcmm: a d x d matrix laid out row by row needs d^2 slots");
  // ONE level. That is the section's whole claim, and the assert is where it
  // is kept honest: nothing below may add a second.
  output_level_ = input_level_ - 1;
  AssertTrue(output_level_ >= 0,
             "SylphPcmm: Eq. (5) needs one level below the input");
}

template <typename word>
void SylphPcmm<word>::Compile() {
  if (!cache_) {
    compiled_ = true;
    return;
  }
  const int slots = context_->param_.degree_;
  const double out_scale = context_->param_.GetScale(output_level_);
  // The plaintext rides at whatever scale makes the product land canonically,
  // which is the same bookkeeping every other plaintext multiply in this tree
  // does: `pt_scale * ct_scale = out_scale * rescale_prime_prod`.
  const double pt_scale =
      out_scale * context_->param_.GetRescalePrimeProd(input_level_) /
      input_scale_;

  pt_.clear();
  pt_.resize(static_cast<size_t>(plan_.b) * plan_.g);
  std::vector<Complex> msg(slots, Complex(0.0, 0.0));
  for (int j = 0; j < plan_.g; j++) {
    for (int i = 0; i < plan_.b; i++) {
      const sylph_pcmm::Mat m = sylph_pcmm::PlaintextFor(plan_, i, j);
      std::fill(msg.begin(), msg.end(), Complex(0.0, 0.0));
      for (size_t s = 0; s < m.size(); s++) msg[s] = Complex(m[s], 0.0);
      context_->encoder_.Encode(pt_[static_cast<size_t>(j) * plan_.b + i],
                                input_level_, pt_scale, msg);
    }
  }
  compiled_ = true;
}

template <typename word>
void SylphPcmm<word>::Apply(Ct &res, const Ct &tau_b,
                            const EvkMap<word> &evk) const {
  AssertTrue(compiled_, "SylphPcmm: Compile() must run before Apply()");
  AssertTrue(context_->param_.NPToLevel(tau_b.GetNP()) == input_level_,
             "SylphPcmm: the input is not at the compiled level");
  AssertTrue(&res != &tau_b, "SylphPcmm: Apply must not be in place");
  const int d = plan_.d;

  // --- the baby steps, hoisted -------------------------------------------
  //
  // Every giant step reads the same `rot_R^i(tau^(l+1) B)`, so the b - 1
  // rotations are paid once. Without this the count would be `g (b - 1)` and
  // the square root in section 4.2's `O(sqrt d)` would be gone.
  std::vector<Ct> baby(plan_.b);
  context_->Copy(baby[0], tau_b);
  for (int i = 1; i < plan_.b; i++) {
    context_->HRot(baby[i], tau_b, evk.GetRotationKey(i * d), i * d);
  }

  // --- Eq. (5) ------------------------------------------------------------
  Ct inner, prod, rotated;
  bool first_giant = true;
  for (int j = 0; j < plan_.g; j++) {
    bool first_baby = true;
    for (int i = 0; i < plan_.b; i++) {
      const Pt *pt = nullptr;
      Pt built;
      if (cache_) {
        pt = &pt_[static_cast<size_t>(j) * plan_.b + i];
      } else {
        // Section 4.2's memory choice: rebuild `pt_{A,i,j,l}` from the one
        // stored `tau^l sigma(A)` with two plaintext rotations, and encode on
        // the device so the host is not the bottleneck (Doing.md 3.25 is what
        // happens when it is).
        const sylph_pcmm::Mat m = sylph_pcmm::PlaintextFor(plan_, i, j);
        std::vector<Complex> msg(context_->param_.degree_, Complex(0.0, 0.0));
        for (size_t s = 0; s < m.size(); s++) msg[s] = Complex(m[s], 0.0);
        const double out_scale = context_->param_.GetScale(output_level_);
        const double pt_scale =
            out_scale * context_->param_.GetRescalePrimeProd(input_level_) /
            input_scale_;
        context_->gpu_encoder_.Encode(built, input_level_, pt_scale, msg);
        pt = &built;
      }
      // MultUnsafe: no rescale here. The whole inner sum is accumulated at the
      // product scale and rescaled ONCE at the end, which is what keeps Eq. (5)
      // at one level rather than one per term.
      context_->MultUnsafe(prod, baby[i], *pt);
      if (first_baby) {
        context_->Copy(inner, prod);
        first_baby = false;
      } else {
        context_->Add(inner, inner, prod);
      }
    }
    if (j == 0) {
      context_->Copy(res, inner);
      first_giant = false;
    } else {
      context_->HRot(rotated, inner, evk.GetRotationKey(j * plan_.b * d),
                     j * plan_.b * d);
      context_->Add(res, res, rotated);
    }
  }
  AssertFalse(first_giant, "SylphPcmm: the giant-step loop did not run");
  // Explicit temp rather than in-place: `Context::Rescale` supports aliasing
  // but Warns for it, and this is a per-call path.
  Ct scaled;
  context_->Rescale(scaled, res);
  res = std::move(scaled);
  AssertTrue(context_->param_.NPToLevel(res.GetNP()) == output_level_,
             "SylphPcmm: Eq. (5) spent more than the one level it claims");
}

template <typename word>
size_t SylphPcmm<word>::GetPlaintextBytes() const {
  size_t bytes = 0;
  for (const Pt &pt : pt_) {
    bytes += static_cast<size_t>(pt.GetNP().GetNumTotal()) *
             context_->param_.degree_ * sizeof(word);
  }
  return bytes;
}

template class SylphPcmm<uint32_t>;
template class SylphPcmm<uint64_t>;

}  // namespace cheddar
