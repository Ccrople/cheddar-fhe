#include <algorithm>
#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/ChebyshevFit.h"
#include "extension/GeLu.h"
#include "extension/Profile.h"

namespace cheddar {

namespace {

//! The exact (erf) GELU the BERT checkpoints were trained with, not the tanh
//! approximation: the two differ by up to 3e-3, which is above what the fit
//! below is asked to deliver, so approximating an approximation would put a
//! floor under the whole feed-forward for nothing.
double GeLu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

}  // namespace

template <typename word>
GeLuHandler<word>::GeLuHandler(ConstContextPtr<word> context,
                               const std::vector<Group> &groups,
                               int input_level)
    : context_{std::move(context)}, groups_{groups}, input_level_{input_level} {
  AssertTrue(!groups_.empty(), "GeLu: at least one group");
  const double in_scale = context_->param_.GetScale(input_level_);
  polys_.resize(groups_.size());
  // The fitted groups' polynomials live on [-1, 1] and are fitted to
  // `GELU(range * v)`, so their output is already GELU(u) with nothing to
  // undo; what they need is `u / range`, which `Apply` takes as given (see
  // `SiLu.h` for why the division is not done here). The saturated groups
  // have no polynomial at all -- that is the whole point of the class.
  std::vector<int> levels;
  double fit_range = 0.0;
  bool have_fit = false;
  for (size_t i = 0; i < groups_.size(); i++) {
    const auto &g = groups_[i];
    if (g.kind != Kind::kFit) {
      // `u` itself costs no level: it arrives at the input level and is only
      // levelled down to meet the fitted group.
      levels.push_back(input_level_);
      continue;
    }
    AssertTrue(g.range > 0.0, "GeLu: range must be positive");
    AssertTrue(g.degree > 0, "GeLu: degree must be positive");
    AssertTrue(!have_fit || g.range == fit_range,
               "GeLu: every fitted group must state the same range -- there "
               "is one ciphertext and one division");
    fit_range = g.range;
    have_fit = true;
    const double r = g.range;
    auto coeffs =
        chebfit::Interpolate([r](double v) { return GeLu(r * v); }, g.degree);
    const int degree_used =
        EvalPoly<word>(coeffs, input_level_, in_scale, in_scale, true)
            .GetPolyDegree();
    levels.push_back(input_level_ - Log2Ceil(degree_used + 1));
  }
  AssertTrue(have_fit, "GeLu: at least one fitted group");
  // The groups are SUMMED, so they meet at the deepest of them; a group that
  // lands higher is levelled down before its mask.
  poly_out_level_ = *std::min_element(levels.begin(), levels.end());
  AssertTrue(poly_out_level_ >= 1,
             "GeLu: the polynomial and its mask do not fit below the input "
             "level");
  for (size_t i = 0; i < groups_.size(); i++) {
    if (groups_[i].kind != Kind::kFit) continue;
    const double r = groups_[i].range;
    auto coeffs = chebfit::Interpolate([r](double v) { return GeLu(r * v); },
                                       groups_[i].degree);
    // The target scale is the canonical scale of the level the tree LANDS on
    // -- EvalPoly stamps it on unchecked, and under grafting the two scales
    // differ enough to fail the next Add (see `SiLu.cu`).
    polys_[i] = std::make_unique<EvalPoly<word>>(
        coeffs, input_level_, in_scale, context_->param_.GetScale(levels[i]),
        /*chebyshev=*/true);
    polys_[i]->Compile(context_);
  }
  // One group needs no mask and no multiply; more than one costs the rescale
  // that the group sum's doubled scale asks for.
  out_level_ = groups_.size() == 1 ? poly_out_level_ : poly_out_level_ - 1;
}

template <typename word>
double GeLuHandler<word>::GetRange() const {
  for (const auto &g : groups_) {
    if (g.kind == Kind::kFit) return g.range;
  }
  return 1.0;
}

template <typename word>
double GeLuHandler<word>::PlainGeLu(int g, double u) const {
  switch (groups_[g].kind) {
    case Kind::kIdentity:
      return u;
    case Kind::kZero:
      return 0.0;
    default:
      return polys_[g]->PlainEvaluate(u / groups_[g].range);
  }
}

template <typename word>
void GeLuHandler<word>::Prepare(
    const std::vector<std::vector<Complex>> &mask) const {
  if (groups_.size() == 1) return;
  AssertTrue(mask.size() == groups_.size(), "GeLu: one mask per group");
  if (cached_mask_level_ == poly_out_level_ && cached_mask_ == mask) return;
  mask_pt_.clear();
  mask_pt_.resize(mask.size());
  const double scale = context_->param_.GetScale(poly_out_level_);
  std::vector<Complex> scaled;
  for (size_t i = 0; i < mask.size(); i++) {
    if (groups_[i].kind == Kind::kZero) continue;
    if (groups_[i].kind == Kind::kIdentity) {
      // The input arrives as `u / range`, so the identity group's answer is
      // `range * v` -- and the factor rides the mask's own plaintext, for
      // nothing.
      scaled.assign(mask[i].size(), Complex(0.0, 0.0));
      for (size_t s = 0; s < mask[i].size(); s++) {
        scaled[s] = mask[i][s] * GetRange();
      }
      context_->gpu_encoder_.Encode(mask_pt_[i], poly_out_level_, scale,
                                    scaled);
      continue;
    }
    context_->gpu_encoder_.Encode(mask_pt_[i], poly_out_level_, scale, mask[i]);
  }
  cached_mask_ = mask;
  cached_mask_level_ = poly_out_level_;
}

template <typename word>
size_t GeLuHandler<word>::GetPlaintextBytes() const {
  if (cached_mask_level_ < 0) return 0;
  const size_t words =
      static_cast<size_t>(
          context_->param_.LevelToNP(cached_mask_level_).GetNumTotal()) *
      context_->param_.degree_;
  size_t count = 0;
  for (const auto &g : groups_) {
    if (g.kind != Kind::kZero) count++;
  }
  return count * words * sizeof(word);
}

template <typename word>
void GeLuHandler<word>::Apply(Ct &res, const Ct &normalised_u,
                              const std::vector<std::vector<Complex>> &mask,
                              const EvkMap<word> &evk_map) const {
  NvtxScope _nv("gelu: Apply");
  const auto &mult_key = evk_map.GetMultiplicationKey();
  if (groups_.size() == 1) {
    polys_[0]->Evaluate(context_, res, normalised_u, mult_key);
    return;
  }
  Prepare(mask);
  Ct piece, levelled, term, acc;
  bool first = true;
  for (size_t i = 0; i < groups_.size(); i++) {
    // A `kZero` group contributes nothing, which is exactly why it is free:
    // GELU below the transition is zero to within 1.3e-04, and those slots
    // are most of the outliers (measured: 3835 of 3907 at layer 0).
    if (groups_[i].kind == Kind::kZero) continue;
    if (groups_[i].kind == Kind::kIdentity) {
      context_->LevelDown(levelled, normalised_u, poly_out_level_);
    } else {
      polys_[i]->Evaluate(context_, piece, normalised_u, mult_key);
      context_->LevelDown(levelled, piece, poly_out_level_);
    }
    context_->Mult(term, levelled, mask_pt_[i]);
    if (first) {
      context_->Copy(acc, term);
      first = false;
    } else {
      context_->Add(acc, acc, term);
    }
  }
  // `Mult(Ct, Pt)` does not rescale; this is the level the header charges the
  // masks, and it is the one Llama's gate multiply spends.
  context_->Rescale(res, acc);
}

template class GeLuHandler<uint32_t>;
template class GeLuHandler<uint64_t>;

}  // namespace cheddar
