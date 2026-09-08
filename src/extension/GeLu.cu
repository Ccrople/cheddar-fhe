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
  polys_.resize(groups_.size());
  // The fitted groups' polynomials live on [-1, 1] and are fitted to
  // `GELU(range * v)`, so their output is already GELU(u) with nothing to
  // undo; what they need is `u / range`, which `Apply` takes as given (see
  // `SiLu.h` for why the division is not done here). The saturated groups
  // have no polynomial at all -- that is the whole point of the class.
  // THE MASK GOES IN FRONT OF THE POLYNOMIAL, NOT BEHIND IT. A saturated
  // slot's input is `u / R` with |u| far past `R` -- that is what makes it
  // saturated -- and a Chebyshev polynomial evaluated there grows like
  // cosh(d arccosh(v)): at |v| = 16 and degree 31 that is 1e46, which
  // overflows the modulus and destroys EVERY slot of the ciphertext, the
  // masked-out ones included. Measured before the fix, with the masks on the
  // output: the bulk slots came back at 6.3e+16. So the fitted group's mask
  // multiplies the INPUT -- the saturated slots become zero, where the fit
  // returns GELU(0) = 0, which is also the right answer for the negative
  // saturated group -- and the masks cost the same one level either way.
  multi_ = groups_.size() > 1;
  const int fit_in_level = multi_ ? input_level_ - 1 : input_level_;
  const double fit_scale = context_->param_.GetScale(fit_in_level);
  std::vector<int> levels;
  double fit_range = 0.0;
  bool have_fit = false;
  for (size_t i = 0; i < groups_.size(); i++) {
    const auto &g = groups_[i];
    if (g.kind != Kind::kFit) {
      // `u` itself needs one level for its own mask multiply, and it takes it
      // wherever the fitted group lands.
      levels.push_back(input_level_);
      continue;
    }
    AssertTrue(g.range > 0.0, "GeLu: range must be positive");
    AssertTrue(g.degree > 0, "GeLu: degree must be positive");
    // The FIRST fitted group's range is the one the caller divided by; a
    // second one states its own and its input mask carries the ratio, which
    // is free (the mask is already a multiply). That is what lets a handful
    // of slots reaching 135 be answered at all while the other 393,000 keep
    // a range of 24 and its degree -- see `Calibration::gelu_group`.
    if (!have_fit) fit_range = g.range;
    have_fit = true;
    const double r = g.range;
    auto coeffs =
        chebfit::Interpolate([r](double v) { return GeLu(r * v); }, g.degree);
    const int degree_used =
        EvalPoly<word>(coeffs, fit_in_level, fit_scale, fit_scale, true)
            .GetPolyDegree();
    levels.push_back(fit_in_level - Log2Ceil(degree_used + 1));
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
        coeffs, fit_in_level, fit_scale, context_->param_.GetScale(levels[i]),
        /*chebyshev=*/true);
    polys_[i]->Compile(context_);
  }
  // One group needs no mask and no multiply at all. More than one costs
  // exactly one level -- the input mask's rescale, already inside
  // `fit_in_level`; the saturated groups' own multiplies rescale onto the
  // level the fit lands on and cost nothing further.
  out_level_ = poly_out_level_;
  AssertTrue(!multi_ || poly_out_level_ + 1 <= fit_in_level,
             "GeLu: the saturated groups have no level to be multiplied at");
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
  if (cached_mask_level_ == input_level_ && cached_mask_ == mask) return;
  mask_pt_.clear();
  mask_pt_.resize(mask.size());
  std::vector<Complex> scaled;
  for (size_t i = 0; i < mask.size(); i++) {
    if (groups_[i].kind == Kind::kZero) continue;
    if (groups_[i].kind == Kind::kIdentity) {
      // The input arrives as `u / range`, so the identity group's answer is
      // `range * v` -- and the factor rides the mask's own plaintext, for
      // nothing. It meets the fit one level above where the fit lands, so
      // that its own rescale puts it exactly there.
      const int lvl = poly_out_level_ + 1;
      scaled.assign(mask[i].size(), Complex(0.0, 0.0));
      for (size_t s = 0; s < mask[i].size(); s++) {
        scaled[s] = mask[i][s] * GetRange();
      }
      context_->gpu_encoder_.Encode(mask_pt_[i], lvl,
                                    context_->param_.GetScale(lvl), scaled);
      continue;
    }
    // The fitted group's mask multiplies the INPUT, at the input level, and
    // carries `caller's range / this group's range` so that every group sees
    // its own argument in [-1, 1] off ONE division.
    const double ratio = GetRange() / groups_[i].range;
    scaled.assign(mask[i].size(), Complex(0.0, 0.0));
    for (size_t sx = 0; sx < mask[i].size(); sx++) {
      scaled[sx] = mask[i][sx] * ratio;
    }
    context_->gpu_encoder_.Encode(mask_pt_[i], input_level_,
                                  context_->param_.GetScale(input_level_),
                                  scaled);
  }
  cached_mask_ = mask;
  cached_mask_level_ = input_level_;
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
  Ct product, clamped, levelled, term;
  bool first = true;
  for (size_t i = 0; i < groups_.size(); i++) {
    // A `kZero` group contributes nothing, which is exactly why it is free:
    // GELU below the transition is zero to within 1.3e-04, and those slots
    // are most of the outliers (measured: 3835 of 3907 at layer 0). The
    // fitted group's input mask has already set them to zero, and the fit
    // returns GELU(0) = 0 there, so nothing has to put them back.
    if (groups_[i].kind == Kind::kZero) continue;
    if (groups_[i].kind == Kind::kFit) {
      // Clamp first, evaluate second. See the constructor.
      context_->Mult(product, normalised_u, mask_pt_[i]);
      context_->Rescale(clamped, product);
      polys_[i]->Evaluate(context_, term, clamped, mult_key);
    } else {
      context_->LevelDown(levelled, normalised_u, poly_out_level_ + 1);
      context_->Mult(product, levelled, mask_pt_[i]);
      context_->Rescale(term, product);
    }
    if (first) {
      context_->Copy(res, term);
      first = false;
    } else {
      context_->Add(res, res, term);
    }
  }
}

template class GeLuHandler<uint32_t>;
template class GeLuHandler<uint64_t>;

}  // namespace cheddar
