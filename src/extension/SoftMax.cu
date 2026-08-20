#include <algorithm>
#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/ChebyshevFit.h"
#include "extension/SoftMax.h"

namespace cheddar {

template <typename word>
std::unique_ptr<EvalPoly<word>> SoftMaxHandler<word>::MakeInvSqrt(double lo,
                                                                  double hi,
                                                                  int degree,
                                                                  int level) {
  // The polynomial lives on [-1, 1], so it is fitted to 1/sqrt(a v + b) with
  // a, b the half-width and centre of the calibrated norm interval. The
  // affine map itself is applied in Apply as a constant multiply and a
  // constant add; reinterpreting the scale instead would be free but hands
  // EvalPoly a non-canonical input scale, which is what silently broke
  // RMSNorm.
  const double a = 0.5 * (hi - lo);
  const double b = 0.5 * (hi + lo);
  auto coeffs = chebfit::Interpolate(
      [a, b](double v) { return 1.0 / std::sqrt(a * v + b); }, degree);

  const double in_scale = context_->param_.GetScale(level);
  const int degree_used =
      EvalPoly<word>(coeffs, level, in_scale, in_scale, true).GetPolyDegree();
  const int out_level = level - Log2Ceil(degree_used + 1);
  AssertTrue(out_level >= 0,
             "SoftMax: the inverse square root does not fit below its level");
  auto poly = std::make_unique<EvalPoly<word>>(
      coeffs, level, in_scale, context_->param_.GetScale(out_level), true);
  poly->Compile(context_);
  return poly;
}

template <typename word>
SoftMaxHandler<word>::SoftMaxHandler(ConstContextPtr<word> context,
                                     int num_keys, double range,
                                     int input_level, int num_iters,
                                     const std::vector<double> &norm_lo,
                                     const std::vector<double> &norm_hi,
                                     int exp_degree, int inv_sqrt_degree,
                                     int early_inv_sqrt_degree, bool boot_aux,
                                     int aux_return_level, double aux_boot_max,
                                     int group_size)
    : context_{std::move(context)},
      num_keys_{num_keys},
      group_size_{group_size},
      range_{range},
      input_level_{input_level},
      num_iters_{num_iters},
      norm_lo_{norm_lo},
      norm_hi_{norm_hi},
      boot_aux_{boot_aux},
      aux_return_level_{aux_return_level},
      aux_in_level_{-1} {
  AssertTrue(num_keys_ > 1 && (num_keys_ & (num_keys_ - 1)) == 0,
             "SoftMax: num_keys must be a power of two");
  AssertTrue(group_size_ >= 1 && (group_size_ & (group_size_ - 1)) == 0,
             "SoftMax: group_size must be a power of two");
  AssertTrue(range_ > 0.0, "SoftMax: range must be positive");
  AssertTrue(num_iters_ >= 1, "SoftMax: at least one iteration");
  AssertTrue(static_cast<int>(norm_lo_.size()) == num_iters_ &&
                 static_cast<int>(norm_hi_.size()) == num_iters_,
             "SoftMax: one calibrated norm interval per iteration");
  num_slots_ = context_->param_.degree_ / 2;
  AssertTrue(num_slots_ % num_keys_ == 0,
             "SoftMax: the row length must divide the slot count");
  num_rows_ = num_slots_ / num_keys_;

  // THE KEY AXIS IS STRIDED, NOT CONTIGUOUS: slot = row + key * num_rows.
  //
  // Cheddar's rotation is cyclic over the whole slot vector, not within a
  // block. With the keys of a row in num_keys *consecutive* slots, rotating by
  // 1, 2, ... num_keys/2 gives each slot the sum of the num_keys slots that
  // follow it, which straddles two rows for every slot but the first of each.
  // The corrupted norm then lands outside the calibrated interval and the
  // inverse square root diverges -- and the result is squared afterwards.
  //
  // Striding the key axis makes the wrap-around exactly right, because the
  // pattern is periodic in the rotation distance: rotating by num_rows moves
  // to the next key of the same row. This is also the axis convention
  // RmsNormHandler already reduces along, so the two operators agree.
  for (int d = num_rows_; d < num_slots_; d *= 2) {
    rotation_distances_.push_back(d);
  }

  // The exponential on [-M/2^k, 0], reached from v in [-1, 1] by
  // x' = M (v - 1) / 2^(k+1).
  const double half = range_ / std::pow(2.0, num_iters_ + 1);
  auto exp_coeffs = chebfit::Interpolate(
      [half](double v) { return std::exp(half * (v - 1.0)); }, exp_degree);
  const double in_scale = context_->param_.GetScale(input_level_);
  const int exp_used =
      EvalPoly<word>(exp_coeffs, input_level_, in_scale, in_scale, true)
          .GetPolyDegree();
  const int exp_out = input_level_ - Log2Ceil(exp_used + 1);
  AssertTrue(exp_out >= 0, "SoftMax: the exponential does not fit");
  exp_poly_ = std::make_unique<EvalPoly<word>>(
      exp_coeffs, input_level_, in_scale, context_->param_.GetScale(exp_out),
      true);
  exp_poly_->Compile(context_);

  // The inverse square roots are compiled lazily against the level each
  // iteration actually reaches, which depends on the ones before it. Walk the
  // levels the same way Apply will.
  //
  // Per iteration the main track spends: one square for the norm, one constant
  // multiply for the affine map, the polynomial, one multiply by the
  // normalisation, one square. [SYLPH] figure 2 keeps the first three off the
  // main track by bootstrapping the auxiliary track separately; without that
  // they land here, which is why the depth below exceeds section 4.3's 8.
  exp_out_level_ = exp_out;
  int level = exp_out - 1;  // the causal mask multiply
  for (int j = 0; j < num_iters_; j++) {
    const int norm_level = level - 1;   // square for ||y||^2
    const int affine_out = norm_level - 1;  // the affine map's constant multiply
    // With a hook the polynomial is compiled where the hook returns, not where
    // the affine map leaves off -- that is the whole point of the hook, and
    // getting it wrong is not an approximation error but an EvalPoly assert.
    const int poly_in =
        (aux_return_level_ >= 0) ? aux_return_level_ : affine_out;
    if (j == 0) aux_in_level_ = affine_out;
    // ||y||^2 / a reaches 2 hi / (hi - lo). The hook cannot carry that, so the
    // affine multiply divides by an extra constant and the hook multiplies it
    // back -- both halves ride multiplies that are already there.
    const double a = 0.5 * (norm_hi_[j] - norm_lo_[j]);
    aux_shrink_.push_back((norm_hi_[j] / a) / aux_boot_max);
    const int degree =
        (j + 1 == num_iters_) ? inv_sqrt_degree : early_inv_sqrt_degree;
    inv_sqrt_.push_back(
        MakeInvSqrt(norm_lo_[j], norm_hi_[j], degree, poly_in));
    int used = 0;
    while ((1 << used) < degree + 1) used++;
    if (aux_return_level_ >= 0 || boot_aux_) {
      // The normalisation comes back at the bootstrap's landing level and is
      // brought down to the main track, so the auxiliary depth never lands
      // here: the main track pays only the multiply and the square. This is
      // what [SYLPH] figure 2's orange triangles buy.
      level -= 2;
    } else {
      // Fused into one track: the main track meets the normalisation at the
      // polynomial's output level, so all of the auxiliary depth lands here.
      level = poly_in - used - 2;
    }
    AssertTrue(level >= 0, "SoftMax: does not fit below the input level");
  }
}

template <typename word>
int SoftMaxHandler<word>::GetMainTrackDepth() const {
  // exp, the causal mask, then k times (multiply by the normalisation, square).
  return (input_level_ - exp_out_level_) + 1 + 2 * num_iters_;
}

template <typename word>
int SoftMaxHandler<word>::GetAuxTrackDepth() const {
  // the norm square, the affine map, and the inverse square root
  return 2 + Log2Ceil(inv_sqrt_.back()->GetPolyDegree() + 1);
}

template <typename word>
std::vector<double> SoftMaxHandler<word>::PlainSoftMax(
    const std::vector<double> &x) const {
  const int d = static_cast<int>(x.size());
  std::vector<double> y(d);
  for (int i = 0; i < d; i++) {
    // the same argument the encrypted path sees: v in [-1, 1]
    y[i] = exp_poly_->PlainEvaluate(2.0 * x[i] / range_ + 1.0);
  }
  for (int j = 0; j < num_iters_; j++) {
    double sq = 0.0;
    for (double v : y) sq += v * v;
    const double a = 0.5 * (norm_hi_[j] - norm_lo_[j]);
    const double b = 0.5 * (norm_hi_[j] + norm_lo_[j]);
    const double r = inv_sqrt_[j]->PlainEvaluate((sq - b) / a);
    for (double &v : y) {
      v *= r;
      v = v * v;
    }
  }
  return y;
}

template <typename word>
void SoftMaxHandler<word>::Prepare(
    const std::vector<Complex> &causal_mask) const {
  Prepare(std::vector<std::vector<Complex>>{causal_mask});
}

template <typename word>
void SoftMaxHandler<word>::Prepare(
    const std::vector<std::vector<Complex>> &causal_mask) const {
  AssertTrue(static_cast<int>(causal_mask.size()) == group_size_,
             "SoftMax: one causal mask per ciphertext of the group");
  // The mask meets y right after the exponential, so that is the level it must
  // be encoded at -- known at construction, which is what lets this run in
  // setup rather than on the first Apply.
  const int level = exp_out_level_;
  if (level == cached_mask_level_ && causal_mask == cached_mask_) return;
  mask_pt_.clear();
  mask_pt_.resize(group_size_);
  for (int g = 0; g < group_size_; g++) {
    AssertTrue(static_cast<int>(causal_mask[g].size()) == num_slots_,
               "SoftMax: the mask must cover every slot");
    context_->encoder_.Encode(mask_pt_[g], level,
                              context_->param_.GetScale(level), causal_mask[g]);
  }
  cached_mask_ = causal_mask;
  cached_mask_level_ = level;
}

template <typename word>
size_t SoftMaxHandler<word>::GetPlaintextBytes() const {
  if (cached_mask_level_ < 0) return 0;
  return static_cast<size_t>(
             context_->param_.LevelToNP(cached_mask_level_).GetNumTotal()) *
         context_->param_.degree_ * sizeof(word) * mask_pt_.size();
}

template <typename word>
void SoftMaxHandler<word>::Apply(Ct &res, const Ct &x_scaled,
                                 const std::vector<Complex> &causal_mask,
                                 const EvkMap<word> &evk_map,
                                 const BootContext<word> *boot_context,
                                 const AuxBoot *aux_boot) const {
  AssertTrue(group_size_ == 1,
             "SoftMax: this handler was built for rows spanning several "
             "ciphertexts, so Apply needs the whole group");
  std::vector<Ct> out, in(1);
  context_->Copy(in[0], x_scaled);
  Apply(out, in, std::vector<std::vector<Complex>>{causal_mask}, evk_map,
        boot_context, aux_boot);
  context_->Copy(res, out[0]);
}

template <typename word>
void SoftMaxHandler<word>::Apply(
    std::vector<Ct> &res, const std::vector<Ct> &x_scaled,
    const std::vector<std::vector<Complex>> &causal_mask,
    const EvkMap<word> &evk_map, const BootContext<word> *boot_context,
    const AuxBoot *aux_boot) const {
  AssertTrue(static_cast<int>(x_scaled.size()) == group_size_,
             "SoftMax: one input ciphertext per member of the group");
  AssertTrue(aux_return_level_ < 0 || aux_boot != nullptr,
             "SoftMax: an auxiliary return level was configured, so Apply "
             "needs the hook that returns there");
  AssertTrue(aux_return_level_ >= 0 || !boot_aux_ || boot_context != nullptr,
             "SoftMax: boot_aux was requested, so Apply needs a BootContext");
  const auto &mult_key = evk_map.GetMultiplicationKey();

  // 1. y = exp(x'). The argument arrives already on [-1, 1]; see the header
  //    for why the mapping is the caller's and not this function's. This and
  //    everything else on the main track is elementwise, so it is simply run
  //    once per ciphertext of the group.
  std::vector<Ct> y(group_size_);
  for (int g = 0; g < group_size_; g++) {
    exp_poly_->Evaluate(context_, y[g], x_scaled[g], mult_key);
  }

  // 2. Causal mask. The exponential is strictly positive, so masked positions
  //    have to be zeroed explicitly or they would join the norm and the
  //    output. Causality is public, hence a plaintext -- one per member of the
  //    group, because which keys a ciphertext carries is what distinguishes
  //    them.
  {
    AssertTrue(context_->param_.NPToLevel(y[0].GetNP()) == exp_out_level_,
               "SoftMax: the exponential did not land where Prepare assumed");
    Prepare(causal_mask);
    Ct masked;
    for (int g = 0; g < group_size_; g++) {
      context_->Mult(masked, y[g], mask_pt_[g]);
      context_->Rescale(y[g], masked);
    }
  }

  Ct sq, term, rotated, r, scaled;
  for (int j = 0; j < num_iters_; j++) {
    // 3. ||y||_2^2 over the row. Euclidean, not the sum -- that is what makes
    //    the squaring below land on a vector that already sums to one. The row
    //    spans the group, so the squares are added across it first and the
    //    strided rotate-and-add then finishes the reduction inside what is now
    //    a single ciphertext. Everything from here to step 6 runs ONCE for the
    //    whole group; that is what the group is for.
    context_->HMult(sq, y[0], y[0], mult_key);
    for (int g = 1; g < group_size_; g++) {
      context_->HMult(term, y[g], y[g], mult_key);
      context_->Add(sq, sq, term);
    }
    for (int d : rotation_distances_) {
      // HRotAdd is res = (a << dist) + b, so res must not alias its inputs.
      context_->HRotAdd(rotated, sq, sq, evk_map.GetRotationKey(d), d);
      context_->Copy(sq, rotated);
    }

    // 4. Affine map onto the polynomial's domain. Mult(Ct, Const) does not
    //    rescale in Cheddar, so the scale is restored explicitly; that is the
    //    level this costs.
    const double a = 0.5 * (norm_hi_[j] - norm_lo_[j]);
    const double b = 0.5 * (norm_hi_[j] + norm_lo_[j]);
    const double shrink = (aux_boot != nullptr) ? aux_shrink_[j] : 1.0;
    const int sq_level = context_->param_.NPToLevel(sq.GetNP());
    Constant<word> inv_a;
    context_->encoder_.EncodeConstant(inv_a, sq_level,
                                      context_->param_.GetScale(sq_level),
                                      1.0 / (a * shrink));
    context_->Mult(scaled, sq, inv_a);
    context_->Rescale(sq, scaled);

    // 4b. The auxiliary bootstrap, if the caller supplied one. It goes here
    //     and not after the polynomial, because this is the only point in the
    //     iteration that lands on the schedule's StC level -- the header says
    //     why that is forced rather than chosen. The shift below is applied
    //     after, at whatever level the hook returns.
    if (aux_boot != nullptr) {
      Ct fresh;
      (*aux_boot)(fresh, sq, shrink);
      AssertTrue(context_->param_.NPToLevel(fresh.GetNP()) == aux_return_level_,
                 "SoftMax: the auxiliary hook did not return at the level the "
                 "inverse square root was compiled for");
      context_->Copy(sq, fresh);
    }

    const int v_level = context_->param_.NPToLevel(sq.GetNP());
    Constant<word> shift;
    context_->encoder_.EncodeConstant(
        shift, v_level, context_->param_.GetScale(v_level), -b / a);
    context_->Add(sq, sq, shift);

    // 5. r = 1 / ||y||_2. Only the last of these has to be accurate: an error
    //    here multiplies the row by a constant, the squaring makes it a square
    //    constant, and the next normalisation divides it straight back out.
    inv_sqrt_[j]->Evaluate(context_, r, sq, mult_key);

    // 6. Bring the normalisation back up, if asked. r = 1/||y||_2 with
    //    ||y||_2^2 in the calibrated interval, so r is already inside the
    //    [-1, 1] that CKKS bootstrapping needs -- [SYLPH] section 3.1.3's 1/B
    //    pre-scaling is unnecessary here. The main track then never sees the
    //    auxiliary depth.
    if (boot_aux_ && aux_boot == nullptr) {
      const int y_level = context_->param_.NPToLevel(y[0].GetNP());
      Ct boosted;
      boot_context->Boot(boosted, r, evk_map);
      const int boosted_level = context_->param_.NPToLevel(boosted.GetNP());
      AssertTrue(boosted_level >= y_level,
                 "SoftMax: the bootstrap landed below the main track, so it "
                 "cannot carry the normalisation back");
      context_->LevelDown(r, boosted, y_level);
    }

    // 7. y <- (y * r)^2. Now sum(y) = 1 for the last iteration, with no
    //    further normalisation needed. The normalisation broadcasts across the
    //    group for free: after the reduction every slot of a row holds that
    //    row's whole norm, so every ciphertext multiplies by the same r.
    //
    // Which of the two is higher depends on where the auxiliary track came
    // back. With `boot_aux` the normalisation is brought down to the main
    // track above; with the hook it returns at the operator's own level and
    // the polynomial leaves it *above* the main track, because the main track
    // has been sitting still since the causal mask. So meet at whichever is
    // lower rather than assuming.
    const int r_level = context_->param_.NPToLevel(r.GetNP());
    const int main_level = context_->param_.NPToLevel(y[0].GetNP());
    const int meet = std::min(r_level, main_level);
    Ct levelled_r;
    context_->LevelDown(levelled_r, r, meet);
    context_->Copy(r, levelled_r);
    for (int g = 0; g < group_size_; g++) {
      Ct levelled, prod;
      context_->LevelDown(levelled, y[g], meet);
      context_->HMult(prod, levelled, r, mult_key);
      context_->HMult(y[g], prod, prod, mult_key);
    }
  }
  res.clear();
  res.resize(group_size_);
  for (int g = 0; g < group_size_; g++) context_->Copy(res[g], y[g]);
}

template class SoftMaxHandler<uint32_t>;
template class SoftMaxHandler<uint64_t>;

}  // namespace cheddar
