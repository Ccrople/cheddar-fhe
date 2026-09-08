#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/ChebyshevFit.h"
#include "extension/LayerNorm.h"
#include "extension/Profile.h"

namespace cheddar {

template <typename word>
LayerNormHandler<word>::LayerNormHandler(ConstContextPtr<word> context,
                                         int num_tokens, int num_channels,
                                         double layer_constant, int input_level,
                                         double eps, double window_ratio,
                                         int degree, int channel_stride,
                                         int live_channels)
    : context_{std::move(context)},
      num_tokens_{num_tokens},
      num_channels_{num_channels},
      layer_constant_{layer_constant},
      eps_{eps},
      input_level_{input_level} {
  AssertTrue(num_tokens_ > 0 && IsPowOfTwo(num_tokens_),
             "LayerNorm: num_tokens must be a power of two");
  AssertTrue(num_channels_ > 0, "LayerNorm: num_channels must be positive");
  AssertTrue(layer_constant_ > 0.0,
             "LayerNorm: layer constant must be positive");
  AssertTrue(window_ratio > 1.0, "LayerNorm: window ratio must exceed one");
  AssertTrue(live_channels >= 0 && live_channels <= num_channels_,
             "LayerNorm: live_channels must not exceed the declared width");
  window_lo_ = 1.0 / std::sqrt(window_ratio);
  window_hi_ = std::sqrt(window_ratio);

  num_slots_ = context_->param_.MaxNumSlots();
  AssertTrue(num_slots_ % num_tokens_ == 0,
             "LayerNorm: tokens must divide the slot count");
  const long long total = 1LL * num_tokens_ * num_channels_;
  AssertTrue(total % num_slots_ == 0,
             "LayerNorm: T * H must be a multiple of the slot count");
  num_ct_ = static_cast<int>(total / num_slots_);

  // The same rotate-and-add tree as `RmsNormHandler`'s, and for the same
  // reason: it reduces and broadcasts at once, so both the mean and the
  // variance land in every slot with nothing to redistribute. `channel_stride`
  // keeps the banded convention's two slot parities apart (see `RmsNorm.h`).
  AssertTrue(channel_stride >= 1 && IsPowOfTwo(channel_stride),
             "LayerNorm: channel_stride must be a power of two");
  AssertTrue(num_tokens_ * channel_stride < num_slots_,
             "LayerNorm: channel_stride leaves the reduction nothing to sum");
  for (int d = num_tokens_ * channel_stride; d < num_slots_; d *= 2) {
    rotation_distances_.push_back(d);
  }

  // The centring: one plaintext multiply and a rescale, per the header.
  centre_level_ = input_level_ - 1;
  AssertTrue(centre_level_ >= 0,
             "LayerNorm: the centring does not fit below the input level");

  const double a = 0.5 * (window_hi_ - window_lo_);
  const double b = 0.5 * (window_hi_ + window_lo_);
  auto coeffs = chebfit::Interpolate(
      [a, b](double v) { return 1.0 / std::sqrt(a * v + b); }, degree);

  // One level for the square, one for the affine map's multiplicative half.
  // Both scales stay canonical: the affine map by reinterpretation is free
  // and arithmetically sound but hands EvalPoly a non-canonical input, which
  // is what silently broke RMSNorm (see `RmsNorm.cu`).
  affine_scale_ = a;
  const int poly_level = centre_level_ - 2;
  const double poly_scale = context_->param_.GetScale(poly_level);
  const int degree_used =
      EvalPoly<word>(coeffs, poly_level, poly_scale, poly_scale, true)
          .GetPolyDegree();
  const int out_level = poly_level - Log2Ceil(degree_used + 1);
  AssertTrue(out_level >= 2,
             "LayerNorm: the inverse square root, the weight and the bias do "
             "not fit below the input level");
  inv_sqrt_ = std::make_unique<EvalPoly<word>>(
      coeffs, poly_level, poly_scale, context_->param_.GetScale(out_level),
      /*chebyshev=*/true);
  inv_sqrt_->Compile(context_);
  // `Apply` multiplies the centred image by `r` first (landing one under
  // `r`'s level), then by the gain, then rescales -- so the gain sits one
  // under `r` and the bias one under the gain.
  weight_level_ = out_level - 1;
  bias_level_ = weight_level_ - 1;
  // Where the mask meets the channel sum: the sum costs no level, so it is
  // still at the input's.
  (void)live_channels;
  live_ = live_channels > 0 ? live_channels : num_channels_;
}

template <typename word>
double LayerNormHandler<word>::PlainInvSqrt(double u) const {
  const double b = 0.5 * (window_hi_ + window_lo_);
  return inv_sqrt_->PlainEvaluate((u - b) / affine_scale_);
}

template <typename word>
void LayerNormHandler<word>::PrepareMask(
    const std::vector<std::vector<Complex>> &mask) const {
  AssertTrue(static_cast<int>(mask.size()) == num_ct_,
             "LayerNorm: one mask vector per input ciphertext");
  if (cached_mask_level_ == input_level_ && cached_mask_ == mask) return;
  mask_pt_.clear();
  mask_pt_.resize(num_ct_);
  const double scale = context_->param_.GetScale(input_level_);
  const double inv = 1.0 / static_cast<double>(live_);
  std::vector<Complex> scaled;
  for (int i = 0; i < num_ct_; i++) {
    scaled.assign(mask[i].size(), Complex(0.0, 0.0));
    for (size_t s = 0; s < mask[i].size(); s++) scaled[s] = mask[i][s] * inv;
    context_->gpu_encoder_.Encode(mask_pt_[i], input_level_, scale, scaled);
  }
  cached_mask_ = mask;
  cached_mask_level_ = input_level_;
}

template <typename word>
void LayerNormHandler<word>::Prepare(
    const std::vector<std::vector<Complex>> &weight,
    const std::vector<std::vector<Complex>> &bias) const {
  AssertTrue(static_cast<int>(weight.size()) == num_ct_,
             "LayerNorm: one weight vector per input ciphertext");
  if (cached_weight_level_ != weight_level_ || cached_weight_ != weight) {
    weight_pt_.clear();
    weight_pt_.resize(num_ct_);
    const double scale = context_->param_.GetScale(weight_level_);
    for (int i = 0; i < num_ct_; i++) {
      context_->gpu_encoder_.Encode(weight_pt_[i], weight_level_, scale,
                                    weight[i]);
    }
    cached_weight_ = weight;
    cached_weight_level_ = weight_level_;
  }
  if (bias.empty()) {
    bias_pt_.clear();
    cached_bias_.clear();
    cached_bias_level_ = -1;
    return;
  }
  AssertTrue(static_cast<int>(bias.size()) == num_ct_,
             "LayerNorm: one bias vector per input ciphertext");
  if (cached_bias_level_ == bias_level_ && cached_bias_ == bias) return;
  bias_pt_.clear();
  bias_pt_.resize(num_ct_);
  const double scale = context_->param_.GetScale(bias_level_);
  for (int i = 0; i < num_ct_; i++) {
    context_->gpu_encoder_.Encode(bias_pt_[i], bias_level_, scale, bias[i]);
  }
  cached_bias_ = bias;
  cached_bias_level_ = bias_level_;
}

template <typename word>
size_t LayerNormHandler<word>::GetPlaintextBytes() const {
  size_t bytes = 0;
  auto add = [&](int level, size_t count) {
    if (level < 0) return;
    const size_t words =
        static_cast<size_t>(context_->param_.LevelToNP(level).GetNumTotal()) *
        context_->param_.degree_;
    bytes += count * words * sizeof(word);
  };
  add(cached_mask_level_, mask_pt_.size());
  add(cached_weight_level_, weight_pt_.size());
  add(cached_bias_level_, bias_pt_.size());
  return bytes;
}

template <typename word>
void LayerNormHandler<word>::ChannelSum(Ct &acc, const std::vector<Ct> &x,
                                        const EvkMap<word> &evk) const {
  AssertTrue(static_cast<int>(x.size()) == num_ct_,
             "LayerNorm: wrong number of input ciphertexts");
  // The channel axis is crossed between ciphertexts by a plain add; the rest
  // of the reduction is inside one.
  for (int i = 0; i < num_ct_; i++) {
    if (i == 0) {
      context_->Copy(acc, x[i]);
    } else {
      context_->Add(acc, acc, x[i]);
    }
  }
  Ct rotated;
  for (int d : rotation_distances_) {
    // HRotAdd is res = (a << dist) + b, so res must not alias either input.
    context_->HRotAdd(rotated, acc, acc, evk.GetRotationKey(d), d);
    context_->Copy(acc, rotated);
  }
}

template <typename word>
void LayerNormHandler<word>::SumOfSquares(Ct &acc, const std::vector<Ct> &xc,
                                          const EvkMap<word> &evk) const {
  AssertTrue(static_cast<int>(xc.size()) == num_ct_,
             "LayerNorm: wrong number of input ciphertexts");
  const auto &mult_key = evk.GetMultiplicationKey();
  Ct sq;
  for (int i = 0; i < num_ct_; i++) {
    context_->HMult(sq, xc[i], xc[i], mult_key);
    if (i == 0) {
      context_->Copy(acc, sq);
    } else {
      context_->Add(acc, acc, sq);
    }
  }
  Ct rotated;
  for (int d : rotation_distances_) {
    context_->HRotAdd(rotated, acc, acc, evk.GetRotationKey(d), d);
    context_->Copy(acc, rotated);
  }
}

template <typename word>
void LayerNormHandler<word>::Centre(
    std::vector<Ct> &xc, const std::vector<Ct> &x,
    const std::vector<std::vector<Complex>> &mask,
    const EvkMap<word> &evk) const {
  NvtxScope _nv("layernorm: Centre");
  PrepareMask(mask);
  Ct sum;
  ChannelSum(sum, x, evk);
  xc.clear();
  xc.resize(num_ct_);
  Ct scaled, levelled;
  for (int i = 0; i < num_ct_; i++) {
    // `mu` masked and divided in ONE multiply: the plaintext carries
    // `1/live` where the image has data and zero where it does not, so the
    // centred image is exactly zero off the data and the variance below
    // collects nothing that is not there (the header's contract).
    context_->Mult(scaled, sum, mask_pt_[i]);
    Ct mu;
    context_->Rescale(mu, scaled);
    context_->LevelDown(levelled, x[i], centre_level_);
    context_->Sub(xc[i], levelled, mu);
  }
}

template <typename word>
void LayerNormHandler<word>::Apply(
    std::vector<Ct> &res, const std::vector<Ct> &x,
    const std::vector<std::vector<Complex>> &weight,
    const std::vector<std::vector<Complex>> &bias,
    const std::vector<std::vector<Complex>> &mask,
    const EvkMap<word> &evk_map) const {
  NvtxScope _nv("layernorm: Apply");
  AssertTrue(static_cast<int>(x.size()) == num_ct_,
             "LayerNorm: wrong number of input ciphertexts");
  const auto &mult_key = evk_map.GetMultiplicationKey();

  // 1. Centre. One level, and the variance below is the mean square of what
  //    comes out -- no `mean(x^2) - mu^2` cancellation anywhere.
  std::vector<Ct> xc;
  Centre(xc, x, mask, evk_map);

  // 2. The variance, broadcast to every slot.
  Ct acc;
  SumOfSquares(acc, xc, evk_map);

  // 3. The affine map onto [-1, 1], as one constant multiply and one constant
  //    add: u = alpha * (S / live + eps), v = (u - b) / a.
  const double a = affine_scale_;
  const double b = 0.5 * (window_hi_ + window_lo_);
  const double k = layer_constant_ / (live_ * a);
  const int acc_level = context_->param_.NPToLevel(acc.GetNP());
  Constant<word> k_const;
  context_->encoder_.EncodeConstant(
      k_const, acc_level, context_->param_.GetScale(acc_level), k);
  Ct scaled;
  context_->Mult(scaled, acc, k_const);
  context_->Rescale(acc, scaled);
  const int v_level = context_->param_.NPToLevel(acc.GetNP());
  Constant<word> shift_const;
  context_->encoder_.EncodeConstant(shift_const, v_level,
                                    context_->param_.GetScale(v_level),
                                    (layer_constant_ * eps_ - b) / a);
  context_->Add(acc, acc, shift_const);

  // 4. The inverse square root.
  Ct r;
  inv_sqrt_->Evaluate(context_, r, acc, mult_key);

  // 5. Apply, then the gain, then the bias. The gain plaintexts carry
  //    `sqrt(alpha_L)` by this class's contract, since
  //    `1/sqrt(var) = sqrt(alpha_L) * r`; the bias is in the stream's units.
  const int r_level = context_->param_.NPToLevel(r.GetNP());
  AssertTrue(r_level - 1 == weight_level_,
             "LayerNorm: the inverse square root did not land where Prepare "
             "assumed");
  Prepare(weight, bias);
  res.clear();
  res.resize(num_ct_);
  Ct levelled, product;
  for (int i = 0; i < num_ct_; i++) {
    context_->LevelDown(levelled, xc[i], r_level);
    context_->HMult(product, levelled, r, mult_key);
    // `Mult(Ct, Pt)` does not rescale, so the rescale here is what puts the
    // output back on its level's canonical scale -- which is what lets the
    // bias be an ordinary plaintext add.
    context_->Mult(product, product, weight_pt_[i]);
    context_->Rescale(res[i], product);
    if (!bias_pt_.empty()) context_->Add(res[i], res[i], bias_pt_[i]);
  }
}

template class LayerNormHandler<uint32_t>;
template class LayerNormHandler<uint64_t>;

}  // namespace cheddar
