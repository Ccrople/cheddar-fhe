#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/RmsNorm.h"

namespace cheddar {

namespace {

// Chebyshev interpolation of f on [-1, 1] at degree n, returned in the
// Chebyshev basis. Interpolating at the Chebyshev nodes rather than fitting in
// the monomial basis is not a stylistic choice: on this window the monomial
// coefficients of an inverse square root span many orders of magnitude, and
// EvalPoly silently drops any coefficient below 1e-9 (Doing.md 1.4).
template <typename F>
std::vector<double> ChebyshevInterpolate(F f, int n) {
  const int m = n + 1;
  std::vector<double> node(m), value(m);
  for (int j = 0; j < m; j++) {
    node[j] = std::cos(M_PI * (j + 0.5) / m);
    value[j] = f(node[j]);
  }
  std::vector<double> c(m, 0.0);
  for (int k = 0; k < m; k++) {
    double acc = 0.0;
    for (int j = 0; j < m; j++) {
      acc += value[j] * std::cos(M_PI * k * (j + 0.5) / m);
    }
    c[k] = (k == 0 ? 1.0 : 2.0) * acc / m;
  }
  return c;
}

}  // namespace

template <typename word>
RmsNormHandler<word>::RmsNormHandler(ConstContextPtr<word> context,
                                     int num_tokens, int num_channels,
                                     double layer_constant, int input_level,
                                     double eps, double window_ratio,
                                     int degree)
    : context_{std::move(context)},
      num_tokens_{num_tokens},
      num_channels_{num_channels},
      layer_constant_{layer_constant},
      eps_{eps},
      input_level_{input_level} {
  AssertTrue(num_tokens_ > 0 && IsPowOfTwo(num_tokens_),
             "RmsNorm: num_tokens must be a power of two");
  AssertTrue(num_channels_ > 0 && IsPowOfTwo(num_channels_),
             "RmsNorm: num_channels must be a power of two");
  AssertTrue(layer_constant_ > 0.0, "RmsNorm: layer constant must be positive");
  AssertTrue(window_ratio > 1.0, "RmsNorm: window ratio must exceed one");
  window_lo_ = 1.0 / std::sqrt(window_ratio);
  window_hi_ = std::sqrt(window_ratio);

  num_slots_ = context_->param_.degree_ / 2;
  AssertTrue(num_slots_ % num_tokens_ == 0,
             "RmsNorm: tokens must divide the slot count");
  const long long total = 1LL * num_tokens_ * num_channels_;
  AssertTrue(total % num_slots_ == 0,
             "RmsNorm: T * H must be a multiple of the slot count");
  num_ct_ = static_cast<int>(total / num_slots_);

  // Rotate-and-add over whole token blocks. The same sequence reduces and
  // broadcasts, so the sum lands in every block and nothing has to be
  // redistributed afterwards.
  for (int d = num_tokens_; d < num_slots_; d *= 2) {
    rotation_distances_.push_back(d);
  }

  // The polynomial lives on [-1, 1], so the argument is mapped there first:
  // v = (u - b) / a with a, b the half-width and centre of the window.
  const double a = 0.5 * (window_hi_ - window_lo_);
  const double b = 0.5 * (window_hi_ + window_lo_);
  auto coeffs = ChebyshevInterpolate(
      [a, b](double v) { return 1.0 / std::sqrt(a * v + b); }, degree);

  // The multiplicative half of the affine map is applied by *reinterpreting the
  // scale*, not by a constant multiplication. A ciphertext holding round(S*D)
  // and declared to be at scale D/k decodes as k*S -- no kernel, no level, and
  // no loss. Cheddar's Mult(Ct, Const) does not rescale either, so taking that
  // route would have left the scale at D^2 and forced a Rescale to recover, at
  // the cost of the level this avoids.
  affine_scale_ = 0.5 * (window_hi_ - window_lo_);
  // The polynomial is compiled at the canonical scale of its own level. An
  // earlier version applied the affine map by reinterpreting the scale, which
  // is free and arithmetically sound, but left EvalPoly with a non-canonical
  // input scale; the encrypted error then grew monotonically with the
  // argument while the same coefficients were exact in the clear. The
  // multiply-and-rescale below costs a level and keeps every scale canonical.
  const int poly_level = input_level_ - 2;
  const double poly_scale = context_->param_.GetScale(poly_level);
  inv_sqrt_ = std::make_unique<EvalPoly<word>>(
      coeffs, poly_level, poly_scale, poly_scale, /*chebyshev=*/true);
  inv_sqrt_->Compile(context_);
}

template <typename word>
double RmsNormHandler<word>::PlainInvSqrt(double u) const {
  const double a = affine_scale_;
  const double b = 0.5 * (window_hi_ + window_lo_);
  return inv_sqrt_->PlainEvaluate((u - b) / a);
}

template <typename word>
void RmsNormHandler<word>::Apply(
    std::vector<Ct> &res, const std::vector<Ct> &x,
    const std::vector<std::vector<Complex>> &weight,
    const EvkMap<word> &evk_map) const {
  AssertTrue(static_cast<int>(x.size()) == num_ct_,
             "RmsNorm: wrong number of input ciphertexts");
  AssertTrue(weight.size() == x.size(),
             "RmsNorm: one weight plaintext per input ciphertext");

  const auto &mult_key = evk_map.GetMultiplicationKey();

  // 1. Square, and accumulate across the ciphertexts that hold one token's
  //    channels. This is the only place the channel axis is crossed between
  //    ciphertexts; the rest of the reduction is inside one.
  Ct acc, sq;
  for (int i = 0; i < num_ct_; i++) {
    context_->HMult(sq, x[i], x[i], mult_key);
    if (i == 0) {
      context_->Copy(acc, sq);
    } else {
      context_->Add(acc, acc, sq);
    }
  }

  // 2. All-reduce over the channel axis within the ciphertext. The same
  //    sequence reduces and broadcasts, so afterwards every slot already holds
  //    its own token's sum and nothing has to be redistributed.
  //
  //    HRotAdd is res = (a << dist) + b, so res must not alias a or b: the
  //    rotation would read words the accumulation has already overwritten.
  Ct rotated;
  for (int d : rotation_distances_) {
    context_->HRotAdd(rotated, acc, acc, evk_map.GetRotationKey(d), d);
    context_->Copy(acc, rotated);
  }

  // 3. Affine map onto the polynomial's domain, folded into one constant
  //    multiply and one constant add:
  //        u = alpha_L * (S / H + eps),   v = (u - b) / a
  //    so the multiplicative half is alpha_L / (H * a) and the additive half
  //    carries the epsilon: (alpha_L * eps - b) / a. Llama's epsilon is not
  //    negligible at this scale -- mean square ~5e-4 against eps 1e-5 -- and it
  //    rides along for free, since constant addition costs no level.
  //
  //    [SYLPH] fuses scalings like this into the operation that produced x;
  //    kept explicit here so the caller can see -- and move -- the level it
  //    costs.
  const double a = affine_scale_;
  const double b = 0.5 * (window_hi_ + window_lo_);
  const double k = layer_constant_ / (num_channels_ * a);
  const int acc_level = context_->param_.NPToLevel(acc.GetNP());

  // Multiplicative half. Mult(Ct, Const) does not rescale in Cheddar, so the
  // scale has to be brought back explicitly; that is the level this costs.
  Constant<word> k_const;
  context_->encoder_.EncodeConstant(
      k_const, acc_level, context_->param_.GetScale(acc_level), k);
  Ct scaled;
  context_->Mult(scaled, acc, k_const);
  context_->Rescale(acc, scaled);

  // Additive half. Constant addition is level-free, which is why the epsilon
  // rides along at no cost.
  const int v_level = context_->param_.NPToLevel(acc.GetNP());
  Constant<word> shift_const;
  context_->encoder_.EncodeConstant(shift_const, v_level,
                                    context_->param_.GetScale(v_level),
                                    (layer_constant_ * eps_ - b) / a);
  context_->Add(acc, acc, shift_const);

  // 4. The inverse square root.
  Ct r;
  inv_sqrt_->Evaluate(context_, r, acc, mult_key);

  // 5. Apply. The weight plaintexts are expected to carry sqrt(alpha_L)
  //    already, since 1/sqrt(mean) = sqrt(alpha_L) * r.
  res.clear();
  res.resize(num_ct_);
  const int r_level = context_->param_.NPToLevel(r.GetNP());
  Ct levelled;
  Pt w_pt;
  for (int i = 0; i < num_ct_; i++) {
    context_->LevelDown(levelled, x[i], r_level);
    context_->HMult(res[i], levelled, r, mult_key);
    // Encode the weight here rather than take it pre-encoded: it has to sit at
    // the level the polynomial left behind, and that depends on the Chebyshev
    // degree. Making the caller supply a plaintext at the right level makes an
    // internal detail of the circuit part of its interface, and getting it
    // wrong surfaces as "Number of primes differ" from inside the multiply.
    const int y_level = context_->param_.NPToLevel(res[i].GetNP());
    context_->encoder_.Encode(w_pt, y_level,
                              context_->param_.GetScale(y_level), weight[i]);
    context_->Mult(res[i], res[i], w_pt);
  }
}

template class RmsNormHandler<uint32_t>;
template class RmsNormHandler<uint64_t>;

}  // namespace cheddar
