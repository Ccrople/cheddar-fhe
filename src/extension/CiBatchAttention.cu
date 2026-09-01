#include "extension/CiBatchAttention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/EncodeGpu.h"
#include "extension/ChebyshevFit.h"
#include "extension/EvalPoly.h"
#include "extension/Profile.h"

namespace cheddar {

template <typename word>
CiBatchAttention<word>::CiBatchAttention(
    std::shared_ptr<const BootContext<word>> boot,
    ConstContextPtr<word> switch_ctx, ConstContextPtr<word> small_ctx,
    ConstContextPtr<word> lifted_ctx, const Config &cfg)
    : boot_{std::move(boot)},
      switch_ctx_{std::move(switch_ctx)},
      small_ctx_{std::move(small_ctx)},
      lifted_ctx_{std::move(lifted_ctx)},
      cfg_{cfg},
      chain_{switch_ctx_->param_.degree_, small_ctx_->param_.degree_,
             cfg.sub_degree},
      layout_{boot_->param_.MaxNumSlots(), cfg.num_tokens, cfg.sub_degree,
              chain_.rank},
      switcher_{switch_ctx_, small_ctx_},
      lift_{small_ctx_, lifted_ctx_},
      ccmm_{lifted_ctx_->param_, lifted_ctx_->ntt_handler_} {
  AssertTrue(boot_->param_.degree_ == switch_ctx_->param_.degree_,
             "CiBatchAttention: the switching ring must have the layer's "
             "degree");
  AssertTrue(chain_.dim == cfg_.num_tokens,
             "CiBatchAttention: the product ring must hold exactly one "
             "token per block -- d = " +
                 std::to_string(chain_.dim) + " against T = " +
                 std::to_string(cfg_.num_tokens));
  AssertTrue(cfg_.head_dim == cfg_.num_tokens,
             "CiBatchAttention: Algorithm 4 is square here: head_dim must "
             "equal the token count");
  AssertTrue(cfg_.num_heads % cfg_.num_kv_heads == 0,
             "CiBatchAttention: GQA needs num_kv_heads | num_heads");
  AssertTrue(cfg_.forward_level - 1 >= 1 && cfg_.inverse_level >= 1 &&
                 cfg_.rope_level - 1 >= cfg_.forward_level,
             "CiBatchAttention: the level ladder does not close");

  // The two converters. The second call's key tokens 64..127 must sit at
  // blocks 0..63 where the contract wants them: under the chain addressing
  // `BlockOf(t + T/2, g) = BlockOf(t, g) + 1` (bit T/2 of the token is bit
  // 0 of the block index), so the shift down is one slot rotation by
  // `lanes` on the masked ciphertext -- checked here, so a layout change
  // fails loudly rather than in the product.
  const int T = cfg_.num_tokens;
  for (int t = 0; t < T / 2; t++) {
    for (int g = 0; g < chain_.rank; g++) {
      AssertTrue(layout_.BlockOf(t + T / 2, g) == layout_.BlockOf(t, g) + 1,
                 "CiBatchAttention: the key-token shift is not one block");
    }
  }
  fwd_ = std::make_unique<CiSinCConverter<word>>(
      switch_ctx_, cfg_.sub_degree, cfg_.forward_level, /*inverse_level=*/-1,
      &chain_, nullptr, cfg_.converter_baby_steps);
  inv_ = std::make_unique<CiSinCConverter<word>>(
      switch_ctx_, cfg_.sub_degree, /*forward_level=*/-1, cfg_.inverse_level,
      &chain_, nullptr, cfg_.converter_baby_steps);
  BuildRope();
  if (cfg_.verbose) {
    std::cout << "  [batch] attention: chain d " << chain_.dim << ", rank "
              << chain_.rank << ", lanes " << chain_.lanes << "; forward @"
              << cfg_.forward_level << ", chain @" << GetChainLevel()
              << ", inverse @" << cfg_.inverse_level << std::endl;
  }
}

template <typename word>
void CiBatchAttention<word>::AddSwitchRotations(EvkRequest &req) const {
  fwd_->AddRequiredRotations(req);
  inv_->AddRequiredRotations(req);
}

template <typename word>
void CiBatchAttention<word>::AddBootRotations(EvkRequest &req) const {
  req.AddRequest(GetShiftRotation(), cfg_.forward_level);
}

template <typename word>
void CiBatchAttention<word>::BuildRope() {
  // rotate_half: pair m with m + head_dim / 2, angle token * theta_m,
  // theta_m = base^(-2m / head_dim) (`reference_forward.py`'s rope). One
  // plaintext per (mask, pair, cos|sin), constant over the instances of a
  // token, encoded at rope_level's scale so that a multiply and one rescale
  // land canonical one level below.
  const int T = cfg_.num_tokens;
  const int half = cfg_.head_dim / 2;
  const GpuEncoder<word> &encoder = boot_->gpu_encoder_;
  const double scale = boot_->param_.GetScale(cfg_.rope_level);
  std::vector<double> c(T), s(T);
  std::vector<Complex> msg;
  for (int mask = 0; mask < 3; mask++) {
    rope_cos_[mask].resize(half);
    rope_sin_[mask].resize(half);
    for (int m = 0; m < half; m++) {
      const double theta =
          std::pow(cfg_.rope_base, -2.0 * static_cast<double>(m) / cfg_.head_dim);
      for (int t = 0; t < T; t++) {
        const bool live = mask == 0 || (t / (T / 2)) == mask - 1;
        c[t] = live ? std::cos(t * theta) : 0.0;
        s[t] = live ? std::sin(t * theta) : 0.0;
      }
      layout_.PackPerToken(msg, c);
      encoder.Encode(rope_cos_[mask][m], cfg_.rope_level, scale, msg);
      layout_.PackPerToken(msg, s);
      encoder.Encode(rope_sin_[mask][m], cfg_.rope_level, scale, msg);
    }
  }
  // V's call masks: the call's key tokens kept.
  for (int call = 0; call < 2; call++) {
    for (int t = 0; t < T; t++) c[t] = (t / (T / 2) == call) ? 1.0 : 0.0;
    layout_.PackPerToken(msg, c);
    encoder.Encode(call_mask_[call], cfg_.rope_level, scale, msg);
  }
}

template <typename word>
void CiBatchAttention<word>::Rope(std::vector<Ct> &cts, int call) const {
  NvtxScope _nv("batch attn: RoPE");
  const int half = cfg_.head_dim / 2;
  AssertTrue(static_cast<int>(cts.size()) == cfg_.head_dim,
             "CiBatchAttention::Rope: one head's channels");
  const int mask = call < 0 ? 0 : call + 1;
  const NPInfo np = boot_->param_.LevelToNP(cfg_.rope_level);
  for (int m = 0; m < half; m++) {
    Ct &lo = cts[m];
    Ct &hi = cts[m + half];
    AssertTrue(lo.GetNP() == np && hi.GetNP() == np,
               "CiBatchAttention::Rope: the channels are not at rope_level");
    Ct aa, bb, t;
    boot_->Mult(aa, lo, rope_cos_[mask][m]);
    boot_->Mult(t, hi, rope_sin_[mask][m]);
    boot_->Sub(aa, aa, t);
    boot_->Mult(bb, hi, rope_cos_[mask][m]);
    boot_->Mult(t, lo, rope_sin_[mask][m]);
    boot_->Add(bb, bb, t);
    boot_->Rescale(lo, aa);
    boot_->Rescale(hi, bb);
  }
}

template <typename word>
void CiBatchAttention<word>::Descend(std::vector<Ct> &lifted, Ct &ct,
                                     int call, const Keys &keys) const {
  NvtxScope _nv("batch attn: descend");
  // On the LAYER's Context: the switching ring shares ci16_35's levels
  // 0..4 only, and a channel arrives above them.
  Ct down;
  boot_->LevelDown(down, ct, cfg_.forward_level);
  ct = Ct();
  if (call == 1) {
    // The second call's key tokens, one block up, brought down: slot s ->
    // s - lanes (`HRot` by r moves slot i + r to i).
    AssertTrue(keys.boot != nullptr,
               "CiBatchAttention::Descend: the shift needs Keys::boot");
    const int r = GetShiftRotation();
    Ct shifted;
    boot_->HRot(shifted, down, keys.boot->GetRotationKey(r), r);
    down = std::move(shifted);
  }
  Ct sinc;
  fwd_->SlotToSinC(switch_ctx_, sinc, down, *keys.swtch);
  down = Ct();
  std::vector<Ct> parts;
  switcher_.Switch(parts, sinc, *keys.ring_switch);
  AssertTrue(static_cast<int>(parts.size()) == chain_.rank,
             "CiBatchAttention::Descend: the switch returned the wrong "
             "number of parts");
  lifted.clear();
  lifted.resize(chain_.rank);
  for (int g = 0; g < chain_.rank; g++) lift_.Lift(lifted[g], parts[g]);
}

template <typename word>
void CiBatchAttention<word>::Return(Ct &res, const std::vector<Ct> &parts,
                                    const Keys &keys) const {
  NvtxScope _nv("batch attn: return");
  Ct big;
  switcher_.SwitchBack(big, parts, *keys.inverse_ring_switch);
  inv_->SinCToSlot(switch_ctx_, res, big, *keys.swtch);
}

template <typename word>
void CiBatchAttention<word>::Scores(std::vector<Ct> &res, std::vector<Ct> &q,
                                    const std::vector<Ct> &k,
                                    const Keys &keys) const {
  NvtxScope _nv("batch attn: Scores");
  const int T = cfg_.num_tokens;
  const int D = cfg_.head_dim;
  const int rank = chain_.rank;
  AssertTrue(static_cast<int>(q.size()) == D && static_cast<int>(k.size()) == D,
             "CiBatchAttention::Scores: one head's channels each");
  AssertTrue(keys.swtch != nullptr && keys.lifted != nullptr &&
                 keys.ring_switch != nullptr &&
                 keys.inverse_ring_switch != nullptr,
             "CiBatchAttention::Scores: keys");

  // Q: RoPE, then every channel down to its 16 group parts on the lifted
  // ring. lq[g][c] is group g's column c of the lhs.
  Rope(q, -1);
  std::vector<std::vector<Ct>> lq(rank);
  for (int c = 0; c < D; c++) {
    std::vector<Ct> parts;
    Descend(parts, q[c], -1, keys);
    for (int g = 0; g < rank; g++) lq[g].push_back(std::move(parts[g]));
  }

  res.clear();
  res.resize(T);
  for (int call = 0; call < 2; call++) {
    NvtxScope _c("batch attn: scores call");
    // K: RoPE with the call's key tokens kept, down to its parts.
    std::vector<Ct> kc(D);
    for (int c = 0; c < D; c++) boot_->Copy(kc[c], k[c]);
    Rope(kc, call);
    std::vector<std::vector<Ct>> lk(rank);
    for (int c = 0; c < D; c++) {
      std::vector<Ct> parts;
      Descend(parts, kc[c], call, keys);
      for (int g = 0; g < rank; g++) lk[g].push_back(std::move(parts[g]));
    }
    // Per group: the elided Algorithm 4, the live half of its columns
    // descended to the product ring.
    std::vector<std::vector<Ct>> out(rank);
    for (int g = 0; g < rank; g++) {
      std::vector<Ct> prod;
      ccmm_.Multiply(lifted_ctx_, prod, lq[g], lk[g], 2 * cfg_.sub_degree,
                     *keys.lifted, /*rhs_row_wise=*/true);
      lk[g].clear();
      out[g].resize(T / 2);
      for (int l = 0; l < T / 2; l++) lift_.Descend(out[g][l], prod[l]);
    }
    // Per key token: the groups' parts back into one big ciphertext and
    // into slots.
    for (int l = 0; l < T / 2; l++) {
      std::vector<Ct> parts(rank);
      for (int g = 0; g < rank; g++) parts[g] = std::move(out[g][l]);
      Return(res[call * (T / 2) + l], parts, keys);
    }
  }
}

// ---------------------------------------------------------------------------
// The softmax walk
// ---------------------------------------------------------------------------

namespace {

// The worst error of the degree-`degree` Chebyshev interpolant of
// exp(hb (v - 1)) on [-1, 1], by Clenshaw on a fine grid.
double ExpFitError(double hb, int degree) {
  auto c = chebfit::Interpolate(
      [hb](double v) { return std::exp(hb * (v - 1.0)); }, degree);
  double worst = 0.0;
  for (int i = 0; i <= 400; i++) {
    const double v = -1.0 + 2.0 * i / 400.0;
    double b0 = 0.0, b1 = 0.0;
    for (size_t j = c.size() - 1; j > 0; j--) {
      const double t = 2.0 * v * b0 - b1 + c[j];
      b1 = b0;
      b0 = t;
    }
    worst = std::max(worst, std::abs(v * b0 - b1 + c[0] -
                                     std::exp(hb * (v - 1.0))));
  }
  return worst;
}

// The degree that reaches sixteen bits, capped at 15 (the level budget:
// see `CiSinCAttention`'s rule, which this repeats).
int ExpDegree(double m_eff) {
  const double hb = std::max(m_eff, 0.0) / 4.0;
  for (int d : {7, 9, 15}) {
    if (ExpFitError(hb, d) < std::pow(2.0, -16.0)) return d;
  }
  return 15;
}

}  // namespace

template <typename word>
void CiBatchAttention<word>::PrepareSoftMax(const SoftMaxCalibration &calib) {
  calib_ = calib;
  const Parameter<word> &param = boot_->param_;
  const int T = cfg_.num_tokens;
  const int top = GetTopLevel();
  exp_in_ = top - 1;
  const int exp_degree =
      (calib_.exp_degree > 0) ? calib_.exp_degree : ExpDegree(calib_.m_eff);
  // k = 1 (Cho): y = exp(m_eff (u - 1) / 4), squared later by the norm.
  const double hb = calib_.m_eff / 4.0;
  auto exp_coeffs = chebfit::Interpolate(
      [hb](double v) { return std::exp(hb * (v - 1.0)); }, exp_degree);
  const int exp_used = EvalPoly<word>(exp_coeffs, exp_in_,
                                      param.GetScale(exp_in_),
                                      param.GetScale(exp_in_), true)
                           .GetPolyDegree();
  exp_out_ = exp_in_ - Log2Ceil(exp_used + 1);
  polys_.clear();
  polys_.push_back(std::make_unique<EvalPoly<word>>(
      exp_coeffs, exp_in_, param.GetScale(exp_in_), param.GetScale(exp_out_),
      true));
  polys_[0]->Compile(boot_);

  mask_level_ = exp_out_;
  sq_level_ = (calib_.causal ? exp_out_ - 1 : exp_out_) - 1;
  poly_in_ = sq_level_ - 1;
  const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
  const double aff_b = 0.5 * (calib_.norm_hi + calib_.norm_lo);
  auto inv_coeffs = chebfit::Interpolate(
      [aff_a, aff_b](double v) { return 1.0 / std::sqrt(aff_a * v + aff_b); },
      calib_.inv_degree);
  const int inv_used = EvalPoly<word>(inv_coeffs, poly_in_,
                                      param.GetScale(poly_in_),
                                      param.GetScale(poly_in_), true)
                           .GetPolyDegree();
  const int inv_out = poly_in_ - Log2Ceil(inv_used + 1);
  AssertTrue(inv_out - 2 >= cfg_.forward_level,
             "CiBatchAttention: the softmax walk overspends its levels; P "
             "would land below forward_level");
  polys_.push_back(std::make_unique<EvalPoly<word>>(
      inv_coeffs, poly_in_, param.GetScale(poly_in_), param.GetScale(inv_out),
      true));
  polys_[1]->Compile(boot_);

  // The per-head row shift as a per-token plaintext at exp_in_'s canonical
  // scale (it is ADDED to the affine's rescaled output). Masked keys keep
  // the global shift so u stays inside the fit domain; here the shift is per
  // query token, so every key column of a row gets the row's own.
  a0_.clear();
  if (calib_.causal) {
    AssertTrue(static_cast<int>(calib_.row_shift.size()) == cfg_.num_heads &&
                   static_cast<int>(calib_.row_shift[0].size()) == T,
               "CiBatchAttention: causal calibration needs a [heads][tokens] "
               "row_shift table");
    AssertTrue(calib_.row_norm.empty() ||
                   (static_cast<int>(calib_.row_norm.size()) ==
                        cfg_.num_heads &&
                    static_cast<int>(calib_.row_norm[0].size()) == T),
               "CiBatchAttention: row_norm must be a [heads][tokens] table");
    a0_.resize(cfg_.num_heads);
    std::vector<double> a0(T);
    std::vector<Complex> msg;
    for (int h = 0; h < cfg_.num_heads; h++) {
      for (int t = 0; t < T; t++) {
        a0[t] = 1.0 - 2.0 * calib_.row_shift[h][t] / calib_.span;
      }
      layout_.PackPerToken(msg, a0);
      boot_->gpu_encoder_.Encode(a0_[h], exp_in_, param.GetScale(exp_in_),
                                 msg);
    }
  }
  softmax_ready_ = true;
  if (cfg_.verbose) {
    std::cout << "  [batch] softmax: exp deg " << exp_used << " @" << exp_in_
              << ".." << exp_out_ << ", mask @" << mask_level_ << ", sq @"
              << sq_level_ << ", invsqrt deg " << inv_used << " @" << poly_in_
              << ".." << inv_out << ", P @" << cfg_.forward_level << std::endl;
  }
}

template <typename word>
void CiBatchAttention<word>::BuildMasks(std::vector<Pt> &masks,
                                        int head) const {
  // Key token l is live for query tokens t >= l; the live value carries the
  // row's norm estimate as est^-1/2 (so the Euclidean norm computes
  // sq / est and est cancels in P = (y r)^2), or 1 without it.
  const int T = cfg_.num_tokens;
  const Parameter<word> &param = boot_->param_;
  masks.clear();
  masks.resize(T);
  std::vector<double> m(T);
  std::vector<Complex> msg;
  for (int l = 0; l < T; l++) {
    for (int t = 0; t < T; t++) {
      double v = 0.0;
      if (t >= l) {
        v = calib_.row_norm.empty()
                ? 1.0
                : 1.0 / std::sqrt(calib_.row_norm[head][t]);
      }
      m[t] = v;
    }
    layout_.PackPerToken(msg, m);
    boot_->gpu_encoder_.Encode(masks[l], mask_level_,
                               param.GetScale(mask_level_), msg);
  }
}

template <typename word>
void CiBatchAttention<word>::SoftMax(std::vector<Ct> &P,
                                     const std::vector<Ct> &scores, int head,
                                     double carried,
                                     const EvkMap<word> &evk) const {
  NvtxScope _nv("batch attn: SoftMax");
  AssertTrue(softmax_ready_, "CiBatchAttention: call PrepareSoftMax first");
  const Parameter<word> &param = boot_->param_;
  const int T = cfg_.num_tokens;
  const int top = GetTopLevel();
  AssertTrue(static_cast<int>(scores.size()) == T,
             "CiBatchAttention::SoftMax: one head's key-token ciphertexts");
  AssertTrue(param.NPToLevel(scores[0].GetNP()) == top,
             "CiBatchAttention::SoftMax: the scores must be booted to the "
             "top level");
  AssertTrue(carried > 0.0, "CiBatchAttention::SoftMax: carried");
  const auto &mult_key = evk.GetMultiplicationKey();

  std::vector<Pt> masks;
  if (calib_.causal) BuildMasks(masks, head);

  // Affine onto the fit domain (carried divides out here), exp, the mask.
  const double a1 = 2.0 / (calib_.span * carried);
  Constant<word> c1;
  boot_->encoder_.EncodeConstant(c1, top, param.GetScale(top), a1);
  std::vector<Ct> y(T);
  Ct sq_acc;
  for (int l = 0; l < T; l++) {
    Ct t1, u;
    boot_->Mult(t1, scores[l], c1);
    boot_->Rescale(u, t1);
    if (calib_.causal) {
      boot_->Add(u, u, a0_[head]);
    } else {
      Constant<word> c0;
      boot_->encoder_.EncodeConstant(c0, exp_in_, u.GetScale(),
                                     1.0 - 2.0 * calib_.shift / calib_.span);
      boot_->Add(u, u, c0);
    }
    Ct y_full;
    polys_[0]->Evaluate(boot_, y_full, u, mult_key);
    if (calib_.causal) {
      Ct t2;
      boot_->Mult(t2, y_full, masks[l]);
      boot_->Rescale(y[l], t2);
    } else {
      y[l] = std::move(y_full);
    }
    // The Euclidean norm over the key axis is a sum over these
    // ciphertexts: the tensor squares accumulate, one relinearization.
    Ct sq;
    boot_->Mult(sq, y[l], y[l]);
    if (l == 0) {
      sq_acc = std::move(sq);
    } else {
      boot_->Add(sq_acc, sq_acc, sq);
    }
  }
  masks.clear();
  Ct sq;
  boot_->RelinearizeRescale(sq, sq_acc, mult_key);
  sq_acc = Ct();
  AssertTrue(param.NPToLevel(sq.GetNP()) == sq_level_,
             "CiBatchAttention::SoftMax: the square did not land at "
             "sq_level");
  {
    const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
    const double aff_b = 0.5 * (calib_.norm_hi + calib_.norm_lo);
    Constant<word> inv_a;
    boot_->encoder_.EncodeConstant(inv_a, sq_level_, param.GetScale(sq_level_),
                                   1.0 / aff_a);
    Ct scaled;
    boot_->Mult(scaled, sq, inv_a);
    boot_->Rescale(sq, scaled);
    Constant<word> shift;
    boot_->encoder_.EncodeConstant(shift, poly_in_, sq.GetScale(),
                                   -aff_b / aff_a);
    boot_->Add(sq, sq, shift);
  }
  Ct r;
  polys_[1]->Evaluate(boot_, r, sq, mult_key);
  const int meet = param.NPToLevel(r.GetNP());
  P.clear();
  P.resize(T);
  for (int l = 0; l < T; l++) {
    Ct levelled, prod;
    boot_->LevelDown(levelled, y[l], meet);
    y[l] = Ct();
    boot_->HMult(prod, levelled, r, mult_key);
    boot_->HMult(P[l], prod, prod, mult_key);
  }
  const int p_level = param.NPToLevel(P[0].GetNP());
  AssertTrue(p_level >= cfg_.forward_level,
             "CiBatchAttention::SoftMax: P landed below forward_level");
  if (p_level > cfg_.forward_level) {
    for (int l = 0; l < T; l++) {
      Ct down;
      boot_->LevelDown(down, P[l], cfg_.forward_level);
      P[l] = std::move(down);
    }
  }
}

// ---------------------------------------------------------------------------
// P V
// ---------------------------------------------------------------------------

template <typename word>
void CiBatchAttention<word>::ZeroLifted(std::vector<Ct> &res, const Ct &like,
                                        int count) const {
  const NPInfo np = like.GetNP();
  const size_t component_bytes = static_cast<size_t>(np.GetNumTotal()) *
                                 lifted_ctx_->param_.degree_ * sizeof(word);
  res.clear();
  res.resize(count);
  for (auto &zero : res) {
    zero.RemoveRx();
    zero.ModifyNP(np);
    zero.SetScale(like.GetScale());
    zero.SetNumSlots(like.GetNumSlots());
    cudaMemsetAsync(zero.bx_.data(), 0, component_bytes, cudaStreamLegacy);
    cudaMemsetAsync(zero.ax_.data(), 0, component_bytes, cudaStreamLegacy);
  }
}

template <typename word>
void CiBatchAttention<word>::Values(std::vector<Ct> &res, std::vector<Ct> &P,
                                    const std::vector<Ct> &v,
                                    const Keys &keys) const {
  NvtxScope _nv("batch attn: Values");
  const int T = cfg_.num_tokens;
  const int D = cfg_.head_dim;
  const int rank = chain_.rank;
  const int half = T / 2;
  AssertTrue(static_cast<int>(P.size()) == T && static_cast<int>(v.size()) == D,
             "CiBatchAttention::Values: P's key tokens and one kv head's "
             "channels");
  AssertTrue(boot_->param_.NPToLevel(P[0].GetNP()) == cfg_.forward_level,
             "CiBatchAttention::Values: P must be at forward_level");

  // P's descent: lp[g][l] is group g's column l (key token).
  std::vector<std::vector<Ct>> lp(rank);
  for (int l = 0; l < T; l++) {
    std::vector<Ct> parts;
    Descend(parts, P[l], -1, keys);
    for (int g = 0; g < rank; g++) lp[g].push_back(std::move(parts[g]));
  }

  // out[g][c]: the product's column c on the product ring, summed over the
  // two calls.
  std::vector<std::vector<Ct>> out(rank);
  for (int call = 0; call < 2; call++) {
    NvtxScope _c("batch attn: values call");
    // V with the call's key tokens kept, shifted down for the odd call by
    // the converter's premap, down to its parts.
    std::vector<std::vector<Ct>> lv(rank);
    for (int c = 0; c < D; c++) {
      Ct masked, t;
      boot_->Mult(t, v[c], call_mask_[call]);
      boot_->Rescale(masked, t);
      std::vector<Ct> parts;
      Descend(parts, masked, call, keys);
      for (int g = 0; g < rank; g++) lv[g].push_back(std::move(parts[g]));
    }
    for (int g = 0; g < rank; g++) {
      // The contract's lhs: P's key tokens of this call as columns 0..63,
      // the dead half exact zeros.
      std::vector<Ct> lhs;
      lhs.reserve(T);
      for (int l = 0; l < half; l++) lhs.push_back(std::move(lp[g][call * half + l]));
      std::vector<Ct> zeros;
      ZeroLifted(zeros, lhs[0], T - half);
      for (auto &z : zeros) lhs.push_back(std::move(z));
      std::vector<Ct> prod;
      ccmm_.Multiply(lifted_ctx_, prod, lhs, lv[g], 2 * cfg_.sub_degree,
                     *keys.lifted, /*rhs_row_wise=*/false);
      lv[g].clear();
      if (call == 0) {
        out[g].resize(D);
        for (int c = 0; c < D; c++) lift_.Descend(out[g][c], prod[c]);
      } else {
        for (int c = 0; c < D; c++) {
          Ct part;
          lift_.Descend(part, prod[c]);
          small_ctx_->Add(out[g][c], out[g][c], part);
        }
      }
    }
  }
  // Per channel: the groups' parts back into one big ciphertext, to slots.
  res.clear();
  res.resize(D);
  for (int c = 0; c < D; c++) {
    std::vector<Ct> parts(rank);
    for (int g = 0; g < rank; g++) parts[g] = std::move(out[g][c]);
    Return(res[c], parts, keys);
  }
}

template class CiBatchAttention<uint32_t>;
template class CiBatchAttention<uint64_t>;

}  // namespace cheddar
