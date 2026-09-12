#include "extension/CiBertBase.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/EncodeGpu.h"
#include "extension/ChebyshevFit.h"
#include "extension/Profile.h"

namespace cheddar {

#ifdef USE_CUBLAS

namespace {
using Clock = std::chrono::steady_clock;
double Since(Clock::time_point t0) {
  cudaDeviceSynchronize();
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
}  // namespace

namespace kernel {
//! `dst[c][o] = src[c][cperm[o]] * rgain[c] * cgain[o]` (null = identity/1):
//! every way a public factor enters a weight that is read by columns.
__global__ void FoldColsKernel(float *dst, const float *src, const int *cperm,
                               const float *rgain, const float *cgain, int in,
                               int out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(in) * out) return;
  const int c = static_cast<int>(idx / out), o = static_cast<int>(idx % out);
  float v = src[static_cast<size_t>(c) * out + (cperm ? cperm[o] : o)];
  if (rgain) v *= rgain[c];
  if (cgain) v *= cgain[o];
  dst[idx] = v;
}
//! `dst[j][o] = src[rperm[j]][o] * cgain[o]`: the same for a weight read by
//! rows (the down projection, whose rows are the permuted hidden axis).
__global__ void FoldRowsKernel(float *dst, const float *src, const int *rperm,
                               const float *cgain, int rows, int cols) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(rows) * cols) return;
  const int j = static_cast<int>(idx / cols), o = static_cast<int>(idx % cols);
  float v = src[static_cast<size_t>(rperm[j]) * cols + o];
  if (cgain) v *= cgain[o];
  dst[idx] = v;
}
}  // namespace kernel

// ------------------------------------------------------------ construction
template <typename word>
CiBertBaseLayer<word>::CiBertBaseLayer(std::shared_ptr<BootContext<word>> boot,
                                       const Config &cfg)
    : boot_{std::move(boot)},
      cfg_{cfg},
      layout_{boot_->param_.MaxNumSlots(), cfg.shape.tokens} {
  const Shape &s = cfg_.shape;
  AssertTrue(boot_->param_.conjugate_invariant_,
             "CiBertBaseLayer: the batched layout is written for R+");
  AssertTrue(s.heads * s.head_dim == s.model,
             "CiBertBaseLayer: heads * head_dim must be the model width");
  AssertTrue(cfg_.rows_per_tile >= 1, "CiBertBaseLayer: rows_per_tile");
  AssertTrue(s.hidden % cfg_.rows_per_tile == 0 || s.hidden < cfg_.rows_per_tile,
             "CiBertBaseLayer: rows_per_tile must divide the hidden width "
             "(the feed-forward's tiles are the down projection's operands)");
  const int T = s.tokens;
  if (cfg_.baby_steps > 0) {
    baby_ = cfg_.baby_steps;
  } else {
    baby_ = 1;
    while (baby_ * baby_ < T) baby_ <<= 1;
  }
  AssertTrue(T % baby_ == 0, "CiBertBaseLayer: baby steps must divide T");
  giant_ = T / baby_;
  ffn_tiles_ = DivCeil(s.hidden, std::min(cfg_.rows_per_tile, s.hidden));
  typename CiBatchProjection<word>::Config pcfg;
  pcfg.rows_per_tile = std::min(cfg_.rows_per_tile, s.hidden);
  pcfg.verbose = false;
  proj_ = std::make_unique<CiBatchProjection<word>>(boot_, pcfg);
}

template <typename word>
int CiBertBaseLayer<word>::TopLevel() const {
  return boot_->GetBootParameter().GetEndLevel();
}

template <typename word>
int CiBertBaseLayer<word>::Level(const Ct &ct) const {
  return boot_->param_.NPToLevel(ct.GetNP());
}

template <typename word>
int CiBertBaseLayer<word>::Levels(int degree) {
  return Log2Ceil(degree + 1);
}

template <typename word>
double CiBertBaseLayer<word>::InputCarry() const {
  AssertTrue(prepared_, "CiBertBaseLayer::InputCarry: Prepare first");
  return cfg_.ride / cal_.in_absmax;
}

template <typename word>
void CiBertBaseLayer<word>::AddRequiredRotations(EvkRequest &req) const {
  boot_->AddRequiredRotations(req, layout_.num_slots);
  const int B = layout_.num_instances, N = layout_.num_slots;
  const int level = TopLevel();
  for (int b = 1; b < baby_; b++) req.AddRequest((b * B) % N, level);
  for (int g = 1; g < giant_; g++) {
    const int d = (g * baby_ * B) % N;
    req.AddRequest(d, level);
    req.AddRequest((N - d) % N, level);
  }
}

// ---------------------------------------------------------------- helpers
template <typename word>
void CiBertBaseLayer<word>::MultInt(Ct &res, const Ct &a, double value) const {
  Const c;
  boot_->encoder_.EncodeConstant(c, Level(a), 1.0, value);
  boot_->Mult(res, a, c);
}

template <typename word>
void CiBertBaseLayer<word>::MultScalar(Ct &res, const Ct &a, double value) const {
  const int lv = Level(a);
  AssertTrue(lv >= 1, "CiBertBaseLayer::MultScalar: no level left");
  Const c;
  boot_->encoder_.EncodeConstant(c, lv, boot_->param_.GetScale(lv), value);
  Ct t;
  boot_->Mult(t, a, c);
  boot_->Rescale(res, t);
}

template <typename word>
void CiBertBaseLayer<word>::AddScalar(Ct &res, const Ct &a, double value) const {
  Const c;
  boot_->encoder_.EncodeConstant(c, Level(a), a.GetScale(), value);
  boot_->Add(res, a, c);
}

template <typename word>
void CiBertBaseLayer<word>::PerToken(Pt &pt, const std::vector<double> &per_token,
                                     const Ct &like) const {
  std::vector<Complex> msg;
  layout_.PackPerToken(msg, per_token);
  boot_->gpu_encoder_.Encode(pt, Level(like), like.GetScale(), msg);
}

template <typename word>
void CiBertBaseLayer<word>::PerSlot(Pt &pt, const std::vector<double> &values,
                                    const Ct &like) const {
  std::vector<Complex> msg;
  layout_.Pack(msg, values);
  boot_->gpu_encoder_.Encode(pt, Level(like), like.GetScale(), msg);
}

// ------------------------------------------------------------------ fold
template <typename word>
bool CiBertBaseLayer<word>::Folds(int j) const {
  return cfg_.fold && j < static_cast<int>(cal_.est.size()) &&
         !cal_.est[j].empty();
}

template <typename word>
void CiBertBaseLayer<word>::FoldEstimate(std::vector<double> &est, int j,
                                         int head) const {
  const int T = cfg_.shape.tokens, B = layout_.num_instances;
  const auto &row = cal_.est[j][head];
  AssertTrue(static_cast<int>(row.size()) == T,
             "CiBertBaseLayer: the fold estimate is [pass][head][token]");
  const int p = (j < static_cast<int>(cal_.est_live_pow.size()))
                    ? cal_.est_live_pow[j]
                    : 0;
  est.assign(static_cast<size_t>(B) * T, 0.0);
  for (int b = 0; b < B; b++) {
    const double w = std::pow(live_[b], static_cast<double>(p));
    for (int t = 0; t < T; t++) {
      const double e = row[t] * w;
      AssertTrue(e > 0.0, "CiBertBaseLayer: a fold estimate is not positive");
      est[static_cast<size_t>(b) * T + t] = e;
    }
  }
}

template <typename word>
const typename CiBertBaseLayer<word>::Pt &CiBertBaseLayer<word>::FoldScale(
    int j, int head, const Ct &like, double unit, double hi) {
  const int idx = j * cfg_.shape.heads + head;
  const int lv = Level(like);
  const double sc = like.GetScale();
  if (fold_a_lv_[idx] != lv || std::abs(fold_a_sc_[idx] - sc) > 1e-9 * sc) {
    std::vector<double> est;
    FoldEstimate(est, j, head);
    for (auto &e : est) e = cfg_.ride / (e * unit * hi);
    PerSlot(fold_a_[idx], est, like);
    fold_a_lv_[idx] = lv;
    fold_a_sc_[idx] = sc;
  }
  return fold_a_[idx];
}

template <typename word>
const typename CiBertBaseLayer<word>::Pt &CiBertBaseLayer<word>::FoldUndo(
    int j, int head, const Ct &like, double unit, double tail) {
  const int idx = j * cfg_.shape.heads + head;
  const int lv = Level(like);
  const double sc = like.GetScale();
  if (fold_b_lv_[idx] != lv || std::abs(fold_b_sc_[idx] - sc) > 1e-9 * sc) {
    std::vector<double> est;
    FoldEstimate(est, j, head);
    for (auto &e : est) e = tail / std::sqrt(e * unit);
    PerSlot(fold_b_[idx], est, like);
    fold_b_lv_[idx] = lv;
    fold_b_sc_[idx] = sc;
  }
  return fold_b_[idx];
}

template <typename word>
template <typename F>
std::unique_ptr<EvalPoly<word>> CiBertBaseLayer<word>::Compile(
    F g, int degree, int level, double in_scale, int *landing) const {
  auto coeffs = chebfit::Interpolate(g, degree);
  const int used =
      EvalPoly<word>(coeffs, level, in_scale, in_scale, true).GetPolyDegree();
  const int lr = level - Levels(used);
  AssertTrue(lr >= 0, "CiBertBaseLayer: a polynomial runs out of levels");
  auto p = std::make_unique<EvalPoly<word>>(coeffs, level, in_scale,
                                            boot_->param_.GetScale(lr), true);
  p->Compile(boot_);
  if (landing != nullptr) *landing = lr;
  return p;
}

template <typename word>
void CiBertBaseLayer<word>::Rotate(Ct &res, const Ct &a, int dist,
                                   const EvkMap<word> &evk) {
  const int N = layout_.num_slots;
  int idx = dist % N;
  if (idx < 0) idx += N;
  if (idx == 0) {
    boot_->Copy(res, a);
    return;
  }
  boot_->HRot(res, a, evk.GetRotationKey(idx), idx);
  stages_.rotations++;
}

template <typename word>
void CiBertBaseLayer<word>::RotateMany(std::vector<Ct> &res, const Ct &a,
                                       const std::vector<int> &dists,
                                       const EvkMap<word> &evk) {
  const int N = layout_.num_slots;
  const int n = static_cast<int>(dists.size());
  res.clear();
  res.resize(n);
  std::vector<int> idx(n);
  for (int i = 0; i < n; i++) {
    int d = dists[i] % N;
    if (d < 0) d += N;
    idx[i] = d;
  }
  int switches = 0;
  for (int i = 0; i < n; i++) switches += (idx[i] != 0) ? 1 : 0;
  if (!cfg_.hoist || switches <= 1) {
    for (int i = 0; i < n; i++) Rotate(res[i], a, idx[i], evk);
    return;
  }
  NvtxScope _nv("bert-base: hoisted rotations");
  // One decomposition of the a-part serves every distance: what a key switch
  // does per rotation is then the key product, the mod-down and the
  // permutation -- `MultKey`'s own words, since `MultKeyNoModDown` takes the
  // mod-up as an argument and the b-part rides its pseudo mod-up exactly as
  // it does inside `MultKey`.
  const Parameter<word> &param = boot_->param_;
  const NPInfo np = a.GetNP();
  const int level = param.NPToLevel(np);
  const int degree = param.degree_;
  const int num_q = np.GetNumQ();
  const int num_aux = param.alpha_;
  const int prime_offset = param.GetMaxNumTer() - np.num_ter_;
  const int beta = DivCeil(num_q + prime_offset, num_aux);
  AssertTrue(level >= 1 && np.num_aux_ == 0,
             "CiBertBaseLayer::RotateMany: a canonical ciphertext above "
             "level 0");
  const auto &ms = boot_->GetModSwitchHandler(level, num_aux);

  std::vector<DeviceVector<word>> modup(beta);
  std::vector<DvView<word>> modup_view;
  for (int i = 0; i < beta; i++) {
    modup[i].resize(static_cast<size_t>(num_q + num_aux) * degree);
    modup_view.push_back(modup[i].View(num_aux * degree));
  }
  ms.ModUp(modup_view, a.AxConstView());
  DeviceVector<word> bxp(static_cast<size_t>(num_q) * degree);
  {
    DvView<word> bxp_view = bxp.View();
    DvConstView<word> p_prod_view(boot_->p_prod_.data() + prime_offset, num_q);
    ms.PseudoModUp(bxp_view, a.BxConstView(), p_prod_view);
  }

  for (int i = 0; i < n; i++) {
    if (idx[i] == 0) {
      boot_->Copy(res[i], a);
      continue;
    }
    Ct accum;
    boot_->MultKeyNoModDown(accum, modup, a, evk.GetRotationKey(idx[i]));
    {  // the b-part, which the hoisted entry point leaves out
      DvView<word> acc_bx(accum.bx_.data(), static_cast<size_t>(num_q) * degree, 0);
      std::vector<DvView<word>> dst = {acc_bx};
      std::vector<DvConstView<word>> lhs = {
          DvConstView<word>(accum.bx_.data(), static_cast<size_t>(num_q) * degree, 0)};
      std::vector<DvConstView<word>> rhs = {bxp.ConstView()};
      boot_->elem_handler_.Add(dst, np, lhs, rhs);
    }
    Ct down;
    down.RemoveRx();
    down.ModifyNP(np);
    down.SetScale(a.GetScale());
    down.SetNumSlots(a.GetNumSlots());
    auto bx_view = down.BxView();
    auto ax_view = down.AxView();
    ms.ModDown(bx_view, accum.BxConstView());
    ms.ModDown(ax_view, accum.AxConstView());
    accum = Ct();
    boot_->Permute(res[i], down, idx[i]);
    stages_.rotations++;
  }
}

template <typename word>
void CiBertBaseLayer<word>::EvalMany(std::vector<Ct> &cts,
                                     const EvalPoly<word> &poly,
                                     const Evk &mult_key) {
  const int n = static_cast<int>(cts.size());
  const int group = (cfg_.poly_batch > 1) ? std::min(cfg_.poly_batch, n) : 1;
  if (group <= 1) {
    for (int i = 0; i < n; i++) {
      Ct out;
      poly.Evaluate(boot_, out, cts[i], mult_key);
      cts[i] = std::move(out);
    }
    return;
  }
  for (int c0 = 0; c0 < n; c0 += group) {
    const int g = std::min(n - c0, group);
    CtBatch<word> in;
    const NPInfo np = cts[c0].GetNP();
    in.Allocate(np, g, false);
    in.scale_ = cts[c0].GetScale();
    in.num_slots_ = cts[c0].GetNumSlots();
    const size_t bytes = in.PolyWords() * sizeof(word);
    for (int b = 0; b < g; b++) {
      AssertTrue(cts[c0 + b].GetNP() == np && !cts[c0 + b].HasRx(),
                 "CiBertBaseLayer::EvalMany: the group is not at one level");
      AssertTrue(std::abs(cts[c0 + b].GetScale() - in.scale_) <= 1e-9 * in.scale_,
                 "CiBertBaseLayer::EvalMany: the group is not at one scale");
      cudaMemcpyAsync(in.CtData(b), cts[c0 + b].bx_.data(), bytes,
                      cudaMemcpyDeviceToDevice, cudaStreamLegacy);
      cudaMemcpyAsync(in.CtData(b) + in.PolyWords(), cts[c0 + b].ax_.data(),
                      bytes, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    }
    CtBatch<word> out;
    poly.EvaluateBatch(boot_, out, in, mult_key);
    in = CtBatch<word>();
    const size_t out_bytes = out.PolyWords() * sizeof(word);
    for (int b = 0; b < g; b++) {
      Ct &dst = cts[c0 + b];
      dst.RemoveRx();
      dst.ModifyNP(out.np_);
      dst.SetScale(out.scale_);
      dst.SetNumSlots(out.num_slots_);
      cudaMemcpyAsync(dst.bx_.data(), out.CtData(b), out_bytes,
                      cudaMemcpyDeviceToDevice, cudaStreamLegacy);
      cudaMemcpyAsync(dst.ax_.data(), out.CtData(b) + out.PolyWords(),
                      out_bytes, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    }
  }
}

template <typename word>
void CiBertBaseLayer<word>::Tap(const std::string &name, const Ct &ct,
                                double factor) const {
  if (!probe_) return;
  std::vector<Ct> one(1);
  boot_->Copy(one[0], ct);
  probe_(name, one, factor);
}

template <typename word>
void CiBertBaseLayer<word>::BootMany(std::vector<Ct> &cts,
                                     const EvkMap<word> &evk) {
  if (!boot_->IsBootPrepared(layout_.num_slots)) {
    boot_->PrepareEvalSpecialFFT(layout_.num_slots);
  }
  const int n = static_cast<int>(cts.size());
  const int group = std::max(1, cfg_.boot_group);
  for (int c0 = 0; c0 < n; c0 += group) {
    const int g = std::min(n - c0, group);
    if (g == 1) {
      Ct up;
      boot_->Boot(up, cts[c0], evk);
      cts[c0] = std::move(up);
    } else {
      std::vector<const Ct *> in(g);
      for (int j = 0; j < g; j++) in[j] = &cts[c0 + j];
      std::vector<Ct> up;
      boot_->BootBatch(up, in, evk);
      for (int j = 0; j < g; j++) cts[c0 + j] = std::move(up[j]);
    }
  }
  stages_.wide_boots += n;
}

template <typename word>
void CiBertBaseLayer<word>::Lift(Stream &s, int need, const EvkMap<word> &evk) {
  AssertTrue(!s.cts.empty(), "CiBertBaseLayer::Lift: empty stream");
  if (Level(s.cts[0]) >= need) return;
  auto t0 = Clock::now();
  BootMany(s.cts, evk);
  stages_.boot += Since(t0);
}

template <typename word>
double CiBertBaseLayer<word>::NarrowLift(Ct &ct, int need, double absmax,
                                         const EvkMap<word> &evk) {
  if (Level(ct) >= need) return 1.0;
  AssertTrue(Level(ct) >= 1,
             "CiBertBaseLayer::NarrowLift: a level-0 accumulator cannot be "
             "scaled into its bootstrap");
  const double c = cfg_.ride / absmax;
  Ct s;
  MultScalar(s, ct, c);
  if (!boot_->IsBootPrepared(layout_.num_slots)) {
    boot_->PrepareEvalSpecialFFT(layout_.num_slots);
  }
  auto t0 = Clock::now();
  Ct up;
  boot_->Boot(up, s, evk);
  ct = std::move(up);
  stages_.narrow_boots++;
  stages_.boot += Since(t0);
  return 1.0 / c;
}

// ---------------------------------------------------------------- prepare
template <typename word>
void CiBertBaseLayer<word>::Prepare(const Weights &w, const Calibration &c) {
  const Shape &s = cfg_.shape;
  const int H = s.model, I = s.hidden, T = s.tokens, D = s.head_dim;
  AssertTrue(c.niter >= 1 && static_cast<int>(c.inv.size()) == c.niter,
             "CiBertBaseLayer::Prepare: one inverse-square-root window per "
             "Cho pass");
  AssertTrue(static_cast<int>(c.shift.size()) == s.heads,
             "CiBertBaseLayer::Prepare: shift is [heads][tokens]");
  for (const auto &row : c.shift) {
    AssertTrue(static_cast<int>(row.size()) == T,
               "CiBertBaseLayer::Prepare: shift is [heads][tokens]");
  }
  AssertTrue(w.wq && w.wk && w.wv && w.wo && w.wint && w.wout,
             "CiBertBaseLayer::Prepare: six tensors");
  AssertTrue(static_cast<int>(w.bq.size()) == H && w.bk.size() == w.bq.size() &&
                 w.bv.size() == w.bq.size() && w.bo.size() == w.bq.size() &&
                 static_cast<int>(w.bint.size()) == I &&
                 static_cast<int>(w.bout.size()) == H &&
                 static_cast<int>(w.attn_norm.size()) == H &&
                 static_cast<int>(w.attn_norm_bias.size()) == H &&
                 static_cast<int>(w.ffn_norm.size()) == H &&
                 static_cast<int>(w.ffn_norm_bias.size()) == H,
             "CiBertBaseLayer::Prepare: the vectors' widths");
  cal_ = c;
  w_ = w;
  const int top = TopLevel();
  const Parameter<word> &param = boot_->param_;

  // ---- the public live counts and the fold's slots -----------------------
  {
    const int B = layout_.num_instances;
    live_.assign(B, static_cast<double>(T));
    if (has_mask_) {
      for (int b = 0; b < B; b++) {
        int n = 0;
        for (int t = 0; t < T; t++) {
          n += mask_valid_[static_cast<size_t>(b) * T + t] ? 1 : 0;
        }
        AssertTrue(n > 0, "CiBertBaseLayer::Prepare: an instance has no real "
                          "token (the batch must be full of real prompts)");
        live_[b] = n;
      }
    }
    const int nf = c.niter * s.heads;
    fold_a_.clear(); fold_b_.clear();
    fold_a_.resize(nf); fold_b_.resize(nf);
    fold_a_lv_.assign(nf, -1); fold_b_lv_.assign(nf, -1);
    fold_a_sc_.assign(nf, 0.0); fold_b_sc_.assign(nf, 0.0);
    AssertTrue(c.est.empty() || static_cast<int>(c.est.size()) == c.niter,
               "CiBertBaseLayer::Prepare: one fold estimate per Cho pass");
  }

  // ---- the level plan (see the header) ----------------------------------
  exp_levels_ = Levels(c.exp.degree);
  const int mask_levels = has_mask_ ? 1 : 0;
  l_s_ = exp_levels_ + (c.niter == 1 ? 6 : 2) + mask_levels;
  {
    int y = l_s_ - exp_levels_ - mask_levels;
    for (int j = 0; j < c.niter; j++) {
      const int lv = Levels(c.inv[j].degree);
      const bool fold = Folds(j);
      // the tensor square, then the scaling that precedes the narrow boot --
      // a plaintext (the fold) or a constant, one level either way when a
      // boot happens; the fold pays it even when none does
      const int sq = y - 1;
      const int s_in = fold ? sq - 1 : sq;
      const int r_in = (s_in < lv + 2) ? top : s_in;
      int r = r_in - 1 - lv;          // the affine, then the polynomial
      // the constant on r: the fold's `1/sqrt(est)` always, the between-pass
      // sqrt(ride) only when another pass follows
      if (fold || j + 1 < c.niter) r -= 1;
      const int p = std::min(y, r) - 2;
      y = (j + 1 < c.niter) ? top : p;
    }
    l_p_ = y;
  }
  AssertTrue(l_p_ >= 4, "CiBertBaseLayer::Prepare: P must leave P V, O, the "
                        "residual and LN1 a level each (l_p " +
                            std::to_string(l_p_) + ")");
  l_qk_in_ = l_s_ + 2;
  l_v_in_ = l_p_ + 1;
  l_o_in_ = l_p_ - 1;
  AssertTrue(l_qk_in_ <= top, "CiBertBaseLayer::Prepare: the exp degree "
                              "does not fit under the boot's landing");

  // ---- carries and folds ------------------------------------------------
  const double c_in = cfg_.ride / c.in_absmax;
  const double a_e = 0.5 * (c.exp.hi - c.exp.lo), b_e = 0.5 * (c.exp.hi + c.exp.lo);
  q_fold_ = 1.0 / (std::sqrt(static_cast<double>(D)) * std::ldexp(1.0, c.niter) * a_e);
  const double c_h = cfg_.ride / c.ln1.out_absmax;
  c_in_ = c_in;
  c_h_ = c_h;

  // ---- the feed-forward's tiles: one GELU each -------------------------
  AssertTrue(static_cast<int>(c.gelu.size()) == ffn_tiles_,
             "CiBertBaseLayer::Prepare: one GELU fit per feed-forward tile "
             "(the calibration's tile size and Config::rows_per_tile are one "
             "decision; got " + std::to_string(c.gelu.size()) + " fits for " +
             std::to_string(ffn_tiles_) + " tiles)");
  AssertTrue(static_cast<int>(c.gelu_perm.size()) == I,
             "CiBertBaseLayer::Prepare: the GELU permutation is [hidden]");
  // The tiles are compiled BEFORE the down projection is prepared, because
  // their LANDING is that operand's level, and they all have to land
  // together: `Levels(degree)` is what a fit spends only when the
  // interpolant's top coefficient survives `EvalPoly`'s own degree
  // reduction, and the tiles' degrees differ by design.
  gelu_poly_.clear();
  gelu_poly_.resize(ffn_tiles_);
  gelu_landing_ = top;
  std::vector<int> landing(ffn_tiles_);
  for (int t = 0; t < ffn_tiles_; t++) {
    const double a = 0.5 * (c.gelu[t].hi - c.gelu[t].lo);
    const double b = 0.5 * (c.gelu[t].hi + c.gelu[t].lo);
    gelu_poly_[t] = Compile(
        [a, b](double x) {
          const double u = a * x + b;
          return 0.5 * u * (1.0 + std::erf(u / std::sqrt(2.0)));
        },
        c.gelu[t].degree, top - 1, param.GetScale(top - 1), &landing[t]);
    gelu_landing_ = std::min(gelu_landing_, landing[t]);
  }
  AssertTrue(gelu_landing_ >= 3,
             "CiBertBaseLayer::Prepare: the GELU degree does not leave the "
             "down projection, the residual and LN2 their levels (landing " +
                 std::to_string(gelu_landing_) + ")");
  FoldWeights(w, c);

  proj_->Prepare("q", fq_.data(), H, H, l_qk_in_, q_fold_ / c_in);
  proj_->Prepare("k", fk_.data(), H, H, l_qk_in_, 1.0 / c_in);
  proj_->Prepare("v", fv_.data(), H, H, l_v_in_, 1.0 / c_in);
  proj_->Prepare("o", fo_.data(), H, H, l_o_in_, c_in);
  // The feed-forward, tiled over the hidden axis: ONE up operand whose tiles
  // are `rows_per_tile` hidden channels (its columns already permuted and
  // carrying each tile's `1 / a`), and one DOWN operand per tile, reading
  // that tile's rows of the permuted `wout`. The products are accumulated --
  // the same sum, in tile order.
  proj_->Prepare("up", fup_.data(), H, I, top, 1.0 / c_h);
  AssertTrue(proj_->NumTiles("up") == ffn_tiles_,
             "CiBertBaseLayer::Prepare: the projection's tiling is not the "
             "calibration's");
  for (int t = 0; t < ffn_tiles_; t++) {
    const int j0 = proj_->TileStart("up", t);
    const int rows = std::min(I - j0, std::min(cfg_.rows_per_tile, I));
    proj_->Prepare(DownName(t), fdown_.data() + static_cast<size_t>(j0) * H,
                   rows, H, gelu_landing_, c_h);
  }
  bq_.resize(H); bk_.resize(H); bv_.resize(H); bo_.resize(H);
  bint_.resize(I); bout_.resize(H);
  for (int i = 0; i < H; i++) {
    // q / k / v write head channels, which carry no suppression; o and down
    // write the streams back, so their biases carry the same `1 / chan`
    // their columns do
    const double s_in = c.in_chan.empty() ? 1.0 : c.in_chan[i];
    const double s_h = c.ln1_chan.empty() ? 1.0 : c.ln1_chan[i];
    bq_[i] = q_fold_ * w.bq[i];
    bk_[i] = w.bk[i];
    bv_[i] = w.bv[i];
    bo_[i] = c_in * w.bo[i] / s_in;
    bout_[i] = c_h * w.bout[i] / s_h;
  }
  {  // the bias in the permuted order, each tile's affine applied to its own
    const int rows = std::min(cfg_.rows_per_tile, I);
    for (int j = 0; j < I; j++) {
      const int t = std::min(j / rows, ffn_tiles_ - 1);
      const double a = 0.5 * (c.gelu[t].hi - c.gelu[t].lo);
      const double b = 0.5 * (c.gelu[t].hi + c.gelu[t].lo);
      bint_[j] = (w.bint[c.gelu_perm[j]] - b) / a;
    }
  }

  // ---- the softmax's shift, folded with the exp's affine ----------------
  //     t = (S_model - shift) / (2^k a) - b/a ; the multiply is in W_Q
  shift_pt_.clear();
  shift_pt_.resize(s.heads);
  {
    std::vector<Complex> msg;
    std::vector<double> row(T);
    for (int h = 0; h < s.heads; h++) {
      for (int t = 0; t < T; t++) {
        row[t] = -c.shift[h][t] / (std::ldexp(1.0, c.niter) * a_e) - b_e / a_e;
      }
      layout_.PackPerToken(msg, row);
      boot_->gpu_encoder_.Encode(shift_pt_[h], l_s_, param.GetScale(l_s_), msg);
    }
  }
  exp_poly_ = Compile([a_e, b_e](double t) { return std::exp(a_e * t + b_e); },
                      c.exp.degree, l_s_, param.GetScale(l_s_));
  prepared_ = true;
  if (cfg_.verbose) {
    std::cout << "  [bert-base] plan: top " << top << ", S at " << l_s_
              << " (exp deg " << c.exp.degree << " = " << exp_levels_
              << " lv), P at " << l_p_ << ", q/k in " << l_qk_in_ << ", v in "
              << l_v_in_ << ", o in " << l_o_in_ << ", FFN " << ffn_tiles_
              << " x " << (I / std::max(1, ffn_tiles_)) << " GELU deg";
    for (int t = 0; t < ffn_tiles_; t++) {
      std::cout << (t ? "/" : " ") << c.gelu[t].degree;
    }
    std::cout << " (|u| <= " << c.gelu[0].hi << ".." << c.gelu[ffn_tiles_ - 1].hi
              << ") -> " << gelu_landing_
              << ", Cho k " << c.niter << ", BSGS " << baby_ << " x " << giant_
              << ", carries in " << c_in << " h " << c_h
              << (has_mask_ ? ", masked" : "");
    std::cout << "; fold ";
    if (c.est.empty()) {
      std::cout << "none";
    } else if (!cfg_.fold) {
      std::cout << "OFF (the calibration has one)";
    } else {
      for (int j = 0; j < c.niter; j++) {
        std::cout << (j ? "," : "") << (Folds(j) ? "on" : "-");
      }
      std::cout << " (live^";
      for (int j = 0; j < c.niter; j++) {
        std::cout << (j ? "," : "")
                  << (j < static_cast<int>(c.est_live_pow.size())
                          ? c.est_live_pow[j]
                          : 0);
      }
      std::cout << ")";
    }
    std::cout << ", hoist " << (cfg_.hoist ? "on" : "off") << ", poly batch "
              << cfg_.poly_batch << ", operands "
              << (proj_->Bytes() >> 20) << " MiB" << std::endl;
  }
}

namespace {
//! A host vector of doubles as device floats, or an empty view for null.
template <typename T, typename U>
void ToDev(DeviceVector<T> &d, const std::vector<U> *src) {
  if (src == nullptr || src->empty()) return;
  HostVector<T> h(src->size());
  for (size_t i = 0; i < src->size(); i++) h[i] = static_cast<T>((*src)[i]);
  d.resize(static_cast<int>(h.size()));
  CopyHostToDevice(d, h);
}
}  // namespace

template <typename word>
void CiBertBaseLayer<word>::FoldCols(DeviceVector<float> &dst, const float *src,
                                     int in, int out,
                                     const std::vector<int> *cperm,
                                     const std::vector<double> *rgain,
                                     const std::vector<double> *cgain) const {
  DeviceVector<int> perm;
  DeviceVector<float> rg, cg;
  ToDev(perm, cperm);
  ToDev(rg, rgain);
  ToDev(cg, cgain);
  const size_t n = static_cast<size_t>(in) * out;
  AssertTrue(n < (static_cast<size_t>(1) << 31),
             "CiBertBaseLayer::FoldCols: the tensor does not fit an int index");
  dst.resize(static_cast<int>(n));
  constexpr int kBlock = 256;
  const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  kernel::FoldColsKernel<<<grid, kBlock>>>(
      dst.data(), src, perm.size() ? perm.data() : nullptr,
      rg.size() ? rg.data() : nullptr, cg.size() ? cg.data() : nullptr, in, out);
}

template <typename word>
void CiBertBaseLayer<word>::FoldRows(DeviceVector<float> &dst, const float *src,
                                     int rows, int cols,
                                     const std::vector<int> &rperm,
                                     const std::vector<double> *cgain) const {
  DeviceVector<int> perm;
  DeviceVector<float> cg;
  ToDev(perm, &rperm);
  ToDev(cg, cgain);
  const size_t n = static_cast<size_t>(rows) * cols;
  AssertTrue(n < (static_cast<size_t>(1) << 31),
             "CiBertBaseLayer::FoldRows: the tensor does not fit an int index");
  dst.resize(static_cast<int>(n));
  constexpr int kBlock = 256;
  const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  kernel::FoldRowsKernel<<<grid, kBlock>>>(dst.data(), src, perm.data(),
                                           cg.size() ? cg.data() : nullptr,
                                           rows, cols);
}

template <typename word>
void CiBertBaseLayer<word>::FoldWeights(const Weights &w,
                                        const Calibration &c) {
  const int H = cfg_.shape.model, I = cfg_.shape.hidden;
  const int rows = std::min(cfg_.rows_per_tile, I);
  // The three streams' public per-channel factors. `d_in` is what enters the
  // layer and what the attention's residual writes back; `d_h` is LN1's
  // output and what the feed-forward's residual writes back.
  const std::vector<double> &d_in = c.in_chan;
  const std::vector<double> &d_h = c.ln1_chan;
  AssertTrue(d_in.empty() || static_cast<int>(d_in.size()) == H,
             "CiBertBaseLayer: in_chan is [model]");
  AssertTrue(d_h.empty() || static_cast<int>(d_h.size()) == H,
             "CiBertBaseLayer: ln1_chan is [model]");
  std::vector<double> inv_in, inv_h;
  auto reciprocal = [](const std::vector<double> &v, std::vector<double> &r) {
    r.resize(v.size());
    for (size_t i = 0; i < v.size(); i++) {
      AssertTrue(v[i] > 0.0, "CiBertBaseLayer: a suppression is not positive");
      r[i] = 1.0 / v[i];
    }
  };
  reciprocal(d_in, inv_in);
  reciprocal(d_h, inv_h);
  const std::vector<double> *rin = d_in.empty() ? nullptr : &d_in;
  const std::vector<double> *cin = inv_in.empty() ? nullptr : &inv_in;
  const std::vector<double> *rh = d_h.empty() ? nullptr : &d_h;
  const std::vector<double> *ch = inv_h.empty() ? nullptr : &inv_h;
  // q / k / v read the stream; o writes it back
  FoldCols(fq_, w.wq, H, H, nullptr, rin, nullptr);
  FoldCols(fk_, w.wk, H, H, nullptr, rin, nullptr);
  FoldCols(fv_, w.wv, H, H, nullptr, rin, nullptr);
  FoldCols(fo_, w.wo, H, H, nullptr, nullptr, cin);
  // the feed-forward: `up` reads h and its columns are the permuted hidden
  // axis, each carrying its GELU tile's `1 / a`; `down` reads the permuted
  // hidden axis by rows and writes h's stream back
  std::vector<double> ga(I);
  for (int j = 0; j < I; j++) {
    const int t = std::min(j / rows, ffn_tiles_ - 1);
    const double a = 0.5 * (c.gelu[t].hi - c.gelu[t].lo);
    AssertTrue(a > 0.0, "CiBertBaseLayer: a GELU tile has an empty interval");
    AssertTrue(c.gelu_perm[j] >= 0 && c.gelu_perm[j] < I,
               "CiBertBaseLayer: the GELU permutation is out of range");
    ga[j] = 1.0 / a;
  }
  FoldCols(fup_, w.wint, H, I, &c.gelu_perm, rh, &ga);
  FoldRows(fdown_, w.wout, I, H, c.gelu_perm, ch);
}

template <typename word>
void CiBertBaseLayer<word>::SetMask(const std::vector<uint8_t> &valid) {
  const size_t n = static_cast<size_t>(layout_.num_instances) * cfg_.shape.tokens;
  AssertTrue(valid.empty() || valid.size() == n,
             "CiBertBaseLayer::SetMask: valid is [instances][tokens]");
  mask_valid_ = valid;
  has_mask_ = !valid.empty();
  mask_pt_.clear();
  mask_level_ = -1;
}

template <typename word>
void CiBertBaseLayer<word>::MaskPlaintexts(const Ct &like) {
  const int T = cfg_.shape.tokens, B = layout_.num_instances;
  const int level = Level(like);
  const double scale = like.GetScale();
  if (mask_level_ == level && std::abs(mask_scale_ - scale) <= 1e-9 * scale &&
      static_cast<int>(mask_pt_.size()) == T) {
    return;
  }
  mask_pt_.clear();
  mask_pt_.resize(T);
  std::vector<double> values(static_cast<size_t>(B) * T);
  std::vector<Complex> msg;
  for (int d = 0; d < T; d++) {
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        values[static_cast<size_t>(b) * T + t] =
            mask_valid_[static_cast<size_t>(b) * T + (t + d) % T] ? 1.0 : 0.0;
      }
    }
    layout_.Pack(msg, values);
    boot_->gpu_encoder_.Encode(mask_pt_[d], level, scale, msg);
  }
  mask_level_ = level;
  mask_scale_ = scale;
}

// ------------------------------------------------------------------ head
template <typename word>
void CiBertBaseLayer<word>::PrepareHead(const HeadWeights &w,
                                        const HeadCalibration &c,
                                        double in_absmax,
                                        const std::vector<double> &in_chan) {
  const int H = cfg_.shape.model;
  AssertTrue(w.pool_w != nullptr && w.cls_w != nullptr,
             "CiBertBaseLayer::PrepareHead: two tensors");
  AssertTrue(static_cast<int>(w.pool_b.size()) == H && !w.cls_b.empty(),
             "CiBertBaseLayer::PrepareHead: the biases' widths");
  hw_ = w;
  hc_ = c;
  classes_ = static_cast<int>(w.cls_b.size());
  c_head_ = cfg_.ride / in_absmax;
  const int top = TopLevel();
  const Parameter<word> &param = boot_->param_;
  // t = (z W + b - b_t) / a_t rides the pooler's weight; tanh(a_t t + b_t)
  const double a_t = 0.5 * (c.tanh.hi - c.tanh.lo), b_t = 0.5 * (c.tanh.hi + c.tanh.lo);
  // the pooler reads the last layer's output stream, so its rows carry that
  // stream's public per-channel suppression
  FoldCols(fpool_, w.pool_w, H, H, nullptr,
           in_chan.empty() ? nullptr : &in_chan, nullptr);
  head_chan_ = in_chan;
  proj_->Prepare("pool", fpool_.data(), H, H, top, 1.0 / (a_t * c_head_));
  hpb_.resize(H);
  for (int i = 0; i < H; i++) hpb_[i] = (w.pool_b[i] - b_t) / a_t;
  int landing = 0;
  tanh_poly_ = Compile([a_t, b_t](double t) { return std::tanh(a_t * t + b_t); },
                       c.tanh.degree, top - 1, param.GetScale(top - 1), &landing);
  proj_->Prepare("cls", w.cls_w, H, classes_, landing, 1.0);
  hcb_ = w.cls_b;
  if (cfg_.verbose) {
    std::cout << "  [bert-base] head: pooler at " << top << " -> " << top - 1
              << ", tanh deg " << c.tanh.degree << " on [" << c.tanh.lo << ", "
              << c.tanh.hi << "] -> " << landing << ", classifier -> "
              << landing - 1 << ", " << classes_ << " classes" << std::endl;
  }
}

template <typename word>
void CiBertBaseLayer<word>::Head(std::vector<Ct> &logits, Stream &z,
                                 const EvkMap<word> &evk) {
  NvtxScope _nv("bert-base: head");
  const int H = cfg_.shape.model;
  const auto &mult_key = evk.GetMultiplicationKey();
  AssertTrue(tanh_poly_ != nullptr, "CiBertBaseLayer::Head: PrepareHead first");
  AssertTrue(static_cast<int>(z.cts.size()) == H, "Head: model channels");
  AssertTrue(std::abs(z.carry - c_head_) <= 1e-9 * c_head_,
             "CiBertBaseLayer::Head: the stream's carry is not the head's "
             "(ride / the last LN2's out_absmax)");
  AssertTrue(z.chan == head_chan_,
             "CiBertBaseLayer::Head: the stream's suppression is not the one "
             "PrepareHead folded into the pooler");
  auto t0 = Clock::now();
  Lift(z, TopLevel(), evk);
  std::vector<Ct> u;
  proj_->Project(u, z.cts, "pool");
  for (int i = 0; i < H; i++) AddScalar(u[i], u[i], hpb_[i]);
  Tap("t_tanh", u, 1.0);
  EvalMany(u, *tanh_poly_, mult_key);
  Tap("pooled", u, 1.0);
  proj_->Project(logits, u, "cls");
  for (int c = 0; c < classes_; c++) AddScalar(logits[c], logits[c], hcb_[c]);
  Tap("logits", logits, 1.0);
  stages_.o += Since(t0);
}

// ---------------------------------------------------- diagonal products
template <typename word>
void CiBertBaseLayer<word>::DiagonalProducts(std::vector<Ct> &res,
                                             const std::vector<Ct> &lhs,
                                             const std::vector<Ct> &rhs,
                                             const EvkMap<word> &evk) {
  NvtxScope _nv("bert-base: Q K^T");
  const int n = static_cast<int>(lhs.size());
  const int B = layout_.num_instances, T = cfg_.shape.tokens;
  const auto &mult_key = evk.GetMultiplicationKey();
  AssertTrue(static_cast<int>(rhs.size()) == n, "DiagonalProducts: widths");
  // baby copies of the rhs: rb[b][i] = rot(rhs[i], b B), and giant copies of
  // the lhs: lg[g][i] = rot(lhs[i], -g baby B). Each source is rotated by
  // several distances, so each costs ONE decomposition (`RotateMany`).
  std::vector<int> baby_dists(baby_), giant_dists(giant_);
  for (int b = 0; b < baby_; b++) baby_dists[b] = b * B;
  for (int g = 0; g < giant_; g++) giant_dists[g] = -g * baby_ * B;
  std::vector<std::vector<Ct>> rb(baby_), lg(giant_);
  for (int b = 0; b < baby_; b++) rb[b].resize(n);
  for (int g = 0; g < giant_; g++) lg[g].resize(n);
  for (int i = 0; i < n; i++) {
    std::vector<Ct> one;
    RotateMany(one, rhs[i], baby_dists, evk);
    for (int b = 0; b < baby_; b++) rb[b][i] = std::move(one[b]);
    RotateMany(one, lhs[i], giant_dists, evk);
    for (int g = 0; g < giant_; g++) lg[g][i] = std::move(one[g]);
  }
  res.clear();
  res.resize(T);
  for (int g = 0; g < giant_; g++) {
    const int G = g * baby_ * B;
    for (int b = 0; b < baby_; b++) {
      Ct acc;
      for (int i = 0; i < n; i++) {
        Ct t;
        boot_->Mult(t, lg[g][i], rb[b][i]);
        if (i == 0) {
          acc = std::move(t);
        } else {
          boot_->Add(acc, acc, t);
        }
      }
      Ct rel;
      boot_->RelinearizeRescale(rel, acc, mult_key);
      stages_.relins++;
      const int d = g * baby_ + b;
      if (g == 0) {
        res[d] = std::move(rel);
      } else {
        Rotate(res[d], rel, G, evk);
      }
    }
  }
}

template <typename word>
void CiBertBaseLayer<word>::DiagonalContract(std::vector<Ct> &res,
                                             const std::vector<Ct> &lhs,
                                             const std::vector<Ct> &rhs,
                                             const EvkMap<word> &evk) {
  NvtxScope _nv("bert-base: P V");
  const int n = static_cast<int>(rhs.size());
  const int B = layout_.num_instances, T = cfg_.shape.tokens;
  const auto &mult_key = evk.GetMultiplicationKey();
  AssertTrue(static_cast<int>(lhs.size()) == T, "DiagonalContract: T diagonals");
  // the rhs again takes baby distances off one decomposition each; the lhs
  // here is a DIFFERENT ciphertext per (giant, baby), so its one rotation
  // has nothing to share
  std::vector<int> baby_dists(baby_);
  for (int b = 0; b < baby_; b++) baby_dists[b] = b * B;
  std::vector<std::vector<Ct>> rb(baby_);
  for (int b = 0; b < baby_; b++) rb[b].resize(n);
  for (int i = 0; i < n; i++) {
    std::vector<Ct> one;
    RotateMany(one, rhs[i], baby_dists, evk);
    for (int b = 0; b < baby_; b++) rb[b][i] = std::move(one[b]);
  }
  res.clear();
  res.resize(n);
  for (int g = 0; g < giant_; g++) {
    const int G = g * baby_ * B;
    std::vector<Ct> lg(baby_);
    for (int b = 0; b < baby_; b++) Rotate(lg[b], lhs[g * baby_ + b], -G, evk);
    for (int i = 0; i < n; i++) {
      Ct acc;
      for (int b = 0; b < baby_; b++) {
        Ct t;
        boot_->Mult(t, lg[b], rb[b][i]);
        if (b == 0) {
          acc = std::move(t);
        } else {
          boot_->Add(acc, acc, t);
        }
      }
      Ct rel;
      boot_->RelinearizeRescale(rel, acc, mult_key);
      stages_.relins++;
      if (g == 0) {
        res[i] = std::move(rel);
      } else {
        Ct part;
        Rotate(part, rel, G, evk);
        boot_->Add(res[i], res[i], part);
      }
    }
  }
}

// ------------------------------------------------------------- softmax
template <typename word>
void CiBertBaseLayer<word>::SoftMax(std::vector<Ct> &P, std::vector<Ct> &S,
                                    int head, const EvkMap<word> &evk) {
  NvtxScope _nv("bert-base: softmax");
  const int T = cfg_.shape.tokens, k = cal_.niter;
  const auto &mult_key = evk.GetMultiplicationKey();
  AssertTrue(Level(S[0]) == l_s_, "SoftMax: the scores are not at l_s");
  // y = exp((S - shift) / 2^k), the affine folded into W_Q and the shift
  const double a_e = 0.5 * (cal_.exp.hi - cal_.exp.lo);
  if (head == 0) Tap("s", S, 1.0 / (std::ldexp(1.0, k) * a_e));
  std::vector<Ct> y(T);
  for (int d = 0; d < T; d++) {
    boot_->Add(y[d], S[d], shift_pt_[head]);
    if (head == 0 && d == 0) Tap("t_exp", y[d], 1.0);
    S[d] = Ct();
  }
  EvalMany(y, *exp_poly_, mult_key);
  if (head == 0) Tap("y", y, 1.0);
  if (has_mask_) {
    // pads out of the row: y_d <- y_d (.) M_d, one wide level
    MaskPlaintexts(y[0]);
    for (int d = 0; d < T; d++) {
      Ct t;
      boot_->Mult(t, y[d], mask_pt_[d]);
      boot_->Rescale(y[d], t);
    }
    if (head == 0) Tap("y_masked", y, 1.0);
  }
  for (int j = 0; j < k; j++) {
    // sq = sum_d y_d^2: one relinearization
    Ct acc;
    for (int d = 0; d < T; d++) {
      Ct t;
      boot_->Mult(t, y[d], y[d]);
      if (d == 0) {
        acc = std::move(t);
      } else {
        boot_->Add(acc, acc, t);
      }
    }
    Ct sq;
    boot_->RelinearizeRescale(sq, acc, mult_key);
    stages_.relins++;
    acc = Ct();
    if (head == 0) Tap("sq" + std::to_string(j), sq, 1.0);
    // pass j's window, in the units the walk is in: after pass j - 1 the
    // main path carries ride (the sqrt(ride) on r), so sq carries ride^2
    const double unit = (j == 0) ? 1.0 : cfg_.ride * cfg_.ride;
    const int degree = cal_.inv[j].degree;
    const int need = Levels(degree) + 2;
    const bool fold = Folds(j);
    double kappa = 1.0, a = 0.0, b = 0.0;
    if (fold) {
      // 1/sqrt(sq) = (1/sqrt(est)) (1/sqrt(sq/est)): the polynomial sees the
      // RATIO, whose window is one row's population spread. The estimate
      // rides the scaling that precedes the boot, so it costs no level there.
      const double lo = cal_.inv[j].lo, hi = cal_.inv[j].hi;  // of the ratio
      Ct t;
      boot_->Mult(t, sq, FoldScale(j, head, sq, unit, hi));
      boot_->Rescale(sq, t);                        // = ride (sq/est) / hi
      if (Level(sq) < need) {
        if (!boot_->IsBootPrepared(layout_.num_slots)) {
          boot_->PrepareEvalSpecialFFT(layout_.num_slots);
        }
        auto tb = Clock::now();
        Ct up;
        boot_->Boot(up, sq, evk);
        sq = std::move(up);
        stages_.narrow_boots++;
        stages_.boot += Since(tb);
      }
      kappa = hi / cfg_.ride;
      a = 0.5 * (hi - lo);
      b = 0.5 * (hi + lo);
    } else {
      const double lo = cal_.inv[j].lo * unit, hi = cal_.inv[j].hi * unit;
      kappa = NarrowLift(sq, need, hi, evk);
      a = 0.5 * (hi - lo);
      b = 0.5 * (hi + lo);
    }
    Ct t;
    MultScalar(t, sq, kappa / a);
    AddScalar(t, t, -b / a);
    auto inv = Compile([a, b](double x) { return 1.0 / std::sqrt(a * x + b); },
                       degree, Level(t), t.GetScale());
    Ct r;
    inv->Evaluate(boot_, r, t, mult_key);
    if (fold) {
      // r <- r / sqrt(est unit), and the between-pass sqrt(ride) with it
      const double tail = (j + 1 < k) ? std::sqrt(cfg_.ride) : 1.0;
      Ct rs;
      boot_->Mult(rs, r, FoldUndo(j, head, r, unit, tail));
      boot_->Rescale(r, rs);
      // the tap is the same quantity either way: 1/sqrt(sq) in ct units,
      // times the sqrt(ride) a following pass carries
      if (head == 0) Tap("r" + std::to_string(j), r, tail);
    } else {
      if (head == 0) Tap("r" + std::to_string(j), r, 1.0);
      if (j + 1 < k) {
        Ct rs;
        MultScalar(rs, r, std::sqrt(cfg_.ride));
        r = std::move(rs);
      }
    }
    // y <- (y r)^2
    const int lw = std::min(Level(y[0]), Level(r));
    Ct r_dn;
    boot_->LevelDown(r_dn, r, lw);
    for (int d = 0; d < T; d++) {
      Ct yd, w, w2, sqd;
      boot_->LevelDown(yd, y[d], lw);
      boot_->Mult(w, yd, r_dn);
      boot_->RelinearizeRescale(w2, w, mult_key);
      boot_->Mult(sqd, w2, w2);
      boot_->RelinearizeRescale(y[d], sqd, mult_key);
      stages_.relins += 2;
    }
    if (j + 1 < k) {
      auto t0 = Clock::now();
      BootMany(y, evk);
      stages_.boot += Since(t0);
    }
  }
  P = std::move(y);
  if (head == 0) Tap("p", P, 1.0);
}

// ----------------------------------------------------------- attention
template <typename word>
void CiBertBaseLayer<word>::Attention(Stream &attn_out, Stream &x,
                                      const EvkMap<word> &evk) {
  NvtxScope _nv("bert-base: attention");
  const Shape &s = cfg_.shape;
  const int H = s.model, D = s.head_dim, NH = s.heads;
  AssertTrue(prepared_, "CiBertBaseLayer::Attention: Prepare first");
  AssertTrue(static_cast<int>(x.cts.size()) == H, "Attention: model channels");
  AssertTrue(std::abs(x.carry - c_in_) <= 1e-9 * c_in_,
             "CiBertBaseLayer::Attention: the stream's carry is not the "
             "calibration's (ride / in_absmax)");
  AssertTrue(Level(x.cts[0]) >= l_qk_in_, "Attention: the stream is below "
                                          "the q/k projection's level");
  AssertTrue(x.chan == cal_.in_chan,
             "CiBertBaseLayer::Attention: the stream's public suppression is "
             "not the calibration's in_chan (the projections carry it)");
  auto t0 = Clock::now();
  std::vector<Ct> q, k, v;
  {
    std::vector<Ct> xin(H);
    for (int c = 0; c < H; c++) boot_->LevelDown(xin[c], x.cts[c], l_qk_in_);
    proj_->Project(q, xin, "q");
    proj_->Project(k, xin, "k");
    for (int c = 0; c < H; c++) boot_->LevelDown(xin[c], x.cts[c], l_v_in_);
    proj_->Project(v, xin, "v");
  }
  // The residual reads `x` one level under the O projection's output, and
  // nothing else does: at BERT-Base's width holding it at the projections'
  // level for the whole head loop is gigabytes for nothing.
  for (int c = 0; c < H; c++) {
    Ct d;
    boot_->LevelDown(d, x.cts[c], l_o_in_ - 1);
    x.cts[c] = std::move(d);
  }
  for (int c = 0; c < H; c++) {
    AddScalar(q[c], q[c], bq_[c]);
    AddScalar(k[c], k[c], bk_[c]);
    AddScalar(v[c], v[c], bv_[c]);
  }
  stages_.qkv += Since(t0);
  Tap("q", q, q_fold_);
  Tap("k", k, 1.0);
  Tap("v", v, 1.0);

  std::vector<Ct> o_in(H);
  for (int h = 0; h < NH; h++) {
    std::vector<Ct> qh(D), kh(D), vh(D);
    for (int d = 0; d < D; d++) {
      qh[d] = std::move(q[h * D + d]);
      kh[d] = std::move(k[h * D + d]);
      vh[d] = std::move(v[h * D + d]);
    }
    t0 = Clock::now();
    std::vector<Ct> S;
    DiagonalProducts(S, qh, kh, evk);
    qh.clear();
    kh.clear();
    stages_.scores += Since(t0);
    t0 = Clock::now();
    std::vector<Ct> P;
    SoftMax(P, S, h, evk);
    stages_.softmax += Since(t0);
    t0 = Clock::now();
    const int lp = std::min(Level(P[0]), Level(vh[0]));
    for (auto &pd : P) {
      Ct t;
      boot_->LevelDown(t, pd, lp);
      pd = std::move(t);
    }
    for (auto &vd : vh) {
      Ct t;
      boot_->LevelDown(t, vd, lp);
      vd = std::move(t);
    }
    std::vector<Ct> oh;
    DiagonalContract(oh, P, vh, evk);
    for (int d = 0; d < D; d++) {
      boot_->LevelDown(o_in[h * D + d], oh[d], l_o_in_);
    }
    stages_.values += Since(t0);
  }
  Tap("o_in", o_in, 1.0);
  t0 = Clock::now();
  proj_->Project(attn_out.cts, o_in, "o");
  o_in.clear();
  for (int c = 0; c < H; c++) AddScalar(attn_out.cts[c], attn_out.cts[c], bo_[c]);
  attn_out.carry = c_in_;
  attn_out.chan = x.chan;    // the O projection's columns carry 1 / chan
  stages_.o += Since(t0);
  Tap("attn", attn_out.cts, c_in_);
}

// ----------------------------------------------------------- LayerNorm
template <typename word>
void CiBertBaseLayer<word>::LayerNorm(Stream &out, const Stream &pre,
                                      const std::vector<double> &g,
                                      const std::vector<double> &b,
                                      const typename Calibration::Norm &n,
                                      const std::vector<double> &out_chan,
                                      const EvkMap<word> &evk,
                                      const std::string &tag) {
  NvtxScope _nv("bert-base: LayerNorm");
  const int H = cfg_.shape.model;
  const double eps = cfg_.shape.eps;
  const auto &mult_key = evk.GetMultiplicationKey();
  AssertTrue(static_cast<int>(pre.cts.size()) == H, "LayerNorm: model channels");
  const int l = Level(pre.cts[0]);
  AssertTrue(l >= 2, "CiBertBaseLayer::LayerNorm(" + tag +
                         "): the stream needs two levels (variance, apply)");
  auto t0 = Clock::now();
  const double c = pre.carry;
  // centred' = H d_c x_c - sum_k d_k x_k  (= H c (x - mu)). The stream's
  // public suppression comes out HERE, and this is why its entries are
  // powers of two: an integer constant is encoded at scale 1, so both
  // multiplies are exact and neither costs a level.
  auto sup = [&pre](int i) {
    return pre.chan.empty() ? 1.0 : pre.chan[i];
  };
  Ct sum;
  for (int i = 0; i < H; i++) {
    Ct dx;
    MultInt(dx, pre.cts[i], sup(i));
    if (i == 0) {
      sum = std::move(dx);
    } else {
      boot_->Add(sum, sum, dx);
    }
  }
  std::vector<Ct> cen(H);
  for (int i = 0; i < H; i++) {
    Ct hx;
    MultInt(hx, pre.cts[i], static_cast<double>(H) * sup(i));
    boot_->Sub(cen[i], hx, sum);
  }
  sum = Ct();
  Tap(tag + "_cen", cen, static_cast<double>(H) * c);
  // V' = sum centred'^2 = H^3 c^2 var: one relinearization
  Ct acc;
  for (int i = 0; i < H; i++) {
    Ct t;
    boot_->Mult(t, cen[i], cen[i]);
    if (i == 0) {
      acc = std::move(t);
    } else {
      boot_->Add(acc, acc, t);
    }
  }
  Ct V;
  boot_->RelinearizeRescale(V, acc, mult_key);
  stages_.relins++;
  acc = Ct();
  const double kappa0 = 1.0 / (static_cast<double>(H) * H * H * c * c);
  Tap(tag + "_var", V, 1.0 / kappa0);
  const int degree = n.inv.degree;
  const double undo = NarrowLift(V, Levels(degree) + 2, n.inv.hi / kappa0, evk);
  const double a = 0.5 * (n.inv.hi - n.inv.lo), bb = 0.5 * (n.inv.hi + n.inv.lo);
  Ct t;
  MultScalar(t, V, kappa0 * undo / a);
  AddScalar(t, t, -bb / a);
  auto inv = Compile(
      [a, bb, eps](double x) { return 1.0 / std::sqrt(a * x + bb + eps); },
      degree, Level(t), t.GetScale());
  Ct r;
  inv->Evaluate(boot_, r, t, mult_key);
  Tap(tag + "_r", r, 1.0);
  // r = 1/sqrt(var + eps) in true units; the apply wants it a level above
  // the channels (its per-channel scalar copies cost one)
  double undo_r = 1.0;
  if (Level(r) < l + 1) undo_r = NarrowLift(r, l + 1, n.r_max, evk);
  const double c_out = cfg_.ride / n.out_absmax;
  const int lr = Level(r);
  out.cts.clear();
  out.cts.resize(H);
  const int lw = std::min(l, lr - 1);
  for (int i = 0; i < H; i++) {
    // out_i = c_out (g_i (x - mu) r + b_i) / d_i;  centred'_i = H c (x - mu).
    // The OUTPUT stream's suppression rides the per-channel constant that is
    // already on r, and its bias -- free, like the input's.
    const double d = out_chan.empty() ? 1.0 : out_chan[i];
    Ct rc, rd, cd, w;
    MultScalar(rc, r, c_out * g[i] * undo_r / (static_cast<double>(H) * c * d));
    boot_->LevelDown(rd, rc, lw);
    boot_->LevelDown(cd, cen[i], lw);
    cen[i] = Ct();
    boot_->Mult(w, cd, rd);
    boot_->RelinearizeRescale(out.cts[i], w, mult_key);
    AddScalar(out.cts[i], out.cts[i], c_out * b[i] / d);
    stages_.relins++;
  }
  out.carry = c_out;
  out.chan = out_chan;
  stages_.ln += Since(t0);
  Tap(tag + "_out", out.cts, c_out);
}

// --------------------------------------------------------- feed-forward
template <typename word>
void CiBertBaseLayer<word>::FeedForward(Stream &out, const Stream &h,
                                        const EvkMap<word> &evk) {
  NvtxScope _nv("bert-base: feed-forward");
  const int H = cfg_.shape.model;
  const auto &mult_key = evk.GetMultiplicationKey();
  AssertTrue(static_cast<int>(h.cts.size()) == H, "FeedForward: model channels");
  AssertTrue(std::abs(h.carry - c_h_) <= 1e-9 * c_h_,
             "CiBertBaseLayer::FeedForward: h's carry is not ride / "
             "ln1.out_absmax");
  AssertTrue(Level(h.cts[0]) == TopLevel(),
             "CiBertBaseLayer::FeedForward: h must be booted (Lift) first");
  // ONE split of the normalised stream serves every tile of `up`; the
  // hidden channels then exist `rows_per_tile` at a time, and `down` is a
  // separate operand per tile whose products are accumulated.
  auto t0 = Clock::now();
  Source src;
  proj_->Split(src, h.cts, "up");
  stages_.ffn += Since(t0);
  out.cts.clear();
  for (int tile = 0; tile < ffn_tiles_; tile++) {
    t0 = Clock::now();
    std::vector<Ct> u;
    proj_->Project(u, src, "up", tile);
    const int j0 = proj_->TileStart("up", tile);
    const int rows = static_cast<int>(u.size());
    for (int j = 0; j < rows; j++) AddScalar(u[j], u[j], bint_[j0 + j]);
    stages_.ffn += Since(t0);
    if (tile == 0) Tap("t_gelu", u, 1.0);
    t0 = Clock::now();
    EvalMany(u, *gelu_poly_[tile], mult_key);
    // the tiles' degrees differ, so their fits land at different levels; the
    // down projection reads one level
    if (Level(u[0]) > gelu_landing_) {
      for (int j = 0; j < rows; j++) {
        Ct d;
        boot_->LevelDown(d, u[j], gelu_landing_);
        u[j] = std::move(d);
      }
    }
    stages_.gelu += Since(t0);
    if (tile == 0) Tap("g", u, 1.0);
    t0 = Clock::now();
    std::vector<Ct> part;
    proj_->Project(part, u, DownName(tile));
    u.clear();
    if (tile == 0) {
      out.cts = std::move(part);
    } else {
      for (int i = 0; i < H; i++) boot_->Add(out.cts[i], out.cts[i], part[i]);
    }
    stages_.ffn += Since(t0);
  }
  for (int i = 0; i < H; i++) AddScalar(out.cts[i], out.cts[i], bout_[i]);
  out.carry = c_h_;
  out.chan = h.chan;         // the down projection's columns carry 1 / chan
  Tap("y_ffn", out.cts, c_h_);
}

// ---------------------------------------------------------------- layer
template <typename word>
void CiBertBaseLayer<word>::Layer(Stream &out, Stream &in,
                                  const EvkMap<word> &evk) {
  NvtxScope _nv("bert-base: layer");
  const int H = cfg_.shape.model;
  AssertTrue(prepared_, "CiBertBaseLayer::Layer: Prepare first");
  AssertTrue(static_cast<int>(in.cts.size()) == H, "Layer: model channels");
  stages_ = Stages{};
  auto t_all = Clock::now();
  // the stream, booted to where the projections read it. `in` is CONSUMED:
  // at BERT-Base's width a copy of the residual stream is gigabytes.
  Stream x;
  x.carry = in.carry;
  x.chan = in.chan;
  x.cts = std::move(in.cts);
  in.cts.clear();
  Lift(x, l_qk_in_, evk);

  Stream attn;
  Attention(attn, x, evk);      // leaves x at l_o_in - 1, the residual's level
  // h_pre = x + attn (both carry c_in)
  Stream h_pre;
  h_pre.carry = c_in_;
  h_pre.chan = x.chan;
  h_pre.cts.resize(H);
  const int la = Level(attn.cts[0]);
  for (int c = 0; c < H; c++) {
    Ct xd;
    boot_->LevelDown(xd, x.cts[c], la);
    x.cts[c] = Ct();
    boot_->Add(h_pre.cts[c], xd, attn.cts[c]);
    attn.cts[c] = Ct();
  }
  x.cts.clear();
  attn.cts.clear();
  Tap("h_pre", h_pre.cts, c_in_);
  Stream h;
  LayerNorm(h, h_pre, w_.attn_norm, w_.attn_norm_bias, cal_.ln1,
            cal_.ln1_chan, evk, "ln1");
  h_pre.cts.clear();

  Lift(h, TopLevel(), evk);
  Stream y;
  FeedForward(y, h, evk);
  Stream z_pre;
  z_pre.carry = c_h_;
  z_pre.chan = h.chan;
  z_pre.cts.resize(H);
  const int ly = Level(y.cts[0]);
  for (int c = 0; c < H; c++) {
    Ct hd;
    boot_->LevelDown(hd, h.cts[c], ly);
    h.cts[c] = Ct();
    boot_->Add(z_pre.cts[c], hd, y.cts[c]);
    y.cts[c] = Ct();
  }
  h.cts.clear();
  y.cts.clear();
  Tap("z_pre", z_pre.cts, c_h_);
  LayerNorm(out, z_pre, w_.ffn_norm, w_.ffn_norm_bias, cal_.ln2,
            cal_.ln2_chan, evk, "ln2");
  stages_.total = Since(t_all);
  if (cfg_.verbose) {
    const Stages &s = stages_;
    std::cout << "  [bert-base] layer " << s.total << " s: boot " << s.boot
              << " (wide " << s.wide_boots << ", narrow " << s.narrow_boots
              << "), qkv " << s.qkv << ", scores " << s.scores << ", softmax "
              << s.softmax << ", values " << s.values << ", o " << s.o
              << ", ln " << s.ln << ", ffn " << s.ffn << ", gelu " << s.gelu
              << "; rotations " << s.rotations << ", relins " << s.relins
              << "; out at level " << Level(out.cts[0]) << std::endl;
  }
}

template class CiBertBaseLayer<uint32_t>;
template class CiBertBaseLayer<uint64_t>;

#endif  // USE_CUBLAS

}  // namespace cheddar
