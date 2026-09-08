#include <algorithm>
#include <cmath>
#include <cstring>

#include <iostream>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/MemoryPool.h"
#include "extension/CiBertLayer.h"
#include "extension/Profile.h"

namespace cheddar {

namespace {

int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

}  // namespace

template <typename word>
typename CiLlamaLayer<word>::Config BaseConfig(
    const typename CiBertLayer<word>::Config &cfg) {
  typename CiLlamaLayer<word>::Config base;
  base.num_tokens = cfg.num_tokens;
  base.proj_rank = cfg.proj_rank;
  base.model_declared = cfg.model_declared;
  base.hidden_declared = cfg.hidden_declared;
  base.model_live = cfg.model_live;
  base.eps = cfg.eps;
  base.product_level = cfg.product_level;
  base.parents_per_tile = cfg.parents_per_tile;
  // BERT runs on the module basis and nothing else: its 768 channels are two
  // dense rank-512 ciphertexts, and the banded convention's `rank/2 - 1` live
  // channels would need three with a duplicate band nobody reads.
  base.module_basis = true;
  base.verbose = cfg.verbose;
  return base;
}

template <typename word>
CiBertLayer<word>::CiBertLayer(
    std::shared_ptr<const BootContext<word>> boot,
    const CiSwitchedCcmmLayout &layout,
    std::vector<const EvaluationKey<word> *> modpack_keys, const Config &cfg)
    : boot_{boot},
      cfg_{cfg},
      base_{boot, layout, std::move(modpack_keys), BaseConfig<word>(cfg)},
      sched_{boot, boot->param_.MaxNumSlots()} {
  num_slots_ = boot_->param_.MaxNumSlots();
  AssertTrue(cfg_.model_declared % cfg_.proj_rank == 0 &&
                 cfg_.hidden_declared % cfg_.proj_rank == 0,
             "CiBertLayer: both declared widths must be a whole number of "
             "ciphertexts");
  num_model_cts_ = cfg_.model_declared / cfg_.proj_rank;
  num_hidden_cts_ = cfg_.hidden_declared / cfg_.proj_rank;
  AssertTrue(cfg_.head_dim % layout.rank == 0,
             "CiBertLayer: the head width must be a whole number of chain "
             "columns");
  num_images_ = cfg_.head_dim / layout.rank;
  AssertTrue(cfg_.num_tokens * cfg_.proj_rank == num_slots_,
             "CiBertLayer: T * rank must be the slot count");

  // Both DERIVED, never fitted (Doing.md 1.5dk): `ToSlot` multiplies the
  // message by `crossing` and `ToCoeff` by `kappa`, and their product is the
  // nominal ratio that makes `Boot` message preserving.
  crossing_ = boot_->GetMessageRatio();
  AssertTrue(crossing_ > 0.0,
             "CiBertLayer: PrepareEvalMod must run before construction");
  kappa_ =
      std::pow(2.0, -boot_->GetBootParameter().GetLogMessageRatio()) / crossing_;

  slot_level_ = sched_.GetSlotLevel();
  op_level_ = slot_level_ - 1;
  sched_.SetModuleBasis(base_.GetModuleBasis());
  if (cfg_.verbose) {
    std::cout << "CiBertLayer: slot " << slot_level_ << ", op " << op_level_
              << ", StC " << sched_.GetStCLevel() << "; crossing " << crossing_
              << ", kappa " << kappa_ << "; " << num_model_cts_
              << " model cts, " << num_hidden_cts_ << " hidden, "
              << num_images_ << " images a tensor" << std::endl;
  }
}

template <typename word>
void CiBertLayer<word>::AddRequiredRotations(EvkRequest &req) const {
  base_.AddRequiredRotations(req);
  // The two LayerNorms' reduction trees. They are the same distances the
  // RMSNorm of the same width asks for -- the mean and the sum of squares
  // walk one tree each and the tree depends on the shape alone -- so the
  // base layer's request already covers them; asked again here because a
  // caller reading this file should not have to know that.
  for (int d = cfg_.num_tokens; d < num_slots_; d *= 2) {
    req.AddRequest(d, op_level_);
  }
}

template <typename word>
int CiBertLayer<word>::NormDegree(double window) const {
  if (cfg_.norm_degree > 0) return cfg_.norm_degree;
  // `RmsNorm.h`'s measured table, on the window's ratio. With the per-token
  // public rescale in front of it the window is the calibration's own error,
  // which is where the cheap end of this table lives.
  if (window <= 4.18) return 7;
  if (window <= 6.0) return 9;
  if (window <= 12.0) return 15;
  return 23;
}

template <typename word>
double CiBertLayer<word>::PlainNormInvSqrt(bool ffn, double alpha,
                                           double window, double var) const {
  (void)ffn;
  const double beta2 = alpha;
  LayerNormHandler<word> ln(boot_, cfg_.num_tokens, cfg_.model_declared, 1.0,
                            op_level_, beta2 * cfg_.eps, window,
                            NormDegree(window), 1, cfg_.model_live);
  return ln.PlainInvSqrt(alpha * var);
}

template <typename word>
void CiBertLayer<word>::Canonicalise(Ct &ct, double factor) const {
  Constant<word> k;
  boot_->encoder_.EncodeConstant(
      k, slot_level_,
      boot_->param_.GetScale(op_level_) *
          boot_->param_.GetRescalePrimeProd(slot_level_) / ct.GetScale(),
      factor);
  boot_->Mult(ct, ct, k);
  boot_->Rescale(ct, ct);
}

template <typename word>
void CiBertLayer<word>::Canonicalise(Ct &ct, const Pt &pt) const {
  boot_->Mult(ct, ct, pt);
  boot_->Rescale(ct, ct);
}

template <typename word>
Plaintext<word> CiBertLayer<word>::TokenPlaintext(
    double factor, const std::vector<double> &per_token, double scale) const {
  AssertTrue(static_cast<int>(per_token.size()) == cfg_.num_tokens,
             "CiBertLayer: the per-token rescale needs one factor per token");
  // The stream's slot address is `channel * num_tokens + rev(token)` (1.5du),
  // so a per-token factor is a stride-`num_tokens` pattern. On the module
  // basis there is no duplicate band, so it is that and nothing else.
  const int log_t = Log2Ceil(cfg_.num_tokens);
  std::vector<Complex> vals(num_slots_, Complex(0.0, 0.0));
  for (int t = 0; t < cfg_.num_tokens; t++) {
    const int p = Rev(t, log_t);
    for (int c = 0; p + c * cfg_.num_tokens < num_slots_; c++) {
      vals[p + c * cfg_.num_tokens] = Complex(factor * per_token[t], 0.0);
    }
  }
  Plaintext<word> pt;
  boot_->encoder_.Encode(pt, slot_level_,
                         boot_->param_.GetScale(op_level_) *
                             boot_->param_.GetRescalePrimeProd(slot_level_) /
                             scale,
                         vals);
  return pt;
}

template <typename word>
void CiBertLayer<word>::AddBias(std::vector<Ct> &imgs,
                                const std::vector<double> &bias_declared,
                                double scale,
                                const std::vector<double> &per_token) const {
  NvtxScope _nv("bert: bias");
  const int rank = cfg_.proj_rank;
  const int log_rank = Log2Ceil(rank);
  const int T = cfg_.num_tokens;
  AssertTrue(static_cast<int>(bias_declared.size()) ==
                 static_cast<int>(imgs.size()) * rank,
             "CiBertLayer: the bias must be stated at every declared output");
  for (size_t g = 0; g < imgs.size(); g++) {
    // A bias is constant along the token axis, so its ring element has
    // `comp_i[t] = b_i` for every `t` -- and a coefficient image IS the
    // banded recomposition of its components,
    // `rec[t*rank + i] = comp_i[t] + [i != 0] comp_{rank-i}[t+1]`, whatever
    // basis the transforms read it in. The last token has no partner.
    // Module row `i` of group `g` is declared output `g*rank + rev(i)`, which
    // is `Project`'s own contract and the reason the caller states the bias
    // at declared indices and never writes a reversal of its own.
    std::vector<double> coeff(static_cast<size_t>(rank) * T, 0.0);
    for (int i = 0; i < rank; i++) {
      const double bi =
          bias_declared[static_cast<size_t>(g) * rank + Rev(i, log_rank)];
      const double partner =
          i == 0 ? 0.0
                 : bias_declared[static_cast<size_t>(g) * rank +
                                 Rev(rank - i, log_rank)];
      for (int t = 0; t < T; t++) {
        // A suppressed row's bias is suppressed with it: the coefficient
        // position IS the token (Doing.md 1.5du), so the factor is a
        // per-position one and the partner deposit takes the NEXT token's.
        const double ft = per_token.empty() ? 1.0 : per_token[t];
        const double fn =
            per_token.empty() ? 1.0 : per_token[std::min(t + 1, T - 1)];
        coeff[static_cast<size_t>(t) * rank + i] =
            scale * (ft * bi + (t + 1 < T ? fn * partner : 0.0));
      }
    }
    Plaintext<word> pt;
    const int level = boot_->param_.NPToLevel(imgs[g].GetNP());
    boot_->encoder_.EncodeCoeff(pt, level, imgs[g].GetScale(), coeff);
    boot_->Add(imgs[g], imgs[g], pt);
  }
}

template <typename word>
void CiBertLayer<word>::Emit(std::vector<std::vector<Ct>> &qkv,
                             const std::vector<Ct> &stream, int qkv_declared,
                             const Weights &w, const Calibration &c) {
  NvtxScope _nv("bert: Emit (q, k, v)");
  AssertTrue(static_cast<int>(stream.size()) == num_model_cts_,
             "CiBertLayer: the residual stream is " +
                 std::to_string(num_model_cts_) + " ciphertexts");
  AssertTrue(w.q.Given() && w.k.Given() && w.v.Given(),
             "CiBertLayer: q, k and v must each be given in exactly one form");
  std::vector<Ct> ins(num_model_cts_);
  for (int i = 0; i < num_model_cts_; i++) {
    boot_->LevelDown(ins[i], stream[i], cfg_.product_level);
  }
  const ProjectionWeight *pw[3] = {&w.q, &w.k, &w.v};
  const std::vector<double> *bias[3] = {w.bq, w.bk, w.bv};
  const double sc[3] = {c.q_scale, c.k_scale, c.v_scale};
  const char *nm[3] = {".q", ".k", ".v"};
  // What one model unit is worth in the stream this reads (the header's
  // table): the coefficient stream carries `stream_scale`, exactly as the
  // Llama layer's does, because the norm's gain took `kappa` out on the way.
  const double unit = c.stream_scale;
  qkv.resize(3);
  for (int j = 0; j < 3; j++) {
    base_.Project(qkv[j], ins, cfg_.model_declared, qkv_declared, *pw[j],
                  sc[j], (w.tag + nm[j]).c_str(), /*in_density=*/1,
                  /*out_density=*/1);
    AssertTrue(static_cast<int>(qkv[j].size()) == qkv_declared / cfg_.proj_rank,
               "CiBertLayer: a projection did not land in the declared "
               "number of ciphertexts");
    if (bias[j] != nullptr) AddBias(qkv[j], *bias[j], unit * sc[j]);
  }
}

template <typename word>
void CiBertLayer<word>::NormTurn(std::vector<Ct> &res,
                                 const std::vector<Ct> &stream,
                                 const std::vector<double> &gain,
                                 const std::vector<double> &bias, double alpha,
                                 double window,
                                 const std::vector<double> &token_scale,
                                 double in_scale, double out_scale,
                                 const std::vector<double> &row_suppress,
                                 const EvkMap<word> &evk) {
  NvtxScope _nv("bert: NormTurn");
  // THE OPERATOR'S INPUT RIDES AT ONE, AND THAT IS WHERE ITS PRECISION GOES.
  // LayerNorm is scale invariant, so feeding it `beta * x` with `alpha /
  // beta^2` in place of `alpha` computes the same function -- and a
  // ciphertext's added error is ABSOLUTE, so a message riding at the model's
  // own magnitude takes the square, the two reductions and the closing
  // multiply at `1/alpha` of the precision the same circuit gets at one.
  // `beta = sqrt(alpha)` is exactly the constant that puts it there, and it
  // folds into the crossing's own multiply.
  const double beta = std::sqrt(alpha);
  std::vector<Ct> slots;
  {
    std::vector<const Ct *> xs(num_model_cts_);
    for (int i = 0; i < num_model_cts_; i++) xs[i] = &stream[i];
    sched_.ToSlotBatch(slots, xs, evk, /*min_ks=*/false);
  }
  // THE STREAM'S FACTOR AND THE CROSSING'S GO OUT HERE, TOGETHER WITH THE
  // PUBLIC PER-TOKEN RESCALE. The coefficient stream carries `stream_scale`
  // per model unit (the class comment's table) and `ToSlot` has just
  // multiplied by `crossing`; what is left after this multiply is `beta`
  // times the model's own value, which is what the handler is calibrated for
  // -- and the per-token factor, which LayerNorm cancels identically.
  const double factor = beta / (crossing_ * in_scale);
  Pt token_pt;
  if (!token_scale.empty()) {
    token_pt = TokenPlaintext(factor, token_scale, slots[0].GetScale());
  }
  for (int i = 0; i < num_model_cts_; i++) {
    if (token_scale.empty()) {
      Canonicalise(slots[i], factor);
    } else {
      Canonicalise(slots[i], token_pt);
    }
  }

  // The handler sees `beta * x`, so its own layer constant is one and its
  // epsilon carries `beta^2`; the gain then carries no `sqrt(alpha)` at all,
  // only the model's gain and the stream factor the output has to leave with.
  const int degree = NormDegree(window);
  LayerNormHandler<word> ln(boot_, cfg_.num_tokens, cfg_.model_declared,
                            /*layer_constant=*/1.0, op_level_,
                            beta * beta * cfg_.eps, window, degree,
                            /*channel_stride=*/1, cfg_.model_live);
  AssertTrue(ln.GetNumCiphertexts() == num_model_cts_,
             "CiBertLayer: LayerNormHandler disagrees about the width");
  AssertTrue(ln.GetOutputLevel() >= sched_.GetStCLevel(),
             "CiBertLayer: the LayerNorm lands at level " +
                 std::to_string(ln.GetOutputLevel()) + " but StC is compiled "
                 "at " + std::to_string(sched_.GetStCLevel()) +
                 " -- the norm spent " +
                 std::to_string(sched_.GetStCLevel() - ln.GetOutputLevel()) +
                 " levels too many. The slack is a BootParameter setting.");

  // The gain, the bias and the live mask, per DECLARED channel: in slots the
  // address is `channel * num_tokens + token`, so all three are constant
  // along the token axis (`NormWeights` in the Llama layer, same convention).
  const int rank = cfg_.proj_rank;
  std::vector<std::vector<Complex>> wts(num_model_cts_), bs(num_model_cts_),
      mask(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    wts[k].assign(num_slots_, Complex(0.0, 0.0));
    bs[k].assign(num_slots_, Complex(0.0, 0.0));
    mask[k].assign(num_slots_, Complex(0.0, 0.0));
    for (int ch = 0; ch < rank; ch++) {
      const int declared = k * rank + ch;
      const bool live = declared < cfg_.model_live;
      // THE OUTPUT'S UNITS ARE THE STREAM'S. LayerNorm is scale invariant, so
      // whatever came in is gone; the stream factor has to be put back here,
      // on the gain AND on the bias, because the next residual add meets a
      // projection output that carries it.
      // `stream_scale / kappa`: the output has to LEAVE in the stream's
      // units, and `ToCoeff` below multiplies by `kappa` on the way out, so
      // the gain and the bias carry the quotient and the coefficient stream
      // carries exactly `stream_scale`.
      const double g = live ? gain[declared] * out_scale / kappa_ : 0.0;
      const double b = live ? bias[declared] * out_scale / kappa_ : 0.0;
      for (int t = 0; t < cfg_.num_tokens; t++) {
        // In slots the address is `channel * num_tokens + rev(token)`; the
        // gain and the bias are token-uniform except for the row
        // suppression, which is exactly why it is free here.
        const size_t s = static_cast<size_t>(ch) * cfg_.num_tokens +
                         Rev(t, Log2Ceil(cfg_.num_tokens));
        const double ft = row_suppress.empty() ? 1.0 : row_suppress[t];
        wts[k][s] = Complex(g * ft, 0.0);
        bs[k][s] = Complex(b * ft, 0.0);
        mask[k][s] = Complex(live ? 1.0 : 0.0, 0.0);
      }
    }
  }
  std::vector<Ct> outv;
  ln.Apply(outv, slots, wts, bs, mask, evk);
  if (keep_norm_slots_) {
    norm_slots_.clear();
    norm_slots_.resize(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->Copy(norm_slots_[k], outv[k]);
    }
  }
  res.resize(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    sched_.ToCoeff(res[k], outv[k], evk, /*min_ks=*/false);
  }
}

template <typename word>
void CiBertLayer<word>::AttentionTurn(std::vector<Ct> &res,
                                      const std::vector<Ct> &seamed,
                                      const std::vector<Ct> &stream,
                                      const Weights &w, const Calibration &c,
                                      const EvkMap<word> &evk) {
  NvtxScope _nv("bert: AttentionTurn (O, residual, LayerNorm)");
  AssertTrue(w.o.Given() && w.attn_gain != nullptr && w.attn_bias != nullptr,
             "CiBertLayer: O and the post-attention norm must be given");
  AssertTrue(!w.tag.empty(),
             "CiBertLayer: a layer's weights need a tag -- the projection leg "
             "caches by name");
  const int attn_declared =
      static_cast<int>(seamed.size()) * cfg_.proj_rank;

  std::vector<Ct> h(num_model_cts_);
  {
    std::vector<Ct> ins(seamed.size());
    for (size_t i = 0; i < seamed.size(); i++) {
      boot_->LevelDown(ins[i], seamed[i], cfg_.product_level);
    }
    std::vector<Ct> o_out;
    base_.Project(o_out, ins, attn_declared, cfg_.model_declared, w.o,
                  c.o_scale, (w.tag + ".o").c_str(), /*in_density=*/1,
                  /*out_density=*/1);
    AssertTrue(static_cast<int>(o_out.size()) == num_model_cts_,
               "CiBertLayer: the O projection did not land in " +
                   std::to_string(num_model_cts_) + " ciphertexts");
    // `o_scale` is sized so this output IS the stream's units, which is what
    // lets the residual below be a plain add and the bias be one constant.
    if (w.bo != nullptr) AddBias(o_out, *w.bo, c.stream_scale);
    // THE RESIDUAL, AND IT IS THE LAYER'S INPUT THAT MOVES. Post-norm adds
    // first and normalises second, so `x` has to meet the projection's output
    // at ITS level, not the other way round.
    for (int k = 0; k < num_model_cts_; k++) {
      Ct down;
      boot_->LevelDown(down, stream[k],
                       boot_->param_.NPToLevel(o_out[k].GetNP()));
      boot_->Add(h[k], down, o_out[k]);
    }
  }
  MemoryPool::Report("bert: after the O projection and the residual");
  // What this layer writes, which is what the next one will read.
  const double out_scale =
      c.stream_out > 0.0 ? c.stream_out : c.stream_scale;
  // The post-attention norm is where the row suppression goes ON: `H`
  // carries it, and the OUTPUT norm below takes it back out.
  NormTurn(res, h, *w.attn_gain, *w.attn_bias, c.attn_alpha, c.attn_window,
           c.attn_scale, c.stream_scale, out_scale, c.row_suppress, evk);
  MemoryPool::Report("bert: after the post-attention LayerNorm");
}

template <typename word>
void CiBertLayer<word>::FeedForward(std::vector<Ct> &res,
                                    const std::vector<Ct> &h,
                                    const Weights &w, const Calibration &c,
                                    const EvkMap<word> &evk) {
  NvtxScope _nv("bert: FeedForward");
  AssertTrue(w.inter.Given() && w.out.Given() && w.ffn_gain != nullptr &&
                 w.ffn_bias != nullptr,
             "CiBertLayer: the feed-forward's weights must be given");
  AssertTrue(static_cast<int>(h.size()) == num_model_cts_,
             "CiBertLayer: the feed-forward's input is " +
                 std::to_string(num_model_cts_) + " ciphertexts");
  // `h` is what `AttentionTurn` wrote, so everything here is stated in the
  // OUTPUT's units, not the layer input's.
  const double out_scale =
      c.stream_out > 0.0 ? c.stream_out : c.stream_scale;

  // ---- the intermediate projection ---------------------------------------
  std::vector<Ct> u;
  {
    std::vector<Ct> ins(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->LevelDown(ins[k], h[k], cfg_.product_level);
    }
    base_.Project(u, ins, cfg_.model_declared, cfg_.hidden_declared, w.inter,
                  c.int_scale, (w.tag + ".int").c_str(), 1, 1);
    AssertTrue(static_cast<int>(u.size()) == num_hidden_cts_,
               "CiBertLayer: the intermediate projection did not land in " +
                   std::to_string(num_hidden_cts_) + " ciphertexts");
  }
  MemoryPool::Report("bert: after the intermediate projection");

  // ---- the crossing, the bias, GELU --------------------------------------
  std::vector<Ct> act(num_hidden_cts_);
  {
    std::vector<Ct> ups;
    {
      std::vector<const Ct *> xs(num_hidden_cts_);
      for (int i = 0; i < num_hidden_cts_; i++) xs[i] = &u[i];
      sched_.ToSlotBatch(ups, xs, evk, /*min_ks=*/false);
      u.clear();
    }
    // The GELU's groups, per SLOT: the outliers are two hidden channels of
    // 3072 (`GeLu.h`), so the assignment is per (token, channel) and not per
    // token. In slots the address is `channel * num_tokens + rev(token)`.
    std::vector<typename GeLuHandler<word>::Group> groups = {
        {GeLuHandler<word>::Kind::kFit, c.gelu_range, c.gelu_degree},
        {GeLuHandler<word>::Kind::kIdentity, c.gelu_range, 0},
        {GeLuHandler<word>::Kind::kZero, c.gelu_range, 0}};
    GeLuHandler<word> gelu(boot_, groups, op_level_);
    const int log_t = Log2Ceil(cfg_.num_tokens);
    const int rank = cfg_.proj_rank;
    std::vector<std::vector<Complex>> mask(3);
    for (int i = 0; i < num_hidden_cts_; i++) {
      // `1/range` and the crossing's own constants ride the same multiply;
      // the bias is a plaintext add at the level that leaves, in the same
      // units. Neither costs a level.
      // The intermediate output carries `int_scale * stream_scale` per
      // model unit and `ToSlot` has just multiplied by `crossing`; what is
      // left after this multiply is `u / range`, which is what the fit takes.
      // `H` carries the row suppression and GELU is not scale invariant, so
      // the argument has it divided out here -- on the multiply the crossing
      // is already paying for -- and the ANSWER gets it back through the
      // masks below.
      const double base =
          1.0 / (crossing_ * c.int_scale * out_scale * c.gelu_range);
      if (c.row_suppress.empty()) {
        Canonicalise(ups[i], base);
      } else {
        std::vector<double> inv(cfg_.num_tokens);
        for (int t = 0; t < cfg_.num_tokens; t++) {
          inv[t] = 1.0 / c.row_suppress[t];
        }
        if (i == 0) gelu_pt_ = TokenPlaintext(base, inv, ups[i].GetScale());
        Canonicalise(ups[i], gelu_pt_);
      }
      if (w.bint != nullptr) {
        std::vector<Complex> bmsg(num_slots_, Complex(0.0, 0.0));
        for (int ch = 0; ch < rank; ch++) {
          const int declared = i * rank + ch;
          const double b = declared < cfg_.hidden_live
                               ? (*w.bint)[declared] / c.gelu_range
                               : 0.0;
          for (int t = 0; t < cfg_.num_tokens; t++) {
            bmsg[static_cast<size_t>(ch) * cfg_.num_tokens + t] =
                Complex(b, 0.0);
          }
        }
        Pt bpt;
        boot_->gpu_encoder_.Encode(bpt, op_level_,
                                   boot_->param_.GetScale(op_level_), bmsg);
        boot_->Add(ups[i], ups[i], bpt);
      }
      for (int gsel = 0; gsel < 3; gsel++) {
        mask[gsel].assign(num_slots_, Complex(0.0, 0.0));
      }
      for (int ch = 0; ch < rank; ch++) {
        const int declared = i * rank + ch;
        for (int t = 0; t < cfg_.num_tokens; t++) {
          const size_t s =
              static_cast<size_t>(ch) * cfg_.num_tokens + Rev(t, log_t);
          int which = 0;
          if (declared < cfg_.hidden_live && !c.gelu_group.empty()) {
            which = c.gelu_group[static_cast<size_t>(t) * cfg_.hidden_live +
                                 declared];
          }
          mask[which][s] = Complex(1.0, 0.0);
        }
      }
      gelu.Apply(act[i], ups[i], mask, evk);
      ups[i] = Ct{};
      if (!c.row_suppress.empty()) {
        // THE SUPPRESSION GOES ON THE ANSWER, and it cannot ride a mask: the
        // fitted group's mask multiplies the INPUT (that is what stops a
        // saturated slot from being evaluated at |v| = 16, `GeLu.h`), so a
        // factor put there would scale the argument instead of the value --
        // measured, that alone took the chain to 2^+24 by layer 1. It costs
        // one of the two levels between the fit's landing and StC's.
        std::vector<Complex> msg(num_slots_, Complex(0.0, 0.0));
        const int log_t = Log2Ceil(cfg_.num_tokens);
        for (int ch = 0; ch < rank; ch++) {
          for (int t = 0; t < cfg_.num_tokens; t++) {
            msg[static_cast<size_t>(ch) * cfg_.num_tokens + Rev(t, log_t)] =
                Complex(c.row_suppress[t], 0.0);
          }
        }
        const int lvl = boot_->param_.NPToLevel(act[i].GetNP());
        Pt spt;
        boot_->gpu_encoder_.Encode(spt, lvl, boot_->param_.GetScale(lvl), msg);
        Ct scaled;
        boot_->Mult(scaled, act[i], spt);
        boot_->Rescale(act[i], scaled);
      }
    }
  }
  MemoryPool::Report("bert: after GELU");

  // ---- the output projection, the residual, and the second norm ----------
  {
    std::vector<Ct> ins(num_hidden_cts_);
    for (int i = 0; i < num_hidden_cts_; i++) {
      Ct co;
      sched_.ToCoeff(co, act[i], evk, /*min_ks=*/false);
      boot_->LevelDown(ins[i], co, cfg_.product_level);
      act[i] = Ct{};
    }
    std::vector<Ct> y;
    base_.Project(y, ins, cfg_.hidden_declared, cfg_.model_declared, w.out,
                  c.out_scale, (w.tag + ".out").c_str(), 1, 1);
    AssertTrue(static_cast<int>(y.size()) == num_model_cts_,
               "CiBertLayer: the output projection did not land in " +
                   std::to_string(num_model_cts_) + " ciphertexts");
    if (w.bout != nullptr) {
      AddBias(y, *w.bout, out_scale, c.row_suppress);
    }
    std::vector<Ct> z(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      Ct down;
      boot_->LevelDown(down, h[k], boot_->param_.NPToLevel(y[k].GetNP()));
      boot_->Add(z[k], down, y[k]);
    }
    MemoryPool::Report("bert: after the output projection and the residual");
    // And OFF: LayerNorm is exactly scale invariant per token, so the
    // factor both halves of `z` carry cancels identically here.
    NormTurn(res, z, *w.ffn_gain, *w.ffn_bias, c.ffn_alpha, c.ffn_window,
             c.ffn_scale, out_scale, out_scale, {}, evk);
  }
  MemoryPool::Report("bert: after the output LayerNorm");
}

template class CiBertLayer<uint32_t>;
template class CiBertLayer<uint64_t>;

}  // namespace cheddar
