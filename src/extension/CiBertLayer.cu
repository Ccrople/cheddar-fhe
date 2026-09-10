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
    double factor, const std::vector<double> &per_token, double scale,
    const std::vector<double> &per_channel, int ct_index) const {
  AssertTrue(per_token.empty() ||
                 static_cast<int>(per_token.size()) == cfg_.num_tokens,
             "CiBertLayer: the per-token rescale needs one factor per token");
  AssertTrue(per_channel.empty() ||
                 static_cast<int>(per_channel.size()) >= cfg_.model_declared,
             "CiBertLayer: the per-channel suppression needs one factor per "
             "declared model channel");
  // The stream's slot address is `channel * num_tokens + rev(token)` (1.5du),
  // so a per-token factor is a stride-`num_tokens` pattern. On the module
  // basis there is no duplicate band, so it is that and nothing else.
  const int log_t = Log2Ceil(cfg_.num_tokens);
  std::vector<Complex> vals(num_slots_, Complex(0.0, 0.0));
  // Both axes ride the SAME plaintext: the slot address is
  // `channel * num_tokens + rev(token)`, so a per-token factor is a
  // stride-`num_tokens` pattern and a per-channel one is constant within each
  // stride. `1 / d_c` is what the norm needs to see the TRUE variance, and
  // putting it here costs nothing because the multiply was happening anyway.
  for (int t = 0; t < cfg_.num_tokens; t++) {
    const int p = Rev(t, log_t);
    const double ft = per_token.empty() ? 1.0 : per_token[t];
    for (int c = 0; p + c * cfg_.num_tokens < num_slots_; c++) {
      double fc = 1.0;
      if (!per_channel.empty()) {
        const int declared = ct_index * cfg_.proj_rank + c;
        const double d = declared < static_cast<int>(per_channel.size())
                             ? per_channel[declared]
                             : 1.0;
        fc = d > 0.0 ? 1.0 / d : 1.0;
      }
      vals[p + c * cfg_.num_tokens] = Complex(factor * ft * fc, 0.0);
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
                                 double window, int degree_in,
                                 const std::vector<double> &invsqrt,
                                 const std::vector<double> &token_scale,
                                 double in_scale, double out_scale,
                                 const std::vector<double> &row_suppress,
                                 const EvkMap<word> &evk,
                                 const NormStages *stages) {
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
  const std::vector<double> *chan =
      (stages != nullptr && stages->channel_suppress != nullptr &&
       !stages->channel_suppress->empty())
          ? stages->channel_suppress
          : nullptr;
  if (chan != nullptr) {
    // ONE PLAINTEXT PER CIPHERTEXT, because `1 / d_c` differs along the
    // channel axis and a ciphertext holds `proj_rank` channels. It is the
    // same multiply and the same single level the per-token factor already
    // used -- the channel half is free.
    for (int i = 0; i < num_model_cts_; i++) {
      Pt pt = TokenPlaintext(factor, token_scale, slots[i].GetScale(), *chan,
                             i);
      Canonicalise(slots[i], pt);
    }
  } else if (token_scale.empty()) {
    for (int i = 0; i < num_model_cts_; i++) Canonicalise(slots[i], factor);
  } else {
    const Pt token_pt =
        TokenPlaintext(factor, token_scale, slots[0].GetScale());
    for (int i = 0; i < num_model_cts_; i++) Canonicalise(slots[i], token_pt);
  }

  // The live mask, which the centring needs and therefore has to exist
  // before the crude stage rather than beside the gain below.
  const int rank_m = cfg_.proj_rank;
  std::vector<std::vector<Complex>> live_mask(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    live_mask[k].assign(num_slots_, Complex(0.0, 0.0));
    for (int ch = 0; ch < rank_m; ch++) {
      const bool live = k * rank_m + ch < cfg_.model_live;
      for (int t = 0; t < cfg_.num_tokens; t++) {
        live_mask[k][static_cast<size_t>(ch) * cfg_.num_tokens +
                     Rev(t, Log2Ceil(cfg_.num_tokens))] =
            Complex(live ? 1.0 : 0.0, 0.0);
      }
    }
  }

  // THE INVERSE SQUARE ROOT IN TWO STAGES, when the calibration asks for it.
  //
  // The crude stage runs on the layer's own window and hands back
  // `y0 = (x - mu) r0`; the stream then makes an ordinary crossing, which
  // costs one bootstrap and buys the refine stage a fresh span; the refine
  // stage reads `mean(y0^2)`, which is `(1 + e)^2` IDENTICALLY for the crude
  // stage's relative error `e`, so its window is a theorem rather than a
  // statistic. `LayerNorm.h`'s Mode comment carries the argument in full.
  //
  // THE CRUDE STAGE'S OUTPUT HAS TO RIDE THE CROSSING, and `y0` is the
  // NORMALISED row -- magnitude ten to fifteen -- where the coefficient
  // stream between layers rides at `out_scale` times the model, two orders
  // smaller. Handed over raw it is far outside the crossing's message range
  // and EvalMod returns noise (measured: the twelve-layer chain at 2^+9 to
  // 2^+16). The fix costs no level: the handler's own layer constant scales
  // its output as `1 / sqrt(alpha)`, so `alpha = 1 / out_scale^2` puts `y0`
  // at exactly the stream's ride, the refine stage takes the SAME constant
  // (its argument is then `out_scale^2 * (1/out_scale^2) = 1`, the window's
  // centre), and the `1 / out_scale` that has to come back out rides the
  // gain plaintext, which is a multiply that was happening anyway.
  const bool staged = stages != nullptr && stages->crude_degree > 0;
  std::vector<Ct> staged_slots;
  if (staged) {
    LayerNormHandler<word> crude(
        boot_, cfg_.num_tokens, cfg_.model_declared, /*layer_constant=*/1.0,
        op_level_, beta * beta * cfg_.eps, window, stages->crude_degree,
        /*channel_stride=*/1, cfg_.model_live,
        stages->crude_invsqrt ? *stages->crude_invsqrt : std::vector<double>{},
        LayerNormHandler<word>::Mode::kCrude);
    AssertTrue(crude.GetOutputLevel() >= sched_.GetStCLevel(),
               "CiBertLayer: the crude invsqrt lands at level " +
                   std::to_string(crude.GetOutputLevel()) + " but StC is at " +
                   std::to_string(sched_.GetStCLevel()) + " -- degree " +
                   std::to_string(stages->crude_degree) + " is " +
                   std::to_string(sched_.GetStCLevel() -
                                  crude.GetOutputLevel()) +
                   " levels too many for the first span.");
    // THE CRUDE STAGE'S GAIN IS NOT THE MODEL'S. `y0` is the NORMALISED row,
    // magnitude ten to fifteen, and the coefficient stream between layers
    // rides at `out_scale` times the model -- two orders smaller. Handed to
    // the crossing raw it is far outside the message range and EvalMod
    // returns noise (measured: the chain at 2^+9 to 2^+16). So the crude
    // stage's one constant gain is `out_scale / kappa`: `ToCoeff` multiplies
    // by `kappa` on the way out, and what the coefficient stream then carries
    // is exactly `out_scale` times the normalised row, which is what every
    // other stream in this layer carries.
    //
    // It cannot ride `layer_constant` instead. That scales the ARGUMENT by
    // `alpha` as well as the output by `1/sqrt(alpha)`, and the argument has
    // to stay inside the window -- `alpha = 1/out_scale^2` puts it at 2192
    // against a window of [0.8, 1.2] (measured: the chain at 2^+11).
    std::vector<std::vector<Complex>> crude_gain(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      crude_gain[k].assign(num_slots_, Complex(0.0, 0.0));
      for (int ch = 0; ch < rank_m; ch++) {
        const bool live = k * rank_m + ch < cfg_.model_live;
        for (int t = 0; t < cfg_.num_tokens; t++) {
          crude_gain[k][static_cast<size_t>(ch) * cfg_.num_tokens +
                        Rev(t, Log2Ceil(cfg_.num_tokens))] =
              Complex(live ? out_scale / kappa_ : 0.0, 0.0);
        }
      }
    }
    std::vector<Ct> y0;
    crude.Apply(y0, slots, crude_gain, /*bias=*/{}, live_mask, evk);
    // The crossing. `ToCoeff` multiplies by `kappa` and `ToSlot` by
    // `crossing`, so the round trip has to be divided back out or the refine
    // stage's argument is off its window by the square of the product.
    std::vector<Ct> coeffed(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      sched_.ToCoeff(coeffed[k], y0[k], evk, /*min_ks=*/false);
    }
    {
      std::vector<const Ct *> xs2(num_model_cts_);
      for (int k = 0; k < num_model_cts_; k++) xs2[k] = &coeffed[k];
      sched_.ToSlotBatch(staged_slots, xs2, evk, /*min_ks=*/false);
    }
    // `ToSlot` multiplied by `crossing`, and the stream carries `out_scale`;
    // both come off here, so the refine stage reads the normalised row and
    // its `mean(y0^2)` is one -- the centre of its window.
    for (int k = 0; k < num_model_cts_; k++) {
      Canonicalise(staged_slots[k], 1.0 / (crossing_ * out_scale));
    }
  }
  const std::vector<Ct> &norm_in = staged ? staged_slots : slots;

  // The handler sees `beta * x`, so its own layer constant is one and its
  // epsilon carries `beta^2`; the gain then carries no `sqrt(alpha)` at all,
  // only the model's gain and the stream factor the output has to leave with.
  // Staged, it sees `y0` instead: already centred, `mean(y0^2) ~ 1`, and its
  // epsilon already spent -- so the constant is one, the epsilon zero, and
  // the window the crude stage's own error bound.
  // The window the handler below actually reads -- the refine stage's when
  // staged -- so the degree fallback is taken against the right one. Without
  // this a calibration that leaves the degree at zero (the served prompt's
  // does) hands the refine stage a degree of zero and `EvalPoly` refuses it.
  const double use_window = staged ? stages->refine_window : window;
  // STAGED, `degree_in` IS THE ONE-STAGE DEGREE and belongs to neither
  // stage: the plan's single-stage entry is what the norm would need without
  // the split (511 at the feed-forward norms of layers 9 and 10), and asking
  // the refine tree for it spends the span twice over. The refine degree is
  // the length of its own fitted vector, and the fallback is the rate law on
  // the refine WINDOW.
  int degree = degree_in > 0 ? degree_in : NormDegree(use_window);
  if (staged) {
    degree = (stages->refine_invsqrt != nullptr &&
              !stages->refine_invsqrt->empty())
                 ? static_cast<int>(stages->refine_invsqrt->size()) - 1
                 : NormDegree(use_window);
  }
  LayerNormHandler<word> ln(
      boot_, cfg_.num_tokens, cfg_.model_declared, /*layer_constant=*/1.0,
      op_level_, staged ? 0.0 : beta * beta * cfg_.eps, use_window,
      degree, /*channel_stride=*/1, cfg_.model_live,
      staged && stages->refine_invsqrt ? *stages->refine_invsqrt : invsqrt,
      staged ? LayerNormHandler<word>::Mode::kRefine
             : LayerNormHandler<word>::Mode::kWhole);
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
  std::vector<std::vector<Complex>> wts(num_model_cts_), bs(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    wts[k].assign(num_slots_, Complex(0.0, 0.0));
    bs[k].assign(num_slots_, Complex(0.0, 0.0));
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
      // The gain and the bias put `d_c` BACK, for the stream that leaves.
      // Everything else that produces this stream (the O and down
      // projections and their biases) folds the same factor into its weights,
      // so the residual add downstream is consistent.
      const double dc =
          (chan != nullptr && declared < static_cast<int>(chan->size()))
              ? (*chan)[declared]
              : 1.0;
      const double g = live ? gain[declared] * out_scale * dc / kappa_ : 0.0;
      const double b = live ? bias[declared] * out_scale * dc / kappa_ : 0.0;
      for (int t = 0; t < cfg_.num_tokens; t++) {
        // In slots the address is `channel * num_tokens + rev(token)`; the
        // gain and the bias are token-uniform except for the row
        // suppression, which is exactly why it is free here.
        const size_t s = static_cast<size_t>(ch) * cfg_.num_tokens +
                         Rev(t, Log2Ceil(cfg_.num_tokens));
        const double ft = row_suppress.empty() ? 1.0 : row_suppress[t];
        wts[k][s] = Complex(g * ft, 0.0);
        bs[k][s] = Complex(b * ft, 0.0);
      }
    }
  }
  std::vector<Ct> outv;
  ln.Apply(outv, norm_in, wts, bs, live_mask, evk);
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
  NormStages attn_stages;
  attn_stages.crude_degree = c.attn_crude_degree;
  attn_stages.refine_window = c.attn_refine_window;
  attn_stages.crude_invsqrt = &c.attn_crude_invsqrt;
  attn_stages.refine_invsqrt = &c.attn_refine_invsqrt;
  attn_stages.channel_suppress = &c.channel_suppress;
  NormTurn(res, h, *w.attn_gain, *w.attn_bias, c.attn_alpha, c.attn_window,
           c.attn_degree, c.attn_invsqrt, c.attn_scale, c.stream_scale,
           out_scale, c.row_suppress, evk, &attn_stages);
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
    // Four groups at most: the bulk fit, the two saturated ones, and -- when
    // the calibration says a handful of slots reach past the bulk range with
    // no stable sign -- a WIDE fit for them. Measured over a corpus, that is
    // 30 slots of 393,216 at layers 9 and 10 and none anywhere else, and
    // without it those slots would be a Chebyshev evaluated outside its
    // interval, which is unbounded.
    // ---- a MODE plan replaces the groups entirely ------------------------
    //
    // `v_j = (u_j - c_j) / rad_j` is the channel's OWN certified interval,
    // and both halves of that affine are public and free: `1/rad_j` rides
    // the crossing's per-slot multiply below (which is already happening)
    // and `-c_j/rad_j` rides the intermediate bias, which is already a slot
    // add. So a mode plan costs one evaluation a MODE and no mask at all.
    const bool use_modes = !c.gelu_modes.empty();
    std::unique_ptr<GeLuHandler<word>> mode_gelu;
    if (use_modes) {
      AssertTrue(static_cast<int>(c.gelu_rad.size()) >= cfg_.hidden_live &&
                     static_cast<int>(c.gelu_centre.size()) >=
                         cfg_.hidden_live,
                 "CiBertLayer: a mode plan needs gelu_rad and gelu_centre");
      std::vector<typename GeLuHandler<word>::Mode> ms(c.gelu_modes.size());
      for (size_t r = 0; r < c.gelu_modes.size(); r++) {
        ms[r].coeffs = c.gelu_modes[r].coeffs;
      }
      mode_gelu = std::make_unique<GeLuHandler<word>>(boot_, ms, op_level_);
    }
    std::vector<typename GeLuHandler<word>::Group> groups;
    if (!c.gelu_bands.empty()) {
      // The CERTIFIED plan: every band a fit over an interval `u` cannot
      // leave. No identity and no zero group, because those two are the
      // only ones whose failure is a wrong answer rather than a bounded
      // one, and their assignment was the per-prompt part.
      for (const auto &b : c.gelu_bands) {
        groups.push_back({GeLuHandler<word>::Kind::kFit, b.range, b.degree,
                          b.coeffs});
      }
    } else {
      groups = {{GeLuHandler<word>::Kind::kFit, c.gelu_range, c.gelu_degree},
                {GeLuHandler<word>::Kind::kIdentity, c.gelu_range, 0},
                {GeLuHandler<word>::Kind::kZero, c.gelu_range, 0}};
      if (c.gelu_wide_range > 0.0) {
        groups.push_back({GeLuHandler<word>::Kind::kFit, c.gelu_wide_range,
                          c.gelu_wide_degree});
      }
    }
    const int num_groups = static_cast<int>(groups.size());
    GeLuHandler<word> gelu(boot_, groups, op_level_);
    // The caller divides by the handler's OWN range; a band whose range
    // differs recovers it through its mask (`GeLu.h`). A mode plan divides
    // per CHANNEL instead, so its "range" is one.
    const double gelu_range = use_modes ? 1.0 : gelu.GetRange();
    const int log_t = Log2Ceil(cfg_.num_tokens);
    const int rank = cfg_.proj_rank;
    std::vector<std::vector<Complex>> mask(num_groups);
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
          1.0 / (crossing_ * c.int_scale * out_scale * gelu_range);
      if (use_modes) {
        // The per-CHANNEL divide, on the multiply the crossing is already
        // paying for. The row suppression, when there is one, simply
        // multiplies it -- both are public and both are per slot.
        std::vector<Complex> msg(num_slots_, Complex(0.0, 0.0));
        for (int ch = 0; ch < rank; ch++) {
          const int declared = i * rank + ch;
          const double r = declared < cfg_.hidden_live
                               ? base / c.gelu_rad[declared]
                               : 0.0;
          for (int t = 0; t < cfg_.num_tokens; t++) {
            const double sup =
                c.row_suppress.empty() ? 1.0 : 1.0 / c.row_suppress[t];
            msg[static_cast<size_t>(ch) * cfg_.num_tokens + Rev(t, log_t)] =
                Complex(r * sup, 0.0);
          }
        }
        // The scale is `TokenPlaintext`'s: the multiply and the rescale
        // together have to land the stream back on `op_level_`'s canonical
        // scale, so the plaintext carries
        // `GetScale(op) * RescalePrimeProd(slot) / ct.GetScale()`.
        Pt pt;
        boot_->gpu_encoder_.Encode(
            pt, slot_level_,
            boot_->param_.GetScale(op_level_) *
                boot_->param_.GetRescalePrimeProd(slot_level_) /
                ups[i].GetScale(),
            msg);
        Canonicalise(ups[i], pt);
      } else if (c.row_suppress.empty()) {
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
          const double b =
              declared >= cfg_.hidden_live
                  ? 0.0
                  : (use_modes ? ((*w.bint)[declared] - c.gelu_centre[declared]) /
                                     c.gelu_rad[declared]
                               : (*w.bint)[declared] / gelu_range);
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
      if (use_modes) {
        // One weight vector a mode, for THIS ciphertext's channels. The
        // weights are per channel and constant along the token axis --
        // there is nothing per prompt anywhere in a mode plan.
        std::vector<std::vector<Complex>> wt(c.gelu_modes.size());
        for (size_t r = 0; r < c.gelu_modes.size(); r++) {
          wt[r].assign(num_slots_, Complex(0.0, 0.0));
          for (int ch = 0; ch < rank; ch++) {
            const int declared = i * rank + ch;
            if (declared >= cfg_.hidden_live) continue;
            const double b = c.gelu_modes[r].weight[declared];
            for (int t = 0; t < cfg_.num_tokens; t++) {
              wt[r][static_cast<size_t>(ch) * cfg_.num_tokens + Rev(t, log_t)] =
                  Complex(b, 0.0);
            }
          }
        }
        mode_gelu->ApplyModes(act[i], ups[i], wt, evk);
      } else {
        for (int gsel = 0; gsel < num_groups; gsel++) {
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
              if (which >= num_groups) which = 0;
            }
            mask[which][s] = Complex(1.0, 0.0);
          }
        }
        gelu.Apply(act[i], ups[i], mask, evk);
      }
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
    NormStages ffn_stages;
    ffn_stages.crude_degree = c.ffn_crude_degree;
    ffn_stages.refine_window = c.ffn_refine_window;
    ffn_stages.crude_invsqrt = &c.ffn_crude_invsqrt;
    ffn_stages.refine_invsqrt = &c.ffn_refine_invsqrt;
    ffn_stages.channel_suppress = &c.channel_suppress;
    NormTurn(res, z, *w.ffn_gain, *w.ffn_bias, c.ffn_alpha, c.ffn_window,
             c.ffn_degree, c.ffn_invsqrt, c.ffn_scale, out_scale, out_scale,
             {}, evk, &ffn_stages);
  }
  MemoryPool::Report("bert: after the output LayerNorm");
}

// ---------------------------------------------------------------------------
// The three prediction heads
// ---------------------------------------------------------------------------

template <typename word>
void CiBertLayer<word>::Pooler(std::vector<Ct> &logits,
                               const std::vector<Ct> &stream,
                               const HeadWeights &w, const HeadCalibration &c,
                               const EvkMap<word> &evk) {
  NvtxScope _nv("bert: pooler + NSP classifier");
  AssertTrue(w.pool.Given() && w.cls.Given(),
             "CiBertLayer: the pooler and the classifier must be given");
  AssertTrue(!w.tag.empty(), "CiBertLayer: a head needs a tag");
  AssertTrue(c.tanh_degree >= 2 && !c.tanh_coeffs.empty(),
             "CiBertLayer: the pooler needs a tanh fit of degree >= 2");
  AssertTrue(static_cast<int>(stream.size()) == num_model_cts_,
             "CiBertLayer: the head reads " + std::to_string(num_model_cts_) +
                 " ciphertexts");

  // ---- x W_p + b_p, every token at once ----------------------------------
  std::vector<Ct> u;
  {
    std::vector<Ct> ins(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->LevelDown(ins[k], stream[k], cfg_.product_level);
    }
    base_.Project(u, ins, cfg_.model_declared, cfg_.model_declared, w.pool,
                  c.pool_scale, (w.tag + ".pool").c_str(), 1, 1);
  }

  // ---- the crossing, the bias, tanh --------------------------------------
  std::vector<Ct> p(num_model_cts_);
  {
    std::vector<Ct> us;
    {
      std::vector<const Ct *> xs(num_model_cts_);
      for (int k = 0; k < num_model_cts_; k++) xs[k] = &u[k];
      sched_.ToSlotBatch(us, xs, evk, /*min_ks=*/false);
      u.clear();
    }
    std::vector<typename GeLuHandler<word>::Group> groups{
        {GeLuHandler<word>::Kind::kFit, c.tanh_range, c.tanh_degree,
         c.tanh_coeffs}};
    GeLuHandler<word> tanh_h(boot_, groups, op_level_);
    const double range = tanh_h.GetRange();
    // The projection's output carries `pool_scale * stream_scale` per model
    // unit and `ToSlot` has just multiplied by `crossing`; what is left after
    // this multiply is `u / range`, which is what the fit takes. The stream's
    // per-channel suppression is gone by here -- it lives on the INPUT axis
    // and `w.pool` was folded with it, so `u` is already in model units.
    const double base =
        1.0 / (crossing_ * c.pool_scale * c.stream_scale * range);
    const int rank = cfg_.proj_rank;
    for (int i = 0; i < num_model_cts_; i++) {
      Canonicalise(us[i], base);
      if (w.pool_bias != nullptr) {
        std::vector<Complex> bmsg(num_slots_, Complex(0.0, 0.0));
        for (int ch = 0; ch < rank; ch++) {
          const int declared = i * rank + ch;
          const double b = declared >= cfg_.model_live
                               ? 0.0
                               : (*w.pool_bias)[declared] / range;
          for (int t = 0; t < cfg_.num_tokens; t++) {
            bmsg[static_cast<size_t>(ch) * cfg_.num_tokens + t] =
                Complex(b, 0.0);
          }
        }
        Pt bpt;
        boot_->gpu_encoder_.Encode(bpt, op_level_,
                                   boot_->param_.GetScale(op_level_), bmsg);
        boot_->Add(us[i], us[i], bpt);
      }
      // One group, so no mask: `Apply` reads an empty vector as "all of it".
      tanh_h.Apply(p[i], us[i], {}, evk);
      us[i] = Ct{};
    }
  }
  MemoryPool::Report("bert: after the pooler's tanh");

  // ---- p W_c + b_c -------------------------------------------------------
  {
    std::vector<Ct> ins(num_model_cts_);
    for (int i = 0; i < num_model_cts_; i++) {
      Ct co;
      sched_.ToCoeff(co, p[i], evk, /*min_ks=*/false);
      boot_->LevelDown(ins[i], co, cfg_.product_level);
      p[i] = Ct{};
    }
    base_.Project(logits, ins, cfg_.model_declared, cfg_.proj_rank, w.cls,
                  c.cls_scale, (w.tag + ".cls").c_str(), 1, 1);
    // `ToCoeff` multiplied by `kappa`, so the classifier's input carries it
    // and its bias has to as well.
    if (w.cls_bias != nullptr) {
      AddBias(logits, *w.cls_bias, c.cls_scale * kappa_);
    }
  }
  MemoryPool::Report("bert: after the NSP classifier");
}

template <typename word>
void CiBertLayer<word>::MlmHead(std::vector<Ct> &logits,
                                const std::vector<Ct> &stream,
                                const HeadWeights &w, const HeadCalibration &c,
                                const EvkMap<word> &evk,
                                std::vector<Ct> *norm_out) {
  NvtxScope _nv("bert: MLM head");
  AssertTrue(w.mlm.Given() && w.dec.Given() && w.mlm_gain != nullptr &&
                 w.mlm_norm_bias != nullptr,
             "CiBertLayer: the MLM transform, its norm and the tied decoder "
             "must be given");
  AssertTrue(!w.tag.empty(), "CiBertLayer: a head needs a tag");
  AssertTrue(c.gelu_degree >= 2 && !c.gelu_coeffs.empty(),
             "CiBertLayer: the MLM head needs a GELU fit of degree >= 2");
  AssertTrue(c.vocab_declared > 0 && c.vocab_declared % cfg_.proj_rank == 0,
             "CiBertLayer: the vocabulary must be declared at a multiple of "
             "the module rank");
  AssertTrue(static_cast<int>(stream.size()) == num_model_cts_,
             "CiBertLayer: the head reads " + std::to_string(num_model_cts_) +
                 " ciphertexts");

  // ---- x W_t + b_t -------------------------------------------------------
  std::vector<Ct> u;
  {
    std::vector<Ct> ins(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->LevelDown(ins[k], stream[k], cfg_.product_level);
    }
    base_.Project(u, ins, cfg_.model_declared, cfg_.model_declared, w.mlm,
                  c.mlm_scale, (w.tag + ".mlm").c_str(), 1, 1);
  }

  // ---- the crossing, the bias, GELU --------------------------------------
  std::vector<Ct> g(num_model_cts_);
  {
    std::vector<Ct> us;
    {
      std::vector<const Ct *> xs(num_model_cts_);
      for (int k = 0; k < num_model_cts_; k++) xs[k] = &u[k];
      sched_.ToSlotBatch(us, xs, evk, /*min_ks=*/false);
      u.clear();
    }
    // The answer's ride rides the FIT (see `gelu_out_scale`): the handler
    // evaluates the vector it is handed, so a constant factor on it is free.
    std::vector<double> fit(c.gelu_coeffs);
    if (c.gelu_out_scale != 1.0) {
      for (double &x : fit) x *= c.gelu_out_scale;
    }
    std::vector<typename GeLuHandler<word>::Group> groups{
        {GeLuHandler<word>::Kind::kFit, c.gelu_range, c.gelu_degree, fit}};
    GeLuHandler<word> gelu(boot_, groups, op_level_);
    const double range = gelu.GetRange();
    const double base =
        1.0 / (crossing_ * c.mlm_scale * c.stream_scale * range);
    const int rank = cfg_.proj_rank;
    for (int i = 0; i < num_model_cts_; i++) {
      Canonicalise(us[i], base);
      if (w.mlm_bias != nullptr) {
        std::vector<Complex> bmsg(num_slots_, Complex(0.0, 0.0));
        for (int ch = 0; ch < rank; ch++) {
          const int declared = i * rank + ch;
          const double b = declared >= cfg_.model_live
                               ? 0.0
                               : (*w.mlm_bias)[declared] / range;
          for (int t = 0; t < cfg_.num_tokens; t++) {
            bmsg[static_cast<size_t>(ch) * cfg_.num_tokens + t] =
                Complex(b, 0.0);
          }
        }
        Pt bpt;
        boot_->gpu_encoder_.Encode(bpt, op_level_,
                                   boot_->param_.GetScale(op_level_), bmsg);
        boot_->Add(us[i], us[i], bpt);
      }
      gelu.Apply(g[i], us[i], {}, evk);
      us[i] = Ct{};
    }
  }
  MemoryPool::Report("bert: after the MLM transform's GELU");

  // ---- its LayerNorm, and the tied decoder that reads it -----------------
  {
    std::vector<Ct> gc(num_model_cts_);
    for (int i = 0; i < num_model_cts_; i++) {
      sched_.ToCoeff(gc[i], g[i], evk, /*min_ks=*/false);
      g[i] = Ct{};
    }
    // `ToCoeff` multiplied by `kappa`, so that is what the norm's input
    // carries per model unit. There is NO channel suppression here: the
    // stream's lives on the encoder's residual and the transform above has
    // already taken it out on the input axis.
    NormStages mlm_stages;
    mlm_stages.crude_degree = c.mlm_crude_degree;
    mlm_stages.refine_window = c.mlm_refine_window;
    mlm_stages.crude_invsqrt = &c.mlm_crude_invsqrt;
    mlm_stages.refine_invsqrt = &c.mlm_refine_invsqrt;
    std::vector<Ct> n;
    NormTurn(n, gc, *w.mlm_gain, *w.mlm_norm_bias, c.mlm_alpha, c.mlm_window,
             c.mlm_degree, c.mlm_invsqrt, c.mlm_norm_token_scale,
             kappa_ * c.gelu_out_scale,
             c.mlm_out_scale, {}, evk,
             c.mlm_crude_degree > 0 ? &mlm_stages : nullptr);
    std::vector<Ct> ins(num_model_cts_);
    for (int i = 0; i < num_model_cts_; i++) {
      boot_->LevelDown(ins[i], n[i], cfg_.product_level);
    }
    if (norm_out != nullptr) *norm_out = std::move(n);
    n.clear();
    base_.Project(logits, ins, cfg_.model_declared, c.vocab_declared, w.dec,
                  c.dec_scale, (w.tag + ".dec").c_str(), 1, 1);
    AssertTrue(static_cast<int>(logits.size()) ==
                   c.vocab_declared / cfg_.proj_rank,
               "CiBertLayer: the tied decoder did not land in " +
                   std::to_string(c.vocab_declared / cfg_.proj_rank) +
                   " ciphertexts");
    if (w.dec_bias != nullptr) {
      AddBias(logits, *w.dec_bias, c.dec_scale * c.mlm_out_scale);
    }
  }
  MemoryPool::Report("bert: after the tied decoder");
}

template class CiBertLayer<uint32_t>;
template class CiBertLayer<uint64_t>;

}  // namespace cheddar
