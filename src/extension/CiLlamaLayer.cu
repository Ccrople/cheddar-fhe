#include "extension/CiLlamaLayer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

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

// The projection leg with its ciphertext-ciphertext half stubbed out. On R+
// those are `CiSinCAttention`'s, by a different road entirely -- SinC operands,
// the ring switch, the lifted product -- so a leg reached from here can only
// ever be asked for `Project`, and saying so is better than a silent fallback.
template <typename word>
class CiProjectionLeg : public CoeffLinearLeg<word> {
 public:
  using CoeffLinearLeg<word>::CoeffLinearLeg;
  void Scores(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double,
              const std::vector<double> &) const override {
    Fail("CiProjectionLeg: the CI score product is CiSinCAttention's");
  }
  void Values(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double) const override {
    Fail("CiProjectionLeg: the CI value product is CiSinCAttention's");
  }
  void LocateScore(int, int, int, int &, int &) const override {
    Fail("CiProjectionLeg: the CI score layout is CiSinCAttention's");
  }
};

template <typename word>
CiLlamaLayer<word>::CiLlamaLayer(
    std::shared_ptr<const BootContext<word>> boot,
    const CiSwitchedCcmmLayout &layout,
    std::vector<const EvaluationKey<word> *> modpack_keys, const Config &cfg)
    : boot_{std::move(boot)},
      cfg_{cfg},
      sched_{boot_, boot_->param_.MaxNumSlots()} {
  num_slots_ = boot_->param_.MaxNumSlots();
  attn_channels_ = 2 * layout.num_cts * cfg_.proj_rank;

  AssertTrue(cfg_.model_declared % cfg_.proj_rank == 0 &&
                 cfg_.hidden_declared % cfg_.proj_rank == 0,
             "CiLlamaLayer: both declared widths must be a whole number of "
             "half-density ciphertexts");
  num_model_cts_ = cfg_.model_declared / cfg_.proj_rank;
  num_hidden_cts_ = cfg_.hidden_declared / cfg_.proj_rank;
  // A half-density ciphertext carries `rank/2 - 1` live model channels --
  // component zero has no partner -- and `rank/2` hidden ones, because
  // nothing reduces over the hidden axis. Seventeen of the first hold Llama's
  // 4096; sixteen do not.
  AssertTrue(num_model_cts_ * (cfg_.proj_rank / 2 - 1) >= cfg_.model_live,
             "CiLlamaLayer: " + std::to_string(num_model_cts_) +
                 " half-density ciphertexts carry " +
                 std::to_string(num_model_cts_ * (cfg_.proj_rank / 2 - 1)) +
                 " live model channels, short of " +
                 std::to_string(cfg_.model_live) +
                 " -- component zero has no partner");

  // THE CROSSING CONSTANT IS DERIVED, NOT FITTED. `HalfBoot` multiplies the
  // message by `level_zero_scale / q0` and `BootContext` computes both halves
  // already; the nominal `2^-log_message_ratio` is what the design ASKS for
  // and differs from what it gets by the rounding in `log_scaleup_`
  // (Doing.md 1.5dk). A constant fitted on one preset and carried to another
  // is wrong by percents.
  crossing_ = boot_->GetMessageRatio();
  AssertTrue(crossing_ > 0.0,
             "CiLlamaLayer: PrepareEvalMod must run before construction");
  // What a FULL turn carries: `ToCoeff` undoes the crossing by the NOMINAL
  // ratio, deliberately, because that is what makes `Boot` message
  // preserving. Every linear stage absorbs the difference and RMSNorm is
  // scale invariant, so it reaches SiLU unchallenged -- and
  // `SiLU(kx)/k - SiLU(x)` is not noise, it is a different function.
  kappa_ =
      std::pow(2.0, -boot_->GetBootParameter().GetLogMessageRatio()) / crossing_;

  slot_level_ = sched_.GetSlotLevel();
  op_level_ = slot_level_ - 1;

  typename CiLlamaSeam<word>::Config scfg;
  scfg.proj_rank = cfg_.proj_rank;
  scfg.verbose = cfg_.verbose;
  seam_ = std::make_unique<CiLlamaSeam<word>>(boot_, layout,
                                              sched_.GetStCLevel(), scfg);

  typename CoeffLinearLeg<word>::Config lcfg;
  lcfg.num_tokens = cfg_.num_tokens;
  lcfg.product_level = cfg_.product_level;
  lcfg.parents_per_tile = cfg_.parents_per_tile;
  // HALF DENSITY ON BOTH AXES, which is a statement about these operands and
  // not a tuning choice. Every parent here is a banded half-density image, so
  // its live module components are a contiguous prefix and the descent stops
  // there; every output is one too, so half of `GatherWeights`'s rows are
  // exact zeros that the operand would store, the GEMM would multiply and
  // `ModPack` would recompose. Measured exact at both settings and 1.9991x /
  // 1.69x apart in work (Doing.md 1.5db/1.5dh).
  lcfg.input_density = 2;
  lcfg.output_density = 2;
  leg_ = std::make_unique<CiProjectionLeg<word>>(boot_, lcfg,
                                                 std::move(modpack_keys));

  if (cfg_.verbose) {
    std::cout << "CiLlamaLayer: slot " << sched_.GetSlotLevel() << ", StC "
              << sched_.GetStCLevel() << ", coeff " << sched_.GetCoeffLevel()
              << ", seam input at " << seam_->GetInputLevel()
              << ", crossing " << crossing_ << " (2^"
              << std::log2(std::abs(crossing_)) << "), kappa " << kappa_
              << std::endl;
  }
}

template <typename word>
void CiLlamaLayer<word>::AddRequiredRotations(EvkRequest &req) const {
  seam_->AddRequiredRotations(req);
  // RMSNorm's distances are a property of the SHAPE -- the reduction tree over
  // `num_channels` at `channel_stride` -- while its `alpha` and window are
  // per-layer calibration that does not outlive one `FeedForward`. So a
  // handler is built here only to be asked what it will rotate by.
  // A concrete degree, not `cfg_.rms_degree`: zero there means "derive it from
  // the window", and the window is per-layer calibration that does not exist
  // yet. The rotation distances do not depend on either.
  RmsNormHandler<word> probe(boot_, cfg_.num_tokens, cfg_.model_declared, 1.0,
                             op_level_, 1e-5, 2.0, NormDegree(2.0),
                             /*channel_stride=*/2);
  for (int d : probe.GetRotationDistances()) req.AddRequest(d, op_level_);
}

template <typename word>
void CiLlamaLayer<word>::PrepareSeamHalf(int half) {
  seam_->PrepareHalf(half);
}

template <typename word>
void CiLlamaLayer<word>::AddSeamHalfRotations(EvkRequest &req) const {
  seam_->AddHalfRotations(req);
}

template <typename word>
void CiLlamaLayer<word>::DropSeamHalf() {
  seam_->DropHalf();
}

template <typename word>
void CiLlamaLayer<word>::Seam(Ct &res, const Ct &booted,
                              const EvkMap<word> &evk) {
  seam_->Apply(res, booted, sched_, evk, cfg_.min_ks);
}

template <typename word>
int CiLlamaLayer<word>::NormDegree(double window) const {
  if (cfg_.rms_degree > 0) return cfg_.rms_degree;
  // A Chebyshev fit's error is uniform over its interval, so the degree has to
  // follow the window rather than be typed beside it.
  if (window <= 2.5) return 9;
  if (window <= 12.0) return 15;
  return 31;
}

template <typename word>
std::vector<std::vector<Complex>> CiLlamaLayer<word>::NormWeights(
    const std::vector<double> &gain, double alpha) const {
  const int rank = cfg_.proj_rank;
  const int log_rank = Log2Ceil(rank);
  const int log_t = Log2Ceil(cfg_.num_tokens);
  (void)log_t;
  const double root_alpha = std::sqrt(alpha);

  std::vector<std::vector<Complex>> wts(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    wts[k].assign(num_slots_, Complex(0.0, 0.0));
    for (int c = 0; c < rank; c++) {
      const int I = Rev(c, log_rank);
      // AT AN ODD DECLARED CHANNEL THE WEIGHT IS THE PARTNER'S. The duplicate
      // band at channel `c` (odd) holds component `rank - I`, whose live
      // address is `Rev(rank - I)`; the reduction at `channel_stride = 2` sums
      // the two parities apart, so it must see the same gain at both.
      const int src = (c % 2 == 0) ? c : Rev(rank - I, log_rank);
      const double v = gain[static_cast<size_t>(k) * rank + src];
      for (int t = 0; t < cfg_.num_tokens; t++) {
        wts[k][static_cast<size_t>(c) * cfg_.num_tokens + t] =
            Complex(v * root_alpha, 0.0);
      }
    }
  }
  return wts;
}

template <typename word>
Plaintext<word> CiLlamaLayer<word>::CrossingPlaintext(
    double factor, const std::vector<double> &sink, double at_scale) const {
  AssertTrue(static_cast<int>(sink.size()) == cfg_.num_tokens,
             "CiLlamaLayer: the sink rescale needs one factor per token");
  // The stream's slot address is `channel * num_tokens + rev(token)` (1.5du),
  // so a per-token factor is a stride-`num_tokens` pattern -- but it is NOT
  // the same pattern on both bands. The banded convention is
  // `rec[p*rank + I] = comp_I[p] + [I!=0] comp_{rank-I}[p+1]`, so a DEAD
  // component (declared channel odd, the duplicate half) at position `p`
  // carries its partner's value at position `p + 1`: a token's duplicate sits
  // one position BACK from its live copy. Applied uniformly, this multiply
  // scales token `t`'s duplicate by `sink[t-1]` -- which leaves the live band
  // right and corrupts the duplicate of the first user token by the last
  // sink's factor, measured as live 2^-5.99 against duplicate 2^-1.06 and a
  // layer at relative 265. Token 0 has no duplicate (position -1 is off the
  // image), exactly as the last position has no partner.
  const int log_t = Log2Ceil(cfg_.num_tokens);
  std::vector<Complex> vals(num_slots_, Complex(factor, 0.0));
  for (int t = 0; t < cfg_.num_tokens; t++) {
    if (sink[t] == 1.0) continue;
    const int p = static_cast<int>(BitReverseInt(t, log_t));
    for (int c = 0; p + c * cfg_.num_tokens < num_slots_; c += 2) {
      vals[p + c * cfg_.num_tokens] = Complex(factor * sink[t], 0.0);
    }
    if (t == 0) continue;
    const int q = static_cast<int>(BitReverseInt(t - 1, log_t));
    for (int c = 1; q + c * cfg_.num_tokens < num_slots_; c += 2) {
      vals[q + c * cfg_.num_tokens] = Complex(factor * sink[t], 0.0);
    }
  }
  Plaintext<word> pt;
  boot_->encoder_.Encode(pt, slot_level_,
                         boot_->param_.GetScale(op_level_) *
                             boot_->param_.GetRescalePrimeProd(slot_level_) /
                             at_scale,
                         vals);
  return pt;
}

template <typename word>
void CiLlamaLayer<word>::Canonicalise(Ct &ct, const Plaintext<word> &pt) const {
  boot_->Mult(ct, ct, pt);
  boot_->Rescale(ct, ct);
}

template <typename word>
void CiLlamaLayer<word>::Canonicalise(Ct &ct, double factor) const {
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
void CiLlamaLayer<word>::NormTurn(std::vector<Ct> &res,
                                  const std::vector<Ct> &stream,
                                  const std::vector<double> &gain,
                                  double alpha, double window,
                                  double stream_scale,
                                  const std::vector<double> &sink,
                                  const EvkMap<word> &evk) {
  AssertTrue(static_cast<int>(stream.size()) == num_model_cts_,
             "CiLlamaLayer: the residual stream is " +
                 std::to_string(num_model_cts_) + " ciphertexts");
  std::vector<Ct> slots(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    sched_.ToSlot(slots[k], stream[k], evk, cfg_.min_ks);
    // What is divided out here is the CROSSING ALONE. A residual carries the
    // O projection's own factor at both ends -- the stream was encrypted with
    // it and the O output already has it -- so a fit measured on this
    // ciphertext would be right here and wrong at the gate's crossing, which
    // carries no such factor (1.5cu). And RMSNorm is scale invariant, which is
    // exactly what hid that mistake for a whole increment.
    // THE STREAM FACTOR GOES OUT HERE, WITH THE CROSSING. The residual
    // carries a global factor so that its crossing rides at `Config::ride`,
    // and `RmsNormHandler` wants `alpha * mean(x^2)` near 1 for the `x` it is
    // handed -- so either this multiply takes the factor out and the
    // calibration is the MODEL's, or it does not and every alpha and epsilon
    // downstream has to carry `stream_scale^2`. The first is what the
    // full-width FFN test does (`1 / (boundary * beta)`) and it is far less
    // to get wrong: measured, leaving the factor in put the invsqrt's
    // argument at 0.0038 where its window is [0.77, 1.3], and outside its
    // interval a Chebyshev fit does whatever it likes -- relative 1.1 at the
    // norm, with a fitted factor that moved between identical runs.
    //
    // Everything downstream of RMSNorm is then in MODEL units, because
    // RMSNorm is scale invariant. The two projections that write the stream
    // back -- O and down -- put the factor on again through their weights.
    // THE SINK RESCALE RIDES THIS MULTIPLY. [SYLPH] 3.1.1's prefix is
    // prompt-independent and so public, which is what makes a per-token
    // factor legal here; and it has to happen at EVERY norm, because the
    // stream's sink rows do not stay in range on their own -- layer 1's
    // output carries them at 74327x the user rows' mean square. Folding the
    // factors into the constant this crossing already pays costs no level and
    // no operation. The plaintext is built once, off the first ciphertext's
    // scale, because `ToSlot` leaves all of them at the same one.
    if (sink.empty()) {
      Canonicalise(slots[k], 1.0 / (crossing_ * stream_scale));
    } else {
      if (k == 0) {
        crossing_pt_ = CrossingPlaintext(1.0 / (crossing_ * stream_scale),
                                         sink, slots[0].GetScale());
      }
      Canonicalise(slots[k], crossing_pt_);
    }
  }

  // RMSNorm DIVIDES BY THE WIDTH IT IS TOLD, and that is the DECLARED one.
  // Llama divides by `model_live`, so the two scalings below cancel exactly:
  // `eps * live / declared` makes the bracket `(live/declared)(S/live + eps)`
  // and `alpha * declared / live` puts its geometric mean back at 1. The
  // leftover `sqrt(declared/live)` is taken out by the weight, which already
  // carries `sqrt(alpha)`. Left alone this is not a scale error a fit absorbs:
  // it puts the polynomial's argument at 0.47 instead of 1 and the bottom
  // sixth of the data outside the fitted window (1.5dd, measured 2^-5.09).
  const double ratio =
      static_cast<double>(cfg_.model_declared) / cfg_.model_live;
  // The handler compiles its polynomial here and encodes its weight
  // plaintexts in `Prepare`; both are per-layer preparation, so they are
  // timed apart from the arithmetic below. `Apply` would do the encode on its
  // own at first use, which is what hid it inside the online row.
  const auto prep0 = std::chrono::steady_clock::now();
  RmsNormHandler<word> rms(boot_, cfg_.num_tokens, cfg_.model_declared,
                           alpha * ratio, op_level_, cfg_.eps / ratio, window,
                           NormDegree(window), /*channel_stride=*/2);
  AssertTrue(rms.GetNumCiphertexts() == num_model_cts_,
             "CiLlamaLayer: RmsNormHandler disagrees about the stream width");
  const auto wts = NormWeights(gain, alpha);
  rms.Prepare(wts);
  cudaDeviceSynchronize();
  prepare_seconds_ +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - prep0)
          .count();
  std::vector<Ct> outv;
  rms.Apply(outv, slots, wts, evk);
  res.resize(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    sched_.ToCoeff(res[k], outv[k], evk, cfg_.min_ks);
  }
}

template <typename word>
std::vector<double> CiLlamaLayer<word>::PlainNormInvSqrt(
    double alpha, double window,
    const std::vector<double> &mean_square) const {
  const double ratio =
      static_cast<double>(cfg_.model_declared) / cfg_.model_live;
  RmsNormHandler<word> probe(boot_, cfg_.num_tokens, cfg_.model_declared,
                             alpha * ratio, op_level_, cfg_.eps / ratio,
                             window, NormDegree(window),
                             /*channel_stride=*/2);
  // The bracket the circuit evaluates: the DECLARED width and the scaled
  // epsilon, whose two corrections cancel (see `NormTurn`).
  const double root = std::sqrt(alpha);
  std::vector<double> res(mean_square.size());
  for (size_t i = 0; i < mean_square.size(); i++) {
    const double u =
        alpha * ratio *
        (mean_square[i] * cfg_.model_live / cfg_.model_declared +
         cfg_.eps / ratio);
    res[i] = root * probe.PlainInvSqrt(u);
  }
  return res;
}

template <typename word>
void CiLlamaLayer<word>::Project(std::vector<Ct> &res,
                                 const std::vector<Ct> &x, int in_declared,
                                 int out_declared,
                                 const std::vector<double> &w, double w_scale,
                                 const char *tag) const {
  leg_->Project(res, x, in_declared, out_declared, w, w_scale, tag);
}

template <typename word>
void CiLlamaLayer<word>::AttentionNorm(std::vector<Ct> &res,
                                       const std::vector<Ct> &stream,
                                       const std::vector<double> &gain,
                                       const Calibration &c,
                                       const EvkMap<word> &evk) {
  NormTurn(res, stream, gain, c.attn_alpha, c.attn_norm_window,
           c.stream_scale, c.attn_sink, evk);
}

template <typename word>
void CiLlamaLayer<word>::FeedForward(std::vector<Ct> &res,
                                     const std::vector<Ct> &h_cts,
                                     const std::vector<Ct> &stream,
                                     const Weights &w, const Calibration &c,
                                     const EvkMap<word> &evk) {
  AssertTrue(w.o != nullptr && w.gate != nullptr && w.up != nullptr &&
                 w.down != nullptr && w.ffn_norm != nullptr,
             "CiLlamaLayer: every weight must be given");
  AssertTrue(!w.tag.empty(),
             "CiLlamaLayer: a layer's weights need a tag -- the projection "
             "leg caches by name and a repeated name with different weights "
             "is a wrong layer that still decrypts");
  AssertTrue(static_cast<int>(h_cts.size()) * cfg_.proj_rank == attn_channels_,
             "CiLlamaLayer: the seam did not hand over the whole attention "
             "output");
  AssertTrue(static_cast<int>(stream.size()) == num_model_cts_,
             "CiLlamaLayer: the residual stream is " +
                 std::to_string(num_model_cts_) + " ciphertexts");

  const auto &p = boot_->param_;

  // ---- the O projection, then the residual -------------------------------
  std::vector<Ct> h_ct(num_model_cts_);
  {
    std::vector<Ct> ins(h_cts.size());
    for (size_t k = 0; k < h_cts.size(); k++) {
      boot_->LevelDown(ins[k], h_cts[k], cfg_.product_level);
    }
    std::vector<Ct> out;
    leg_->Project(out, ins, attn_channels_, cfg_.model_declared, *w.o,
                  c.res_scale, (w.tag + ".o").c_str());
    AssertTrue(static_cast<int>(out.size()) == num_model_cts_,
               "CiLlamaLayer: the O projection did not land in " +
                   std::to_string(num_model_cts_) + " ciphertexts");
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->Add(h_ct[k], stream[k], out[k]);
    }
  }

  // ---- the crossing, RMSNorm, and back to coefficients --------------------
  std::vector<Ct> normed;
  NormTurn(normed, h_ct, *w.ffn_norm, c.alpha, c.norm_window,
           c.stream_scale, c.ffn_sink, evk);

  // ---- gate and up -------------------------------------------------------
  std::vector<Ct> gate, upv;
  {
    std::vector<Ct> ins(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->LevelDown(ins[k], normed[k], cfg_.product_level);
    }
    leg_->Project(gate, ins, cfg_.model_declared, cfg_.hidden_declared,
                  *w.gate, c.gate_scale, (w.tag + ".gate").c_str());
    leg_->Project(upv, ins, cfg_.model_declared, cfg_.hidden_declared, *w.up,
                  c.gate_scale, (w.tag + ".up").c_str());
  }
  AssertTrue(static_cast<int>(gate.size()) == num_hidden_cts_ &&
                 static_cast<int>(upv.size()) == num_hidden_cts_,
             "CiLlamaLayer: the gate and up projections did not land in " +
                 std::to_string(num_hidden_cts_) + " ciphertexts");

  // ---- SiLU and the gate multiply ----------------------------------------
  std::vector<Ct> prod(num_hidden_cts_);
  {
    SiLuHandler<word> silu(boot_, c.silu_range, op_level_, cfg_.silu_degree);
    for (int i = 0; i < num_hidden_cts_; i++) {
      Ct g_up, u_up, sv, u_low;
      sched_.ToSlot(g_up, gate[i], evk, cfg_.min_ks);
      sched_.ToSlot(u_up, upv[i], evk, cfg_.min_ks);
      // `crossing_`, NOT a fit taken on the residual: these carry no O factor.
      // And `kappa_` beside it, which is the same mistake one turn further out
      // -- see the class comment.
      Canonicalise(g_up,
                   1.0 / (kappa_ * crossing_ * c.gate_scale * c.silu_range));
      Canonicalise(u_up, 1.0 / (kappa_ * crossing_ * c.gate_scale));
      silu.Apply(sv, g_up, evk);
      boot_->LevelDown(u_low, u_up, p.NPToLevel(sv.GetNP()));
      boot_->HMult(prod[i], sv, u_low, evk.GetMultiplicationKey());
    }
  }

  // ---- the down projection ------------------------------------------------
  {
    std::vector<Ct> ins(num_hidden_cts_);
    for (int i = 0; i < num_hidden_cts_; i++) {
      Ct c2;
      sched_.ToCoeff(c2, prod[i], evk, cfg_.min_ks);
      boot_->LevelDown(ins[i], c2, cfg_.product_level);
    }
    std::vector<Ct> y;
    // `stream_scale`, not 1: RMSNorm is scale invariant, so `y` comes back in
    // the model's own units while `h_ct` carries the stream's factor, and the
    // two cannot be added until they agree. The weight is a plaintext, so
    // putting it back costs nothing.
    leg_->Project(y, ins, cfg_.hidden_declared, cfg_.model_declared, *w.down,
                  c.stream_scale, (w.tag + ".down").c_str());
    AssertTrue(static_cast<int>(y.size()) == num_model_cts_,
               "CiLlamaLayer: the down projection did not land in " +
                   std::to_string(num_model_cts_) + " ciphertexts");
    // THE SECOND RESIDUAL. The correctness-width layer test stopped at the
    // down projection and compared against the down projection, so it never
    // needed this; a layer that feeds its successor does.
    res.resize(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->Add(res[k], h_ct[k], y[k]);
    }
  }
}

template class CiProjectionLeg<uint32_t>;
template class CiProjectionLeg<uint64_t>;
template class CiLlamaLayer<uint32_t>;
template class CiLlamaLayer<uint64_t>;

}  // namespace cheddar
