#include "extension/CiLlamaLayer.h"

#include <algorithm>
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
  RmsNormHandler<word> probe(boot_, cfg_.num_tokens, cfg_.model_declared, 1.0,
                             op_level_, 1e-5, 2.0, cfg_.rms_degree,
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
  std::vector<Ct> normed(num_model_cts_);
  {
    std::vector<Ct> slots(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      sched_.ToSlot(slots[k], h_ct[k], evk, cfg_.min_ks);
      // The residual carries the O projection's own factor at BOTH ends -- the
      // stream was encrypted with it and the O output already has it -- so
      // what is divided out here is the crossing ALONE. A fit measured on this
      // ciphertext instead would be right here and wrong at the gate's
      // crossing, which carries no such factor (1.5cu).
      Canonicalise(slots[k], 1.0 / crossing_);
    }

    // RMSNorm DIVIDES BY THE WIDTH IT IS TOLD, and that is the declared one.
    // Llama divides by `model_live`, so the two scalings below cancel exactly:
    // `eps * live / declared` makes the bracket `(live/declared)(S/live + eps)`
    // and `alpha * declared / live` puts its geometric mean back at 1. The
    // leftover `sqrt(declared/live)` is taken out by the weight, which already
    // carries `sqrt(alpha)`.
    const double ratio =
        static_cast<double>(cfg_.model_declared) / cfg_.model_live;
    RmsNormHandler<word> rms(boot_, cfg_.num_tokens, cfg_.model_declared,
                             c.alpha * ratio, op_level_, cfg_.eps / ratio,
                             c.norm_window, NormDegree(c.norm_window),
                             /*channel_stride=*/2);
    AssertTrue(rms.GetNumCiphertexts() == num_model_cts_,
               "CiLlamaLayer: RmsNormHandler disagrees about the stream width");
    const auto wts = NormWeights(*w.ffn_norm, c.alpha);
    std::vector<Ct> outv;
    rms.Apply(outv, slots, wts, evk);
    for (int k = 0; k < num_model_cts_; k++) {
      sched_.ToCoeff(normed[k], outv[k], evk, cfg_.min_ks);
    }
  }

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
    leg_->Project(res, ins, cfg_.hidden_declared, cfg_.model_declared, *w.down,
                  1.0, (w.tag + ".down").c_str());
    AssertTrue(static_cast<int>(res.size()) == num_model_cts_,
               "CiLlamaLayer: the down projection did not land in " +
                   std::to_string(num_model_cts_) + " ciphertexts");
  }
}

template class CiProjectionLeg<uint32_t>;
template class CiProjectionLeg<uint64_t>;
template class CiLlamaLayer<uint32_t>;
template class CiLlamaLayer<uint64_t>;

}  // namespace cheddar
