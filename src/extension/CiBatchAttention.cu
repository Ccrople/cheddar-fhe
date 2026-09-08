#include "extension/CiBatchAttention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

namespace {

// The layout the attention runs on: chain-addressed by default, the plain
// map `Slot(t, b) = t * B + b` when the config asks for it. A function
// because the member initialiser list cannot branch between two
// constructors.
CiBatchLayout MakeBatchLayout(int num_slots, int num_tokens, int lanes,
                              int rank, bool plain) {
  if (plain) return CiBatchLayout(num_slots, num_tokens);
  return CiBatchLayout(num_slots, num_tokens, lanes, rank);
}

}  // namespace

template <typename word>
CiBatchAttention<word>::CiBatchAttention(
    std::shared_ptr<const BootContext<word>> boot,
    ConstContextPtr<word> switch_ctx, ConstContextPtr<word> small_ctx,
    ConstContextPtr<word> lifted_ctx, const Config &cfg,
    std::shared_ptr<const BootContext<word>> tower /*= nullptr*/)
    : boot_{std::move(boot)},
      switch_ctx_{std::move(switch_ctx)},
      small_ctx_{std::move(small_ctx)},
      lifted_ctx_{std::move(lifted_ctx)},
      cfg_{cfg},
      chain_{switch_ctx_->param_.degree_, small_ctx_->param_.degree_,
             cfg.sub_degree},
      layout_{MakeBatchLayout(boot_->param_.MaxNumSlots(), cfg.num_tokens,
                              cfg.sub_degree, chain_.rank, cfg.plain_map)},
      switcher_{switch_ctx_, small_ctx_},
      lift_{small_ctx_, lifted_ctx_},
      ccmm_{lifted_ctx_->param_, lifted_ctx_->ntt_handler_},
      tower_{std::move(tower)} {
  {
    // The softmax's exp is one polynomial over `num_tokens` ciphertexts that
    // differ only in their data -- exactly the shape `EvaluateBatch` is for.
    // The knob bounds the live basis, and <= 1 restores the per-ciphertext
    // loop for an A/B.
    const char *e = std::getenv("CHEDDAR_BATCH_SOFTMAX_EXP");
    if (e != nullptr && e[0] != '\0') exp_batch_ = std::atoi(e);
  }
  AssertTrue(boot_->param_.degree_ == switch_ctx_->param_.degree_,
             "CiBatchAttention: the switching ring must have the layer's "
             "degree");
  AssertTrue(chain_.dim == cfg_.num_tokens,
             "CiBatchAttention: the product ring must hold exactly one "
             "token per block -- d = " +
                 std::to_string(chain_.dim) + " against T = " +
                 std::to_string(cfg_.num_tokens));
  // NOT the CC-MM's constraint any more. `BatchCcmm::Multiply` contracts a
  // rectangular inner dimension (BatchCcmmContractsARectangularInner), and
  // `Scores` already passes K row-wise, so the score product would take any
  // head_dim. What still pins it is `Values`: its V is one ciphertext a
  // CHANNEL, so it goes to Algorithm 4 column-wise and step 1's CMT
  // transposes it -- and [KANG] Algorithm 3 is square by construction,
  // exactly `degree / sub_degree` ciphertexts. Freeing head_dim means handing
  // `Values` a row-wise V, which is the transpose CMT was there to do.
  AssertTrue(cfg_.head_dim == cfg_.num_tokens,
             "CiBatchAttention: Values hands V to Algorithm 4 column-wise, so "
             "step 1's square CMT pins head_dim to the token count");
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
  if (!cfg_.plain_map) {
    for (int t = 0; t < T / 2; t++) {
      for (int g = 0; g < chain_.rank; g++) {
        AssertTrue(layout_.BlockOf(t + T / 2, g) == layout_.BlockOf(t, g) + 1,
                   "CiBatchAttention: the key-token shift is not one block");
      }
    }
  } else {
    // The plain map's own version of the same claim: the token is the SLOW
    // axis, so the shift is `T/2 * B` slots -- one rotation, and still
    // lane-preserving so nothing downstream notices its size.
    const int r = GetShiftRotation();
    for (int t = 0; t < T / 2; t++) {
      for (int b = 0; b < layout_.num_instances; b++) {
        AssertTrue(layout_.Slot(t + T / 2, b) - layout_.Slot(t, b) == r,
                   "CiBatchAttention: the key-token shift is not one "
                   "rotation under the plain map");
      }
    }
    AssertTrue(r % cfg_.sub_degree == 0,
               "CiBatchAttention: the key-token shift must preserve lanes");
  }

  // The premap the plain map needs, built from the two layouts themselves so
  // that a layout change cannot silently disagree with a transcription of it.
  // `premap_[b]` is the block the CHAIN addressing holds what the plain map
  // holds at block `b`; lanes -- the low log2(sub_degree) slot bits -- are
  // untouched, which is the contract `forward_premap` states. The same vector
  // serves the inverse, a block bit reversal being its own inverse.
  const std::vector<int> *premap = nullptr;
  if (cfg_.plain_map) {
    AssertTrue(!cfg_.fused_scores,
               "CiBatchAttention: the tower's lane prefix mixes only within a "
               "lane group, so it cannot carry a block premap (63 -> 15309 "
               "diagonals) -- the plain map's scores must come back through "
               "the inverse converter");
    const CiBatchLayout chain_layout(layout_.num_slots, T, cfg_.sub_degree,
                                     chain_.rank);
    const int num_blocks = layout_.num_slots / cfg_.sub_degree;
    premap_.assign(num_blocks, -1);
    for (int t = 0; t < T; t++) {
      for (int b = 0; b < layout_.num_instances; b++) {
        const int ps = layout_.Slot(t, b), cs = chain_layout.Slot(t, b);
        AssertTrue(ps % cfg_.sub_degree == cs % cfg_.sub_degree,
                   "CiBatchAttention: the premap is not lane-preserving");
        const int pb = ps / cfg_.sub_degree, cb = cs / cfg_.sub_degree;
        AssertTrue(premap_[pb] < 0 || premap_[pb] == cb,
                   "CiBatchAttention: the premap is not well defined");
        premap_[pb] = cb;
      }
    }
    std::vector<int> seen(num_blocks, 0);
    for (int b = 0; b < num_blocks; b++) {
      AssertTrue(premap_[b] >= 0, "CiBatchAttention: the premap misses a block");
      seen[premap_[b]]++;
    }
    for (int b = 0; b < num_blocks; b++) {
      AssertTrue(seen[b] == 1,
                 "CiBatchAttention: the premap is not a bijection over blocks");
    }
    premap = &premap_;
  }

  fwd_ = std::make_unique<CiSinCConverter<word>>(
      switch_ctx_, cfg_.sub_degree, cfg_.forward_level, /*inverse_level=*/-1,
      &chain_, premap, cfg_.converter_baby_steps);
  inv_ = std::make_unique<CiSinCConverter<word>>(
      switch_ctx_, cfg_.sub_degree, /*forward_level=*/-1, cfg_.inverse_level,
      &chain_, nullptr, cfg_.converter_baby_steps, premap);
  BuildRope();

  AssertTrue(!cfg_.affine_in_prefix || cfg_.fused_scores,
             "CiBatchAttention: affine_in_prefix rides the fused score "
             "boot's prefix -- it needs fused_scores");
  if (cfg_.fused_scores) {
    // THE FUSED SCORE BOOT (idea [4], the 3.16 pattern): the scores stay in
    // the nested SinC element after the switch-back, and the tower ring's
    // HalfBoot -- whose CoeffToSlot is the tower CtS' -- lands their
    // coordinates in slots; the lane prefix finishes the return. The prefix
    // folds the inverse of the tower's message ratio (so the scores come
    // back as a `Boot` would have left them, `carried * m`) and the
    // plaintext scale that lands them CANONICAL at the layer's top level.
    AssertTrue(tower_ != nullptr,
               "CiBatchAttention: fused_scores needs the tower ring's "
               "BootContext");
    AssertTrue(tower_->GetStCInputScale() > 0.0 &&
                   tower_->GetMessageRatio() != 0.0,
               "CiBatchAttention: PrepareEvalMod must run on the tower ring "
               "before construction");
    const auto &tp = tower_->GetBootParameter();
    // The prefix lands where the LAYER's own Boot does (the softmax is
    // compiled there). A tower whose EvalMod ends ABOVE that landing is
    // fine: the HalfBootTower output is LevelDowned (exact) to the
    // prefix's entry first -- so a landing-15 layer rides the SHIPPED
    // land17c3e10 tower rather than a junction L16 ladder (whose EvalMod
    // measured 2^-7 against land17's 2^-15.5, Doing 7.36).
    // [3]: Config::score_top lowers the landing below the layer boot's
    // (the aux boot split frees the levels the softmax no longer needs).
    const int prefix_out = GetTopLevel();
    AssertTrue(prefix_out + 1 <= tp.GetEvalModEndLevel(),
               "CiBatchAttention: the tower's EvalMod ends below the layer "
               "boot's landing -- the prefix cannot reach it");
    for (int L = 0; L <= prefix_out; L++) {
      AssertTrue(
          boot_->param_.GetPrimeVector(boot_->param_.LevelToNP(L)) ==
              tower_->param_.GetPrimeVector(tower_->param_.LevelToNP(L)),
          "CiBatchAttention: the tower ring must share the layer ring's "
          "levels up to the prefix's landing (" + std::to_string(L) + ")");
    }
    basis_ = std::make_unique<CiSinCBasis<word>>(
        switch_ctx_->param_.degree_, small_ctx_->param_.degree_,
        cfg_.sub_degree);
    basis_->PrepareCtS(tower_, tp.GetCtSStartLevel(),
                       layout_.num_slots * tower_->GetCtSConst());
    const int prefix_level = prefix_out + 1;
    const double target = boot_->param_.GetScale(prefix_level - 1);
    const double pt_scale = target *
                            boot_->param_.GetRescalePrimeProd(prefix_level) /
                            tower_->GetStCInputScale();
    basis_->PreparePrefix(tower_, prefix_level,
                          /*constant=*/1.0 / tower_->GetMessageRatio(),
                          pt_scale);
    if (cfg_.verbose) {
      std::cout << "  [batch] fused scores: the tower CtS' at "
                << tp.GetCtSStartLevel() << ", the prefix at " << prefix_level
                << " landing " << prefix_out << " at scale 2^"
                << std::log2(target) << ", ratio "
                << tower_->GetMessageRatio() << std::endl;
    }
  }

  if (cfg_.verbose) {
    std::cout << "  [batch] attention: chain d " << chain_.dim << ", rank "
              << chain_.rank << ", lanes " << chain_.lanes << "; forward @"
              << cfg_.forward_level << ", chain @" << GetChainLevel()
              << ", inverse @" << cfg_.inverse_level << std::endl;
  }
}

template <typename word>
void CiBatchAttention<word>::AddTowerRotations(EvkRequest &req) const {
  AssertTrue(cfg_.fused_scores,
             "CiBatchAttention: no tower ring without fused_scores");
  basis_->AddCtSRotations(req);
  basis_->AddPrefixRotations(req);
}

template <typename word>
void CiBatchAttention<word>::BootScoresFused(std::vector<Ct> &booted,
                                             std::vector<Ct> &sinc,
                                             const Keys &keys, int group,
                                             double carried) const {
  NvtxScope _nv("batch attn: fused score boot");
  AssertTrue(cfg_.fused_scores && keys.tower != nullptr,
             "CiBatchAttention::BootScoresFused: fused_scores and "
             "Keys::tower");
  if (cfg_.affine_in_prefix) {
    AssertTrue(softmax_ready_ && calib_.causal,
               "CiBatchAttention::BootScoresFused: affine_in_prefix needs "
               "the causal softmax calibration (PrepareSoftMax first)");
    AssertTrue(carried > 0.0,
               "CiBatchAttention::BootScoresFused: affine_in_prefix needs "
               "the chain's carried factor");
    AssertTrue(prefix_affine_carried_ == 0.0 ||
                   std::abs(prefix_affine_carried_ - carried) <=
                       1e-9 * std::abs(carried),
               "CiBatchAttention::BootScoresFused: the chain's carried "
               "factor moved between calls -- the scale walk is expected "
               "deterministic");
  }
  const int T = static_cast<int>(sinc.size());
  booted.clear();
  booted.resize(T);
  const int g_max = Max(group, 1);
  const int prefix_level = GetTopLevel() + 1;
  for (int l0 = 0; l0 < T; l0 += g_max) {
    const int g = Min(T - l0, g_max);
    std::vector<const Ct *> xs(g);
    for (int j = 0; j < g; j++) xs[j] = &sinc[l0 + j];
    std::vector<Ct> halves;
    tower_->HalfBootTowerBatch(halves, xs, *keys.tower, *basis_);
    for (int j = 0; j < g; j++) {
      sinc[l0 + j] = Ct();
      // A tower whose EvalMod ends above the prefix's entry: the exact
      // LevelDown first (the declared-scale offset is preserved).
      if (tower_->param_.NPToLevel(halves[j].GetNP()) > prefix_level) {
        Ct down;
        tower_->LevelDown(down, halves[j], prefix_level);
        halves[j] = std::move(down);
      }
      // The prefix's plaintexts are (re-)encoded at the first ciphertext.
      // The measured HalfBootTower output scale EQUALS the nominal
      // GetStCInputScale the ctor used (ci_sinc_basis_test hunt1: 0 ppm on
      // both the v2 and v3 towers -- the boot SetScales exactly that number),
      // so the scale half of this re-encode reproduces the ctor's plaintexts
      // and is redundant. It stays for the AFFINE fold under affine_in_prefix:
      // the softmax's multiply `a1 = 2 / (span * carried)` (the landing-15
      // lever), folded into the same prefix. Same structure, same rotations --
      // only the values change, so the keys stand. (The 7.37 "0.28%" was the
      // v2 EvalMod landing WANDER, 2^51.8 vs the 2^58 contract, not a
      // measured/nominal miss; the v3 tower default lands 2^58 exactly.)
      const double in_scale = halves[j].GetScale();
      if (std::abs(prefix_in_scale_ - in_scale) > 1e-9 * in_scale) {
        AssertTrue(prefix_in_scale_ == 0.0,
                   "CiBatchAttention::BootScoresFused: the tower boot's "
                   "output scale moved between calls");
        const double target = boot_->param_.GetScale(prefix_level - 1);
        const double pt_scale =
            target * tower_->param_.GetRescalePrimeProd(prefix_level) /
            in_scale;
        const double a1 =
            cfg_.affine_in_prefix ? 2.0 / (calib_.span * carried) : 1.0;
        basis_->PreparePrefix(tower_, prefix_level,
                              /*constant=*/a1 / tower_->GetMessageRatio(),
                              pt_scale);
        prefix_in_scale_ = in_scale;
        prefix_affine_carried_ = cfg_.affine_in_prefix ? carried : 0.0;
        if (cfg_.verbose) {
          std::cout << "  [batch] fused scores: the prefix re-encoded from "
                    << "the measured scale 2^" << std::log2(in_scale)
                    << " -> landing 2^" << std::log2(target)
                    << (cfg_.affine_in_prefix ? " with the affine folded"
                                              : "")
                    << " (a1 " << a1 << ", carried " << carried << ")"
                    << std::endl;
        }
      }
      basis_->Prefix(booted[l0 + j], halves[j], *keys.tower);
      halves[j] = Ct();
    }
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
  t_descend_.Begin();
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
  t_descend_.End();
}

template <typename word>
void CiBatchAttention<word>::Return(Ct &res, const std::vector<Ct> &parts,
                                    const Keys &keys) const {
  NvtxScope _nv("batch attn: return");
  t_return_.Begin();
  Ct big;
  switcher_.SwitchBack(big, parts, *keys.inverse_ring_switch);
  inv_->SinCToSlot(switch_ctx_, res, big, *keys.swtch);
  t_return_.End();
}

template <typename word>
bool CiBatchAttention<word>::conv_serial_ =
    (std::getenv("CHEDDAR_CI_BATCH_CONV_SERIAL") != nullptr &&
     std::getenv("CHEDDAR_CI_BATCH_CONV_SERIAL")[0] == '1');

template <typename word>
void CiBatchAttention<word>::SetConvSerial(bool serial) {
  conv_serial_ = serial;
}

template <typename word>
void CiBatchAttention<word>::DescendBatch(std::vector<std::vector<Ct>> &lifted,
                                          std::vector<Ct> &cts, int call,
                                          const Keys &keys) const {
  const int n = static_cast<int>(cts.size());
  lifted.clear();
  lifted.resize(n);
  if (conv_serial_ || n == 1) {
    for (int c = 0; c < n; c++) Descend(lifted[c], cts[c], call, keys);
    return;
  }
  NvtxScope _nv("batch attn: descend");
  t_descend_.Begin();
  // The per-channel prologue: LevelDown, and the odd call's key-token shift.
  t_desc_pre_.Begin();
  std::vector<Ct> down(n);
  for (int c = 0; c < n; c++) {
    boot_->LevelDown(down[c], cts[c], cfg_.forward_level);
    cts[c] = Ct();
    if (call == 1) {
      AssertTrue(keys.boot != nullptr,
                 "CiBatchAttention::DescendBatch: the shift needs Keys::boot");
      const int r = GetShiftRotation();
      Ct shifted;
      boot_->HRot(shifted, down[c], keys.boot->GetRotationKey(r), r);
      down[c] = std::move(shifted);
    }
  }
  t_desc_pre_.End();
  // ONE ct-batched forward conversion over the group.
  t_desc_conv_.Begin();
  std::vector<Ct> sinc(n);
  {
    std::vector<Ct *> outs(n);
    std::vector<const Ct *> ins(n);
    for (int c = 0; c < n; c++) {
      outs[c] = &sinc[c];
      ins[c] = &down[c];
    }
    fwd_->SlotToSinCBatch(switch_ctx_, outs, ins, *keys.swtch);
  }
  down.clear();
  t_desc_conv_.End();
  // The ring switches as one batched group, then the lifts per part.
  t_desc_switch_.Begin();
  std::vector<std::vector<Ct>> parts_all;
  {
    std::vector<const Ct *> sinc_ptrs(n);
    for (int c = 0; c < n; c++) sinc_ptrs[c] = &sinc[c];
    switcher_.SwitchBatch(parts_all, sinc_ptrs, *keys.ring_switch);
  }
  sinc.clear();
  t_desc_switch_.End();
  t_desc_lift_.Begin();
  for (int c = 0; c < n; c++) {
    AssertTrue(static_cast<int>(parts_all[c].size()) == chain_.rank,
               "CiBatchAttention::DescendBatch: the switch returned the "
               "wrong number of parts");
    lifted[c].resize(chain_.rank);
    for (int g = 0; g < chain_.rank; g++) {
      lift_.Lift(lifted[c][g], parts_all[c][g]);
    }
  }
  t_desc_lift_.End();
  t_descend_.End();
}

template <typename word>
void CiBatchAttention<word>::ReturnBatch(
    const std::vector<Ct *> &res, std::vector<std::vector<Ct>> &parts_list,
    const Keys &keys, bool to_slots /*= true*/) const {
  const int n = static_cast<int>(parts_list.size());
  AssertTrue(static_cast<int>(res.size()) == n,
             "CiBatchAttention::ReturnBatch: outputs disagree with inputs");
  if (conv_serial_ || n == 1) {
    for (int i = 0; i < n; i++) {
      if (to_slots) {
        Return(*res[i], parts_list[i], keys);
      } else {
        t_return_.Begin();
        switcher_.SwitchBack(*res[i], parts_list[i],
                             *keys.inverse_ring_switch);
        t_return_.End();
      }
      parts_list[i].clear();
    }
    return;
  }
  NvtxScope _nv("batch attn: return");
  t_return_.Begin();
  if (!to_slots) {
    // The fused scores stop in the nested SinC element: switch back only.
    std::vector<const std::vector<Ct> *> parts_ptrs(n);
    for (int i = 0; i < n; i++) parts_ptrs[i] = &parts_list[i];
    switcher_.SwitchBackBatch(res, parts_ptrs, *keys.inverse_ring_switch);
    for (int i = 0; i < n; i++) parts_list[i].clear();
    t_return_.End();
    return;
  }
  std::vector<Ct> big(n);
  {
    std::vector<Ct *> big_ptrs(n);
    std::vector<const std::vector<Ct> *> parts_ptrs(n);
    for (int i = 0; i < n; i++) {
      big_ptrs[i] = &big[i];
      parts_ptrs[i] = &parts_list[i];
    }
    switcher_.SwitchBackBatch(big_ptrs, parts_ptrs,
                              *keys.inverse_ring_switch);
  }
  for (int i = 0; i < n; i++) parts_list[i].clear();
  {
    std::vector<Ct *> outs(n);
    std::vector<const Ct *> ins(n);
    for (int i = 0; i < n; i++) {
      outs[i] = res[i];
      ins[i] = &big[i];
    }
    inv_->SinCToSlotBatch(switch_ctx_, outs, ins, *keys.swtch);
  }
  t_return_.End();
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
  {
    std::vector<std::vector<Ct>> lifted;
    DescendBatch(lifted, q, -1, keys);
    for (int g = 0; g < rank; g++) lq[g].resize(D);
    for (int c = 0; c < D; c++) {
      for (int g = 0; g < rank; g++) lq[g][c] = std::move(lifted[c][g]);
    }
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
    {
      std::vector<std::vector<Ct>> lifted;
      DescendBatch(lifted, kc, call, keys);
      for (int g = 0; g < rank; g++) lk[g].resize(D);
      for (int c = 0; c < D; c++) {
        for (int g = 0; g < rank; g++) lk[g][c] = std::move(lifted[c][g]);
      }
    }
    // Per group: the elided Algorithm 4, the live half of its columns
    // descended to the product ring.
    std::vector<std::vector<Ct>> out(rank);
    for (int g = 0; g < rank; g++) {
      std::vector<Ct> prod;
      t_mult_.Begin();
      ccmm_.Multiply(lifted_ctx_, prod, lq[g], lk[g], 2 * cfg_.sub_degree,
                     *keys.lifted, /*rhs_row_wise=*/true);
      t_mult_.End();
      lk[g].clear();
      out[g].resize(T / 2);
      t_lift_descend_.Begin();
      for (int l = 0; l < T / 2; l++) lift_.Descend(out[g][l], prod[l]);
      t_lift_descend_.End();
    }
    // Per key token: the groups' parts back into one big ciphertext and
    // into slots.
    {
      std::vector<std::vector<Ct>> parts_list(T / 2);
      std::vector<Ct *> outs(T / 2);
      for (int l = 0; l < T / 2; l++) {
        parts_list[l].resize(rank);
        for (int g = 0; g < rank; g++) {
          parts_list[l][g] = std::move(out[g][l]);
        }
        outs[l] = &res[call * (T / 2) + l];
      }
      // Fused scores stop at the switch-back (the SinC element goes to
      // `BootScoresFused`); otherwise the full return to slots.
      ReturnBatch(outs, parts_list, keys, /*to_slots=*/!cfg_.fused_scores);
    }
  }
}

template <typename word>
void CiBatchAttention<word>::DescendKV(DescendedKV &dkv,
                                       const std::vector<Ct> &k,
                                       const std::vector<Ct> &v,
                                       const Keys &keys) const {
  NvtxScope _nv("batch attn: descend kv");
  const int D = cfg_.head_dim;
  const int rank = chain_.rank;
  AssertTrue(static_cast<int>(k.size()) == D &&
                 static_cast<int>(v.size()) == D,
             "CiBatchAttention::DescendKV: one kv head's channels each");
  dkv.lk.clear();
  dkv.lv.clear();
  dkv.lk.resize(2);
  dkv.lv.resize(2);
  for (int call = 0; call < 2; call++) {
    // K: RoPE with the call's key tokens kept, down to its parts -- what
    // Scores' call loop does, once instead of once per Q head.
    std::vector<Ct> kc(D);
    for (int c = 0; c < D; c++) boot_->Copy(kc[c], k[c]);
    Rope(kc, call);
    auto &lk = dkv.lk[call];
    lk.resize(rank);
    {
      std::vector<std::vector<Ct>> lifted;
      DescendBatch(lifted, kc, call, keys);
      for (int g = 0; g < rank; g++) lk[g].resize(D);
      for (int c = 0; c < D; c++) {
        for (int g = 0; g < rank; g++) lk[g][c] = std::move(lifted[c][g]);
      }
    }
    // V with the call's key tokens kept, shifted down for the odd call by
    // the converter's premap -- what Values' call loop does.
    auto &lv = dkv.lv[call];
    lv.resize(rank);
    {
      std::vector<Ct> masked(D);
      for (int c = 0; c < D; c++) {
        Ct t;
        boot_->Mult(t, v[c], call_mask_[call]);
        boot_->Rescale(masked[c], t);
      }
      std::vector<std::vector<Ct>> lifted;
      DescendBatch(lifted, masked, call, keys);
      for (int g = 0; g < rank; g++) lv[g].resize(D);
      for (int c = 0; c < D; c++) {
        for (int g = 0; g < rank; g++) lv[g][c] = std::move(lifted[c][g]);
      }
    }
  }
}

template <typename word>
void CiBatchAttention<word>::Scores(std::vector<Ct> &res, std::vector<Ct> &q,
                                    const DescendedKV &dkv,
                                    const Keys &keys) const {
  NvtxScope _nv("batch attn: Scores");
  const int T = cfg_.num_tokens;
  const int D = cfg_.head_dim;
  const int rank = chain_.rank;
  AssertTrue(static_cast<int>(q.size()) == D &&
                 static_cast<int>(dkv.lk.size()) == 2,
             "CiBatchAttention::Scores: one head's channels and a hoisted "
             "kv descent");
  AssertTrue(keys.swtch != nullptr && keys.lifted != nullptr &&
                 keys.ring_switch != nullptr &&
                 keys.inverse_ring_switch != nullptr,
             "CiBatchAttention::Scores: keys");

  Rope(q, -1);
  std::vector<std::vector<Ct>> lq(rank);
  {
    std::vector<std::vector<Ct>> lifted;
    DescendBatch(lifted, q, -1, keys);
    for (int g = 0; g < rank; g++) lq[g].resize(D);
    for (int c = 0; c < D; c++) {
      for (int g = 0; g < rank; g++) lq[g][c] = std::move(lifted[c][g]);
    }
  }

  res.clear();
  res.resize(T);
  for (int call = 0; call < 2; call++) {
    NvtxScope _c("batch attn: scores call");
    const auto &lk = dkv.lk[call];
    std::vector<std::vector<Ct>> out(rank);
    for (int g = 0; g < rank; g++) {
      std::vector<Ct> prod;
      t_mult_.Begin();
      ccmm_.Multiply(lifted_ctx_, prod, lq[g], lk[g], 2 * cfg_.sub_degree,
                     *keys.lifted, /*rhs_row_wise=*/true);
      t_mult_.End();
      out[g].resize(T / 2);
      t_lift_descend_.Begin();
      for (int l = 0; l < T / 2; l++) lift_.Descend(out[g][l], prod[l]);
      t_lift_descend_.End();
    }
    {
      std::vector<std::vector<Ct>> parts_list(T / 2);
      std::vector<Ct *> outs(T / 2);
      for (int l = 0; l < T / 2; l++) {
        parts_list[l].resize(rank);
        for (int g = 0; g < rank; g++) {
          parts_list[l][g] = std::move(out[g][l]);
        }
        outs[l] = &res[call * (T / 2) + l];
      }
      // Fused scores stop at the switch-back (the SinC element goes to
      // `BootScoresFused`); otherwise the full return to slots.
      ReturnBatch(outs, parts_list, keys, /*to_slots=*/!cfg_.fused_scores);
    }
  }
}

template <typename word>
void CiBatchAttention<word>::Values(std::vector<Ct> &res, std::vector<Ct> &P,
                                    const DescendedKV &dkv,
                                    const Keys &keys) const {
  NvtxScope _nv("batch attn: Values");
  const int T = cfg_.num_tokens;
  const int D = cfg_.head_dim;
  const int rank = chain_.rank;
  const int half = T / 2;
  AssertTrue(static_cast<int>(P.size()) == T &&
                 static_cast<int>(dkv.lv.size()) == 2,
             "CiBatchAttention::Values: P's key tokens and a hoisted kv "
             "descent");
  AssertTrue(boot_->param_.NPToLevel(P[0].GetNP()) == cfg_.forward_level,
             "CiBatchAttention::Values: P must be at forward_level");

  std::vector<std::vector<Ct>> lp(rank);
  {
    std::vector<std::vector<Ct>> lifted;
    DescendBatch(lifted, P, -1, keys);
    for (int g = 0; g < rank; g++) lp[g].resize(T);
    for (int l = 0; l < T; l++) {
      for (int g = 0; g < rank; g++) lp[g][l] = std::move(lifted[l][g]);
    }
  }

  std::vector<std::vector<Ct>> out(rank);
  for (int call = 0; call < 2; call++) {
    NvtxScope _c("batch attn: values call");
    const auto &lv = dkv.lv[call];
    for (int g = 0; g < rank; g++) {
      std::vector<Ct> lhs;
      lhs.reserve(T);
      for (int l = 0; l < half; l++) {
        lhs.push_back(std::move(lp[g][call * half + l]));
      }
      std::vector<Ct> zeros;
      ZeroLifted(zeros, lhs[0], T - half);
      for (auto &z : zeros) lhs.push_back(std::move(z));
      std::vector<Ct> prod;
      t_mult_.Begin();
      ccmm_.Multiply(lifted_ctx_, prod, lhs, lv[g], 2 * cfg_.sub_degree,
                     *keys.lifted, /*rhs_row_wise=*/false);
      t_mult_.End();
      t_lift_descend_.Begin();
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
      t_lift_descend_.End();
    }
  }
  res.clear();
  res.resize(D);
  {
    std::vector<std::vector<Ct>> parts_list(D);
    std::vector<Ct *> outs(D);
    for (int c = 0; c < D; c++) {
      parts_list[c].resize(rank);
      for (int g = 0; g < rank; g++) parts_list[c][g] = std::move(out[g][c]);
      outs[c] = &res[c];
    }
    ReturnBatch(outs, parts_list, keys);
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

// The DATA-INDEPENDENT worst-case of the LATER sum-of-squares, given the crude
// degree-`iter_degree` FIRST inverse square root over the wide window
// [first_lo, first_hi]. The Cho iteration is
//     y1 = (y0 * r0)^2 ,  sq1 = sum_t y1[t]^2 = r0^4 * sum_t y0[t]^4 ,
// and sum_t y0^4 <= (sum_t y0^2)^2 = sq0^2 (a row concentrated on one key --
// reachable: a sink-dominated early row). With r0 = finv(sq0) the crude
// invsqrt, sq1 <= (finv(sq0) * sqrt(sq0))^4, maximised over sq0 in the window.
// The FIRST invsqrt is deliberately crude, so this overshoots 1 (host: ~2.28 at
// [0.016, 128] deg 31) and the UNCLAMPED last invsqrt blows (cosh) unless its
// window's upper end covers it. This bound depends ONLY on the fit, not on any
// prompt, so it makes the last window population-safe by construction.
// See [[quarot-heterogeneous-softmax]] and reference/scripts (robust_windows.py).
double WorstCaseChoLaterSq(double first_lo, double first_hi, int iter_degree) {
  const double aff_a = 0.5 * (first_hi - first_lo);
  const double aff_b = 0.5 * (first_hi + first_lo);
  const auto c = chebfit::Interpolate(
      [aff_a, aff_b](double v) { return 1.0 / std::sqrt(aff_a * v + aff_b); },
      iter_degree);
  double worst = 0.0;
  const int grid = 4000;
  for (int i = 0; i <= grid; i++) {
    const double sq0 = first_lo + (first_hi - first_lo) * i / grid;
    const double v = (sq0 - aff_b) / aff_a;  // in [-1, 1]
    double b0 = 0.0, b1 = 0.0;               // Clenshaw of sum c_k T_k(v)
    for (size_t j = c.size() - 1; j > 0; j--) {
      const double t = 2.0 * v * b0 - b1 + c[j];
      b1 = b0;
      b0 = t;
    }
    const double finv = v * b0 - b1 + c[0];
    const double factor = finv * std::sqrt(sq0);  // (1 + relative error)
    worst = std::max(worst, factor * factor * factor * factor);
  }
  return worst;
}

}  // namespace

template <typename word>
void CiBatchAttention<word>::PrepareSoftMax(const SoftMaxCalibration &calib) {
  calib_ = calib;
  const Parameter<word> &param = boot_->param_;
  const int T = cfg_.num_tokens;
  const int top = GetTopLevel();
  // affine_in_prefix: the multiply rides the fused boot's prefix, so the
  // walk starts at exp on the landing itself -- 11 levels above forward
  // instead of 12 (the landing-15 lever).
  AssertTrue(!cfg_.affine_in_prefix || calib_.causal,
             "CiBatchAttention::PrepareSoftMax: affine_in_prefix needs the "
             "causal calibration (the row shift is per token)");
  exp_in_ = cfg_.affine_in_prefix ? top : top - 1;
  const int exp_degree =
      (calib_.exp_degree > 0) ? calib_.exp_degree : ExpDegree(calib_.m_eff);
  // k = 1 (Cho): y = exp(m_eff (u - 1) / 4), squared later by the norm.
  // niter = k > 0: the 2^k down-scale -- y0 = exp((S-shift)/2^k), the k
  // squarings raise it to exp(S-shift). The affine a1/a0 is unchanged (it
  // maps (S-shift)/span into [-1,0]); only the exp's slope shrinks:
  // hb_k = m_eff / 2^(k+1) so exp(hb_k (v-1)) = exp(m_eff (S-shift)/(span 2^k)).
  const double hb = (calib_.niter > 0)
                        ? calib_.m_eff / std::ldexp(1.0, calib_.niter + 1)
                        : calib_.m_eff / 4.0;
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
  // Causal: the Euclidean norm's affine MULTIPLY rides the mask -- each
  // mask value carries 1/sqrt(a), so the squared sum arrives already
  // divided by the window's half-width and only the level-free constant
  // remains. One level, and with it the whole walk fits a Boot landing at
  // 16: ci16_35 as shipped, no thinner StC.
  // [3] with the aux boot the accumulator leaves the walk at sq_level_ and
  // comes back at the aux ring's landing; the inverse square root is
  // compiled THERE (causal only: the shift is level-free).
  if (aux_boot_ != nullptr) {
    AssertTrue(calib_.causal,
               "CiBatchAttention::PrepareSoftMax: the aux boot split needs "
               "the causal calibration (its affine shift is level-free)");
    AssertTrue(sq_level_ >= 0,
               "CiBatchAttention::PrepareSoftMax: the walk exhausts its "
               "levels before the accumulator");
    poly_in_ = aux_boot_->GetBootParameter().GetEndLevel();
    // The accumulator must enter its bootstrap INSIDE EvalMod's ride
    // (~0.3): its raw message is S/a in [norm_lo, norm_hi] / aff_a --
    // up to 4.5 at the default window, an extrapolation of the fitted
    // polynomial (the 7.38 bisection). gamma rides the masks (each
    // carries sqrt(gamma) beside 1/sqrt(a)), the booted sum re-declares
    // its scale by gamma (free), and the inverse square root gives
    // 1/sqrt(gamma) back -- P is exact in gamma.
    const double ride = [] {
      const char *e = std::getenv("CHEDDAR_CI_BATCH_ACCUM_RIDE");
      return (e && e[0] != '\0') ? std::atof(e) : 0.3;
    }();
    const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
    aux_gamma_ = std::min(1.0, ride * aff_a / calib_.norm_hi);
  } else {
    poly_in_ = calib_.causal ? sq_level_ : sq_level_ - 1;
    aux_gamma_ = 1.0;
  }
  const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
  const double aff_b = 0.5 * (calib_.norm_hi + calib_.norm_lo);
  // Causal: the mask carries 1/sqrt(a), so the NUMERATOR y is scaled by it
  // too and P = (y r)^2 by 1/a. The polynomial must return sqrt(a) to
  // cancel it: 1/sqrt(t + b/a) = sqrt(a)/sqrt(a t + b). Without the fold
  // (non-causal) the plain 1/sqrt(a t + b) stands. The host mirror
  // (softmax_mirror.py) puts the difference at 10x vs 2^-15.5 on the real
  // layer-0 scores.
  // The aux path's masks carry sqrt(gamma) (BuildMasks), so the booted sum
  // reads S gamma / a; its scale is re-declared by gamma in SoftMax (free)
  // and the polynomial returns the leftover 1/sqrt(gamma) -- so its INPUT
  // scale is gamma * canonical and its values carry gamma^-1/2.
  const double gi = 1.0 / std::sqrt(aux_gamma_);
  auto inv_coeffs =
      calib_.causal
          ? chebfit::Interpolate(
                [aff_a, aff_b, gi](double v) {
                  return gi / std::sqrt(v + aff_b / aff_a);
                },
                calib_.inv_degree)
          : chebfit::Interpolate(
                [aff_a, aff_b](double v) {
                  return 1.0 / std::sqrt(aff_a * v + aff_b);
                },
                calib_.inv_degree);
  const double inv_in_scale = aux_gamma_ * param.GetScale(poly_in_);
  const int inv_used = EvalPoly<word>(inv_coeffs, poly_in_, inv_in_scale,
                                      param.GetScale(poly_in_), true)
                           .GetPolyDegree();
  const int inv_out = poly_in_ - Log2Ceil(inv_used + 1);
  AssertTrue(inv_out - 2 >= cfg_.forward_level,
             "CiBatchAttention: the softmax walk overspends its levels; P "
             "would land below forward_level");
  polys_.push_back(std::make_unique<EvalPoly<word>>(
      inv_coeffs, poly_in_, inv_in_scale, param.GetScale(inv_out), true));
  polys_[1]->Compile(boot_);
  if (aux_boot_ != nullptr) {
    aux_inv_coeffs_ = inv_coeffs;
    aux_inv_out_ = inv_out;
    aux_inv_in_scale_ = inv_in_scale;
  }

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
  // niter = k > 0: the full Cho iteration's invsqrt polynomials. The main
  // path is booted each iteration, so sq always arrives at `top - 1`; every
  // invsqrt is compiled THERE. [0] the WIDE first window (population underflow
  // not yet compressed by the squarings), [1] the crude intermediate and [2]
  // the accurate last, both over [norm_lo, norm_hi] ~ [1/n, 1]. No est fold
  // (the mask is plain causal), so the plain 1/sqrt(a v + b) form. See
  // SoftMaxCho and [[quarot-heterogeneous-softmax]].
  cho_inv_.clear();
  if (calib_.niter > 0) {
    AssertTrue(calib_.causal,
               "CiBatchAttention::PrepareSoftMax: niter>0 needs the causal "
               "calibration (per-token row shift)");
    AssertTrue(!cfg_.affine_in_prefix && !cfg_.fused_scores,
               "CiBatchAttention::PrepareSoftMax: niter>0 is the plain causal "
               "path (no fused/affine-prefix yet)");
    // Booted iterations: sq = sum(y^2) lands at top-1, the invsqrt's affine
    // multiply rescales to top-2. The FIRST iteration skips the boot (y0 is
    // fresh from exp at exp_out_), so its sq lands at exp_out_-1 and its affine
    // at exp_out_-2 -- compile the first invsqrt THERE.
    // sq = sum(y^2) costs a level, its affine multiply another, so the invsqrt
    // reads at (y level) - 2. Booted y is at `top`; the un-booted y0 is at
    // exp_out_-1 (exp then the causal-mask rescale).
    cho_inv_in_ = top - 2;
    cho_first_in_ = exp_out_ - 3;
    AssertTrue(cho_first_in_ > 0,
               "CiBatchAttention::PrepareSoftMax: niter>0 needs exp_out above 3");
    if (!calib_.cho_est.empty()) {
      AssertTrue(static_cast<int>(calib_.cho_est.size()) == calib_.niter,
                 "CiBatchAttention::PrepareSoftMax: cho_est is one estimate "
                 "set an ITERATION");
      for (const auto &per_head : calib_.cho_est) {
        AssertTrue(!per_head.empty(),
                   "CiBatchAttention::PrepareSoftMax: cho_est needs a head");
        for (const auto &rows : per_head) {
          AssertTrue(static_cast<int>(rows.size()) == T,
                     "CiBatchAttention::PrepareSoftMax: cho_est is one "
                     "estimate a ROW");
          for (double e : rows) {
            AssertTrue(e > 0.0,
                       "CiBatchAttention::PrepareSoftMax: a cho_est entry is "
                       "not positive, so its square root is not a fold");
          }
        }
      }
    }
    cho_first_lo_ = (calib_.first_lo > 0.0) ? calib_.first_lo : calib_.norm_lo;
    cho_first_hi_ = (calib_.first_hi > 0.0) ? calib_.first_hi : calib_.norm_hi;
    // The later window: start from the calibrated [norm_lo, norm_hi] ~ [1/n, 1],
    // but WIDEN its upper end (by construction, no data) to cover the crude
    // first invsqrt's worst-case overshoot -- else the unclamped last invsqrt
    // blows (cosh) for a prompt whose row concentrates near the crude region
    // (host: sq1 up to ~2.28 at [0.016,128] deg 31, REACHABLE by a sink-dominated
    // row). A 10% margin above the bound; this only relaxes the last invsqrt's
    // fit (accuracy is set by the LOW end, host robust_windows.py: negligible)
    // and costs no level or bootstrap. See [[quarot-heterogeneous-softmax]].
    cho_later_lo_ = calib_.norm_lo;
    // The overshoot is a CHAIN, not one step: each normalization's crude `r`
    // inflates the NEXT iteration's `sq`, so the window that has to cover it is
    // the next one's. With k = 2 the last iteration follows the first directly
    // and this is the original bound; with k >= 3 the intermediate sits between
    // them and owes its own.
    const double worst_first =
        WorstCaseChoLaterSq(cho_first_lo_, cho_first_hi_, calib_.iter_inv_degree);
    cho_mid_deg_ = (calib_.mid_inv_degree > 0) ? calib_.mid_inv_degree
                                               : calib_.iter_inv_degree;
    // A SUPPLIED intermediate window is taken as given, and must already carry
    // the first invsqrt's error (the generator walks the calibration split with
    // the crude `r_0` in place). It is not widened by `worst_first`, because
    // that bound uses `sum y^4 <= (sum y^2)^2` -- attained only by a row
    // concentrated on ONE key. At the LAST iteration's temperature the data
    // really does reach it, which is why the original bound costs nothing
    // there; at an intermediate temperature the collision probability is two
    // orders below it (measured 7.0e-3 against the bound's 1.0), and applying
    // it turns a 36x window into a 5700x one -- the very thing this window
    // exists to avoid.
    const bool have_mid = calib_.mid_hi > 0.0;
    cho_mid_lo_ = have_mid ? calib_.mid_lo : calib_.norm_lo;
    cho_mid_hi_ = have_mid ? calib_.mid_hi
                           : std::max(calib_.norm_hi, 1.10 * worst_first);
    const double worst_mid =
        (calib_.niter >= 3)
            ? WorstCaseChoLaterSq(cho_mid_lo_, cho_mid_hi_, cho_mid_deg_)
            : worst_first;
    cho_later_hi_ = std::max(calib_.norm_hi, 1.10 * worst_mid);
    // `is_last` lands P at forward_level (its apply is the final square);
    // the first/intermediate invsqrts' output is booted, so they only need
    // their apply to stay above 0.
    auto compile_inv = [&](double lo, double hi, int degree, int in_level,
                           bool is_last) {
      const double aff_a = 0.5 * (hi - lo);
      const double aff_b = 0.5 * (hi + lo);
      auto coeffs = chebfit::Interpolate(
          [aff_a, aff_b](double v) { return 1.0 / std::sqrt(aff_a * v + aff_b); },
          degree);
      const int used = EvalPoly<word>(coeffs, in_level, param.GetScale(in_level),
                                      param.GetScale(in_level), true)
                           .GetPolyDegree();
      const int out = in_level - Log2Ceil(used + 1);
      // Taking a per-row estimate back out of `r` is one more plaintext
      // multiply on the last iteration's output, so its floor rises by one.
      const int floor =
          is_last ? cfg_.forward_level + 2 + (calib_.cho_est.empty() ? 0 : 1)
                  : 3;
      AssertTrue(out >= floor,
                 "CiBatchAttention::PrepareSoftMax: niter invsqrt overspends "
                 "its levels");
      auto p = std::make_unique<EvalPoly<word>>(coeffs, in_level,
                                                param.GetScale(in_level),
                                                param.GetScale(out), true);
      p->Compile(boot_);
      return p;
    };
    cho_inv_.push_back(compile_inv(cho_first_lo_, cho_first_hi_,
                                   calib_.iter_inv_degree, cho_first_in_, false));
    cho_inv_.push_back(
        compile_inv(cho_mid_lo_, cho_mid_hi_, cho_mid_deg_, cho_inv_in_, false));
    cho_inv_.push_back(compile_inv(cho_later_lo_, cho_later_hi_,
                                   calib_.last_inv_degree, cho_inv_in_, true));
    // The plain-causal 0/1 masks at exp_out_ are HEAD-INDEPENDENT, so encode the
    // 128 of them ONCE here (SoftMaxCho, called once per head, read cho_masks_[l]
    // instead of re-encoding 128 x NHEAD a layer). mask[l] is live at t >= l.
    cho_masks_.clear();
    cho_masks_.resize(T);
    {
      std::vector<double> m(T);
      std::vector<Complex> msg;
      for (int l = 0; l < T; l++) {
        for (int t = 0; t < T; t++) m[t] = (t >= l) ? 1.0 : 0.0;
        layout_.PackPerToken(msg, m);
        boot_->gpu_encoder_.Encode(cho_masks_[l], exp_out_,
                                   param.GetScale(exp_out_), msg);
      }
    }
    if (cfg_.verbose) {
      std::cout << "  [batch] softmax Cho: niter " << calib_.niter
                << ", exp hb " << hb << " @" << exp_in_ << ".." << exp_out_
                << ", invsqrt @" << cho_inv_in_ << " first[" << cho_first_lo_
                << "," << cho_first_hi_ << "] deg " << calib_.iter_inv_degree
                << ", mid[" << cho_mid_lo_ << "," << cho_mid_hi_ << "] deg "
                << cho_mid_deg_ << ", later[" << cho_later_lo_ << ","
                << cho_later_hi_ << "] (calib_hi " << calib_.norm_hi
                << ", worst " << worst_first << "/" << worst_mid
                << "), last deg " << calib_.last_inv_degree << std::endl;
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
  // The affine's multiplicative half, folded (see PrepareSoftMax); the aux
  // path's ride factor gamma rides here too (1 without the aux boot).
  const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
  const double fold = std::sqrt(aux_gamma_ / aff_a);
  for (int l = 0; l < T; l++) {
    for (int t = 0; t < T; t++) {
      double v = 0.0;
      if (t >= l) {
        v = fold * (calib_.row_norm.empty()
                        ? 1.0
                        : 1.0 / std::sqrt(calib_.row_norm[head][t]));
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
                                     double carried, const EvkMap<word> &evk,
                                     const PublicHalf *pub,
                                     Ct *pub_scale) const {
  NvtxScope _nv("batch attn: SoftMax");
  AssertTrue(softmax_ready_, "CiBatchAttention: call PrepareSoftMax first");
  if (calib_.niter > 0) {
    SoftMaxCho(P, scores, head, carried, evk, pub, pub_scale);
    return;
  }
  AssertTrue(pub == nullptr,
             "CiBatchAttention::SoftMax: the public half joins the Cho walk "
             "(niter > 0); the single-shot softmax has no iteration to join");
  const Parameter<word> &param = boot_->param_;
  const int T = cfg_.num_tokens;
  const int top = GetTopLevel();
  AssertTrue(static_cast<int>(scores.size()) == T,
             "CiBatchAttention::SoftMax: one head's key-token ciphertexts");
  AssertTrue(param.NPToLevel(scores[0].GetNP()) >= top,
             "CiBatchAttention::SoftMax: the scores must be booted to at "
             "least the top level");
  AssertTrue(carried > 0.0, "CiBatchAttention::SoftMax: carried");
  const auto &mult_key = evk.GetMultiplicationKey();

  std::vector<Pt> masks;
  if (calib_.causal) BuildMasks(masks, head);

  // Affine onto the fit domain (carried divides out here), exp, the mask.
  // Under affine_in_prefix the multiply already rode the fused boot's
  // prefix (BootScoresFused folded a1 with this carried -- asserted), so
  // only the row shift is added and the walk starts one level higher.
  const double a1 = 2.0 / (calib_.span * carried);
  Constant<word> c1;
  if (cfg_.affine_in_prefix) {
    AssertTrue(std::abs(prefix_affine_carried_ - carried) <=
                   1e-9 * std::abs(carried),
               "CiBatchAttention::SoftMax: the prefix was folded with a "
               "different carried factor");
  } else {
    boot_->encoder_.EncodeConstant(c1, top, param.GetScale(top), a1);
  }
  // The affine is per key token and key-switch free; the exp is ONE
  // polynomial over ciphertexts that differ only in their data, so it CAN go
  // through `EvaluateBatch` -- the compiled tree walked once, every key
  // switch `MultKeyBatch`-shaped, word for word the per-ciphertext loop.
  //
  // It does not pay here, and the default is the loop. Measured on the A100
  // (`ci16_35`, T = 128, one head, exp from a deg-15 fit), one softmax:
  //
  //     batch 1 (the loop) 1.088 s | 8  1.042 s | 32  1.089 s | 128  1.041 s
  //
  // flat inside the run-to-run noise, and word for word identical at every
  // width (TheBatchedSoftMaxExpIsTheSerialOneWordForWord). The reason is
  // that this evaluation is not launch-bound: each ciphertext is 65536 slots
  // at a high level, so one exp already fills the card. The same lever moved
  // the DECODE's heads 15.5 -> 8.3 s because there the per-head work is
  // small. Kept behind `exp_batch_` for a shape where that changes; the
  // chunk bounds the live basis, which a deg-15 tree holds several of.
  std::vector<Ct> y(T);
  const int chunk = (exp_batch_ > 1) ? Min(exp_batch_, T) : 1;
  const auto affine = [&](Ct &u, int l) {
    if (cfg_.affine_in_prefix) {
      // A LevelDown to the ciphertext's own level is a copy; the affine's
      // multiply already rode the prefix. (A serial boot landing above
      // `score_top` comes down here too.)
      boot_->LevelDown(u, scores[l], top);
    } else {
      Ct t1;
      if (param.NPToLevel(scores[l].GetNP()) > top) {
        Ct down;
        boot_->LevelDown(down, scores[l], top);
        boot_->Mult(t1, down, c1);
      } else {
        boot_->Mult(t1, scores[l], c1);
      }
      boot_->Rescale(u, t1);
    }
    if (calib_.causal) {
      boot_->Add(u, u, a0_[head]);
    } else {
      Constant<word> c0;
      boot_->encoder_.EncodeConstant(c0, exp_in_, u.GetScale(),
                                     1.0 - 2.0 * calib_.shift / calib_.span);
      boot_->Add(u, u, c0);
    }
  };

  {
    NvtxScope _e("batch attn: softmax exp");
    for (int start = 0; start < T; start += chunk) {
      const int end = Min(T, start + chunk);
      if (chunk == 1) {
        Ct u;
        affine(u, start);
        polys_[0]->Evaluate(boot_, y[start], u, mult_key);
        continue;
      }
      std::vector<Ct> u(end - start);
      for (int l = start; l < end; l++) affine(u[l - start], l);

      CtBatch<word> in;
      const NPInfo in_np = u[0].GetNP();
      in.Allocate(in_np, end - start, false);
      in.scale_ = u[0].GetScale();
      in.num_slots_ = u[0].GetNumSlots();
      const size_t in_bytes = in.PolyWords() * sizeof(word);
      for (int b = 0; b < end - start; b++) {
        AssertTrue(u[b].GetNP() == in_np && !u[b].HasRx(),
                   "CiBatchAttention::SoftMax: the affine left the group at "
                   "different levels");
        AssertTrue(std::abs(u[b].GetScale() - in.scale_) <= 1e-9 * in.scale_,
                   "CiBatchAttention::SoftMax: the affine left the group at "
                   "different scales");
        cudaMemcpyAsync(in.CtData(b), u[b].bx_.data(), in_bytes,
                        cudaMemcpyDeviceToDevice, cudaStreamLegacy);
        cudaMemcpyAsync(in.CtData(b) + in.PolyWords(), u[b].ax_.data(),
                        in_bytes, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
      }
      u.clear();

      CtBatch<word> out;
      polys_[0]->EvaluateBatch(boot_, out, in, mult_key);
      in = CtBatch<word>();

      const size_t out_bytes = out.PolyWords() * sizeof(word);
      for (int b = 0; b < end - start; b++) {
        Ct &dst = y[start + b];
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

  Ct sq_acc;
  for (int l = 0; l < T; l++) {
    if (calib_.causal) {
      Ct t2;
      boot_->Mult(t2, y[l], masks[l]);
      boot_->Rescale(y[l], t2);
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
  // [3] the aux boot split: the ONE accumulator rides its own short ring
  // back up, and the inverse square root walks from that landing.
  if (aux_boot_ != nullptr) {
    AssertTrue(aux_evk_ != nullptr,
               "CiBatchAttention::SoftMax: SetAuxBoot without its keys");
    Ct up;
    aux_boot_->Boot(up, sq, *aux_evk_);
    sq = std::move(up);
    AssertTrue(param.NPToLevel(sq.GetNP()) == poly_in_,
               "CiBatchAttention::SoftMax: the aux boot did not land at "
               "the compiled inverse square root's level");
    // The masks carried sqrt(gamma): re-declare the scale by gamma (free)
    // so the message reads S / a again; the polynomial was compiled at
    // gamma * canonical and returns the leftover 1 / sqrt(gamma).
    sq.SetScale(sq.GetScale() * aux_gamma_);
    // The aux ladder's EvalMod may land off the nominal scale (the 7.37
    // drift): recompile the inverse square root ONCE at the measured
    // input scale.
    const double in_scale = sq.GetScale();
    if (std::abs(in_scale - aux_inv_in_scale_) > 1e-9 * in_scale) {
      polys_[1] = std::make_unique<EvalPoly<word>>(
          aux_inv_coeffs_, poly_in_, in_scale,
          param.GetScale(aux_inv_out_), true);
      polys_[1]->Compile(boot_);
      aux_inv_in_scale_ = in_scale;
      if (cfg_.verbose) {
        std::cout << "  [batch] softmax: invsqrt recompiled at the "
                  << "measured aux landing scale 2^" << std::log2(in_scale)
                  << std::endl;
      }
    }
  }
  {
    const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
    const double aff_b = 0.5 * (calib_.norm_hi + calib_.norm_lo);
    if (calib_.causal) {
      // 1/a arrived through the masks (BuildMasks); only the level-free
      // constant is left.
      Constant<word> shift;
      boot_->encoder_.EncodeConstant(shift, poly_in_, sq.GetScale(),
                                     -aff_b / aff_a);
      boot_->Add(sq, sq, shift);
    } else {
      Constant<word> inv_a;
      boot_->encoder_.EncodeConstant(inv_a, sq_level_,
                                     param.GetScale(sq_level_), 1.0 / aff_a);
      Ct scaled;
      boot_->Mult(scaled, sq, inv_a);
      boot_->Rescale(sq, scaled);
      Constant<word> shift;
      boot_->encoder_.EncodeConstant(shift, poly_in_, sq.GetScale(),
                                     -aff_b / aff_a);
      boot_->Add(sq, sq, shift);
    }
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
// The FULL Cho [25] iteration -- 512 DIFFERENT prompts, one population calib.
// ---------------------------------------------------------------------------
template <typename word>
void CiBatchAttention<word>::SoftMaxCho(std::vector<Ct> &P,
                                        const std::vector<Ct> &scores, int head,
                                        double carried,
                                        const EvkMap<word> &evk,
                                        const PublicHalf *pub,
                                        Ct *pub_scale) const {
  NvtxScope _nv("batch attn: SoftMaxCho");
  const Parameter<word> &param = boot_->param_;
  const int T = cfg_.num_tokens;
  const int top = GetTopLevel();
  const int k = calib_.niter;
  AssertTrue(static_cast<int>(scores.size()) == T,
             "CiBatchAttention::SoftMaxCho: one head's key-token ciphertexts");
  AssertTrue(param.NPToLevel(scores[0].GetNP()) >= top,
             "CiBatchAttention::SoftMaxCho: the scores must be booted to top");
  AssertTrue(carried > 0.0, "CiBatchAttention::SoftMaxCho: carried");
  AssertTrue(calib_.causal, "CiBatchAttention::SoftMaxCho: causal only");
  AssertTrue(static_cast<int>(cho_inv_.size()) == 3,
             "CiBatchAttention::SoftMaxCho: call PrepareSoftMax with niter>0");
  const auto &mult_key = evk.GetMultiplicationKey();

  // The plain causal 0/1 masks at exp_out_ are HEAD-INDEPENDENT (no est/gamma
  // fold -- the iteration does the normalization), so they were encoded ONCE in
  // PrepareSoftMax; read cho_masks_[l] here. mask[l] is live (1) at t >= l.
  AssertTrue(static_cast<int>(cho_masks_.size()) == T,
             "CiBatchAttention::SoftMaxCho: PrepareSoftMax must build cho_masks_");

  // y0 = exp((S - shift)/2^k) (.) causal.  u = a1 S + a0[row], exp, mask.
  const double a1 = 2.0 / (calib_.span * carried);
  Constant<word> c1;
  boot_->encoder_.EncodeConstant(c1, top, param.GetScale(top), a1);
  std::vector<Ct> y(T);
  for (int l = 0; l < T; l++) {
    Ct t1, u;
    if (param.NPToLevel(scores[l].GetNP()) > top) {
      Ct down;
      boot_->LevelDown(down, scores[l], top);
      boot_->Mult(t1, down, c1);
    } else {
      boot_->Mult(t1, scores[l], c1);
    }
    boot_->Rescale(u, t1);              // exp_in_ = top - 1
    boot_->Add(u, u, a0_[head]);
    Ct yf, t2;
    polys_[0]->Evaluate(boot_, yf, u, mult_key);   // exp -> exp_out_
    boot_->Mult(t2, yf, cho_masks_[l]);
    boot_->Rescale(y[l], t2);           // exp_out_ - 1
  }

  // The main-path boots batch across the T key-token ciphertexts, exactly the
  // score boots' CHEDDAR_CI_BATCH_BOOT_GROUP (BootBatch is word-for-word equal
  // to the loop; default 8, matching CiBatchLayer::BootGroupSize; group 1 = the
  // serial A/B baseline).
  static const int boot_group = [] {
    const char *e = std::getenv("CHEDDAR_CI_BATCH_BOOT_GROUP");
    const int v = (e != nullptr) ? std::atoi(e) : 0;
    return v >= 1 ? v : 8;
  }();

  // The public half's running scalar. `R_0 = 1` is not a ciphertext, so it is
  // a flag; from then on `R_{j+1} = R_j^2 (r^(j))^2`. It carries no key index,
  // which is the whole reason the public keys need not survive an iteration.
  Ct R;
  bool have_R = false;
  if (pub != nullptr) {
    AssertTrue(pub->pow != nullptr && static_cast<int>(pub->pow->size()) == k,
               "CiBatchAttention::SoftMaxCho: the public half owes one power "
               "sum an iteration (CiPcAttention::GetNumPowers)");
    AssertTrue(pub_scale != nullptr,
               "CiBatchAttention::SoftMaxCho: the public value accumulator "
               "needs R_k, so pub_scale must be given with pub");
    AssertTrue(static_cast<int>(pub->pow_scale.size()) == k,
               "CiBatchAttention::SoftMaxCho: pow_scale is one row of "
               "constants an iteration");
    for (const auto &row : pub->pow_scale) {
      AssertTrue(static_cast<int>(row.size()) == T,
                 "CiBatchAttention::SoftMaxCho: pow_scale is one constant a "
                 "query token");
    }
    if (!calib_.pub_r_est.empty()) {
      AssertTrue(static_cast<int>(calib_.pub_r_est.size()) == k,
                 "CiBatchAttention::SoftMaxCho: pub_r_est is one estimate set "
                 "an ITERATION");
      for (const auto &per_head : calib_.pub_r_est) {
        AssertTrue(head < static_cast<int>(per_head.size()),
                   "CiBatchAttention::SoftMaxCho: pub_r_est has no such head");
        AssertTrue(static_cast<int>(per_head[head].size()) == T,
                   "CiBatchAttention::SoftMaxCho: pub_r_est is one estimate a "
                   "query token");
        for (double e : per_head[head]) {
          AssertTrue(e > 0.0,
                     "CiBatchAttention::SoftMaxCho: a pub_r_est entry is not "
                     "positive, so it is not a normaliser");
        }
      }
    }
  }
  // `R` is carried NORMALISED (`R / G_j`) so the bootstrap at the end of each
  // iteration sees an O(1) message -- see `SoftMaxCalibration::pub_r_est`.
  // `G_0 = 1`; `G_{j+1} = pub_r_est[j][head]`.
  const bool r_normed = (pub != nullptr) && !calib_.pub_r_est.empty();
  const std::vector<double> ones(T, 1.0);
  auto g_at = [&](int j) -> const std::vector<double> & {
    return (!r_normed || j == 0) ? ones : calib_.pub_r_est[j - 1][head];
  };

  // k normalize-and-square iterations. Boot the MAIN path to top each time
  // (so sq arrives at top-1 uniformly), norm via invsqrt, then y = (y r)^2.
  for (int j = 0; j < k; j++) {
    // (1) main-path bootstrap to top -- SKIPPED on iteration 0 (y0 is fresh
    //     from exp, still high), so the first invsqrt reads at cho_first_in_.
    if (j > 0) {
      for (int l0 = 0; l0 < T; l0 += boot_group) {
        const int g = Min(T - l0, boot_group);
        if (g == 1) {
          Ct up;
          boot_->Boot(up, y[l0], evk);
          y[l0] = std::move(up);
          continue;
        }
        std::vector<const Ct *> in(g);
        for (int j2 = 0; j2 < g; j2++) in[j2] = &y[l0 + j2];
        std::vector<Ct> out_g;
        boot_->BootBatch(out_g, in, evk);
        for (int j2 = 0; j2 < g; j2++) y[l0 + j2] = std::move(out_g[j2]);
      }
    }
    // (2) sq = sum_l y_l^2  (one relinearization), landing at top-1.
    Ct sq_acc;
    for (int l = 0; l < T; l++) {
      Ct sq;
      boot_->Mult(sq, y[l], y[l]);
      if (l == 0)
        sq_acc = std::move(sq);
      else
        boot_->Add(sq_acc, sq_acc, sq);
    }
    Ct sq;
    boot_->RelinearizeRescale(sq, sq_acc, mult_key);
    sq_acc = Ct();
    Ct rsq;
    bool have_rsq = false;
    // (3) the window and its invsqrt: first iteration wide, later [norm_lo,hi];
    //     crude except the last.
    const bool first = (j == 0);
    const bool last = (j == k - 1);
    const double lo =
        first ? cho_first_lo_ : (last ? cho_later_lo_ : cho_mid_lo_);
    const double hi =
        first ? cho_first_hi_ : (last ? cho_later_hi_ : cho_mid_hi_);
    EvalPoly<word> *inv =
        first ? cho_inv_[0].get() : (last ? cho_inv_[2].get() : cho_inv_[1].get());
    const double aff_a = 0.5 * (hi - lo);
    const double aff_b = 0.5 * (hi + lo);
    const int sq_lvl = param.NPToLevel(sq.GetNP());
    Ct scaled, sqv;
    // The affine's multiply. With a per-row estimate it carries the estimate
    // too -- `sq / est` instead of `sq` -- so the window the polynomial was
    // fitted on is the RATIO and not the span of the rows' concentrations.
    // Same multiply, same level; only taking the estimate back out of `r`
    // below costs one. See `SoftMaxCalibration::cho_est`.
    const bool folded = !calib_.cho_est.empty();
    std::vector<double> row_g;   //!< 1 / sqrt(est), kept for `r`
    if (folded) {
      const auto &est = calib_.cho_est[j];
      AssertTrue(head >= 0 && head < static_cast<int>(est.size()),
                 "CiBatchAttention::SoftMaxCho: cho_est has no such head");
      std::vector<double> row_a(T);
      row_g.resize(T);
      for (int t = 0; t < T; t++) {
        row_a[t] = 1.0 / (est[head][t] * aff_a);
        row_g[t] = 1.0 / std::sqrt(est[head][t]);
      }
      std::vector<Complex> msg;
      Pt pa;
      layout_.PackPerToken(msg, row_a);
      boot_->gpu_encoder_.Encode(pa, sq_lvl, param.GetScale(sq_lvl), msg);
      boot_->Mult(scaled, sq, pa);
    } else {
      Constant<word> inva;
      boot_->encoder_.EncodeConstant(inva, sq_lvl, param.GetScale(sq_lvl),
                                     1.0 / aff_a);
      boot_->Mult(scaled, sq, inva);
    }
    boot_->Rescale(sqv, scaled);        // cho_inv_in_ = top - 2
    Constant<word> shift;
    boot_->encoder_.EncodeConstant(shift, param.NPToLevel(sqv.GetNP()),
                                   sqv.GetScale(), -aff_b / aff_a);
    boot_->Add(sqv, sqv, shift);
    // (2b) THE PUBLIC HALF OF THE SAME DENOMINATOR, for T = 4096:
    //          sq^(j) += R_j^2 * sum_p (y0_p)^(2^(j+1)) .
    // `R_j` is one number a query token, so it left the sum and the public
    // keys are gone -- only their power sum is here.
    //
    // It joins AFTER the affine, and that is not cosmetic. The public term
    // arrives from the public branch far below the walk and has to cross a
    // bootstrap, which wants an O(1) message; the raw power sum over 3968
    // keys runs to the hundreds and EvalMod's sine would be evaluated
    // outside its range -- returning garbage, not an error. Every per-row
    // constant (`pow_scale`, the row's `cho_est`, and `1 / aff_a`) is
    // therefore folded in BEFORE the boot, so what crosses it is the public
    // half of the invsqrt's ARGUMENT.
    if (pub != nullptr) {
      const Ct &M = (*pub->pow)[j];
      // The constants ride M, which is the highest of the three: a plaintext
      // multiply costs a level wherever it is taken, and `R_j^2` has the
      // least to give (two squarings below the first invsqrt's landing).
      // `R_j^2 = (R_j / G_j)^2 G_j^2`, and the carried ciphertext is the
      // normalised one, so `G_j^2` comes back out here -- on a per-row
      // constant this term already carries, at no level.
      const std::vector<double> &gj = g_at(j);
      std::vector<double> c(T);
      for (int t = 0; t < T; t++) {
        const double est = folded ? calib_.cho_est[j][head][t] : 1.0;
        c[t] = pub->pow_scale[j][t] * gj[t] * gj[t] / (est * aff_a);
      }
      const int m_lvl = param.NPToLevel(M.GetNP());
      AssertTrue(m_lvl >= 1,
                 "CiBatchAttention::SoftMaxCho: the public power sum arrived "
                 "with no level to fold its constants into");
      std::vector<Complex> msg;
      Pt pc;
      layout_.PackPerToken(msg, c);
      boot_->gpu_encoder_.Encode(pc, m_lvl, param.GetScale(m_lvl), msg);
      Ct mc, t1;
      boot_->Mult(t1, M, pc);
      boot_->Rescale(mc, t1);

      Ct pre;
      if (!have_R) {
        pre = std::move(mc);  // R_0 = 1
      } else {
        boot_->HMult(rsq, R, R, mult_key);  // R_j^2, reused by (3b)
        have_rsq = true;
        const int low = Min(param.NPToLevel(rsq.GetNP()),
                            param.NPToLevel(mc.GetNP()));
        AssertTrue(low >= 1,
                   "CiBatchAttention::SoftMaxCho: R_j^2 and the public power "
                   "sum have no level left to meet on");
        Ct a2, b2;
        boot_->LevelDown(a2, rsq, low);
        boot_->LevelDown(b2, mc, low);
        boot_->HMult(pre, a2, b2, mult_key);
      }
      // What crosses the bootstrap is the public half of the invsqrt's
      // ARGUMENT, which is O(1) -- see `PublicHalf::pow_scale`.
      Ct term;
      boot_->Boot(term, pre, evk);
      const int v_lvl = param.NPToLevel(sqv.GetNP());
      AssertTrue(param.NPToLevel(term.GetNP()) >= v_lvl,
                 "CiBatchAttention::SoftMaxCho: the public term landed below "
                 "the argument it joins");
      Ct down;
      boot_->LevelDown(down, term, v_lvl);
      boot_->Add(sqv, sqv, down);
    }
    Ct r;
    inv->Evaluate(boot_, r, sqv, mult_key);
    if (folded) {
      // The polynomial was handed `sq / est`, so it returned
      // `sqrt(est) * r_true`. One plaintext multiply on ONE ciphertext takes
      // the estimate back out, and the walk has the level for it: at
      // `cho_inv_in_` = top - 2 = 14 a degree-63 invsqrt lands `r` at 8
      // against the compile-time floor `forward_level + 2 + 1` = 7.
      const int r_lvl = param.NPToLevel(r.GetNP());
      std::vector<Complex> msg;
      Pt pg;
      layout_.PackPerToken(msg, row_g);
      boot_->gpu_encoder_.Encode(pg, r_lvl, param.GetScale(r_lvl), msg);
      Ct t3, rf;
      boot_->Mult(t3, r, pg);
      boot_->Rescale(rf, t3);
      r = std::move(rf);
    }
    // (3b) R_{j+1} = R_j^2 (r^(j))^2 -- the next iteration's public factor,
    //      and after the last one the factor the public VALUE accumulator
    //      needs before it joins the output of `Values`.
    if (pub != nullptr) {
      Ct r2;
      boot_->HMult(r2, r, r, mult_key);
      if (r_normed) {
        // R~_{j+1} = R~_j^2 r_j^2 (G_j^2 / G_{j+1}). The factor rides `r^2`,
        // which exists only for this recursion and is booted immediately
        // after, so the level it costs is one the boot gives back.
        const std::vector<double> &gj = g_at(j);
        const std::vector<double> &gn = g_at(j + 1);
        std::vector<double> f(T);
        for (int t = 0; t < T; t++) f[t] = gj[t] * gj[t] / gn[t];
        const int lvl = param.NPToLevel(r2.GetNP());
        AssertTrue(lvl >= 1,
                   "CiBatchAttention::SoftMaxCho: no level left to normalise "
                   "the public scalar's recursion");
        std::vector<Complex> msg;
        Pt pf;
        layout_.PackPerToken(msg, f);
        boot_->gpu_encoder_.Encode(pf, lvl, param.GetScale(lvl), msg);
        Ct t4, rn;
        boot_->Mult(t4, r2, pf);
        boot_->Rescale(rn, t4);
        r2 = std::move(rn);
      }
      if (!have_R) {
        R = std::move(r2);  // R_1 = (r^(0))^2
        have_R = true;
      } else {
        if (!have_rsq) {
          boot_->HMult(rsq, R, R, mult_key);
          have_rsq = true;
        }
        const int low = Min(param.NPToLevel(rsq.GetNP()),
                            param.NPToLevel(r2.GetNP()));
        Ct a, b, prod;
        boot_->LevelDown(a, rsq, low);
        boot_->LevelDown(b, r2, low);
        boot_->HMult(prod, a, b, mult_key);
        R = std::move(prod);
      }
      // `R` costs levels every iteration (a square and a multiply) and it has
      // to survive all of them AND the multiply onto the public value
      // accumulator afterwards, so it rides back up here. It is O(1) by
      // construction -- `R_j` is a product of inverse square roots of sums
      // that are themselves normalised -- so the bootstrap is well posed.
      // One boot an iteration a head, 64 a layer.
      Ct up;
      boot_->Boot(up, R, evk);
      R = std::move(up);
    }
    // (4) y = (y r)^2  -- each y_l meets r, multiplies, squares; the result
    //     sums to 1 over live keys (r = 1/||y||).
    const int meet = param.NPToLevel(r.GetNP());
    for (int l = 0; l < T; l++) {
      Ct levelled, prod, out;
      boot_->LevelDown(levelled, y[l], meet);
      boot_->HMult(prod, levelled, r, mult_key);
      boot_->HMult(out, prod, prod, mult_key);
      y[l] = std::move(out);
    }
  }

  if (pub != nullptr) {
    AssertTrue(have_R,
               "CiBatchAttention::SoftMaxCho: the walk did not run, so there "
               "is no R_k for the public accumulator");
    *pub_scale = std::move(R);
  }

  // P = y_k, landed at forward_level.
  P.clear();
  P.resize(T);
  const int y_level = param.NPToLevel(y[0].GetNP());
  AssertTrue(y_level >= cfg_.forward_level,
             "CiBatchAttention::SoftMaxCho: P landed below forward_level");
  for (int l = 0; l < T; l++) {
    if (y_level > cfg_.forward_level)
      boot_->LevelDown(P[l], y[l], cfg_.forward_level);
    else
      P[l] = std::move(y[l]);
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
  {
    std::vector<std::vector<Ct>> lifted;
    DescendBatch(lifted, P, -1, keys);
    for (int g = 0; g < rank; g++) lp[g].resize(T);
    for (int l = 0; l < T; l++) {
      for (int g = 0; g < rank; g++) lp[g][l] = std::move(lifted[l][g]);
    }
  }

  // out[g][c]: the product's column c on the product ring, summed over the
  // two calls.
  std::vector<std::vector<Ct>> out(rank);
  for (int call = 0; call < 2; call++) {
    NvtxScope _c("batch attn: values call");
    // V with the call's key tokens kept, shifted down for the odd call by
    // the converter's premap, down to its parts.
    std::vector<std::vector<Ct>> lv(rank);
    {
      std::vector<Ct> masked(D);
      for (int c = 0; c < D; c++) {
        Ct t;
        boot_->Mult(t, v[c], call_mask_[call]);
        boot_->Rescale(masked[c], t);
      }
      std::vector<std::vector<Ct>> lifted;
      DescendBatch(lifted, masked, call, keys);
      for (int g = 0; g < rank; g++) lv[g].resize(D);
      for (int c = 0; c < D; c++) {
        for (int g = 0; g < rank; g++) lv[g][c] = std::move(lifted[c][g]);
      }
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
      t_mult_.Begin();
      ccmm_.Multiply(lifted_ctx_, prod, lhs, lv[g], 2 * cfg_.sub_degree,
                     *keys.lifted, /*rhs_row_wise=*/false);
      t_mult_.End();
      lv[g].clear();
      t_lift_descend_.Begin();
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
      t_lift_descend_.End();
    }
  }
  // Per channel: the groups' parts back into one big ciphertext, to slots.
  res.clear();
  res.resize(D);
  {
    std::vector<std::vector<Ct>> parts_list(D);
    std::vector<Ct *> outs(D);
    for (int c = 0; c < D; c++) {
      parts_list[c].resize(rank);
      for (int g = 0; g < rank; g++) parts_list[c][g] = std::move(out[g][c]);
      outs[c] = &res[c];
    }
    ReturnBatch(outs, parts_list, keys);
  }
}

template class CiBatchAttention<uint32_t>;
template class CiBatchAttention<uint64_t>;

}  // namespace cheddar
