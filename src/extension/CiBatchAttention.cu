#include "extension/CiBatchAttention.h"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/EncodeGpu.h"
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

  // The three converters. The shifted forward's premap moves token t to
  // token (t + T/2) % T inside every group, so that the second call's key
  // tokens 64..127 sit at blocks 0..63 where the contract wants them.
  const int T = cfg_.num_tokens;
  std::vector<int> premap(chain_.rank * T);
  for (int t = 0; t < T; t++) {
    for (int g = 0; g < chain_.rank; g++) {
      premap[layout_.BlockOf(t, g)] = layout_.BlockOf((t + T / 2) % T, g);
    }
  }
  fwd_[0] = std::make_unique<CiSinCConverter<word>>(
      switch_ctx_, cfg_.sub_degree, cfg_.forward_level, /*inverse_level=*/-1,
      &chain_, nullptr, cfg_.converter_baby_steps);
  fwd_[1] = std::make_unique<CiSinCConverter<word>>(
      switch_ctx_, cfg_.sub_degree, cfg_.forward_level, /*inverse_level=*/-1,
      &chain_, &premap, cfg_.converter_baby_steps);
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
  fwd_[0]->AddRequiredRotations(req);
  fwd_[1]->AddRequiredRotations(req);
  inv_->AddRequiredRotations(req);
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
  Ct down;
  switch_ctx_->LevelDown(down, ct, cfg_.forward_level);
  ct = Ct();
  Ct sinc;
  fwd_[call == 1 ? 1 : 0]->SlotToSinC(switch_ctx_, sinc, down, *keys.swtch);
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

template class CiBatchAttention<uint32_t>;
template class CiBatchAttention<uint64_t>;

}  // namespace cheddar
