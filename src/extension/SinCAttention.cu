#include "extension/SinCAttention.h"

#include "extension/Profile.h"

#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

namespace {

int BitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

std::pair<int, int> SplitBSGS(int num_diag) {
  int bs = 1;
  while (bs * bs < num_diag && bs < 128) bs <<= 1;
  return {bs, DivCeil(num_diag, bs)};
}

}  // namespace

template <typename word>
SinCAttention<word>::SinCAttention(
    std::shared_ptr<const BootContext<word>> boot,
    ConstContextPtr<word> switch_ctx, ConstContextPtr<word> small_ctx,
    const Config &cfg)
    : boot_{std::move(boot)},
      switch_ctx_{std::move(switch_ctx)},
      cfg_{cfg},
      num_slots_{0},
      prefix_window_{0},
      ccmm_{switch_ctx_, small_ctx, cfg.sub_degree} {
  num_slots_ = boot_->param_.degree_ / 2;
  const auto &layout = ccmm_.GetLayout();
  AssertTrue(layout.dim == cfg_.num_tokens && layout.dim == cfg_.head_dim,
             "SinCAttention: the product's width is " +
                 std::to_string(layout.dim) + ", which is neither T nor "
                 "head_dim -- pick sub_degree so it is both");
  AssertTrue(layout.lanes == cfg_.lanes,
             "SinCAttention: sub_degree/2 must be the lane count");
  AssertTrue(cfg_.gqa_group >= 1 && cfg_.lanes % cfg_.gqa_group == 0,
             "SinCAttention: the lanes must divide into whole KV groups");
  AssertTrue(boot_->GetSinCNumPhases(num_slots_) == cfg_.sinc_phases,
             "SinCAttention: call PrepareSinC(num_slots, sub_degree, "
             "sinc_level, ..., sinc_phases) first");
  AssertTrue(cfg_.sinc_level <= cfg_.swap_level - 3,
             "SinCAttention: K and V spend two swaps and the exchange before "
             "SlotToSinC, so sinc_level cannot exceed swap_level - 3");
  AssertTrue(cfg_.sinc_level - cfg_.sinc_phases >= cfg_.product_level,
             "SinCAttention: the operands cannot reach the product level");
  AssertTrue(boot_->param_.GetScale(0) == switch_ctx_->param_.GetScale(0),
             "SinCAttention: the block ring and the switching ring disagree "
             "on the level-0 scale, so the product's result cannot cross back");

  const int channels_per_ct = num_slots_ / cfg_.num_tokens;
  const int kv_heads = cfg_.lanes / cfg_.gqa_group;
  const int num_src = kv_heads * cfg_.head_dim / channels_per_ct;
  AssertTrue(num_src >= 1 && layout.num_cts % num_src == 0,
             "SinCAttention: the KV tensor does not fill a whole number of "
             "ciphertexts");
  const int log_cts = Log2Ceil(layout.num_cts);
  const int log_src = Log2Ceil(num_src);

  // THE FOUR TRANSFORMS, AND WHY THERE ARE ONLY FOUR. Q, K and V all enter at
  // the same level, and `[4|4]@3` is the first half of both `[4|7]` and
  // `[8|4]@3`, so one object serves all three. Only the second half differs.
  swap_a_ = std::make_unique<SlotPermute<word>>(
      boot_, SwapAdjacentFields(num_slots_, 4, 4, 3), cfg_.swap_level);
  swap_qv_b_ = std::make_unique<SlotPermute<word>>(
      boot_, SwapAdjacentFields(num_slots_, 3, 4), cfg_.swap_level - 1);
  swap_k_b_ = std::make_unique<SlotPermute<word>>(
      boot_, SwapAdjacentFields(num_slots_, 4, 4, 7), cfg_.swap_level - 1);
  // K and V share it: 2 ciphertexts to 8, three bits at offset 0, with GQA's
  // replication in the low bits of the field.
  exchange_ = std::make_unique<CtAxisExchange<word>>(
      boot_, log_cts, /*field_offset=*/0, cfg_.swap_level - 2, log_src);

  BuildPrefixes();
}

template <typename word>
void SinCAttention<word>::BuildPrefixes() {
  // THE PREFIX PAIR. Canonicalise is one constant multiply and one rescale at
  // the level HalfBoot lands on, and the prefix is one LinearTransform at the
  // same level, so the first folds into the second for nothing -- and folding
  // it is what restores the magnitude HalfBoot divided out. The two products
  // want different magnitudes, hence two.
  prefix_.clear();
  const StripedMatrix raw =
      boot_->SinCPrefixMatrix(num_slots_, cfg_.sub_degree, prefix_window_);
  auto [bs, gs] = SplitBSGS(raw.GetNumDiag());
  const double prod = boot_->param_.GetRescalePrimeProd(cfg_.prefix_level);
  const double target = boot_->param_.GetScale(cfg_.prefix_level - 1);
  const double magnitudes[2] = {cfg_.score_magnitude, cfg_.value_magnitude};
  for (int i = 0; i < 2; i++) {
    // The scale is Canonicalise's: `Mult` multiplies the scales and `Rescale`
    // divides by the level's actual prime product, so this is exactly what
    // lands the result on the canonical scale of `prefix_level - 1`. Without a
    // HalfBoot scale to work from, fall back to LinearTransform's own
    // convention, which preserves whatever it is handed.
    const double scale =
        (cfg_.halfboot_scale > 0.0) ? target * prod / cfg_.halfboot_scale
                                    : prod;
    const StripedMatrix m =
        (magnitudes[i] == 1.0)
            ? raw
            : StripedMatrix::Mult(raw, Complex(magnitudes[i], 0.0));
    prefix_.emplace_back(boot_, m, cfg_.prefix_level, scale, bs, gs,
                         /*pre_rotation=*/-prefix_window_,
                         /*additional_pt_rot=*/prefix_window_);
  }
}

template <typename word>
void SinCAttention<word>::SetMagnitudes(double score_magnitude,
                                        double value_magnitude) {
  cfg_.score_magnitude = score_magnitude;
  cfg_.value_magnitude = value_magnitude;
  BuildPrefixes();
}

template <typename word>
void SinCAttention<word>::AddRequiredRotations(EvkRequest &req) const {
  swap_a_->AddRequiredRotations(req);
  swap_qv_b_->AddRequiredRotations(req);
  swap_k_b_->AddRequiredRotations(req);
  exchange_->AddRequiredRotations(req);
  boot_->AddRequiredSinCRotations(req, num_slots_);
  for (const auto &lt : prefix_) lt.AddRequiredRotations(req);
  // The window the prefix leaves, undone on its output.
  req.AddRequest(num_slots_ - prefix_window_, cfg_.prefix_level - 1);
}

template <typename word>
void SinCAttention<word>::Descend(std::vector<Ct> &res,
                                  const std::vector<const Ct *> &x, Leg leg,
                                  const EvkMap<word> &evk) const {
  const auto &layout = ccmm_.GetLayout();
  const int n = layout.num_cts;
  const int sinc_level = GetSinCLevel();

  // The swaps are compiled at `swap_level`, so whatever the caller hands over
  // has to arrive there. It generally does not: the block's Q, K and V leave
  // RoPE at the block's operand level, and the leg is deliberately compiled
  // lower -- the same three transforms cost fewer limbs down there, and the
  // levels in between would be dropped before SlotToSinC anyway. So this is a
  // real LevelDown, not a formality.
  auto at_swap_level = [&](Ct &dst, const Ct &src) {
    const int level = boot_->param_.NPToLevel(src.GetNP());
    AssertTrue(level >= cfg_.swap_level,
               "SinCAttention: an operand arrived at level " +
                   std::to_string(level) + ", below the swap level " +
                   std::to_string(cfg_.swap_level));
    boot_->LevelDown(dst, src, cfg_.swap_level);
  };

  std::vector<Ct> stage(n);
  if (leg == Leg::kProb) {
    // SoftMax leaves P exactly where the product wants it. This is the whole
    // saving and it is also why it is a LevelDown and nothing else.
    AssertTrue(static_cast<int>(x.size()) == n,
               "SinCAttention: P is one ciphertext per product ciphertext");
    for (int i = 0; i < n; i++) {
      boot_->LevelDown(stage[i], *x[i], sinc_level);
    }
  } else if (leg == Leg::kQuery) {
    AssertTrue(static_cast<int>(x.size()) == n,
               "SinCAttention: Q is one ciphertext per product ciphertext");
    for (int i = 0; i < n; i++) {
      Ct in, a;
      at_swap_level(in, *x[i]);
      swap_a_->Evaluate(boot_, a, in, evk);
      // Q skips the exchange, so it arrives a level above K and V. The drop
      // that reconciles them is taken below, with everyone else's.
      swap_qv_b_->Evaluate(boot_, stage[i], a, evk);
    }
  } else {
    AssertTrue(static_cast<int>(x.size()) == exchange_->GetNumSrcCts(),
               "SinCAttention: the KV tensor is " +
                   std::to_string(exchange_->GetNumSrcCts()) +
                   " ciphertexts before the exchange");
    std::vector<Ct> swapped(x.size());
    for (size_t i = 0; i < x.size(); i++) {
      Ct in, a;
      at_swap_level(in, *x[i]);
      swap_a_->Evaluate(boot_, a, in, evk);
      if (leg == Leg::kKey) {
        swap_k_b_->Evaluate(boot_, swapped[i], a, evk);
      } else {
        swap_qv_b_->Evaluate(boot_, swapped[i], a, evk);
      }
    }
    std::vector<Ct> exchanged;
    {
      NvtxScope _n("sinc: ct-axis exchange");
      exchange_->Evaluate(exchanged, swapped, evk);
    }
    // The exchange hands back the field's value as its array index; the layout
    // wants `column / rank`, which is that value bit-reversed. Eight pointers.
    const int log_cts = Log2Ceil(n);
    for (int y = 0; y < n; y++) {
      stage[BitRev(y, log_cts)] = std::move(exchanged[y]);
    }
  }

  if (cfg_.verbose) {
    std::cout << "  SinCAttention: leg " << static_cast<int>(leg)
              << " at level " << boot_->param_.NPToLevel(stage[0].GetNP())
              << " before SlotToSinC" << std::endl;
  }
  res.clear();
  res.resize(n);
  NvtxScope _n("sinc: SlotToSinC");
  for (int i = 0; i < n; i++) {
    // SlotToSinC is compiled at `sinc_level`, and the three legs reach this
    // point at three different levels: P exactly there, Q at swap_level - 2,
    // K and V at swap_level - 3 because the exchange took one more. THIS is
    // where they are made to agree. Without it the K and V legs went into
    // SlotToSinC at whatever the exchange happened to leave, which worked only
    // when the caller had chosen sinc_level == swap_level - 3 exactly -- true
    // of SinCAttentionTest, false of the block, where it asserted inside a
    // hoist with no indication of which leg or which level was wrong.
    Ct dropped, sinc;
    boot_->LevelDown(dropped, stage[i], sinc_level);
    boot_->SlotToSinC(sinc, num_slots_, dropped, evk);
    CanonicalDown(res[i], sinc, cfg_.product_level);
  }
}

template <typename word>
void SinCAttention<word>::CanonicalDown(Ct &res, const Ct &x,
                                        int level) const {
  const auto &param = boot_->param_;
  const int from = param.NPToLevel(x.GetNP());
  AssertTrue(from >= level, "SinCAttention: CanonicalDown from level " +
                                std::to_string(from) + " to " +
                                std::to_string(level));
  if (from == level) {
    // Already there; LevelDown at its own level is the library's copy.
    boot_->LevelDown(res, x, level);
    return;
  }
  // One multiply by an exact 1.0, encoded at the scale that makes
  // `x.GetScale() * scale / prod(from)` land on canonical(from - 1) -- the
  // prefix pair's idiom, and SylphSchedule::Canonicalise's without its
  // restore constant. `1.0` is exact at any scale, so nothing is lost that an
  // ordinary rescale would not also lose.
  const double target = param.GetScale(from - 1);
  const double scale = target * param.GetRescalePrimeProd(from) / x.GetScale();
  Constant<word> one;
  boot_->encoder_.EncodeConstant(one, from, scale, 1.0);
  Ct tmp;
  boot_->Mult(tmp, x, one);
  Ct canon;
  boot_->Rescale(canon, tmp);
  canon.SetScale(target);
  // From canonical, LevelDown stays canonical: scale(l)^2 / prod(l) is
  // scale(l - 1) by Parameter's own recursion.
  if (from - 1 == level) {
    res = std::move(canon);
  } else {
    boot_->LevelDown(res, canon, level);
  }
}

template <typename word>
void SinCAttention<word>::Ascend(
    std::vector<Ct> &res, std::vector<Ct> &product,
    const LinearTransform<word> &prefix,
    const std::vector<std::vector<Complex>> &shift,
    const EvkMap<word> &evk) const {
  const int n = static_cast<int>(product.size());

  // The shift goes in HERE, at level 0 and in SinC form, because this is the
  // last point before the bootstrap and the bootstrap is what has no room for
  // its magnitude. It costs no level: a plaintext addition never does.
  if (!shift.empty()) {
    AssertTrue(static_cast<int>(shift.size()) == n,
               "SinCAttention: one shift per result ciphertext");
    const double scale = product[0].GetScale();
    if (shift != shift_cache_ || static_cast<int>(shift_pt_.size()) != n) {
      shift_pt_.clear();
      shift_pt_.resize(n);
      for (int i = 0; i < n; i++) {
        AssertTrue(static_cast<int>(shift[i].size()) == num_slots_,
                   "SinCAttention: the shift must cover every slot");
        switch_ctx_->encoder_.EncodeSinC(shift_pt_[i], 0, scale, shift[i],
                                         cfg_.sub_degree);
      }
      shift_cache_ = shift;
    }
    for (int i = 0; i < n; i++) {
      switch_ctx_->Add(product[i], product[i], shift_pt_[i]);
    }
  }

  if (cfg_.verbose) {
    std::cout << "  SinCAttention: the product left " << n
              << " ciphertexts at " << product[0].GetNP().num_main_ << " main + "
              << product[0].GetNP().num_ter_ << " terminal, scale "
              << product[0].GetScale() << std::endl;
  }
  res.clear();
  res.resize(n);
  for (int i = 0; i < n; i++) {
    // The product's result lives on the switching ring, which shares the
    // block ring's primes and secret at level 0, so it crosses back by being
    // handed over -- no words move.
    Ct half, shifted;
    boot_->HalfBoot(half, product[i], evk);
    product[i] = Ct{};  // 8 big ciphertexts at the bootstrap's level is real
    if (cfg_.verbose && i == 0) {
      std::cout << "  SinCAttention: HalfBoot landed at "
                << boot_->param_.NPToLevel(half.GetNP()) << ", scale "
                << half.GetScale() << " (the prefix was compiled at "
                << cfg_.prefix_level << " for scale " << cfg_.halfboot_scale
                << ")" << std::endl;
    }
    prefix.Evaluate(boot_, shifted, half, evk);
    const int back = num_slots_ - prefix_window_;
    boot_->HRot(res[i], shifted, evk.GetRotationKey(back), back);
  }
}

namespace {
template <typename Ct>
std::vector<const Ct *> Pointers(const std::vector<Ct> &v) {
  std::vector<const Ct *> res;
  res.reserve(v.size());
  for (const auto &c : v) res.push_back(&c);
  return res;
}
}  // namespace

template <typename word>
void SinCAttention<word>::Scores(
    std::vector<Ct> &res, const std::vector<Ct> &q, const std::vector<Ct> &k,
    const Keys &keys, const std::vector<std::vector<Complex>> &shift) const {
  Scores(res, Pointers(q), Pointers(k), keys, shift);
}

template <typename word>
void SinCAttention<word>::Values(std::vector<Ct> &res,
                                 const std::vector<Ct> &p,
                                 const std::vector<Ct> &v,
                                 const Keys &keys) const {
  Values(res, Pointers(p), Pointers(v), keys);
}

template <typename word>
void SinCAttention<word>::Scores(
    std::vector<Ct> &res, const std::vector<const Ct *> &q,
    const std::vector<const Ct *> &k, const Keys &keys,
    const std::vector<std::vector<Complex>> &shift) const {
  AssertTrue(keys.big != nullptr && keys.small != nullptr &&
                 keys.ring_switch != nullptr &&
                 keys.inverse_ring_switch != nullptr,
             "SinCAttention: Scores needs all four key sets");
  std::vector<Ct> q_op, k_op;
  {
    NvtxScope _n("sinc: descend Q");
    Descend(q_op, q, Leg::kQuery, *keys.big);
  }
  {
    NvtxScope _n("sinc: descend K");
    Descend(k_op, k, Leg::kKey, *keys.big);
  }
  std::vector<Ct> product;
  {
    NvtxScope _n("sinc: CC-MM");
    ccmm_.Multiply(product, q_op, k_op, *keys.ring_switch,
                   *keys.inverse_ring_switch, *keys.small);
  }
  q_op.clear();
  k_op.clear();
  {
    NvtxScope _n("sinc: ascend");
    Ascend(res, product, prefix_[0], shift, *keys.big);
  }
}

template <typename word>
void SinCAttention<word>::Values(std::vector<Ct> &res,
                                 const std::vector<const Ct *> &p,
                                 const std::vector<const Ct *> &v,
                                 const Keys &keys) const {
  AssertTrue(keys.big != nullptr && keys.small != nullptr &&
                 keys.ring_switch != nullptr &&
                 keys.inverse_ring_switch != nullptr,
             "SinCAttention: Values needs all four key sets");
  std::vector<Ct> p_op, v_op;
  {
    NvtxScope _n("sinc: descend P");
    Descend(p_op, p, Leg::kProb, *keys.big);
  }
  {
    NvtxScope _n("sinc: descend V");
    Descend(v_op, v, Leg::kValue, *keys.big);
  }
  std::vector<Ct> product;
  {
    NvtxScope _n("sinc: CC-MM");
    ccmm_.Multiply(product, p_op, v_op, *keys.ring_switch,
                   *keys.inverse_ring_switch, *keys.small);
  }
  p_op.clear();
  v_op.clear();
  {
    NvtxScope _n("sinc: ascend");
    Ascend(res, product, prefix_[1], {}, *keys.big);
  }
}

template class SinCAttention<uint32_t>;
template class SinCAttention<uint64_t>;

}  // namespace cheddar
