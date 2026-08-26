#include "extension/LlamaBlock.h"

#include "extension/Profile.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/AttentionPacking.h"

namespace cheddar {

namespace {

// PAIRING THE LIFTS, and why this one defaults ON where the ring switch did
// not. The merge is sound only while every ciphertext reaching `Lift` carries
// payload in coefficients 0..N/2-1 alone -- a property of the pipeline, not of
// the call -- and in this block it is one: `Lower` is StC applied to real-axis
// slots, which leaves the upper half zero, and the PC-MM between them mixes
// module components entrywise, so zeros stay zero. Against that contract the
// paired path is not an approximation of the unpaired one, it is the same
// arithmetic: measured 3.8e-10 against a two-HalfBoot reference, and identical
// payload precision to six decimal places (`HalfBootPairCostsOneCrossing`).
// `CHEDDAR_BOOT_PAIR=0` is the escape hatch, not the default.
bool BootPairEnabled() {
  static const bool on = []() {
    const char *env = std::getenv("CHEDDAR_BOOT_PAIR");
    return env == nullptr || std::string(env) != "0";
  }();
  return on;
}

}  // namespace

template <typename word>
LlamaBlock<word>::LlamaBlock(std::shared_ptr<const BootContext<word>> boot,
                             const Config &cfg, const Calibration &cal,
                             const LinearLeg &leg)
    : boot_{std::move(boot)}, cfg_{cfg}, cal_{cal}, leg_{&leg} {
  AssertTrue(boot_ != nullptr, "LlamaBlock: no BootContext");
  num_slots_ = boot_->param_.degree_ / 2;
  AssertTrue(cfg_.num_tokens > 0 && num_slots_ % cfg_.num_tokens == 0,
             "LlamaBlock: the token count must divide the slot count");
  channels_per_ct_ = num_slots_ / cfg_.num_tokens;
  AssertTrue(cfg_.head_dim > 0 && channels_per_ct_ % cfg_.head_dim == 0,
             "LlamaBlock: the head dimension must divide the channels one "
             "ciphertext holds, or a head would straddle two of them");
  const std::vector<int> widths = {cfg_.num_channels, cfg_.num_kv_channels,
                                   cfg_.hidden};
  for (int ch : widths) {
    AssertTrue(ch % channels_per_ct_ == 0,
               "LlamaBlock: every tensor width must be a whole number of "
               "ciphertexts");
  }

  sched_ = std::make_unique<SylphSchedule<word>>(boot_, num_slots_);
  // Every operator runs one level below where the bootstrap lands, because
  // Canonicalise spends that level putting the ciphertext on the scale
  // EvalPoly asserts. See SylphSchedule.h.
  const int op_level = sched_->GetSlotLevel() - 1;

  // The layer constants are stated for the unscaled activation and the
  // ciphertext carries `residual`, so what reaches the polynomial is
  // alpha * mean((residual * x)^2) = (alpha / residual^2) * mean(x^2). The
  // epsilon moves the same way, and both are free -- the multiplicative half
  // of the affine map is a reinterpretation of the scale and the additive half
  // is a constant addition (RmsNorm.h).
  const double r2 = cal_.residual * cal_.residual;
  attn_norm_ = std::make_unique<RmsNormHandler<word>>(
      boot_, cfg_.num_tokens, cfg_.num_channels, cal_.attn_alpha / r2, op_level,
      cfg_.eps * r2, cal_.rms_window, cal_.rms_degree);
  ffn_norm_ = std::make_unique<RmsNormHandler<word>>(
      boot_, cfg_.num_tokens, cfg_.num_channels, cal_.ffn_alpha / r2, op_level,
      cfg_.eps * r2, cal_.rms_window, cal_.rms_degree);
  // THE LEG'S LAYOUT, TAKEN ONCE. Everything below depends on it: RoPE's
  // tables are per ciphertext when the channels are permuted, SoftMax's group
  // size is the leg's score grouping, and the causal mask is built in the
  // leg's score layout. That is why the leg is a constructor argument.
  leg_->ChannelOrder(q_order_, LinearLeg::Tensor::kQuery);
  leg_->ChannelOrder(k_order_, LinearLeg::Tensor::kKey);
  leg_->ChannelOrder(v_order_, LinearLeg::Tensor::kValue);
  leg_->ChannelOrder(o_order_, LinearLeg::Tensor::kAttnOut);
  for (const auto *o : {&q_order_, &o_order_}) {
    AssertTrue(o->empty() || static_cast<int>(o->size()) == cfg_.num_channels,
               "LlamaBlock: a channel order must cover every channel");
  }
  for (const auto *o : {&k_order_, &v_order_}) {
    AssertTrue(o->empty() ||
                   static_cast<int>(o->size()) == cfg_.num_kv_channels,
               "LlamaBlock: a KV channel order must cover every KV channel");
  }

  // RoPE runs one level below the operators, on Q and K only, and each of them
  // gets its own handler because their channel orders differ. Only the
  // *distinct* maps are built: a tensor's ciphertexts repeat their layout once
  // per head group, and a duplicate map would be a duplicate set of plaintexts
  // encoded on the host for nothing.
  {
    std::vector<std::vector<int>> maps;
    int step = 0;
    BuildRoPeLayout(maps, q_variant_, step, LinearLeg::Tensor::kQuery,
                    cfg_.num_channels);
    q_rope_ = maps.empty()
                  ? std::make_unique<RoPeHandler<word>>(
                        boot_, cfg_.num_tokens, cfg_.head_dim, op_level - 1,
                        cfg_.rope_theta)
                  : std::make_unique<RoPeHandler<word>>(
                        boot_, cfg_.num_tokens, cfg_.head_dim, op_level - 1,
                        cfg_.rope_theta, std::move(maps), step);
    BuildRoPeLayout(maps, k_variant_, step, LinearLeg::Tensor::kKey,
                    cfg_.num_kv_channels);
    k_rope_ = maps.empty()
                  ? std::make_unique<RoPeHandler<word>>(
                        boot_, cfg_.num_tokens, cfg_.head_dim, op_level - 1,
                        cfg_.rope_theta)
                  : std::make_unique<RoPeHandler<word>>(
                        boot_, cfg_.num_tokens, cfg_.head_dim, op_level - 1,
                        cfg_.rope_theta, std::move(maps), step);
  }

  // The untranspose of turn D. Split into two narrow swaps for the reason
  // SlotPermute.h gives: 256 + 128 diagonals against 2048, one more level and
  // an eighth of the plaintext memory, and the memory is what binds.
  //
  // AND COMPILED AT THE BOTTOM OF TURN D, NOT THE TOP. A diagonal is a full
  // plaintext at the transform's own level, so 384 of them cost a third less
  // at level 13 than at level 18 -- and turn D has nothing else in its slot
  // leg, so descending to meet StC first is free. That is the schedule's own
  // idiom (`SylphSchedule.h`: a stage shallower than the slack descends to
  // meet StC, measured at 19.7101 bits against 19.7099 over four levels).
  if (leg_->NeedsOutputSwap()) {
    const int landing = sched_->GetStCLevel();
    AssertTrue(landing + 2 <= op_level,
               "LlamaBlock: turn D has no room for the untranspose above StC");
    untranspose_a_ = std::make_unique<SlotPermute<word>>(
        boot_, SwapAdjacentFields(num_slots_, 4, 3), landing + 2);
    untranspose_b_ = std::make_unique<SlotPermute<word>>(
        boot_, SwapAdjacentFields(num_slots_, 4, 4, 3), landing + 1);
  }
  // The auxiliary track goes round this cycle, not through `Boot`. `Boot`
  // lands at GetEndLevel(), which the slack pushes below the main track, so
  // SoftMax's own boot_aux asserts on any set that reserves slack -- which is
  // every set this schedule runs on. The hook Run installs is StC, HalfBoot
  // and Canonicalise, which is one whole turn of the cycle taken by a value
  // that is not the main track.
  //
  // A score row spans `group_size` ciphertexts on a slot-domain leg, because
  // the CC-MM puts three bits of the key on the ciphertext axis. `num_keys` is
  // therefore per ciphertext and the group carries the rest.
  const int group_size = leg_->GetScoreGroupSize();
  AssertTrue(group_size >= 1 && cfg_.num_tokens % group_size == 0,
             "LlamaBlock: the score group size must divide the token count");
  softmax_ = std::make_unique<SoftMaxHandler<word>>(
      boot_, cfg_.num_tokens / group_size, cal_.softmax_range, op_level,
      cal_.softmax_iters, cal_.softmax_norm_lo, cal_.softmax_norm_hi,
      cal_.softmax_exp_degree, cal_.softmax_inv_sqrt_degree,
      cal_.softmax_early_inv_sqrt_degree,
      /*boot_aux=*/false, /*aux_return_level=*/op_level,
      /*aux_boot_max=*/cal_.boot_max, group_size);
  AssertTrue(softmax_->GetAuxCallLevel() == sched_->GetStCLevel(),
             "LlamaBlock: SoftMax's auxiliary value lands on level " +
                 std::to_string(softmax_->GetAuxCallLevel()) +
                 " but StC is compiled at " +
                 std::to_string(sched_->GetStCLevel()) +
                 ", so the cycle cannot pick it up. The exponential's degree "
                 "and the number of iterations set that level.");
  silu_ = std::make_unique<SiLuHandler<word>>(boot_, cal_.silu_range, op_level,
                                              cal_.silu_degree);
}

template <typename word>
int LlamaBlock<word>::NumCiphertexts(int channels) const {
  return channels / channels_per_ct_;
}

template <typename word>
int LlamaBlock<word>::OutSwapDepth() const {
  // The two swaps cost two levels, but they are placed at the bottom of the
  // turn, so the turn's slot leg spends everything down to StC. That is what
  // the budget has to be checked against, and it is the whole budget -- which
  // is correct, because turn D has nothing else to spend it on.
  return leg_->NeedsOutputSwap() ? (GetResultLevel() - sched_->GetStCLevel())
                                 : 0;
}

template <typename word>
int LlamaBlock<word>::NumScoreCiphertexts() const {
  const int heads = cfg_.num_channels / cfg_.head_dim;
  return heads * cfg_.num_tokens * cfg_.num_tokens / num_slots_;
}

template <typename word>
const std::vector<double> &LlamaBlock<word>::Reorder(
    const std::vector<double> &w, int rows, int cols,
    typename LinearLeg::Tensor which, bool permute_rows) const {
  const std::vector<int> *order = nullptr;
  int slot = 0;
  switch (which) {
    case LinearLeg::Tensor::kQuery: order = &q_order_; slot = 0; break;
    case LinearLeg::Tensor::kKey: order = &k_order_; slot = 1; break;
    case LinearLeg::Tensor::kValue: order = &v_order_; slot = 2; break;
    case LinearLeg::Tensor::kAttnOut: order = &o_order_; slot = 3; break;
  }
  // The common case, and the one worth not copying: 14336 x 4096 doubles is
  // 470 MB and the FFN's three matrices are never reordered.
  if (order->empty()) return w;

  Reordered &held = reordered_[slot];
  if (held.source == w.data() && held.size == w.size()) return held.data;
  std::vector<double> &buf = held.data;
  held.source = w.data();
  held.size = w.size();
  AssertTrue(w.size() == static_cast<size_t>(rows) * cols,
             "LlamaBlock::Reorder: the weight is not rows x cols");
  buf.assign(w.size(), 0.0);
  if (permute_rows) {
    AssertTrue(static_cast<int>(order->size()) == rows,
               "LlamaBlock::Reorder: the row order must cover every row");
    for (int i = 0; i < rows; i++) {
      std::copy(w.begin() + static_cast<size_t>(i) * cols,
                w.begin() + static_cast<size_t>(i + 1) * cols,
                buf.begin() + static_cast<size_t>((*order)[i]) * cols);
    }
  } else {
    AssertTrue(static_cast<int>(order->size()) == cols,
               "LlamaBlock::Reorder: the column order must cover every column");
    for (int i = 0; i < rows; i++) {
      const double *src = &w[static_cast<size_t>(i) * cols];
      double *dst = &buf[static_cast<size_t>(i) * cols];
      for (int j = 0; j < cols; j++) dst[(*order)[j]] = src[j];
    }
  }
  return buf;
}

template <typename word>
void LlamaBlock<word>::BuildRoPeLayout(std::vector<std::vector<int>> &maps,
                                       std::vector<int> &variant, int &step,
                                       typename LinearLeg::Tensor which,
                                       int channels) const {
  const std::vector<int> &order =
      (which == LinearLeg::Tensor::kQuery) ? q_order_ : k_order_;
  maps.clear();
  variant.assign(NumCiphertexts(channels), 0);
  step = 0;
  if (order.empty()) return;  // the default packing; RoPeHandler knows it

  // position -> (head * head_dim + channel), the inverse of the leg's order
  std::vector<int> inv(channels, -1);
  for (int i = 0; i < channels; i++) {
    AssertTrue(order[i] >= 0 && order[i] < channels && inv[order[i]] == -1,
               "LlamaBlock: a channel order is not a permutation");
    inv[order[i]] = i;
  }

  const int num_ct = NumCiphertexts(channels);
  const int T = cfg_.num_tokens;
  const int D = cfg_.head_dim;
  const int half = D / 2;

  // The partner distance, read off slot 0 of ciphertext 0 and then verified
  // over every slot of every map by RoPeHandler's own constructor.
  const int m0 = inv[0];
  const int c0 = m0 % D;
  const int m1 = (c0 < half) ? m0 + half : m0 - half;
  const int p1 = order[m1];
  AssertTrue(p1 / channels_per_ct_ == 0,
             "LlamaBlock: the channel order puts a RoPE pair in two different "
             "ciphertexts, so the partner cannot be rotated in");
  const int s1 = (p1 % channels_per_ct_) * T;
  step = (c0 < half) ? s1 : (num_slots_ - s1);

  // ONE MAP PER DISTINCT LAYOUT, NOT ONE PER CIPHERTEXT. A tensor's layout
  // repeats once per head group, so Q's sixteen ciphertexts hold eight
  // distinct maps -- and a map is three encoded plaintexts on the host, which
  // is the expensive part of RoPE.
  std::vector<std::vector<int>> all(num_ct, std::vector<int>(num_slots_, 0));
  for (int ct = 0; ct < num_ct; ct++) {
    for (int s = 0; s < num_slots_; s++) {
      all[ct][s] = inv[ct * channels_per_ct_ + s / T] % D;
    }
  }
  for (int ct = 0; ct < num_ct; ct++) {
    int found = -1;
    for (size_t j = 0; j < maps.size(); j++) {
      if (maps[j] == all[ct]) { found = static_cast<int>(j); break; }
    }
    if (found < 0) {
      found = static_cast<int>(maps.size());
      maps.push_back(all[ct]);
    }
    variant[ct] = found;
  }
}

template <typename word>
void LlamaBlock<word>::ApplyRoPe(std::vector<Ct> &res,
                                 const std::vector<Ct> &x,
                                 const RoPeHandler<word> &rope,
                                 const std::vector<int> &variant,
                                 const EvkMap<word> &evk_map) const {
  AssertTrue(variant.size() == x.size(),
             "LlamaBlock::ApplyRoPe: one variant per ciphertext");
  res.resize(x.size());
  // VARIANT-MAJOR, AND THAT IS THE WHOLE REASON THIS HELPER EXISTS. RoPeHandler
  // keeps one set of encoded tables, because holding all of them would be
  // three plaintexts per distinct layout at the operator's level. Encoding a
  // set is a SpecialIFFT plus a prime-by-prime reduction over every slot on
  // the host -- RoPE's measured 171 ms, against SiLU's 7.8 ms for a much
  // heavier circuit -- so the loop is ordered to pay it once per layout.
  const int variants = rope.GetNumVariants();
  for (int v = 0; v < variants; v++) {
    for (size_t i = 0; i < x.size(); i++) {
      if (variant[i] != v) continue;
      rope.Apply(res[i], x[i], cfg_.first_position, evk_map, v);
    }
  }
}

template <typename word>
void LlamaBlock<word>::BuildCausalMask(
    std::vector<std::vector<Complex>> &res) const {
  const int T = cfg_.num_tokens;
  const int heads = cfg_.num_channels / cfg_.head_dim;
  res.assign(NumScoreCiphertexts(),
             std::vector<Complex>(num_slots_, Complex(0.0, 0.0)));
  for (int h = 0; h < heads; h++) {
    for (int query = 0; query < T; query++) {
      for (int key = 0; key <= query; key++) {
        int ct = 0, slot = 0;
        leg_->LocateScore(h, query, key, ct, slot);
        AssertTrue(ct >= 0 && ct < static_cast<int>(res.size()) && slot >= 0 &&
                       slot < num_slots_,
                   "LlamaBlock: the leg's score layout is out of range");
        res[ct][slot] = Complex(1.0, 0.0);
      }
    }
  }
}

template <typename word>
void LlamaBlock<word>::Untranspose(std::vector<Ct> &res,
                                   const std::vector<Ct> &x,
                                   const EvkMap<word> &evk_map) const {
  res.resize(x.size());
  const int entry = sched_->GetStCLevel() + 2;
  for (size_t i = 0; i < x.size(); i++) {
    Ct dropped, a;
    boot_->LevelDown(dropped, x[i], entry);
    untranspose_a_->Evaluate(boot_, a, dropped, evk_map);
    untranspose_b_->Evaluate(boot_, res[i], a, evk_map);
  }
}

template <typename word>
void LlamaBlock<word>::AddRequiredRotations(EvkRequest &req) const {
  const int op_level = sched_->GetSlotLevel() - 1;
  boot_->AddRequiredRotations(req, num_slots_);
  for (int d : attn_norm_->GetRotationDistances()) req.AddRequest(d, op_level);
  for (int d : q_rope_->GetRotationDistances()) req.AddRequest(d, op_level - 1);
  for (int d : k_rope_->GetRotationDistances()) req.AddRequest(d, op_level - 1);
  for (int d : softmax_->GetRotationDistances()) req.AddRequest(d, op_level);
  if (untranspose_a_ != nullptr) {
    untranspose_a_->AddRequiredRotations(req);
    untranspose_b_->AddRequiredRotations(req);
  }
}

// The six turns, as the schedule sees them. `Stage::linear_depth` is what the
// product leg spends below StC's output level; the product itself runs at the
// bottom of the ladder, so this is the descent plus the one level [BAE]'s
// PC-MM and [KANG]'s CC-MM each consume.
namespace {
template <typename word>
std::vector<typename SylphSchedule<word>::Stage> TurnsOf(
    const typename LlamaBlock<word>::Calibration &cal, int out_swap) {
  const int rms = 1 + Log2Ceil(cal.rms_degree + 1) + 2 + 1;
  // RmsNormHandler::Apply consumes one level for the square, ceil(log2(d+1))
  // for the polynomial and two more to apply the result and the weight -- and
  // then *owes* one, because it ends with Mult(res, res, weight_pt_) and no
  // Rescale. A schedule that counts only what an operator consumes puts the
  // ciphertext on StC's level still owing a rescale, and the failure is silent
  // and total (Doing.md 1.5aa). So the budget is consumed + owed.
  const int softmax = 1 + Log2Ceil(cal.softmax_exp_degree + 1) +
                      2 * cal.softmax_iters;
  const int silu = Log2Ceil(cal.silu_degree + 1) + 1;  // the fit, then * up
  using Stage = typename SylphSchedule<word>::Stage;
  // Turns B and C spend NOTHING on the linear leg of this schedule. Their
  // products take the SinC road instead -- swaps, exchange, SlotToSinC, the
  // ring switch, the product, HalfBoot and the prefix -- and the leg checks
  // that road's own arithmetic, which is a different set of constraints
  // (`SinCAttention`'s constructor states them). What this schedule still owns
  // for those turns is the slot leg above the handover: RoPE in B, SoftMax in
  // C, both of which must land on the level the leg takes its operands at.
  return {
      Stage{"A  RMSNorm(attn) -> Q,K,V", rms, 2},
      Stage{"B  RoPE(Q), RoPE(K), carry V -> S = QK^T", 1, 0},
      Stage{"C  SoftMax(S) -> A = PV", softmax, 0},
      Stage{"D  untranspose A -> O = A W_o", out_swap, 2},
      Stage{"E  RMSNorm(ffn) -> G,U", rms, 2},
      Stage{"F  SiLU(G) * U -> Y = . W_down", silu, 2},
  };
}
}  // namespace

template <typename word>
bool LlamaBlock<word>::Fits(std::string *why) const {
  for (const auto &stage : TurnsOf<word>(cal_, OutSwapDepth())) {
    if (!sched_->Fits(stage, why)) return false;
  }
  // SoftMax has to end exactly where the leg picks P up. One level above and
  // the LevelDown inside the leg absorbs it silently; one below and there is
  // nothing to absorb, and the failure is a prime-count mismatch four frames
  // deep in a transform.
  if (softmax_->GetAuxCallLevel() < GetProbLevel()) {
    if (why != nullptr) {
      *why = "C: SoftMax leaves P at level " +
             std::to_string(softmax_->GetAuxCallLevel()) +
             " but the product leg takes it at " +
             std::to_string(GetProbLevel());
    }
    return false;
  }
  return true;
}

template <typename word>
std::string LlamaBlock<word>::DescribePlan() const {
  std::ostringstream os;
  os << sched_->DescribePlan(TurnsOf<word>(cal_, OutSwapDepth()));
  os << "packing: " << cfg_.num_tokens << " tokens x " << channels_per_ct_
     << " channels per ciphertext" << std::endl;
  os << "  hidden state " << NumCiphertexts(cfg_.num_channels) << " ct, kv "
     << NumCiphertexts(cfg_.num_kv_channels) << " ct, ffn "
     << NumCiphertexts(cfg_.hidden) << " ct, scores "
     << (cfg_.num_channels / cfg_.head_dim) * cfg_.num_tokens * cfg_.num_tokens /
            num_slots_
     << " ct" << std::endl;
  // Every ciphertext crossing a turn boundary is bootstrapped, including the
  // ones whose turn has no operator in it. That count is the block's cost.
  const int h = NumCiphertexts(cfg_.num_channels);
  const int kv = NumCiphertexts(cfg_.num_kv_channels);
  const int ffn = NumCiphertexts(cfg_.hidden);
  const int scores = (cfg_.num_channels / cfg_.head_dim) * cfg_.num_tokens *
                     cfg_.num_tokens / num_slots_;
  // WHERE THE BOOTSTRAPS ARE NOW. Turn B lifts Q, K and V and the leg
  // bootstraps every score ciphertext on the way out of the product; turn C
  // runs SoftMax's auxiliary track once per row group and the leg bootstraps
  // the attention output; turn D lifts nothing at all, because the attention
  // output came back in slots.
  const int groups = scores / std::max(1, leg_->GetScoreGroupSize());
  const int total = h + (h + kv + kv + scores) + (groups + scores) + h + 2 * ffn;
  os << "bootstraps: A " << h << ", B " << (h + kv + kv) << " + " << scores
     << " in the leg, C " << groups << " aux + " << scores
     << " in the leg, D 0, E " << h << ", F " << (2 * ffn) << " = " << total
     << " per block" << std::endl;

  // WHAT THE SCORE BOOTSTRAP ACTUALLY CARRIES, which is not a free knob --
  // see Calibration::size_scores. The shift centres each row on its own
  // calibrated maximum before the crossing, so what has to fit is half the
  // SoftMax interval and not the raw scores.
  const double crossing = cal_.size_q * cal_.size_k *
                          std::sqrt(static_cast<double>(cfg_.head_dim)) *
                          cal_.softmax_range / 2.0;
  os << "score crossing: size_q * size_k * sqrt(head_dim) * range / 2 = "
     << crossing << " against boot_max " << cal_.boot_max;
  if (crossing > cal_.boot_max) os << "  (TOO LARGE -- lower size_q/size_k)";
  os << std::endl;
  return os.str();
}

// A hundred bootstraps is minutes of silence, and an assert inside one of them
// says only that the prime counts differed. Naming the turn costs a line of
// output and turns "somewhere in the block" into "turn F, the gate multiply".
template <typename word>
void LlamaBlock<word>::Announce(const char *what, const std::vector<Ct> &cts,
                                int expect_level) const {
  const int level = cts.empty() ? -1 : boot_->param_.NPToLevel(cts[0].GetNP());
  std::cout << "  [block] " << what << ": " << cts.size() << " ct at level "
            << level;
  if (expect_level >= 0 && level != expect_level) {
    std::cout << "  (EXPECTED " << expect_level << ")";
  }
  // Device-free at every turn boundary. Memory is what binds this block --
  // the setup alone is tens of gigabytes of keys and transform plaintexts --
  // and an allocation failure four frames inside a bootstrap says only
  // "out_of_memory", never which turn asked.
  size_t dev_free = 0, dev_total = 0;
  if (cudaMemGetInfo(&dev_free, &dev_total) == cudaSuccess) {
    std::cout << ", " << (dev_free >> 20) << " MiB free";
  }
  std::cout << std::endl;
}

template <typename word>
void LlamaBlock<word>::Lift(std::vector<Ct> &res, const std::vector<Ct> &x,
                            double magnitude, const EvkMap<word> &evk_map,
                            double shift) const {
  res.resize(x.size());
  const int level = sched_->GetSlotLevel() - 1;
  Constant<word> shift_const;
  if (shift != 0.0) {
    boot_->encoder_.EncodeConstant(shift_const, level,
                                   boot_->param_.GetScale(level), shift);
  }
  // Two ciphertexts to a crossing where the pipeline allows it. The magnitude
  // and the shift are applied after the split, per ciphertext, so pairing
  // changes nothing downstream: `Canonicalise` sees exactly the ciphertext it
  // would have seen.
  size_t i = 0;
  if (BootPairEnabled()) {
    for (; i + 1 < x.size(); i += 2) {
      Ct landed_lo, landed_hi;
      sched_->ToSlotPair(landed_lo, landed_hi, x[i], x[i + 1], evk_map);
      sched_->Canonicalise(res[i], landed_lo, magnitude);
      sched_->Canonicalise(res[i + 1], landed_hi, magnitude);
      if (shift != 0.0) {
        boot_->Add(res[i], res[i], shift_const);
        boot_->Add(res[i + 1], res[i + 1], shift_const);
      }
    }
  }
  for (; i < x.size(); i++) {
    Ct landed;
    sched_->ToSlot(landed, x[i], evk_map);
    sched_->Canonicalise(res[i], landed, magnitude);
    if (shift != 0.0) boot_->Add(res[i], res[i], shift_const);
  }
}

template <typename word>
void LlamaBlock<word>::Lower(std::vector<Ct> &res, const std::vector<Ct> &x,
                             const EvkMap<word> &evk_map) const {
  res.resize(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    sched_->ToCoeff(res[i], x[i], evk_map);
  }
}

template <typename word>
void LlamaBlock<word>::SpreadNormWeight(
    std::vector<std::vector<Complex>> &res, const std::vector<double> &w,
    double alpha) const {
  const int num_ct = NumCiphertexts(cfg_.num_channels);
  // RmsNormHandler asks for sqrt(alpha_L) folded into the weight, because the
  // circuit evaluates 1/sqrt(alpha * mean(x^2)) and the missing sqrt(alpha) has
  // to come back somewhere. The weight plaintext is where it costs nothing.
  const double root = std::sqrt(alpha);
  res.assign(num_ct, std::vector<Complex>(num_slots_, Complex(0.0, 0.0)));
  for (int i = 0; i < num_ct; i++) {
    for (int s = 0; s < num_slots_; s++) {
      const int c = i * channels_per_ct_ + s / cfg_.num_tokens;
      res[i][s] = Complex(w[c] * root, 0.0);
    }
  }
}

template <typename word>
void LlamaBlock<word>::InjectSinks(std::vector<Ct> &cts,
                                   const std::vector<double> &want,
                                   const std::vector<double> &got, int channels,
                                   double scale,
                                   const std::vector<int> &order) const {
  const int sinks = cfg_.num_sink_tokens;
  if (sinks <= 0) return;
  const size_t expect = static_cast<size_t>(sinks) * channels;
  AssertTrue(want.size() == expect && got.size() == expect,
             "LlamaBlock::InjectSinks: the public sink rows must be "
             "[num_sink_tokens][channels]");
  AssertTrue(static_cast<int>(cts.size()) == NumCiphertexts(channels),
             "LlamaBlock::InjectSinks: wrong ciphertext count");

  // The projection's columns were reordered, so slot position `p` holds the
  // model's channel `inv[p]` and not `p`. The sink rows are stated in the
  // model's order, which is the only order they mean anything in.
  std::vector<int> inv;
  if (!order.empty()) {
    AssertTrue(static_cast<int>(order.size()) == channels,
               "LlamaBlock::InjectSinks: the channel order must cover every "
               "channel of the tensor");
    inv.assign(channels, -1);
    for (int c = 0; c < channels; c++) inv[order[c]] = c;
  }

  const int degree = boot_->param_.degree_;
  const int level = 0;
  for (size_t i = 0; i < cts.size(); i++) {
    AssertTrue(boot_->param_.NPToLevel(cts[i].GetNP()) == level,
               "LlamaBlock::InjectSinks: the projection must have landed at "
               "level 0 before a plaintext can be added to it");
    std::vector<double> coeffs(degree, 0.0);
    bool any = false;
    for (int t = 0; t < sinks; t++) {
      for (int c = 0; c < channels_per_ct_; c++) {
        const int position = static_cast<int>(i) * channels_per_ct_ + c;
        const int channel = inv.empty() ? position : inv[position];
        const size_t idx = static_cast<size_t>(t) * channels + channel;
        const double d = scale * (want[idx] - got[idx]);
        if (d == 0.0) continue;
        const int slot = t + cfg_.num_tokens * c;
        coeffs[AttentionPacking::CoeffOfSlot({slot, false}, degree)] = d;
        any = true;
      }
    }
    if (!any) continue;
    Plaintext<word> pt;
    boot_->encoder_.EncodeCoeff(pt, level, cts[i].GetScale(), coeffs);
    Ct sum;
    boot_->Add(sum, cts[i], pt);
    cts[i] = std::move(sum);
  }
}

template <typename word>
void LlamaBlock<word>::Run(std::vector<Ct> &res, const std::vector<Ct> &x,
                           const Weights &w, const PublicSinks &sinks,
                           const EvkMap<word> &evk_map) const {
  std::string why;
  AssertTrue(Fits(&why), "LlamaBlock: the plan does not close -- " + why);
  AssertTrue(static_cast<int>(x.size()) == NumCiphertexts(cfg_.num_channels),
             "LlamaBlock: wrong number of input ciphertexts");

  const LinearLeg &leg = *leg_;
  const double r = cal_.residual;

  // ---- turn A: RMSNorm(attn), then the QKV projection -------------------
  //
  // The input carries `residual`, and RMSNorm is invariant to it -- that is
  // exactly why the layer constant was divided by residual^2 in the
  // constructor. So the normalised activation is the true one and every
  // crossing constant below is stated against unscaled tensors.
  std::vector<Ct> slots, normed, coeff;
  Announce("A  input", x, 0);
  {
    ProfileScope _p("A  lift x");
    Lift(slots, x, 1.0, evk_map);
  }
  Announce("A  lifted", slots, sched_->GetSlotLevel() - 1);
  std::vector<std::vector<Complex>> attn_w;
  {
    ProfileScope _p("A  RMSNorm(attn)");
    SpreadNormWeight(attn_w, w.attn_norm, cal_.attn_alpha / (r * r));
    attn_norm_->Apply(normed, slots, attn_w, evk_map);
  }
  // One above StC, not on it: Apply ends with Mult(res, res, weight_pt_) and
  // no Rescale, so the ciphertext arrives owing one. ToCoeff settles it.
  Announce("A  RMSNorm (owes a rescale)", normed, sched_->GetStCLevel() + 1);
  {
    ProfileScope _p("A  lower");
    Lower(coeff, normed, evk_map);
  }
  Announce("A  lowered", coeff, sched_->GetCoeffLevel());

  // THE CHANNEL ORDER RIDES ON THE WEIGHT AND COSTS NOTHING. A projection's
  // output channel order is the column order of its matrix, so the layout the
  // ciphertext-ciphertext product needs -- head in the four slot bits above
  // the token, three channel bits on the ciphertext axis -- is bought here, in
  // a host-side permutation of a plaintext, rather than with a slot transform
  // and a level on each of Q, K and V.
  std::vector<Ct> q, k, v;
  {
    ProfileScope _p("A  project Q");
    leg.Project(q, coeff, cfg_.num_channels, cfg_.num_channels,
                Reorder(w.wq, cfg_.num_channels, cfg_.num_channels,
                        LinearLeg::Tensor::kQuery, false),
                cal_.size_q, "Q");
  }
  {
    ProfileScope _p("A  project K");
    leg.Project(k, coeff, cfg_.num_channels, cfg_.num_kv_channels,
                Reorder(w.wk, cfg_.num_channels, cfg_.num_kv_channels,
                        LinearLeg::Tensor::kKey, false),
                cal_.size_k, "K");
  }
  {
    ProfileScope _p("A  project V");
    leg.Project(v, coeff, cfg_.num_channels, cfg_.num_kv_channels,
                Reorder(w.wv, cfg_.num_channels, cfg_.num_kv_channels,
                        LinearLeg::Tensor::kValue, false),
                cal_.size_v, "V");
  }

  // The sink tokens' K and V, put back. Their hidden state never reached the
  // encrypted RMSNorm -- a public filler stood in for it, which is the only
  // way the layer constant covers every token at once -- so what the
  // projection just produced on those rows is a public number, and so is what
  // it should have been. Adding the difference is a plaintext addition and
  // costs nothing. It happens before RoPE, which then rotates the sink keys by
  // their own positions exactly as it would have.
  {
    ProfileScope _p("A  inject sinks");
    InjectSinks(k, sinks.k, sinks.computed_k, cfg_.num_kv_channels, cal_.size_k,
                k_order_);
    InjectSinks(v, sinks.v, sinks.computed_v, cfg_.num_kv_channels, cal_.size_v,
                v_order_);
  }

  // ---- turn B: RoPE on Q and K, then the score product ------------------
  //
  // RoPE is linear, so it commutes with the crossing constant and the
  // ciphertexts are lifted at magnitude 1 -- whatever they carry, they carry
  // it through unchanged and `Scores` divides both out.
  std::vector<Ct> q_slots, k_slots, v_slots;
  Announce("B  Q from the projection", q, 0);
  {
    ProfileScope _p("B  lift Q,K,V");
    Lift(q_slots, q, 1.0, evk_map);
    Lift(k_slots, k, 1.0, evk_map);
    Lift(v_slots, v, 1.0, evk_map);
  }
  q.clear();
  k.clear();
  v.clear();
  std::vector<Ct> q_rot(q_slots.size()), k_rot(k_slots.size());
  {
    ProfileScope _p("B  RoPE Q");
    ApplyRoPe(q_rot, q_slots, *q_rope_, q_variant_, evk_map);
  }
  {
    ProfileScope _p("B  RoPE K");
    ApplyRoPe(k_rot, k_slots, *k_rope_, k_variant_, evk_map);
  }
  q_slots.clear();
  k_slots.clear();
  Announce("B  RoPE", q_rot, GetOperandLevel());
  // V takes no part in turn B's slot leg, but it has to arrive at the product
  // alongside Q and K, so it spends RoPE's level doing nothing. A LevelDown is
  // free; what is not free is the bootstrap above, and that one is unavoidable
  // -- the projection left V at level 0 and the product does not start there.
  {
    ProfileScope _p("B  V level down");
    for (size_t i = 0; i < v_slots.size(); i++) {
      Ct dropped;
      boot_->LevelDown(dropped, v_slots[i], GetOperandLevel());
      v_slots[i] = std::move(dropped);
    }
  }

  // WHAT SOFTMAX WANTS IS 2 (s - c) / M + 1 ON [-1, 1], AND THE WHOLE AFFINE
  // MAP IS FREE -- but only in this order.
  //
  // The shift goes in first, on the raw product, before the bootstrap; the
  // scaling goes in afterwards, in the transform that replaces Canonicalise.
  // Both are free -- an addition is a plaintext and the scaling rides a
  // multiply that was already being paid for -- so the order is not about
  // cost. It is about what the bootstrap in between has to carry.
  //
  // Shift first, and the crossing carries s - c, which the calibration bounds
  // by range/2 because c IS the row's calibrated maximum. Scale first and the
  // crossing carries s, which nothing bounds: a row's scores sit wherever its
  // own maximum puts them, and on the real layer-2 scores that is twelve nats
  // of spread between rows.
  //
  // The +1 rides in the same shift, for nothing, because the shift is already
  // per entry -- see Calibration::softmax_mask_shift for why it has to be.
  std::vector<Ct> scores;
  const int T = cfg_.num_tokens;
  const double root = std::sqrt(static_cast<double>(cfg_.head_dim));
  // The product carries both operands' crossing constants and a factor
  // sqrt(head_dim) from the contraction; `magnitude` divides all three out and
  // applies SoftMax's own 2/range. It is applied AFTER the shift, which is
  // what makes the crossing carry range/2 instead of the raw scores.
  const double raw = cal_.size_q * cal_.size_k * root;
  const double score_magnitude = 2.0 / (cal_.softmax_range * raw);
  const int num_rows = (cfg_.num_channels / cfg_.head_dim) * T;
  AssertTrue(static_cast<int>(cal_.softmax_shift.size()) == num_rows,
             "LlamaBlock: SoftMax needs one calibrated shift per row -- see "
             "Calibration::softmax_shift for why a single global one does not "
             "work on real scores");
  std::vector<double> score_shift(static_cast<size_t>(num_rows) * T);
  {
    ProfileScope _p("B  score shift (host)");
    for (int rr = 0; rr < num_rows; rr++) {
      const int query = rr % T;
      for (int key = 0; key < T; key++) {
        // Masked entries get pushed down by the calibrated gap so that the
        // exponential still sees an in-interval argument; the mask zeroes them
        // a level later, but the polynomial has already run by then.
        const double extra = (key <= query) ? 0.0 : cal_.softmax_mask_shift;
        // magnitude * (raw_product + shift) = 2 (u - c - extra) / range + 1, so
        // the shift is the whole affine map's constant carried back through the
        // magnitude -- the +1 included, which is why SoftMax's argument needs no
        // further addition once the product has come back.
        score_shift[static_cast<size_t>(rr) * T + key] =
            (cal_.softmax_range / 2.0 - cal_.softmax_shift[rr] - extra) * raw;
      }
    }
  }
  {
    ProfileScope _p("B  score product Q K^T");
    leg.Scores(scores, q_rot, k_rot, score_magnitude, score_shift);
  }
  q_rot.clear();
  k_rot.clear();
  AssertTrue(static_cast<int>(scores.size()) == NumScoreCiphertexts(),
             "LlamaBlock: the leg returned " + std::to_string(scores.size()) +
                 " score ciphertexts against the " +
                 std::to_string(NumScoreCiphertexts()) + " the packing holds");

  // ---- turn C: SoftMax over a row that may span several ciphertexts ------
  //
  // The scores arrive from the leg already in slots, on the operator's level
  // and its canonical scale, with the affine map already applied -- there is
  // no Lift here, and that is the seam. The transform that stands in for
  // `Canonicalise` inside the leg did the same job on the way out of the
  // product's bootstrap.
  typename SoftMaxHandler<word>::AuxBoot aux =
      [this, &evk_map](Ct &out, const Ct &in, double magnitude) {
        Ct coeff, landed;
        sched_->ToCoeff(coeff, in, evk_map);
        sched_->ToSlot(landed, coeff, evk_map);
        sched_->Canonicalise(out, landed, magnitude);
      };
  Announce("C  scores from the product", scores, GetResultLevel());
  std::vector<std::vector<Complex>> causal_mask;
  {
    ProfileScope _p("C  causal mask (host)");
    BuildCausalMask(causal_mask);
  }
  const int group = leg.GetScoreGroupSize();
  std::vector<Ct> probs(scores.size());
  {
    ProfileScope _p("C  SoftMax");
    for (size_t base = 0; base < scores.size(); base += group) {
      std::vector<Ct> in, out;
      std::vector<std::vector<Complex>> mask;
      in.reserve(group);
      mask.reserve(group);
      // Moved in, not copied: a row group is `group` ciphertexts at the
      // operator's level and SoftMax holds several more of each while it runs,
      // so the peak is one group rather than the layer.
      for (int j = 0; j < group; j++) {
        in.push_back(std::move(scores[base + j]));
        mask.push_back(causal_mask[base + j]);
      }
      softmax_->Apply(out, in, mask, evk_map, nullptr, &aux);
      for (int j = 0; j < group; j++) probs[base + j] = std::move(out[j]);
    }
  }
  scores.clear();
  causal_mask.clear();
  Announce("C  SoftMax", probs, GetProbLevel());

  std::vector<Ct> attn;
  {
    ProfileScope _p("C  value product P V");
    leg.Values(attn, probs, v_slots, cal_.size_attn / cal_.size_v);
  }
  probs.clear();
  v_slots.clear();

  // ---- turn D: carry the attention output into the O projection ---------
  //
  // O is a second product and the product ring holds one level, so it cannot
  // follow PV directly. `residual` rather than a private constant, because
  // this output is added to the block input.
  //
  // There is no Lift here either. What there is instead is the untranspose:
  // the CC-MM's lane index is the four slot bits above the token, so the
  // attention output comes back with the head there and the token above it,
  // and the O projection reads token-fastest. Two narrow field swaps put it
  // back, and W_o's ROWS carry the rest of the layout for nothing.
  std::vector<Ct> attn_coeff, attn_out;
  Announce("D  attention values", attn, GetResultLevel());
  if (untranspose_a_ != nullptr) {
  {
    ProfileScope _p("D  untranspose");
      std::vector<Ct> flat;
      Untranspose(flat, attn, evk_map);
      attn = std::move(flat);
      Announce("D  untransposed", attn, sched_->GetStCLevel());
  }
  }
  {
    ProfileScope _p("D  lower");
    Lower(attn_coeff, attn, evk_map);
  }
  attn.clear();
  {
    ProfileScope _p("D  project O");
    leg.Project(attn_out, attn_coeff, cfg_.num_channels, cfg_.num_channels,
                Reorder(w.wo, cfg_.num_channels, cfg_.num_channels,
                        LinearLeg::Tensor::kAttnOut, true),
                r / cal_.size_attn, "O");
  }

  std::vector<Ct> h(x.size());
  {
    ProfileScope _p("D  residual + attn");
    for (size_t i = 0; i < x.size(); i++) {
      boot_->Add(h[i], x[i], attn_out[i]);
    }
  }

  // ---- turn E: RMSNorm(ffn), then the gate and up projections -----------
  std::vector<Ct> h_slots, h_normed, h_coeff;
  {
    ProfileScope _p("E  lift");
    Lift(h_slots, h, 1.0, evk_map);
  }
  std::vector<std::vector<Complex>> ffn_w;
  {
    ProfileScope _p("E  RMSNorm(ffn)");
    SpreadNormWeight(ffn_w, w.ffn_norm, cal_.ffn_alpha / (r * r));
    ffn_norm_->Apply(h_normed, h_slots, ffn_w, evk_map);
  }
  Announce("E  RMSNorm (owes a rescale)", h_normed,
           sched_->GetStCLevel() + 1);
  {
    ProfileScope _p("E  lower");
    Lower(h_coeff, h_normed, evk_map);
  }

  std::vector<Ct> gate, up;
  // SiLU is the one operator whose answer depends on the magnitude of its
  // argument, so 1/range is not optional bookkeeping -- it is part of the
  // function. It folds into the gate projection's plaintext, which is where
  // SiLu.h says it belongs, and the crossing constant rides along with it.
  {
    ProfileScope _p("E  project gate");
    leg.Project(gate, h_coeff, cfg_.num_channels, cfg_.hidden, w.wgate,
                cal_.size_gate / cal_.silu_range, "gate");
  }
  {
    ProfileScope _p("E  project up");
    leg.Project(up, h_coeff, cfg_.num_channels, cfg_.hidden, w.wup, cal_.size_up,
                "up");
  }

  // ---- turn F: SiLU, the elementwise gate, then the down projection -----
  //
  // The gate is grown back to exactly the interval the polynomial was fitted
  // on. The bootstrap before it saw size_gate/range * g, at most boot_max.
  std::vector<Ct> gate_slots, up_slots;
  Announce("F  gate", gate, 0);
  {
    ProfileScope _p("F  lift gate,up");
    Lift(gate_slots, gate, 1.0 / cal_.size_gate, evk_map);
    Lift(up_slots, up, 1.0, evk_map);
  }
  const auto &mult_key = evk_map.GetMultiplicationKey();
  std::vector<Ct> gated(gate_slots.size());
  {
    ProfileScope _p("F  SiLU x up");
    for (size_t i = 0; i < gate_slots.size(); i++) {
      Ct activated;
      silu_->Apply(activated, gate_slots[i], evk_map);
      // `up` was lifted alongside the gate and has spent nothing since, while
      // SiLU's polynomial has descended ceil(log2(degree+1)) levels. Cheddar's
      // HMult requires the two operands at the same level and reports the
      // mismatch as "Number of primes differ" from inside the tensor product,
      // which names neither operand -- so bring `up` down explicitly.
      const int level = boot_->param_.NPToLevel(activated.GetNP());
      Ct levelled;
      boot_->LevelDown(levelled, up_slots[i], level);
      boot_->HMult(gated[i], activated, levelled, mult_key);
    }
  }
  Announce("F  SiLU * up", gated, -1);
  std::vector<Ct> gated_coeff, ffn_out;
  {
    ProfileScope _p("F  lower");
    Lower(gated_coeff, gated, evk_map);
  }
  {
    ProfileScope _p("F  project down");
    leg.Project(ffn_out, gated_coeff, cfg_.hidden, cfg_.num_channels, w.wdown,
                r / cal_.size_up, "down");
  }

  res.resize(x.size());
  {
    ProfileScope _p("F  residual + ffn");
    for (size_t i = 0; i < x.size(); i++) {
      boot_->Add(res[i], h[i], ffn_out[i]);
    }
  }
}

template <typename word>
void LlamaBlock<word>::PlainRun(std::vector<double> &res,
                                const std::vector<double> &x,
                                const Weights &w) const {
  const int T = cfg_.num_tokens;
  const int H = cfg_.num_channels;
  const int KV = cfg_.num_kv_channels;
  const int D = cfg_.head_dim;
  const int F = cfg_.hidden;
  const int heads = H / D;
  const int kv_heads = KV / D;
  const int group = heads / kv_heads;

  auto norm = [&](const std::vector<double> &in, const std::vector<double> &g,
                  std::vector<double> &out) {
    out.assign(static_cast<size_t>(T) * H, 0.0);
    for (int t = 0; t < T; t++) {
      double sq = 0.0;
      for (int c = 0; c < H; c++) {
        const double u = in[static_cast<size_t>(t) * H + c];
        sq += u * u;
      }
      const double inv = 1.0 / std::sqrt(sq / H + cfg_.eps);
      for (int c = 0; c < H; c++) {
        out[static_cast<size_t>(t) * H + c] =
            in[static_cast<size_t>(t) * H + c] * inv * g[c];
      }
    }
  };
  auto gemm = [&](const std::vector<double> &a, const std::vector<double> &b,
                  int inner, int outer, std::vector<double> &out) {
    out.assign(static_cast<size_t>(T) * outer, 0.0);
    for (int t = 0; t < T; t++) {
      for (int i = 0; i < inner; i++) {
        const double av = a[static_cast<size_t>(t) * inner + i];
        if (av == 0.0) continue;
        const double *bp = &b[static_cast<size_t>(i) * outer];
        double *op = &out[static_cast<size_t>(t) * outer];
        for (int o = 0; o < outer; o++) op[o] += av * bp[o];
      }
    }
  };
  auto rope = [&](std::vector<double> &m, int width) {
    const int half = D / 2;
    std::vector<double> src(m);
    for (int t = 0; t < T; t++) {
      const double p = cfg_.first_position + t;
      for (int c = 0; c < width; c++) {
        const int j = c % D;
        const int base = c - j;
        const double f =
            std::pow(cfg_.rope_theta, -2.0 * (j % half) / static_cast<double>(D));
        const double cs = std::cos(p * f), sn = std::sin(p * f);
        const size_t idx = static_cast<size_t>(t) * width + c;
        const size_t partner =
            static_cast<size_t>(t) * width + base + ((j < half) ? j + half : j - half);
        m[idx] = src[idx] * cs + (j < half ? -1.0 : 1.0) * src[partner] * sn;
      }
    }
  };

  std::vector<double> normed;
  norm(x, w.attn_norm, normed);
  std::vector<double> q, k, v;
  gemm(normed, w.wq, H, H, q);
  gemm(normed, w.wk, H, KV, k);
  gemm(normed, w.wv, H, KV, v);
  rope(q, H);
  rope(k, KV);

  std::vector<double> attn(static_cast<size_t>(T) * H, 0.0);
  const double inv_root = 1.0 / std::sqrt(static_cast<double>(D));
  for (int hd = 0; hd < heads; hd++) {
    const int kvh = hd / group;
    for (int t = 0; t < T; t++) {
      std::vector<double> s(T, 0.0);
      double best = -1e300;
      for (int u = 0; u <= t; u++) {
        double dot = 0.0;
        for (int d = 0; d < D; d++) {
          dot += q[static_cast<size_t>(t) * H + hd * D + d] *
                 k[static_cast<size_t>(u) * KV + kvh * D + d];
        }
        s[u] = dot * inv_root;
        best = std::max(best, s[u]);
      }
      double sum = 0.0;
      for (int u = 0; u <= t; u++) {
        s[u] = std::exp(s[u] - best);
        sum += s[u];
      }
      for (int u = 0; u <= t; u++) {
        const double p = s[u] / sum;
        for (int d = 0; d < D; d++) {
          attn[static_cast<size_t>(t) * H + hd * D + d] +=
              p * v[static_cast<size_t>(u) * KV + kvh * D + d];
        }
      }
    }
  }

  std::vector<double> proj;
  gemm(attn, w.wo, H, H, proj);
  std::vector<double> hidden(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < hidden.size(); i++) hidden[i] = x[i] + proj[i];

  std::vector<double> fnormed;
  norm(hidden, w.ffn_norm, fnormed);
  std::vector<double> g, u;
  gemm(fnormed, w.wgate, H, F, g);
  gemm(fnormed, w.wup, H, F, u);
  std::vector<double> act(g.size());
  for (size_t i = 0; i < g.size(); i++) {
    act[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
  }
  std::vector<double> down;
  gemm(act, w.wdown, F, H, down);
  res.assign(hidden.size(), 0.0);
  for (size_t i = 0; i < res.size(); i++) res[i] = hidden[i] + down[i];
}

template class LlamaBlock<uint32_t>;
template class LlamaBlock<uint64_t>;

}  // namespace cheddar
