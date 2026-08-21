#include "extension/LlamaAttention.h"

#include "extension/Profile.h"

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

}  // namespace

template <typename word>
SinCLinearLeg<word>::SinCLinearLeg(
    std::shared_ptr<const BootContext<word>> boot,
    ConstContextPtr<word> switch_ctx, ConstContextPtr<word> small_ctx,
    const Config &cfg, const typename SinCAttention<word>::Config &attn_cfg,
    const typename SinCAttention<word>::Keys &keys,
    const typename CoeffLinearLeg<word>::Config &linear_cfg,
    std::vector<const EvaluationKey<word> *> modpack_keys)
    : CoeffLinearLeg<word>(boot, linear_cfg, std::move(modpack_keys)),
      cfg_{cfg},
      attn_{boot, std::move(switch_ctx), std::move(small_ctx), attn_cfg},
      keys_{keys} {
  const auto &layout = attn_.GetLayout();
  num_slots_ = boot->param_.degree_ / 2;
  num_cts_ = layout.num_cts;
  lanes_ = layout.lanes;
  head_dim_ = attn_cfg.head_dim;
  num_tokens_ = attn_cfg.num_tokens;
  channels_per_ct_ = num_slots_ / num_tokens_;

  AssertTrue(cfg_.num_heads > 0 && cfg_.num_heads % lanes_ == 0,
             "SinCLinearLeg: one call of the batch CC-MM does " +
                 std::to_string(lanes_) +
                 " heads, so the head count must be a whole number of calls");
  AssertTrue(cfg_.num_kv_heads > 0 &&
                 cfg_.num_heads / cfg_.num_kv_heads == attn_cfg.gqa_group,
             "SinCLinearLeg: the GQA group in the attention config does not "
             "match the head counts here");
  AssertTrue(lanes_ % attn_cfg.gqa_group == 0,
             "SinCLinearLeg: the lanes must divide into whole KV groups");
  const int kv_per_group = lanes_ / attn_cfg.gqa_group;
  num_src_cts_ = kv_per_group * head_dim_ / channels_per_ct_;
  AssertTrue(num_src_cts_ >= 1 &&
                 num_src_cts_ * channels_per_ct_ ==
                     kv_per_group * head_dim_,
             "SinCLinearLeg: one group's KV heads do not fill a whole number "
             "of ciphertexts");
  // The three field widths have to tile the slot index exactly, or the
  // channel orders below run off the end of a ciphertext.
  AssertTrue((head_dim_ / num_cts_) * lanes_ == channels_per_ct_,
             "SinCLinearLeg: (head_dim / num_cts) * lanes must be the channels "
             "one ciphertext holds");
  AssertTrue(IsPowOfTwo(num_cts_) && IsPowOfTwo(num_src_cts_),
             "SinCLinearLeg: both ciphertext counts must be powers of two");

  score_magnitude_ = attn_cfg.score_magnitude;
  value_magnitude_ = attn_cfg.value_magnitude;
}

template <typename word>
void SinCLinearLeg<word>::SetChainConstant(double c) {
  AssertTrue(c != 0.0, "SinCLinearLeg: the chain constant cannot be zero");
  cfg_.chain_constant = c;
  attn_.SetMagnitudes(score_magnitude_ / c, value_magnitude_ / c);
}

template <typename word>
void SinCLinearLeg<word>::Retune(double score, double value) const {
  if (score == score_magnitude_ && value == value_magnitude_) return;
  score_magnitude_ = score;
  value_magnitude_ = value;
  // Only the two prefixes are rebuilt -- 31 diagonals each -- and NO rotation
  // key changes, because a constant does not change a transform's diagonal
  // structure. So this is cheap enough to do lazily and still be a no-op on
  // every block after the first.
  attn_.SetMagnitudes(score / cfg_.chain_constant, value / cfg_.chain_constant);
}

// ---------------------------------------------------------------- the layouts
//
// Every one of these is a bijection of the channel index, and every one is
// free: the block permutes the projection's weight columns and the ciphertext
// comes out packed. What they encode is where the swap chain and the exchange
// need each bit to start from, which AttentionTransportTest derived and
// checked index by index over all 262144 positions.

template <typename word>
void SinCLinearLeg<word>::ChannelOrder(std::vector<int> &res,
                                       Tensor which) const {
  const int D = head_dim_;
  const int log_cts = Log2Ceil(num_cts_);
  const int per_group = channels_per_ct_ * num_cts_;  // lanes_ * D
  res.clear();

  if (which == Tensor::kQuery || which == Tensor::kAttnOut) {
    // ct = BitRev(c mod num_cts), in-ct channel = (c / num_cts) * lanes + head.
    // Three channel bits go to the ciphertext axis so the head can have the
    // four slot bits above the token, which is the CC-MM's lane index; the
    // bit reversal is the layout's own naming of `column / rank`.
    //
    // The attention output comes back in Q's packing, which is why one order
    // serves both: it is what makes the block close on itself.
    res.assign(static_cast<size_t>(cfg_.num_heads) * D, 0);
    for (int head = 0; head < cfg_.num_heads; head++) {
      const int g = head / lanes_;
      const int h = head % lanes_;
      for (int c = 0; c < D; c++) {
        const int ct = BitRev(c % num_cts_, log_cts);
        const int within = (c / num_cts_) * lanes_ + h;
        res[static_cast<size_t>(head) * D + c] =
            g * per_group + ct * channels_per_ct_ + within;
      }
    }
    return;
  }

  // K and V. Both spend nothing on the ciphertext axis for the channel,
  // because the exchange takes it -- for the KEY in K's case, and for GQA's
  // fourfold replication in V's. What differs is where the channel's low bits
  // sit, and that is set by which second swap the operand goes through.
  const int kv_per_group = lanes_ / (cfg_.num_heads / cfg_.num_kv_heads);
  const int kv_groups = cfg_.num_kv_heads / kv_per_group;
  const int stride = kv_per_group / num_src_cts_;
  res.assign(static_cast<size_t>(cfg_.num_kv_heads) * D, 0);
  for (int kv = 0; kv < cfg_.num_kv_heads; kv++) {
    const int g = kv / kv_per_group;
    const int j = kv % kv_per_group;
    const int ct = j % num_src_cts_;
    const int hi = j / num_src_cts_;
    for (int c = 0; c < D; c++) {
      int within = 0;
      if (which == Tensor::kKey) {
        // [ c | hi ]: the key ends up on the ciphertext axis after the
        // `[4|4]@7` swap, so the channel keeps the top of the slot index.
        within = c * stride + hi;
      } else {
        // [ c/num_cts | hi | c mod num_cts ]: V follows Q's `[4|7]` swap, so
        // its low channel bits have to be where the exchange will find them.
        within = (c / num_cts_) * lanes_ + hi * num_cts_ + (c % num_cts_);
      }
      res[static_cast<size_t>(kv) * D + c] =
          g * num_src_cts_ * channels_per_ct_ + ct * channels_per_ct_ + within;
    }
  }
  AssertTrue(kv_groups * kv_per_group == cfg_.num_kv_heads,
             "SinCLinearLeg: the KV heads do not divide into whole groups");
}

template <typename word>
void SinCLinearLeg<word>::LocateLane(int head, int token, int index, int &ct,
                                     int &slot) const {
  const int g = head / lanes_;
  const int h = head % lanes_;
  const int log_cts = Log2Ceil(num_cts_);
  ct = g * num_cts_ + BitRev(index % num_cts_, log_cts);
  slot = (index / num_cts_) * (num_tokens_ * lanes_) + token * lanes_ + h;
}

template <typename word>
void SinCLinearLeg<word>::LocateScore(int head, int query, int key, int &ct,
                                      int &slot) const {
  LocateLane(head, query, key, ct, slot);
}

// ------------------------------------------------------------------ the shift
//
// The block states the shift per (head, query, key), in the raw product's own
// units. `SinCAttention` adds it at level 0 in SinC form, which is indexed by
// the message index and not by the slot -- so each entry has to be carried
// from the slot it will occupy to the SinC index it occupies now. The layout
// owns both halves of that: `Position` inverts the slot map, `LocateSinC`
// gives the index. Neither is trusted here; both are the layout's own.

template <typename word>
void SinCLinearLeg<word>::BuildShift(std::vector<std::vector<Complex>> &res,
                                     const std::vector<double> &shift,
                                     int group) const {
  const auto &layout = attn_.GetLayout();
  const int T = num_tokens_;
  res.assign(num_cts_, std::vector<Complex>(num_slots_, Complex(0.0, 0.0)));
  for (int h = 0; h < lanes_; h++) {
    const int head = group * lanes_ + h;
    for (int query = 0; query < T; query++) {
      for (int key = 0; key < T; key++) {
        int ct = 0, slot = 0;
        LocateLane(h, query, key, ct, slot);  // group-local
        int row = 0, column = 0, lane = 0;
        AssertTrue(layout.Position(ct, slot, row, column, lane),
                   "SinCLinearLeg: the score layout names a slot the product "
                   "does not use");
        int sinc_ct = 0, index = 0;
        layout.LocateSinC(row, column, lane, sinc_ct, index);
        AssertTrue(sinc_ct == ct,
                   "SinCLinearLeg: the SinC index of an entry is in a "
                   "different ciphertext from its slot, which the layout's "
                   "own identity forbids");
        res[ct][index] = Complex(
            shift[(static_cast<size_t>(head) * T + query) * T + key], 0.0);
      }
    }
  }
}

// ------------------------------------------------------------- the two products

template <typename word>
void SinCLinearLeg<word>::Scores(std::vector<Ct> &res,
                                 const std::vector<Ct> &q,
                                 const std::vector<Ct> &k, double magnitude,
                                 const std::vector<double> &shift) const {
  const int groups = cfg_.num_heads / lanes_;
  AssertTrue(static_cast<int>(q.size()) == groups * num_cts_,
             "SinCLinearLeg::Scores: Q is " + std::to_string(q.size()) +
                 " ciphertexts against the " +
                 std::to_string(groups * num_cts_) + " the packing holds");
  AssertTrue(static_cast<int>(k.size()) == groups * num_src_cts_,
             "SinCLinearLeg::Scores: K is " + std::to_string(k.size()) +
                 " ciphertexts against the " +
                 std::to_string(groups * num_src_cts_) + " expected");
  AssertTrue(shift.empty() ||
                 shift.size() == static_cast<size_t>(cfg_.num_heads) *
                                     num_tokens_ * num_tokens_,
             "SinCLinearLeg::Scores: the shift must be one entry per score");

  {
    NvtxScope _n("leg: Retune");
    Retune(magnitude, value_magnitude_);
  }
  res.clear();
  res.resize(static_cast<size_t>(groups) * num_cts_);
  for (int g = 0; g < groups; g++) {
    std::vector<const Ct *> qg, kg;
    for (int i = 0; i < num_cts_; i++) qg.push_back(&q[g * num_cts_ + i]);
    for (int i = 0; i < num_src_cts_; i++) {
      kg.push_back(&k[g * num_src_cts_ + i]);
    }
    std::vector<std::vector<Complex>> sh;
    if (!shift.empty()) {
      NvtxScope _n("leg: BuildShift");
      BuildShift(sh, shift, g);
    }
    std::vector<Ct> out;
    attn_.Scores(out, qg, kg, keys_, sh);
    for (int i = 0; i < num_cts_; i++) {
      res[static_cast<size_t>(g) * num_cts_ + i] = std::move(out[i]);
    }
  }
}

template <typename word>
void SinCLinearLeg<word>::Values(std::vector<Ct> &res,
                                 const std::vector<Ct> &p,
                                 const std::vector<Ct> &v,
                                 double magnitude) const {
  const int groups = cfg_.num_heads / lanes_;
  AssertTrue(static_cast<int>(p.size()) == groups * num_cts_,
             "SinCLinearLeg::Values: P is " + std::to_string(p.size()) +
                 " ciphertexts against the " +
                 std::to_string(groups * num_cts_) + " expected");
  AssertTrue(static_cast<int>(v.size()) == groups * num_src_cts_,
             "SinCLinearLeg::Values: V is " + std::to_string(v.size()) +
                 " ciphertexts against the " +
                 std::to_string(groups * num_src_cts_) + " expected");

  {
    NvtxScope _n("leg: Retune");
    Retune(score_magnitude_, magnitude);
  }
  res.clear();
  res.resize(static_cast<size_t>(groups) * num_cts_);
  for (int g = 0; g < groups; g++) {
    std::vector<const Ct *> pg, vg;
    for (int i = 0; i < num_cts_; i++) pg.push_back(&p[g * num_cts_ + i]);
    for (int i = 0; i < num_src_cts_; i++) {
      vg.push_back(&v[g * num_src_cts_ + i]);
    }
    std::vector<Ct> out;
    attn_.Values(out, pg, vg, keys_);
    for (int i = 0; i < num_cts_; i++) {
      res[static_cast<size_t>(g) * num_cts_ + i] = std::move(out[i]);
    }
  }
}

template class SinCLinearLeg<uint32_t>;
template class SinCLinearLeg<uint64_t>;

}  // namespace cheddar
