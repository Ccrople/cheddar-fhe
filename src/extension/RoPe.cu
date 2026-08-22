#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/RoPe.h"

namespace cheddar {

namespace {

// The three plaintext patterns, built once per Apply since they depend on the
// starting position. cos everywhere; -sin on the lower half of each head; +sin
// on the upper half. The masks are what let one pair of rotations serve both
// halves without crossing a head boundary.
struct Tables {
  std::vector<Complex> cos_t, lower, upper;
};

Tables BuildTables(int num_slots, int num_tokens, int head_dim,
                   int first_position, double theta,
                   const std::vector<int> *channel) {
  const int half = head_dim / 2;
  Tables t;
  t.cos_t.assign(num_slots, Complex(0.0, 0.0));
  t.lower.assign(num_slots, Complex(0.0, 0.0));
  t.upper.assign(num_slots, Complex(0.0, 0.0));
  for (int s = 0; s < num_slots; s++) {
    const int p = first_position + (s % num_tokens);
    const int j =
        (channel != nullptr) ? (*channel)[s] : (s / num_tokens) % head_dim;
    // The frequencies are duplicated across the upper half, so the angle uses
    // j mod half -- this is what makes cos = cat(freqs, freqs) in Hugging
    // Face's implementation.
    const double f =
        std::pow(theta, -2.0 * (j % half) / static_cast<double>(head_dim));
    const double ang = p * f;
    t.cos_t[s] = Complex(std::cos(ang), 0.0);
    if (j < half) {
      t.lower[s] = Complex(-std::sin(ang), 0.0);
    } else {
      t.upper[s] = Complex(std::sin(ang), 0.0);
    }
  }
  return t;
}

}  // namespace

template <typename word>
RoPeHandler<word>::RoPeHandler(ConstContextPtr<word> context, int num_tokens,
                               int head_dim, int input_level, double theta)
    : context_{std::move(context)},
      num_tokens_{num_tokens},
      head_dim_{head_dim},
      input_level_{input_level},
      theta_{theta} {
  AssertTrue(num_tokens_ > 0, "RoPe: num_tokens must be positive");
  AssertTrue(head_dim_ > 1 && head_dim_ % 2 == 0,
             "RoPe: head_dim must be even");
  AssertTrue(theta_ > 1.0, "RoPe: theta must exceed one");
  num_slots_ = context_->param_.degree_ / 2;
  AssertTrue(num_slots_ % num_tokens_ == 0,
             "RoPe: num_tokens must divide the slot count");
  const int channels = num_slots_ / num_tokens_;
  AssertTrue(channels % head_dim_ == 0,
             "RoPe: head_dim must divide the channels in one ciphertext, or a "
             "head would straddle two ciphertexts and the partner channel "
             "would not be present");

  // result[s] = input[s + d], so the lower half reads its partner d = +half*T
  // slots away and the upper half reads it -half*T away, the latter expressed
  // as a positive cyclic distance.
  const int step = (head_dim_ / 2) * num_tokens_;
  up_ = step;
  down_ = num_slots_ - step;
  rotation_distances_ = {up_, down_};
}

template <typename word>
RoPeHandler<word>::RoPeHandler(ConstContextPtr<word> context, int num_tokens,
                               int head_dim, int input_level, double theta,
                               std::vector<std::vector<int>> head_channel,
                               int partner_step)
    : context_{std::move(context)},
      num_tokens_{num_tokens},
      head_dim_{head_dim},
      input_level_{input_level},
      theta_{theta},
      head_channel_{std::move(head_channel)} {
  AssertTrue(num_tokens_ > 0, "RoPe: num_tokens must be positive");
  AssertTrue(head_dim_ > 1 && head_dim_ % 2 == 0,
             "RoPe: head_dim must be even");
  AssertTrue(theta_ > 1.0, "RoPe: theta must exceed one");
  num_slots_ = context_->param_.degree_ / 2;
  AssertTrue(num_slots_ % num_tokens_ == 0,
             "RoPe: num_tokens must divide the slot count");
  AssertTrue(!head_channel_.empty(),
             "RoPe: the layout constructor needs at least one channel map");
  AssertTrue(partner_step > 0 && partner_step < num_slots_,
             "RoPe: the partner step must be a positive rotation");

  // THE STEP IS VERIFIED, NOT TRUSTED. A wrong one is not a crash: it pairs
  // each channel with some other channel, and RoPE then comes back a
  // plausible rotation of the wrong thing, which the residual carries all the
  // way to the block's output.
  const int half = head_dim_ / 2;
  for (size_t v = 0; v < head_channel_.size(); v++) {
    AssertTrue(static_cast<int>(head_channel_[v].size()) == num_slots_,
               "RoPe: every channel map must cover every slot");
    for (int s = 0; s < num_slots_; s++) {
      const int j = head_channel_[v][s];
      AssertTrue(j >= 0 && j < head_dim_,
                 "RoPe: a channel map holds an index outside the head");
      const int partner =
          (j < half) ? (s + partner_step) % num_slots_
                     : (s + num_slots_ - partner_step) % num_slots_;
      const int want = (j < half) ? j + half : j - half;
      AssertTrue(head_channel_[v][partner] == want,
                 "RoPe: the partner step does not carry channel " +
                     std::to_string(j) + " of slot " + std::to_string(s) +
                     " to channel " + std::to_string(want) +
                     ", so the layout and the step disagree");
    }
  }

  up_ = partner_step;
  down_ = num_slots_ - partner_step;
  rotation_distances_ = (up_ == down_) ? std::vector<int>{up_}
                                       : std::vector<int>{up_, down_};
}

template <typename word>
const std::vector<int> *RoPeHandler<word>::ChannelMap(int variant) const {
  if (head_channel_.empty()) return nullptr;
  AssertTrue(variant >= 0 && variant < static_cast<int>(head_channel_.size()),
             "RoPe: channel map " + std::to_string(variant) +
                 " was asked for but only " +
                 std::to_string(head_channel_.size()) + " were built");
  return &head_channel_[variant];
}

template <typename word>
void RoPeHandler<word>::PlainApply(std::vector<double> &res,
                                   const std::vector<double> &x,
                                   int first_position, int variant) const {
  AssertTrue(static_cast<int>(x.size()) == num_slots_,
             "RoPe: input must cover every slot");
  auto t = BuildTables(num_slots_, num_tokens_, head_dim_, first_position,
                       theta_, ChannelMap(variant));
  res.assign(num_slots_, 0.0);
  for (int s = 0; s < num_slots_; s++) {
    const int su = (s + up_) % num_slots_;
    const int sd = (s + down_) % num_slots_;
    res[s] = x[s] * t.cos_t[s].real() + x[su] * t.lower[s].real() +
             x[sd] * t.upper[s].real();
  }
}

template <typename word>
const typename RoPeHandler<word>::Encoded &RoPeHandler<word>::GetTables(
    int first_position, int level, int variant) const {
  if (level < 0) level = input_level_;
  // One position at a time. Every table in the map is built for the position
  // held here, so a move drops all of them together -- which is what keeps a
  // decode loop from accumulating a set per token.
  if (first_position != cached_position_) {
    tables_.clear();
    cached_position_ = first_position;
  }
  const std::pair<int, int> key(level, variant);
  auto it = tables_.find(key);
  if (it != tables_.end()) return it->second;

  const double scale = context_->param_.GetScale(level);
  auto t = BuildTables(num_slots_, num_tokens_, head_dim_, first_position,
                       theta_, ChannelMap(variant));
  Encoded built;
  context_->encoder_.Encode(built.cos_pt, level, scale, t.cos_t);
  context_->encoder_.Encode(built.lower_pt, level, scale, t.lower);
  context_->encoder_.Encode(built.upper_pt, level, scale, t.upper);
  return tables_.emplace(key, std::move(built)).first->second;
}

template <typename word>
void RoPeHandler<word>::Prepare(int first_position, int level,
                                int variant) const {
  GetTables(first_position, level, variant);
}

template <typename word>
size_t RoPeHandler<word>::GetPlaintextBytes() const {
  size_t bytes = 0;
  for (const auto &e : tables_) {
    bytes += (e.second.cos_pt.mx_.size() + e.second.lower_pt.mx_.size() +
              e.second.upper_pt.mx_.size()) *
             sizeof(word);
  }
  return bytes;
}

template <typename word>
void RoPeHandler<word>::Apply(Ct &res, const Ct &x, int first_position,
                              const EvkMap<word> &evk_map, int variant) const {
  const int level = context_->param_.NPToLevel(x.GetNP());
  const double scale = context_->param_.GetScale(level);

  // Build the plaintexts only the first time this (position, level, variant)
  // is asked for. A Plaintext is fixed by its values, its level and its scale
  // together -- the same table at a different level is a different object with
  // a different prime count, and reusing one across levels fails inside the
  // multiply with "Number of primes differ". See the cache's own comment in
  // the header for why it holds every variant of one position.
  const Encoded &tables = GetTables(first_position, level, variant);
  const Pt &cos_pt = tables.cos_pt, &lower_pt = tables.lower_pt,
           &upper_pt = tables.upper_pt;

  // Both rotations read the input, so they run at the input level and neither
  // depends on the other.
  Ct rot_up, rot_down;
  context_->HRot(rot_up, x, evk_map.GetRotationKey(up_), up_);
  context_->HRot(rot_down, x, evk_map.GetRotationKey(down_), down_);

  // Mult(ct, pt) does not rescale in Cheddar, so all three products sit at
  // scale^2 and can be added; one Rescale at the end brings the scale back.
  // That is the single multiplicative level [SYLPH] section 3.2 quotes.
  Ct acc, term;
  context_->Mult(acc, x, cos_pt);
  context_->Mult(term, rot_up, lower_pt);
  context_->Add(acc, acc, term);
  context_->Mult(term, rot_down, upper_pt);
  context_->Add(acc, acc, term);
  context_->Rescale(res, acc);
}

template class RoPeHandler<uint32_t>;
template class RoPeHandler<uint64_t>;

}  // namespace cheddar
