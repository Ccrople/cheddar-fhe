#include "extension/AttentionPacking.h"

#include <algorithm>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

AttentionPacking::AttentionPacking(int num_heads, int head_dim, int num_tokens,
                                   int degree)
    : num_heads_{num_heads},
      head_dim_{head_dim},
      num_tokens_{num_tokens},
      degree_{degree} {
  AssertTrue(num_heads > 0 && IsPowOfTwo(num_heads),
             "AttentionPacking: num_heads must be a power of two");
  AssertTrue(head_dim > 0 && IsPowOfTwo(head_dim),
             "AttentionPacking: head_dim must be a power of two");
  AssertTrue(num_tokens > 0 && IsPowOfTwo(num_tokens),
             "AttentionPacking: num_tokens must be a power of two");
  AssertTrue(degree >= 2 && IsPowOfTwo(degree),
             "AttentionPacking: degree must be a power of two");

  // One ciphertext holds `degree` coefficients, laid out as
  // [channel-within-ciphertext | head | token]. The token and head fields are
  // fixed by the shape, so what is left for the channel field is what decides
  // how many ciphertexts the tensor needs.
  const long long per_channel =
      static_cast<long long>(num_tokens) * num_heads;
  AssertTrue(per_channel <= degree,
             "AttentionPacking: num_tokens * num_heads exceeds the ring "
             "degree, so one channel does not fit in one ciphertext");
  AssertTrue(degree % per_channel == 0,
             "AttentionPacking: num_tokens * num_heads must divide the ring "
             "degree");
  channels_per_ct_ = static_cast<int>(degree / per_channel);
  AssertTrue(head_dim % channels_per_ct_ == 0,
             "AttentionPacking: the channels one ciphertext carries must "
             "divide head_dim");
  num_ciphertexts_ = head_dim / channels_per_ct_;

  coeff_axis_order_ = DeriveAxisOrder(false);
  slot_axis_order_ = DeriveAxisOrder(true);
  imaginary_flag_axis_ = DeriveImaginaryFlagAxis();
}

int AttentionPacking::TensorOffset(const TensorIndex &t) const {
  AssertTrue(t.head >= 0 && t.head < num_heads_, "TensorOffset: bad head");
  AssertTrue(t.channel >= 0 && t.channel < head_dim_,
             "TensorOffset: bad channel");
  AssertTrue(t.token >= 0 && t.token < num_tokens_, "TensorOffset: bad token");
  return (t.head * head_dim_ + t.channel) * num_tokens_ + t.token;
}

AttentionPacking::TensorIndex AttentionPacking::CoeffPosition(int ct,
                                                              int coeff) const {
  AssertTrue(ct >= 0 && ct < num_ciphertexts_,
             "CoeffPosition: ciphertext index out of range");
  AssertTrue(coeff >= 0 && coeff < degree_,
             "CoeffPosition: coefficient index out of range");
  TensorIndex t;
  t.token = coeff % num_tokens_;
  t.head = (coeff / num_tokens_) % num_heads_;
  t.channel = ct * channels_per_ct_ + coeff / (num_tokens_ * num_heads_);
  return t;
}

void AttentionPacking::CoeffIndexOf(const TensorIndex &t, int &ct,
                                    int &coeff) const {
  AssertTrue(t.head >= 0 && t.head < num_heads_, "CoeffIndexOf: bad head");
  AssertTrue(t.channel >= 0 && t.channel < head_dim_,
             "CoeffIndexOf: bad channel");
  AssertTrue(t.token >= 0 && t.token < num_tokens_, "CoeffIndexOf: bad token");
  ct = t.channel / channels_per_ct_;
  const int channel_in_ct = t.channel % channels_per_ct_;
  coeff = (channel_in_ct * num_heads_ + t.head) * num_tokens_ + t.token;
}

// The homomorphic CtS/StC run EvalSpecialFFT's butterfly stages and nothing
// else, while the host encoder's SpecialFFT/SpecialIFFT each carry one
// BitReverseVector. The difference between the two is therefore exactly one bit
// reversal, on the slot count rather than on the ring degree, with the top
// coefficient bit selecting the real or the imaginary part of the slot. The
// class comment gives the reasoning; AttentionPackingTest measures it.
AttentionPacking::SlotPosition AttentionPacking::SlotOfCoeff(int coeff,
                                                             int degree) {
  AssertTrue(degree >= 2 && IsPowOfTwo(degree),
             "SlotOfCoeff: degree must be a power of two");
  AssertTrue(coeff >= 0 && coeff < degree,
             "SlotOfCoeff: coefficient index out of range");
  const int num_slots = degree / 2;
  const int log_slots = Log2Ceil(num_slots);
  SlotPosition pos;
  pos.imaginary = coeff >= num_slots;
  pos.slot = static_cast<int>(
      BitReverseInt(pos.imaginary ? coeff - num_slots : coeff, log_slots));
  return pos;
}

int AttentionPacking::CoeffOfSlot(const SlotPosition &pos, int degree) {
  AssertTrue(degree >= 2 && IsPowOfTwo(degree),
             "CoeffOfSlot: degree must be a power of two");
  const int num_slots = degree / 2;
  AssertTrue(pos.slot >= 0 && pos.slot < num_slots,
             "CoeffOfSlot: slot index out of range");
  const int log_slots = Log2Ceil(num_slots);
  const int base = static_cast<int>(BitReverseInt(pos.slot, log_slots));
  return pos.imaginary ? base + num_slots : base;
}

AttentionPacking::TensorIndex AttentionPacking::SlotToTensor(
    int ct, const SlotPosition &pos) const {
  return CoeffPosition(ct, CoeffOfSlot(pos, degree_));
}

const char *AttentionPacking::AxisName(Axis axis) {
  switch (axis) {
    case Axis::kHead:
      return "head";
    case Axis::kChannel:
      return "channel";
    case Axis::kToken:
      return "token";
  }
  return "?";
}

// Both maps are bit-field maps, so every index bit belongs to exactly one axis
// and each axis owns a contiguous block. Rather than assert that, find it:
// toggle one bit at a time and see which component of the tensor position
// moves. The result is the ordering claim of [SYLPH] 3.2 turned into something
// that fails loudly if it stops holding.
//
// The slot domain is walked over the slot index alone. The real/imaginary flag
// is a further bit of the coefficient index and it is *not* contiguous with the
// rest of its axis -- for the [SYLPH] shape the channel axis owns slot bits 0-2
// and the flag, so folding the flag in would make the block non-contiguous and
// abort here. It is reported by DeriveImaginaryFlagAxis instead.
std::array<AttentionPacking::Axis, 3> AttentionPacking::DeriveAxisOrder(
    bool in_slot_domain) const {
  const int num_slots = degree_ / 2;
  const int num_bits = in_slot_domain ? Log2Ceil(num_slots) : Log2Ceil(degree_);

  auto position = [&](int index) {
    if (!in_slot_domain) return CoeffPosition(0, index);
    return SlotToTensor(0, SlotPosition{index, false});
  };

  const TensorIndex base = position(0);
  auto component = [](const TensorIndex &t, int axis) {
    return axis == 0 ? t.head : axis == 1 ? t.channel : t.token;
  };

  // lowest and highest index bit each axis controls, -1 if it controls none
  int lo[3] = {-1, -1, -1};
  int hi[3] = {-1, -1, -1};
  const Axis axes[3] = {Axis::kHead, Axis::kChannel, Axis::kToken};

  for (int bit = 0; bit < num_bits; bit++) {
    const TensorIndex moved = position(1 << bit);
    int owner = -1;
    for (int a = 0; a < 3; a++) {
      if (component(moved, a) == component(base, a)) continue;
      AssertTrue(owner == -1,
                 "AttentionPacking: an index bit moves two axes at once, so "
                 "the layout is not a bit-field map and the axis order is not "
                 "well defined");
      owner = a;
    }
    AssertTrue(owner != -1,
               "AttentionPacking: an index bit moves no axis, so the map is "
               "not injective");
    if (lo[owner] == -1) lo[owner] = bit;
    hi[owner] = bit;
  }

  for (int a = 0; a < 3; a++) {
    AssertTrue(lo[a] != -1,
               "AttentionPacking: an axis controls no index bit; check the "
               "shape against the ring degree");
    for (int bit = lo[a]; bit <= hi[a]; bit++) {
      const TensorIndex moved = position(1 << bit);
      AssertTrue(component(moved, a) != component(base, a),
                 "AttentionPacking: an axis does not own a contiguous block "
                 "of index bits, so calling one axis faster than another is "
                 "meaningless");
    }
  }

  int order[3] = {0, 1, 2};
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      if (lo[order[j]] < lo[order[i]]) std::swap(order[i], order[j]);
    }
  }
  return {axes[order[0]], axes[order[1]], axes[order[2]]};
}

AttentionPacking::Axis AttentionPacking::DeriveImaginaryFlagAxis() const {
  const TensorIndex real = SlotToTensor(0, SlotPosition{0, false});
  const TensorIndex imag = SlotToTensor(0, SlotPosition{0, true});
  const bool changed[3] = {imag.head != real.head,
                           imag.channel != real.channel,
                           imag.token != real.token};
  const Axis axes[3] = {Axis::kHead, Axis::kChannel, Axis::kToken};
  int owner = -1;
  for (int a = 0; a < 3; a++) {
    if (!changed[a]) continue;
    AssertTrue(owner == -1,
               "AttentionPacking: the real/imaginary flag moves two axes at "
               "once");
    owner = a;
  }
  AssertTrue(owner != -1,
             "AttentionPacking: the real/imaginary flag moves no axis");
  return axes[owner];
}

void AttentionPacking::PackCoeff(std::vector<std::vector<double>> &res,
                                 const std::vector<double> &tensor) const {
  AssertTrue(static_cast<int>(tensor.size()) == GetTensorSize(),
             "PackCoeff: tensor size does not match the layout");
  res.assign(num_ciphertexts_, std::vector<double>(degree_, 0.0));
  for (int ct = 0; ct < num_ciphertexts_; ct++) {
    for (int coeff = 0; coeff < degree_; coeff++) {
      res[ct][coeff] = tensor[TensorOffset(CoeffPosition(ct, coeff))];
    }
  }
}

void AttentionPacking::UnpackCoeff(
    std::vector<double> &res,
    const std::vector<std::vector<double>> &cts) const {
  AssertTrue(static_cast<int>(cts.size()) == num_ciphertexts_,
             "UnpackCoeff: wrong ciphertext count");
  res.assign(GetTensorSize(), 0.0);
  for (int ct = 0; ct < num_ciphertexts_; ct++) {
    AssertTrue(static_cast<int>(cts[ct].size()) == degree_,
               "UnpackCoeff: wrong coefficient count");
    for (int coeff = 0; coeff < degree_; coeff++) {
      res[TensorOffset(CoeffPosition(ct, coeff))] = cts[ct][coeff];
    }
  }
}

void AttentionPacking::PackSlot(std::vector<std::vector<Complex>> &res,
                                const std::vector<double> &tensor) const {
  AssertTrue(static_cast<int>(tensor.size()) == GetTensorSize(),
             "PackSlot: tensor size does not match the layout");
  res.assign(num_ciphertexts_, std::vector<Complex>(degree_ / 2, Complex(0)));
  for (int ct = 0; ct < num_ciphertexts_; ct++) {
    for (int coeff = 0; coeff < degree_; coeff++) {
      const double value = tensor[TensorOffset(CoeffPosition(ct, coeff))];
      const SlotPosition pos = SlotOfCoeff(coeff, degree_);
      Complex &slot = res[ct][pos.slot];
      slot = pos.imaginary ? Complex(slot.real(), value)
                           : Complex(value, slot.imag());
    }
  }
}

void AttentionPacking::UnpackSlot(
    std::vector<double> &res,
    const std::vector<std::vector<Complex>> &cts) const {
  AssertTrue(static_cast<int>(cts.size()) == num_ciphertexts_,
             "UnpackSlot: wrong ciphertext count");
  res.assign(GetTensorSize(), 0.0);
  for (int ct = 0; ct < num_ciphertexts_; ct++) {
    AssertTrue(static_cast<int>(cts[ct].size()) == degree_ / 2,
               "UnpackSlot: wrong slot count");
    for (int coeff = 0; coeff < degree_; coeff++) {
      const SlotPosition pos = SlotOfCoeff(coeff, degree_);
      const Complex &slot = cts[ct][pos.slot];
      res[TensorOffset(CoeffPosition(ct, coeff))] =
          pos.imaginary ? slot.imag() : slot.real();
    }
  }
}

}  // namespace cheddar
