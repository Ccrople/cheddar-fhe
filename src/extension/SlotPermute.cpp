#include "extension/SlotPermute.h"

#include <algorithm>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

std::vector<int> SwapAdjacentFields(int num_slots, int low_bits, int high_bits,
                                    int offset /*= 0*/) {
  AssertTrue(low_bits > 0 && high_bits > 0 && offset >= 0,
             "SwapAdjacentFields: both fields must be non-empty");
  const int window = 1 << (low_bits + high_bits);
  const int below = 1 << offset;
  AssertTrue(num_slots % (window * below) == 0,
             "SwapAdjacentFields: the two fields must fit the slot index");
  const int low = 1 << low_bits;
  const int high = 1 << high_bits;

  std::vector<int> perm(num_slots);
  for (int s = 0; s < num_slots; s++) {
    const int keep = s % below;             // the bits below the pair
    const int up = s / below;
    const int rest = up / window;           // the bits above the pair
    const int in = up % window;
    const int a = in / low;                 // the high field
    const int b = in % low;                 // the low field
    perm[s] = (rest * window + b * high + a) * below + keep;
  }
  return perm;
}

template <typename word>
SlotPermute<word>::SlotPermute(ConstContextPtr<word> context,
                              const std::vector<int> &perm, int level)
    : num_slots_{static_cast<int>(perm.size())}, level_{level} {
  AssertTrue(IsPowOfTwo(num_slots_),
             "SlotPermute: the slot count must be a power of two");
  AssertTrue(num_slots_ == context->param_.degree_ / 2,
             "SlotPermute: the permutation must cover every slot");
  AssertTrue(level >= 1, "SlotPermute: the transform spends a level");

  // A permutation is a striped matrix with one diagonal per distinct offset.
  // StripedMatrix stores `A[k][(k + d) mod w] = diag_d[k]` and we want
  // `out[dst] = in[src]`, so `d = (src - dst) mod n` and `diag_d[dst] = 1`.
  StripedMatrix matrix(num_slots_, num_slots_);
  std::vector<bool> hit(num_slots_, false);
  for (int s = 0; s < num_slots_; s++) {
    const int dst = perm[s];
    AssertTrue(dst >= 0 && dst < num_slots_ && !hit[dst],
               "SlotPermute: the map is not a bijection");
    hit[dst] = true;
    int d = (s - dst) % num_slots_;
    if (d < 0) d += num_slots_;
    matrix.try_emplace(d, num_slots_, Complex(0));
    matrix.at(d)[dst] = Complex(1.0, 0.0);
  }
  num_diag_ = matrix.GetNumDiag();
  AssertTrue(num_diag_ > 1,
             "SlotPermute: a single-diagonal map is a rotation, not a "
             "LinearTransform");

  // THE WINDOW, AND WHY IT IS NOT `LinearTransform::pre_rotation`.
  //
  // That parameter does not shift a window. `DetermineStride` reduces every
  // offset by it and the hoisting engine then rotates by the *reduced* amount,
  // so the input is assumed to arrive already rotated by it. Inside
  // EvalSpecialFFT that is free -- the previous phase produced exactly that
  // rotation, which is what the "min-KS adjustment" comment there is about --
  // but a standalone transform has no previous phase, and a nonzero
  // pre_rotation returns garbage rather than a shifted answer. Measured.
  //
  // So the shift is done honestly: the matrix is built for the permutation
  // composed with a rotation, and that rotation is undone afterwards with one
  // HRot. It is worth doing only when it buys something, so both options are
  // costed and the cheaper one wins -- a shift can *destroy* a common stride,
  // and a square transpose's offsets are all multiples of n-1 while the gap
  // start need not be.
  std::vector<int> offsets;
  offsets.reserve(num_diag_);
  for (const auto &kv : matrix) offsets.push_back(kv.first);
  std::sort(offsets.begin(), offsets.end());

  auto cost = [&](int shift) {
    int max_rot = 0, g = 0;
    for (int o : offsets) {
      int rot = (o - shift) % num_slots_;
      if (rot < 0) rot += num_slots_;
      max_rot = Max(max_rot, rot);
      g = (g == 0) ? rot : GCD(g, rot);
    }
    if (g == 0) g = 1;
    return std::pair<int, int>{max_rot / g + 1, g};
  };

  int gap_start = offsets.front();
  int gap = offsets.front() + num_slots_ - offsets.back();
  for (size_t i = 1; i < offsets.size(); i++) {
    const int g = offsets[i] - offsets[i - 1];
    if (g > gap) {
      gap = g;
      gap_start = offsets[i];
    }
  }

  const auto plain = cost(0);
  const auto shifted = cost(gap_start);
  const bool use_shift = shifted.first < plain.first;
  shift_ = use_shift ? gap_start : 0;
  const auto chosen = use_shift ? shifted : plain;
  stride_ = chosen.second;

  // `out2[(dst + shift) mod n] = in[src]`, so diagonal `d - shift` carries row
  // `dst + shift` of what diagonal `d` carried at row `dst`.
  StripedMatrix built(num_slots_, num_slots_);
  if (shift_ == 0) {
    built = matrix;
  } else {
    for (const auto &kv : matrix) {
      int d = (kv.first - shift_) % num_slots_;
      if (d < 0) d += num_slots_;
      built.try_emplace(d, num_slots_, Complex(0));
      auto &out_diag = built.at(d);
      const auto &in_diag = kv.second;
      for (int k = 0; k < num_slots_; k++) {
        out_diag[(k + shift_) % num_slots_] = in_diag[k];
      }
    }
  }

  // bs * gs has to cover the BSGS steps, and the hoisting engine caps bs at
  // 2^7 (Hoist.h). Split as squarely as that allows.
  int bs = 1;
  while (bs * bs < chosen.first && bs < 128) bs <<= 1;
  bs_ = bs;
  gs_ = DivCeil(chosen.first, bs_);

  lt_.emplace_back(context, built, level,
                   context->param_.GetRescalePrimeProd(level), bs_, gs_, 0, 0);
}

template <typename word>
void SlotPermute<word>::AddRequiredRotations(EvkRequest &req) const {
  for (const auto &t : lt_) t.AddRequiredRotations(req);
  // The compensating rotation runs on the transform's output, one level down.
  if (shift_ != 0) req.AddRequest(shift_, level_ - 1);
}

template <typename word>
void SlotPermute<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                                 const Ct &input,
                                 const EvkMap<word> &evk_map) const {
  if (shift_ == 0) {
    lt_.front().Evaluate(context, res, input, evk_map);
    return;
  }
  Ct shifted;
  lt_.front().Evaluate(context, shifted, input, evk_map);
  context->HRot(res, shifted, evk_map.GetRotationKey(shift_), shift_);
}

template class SlotPermute<uint32_t>;
template class SlotPermute<uint64_t>;

}  // namespace cheddar
