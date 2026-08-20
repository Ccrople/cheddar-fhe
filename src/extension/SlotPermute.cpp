#include "extension/SlotPermute.h"

#include <algorithm>
#include <set>
#include <string>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

std::vector<int> SwapAdjacentFields(int num_slots, int low_bits,
                                    int high_bits) {
  AssertTrue(low_bits > 0 && high_bits > 0,
             "SwapAdjacentFields: both fields must be non-empty");
  const int window = 1 << (low_bits + high_bits);
  AssertTrue(num_slots % window == 0,
             "SwapAdjacentFields: the two fields must fit the slot index");
  const int low = 1 << low_bits;
  const int high = 1 << high_bits;

  std::vector<int> perm(num_slots);
  for (int s = 0; s < num_slots; s++) {
    const int rest = s / window;
    const int in = s % window;
    const int a = in / low;   // the high field
    const int b = in % low;   // the low field
    perm[s] = rest * window + b * high + a;
  }
  return perm;
}

template <typename word>
SlotPermute<word>::SlotPermute(ConstContextPtr<word> context,
                               const std::vector<int> &perm, int level)
    : num_slots_{static_cast<int>(perm.size())} {
  AssertTrue(IsPowOfTwo(num_slots_),
             "SlotPermute: the slot count must be a power of two");
  AssertTrue(num_slots_ == context->param_.degree_ / 2,
             "SlotPermute: the permutation must cover every slot");

  // A permutation is a striped matrix with one diagonal per distinct offset:
  // row `dst` takes column `src`, and StripedMatrix stores
  // `A[k][(k + d) mod w] = diag_d[k]`, so `d = (src - dst) mod n`.
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

  // THE WINDOW. The raw offsets straddle zero -- a transpose moves data both
  // ways -- and reduced mod the slot count they then span almost the whole
  // ring, which would make DetermineStride demand bs * gs >= num_slots.
  // Rotating the window to start just after the largest circular gap collapses
  // that to the true spread. Nothing else in the file matters as much.
  std::vector<int> offsets;
  offsets.reserve(num_diag_);
  for (const auto &kv : matrix) offsets.push_back(kv.first);
  std::sort(offsets.begin(), offsets.end());

  int gap = offsets.front() + num_slots_ - offsets.back();
  pre_rotation_ = offsets.front();
  for (size_t i = 1; i < offsets.size(); i++) {
    const int g = offsets[i] - offsets[i - 1];
    if (g > gap) {
      gap = g;
      pre_rotation_ = offsets[i];
    }
  }

  int max_rot = 0;
  stride_ = 0;
  for (int o : offsets) {
    int rot = (o - pre_rotation_) % num_slots_;
    if (rot < 0) rot += num_slots_;
    max_rot = Max(max_rot, rot);
    stride_ = (stride_ == 0) ? rot : GCD(stride_, rot);
  }
  if (stride_ == 0) stride_ = 1;

  // bs * gs has to cover (max_rot / stride) + 1 steps, and the hoisting engine
  // caps bs at 2^7 (Hoist.h). Split as squarely as that allows.
  const int steps = max_rot / stride_ + 1;
  int bs = 1;
  while (bs * bs < steps && bs < 128) bs <<= 1;
  bs_ = bs;
  gs_ = DivCeil(steps, bs_);
  AssertTrue(static_cast<long long>(bs_) * gs_ * stride_ > max_rot,
             "SlotPermute: the permutation is too unstructured for BSGS -- "
             "num_diag " + std::to_string(num_diag_) + ", stride " +
                 std::to_string(stride_) + ", max_rot " +
                 std::to_string(max_rot));

  lt_.emplace_back(context, matrix, level,
                   context->param_.GetRescalePrimeProd(level), bs_, gs_,
                   pre_rotation_, 0);
}

template <typename word>
void SlotPermute<word>::AddRequiredRotations(EvkRequest &req) const {
  for (const auto &t : lt_) t.AddRequiredRotations(req);
}

template <typename word>
void SlotPermute<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                                 const Ct &input,
                                 const EvkMap<word> &evk_map) const {
  lt_.front().Evaluate(context, res, input, evk_map);
}

template class SlotPermute<uint32_t>;
template class SlotPermute<uint64_t>;

}  // namespace cheddar
