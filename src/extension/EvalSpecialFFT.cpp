#include "extension/EvalSpecialFFT.h"

#include <cmath>
#include <utility>
#include <vector>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
EvalSpecialFFT<word>::EvalSpecialFFT(ConstContextPtr<word> context,
                                     const BootParameter &boot_param,
                                     int num_slots, double cts_const,
                                     double stc_const)
    : num_slots_{num_slots},
      boot_param_{boot_param},
      cts_const_{cts_const},
      stc_const_{stc_const},
      full_slot_{num_slots == context->param_.degree_ / 2} {
  AssertTrue(num_slots >= 256,
             "Currently only high number of slots are supported");
  AssertTrue(IsPowOfTwo(num_slots), "Number of slots must be a power of 2");
  AssertTrue(num_slots <= context->param_.degree_ / 2,
             "Number of slots exceeds the maximum possible");
  PopulatePlainMatrices(context);
  PreparePlaintexts(context);
}

template <typename word>
std::pair<int, int> EvalSpecialFFT<word>::BSGSSplit(int num_diag) const {
  AssertTrue(IsPowOfTwo(num_diag) || IsPowOfTwo(num_diag + 1),
             "Invalid number of diagonals for EvalSpecialFFT");
  // this is somewhat heuristic
  int bs, gs;
  if (num_diag <= 4) {
    return {num_diag, 1};
  }

  switch (num_diag) {
    case 7:
    case 8:
    case 15:
      // consider using bs = 5;
    case 16:
      bs = 4;
      break;
    case 31:
    case 32:
    case 63:
      // consider using bs = 9, 11;
    case 64:
      // consider using bs = 11;
      bs = 8;
      break;
    default:  // over 127, don't care actually
      bs = 1 << DivCeil(Log2Ceil(num_diag), 2);
      break;
  }
  gs = DivCeil(num_diag, bs);

  return {bs, gs};
}

template <typename word>
void EvalSpecialFFT<word>::PopulatePlainMatrices(
    ConstContextPtr<word> context) {
  int M = context->param_.degree_ * 2;
  const auto &encoder = context->encoder_;

  int num_stages = Log2Ceil(num_slots_);
  plain_fft_stages_.resize(num_stages);
  plain_ifft_stages_.resize(num_stages);

  for (int i = 0; i < num_stages; i++) {
    int stride = 1 << i;
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M / st8;

    // Multiplication order left (0) --> right (num_stages - 1)
    auto &fft_target = plain_fft_stages_[i];
    auto &ifft_target = plain_ifft_stages_[num_stages - i - 1];

    fft_target = StripedMatrix(num_slots_, num_slots_);
    ifft_target = StripedMatrix(num_slots_, num_slots_);

    fft_target.try_emplace(0, num_slots_, Complex(0));
    fft_target.try_emplace(stride, num_slots_, Complex(0));
    if (i != num_stages - 1) {
      fft_target.try_emplace(num_slots_ - stride, num_slots_, Complex(0));
    }
    auto &fft_diag_0 = fft_target[0];
    auto &fft_diag_plus = fft_target[stride];
    auto &fft_diag_minus = fft_target[num_slots_ - stride];

    ifft_target.try_emplace(0, num_slots_, Complex(0));
    ifft_target.try_emplace(stride, num_slots_, Complex(0));
    if (i != num_stages - 1) {
      ifft_target.try_emplace(num_slots_ - stride, num_slots_, Complex(0));
    }
    auto &ifft_diag_0 = ifft_target[0];
    auto &ifft_diag_plus = ifft_target[stride];
    auto &ifft_diag_minus = ifft_target[num_slots_ - stride];

    for (int j = 0; j < stride; j++) {
      int fft_twiddle_index = (context->param_.GetGaloisFactor(j) % st8) * gap;
      int ifft_twiddle_index =
          (st8 - (context->param_.GetGaloisFactor(j) % st8)) * gap;
      Complex fft_twiddle = encoder.GetTwiddleFactor(fft_twiddle_index);
      Complex ifft_twiddle = encoder.GetTwiddleFactor(ifft_twiddle_index);

      // FFT
      // (x, y) = (x + y * twiddle, x - y * twiddle)
      fft_diag_0[j] = 1;
      fft_diag_plus[j] = fft_twiddle;
      fft_diag_minus[j + stride] = 1;
      fft_diag_0[j + stride] = -fft_twiddle;
      /* Matrix form
      fft_target[j][j] = 1;
      fft_target[j][j + stride] = fft_twiddle;
      fft_target[j + stride][j] = 1;
      fft_target[j + stride][j + stride] = -fft_twiddle;
      */

      // IFFT
      // (x, y) = (x + y, (x - y) * twiddle)
      ifft_diag_0[j] = 1;
      ifft_diag_plus[j] = 1;
      ifft_diag_minus[j + stride] = ifft_twiddle;
      ifft_diag_0[j + stride] = -ifft_twiddle;

      /* Matrix form
      ifft_target[j][j] = 1;
      ifft_target[j][j + stride] = 1;
      ifft_target[j + stride][j] = ifft_twiddle;
      ifft_target[j + stride][j + stride] = -ifft_twiddle;
      */
    }

    // For the rest, we can simply copy the values
    int num_double = Log2Ceil(num_slots_ / stride_group_size);

    for (int r = 0; r < num_double; r++) {
      std::copy(fft_diag_0.begin(),
                fft_diag_0.begin() + stride_group_size * (1 << r),
                fft_diag_0.begin() + stride_group_size * (1 << r));
      std::copy(fft_diag_plus.begin(),
                fft_diag_plus.begin() + stride_group_size * (1 << r),
                fft_diag_plus.begin() + stride_group_size * (1 << r));
      if (i != num_stages - 1) {
        std::copy(fft_diag_minus.begin(),
                  fft_diag_minus.begin() + stride_group_size * (1 << r),
                  fft_diag_minus.begin() + stride_group_size * (1 << r));
      }
      std::copy(ifft_diag_0.begin(),
                ifft_diag_0.begin() + stride_group_size * (1 << r),
                ifft_diag_0.begin() + stride_group_size * (1 << r));
      std::copy(ifft_diag_plus.begin(),
                ifft_diag_plus.begin() + stride_group_size * (1 << r),
                ifft_diag_plus.begin() + stride_group_size * (1 << r));
      if (i != num_stages - 1) {
        std::copy(ifft_diag_minus.begin(),
                  ifft_diag_minus.begin() + stride_group_size * (1 << r),
                  ifft_diag_minus.begin() + stride_group_size * (1 << r));
      }
    }
  }
}

template <typename word>
void EvalSpecialFFT<word>::PreparePlaintexts(ConstContextPtr<word> context) {
  int num_cts_phases = boot_param_.num_cts_levels_;
  int num_stc_phases = boot_param_.num_stc_levels_;
  int log_num_slots = Log2Ceil(num_slots_);

  int cts_level = boot_param_.GetCtSStartLevel();
  int stc_level = boot_param_.GetStCStartLevel();
  AssertTrue(num_cts_phases >= 2, "Use at least 2 levels for CtS");
  AssertTrue(num_stc_phases >= 2, "Use at least 2 levels for StC");

  int cts_stages_left = log_num_slots;
  int cts_stages_cumul = 0;
  double cts_const_div = std::pow(cts_const_, 1.0 / num_cts_phases);
  // std::cout << "cts_const_div: " << cts_const_div << std::endl;
  double stc_const_div = std::pow(stc_const_, 1.0 / num_stc_phases);
  // std::cout << "stc_const_div: " << stc_const_div << std::endl;

  // We will use different scaling methodology for CtS and StC
  double cts_scale = 1.0;
  for (int i = 0; i < cts_level; i++) {
    cts_scale *= context->param_.GetRescalePrimeProd(cts_level - i);
  }
  cts_scale = std::pow(cts_scale, 1.0 / num_cts_phases);

  for (int i = 0; i < num_cts_phases; i++) {
    std::cout << "CtS preparation phase " << i << std::endl;
    // CtS: high strides (num_slots / 2) --> low strides (1)
    int num_stages;
    if (i == 0) {
      num_stages = DivCeil(cts_stages_left, num_cts_phases);
    } else {
      num_stages = cts_stages_left / (num_cts_phases - i);
    }
    cts_stages_left -= num_stages;

    StripedMatrix phase_matrix = plain_ifft_stages_[cts_stages_cumul];
    for (int j = cts_stages_cumul + 1; j < cts_stages_cumul + num_stages; j++) {
      phase_matrix = StripedMatrix::Mult(plain_ifft_stages_[j], phase_matrix);
    }

    // Decomposing into Wx and -iWx part for later decomposition of real and
    // imag part for non-full-slot cases
    if (i == num_cts_phases - 1 && !full_slot_) {
      StripedMatrix extended(num_slots_ * 2, num_slots_ * 2);
      for (auto &[i, diag] : phase_matrix) {
        int dst_idx = i;
        if (i >= num_slots_ / 2) dst_idx += num_slots_;
        extended.try_emplace(dst_idx, num_slots_ * 2, Complex(0));
        for (int j = 0; j < num_slots_; j++) {
          extended.at(dst_idx)[j] = diag[j];
          extended.at(dst_idx)[j + num_slots_] = diag[j] * Complex(0, -1);
        }
      }
      phase_matrix = extended;
    }
    phase_matrix = StripedMatrix::Mult(phase_matrix, cts_const_div);

    int num_eff_diag = phase_matrix.GetNumDiag();
    if (i == num_cts_phases - 1) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    // std::cout << "CtS phase " << i << ": bs = " << bs << ", gs = " << gs
    //          << std::endl;

    // Min-KS adjustment (can be used also for hoisting)
    int pre_rotation;
    int additional_pt_rot = -(1 << cts_stages_left);
    if (i == 0) {
      pre_rotation = (1 << cts_stages_left);
    } else if (i == num_cts_phases - 1) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << cts_stages_left);
    }
    // std::cout << "Pre rotation: " << pre_rotation << std::endl;
    // std::cout << "Additional pt rot: " << additional_pt_rot << std::endl;

    cts_phases_.emplace_back(context, phase_matrix, cts_level - i,
                             context->param_.GetRescalePrimeProd(cts_level - i),
                             bs, gs, pre_rotation, additional_pt_rot);
    cts_stages_cumul += num_stages;
  }

  // 2. StC initialization
  int stc_stages_left = log_num_slots;
  int stc_stages_cumul = 0;
  for (int i = 0; i < num_stc_phases; i++) {
    std::cout << "StC preparation phase " << i << std::endl;
    // StC: low strides (1) --> high strides (num_slots / 2)
    int num_stages = stc_stages_left / (num_stc_phases - i);
    stc_stages_left -= num_stages;

    StripedMatrix phase_matrix = plain_fft_stages_[stc_stages_cumul];
    for (int j = stc_stages_cumul + 1; j < stc_stages_cumul + num_stages; j++) {
      phase_matrix = StripedMatrix::Mult(plain_fft_stages_[j], phase_matrix);
    }

    if (i == 0 && !full_slot_) {
      StripedMatrix extended(num_slots_ * 2, num_slots_ * 2);
      for (auto &[i, diag] : phase_matrix) {
        int dst_idx = i;
        if (i >= num_slots_ / 2) dst_idx += num_slots_;
        extended.try_emplace(dst_idx, num_slots_ * 2, Complex(0));
        for (int j = 0; j < num_slots_; j++) {
          extended.at(dst_idx)[j] = diag[j];
          extended.at(dst_idx)[j + num_slots_] = diag[j] * Complex(0, 1);
        }
      }
      phase_matrix = extended;
    }
    phase_matrix = StripedMatrix::Mult(phase_matrix, stc_const_div);

    int num_eff_diag = phase_matrix.GetNumDiag();
    if (i == 0) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    // std::cout << "StC phase " << i << ": bs = " << bs << ", gs = " << gs
    //          << std::endl;

    // Min-KS adjustment (can be used also for hoisting)
    int pre_rotation, additional_pt_rot;
    if (i == 0) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = (1 << num_stages);
    } else if (i == num_stc_phases - 1) {
      pre_rotation = (1 << stc_stages_cumul);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << stc_stages_cumul);
      additional_pt_rot = (1 << (num_stages + stc_stages_cumul));
    }
    // std::cout << "Pre rotation: " << pre_rotation << std::endl;
    // std::cout << "Additional pt rot: " << additional_pt_rot << std::endl;

    // double stc_scale = context->param_.GetScale(stc_level - i);
    double stc_scale = context->param_.GetRescalePrimeProd(stc_level - i);
    stc_phases_.emplace_back(context, phase_matrix, stc_level - i, stc_scale,
                             bs, gs, pre_rotation, additional_pt_rot);
    stc_stages_cumul += num_stages;
  }
}

template <typename word>
void EvalSpecialFFT<word>::AddRequiredRotations(EvkRequest &req,
                                                bool min_ks) const {
  for (const auto &cts_phase : cts_phases_) {
    cts_phase.AddRequiredRotations(req, min_ks);
  }
  for (const auto &stc_phase : stc_phases_) {
    stc_phase.AddRequiredRotations(req, min_ks);
  }
  if (!full_slot_) {
    req.AddRequest(num_slots_, boot_param_.GetEndLevel());
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateCtS(ConstContextPtr<word> context, Ct &res,
                                       const Ct &input,
                                       const EvkMap<word> &evk_map,
                                       bool min_ks) const {
  int num_cts_phases = cts_phases_.size();
  cts_phases_.at(0).Evaluate(context, res, input, evk_map, min_ks);
  for (int i = 1; i < num_cts_phases; i++) {
    cts_phases_.at(i).Evaluate(context, res, res, evk_map, min_ks);
  }
  if (!full_slot_) {
    res.SetNumSlots(num_slots_ * 2);
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateStC(ConstContextPtr<word> context, Ct &res,
                                       const Ct &input,
                                       const EvkMap<word> &evk_map,
                                       bool min_ks) const {
  int num_stc_phases = stc_phases_.size();
  stc_phases_.at(0).Evaluate(context, res, input, evk_map, min_ks);
  for (int i = 1; i < num_stc_phases; i++) {
    stc_phases_.at(i).Evaluate(context, res, res, evk_map, min_ks);
  }

  Ct tmp;
  if (!full_slot_) {
    res.SetNumSlots(num_slots_ * 2);
    // res += HRot(res, num_slots_)
    context->HRotAdd(res, res, res, evk_map.GetRotationKey(num_slots_),
                     num_slots_);
  }
  res.SetNumSlots(num_slots_);
}

template <typename word>
void EvalSpecialFFT<word>::PrepareSinC(ConstContextPtr<word> context,
                                       int sub_degree, int stc_level,
                                       int cts_level, int num_phases) {
  const int degree = context->param_.degree_;
  AssertTrue(full_slot_,
             "PrepareSinC: the SinC conversions are defined on the full slot "
             "count; the sparse-packing path is a different transform");
  AssertTrue(IsPowOfTwo(sub_degree) && sub_degree >= 2 && sub_degree < degree,
             "PrepareSinC: sub_degree must be a power of two in [2, degree); "
             "SinC(degree) is the ordinary slot encoding and needs no "
             "conversion");
  AssertTrue(degree % sub_degree == 0, "PrepareSinC: sub_degree must divide "
                                       "the ring degree");

  const int num_stages = Log2Ceil(num_slots_);
  const int d = degree / sub_degree;
  const int p = Log2Ceil(d);
  AssertTrue(p <= num_stages,
             "PrepareSinC: sub_degree is smaller than the transform allows");

  AssertTrue(p >= 1, "PrepareSinC: nothing to do");
  AssertTrue(num_phases >= 1 && num_phases <= p,
             "PrepareSinC: the phase count must be between one and the stage "
             "count");
  AssertTrue(stc_level - num_phases >= 0 && cts_level - num_phases >= 0,
             "PrepareSinC: the transform spends one level per phase and there "
             "are not that many below the level it starts at");
  sinc_sub_degree_ = sub_degree;
  sinc_stc_.clear();
  sinc_cts_.clear();

  // HOW THE p STAGES ARE SPLIT, AND WHY THAT IS THE WHOLE COST QUESTION.
  //
  // A product of `q` butterfly stages has 2^q diagonals, and a diagonal is a
  // full plaintext at the transform's own limb count. So one phase carrying
  // all p stages is 2^p plaintexts -- 2048 for the sub_degree = 32 the
  // attention product wants, which is gigabytes -- while three phases of
  // 4 + 4 + 3 are 16 + 16 + 8 = 40. That is the same trade `PreparePlaintexts`
  // makes for StC itself (`num_stc_levels_`, three on every logN=16 preset),
  // and it is level-neutral in the pipeline: a tensor bound for the product
  // pays SlotToSinC *instead of* SlotToCoeff, not on top of it.
  //
  // The stages are split as evenly as p allows, largest group first, matching
  // how StC's own phases are apportioned.
  auto split = [&](int total, int phases) {
    std::vector<int> counts;
    int left = total;
    for (int i = 0; i < phases; i++) {
      const int take = (i == 0) ? DivCeil(left, phases) : left / (phases - i);
      counts.push_back(take);
      left -= take;
    }
    return counts;
  };
  const std::vector<int> counts = split(p, num_phases);

  // THE SHIFT BETWEEN PHASES, AND WHY A SPLIT TRANSFORM CANNOT DO WITHOUT ONE.
  //
  // A group of `q` consecutive butterfly stages starting at stride 2^c has its
  // offsets spread over multiples of 2^c in +-(2^(c+q) - 2^c) -- so they
  // STRADDLE ZERO, and reduced mod the slot count the negative ones land near
  // the top. `LinearTransform::DetermineStride` then sees a spread of nearly
  // the whole ring and demands `bs * gs >= num_slots / 2^c`, which for the
  // first phase of the sub_degree = 32 suffix is 2048 against 31 diagonals. It
  // does not merely cost keys; it refuses to build.
  //
  // A transform whose stages are ALL of them does not have this problem -- the
  // offsets wrap around and cover every residue at their common stride, which
  // is why the single-phase form works and why nothing needed this until now.
  //
  // The fix is the one `PreparePlaintexts` already uses for StC's own phases,
  // and it is free: `LinearTransform` computes
  // `rot(M . rot(x, -(a + p)), a)` for `p = pre_rotation` and
  // `a = additional_pt_rot`, so a chain of phases is exact iff
  //
  //     p_0 + a_0 = 0        the first phase takes an unrotated input
  //     p_{i+1} = a_i - a_{i+1}   each phase undoes the last one's rotation
  //     a_{last} = 0         and the last one leaves it unrotated
  //
  // Choosing `a_i = 2^(cumul_{i+1})` -- the stride the NEXT phase starts at --
  // satisfies all three and puts every phase's reduced offsets in
  // `[0, 2 (2^q - 1) 2^c]`, i.e. `bs * gs >= 2^(q+1) - 1`, which is exactly the
  // diagonal count. It is the same rule StC uses; the only difference here is
  // that the suffix starts at stride `2^(num_stages - p)` rather than at 1, and
  // for `num_phases == 1` every shift is zero and this reduces to the plain
  // transform the single-phase form was.
  int cursor = num_stages - p;
  int prev_a = 0;
  for (int phase = 0; phase < num_phases; phase++) {
    StripedMatrix forward = plain_fft_stages_[cursor];
    for (int j = cursor + 1; j < cursor + counts[phase]; j++) {
      forward = StripedMatrix::Mult(plain_fft_stages_[j], forward);
    }
    cursor += counts[phase];
    const bool last = (phase == num_phases - 1);
    const int a = last ? 0 : (1 << cursor);
    const int pre_rotation = (phase == 0) ? -a : (prev_a - a);
    prev_a = a;

    const int level = stc_level - phase;
    auto [fbs, fgs] = BSGSSplit(forward.GetNumDiag());
    sinc_stc_.emplace_back(context, forward, level,
                           context->param_.GetRescalePrimeProd(level), fbs,
                           fgs, pre_rotation, a);
  }

  // SinC -> slots: the PREFIX of CtS, which is the same set of strides in the
  // opposite order, times 1/d. `plain_ifft_stages_[num_stages-1-i]` holds
  // stride 2^i, so index 0 is the highest stride -- the one the forward
  // applied last, and therefore the one the inverse applies first. The 1/d
  // rides the first phase; each stage pair composes to 2I, so p of them
  // compose to 2^p = d.
  // The same chain, mirrored. `plain_ifft_stages_[j]` holds stride
  // 2^(num_stages-1-j), so a phase ending at index `cursor` has its LOWEST
  // stride at 2^(num_stages - cursor); that exponent is what `a` is built from,
  // negated because CtS descends where StC ascends.
  cursor = 0;
  prev_a = 0;
  for (int phase = 0; phase < num_phases; phase++) {
    StripedMatrix inverse = plain_ifft_stages_[cursor];
    for (int j = cursor + 1; j < cursor + counts[phase]; j++) {
      inverse = StripedMatrix::Mult(plain_ifft_stages_[j], inverse);
    }
    cursor += counts[phase];
    if (phase == 0) {
      inverse = StripedMatrix::Mult(inverse, 1.0 / static_cast<double>(d));
    }
    const bool last = (phase == num_phases - 1);
    const int a = last ? 0 : -(1 << (num_stages - cursor));
    const int pre_rotation = (phase == 0) ? -a : (prev_a - a);
    prev_a = a;

    const int level = cts_level - phase;
    auto [ibs, igs] = BSGSSplit(inverse.GetNumDiag());
    sinc_cts_.emplace_back(context, inverse, level,
                           context->param_.GetRescalePrimeProd(level), ibs,
                           igs, pre_rotation, a);
  }
}

template <typename word>
void EvalSpecialFFT<word>::AddRequiredSinCRotations(EvkRequest &req) const {
  for (const auto &lt : sinc_stc_) lt.AddRequiredRotations(req);
  for (const auto &lt : sinc_cts_) lt.AddRequiredRotations(req);
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateSlotToSinC(
    ConstContextPtr<word> context, Ct &res, const Ct &input,
    const EvkMap<word> &evk_map) const {
  AssertTrue(!sinc_stc_.empty(), "EvaluateSlotToSinC: call PrepareSinC first");
  sinc_stc_.front().Evaluate(context, res, input, evk_map);
  for (size_t i = 1; i < sinc_stc_.size(); i++) {
    Ct next;
    sinc_stc_[i].Evaluate(context, next, res, evk_map);
    res = std::move(next);
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateSinCToSlot(
    ConstContextPtr<word> context, Ct &res, const Ct &input,
    const EvkMap<word> &evk_map) const {
  AssertTrue(!sinc_cts_.empty(), "EvaluateSinCToSlot: call PrepareSinC first");
  sinc_cts_.front().Evaluate(context, res, input, evk_map);
  for (size_t i = 1; i < sinc_cts_.size(); i++) {
    Ct next;
    sinc_cts_[i].Evaluate(context, next, res, evk_map);
    res = std::move(next);
  }
}

template class EvalSpecialFFT<uint32_t>;
template class EvalSpecialFFT<uint64_t>;

}  // namespace cheddar
