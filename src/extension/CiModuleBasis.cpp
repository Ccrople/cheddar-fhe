#include "extension/CiModuleBasis.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>

#include "common/Assert.h"
#include "extension/EvalSpecialFFT.h"

namespace cheddar {

namespace {

// Entry (row, col) of a StripedMatrix is `m[(col - row) mod n][row]`; both
// scalings below walk every stored entry once.
void ScaleColumns(StripedMatrix &m, const std::vector<double> &scale) {
  const int n = static_cast<int>(scale.size());
  for (auto &[idx, diag] : m) {
    for (int row = 0; row < n; row++) {
      const int col = ((row + idx) % n + n) % n;
      diag[row] *= scale[col];
    }
  }
}

void ScaleRows(StripedMatrix &m, const std::vector<double> &scale) {
  const int n = static_cast<int>(scale.size());
  for (auto &[idx, diag] : m) {
    for (int row = 0; row < n; row++) diag[row] *= scale[row];
  }
}

// THE WINDOW. `LinearTransform` maps every offset to `(i - pre_rotation) mod
// n` and wants those to fit in `(bs * gs - 1) * gcd`; a phase whose offsets
// straddle zero therefore needs a pre-rotation, and the SinC prefix's chain
// rule (EvalSpecialFFT.cpp, PrepareSinCPrefix) is how consecutive phases
// pass it along without a rotation between them:
//
//     p_0 = -a_0,   p_{i+1} = a_i - a_{i+1},   result = rot(M x, a_last).
//
// Here `a` is the tightest window: minus the most negative signed offset.
// A phase covering the WHOLE stride lattice -- every twist phase that
// contains the top stride, and the correction-folded one -- fits any
// pre-rotation, so its window is free, and a chain that ends on one ends
// clean. A chain that ends elsewhere is finished by one HRot.
int Window(const StripedMatrix &m, int n) {
  int gcd = 0;
  int most_negative = 0;
  for (const auto &[idx, _] : m) {
    const int off = ((idx % n) + n) % n;
    gcd = GCD(gcd, off);
    const int signed_off = (off <= n / 2) ? off : off - n;
    most_negative = std::min(most_negative, signed_off);
  }
  if (gcd > 0 && static_cast<int>(m.size()) * gcd == n) return -1;  // full
  return -most_negative;
}

// Real in, complex through the middle, real out -- one group of phases.
template <typename word>
void RunGroup(ConstContextPtr<word> context,
              const std::vector<ComplexLinearTransform<word>> &group,
              Ciphertext<word> &res, const Ciphertext<word> &input,
              const EvkMap<word> &evk_map) {
  Ciphertext<word> re, im;
  group.front().EvaluateFromReal(context, re, im, input, evk_map);
  for (size_t j = 1; j + 1 < group.size(); j++) {
    group[j].EvaluatePair(context, re, im, re, im, evk_map);
  }
  group.back().EvaluateToReal(context, res, re, im, evk_map);
}

}  // namespace

template <typename word>
std::pair<int, int> CiModuleBasis<word>::Split(int num_diag) {
  // The converter's split, under the fused complex giant step's two limits:
  // at most 32 baby steps (GSFusedComplexPAccum's instantiation cap) and
  // never gs == 1 (the swapped layout it cannot evaluate).
  int bs = 1 << DivCeil(Log2Ceil(num_diag), 2);
  if (bs > 32) bs = 32;
  if (bs > num_diag) bs = 1 << Log2Ceil(num_diag);
  if (bs < 1) bs = 1;
  int gs = DivCeil(num_diag, bs);
  if (gs < 2) {
    bs = DivCeil(num_diag, 2);
    gs = 2;
  }
  return {bs, gs};
}

template <typename word>
StripedMatrix CiModuleBasis<word>::Correction(
    const Parameter<word> &param, const Encoder<word> &encoder) const {
  const int n = num_slots_;
  const int64_t M = param.CyclotomicIndex();
  StripedMatrix corr(n, n);
  auto put = [&](int row, int col, Complex v) {
    const int off = ((col - row) % n + n) % n;
    corr.try_emplace(off, n, Complex(0.0, 0.0));
    corr[off][row] = v;
  };
  for (int p = 0; p < n; p++) {
    const int lane = p & (small_degree_ - 1);
    const int block = p >> log_small_;
    const int i = static_cast<int>(BitReverseInt(block, log_rank_));
    if (i == 0) {
      // v = 2 s: the class with no partner.
      put(p, p, Complex(0.5, 0.0));
      continue;
    }
    // The conjugate of the small root the slot's residue mod T sits at:
    // zeta_s^-k depends on s mod T alone.
    const int64_t idx = (static_cast<int64_t>(rank_) *
                         static_cast<int64_t>(param.GetGaloisFactor(lane))) %
                        M;
    const Complex w = std::conj(encoder.GetTwiddleFactor(static_cast<int>(idx)));
    if (i == rank_ / 2) {
      // Its own mirror: v = (1 + w) s.
      put(p, p, Complex(1.0, 0.0) / (Complex(1.0, 0.0) + w));
      continue;
    }
    const int mirror = (rank_ - i) & (rank_ - 1);
    const int q = (static_cast<int>(BitReverseInt(mirror, log_rank_))
                   << log_small_) |
                  lane;
    const Complex den = Complex(1.0, 0.0) - w * w;
    put(p, p, Complex(1.0, 0.0) / den);
    put(p, q, -w / den);
  }
  return corr;
}

template <typename word>
int CiModuleBasis<word>::Chain(ConstContextPtr<word> context,
                               std::vector<StripedMatrix> &matrices,
                               int start_level, std::vector<Transform> &dst,
                               std::vector<int> &diagonals) const {
  const auto &param = context->param_;
  const int n = num_slots_;
  const int num_phases = static_cast<int>(matrices.size());
  std::vector<int> a(num_phases, 0);
  for (int j = 0; j < num_phases; j++) {
    const int w = Window(matrices[j], n);
    // A free window takes whatever closes the chain cleanly: the previous
    // phase's window when it is not the last, zero when it is.
    a[j] = (w >= 0) ? w : 0;
    if (w < 0 && j + 1 < num_phases) a[j] = (j > 0) ? a[j - 1] : 0;
  }
  int prev_a = 0;
  for (int j = 0; j < num_phases; j++) {
    const int pre_rotation = (j == 0) ? -a[0] : (prev_a - a[j]);
    prev_a = a[j];
    const int level = start_level - j;
    const int nd = matrices[j].GetNumDiag();
    auto [bs, gs] = Split(nd);
    dst.emplace_back(context, matrices[j], level,
                     param.GetRescalePrimeProd(level), bs, gs, pre_rotation,
                     a[j]);
    diagonals.push_back(nd);
    std::cout << "  phase " << j << ": " << nd << " diagonals, level "
              << level << ", BSGS " << bs << "x" << gs << ", pre_rotation "
              << pre_rotation << ", pt_rot " << a[j] << std::endl;
  }
  return prev_a;
}

template <typename word>
CiModuleBasis<word>::CiModuleBasis(ConstContextPtr<word> context,
                                   int small_degree, int stc_level,
                                   int cts_level, const Phases &phases,
                                   double stc_const, double cts_const)
    : num_slots_{context->param_.MaxNumSlots()},
      small_degree_{small_degree},
      stc_level_{stc_level},
      cts_level_{cts_level} {
  const auto &param = context->param_;
  const auto &encoder = context->encoder_;
  AssertTrue(param.conjugate_invariant_,
             "CiModuleBasis: the module basis is a conjugate-invariant "
             "object");
  AssertTrue(small_degree > 0 && IsPowOfTwo(small_degree) &&
                 small_degree < num_slots_,
             "CiModuleBasis: small_degree must be a power of two below the "
             "slot count");
  rank_ = num_slots_ / small_degree_;
  log_slots_ = Log2Ceil(num_slots_);
  log_small_ = Log2Ceil(small_degree_);
  log_rank_ = Log2Ceil(rank_);
  const int n = num_slots_;

  auto sum = [](const std::vector<int> &v) {
    return std::accumulate(v.begin(), v.end(), 0);
  };
  auto check_group = [&](const std::vector<int> &v, int stages,
                         const char *who) {
    AssertTrue(v.size() >= 2, std::string("CiModuleBasis: ") + who +
                                  " needs at least two phases (the first "
                                  "lifts to a pair, the last drops to real)");
    AssertTrue(sum(v) == stages, std::string("CiModuleBasis: ") + who +
                                     " must cover exactly its stages");
    for (int c : v) AssertTrue(c >= 1, "CiModuleBasis: an empty phase");
  };

  // The coefficient index each slot position stands for, and its (i, t).
  auto module_index = [&](int pos) {
    return static_cast<int>(BitReverseInt(pos, log_slots_));
  };
  auto i_of = [&](int flat) { return flat & (rank_ - 1); };
  auto t_of = [&](int flat) { return flat >> log_rank_; };

  std::cout << "CiModuleBasis: slots " << n << ", T " << small_degree_
            << ", k " << rank_ << std::endl;

  // ---- SlotToCoeff -------------------------------------------------------
  if (stc_level >= 0) {
    check_group(phases.stc_small, log_small_, "stc_small");
    check_group(phases.stc_twist, log_rank_, "stc_twist");
    const int num_small = static_cast<int>(phases.stc_small.size());
    const int num_phases = num_small + static_cast<int>(phases.stc_twist.size());
    AssertTrue(stc_level >= num_phases,
               "CiModuleBasis: StC needs one level per phase");
    const double const_div = std::pow(stc_const, 1.0 / num_phases);

    // The module scaling, on the coefficient index: 2^[i != 0] 2^[t != 0].
    std::vector<double> col_scale(n, 1.0);
    for (int pos = 0; pos < n; pos++) {
      const int flat = module_index(pos);
      col_scale[pos] = (i_of(flat) != 0 ? 2.0 : 1.0) *
                       (t_of(flat) != 0 ? 2.0 : 1.0);
    }

    std::vector<StripedMatrix> matrices;
    int cumul = 0;
    for (const auto *group : {&phases.stc_small, &phases.stc_twist}) {
      for (int count : *group) {
        std::vector<int> stages;
        for (int s = cumul; s < cumul + count; s++) stages.push_back(s);
        StripedMatrix m =
            CiButterflyStages(param, encoder, n, stages, /*inverse_dir=*/false);
        if (matrices.empty()) ScaleColumns(m, col_scale);
        matrices.push_back(
            StripedMatrix::Mult(m, Complex(const_div, 0.0)));
        cumul += count;
      }
    }
    std::cout << " StC at level " << stc_level << ":" << std::endl;
    std::vector<Transform> all;
    stc_shift_ = Chain(context, matrices, stc_level, all, stc_diagonals_);
    for (int j = 0; j < num_phases; j++) {
      (j < num_small ? stc_small_ : stc_twist_).push_back(std::move(all[j]));
    }
  }

  // ---- CoeffToSlot -------------------------------------------------------
  if (cts_level >= 0) {
    check_group(phases.cts_twist, log_rank_, "cts_twist");
    check_group(phases.cts_small, log_small_, "cts_small");
    const int num_twist = static_cast<int>(phases.cts_twist.size());
    const int num_phases = num_twist + static_cast<int>(phases.cts_small.size());
    AssertTrue(cts_level >= num_phases,
               "CiModuleBasis: CtS needs one level per phase");
    const double const_div = std::pow(cts_const, 1.0 / num_phases);
    // The twist stages return k/2 times (s + w Flip s); the small stages
    // return T times the coordinates. Both are spread over their phases so
    // that no single plaintext carries a 1/256.
    const double twist_div = std::pow(2.0 / rank_, 1.0 / num_twist);
    const double small_div =
        std::pow(1.0 / small_degree_, 1.0 / phases.cts_small.size());

    std::vector<double> row_scale(n, 1.0);
    for (int pos = 0; pos < n; pos++) {
      row_scale[pos] = (i_of(module_index(pos)) != 0) ? 0.5 : 1.0;
    }

    std::vector<StripedMatrix> matrices;
    int top = log_slots_ - 1;
    for (size_t j = 0; j < phases.cts_twist.size(); j++) {
      const int count = phases.cts_twist[j];
      std::vector<int> stages;
      for (int s = top; s > top - count; s--) stages.push_back(s);
      StripedMatrix m =
          CiButterflyStages(param, encoder, n, stages, /*inverse_dir=*/true);
      if (j + 1 == phases.cts_twist.size()) {
        m = StripedMatrix::Mult(Correction(param, encoder), m);
      }
      matrices.push_back(
          StripedMatrix::Mult(m, Complex(const_div * twist_div, 0.0)));
      top -= count;
    }
    AssertTrue(top == log_small_ - 1, "CiModuleBasis: twist stages miscounted");
    for (size_t j = 0; j < phases.cts_small.size(); j++) {
      const int count = phases.cts_small[j];
      std::vector<int> stages;
      for (int s = top; s > top - count; s--) stages.push_back(s);
      StripedMatrix m =
          CiButterflyStages(param, encoder, n, stages, /*inverse_dir=*/true);
      if (j + 1 == phases.cts_small.size()) ScaleRows(m, row_scale);
      matrices.push_back(
          StripedMatrix::Mult(m, Complex(const_div * small_div, 0.0)));
      top -= count;
    }
    AssertTrue(top == -1, "CiModuleBasis: small stages miscounted");
    std::cout << " CtS at level " << cts_level << ":" << std::endl;
    std::vector<Transform> all;
    cts_shift_ = Chain(context, matrices, cts_level, all, cts_diagonals_);
    for (int j = 0; j < num_phases; j++) {
      (j < num_twist ? cts_twist_ : cts_small_).push_back(std::move(all[j]));
    }
  }
  std::cout << " closing rotations: StC " << stc_shift_ << ", CtS "
            << cts_shift_ << std::endl;
}

template <typename word>
void CiModuleBasis<word>::AddRequiredRotations(EvkRequest &req,
                                               bool min_ks) const {
  for (const auto *group : {&stc_small_, &stc_twist_, &cts_twist_, &cts_small_}) {
    for (const auto &t : *group) t.AddRequiredRotations(req, min_ks);
  }
  if (stc_shift_ != 0) {
    req.AddRequest(num_slots_ - stc_shift_, stc_level_ - GetStCNumLevels());
  }
  if (cts_shift_ != 0) {
    req.AddRequest(num_slots_ - cts_shift_, cts_level_ - GetCtSNumLevels());
  }
}

template <typename word>
void CiModuleBasis<word>::EvaluateStC(ConstContextPtr<word> context, Ct &res,
                                      const Ct &input,
                                      const EvkMap<word> &evk_map) const {
  AssertTrue(!stc_small_.empty(), "CiModuleBasis: StC was not built");
  Ct mid, out;
  RunGroup(context, stc_small_, mid, input, evk_map);
  RunGroup(context, stc_twist_, out, mid, evk_map);
  if (stc_shift_ != 0) {
    const int back = num_slots_ - stc_shift_;
    context->HRot(res, out, evk_map.GetRotationKey(back), back);
  } else {
    res = std::move(out);
  }
  res.SetNumSlots(num_slots_);
}

template <typename word>
void CiModuleBasis<word>::EvaluateCtS(ConstContextPtr<word> context, Ct &res,
                                      const Ct &input,
                                      const EvkMap<word> &evk_map) const {
  AssertTrue(!cts_twist_.empty(), "CiModuleBasis: CtS was not built");
  Ct mid, out;
  RunGroup(context, cts_twist_, mid, input, evk_map);
  RunGroup(context, cts_small_, out, mid, evk_map);
  if (cts_shift_ != 0) {
    const int back = num_slots_ - cts_shift_;
    context->HRot(res, out, evk_map.GetRotationKey(back), back);
  } else {
    res = std::move(out);
  }
  res.SetNumSlots(num_slots_);
}

template class CiModuleBasis<uint32_t>;
template class CiModuleBasis<uint64_t>;

}  // namespace cheddar
