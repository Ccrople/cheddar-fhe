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

  // ---- SlotToCoeff -------------------------------------------------------
  if (stc_level >= 0) {
    check_group(phases.stc_small, log_small_, "stc_small");
    check_group(phases.stc_twist, log_rank_, "stc_twist");
    const int num_phases =
        static_cast<int>(phases.stc_small.size() + phases.stc_twist.size());
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

    int level = stc_level;
    int cumul = 0;
    bool first = true;
    auto build = [&](const std::vector<int> &group,
                     std::vector<Transform> &dst) {
      for (int count : group) {
        std::vector<int> stages;
        for (int s = cumul; s < cumul + count; s++) stages.push_back(s);
        StripedMatrix m =
            CiButterflyStages(param, encoder, n, stages, /*inverse_dir=*/false);
        if (first) ScaleColumns(m, col_scale);
        first = false;
        m = StripedMatrix::Mult(m, Complex(const_div, 0.0));
        const int nd = m.GetNumDiag();
        auto [bs, gs] = Split(nd);
        dst.emplace_back(context, m, level, param.GetRescalePrimeProd(level),
                         bs, gs, 0, 0);
        stc_diagonals_.push_back(nd);
        level--;
        cumul += count;
      }
    };
    build(phases.stc_small, stc_small_);
    build(phases.stc_twist, stc_twist_);
  }

  // ---- CoeffToSlot -------------------------------------------------------
  if (cts_level >= 0) {
    check_group(phases.cts_twist, log_rank_, "cts_twist");
    check_group(phases.cts_small, log_small_, "cts_small");
    const int num_phases =
        static_cast<int>(phases.cts_twist.size() + phases.cts_small.size());
    AssertTrue(cts_level >= num_phases,
               "CiModuleBasis: CtS needs one level per phase");
    const double const_div = std::pow(cts_const, 1.0 / num_phases);
    // The twist stages return k/2 times (s + w Flip s); the small stages
    // return T times the coordinates. Both are spread over their phases so
    // that no single plaintext carries a 1/256.
    const double twist_div =
        std::pow(2.0 / rank_, 1.0 / phases.cts_twist.size());
    const double small_div =
        std::pow(1.0 / small_degree_, 1.0 / phases.cts_small.size());

    std::vector<double> row_scale(n, 1.0);
    for (int pos = 0; pos < n; pos++) {
      row_scale[pos] = (i_of(module_index(pos)) != 0) ? 0.5 : 1.0;
    }

    int level = cts_level;
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
      m = StripedMatrix::Mult(m, Complex(const_div * twist_div, 0.0));
      const int nd = m.GetNumDiag();
      auto [bs, gs] = Split(nd);
      cts_twist_.emplace_back(context, m, level,
                              param.GetRescalePrimeProd(level), bs, gs, 0, 0);
      cts_diagonals_.push_back(nd);
      level--;
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
      m = StripedMatrix::Mult(m, Complex(const_div * small_div, 0.0));
      const int nd = m.GetNumDiag();
      auto [bs, gs] = Split(nd);
      cts_small_.emplace_back(context, m, level,
                              param.GetRescalePrimeProd(level), bs, gs, 0, 0);
      cts_diagonals_.push_back(nd);
      level--;
      top -= count;
    }
    AssertTrue(top == -1, "CiModuleBasis: small stages miscounted");
  }

  std::cout << "CiModuleBasis: slots " << n << ", T " << small_degree_
            << ", k " << rank_;
  if (stc_level >= 0) {
    std::cout << "; StC at level " << stc_level << " in " << GetStCNumLevels()
              << " phases, diagonals";
    for (int d : stc_diagonals_) std::cout << " " << d;
  }
  if (cts_level >= 0) {
    std::cout << "; CtS at level " << cts_level << " in " << GetCtSNumLevels()
              << " phases, diagonals";
    for (int d : cts_diagonals_) std::cout << " " << d;
  }
  std::cout << std::endl;
}

template <typename word>
void CiModuleBasis<word>::AddRequiredRotations(EvkRequest &req,
                                               bool min_ks) const {
  for (const auto *group : {&stc_small_, &stc_twist_, &cts_twist_, &cts_small_}) {
    for (const auto &t : *group) t.AddRequiredRotations(req, min_ks);
  }
}

namespace {

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
void CiModuleBasis<word>::EvaluateStC(ConstContextPtr<word> context, Ct &res,
                                      const Ct &input,
                                      const EvkMap<word> &evk_map) const {
  AssertTrue(!stc_small_.empty(), "CiModuleBasis: StC was not built");
  Ct mid;
  RunGroup(context, stc_small_, mid, input, evk_map);
  RunGroup(context, stc_twist_, res, mid, evk_map);
  res.SetNumSlots(num_slots_);
}

template <typename word>
void CiModuleBasis<word>::EvaluateCtS(ConstContextPtr<word> context, Ct &res,
                                      const Ct &input,
                                      const EvkMap<word> &evk_map) const {
  AssertTrue(!cts_twist_.empty(), "CiModuleBasis: CtS was not built");
  Ct mid;
  RunGroup(context, cts_twist_, mid, input, evk_map);
  RunGroup(context, cts_small_, res, mid, evk_map);
  res.SetNumSlots(num_slots_);
}

template class CiModuleBasis<uint32_t>;
template class CiModuleBasis<uint64_t>;

}  // namespace cheddar
