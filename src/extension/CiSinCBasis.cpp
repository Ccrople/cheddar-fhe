#include "extension/Profile.h"
#include "extension/CiSinCBasis.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>

#include "common/Assert.h"
#include "extension/EvalSpecialFFT.h"

namespace cheddar {

namespace {

// Entry (row, col) of a StripedMatrix is `m[(col - row) mod n][row]`.
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

StripedMatrix RealPart(const StripedMatrix &m) {
  StripedMatrix res(m.GetHeight(), m.GetWidth());
  for (const auto &[idx, diag] : m) {
    std::vector<Complex> re(diag.size());
    for (size_t j = 0; j < diag.size(); j++) re[j] = Complex(diag[j].real(), 0.0);
    res.emplace(idx, std::move(re));
  }
  return res;
}

// The window rule of CiModuleBasis / the SinC prefix: a phase whose offsets
// straddle zero rides a pre-rotation of minus its most negative offset, a
// whole-lattice phase none, and the chain closes with one HRot.
int Window(const StripedMatrix &m, int n) {
  int gcd = 0;
  int most_negative = 0;
  for (const auto &[idx, _] : m) {
    const int off = ((idx % n) + n) % n;
    gcd = GCD(gcd, off);
    const int signed_off = (off <= n / 2) ? off : off - n;
    most_negative = std::min(most_negative, signed_off);
  }
  if (gcd > 0 && static_cast<int>(m.size()) * gcd == n) return 0;
  return -most_negative;
}

// A lane-preserving block permutation folded on the column side (Doing.md
// 1.5bx, CiSinCConverter's FoldColumnPremap): with `inv` the inverse of the
// premap, column `B k + lane` of `m` moves to `inv[B] k + lane`, so the
// composed matrix consumes the caller's layout directly.
StripedMatrix FoldColumnPremap(const StripedMatrix &m,
                               const std::vector<int> &inv, int k) {
  const int n = m.GetHeight();
  StripedMatrix res(n, n);
  for (const auto &[idx, diag] : m) {
    AssertTrue(((idx % k) + k) % k == 0,
               "CiSinCBasis: a premap fold needs every offset on the "
               "stride-T_l lattice");
    for (int row = 0; row < n; row++) {
      if (diag[row] == Complex(0.0, 0.0)) continue;
      const int col = ((row + idx) % n + n) % n;
      const int new_col = inv[col / k] * k + (col % k);
      const int off = ((new_col - row) % n + n) % n;
      res.try_emplace(off, n, Complex(0.0, 0.0));
      res[off][row] += diag[row];
    }
  }
  return res;
}

std::vector<int> DefaultGroup(const std::vector<int> &given, int stages,
                              const char *who) {
  std::vector<int> v = given.empty() ? std::vector<int>{stages} : given;
  AssertTrue(std::accumulate(v.begin(), v.end(), 0) == stages,
             std::string("CiSinCBasis: ") + who +
                 " must cover exactly its stages");
  for (int c : v) AssertTrue(c >= 1, "CiSinCBasis: an empty phase");
  return v;
}

}  // namespace

template <typename word>
std::pair<int, int> CiSinCBasis<word>::Split(int num_diag) {
  // As CiModuleBasis: at most 32 baby steps for the fused complex giant
  // step, never a gs == 1 layout.
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
CiSinCBasis<word>::CiSinCBasis(int big_degree, int small_degree,
                               int sub_degree)
    : num_slots_{big_degree},
      outer_rank_{big_degree / small_degree},
      inner_rank_{small_degree / sub_degree},
      lane_degree_{sub_degree} {
  AssertTrue(IsPowOfTwo(big_degree) && IsPowOfTwo(small_degree) &&
                 IsPowOfTwo(sub_degree) && sub_degree >= 2 &&
                 small_degree > sub_degree && big_degree > small_degree &&
                 big_degree % small_degree == 0 &&
                 small_degree % sub_degree == 0,
             "CiSinCBasis: degrees must be powers of two with sub_degree < "
             "small_degree < big_degree");
  log_slots_ = Log2Ceil(num_slots_);
  log_outer_ = Log2Ceil(outer_rank_);
  log_inner_ = Log2Ceil(inner_rank_);
  log_lane_ = Log2Ceil(lane_degree_);
  std::cout << "CiSinCBasis: slots " << num_slots_ << ", tower (" << outer_rank_
            << "; " << inner_rank_ << "; " << lane_degree_ << ")" << std::endl;
}

template <typename word>
StripedMatrix CiSinCBasis<word>::Correction(const Parameter<word> &param,
                                            const Encoder<word> &encoder,
                                            bool inner) const {
  const int n = num_slots_;
  const int64_t M = param.CyclotomicIndex();
  // The pair correction of Doing.md 3.5, `s = (v - w Flip v) / (1 - w^2)`,
  // on the module the twist group just diagonalised: the outer one (T = the
  // small degree, k = k_o, the class field the top log2 k_o bits, w the
  // conjugate small root at the slot's residue mod T) or the inner one
  // inside each part (T = T_l, k = k_i, the field the next log2 k_i bits,
  // w the conjugate LANE root, index k_o k_i times the Galois factor).
  const int T = inner ? lane_degree_ : GetOuterSmallDegree();
  const int k = inner ? inner_rank_ : outer_rank_;
  const int log_T = inner ? log_lane_ : (log_inner_ + log_lane_);
  const int log_k = inner ? log_inner_ : log_outer_;
  const int root_rank = inner ? outer_rank_ * inner_rank_ : outer_rank_;
  StripedMatrix corr(n, n);
  auto put = [&](int row, int col, Complex v) {
    const int off = ((col - row) % n + n) % n;
    corr.try_emplace(off, n, Complex(0.0, 0.0));
    corr[off][row] = v;
  };
  for (int p = 0; p < n; p++) {
    const int lane = p & (T - 1);
    const int field = (p >> log_T) & (k - 1);
    const int i = static_cast<int>(BitReverseInt(field, log_k));
    if (i == 0) {
      put(p, p, Complex(0.5, 0.0));
      continue;
    }
    const int64_t idx = (static_cast<int64_t>(root_rank) *
                         static_cast<int64_t>(param.GetGaloisFactor(lane))) %
                        M;
    const Complex w = std::conj(encoder.GetTwiddleFactor(static_cast<int>(idx)));
    if (i == k / 2) {
      put(p, p, Complex(1.0, 0.0) / (Complex(1.0, 0.0) + w));
      continue;
    }
    const int mirror = (k - i) & (k - 1);
    const int q = (p & ~((k - 1) << log_T)) |
                  (static_cast<int>(BitReverseInt(mirror, log_k)) << log_T);
    const Complex den = Complex(1.0, 0.0) - w * w;
    put(p, p, Complex(1.0, 0.0) / den);
    put(p, q, -w / den);
  }
  return corr;
}

template <typename word>
void CiSinCBasis<word>::Compile(ConstContextPtr<word> context,
                                std::vector<StripedMatrix> &matrices,
                                const std::vector<int> &group_sizes,
                                int start_level, Chain &chain) const {
  const auto &param = context->param_;
  const int n = num_slots_;
  chain.context = context;
  chain.level = start_level;
  chain.groups.clear();
  chain.diagonals.clear();
  int carried = 0;
  int j = 0;
  int level = start_level;
  for (int size : group_sizes) {
    Group group;
    for (int q = 0; q < size; q++, j++) {
      const bool single = (size == 1);
      StripedMatrix m = single ? RealPart(matrices[j]) : matrices[j];
      const int w = Window(m, n);
      const int pre_rotation = -w;
      carried += w;
      int gcd = 0;
      int max_rot = 0;
      for (const auto &[idx, _] : m) {
        const int rot = ((idx - pre_rotation) % n + n) % n;
        gcd = GCD(gcd, rot);
        max_rot = std::max(max_rot, rot);
      }
      const int nd = m.GetNumDiag();
      const int span = (gcd > 0) ? max_rot / gcd + 1 : nd;
      auto [bs, gs] = Split(std::max(nd, span));
      const double scale = param.GetRescalePrimeProd(level);
      if (single) {
        group.real.emplace_back(context, m, level, scale, bs, gs, pre_rotation,
                                carried);
      } else {
        group.pair.emplace_back(context, m, level, scale, bs, gs, pre_rotation,
                                carried);
      }
      chain.diagonals.push_back(nd);
      std::cout << "  phase " << j << (single ? " (real)" : " (pair)") << ": "
                << nd << " diagonals on stride " << gcd << ", span " << span
                << ", level " << level << ", BSGS " << bs << "x" << gs
                << ", pre_rotation " << pre_rotation << ", pt_rot " << carried
                << std::endl;
      level--;
    }
    chain.groups.push_back(std::move(group));
  }
  chain.shift = carried;
  std::cout << "  closing rotation " << carried << std::endl;
}

template <typename word>
void CiSinCBasis<word>::PrepareForward(ConstContextPtr<word> context,
                                       const std::string &name, int level,
                                       const std::vector<int> *premap,
                                       const Phases &phases, bool fold_premap) {
  const auto &param = context->param_;
  const auto &encoder = context->encoder_;
  AssertTrue(param.conjugate_invariant_ && param.degree_ == num_slots_ &&
                 param.MaxNumSlots() == num_slots_,
             "CiSinCBasis: the forward runs on the conjugate-invariant big "
             "ring, one real slot per coefficient");
  const std::vector<int> inner =
      DefaultGroup(phases.forward_inner, log_inner_, "forward_inner");
  const std::vector<int> outer =
      DefaultGroup(phases.forward_outer, log_outer_, "forward_outer");
  const bool standalone = (premap != nullptr && !fold_premap);
  const int num_phases =
      static_cast<int>(inner.size() + outer.size()) + (standalone ? 1 : 0);
  AssertTrue(level >= num_phases,
             "CiSinCBasis: the forward needs one level per phase");
  const int n = num_slots_;

  // The column scaling on the primary addresses, `2^[i != 0] 2^[j != 0]`
  // (the host derivation's check 2): slot s is block BitRev(j k_o + i).
  std::vector<double> col(n, 1.0);
  for (int s = 0; s < n; s++) {
    const int flat = static_cast<int>(
        BitReverseInt(s >> log_lane_, log_outer_ + log_inner_));
    const int i = flat & (outer_rank_ - 1);
    const int jj = flat >> log_outer_;
    col[s] = (i != 0 ? 2.0 : 1.0) * (jj != 0 ? 2.0 : 1.0);
  }
  std::vector<int> inv;
  if (premap != nullptr) {
    const int nb = n / lane_degree_;
    AssertTrue(static_cast<int>(premap->size()) == nb,
               "CiSinCBasis: the premap must cover every block");
    inv.assign(nb, -1);
    for (int b = 0; b < nb; b++) {
      const int dst = (*premap)[b];
      AssertTrue(dst >= 0 && dst < nb && inv[dst] == -1,
                 "CiSinCBasis: the premap is not a bijection");
      inv[dst] = b;
    }
  }

  std::vector<StripedMatrix> matrices;
  if (standalone) {
    // The premap as a permutation of its own: the caller's slot s (block b)
    // moves to the primary block (*premap)[b], lane untouched.
    StripedMatrix perm(n, n);
    for (int s = 0; s < n; s++) {
      const int dst = (*premap)[s >> log_lane_] * lane_degree_ +
                      (s & (lane_degree_ - 1));
      const int off = ((s - dst) % n + n) % n;
      perm.try_emplace(off, n, Complex(0.0, 0.0));
      perm[off][dst] = Complex(1.0, 0.0);
    }
    matrices.push_back(std::move(perm));
  }
  int cumul = log_lane_;
  bool first_stage_phase = true;
  for (const auto *group : {&inner, &outer}) {
    for (int count : *group) {
      std::vector<int> stages;
      for (int s = cumul; s < cumul + count; s++) stages.push_back(s);
      StripedMatrix m =
          CiButterflyStages(param, encoder, n, stages, /*inverse_dir=*/false);
      if (first_stage_phase) {
        ScaleColumns(m, col);
        // The premap composes OUTERMOST, after the scaling: it relabels
        // which caller slot feeds each primary address.
        if (premap != nullptr && fold_premap) {
          m = FoldColumnPremap(m, inv, lane_degree_);
        }
        first_stage_phase = false;
      }
      matrices.push_back(std::move(m));
      cumul += count;
    }
  }
  AssertTrue(cumul == log_slots_, "CiSinCBasis: forward stages miscounted");
  for (auto &f : forwards_) {
    AssertTrue(f.name != name, "CiSinCBasis: a forward named " + name +
                                   " is already compiled");
  }
  std::cout << "CiSinCBasis forward '" << name << "' at level " << level
            << (premap == nullptr ? ""
                                  : (fold_premap ? " (premap folded)"
                                                 : " (premap standalone)"))
            << ":" << std::endl;
  NamedForward nf;
  nf.name = name;
  std::vector<int> sizes;
  if (standalone) sizes.push_back(1);
  sizes.push_back(static_cast<int>(inner.size()));
  sizes.push_back(static_cast<int>(outer.size()));
  Compile(context, matrices, sizes, level, nf.chain);
  forwards_.push_back(std::move(nf));
}

template <typename word>
void CiSinCBasis<word>::PrepareCtS(ConstContextPtr<word> context, int level,
                                   double cts_const, const Phases &phases) {
  const auto &param = context->param_;
  const auto &encoder = context->encoder_;
  AssertTrue(param.conjugate_invariant_ && param.degree_ == num_slots_ &&
                 param.MaxNumSlots() == num_slots_,
             "CiSinCBasis: the CtS runs on the conjugate-invariant big ring");
  const std::vector<int> outer =
      DefaultGroup(phases.cts_outer, log_outer_, "cts_outer");
  const std::vector<int> inner =
      DefaultGroup(phases.cts_inner, log_inner_, "cts_inner");
  const std::vector<int> lane =
      DefaultGroup(phases.cts_lane, log_lane_, "cts_lane");
  const int num_phases =
      static_cast<int>(outer.size() + inner.size() + lane.size());
  // Leading thin single-terminal levels (an even landing's top), set aside
  // for the pure-rescale prologue in EvaluateCtS; the compiled phases start
  // at the topmost THICK level.
  constexpr double kThinRescaleProd = 1073741824.0;  // 2^30
  cts_thin_levels_.clear();
  cts_thin_consts_.clear();
  int thick_level = level;
  while (param.GetRescalePrimeProd(thick_level) < kThinRescaleProd) {
    cts_thin_levels_.push_back(thick_level);
    Constant<word> thin_c;
    encoder.EncodeConstant(thin_c, thick_level,
                           param.GetRescalePrimeProd(thick_level), 1.0);
    cts_thin_consts_.push_back(std::move(thin_c));
    thick_level--;
  }
  if (!cts_thin_levels_.empty()) {
    std::cout << "CiSinCBasis CtS: " << cts_thin_levels_.size()
              << " thin single-terminal level(s) at the top, consumed by a "
                 "pure rescale" << std::endl;
  }
  AssertTrue(thick_level >= num_phases,
             "CiSinCBasis: the CtS needs one THICK level per phase (the thin "
             "single-terminal levels take none)");
  const int n = num_slots_;
  const double const_div = std::pow(cts_const, 1.0 / num_phases);
  // The twist groups return k/2 times the corrected pair sums and the lane
  // stages T_l times the coordinates (the host derivation's check 3), each
  // spread over its own phases; the residual `2^[i != 0] 2^[j != 0]` rides
  // the rows of the last phase.
  const double outer_div = std::pow(2.0 / outer_rank_, 1.0 / outer.size());
  const double inner_div = std::pow(2.0 / inner_rank_, 1.0 / inner.size());
  const double lane_div = std::pow(1.0 / lane_degree_, 1.0 / lane.size());
  std::vector<double> row_scale(n, 1.0);
  for (int s = 0; s < n; s++) {
    const int flat = static_cast<int>(BitReverseInt(s, log_slots_));
    const int i = flat & (outer_rank_ - 1);
    const int jj = (flat >> log_outer_) & (inner_rank_ - 1);
    row_scale[s] = (i != 0 ? 0.5 : 1.0) * (jj != 0 ? 0.5 : 1.0);
  }

  std::vector<StripedMatrix> matrices;
  int top = log_slots_ - 1;
  auto take = [&](const std::vector<int> &group, bool correct_last,
                  bool inner_corr, bool rows_last, double div) {
    for (size_t j = 0; j < group.size(); j++) {
      const int count = group[j];
      std::vector<int> stages;
      for (int s = top; s > top - count; s--) stages.push_back(s);
      StripedMatrix m =
          CiButterflyStages(param, encoder, n, stages, /*inverse_dir=*/true);
      const bool last = (j + 1 == group.size());
      if (last && correct_last) {
        m = StripedMatrix::Mult(Correction(param, encoder, inner_corr), m);
      }
      if (last && rows_last) ScaleRows(m, row_scale);
      matrices.push_back(StripedMatrix::Mult(m, Complex(const_div * div, 0.0)));
      top -= count;
    }
  };
  take(outer, /*correct_last=*/true, /*inner_corr=*/false, false, outer_div);
  AssertTrue(top == log_inner_ + log_lane_ - 1,
             "CiSinCBasis: outer twist stages miscounted");
  take(inner, /*correct_last=*/true, /*inner_corr=*/true, false, inner_div);
  AssertTrue(top == log_lane_ - 1, "CiSinCBasis: inner twist stages miscounted");
  take(lane, false, false, /*rows_last=*/true, lane_div);
  AssertTrue(top == -1, "CiSinCBasis: lane stages miscounted");
  std::cout << "CiSinCBasis CtS at level " << level
            << (cts_thin_levels_.empty()
                    ? ""
                    : " (phases from " + std::to_string(thick_level) + ")")
            << ":" << std::endl;
  const std::vector<int> sizes{static_cast<int>(outer.size()),
                               static_cast<int>(inner.size()),
                               static_cast<int>(lane.size())};
  Compile(context, matrices, sizes, thick_level, cts_);
  // The public entry level is where the caller hands its ciphertext (the
  // thin prologue runs there); the compiled phases sit below it.
  cts_.level = level;
}

template <typename word>
void CiSinCBasis<word>::PreparePrefix(ConstContextPtr<word> context, int level,
                                      double constant, double pt_scale,
                                      const Phases &phases) {
  const auto &param = context->param_;
  const auto &encoder = context->encoder_;
  AssertTrue(param.conjugate_invariant_ && param.degree_ == num_slots_ &&
                 param.MaxNumSlots() == num_slots_,
             "CiSinCBasis: the prefix runs on the conjugate-invariant big "
             "ring");
  const std::vector<int> lane = DefaultGroup(phases.prefix, log_lane_, "prefix");
  AssertTrue(level >= static_cast<int>(lane.size()),
             "CiSinCBasis: the prefix needs one level per phase");
  const int n = num_slots_;
  // The lane coefficient's column scaling `2^[t != 0]` (check 4): slot s
  // holds tower coordinate BitRev(s), whose t is its top log2 T_l bits.
  std::vector<double> col(n, 1.0);
  for (int s = 0; s < n; s++) {
    const int flat = static_cast<int>(BitReverseInt(s, log_slots_));
    const int t = flat >> (log_outer_ + log_inner_);
    col[s] = (t != 0) ? 2.0 : 1.0;
  }
  std::vector<StripedMatrix> matrices;
  int cumul = 0;
  for (int count : lane) {
    std::vector<int> stages;
    for (int s = cumul; s < cumul + count; s++) stages.push_back(s);
    StripedMatrix m =
        CiButterflyStages(param, encoder, n, stages, /*inverse_dir=*/false);
    if (matrices.empty()) {
      ScaleColumns(m, col);
      if (constant != 1.0) m = StripedMatrix::Mult(m, Complex(constant, 0.0));
    }
    matrices.push_back(std::move(m));
    cumul += count;
  }
  std::cout << "CiSinCBasis prefix at level " << level << ":" << std::endl;
  Compile(context, matrices, {static_cast<int>(lane.size())}, level, prefix_);
  if (pt_scale > 0.0) {
    // The first phase's plaintexts at the caller's scale (a Canonicalise
    // fold, as the ordinary SinC prefix): rebuild it alone.
    AssertTrue(lane.size() == 1,
               "CiSinCBasis: a custom prefix scale needs a one-phase prefix");
    Group &g = prefix_.groups.front();
    const StripedMatrix m = RealPart(matrices.front());
    const int w = Window(m, n);
    int gcd = 0, max_rot = 0;
    for (const auto &[idx, _] : m) {
      const int rot = ((idx + w) % n + n) % n;
      gcd = GCD(gcd, rot);
      max_rot = std::max(max_rot, rot);
    }
    const int nd = m.GetNumDiag();
    const int span = (gcd > 0) ? max_rot / gcd + 1 : nd;
    auto [bs, gs] = Split(std::max(nd, span));
    g.real.clear();
    g.real.emplace_back(context, m, level, pt_scale, bs, gs, -w, w);
    std::cout << "  (prefix plaintexts at scale 2^" << std::log2(pt_scale)
              << ")" << std::endl;
  }
}

template <typename word>
int CiSinCBasis<word>::NumLevels(const Chain &chain) {
  int levels = 0;
  for (const auto &g : chain.groups) {
    levels += static_cast<int>(g.real.size() + g.pair.size());
  }
  return levels;
}

template <typename word>
const typename CiSinCBasis<word>::Chain &CiSinCBasis<word>::FindForward(
    const std::string &name) const {
  for (const auto &f : forwards_) {
    if (f.name == name) return f.chain;
  }
  AssertTrue(false, "CiSinCBasis: no forward named " + name);
  return forwards_.front().chain;
}

template <typename word>
bool CiSinCBasis<word>::HasForward(const std::string &name) const {
  for (const auto &f : forwards_) {
    if (f.name == name) return true;
  }
  return false;
}

template <typename word>
int CiSinCBasis<word>::GetForwardLevel(const std::string &name) const {
  return FindForward(name).level;
}

template <typename word>
int CiSinCBasis<word>::GetForwardNumLevels(const std::string &name) const {
  return NumLevels(FindForward(name));
}

template <typename word>
const std::vector<int> &CiSinCBasis<word>::GetForwardDiagonals(
    const std::string &name) const {
  return FindForward(name).diagonals;
}

template <typename word>
void CiSinCBasis<word>::AddChainRotations(const Chain &chain, int num_slots,
                                          EvkRequest &req, bool min_ks) {
  for (const auto &g : chain.groups) {
    for (const auto &t : g.real) t.AddRequiredRotations(req, min_ks);
    for (const auto &t : g.pair) t.AddRequiredRotations(req, min_ks);
  }
  if (chain.shift != 0) {
    req.AddRequest(num_slots - chain.shift, chain.level - NumLevels(chain));
  }
}

template <typename word>
void CiSinCBasis<word>::AddForwardRotations(EvkRequest &req,
                                            bool min_ks) const {
  for (const auto &f : forwards_) {
    AddChainRotations(f.chain, num_slots_, req, min_ks);
  }
}

template <typename word>
void CiSinCBasis<word>::AddCtSRotations(EvkRequest &req, bool min_ks) const {
  if (cts_.level >= 0) AddChainRotations(cts_, num_slots_, req, min_ks);
}

template <typename word>
void CiSinCBasis<word>::AddPrefixRotations(EvkRequest &req,
                                           bool min_ks) const {
  if (prefix_.level >= 0) AddChainRotations(prefix_, num_slots_, req, min_ks);
}

template <typename word>
void CiSinCBasis<word>::AddRequiredRotations(EvkRequest &req,
                                             bool min_ks) const {
  AddForwardRotations(req, min_ks);
  AddCtSRotations(req, min_ks);
  AddPrefixRotations(req, min_ks);
}

template <typename word>
void CiSinCBasis<word>::RunGroup(ConstContextPtr<word> context,
                                 const Group &group, Ct &res, const Ct &input,
                                 const EvkMap<word> &evk_map) {
  if (!group.real.empty()) {
    group.real.front().Evaluate(context, res, input, evk_map);
    return;
  }
  Ct re, im;
  group.pair.front().EvaluateFromReal(context, re, im, input, evk_map);
  for (size_t j = 1; j + 1 < group.pair.size(); j++) {
    group.pair[j].EvaluatePair(context, re, im, re, im, evk_map);
  }
  group.pair.back().EvaluateToReal(context, res, re, im, evk_map);
}

template <typename word>
void CiSinCBasis<word>::Run(const Chain &chain, Ct &res, const Ct &input,
                            const EvkMap<word> &evk_map) const {
  AssertTrue(!chain.groups.empty(), "CiSinCBasis: this chain was not built");
  const auto &context = chain.context;
  Ct cur;
  RunGroup(context, chain.groups[0], cur, input, evk_map);
  for (size_t g = 1; g < chain.groups.size(); g++) {
    Ct next;
    RunGroup(context, chain.groups[g], next, cur, evk_map);
    cur = std::move(next);
  }
  if (chain.shift != 0) {
    const int back = num_slots_ - chain.shift;
    context->HRot(res, cur, evk_map.GetRotationKey(back), back);
  } else {
    res = std::move(cur);
  }
  res.SetNumSlots(num_slots_);
}

template <typename word>
void CiSinCBasis<word>::Forward(const std::string &name, Ct &res,
                                const Ct &input,
                                const EvkMap<word> &evk_map) const {
  NvtxScope _nv("tower forward");
  Run(FindForward(name), res, input, evk_map);
}

template <typename word>
void CiSinCBasis<word>::EvaluateCtS(Ct &res, const Ct &input,
                                    const EvkMap<word> &evk_map) const {
  NvtxScope _nv("tower CtS");
  // Consume the thin single-terminal levels with a pure rescale (a
  // constant-1 multiply at the level's own rescale product, then a rescale),
  // exactly as EvalSpecialFFT's CoeffToSlot does.
  if (!cts_thin_levels_.empty()) {
    const auto &context = cts_.context;
    Ct pre;
    context->MultUnsafe(pre, input, cts_thin_consts_[0], cts_thin_levels_[0]);
    context->Rescale(pre, pre);
    for (size_t k = 1; k < cts_thin_levels_.size(); k++) {
      context->MultUnsafe(pre, pre, cts_thin_consts_[k], cts_thin_levels_[k]);
      context->Rescale(pre, pre);
    }
    Run(cts_, res, pre, evk_map);
    return;
  }
  Run(cts_, res, input, evk_map);
}

template <typename word>
void CiSinCBasis<word>::Prefix(Ct &res, const Ct &input,
                               const EvkMap<word> &evk_map) const {
  NvtxScope _nv("tower prefix");
  Run(prefix_, res, input, evk_map);
}

template class CiSinCBasis<uint32_t>;
template class CiSinCBasis<uint64_t>;

}  // namespace cheddar
