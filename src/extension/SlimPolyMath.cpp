#include "extension/SlimPolyMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>

#include "common/Assert.h"

namespace cheddar {
namespace slim {

namespace {

using Cplx = std::complex<double>;

constexpr double kTiny = 1e-300;

//! A polynomial's own scale, for turning absolute residuals into relative
//! ones. The largest coefficient is the right yardstick here: the roots are
//! found from the coefficients, so a residual is only meaningful against them.
double CoeffScale(const ChebPoly &c) {
  double s = 0.0;
  for (double v : c) s = std::max(s, std::abs(v));
  return std::max(s, kTiny);
}

//! Drop trailing coefficients that are pure round-off, so that `Degree` and
//! the leading-coefficient tests agree with each other.
void Trim(ChebPoly &c) {
  const double eps = 1e-14 * CoeffScale(c);
  while (c.size() > 1 && std::abs(c.back()) <= eps) c.pop_back();
}

}  // namespace

int Degree(const ChebPoly &c) {
  ChebPoly t = c;
  Trim(t);
  return static_cast<int>(t.size()) - 1;
}

double Eval(const ChebPoly &c, double x) {
  const int n = static_cast<int>(c.size()) - 1;
  if (n < 0) return 0.0;
  if (n == 0) return c[0];
  double b0 = 0.0, b1 = 0.0;
  for (int k = n; k >= 1; k--) {
    const double t = 2.0 * x * b0 - b1 + c[k];
    b1 = b0;
    b0 = t;
  }
  return x * b0 - b1 + c[0];
}

Cplx EvalComplex(const ChebPoly &c, const Cplx &z) {
  const int n = static_cast<int>(c.size()) - 1;
  if (n < 0) return Cplx(0.0, 0.0);
  if (n == 0) return Cplx(c[0], 0.0);
  Cplx b0(0.0, 0.0), b1(0.0, 0.0);
  for (int k = n; k >= 1; k--) {
    const Cplx t = 2.0 * z * b0 - b1 + c[k];
    b1 = b0;
    b0 = t;
  }
  return z * b0 - b1 + c[0];
}

ChebPoly Add(const ChebPoly &a, const ChebPoly &b) {
  ChebPoly r(std::max(a.size(), b.size()), 0.0);
  for (size_t i = 0; i < a.size(); i++) r[i] += a[i];
  for (size_t i = 0; i < b.size(); i++) r[i] += b[i];
  return r;
}

ChebPoly Sub(const ChebPoly &a, const ChebPoly &b) {
  ChebPoly r(std::max(a.size(), b.size()), 0.0);
  for (size_t i = 0; i < a.size(); i++) r[i] += a[i];
  for (size_t i = 0; i < b.size(); i++) r[i] -= b[i];
  return r;
}

ChebPoly Scale(const ChebPoly &a, double s) {
  ChebPoly r = a;
  for (double &v : r) v *= s;
  return r;
}

ChebPoly Mul(const ChebPoly &a, const ChebPoly &b) {
  if (a.empty() || b.empty()) return ChebPoly{0.0};
  // T_i T_j = (T_{i+j} + T_{|i-j|}) / 2, which is the whole rule. Doing this
  // in the Chebyshev basis rather than converting to monomials and back is
  // the point: at degree 32 the round trip loses more than the products cost.
  ChebPoly r(a.size() + b.size() - 1, 0.0);
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] == 0.0) continue;
    for (size_t j = 0; j < b.size(); j++) {
      if (b[j] == 0.0) continue;
      const double h = 0.5 * a[i] * b[j];
      r[i + j] += h;
      r[i > j ? i - j : j - i] += h;
    }
  }
  return r;
}

ChebPoly Deriv(const ChebPoly &c) {
  const int n = static_cast<int>(c.size()) - 1;
  if (n <= 0) return ChebPoly{0.0};
  // b_{k-1} = b_{k+1} + 2 k a_k, downward, then b_0 halves. Checked by hand
  // against T_1 (-> 1), T_2 (-> 4 T_1) and T_3 (-> 3 T_0 + 6 T_2) before it
  // was used for anything.
  std::vector<double> b(n + 3, 0.0);
  for (int k = n; k >= 1; k--) b[k - 1] = b[k + 1] + 2.0 * k * c[k];
  ChebPoly d(b.begin(), b.begin() + n);
  d[0] *= 0.5;
  return d;
}

double SupNorm(const ChebPoly &c, double lo, double hi, int grid) {
  grid = std::max(grid, 2);
  double s = 0.0;
  for (int i = 0; i < grid; i++) {
    const double x = lo + (hi - lo) * i / (grid - 1);
    s = std::max(s, std::abs(Eval(c, x)));
  }
  return s;
}

std::vector<Cplx> Roots(const ChebPoly &c_in, double *residual) {
  ChebPoly c = c_in;
  Trim(c);
  const int n = static_cast<int>(c.size()) - 1;
  std::vector<Cplx> z;
  if (n <= 0) {
    if (residual != nullptr) *residual = 0.0;
    return z;
  }
  const ChebPoly dc = Deriv(c);

  // Aberth-Ehrlich, started from points spread over an annulus around the
  // interval. These polynomials are fits on [-1, 1], so their roots sit in a
  // Bernstein ellipse of modest eccentricity; a single circle is a poor start
  // because it makes every iterate collide, and the offset angle is what keeps
  // conjugate pairs from being initialised on top of each other.
  z.resize(n);
  for (int i = 0; i < n; i++) {
    const double frac = (i + 0.5) / n;
    const double ang = 2.0 * M_PI * frac + 0.35;
    const double rad = 0.45 + 1.35 * ((i % 3) / 2.0);
    z[i] = Cplx(rad * std::cos(ang), rad * std::sin(ang));
  }

  constexpr int kIters = 4000;
  for (int it = 0; it < kIters; it++) {
    double move = 0.0;
    for (int i = 0; i < n; i++) {
      const Cplx f = EvalComplex(c, z[i]);
      const Cplx fp = EvalComplex(dc, z[i]);
      if (std::abs(fp) < kTiny) continue;
      const Cplx ratio = f / fp;
      Cplx sum(0.0, 0.0);
      for (int j = 0; j < n; j++) {
        if (j == i) continue;
        const Cplx d = z[i] - z[j];
        if (std::abs(d) < kTiny) continue;
        sum += 1.0 / d;
      }
      const Cplx den = 1.0 - ratio * sum;
      if (std::abs(den) < kTiny) continue;
      const Cplx step = ratio / den;
      z[i] -= step;
      move = std::max(move, std::abs(step));
    }
    if (move < 1e-15) break;
  }

  if (residual != nullptr) {
    // Relative to the coefficient scale, and to the polynomial's own size at
    // the root's radius -- a root far outside the interval has a huge |p'| and
    // an absolute residual there means nothing.
    const double scale = CoeffScale(c);
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
      const double denom =
          std::max(scale * std::pow(std::max(1.0, std::abs(z[i])), n), kTiny);
      worst = std::max(worst, std::abs(EvalComplex(c, z[i])) / denom);
    }
    *residual = worst;
  }
  return z;
}

double GlobalMin(const ChebPoly &c) {
  ChebPoly p = c;
  Trim(p);
  const int n = Degree(p);
  if (n <= 0) return p.empty() ? 0.0 : p[0];
  AssertTrue(n % 2 == 0,
             "slim::GlobalMin: odd degree has no finite minimum over R");
  AssertTrue(p[n] > 0.0,
             "slim::GlobalMin: a negative leading coefficient has no finite "
             "minimum over R (lemma 1 needs a positive one)");

  double best = std::numeric_limits<double>::infinity();
  // The minimum is at a critical point, so the real roots of p' are the whole
  // answer; the grid below only exists to catch a root finder that missed one,
  // and it is checked rather than trusted.
  const ChebPoly dp = Deriv(p);
  const std::vector<Cplx> cr = Roots(dp);
  for (const Cplx &r : cr) {
    if (std::abs(r.imag()) > 1e-7 * std::max(1.0, std::abs(r.real()))) continue;
    best = std::min(best, Eval(p, r.real()));
  }
  double grid_best = std::numeric_limits<double>::infinity();
  double reach = 1.0;
  for (const Cplx &r : cr) reach = std::max(reach, std::abs(r.real()));
  reach = std::min(reach * 1.5 + 0.5, 64.0);
  constexpr int kGrid = 40001;
  for (int i = 0; i < kGrid; i++) {
    const double x = -reach + 2.0 * reach * i / (kGrid - 1);
    grid_best = std::min(grid_best, Eval(p, x));
  }
  return std::min(best, grid_best);
}

namespace {

//! One conjugate pair as `(X - re)^2 + im^2`, which is [SYLPH]'s
//! `(X - a_{2i})(X - a_{2i+1}) = (X - Re a_{2i})^2 + (Im a_{2i})^2`.
struct Pair {
  double re = 0.0;
  double im = 0.0;
};

//! Sort the roots into conjugate pairs. Real roots of `P + m` come in even
//! multiplicities (lemma 1's proof says so, and the epsilon below makes them
//! strictly complex in practice), so any left over are averaged in pairs.
std::vector<Pair> PairRoots(std::vector<Cplx> z) {
  std::vector<Pair> pairs;
  std::vector<bool> used(z.size(), false);
  for (size_t i = 0; i < z.size(); i++) {
    if (used[i]) continue;
    used[i] = true;
    // The partner is whichever unused root is nearest to conj(z_i).
    const Cplx want = std::conj(z[i]);
    size_t best = z.size();
    double bd = 0.0;
    for (size_t j = 0; j < z.size(); j++) {
      if (used[j]) continue;
      const double d = std::abs(z[j] - want);
      if (best == z.size() || d < bd) {
        best = j;
        bd = d;
      }
    }
    if (best == z.size()) {
      // An odd count can only happen if the root finder lost one; treat the
      // leftover as real so the caller's residual check can reject the plan.
      pairs.push_back({z[i].real(), 0.0});
      continue;
    }
    used[best] = true;
    const double re = 0.5 * (z[i].real() + z[best].real());
    const double im = 0.5 * (std::abs(z[i].imag()) + std::abs(z[best].imag()));
    pairs.push_back({re, im});
  }
  return pairs;
}

//! [SYLPH] eq. (2), one step: combine two sums of squares into one, rotated by
//! -pi/4. The rotation is what keeps both halves at full degree.
void CombineEq2(const ChebPoly &u0, const ChebPoly &v0, const ChebPoly &u1,
                const ChebPoly &v1, ChebPoly &u, ChebPoly &v) {
  const ChebPoly a = Sub(Mul(u0, u1), Mul(v0, v1));   // U0 U1 - V0 V1
  const ChebPoly b = Add(Mul(u0, v1), Mul(u1, v0));   // U0 V1 + U1 V0
  const double inv = 1.0 / std::sqrt(2.0);
  u = Scale(Add(a, b), inv);
  v = Scale(Sub(a, b), inv);
}

//! Rotate `(u, v)` so both carry the full degree with a positive leading
//! coefficient. `u + i v` is the Gaussian polynomial `W` and `u^2 + v^2` is
//! `|W|^2`, so any global phase is free; this picks the one that puts the
//! leading coefficient's argument at pi/4, which is the exact form of the
//! approximate balance eq. (2) is aiming at.
void NormaliseLeading(ChebPoly &u, ChebPoly &v, int degree) {
  u.resize(degree + 1, 0.0);
  v.resize(degree + 1, 0.0);
  const Cplx lead(u[degree], v[degree]);
  if (std::abs(lead) < kTiny) return;
  const double theta = 0.25 * M_PI - std::arg(lead);
  const double cs = std::cos(theta), sn = std::sin(theta);
  ChebPoly nu = Sub(Scale(u, cs), Scale(v, sn));
  ChebPoly nv = Add(Scale(u, sn), Scale(v, cs));
  u = std::move(nu);
  v = std::move(nv);
}

//! Build `(U, V)` for one choice of conjugate representatives, following
//! [SYLPH] eq. (2) from the first pair through the last.
void BuildFromPairs(const std::vector<Pair> &pairs,
                    const std::vector<char> &flip, double lambda, ChebPoly &u,
                    ChebPoly &v) {
  const double s0 = flip.empty() || !flip[0] ? 1.0 : -1.0;
  u = ChebPoly{-pairs[0].re * lambda, lambda};
  v = ChebPoly{s0 * pairs[0].im * lambda};
  for (size_t i = 1; i < pairs.size(); i++) {
    const double s = (i < flip.size() && flip[i]) ? -1.0 : 1.0;
    const ChebPoly u1{-pairs[i].re * lambda, lambda};
    const ChebPoly v1{s * pairs[i].im * lambda};
    ChebPoly nu, nv;
    CombineEq2(u, v, u1, v1, nu, nv);
    u = std::move(nu);
    v = std::move(nv);
  }
}

}  // namespace

Decomposition Decompose(const ChebPoly &p_in, const DecomposeOptions &opt) {
  Decomposition out;
  ChebPoly p = p_in;
  Trim(p);
  const int d = Degree(p);
  if (d <= 0 || d % 2 != 0) {
    out.why = "lemma 1 needs an even, positive degree";
    return out;
  }
  if (p[d] <= 0.0) {
    out.why = "lemma 1 needs a positive leading coefficient";
    return out;
  }

  // m = max_x(-P(x)) = -min_x P(x), over the WHOLE line: a polynomial that is
  // non-negative only on an interval is not a sum of two squares of half its
  // degree. The epsilon is not slack for its own sake -- it separates the
  // double real root that the exact minimum creates into a conjugate pair,
  // which is the difference between a well-conditioned root finder and one
  // that returns two nearly equal reals it cannot pair.
  const double interval_scale =
      std::max(SupNorm(p, opt.lo, opt.hi), CoeffScale(p));
  const double gmin = GlobalMin(p);
  const double m = -gmin + 1e-9 * interval_scale;

  ChebPoly q = p;
  q[0] += m;  // T_0 is the constant, so this is exactly `P + m`
  double root_residual = 0.0;
  std::vector<Cplx> z = Roots(q, &root_residual);
  if (static_cast<int>(z.size()) != d) {
    out.why = "the root finder did not return deg(P) roots";
    return out;
  }
  if (root_residual > opt.root_residual_max) {
    out.why = "the roots do not reproduce P + m (residual " +
              std::to_string(root_residual) + ")";
    return out;
  }
  const std::vector<Pair> pairs = PairRoots(z);
  if (static_cast<int>(pairs.size()) * 2 != d) {
    out.why = "the roots did not pair into conjugates";
    return out;
  }

  // `q = p_0 prod (X - a_i)` with `p_0` the MONOMIAL leading coefficient, and
  // T_d leads with 2^(d-1). Spreading `sqrt(p_0)` evenly over the pair factors
  // instead of applying it at the end keeps every intermediate O(1): the monic
  // product's coefficients are around 2^-d, and sqrt(p_0) is around 2^(d/2).
  const double p0 = q[d] * std::pow(2.0, d - 1);
  if (!(p0 > 0.0)) {
    out.why = "P + m has a non-positive leading coefficient";
    return out;
  }
  const double lambda = std::pow(p0, 1.0 / static_cast<double>(d));

  // Appendix D's objective, both halves. The first is the visible one -- the
  // loop costs about `1 + log2(|U| + |V|)` bits, so small on the interval is
  // better. The second is the one that actually decides a deep tree: a child's
  // own `m` is `-min U` over the WHOLE line, and a degree-16 polynomial that
  // is O(1) on [-1, 1] can dive to -1e5 just outside it. Appendix D asks for
  // exactly this lookahead ("in search of U^(l-1)_i with a small associated
  // m^(l-1)_i"); without it the search optimises the half that is already
  // small and lets the half that is not run away.
  auto child_m = [&](const ChebPoly &c) {
    ChebPoly t = c;
    Trim(t);
    const int td = Degree(t);
    if (td <= 0 || td % 2 != 0 || t[td] <= 0.0) return 0.0;
    return std::max(0.0, -GlobalMin(t));
  };
  auto score = [&](const ChebPoly &u, const ChebPoly &v) {
    const double sup = std::max(SupNorm(u, opt.lo, opt.hi, 513),
                                SupNorm(v, opt.lo, opt.hi, 513));
    double s = std::log2(1.0 + sup);
    if (opt.lookahead && d / 2 >= 2) {
      const double mm = std::max(child_m(u), child_m(v));
      s += opt.lookahead_weight * std::log2(1.0 + mm);
    }
    return s;
  };

  const size_t np = pairs.size();
  std::vector<char> flip(np, 0);
  ChebPoly bu, bv;
  BuildFromPairs(pairs, flip, lambda, bu, bv);
  NormaliseLeading(bu, bv, d / 2);
  double best = score(bu, bv);

  if (opt.search > 0 && np > 1) {
    std::mt19937 rng(opt.seed);
    std::vector<char> cand(np, 0);
    for (int t = 0; t < opt.search; t++) {
      for (size_t i = 0; i < np; i++) cand[i] = (rng() & 1u) ? 1 : 0;
      ChebPoly u, v;
      BuildFromPairs(pairs, cand, lambda, u, v);
      NormaliseLeading(u, v, d / 2);
      const double s = score(u, v);
      if (s < best) {
        best = s;
        bu = std::move(u);
        bv = std::move(v);
      }
    }
  }

  out.u = std::move(bu);
  out.v = std::move(bv);
  out.m = m;
  out.magnitude = best;

  // The identity itself, on the interval it will be evaluated over. Cheap, and
  // the only thing that can catch a pairing that went wrong in a way the root
  // residual did not see.
  const ChebPoly check = Sub(Add(Mul(out.u, out.u), Mul(out.v, out.v)), p);
  double worst = 0.0;
  constexpr int kGrid = 2049;
  for (int i = 0; i < kGrid; i++) {
    const double x = opt.lo + (opt.hi - opt.lo) * i / (kGrid - 1);
    worst = std::max(worst, std::abs(Eval(check, x) - m));
  }
  out.residual = worst;
  out.ok = true;
  return out;
}

double SlimPlan::PlainEvaluate(double x) const {
  const int blocks = 1 << j;
  std::vector<double> val(blocks, 0.0);
  for (int i = 0; i < blocks; i++) {
    val[i] = Eval(leaf[i % static_cast<int>(leaf.size())], x);
  }
  // Algorithm 1's loop, block for block: square, add the rotation by
  // `2^(l-1)` blocks, subtract `m^(l-1)`.
  for (int l = j; l >= 1; l--) {
    const int shift = 1 << (l - 1);
    std::vector<double> next(blocks, 0.0);
    for (int i = 0; i < blocks; i++) {
      const double a = val[i];
      const double b = val[(i + shift) % blocks];
      next[i] = a * a + b * b - m[l - 1][i % shift];
    }
    val = std::move(next);
  }
  return val[0];
}

SlimPlan BuildPlan(const ChebPoly &p, int j, const DecomposeOptions &opt) {
  SlimPlan plan;
  ChebPoly root = p;
  Trim(root);
  int d = Degree(root);
  if (d <= 0) {
    plan.why = "slim needs a polynomial of positive degree";
    return plan;
  }
  // Lemma 1 needs a POSITIVE leading coefficient; with a negative one
  // `max_x(-P(x))` is infinite and there is no decomposition at any depth.
  // Padding cannot rescue that -- adding a tiny leading term of the other sign
  // only moves the divergence further out and makes `m` astronomical -- so the
  // fix is to decompose `-P` and tell the caller to negate. For the softmax's
  // auxiliary track that negation is free: it rides the multiply that consumes
  // the inverse square root.
  if (root[d] < 0.0) {
    plan.negated = true;
    root = Scale(root, -1.0);
  }
  int k = 0;
  while ((1 << k) < d) k++;
  if ((1 << k) != d) {
    // A fit whose natural degree is short -- the inverse square root on
    // [0.9, 1.1] interpolated at 16 trims to 10, because the tail is round-off
    // -- has no degree-2^k term for the recursion to split. Give it one, and
    // report how big the addition was: the plan then evaluates a polynomial
    // the caller did not hand in, and the size of that substitution belongs in
    // the plan rather than in a comment.
    if (opt.pad_leading <= 0.0) {
      plan.why = "slim needs a degree that is exactly a power of two (got " +
                 std::to_string(d) + "); pad_leading is off";
      return plan;
    }
    plan.padding = opt.pad_leading * std::max(SupNorm(root, opt.lo, opt.hi),
                                              1e-300);
    root.resize((1 << k) + 1, 0.0);
    root[1 << k] = plan.padding;
    d = 1 << k;
  }
  plan.k = k;
  plan.j = j;
  if (j < 0 || j > k) {
    plan.why = "the recursion depth must satisfy 0 <= j <= k";
    return plan;
  }

  std::vector<ChebPoly> level{root};
  plan.m.assign(j, {});
  for (int l = 1; l <= j; l++) {
    const int half = 1 << (l - 1);
    std::vector<ChebPoly> next(1 << l);
    plan.m[l - 1].assign(half, 0.0);
    for (int i = 0; i < half; i++) {
      const Decomposition dec = Decompose(level[i], opt);
      if (!dec.ok) {
        plan.why = "level " + std::to_string(l) + " node " +
                   std::to_string(i) + ": " + dec.why;
        return plan;
      }
      // Eq. (3): the children of `i` sit at `i` and `i + 2^(l-1)`.
      next[i] = dec.u;
      next[i + half] = dec.v;
      plan.m[l - 1][i] = dec.m;
      plan.worst_magnitude = std::max(plan.worst_magnitude, dec.magnitude);
      plan.worst_m = std::max(plan.worst_m, std::abs(dec.m));
    }
    level = std::move(next);
  }
  plan.leaf = std::move(level);

  // The plan's own loop against the polynomial it claims to evaluate. This is
  // the number a caller should read; the per-node residuals cannot see the
  // cancellation the loop accumulates.
  double worst = 0.0;
  constexpr int kGrid = 4097;
  for (int i = 0; i < kGrid; i++) {
    const double x = opt.lo + (opt.hi - opt.lo) * i / (kGrid - 1);
    worst = std::max(worst, std::abs(plan.PlainEvaluate(x) - Eval(root, x)));
  }
  plan.residual = worst;
  // What the identity costs in dynamic range: the ciphertext has to carry
  // `U^2 + V^2 = P + m` where the answer is only as big as `P`.
  const double sup_p = std::max(SupNorm(root, opt.lo, opt.hi), 1e-300);
  plan.range_bits = std::log2((sup_p + plan.worst_m) / sup_p);
  plan.ok = true;
  return plan;
}

SlimBudget Budget(int levels, int j, bool fold_leading) {
  SlimBudget b;
  b.levels = levels;
  const int k = fold_leading ? levels : levels - 1;
  b.slim_degree = (k >= 0) ? (1 << k) : 0;
  b.paterson_degree = (levels >= 0) ? ((1 << levels) - 1) : 0;
  j = std::max(0, std::min(j, k));
  // Paterson-Stockmeyer on degree D is about 2 sqrt(D) ciphertext-ciphertext
  // multiplications (the baby steps are the giant steps' square root); slim
  // pays that on the leaf degree 2^(k-j) and then j more.
  auto ps_mults = [](int degree) {
    if (degree <= 1) return 0;
    return static_cast<int>(std::lround(2.0 * std::sqrt(
        static_cast<double>(degree))));
  };
  b.slim_mults = ps_mults(1 << (k - j)) + j;
  b.paterson_mults = ps_mults(b.paterson_degree);
  b.slim_rotations = j;
  return b;
}

}  // namespace slim
}  // namespace cheddar
