#pragma once

#include <complex>
#include <string>
#include <vector>

namespace cheddar {

/**
 * @brief [SYLPH] section 3.4 and Algorithm 1: *slim* polynomial evaluation.
 *
 * ## What it is, and what it is not
 *
 * A degree-`d` polynomial evaluated on a ciphertext whose message only needs
 * `2^t <= N / (2 d)` slots. The remaining slots are not padding -- they carry
 * **different polynomials of the decomposition tree**, evaluated in parallel
 * by the same SIMD instructions, and folded back together by `j` squarings and
 * `j` rotations.
 *
 * By [SYLPH] theorem 1 (appendix D) it is **not a level optimisation**:
 * degree `2^k` costs `k + 1` levels, which is what Paterson-Stockmeyer spends
 * on degree `2^(k+1) - 1` -- twice the degree for the same budget -- and `k`
 * levels if the leading plaintext coefficient is folded into a preceding
 * multiplication, which is the case where the two are even. What it buys is
 *
 *     O(2^((k-j)/2)) + j   ciphertext-ciphertext multiplications, and
 *     j                    rotations
 *
 * against Paterson-Stockmeyer's `O(2^(k/2))`. With `j = k/2` that is
 * `O(2^(k/4))`, so the win is in **key switchings**, which is what a
 * sparsely-packed auxiliary track is bound by.
 *
 * The corollary matters for calibration: at a fixed level budget `L`,
 * Paterson-Stockmeyer buys degree `2^L - 1` and slim buys degree `2^(L-1)`
 * (or `2^L` with the fold). **Slim is chosen for speed, never for accuracy**;
 * on a window where the fit is the limiter, plain `EvalPoly` at the same
 * levels is strictly better. See `SlimBudget` below, which states that
 * trade in one place instead of leaving it to a caller's arithmetic.
 *
 * ## The decomposition (lemma 1)
 *
 * For `P` of even degree with positive leading coefficient there are `U`, `V`
 * of degree `<= deg(P)/2` and a scalar `m` with
 *
 *     P(x) = U(x)^2 + V(x)^2 - m .
 *
 * `m = max_x(-P(x)) = -min_x P(x)` over the whole real line makes `P + m`
 * non-negative on **R**, which is what makes it a sum of two squares at all.
 * A polynomial non-negative only on an interval has no such certificate at
 * this degree, so `m` is a global quantity even though the fit is not, and
 * `m` is the number that decides whether slim is usable: `U^2 + V^2` reaches
 * `max_I P + m`, so the evaluation carries `log2(1 + m / max_I |P|)` bits of
 * cancellation that the direct evaluation does not. `Decomposition::m` is
 * reported for exactly this reason -- measure it before choosing slim.
 *
 * The construction is [SYLPH]'s, verbatim: write `P + m = p_0 prod (X - a_i)`,
 * order the roots in conjugate pairs, turn each pair into
 * `(X - Re a)^2 + (Im a)^2`, and combine pairs with
 *
 *     (U0^2 + V0^2)(U1^2 + V1^2)
 *       = ((U0 U1 - V0 V1 + U0 V1 + U1 V0) / sqrt 2)^2
 *       + ((U0 U1 - V0 V1 - U0 V1 - U1 V0) / sqrt 2)^2 .          (eq. 2)
 *
 * Equivalently `W = sqrt(p_0) prod_{a in S}(X - a)` over one root of each
 * conjugate pair, `U = Re W`, `V = Im W`, `P + m = |W|^2` on the real line.
 * **The `1/sqrt 2` is not cosmetic**: it is a rotation by `-pi/4` of the
 * Gaussian polynomial `U + iV`, and it is what keeps both `U` and `V` at the
 * full degree `deg(P)/2` with positive leading coefficients. Without it `V`
 * would lose its leading term and the recursion could not continue. That is
 * why this file rotates at every combination step rather than only at the end.
 *
 * ## The tree (eq. 3) and Algorithm 1
 *
 * Apply the lemma recursively. `U^(0)_0 = P`; level `l` holds
 * `U^(l)_0 .. U^(l)_{2^l - 1}` of degree `2^(k-l)` with
 *
 *     (U^(l)_i)^2 + (U^(l)_{i + 2^(l-1)})^2 = U^(l-1)_i + m^(l-1)_i .   (eq. 3)
 *
 * The child indices are `i` and `i + 2^(l-1)` -- half the level's width apart,
 * which is what makes the fold a single rotation. Cut the tree at level `j`
 * and lay the `2^j` leaves out in blocks of `2^t` slots, block `i` carrying
 * `U^(j)_{i mod 2^j}`. Then, writing `B = 2^t` for the block:
 *
 *     ct <- Evaluate(ct, v^(0), ..., v^(2^(k-j)))     Paterson-Stockmeyer,
 *                                                     plaintext coefficients
 *     for l = j down to 1:
 *       ct <- ct * ct
 *       ct <- ct + Rot(ct, B * 2^(l-1)) - m^(l-1)
 *
 * The invariant is worth stating because it is the whole proof: **before
 * iteration `l`, block `i` holds `U^(l)_{i mod 2^l}(x)`; after it, block `i`
 * holds `U^(l-1)_{i mod 2^(l-1)}(x)`.** At `l = 1` every block holds
 * `U^(0)_0 = P`, so the answer is in the first block and in every other one.
 * `SlimPlan::PlainEvaluate` runs exactly this loop in the clear, which is how
 * the encrypted path is checked.
 *
 * ## Numerical stability, and the search appendix D asks for
 *
 * `|x^2 - y^2| <= 2 max(|x|,|y|) |x - y|`, so each iteration costs about
 * `1 + log2(|U^(l)_i| + |U^(l)_{i + 2^(l-1)}|)` bits in slot `i`. The
 * decomposition is not unique -- taking the other root of a conjugate pair, or
 * rotating `U + iV` by any phase, gives another one -- so appendix D asks for
 * a search over decompositions whose `U` and `V` stay small on the interval,
 * and whose children's `m` stay small. `DecomposeOptions::search` is that
 * search; it is a bounded randomised one with a fixed seed, so a plan is
 * reproducible, and `SlimPlan::worst_magnitude` reports what it achieved.
 *
 * @tparam word uint32_t or uint64_t
 */
namespace slim {

/**
 * @brief A polynomial in the Chebyshev basis of `[-1, 1]`, in `chebfit`'s
 * convention: `p(x) = sum_k c[k] T_k(x)` with the halving already folded into
 * `c[0]`, which is what `EvalPoly(..., chebyshev = true)` consumes.
 *
 * Everything here stays in this basis. Converting to the monomial basis to
 * find roots would be the obvious move and it is the wrong one: at degree 32
 * the conversion alone loses more digits than the root finder could recover,
 * and `EvalPoly` silently drops monomial coefficients below `1e-9`
 * (`ChebyshevFit.h`). Clenshaw evaluation, the product rule
 * `T_a T_b = (T_{a+b} + T_{|a-b|}) / 2` and an Aberth iteration driven by
 * Clenshaw all work directly on this representation.
 */
using ChebPoly = std::vector<double>;

/** @brief The degree, ignoring trailing coefficients that round to zero. */
int Degree(const ChebPoly &c);

/** @brief `p(x)` by Clenshaw. Valid for `|x| > 1` too. */
double Eval(const ChebPoly &c, double x);

/** @brief `p(z)` by Clenshaw at a complex argument, for the root finder. */
std::complex<double> EvalComplex(const ChebPoly &c, const std::complex<double> &z);

/** @brief `a + b`, at the longer length. */
ChebPoly Add(const ChebPoly &a, const ChebPoly &b);

/** @brief `a - b`. */
ChebPoly Sub(const ChebPoly &a, const ChebPoly &b);

/** @brief `s * a`. */
ChebPoly Scale(const ChebPoly &a, double s);

/** @brief `a * b` through `T_a T_b = (T_{a+b} + T_{|a-b|}) / 2`. */
ChebPoly Mul(const ChebPoly &a, const ChebPoly &b);

/** @brief `p'`, by the standard Chebyshev derivative recurrence. */
ChebPoly Deriv(const ChebPoly &c);

/** @brief `max |p|` over `[lo, hi]`, sampled; the interval is the fit's. */
double SupNorm(const ChebPoly &c, double lo, double hi, int grid = 4001);

/**
 * @brief Every root of `p`, by an Aberth-Ehrlich iteration on the Chebyshev
 * series.
 *
 * Reported through `residual`: the largest `|p(root)|` relative to `p`'s
 * coefficient scale. A decomposition built on roots that do not reproduce the
 * polynomial is worse than no decomposition, and this is the only place that
 * can notice, so callers are expected to check it rather than trust it.
 */
std::vector<std::complex<double>> Roots(const ChebPoly &c, double *residual = nullptr);

/**
 * @brief `min p` over the whole real line, which is `-m` of lemma 1.
 *
 * Global, not over the fit interval: see the header. Requires even degree and
 * a positive leading coefficient, without which the minimum is `-infinity`.
 */
double GlobalMin(const ChebPoly &c);

/** @brief Knobs for the appendix D search. */
struct DecomposeOptions {
  double lo = -1.0;  //!< the interval the fit lives on, for the objective
  double hi = 1.0;
  //! Candidate decompositions to try beyond the canonical one. 0 = take the
  //! canonical pairing and the -pi/4 rotation, which is [SYLPH] eq. (2) read
  //! literally. Appendix D's search is what the rest buys.
  int search = 64;
  unsigned seed = 0x5115u;  //!< fixed, so a plan is reproducible
  //! Refuse a decomposition whose roots do not reproduce the polynomial.
  double root_residual_max = 1e-6;
  //! Appendix D asks for more than a small `U` and `V` on the interval: it
  //! asks to explore "the set of possible decompositions at step l-1 in search
  //! of U^(l-1)_i with a small associated m^(l-1)_i", i.e. to score a
  //! candidate by the `m` its CHILDREN will need. That is a global minimum per
  //! candidate and costs a root find each, so it is a switch; without it the
  //! objective is only the interval sup norm, which is the visible half of
  //! appendix D and not the half that decides whether a deep tree is usable.
  bool lookahead = true;
  //! Weight of the children's `m` against the interval sup norm in that
  //! objective. Both enter as `log2(1 + .)`, so the sum is in bits.
  double lookahead_weight = 1.0;
  //! A fit whose natural degree is below `2^k` has no leading term for lemma 1
  //! to work with. Padding adds `pad * scale * T_{2^k}`, perturbing the
  //! polynomial by exactly that much; it is reported in `SlimPlan::padding`.
  //! 0 disables it, and a short polynomial is then refused rather than
  //! silently changed.
  double pad_leading = 1e-9;
};

/** @brief `P = u^2 + v^2 - m`, lemma 1. */
struct Decomposition {
  ChebPoly u, v;
  double m = 0.0;
  //! `max(|u|, |v|)` over the interval -- appendix D's stability objective.
  double magnitude = 0.0;
  //! `max |u^2 + v^2 - m - P|` over the interval; the identity's own residual.
  double residual = 0.0;
  bool ok = false;
  std::string why;  //!< why not, when `ok` is false
};

/** @brief Lemma 1 on one polynomial of even degree, positive leading term. */
Decomposition Decompose(const ChebPoly &p, const DecomposeOptions &opt);

/**
 * @brief The tree of eq. (3) cut at level `j`, ready to become plaintexts.
 *
 * `leaf[i]` is `U^(j)_i` for `0 <= i < 2^j`, each of degree `2^(k-j)`.
 * `m[l][i]` is `m^(l)_i` for `0 <= l < j` and `0 <= i < 2^l`, consumed by the
 * iteration at `l + 1`.
 */
struct SlimPlan {
  int k = 0;  //!< the polynomial's degree is `2^k`
  int j = 0;  //!< recursion depth; `2^j` leaves, `j` squarings and rotations
  std::vector<ChebPoly> leaf;
  std::vector<std::vector<double>> m;
  //! The largest `|U^(l)_i|` over the interval anywhere in the tree, and the
  //! largest `|m|`: appendix D's two stability numbers, reported rather than
  //! assumed. `worst_magnitude` drives the `1 + log2(...)` bit loss.
  double worst_magnitude = 0.0;
  double worst_m = 0.0;
  //! `max |PlainEvaluate(x) - P(x)|` over the interval, from the plan's own
  //! loop. This is the number that says whether the plan is usable.
  double residual = 0.0;
  //! How much `T_{2^k}` had to be added to give the fit a leading term, 0 when
  //! it already had one. The plan evaluates the PADDED polynomial, so this is
  //! an error the caller inherits.
  double padding = 0.0;
  //! Lemma 1 has no decomposition for a negative leading coefficient, so the
  //! plan is built for `-P` and this says so: the caller must negate the
  //! result. In the softmax's auxiliary track that is free, riding the
  //! multiply that consumes the inverse square root.
  bool negated = false;
  //! Bits of dynamic range the identity spends that a direct evaluation does
  //! not: `U^2 + V^2` reaches `max_I P + m` where the answer is `max_I |P|`,
  //! so a ciphertext carrying it has `log2((max_I P + m) / max_I |P|)` fewer
  //! bits for the answer. THIS is the number that decides whether a depth is
  //! usable, and it is why `worst_m` is reported beside it.
  double range_bits = 0.0;
  bool ok = false;
  std::string why;

  /** @brief Levels Algorithm 1 spends: `k + 1`, or `k` with the fold. */
  int NumLevels(bool fold_leading = false) const {
    return fold_leading ? k : k + 1;
  }
  /** @brief Slots one block occupies must satisfy `2^t * 2^j <= num_slots`. */
  int NumBlocks() const { return 1 << j; }

  /** @brief Algorithm 1 in the clear, block by block, exactly as it runs. */
  double PlainEvaluate(double x) const;
};

/**
 * @brief Build the tree for `p` (degree `2^k`) cut at depth `j`.
 *
 * `j = 0` is the degenerate case and returns `p` itself as the single leaf,
 * which makes `SlimPolyHandler` reduce to an ordinary evaluation and gives
 * every test a baseline that cannot be wrong.
 */
SlimPlan BuildPlan(const ChebPoly &p, int j, const DecomposeOptions &opt);

/**
 * @brief What a level budget buys, slim against Paterson-Stockmeyer.
 *
 * Stated once, here, because the comparison is the thing callers get wrong:
 * both spend the same levels, and slim's degree is the smaller one.
 */
struct SlimBudget {
  int levels = 0;
  int slim_degree = 0;   //!< `2^levels` with the fold, `2^(levels-1)` without
  int paterson_degree = 0;  //!< `2^levels - 1`
  int slim_mults = 0;    //!< `O(2^((k-j)/2)) + j`
  int paterson_mults = 0;
  int slim_rotations = 0;
};

/** @brief The table above for one budget and one recursion depth. */
SlimBudget Budget(int levels, int j, bool fold_leading = false);

}  // namespace slim

}  // namespace cheddar
