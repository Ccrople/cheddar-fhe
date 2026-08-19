#pragma once

#include <cmath>
#include <vector>

namespace cheddar {
namespace chebfit {

/**
 * @brief Chebyshev interpolation of f on [-1, 1] at degree n, in the Chebyshev
 * basis.
 *
 * Interpolating at the Chebyshev nodes rather than fitting in the monomial
 * basis is not a stylistic choice. Monomial coefficients of these functions
 * span many orders of magnitude on the intervals they are approximated over,
 * and EvalPoly silently drops any coefficient below 1e-9 (Doing.md 1.4), so a
 * monomial fit loses terms without saying so.
 *
 * The convention returned is f ~ sum_k c_k T_k, with the halving already
 * folded into c_0. Cheddar's EvalPoly with chebyshev = true takes exactly
 * this; it was checked against a plaintext evaluation rather than assumed.
 */
template <typename F>
std::vector<double> Interpolate(F f, int n) {
  const int m = n + 1;
  std::vector<double> node(m), value(m);
  for (int j = 0; j < m; j++) {
    node[j] = std::cos(M_PI * (j + 0.5) / m);
    value[j] = f(node[j]);
  }
  std::vector<double> c(m, 0.0);
  for (int k = 0; k < m; k++) {
    double acc = 0.0;
    for (int j = 0; j < m; j++) {
      acc += value[j] * std::cos(M_PI * k * (j + 0.5) / m);
    }
    c[k] = (k == 0 ? 1.0 : 2.0) * acc / m;
  }
  return c;
}

}  // namespace chebfit
}  // namespace cheddar
