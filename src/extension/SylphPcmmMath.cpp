#include "extension/SylphPcmmMath.h"

#include <cmath>
#include <string>

namespace cheddar {
namespace sylph_pcmm {

namespace {

inline int Mod(int x, int d) {
  x %= d;
  return (x < 0) ? x + d : x;
}

}  // namespace

Mat Sigma(const Mat &a, int d) {
  Mat r(static_cast<size_t>(d) * d, 0.0);
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) r[i * d + j] = a[i * d + Mod(i + j, d)];
  }
  return r;
}

Mat Tau(const Mat &a, int d) {
  Mat r(static_cast<size_t>(d) * d, 0.0);
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) r[i * d + j] = a[Mod(i + j, d) * d + j];
  }
  return r;
}

Mat TauPow(const Mat &a, int d, int n) {
  Mat r = a;
  for (int t = 0; t < n; t++) r = Tau(r, d);
  return r;
}

Mat RotR(const Mat &a, int d, int k) {
  Mat r(static_cast<size_t>(d) * d, 0.0);
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) r[i * d + j] = a[Mod(i + k, d) * d + j];
  }
  return r;
}

Mat RotC(const Mat &a, int d, int k) {
  Mat r(static_cast<size_t>(d) * d, 0.0);
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) r[i * d + j] = a[i * d + Mod(j + k, d)];
  }
  return r;
}

Mat MatMul(const Mat &a, const Mat &b, int d) {
  Mat c(static_cast<size_t>(d) * d, 0.0);
  for (int i = 0; i < d; i++) {
    for (int k = 0; k < d; k++) {
      const double av = a[i * d + k];
      if (av == 0.0) continue;
      for (int j = 0; j < d; j++) c[i * d + j] += av * b[k * d + j];
    }
  }
  return c;
}

std::vector<int> SylphPcmmPlan::RotationDistances() const {
  std::vector<int> v;
  // The inner loop rotates the ciphertext by one row at a time; the outer one
  // by `b` rows. Both are `rot_R`, so both are slot rotations by a multiple of
  // `d` -- there is no column rotation on a ciphertext anywhere in Eq. (5),
  // which is the level that was saved.
  for (int i = 1; i < b; i++) v.push_back(i * d);
  for (int j = 1; j < g; j++) v.push_back(j * b * d);
  return v;
}

SylphPcmmPlan BuildPlan(const Mat &a, int d, int tau_power, int b) {
  SylphPcmmPlan plan;
  plan.d = d;
  plan.tau_power = tau_power;
  if (d <= 0 || a.size() != static_cast<size_t>(d) * d) {
    plan.why = "the matrix is not d x d";
    return plan;
  }
  if (tau_power < 0) {
    plan.why = "tau_power must be non-negative";
    return plan;
  }
  if (b <= 0) {
    // The BSGS optimum is sqrt(d); walk outward to the nearest divisor so that
    // `g b = d` exactly, which Eq. (5) needs -- a ragged split would leave the
    // last giant step short and the index algebra does not allow for that.
    const int root = static_cast<int>(std::lround(std::sqrt(
        static_cast<double>(d))));
    for (int off = 0; off <= d; off++) {
      if (root - off >= 1 && d % (root - off) == 0) {
        b = root - off;
        break;
      }
      if (root + off <= d && d % (root + off) == 0) {
        b = root + off;
        break;
      }
    }
  }
  if (b <= 0 || d % b != 0) {
    plan.why = "the baby-step count must divide d";
    return plan;
  }
  plan.b = b;
  plan.g = d / b;
  // Section 4.2 keeps ONLY this, and rebuilds every `pt_{A,i,j,l}` from it.
  plan.base = TauPow(Sigma(a, d), d, tau_power);
  plan.ok = true;
  return plan;
}

Mat PlaintextFor(const SylphPcmmPlan &plan, int i, int j) {
  const int d = plan.d;
  const int k = i + j * plan.b;
  // `rot_R^(-l k - j b) . rot_C^k (tau^l sigma A)`, two rotations, exactly as
  // section 4.2 says. The row shift carries both the `-l k` that lemma 2
  // produces each time `tau` passes a column rotation and the `-j b` that
  // hoists the giant step out of the inner sum.
  const Mat col = RotC(plan.base, d, k);
  return RotR(col, d, -(plan.tau_power * k + j * plan.b));
}

std::vector<int> TauPermutation(int d, int n, int num_slots) {
  std::vector<int> perm(num_slots);
  for (int s = 0; s < num_slots; s++) perm[s] = s;
  // `(tau^n(A))_{i,j} = A_{i + n j, j}`, so output slot `i d + j` holds input
  // slot `((i + n j) mod d) d + j`. Inverting that for `SlotPermute`'s
  // convention -- output slot `perm[s]` receives input slot `s` -- input slot
  // `i d + j` goes to `((i - n j) mod d) d + j`.
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      perm[i * d + j] = Mod(i - n * j, d) * d + j;
    }
  }
  return perm;
}

Mat PlainApply(const SylphPcmmPlan &plan, const Mat &tau_b) {
  const int d = plan.d;
  Mat res(static_cast<size_t>(d) * d, 0.0);
  for (int j = 0; j < plan.g; j++) {
    Mat inner(static_cast<size_t>(d) * d, 0.0);
    for (int i = 0; i < plan.b; i++) {
      const Mat pt = PlaintextFor(plan, i, j);
      const Mat rb = RotR(tau_b, d, i);
      for (size_t s = 0; s < inner.size(); s++) inner[s] += pt[s] * rb[s];
    }
    const Mat hoisted = RotR(inner, d, j * plan.b);
    for (size_t s = 0; s < res.size(); s++) res[s] += hoisted[s];
  }
  return res;
}

}  // namespace sylph_pcmm
}  // namespace cheddar
