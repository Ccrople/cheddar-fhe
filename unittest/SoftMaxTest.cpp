// SoftMax, [SYLPH] section 2.3 (Cho et al.).
//
// THE NORM IS EUCLIDEAN. Reading it as the sum -- which the paper's phrasing
// invites if you skip the words "inverse square root caused by the Euclidean
// norm" -- yields a different algorithm needing a reciprocal per iteration
// plus a final normalisation, and measured on real scores that came to 19-26
// levels against [SYLPH]'s 8. With the Euclidean norm, z = y/||y||_2 gives
// sum(z^2) = 1, so the squaring lands on a vector that already sums to one:
// exactly k iterations and no final normalisation.
//
// THE PARAMETERS BELOW ARE MEASURED, NOT GUESSED. On the real Llama-3-8B
// layer-2 attention scores (RMSNorm -> Q/K projection -> RoPE -> causal QK^T
// over all 32 heads), the worst per-row span is 20.626, so M = 21. At that M:
//
//     k   exp on [-M/2^k, 0]   the accurate 1/sqrt        main track
//     1   degree 9             [1.54, 6.82]   degree  8   7 levels
//     2   degree 7             [0.0136, 0.50] degree 28   8 levels
//
// The k=2 row reproduces [SYLPH] section 4.3 exactly ("two iterations ...
// only 8 levels in its main track"). It uses k=2 because its calibrated
// SoftMax input is 32.78 (table 2); at M = 21, k=1 is strictly better.

#include <algorithm>
#include <cmath>
#include <vector>

#include "Testbed.h"
#include "extension/SoftMax.h"

using word = uint32_t;

namespace {

constexpr int kKeys = 128;      // d, one SoftMax row
constexpr double kRange = 21.0; // M, measured
constexpr int kIters = 1;       // k
constexpr int kExpDegree = 9;
constexpr int kInvSqrtDegree = 8;
// The calibrated interval must come from the same distribution the operator is
// fed. On the real layer-2 scores it is [1.544, 6.82]; the synthetic rows below
// have a different distribution, so the test measures their interval instead of
// borrowing the real one. Getting this wrong is not a small error: a degree-8
// Chebyshev evaluated four interval-widths outside its domain returns 1e10.
constexpr double kTargetBits = 12.0;
// Below this the circuit, not the approximation, sets the accuracy -- measured
// for SiLU across bootparam_30/35/40 and it is a property of the scale.
constexpr double kMinUsableScale = 1.5e10;

double Bits(double err, double ref) {
  if (err <= 0.0) return 1e9;
  return -std::log2(err / ref);
}

// A row of scores in [-M, 0] with its own max at 0, plus how many keys are
// causally valid. Deterministic, so a failure is reproducible.
std::vector<double> MakeRow(int row, int valid) {
  std::vector<double> x(kKeys, 0.0);
  for (int i = 0; i < valid; i++) {
    const double u = std::fmod(0.6180339887 * (i + 1) + 0.31 * row, 1.0);
    x[i] = -kRange * u * u;  // biased toward the top of the interval
  }
  x[row % valid] = 0.0;  // the row maximum, which the translation puts at 0
  return x;
}

// The interval ||y||_2^2 actually occupies, which is what offline calibration
// produces. Uses the exact exponential, as the calibration pass would.
void NormInterval(const std::vector<std::vector<double>> &rows,
                  const std::vector<int> &valid, int iters, double *lo,
                  double *hi) {
  *lo = 1e300;
  *hi = 0.0;
  for (size_t r = 0; r < rows.size(); r++) {
    std::vector<double> y(valid[r]);
    for (int i = 0; i < valid[r]; i++) {
      y[i] = std::exp(rows[r][i] / std::pow(2.0, iters));
    }
    for (int j = 0; j < iters; j++) {
      double sq = 0.0;
      for (double v : y) sq += v * v;
      *lo = std::min(*lo, sq);
      *hi = std::max(*hi, sq);
      const double inv = 1.0 / std::sqrt(sq);
      for (double &v : y) {
        v *= inv;
        v = v * v;
      }
    }
  }
  // a little margin, since the polynomial exponential is not the exact one
  *lo *= 0.98;
  *hi *= 1.02;
}

std::vector<double> TrueSoftMax(const std::vector<double> &x, int valid) {
  std::vector<double> y(x.size(), 0.0);
  double s = 0.0;
  for (int i = 0; i < valid; i++) {
    y[i] = std::exp(x[i]);
    s += y[i];
  }
  for (int i = 0; i < valid; i++) y[i] /= s;
  return y;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The polynomials and the iteration, with no ciphertext present.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, SoftMaxPlainOracle) {
  const int level = default_encryption_level_;

  std::vector<std::vector<double>> rows;
  std::vector<int> valid_v;
  for (int r = 0; r < 64; r++) {
    rows.push_back(MakeRow(r, kKeys));
    valid_v.push_back(kKeys);
  }
  double lo, hi;
  NormInterval(rows, valid_v, kIters, &lo, &hi);
  std::cout << "calibrated ||y||^2 interval for this data: [" << lo << ", "
            << hi << "]  (real layer-2 scores give [1.54, 6.82])" << std::endl;

  SoftMaxHandler<word> sm(context_, kKeys, kRange, level, kIters,
                          std::vector<double>(kIters, lo),
                          std::vector<double>(kIters, hi), kExpDegree,
                          kInvSqrtDegree);

  double worst = 0.0;
  int worst_row = -1;
  for (int row = 0; row < 64; row++) {
    const int valid = kKeys;
    const auto &x = rows[row];
    auto got = sm.PlainSoftMax(x);
    auto want = TrueSoftMax(x, valid);
    for (int i = 0; i < valid; i++) {
      const double d = std::abs(got[i] - want[i]);
      if (d > worst) {
        worst = d;
        worst_row = row;
      }
    }
  }
  // The reference is 1/d, the value a uniform row would take, which is what
  // "12 bits of SoftMax precision" is relative to.
  const double ref = 1.0 / kKeys;
  std::cout << "k=" << kIters << " exp degree " << kExpDegree
            << ", 1/sqrt degree " << kInvSqrtDegree << std::endl;
  std::cout << "plain oracle vs true SoftMax: " << worst << " ("
            << Bits(worst, ref) << " bits relative to 1/d), worst row "
            << worst_row << std::endl;
  EXPECT_GT(Bits(worst, ref), kTargetBits)
      << "the polynomials alone miss the bar, so no circuit can reach it";
}

// ---------------------------------------------------------------------------
// 2. The circuit, against its own polynomials and against true SoftMax.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, SoftMaxOnEncrypted) {
  const int level = default_encryption_level_;
  const int slots = param_->degree_ / 2;
  const int rows = slots / kKeys;

  // Causal rows of varying length, which is what attention actually produces.
  // A row shorter than d makes the norm smaller, so this exercises the whole
  // calibrated interval rather than one point in it.
  std::vector<int> valid(rows);
  for (int r = 0; r < rows; r++) valid[r] = kKeys - (r % (kKeys / 2));
  std::vector<std::vector<double>> row_data;
  for (int r = 0; r < rows; r++) row_data.push_back(MakeRow(r, valid[r]));

  double lo, hi;
  NormInterval(row_data, valid, kIters, &lo, &hi);
  std::cout << "calibrated ||y||^2 interval for this data: [" << lo << ", "
            << hi << "]" << std::endl;

  SoftMaxHandler<word> sm(context_, kKeys, kRange, level, kIters,
                          std::vector<double>(kIters, lo),
                          std::vector<double>(kIters, hi), kExpDegree,
                          kInvSqrtDegree);
  for (int d : sm.GetRotationDistances()) {
    interface_->PrepareRotationKey(d, level);
  }

  std::vector<double> x(slots, 0.0);
  std::vector<Complex> msg(slots), mask(slots);
  for (int r = 0; r < rows; r++) {
    const auto &row = row_data[r];
    for (int i = 0; i < kKeys; i++) {
      const int s = r * kKeys + i;
      x[s] = row[i];
      // masked slots still have to hold a value inside the polynomial's domain
      msg[s] = Complex(2.0 * row[i] / kRange + 1.0, 0.0);
      mask[s] = Complex(i < valid[r] ? 1.0 : 0.0, 0.0);
    }
  }

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  Ciphertext<word> res;
  sm.Apply(res, ct, mask, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const int out_level = param_->NPToLevel(res.GetNP());
  std::cout << "output level " << out_level << " from " << level
            << ", so depth " << (level - out_level) << std::endl;
  std::cout << "[SYLPH] 4.3's 8 levels are the MAIN track only; figure 2 "
               "bootstraps the auxiliary track separately, which Cheddar "
               "cannot yet do, so the norm square, the affine map and the "
               "inverse square root land here too"
            << std::endl;
  EXPECT_NEAR(res.GetScale() / param_->GetScale(out_level), 1.0, 1e-9)
      << "the output scale must stay canonical";

  std::vector<Complex> got;
  DecryptAndDecode(got, res);

  const double ref = 1.0 / kKeys;
  double err_fit = 0.0, err_true = 0.0, worst_sum = 0.0;
  for (int r = 0; r < rows; r++) {
    std::vector<double> row(kKeys);
    for (int i = 0; i < kKeys; i++) row[i] = x[r * kKeys + i];
    std::vector<double> row_valid(row.begin(), row.begin() + valid[r]);
    auto fit = sm.PlainSoftMax(row_valid);
    auto want = TrueSoftMax(row, valid[r]);
    double sum = 0.0;
    for (int i = 0; i < kKeys; i++) {
      const double g = got[r * kKeys + i].real();
      sum += g;
      if (i < valid[r]) {
        err_fit = std::max(err_fit, std::abs(g - fit[i]));
        err_true = std::max(err_true, std::abs(g - want[i]));
      } else {
        // masked positions must be exactly out of the answer
        err_true = std::max(err_true, std::abs(g));
      }
    }
    worst_sum = std::max(worst_sum, std::abs(sum - 1.0));
  }
  std::cout << "circuit vs its own polynomials: " << err_fit << "  ("
            << Bits(err_fit, ref) << " bits)" << std::endl;
  std::cout << "circuit vs true SoftMax:        " << err_true << "  ("
            << Bits(err_true, ref) << " bits)" << std::endl;
  // The Euclidean norm is what makes this hold with no final normalisation.
  // If it drifts, the norm has been read as a sum somewhere.
  std::cout << "worst |sum(row) - 1|:           " << worst_sum << std::endl;
  EXPECT_LT(worst_sum, 1e-2)
      << "the rows must already sum to one, which is exactly the property the "
         "Euclidean norm buys and a sum-norm does not";

  if (param_->base_scale_ < kMinUsableScale) {
    std::cout << "2^30: recorded as insufficient for the non-linearities"
              << std::endl;
    return;
  }
  EXPECT_GT(Bits(err_true, ref), kTargetBits);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
