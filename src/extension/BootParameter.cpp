#include "extension/BootParameter.h"

#include <cmath>
#include <cstdlib>

#include "common/CommonUtils.h"

namespace cheddar {

namespace {

// `CHEDDAR_BOOT_DOUBLE_ANGLE=n` widens EvalMod's range: the sine is
// approximated over initial_K * 2^n wrap-arounds, one level per doubling.
// The default 3 (K = 16) is what every measured bootstrap ran with; a
// module-centred ModRaise under a module-sparse secret sees a wrap-around
// std of ~4 against ~2.3 (Doing.md 3.5, check 6) and wants 4.
int DoubleAngleCount() {
  const char *env = std::getenv("CHEDDAR_BOOT_DOUBLE_ANGLE");
  if (env == nullptr || env[0] == 0) return 3;
  const int n = std::atoi(env);
  return (n >= 1 && n <= 6) ? n : 3;
}

// THE COEFFICIENTS ARE TIED TO THE DOUBLE-ANGLE COUNT BY ONE FACTOR. The
// table below is the Chebyshev expansion on [-1, 1] of
//
//     (1 / 2 pi)^(1/8) * cos(4 pi x)
//
// (checked by quadrature to 1e-5, Doing.md 3.9): cos(2 pi initial_K x) with
// the 1 / 2 pi that turns sin(2 pi t / q0) into t / q0 spread over the
// doublings as `EvalMod` does -- (1/2pi)^(1/2^r) on the polynomial and
// (1/2pi)^(2^k/2^r) on the k-th double angle's constant. So the table is
// right for r = 3 and, unscaled, wrong for any other r: at r = 4 the double
// angles expect (1/2pi)^(1/16) in front and get (1/2pi)^(1/8), which is why
// `CHEDDAR_BOOT_DOUBLE_ANGLE=4` used to break the library's own HalfBoot.
// Scaling the table by (2 pi)^(1/8 - 1/2^r) is the whole correction; the
// function under it does not depend on r.
// THE BASE POLYNOMIAL'S PERIOD COUNT, which was pinned at 2 and is the
// dominant precision knob nobody had.
//
// `K = initial_K * 2^r`, and each double angle is a `2x^2 - 1` that amplifies
// whatever error it is handed -- measured on the A100 this session at **~1.2
// bits a turn** (K 16/32/64 at 19.5/18.3/17.2 on one ladder). The SAME K is
// reachable with fewer turns by fitting more periods in the base polynomial,
// and the level ledger does not object: `GetNumEvalModLevels` is
// `Log2Ceil(coeffs) + r`, so
//
//     K = 64:  K0 = 2, deg 26, r = 5  ->  5 + 5 = 10 levels, 5 double angles
//              K0 = 16, deg 126, r = 2 ->  7 + 2 =  9 levels, 2 double angles
//
// -- one level CHEAPER and three turns fewer. The table for any period count
// is a Bessel expansion, `cos(a x) = J_0(a) + 2 sum (-1)^k J_2k(a) T_2k(x)`
// with `a = 2 pi K0`, which reproduces the shipped literal to 5e-05 (its own
// comment says it was checked by quadrature to 1e-05). The literal is kept
// verbatim for `K0 = 2` so that every existing preset is bit-for-bit
// unchanged; anything else is computed.
std::vector<double> BesselModCoefficients(int initial_K, int num_double_angle,
                                          double tol = 1e-6) {
  const double a = 2.0 * M_PI * initial_K;
  const double pref =
      std::pow(0.5 / M_PI, 1.0 / static_cast<double>(1 << num_double_angle));
  int deg = 2;
  while (deg < 400 && std::abs(2.0 * std::cyl_bessel_j(deg, a)) >= tol) {
    deg += 2;
  }
  std::vector<double> c(deg + 1, 0.0);
  for (int k = 0; 2 * k <= deg; k++) {
    const double j = std::cyl_bessel_j(2 * k, a);
    c[2 * k] = (k == 0 ? j : 2.0 * ((k % 2) ? -j : j)) * pref;
  }
  return c;
}

std::vector<double> ScaledModCoefficients(int num_double_angle) {
  std::vector<double> c{
      0.12517186708929745802,    0.0, 0.2894364973331168731,      0.0,
      0.36272381596524499154,    0.0, 0.3011054704600794278,      0.0,
      -0.10550875667295944105,   0.0, -0.43588877795190139706,    0.0,
      0.37482647434055190702,    0.0, -0.14821069913569220404,    0.0,
      0.03665437786710548091,    0.0, -0.0063882548960017121343,  0.0,
      0.00083684232451067872756, 0.0, -8.6443599931576702305e-05, 0.0,
      7.0966437900548814324e-06, 0.0, -5.228015817181348194e-07,  0.0,
      2.2714690137973883081e-08, 0.0, -2.3761936068138980797e-09};
  const double factor =
      std::pow(2.0 * M_PI, 1.0 / 8.0 - 1.0 / static_cast<double>(1 << num_double_angle));
  for (double &v : c) v *= factor;
  return c;
}

}  // namespace

BootParameter::BootParameter(int max_level, int num_cts_levels,
                             int num_stc_levels, int log_message_ratio /* = 5*/,
                             int num_slack_levels /* = 0*/,
                             int num_double_angle /* = 0*/,
                             int initial_K /* = 2*/)
    : max_level_{max_level},
      num_cts_levels_{num_cts_levels},
      num_stc_levels_{num_stc_levels},
      num_slack_levels_{num_slack_levels},
      log_message_ratio_{log_message_ratio},
      mod_coefficients_{
          (initial_K <= 2)
              ? ScaledModCoefficients(num_double_angle > 0
                                          ? num_double_angle
                                          : DoubleAngleCount())
              : BesselModCoefficients(initial_K,
                                      num_double_angle > 0
                                          ? num_double_angle
                                          : DoubleAngleCount())},
      num_double_angle_{num_double_angle > 0 ? num_double_angle
                                             : DoubleAngleCount()},
      initial_K_{initial_K > 0 ? initial_K : 2} {}

int BootParameter::GetNumEvalModLevels() const {
  return Log2Ceil(mod_coefficients_.size()) + num_double_angle_;
}

int BootParameter::GetMaxLevel() const { return max_level_; }
int BootParameter::GetCtSStartLevel() const { return max_level_; }
int BootParameter::GetEvalModStartLevel() const {
  return max_level_ - num_cts_levels_;
}
int BootParameter::GetEvalModEndLevel() const {
  return GetEvalModStartLevel() - GetNumEvalModLevels();
}
int BootParameter::GetStCStartLevel() const {
  return GetEvalModEndLevel() - num_slack_levels_;
}
int BootParameter::GetStartLevel() const { return max_level_; }
int BootParameter::GetEndLevel() const {
  return GetStCStartLevel() - num_stc_levels_;
}

}  // namespace cheddar