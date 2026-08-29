// The crossing constant is a property of the level-zero primes, not a fit.
//
// `HalfBoot` multiplies the message by `level_zero_scale / q0`. Two headers in
// this tree used to say that constant "has to be measured, not derived", and
// two independent fits were carried in the source because of it: 0.0298533 in
// `LlamaBlockTest` (fitted on `sylphflow16_35` by `SinCAttentionTest`) and
// 2^-4.9829 in the conjugate-invariant FFN's `boundary`.
//
// It is derived. `BootContext`'s constructor already computes both halves:
//
//     log_scaleup_   = round(log2 q0_prod) - round(log2 base_scale) - lmr
//     message_ratio_ = base_scale * 2^log_scaleup_ / q0_prod
//
// The rounding is the whole story. `2^-log_message_ratio` is what the design
// ASKS for; what it GETS is that divided by `q0_prod / 2^round(log2 q0_prod)`,
// and a product of NTT-friendly primes is not a power of two. On the shipped
// presets the correction runs from 1.0097 to 1.1856 -- so a constant fitted on
// one preset and carried to another is wrong by percents, which is exactly
// what had happened: `bootparam_35`'s true ratio is 0.0310414 and the block
// was handing it `sylphflow16_35`'s 0.0298533, 4.0% out on every score and
// every value.
//
// THE MEASUREMENT IS PERMUTATION-FREE, WHICH IS THE POINT. `CoeffToSlot`
// carries a bit reversal and the ordinary ring's full-slot HalfBoot folds the
// two coefficient halves onto the real and imaginary axes, so there is no
// index map to get right here and none is used: for a bijection of the
// coefficients onto the slots' real components,
//
//     sum_j |slot_j|^2  =  ratio^2 * sum_i coeff_i^2
//
// whatever the bijection is. `HalfBootTest` spent five attempts on the index
// map before going round it; the norm goes round it by construction. The
// sorted-magnitude check below then confirms it really is a bijection and not
// a mixing, which the norm alone could not tell.
//
// RIDE LOW. `EvalMod` is a Chebyshev sine with no arcsine correction, so it
// returns `m - a m^3` with `a` about 0.0026 (Doing.md 1.5cv). At |m| ~ 1 that
// is a systematic 9e-4 on the fit -- enough to hide the derivation's own
// agreement. At |m| ~ 0.05 it is 6e-6, and the additive floor averages down
// over `degree` coefficients, so the fit resolves the ratio to ~1e-6 and the
// claim is testable rather than merely consistent.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "Testbed.h"

using word = uint32_t;

namespace {

// Small enough that EvalMod's cubic is 6e-6 of the value; see the header.
constexpr double kRide = 0.05;

}  // namespace

TEST_P(Testbed32, TheCrossingConstantIsTheLevelZeroPrimeRatio) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  ASSERT_FALSE(param_->conjugate_invariant_)
      << "the real subring's HalfBoot lands real slots and needs its own "
         "norm identity; the CI branch checks this through its own fit";

  const int degree = param_->degree_;
  const int num_slots = param_->MaxNumSlots();
  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  // Coefficients, not slots: this is the direction the pipeline crosses in.
  std::mt19937 rng(20260829);
  std::uniform_real_distribution<double> dist(-kRide, kRide);
  std::vector<double> coeffs(degree);
  for (double &c : coeffs) c = dist(rng);

  Plaintext<word> ptxt;
  const double level_zero_scale = param_->GetScale(0);
  context_->encoder_.EncodeCoeff(ptxt, 0, level_zero_scale, coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, ptxt);
  ct.SetNumSlots(num_slots);

  Ciphertext<word> slots;
  boot->HalfBoot(slots, ct, interface_->GetEvkMap());
  std::vector<Complex> got;
  DecryptAndDecode(got, slots);

  // The norm ratio. No index map, no conjugation convention, no sign.
  double num = 0.0, den = 0.0;
  for (double c : coeffs) den += c * c;
  for (int j = 0; j < num_slots; j++) num += std::norm(got[j]);
  const double measured = std::sqrt(num / den);
  const double derived = boot->GetMessageRatio();
  const double nominal =
      std::pow(2.0, -boot->GetBootParameter().GetLogMessageRatio());

  std::cout << GetParam() << ":\n"
            << "  crossing constant  measured " << measured << " = 2^"
            << std::log2(measured) << "\n"
            << "                     derived  " << derived << " = 2^"
            << std::log2(derived) << "\n"
            << "                     nominal  " << nominal << " = 2^"
            << std::log2(nominal) << "\n"
            << "  measured/derived " << measured / derived
            << ",  nominal/derived " << nominal / derived << std::endl;

  // The derivation, to the fit's own resolution.
  EXPECT_NEAR(measured / derived, 1.0, 2e-4)
      << "the level-zero prime ratio is not what the crossing applies";

  // And that it is a real claim: the nominal power of two is a DIFFERENT
  // number on every preset here, by between 1% and 19%, so a test that passed
  // against the nominal would not be testing anything.
  const double gap = std::abs(nominal / derived - 1.0);
  std::cout << "  the nominal is out by " << 100.0 * gap << "%" << std::endl;

  // A bijection of coefficients onto real slot components, not a mixing: the
  // sorted magnitudes have to agree one for one. Only the norm is used above,
  // and the norm cannot tell the two apart.
  std::vector<double> want(degree), have;
  have.reserve(degree);
  for (int i = 0; i < degree; i++) want[i] = std::abs(coeffs[i]) * derived;
  for (int j = 0; j < num_slots; j++) {
    have.push_back(std::abs(got[j].real()));
    if (static_cast<int>(have.size()) < degree)
      have.push_back(std::abs(got[j].imag()));
  }
  ASSERT_EQ(static_cast<int>(have.size()), degree);
  std::sort(want.begin(), want.end());
  std::sort(have.begin(), have.end());
  double worst = 0.0;
  for (int i = 0; i < degree; i++)
    worst = std::max(worst, std::abs(have[i] - want[i]));
  std::cout << "  sorted magnitudes agree to " << worst << " against a ride of "
            << kRide * derived << std::endl;
  EXPECT_LT(worst, 0.02 * kRide * derived)
      << "HalfBoot did not carry the coefficients onto the slots one for one";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "sylphflow16_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
