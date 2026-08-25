// CoeffToSlot and SlotToCoeff on the conjugate-invariant ring.
//
// This is step 5 of Doing.md 1.5av, and the part of it that carries the risk.
// The bootstrap is CtS -> EvalMod -> StC, and EvalMod is slot-wise polynomial
// evaluation that does not know which ring it is in. The two transforms do.
//
// WHAT IS BEING ASSERTED. R+ is totally real, so both ends of CtS are real --
// `E^-1 = Re(SpecialIFFT)`. But the butterfly factorisation that makes CtS
// cheap is a factorisation of the COMPLEX transform, and taking real parts does
// not commute with it. So the intermediate is complex and is carried as a pair
// of real ciphertexts (ComplexLinearTransform), while the stage matrices
// themselves are the ordinary ring's, unchanged. Three claims follow, and the
// two tests here are exactly those claims run on the GPU:
//
//   1. the composed plain_ifft_stages_ product A satisfies
//        Re(A z) = num_slots * BitReverse(E^-1 z)
//      -- so CtS needs no new matrices and no new scalar, cts_const_ included;
//   2. plain_fft_stages_ composed is A conjugate-transposed, so the same is
//      true in the other direction, up to E^T E = diag(n, 2n, ..., 2n) -- the
//      basis vector 1 has no conjugate partner where every c_j has one. That
//      diagonal is folded into StC's first phase as a column scaling;
//   3. therefore stc_const_ is unchanged too.
//
// The reference is the ENCODER, not another run of the transform: the expected
// slot vector is written down from the coefficients directly, and the expected
// coefficient vector likewise. Both tests run unchanged on the ordinary ring's
// bootparam_35, where the fold is complex and the code takes the other path, so
// a half-applied change cannot pass quietly.
//
// The transforms are compiled standalone rather than through BootContext, whose
// constructor still refuses the conjugate-invariant flag. cts_const is set to
// 1 / num_slots and stc_const to 1 so that what comes back is the answer itself
// rather than the answer times a bootstrap-specific scalar.

#include "Testbed.h"
#include "extension/EvalSpecialFFT.h"

using word = uint32_t;

namespace {

int BitReverse(int value, int num_bits) {
  int res = 0;
  for (int i = 0; i < num_bits; i++) {
    res = (res << 1) | ((value >> i) & 1);
  }
  return res;
}

}  // namespace

class CiFft : public Testbed<word> {
 protected:
  // EvalSpecialFFT only needs a Context; BootContext would refuse the flag.
  bool UseBootContext() const override { return false; }

  BootParameter MakeBootParameter() const {
    return BootParameter(param_->max_level_, num_cts_levels_, num_stc_levels_,
                         5, 0);
  }

  // The coefficient vector both directions are written down from.
  //
  // A sparsely packed message occupies coefficient `j * gap` with
  // `gap = MaxNumSlots() / num_slots` -- which is `degree / num_slots` on the
  // real subring and `(degree / 2) / num_slots` on the ordinary one, i.e. the
  // same expression. The ordinary ring additionally carries the imaginary half
  // of slot `j` at `j * gap + degree / 2`; the real subring has no imaginary
  // half, and the coefficient that would mirror it is the one the fold reads
  // back as `-a_(n-j)`.
  void DrawCoefficients(std::vector<double> &coeffs, int num_slots) {
    const int degree = 1 << log_degree_;
    const int gap = param_->MaxNumSlots() / num_slots;
    const bool ci = param_->conjugate_invariant_;

    std::vector<Complex> draw;
    GenerateRandomMessage(draw, num_slots, -1.0, 1.0, /*complex=*/true);

    coeffs.assign(degree, 0.0);
    for (int j = 0; j < num_slots; j++) {
      coeffs[j * gap] = draw[j].real();
      if (!ci) coeffs[j * gap + degree / 2] = draw[j].imag();
    }
  }

  // What CtS must produce in the slots: the coefficient vector folded into the
  // ring's own transform input, in bit-reversed order.
  void ExpectedSlots(std::vector<Complex> &expected,
                     const std::vector<double> &coeffs, int num_slots) const {
    const int degree = 1 << log_degree_;
    const int gap = param_->MaxNumSlots() / num_slots;
    const bool ci = param_->conjugate_invariant_;
    const int log_num_slots = Log2Ceil(num_slots);

    expected.assign(num_slots, Complex(0, 0));
    for (int j = 0; j < num_slots; j++) {
      Complex value = ci ? Complex(coeffs[j * gap], 0.0)
                         : Complex(coeffs[j * gap],
                                   coeffs[j * gap + degree / 2]);
      expected[BitReverse(j, log_num_slots)] = value;
    }
  }
};

TEST_P(CiFft, CoeffToSlot) {
  const int num_slots = param_->MaxNumSlots();
  const int level = MakeBootParameter().GetCtSStartLevel();

  EvalSpecialFFT<word> fft(context_, MakeBootParameter(), num_slots,
                           1.0 / num_slots, 1.0);
  EvkRequest req;
  fft.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<double> coeffs;
  DrawCoefficients(coeffs, num_slots);

  Plaintext<word> ptxt;
  context_->encoder_.EncodeCoeff(ptxt, level, DetermineScale(level), coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, ptxt);

  Ciphertext<word> res_ct;
  fft.EvaluateCtS(context_, res_ct, ct, interface_->GetEvkMap());

  std::vector<Complex> res;
  DecryptAndDecode(res, res_ct);

  std::vector<Complex> expected;
  ExpectedSlots(expected, coeffs, num_slots);
  CompareMessages(expected, res);
}

TEST_P(CiFft, SlotToCoeff) {
  const int num_slots = param_->MaxNumSlots();
  const int level = MakeBootParameter().GetStCStartLevel();

  EvalSpecialFFT<word> fft(context_, MakeBootParameter(), num_slots,
                           1.0 / num_slots, 1.0);
  EvkRequest req;
  fft.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  std::vector<double> coeffs;
  DrawCoefficients(coeffs, num_slots);
  std::vector<Complex> slots;
  ExpectedSlots(slots, coeffs, num_slots);

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, slots, level);

  Ciphertext<word> res_ct;
  fft.EvaluateStC(context_, res_ct, ct, interface_->GetEvkMap());

  Plaintext<word> res_ptxt;
  interface_->Decrypt(res_ptxt, res_ct);
  std::vector<double> res_coeffs;
  context_->encoder_.DecodeCoeff(res_coeffs, res_ptxt);

  // DecodeCoeff returns every coefficient, and at full slot packing every one
  // of them carries a value, so the comparison is over the whole vector.
  std::vector<Complex> expected, obtained;
  for (size_t i = 0; i < coeffs.size(); i++) {
    expected.emplace_back(coeffs[i], 0.0);
    obtained.emplace_back(res_coeffs[i], 0.0);
  }
  CompareMessages(expected, obtained);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, CiFft, testing::Values("ci16_35.json", "bootparam_35.json"),
    [](const testing::TestParamInfo<CiFft::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
