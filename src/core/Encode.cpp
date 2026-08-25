#include "core/Encode.h"

#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"
#include "core/BigInt.h"

namespace cheddar {

template <typename word>
Encoder<word>::Encoder(const Parameter<word> &param,
                       const NTTHandler<word> &ntt_handler)
    : param_{param},
      ntt_handler_{ntt_handler},
      M_{param.CyclotomicIndex()},
      twiddle_factors_(param.CyclotomicIndex()) {
  for (int i = 0; i < M_; i++) {
    // e^(2*pi*sqrt(-1)*i/M)
    twiddle_factors_[i] = std::polar(1.0, 2.0 * M_PI * i / M_);
  }
}

template <typename word>
void Encoder<word>::Encode(Plaintext<word> &ptxt, int level, double scale,
                           const std::vector<Complex> &message,
                           int num_aux /*= 0*/) const {
  int msg_length = message.size();
  int num_slots = 1 << Log2Ceil<int>(msg_length);
  AssertTrue(num_slots <= param_.MaxNumSlots(),
             "Encode: too many slots for this ring");
  std::vector<Complex> padded_msg(num_slots);
  std::copy(message.begin(), message.end(), padded_msg.begin());

  if (param_.conjugate_invariant_) {
    // R+ is totally real: a slot is one real number and there is nowhere for
    // an imaginary part to go. Dropping it quietly would encode something the
    // caller did not ask for, so it is a caller error. The bound is relative,
    // because a message built out of cosines carries a rounding-sized
    // imaginary residue that means nothing.
    double max_real = 0.0;
    double max_imag = 0.0;
    for (const auto &value : padded_msg) {
      max_real = std::max(max_real, std::abs(value.real()));
      max_imag = std::max(max_imag, std::abs(value.imag()));
    }
    AssertTrue(max_imag <= 1e-8 * std::max(max_real, 1.0),
               "Encode: the conjugate-invariant ring has real slots, and this "
               "message has an imaginary part");
  }

  EncodeWorker(ptxt, level, scale, padded_msg, num_aux);
}

template <typename word>
void Encoder<word>::EncodeWorker(Plaintext<word> &ptxt, int level, double scale,
                                 std::vector<Complex> &message,
                                 int num_aux /*= 0*/) const {
  SpecialIFFT(message);
  if (param_.conjugate_invariant_) {
    FoldedVectorToPlaintext(ptxt, level, scale, message, num_aux);
  } else {
    ComplexVectorToPlaintext(ptxt, level, scale, message, num_aux);
  }
  NPInfo np = param_.LevelToNP(level, num_aux);
  auto mx_temp = ptxt.View();
  ntt_handler_.NTT(mx_temp, np, ptxt.ConstView(), true);
}

template <typename word>
void Encoder<word>::Decode(std::vector<Complex> &message,
                           const Plaintext<word> &ptxt) const {
  Plaintext<word> tmp;
  NPInfo np = ptxt.GetNP();
  AssertTrue(np.num_aux_ == 0, "Decode: Aux primes not supported");

  DeviceVector<word> icrt1_dv = MakeICRTConstants(np);

  tmp.ModifyNP(np);
  tmp.SetNumSlots(ptxt.GetNumSlots());
  tmp.SetScale(ptxt.GetScale());

  auto mx_temp = tmp.View();
  ntt_handler_.INTTAndMultConst(mx_temp, np, ptxt.ConstView(),
                                icrt1_dv.ConstView());
  if (param_.conjugate_invariant_) {
    PlaintextToFoldedVector(message, tmp);
  } else {
    PlaintextToComplexVector(message, tmp);
  }
  SpecialFFT(message);
}

// Constants for the inverse CRT performed on the host during decoding:
// icrt1[i] = ( degree * prod_{j != i} p_j )^{-1} mod p_i. The INTT multiplies
// each limb by this, so PlaintextToRealVector / PlaintextToComplexVector only
// have to scale by prod_{j != i} p_j over the integers and sum.
template <typename word>
DeviceVector<word> Encoder<word>::MakeICRTConstants(const NPInfo &np) const {
  auto primes = param_.GetPrimeVector(np);
  int num_total_primes = np.GetNumTotal();
  word degree = param_.degree_;

  HostVector<word> icrt1(num_total_primes);
  for (int i = 0; i < num_total_primes; i++) {
    word mod_prime = primes[i];
    word prod = 1;
    for (int j = 0; j < num_total_primes; j++) {
      if (i == j) continue;
      prod = primeutil::MultMod(prod, primes[j], mod_prime);
    }
    prod = primeutil::MultMod(prod, degree, mod_prime);
    // deliberately not using the Montgomery form here
    icrt1[i] = primeutil::InvMod(prod, mod_prime);
  }

  DeviceVector<word> icrt1_dv(num_total_primes);
  CopyHostToDevice(icrt1_dv, icrt1);
  return icrt1_dv;
}

template <typename word>
void Encoder<word>::EncodeCoeff(Plaintext<word> &ptxt, int level, double scale,
                                const std::vector<double> &coeffs,
                                int num_aux /*= 0*/) const {
  RealVectorToPlaintext(ptxt, level, scale, coeffs, num_aux);
  NPInfo np = param_.LevelToNP(level, num_aux);
  auto mx_temp = ptxt.View();
  ntt_handler_.NTT(mx_temp, np, ptxt.ConstView(), true);
}

template <typename word>
void Encoder<word>::DecodeCoeff(std::vector<double> &coeffs,
                                const Plaintext<word> &ptxt) const {
  Plaintext<word> tmp;
  NPInfo np = ptxt.GetNP();
  AssertTrue(np.num_aux_ == 0, "DecodeCoeff: Aux primes not supported");

  DeviceVector<word> icrt1_dv = MakeICRTConstants(np);

  tmp.ModifyNP(np);
  tmp.SetNumSlots(ptxt.GetNumSlots());
  tmp.SetScale(ptxt.GetScale());

  auto mx_temp = tmp.View();
  ntt_handler_.INTTAndMultConst(mx_temp, np, ptxt.ConstView(),
                                icrt1_dv.ConstView());
  PlaintextToRealVector(coeffs, tmp);
}

template <typename word>
void Encoder<word>::RealVectorToPlaintext(Plaintext<word> &ptxt, int level,
                                          double scale,
                                          const std::vector<double> &coeffs,
                                          int num_aux /*= 0*/) const {
  int degree = param_.degree_;
  int num_coeffs = static_cast<int>(coeffs.size());

  AssertTrue(num_coeffs > 0 && num_coeffs <= degree,
             "RealVectorToPlaintext: invalid number of coefficients");

  NPInfo np = param_.LevelToNP(level, num_aux);
  auto primes = param_.GetPrimeVector(np);
  int num_total_primes = np.GetNumTotal();
  HostVector<word> mx(num_total_primes * degree, 0);

  std::vector<BigInt> big_primes;
  for (int j = 0; j < num_total_primes; j++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[j]));
  }

  BigInt residue(static_cast<uint64_t>(0));
  for (int i = 0; i < num_coeffs; i++) {
    // BigInt(double) truncates, so round here to get the nearest integer.
    BigInt value(std::round(coeffs[i] * scale));
    for (int j = 0; j < num_total_primes; j++) {
      // BigInt::Mod is always non-negative, which is what the RNS limb needs
      // even for a negative coefficient.
      BigInt::Mod(residue, value, big_primes[j]);
      mx[i + j * degree] = residue.GetUnsigned();
    }
  }

  ptxt.ModifyNP(np);
  // Coefficient encoding occupies every coefficient, so the slot metadata is
  // set to the full width; it is not meaningful for this encoding.
  ptxt.SetNumSlots(param_.MaxNumSlots());
  ptxt.SetScale(scale);
  CopyHostToDevice(ptxt.mx_, mx);
}

template <typename word>
void Encoder<word>::PlaintextToRealVector(std::vector<double> &coeffs,
                                          const Plaintext<word> &ptxt) const {
  NPInfo np = ptxt.GetNP();
  int num_total_primes = np.GetNumTotal();
  auto primes = param_.GetPrimeVector(np);
  int degree = param_.degree_;
  double scale = ptxt.GetScale();

  HostVector<word> intt_res;
  CopyDeviceToHost(intt_res, ptxt.mx_);

  std::vector<BigInt> big_primes;
  BigInt prime_prod(static_cast<uint64_t>(1));
  for (int i = 0; i < num_total_primes; i++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[i]));
    BigInt::Mult(prime_prod, prime_prod, big_primes[i]);
  }
  BigInt half_prime_prod(static_cast<uint64_t>(0));
  BigInt::Div2(half_prime_prod, prime_prod);

  // prod_{k != j} p_k does not depend on the coefficient index, so hoist it out
  // of the loop over `degree` coefficients.
  std::vector<BigInt> crt_weight;
  for (int j = 0; j < num_total_primes; j++) {
    BigInt prod(static_cast<uint64_t>(1));
    for (int k = 0; k < num_total_primes; k++) {
      if (j == k) continue;
      BigInt::Mult(prod, prod, big_primes[k]);
    }
    crt_weight.push_back(prod);
  }

  coeffs.resize(degree);
  BigInt tmp(static_cast<uint64_t>(0));
  for (int i = 0; i < degree; i++) {
    BigInt icrt(static_cast<uint64_t>(0));
    for (int j = 0; j < num_total_primes; j++) {
      BigInt limb(static_cast<uint64_t>(intt_res[j * degree + i]));
      BigInt::Mult(tmp, limb, crt_weight[j]);
      BigInt::Add(icrt, icrt, tmp);
    }
    BigInt::NormalizeMod(icrt, icrt, prime_prod, half_prime_prod);
    coeffs[i] = icrt.GetDouble() / scale;
  }
}

// The conjugate-invariant counterparts of the two functions above.
//
// SpecialIFFT at S slots returns the fold of the rank-S subring element, so
// the ring basis is its real part and its imaginary part is the same
// coefficients mirrored:
//
//     f[r] = a_(r*gap) - i a_((S-r)*gap),   gap = degree / S,   f[0] = a_0
//
// -- f[0] has no mirror partner because the lift's coefficient at Y^N is zero,
// which is the one place the antisymmetry pins a value rather than relating
// two. That the imaginary half carries nothing new is exactly the halving,
// restated on the host: N real slots out of the same N coefficients.
template <typename word>
void Encoder<word>::FoldedVectorToPlaintext(Plaintext<word> &ptxt, int level,
                                            double scale,
                                            const std::vector<Complex> &data,
                                            int num_aux /*= 0*/) const {
  int num_slots = data.size();
  int degree = param_.degree_;

  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "FoldedVectorToPlaintext: Power of 2 num slots only");
  AssertTrue(num_slots <= degree, "FoldedVectorToPlaintext: Too many slots");
  int gap = degree / num_slots;

  std::vector<double> coeffs(degree, 0.0);
  for (int i = 0; i < num_slots; i++) {
    coeffs[i * gap] = data[i].real();
  }

  RealVectorToPlaintext(ptxt, level, scale, coeffs, num_aux);
  // RealVectorToPlaintext has no slot count of its own to record; this one
  // does, and a sparsely packed message needs it to decode.
  ptxt.SetNumSlots(num_slots);
}

template <typename word>
void Encoder<word>::PlaintextToFoldedVector(std::vector<Complex> &data,
                                            const Plaintext<word> &ptxt) const {
  int num_slots = ptxt.GetNumSlots();
  int degree = param_.degree_;

  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "PlaintextToFoldedVector: Power of 2 num slots only");
  AssertTrue(num_slots <= degree, "PlaintextToFoldedVector: Too many slots");
  int gap = degree / num_slots;

  std::vector<double> coeffs;
  PlaintextToRealVector(coeffs, ptxt);

  data.resize(num_slots);
  data[0] = Complex(coeffs[0], 0.0);
  for (int i = 1; i < num_slots; i++) {
    data[i] = Complex(coeffs[i * gap], -coeffs[(num_slots - i) * gap]);
  }
}

template <typename word>
void Encoder<word>::SpecialIFFT(std::vector<Complex> &data) const {
  int num_slots = data.size();
  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "Power of 2 num slots only");

  for (int stride = num_slots / 2; stride >= 1; stride /= 2) {
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M_ / st8;
    for (int i = 0; i < num_slots; i += stride_group_size) {
      for (int j = 0; j < stride; j++) {
        int twiddle_index = (st8 - (param_.GetGaloisFactor(j) % st8)) * gap;
        auto x = data[i + j] + data[i + j + stride];
        auto y = data[i + j] - data[i + j + stride];
        y *= twiddle_factors_[twiddle_index];
        data[i + j] = x;
        data[i + j + stride] = y;
      }
    }
  }
  BitReverseVector(data);
  for (int i = 0; i < num_slots; ++i) {
    data[i] /= num_slots;
  }
}

template <typename word>
void Encoder<word>::SpecialFFT(std::vector<Complex> &data) const {
  int num_slots = data.size();
  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "Power of 2 num slots only");
  BitReverseVector(data);

  for (int stride = 1; stride < num_slots; stride *= 2) {
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M_ / st8;
    for (int i = 0; i < num_slots; i += stride_group_size) {
      for (int j = 0; j < stride; j++) {
        int twiddle_index = (param_.GetGaloisFactor(j) % st8) * gap;
        auto x = data[i + j];
        auto y = data[i + j + stride];
        y *= twiddle_factors_[twiddle_index];
        data[i + j] = x + y;
        data[i + j + stride] = x - y;
      }
    }
  }
}

template <typename word>
void Encoder<word>::ComplexVectorToPlaintext(Plaintext<word> &ptxt, int level,
                                             double scale,
                                             const std::vector<Complex> &data,
                                             int num_aux /*= 0*/) const {
  int num_slots = data.size();
  int degree = param_.degree_;
  int half_degree = degree / 2;
  int gap = half_degree / num_slots;

  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "ComplexVectorToPlaintext: Power of 2 num slots only");
  AssertTrue(num_slots <= half_degree,
             "ComplexVectorToPlaintext: Too many slots");

  NPInfo np = param_.LevelToNP(level, num_aux);
  auto primes = param_.GetPrimeVector(np);
  int num_total_primes = np.GetNumTotal();
  HostVector<word> mx(num_total_primes * degree, 0);

  std::vector<BigInt> big_primes;

  for (int j = 0; j < num_total_primes; j++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[j]));
  }

  for (int i = 0; i < num_slots; i++) {
    Complex value = data[i] * scale;
    double real = value.real();
    double imag = value.imag();

    BigInt big_real(real);
    BigInt big_imag(imag);

    int real_degree_index = i * gap;
    int imag_degree_index = real_degree_index + half_degree;

    BigInt real_tmp(0.0);
    BigInt imag_tmp(0.0);

    for (int j = 0; j < num_total_primes; j++) {
      const BigInt &prime = big_primes[j];
      BigInt::Mod(real_tmp, big_real, prime);
      BigInt::Mod(imag_tmp, big_imag, prime);
      word real_mod = real_tmp.GetUnsigned();
      word imag_mod = imag_tmp.GetUnsigned();
      mx[real_degree_index + j * param_.degree_] = real_mod;
      mx[imag_degree_index + j * param_.degree_] = imag_mod;
    }
  }

  ptxt.ModifyNP(np);
  ptxt.SetNumSlots(num_slots);
  ptxt.SetScale(scale);
  CopyHostToDevice(ptxt.mx_, mx);
}

template <typename word>
void Encoder<word>::PlaintextToComplexVector(
    std::vector<Complex> &data, const Plaintext<word> &ptxt) const {
  int num_slots = ptxt.GetNumSlots();
  double scale = ptxt.GetScale();
  NPInfo np = ptxt.GetNP();
  int num_total_primes = np.GetNumTotal();
  auto primes = param_.GetPrimeVector(np);
  int degree = param_.degree_;
  int half_degree = degree / 2;
  int gap = half_degree / num_slots;

  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "ComplexVectorToPlaintext: Power of 2 num slots only");
  AssertTrue(num_slots <= half_degree,
             "PlaintextToComplexVector: Too many slots");

  data.resize(num_slots);

  HostVector<word> intt_res;
  CopyDeviceToHost(intt_res, ptxt.mx_);

  std::vector<BigInt> big_primes;
  BigInt prime_prod(static_cast<uint64_t>(1));
  for (int i = 0; i < num_total_primes; i++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[i]));
    BigInt::Mult(prime_prod, prime_prod, big_primes[i]);
  }
  BigInt half_prime_prod(static_cast<uint64_t>(0));
  BigInt::Div2(half_prime_prod, prime_prod);

  BigInt tmp(static_cast<uint64_t>(0));
  for (int i = 0; i < num_slots; i++) {
    BigInt real_icrt(static_cast<uint64_t>(0));
    BigInt imag_icrt(static_cast<uint64_t>(0));
    for (int j = 0; j < num_total_primes; j++) {
      BigInt real_value(static_cast<uint64_t>(intt_res[j * degree + i * gap]));
      BigInt imag_value(
          static_cast<uint64_t>(intt_res[j * degree + i * gap + half_degree]));
      BigInt prod(static_cast<uint64_t>(1));
      for (int k = 0; k < num_total_primes; k++) {
        if (j == k) continue;
        BigInt::Mult(prod, prod, big_primes[k]);
      }
      BigInt::Mult(tmp, real_value, prod);
      BigInt::Add(real_icrt, real_icrt, tmp);
      BigInt::Mult(tmp, imag_value, prod);
      BigInt::Add(imag_icrt, imag_icrt, tmp);
    }
    BigInt::NormalizeMod(real_icrt, real_icrt, prime_prod, half_prime_prod);
    BigInt::NormalizeMod(imag_icrt, imag_icrt, prime_prod, half_prime_prod);

    double real = real_icrt.GetDouble() / scale;
    double imag = imag_icrt.GetDouble() / scale;
    data[i] = Complex(real, imag);
  }
}

// SinC splits the message into N/k blocks of k/2 slots, sends each through
// iDFT_k, and interleaves the resulting subring elements with stride N/k. Both
// directions share that block geometry, so it is derived once here.
//
// The k/2 complex outputs of SpecialIFFT carry the k real coefficients of the
// subring element as (real parts | imaginary parts), the same split
// ComplexVectorToPlaintext performs at the full degree.
namespace {

struct SinCLayout {
  int num_blocks;       // N/k, and also the coefficient stride
  int slots_per_block;  // k/2
};

SinCLayout MakeSinCLayout(int degree, int sub_degree) {
  AssertTrue(sub_degree >= 2 && sub_degree <= degree &&
                 IsPowOfTwo(sub_degree) && degree % sub_degree == 0,
             "SinC: sub_degree must be a power of two dividing the ring "
             "degree, and at least 2");
  return SinCLayout{degree / sub_degree, sub_degree / 2};
}

// The conjugate-invariant ring is free over its rank-k subring on
// {1, c_1, ..., c_{d-1}} rather than on monomials, and the change of basis
// between the full ring's c-coefficients and the d module components is the
// banded two-term map of Doing.md 1.5ba -- the same one MlweHandler's
// ModDecomp implements on the device, in real arithmetic here because the
// encoder works on messages. Recompose:
//
//     coeffs[t*d + i] = comp_i[t] + comp_{d-i}[t+1],    comp_.[k] = 0
//
// with the i = 0 class pure, and the inverse the alternating-sign suffix-sum
// scan down each class pair (i, d-i).
std::vector<double> CiSinCRecompose(
    const std::vector<std::vector<double>> &comp, int num_blocks,
    int sub_degree) {
  std::vector<double> out(static_cast<size_t>(num_blocks) * sub_degree, 0.0);
  for (int t = 0; t < sub_degree; t++) {
    for (int i = 0; i < num_blocks; i++) {
      double v = comp[i][t];
      if (i != 0 && t + 1 < sub_degree) v += comp[num_blocks - i][t + 1];
      out[static_cast<size_t>(t) * num_blocks + i] = v;
    }
  }
  return out;
}

std::vector<std::vector<double>> CiSinCDecompose(
    const std::vector<double> &coeffs, int num_blocks, int sub_degree) {
  std::vector<std::vector<double>> comp(
      num_blocks, std::vector<double>(sub_degree, 0.0));
  for (int t = 0; t < sub_degree; t++) {
    comp[0][t] = coeffs[static_cast<size_t>(t) * num_blocks];
  }
  for (int i = 1; i <= num_blocks / 2; i++) {
    const int mi = num_blocks - i;
    double acc_i = 0.0;
    double acc_m = 0.0;
    for (int t = sub_degree - 1; t >= 0; t--) {
      const double new_i =
          coeffs[static_cast<size_t>(t) * num_blocks + i] - acc_m;
      const double new_m =
          coeffs[static_cast<size_t>(t) * num_blocks + mi] - acc_i;
      comp[i][t] = new_i;
      comp[mi][t] = new_m;  // mi == i rewrites the same value
      acc_i = new_i;
      acc_m = new_m;
    }
  }
  return comp;
}

}  // namespace

template <typename word>
void Encoder<word>::EncodeSinC(Plaintext<word> &ptxt, int level, double scale,
                               const std::vector<Complex> &message,
                               int sub_degree, int num_aux /*= 0*/) const {
  const int degree = param_.degree_;
  const SinCLayout layout = MakeSinCLayout(degree, sub_degree);

  if (param_.conjugate_invariant_) {
    // R+ is free over its rank-k subring on {1, c_1, ..., c_{d-1}}, so a
    // block is still one subring element per module component -- but the
    // component-to-coefficient map is the banded two-term recomposition of
    // Doing.md 1.5ba, not a stride, and a block holds k REAL slots, the
    // subring being itself totally real. The per-block transform is the
    // conjugate-invariant encoder at size k: SpecialIFFT and the real part,
    // which is EncodeWorker's own fold at the subring's degree.
    const int num_blocks = degree / sub_degree;
    AssertTrue(static_cast<int>(message.size()) == degree,
               "EncodeSinC: message must fill every real slot of the ring");
    double max_real = 0.0;
    double max_imag = 0.0;
    for (const auto &value : message) {
      max_real = std::max(max_real, std::abs(value.real()));
      max_imag = std::max(max_imag, std::abs(value.imag()));
    }
    AssertTrue(max_imag <= 1e-8 * std::max(max_real, 1.0),
               "EncodeSinC: the conjugate-invariant ring has real slots, and "
               "this message has an imaginary part");

    std::vector<std::vector<double>> comp(
        num_blocks, std::vector<double>(sub_degree));
    std::vector<Complex> block(sub_degree);
    for (int i = 0; i < num_blocks; i++) {
      std::copy(message.begin() + static_cast<size_t>(i) * sub_degree,
                message.begin() + static_cast<size_t>(i + 1) * sub_degree,
                block.begin());
      SpecialIFFT(block);
      for (int t = 0; t < sub_degree; t++) comp[i][t] = block[t].real();
    }

    EncodeCoeff(ptxt, level, scale,
                CiSinCRecompose(comp, num_blocks, sub_degree), num_aux);
    return;
  }

  AssertTrue(static_cast<int>(message.size()) == degree / 2,
             "EncodeSinC: message must fill every slot of the ring");

  std::vector<double> coeffs(degree, 0.0);
  std::vector<Complex> block(layout.slots_per_block);

  for (int i = 0; i < layout.num_blocks; i++) {
    const auto first = message.begin() + i * layout.slots_per_block;
    std::copy(first, first + layout.slots_per_block, block.begin());

    // iDFT_2 is the identity on a single complex number, and SpecialIFFT's
    // bit reversal is not defined for a one-element vector, so skip it.
    if (layout.slots_per_block > 1) SpecialIFFT(block);

    for (int t = 0; t < layout.slots_per_block; t++) {
      const int real_index = i + t * layout.num_blocks;
      const int imag_index =
          i + (t + layout.slots_per_block) * layout.num_blocks;
      coeffs[real_index] = block[t].real();
      coeffs[imag_index] = block[t].imag();
    }
  }

  EncodeCoeff(ptxt, level, scale, coeffs, num_aux);
}

template <typename word>
void Encoder<word>::DecodeSinC(std::vector<Complex> &message,
                               const Plaintext<word> &ptxt,
                               int sub_degree) const {
  const int degree = param_.degree_;
  const SinCLayout layout = MakeSinCLayout(degree, sub_degree);

  std::vector<double> coeffs;
  DecodeCoeff(coeffs, ptxt);
  AssertTrue(static_cast<int>(coeffs.size()) == degree,
             "DecodeSinC: unexpected coefficient count");

  if (param_.conjugate_invariant_) {
    // The inverse of the conjugate-invariant EncodeSinC: the scan back to
    // the module components, then per block the fold-and-transform that
    // PlaintextToFoldedVector performs, at the subring's size.
    const int num_blocks = degree / sub_degree;
    const auto comp = CiSinCDecompose(coeffs, num_blocks, sub_degree);

    message.resize(degree);
    std::vector<Complex> block(sub_degree);
    for (int i = 0; i < num_blocks; i++) {
      block[0] = Complex(comp[i][0], 0.0);
      for (int t = 1; t < sub_degree; t++) {
        block[t] = Complex(comp[i][t], -comp[i][sub_degree - t]);
      }
      SpecialFFT(block);
      std::copy(block.begin(), block.end(),
                message.begin() + static_cast<size_t>(i) * sub_degree);
    }
    return;
  }

  message.resize(degree / 2);
  std::vector<Complex> block(layout.slots_per_block);

  for (int i = 0; i < layout.num_blocks; i++) {
    for (int t = 0; t < layout.slots_per_block; t++) {
      const int real_index = i + t * layout.num_blocks;
      const int imag_index =
          i + (t + layout.slots_per_block) * layout.num_blocks;
      block[t] = Complex(coeffs[real_index], coeffs[imag_index]);
    }

    if (layout.slots_per_block > 1) SpecialFFT(block);

    std::copy(block.begin(), block.end(),
              message.begin() + i * layout.slots_per_block);
  }
}

template <typename word>
void Encoder<word>::EncodeConstant(Constant<word> &constant, int level,
                                   double scale, double number,
                                   int num_aux /*= 0*/) const {
  BigInt big_int(number * scale);
  NPInfo np = param_.LevelToNP(level, num_aux);
  int num_total_primes = np.GetNumTotal();

  auto primes = param_.GetPrimeVector(np);

  HostVector<word> cx(num_total_primes);

  for (int i = 0; i < num_total_primes; i++) {
    BigInt prime(static_cast<uint64_t>(primes[i]));

    BigInt big_mod(static_cast<uint64_t>(0));
    BigInt::Mod(big_mod, big_int, prime);

    word result = big_mod.GetUnsigned();
    cx[i] = primeutil::ToMontgomery(result, primes[i]);
  }

  constant.ModifyNP(np);
  constant.SetScale(scale);
  CopyHostToDevice(constant.cx_, cx);
}

template <typename word>
Complex Encoder<word>::GetTwiddleFactor(int index) const {
  return twiddle_factors_[index];
}

template class Encoder<uint32_t>;
template class Encoder<uint64_t>;

}  // namespace cheddar
