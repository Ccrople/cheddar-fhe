#include <unordered_map>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"
#include "core/Cmt.h"

namespace cheddar {

namespace {

// Inverse modulo a power of two, by Newton iteration: each step doubles the
// number of correct low bits, and x = 1 is already correct modulo 2 for odd a.
int InvModPowerOfTwo(int a, int modulus) {
  long long x = 1;
  for (int i = 0; i < 32; i++) {
    x = (x * (2 - static_cast<long long>(a) * x)) % modulus;
    if (x < 0) x += modulus;
  }
  return static_cast<int>(x);
}

}  // namespace

template <typename word>
CmtHandler<word>::CmtHandler(const Parameter<word> &param,
                             const NTTHandler<word> &ntt_handler)
    : param_{param}, ntt_handler_{ntt_handler} {}

template <typename word>
void CmtHandler<word>::EncodeMonomial(Pt &res, int level, int exponent,
                                      int num_aux /*= 0*/) const {
  const int degree = param_.degree_;

  // X^(e + N) = -X^e, so every exponent reduces to a single signed coefficient.
  int reduced = exponent % (2 * degree);
  if (reduced < 0) reduced += 2 * degree;
  const bool negate = reduced >= degree;
  const int position = negate ? reduced - degree : reduced;

  const NPInfo np = param_.LevelToNP(level, num_aux);
  const auto primes = param_.GetPrimeVector(np);
  const int num_total_primes = np.GetNumTotal();

  // Built limb by limb rather than through EncodeCoeff: the coefficient is
  // exactly +-1, so there is nothing for BigInt to do and the result is exact
  // by construction.
  HostVector<word> mx(num_total_primes * degree, 0);
  for (int j = 0; j < num_total_primes; j++) {
    mx[j * degree + position] = negate ? primes[j] - 1 : 1;
  }

  res.ModifyNP(np);
  res.SetNumSlots(degree / 2);
  res.SetScale(1.0);
  CopyHostToDevice(res.mx_, mx);

  auto view = res.View();
  ntt_handler_.NTT(view, np, res.ConstView(), true);
}

template <typename word>
const Plaintext<word> &CmtHandler<word>::GetMonomial(MonomialCache &cache,
                                                     int level,
                                                     int exponent) const {
  auto it = cache.find(exponent);
  if (it != cache.end()) return it->second;

  Pt monomial;
  EncodeMonomial(monomial, level, exponent);
  return cache.emplace(exponent, std::move(monomial)).first->second;
}

// The Cooley-Tukey body. `cts` is transformed in place, which keeps the number
// of live ciphertexts at one level's worth rather than one per recursion depth.
template <typename word>
void CmtHandler<word>::TweakWorker(ConstContextPtr<word> context,
                                   std::vector<Ct> &cts, int sub_degree,
                                   int sign, int level,
                                   MonomialCache &cache) const {
  const int d = static_cast<int>(cts.size());
  if (d == 1) return;

  const int half = d / 2;

  // Split by parity of j, recurse with d halved and k doubled -- the twiddle
  // X^(4*k*i*j) of the sub-transform is exactly TWEAK's own at parameter 2k.
  std::vector<Ct> even, odd;
  even.reserve(half);
  odd.reserve(half);
  for (int j = 0; j < half; j++) {
    even.push_back(std::move(cts[2 * j]));
    odd.push_back(std::move(cts[2 * j + 1]));
  }

  TweakWorker(context, even, sub_degree * 2, sign, level, cache);
  TweakWorker(context, odd, sub_degree * 2, sign, level, cache);

  Ct twisted;
  for (int j = 0; j < half; j++) {
    const int exponent = 2 * sub_degree * j * sign;
    context->Mult(twisted, odd[j], GetMonomial(cache, level, exponent));
    // At i = j + d/2 the twiddle gains X^(k*d) = X^N = -1, which is the whole
    // of the butterfly's minus sign.
    context->Sub(cts[j + half], even[j], twisted);
    context->Add(cts[j], even[j], twisted);
  }
}

template <typename word>
void CmtHandler<word>::Tweak(ConstContextPtr<word> context,
                             std::vector<Ct> &res, const std::vector<Ct> &cts,
                             int sub_degree, int sign) const {
  const int degree = param_.degree_;
  const int d = static_cast<int>(cts.size());

  AssertTrue(sign == 1 || sign == -1, "Tweak: sign must be +1 or -1");
  AssertTrue(sub_degree >= 1 && sub_degree <= degree &&
                 IsPowOfTwo(sub_degree) && degree / sub_degree == d,
             "Tweak: the number of ciphertexts must be degree / sub_degree");
  AssertTrue(d >= 1, "Tweak: no input ciphertexts");

  const NPInfo np = cts.at(0).GetNP();
  const double scale = cts.at(0).GetScale();
  for (const auto &ct : cts) {
    AssertTrue(ct.GetNP() == np, "Tweak: ciphertexts differ in NP");
    AssertTrue(!ct.HasRx(),
               "Tweak: ciphertexts must not carry an rx_ part");
  }
  for (const auto &r : res) {
    for (const auto &c : cts) {
      AssertTrue(&r != &c, "Tweak: in-place operation is not supported");
    }
  }

  const int level = param_.NPToLevel(np);
  AssertTrue(level >= 0, "Tweak: inputs are not at a valid level");

  res.resize(d);
  for (int j = 0; j < d; j++) {
    context->Copy(res[j], cts[j]);
  }

  MonomialCache cache;
  TweakWorker(context, res, sub_degree, sign, level, cache);

  // Monomial multiplication changes neither, but say so where it is checkable.
  for (auto &r : res) {
    AssertTrue(r.GetNP() == np, "Tweak: the transform changed the level");
    context->AssertSameScale(r, scale);
  }
}

template <typename word>
std::unordered_map<int, int> CmtHandler<word>::GaloisIndexTable() const {
  const int half_degree = param_.degree_ / 2;
  std::unordered_map<int, int> table;
  table.reserve(half_degree * 2);
  for (int i = 0; i < half_degree; i++) {
    table.emplace(param_.GetGaloisFactor(i), i);
  }
  return table;
}

template <typename word>
std::vector<int> CmtHandler<word>::ScrambleAutoRotationIndices(
    int sub_degree) const {
  const int degree = param_.degree_;
  AssertTrue(sub_degree >= 1 && sub_degree <= degree && IsPowOfTwo(sub_degree),
             "ScrambleAutoRotationIndices: invalid sub_degree");
  const int d = degree / sub_degree;
  const int two_n = 2 * degree;

  const std::unordered_map<int, int> table = GaloisIndexTable();
  std::vector<int> indices;
  indices.reserve(d > 0 ? d - 1 : 0);
  for (int t = 1; t < d; t++) {
    const int galois_factor = (2 * sub_degree * t + 1) % two_n;
    const auto it = table.find(galois_factor);
    AssertTrue(it != table.end(),
               "ScrambleAutoRotationIndices: 2kt+1 is not a power of 5");
    indices.push_back(it->second);
  }
  return indices;
}

template <typename word>
void CmtHandler<word>::ScrambleAuto(ConstContextPtr<word> context,
                                    std::vector<Ct> &res,
                                    const std::vector<Ct> &cts, int sub_degree,
                                    const EvkMap<word> &evk_map) const {
  const int degree = param_.degree_;
  const int d = static_cast<int>(cts.size());
  AssertTrue(sub_degree >= 1 && sub_degree <= degree &&
                 IsPowOfTwo(sub_degree) && degree / sub_degree == d,
             "ScrambleAuto: the number of ciphertexts must be degree / "
             "sub_degree");

  const NPInfo np = cts.at(0).GetNP();
  for (const auto &ct : cts) {
    AssertTrue(ct.GetNP() == np, "ScrambleAuto: ciphertexts differ in NP");
    AssertTrue(!ct.HasRx(),
               "ScrambleAuto: ciphertexts must not carry an rx_ part");
  }
  for (const auto &r : res) {
    for (const auto &c : cts) {
      AssertTrue(&r != &c,
                 "ScrambleAuto: in-place operation is not supported");
    }
  }

  const int two_n = 2 * degree;
  const std::unordered_map<int, int> table = GaloisIndexTable();
  res.resize(d);

  Ct switched;
  for (int t = 0; t < d; t++) {
    const int galois_factor = (2 * sub_degree * t + 1) % two_n;
    const int inverse = InvModPowerOfTwo(galois_factor, two_n);

    // The inverse lies in the same subgroup, so it is 2k*t_star + 1 and the
    // division below is exact. Asserted rather than assumed: it is the one
    // step where a wrong subgroup would go unnoticed.
    AssertTrue((inverse - 1) % (2 * sub_degree) == 0,
               "ScrambleAuto: the inverse left the subgroup");
    const int t_star = ((inverse - 1) / (2 * sub_degree)) % d;

    if (galois_factor == 1) {
      // t = 0 is the identity automorphism, so it needs no key switch.
      context->Copy(res[t], cts[t_star]);
      continue;
    }

    const auto it = table.find(galois_factor);
    AssertTrue(it != table.end(),
               "ScrambleAuto: 2kt+1 is not a power of 5 modulo 2N");
    const int index = it->second;
    // MultKey then Permute rather than HRot: HRot reduces its argument modulo
    // the slot count, which is a rotation distance and not what this index is.
    context->MultKey(switched, cts[t_star], evk_map.GetRotationKey(index));
    context->Permute(res[t], switched, index);
  }
}

template <typename word>
void CmtHandler<word>::Cmt(ConstContextPtr<word> context,
                           std::vector<Ct> &res, const std::vector<Ct> &cts,
                           int sub_degree, const EvkMap<word> &evk_map) const {
  const int degree = param_.degree_;
  const int d = static_cast<int>(cts.size());
  AssertTrue(sub_degree >= 1 && sub_degree <= degree &&
                 IsPowOfTwo(sub_degree) && degree / sub_degree == d,
             "Cmt: the number of ciphertexts must be degree / sub_degree");

  const NPInfo np = cts.at(0).GetNP();
  const int level = param_.NPToLevel(np);
  AssertTrue(level >= 0, "Cmt: inputs are not at a valid level");

  MonomialCache cache;

  // Adjust, first half: X^i * ct_i. Together with TWEAK's X^(2kij) twiddle
  // this is Adjust's X^(i(2kt+1)), since i(2kt+1) = 2kit + i.
  std::vector<Ct> staged(d);
  for (int i = 0; i < d; i++) {
    context->Mult(staged[i], cts[i], GetMonomial(cache, level, i));
  }

  std::vector<Ct> transformed;
  Tweak(context, transformed, staged, sub_degree, 1);

  // Adjust, second half: d^-1, as [KANG] Algorithm 3 line 5 writes it -- the
  // inverse of d modulo each prime, which exists because d is a power of two
  // and the primes are odd. That divides the message exactly, at scale 1, so
  // it costs no level and leaves the scale where it was.
  //
  // Relabelling the scale instead would decode to the same value, but it
  // leaves the recorded scale d times larger, and the batch CC-MM composes
  // three of these before a single rescale.
  Constant<word> inv_d;
  {
    const auto primes = param_.GetPrimeVector(np);
    const int num_total_primes = np.GetNumTotal();
    HostVector<word> host(num_total_primes);
    for (int i = 0; i < num_total_primes; i++) {
      host[i] = primeutil::ToMontgomery<word>(
          primeutil::InvMod<word>(static_cast<word>(d), primes[i]), primes[i]);
    }
    inv_d.ModifyNP(np);
    inv_d.SetScale(1.0);
    CopyHostToDevice(inv_d.cx_, host);
  }
  Ct scaled;
  for (auto &ct : transformed) {
    context->Mult(scaled, ct, inv_d);
    ct = std::move(scaled);
  }

  std::vector<Ct> scrambled;
  ScrambleAuto(context, scrambled, transformed, sub_degree, evk_map);

  std::vector<Ct> untransformed;
  Tweak(context, untransformed, scrambled, sub_degree, -1);

  // invAdjust, second half: X^-j.
  res.resize(d);
  for (int j = 0; j < d; j++) {
    context->Mult(res[j], untransformed[j], GetMonomial(cache, level, -j));
  }

  for (const auto &r : res) {
    AssertTrue(r.GetNP() == np, "Cmt: the transform changed the level");
  }
}

template class CmtHandler<uint32_t>;
template class CmtHandler<uint64_t>;

}  // namespace cheddar
