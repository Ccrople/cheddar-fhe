#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/Cmt.h"

namespace cheddar {

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

template class CmtHandler<uint32_t>;
template class CmtHandler<uint64_t>;

}  // namespace cheddar
