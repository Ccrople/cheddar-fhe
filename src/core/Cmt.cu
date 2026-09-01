#include <algorithm>
#include <cstdlib>
#include <functional>
#include <unordered_map>

#include "common/Assert.h"
#include "common/Basic.cuh"
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
void CmtHandler<word>::CmtSerial(ConstContextPtr<word> context,
                                 std::vector<Ct> &res,
                                 const std::vector<Ct> &cts, int sub_degree,
                                 const EvkMap<word> &evk_map) const {
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

// ----- The batched path ----- //

namespace kernel {

// dst[z] = src_ptrs[z] * bank[mono_idx[z]]: the X^i of Adjust folded into the
// gather of the d input ciphertexts into one buffer. Ciphertext z, polynomial
// blockIdx.y, word i of its limbs. The product is CPAccum's single-term case
// (Context::Mult(Ct, Pt)): one MultMontgomery, so the same word.
template <typename word>
__global__ void CmtGather(int log_degree, word *dst, int ct_words,
                          int poly_words, const word *const *src_ptrs,
                          const word *bank, const int *mono_idx,
                          const word *primes,
                          const make_signed_t<word> *inv_primes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= poly_words) return;
  const int z = blockIdx.z;
  const int prime_index = i >> log_degree;
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const make_signed_t<word> inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  const word m = basic::StreamingLoad(bank + mono_idx[z] * poly_words + i);
  const word s = basic::StreamingLoad(src_ptrs[2 * z + blockIdx.y] + i);
  dst[z * ct_words + blockIdx.y * poly_words + i] =
      basic::MultMontgomery(m, s, prime, inv_prime);
}

// The inverse: dst_ptrs[z] = src[z] * bank[mono_idx[z]], the X^-j of
// invAdjust folded into the scatter back to d ciphertexts.
template <typename word>
__global__ void CmtScatter(int log_degree, word *const *dst_ptrs,
                           int ct_words, int poly_words, const word *src,
                           const word *bank, const int *mono_idx,
                           const word *primes,
                           const make_signed_t<word> *inv_primes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= poly_words) return;
  const int z = blockIdx.z;
  const int prime_index = i >> log_degree;
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const make_signed_t<word> inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  const word m = basic::StreamingLoad(bank + mono_idx[z] * poly_words + i);
  const word s =
      basic::StreamingLoad(src + z * ct_words + blockIdx.y * poly_words + i);
  dst_ptrs[2 * z + blockIdx.y][i] = basic::MultMontgomery(m, s, prime, inv_prime);
}

// One pass of TWEAK over every list of one size: pair z is (even, odd) ->
// (lo, hi) with twiddle bank[m]:
//   t = odd * X^e, lo = even + t, hi = even - t   (and both times `post`)
// -- TweakWorker's Mult / Sub / Add, whose words these are. `post` is the
// per-limb d^-1 the last pass of TWEAK(+1) applies (Cmt's MultConst), or
// null.
template <typename word>
__global__ void CmtButterfly(int log_degree, word *dst, const word *src,
                             int ct_words, int poly_words, const word *bank,
                             const int *pairs, const word *post,
                             const word *primes,
                             const make_signed_t<word> *inv_primes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= poly_words) return;
  const int *p = pairs + 5 * blockIdx.z;
  const int e = p[0], o = p[1], lo = p[2], hi = p[3], m = p[4];
  const int prime_index = i >> log_degree;
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const make_signed_t<word> inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  const int poly_off = blockIdx.y * poly_words + i;
  const word tw = basic::StreamingLoad(bank + m * poly_words + i);
  const word odd = basic::StreamingLoad(src + o * ct_words + poly_off);
  const word even = basic::StreamingLoad(src + e * ct_words + poly_off);
  const word t = basic::MultMontgomery(tw, odd, prime, inv_prime);
  word sum = basic::Add(even, t, prime);
  word diff = basic::Sub(even, t, prime);
  if (post != nullptr) {
    const word c = basic::StreamingLoadConst(post + prime_index);
    sum = basic::MultMontgomery(c, sum, prime, inv_prime);
    diff = basic::MultMontgomery(c, diff, prime, inv_prime);
  }
  dst[lo * ct_words + poly_off] = sum;
  dst[hi * ct_words + poly_off] = diff;
}

}  // namespace kernel

template <typename word>
struct CmtHandler<word>::BatchPlan {
  int level = 0;
  int sub_degree = 0;
  int d = 0;
  NPInfo np;
  // The monomial bank, one limb set per distinct exponent the call uses.
  DeviceVector<word> bank;
  std::map<int, int> bank_index;
  // Adjust's X^i and invAdjust's X^-j, as bank indices per ciphertext.
  DeviceVector<int> adjust_idx;
  DeviceVector<int> final_idx;
  // TWEAK(+1) and TWEAK(-1): one pair table per pass, d/2 pairs of
  // (even, odd, lo, hi, monomial), passes in order of list size 2 .. d. The
  // last pass of TWEAK(+1) writes through ScrambleAuto's input permutation.
  std::vector<DeviceVector<int>> pairs_plus;
  std::vector<DeviceVector<int>> pairs_minus;
  // d^-1 per limb, Montgomery form: the last pass of TWEAK(+1) applies it.
  DeviceVector<word> inv_d;
  // ScrambleAuto: the key index per t (-1 at the identity) and the
  // automorphism per t as PermuteBatch reads it.
  std::vector<int> key_index;
  DeviceVector<uint32_t> galois_factor;
  DeviceVector<uint32_t> galois_offset;
};

template <typename word>
const typename CmtHandler<word>::BatchPlan &CmtHandler<word>::GetPlan(
    int level, int sub_degree) const {
  auto it = plans_.find({level, sub_degree});
  if (it != plans_.end()) return *it->second;

  auto plan = std::make_shared<BatchPlan>();
  const int degree = param_.degree_;
  const int d = degree / sub_degree;
  const int two_n = 2 * degree;
  plan->level = level;
  plan->sub_degree = sub_degree;
  plan->d = d;
  plan->np = param_.LevelToNP(level, 0);
  const int num_total_primes = plan->np.GetNumTotal();
  const int poly_words = num_total_primes * degree;

  // 1. The butterfly structure: TweakWorker's recursion on slot indices, the
  //    pairs of every list of size m collected into pass log2(m) - 1.
  //    Recursion on a list L of length m, at parameter k: E = L[even
  //    positions], O = L[odd positions] hold the two sub-transforms (in their
  //    own slots) once the recursion returns, and the pass writes
  //    L[j] = E[j] + X^(2kj s) O[j], L[j + m/2] = E[j] - X^(2kj s) O[j].
  const int num_passes = [&] {
    int n = 0;
    for (int m = d; m > 1; m /= 2) n++;
    return n;
  }();
  struct Pair {
    int e, o, lo, hi, exponent;
  };
  std::vector<std::vector<Pair>> passes(num_passes);
  std::function<void(const std::vector<int> &, int)> recurse =
      [&](const std::vector<int> &L, int k) {
        const int m = static_cast<int>(L.size());
        if (m == 1) return;
        const int half = m / 2;
        std::vector<int> E, O;
        for (int j = 0; j < half; j++) {
          E.push_back(L[2 * j]);
          O.push_back(L[2 * j + 1]);
        }
        recurse(E, 2 * k);
        recurse(O, 2 * k);
        int pass = 0;
        for (int mm = 2; mm < m; mm *= 2) pass++;
        for (int j = 0; j < half; j++) {
          passes[pass].push_back({E[j], O[j], L[j], L[j + half], 2 * k * j});
        }
      };
  std::vector<int> all(d);
  for (int i = 0; i < d; i++) all[i] = i;
  recurse(all, sub_degree);
  for (const auto &pass : passes) {
    AssertTrue(static_cast<int>(pass.size()) == d / 2,
               "Cmt: a TWEAK pass does not cover every slot once");
  }

  // 2. ScrambleAuto: res[t] = cts[t*](X^(2kt+1)). Slot t of the batch that
  //    enters the key switches must hold logical ciphertext t*, so the last
  //    TWEAK(+1) pass writes logical index L to slot phi(L) with t*(phi(L)) =
  //    L.
  const std::unordered_map<int, int> table = GaloisIndexTable();
  std::vector<int> phi(d, -1);
  plan->key_index.assign(d, -1);
  HostVector<uint32_t> gf(d), go(d);
  for (int t = 0; t < d; t++) {
    const int galois_factor = (2 * sub_degree * t + 1) % two_n;
    const int inverse = InvModPowerOfTwo(galois_factor, two_n);
    AssertTrue((inverse - 1) % (2 * sub_degree) == 0,
               "Cmt: the inverse left the subgroup");
    const int t_star = ((inverse - 1) / (2 * sub_degree)) % d;
    AssertTrue(phi[t_star] == -1, "Cmt: t -> t* is not a permutation");
    phi[t_star] = t;
    if (galois_factor == 1) {
      gf[t] = 1;
      go[t] = 0;
      continue;
    }
    const auto it = table.find(galois_factor);
    AssertTrue(it != table.end(), "Cmt: 2kt+1 is not a power of 5 modulo 2N");
    const int index = it->second;
    plan->key_index[t] = index;
    // Context::Permute(res, switched, index): the automorphism of rotation
    // index `index` (the ciphertext carries degree / 2 slots after the first
    // monomial product, so the reduction modulo the slot count is void).
    const int factor = param_.GetGaloisFactor(index);
    gf[t] = static_cast<uint32_t>(factor);
    go[t] = static_cast<uint32_t>(param_.GetGaloisOffset(factor));
  }
  CopyHostToDevice(plan->galois_factor, gf);
  CopyHostToDevice(plan->galois_offset, go);

  // 3. The exponents the call uses, and the bank.
  auto index_of = [&](int exponent) {
    auto f = plan->bank_index.find(exponent);
    if (f != plan->bank_index.end()) return f->second;
    const int idx = static_cast<int>(plan->bank_index.size());
    plan->bank_index.emplace(exponent, idx);
    return idx;
  };
  HostVector<int> adjust(d), final_ids(d);
  for (int i = 0; i < d; i++) adjust[i] = index_of(i);
  for (int j = 0; j < d; j++) final_ids[j] = index_of(-j);
  std::vector<HostVector<int>> plus(num_passes), minus(num_passes);
  for (int pass = 0; pass < num_passes; pass++) {
    const bool last = (pass == num_passes - 1);
    for (const Pair &pr : passes[pass]) {
      const int lo = last ? phi[pr.lo] : pr.lo;
      const int hi = last ? phi[pr.hi] : pr.hi;
      for (int v : {pr.e, pr.o, lo, hi, index_of(pr.exponent)}) {
        plus[pass].push_back(v);
      }
      for (int v : {pr.e, pr.o, pr.lo, pr.hi, index_of(-pr.exponent)}) {
        minus[pass].push_back(v);
      }
    }
  }
  const int num_monomials = static_cast<int>(plan->bank_index.size());
  plan->bank.resize(static_cast<size_t>(num_monomials) * poly_words);
  for (const auto &[exponent, idx] : plan->bank_index) {
    Pt monomial;
    EncodeMonomial(monomial, level, exponent);
    AssertTrue(static_cast<int>(monomial.mx_.size()) == poly_words,
               "Cmt: monomial size mismatch");
    cudaMemcpyAsync(plan->bank.data() + static_cast<size_t>(idx) * poly_words,
                    monomial.mx_.data(), poly_words * sizeof(word),
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  }
  CopyHostToDevice(plan->adjust_idx, adjust);
  CopyHostToDevice(plan->final_idx, final_ids);
  plan->pairs_plus.resize(num_passes);
  plan->pairs_minus.resize(num_passes);
  for (int pass = 0; pass < num_passes; pass++) {
    CopyHostToDevice(plan->pairs_plus[pass], plus[pass]);
    CopyHostToDevice(plan->pairs_minus[pass], minus[pass]);
  }

  // 4. d^-1 per limb, as Cmt builds its Constant.
  {
    const auto primes = param_.GetPrimeVector(plan->np);
    HostVector<word> host(num_total_primes);
    for (int i = 0; i < num_total_primes; i++) {
      host[i] = primeutil::ToMontgomery<word>(
          primeutil::InvMod<word>(static_cast<word>(d), primes[i]), primes[i]);
    }
    CopyHostToDevice(plan->inv_d, host);
  }

  return *plans_.emplace(std::make_pair(level, sub_degree), plan)
              .first->second;
}

template <typename word>
void CmtHandler<word>::CmtBatched(ConstContextPtr<word> context,
                                  std::vector<Ct> &res,
                                  const std::vector<Ct> &cts, int sub_degree,
                                  const EvkMap<word> &evk_map) const {
  const int degree = param_.degree_;
  const int d = static_cast<int>(cts.size());
  const NPInfo np = cts.at(0).GetNP();
  const int level = param_.NPToLevel(np);
  const double scale = cts.at(0).GetScale();
  for (const auto &ct : cts) {
    AssertTrue(ct.GetNP() == np, "Cmt: ciphertexts differ in NP");
    AssertTrue(!ct.HasRx(), "Cmt: ciphertexts must not carry an rx_ part");
    AssertTrue(ct.GetScale() == scale, "Cmt: ciphertexts differ in scale");
  }
  for (const auto &r : res) {
    for (const auto &c : cts) {
      AssertTrue(&r != &c, "Cmt: in-place operation is not supported");
    }
  }
  const BatchPlan &plan = GetPlan(level, sub_degree);
  AssertTrue(plan.d == d, "Cmt: plan shape mismatch");

  const int poly_words = np.GetNumTotal() * degree;
  const int ct_words = 2 * poly_words;
  const word *primes = param_.GetPrimesPtr(np);
  const auto *inv_primes = param_.GetInvPrimesPtr(np);
  constexpr int block = 256;
  const dim3 grid(DivCeil(poly_words, block), 2, d);
  const int num_passes = static_cast<int>(plan.pairs_plus.size());

  DeviceVector<word> buf_a(d * ct_words), buf_b(d * ct_words);
  word *cur = buf_a.data();
  word *nxt = buf_b.data();

  // 1. Gather with Adjust's X^i.
  {
    HostVector<uint64_t> ptrs(2 * d);
    for (int i = 0; i < d; i++) {
      ptrs[2 * i] = reinterpret_cast<uint64_t>(cts[i].bx_.data());
      ptrs[2 * i + 1] = reinterpret_cast<uint64_t>(cts[i].ax_.data());
    }
    DeviceVector<uint64_t> ptrs_dev;
    CopyHostToDevice(ptrs_dev, ptrs);
    kernel::CmtGather<word><<<grid, block>>>(
        param_.log_degree_, cur, ct_words, poly_words,
        reinterpret_cast<const word *const *>(ptrs_dev.data()),
        plan.bank.data(), plan.adjust_idx.data(), primes, inv_primes);
  }

  // 2. TWEAK(+1), d^-1 on the last pass, which also lays the slots out in
  //    ScrambleAuto's input order.
  for (int pass = 0; pass < num_passes; pass++) {
    const bool last = (pass == num_passes - 1);
    kernel::CmtButterfly<word><<<dim3(DivCeil(poly_words, block), 2, d / 2),
                                 block>>>(
        param_.log_degree_, nxt, cur, ct_words, poly_words, plan.bank.data(),
        plan.pairs_plus[pass].data(), last ? plan.inv_d.data() : nullptr,
        primes, inv_primes);
    std::swap(cur, nxt);
  }

  // 3. ScrambleAuto: slot 0 is the identity and is copied; slots 1 .. d-1 are
  //    switched with their own keys in one group, then every slot is
  //    permuted by its automorphism.
  {
    std::vector<const Evk *> keys;
    keys.reserve(d - 1);
    for (int t = 1; t < d; t++) {
      AssertTrue(plan.key_index[t] >= 0, "Cmt: the identity is not at t = 0");
      keys.push_back(&evk_map.GetRotationKey(plan.key_index[t]));
    }
    AssertTrue(plan.key_index[0] == -1, "Cmt: t = 0 is not the identity");
    cudaMemcpyAsync(nxt, cur, static_cast<size_t>(ct_words) * sizeof(word),
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    context->MultKeyBatch(nxt + ct_words, ct_words, cur + ct_words, ct_words,
                          np, keys, d - 1);
    std::swap(cur, nxt);
    context->elem_handler_.PermuteBatch(nxt, cur, ct_words, poly_words, 2, np,
                                        plan.galois_factor.data(),
                                        plan.galois_offset.data(), d);
    std::swap(cur, nxt);
  }

  // 4. TWEAK(-1).
  for (int pass = 0; pass < num_passes; pass++) {
    kernel::CmtButterfly<word><<<dim3(DivCeil(poly_words, block), 2, d / 2),
                                 block>>>(
        param_.log_degree_, nxt, cur, ct_words, poly_words, plan.bank.data(),
        plan.pairs_minus[pass].data(), nullptr, primes, inv_primes);
    std::swap(cur, nxt);
  }

  // 5. Scatter with invAdjust's X^-j. The serial path's outputs carry
  //    degree / 2 slots (Mult(Ct, Pt) takes the larger count and the
  //    monomial's is degree / 2); so do these.
  res.resize(d);
  {
    HostVector<uint64_t> ptrs(2 * d);
    for (int j = 0; j < d; j++) {
      Ct &r = res[j];
      r.RemoveRx();
      r.ModifyNP(np);
      r.SetScale(scale);
      r.SetNumSlots(Max(cts[j].GetNumSlots(), degree / 2));
      ptrs[2 * j] = reinterpret_cast<uint64_t>(r.bx_.data());
      ptrs[2 * j + 1] = reinterpret_cast<uint64_t>(r.ax_.data());
    }
    DeviceVector<uint64_t> ptrs_dev;
    CopyHostToDevice(ptrs_dev, ptrs);
    kernel::CmtScatter<word><<<grid, block>>>(
        param_.log_degree_, reinterpret_cast<word *const *>(ptrs_dev.data()),
        ct_words, poly_words, cur, plan.bank.data(), plan.final_idx.data(),
        primes, inv_primes);
  }
}

template <typename word>
bool CmtHandler<word>::BatchEnabled() {
  const char *env = std::getenv("CHEDDAR_CMT_SERIAL");
  return !(env != nullptr && env[0] == '1');
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
  if (!BatchEnabled() || param_.conjugate_invariant_ || d < 2) {
    CmtSerial(context, res, cts, sub_degree, evk_map);
    return;
  }
  CmtBatched(context, res, cts, sub_degree, evk_map);
}

template class CmtHandler<uint32_t>;
template class CmtHandler<uint64_t>;

}  // namespace cheddar
