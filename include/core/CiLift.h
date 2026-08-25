#pragma once

#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/DeviceVector.h"
#include "core/NPInfo.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief The keyless embedding of a conjugate-invariant ring into the
 * ordinary cyclotomic ring of the same conductor, and the keyless way back.
 *
 * ## Why this exists (Doing.md 1.5bk)
 *
 * R+ of degree n has conductor 4n, and the ordinary ring of that conductor
 * is the cyclotomic ring of degree N = 2n: R+ IS its real subring, with
 * c_j = X^j + X^-j = X^j - X^{N-j}. [KANG]'s batch CC-MM machinery -- TWEAK's
 * monomial twiddles and the CMT built on them -- has no conjugate-invariant
 * form: inside R+ the automorphisms that fix the CC-MM's subring act on the
 * module pairs (i, d-i) jointly, so every projection they can build reaches
 * only the pair-symmetric combination, and separating a pair needs the odd
 * part X^a - X^-a -- exactly what the totally real ring quotients away. So
 * the CC-MM stretch lifts: run [KANG] on the degree-N ordinary ring, where
 * monomials and the full automorphism group exist, and come back down.
 *
 * ## Both directions are keyless
 *
 * **Lift.** The inclusion is a ring homomorphism, so a ciphertext (b, a)
 * over R+ with secret s satisfies b + a*s = m verbatim in the big ring under
 * the embedded secret LiftSecret(s): reindex the coefficients and transform.
 * No key, no noise, no security statement beyond the conjugate-invariant
 * scheme's own -- the embedded secret in the degree-N ring IS how CI
 * security is defined, and nothing is published.
 *
 * **Descend.** With T = id + conj, the trace: conj is an automorphism of
 * the big ring that fixes both the embedded secret and any lifted message,
 * so conj(b) + conj(a)*s = m alongside b + a*s = m, and summing,
 * T(b) + T(a)*s = 2m. On coefficients T composed with the read-off of the
 * c-basis is one linear map -- out[j] = in[j] - in[N-j] -- so the pair
 * (T(b), T(a)) reinterprets at degree n as a conjugate-invariant ciphertext
 * of m under s at TWICE the recorded scale. No key switch: conjugation
 * ordinarily moves the secret, and the embedded secret is its own
 * conjugate. A message with an odd part loses it -- T/2 is the even-part
 * projection -- and position N/2 is annihilated (c_n = 0 in the big ring).
 * Any product of lifted ciphertexts has a purely even message, the real
 * subring being closed under multiplication, so the lift -> multiply ->
 * descend sandwich is exact.
 *
 * Why the trace and not the average: E = T/2 is the same projection of the
 * ELEMENT, but 2^-1 mod q turns an odd small integer into a residue near
 * q/2, so E does not preserve coefficient size. The ciphertext noise has
 * odd antisymmetric parts in about half its positions; E of it puts ~q/2
 * there and the descent decrypts to garbage -- while on lifted ciphertexts,
 * whose components are exactly mirror-antisymmetric, the halving collapses
 * to the identity and the defect is invisible. The factor 2 rides the
 * recorded scale instead, where it is exact (Doing.md 1.5bk).
 *
 * ## What it costs
 *
 * Per direction: one INTT at the source degree, one reindexing kernel, one
 * NTT at the target degree. The two parameter sets must share their prime
 * chain -- 1 mod 4n and 1 mod 2N are the same condition at N = 2n, so the
 * same primes serve both rings (`ci12_35` / `ringdegree13_35` are the
 * shipped pair).
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiLiftHandler {
 private:
  using Ct = Ciphertext<word>;

  ConstContextPtr<word> ci_;
  ConstContextPtr<word> big_;

  static constexpr int kernel_block_dim_ = 256;

 public:
  /**
   * @param ci the conjugate-invariant Context, degree n
   * @param big the ordinary Context of the same conductor, degree 2n,
   * holding the same prime chain at every level
   */
  CiLiftHandler(ConstContextPtr<word> ci, ConstContextPtr<word> big);

  // disable copying (or moving also)
  CiLiftHandler(const CiLiftHandler &) = delete;
  CiLiftHandler &operator=(const CiLiftHandler &) = delete;

  /**
   * @brief Embed one conjugate-invariant ciphertext into the ordinary ring:
   * coefficient c_j becomes X^j - X^{N-j}. Keyless and exact; the result
   * decrypts under the embedded secret, LiftSecret of the CI one.
   *
   * @param res output, a ciphertext of the big Context at the same level
   * @param ct input, at the CI Context, no aux primes and no rx_ part
   */
  void Lift(Ct &res, const Ct &ct) const;

  /**
   * @brief The keyless way back: the trace T = id + conj on both components
   * fused with the c-basis read-off, reinterpreted at degree n, at twice
   * the input's recorded scale. Exact on ciphertexts whose message lies in
   * the lifted image; a message with an odd part loses it to the even-part
   * projection.
   *
   * @param res output, a ciphertext of the CI Context at the same level,
   * with scale 2x the input's
   * @param ct input, at the big Context under the embedded secret, no aux
   * primes and no rx_ part
   */
  void Descend(Ct &res, const Ct &ct) const;

  /**
   * @brief The embedded secret, for building the big ring's UserInterface:
   * position j's coefficient lands at j and negated at N - j. Ternary in,
   * ternary out, twice the Hamming weight.
   */
  static std::vector<int> LiftSecret(const std::vector<int> &ci_secret);
};

}  // namespace cheddar
