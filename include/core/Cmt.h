#pragma once

#include <map>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/NPInfo.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Ciphertext matrix transpose (CMT) of Cheon, Kang and Lee (ePrint
 * 2025/1957), section 4.2 -- and the monomial arithmetic it is built out of.
 *
 * CMT turns a column-wise subring matrix encryption into a row-wise one, which
 * is what makes the block products of the batch CC-MM line up. [KANG]
 * Algorithm 3 runs it in five stages:
 *
 *     X^i * ct  ->  TWEAK(+1)  ->  d^-1 and Auto(.; 2kt+1)  ->  TWEAK(-1)
 *               ->  X^-i * ct
 *
 * ## What is implemented here
 *
 * The two stages that need **no key material and no multiplicative level**:
 * monomial multiplication, and `TWEAK` ([KANG] Algorithm 2). Together they are
 * the whole `O(dN log d)` half of CMT.
 *
 * `ScrambleAuto` -- the automorphisms `X -> X^(2kt+1)`, and with them the d key
 * switchings that dominate CMT's cost -- is **not** here yet. It is separated
 * on purpose: the key inventory is a real decision (see below) and it should be
 * made against measurements rather than folded into a transform that has
 * nothing to do with it.
 *
 * ## TWEAK
 *
 * `TWEAK^N_k({m_j}, s)` produces `ct'_i = sum_j X^(2*k*i*j*s) * m_j`, a
 * length-d Cooley-Tukey transform over ciphertexts whose twiddles are
 * monomials, with `d = N/k`. The recursion halves d and doubles k:
 *
 *     aux_even <- TWEAK_{2k}(cts at even j),  aux_odd <- TWEAK_{2k}(cts at odd)
 *     ct'_j        = aux_even_j + X^(2kj*s) * aux_odd_j
 *     ct'_{j+d/2}  = aux_even_j - X^(2kj*s) * aux_odd_j
 *
 * The minus sign is not a convention: at `i = j + d/2` the twiddle picks up
 * `X^(k*d) = X^N = -1`, and the even/odd sub-transforms are unchanged because
 * they pick up `X^(2N) = 1`. The negacyclic ring supplies both.
 *
 * `X^(2k)` has order exactly d in the ring, so `TWEAK(-1) . TWEAK(+1) = d`,
 * which is why Algorithm 3 carries a `d^-1`. That round trip is checked
 * directly rather than assumed.
 *
 * ## Monomial multiplication costs nothing, and why
 *
 * `X^e` is encoded as a plaintext holding a single coefficient of `+-1` at
 * scale **1**, so `Context::Mult(Ct, Pt)` -- which multiplies the scales and
 * does not rescale -- returns the product at the input's own scale and level.
 * The plaintext is exact: no BigInt, no rounding, just a unit RNS limb, so the
 * only error TWEAK introduces is the ciphertext noise growth of a
 * multiplication by a polynomial of norm 1.
 *
 * Cheddar already had one instance of this (`MultImaginaryUnit` is
 * multiplication by `X^(N/2)` against a precomputed NTT-domain constant); this
 * is the same idea for a general exponent, with the negacyclic wrap
 * `X^(e+N) = -X^e` handled when the monomial is built.
 *
 * ## The key decision that is deliberately still open
 *
 * `ScrambleAuto` needs the automorphisms `X -> X^(2kt+1)` for `t` in `[d)`.
 * Those form the subgroup `H = <5^(k/2)>` of order d, so there are two ways to
 * supply them:
 *
 *   * **d automorphism keys**, one per t: d key switchings total.
 *   * **one master key** for `X -> X^(5^(k/2))`, applied t times to reach the
 *     t-th element: one key, but `O(d^2)` key switchings.
 *
 * [KANG] section 4.3 takes the second. That is the right default *here* too --
 * Cheddar cannot serialize keys, so every key is regenerated per process, and
 * a rotation key was measured at ~300 ms and ~126 MiB (`Doing.md` section 2) --
 * but at `d = 32` it is 496 key switchings against 32, which is not obviously
 * the cheaper end. **The choice should be measured, not assumed**, and nothing
 * in this file presumes either.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CmtHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

  const Parameter<word> &param_;
  const NTTHandler<word> &ntt_handler_;

  // Monomials are reused across the recursion, and each one costs an NTT.
  using MonomialCache = std::map<int, Pt>;

  const Pt &GetMonomial(MonomialCache &cache, int level, int exponent) const;

  void TweakWorker(ConstContextPtr<word> context, std::vector<Ct> &cts,
                   int sub_degree, int sign, int level,
                   MonomialCache &cache) const;

 public:
  CmtHandler(const Parameter<word> &param, const NTTHandler<word> &ntt_handler);

  // disable copying (or moving also)
  CmtHandler(const CmtHandler &) = delete;
  CmtHandler &operator=(const CmtHandler &) = delete;

  /**
   * @brief Encode the monomial `X^exponent` as an exact plaintext of scale 1.
   *
   * The exponent may be negative or beyond the ring degree; it is reduced
   * modulo 2N and `X^(e+N) = -X^e` supplies the sign. Multiplying by this
   * plaintext preserves both the scale and the level of a ciphertext.
   *
   * @param res output plaintext (NTT-applied, as for EncodeCoeff)
   * @param level level to encode at
   * @param exponent the power of X, any integer
   * @param num_aux number of auxiliary primes (default: 0 --> none)
   */
  void EncodeMonomial(Pt &res, int level, int exponent,
                      int num_aux = 0) const;

  /**
   * @brief [KANG] Algorithm 2: `res[i] = sum_j X^(2*k*i*j*sign) * cts[j]`.
   *
   * Consumes no level and uses no key. The number of input ciphertexts must be
   * `degree / sub_degree`, which is what the recursion's halving of d against
   * its doubling of k requires.
   *
   * @param context the CKKS context
   * @param res output, resized to `cts.size()`; may not alias `cts`
   * @param cts input ciphertexts, all at one level and without an rx_ part
   * @param sub_degree k
   * @param sign +1 or -1
   */
  void Tweak(ConstContextPtr<word> context, std::vector<Ct> &res,
             const std::vector<Ct> &cts, int sub_degree, int sign) const;
};

}  // namespace cheddar
