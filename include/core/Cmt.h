#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
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
 * All of it: monomial multiplication, `TWEAK` ([KANG] Algorithm 2),
 * `ScrambleAuto`, and Algorithm 3 on top of them.
 *
 * **The whole of CMT is level-free.** TWEAK is monomial multiplications and
 * additions, ScrambleAuto is key switches and permutations, and the `d^-1` is
 * a change to the recorded scale rather than an operation. That matters
 * because the batch CC-MM spends this ring's only multiplicative level on its
 * tensor product, so a CMT that cost one would leave the algorithm with
 * nowhere to go.
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
 * [KANG] section 4.3 takes the second, because it cares about key memory.
 * **This implementation takes the first**, which is the cost model Algorithm 3
 * is stated against ("d key switchings, all in ScrambleAuto"), and it is the
 * cheaper end here: at `d = 32` the master key would be 496 key switchings
 * against 32, and at this ring a key is a few hundred KiB rather than the
 * ~126 MiB measured at log_degree 16 (`Doing.md` section 2). The d keys cost
 * about 12 MiB in total.
 *
 * That is a judgement, not a measurement -- the timing has not been taken. If
 * key memory ever binds (it would at log_degree 16, or at a larger d), the
 * master-key variant replaces `ScrambleAuto` alone and nothing else in this
 * file changes. `ScrambleAutoRotationIndices` exists so a caller can see the
 * inventory it is being asked for.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CmtHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  const Parameter<word> &param_;
  const NTTHandler<word> &ntt_handler_;

  // Monomials are reused across the recursion, and each one costs an NTT.
  using MonomialCache = std::map<int, Pt>;

  const Pt &GetMonomial(MonomialCache &cache, int level, int exponent) const;

  void TweakWorker(ConstContextPtr<word> context, std::vector<Ct> &cts,
                   int sub_degree, int sign, int level,
                   MonomialCache &cache) const;

  // factor -> i with 5^i == factor (mod 2N), read out of Parameter's own table
  // so the convention is the library one rather than a re-derivation. Built in
  // a single pass and reused: a scan per lookup would be d passes over N/2
  // entries, which is the same order as a key switch and would be the only
  // host-side cost in ScrambleAuto worth naming.
  std::unordered_map<int, int> GaloisIndexTable() const;

  // ----- The batched path ----- //
  //
  // One Cmt call is d ciphertexts through the same five stages with the same
  // monomials and the same d automorphisms, and every stage is a map over
  // the ciphertext index that touches each ciphertext once. Run
  // ciphertext by ciphertext that is ~6.5k launches of ~10 us kernels per
  // call at d = 128 (Doing 3.21: 78k launches a layer, 22 % busy); run over
  // the d ciphertexts laid out back to back in one buffer it is ~40 launches
  // with the ciphertext index on blockIdx.z. The words are the same: every
  // stage is modular arithmetic through the same kernels (the key switch)
  // or the same per-element operations (the monomials and the butterflies),
  // and a modular sum does not care what order its terms arrive in.
  //
  // A plan holds what depends only on (level, sub_degree): the monomial bank
  // (built once, not per call as the serial path's cache is), TWEAK's
  // butterfly tables per pass, the automorphisms and the keys they ask for.
  struct BatchPlan;
  mutable std::map<std::pair<int, int>, std::shared_ptr<BatchPlan>> plans_;
  const BatchPlan &GetPlan(int level, int sub_degree) const;

  void CmtBatched(ConstContextPtr<word> context, std::vector<Ct> &res,
                  const std::vector<Ct> &cts, int sub_degree,
                  const EvkMap<word> &evk_map) const;

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

  /**
   * @brief The rotation indices `ScrambleAuto` will ask for, so that a caller
   * can generate exactly those keys and no others.
   *
   * The automorphisms are `X -> X^(2kt+1)` for `t` in `[d)`. Those elements
   * form the subgroup `H = <5^(k/2)>` of order d, so each is `5^i` for some i
   * and Cheddar can supply it as an ordinary rotation key. `t = 0` is the
   * identity and is omitted.
   *
   * @param sub_degree k
   * @return the indices, in increasing t order, without the identity
   */
  std::vector<int> ScrambleAutoRotationIndices(int sub_degree) const;

  /**
   * @brief [KANG] ScrambleAuto: `res[t] = cts[t*](X^(2kt+1))`, where
   * `2k*t* + 1 = (2kt+1)^-1 mod 2N`.
   *
   * A permutation of the inputs followed by one automorphism each. Consumes no
   * multiplicative level -- a key switch and a permutation are both level-free
   * -- but it is where CMT's d key switchings live, and the only stage of the
   * algorithm that needs key material at all.
   *
   * @param context the CKKS context
   * @param res output, resized to `cts.size()`; may not alias `cts`
   * @param cts input ciphertexts, `degree / sub_degree` of them
   * @param sub_degree k
   * @param evk_map rotation keys for `ScrambleAutoRotationIndices(k)`
   */
  void ScrambleAuto(ConstContextPtr<word> context, std::vector<Ct> &res,
                    const std::vector<Ct> &cts, int sub_degree,
                    const EvkMap<word> &evk_map) const;

  /**
   * @brief [KANG] Algorithm 3, the whole CMT:
   *
   *     X^i * ct  ->  TWEAK(+1)  ->  d^-1, ScrambleAuto  ->  TWEAK(-1)
   *               ->  X^-j * ct
   *
   * The `X^i` and `X^-j` wrappers are what turn TWEAK's `X^(2kij)` twiddle
   * into Adjust's `X^(i(2kt+1))`, since `i(2kt+1) = 2kit + i`.
   *
   * **`d^-1` is free.** The scale is host metadata, so dividing the message by
   * d is recording a scale d times larger; the ciphertext words do not move
   * and no level is spent. Encoding `1/d` as a constant instead would either
   * round to zero at scale 1 or cost a rescale, and CMT cannot afford a level:
   * the batch CC-MM spends this ring's only one on its tensor product.
   *
   * @param context the CKKS context
   * @param res output, resized to `cts.size()`; may not alias `cts`
   * @param cts input ciphertexts, `degree / sub_degree` of them
   * @param sub_degree k
   * @param evk_map rotation keys for `ScrambleAutoRotationIndices(k)`
   */
  void Cmt(ConstContextPtr<word> context, std::vector<Ct> &res,
           const std::vector<Ct> &cts, int sub_degree,
           const EvkMap<word> &evk_map) const;

  /**
   * @brief Algorithm 3 ciphertext by ciphertext, through `Context`'s
   * per-ciphertext operations: the reference the batched path is measured
   * against word for word (`CmtTest`). `Cmt` takes this path on the
   * conjugate-invariant ring, at d = 1, or under `CHEDDAR_CMT_SERIAL=1`.
   */
  void CmtSerial(ConstContextPtr<word> context, std::vector<Ct> &res,
                 const std::vector<Ct> &cts, int sub_degree,
                 const EvkMap<word> &evk_map) const;

  /** @brief Whether `Cmt` batches: `CHEDDAR_CMT_SERIAL` unset or not 1. */
  static bool BatchEnabled();
};

}  // namespace cheddar
