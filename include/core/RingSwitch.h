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
 * @brief Ring switching of Bae, Cheon, Hanrot, Park and Stehle (CRYPTO 2024),
 * appendix A -- moving a ciphertext from degree N down to k = N/N'
 * ciphertexts of degree N'.
 *
 * ## The construction, and why it is only two steps
 *
 *     1. (a~, b~) <- KeySwitch( (a,b), swk_{sk -> sk'} )   sk' in the subring
 *     2. (a_i, b_i) = ( e*_i(a~), e*_i(b~) )               0 <= i < k
 *
 * The target secret sk' has nonzero coefficients only at multiples of X^k, so
 * its X^k-adic view is (s_small, 0, ..., 0). Feeding that into the MLWE
 * identity of appendix A, every module component past the first is multiplied
 * by zero and the rank-k relation collapses to
 *
 *     e*_i(a~) * s_small + e*_i(b~) = e*_i(m)   in R_{N'}
 *
 * which is an ordinary RLWE ciphertext at degree N'. That collapse is what
 * separates this from ModDecomp: there the secret is the full sk, all k
 * components survive, and the result is genuinely MLWE with a Toeplitz-like
 * arrangement of a. Here step 2 is a plain stride-k gather on both components
 * and nothing else.
 *
 * On the conjugate-invariant ring the collapse survives verbatim -- the
 * embedded secret zeroes every module component but the first, and with it
 * every c_k-carrying term of the a~ arrangement -- but the components are the
 * alternating-sign suffix-sum scan of Mlwe.cu rather than stride slices, and
 * the recomposition its banded inverse. `ci_ringswitch16_35.json` and
 * `ci12_35.json` are the conjugate-invariant pair, on 1-mod-4N primes.
 *
 * ## Why this needs its own parameter set
 *
 * The switching key is published at modulus P*Q, and an attacker can apply
 * e*_i to it and read RLWE samples **at degree N'**. So log2(PQ) has to fit
 * the small ring's budget rather than the big one's -- [BAE] appendix A states
 * the condition outright, and it is the reason a ring switch is scheduled at a
 * low level rather than wherever is convenient.
 *
 * Concretely, the shipped presets carry alpha = 12, which puts P alone at
 * 2^372 against a budget near 2^104 for degree 4096. Lowering Q cannot rescue
 * that; the key needs a parameter set whose alpha is 1. That is what
 * `ringswitch16_30.json` is: the same ring degree and the *same primes* as
 * `bootparam_30`'s level 1, but with a single auxiliary prime. A ciphertext
 * therefore crosses from the big Context to the switching Context without a
 * single word changing -- only which handlers act on it.
 *
 * The alternative of reusing the existing short base (its num_aux is level 0's
 * prime count, so log2 PQ is 79.5 and secure) fails for a different reason:
 * `Context::AdjustLevelForMultKey` only routes it at level 0, where Q is about
 * ten bits above the scale. After the message and the ciphertext scale there
 * are roughly five bits left for the plaintext matrix, and a typical
 * Llama-3 weight of 0.0175 rounds to nothing. One extra prime is what buys the
 * thirty-five bits the product actually needs.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class RingSwitchHandler {
 private:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;

  ConstContextPtr<word> big_;
  ConstContextPtr<word> small_;
  int rank_ = 0;

  static constexpr int kernel_block_dim_ = 256;

 public:
  /**
   * @param big the Context hosting the key switch. Its degree is N and its
   * alpha must be small enough that the switching key fits the small ring's
   * security budget.
   * @param small the Context the switched ciphertexts belong to, at degree N'.
   * Its level configuration must match `big`'s so that levels correspond.
   */
  RingSwitchHandler(ConstContextPtr<word> big, ConstContextPtr<word> small);

  // disable copying (or moving also)
  RingSwitchHandler(const RingSwitchHandler &) = delete;
  RingSwitchHandler &operator=(const RingSwitchHandler &) = delete;

  int GetRank() const { return rank_; }

  /**
   * @brief Switch one degree-N ciphertext into `rank` degree-N' ciphertexts.
   *
   * @param res output, resized to rank; each belongs to the small Context
   * @param ct input, at the big Context, no aux primes and no rx_ part
   * @param swk the ring-switching key, from
   * UserInterface::PrepareRingSwitchKey on the big Context
   */
  void Switch(std::vector<Ct> &res, const Ct &ct, const Evk &swk) const;

  /**
   * @brief The way back: recompose `rank` degree-N' ciphertexts into one at
   * degree N.
   *
   * X^k-adic recomposition first, which is free and needs no key. It is also
   * already *correct* as a ciphertext: because sk' lies in the subring,
   * multiplying by it does not mix the X^k-adic components, so
   * e*_i(a~ * sk') = e*_i(a~) * s_small and the interleaved pair decrypts to
   * the interleaved message under sk' at degree N. Only then does a single key
   * switch move it off sk' and onto the ordinary secret.
   *
   * @param res output, one ciphertext at the big degree
   * @param parts the rank inputs, all at the small Context, same level, no aux
   * primes and no rx_ part
   * @param swk the inverse ring-switching key, from
   * UserInterface::PrepareInverseRingSwitchKey on the big Context
   */
  void SwitchBack(Ct &res, const std::vector<Ct> &parts, const Evk &swk) const;
};

}  // namespace cheddar
