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
 * @brief RLWE-to-MLWE decomposition of Bae, Cheon, Hanrot, Park and Stehle
 * (CRYPTO 2024), appendix A -- the format the PC-MM uses when the encrypted
 * matrix is narrower than the ring degree.
 *
 * ## Why this exists
 *
 * An RLWE ciphertext at degree N holds N coefficients, so using one ciphertext
 * per row of a d-wide matrix wastes a factor N/d. MLWE fixes that: the same
 * data becomes k = N/N' ciphertexts over the subring of degree N', each of
 * module rank k, and one of them holds exactly one row.
 *
 * Crucially the security dimension is unchanged, because MLWE of rank k over
 * degree N' is as hard as RLWE over degree kN' = N. **ModDecomp needs no
 * switching key and therefore costs no security at all** -- unlike RingSwitch,
 * which does need one and is only safe at a small modulus.
 *
 * ## The decomposition
 *
 * With N' | N and k = N/N', identify Y = X^k, so R_{N'} = Z[Y]/(Y^{N'} + 1).
 * Every a in R_N splits X^k-adically (paper, eq. 14) into k elements of
 * R_{N'}:
 *
 *     e*_i(a)[s] = a[i + k*s],     0 <= i < k,  0 <= s < N'
 *
 * that is, e*_i(a) is the stride-k slice of the coefficient vector starting at
 * offset i. Then (paper, eq. 15), with
 *
 *     sk'    = ( e*_0(sk), ..., e*_{k-1}(sk) )                 in R^k_{N'}
 *     a~_i[j] = e*_{i-j}(a)          for j <= i
 *             = Y * e*_{i-j+k}(a)    for j >  i
 *
 * the pair (a~_i, e*_i(b)) is an MLWE^k ciphertext of e*_i(m) under sk'.
 *
 * The wrap carries no sign change: it comes from X^{i+k} = X^i * Y, and
 * i + k < 2k <= N, so the negacyclic reduction of R_N is never reached. The
 * negacyclic wrap of R_{N'} does appear inside the multiplication by Y:
 * (Y*q)[0] = -q[N'-1], (Y*q)[s] = q[s-1] for s >= 1.
 *
 * ## Representation
 *
 * The output is in the **coefficient** domain, not the NTT domain. That is not
 * a convenience: the whole point of this format is to feed the two plaintext
 * matrix products, which are defined on coefficient vectors, and the result is
 * converted back by ModPack. No NTT at degree N' is required *by this path*.
 *
 * That is a statement about the PC-MM only, and must not be read as a general
 * escape from small-degree NTT. Sylph's other small-ring operator, the batch
 * CC-MM, lives at degree 2^12 in the SinC encoding ([SYLPH] table 4) and does
 * need a working transform there, as does every key switch and rescale
 * performed on a ring-switched degree-2^12 ciphertext. Cheddar's NTT launch
 * configuration is only tuned for log_degree 16 and degenerates below it
 * (NTTUtils.cuh:399-432), so small-degree NTT remains a prerequisite of the
 * pipeline as a whole -- just not of the decomposition implemented here.
 *
 * ## Cost, and why RingSwitch comes first
 *
 * Materializing a~_i costs k times the original a-part, because each of the k
 * ciphertexts carries all k blocks. Two things keep that affordable, and both
 * are scheduling decisions rather than accidents:
 *
 *   * the product is placed at the lowest level, where a ciphertext has two or
 *     three limbs rather than fifty;
 *   * a RingSwitch from N down to N1 before decomposing reduces k from N/N' to
 *     N1/N', which is why Sylph reaches degree 256 by "ring-switching followed
 *     by an MLWE decomposition" rather than decomposing straight from 2^16.
 *
 * The k blocks are shared between all k ciphertexts, so a compact
 * representation that stores them once and applies the rotation implicitly is
 * possible; this implementation materializes them, which is simpler and is
 * what the correctness tests check.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class MlweCiphertext {
 public:
  MlweCiphertext() = default;

  // Movable, but not copyable (it owns device memory).
  MlweCiphertext(MlweCiphertext &&) = default;
  MlweCiphertext &operator=(MlweCiphertext &&) = default;

  int GetRank() const;
  int GetDegree() const;
  NPInfo GetNP() const;
  double GetScale() const;

  // member variables (public, as elsewhere in Cheddar)
  int rank_ = 0;       // k
  int degree_ = 0;     // N'
  NPInfo np_;
  double scale_ = 1.0;

  // a_[(limb * rank_ + j) * degree_ + s] = coefficient s of a~_i[j]
  DeviceVector<word> a_;
  // b_[limb * degree_ + s] = coefficient s of e*_i(b)
  DeviceVector<word> b_;
};

template <typename word>
class MlweHandler {
 private:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;

  const Parameter<word> &param_;
  const NTTHandler<word> &ntt_handler_;

  static constexpr int kernel_block_dim_ = 256;

 public:
  MlweHandler(const Parameter<word> &param,
              const NTTHandler<word> &ntt_handler);

  // disable copying (or moving also)
  MlweHandler(const MlweHandler &) = delete;
  MlweHandler &operator=(const MlweHandler &) = delete;

  /**
   * @brief Decompose one NTT-domain RLWE ciphertext into k = degree /
   * small_degree coefficient-domain MLWE ciphertexts of rank k.
   *
   * The input is inverse-transformed internally, since Cheddar keeps
   * ciphertexts in the NTT domain but the decomposition is defined on
   * coefficients. No key is used and no security is spent.
   *
   * On the conjugate-invariant ring the API and the container shapes are
   * unchanged, but the map inside is not a stride gather: R+ is free over its
   * rank-N' subring on {1, c_1, ..., c_{k-1}}, and c_i c_{tk} = c_{tk+i} +
   * c_{tk-i} hits two coefficient classes, so the components come out of an
   * alternating-sign suffix-sum scan and the a~ arrangement mixes them with
   * subring coefficients {1, -1, 2, c_k} (Doing.md 1.5ba, and the kernel
   * comments in Mlwe.cu). Still no key and no security cost; what it does
   * cost is coefficient growth in the components, about 0.68 sqrt(N') rms.
   *
   * @param res output, resized to k
   * @param ct input ciphertext, no aux primes and no rx_ part
   * @param small_degree N', a power of two dividing the ring degree
   */
  void ModDecomp(std::vector<MlweCiphertext<word>> &res, const Ct &ct,
                 int small_degree) const;

  /**
   * @brief Pack k = degree / small_degree coefficient-domain MLWE ciphertexts
   * of rank k back into one NTT-domain RLWE ciphertext at the ring degree.
   * This is the inverse of ModDecomp, and it is the step [BAE] section 2.3
   * calls ModPack.
   *
   * ## The algorithm ([BAE] appendix A, p. 36)
   *
   * The k inputs (a_i, b_i) satisfy <a_i, sk'> + b_i = m_i over R_{N'}, with
   * sk' = ( e*_0(sk), ..., e*_{k-1}(sk) ). Recompose X^k-adically -- the same
   * interleaving ModDecomp undoes, and equally free:
   *
   *     A_j[i + k*s] = a_i[j][s],     B[i + k*s] = b_i[s]
   *
   * Because each sk'_j lies in the subring (its nonzero coefficients sit only
   * on multiples of X^k) and X^i commutes with it,
   *
   *     sum_j A_j * sk'_j  +  B  =  sum_i m_i X^i  =  m     in R_N.
   *
   * That is an MLWE ciphertext of rank k *at the full degree*. Reducing it to
   * rank 1 is k key switches: for each j, switch (A_j, 0) from sk'_j to the
   * ordinary secret sk, then sum the results and add B.
   *
   * ## What it costs, and why it is not free like ModDecomp
   *
   * ModDecomp needs no key and spends no security, because rank k over degree
   * N' is exactly as hard as RLWE over degree kN'. **The reverse direction is
   * not symmetric**: it needs k switching keys, one per module component, from
   * a secret that is *not* sk. [BAE] section 2.3 states this outright, and it
   * is the reason the PC-MM is scheduled at the lowest level -- k key switches
   * at fifty limbs would dominate the two plaintext products they exist to
   * serve.
   *
   * The keys are ordinary CKKS switching keys, so nothing new is needed on the
   * key-generation side beyond the embedded secrets themselves; see
   * UserInterface::PrepareModPackKeys.
   *
   * The k switches are accumulated *before* the single mod-down, following the
   * MultKeyNoModDown / ModDown split that Context::MultKey performs internally.
   * Mod-down is the expensive half of a key switch (an INTT over the auxiliary
   * primes and an NTT back), so this is k mod-downs turned into one. It also
   * rounds once instead of k times, so the accumulated result is no noisier
   * than the naive order.
   *
   * The inputs are in the coefficient domain, which is where the PC-MM leaves
   * them; the recomposed polynomials are transformed here, once each, because
   * key switching is defined in the NTT domain. Cheddar's NTT is only tuned
   * for log_degree 16 (NTTUtils.cuh:399-432) but it is the *full* ring degree
   * that is transformed here, never the small one -- ModPack asks nothing of
   * the small-degree transform.
   *
   * @param context the CKKS context, for key switching
   * @param res output ciphertext, at the ring degree and in the NTT domain
   * @param cts the k input MLWE ciphertexts, in decomposition-index order,
   *        all sharing one NP, rank and degree, and with no auxiliary primes
   * On the conjugate-invariant ring the recomposition is the banded two-term
   * inverse of the scan (see ModDecomp) rather than an interleave, and the
   * embedded secrets the keys switch from are the scan of the secret's
   * coefficients rather than stride slices -- UserInterface::
   * PrepareModPackKeys builds them accordingly. The switches themselves are
   * ordinary key switches either way.
   *
   * @param keys the k switching keys, keys[j] switching from the embedded
   *        j-th module component of the secret to the ordinary secret
   */
  void ModPack(ConstContextPtr<word> context, Ct &res,
               const std::vector<MlweCiphertext<word>> &cts,
               const std::vector<const Evk *> &keys) const;

  /**
   * @brief res = lo + Y^(N'/2) * hi, on one module component.
   *
   * TWO PAYLOADS, ONE PACK. `ModPack` costs `rank` key switches per output
   * ciphertext and that is what a projection spends its time on -- measured at
   * 81% of the block's seven projections, against 6% for the product itself.
   * Halving the pack count therefore nearly halves the projection, and the way
   * to halve it is to put two outputs in one ciphertext before packing rather
   * than after.
   *
   * The shift is the module-level image of the ring's own `X^(N/2)`: packing
   * sends entry `s` of component `n` to big coefficient `n + rank*s`, so
   * `Y^(N'/2)` on every component is `X^(rank*N'/2)` = `X^(N/2)` on the packed
   * ciphertext. That is the same merge `BootContext::HalfBootPair` performs on
   * the big ring, arrived at one pack earlier.
   *
   * The arithmetic is unconditional -- `Y^N' = -1`, so the wrap negates, and
   * nothing here assumes an empty half. What does assume it is the *use*: the
   * two payloads only stay separable while each occupies coefficients
   * `0 .. N'/2-1` alone, which is the contract `HalfBootPair` documents and
   * which the caller owns.
   */
  void AddShiftedHalf(MlweCiphertext<word> &res, const MlweCiphertext<word> &lo,
                      const MlweCiphertext<word> &hi) const;

  /**
   * @brief The scan (P^-1) applied IN PLACE to one coefficient-domain
   * polynomial laid out `[limb][t * rank + i]`, every limb of `np`.
   *
   * What a module-centred ModRaise needs (Doing.md 3.5): the wrap-around
   * integer a bootstrap has to remove is small in whichever coordinates the
   * level-zero representatives were centred in, and the module-basis
   * CoeffToSlot reads module coordinates, so the representatives are centred
   * there -- scan, lift, recompose. Both maps are integer and linear, so per
   * residue is exact. Conjugate-invariant rings only.
   */
  void ScanInPlace(DvView<word> &poly, const NPInfo &np,
                   int small_degree) const;

  /** @brief The banded recomposition (P), in place; the inverse of the above. */
  void RecomposeInPlace(DvView<word> &poly, const NPInfo &np,
                        int small_degree) const;

  /**
   * @brief Inverse-transform a ciphertext component into the coefficient
   * domain and copy it to the host. Exposed because the decomposition is only
   * meaningful against coefficient vectors, so tests need them.
   *
   * @param res output host vector, num_total_primes * degree entries
   * @param src NTT-domain component view
   * @param np the NPInfo of the component
   */
  void ComponentToHostCoeffs(HostVector<word> &res,
                             const DvConstView<word> &src,
                             const NPInfo &np) const;
};

}  // namespace cheddar
