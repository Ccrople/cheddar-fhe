#pragma once

/**
 * @brief The code inside this file is for test purposes and are NOT secure
 * implementations.
 *
 */

#include "Random.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"

namespace cheddar {

/**
 * @brief This class provides a simple unoptimized client interface for CKKS.
 * The security of this class is not guaranteed and should not be used in
 * production. This class is intended for testing purposes only.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class UserInterface {
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;


 public:
  /**
   * @brief Construct a new UserInterface object. Random secrets will be sampled
   * and basic evaluation keys (multiplication / conjugation / dense-to-sparse /
   * sparse-to-dense) will be prepared automatically.
   *
   * @param context CKKS context (can be a BootContext or a Context)
   */
  explicit UserInterface(ContextPtr<word> context);

  /**
   * @brief Encrypt a plaintext into a ciphertext.
   *
   * @param ctxt output ciphertext
   * @param ptxt input plaintext
   */
  void Encrypt(Ct &ctxt, const Pt &ptxt) const;

  /**
   * @brief Decrypt a ciphertext into a plaintext.
   *
   * @param ptxt output plaintext
   * @param ctxt input ciphertext
   */
  void Decrypt(Pt &ptxt, const Ct &ctxt) const;

  // Get const reference to an evaluation key
  const Evk &GetRotationKey(int rot_idx) const;
  const Evk &GetMultiplicationKey() const;
  const Evk &GetConjugationKey() const;
  const Evk &GetDenseToSparseKey() const;
  const Evk &GetSparseToDenseKey() const;
  const Evk &GetModPackKey(int rank, int j) const;
  const Evk &GetRingSwitchKey(int rank) const;

  /**
   * @brief This ring's secret as signed ternary coefficients.
   *
   * Exposed because a ring switch is a key switch between two *different*
   * Contexts: the key is built in the big ring but its target secret is the
   * small ring's own, independently sampled, embedded into the big ring. The
   * small ring's UserInterface is the only thing that knows those
   * coefficients, so it has to hand them over.
   *
   * This obviously has no place outside a test harness, which is what
   * UserInterface already is (see the warning it prints on construction).
   */
  const std::vector<int> &GetSecretCoeffs() const;

  /**
   * @brief Getter for the evaluation key map.
   *
   * @return const EvkMap<word>& const reference to the evaluation key map
   */
  const EvkMap<word> &GetEvkMap() const;

  /**
   * @brief Prepare a rotation key for the given rotation distance.
   *
   * @param rot_idx rotation distance
   * @param max_level maximum level for the rotation key (default: -1 -->
   * param_->max_level_)
   */
  void PrepareRotationKey(int rot_idx, int max_level = -1);

  /**
   * @brief Prepare rotation keys for the given EvkRequest.
   *
   * @param evk_request The request containing rotation distances and levels.
   */
  void PrepareRotationKey(const EvkRequest &evk_request);

  /**
   * @brief Prepare the k = degree / small_degree switching keys that
   * MlweHandler::ModPack needs to come back from the MLWE format.
   *
   * ModPack leaves a rank-k MLWE ciphertext at the full ring degree under the
   * secret ( e*_0(sk), ..., e*_{k-1}(sk) ), whose j-th component embeds into
   * R_N as the polynomial holding sk's coefficients j, j+k, j+2k, ... on the
   * powers X^0, X^k, X^{2k}, ... Reducing that to rank 1 means switching each
   * component to the ordinary secret, so this generates one ordinary
   * evaluation key per component. They are the *only* key material the whole
   * Bae PC-MM path uses -- the product itself needs none, and ModDecomp needs
   * none.
   *
   * Note the direction: the keys are indexed by module rank as well as by
   * component, because a different rank decomposes the secret differently.
   *
   * @param small_degree N', the degree the MLWE ciphertexts live at
   * @param max_level maximum level for the keys (default: -1 -->
   * param_->max_level_)
   */
  void PrepareModPackKeys(int small_degree, int max_level = -1);

  /**
   * @brief Prepare the key that switches a ciphertext of this ring onto a
   * secret lying in the degree-`small_degree` subring, which is what lets it
   * be split into `rank` ciphertexts of that degree ([BAE] appendix A).
   *
   * The direction is the mirror of PrepareModPackKeys. There the ciphertext
   * arrives under an embedded secret and leaves under the ordinary one; here
   * it arrives under the ordinary secret and leaves under the embedded one,
   * so the roles of encryption and target secret are exchanged.
   *
   * Only one key is needed, not `rank` of them: the target secret has nonzero
   * coefficients only at multiples of X^k, so its X^k-adic view is
   * (s_small, 0, ..., 0) and the MLWE relation collapses to rank 1.
   *
   * **The modulus this key is published at is a security parameter.** An
   * attacker can apply e*_i to it and read RLWE samples at degree
   * `small_degree`, so log2(PQ) has to fit that degree's budget, not this
   * one's. [BAE] states the condition outright. In practice this means the
   * key must be built in a parameter set whose alpha is small -- the ordinary
   * presets carry alpha=12, and P alone is then 2^372 against a budget near
   * 2^104.
   *
   * @param small_degree N', the degree the switched ciphertexts will live at
   * @param small_secret_coeffs the small ring's secret, from its own
   * UserInterface::GetSecretCoeffs; exactly `small_degree` entries
   * @param max_level maximum level for the key (default: -1 --> max_level_)
   */
  void PrepareRingSwitchKey(int small_degree,
                            const std::vector<int> &small_secret_coeffs,
                            int max_level = -1);

 private:
  static inline constexpr double kErrorStandardDeviation = 3.2;
  static inline constexpr int kernel_block_dim_ = 256;

  ContextPtr<word> context_;
  Dv main_secret_;
  Dv sparse_secret_;
  // main_secret_ as signed ternary coefficients, for PrepareModPackKeys
  std::vector<int> main_secret_coeffs_;

  EvkMap<word> evk_map_;

  std::vector<word> all_primes_;

  DvView<word> MainSecretView(int front_ignore = 0);
  DvConstView<word> MainSecretConstView(int front_ignore = 0) const;
  DvView<word> SparseSecretView(int front_ignore = 0);
  DvConstView<word> SparseSecretConstView(int front_ignore = 0) const;

  // Initialization sequences;
  void PrepareSecrets();
  void PrepareBasicEvks();

  void PrepareEvk(int key_idx, const NPInfo &np, const Dv &encryption_secret,
                  const Dv &target_secret);

  void SampleRandomPolynomial(Dv &poly, const NPInfo &np) const;
  void SampleError(Dv &poly, const NPInfo &np) const;

  NPInfo GetNPForEvk(int max_level) const;
};

}  // namespace cheddar
