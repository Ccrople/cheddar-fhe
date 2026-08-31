#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "core/Container.h"
#include "core/ElementWise.h"
#include "core/Encode.h"
#include "core/EncodeGpu.h"
#include "core/MemoryPool.h"
#include "core/ModSwitch.h"
#include "core/MultiLevelCiphertext.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

template <typename word>
class Context {
 protected:
  // short-hand notations
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Const = Constant<word>;

  Context(const Parameter<word> &param);

  void MatchResultWith(Ct &res, const Ct &a) const;
  void MatchResultWith(Ct &res, const Ct &a, const Ct &b) const;
  void AdjustLevelForMultKey(int &level, const int num_q,
                             const int num_aux) const;

  DvConstView<word> GetPProd(NPInfo &np) const;
  const ModSwitchHandler<word> &GetDtSModSwitchHandler() const;
  const ModSwitchHandler<word> &GetStDModSwitchHandler() const;


 public:
  void AssertSameScale(const double &scale1, const double &scale2) const;

  template <typename Container1>
  void AssertSameScale(const Container1 &a, const double &scale) const {
    AssertSameScale(scale, a.GetScale());
  }

  template <typename Container1>
  void AssertSameScale(const double &scale, const Container1 &a) const {
    AssertSameScale(scale, a.GetScale());
  }

  template <typename Container1, typename Container2>
  void AssertSameScale(const Container1 &a, const Container2 &b) const {
    AssertSameScale(a.GetScale(), b.GetScale());
  }

  template <typename Container1, typename Container2, typename... Args>
  void AssertSameScale(const Container1 &a, const Container2 &b,
                       Args... args) const {
    AssertSameScale(a, b);
    AssertSameScale(b, args...);
  }

  /**
   * @brief Create a new Context object. This is the only way to create a new
   * Context and should be used instead of the constructor.
   *
   * @param param CKKS parameter
   * @return std::shared_ptr<Context<word>> a shared pointer to the new Context
   */
  static std::shared_ptr<Context<word>> Create(const Parameter<word> &param);

  // disable copying (or moving also)
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  // For forwarding purposes
  Context(Context &&) = default;

  // Make it polymorphic
  virtual ~Context();

  // The order matters here.
  const Parameter<word> &param_;
  MemoryPool memory_pool_;
  ElementWiseHandler<word> elem_handler_;
  NTTHandler<word> ntt_handler_;
  std::vector<ModSwitchHandler<word>> mod_switch_handlers_;
  Encoder<word> encoder_;
  /**
   * @brief The same encodings on the device.
   *
   * `encoder_` is one host thread and a GMP reduction per (value, prime); it
   * is the reference and the decoder. Whatever the library encodes in bulk --
   * a transform's diagonals (`HoistHandler::CompilePlaintexts`), a
   * projection's operands (`CoeffLinearLeg`) -- goes through this one, which
   * is limb-identical on the coefficient and matrix encodings and rounds to
   * nearest where the host's slot encoding truncates (`EncodeGpu.h`).
   */
  GpuEncoder<word> gpu_encoder_;

  DeviceVector<word> p_prod_;
  DeviceVector<word> p_prod_dts_;

  /**
   * @brief Extra mod-switch machinery for key switches on a narrow auxiliary
   * basis, keyed by (level, num_aux).
   *
   * `alpha_` is one number for the whole parameter set and it is sized for the
   * deepest key switch in it. A switch low in the chain pays for that: ModPack
   * runs `rank` of them at level 1, where the ciphertext has three primes and
   * the extended basis has fifteen. Building a second handler for the same
   * level with fewer auxiliary primes is the whole fix, and it is opt-in
   * because it changes the key-switch noise -- P must still exceed the digit,
   * which the caller is responsible for.
   *
   * Empty unless `PrepareNarrowKeySwitch` was called, so nothing that does not
   * ask for one is affected.
   */
  mutable std::map<std::pair<int, int>, ModSwitchHandler<word>>
      narrow_handlers_;
  mutable std::map<std::pair<int, int>, DeviceVector<word>> narrow_p_prod_;
  std::vector<Const> level_down_consts_;

  /**
   * @brief Copy a ciphertext to another ciphertext. Falls back to nop if the
   * two ciphertexts are the same.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void Copy(Ct &res, const Ct &a) const;

  // Basic functions
  // The functions in the Context are meant to be only used for
  // operands using the same prime set. (with same scale)

  /**
   * @brief Add two ciphertexts. res = a + b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   */
  void Add(Ct &res, const Ct &a, const Ct &b) const;

  /**
   * @brief Add a ciphertext with a plaintext. res = a + b
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input plaintext
   */
  void Add(Ct &res, const Ct &a, const Pt &b) const;

  /**
   * @brief Add a ciphertext with a constant. res = a + b
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   */
  void Add(Ct &res, const Ct &a, const Const &b) const;

  /**
   * @brief Subtract two ciphertexts. res = a - b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   */
  void Sub(Ct &res, const Ct &a, const Ct &b) const;

  /**
   * @brief Subtract a plaintext from a ciphertext. res = a - b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input plaintext (right)
   */
  void Sub(Ct &res, const Ct &a, const Pt &b) const;

  /**
   * @brief Subtract a constant from a ciphertext. res = a - b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input constant (right)
   */
  void Sub(Ct &res, const Ct &a, const Const &b) const;

  /**
   * @brief Subtract a ciphertext from a plaintext. res = a - b
   *
   * @param res result ciphertext
   * @param a input plaintext (left)
   * @param b input ciphertext (right)
   */
  void Sub(Ct &res, const Pt &a, const Ct &b) const;

  /**
   * @brief Subtract a ciphertext from a constant. res = a - b
   *
   * @param res result ciphertext
   * @param a input constant (left)
   * @param b input ciphertext (right)
   */
  void Sub(Ct &res, const Const &a, const Ct &b) const;

  /**
   * @brief Negate a ciphertext. res = -a
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void Neg(Ct &res, const Ct &a) const;

  /**
   * @brief Multiply two ciphertexts (only perform tensor). res = a * b.
   * Relinearization or rescaling will not be performed
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   */
  void Mult(Ct &res, const Ct &a, const Ct &b) const;

  /**
   * @brief Multiply a ciphertext with a plaintext. res = a * b. Rescaling will
   * not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input plaintext
   */
  void Mult(Ct &res, const Ct &a, const Pt &b) const;

  /**
   * @brief Multiply a ciphertext with a constant. res = a * b. Rescaling will
   * not be performed
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   */
  void Mult(Ct &res, const Ct &a, const Const &b) const;

  // Unsafe functions are used for performance reasons and they only work
  // for specific cases. Do not use them unless you know what you are doing.

  // For Ct x Ct/Pt/Const mult, we should be able to perform operations
  // between operands at different levels. However, it will make scale
  // management very difficult.

  /**
   * @brief Check if two levels are compatible for MultUnsafe.
   *
   * @param level1 level of the first operand
   * @param level2 level of the second operand
   * @return true if the two levels are compatible
   * @return false if the two levels are not compatible
   */
  bool IsMultUnsafeCompatible(int level1, int level2) const;

  /**
   * @brief Multiply two ciphertexts at a designated level. res = a * b.
   * Relinearization or rescaling will not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   * @param level (default: -1 --> min(a's level, b's level))
   */
  void MultUnsafe(Ct &res, const Ct &a, const Ct &b, int level = -1) const;

  /**
   * @brief Multiply a ciphertext with a plaintext at a designated level. res =
   * a * b. Rescaling will not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input plaintext
   * @param level (default: -1 --> min(a's level, b's level))
   */
  void MultUnsafe(Ct &res, const Ct &a, const Pt &b, int level = -1) const;

  /**
   * @brief Multiply a ciphertext with a constant at a designated level. res = a
   * * b. Rescaling will not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   * @param level (default: -1 --> min(a's level, b's level))
   */
  void MultUnsafe(Ct &res, const Ct &a, const Const &b, int level = -1) const;

  // Should be used after MultKey

  /**
   * @brief Permute the polynomials in the ciphertext according to the rotation
   * distance.
   *
   * @param res result_ciphertext
   * @param a input ciphertext
   * @param rot_dist rotation distance
   */
  void Permute(Ct &res, const Ct &a, int rot_dist) const;

  /**
   * @brief Permute the polynomials in the ciphertext for conjugation.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void PermuteConjugate(Ct &res, const Ct &a) const;

  /**
   * @brief Multiply sqrt(-1) to the ciphertext. This does not require
   * rescaling.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void MultImaginaryUnit(Ct &res, const Ct &a) const;

  /**
   * @brief Perform rescaling on the ciphertext. Level will be reduced by 1.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void Rescale(Ct &res, const Ct &a) const;

  // Key-related operations
  // Performs ModUp -> key mult -> ModDown (+ Rescale)

  /**
   * @brief Perform relinearization on a ciphertext with three polynomials.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param key multiplication key
   */
  void Relinearize(Ct &res, const Ct &a, const Evk &key) const;

  /**
   * @brief Perform relinearization and rescaling on a ciphertext with three
   * polynomials. This function is faster than performing relinearization and
   * rescaling separately and its cost is similar to just a single
   * relinearization.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param key multiplication key
   */
  void RelinearizeRescale(Ct &res, const Ct &a, const Evk &key) const;

  /**
   * @brief Multiply a ciphertext with a evaluation key.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param key any evaluation key
   */
  void MultKey(Ct &res, const Ct &a, const Evk &key) const;

  // Short-hand functions

  /**
   * @brief MultKey + Permute
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param rot_key rotation key
   * @param rot_dist rotation distance
   */
  void HRot(Ct &res, const Ct &a, const Evk &rot_key, int rot_dist) const;

  /**
   * @brief MultKey + PermuteConjugate
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param conj_key conjugation key
   */
  void HConj(Ct &res, const Ct &a, const Evk &conj_key) const;

  /**
   * @brief Mult + (Relinearize or RelinearizeRescale)
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   * @param mult_key multiplication key
   * @param rescale whether to rescale the result
   */
  void HMult(Ct &res, const Ct &a, const Ct &b, const Evk &mult_key,
             bool rescale = true) const;

  /**
   * @brief res = res + a * b, but faster.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   */
  void MadUnsafe(Ct &res, const Ct &a, const Const &b) const;

  /**
   * @brief res = (a << rot_dist) + b, but faster.
   *
   * @param res result ciphertext
   * @param a input ciphertext to perform HRot
   * @param b input ciphertext to add
   * @param rot_key rotation key
   * @param rot_dist rotation distance
   */
  void HRotAdd(Ct &res, const Ct &a, const Ct &b, const Evk &rot_key,
               int rot_dist) const;

  /**
   * @brief res = conj(a) + b, but faster.
   *
   * @param res result ciphertext
   * @param a input ciphertext to perform HConj
   * @param b input ciphertext to add
   * @param conj_key conjugation key
   */
  void HConjAdd(Ct &res, const Ct &a, const Ct &b, const Evk &conj_key) const;

  /**
   * @brief Reduce the level of the input ciphertext to a target level.
   * This is not an optimized implementation of LevelDown.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param target_level target level (<= a's level)
   */
  void LevelDown(Ct &res, const Ct &a, int target_level) const;

  /**
   * @brief Add lower-level versions in the MultiLevelCiphertext.
   *
   * @param ml_ct the MultiLevelCiphertext
   * @param min_level the minimum level to support
   */
  void AddLowerLevelsUntil(MultiLevelCiphertext<word> &ml_ct,
                           int min_level) const;

  // Special-purpose functions for bootstrapping/hoisting
  void MultKeyNoModDown(Ct &accum, const std::vector<Dv> &a_modup,
                        const Ct &a_orig, const Evk &key) const;
  void MultKeyNoModDown(Ct &accum, const Ct &a, const Evk &key) const;

  /**
   * @brief accumulate <key[k], mod_up[k]> over k, in as few launches as the
   * accumulating kernel allows.
   *
   * The counterpart to `MultKeyNoModDown(accum, a_modup, a_orig, key)` for a
   * caller holding several switches at once. `PAccum` takes several
   * (key, mod-up) pairs per launch and, with `accumulate`, folds the running
   * accumulator in as one more term, so a group of switches costs one launch
   * rather than one product and one addition each. Pass no more than
   * `ElementWiseHandler<word>::max_num_accum_` terms -- keys times beta -- to
   * stay on the single-launch path.
   *
   * The `p_prod * bx` term that `MultKeyNoModDown` folds in is deliberately
   * absent: the only caller switches ciphertexts whose b-part is zero.
   */
  void MultKeyAccumNoModDown(
      Ct &accum, const std::vector<std::vector<DvConstView<word>>> &a_modups,
      const Ct &a_orig, const std::vector<const Evk *> &keys,
      bool accumulate) const;

  /**
   * @brief The mod-up half for a whole group of switches, into one buffer.
   *
   * Every switch in the group shares a level, a key shape and a set of
   * conversion tables, so on a single decomposition group they share three
   * kernel launches instead of taking five each -- and, more to the point, one
   * grid wide enough to fill the card. `a_coeffs` holds the group's
   * coefficient-domain inputs back to back; `buffer` comes back holding the
   * extended-basis results the same way, with `mod_up_views` indexing them per
   * switch and per decomposition group.
   */
  void ModUpForKeySwitchBatch(
      Dv &buffer, std::vector<std::vector<DvConstView<word>>> &mod_up_views,
      const Ct &a, const Evk &key, const DvConstView<word> &a_coeffs,
      int batch) const;

  /**
   * @brief Build the mod-switch machinery for key switches at `level` against
   * keys carrying `num_aux` auxiliary primes instead of `alpha_`.
   *
   * Call once, before any switch that uses such a key; the keys themselves are
   * made by asking `UserInterface` for that `num_aux`. Idempotent.
   *
   * **This is per-Context state.** The handler and P product built here live
   * in this Context, so a key made with a narrow basis and then switched
   * through a DIFFERENT Context over the same primes fails at
   * `MultKeyNoModDown` with "Invalid setting" -- not at the call that made the
   * key, and not with the narrow basis named in the message. A pipeline
   * holding two Contexts over one secret has to prepare both;
   * `UserInterface::PrepareModPackKeys` returns the count it chose for exactly
   * that reason.
   *
   * ## What the caller is responsible for
   *
   * The narrow P must still exceed the key-switch digit, or the switch adds
   * more noise than it removes. With `beta == 1` the digit is the whole
   * modulus at `level`, so the condition is `prod(first num_aux aux primes) >
   * Q_level`, and it wants margin rather than equality. On `sylphflow16_35` at
   * level 1 that is 2^75.1 against 2^31 per auxiliary prime: three of them is
   * 2^93 and four is 2^124. Nothing here checks it -- the accuracy of the
   * circuit is what checks it.
   *
   * @param level the level the switch happens at
   * @param num_aux auxiliary primes in the extended basis, in [1, alpha_]
   */
  void PrepareNarrowKeySwitch(int level, int num_aux) const;

  /**
   * @brief The mod-switch handler and P product for a key switch at `level`
   * whose key carries `num_aux` auxiliary primes.
   *
   * `num_aux == param_.alpha_` gives the ordinary pair, which is every switch
   * the library made until narrow bases existed. Anything else must have been
   * asked for through `PrepareNarrowKeySwitch` first.
   */
  const ModSwitchHandler<word> &GetModSwitchHandler(int level,
                                                    int num_aux) const;
  DvConstView<word> GetPProd(NPInfo &np, int num_aux) const;
};

template <typename word>
using ContextPtr = std::shared_ptr<Context<word>>;

template <typename word>
using ConstContextPtr = std::shared_ptr<const Context<word>>;

}  // namespace cheddar
