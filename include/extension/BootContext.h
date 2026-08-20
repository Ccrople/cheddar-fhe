#pragma once

#include <map>
#include <memory>
#include <vector>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootParameter.h"
#include "extension/EvalMod.h"
#include "extension/EvalSpecialFFT.h"

namespace cheddar {

enum class BootVariant {
  kNormal,             // Normal complex bootstrapping
  kImaginaryRemoving,  // Removes the imaginary part at the end
  kMergeTwoReal        // For developers' internal use
};

/**
 * @brief BootContext class for bootstrapping. This class is used to handle all
 * the precomputed data and to create an optimized computational flow for
 * bootstrapping. Minimum key-switching (min_ks) is supported, which
 * significantly reduces the number of evaluation keys at the cost of slower
 * execution time. To understand what min_ks does, refer to Kim, Jongmin, et al.
 * "ARK: Fully Homomorphic Encryption Accelerator with Runtime Data Generation
 * and Inter-Operation Key Reuse." 2022 55th IEEE/ACM International Symposium on
 * Microarchitecture (MICRO). IEEE, 2022.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class BootContext : public Context<word>,
                    public std::enable_shared_from_this<BootContext<word>> {
 private:
  using Base = Context<word>;
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  int log_scaleup_;
  Constant<word> scaleup_const_;

  double cts_const_;
  double stc_const_;

  DeviceVector<word> mod_max_intt_const_;

  std::map<int, EvalSpecialFFT<word>> eval_fft_;
  std::map<int, BootVariant> boot_variant_;
  std::unique_ptr<EvalMod<word>> eval_mod_;

  BootContext(const Parameter<word> &, const BootParameter &);

  int GetBootEnabledNumSlots(int num_slots) const;

  ContextPtr<word> GetContext();
  ConstContextPtr<word> GetContext() const;

 public:
  const BootParameter boot_param_;

  /**
   * @brief Creates a new instance of BootContext.
   *
   * @param param CKKS parameters
   * @param boot_param bootstrapping parameters
   * @return std::shared_ptr<BootContext<word>> a shared pointer to the new
   * BootContext instance
   */
  static std::shared_ptr<BootContext<word>> Create(
      const Parameter<word> &param, const BootParameter &boot_param);

  BootContext(const BootContext &) = delete;
  BootContext &operator=(const BootContext &) = delete;

  BootContext(BootContext &&) = default;

  virtual ~BootContext() = default;

  // To perform bootstrapping, follow the following steps.

  // 1. first prepare evalmod (only once) and special fft (for each num_slots)

  /**
   * @brief Prepares homomorphic modular reduction evaluation in a BootContext
   *
   */
  void PrepareEvalMod();

  /**
   * @brief Prepares homomorphic special FFT and IFFT evaluations in a
   * BootContext.
   *
   * @param num_slots number of slots in the ciphertext to be bootstrapped
   * @param variant boot variant (BootVariant::kNormal (default) /
   * BootVariant::kImaginaryRemoving / BootVariant::kMergeTwoReal)
   */
  void PrepareEvalSpecialFFT(int num_slots,
                             BootVariant variant = BootVariant::kNormal);

  // 2. Retrieve required rotation distances for performing bootstrapping.

  /**
   * @brief Add required rotation distances to an EvkRequest. The client needs
   * to create an EvkMap based on the information in the resulting EvkRequest.
   *
   * @param req EvkRequest to add the required rotation distances
   * @param num_slots number of slots in the ciphertext to be bootstrapped
   * @param min_ks whether to use minimum key-switching
   */
  void AddRequiredRotations(EvkRequest &req, int num_slots,
                            bool min_ks = false) const;

  // 3. Actual evaluation

  /**
   * @brief Perform bootstrapping. PrepareEvalMod() and PrepareEvalSpecialFFT()
   * should have been already done. Also, the client should provide all the
   * required evaluation keys in the evk_map by using the information obtained
   * from AddRequiredRotations().
   *
   * @param res bootstrapping result ciphertext
   * @param input input ciphertext
   * @param evk_map client-provided EvkMap
   * @param min_ks whether to use minimum key-switching
   */
  // The scaling constants the CtS and StC transforms bake in, and EvalMod
  // itself. Public for the same reason as the conversions below: standalone
  // SlotToCoeff leaves stc_const_ applied, and inside Boot that is undone by
  // scaleup_const_ and EvalMod. A caller crossing the boundary outside
  // bootstrapping has to compensate it explicitly, which it cannot do without
  // being able to read the constant.
  /** @brief The BootParameter this context was built with. */
  const BootParameter &GetBootParameter() const { return boot_param_; }

  /**
   * @brief The input scale SlotToCoeff's phases were compiled against.
   *
   * `stc_phases_` are LinearTransforms pinned to levels
   * `GetStCStartLevel() - i` with per-phase scales, and `stc_const_` is split
   * across them as its cube root -- so a wrong input scale does not shift the
   * result by a constant, it stops the three phases composing. Inside Boot the
   * input is whatever EvalMod left, not the canonical scale of that level, so a
   * caller invoking SlotToCoeff standalone has to reinterpret its ciphertext's
   * scale to this. That costs no level and no kernel: it changes the declared
   * scale, not the data.
   */
  double GetStCInputScale() const;

  double GetCtSConst() const;
  double GetStCConst(BootVariant variant = BootVariant::kNormal) const;
  void EvaluateMod(Ct &res, const Ct &input, const Evk &mult_key) const;

  // The encoding conversions, public because the Llama pipeline needs them and
  // not only Boot does. SlotToCoeff is compiled at GetStCStartLevel(), which is
  // the default encryption level, so an ordinary ciphertext can feed it;
  // CoeffToSlot is compiled at GetCtSStartLevel() = max_level, which only a
  // ModUp'd ciphertext reaches. That asymmetry is why [SYLPH] section 3.2 fuses
  // encoding conversions into bootstrapping, and why HalfBoot below exists.
  /**
   * @brief Lift the level-zero input to `target_level`.
   *
   * @param target_level level to climb to; -1 uses this context's
   * `BootParameter::GetMaxLevel()`, which is what Boot passes. It was formerly
   * always `param_.max_level_`, which is why a bootstrap could only ever land
   * where the parameter set put it.
   */
  void ModUpToLevel(Ct &res, const Ct &input, const EvkMap<word> &evk_map,
                    int target_level = -1) const;
  void CoeffToSlot(Ct &res, int num_slots, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;
  void SlotToCoeff(Ct &res, int num_slots, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;

  /**
   * @brief The partial conversions of [SYLPH] section 3.2: slots <-> the
   * Slots-in-Coefficients encoding the batch CC-MM operates in.
   *
   * `SlotToSinC` is the last `log2(degree / sub_degree)` butterfly stages of
   * SlotToCoeff and nothing else; `SinCToSlot` is the first that many of
   * CoeffToSlot, times `d^-1`. See `EvalSpecialFFT.h` for the identity and why
   * the *prefix* of StC is a different map.
   *
   * `num_phases` splits those stages across that many `LinearTransform`s, one
   * level each, for the same reason StC is split: a phase of `q` stages holds
   * `2^q` plaintexts. At `sub_degree = 512` one phase is 128 of them and one
   * level; at the `sub_degree = 32` the attention product needs it would be
   * 2048, so three phases -- the same three SlotToCoeff would have spent -- are
   * what make it fit.
   *
   * Unlike `CoeffToSlot`, `SinCToSlot` is compiled at a level the caller
   * chooses, so it does not have to live inside a bootstrap.
   *
   * `PrepareEvalSpecialFFT(num_slots)` must have run first.
   */
  void PrepareSinC(int num_slots, int sub_degree, int stc_level,
                   int cts_level, int num_phases = 1);
  /// Levels a conversion spends, one per phase.
  int GetSinCNumPhases(int num_slots) const;
  void AddRequiredSinCRotations(EvkRequest &req, int num_slots) const;
  void SlotToSinC(Ct &res, int num_slots, const Ct &input,
                  const EvkMap<word> &evk_map) const;
  void SinCToSlot(Ct &res, int num_slots, const Ct &input,
                  const EvkMap<word> &evk_map) const;

  void Boot(Ct &res, const Ct &input, const EvkMap<word> &evk_map,
            bool min_ks = false) const;

  /**
   * @brief Bootstrapping stopped before SlotToCoeff: coefficients in, slots out.
   *
   * This is [BAE]'s HalfBTS, and for this project it is not an optimisation but
   * the only way across the domain boundary. `EvalSpecialFFT` compiles
   * SlotToCoeff at `GetStCStartLevel()`, which is exactly
   * `default_encryption_level`, so an ordinary ciphertext can feed it. It
   * compiles CoeffToSlot at `GetCtSStartLevel()` = `max_level`, which nothing
   * but a ModUp'd ciphertext reaches. So slot -> coefficient is a plain call and
   * **coefficient -> slot exists only inside bootstrapping** -- which is why
   * [SYLPH] section 3.2 fuses encoding conversions into it.
   *
   * The level arithmetic closes exactly, which is the sign the parameter set
   * was built for this flow:
   *
   *     slot @19 --StC(3)--> coeff @16 --[ring descent, PCMM, ascent]-->
   *     --ModUp--> @31 --CtS(4)--> @27 --EvalMod(8)--> slot @19
   *
   * @param res output, slot-encoded at `GetEvalModStartLevel() - EvalMod`
   * @param input coefficient-encoded, brought to level 0 as `Boot` does
   * @param evk_map client-provided EvkMap
   * @param min_ks whether to use minimum key-switching
   */
  void HalfBoot(Ct &res, const Ct &input, const EvkMap<word> &evk_map,
                bool min_ks = false) const;

  // Other functions...

  /**
   * @brief Checks if bootstrapping is prepared for the given number of slots.
   *
   * @param num_slots number of slots in the ciphertext to be bootstrapped
   * @return true if bootstrapping is prepared for the given number of slots
   * @return false if bootstrapping is not prepared
   */
  bool IsBootPrepared(int num_slots) const;

  /**
   * @brief Whether PrepareEvalMod() has run, and so whether the scales derived
   * from it -- GetStCInputScale() in particular -- are known yet.
   *
   * A caller that only wants the level arithmetic should not have to prepare
   * anything, and without this it cannot tell the difference between "not
   * prepared" and a wrong scale except by tripping the assert.
   */
  bool IsEvalModPrepared() const { return eval_mod_ != nullptr; }

  /**
   * @brief Performs the trace operation. For s = start_rot_dist, and n =
   * num_accum, res = (input << s) + (input << 2s) + ... + (input << ns).
   *
   * @param res result ciphertext
   * @param start_rot_dist starting rotation amount
   * @param num_accum must be a power of 2
   * @param input input ciphertext
   * @param evk_map client-provided EvkMap
   */
  void Trace(Ct &res, int start_rot_dist, int num_accum, const Ct &input,
             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar