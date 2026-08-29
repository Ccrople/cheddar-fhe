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
  double message_ratio_;

  DeviceVector<word> mod_max_intt_const_;

  std::map<int, EvalSpecialFFT<word>> eval_fft_;
  std::map<int, BootVariant> boot_variant_;
  std::unique_ptr<EvalMod<word>> eval_mod_;

  BootContext(const Parameter<word> &, const BootParameter &);

  int GetBootEnabledNumSlots(int num_slots) const;

  /**
   * @brief Step 3 of the bootstrap, shared by Boot and HalfBoot: take what
   * CoeffToSlot produced through EvalMod.
   *
   * Three shapes, and which one applies is a fact about the ring rather than
   * about the schedule. The ordinary ring's CtS leaves the coefficient vector
   * folded into complex slots, so the halves have to be separated before a
   * real modular reduction can act on them -- two EvalMod calls at full slot,
   * or one plus a conjugation key switch when sparse packing leaves room to
   * merge them. On the real subring the slots are already real and there is
   * nothing to separate: one EvalMod, no conjugation key, and none is built.
   */
  void EvaluateModAfterCtS(Ct &res, Ct &main_ct, bool full_slot,
                           const EvkMap<word> &evk_map) const;

  /**
   * @brief The full-slot separation, stopped before it recombines.
   *
   * `EvaluateModAfterCtS`'s full-slot branch splits the complex slot vector
   * into its two real axes, reduces each, and folds them back together. The
   * fold is the last two lines and nothing before it depends on them, so a
   * caller that wants the axes *as two ciphertexts* gets them for the price of
   * the reduction it was paying anyway -- no extra key switch, no extra level.
   *
   * `main_ct` arrives at EvalMod's start scale (the caller sets it, as
   * `EvaluateModAfterCtS` does); `lo` and `hi` leave carrying the real and
   * imaginary axes respectively, at EvalMod's end scale.
   */
  void SplitAndEvaluateMod(Ct &lo, Ct &hi, const Ct &main_ct,
                           const EvkMap<word> &evk_map) const;

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
   * @param cts_donor when non-null, another BootContext whose CoeffToSlot
   *        tables this one adopts instead of building its own.
   *
   * **Only StC depends on the slack.** `GetCtSStartLevel()` is `max_level_`
   * and EvalMod's start and end are measured down from it; the slack enters at
   * `GetStCStartLevel()` and nowhere above it, and `cts_const_` reads only
   * levels above it. So a pair of BootContexts differing in nothing but the
   * slack -- which is what a conjugate-invariant Llama layer holds, the leg's
   * at slack zero for the softmax walk and the FFN's at slack nine so that
   * `SlotToCoeff` compiles clear of the `num_accum == 1` zone (Doing.md
   * 1.5ct) -- builds the SAME CoeffToSlot plaintexts twice. `MemoryLedger`
   * prepares the FFN's tables both ways in one process and splits them: CtS
   * **3084.0 MiB**, StC 3324.5 at slack zero and 2342.5 at slack nine. The
   * pair of sets is 6408.5 and 5426.5 MiB, the 982.0 between them is the two
   * StCs, and **3084.0 MiB is the duplicate** -- with its build time.
   *
   * Adoption is by `shared_ptr`, so the order the two Contexts are released in
   * does not matter: `ReleaseEvalSpecialFFT` on the donor drops its StC and
   * its reference, and the CtS survives for as long as a borrower holds one.
   *
   * Every condition that makes the donor's tables *these* tables is asserted
   * here -- primes, degree, ring, level configuration, CtS compile level,
   * phase count, constant -- because a plaintext is a device buffer of RNS
   * limbs and carries no record of what it was encoded against. A donor from
   * an unrelated parameter set would decode as noise, silently.
   */
  void PrepareEvalSpecialFFT(int num_slots,
                             BootVariant variant = BootVariant::kNormal,
                             const BootContext<word> *cts_donor = nullptr);

  /**
   * @brief Drop the CoeffToSlot and SlotToCoeff plaintext diagonals compiled
   *        for `num_slots`, keeping everything else this Context holds.
   *
   * They are the largest single object a BootContext owns and they are dead
   * the moment a pipeline stops bootstrapping. `MemoryLedger` measures them at
   * **6408.5 MiB** on `ci16_35` at 65536 slots, against 7059.0 MiB of rotation
   * keys and 61.1 MiB for the Context itself.
   *
   * A conjugate-invariant layer holds two BootContexts over one secret -- the
   * leg's at slack zero and the FFN's at slack nine, because SlotToCoeff is
   * compiled at `GetStCStartLevel()` and the softmax walk needs `GetEndLevel()`
   * at 16 (1.5ct) -- and the leg's is dead after its eight Boots. Dropping it
   * outright is not available: a `ContextPtr` is a `shared_ptr` and the
   * EvkMap every later key lookup goes through lives in a `UserInterface` that
   * holds one. So the Context has to stay and its tables have to go, which is
   * this call and nothing more.
   *
   * After it, `IsBootPrepared(num_slots)` is false again and every entry point
   * that reads the tables -- `Boot`, `HalfBoot`, `CoeffToSlot`, `SlotToCoeff`,
   * `AddRequiredRotations`, the SinC family -- throws at `eval_fft_.at` until
   * `PrepareEvalSpecialFFT` runs again. Rotation keys are the caller's and are
   * untouched, so a re-prepare needs no new key material.
   *
   * If another BootContext adopted these CoeffToSlot tables (see
   * `PrepareEvalSpecialFFT`'s `cts_donor`), this drops the SlotToCoeff half and
   * this Context's reference to the CtS half, and the CtS itself lives on for
   * the borrower -- so the bytes returned are the 3324.5 MiB of StC rather than
   * the 6408.5 of both, and the layer's ledger row shrinks by exactly the
   * 3084.0 the donation had already saved earlier.
   *
   * @param num_slots the slot count the tables were compiled for
   * @return whether anything was dropped
   */
  bool ReleaseEvalSpecialFFT(int num_slots);

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

  /**
   * @brief What `HalfBoot` leaves on the message, exactly.
   *
   * `ModRaise` reads the level-zero ciphertext modulo `q0`, so `EvalMod` acts
   * on `m * level_zero_scale / q0`: the crossing multiplies the message by
   * that ratio, and a caller that stops at slots -- which is every caller of
   * `HalfBoot` -- owes its inverse to whatever reads the result.
   *
   * `BootParameter::GetLogMessageRatio()` is the ratio the design ASKS for and
   * `log_scaleup_` is built from rounded logarithms, so what the crossing
   * actually applies is `2^-log_message_ratio` divided by
   * `q0_prod / 2^round(log2 q0_prod)` -- and a product of NTT-friendly primes
   * is not a power of two. Every bootstrappable preset here, as
   * `nominal / actual`:
   *
   *     bootparam_30        0.843752    2^-4.7549
   *     ci16_35, ci16_40    0.988037    2^-4.9826
   *     bootparam_35, _40   1.009705    2^-5.0139
   *     sylphflow16_35      1.046448    2^-5.0655
   *     bootparam_40_64bit  1.000000    2^-5.0000
   *
   * The last row is why the nominal was ever plausible: its `q0` is the SINGLE
   * 64-bit prime `2^50 + 14337`, so the ratio is right to ten digits. Every
   * 32-bit preset needs two primes to reach the same modulus and their product
   * lands wherever it lands.
   *
   * Two headers used to say this "has to be measured, not derived".
   * `CrossingConstantTest` measures it against this accessor on three of those
   * presets and agrees to **five digits** -- 0.999986, 0.999991, 0.999991 --
   * while the nominal power of two is out by 15.6%, 0.97% and 4.6%
   * respectively, so the test is not passing on a coincidence. The two fits
   * the tree carried agree too: the SinC leg's 0.0298533 on `sylphflow16_35`
   * against 0.0298629, and the conjugate-invariant FFN, which now checks its
   * own in-run fit against this accessor and reports **fit/derived 0.999998**
   * -- six digits, through the leg's own noise rather than on clean
   * coefficients.
   *
   * The fit that this replaces was carried in `LlamaBlockTest` for EVERY
   * preset and is consumed only by `Mode::kFull`, which skips off
   * `sylphflow16_35` -- so it was latent rather than live, and it stays
   * latent exactly as long as that skip does. On `bootparam_35` the same
   * literal is 3.7% wrong and on `bootparam_30` it is 19% wrong, with
   * nothing in the pipeline able to say so.
   *
   * It does NOT appear in a full turn of `SylphSchedule`'s cycle: `ToCoeff`
   * scales down by the nominal ratio and `SlotToCoeff` scales up by this one,
   * so the pair cancels and the coefficient leg simply runs at `nominal /
   * actual` of what the caller believes. Only a leg that crosses one way --
   * `SinCLinearLeg`, which has no `ToCoeff` to cancel against -- pays it.
   */
  double GetMessageRatio() const { return message_ratio_; }

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

  /**
   * @brief The other half of the SinC round trip: what `HalfBoot` leaves
   * undone.
   *
   * `SinCToSlot` above converts a SinC ciphertext to slots *in the same ring
   * and at the caller's level*. It is not the way home from the matrix
   * product, because the product leaves its result at level 0 and the only
   * route out of level 0 is a bootstrap.
   *
   * `HalfBoot` inverts the WHOLE of SlotToCoeff, and a SinC-encoded ciphertext
   * is `StC(P^-1(s))` with `P` StC's prefix -- so what lands in slots is
   * `P^-1(s)`, a twiddle-weighted mixture of the values rather than a
   * permutation of them. `SinCPrefix` applies `P` and finishes the trip.
   *
   * One level, which a [SYLPH] schedule is already spending on
   * `Canonicalise`'s multiply at the same place, plus one HRot.
   */
  void PrepareSinCPrefix(int num_slots, int sub_degree, int level,
                         int num_phases = 1, double constant = 1.0,
                         double pt_scale = -1.0);
  int GetSinCPrefixNumPhases(int num_slots) const;
  /// The composed prefix matrix and its window, for a caller compiling its
  /// own -- see `EvalSpecialFFT::SinCPrefixMatrix`.
  StripedMatrix SinCPrefixMatrix(int num_slots, int sub_degree,
                                 int &window) const;
  void AddRequiredSinCPrefixRotations(EvkRequest &req, int num_slots) const;
  void SinCPrefix(Ct &res, int num_slots, const Ct &input,
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

  /**
   * @brief Two real-payload coefficient ciphertexts through ONE HalfBoot.
   *
   * THE HALF THAT WAS ALREADY BEING PAID FOR. At full slots the ordinary ring's
   * CtS folds coefficients `j` and `j + N/2` into the real and imaginary axes of
   * slot `j`, so `EvaluateModAfterCtS` separates the axes and reduces each --
   * two EvalMods, always. A pipeline whose payload is real-axis-only in slots
   * (which is every operator in this layer: `SpreadNormWeight` and every
   * calibration constant write `Complex(v, 0)`) has StC leave coefficients
   * `N/2..N-1` **zero**, so the second of those two EvalMods runs on an
   * identically-zero ciphertext, once per bootstrap.
   *
   * Filling that half costs one monomial multiply and one add. `X^{N/2}` is the
   * imaginary unit's polynomial -- it evaluates to `i` at every slot, because
   * `5^j = 1 mod 4` -- so `MultImaginaryUnit` *is* the shift by `N/2`, level
   * free and key free, and `hi`'s own upper half being zero means the shift
   * never wraps. Splitting the result back out is free by construction: it is
   * `SplitAndEvaluateMod` stopping one line before the fold.
   *
   * So a pair costs one HalfBoot, not two, and the arithmetic each half sees is
   * the arithmetic it would have seen alone. The one thing the caller owes is
   * the contract:
   *
   * **Both inputs must carry payload in coefficients `0 .. N/2-1` only.** That
   * is what StC produces from real slots and what the PC-MM preserves (its mix
   * is entrywise over each module component, so zeros stay zero), but it is not
   * checked here and cannot be -- a violation silently adds `hi`'s upper half
   * into `lo`'s payload.
   *
   * @param res_lo slots of `lo`, exactly as `HalfBoot(lo)` would have left them
   * @param res_hi slots of `hi`, likewise
   * @param lo coefficient-encoded, upper coefficient half zero
   * @param hi coefficient-encoded, upper coefficient half zero, same level and
   *        scale as `lo`
   */
  void HalfBootPair(Ct &res_lo, Ct &res_hi, const Ct &lo, const Ct &hi,
                    const EvkMap<word> &evk_map, bool min_ks = false) const;

  /**
   * @brief `HalfBootPair` for a caller that already holds the merged form.
   *
   * The merge is `lo + X^(N/2) * hi` and a producer upstream may be able to
   * make it more cheaply than one multiply and one add here -- the projection
   * can, by merging at the module component and halving its ModPack
   * (`MlweHandler::AddShiftedHalf`), which is worth far more than the merge
   * itself. Such a caller has no unmerged pair to hand `HalfBootPair`.
   *
   * @param merged coefficient-encoded, payload of one ciphertext in
   *        coefficients `0 .. N/2-1` and of the other in `N/2 .. N-1`
   */
  void HalfBootSplit(Ct &res_lo, Ct &res_hi, const Ct &merged,
                     const EvkMap<word> &evk_map, bool min_ks = false) const;

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