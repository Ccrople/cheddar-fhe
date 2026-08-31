#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/CiSwitchedCcmm.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootParameter.h"
#include "extension/ComplexLinearTransform.h"
#include "extension/LinearTransform.h"

namespace cheddar {

/**
 * @brief A class for the homomorphic evaluation of special FFT
 * (SlotToCoeff/StC) and IFFT (CoeffToSlot/CtS)
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class EvalSpecialFFT {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Complex = std::complex<double>;

  const int num_slots_;
  BootParameter boot_param_;

  const double cts_const_;
  const double stc_const_;
  const bool full_slot_;
  // The real subring runs the same stage matrices over a complex intermediate
  // carried as a pair of real ciphertexts, so its phases are a different type.
  // Exactly one of the two pairs of vectors below is populated.
  const bool conjugate_invariant_;

 public:
  /**
   * @brief The CoeffToSlot plaintext diagonals, held apart from the rest so
   *        two `BootContext`s over one parameter set can share one copy.
   *
   * **CtS does not depend on the slack and StC does.** `GetCtSStartLevel()` is
   * `max_level_` and `GetEvalModStartLevel()`/`GetEvalModEndLevel()` are
   * measured down from it; the slack enters only at
   * `GetStCStartLevel() = GetEvalModEndLevel() - num_slack_levels_`, and
   * `stc_const_` reads `GetEndLevel()` below that. So two BootContexts that
   * differ in nothing but the slack -- which is exactly the pair a
   * conjugate-invariant Llama layer holds, the leg's at slack zero for the
   * softmax walk and the FFN's at slack nine so `SlotToCoeff` compiles above
   * the `num_accum == 1` zone (Doing.md 1.5ct) -- compile CtS at the same
   * levels, from the same stage matrices, against the same constant, and
   * produce bit-identical plaintexts. `MemoryLedger` prepares the FFN's tables
   * both ways in one process and splits them: **CtS 3084.0 MiB**, StC 3324.5
   * at slack zero and 2342.5 at slack nine. So the pair of table sets is
   * 6408.5 and 5426.5 MiB, the 982.0 between them is the two StCs, and
   * **3084.0 MiB was one object built and held twice**.
   *
   * The vectors are the two possible phase types and exactly one is populated,
   * as for StC below.
   */
  struct CtSTables {
    std::vector<LinearTransform<word>> phases;
    std::vector<ComplexLinearTransform<word>> ci_phases;
  };

 private:
  // Never null after construction: either built here or adopted from a donor
  // whose parameters this one's constructor checked.
  std::shared_ptr<CtSTables> cts_;
  std::vector<LinearTransform<word>> stc_phases_;

  std::vector<ComplexLinearTransform<word>> stc_ci_phases_;

  std::vector<StripedMatrix> plain_fft_stages_;
  std::vector<StripedMatrix> plain_ifft_stages_;

  // The partial conversions of [SYLPH] section 3.2. Zero or one entry each;
  // a vector rather than an optional because LinearTransform has no default
  // constructor and owns device state.
  int sinc_sub_degree_ = 0;
  std::vector<LinearTransform<word>> sinc_stc_;
  std::vector<LinearTransform<word>> sinc_cts_;
  // The conjugate-invariant conversions run the same composed stage products
  // over a complex intermediate carried as a pair of real ciphertexts, like
  // CtS and StC themselves. Forward: the first phase's input columns outside
  // block 0 are doubled (the suffix analogue of the StC phase-0 correction).
  // Inverse: the last phase's rows carry the complex lambda correction of
  // Doing.md 1.5bn, solved on a 4k reference ring at PrepareSinC time; it is
  // only built for sub_degree <= 256, and reading the banded basis back
  // amplifies noise by ~2k/pi -- the same conditioning 1.5bj measured through
  // the ring switch, intrinsic to the direction, not to this transform.
  std::vector<ComplexLinearTransform<word>> sinc_stc_ci_;
  std::vector<ComplexLinearTransform<word>> sinc_cts_ci_;
  std::vector<LinearTransform<word>> sinc_prefix_;
  int sinc_prefix_shift_ = 0;
  int sinc_prefix_level_ = -1;

  std::pair<int, int> BSGSSplit(int num_diag) const;
  void PopulatePlainMatrices(ConstContextPtr<word> context);
  void PreparePlaintexts(ConstContextPtr<word> context);

 public:
  /**
   * @param shared_cts when non-null, the CoeffToSlot tables to adopt instead
   *        of building. **The caller owes the check**: the donor must have
   *        been compiled for the same `num_slots`, the same `cts_const`, the
   *        same `GetCtSStartLevel()` and `num_cts_levels_`, and against a
   *        Context with the same primes, because a plaintext is a device
   *        buffer of RNS limbs and nothing inside it records which parameter
   *        set it belongs to. `BootContext::PrepareEvalSpecialFFT` does that
   *        checking; going through it is the supported route.
   */
  EvalSpecialFFT(ConstContextPtr<word> context, const BootParameter &boot_param,
                 int num_slots, double cts_const, double stc_const,
                 std::shared_ptr<CtSTables> shared_cts = nullptr);

  EvalSpecialFFT(const EvalSpecialFFT &) = delete;
  EvalSpecialFFT &operator=(const EvalSpecialFFT &) = delete;
  EvalSpecialFFT(EvalSpecialFFT &&) = default;

  /// The CoeffToSlot tables, for another EvalSpecialFFT to adopt. Never null.
  std::shared_ptr<CtSTables> GetCtSTables() const { return cts_; }

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  void EvaluateCtS(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;
  void EvaluateStC(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;

  /**
   * @brief SLOTS <-> SinC, the "partial bit-reversal operation which occurs
   * when moving to SinC encoding" of [SYLPH] section 3.2.
   *
   * ## The identity
   *
   * Writing `n = num_slots`, `L = log2 n`, `d = degree / sub_degree` and
   * `p = log2 d`, the slot -> SinC(sub_degree) map is **exactly the last `p`
   * butterfly stages of StC**, in StC's own ascending-stride order, with no
   * scalar:
   *
   *     U = plain_fft_stages_[L-1] . ... . plain_fft_stages_[L-p]
   *
   *     input slot s  ->  SinC block  BitRev_p( s >> log2(sub_degree/2) )
   *                       SinC lane   s mod (sub_degree/2)
   *
   * The bit reversal is on the **block index only**; the lanes are untouched.
   * Both endpoints are the ones the definition collapses to: `sub_degree == 2`
   * gives `p = L`, the whole of StC and hence `EncodeCoeff`; `sub_degree ==
   * degree` gives `p = 0`, the identity and hence `Encode`.
   *
   * **The prefix is a different map.** The low-stride stages act inside a lane
   * block, which is the wrong axis, and using them gives the wrong answer for
   * every non-degenerate case. This is a suffix.
   *
   * The inverse is the mirror image: the **first** `p` stages of CtS, in CtS's
   * own order (descending stride), times `1/d` -- `ifft_i . fft_i = 2I` stage
   * by stage, so `p` stages compose to `2^p = d`.
   *
   * ## What it costs
   *
   * The suffix product has exactly `d` diagonals at stride `sub_degree/2`. At
   * `sub_degree = 512` that is 128 of them and one `LinearTransform` -- one
   * level, against the full StC's `num_stc_levels_`. At `sub_degree = 32`,
   * which is what Llama-3's per-head `128 x 128` product needs, it is 2048,
   * and 2048 plaintexts at eleven limbs do not fit anywhere. So the stages are
   * splittable across `num_phases` transforms exactly as StC's own are: three
   * phases of 4 + 4 + 3 stages are 16 + 16 + 8 = 40 plaintexts and three
   * levels, which is what StC would have cost anyway.
   *
   * Nothing here is min-KS or pre-rotated: a standalone conversion has no
   * neighbouring phase to share a rotation with.
   *
   * ## Why it is not `EvalSpecialFFT` with a different `num_slots`
   *
   * Because that selects the *low*-stride stages, not the high ones -- stage
   * `i` of an `n'`-slot transform uses the twiddle modulus `2^(i+3)`, and no
   * `n'` makes `{0..log2 n'-1}` coincide with `{L-p..L-1}`. A smaller
   * `num_slots` also switches on the sparse-packing path, which is a different
   * thing again.
   *
   * @param sub_degree k, a power of two dividing the ring degree
   * @param stc_level the level the slots -> SinC transform is compiled at; it
   *        returns `num_phases` below
   * @param cts_level likewise for SinC -> slots
   * @param num_phases how many `LinearTransform`s the `p` stages are split
   *        into, one level each. **This is the whole cost question.** A
   *        product of `q` stages has `2^q` diagonals and a diagonal is a full
   *        plaintext at the transform's limb count, so one phase carrying all
   *        `p` is `2^p` plaintexts -- 2048, i.e. gigabytes, at the
   *        `sub_degree = 32` the attention product wants -- while three phases
   *        of 4 + 4 + 3 are 40. It is the same trade StC itself makes
   *        (`num_stc_levels_`, three on every logN=16 preset), and it is
   *        level-neutral in the pipeline, because a tensor bound for the
   *        product pays this *instead of* SlotToCoeff rather than on top of it.
   */
  void PrepareSinC(ConstContextPtr<word> context, int sub_degree,
                   int stc_level, int cts_level, int num_phases = 1);

  /// How many levels a conversion spends: one per phase.
  int GetSinCNumPhases() const {
    return static_cast<int>(conjugate_invariant_ ? sinc_stc_ci_.size()
                                                 : sinc_stc_.size());
  }
  /// The BSGS split of each forward phase, for reports.
  std::pair<int, int> GetSinCPhaseBSGS(int phase) const {
    if (conjugate_invariant_) {
      return {sinc_stc_ci_.at(phase).GetBS(), sinc_stc_ci_.at(phase).GetGS()};
    }
    return {sinc_stc_.at(phase).GetBS(), sinc_stc_.at(phase).GetGS()};
  }

  /// The `sub_degree` PrepareSinC was called with, or 0.
  int GetSinCSubDegree() const { return sinc_sub_degree_; }

  /**
   * @brief The PREFIX of SlotToCoeff -- the part HalfBoot leaves undone.
   *
   * `SlotToCoeff` is `num_stages` butterfly stages and `SlotToSinC` is the
   * last `p = log2(degree/sub_degree)` of them, so `StC = SlotToSinC . P` with
   * `P` the first `num_stages - p`. A ciphertext holding `SlotToSinC(s)` is
   * therefore `StC(P^-1(s))`, and `HalfBoot` -- which inverts the whole of StC
   * -- hands back `P^-1(s)`, not `s`. This is `P`.
   *
   * It is not a relabelling that could be absorbed into the layout: the
   * butterfly stages combine values with twiddle factors, so what comes back
   * without it is a linear MIXTURE of the scores rather than a permutation of
   * them.
   *
   * At the attention product's `sub_degree = 32` it is 4 stages and 31
   * diagonals -- one level, which the schedule spends on `Canonicalise`'s
   * multiply anyway, plus one HRot for the window (see the .cpp).
   *
   * @param level the level it is compiled at, i.e. where HalfBoot lands
   * @param num_phases how many LinearTransforms to split the stages across,
   *        one level each; 1 is right at `sub_degree = 32`
   * @param constant a scalar folded into the transform's own matrix, and
   *        `pt_scale` the scale its plaintexts are encoded at. Together they
   *        are `SylphSchedule::Canonicalise`: that operator is one multiply by
   *        a constant and one rescale, at the level HalfBoot lands on, which
   *        is exactly this level. Folding it in here is what makes the prefix
   *        free -- the schedule was spending the level anyway -- and it is
   *        also what restores the magnitude HalfBoot divided out, which is
   *        worth about five bits. `pt_scale <= 0` means the default,
   *        `GetRescalePrimeProd(level)`, which preserves the input scale and
   *        folds nothing.
   */
  void PrepareSinCPrefix(ConstContextPtr<word> context, int sub_degree,
                         int level, int num_phases = 1,
                         double constant = 1.0, double pt_scale = -1.0);

  /**
   * @brief The composed prefix matrix and the window rotation it needs, for a
   * caller that wants to compile more than one of them.
   *
   * `PrepareSinCPrefix` holds exactly one, which is enough when the constant
   * folded into it is the same every time. The two attention products need
   * different ones -- `Canonicalise`'s magnitude is `2 / size_scores` for the
   * scores and 1 for the values -- so `SinCAttention` builds its own pair from
   * this. The window is the `additional_pt_rot` the transform must be given,
   * with `pre_rotation = -window`; the result then comes back rotated by
   * `window` and one HRot by `-window` finishes it. See the .cpp for why a
   * window is needed at all.
   *
   * @param sub_degree k
   * @param window receives `2^(num_stages - p)`
   */
  StripedMatrix SinCPrefixMatrix(int sub_degree, int &window) const;

  /// Levels the prefix spends, or 0 if it was never prepared.
  int GetSinCPrefixNumPhases() const {
    return static_cast<int>(sinc_prefix_.size());
  }
  /// The residual rotation the window leaves, undone inside EvaluateSinCPrefix.
  int GetSinCPrefixShift() const { return sinc_prefix_shift_; }

  void AddRequiredSinCRotations(EvkRequest &req) const;
  void AddRequiredSinCPrefixRotations(EvkRequest &req) const;

  void EvaluateSlotToSinC(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
  void EvaluateSinCToSlot(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
  void EvaluateSinCPrefix(ConstContextPtr<word> context, Ct &res,
                          const Ct &input, const EvkMap<word> &evk_map) const;
};

/**
 * @brief The one-phase, one-level form of the conjugate-invariant SinC
 * conversions, for rings that cannot host `EvalSpecialFFT` at all.
 *
 * ## Why this exists (Doing.md 1.5bo)
 *
 * The CI CC-MM chain needs a PER-PART conversion on the small product ring:
 * after `SlotToSinC(degree / rank)` on the big ring, the switch's parts land
 * in slot form, one small-ring conversion short of the Algorithm 4 operand.
 * The small ring holds two levels above its floor and `EvalSpecialFFT`'s
 * constructor compiles a full CtS/StC against a BootParameter -- neither fits.
 *
 * What makes one phase possible where `PrepareSinC` demands two on R+: the
 * pair chain exists to carry a complex INTERMEDIATE between phases, and a
 * single phase has none. For real input x the whole conversion is
 * `Re(M x) = Re(M) x` -- one plain `LinearTransform` over the real part of
 * the corrected stage product, one level, ordinary BSGS (a gs = 1 layout is
 * fine here; there is no fused complex giant step involved).
 *
 * The matrices carry the corrections of Doing.md 1.5bn: the forward doubles
 * every input column outside block 0; the inverse rides 1/d and the complex
 * per-row lambda (solved on the 4k reference ring, capped at
 * sub_degree 256), taken real AFTER the lambda multiplication.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiSinCConverter {
 private:
  using Ct = Ciphertext<word>;

  int sub_degree_ = 0;
  // Zero or one entry each; a vector because LinearTransform has no default
  // constructor and owns device state.
  std::vector<LinearTransform<word>> forward_;
  std::vector<LinearTransform<word>> inverse_;

  // Deserializing constructor; see Save/Load.
  struct FromArchive {};
  CiSinCConverter(FromArchive, ArchiveReader &ar);

 public:
  /**
   * @param context a conjugate-invariant Context
   * @param sub_degree k
   * @param forward_level level the slots -> SinC direction is compiled at
   *        (it returns one below), or -1 to skip building it
   * @param inverse_level likewise for SinC -> slots; the inverse is only
   *        buildable for sub_degree <= 256 (the lambda cap)
   * @param chain when non-null, the CC-MM chain layout whose operand the
   *        conversions serve, and both directions fold the mixed-radix
   *        identity's block maps in (Doing.md 1.5bq). The FORWARD then
   *        consumes slots holding each entry at its PRIMARY `LocateSlot`
   *        address only -- a bijective layout, no host-side sums -- and
   *        produces the chain operand: the copy-add `y = x + pi2 x` is one
   *        column relabelling of the composed matrix, and the INVERSE folds
   *        g's block-level scan in, so it consumes the chain's output and
   *        returns the true (unsummed) values at their primary addresses.
   *        Every map here lives on the stride-`sub_degree` diagonal lattice,
   *        so the fold cannot grow the diagonal count past `degree /
   *        sub_degree` -- the composed transform's own ceiling: no extra
   *        level, no extra rotations beyond the BSGS the count implies.
   * @param forward_premap when non-null, a lane-preserving slot permutation
   *        folded into the FORWARD on its input side (Doing.md 1.5bx), given
   *        at block granularity: `(*forward_premap)[b]` is the block address
   *        the forward's own convention (primary `LocateSlot` blocks under
   *        `chain`, flat SinC blocks without) assigns to the entry the
   *        caller's layout holds at block `b` of the slot index, lanes -- the
   *        low `log2(sub_degree)` slot bits -- untouched. This is how a
   *        transport whose composed map is a block permutation (the Llama
   *        leg's Q/V transport, and the intra-ciphertext remainder of K's)
   *        rides the conversion for free: a column relabelling of the
   *        composed matrix, on the same lattice, same ceiling. Must be a
   *        bijection over `degree / sub_degree` blocks; ignored when the
   *        forward is not built.
   * @param baby_steps when positive, the BSGS baby-step count to compile
   *        both directions with, overriding the default split.
   *
   *        The default is `sqrt(num_diag)`, which balances a baby step
   *        against a giant step -- and on R+ those do NOT cost the same.
   *        1.5be measured the ratio at ~7 (a baby step rides the shared
   *        ModUp as one fused key multiply; a giant step pays its own
   *        ModDown + ModUp), so the balanced split is `sqrt(7 num_diag)`,
   *        and `BSGSSplit` already uses it for the bootstrap's CI phases.
   *        It caps those at 16 baby steps for `GSFusedComplexKernel`'s
   *        registers -- a cap that does NOT apply here, since these are
   *        single-ciphertext `LinearTransform`s and the fused complex
   *        giant step lives in `ComplexLinearTransform`. The default is
   *        left on the balanced split until the ratio is measured at the
   *        converter's own level and shape; this parameter is how it gets
   *        measured.
   */
  CiSinCConverter(ConstContextPtr<word> context, int sub_degree,
                  int forward_level, int inverse_level,
                  const CiSwitchedCcmmLayout *chain = nullptr,
                  const std::vector<int> *forward_premap = nullptr,
                  int baby_steps = 0);

  // disable copying (or moving also)
  CiSinCConverter(const CiSinCConverter &) = delete;
  CiSinCConverter &operator=(const CiSinCConverter &) = delete;

  /**
   * @brief Write the compiled conversions.
   *
   * This is the single most expensive object the Llama leg builds. The
   * constructor's work is host-side matrix composition -- the SinC stage
   * matrices with the chain layout's block maps and the transport premap, the
   * copy-add and its inverse folded into the diagonals -- and then 2048
   * diagonals per direction encoded on the device. Measured on the A100
   * (96 host threads): the P/V pair is 37 s -- forward: stages 5 s, folds
   * 1.3 s, compile 3 s; inverse: stages 5 s, folds 19.5 s, compile 2.7 s
   * -- against ~255 s with the encoding on the host and the folds on one
   * core. The leg's three were 728-803 s then; every one of the model's 32
   * layers reuses them unchanged.
   *
   * What is NOT written is the recipe: the sub-degree and the levels are
   * recorded, but the chain layout and the premap that shaped the matrices are
   * not, because they are already baked into the diagonals. A caller who
   * changes either must invalidate its cache; the archive identity only sees
   * the parameter set, so it cannot catch that.
   */
  void Save(ArchiveWriter &ar) const;

  /**
   * @brief Rebuild a converter written by `Save`.
   *
   * Returns a `unique_ptr` because this class is neither copyable nor
   * movable, which is also how `CiSinCAttention` already holds its three.
   */
  static std::unique_ptr<CiSinCConverter> Load(ArchiveReader &ar);

  int GetSubDegree() const { return sub_degree_; }

  /// The compiled directions, or null for one that was not built.
  const LinearTransform<word> *GetForward() const {
    return forward_.empty() ? nullptr : &forward_.front();
  }
  const LinearTransform<word> *GetInverse() const {
    return inverse_.empty() ? nullptr : &inverse_.front();
  }

  void AddRequiredRotations(EvkRequest &req) const;

  void SlotToSinC(ConstContextPtr<word> context, Ct &res, const Ct &input,
                  const EvkMap<word> &evk_map) const;
  void SinCToSlot(ConstContextPtr<word> context, Ct &res, const Ct &input,
                  const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar