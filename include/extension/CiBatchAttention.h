#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/GpuTimer.h"
#include "core/BatchCcmm.h"
#include "core/CiLift.h"
#include "core/CiSwitchedCcmm.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "core/RingSwitch.h"
#include "extension/BootContext.h"
#include "extension/CiBatch.h"
#include "extension/CiSinCBasis.h"
#include "extension/EvalPoly.h"
#include "extension/EvalSpecialFFT.h"

namespace cheddar {

/**
 * @brief The attention products on the BATCHED layout (`CiBatchLayout`):
 * per head, `Q K^T` and `P V` as [KANG] Algorithm 4 on the lifted CI chain,
 * one instance GROUP at a time.
 *
 * ## The layout the products see
 *
 * A channel ciphertext of the batched layout, converted slots -> SinC at
 * sub-degree `lanes` (32) with the chain layout folded in, holds SinC block
 * `BitRev(token * rank + group)`, lane `instance % lanes` (`CiBatchLayout`
 * chain-addressed). The CI ring switch then yields `rank` (16) parts, and
 * part `g` is the SAME channel over the 128 tokens of the 32 instances of
 * group `g` -- exactly one column of a [KANG] matrix encryption at d = 128
 * on the product ring. So per (head, group) the 128 channel parts of Q are
 * the column-wise encryption of `Q_hg` (rows = tokens), and everything the
 * single-prompt leg did to bring heads and tokens into place (the
 * exchange, the cross, the merge) does not exist here.
 *
 * ## `Q K^T` transposes nothing
 *
 * Algorithm 4 contracts the CIPHERTEXT index of both operands once its step
 * 1 has made the rhs row-wise. K as projected -- ciphertext = channel,
 * blocks = tokens -- IS the row-wise encryption of `K^T`, so the score
 * product runs `BatchCcmmHandler::Multiply(..., rhs_row_wise = true)` on
 * Q's and K's parts as they are. The lifted twist of Doing.md 1.5bl then
 * lands on the KEY-TOKEN axis (column l comes back as
 * `S[:, l] + cos(theta) S[:, d - l]`), and confining K's live key tokens
 * to blocks < d/2 kills it for every l < d/2: measured exact
 * (`CiBatch.TheElidedScoreProductHoldsUnderTheContract`, 2.9e-07 at the
 * chain's floor). Two calls a head cover the key tokens 0..63 and
 * 64..127: the second half is masked in K's own RoPE plaintexts and shifted
 * down by 64 through a forward converter whose block premap is that shift.
 *
 * ## `P V` is the standard orientation
 *
 * P comes back with ciphertext = key token and blocks = query tokens; V is
 * ciphertext = channel, blocks = key tokens. The contraction is P's
 * ciphertext index against V's BLOCK index -- Algorithm 4 as written, V
 * through step 1's CMT -- under 1.5bl's contract as the single-prompt leg
 * runs it: P's ciphertexts 0..63 / 64..127 are the two calls' lhs
 * verbatim, V's key tokens confined per call (masked, the odd call
 * through the shifted converter). The output is ciphertext = channel,
 * blocks = query tokens: the layout's own form, straight into the O
 * projection after the return.
 *
 * ## Levels, tight
 *
 *     projection output @ rope_level (5)  --RoPE(+mask)-->  4
 *       = forward_level  --SlotToSinC-->  3 = chain level
 *       --RingSwitch--> parts @3  --Lift-->  Algorithm 4 @3 -> 2
 *       --Descend, SwitchBack @2--> --SinCToSlot @ inverse_level (2)--> 1
 *
 * so a product's output lands in slots at level 1: the scores go to a
 * `Boot` (which takes any level), the attention output to the O projection
 * (1 -> 0) and the residual at 0. Nothing is LevelDowned past work: the
 * attention norm holds at 7 (y at 6), the projections land at 5, RoPE at
 * 4 = the forward level. The softmax fits the shipped `ci16_35` landing
 * (16) because the Euclidean norm's affine multiply rides the causal mask.
 *
 * ## Keys, on three rings
 *
 * The forward and inverse ring-switching keys on the switching Context (at
 * the chain level), the rotations of the three converters on the same
 * Context (`AddSwitchRotations`), the automorphism and multiplication keys
 * on the lifted ring (`LiftedRotationIndices`).
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiBatchAttention {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

 public:
  struct Config {
    int num_tokens = 128;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    //! The big-ring SinC sub-degree = the lanes of a group. 32 on the
    //! shipped chain (`ci16_35` / `ci12_35_boot` / `ringdegree13_35_boot`):
    //! the product ring's 4096 = 128 tokens x 32 lanes.
    int sub_degree = 32;
    //! The projections' output level; RoPE lands one below, ON the forward
    //! level when the attention norm holds at 7 (y at 6, projections at 5).
    int rope_level = 5;
    //! The slots -> SinC forward runs here and lands one below, where the
    //! chain runs.
    int forward_level = 4;
    //! The SinC -> slots return runs here and lands one below.
    int inverse_level = 2;
    double rope_base = 500000.0;  //!< Llama-3's theta
    int converter_baby_steps = 256;
    //! B512_ccmm_ideas idea [4], the way Doing.md 3.16 fused the leg: the
    //! SCORES' return conversion is absorbed into their bootstrap. Scores
    //! then hands back the SwitchBack outputs (the nested SinC element,
    //! chain scale) and `BootScoresFused` runs `HalfBootTowerBatch` on the
    //! TOWER ring plus the lane prefix, landing the booted scores at the
    //! layer's top level with the carried factor in the message -- no
    //! `SinCToSlot`, no full Boot, one level given back to the chain's
    //! side. Needs the `tower` BootContext (K = 64, `ci16_35_land17c3e10`,
    //! its SSE secret tower-sparse). Values' returns keep the converter.
    bool fused_scores = false;
    //! The landing-15 lever (the level idea [4] freed): the softmax's
    //! affine MULTIPLY `a1 = 2 / (span * carried)` is folded into the
    //! fused boot's lane-prefix plaintexts (both factors are configuration
    //! constants; the prefix is re-encoded lazily at the first
    //! `BootScoresFused` once `carried` is observed), so the walk starts
    //! at exp directly -- 11 levels above `forward_level` 4, which is what
    //! a Boot landing at 15 has. Requires `fused_scores` and a causal
    //! calibration; `SoftMax` then expects the affine already applied and
    //! only adds the row shift.
    bool affine_in_prefix = false;
    //! [3] the aux boot split (Doing.md 7.39): where the booted scores
    //! land. 0 = the layer boot's own landing (no change). With
    //! `SetAuxBoot` the softmax's Euclidean-norm accumulator (ONE
    //! ciphertext a head) bootstraps on its own short ring and the walk
    //! above `forward_level` shrinks to exp + mask + the two products --
    //! 8 levels at exp degree 15 -- so the scores can land at 12: a
    //! shorter tower for the fused boots (`ci16_35_land13c3e10v3`, EvalMod
    //! ending at 13), and the serial route LevelDowns to it. Requires
    //! `affine_in_prefix` on the fused route (the affine multiply has no
    //! level of its own at 12).
    int score_top = 0;
    //! Run on the PLAIN slot map `Slot(t, b) = t * B + b` instead of the
    //! chain addressing, with the chain's block map folded into the two
    //! converters as a premap (`CiSinCConverter`'s `forward_premap` /
    //! `inverse_premap`, the block index's bit reversal). It is free: those
    //! transforms are already on the stride-`sub_degree` diagonal lattice
    //! and already at its `degree / sub_degree` ceiling, so a block
    //! relabelling costs no diagonal, no plaintext byte and no key
    //! (PcPremapTest). Nothing inside the attention reads the map except
    //! the per-token plaintexts, which follow it.
    //!
    //! What it BUYS is outside: a PC-attention operand is constant over the
    //! token axis and varies over the instance axis, and under this map that
    //! is a k = B subring element -- 512 coefficients of 65536, and lane t
    //! is instance t exactly. Under the chain map the token sits in the
    //! middle slot bits and the same operand is dense.
    //!
    //! Not compatible with `fused_scores`: the tower's lane prefix mixes
    //! only WITHIN a lane group, so its offsets are off the block lattice
    //! and a block relabelling multiplies the two sets (63 -> 15309).
    bool plain_map = false;
    bool verbose = false;
  };

  /** @brief The keys one call needs, on three rings (four when fused). */
  struct Keys {
    const EvkMap<word> *boot = nullptr;    //!< the layer's ring (the shift)
    const EvkMap<word> *swtch = nullptr;   //!< the switching ring
    const EvkMap<word> *lifted = nullptr;  //!< the lifted ordinary ring
    const EvkMap<word> *tower = nullptr;   //!< the tower ring (fused scores)
    const Evk *ring_switch = nullptr;
    const Evk *inverse_ring_switch = nullptr;
  };

  /**
   * @param boot the batched layer's ring (`ci16_35`)
   * @param switch_ctx the switching Context sharing `boot`'s bottom primes
   *        and secret (`ci_ringswitch16_35_boot`)
   * @param small_ctx the conjugate-invariant product ring (`ci12_35_boot`)
   * @param lifted_ctx the ordinary ring of the small ring's conductor
   *        (`ringdegree13_35_boot`), its keys on the lifted secret
   */
  CiBatchAttention(std::shared_ptr<const BootContext<word>> boot,
                   ConstContextPtr<word> switch_ctx,
                   ConstContextPtr<word> small_ctx,
                   ConstContextPtr<word> lifted_ctx, const Config &cfg,
                   std::shared_ptr<const BootContext<word>> tower = nullptr);

  CiBatchAttention(const CiBatchAttention &) = delete;
  CiBatchAttention &operator=(const CiBatchAttention &) = delete;

  //! The chain-addressed batched layout every channel ciphertext must use.
  const CiBatchLayout &GetLayout() const { return layout_; }
  const CiSwitchedCcmmLayout &GetChain() const { return chain_; }
  int GetChainLevel() const { return cfg_.forward_level - 1; }
  //! The level a product's output lands at, in slots.
  int GetOutputLevel() const { return cfg_.inverse_level - 1; }

  /**
   * @brief Run every descend/return group through the old per-channel loop
   * (serial converter, serial ring switch) instead of the ct-batched path;
   * the A/B of `CiBatch.TheBatchedConverterIsWordForWord`. Initialised from
   * `CHEDDAR_CI_BATCH_CONV_SERIAL`.
   */
  static void SetConvSerial(bool serial);

  //! Whether the scores' return rides their bootstrap (`Config::
  //! fused_scores` with a tower ring).
  bool FusedScores() const { return cfg_.fused_scores; }
  //! Flip the fused return at runtime (the A/B of the fused-vs-serial
  //! diagnostic). Turning it on requires the tower basis to have been built
  //! at construction (`Config::fused_scores` true then).
  void SetFusedScores(bool on) {
    AssertTrue(!on || basis_ != nullptr,
               "CiBatchAttention: the tower basis was not built");
    cfg_.fused_scores = on;
  }
  //! The level the fused score boot lands at: the LAYER boot's own landing
  //! (the prefix runs one level above it; a tower whose EvalMod ends
  //! higher is LevelDowned to the prefix's entry first), so `SoftMax`
  //! reads either path the same.
  int GetFusedTopLevel() const { return GetTopLevel(); }
  //! The tower ring's rotations (the CtS' and the prefix); fused only.
  void AddTowerRotations(EvkRequest &req) const;
  /**
   * @brief The fused score bootstrap: groups of `group` SinC-form score
   * ciphertexts (what `Scores` hands back under `fused_scores`) through
   * `HalfBootTowerBatch` and the lane prefix, landing in slots at
   * `GetFusedTopLevel()` canonical with the chain's carried factor in the
   * message, exactly as a `Boot` would have left them. `sinc` is consumed.
   *
   * Under `Config::affine_in_prefix` the prefix also carries the softmax's
   * affine multiply, so the landed message is `u - a0 = 2 S / span` and
   * `SoftMax` starts at exp: pass the chain's `carried` (required then,
   * ignored otherwise) and call `PrepareSoftMax` first (the fold needs
   * `span`). The prefix is (re-)encoded at the first call and whenever
   * `carried` changes.
   */
  void BootScoresFused(std::vector<Ct> &booted, std::vector<Ct> &sinc,
                       const Keys &keys, int group,
                       double carried = 0.0) const;

  //! Rotations on the switching ring: the two converters.
  void AddSwitchRotations(EvkRequest &req) const;
  //! Rotations on the layer's ring: the key-token shift of the second
  //! call. Under the chain addressing token t + T/2 sits at block + 1, so
  //! the shift down is ONE slot rotation by `lanes` -- no second forward
  //! converter (3.8 GiB of plaintexts) for it.
  void AddBootRotations(EvkRequest &req) const;
  //! Under the PLAIN map the token is the SLOW axis instead, so the same
  //! shift is `T/2 * B` slots -- a bigger index, still ONE rotation, still
  //! lane-preserving (PcPremapTest).
  int GetShiftRotation() const {
    return layout_.lanes == 0
               ? (cfg_.num_tokens / 2) * layout_.num_instances
               : cfg_.sub_degree;
  }
  //! Automorphism indices on the lifted ring.
  std::vector<int> LiftedRotationIndices() const {
    return ccmm_.RotationIndices(2 * cfg_.sub_degree);
  }

  /**
   * @brief `res = RoPE(Q_h) RoPE(K_kv)^T` for one head: the 128 score
   * ciphertexts (ciphertext = key token, blocks = query tokens, every
   * instance), in slots at `GetOutputLevel()`, the chain's message factor
   * in the recorded scale.
   *
   * @param q the head's `head_dim` channel ciphertexts at `rope_level`,
   *        CONSUMED
   * @param k the kv head's `head_dim` channel ciphertexts at `rope_level`,
   *        read (RoPE'd per call into copies)
   */
  void Scores(std::vector<Ct> &res, std::vector<Ct> &q,
              const std::vector<Ct> &k, const Keys &keys) const;

  /**
   * @brief One kv head's K and V descended ONCE for both calls, shared by
   * its GQA group's four Q heads. `Scores`/`Values` on the raw channels
   * redo the kv head's RoPE-and-descent per Q head -- four times for the
   * same words. `lk[call][g][c]` / `lv[call][g][c]` is group `g`'s column
   * `c` on the lifted ring, exactly what the per-head path builds
   * (deterministic kernels on the same inputs, so bit-identical).
   */
  struct DescendedKV {
    std::vector<std::vector<std::vector<Ct>>> lk, lv;
  };
  /** @brief Fill `dkv` from one kv head's K and V, both calls. */
  void DescendKV(DescendedKV &dkv, const std::vector<Ct> &k,
                 const std::vector<Ct> &v, const Keys &keys) const;
  /** @brief `Scores` reading the hoisted K parts. */
  void Scores(std::vector<Ct> &res, std::vector<Ct> &q,
              const DescendedKV &dkv, const Keys &keys) const;
  /** @brief `Values` reading the hoisted V parts. */
  void Values(std::vector<Ct> &res, std::vector<Ct> &P,
              const DescendedKV &dkv, const Keys &keys) const;

  /**
   * @brief Device seconds of the four CC-MM phases since construction
   * (`EventSpanTimer` brackets, resolved here): the slot -> lifted descent,
   * Algorithm 4's product, the product's drop to the small ring, and the
   * way back to slots. The B512_ccmm_ideas step-0 attribution without a
   * profiler.
   */
  struct PhaseSeconds {
    double descend, multiply, lift_descend, ret;
    //! descend's split: the LevelDown/shift prologue, the batched forward
    //! conversion, the ring switch, the lifts (batched path only; the
    //! serial A/B leaves them zero).
    double desc_pre, desc_conv, desc_switch, desc_lift;
  };
  PhaseSeconds GetPhaseSeconds() const {
    return {t_descend_.Seconds(),     t_mult_.Seconds(),
            t_lift_descend_.Seconds(), t_return_.Seconds(),
            t_desc_pre_.Seconds(),    t_desc_conv_.Seconds(),
            t_desc_switch_.Seconds(), t_desc_lift_.Seconds()};
  }

  /** @brief What the softmax walk needs to know about the data, in CHAIN
   * units: the raw scores times the factor the Q and K weights carried
   * (`cq * ck`). */
  struct SoftMaxCalibration {
    double m_eff = 8.0;   //!< `span_raw / sqrt(D)`, the fitted exp's span
    double span = 1.0;    //!< the calibrated score span, chain units
    double shift = 0.0;   //!< the calibrated score max, chain units
    //! Causal only: [head][row] live-key maximum (chain units) and the
    //! live row-norm estimate `sum_l exp(m_eff (S - shift) / span)`; the
    //! latter folds into the mask as `est^-1/2` so the inverse square
    //! root's interval collapses to the actual / estimate ratio.
    std::vector<std::vector<double>> row_shift, row_norm;
    double norm_lo = 0.9, norm_hi = 1.1;  //!< the invsqrt interval (ratio)
    int exp_degree = 0;   //!< 0 = derive from `m_eff`
    //! 7 (three levels): with the affine's multiply folded into the causal
    //! mask, the walk is 12 levels above `forward_level` 4 -- exp's four,
    //! the mask, the square, three here and the closing two -- which is
    //! exactly what a Boot landing at 16 (ci16_35 as shipped) has. Degree 7
    //! on [0.9, 1.1] is 2^-13.
    int inv_degree = 7;
    bool causal = true;
    /**
     * @brief `niter > 0`: a per-ITERATION, per-HEAD, per-ROW estimate of
     * `sq = sum_l y_l^2`, which the invsqrt's argument is divided by.
     * `[iter][head][row]`; empty is 1 everywhere, i.e. today's behaviour.
     *
     * ## Why 4096 keys need it
     *
     * After the first normalise-and-square the row is a PROBABILITY VECTOR
     * (`sum_l (y_l r)^2 = 1` by construction of `r`), so every later `sq` is
     * that distribution's COLLISION PROBABILITY -- bounded below by
     * `1 / live` and above by how concentrated the row is. The window a
     * single affine has to cover therefore spans the rows' concentrations,
     * and its lower edge falls with the key count.
     *
     * Measured on the real layer-0 weights and a real 4096-token prompt
     * (`reference/scripts/cho_window.py`, 8 heads, niter 2):
     *
     *     later window   T = 128   [0.0131 .. 1.000]     deg-63  2^-19.4
     *                    T = 4096  [0.00051 .. 0.1732]   deg-63  2^-5.4
     *
     * -- the lower edge drops by exactly the key ratio (1/128 -> 1/4096) and
     * the fit collapses. It does NOT blow up, which is the dangerous part:
     * the polynomial saturates, `r` comes back at 0.354x its true value, and
     * the attention output is silently scaled down.
     *
     * Dividing `sq` by a per-row estimate collapses the window to the RATIO
     * actual/estimate, which is what `row_norm` already does for the
     * single-shot path. It is nearly free: the invsqrt's affine ALREADY
     * multiplies `sq` by `1 / aff_a`, so the estimate rides that constant at
     * no extra level, and only taking it back out of `r` costs one -- which
     * the walk has (deg-63 lands `r` at 8 against a floor of
     * `forward_level + 2`).
     *
     * And it buys accuracy back rather than spending it. On the same fit:
     *
     *     ratio window   [0.25, 4] = 16x   deg-31  6.1e-08   deg-63  2.0e-14
     *                    [0.5, 2]  =  4x   deg-31  4.8e-15
     *
     * so a fold that lands inside 16x lets `last_inv_degree` come DOWN from
     * 63 to 31 and gives a level back.
     */
    std::vector<std::vector<std::vector<double>>> cho_est;

    // --- Full Cho [25] normalize-and-square iteration (heterogeneous B=512) ---
    //! Extra squaring iterations `k`. 0 = the single-square shortcut above
    //! (one prompt, or the ride-factor proxy): exp -> sq -> one invsqrt(sq/est)
    //! -> P=(y r)^2, valid only when sq/est stays in [norm_lo,norm_hi].
    //!
    //! k>0 = the FAITHFUL Cho circuit for 512 DIFFERENT prompts sharing ONE
    //! population calibration. The shared `row_shift` is the POPULATION row-max,
    //! so a prompt whose true row-max sits far below it underflows the exp and
    //! sq collapses out of any fixed window; the single-square path then fails
    //! (whole layer 2^-0.3). Cho fixes it by DOWN-SCALING the exp argument by
    //! 2^k and doing k squarings:
    //!     x' = (S - shift)/2^k ;  y0 = exp(x') ;
    //!     y^j = ( y^{j-1} / || y^{j-1} || )^2 ,  j = 1..k ;  P = y^k .
    //! The 2^k down-scale compresses the FIRST normalization's cross-prompt
    //! window (host: k=4 3010x, k=5 617x, k=6 280x); the LATER normalizations
    //! see the data-INDEPENDENT [1/n,1] (~128x). Host-proven across 512 prompts
    //! (cho_softmax.py) to 2^-45 with exact ops. See [[quarot-heterogeneous-softmax]].
    //!
    //! Level cost: each squaring + its invsqrt-normalize descends the MAIN path
    //! (T ciphertexts), so ~k iterations need main-path bootstraps (Sylph's
    //! "wide main path" -- the aux boot already handles the narrow norm path).
    //! Intermediate invsqrts "need not be accurate" (a per-row scalar error is
    //! absorbed by the next normalize), so a LOW `iter_inv_degree` minimax over
    //! the ~128x [1/n,1] window suffices (host probe: deg-7 ~2^-2, fine as an
    //! intermediate scalar). Only the LAST normalization sets the final scale;
    //! it uses `last_inv_degree` (host probe: deg-31 ~2^-5.9 in 5 levels, cheaper
    //! than Newton's ~3 levels/step for the same accuracy) over [norm_lo,norm_hi].
    //! Both are the library's single-Chebyshev invsqrt idiom (as RmsNorm), on a
    //! landing ring that affords the degree; no Newton needed.
    int niter = 0;
    int iter_inv_degree = 7;    //!< intermediate normalizations (crude, ~128x)
    int last_inv_degree = 31;   //!< the final normalization (accurate, ~128x)
    //! The FIRST normalization's window (wider, k-dependent: the underflow
    //! spread not yet compressed). 0 = fall back to [norm_lo,norm_hi]. Later
    //! iterations use the data-independent [norm_lo,norm_hi] ~ [1/n,1].
    double first_lo = 0.0, first_hi = 0.0;
    /**
     * @brief The INTERMEDIATE normalizations' window (0 < j < k-1), and the
     * degree fitted on it. 0 = fall back to [norm_lo, norm_hi] and
     * `iter_inv_degree`, which is what k = 2 wants (there IS no intermediate)
     * and what k >= 3 must not have.
     *
     * ## Why an intermediate needs its own window
     *
     * `y_j` for j >= 1 is the tempered distribution NORMALIZED TO SUM ONE, so
     * `sq_j` is its collision probability and the LAST iteration -- always at
     * temperature 2, whatever k is -- spans `[1/live, 1]`: 2069x at T = 4096
     * against 350x at T = 128 (`reference/scripts/cho_ideal512.py`). The
     * intermediates are at temperature 4, 8, 16 ... and span 25x, 5x, 18x.
     * Compiling them on the LAST one's window is therefore a hole that only
     * opens at k >= 3, and it opened: with `iter_inv_degree` 31 over a 5900x
     * window the intermediate `r` is ~38% out, `y = (y r)^2` carries that to
     * the fourth power, and `sq_{k-1}` leaves the window the last invsqrt was
     * fitted on -- whose unclamped Chebyshev then grows like cosh. Measured on
     * a T = 4096 population calibration: softmax 2^-3.7, and with the served
     * context's own (tighter) last window 2.3e+49.
     *
     * It cannot be fixed by raising `iter_inv_degree`, because that degree is
     * also the FIRST invsqrt's and the first is compiled at `exp_out_ - 3`
     * with five levels to spend ("overspends its levels" at 63).
     */
    double mid_lo = 0.0, mid_hi = 0.0;
    int mid_inv_degree = 0;  //!< 0 = `iter_inv_degree`
    /**
     * @brief `[iter][head][row]`: a per-row estimate of the public half's
     * running scalar `R_{j+1}`, so the ciphertext that is BOOTSTRAPPED carries
     * `R / est` instead of `R`. Empty is 1 everywhere -- today's behaviour.
     *
     * ## Why `R` is not O(1)
     *
     * `SoftMaxCho` carries `R_j` because the public keys leave the sum: with
     * `y^(j)_l = (y0_l)^(2^j) R_j`, `R_j` has no key index, so the public half
     * owes only power sums. But `sum_l y^(j)_l = 1`, so
     *
     *     R_j  =  1 / sum_l (y0_l)^(2^j)
     *
     * -- it IS the softmax's normaliser, and that spans orders of magnitude
     * with the data. At T = 128 with a per-prompt calibration it happens to sit
     * near 1 (measured 0.974) and the bootstrap at the end of each iteration is
     * well posed. At T = 4096 with a POPULATION calibration the same quantity
     * reaches 42 (host), the boot's message leaves EvalMod's range, and the
     * result is 2.2e+49 -- silently, because a bootstrap out of range does not
     * raise.
     *
     * With an estimate the boot sees `R / est ~ 1`. It costs no level of its
     * own: the factor rides `r^2`, which is already formed for the recursion
     * and is followed immediately by the boot, and the estimate comes back out
     * where `R` is CONSUMED -- `est^2` folds into the per-row constant the
     * public power sum already carries. The one thing the caller must do is
     * take `est` back out of `pub_scale`, which comes back as
     * `R_k / pub_r_est[niter-1][head][row]`; fold it into the public value
     * accumulator, which carries a per-row constant of its own.
     *
     * The estimate itself is offline: `R_j = 1 / sum_l (y0_l)^(2^j)` is a
     * function of the scores, so a calibration split gives it
     * (reference/scripts/gen_t4096_pop.py).
     */
    std::vector<std::vector<std::vector<double>>> pub_r_est;
  };

  /**
   * @brief Compile the softmax walk: the two polynomials and the per-head
   * per-token plaintexts of the causal mask and the row shift. Cheap.
   */
  void PrepareSoftMax(const SoftMaxCalibration &calib);
  //! How many exp evaluations `SoftMax` batches (<= 1 = the per-ciphertext
  //! loop, the default). `CHEDDAR_BATCH_SOFTMAX_EXP` overrides; a setter so
  //! one object can run both routes for a word-for-word comparison.
  void SetExpBatch(int n) { exp_batch_ = n; }
  int GetExpBatch() const { return exp_batch_; }
  //! Where `SoftMax` expects its booted scores: the boot's landing.
  int GetTopLevel() const {
    return cfg_.score_top > 0 ? cfg_.score_top
                              : boot_->GetBootParameter().GetEndLevel();
  }

  /**
   * @brief [3] Give the softmax's Euclidean-norm accumulator its own
   * bootstrap ring (a gen_landing sub-ladder on the layer's secret and
   * bottom primes -- the norm channel ring, landing 9, serves). Call
   * BEFORE `PrepareSoftMax`: the inverse square root is then compiled at
   * `aux`'s landing (9 -> deg 7 lands at 6, P at `forward_level` 4).
   * Causal calibrations only (the affine shift is level-free there). The
   * caller keeps `aux`'s EvalMod and FFT tables prepared around `SoftMax`.
   */
  void SetAuxBoot(std::shared_ptr<const BootContext<word>> aux,
                  const EvkMap<word> *aux_evk) {
    aux_boot_ = std::move(aux);
    aux_evk_ = aux_evk;
  }

  /**
   * @brief One head's softmax on the batched layout: the key axis is the
   * ciphertext index, so the row sums are sums of ciphertexts.
   *
   *     u = a1 S + a0[row]  ->  y = exp(m_eff (u - 1) / 4) (.) mask[l]
   *     sq = sum_l y_l^2 (one relinearization)  ->  r = invsqrt(sq)
   *     P_l = (y_l r)^2                                  (Cho, k = 1)
   *
   * @param P the 128 key-token ciphertexts of P at `forward_level`
   * @param scores the booted scores at `GetTopLevel()`, read
   * @param head the head, for its row shift and norm estimate
   * @param carried the scores' recorded-over-canonical factor before their
   *        Boot, divided out in the affine map
   */
  /**
   * @brief The PUBLIC half of a T = 4096 head, exactly as `CiPcAttention`
   * leaves it. Absent (a null pointer) is the T = 128 layer, and then not one
   * instruction below changes -- the whole join is inside
   * `if (pub != nullptr)`.
   *
   * Unrolling `SoftMaxCho`'s walk gives
   * `y^(j)_l = (y0_l)^(2^j) R_j` with `R_j = prod_{i<j} (r^(i))^(2^(j-i))`,
   * and `R_j` carries no key index -- it is one number a QUERY token. So it
   * comes out of the denominator,
   *
   *     sq^(j) = sum_l (y^(j)_l)^2 = R_j^2 sum_l (y0_l)^(2^(j+1)) ,
   *
   * and the public keys never have to survive an iteration. What they hand
   * over instead is `pow[j] = sum_p (y0_p)^(2^(j+1))`, one a iteration, and
   * one value accumulator at the top power. `CiPcAttention::Head` returns
   * exactly those.
   */
  struct PublicHalf {
    //! `pow[j] = sum_p (y0_p)^(2^(j+1))` DIVIDED by `pow_scale[j]`, `niter`
    //! of them.
    const std::vector<Ct> *pow = nullptr;
    /**
     * @brief The per-ITERATION, per-ROW constant the caller divided `pow[j]`
     * by. `[iter][row]`; required whenever `pow` is given.
     *
     * The public term has to be bootstrapped -- it arrives from the public
     * branch far below the walk -- and a bootstrap wants an O(1) message
     * (`GetMessageRatio`). The raw power sums are not: over 3968 keys
     * `sum_p y0^2` runs to the hundreds, which puts EvalMod's sine outside
     * its range and returns garbage rather than an error. So the caller
     * hands over the sum already divided by a calibrated estimate of its own
     * size, and this class multiplies the estimate back in BEFORE the boot,
     * together with `1 / aff_a` and the row's `cho_est` -- one plaintext,
     * and the thing that crosses the bootstrap is the public half of the
     * invsqrt's ARGUMENT, which is O(1) by construction.
     */
    std::vector<std::vector<double>> pow_scale;
  };

  /**
   * @param pub the public half of a T = 4096 head, or null for T = 128
   * @param pub_scale out: `R_niter`, the per-query-token factor the public
   *        VALUE accumulator has to be multiplied by before it joins the
   *        output of `Values`. Written only when `pub` is given.
   */
  void SoftMax(std::vector<Ct> &P, const std::vector<Ct> &scores, int head,
               double carried, const EvkMap<word> &evk,
               const PublicHalf *pub = nullptr,
               Ct *pub_scale = nullptr) const;

  /**
   * @brief The FULL Cho [25] iteration (SoftMaxCalibration::niter > 0) for 512
   * DIFFERENT prompts sharing one population calibration. Downscales the exp
   * argument by 2^k and runs k squarings, re-bootstrapping the MAIN path (the
   * T ciphertexts) between iterations:
   *     y0 = exp((S - shift)/2^k) (.) causal ;
   *     for j = 0..k-1: sq = sum_l y_l^2 ; r = invsqrt_j(sq) ; y = (y r)^2 [; boot y]
   *     P = y_k .
   * Booting the main path each iteration keeps sq at a high level, so no aux
   * boot is needed. invsqrt_0 uses the wide first window; the rest the data-
   * independent [norm_lo,norm_hi]~[1/n,1], crude (iter_inv_degree) except the
   * last (last_inv_degree). Plain causal path only (no fused/affine-prefix).
   */
  void SoftMaxCho(std::vector<Ct> &P, const std::vector<Ct> &scores, int head,
                  double carried, const EvkMap<word> &evk,
                  const PublicHalf *pub = nullptr,
                  Ct *pub_scale = nullptr) const;

  /**
   * @brief `res = P V` for one head: the 128 attention-output channel
   * ciphertexts (blocks = query tokens) in slots at `GetOutputLevel()`,
   * the chain's factor in the recorded scale.
   *
   * @param P the head's 128 key-token ciphertexts at `forward_level`,
   *        CONSUMED
   * @param v the kv head's 128 channel ciphertexts at `rope_level`, read
   */
  void Values(std::vector<Ct> &res, std::vector<Ct> &P,
              const std::vector<Ct> &v, const Keys &keys) const;

 private:
  //! V's per-call plaintext: the call's key tokens kept, at `rope_level`.
  Pt call_mask_[2];
  //! The compiled softmax walk.
  SoftMaxCalibration calib_;
  bool softmax_ready_ = false;
  //! How many of the softmax's `num_tokens` exp evaluations go through
  //! `EvalPoly::EvaluateBatch` at once. <= 1 is the per-ciphertext loop, and
  //! that is the DEFAULT because the batch measured flat -- see `SoftMax`.
  //! `CHEDDAR_BATCH_SOFTMAX_EXP` overrides.
  int exp_batch_ = 1;
  int exp_in_ = 0, exp_out_ = 0, mask_level_ = 0, sq_level_ = 0,
      poly_in_ = 0;
  // [1] is recompiled lazily on the aux path (its ladder's EvalMod can
  // land ~0.3% off the nominal scale, the 7.37 drift), so mutable.
  mutable std::vector<std::unique_ptr<EvalPoly<word>>> polys_;  // [0] exp, [1] invsqrt
  //! [3] the invsqrt's fit and landing, kept so SoftMax can recompile [1]
  //! at the MEASURED aux landing scale; in_scale 0 = not yet compiled.
  std::vector<double> aux_inv_coeffs_;
  int aux_inv_out_ = -1;
  mutable double aux_inv_in_scale_ = 0.0;
  //! Per head: the row shift's per-token plaintext at `exp_in_`, and the
  //! 128 per-key-token causal masks (with `est^-1/2` folded) at `exp_out_`.
  //! The masks are built per head at its call (`BuildMasks`), 128
  //! plaintexts at a time.
  std::vector<Pt> a0_;
  //! niter>0 (SoftMaxCho): the k+... invsqrt polynomials -- [0] the wide FIRST
  //! window, [1] the crude intermediate (iter_inv_degree), [2] the accurate
  //! LAST (last_inv_degree); both later ones over [norm_lo,norm_hi]. Plus their
  //! affine (fit domain -> window) and the exp's downscaled landing.
  mutable std::vector<std::unique_ptr<EvalPoly<word>>> cho_inv_;  // 0 first,1 mid,2 last
  double cho_first_lo_ = 0.0, cho_first_hi_ = 0.0;  //!< the first window
  //! The LATER window (iterations >0), starting from calib [norm_lo,norm_hi] but
  //! with its upper end WIDENED by construction to cover the crude first
  //! invsqrt's data-independent overshoot (WorstCaseChoLaterSq) so the unclamped
  //! last invsqrt cannot blow for any prompt. Used by both PrepareSoftMax (the
  //! compiled poly) and SoftMaxCho (the runtime affine) -- they must agree.
  double cho_later_lo_ = 0.0, cho_later_hi_ = 0.0;
  //! The INTERMEDIATE window (0 < j < k-1) and its degree. Separate from the
  //! later one because the intermediates are at temperature 4, 8, 16 ... and
  //! span tens, while the last is always at temperature 2 and spans `live`.
  //! Its upper end carries the FIRST invsqrt's overshoot, and the LAST one's
  //! carries this one's -- the chain, not one step. See
  //! `SoftMaxCalibration::mid_lo`.
  double cho_mid_lo_ = 0.0, cho_mid_hi_ = 0.0;
  int cho_mid_deg_ = 0;
  int cho_inv_in_ = 0;   //!< the level the later invsqrts read sq at (booted)
  //! The FIRST iteration skips the main-path boot (y0 is fresh from exp), so its
  //! invsqrt reads sq at a LOWER level -- compiled separately here.
  int cho_first_in_ = 0;
  //! niter>0 (SoftMaxCho): the 128 plain-causal 0/1 masks at exp_out_. They are
  //! HEAD-INDEPENDENT (the Cho iteration does the normalization -- no est/gamma
  //! fold, unlike BuildMasks), so they are encoded ONCE in PrepareSoftMax rather
  //! than rebuilt per head call (128 x NHEAD -> 128 encodes a layer). mask[l] is
  //! live (1) at query tokens t >= l.
  std::vector<Pt> cho_masks_;
  //! `affine_in_prefix`: the `carried` the prefix plaintexts were encoded
  //! with (0 = the plain ctor encode, no affine folded). The chain's scale
  //! walk is deterministic, so after the first fold this never changes.
  mutable double prefix_affine_carried_ = 0.0;
  //! The MEASURED tower-boot output scale the prefix plaintexts were
  //! encoded from (0 = the ctor's nominal-StCInputScale encode, which can
  //! be ~0.3% off the true EvalMod-tree scale and then misses the
  //! canonical landing the softmax's first Add requires).
  mutable double prefix_in_scale_ = 0.0;
  void BuildMasks(std::vector<Pt> &masks, int head) const;
  //! Zero ciphertexts on the lifted ring, the shape of `like`, `count` of
  //! them: the contract's dead lhs columns.
  void ZeroLifted(std::vector<Ct> &res, const Ct &like, int count) const;
  void BuildRope();
  //! RoPE in place on one head's `head_dim` channel ciphertexts, with the
  //! key-token half `call` kept (-1: every token), `rope_level` -> one below.
  void Rope(std::vector<Ct> &cts, int call) const;
  //! One channel ciphertext down to the lifted ring: LevelDown, for call 1
  //! the key-token shift (one rotation), the forward converter, the ring
  //! switch, the lift. `lifted[g]` is group `g`'s part. `ct` is consumed.
  void Descend(std::vector<Ct> &lifted, Ct &ct, int call,
               const Keys &keys) const;
  //! `Descend` over a GROUP of channels: the LevelDowns (and the odd call's
  //! shifts) per channel, then ONE ct-batched forward conversion
  //! (`CiSinCConverter::SlotToSinCBatch` -- the B512_ccmm_ideas idea [2]:
  //! the diagonal table streamed once for the group instead of once per
  //! channel), then the ring switch and lift per channel. `lifted[c][g]` is
  //! channel c's group-g part; `cts` are consumed. Word for word the loop
  //! of `Descend` calls; `CHEDDAR_CI_BATCH_CONV_SERIAL=1` is that loop.
  void DescendBatch(std::vector<std::vector<Ct>> &lifted,
                    std::vector<Ct> &cts, int call, const Keys &keys) const;
  //! The way back for one column: the `rank` groups' parts (product ring)
  //! switched back into one big ciphertext, then SinC -> slots.
  void Return(Ct &res, const std::vector<Ct> &parts, const Keys &keys) const;
  //! `Return` over a GROUP of columns: the ring switch-backs per column,
  //! then ONE ct-batched inverse conversion. `*res[i]` answers
  //! `parts_list[i]` (consumed). Word for word the loop of `Return` calls.
  //! `to_slots` false stops after the switch-back (the fused scores' SinC
  //! form; no inverse conversion).
  void ReturnBatch(const std::vector<Ct *> &res,
                   std::vector<std::vector<Ct>> &parts_list, const Keys &keys,
                   bool to_slots = true) const;

  std::shared_ptr<const BootContext<word>> boot_;
  ConstContextPtr<word> switch_ctx_;
  ConstContextPtr<word> small_ctx_;
  ConstContextPtr<word> lifted_ctx_;
  Config cfg_;
  CiSwitchedCcmmLayout chain_;
  CiBatchLayout layout_;
  RingSwitchHandler<word> switcher_;
  CiLiftHandler<word> lift_;
  BatchCcmmHandler<word> ccmm_;
  //! `plain_map` only: the block permutation both converters fold, built
  //! from the two layouts rather than transcribed from them. Held because
  //! the converters take a pointer to it.
  std::vector<int> premap_;
  //! The forward (slots -> SinC) and the inverse converter.
  std::unique_ptr<CiSinCConverter<word>> fwd_;
  std::unique_ptr<CiSinCConverter<word>> inv_;
  //! The fused scores' tower ring and its basis (CtS' + prefix).
  std::shared_ptr<const BootContext<word>> tower_;
  //! [3] the aux accumulator's own bootstrap ring (`SetAuxBoot`), or null.
  std::shared_ptr<const BootContext<word>> aux_boot_;
  const EvkMap<word> *aux_evk_ = nullptr;
  //! [3] the accumulator's ride factor: the masks carry sqrt of it, the
  //! booted sum re-declares its scale by it, the inverse square root
  //! returns it. 1 without the aux boot.
  double aux_gamma_ = 1.0;
  std::unique_ptr<CiSinCBasis<word>> basis_;

 public:
  /**
   * @brief Residency of the two converters' compiled plaintexts (~7.5 GiB;
   * `HoistHandler::Unstage`). The feed-forward half never touches them and
   * the layer's seam is memory-bound: the layer unstages them for the FFN
   * and stages them back for the next attention.
   */
  void UnstageConverters() const {
    if (fwd_) {
      if (fwd_->GetForward() != nullptr) fwd_->GetForward()->Unstage();
      if (fwd_->GetInverse() != nullptr) fwd_->GetInverse()->Unstage();
    }
    if (inv_) {
      if (inv_->GetForward() != nullptr) inv_->GetForward()->Unstage();
      if (inv_->GetInverse() != nullptr) inv_->GetInverse()->Unstage();
    }
  }
  void StageConverters() const {
    if (fwd_) {
      if (fwd_->GetForward() != nullptr) fwd_->GetForward()->Stage();
      if (fwd_->GetInverse() != nullptr) fwd_->GetInverse()->Stage();
    }
    if (inv_) {
      if (inv_->GetForward() != nullptr) inv_->GetForward()->Stage();
      if (inv_->GetInverse() != nullptr) inv_->GetInverse()->Stage();
    }
  }

 private:
  //! RoPE's per-token plaintexts at `rope_level`: [mask][pair], mask 0 =
  //! every token (Q), 1 / 2 = the key-token halves (K's two calls).
  std::vector<Pt> rope_cos_[3], rope_sin_[3];
  //! The phase ledger (`GetPhaseSeconds`).
  mutable EventSpanTimer t_descend_, t_mult_, t_lift_descend_, t_return_;
  mutable EventSpanTimer t_desc_pre_, t_desc_conv_, t_desc_switch_,
      t_desc_lift_;
  static bool conv_serial_;
};

}  // namespace cheddar
