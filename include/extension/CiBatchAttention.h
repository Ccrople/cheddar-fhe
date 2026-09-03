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
  int GetShiftRotation() const { return cfg_.sub_degree; }
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
  };

  /**
   * @brief Compile the softmax walk: the two polynomials and the per-head
   * per-token plaintexts of the causal mask and the row shift. Cheap.
   */
  void PrepareSoftMax(const SoftMaxCalibration &calib);
  //! Where `SoftMax` expects its booted scores: the boot's landing.
  int GetTopLevel() const { return boot_->GetBootParameter().GetEndLevel(); }

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
  void SoftMax(std::vector<Ct> &P, const std::vector<Ct> &scores, int head,
               double carried, const EvkMap<word> &evk) const;

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
  int exp_in_ = 0, exp_out_ = 0, mask_level_ = 0, sq_level_ = 0,
      poly_in_ = 0;
  std::vector<std::unique_ptr<EvalPoly<word>>> polys_;  // [0] exp, [1] invsqrt
  //! Per head: the row shift's per-token plaintext at `exp_in_`, and the
  //! 128 per-key-token causal masks (with `est^-1/2` folded) at `exp_out_`.
  //! The masks are built per head at its call (`BuildMasks`), 128
  //! plaintexts at a time.
  std::vector<Pt> a0_;
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
  //! The forward (slots -> SinC) and the inverse converter.
  std::unique_ptr<CiSinCConverter<word>> fwd_;
  std::unique_ptr<CiSinCConverter<word>> inv_;
  //! The fused scores' tower ring and its basis (CtS' + prefix).
  std::shared_ptr<const BootContext<word>> tower_;
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
