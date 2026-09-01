#pragma once

#include <memory>
#include <string>
#include <vector>

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
 * ## Levels
 *
 *     projection output @ rope_level (6)  --RoPE(+mask)-->  5
 *       --LevelDown--> forward_level (4)  --SlotToSinC-->  3 = chain level
 *       --RingSwitch--> parts @3  --Lift-->  Algorithm 4 @3 -> 2
 *       --Descend, SwitchBack @2--> --SinCToSlot @ inverse_level (2)--> 1
 *
 * so a product's output lands in slots at level 1: the scores go to a
 * `Boot` (which takes any level), the attention output to the O projection
 * (1 -> 0) and the residual at 0.
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
    //! The projections' output level; RoPE lands one below.
    int rope_level = 6;
    //! The slots -> SinC forward runs here and lands one below, where the
    //! chain runs.
    int forward_level = 4;
    //! The SinC -> slots return runs here and lands one below.
    int inverse_level = 2;
    double rope_base = 500000.0;  //!< Llama-3's theta
    int converter_baby_steps = 256;
    bool verbose = false;
  };

  /** @brief The keys one call needs, on three rings. */
  struct Keys {
    const EvkMap<word> *swtch = nullptr;   //!< the switching ring
    const EvkMap<word> *lifted = nullptr;  //!< the lifted ordinary ring
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
                   ConstContextPtr<word> lifted_ctx, const Config &cfg);

  CiBatchAttention(const CiBatchAttention &) = delete;
  CiBatchAttention &operator=(const CiBatchAttention &) = delete;

  //! The chain-addressed batched layout every channel ciphertext must use.
  const CiBatchLayout &GetLayout() const { return layout_; }
  const CiSwitchedCcmmLayout &GetChain() const { return chain_; }
  int GetChainLevel() const { return cfg_.forward_level - 1; }
  //! The level a product's output lands at, in slots.
  int GetOutputLevel() const { return cfg_.inverse_level - 1; }

  //! Rotations on the switching ring: the three converters.
  void AddSwitchRotations(EvkRequest &req) const;
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

 private:
  void BuildRope();
  //! RoPE in place on one head's `head_dim` channel ciphertexts, with the
  //! key-token half `call` kept (-1: every token), `rope_level` -> one below.
  void Rope(std::vector<Ct> &cts, int call) const;
  //! One channel ciphertext down to the lifted ring: LevelDown, the forward
  //! converter of `call` (-1 or 0: plain; 1: the shifted one), the ring
  //! switch, the lift. `lifted[g]` is group `g`'s part. `ct` is consumed.
  void Descend(std::vector<Ct> &lifted, Ct &ct, int call,
               const Keys &keys) const;
  //! The way back for one column: the `rank` groups' parts (product ring)
  //! switched back into one big ciphertext, then SinC -> slots.
  void Return(Ct &res, const std::vector<Ct> &parts, const Keys &keys) const;

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
  //! [0]: the plain forward; [1]: the forward with the key-token shift
  //! folded in; the inverse.
  std::unique_ptr<CiSinCConverter<word>> fwd_[2];
  std::unique_ptr<CiSinCConverter<word>> inv_;
  //! RoPE's per-token plaintexts at `rope_level`: [mask][pair], mask 0 =
  //! every token (Q), 1 / 2 = the key-token halves (K's two calls).
  std::vector<Pt> rope_cos_[3], rope_sin_[3];
};

}  // namespace cheddar
