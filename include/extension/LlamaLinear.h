#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#include "extension/LlamaBlock.h"

namespace cheddar {

/**
 * @brief `LlamaBlock::LinearLeg::Project` for real: the Bae PC-MM on the
 * block's own ciphertexts, at the block's own ring degree, with **no ring
 * switch**.
 *
 * ## Why there is no ring switch here
 *
 * Doing.md 1.5y recorded the products as blocked on a ring-switching parameter
 * pair, and its own closing sentence says otherwise -- "ring switching is an
 * optimisation of the linear leg, not a correctness requirement". Two separate
 * findings turn that from a preference into the only route:
 *
 * 1. `bootparam_35` admits no *buildable* switching pair at any level.
 *    `Parameter.cu:86-91` requires the top `level_config` entry to hold every
 *    main and terminal prime, so a truncated ladder is constructible only if
 *    its top level dominates every lower one componentwise. `bootparam_35`'s
 *    bottom ladder regrafts -- (0,2) -> (2,1) -> (4,0) -> (1,5) -- so every
 *    window tops out at level 31, about 1300 bits over the degree-4096 cap.
 *    The "fifteen bits of headroom" in 1.5y describes a set that cannot be
 *    instantiated.
 *
 * 2. Nothing needs it. What the PC-MM requires is one channel per ciphertext,
 *    because `res[i] = sum_j u[i][j] * cts[j]` is a *scalar* combination: it
 *    never mixes coefficients within a ciphertext, so output coefficient p
 *    depends only on input coefficient p. `ModDecomp` supplies exactly that
 *    and is free -- no key, no security spent, since rank k over degree N' is
 *    RLWE over kN' (`Mlwe.h:26`). ModPack's switching keys bind at the *full*
 *    degree, not a subring, so no ring-switch key is involved either.
 *
 * ## The alignment, which is what makes this cheap
 *
 * `ModDecomp` sends coefficient `i + rank*s` to position `s` of MLWE
 * ciphertext `i` (`Mlwe.cu:21`), so it groups by residue mod rank: it isolates
 * whichever axis owns the **low coefficient bits**. The block's slot layout is
 * `slot s -> token s % T, channel s / T`, and `CoeffOfSlot` is
 * `BitReverse(s, log2 slots)` (`AttentionPacking.cpp:102-110`), so low slot
 * bits become high coefficient bits and the **channel** ends up low. It is
 * already the axis the product wants; no repacking, no rotation, no mask.
 *
 * The alignment is exact only at `small_degree = 2 * num_tokens`, and only when
 * the block's channels-per-ciphertext equals the resulting rank. At degree
 * 65536 that means **T = 128 and rank 256**: T = 64 puts two channels in each
 * module component at the only legal rank, because `Mlwe.cu:158` requires
 * `small_degree % 256 == 0`. The constructor checks this rather than assuming
 * it.
 *
 * The MLWE index carries the channel bit-reversed, and so does the output row
 * index. Both are fixed permutations folded into the plaintext operand, where
 * they cost nothing.
 *
 * ## What it costs
 *
 * `ModDecomp` to rank k multiplies the a-part by k: one parent's
 * `num_primes * degree` words become k MLWE ciphertexts of that size each.
 * At 4096 input channels and level 1 that is about 3.2 GB, and it is precisely
 * what [SYLPH]'s ring switch to degree 4096 removes (k = 16, not 256). The
 * alternative to `ModDecomp` -- moving the channel axis onto the ciphertext
 * axis by Galois automorphisms -- costs 511 key switches per parent and has
 * never been built.
 *
 * ## Scale, and the one answer
 *
 * The product does not rescale (following `Context::Mult`), so the caller's
 * weight scale decides where the rescaled result lands. It is canonical at
 * level 0 for exactly one choice:
 *
 *     GetScale(L) * w / GetRescalePrimeProd(L) = GetScale(L-1)  =>  w = GetScale(L)
 *
 * `EncodeWeights` uses that and nothing else. A merely plausible scale
 * survives every plain operation silently and stops being silent inside
 * `EvalPoly`, which aborts the process -- three layers above here.
 *
 * ## What this class does NOT implement
 *
 * `Scores` and `Values` are ciphertext-ciphertext products and need
 * `BatchCcmmHandler`, not this one. They are deliberately left pure virtual,
 * so a caller must say what it is using for them; there is no silent fallback.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CoeffLinearLeg : public LlamaBlock<word>::LinearLeg {
 private:
  using Ct = Ciphertext<word>;

 public:
  struct Config {
    int num_tokens = 128;   //!< T; must satisfy small_degree == 2 * T
    int product_level = 1;  //!< the product runs here and rescales to level - 1
  };

  /**
   * @brief Build the leg.
   *
   * @param context the CKKS context the block runs in
   * @param cfg see Config
   * @param modpack_keys exactly `rank` keys, `keys[j]` switching the j-th
   *        embedded module component to the ordinary secret. Generate with
   *        `UserInterface::PrepareModPackKeys(small_degree, product_level)`
   *        and collect with `GetModPackKey(rank, j)`. The leg does not own
   *        them and they must outlive it.
   */
  CoeffLinearLeg(ConstContextPtr<word> context, const Config &cfg,
                 std::vector<const EvaluationKey<word> *> modpack_keys);

  // disable copying (or moving also)
  CoeffLinearLeg(const CoeffLinearLeg &) = delete;
  CoeffLinearLeg &operator=(const CoeffLinearLeg &) = delete;

  /// k = degree / small_degree, and also the block's channels per ciphertext.
  int GetRank() const { return rank_; }
  /// N' = 2 * num_tokens.
  int GetSmallDegree() const { return small_degree_; }
  /// The ModPack keys a caller has to generate, for `PrepareModPackKeys`.
  static int SmallDegreeFor(int num_tokens) { return 2 * num_tokens; }

  void Project(std::vector<Ct> &res, const std::vector<Ct> &x, int in_channels,
               int out_channels, const std::vector<double> &w, double w_scale,
               const char *name) const override;

 private:
  // Row r of the operand carries output channel `group * rank + BitRev(r)`,
  // and column `p * rank + i` carries input channel `p * rank + BitRev(i)`.
  // Both reversals are their own inverse.
  void EncodeWeights(PlainMatrix<word> &res, const std::vector<double> &w,
                     int in_channels, int out_channels, int group,
                     double w_scale) const;

  ConstContextPtr<word> context_;
  Config cfg_;
  int small_degree_;
  int rank_;
  std::vector<const EvaluationKey<word> *> modpack_keys_;
  MlweHandler<word> mlwe_;
  PcmmHandler<word> pcmm_;
};

}  // namespace cheddar
