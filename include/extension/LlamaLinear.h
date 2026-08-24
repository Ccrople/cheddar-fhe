#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#include "core/RingSwitch.h"
#ifdef USE_CUBLAS
#include "core/PcmmBlas.h"
#endif
#include "extension/LlamaBlock.h"

namespace cheddar {

/**
 * @brief `LlamaBlock::LinearLeg::Project` for real: the Bae PC-MM on the
 * block's own ciphertexts, by either of the two routes down to degree 256.
 *
 * ## The two descents, and why the ring-switched one is [SYLPH]'s
 *
 * The PC-MM wants one channel per ciphertext, because `res[i] = sum_j u[i][j]
 * * cts[j]` is a *scalar* combination: it never mixes coefficients within a
 * ciphertext, so output coefficient p depends only on input coefficient p.
 * There are two ways to get there and they differ by an order of magnitude in
 * what the return trip costs.
 *
 * **Direct (`Descent` absent).** `ModDecomp` straight from the block's degree,
 * rank `k = 65536 / 256 = 256`. It is free going down -- no key, no security
 * spent, since rank k over degree N' is RLWE over kN' (`Mlwe.h:26`) -- and the
 * bill arrives on the way back: ModPack is **256 key switches at degree
 * 65536** per output group, and the decomposition materialises `k` times the
 * parent's a-part, about 201 MB per parent at level 1. That is what
 * `parents_per_tile` exists to bound.
 *
 * **Ring-switched (`Descent` present).** [SYLPH] section 3.2 reaches degree
 * 256 by "ring-switching followed by an MLWE decomposition", and `Mlwe.h`
 * names the same route: a RingSwitch from N down to N1 first reduces k from
 * N/N' to N1/N'. Concretely
 *
 *     1 x 65536  --RingSwitch-->  16 x 4096  --ModDecomp-->  16 x 16 MLWE@256
 *
 * so the 256 module components are the same 256 components -- only their
 * *index* changes, and `Component()` below is that change. The return trip is
 * then 16 ModPacks of 16 switches **at degree 4096**, which is a sixteenth of
 * a big one each, plus one inverse ring switch at 65536. Against 256 big
 * switches that is about 15x, and the decomposition is 16x smaller, which
 * removes the reason tiling existed.
 *
 * The one thing it costs is the switching key, published at PQ and readable at
 * degree N1, so it needs a parameter set whose alpha is 1 -- `ringswitch16_35`
 * and `ringdegree12_35`, the same pair the batch CC-MM already descends
 * through. Doing.md 1.5ac and 1.5ae concluded no such pair was buildable on
 * `bootparam_35` and this class was written against that; 1.5af superseded it.
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
 * ## What it costs, and what the descent changes about that
 *
 * `ModDecomp` to rank k multiplies the a-part by k: one parent's
 * `num_primes * degree` words become k MLWE ciphertexts of that size each. The
 * alternative -- moving the channel axis onto the ciphertext axis by Galois
 * automorphisms -- costs 511 key switches per parent and has never been built.
 *
 * Direct, at 4096 input channels and level 1, that is about 3.2 GB, and it is
 * why the contraction is **tiled**: Llama's down projection contracts over
 * 14336 channels, which is 56 parents and about 11.3 GB of module components
 * held at once, more than a 48 GiB card has free in the middle of a block that
 * is already holding a dozen tensors. `Config::parents_per_tile` caps how many
 * parents are decomposed at a time; each tile's partial products are ModPacked
 * and **added as ordinary RLWE ciphertexts** before a single rescale at the
 * end. Summing after ModPack rather than before it is not a preference: there
 * is no MLWE addition in the library, and a partial sum that has already been
 * repacked is an ordinary ciphertext at the product's level.
 *
 * The cost of a tile is one extra ModPack per output group, which is `rank`
 * key switches -- so the tiling trades roughly 46 ms per group per extra tile
 * against gigabytes. It does not change the result: the sum is exact, and the
 * single rescale at the end is the same rescale the untiled path performs.
 *
 * With the `Descent` the same 56 parents hold about 705 MB rather than 11.3 GB,
 * because the rank is 16 and not 256, so the tiling has nothing left to buy and
 * `parents_per_tile = 0` is the setting. The accumulation moves to the small
 * ring, where an add is a sixteenth of the size, and the inverse ring switch
 * runs **once per output group** rather than once per tile -- so a tile costs
 * one extra ModPack at degree 4096 and nothing else.
 *
 * ## Scale, and the one answer
 *
 * The product does not rescale (following `Context::Mult`), so the caller's
 * weight scale decides where the rescaled result lands. It is canonical at
 * level 0 for exactly one choice:
 *
 *     GetScale(L) * w / GetRescalePrimeProd(L) = GetScale(L-1)  =>  w = GetScale(L)
 *
 * `BuildOperands` uses that and nothing else. A merely plausible scale
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
    //! Input ciphertexts decomposed at once. Each one costs `rank` module
    //! components of the parent's own size -- about 201 MB at level 1 on
    //! `bootparam_35` -- so 16 is roughly 3.2 GB. 0 means no tiling.
    int parents_per_tile = 16;
  };

  /**
   * @brief [SYLPH] section 3.2's descent: ring-switch first, decompose after.
   *
   * Leaving this empty is the direct route and changes nothing. Filling it in
   * puts the product on the small ring, which is where the paper runs it; see
   * the class comment for what the two cost.
   *
   * The two contexts are the same pair the batch CC-MM uses and are shared
   * with it when both are on. `switch_context` holds the **block's own
   * secret** -- a ciphertext walks down the ladder without a word changing --
   * and exists only because its `alpha` is small enough that a switching key
   * published at PQ still fits the small ring's budget. `small_context` has
   * its own secret and its own ModPack keys.
   */
  struct Descent {
    //! Degree N, block primes, alpha 1. `ringswitch16_35.json`.
    ConstContextPtr<word> switch_context;
    //! Degree N1, the ring the product runs in. `ringdegree12_35.json`.
    ConstContextPtr<word> small_context;
    //! `UserInterface::GetRingSwitchKey(N / N1)` on `switch_context`, from
    //! `PrepareRingSwitchKey(N1, small_secret_coeffs, product_level)`.
    const EvaluationKey<word> *forward = nullptr;
    //! `GetInverseRingSwitchKey(N / N1)`, from `PrepareInverseRingSwitchKey`
    //! with the same two secrets and level.
    const EvaluationKey<word> *inverse = nullptr;
    //! Exactly `N1 / small_degree` keys, from **`small_context`'s** own
    //! `PrepareModPackKeys(small_degree, product_level)`. These replace the
    //! `rank` big-ring keys the direct route needs, and there are sixteen
    //! times fewer of them at a sixteenth of the degree.
    std::vector<const EvaluationKey<word> *> modpack_keys;

    bool Enabled() const { return small_context != nullptr; }
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
   *        them and they must outlive it. **Ignored, and may be empty, when
   *        `descent` is filled in**: the repacking then happens on the small
   *        ring with `Descent::modpack_keys` instead.
   * @param descent [SYLPH]'s ring-switched route, or empty for the direct one
   */
  CoeffLinearLeg(ConstContextPtr<word> context, const Config &cfg,
                 std::vector<const EvaluationKey<word> *> modpack_keys,
                 Descent descent = Descent{});

  // disable copying (or moving also)
  CoeffLinearLeg(const CoeffLinearLeg &) = delete;
  CoeffLinearLeg &operator=(const CoeffLinearLeg &) = delete;

  /// k = degree / small_degree, and also the block's channels per ciphertext.
  int GetRank() const { return rank_; }
  /// N' = 2 * num_tokens.
  int GetSmallDegree() const { return small_degree_; }
  /// The ModPack keys a caller has to generate, for `PrepareModPackKeys`.
  static int SmallDegreeFor(int num_tokens) { return 2 * num_tokens; }
  /// Whether the product descends through [SYLPH]'s ring switch.
  bool IsRingSwitched() const { return descent_.Enabled(); }
  /// N / N1, the ring-switch rank. 1 without the descent.
  int GetRingRank() const { return ring_rank_; }
  /// N1 / small_degree, the ModDecomp rank. `rank_` without the descent.
  int GetSubRank() const { return sub_rank_; }

  void Project(std::vector<Ct> &res, const std::vector<Ct> &x, int in_channels,
               int out_channels, const std::vector<double> &w, double w_scale,
               const char *name) const override;

 private:
  /**
   * @brief The plaintext operands of one projection, converted once.
   *
   * A projection's weight matrix enters the product as `tiles * groups`
   * separate operands, and building them was **31% of the layer** -- 61.5 s of
   * host time per block against 0.43 s of GPU behind it (Doing.md 1.5an).
   * Every input to that work is a constant: the weights, the scale, the level,
   * the tiling. It was being redone 832 times per block and would have been
   * redone again for each of the 32 layers and every prompt.
   *
   * [SYLPH] section 5.3 calls this model conversion and does it once, offline,
   * keeping the result on the GPU (its section 5.1). That is what this holds.
   * The whole set is `sum(in * out)` entries over the seven projections, which
   * is 218M -- about 2.6 GB at three limbs, either as `PlainMatrix` words or
   * as int8 pieces.
   */
  struct Operands {
    int tiles = 0;
    int groups = 0;
    int tile = 0;
    int in_channels = 0;
    int out_channels = 0;
    double w_scale = 0.0;
    //! Of the weight matrix, so a second tensor arriving under the same name
    //! is caught rather than silently answered with the first one's operands.
    uint64_t fingerprint = 0;
    size_t bytes = 0;
    //! Indexed `tile_index * groups + group`.
    std::vector<PlainMatrix<word>> u;
#ifdef USE_CUBLAS
    std::vector<typename PcmmBlasHandler<word>::SplitMatrix> split;
#endif
  };

  // Row r of the operand carries output channel `group * rank + BitRev(r)`,
  // and column `p * rank + i` carries input channel
  // `(first_parent + p) * rank + BitRev(i)`. Both reversals are their own
  // inverse. Only the `num_parents` parents of the current tile are encoded,
  // so the operand is `rank x (num_parents * rank)`.
  void GatherWeights(std::vector<double> &values, const std::vector<double> &w,
                     int in_channels, int out_channels, int group,
                     double w_scale, int first_parent, int num_parents) const;

  //! Append one tile's `groups` operands to `res`, in group order.
  void BuildOperands(Operands &res, const std::vector<double> &w,
                     int in_channels, int out_channels, double w_scale,
                     int first_parent, int num_parents, int groups) const;

  //! The cached operands of `name`, built on first use. Asserts that the
  //! weight matrix is the one they were built from.
  const Operands &GetOperands(const char *name, const std::vector<double> &w,
                              int in_channels, int out_channels, double w_scale,
                              int parents, int groups, int tile) const;

  //! Size, scale and a strided sample of the entries. Cheap enough to run on
  //! every call and specific enough that a different tensor cannot pass.
  static uint64_t Fingerprint(const std::vector<double> &w, double w_scale);

  /**
   * @brief Which module component the operand's `flat`-th row or column is.
   *
   * The channel map is stated against the **direct** decomposition, where
   * `ModDecomp` sends the parent's coefficient `c` to component `c mod rank`
   * and so the component index simply *is* `c mod rank`. The descent reaches
   * the same 256 components by two strides instead of one:
   *
   *     small ciphertext  i = c mod ring_rank
   *     component         n = (c / ring_rank) mod sub_rank
   *
   * and the operand enumerates them in that order, `flat = i * sub_rank + n`,
   * because that is the order `Project` builds `columns` in and the order
   * `ModPack` reads its 16 rows back out in. Composing the two strides gives
   * `c mod rank = i + ring_rank * n`, which is what this returns -- so one
   * permutation of the plaintext, applied to both axes, is the whole
   * difference between the two routes. It rides `GatherWeights` and costs
   * nothing at all.
   *
   * Without the descent `ring_rank_` is 1 and this is the identity.
   */
  int Component(int flat) const {
    if (ring_rank_ == 1) return flat;
    return (flat / sub_rank_) + ring_rank_ * (flat % sub_rank_);
  }

  //! One tile's module components, in `Component()`'s order: `ModDecomp`
  //! straight from the parent, or a ring switch and then `ModDecomp` on each
  //! part. `span * rank_` of them either way.
  void Decompose(std::vector<MlweCiphertext<word>> &columns,
                 const std::vector<Ct> &x, int base, int span) const;

  ConstContextPtr<word> context_;
  Config cfg_;
  Descent descent_;
  int small_degree_;
  int rank_;
  //! N / N1, and N1 / small_degree. Their product is always `rank_`; without
  //! the descent they are 1 and `rank_`.
  int ring_rank_;
  int sub_rank_;
  std::vector<const EvaluationKey<word> *> modpack_keys_;
  MlweHandler<word> mlwe_;
  PcmmHandler<word> pcmm_;
  //! The descent's handlers, all on the small ring. Null without it.
  std::unique_ptr<RingSwitchHandler<word>> switcher_;
  std::unique_ptr<MlweHandler<word>> small_mlwe_;
  std::unique_ptr<PcmmHandler<word>> small_pcmm_;
  //! Where the product, ModPack and the rescale actually run: the small
  //! context with the descent, this leg's own without it. `Project` reads
  //! these rather than branching on `descent_` at every step, because the two
  //! routes differ only in the descend and return, not in between.
  const Parameter<word> *product_param_;
  const MlweHandler<word> *product_mlwe_;
  const PcmmHandler<word> *product_pcmm_;
  ConstContextPtr<word> product_context_;
  const std::vector<const EvaluationKey<word> *> *pack_keys_;
  //! `CHEDDAR_WEIGHT_CACHE=0` restores the recompute-every-call behaviour,
  //! which is what a card without room for the converted model needs and what
  //! the before/after measurement is against.
  bool cache_weights_;
  //! `CHEDDAR_PCMM_BLAS=0` puts the product back on `PcmmAccum`. Always false
  //! in a build without USE_CUBLAS.
  bool use_blas_;
  mutable std::map<std::string, Operands> operands_;
#ifdef USE_CUBLAS
  std::unique_ptr<PcmmBlasHandler<word>> blas_;
  std::unique_ptr<PcmmBlasHandler<word>> small_blas_;
  PcmmBlasHandler<word> *product_blas_ = nullptr;
#endif
};

}  // namespace cheddar
