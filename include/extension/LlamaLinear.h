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
#ifdef USE_CUBLAS
#include "core/PcmmBlas.h"
#endif
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
 * That is why the contraction is **tiled**. Llama's down projection contracts
 * over 14336 channels, which is 56 parents and about 11.3 GB of module
 * components held at once -- more than a 48 GiB card has free in the middle of
 * a block that is already holding a dozen tensors. `Config::parents_per_tile`
 * caps how many parents are decomposed at a time; each tile's partial products
 * are ModPacked and **added as ordinary RLWE ciphertexts** before a single
 * rescale at the end. Summing after ModPack rather than before it is not a
 * preference: there is no MLWE addition in the library, and a partial sum that
 * has already been repacked is an ordinary ciphertext at the product's level.
 *
 * The cost of a tile is one extra ModPack per output group, which is `rank`
 * key switches at the lowest level -- so the tiling trades roughly 46 ms per
 * group per extra tile against gigabytes. It does not change the result: the
 * sum is exact, and the single rescale at the end is the same rescale the
 * untiled path performs.
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

  ConstContextPtr<word> context_;
  Config cfg_;
  int small_degree_;
  int rank_;
  std::vector<const EvaluationKey<word> *> modpack_keys_;
  MlweHandler<word> mlwe_;
  PcmmHandler<word> pcmm_;
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
#endif
};

}  // namespace cheddar
