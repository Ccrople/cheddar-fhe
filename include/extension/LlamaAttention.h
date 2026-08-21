#pragma once

#include <memory>
#include <vector>

#include "extension/LlamaBlock.h"
#include "extension/LlamaLinear.h"
#include "extension/SinCAttention.h"

namespace cheddar {

/**
 * @brief The whole product leg: `CoeffLinearLeg` for the seven plaintext
 * products, `SinCAttention` for the two ciphertext-ciphertext ones.
 *
 * `CoeffLinearLeg` deliberately leaves `Scores` and `Values` pure virtual --
 * they are a different primitive, in a different domain, at a different ring
 * degree, and a silent fallback would be worse than a link error. This is what
 * fills them in, and with it a `LlamaBlock` runs with nothing on the host.
 *
 * ## What this class is, beyond forwarding
 *
 * Three things, and none of them is the product itself.
 *
 * **The layout, stated once.** `SinCAttention` reads its operands in three
 * packings that nothing else in the block uses -- Q's, K's and V's -- and hands
 * its results back in a fourth. The block cannot guess them and should not:
 * they are consequences of the batch CC-MM's lane index and of the field swaps
 * that reach it. So they live here, as `ChannelOrder` and `LocateScore`, and
 * the block reads them through `LinearLeg`.
 *
 * The important half of that is that they cost **nothing**. A projection's
 * output channel order is the column order of its weight, so the block permutes
 * a plaintext offline and the ciphertext comes out packed. The one piece that
 * cannot ride the weights is the token axis, because the token's position is
 * fixed by the PC-MM ladder rather than chosen -- and that is exactly the one
 * piece the block pays a transform for, in turn D's untranspose.
 *
 * **The head groups.** One call of the batch CC-MM does `lanes` heads. Llama-3
 * has 32 and `sub_degree = 32` gives 16 lanes, so every product is two calls,
 * over disjoint slices of the same tensors. Ciphertexts do not copy, so the
 * slices are handed over as pointers.
 *
 * **The chain's constant.** `SinCAttention`'s route ends in `HalfBoot` and a
 * transform standing in for `Canonicalise`, and what that leaves is a constant
 * the parameter set fixes and **which has to be measured** -- HalfBoot says so
 * in its own comment. `Config::chain_constant` is where the measurement goes;
 * the leg divides it out so the block's `magnitude` means what the block
 * thinks it means. See `Config::chain_constant` for how to obtain one.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SinCLinearLeg : public CoeffLinearLeg<word> {
 private:
  using Ct = Ciphertext<word>;
  using Base = CoeffLinearLeg<word>;
  using Tensor = typename LlamaBlock<word>::LinearLeg::Tensor;

 public:
  struct Config {
    int num_heads = 32;   //!< query heads in the layer
    int num_kv_heads = 8;  //!< KV heads; `num_heads / num_kv_heads` is the GQA
                           //!< group

    /**
     * @brief What the chain leaves when `magnitude` is 1, measured.
     *
     * `HalfBoot` declares its output at `eval_mod_->end_scale_` and its own
     * comment says the remaining constant "gets measured rather than derived
     * through cts_const_, stc_const_ and q0". On `sylphflow16_35` with slack 8
     * it is **0.0298533 = 2^-5.066**, against the `2^-log_message_ratio = 2^-5`
     * the design intends; the few percent it misses by is the part that does
     * not cancel because this leg has no `ToCoeff` to cancel against.
     *
     * To measure one on a set that has not been measured: run `Scores` with
     * this at 1.0 on operands whose product is known, fit the constant, and
     * put its value here. `SinCAttentionTest` does exactly that and prints it.
     * Leaving it at 1.0 is not an error, it is a different circuit: everything
     * downstream then sees a message 33x too small, and the first thing to
     * notice will be SoftMax's polynomial being evaluated near zero.
     */
    double chain_constant = 1.0;
  };

  /**
   * @param cfg the head counts and the chain constant
   * @param attn_cfg levels and shape for `SinCAttention`; the block's
   *        `GetOperandLevel()`, `GetProbLevel()` and `GetSlotLevel()` set
   *        `swap_level`, `sinc_level` and `prefix_level`
   * @param keys the four key sets one call needs, on both rings
   * @param linear_cfg `CoeffLinearLeg`'s own configuration
   * @param modpack_keys `CoeffLinearLeg`'s ModPack keys
   */
  SinCLinearLeg(std::shared_ptr<const BootContext<word>> boot,
                ConstContextPtr<word> switch_ctx,
                ConstContextPtr<word> small_ctx, const Config &cfg,
                const typename SinCAttention<word>::Config &attn_cfg,
                const typename SinCAttention<word>::Keys &keys,
                const typename CoeffLinearLeg<word>::Config &linear_cfg,
                std::vector<const EvaluationKey<word> *> modpack_keys);

  // disable copying (or moving also)
  SinCLinearLeg(const SinCLinearLeg &) = delete;
  SinCLinearLeg &operator=(const SinCLinearLeg &) = delete;

  const SinCAttention<word> &GetAttention() const { return attn_; }

  /** @brief Every rotation the two products need on the block's ring. */
  void AddRequiredRotations(EvkRequest &req) const {
    attn_.AddRequiredRotations(req);
  }
  /** @brief The same on the product ring. */
  std::vector<int> SmallRotationIndices() const {
    return attn_.SmallRotationIndices();
  }

  /** @brief Install a measured chain constant after construction. */
  void SetChainConstant(double c);
  double GetChainConstant() const { return cfg_.chain_constant; }

  // ---- LinearLeg -------------------------------------------------------
  void ChannelOrder(std::vector<int> &res, Tensor which) const override;
  int GetScoreGroupSize() const override { return num_cts_; }
  void LocateScore(int head, int query, int key, int &ct,
                   int &slot) const override;
  bool NeedsOutputSwap() const override { return true; }

  void Scores(std::vector<Ct> &res, const std::vector<Ct> &q,
              const std::vector<Ct> &k, double magnitude,
              const std::vector<double> &shift) const override;
  void Values(std::vector<Ct> &res, const std::vector<Ct> &p,
              const std::vector<Ct> &v, double magnitude) const override;

 private:
  // The score and output layouts differ only in what the second index is --
  // the key for one, the channel for the other -- so they are one function.
  void LocateLane(int head, int token, int index, int &ct, int &slot) const;
  // The block's shift, one entry per (head, query, key), turned into the SinC
  // message the product's own output is indexed by.
  void BuildShift(std::vector<std::vector<Complex>> &res,
                  const std::vector<double> &shift, int group) const;
  // Rebuild the prefixes only when a magnitude has actually moved. Both are
  // set together, so both are remembered.
  void Retune(double score, double value) const;

  Config cfg_;
  // Mutable because Retune is the only place a magnitude is installed and it
  // runs from the const Scores and Values. What it rebuilds is two transforms
  // of 31 diagonals; no key and no ciphertext is touched.
  mutable SinCAttention<word> attn_;
  typename SinCAttention<word>::Keys keys_;
  int num_slots_;
  int num_cts_;
  int lanes_;
  int head_dim_;
  int num_tokens_;
  int channels_per_ct_;
  int num_src_cts_;  //!< ciphertexts the KV tensor of one group occupies
  mutable double score_magnitude_ = 1.0;
  mutable double value_magnitude_ = 1.0;
};

}  // namespace cheddar
