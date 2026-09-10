#pragma once

#include <vector>

#include "common/Assert.h"
#include "core/Container.h"

namespace cheddar {

template <typename word>
class Context;

/**
 * @brief The interface every projection leg implements: the seven plaintext
 * products, and the two ciphertext-ciphertext ones an attention needs.
 *
 * ## Why this is its own header
 *
 * It was `LlamaBlock<word>::LinearLeg`, a nested class of the ORDINARY-ring
 * block. But the CI path's own projection, `CoeffLinearLeg`, implements it --
 * and so, through it, do `CiProjectionLeg` and every `ProjectOnlyLeg` in the
 * tests. The interface therefore outlived the block that declared it, and
 * when the ordinary block was removed (2026-09-10) it had to be somewhere the
 * CI path could reach without dragging six turns of `SylphSchedule`
 * bookkeeping behind it.
 *
 * The doc comments below still name `LlamaBlock::GetOperandLevel` and its
 * siblings, because that is where the LEVELS these signatures speak of were
 * defined and the CI path chose the same ones (`CiSinCAttention::Config`'s
 * `forward_level` / `inverse_level`). They are a description of the contract,
 * not a live dependency.
 */
template <typename word>
class LinearLeg {
 public:
  using Ct = Ciphertext<word>;

  virtual ~LinearLeg() = default;

  /** @brief Which tensor a channel order is being asked for. */
  enum class Tensor {
    kQuery,   //!< W_q's columns
    kKey,     //!< W_k's columns
    kValue,   //!< W_v's columns
    kAttnOut  //!< the attention output, so W_o's ROWS
  };

  /**
   * @brief The output-channel order this leg wants, or empty for the block's
   * own `[head][channel]` order.
   *
   * A projection's output channel order is the column order of its weight
   * matrix, so **any** permutation of the channels is free: the block
   * reorders the plaintext once, offline, and the ciphertext comes out of
   * the product already packed the way the next stage reads it. That is what
   * pays for the CC-MM's layout, which otherwise needs a slot permutation --
   * a `LinearTransform`, a level and hundreds of plaintexts -- on Q, K and V
   * alike.
   *
   * `kAttnOut` is the same statement read backwards: the attention output
   * arrives in a layout the block did not choose, and permuting W_o's *rows*
   * is how the O projection reads it.
   *
   * @param res receives a permutation of `[head * head_dim + channel]`,
   * giving the leg's own channel index, or is left empty
   */
  virtual void ChannelOrder(std::vector<int> &res, Tensor which) const {
    res.clear();
  }

  /** @brief Ciphertexts one score row spans; 1 when a row fits in one. */
  virtual int GetScoreGroupSize() const { return 1; }

  /**
   * @brief Where score entry `(head, query, key)` lives, in slots.
   *
   * The block needs this for the causal mask, which it builds rather than
   * being handed: `key <= query` is arithmetic, and a caller that had to
   * know the leg's score layout to state it would be doing the leg's job.
   */
  virtual void LocateScore(int head, int query, int key, int &ct,
                           int &slot) const = 0;

  /**
   * @brief Whether `Values` returns with the head above the token, so that
   * the block has to untranspose before the O projection.
   */
  virtual bool NeedsOutputSwap() const { return false; }

  /**
   * @brief res = w_scale * (x @ w), a plaintext projection.
   *
   * @param res output, `out_channels` wide, resized by the callee
   * @param x input, `in_channels` wide
   * @param in_channels inner dimension
   * @param out_channels output width
   * @param w row-major [in_channels][out_channels]
   * @param w_scale a constant multiplying every entry of w; this is where
   * every crossing constant and every 1/range folds in for free
   * @param name for diagnostics, e.g. "Q"
   */
  virtual void Project(std::vector<Ct> &res, const std::vector<Ct> &x,
                       int in_channels, int out_channels,
                       const std::vector<double> &w, double w_scale,
                       const char *name) const = 0;

  /**
   * @brief `Project`, with consecutive output ciphertexts merged in pairs.
   *
   * Output ciphertext `2m` lands in coefficients `0 .. N/2-1` of `res[m]` and
   * `2m+1` in `N/2 .. N-1`. That is the form `BootContext::HalfBootSplit`
   * consumes, so the block's next step reads it directly.
   *
   * WHY THE PROJECTION AND NOT THE BOOTSTRAP DOES THIS. The merge itself is
   * one multiply and one add wherever it happens; what makes the position
   * matter is `ModPack`, which costs `rank` key switches per output
   * ciphertext and is 81% of the block's seven projections against 6% for
   * the product. Merging before the pack halves the pack. A leg that cannot
   * do that still answers correctly through the default below -- it projects
   * as usual and merges on the big ring -- and pays what it would have paid.
   *
   * @param res `out_channels / (2 * channels_per_ct)` merged ciphertexts
   */
  virtual void ProjectMerged(std::vector<Ct> &res, const std::vector<Ct> &x,
                             int in_channels, int out_channels,
                             const std::vector<double> &w, double w_scale,
                             const char *name,
                             const Context<word> &context) const {
    std::vector<Ct> plain;
    Project(plain, x, in_channels, out_channels, w, w_scale, name);
    AssertTrue(plain.size() % 2 == 0,
               "ProjectMerged: an odd number of output ciphertexts has no "
               "pairing");
    res.resize(plain.size() / 2);
    for (size_t m = 0; m < res.size(); m++) {
      Ct shifted;
      context.MultImaginaryUnit(shifted, plain[2 * m + 1]);
      context.Add(res[m], shifted, plain[2 * m]);
    }
  }

  /**
   * @brief res = magnitude * (Q K^T + shift), per head, with GQA repetition.
   *
   * ## The order is not the obvious one, and it is what makes the bootstrap
   * ## carry the SoftMax interval rather than the raw scores
   *
   * `shift` is added to the **raw** product and the whole sum is scaled
   * afterwards. That is deliberate. On a SinC leg the addition happens at
   * level 0, in SinC form, immediately before the bootstrap, and the scaling
   * happens after it, in the transform that replaces `Canonicalise`. So what
   * crosses is `raw - c`, which the calibration bounds by `range/2`, and not
   * `raw`, which nothing bounds -- a row's scores sit wherever its own
   * maximum puts them. A plaintext addition costs no level in either place;
   * the difference is entirely in what the bootstrap has to carry.
   *
   * @param res output, `num_heads * T` rows of `T` keys, slot-encoded in the
   * leg's own score layout at `LlamaBlock::GetResultLevel()`
   * @param q the rotated queries, `num_channels` wide, slot-encoded at
   * `LlamaBlock::GetOperandLevel()`
   * @param k the rotated keys, `num_kv_channels` wide, at the same level
   * @param magnitude multiplies the shifted product; it carries
   * 1/sqrt(head_dim), 2/range and both operands' crossing constants
   * @param shift added to the raw product, one entry per score in the order
   * `(head * T + query) * T + key`. Per entry rather than per row because
   * the masked entries need their own correction; see
   * `Calibration::softmax_shift` and `Calibration::softmax_mask_shift`.
   */
  virtual void Scores(std::vector<Ct> &res, const std::vector<Ct> &q,
                      const std::vector<Ct> &k, double magnitude,
                      const std::vector<double> &shift) const = 0;

  /**
   * @brief res = magnitude * (P V), per head, with GQA repetition.
   *
   * @param res output, `num_channels` wide, slot-encoded at
   * `LlamaBlock::GetResultLevel()` and, if `NeedsOutputSwap()`, transposed
   * @param p the SoftMax probabilities, slot-encoded at or above
   * `LlamaBlock::GetProbLevel()`, in the leg's own score layout
   * @param v the values, `num_kv_channels` wide, slot-encoded at
   * `LlamaBlock::GetOperandLevel()`
   * @param magnitude multiplies every entry
   */
  virtual void Values(std::vector<Ct> &res, const std::vector<Ct> &p,
                      const std::vector<Ct> &v,
                      double magnitude) const = 0;
};

}  // namespace cheddar
