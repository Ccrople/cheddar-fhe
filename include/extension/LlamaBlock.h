#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/Assert.h"
#include "core/Container.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/RmsNorm.h"
#include "extension/RoPe.h"
#include "extension/SiLu.h"
#include "extension/SlotPermute.h"
#include "extension/SoftMax.h"
#include "extension/SylphSchedule.h"

namespace cheddar {

/**
 * @brief One Llama-3 decoder block, as a sequence of `SylphSchedule` turns.
 *
 * `SylphSchedule` owns one turn of [SYLPH] figure 2's cycle -- the transport
 * between the slot and coefficient domains and the level and scale bookkeeping
 * at the two joints. This owns the block: which operator runs in which turn,
 * which tensor has to be carried across a turn it takes no part in, and what
 * every ciphertext's magnitude has to be when it reaches the next bootstrap.
 *
 * ## The block is six turns, not four
 *
 * The count is forced by two facts that are not about Llama at all.
 *
 * **A product consumes a level and the product ring has two.** [BAE]'s PC-MM
 * and [KANG]'s batch CC-MM both run at the bottom of the ladder because their
 * key material is published at PQ and PQ has to fit the *small* ring's budget
 * (`RingSwitch.h`). `ringdegree12_*` therefore has `max_level_ = 1`: one
 * product, level 1 to level 0, and no second one. Two products in a row are
 * separated by a bootstrap whether or not any operator sits between them.
 *
 * **A tensor that skips a turn still has to be carried.** V is produced by the
 * QKV projection and consumed by the AV product two turns later; K is produced
 * there and consumed one turn later. Both land at level 0, and level 0 is
 * below the product's input level, so they cannot simply wait -- they are
 * bootstrapped in the turn that consumes them, with no operator in the slot
 * leg. That transport is not overhead that better scheduling removes; it is
 * what the two-level product ring costs.
 *
 *     turn  slot leg                     product leg
 *     ----  ---------------------------  --------------------------------
 *     A     RMSNorm(attn)                Q, K, V  = h @ W_{q,k,v}   PC-MM
 *     B     RoPE(Q), RoPE(K); V carried  S        = Q K^T           CC-MM
 *     C     SoftMax(S)                   A        = P V             CC-MM
 *     D     untranspose A                O        = A @ W_o         PC-MM
 *     E     RMSNorm(ffn)                 G, U     = h @ W_{gate,up} PC-MM
 *     F     SiLU(G) * U                  Y        = . @ W_down      PC-MM
 *
 * The two residual additions are ordinary `Add`s in the coefficient domain at
 * level 0, between turn D and turn E and after turn F.
 *
 * ## The two ciphertext-ciphertext products do not use the cycle's transport
 *
 * `Project` is a coefficient-domain product: it is handed level-0 coefficient
 * ciphertexts and returns level-0 coefficient ciphertexts, so the block wraps
 * it in `Lift` on the way in and `Lower` on the way out. `Scores` and `Values`
 * are **not**. They take slot-domain operands and hand back slot-domain
 * results, because the SinC route the batch CC-MM needs -- the field swaps,
 * the ciphertext-axis exchange, `SlotToSinC`, the ring switch, the product,
 * `HalfBoot` and the StC prefix -- *is* a whole turn of the cycle, taken by a
 * different road. Handing it a coefficient ciphertext would mean bootstrapping
 * into slots only to descend again, which is one wasted bootstrap per score
 * ciphertext per block.
 *
 * So turns B and C have no `Lower` at their end and turn C and D have no
 * `Lift` at their start. The seam is exactly those two calls, and what the
 * block keeps is the level bookkeeping around them: `GetOperandLevel()` is
 * where Q, K and V must reach the leg, `GetProbLevel()` where P does, and
 * `GetResultLevel()` where both products come back.
 *
 * ## Turn D untransposes, and that is not bookkeeping either
 *
 * The CC-MM's lane index is the four slot bits directly above the token, so
 * the attention output comes back with the head there and the token above it
 * -- the transpose of what a coefficient-domain product reads. One inverse
 * `[4|7]` field swap puts it back, as two narrow swaps for the same reason the
 * forward pair is split (`SlotPermute.h`: 256 + 128 diagonals against 2048).
 * It costs two levels out of turn D's budget, which had nothing else in it.
 *
 * Everything else the layout needs is **free**, because a projection's output
 * channel order is the column order of its weight -- see `ChannelOrder`.
 *
 * ## Magnitude is a schedule variable, and it is the one that bites
 *
 * Every turn ends at a bootstrap, and `Boot` needs its decoded message inside
 * (-1, 1) -- measured, an arriving max of 9.4 left 1.20 bits of the message
 * and an arriving max of 0.5 left 10.95. So *every* tensor crossing a turn
 * boundary carries a calibrated scaling constant, and the constant is chosen
 * so the crossing magnitude is `Calibration::boot_max`.
 *
 * That constant is free in both directions and it is worth being explicit
 * about where each half lives:
 *
 *  - **Shrinking** rides on the product's plaintext. Every projection already
 *    multiplies by a weight matrix, so `w_scale` folds in for nothing. The two
 *    ciphertext-ciphertext products have no plaintext to fold into, so theirs
 *    rides on the *shift* instead: `Scores` adds the calibrated row maximum
 *    before the bootstrap, which is what bounds the crossing by `range/2`
 *    rather than by whatever the raw scores happen to reach.
 *  - **Growing** rides on `Canonicalise`, or -- on the ciphertext-ciphertext
 *    leg, which has no `Canonicalise` -- on the transform that stands in its
 *    place. It is a multiply by a constant, and the constant is arbitrary, so
 *    `magnitude` restores whatever the crossing divided out at no extra level.
 *    This is what lets SiLU and SoftMax receive their arguments on exactly the
 *    interval their polynomials were fitted on while the bootstrap before them
 *    still sees half of it.
 *
 * The second point is not a convenience. `SiLuHandler::Apply` takes `x / range`
 * and `SoftMaxHandler::Apply` takes `2(x - c)/M + 1`, both on [-1, 1]; a
 * bootstrap that wants at most 0.5 and a polynomial that wants exactly 1.0
 * cannot both be satisfied by the same ciphertext, and widening the fit
 * interval to 2x costs roughly a doubling of degree, which is a level. Free
 * growth at the canonicalisation is what removes the conflict.
 *
 * ## The residual chain pins one constant globally
 *
 * `h = x + Attn(...)` is an addition of two ciphertexts, so both operands must
 * carry the *same* constant. The block input, the O projection's output and
 * the down projection's output therefore all use `Calibration::residual` --
 * one number, calibrated against the largest magnitude anything in that chain
 * reaches, not per-tensor. Everything else is free to be sized per-tensor.
 *
 * ## What is not here
 *
 * The products. `LinearLeg` is a pure interface, and the block never sees a
 * ring, a ring switch, a tile or a key beyond its own. What it does see is the
 * *layout* the leg works in, because RoPE, SoftMax and the causal mask all
 * read it -- and it sees that through three narrow queries (`ChannelOrder`,
 * `GetScoreGroupSize`, `LocateScore`) rather than through the product itself.
 * With the interface, the same block runs against a host stand-in and against
 * the real path, and the difference between the two numbers is what the real
 * path costs.
 *
 * Attention sinks, on the other hand, ARE here, and they have to be: a query
 * that does not attend to the sink keys is computing a different function.
 * What is not here is their *hidden state*, which cannot pass the encrypted
 * RMSNorm at any polynomial degree -- see `Config::num_sink_tokens` for the
 * measurement and `PublicSinks` for what replaces it. [SYLPH] section 3.1.1 is
 * the same arrangement.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class LlamaBlock {
 private:
  using Ct = Ciphertext<word>;

 public:
  /** @brief Shape. Everything here is a property of the model, not the run. */
  struct Config {
    int num_tokens = 128;      //!< T, every token of the prompt, sinks included
    int num_channels = 4096;   //!< H
    int num_kv_channels = 1024;  //!< H_kv = num_kv_heads * head_dim
    int hidden = 14336;        //!< the FFN inner dimension
    int head_dim = 128;
    int first_position = 0;    //!< RoPE position of the first token
    double rope_theta = 500000.0;
    double eps = 1e-5;

    /**
     * @brief How many leading tokens are attention sinks, [SYLPH] 3.1.1.
     *
     * These occupy their token slots like any other -- they are *keys* that
     * every query attends to, and dropping them changes the function -- but
     * their hidden state never goes through the encrypted RMSNorm, because it
     * cannot. Measured on the real layer-2 bundle: the two leading
     * beginning-of-sequence tokens carry a per-token mean square of 33.1
     * against 3.4e-4 for the rest, a **285,946x** window, and one layer
     * constant plus one Chebyshev interval has to cover every token at once.
     * `RmsNorm.h` puts degree 23 at 30x. There is no degree that reaches
     * 285,946x, and the failure is not graceful: a Chebyshev polynomial
     * outside its interval grows like cosh(d arccosh(v)), so the sink slots
     * would not merely be wrong, they would take the whole ciphertext out of
     * the next bootstrap's range.
     *
     * So the caller substitutes a **public** filler for those tokens' hidden
     * state -- a rescaled copy of the true one, which is itself public,
     * because a prefix of beginning-of-sequence tokens is prompt-independent
     * and its trajectory through every layer is a constant of the model -- and
     * hands over `PublicSinks` to put the true K and V back. With the window
     * down to 5.2x the encrypted run then reproduces the true layer **exactly**
     * on every non-sink token: only the sink rows of Q, of the attention
     * output and of the FFN differ, and those are discarded.
     */
    int num_sink_tokens = 0;
  };

  /**
   * @brief The sink tokens' true K and V, and what the block will compute for
   * them from the public filler, so the difference can be added back.
   *
   * Both are **pre-RoPE** and unscaled: the block folds in the crossing
   * constants itself, and RoPE runs afterwards at the sink tokens' own
   * positions, which is what it would have done anyway.
   *
   * Every entry here is public. The correction is a plaintext addition in the
   * coefficient domain, so it costs no level and no key.
   */
  struct PublicSinks {
    std::vector<double> k;  //!< [num_sink_tokens][H_kv], the true keys
    std::vector<double> v;  //!< [num_sink_tokens][H_kv], the true values
    std::vector<double> computed_k;  //!< what the filler produces, same shape
    std::vector<double> computed_v;  //!< what the filler produces, same shape
  };

  /**
   * @brief Everything [SYLPH] section 3.1.3 calls calibration.
   *
   * None of these can be derived from the model alone -- they are properties
   * of the activations, measured offline on representative inputs. A caller
   * that guesses is not approximated badly, it is *wrong*, silently: an input
   * outside a Chebyshev interval is evaluated as whatever the polynomial does
   * out there, and a crossing magnitude above `boot_max` loses the message.
   */
  struct Calibration {
    double boot_max = 0.5;  //!< magnitude every tensor crosses a turn at
    double residual = 1.0;  //!< the one constant the residual chain shares

    // RMSNorm: alpha_L such that alpha_L * mean(x^2) lands in the window.
    // Stated for the *unscaled* activation; the block divides out whatever
    // constant the ciphertext actually carries.
    double attn_alpha = 1.0;
    double ffn_alpha = 1.0;
    double rms_window = 4.18;
    int rms_degree = 7;

    // SoftMax: row R's argument is taken to lie in [shift[R] - range, shift[R]].
    //
    // THE SHIFT IS PER ROW AND THAT IS NOT AN OPTIMISATION.
    //
    // With one global shift, measured on the real layer-2 scores, ||y||_2^2
    // spans **5.39e9x** across rows -- the rows differ in their own maximum by
    // about twelve nats, so a row whose scores all sit far below the global
    // maximum exponentiates to almost nothing. No polynomial covers that, at
    // any degree, and the failure is not gradual: the normalisation for such a
    // row is wrong by orders of magnitude and the squaring that follows makes
    // it worse.
    //
    // Shifting each row by its own calibrated maximum puts every row's largest
    // exponential at 1 by construction, so `||y||_2^2` lands in [1, keys] and
    // the spread is the row length rather than the score dispersion. That is
    // 64x here against 5.39e9x, and 64x is an ordinary Chebyshev problem.
    //
    // It costs nothing. The shift is added by the score product in the
    // coefficient domain, where a per-row plaintext and a uniform one are the
    // same `Add(Ct, Pt)`. And it is legitimate: [SYLPH] section 3.1.3's
    // calibration is exactly this -- offline statistics of the activations,
    // per layer, used as public constants.
    // The masked entries are the other half of the story, and they are why
    // the shift is per *entry* and not merely per row.
    //
    // The row maximum that makes ||y||_2^2 land in [1, keys] has to be taken
    // over the causal entries only -- a row's own largest exponential is the
    // one it will actually normalise by. But the exponential still runs on
    // the masked entries; the mask multiplies them away afterwards, not
    // before. Their argument then sits above the interval, by up to the gap
    // between a row's overall maximum and its causal one, and a Chebyshev
    // polynomial outside its interval grows like cosh(d * arccosh(v)).
    //
    // `softmax_mask_shift` is the calibrated size of that gap. Subtracting it
    // from every masked entry puts them back inside, at the cost of widening
    // the range that has to cover them. Both halves are plaintext additions
    // in the coefficient domain, so the whole arrangement is free.
    double softmax_range = 21.0;
    std::vector<double> softmax_shift;  //!< one per row, head * T + query
    double softmax_mask_shift = 0.0;    //!< extra, on masked entries only
    std::vector<double> softmax_norm_lo{1.0};
    std::vector<double> softmax_norm_hi{64.0};
    int softmax_iters = 1;
    int softmax_exp_degree = 9;
    // The inverse square root rides the auxiliary track, which comes back from
    // a bootstrap with the operator's whole level budget, so its degree is
    // nearly free -- but not entirely. It leaves the auxiliary value at
    // `aux_return_level - ceil(log2(d+1))`, and the main track then spends two
    // more levels, so `d = 31` (five levels) is the ceiling on a slack-8 set
    // and `d = 63` lands the result one level below StC.
    int softmax_inv_sqrt_degree = 31;
    int softmax_early_inv_sqrt_degree = 4;

    // SiLU: the fit half-interval. Set it to twice the measured maximum, so
    // the crossing sits at boot_max and Canonicalise grows it back to exactly
    // the interval the polynomial owns.
    double silu_range = 2.5;
    int silu_degree = 15;

    // Crossing constants for the tensors that are not in the residual chain.
    // Each is 1 / (the tensor's calibrated maximum) times boot_max.
    double size_q = 1.0;
    double size_k = 1.0;
    double size_v = 1.0;
    //! **Not a free knob on a slot-domain leg, and DescribePlan reports what
    //! it actually is.** The score product has no plaintext operand, so
    //! nothing can scale its output before the bootstrap that follows it. What
    //! the bootstrap carries is fixed by the operands and by the shift:
    //! `size_q * size_k * sqrt(head_dim) * softmax_range / 2`, because the
    //! shift centres the row on its own calibrated maximum and the SoftMax
    //! interval is what is left. Calibrating the scores means moving `size_q`
    //! and `size_k`; this field is kept for a coefficient-domain leg, which
    //! does have a plaintext to fold it into.
    double size_scores = 1.0;
    double size_probs = 1.0;
    double size_attn = 1.0;
    double size_gate = 1.0;
    double size_up = 1.0;
  };

  /**
   * @brief The product leg: everything that mixes channels or tokens.
   *
   * `Project` takes and returns **coefficient-encoded** ciphertexts in the
   * block's packing -- slot `s` of ciphertext `i` holds token `s % T` of
   * channel `i * (slots/T) + s / T`, placed by
   * `AttentionPacking::CoeffOfSlot`. The implementation owns the level it
   * works at, the ring it works in, and the descent to it; it must return at
   * level 0, which is where `SylphSchedule::ToSlot` expects its input.
   *
   * `Scores` and `Values` take and return **slot-encoded** ciphertexts, at
   * the levels `LlamaBlock::GetOperandLevel`, `GetProbLevel` and
   * `GetResultLevel` name. See the class comment for why the two products do
   * not share `Project`'s domain.
   */
  class LinearLeg {
   public:
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

  /** @brief The layer's plaintext weights, row-major and already transposed. */
  struct Weights {
    std::vector<double> attn_norm;  //!< H
    std::vector<double> ffn_norm;   //!< H
    std::vector<double> wq;         //!< [H][H]
    std::vector<double> wk;         //!< [H][H_kv]
    std::vector<double> wv;         //!< [H][H_kv]
    std::vector<double> wo;         //!< [H][H]
    std::vector<double> wgate;      //!< [H][hidden]
    std::vector<double> wup;        //!< [H][hidden]
    std::vector<double> wdown;      //!< [hidden][H]
  };

  /**
   * @param leg the product implementation. It is a constructor argument and
   * not a `Run` argument because the block's own operators depend on its
   * layout: RoPE's tables are per ciphertext once the channels are permuted,
   * SoftMax's group size is the leg's score grouping, and the causal mask is
   * built in the leg's score layout. The reference must outlive the block.
   */
  LlamaBlock(std::shared_ptr<const BootContext<word>> boot, const Config &cfg,
             const Calibration &cal, const LinearLeg &leg);

  // disable copying (or moving also)
  LlamaBlock(const LlamaBlock &) = delete;
  LlamaBlock &operator=(const LlamaBlock &) = delete;

  /** @brief Ciphertexts a tensor of `channels` columns occupies. */
  int NumCiphertexts(int channels) const;

  /** @brief Every rotation distance the block's operators need keys for. */
  void AddRequiredRotations(EvkRequest &req) const;

  /**
   * @brief Whether every turn fits the schedule, and what the plan is.
   *
   * Arithmetic only -- nothing is encrypted, so a block that does not fit is
   * found in milliseconds rather than after the first hour of key generation.
   */
  std::string DescribePlan() const;

  /** @brief False if any turn's slot leg is deeper than the slack allows. */
  bool Fits(std::string *why = nullptr) const;

  /**
   * @brief Run the whole block.
   *
   * @param res output, the residual stream after the block, coefficient
   * encoded at level 0 and carrying `Calibration::residual`
   * @param x input, the residual stream, coefficient encoded at level 0 and
   * carrying `Calibration::residual`
   * @param w the layer's weights
   * @param sinks the public K and V of the first `Config::num_sink_tokens`
   * tokens, and what the filler produces for them; empty when there are none
   * @param evk_map every key the block needs
   */
  void Run(std::vector<Ct> &res, const std::vector<Ct> &x, const Weights &w,
           const PublicSinks &sinks, const EvkMap<word> &evk_map) const;

  /**
   * @brief The same block in the clear, for a reference the encrypted run can
   * be measured against.
   *
   * Uses the *true* functions, not the handlers' polynomials, so the
   * difference it reports is the whole cost of running the block encrypted --
   * approximation and circuit together. Each handler's `Plain*` gives the
   * other split when it is wanted.
   *
   * @param res output, T x H row-major, unscaled
   * @param x input, T x H row-major, unscaled
   */
  void PlainRun(std::vector<double> &res, const std::vector<double> &x,
                const Weights &w) const;

  const Config &GetConfig() const { return cfg_; }
  const Calibration &GetCalibration() const { return cal_; }
  const SylphSchedule<word> &GetSchedule() const { return *sched_; }

  /** @brief Level every slot-domain operator runs at. */
  int GetOperatorLevel() const { return sched_->GetSlotLevel() - 1; }
  /** @brief Level Q, K and V reach `Scores` and `Values` at: after RoPE. */
  int GetOperandLevel() const { return GetOperatorLevel() - 1; }
  /** @brief Level P reaches `Values` at: where SoftMax leaves it. */
  int GetProbLevel() const { return sched_->GetStCLevel(); }
  /** @brief Level both products must return at, which is the operator one. */
  int GetResultLevel() const { return GetOperatorLevel(); }

  /** @brief Score ciphertexts the whole layer holds. */
  int NumScoreCiphertexts() const;

  /** @brief The causal mask, in the leg's score layout, one per ciphertext. */
  void BuildCausalMask(std::vector<std::vector<Complex>> &res) const;

 private:
  // One turn's slot leg: bootstrap every ciphertext, canonicalise it onto the
  // operator's scale and magnitude, and hand back the result. `magnitude` is
  // what the crossing constant has to be undone by; `shift` is added after,
  // for the one operator whose argument is affine rather than linear.
  void Lift(std::vector<Ct> &res, const std::vector<Ct> &x, double magnitude,
            const EvkMap<word> &evk_map, double shift = 0.0) const;
  // The same crossing for a producer that already merged its output in pairs
  // (`LinearLeg::ProjectMerged`): `res` is twice as long as `merged`, and what
  // it holds is what `Lift` would have handed back.
  void LiftMerged(std::vector<Ct> &res, const std::vector<Ct> &merged,
                  double magnitude, const EvkMap<word> &evk_map,
                  double shift = 0.0) const;
  // The other half: StC every ciphertext back to the coefficient domain.
  void Lower(std::vector<Ct> &res, const std::vector<Ct> &x,
             const EvkMap<word> &evk_map) const;

  // A tensor's weight with its columns (or, for W_o, its rows) put in the
  // order the leg asked for. Returns `w` itself when the leg wants none, so
  // the untouched matrices are never copied.
  //
  // The permuted copy is kept per tensor rather than rebuilt per call. It is a
  // pure function of the weight and the order, both of which are fixed for the
  // life of the block, and W_q is 4096 x 4096 doubles -- 134 MB allocated,
  // scattered and freed on every call, for a result that could not have
  // changed. The source pointer and size are checked so a different tensor
  // cannot be answered with a held permutation.
  const std::vector<double> &Reorder(const std::vector<double> &w, int rows,
                                     int cols,
                                     typename LinearLeg::Tensor which,
                                     bool permute_rows) const;

  // RoPE's channel map for each ciphertext of a tensor, derived from the
  // leg's channel order, and the slot distance to the partner channel. Only
  // the DISTINCT maps come back, with `variant` saying which one each
  // ciphertext uses -- a map is three host-encoded plaintexts and a tensor
  // repeats its layout once per head group.
  void BuildRoPeLayout(std::vector<std::vector<int>> &maps,
                       std::vector<int> &variant, int &step,
                       typename LinearLeg::Tensor which, int channels) const;

  // RoPE over a whole tensor, variant-major so the handler's one plaintext
  // cache is rebuilt once per distinct layout rather than once per ciphertext.
  void ApplyRoPe(std::vector<Ct> &res, const std::vector<Ct> &x,
                 const RoPeHandler<word> &rope,
                 const std::vector<int> &variant,
                 const EvkMap<word> &evk_map) const;

  // Levels turn D's untranspose spends: two, or none when the leg hands the
  // attention output back in the block's own packing.
  int OutSwapDepth() const;

  // The untranspose of turn D: the inverse of the [4|7] field swap the CC-MM
  // leaves behind, as two narrow swaps.
  void Untranspose(std::vector<Ct> &res, const std::vector<Ct> &x,
                   const EvkMap<word> &evk_map) const;

  // Say which turn is running and what level it is on. A hundred bootstraps
  // is minutes of silence, and an assert inside one of them reports only that
  // the prime counts differed.
  void Announce(const char *what, const std::vector<Ct> &cts,
                int expect_level = -1) const;

  // Per-channel weights, broadcast into the block's packing, with sqrt(alpha)
  // folded in the way RmsNormHandler asks for.
  void SpreadNormWeight(std::vector<std::vector<Complex>> &res,
                        const std::vector<double> &w, double alpha) const;

  // Add `scale * (want - got)` on the sink tokens' slots and nothing anywhere
  // else. Both operands are public, so this is a coefficient-domain plaintext
  // addition: no level, no key, and it lands before RoPE, which is where the
  // cached K of a sink token belongs.
  void InjectSinks(std::vector<Ct> &cts, const std::vector<double> &want,
                   const std::vector<double> &got, int channels, double scale,
                   const std::vector<int> &order, bool merged = false) const;

  std::shared_ptr<const BootContext<word>> boot_;
  Config cfg_;
  Calibration cal_;
  const LinearLeg *leg_;
  int num_slots_;
  int channels_per_ct_;
  // The leg's channel orders, taken once, and their inverses, which is what
  // the sink injection and RoPE's tables actually read.
  std::vector<int> q_order_, k_order_, v_order_, o_order_;
  // Reorder's held permutations, one per tensor, with the source they were
  // built from.
  struct Reordered {
    const double *source = nullptr;
    size_t size = 0;
    std::vector<double> data;
  };
  mutable Reordered reordered_[4];
  std::vector<int> q_variant_, k_variant_;

  std::unique_ptr<SylphSchedule<word>> sched_;
  std::unique_ptr<RmsNormHandler<word>> attn_norm_;
  std::unique_ptr<RmsNormHandler<word>> ffn_norm_;
  std::unique_ptr<RoPeHandler<word>> q_rope_, k_rope_;
  std::unique_ptr<SoftMaxHandler<word>> softmax_;
  std::unique_ptr<SiLuHandler<word>> silu_;
  std::unique_ptr<SlotPermute<word>> untranspose_a_, untranspose_b_;
};

}  // namespace cheddar
