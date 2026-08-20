#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/Container.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/RmsNorm.h"
#include "extension/RoPe.h"
#include "extension/SiLu.h"
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
 *     B     RoPE(Q), RoPE(K)             S        = Q K^T           CC-MM
 *     C     SoftMax(S); V transported    A        = P V             CC-MM
 *     D     (transport only)             O        = A @ W_o         PC-MM
 *     E     RMSNorm(ffn)                 G, U     = h @ W_{gate,up} PC-MM
 *     F     SiLU(G) * U                  Y        = . @ W_down      PC-MM
 *
 * The two residual additions are ordinary `Add`s in the coefficient domain at
 * level 0, between turn D and turn E and after turn F.
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
 *    ciphertext-ciphertext products take an explicit `scale` for the same
 *    reason -- the 1/sqrt(head_dim) is already there.
 *  - **Growing** rides on `Canonicalise`. It is a multiply by a constant and a
 *    rescale, and the constant is arbitrary, so `magnitude` restores whatever
 *    the crossing divided out at no extra level. This is what lets SiLU and
 *    SoftMax receive their arguments on exactly the interval their polynomials
 *    were fitted on while the bootstrap before them still sees half of it.
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
 * weight matrix, a head, a transpose or a ring switch. That is deliberate:
 * the coefficient-domain product path (RingSwitch -> ModDecomp -> PC-MM /
 * CC-MM -> ModPack -> RingSwitch) is the one part of the block that does not
 * yet run on `bootparam_35`, and a block that hard-coded it could not be run
 * at all until it did. With the interface, the same block runs today against
 * any implementation and against the real one when it exists.
 *
 * Also not here: attention sinks. [SYLPH] section 3.1.1 removes them from the
 * encrypted path entirely and injects their KV states from a static cache, and
 * `RmsNorm.h` records why -- on this bundle the sink tokens make the per-token
 * mean square span 285,946x against the 30x window a layer constant can cover.
 * `num_tokens` is the count of *user* tokens and the caller must have excluded
 * them already.
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
    int num_tokens = 64;       //!< T, user tokens; sinks already excluded
    int num_channels = 4096;   //!< H
    int num_kv_channels = 1024;  //!< H_kv = num_kv_heads * head_dim
    int hidden = 14336;        //!< the FFN inner dimension
    int head_dim = 128;
    int first_position = 0;    //!< RoPE position of the first user token
    double rope_theta = 500000.0;
    double eps = 1e-5;
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
    double softmax_range = 21.0;
    std::vector<double> softmax_shift;  //!< one per row, head * T + query
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
    double size_scores = 1.0;  //!< applied to (s - shift) / range
    double size_probs = 1.0;
    double size_attn = 1.0;
    double size_gate = 1.0;
    double size_up = 1.0;
  };

  /**
   * @brief The product leg: everything that mixes channels or tokens.
   *
   * All arguments and results are **coefficient-encoded** ciphertexts in the
   * block's packing -- slot `s` of ciphertext `i` holds token `s % T` of
   * channel `i * (slots/T) + s / T`, placed by
   * `AttentionPacking::CoeffOfSlot`. The implementation owns the level it
   * works at, the ring it works in, and the descent to it; it must return at
   * level 0, which is where `SylphSchedule::ToSlot` expects its input.
   */
  class LinearLeg {
   public:
    virtual ~LinearLeg() = default;

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
     * @brief res = scale * (Q K^T) + shift, per head, with GQA repetition.
     *
     * The shift is an additive constant on every entry. In the coefficient
     * domain that is a plaintext whose coefficients are all equal, so it costs
     * no level -- which is why SoftMax's affine map is split here rather than
     * done in the slot domain where it would cost one.
     *
     * @param res output, `num_heads * T` rows of `T` keys
     * @param q the rotated queries, `num_channels` wide
     * @param k the rotated keys, `num_kv_channels` wide
     * @param scale multiplies every entry, and carries 1/sqrt(head_dim), the
     * SoftMax range, and both operands' crossing constants
     * @param shift added after scaling, one entry per row in the order
     * `head * T + query`; see `Calibration::softmax_shift`
     */
    virtual void Scores(std::vector<Ct> &res, const std::vector<Ct> &q,
                        const std::vector<Ct> &k, double scale,
                        const std::vector<double> &shift) const = 0;

    /**
     * @brief res = scale * (P V), per head, with GQA repetition.
     *
     * @param res output, `num_channels` wide
     * @param p the SoftMax probabilities
     * @param v the values, `num_kv_channels` wide
     * @param scale multiplies every entry
     */
    virtual void Values(std::vector<Ct> &res, const std::vector<Ct> &p,
                        const std::vector<Ct> &v, double scale) const = 0;
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

  LlamaBlock(std::shared_ptr<const BootContext<word>> boot, const Config &cfg,
             const Calibration &cal);

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
   * @param causal_mask 1 where key <= query and 0 elsewhere, in the score
   * packing, one vector per score ciphertext
   * @param leg the product implementation
   * @param evk_map every key the block needs
   */
  void Run(std::vector<Ct> &res, const std::vector<Ct> &x, const Weights &w,
           const std::vector<std::vector<Complex>> &causal_mask,
           const LinearLeg &leg, const EvkMap<word> &evk_map) const;

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

 private:
  // One turn's slot leg: bootstrap every ciphertext, canonicalise it onto the
  // operator's scale and magnitude, and hand back the result. `magnitude` is
  // what the crossing constant has to be undone by; `shift` is added after,
  // for the one operator whose argument is affine rather than linear.
  void Lift(std::vector<Ct> &res, const std::vector<Ct> &x, double magnitude,
            const EvkMap<word> &evk_map, double shift = 0.0) const;
  // The other half: StC every ciphertext back to the coefficient domain.
  void Lower(std::vector<Ct> &res, const std::vector<Ct> &x,
             const EvkMap<word> &evk_map) const;

  // Per-channel weights, broadcast into the block's packing, with sqrt(alpha)
  // folded in the way RmsNormHandler asks for.
  void SpreadNormWeight(std::vector<std::vector<Complex>> &res,
                        const std::vector<double> &w, double alpha) const;

  std::shared_ptr<const BootContext<word>> boot_;
  Config cfg_;
  Calibration cal_;
  int num_slots_;
  int channels_per_ct_;

  std::unique_ptr<SylphSchedule<word>> sched_;
  std::unique_ptr<RmsNormHandler<word>> attn_norm_;
  std::unique_ptr<RmsNormHandler<word>> ffn_norm_;
  std::unique_ptr<RoPeHandler<word>> rope_;
  std::unique_ptr<SoftMaxHandler<word>> softmax_;
  std::unique_ptr<SiLuHandler<word>> silu_;
};

}  // namespace cheddar
