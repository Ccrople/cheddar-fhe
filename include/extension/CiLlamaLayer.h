#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/CiSwitchedCcmm.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/CiLlamaSeam.h"
#include "extension/LlamaLinear.h"
#include "extension/RmsNorm.h"
#include "extension/SiLu.h"
#include "extension/SylphSchedule.h"

namespace cheddar {

/**
 * @brief Everything of a Llama-3 decoder layer on R+ that is not the attention
 * leg: the seam, the O projection, both residuals, RMSNorm, gate and up, SiLU
 * and the gate multiply, and the down projection.
 *
 * ## Where this sits
 *
 * `CiSinCAttention` is the leg -- half-images in at the HalfBoot landing,
 * attention output out in the CC-MM chain's layout. This is the other half,
 * and between them they are a layer. The split is not arbitrary: the leg runs
 * on a `BootContext` at slack ZERO because its softmax walk needs
 * `GetEndLevel()` at 16, and everything here runs on a second `BootContext`
 * over the same primes and the same secret at slack NINE, because
 * `SlotToCoeff` is compiled at `GetStCStartLevel()` and an operator eight
 * levels deep cannot reach it without slack (Doing.md 1.5ct). The two
 * Contexts are the caller's; this class holds only the second.
 *
 *     [caller]  X -> PC-MM emissions -> HalfBoot -> q/k/v half-images
 *     leg       Scores -> [caller] Boot x8 -> SoftMax -> Values -> 8 chain cts
 *     [caller]  Boot x8
 *     THIS      Seam         -> 16 half-density coefficient ciphertexts
 *     THIS      FeedForward  -> the next residual stream, at level 0
 *
 * ## What the caller owns, and what it does not
 *
 * The caller owns the Contexts, every key, and the CALIBRATION -- which is
 * what [SYLPH] section 3.1 says it should be: ranges fitted offline on the
 * clear model, not measured in the run. What this class will NOT do is
 * decrypt anything to find a constant. Two constants that earlier code fitted
 * in-run are derived here instead:
 *
 *  - **the crossing constant** is `BootContext::GetMessageRatio()`, exact
 *    (Doing.md 1.5dk), rather than a fit against a decrypted twin;
 *  - **kappa**, what a full turn through the coefficient domain carries, is
 *    `2^-log_message_ratio / crossing` -- `SylphSchedule::ToCoeff` undoes the
 *    crossing by the NOMINAL ratio, deliberately, so a one-way crossing is
 *    off by exactly that and SiLU is the only stage that cannot absorb it
 *    (1.5cv). Folding it in moved the FFN from 2^-8.21 to 2^-12.28.
 *
 * ## The contracts, all of which have cost a wrong layer
 *
 * - **Half density, with its duplicates.** Every slot-domain operator here is
 *   duplicate-preserving: the canonicalise is a uniform constant rather than
 *   a mask, `RmsNormHandler` reduces at `channel_stride = 2` so the two slot
 *   parities stay apart, and its weight carries the PARTNER channel's value
 *   at the duplicate slots (1.5cs).
 * - **Component zero has no partner.** `I -> rank - I` has exactly two fixed
 *   points, so a half-density ciphertext carries at most `rank/2 - 1` live
 *   channels. Everything that WRITES the model dimension -- the O projection
 *   and the down projection -- must leave channel zero empty, or the next
 *   RMSNorm normalises by a mean square that is short one channel and
 *   `ModDecomp`'s suffix recursion hands all of it to the next projection's
 *   live components (1.5cu). `Config::keep_component_zero` exists only to
 *   reproduce the old, wrong contract in a test.
 * - **The ride height has an optimum, not a ceiling.** EvalMod's own
 *   approximation is cubic, so the crossing's relative cost is `a * ride^2`
 *   with `a = -0.00258` and every doubling costs two bits, until the additive
 *   floor takes over below ~0.02 (1.5cv). 0.2 is the measured bottom of that
 *   U for the FFN.
 * - **A window follows a measured spread.** A Chebyshev fit's error is
 *   uniform over its interval, so a window wider than the data throws that
 *   ratio away and a window NARROWER than the data evaluates the polynomial
 *   where it was never fitted -- silently. Both are invisible end to end,
 *   which is why the caller supplies them from calibration.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiLlamaLayer {
 private:
  using Ct = Ciphertext<word>;

 public:
  struct Config {
    int num_tokens = 128;   //!< T; the small degree is 2T
    int proj_rank = 512;    //!< the module rank the banded convention uses
    //! Declared channels of the residual stream and of the FFN's inner
    //! dimension. Declared, not live: half density means a ciphertext carries
    //! `proj_rank / 2 - 1` live model channels and `proj_rank / 2` hidden
    //! ones, because nothing reduces over the hidden axis.
    int model_declared = 8704;
    int hidden_declared = 28672;
    //! The model's own channel count, which is NOT the declared one. RMSNorm
    //! divides by the width it is told and Llama divides by 4096, so feeding
    //! it 8704 puts the polynomial's argument at 0.47 instead of 1 and the
    //! bottom sixth of the data outside the fitted window -- 1.5cv's silent
    //! failure, measured at 2^-5.09. Both halves cancel exactly when `eps` and
    //! `alpha` are scaled by the ratio, which is what `FeedForward` does.
    int model_live = 4096;
    double eps = 1e-5;
    int product_level = 1;
    //! Input ciphertexts decomposed at once, on the direct route. Sixteen
    //! parents at rank 512 and five limbs is 10.7 GB standing at once; four
    //! is 2.7, paid for in ModPacks (1.5ct).
    int parents_per_tile = 4;
    //! How hot the message rides into HalfBoot. See the class comment: this
    //! is the bottom of a U, not a ceiling to sit under.
    double ride = 0.2;
    int silu_degree = 31;
    //! 0 derives the invsqrt degree from the window, as a Chebyshev fit's
    //! uniform error requires: 9 up to 2.5, 15 up to 12, 31 beyond.
    int rms_degree = 0;
    //! Leave channel zero of the model dimension empty. See the class
    //! comment; false is the wrong contract and exists only to reproduce it.
    bool keep_component_zero = false;
    bool min_ks = false;
    bool verbose = false;
  };

  /** @brief One layer's plaintext weights, at the DECLARED widths. */
  struct Weights {
    //! `attn_channels x model_declared`, indexed `[in * model_declared + out]`.
    const std::vector<double> *o = nullptr;
    //! `model_declared x hidden_declared`.
    const std::vector<double> *gate = nullptr;
    const std::vector<double> *up = nullptr;
    //! `hidden_declared x model_declared`.
    const std::vector<double> *down = nullptr;
    //! `model_declared` RMSNorm gains, at the declared channels.
    const std::vector<double> *ffn_norm = nullptr;
    //! Distinguishes this layer's converted weights in the projection leg's
    //! cache. MUST differ between layers: the cache is keyed by name and
    //! asserts the fingerprint, so a repeated name with different weights is
    //! caught rather than silently answered with the first layer's operands.
    std::string tag;
  };

  /** @brief What [SYLPH] 3.1 fits offline on the clear model. */
  struct Calibration {
    //! The factor the O projection's weight carries, chosen so the residual
    //! stream reaches `Config::ride` at the crossing.
    double res_scale = 1.0;
    //! RMSNorm's layer constant for the FFN's norm: `1 / geometric mean of
    //! the mean squares`, which puts the argument's geometric mean at 1 by
    //! construction.
    double alpha = 1.0;
    //! The invsqrt window. `(measured argument ratio) * margin`, margin 1.3.
    double norm_window = 2.0;
    //! The SAME two, for the PRE-ATTENTION norm. They are different numbers:
    //! that norm's input is the layer's own input and the FFN's is the
    //! post-attention residual, and a window fitted on one is not a window for
    //! the other -- too wide throws away the ratio, too narrow evaluates the
    //! polynomial where it was never fitted, silently.
    double attn_alpha = 1.0;
    double attn_norm_window = 2.0;
    //! SiLU's fitted range, `margin * max|gate|` with margin ~1.2.
    double silu_range = 1.0;
    //! The factor the gate and up weights carry, sizing their crossing.
    double gate_scale = 1.0;
    //! THE FACTOR THE RESIDUAL STREAM CARRIES, and the down projection's
    //! weight scale.
    //!
    //! RMSNorm is scale invariant, so everything downstream of it is in the
    //! MODEL's own units however the stream was scaled -- which means the
    //! feed-forward output comes back unscaled and cannot be added to a
    //! stream that is. The down projection's weight puts it back, for
    //! nothing, exactly as the O projection's does for the attention half.
    double stream_scale = 1.0;
  };

  /**
   * @param boot the FFN's BootContext -- same primes and secret as the leg's,
   *        slack nine
   * @param layout the chain layout the attention output arrives in
   * @param modpack_keys the projection's ModPack keys, on `boot`'s Context
   * @param cfg the shapes and the two dials
   */
  CiLlamaLayer(std::shared_ptr<const BootContext<word>> boot,
               const CiSwitchedCcmmLayout &layout,
               std::vector<const EvaluationKey<word> *> modpack_keys,
               const Config &cfg);

  CiLlamaLayer(const CiLlamaLayer &) = delete;
  CiLlamaLayer &operator=(const CiLlamaLayer &) = delete;

  /**
   * @brief Every rotation this class needs except the seam's per-half stages.
   *
   * RMSNorm's distances are read off a handler built here for the purpose:
   * they depend on the shape alone, while the handler's calibration does not
   * outlive one layer.
   */
  void AddRequiredRotations(EvkRequest &req) const;

  //! Build the seam's T1 for `half`; see `CiLlamaSeam`.
  void PrepareSeamHalf(int half);
  void AddSeamHalfRotations(EvkRequest &req) const;
  void DropSeamHalf();

  /**
   * @brief One booted chain-layout ciphertext to its banded coefficient image,
   * for the half `PrepareSeamHalf` last built.
   */
  void Seam(Ct &res, const Ct &booted, const EvkMap<word> &evk);

  /**
   * @brief The pre-attention RMSNorm: the residual stream crossed into slots,
   * normalised, and returned to coefficients for the Q/K/V product.
   *
   * The correctness-width layer test never had this -- it projected Q, K and V
   * straight off a synthetic stream -- and the full-width leg test did it on
   * the HOST. A layer that feeds its own successor cannot do either.
   *
   * @param res `model_declared / proj_rank` coefficient ciphertexts at
   *        `GetSchedule().GetCoeffLevel()`
   * @param stream the residual stream at level 0
   * @param gain the RMSNorm weights at DECLARED channels
   */
  /**
   * @brief One projection through the layer's own leg, DECLARED at both ends.
   *
   * The Q, K and V emissions are the same operator as O, gate, up and down --
   * a banded half-density coefficient image contracted against a weight whose
   * two axes are declared channels -- and running them through a raw
   * `PcmmHandler` instead costs FOUR TIMES the work: `Config::input_density`
   * skips the dead half of every parent, `output_density` skips the dead half
   * of every weight operand, and `parents_per_tile` stops repeating a ModPack
   * per output group. Measured at the model's width, 1.5dd: 228.2 ms per
   * output ciphertext at tile 4 with neither skip, 49.8 with both at tile 16.
   *
   * `w` is indexed `[in_declared][out_declared]` and the leg does the
   * bit reversal on both axes: module component `i` of parent `p` is declared
   * input `p * rank + rev(i)`, and module row `r` of group `g` is declared
   * output `g * rank + rev(r)`. So the caller states the map in the same
   * declared indices `ModelSlot` and the seam use, and never writes a
   * reversal of its own.
   *
   * @param res one ciphertext per group, at the level a `HalfBoot` takes
   * @param x the parents, at `Config::product_level`
   * @param tag the weight-cache name; a repeated tag with different weights is
   *        a wrong layer that still decrypts
   */
  void Project(std::vector<Ct> &res, const std::vector<Ct> &x, int in_declared,
               int out_declared, const std::vector<double> &w, double w_scale,
               const char *tag) const;

  void AttentionNorm(std::vector<Ct> &res, const std::vector<Ct> &stream,
                     const std::vector<double> &gain, const Calibration &c,
                     const EvkMap<word> &evk);

  /**
   * @brief The O projection, the residual, and the whole feed-forward network.
   *
   * @param res the next residual stream, `model_declared / proj_rank`
   *        half-density coefficient ciphertexts at level 0
   * @param h_cts the seam's `2 * num_cts` half-density images
   * @param stream the residual stream coming in, at level 0, already carrying
   *        whatever factor the O projection's output will carry
   * @param w this layer's weights
   * @param c this layer's calibration
   */
  void FeedForward(std::vector<Ct> &res, const std::vector<Ct> &h_cts,
                   const std::vector<Ct> &stream, const Weights &w,
                   const Calibration &c, const EvkMap<word> &evk);

  //! The crossing constant, derived from the BootParameter (1.5dk).
  double GetCrossing() const { return crossing_; }
  //! What a full turn through the coefficient domain carries.
  double GetKappa() const { return kappa_; }
  //! `SylphSchedule`'s levels, for a caller placing its own stages.
  const SylphSchedule<word> &GetSchedule() const { return sched_; }
  //! The projection leg, for a caller that wants its own `Project` calls.
  CoeffLinearLeg<word> &GetProjectionLeg() { return *leg_; }

  /**
   * @brief Host seconds spent PREPARING this layer's slot operators, which is
   * per-layer work and not arithmetic.
   *
   * `RmsNormHandler` compiles its polynomial at construction and encodes its
   * weight plaintexts on first use -- 1308.8 ms against 31.9 ms for the second
   * call (1.5dd) -- and a layer calls it twice, with two different weights, so
   * nothing inside a layer amortises it. Like the projection leg's model
   * conversion it belongs beside the weights, not in the online row.
   */
  /**
   * @brief The library's OWN compiled inverse square root, in double.
   *
   * `CiFfn.TheFitsAloneExplainTheFfnError` capped the feed-forward at 8.95
   * bits before any ciphertext existed by running it against `PlainInvSqrt`
   * and `PlainSiLu` (1.5cu). The same question for a layer's RMSNorm needs the
   * handler built at the LAYER's own level, window and degree, and a caller
   * that reconstructs those from the outside gets one of them wrong -- the
   * first attempt passed `max_level_ - 1` for the operator level and the
   * library answered `GetScale: Invalid level`. So the layer answers it.
   *
   * @param alpha the calibration `alpha`, unscaled -- the declared-width
   *        correction is applied here, exactly as `NormTurn` applies it
   * @param window the invsqrt window, which fixes the degree
   * @param mean_square one per token, over the LIVE channels
   */
  std::vector<double> PlainNormInvSqrt(
      double alpha, double window,
      const std::vector<double> &mean_square) const;

  double GetPrepareSeconds() const { return prepare_seconds_; }
  void ResetPrepareTimer() const { prepare_seconds_ = 0.0; }

 private:
  //! The RMSNorm weight plaintexts for one layer, duplicates included: at an
  //! ODD declared channel the weight carries the PARTNER channel's gain,
  //! because the duplicate band holds `comp_{rank-I}` and the reduction must
  //! see the same value there as at the live address (1.5cs).
  std::vector<std::vector<Complex>> NormWeights(const std::vector<double> &gain,
                                                double alpha) const;

  //! The invsqrt degree `Config::rms_degree` asks for, or the one the window
  //! implies when it is zero.
  int NormDegree(double window) const;

  //! HalfBoot into slots, canonicalise by the crossing, RMSNorm, and back to
  //! coefficients. The two RMSNorms of a layer differ only in their gains and
  //! their calibration, so they are one function.
  void NormTurn(std::vector<Ct> &res, const std::vector<Ct> &stream,
                const std::vector<double> &gain, double alpha, double window,
                double stream_scale, const EvkMap<word> &evk);

  //! Multiply by a constant at `GetSlotLevel()` and rescale, landing the
  //! result canonical one level below. Duplicate-preserving by construction:
  //! it is a uniform constant, not a mask.
  void Canonicalise(Ct &ct, double factor) const;

  std::shared_ptr<const BootContext<word>> boot_;
  Config cfg_;
  int num_slots_ = 0;
  int attn_channels_ = 0;
  int num_model_cts_ = 0;
  int num_hidden_cts_ = 0;
  double crossing_ = 0.0;
  double kappa_ = 1.0;
  int slot_level_ = 0;
  int op_level_ = 0;

  SylphSchedule<word> sched_;
  std::unique_ptr<CiLlamaSeam<word>> seam_;
  std::unique_ptr<CoeffLinearLeg<word>> leg_;
  //! Per-layer operator preparation; see `GetPrepareSeconds`.
  mutable double prepare_seconds_ = 0.0;
};

}  // namespace cheddar
