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
    //! 0 derives SiLU's degree from its RANGE, which is the rule 1.5cw states
    //! and the tree owed here: a Chebyshev error is uniform in ABSOLUTE terms
    //! over its interval, so a range set by rows that are not the answer puts
    //! the whole of it on the rows that are. Layer 1 of Llama-3-8B is the
    //! model's massive-activation layer and its gate reaches 15.25 at the two
    //! sink rows against 3.71 at the 126 user rows, so `1.2 * gate_absmax` is
    //! 18.30 and degree 31 fits SiLU there to 1.69e-2 -- measured in float64,
    //! that ALONE puts the layer at 2^-2.97 against the encrypted run's
    //! 2^-3.06, i.e. it is the entire error and every ciphertext stage is
    //! invisible behind it. Narrowing the range instead is not available: the
    //! sink gate would sit outside the interval, where a degree-31 Chebyshev
    //! is cosh(31 arccosh(3.42)) ~ 1e25 and takes the ciphertext with it.
    int silu_degree = 0;
    //! 0 derives the invsqrt degree from the window, as a Chebyshev fit's
    //! uniform error requires: 9 up to 2.5, 15 up to 12, 31 beyond.
    int rms_degree = 0;
    //! Leave channel zero of the model dimension empty. See the class
    //! comment; false is the wrong contract and exists only to reproduce it.
    bool keep_component_zero = false;
    bool min_ks = false;
    bool verbose = false;
    //! Keep `RmsNormHandler`'s output in SLOTS, before `SlotToCoeff`, so a
    //! caller can read it. Diagnostic only: the coefficient read cannot say
    //! whether an error was made by the norm or by the conversion under it,
    //! and the two are separated by ten bits of the layer's budget. Costs
    //! `num_model_cts` ciphertexts of residency while it is on.
    bool keep_norm_slots = false;
  };

  using DeviceWeights = typename CoeffLinearLeg<word>::DeviceWeights;

  /**
   * @brief One projection's weight, in one of the leg's two forms.
   *
   * `host` is the matrix at the DECLARED widths, `[in * out_declared + out]`,
   * built by the caller from the model's tensor and the slot maps -- four
   * times the tensor and mostly zeros, converted on the host (a layer's whole
   * `pcmm: convert weights` row, 87 s). `device` is the tensor itself on the
   * GPU with the two slot maps, encoded by the leg where it is read
   * (`CoeffLinearLeg::DeviceWeights`). Exactly one is set; the leg's
   * `device_weights_test` shows the two project to the same words.
   */
  struct ProjectionWeight {
    const std::vector<double> *host = nullptr;
    const DeviceWeights *device = nullptr;
    bool Given() const { return (host != nullptr) != (device != nullptr); }
  };

  /** @brief One layer's plaintext weights. */
  struct Weights {
    //! `attn_channels x model_declared`.
    ProjectionWeight o;
    //! `model_declared x hidden_declared`.
    ProjectionWeight gate;
    ProjectionWeight up;
    //! `hidden_declared x model_declared`.
    ProjectionWeight down;
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

    //! THE SINK RESCALE, PER TOKEN, AT EACH OF THE TWO NORMS.
    //!
    //! [SYLPH] 3.1.1: a prefix of beginning-of-sequence tokens is
    //! prompt-independent, so its hidden state is a constant of the model and
    //! rescaling it is a PUBLIC per-token multiplier -- which is the only
    //! reason it can be done at all under encryption. It has to happen at
    //! EVERY norm, not once at the model's input: measured on the rotated
    //! model, layer 1's OUTPUT carries sink rows at mean square 26.72 against
    //! the user rows' 1.2e-04 to 6.2e-04, a ratio of 74327, and an invsqrt
    //! handed that grows like cosh(d arccosh(v)). Without it a 32-layer run
    //! dies at layer 1 with `carried 1693` against a stream scale of 0.019.
    //!
    //! One entry per token, 1.0 at the user tokens. Empty means no rescale,
    //! which is right only for a single layer whose input was rescaled by the
    //! caller. The factors ride the crossing's own constant multiply, so they
    //! cost NO level and no operation -- `Canonicalise` becomes a plaintext
    //! multiply instead of a constant one.
    std::vector<double> attn_sink, ffn_sink;

    //! THE FEED-FORWARD'S OWN SINK ROWS, suppressed by a public per-token
    //! factor. Layer 1 is the model's massive-activation layer: its
    //! `down` output is 194x bigger at the two sink rows than at the 126 user
    //! rows (`reference/whererows.py`), and no rescale of the layer's INPUT
    //! reaches that, because the layer makes it. The factor folds into the
    //! `up` crossing's restore multiply, which is a constant multiply already,
    //! so it costs nothing: `gu = SiLU(g) * (u * s)` and the gate -- the one
    //! operand SiLU makes non-linear use of -- is untouched, so `y` comes out
    //! row-scaled exactly.
    //!
    //! One entry per token, 1.0 at the user tokens. Empty means no
    //! suppression. A run that uses it must suppress the residual stream to
    //! match, or the two conventions meet at the second residual add.
    std::vector<double> up_sink;
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
  /// The same, from the tensor on the device.
  void Project(std::vector<Ct> &res, const std::vector<Ct> &x, int in_declared,
               int out_declared, const DeviceWeights &w, double w_scale,
               const char *tag) const;
  /// Either form.
  void Project(std::vector<Ct> &res, const std::vector<Ct> &x, int in_declared,
               int out_declared, const ProjectionWeight &w, double w_scale,
               const char *tag) const;

  /**
   * @brief The half-density slot conventions, stated once.
   *
   * A residual-stream ciphertext carries `rank/2 - 1` live model channels at
   * the even declared indices 2, 4, ..., rank-2 -- component zero has no
   * partner under the banded convention (Doing.md 1.5cu) -- and a hidden one
   * carries `rank/2` at 0, 2, ..., rank-2, because nothing reduces over the
   * hidden axis. `ModelSlot`/`HiddenSlot` are live -> declared;
   * `ModelMap`/`HiddenMap` are declared -> live with -1 at the dead indices,
   * which is what `DeviceWeights` takes. `CiSinCAttention` and `CiLlamaSeam`
   * address the same declared indices.
   */
  static int ModelSlot(int m, int rank) {
    const int per = rank / 2 - 1;
    return (m / per) * rank + 2 * (m % per + 1);
  }
  static int HiddenSlot(int j, int rank) {
    const int per = rank / 2;
    return (j / per) * rank + 2 * (j % per);
  }
  //! `live` is the tensor's own channel count (4096 model, 14336 hidden);
  //! declared slots past it are dead, as the ciphertexts' tails are.
  static void ModelMap(std::vector<int> &slot, int declared, int rank,
                       int live) {
    slot.assign(declared, -1);
    for (int m = 0; m < live; m++) slot[ModelSlot(m, rank)] = m;
  }
  static void HiddenMap(std::vector<int> &slot, int declared, int rank,
                        int live) {
    slot.assign(declared, -1);
    for (int j = 0; j < live; j++) slot[HiddenSlot(j, rank)] = j;
  }

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

  //! SiLU's degree from its range; see `Config::silu_degree`.
  int SiLuDegree(double range) const;

  //! HalfBoot into slots, canonicalise by the crossing, RMSNorm, and back to
  //! coefficients. The two RMSNorms of a layer differ only in their gains and
  //! their calibration, so they are one function.
  void NormTurn(std::vector<Ct> &res, const std::vector<Ct> &stream,
                const std::vector<double> &gain, double alpha, double window,
                double stream_scale, const std::vector<double> &sink,
                const EvkMap<word> &evk);

  //! Multiply by a constant at `GetSlotLevel()` and rescale, landing the
  //! result canonical one level below. Duplicate-preserving by construction:
  //! it is a uniform constant, not a mask.
  void Canonicalise(Ct &ct, double factor) const;

 public:
  //! The last `NormTurn`'s output in SLOTS, empty unless
  //! `Config::keep_norm_slots`.
  const std::vector<Ct> &GetNormSlots() const { return norm_slots_; }

  //! The last `NormTurn`'s channel sum of squares, in slots. Only meaningful
  //! under `Config::keep_norm_slots`.
  const Ct &GetNormAcc() const { return norm_acc_; }

 private:
  //! The same multiply with a per-TOKEN factor instead of a scalar, which is
  //! what carries the sink rescale for free. Duplicate-preserving, but NOT by
  //! being uniform across channels: the banded convention puts a channel's
  //! duplicate one token position BACK from its live copy, so the two bands
  //! need the same factor at different slot addresses. `at_scale` is the
  //! scale of the ciphertexts it will meet.
  Plaintext<word> CrossingPlaintext(double factor,
                                    const std::vector<double> &sink,
                                    double at_scale) const;
  void Canonicalise(Ct &ct, const Plaintext<word> &pt) const;

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
  //! `Config::keep_norm_slots`: the last norm's channel sum of squares, which
  //! is the half of the operator whose error is NOT a fit.
  mutable Ct norm_acc_;

  //! `Config::keep_norm_slots`: the last norm's output before the conversion.
  mutable std::vector<Ct> norm_slots_;

  //! Per-layer operator preparation; see `GetPrepareSeconds`.
  mutable double prepare_seconds_ = 0.0;
  //! The crossing's plaintext when it carries a sink rescale; built once per
  //! `NormTurn` off the first ciphertext's scale.
  Plaintext<word> crossing_pt_;
  //! The `up` crossing's, when `Calibration::up_sink` is in play.
  Plaintext<word> up_pt_;
};

}  // namespace cheddar
