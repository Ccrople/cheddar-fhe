#pragma once

#include <memory>
#include <vector>

#include "core/CiSwitchedCcmm.h"
#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/EvalPoly.h"
#include "extension/EvalSpecialFFT.h"
#include "extension/LinearTransform.h"

namespace cheddar {

/**
 * @brief The attention leg on the conjugate-invariant ring: from the
 * HalfBoot-landed projection half-images to the attention output, at the
 * Llama alignment. The library form of the measured walk of Doing.md
 * 1.5bx-1.5cc; `CiBootSet.TheAttentionLegClosesEndToEnd` is its reference
 * implementation.
 *
 * ## Where this sits, against `SinCAttention`
 *
 * The ordinary-ring `SinCAttention` is "the two products, slots in, slots
 * out": its operands arrive in the block's slot packing and its exits ride
 * `HalfBoot` plus an StC prefix. On R+ neither end works that way. The
 * projections' images land where the banded ModPack physics puts them (the
 * half-density doorstep of 1.5by), the transport into the chain layout is
 * fused into this class's own multiplies, and the return to slots is the
 * nested converter's inverse followed by the caller's full `Boot` -- the
 * fused HalfBoot + CI-prefix exit does not exist yet (1.5bu). So this class
 * spans more of the leg: transport, both contractions, and the softmax
 * between them, with the bootstrap crossings left to the caller.
 *
 *     q/k/v half-images @ land_level          (16 half cts per tensor)
 *       --RoPE+restore+kill (Q,K) / kill (V)--> merge --> exchange
 *       --[K: cross] [V: call align]--> premapped descents @ forward_level
 *       --chain x2--> @ inverse_level --SinCToSlot--> scores @ 0
 *     caller: Boot each ciphertext to GetEndLevel()
 *       --SoftMax (optionally causal)--> P @ forward_level
 *       --nested descent + chain x2 + SinCToSlot--> output @ 0
 *
 * ## The contracts this class embodies
 *
 * - **The half-density doorstep (1.5by).** A projection half-image carries
 *   256 live components (heads of one family, channels of one group) and
 *   recomposes clean; the duplicate deposits sit at addresses the
 *   RoPE/restore masks already zero. `Scores`/`Values` consume the 2 x 8
 *   half ciphertexts per tensor exactly as the PC-MM emits them (1.5ca).
 * - **The transport canonicalises (1.5cb).** HalfBoot declares its output
 *   at `GetStCInputScale()` (~2^58); riding that through the chain is
 *   1.5bz's five-bit floor but breaks EvalMod at the Boot boundary. So
 *   `gamma = sqrt(GetScale(cross_level) / GetStCInputScale())` is folded
 *   into BOTH the RoPE/restore masks and the exchange's plaintexts --
 *   integers stay ~2^28, no single-fold 2^-12 cliff -- landing the operands
 *   canonical at `cross_level`.
 * - **The causal mask folds into the softmax walk (1.5cc).** One 0/1
 *   plaintext multiply on y = exp(...), its level paid by exp's fit
 *   (deg 9 -> 7); the affine's shift goes per-row, which closes the
 *   invsqrt interval to [1, live row sum] by construction. Causality is
 *   two per-ciphertext Encodes, carried by `SoftMaxCalibration::causal`.
 *
 * ## What the caller owns
 *
 * The projections (PC-MM emission + `HalfBoot`, 1.5ca), the `Boot` of each
 * scores ciphertext, every calibration number in `Config` and
 * `SoftMaxCalibration` (restore is 1.5bz's measured boundary constant;
 * span/shift/interval are read off calibration data as [SYLPH] reads its
 * off the real layer), and all key preparation. The one-time cost is the
 * constructor's three converter builds (~11 min on A100 at sub 32, host
 * side); one instance amortises over all layers.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiSinCAttention {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

 public:
  struct Config {
    int sub_degree = 32;      //!< k; fixes the Llama alignment below
    int land_level = 19;      //!< where HalfBoot lands the half-images
    int exchange_level = 18;  //!< the merge and the 63-diagonal exchange
    int cross_level = 17;     //!< K's cross and V's call alignment
    int forward_level = 3;    //!< the nested descents (SlotToSinC)
    int chain_level = 2;      //!< the chain multiplies here
    int inverse_level = 1;    //!< the return conversions (SinCToSlot)
    double rope_base = 10000.0;  //!< theta[m] = rope_base^(-2m / dim)
    //! 1 / the measured HalfBoot boundary constant (Doing.md 1.5bz); folded
    //! into the RoPE/restore masks. 1.0 leaves the images as HalfBoot
    //! scaled them.
    double restore = 1.0;
    bool verbose = false;
  };

  /** @brief What the softmax walk needs to know about the data. */
  struct SoftMaxCalibration {
    double m_eff = 8.0;  //!< effective span of the fitted exp
    double span = 1.0;   //!< calibrated score span (max - min)
    double shift = 0.0;  //!< calibrated score max (the affine's shift)
    //! The invsqrt interval, margins included. With `row_norm` folded
    //! (below) these are RATIO bounds around 1; without it, the interval
    //! of the row norms themselves.
    double norm_lo = 0.5;
    double norm_hi = 2.0;
    int exp_degree = 0;   //!< 0 = auto: 7 when causal, 9 when not (1.5cc)
    int inv_degree = 15;  //!< the walk has exactly four levels for it
    bool causal = false;
    //! Causal only: each (lane, row)'s calibrated live-key maximum, indexed
    //! [lane][row] with lane the LAYOUT lane (BitRev of the head). Masked
    //! slots fall back to `shift` so u stays inside the fit domain.
    std::vector<std::vector<double>> row_shift;
    //! Causal only, same indexing: each row's calibrated live-norm
    //! estimate, sum over live keys of exp(m_eff (S - row_shift) / span).
    //! When non-empty it folds into the mask as est^-1/2: y arrives
    //! pre-divided by sqrt(est), the Euclidean norm computes sq / est by
    //! itself, and est cancels IDENTICALLY in P = (y r)^2 -- the same
    //! row-invariance play as `row_shift`, absorbing the norm's magnitude
    //! instead of the max. The invsqrt interval then collapses to the
    //! actual / estimate ratio and `norm_lo` / `norm_hi` become ratio
    //! bounds (0.9 / 1.1 on calibration-quality data). Without it the
    //! interval is the live row sums themselves, and at 128 keys invsqrt
    //! deg 15 costs ~1.4e-2 on the row sums (measured) -- this fold is
    //! the fix, at zero levels and zero new mechanism.
    std::vector<std::vector<double>> row_norm;
  };

  /** @brief The keys one call needs, on three rings. */
  struct Keys {
    const EvkMap<word> *boot = nullptr;    //!< the big CI ring
    const EvkMap<word> *swtch = nullptr;   //!< the switching ring (descents)
    const EvkMap<word> *lifted = nullptr;  //!< the lifted ordinary ring
    const Evk *ring_switch = nullptr;
    const Evk *inverse_ring_switch = nullptr;
  };

  /**
   * @param boot the conjugate-invariant bootstrappable Context. Its
   *        `PrepareEvalMod` must have run: the canonicalising gamma reads
   *        `GetStCInputScale()` in this constructor.
   * @param switch_ctx the switching Context sharing `boot`'s bottom primes
   *        and secret (the _boot trio of Doing.md 1.5bt)
   * @param small_ctx the conjugate-invariant product ring
   * @param lifted_ctx the ordinary ring of the small ring's conductor
   */
  CiSinCAttention(std::shared_ptr<const BootContext<word>> boot,
                  ConstContextPtr<word> switch_ctx,
                  ConstContextPtr<word> small_ctx,
                  ConstContextPtr<word> lifted_ctx, const Config &cfg);

  // disable copying (or moving also)
  CiSinCAttention(const CiSinCAttention &) = delete;
  CiSinCAttention &operator=(const CiSinCAttention &) = delete;

  const CiSwitchedCcmmLayout &GetLayout() const { return ccmm_.GetLayout(); }
  /// Ciphertexts one operand occupies, and one result.
  int GetNumCiphertexts() const { return ccmm_.GetLayout().num_cts; }
  /// Where SoftMax expects its booted input.
  int GetTopLevel() const { return boot_->GetBootParameter().GetEndLevel(); }

  /// Rotations on the boot ring: the exchange and its window return, the
  /// merge, K's cross and V's alignment, and the softmax reduction tree
  /// (whose indices the bootstrap's own FFT keys already hold). The Boot's
  /// own rotations are the caller's business.
  void AddRequiredRotations(EvkRequest &req) const;
  /// Rotations on the switching ring: the three converters.
  void AddSwitchRotations(EvkRequest &req) const;
  /// Automorphism indices on the lifted ring.
  std::vector<int> LiftedRotationIndices() const {
    return ccmm_.LiftedRotationIndices();
  }

  /**
   * @brief `res = RoPE(Q) RoPE(K)^T`, one T x T per lane, from the
   * half-images to the level-0 scores in slots.
   *
   * @param res `num_cts` slot ciphertexts at level 0, message carrying the
   *        chain's factor: `res[i].GetScale() / base_scale` is the
   *        `carried` that `SoftMax` divides out (Doing.md 1.5bu)
   * @param q_a,q_b the queries' 8 + 8 half-images at `land_level`, heads
   *        0..15 / 16..31; CONSUMED (they are the walk's workspace)
   * @param k_a,k_b the keys' half-images, likewise consumed
   */
  void Scores(std::vector<Ct> &res, std::vector<Ct> &q_a,
              std::vector<Ct> &q_b, std::vector<Ct> &k_a,
              std::vector<Ct> &k_b, const Keys &keys) const;

  /**
   * @brief Compile the softmax walk for one calibration. Rebuilds the two
   * polynomial evaluators and, when causal, the per-ciphertext affine and
   * mask plaintexts. Must run before `SoftMax`; cheap next to the
   * constructor.
   */
  void PrepareSoftMax(const SoftMaxCalibration &calib);

  /**
   * @brief The softmax walk of 1.5bv/1.5cc on the booted scores: affine ->
   * exp -> [causal mask] -> Euclidean norm over the key axis -> invsqrt ->
   * P = (y r)^2, landing P at `forward_level`.
   *
   * @param P output, `num_cts` slot ciphertexts at `forward_level`
   * @param scores the booted scores at `GetTopLevel()`
   * @param carried the message factor at the Boot boundary: the scores'
   *        pre-Boot scale over the base scale, read off `Scores`' output
   */
  void SoftMax(std::vector<Ct> &P, const std::vector<Ct> &scores,
               double carried, const EvkMap<word> &evk) const;

  /**
   * @brief `res = P V`, from SoftMax's P and the values' half-images to the
   * attention output in slots at level 0.
   *
   * P's own ciphertexts are the two calls' lhs halves verbatim (1.5bw); V
   * rides Q's converter with a call mask and, for the odd call, one
   * rotation (1.5cb).
   *
   * @param res `num_cts` slot ciphertexts at level 0
   * @param p SoftMax's output at `forward_level`; read, not consumed
   * @param v_a,v_b the values' half-images at `land_level`; consumed
   */
  void Values(std::vector<Ct> &res, std::vector<Ct> &p, std::vector<Ct> &v_a,
              std::vector<Ct> &v_b, const Keys &keys) const;

 private:
  void BuildPremaps();
  void BuildTransportPlaintexts();
  // RoPE + restore + kill over one tensor's halves (with_angles), or the
  // plain restore + kill (V). In place, land_level -> land_level - 1.
  void RopeAndKill(std::vector<Ct> &a_cts, std::vector<Ct> &b_cts,
                   bool with_angles) const;
  // b's halves fold onto a's by one rotation: rev5(16 + v) = rev5(v) + 1.
  void Merge(std::vector<Ct> &a_cts, std::vector<Ct> &b_cts,
             const EvkMap<word> &evk) const;
  // The 63-diagonal token/head field swap, window convention (1.5by).
  void ExchangeAll(std::vector<Ct> &cts, const EvkMap<word> &evk) const;
  // K's cross: token top bits become the ciphertext index (1.5bx).
  std::vector<Ct> Cross(const std::vector<Ct> &k_cts, int call,
                        const EvkMap<word> &evk) const;
  // V's per-call alignment: mask the call bit, odd call shifts by 128.
  std::vector<Ct> VCall(const std::vector<Ct> &v_cts, int call,
                        const EvkMap<word> &evk) const;
  // LevelDown to forward_level, then the converter's forward.
  void Convert(const CiSinCConverter<word> &conv, std::vector<Ct> &cts,
               const EvkMap<word> &evk) const;
  // Two chain calls with per-call rhs, summed; then SinCToSlot to level 0.
  void ChainAndReturn(std::vector<Ct> &res, std::vector<Ct> &lhs_sinc,
                      const std::vector<Ct> &rhs_source, bool rhs_is_k,
                      const Keys &keys) const;

  std::shared_ptr<const BootContext<word>> boot_;
  ConstContextPtr<word> switch_ctx_;
  Config cfg_;
  int num_slots_ = 0;
  int degree_ = 0;
  int window_ = 0;
  double gamma_ = 1.0;

  CiSwitchedCcmmHandler<word> ccmm_;
  std::vector<int> pre_q_, pre_k_;
  std::unique_ptr<CiSinCConverter<word>> conv_q_, conv_k_, conv_pv_;
  std::vector<LinearTransform<word>> exchange_;  // zero or one entry

  // The transport's plaintexts: 4 x (cos, sin, -sin) shared by Q and K,
  // V's single restore mask, K's 8 selectors, V's 2 call selectors.
  std::vector<Pt> rope_cos_, rope_sin_, rope_neg_sin_;
  Pt kill_;
  std::vector<Pt> cross_sel_, call_sel_;

  // The compiled softmax walk.
  SoftMaxCalibration calib_;
  bool softmax_ready_ = false;
  int exp_in_ = 0, exp_out_ = 0, sq_level_ = 0, poly_in_ = 0;
  std::vector<std::unique_ptr<EvalPoly<word>>> polys_;  // [0] exp, [1] invsqrt
  std::vector<Pt> causal_a0_, causal_mask_;             // per ciphertext
  std::vector<int> reduce_dist_;
};

}  // namespace cheddar
