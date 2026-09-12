#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootContext.h"
#include "extension/CiBatch.h"
#include "extension/EvalPoly.h"

namespace cheddar {

#ifdef USE_CUBLAS

/**
 * @brief A BERT encoder layer on the BATCHED layout: one channel per
 * ciphertext, the ciphertext's slots `[token][instance]` (`CiBatchLayout`,
 * plain map, instance fastest), ONE ring, no ring switch.
 *
 * This is `CiBertTinyLayer`'s design at BERT-BASE's width (H 768, 12 heads of
 * 64, hidden 3072, 12 layers). The algorithm is the same operator for
 * operator; what the width changes is MEMORY, and there is exactly one
 * structural difference because of it:
 *
 *   **the feed-forward is tiled over the hidden axis.** At H = 768 a stream
 *   is 768 ciphertexts (7.5 GiB at the boot's landing on `ci16_35_k16_w58`,
 *   `word` = uint32_t); the 3072 intermediates would be 28 GiB at the level
 *   the GELU reads them at, and their int8 split another 22. So `up` is
 *   projected one tile of `Config::rows_per_tile` hidden channels at a time
 *   (one split of `h` serves every tile), the tile is GELU'd, and `down` is
 *   a SEPARATE operand per tile whose products are ACCUMULATED -- the same
 *   sum in a different order, exact. Peak: one tile, not the hidden width.
 *
 * Everything else -- the diagonal products, the Cho softmax and its fold, the
 * LayerNorms on the one variance ciphertext, the head -- is what BERT-Tiny
 * does, at 12 heads and 12 layers.
 *
 * ## The layer, operator by operator
 *
 *   projections   [KANG] Alg. 1 = `CiBatchProjection` (int8 GEMM over whole
 *                 ciphertexts, no key, one rescale). Biases are constants.
 *   Q K^T         the DIAGONAL product: `S_d = sum_c Q_c (.) rot_{dB}(K_c)`,
 *                 d = 0..T-1, one ciphertext per key offset d (slots = query
 *                 token x instance). A token shift inside every prompt is ONE
 *                 cyclic slot rotation by d * B because T * B = N. Baby-step
 *                 giant-step over d; T relinearizations a head.
 *   softmax       Cho: y = exp((S - shift)/2^k), k times y <- (y/|y|)^2.
 *                 The key axis is the ciphertext index, so a row sum is T
 *                 ciphertext adds and the inverse square root runs on ONE
 *                 ciphertext (the NARROW path: booted whenever it needs
 *                 levels, degree free). The shift is a public per-(head,
 *                 query position) plaintext; `SetMask` zeroes the pads.
 *   P V           the same diagonal product the other way round:
 *                 `O_c = sum_d P_d (.) rot_{dB}(V_c)`; output = the layout.
 *   LayerNorm     mean and variance are ciphertext sums (no rotation); the
 *                 centring is `H x_c - sum` (an integer multiply, no level);
 *                 the inverse square root on the ONE variance ciphertext --
 *                 which is the narrow path, so BERT-Base's wide L9/L10
 *                 variance windows cost a DEGREE and not a level; the gain
 *                 rides that ciphertext's H scalar copies, the bias is a
 *                 constant. One WIDE level.
 *   GELU          one Chebyshev polynomial per hidden ciphertext, evaluated
 *                 tile by tile (today one shared interval per layer).
 *
 * ## Units: every ciphertext carries a public factor
 *
 * `Stream::carry`: message = carry * model value. A bootstrap wants |message|
 * <= `Config::ride`, so the carry of what enters one is `ride / absmax`
 * with `absmax` from the calibration; projections fold `carry_out /
 * carry_in` into the weight, the softmax folds its affine into W_Q and the
 * shift, the norms into the variance affine. Nothing else about units is
 * anywhere.
 *
 * ## Levels (boot landing `top`, k Cho passes)
 *
 *   x booted -> top; q/k at l_S + 1, S_d at l_S = exp levels + (k == 1 ? 6
 *   : 2) + mask; exp -> y; sq (narrow, booted) -> r; w = y r; P = y_k at
 *   l_p >= 4; P V -> l_p - 1; O -> l_p - 2; residual; LN1 -> one below
 *   (wide: one level); boot -> top; up -> top - 1; GELU -> its landing;
 *   down; residual; LN2 -> one below.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiBertBaseLayer {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Const = Constant<word>;
  using Evk = EvaluationKey<word>;
  using Source = typename CiBatchProjection<word>::Source;

 public:
  struct Shape {
    int model = 768;
    int hidden = 3072;
    int heads = 12;
    int head_dim = 64;
    int tokens = 128;
    double eps = 1e-12;
  };

  struct Config {
    Shape shape;
    //! |message| a bootstrap input is scaled to (EvalMod's cubic: 2^-15 at
    //! 0.35).
    double ride = 0.35;
    //! Full boots per `BootBatch` call.
    int boot_group = 8;
    //! Baby steps of the diagonal products (must divide T); 0 = ~sqrt(T).
    int baby_steps = 0;
    //! Hidden channels per feed-forward tile (must divide `hidden`): the
    //! peak of the whole layer. 512 at BERT-Base's 3072 is six tiles.
    int rows_per_tile = 512;
    //! Use `Calibration::est` when the calibration carries one (A/B switch).
    bool fold = true;
    //! Rotate a ciphertext by several distances off ONE decomposition
    //! (`Context::MultKeyNoModDown` on a shared mod-up): the BSGS steps of
    //! the diagonal products. False = one `HRot` each, the A/B baseline.
    bool hoist = true;
    //! Evaluate the exp / GELU / tanh polynomials over a batch of
    //! ciphertexts (`EvalPoly::EvaluateBatch`); 1 = the per-ciphertext loop.
    int poly_batch = 1;
    bool verbose = false;
  };

  /** @brief A Chebyshev fit's domain and degree, as sim.py states it. */
  struct PolySpec {
    double lo = -1.0, hi = 1.0;
    int degree = 15;
  };

  /** @brief One layer's calibration (sim.py's calib.json), MODEL units. */
  struct Calibration {
    double in_absmax = 1.0;  //!< the stream entering the layer
    // softmax
    int niter = 1;                                //!< Cho passes k
    std::vector<std::vector<double>> shift;       //!< [heads][tokens]
    PolySpec exp;                                 //!< on (S - shift)/2^k
    std::vector<PolySpec> inv;                    //!< pass j's 1/sqrt window
    /**
     * @brief The FOLD: `[pass][head][token]`, a public estimate of the row's
     * sum of squares, empty (or an empty pass) = no fold for that pass.
     *
     * `1 / sqrt(sq) = (1 / sqrt(est)) (1 / sqrt(sq / est))`, so with a public
     * `est` the polynomial sees the RATIO and its window is what one row's
     * population spread is, not what the whole population's is. The estimate
     * that a served batch uses is
     *
     *     est[b][t] = `est[j][head][t]` * live_b ^ `est_live_pow[j]`
     *
     * with `live_b` the instance's real-token count, which the mask makes
     * public -- so the length dependence of `sq_0 = sum over real keys`
     * leaves the window exactly. `1 / est` rides the scaling that precedes
     * the narrow path's bootstrap (no level of its own where a boot happens)
     * and `1 / sqrt(est)` rides the constant on `r` (one narrow level).
     */
    std::vector<std::vector<std::vector<double>>> est;
    std::vector<int> est_live_pow;                //!< [pass], 0 when absent
    // norms
    struct Norm {
      PolySpec inv;          //!< 1/sqrt(var + eps) window
      double r_max = 1.0;    //!< 1/sqrt(lo + eps)
      double out_absmax = 1.0;
    } ln1, ln2;
    /**
     * @brief The GELU: ONE FIT PER FEED-FORWARD TILE, in the calibration's
     * own channel order (`gelu_perm`, a permutation of the hidden axis).
     *
     * A hidden channel is a WHOLE CIPHERTEXT in this layout, so a per-channel
     * interval costs nothing -- no mask, no band plaintext, no low-rank mode
     * plan: sort the channels by their own `|u|` and the tile IS the band.
     * That matters because BERT's outliers are dimension-wise: at BERT-Base
     * layer 10 one channel reaches `|u| = 130` where the 99th percentile is
     * 5.5, and a Chebyshev interpolant's error depends on `radius / degree`
     * alone (0.25 -> 5e-5, 0.125 -> 1e-12), so one shared interval charges
     * every channel for the worst one. The permutation is public: it is a
     * property of the weights and the population, not of a prompt, and it is
     * applied to `wint`'s columns and `wout`'s rows, which leaves the layer's
     * output exactly unchanged (a sum over the hidden axis).
     *
     * Each tile's `1 / a` also rides `wint`'s columns, so the polynomial sees
     * `[-1, 1]` as `EvalPoly` requires and the tiles cost nothing but their
     * own degrees.
     */
    std::vector<PolySpec> gelu;      //!< [ffn tile]
    std::vector<int> gelu_perm;      //!< [hidden], the order the tiles cut
    double h_pre_absmax = 1.0, z_pre_absmax = 1.0;
  };

  /** @brief One layer's tensors: device `[in][out]` f32, host vectors. */
  struct Weights {
    const float *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const float *wint = nullptr, *wout = nullptr;
    std::vector<double> bq, bk, bv, bo, bint, bout;
    std::vector<double> attn_norm, attn_norm_bias, ffn_norm, ffn_norm_bias;
  };

  /** @brief `model` (or `hidden`) ciphertexts and their public factor. */
  struct Stream {
    std::vector<Ct> cts;
    double carry = 1.0;
  };

  struct Stages {
    double boot = 0, qkv = 0, scores = 0, softmax = 0, values = 0, o = 0,
           ln = 0, ffn = 0, gelu = 0, total = 0;
    int wide_boots = 0, narrow_boots = 0, rotations = 0, relins = 0;
  };

  CiBertBaseLayer(std::shared_ptr<BootContext<word>> boot, const Config &cfg);
  CiBertBaseLayer(const CiBertBaseLayer &) = delete;
  CiBertBaseLayer &operator=(const CiBertBaseLayer &) = delete;

  const CiBatchLayout &GetLayout() const { return layout_; }
  //! The boot's rotations plus the diagonal products' BSGS set.
  void AddRequiredRotations(EvkRequest &req) const;
  //! The carry the layer's input is expected with (`ride / in_absmax`).
  double InputCarry() const;

  /**
   * @brief Convert the weights (GEMM operands at their levels, the biases
   * and folds), compile the polynomials, encode the shift plaintexts. Call
   * once per layer, after the keys exist; a repeated call REPLACES every
   * operand by name, so a 12-layer chain holds one layer's pieces.
   */
  void Prepare(const Weights &w, const Calibration &c);

  /**
   * @brief The whole layer: `out = LN2(h + FFN(h)), h = LN1(x + Attn(x))`.
   * `in` is CONSUMED (its ciphertexts are moved out): at this width a copy
   * of the residual stream is gigabytes, and a chain never reads it again.
   */
  void Layer(Stream &out, Stream &in, const EvkMap<word> &evk);

  // The halves, exposed for the tests.
  void Attention(Stream &attn_out, Stream &x, const EvkMap<word> &evk);
  void LayerNorm(Stream &out, const Stream &pre, const std::vector<double> &g,
                 const std::vector<double> &b,
                 const typename Calibration::Norm &n, const EvkMap<word> &evk,
                 const std::string &tag);
  void FeedForward(Stream &out, const Stream &h, const EvkMap<word> &evk);

  const Stages &GetStages() const { return stages_; }
  int TopLevel() const;
  //! Device bytes the prepared GEMM operands hold.
  size_t OperandBytes() const { return proj_->Bytes(); }

  /**
   * @brief The attention mask, `valid[b * T + t]` = 1 for a real token of
   * instance b (0 for [PAD]). Pads are projected, scored and exp'd like
   * every key -- so the calibration's exp domain covers them -- and their
   * `y` is zeroed by a per-slot plaintext before the row sum: one wide
   * level, which `Prepare` reserves, so call this BEFORE `Prepare`. Empty
   * = no mask (every token real).
   */
  void SetMask(const std::vector<uint8_t> &valid);

  /** @brief The pooler + classifier: `tanh(z[CLS] W + b) W_c + b_c`. */
  struct HeadWeights {
    const float *pool_w = nullptr;  //!< `[model][model]`
    const float *cls_w = nullptr;   //!< `[model][classes]`
    std::vector<double> pool_b, cls_b;
  };
  struct HeadCalibration {
    PolySpec tanh;  //!< the pooler input's CERTIFIED interval (the sphere)
  };
  //! `in_absmax`: the stream the head reads (the last layer's LN2 output).
  void PrepareHead(const HeadWeights &w, const HeadCalibration &c,
                   double in_absmax);
  /**
   * @brief `logits`: `classes` ciphertexts in model units; the answer for
   * instance b is at token 0 (the [CLS] slot). Every other token's slot is
   * the same head on that token -- inside the certified interval, so no
   * escape anywhere. `z` is booted in place.
   */
  void Head(std::vector<Ct> &logits, Stream &z, const EvkMap<word> &evk);

  /**
   * @brief A tap on every intermediate: `probe(name, cts, factor)` where
   * `decrypted / factor` is the model-unit quantity `bert_base/debug.py`
   * recomputes on the host. Costs nothing when unset.
   */
  using Probe = std::function<void(const std::string &, const std::vector<Ct> &,
                                   double)>;
  void SetProbe(Probe p) { probe_ = std::move(p); }

 private:
  void Tap(const std::string &name, const std::vector<Ct> &cts, double factor) const {
    if (probe_) probe_(name, cts, factor);
  }
  void Tap(const std::string &name, const Ct &ct, double factor) const;
  Probe probe_;
  int Level(const Ct &ct) const;
  //! Boot every ciphertext of `s` that sits below `need` (all land at top).
  void Lift(Stream &s, int need, const EvkMap<word> &evk);
  //! Boot ONE ciphertext (the narrow path) if below `need`, scaled so that
  //! its message is <= ride given |message| <= `absmax`; returns the factor
  //! the caller must multiply by to undo (1 when nothing was done).
  double NarrowLift(Ct &ct, int need, double absmax, const EvkMap<word> &evk);
  void BootMany(std::vector<Ct> &cts, const EvkMap<word> &evk);
  //! `res = a * value` at the constant's scale 1 (an integer, no rescale).
  void MultInt(Ct &res, const Ct &a, double value) const;
  //! `res = Rescale(a * value)`: one level.
  void MultScalar(Ct &res, const Ct &a, double value) const;
  void AddScalar(Ct &res, const Ct &a, double value) const;
  //! A per-query-token plaintext at the ciphertext's level and scale.
  void PerToken(Pt &pt, const std::vector<double> &per_token, const Ct &like) const;
  //! A per-slot plaintext, `values[b * T + t]`, at `like`'s level and scale.
  void PerSlot(Pt &pt, const std::vector<double> &values, const Ct &like) const;
  //! Does pass `j` carry a fold estimate?
  bool Folds(int j) const;
  //! `est[b][t]` for pass j, head h: the calibration's row estimate times
  //! the instance's public live count.
  void FoldEstimate(std::vector<double> &est, int j, int head) const;
  //! The two fold plaintexts of (pass, head), cached at the level and scale
  //! they are wanted at: `ride / (est unit hi)` before the narrow boot, and
  //! `tail / sqrt(est unit)` on `r` after the polynomial.
  const Pt &FoldScale(int j, int head, const Ct &like, double unit, double hi);
  const Pt &FoldUndo(int j, int head, const Ct &like, double unit, double tail);
  //! `cts <- poly(cts)` through `EvalPoly::EvaluateBatch` in groups of
  //! `Config::poly_batch` (1 = the per-ciphertext loop).
  void EvalMany(std::vector<Ct> &cts, const EvalPoly<word> &poly,
                const Evk &mult_key);
  //! `res[i] = rot(a, dists[i])` off ONE decomposition of `a` when
  //! `Config::hoist`, else one `HRot` each. `res` is resized.
  void RotateMany(std::vector<Ct> &res, const Ct &a,
                  const std::vector<int> &dists, const EvkMap<word> &evk);
  //! Chebyshev interpolant of `g` on [-1, 1] (the affine already applied),
  //! compiled at `level` for inputs at `in_scale`, landing canonical.
  template <typename F>
  std::unique_ptr<EvalPoly<word>> Compile(F g, int degree, int level,
                                          double in_scale,
                                          int *landing = nullptr) const;
  static int Levels(int degree);
  void Rotate(Ct &res, const Ct &a, int dist, const EvkMap<word> &evk);
  //! `res[d] = sum_i lhs[i] (.) rot(rhs[i], d B)` for d = 0..T-1 (BSGS).
  void DiagonalProducts(std::vector<Ct> &res, const std::vector<Ct> &lhs,
                        const std::vector<Ct> &rhs, const EvkMap<word> &evk);
  //! `res[i] = sum_d lhs[d] (.) rot(rhs[i], d B)` (the transpose: P V).
  void DiagonalContract(std::vector<Ct> &res, const std::vector<Ct> &lhs,
                        const std::vector<Ct> &rhs, const EvkMap<word> &evk);
  void SoftMax(std::vector<Ct> &P, std::vector<Ct> &S, int head,
               const EvkMap<word> &evk);
  //! The down projection's operand name for hidden tile `t`.
  static std::string DownName(int t) { return "down" + std::to_string(t); }

  std::shared_ptr<BootContext<word>> boot_;
  Config cfg_;
  CiBatchLayout layout_;
  std::unique_ptr<CiBatchProjection<word>> proj_;
  Calibration cal_;
  Weights w_;
  bool prepared_ = false;
  // levels, planned in Prepare
  int l_s_ = 0, l_qk_in_ = 0, l_p_ = 0, l_v_in_ = 0, l_o_in_ = 0;
  int exp_levels_ = 0, gelu_landing_ = 0;
  int baby_ = 0, giant_ = 0;
  int ffn_tiles_ = 0;
  // carries and folds
  double c_in_ = 1.0, c_h_ = 1.0;     //!< ride / in_absmax, ride / ln1 out
  double q_fold_ = 1.0;               //!< W_Q's factor 1/(sqrt(D) 2^k a_exp)
  std::vector<double> bq_, bk_, bv_, bo_, bint_, bout_;  //!< folded biases
  std::vector<Pt> shift_pt_;          //!< per head, at l_s
  std::unique_ptr<EvalPoly<word>> exp_poly_;
  //! one GELU per feed-forward tile, and the permuted (and per-tile scaled)
  //! feed-forward tensors the tiles are cut from
  std::vector<std::unique_ptr<EvalPoly<word>>> gelu_poly_;
  DeviceVector<float> wint_, wout_;
  DeviceVector<int> perm_d_;
  DeviceVector<float> gscale_d_;
  void FoldFeedForwardWeights(const Weights &w, const Calibration &c);
  // the mask: T per-slot 0/1 plaintexts `M_d[t][b] = valid[b][(t + d) % T]`,
  // encoded at the level and scale `y` has (rebuilt when those change)
  std::vector<uint8_t> mask_valid_;
  bool has_mask_ = false;
  std::vector<Pt> mask_pt_;
  int mask_level_ = -1;
  double mask_scale_ = 0.0;
  void MaskPlaintexts(const Ct &like);
  //! the fold, indexed `j * heads + head`; the level/scale each was built at
  std::vector<Pt> fold_a_, fold_b_;
  std::vector<int> fold_a_lv_, fold_b_lv_;
  std::vector<double> fold_a_sc_, fold_b_sc_;
  std::vector<double> live_;  //!< real tokens an instance (public), T if no mask
  // the head
  HeadWeights hw_;
  HeadCalibration hc_;
  std::vector<double> hpb_, hcb_;     //!< folded pooler / classifier biases
  std::unique_ptr<EvalPoly<word>> tanh_poly_;
  double c_head_ = 0.0;               //!< the carry `Head` expects
  int classes_ = 0;
  mutable Stages stages_;
};

#endif  // USE_CUBLAS

}  // namespace cheddar
