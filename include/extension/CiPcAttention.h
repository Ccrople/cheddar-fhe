#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/SubringMatrix.h"
#include "extension/BootContext.h"
#include "extension/CiBatch.h"
#include "extension/EvalPoly.h"

namespace cheddar {

/**
 * @brief [SYLPH] section 4's PUBLIC-CONTEXT attention on the batched layout:
 * the encrypted query tokens against a public KV cache that differs per
 * instance, as [KANG] Algorithm 1 twice with one exp between.
 *
 * This is what the plain slot map was for. Everything before it -- the
 * converter premaps, the compact subring store, the device encode -- was a
 * part; this is the thing that consumes them.
 *
 * ## The shape, and why it is Algorithm 1 and not a matrix product
 *
 * Sylph's heterogeneous prefill has a 4096-token context of which ~128 are
 * encrypted and the rest are public. A public key token's contribution to a
 * score is
 *
 *     S[p][t][b] = sum_c  Kpub[b][p][c] * Q[c][t][b]
 *
 * -- `t` the encrypted query token, `b` the instance, `c` the channel. The
 * PLAINTEXT operand `Kpub[b][p][c]` is constant over `t` and varies over `b`,
 * and under the plain map `Slot(t, b) = t * B + b` a vector like that is
 * periodic with period B in the slot index. `SubringMatrixHandler` is exactly
 * that operand's handler: B = 512 is the sub-degree, the 512 lanes ARE the
 * instances in order (PcPremap's TheSubringLaneIsThePlainMapsInstance,
 * measured), the 128 blocks are the query tokens, and the product above is
 * `res[l] = sum_j u[j][l] * ct[j]` with `j` = channel and `l` = public token.
 *
 * So the whole linear half is **depth 1, no rotation, no automorphism key, no
 * relinearization key, no encoding conversion**. Under the chain addressing
 * the same operand is dense in the slot index and the encode alone is 4.06M
 * full-degree plaintexts a layer; that is the entire reason `Config::
 * plain_map` exists.
 *
 * The value half is the same call with the indices exchanged:
 *
 *     acc[c][t][b] = sum_p  w[p][t][b] * Vpub[b][p][c]
 *
 * `j` = public token, `l` = channel. Both directions contract the CIPHERTEXT
 * index and never touch the block index, which is why neither needs the
 * Vec dimension to be anything in particular.
 *
 * ## Streaming, and what stays alive
 *
 * The public context is long and its KV is per instance, so it cannot be
 * resident: at B = 512, head_dim = 128 and 3968 public tokens one head's K is
 * 260M reals. It does not have to be. The two products above are a sum over
 * `p`, so the context enters in CHUNKS and what survives a chunk is
 *
 *     sq       one ciphertext          sum_p w[p]
 *     acc      head_dim ciphertexts    sum_p w[p] Vpub[p][.]
 *
 * independent of how many public tokens there are. The live set is
 * `head_dim + 1 + chunk` ciphertexts and the chunk is a knob. `Head` drives
 * that loop against a `ChunkSource`; `Scores`/`Weights`/`Accumulate` are the
 * steps, exposed because a caller with its own schedule (a prefetch, a
 * different chunking, a host cache) should not have to take this one.
 *
 * ## The softmax, and how it joins the encrypted half
 *
 * `CiBatchAttention::SoftMax` normalises by the EUCLIDEAN norm over the key
 * axis (Cho, k = 1): `y_l = exp(m_eff (u_l - 1) / 4) (.) mask`,
 * `r = invsqrt(sum_l y_l^2)`, `P_l = (y_l r)^2`. The quantity that accumulates
 * is therefore `y^2`, and this class computes it DIRECTLY -- `w = y^2` is
 * `exp` at twice the argument, so the square is free and there is no
 * relinearization anywhere in the public branch.
 *
 * What comes back is unnormalised on purpose:
 *
 *     sq_pub  = sum_{p public}   w[p]
 *     acc_pub = sum_{p public}   w[p] Vpub[p][.]
 *
 * and the joint softmax over all 4096 keys is then
 *
 *     r = invsqrt(sq_pub + sq_enc),   out = r^2 (acc_pub + acc_enc)
 *
 * -- two ADDS, because both halves are in the plain map. That is the second
 * thing the plain map buys and it is easy to miss: with `fused_scores` the
 * scores come back chain-addressed (the tower's lane prefix cannot carry a
 * block premap, 63 -> 15309) and the join costs 32 standalone 243-diagonal
 * permutations a layer instead. `Config::plain_map` rejects `fused_scores`
 * for that reason, and this is the other side of that trade.
 *
 * The affine's MULTIPLY rides the key weights (`a1 = 2 / (span * carried)` is
 * a constant, so it is folded into `Kpub` at encode time and costs nothing);
 * its per-query-token ADD is one plaintext add at no level. The mask fold
 * that `BuildMasks` puts on every key -- `sqrt(gamma / aff_a / row_norm)` --
 * is per query token here, not per key, so it is applied ONCE at the end to
 * `sq` and `acc` rather than to every public token: 1 + head_dim plaintext
 * multiplies a head against `ptok` of them. It cancels in `P = (y r)^2` and
 * exists only to put the inverse square root's argument in its interval.
 *
 * ## Levels
 *
 *     q @ q_level  --Algorithm 1-->  scores one below
 *       --affine (add only)-->  --exp, deg d-->  ceil(log2(d+1)) levels
 *       --Algorithm 1-->  acc one below  --row fold-->  one below
 *
 * At `q_level` 16 and a degree-15 exp that is 16 -> 15 -> 11 -> 10 -> 9: the
 * public branch needs NO bootstrap of its own, which is the other half of
 * Sylph's "89 % of PC-attention is linear".
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiPcAttention {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

 public:
  struct Config {
    //! T, the ENCRYPTED query tokens -- the subring's blocks.
    int num_tokens = 128;
    //! B, the batch -- the subring's sub-degree and its lanes.
    int num_instances = 512;
    int head_dim = 128;
    //! Public key tokens an Algorithm 1 call. Bounds the weight store
    //! (`head_dim * chunk` subring entries) and the ciphertexts alive
    //! between the two products; nothing else reads it.
    int chunk = 64;
    //! The level the queries arrive at. The scores land one below.
    int q_level = 16;
    bool verbose = false;
  };

  /**
   * @brief What the exp needs to know about the data, in the same units as
   * `CiBatchAttention::SoftMaxCalibration` so that the two halves of the
   * softmax can be added.
   */
  struct Calibration {
    double m_eff = 8.0;   //!< the fitted exp's span
    double span = 1.0;    //!< the calibrated score span, chain units
    //! The factor the Q and K weights carried (`cq * ck`); it divides out in
    //! the affine, exactly as it does in `CiBatchAttention::SoftMax`.
    double carried = 1.0;
    //! Per QUERY token: the live-key maximum in chain units, exactly
    //! `SoftMaxCalibration::row_shift[head]`. The affine's add is
    //! `1 - 2 * row_shift[t] / span`, as `PrepareSoftMax` builds it. Empty =
    //! the same `shift` every token.
    std::vector<double> row_shift;
    double shift = 0.0;
    //! Per QUERY token: the SQUARE of the fold `BuildMasks` puts on every
    //! key of the encrypted branch (`gamma / aff_a / row_norm[t]`), applied
    //! once at the end here. Empty = 1.
    std::vector<double> row_fold;
    int exp_degree = 0;  //!< 0 = derive from `m_eff`
    /**
     * @brief The encrypted branch's Cho [25] iteration count
     * (`CiBatchAttention::SoftMaxCalibration::niter`). 0 is the single-shot
     * softmax this class was first written against; `k > 0` is what the
     * heterogeneous B = 512 layer actually runs, and it changes what the
     * public half has to hand over.
     *
     * ## Why a joint softmax with the iteration is not "two adds"
     *
     * `SoftMaxCho` walks `y <- (y r)^2` k times, so every iteration's
     * denominator is a sum over ALL 4096 keys and the public keys' `y` would
     * have to survive between iterations -- 3968 ciphertexts. They do not
     * have to. Unrolling the walk,
     *
     *     y^(j)_l = (y^(0)_l)^(2^j) R_j ,  R_j = prod_{i<j} (r^(i))^(2^(j-i))
     *
     * and `R_j` carries no key index -- it is one number a QUERY token. So it
     * comes out of the sum:
     *
     *     sq^(j) = sum_l (y^(j)_l)^2 = R_j^2 sum_l (y^(0)_l)^(2^(j+1))
     *
     * and all the public half owes the iteration is the POWER SUMS
     * `M_m = sum_p (y^(0)_p)^m` at `m = 2, 4, .., 2^k`, plus the value
     * accumulator at the top power. Those are k exponentials of the same
     * score, so they are computed in ONE streaming pass: the linear halves
     * (`Scores`, the value product) are shared and only the exp repeats.
     * `GetNumPowers()` is k, `Head` returns them in `pow`, and `acc` is
     * against `pow`'s LAST weight.
     */
    int niter = 0;
  };

  /**
   * @param boot the batched layer's ring (`ci16_35`), for the exp and the
   *        plaintexts
   */
  CiPcAttention(std::shared_ptr<const BootContext<word>> boot,
                const Config &cfg);

  CiPcAttention(const CiPcAttention &) = delete;
  CiPcAttention &operator=(const CiPcAttention &) = delete;

  //! The plain-map layout the queries and the results are packed in.
  const CiBatchLayout &GetLayout() const { return layout_; }
  //! How many power sums `Head` returns: `max(niter, 1)`. `pow[j]` is
  //! `sum_p (y0_p)^(2^(j+1))` and the value accumulator rides `pow`'s last
  //! weight, `(y0_p)^(2^GetNumPowers())`.
  int GetNumPowers() const { return calib_.niter > 0 ? calib_.niter : 1; }
  //! Where the exp lands: `q_level - 1 - ceil(log2(deg + 1))`.
  int GetWeightLevel() const { return exp_out_; }
  //! Where `Head` leaves `sq` and `acc`. The value product is one below the
  //! exp; the row fold, when the calibration has one, is one below that, and
  //! `sq` is brought down to meet `acc` either way.
  int GetOutputLevel() const { return has_fold_ ? exp_out_ - 2 : exp_out_ - 1; }

  /** @brief Compile the exp and encode the per-query-token plaintexts. */
  void Prepare(const Calibration &calib);

  /**
   * @brief One chunk of public KEYS, encoded as Algorithm 1 weights.
   *
   * `cols_in = head_dim`, `cols_out = width`, the affine's multiply folded
   * in. The input is the lane-major, entry-major array
   * `EncodeWeightsReal` takes:
   *
   *     k[(c * width + p) * num_instances + b] = Kpub[b][start + p][c]
   *
   * so a caller building it from a `[instance][token][channel]` cache walks
   * its own memory in whatever order suits it and this never sees the cache.
   */
  void EncodeKeys(SubringWeights<word> &res, const std::vector<double> &k,
                  int width) const;

  /**
   * @brief One chunk of public VALUES: `cols_in = width`, `cols_out =
   * head_dim`, no fold.
   *
   *     v[(p * head_dim + c) * num_instances + b] = Vpub[b][start + p][c]
   */
  void EncodeValues(SubringWeights<word> &res, const std::vector<double> &v,
                    int width) const;

  /**
   * @brief One chunk's scores: `res[p] = sum_c keys[c][p] q[c]`, `width`
   * ciphertexts at `q_level - 1`, the affine's multiply already in them.
   */
  void Scores(std::vector<Ct> &res, const std::vector<Ct> &q,
              const SubringWeights<word> &keys) const;

  /**
   * @brief One chunk's softmax weights: the affine's add, then
   * `w = exp(m_eff (u - 1) / 2)` -- which is the encrypted branch's
   * `y = exp(m_eff (u - 1) / 4)` SQUARED, the quantity that accumulates.
   * `scores` is consumed.
   */
  void Weights(std::vector<std::vector<Ct>> &w, std::vector<Ct> &scores,
               const EvkMap<word> &evk) const;

  /**
   * @brief Fold one chunk in: `sq += sum_p w[p]` and
   * `acc[c] += sum_p w[p] Vpub[p][c]`. `acc` and `sq` are grown on the first
   * call and read-modify-written after; `w` is consumed.
   */
  void Accumulate(std::vector<Ct> &acc, std::vector<Ct> &pow,
                  std::vector<std::vector<Ct>> &w,
                  const SubringWeights<word> &values) const;

  /**
   * @brief Close a head: the per-query-token fold onto `sq` and every
   * channel of `acc`, landing both at `GetOutputLevel()`.
   */
  void Finish(std::vector<Ct> &acc, std::vector<Ct> &pow) const;

  /**
   * @brief The public half of the OUTPUT: `res[c] += scale * acc[c]`.
   *
   * `Values` gives the encrypted half `sum_l P_l Venc[l]` with `P = y_k`
   * already carrying the walk's per-query-token factor. The public half was
   * accumulated against the LAST power alone, `sum_p (y0_p)^(2^k) Vpub[p]`,
   * so what it still owes is `R_k` -- and that is exactly what
   * `CiBatchAttention::SoftMaxCho` hands back through `pub_scale`.
   *
   * One ciphertext multiply a CHANNEL (128 a head, 4096 a layer), which is
   * the whole cost of joining the two outputs; the denominators joined
   * earlier, inside the walk.
   *
   * `acc` is consumed. `res` is grown on the first call and read-modify-
   * written after, so a caller can hand it `Values`' own output. The scales
   * must already agree -- the encrypted output carries the chain's factor in
   * its RECORDED scale, so a caller that wants to add to it declares the
   * public value weights at the same ratio when it encodes them.
   */
  void JoinOutput(std::vector<Ct> &res, std::vector<Ct> &acc,
                  const Ct &scale, const EvkMap<word> &evk) const;

  /**
   * @brief Fill one chunk's key and value arrays for public tokens
   * `[start, start + width)`, in the layouts `EncodeKeys` / `EncodeValues`
   * document. Both vectors arrive already sized.
   */
  using ChunkSource = std::function<void(int start, int width,
                                         std::vector<double> &k,
                                         std::vector<double> &v)>;

  /**
   * @brief One head against the whole public context, streamed.
   *
   * `pub_tokens` may be any size; the loop holds one chunk's weights at a
   * time and the ciphertexts that survive it are `sq` and `acc`, so the
   * residency does not move with the context length. That claim is the point
   * of the interface and PcAttentionTest measures it.
   */
  void Head(std::vector<Ct> &acc, std::vector<Ct> &pow,
            const std::vector<Ct> &q, int pub_tokens, const ChunkSource &src,
            const EvkMap<word> &evk) const;

  //! The single-power form, for a caller whose encrypted branch does not
  //! iterate (`niter == 0`). Asserts that.
  void Head(std::vector<Ct> &acc, Ct &sq, const std::vector<Ct> &q,
            int pub_tokens, const ChunkSource &src,
            const EvkMap<word> &evk) const;

  /**
   * @brief A GQA GROUP against the whole public context: the group's query
   * heads share ONE encode of the public keys and values.
   *
   * `Kpub` and `Vpub` are per KV head and Llama-3-8B is GQA 32/8, so one
   * encoded pair serves FOUR query heads while the two products are per
   * query head. `Head` encodes inside its own chunk loop, so calling it four
   * times encodes the same numbers four times -- and the encode is half of
   * what a public token costs (B200, `ci16_35`, B = 512, chunk 128: 3.578 ms
   * a token to encode, 3.286 ms a token a head to multiply). Over Sylph's
   * 3968 public tokens that is 871.5 s a layer against 530.8, measured by
   * `PcAttention.TheContextPriceSplitsByHeadCount`.
   *
   * The trade is that the two stores now outlive the exp -- `Head` could
   * drop the key store before it, because one head has no second use for it.
   * Live demand at chunk 128 is 6.5 GiB, which is nothing on a card with 183.
   *
   * `acc` and `sq` are one a query head in `q`'s order and are cleared here.
   * `Head` is this with a group of one, so the two paths are the same code
   * and `TheGroupShareIsTheSeparateHeadsWordForWord` says the sharing did
   * not change a word.
   */
  void HeadGroup(const std::vector<std::vector<Ct> *> &acc,
                 const std::vector<std::vector<Ct> *> &pow,
                 const std::vector<const std::vector<Ct> *> &q,
                 int pub_tokens, const ChunkSource &src,
                 const EvkMap<word> &evk) const;

 private:
  std::shared_ptr<const BootContext<word>> boot_;
  Config cfg_;
  CiBatchLayout layout_;
  SubringMatrixHandler<word> subring_;
  Calibration calib_;
  bool ready_ = false;
  int exp_out_ = 0;
  //! The affine's multiply, folded into every key weight at encode time.
  double a1_ = 1.0;
  //! One a power: `exps_[j]` is `exp(2^(j+1) hb (v - 1))`, `hb` the
  //! encrypted branch's own `y0` exponent.
  std::vector<std::unique_ptr<EvalPoly<word>>> exps_;
  //! Where each power's polynomial actually lands. A smaller argument has
  //! smaller high coefficients, `EvalPoly` trims the ones that vanish, and
  //! the low powers can therefore come out a level higher; `Weights` brings
  //! them down to `exp_out_`, the deepest.
  std::vector<int> exp_level_;
  //! The affine's per-query-token add, at `q_level - 1`.
  Pt a0_;
  //! The mask fold, per query token, at `GetWeightLevel() - 1`.
  Pt fold_;
  bool has_fold_ = false;
};

}  // namespace cheddar
