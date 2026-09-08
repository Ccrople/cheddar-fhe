#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/EvalPoly.h"

namespace cheddar {

/**
 * @brief BERT's GELU, `0.5 x (1 + erf(x / sqrt 2))`, as ONE low-degree
 * polynomial plus two public masks -- which is what makes its outliers cost
 * nothing.
 *
 * ## The problem, measured on the real model
 *
 * The intermediate projection's output does not fit an interval a Chebyshev
 * fit can afford. Measured over the twelve layers
 * (`reference_forward_bert.py`, priced by `gelufit.py`): the median |u| is
 * 1.7 to 2.1 and the 99th percentile 4 to 5, but the maximum reaches 130, and
 * it does so in **two columns of 3072** -- the dimension-wise outliers
 * [SYLPH] section 2 names, and the same phenomenon its orthogonal rotations
 * spread out. A single fit over the whole reach is hopeless at any affordable
 * degree: over +-160, degree 127 (seven levels) still leaves 5.6e-01 of
 * absolute error, and the layer's feed-forward comes out at rms 2.5e-01.
 *
 * Neither of [SYLPH]'s two offline remedies is available here:
 *
 *  - **prepending sink tokens** does not move them. Measured with 0, 2, 4, 8
 *    and 16 prepended `[PAD]`, `[SEP]`, `[MASK]` and `[UNK]` tokens, the
 *    user rows' maximum stays 130: BERT's outliers sit on ordinary content
 *    tokens, not on a fixed prefix, so there is no public prefix to absorb
 *    them (and a bidirectional encoder has no prompt-independent token whose
 *    Key-Value state could be precomputed and injected at all).
 *  - **orthogonal rotations** cannot be folded through a post-LN model: the
 *    LayerNorm gain sits ON the residual stream rather than in front of a
 *    projection, so `Q^T diag(g) Q` is not diagonal and there is nowhere to
 *    absorb it. And a rotation would not touch this anyway -- it is folded
 *    into the weights, so `u = h W` is unchanged by construction.
 *
 * ## What does work: GELU IS ALREADY EXACT OUTSIDE THE TRANSITION
 *
 * `GELU(u) - u` is below 1.3e-04 for `u > 4` and `|GELU(u)|` is below
 * 1.3e-04 for `u < -4`. So the outlier slots need no polynomial at all:
 *
 *     res = mask_bulk * P(u / R)  +  mask_pos * u
 *
 * with `P` fitted on `+-R = +-2 tau` and the masks public, from calibration.
 * The saturated slots are then EXACT, the fitted interval is 8 rather than
 * 160, and the degree collapses. Measured end to end on the feed-forward
 * output, worst layer of the twelve:
 *
 *     tau 4, R 8,  degree 31    rms 3.2e-04     five levels
 *     tau 6, R 12, degree 63    rms 6.4e-07     six levels
 *     one interval +-160, degree 127            rms 2.5e-01
 *
 * ## Which way this fails, and why that is the right way round
 *
 * A slot the calibration called SATURATED that arrives small is answered with
 * `u` instead of `GELU(u)`: an error of at most 0.17, falling as `u` grows.
 * A slot the calibration called BULK that arrives past `R` is evaluated by a
 * Chebyshev polynomial outside its interval, which grows like
 * `cosh(d arccosh(v))` and takes the ciphertext with it. So the margin goes
 * on the BULK side -- `R = 2 tau`, twice the threshold at which a slot would
 * have been called saturated -- and a doubtful slot is assigned to the
 * saturated group, never to the bulk. `Group::kind` states which is which.
 *
 * Everything `SiLu.h` says about a range being a parameter, about guessing
 * high not being the safe side, and about the argument's scale being part of
 * the argument holds here word for word, and is not repeated.
 *
 * ## THE MASK GOES IN FRONT OF THE POLYNOMIAL
 *
 * Not behind it. A saturated slot's input is `u / R` with |u| far past `R` --
 * that is what makes it saturated -- and a Chebyshev polynomial evaluated
 * there grows like cosh(d arccosh(v)): at |v| = 16 and degree 31 that is
 * 1e46, which overflows the modulus and destroys EVERY slot of the
 * ciphertext, the masked-out ones included. Measured, with the masks on the
 * output instead: the BULK slots came back at 6.3e+16.
 *
 * So the fitted group's mask multiplies the INPUT. The saturated slots become
 * zero, where the fit returns `GELU(0) = 0` -- which is also the right answer
 * for the negative saturated group, so it needs no term of its own -- and the
 * positive group is added back afterwards, its own multiply rescaling onto
 * the level the fit lands on.
 *
 * It costs one level. Llama's feed-forward spends that level on the gate
 * multiply `SiLU(g) * u`; BERT's has no gate, so the level the masks take is
 * the one SwiGLU was using. With a single group there is no mask and no
 * multiply at all.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class GeLuHandler {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

 public:
  /** @brief What a group's slots are answered with. */
  enum class Kind {
    kFit,       //!< the Chebyshev fit of GELU on [-range, range]
    kIdentity,  //!< `u` itself: exact where `u` is past the transition
    kZero,      //!< nothing: exact where `u` is below it
  };

  /** @brief One slot group: how it is answered, and over what interval. */
  struct Group {
    Kind kind = Kind::kFit;
    //! `kFit` only: the fitted half-interval, in the units of `u`. The
    //! caller hands the input divided by THIS, so every group of kind
    //! `kFit` in one handler must state the same range (there is one
    //! ciphertext and one division).
    double range = 1.0;
    int degree = 31;
  };

  /**
   * @param context the evaluation context
   * @param groups one per polynomial, in the order the masks are given
   * @param input_level level of the input ciphertexts
   */
  GeLuHandler(ConstContextPtr<word> context, const std::vector<Group> &groups,
              int input_level);

  GeLuHandler(const GeLuHandler &) = delete;
  GeLuHandler &operator=(const GeLuHandler &) = delete;

  int GetNumGroups() const { return static_cast<int>(groups_.size()); }
  double GetRange(int g) const { return groups_[g].range; }
  //! The range the caller must divide by: the one fitted group's.
  double GetRange() const;
  //! The level `Apply` leaves its output at.
  int GetOutputLevel() const { return out_level_; }

  /** @brief What group `g` answers in the clear, on `u`. */
  double PlainGeLu(int g, double u) const;

  /**
   * @brief Encode the group masks up front.
   *
   * @param mask one slot vector per group, `1` at the slots that belong to
   *        the group and `0` elsewhere. The assignment is per SLOT --
   *        (token, hidden channel) -- because that is where the outliers are:
   *        two columns of 3072, across many rows. Ignored when there is one
   *        group.
   */
  void Prepare(const std::vector<std::vector<Complex>> &mask) const;

  /** @brief Device bytes the cached masks hold. */
  size_t GetPlaintextBytes() const;

  /**
   * @brief `res = GELU(u)`, given `v = u / range` per token.
   *
   * @param res output at `GetOutputLevel()`
   * @param normalised_u the input divided by `GetRange()`
   * @param mask the group masks; empty when there is one group
   * @param evk_map supplies the multiplication key
   */
  void Apply(Ct &res, const Ct &normalised_u,
             const std::vector<std::vector<Complex>> &mask,
             const EvkMap<word> &evk_map) const;

 private:
  ConstContextPtr<word> context_;
  std::vector<Group> groups_;
  int input_level_;
  int poly_out_level_ = 0;
  int out_level_ = 0;
  //! One entry per group; null for `kIdentity` and `kZero`.
  std::vector<std::unique_ptr<EvalPoly<word>>> polys_;
  mutable std::vector<std::vector<Complex>> cached_mask_;
  mutable std::vector<Pt> mask_pt_;
  mutable int cached_mask_level_ = -1;
  bool multi_ = false;
};

}  // namespace cheddar
