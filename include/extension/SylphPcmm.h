#pragma once

#include <memory>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/SylphPcmmMath.h"

namespace cheddar {

/**
 * @brief [SYLPH] section 4.2's depth-one PCMM on a ciphertext: Eq. (5).
 *
 * The algebra, lemma 2 and the reference implementation are in
 * `SylphPcmmMath.h`; this is the encrypted loop and nothing else:
 *
 *     tau^l(C) = sum_j rot_R^(j b) ( sum_i pt_{A,i,j,l} . rot_R^i(tau^(l+1) B) )
 *
 * One plaintext multiply deep, `(b - 1) + (g - 1)` rotations wide, and no
 * bootstrap -- which is the whole requirement section 4 states for PC-attention
 * ("the core objective is to avoid bootstrapping as much as possible, ideally
 * computing the PC-attention stage nearly bootstrapping-free").
 *
 * ## Two things that are not obvious from the equation
 *
 * **The rotations are hoisted.** Every inner term reads the SAME ciphertext at
 * a different `rot_R^i`, so the `b - 1` inner rotations are computed once and
 * shared by all `g` giant steps -- `Apply` builds the baby-step set first. That
 * is what makes the count `(b - 1) + (g - 1)` rather than `g (b - 1)`.
 *
 * **The plaintexts are rebuilt, not stored.** Section 4.2 is explicit that
 * keeping all `d` of them is "impractical for large-scale models due to the
 * massive memory overhead", and that each one costs only two plaintext
 * rotations from `tau^l . sigma(A)`. `Config::cache_plaintexts` chooses: on by
 * default because at `d = 128` the whole set is 128 plaintexts and setup is
 * where this project has always paid for its encodes, off when memory is the
 * binding constraint and the encode can ride the GPU encoder instead.
 *
 * ## Where the tau goes, and why SoftMax does not mind
 *
 * A chain of these carries one `tau` from end to end: PC-attention is
 * `PCMM_1 -> SoftMax -> PCMM_2`, and section 4.2 applies `tau^2` once, right
 * after RoPE and before the computation becomes wide, so that the two products
 * consume it one power at a time and the layout is correct again at the end.
 *
 * SoftMax sits between them and sees `tau(M)` rather than `M`. Section 4.3
 * observes that this costs nothing: `(tau(M)_{j,i})_j` is the `i`-th COLUMN of
 * `M` rotated by `i`, so a SoftMax over each column of `tau(M)` is the SoftMax
 * over each column of `M`, each rotated by the same `i` -- that is, exactly
 * `tau(SoftMax(M))`. The operator is unchanged; only the reduction axis has to
 * be the one this layout puts a column on. `SoftMaxSeesColumnsOfTau` in the
 * test states that as an identity rather than as a remark.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class SylphPcmm {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  ConstContextPtr<word> context_;
  sylph_pcmm::SylphPcmmPlan plan_;
  int input_level_ = 0;
  double input_scale_ = 0.0;
  int output_level_ = 0;
  bool cache_ = true;
  bool compiled_ = false;

  // pt_{A,i,j,l}, indexed `j * b + i` when cached.
  std::vector<Pt> pt_;

 public:
  /**
   * @param context the evaluation context
   * @param plan `sylph_pcmm::BuildPlan(A, d, l)`
   * @param input_level the level `tau^(l+1)(B)` arrives at
   * @param input_scale its scale
   * @param cache_plaintexts keep all `d` rearranged plaintexts (setup cost,
   *        no per-call encode) rather than rebuilding them per call
   */
  SylphPcmm(ConstContextPtr<word> context, sylph_pcmm::SylphPcmmPlan plan,
            int input_level, double input_scale, bool cache_plaintexts = true);

  SylphPcmm(const SylphPcmm &) = delete;
  SylphPcmm &operator=(const SylphPcmm &) = delete;

  /** @brief Encode what `Apply` reads. Must run first. */
  void Compile();

  /** @brief The level `tau^l(C)` lands at: one below the input. */
  int GetOutputLevel() const { return output_level_; }

  /** @brief Every `rot_R` distance Eq. (5) needs, all multiples of `d`. */
  std::vector<int> GetRotationDistances() const {
    return plan_.RotationDistances();
  }

  /** @brief Ciphertext rotations one call performs. */
  int GetNumRotations() const { return plan_.NumRotations(); }

  /**
   * @brief `res = tau^l(A B)` from `tau_b = tau^(l+1)(B)`, Eq. (5).
   *
   * `res` must not alias `tau_b`.
   */
  void Apply(Ct &res, const Ct &tau_b, const EvkMap<word> &evk) const;

  /** @brief Device bytes the compiled plaintexts hold, 0 before `Compile`. */
  size_t GetPlaintextBytes() const;
};

}  // namespace cheddar
