#include "extension/SlimPoly.h"

#include <algorithm>
#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
SlimPolyHandler<word>::SlimPolyHandler(ConstContextPtr<word> context,
                                       slim::SlimPlan plan, int block_slots,
                                       int input_level, double input_scale,
                                       double output_scale)
    : context_{std::move(context)},
      plan_{std::move(plan)},
      block_slots_{block_slots},
      input_level_{input_level},
      input_scale_{input_scale},
      output_scale_{output_scale} {
  AssertTrue(plan_.ok, "SlimPoly: the plan is not usable: " + plan_.why);
  AssertTrue(block_slots_ > 0 && (block_slots_ & (block_slots_ - 1)) == 0,
  // MaxNumSlots, not `degree_`: the CI ring has `degree_` real slots and an
  // ordinary one has `degree_ / 2` complex ones, and a message encoded at the
  // wrong length is not an error the encoder reports.
             "SlimPoly: block_slots must be a power of two");
  const int slots = context_->param_.MaxNumSlots();
  AssertTrue(block_slots_ * plan_.NumBlocks() <= slots,
             "SlimPoly: the tree does not fit -- [SYLPH] section 3.4 needs "
             "2^(t+j) <= N, and the input's own slots are 2^t");

  // Levels, and where they go: `k - j` for the leaf's Chebyshev basis, one for
  // the leaf's plaintext multiply, and `j` for the fold. Theorem 1's `k + 1`.
  const int leaf_degree = 1 << (plan_.k - plan_.j);
  const int basis_levels = plan_.k - plan_.j;
  output_level_ = input_level_ - basis_levels - 1 - plan_.j;
  AssertTrue(output_level_ >= 0,
             "SlimPoly: Algorithm 1 needs " +
                 std::to_string(plan_.NumLevels()) + " levels below " +
                 std::to_string(input_level_) + ", and there are not that many");
  AssertTrue(leaf_degree >= 1, "SlimPoly: the leaf degree collapsed");
}

template <typename word>
std::vector<int> SlimPolyHandler<word>::GetRotationDistances() const {
  std::vector<int> d;
  d.reserve(plan_.j);
  for (int l = plan_.j; l >= 1; l--) d.push_back(block_slots_ * (1 << (l - 1)));
  return d;
}

template <typename word>
void SlimPolyHandler<word>::Compile() {
  const int slots = context_->param_.MaxNumSlots();
  const int blocks = plan_.NumBlocks();
  const int leaf_degree = 1 << (plan_.k - plan_.j);

  // v^(delta): slot `i * block_slots + s` carries leaf `i mod 2^j`'s
  // degree-delta Chebyshev coefficient. Only the first `blocks * block_slots`
  // slots are ever read -- the rest of the ciphertext is whatever the caller's
  // layout put there, and the fold never rotates it into view -- but they are
  // filled periodically anyway so that a decrypt of an intermediate reads the
  // same everywhere, which is what makes a host cross-check possible.
  leaf_msg_.assign(leaf_degree + 1, std::vector<Complex>(slots, Complex(0.0)));
  for (int delta = 0; delta <= leaf_degree; delta++) {
    for (int s = 0; s < slots; s++) {
      const int block = (s / block_slots_) % blocks;
      const std::vector<double> &c = plan_.leaf[block];
      const double v =
          (delta < static_cast<int>(c.size())) ? c[delta] : 0.0;
      leaf_msg_[delta][s] = Complex(v, 0.0);
    }
  }

  // m^(l): slot `i * block_slots + s` carries `m^(l)_{i mod 2^l}`, which is
  // the constant iteration `l + 1` subtracts.
  m_msg_.assign(plan_.j, std::vector<Complex>(slots, Complex(0.0)));
  for (int l = 0; l < plan_.j; l++) {
    const int period = 1 << l;
    for (int s = 0; s < slots; s++) {
      const int block = (s / block_slots_) % blocks;
      m_msg_[l][s] = Complex(plan_.m[l][block % period], 0.0);
    }
  }

  // --- encode -----------------------------------------------------------
  // The leaf's plaintexts sit at the level the Chebyshev basis lands on, with
  // the scale divided out per degree exactly as `EvalPolyNode::Compile` does
  // for its scalar constants: the product `T_delta * v^(delta)` has to reach
  // one common scale before the accumulation.
  const int basis_levels = plan_.k - plan_.j;
  const int leaf_work_level = input_level_ - basis_levels;
  const int leaf_out_level = leaf_work_level - 1;
  const double leaf_out_scale = context_->param_.GetScale(leaf_out_level);
  const double leaf_work_scale =
      leaf_out_scale * context_->param_.GetRescalePrimeProd(leaf_work_level);

  basis_scale_.assign(leaf_degree + 1, 1.0);
  if (leaf_degree >= 2) {
    BasisMap<word> probe(input_level_, input_scale_, EvalPolyType::kNormal,
                         true);
    for (int delta = 2; delta <= leaf_degree; delta++) probe.AddBase(context_,
                                                                    delta);
    for (int delta = 2; delta <= leaf_degree; delta++) {
      basis_scale_[delta] = probe.GetBaseLevelAndScale(delta).second;
    }
  }
  basis_scale_[1] = input_scale_;

  leaf_pt_.clear();
  leaf_pt_.resize(leaf_degree + 1);
  for (int delta = 0; delta <= leaf_degree; delta++) {
    const double scale =
        (delta == 0) ? leaf_work_scale : (leaf_work_scale / basis_scale_[delta]);
    context_->encoder_.Encode(leaf_pt_[delta], leaf_work_level, scale,
                              leaf_msg_[delta]);
  }

  // m^(l) is subtracted from a freshly rescaled square, so it sits at that
  // level on that level's canonical scale.
  m_pt_.clear();
  m_pt_.resize(plan_.j);
  for (int l = plan_.j; l >= 1; l--) {
    const int lvl = leaf_out_level - (plan_.j - l + 1);
    context_->encoder_.Encode(m_pt_[l - 1], lvl, context_->param_.GetScale(lvl),
                              m_msg_[l - 1]);
  }
  compiled_ = true;
}

template <typename word>
void SlimPolyHandler<word>::Evaluate(Ct &res, const Ct &input,
                                     const EvkMap<word> &evk) const {
  AssertTrue(compiled_, "SlimPoly: Compile() must run before Evaluate()");
  AssertTrue(context_->param_.NPToLevel(input.GetNP()) == input_level_,
             "SlimPoly: the input is not at the level the plan was compiled "
             "for");
  AssertFalse(input.HasRx(),
              "SlimPoly: relinearization required before Evaluate");
  context_->AssertSameScale(input, input_scale_);
  const Evk &mult_key = evk.GetMultiplicationKey();
  const int leaf_degree = 1 << (plan_.k - plan_.j);
  const int basis_levels = plan_.k - plan_.j;
  const int leaf_work_level = input_level_ - basis_levels;

  // --- line 1: Evaluate, the leaf with PLAINTEXT coefficients -------------
  //
  // `sum_delta v^(delta) T_delta(x)`, which is [SYLPH]'s `Evaluate` subroutine.
  // The paper implements it with Paterson-Stockmeyer over plaintext
  // coefficients; this builds the WHOLE Chebyshev basis instead, which costs
  // O(leaf_degree) ciphertext multiplications rather than O(sqrt(leaf_degree)).
  // That is deliberate and it is cheap where slim is actually used: the leaf
  // degree is `2^(k-j)`, so the depths the appendix D search makes usable
  // (j = k and j = k-1, where `m` stays near 1) leave a leaf of degree 1 or 2
  // and no basis worth a square root. A plaintext-coefficient
  // Paterson-Stockmeyer is the refinement for small `j`, and it is `EvalPoly`'s
  // tree with `PAccum` in place of `CAccum` -- the same change the BERT
  // branch's mode plan wants.
  Ct accum;
  {
    std::map<int, MultiLevelCiphertext<word>> basis;
    Ct input_tmp;
    context_->Copy(input_tmp, input);
    basis.try_emplace(1, context_->param_, std::move(input_tmp));
    if (leaf_degree >= 2) {
      BasisMap<word> basis_map(input_level_, input_scale_,
                               EvalPolyType::kNormal, true);
      for (int delta = 2; delta <= leaf_degree; delta++) {
        basis_map.AddBase(context_, delta);
      }
      basis_map.Evaluate(context_, basis, mult_key);
    }

    NPInfo np = context_->param_.LevelToNP(leaf_work_level);
    std::vector<std::vector<DvConstView<word>>> ct_srcs;
    std::vector<DvConstView<word>> pt_srcs;
    double scale = 0.0;
    int num_slots = 0;
    for (int delta = 1; delta <= leaf_degree; delta++) {
      MultiLevelCiphertext<word> &ml = basis.at(delta);
      int lvl = ml.GetMaxLevel();
      while (!context_->IsMultUnsafeCompatible(lvl, leaf_work_level)) lvl -= 1;
      context_->AddLowerLevelsUntil(ml, lvl);
      const Ct &ct = ml.AtLevel(lvl);
      const int ter_diff = ct.GetNP().num_ter_ - np.num_ter_;
      AssertTrue(ter_diff >= 0, "SlimPoly: leaf level mismatch");
      ct_srcs.push_back(ct.ConstViewVector(ter_diff));
      pt_srcs.push_back(leaf_pt_[delta].ConstView());
      num_slots = Max(num_slots, ct.GetNumSlots());
      const double s = ct.GetScale() * leaf_pt_[delta].GetScale();
      if (scale == 0.0) {
        scale = s;
      } else {
        context_->AssertSameScale(scale, s);
      }
    }
    accum.RemoveRx();
    accum.ModifyNP(np);
    accum.SetScale(scale);
    accum.SetNumSlots(num_slots);
    std::vector<DvView<word>> dst = accum.ViewVector(0, true);
    context_->elem_handler_.PAccum(dst, np, ct_srcs, pt_srcs);
    context_->Add(accum, accum, leaf_pt_[0]);
    context_->Rescale(res, accum);
  }

  // --- lines 2-5: the fold ------------------------------------------------
  //
  //   for l = j down to 1:  ct <- ct^2 ;  ct <- ct + Rot(ct, 2^t 2^(l-1)) - m
  //
  // Before iteration `l`, block `i` holds `U^(l)_{i mod 2^l}(x)`; after it,
  // `U^(l-1)_{i mod 2^(l-1)}(x)`. At `l = 1` every block holds `P(x)`.
  Ct rotated;
  for (int l = plan_.j; l >= 1; l--) {
    context_->HMult(res, res, res, mult_key);
    const int dist = block_slots_ * (1 << (l - 1));
    // HRotAdd is res = (a << dist) + b and must not alias its inputs.
    context_->HRotAdd(rotated, res, res, evk.GetRotationKey(dist), dist);
    context_->Sub(res, rotated, m_pt_[l - 1]);
  }

  AssertTrue(context_->param_.NPToLevel(res.GetNP()) == output_level_,
             "SlimPoly: Algorithm 1 did not land where Compile said it would");
  // As `EvalPoly::Evaluate` does: check the scale rather than assert it into
  // being, then set it exactly so accumulated double error cannot drift.
  context_->AssertSameScale(res, output_scale_);
  res.SetScale(output_scale_);
  if (plan_.negated) {
    // Lemma 1 was applied to -P; the caller was told, but say it again here
    // rather than let a wrong sign leave the class quietly.
    Warn("SlimPoly: the plan is for -P; the caller must negate the result");
  }
}

template <typename word>
size_t SlimPolyHandler<word>::GetPlaintextBytes() const {
  if (!compiled_) return 0;
  size_t bytes = 0;
  for (const Pt &pt : leaf_pt_) bytes += pt.GetNP().GetNumTotal() *
                                        context_->param_.degree_ * sizeof(word);
  for (const Pt &pt : m_pt_) bytes += pt.GetNP().GetNumTotal() *
                                     context_->param_.degree_ * sizeof(word);
  return bytes;
}

template class SlimPolyHandler<uint32_t>;
template class SlimPolyHandler<uint64_t>;

}  // namespace cheddar
