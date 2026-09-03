#include "extension/Profile.h"
#include "extension/BootContext.h"

#include "core/Mlwe.h"
#include "core/Streams.h"
#include "extension/CiModuleBasis.h"
#include "extension/CiSinCBasis.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"

namespace {

int Log2Scale(double scale) {
  // round log2(scale)
  return static_cast<int>(std::log2(scale) + 0.5);
}

}  // namespace

namespace cheddar {

template <typename word>
ContextPtr<word> BootContext<word>::GetContext() {
  return this->shared_from_this();
}

template <typename word>
ConstContextPtr<word> BootContext<word>::GetContext() const {
  return this->shared_from_this();
}

template <typename word>
std::shared_ptr<BootContext<word>> BootContext<word>::Create(
    const Parameter<word> &param, const BootParameter &boot_param) {
  return std::shared_ptr<BootContext<word>>(
      new BootContext<word>(param, boot_param));
}

template <typename word>
BootContext<word>::BootContext(const Parameter<word> &param,
                               const BootParameter &boot_param)
    : Base(param), boot_param_{boot_param} {
  // Check if param and boot_param is consistent.
  //
  // The invariant is on where EvalMod *ends*, not on where StC starts. Those
  // are the same level exactly when there is no slack, which is every shipped
  // preset, so this is the original check for all of them. With a slack gap
  // StC is compiled below EvalMod's output on purpose ([SYLPH] figure 2 puts
  // the non-linear operators in between), and pinning the check to StC would
  // reject precisely the configuration the gap exists for.
  AssertTrue(
      param.max_level_ == boot_param.max_level_ &&
          param.default_encryption_level_ == boot_param.GetEvalModEndLevel(),
      "Parameter mismatch for BootContext");

  // At level 0, the scale is adjusted
  // level_zero_scale --> level_zero_scale * 2^log_scaleup_
  // which is approximately equal to q0 / message_ratio;
  // Bootstrapping assumes message is in range (-1, 1) and
  // scale : q0 = 1 : message_ratio
  // to enable correct operation in the approximate eval mod 1 operation
  // this adjustment at level 0 is for fully utilizing the available data
  // range
  double level_zero_scale = param.base_scale_;
  int log_level_zero_scale = Log2Scale(level_zero_scale);

  std::vector<word> q0 = param.GetPrimeVector(param.LevelToNP(-1));
  double q0_prod = 1.0;
  for (word q : q0) {
    q0_prod *= q;
  }
  int log_q0_prod = Log2Scale(q0_prod);
  log_scaleup_ =
      (log_q0_prod - log_level_zero_scale) - boot_param.log_message_ratio_;
  AssertTrue(log_scaleup_ >= 0, "Invalid level zero scaleup");
  level_zero_scale *= (1 << log_scaleup_);

  AssertTrue(log_q0_prod <= 62, "Invalid q0_prod");
  AssertTrue(log_level_zero_scale <= 62, "Invalid level_zero_scale");

  // THE CROSSING CONSTANT, DERIVED. See the header: this is the factor
  // `HalfBoot` leaves on the message, and it is a property of the level-zero
  // primes rather than something to fit. `log_scaleup_` above is built out of
  // *rounded* logarithms, so the ratio it lands on is `2^-log_message_ratio`
  // only to the extent that `q0_prod` is a power of two -- which it is not.
  message_ratio_ = level_zero_scale / q0_prod;

  int log_eval_mod_start_scale =
      Log2Scale(param.GetRescalePrimeProd(boot_param.GetEvalModStartLevel()));
  int actual_K = (1 << boot_param.num_double_angle_) * boot_param.initial_K_;

  double eval_mod_end_scale = (UINT64_C(1) << log_eval_mod_start_scale);
  int eval_mod_levels = boot_param.GetNumEvalModLevels();
  int eval_mod_start_level = boot_param.GetEvalModStartLevel();
  for (int i = 0; i < eval_mod_levels; i++) {
    eval_mod_end_scale = eval_mod_end_scale * eval_mod_end_scale /
                         param.GetRescalePrimeProd(eval_mod_start_level - i);
  }

  // See the development notes for details.
  //
  // Both constants carry over to the real subring unchanged, which is not
  // obvious and was verified numerically before it was relied on. The 1/degree
  // in cts_const_ is the product of three factors that shift against each
  // other: Trace contributes MaxNumSlots() / num_slots, the CtS stages
  // contribute num_slots, and the ordinary ring's real/imaginary split
  // contributes a further 2 that the real subring does not need -- because its
  // MaxNumSlots() is degree where the ordinary ring's is degree / 2. Both
  // products are degree. stc_const_ needs no adjustment either: SlotToCoeff
  // lands on E a exactly, once the diag(1, 2, ..., 2) that E^T E leaves behind
  // is folded into its first phase (EvalSpecialFFT.cpp).
  cts_const_ =
      (UINT64_C(1) << (log_eval_mod_start_scale - this->param_.log_degree_)) /
      (q0_prod * actual_K);
  stc_const_ = (q0_prod * param.GetScale(boot_param.GetEndLevel())) /
               (eval_mod_end_scale * level_zero_scale);

  this->encoder_.EncodeConstant(scaleup_const_, -1, 1.0, (1 << log_scaleup_));

  // Populating mod_max_intt_const_;

  int num_base = q0.size();
  HostVector<word> intt_const(num_base, 1);
  for (int i = 0; i < num_base; i++) {
    word mod_prime = q0[i];
    for (int j = 0; j < num_base; j++) {
      if (i != j)
        intt_const[i] = primeutil::MultMod(intt_const[i], q0[j], mod_prime);
    }
    intt_const[i] = primeutil::MultMod(
        intt_const[i], static_cast<word>(this->param_.degree_), mod_prime);
    intt_const[i] = primeutil::InvMod(intt_const[i], mod_prime);
  }
  CopyHostToDevice(mod_max_intt_const_, intt_const);
}

template <typename word>
double BootContext<word>::GetStCInputScale() const {
  AssertTrue(eval_mod_ != nullptr,
             "GetStCInputScale: EvalMod not prepared, so the scale StC expects "
             "is not known yet");
  return eval_mod_->end_scale_;
}

template <typename word>
double BootContext<word>::GetEvalModStartScale() const {
  AssertTrue(eval_mod_ != nullptr,
             "GetEvalModStartScale: EvalMod not prepared");
  return eval_mod_->start_scale_;
}

template <typename word>
double BootContext<word>::GetCtSConst() const {
  return cts_const_;
}

template <typename word>
double BootContext<word>::GetStCConst(BootVariant variant) const {
  return (variant == BootVariant::kImaginaryRemoving ||
          variant == BootVariant::kMergeTwoReal)
             ? stc_const_ / 2
             : stc_const_;
}

template <typename word>
int BootContext<word>::GetBootEnabledNumSlots(int num_slots) const {
  int orig_num_slots = num_slots;
  int max_num_slots = this->param_.MaxNumSlots();
  AssertTrue(num_slots <= max_num_slots, "num_slots exceeds max_num_slots");
  AssertTrue(IsPowOfTwo(num_slots), "Num slots must be power of 2");
  if (!IsBootPrepared(num_slots)) {
    Warn("BootContext not prepared for num slots: " +
         std::to_string(num_slots));
    num_slots *= 2;
    while (num_slots <= max_num_slots) {
      if (IsBootPrepared(num_slots)) {
        Warn("Using BootContext prepared for num slots: " +
             std::to_string(num_slots));
        break;
      }
      num_slots *= 2;
    }
    if (num_slots > max_num_slots) {
      Fail("No BootContext available for num slots: " +
           std::to_string(orig_num_slots));
    }
  }
  return num_slots;
}

template <typename word>
void BootContext<word>::PrepareEvalMod() {
  if (eval_mod_ != nullptr) {
    Warn("EvalMod already prepared");
    return;
  }
  eval_mod_ = std::make_unique<EvalMod<word>>(GetContext(), boot_param_);
}

template <typename word>
void BootContext<word>::PrepareEvalSpecialFFT(
    int num_slots, BootVariant variant,
    const BootContext<word> *cts_donor) {
  AssertTrue(IsPowOfTwo(num_slots), "Only power-of-two slots are supported");
  // Both non-default variants are statements about an imaginary part: one
  // removes it after StC, the other packs two real messages into one complex
  // ciphertext. The real subring has no imaginary part to remove and no second
  // axis to pack into -- its slots are already real and already all used.
  AssertTrue(!this->param_.conjugate_invariant_ ||
                 variant == BootVariant::kNormal,
             "PrepareEvalSpecialFFT: kImaginaryRemoving and kMergeTwoReal are "
             "statements about an imaginary part the real subring does not "
             "have");
  // A CoeffToSlot plaintext is a device buffer of RNS limbs and carries no
  // record of the parameter set it was encoded against, so every condition
  // that makes the donor's tables the same tables is checked HERE rather than
  // left to the caller's judgement. The list is exactly what `PreparePlaintexts`
  // reads on the CtS side: the slot count, the stage matrices (degree and
  // ring), the compile levels, the phase count and the constant.
  std::shared_ptr<typename EvalSpecialFFT<word>::CtSTables> shared_cts;
  if (cts_donor != nullptr) {
    AssertTrue(cts_donor != this,
               "PrepareEvalSpecialFFT: a BootContext cannot donate to itself");
    const auto &dparam = cts_donor->param_;
    const auto &dboot = cts_donor->boot_param_;
    AssertTrue(dparam.log_degree_ == this->param_.log_degree_ &&
                   dparam.conjugate_invariant_ ==
                       this->param_.conjugate_invariant_ &&
                   dparam.main_primes_ == this->param_.main_primes_ &&
                   dparam.ter_primes_ == this->param_.ter_primes_ &&
                   dparam.aux_primes_ == this->param_.aux_primes_ &&
                   dparam.level_config_ == this->param_.level_config_,
               "PrepareEvalSpecialFFT: the CtS donor's Parameter differs from "
               "this one's; the tables are encoded against its primes");
    AssertTrue(dboot.GetCtSStartLevel() == boot_param_.GetCtSStartLevel() &&
                   dboot.num_cts_levels_ == boot_param_.num_cts_levels_,
               "PrepareEvalSpecialFFT: the CtS donor compiles CoeffToSlot at a "
               "different level or across a different number of phases");
    AssertTrue(cts_donor->GetCtSConst() == GetCtSConst(),
               "PrepareEvalSpecialFFT: the CtS donor's cts_const differs");
    AssertTrue(cts_donor->eval_fft_.count(num_slots) != 0,
               "PrepareEvalSpecialFFT: the CtS donor has no tables for " +
                   std::to_string(num_slots) + " slots");
    shared_cts = cts_donor->eval_fft_.at(num_slots).GetCtSTables();
  }
  // TODO: Implement PrepareBootConversionMatrices
  eval_fft_.try_emplace(num_slots, GetContext(), boot_param_, num_slots,
                        GetCtSConst(), GetStCConst(variant),
                        std::move(shared_cts));
  boot_variant_.try_emplace(num_slots, variant);
}

template <typename word>
bool BootContext<word>::ReleaseEvalSpecialFFT(int num_slots) {
  // A plain erase. The tables are the map's mapped_type, so this is the whole
  // release; everything that reads them goes through `eval_fft_.at`, which
  // then throws with the slot count in the message rather than reading freed
  // memory.
  // The variant goes with them. `PrepareEvalSpecialFFT` emplaces both with
  // `try_emplace`, so leaving the variant behind would make a re-prepare at a
  // different variant silently keep the old one.
  boot_variant_.erase(num_slots);
  return eval_fft_.erase(num_slots) != 0;
}

template <typename word>
bool BootContext<word>::ReleaseCtS(int num_slots) {
  auto it = eval_fft_.find(num_slots);
  if (it == eval_fft_.end() || !it->second.HasCtS()) return false;
  it->second.DropCtS();
  return true;
}

template <typename word>
bool BootContext<word>::IsBootPrepared(int num_slots) const {
  return (eval_mod_ != nullptr) &&
         (eval_fft_.find(num_slots) != eval_fft_.end());
}

template <typename word>
void BootContext<word>::AddRequiredRotations(EvkRequest &req, int num_slots,
                                             bool min_ks) const {
  int max_num_slots = this->param_.MaxNumSlots();
  num_slots = GetBootEnabledNumSlots(num_slots);
  // Trace and rotations for possible slot modification after StC
  for (int ns = num_slots; ns < max_num_slots; ns *= 2) {
    req.AddRequest(ns, boot_param_.GetMaxLevel());
  }
  eval_fft_.at(num_slots).AddRequiredRotations(req, min_ks);
}

template <typename word>
void BootContext<word>::ModUpToLevel(Ct &res, const Ct &input,
                                     const EvkMap<word> &evk_map,
                                     int target_level, int module_small_degree,
                                     int tower_inner_rank) const {
  NvtxScope _nv("boot: ModRaise");
  if (target_level < 0) target_level = boot_param_.GetMaxLevel();
  AssertTrue(target_level > 0 && target_level <= this->param_.max_level_,
             "ModUpToLevel: target level out of range");
  // The number of q primes at the target, which was param_.L_ back when the
  // target was always the parameter set's maximum.
  NPInfo target_np = this->param_.LevelToNP(target_level);
  const int L = target_np.GetNumQ();
  const int alpha = this->param_.alpha_;
  const int degree = this->param_.degree_;
  NPInfo np = this->param_.LevelToNP(-1);
  AssertTrue(input.GetNP() == np, "ModUpToLevel: input NP mismatch");
  AssertTrue(!input.HasRx(), "ModUpToLevel: input has Rx");
  res.RemoveRx();
  res.ModifyNP(np);
  res.SetNumSlots(input.GetNumSlots());
  res.SetScale(input.GetScale());

  bool sse = this->param_.IsUsingSparseSecretEncapsulation();

  // SSE case
  const Ct *working_ct = &input;
  if (sse) {
    // Dense to sparse key-switch
    const auto &dts_key = evk_map.GetDenseToSparseKey();

    // DtS key-switch
    this->MultKey(res, input, dts_key);
    working_ct = &res;
  }
  // ModUp sequence
  Dv tmp_bx(L * degree);
  int tmp_ax_num_aux = sse ? alpha : 0;
  Dv tmp_ax((L + tmp_ax_num_aux) * degree);

  DvView<word> res_bx_view = res.BxView();
  DvView<word> res_ax_view = res.AxView();
  DvView<word> tmp_bx_view = tmp_bx.View(0);
  DvView<word> tmp_ax_view = tmp_ax.View(tmp_ax_num_aux * degree);

  // The module-centred form (Doing.md 3.5): the scan takes the level-zero
  // residues to module coordinates, the lift centres THOSE, and the
  // recomposition returns to native coefficients at the target modulus. Both
  // maps are integer and linear, so per residue they are exact, and what
  // changes is only which representative of the same ciphertext mod q0 gets
  // lifted -- the one whose module coordinates lie in (-q0/2, q0/2].
  std::unique_ptr<MlweHandler<word>> mlwe;
  if (module_small_degree > 0) {
    AssertTrue(this->param_.conjugate_invariant_,
               "ModUpToLevel: the module-centred lift is a conjugate-"
               "invariant object");
    mlwe = std::make_unique<MlweHandler<word>>(this->param_,
                                               this->ntt_handler_);
  }

  // TODO(jongmin.kim): We can remove some redundant NTT here.
  // ModUpToLevel(bx)
  NPInfo max_level_np = target_np;
  this->ntt_handler_.INTTAndMultConst(res_bx_view, np,
                                      working_ct->BxConstView(),
                                      mod_max_intt_const_.ConstView(0));
  if (mlwe) {
    mlwe->ScanInPlace(res_bx_view, np, module_small_degree, tower_inner_rank);
  }
  this->elem_handler_.ModUpToLevel(tmp_bx_view, res.BxConstView(),
                                   target_level);
  if (mlwe) {
    mlwe->RecomposeInPlace(tmp_bx_view, max_level_np, module_small_degree,
                           tower_inner_rank);
  }
  this->ntt_handler_.NTT(tmp_bx_view, max_level_np, tmp_bx.ConstView(0), true);

  // ModUpToLevel(ax)
  max_level_np.num_aux_ = tmp_ax_num_aux;
  this->ntt_handler_.INTTAndMultConst(res_ax_view, np,
                                      working_ct->AxConstView(),
                                      mod_max_intt_const_.ConstView(0));
  if (mlwe) {
    mlwe->ScanInPlace(res_ax_view, np, module_small_degree, tower_inner_rank);
  }
  this->elem_handler_.ModUpToLevel(tmp_ax_view, res.AxConstView(),
                                   target_level);
  if (mlwe) {
    mlwe->RecomposeInPlace(tmp_ax_view, max_level_np, module_small_degree,
                           tower_inner_rank);
  }
  this->ntt_handler_.NTT(tmp_ax_view, max_level_np,
                         tmp_ax.ConstView(tmp_ax_num_aux * degree), true);

  if (sse) {
    // StD key-switch
    const auto &std_key = evk_map.GetSparseToDenseKey();
    const auto &std_mod_switcher = this->GetStDModSwitchHandler();
    // MultKey
    Ct tmp_std(max_level_np);
    std::vector<DvView<word>> tmp_std_view = tmp_std.ViewVector();
    this->elem_handler_.PMult(tmp_std_view, max_level_np,
                              std_key.ConstViewVector(0),
                              tmp_ax.ConstView(tmp_ax_num_aux * degree));

    // PseudoModUp: tmp_std.bx_ += tmp_bx * p_prod
    DvView<word> tmp_std_bx_view(tmp_std.bx_.data(), L * degree, 0);
    std::vector<DvView<word>> caccum_res{tmp_std_bx_view};
    std::vector<std::vector<DvConstView<word>>> caccum_input;
    caccum_input.push_back(std::vector<DvConstView<word>>{tmp_bx.ConstView(0)});
    caccum_input.push_back(
        std::vector<DvConstView<word>>{tmp_std.BxConstView()});
    max_level_np.num_aux_ = 0;
    this->elem_handler_.CAccum(caccum_res, max_level_np, caccum_input,
                               {this->GetPProd(max_level_np)});
    // ModDown
    res.ModifyNP(max_level_np);
    DvView<word> final_bx_view = res.BxView();
    DvView<word> final_ax_view = res.AxView();
    std_mod_switcher.ModDown(final_bx_view, tmp_std.BxConstView());
    std_mod_switcher.ModDown(final_ax_view, tmp_std.AxConstView());
  } else {
    res.bx_ = std::move(tmp_bx);
    res.ax_ = std::move(tmp_ax);
    res.ModifyNP(max_level_np);
  }
}

template <typename word>
void BootContext<word>::CoeffToSlot(Ct &res, int num_slots, const Ct &input,
                                    const EvkMap<word> &evk_map,
                                    bool min_ks /*= false*/) const {
  NvtxScope _nv("boot: CtS native");
  eval_fft_.at(num_slots).EvaluateCtS(GetContext(), res, input, evk_map,
                                      min_ks);
}

template <typename word>
void BootContext<word>::SlotToCoeff(Ct &res, int num_slots, const Ct &input,
                                    const EvkMap<word> &evk_map,
                                    bool min_ks /*= false*/) const {
  NvtxScope _nv("boot: StC native");
  eval_fft_.at(num_slots).EvaluateStC(GetContext(), res, input, evk_map,
                                      min_ks);
}

template <typename word>
void BootContext<word>::CoeffToSlotBatch(
    std::vector<Ct> &res, int num_slots,
    const std::vector<const Ct *> &inputs,
    const EvkMap<word> &evk_map) const {
  NvtxScope _nv("boot: CtS native batch");
  eval_fft_.at(num_slots).EvaluateCtSBatch(GetContext(), res, inputs, evk_map);
}

template <typename word>
void BootContext<word>::SlotToCoeffBatch(
    std::vector<Ct> &res, int num_slots,
    const std::vector<const Ct *> &inputs,
    const EvkMap<word> &evk_map) const {
  NvtxScope _nv("boot: StC native batch");
  eval_fft_.at(num_slots).EvaluateStCBatch(GetContext(), res, inputs, evk_map);
}

template <typename word>
void BootContext<word>::PrepareSinC(int num_slots, int sub_degree,
                                    int stc_level, int cts_level,
                                    int num_phases) {
  AssertTrue(eval_fft_.count(num_slots) != 0,
             "PrepareSinC: call PrepareEvalSpecialFFT first");
  eval_fft_.at(num_slots).PrepareSinC(GetContext(), sub_degree, stc_level,
                                      cts_level, num_phases);
}

template <typename word>
int BootContext<word>::GetSinCNumPhases(int num_slots) const {
  return eval_fft_.at(num_slots).GetSinCNumPhases();
}

template <typename word>
void BootContext<word>::AddRequiredSinCRotations(EvkRequest &req,
                                                 int num_slots) const {
  eval_fft_.at(num_slots).AddRequiredSinCRotations(req);
}

template <typename word>
void BootContext<word>::SlotToSinC(Ct &res, int num_slots, const Ct &input,
                                   const EvkMap<word> &evk_map) const {
  eval_fft_.at(num_slots).EvaluateSlotToSinC(GetContext(), res, input,
                                             evk_map);
}

template <typename word>
void BootContext<word>::SinCToSlot(Ct &res, int num_slots, const Ct &input,
                                   const EvkMap<word> &evk_map) const {
  eval_fft_.at(num_slots).EvaluateSinCToSlot(GetContext(), res, input,
                                             evk_map);
}

template <typename word>
void BootContext<word>::PrepareSinCPrefix(int num_slots, int sub_degree,
                                          int level, int num_phases,
                                          double constant, double pt_scale) {
  AssertTrue(eval_fft_.count(num_slots) != 0,
             "PrepareSinCPrefix: call PrepareEvalSpecialFFT first");
  eval_fft_.at(num_slots).PrepareSinCPrefix(GetContext(), sub_degree, level,
                                            num_phases, constant, pt_scale);
}

template <typename word>
int BootContext<word>::GetSinCPrefixNumPhases(int num_slots) const {
  return eval_fft_.at(num_slots).GetSinCPrefixNumPhases();
}

template <typename word>
StripedMatrix BootContext<word>::SinCPrefixMatrix(int num_slots,
                                                  int sub_degree,
                                                  int &window) const {
  return eval_fft_.at(num_slots).SinCPrefixMatrix(sub_degree, window);
}

template <typename word>
void BootContext<word>::AddRequiredSinCPrefixRotations(EvkRequest &req,
                                                       int num_slots) const {
  eval_fft_.at(num_slots).AddRequiredSinCPrefixRotations(req);
}

template <typename word>
void BootContext<word>::SinCPrefix(Ct &res, int num_slots, const Ct &input,
                                   const EvkMap<word> &evk_map) const {
  eval_fft_.at(num_slots).EvaluateSinCPrefix(GetContext(), res, input, evk_map);
}

template <typename word>
void BootContext<word>::EvaluateMod(Ct &res, const Ct &input,
                                    const Evk &mult_key) const {
  NvtxScope _nv("boot: EvalMod");
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");
  // A launch-bound window (EvalMod keeps the card ~20% busy, Doing.md
  // 3.18): the next layer's preparation may issue a bounded chunk here.
  IdleWindow::Notify("evalmod");
  this->AssertSameScale(input, eval_mod_->start_scale_);
  eval_mod_->Evaluate(GetContext(), res, input, mult_key);
  this->AssertSameScale(res, eval_mod_->end_scale_);
}

template <typename word>
void BootContext<word>::EvaluateModAfterCtS(Ct &res, Ct &main_ct,
                                            bool full_slot,
                                            const EvkMap<word> &evk_map) const {
  NvtxScope _nv("boot: EvalMod");
  main_ct.SetScale(eval_mod_->start_scale_);

  if (this->param_.conjugate_invariant_) {
    // Nothing to unpick. CoeffToSlot on the real subring lands the coefficient
    // vector in REAL slots, so the modular reduction runs once over all of
    // them. The ordinary ring spends two EvalMod calls separating a real half
    // from an imaginary one, or one plus a conjugation key switch when there
    // are spare slots to merge into -- and it needs two ciphertexts to carry
    // what this carries in one. That is where the branch earns back the
    // transform costing twice as much (Doing.md 1.5bb).
    EvaluateMod(res, main_ct, evk_map.GetMultiplicationKey());
    return;
  }

  if (full_slot) {
    // Split the axes, reduce each, fold them back. The first two steps are
    // `SplitAndEvaluateMod`; `HalfBootPair` is the same call without the fold.
    Ct ct_conj;
    SplitAndEvaluateMod(res, ct_conj, main_ct, evk_map);
    this->MultImaginaryUnit(ct_conj, ct_conj);
    this->Add(res, res, ct_conj);
  } else {
    // Can merge real and imag part using extra slots
    this->HConjAdd(res, main_ct, main_ct, evk_map.GetConjugationKey());
    EvaluateMod(res, res, evk_map.GetMultiplicationKey());
  }
}

template <typename word>
void BootContext<word>::SplitAndEvaluateMod(Ct &lo, Ct &hi, const Ct &main_ct,
                                            const EvkMap<word> &evk_map) const {
  NvtxScope _nv("boot: EvalMod split");
  // main = a + i b. Conjugation gives a - i b, and the two combinations are
  // real: lo = 2a, and hi = i * (conj - main) = i * (-2 i b) = 2b. Whatever
  // constant CtS folded in rides both identically, which is why the calibration
  // that makes the real axis right makes the imaginary one right too.
  this->HConj(hi, main_ct, evk_map.GetConjugationKey());
  this->Add(lo, main_ct, hi);
  this->Sub(hi, hi, main_ct);
  this->MultImaginaryUnit(hi, hi);
  EvaluateMod(lo, lo, evk_map.GetMultiplicationKey());
  EvaluateMod(hi, hi, evk_map.GetMultiplicationKey());
}

template <typename word>
void BootContext<word>::HalfBootPair(Ct &res_lo, Ct &res_hi, const Ct &lo,
                                     const Ct &hi, const EvkMap<word> &evk_map,
                                     bool min_ks) const {
  NvtxScope _nv("boot: HalfBootPair");
  counts_.pair++;
  AssertTrue(!this->param_.conjugate_invariant_,
             "HalfBootPair: the real subring's CtS lands real slots, so there "
             "is no second axis to fill and no pair to make");
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");
  const int max_num_slots = this->param_.MaxNumSlots();
  const int input_num_slots = lo.GetNumSlots();
  AssertTrue(hi.GetNumSlots() == input_num_slots,
             "HalfBootPair: the two inputs must carry the same slot count");
  const int num_slots = GetBootEnabledNumSlots(input_num_slots);
  AssertTrue(num_slots == max_num_slots,
             "HalfBootPair: only the full-slot bootstrap separates the axes; a "
             "sparse one merges them into spare slots and has no free half");

  const int lo_level = this->param_.NPToLevel(lo.GetNP());
  const int hi_level = this->param_.NPToLevel(hi.GetNP());
  AssertTrue(lo_level == hi_level,
             "HalfBootPair: the two inputs must be at the same level");
  if (lo_level > 0) {
    Ct lo_down, hi_down;
    this->LevelDown(lo_down, lo, 0);
    this->LevelDown(hi_down, hi, 0);
    HalfBootPair(res_lo, res_hi, lo_down, hi_down, evk_map, min_ks);
    return;
  }
  AssertTrue(lo.GetScale() == hi.GetScale(),
             "HalfBootPair: the two inputs must be at the same scale -- they "
             "share one ModRaise and one EvalMod calibration");

  // THE MERGE. `hi`'s payload is coefficients 0..N/2-1 by contract, and
  // multiplying by X^{N/2} moves it to N/2..N-1 with nothing wrapping round the
  // negacyclic sign, so the sum holds both payloads and neither is disturbed.
  Ct merged;
  this->MultImaginaryUnit(merged, hi);
  this->Add(merged, merged, lo);
  HalfBootSplit(res_lo, res_hi, merged, evk_map, min_ks);
}

template <typename word>
void BootContext<word>::HalfBootSplit(Ct &res_lo, Ct &res_hi, const Ct &merged,
                                      const EvkMap<word> &evk_map,
                                      bool min_ks) const {
  NvtxScope _nv("boot: HalfBootSplit");
  counts_.pair++;
  AssertTrue(!this->param_.conjugate_invariant_,
             "HalfBootSplit: the real subring's CtS lands real slots, so there "
             "is no second axis to split");
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");
  const int max_num_slots = this->param_.MaxNumSlots();
  const int input_num_slots = merged.GetNumSlots();
  const int num_slots = GetBootEnabledNumSlots(input_num_slots);
  AssertTrue(num_slots == max_num_slots,
             "HalfBootSplit: only the full-slot bootstrap separates the axes");
  if (this->param_.NPToLevel(merged.GetNP()) > 0) {
    Ct down;
    this->LevelDown(down, merged, 0);
    HalfBootSplit(res_lo, res_hi, down, evk_map, min_ks);
    return;
  }

  // From here it is `HalfBoot`, verbatim, on the merged ciphertext.
  Ct main_ct;
  NPInfo min_np = this->param_.LevelToNP(-1);
  AssertTrue(min_np.IsSubsetOf(merged.GetNP()), "HalfBootPair: Invalid input NP");
  this->MultUnsafe(main_ct, merged, scaleup_const_, -1);
  ModUpToLevel(main_ct, main_ct, evk_map, boot_param_.GetMaxLevel());
  main_ct.SetNumSlots(max_num_slots);
  Trace(main_ct, num_slots, (max_num_slots / num_slots), main_ct, evk_map);
  main_ct.SetNumSlots(num_slots);
  CoeffToSlot(main_ct, num_slots, main_ct, evk_map, min_ks);

  AssertTrue(boot_variant_.at(num_slots) != BootVariant::kImaginaryRemoving,
             "HalfBootPair: kImaginaryRemoving folds work into StC, which this "
             "does not run");
  main_ct.SetScale(eval_mod_->start_scale_);
  SplitAndEvaluateMod(res_lo, res_hi, main_ct, evk_map);

  res_lo.SetNumSlots(input_num_slots);
  res_hi.SetNumSlots(input_num_slots);
  res_lo.SetScale(eval_mod_->end_scale_);
  res_hi.SetScale(eval_mod_->end_scale_);
}

template <typename word>
int BootContext<word>::BootFrontPreCtS(Ct &main_ct, const Ct &input,
                                       const EvkMap<word> &evk_map) const {
  const int max_num_slots = this->param_.MaxNumSlots();
  const int input_num_slots = input.GetNumSlots();
  const int num_slots = GetBootEnabledNumSlots(input_num_slots);
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");

  const Ct *src = &input;
  Ct low;
  if (this->param_.NPToLevel(input.GetNP()) > 0) {
    this->LevelDown(low, input, 0);
    src = &low;
  }

  // 0. Scale up
  NPInfo min_np = this->param_.LevelToNP(-1);
  AssertTrue(min_np.IsSubsetOf(src->GetNP()), "Boot: Invalid input NP");
  this->MultUnsafe(main_ct, *src, scaleup_const_, -1);

  // 1. ModUp with optional DtS/StD key-switch + Trace.
  //
  // Climb only to the level this BootContext's BootParameter asks for, not to
  // the parameter set's maximum. CoeffToSlot, EvalMod and SlotToCoeff are
  // already compiled against boot_param_ (GetCtSStartLevel, GetEvalModStartLevel,
  // GetStCStartLevel all derive from its max_level_), and the rotation keys are
  // requested at boot_param_.GetMaxLevel() too -- ModUp was the one place still
  // hard-wired to param_.max_level_. A BootParameter built with a smaller
  // max_level therefore yields a bootstrap that lands at a chosen level and
  // does every limb operation in between on fewer primes.
  ModUpToLevel(main_ct, main_ct, evk_map, boot_param_.GetMaxLevel());

  // Perform trace
  main_ct.SetNumSlots(max_num_slots);
  Trace(main_ct, num_slots, (max_num_slots / num_slots), main_ct, evk_map);
  main_ct.SetNumSlots(num_slots);
  return input_num_slots;
}

template <typename word>
int BootContext<word>::BootFront(Ct &slots, const Ct &input,
                                 const EvkMap<word> &evk_map,
                                 bool min_ks) const {
  Ct main_ct;
  const int input_num_slots = BootFrontPreCtS(main_ct, input, evk_map);
  const int num_slots = GetBootEnabledNumSlots(input_num_slots);

  // 2. Perform CtS
  CoeffToSlot(slots, num_slots, main_ct, evk_map, min_ks);
  return input_num_slots;
}

template <typename word>
void BootContext<word>::Boot(Ct &res, const Ct &input,
                             const EvkMap<word> &evk_map, bool min_ks) const {
  NvtxScope _nv("boot: Boot");
  counts_.full++;
  const bool full_slot =
      (GetBootEnabledNumSlots(input.GetNumSlots()) == this->param_.MaxNumSlots());

  Ct main_ct;
  const int input_num_slots = BootFront(main_ct, input, evk_map, min_ks);

  // 3. Take the coefficients through EvalMod.
  EvaluateModAfterCtS(res, main_ct, full_slot, evk_map);
  BootBack(res, res, input_num_slots, evk_map, min_ks);
}

template <typename word>
void BootContext<word>::BootBack(Ct &res, Ct &slots, int input_num_slots,
                                 const EvkMap<word> &evk_map,
                                 bool min_ks) const {
  const int num_slots = GetBootEnabledNumSlots(input_num_slots);

  // 4. Finally, perform StC.
  //
  // With slack configured, StC is compiled below where EvalMod ends -- that gap
  // exists so the caller can put the non-linear operators in it ([SYLPH] figure
  // 2). Boot itself puts nothing there, so it has to cross the gap, and
  // LevelDown is a multiply by a level-down constant plus a rescale, which
  // leaves the declared scale alone. That matters here: what StC receives
  // inside Boot is still EvalMod's end scale, exactly as with no slack.
  //
  // The drift is not cosmetic. LevelDown's rescales divide by the actual prime
  // products, which miss the nominal 2^35 per level by up to a part in 800, so
  // the scale it leaves is not the scale it was handed. The message is
  // untouched -- the scale field tracks it exactly -- but the canonical scale
  // this function declares at the end no longer describes the data, and the
  // whole of that mismatch shows up as error.
  //
  // Measured before this line existed, over slack 1 to 4: predicted error
  // -log2|drift - 1| of 11.30, 10.43, 6.89, 9.75 bits against measured 11.78,
  // 10.92, 7.39, 10.24 -- the same +0.49 offset four times over, which is what
  // a pure constant bias looks like under a max-absolute metric. So the gap
  // costs no precision at all; declaring the wrong scale afterwards did.
  double leveldown_drift = 1.0;
  if (boot_param_.GetNumSlackLevels() > 0) {
    const double before = slots.GetScale();
    this->LevelDown(slots, slots, boot_param_.GetStCStartLevel());
    leveldown_drift = slots.GetScale() / before;
  }
  SlotToCoeff(res, num_slots, slots, evk_map, min_ks);

  if (boot_variant_.at(num_slots) == BootVariant::kImaginaryRemoving) {
    // res += HConJ(res)
    this->HConjAdd(res, res, res, evk_map.GetConjugationKey());
  }
  // For kNormal of kMergeTwoReal, no additional operation is needed inside
  // this function. For kMergeTwoReal, extra ops are required after returing.

  // Restore num slots and set scale (just in case)
  res.SetNumSlots(input_num_slots);
  double final_scale =
      this->param_.GetScale(boot_param_.GetEndLevel()) * leveldown_drift;
  res.SetScale(final_scale);
}

template <typename word>
void BootContext<word>::HalfBoot(Ct &res, const Ct &input,
                             const EvkMap<word> &evk_map, bool min_ks) const {
  NvtxScope _nv("boot: HalfBoot");
  counts_.half++;
  int max_num_slots = this->param_.MaxNumSlots();
  int input_num_slots = input.GetNumSlots();
  int num_slots = GetBootEnabledNumSlots(input_num_slots);
  bool full_slot = (num_slots == max_num_slots);
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");

  Ct main_ct;
  int input_level = this->param_.NPToLevel(input.GetNP());
  if (input_level > 0) {
    this->LevelDown(main_ct, input, 0);
    HalfBoot(res, main_ct, evk_map, min_ks);
    return;
  }

  // 0. Scale up
  NPInfo min_np = this->param_.LevelToNP(-1);
  AssertTrue(min_np.IsSubsetOf(input.GetNP()), "Boot: Invalid input NP");
  this->MultUnsafe(main_ct, input, scaleup_const_, -1);

  // 1. ModUp with optional DtS/StD key-switch + Trace.
  //
  // Climb only to the level this BootContext's BootParameter asks for, not to
  // the parameter set's maximum. CoeffToSlot, EvalMod and SlotToCoeff are
  // already compiled against boot_param_ (GetCtSStartLevel, GetEvalModStartLevel,
  // GetStCStartLevel all derive from its max_level_), and the rotation keys are
  // requested at boot_param_.GetMaxLevel() too -- ModUp was the one place still
  // hard-wired to param_.max_level_. A BootParameter built with a smaller
  // max_level therefore yields a bootstrap that lands at a chosen level and
  // does every limb operation in between on fewer primes.
  ModUpToLevel(main_ct, main_ct, evk_map, boot_param_.GetMaxLevel());

  // Perform trace
  main_ct.SetNumSlots(max_num_slots);
  Trace(main_ct, num_slots, (max_num_slots / num_slots), main_ct, evk_map);
  main_ct.SetNumSlots(num_slots);

  // 2. Perform CtS
  CoeffToSlot(main_ct, num_slots, main_ct, evk_map, min_ks);

  // 3. Take the coefficients through EvalMod.
  EvaluateModAfterCtS(res, main_ct, full_slot, evk_map);

  // 4. NO StC. Boot's CoeffToSlot/SlotToCoeff pair is what makes it
  // domain-preserving; stopping here leaves the input's *coefficients* sitting
  // in the output's *slots*, which is the conversion the pipeline needs and
  // cannot get any other way -- CoeffToSlot alone is compiled at max_level and
  // only a ModUp'd ciphertext lives there.
  //
  // kImaginaryRemoving's extra HConjAdd belongs after StC, so it is not applied
  // here; a variant needing it would have to be handled by the caller.
  AssertTrue(boot_variant_.at(num_slots) != BootVariant::kImaginaryRemoving,
             "HalfBoot: kImaginaryRemoving folds work into StC, which this "
             "does not run");

  res.SetNumSlots(input_num_slots);
  // The honest scale: EvaluateMod asserts its own end_scale_, and skipping StC
  // means none of StC's constant has been applied. Declaring
  // GetScale(GetStCStartLevel()) here -- as if StC had run -- put the values
  // out by ~2.6e5. What the remaining constant is gets measured rather than
  // derived through cts_const_, stc_const_ and q0.
  res.SetScale(eval_mod_->end_scale_);
}

template <typename word>
int BootContext<word>::HalfBootModuleFront(
    Ct &slots, const Ct &input, const EvkMap<word> &evk_map,
    const CiModuleBasis<word> &basis) const {
  const int max_num_slots = this->param_.MaxNumSlots();
  const int input_num_slots = input.GetNumSlots();
  // Not `GetBootEnabledNumSlots`: that reads the native tables, and a ring
  // whose crossings are all module ones need not build them.
  AssertTrue(input_num_slots == max_num_slots,
             "HalfBootModule: the module basis is a full-slot object");
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");
  AssertTrue(basis.GetCtSNumLevels() == boot_param_.num_cts_levels_,
             "HalfBootModule: the module CtS must spend exactly the boot "
             "parameter's CtS levels, or EvalMod starts at the wrong level");

  Ct main_ct;
  const Ct *src = &input;
  Ct low;
  const int input_level = this->param_.NPToLevel(input.GetNP());
  if (input_level > 0) {
    this->LevelDown(low, input, 0);
    src = &low;
  }

  NPInfo min_np = this->param_.LevelToNP(-1);
  AssertTrue(min_np.IsSubsetOf(src->GetNP()), "Boot: Invalid input NP");
  this->MultUnsafe(main_ct, *src, scaleup_const_, -1);

  ModUpToLevel(main_ct, main_ct, evk_map, boot_param_.GetMaxLevel(),
               basis.GetSmallDegree());
  main_ct.SetNumSlots(max_num_slots);

  basis.EvaluateCtS(GetContext(), slots, main_ct, evk_map);
  return input_num_slots;
}

template <typename word>
void BootContext<word>::HalfBootModule(Ct &res, const Ct &input,
                                       const EvkMap<word> &evk_map,
                                       const CiModuleBasis<word> &basis) const {
  NvtxScope _nv("boot: HalfBootModule");
  counts_.half++;
  Ct slots;
  const int input_num_slots =
      HalfBootModuleFront(slots, input, evk_map, basis);
  EvaluateModAfterCtS(res, slots, /*full_slot=*/true, evk_map);
  res.SetNumSlots(input_num_slots);
  res.SetScale(eval_mod_->end_scale_);
}

template <typename word>
int BootContext<word>::HalfBootTowerFront(
    Ct &slots, const Ct &input, const EvkMap<word> &evk_map,
    const CiSinCBasis<word> &basis) const {
  const int max_num_slots = this->param_.MaxNumSlots();
  const int input_num_slots = input.GetNumSlots();
  // Not `GetBootEnabledNumSlots`: that reads the native tables, and a ring
  // whose only crossing is this one never builds them (`PrepareEvalMod`
  // alone; the tower CtS' is the basis's).
  AssertTrue(input_num_slots == max_num_slots,
             "HalfBootTower: the tower basis is a full-slot object");
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");
  AssertTrue(basis.GetNumSlots() == max_num_slots,
             "HalfBootTower: the basis was built for another ring");
  AssertTrue(basis.GetCtSLevel() == boot_param_.GetCtSStartLevel() &&
                 basis.GetCtSNumLevels() == boot_param_.num_cts_levels_,
             "HalfBootTower: the tower CtS must be compiled at the CtS start "
             "level and spend exactly the boot parameter's CtS levels, or "
             "EvalMod starts at the wrong level");

  Ct main_ct;
  const Ct *src = &input;
  Ct low;
  const int input_level = this->param_.NPToLevel(input.GetNP());
  if (input_level > 0) {
    this->LevelDown(low, input, 0);
    src = &low;
  }

  NPInfo min_np = this->param_.LevelToNP(-1);
  AssertTrue(min_np.IsSubsetOf(src->GetNP()), "Boot: Invalid input NP");
  this->MultUnsafe(main_ct, *src, scaleup_const_, -1);

  ModUpToLevel(main_ct, main_ct, evk_map, boot_param_.GetMaxLevel(),
               basis.GetOuterSmallDegree(), basis.GetInnerRank());
  main_ct.SetNumSlots(max_num_slots);

  basis.EvaluateCtS(slots, main_ct, evk_map);
  return input_num_slots;
}

template <typename word>
void BootContext<word>::HalfBootTower(Ct &res, const Ct &input,
                                      const EvkMap<word> &evk_map,
                                      const CiSinCBasis<word> &basis) const {
  NvtxScope _nv("boot: HalfBootTower");
  counts_.half++;
  Ct slots;
  const int input_num_slots = HalfBootTowerFront(slots, input, evk_map, basis);
  EvaluateModAfterCtS(res, slots, /*full_slot=*/true, evk_map);
  res.SetNumSlots(input_num_slots);
  res.SetScale(eval_mod_->end_scale_);
}

namespace {
bool &EvalModSerialFlag() {
  static bool serial = [] {
    const char *env = std::getenv("CHEDDAR_EVALMOD_SERIAL");
    return env != nullptr && env[0] == '1';
  }();
  return serial;
}
}  // namespace

template <typename word>
bool BootContext<word>::EvalModSerial() {
  return EvalModSerialFlag();
}

template <typename word>
void BootContext<word>::SetEvalModSerial(bool serial) {
  EvalModSerialFlag() = serial;
}

template <typename word>
void BootContext<word>::EvaluateModBatch(std::vector<Ct *> &cts,
                                         const Evk &mult_key) const {
  AssertTrue(eval_mod_ != nullptr, "EvalMod not prepared");
  const int batch = static_cast<int>(cts.size());
  if (batch == 0) return;
  if (batch == 1 || EvalModSerial()) {
    for (Ct *ct : cts) {
      EvaluateMod(*ct, *ct, mult_key);
    }
    return;
  }
  // The batch buffers hold every basis entry times the group, so the group
  // is chunked: 8 ciphertexts keep the extra residency under ~2 GiB on the
  // landing rings. `CHEDDAR_EVALMOD_BATCH` widens or narrows it.
  static const int max_batch = [] {
    const char *env = std::getenv("CHEDDAR_EVALMOD_BATCH");
    const int value = (env != nullptr) ? std::atoi(env) : 0;
    return value >= 1 ? value : 8;
  }();
  if (batch > max_batch) {
    for (int start = 0; start < batch; start += max_batch) {
      std::vector<Ct *> chunk(cts.begin() + start,
                              cts.begin() + Min(batch, start + max_batch));
      EvaluateModBatch(chunk, mult_key);
    }
    return;
  }
  NvtxScope _nv("boot: EvalModBatch");
  // A launch-bound window in the serial shape; batched it is much shorter,
  // but the prefetch can still use it.
  IdleWindow::Notify("evalmod");

  // Gather the group into one strided buffer. Every ciphertext must stand
  // exactly where the serial EvaluateMod expects it.
  const NPInfo np = cts[0]->GetNP();
  CtBatch<word> state;
  state.Allocate(np, batch, false);
  state.scale_ = eval_mod_->start_scale_;
  state.num_slots_ = cts[0]->GetNumSlots();
  const size_t poly_bytes = state.PolyWords() * sizeof(word);
  for (int b = 0; b < batch; b++) {
    const Ct &ct = *cts[b];
    AssertTrue(ct.GetNP() == np,
               "EvaluateModBatch: the group must share one level");
    AssertTrue(!ct.HasRx(), "EvaluateModBatch: relinearization required");
    this->AssertSameScale(ct, eval_mod_->start_scale_);
    cudaMemcpyAsync(state.CtData(b), ct.bx_.data(), poly_bytes,
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    cudaMemcpyAsync(state.CtData(b) + state.PolyWords(), ct.ax_.data(),
                    poly_bytes, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  }

  CtBatch<word> out;
  eval_mod_->EvaluateBatch(GetContext(), out, state, mult_key);
  this->AssertSameScale(out.scale_, eval_mod_->end_scale_);

  // Scatter the results back into the callers' ciphertexts.
  const size_t out_bytes = out.PolyWords() * sizeof(word);
  for (int b = 0; b < batch; b++) {
    Ct &ct = *cts[b];
    const int num_slots = ct.GetNumSlots();
    ct.RemoveRx();
    ct.ModifyNP(out.np_);
    ct.SetScale(eval_mod_->end_scale_);
    ct.SetNumSlots(num_slots);
    cudaMemcpyAsync(ct.bx_.data(), out.CtData(b), out_bytes,
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    cudaMemcpyAsync(ct.ax_.data(), out.CtData(b) + out.PolyWords(), out_bytes,
                    cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  }
}

template <typename word>
void BootContext<word>::HalfBootModuleBatch(
    std::vector<Ct> &res, const std::vector<const Ct *> &inputs,
    const EvkMap<word> &evk_map, const CiModuleBasis<word> &basis) const {
  NvtxScope _nv("boot: HalfBootModuleBatch");
  AssertTrue(this->param_.conjugate_invariant_,
             "HalfBootModuleBatch: the module basis lives on R+, and the "
             "batched EvalMod runs the real subring's single reduction");
  const int n = static_cast<int>(inputs.size());
  res.resize(n);
  std::vector<int> num_slots(n);
  std::vector<Ct *> ptrs(n);
  for (int i = 0; i < n; i++) {
    counts_.half++;
    num_slots[i] = HalfBootModuleFront(res[i], *inputs[i], evk_map, basis);
    res[i].SetScale(eval_mod_->start_scale_);
    ptrs[i] = &res[i];
  }
  EvaluateModBatch(ptrs, evk_map.GetMultiplicationKey());
  for (int i = 0; i < n; i++) {
    res[i].SetNumSlots(num_slots[i]);
    res[i].SetScale(eval_mod_->end_scale_);
  }
}

namespace {
bool HoistCtSerial() {
  static const bool serial = [] {
    const char *env = std::getenv("CHEDDAR_HOIST_CT_SERIAL");
    return env != nullptr && env[0] == '1';
  }();
  return serial;
}
}  // namespace

template <typename word>
void BootContext<word>::BootBatch(std::vector<Ct> &res,
                                  const std::vector<const Ct *> &inputs,
                                  const EvkMap<word> &evk_map,
                                  bool min_ks) const {
  NvtxScope _nv("boot: BootBatch");
  AssertTrue(this->param_.conjugate_invariant_,
             "BootBatch: the batched EvalMod runs the real subring's single "
             "reduction; the ordinary ring's axis split is not batched");
  const int n = static_cast<int>(inputs.size());
  res.resize(n);
  std::vector<int> num_slots(n);
  std::vector<Ct> slots(n);
  std::vector<Ct *> ptrs(n);

  // The batched conversions share one compiled table set, so the group must
  // share one slot count; a mixed group falls back to the per-ciphertext
  // conversions (CHEDDAR_HOIST_CT_SERIAL=1 forces that path, the A/B).
  bool ct_batched = !HoistCtSerial() && !min_ks && n > 1;
  for (int i = 1; ct_batched && i < n; i++) {
    ct_batched = (inputs[i]->GetNumSlots() == inputs[0]->GetNumSlots());
  }

  if (ct_batched) {
    std::vector<Ct> mains(n);
    std::vector<const Ct *> main_ptrs(n);
    for (int i = 0; i < n; i++) {
      counts_.full++;
      num_slots[i] = BootFrontPreCtS(mains[i], *inputs[i], evk_map);
      main_ptrs[i] = &mains[i];
    }
    CoeffToSlotBatch(slots, GetBootEnabledNumSlots(num_slots[0]), main_ptrs,
                     evk_map);
    mains.clear();
  } else {
    for (int i = 0; i < n; i++) {
      counts_.full++;
      num_slots[i] = BootFront(slots[i], *inputs[i], evk_map, min_ks);
    }
  }
  for (int i = 0; i < n; i++) {
    // What EvaluateModAfterCtS's conjugate-invariant branch does before its
    // single reduction.
    slots[i].SetScale(eval_mod_->start_scale_);
    ptrs[i] = &slots[i];
  }
  EvaluateModBatch(ptrs, evk_map.GetMultiplicationKey());

  if (ct_batched) {
    // BootBack's steps, with the group's StC as ONE batched conversion: the
    // slack's LevelDown per ciphertext, the conversion, the epilogue.
    const int group_num_slots = GetBootEnabledNumSlots(num_slots[0]);
    std::vector<double> leveldown_drift(n, 1.0);
    if (boot_param_.GetNumSlackLevels() > 0) {
      for (int i = 0; i < n; i++) {
        const double before = slots[i].GetScale();
        this->LevelDown(slots[i], slots[i], boot_param_.GetStCStartLevel());
        leveldown_drift[i] = slots[i].GetScale() / before;
      }
    }
    std::vector<const Ct *> slot_ptrs(n);
    for (int i = 0; i < n; i++) slot_ptrs[i] = &slots[i];
    SlotToCoeffBatch(res, group_num_slots, slot_ptrs, evk_map);
    for (int i = 0; i < n; i++) {
      if (boot_variant_.at(group_num_slots) ==
          BootVariant::kImaginaryRemoving) {
        this->HConjAdd(res[i], res[i], res[i], evk_map.GetConjugationKey());
      }
      res[i].SetNumSlots(num_slots[i]);
      res[i].SetScale(this->param_.GetScale(boot_param_.GetEndLevel()) *
                      leveldown_drift[i]);
      slots[i] = Ct();
    }
  } else {
    for (int i = 0; i < n; i++) {
      BootBack(res[i], slots[i], num_slots[i], evk_map, min_ks);
      slots[i] = Ct();
    }
  }
}

template <typename word>
void BootContext<word>::HalfBootTowerBatch(
    std::vector<Ct> &res, const std::vector<const Ct *> &inputs,
    const EvkMap<word> &evk_map, const CiSinCBasis<word> &basis) const {
  NvtxScope _nv("boot: HalfBootTowerBatch");
  AssertTrue(this->param_.conjugate_invariant_,
             "HalfBootTowerBatch: the tower basis lives on R+, and the "
             "batched EvalMod runs the real subring's single reduction");
  const int n = static_cast<int>(inputs.size());
  res.resize(n);
  std::vector<int> num_slots(n);
  std::vector<Ct *> ptrs(n);
  for (int i = 0; i < n; i++) {
    counts_.half++;
    num_slots[i] = HalfBootTowerFront(res[i], *inputs[i], evk_map, basis);
    res[i].SetScale(eval_mod_->start_scale_);
    ptrs[i] = &res[i];
  }
  EvaluateModBatch(ptrs, evk_map.GetMultiplicationKey());
  for (int i = 0; i < n; i++) {
    res[i].SetNumSlots(num_slots[i]);
    res[i].SetScale(eval_mod_->end_scale_);
  }
}

template <typename word>
void BootContext<word>::Trace(Ct &res, int start_rot_dist, int num_accum,
                              const Ct &input,
                              const EvkMap<word> &evk_map) const {
  int num_slots = input.GetNumSlots();
  AssertTrue(IsPowOfTwo(num_accum), "Num accum must be power of 2");
  int log_num_accum = Log2Ceil(num_accum);

  if (num_accum == 1) {
    this->Copy(res, input);
    return;
  }

  NPInfo np = input.GetNP();
  AssertTrue(np.num_aux_ == 0, "Trace: Aux primes are not allowed");

  res.RemoveRx();
  res.ModifyNP(np);
  res.SetNumSlots(num_slots);
  res.SetScale(input.GetScale());
  std::vector<DvView<word>> res_view = res.ViewVector();

  Ct tmp;
  for (int i = 0; i < log_num_accum; i++) {
    int rot_idx = (start_rot_dist * (1 << i)) % num_slots;
    if (rot_idx < 0) rot_idx += num_slots;
    const auto &evk = evk_map.GetRotationKey(rot_idx);
    if (i == 0) {
      // res = HRot(input, rot_idx) + input
      this->HRotAdd(res, input, input, evk, rot_idx);
    } else {
      // res += HRot(res, rot_idx)
      this->HRotAdd(res, res, res, evk, rot_idx);
    }
  }
}

template class BootContext<uint32_t>;
template class BootContext<uint64_t>;

}  // namespace cheddar
