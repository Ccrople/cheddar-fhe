#include "extension/LlamaBlock.h"

#include <cmath>
#include <iostream>
#include <sstream>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
LlamaBlock<word>::LlamaBlock(std::shared_ptr<const BootContext<word>> boot,
                             const Config &cfg, const Calibration &cal)
    : boot_{std::move(boot)}, cfg_{cfg}, cal_{cal} {
  AssertTrue(boot_ != nullptr, "LlamaBlock: no BootContext");
  num_slots_ = boot_->param_.degree_ / 2;
  AssertTrue(cfg_.num_tokens > 0 && num_slots_ % cfg_.num_tokens == 0,
             "LlamaBlock: the token count must divide the slot count");
  channels_per_ct_ = num_slots_ / cfg_.num_tokens;
  AssertTrue(cfg_.head_dim > 0 && channels_per_ct_ % cfg_.head_dim == 0,
             "LlamaBlock: the head dimension must divide the channels one "
             "ciphertext holds, or a head would straddle two of them");
  const std::vector<int> widths = {cfg_.num_channels, cfg_.num_kv_channels,
                                   cfg_.hidden};
  for (int ch : widths) {
    AssertTrue(ch % channels_per_ct_ == 0,
               "LlamaBlock: every tensor width must be a whole number of "
               "ciphertexts");
  }

  sched_ = std::make_unique<SylphSchedule<word>>(boot_, num_slots_);
  // Every operator runs one level below where the bootstrap lands, because
  // Canonicalise spends that level putting the ciphertext on the scale
  // EvalPoly asserts. See SylphSchedule.h.
  const int op_level = sched_->GetSlotLevel() - 1;

  // The layer constants are stated for the unscaled activation and the
  // ciphertext carries `residual`, so what reaches the polynomial is
  // alpha * mean((residual * x)^2) = (alpha / residual^2) * mean(x^2). The
  // epsilon moves the same way, and both are free -- the multiplicative half
  // of the affine map is a reinterpretation of the scale and the additive half
  // is a constant addition (RmsNorm.h).
  const double r2 = cal_.residual * cal_.residual;
  attn_norm_ = std::make_unique<RmsNormHandler<word>>(
      boot_, cfg_.num_tokens, cfg_.num_channels, cal_.attn_alpha / r2, op_level,
      cfg_.eps * r2, cal_.rms_window, cal_.rms_degree);
  ffn_norm_ = std::make_unique<RmsNormHandler<word>>(
      boot_, cfg_.num_tokens, cfg_.num_channels, cal_.ffn_alpha / r2, op_level,
      cfg_.eps * r2, cal_.rms_window, cal_.rms_degree);
  rope_ = std::make_unique<RoPeHandler<word>>(boot_, cfg_.num_tokens,
                                              cfg_.head_dim, op_level,
                                              cfg_.rope_theta);
  // The auxiliary track goes round this cycle, not through `Boot`. `Boot`
  // lands at GetEndLevel(), which the slack pushes below the main track, so
  // SoftMax's own boot_aux asserts on any set that reserves slack -- which is
  // every set this schedule runs on. The hook below is StC, HalfBoot and
  // Canonicalise, which is one whole turn of the cycle taken by a value that
  // is not the main track.
  softmax_ = std::make_unique<SoftMaxHandler<word>>(
      boot_, cfg_.num_tokens, cal_.softmax_range, op_level, cal_.softmax_iters,
      cal_.softmax_norm_lo, cal_.softmax_norm_hi, cal_.softmax_exp_degree,
      cal_.softmax_inv_sqrt_degree, cal_.softmax_early_inv_sqrt_degree,
      /*boot_aux=*/false, /*aux_return_level=*/op_level,
      /*aux_boot_max=*/cal_.boot_max);
  AssertTrue(softmax_->GetAuxCallLevel() == sched_->GetStCLevel(),
             "LlamaBlock: SoftMax's auxiliary value lands on level " +
                 std::to_string(softmax_->GetAuxCallLevel()) +
                 " but StC is compiled at " +
                 std::to_string(sched_->GetStCLevel()) +
                 ", so the cycle cannot pick it up. The exponential's degree "
                 "and the number of iterations set that level.");
  silu_ = std::make_unique<SiLuHandler<word>>(boot_, cal_.silu_range, op_level,
                                              cal_.silu_degree);
}

template <typename word>
int LlamaBlock<word>::NumCiphertexts(int channels) const {
  return channels / channels_per_ct_;
}

template <typename word>
void LlamaBlock<word>::AddRequiredRotations(EvkRequest &req) const {
  const int op_level = sched_->GetSlotLevel() - 1;
  boot_->AddRequiredRotations(req, num_slots_);
  for (int d : attn_norm_->GetRotationDistances()) req.AddRequest(d, op_level);
  for (int d : rope_->GetRotationDistances()) req.AddRequest(d, op_level);
  for (int d : softmax_->GetRotationDistances()) req.AddRequest(d, op_level);
}

// The six turns, as the schedule sees them. `Stage::linear_depth` is what the
// product leg spends below StC's output level; the product itself runs at the
// bottom of the ladder, so this is the descent plus the one level [BAE]'s
// PC-MM and [KANG]'s CC-MM each consume.
namespace {
template <typename word>
std::vector<typename SylphSchedule<word>::Stage> TurnsOf(
    const typename LlamaBlock<word>::Calibration &cal) {
  const int rms = 1 + Log2Ceil(cal.rms_degree + 1) + 2 + 1;
  // RmsNormHandler::Apply consumes one level for the square, ceil(log2(d+1))
  // for the polynomial and two more to apply the result and the weight -- and
  // then *owes* one, because it ends with Mult(res, res, weight_pt_) and no
  // Rescale. A schedule that counts only what an operator consumes puts the
  // ciphertext on StC's level still owing a rescale, and the failure is silent
  // and total (Doing.md 1.5aa). So the budget is consumed + owed.
  const int softmax = 1 + Log2Ceil(cal.softmax_exp_degree + 1) +
                      2 * cal.softmax_iters;
  const int silu = Log2Ceil(cal.silu_degree + 1) + 1;  // the fit, then * up
  using Stage = typename SylphSchedule<word>::Stage;
  return {
      Stage{"A  RMSNorm(attn) -> Q,K,V", rms, 2},
      Stage{"B  RoPE(Q), RoPE(K) -> S = QK^T", 1, 2},
      Stage{"C  SoftMax(S), carry V -> A = PV", softmax, 2},
      Stage{"D  carry A -> O = A W_o", 0, 2},
      Stage{"E  RMSNorm(ffn) -> G,U", rms, 2},
      Stage{"F  SiLU(G) * U -> Y = . W_down", silu, 2},
  };
}
}  // namespace

template <typename word>
bool LlamaBlock<word>::Fits(std::string *why) const {
  for (const auto &stage : TurnsOf<word>(cal_)) {
    if (!sched_->Fits(stage, why)) return false;
  }
  return true;
}

template <typename word>
std::string LlamaBlock<word>::DescribePlan() const {
  std::ostringstream os;
  os << sched_->DescribePlan(TurnsOf<word>(cal_));
  os << "packing: " << cfg_.num_tokens << " tokens x " << channels_per_ct_
     << " channels per ciphertext" << std::endl;
  os << "  hidden state " << NumCiphertexts(cfg_.num_channels) << " ct, kv "
     << NumCiphertexts(cfg_.num_kv_channels) << " ct, ffn "
     << NumCiphertexts(cfg_.hidden) << " ct, scores "
     << (cfg_.num_channels / cfg_.head_dim) * cfg_.num_tokens * cfg_.num_tokens /
            num_slots_
     << " ct" << std::endl;
  // Every ciphertext crossing a turn boundary is bootstrapped, including the
  // ones whose turn has no operator in it. That count is the block's cost.
  const int h = NumCiphertexts(cfg_.num_channels);
  const int kv = NumCiphertexts(cfg_.num_kv_channels);
  const int ffn = NumCiphertexts(cfg_.hidden);
  const int scores = (cfg_.num_channels / cfg_.head_dim) * cfg_.num_tokens *
                     cfg_.num_tokens / num_slots_;
  // Turn C bootstraps three things: the scores on the way in, V which was
  // produced two turns ago, and SoftMax's own auxiliary track, which takes a
  // whole turn of the cycle inside the operator.
  os << "bootstraps: A " << h << ", B " << (h + kv) << ", C "
     << (2 * scores + kv) << " (" << scores << " aux), D " << h << ", E " << h
     << ", F " << (2 * ffn) << " = "
     << (h + h + kv + 2 * scores + kv + h + h + 2 * ffn) << " per block"
     << std::endl;
  return os.str();
}

// A hundred bootstraps is minutes of silence, and an assert inside one of them
// says only that the prime counts differed. Naming the turn costs a line of
// output and turns "somewhere in the block" into "turn F, the gate multiply".
template <typename word>
void LlamaBlock<word>::Announce(const char *what, const std::vector<Ct> &cts,
                                int expect_level) const {
  const int level = cts.empty() ? -1 : boot_->param_.NPToLevel(cts[0].GetNP());
  std::cout << "  [block] " << what << ": " << cts.size() << " ct at level "
            << level;
  if (expect_level >= 0 && level != expect_level) {
    std::cout << "  (EXPECTED " << expect_level << ")";
  }
  std::cout << std::endl;
}

template <typename word>
void LlamaBlock<word>::Lift(std::vector<Ct> &res, const std::vector<Ct> &x,
                            double magnitude, const EvkMap<word> &evk_map,
                            double shift) const {
  res.resize(x.size());
  const int level = sched_->GetSlotLevel() - 1;
  Constant<word> shift_const;
  if (shift != 0.0) {
    boot_->encoder_.EncodeConstant(shift_const, level,
                                   boot_->param_.GetScale(level), shift);
  }
  for (size_t i = 0; i < x.size(); i++) {
    Ct landed;
    sched_->ToSlot(landed, x[i], evk_map);
    sched_->Canonicalise(res[i], landed, magnitude);
    if (shift != 0.0) boot_->Add(res[i], res[i], shift_const);
  }
}

template <typename word>
void LlamaBlock<word>::Lower(std::vector<Ct> &res, const std::vector<Ct> &x,
                             const EvkMap<word> &evk_map) const {
  res.resize(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    sched_->ToCoeff(res[i], x[i], evk_map);
  }
}

template <typename word>
void LlamaBlock<word>::SpreadNormWeight(
    std::vector<std::vector<Complex>> &res, const std::vector<double> &w,
    double alpha) const {
  const int num_ct = NumCiphertexts(cfg_.num_channels);
  // RmsNormHandler asks for sqrt(alpha_L) folded into the weight, because the
  // circuit evaluates 1/sqrt(alpha * mean(x^2)) and the missing sqrt(alpha) has
  // to come back somewhere. The weight plaintext is where it costs nothing.
  const double root = std::sqrt(alpha);
  res.assign(num_ct, std::vector<Complex>(num_slots_, Complex(0.0, 0.0)));
  for (int i = 0; i < num_ct; i++) {
    for (int s = 0; s < num_slots_; s++) {
      const int c = i * channels_per_ct_ + s / cfg_.num_tokens;
      res[i][s] = Complex(w[c] * root, 0.0);
    }
  }
}

template <typename word>
void LlamaBlock<word>::Run(std::vector<Ct> &res, const std::vector<Ct> &x,
                           const Weights &w,
                           const std::vector<std::vector<Complex>> &causal_mask,
                           const LinearLeg &leg,
                           const EvkMap<word> &evk_map) const {
  std::string why;
  AssertTrue(Fits(&why), "LlamaBlock: the plan does not close -- " + why);
  AssertTrue(static_cast<int>(x.size()) == NumCiphertexts(cfg_.num_channels),
             "LlamaBlock: wrong number of input ciphertexts");

  const double r = cal_.residual;

  // ---- turn A: RMSNorm(attn), then the QKV projection -------------------
  //
  // The input carries `residual`, and RMSNorm is invariant to it -- that is
  // exactly why the layer constant was divided by residual^2 in the
  // constructor. So the normalised activation is the true one and every
  // crossing constant below is stated against unscaled tensors.
  std::vector<Ct> slots, normed, coeff;
  Announce("A  input", x, 0);
  Lift(slots, x, 1.0, evk_map);
  Announce("A  lifted", slots, sched_->GetSlotLevel() - 1);
  std::vector<std::vector<Complex>> attn_w;
  SpreadNormWeight(attn_w, w.attn_norm, cal_.attn_alpha / (r * r));
  attn_norm_->Apply(normed, slots, attn_w, evk_map);
  Announce("A  RMSNorm", normed, sched_->GetStCLevel());
  Lower(coeff, normed, evk_map);
  Announce("A  lowered", coeff, sched_->GetCoeffLevel());

  std::vector<Ct> q, k, v;
  leg.Project(q, coeff, cfg_.num_channels, cfg_.num_channels, w.wq, cal_.size_q,
              "Q");
  leg.Project(k, coeff, cfg_.num_channels, cfg_.num_kv_channels, w.wk,
              cal_.size_k, "K");
  leg.Project(v, coeff, cfg_.num_channels, cfg_.num_kv_channels, w.wv,
              cal_.size_v, "V");

  // ---- turn B: RoPE on Q and K, then the score product ------------------
  //
  // RoPE is linear, so it commutes with the crossing constant and the
  // ciphertexts are lifted at magnitude 1 -- whatever they carry, they carry
  // it through unchanged and `Scores` divides both out.
  std::vector<Ct> q_slots, k_slots;
  Announce("B  Q from the projection", q, 0);
  Lift(q_slots, q, 1.0, evk_map);
  Lift(k_slots, k, 1.0, evk_map);
  std::vector<Ct> q_rot(q_slots.size()), k_rot(k_slots.size());
  for (size_t i = 0; i < q_slots.size(); i++) {
    rope_->Apply(q_rot[i], q_slots[i], cfg_.first_position, evk_map);
  }
  for (size_t i = 0; i < k_slots.size(); i++) {
    rope_->Apply(k_rot[i], k_slots[i], cfg_.first_position, evk_map);
  }
  Announce("B  RoPE", q_rot, sched_->GetSlotLevel() - 2);
  std::vector<Ct> q_coeff, k_coeff;
  Lower(q_coeff, q_rot, evk_map);
  Lower(k_coeff, k_rot, evk_map);

  // What SoftMax wants is 2 (s - c) / M + 1 on [-1, 1], and what the
  // bootstrap can carry is at most boot_max. The affine map is therefore split
  // three ways, and none of the three costs a level: the product carries
  // (s - c) * size_scores / M, which is small enough to cross; Canonicalise
  // multiplies by 2 / size_scores on its way through; and the +1 is a constant
  // addition in the slot domain.
  //
  // The subtraction of c happens inside the product because there it is a
  // plaintext whose coefficients are all equal -- free -- while in the slot
  // domain it would have to wait until after the crossing, where the magnitude
  // it would add is exactly what the crossing has no room for.
  std::vector<Ct> scores;
  const double score_scale = cal_.size_scores /
                             (cal_.softmax_range * std::sqrt(cfg_.head_dim) *
                              cal_.size_q * cal_.size_k);
  const int T = cfg_.num_tokens;
  const int num_rows = (cfg_.num_channels / cfg_.head_dim) * T;
  AssertTrue(static_cast<int>(cal_.softmax_shift.size()) == num_rows,
             "LlamaBlock: SoftMax needs one calibrated shift per row -- see "
             "Calibration::softmax_shift for why a single global one does not "
             "work on real scores");
  std::vector<double> score_shift(static_cast<size_t>(num_rows) * T);
  for (int rr = 0; rr < num_rows; rr++) {
    const int query = rr % T;
    for (int key = 0; key < T; key++) {
      // Masked entries get pushed down by the calibrated gap so that the
      // exponential still sees an in-interval argument; the mask zeroes them
      // a level later, but the polynomial has already run by then.
      const double extra = (key <= query) ? 0.0 : cal_.softmax_mask_shift;
      score_shift[static_cast<size_t>(rr) * T + key] =
          -cal_.size_scores * (cal_.softmax_shift[rr] + extra) /
          cal_.softmax_range;
    }
  }
  leg.Scores(scores, q_coeff, k_coeff, score_scale, score_shift);

  // ---- turn C: SoftMax, and V carried across ----------------------------
  std::vector<Ct> score_slots;
  Lift(score_slots, scores, 2.0 / cal_.size_scores, evk_map, /*shift=*/1.0);
  AssertTrue(causal_mask.size() == score_slots.size(),
             "LlamaBlock: one causal mask per score ciphertext is required");
  typename SoftMaxHandler<word>::AuxBoot aux =
      [this, &evk_map](Ct &out, const Ct &in, double magnitude) {
        Ct coeff, landed;
        sched_->ToCoeff(coeff, in, evk_map);
        sched_->ToSlot(landed, coeff, evk_map);
        sched_->Canonicalise(out, landed, magnitude);
      };
  Announce("C  scores lifted", score_slots, sched_->GetSlotLevel() - 1);
  std::vector<Ct> probs(score_slots.size());
  for (size_t i = 0; i < score_slots.size(); i++) {
    softmax_->Apply(probs[i], score_slots[i], causal_mask[i], evk_map, nullptr,
                    &aux);
  }
  Announce("C  SoftMax", probs, sched_->GetStCLevel());
  std::vector<Ct> prob_coeff;
  Lower(prob_coeff, probs, evk_map);

  // V was produced two turns ago and left at level 0, which is below the
  // product's input level, so it is bootstrapped here with nothing in its slot
  // leg. That transport is what the two-level product ring costs; it is not
  // recoverable by reordering.
  std::vector<Ct> v_slots, v_coeff;
  Lift(v_slots, v, 1.0, evk_map);
  Lower(v_coeff, v_slots, evk_map);

  std::vector<Ct> attn;
  leg.Values(attn, prob_coeff, v_coeff, cal_.size_attn / cal_.size_v);

  // ---- turn D: carry the attention output into the O projection ---------
  //
  // O is a second product and the product ring holds one level, so it cannot
  // follow PV directly. `residual` rather than a private constant, because
  // this output is added to the block input.
  std::vector<Ct> attn_slots, attn_coeff, attn_out;
  Announce("D  attention values", attn, 0);
  Lift(attn_slots, attn, 1.0, evk_map);
  Lower(attn_coeff, attn_slots, evk_map);
  leg.Project(attn_out, attn_coeff, cfg_.num_channels, cfg_.num_channels, w.wo,
              r / cal_.size_attn, "O");

  std::vector<Ct> h(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    boot_->Add(h[i], x[i], attn_out[i]);
  }

  // ---- turn E: RMSNorm(ffn), then the gate and up projections -----------
  std::vector<Ct> h_slots, h_normed, h_coeff;
  Lift(h_slots, h, 1.0, evk_map);
  std::vector<std::vector<Complex>> ffn_w;
  SpreadNormWeight(ffn_w, w.ffn_norm, cal_.ffn_alpha / (r * r));
  ffn_norm_->Apply(h_normed, h_slots, ffn_w, evk_map);
  Announce("E  RMSNorm", h_normed, sched_->GetStCLevel());
  Lower(h_coeff, h_normed, evk_map);

  std::vector<Ct> gate, up;
  // SiLU is the one operator whose answer depends on the magnitude of its
  // argument, so 1/range is not optional bookkeeping -- it is part of the
  // function. It folds into the gate projection's plaintext, which is where
  // SiLu.h says it belongs, and the crossing constant rides along with it.
  leg.Project(gate, h_coeff, cfg_.num_channels, cfg_.hidden, w.wgate,
              cal_.size_gate / cal_.silu_range, "gate");
  leg.Project(up, h_coeff, cfg_.num_channels, cfg_.hidden, w.wup, cal_.size_up,
              "up");

  // ---- turn F: SiLU, the elementwise gate, then the down projection -----
  //
  // The gate is grown back to exactly the interval the polynomial was fitted
  // on. The bootstrap before it saw size_gate/range * g, at most boot_max.
  std::vector<Ct> gate_slots, up_slots;
  Announce("F  gate", gate, 0);
  Lift(gate_slots, gate, 1.0 / cal_.size_gate, evk_map);
  Lift(up_slots, up, 1.0, evk_map);
  const auto &mult_key = evk_map.GetMultiplicationKey();
  std::vector<Ct> gated(gate_slots.size());
  for (size_t i = 0; i < gate_slots.size(); i++) {
    Ct activated;
    silu_->Apply(activated, gate_slots[i], evk_map);
    // `up` was lifted alongside the gate and has spent nothing since, while
    // SiLU's polynomial has descended ceil(log2(degree+1)) levels. Cheddar's
    // HMult requires the two operands at the same level and reports the
    // mismatch as "Number of primes differ" from inside the tensor product,
    // which names neither operand -- so bring `up` down explicitly.
    const int level = boot_->param_.NPToLevel(activated.GetNP());
    Ct levelled;
    boot_->LevelDown(levelled, up_slots[i], level);
    boot_->HMult(gated[i], activated, levelled, mult_key);
  }
  Announce("F  SiLU * up", gated, -1);
  std::vector<Ct> gated_coeff, ffn_out;
  Lower(gated_coeff, gated, evk_map);
  leg.Project(ffn_out, gated_coeff, cfg_.hidden, cfg_.num_channels, w.wdown,
              r / cal_.size_up, "down");

  res.resize(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    boot_->Add(res[i], h[i], ffn_out[i]);
  }
}

template <typename word>
void LlamaBlock<word>::PlainRun(std::vector<double> &res,
                                const std::vector<double> &x,
                                const Weights &w) const {
  const int T = cfg_.num_tokens;
  const int H = cfg_.num_channels;
  const int KV = cfg_.num_kv_channels;
  const int D = cfg_.head_dim;
  const int F = cfg_.hidden;
  const int heads = H / D;
  const int kv_heads = KV / D;
  const int group = heads / kv_heads;

  auto norm = [&](const std::vector<double> &in, const std::vector<double> &g,
                  std::vector<double> &out) {
    out.assign(static_cast<size_t>(T) * H, 0.0);
    for (int t = 0; t < T; t++) {
      double sq = 0.0;
      for (int c = 0; c < H; c++) {
        const double u = in[static_cast<size_t>(t) * H + c];
        sq += u * u;
      }
      const double inv = 1.0 / std::sqrt(sq / H + cfg_.eps);
      for (int c = 0; c < H; c++) {
        out[static_cast<size_t>(t) * H + c] =
            in[static_cast<size_t>(t) * H + c] * inv * g[c];
      }
    }
  };
  auto gemm = [&](const std::vector<double> &a, const std::vector<double> &b,
                  int inner, int outer, std::vector<double> &out) {
    out.assign(static_cast<size_t>(T) * outer, 0.0);
    for (int t = 0; t < T; t++) {
      for (int i = 0; i < inner; i++) {
        const double av = a[static_cast<size_t>(t) * inner + i];
        if (av == 0.0) continue;
        const double *bp = &b[static_cast<size_t>(i) * outer];
        double *op = &out[static_cast<size_t>(t) * outer];
        for (int o = 0; o < outer; o++) op[o] += av * bp[o];
      }
    }
  };
  auto rope = [&](std::vector<double> &m, int width) {
    const int half = D / 2;
    std::vector<double> src(m);
    for (int t = 0; t < T; t++) {
      const double p = cfg_.first_position + t;
      for (int c = 0; c < width; c++) {
        const int j = c % D;
        const int base = c - j;
        const double f =
            std::pow(cfg_.rope_theta, -2.0 * (j % half) / static_cast<double>(D));
        const double cs = std::cos(p * f), sn = std::sin(p * f);
        const size_t idx = static_cast<size_t>(t) * width + c;
        const size_t partner =
            static_cast<size_t>(t) * width + base + ((j < half) ? j + half : j - half);
        m[idx] = src[idx] * cs + (j < half ? -1.0 : 1.0) * src[partner] * sn;
      }
    }
  };

  std::vector<double> normed;
  norm(x, w.attn_norm, normed);
  std::vector<double> q, k, v;
  gemm(normed, w.wq, H, H, q);
  gemm(normed, w.wk, H, KV, k);
  gemm(normed, w.wv, H, KV, v);
  rope(q, H);
  rope(k, KV);

  std::vector<double> attn(static_cast<size_t>(T) * H, 0.0);
  const double inv_root = 1.0 / std::sqrt(static_cast<double>(D));
  for (int hd = 0; hd < heads; hd++) {
    const int kvh = hd / group;
    for (int t = 0; t < T; t++) {
      std::vector<double> s(T, 0.0);
      double best = -1e300;
      for (int u = 0; u <= t; u++) {
        double dot = 0.0;
        for (int d = 0; d < D; d++) {
          dot += q[static_cast<size_t>(t) * H + hd * D + d] *
                 k[static_cast<size_t>(u) * KV + kvh * D + d];
        }
        s[u] = dot * inv_root;
        best = std::max(best, s[u]);
      }
      double sum = 0.0;
      for (int u = 0; u <= t; u++) {
        s[u] = std::exp(s[u] - best);
        sum += s[u];
      }
      for (int u = 0; u <= t; u++) {
        const double p = s[u] / sum;
        for (int d = 0; d < D; d++) {
          attn[static_cast<size_t>(t) * H + hd * D + d] +=
              p * v[static_cast<size_t>(u) * KV + kvh * D + d];
        }
      }
    }
  }

  std::vector<double> proj;
  gemm(attn, w.wo, H, H, proj);
  std::vector<double> hidden(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < hidden.size(); i++) hidden[i] = x[i] + proj[i];

  std::vector<double> fnormed;
  norm(hidden, w.ffn_norm, fnormed);
  std::vector<double> g, u;
  gemm(fnormed, w.wgate, H, F, g);
  gemm(fnormed, w.wup, H, F, u);
  std::vector<double> act(g.size());
  for (size_t i = 0; i < g.size(); i++) {
    act[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
  }
  std::vector<double> down;
  gemm(act, w.wdown, F, H, down);
  res.assign(hidden.size(), 0.0);
  for (size_t i = 0; i < res.size(); i++) res[i] = hidden[i] + down[i];
}

template class LlamaBlock<uint32_t>;
template class LlamaBlock<uint64_t>;

}  // namespace cheddar
