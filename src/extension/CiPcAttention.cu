#include "extension/CiPcAttention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/EncodeGpu.h"
#include "extension/ChebyshevFit.h"

namespace cheddar {

namespace {

// The worst error of the degree-n Chebyshev interpolant of exp(hb (v - 1)) on
// [-1, 1], and the degree that reaches sixteen bits -- `CiBatchAttention`'s
// own rule, repeated rather than shared because the argument here is twice
// its: the public branch accumulates y^2 directly, so its exponent is
// m_eff / 2 where the encrypted branch's is m_eff / 4.
double ExpFitError(double hb, int n) {
  const auto c = chebfit::Interpolate(
      [hb](double v) { return std::exp(hb * (v - 1.0)); }, n);
  double worst = 0.0;
  for (int i = 0; i <= 2048; i++) {
    const double v = -1.0 + 2.0 * i / 2048.0;
    double b0 = 0.0, b1 = 0.0;
    for (int j = n; j >= 1; j--) {
      const double t = 2.0 * v * b0 - b1 + c[j];
      b1 = b0;
      b0 = t;
    }
    worst = std::max(worst,
                     std::abs(v * b0 - b1 + c[0] - std::exp(hb * (v - 1.0))));
  }
  return worst;
}

int ExpDegree(double hb) {
  for (int d : {7, 9, 15, 31}) {
    if (ExpFitError(hb, d) < std::pow(2.0, -16.0)) return d;
  }
  return 31;
}

}  // namespace

template <typename word>
CiPcAttention<word>::CiPcAttention(
    std::shared_ptr<const BootContext<word>> boot, const Config &cfg)
    : boot_{std::move(boot)},
      cfg_{cfg},
      layout_{cfg.num_tokens * cfg.num_instances, cfg.num_tokens},
      subring_{boot_->param_, boot_->encoder_} {
  const Parameter<word> &param = boot_->param_;
  AssertTrue(param.conjugate_invariant_,
             "CiPcAttention: the batched layout is conjugate-invariant");
  // The subring view IS the plain map, and it is the plain map at exactly one
  // sub-degree: `num_instances` lanes and `num_tokens` blocks have to fill
  // the ring between them, or the operand this class exists for is not a
  // subring element at all.
  AssertTrue(cfg_.num_tokens * cfg_.num_instances == param.degree_,
             "CiPcAttention: tokens x instances must be the ring's real "
             "slots -- the public operand is a subring element of exactly "
             "that sub-degree");
  AssertTrue(IsPowOfTwo(cfg_.num_instances) && cfg_.num_instances >= 2,
             "CiPcAttention: the batch is the subring's sub-degree, so it is "
             "a power of two");
  AssertTrue(cfg_.head_dim > 0 && cfg_.chunk > 0,
             "CiPcAttention: head_dim and chunk must be positive");
  AssertTrue(cfg_.q_level >= 1 && cfg_.q_level <= param.max_level_,
             "CiPcAttention: q_level is not on this ring");
}

template <typename word>
void CiPcAttention<word>::Prepare(const Calibration &calib) {
  calib_ = calib;
  const Parameter<word> &param = boot_->param_;
  const int T = cfg_.num_tokens;
  AssertTrue(calib_.span > 0.0 && calib_.carried > 0.0,
             "CiPcAttention::Prepare: span and carried must be positive");
  AssertTrue(calib_.row_shift.empty() ||
                 static_cast<int>(calib_.row_shift.size()) == T,
             "CiPcAttention::Prepare: row_shift is one entry a QUERY token");
  AssertTrue(calib_.row_fold.empty() ||
                 static_cast<int>(calib_.row_fold.size()) == T,
             "CiPcAttention::Prepare: row_fold is one entry a QUERY token");

  a1_ = 2.0 / (calib_.span * calib_.carried);

  // w = y^2 = exp(m_eff (u - 1) / 2). The encrypted branch squares its y
  // after the fact and pays a relinearization for it; here the square is a
  // constant in the exponent and costs nothing.
  const double hb = calib_.m_eff / 2.0;
  const int degree =
      (calib_.exp_degree > 0) ? calib_.exp_degree : ExpDegree(hb);
  const int exp_in = cfg_.q_level - 1;
  auto coeffs = chebfit::Interpolate(
      [hb](double v) { return std::exp(hb * (v - 1.0)); }, degree);
  const int used = EvalPoly<word>(coeffs, exp_in, param.GetScale(exp_in),
                                  param.GetScale(exp_in), true)
                       .GetPolyDegree();
  exp_out_ = exp_in - Log2Ceil(used + 1);
  AssertTrue(exp_out_ >= 2,
             "CiPcAttention::Prepare: the exp exhausts the level budget "
             "before the value product and the fold");
  exp_ = std::make_unique<EvalPoly<word>>(coeffs, exp_in,
                                          param.GetScale(exp_in),
                                          param.GetScale(exp_out_), true);
  exp_->Compile(boot_);

  // The affine's ADD, per query token, at the level the scores land on. The
  // multiply is not here: it is a constant, so it rides the key weights'
  // encode (see `EncodeKeys`) and costs no level at all.
  std::vector<double> a0(T);
  for (int t = 0; t < T; t++) {
    const double s = calib_.row_shift.empty() ? calib_.shift
                                              : calib_.row_shift[t];
    a0[t] = 1.0 - 2.0 * s / calib_.span;
  }
  std::vector<Complex> msg;
  layout_.PackPerToken(msg, a0);
  boot_->gpu_encoder_.Encode(a0_, exp_in, param.GetScale(exp_in), msg);

  // The mask fold, per query token. In the encrypted branch it rides every
  // KEY's mask because the mask is there anyway; here there is no causal mask
  // to ride -- every public token precedes every encrypted one -- so it is
  // applied once at the end instead of `ptok` times.
  has_fold_ = !calib_.row_fold.empty();
  if (has_fold_) {
    layout_.PackPerToken(msg, calib_.row_fold);
    boot_->gpu_encoder_.Encode(fold_, exp_out_ - 1,
                               param.GetScale(exp_out_ - 1), msg);
  }
  ready_ = true;
  if (cfg_.verbose) {
    std::cout << "  [pc] exp deg " << used << " @" << exp_in << ".."
              << exp_out_ << ", a1 " << a1_ << ", fold "
              << (has_fold_ ? "yes" : "no") << ", out @" << GetOutputLevel()
              << std::endl;
  }
}

template <typename word>
void CiPcAttention<word>::EncodeKeys(SubringWeights<word> &res,
                                     const std::vector<double> &k,
                                     int width) const {
  AssertTrue(ready_, "CiPcAttention: call Prepare first");
  AssertTrue(width > 0 && width <= cfg_.chunk,
             "CiPcAttention::EncodeKeys: width must be at most the chunk");
  const double scale = boot_->param_.GetScale(cfg_.q_level);
  // The affine's multiply, folded. Encoding K at `a1 * scale` makes the
  // plaintext integers `round(a1 * scale * K)`, which is the encoding of
  // `a1 * K` at `scale`; declaring the canonical scale afterwards is what
  // says so. No copy of the caller's array, and no level.
  subring_.EncodeWeightsReal(res, boot_->gpu_encoder_, cfg_.q_level,
                             a1_ * scale, k, cfg_.head_dim, width,
                             cfg_.num_instances);
  res.scale_ = scale;
}

template <typename word>
void CiPcAttention<word>::EncodeValues(SubringWeights<word> &res,
                                       const std::vector<double> &v,
                                       int width) const {
  AssertTrue(ready_, "CiPcAttention: call Prepare first");
  AssertTrue(width > 0 && width <= cfg_.chunk,
             "CiPcAttention::EncodeValues: width must be at most the chunk");
  // At the exp's landing, which is where the weights it multiplies are.
  subring_.EncodeWeightsReal(res, boot_->gpu_encoder_, exp_out_,
                             boot_->param_.GetScale(exp_out_), v, width,
                             cfg_.head_dim, cfg_.num_instances);
}

template <typename word>
void CiPcAttention<word>::Scores(std::vector<Ct> &res,
                                 const std::vector<Ct> &q,
                                 const SubringWeights<word> &keys) const {
  AssertTrue(ready_, "CiPcAttention: call Prepare first");
  AssertTrue(static_cast<int>(q.size()) == cfg_.head_dim,
             "CiPcAttention::Scores: one ciphertext a channel");
  AssertTrue(keys.GetColsIn() == cfg_.head_dim,
             "CiPcAttention::Scores: the contraction is over the channels");
  AssertTrue(keys.GetSubDegree() == cfg_.num_instances,
             "CiPcAttention::Scores: the key weights are not at the batch's "
             "sub-degree");
  AssertTrue(boot_->param_.NPToLevel(q[0].GetNP()) == cfg_.q_level,
             "CiPcAttention::Scores: the queries are not at q_level");
  subring_.Multiply(boot_, res, keys, q);
}

template <typename word>
void CiPcAttention<word>::Weights(std::vector<Ct> &w, std::vector<Ct> &scores,
                                  const EvkMap<word> &evk) const {
  AssertTrue(ready_, "CiPcAttention: call Prepare first");
  const auto &mult_key = evk.GetMultiplicationKey();
  const int exp_in = cfg_.q_level - 1;
  const double want = boot_->param_.GetScale(exp_in);
  w.clear();
  w.resize(scores.size());
  for (size_t p = 0; p < scores.size(); p++) {
    AssertTrue(boot_->param_.NPToLevel(scores[p].GetNP()) == exp_in,
               "CiPcAttention::Weights: the score product did not land one "
               "level below the queries");
    AssertTrue(std::abs(scores[p].GetScale() - want) <= 1e-9 * want,
               "CiPcAttention::Weights: the scores are off the canonical "
               "scale, so the affine's add would not line up");
    boot_->Add(scores[p], scores[p], a0_);
    exp_->Evaluate(boot_, w[p], scores[p], mult_key);
    scores[p] = Ct();
  }
  scores.clear();
}

template <typename word>
void CiPcAttention<word>::Accumulate(std::vector<Ct> &acc, Ct &sq,
                                     std::vector<Ct> &w,
                                     const SubringWeights<word> &values) const {
  AssertTrue(ready_, "CiPcAttention: call Prepare first");
  AssertTrue(!w.empty(), "CiPcAttention::Accumulate: no weights");
  AssertTrue(values.GetColsIn() == static_cast<int>(w.size()),
             "CiPcAttention::Accumulate: the value weights do not contract "
             "this chunk");
  AssertTrue(values.GetColsOut() == cfg_.head_dim,
             "CiPcAttention::Accumulate: one output ciphertext a channel");
  const bool first = acc.empty();

  // The denominator is a SUM OF CIPHERTEXTS -- no rotation, no key, no level
  // -- because the key axis is the ciphertext index. That is the batched
  // layout's own property and it is why the public branch never reduces.
  // The weights are read, not consumed (the value product below reads them
  // too), so the first term of the first chunk is a LevelDown to its own
  // level: a copy, a ciphertext not being copyable.
  size_t p = 0;
  if (first) {
    boot_->LevelDown(sq, w[0], exp_out_);
    p = 1;
  }
  for (; p < w.size(); p++) boot_->Add(sq, sq, w[p]);

  std::vector<Ct> prod;
  subring_.Multiply(boot_, prod, values, w);
  w.clear();
  if (first) {
    acc = std::move(prod);
  } else {
    AssertTrue(acc.size() == prod.size(),
               "CiPcAttention::Accumulate: the chunks disagree on the "
               "channel count");
    for (size_t c = 0; c < acc.size(); c++) {
      boot_->Add(acc[c], acc[c], prod[c]);
    }
  }
}

template <typename word>
void CiPcAttention<word>::Finish(std::vector<Ct> &acc, Ct &sq) const {
  AssertTrue(ready_, "CiPcAttention: call Prepare first");
  AssertTrue(!acc.empty(), "CiPcAttention::Finish: nothing was accumulated");
  // `sq` is one level above `acc` (it never went through the value product),
  // so it comes down to meet it before the fold and both land together.
  Ct down;
  boot_->LevelDown(down, sq, exp_out_ - 1);
  sq = std::move(down);
  if (!has_fold_) return;
  Ct t;
  boot_->Mult(t, sq, fold_);
  boot_->Rescale(sq, t);
  for (auto &c : acc) {
    Ct u;
    boot_->Mult(u, c, fold_);
    boot_->Rescale(c, u);
  }
}

template <typename word>
void CiPcAttention<word>::HeadGroup(
    const std::vector<std::vector<Ct> *> &acc, const std::vector<Ct *> &sq,
    const std::vector<const std::vector<Ct> *> &q, int pub_tokens,
    const ChunkSource &src, const EvkMap<word> &evk) const {
  AssertTrue(ready_, "CiPcAttention: call Prepare first");
  AssertTrue(pub_tokens > 0,
             "CiPcAttention::HeadGroup: an empty public context");
  const int heads = static_cast<int>(q.size());
  AssertTrue(heads > 0, "CiPcAttention::HeadGroup: an empty group");
  AssertTrue(static_cast<int>(acc.size()) == heads &&
                 static_cast<int>(sq.size()) == heads,
             "CiPcAttention::HeadGroup: one accumulator pair a query head");
  for (int h = 0; h < heads; h++) {
    AssertTrue(q[h] != nullptr && acc[h] != nullptr && sq[h] != nullptr,
               "CiPcAttention::HeadGroup: a null member");
    acc[h]->clear();
    *sq[h] = Ct();
  }

  const size_t lanes = static_cast<size_t>(cfg_.num_instances);
  std::vector<double> kbuf, vbuf;
  SubringWeights<word> kw, vw;
  for (int start = 0; start < pub_tokens; start += cfg_.chunk) {
    const int width = std::min(cfg_.chunk, pub_tokens - start);
    kbuf.assign(static_cast<size_t>(cfg_.head_dim) * width * lanes, 0.0);
    vbuf.assign(static_cast<size_t>(width) * cfg_.head_dim * lanes, 0.0);
    src(start, width, kbuf, vbuf);

    // ONE encode of the chunk for the whole group -- the reason this form
    // exists. Both stores stay alive across the group's exps; that is the
    // trade, and at chunk 128 it is 6.5 GiB.
    EncodeKeys(kw, kbuf, width);
    EncodeValues(vw, vbuf, width);
    for (int h = 0; h < heads; h++) {
      std::vector<Ct> s;
      Scores(s, *q[h], kw);
      std::vector<Ct> w;
      Weights(w, s, evk);
      Accumulate(*acc[h], *sq[h], w, vw);
    }
    kw = SubringWeights<word>();
    vw = SubringWeights<word>();
  }
  for (int h = 0; h < heads; h++) Finish(*acc[h], *sq[h]);
}

template <typename word>
void CiPcAttention<word>::Head(std::vector<Ct> &acc, Ct &sq,
                               const std::vector<Ct> &q, int pub_tokens,
                               const ChunkSource &src,
                               const EvkMap<word> &evk) const {
  HeadGroup({&acc}, {&sq}, {&q}, pub_tokens, src, evk);
}

template class CiPcAttention<uint32_t>;
template class CiPcAttention<uint64_t>;

}  // namespace cheddar
