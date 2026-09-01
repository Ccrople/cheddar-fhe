#include "extension/CiBatchLayer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/EncodeGpu.h"
#include "extension/ChebyshevFit.h"
#include "extension/Profile.h"
#include "extension/SiLu.h"

namespace cheddar {

#ifdef USE_CUBLAS

namespace {
using Clock = std::chrono::steady_clock;
double SinceSeconds(Clock::time_point t0) {
  cudaDeviceSynchronize();
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
}  // namespace

template <typename word>
CiBatchLayer<word>::CiBatchLayer(std::shared_ptr<const BootContext<word>> boot,
                                 const Config &cfg)
    : boot_{std::move(boot)},
      cfg_{cfg},
      layout_{boot_->param_.MaxNumSlots(), cfg.num_tokens} {
  AssertTrue(boot_->param_.conjugate_invariant_,
             "CiBatchLayer: the batched layout is written for R+");
  AssertTrue(cfg_.hidden % cfg_.rows_per_tile == 0,
             "CiBatchLayer: rows_per_tile must divide the hidden width");
  AssertTrue(cfg_.norm_apply_level >= 1,
             "CiBatchLayer: norm_apply_level must leave a level for the "
             "apply");
  typename CiBatchProjection<word>::Config pcfg;
  pcfg.rows_per_tile = cfg_.rows_per_tile;
  pcfg.verbose = cfg_.verbose;
  proj_ = std::make_unique<CiBatchProjection<word>>(boot_, pcfg);
}

template <typename word>
void CiBatchLayer<word>::AddRequiredRotations(EvkRequest &req) const {
  boot_->AddRequiredRotations(req, layout_.num_slots);
}

template <typename word>
int CiBatchLayer<word>::NormDegree(double window) const {
  if (cfg_.rms_degree > 0) return cfg_.rms_degree;
  if (window <= 2.5) return 9;
  if (window <= 12.0) return 15;
  return 31;
}

template <typename word>
void CiBatchLayer<word>::NormTurn(typename CiBatchProjection<word>::Source &src,
                                  const std::vector<Ct> &stream, double alpha,
                                  double window,
                                  const std::vector<double> &sink,
                                  double stream_scale,
                                  const EvkMap<word> &evk) const {
  NvtxScope _nv("batch: NormTurn");
  const int model = cfg_.model;
  const int T = cfg_.num_tokens;
  AssertTrue(static_cast<int>(stream.size()) == model,
             "CiBatchLayer::NormTurn: the stream is " + std::to_string(model) +
                 " ciphertexts");
  AssertTrue(sink.empty() || static_cast<int>(sink.size()) == T,
             "CiBatchLayer::NormTurn: one sink factor per token");
  AssertTrue(alpha > 0.0 && window > 1.0 && stream_scale > 0.0,
             "CiBatchLayer::NormTurn: bad calibration");
  const auto &mult_key = evk.GetMultiplicationKey();
  const Parameter<word> &param = boot_->param_;
  const int top = boot_->GetBootParameter().GetEndLevel();
  const int hold = cfg_.norm_apply_level;
  AssertTrue(hold + 1 <= top,
             "CiBatchLayer::NormTurn: norm_apply_level is above the boot's "
             "landing");

  // Pass A: every channel booted, its square accumulated WITHOUT
  // relinearization (the tensor product's three components add), and the
  // channel itself either kept at the level the apply will meet it or
  // dropped to be booted again (`Config::hold_channels`).
  auto t0 = Clock::now();
  std::vector<Ct> xs(cfg_.hold_channels ? model : 0);
  Ct acc;
  {
    NvtxScope _a("batch: norm pass A");
    for (int c = 0; c < model; c++) {
      Ct up;
      boot_->Boot(up, stream[c], evk);
      Ct sq;
      boot_->Mult(sq, up, up);
      if (c == 0) {
        acc = std::move(sq);
      } else {
        boot_->Add(acc, acc, sq);
      }
      if (cfg_.hold_channels) boot_->LevelDown(xs[c], up, hold);
    }
  }
  stages_.boot += SinceSeconds(t0);
  t0 = Clock::now();

  // ONE relinearization for the whole channel sum.
  Ct s2;
  boot_->RelinearizeRescale(s2, acc, mult_key);
  acc = Ct();

  // The affine map onto the polynomial's domain, with the per-token sink
  // and the stream's factor folded into its one multiply:
  //     u = alpha * (sink_t^2 * S / (H * s^2) + eps),   v = (u - b) / a
  // where S is the sum the ciphertexts hold (stream_scale^2 times the
  // model's), a and b the window's half-width and centre.
  const double wl = 1.0 / std::sqrt(window), wh = std::sqrt(window);
  const double a = 0.5 * (wh - wl), b = 0.5 * (wh + wl);
  const int l2 = param.NPToLevel(s2.GetNP());
  std::vector<double> kt(T);
  for (int t = 0; t < T; t++) {
    const double s = sink.empty() ? 1.0 : sink[t];
    kt[t] = alpha * s * s / (static_cast<double>(model) * stream_scale *
                             stream_scale * a);
  }
  std::vector<Complex> msg;
  Pt pk;
  layout_.PackPerToken(msg, kt);
  boot_->gpu_encoder_.Encode(pk, l2, param.GetScale(l2), msg);
  Ct v;
  {
    Ct tmp;
    boot_->Mult(tmp, s2, pk);
    boot_->Rescale(v, tmp);
  }
  const int lv = param.NPToLevel(v.GetNP());
  {
    Constant<word> shift;
    boot_->encoder_.EncodeConstant(shift, lv, param.GetScale(lv),
                                   (alpha * cfg_.eps - b) / a);
    boot_->Add(v, v, shift);
  }

  // The inverse square root, compiled at v's level and landing canonical.
  const int degree = NormDegree(window);
  auto coeffs = chebfit::Interpolate(
      [a, b](double t) { return 1.0 / std::sqrt(a * t + b); }, degree);
  const double in_scale = param.GetScale(lv);
  const int used =
      EvalPoly<word>(coeffs, lv, in_scale, in_scale, true).GetPolyDegree();
  const int lr = lv - Log2Ceil(used + 1);
  AssertTrue(lr >= hold + (sink.empty() ? 0 : 1),
             "CiBatchLayer::NormTurn: the inverse square root lands below "
             "norm_apply_level");
  EvalPoly<word> inv(coeffs, lv, in_scale, param.GetScale(lr), true);
  inv.Compile(boot_);
  Ct r;
  inv.Evaluate(boot_, r, v, mult_key);

  // The per-token factor onto r: y = (sink x) r = x (sink r).
  Ct rs;
  if (sink.empty()) {
    rs = std::move(r);
  } else {
    Pt ps;
    layout_.PackPerToken(msg, sink);
    boot_->gpu_encoder_.Encode(ps, lr, param.GetScale(lr), msg);
    Ct tmp;
    boot_->Mult(tmp, r, ps);
    boot_->Rescale(rs, tmp);
  }
  Ct rh;
  boot_->LevelDown(rh, rs, hold);

  // Pass B: the apply, one relinearization per channel, each normalised
  // channel split into the projection source the moment it exists.
  stages_.norm += SinceSeconds(t0);
  t0 = Clock::now();
  proj_->BeginSplit(src, model, hold - 1, layout_.num_slots);
  {
    NvtxScope _b("batch: norm pass B");
    for (int c = 0; c < model; c++) {
      Ct x_c;
      if (cfg_.hold_channels) {
        x_c = std::move(xs[c]);
      } else {
        Ct up;
        boot_->Boot(up, stream[c], evk);
        boot_->LevelDown(x_c, up, hold);
      }
      Ct y_c;
      boot_->HMult(y_c, x_c, rh, mult_key);
      proj_->AddColumn(src, c, y_c);
    }
  }
  if (cfg_.hold_channels) {
    stages_.norm += SinceSeconds(t0);
  } else {
    stages_.boot += SinceSeconds(t0);
  }
  if (cfg_.verbose) {
    std::cout << "  [batch] NormTurn: window " << window << " degree " << used
              << ", r at level " << lr << ", y at level " << (hold - 1)
              << ", split " << (cfg_.hold_channels ? "from held channels"
                                                    : "from a second boot")
              << std::endl;
  }
}

template <typename word>
void CiBatchLayer<word>::FeedForward(std::vector<Ct> &res,
                                     const std::vector<Ct> &stream,
                                     const Weights &w, const Calibration &c,
                                     const EvkMap<word> &evk) {
  NvtxScope _nv("batch: FeedForward");
  const int model = cfg_.model, hidden = cfg_.hidden;
  const Parameter<word> &param = boot_->param_;
  const auto &mult_key = evk.GetMultiplicationKey();
  AssertTrue(w.gate != nullptr && w.up != nullptr && w.down != nullptr,
             "CiBatchLayer::FeedForward: the three tensors are needed");
  AssertTrue(static_cast<int>(w.ffn_norm.size()) == model,
             "CiBatchLayer::FeedForward: one gain per model channel");
  AssertTrue(c.silu_range > 0.0, "CiBatchLayer::FeedForward: SiLU range");
  stages_ = Stages{};
  auto t_all = Clock::now();

  // 1. The norm, straight into the split the projections read.
  typename CiBatchProjection<word>::Source src_y;
  NormTurn(src_y, stream, c.alpha, c.norm_window, c.ffn_sink, c.stream_scale,
           evk);
  const int ly = src_y.level;

  // 2. The weights: the gain on gate and up, 1/range on gate so SiLU's
  //    polynomial sees [-1, 1], and the stream's factor on down so that its
  //    output adds to the stream as it is.
  auto t0 = Clock::now();
  {
    NvtxScope _w("batch: ffn weights");
    std::vector<double> gg(model);
    for (int i = 0; i < model; i++) gg[i] = w.ffn_norm[i] / c.silu_range;
    DeviceVector<float> gate_f, up_f;
    CiBatchProjection<word>::FoldGain(gate_f, w.gate, model, hidden, gg);
    CiBatchProjection<word>::FoldGain(up_f, w.up, model, hidden, w.ffn_norm);
    proj_->Prepare("ffn.gate", gate_f.data(), model, hidden, ly);
    proj_->Prepare("ffn.up", up_f.data(), model, hidden, ly);
  }
  // SiLU on the gate at ly - 1; the product one below its output; down one
  // below that.
  const int lg = ly - 1;
  SiLuHandler<word> silu(boot_, 1.0, lg, cfg_.silu_degree);
  int lh = -1;  // the level the product lands on, read off the first chunk
  const int chunk = cfg_.rows_per_tile;
  const int num_chunks = hidden / chunk;
  double t_weights = SinceSeconds(t0);

  // 3. The split of y is shared by gate and up over every chunk.
  std::vector<Ct> d_acc;
  for (int j = 0; j < num_chunks; j++) {
    NvtxScope _c("batch: ffn chunk");
    t0 = Clock::now();
    std::vector<Ct> g, u;
    proj_->Project(g, src_y, "ffn.gate", j);
    proj_->Project(u, src_y, "ffn.up", j);
    stages_.gate_up += SinceSeconds(t0);

    t0 = Clock::now();
    std::vector<Ct> h(g.size());
    for (size_t i = 0; i < g.size(); i++) {
      Ct s;
      silu.Apply(s, g[i], evk);
      g[i] = Ct();
      const int ls = param.NPToLevel(s.GetNP());
      Ct ul;
      boot_->LevelDown(ul, u[i], ls);
      u[i] = Ct();
      boot_->HMult(h[i], s, ul, mult_key);
    }
    stages_.silu += SinceSeconds(t0);

    t0 = Clock::now();
    if (lh < 0) {
      lh = param.NPToLevel(h[0].GetNP());
      AssertTrue(lh >= 1,
                 "CiBatchLayer::FeedForward: no level left for the down "
                 "projection; raise norm_apply_level");
    }
    // The down operand for this chunk's rows: a contiguous row slice of
    // the `[hidden][model]` tensor, the stream's factor folded in.
    const std::string dn = "ffn.down." + std::to_string(j);
    proj_->Prepare(dn, w.down + static_cast<size_t>(j) * chunk * model, chunk,
                   model, lh, c.stream_scale);
    std::vector<Ct> d;
    proj_->Project(d, h, dn);
    proj_->Release(dn);
    h.clear();
    if (j == 0) {
      d_acc = std::move(d);
    } else {
      for (int i = 0; i < model; i++) boot_->Add(d_acc[i], d_acc[i], d[i]);
    }
    stages_.down += SinceSeconds(t0);
  }
  proj_->Release("ffn.gate");
  proj_->Release("ffn.up");
  src_y = typename CiBatchProjection<word>::Source();

  // 4. The residual, both sides at the lower of the two levels.
  t0 = Clock::now();
  const int ld = param.NPToLevel(d_acc[0].GetNP());
  const int ls = param.NPToLevel(stream[0].GetNP());
  const int lo = std::min(ld, ls);
  res.clear();
  res.resize(model);
  for (int i = 0; i < model; i++) {
    Ct a, b;
    boot_->LevelDown(a, stream[i], lo);
    boot_->LevelDown(b, d_acc[i], lo);
    boot_->Add(res[i], a, b);
    d_acc[i] = Ct();
  }
  stages_.down += SinceSeconds(t0);
  stages_.total = SinceSeconds(t_all);
  if (cfg_.verbose) {
    std::cout << "  [batch] FeedForward: y at " << ly << ", gate at " << lg
              << ", product at " << lh << ", down at " << ld
              << ", residual at " << lo << "; weights " << t_weights
              << " s, boot " << stages_.boot << " s, norm " << stages_.norm
              << " s, gate/up " << stages_.gate_up << " s, silu "
              << stages_.silu << " s, down " << stages_.down << " s, total "
              << stages_.total << " s" << std::endl;
  }
}

template class CiBatchLayer<uint32_t>;
template class CiBatchLayer<uint64_t>;

#endif  // USE_CUBLAS

}  // namespace cheddar
