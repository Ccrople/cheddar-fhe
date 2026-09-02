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
#include "core/MemoryPool.h"
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
CiBatchLayer<word>::CiBatchLayer(std::shared_ptr<BootContext<word>> boot,
                                 const Config &cfg)
    : boot_{std::move(boot)},
      cfg_{cfg},
      layout_{cfg.lanes > 0
                  ? CiBatchLayout(boot_->param_.MaxNumSlots(), cfg.num_tokens,
                                  cfg.lanes, cfg.rank)
                  : CiBatchLayout(boot_->param_.MaxNumSlots(),
                                  cfg.num_tokens)} {
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
void CiBatchLayer<word>::Park(ParkedStream &parked,
                              std::vector<Ct> &stream) const {
  NvtxScope _nv("batch: park the stream");
  const size_t n = stream.size();
  parked.bx.resize(n);
  parked.ax.resize(n);
  parked.np.resize(n);
  parked.scale.resize(n);
  parked.slots.resize(n);
  for (size_t i = 0; i < n; i++) {
    Ct &ct = stream[i];
    AssertFalse(ct.HasRx(), "CiBatchLayer::Park: a stream ciphertext has rx");
    parked.np[i] = ct.GetNP();
    parked.scale[i] = ct.GetScale();
    parked.slots[i] = ct.GetNumSlots();
    CopyDeviceToHost(parked.bx[i], ct.bx_);
    CopyDeviceToHost(parked.ax[i], ct.ax_);
    ct = Ct();
  }
}

template <typename word>
void CiBatchLayer<word>::Unpark(std::vector<Ct> &stream,
                                ParkedStream &parked) const {
  NvtxScope _nv("batch: unpark the stream");
  const size_t n = parked.bx.size();
  stream.clear();
  stream.resize(n);
  for (size_t i = 0; i < n; i++) {
    Ct &ct = stream[i];
    ct.RemoveRx();
    ct.ModifyNP(parked.np[i]);
    ct.SetScale(parked.scale[i]);
    ct.SetNumSlots(parked.slots[i]);
    CopyHostToDevice(ct.bx_, parked.bx[i]);
    CopyHostToDevice(ct.ax_, parked.ax[i]);
    parked.bx[i] = HostVector<word>();
    parked.ax[i] = HostVector<word>();
  }
  parked = ParkedStream();
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
                                  const EvkMap<word> &evk, bool hold_ch,
                                  int hold_level) const {
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
  const int hold = hold_level;
  AssertTrue(hold + 1 <= top,
             "CiBatchLayer::NormTurn: norm_apply_level is above the boot's "
             "landing");

  // The boot's tables, if the previous norm dropped them.
  auto t0 = Clock::now();
  if (!boot_->IsBootPrepared(layout_.num_slots)) {
    NvtxScope _p("batch: prepare boot tables");
    boot_->PrepareEvalSpecialFFT(layout_.num_slots);
    prepare_seconds_ += SinceSeconds(t0);
    t0 = Clock::now();
  }

  // The split's buffers FIRST -- twenty of a gigabyte -- so they sit below
  // everything the two passes allocate and free, rather than among it.
  proj_->BeginSplit(src, model, hold - 1, layout_.num_slots);

  // Pass A: every channel booted, its square accumulated WITHOUT
  // relinearization (the tensor product's three components add), and the
  // channel itself either kept at the level the apply will meet it or
  // dropped to be booted again (`Config::hold_channels`).
  std::vector<Ct> xs(hold_ch ? model : 0);
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
      if (hold_ch) boot_->LevelDown(xs[c], up, hold);
    }
  }
  // The last bootstrap of this norm, when the channels are held: the
  // tables can go now.
  if (hold_ch && cfg_.release_boot_tables) {
    boot_->ReleaseEvalSpecialFFT(layout_.num_slots);
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
  AssertTrue(lr >= hold + 1,
             "CiBatchLayer::NormTurn: the inverse square root lands below "
             "norm_apply_level + 1");
  EvalPoly<word> inv(coeffs, lv, in_scale, param.GetScale(lr), true);
  inv.Compile(boot_);
  Ct r;
  inv.Evaluate(boot_, r, v, mult_key);

  // The per-token factor onto r, and the stream's factor OUT: the channel
  // the apply reads still carries `stream_scale`, and the norm is to land
  // in the model's units -- y = (sink x) r = (s x) (sink r / s).
  Ct rs;
  {
    std::vector<double> f(T);
    for (int t = 0; t < T; t++) {
      f[t] = (sink.empty() ? 1.0 : sink[t]) / stream_scale;
    }
    Pt ps;
    layout_.PackPerToken(msg, f);
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
  {
    NvtxScope _b("batch: norm pass B");
    for (int c = 0; c < model; c++) {
      Ct x_c;
      if (hold_ch) {
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
  if (!hold_ch && cfg_.release_boot_tables) {
    boot_->ReleaseEvalSpecialFFT(layout_.num_slots);
  }
  if (hold_ch) {
    stages_.norm += SinceSeconds(t0);
  } else {
    stages_.boot += SinceSeconds(t0);
  }
  if (cfg_.verbose) {
    std::cout << "  [batch] NormTurn: window " << window << " degree " << used
              << ", r at level " << lr << ", y at level " << (hold - 1)
              << ", split " << (hold_ch ? "from held channels"
                                                    : "from a second boot")
              << std::endl;
  }
}

template <typename word>
void CiBatchLayer<word>::FeedForward(std::vector<Ct> &res,
                                     std::vector<Ct> &stream,
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
           evk, cfg_.hold_channels_ffn, cfg_.norm_apply_level);
  const int ly = src_y.level;
  ParkedStream parked;
  Park(parked, stream);
  if (cfg_.verbose) MemoryPool::Report("batch: after the norm");

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
  // The handler fits SiLU(range * v) on [-1, 1] and wants g / range as its
  // input, which the gate weight carries: so the range it is told is the
  // calibration's, not 1 (which would evaluate SiLU(g / range) -- bounded,
  // wrong).
  const int lg = ly - 1;
  SiLuHandler<word> silu(boot_, c.silu_range, lg, cfg_.silu_degree);
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
      if (cfg_.verbose) MemoryPool::Report("batch: before the first down");
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
  Unpark(stream, parked);
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

template <typename word>
void CiBatchLayer<word>::Attention(
    std::vector<Ct> &res, std::vector<Ct> &stream, const AttnWeights &w,
    const Calibration &c, CiBatchAttention<word> &attn,
    const typename CiBatchAttention<word>::Keys &akeys,
    const EvkMap<word> &evk) {
  NvtxScope _nv("batch: Attention");
  const int model = cfg_.model;
  const Parameter<word> &param = boot_->param_;
  const CiBatchLayout &alayout = attn.GetLayout();
  AssertTrue(alayout.num_tokens == layout_.num_tokens &&
                 alayout.num_slots == layout_.num_slots &&
                 alayout.lanes == layout_.lanes && alayout.rank == layout_.rank,
             "CiBatchLayer::Attention: the attention's layout is not this "
             "layer's (Config::lanes / rank must be the chain's)");
  AssertTrue(w.q != nullptr && w.k != nullptr && w.v != nullptr &&
                 w.o != nullptr,
             "CiBatchLayer::Attention: the four tensors are needed");
  AssertTrue(static_cast<int>(w.attn_norm.size()) == model,
             "CiBatchLayer::Attention: one gain per model channel");
  // The heads: read off the attention's configuration through its layout
  // is not possible, so the shapes are the layer's contract.
  const int T = cfg_.num_tokens;
  const int D = T;  // Algorithm 4 is square: head_dim = tokens
  const int heads = 32, kv_heads = 8;  // Llama-3-8B; asserted below
  AssertTrue(cfg_.rows_per_tile % D == 0 && D % 1 == 0,
             "CiBatchLayer::Attention: a projection tile must hold whole "
             "heads");
  const int heads_per_tile = cfg_.rows_per_tile / D;
  AssertTrue(heads % heads_per_tile == 0 && kv_heads % heads_per_tile == 0,
             "CiBatchLayer::Attention: the tile must divide the head counts");
  const int group = heads / kv_heads;
  stages_ = Stages{};
  auto t_all = Clock::now();

  // 1. The pre-attention norm, into the split the three projections read.
  typename CiBatchProjection<word>::Source src_y;
  NormTurn(src_y, stream, c.attn_alpha, c.attn_norm_window, c.attn_sink,
           c.stream_scale, evk, cfg_.hold_channels, cfg_.norm_apply_level_attn);
  const int ly = src_y.level;
  ParkedStream parked;
  Park(parked, stream);
  if (cfg_.verbose) MemoryPool::Report("batch: after the attention norm");

  // 2. The weights: the gain on all three, cq on Q and ck on K (the chain
  //    factor the softmax works in), and the projections at ly -> ly - 1 =
  //    the attention's rope_level.
  auto t0 = Clock::now();
  {
    NvtxScope _w("batch: attention weights");
    std::vector<double> gq(model), gk(model);
    for (int i = 0; i < model; i++) {
      gq[i] = w.attn_norm[i] * c.cq;
      gk[i] = w.attn_norm[i] * c.ck;
    }
    DeviceVector<float> q_f, k_f, v_f;
    CiBatchProjection<word>::FoldGain(q_f, w.q, model, heads * D, gq);
    CiBatchProjection<word>::FoldGain(k_f, w.k, model, kv_heads * D, gk);
    CiBatchProjection<word>::FoldGain(v_f, w.v, model, kv_heads * D,
                                      w.attn_norm);
    proj_->Prepare("attn.q", q_f.data(), model, heads * D, ly);
    proj_->Prepare("attn.k", k_f.data(), model, kv_heads * D, ly);
    proj_->Prepare("attn.v", v_f.data(), model, kv_heads * D, ly);
  }
  stages_.qkv += SinceSeconds(t0);

  // 3. The softmax's calibration in chain units.
  {
    typename CiBatchAttention<word>::SoftMaxCalibration sc;
    const double cqk = c.cq * c.ck;
    sc.m_eff = c.m_eff;
    sc.span = cqk * c.span_raw;
    sc.shift = cqk * c.s_raw_max;
    sc.causal = true;
    sc.inv_degree = 15;
    AssertTrue(static_cast<int>(c.row_shift_raw.size()) == heads,
               "CiBatchLayer::Attention: row_shift_raw is [heads][tokens]");
    sc.row_shift.assign(heads, std::vector<double>(T, 0.0));
    for (int h = 0; h < heads; h++) {
      for (int t = 0; t < T; t++) sc.row_shift[h][t] = cqk * c.row_shift_raw[h][t];
    }
    sc.row_norm = c.row_norm;
    attn.PrepareSoftMax(sc);
  }

  // 4. Per kv group: the group's K and V heads and its four Q heads
  //    projected, then per head scores -> Boot -> softmax -> P V, and the
  //    group's slice of the O projection accumulated at once (the attention
  //    output never stands whole: 4096 ciphertexts at level 1 are 6 GiB).
  std::vector<Ct> o_acc;
  std::vector<Ct> attn_out(static_cast<size_t>(group) * D);
  std::vector<std::vector<Ct>> k_heads, v_heads, q_heads;
  double o_ratio = 0.0;
  int o_level = -1;
  const int top = boot_->GetBootParameter().GetEndLevel();
  // A projection tile split into its heads.
  auto split_heads = [&](std::vector<Ct> &tile,
                         std::vector<std::vector<Ct>> &heads_out) {
    // (`assign(n, value)` would copy a vector of non-copyable ciphertexts.)
    heads_out.clear();
    heads_out.resize(heads_per_tile);
    for (int i = 0; i < heads_per_tile; i++) {
      for (int cc = 0; cc < D; cc++) {
        heads_out[i].push_back(std::move(tile[i * D + cc]));
      }
    }
    tile.clear();
  };
  for (int kv = 0; kv < kv_heads; kv++) {
    NvtxScope _g("batch: kv group");
    if (kv % heads_per_tile == 0) {
      t0 = Clock::now();
      std::vector<Ct> kt, vt;
      proj_->Project(kt, src_y, "attn.k", kv / heads_per_tile);
      proj_->Project(vt, src_y, "attn.v", kv / heads_per_tile);
      split_heads(kt, k_heads);
      split_heads(vt, v_heads);
      stages_.qkv += SinceSeconds(t0);
    }
    const std::vector<Ct> &k_kv = k_heads[kv % heads_per_tile];
    const std::vector<Ct> &v_kv = v_heads[kv % heads_per_tile];
    for (int hi = 0; hi < group; hi++) {
      const int h = kv * group + hi;
      NvtxScope _h("batch: head");
      // The head's Q: its tile projected when its first head comes up.
      t0 = Clock::now();
      if (h % heads_per_tile == 0) {
        std::vector<Ct> qt;
        proj_->Project(qt, src_y, "attn.q", h / heads_per_tile);
        split_heads(qt, q_heads);
      }
      std::vector<Ct> q_h = std::move(q_heads[h % heads_per_tile]);
      stages_.qkv += SinceSeconds(t0);

      t0 = Clock::now();
      std::vector<Ct> scores;
      attn.Scores(scores, q_h, k_kv, akeys);
      stages_.scores += SinceSeconds(t0);

      // The scores' bootstraps, the chain's factor read off before them.
      t0 = Clock::now();
      const int ls = param.NPToLevel(scores[0].GetNP());
      const double carried = scores[0].GetScale() / param.GetScale(ls);
      if (!boot_->IsBootPrepared(layout_.num_slots)) {
        NvtxScope _p("batch: prepare boot tables");
        boot_->PrepareEvalSpecialFFT(layout_.num_slots);
        prepare_seconds_ += SinceSeconds(t0);
        t0 = Clock::now();
      }
      std::vector<Ct> booted(T);
      for (int l = 0; l < T; l++) {
        boot_->Boot(booted[l], scores[l], evk);
        scores[l] = Ct();
      }
      AssertTrue(param.NPToLevel(booted[0].GetNP()) == top,
                 "CiBatchLayer::Attention: the score bootstrap did not "
                 "land at the top level");
      stages_.boot += SinceSeconds(t0);

      t0 = Clock::now();
      std::vector<Ct> P;
      attn.SoftMax(P, booted, h, carried, evk);
      booted.clear();
      stages_.softmax += SinceSeconds(t0);

      t0 = Clock::now();
      std::vector<Ct> out_h;
      attn.Values(out_h, P, v_kv, akeys);
      P.clear();
      for (int cc = 0; cc < D; cc++) {
        attn_out[static_cast<size_t>(hi) * D + cc] = std::move(out_h[cc]);
      }
      stages_.values += SinceSeconds(t0);
      if (cfg_.verbose) {
        std::cout << "  [batch] head " << h << " done: scores "
                  << stages_.scores << " boot " << stages_.boot << " softmax "
                  << stages_.softmax << " values " << stages_.values
                  << " s so far" << std::endl;
      }
    }
    // 5. The group's slice of the O projection: rows kv * group * D .. of
    //    the `[heads * D][model]` tensor, the stream's factor on the weight,
    //    the chain's factor in the inputs' recorded scale.
    t0 = Clock::now();
    if (o_level < 0) {
      o_level = param.NPToLevel(attn_out[0].GetNP());
      o_ratio = attn_out[0].GetScale() / param.GetScale(o_level);
      if (cfg_.verbose) {
        MemoryPool::Report("batch: before the first O projection");
      }
    }
    const std::string on = "attn.o." + std::to_string(kv);
    proj_->Prepare(on, w.o + static_cast<size_t>(kv) * group * D * model,
                   group * D, model, o_level, c.stream_scale, o_ratio);
    std::vector<Ct> o_part;
    proj_->Project(o_part, attn_out, on);
    proj_->Release(on);
    for (auto &a : attn_out) a = Ct();
    if (kv == 0) {
      o_acc = std::move(o_part);
    } else {
      for (int i = 0; i < model; i++) boot_->Add(o_acc[i], o_acc[i], o_part[i]);
    }
    stages_.o += SinceSeconds(t0);
  }
  k_heads.clear();
  v_heads.clear();
  proj_->Release("attn.q");
  proj_->Release("attn.k");
  proj_->Release("attn.v");
  src_y = typename CiBatchProjection<word>::Source();
  if (cfg_.release_boot_tables) boot_->ReleaseEvalSpecialFFT(layout_.num_slots);

  // 6. The residual.
  t0 = Clock::now();
  const int lo_in = o_level;
  const double ratio = o_ratio;
  std::vector<Ct> o = std::move(o_acc);
  Unpark(stream, parked);
  const int ld = param.NPToLevel(o[0].GetNP());
  const int ls2 = param.NPToLevel(stream[0].GetNP());
  const int lo = std::min(ld, ls2);
  res.clear();
  res.resize(model);
  for (int i = 0; i < model; i++) {
    Ct a, b;
    boot_->LevelDown(a, stream[i], lo);
    boot_->LevelDown(b, o[i], lo);
    boot_->Add(res[i], a, b);
    o[i] = Ct();
  }
  stages_.o += SinceSeconds(t0);
  stages_.total = SinceSeconds(t_all);
  if (cfg_.verbose) {
    std::cout << "  [batch] Attention: y at " << ly << ", O in at " << lo_in
              << " (ratio " << ratio << "), residual at " << lo << "; boot "
              << stages_.boot << " s, norm " << stages_.norm << " s, q/k/v "
              << stages_.qkv << " s, scores " << stages_.scores
              << " s, softmax " << stages_.softmax << " s, values "
              << stages_.values << " s, o " << stages_.o << " s, total "
              << stages_.total << " s" << std::endl;
  }
}

template class CiBatchLayer<uint32_t>;
template class CiBatchLayer<uint64_t>;

#endif  // USE_CUBLAS

}  // namespace cheddar
