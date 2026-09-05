#include "extension/CiDecodeLayer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/ParallelFor.h"
#include "core/EncodeGpu.h"
#include "extension/ChebyshevFit.h"
#include "extension/EvalPoly.h"
#include "extension/Profile.h"

namespace cheddar {

#ifdef USE_CUBLAS

namespace {
using Clock = std::chrono::steady_clock;
double SinceSeconds(Clock::time_point t0) {
  cudaDeviceSynchronize();
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

void ToDev(DeviceVector<float> &d, const float *h, size_t n) {
  HostVector<float> hv(n);
  std::copy(h, h + n, hv.begin());
  d.resize(static_cast<int>(n));
  CopyHostToDevice(d, hv);
}

// diag(gain) W on the host tensor (the projection reads the normalised
// stream, whose gain was not applied in the norm).
std::vector<float> FoldRows(const float *w, int in, int out,
                            const std::vector<double> &gain) {
  std::vector<float> res(static_cast<size_t>(in) * out);
  ParallelFor(in, [&](int begin, int end) {
    for (int c = begin; c < end; c++) {
      const double g = gain[c];
      const float *row = w + static_cast<size_t>(c) * out;
      float *dst = res.data() + static_cast<size_t>(c) * out;
      for (int o = 0; o < out; o++) {
        dst[o] = static_cast<float>(g * row[o]);
      }
    }
  });
  return res;
}
}  // namespace

template <typename word>
CiDecodeLayer<word>::CiDecodeLayer(std::shared_ptr<BootContext<word>> boot,
                                   const Config &cfg)
    : boot_{std::move(boot)},
      cfg_{cfg},
      layout_{boot_->param_.MaxNumSlots(), cfg.num_tokens} {
  AssertTrue(boot_->param_.conjugate_invariant_,
             "CiDecodeLayer: the batched layout is written for R+");
  AssertTrue(cfg_.model % cfg_.num_tokens == 0 &&
                 cfg_.model / cfg_.num_heads == cfg_.num_tokens,
             "CiDecodeLayer: the head dimension must equal the token rows");
  AssertTrue(cfg_.num_heads % cfg_.num_kv_heads == 0,
             "CiDecodeLayer: GQA wants whole query groups");
  AssertTrue(cfg_.hidden % cfg_.rows_per_tile == 0 &&
                 cfg_.rows_per_tile % cfg_.num_tokens == 0,
             "CiDecodeLayer: rows_per_tile must cut the hidden into whole "
             "pack groups");
  AssertTrue(cfg_.exp_degree <= 7 && cfg_.recip_degree <= 7 &&
                 cfg_.invsqrt_degree <= 7,
             "CiDecodeLayer: the attention/norm level plan (le, lf, land - 5) "
             "allocates the levels a used degree <= 7 consumes; a used degree "
             "of 8 needs one more and overruns a downstream LevelDown. This is "
             "now a level-budget bound only -- the EvalPoly 2^400 that used to "
             "ALSO strike a used degree of 8 is fixed (Context::Add Rx "
             "aliasing, Doing.md 7.52), proven by param_robust_test's "
             "DecodeVerbatimEntry at degrees 2..15");
  typename CiBatchProjection<word>::Config pcfg;
  pcfg.rows_per_tile = cfg_.rows_per_tile;
  pcfg.verbose = cfg_.verbose;
  proj_ = std::make_unique<CiBatchProjection<word>>(boot_, pcfg);
  const int T = cfg_.num_tokens, B = layout_.num_instances;
  row_masks_.resize(T);
  std::vector<double> mv(static_cast<size_t>(B) * T);
  for (int j = 0; j < T; j++) {
    std::fill(mv.begin(), mv.end(), 0.0);
    for (int b = 0; b < B; b++) {
      mv[static_cast<size_t>(b) * T + j] = 1.0;
    }
    layout_.Pack(row_masks_[j], mv);
  }
}

template <typename word>
void CiDecodeLayer<word>::AddRequiredRotations(EvkRequest &req) const {
  const int land = boot_->GetBootParameter().GetEndLevel();
  const int T = cfg_.num_tokens, B = layout_.num_instances;
  // The unpacks' babies run at their pt levels (<= the fan-out's 7); a
  // land-level key per baby is ~35 MiB of dead weight times 120. The
  // ladder's strides do run at land - 1, so they are requested LAST and
  // keep the deep key.
  for (int b = 1; b < T; b++) {
    req.AddRequest(b * B, FanoutLevel() + 1);
  }
  for (int s = B; s < B * T; s <<= 1) {
    req.AddRequest(s, land);  // the token ladder
  }
}

template <typename word>
int CiDecodeLayer<word>::ExpSquarings(const Calibration &c) const {
  const double sqd = std::sqrt(static_cast<double>(cfg_.num_tokens));
  double r_w = 0.0;
  for (int h = 0; h < cfg_.num_heads; h++) {
    r_w = std::max(r_w, (c.s_hi[h] - c.s_lo[h]) / sqd);
  }
  const int land = boot_->GetBootParameter().GetEndLevel();
  const int le = land - 1 - Log2Ceil(cfg_.exp_degree + 1);
  int k = 0;
  while (r_w / (1 << k) > 2.0 && k < le - 7) k++;
  return k;
}

template <typename word>
void CiDecodeLayer<word>::LowerTo(std::vector<Ct> &x, int level) const {
  for (auto &ct : x) {
    Ct t;
    boot_->LevelDown(t, ct, level);
    ct = std::move(t);
  }
}

template <typename word>
void CiDecodeLayer<word>::Park(Parked &parked, std::vector<Ct> &stream) const {
  NvtxScope _nv("decode: park the stream");
  const size_t n = stream.size();
  parked.bx.resize(n);
  parked.ax.resize(n);
  parked.np.resize(n);
  parked.scale.resize(n);
  parked.slots.resize(n);
  for (size_t i = 0; i < n; i++) {
    Ct &ct = stream[i];
    AssertFalse(ct.HasRx(), "CiDecodeLayer::Park: a stream ciphertext has rx");
    parked.np[i] = ct.GetNP();
    parked.scale[i] = ct.GetScale();
    parked.slots[i] = ct.GetNumSlots();
    CopyDeviceToHost(parked.bx[i], ct.bx_);
    CopyDeviceToHost(parked.ax[i], ct.ax_);
    ct = Ct();
  }
  stream.clear();
}

template <typename word>
void CiDecodeLayer<word>::Unpark(std::vector<Ct> &stream,
                                 Parked &parked) const {
  NvtxScope _nv("decode: unpark the stream");
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
  parked = Parked();
}

template <typename word>
void CiDecodeLayer<word>::BootMany(std::vector<Ct> &out,
                                   const std::vector<Ct> &in,
                                   const EvkMap<word> &evk) const {
  const int n = static_cast<int>(in.size());
  const int group = Max(cfg_.boot_group, 1);
  out.resize(n);
  for (int i0 = 0; i0 < n; i0 += group) {
    const int m = Min(n - i0, group);
    if (m == 1) {
      boot_->Boot(out[i0], in[i0], evk);
      continue;
    }
    std::vector<const Ct *> ptrs(m);
    for (int j = 0; j < m; j++) ptrs[j] = &in[i0 + j];
    std::vector<Ct> up_g;
    boot_->BootBatch(up_g, ptrs, evk);
    for (int j = 0; j < m; j++) out[i0 + j] = std::move(up_g[j]);
  }
}

template <typename word>
void CiDecodeLayer<word>::Note(const char *what) const {
  if (!cfg_.verbose) return;
  size_t f = 0, t = 0;
  cudaDeviceSynchronize();
  cudaMemGetInfo(&f, &t);
  std::cout << "  [decode] " << what << ": " << (f >> 20) << " MiB free"
            << std::endl;
}

template <typename word>
void CiDecodeLayer<word>::CanonicalTo(Ct &ct, int target) const {
  const Parameter<word> &param = boot_->param_;
  int l = param.NPToLevel(ct.GetNP());
  AssertTrue(l >= target, "CiDecodeLayer::CanonicalTo: cannot ascend");
  while (l > target) {
    Pt one;
    EncodeFull(one, l, param.GetScale(l), 1.0);
    Ct tmp;
    boot_->Mult(tmp, ct, one);
    boot_->Rescale(ct, tmp);
    l--;
  }
}

template <typename word>
void CiDecodeLayer<word>::EncodeFull(Pt &pt, int level, double scale,
                                     double value) const {
  const int T = cfg_.num_tokens, B = layout_.num_instances;
  std::vector<double> mv(static_cast<size_t>(B) * T, value);
  std::vector<Complex> msg;
  layout_.Pack(msg, mv);
  boot_->gpu_encoder_.Encode(pt, level, scale, msg);
}

template <typename word>
std::vector<typename CiDecodeLayer<word>::Pt>
CiDecodeLayer<word>::MakeRowMasks(int level, double value) const {
  const int T = cfg_.num_tokens, B = layout_.num_instances;
  const double scale = boot_->param_.GetScale(level);
  std::vector<Pt> masks(T);
  std::vector<double> mv(static_cast<size_t>(B) * T);
  std::vector<Complex> msg;
  for (int j = 0; j < T; j++) {
    std::fill(mv.begin(), mv.end(), 0.0);
    for (int b = 0; b < B; b++) {
      mv[static_cast<size_t>(b) * T + j] = value;
    }
    layout_.Pack(msg, mv);
    boot_->gpu_encoder_.Encode(masks[j], level, scale, msg);
  }
  return masks;
}

template <typename word>
void CiDecodeLayer<word>::PackGroup(Ct &dense, const std::vector<Ct> &chan,
                                    int c0,
                                    const std::vector<Pt> &masks) const {
  Ct acc;
  for (int r = 0; r < cfg_.num_tokens; r++) {
    Ct m;
    boot_->Mult(m, chan[c0 + r], masks[r]);
    if (r == 0) {
      acc = std::move(m);
    } else {
      boot_->Add(acc, acc, m);
    }
  }
  boot_->Rescale(dense, acc);
}

template <typename word>
CiDecodeUnpack<word> &CiDecodeLayer<word>::UnpackAt(int pt_level) {
  auto it = unpacks_.find(pt_level);
  if (it == unpacks_.end()) {
    it = unpacks_
             .emplace(pt_level, std::make_unique<CiDecodeUnpack<word>>(
                                    boot_, row_masks_, layout_.num_instances,
                                    pt_level,
                                    boot_->param_.GetRescalePrimeProd(
                                        pt_level)))
             .first;
  }
  return *it->second;
}

template <typename word>
void CiDecodeLayer<word>::EvalAtDepth(Ct &out, const Ct &in, double pk_value,
                                      double shift_value,
                                      const std::function<double(double)> &fn,
                                      int degree,
                                      const EvkMap<word> &evk) const {
  const Parameter<word> &param = boot_->param_;
  const auto &mult_key = evk.GetMultiplicationKey();
  const int l = param.NPToLevel(in.GetNP());
  Pt pk;
  EncodeFull(pk, l, param.GetScale(l), pk_value);
  Ct v;
  {
    Ct tmp;
    boot_->Mult(tmp, in, pk);
    boot_->Rescale(v, tmp);
  }
  const int lv = param.NPToLevel(v.GetNP());
  if (shift_value != 0.0) {
    Constant<word> shift;
    boot_->encoder_.EncodeConstant(shift, lv, v.GetScale(), shift_value);
    boot_->Add(v, v, shift);
  }
  auto coeffs = chebfit::Interpolate(fn, degree);
  const double in_scale = v.GetScale();
  const int used =
      EvalPoly<word>(coeffs, lv, in_scale, in_scale, true).GetPolyDegree();
  const int lr = lv - Log2Ceil(used + 1);
  EvalPoly<word> poly(coeffs, lv, in_scale, param.GetScale(lr), true);
  poly.Compile(boot_);
  poly.Evaluate(boot_, out, v, mult_key);
}

template <typename word>
void CiDecodeLayer<word>::NormTurn(std::vector<Ct> &y,
                                   const std::vector<Ct> &stream, double alpha,
                                   double window, double stream_scale,
                                   double norm_scale,
                                   int unpack_pt_level,
                                   const EvkMap<word> &evk) {
  NvtxScope _nv("decode: NormTurn");
  const Parameter<word> &param = boot_->param_;
  const auto &mult_key = evk.GetMultiplicationKey();
  const int T = cfg_.num_tokens, model = cfg_.model, B = layout_.num_instances;
  const int groups = model / T;
  const int ls = param.NPToLevel(stream[0].GetNP());
  AssertTrue(ls >= 1,
             "CiDecodeLayer::NormTurn: the stream needs a level to pack");
  const int land = boot_->GetBootParameter().GetEndLevel();
  // The pack rides the residual at `norm_scale` instead of the carried
  // `stream_scale`: fold the ratio into the mask value (free), and use
  // `norm_scale` wherever `stream_scale` set the boot height below (the
  // invsqrt's argument and the apply's declared scale both cancel it, so
  // `y` is unchanged -- only the boot's message magnitude moves).
  const double ns = norm_scale > 0.0 ? norm_scale : stream_scale;
  auto masks = MakeRowMasks(ls, ns / stream_scale);

  // Pack each group, the boots GROUPED through BootBatch, and the
  // squares AFTER the boot (dense^2 + the token ladder on the summed
  // groups); no accumulator bootstraps -- the decode norm has no ride
  // problem (Doing.md 7.42).
  std::vector<Ct> up;
  Ct s2;
  {
    std::vector<Ct> dense(groups);
    for (int g = 0; g < groups; g++) {
      PackGroup(dense[g], stream, g * T, masks);
    }
    BootMany(up, dense, evk);
  }
  for (int g = 0; g < groups; g++) {
    Ct sq;
    boot_->HMult(sq, up[g], up[g], mult_key);
    if (g == 0) {
      s2 = std::move(sq);
    } else {
      boot_->Add(s2, s2, sq);
    }
  }
  for (int s = B; s < B * T; s <<= 1) {
    Ct r;
    boot_->HRot(r, s2, evk.GetRotationKey(s), s);
    boot_->Add(s2, s2, r);
  }

  // u = alpha (S / (model s^2) + eps) mapped onto the window, the inverse
  // square root at depth, beta = sqrt(alpha) and the stream's factor OUT
  // through the declared scale (free).
  const double wl = 1.0 / std::sqrt(window), wh = std::sqrt(window);
  const double a = 0.5 * (wh - wl), b = 0.5 * (wh + wl);
  Ct r_inv;
  EvalAtDepth(
      r_inv, s2,
      alpha / (static_cast<double>(model) * ns * ns * a),
      (alpha * cfg_.eps - b) / a,
      [a, b](double t) { return 1.0 / std::sqrt(a * t + b); },
      cfg_.invsqrt_degree, evk);
  const int lr = land - 5;
  CanonicalTo(r_inv, lr);
  r_inv.SetScale(r_inv.GetScale() * ns / std::sqrt(alpha));

  // Apply: ONE multiply normalises a group's 128 channels; the product
  // then descends to the caller's level BEFORE the unpack -- 4096
  // channels cost 2.1 GiB a limb, so y stands as low as its consumers
  // allow.
  auto &unpack = UnpackAt(unpack_pt_level);
  y.clear();
  y.resize(model);
  for (int g = 0; g < groups; g++) {
    Ct dd, dn;
    boot_->LevelDown(dd, up[g], lr);
    up[g] = Ct();
    boot_->HMult(dn, dd, r_inv, mult_key);
    Ct dn8;
    boot_->LevelDown(dn8, dn, unpack_pt_level);
    std::vector<Ct> chans;
    unpack.Evaluate(boot_, chans, dn8, evk);
    for (int r = 0; r < T; r++) {
      y[g * T + r] = std::move(chans[r]);
    }
  }
  Note("norm unpacked");
}

template <typename word>
std::vector<float> CiDecodeLayer<word>::FoldQK(
    const float *w, int out, const std::vector<double> &gain,
    const std::vector<double> &head_scale, int position) const {
  const int in = cfg_.model, D = cfg_.num_tokens, half = D / 2;
  const int heads = out / D;
  std::vector<double> cs(half), sn(half);
  for (int d = 0; d < half; d++) {
    const double ang = position * std::pow(cfg_.rope_base, -2.0 * d / D);
    cs[d] = std::cos(ang);
    sn[d] = std::sin(ang);
  }
  std::vector<float> res(static_cast<size_t>(in) * out);
  ParallelFor(in, [&](int begin, int end) {
    for (int c = begin; c < end; c++) {
      const float *row = w + static_cast<size_t>(c) * out;
      float *dst = res.data() + static_cast<size_t>(c) * out;
      const double g = gain[c];
      for (int h = 0; h < heads; h++) {
        const double hs = head_scale.empty() ? 1.0 : head_scale[h];
        for (int d = 0; d < half; d++) {
          const double lo = row[h * D + d], hi = row[h * D + d + half];
          dst[h * D + d] = static_cast<float>(g * hs * (lo * cs[d] - hi * sn[d]));
          dst[h * D + d + half] =
              static_cast<float>(g * hs * (hi * cs[d] + lo * sn[d]));
        }
      }
    }
  });
  return res;
}

template <typename word>
void CiDecodeLayer<word>::Step(std::vector<Ct> &next, std::vector<Ct> &stream,
                               Cache &cache, const HostWeights &w,
                               const Calibration &c, int position,
                               const EvkMap<word> &evk, Debug *dbg) {
  NvtxScope _nv("decode: step");
  const Parameter<word> &param = boot_->param_;
  const auto &mult_key = evk.GetMultiplicationKey();
  const int T = cfg_.num_tokens, model = cfg_.model, hidden = cfg_.hidden;
  const int B = layout_.num_instances, D = T;
  const int heads = cfg_.num_heads, kv_heads = cfg_.num_kv_heads;
  const int qgroup = heads / kv_heads;
  const double sqd = std::sqrt(static_cast<double>(D));
  AssertTrue(position == T - 1,
             "CiDecodeLayer::Step: only a full cache (position = T - 1); a "
             "shorter one leaves exp(shift) in Z's dead rows");
  AssertTrue(static_cast<int>(stream.size()) == model,
             "CiDecodeLayer::Step: the stream is `model` ciphertexts");
  AssertTrue(static_cast<int>(c.s_lo.size()) == heads &&
                 static_cast<int>(c.s_hi.size()) == heads &&
                 static_cast<int>(c.z_lo.size()) == heads &&
                 static_cast<int>(c.z_hi.size()) == heads,
             "CiDecodeLayer::Step: per-head windows");
  AssertTrue(static_cast<int>(cache.kc.size()) == kv_heads &&
                 static_cast<int>(cache.vt.size()) == kv_heads,
             "CiDecodeLayer::Step: one K and one V set per kv head");
  const int land = boot_->GetBootParameter().GetEndLevel();
  const int k_sq = ExpSquarings(c);
  const int le = land - 1 - Log2Ceil(cfg_.exp_degree + 1);
  const int lf = FanoutLevel();  // 7; the walk descends canonically to it
  AssertTrue(le - k_sq >= lf,
             "CiDecodeLayer::Step: the level budget does not close (the O "
             "output must reach level 1)");
  stages_ = Stages();
  auto t_all = Clock::now();
  auto t0 = Clock::now();

  // ---- The attention norm; the stream then parks in host memory until
  // its residual add (the card wants every byte for y and the splits).
  std::vector<Ct> y;
  NormTurn(y, stream, c.attn_alpha, c.attn_window, c.stream_scale,
           c.attn_norm_scale, lf, evk);
  Parked parked;
  Park(parked, stream);
  stages_.norm1 = SinceSeconds(t0);
  t0 = Clock::now();

  // ---- q/k and the appends. The norm gain, the per-head score gamma (q)
  // and RoPE at the step's position fold into the weights on the host.
  // gamma rides the PROJECTION WEIGHTS, never a post-product multiply
  // (Doing.md 7.45: an unscaled partial score wraps the low-level message
  // headroom).
  std::vector<double> gam(heads);
  for (int h = 0; h < heads; h++) {
    gam[h] = cfg_.ride / Max(c.s_hi[h] - c.s_lo[h], 1e-9);
  }
  std::vector<Ct> q, knew;
  auto ratio_of = [&](const Ct &x) {
    return x.GetScale() / param.GetScale(param.NPToLevel(x.GetNP()));
  };
  {
    NvtxScope _w("decode: qkv");
    AssertTrue(param.NPToLevel(y[0].GetNP()) == lf - 1,
               "CiDecodeLayer::Step: the norm must land y at the V "
               "append's level");
    // The V APPEND first, straight off y at 6: there is no V projection.
    // vt[position] = sum_d mask_d (sum_c W_v[c][d] y_c)
    //             = sum_c (sum_d W_v[c][d] mask_d) y_c,
    // and the inner sum is a PER-TOKEN plaintext (row d carries the
    // weight), so the append is `model` plaintext multiplies per kv head
    // and ONE rescale -- no 17 GiB level-7 split of y, which is what
    // OOM'd this step twice. The plaintext's scale is chosen so the pack
    // lands CANONICAL at VCacheLevel; the per-step encodes (~kv * model)
    // are the cost, and folding them away is an open lever.
    {
      NvtxScope _v("decode: v append");
      const int ly = lf - 1;
      const double pt_scale =
          param.GetScale(ly) * param.GetScale(ly) / y[0].GetScale();
      std::vector<double> vals(D);
      std::vector<Complex> msg;
      for (int kv = 0; kv < kv_heads; kv++) {
        Ct acc;
        for (int cc = 0; cc < model; cc++) {
          for (int d = 0; d < D; d++) {
            vals[d] = w.attn_norm[cc] *
                      static_cast<double>(
                          w.v[static_cast<size_t>(cc) * kv_heads * D +
                              kv * D + d]);
          }
          layout_.PackPerToken(msg, vals);
          Pt pt;
          boot_->gpu_encoder_.Encode(pt, ly, pt_scale, msg);
          Ct m;
          boot_->Mult(m, y[cc], pt);
          if (cc == 0) {
            acc = std::move(m);
          } else {
            boot_->Add(acc, acc, m);
          }
        }
        boot_->Rescale(cache.vt[kv][position], acc);
      }
    }
    Note("v appended");
    stages_.append = SinceSeconds(t0);
    t0 = Clock::now();

    // k and q per HALF: the operand a sliced-row half of the folded
    // tensor, the source that half's incremental split at y's own scale
    // (`BeginSplit`), the halves' products ADDED -- a whole-width int8
    // copy never coexists with y. The scale ratio is per LEVEL (LevelDown
    // keeps the scale, the canonical one moves).
    LowerTo(y, 3);
    {
      auto kf = FoldQK(w.k, kv_heads * D, w.attn_norm, {}, position);
      const double kr = ratio_of(y[0]);
      const int block = model / 2;
      for (int b0 = 0; b0 < model; b0 += block) {
        DeviceVector<float> kd;
        ToDev(kd, kf.data() + static_cast<size_t>(b0) * kv_heads * D,
              static_cast<size_t>(block) * kv_heads * D);
        proj_->Prepare("dec.k", kd.data(), block, kv_heads * D, 3, 1.0, kr);
        typename CiBatchProjection<word>::Source src;
        proj_->BeginSplit(src, block, 3, layout_.num_slots, y[0].GetScale());
        for (int i = 0; i < block; i++) {
          proj_->AddColumn(src, i, y[b0 + i]);
        }
        std::vector<Ct> part;
        for (int t2 = 0; t2 < proj_->NumTiles("dec.k"); t2++) {
          std::vector<Ct> rows;
          proj_->Project(rows, src, "dec.k", t2);
          for (auto &ct : rows) part.push_back(std::move(ct));
        }
        if (b0 == 0) {
          knew = std::move(part);
        } else {
          for (size_t i = 0; i < knew.size(); i++) {
            boot_->Add(knew[i], knew[i], part[i]);
          }
        }
      }
      proj_->Release("dec.k");
    }
    Note("k projected");
    LowerTo(y, 2);
    {
      auto qf = FoldQK(w.q, heads * D, w.attn_norm, gam, position);
      const double qr = ratio_of(y[0]);
      const int block = model / 2;
      for (int b0 = 0; b0 < model; b0 += block) {
        DeviceVector<float> qd;
        ToDev(qd, qf.data() + static_cast<size_t>(b0) * heads * D,
              static_cast<size_t>(block) * heads * D);
        proj_->Prepare("dec.q", qd.data(), block, heads * D, 2, 1.0, qr);
        typename CiBatchProjection<word>::Source src;
        proj_->BeginSplit(src, block, 2, layout_.num_slots, y[0].GetScale());
        for (int i = 0; i < block; i++) {
          proj_->AddColumn(src, i, y[b0 + i]);
        }
        std::vector<Ct> part;
        for (int t2 = 0; t2 < proj_->NumTiles("dec.q"); t2++) {
          std::vector<Ct> rows;
          proj_->Project(rows, src, "dec.q", t2);
          for (auto &ct : rows) part.push_back(std::move(ct));
        }
        if (b0 == 0) {
          q = std::move(part);
        } else {
          for (size_t i = 0; i < q.size(); i++) {
            boot_->Add(q[i], q[i], part[i]);
          }
        }
      }
      proj_->Release("dec.q");
    }
    y.clear();
    Note("q projected");
  }
  stages_.qkv = SinceSeconds(t0);
  t0 = Clock::now();

  // ---- The K append: one masked add per channel (Doing.md 7.45's [4]).
  {
    NvtxScope _a("decode: k append");
    Pt kmask;
    {
      std::vector<double> mv(static_cast<size_t>(B) * T, 0.0);
      for (int b = 0; b < B; b++) {
        mv[static_cast<size_t>(b) * T + position] = 1.0;
      }
      std::vector<Complex> msg;
      layout_.Pack(msg, mv);
      boot_->gpu_encoder_.Encode(kmask, 2, param.GetScale(2), msg);
    }
    for (int kv = 0; kv < kv_heads; kv++) {
      for (int d = 0; d < D; d++) {
        Ct m, r;
        boot_->Mult(m, knew[kv * D + d], kmask);
        boot_->Rescale(r, m);
        boot_->Add(cache.kc[kv][d], cache.kc[kv][d], r);
      }
    }
    knew.clear();
  }
  Note("appends done");
  stages_.append += SinceSeconds(t0);
  t0 = Clock::now();

  // ---- Per head: scores, the max shift, ONE boot, the exp walk, Z and
  // its reciprocal, the fan-out, ScoreV, the reciprocal LAST, the unpack.
  auto &fanout = UnpackAt(lf);
  auto &hunpack = UnpackAt(3);
  std::vector<Ct> o_acc;
  DeviceVector<float> o_dev;
  ToDev(o_dev, w.o, static_cast<size_t>(model) * model);
  // Every head's shifted score first, then the boots GROUPED -- and q
  // dies before the per-head transients (its 4096 channels are 8 GiB).
  std::vector<Ct> s_ups;
  {
    NvtxScope _s("decode: scores");
    std::vector<Ct> scs(heads);
    for (int h = 0; h < heads; h++) {
      const int kv = h / qgroup;
      Ct acc;
      for (int d = 0; d < D; d++) {
        Ct m;
        boot_->Mult(m, q[h * D + d], cache.kc[kv][d]);
        if (d == 0) {
          acc = std::move(m);
        } else {
          boot_->Add(acc, acc, m);
        }
      }
      boot_->RelinearizeRescale(scs[h], acc, mult_key);
      const int l0 = param.NPToLevel(scs[h].GetNP());
      Constant<word> shift;
      boot_->encoder_.EncodeConstant(shift, l0, scs[h].GetScale(),
                                     -gam[h] * c.s_hi[h]);
      boot_->Add(scs[h], scs[h], shift);
    }
    q.clear();
    BootMany(s_ups, scs, evk);
  }
  Note("scores booted");
  for (int h = 0; h < heads; h++) {
    NvtxScope _h("decode: head");
    if (h % 8 == 0) Note(("head " + std::to_string(h)).c_str());
    const int kv = h / qgroup;
    Ct s_up = std::move(s_ups[h]);
    // e = exp((s - s_hi) / sqd): a deg-7 Chebyshev on the 2^-k chunk, k
    // squarings, a canonical descent wherever the trim lands high (the
    // squarings SQUARE a declared-scale offset, so none may enter them).
    Ct e;
    {
      const double rg = (c.s_hi[h] - c.s_lo[h]) / sqd / (1 << k_sq);
      const double lo = -rg - 0.02 * rg - 1e-9, hi = 0.02 * rg + 1e-9;
      const double a = 0.5 * (hi - lo), b = 0.5 * (hi + lo);
      const double in_factor = gam[h] * sqd * (1 << k_sq);
      EvalAtDepth(e, s_up, 1.0 / (in_factor * a), -b / a,
                  [a, b](double t) { return std::exp(a * t + b); },
                  cfg_.exp_degree, evk);
      CanonicalTo(e, le);
      for (int j = 0; j < k_sq; j++) {
        Ct e2;
        boot_->HMult(e2, e, e, mult_key);
        e = std::move(e2);
      }
      CanonicalTo(e, lf);
    }
    Ct r_inv;
    {
      Ct z;
      boot_->LevelDown(z, e, lf);
      for (int s = B; s < B * T; s <<= 1) {
        Ct r;
        boot_->HRot(r, z, evk.GetRotationKey(s), s);
        boot_->Add(z, z, r);
      }
      const double zl = c.z_lo[h] * 0.9, zh = c.z_hi[h] * 1.1;
      const double a = 0.5 * (zh - zl), b = 0.5 * (zh + zl);
      EvalAtDepth(r_inv, z, 1.0 / a, -b / a,
                  [a, b](double t) { return 1.0 / (a * t + b); },
                  cfg_.recip_degree, evk);
      CanonicalTo(r_inv, 4);
    }
    std::vector<Ct> e_t;
    fanout.Evaluate(boot_, e_t, e, evk);
    Ct out_raw;
    {
      Ct acc;
      for (int t = 0; t < T; t++) {
        Ct et;
        boot_->LevelDown(et, e_t[t], VCacheLevel());
        e_t[t] = Ct();
        Ct m;
        boot_->Mult(m, et, cache.vt[kv][t]);
        if (t == 0) {
          acc = std::move(m);
        } else {
          boot_->Add(acc, acc, m);
        }
      }
      boot_->RelinearizeRescale(out_raw, acc, mult_key);
    }
    e_t.clear();
    Ct out_ct;
    {
      Ct dd;
      boot_->LevelDown(dd, out_raw, 4);
      boot_->HMult(out_ct, dd, r_inv, mult_key);
    }
    std::vector<Ct> chans;
    hunpack.Evaluate(boot_, chans, out_ct, evk);
    // The head's slice of O, projected NOW and accumulated -- 128
    // channels split per head instead of 4096 standing until one big O.
    {
      const double ratio = ratio_of(chans[0]);
      const int lo_in = param.NPToLevel(chans[0].GetNP());
      proj_->Prepare("dec.o",
                     o_dev.data() + static_cast<size_t>(h) * D * model, D,
                     model, lo_in, c.stream_scale, ratio);
      std::vector<Ct> o_part;
      proj_->Project(o_part, chans, "dec.o");
      proj_->Release("dec.o");
      if (h == 0) {
        o_acc = std::move(o_part);
      } else {
        for (int i = 0; i < model; i++) {
          boot_->Add(o_acc[i], o_acc[i], o_part[i]);
        }
      }
    }
  }
  stages_.heads = SinceSeconds(t0);
  t0 = Clock::now();

  // ---- The attention residual (the stream's factor rode the O weights).
  std::vector<Ct> mid(model);
  {
    NvtxScope _o("decode: attention residual");
    Unpark(stream, parked);
    for (int i = 0; i < model; i++) {
      CanonicalTo(o_acc[i], StreamLevel());
      boot_->Add(mid[i], stream[i], o_acc[i]);
    }
    stream.clear();
    o_acc.clear();
  }
  Note("attention residual");
  stages_.o = SinceSeconds(t0);
  t0 = Clock::now();

  // ---- The feed-forward norm; mid then parks until its residual add.
  std::vector<Ct> y2;
  NormTurn(y2, mid, c.ffn_alpha, c.ffn_window, c.stream_scale,
           c.ffn_norm_scale, 3, evk);
  Parked parked_mid;
  Park(parked_mid, mid);
  stages_.norm2 = SinceSeconds(t0);
  t0 = Clock::now();

  // ---- gate/up (gain folded), the ride gammas IN THE PACK MASKS (free),
  // two boots a group, SiLU at depth, the up gamma unfolded through the
  // declared scale, the product, the unpack, and down per input-row tile.
  const double gam_g = cfg_.ride / c.silu_gmax;
  const double gam_u = cfg_.ride / c.up_umax;
  {
    NvtxScope _w("decode: ffn weights");
    auto gf = FoldRows(w.gate, model, hidden, w.ffn_norm);
    auto uf = FoldRows(w.up, model, hidden, w.ffn_norm);
    DeviceVector<float> gd, ud;
    ToDev(gd, gf.data(), gf.size());
    ToDev(ud, uf.data(), uf.size());
    const double r2 = y2[0].GetScale() / param.GetScale(2);
    proj_->Prepare("dec.gate", gd.data(), model, hidden, 2, 1.0, r2);
    proj_->Prepare("dec.up", ud.data(), model, hidden, 2, 1.0, r2);
  }
  typename CiBatchProjection<word>::Source src2;
  {
    proj_->Split(src2, y2, "dec.gate");
    y2.clear();
  }
  Note("ffn split");
  auto gmasks = MakeRowMasks(1, gam_g);
  auto umasks = MakeRowMasks(1, gam_u);
  const int silu_land = land - 1 - Log2Ceil(cfg_.silu_degree + 1);
  const double R = c.silu_gmax * 1.02;
  auto &munpack = UnpackAt(3);
  std::vector<Ct> d_acc;
  const int tiles = hidden / cfg_.rows_per_tile;
  const int groups_per_tile = cfg_.rows_per_tile / T;
  for (int j = 0; j < tiles; j++) {
    NvtxScope _t("decode: ffn tile");
    auto tt = Clock::now();
    std::vector<Ct> g, u;
    proj_->Project(g, src2, "dec.gate", j);
    proj_->Project(u, src2, "dec.up", j);
    stages_.gate_up += SinceSeconds(tt);
    tt = Clock::now();
    // The tile's gate/up pairs packed first, their boots GROUPED.
    std::vector<Ct> ups;
    {
      std::vector<Ct> denses(2 * groups_per_tile);
      for (int gi = 0; gi < groups_per_tile; gi++) {
        PackGroup(denses[2 * gi], g, gi * T, gmasks);
        PackGroup(denses[2 * gi + 1], u, gi * T, umasks);
      }
      g.clear();
      u.clear();
      BootMany(ups, denses, evk);
    }
    std::vector<Ct> chans(cfg_.rows_per_tile);
    for (int gi = 0; gi < groups_per_tile; gi++) {
      Ct dg_up = std::move(ups[2 * gi]);
      Ct du_up = std::move(ups[2 * gi + 1]);
      Ct sg;
      EvalAtDepth(sg, dg_up, 1.0 / (gam_g * R), 0.0,
                  [R](double t) {
                    const double x = R * t;
                    return x / (1.0 + std::exp(-x));
                  },
                  cfg_.silu_degree, evk);
      CanonicalTo(sg, silu_land);
      du_up.SetScale(du_up.GetScale() * gam_u);
      Ct dm;
      {
        Ct dd;
        boot_->LevelDown(dd, du_up, param.NPToLevel(sg.GetNP()));
        boot_->HMult(dm, sg, dd, mult_key);
      }
      Ct dml;
      boot_->LevelDown(dml, dm, 3);
      std::vector<Ct> gc;
      munpack.Evaluate(boot_, gc, dml, evk);
      for (int r = 0; r < T; r++) {
        chans[gi * T + r] = std::move(gc[r]);
      }
    }
    stages_.mid += SinceSeconds(tt);
    tt = Clock::now();
    // This tile's rows of down, the stream's factor folded in.
    const double ratio = chans[0].GetScale() / param.GetScale(2);
    DeviceVector<float> dd;
    ToDev(dd,
          w.down + static_cast<size_t>(j) * cfg_.rows_per_tile * model,
          static_cast<size_t>(cfg_.rows_per_tile) * model);
    const std::string dn = "dec.down." + std::to_string(j);
    proj_->Prepare(dn, dd.data(), cfg_.rows_per_tile, model, 2,
                   c.stream_scale, ratio);
    std::vector<Ct> d;
    proj_->Project(d, chans, dn);
    proj_->Release(dn);
    chans.clear();
    if (j == 0) {
      d_acc = std::move(d);
      Note("ffn tile 0");
    } else {
      for (int i = 0; i < model; i++) {
        boot_->Add(d_acc[i], d_acc[i], d[i]);
      }
    }
    stages_.down += SinceSeconds(tt);
  }
  proj_->Release("dec.gate");
  proj_->Release("dec.up");

  // ---- The feed-forward residual: the next stream, level 1, chainable.
  Unpark(mid, parked_mid);
  next.clear();
  next.resize(model);
  for (int i = 0; i < model; i++) {
    CanonicalTo(d_acc[i], StreamLevel());
    boot_->Add(next[i], mid[i], d_acc[i]);
  }
  if (dbg != nullptr) {
    dbg->mid.resize(model);
    for (int i = 0; i < model; i++) {
      boot_->LevelDown(dbg->mid[i], mid[i], param.NPToLevel(mid[i].GetNP()));
    }
  }
  stages_.total = SinceSeconds(t_all);
}

template class CiDecodeLayer<uint32_t>;
template class CiDecodeLayer<uint64_t>;

#endif  // USE_CUBLAS

}  // namespace cheddar
