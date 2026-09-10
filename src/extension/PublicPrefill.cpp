#include "extension/PublicPrefill.h"

#include <cmath>
#include <vector>

#include "common/Assert.h"
#include "common/ParallelFor.h"

namespace cheddar {

void PublicSinkRescale(std::vector<double> &factor, const double *hidden,
                       int model, int num_instances, int tokens,
                       int sink_tokens) {
  AssertTrue(hidden != nullptr, "PublicSinkRescale: no hidden state");
  AssertTrue(model > 0 && num_instances > 0 && tokens > 0,
             "PublicSinkRescale: an empty shape");
  AssertTrue(sink_tokens >= 0 && sink_tokens < tokens,
             "PublicSinkRescale: the sinks must leave a body to average");
  factor.assign(static_cast<size_t>(num_instances) * tokens, 1.0);
  if (sink_tokens == 0) return;

  ParallelFor(num_instances, [&](int begin, int end) {
    std::vector<double> ms(tokens);
    for (int b = begin; b < end; b++) {
      const double *xb = hidden + static_cast<size_t>(b) * tokens * model;
      for (int t = 0; t < tokens; t++) {
        const double *x = xb + static_cast<size_t>(t) * model;
        double s = 0.0;
        for (int c = 0; c < model; c++) s += x[c] * x[c];
        ms[t] = s / model;
      }
      // The geometric mean of the body's power, which is what the sinks are
      // brought to. In logs, so a long context cannot overflow the product.
      double logsum = 0.0;
      for (int t = sink_tokens; t < tokens; t++) logsum += std::log(ms[t]);
      const double target = std::exp(logsum / (tokens - sink_tokens));
      for (int t = 0; t < sink_tokens; t++) {
        factor[static_cast<size_t>(b) * tokens + t] =
            (ms[t] > 0.0) ? std::sqrt(target / ms[t]) : 1.0;
      }
    }
  });
}

void PublicPrefillChunk(std::vector<double> &k, std::vector<double> &v,
                        const PublicPrefillLayer &layer, const double *hidden,
                        int num_instances, int width, int kv_head,
                        int rope_pos0) {
  AssertTrue(layer.wk != nullptr && layer.wv != nullptr &&
                 layer.attn_norm != nullptr,
             "PublicPrefillChunk: the layer's weights");
  AssertTrue(hidden != nullptr, "PublicPrefillChunk: no hidden state");
  AssertTrue(width > 0 && num_instances > 0,
             "PublicPrefillChunk: an empty chunk");
  AssertTrue(kv_head >= 0 && kv_head < layer.kv_heads,
             "PublicPrefillChunk: the kv head is out of range");
  const int model = layer.model;
  const int dim = layer.head_dim;
  const int half = dim / 2;
  const int cols = layer.kv_heads * dim;  //!< the tensor's output width
  const int base = kv_head * dim;         //!< this head's slice of it
  const size_t lanes = static_cast<size_t>(num_instances);
  AssertTrue(k.size() == static_cast<size_t>(dim) * width * lanes,
             "PublicPrefillChunk: the key array is not [c][p][b]");
  AssertTrue(v.size() == static_cast<size_t>(width) * dim * lanes,
             "PublicPrefillChunk: the value array is not [p][c][b]");

  // One (instance, token) at a time: the norm is over that row's channels and
  // the two projections read the same normalised row, so the row is formed
  // once and both contractions walk it.
  const size_t rows = static_cast<size_t>(num_instances) * width;
  ParallelFor(static_cast<int>(rows), [&](int begin, int end) {
    std::vector<double> y(model), kk(dim), vv(dim);
    for (int r = begin; r < end; r++) {
      const int b = r / width;
      const int p = r % width;
      const double *x = hidden + static_cast<size_t>(r) * model;

      double ms = 0.0;
      for (int c = 0; c < model; c++) ms += x[c] * x[c];
      const double inv = 1.0 / std::sqrt(ms / model + layer.eps);
      for (int c = 0; c < model; c++) {
        y[c] = x[c] * inv * static_cast<double>(layer.attn_norm[c]);
      }

      for (int d = 0; d < dim; d++) {
        kk[d] = 0.0;
        vv[d] = 0.0;
      }
      for (int c = 0; c < model; c++) {
        const double yc = y[c];
        if (yc == 0.0) continue;
        const float *wkr = layer.wk + static_cast<size_t>(c) * cols + base;
        const float *wvr = layer.wv + static_cast<size_t>(c) * cols + base;
        for (int d = 0; d < dim; d++) {
          kk[d] += yc * static_cast<double>(wkr[d]);
          vv[d] += yc * static_cast<double>(wvr[d]);
        }
      }

      // RoPE on K only (V is never rotated), rotate_half, at the position
      // this chunk was told to use -- `start - ptok` for Sylph's split, so
      // that the relative angle to an encrypted query at `t` is
      // `t - (p - ptok)` = the true `(ptok + t) - p`.
      const double pos = static_cast<double>(rope_pos0 + p);
      for (int d = 0; d < half; d++) {
        const double theta =
            std::pow(layer.rope_base, -2.0 * static_cast<double>(d) / dim);
        const double a = pos * theta;
        const double ca = std::cos(a), sa = std::sin(a);
        const double lo = kk[d], hi = kk[d + half];
        kk[d] = lo * ca - hi * sa;
        kk[d + half] = hi * ca + lo * sa;
      }

      // Into the two layouts. `ck` rides the key so the public scores land in
      // the encrypted branch's units; the value carries no such factor.
      for (int d = 0; d < dim; d++) {
        k[(static_cast<size_t>(d) * width + p) * lanes + b] = layer.ck * kk[d];
        v[(static_cast<size_t>(p) * dim + d) * lanes + b] = vv[d];
      }
    }
  });
}

}  // namespace cheddar
