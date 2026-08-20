// One whole Llama-3-8B decoder block, assembled.
//
// SylphScheduleTest proved one turn of [SYLPH] figure 2's cycle and then two
// turns of it, with RMSNorm in the slot leg and a descent to level 0 standing
// in for the product. This is the block: all four non-linear operators, in the
// order the model puts them, with the residual connections, on real layer-2
// weights.
//
// WHAT IS ENCRYPTED AND WHAT IS NOT, stated plainly and up front.
//
// Encrypted, on the GPU, in the real parameter set: every bootstrap, every
// slot/coefficient conversion, RMSNorm twice, RoPE, SoftMax, SiLU, the
// elementwise gate, and both residual additions.
//
// **Not** encrypted: the seven matrix products. `HostLinearLeg` decrypts,
// multiplies in the clear and re-encrypts. That is a stand-in and it is
// labelled as one everywhere it appears. The reason is recorded in Doing.md
// 1.5y: the coefficient-domain product path needs a ring-switching parameter
// pair, the pair for `bootparam_35` exists on paper but has never executed on
// a GPU, and a block that hard-coded the product could not be run at all until
// it had. With `LlamaBlock::LinearLeg` the same block runs today and takes the
// real product as a drop-in when it is ready.
//
// So the error this test reports is **the whole cost of everything except the
// products**, measured against the true block in the clear. That is a real
// number about a real question -- whether the non-linear half of a Llama-3
// layer survives six bootstraps and four polynomial approximations -- and it
// is not answerable any other way until the product lands.
//
// THE CALIBRATION IS COMPUTED HERE, FROM THE HOST RUN.
//
// [SYLPH] section 3.1.3 calibrates offline on representative inputs. There is
// no representative input here, there is *the* input, so the test runs the
// block in the clear first and reads the constants off it: the two layer
// constants, the SoftMax range and shift, the inverse-square-root intervals,
// the SiLU range, and the crossing constant of every tensor that passes a
// bootstrap. That is generous to the encrypted run -- it is the best any
// calibration could do -- and it is the right way round, because a
// miscalibration would otherwise be indistinguishable from an approximation
// error.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Testbed.h"
#include "extension/AttentionPacking.h"
#include "extension/LlamaBlock.h"

using word = uint32_t;
using Block = LlamaBlock<word>;

namespace {

int SlackFromEnv() {
  const char *env = std::getenv("CHEDDAR_SLACK");
  return env ? std::atoi(env) : 8;
}
const int kSlack = SlackFromEnv();

constexpr int kAllTokens = 128;
constexpr int kChannels = 4096;
constexpr int kKvChannels = 1024;
constexpr int kHidden = 14336;
constexpr int kHeadDim = 128;
constexpr int kTokens = 64;      // the user segment, clear of the sinks
constexpr int kFirstToken = 64;  // where it starts in the 128-token prompt
constexpr double kEps = 1e-5;
constexpr double kBootMax = 0.5;

std::string DataDir() {
  const char *env = std::getenv("LLAMA3_REAL_DIR");
  return env ? std::string(env) : std::string();
}

bool ReadF32(const std::string &path, size_t count, std::vector<double> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<float> raw(count);
  f.read(reinterpret_cast<char *>(raw.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  if (static_cast<size_t>(f.gcount()) != count * sizeof(float)) return false;
  out.assign(raw.begin(), raw.end());
  return true;
}

double MaxAbs(const std::vector<double> &v) {
  double m = 0.0;
  for (double u : v) m = std::max(m, std::abs(u));
  return m;
}

// ---------------------------------------------------------------------------
// The packing, in one place.
//
// Both layouts are the block's contract with the product leg, so they are
// written once here and used by the encoder, the decoder and the reference.
//
//   channel packing   ciphertext i, slot s  ->  token s % T,
//                                               channel i * (slots/T) + s / T
//   score packing     key j of row R        ->  ciphertext R / rows_per_ct,
//                                               slot (R % rows_per_ct)
//                                                   + j * rows_per_ct
//
// The score rows are R = head * T + query, so the rows of one head stay inside
// one ciphertext. The key axis is strided rather than contiguous because
// Cheddar rotates cyclically over the whole slot vector, so a contiguous row
// would have SoftMax's rotate-and-add straddle row boundaries -- SoftMax.h.
struct Packing {
  int tokens = kTokens;
  int slots = 0;
  int degree = 0;
  int channels_per_ct() const { return slots / tokens; }
  int rows_per_ct() const { return slots / tokens; }
  int coeff(int slot) const {
    return AttentionPacking::CoeffOfSlot({slot, false}, degree);
  }
};

// ---------------------------------------------------------------------------
// The clear-text run, with every intermediate kept, because every calibration
// constant is read off one of them.
struct ClearRun {
  std::vector<double> normed, q, k, v, scores, probs, attn, proj, hidden;
  std::vector<double> fnormed, gate, up, down, out;
  std::vector<double> row_max;  //!< the per-row shift, head * T + query
  double score_max = 0.0, score_min = 0.0, span = 0.0, mask_gap = 0.0;
  double norm_lo = 0.0, norm_hi = 0.0;
  double attn_alpha = 1.0, ffn_alpha = 1.0;
  double attn_window = 1.0, ffn_window = 1.0;
};

void Gemm(const std::vector<double> &a, const std::vector<double> &b, int rows,
          int inner, int outer, std::vector<double> &out) {
  out.assign(static_cast<size_t>(rows) * outer, 0.0);
  for (int t = 0; t < rows; t++) {
    double *op = &out[static_cast<size_t>(t) * outer];
    for (int i = 0; i < inner; i++) {
      const double av = a[static_cast<size_t>(t) * inner + i];
      if (av == 0.0) continue;
      const double *bp = &b[static_cast<size_t>(i) * outer];
      for (int o = 0; o < outer; o++) op[o] += av * bp[o];
    }
  }
}

void RmsNormWithStats(const std::vector<double> &in,
                      const std::vector<double> &g, int rows, int cols,
                      std::vector<double> &out, double &alpha, double &window) {
  out.assign(static_cast<size_t>(rows) * cols, 0.0);
  double log_sum = 0.0, lo = 1e300, hi = 0.0;
  for (int t = 0; t < rows; t++) {
    double sq = 0.0;
    for (int c = 0; c < cols; c++) {
      const double u = in[static_cast<size_t>(t) * cols + c];
      sq += u * u;
    }
    const double ms = sq / cols;
    log_sum += std::log(ms);
    lo = std::min(lo, ms);
    hi = std::max(hi, ms);
    const double inv = 1.0 / std::sqrt(ms + kEps);
    for (int c = 0; c < cols; c++) {
      out[static_cast<size_t>(t) * cols + c] =
          in[static_cast<size_t>(t) * cols + c] * inv * g[c];
    }
  }
  alpha = 1.0 / std::exp(log_sum / rows);
  window = hi / lo;
}

void ApplyRoPe(std::vector<double> &m, int rows, int width, int head_dim,
               int first_position, double theta) {
  const int half = head_dim / 2;
  const std::vector<double> src(m);
  for (int t = 0; t < rows; t++) {
    const double p = first_position + t;
    for (int c = 0; c < width; c++) {
      const int j = c % head_dim;
      const int base = c - j;
      const double f =
          std::pow(theta, -2.0 * (j % half) / static_cast<double>(head_dim));
      const double cs = std::cos(p * f), sn = std::sin(p * f);
      const size_t idx = static_cast<size_t>(t) * width + c;
      const size_t partner = static_cast<size_t>(t) * width + base +
                             ((j < half) ? j + half : j - half);
      m[idx] = src[idx] * cs + (j < half ? -1.0 : 1.0) * src[partner] * sn;
    }
  }
}

// The block in the clear, keeping everything. `probs` is the true SoftMax; the
// encrypted path's iteration reproduces it exactly in exact arithmetic, so any
// difference is approximation and not algorithm (SoftMax.h).
void TraceClearBlock(const Block::Config &cfg, const Block::Weights &w,
                     const std::vector<double> &x, ClearRun &r) {
  const int T = cfg.num_tokens, H = cfg.num_channels, KV = cfg.num_kv_channels;
  const int D = cfg.head_dim, F = cfg.hidden;
  const int heads = H / D, group = heads / (KV / D);

  RmsNormWithStats(x, w.attn_norm, T, H, r.normed, r.attn_alpha, r.attn_window);
  Gemm(r.normed, w.wq, T, H, H, r.q);
  Gemm(r.normed, w.wk, T, H, KV, r.k);
  Gemm(r.normed, w.wv, T, H, KV, r.v);
  ApplyRoPe(r.q, T, H, D, cfg.first_position, cfg.rope_theta);
  ApplyRoPe(r.k, T, KV, D, cfg.first_position, cfg.rope_theta);

  const double inv_root = 1.0 / std::sqrt(static_cast<double>(D));
  r.scores.assign(static_cast<size_t>(heads) * T * T, 0.0);
  r.row_max.assign(static_cast<size_t>(heads) * T, 0.0);
  r.score_max = -1e300;
  r.score_min = 1e300;
  r.span = 0.0;
  // Pass one: the scores, and the per-row *causal* maximum. Causal, because
  // that is the entry the row will normalise by, and taking the maximum over
  // every entry instead is what left ||y||^2 spanning 1.35e5x.
  for (int hd = 0; hd < heads; hd++) {
    const int kvh = hd / group;
    for (int t = 0; t < T; t++) {
      double causal_hi = -1e300;
      for (int u = 0; u < T; u++) {
        double dot = 0.0;
        for (int d = 0; d < D; d++) {
          dot += r.q[static_cast<size_t>(t) * H + hd * D + d] *
                 r.k[static_cast<size_t>(u) * KV + kvh * D + d];
        }
        const double s = dot * inv_root;
        r.scores[(static_cast<size_t>(hd) * T + t) * T + u] = s;
        if (u <= t) causal_hi = std::max(causal_hi, s);
        r.score_max = std::max(r.score_max, s);
        r.score_min = std::min(r.score_min, s);
      }
      r.row_max[static_cast<size_t>(hd) * T + t] = causal_hi;
    }
  }
  // Pass two: how far above its causal maximum a row's masked entries reach.
  // The exponential runs on them -- the mask only zeroes them a level later --
  // so they have to be pushed back into the interval, and the range then has
  // to cover them where they land.
  r.mask_gap = 0.0;
  for (int hd = 0; hd < heads; hd++) {
    for (int t = 0; t < T; t++) {
      const double c = r.row_max[static_cast<size_t>(hd) * T + t];
      for (int u = t + 1; u < T; u++) {
        r.mask_gap = std::max(
            r.mask_gap,
            r.scores[(static_cast<size_t>(hd) * T + t) * T + u] - c);
      }
    }
  }
  r.span = 0.0;
  for (int hd = 0; hd < heads; hd++) {
    for (int t = 0; t < T; t++) {
      const double c = r.row_max[static_cast<size_t>(hd) * T + t];
      for (int u = 0; u < T; u++) {
        const double shifted =
            r.scores[(static_cast<size_t>(hd) * T + t) * T + u] - c -
            (u > t ? r.mask_gap : 0.0);
        r.span = std::max(r.span, -shifted);
      }
    }
  }

  // ||y||_2^2 per row, with y = exp((s - c_r)/2) -- the half is the k = 1
  // iteration's own, since exp is evaluated on [-M/2, 0] and the squaring at
  // the end of the iteration restores the other half -- and the causal mask
  // applied. This is what the inverse square root actually sees.
  r.probs.assign(r.scores.size(), 0.0);
  r.norm_lo = 1e300;
  r.norm_hi = 0.0;
  for (int hd = 0; hd < heads; hd++) {
    for (int t = 0; t < T; t++) {
      const double c = r.row_max[static_cast<size_t>(hd) * T + t];  // causal
      double sq = 0.0, sum = 0.0;
      for (int u = 0; u <= t; u++) {
        const double half = std::exp(
            0.5 * (r.scores[(static_cast<size_t>(hd) * T + t) * T + u] - c));
        sq += half * half;
        const double y = half * half;
        sum += y;
        r.probs[(static_cast<size_t>(hd) * T + t) * T + u] = y;
      }
      r.norm_lo = std::min(r.norm_lo, sq);
      r.norm_hi = std::max(r.norm_hi, sq);
      for (int u = 0; u <= t; u++) {
        r.probs[(static_cast<size_t>(hd) * T + t) * T + u] /= sum;
      }
    }
  }

  r.attn.assign(static_cast<size_t>(T) * H, 0.0);
  for (int hd = 0; hd < heads; hd++) {
    const int kvh = hd / group;
    for (int t = 0; t < T; t++) {
      for (int u = 0; u <= t; u++) {
        const double p = r.probs[(static_cast<size_t>(hd) * T + t) * T + u];
        for (int d = 0; d < D; d++) {
          r.attn[static_cast<size_t>(t) * H + hd * D + d] +=
              p * r.v[static_cast<size_t>(u) * KV + kvh * D + d];
        }
      }
    }
  }
  Gemm(r.attn, w.wo, T, H, H, r.proj);
  r.hidden.assign(static_cast<size_t>(T) * H, 0.0);
  for (size_t i = 0; i < r.hidden.size(); i++) r.hidden[i] = x[i] + r.proj[i];

  RmsNormWithStats(r.hidden, w.ffn_norm, T, H, r.fnormed, r.ffn_alpha,
                   r.ffn_window);
  Gemm(r.fnormed, w.wgate, T, H, F, r.gate);
  Gemm(r.fnormed, w.wup, T, H, F, r.up);
  std::vector<double> act(r.gate.size());
  for (size_t i = 0; i < act.size(); i++) {
    act[i] = r.gate[i] / (1.0 + std::exp(-r.gate[i])) * r.up[i];
  }
  Gemm(act, w.wdown, T, F, H, r.down);
  r.out.assign(r.hidden.size(), 0.0);
  for (size_t i = 0; i < r.out.size(); i++) r.out[i] = r.hidden[i] + r.down[i];
}

}  // namespace

// ---------------------------------------------------------------------------
// THE STAND-IN.
//
// Decrypt, multiply on the host, re-encrypt at level 0 in the coefficient
// domain -- which is exactly where the real product would leave the result.
// Everything the real path has to get right about layout is here in the clear,
// so when it is written this class is the specification it has to match.
class HostLinearLeg : public Block::LinearLeg {
 public:
  HostLinearLeg(const Testbed32 &bed, const Packing &pack,
                const Block::Config &cfg)
      : bed_{bed}, pack_{pack}, cfg_{cfg} {}

  void Project(std::vector<Ciphertext<word>> &res,
               const std::vector<Ciphertext<word>> &x, int in_channels,
               int out_channels, const std::vector<double> &w, double w_scale,
               const char *name) const override {
    std::vector<double> a;
    Gather(a, x, in_channels);
    std::vector<double> y;
    Gemm(a, w, pack_.tokens, in_channels, out_channels, y);
    for (double &u : y) u *= w_scale;
    log_.emplace_back(name, MaxAbs(y));
    Scatter(res, y, out_channels);
  }

  void Scores(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &q,
              const std::vector<Ciphertext<word>> &k, double scale,
              const std::vector<double> &shift) const override {
    std::vector<double> qa, ka;
    Gather(qa, q, cfg_.num_channels);
    Gather(ka, k, cfg_.num_kv_channels);
    const int T = pack_.tokens, D = cfg_.head_dim;
    const int heads = cfg_.num_channels / D;
    const int group = heads / (cfg_.num_kv_channels / D);
    std::vector<double> s(static_cast<size_t>(heads) * T * T, 0.0);
    for (int hd = 0; hd < heads; hd++) {
      const int kvh = hd / group;
      for (int t = 0; t < T; t++) {
        for (int u = 0; u < T; u++) {
          double dot = 0.0;
          for (int d = 0; d < D; d++) {
            dot += qa[static_cast<size_t>(t) * cfg_.num_channels + hd * D + d] *
                   ka[static_cast<size_t>(u) * cfg_.num_kv_channels + kvh * D + d];
          }
          s[(static_cast<size_t>(hd) * T + t) * T + u] =
              dot * scale + shift[(static_cast<size_t>(hd) * T + t) * T + u];
        }
      }
    }
    log_.emplace_back("scores", MaxAbs(s));
    ScatterScores(res, s);
  }

  void Values(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &p,
              const std::vector<Ciphertext<word>> &v,
              double scale) const override {
    std::vector<double> pa, va;
    GatherScores(pa, p);
    Gather(va, v, cfg_.num_kv_channels);
    const int T = pack_.tokens, D = cfg_.head_dim, H = cfg_.num_channels;
    const int heads = H / D;
    const int group = heads / (cfg_.num_kv_channels / D);
    std::vector<double> y(static_cast<size_t>(T) * H, 0.0);
    for (int hd = 0; hd < heads; hd++) {
      const int kvh = hd / group;
      for (int t = 0; t < T; t++) {
        for (int u = 0; u < T; u++) {
          const double pw = pa[(static_cast<size_t>(hd) * T + t) * T + u];
          if (pw == 0.0) continue;
          for (int d = 0; d < D; d++) {
            y[static_cast<size_t>(t) * H + hd * D + d] +=
                pw * va[static_cast<size_t>(u) * cfg_.num_kv_channels + kvh * D + d];
          }
        }
      }
    }
    for (double &u : y) u *= scale;
    log_.emplace_back("attn", MaxAbs(y));
    Scatter(res, y, H);
  }

  // What every product produced, so a magnitude that leaves the bootstrap's
  // range is attributable to a stage rather than to the block as a whole.
  mutable std::vector<std::pair<std::string, double>> log_;

  // The last thing each product read, kept so a stage can be compared against
  // the clear run at the point it entered the product rather than only at the
  // end of the block.
  mutable std::vector<std::pair<std::string, std::vector<double>>> seen_;

 private:
  void Gather(std::vector<double> &out,
              const std::vector<Ciphertext<word>> &cts, int channels) const {
    out.assign(static_cast<size_t>(pack_.tokens) * channels, 0.0);
    const int cpc = pack_.channels_per_ct();
    for (size_t i = 0; i < cts.size(); i++) {
      Plaintext<word> ptxt;
      bed_.interface_->Decrypt(ptxt, cts[i]);
      std::vector<double> coeffs;
      bed_.context_->encoder_.DecodeCoeff(coeffs, ptxt);
      for (int s = 0; s < pack_.slots; s++) {
        const int t = s % pack_.tokens;
        const int c = static_cast<int>(i) * cpc + s / pack_.tokens;
        out[static_cast<size_t>(t) * channels + c] = coeffs[pack_.coeff(s)];
      }
    }
  }

  void Scatter(std::vector<Ciphertext<word>> &res, const std::vector<double> &v,
               int channels) const {
    const int cpc = pack_.channels_per_ct();
    const int num_ct = channels / cpc;
    res.resize(num_ct);
    for (int i = 0; i < num_ct; i++) {
      std::vector<double> coeffs(pack_.degree, 0.0);
      for (int s = 0; s < pack_.slots; s++) {
        const int t = s % pack_.tokens;
        const int c = i * cpc + s / pack_.tokens;
        coeffs[pack_.coeff(s)] = v[static_cast<size_t>(t) * channels + c];
      }
      Plaintext<word> ptxt;
      bed_.context_->encoder_.EncodeCoeff(
          ptxt, 0, bed_.context_->param_.GetScale(0), coeffs);
      bed_.interface_->Encrypt(res[i], ptxt);
    }
  }

  void GatherScores(std::vector<double> &out,
                    const std::vector<Ciphertext<word>> &cts) const {
    const int T = pack_.tokens, rows = pack_.rows_per_ct();
    out.assign(cts.size() * static_cast<size_t>(pack_.slots), 0.0);
    for (size_t i = 0; i < cts.size(); i++) {
      Plaintext<word> ptxt;
      bed_.interface_->Decrypt(ptxt, cts[i]);
      std::vector<double> coeffs;
      bed_.context_->encoder_.DecodeCoeff(coeffs, ptxt);
      for (int r = 0; r < rows; r++) {
        for (int j = 0; j < T; j++) {
          out[(i * rows + r) * T + j] = coeffs[pack_.coeff(r + j * rows)];
        }
      }
    }
  }

  void ScatterScores(std::vector<Ciphertext<word>> &res,
                     const std::vector<double> &v) const {
    const int T = pack_.tokens, rows = pack_.rows_per_ct();
    const int num_ct = static_cast<int>(v.size()) / (rows * T);
    res.resize(num_ct);
    for (int i = 0; i < num_ct; i++) {
      std::vector<double> coeffs(pack_.degree, 0.0);
      for (int r = 0; r < rows; r++) {
        for (int j = 0; j < T; j++) {
          coeffs[pack_.coeff(r + j * rows)] =
              v[(static_cast<size_t>(i) * rows + r) * T + j];
        }
      }
      Plaintext<word> ptxt;
      bed_.context_->encoder_.EncodeCoeff(
          ptxt, 0, bed_.context_->param_.GetScale(0), coeffs);
      bed_.interface_->Encrypt(res[i], ptxt);
    }
  }

  const Testbed32 &bed_;
  Packing pack_;
  Block::Config cfg_;
};

class LlamaBlockFixture : public Testbed32 {
 protected:
  int BootSlackLevels() const override { return kSlack; }

  static Block::Config MakeConfig() {
    Block::Config cfg;
    cfg.num_tokens = kTokens;
    cfg.num_channels = kChannels;
    cfg.num_kv_channels = kKvChannels;
    cfg.hidden = kHidden;
    cfg.head_dim = kHeadDim;
    cfg.first_position = kFirstToken;
    cfg.eps = kEps;
    return cfg;
  }

  bool LoadWeights(Block::Weights &w, std::vector<double> &x) {
    const std::string d = DataDir();
    if (d.empty()) return false;
    std::vector<double> all;
    if (!ReadF32(d + "/input.f32", static_cast<size_t>(kAllTokens) * kChannels,
                 all))
      return false;
    x.assign(
        all.begin() + static_cast<size_t>(kFirstToken) * kChannels,
        all.begin() + static_cast<size_t>(kFirstToken + kTokens) * kChannels);
    return ReadF32(d + "/attn_norm.f32", kChannels, w.attn_norm) &&
           ReadF32(d + "/ffn_norm.f32", kChannels, w.ffn_norm) &&
           ReadF32(d + "/wq.f32", static_cast<size_t>(kChannels) * kChannels,
                   w.wq) &&
           ReadF32(d + "/wk.f32", static_cast<size_t>(kChannels) * kKvChannels,
                   w.wk) &&
           ReadF32(d + "/wv.f32", static_cast<size_t>(kChannels) * kKvChannels,
                   w.wv) &&
           ReadF32(d + "/wo.f32", static_cast<size_t>(kChannels) * kChannels,
                   w.wo) &&
           ReadF32(d + "/wgate.f32", static_cast<size_t>(kChannels) * kHidden,
                   w.wgate) &&
           ReadF32(d + "/wup.f32", static_cast<size_t>(kChannels) * kHidden,
                   w.wup) &&
           ReadF32(d + "/wdown.f32", static_cast<size_t>(kHidden) * kChannels,
                   w.wdown);
  }

  // Every constant [SYLPH] calls calibration, read off the clear run.
  static Block::Calibration Calibrate(const ClearRun &r,
                                      const std::vector<double> &x) {
    Block::Calibration cal;
    cal.boot_max = kBootMax;
    // One constant for the whole residual chain: the input, the O projection's
    // output and the down projection's output are added to each other, so they
    // cannot be sized separately.
    cal.residual =
        kBootMax / std::max({MaxAbs(x), MaxAbs(r.hidden), MaxAbs(r.out)});
    cal.attn_alpha = r.attn_alpha;
    cal.ffn_alpha = r.ffn_alpha;
    // One handler shape serves both norms, so the window has to cover the
    // wider of the two spreads.
    cal.rms_window = std::max(r.attn_window, r.ffn_window);
    cal.rms_degree = 7;

    // One shift per row, its own maximum. The range only has to cover the
    // widest single row, not the dispersion across rows, which is the other
    // half of what the per-row shift buys.
    cal.softmax_shift = r.row_max;
    cal.softmax_mask_shift = r.mask_gap;
    cal.softmax_range = r.span;
    cal.softmax_iters = 1;
    // Degree 15 costs the same four levels as degree 9 -- EvalPoly spends
    // ceil(log2(d+1)) -- and the range now has to cover the pushed-down
    // masked entries as well, so there is no reason to take the smaller fit.
    cal.softmax_exp_degree = 15;
    cal.softmax_inv_sqrt_degree = 31;
    cal.softmax_early_inv_sqrt_degree = 4;
    cal.softmax_norm_lo = {r.norm_lo};
    cal.softmax_norm_hi = {r.norm_hi};

    // SiLU's interval is the measured maximum, tight, because the crossing is
    // handled by the free growth at Canonicalise rather than by widening the
    // fit. Widening would cost roughly a doubling of degree, which is a level.
    cal.silu_range = MaxAbs(r.gate) * 1.02;
    cal.silu_degree = 15;

    cal.size_q = kBootMax / MaxAbs(r.q);
    cal.size_k = kBootMax / MaxAbs(r.k);
    cal.size_v = kBootMax / MaxAbs(r.v);
    cal.size_scores = kBootMax;  // (s - c) / M already lies in [-1, 0]
    cal.size_attn = kBootMax / MaxAbs(r.attn);
    cal.size_gate = kBootMax;  // the ciphertext carries size_gate * g / range
    cal.size_up = kBootMax / MaxAbs(r.up);
    return cal;
  }
};

INSTANTIATE_TEST_SUITE_P(Cheddar, LlamaBlockFixture,
                         testing::Values("bootparam_35.json"),
                         [](const testing::TestParamInfo<const char *> &info) {
                           std::string name = info.param;
                           name = name.substr(0, name.find('.'));
                           return name + "_json";
                         });

// ---------------------------------------------------------------------------
// 1. Arithmetic only. Does the block fit, and what does it cost?
//
// Nothing is encrypted, so a plan that does not close is found in milliseconds
// rather than after key generation. It also prints the bootstrap count, which
// is the block's real cost and is not obvious from the operator list -- two of
// the six turns exist only to carry a tensor across a level boundary.
TEST_P(LlamaBlockFixture, ThePlanClosesForTheBlock) {
  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  boot->PrepareEvalMod();

  Block::Calibration cal;
  Block block(boot, MakeConfig(), cal);
  std::cout << block.DescribePlan() << std::endl;

  std::string why;
  EXPECT_TRUE(block.Fits(&why)) << why;
}

// ---------------------------------------------------------------------------
// 2. The calibration, in the clear, with no GPU.
//
// Every constant the encrypted run needs, printed against the interval the
// operator that consumes it was built for. A block that cannot be calibrated
// cannot be run, and finding that out here costs a second instead of an hour.
TEST_P(LlamaBlockFixture, TheCalibrationIsReachable) {
  Block::Weights w;
  std::vector<double> x;
  if (!LoadWeights(w, x)) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  const auto cfg = MakeConfig();
  ClearRun r;
  TraceClearBlock(cfg, w, x, r);
  const auto cal = Calibrate(r, x);

  std::cout << "residual chain: |x| " << MaxAbs(x) << ", |h| "
            << MaxAbs(r.hidden) << ", |out| " << MaxAbs(r.out)
            << " -> residual constant " << cal.residual << std::endl;
  std::cout << "RMSNorm windows: attn " << r.attn_window << "x, ffn "
            << r.ffn_window << "x  (RmsNorm.h: degree 7 reaches 12 bits at "
            << "4.18x, degree 9 at 6x, degree 23 at 30x)" << std::endl;
  std::cout << "RMSNorm layer constants: attn " << cal.attn_alpha << ", ffn "
            << cal.ffn_alpha << std::endl;
  std::cout << "SoftMax: scores in [" << r.score_min << ", " << r.score_max
            << "] globally; widest single row " << cal.softmax_range
            << " -> range M" << std::endl;
  std::cout << "SoftMax per-row shift: " << cal.softmax_shift.size()
            << " causal row maxima, in ["
            << *std::min_element(r.row_max.begin(), r.row_max.end()) << ", "
            << *std::max_element(r.row_max.begin(), r.row_max.end())
            << "] -- that spread is what one global shift would have to absorb"
            << std::endl;
  std::cout << "SoftMax masked-entry gap: " << cal.softmax_mask_shift
            << ", so the range carries it and every argument stays in "
            << "[-1, 0] before the +1" << std::endl;
  std::cout << "SoftMax ||y||^2 over rows: [" << r.norm_lo << ", " << r.norm_hi
            << "], spread " << (r.norm_hi / r.norm_lo)
            << "x  (with one global shift this measured 5.39e9x, which no "
            << "degree covers)" << std::endl;
  std::cout << "SiLU: |gate| " << MaxAbs(r.gate) << " -> range "
            << cal.silu_range << std::endl;
  std::cout << "crossing magnitudes: |q| " << MaxAbs(r.q) << ", |k| "
            << MaxAbs(r.k) << ", |v| " << MaxAbs(r.v) << ", |attn| "
            << MaxAbs(r.attn) << ", |up| " << MaxAbs(r.up) << std::endl;

  // The one that is not a matter of degree. Every other constant can be
  // absorbed by a wider interval at the cost of levels; this one is a property
  // of the data and if the spread is too wide there is no polynomial that
  // covers it.
  EXPECT_GT(r.norm_lo, 0.0);
}

// ---------------------------------------------------------------------------
// 3. The block, encrypted, end to end.
TEST_P(LlamaBlockFixture, TheBlockRunsEndToEnd) {
  Block::Weights w;
  std::vector<double> x;
  if (!LoadWeights(w, x)) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  const auto cfg = MakeConfig();

  ClearRun r;
  TraceClearBlock(cfg, w, x, r);
  const auto cal = Calibrate(r, x);

  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);
  Block block(boot, cfg, cal);
  std::string why;
  ASSERT_TRUE(block.Fits(&why)) << why;
  std::cout << block.DescribePlan() << std::endl;

  EvkRequest req;
  block.AddRequiredRotations(req);
  interface_->PrepareRotationKey(req);

  Packing pack;
  pack.tokens = kTokens;
  pack.slots = num_slots;
  pack.degree = param_->degree_;

  // The input, coefficient encoded at level 0 -- where the previous block's
  // down projection would have left it.
  const int num_ct = kChannels / pack.channels_per_ct();
  std::vector<Ciphertext<word>> state(num_ct);
  for (int i = 0; i < num_ct; i++) {
    std::vector<double> coeffs(pack.degree, 0.0);
    for (int s = 0; s < num_slots; s++) {
      const int t = s % kTokens;
      const int c = i * pack.channels_per_ct() + s / kTokens;
      coeffs[pack.coeff(s)] =
          cal.residual * x[static_cast<size_t>(t) * kChannels + c];
    }
    Plaintext<word> ptxt;
    context_->encoder_.EncodeCoeff(ptxt, 0, param_->GetScale(0), coeffs);
    interface_->Encrypt(state[i], ptxt);
  }

  // The causal mask, in the score packing. Causality is public, so it is a
  // plaintext.
  const int heads = kChannels / kHeadDim;
  const int rows = pack.rows_per_ct();
  const int num_score_ct = heads * kTokens / rows;
  std::vector<std::vector<Complex>> mask(
      num_score_ct, std::vector<Complex>(num_slots, Complex(0.0, 0.0)));
  for (int i = 0; i < num_score_ct; i++) {
    for (int rr = 0; rr < rows; rr++) {
      const int query = (i * rows + rr) % kTokens;
      for (int j = 0; j < kTokens; j++) {
        mask[i][rr + j * rows] = Complex(j <= query ? 1.0 : 0.0, 0.0);
      }
    }
  }

  HostLinearLeg leg(*this, pack, cfg);
  std::vector<Ciphertext<word>> out;
  block.Run(out, state, w, mask, leg, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "product-leg magnitudes, against the bootstrap's " << kBootMax
            << ":" << std::endl;
  for (const auto &e : leg.log_) {
    std::cout << "   " << e.first << " max " << e.second << std::endl;
  }

  // The output carries `residual`, coefficient encoded at level 0.
  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_ct; i++) {
    Plaintext<word> ptxt;
    interface_->Decrypt(ptxt, out[i]);
    std::vector<double> coeffs;
    context_->encoder_.DecodeCoeff(coeffs, ptxt);
    for (int s = 0; s < num_slots; s++) {
      const int t = s % kTokens;
      const int c = i * pack.channels_per_ct() + s / kTokens;
      const double want = r.out[static_cast<size_t>(t) * kChannels + c];
      const double got = coeffs[pack.coeff(s)] / cal.residual;
      worst = std::max(worst, std::abs(got - want));
      absmax = std::max(absmax, std::abs(want));
    }
  }
  std::cout << "one whole block vs the clear model: max abs err " << worst
            << " against |out| <= " << absmax << " ("
            << -std::log2(worst / absmax) << " bits)" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 4.0);
}
