// One whole Llama-3-8B decoder block, assembled.
//
// SylphScheduleTest proved one turn of [SYLPH] figure 2's cycle and then two
// turns of it, with RMSNorm in the slot leg and a descent to level 0 standing
// in for the product. This is the block: all four non-linear operators, in the
// order the model puts them, with the residual connections, on real layer-2
// weights.
//
// WHAT IS COMPUTED, stated plainly and up front.
//
// Llama-3-8B layer 2 on a 128-token prompt, T = 128, first position 0. Query
// 127 attends to keys 0..127, sinks included. The final comparison is against
// the true layer -- the one computed on the real hidden state of every token,
// with nothing substituted -- over tokens 2..127, which are the tokens the
// encrypted layer is being run for.
//
// This ran at kTokens = 64 from token 64 until now, which is attention over a
// 64-token window with no key history and is a different function. Because the
// clear reference truncated identically, no error figure could see it.
//
// The two leading tokens are attention sinks and their hidden state does not
// go through the encrypted RMSNorm, because no polynomial degree covers a
// 285,946x window. See kSinkTokens below for what happens instead; the
// arrangement is exact off the sink rows and the test checks that in the clear
// before it encrypts anything.
//
// WHAT IS ENCRYPTED AND WHAT IS NOT.
//
// Encrypted, on the GPU, in the real parameter set, in both tests below: every
// bootstrap, every slot/coefficient conversion, RMSNorm twice, RoPE, SoftMax,
// SiLU, the elementwise gate, and both residual additions.
//
// `TheBlockRunsWithEncryptedProjections` additionally runs **five of the seven
// matrix products** for real -- Q, K, V, O, gate, up and down go through
// `CoeffLinearLeg`: ModDecomp onto the channel axis, the Bae PC-MM, ModPack,
// one rescale, at the block's own ring degree with no ring switch. That is
// seven `Project` calls; the two that remain are `Scores` (Q K^T) and `Values`
// (P V), which are ciphertext-ciphertext products and need a different
// primitive.
//
// Those two still go through `HostLinearLeg`, which decrypts, multiplies in
// the clear and re-encrypts. It is a stand-in and it is labelled as one
// everywhere it appears.
//
// `TheBlockRunsEndToEnd` keeps every product on the host, so the difference
// between the two numbers is exactly what the real projections cost.
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
#include <memory>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "Testbed.h"
#include "extension/AttentionPacking.h"
#include "extension/LlamaAttention.h"
#include "extension/LlamaBlock.h"
#include "extension/LlamaLinear.h"

using word = uint32_t;
using Block = LlamaBlock<word>;

namespace {

int SlackFromEnv() {
  const char *env = std::getenv("CHEDDAR_SLACK");
  return env ? std::atoi(env) : 8;
}
const int kSlack = SlackFromEnv();

// How many input ciphertexts CoeffLinearLeg decomposes at once. Each one costs
// `rank` module components of the parent's own size -- about 201 MB at level 1
// on bootparam_35 -- and the block is already holding a dozen tensors and
// every bootstrap key by the time the O projection runs. Four is 0.8 GB.
// Bigger is faster (one ModPack per output group per tile is the whole cost of
// a tile) and this is the knob to turn when a card has room.
int TileFromEnv() {
  const char *env = std::getenv("CHEDDAR_PARENTS_PER_TILE");
  return env ? std::atoi(env) : 4;
}
const int kParentsPerTile = TileFromEnv();

constexpr int kAllTokens = 128;
constexpr int kChannels = 4096;
constexpr int kKvChannels = 1024;
constexpr int kHidden = 14336;
constexpr int kHeadDim = 128;
// THE WHOLE PROMPT, and that is the point.
//
// This ran at kTokens = 64 from kFirstToken = 64 until now, which is attention
// over a 64-token window with no key history: query 100 attended to keys
// 64..100 instead of 0..100, and keys 0..63 were not cached anywhere, they
// were absent. That is a different function from Llama-3, not an
// approximation of it, and because TraceClearBlock truncated identically the
// error figure could not see it.
//
// T = 128 is also the count the encrypted product needs. ModDecomp splits a
// parent into `degree / (2T)` module components and the block puts
// `slots / T` channels in a ciphertext; the Bae PC-MM separates channels only
// when those two agree, which at degree 65536 happens at T = 128 and nowhere
// else (Doing.md 1.5ac). Two holes, one constant.
constexpr int kTokens = kAllTokens;
constexpr int kFirstToken = 0;

// THE SINC LEG'S SHAPE. `sub_degree = 32` at ring degree 4096 gives d = 128 --
// which has to be both T and head_dim at once, and is -- and 16 lanes, so one
// batch CC-MM does 16 of the layer's 32 heads and every product is two calls.
constexpr int kSubDegree = 32;
constexpr int kSinCPhases = 3;
constexpr int kProductLevel = 1;

// The constant HalfBoot and the StC prefix leave when the magnitude is 1.
// Measured on sylphflow16_35 with slack 8 by SinCAttentionTest, which prints
// it; see SinCLinearLeg::Config::chain_constant for why it cannot be derived.
double ChainConstant() {
  const char *env = std::getenv("CHEDDAR_CHAIN_CONSTANT");
  return env ? std::atof(env) : 0.0298533;
}

// THE SINKS, AND WHY THEY ARE NOT MERELY SKIPPED.
//
// The prompt of `input_nosink.f32` is two beginning-of-sequence tokens
// followed by 126 text tokens. Those two are attention sinks and their hidden
// state is enormous: measured on this bundle, per-token mean square 33.1
// against 3.4e-4 for every other token, a 285,946x window. RmsNorm.h puts
// degree 23 at 30x, so no degree covers it, and a Chebyshev polynomial outside
// its interval grows like cosh(d arccosh(v)) -- the sink slots would take the
// whole ciphertext out of the next bootstrap's range, not just be wrong.
//
// They cannot be dropped either. Query t attends to keys 0..t, sinks included,
// and a query that does not is computing a different function -- which is the
// whole complaint against the 64-token window this test used to run.
//
// So they are handled the way [SYLPH] 3.1.1 handles them. A prefix of
// beginning-of-sequence tokens is prompt-independent, so its hidden state at
// every layer is a constant of the model and therefore public. A public
// rescaled copy of it stands in for the encrypted RMSNorm's input -- which
// brings the window down to 5.2x -- and the true K and V are added back as a
// plaintext correction straight after the projection. Nothing else changes,
// and the result is exact: only the sink rows of Q, of the attention output
// and of the FFN differ from the true layer, and those are discarded.
constexpr int kSinkTokens = 2;
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
  //! K and V as the projection leaves them: before RoPE and before the sink
  //! override. The block injects its correction at exactly this point, so
  //! these are the two halves the correction is built from.
  std::vector<double> k_pre, v_pre;
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
                     const std::vector<double> &x, ClearRun &r,
                     const std::vector<double> *sink_k = nullptr,
                     const std::vector<double> *sink_v = nullptr) {
  const int T = cfg.num_tokens, H = cfg.num_channels, KV = cfg.num_kv_channels;
  const int D = cfg.head_dim, F = cfg.hidden;
  const int heads = H / D, group = heads / (KV / D);

  RmsNormWithStats(x, w.attn_norm, T, H, r.normed, r.attn_alpha, r.attn_window);
  Gemm(r.normed, w.wq, T, H, H, r.q);
  Gemm(r.normed, w.wk, T, H, KV, r.k);
  Gemm(r.normed, w.wv, T, H, KV, r.v);
  // Kept before anything touches them: the block's sink correction is built
  // from exactly these rows, and it is added at exactly this point.
  r.k_pre = r.k;
  r.v_pre = r.v;
  // The sinks' true K and V, in the same place LlamaBlock::InjectSinks puts
  // them -- after the projection, before RoPE.
  if (sink_k != nullptr && cfg.num_sink_tokens > 0) {
    const size_t n = static_cast<size_t>(cfg.num_sink_tokens) * KV;
    std::copy(sink_k->begin(), sink_k->begin() + n, r.k.begin());
    std::copy(sink_v->begin(), sink_v->begin() + n, r.v.begin());
  }
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

  // WHERE THE STAND-IN HAS TO PUT THINGS. `Project` is unchanged -- level 0,
  // coefficient domain -- but `Scores` and `Values` are slot-domain now, and
  // the level they return at is the block's, not one this class can pick. The
  // block reports it; this takes it after construction, which is the only
  // ordering that works when the block also needs the leg to build itself.
  void SetLevels(int operand, int prob, int result) {
    operand_level_ = operand;
    prob_level_ = prob;
    result_level_ = result;
  }

  // The score layout of the block's own packing: row `head * T + query`, keys
  // strided by the rows one ciphertext holds. This is what SoftMax reads with
  // a group of one, which is what makes the stand-in a control rather than a
  // second implementation of the real leg's layout.
  void LocateScore(int head, int query, int key, int &ct,
                   int &slot) const override {
    const int rows = pack_.rows_per_ct();
    const int row = head * pack_.tokens + query;
    ct = row / rows;
    slot = (row % rows) + key * rows;
  }

  void Project(std::vector<Ciphertext<word>> &res,
               const std::vector<Ciphertext<word>> &x, int in_channels,
               int out_channels, const std::vector<double> &w, double w_scale,
               const char *name) const override {
    std::vector<double> a;
    Gather(a, x, in_channels);
    // What the product read is the encrypted turn's *result*, so recording it
    // here is the per-turn error ledger with no extra decryption: the block
    // hands every operator's output to a product and nowhere else.
    seen_.emplace_back(std::string("in:") + name, a);
    std::vector<double> y;
    Gemm(a, w, pack_.tokens, in_channels, out_channels, y);
    for (double &u : y) u *= w_scale;
    log_.emplace_back(name, MaxAbs(y));
    Scatter(res, y, out_channels);
  }

  void Scores(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &q,
              const std::vector<Ciphertext<word>> &k, double magnitude,
              const std::vector<double> &shift) const override {
    std::vector<double> qa, ka;
    GatherSlots(qa, q, cfg_.num_channels);
    GatherSlots(ka, k, cfg_.num_kv_channels);
    seen_.emplace_back("in:q", qa);
    seen_.emplace_back("in:k", ka);
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
          // SHIFT FIRST, THEN SCALE. On the real leg the shift goes in at
          // level 0 before the bootstrap and the scale comes out of the
          // transform after it, so what crosses is the centred score. The
          // stand-in has to compute the same function in the same order or the
          // two runs are not comparable.
          s[(static_cast<size_t>(hd) * T + t) * T + u] =
              (dot + shift[(static_cast<size_t>(hd) * T + t) * T + u]) *
              magnitude;
        }
      }
    }
    log_.emplace_back("scores", MaxAbs(s));
    ScatterScores(res, s);
  }

  void Values(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &p,
              const std::vector<Ciphertext<word>> &v,
              double magnitude) const override {
    std::vector<double> pa, va;
    GatherScores(pa, p);
    GatherSlots(va, v, cfg_.num_kv_channels);
    seen_.emplace_back("in:probs", pa);
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
    for (double &u : y) u *= magnitude;
    log_.emplace_back("attn", MaxAbs(y));
    ScatterSlots(res, y, H);
  }

  // What every product produced, so a magnitude that leaves the bootstrap's
  // range is attributable to a stage rather than to the block as a whole.
  mutable std::vector<std::pair<std::string, double>> log_;

  // The last thing each product read, kept so a stage can be compared against
  // the clear run at the point it entered the product rather than only at the
  // end of the block.
  mutable std::vector<std::pair<std::string, std::vector<double>>> seen_;

  // ---- the same two records, taken WITHOUT standing in for anything ------
  //
  // A probe decrypts a copy and puts the ciphertext back untouched; the
  // encrypted product then runs on it exactly as it would have. That is the
  // difference between this and Project above, and it is the whole difference:
  // one substitutes for the computation, the other only watches it. Keeping
  // the turn-by-turn ledger is worth a decryption in a test, because without
  // it a block that degrades reports one number and no address.
  void Probe(const std::vector<Ciphertext<word>> &x, int channels,
             const char *name) const {
    std::vector<double> a;
    Gather(a, x, channels);
    seen_.emplace_back(std::string("in:") + name, a);
  }

  void ProbeOut(const std::vector<Ciphertext<word>> &res, int channels,
                const char *name) const {
    std::vector<double> y;
    Gather(y, res, channels);
    log_.emplace_back(name, MaxAbs(y));
  }

  // The same two records, written by a probe that did its own decoding
  // because the tensor is not in this class's packing.
  void Record(const std::string &tag, const std::vector<double> &v) const {
    seen_.emplace_back(tag, v);
  }
  void RecordMax(const std::string &tag, const std::vector<double> &v) const {
    log_.emplace_back(tag, MaxAbs(v));
  }

 private:
  // The same packing as `Gather`, read as slots rather than as coefficients.
  // Both index the ciphertext identically -- channel `s / T`, token `s % T`;
  // what differs is which encoder reads it, and that is the whole content of
  // the seam this block was rewired around.
  void GatherSlots(std::vector<double> &out,
                   const std::vector<Ciphertext<word>> &cts,
                   int channels) const {
    out.assign(static_cast<size_t>(pack_.tokens) * channels, 0.0);
    const int cpc = pack_.channels_per_ct();
    for (size_t i = 0; i < cts.size(); i++) {
      Plaintext<word> ptxt;
      bed_.interface_->Decrypt(ptxt, cts[i]);
      std::vector<Complex> msg;
      bed_.context_->encoder_.Decode(msg, ptxt);
      for (int s = 0; s < pack_.slots; s++) {
        const int t = s % pack_.tokens;
        const int c = static_cast<int>(i) * cpc + s / pack_.tokens;
        out[static_cast<size_t>(t) * channels + c] = msg[s].real();
      }
    }
  }

  void ScatterSlots(std::vector<Ciphertext<word>> &res,
                    const std::vector<double> &v, int channels) const {
    AssertLevel();
    const int cpc = pack_.channels_per_ct();
    const int num_ct = channels / cpc;
    res.resize(num_ct);
    for (int i = 0; i < num_ct; i++) {
      std::vector<Complex> msg(pack_.slots, Complex(0.0, 0.0));
      for (int s = 0; s < pack_.slots; s++) {
        const int t = s % pack_.tokens;
        const int c = i * cpc + s / pack_.tokens;
        msg[s] = Complex(v[static_cast<size_t>(t) * channels + c], 0.0);
      }
      Plaintext<word> ptxt;
      bed_.context_->encoder_.Encode(
          ptxt, result_level_,
          bed_.context_->param_.GetScale(result_level_), msg);
      bed_.interface_->Encrypt(res[i], ptxt);
    }
  }

  void AssertLevel() const {
    ASSERT_GE(result_level_, 0)
        << "HostLinearLeg::SetLevels was never called, so the stand-in does "
           "not know where the block expects its results";
  }

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
      std::vector<Complex> msg;
      bed_.context_->encoder_.Decode(msg, ptxt);
      for (int r = 0; r < rows; r++) {
        for (int j = 0; j < T; j++) {
          out[(i * rows + r) * T + j] = msg[r + j * rows].real();
        }
      }
    }
  }

  void ScatterScores(std::vector<Ciphertext<word>> &res,
                     const std::vector<double> &v) const {
    AssertLevel();
    const int T = pack_.tokens, rows = pack_.rows_per_ct();
    const int num_ct = static_cast<int>(v.size()) / (rows * T);
    res.resize(num_ct);
    for (int i = 0; i < num_ct; i++) {
      std::vector<Complex> msg(pack_.slots, Complex(0.0, 0.0));
      for (int r = 0; r < rows; r++) {
        for (int j = 0; j < T; j++) {
          msg[r + j * rows] = Complex(
              v[(static_cast<size_t>(i) * rows + r) * T + j], 0.0);
        }
      }
      Plaintext<word> ptxt;
      bed_.context_->encoder_.Encode(
          ptxt, result_level_,
          bed_.context_->param_.GetScale(result_level_), msg);
      bed_.interface_->Encrypt(res[i], ptxt);
    }
  }

  const Testbed32 &bed_;
  Packing pack_;
  Block::Config cfg_;
  int operand_level_ = -1;
  int prob_level_ = -1;
  int result_level_ = -1;
};

// ---------------------------------------------------------------------------
// FIVE OF THE SEVEN PRODUCTS, FOR REAL.
//
// `CoeffLinearLeg` runs the Bae PC-MM on the block's own ciphertexts:
// ModDecomp splits the channel axis onto the ciphertext axis, one plaintext
// matrix product contracts it, ModPack puts the result back, one rescale lands
// it at level 0. No ring switch, no key material in the product itself, and
// the output layout is the input layout, which is what makes it a drop-in.
//
// `Scores` and `Values` are ciphertext-ciphertext products and are NOT
// implemented by that class -- it leaves them pure virtual precisely so that a
// caller has to say what it is using instead. Here that is `HostLinearLeg`,
// and it is still a stand-in, still labelled as one. What this class buys is
// that the QKV, O, gate, up and down projections are now real: the eight-level
// descent is performed rather than fabricated, the imaginary half of every
// polynomial is carried forward rather than zeroed, and the product's own
// approximation error is in the number.
class EncryptedProjectionLeg : public cheddar::CoeffLinearLeg<word> {
 private:
  using Base = cheddar::CoeffLinearLeg<word>;

 public:
  EncryptedProjectionLeg(cheddar::ConstContextPtr<word> context,
                         const Base::Config &lcfg,
                         std::vector<const EvaluationKey<word> *> keys,
                         const HostLinearLeg &host)
      : Base(context, lcfg, std::move(keys)), host_{host} {}

  void Project(std::vector<Ciphertext<word>> &res,
               const std::vector<Ciphertext<word>> &x, int in_channels,
               int out_channels, const std::vector<double> &w, double w_scale,
               const char *name) const override {
    host_.Probe(x, in_channels, name);
    Base::Project(res, x, in_channels, out_channels, w, w_scale, name);
    host_.ProbeOut(res, out_channels, name);
  }

  // Still the stand-in. Turn C's and turn B's products decrypt, multiply on
  // the host and re-encrypt.
  void Scores(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &q,
              const std::vector<Ciphertext<word>> &k, double magnitude,
              const std::vector<double> &shift) const override {
    host_.Scores(res, q, k, magnitude, shift);
  }

  void Values(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &p,
              const std::vector<Ciphertext<word>> &v,
              double magnitude) const override {
    host_.Values(res, p, v, magnitude);
  }

  // The stand-in's score layout, so the block builds the causal mask in it.
  void LocateScore(int head, int query, int key, int &ct,
                   int &slot) const override {
    host_.LocateScore(head, query, key, ct, slot);
  }

 private:
  const HostLinearLeg &host_;
};

// ---------------------------------------------------------------------------
// THE REAL LEG, WITH THE SAME LEDGER.
//
// `SinCLinearLeg` is the whole product path. What this adds is the per-turn
// record: every operator's output is handed to a product and to nothing else,
// so reading it at the product's door attributes the block's error to a turn.
//
// Reading it means undoing the leg's own layout, and that is the point of
// keeping the probe here rather than inside the leg: the probe has to know the
// permutation to invert it, and asking the leg for the permutation it applied
// is exactly the check that the permutation is what the block thinks it is.
class ProbedSinCLeg : public cheddar::SinCLinearLeg<word> {
 private:
  using Base = cheddar::SinCLinearLeg<word>;
  using Tensor = Block::LinearLeg::Tensor;

 public:
  ProbedSinCLeg(std::shared_ptr<const cheddar::BootContext<word>> boot,
                cheddar::ConstContextPtr<word> sw,
                cheddar::ConstContextPtr<word> small, const Base::Config &cfg,
                const cheddar::SinCAttention<word>::Config &acfg,
                const cheddar::SinCAttention<word>::Keys &keys,
                const cheddar::CoeffLinearLeg<word>::Config &lcfg,
                std::vector<const EvaluationKey<word> *> modpack_keys,
                const Testbed32 &bed, const Packing &pack,
                const HostLinearLeg &host)
      : Base(std::move(boot), std::move(sw), std::move(small), cfg, acfg, keys,
             lcfg, std::move(modpack_keys)),
        bed_{bed},
        pack_{pack},
        host_{host},
        cfg_{cfg} {}

  void Project(std::vector<Ciphertext<word>> &res,
               const std::vector<Ciphertext<word>> &x, int in_channels,
               int out_channels, const std::vector<double> &w, double w_scale,
               const char *name) const override {
    host_.Probe(x, in_channels, name);
    Base::Project(res, x, in_channels, out_channels, w, w_scale, name);
    host_.ProbeOut(res, out_channels, name);
  }

  void Scores(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &q,
              const std::vector<Ciphertext<word>> &k, double magnitude,
              const std::vector<double> &shift) const override {
    ProbeChannels(q, kChannels, Tensor::kQuery, "q");
    ProbeChannels(k, kKvChannels, Tensor::kKey, "k");
    Base::Scores(res, q, k, magnitude, shift);
    ProbeLanes(res, kTokens, "scores");
  }

  void Values(std::vector<Ciphertext<word>> &res,
              const std::vector<Ciphertext<word>> &p,
              const std::vector<Ciphertext<word>> &v,
              double magnitude) const override {
    ProbeLanes(p, kTokens, "probs");
    Base::Values(res, p, v, magnitude);
    // head_dim == num_tokens here, and not by coincidence: SinCAttention
    // asserts that the product's width is both, because d is one number. So
    // the score layout and the output layout are the same function and one
    // probe reads both.
    ProbeLanes(res, kHeadDim, "attn_out");
  }

 private:
  // A channel tensor in the leg's own packing, read back in the model's
  // [token][head * head_dim + channel] order.
  void ProbeChannels(const std::vector<Ciphertext<word>> &cts, int channels,
                     Tensor which, const char *name) const {
    std::vector<int> order;
    ChannelOrder(order, which);
    std::vector<int> inv(channels, 0);
    for (int c = 0; c < channels; c++) inv[order[c]] = c;
    std::vector<double> out(static_cast<size_t>(kTokens) * channels, 0.0);
    const int cpc = pack_.channels_per_ct();
    for (size_t i = 0; i < cts.size(); i++) {
      Plaintext<word> ptxt;
      bed_.interface_->Decrypt(ptxt, cts[i]);
      std::vector<Complex> msg;
      bed_.context_->encoder_.Decode(msg, ptxt);
      for (int s = 0; s < pack_.slots; s++) {
        const int t = s % kTokens;
        const int p = static_cast<int>(i) * cpc + s / kTokens;
        out[static_cast<size_t>(t) * channels + inv[p]] = msg[s].real();
      }
    }
    host_.Record(std::string("in:") + name, out);
  }

  // A lane tensor -- scores, probabilities or the attention output -- read
  // back in [head][query][index] order through the leg's own score layout.
  void ProbeLanes(const std::vector<Ciphertext<word>> &cts, int width,
                  const char *name) const {
    std::vector<std::vector<Complex>> msg(cts.size());
    for (size_t i = 0; i < cts.size(); i++) {
      Plaintext<word> ptxt;
      bed_.interface_->Decrypt(ptxt, cts[i]);
      bed_.context_->encoder_.Decode(msg[i], ptxt);
    }
    const int heads = cfg_.num_heads;
    std::vector<double> out(
        static_cast<size_t>(heads) * kTokens * width, 0.0);
    for (int h = 0; h < heads; h++) {
      for (int t = 0; t < kTokens; t++) {
        for (int j = 0; j < width; j++) {
          int ct = 0, slot = 0;
          LocateScore(h, t, j, ct, slot);
          out[(static_cast<size_t>(h) * kTokens + t) * width + j] =
              msg[ct][slot].real();
        }
      }
    }
    host_.Record(std::string("in:") + name, out);
    host_.RecordMax(name, out);
  }

  const Testbed32 &bed_;
  Packing pack_;
  const HostLinearLeg &host_;
  Base::Config cfg_;
};

class LlamaBlockFixture : public Testbed32 {
 protected:
  int BootSlackLevels() const override { return kSlack; }

  // WHICH PRODUCTS ARE REAL. Everything else about the run is identical,
  // which is what makes the three numbers comparable.
  enum class Mode {
    kHost,         //!< every product on the host: the non-linear half alone
    kProjections,  //!< the seven plaintext products real, the two CC ones not
    kFull          //!< nothing on the host
  };
  void RunWholeBlock(Mode mode);

  static Block::Config MakeConfig() {
    Block::Config cfg;
    cfg.num_tokens = kTokens;
    cfg.num_channels = kChannels;
    cfg.num_kv_channels = kKvChannels;
    cfg.hidden = kHidden;
    cfg.head_dim = kHeadDim;
    cfg.first_position = kFirstToken;
    cfg.num_sink_tokens = kSinkTokens;
    cfg.eps = kEps;
    return cfg;
  }

  bool LoadWeights(Block::Weights &w, std::vector<double> &x) {
    const std::string d = DataDir();
    if (d.empty()) return false;
    // input_nosink.f32, not input.f32. Both are 128-token layer-2 bundles;
    // this one's prompt is `<bos> <bos> The ...`, so its sinks are exactly the
    // two leading tokens. input.f32's prompt has a third BOS at position 3, so
    // its sinks are {0, 1, 3} -- non-contiguous, and a leading-prefix
    // treatment would not cover it.
    std::vector<double> all;
    if (!ReadF32(d + "/input_nosink.f32",
                 static_cast<size_t>(kAllTokens) * kChannels, all))
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

  // The public stand-in for the sink tokens' hidden state.
  //
  // A rescaled copy of the true one. That is public -- a prefix of
  // beginning-of-sequence tokens is prompt-independent, so its hidden state at
  // layer 2 is a constant of the model -- and the scale is chosen to put its
  // mean square at the geometric centre of the other tokens' band, which is
  // where the layer constant will place the Chebyshev interval. It widens the
  // RMSNorm window by nothing.
  static std::vector<double> WithFilledSinks(const std::vector<double> &x) {
    std::vector<double> out(x);
    double log_sum = 0.0;
    for (int t = kSinkTokens; t < kTokens; t++) {
      double sq = 0.0;
      for (int c = 0; c < kChannels; c++) {
        const double u = x[static_cast<size_t>(t) * kChannels + c];
        sq += u * u;
      }
      log_sum += std::log(sq / kChannels);
    }
    const double target = std::exp(log_sum / (kTokens - kSinkTokens));
    for (int t = 0; t < kSinkTokens; t++) {
      double sq = 0.0;
      for (int c = 0; c < kChannels; c++) {
        const double u = x[static_cast<size_t>(t) * kChannels + c];
        sq += u * u;
      }
      const double f = std::sqrt(target / (sq / kChannels));
      for (int c = 0; c < kChannels; c++) {
        out[static_cast<size_t>(t) * kChannels + c] *= f;
      }
    }
    return out;
  }

  // The four public row blocks the block needs: the sinks' true K and V, and
  // what the filler makes the projection produce for them.
  static Block::PublicSinks MakeSinks(const ClearRun &truth,
                                      const ClearRun &filled) {
    const size_t n = static_cast<size_t>(kSinkTokens) * kKvChannels;
    Block::PublicSinks s;
    s.k.assign(truth.k_pre.begin(), truth.k_pre.begin() + n);
    s.v.assign(truth.v_pre.begin(), truth.v_pre.begin() + n);
    s.computed_k.assign(filled.k_pre.begin(), filled.k_pre.begin() + n);
    s.computed_v.assign(filled.v_pre.begin(), filled.v_pre.begin() + n);
    return s;
  }

  // Both clear runs, the filler and the public sink rows, in one place.
  //
  // `truth` is the layer as Llama-3 computes it, on the real hidden state of
  // every token. `filled` is what the encrypted path computes: the same layer
  // on the filled input, with the sinks' true K and V put back. They agree
  // exactly on every non-sink token, and the test asserts that rather than
  // assuming it -- it is the whole justification for the arrangement.
  static void BuildReference(const Block::Config &cfg,
                             const Block::Weights &w,
                             const std::vector<double> &x_true,
                             std::vector<double> &x_fill, ClearRun &truth,
                             ClearRun &filled, Block::PublicSinks &sinks) {
    Block::Config plain = cfg;
    plain.num_sink_tokens = 0;
    TraceClearBlock(plain, w, x_true, truth);

    x_fill = WithFilledSinks(x_true);
    ClearRun probe;
    TraceClearBlock(plain, w, x_fill, probe);  // no override: the "computed"
    sinks = MakeSinks(truth, probe);
    TraceClearBlock(cfg, w, x_fill, filled, &sinks.k, &sinks.v);
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

    // THE SCORE CROSSING IS A SECOND CONSTRAINT ON THE SAME TWO CONSTANTS,
    // AND IT IS NOT IMPLIED BY THE FIRST.
    //
    // On a slot-domain leg the score product has no plaintext operand, so
    // nothing scales its output before the bootstrap that follows it. What
    // that bootstrap carries is fixed by the operands and by the shift:
    // `size_q * size_k * sqrt(head_dim) * range / 2`, the shift having centred
    // each row on its own calibrated maximum. Sizing Q and K only against
    // their own maxima says nothing about that number, and it is a product of
    // two small constants against a sqrt(128) * 21 / 2 that is not small.
    //
    // Both are shrunk by the same factor when it binds, because there is no
    // reason to prefer one: the product is what has to fit and only the
    // product appears in it.
    const double crossing = cal.size_q * cal.size_k *
                            std::sqrt(static_cast<double>(kHeadDim)) *
                            r.span / 2.0;
    if (crossing > kBootMax) {
      const double shrink = std::sqrt(kBootMax / crossing);
      std::cout << "  the score crossing would be " << crossing << " against "
                << kBootMax << ", so size_q and size_k are both scaled by "
                << shrink << std::endl;
      cal.size_q *= shrink;
      cal.size_k *= shrink;
    }

    cal.size_v = kBootMax / MaxAbs(r.v);
    cal.size_scores = kBootMax;  // unused by a slot-domain leg; see the header
    cal.size_attn = kBootMax / MaxAbs(r.attn);
    cal.size_gate = kBootMax;  // the ciphertext carries size_gate * g / range
    cal.size_up = kBootMax / MaxAbs(r.up);
    return cal;
  }
};

INSTANTIATE_TEST_SUITE_P(Cheddar, LlamaBlockFixture,
                         testing::Values("bootparam_35.json",
                                         "sylphflow16_35.json"),
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
  Packing pack;
  pack.tokens = kTokens;
  pack.slots = param_->degree_ / 2;
  pack.degree = param_->degree_;
  HostLinearLeg host(*this, pack, MakeConfig());
  Block block(boot, MakeConfig(), cal, host);
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
  ClearRun truth, r;
  std::vector<double> x_fill;
  Block::PublicSinks sinks;
  BuildReference(cfg, w, x, x_fill, truth, r, sinks);
  const auto cal = Calibrate(r, x_fill);

  // THE CLAIM THE WHOLE SINK ARRANGEMENT RESTS ON, checked rather than
  // asserted: substituting a public filler for the sink tokens' hidden state
  // and putting their true K and V back changes nothing at all on the tokens
  // the layer is being run for.
  double sink_err = 0.0, sink_mag = 0.0;
  for (int t = kSinkTokens; t < kTokens; t++) {
    for (int c = 0; c < kChannels; c++) {
      const size_t i = static_cast<size_t>(t) * kChannels + c;
      sink_err = std::max(sink_err, std::abs(r.out[i] - truth.out[i]));
      sink_mag = std::max(sink_mag, std::abs(truth.out[i]));
    }
  }
  std::cout << "sinks: " << kSinkTokens << " public tokens; the filled run vs "
            << "the true layer on tokens " << kSinkTokens << ".." << (kTokens - 1)
            << ": max abs diff " << sink_err << " against |out| " << sink_mag
            << std::endl;
  EXPECT_LT(sink_err, 1e-12 * std::max(sink_mag, 1e-12))
      << "the public-filler substitution must be exact off the sink rows";
  std::cout << "RMSNorm window without the substitution: attn "
            << truth.attn_window << "x, ffn " << truth.ffn_window
            << "x -- which is why there is one" << std::endl;

  std::cout << "residual chain: |x| " << MaxAbs(x_fill) << ", |h| "
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
void LlamaBlockFixture::RunWholeBlock(Mode mode) {
  const bool encrypted_projections = (mode != Mode::kHost);
  Block::Weights w;
  std::vector<double> x;
  if (!LoadWeights(w, x)) GTEST_SKIP() << "LLAMA3_REAL_DIR is not set";

  auto boot = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot, nullptr);
  const int num_slots = param_->degree_ / 2;
  const auto cfg = MakeConfig();

  ClearRun truth, r;
  std::vector<double> x_fill;
  Block::PublicSinks sinks;
  BuildReference(cfg, w, x, x_fill, truth, r, sinks);
  const auto cal = Calibrate(r, x_fill);

  boot->PrepareEvalMod();
  boot->PrepareEvalSpecialFFT(num_slots);

  Packing pack;
  pack.tokens = kTokens;
  pack.slots = num_slots;
  pack.degree = param_->degree_;

  // THE PRODUCTS, BUILT BEFORE THE BLOCK.
  //
  // `host` is always constructed: with encrypted projections it is still the
  // stand-in for the two ciphertext-ciphertext products, and it is also where
  // the per-turn ledger lives, because the probes write into its records.
  //
  // The leg comes first because the block's own operators depend on its
  // layout -- RoPE's tables, SoftMax's group size and the causal mask all read
  // it -- so `LlamaBlock` takes it as a constructor argument. The levels go
  // the other way, and go in afterwards.
  HostLinearLeg host(*this, pack, cfg);
  std::unique_ptr<EncryptedProjectionLeg> real_leg;
  std::unique_ptr<ProbedSinCLeg> sinc_leg;
  std::unique_ptr<ringfixture::Ring<word>> sw_ring, small_ring;
  std::vector<const EvaluationKey<word> *> modpack_keys;

  const Block::LinearLeg *leg = &host;
  if (encrypted_projections) {
    // rank = degree / (2T) = the block's channels per ciphertext, and one
    // switching key per module component. These are the only keys the product
    // needs; the PC-MM itself uses none.
    using Leg = cheddar::CoeffLinearLeg<word>;
    Leg::Config lcfg;
    lcfg.num_tokens = kTokens;
    lcfg.product_level = 1;
    // 14336 input channels is 56 parents and about 11.3 GB of module
    // components held at once. Bounding it costs one extra ModPack per output
    // group per tile and nothing else; see LlamaLinear.h.
    lcfg.parents_per_tile = kParentsPerTile;

    const int small_degree = Leg::SmallDegreeFor(kTokens);
    const int rank = param_->degree_ / small_degree;
    size_t dev_free = 0, dev_total = 0;
    cudaMemGetInfo(&dev_free, &dev_total);
    std::cout << "preparing " << rank << " ModPack keys at level "
              << lcfg.product_level << " (small degree " << small_degree
              << "), " << lcfg.parents_per_tile << " parents per tile; device "
              << (dev_free >> 20) << " MiB free of " << (dev_total >> 20)
              << " MiB before keygen" << std::endl;
    interface_->PrepareModPackKeys(small_degree, lcfg.product_level);
    modpack_keys.resize(rank);
    for (int j = 0; j < rank; j++) {
      modpack_keys[j] = &interface_->GetModPackKey(rank, j);
    }
    cudaMemGetInfo(&dev_free, &dev_total);
    std::cout << "device " << (dev_free >> 20) << " MiB free after the "
              << rank << " ModPack keys" << std::endl;

    if (mode == Mode::kProjections) {
      real_leg = std::make_unique<EncryptedProjectionLeg>(context_, lcfg,
                                                          modpack_keys, host);
      leg = real_leg.get();
    } else {
      // ---- THE OTHER TWO RINGS ------------------------------------------
      //
      // The switching ring shares the block ring's primes and its SECRET --
      // the ciphertext walks down the ladder unchanged, so a Context of its
      // own would be a different key. What it does not share is `alpha`, and
      // that is the whole reason it exists: the ring-switching key is
      // published at PQ, and PQ has to fit the small ring's budget.
      const int operand_level =
          boot->GetBootParameter().GetEvalModEndLevel() - 2;
      const int prob_level = boot->GetBootParameter().GetStCStartLevel();
      std::cout << "preparing SlotToSinC at level " << prob_level << ", "
                << kSinCPhases << " phases" << std::endl;
      boot->PrepareSinC(num_slots, kSubDegree, prob_level, prob_level,
                        kSinCPhases);
      sw_ring = std::make_unique<ringfixture::Ring<word>>(
          "ringswitch16_35.json", interface_->GetSecretCoeffs());
      small_ring =
          std::make_unique<ringfixture::Ring<word>>("ringdegree12_35.json");

      cheddar::SinCLinearLeg<word>::Config scfg;
      scfg.num_heads = kChannels / kHeadDim;
      scfg.num_kv_heads = kKvChannels / kHeadDim;
      scfg.chain_constant = ChainConstant();
      cheddar::SinCAttention<word>::Config acfg;
      acfg.num_tokens = kTokens;
      acfg.head_dim = kHeadDim;
      acfg.lanes = kSubDegree / 2;
      acfg.gqa_group = scfg.num_heads / scfg.num_kv_heads;
      acfg.sub_degree = kSubDegree;
      acfg.sinc_phases = kSinCPhases;
      acfg.product_level = kProductLevel;
      acfg.swap_level = operand_level;
      acfg.sinc_level = prob_level;
      acfg.prefix_level = boot->GetBootParameter().GetEvalModEndLevel();
      acfg.halfboot_scale = boot->GetStCInputScale();
      acfg.verbose = std::getenv("CHEDDAR_VERBOSE") != nullptr;

      // The keys go in after the leg exists, because what has to be generated
      // is what the leg asks for. See SinCLinearLeg::SetKeys.
      cheddar::SinCAttention<word>::Keys empty;
      sinc_leg = std::make_unique<ProbedSinCLeg>(
          boot, sw_ring->context, small_ring->context, scfg, acfg, empty, lcfg,
          modpack_keys, *this, pack, host);
      leg = sinc_leg.get();
      std::cout << "the SinC leg: swaps at " << operand_level << "/"
                << operand_level - 1 << ", exchange at " << operand_level - 2
                << ", SinC at " << prob_level << ".."
                << prob_level - kSinCPhases + 1 << ", product at "
                << kProductLevel << ", HalfBoot lands at " << acfg.prefix_level
                << ", chain constant " << scfg.chain_constant << std::endl;
    }
  }

  Block block(boot, cfg, cal, *leg);
  host.SetLevels(block.GetOperandLevel(), block.GetProbLevel(),
                 block.GetResultLevel());
  std::string why;
  ASSERT_TRUE(block.Fits(&why)) << why;
  std::cout << block.DescribePlan() << std::endl;
  std::cout << "the seam: operands at " << block.GetOperandLevel()
            << ", P at " << block.GetProbLevel() << ", results at "
            << block.GetResultLevel() << std::endl;

  EvkRequest req;
  block.AddRequiredRotations(req);
  if (sinc_leg != nullptr) sinc_leg->AddRequiredRotations(req);
  {
    size_t before = 0, total = 0, after = 0;
    cudaMemGetInfo(&before, &total);
    interface_->PrepareRotationKey(req);
    cudaMemGetInfo(&after, &total);
    std::cout << "big-ring rotation keys: " << ((before - after) >> 20)
              << " MiB, " << (after >> 20) << " MiB free after" << std::endl;
  }

  if (sinc_leg != nullptr) {
    // The block's own accessors are the contract; the levels above were
    // derived from the BootParameter to break the ordering cycle, and this is
    // where the two are required to agree.
    const auto &acfg_layout = sinc_leg->GetAttention().GetLayout();
    ASSERT_EQ(sinc_leg->GetAttention().GetSinCLevel(), block.GetProbLevel());
    ASSERT_EQ(sinc_leg->GetAttention().GetOutputLevel(),
              block.GetResultLevel());
    const int rank = acfg_layout.rank;
    size_t before = 0, total = 0, after = 0;
    cudaMemGetInfo(&before, &total);
    sw_ring->ui->PrepareRingSwitchKey(small_ring->Degree(),
                                      small_ring->ui->GetSecretCoeffs(),
                                      kProductLevel);
    sw_ring->ui->PrepareInverseRingSwitchKey(small_ring->Degree(),
                                             small_ring->ui->GetSecretCoeffs(),
                                             kProductLevel);
    for (int idx : sinc_leg->SmallRotationIndices()) {
      small_ring->ui->PrepareRotationKey(idx, kProductLevel);
    }
    cudaMemGetInfo(&after, &total);
    std::cout << "ring-switch and small-ring keys: " << ((before - after) >> 20)
              << " MiB, " << (after >> 20) << " MiB free after" << std::endl;

    cheddar::SinCAttention<word>::Keys keys;
    keys.big = &interface_->GetEvkMap();
    keys.small = &small_ring->ui->GetEvkMap();
    keys.ring_switch = &sw_ring->ui->GetRingSwitchKey(rank);
    keys.inverse_ring_switch = &sw_ring->ui->GetInverseRingSwitchKey(rank);
    sinc_leg->SetKeys(keys);
  }

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
          cal.residual * x_fill[static_cast<size_t>(t) * kChannels + c];
    }
    Plaintext<word> ptxt;
    context_->encoder_.EncodeCoeff(ptxt, 0, param_->GetScale(0), coeffs);
    interface_->Encrypt(state[i], ptxt);
  }

  std::vector<Ciphertext<word>> out;
  block.Run(out, state, w, sinks, interface_->GetEvkMap());
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "product-leg magnitudes, against the bootstrap's " << kBootMax
            << ":" << std::endl;
  for (const auto &e : host.log_) {
    std::cout << "   " << e.first << " max " << e.second << std::endl;
  }

  // WHERE THE BLOCK'S BITS GO, TURN BY TURN.
  //
  // Every operator's output is handed to a product and to nothing else, so
  // what the stand-in read at each call *is* that turn's encrypted result.
  // Comparing each against the clear run at the same point attributes the
  // block's error to a turn instead of to the block.
  //
  // The constant each one carries is the calibration's, divided out here, so
  // the bits reported are relative to the true tensor.
  struct Point {
    const char *tag;
    const std::vector<double> *want;
    double carried;
  };
  std::vector<double> gated(r.gate.size());
  for (size_t i = 0; i < gated.size(); i++) {
    gated[i] = r.gate[i] / (1.0 + std::exp(-r.gate[i])) * r.up[i];
  }
  const std::vector<Point> points = {
      {"A  RMSNorm(attn) -> the QKV product", &r.normed, 1.0},
      {"B  RoPE(Q) -> the score product", &r.q, cal.size_q},
      {"B  RoPE(K) -> the score product", &r.k, cal.size_k},
      {"C  SoftMax -> the value product", &r.probs, 1.0},
      {"D  attention values -> the O product", &r.attn, cal.size_attn},
      {"E  RMSNorm(ffn) -> the gate/up product", &r.fnormed, 1.0},
      {"F  SiLU(G) * U -> the down product", &gated, cal.size_up},
  };
  const char *tags[] = {"in:Q", "in:q", "in:k", "in:probs",
                        "in:O", "in:gate", "in:down"};
  // The SinC leg reads the attention output in its own layout, before the
  // block untransposes it, so its probe writes a different tag. The O
  // projection then sees the untransposed one under "in:O" as usual.
  const char *alt[] = {nullptr, nullptr, nullptr, nullptr,
                       "in:attn_out", nullptr, nullptr};
  std::cout << "where the block's bits go, turn by turn:" << std::endl;
  for (size_t j = 0; j < points.size(); j++) {
    const std::vector<double> *got = nullptr;
    for (const auto &e : host.seen_) {
      if (e.first == tags[j]) {
        got = &e.second;
        break;
      }
    }
    if (got == nullptr && alt[j] != nullptr) {
      for (const auto &e : host.seen_) {
        if (e.first == alt[j]) {
          got = &e.second;
          break;
        }
      }
    }
    if (got == nullptr || got->size() != points[j].want->size()) {
      std::cout << "   " << points[j].tag << ": not recorded" << std::endl;
      continue;
    }
    double e = 0.0, m = 0.0;
    for (size_t i = 0; i < got->size(); i++) {
      const double want = (*points[j].want)[i];
      e = std::max(e, std::abs((*got)[i] / points[j].carried - want));
      m = std::max(m, std::abs(want));
    }
    std::cout << "   " << points[j].tag << ": " << -std::log2(e / m)
              << " bits (max |.| " << m << ")" << std::endl;
  }

  // The output carries `residual`, coefficient encoded at level 0.
  //
  // Compared against `truth` -- the layer on the real hidden state of every
  // token, with nothing substituted anywhere -- and over the tokens the layer
  // is being run for. The sink rows are skipped because their hidden state was
  // never in the ciphertext; their K and V were, and every query used them.
  double worst = 0.0, absmax = 0.0;
  for (int i = 0; i < num_ct; i++) {
    Plaintext<word> ptxt;
    interface_->Decrypt(ptxt, out[i]);
    std::vector<double> coeffs;
    context_->encoder_.DecodeCoeff(coeffs, ptxt);
    for (int s = 0; s < num_slots; s++) {
      const int t = s % kTokens;
      if (t < kSinkTokens) continue;
      const int c = i * pack.channels_per_ct() + s / kTokens;
      const double want = truth.out[static_cast<size_t>(t) * kChannels + c];
      const double got = coeffs[pack.coeff(s)] / cal.residual;
      worst = std::max(worst, std::abs(got - want));
      absmax = std::max(absmax, std::abs(want));
    }
  }
  std::cout << "one whole Llama-3-8B layer-2 block, tokens " << kSinkTokens
            << ".." << (kTokens - 1) << " of a " << kTokens
            << "-token prompt, vs the true layer: max abs err " << worst
            << " against |out| <= " << absmax << " ("
            << -std::log2(worst / absmax) << " bits)" << std::endl;
  EXPECT_GT(-std::log2(worst / absmax), 4.0);
}

// The control: every product on the host, so the number is the cost of the
// non-linear half alone. Keeping it is what lets the run below be attributed --
// the difference between the two IS the products' contribution.
//
// ONE THING THIS CONTROL NO LONGER MODELS. The stand-in returns its results
// already in slots at the operator's level, so the score product and the value
// product each skip a bootstrap that the real leg performs inside itself. That
// is not a flaw in the stand-in; it is what a stand-in for a product that owns
// its own bootstrap can be. It means the gap between this run and the full one
// contains two bootstraps per score ciphertext that this run never paid.
TEST_P(LlamaBlockFixture, TheBlockRunsEndToEnd) {
  RunWholeBlock(Mode::kHost);
}

// The block with its five plaintext projections running encrypted: ModDecomp,
// Bae PC-MM, ModPack, rescale, at the block's own ring degree with no ring
// switch. Two of the seven products -- Q K^T and P V -- are still the host
// stand-in, and that is stated rather than hidden.
TEST_P(LlamaBlockFixture, TheBlockRunsWithEncryptedProjections) {
  RunWholeBlock(Mode::kProjections);
}

// ---------------------------------------------------------------------------
// THE LAYER, WITH NOTHING ON THE HOST.
//
// All nine products encrypted: seven through the Bae PC-MM at the block's own
// ring degree, and Q K^T and P V through `SinCAttention` -- field swaps,
// ciphertext-axis exchange, SlotToSinC, the ring switch to degree 4096,
// [KANG] Algorithm 4, HalfBoot and the StC prefix.
//
// It needs three Contexts and therefore its own parameter set: the block ring
// `sylphflow16_35`, the switching ring `ringswitch16_35` sharing its primes
// and its SECRET, and the product ring `ringdegree12_35`. The other two tests
// run on `bootparam_35` and this one skips there, so a filtered run gets what
// it asked for and nothing else.
TEST_P(LlamaBlockFixture, TheLayerRunsFullyEncrypted) {
  if (std::string(GetParam()).find("sylphflow") == std::string::npos) {
    GTEST_SKIP() << "the fully encrypted layer needs sylphflow16_35, whose "
                    "ladder the SinC band was solved for";
  }
  RunWholeBlock(Mode::kFull);
}
