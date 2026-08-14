/**
 * @brief S3 chain probe.
 *
 * Repeated operation chains shaped like the four Llama-3 sublayer paths, run
 * end to end on encrypted data and validated step by step against a host
 * reference. The point is not to implement Llama -- no attention or FFN kernel
 * is built here -- but to measure what each shape costs in levels, scale drift
 * and error, and to find where a bootstrap becomes mandatory.
 *
 * Each chain writes one CSV trace with a row per operation.
 *
 *   ./chain_probe --gtest_filter='Cheddar/ChainBed.*<name>*<param>'
 */

#include "ProbeSupport.h"

using namespace cheddar;
using word = uint32_t;

namespace {

constexpr int kSlots = 128;

// The reduction width of one RMSNorm / SoftMax row inside the slot vector.
constexpr int kFieldWidth = 16;

std::string TraceId(const std::string &name, const std::string &param) {
  std::string clean = param;
  std::replace(clean.begin(), clean.end(), '.', '_');
  return "s3_chain_" + name + "_" + clean;
}

double PredictEvalPolyTargetScale(const Parameter<word> &param, int input_level,
                                  double input_scale, int level_consumption) {
  double scale = input_scale;
  for (int i = 0; i < level_consumption; ++i) {
    scale = scale * scale / param.GetRescalePrimeProd(input_level - i);
  }
  return scale;
}

/**
 * @brief Host model of the rotate-and-accumulate reduction tree: after
 * log2(width) steps every slot holds the sum of the `width` values starting at
 * it (wrapping around the slot vector). This is the primitive an RMSNorm sum
 * and a SoftMax row sum are both built from.
 */
std::vector<Complex> HostWindowSum(const std::vector<Complex> &in, int width) {
  const int n = static_cast<int>(in.size());
  std::vector<Complex> out(n, Complex(0.0, 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < width; ++j) out[i] += in[(i + j) % n];
  }
  return out;
}

}  // namespace

class ChainBed : public s3::ProbeBed {
 public:
  /**
   * @brief Rotate-and-accumulate reduction over `width` slots. Consumes no
   * levels; costs log2(width) key switches.
   */
  void ReduceWindow(Ciphertext<word> &res, const Ciphertext<word> &in,
                    int width) {
    context_->Copy(res, in);
    for (int r = 1; r < width; r *= 2) {
      Ciphertext<word> tmp;
      context_->HRotAdd(tmp, res, res, interface_->GetRotationKey(r), r);
      res = std::move(tmp);
    }
  }

  /** @brief Ensure the rotation keys a reduction of `width` needs exist. */
  void PrepareReductionKeys(int width, int level) {
    EvkRequest req;
    for (int r = 1; r < width; r *= 2) req.AddRequest(r, level);
    interface_->PrepareRotationKey(req);
  }

  /**
   * @brief Affine map into a polynomial's [-1,1] domain: t = a*x + b.
   * Costs one level (the constant product must be rescaled).
   */
  void AffineToUnit(Ciphertext<word> &res, const Ciphertext<word> &in,
                    const s3::DomainMap &domain, int level) {
    const double a = 2.0 / (domain.hi - domain.lo);
    const double b = -2.0 * domain.lo / (domain.hi - domain.lo) - 1.0;
    Constant<word> ca, cb;
    EncodeConstant(ca, a, level);
    Ciphertext<word> scaled, rescaled;
    context_->Mult(scaled, in, ca);
    context_->Rescale(rescaled, scaled);
    EncodeConstant(cb, b, param_->NPToLevel(rescaled.GetNP()));
    context_->Add(res, rescaled, cb);
  }

  /** @brief Build and compile an EvalPoly for a fit at a given input level. */
  std::unique_ptr<EvalPoly<word>> MakePoly(const s3::NonlinearFit &fit,
                                           int input_level) {
    const int levels = Log2Ceil(fit.effective_degree + 1);
    const double input_scale = DetermineScale(input_level);
    const double target_scale = PredictEvalPolyTargetScale(
        *param_, input_level, input_scale, levels);
    auto poly = std::make_unique<EvalPoly<word>>(fit.coeffs, input_level,
                                                 input_scale, target_scale,
                                                 /*chebyshev=*/true);
    poly->Compile(context_);
    return poly;
  }
};

// ===========================================================================
// Chain 1: RMSNorm polynomial path
// ===========================================================================

TEST_P(ChainBed, RMSNormPath) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("rmsnorm", GetParam()));
  tracer.SetChain("rmsnorm");
  s3::DetRandom rng;

  const int entry_level = param_->default_encryption_level_;
  PrepareReductionKeys(kFieldWidth, entry_level);

  std::vector<Complex> x;
  rng.Real(x, kSlots, -1.0, 1.0);

  const s3::LogicalMeta meta{"SLOT", "field=" + std::to_string(kFieldWidth)};
  Ciphertext<word> ct_x;
  EncodeAndEncrypt(ct_x, x, entry_level);
  tracer.Trace("entry", ct_x, x, meta, 0.0, "RMSNorm input");

  // 1. square (1 level)
  std::vector<Complex> ref_sq(kSlots);
  for (int i = 0; i < kSlots; ++i) ref_sq[i] = x[i] * x[i];
  Ciphertext<word> ct_sq;
  tracer.TraceStep("square HMult(x,x)", ct_sq, ref_sq,
                   [&]() {
                     context_->HMult(ct_sq, ct_x, ct_x,
                                     interface_->GetMultiplicationKey(), true);
                   },
                   meta, "level -1");

  // 2. reduction over the field (0 levels, log2(width) key switches)
  const std::vector<Complex> ref_sum = HostWindowSum(ref_sq, kFieldWidth);
  Ciphertext<word> ct_sum;
  tracer.TraceStep("reduce window sum", ct_sum, ref_sum,
                   [&]() { ReduceWindow(ct_sum, ct_sq, kFieldWidth); }, meta,
                   "0 levels, log2(width) key switches");

  // 3. affine map into the rsqrt domain (1 level)
  const s3::NonlinearFit rsqrt = s3::RsqrtFit(0.05, 8.0, 15);
  const int sum_level = param_->NPToLevel(ct_sum.GetNP());
  std::vector<Complex> ref_unit(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    ref_unit[i] = Complex(rsqrt.domain.ToUnit(ref_sum[i].real() / kFieldWidth),
                          0.0);
  }
  // Fold the 1/width mean into the same affine map.
  s3::DomainMap scaled_domain{rsqrt.domain.lo * kFieldWidth,
                              rsqrt.domain.hi * kFieldWidth};
  Ciphertext<word> ct_unit;
  tracer.TraceStep("affine to [-1,1]", ct_unit, ref_unit,
                   [&]() {
                     AffineToUnit(ct_unit, ct_sum, scaled_domain, sum_level);
                   },
                   meta, "folds the 1/width mean; level -1");

  // 4. rsqrt fit (ceil(log2(deg+1)) levels)
  const int poly_level = param_->NPToLevel(ct_unit.GetNP());
  auto poly = MakePoly(rsqrt, poly_level);
  std::vector<Complex> ref_rsqrt(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    ref_rsqrt[i] = Complex(poly->PlainEvaluate(ref_unit[i].real()), 0.0);
  }
  Ciphertext<word> ct_rsqrt;
  tracer.TraceStep(
      "EvalPoly rsqrt deg" + std::to_string(rsqrt.effective_degree), ct_rsqrt,
      ref_rsqrt,
      [&]() {
        poly->Evaluate(context_, ct_rsqrt, ct_unit,
                       interface_->GetMultiplicationKey());
      },
      meta, "level -" + std::to_string(Log2Ceil(rsqrt.effective_degree + 1)));

  // 5. bring x down and normalize (1 level)
  const int norm_level = param_->NPToLevel(ct_rsqrt.GetNP());
  Ciphertext<word> ct_x_down;
  tracer.TraceStep("LevelDown(x) to meet the fit", ct_x_down, x,
                   [&]() { context_->LevelDown(ct_x_down, ct_x, norm_level); },
                   meta, "residual-style level alignment");

  std::vector<Complex> ref_out(kSlots);
  for (int i = 0; i < kSlots; ++i) ref_out[i] = x[i] * ref_rsqrt[i];
  Ciphertext<word> ct_out;
  tracer.TraceStep("normalize HMult", ct_out, ref_out,
                   [&]() {
                     context_->HMult(ct_out, ct_x_down, ct_rsqrt,
                                     interface_->GetMultiplicationKey(), true);
                   },
                   meta, "level -1");

  const int exit_level = param_->NPToLevel(ct_out.GetNP());
  std::cout << "RMSNorm chain: entry_level=" << entry_level
            << " exit_level=" << exit_level
            << " levels_consumed=" << entry_level - exit_level << "\n";
  std::cout << "entry_scale=2^" << std::log2(DetermineScale(entry_level))
            << " exit_scale=2^" << std::log2(ct_out.GetScale()) << "\n";
  std::cout << "trace written to " << tracer.GetPath() << "\n";
  std::cout << "peak device MiB delta: " << tracer.GetPeakDeltaMiB() << "\n";
}

// ===========================================================================
// Chain 2: projection + residual
// ===========================================================================

TEST_P(ChainBed, ProjectionResidualPath) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("projection_residual", GetParam()));
  tracer.SetChain("projection_residual");
  s3::DetRandom rng;

  const int entry_level = param_->default_encryption_level_;
  const double pt_scale = DetermineScale(entry_level);

  // A square diagonal-encoded map stands in for one projection tile.
  const int bs = 8, gs = 4;
  StripedMatrix matrix(kSlots, kSlots);
  for (int d = 0; d < bs * gs; d += 2) {
    std::vector<Complex> diag;
    rng.ComplexMsg(diag, kSlots, -0.3, 0.3);
    matrix[d] = diag;
  }
  LinearTransform<word> lt(context_, matrix, entry_level, pt_scale, bs, gs);
  EvkRequest req;
  lt.AddRequiredRotations(req);
  const double keygen_us = TimeRotationKeys(req);
  std::cout << "projection rotation keys=" << req.size()
            << " keygen_us=" << keygen_us << " (OFFLINE SETUP)\n";

  std::vector<Complex> x;
  rng.Real(x, kSlots, -1.0, 1.0);
  Ciphertext<word> ct_x;
  EncodeAndEncrypt(ct_x, x, entry_level);
  const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kSlots)};
  tracer.Trace("entry (also the residual)", ct_x, x, meta, 0.0, "");

  std::vector<Complex> ref_proj(kSlots, Complex(0.0, 0.0));
  for (const auto &[d, diag] : matrix) {
    for (int i = 0; i < kSlots; ++i) {
      ref_proj[i] += diag[i] * x[(i + d) % kSlots];
    }
  }

  Ciphertext<word> ct_proj;
  tracer.TraceStep("projection LinearTransform", ct_proj, ref_proj,
                   [&]() {
                     lt.Evaluate(context_, ct_proj, ct_x,
                                 interface_->GetEvkMap());
                   },
                   meta, "square tile stand-in for a PCMM");

  // The residual is still at the entry level while the projection output has
  // dropped. Add asserts a matching prime set, so the residual must be brought
  // down first. This is the level-alignment cost every Llama residual pays.
  const int proj_level = param_->NPToLevel(ct_proj.GetNP());
  std::cout << "projection consumed " << entry_level - proj_level
            << " level(s); residual must be level-aligned before Add\n";

  Ciphertext<word> ct_res_down;
  tracer.TraceStep("LevelDown(residual)", ct_res_down, x,
                   [&]() { context_->LevelDown(ct_res_down, ct_x, proj_level); },
                   meta, "aligns residual with projection output");

  std::vector<Complex> ref_sum(kSlots);
  for (int i = 0; i < kSlots; ++i) ref_sum[i] = ref_proj[i] + x[i];
  Ciphertext<word> ct_sum;
  tracer.TraceStep("residual Add", ct_sum, ref_sum,
                   [&]() { context_->Add(ct_sum, ct_proj, ct_res_down); }, meta,
                   "0 levels");

  // Repeat the projection+residual block until the level budget runs out, to
  // measure error growth and find the bootstrap point.
  std::cout << "repeat,level,max_abs_err,scale_log2\n";
  Ciphertext<word> cur;
  context_->Copy(cur, ct_sum);
  std::vector<Complex> ref_cur = ref_sum;
  int repeats = 0;
  while (param_->NPToLevel(cur.GetNP()) >= 2) {
    const int lvl = param_->NPToLevel(cur.GetNP());
    // A fresh transform is required at each level: LinearTransform bakes its
    // plaintext level and scale in at construction.
    LinearTransform<word> lt_i(context_, matrix, lvl, DetermineScale(lvl), bs,
                               gs);
    EvkRequest req_i;
    lt_i.AddRequiredRotations(req_i);
    interface_->PrepareRotationKey(req_i);

    std::vector<Complex> ref_next(kSlots, Complex(0.0, 0.0));
    for (const auto &[d, diag] : matrix) {
      for (int i = 0; i < kSlots; ++i) {
        ref_next[i] += diag[i] * ref_cur[(i + d) % kSlots];
      }
    }
    Ciphertext<word> proj_i, down_i, sum_i;
    lt_i.Evaluate(context_, proj_i, cur, interface_->GetEvkMap());
    const int lvl_i = param_->NPToLevel(proj_i.GetNP());
    context_->LevelDown(down_i, cur, lvl_i);
    context_->Add(sum_i, proj_i, down_i);
    for (int i = 0; i < kSlots; ++i) ref_next[i] += ref_cur[i];

    ++repeats;
    const s3::ErrorStats stats = tracer.Trace(
        "repeat " + std::to_string(repeats) + " proj+residual", sum_i, ref_next,
        meta, 0.0, "level " + std::to_string(lvl_i));
    std::cout << repeats << ',' << lvl_i << ',' << stats.max_abs << ','
              << std::log2(sum_i.GetScale()) << '\n';

    cur = std::move(sum_i);
    ref_cur = std::move(ref_next);
  }
  std::cout << "projection+residual repeats before the level budget from "
            << entry_level << " is exhausted: " << repeats << "\n";
  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// Chain 3: attention-score polynomial path (exp -> mask -> sum -> 1/x)
// ===========================================================================

TEST_P(ChainBed, AttentionScorePath) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("attention_score", GetParam()));
  tracer.SetChain("attention_score");
  s3::DetRandom rng;

  const int entry_level = param_->default_encryption_level_;
  PrepareReductionKeys(kFieldWidth, entry_level);

  // Scores are assumed already shifted so the row max is 0, which is what a
  // numerically stable SoftMax does on the host side.
  std::vector<Complex> scores;
  rng.Real(scores, kSlots, -6.0, 0.0);

  const s3::NonlinearFit exp_fit = s3::ExpFit(-8.0, 0.0, 15);
  const s3::LogicalMeta meta{"SLOT", "row=" + std::to_string(kFieldWidth)};

  // Feed the polynomial its unit-domain argument directly; the affine map is
  // charged separately in the RMSNorm chain and would double-count here.
  std::vector<Complex> unit(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    unit[i] = Complex(exp_fit.domain.ToUnit(scores[i].real()), 0.0);
  }
  Ciphertext<word> ct_unit;
  EncodeAndEncrypt(ct_unit, unit, entry_level);
  tracer.Trace("entry (scores in [-1,1])", ct_unit, unit, meta, 0.0, "");

  // 1. exp fit
  auto exp_poly = MakePoly(exp_fit, entry_level);
  std::vector<Complex> ref_exp(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    ref_exp[i] = Complex(exp_poly->PlainEvaluate(unit[i].real()), 0.0);
  }
  Ciphertext<word> ct_exp;
  tracer.TraceStep(
      "EvalPoly exp deg" + std::to_string(exp_fit.effective_degree), ct_exp,
      ref_exp,
      [&]() {
        exp_poly->Evaluate(context_, ct_exp, ct_unit,
                           interface_->GetMultiplicationKey());
      },
      meta);

  // 2. causal mask as a plaintext product (1 level)
  const int mask_level = param_->NPToLevel(ct_exp.GetNP());
  std::vector<Complex> mask(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    mask[i] = ((i % kFieldWidth) <= (i / kFieldWidth) % kFieldWidth)
                  ? Complex(1.0, 0.0)
                  : Complex(0.0, 0.0);
  }
  Plaintext<word> pt_mask;
  context_->encoder_.Encode(pt_mask, mask_level, ct_exp.GetScale(), mask);
  std::vector<Complex> ref_masked(kSlots);
  for (int i = 0; i < kSlots; ++i) ref_masked[i] = ref_exp[i] * mask[i];

  Ciphertext<word> ct_masked_raw, ct_masked;
  tracer.TraceStep("causal mask Mult(ct,pt)", ct_masked_raw, ref_masked,
                   [&]() { context_->Mult(ct_masked_raw, ct_exp, pt_mask); },
                   meta, "no rescale yet");
  tracer.TraceStep("mask Rescale", ct_masked, ref_masked,
                   [&]() { context_->Rescale(ct_masked, ct_masked_raw); }, meta,
                   "level -1");

  // 3. row sum (0 levels)
  const std::vector<Complex> ref_rowsum =
      HostWindowSum(ref_masked, kFieldWidth);
  Ciphertext<word> ct_rowsum;
  tracer.TraceStep("row sum (rotate-add)", ct_rowsum, ref_rowsum,
                   [&]() { ReduceWindow(ct_rowsum, ct_masked, kFieldWidth); },
                   meta, "0 levels");

  std::cout << "SoftMax denominator observed range: ";
  double dmin = 1e30, dmax = -1e30;
  for (const auto &v : ref_rowsum) {
    dmin = std::min(dmin, v.real());
    dmax = std::max(dmax, v.real());
  }
  std::cout << "[" << dmin << ", " << dmax << "]\n";

  // 4. reciprocal fit over the measured denominator range
  const s3::NonlinearFit recip_fit =
      s3::ReciprocalFit(std::max(0.25, dmin * 0.9), dmax * 1.1, 63);
  const int recip_in_level = param_->NPToLevel(ct_rowsum.GetNP());
  std::cout << "reciprocal fit: range=[" << recip_fit.domain.lo << ","
            << recip_fit.domain.hi
            << "] effective_degree=" << recip_fit.effective_degree
            << " levels=" << Log2Ceil(recip_fit.effective_degree + 1) << "\n";

  Ciphertext<word> ct_recip_unit;
  std::vector<Complex> ref_recip_unit(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    ref_recip_unit[i] =
        Complex(recip_fit.domain.ToUnit(ref_rowsum[i].real()), 0.0);
  }
  tracer.TraceStep("affine to [-1,1] for 1/x", ct_recip_unit, ref_recip_unit,
                   [&]() {
                     AffineToUnit(ct_recip_unit, ct_rowsum, recip_fit.domain,
                                  recip_in_level);
                   },
                   meta, "level -1");

  const int recip_level = param_->NPToLevel(ct_recip_unit.GetNP());
  if (recip_level <= Log2Ceil(recip_fit.effective_degree + 1)) {
    std::cout << "BLOCKER: only " << recip_level
              << " levels remain but the degree-"
              << recip_fit.effective_degree << " reciprocal needs "
              << Log2Ceil(recip_fit.effective_degree + 1)
              << ". A bootstrap is mandatory inside the SoftMax at this "
                 "entry level.\n";
  } else {
    auto recip_poly = MakePoly(recip_fit, recip_level);
    std::vector<Complex> ref_recip(kSlots);
    for (int i = 0; i < kSlots; ++i) {
      ref_recip[i] =
          Complex(recip_poly->PlainEvaluate(ref_recip_unit[i].real()), 0.0);
    }
    Ciphertext<word> ct_recip;
    tracer.TraceStep(
        "EvalPoly 1/x deg" + std::to_string(recip_fit.effective_degree),
        ct_recip, ref_recip,
        [&]() {
          recip_poly->Evaluate(context_, ct_recip, ct_recip_unit,
                               interface_->GetMultiplicationKey());
        },
        meta);

    // 5. normalize
    const int norm_level = param_->NPToLevel(ct_recip.GetNP());
    Ciphertext<word> ct_num_down, ct_soft;
    context_->LevelDown(ct_num_down, ct_masked, norm_level);
    std::vector<Complex> ref_soft(kSlots);
    for (int i = 0; i < kSlots; ++i) {
      ref_soft[i] = ref_masked[i] * ref_recip[i];
    }
    tracer.TraceStep("normalize HMult", ct_soft, ref_soft,
                     [&]() {
                       context_->HMult(ct_soft, ct_num_down, ct_recip,
                                       interface_->GetMultiplicationKey(),
                                       true);
                     },
                     meta, "level -1");
    std::cout << "attention-score chain: entry_level=" << entry_level
              << " exit_level=" << param_->NPToLevel(ct_soft.GetNP())
              << " levels_consumed="
              << entry_level - param_->NPToLevel(ct_soft.GetNP()) << "\n";
  }
  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// Chain 4: SiLU / SwiGLU
// ===========================================================================

TEST_P(ChainBed, SiluSwigluPath) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("silu_swiglu", GetParam()));
  tracer.SetChain("silu_swiglu");
  s3::DetRandom rng;

  const int entry_level = param_->default_encryption_level_;
  const s3::NonlinearFit silu = s3::SiluFit(-8.0, 8.0, 15);
  const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kSlots)};

  std::vector<Complex> gate_unit, up;
  rng.Real(gate_unit, kSlots, -0.95, 0.95);
  rng.Real(up, kSlots, -1.0, 1.0);

  Ciphertext<word> ct_gate, ct_up;
  EncodeAndEncrypt(ct_gate, gate_unit, entry_level);
  EncodeAndEncrypt(ct_up, up, entry_level);
  tracer.Trace("gate entry", ct_gate, gate_unit, meta, 0.0, "");
  tracer.Trace("up entry", ct_up, up, meta, 0.0, "");

  // 1. SiLU fit
  auto poly = MakePoly(silu, entry_level);
  std::vector<Complex> ref_silu(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    ref_silu[i] = Complex(poly->PlainEvaluate(gate_unit[i].real()), 0.0);
  }
  Ciphertext<word> ct_silu;
  tracer.TraceStep("EvalPoly silu deg" + std::to_string(silu.effective_degree),
                   ct_silu, ref_silu,
                   [&]() {
                     poly->Evaluate(context_, ct_silu, ct_gate,
                                    interface_->GetMultiplicationKey());
                   },
                   meta);

  // 2. gate product against the up projection (1 level)
  const int gate_level = param_->NPToLevel(ct_silu.GetNP());
  Ciphertext<word> ct_up_down;
  tracer.TraceStep("LevelDown(up) to meet the gate", ct_up_down, up,
                   [&]() { context_->LevelDown(ct_up_down, ct_up, gate_level); },
                   meta);

  std::vector<Complex> ref_out(kSlots);
  for (int i = 0; i < kSlots; ++i) ref_out[i] = ref_silu[i] * up[i];
  Ciphertext<word> ct_out;
  tracer.TraceStep("SwiGLU gate HMult", ct_out, ref_out,
                   [&]() {
                     context_->HMult(ct_out, ct_silu, ct_up_down,
                                     interface_->GetMultiplicationKey(), true);
                   },
                   meta, "level -1");

  const int exit_level = param_->NPToLevel(ct_out.GetNP());
  std::cout << "SwiGLU chain: entry_level=" << entry_level
            << " exit_level=" << exit_level
            << " levels_consumed=" << entry_level - exit_level << "\n";

  // Repeat the SiLU+gate shape to find how many fit in one level budget.
  int repeats = 0;
  Ciphertext<word> cur;
  context_->Copy(cur, ct_out);
  std::vector<Complex> ref_cur = ref_out;
  const int depth = entry_level - exit_level;
  while (param_->NPToLevel(cur.GetNP()) > depth) {
    const int lvl = param_->NPToLevel(cur.GetNP());
    auto poly_i = MakePoly(silu, lvl);
    Ciphertext<word> silu_i, down_i, out_i;
    poly_i->Evaluate(context_, silu_i, cur, interface_->GetMultiplicationKey());
    const int lvl_i = param_->NPToLevel(silu_i.GetNP());
    context_->LevelDown(down_i, cur, lvl_i);
    context_->HMult(out_i, silu_i, down_i,
                    interface_->GetMultiplicationKey(), true);

    std::vector<Complex> ref_next(kSlots);
    for (int i = 0; i < kSlots; ++i) {
      ref_next[i] = Complex(poly_i->PlainEvaluate(ref_cur[i].real()), 0.0) *
                    ref_cur[i];
    }
    ++repeats;
    const s3::ErrorStats stats =
        tracer.Trace("repeat " + std::to_string(repeats) + " silu+gate", out_i,
                     ref_next, meta, 0.0,
                     "level " + std::to_string(param_->NPToLevel(out_i.GetNP())));
    std::cout << "repeat " << repeats
              << " level=" << param_->NPToLevel(out_i.GetNP())
              << " max_abs_err=" << stats.max_abs << "\n";
    cur = std::move(out_i);
    ref_cur = std::move(ref_next);
  }
  std::cout << "SwiGLU-shaped stages that fit in one budget from level "
            << entry_level << ": " << repeats + 1 << "\n";
  std::cout << "trace written to " << tracer.GetPath() << "\n";
  std::cout << "peak device MiB delta: " << tracer.GetPeakDeltaMiB() << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, ChainBed,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json"),
    [](const testing::TestParamInfo<ChainBed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
