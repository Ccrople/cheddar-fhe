// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// llama3_oracle_cli -- the deterministic plaintext oracle, as an executable.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "oracle/Calibrate.h"
#include "oracle/Cheby.h"
#include "oracle/Compare.h"
#include "oracle/Config.h"
#include "oracle/Json.h"
#include "oracle/Ops.h"
#include "oracle/Serialize.h"
#include "oracle/Vectors.h"
#include "oracle/Weights.h"

namespace {

using namespace oracle;

struct Args {
  std::string command;
  std::map<std::string, std::string> opts;

  bool Has(const std::string& k) const { return opts.count(k) > 0; }
  std::string Str(const std::string& k, const std::string& d = "") const {
    auto it = opts.find(k);
    return it == opts.end() ? d : it->second;
  }
  int64_t Int(const std::string& k, int64_t d) const {
    auto it = opts.find(k);
    return it == opts.end() ? d : std::stoll(it->second);
  }
  double Dbl(const std::string& k, double d) const {
    auto it = opts.find(k);
    return it == opts.end() ? d : std::stod(it->second);
  }
  bool Flag(const std::string& k) const {
    auto it = opts.find(k);
    if (it == opts.end()) return false;
    return it->second != "0" && it->second != "false";
  }
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  if (argc > 1) a.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) != 0) continue;
    s = s.substr(2);
    const size_t eq = s.find('=');
    if (eq == std::string::npos) {
      a.opts[s] = "1";
    } else {
      a.opts[s.substr(0, eq)] = s.substr(eq + 1);
    }
  }
  return a;
}

void PrintUsage() {
  std::cout <<
R"(llama3_oracle_cli -- deterministic plaintext oracle for Llama-3-8B on Cheddar

USAGE
  llama3_oracle_cli <command> [--option=value ...]

COMMANDS
  config          print the verified configuration and its provenance
  conventions     print the numerical conventions and the serialization format
  boundaries      list every exported module boundary and its metadata
  thresholds      print the acceptance thresholds and their rationale
  vectors         print the test-vector catalogue
  weights-howto   how to connect locally available Llama-3 weights
  dump            run the model and write every boundary tensor
  calibrate       run the model and write the range/calibration report
  fit             degree search over calibrated (or supplied) intervals
  selftest        run every vector, check the invariants, report
  compare         compare two tensor files or two dump directories

SHAPE OPTIONS (all commands that run the model)
  --tokens=N          sequence length            (default 128)
  --layers=N          decoder blocks to run      (default 1)
  --seed=N            synthetic RNG seed         (default 20260814)
  --reduced           use a small shape for a fast run:
                      heads 4, kv_heads 2, head_dim 8, intermediate 32
  --heads / --kv-heads / --head-dim / --intermediate
                      override individual dimensions
  --rope=half_split|interleaved     RoPE channel pairing (default half_split)
  --no-rope           disable RoPE entirely (a circuit simplification)
  --softmax=row_max|fixed_shift|naive
  --shift=X           the constant for --softmax=fixed_shift
  --weights-dir=DIR   load locally supplied float32 weights (see weights-howto)
  --dtype=f64|f32     payload type for dump (default f64)

DUMP OPTIONS
  --out=DIR           output directory (required)
  --set=minimal|boundaries|full     which tensors to write (default boundaries)

CALIBRATE OPTIONS
  --out=DIR           write calibration.txt / calibration.json / fits.txt
  --margin=X          fractional widening of the observed interval (default 0.05)
  --target=X          target max relative approximation error (default 1e-4)

COMPARE OPTIONS
  --test=PATH --ref=PATH            two .tensor files
  --test-dir=DIR --ref-dir=DIR      two dump directories, matched by name
  --tier=primitive|module|block|multi_block   or --module=rmsnorm|softmax|...
  --axis=0|1                        also report per-token (0) or per-channel (1)

EXIT STATUS
  0 success / all thresholds met. 1 on a threshold failure or an error.
)";
}

Llama3Config ConfigFromArgs(const Args& a) {
  Llama3Config c = Llama3_8B();
  if (a.Flag("reduced")) c = Llama3Reduced(4, 2, 8, 32);
  if (a.Has("heads") || a.Has("kv-heads") || a.Has("head-dim") ||
      a.Has("intermediate")) {
    c = Llama3Reduced(a.Int("heads", c.num_heads), a.Int("kv-heads", c.num_kv_heads),
                      a.Int("head-dim", c.head_dim),
                      a.Int("intermediate", c.intermediate_size));
  }
  const std::string rope = a.Str("rope", "half_split");
  if (rope == "interleaved") c.rope_convention = RopeConvention::kInterleaved;
  else if (rope != "half_split") throw std::invalid_argument("--rope must be half_split or interleaved");
  if (a.Flag("no-rope")) c.rope_enabled = false;
  const std::string sm = a.Str("softmax", "row_max");
  if (sm == "fixed_shift") c.softmax_mode = SoftmaxMode::kFixedShift;
  else if (sm == "naive") c.softmax_mode = SoftmaxMode::kNaive;
  else if (sm != "row_max") throw std::invalid_argument("--softmax must be row_max, fixed_shift or naive");
  c.softmax_fixed_shift = a.Dbl("shift", 0.0);
  c.num_layers = a.Int("layers", 1);
  c.Validate();
  return c;
}

struct RunInputs {
  Llama3Config config;
  SyntheticSpec spec;
  std::vector<LayerWeights> layers;
  Tensor x;
  bool real_weights = false;
  std::string weights_dir;
};

RunInputs PrepareRun(const Args& a) {
  RunInputs r;
  r.config = ConfigFromArgs(a);
  r.spec.seed = static_cast<uint64_t>(a.Int("seed", 20260814));
  const int64_t tokens = a.Int("tokens", 128);
  const int64_t layers = a.Int("layers", 1);

  r.weights_dir = a.Str("weights-dir");
  if (!r.weights_dir.empty()) {
    WeightPaths p;
    p.dir = r.weights_dir;
    LayerWeights w;
    std::string err;
    if (!LoadLayerWeights(p, r.config, &w, &err))
      throw std::runtime_error("weight load failed: " + err);
    r.real_weights = true;
    // One real block's weights, reused at every layer. Reusing them is the
    // honest option: the bundle format holds ONE layer, and pretending
    // otherwise would silently make a multi-layer calibration fictional.
    for (int64_t i = 0; i < layers; ++i) r.layers.push_back(w);
    Tensor xin;
    if (LoadInputActivations(p, r.config, tokens, &xin, &err)) {
      r.x = xin;
    } else {
      std::cerr << "note: " << err
                << "\n      falling back to synthetic activations.\n";
      r.x = MakeSyntheticActivations(r.config, r.spec, tokens);
    }
  } else {
    r.layers = MakeSyntheticLayers(r.config, r.spec, layers);
    r.x = MakeSyntheticActivations(r.config, r.spec, tokens);
  }
  return r;
}

Json RunMetadata(const RunInputs& r, const Args& a) {
  Json j = Json::Object();
  j.Set("oracle", "cheddar llama3 plaintext oracle");
  j.Set("config_identity", r.config.Identity());
  Json cfg = Json::Object();
  cfg.Set("hidden_size", r.config.hidden_size);
  cfg.Set("intermediate_size", r.config.intermediate_size);
  cfg.Set("num_layers", static_cast<int64_t>(r.layers.size()));
  cfg.Set("num_heads", r.config.num_heads);
  cfg.Set("num_kv_heads", r.config.num_kv_heads);
  cfg.Set("head_dim", r.config.head_dim);
  cfg.Set("rms_norm_eps", r.config.rms_norm_eps);
  cfg.Set("rope_theta", r.config.rope_theta);
  cfg.Set("rope_convention", std::string(ToString(r.config.rope_convention)));
  cfg.Set("rope_enabled", r.config.rope_enabled);
  cfg.Set("softmax_mode", std::string(ToString(r.config.softmax_mode)));
  cfg.Set("softmax_fixed_shift", r.config.softmax_fixed_shift);
  cfg.Set("causal", r.config.causal);
  cfg.Set("tokens", r.x.Dim(0));
  j.Set("config", cfg);
  j.Set("weight_source", r.real_weights ? "loaded" : "synthetic");
  j.Set("weight_dir", r.weights_dir);
  j.Set("seed", static_cast<int64_t>(r.spec.seed));
  j.Set("weight_scaling",
        std::string(r.spec.scaling == WeightScaling::kUnitVariance
                        ? "1/sqrt(fan_in)"
                        : "unscaled"));
  j.Set("numerical_conventions", NumericalConventions());
  j.Set("weight_source_note", WeightSourceNote(r.real_weights, r.weights_dir));
  (void)a;
  return j;
}

bool WriteTextFile(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::trunc);
  if (!f) return false;
  f.write(text.data(), static_cast<std::streamsize>(text.size()));
  return static_cast<bool>(f);
}

// ---------------------------------------------------------------------------

int CmdDump(const Args& a) {
  const std::string out = a.Str("out");
  if (out.empty()) {
    std::cerr << "dump: --out=DIR is required\n";
    return 1;
  }
  std::string err;
  if (!EnsureDirectory(out, &err)) {
    std::cerr << "dump: " << err << "\n";
    return 1;
  }
  RunInputs r = PrepareRun(a);
  const TensorDType dtype =
      a.Str("dtype", "f64") == "f32" ? TensorDType::kFloat32 : TensorDType::kFloat64;
  FileSink sink(out, dtype);
  sink.SetNameFilter(DumpSet(a.Str("set", "boundaries")));

  Tensor y = Forward(r.x, r.layers, r.config, a.Int("offset", 0), &sink);
  (void)y;

  Json meta = RunMetadata(r, a);
  meta.Set("dump_set", a.Str("set", "boundaries"));
  meta.Set("serialization_format", SerializationFormatDoc());
  if (!sink.WriteManifest(meta, &err)) {
    std::cerr << "dump: " << err << "\n";
    return 1;
  }
  std::cout << WeightSourceNote(r.real_weights, r.weights_dir) << "\n";
  std::cout << "wrote " << sink.Records().size() << " tensors to " << out
            << "/ (manifest.json lists them)\n";
  for (const std::string& e : sink.Errors()) std::cerr << "  error: " << e << "\n";
  return sink.Errors().empty() ? 0 : 1;
}

int CmdCalibrate(const Args& a) {
  RunInputs r = PrepareRun(a);

  // Pass 1: row-max SoftMax. This measures the score distribution, which is
  // what a fixed shift has to be chosen from.
  Calibrator cal1;
  Forward(r.x, r.layers, r.config, a.Int("offset", 0), &cal1);
  Calibrator m1 = cal1.MergedAcrossLayers();

  double score_max = 0.0, score_min = 0.0;
  bool have_scores = m1.Has("attn.scores") && m1.Stat("attn.scores").Count() > 0;
  if (have_scores) {
    score_max = m1.Stat("attn.scores").Max();
    score_min = m1.Stat("attn.scores").Min();
  }

  // Pass 2: fixed-shift SoftMax at the measured global maximum. This is the
  // circuit an encrypted SoftMax actually runs, so its ranges are the ones the
  // approximation must be built against.
  Llama3Config c2 = r.config;
  c2.softmax_mode = SoftmaxMode::kFixedShift;
  c2.softmax_fixed_shift = score_max;
  Calibrator cal2;
  Forward(r.x, r.layers, c2, a.Int("offset", 0), &cal2);
  Calibrator m2 = cal2.MergedAcrossLayers();

  const double margin = a.Dbl("margin", 0.05);
  std::vector<FitInterval> iv1 = DeriveFitIntervals(m1, margin);
  std::vector<FitInterval> iv2 = DeriveFitIntervals(m2, margin);

  std::string text;
  {
    std::ostringstream os;
    os << WeightSourceNote(r.real_weights, r.weights_dir) << "\n";
    os << "config: " << r.config.Identity() << "\n";
    os << "tokens: " << r.x.Dim(0) << "   layers: " << r.layers.size() << "\n\n";

    os << "PASS 1 -- row-max SoftMax (the mathematical reference)\n";
    os << "======================================================\n";
    os << CalibrationReport(m1, iv1, "SoftMax shift: per-row maximum");

    os << "\n\n";
    os << "PASS 2 -- fixed-shift SoftMax (what an encrypted circuit runs)\n";
    os << "=============================================================\n";
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "Measured score range over pass 1: [%.6g, %.6g].\n"
                  "The fixed shift is set to the measured MAXIMUM, %.6g, which "
                  "is the only\nchoice that keeps every exp argument at or "
                  "below zero and therefore keeps\nexp bounded by 1. A shift "
                  "chosen any lower lets exp grow; chosen higher, the\n"
                  "denominator underflows towards the bottom of the reciprocal "
                  "interval.\n"
                  "This shift is a property of the MEASURED score "
                  "distribution and must be\nre-measured whenever the weights, "
                  "the sequence length or the scale change.\n\n",
                  score_min, score_max, score_max);
    os << buf;
    os << CalibrationReport(m2, iv2, "SoftMax shift: fixed at the measured max");

    // The comparison is the point of running both passes. It is computed here
    // rather than left to the reader because the two numbers are the whole
    // trade-off in the encrypted SoftMax and they are easy to miss in two
    // separate tables.
    if (m1.Has("attn.denominator") && m2.Has("attn.denominator") &&
        m1.Has("attn.exp_input") && m2.Has("attn.exp_input")) {
      const RangeStat& d1 = m1.Stat("attn.denominator");
      const RangeStat& d2 = m2.Stat("attn.denominator");
      const RangeStat& e1 = m1.Stat("attn.exp_input");
      const RangeStat& e2 = m2.Stat("attn.exp_input");
      os << "\n\nSHIFT STRATEGY -- the measured cost of not being able to take "
            "a maximum\n"
            "=================================================================="
            "========\n";
      std::snprintf(buf, sizeof(buf),
                    "                          exp argument            "
                    "denominator (1/x argument)\n"
                    "  per-row maximum   [%10.4g, %8.4g]   [%10.4g, %8.4g]  "
                    "ratio %8.4g\n"
                    "  one fixed shift   [%10.4g, %8.4g]   [%10.4g, %8.4g]  "
                    "ratio %8.4g\n",
                    e1.Min(), e1.Max(), d1.Min(), d1.Max(),
                    d1.Min() > 0.0 ? d1.Max() / d1.Min() : 0.0, e2.Min(),
                    e2.Max(), d2.Min(), d2.Max(),
                    d2.Min() > 0.0 ? d2.Max() / d2.Min() : 0.0);
      os << buf;
      os <<
R"(
Read it this way. A per-row maximum normalises every row, so its denominator
sits in a narrow band and 1/x is easy to fit -- but a maximum is not a
low-degree polynomial and CKKS cannot compute one. A single global shift IS
computable, and it pays for that by widening the denominator range by the full
spread of the per-row maxima: a row whose own scores are far below the global
maximum contributes exponentially small terms.

The ratio in the right-hand column is what the 1/x fit has to span, and the
degree table above shows what that costs. If the fixed-shift ratio is large
enough that no candidate degree fits it, the answer is NOT a higher degree.
The options, in the order they should be considered:

  1. Shift per row-block rather than globally, if the packing lets a block of
     rows share one plaintext constant. This narrows the ratio by exactly the
     spread of the per-block maxima and costs nothing but the constants.
  2. Refresh the denominator on a narrow auxiliary track so the reciprocal is
     evaluated at a fixed depth over a fixed range (the "refresh_denominator"
     idea in the HEonGPU record, which reports the fit moving off the wide
     track entirely).
  3. Split the 1/x domain and select with a low-degree comparison -- expensive,
     and the last resort.

This measurement is the input to that decision; it is not the decision. S0 owns
the choice, and it cannot be made without knowing the packing.
)";
    }
    text = os.str();
  }

  // Degree recommendations over BOTH passes' intervals. Pass 2 is what an
  // encrypted circuit with one global shift actually faces; pass 1 is the
  // unreachable ideal, and the gap between the two degree tables is what a
  // better shift strategy is worth. Reporting only one of them would hide
  // exactly the decision S5 has to make.
  const double target = a.Dbl("target", 1e-4);
  const std::string weight_note =
      r.real_weights ? " on loaded weights" : " on SYNTHETIC weights";
  auto recommend = [&](const std::vector<FitInterval>& iv) {
    std::vector<DegreeRecommendation> out;
    for (const FitInterval& fi : iv)
      out.push_back(RecommendDegree(fi.function, FunctionByName(fi.function),
                                    fi.lo, fi.hi, target, fi.measured,
                                    fi.source + weight_note));
    return out;
  };
  const std::vector<DegreeRecommendation> recs = recommend(iv2);
  const std::vector<DegreeRecommendation> recs_rowmax = recommend(iv1);

  std::string fits;
  {
    std::ostringstream os;
    os << "PART A -- over the FIXED-SHIFT intervals (the circuit that is\n"
          "actually implementable under CKKS). These are the degrees to plan\n"
          "against unless the shift strategy changes.\n\n";
    os << DegreeReport(recs);
    os << "\n\nPART B -- over the PER-ROW-MAXIMUM intervals. A row maximum is\n"
          "NOT computable under CKKS, so this table is not a plan; it is the\n"
          "upper bound on what any better shift strategy could buy. Where a\n"
          "row here needs a lower degree than the same row in Part A, that\n"
          "difference is the prize for narrowing the shift -- per row-block,\n"
          "or via a refreshed auxiliary denominator track.\n\n";
    os << DegreeReport(recs_rowmax);
    fits = os.str();
  }

  std::cout << text << "\n" << fits;

  const std::string out = a.Str("out");
  if (!out.empty()) {
    std::string err;
    if (!EnsureDirectory(out, &err)) {
      std::cerr << "calibrate: " << err << "\n";
      return 1;
    }
    Json j = RunMetadata(r, a);
    j.Set("softmax_shift_measured", score_max);
    j.Set("score_min", score_min);
    j.Set("score_max", score_max);
    j.Set("ranges_rowmax", m1.ToJson());
    j.Set("ranges_fixed_shift", m2.ToJson());
    Json ivj = Json::Array();
    for (const FitInterval& fi : iv2) {
      Json o = Json::Object();
      o.Set("function", fi.function);
      o.Set("source", fi.source);
      o.Set("lo", fi.lo);
      o.Set("hi", fi.hi);
      o.Set("observed_lo", fi.observed_lo);
      o.Set("observed_hi", fi.observed_hi);
      o.Set("margin", fi.margin);
      o.Set("dynamic_range", fi.dynamic_range);
      o.Set("samples", fi.samples);
      o.Set("measured", fi.measured);
      o.Set("note", fi.note);
      ivj.Push(o);
    }
    j.Set("fit_intervals", ivj);
    Json rj = Json::Array();
    for (const DegreeRecommendation& d : recs) rj.Push(d.ToJson());
    j.Set("degree_recommendations", rj);
    Json rj1 = Json::Array();
    for (const DegreeRecommendation& d : recs_rowmax) rj1.Push(d.ToJson());
    j.Set("degree_recommendations_rowmax_upper_bound", rj1);
    j.Set("caveats", CalibrationCaveats());

    bool ok = WriteTextFile(out + "/calibration.txt", text);
    ok = WriteTextFile(out + "/fits.txt", fits) && ok;
    ok = WriteTextFile(out + "/calibration.json", j.Dump(2)) && ok;
    if (!ok) {
      std::cerr << "calibrate: failed to write one or more report files\n";
      return 1;
    }
    std::cout << "\nwrote calibration.txt, fits.txt and calibration.json to "
              << out << "/\n";
  }
  return 0;
}

int CmdFit(const Args& a) {
  const std::string fn = a.Str("function");
  if (fn.empty() || !a.Has("lo") || !a.Has("hi")) {
    std::cerr << "fit: --function=rsqrt|exp|reciprocal|silu --lo=X --hi=Y "
                 "[--target=1e-4]\n"
                 "     For measured intervals use `calibrate` instead; this "
                 "command exists\n"
                 "     for exploring a HAND-CHOSEN interval, and its report "
                 "says so.\n";
    return 1;
  }
  const double lo = a.Dbl("lo", 0.0), hi = a.Dbl("hi", 1.0);
  const double target = a.Dbl("target", 1e-4);
  DegreeRecommendation r =
      RecommendDegree(fn, FunctionByName(fn), lo, hi, target,
                      /*interval_measured=*/false, "supplied on the command line");
  std::cout << DegreeReport({r});
  return r.met_target ? 0 : 1;
}

int CmdSelfTest(const Args& a) {
  Llama3Config base = ConfigFromArgs(a);
  if (!a.Flag("reduced") && !a.Has("heads")) base = Llama3Reduced(4, 2, 8, 32);
  SyntheticSpec spec;
  spec.seed = static_cast<uint64_t>(a.Int("seed", 20260814));
  const int64_t tokens = a.Int("tokens", 8);

  int failures = 0;
  std::cout << "vector selftest -- config " << base.Identity() << "\n\n";
  for (const std::string& name : TestVectorNames()) {
    TestVector v = BuildTestVector(name, base, spec, tokens);
    RecordingSink rec;
    Tensor out = DecoderBlock(v.activations, v.weights, v.config,
                              v.position_offset, &rec, "L0.");
    std::vector<InvariantResult> inv = CheckInvariants(rec, v.config, "L0.");
    int bad = 0;
    for (const InvariantResult& i : inv)
      if (!i.ok) ++bad;
    Calibrator cal;
    DecoderBlock(v.activations, v.weights, v.config, v.position_offset, &cal, "L0.");
    Calibrator m = cal.MergedAcrossLayers();

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-26s tokens %4lld  invariants %d/%d%s\n",
                  name.c_str(), static_cast<long long>(v.Tokens()),
                  static_cast<int>(inv.size()) - bad, static_cast<int>(inv.size()),
                  v.is_control ? "   [CONTROL]" : "");
    std::cout << buf;
    for (const InvariantResult& i : inv)
      if (!i.ok)
        std::cout << "    FAIL " << i.name << ": measured " << i.measured
                  << " bound " << i.bound << "  " << i.detail << "\n";
    if (m.Has("attn.exp_input") && m.Stat("attn.exp_input").Count() > 0) {
      const RangeStat& s = m.Stat("attn.exp_input");
      std::snprintf(buf, sizeof(buf),
                    "    exp arg [%.4g, %.4g]   denom [%.4g, %.4g]   "
                    "mean_sq [%.4g, %.4g]\n",
                    s.Min(), s.Max(),
                    m.Has("attn.denominator") ? m.Stat("attn.denominator").Min() : 0.0,
                    m.Has("attn.denominator") ? m.Stat("attn.denominator").Max() : 0.0,
                    m.Has("attn_norm.mean_sq") ? m.Stat("attn_norm.mean_sq").Min() : 0.0,
                    m.Has("attn_norm.mean_sq") ? m.Stat("attn_norm.mean_sq").Max() : 0.0);
      std::cout << buf;
    }
    failures += bad;
    (void)out;
  }
  std::cout << "\n" << (failures == 0 ? "all invariants hold" : "INVARIANT FAILURES")
            << "\n";
  return failures == 0 ? 0 : 1;
}

int CmdCompare(const Args& a) {
  Threshold t = ThresholdModule();
  if (a.Has("module")) t = ThresholdForModule(a.Str("module"));
  else {
    const std::string tier = a.Str("tier", "module");
    if (tier == "primitive") t = ThresholdPrimitive();
    else if (tier == "block") t = ThresholdBlock();
    else if (tier == "multi_block") t = ThresholdMultiBlock(a.Int("blocks", 4));
    else if (tier != "module") {
      std::cerr << "compare: unknown --tier=" << tier << "\n";
      return 1;
    }
  }

  auto report_one = [&](const std::string& label, const Tensor& test,
                        const Tensor& ref) {
    ErrorReport e = Compare(test, ref);
    std::string why;
    const bool ok = Accept(e, t, &why);
    std::cout << (ok ? "PASS  " : "FAIL  ") << label << "\n        "
              << e.ToString() << "\n";
    if (!ok) std::cout << "        against " << t.name << ": " << why << "\n";
    if (a.Has("axis") && test.Rank() == 2) {
      const int64_t axis = a.Int("axis", 0);
      WorstSlice w = WorstAlongAxis(CompareAlongAxis(test, ref, axis));
      std::cout << "        worst " << (axis == 0 ? "token " : "channel ")
                << w.index << ": " << w.report.ToString() << "\n";
    }
    return ok;
  };

  if (a.Has("test") && a.Has("ref")) {
    Tensor tt, rr;
    std::string n1, n2, m1, m2, err;
    if (!ReadTensorFile(a.Str("test"), &tt, &n1, &m1, &err) ||
        !ReadTensorFile(a.Str("ref"), &rr, &n2, &m2, &err)) {
      std::cerr << "compare: " << err << "\n";
      return 1;
    }
    if (!tt.SameShape(rr)) {
      std::cerr << "compare: shape mismatch " << tt.ShapeString() << " vs "
                << rr.ShapeString() << "\n";
      return 1;
    }
    return report_one(n1.empty() ? a.Str("test") : n1, tt, rr) ? 0 : 1;
  }

  if (a.Has("test-dir") && a.Has("ref-dir")) {
    const std::string td = a.Str("test-dir"), rd = a.Str("ref-dir");
    std::error_code ec;
    if (!std::filesystem::is_directory(rd, ec)) {
      std::cerr << "compare: " << rd << " is not a directory\n";
      return 1;
    }
    int checked = 0, failed = 0, missing = 0;
    std::vector<std::filesystem::path> refs;
    for (const auto& e : std::filesystem::directory_iterator(rd, ec))
      if (e.is_regular_file() && e.path().extension() == ".tensor")
        refs.push_back(e.path());
    // Directory iteration order is not specified; sort so two runs of this
    // command produce the same report.
    std::sort(refs.begin(), refs.end());

    for (const auto& rp : refs) {
      const std::string tp = td + "/" + rp.filename().string();
      Tensor tt, rr;
      std::string n1, n2, m1, m2, err;
      if (!ReadTensorFile(rp.string(), &rr, &n2, &m2, &err)) {
        std::cout << "SKIP  " << rp.filename().string() << ": " << err << "\n";
        continue;
      }
      if (!ReadTensorFile(tp, &tt, &n1, &m1, &err)) {
        std::cout << "MISS  " << n2 << ": " << err << "\n";
        ++missing;
        continue;
      }
      if (!tt.SameShape(rr)) {
        std::cout << "FAIL  " << n2 << ": shape " << tt.ShapeString() << " vs "
                  << rr.ShapeString() << "\n";
        ++failed;
        continue;
      }
      ++checked;
      if (!report_one(n2, tt, rr)) ++failed;
    }
    std::cout << "\n" << checked << " compared, " << failed << " failed, "
              << missing << " missing, against " << t.name << "\n";
    if (missing > 0)
      std::cout << "a MISSING tensor is not a pass. The reference directory "
                   "declares the boundary contract;\nan encrypted run that "
                   "does not produce a boundary has not implemented it.\n";
    return (failed == 0 && missing == 0) ? 0 : 1;
  }

  std::cerr << "compare: give --test=FILE --ref=FILE, or "
               "--test-dir=DIR --ref-dir=DIR\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = ParseArgs(argc, argv);
    if (a.command.empty() || a.command == "help" || a.command == "--help") {
      PrintUsage();
      return 0;
    }
    if (a.command == "config") {
      std::cout << ConfigProvenance() << "\n" << RopeConventionNote() << "\n";
      std::cout << "Resolved configuration: " << ConfigFromArgs(a).Identity() << "\n";
      return 0;
    }
    if (a.command == "conventions") {
      std::cout << NumericalConventions() << "\n"
                << SerializationFormatDoc() << "\n"
                << PythonReaderSnippet() << "\n";
      return 0;
    }
    if (a.command == "boundaries") {
      std::cout << "Module boundaries exported per layer, in emission order.\n"
                << "Names are prefixed with the layer, e.g. \"L0.attn.probs\".\n\n";
      char buf[512];
      std::snprintf(buf, sizeof(buf), "%-22s %-18s %-28s %-12s\n", "suffix",
                    "axes", "shape", "fit role");
      std::cout << buf;
      for (const BoundaryDecl& d : LayerBoundaries()) {
        std::snprintf(buf, sizeof(buf), "%-22s %-18s %-28s %-12s\n",
                      d.suffix.c_str(), d.dims.c_str(), d.shape_expr.c_str(),
                      d.fit_role.empty() ? "-" : d.fit_role.c_str());
        std::cout << buf;
        std::cout << "    " << d.description << "\n";
      }
      return 0;
    }
    if (a.command == "thresholds") {
      std::cout << ThresholdRationale();
      return 0;
    }
    if (a.command == "vectors") {
      std::cout << TestVectorCatalogue();
      return 0;
    }
    if (a.command == "weights-howto") {
      std::cout << LocalWeightsHowto();
      return 0;
    }
    if (a.command == "dump") return CmdDump(a);
    if (a.command == "calibrate") return CmdCalibrate(a);
    if (a.command == "fit") return CmdFit(a);
    if (a.command == "selftest") return CmdSelfTest(a);
    if (a.command == "compare") return CmdCompare(a);

    std::cerr << "unknown command '" << a.command << "'\n\n";
    PrintUsage();
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
