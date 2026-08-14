/**
 * @brief S3 capability probe.
 *
 * Focused correctness probes for exactly the Cheddar operations a Llama-3
 * decoder block needs, each one traced through OpTracer so that the level,
 * scale, modulus counts, component count and decryption error are recorded at
 * every step.
 *
 * Every probe validates against a host-computed reference, never against a
 * second run of the same GPU code.
 *
 * Run one parameter set at a time:
 *   ./capability_probe --gtest_filter='Cheddar/ProbeBed.*<name>*<param>'
 */

#include "ProbeSupport.h"

using namespace cheddar;
using word = uint32_t;

namespace {

// Host decode is O(num_slots * num_primes^2) BigInt work, so every traced
// probe uses a small slot count. Timing-only probes use whatever they need.
constexpr int kTraceSlots = 128;

// Bootstrapping is prepared per slot count; a small one keeps the host-side
// decode of the probe cheap without changing what the GPU path executes.
constexpr int kBootSlots = 1024;

std::string TraceId(const std::string &name, const std::string &param) {
  std::string clean = param;
  std::replace(clean.begin(), clean.end(), '.', '_');
  return "s3_" + name + "_" + clean;
}

/**
 * @brief Reproduce the target-scale rule that EvalMod uses when it builds an
 * EvalPoly (EvalMod.cpp:28-35). The output scale of a polynomial evaluation is
 * NOT the default scale of the target level; it is the scale that emerges from
 * repeated squaring and rescaling. Module code must carry this scale forward.
 */
double PredictEvalPolyTargetScale(const Parameter<word> &param, int input_level,
                                  double input_scale, int level_consumption) {
  double scale = input_scale;
  for (int i = 0; i < level_consumption; ++i) {
    scale = scale * scale / param.GetRescalePrimeProd(input_level - i);
  }
  return scale;
}

int LevelConsumption(int degree) { return Log2Ceil(degree + 1); }

}  // namespace

// ===========================================================================
// 1. Environment and parameter-set report
// ===========================================================================

TEST_P(ProbeBed, Environment) {
  s3::PrintEnvironment(std::cout);
  s3::PrintParameterSet(std::cout, *this, GetParam());

  std::cout << "-------------------- setup phases --------------------\n";
  std::cout << "context_create_us : " << context_create_us_
            << "  (total setup minus isolated basic keygen)\n";
  std::cout << "basic_keygen_us   : " << basic_keygen_us_
            << "  (secrets + mult/conj/dts/std keys, measured in isolation)\n";
  std::cout << "setup_total_us    : " << setup_total_us_ << "\n";
  std::cout << "NOTE: key generation is offline setup. It is never included in "
               "any online latency reported below.\n";

  std::cout << "-------------------- per-level ledger --------------------\n";
  std::cout << "level,num_main,num_ter,num_aux,num_q,scale_log2\n";
  for (int level = 0; level <= param_->max_level_; ++level) {
    const NPInfo np = param_->LevelToNP(level);
    std::cout << level << ',' << np.num_main_ << ',' << np.num_ter_ << ','
              << np.num_aux_ << ',' << np.GetNumQ() << ',';
    if (level <= param_->default_encryption_level_) {
      std::cout << std::log2(param_->GetScale(level));
    } else {
      std::cout << "n/a(above default_encryption_level)";
    }
    std::cout << '\n';
  }

  // logN=16 audit: what the library itself claims versus what is configured.
  std::cout << "-------------------- logN audit --------------------\n";
  std::cout << "NTTHandler::min_log_degree_ : " << NTTHandler<word>::min_log_degree_
            << "\n";
  std::cout << "NTTHandler::max_log_degree_ : " << NTTHandler<word>::max_log_degree_
            << "\n";
  std::cout << "configured log_degree       : " << param_->log_degree_ << "\n";
  std::cout << "This probe exercises ONLY the configured ring. A ring in the "
               "declared [min,max] range is not thereby verified.\n";
  EXPECT_EQ(param_->log_degree_, 16)
      << "S3 baseline is logN=16; a different ring is out of this session's "
         "scope and belongs to S1.";
}

// ===========================================================================
// 2. Encode / decode and encrypt / decrypt
// ===========================================================================

TEST_P(ProbeBed, EncodeDecode) {
  s3::DetRandom rng;
  std::cout << "level,max_abs_err,mean_abs_err,snr\n";
  for (int level = 0; level <= param_->max_level_; ++level) {
    std::vector<Complex> msg;
    rng.ComplexMsg(msg, kTraceSlots);

    Plaintext<word> pt;
    Encode(pt, msg, level);
    std::vector<Complex> decoded;
    Decode(decoded, pt);

    const s3::ErrorStats stats = s3::CompareDecoded(msg, decoded);
    std::cout << level << ',' << stats.max_abs << ',' << stats.mean_abs << ','
              << stats.snr << '\n';
    EXPECT_LT(stats.max_abs, 1e-5) << "encode/decode round trip at level "
                                   << level;
  }
}

TEST_P(ProbeBed, EncryptDecrypt) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("encrypt_decrypt", GetParam()));
  tracer.SetChain("encrypt_decrypt");
  s3::DetRandom rng;

  for (int level = 0; level <= param_->max_level_; ++level) {
    std::vector<Complex> msg;
    rng.ComplexMsg(msg, kTraceSlots);

    Ciphertext<word> ct;
    tracer.TraceStep("Encrypt", ct, msg,
                     [&]() { EncodeAndEncrypt(ct, msg, level); },
                     {"SLOT", "slots=" + std::to_string(kTraceSlots)},
                     "level=" + std::to_string(level));
  }
  std::cout << "trace written to " << tracer.GetPath() << "\n";
  std::cout << "peak device MiB (delta over baseline): "
            << tracer.GetPeakDeltaMiB() << "\n";
}

// ===========================================================================
// 3. Add / Sub
// ===========================================================================

TEST_P(ProbeBed, AddSub) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("add_sub", GetParam()));
  tracer.SetChain("add_sub");
  s3::DetRandom rng;

  const int level = param_->max_level_;
  std::vector<Complex> m1, m2;
  rng.ComplexMsg(m1, kTraceSlots);
  rng.ComplexMsg(m2, kTraceSlots);
  const double const_value = m2[0].real();

  Ciphertext<word> ct1, ct2, res;
  Plaintext<word> pt2;
  Constant<word> c2;
  EncodeAndEncrypt(ct1, m1, level);
  EncodeAndEncrypt(ct2, m2, level);
  Encode(pt2, m2, level);
  EncodeConstant(c2, const_value, level);

  const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kTraceSlots)};
  auto expect = [&](auto op) {
    std::vector<Complex> ref(kTraceSlots);
    for (int i = 0; i < kTraceSlots; ++i) ref[i] = op(m1[i], m2[i]);
    return ref;
  };

  tracer.TraceStep("Add(ct,ct)", res,
                   expect([](Complex a, Complex b) { return a + b; }),
                   [&]() { context_->Add(res, ct1, ct2); }, meta);
  tracer.TraceStep("Add(ct,pt)", res,
                   expect([](Complex a, Complex b) { return a + b; }),
                   [&]() { context_->Add(res, ct1, pt2); }, meta);
  tracer.TraceStep(
      "Add(ct,const)", res,
      expect([&](Complex a, Complex) { return a + const_value; }),
      [&]() { context_->Add(res, ct1, c2); }, meta);
  tracer.TraceStep("Sub(ct,ct)", res,
                   expect([](Complex a, Complex b) { return a - b; }),
                   [&]() { context_->Sub(res, ct1, ct2); }, meta);
  tracer.TraceStep("Sub(ct,pt)", res,
                   expect([](Complex a, Complex b) { return a - b; }),
                   [&]() { context_->Sub(res, ct1, pt2); }, meta);
  tracer.TraceStep(
      "Sub(ct,const)", res,
      expect([&](Complex a, Complex) { return a - const_value; }),
      [&]() { context_->Sub(res, ct1, c2); }, meta);
  tracer.TraceStep("Neg(ct)", res,
                   expect([](Complex a, Complex) { return -a; }),
                   [&]() { context_->Neg(res, ct1); }, meta);

  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// 4. Mult / Relinearize / Rescale, step by step
// ===========================================================================

TEST_P(ProbeBed, MultRelinRescale) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("mult_relin_rescale", GetParam()));
  tracer.SetChain("mult_relin_rescale");
  s3::DetRandom rng;

  const int level = param_->max_level_;
  std::vector<Complex> m1, m2, ref(kTraceSlots);
  rng.ComplexMsg(m1, kTraceSlots);
  rng.ComplexMsg(m2, kTraceSlots);
  for (int i = 0; i < kTraceSlots; ++i) ref[i] = m1[i] * m2[i];

  Ciphertext<word> ct1, ct2;
  EncodeAndEncrypt(ct1, m1, level);
  EncodeAndEncrypt(ct2, m2, level);
  const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kTraceSlots)};

  // Unfused path: tensor -> relinearize -> rescale
  Ciphertext<word> tensored, relin, rescaled;
  tracer.TraceStep("Mult(ct,ct) [tensor only]", tensored, ref,
                   [&]() { context_->Mult(tensored, ct1, ct2); }, meta,
                   "3 components, scale=scale^2, level unchanged");
  tracer.TraceStep(
      "Relinearize", relin, ref,
      [&]() {
        context_->Relinearize(relin, tensored, interface_->GetMultiplicationKey());
      },
      meta, "back to 2 components, still scale^2");
  tracer.TraceStep("Rescale", rescaled, ref,
                   [&]() { context_->Rescale(rescaled, relin); }, meta,
                   "level-1, scale back to ~default");

  // Fused path
  Ciphertext<word> fused;
  tracer.TraceStep(
      "RelinearizeRescale [fused]", fused, ref,
      [&]() {
        context_->RelinearizeRescale(fused, tensored,
                                     interface_->GetMultiplicationKey());
      },
      meta, "fused; cost ~ one relinearize");

  Ciphertext<word> hmult;
  tracer.TraceStep(
      "HMult(rescale=true)", hmult, ref,
      [&]() {
        context_->HMult(hmult, ct1, ct2, interface_->GetMultiplicationKey(),
                        true);
      },
      meta);

  // Plaintext and constant products, which is what a PCMM will actually use.
  Plaintext<word> pt2;
  Encode(pt2, m2, level);
  Ciphertext<word> ctpt;
  tracer.TraceStep("Mult(ct,pt)", ctpt, ref,
                   [&]() { context_->Mult(ctpt, ct1, pt2); }, meta,
                   "no rescale performed");
  Ciphertext<word> ctpt_rs;
  tracer.TraceStep("Rescale after Mult(ct,pt)", ctpt_rs, ref,
                   [&]() { context_->Rescale(ctpt_rs, ctpt); }, meta);

  EXPECT_EQ(param_->NPToLevel(rescaled.GetNP()), level - 1)
      << "Rescale must consume exactly one level";
  EXPECT_EQ(param_->NPToLevel(fused.GetNP()), level - 1)
      << "Fused relin-rescale must consume exactly one level";
  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// 5. MultUnsafe: the cross-level product a residual path needs
// ===========================================================================

TEST_P(ProbeBed, MultUnsafeCompatibility) {
  std::cout << "IsMultUnsafeCompatible(level_a, level_b) over all pairs.\n";
  std::cout << "A '.' means the lower level's prime set is NOT a subset of the "
               "higher one, so a cross-level product is refused.\n";
  int incompatible = 0;
  std::cout << "     ";
  for (int b = 0; b <= param_->max_level_; ++b) std::cout << (b % 10);
  std::cout << '\n';
  for (int a = 0; a <= param_->max_level_; ++a) {
    std::cout << std::setw(4) << a << ' ';
    for (int b = 0; b <= param_->max_level_; ++b) {
      const bool ok = context_->IsMultUnsafeCompatible(a, b);
      if (!ok) ++incompatible;
      std::cout << (ok ? 'X' : '.');
    }
    std::cout << '\n';
  }
  std::cout << "incompatible_pairs=" << incompatible << "\n";

  // Now actually run one cross-level product and validate it.
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("mult_unsafe", GetParam()));
  tracer.SetChain("mult_unsafe");
  s3::DetRandom rng;

  const int high = param_->max_level_;
  const int low = param_->max_level_ - 3;
  ASSERT_TRUE(context_->IsMultUnsafeCompatible(high, low))
      << "no compatible cross-level pair to probe";

  std::vector<Complex> m1, m2, ref(kTraceSlots);
  rng.ComplexMsg(m1, kTraceSlots);
  rng.ComplexMsg(m2, kTraceSlots);
  for (int i = 0; i < kTraceSlots; ++i) ref[i] = m1[i] * m2[i];

  Ciphertext<word> ct_high, ct_low, res;
  EncodeAndEncrypt(ct_high, m1, high);
  EncodeAndEncrypt(ct_low, m2, low);

  const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kTraceSlots)};
  // The result scale is the PRODUCT of two different default scales, which is
  // exactly the scale-management hazard the header warns about.
  std::cout << "scale(high level " << high << ") = " << ct_high.GetScale()
            << "\nscale(low level " << low << ")  = " << ct_low.GetScale()
            << "\nproduct scale                = "
            << ct_high.GetScale() * ct_low.GetScale() << "\n";

  tracer.TraceStep("MultUnsafe(ct@" + std::to_string(high) + ",ct@" +
                       std::to_string(low) + ")",
                   res, ref,
                   [&]() { context_->MultUnsafe(res, ct_high, ct_low); }, meta,
                   "result lands at min(level)");
  EXPECT_EQ(param_->NPToLevel(res.GetNP()), low);
  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// 6. HRot, and the rotation-key rules
// ===========================================================================

TEST_P(ProbeBed, HRot) {
  s3::OpTracer<word> tracer(context_, *interface_, TraceId("hrot", GetParam()));
  tracer.SetChain("hrot");
  s3::DetRandom rng;

  const int level = param_->max_level_;
  const std::vector<int> distances{1, 2, 8, kTraceSlots / 2, kTraceSlots - 1};

  std::cout << "rot_dist,keygen_us\n";
  for (int d : distances) {
    const double keygen_us = TimeRotationKey(d, level);
    std::cout << d << ',' << keygen_us << "   (offline, not online latency)\n";
  }

  std::vector<Complex> msg;
  rng.ComplexMsg(msg, kTraceSlots);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, level);

  const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kTraceSlots)};
  for (int d : distances) {
    std::vector<Complex> ref(kTraceSlots);
    // Cheddar's convention, read off BasicTest.cpp:386 and verified here:
    // res[i] = msg[(i + d) mod num_slots], i.e. a LEFT rotation by d.
    for (int i = 0; i < kTraceSlots; ++i) ref[i] = msg[(i + d) % kTraceSlots];

    Ciphertext<word> res;
    tracer.TraceStep(
        "HRot(" + std::to_string(d) + ")", res, ref,
        [&]() {
          context_->HRot(res, ct, interface_->GetRotationKey(d), d);
        },
        meta, "left rotation: res[i]=msg[(i+d) mod n]");
  }

  // A negative (right) rotation is the same key as num_slots - d. Cheddar
  // normalizes rot_dist into [0, num_slots) inside HRot, so module code must
  // request the normalized index from the key map, not the signed one.
  {
    const int signed_rot = -1;
    const int normalized = kTraceSlots + signed_rot;
    std::vector<Complex> ref(kTraceSlots);
    for (int i = 0; i < kTraceSlots; ++i) {
      ref[i] = msg[((i + signed_rot) % kTraceSlots + kTraceSlots) % kTraceSlots];
    }
    Ciphertext<word> res;
    tracer.TraceStep(
        "HRot(-1) via key " + std::to_string(normalized), res, ref,
        [&]() {
          context_->HRot(res, ct, interface_->GetRotationKey(normalized),
                         signed_rot);
        },
        meta, "negative rotation needs key num_slots-d");
  }

  // HRotAdd: the fused form a reduction tree wants.
  {
    const int d = 8;
    std::vector<Complex> m2, ref(kTraceSlots);
    rng.ComplexMsg(m2, kTraceSlots);
    Ciphertext<word> ct2;
    EncodeAndEncrypt(ct2, m2, level);
    for (int i = 0; i < kTraceSlots; ++i) {
      ref[i] = msg[(i + d) % kTraceSlots] + m2[i];
    }
    Ciphertext<word> res;
    tracer.TraceStep(
        "HRotAdd(" + std::to_string(d) + ")", res, ref,
        [&]() {
          context_->HRotAdd(res, ct, ct2, interface_->GetRotationKey(d), d);
        },
        meta, "fused rotate-and-accumulate");
  }

  // HConj, needed wherever a real-only value must be extracted.
  {
    std::vector<Complex> ref(kTraceSlots);
    for (int i = 0; i < kTraceSlots; ++i) ref[i] = std::conj(msg[i]);
    Ciphertext<word> res;
    tracer.TraceStep(
        "HConj", res, ref,
        [&]() {
          context_->HConj(res, ct, interface_->GetConjugationKey());
        },
        meta);
  }

  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// 7. Rotation-key inventory for a Llama-shaped rotation set
// ===========================================================================

TEST_P(ProbeBed, RotationKeyInventory) {
  const int num_slots = kTraceSlots;

  // A slot-resident reduction over a field of `num_slots` needs the
  // power-of-two rotations; a BSGS linear map needs its baby and giant steps;
  // an attention/token shift needs signed offsets. Deliberately includes a
  // zero, a duplicate and an aliasing negative so the checker is exercised.
  std::vector<int> requested;
  for (int r = 1; r < num_slots; r *= 2) requested.push_back(r);
  for (int r = 1; r <= 8; ++r) requested.push_back(r);        // baby steps
  for (int r = 8; r < num_slots; r += 8) requested.push_back(r);  // giant steps
  requested.push_back(0);                                     // no-op
  requested.push_back(-1);                                    // aliases n-1
  requested.push_back(-8);                                    // aliases n-8
  requested.push_back(4);                                     // duplicate

  const s3::RotationInventory inv =
      s3::BuildRotationInventory(requested, num_slots);

  std::cout << "requested_raw          : " << requested.size() << "\n";
  std::cout << "distinct_keys_required : " << inv.NumKeysRequired() << "\n";
  std::cout << "duplicate_requests     : " << inv.num_duplicate_requests
            << "\n";
  std::cout << "zero_requests (no-op, and GetRotationKey(0) would ABORT): "
            << inv.zero_requests.size() << "\n";
  std::cout << "out_of_range           : " << inv.out_of_range.size() << "\n";
  std::cout << "aliased (>1 raw request mapping to one key):\n";
  for (const auto &[norm, raws] : inv.aliases) {
    if (raws.size() < 2) continue;
    std::cout << "  key " << norm << " <-";
    for (int r : raws) std::cout << ' ' << r;
    std::cout << '\n';
  }
  EXPECT_TRUE(inv.out_of_range.empty());
  EXPECT_FALSE(inv.zero_requests.empty())
      << "the probe intends to exercise the zero-rotation case";

  // Generate the whole set through the library and time it as offline setup.
  EvkRequest req;
  for (int idx : inv.normalized) req.AddRequest(idx, param_->max_level_);
  const double keygen_us = TimeRotationKeys(req);
  const s3::MemSample mem = s3::SampleMemory();

  std::cout << "EvkRequest size        : " << req.size() << "\n";
  std::cout << "rotation_keygen_us     : " << keygen_us
            << "   (OFFLINE SETUP -- not an inference latency)\n";
  std::cout << "us_per_key             : " << keygen_us / req.size() << "\n";
  std::cout << "device_used_mib_after  : " << mem.used_mib << " of "
            << mem.total_mib << " (shared card)\n";

  // Every requested key must now resolve. GetEvk asserts on a miss, so a
  // missing key aborts the process rather than returning; the check below is
  // therefore a positive confirmation that the inventory was complete.
  for (int idx : inv.normalized) {
    const auto &key = interface_->GetRotationKey(idx);
    EXPECT_GT(key.GetBeta(), 0) << "rotation key " << idx << " is empty";
  }
  std::cout << "all " << inv.NumKeysRequired()
            << " rotation keys resolved; none missing.\n";
}

// ===========================================================================
// 8. LinearTransform / BSGS -- the closest existing thing to a PCMM
// ===========================================================================

TEST_P(ProbeBed, LinearTransformBSGS) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("linear_transform", GetParam()));
  tracer.SetChain("linear_transform");
  s3::DetRandom rng;

  const int n = kTraceSlots;
  const int pt_level = param_->max_level_;
  const double pt_scale = DetermineScale(pt_level);

  // LinearTransform asserts a SQUARE power-of-two matrix (height == width),
  // so this probe builds an n x n map. That constraint is a capability result
  // in its own right: a rectangular Llama projection cannot be handed to
  // LinearTransform directly.
  const int bs = 8;
  const int gs = 4;  // bs * gs = 32 diagonal slots available at stride 1
  StripedMatrix matrix(n, n);
  std::vector<int> diag_indices;
  for (int d = 0; d < bs * gs; d += 2) diag_indices.push_back(d);

  for (int d : diag_indices) {
    std::vector<Complex> diag;
    rng.ComplexMsg(diag, n, -0.5, 0.5);
    matrix[d] = diag;
  }
  std::cout << "matrix " << matrix.GetHeight() << "x" << matrix.GetWidth()
            << ", diagonals=" << matrix.GetNumDiag() << ", bs=" << bs
            << ", gs=" << gs << "\n";

  LinearTransform<word> lt(context_, matrix, pt_level, pt_scale, bs, gs);
  std::cout << "using_bsgs=" << lt.IsUsingBSGS() << " bs=" << lt.GetBS()
            << " gs=" << lt.GetGS() << "\n";

  EvkRequest req;
  lt.AddRequiredRotations(req);
  std::cout << "rotation keys required by this transform: " << req.size()
            << "\n  indices:";
  for (const auto &[idx, lvl] : req) std::cout << ' ' << idx << "@L" << lvl;
  std::cout << '\n';
  const double lt_keygen_us = TimeRotationKeys(req);
  std::cout << "lt_rotation_keygen_us : " << lt_keygen_us
            << "   (OFFLINE SETUP)\n";

  // min_ks needs a different, smaller key set; report both inventories.
  EvkRequest req_minks;
  lt.AddRequiredRotations(req_minks, true);
  std::cout << "rotation keys required with min_ks: " << req_minks.size()
            << "\n";

  std::vector<Complex> msg;
  rng.ComplexMsg(msg, n, -1.0, 1.0);

  // Two plausible diagonal conventions differ only in the sign of the shift.
  // The probe computes both on the host and reports which one the library
  // actually implements, rather than assuming one and reporting a failure.
  std::vector<Complex> ref_left(n, Complex(0.0, 0.0));
  std::vector<Complex> ref_right(n, Complex(0.0, 0.0));
  for (const auto &[d, diag] : matrix) {
    for (int i = 0; i < n; ++i) {
      ref_left[i] += diag[i] * msg[(i + d) % n];
      ref_right[i] += diag[i] * msg[((i - d) % n + n) % n];
    }
  }

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, pt_level);
  const s3::LogicalMeta meta{"SLOT", "square " + std::to_string(n) + "x" +
                                         std::to_string(n)};

  Ciphertext<word> res;
  const s3::ErrorStats stats = tracer.TraceStep(
      "LinearTransform(BSGS)", res, ref_left,
      [&]() { lt.Evaluate(context_, res, ct, interface_->GetEvkMap()); }, meta,
      "diagonal-encoded square map, vs left-rotation convention");

  const std::vector<Complex> obtained = tracer.Peek(res);
  const s3::ErrorStats err_left = s3::CompareDecoded(ref_left, obtained);
  const s3::ErrorStats err_right = s3::CompareDecoded(ref_right, obtained);
  std::cout << "diagonal convention check:\n"
            << "  out[i] = sum_d diag_d[i]*in[(i+d) mod n] -> max_err "
            << err_left.max_abs << "\n"
            << "  out[i] = sum_d diag_d[i]*in[(i-d) mod n] -> max_err "
            << err_right.max_abs << "\n"
            << "  MATCHED CONVENTION: "
            << (err_left.max_abs < err_right.max_abs ? "(i+d), same direction "
                                                       "as HRot"
                                                     : "(i-d), OPPOSITE to "
                                                       "HRot")
            << "\n";
  EXPECT_LT(std::min(err_left.max_abs, err_right.max_abs), 1e-3)
      << "BSGS linear transform matched neither host matvec convention";
  std::cout << "level in=" << pt_level
            << " level out=" << param_->NPToLevel(res.GetNP())
            << "  (levels consumed="
            << pt_level - param_->NPToLevel(res.GetNP()) << ")\n";

  // Online latency, warm-up separated.
  const s3::LatencyStats lat = s3::TimeRepeated(
      [&]() { lt.Evaluate(context_, res, ct, interface_->GetEvkMap()); }, 3, 10);
  std::cout << "linear_transform warm_up_us=" << lat.warm_up_us
            << " median_us=" << lat.median_us << " min_us=" << lat.min_us
            << " max_us=" << lat.max_us << " iters=" << lat.iters << "\n";
  std::cout << "peak device MiB (delta over baseline): "
            << tracer.GetPeakDeltaMiB() << "\n";
  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// 9. EvalPoly: the four Llama non-linearity fits
// ===========================================================================

TEST_P(ProbeBed, EvalPolyNonlinearities) {
  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("eval_poly", GetParam()));
  s3::DetRandom rng;

  // The ranges here are placeholders until S2's calibration lands. The probe
  // reports the fit error separately from the ciphertext error so the two are
  // never confused.
  std::vector<s3::NonlinearFit> fits{s3::RsqrtFit(), s3::ExpFit(),
                                     s3::SiluFit(), s3::ReciprocalFit()};

  for (auto &fit : fits) {
    SCOPED_TRACE(fit.name);
    tracer.SetChain("evalpoly_" + fit.name);

    // The level consumption must be derived from the degree EvalPoly will
    // actually compile (after near-zero trimming), because the target scale
    // depends on it and a scale mismatch aborts the process.
    const int actual_degree = fit.effective_degree;
    const int predicted_levels = LevelConsumption(actual_degree);

    // A real chain sits at the level a bootstrap returns to, not at the very
    // top of the ladder, so the probe uses default_encryption_level_ where the
    // scale ladder is the clean power-of-two one.
    const int input_level = param_->default_encryption_level_;
    ASSERT_GT(input_level, predicted_levels)
        << "not enough levels for a degree-" << actual_degree << " fit";
    const double input_scale = DetermineScale(input_level);
    const double target_scale = PredictEvalPolyTargetScale(
        *param_, input_level, input_scale, predicted_levels);

    EvalPoly<word> poly(fit.coeffs, input_level, input_scale, target_scale,
                        /*chebyshev=*/true);
    ASSERT_EQ(poly.GetPolyDegree(), actual_degree)
        << "the probe's near-zero trimming disagrees with EvalPoly's";
    poly.Compile(context_);

    std::cout << "---- " << fit.name << " ----\n";
    std::cout << "range=[" << fit.domain.lo << "," << fit.domain.hi
              << "] requested_degree=" << fit.degree
              << " actual_degree=" << actual_degree
              << " (coefficients below " << kZeroCoeffThreshold
              << " are dropped by EvalPoly)\n";
    std::cout << "levels: consumption=" << predicted_levels
              << " input_level=" << input_level << " expected_out_level="
              << input_level - predicted_levels << "\n";
    std::cout << "scales: input=2^" << std::log2(input_scale) << " target=2^"
              << std::log2(target_scale)
              << "   NOTE: the target scale is NOT the default scale of the "
                 "output level; it is scale^2/rescale_prime_prod iterated.\n";
    std::cout << "chebyshev_fit_max_error=" << fit.fit_error
              << "   (approximation error alone, no ciphertext involved)\n";

    // Host cross-check of the coefficient convention against the library's own
    // plaintext evaluator, before anything is encrypted.
    double worst_plain = 0.0;
    for (int i = 0; i <= 200; ++i) {
      const double t = -1.0 + 2.0 * i / 200.0;
      worst_plain =
          std::max(worst_plain, std::abs(poly.PlainEvaluate(t) - fit.unit_fn(t)));
    }
    std::cout << "evalpoly_plain_vs_truth_max_error=" << worst_plain << "\n";
    EXPECT_LT(worst_plain, 10.0 * fit.fit_error + 1e-6)
        << "coefficient convention mismatch: EvalPoly's plaintext evaluation "
           "disagrees with the host Chebyshev fit";

    // Ciphertext evaluation.
    std::vector<Complex> msg;
    rng.Real(msg, kTraceSlots, -0.95, 0.95);
    std::vector<Complex> ref(kTraceSlots);
    for (int i = 0; i < kTraceSlots; ++i) {
      // Compare against the polynomial the circuit actually evaluates, so the
      // measured error is ciphertext error and not approximation error.
      ref[i] = Complex(poly.PlainEvaluate(msg[i].real()), 0.0);
    }

    Ciphertext<word> ct, res;
    EncodeAndEncrypt(ct, msg, input_level);
    const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kTraceSlots)};
    tracer.Trace("input", ct, {}, meta, 0.0, fit.name);

    const s3::ErrorStats stats = tracer.TraceStep(
        "EvalPoly(" + fit.name + ",deg" + std::to_string(actual_degree) + ")",
        res, ref,
        [&]() {
          poly.Evaluate(context_, res, ct, interface_->GetMultiplicationKey());
        },
        meta, "vs the same polynomial evaluated on the host");
    EXPECT_LT(stats.max_abs, 1e-2)
        << "ciphertext polynomial evaluation error for " << fit.name;

    const int out_level = param_->NPToLevel(res.GetNP());
    EXPECT_EQ(out_level, input_level - predicted_levels)
        << "level consumption did not match ceil(log2(deg+1))";
    std::cout << "measured_out_level=" << out_level
              << " measured_out_scale=2^" << std::log2(res.GetScale()) << "\n";

    const s3::LatencyStats lat = s3::TimeRepeated(
        [&]() {
          poly.Evaluate(context_, res, ct, interface_->GetMultiplicationKey());
        },
        3, 10);
    std::cout << "evalpoly warm_up_us=" << lat.warm_up_us
              << " median_us=" << lat.median_us << " min_us=" << lat.min_us
              << " max_us=" << lat.max_us << "\n";
  }
  std::cout << "peak device MiB (delta over baseline): "
            << tracer.GetPeakDeltaMiB() << "\n";
  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

// ===========================================================================
// 10. Bootstrapping
// ===========================================================================

TEST_P(ProbeBed, Bootstrap) {
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr)
      << "parameter set does not enable bootstrapping";

  s3::OpTracer<word> tracer(context_, *interface_,
                            TraceId("bootstrap", GetParam()));
  tracer.SetChain("bootstrap");
  s3::DetRandom rng;

  const double prep_evalmod_us =
      s3::TimeOnce([&]() { boot_context->PrepareEvalMod(); });
  const double prep_fft_us = s3::TimeOnce(
      [&]() { boot_context->PrepareEvalSpecialFFT(kBootSlots); });

  EvkRequest req;
  boot_context->AddRequiredRotations(req, kBootSlots);
  std::cout << "boot rotation keys required (basic): " << req.size() << "\n";
  EvkRequest req_minks;
  boot_context->AddRequiredRotations(req_minks, kBootSlots, true);
  std::cout << "boot rotation keys required (min_ks): " << req_minks.size()
            << "\n";

  const double boot_keygen_us = TimeRotationKeys(req);
  const double minks_keygen_us = TimeRotationKeys(req_minks);

  std::cout << "OFFLINE SETUP (never part of an inference latency):\n";
  std::cout << "  prepare_evalmod_us     : " << prep_evalmod_us << "\n";
  std::cout << "  prepare_specialfft_us  : " << prep_fft_us << "\n";
  std::cout << "  boot_rotation_keygen_us: " << boot_keygen_us << "\n";
  std::cout << "  minks_extra_keygen_us  : " << minks_keygen_us << "\n";
  std::cout << "  basic_keygen_us        : " << basic_keygen_us_ << "\n";
  std::cout << "  context_create_us      : " << context_create_us_ << "\n";

  ASSERT_TRUE(boot_context->IsBootPrepared(kBootSlots));

  std::vector<Complex> msg;
  rng.ComplexMsg(msg, kBootSlots, -1.0, 1.0);
  Ciphertext<word> ct, res;
  EncodeAndEncrypt(ct, msg, 0);

  const s3::LogicalMeta meta{"SLOT", "slots=" + std::to_string(kBootSlots)};
  tracer.Trace("pre-boot input", ct, msg, meta, 0.0, "encrypted at level 0");

  const s3::ErrorStats stats = tracer.TraceStep(
      "Boot(basic)", res, msg,
      [&]() { boot_context->Boot(res, ct, interface_->GetEvkMap()); }, meta,
      "first call: includes warm-up");
  EXPECT_LT(stats.max_abs, 1e-2) << "bootstrap correctness";

  const int out_level = param_->NPToLevel(res.GetNP());
  std::cout << "boot: level 0 -> " << out_level
            << " (default_encryption_level=" << param_->default_encryption_level_
            << ")\n";

  const s3::LatencyStats lat = s3::TimeRepeated(
      [&]() { boot_context->Boot(res, ct, interface_->GetEvkMap()); }, 3, 5);
  std::cout << "boot_basic warm_up_us=" << lat.warm_up_us
            << " median_us=" << lat.median_us << " min_us=" << lat.min_us
            << " max_us=" << lat.max_us << "\n";

  Ciphertext<word> res_minks;
  const s3::ErrorStats minks_stats = tracer.TraceStep(
      "Boot(min_ks)", res_minks, msg,
      [&]() {
        boot_context->Boot(res_minks, ct, interface_->GetEvkMap(), true);
      },
      meta);
  EXPECT_LT(minks_stats.max_abs, 1e-2) << "min_ks bootstrap correctness";
  const s3::LatencyStats lat_minks = s3::TimeRepeated(
      [&]() {
        boot_context->Boot(res_minks, ct, interface_->GetEvkMap(), true);
      },
      3, 5);
  std::cout << "boot_minks warm_up_us=" << lat_minks.warm_up_us
            << " median_us=" << lat_minks.median_us << "\n";

  std::cout << "peak device MiB (delta over baseline): "
            << tracer.GetPeakDeltaMiB() << " of " << s3::SampleMemory().total_mib
            << " total (shared card)\n";
  std::cout << "trace written to " << tracer.GetPath() << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, ProbeBed,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json"),
    [](const testing::TestParamInfo<ProbeBed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
