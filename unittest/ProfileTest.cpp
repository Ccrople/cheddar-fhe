// Online timings for the Llama non-linear operators.
//
// Every run in this project so far has been a correctness run; nothing built
// here has ever been timed. Per CLAUDE.md, key generation, warm-up and online
// evaluation are separated, and the GPU, toolchain, preset and problem shape are
// named in the report rather than implied.
//
// THE NUMBER THIS EXISTS FOR. SoftMax can run two ways: fused, 13 levels and no
// bootstrap, or with the auxiliary track bootstrapped, 7 levels and one
// bootstrap ([SYLPH] figure 2). Six levels against one bootstrap is a real
// trade and the level budget cannot settle it -- only a measurement can. Both
// are timed back to back here on the same ciphertext.
//
// NOT MEASURED HERE: PCMM and CCMM. By [SYLPH] table 6 they are the larger
// share of a layer (QKV 326 ms, gate/up 1040 ms, QK 316, ScoreV 452 against
// SoftMax's 440 and the norms' 40), so a per-layer total cannot be assembled
// from this file alone. What it does settle is the SoftMax question and the cost
// of each non-linear operator we wrote.

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "Testbed.h"
#include "extension/RmsNorm.h"
#include "extension/RoPe.h"
#include "extension/SiLu.h"
#include "extension/SoftMax.h"

using word = uint32_t;

namespace {

constexpr int kWarmUp = 3;
constexpr int kReps = 10;

// Warm up, then time kReps calls and average. The Testbed macro times a single
// iteration after its warm-up, which is too noisy to compare 40 ms against
// 100 ms.
template <typename F>
double TimeMs(F f) {
  for (int i = 0; i < kWarmUp; i++) f();
  cudaDeviceSynchronize();
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kReps; i++) f();
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / kReps;
}

void Report(const std::string &what, double ms, const std::string &shape) {
  std::cout << "  " << what << std::string(std::max(0, 34 - (int)what.size()), ' ')
            << ms << " ms   " << shape << std::endl;
}

}  // namespace

TEST_P(Testbed32, ProfileNonLinearOperators) {
  auto boot_context = std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr);
  const int level = default_encryption_level_;
  const int slots = param_->degree_ / 2;

  std::cout << "preset " << GetParam() << ", logN " << param_->log_degree_
            << ", scale 2^" << (int)std::round(std::log2(param_->base_scale_))
            << ", encryption level " << level << ", " << slots << " slots"
            << std::endl;

  // ---- key generation, outside every timed region -------------------------
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, slots);
  interface_->PrepareRotationKey(req);

  constexpr int kTokens = 64;
  constexpr int kChannels = 4096;
  constexpr int kKeys = 128;
  constexpr double kSmRange = 21.0;
  constexpr int kSmExpDeg = 13, kSmInvDeg = 12;
  const double kNormLo = 1.5, kNormHi = 30.0;

  RmsNormHandler<word> rms(context_, kTokens, kChannels, 1.0, level, 1e-5, 6.0,
                           9);
  for (int d : rms.GetRotationDistances()) interface_->PrepareRotationKey(d, level);
  SiLuHandler<word> silu(context_, 12.0, level, 31);
  SoftMaxHandler<word> sm_fused(context_, kKeys, kSmRange, level, 1,
                                {kNormLo}, {kNormHi}, kSmExpDeg, kSmInvDeg);
  SoftMaxHandler<word> sm_boot(context_, kKeys, kSmRange, level, 1, {kNormLo},
                               {kNormHi}, kSmExpDeg, kSmInvDeg, 4, true);
  for (int d : sm_fused.GetRotationDistances())
    interface_->PrepareRotationKey(d, level);
  RoPeHandler<word> rope(context_, kTokens, 128, level, 500000.0);
  for (int d : rope.GetRotationDistances()) interface_->PrepareRotationKey(d, level);

  // ---- inputs ------------------------------------------------------------
  std::vector<Complex> msg(slots), mask(slots, Complex(1.0, 0.0));
  for (int s = 0; s < slots; s++) {
    msg[s] = Complex(0.3 * std::sin(0.01 * s), 0.0);
  }
  Ciphertext<word> ct, res;
  EncodeAndEncrypt(ct, msg, level);

  const int num_ct = rms.GetNumCiphertexts();
  std::vector<Ciphertext<word>> rms_in(num_ct), rms_out;
  std::vector<std::vector<Complex>> wts(num_ct,
                                        std::vector<Complex>(slots,
                                                             Complex(1.0, 0.0)));
  for (int i = 0; i < num_ct; i++) EncodeAndEncrypt(rms_in[i], msg, level);

  Ciphertext<word> boot_in;
  EncodeAndEncrypt(boot_in, msg, 0);

  // ---- plaintext conversion, explicitly in setup --------------------------
  //
  // [SYLPH] section 5.3 makes the model's plaintext conversion its own stage
  // and section 5.1 keeps the result on the GPU for the whole run. Doing it
  // here rather than letting Apply do it lazily is what makes the timings
  // below online numbers rather than a first-call encode in disguise.
  rope.Prepare(0, level);
  sm_fused.Prepare(mask);
  sm_boot.Prepare(mask);
  rms.Prepare(wts);
  const size_t pt_bytes = rope.GetPlaintextBytes() +
                          sm_fused.GetPlaintextBytes() +
                          rms.GetPlaintextBytes();
  std::cout << "resident plaintexts: RoPE " << rope.GetPlaintextBytes() / 1e6
            << " MB, SoftMax mask " << sm_fused.GetPlaintextBytes() / 1e6
            << " MB, RMSNorm weights " << rms.GetPlaintextBytes() / 1e6
            << " MB, total " << pt_bytes / 1e6 << " MB" << std::endl;

  // ---- online evaluation --------------------------------------------------
  std::cout << "online, " << kWarmUp << " warm-up then mean of " << kReps
            << ":" << std::endl;
  const auto &evk = interface_->GetEvkMap();

  Report("Bootstrap", TimeMs([&] {
           Ciphertext<word> o;
           boot_context->Boot(o, boot_in, evk);
         }), "1 ct, all slots");

  Report("RoPE", TimeMs([&] { rope.Apply(res, ct, 0, evk); }),
         "1 ct, T=64 D=128");

  Report("SiLU", TimeMs([&] { silu.Apply(res, ct, evk); }), "1 ct, degree 31");

  Report("SoftMax fused (13 levels)",
         TimeMs([&] { sm_fused.Apply(res, ct, mask, evk); }), "1 ct, d=128");

  Report("SoftMax + aux boot (7 levels)",
         TimeMs([&] { sm_boot.Apply(res, ct, mask, evk, boot_context.get()); }),
         "1 ct, d=128");

  Report("RMSNorm", TimeMs([&] {
           rms.Apply(rms_out, rms_in, wts, evk);
         }), std::to_string(num_ct) + " ct, T=64 H=4096");

  // ---- what a plaintext encode costs on its own ---------------------------
  //
  // The claim being tested: these operator timings are dominated by host-side
  // plaintext encoding, not by homomorphic work. Encoder::Encode runs
  // SpecialIFFT over 32768 slots and then num_primes * degree BigInt::Mod
  // reductions, all single-threaded on the CPU, before anything reaches the
  // GPU. EncodeConstant does neither: one scalar, num_primes reductions, no
  // IFFT. SiLU calls neither and comes in at 7.8 ms; RoPE calls Encode three
  // times. If the numbers below are what they look like, that is the whole
  // story rather than an inference from SiLU.
  std::cout << "plaintext preparation, same warm-up and repeats:" << std::endl;
  std::vector<Complex> table(slots, Complex(0.5, 0.0));
  for (int lv : {level, level / 2}) {
    Plaintext<word> pt;
    const double sc = param_->GetScale(lv);
    Report("Encode (full width), level " + std::to_string(lv),
           TimeMs([&] { context_->encoder_.Encode(pt, lv, sc, table); }),
           std::to_string(param_->LevelToNP(lv).GetNumTotal()) + " primes");
  }
  {
    Constant<word> c;
    const double sc = param_->GetScale(level);
    Report("EncodeConstant, level " + std::to_string(level),
           TimeMs([&] { context_->encoder_.EncodeConstant(c, level, sc, 0.5); }),
           "scalar");
  }

  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::cout << "PCMM and CCMM are not in this file; by [SYLPH] table 6 they are "
               "the larger share of a layer, so no per-layer total should be "
               "assembled from these numbers alone."
            << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
