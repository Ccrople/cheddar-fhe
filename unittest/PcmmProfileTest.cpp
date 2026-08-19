// PCMM: setup against online, and what the plaintexts and ciphertexts weigh.
//
// TWO THINGS THIS SETTLES.
//
// 1. THE SPLIT. [SYLPH] section 5.3 converts the model's weights offline into
//    the layout the product consumes -- and into the NTT domain where the
//    algorithm needs it -- and section 5.1 keeps the result on the GPU for the
//    whole run. So EncodeMatrix is setup and Multiply is online, and a
//    steady-state number must not fold the first into the second. Section 4.2's
//    runtime work is only the *rearrangement* of an already-encoded copy, by
//    two plaintext rotations, which is a permutation and not an encode.
//
// 2. THE FOOTPRINT, MEASURED RATHER THAN DERIVED. Table 5 puts the converted
//    Llama-3-8B weights at 52.0 GiB against 13.0 GiB of FP16, on a machine with
//    eight GPUs holding 1/g each. One A6000 has 48 GiB. Before planning around
//    per-layer streaming I want our own bytes, not the paper's: the arithmetic
//    I did by hand says 4096 degree-65536 ciphertexts need 48 GiB against a
//    47.5 GiB card. Measured: it is right, and the allocation dies at ~3072.
//    What was wrong was the ring. PCMM does not run at 2^16 -- [SYLPH] table 4
//    puts it at ring degree 256 in coefficient encoding, and `llama_pcmm_test`
//    instantiates on `ringdegree12_30`, degree 4096. There a ciphertext is
//    98 KB rather than 12.6 MB and 4096 of them are 403 MB. This test therefore
//    runs where PCMM actually runs; profiling it at 2^16 was measuring a
//    configuration the pipeline never uses.

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "Testbed.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"

using word = uint32_t;

namespace {
constexpr int kWarmUp = 2;
constexpr int kReps = 5;

template <typename F>
double TimeMs(F f, int warm, int reps) {
  for (int i = 0; i < warm; i++) f();
  cudaDeviceSynchronize();
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; i++) f();
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

double GiB(size_t b) { return b / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

TEST_P(Testbed32, ProfilePcmm) {
  constexpr int kTokens = 128;
  constexpr int kInChannels = 4096;
  constexpr int kOutChannels = 128;
  constexpr double kWeightScale = 1073741824.0;  // 2^30

  const int degree = param_->degree_;
  const int level = default_encryption_level_;
  const double ct_scale = DetermineScale(level);
  const int primes = param_->LevelToNP(level).GetNumTotal();

  // The sizes, before allocating anything, so the plan is checkable.
  const size_t ct_bytes =
      static_cast<size_t>(primes) * degree * sizeof(word) * 2;
  std::cout << "preset " << GetParam() << ", level " << level << ", " << primes
            << " primes, degree " << degree << std::endl;
  std::cout << "one ciphertext = " << ct_bytes / 1e6 << " MB; "
            << kInChannels << " of them = " << GiB(ct_bytes * kInChannels)
            << " GiB" << std::endl;
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  std::cout << "card: " << GiB(total_b) << " GiB total, " << GiB(free_b)
            << " GiB free outside the RMM pool (the pool takes what was free "
               "at context creation, so this is not a budget)"
            << std::endl;

  // Allocate the input ciphertexts the way LlamaPcmmTest does, reporting as we
  // go. If the hand arithmetic is right this cannot finish; if it finishes, the
  // arithmetic is wrong and the printed total says by how much.
  PcmmHandler<word> pcmm(*param_);
  std::vector<Ciphertext<word>> cts(kInChannels);
  {
    std::vector<double> coeffs(degree, 0.0);
    Plaintext<word> pt;
    for (int c = 0; c < kInChannels; c++) {
      for (int t = 0; t < kTokens; t++) coeffs[t] = 0.01 * ((c + t) % 97);
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(cts[c], pt);
      if ((c + 1) % 1024 == 0) {
        const int p = cts[c].GetNP().GetNumTotal();
        std::cout << "  " << (c + 1) << " ciphertexts held, each " << p
                  << " primes" << std::endl;
      }
    }
  }
  const int held_primes = cts[0].GetNP().GetNumTotal();
  std::cout << "held: " << kInChannels << " ciphertexts at " << held_primes
            << " primes = "
            << GiB(static_cast<size_t>(held_primes) * degree * sizeof(word) *
                   2 * kInChannels)
            << " GiB" << std::endl;

  // ---- setup: the model conversion --------------------------------------
  std::vector<double> u_values(static_cast<size_t>(kOutChannels) * kInChannels);
  for (size_t i = 0; i < u_values.size(); i++) {
    u_values[i] = 0.001 * static_cast<double>((i * 7919) % 211);
  }
  PlainMatrix<word> u;
  const double encode_ms = TimeMs(
      [&] {
        pcmm.EncodeMatrix(u, level, kWeightScale, u_values, kOutChannels,
                          kInChannels);
      },
      1, 2);
  std::cout << "SETUP  EncodeMatrix " << kOutChannels << "x" << kInChannels
            << ": " << encode_ms << " ms" << std::endl;

  // ---- online: the product ------------------------------------------------
  std::vector<Ciphertext<word>> res;
  const double mult_ms =
      TimeMs([&] { pcmm.Multiply(res, u, cts); }, kWarmUp, kReps);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::cout << "ONLINE Multiply     " << kOutChannels << "x" << kInChannels
            << " by " << kTokens << " tokens: " << mult_ms << " ms" << std::endl;

  // The converted weights are what [SYLPH] table 5 measures at 52.0 GiB for the
  // whole model. Ours, from the encoded matrix rather than from arithmetic.
  {
    const size_t u_words = static_cast<size_t>(u.GetRows()) * u.GetCols() *
                           u.GetNP().GetNumTotal();
    const double per_value = static_cast<double>(u_words) * sizeof(word) /
                             (static_cast<double>(u.GetRows()) * u.GetCols());
    std::cout << "converted weights: " << kOutChannels << "x" << kInChannels
              << " = " << u_words * sizeof(word) / 1e6 << " MB, "
              << per_value << " bytes per weight" << std::endl;
    const long long per_layer_values =
        4096LL * 6144 + 4096LL * 4096 + 4096LL * 28672 + 14336LL * 4096;
    std::cout << "  scaled to one Llama-3-8B layer ("
              << per_layer_values / 1e6 << "M weights): "
              << per_layer_values * per_value / 1e9 << " GB; 32 layers: "
              << per_layer_values * per_value * 32 / 1e9
              << " GB against [SYLPH] table 5's 55.8 GB and this card's 47.5"
              << std::endl;
  }

  // Scale to the real projections, which is the number the level budget wants.
  // Cost is linear in the output width for a fixed input width, so this is a
  // scaling of a measurement rather than a new one -- and it is labelled as
  // such.
  std::cout << "scaled linearly in output width (a projection, not a "
               "measurement):"
            << std::endl;
  const struct {
    const char *name;
    int out;
  } projections[] = {{"QKV  4096->6144", 6144},
                     {"O    4096->4096", 4096},
                     {"gate/up 4096->28672", 28672}};
  for (const auto &p : projections) {
    std::cout << "  " << p.name << "  ~"
              << mult_ms * p.out / kOutChannels << " ms online, ~"
              << encode_ms * p.out / kOutChannels / 1000.0 << " s setup"
              << std::endl;
  }
}

// ---------------------------------------------------------------------------
// The configuration [SYLPH] table 4 actually specifies: ring degree 256, MLWE.
//
// The RLWE-at-4096 measurement above is not what Sylph runs. Table 4 puts PCMM
// at ring degree 256 in coefficient encoding, and section 3.3 spells out the
// route: "ring-switching followed by an MLWE decomposition keeps all the token
// values of one given channel co-located in one ciphertext". ModDecomp takes a
// degree-4096 ciphertext to k = 16 MLWE ciphertexts of rank 16 at degree 256.
//
// The product is a scalar linear combination, so it commutes with ModDecomp:
// decompose every channel, then run one PCMM per MLWE component. A full 4096
// channel set is therefore 16 of these, and the per-product number below is
// what scales.
// ---------------------------------------------------------------------------
TEST_P(Testbed32, ProfilePcmmMlwe) {
  constexpr int kTokens = 128;
  constexpr int kColumns = 4096;
  constexpr int kOutChannels = 128;
  constexpr int kSmallDegree = 256;
  constexpr double kWeightScale = 1073741824.0;

  const int degree = param_->degree_;
  const int level = default_encryption_level_;
  const double ct_scale = DetermineScale(level);
  const int k = degree / kSmallDegree;
  ASSERT_EQ(kColumns % k, 0);

  PcmmHandler<word> pcmm(*param_);
  MlweHandler<word> mlwe(*param_);

  // 4096 columns = 4096/k source ciphertexts, each decomposing into k parts.
  std::vector<MlweCiphertext<word>> cts;
  cts.reserve(kColumns);
  {
    std::vector<double> coeffs(degree, 0.0);
    Plaintext<word> pt;
    Ciphertext<word> ct;
    std::vector<MlweCiphertext<word>> parts;
    for (int s = 0; s < kColumns / k; s++) {
      for (int i = 0; i < kTokens; i++) coeffs[i] = 0.01 * ((s + i) % 97);
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(ct, pt);
      mlwe.ModDecomp(parts, ct, kSmallDegree);
      ASSERT_EQ(static_cast<int>(parts.size()), k);
      for (auto &p : parts) cts.push_back(std::move(p));
    }
  }
  const int rank = cts[0].GetRank();
  const int primes = cts[0].GetNP().GetNumTotal();
  // rank * N' words for a_, N' for b_, per limb.
  const size_t mlwe_bytes = static_cast<size_t>(primes) *
                            (static_cast<size_t>(rank) * kSmallDegree +
                             kSmallDegree) *
                            sizeof(word);
  std::cout << "MLWE: degree " << cts[0].GetDegree() << ", rank " << rank
            << ", " << primes << " primes -> " << mlwe_bytes / 1e3
            << " KB each; " << kColumns << " columns = "
            << GiB(mlwe_bytes * kColumns) << " GiB" << std::endl;

  std::vector<double> u_values(static_cast<size_t>(kOutChannels) * kColumns);
  for (size_t i = 0; i < u_values.size(); i++) {
    u_values[i] = 0.001 * static_cast<double>((i * 7919) % 211);
  }
  PlainMatrix<word> u;
  const double encode_ms = TimeMs(
      [&] {
        pcmm.EncodeMatrix(u, level, kWeightScale, u_values, kOutChannels,
                          kColumns);
      },
      1, 2);

  std::vector<MlweCiphertext<word>> res;
  const double mult_ms =
      TimeMs([&] { pcmm.Multiply(res, u, cts); }, kWarmUp, kReps);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << "SETUP  EncodeMatrix " << kOutChannels << "x" << kColumns << ": "
            << encode_ms << " ms" << std::endl;
  std::cout << "ONLINE Multiply (MLWE, N'=" << kSmallDegree << ", rank " << rank
            << "): " << mult_ms << " ms" << std::endl;
  std::cout << "one component of " << k << "; a full channel set is "
            << k << "x that = " << mult_ms * k << " ms for "
            << kOutChannels << " outputs" << std::endl;

  const double per_out = mult_ms * k / kOutChannels;
  std::cout << "scaled linearly in output width (a projection, not a "
               "measurement), against [SYLPH] table 6 on one GPU:"
            << std::endl;
  const struct {
    const char *name;
    int out;
    int sylph_ms;
  } projections[] = {{"QKV  4096->6144", 6144, 326},
                     {"O    4096->4096", 4096, 315},
                     {"gate/up 4096->28672", 28672, 1040}};
  for (const auto &p : projections) {
    std::cout << "  " << p.name << "  ~" << per_out * p.out
              << " ms  ([SYLPH] " << p.sylph_ms << " ms)" << std::endl;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
