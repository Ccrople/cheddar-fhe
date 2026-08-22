// The cuBLAS PCMM against the hand kernel, which is the reference.
//
// PcmmHandler::Multiply is correct -- pcmm_test and llama_pcmm_test cover it --
// so the only acceptable result here is *equality*, not closeness. Both paths
// write plain residues: PcmmAccum multiplies a Montgomery-form U so its output
// is plain, and the GEMM path multiplies plain residues directly. That makes a
// bit-for-bit comparison meaningful, and anything less would hide a carry or a
// reduction bug behind CKKS noise.
//
// This runs at ringdegree12_30, where PCMM actually runs ([SYLPH] table 4 puts
// it at a small ring in coefficient encoding); at 2^16 the inputs alone exceed
// the card.

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "Testbed.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#ifdef USE_CUBLAS
#include "core/PcmmBlas.h"
#endif

using word = uint32_t;

#ifdef USE_CUBLAS
TEST_P(Testbed32, PcmmBlasMatchesTheHandKernel) {
  constexpr int kTokens = 128;
  constexpr int kCols = 1024;
  constexpr int kRows = 64;
  constexpr double kWeightScale = 1073741824.0;

  const int degree = param_->degree_;
  const int level = default_encryption_level_;
  const double ct_scale = DetermineScale(level);

  PcmmHandler<word> pcmm(*param_);
  PcmmBlasHandler<word> blas(*param_);

  std::vector<Ciphertext<word>> cts(kCols);
  {
    std::vector<double> coeffs(degree, 0.0);
    Plaintext<word> pt;
    for (int c = 0; c < kCols; c++) {
      for (int t = 0; t < kTokens; t++) {
        coeffs[t] = 0.013 * ((c * 31 + t * 7) % 211) - 1.3;
      }
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(cts[c], pt);
    }
  }

  std::vector<double> values(static_cast<size_t>(kRows) * kCols);
  for (size_t i = 0; i < values.size(); i++) {
    values[i] = 0.002 * static_cast<double>((i * 7919) % 401) - 0.4;
  }

  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, level, kWeightScale, values, kRows, kCols);
  typename PcmmBlasHandler<word>::SplitMatrix us;
  blas.SplitMatrixFrom(us, level, kWeightScale, values, kRows, kCols);
  std::cout << "split into " << us.pieces << " int8 pieces per residue, "
            << PcmmBlasHandler<word>::SplitBytes(us) / 1e6 << " MB" << std::endl;

  std::vector<Ciphertext<word>> want, got;
  pcmm.Multiply(want, u, cts);
  blas.Multiply(got, us, cts);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(want.size(), got.size());

  const int primes = param_->LevelToNP(level).GetNumTotal();
  const size_t n = static_cast<size_t>(primes) * degree;
  size_t mismatches = 0;
  for (int i = 0; i < kRows; i++) {
    HostVector<word> hb(n), gb(n), ha(n), ga(n);
    CopyDeviceToHost(hb, want[i].bx_);
    CopyDeviceToHost(gb, got[i].bx_);
    CopyDeviceToHost(ha, want[i].ax_);
    CopyDeviceToHost(ga, got[i].ax_);
    for (size_t k = 0; k < n; k++) {
      if (hb[k] != gb[k] || ha[k] != ga[k]) mismatches++;
    }
  }
  std::cout << "compared " << 2 * kRows * n << " words" << std::endl;
  EXPECT_EQ(mismatches, 0u)
      << "the GEMM path disagrees with PcmmAccum; both write plain residues, so "
         "this must be exact";

  EXPECT_NEAR(got[0].GetScale() / want[0].GetScale(), 1.0, 1e-12);
}

// Equality established above, so the timing means something. Same shape as
// PcmmProfileTest's, so the two numbers are directly comparable.
TEST_P(Testbed32, PcmmBlasIsFasterThanTheHandKernel) {
  constexpr int kTokens = 128;
  constexpr int kCols = 4096;
  constexpr int kRows = 128;
  constexpr double kWeightScale = 1073741824.0;
  constexpr int kWarm = 2, kReps = 5;

  const int degree = param_->degree_;
  const int level = default_encryption_level_;
  const double ct_scale = DetermineScale(level);

  PcmmHandler<word> pcmm(*param_);
  PcmmBlasHandler<word> blas(*param_);

  std::vector<Ciphertext<word>> cts(kCols);
  {
    std::vector<double> coeffs(degree, 0.0);
    Plaintext<word> pt;
    for (int c = 0; c < kCols; c++) {
      for (int t = 0; t < kTokens; t++) coeffs[t] = 0.01 * ((c + t) % 97);
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(cts[c], pt);
    }
  }
  std::vector<double> values(static_cast<size_t>(kRows) * kCols);
  for (size_t i = 0; i < values.size(); i++) {
    values[i] = 0.001 * static_cast<double>((i * 7919) % 211);
  }

  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, level, kWeightScale, values, kRows, kCols);
  typename PcmmBlasHandler<word>::SplitMatrix us;
  blas.SplitMatrixFrom(us, level, kWeightScale, values, kRows, kCols);

  auto time_ms = [&](auto &&f) {
    for (int i = 0; i < kWarm; i++) f();
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kReps; i++) f();
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / kReps;
  };

  std::vector<Ciphertext<word>> r1, r2;
  const double hand = time_ms([&] { pcmm.Multiply(r1, u, cts); });
  const double gemm = time_ms([&] { blas.Multiply(r2, us, cts); });
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  std::cout << kRows << "x" << kCols << " by " << kTokens << " tokens:"
            << std::endl;
  std::cout << "  PcmmAccum (hand kernel) " << hand << " ms" << std::endl;
  std::cout << "  cuBLAS int8             " << gemm << " ms   "
            << (hand / gemm) << "x" << std::endl;
  std::cout << "scaled to the projections, against [SYLPH] table 6 on 1 GPU:"
            << std::endl;
  const struct { const char *n; int out; int sylph; } proj[] = {
      {"QKV  4096->6144", 6144, 326},
      {"O    4096->4096", 4096, 315},
      {"gate/up 4096->28672", 28672, 1040}};
  for (const auto &p : proj) {
    std::cout << "  " << p.n << "  " << gemm * p.out / kRows << " ms  ([SYLPH] "
              << p.sylph << ")" << std::endl;
  }
}
#endif

#ifdef USE_CUBLAS
// The MLWE overload, which is the one that matters.
//
// `CoeffLinearLeg` reaches the product through `ModDecomp`, so every PC-MM the
// Llama block performs is on `MlweCiphertext`, not `Ciphertext` -- the RLWE
// overload the two tests above cover was never on the layer's path. Same
// standard: `PcmmHandler::Multiply(MLWE)` is the reference and the only
// acceptable result is equality.
TEST_P(Testbed32, PcmmBlasMlweMatchesTheHandKernel) {
  constexpr int kSmallDegree = 256;
  constexpr int kParents = 4;
  constexpr double kWeightScale = 1073741824.0;

  const int degree = param_->degree_;
  const int level = default_encryption_level_;
  const double ct_scale = DetermineScale(level);
  const int rank = degree / kSmallDegree;
  const int cols = kParents * rank;
  const int rows = rank;

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);
  PcmmHandler<word> pcmm(*param_);
  PcmmBlasHandler<word> blas(*param_);

  std::vector<MlweCiphertext<word>> columns;
  {
    std::vector<double> coeffs(degree, 0.0);
    Plaintext<word> pt;
    Ciphertext<word> ct;
    for (int p = 0; p < kParents; p++) {
      for (int i = 0; i < degree; i++) {
        coeffs[i] = 0.011 * static_cast<double>((p * 37 + i * 13) % 197) - 1.1;
      }
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(ct, pt);
      std::vector<MlweCiphertext<word>> decomposed;
      mlwe.ModDecomp(decomposed, ct, kSmallDegree);
      for (auto &c : decomposed) columns.push_back(std::move(c));
    }
  }
  ASSERT_EQ(static_cast<int>(columns.size()), cols);

  std::vector<double> values(static_cast<size_t>(rows) * cols);
  for (size_t i = 0; i < values.size(); i++) {
    values[i] = 0.002 * static_cast<double>((i * 7919) % 401) - 0.4;
  }

  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, level, kWeightScale, values, rows, cols);
  typename PcmmBlasHandler<word>::SplitMatrix us;
  blas.SplitMatrixFrom(us, level, kWeightScale, values, rows, cols);

  std::vector<MlweCiphertext<word>> want, got;
  pcmm.Multiply(want, u, columns);
  blas.Multiply(got, us, columns);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(want.size(), got.size());

  size_t mismatches = 0, words = 0;
  for (int i = 0; i < rows; i++) {
    ASSERT_EQ(want[i].a_.size(), got[i].a_.size());
    ASSERT_EQ(want[i].b_.size(), got[i].b_.size());
    HostVector<word> ha(want[i].a_.size()), ga(got[i].a_.size());
    HostVector<word> hb(want[i].b_.size()), gb(got[i].b_.size());
    CopyDeviceToHost(ha, want[i].a_);
    CopyDeviceToHost(ga, got[i].a_);
    CopyDeviceToHost(hb, want[i].b_);
    CopyDeviceToHost(gb, got[i].b_);
    for (size_t k = 0; k < ha.size(); k++) {
      if (ha[k] != ga[k]) mismatches++;
    }
    for (size_t k = 0; k < hb.size(); k++) {
      if (hb[k] != gb[k]) mismatches++;
    }
    words += ha.size() + hb.size();
  }
  std::cout << "rank " << rank << ", " << cols << " columns, " << us.pieces
            << " int8 pieces; compared " << words << " words" << std::endl;
  EXPECT_EQ(mismatches, 0u)
      << "the GEMM path disagrees with PcmmAccum on the MLWE format";
  EXPECT_NEAR(got[0].scale_ / want[0].scale_, 1.0, 1e-12);
}

// The layer prepares the split once per tile and then multiplies it by one
// plaintext matrix per ModPack group -- fifty-six of them for gate and up. That
// reuse is the whole point of `PrepareSource`, and it is also the one thing the
// single-product test above cannot see: a product that left state behind in the
// split, or a second matrix that quietly received the first one's answer, would
// pass there and be wrong here.
TEST_P(Testbed32, PcmmBlasMlweReusesOneSplitSource) {
  constexpr int kSmallDegree = 256;
  constexpr int kParents = 4;
  constexpr int kMatrices = 3;
  constexpr double kWeightScale = 1073741824.0;

  const int degree = param_->degree_;
  const int level = default_encryption_level_;
  const double ct_scale = DetermineScale(level);
  const int rank = degree / kSmallDegree;
  const int cols = kParents * rank;
  const int rows = rank;

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);
  PcmmHandler<word> pcmm(*param_);
  PcmmBlasHandler<word> blas(*param_);

  std::vector<MlweCiphertext<word>> columns;
  {
    std::vector<double> coeffs(degree, 0.0);
    Plaintext<word> pt;
    Ciphertext<word> ct;
    for (int p = 0; p < kParents; p++) {
      for (int i = 0; i < degree; i++) {
        coeffs[i] = 0.009 * static_cast<double>((p * 53 + i * 29) % 211) - 0.9;
      }
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(ct, pt);
      std::vector<MlweCiphertext<word>> decomposed;
      mlwe.ModDecomp(decomposed, ct, kSmallDegree);
      for (auto &c : decomposed) columns.push_back(std::move(c));
    }
  }
  ASSERT_EQ(static_cast<int>(columns.size()), cols);

  // Three unrelated plaintext matrices, standing in for three ModPack groups
  // of one tile, and the reference product for each -- taken before the split
  // exists, because the reference reads the ciphertexts and the point of the
  // test is that the GEMM path no longer does.
  std::vector<typename PcmmBlasHandler<word>::SplitMatrix> us(kMatrices);
  std::vector<std::vector<MlweCiphertext<word>>> want(kMatrices);
  for (int m = 0; m < kMatrices; m++) {
    std::vector<double> values(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < values.size(); i++) {
      values[i] =
          0.002 * static_cast<double>((i * 7919 + m * 1237) % 401) - 0.4;
    }
    PlainMatrix<word> u;
    pcmm.EncodeMatrix(u, level, kWeightScale, values, rows, cols);
    pcmm.Multiply(want[m], u, columns);
    blas.SplitMatrixFrom(us[m], level, kWeightScale, values, rows, cols);
  }

  typename PcmmBlasHandler<word>::SplitSource prepared;
  blas.PrepareSource(prepared, us[0], columns);
  // What `CoeffLinearLeg::Project` does next: the ciphertexts the split came
  // from are dropped, so every product below reads only the split.
  columns.clear();
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  size_t mismatches = 0, words = 0;
  for (int m = 0; m < kMatrices; m++) {
    std::vector<MlweCiphertext<word>> got;
    blas.Multiply(got, us[m], prepared);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(want[m].size(), got.size());
    for (int i = 0; i < rows; i++) {
      ASSERT_EQ(want[m][i].a_.size(), got[i].a_.size());
      ASSERT_EQ(want[m][i].b_.size(), got[i].b_.size());
      HostVector<word> ha(want[m][i].a_.size()), ga(got[i].a_.size());
      HostVector<word> hb(want[m][i].b_.size()), gb(got[i].b_.size());
      CopyDeviceToHost(ha, want[m][i].a_);
      CopyDeviceToHost(ga, got[i].a_);
      CopyDeviceToHost(hb, want[m][i].b_);
      CopyDeviceToHost(gb, got[i].b_);
      for (size_t k = 0; k < ha.size(); k++) {
        if (ha[k] != ga[k]) mismatches++;
      }
      for (size_t k = 0; k < hb.size(); k++) {
        if (hb[k] != gb[k]) mismatches++;
      }
      words += ha.size() + hb.size();
    }
  }
  std::cout << kMatrices << " matrices against one prepared split; compared "
            << words << " words" << std::endl;
  EXPECT_EQ(mismatches, 0u)
      << "a reused split source does not give the same product as PcmmAccum";
}
#endif

INSTANTIATE_TEST_SUITE_P(
    SmallRing, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
