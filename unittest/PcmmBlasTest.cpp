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
#endif

INSTANTIATE_TEST_SUITE_P(
    SmallRing, Testbed32, testing::Values("ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string p = info.param;
      std::replace(p.begin(), p.end(), '.', '_');
      return p;
    });
