// Does a full Context stand up at ring degree 4096?
//
// SmallRingNttTest established that the transform pair is invertible and
// non-degenerate from log_degree 12 to 16. That is a necessary condition and
// nothing more: it exercised NTTHandler alone, at one limb, with no Context,
// no encoder and no key material. This file asks the next question, which is
// the one the pipeline actually depends on -- whether
// parameters/ringdegree12_28.json produces a working CKKS instance.
//
// The parameter set is [KANG] table 2's S12 shape, which is what [SYLPH]
// table 4 needs for the batch CC-MM (ring degree 4096, SinC) and PCMv, and
// which is also the parent ring of the degree-256 MLWE view that carries the
// Bae PC-MM. Its derived quantities are asserted below against S12 so that a
// silent edit to the JSON shows up here rather than as noise in a matrix
// product three layers up.
//
// SEPARATE BINARY, DELIBERATELY. __cm_log_degree is uploaded once per process
// behind a class-level `static inline bool cm_populated_`, so a degree-4096
// parameter set appended to any existing suite would run *at 65536* -- the
// suites instantiate bootparam_* first -- and would pass while measuring
// nothing. SmallRingNttTest.cpp documents the trap; this file avoids it by
// instantiating on one parameter file and linking its own executable.

#undef ENABLE_EXTENSION

#include <cmath>

#include "Testbed.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"

using word = uint32_t;

namespace {

// Largest absolute deviation, and the mean, over a decoded message.
struct Error {
  double max_abs;
  double mean_abs;
};

Error MeasureError(const std::vector<Complex> &got,
                   const std::vector<Complex> &want) {
  double max_abs = 0.0, sum = 0.0;
  const int n = static_cast<int>(want.size());
  for (int i = 0; i < n; i++) {
    const double d = std::abs(got[i] - want[i]);
    max_abs = std::max(max_abs, d);
    sum += d;
  }
  return {max_abs, sum / n};
}

}  // namespace

// The JSON must keep describing S12. These are cheap and they pin the file.
TEST_P(Testbed32, S12ParameterDerivations) {
  EXPECT_EQ(log_degree_, 12);
  EXPECT_EQ(param_->degree_, 4096);

  // L = |main| + |ter| = 3, alpha = |aux| = 2, so dnum = ceil(3/2) = 2, which
  // is S12's dnum. Nothing here is configured directly; all three are derived
  // in Parameter's initialiser list, so this is a real check of the JSON.
  EXPECT_EQ(param_->L_, 3);
  EXPECT_EQ(GetAlpha(), 2);
  EXPECT_EQ(GetDnum(), 2);

  // Exactly one multiplicative level. Two would put log2 QP at 132 and the
  // ring below 128-bit security, which is why [KANG] S12 is "28 x 1" and why
  // Sylph performs a single product down here before going back up.
  EXPECT_EQ(param_->max_level_, 1);
  EXPECT_EQ(default_encryption_level_, 1);

  // The scale is a single 28-bit prime -- no grafting is needed at this ring,
  // because 28 is below the 31-bit cap that forces the big ring's ratios.
  EXPECT_EQ(param_->ter_primes_.size(), 0u);
  const double ratio = param_->GetRescalePrimeProd(1);
  EXPECT_NEAR(std::log2(ratio), 28.0, 0.01);
  EXPECT_NEAR(std::log2(param_->GetScale(0)), 28.0, 0.01);
  EXPECT_NEAR(std::log2(param_->GetScale(1)), 28.0, 0.01);

  std::cout << "degree " << param_->degree_ << ", L " << param_->L_
            << ", alpha " << GetAlpha() << ", dnum " << GetDnum()
            << ", max_level " << param_->max_level_ << ", rescale ratio 2^"
            << std::log2(ratio) << std::endl;
}

// Encoder only: SpecialIFFT / SpecialFFT at 2048 slots. This is a different
// code path from the NTT and has never been run below degree 65536.
TEST_P(Testbed32, SlotEncodeDecodeRoundTrip) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg);
    ASSERT_EQ(static_cast<int>(msg.size()), param_->degree_ / 2);

    Plaintext<word> ptxt;
    Encode(ptxt, msg, level);
    std::vector<Complex> got;
    Decode(got, ptxt);

    const Error e = MeasureError(got, msg);
    std::cout << "level " << level << " encode/decode: max " << e.max_abs
              << ", mean " << e.mean_abs << std::endl;
    ASSERT_LT(e.max_abs, 1e-5) << "slot encoding is lossy at level " << level;
  }
}

// The full path, including key generation and the secret at this degree.
TEST_P(Testbed32, EncryptDecryptRoundTrip) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg);

    Ciphertext<word> ct;
    EncodeAndEncrypt(ct, msg, level);
    std::vector<Complex> got;
    DecryptAndDecode(got, ct);

    const Error e = MeasureError(got, msg);
    std::cout << "level " << level << " encrypt/decrypt: max " << e.max_abs
              << ", mean " << e.mean_abs << std::endl;
    ASSERT_LT(e.max_abs, 1e-4) << "encryption is lossy at level " << level;
  }
}

// Coefficient encoding, which is what both matrix products are defined on.
TEST_P(Testbed32, CoeffEncodeRoundTrip) {
  const int degree = param_->degree_;
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

    Plaintext<word> ptxt;
    context_->encoder_.EncodeCoeff(ptxt, level, DetermineScale(level), coeffs);
    std::vector<double> got;
    context_->encoder_.DecodeCoeff(got, ptxt);

    ASSERT_EQ(static_cast<int>(got.size()), degree);
    double max_abs = 0.0;
    for (int i = 0; i < degree; i++) {
      max_abs = std::max(max_abs, std::abs(got[i] - coeffs[i]));
    }
    std::cout << "level " << level << " coeff encode/decode: max " << max_abs
              << std::endl;
    ASSERT_LT(max_abs, 1e-5) << "coefficient encoding is lossy at level "
                             << level;
  }
}

// The one multiplicative level this ring has: multiply at level 1 by a
// plaintext, rescale down to level 0, and confirm the scale lands back on the
// canonical value rather than drifting.
TEST_P(Testbed32, PlaintextMultThenRescale) {
  std::vector<Complex> a, b;
  GenerateRandomMessage(a);
  GenerateRandomMessage(b);

  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, a, 1);
  Plaintext<word> pt;
  Encode(pt, b, 1);

  Ciphertext<word> prod, res;
  context_->Mult(prod, ct, pt);
  context_->Rescale(res, prod);

  ASSERT_EQ(param_->NPToLevel(res.GetNP()), 0);
  // Parameter chooses scale_1 so that scale_1^2 / ratio_1 == scale_0 exactly;
  // a drift here means the JSON's scale and its prime disagree.
  EXPECT_NEAR(res.GetScale() / param_->GetScale(0), 1.0, 1e-6);

  std::vector<Complex> got;
  DecryptAndDecode(got, res);

  std::vector<Complex> want(a.size());
  for (size_t i = 0; i < a.size(); i++) want[i] = a[i] * b[i];

  const Error e = MeasureError(got, want);
  std::cout << "mult+rescale: level " << param_->NPToLevel(res.GetNP())
            << ", scale 2^" << std::log2(res.GetScale()) << ", max "
            << e.max_abs << ", mean " << e.mean_abs << std::endl;
  ASSERT_LT(e.max_abs, 1e-3);
}

// ModDecomp at the shape Sylph actually uses: 4096 -> 256, so rank 16. The
// existing MlweTest only reaches 4096 and 16384 from the 65536 ring, which is
// a different rank and a different limb count.
TEST_P(Testbed32, ModDecompToSylphDegree256) {
  const int degree = param_->degree_;
  const int small_degree = 256;
  const int rank = degree / small_degree;
  ASSERT_EQ(rank, 16);

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  const int level = 0;
  std::vector<double> coeffs(degree);
  Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  const NPInfo np = ct.GetNP();
  const auto primes = param_->GetPrimeVector(np);
  const int num_total = np.GetNumTotal();

  HostVector<word> host_a, host_b;
  mlwe.ComponentToHostCoeffs(host_a, ct.AxConstView(), np);
  mlwe.ComponentToHostCoeffs(host_b, ct.BxConstView(), np);

  std::vector<MlweCiphertext<word>> parts;
  mlwe.ModDecomp(parts, ct, small_degree);
  ASSERT_EQ(static_cast<int>(parts.size()), rank);

  long long mismatches = 0;
  std::string first;
  for (int i = 0; i < rank; i++) {
    ASSERT_EQ(parts[i].rank_, rank);
    ASSERT_EQ(parts[i].degree_, small_degree);

    HostVector<word> dev_a, dev_b;
    CopyDeviceToHost(dev_a, parts[i].a_);
    CopyDeviceToHost(dev_b, parts[i].b_);

    for (int limb = 0; limb < num_total; limb++) {
      const uint64_t p = primes[limb];
      const word *a_src = host_a.data() + limb * degree;
      const word *b_src = host_b.data() + limb * degree;

      for (int s = 0; s < small_degree; s++) {
        if (b_src[i + rank * s] % p != dev_b[limb * small_degree + s] % p) {
          if (!mismatches) first = "b i=" + std::to_string(i);
          mismatches++;
        }
      }
      for (int j = 0; j < rank; j++) {
        for (int s = 0; s < small_degree; s++) {
          uint64_t want;
          if (j <= i) {
            want = a_src[(i - j) + rank * s];
          } else {
            const int l = i - j + rank;
            want = (s == 0) ? (p - a_src[l + rank * (small_degree - 1)] % p) % p
                            : a_src[l + rank * (s - 1)];
          }
          if (want % p != dev_a[(limb * rank + j) * small_degree + s] % p) {
            if (!mismatches) {
              first = "a i=" + std::to_string(i) + " j=" + std::to_string(j);
            }
            mismatches++;
          }
        }
      }
    }
  }
  std::cout << "ModDecomp 4096 -> 256, rank " << rank << ", " << num_total
            << " limbs: " << mismatches << " mismatches" << std::endl;
  ASSERT_EQ(mismatches, 0) << "first at " << first;
}

// The MLWE PC-MM, checked by commutation rather than against a hand-rolled
// Montgomery reference.
//
// ModDecomp is a re-indexing of coefficients and an INTT; the PC-MM is a scalar
// linear combination. Both are linear, so the two orders must agree:
//
//     ModDecomp( sum_j u_j * ct_j )^(i)  ==  sum_j u_j * ModDecomp(ct_j)^(i)
//
// The left side is the already-validated RLWE overload followed by ModDecomp;
// the right side is the new MLWE overload. Any Montgomery or domain convention
// appears on both sides and cancels, which is precisely what makes this a
// stronger check than reimplementing the convention in the test and agreeing
// with myself.
//
// Comparison is mod p, not word-for-word: the two orders are equal as residues
// but need not produce identical representatives if any stage reduces lazily.
TEST_P(Testbed32, MlwePcmmCommutesWithModDecomp) {
  const int degree = param_->degree_;
  const int small_degree = 256;
  const int rank = degree / small_degree;
  const int rows = 3, cols = 4;
  const int level = 0;

  PcmmHandler<word> pcmm(*param_);
  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  std::vector<Ciphertext<word>> cts(cols);
  for (int j = 0; j < cols; j++) {
    std::vector<double> coeffs(degree);
    Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);
    Plaintext<word> pt;
    context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), coeffs);
    interface_->Encrypt(cts[j], pt);
  }

  std::vector<double> u_values(rows * cols);
  Random::SampleUniformReal(u_values.data(), rows * cols, -1.0, 1.0);
  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, level, 1024.0, u_values, rows, cols);

  // Left side: RLWE product, then decompose each row.
  std::vector<Ciphertext<word>> rlwe_res;
  pcmm.Multiply(rlwe_res, u, cts);
  ASSERT_EQ(static_cast<int>(rlwe_res.size()), rows);
  std::vector<std::vector<MlweCiphertext<word>>> expected(rows);
  for (int r = 0; r < rows; r++) {
    mlwe.ModDecomp(expected[r], rlwe_res[r], small_degree);
  }

  // Right side: decompose each input, then take the product per index i.
  std::vector<std::vector<MlweCiphertext<word>>> parts(cols);
  for (int j = 0; j < cols; j++) {
    mlwe.ModDecomp(parts[j], cts[j], small_degree);
  }

  const NPInfo np = cts[0].GetNP();
  const auto primes = param_->GetPrimeVector(np);
  const int num_total = np.GetNumTotal();
  const int a_stride = rank * small_degree;

  long long mismatches = 0;
  std::string first;
  for (int i = 0; i < rank; i++) {
    // Each part is consumed exactly once, so moving is safe and avoids a copy
    // that MlweCiphertext deliberately does not offer.
    std::vector<MlweCiphertext<word>> column;
    column.reserve(cols);
    for (int j = 0; j < cols; j++) column.push_back(std::move(parts[j][i]));

    std::vector<MlweCiphertext<word>> got;
    pcmm.Multiply(got, u, column);
    ASSERT_EQ(static_cast<int>(got.size()), rows);

    for (int r = 0; r < rows; r++) {
      ASSERT_EQ(got[r].rank_, rank);
      ASSERT_EQ(got[r].degree_, small_degree);
      EXPECT_NEAR(got[r].scale_ / expected[r][i].scale_, 1.0, 1e-9);

      HostVector<word> ga, gb, ea, eb;
      CopyDeviceToHost(ga, got[r].a_);
      CopyDeviceToHost(gb, got[r].b_);
      CopyDeviceToHost(ea, expected[r][i].a_);
      CopyDeviceToHost(eb, expected[r][i].b_);

      for (int limb = 0; limb < num_total; limb++) {
        const uint64_t p = primes[limb];
        for (int s = 0; s < small_degree; s++) {
          const int idx = limb * small_degree + s;
          if (gb[idx] % p != eb[idx] % p) {
            if (!mismatches) {
              first = "b i=" + std::to_string(i) + " r=" + std::to_string(r) +
                      " limb=" + std::to_string(limb) +
                      " s=" + std::to_string(s);
            }
            mismatches++;
          }
        }
        for (int t = 0; t < a_stride; t++) {
          const int idx = limb * a_stride + t;
          if (ga[idx] % p != ea[idx] % p) {
            if (!mismatches) {
              first = "a i=" + std::to_string(i) + " r=" + std::to_string(r) +
                      " limb=" + std::to_string(limb) +
                      " t=" + std::to_string(t);
            }
            mismatches++;
          }
        }
      }
    }
  }

  std::cout << "MLWE PC-MM " << rows << "x" << cols << " at degree "
            << small_degree << ", rank " << rank << ", " << num_total
            << " limbs: " << mismatches << " mismatches" << std::endl;
  ASSERT_EQ(mismatches, 0) << "first at " << first;
}

INSTANTIATE_TEST_SUITE_P(
    SmallRing, Testbed32,
    testing::Values("ringdegree12_28.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
