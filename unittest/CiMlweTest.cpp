// The RLWE-to-MLWE decomposition on the conjugate-invariant ring.
//
// On R+ = Z[Y + Y^-1] the module structure over the rank-N' subring is not a
// stride interleave: the subring is spanned by {1, c_k, c_2k, ...} and R+ is
// free over it on {1, c_1, ..., c_{k-1}}, but c_i c_{tk} = c_{tk+i} + c_{tk-i}
// hits two coefficient classes. ModDecomp is therefore the alternating-sign
// suffix-sum scan of Doing.md 1.5ba, and the a-part mixes components with
// subring coefficients {1, -1, 2, c_k} rather than shifting slices.
//
// Checked the way MlweTest checks the power-basis form, from two directions,
// plus the round trip:
//
//   CiModDecompIdentityHolds -- host-only, on a tiny synthetic R+ mod a small
//       prime: decomposing a and sk, forming the a~ combination, multiplying
//       against the decomposed secret in the subring and recomposing must
//       reproduce a * sk under the ring's own product
//       c_j c_k = c_{j+k} + c_{|j-k|}. Exhaustive over every (i, j) pair at
//       N = 16..64, so every k-crossing, both mirror classes and the self-
//       mirror class are hit with random values.
//
//   CiModDecompMatchesHostIndexing -- the device kernels at the real ring
//       degree, entry by entry against the same formulas evaluated on the
//       host from the inverse-transformed ciphertext components.
//
//   ModPackInvertsModDecomp -- decompose, pack with the embedded-secret
//       switching keys, decrypt. This is the round trip through the real
//       key-switch machinery, and it is also the first conjugate-invariant
//       caller of the coefficient-path mod-up (ModUpFromCoeffBatch on
//       ci16_35, where beta = 1, and the per-ciphertext ModUpFromCoeff on
//       ci12_30, where alpha = 1 forces beta > 1). Its body is ring-agnostic,
//       so ringdegree12_30 runs it too as the control that the dispatch
//       leaves the ordinary ring alone.
//
//   PcmmOnComponentsMatchesHostMap -- the product itself between the two:
//       ModDecomp, the scalar PC-MM across the module components, ModPack,
//       decrypt, against the same map in exact arithmetic on the host.
//
//   PcmmBlasMatchesTheHandKernelOnComponents (USE_CUBLAS) -- the int8
//       tensor-core route against the hand kernel, bit for bit, at the
//       projection's own small degree of 256. On ci16_35 that is rank 256
//       out of degree 65536 -- the real configuration's first device run,
//       which the indexing test above also spot-checks semantically.

#undef ENABLE_EXTENSION

#include "Testbed.h"
#include "core/Mlwe.h"
#include "core/Pcmm.h"
#ifdef USE_CUBLAS
#include "core/PcmmBlas.h"
#endif

using word = uint32_t;

namespace {

// --- host reference, in the c-basis and mod a prime -------------------------

using IPoly = std::vector<uint64_t>;

// Multiplication in R+ of rank `degree` mod p, in the basis
// {1, c_1, ..., c_{degree-1}}. Index 0 is the coefficient of 1, not of
// c_0 = 2, which is why a term landing at index 0 counts twice. The same
// routine at degree N' is the subring product: {1, c_k, c_2k, ...} has the
// same structure constants and the same folds (c_{N'k} = c_N = 0).
IPoly CiMulMod(const IPoly &a, const IPoly &b, uint64_t p) {
  const int degree = static_cast<int>(a.size());
  IPoly res(degree, 0);
  auto add = [&res, degree, p](int m, uint64_t v) {
    if (m == degree) return;  // c_N = 0
    if (m > degree) {         // c_m = -c_{2N-m}
      m = 2 * degree - m;
      v = (p - v) % p;
    }
    if (m == 0) v = 2 * v % p;  // c_0 = 2
    res[m] = (res[m] + v) % p;
  };

  res[0] = a[0] * b[0] % p;
  for (int k = 1; k < degree; k++) res[k] = a[0] * b[k] % p;
  for (int j = 1; j < degree; j++) res[j] = (res[j] + a[j] * b[0]) % p;
  for (int j = 1; j < degree; j++) {
    if (a[j] == 0) continue;
    for (int k = 1; k < degree; k++) {
      if (b[k] == 0) continue;
      const uint64_t v = a[j] * b[k] % p;
      add(j + k, v);
      add(j > k ? j - k : k - j, v);
    }
  }
  return res;
}

// The module decomposition: alpha_i[t] = a[tk + i] - alpha_{k-i}[t+1], with
// the i = 0 class a pure gather. Same zigzag as the device kernel.
std::vector<IPoly> CiDecompose(const IPoly &a, int rank, int small_degree,
                               uint64_t p) {
  std::vector<IPoly> alpha(rank, IPoly(small_degree, 0));
  for (int t = 0; t < small_degree; t++) alpha[0][t] = a[t * rank];
  for (int i = 1; i <= rank / 2; i++) {
    const int mi = rank - i;
    uint64_t acc_i = 0;
    uint64_t acc_m = 0;
    for (int t = small_degree - 1; t >= 0; t--) {
      const uint64_t new_i = (a[t * rank + i] + p - acc_m) % p;
      const uint64_t new_m = (a[t * rank + mi] + p - acc_i) % p;
      alpha[i][t] = new_i;
      alpha[mi][t] = new_m;  // mi == i rewrites the same value
      acc_i = new_i;
      acc_m = new_m;
    }
  }
  return alpha;
}

// The inverse: out[tk + i] = q_i[t] + q_{k-i}[t+1], q_.[N'] = 0.
IPoly CiRecompose(const std::vector<IPoly> &q, int rank, int small_degree,
                  uint64_t p) {
  IPoly out(rank * small_degree, 0);
  for (int t = 0; t < small_degree; t++) {
    for (int i = 0; i < rank; i++) {
      uint64_t v = q[i][t];
      if (i != 0 && t + 1 < small_degree) v = (v + q[rank - i][t + 1]) % p;
      out[t * rank + i] = v;
    }
  }
  return out;
}

// (c'_1 * q)[t] = q[t-1] + q[t+1], with q[-1] = q[1] and q[N'] = 0.
IPoly SubringShift(const IPoly &q, uint64_t p) {
  const int n = static_cast<int>(q.size());
  IPoly res(n, 0);
  for (int t = 0; t < n; t++) {
    const uint64_t lo = q[t == 0 ? 1 : t - 1];
    const uint64_t hi = (t + 1 < n) ? q[t + 1] : 0;
    res[t] = (lo + hi) % p;
  }
  return res;
}

// a~_l[j] as a function of a's module components -- the formula the
// CiModDecompCombine kernel implements.
IPoly ATildeCi(const std::vector<IPoly> &alpha, int l, int j, int rank,
               uint64_t p) {
  const int small_degree = static_cast<int>(alpha[0].size());
  if (j == 0) return alpha[l];
  IPoly res(small_degree, 0);
  if (l == 0) {
    const IPoly shift = SubringShift(alpha[rank - j], p);
    for (int t = 0; t < small_degree; t++) {
      res[t] = (2 * alpha[j][t] + shift[t]) % p;
    }
    return res;
  }
  const int d = (l > j) ? (l - j) : (j - l);
  res = alpha[d];
  if (l + j < rank) {
    for (int t = 0; t < small_degree; t++) {
      res[t] = (res[t] + alpha[l + j][t]) % p;
    }
  }
  if (j > l) {
    const IPoly shift = SubringShift(alpha[rank - (j - l)], p);
    for (int t = 0; t < small_degree; t++) {
      res[t] = (res[t] + shift[t]) % p;
    }
  }
  if (l + j > rank) {
    for (int t = 0; t < small_degree; t++) {
      res[t] = (res[t] + p - alpha[2 * rank - l - j][t]) % p;
    }
  }
  return res;
}

// --- the same maps in real arithmetic, for the end-to-end product test ------

using RPoly = std::vector<double>;

std::vector<RPoly> RealDecompose(const RPoly &a, int rank, int small_degree,
                                 bool conjugate_invariant) {
  std::vector<RPoly> comp(rank, RPoly(small_degree, 0.0));
  if (!conjugate_invariant) {
    for (int i = 0; i < rank; i++) {
      for (int t = 0; t < small_degree; t++) comp[i][t] = a[t * rank + i];
    }
    return comp;
  }
  for (int t = 0; t < small_degree; t++) comp[0][t] = a[t * rank];
  for (int i = 1; i <= rank / 2; i++) {
    const int mi = rank - i;
    double acc_i = 0.0;
    double acc_m = 0.0;
    for (int t = small_degree - 1; t >= 0; t--) {
      const double new_i = a[t * rank + i] - acc_m;
      const double new_m = a[t * rank + mi] - acc_i;
      comp[i][t] = new_i;
      comp[mi][t] = new_m;
      acc_i = new_i;
      acc_m = new_m;
    }
  }
  return comp;
}

RPoly RealRecompose(const std::vector<RPoly> &q, int rank, int small_degree,
                    bool conjugate_invariant) {
  RPoly out(rank * small_degree, 0.0);
  for (int t = 0; t < small_degree; t++) {
    for (int i = 0; i < rank; i++) {
      double v = q[i][t];
      if (conjugate_invariant && i != 0 && t + 1 < small_degree) {
        v += q[rank - i][t + 1];
      }
      out[t * rank + i] = v;
    }
  }
  return out;
}

}  // namespace

// The formula itself: recomposing sum_j a~_l[j] * sigma_j over the components
// l must equal a * sk under the conjugate-invariant product.
TEST(CiMlweDecomposition, CiModDecompIdentityHolds) {
  constexpr uint64_t p = 97;
  std::mt19937_64 gen(12345);
  std::uniform_int_distribution<uint64_t> dist(0, p - 1);

  for (auto shape : std::vector<std::pair<int, int>>{{16, 4}, {32, 8}, {32, 4},
                                                     {64, 8}}) {
    const int degree = shape.first;
    const int small_degree = shape.second;
    const int rank = degree / small_degree;

    IPoly a(degree), sk(degree);
    for (int i = 0; i < degree; i++) {
      a[i] = dist(gen);
      sk[i] = dist(gen);
    }

    // The scan and its inverse are a bijection on their own.
    const auto alpha = CiDecompose(a, rank, small_degree, p);
    const IPoly back = CiRecompose(alpha, rank, small_degree, p);
    for (int c = 0; c < degree; c++) {
      ASSERT_EQ(a[c], back[c]) << "scan/recompose at degree " << degree
                               << ", small_degree " << small_degree
                               << ", coefficient " << c;
    }

    const IPoly expected = CiMulMod(a, sk, p);
    const auto sigma = CiDecompose(sk, rank, small_degree, p);

    std::vector<IPoly> components(rank);
    for (int l = 0; l < rank; l++) {
      IPoly acc(small_degree, 0);
      for (int j = 0; j < rank; j++) {
        const IPoly term =
            CiMulMod(ATildeCi(alpha, l, j, rank, p), sigma[j], p);
        for (int t = 0; t < small_degree; t++) {
          acc[t] = (acc[t] + term[t]) % p;
        }
      }
      components[l] = std::move(acc);
    }
    const IPoly obtained = CiRecompose(components, rank, small_degree, p);

    for (int c = 0; c < degree; c++) {
      ASSERT_EQ(expected[c], obtained[c])
          << "degree " << degree << ", small_degree " << small_degree
          << ", coefficient " << c;
    }
  }
}

// The device kernels, against the same formulas evaluated on the host.
TEST_P(Testbed32, CiModDecompMatchesHostIndexing) {
  if (!param_->conjugate_invariant_) {
    GTEST_SKIP() << "ordinary-ring control runs the ModPack round trip only";
  }
  const int degree = 1 << log_degree_;
  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  const int level = std::min(2, param_->max_level_);
  const double scale = DetermineScale(level);
  const NPInfo np = param_->LevelToNP(level);
  const int num_total_primes = np.GetNumTotal();
  const auto primes = param_->GetPrimeVector(np);

  std::vector<double> coeffs(degree);
  Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  // The coefficient-domain components the decomposition is defined on.
  HostVector<word> host_a, host_b;
  mlwe.ComponentToHostCoeffs(host_a, ct.AxConstView(), np);
  mlwe.ComponentToHostCoeffs(host_b, ct.BxConstView(), np);

  std::vector<int> small_degrees{degree / 4, degree / 16};
  // The projection's own small degree. degree / 16 already is 256 on the
  // degree-4096 presets; on ci16_35 this adds rank 256, the real
  // configuration.
  if (degree / 16 != 256) small_degrees.push_back(256);

  for (int small_degree : small_degrees) {
    const int rank = degree / small_degree;

    // The full a~ table costs rank^2 * N' on the host, so above rank 16 the
    // comparison narrows to a spot set of components and columns that still
    // hits every branch of the formula: the pure column, the l = 0 row, both
    // mirror halves, the self-mirror class, the l+j = k boundary and the
    // l+j > k negative branch. The kernels themselves have no rank-dependent
    // branches beyond these.
    std::vector<int> spots;
    if (rank <= 16) {
      for (int i = 0; i < rank; i++) spots.push_back(i);
    } else {
      spots = {0,        1,            2,            rank / 2 - 1,
               rank / 2, rank / 2 + 1, rank - 2,     rank - 1};
    }

    std::vector<MlweCiphertext<word>> parts;
    mlwe.ModDecomp(parts, ct, small_degree);
    ASSERT_EQ(static_cast<int>(parts.size()), rank);

    std::cout << "CI ModDecomp degree " << degree << " -> " << small_degree
              << ", rank " << rank << ", " << num_total_primes << " limbs, "
              << spots.size() << " of " << rank << " components checked"
              << std::endl;

    // The scan and the a~ combination per limb, from the host coefficients.
    std::vector<std::vector<IPoly>> alpha(num_total_primes);
    std::vector<std::vector<IPoly>> beta(num_total_primes);
    for (int limb = 0; limb < num_total_primes; limb++) {
      const uint64_t p = primes[limb];
      IPoly a_limb(degree), b_limb(degree);
      for (int c = 0; c < degree; c++) {
        a_limb[c] = host_a[limb * degree + c] % p;
        b_limb[c] = host_b[limb * degree + c] % p;
      }
      alpha[limb] = CiDecompose(a_limb, rank, small_degree, p);
      beta[limb] = CiDecompose(b_limb, rank, small_degree, p);
    }

    for (int i : spots) {
      ASSERT_EQ(parts[i].rank_, rank);
      ASSERT_EQ(parts[i].degree_, small_degree);

      HostVector<word> dev_a, dev_b;
      CopyDeviceToHost(dev_a, parts[i].a_);
      CopyDeviceToHost(dev_b, parts[i].b_);

      long long mismatches = 0;
      std::string first_mismatch;

      for (int limb = 0; limb < num_total_primes; limb++) {
        const uint64_t p = primes[limb];

        for (int t = 0; t < small_degree; t++) {
          const uint64_t expected = beta[limb][i][t];
          const uint64_t obtained = dev_b[limb * small_degree + t];
          if ((expected + p - obtained % p) % p != 0) {
            if (mismatches == 0) {
              first_mismatch = "b-part at i=" + std::to_string(i) +
                               " limb=" + std::to_string(limb) +
                               " t=" + std::to_string(t) + " expected " +
                               std::to_string(expected) + " obtained " +
                               std::to_string(obtained);
            }
            mismatches++;
          }
        }

        for (int j : spots) {
          const IPoly expected_a = ATildeCi(alpha[limb], i, j, rank, p);
          for (int t = 0; t < small_degree; t++) {
            const uint64_t obtained =
                dev_a[(limb * rank + j) * small_degree + t];
            if ((expected_a[t] + p - obtained % p) % p != 0) {
              if (mismatches == 0) {
                first_mismatch = "a-part at i=" + std::to_string(i) +
                                 " j=" + std::to_string(j) +
                                 " limb=" + std::to_string(limb) +
                                 " t=" + std::to_string(t) + " expected " +
                                 std::to_string(expected_a[t]) + " obtained " +
                                 std::to_string(obtained);
              }
              mismatches++;
            }
          }
        }
      }

      ASSERT_EQ(mismatches, 0) << "small_degree " << small_degree << ", "
                               << mismatches << " mismatches, first: "
                               << first_mismatch;
    }
  }
}

// The round trip through the embedded-secret key switches. Rank 4 rather than
// MlweTest's rank 2, because rank 4 is the smallest with a cross-mirror class
// pair (1, 3): its embedded secrets and its recomposition exercise the
// two-class zigzag, where rank 2 has only the pure class and the self-mirror.
// The keys stay affordable: four of them at a low level.
TEST_P(Testbed32, ModPackInvertsModDecomp) {
  const int degree = 1 << log_degree_;
  const int rank = 4;
  const int small_degree = degree / rank;

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  const int level = std::min(2, param_->max_level_);
  interface_->PrepareModPackKeys(small_degree, level);
  std::vector<const EvaluationKey<word> *> keys;
  for (int j = 0; j < rank; j++) {
    keys.push_back(&interface_->GetModPackKey(rank, j));
  }

  std::vector<double> coeffs(degree);
  Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  std::vector<MlweCiphertext<word>> parts;
  mlwe.ModDecomp(parts, ct, small_degree);
  ASSERT_EQ(static_cast<int>(parts.size()), rank);

  Ciphertext<word> packed;
  mlwe.ModPack(context_, packed, parts, keys);
  ASSERT_EQ(param_->NPToLevel(packed.GetNP()), level);

  Plaintext<word> out;
  interface_->Decrypt(out, packed);
  std::vector<double> got;
  context_->encoder_.DecodeCoeff(got, out);
  ASSERT_EQ(static_cast<int>(got.size()), degree);

  double max_abs = 0.0;
  for (int i = 0; i < degree; i++) {
    max_abs = std::max(max_abs, std::abs(got[i] - coeffs[i]));
  }
  std::cout << (param_->conjugate_invariant_ ? "CI" : "ordinary")
            << " degree " << degree << " -> " << small_degree << ", rank "
            << rank << ", level " << level << ": max error " << max_abs
            << std::endl;
  ASSERT_LT(max_abs, 1e-3);
}

// The product between decomposition and packing: res_l = sum_j U[l][j] *
// parts[j] over the module components, then packed and decrypted, against the
// same map on the host. `PcmmHandler::Multiply` is a scalar combination and
// so basis-agnostic by construction; what this pins down on R+ is everything
// around that claim -- the container layout the accumulator assumes, the
// scale bookkeeping through EncodeMatrix, and that the mixed components are
// still valid MLWE ciphertexts for the embedded-secret switches.
//
// The weight scale is 2^25: large enough that weight rounding (about
// 2^-26 * the component magnitude, which the scan grows to a few hundred)
// stays far below the 1e-3 bound, and small enough that the unrescaled
// product scale times those components stays several bits inside the
// level-1 modulus of the degree-4096 presets.
TEST_P(Testbed32, PcmmOnComponentsMatchesHostMap) {
  const int degree = 1 << log_degree_;
  const int rank = 4;
  const int small_degree = degree / rank;
  constexpr double kWeightScale = 33554432.0;  // 2^25

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);
  PcmmHandler<word> pcmm(*param_);

  const int level = std::min(2, param_->max_level_);
  interface_->PrepareModPackKeys(small_degree, level);
  std::vector<const EvaluationKey<word> *> keys;
  for (int j = 0; j < rank; j++) {
    keys.push_back(&interface_->GetModPackKey(rank, j));
  }

  std::vector<double> coeffs(degree);
  Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);

  Plaintext<word> pt;
  context_->encoder_.EncodeCoeff(pt, level, DetermineScale(level), coeffs);
  Ciphertext<word> ct;
  interface_->Encrypt(ct, pt);

  std::vector<MlweCiphertext<word>> parts;
  mlwe.ModDecomp(parts, ct, small_degree);
  ASSERT_EQ(static_cast<int>(parts.size()), rank);

  std::vector<double> values(static_cast<size_t>(rank) * rank);
  Random::SampleUniformReal(values.data(), values.size(), -1.0, 1.0);

  PlainMatrix<word> u;
  pcmm.EncodeMatrix(u, level, kWeightScale, values, rank, rank);

  std::vector<MlweCiphertext<word>> mixed;
  pcmm.Multiply(mixed, u, parts);
  ASSERT_EQ(static_cast<int>(mixed.size()), rank);

  Ciphertext<word> packed;
  mlwe.ModPack(context_, packed, mixed, keys);
  ASSERT_EQ(param_->NPToLevel(packed.GetNP()), level);

  Plaintext<word> out;
  interface_->Decrypt(out, packed);
  std::vector<double> got;
  context_->encoder_.DecodeCoeff(got, out);
  ASSERT_EQ(static_cast<int>(got.size()), degree);

  // The same map on the host: decompose, mix the components by U, recompose.
  const bool ci = param_->conjugate_invariant_;
  const auto comp = RealDecompose(coeffs, rank, small_degree, ci);
  std::vector<RPoly> mixed_host(rank, RPoly(small_degree, 0.0));
  for (int l = 0; l < rank; l++) {
    for (int j = 0; j < rank; j++) {
      const double weight = values[l * rank + j];
      for (int t = 0; t < small_degree; t++) {
        mixed_host[l][t] += weight * comp[j][t];
      }
    }
  }
  const RPoly expected = RealRecompose(mixed_host, rank, small_degree, ci);

  double max_abs = 0.0;
  for (int i = 0; i < degree; i++) {
    max_abs = std::max(max_abs, std::abs(got[i] - expected[i]));
  }
  std::cout << (ci ? "CI" : "ordinary") << " PC-MM degree " << degree
            << " -> " << small_degree << ", rank " << rank << ", level "
            << level << ": max error " << max_abs << std::endl;
  ASSERT_LT(max_abs, 1e-3);
}

#ifdef USE_CUBLAS
// The int8 tensor-core route against the hand kernel, bit for bit -- the
// PcmmBlasTest standard, on this ring. The GEMM path splits residues into
// int8 pieces and recombines them mod p, and none of that sees the basis;
// what it must not trip over is the container the conjugate-invariant
// ModDecomp filled and this parameter set's 1-mod-4N primes. kSmallDegree is
// 256, the degree the projections actually run at, so on ci16_35 this drives
// ModDecomp and both products at rank 256 out of degree 65536 -- the real
// configuration -- with the semantic anchor carried by the indexing test's
// spot checks and by the hand kernel's own end-to-end test above.
TEST_P(Testbed32, PcmmBlasMatchesTheHandKernelOnComponents) {
  constexpr int kSmallDegree = 256;
  constexpr int kParents = 2;
  constexpr double kWeightScale = 1073741824.0;  // 2^30, as in PcmmBlasTest

  const int degree = param_->degree_;
  const int level = std::min(2, param_->max_level_);
  const double ct_scale = DetermineScale(level);
  const int rank = degree / kSmallDegree;
  const int cols = kParents * rank;
  const int rows = rank;

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);
  PcmmHandler<word> pcmm(*param_);
  PcmmBlasHandler<word> blas(*param_);

  std::vector<MlweCiphertext<word>> columns;
  {
    std::vector<double> coeffs(degree);
    Plaintext<word> pt;
    Ciphertext<word> ct;
    for (int parent = 0; parent < kParents; parent++) {
      Random::SampleUniformReal(coeffs.data(), degree, -1.0, 1.0);
      context_->encoder_.EncodeCoeff(pt, level, ct_scale, coeffs);
      interface_->Encrypt(ct, pt);
      std::vector<MlweCiphertext<word>> decomposed;
      mlwe.ModDecomp(decomposed, ct, kSmallDegree);
      for (auto &c : decomposed) columns.push_back(std::move(c));
    }
  }
  ASSERT_EQ(static_cast<int>(columns.size()), cols);

  std::vector<double> values(static_cast<size_t>(rows) * cols);
  Random::SampleUniformReal(values.data(), values.size(), -0.5, 0.5);

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
  std::cout << (param_->conjugate_invariant_ ? "CI" : "ordinary")
            << " blas-vs-hand at degree " << degree << " -> " << kSmallDegree
            << ", rank " << rank << ", " << cols << " columns, level " << level
            << ": compared " << words << " words" << std::endl;
  EXPECT_EQ(mismatches, 0u)
      << "the cuBLAS path does not reproduce PcmmAccum on this ring";
}
#endif

INSTANTIATE_TEST_SUITE_P(
    // ringdegree12_30 is the same primes and shape with the flag off -- the
    // control that the dispatch leaves the ordinary ring alone.
    Cheddar, Testbed32,
    testing::Values("ci12_30.json", "ci16_35.json", "ringdegree12_30.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
