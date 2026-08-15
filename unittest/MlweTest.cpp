// Tests for the RLWE-to-MLWE decomposition of Bae et al. (CRYPTO 2024),
// appendix A.
//
// The whole risk in this operation is indexing, so it is checked twice and
// from two directions:
//
//   ModDecompIdentityHolds  -- a pure host test, on a tiny synthetic ring, that
//       the paper's formula is the one being implemented: recomposing
//       sum_i <a~_i, sk'> X^i must reproduce a * sk in R_N for arbitrary a and
//       sk. Small enough (N = 16, 32) that the negacyclic product is exact and
//       exhaustive over every (i, j) pair, including the Y-wrap.
//
//   ModDecompMatchesHostIndexing -- the device kernel at the real ring degree,
//       compared entry by entry against the same formula evaluated on the host
//       from the inverse-transformed ciphertext components.
//
// Together these say the formula is right and that the kernel implements it.

#undef ENABLE_EXTENSION

#include "Testbed.h"
#include "core/Mlwe.h"

using word = uint32_t;

namespace {

// --- host reference for the tiny synthetic ring -----------------------------

using Poly = std::vector<uint64_t>;

uint64_t AddMod(uint64_t a, uint64_t b, uint64_t p) { return (a + b) % p; }
uint64_t SubMod(uint64_t a, uint64_t b, uint64_t p) { return (a + p - b) % p; }
uint64_t MulMod(uint64_t a, uint64_t b, uint64_t p) { return (a * b) % p; }

// Negacyclic product in Z_p[X]/(X^n + 1).
Poly NegacyclicMul(const Poly &a, const Poly &b, uint64_t p) {
  const int n = static_cast<int>(a.size());
  Poly res(n, 0);
  for (int i = 0; i < n; i++) {
    if (a[i] == 0) continue;
    for (int j = 0; j < n; j++) {
      const uint64_t t = MulMod(a[i], b[j], p);
      const int k = i + j;
      if (k < n) {
        res[k] = AddMod(res[k], t, p);
      } else {
        res[k - n] = SubMod(res[k - n], t, p);
      }
    }
  }
  return res;
}

// e*_l(a)[s] = a[l + k*s]
Poly Estar(const Poly &a, int l, int rank, int small_degree) {
  Poly res(small_degree);
  for (int s = 0; s < small_degree; s++) {
    res[s] = a[l + rank * s];
  }
  return res;
}

// (Y * q)[0] = -q[N'-1], (Y * q)[s] = q[s-1]
Poly MulY(const Poly &q, uint64_t p) {
  const int n = static_cast<int>(q.size());
  Poly res(n);
  res[0] = SubMod(0, q[n - 1], p);
  for (int s = 1; s < n; s++) res[s] = q[s - 1];
  return res;
}

// a~_i[j], the paper's Toeplitz-with-Y arrangement.
Poly ATilde(const Poly &a, int i, int j, int rank, int small_degree,
            uint64_t p) {
  if (j <= i) return Estar(a, i - j, rank, small_degree);
  return MulY(Estar(a, i - j + rank, rank, small_degree), p);
}

}  // namespace

// The formula itself: sum_i <a~_i, sk'> X^i must equal a * sk in R_N.
TEST(MlweDecomposition, ModDecompIdentityHolds) {
  constexpr uint64_t p = 97;
  std::mt19937_64 gen(12345);
  std::uniform_int_distribution<uint64_t> dist(0, p - 1);

  for (auto shape : std::vector<std::pair<int, int>>{{16, 4}, {32, 8}, {32, 4},
                                                     {64, 8}}) {
    const int degree = shape.first;
    const int small_degree = shape.second;
    const int rank = degree / small_degree;

    Poly a(degree), sk(degree);
    for (int i = 0; i < degree; i++) {
      a[i] = dist(gen);
      sk[i] = dist(gen);
    }

    const Poly expected = NegacyclicMul(a, sk, p);

    // sk' = ( e*_0(sk), ..., e*_{k-1}(sk) )
    std::vector<Poly> sk_prime(rank);
    for (int j = 0; j < rank; j++) {
      sk_prime[j] = Estar(sk, j, rank, small_degree);
    }

    // obtained[i + k*s] = ( sum_j a~_i[j] * sk'_j )[s], recomposed over X^i.
    Poly obtained(degree, 0);
    for (int i = 0; i < rank; i++) {
      Poly acc(small_degree, 0);
      for (int j = 0; j < rank; j++) {
        const Poly term =
            NegacyclicMul(ATilde(a, i, j, rank, small_degree, p), sk_prime[j],
                          p);
        for (int s = 0; s < small_degree; s++) {
          acc[s] = AddMod(acc[s], term[s], p);
        }
      }
      for (int s = 0; s < small_degree; s++) {
        obtained[i + rank * s] = acc[s];
      }
    }

    for (int c = 0; c < degree; c++) {
      ASSERT_EQ(expected[c], obtained[c])
          << "degree " << degree << ", small_degree " << small_degree
          << ", coefficient " << c;
    }
  }
}

// The device kernel, against the same formula evaluated on the host.
TEST_P(Testbed32, ModDecompMatchesHostIndexing) {
  const int degree = 1 << log_degree_;
  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  // Low level keeps the limb count, and hence the k-fold materialisation of
  // the a-part, small. That is also where the product is scheduled.
  const int level = 2;
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

  for (int small_degree : {4096, 16384}) {
    const int rank = degree / small_degree;

    std::vector<MlweCiphertext<word>> parts;
    mlwe.ModDecomp(parts, ct, small_degree);
    ASSERT_EQ(static_cast<int>(parts.size()), rank);

    std::cout << "ModDecomp degree " << degree << " -> " << small_degree
              << ", rank " << rank << ", " << num_total_primes << " limbs"
              << std::endl;

    for (int i = 0; i < rank; i++) {
      ASSERT_EQ(parts[i].rank_, rank);
      ASSERT_EQ(parts[i].degree_, small_degree);

      HostVector<word> dev_a, dev_b;
      CopyDeviceToHost(dev_a, parts[i].a_);
      CopyDeviceToHost(dev_b, parts[i].b_);

      // Counted rather than asserted per element: there are rank^2 *
      // small_degree * limbs comparisons, and a gtest assertion each would
      // dominate the runtime.
      long long mismatches = 0;
      std::string first_mismatch;

      for (int limb = 0; limb < num_total_primes; limb++) {
        const uint64_t p = primes[limb];
        const word *a_src = host_a.data() + limb * degree;
        const word *b_src = host_b.data() + limb * degree;

        // b-part: e*_i(b)
        for (int s = 0; s < small_degree; s++) {
          const uint64_t expected = b_src[i + rank * s];
          const uint64_t obtained = dev_b[limb * small_degree + s];
          if ((expected + p - obtained % p) % p != 0) {
            if (mismatches == 0) {
              first_mismatch = "b-part at i=" + std::to_string(i) +
                               " limb=" + std::to_string(limb) +
                               " s=" + std::to_string(s) + " expected " +
                               std::to_string(expected) + " obtained " +
                               std::to_string(obtained);
            }
            mismatches++;
          }
        }

        // a-part: a~_i[j]
        for (int j = 0; j < rank; j++) {
          for (int s = 0; s < small_degree; s++) {
            uint64_t expected;
            if (j <= i) {
              expected = a_src[(i - j) + rank * s];
            } else {
              const int l = i - j + rank;
              if (s == 0) {
                const uint64_t tail = a_src[l + rank * (small_degree - 1)];
                expected = (p - tail % p) % p;
              } else {
                expected = a_src[l + rank * (s - 1)];
              }
            }
            const uint64_t obtained =
                dev_a[(limb * rank + j) * small_degree + s];
            if ((expected + p - obtained % p) % p != 0) {
              if (mismatches == 0) {
                first_mismatch = "a-part at i=" + std::to_string(i) +
                                 " j=" + std::to_string(j) +
                                 " limb=" + std::to_string(limb) +
                                 " s=" + std::to_string(s) + " expected " +
                                 std::to_string(expected) + " obtained " +
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

// ModPack at the full ring degree, which is the one case the degree-4096 suite
// cannot cover: bootparam_40 carries terminal primes, so its NPs have a
// nonzero terminal-prime offset and the embedded secrets have to be encoded
// against the right prime for each limb. Grafting is invisible at
// ringdegree12_30, whose terminal prime list is empty.
//
// Rank 2 deliberately. The number of switching keys is the rank, each is the
// size of a rotation key, and there is no serialization in Cheddar -- rank 16
// here would cost a couple of GiB to prove nothing that rank 2 does not.
TEST_P(Testbed32, ModPackInvertsModDecompAtFullDegree) {
  const int degree = 1 << log_degree_;
  const int small_degree = degree / 2;
  const int rank = 2;

  MlweHandler<word> mlwe(*param_, context_->ntt_handler_);

  const int level = 2;
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
  std::cout << "degree " << degree << " -> " << small_degree << ", rank "
            << rank << ", level " << level << ": max error " << max_abs
            << std::endl;
  ASSERT_LT(max_abs, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
