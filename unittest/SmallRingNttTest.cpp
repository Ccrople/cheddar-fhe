// Does Cheddar's NTT actually work below log_degree 16?
//
// NTT.h:46 declares min_log_degree_ = 12 and carries a "TODO: allow for
// different log_degree". Parameter.h:32 says "currently only 16 is supported",
// and all four shipped presets are 16. Nothing in the tree exercises a smaller
// ring, so the declared range is an assertion nobody has checked.
//
// This matters because [SYLPH] table 4 places the batch CC-MM at ring degree
// 4096 in the SinC encoding, and every key switch and rescale performed on a
// ring-switched degree-2^12 ciphertext needs a working transform there. The
// PC-MM path can stay in the coefficient domain, but the pipeline as a whole
// cannot.
//
// TWO TRAPS SHAPE THIS FILE.
//
//   1. __cm_log_degree is written once per process. Every handler guards its
//      constant-memory upload with a *class-level* `static inline bool
//      cm_populated_` (NTT.h:26, ElementWise.h:25, ModSwitch.h:77,
//      Hoist.h:45, UserInterface.h:30). The first handler constructed wins;
//      a second one at a different log_degree silently leaves the old value in
//      place and every kernel then computes at the wrong degree.
//
//      So each degree MUST run in its own process. The tests below are
//      separate TESTs precisely so they can be selected one at a time, and
//      each one asserts that it is the first to build a handler. Running the
//      whole binary unfiltered will fail on that assertion rather than report
//      a false pass. Use run_small_ring_ntt.sh, or one --gtest_filter per
//      invocation.
//
//   2. A round trip alone cannot distinguish a correct transform pair from a
//      pair that both do nothing -- and "silently launches no kernel" is a
//      failure mode this codebase has already produced once (Hoist.cu:317
//      launches nothing when num_accum == 1 and returns uninitialised memory).
//      Every case therefore also asserts that NTT(x) differs from x.
//
// What this test does and does not establish: INTT(NTT(x)) == x together with
// NTT(x) != x says the transform pair is invertible and non-degenerate at that
// degree. It does not by itself pin the twiddle convention, so it is a
// necessary condition, not a full correctness proof. A convolution check
// against a host negacyclic product is the natural follow-up once this passes.

#undef ENABLE_EXTENSION

#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Testbed.h"
#include "core/DeviceVector.h"
#include "core/NPInfo.h"
#include "core/NTT.h"
#include "core/Parameter.h"

using word = uint32_t;

namespace {

// All of these are 1 mod 131072, hence 1 mod 2*2^d for every d <= 16, so the
// same list is NTT-friendly at every degree swept here. Taken from
// parameters/bootparam_30.json so that nothing new has to be trusted.
const std::vector<word> kMainPrimes = {1073872897, 1073479681, 1074266113};
const std::vector<word> kAuxPrimes = {2147352577, 2146959361};

// One handler per process; see trap 1 above.
bool handler_built = false;

void RunRoundTrip(int log_degree) {
  ASSERT_FALSE(handler_built)
      << "A handler was already constructed in this process. __cm_log_degree "
         "is process-global and write-once, so this case would silently run at "
         "the previous degree. Run one --gtest_filter per invocation.";
  handler_built = true;

  const int degree = 1 << log_degree;
  const std::vector<std::pair<int, int>> level_config = {{1, 0}, {2, 0}, {3, 0}};

  Parameter<word> param(log_degree, static_cast<double>(UINT64_C(1) << 28),
                        /*default_encryption_level=*/2, level_config,
                        kMainPrimes, kAuxPrimes);
  NTTHandler<word> ntt(param);

  // Level 0 keeps the limb count at its smallest; the transform is per-limb, so
  // nothing about the question depends on carrying more.
  const NPInfo np = param.LevelToNP(0);
  const int num_total = np.GetNumTotal();
  const auto primes = param.GetPrimeVector(np);
  const int size = num_total * degree;

  std::cout << "log_degree " << log_degree << ", degree " << degree << ", "
            << num_total << " limbs" << std::endl;

  // Random coefficients, each reduced into its own prime.
  std::mt19937_64 gen(0xC0FFEE ^ log_degree);
  HostVector<word> host_src(size);
  for (int limb = 0; limb < num_total; limb++) {
    std::uniform_int_distribution<uint64_t> dist(0, primes[limb] - 1);
    for (int i = 0; i < degree; i++) {
      host_src[limb * degree + i] = static_cast<word>(dist(gen));
    }
  }

  DeviceVector<word> src(size), fwd(size), back(size);
  CopyHostToDevice(src, host_src);

  auto fwd_view = fwd.View(0);
  ntt.NTT(fwd_view, np, src.ConstView(0));

  auto back_view = back.View(0);
  ntt.INTT(back_view, np, fwd.ConstView(0), /*montgomery_conversion=*/false);

  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess)
      << "a kernel failed to launch at log_degree " << log_degree;

  HostVector<word> host_fwd, host_back;
  CopyDeviceToHost(host_fwd, fwd);
  CopyDeviceToHost(host_back, back);

  // Trap 2: the transform must actually transform.
  long long unchanged = 0;
  for (int i = 0; i < size; i++) {
    if (host_fwd[i] == host_src[i]) unchanged++;
  }
  ASSERT_LT(unchanged, size / 2)
      << "NTT left " << unchanged << " of " << size
      << " words untouched at log_degree " << log_degree
      << " -- the forward transform is degenerate or never launched";

  // The round trip itself. Counted rather than asserted per element so that a
  // broken degree reports how broken it is instead of stopping at the first
  // word.
  long long mismatches = 0;
  std::string first;
  for (int limb = 0; limb < num_total; limb++) {
    const uint64_t p = primes[limb];
    for (int i = 0; i < degree; i++) {
      const int idx = limb * degree + i;
      const uint64_t expected = host_src[idx];
      const uint64_t obtained = host_back[idx] % p;
      if (expected % p != obtained) {
        if (mismatches == 0) {
          first = "limb " + std::to_string(limb) + " coefficient " +
                  std::to_string(i) + " expected " + std::to_string(expected) +
                  " obtained " + std::to_string(obtained);
        }
        mismatches++;
      }
    }
  }
  ASSERT_EQ(mismatches, 0)
      << "INTT(NTT(x)) != x at log_degree " << log_degree << ": " << mismatches
      << " of " << size << " words differ, first at " << first;

  std::cout << "log_degree " << log_degree << ": round trip exact over " << size
            << " words" << std::endl;
}

}  // namespace

// 2^16 is the only degree the library claims support for -- the control.
TEST(SmallRingNtt, RoundTripLogDegree16) { RunRoundTrip(16); }

// The degrees between are swept so that a failure at 12 can be located rather
// than merely observed.
TEST(SmallRingNtt, RoundTripLogDegree15) { RunRoundTrip(15); }
TEST(SmallRingNtt, RoundTripLogDegree14) { RunRoundTrip(14); }
TEST(SmallRingNtt, RoundTripLogDegree13) { RunRoundTrip(13); }

// The one Sylph actually needs: batch CC-MM and PCMv both live at 4096.
TEST(SmallRingNtt, RoundTripLogDegree12) { RunRoundTrip(12); }
