// One bootstrap, measured properly.
//
// `Bootstrapping.cpp`'s `Bootstrap` times a single call after five warm-ups
// and is a correctness test that happens to print a number. This file is the
// other thing: a stable latency bench for ONE ciphertext's bootstrap, with a
// device sync on both sides of every timed call so the number is the
// bootstrap's own latency rather than a slice of a pipelined queue, and with
// enough repetitions to separate the floor from the pool's noise.
//
// It also marks each repetition with an NVTX range that CLOSES AFTER the
// sync, so an nsys capture attributes every kernel to the repetition that
// launched it -- the plain `boot: Boot` range ends when the host pops it,
// which on an unsynchronised stream is long before the GPU is done.
//
//   BB_WARM   warm-up bootstraps before timing (default 5)
//   BB_REPS   timed repetitions (default 20)
//   BB_MINKS  1 to bench the min_ks path as well (default 0)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "Testbed.h"
#include "core/MemoryPool.h"
#include "extension/Profile.h"

namespace {

int EnvInt(const char *key, int fallback) {
  const char *v = std::getenv(key);
  return (v != nullptr) ? std::atoi(v) : fallback;
}

struct Stat {
  double min = 0.0, mean = 0.0, median = 0.0, max = 0.0;
};

// Every sample brackets `f` with a device sync, so a repetition cannot borrow
// the previous one's queue. The floor (min) is the honest latency; the mean
// carries whatever the allocator did that run.
template <typename F>
Stat TimeIt(int reps, const char *label, F &&f) {
  std::vector<double> t;
  t.reserve(reps);
  for (int i = 0; i < reps; i++) {
    cudaDeviceSynchronize();
    const auto start = std::chrono::steady_clock::now();
#ifdef CHEDDAR_HAS_NVTX
    nvtxRangePushA(label);
#else
    (void)label;
#endif
    f();
    cudaDeviceSynchronize();
#ifdef CHEDDAR_HAS_NVTX
    nvtxRangePop();
#endif
    const auto end = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::vector<double> s = t;
  std::sort(s.begin(), s.end());
  Stat out;
  out.min = s.front();
  out.max = s.back();
  out.median = s[s.size() / 2];
  double sum = 0.0;
  for (double x : t) sum += x;
  out.mean = sum / static_cast<double>(t.size());
  return out;
}

void Report(const char *what, const Stat &s, int reps) {
  std::cout << "[bench] " << what << ": min " << std::fixed
            << std::setprecision(3) << s.min << " ms, median " << s.median
            << ", mean " << s.mean << ", max " << s.max << "  (" << reps
            << " reps)" << std::endl;
}

}  // namespace

// The bootstrap's level plan is the cost model's other half: CoeffToSlot and
// SlotToCoeff are 86% of the time and both cost (diagonals x limbs), and the
// diagonal count is set by how many levels each is given. These override the
// preset so the curve can be measured; the landing level moves with them and
// the bench prints it, because a faster bootstrap that lands lower is a trade
// and not a win.
class BootBed : public Testbed32 {
 protected:
  int BootMaxLevel() const override {
    return EnvInt("BB_MAXLEVEL", Testbed32::BootMaxLevel());
  }
  int BootCtsLevels() const override {
    return EnvInt("BB_CTS", Testbed32::BootCtsLevels());
  }
  int BootStcLevels() const override {
    return EnvInt("BB_STC", Testbed32::BootStcLevels());
  }
};

TEST_P(BootBed, BootLatency) {
  using word = uint32_t;
  const int num_slots = param_->MaxNumSlots();
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  ASSERT_NE(boot_context, nullptr) << "preset is not a bootstrapping one";

  // WHAT A BOOTSTRAP COSTS IN MEMORY, beside what it costs in time. Three
  // numbers, all read off the objects rather than modelled: the pool's live
  // bytes taken by the boot's tables (EvalMod + the CtS/StC plaintexts), the
  // evaluation keys' own bytes (every key in the map after the boot's
  // rotations are added: rotation keys, the relinearization key and the
  // sparse-secret pair), and one rotation key's bytes -- which is
  // `beta * (num_q + num_aux) * 2 polys * N * sizeof(word)`, so a preset
  // whose digit count or prime count moves shows it here first.
  const auto pool_before = cheddar::MemoryPool::GetUsage();
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  const auto pool_tables = cheddar::MemoryPool::GetUsage();
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots,
                                     EnvInt("BB_MINKS", 0) != 0);
  interface_->PrepareRotationKey(req);
  const auto pool_keys = cheddar::MemoryPool::GetUsage();
  {
    const auto &evk_map = interface_->GetEvkMap();
    size_t evk_bytes = 0, rot_bytes = 0;
    int num_keys = 0, num_rot = 0;
    for (const auto &kv : evk_map) {
      size_t bytes = 0;
      for (const auto &v : kv.second.bx_) bytes += v.size();
      for (const auto &v : kv.second.ax_) bytes += v.size();
      evk_bytes += bytes;
      num_keys++;
      if (kv.first >= 0 && kv.first != EvkMap<word>::kConjugationKeyIndex) {
        rot_bytes += bytes;
        num_rot++;
      }
    }
    const auto top_np = param_->LevelToNP(param_->max_level_);
    std::cout << "[bench] memory: boot tables "
              << ((pool_tables.current_bytes - pool_before.current_bytes) >> 20)
              << " MiB (EvalMod + CtS/StC plaintexts), evaluation keys "
              << (evk_bytes >> 20) << " MiB in " << num_keys << " keys ("
              << num_rot << " rotation keys, " << (rot_bytes >> 20)
              << " MiB; one key "
              << ((num_rot > 0 ? rot_bytes / num_rot : 0) >> 20)
              << " MiB), pool live after keys "
              << (pool_keys.current_bytes >> 20) << " MiB; top NP ("
              << top_np.num_main_ << ", " << top_np.num_ter_ << ") + "
              << param_->GetMaxNumAux() << " aux, "
              << (top_np.num_main_ + top_np.num_ter_) << " Q primes"
              << std::endl;
  }

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots, -1.0, 1.0,
                        /*complex=*/!param_->conjugate_invariant_);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg, 0);

  const int warm = EnvInt("BB_WARM", 5);
  const int reps = EnvInt("BB_REPS", 20);
  Ciphertext<word> res;

  for (int i = 0; i < warm; i++) {
    boot_context->Boot(res, ct, interface_->GetEvkMap());
  }
  cudaDeviceSynchronize();

  const Stat boot = TimeIt(reps, "bench: boot", [&] {
    boot_context->Boot(res, ct, interface_->GetEvkMap());
  });
  Report("Boot", boot, reps);

  if (EnvInt("BB_MINKS", 0) != 0) {
    for (int i = 0; i < warm; i++) {
      boot_context->Boot(res, ct, interface_->GetEvkMap(), true);
    }
    const Stat mk = TimeIt(reps, "bench: boot minks", [&] {
      boot_context->Boot(res, ct, interface_->GetEvkMap(), true);
    });
    Report("Boot(min_ks)", mk, reps);
  }

  // A bootstrap's time means nothing without its precision, so the bench
  // reports both: the landing level it bought and the worst slot it left.
  std::vector<Complex> out;
  DecryptAndDecode(out, res);
  double worst = 0.0;
  for (size_t i = 0; i < msg.size(); i++) {
    worst = std::max(worst, std::abs(out[i] - msg[i]));
  }
  std::cout << "[bench] land level " << param_->NPToLevel(res.GetNP())
            << ", worst |err| 2^" << std::setprecision(2)
            << std::log2(worst) << std::endl;
  CompareMessages(msg, out);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, BootBed,
    testing::ValuesIn(PresetList({"bootparam_35.json", "bootparam_40.json",
                    "sylphflow16_35.json", "sylphflow16_40.json",
                    "ci16_35.json", "ci16_40.json", "ci16_35_stc2.json",
                    "ci16_35_land17c3e10.json", "ci16_35_land17c3e10v3.json",
                    "ci16_35_land13c2e9.json",
                    "ci16_42_k16_w60.json", "ci16_42_k32_w60.json",
                    "ci16_42_k64_w60.json",
                    // The landing LADDER, land19 (= ci16_35's own shape, 60
                    // Q limbs) down to land5 (44). A boot is bandwidth-bound
                    // and most of the bytes are evaluation key, whose size is
                    // `beta * (num_q + num_aux)` with `beta = DivCeil(num_q,
                    // num_aux)` -- so shortening the chain to what the
                    // consumer actually uses pays about quadratically, with a
                    // step wherever beta drops.
                    //
                    // v3 ONLY (2026-09-10). These rungs solve EvalMod's scale
                    // recursion instead of tolerating it, so the landing scale
                    // is 2^58.000 rather than the v2 wander that `param_audit`
                    // measured at -1.8 .. -6.2 bits. The v2 rungs they replace
                    // were deleted with their presets: the comparison they
                    // existed to make is settled and recorded in
                    // `reference/audit/`, and keeping a ladder nothing ships
                    // on costs a preset each. The two junction rungs (15, 9)
                    // need a ninth EvalMod level for the solve to converge, so
                    // they are their own shapes, not twins.
                    "ci16_35_land19v3.json", "ci16_35_land12v3.json",
                    "ci16_35_land6v3.json", "ci16_35_land5v3.json",
                    "ci16_35_land15e9v3.json", "ci16_35_land9e9v3.json",
                    // The second 2^42 cut, one prefix for all three K.
                    "ci16_42_k16.json", "ci16_42_k32.json",
                    "ci16_42_k64.json"})),
    [](const testing::TestParamInfo<BootBed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
