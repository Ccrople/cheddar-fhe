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

  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots,
                                     EnvInt("BB_MINKS", 0) != 0);
  interface_->PrepareRotationKey(req);

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
    testing::Values("bootparam_35.json", "bootparam_40.json",
                    "sylphflow16_35.json", "sylphflow16_40.json",
                    "ci16_35.json", "ci16_40.json", "ci16_35_stc2.json",
                    "ci16_35_land17c3e10.json", "ci16_35_land17c3e10v3.json",
                    "ci16_35_land13c2e9.json"),
    [](const testing::TestParamInfo<BootBed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
