// What a conjugate-invariant layer actually spends its 72 GiB on, measured
// per object instead of guessed.
//
// Section 1.5da attributed the CI layer's peak by stage and named two
// duplicates it could not price: the second BootContext's `UserInterface`,
// whose basic evaluation keys are built and never read, and the leg's
// BootContext, which is dead after its eight Boots but holds its CoeffToSlot
// and SlotToCoeff diagonals across the seam, the O projection and the whole
// FFN. It could not price them because **no tool in the tree could measure
// anything below 40 GiB**: the upstream is `cuda_async_memory_resource`, so
// `cudaMemGetInfo` reports a pool RESERVATION and a high-water mark, and a
// `CHEDDAR_CI_TILE` sweep that moves an operand from 10.7 GB to 2.7 GB
// measured identical driver numbers at every setting.
//
// `MemoryPool::SetStatisticsEnabled` is that tool -- RMM's
// `statistics_resource_adaptor` around the pool -- and this file is what it
// was added for. Everything here is measured on the LEG'S OWN SHAPE without
// running the leg: `CiBootSet.TheWholeLayerRunsOnTheRealSubring` is a quarter
// of an hour and section 1.5ct spent three of them guessing, while this takes
// about two and a half minutes. A number that can be taken off the pipeline
// should be.
//
// The statistics have to be requested before the first Context exists, so this
// test is deliberately alone in its binary: a second test in the same process
// would either share this one's Contexts or find the request refused.

#include <gtest/gtest.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/MemoryPool.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::BootContext;
using cheddar::EvkRequest;
using cheddar::MemoryPool;

namespace {

constexpr const char *kParam = "ci16_35.json";

// The FFN half of the layer runs at slack nine and the leg at slack zero, and
// that difference is the whole reason the layer holds two BootContexts over
// one secret: SlotToCoeff is compiled at `GetStCStartLevel()`, which moves
// with the slack, while the leg's softmax walk needs `GetEndLevel()` at 16 and
// so needs slack zero. See 1.5ct.
constexpr int kLegSlack = 0;
constexpr int kFfnSlack = 9;

// The layer's own projection shape: rank 512 over a degree-65536 ring, so the
// module components live at degree 128, and the product sits at level 1.
constexpr int kProjRank = 512;
constexpr int kPcmmLevel = 1;
constexpr int kProjSmall = 65536 / kProjRank;

double MiB(int64_t bytes) { return static_cast<double>(bytes) / (1 << 20); }

struct Ledger {
  int64_t last = 0;
  void Mark(const std::string &tag) {
    const auto usage = MemoryPool::GetUsage();
    std::cout << "    " << std::left << std::setw(44) << tag << std::right
              << std::fixed << std::setprecision(1) << std::setw(10)
              << MiB(usage.current_bytes - last) << " MiB   (live "
              << std::setw(9) << MiB(usage.current_bytes) << ", peak "
              << std::setw(9) << MiB(usage.peak_bytes) << ")" << std::endl;
    last = usage.current_bytes;
  }
  void Sync() { last = MemoryPool::GetUsage().current_bytes; }
};

}  // namespace

TEST(MemoryLedger, TheLayersBootstrapSetIsPricedPerObject) {
  // Before any Context: `SetStatisticsEnabled` returns false once one exists,
  // because a buffer captures its resource at allocation time. The return is
  // not asserted -- a later test in the same binary would find the adaptor
  // already installed, which is fine -- but `StatisticsEnabled()` is, below,
  // once a Context has forced the pool into being.
  MemoryPool::SetStatisticsEnabled(true);

  Ledger led;
  std::cout << "\n  [mem] the leg's BootContext, " << kParam << ", slack "
            << kLegSlack << "\n";

  // ---- the leg's set ---------------------------------------------------
  auto leg = std::make_unique<Ring>(kParam, std::vector<int>{}, kLegSlack,
                                    /*build_user_interface=*/false);
  ASSERT_TRUE(MemoryPool::StatisticsEnabled())
      << "the adaptor was requested but is not installed";
  led.Mark("Context (primes, twiddles, EvalMod tables)");

  leg->BuildUserInterface();
  led.Mark("UserInterface: secrets + PrepareBasicEvks");

  auto lctx = std::dynamic_pointer_cast<BootContext<word>>(leg->context);
  ASSERT_NE(lctx, nullptr);
  const int num_slots = leg->param->MaxNumSlots();

  lctx->PrepareEvalMod();
  led.Mark("PrepareEvalMod");

  const int64_t leg_before_fft = MemoryPool::GetUsage().current_bytes;
  lctx->PrepareEvalSpecialFFT(num_slots);
  led.Mark("PrepareEvalSpecialFFT (CtS + StC diagonals)");
  const int64_t leg_fft =
      MemoryPool::GetUsage().current_bytes - leg_before_fft;

  const int64_t leg_before_keys = MemoryPool::GetUsage().current_bytes;
  size_t leg_key_count = 0, minks_key_count = 0;
  {
    EvkRequest req;
    lctx->AddRequiredRotations(req, num_slots);
    leg_key_count = req.size();
    EvkRequest minks_req;
    lctx->AddRequiredRotations(minks_req, num_slots, /*min_ks=*/true);
    minks_key_count = minks_req.size();
    leg->ui->PrepareRotationKey(req);
  }
  led.Mark("the bootstrap's rotation keys");
  const int64_t leg_keys =
      MemoryPool::GetUsage().current_bytes - leg_before_keys;
  const int64_t leg_total = MemoryPool::GetUsage().current_bytes;

  // ---- the FFN's set, over the same primes and the same secret ---------
  //
  // 1.5ct: "A second Context over the same primes holding the same secret is
  // what the switching ring already is; here the two differ only in slack."
  // The keys are shared -- one EvkMap serves both -- so what the second set
  // costs is its Context, its EvalMod tables and its own CtS/StC diagonals,
  // and this separates the three for the first time.
  std::cout << "\n  [mem] the FFN's BootContext, same primes and secret, slack "
            << kFfnSlack << "\n";
  led.Sync();
  const int64_t ffn_start = MemoryPool::GetUsage().current_bytes;

  Ring ffn(kParam, leg->ui->GetSecretCoeffs(), kFfnSlack,
           /*build_user_interface=*/false);
  led.Mark("Context");

  auto fctx = std::dynamic_pointer_cast<BootContext<word>>(ffn.context);
  ASSERT_NE(fctx, nullptr);
  fctx->PrepareEvalMod();
  led.Mark("PrepareEvalMod");

  const int64_t ffn_before_fft = MemoryPool::GetUsage().current_bytes;
  fctx->PrepareEvalSpecialFFT(num_slots);
  led.Mark("PrepareEvalSpecialFFT (CtS + StC diagonals)");
  const int64_t ffn_fft =
      MemoryPool::GetUsage().current_bytes - ffn_before_fft;

  // AND ITS ROTATIONS, INTO THE SAME EvkMap. 1.5da removed a verbatim
  // duplicate here by having both Contexts share one map, but a key is per
  // (index, LEVEL) and the slack moves every level the FFN's transforms run
  // at. So sharing the map does not by itself mean sharing the keys, and this
  // row is the first measurement of how much of the second set is genuinely
  // new material rather than a second name for the leg's.
  const int64_t ffn_before_keys = MemoryPool::GetUsage().current_bytes;
  {
    EvkRequest req;
    fctx->AddRequiredRotations(req, num_slots);
    leg->ui->PrepareRotationKey(req);
  }
  led.Mark("its rotations, added to the leg's EvkMap");

  // AND ITS CoeffToSlot HALF IS A BYTE-FOR-BYTE DUPLICATE OF THE LEG'S.
  // `BootParameter::GetCtSStartLevel()` is `max_level_` and
  // `GetEvalModStartLevel()` / `GetEvalModEndLevel()` are derived from it and
  // from `num_cts_levels_`; the slack enters only at `GetStCStartLevel()`,
  // which is `GetEvalModEndLevel() - num_slack_levels_`. So two BootContexts
  // over the same primes that differ only in slack compile CoeffToSlot at the
  // same levels from the same constants, and one of the two copies is dead
  // weight. `PrepareEvalSpecialFFT` builds the pair together, so this is a
  // statement about the code rather than a row in the table above -- but it
  // is the largest duplicate left in the layer, and it is what a shareable
  // CtS would buy.
  EXPECT_EQ(lctx->GetBootParameter().GetCtSStartLevel(),
            fctx->GetBootParameter().GetCtSStartLevel())
      << "CoeffToSlot is supposed to be slack-independent";
  EXPECT_NE(lctx->GetBootParameter().GetStCStartLevel(),
            fctx->GetBootParameter().GetStCStartLevel())
      << "the two Contexts exist precisely because SlotToCoeff is not";
  const int64_t ffn_keys =
      MemoryPool::GetUsage().current_bytes - ffn_before_keys;

  const int64_t ffn_total = MemoryPool::GetUsage().current_bytes - ffn_start;

  // ---- what the deferred UserInterface saved ---------------------------
  //
  // The layer's second BootContext never encrypts and never decrypts; every
  // key lookup in the FFN goes through the leg's EvkMap. So its
  // `PrepareBasicEvks` -- a multiplication key plus, under sparse-secret
  // encapsulation, a dense-to-sparse and a sparse-to-dense key at the top np
  // -- is built and read by nothing. Building one here on purpose prices it.
  std::cout << "\n  [mem] what the FFN Context does NOT have to build\n";
  led.Sync();
  const int64_t ui_before = MemoryPool::GetUsage().current_bytes;
  ffn.BuildUserInterface();
  led.Mark("UserInterface: secrets + PrepareBasicEvks");
  const int64_t ui_cost = MemoryPool::GetUsage().current_bytes - ui_before;

  // ---- the projection's key material -----------------------------------
  //
  // The leg is 79.1% of the layer's peak (1.5da) and its bootstrap set is only
  // 13.8 GiB of the 57 it releases, so most of the leg is somewhere else. The
  // first place to look is `ModPack`: a projection at the layer's shape does
  // `rank` key switches per emission, and section 1.5bh put the key material at
  // "512 big-ring keys ~4.3 GB" from a size calculation nobody could check.
  // Building them here is the check, and it costs seconds rather than the
  // leg's twelve minutes of converters.
  std::cout << "\n  [mem] the projection's ModPack keys, rank "
            << kProjRank << " at level " << kPcmmLevel << "\n";
  led.Sync();
  const int64_t pack_before = MemoryPool::GetUsage().current_bytes;
  leg->ui->PrepareModPackKeys(kProjSmall, kPcmmLevel);
  led.Mark("PrepareModPackKeys");
  const int64_t pack_cost =
      MemoryPool::GetUsage().current_bytes - pack_before;

  // ---- releasing the leg, in two steps ---------------------------------
  //
  // The leg's BootContext is dead after its eight Boots: 1.5da found the only
  // later use of it is a scalar `GetLogMessageRatio()`. But it cannot simply
  // be dropped, and the two rows below are why. A `ContextPtr` is a
  // `shared_ptr`, and the layer holds TWO handles on the leg's Context that it
  // still needs -- `bctx`, for that scalar, and `boot.ui`, which owns the
  // EvkMap every FFN key lookup goes through. Dropping the Ring therefore
  // returns the keys and the Parameter and nothing else; the CoeffToSlot and
  // SlotToCoeff diagonals stay until the LAST handle goes, and the last handle
  // is one the layer cannot give up.
  //
  // So the second row is the price of a `BootContext` method that drops just
  // the EvalSpecialFFT tables, and it is measured here rather than argued. The
  // first run of this file held `lctx` by accident and reported the release as
  // 7331 MiB out of a 13800 MiB set; the missing 6469 was not a leak, it was
  // that handle -- which is exactly the layer's situation, found in two and a
  // half minutes instead of fifteen.
  std::cout << "\n  [mem] releasing the leg, in two steps\n";
  led.Sync();
  const int64_t before_ring = MemoryPool::GetUsage().current_bytes;
  leg.reset();
  led.Mark("the Ring: UserInterface, EvkMap, Parameter");
  const int64_t ring_released =
      before_ring - MemoryPool::GetUsage().current_bytes;

  const int64_t before_ctx = MemoryPool::GetUsage().current_bytes;
  lctx.reset();
  led.Mark("the last handle on its BootContext");
  const int64_t ctx_released =
      before_ctx - MemoryPool::GetUsage().current_bytes;

  // ---- what min_ks would buy on the same key set ------------------------
  //
  // The rotation keys are the largest single object measured above -- 7059
  // MiB against 6408 for the diagonals -- and Cheddar has a lever aimed
  // exactly at them: `min_ks` (ARK-style key reuse) is a runtime flag on
  // `AddRequiredRotations` and on `Boot`, and CLAUDE.md records it as "36%
  // slower -- it buys key count, not time" on the ordinary ring, with the CI
  // bootstrap at 58.2 ms against MinKS 150.8. What it buys has never been
  // measured in bytes, which is the half that decides whether a layer fits.
  //
  // Built into a fresh ring, after the leg has been released, so that the
  // comparison is against `leg_keys` and not against a pool state.
  std::cout << "\n  [mem] the same bootstrap key set under min_ks\n";
  led.Sync();
  int64_t minks_keys = 0;
  {
    Ring alt(kParam, std::vector<int>{}, kLegSlack);
    auto actx = std::dynamic_pointer_cast<BootContext<word>>(alt.context);
    ASSERT_NE(actx, nullptr);
    actx->PrepareEvalMod();
    actx->PrepareEvalSpecialFFT(num_slots);
    led.Mark("a fresh Context and its CtS + StC");
    const int64_t before = MemoryPool::GetUsage().current_bytes;
    EvkRequest req;
    actx->AddRequiredRotations(req, num_slots, /*min_ks=*/true);
    alt.ui->PrepareRotationKey(req);
    led.Mark("its rotation keys, min_ks");
    std::cout << "      " << leg_key_count << " keys plain against "
              << minks_key_count << " under min_ks" << std::endl;
    minks_keys = MemoryPool::GetUsage().current_bytes - before;
  }
  led.Mark("released");

  const auto usage = MemoryPool::GetUsage();
  std::cout << "\n  peak live demand " << std::fixed << std::setprecision(1)
            << MiB(usage.peak_bytes) << " MiB over "
            << usage.peak_allocations << " concurrent allocations; "
            << MiB(usage.total_bytes) / 1024.0 << " GiB allocated in total\n"
            << std::endl;

  // The numbers this file exists to produce, restated so that a reader of the
  // log does not have to subtract rows.
  std::cout << "  the leg's whole boot set        " << MiB(leg_total)
            << " MiB\n"
            << "    of which CtS + StC            " << MiB(leg_fft)
            << " MiB\n"
            << "    of which rotation keys        " << MiB(leg_keys)
            << " MiB\n"
            << "    (same key set, min_ks)        " << MiB(minks_keys)
            << " MiB\n"
            << "  the FFN's second boot set       " << MiB(ffn_total)
            << " MiB\n"
            << "    of which CtS + StC            " << MiB(ffn_fft)
            << " MiB\n"
            << "    of which NEW rotation keys    " << MiB(ffn_keys)
            << " MiB\n"
            << "  a duplicate UserInterface       " << MiB(ui_cost)
            << " MiB\n"
            << "  the projection's ModPack keys   " << MiB(pack_cost)
            << " MiB\n"
            << "  freed by dropping the Ring      " << MiB(ring_released)
            << " MiB\n"
            << "  held by the last Context handle " << MiB(ctx_released)
            << " MiB" << std::endl;

  // Guards, not readings. Each is a fact the layer's schedule depends on, and
  // each would have caught a real mistake:
  //
  //  - the adaptor sees something at all, so a zero row is a broken tool
  //    rather than a free object;
  //  - the deferred UserInterface really does cost something, so passing
  //    `build_user_interface=false` in the layer is a saving and not a no-op;
  //  - the diagonals really are owned by the Context and not by the Ring,
  //    which is what makes a release method worth writing;
  //  - and the two steps together return essentially the whole set, so nothing
  //    here is an accounting artefact.
  EXPECT_GT(leg_total, 0) << "the statistics adaptor counted nothing";
  EXPECT_GT(ui_cost, 0) << "PrepareBasicEvks is supposed to allocate";
  EXPECT_GT(ctx_released, leg_fft / 2)
      << "the CtS/StC diagonals are supposed to be owned by the Context";
  EXPECT_GT(ring_released + ctx_released, leg_total * 9 / 10)
      << "dropping both handles should return nearly the whole leg set";
}
