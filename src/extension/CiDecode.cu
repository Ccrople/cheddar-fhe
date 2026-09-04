#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>

#include "common/Assert.h"
#include "extension/CiDecode.h"

namespace cheddar {

template <typename word>
PlainHoistMap CiDecodeUnpack<word>::MakeMap(
    const std::vector<Message> &row_masks, int stride) {
  const int K = static_cast<int>(row_masks.size());
  PlainHoistMap map;
  auto &bs = map[0];
  for (int b = 0; b < K; b++) {
    bs.try_emplace(b * stride, row_masks.at((K - b) % K));
  }
  return map;
}

template <typename word>
CiDecodeUnpack<word>::CiDecodeUnpack(ConstContextPtr<word> context,
                                     const std::vector<Message> &row_masks,
                                     int stride, int pt_level, double pt_scale)
    : num_channels_(static_cast<int>(row_masks.size())),
      stride_(stride),
      pt_level_(pt_level),
      pt_scale_(pt_scale),
      hoist_(context, MakeMap(row_masks, stride), pt_level, pt_scale,
             /*suppress_bs_swap=*/true) {
  AssertTrue(num_channels_ > 1, "CiDecodeUnpack: nothing to unpack");
}

template <typename word>
void CiDecodeUnpack<word>::AddRequiredRotations(EvkRequest &req) const {
  hoist_.AddRequiredRotations(req);
}

namespace {
// CHEDDAR_DECODE_UNPACK_TIME=1: synchronise and report the three phases.
// A debug knob, not a production bracket.
struct PhaseClock {
  bool on = false;
  std::chrono::steady_clock::time_point last;
  void Start() {
    const char *e = std::getenv("CHEDDAR_DECODE_UNPACK_TIME");
    on = (e != nullptr && e[0] == '1');
    if (on) {
      cudaDeviceSynchronize();
      last = std::chrono::steady_clock::now();
    }
  }
  void Mark(const char *name) {
    if (!on) return;
    cudaDeviceSynchronize();
    auto now = std::chrono::steady_clock::now();
    std::cout << "  [unpack] " << name << " "
              << std::chrono::duration<double, std::milli>(now - last).count()
              << " ms" << std::endl;
    last = now;
  }
};
}  // namespace

template <typename word>
void CiDecodeUnpack<word>::Evaluate(ConstContextPtr<word> context,
                                    std::vector<Ct> &out, const Ct &input,
                                    const EvkMap<word> &evk_map) const {
  const int K = num_channels_;
  PhaseClock clock;
  clock.Start();

  // 1. The K hoisted rotations, shared by every channel (bs[0] is the
  // pseudo-mod-up'ed input, exactly as the serial evaluation reads it).
  std::map<int, Ct> bs;
  hoist_.EvaluateBabyStep(context, bs, input, evk_map);
  clock.Mark("babies");

  // 2. The select: one launch over every (baby, channel) pair, in the
  // extended basis the babies arrive in.
  const auto &pt_map = hoist_.GetPlaintexts().at(0);
  const NPInfo ext_np = bs.at(0).GetNP();
  int num_slots = input.GetNumSlots();

  std::vector<std::vector<DvConstView<word>>> ct_srcs;
  std::vector<DvConstView<word>> pt_srcs;
  ct_srcs.reserve(K);
  pt_srcs.reserve(K);
  for (int b = 0; b < K; b++) {
    ct_srcs.push_back(bs.at(b * stride_).ConstViewVector());
  }
  // pt_srcs[j] must be row_mask_j, which the map holds at baby (K - j) % K.
  for (int j = 0; j < K; j++) {
    const auto &pt = pt_map.at(((K - j) % K) * stride_);
    pt_srcs.push_back(pt.ConstView());
    num_slots = std::max(num_slots, pt.GetNumSlots());
  }

  // The accumulators live in ONE strided arena ([c][bx | ax], the extended
  // basis) so the mod-down can run as one batch.
  const int degree = context->param_.degree_;
  const int ext_words = ext_np.GetNumTotal() * degree;
  const int aux_words = ext_np.num_aux_ * degree;
  DeviceVector<word> arena(2 * K * ext_words);
  std::vector<std::vector<DvView<word>>> dst;
  dst.reserve(K);
  for (int c = 0; c < K; c++) {
    dst.push_back(
        {DvView<word>(arena.data() + static_cast<size_t>(2 * c) * ext_words,
                      ext_words, aux_words),
         DvView<word>(
             arena.data() + static_cast<size_t>(2 * c + 1) * ext_words,
             ext_words, aux_words)});
  }
  context->elem_handler_.PAccumRotBatchCt(dst, ext_np, ct_srcs, pt_srcs);
  clock.Mark("select");

  // 3. The final mod-down as TWO batch calls (`ModDownAndRescaleBatch` is
  // word for word the serial `ModDownAndRescale`), then each channel's
  // halves peeled off the out arena -- EvaluateFinalModDown's arithmetic
  // and scale, so channel 0 stays word-for-word the serial evaluation's.
  const NPInfo next_np = context->param_.LevelToNP(pt_level_ - 1);
  const NPInfo out_np(next_np.num_main_, next_np.num_ter_, 0, next_np.degree_);
  auto &mod_switcher = context->mod_switch_handlers_.at(pt_level_);
  const double out_scale =
      pt_scale_ * input.GetScale() /
      context->param_.GetRescalePrimeProd(pt_level_);
  const int out_words = out_np.GetNumQ() * degree;
  DeviceVector<word> out_arena(2 * K * out_words);
  mod_switcher.ModDownAndRescaleBatch(out_arena.data(), 2 * out_words,
                                      arena.data(), 2 * ext_words, K);
  mod_switcher.ModDownAndRescaleBatch(out_arena.data() + out_words,
                                      2 * out_words, arena.data() + ext_words,
                                      2 * ext_words, K);
  out.clear();
  out.reserve(K);
  for (int c = 0; c < K; c++) {
    out.emplace_back(out_np);
    Ct &res = out.back();
    res.SetNumSlots(num_slots);
    res.SetScale(out_scale);
    cudaMemcpyAsync(res.bx_.data(),
                    out_arena.data() + static_cast<size_t>(2 * c) * out_words,
                    static_cast<size_t>(out_words) * sizeof(word),
                    cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(
        res.ax_.data(),
        out_arena.data() + static_cast<size_t>(2 * c + 1) * out_words,
        static_cast<size_t>(out_words) * sizeof(word),
        cudaMemcpyDeviceToDevice);
  }
  clock.Mark("moddown");
}

template class CiDecodeUnpack<uint32_t>;
template class CiDecodeUnpack<uint64_t>;

}  // namespace cheddar
