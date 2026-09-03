#pragma once

#include <complex>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "core/Serialization.h"

namespace cheddar {

using Message = std::vector<std::complex<double>>;
using PlainHoistMap = std::map<int, std::map<int, Message>>;

/**
 * @brief This class implements the baby-step/giant-step (BSGS) technique
 * used to efficiently evaluate multiple ciphertext rotations combined with
 * plaintext multiplications. The main use case is linear transformation.
 * This implementation supports the "Double Hoisting" technique from
 * Jean-Philippe Bossuat, Christian Mouchet, Juan Troncoso-Pastoriza, and
 * Jean-Pierre Hubaux. "Efficient Bootstrapping for Approximate Homomorphic
 * Encryption with Non-sparse Keys". Advances in Cryptology – EUROCRYPT 2021.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class HoistHandler {
 private:
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Complex = std::complex<double>;

  int pt_level_;
  double pt_scale_;

  std::set<int> bs_indices_;
  std::vector<int> gs_indices_;

  static constexpr int kernel_block_dim_ = 256;

  constexpr static int max_log_beta_ = 4;
  constexpr static int max_log_bs_ = 7;
  // The giant-step key switches one rotation at a time (the old loop;
  // `CHEDDAR_HOIST_GS_SERIAL=1`) instead of as one batched group.
  static bool gs_serial_;
  // The baby steps one ciphertext at a time (`CHEDDAR_HOIST_BS_SERIAL=1`)
  // instead of as one `ModUpBatch` + `BSFusedKernelBatch` group.
  static bool bs_serial_;
  // `EvaluateBatch` one ciphertext at a time (`CHEDDAR_HOIST_EVAL_SERIAL=1`)
  // instead of the whole-transform group.
  static bool eval_serial_;

 public:
  /**
   * @brief Run the giant steps' key switches one by one (the serial loop) or
   * as one group (`Context::MultKeyBatchNoModDown` + `PermuteAccum`); the two
   * are word-for-word equal (`boot_test`).
   */
  static void SetGiantStepSerial(bool serial);
  /** @brief The same switch for the batched baby step. */
  static void SetBabyStepSerial(bool serial);
  /** @brief The same switch for the whole-transform `EvaluateBatch`. */
  static void SetEvaluateSerial(bool serial);

 private:

  // `mutable`: residency (`Stage`/`Unstage`) is not a logical change to the
  // transform, and every reader of the map is const.
  mutable std::map<int, std::map<int, Pt>> hoist_pt_map_;
  // The plaintexts' host copies, in map order, filled by the first `Unstage`
  // and kept: a transform that is unstaged twice copies down once.
  mutable std::vector<HostVector<word>> host_pts_;
  mutable bool on_device_ = true;
  // The host copies' buffers registered with the driver (pinned), to be
  // unregistered when the handler dies. A moved-from handler keeps none.
  mutable std::vector<void *> registered_;
  // The fused kernels' pointer tables, uploaded as one buffer per launch
  // (`TableUpload` in Hoist.cu) and kept: the next launch's copy is behind
  // this launch's kernel on the stream, so the buffer is reused without a
  // wait.
  mutable DeviceVector<uint64_t> table_scratch_;

  // initialization-related methods
  void ExtractBSIndices(const PlainHoistMap &hoist_map);
  void CompilePlaintexts(ConstContextPtr<word> context,
                         const PlainHoistMap &hoist_map);
  std::pair<int, int> CheckStrideMinKS() const;

  // Deserializing constructor. A tag type rather than an overload because the
  // members it fills are exactly the ones the compiling constructor derives,
  // and the two must not be confusable at a call site.
  struct FromArchive {};
  HoistHandler(FromArchive, ArchiveReader &ar);

  // optimization-related methods
  void GSFusedPAccum(ConstContextPtr<word> context, std::map<int, Ct> &results,
                     const std::vector<int> &gs_indices,
                     const std::map<int, Ct> &bs) const;
  static void GSFusedComplexPAccum(ConstContextPtr<word> context,
                                   std::map<int, Ct> &results_re,
                                   std::map<int, Ct> *results_im,
                                   const HoistHandler &re_h,
                                   const HoistHandler &im_h,
                                   const std::map<int, Ct> &bs_re,
                                   const std::map<int, Ct> *bs_im);

  /**
   * @brief `GSFusedComplexPAccum` over a GROUP of ciphertexts in one kernel:
   * the baby-step and output tables carry one slice per ciphertext, the
   * plaintext table is SHARED, and the ciphertext index rides the FAST grid
   * dimension so same-position blocks of different ciphertexts hit the same
   * plaintext lines in L2 -- the diagonal table is streamed from DRAM once
   * per ~group instead of once per ciphertext (Doing.md 7.30: that stream
   * was 20 of a 55 ms bootstrap). Per-ciphertext arithmetic and its order
   * are exactly the serial kernel's, so the words agree exactly.
   */
  static void GSFusedComplexPAccumBatch(
      ConstContextPtr<word> context,
      std::vector<std::map<int, Ct>> &results_re,
      std::vector<std::map<int, Ct>> *results_im, const HoistHandler &re_h,
      const HoistHandler &im_h,
      const std::vector<const std::map<int, Ct> *> &bs_re,
      const std::vector<const std::map<int, Ct> *> *bs_im);

  /// The rotate/fold half of `EvaluateGiantStepComplex` (giant key switches,
  /// PermuteAccum, the final mod-down), shared by the serial call and the
  /// batched group so the two cannot drift.
  static void GSComplexRotateFold(ConstContextPtr<word> context,
                                  const HoistHandler &re_h,
                                  std::map<int, Ct> &accum, Ct &res,
                                  const EvkMap<word> &evk_map,
                                  int input_num_slots, double input_scale);

  /// `GSComplexRotateFold` over EVERY (ciphertext, half) accumulator map of
  /// a batched group at once: all the giant key switches as ONE gather +
  /// `ModDownBatch` + `MultKeyBatchNoModDown` (the Doing 3.23 pattern the
  /// complex path never had), then per map the PermuteAccum folds and the
  /// final mod-down. Word for word the loop of `GSComplexRotateFold` calls
  /// -- the same kernels, and the modular sums in either order. Chunked by
  /// `CHEDDAR_HOIST_GS_CHUNK_MIB` (default 2048) of key-switch output.
  static void GSComplexRotateFoldGroup(ConstContextPtr<word> context,
                                       const HoistHandler &re_h,
                                       std::vector<std::map<int, Ct> *> &accums,
                                       std::vector<Ct *> &results,
                                       const EvkMap<word> &evk_map,
                                       int input_num_slots, double input_scale);
  void BSFusedKeyMult(ConstContextPtr<word> context, std::map<int, Ct> &res,
                      std::vector<Dv> &a_modup, const Ct &a_orig,
                      const EvkMap<word> &keys, std::vector<int> &rotations,
                      const Dv &input_bx_pseudo_modup) const;
  /// `BSFusedKeyMult` over a GROUP of ciphertexts in one kernel: the key and
  /// galois tables are SHARED (one transform, one key set) and the ciphertext
  /// index rides the FAST grid dimension so same-position blocks of different
  /// ciphertexts hit the same key lines in L2. `a_modup[c * num_digits + j]`
  /// holds ciphertext c's digit j at `modup_stride` words; per-ciphertext
  /// arithmetic and its order are exactly the serial kernel's.
  void BSFusedKeyMultBatch(ConstContextPtr<word> context,
                           const std::vector<std::map<int, Ct> *> &res,
                           const std::vector<const word *> &a_modup,
                           const std::vector<const Ct *> &a_origs,
                           const EvkMap<word> &keys,
                           std::vector<int> &rotations,
                           const std::vector<const word *> &pseudo_modup,
                           const std::vector<word *> *dst_b_override = nullptr,
                           const std::vector<word *> *dst_a_override =
                               nullptr) const;

  /**
   * @brief `EvaluateBabyStepBatch` with every baby step written into ONE
   * strided arena per baby index instead of per-ciphertext maps:
   * `arenas[j]` (ordered as `bs_indices_`) holds ciphertext c's b-part at
   * `c * 2 * total_words` and its a-part `total_words` later, at the
   * extended (q + aux) basis. The kernels and their order are exactly
   * `EvaluateBabyStepBatch`'s -- only the destination addresses differ --
   * so the words are the map form's. Feeds the batched giant step's
   * `PAccumBatchCt`, whose sources must be strided over the group.
   */
  void EvaluateBabyStepBatchArena(ConstContextPtr<word> context,
                                  std::vector<Dv> &arenas,
                                  const std::vector<const Ct *> &inputs,
                                  const EvkMap<word> &evk_map) const;

  // evaluation-related methods
  void EvaluateSingleAccum(ConstContextPtr<word> context, Ct &res,
                           const std::map<int, Ct> &bs,
                           const std::map<int, Pt> &pt_map,
                           bool inplace = false) const;
  void EvaluateFinalModDown(ConstContextPtr<word> context, Ct &res,
                            Ct &final_accum, int input_num_slots,
                            double input_scale) const;
  void EvaluateMinKSBabyStep(ConstContextPtr<word> context,
                             std::map<int, Ct> &bs, const Ct &input,
                             const EvkMap<word> &evk_map) const;
  void EvaluateMinKSGiantStep(ConstContextPtr<word> context, Ct &res,
                              const std::map<int, Ct> &bs,
                              const EvkMap<word> &evk_map) const;

  void EvaluateGiantStepOptimized(ConstContextPtr<word> context, Ct &res,
                                  const std::map<int, Ct> &bs,
                                  const EvkMap<word> &evk_map) const;

 public:
  HoistHandler(ConstContextPtr<word> context, const PlainHoistMap &hoist_map,
               int pt_level, double pt_scale, bool suppress_bs_swap = false);

  HoistHandler(const HoistHandler &) = delete;
  HoistHandler &operator=(const HoistHandler &) = delete;
  HoistHandler(HoistHandler &&) = default;
  ~HoistHandler();

  /**
   * @brief Write the compiled plaintexts and the baby/giant structure.
   *
   * A compiled handler is already in RNS form, so nothing here needs a
   * Context: what the constructor spends its time on -- `CompilePlaintexts`,
   * which encodes every diagonal of the matrix at `pt_level` -- is exactly
   * what is being cached, and the caller who reads it back is by construction
   * running against the parameter set the archive's identity names.
   *
   * This was the expensive half of the Llama leg's preparation while
   * `CompilePlaintexts` encoded on the host: the three `CiSinCConverter`s
   * were ~730 s of it, and the seam's per-layer T1 stages 10.4 s. With the
   * encoding on the device and the matrix construction in front of it on
   * every core, those are about a minute and 0.12 s.
   */
  void Save(ArchiveWriter &ar) const;

  /** @brief Rebuild a handler written by `Save`. */
  static HoistHandler Load(ArchiveReader &ar);

  /**
   * @brief The compiled plaintexts, `[giant step][baby step]`, as `Evaluate`
   * reads them. Read-only: for a caller comparing two compilations of the
   * same matrix coefficient by coefficient.
   */
  const std::map<int, std::map<int, Pt>> &GetPlaintexts() const {
    return hoist_pt_map_;
  }

  /**
   * @brief Move the compiled plaintexts off the device into host memory,
   * keeping everything else, and bring them back.
   *
   * A transform that is read once a layer -- the attention leg's three
   * converters, 15.3 GiB of plaintexts on ci16_35 -- need not stand on the
   * device between its uses: `Unstage` copies every diagonal down (once; the
   * host copies are kept) and frees the device buffers, `Stage` allocates
   * and copies them back. `Evaluate` on an unstaged handler is an error.
   * Both are const because residency is not a logical change; the leg's
   * `StageOperands` is the same idea for the projection weights.
   */
  void Unstage() const;
  void Stage() const;
  bool IsOnDevice() const { return on_device_; }
  /// Bytes the plaintexts occupy on the device when staged.
  size_t PlaintextBytes() const;

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const EvkMap<word> &evk_map, bool min_ks = false) const;
  void EvaluateBabyStep(ConstContextPtr<word> context, std::map<int, Ct> &bs,
                        const Ct &input, const EvkMap<word> &evk_map,
                        bool min_ks = false) const;
  /**
   * @brief `EvaluateBabyStep` over a GROUP of ciphertexts at one level: every
   * a-part gathered into one strided buffer, ONE `ModUpBatch` (the NTTs carry
   * the group on `blockIdx.z`), then ONE `BSFusedKernelBatch` with the key
   * tables shared across the group. Word for word the loop of serial calls
   * (`boot_test *WordForWord*`); `CHEDDAR_HOIST_BS_SERIAL=1` is the loop.
   * No min_ks form; `bs[i]` answers `inputs[i]`.
   */
  void EvaluateBabyStepBatch(ConstContextPtr<word> context,
                             const std::vector<std::map<int, Ct> *> &bs,
                             const std::vector<const Ct *> &inputs,
                             const EvkMap<word> &evk_map) const;
  /**
   * @brief The WHOLE transform over a GROUP of ciphertexts at one level and
   * scale: the group's baby steps as one `EvaluateBabyStepBatchArena`, the
   * plaintext accumulation as one `PAccumBatchCt` per giant step (the
   * diagonal table streamed once for the group instead of once per
   * ciphertext), and every giant key switch of every ciphertext through the
   * one batched rotate/fold (`GSComplexRotateFoldGroup` -- generic over
   * accumulator maps, nothing complex about it). Word for word the loop of
   * `Evaluate` calls: the same kernels per ciphertext, and the modular sums
   * accumulator by accumulator (order-exact mod p).
   *
   * Falls back to the serial loop when the group is 1, the transform is
   * trivial (single baby or single accumulator), the digit count exceeds
   * the fused kernel's cap, giant step 0 is absent, the inputs disagree in
   * level/scale/slots, or `CHEDDAR_HOIST_EVAL_SERIAL=1`. The group is
   * chunked so the baby arenas stay under `CHEDDAR_HOIST_EVAL_CHUNK_MIB`
   * (default 24576). No min_ks form; `res[i]` answers `inputs[i]`.
   */
  void EvaluateBatch(ConstContextPtr<word> context,
                     const std::vector<Ct *> &res,
                     const std::vector<const Ct *> &inputs,
                     const EvkMap<word> &evk_map) const;
  void EvaluateGiantStep(ConstContextPtr<word> context, Ct &res,
                         const std::map<int, Ct> &bs,
                         const EvkMap<word> &evk_map,
                         bool min_ks = false) const;

  /**
   * @brief The giant step of a COMPLEX matrix over a pair of real
   * ciphertexts, with every plaintext streamed once.
   *
   *     res_re = Re res_in_re - Im res_in_im
   *     res_im = Im res_in_re + Re res_in_im
   *
   * `re_h` and `im_h` hold the two matrix halves and must have been compiled
   * with identical baby/giant structure (same offsets, stride, bs, gs,
   * pre-rotations) -- ComplexLinearTransform builds them that way and this
   * asserts it. The minus sign lives in the kernel as a modular Sub, so no
   * negated copy of any baby step is ever materialised.
   *
   * `bs_im == nullptr` is the pair lift: one real input, both outputs.
   * `res_im == nullptr` is the drop back to real: both inputs, one output.
   * Baby steps come from `EvaluateBabyStep` of either handler -- the two
   * share baby structure, which is the point.
   *
   * No min_ks form: the caller falls back to four unfused passes there.
   */
  static void EvaluateGiantStepComplex(ConstContextPtr<word> context,
                                       Ct &res_re, Ct *res_im,
                                       const HoistHandler &re_h,
                                       const HoistHandler &im_h,
                                       const std::map<int, Ct> &bs_re,
                                       const std::map<int, Ct> *bs_im,
                                       const EvkMap<word> &evk_map);

  /**
   * @brief `EvaluateGiantStepComplex` over a GROUP of ciphertexts: ONE
   * `GSFusedComplexPAccumBatch` (the plaintext table streamed once for the
   * group), then the per-ciphertext rotate/fold exactly as the serial call
   * runs it. `res_re[i]`/`res_im[i]` answer `bs_re[i]`/`bs_im[i]`; `res_im`
   * nullptr is the drop to real, `bs_im` nullptr the lift from it, as the
   * serial call has them. Word for word the loop of serial calls.
   */
  static void EvaluateGiantStepComplexBatch(
      ConstContextPtr<word> context, std::vector<Ct *> &res_re,
      std::vector<Ct *> *res_im, const HoistHandler &re_h,
      const HoistHandler &im_h,
      const std::vector<const std::map<int, Ct> *> &bs_re,
      const std::vector<const std::map<int, Ct> *> *bs_im,
      const EvkMap<word> &evk_map);
};

}  // namespace cheddar
