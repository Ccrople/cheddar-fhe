#pragma once

#include <vector>

#include "core/DeviceVector.h"
#include "core/Parameter.h"

// ElementWise functions all have the same structure, which we

namespace cheddar {

/**
 * @brief This class is just a collection of element-wise functions
 *
 * @tparam word either uint32_t or uint64_t
 */
template <typename word>
class ElementWiseHandler {
 private:
  const Parameter<word> &param_;

  static constexpr int kernel_block_dim_ = 256;
  static constexpr int max_num_poly_ = 3;
  static constexpr int max_num_accum_ = 8;

  uint32_t PermuteAmountToGaloisFactor(int permute_amount) const;
  void AssertNPMatch(std::vector<DvView<word>> &dst, const NPInfo &np) const;

  template <bool const_accum>
  void CPAccumWorker(std::vector<DvView<word>> &dst, const NPInfo &np,
                     const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
                     const std::vector<DvConstView<word>> &common_srcs) const;

  template <bool const_accum>
  void CPAccumWorkerBatch(
      std::vector<DvView<word>> &dst, const NPInfo &np,
      const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
      const std::vector<DvConstView<word>> &common_srcs, int batch,
      size_t dst_stride, const std::vector<size_t> &src_strides) const;

  void PermuteAccumWorker(
      std::vector<DvView<word>> &dst, const NPInfo &np,
      const std::vector<int> &permute_amounts,
      const std::vector<std::vector<DvConstView<word>>> &srcs) const;

 public:
  ElementWiseHandler(const Parameter<word> &param);

  // disable copying (or moving also)
  ElementWiseHandler(const ElementWiseHandler &) = delete;
  ElementWiseHandler &operator=(const ElementWiseHandler &) = delete;

  // for forwarding purposes
  ElementWiseHandler(ElementWiseHandler &&) = default;

  // ----- Basic functions ----- //
  // dst = src1 + src2
  void Add(std::vector<DvView<word>> &dst, const NPInfo &np,
           const std::vector<DvConstView<word>> &src1,
           const std::vector<DvConstView<word>> &src2) const;
  // dst = src1 - src2
  void Sub(std::vector<DvView<word>> &dst, const NPInfo &np,
           const std::vector<DvConstView<word>> &src1,
           const std::vector<DvConstView<word>> &src2) const;
  // dst = -src1
  void Neg(std::vector<DvView<word>> &dst, const NPInfo &np,
           const std::vector<DvConstView<word>> &src1) const;
  // dst = src1 * src2
  void Mult(std::vector<DvView<word>> &dst, const NPInfo &np,
            const std::vector<DvConstView<word>> &src1,
            const std::vector<DvConstView<word>> &src2) const;
  // dst = src1 * src2 (pt)
  void PMult(std::vector<DvView<word>> &dst, const NPInfo &np,
             const std::vector<DvConstView<word>> &src1,
             const DvConstView<word> &src2) const;

  // dst = src1 + const_src;
  void AddConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                const std::vector<DvConstView<word>> &src1,
                const DvConstView<word> &src_const) const;
  // dst = src1 - const_src;
  void SubConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                const std::vector<DvConstView<word>> &src1,
                const DvConstView<word> &src_const) const;
  // dst = const_src - src1;
  void SubOppositeConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                        const std::vector<DvConstView<word>> &src1,
                        const DvConstView<word> &src_const) const;
  // dst = src1 * const_src;
  void MultConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                 const std::vector<DvConstView<word>> &src1,
                 const DvConstView<word> &src_const) const;

  void Tensor(std::vector<DvView<word>> &dst, const NPInfo &np,
              const std::vector<DvConstView<word>> &src1,
              const std::vector<DvConstView<word>> &src2) const;

  void Permute(std::vector<DvView<word>> &dst, const NPInfo &np,
               int permute_amount,
               const std::vector<DvConstView<word>> &src1) const;

  // ----- Accumulation functions ----- //

  void PermuteAccum(
      std::vector<DvView<word>> &dst, const NPInfo &np,
      const std::vector<int> &permute_amounts,
      const std::vector<std::vector<DvConstView<word>>> &srcs) const;
  void Accum(std::vector<DvView<word>> &dst, const NPInfo &np,
             const std::vector<std::vector<DvConstView<word>>> &srcs) const;
  void PAccum(std::vector<DvView<word>> &dst, const NPInfo &np,
              const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
              const std::vector<DvConstView<word>> &pt_srcs) const;
  void CAccum(std::vector<DvView<word>> &dst, const NPInfo &np,
              const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
              const std::vector<DvConstView<word>> &const_srcs) const;

  // Special functions, only use it when you know what you are doing
  /**
   * @brief Lift from the level-zero base to the basis of `target_level`.
   *
   * The kernel iterates over the primes of `dst`, so the target basis is not
   * special -- it was only ever the parameter set's maximum because that is
   * what a full bootstrap needs. Climbing no higher than the levels actually
   * required makes every limb operation in CoeffToSlot, EvalMod and
   * SlotToCoeff shorter.
   *
   * @param target_level the level to lift to; -1 means the parameter set's
   * maximum, which is the previous behaviour
   */
  void ModUpToLevel(DvView<word> &dst, const DvConstView<word> &src1,
                    int target_level = -1) const;

  void MultImaginaryUnit(std::vector<DvView<word>> &dst, const NPInfo &np,
                         const std::vector<DvConstView<word>> &src1,
                         const DvConstView<word> &src_i_unit) const;

  // ----- Batched-ciphertext elementwise (a polynomial evaluation over a
  // batch of ciphertexts at one level: EvalMod) ----- //
  //
  // Each call runs the serial method's kernel ONCE with gridDim.z = batch.
  // The views describe ciphertext 0 of each batch, exactly as the serial call
  // would receive them; ciphertext b's polynomial j sits at the view's
  // pointer plus b * that buffer's stride (in words). With batch 1 and
  // stride 0 every kernel is the serial one, word for word.

  // dst = src1 + src2, over a batch.
  void AddBatchCt(std::vector<DvView<word>> &dst, const NPInfo &np,
                  const std::vector<DvConstView<word>> &src1,
                  const std::vector<DvConstView<word>> &src2, int batch,
                  size_t dst_stride, size_t src1_stride,
                  size_t src2_stride) const;
  // dst = src1 + const, over a batch (the constant is shared).
  void AddConstBatchCt(std::vector<DvView<word>> &dst, const NPInfo &np,
                       const std::vector<DvConstView<word>> &src1,
                       const DvConstView<word> &src_const, int batch,
                       size_t dst_stride, size_t src1_stride) const;
  // dst = src1 * const, over a batch (the constant is shared).
  void MultConstBatchCt(std::vector<DvView<word>> &dst, const NPInfo &np,
                        const std::vector<DvConstView<word>> &src1,
                        const DvConstView<word> &src_const, int batch,
                        size_t dst_stride, size_t src1_stride) const;
  // The tensor product (b, a, r) of two batches; src1 == src2 (pointerwise,
  // with equal strides) dispatches the square kernel as the serial call does.
  void TensorBatchCt(std::vector<DvView<word>> &dst, const NPInfo &np,
                     const std::vector<DvConstView<word>> &src1,
                     const std::vector<DvConstView<word>> &src2, int batch,
                     size_t dst_stride, size_t src1_stride,
                     size_t src2_stride) const;
  // CAccum over a batch: per ciphertext b of the batch,
  // dst_b = sum_k const_k * ct_srcs[k]_b (+ ct_srcs.back()_b when it has one
  // more entry than const_srcs, exactly as the serial CAccum). Each source's
  // batch stride is src_strides[k]; the constants are shared.
  void CAccumBatchCt(std::vector<DvView<word>> &dst, const NPInfo &np,
                     const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
                     const std::vector<DvConstView<word>> &const_srcs,
                     int batch, size_t dst_stride,
                     const std::vector<size_t> &src_strides) const;
  // PAccum over a batch: per ciphertext b of the batch,
  // dst_b = sum_k pt_k * ct_srcs[k]_b, the plaintexts SHARED across the
  // batch. The serial worker's launches with gridDim.z = batch (the same
  // chunking, source order and kernels), so the words are the serial
  // PAccum's per ciphertext.
  void PAccumBatchCt(std::vector<DvView<word>> &dst, const NPInfo &np,
                     const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
                     const std::vector<DvConstView<word>> &pt_srcs, int batch,
                     size_t dst_stride,
                     const std::vector<size_t> &src_strides) const;

  // ----- Batched key switching (Cmt, and anything else that switches many
  // ciphertexts with many keys at one level) ----- //

  static constexpr int max_batch_digits_ = 8;

  /**
   * @brief The key multiply of `batch` key switches in one launch, each
   * switch with its own key: what `PAccum` over the digits followed by
   * `CAccum` of `p_prod * bx` computes per ciphertext, word for word.
   *
   * Switch `b` accumulates into `dst + b * dst_batch_stride`: its b-part on
   * the first `ext_words = np.GetNumTotal() * degree` words and its a-part on
   * the next. `modup[i] + b * modup_batch_stride` is its digit `i`
   * (`ModSwitchHandler::ModUpBatch`'s layout), `key_table` holds, per switch
   * and per digit, FOUR pointers: the key's b and a q limbs (already offset
   * by the terminal-prime padding, as `EvaluationKey::ConstViewVector(i,
   * offset)` gives them) and its b and a auxiliary limbs -- every key with
   * its own limb layout. `bx + b * bx_batch_stride` is the switch's original b-part, added in
   * times the per-prime `p_prod` on the q limbs; `add_a` (null or the same
   * layout) is added to the a-part the same way -- the relinearization's
   * (D0, D1 + D2), whose P-multiple the mod-down returns exactly.
   */
  void KeyMultBatch(word *dst, int dst_batch_stride, const NPInfo &np,
                    const std::vector<const word *> &modup,
                    int modup_batch_stride, const word *const *key_table,
                    const word *bx, const word *add_a, int bx_batch_stride,
                    const word *p_prod, int batch) const;

  /**
   * @brief `Permute` of `batch` ciphertexts, each by its own automorphism, in
   * one launch. Ciphertext `b` has `num_poly` polynomials of `np` limbs at
   * `src + b * batch_stride + p * poly_stride`, the result lands at the same
   * offsets of `dst`; `galois_factors[b]` / `galois_offsets[b]` are its map
   * in `Parameter::GetGaloisOffset`'s terms. No auxiliary limbs.
   */
  void PermuteBatch(word *dst, const word *src, int batch_stride,
                    int poly_stride, int num_poly, const NPInfo &np,
                    const uint32_t *galois_factors,
                    const uint32_t *galois_offsets, int batch) const;
};

}  // namespace cheddar
