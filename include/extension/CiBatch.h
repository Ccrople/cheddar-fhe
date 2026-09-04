#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/DeviceVector.h"
#include "core/Type.h"
#ifdef USE_CUBLAS
#include "core/PcmmBlas.h"
#endif

namespace cheddar {

/**
 * @brief The BATCHED layout of a Llama layer on R+: one channel per
 * ciphertext, the ciphertext's slots holding every (token, instance) pair of
 * that channel.
 *
 * ## Where this comes from
 *
 * The single-prompt layer (`CiLlamaLayer`, `CiSinCAttention`) packs 512
 * channels of 128 tokens into one ciphertext and reaches the channel axis
 * for a projection by `ModDecomp` -- [BAE]'s MLWE decomposition -- so that
 * the product can contract over module components, and pays `ModPack`'s
 * `rank` key switches per output ciphertext to come back. That is half the
 * cost of every projection (Doing.md 1.5dd), and it exists only because the
 * contracted axis lives INSIDE a ciphertext.
 *
 * [KANG] (ePrint 2025/1957) Algorithm 1 puts it the other way round. Read
 * `R_N` as a rank-`d` module over the subring `R_k`, `k | N`, and a matrix
 * over `R_k` as `d'` ciphertexts, one per COLUMN: `B + Toep(sk) A = M`.
 * Right-multiplying by a plaintext `U` over `R_k` is `(B U, A U)` -- a linear
 * combination of whole ciphertexts, no decomposition, no key switch, one
 * rescale. With a weight that is the SAME for every batch lane, `U`'s
 * entries are constants of `R_k` (the constant polynomial is the unique
 * preimage of a constant lane vector), so the product degenerates further
 * to a SCALAR combination of ciphertexts,
 *
 *     res[o] = sum_c  W[c][o] * ct[c]
 *
 * which is [BAE]'s own identity read with rows = ciphertexts, and which
 * commutes with EVERY map that acts within a ciphertext -- the slot
 * encoding, the SinC encoding, a bootstrap. So the projection needs no
 * encoding at all: it runs on slot-form ciphertexts as they stand, and only
 * the ciphertext-ciphertext products of the attention (Algorithm 4) need the
 * coefficient form. HEonGPU's batch-16 tree measured exactly this
 * (`LLAMA3_8B_LAYER_FLOW.md` 22.7: "project() COMMUTES with the bridge");
 * this file is the Cheddar form of the same layout.
 *
 * ## The layout
 *
 * A channel is a ciphertext. Its `num_slots` real slots (R+ has `degree`
 * of them) hold `num_tokens` tokens of `num_instances = num_slots /
 * num_tokens` INDEPENDENT prompts: on `ci16_35` at T = 128 that is 512
 * prompts a ciphertext. `Slot(token, instance)` is the one place the map is
 * written; every operator here is either slot-wise (RMSNorm's polynomial,
 * SiLU, the softmax's exp, RoPE's per-token angles) or ciphertext-wise (the
 * projections, the channel reductions, the residual adds), so nothing but
 * the host packing and the per-token plaintexts read it.
 *
 * What the batch buys, and what it does not:
 *
 *  - the projections have NO key switch and no encoding conversion -- the
 *    int8 GEMM over `in` whole ciphertexts is the whole cost, and it is
 *    exact mod q (`PcmmBlasHandler`);
 *  - a channel reduction (RMSNorm's mean square, softmax's row sum) is a
 *    SUM OF CIPHERTEXTS -- no rotation, no key, no level;
 *  - the slot-wise non-linear operators cost what they cost per ciphertext,
 *    and a ciphertext now carries `num_instances` prompts: their cost per
 *    prompt falls by the batch and their cost per layer does not;
 *  - the residual stream is `model` ciphertexts (4096) rather than 8: the
 *    memory and the bootstrap count scale with the batch, which is what a
 *    batch is.
 *
 * ## Host helpers
 *
 * `Pack`/`Unpack` move one channel between a `[instance][token]` table and
 * the slot message, so that a test's reference can be a plain host loop and
 * the map is exercised by every read.
 */
struct CiBatchLayout {
  int num_slots = 0;
  int num_tokens = 0;
  int num_instances = 0;
  //! The CC-MM chain's addressing, when the layout serves one (0 = the
  //! plain map): an instance is (group, lane) with `lanes` lanes per
  //! group, and a channel ciphertext's SinC blocks at sub-degree `lanes`
  //! are the (token, group) pairs in the order the chain's forward
  //! converter reads them -- `CiSwitchedCcmmLayout::LocateSlot`'s primary
  //! address with row = token and column % rank = group.
  int lanes = 0;
  int rank = 0;

  CiBatchLayout() = default;
  CiBatchLayout(int num_slots, int num_tokens);
  //! The chain-addressed layout: `num_slots = num_tokens * rank * lanes`.
  CiBatchLayout(int num_slots, int num_tokens, int lanes, int rank);

  //! The slot of token `token` of prompt `instance`. Plain: the instance
  //! fast, the token slow. Chain-addressed: block `BitRev(token * rank +
  //! group)` of the stride-`lanes` block index, lane untouched. Either way
  //! a per-token constant is a plaintext, and nothing but the host packing
  //! and those plaintexts reads this.
  int Slot(int token, int instance) const {
    if (lanes == 0) return token * num_instances + instance;
    const int group = instance / lanes, lane = instance % lanes;
    return BlockOf(token, group) * lanes + lane;
  }
  //! Chain-addressed only: the SinC block of (token, group).
  int BlockOf(int token, int group) const;

  //! `values[instance * num_tokens + token]` -> the slot message.
  void Pack(std::vector<Complex> &msg, const std::vector<double> &values) const;
  //! The inverse: the real parts of the slot message, `[instance][token]`.
  void Unpack(std::vector<double> &values,
              const std::vector<Complex> &msg) const;
  //! The slot message of a per-TOKEN constant (RoPE's angles, a sink
  //! rescale): `per_token[token]` at every instance.
  void PackPerToken(std::vector<Complex> &msg,
                    const std::vector<double> &per_token) const;
};

#ifdef USE_CUBLAS

/**
 * @brief [KANG] Algorithm 1 with a batch-invariant weight: every projection
 * of the batched layer, as the int8 GEMM over whole ciphertexts.
 *
 * ## What it computes
 *
 *     res[o] = Rescale( sum_c  W[c][o] * x[c] ),   0 <= o < out
 *
 * `x` is `in` ciphertexts at `level`, `W` the model's own `[in][out]` f32
 * tensor on the device (`reference/export_layers.py`'s layout, the same
 * blob `CiLlamaLayer::DeviceWeights` reads), and `res` lands one level
 * below, CANONICAL when the weight was encoded at `GetScale(level)`:
 *
 *     GetScale(L) * GetScale(L) / GetRescalePrimeProd(L) = GetScale(L - 1).
 *
 * That is `PcmmBlasHandler::Multiply` on RLWE ciphertexts -- the overload
 * that was never on the single-prompt layer's path, because that layer's
 * contracted axis is a module component -- followed by one rescale each.
 * No `ModDecomp`, no `ModPack`, no ring switch, no key of any kind; the
 * product is bit-exact modulo the primes, so the only error is the weight's
 * rounding to the scale and the ciphertexts' own noise.
 *
 * ## Tiles
 *
 * A weight is split into int8 pieces once (`Prepare`) and held on the
 * device in row tiles of `Config::rows_per_tile` output channels. The tile
 * bounds two things: the int32 accumulator the GEMM needs (`pieces^2 * rows
 * * chunk` words) and the output ciphertexts alive before their rescale.
 * The SOURCE is split once per `Project` and shared by every tile
 * (`PcmmBlasHandler::PrepareSource` on RLWE ciphertexts), so the tiling
 * costs no repeated gather.
 *
 * ## Residency
 *
 * Device-resident operands, by name, exactly as `Prepare` left them; a
 * layer's seven projections at level 1 on `ci16_35` are ~2 GiB of pieces.
 * Host staging and prefetch (`CoeffLinearLeg`'s `kHost`, `CiLayerPrefetch`)
 * are the same machinery and are not repeated here until the batched layer
 * runs more than one layer.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CiBatchProjection {
 private:
  using Ct = Ciphertext<word>;
  using SplitMatrix = typename PcmmBlasHandler<word>::SplitMatrix;
  using SplitSource = typename PcmmBlasHandler<word>::SplitSource;

 public:
  struct Config {
    //! Output channels per tile. 2048 at the model's width is 14336 / 7
    //! tiles for gate and up, and a 2048 x 4096 x 4 x limbs int8 slab.
    int rows_per_tile = 2048;
    bool verbose = false;
  };

  //! One converted weight: its tiles, and what they were cut for.
  struct Operand {
    int in = 0;
    int out = 0;
    int level = 0;
    double w_scale = 1.0;
    double weight_scale = 0.0;  //!< the encoding scale, `GetScale(level)`
    int rows_per_tile = 0;
    std::vector<SplitMatrix> tiles;
    size_t bytes = 0;
  };

  CiBatchProjection(ConstContextPtr<word> context, const Config &cfg);

  // disable copying (or moving also)
  CiBatchProjection(const CiBatchProjection &) = delete;
  CiBatchProjection &operator=(const CiBatchProjection &) = delete;

  /**
   * @brief Convert one weight: encode `w_scale * W` at `GetScale(level)`
   * on the device (`GpuEncoder::EncodeResiduesGathered`) and split it into
   * int8 pieces, tile by tile.
   *
   * @param name the operand's name; a repeated name replaces the operand
   * @param tensor device, `[in][out]` row-major f32
   * @param level the level the INPUT ciphertexts will be at
   * @param w_scale a factor folded into the weight (a calibration's
   *        stream or gate scale, a SiLU range's reciprocal, an RMSNorm gain
   *        folded per input channel is the caller's -- see `FoldGain`)
   */
  /**
   * @param input_scale_ratio the inputs' recorded scale over the level's
   *        canonical one -- 2 for what comes back from the CC-MM chain,
   *        whose descent doubles the recorded scale (Doing.md 1.5bk). The
   *        weight is encoded at `GetScale(level) / ratio` so that the
   *        product still lands canonical one level down.
   */
  void Prepare(const std::string &name, const float *tensor, int in, int out,
               int level, double w_scale = 1.0, double input_scale_ratio = 1.0);

  bool Has(const std::string &name) const {
    return operands_.count(name) != 0;
  }
  void Release(const std::string &name) { operands_.erase(name); }
  const Operand &Get(const std::string &name) const {
    return operands_.at(name);
  }
  //! Device bytes every prepared operand holds.
  size_t Bytes() const;

  /**
   * @brief `res = Rescale(x W)`, `x` the `in` channel ciphertexts at the
   * operand's level, `res` the `out` channel ciphertexts one level below.
   */
  void Project(std::vector<Ct> &res, const std::vector<Ct> &x,
               const std::string &name) const;

  /**
   * @brief A source split once, for every tile of every operand that reads
   * the same `in` ciphertexts at the same level -- the feed-forward's gate
   * and up, tile by tile, against one split of the normalised stream.
   */
  struct Source {
    SplitSource split;
    int in = 0;
    int level = 0;
  };
  //! Split `x` for the operand `name` (whose pieces and level it takes), cut
  //! for `rows_per_tile`; any operand with the same `in` and level may then
  //! be projected from it.
  void Split(Source &src, const std::vector<Ct> &x,
             const std::string &name) const;
  //! The same split built one column at a time, so the inputs never all
  //! exist together: `BeginSplit` for `in` inputs at `level` (their
  //! recorded scale `scale`; 0 = the level's canonical one), then
  //! `AddColumn` per input, which may be dropped once added. Any operand
  //! with that `in` and level projects from it, the product's scale
  //! `u.scale * scale / rescale_prod` -- so an off-canonical stream
  //! projects canonically when its operand was prepared with the
  //! matching `input_scale_ratio`.
  void BeginSplit(Source &src, int in, int level, int num_slots,
                  double scale = 0.0) const;
  void AddColumn(Source &src, int col, const Ct &x) const;
  int NumTiles(const std::string &name) const {
    return static_cast<int>(operands_.at(name).tiles.size());
  }
  //! First output channel of `tile` of `name`.
  int TileStart(const std::string &name, int tile) const {
    return tile * operands_.at(name).rows_per_tile;
  }
  //! The output channels of one tile of `name` from a split source: one
  //! GEMM and its rescales.
  void Project(std::vector<Ct> &res, const Source &src,
               const std::string &name, int tile) const;

  /**
   * @brief `diag(g) W` on the host side of a tensor: RMSNorm's per-channel
   * gain folded into the projection that reads the normalised stream,
   * exactly (`W^T diag(g) y = (diag(g) W)^T y`). The folded tensor is a new
   * device buffer the caller owns.
   */
  static void FoldGain(DeviceVector<float> &res, const float *tensor, int in,
                       int out, const std::vector<double> &gain);

 private:
  //! `CHEDDAR_BATCH_RESCALE_SERIAL=1`: one `Rescale` per output instead
  //! of the tile's batched one (the A/B; the words are the same).
  static bool RescaleSerial();

  ConstContextPtr<word> context_;
  Config cfg_;
  std::unique_ptr<PcmmBlasHandler<word>> blas_;
  std::map<std::string, Operand> operands_;
  //! Scratch for the gathered encode: the residues of one tile, prime-major.
  mutable DeviceVector<word> residues_;
  mutable DeviceVector<int32_t> row_map_, col_map_;
  //! A tile's GEMM output and its batched rescale, kept between calls.
  mutable DeviceVector<word> prod_, rescaled_;
};

#endif  // USE_CUBLAS

}  // namespace cheddar
