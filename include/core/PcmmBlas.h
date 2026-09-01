#pragma once

#ifdef USE_CUBLAS

#include <cublas_v2.h>

#include <vector>

#include "core/Container.h"
#include "core/Parameter.h"
#include "core/Pcmm.h"

namespace cheddar {

/**
 * @brief PCMM through cuBLAS, which is where [SYLPH]'s speed comes from.
 *
 * ## Why this is a GEMM at all
 *
 * [BAE] section 1.1: `(U.A) . Toep(sk) + (U.B) = U.M`, so PCMM *is* two
 * plaintext matrix products. `PcmmAccum` already computes, per RNS prime,
 * `DST[rows x degree] = U[rows x cols] * SRC[cols x degree]` -- a GEMM written
 * by hand. [SYLPH] section 2.2 credits its performance to "offloading most
 * computations to fast BLAS libraries", and we were not.
 *
 * ## Why int8 and not fp64
 *
 * cuBLAS does no modular arithmetic, so an exact product needs operands split
 * into pieces small enough that nothing overflows the accumulator. Measured on
 * this A6000 for one prime slice (128 x 4096 by 4096 x 4096, 4.29 GFLOP):
 *
 *     fp64           7.315 ms    0.6 TFLOP/s   3 splits -> 131.7 ms
 *     fp32           0.205 ms   20.9 TFLOP/s   not exact for 30-bit residues
 *     int8 -> int32  0.086 ms   49.9 TOP/s     -> a few ms
 *
 * against `PcmmAccum`'s 36.7 ms for the same work. **fp64 is a dead end here**:
 * the A6000 runs it at 1/32 of fp32, so `cublasDgemm` would be slower than the
 * kernel it replaces. Measuring this before building the machinery is the only
 * reason it did not get built the wrong way.
 *
 * ## The splitting
 *
 * Residues are below a prime of up to 30 bits. Each value becomes
 * `S = PiecesFor(bits)` BALANCED digits of 8 bits in `[-128, 127]` -- four
 * for a 30-bit prime, where 7-bit unsigned pieces needed five and therefore
 * 25 piece products instead of 16 (Doing.md 3.19: the product is 36 % of the
 * layer's kernel time, so the digit width is not a detail). A piece product
 * is within `+-2^14`, and summing `cols` of them (8704 < 2^14) stays within
 * `+-2^28`, inside int32 with room to spare.
 *
 * Pieces `k` of U and `l` of the source contribute at shift `8 * (k + l)`.
 * Each of the `S^2` products lands in its own int32 slab (`beta = 0`: cuBLAS's
 * integer path does not guarantee an in-place accumulation), and one kernel
 * recombines the slabs modulo the prime, lifting a negative sum by a multiple
 * of the prime first. The recombination, which needs 64-bit intermediates,
 * happens once per output element rather than once per product.
 *
 * ## The layout is the whole performance story
 *
 * int8 cuBLAS has two paths on Ampere and only one of them is the tensor core.
 * Measured on this A100 at the layer's own shape, m = 32768, n = 256, k = 4096:
 *
 *     NN   0.965 ms    71.2 TOPS   ampere_igemm_int8_128x128_ldg4_nn
 *     NT   0.973 ms    70.6 TOPS   ampere_igemm_int8_128x128_ldg4_nt
 *     TT   0.966 ms    71.1 TOPS   ampere_igemm_int8_128x128_ldg4_tt
 *     TN   0.175 ms   392.7 TOPS   cutlass_80_tensorop_i16832gemm_s8_..._tn
 *
 * 71 TOPS is not a poor result, it is exactly this card's DP4A ceiling
 * (19.5 fp32 TFLOP x 4). Three of the four layouts fall on the SIMT integer
 * kernel; **only TN reaches IMMA**, and it does so because TN is the one
 * arrangement where the reduction index is contiguous in *both* operands.
 * U already satisfies it -- `SplitMatrixFrom` writes row-major `[rows][cols]`,
 * which read as a column-major `cols x rows` is exactly `B` for `OP_N`. The
 * source did not, so the gather transposes as it splits. Batching the sixteen
 * shift groups into one `cublasGemmStridedBatchedEx` was measured at the same
 * time and is worth nothing (421.8 against 411.9 TOPS): at this size the
 * launches are already long enough to hide their own overhead.
 *
 * ## Correctness
 *
 * `PcmmHandler::Multiply` stays untouched and is the reference: this path must
 * agree with it exactly, not approximately. `U` is taken as plain residues here
 * rather than in Montgomery form, and both produce plain residues, so equality
 * is bit-for-bit and a test can assert it.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class PcmmBlasHandler {
 private:
  using Ct = Ciphertext<word>;

  const Parameter<word> &param_;
  cublasHandle_t handle_ = nullptr;

  // 8 bits per int8 piece, as BALANCED digits in [-128, 127]: the digit `p`
  // of `r` is byte `p` of `r + 0x80..80` minus 128, so a 30-bit residue is
  // four pieces and a product 16 GEMMs instead of the 25 that 7-bit unsigned
  // pieces cost (Doing.md 3.19). The int32 group sums are then signed, and
  // `CombineGroups` lifts a negative one by a multiple of the prime.
  static constexpr int kPieceBits = 8;
  //! Pieces that cover `max_bits`-bit residues with the bias trick: the
  //! biased value `r + 0x80..80` needs one more bit than `r`.
  static int PiecesFor(int max_bits) {
    return (max_bits + 1 + kPieceBits - 1) / kPieceBits;
  }

 public:
  /** @brief U split into int8 pieces, one set per RNS prime. Setup-time. */
  struct SplitMatrix {
    int rows = 0;
    int cols = 0;
    int pieces = 0;   // S
    double scale = 1.0;
    NPInfo np;
    // pieces * primes * rows * cols int8 values, piece-major
    DeviceVector<int8_t> data;
    // A prefetched operand does not own its device bytes: they sit in a
    // staging arena outside the pool (`CoeffLinearLeg`'s), and this is the
    // window onto them while the operand is staged. Null when `data` holds
    // the bytes, or when the operand is off the device.
    const int8_t *view = nullptr;

    const int8_t *Ptr() const { return view != nullptr ? view : data.data(); }
    /// The bytes the split holds or will hold: `pieces * primes * rows *
    /// cols`, a function of the shape alone.
    size_t Bytes() const {
      return static_cast<size_t>(pieces) * np.GetNumTotal() * rows * cols;
    }
  };

  /**
   * @brief The source ciphertexts split into int8 pieces, once per tile.
   *
   * `CoeffLinearLeg` holds one ModDecomp for a whole tile and multiplies it by
   * one plaintext matrix per ModPack group -- sixteen of them for Q or O and
   * fifty-six for the FFN's gate and up. The split of the source does not
   * depend on which group is being computed, so doing it inside `Multiply` did
   * it `groups` times over. The whole layer's gather traffic was 1.4 TB for a
   * job whose actual source is 43 GB.
   *
   * Holding it costs nothing net: the split is `pieces` bytes per word against
   * a word of four, so at four pieces it is exactly the size of the ciphertexts
   * it came from, and the caller drops those as soon as this exists.
   *
   * The buffers are per (prime, chunk) rather than one block because
   * `DeviceVector` is indexed by `int` and a sixteen-parent tile's a-part is
   * 3.2e9 pieces.
   */
  struct SplitSource {
    int cols = 0;
    int pieces = 0;
    int rank = 0;
    int degree = 0;  // the small degree, i.e. the b-part's vector length
    double scale = 0.0;
    NPInfo np;
    int chunk_b = 0;
    int chunk_a = 0;
    // The RLWE form (`PrepareSource(..., const std::vector<Ct> &, rows)`)
    // records the row count its chunks were cut for and the inputs' slot
    // count, so that a product against a SMALLER row tile can reuse it: the
    // chunk is a function of the rows only through the accumulator budget,
    // and a smaller tile needs less of it, never more.
    int rows = 0;
    int num_slots = 0;
    std::vector<DeviceVector<int8_t>> b_data;  // [prime * chunks + chunk]
    std::vector<DeviceVector<int8_t>> a_data;

    /** @brief Device bytes held. */
    size_t Bytes() const {
      size_t n = 0;
      for (const auto &d : b_data) n += d.size();
      for (const auto &d : a_data) n += d.size();
      return n;
    }
  };

  explicit PcmmBlasHandler(const Parameter<word> &param);
  ~PcmmBlasHandler();

  PcmmBlasHandler(const PcmmBlasHandler &) = delete;
  PcmmBlasHandler &operator=(const PcmmBlasHandler &) = delete;

  /**
   * @brief Split the plaintext matrix into int8 pieces, once, offline.
   *
   * This is [SYLPH] section 5.3's model conversion: the weights are put into
   * the form the product consumes and kept on the GPU (section 5.1). It takes
   * the same `values` `PcmmHandler::EncodeMatrix` does rather than an encoded
   * matrix, because the GEMM wants plain residues and `PlainMatrix` holds
   * Montgomery form.
   */
  void SplitMatrixFrom(SplitMatrix &res, int level, double scale,
                       const std::vector<double> &values, int rows, int cols,
                       int num_aux = 0) const;

  /**
   * @brief The same split, from residues already on the device.
   *
   * `SplitMatrixFrom` is a host loop -- a `BigInt` reduction per (value,
   * prime) pair -- and at the model's width it is the whole of a layer's
   * `pcmm: convert weights` row. `GpuEncoder::EncodeResiduesGathered` writes
   * the same plain residues on the device, prime-major, and this splits them
   * into the piece-major int8 layout `Multiply` consumes without a host byte
   * in between. Same `pieces`, same bytes, same values.
   *
   * @param residues device, `np.GetNumTotal() * rows * cols` words, prime-major
   */
  void SplitMatrixFromResidues(SplitMatrix &res, int level, double scale,
                               const word *residues, int rows, int cols,
                               int num_aux = 0) const;

  /** @brief Device bytes the split matrix holds. */
  static size_t SplitBytes(const SplitMatrix &m) { return m.Bytes(); }

  /**
   * @brief Fill a `SplitMatrix`'s shape -- pieces, NPInfo, scale -- without
   * building it, so a caller can size a buffer for it before any encode.
   */
  void DescribeSplit(SplitMatrix &res, int level, double scale, int rows,
                     int cols, int num_aux = 0) const;

  /**
   * @brief `SplitMatrixFromResidues` into memory the caller owns, on the
   * caller's stream: the prefetch path, whose bytes go to a pinned mirror
   * on the copy stream and never live in the pool. `res.data` stays empty
   * and `res.view` is left for the caller to set when the operand is
   * staged.
   */
  void SplitResiduesInto(SplitMatrix &res, int level, double scale,
                         const word *residues, int rows, int cols, int8_t *dst,
                         cudaStream_t stream, int num_aux = 0) const;

  /**
   * @brief res[i] = sum_j u[i][j] * cts[j], through cuBLAS.
   *
   * Same contract as `PcmmHandler::Multiply`: no rescaling, result at
   * `u.scale * cts[0].scale`.
   */
  void Multiply(std::vector<Ct> &res, const SplitMatrix &u,
                const std::vector<Ct> &cts) const;

  /**
   * @brief The same product on MLWE ciphertexts, which is the format the Llama
   * projections actually run in.
   *
   * `CoeffLinearLeg` reaches the product by `ModDecomp`, so its operands are
   * `MlweCiphertext`, not `Ciphertext`, and the RLWE overload above was never
   * on the layer's path. `PcmmHandler` carries the same pair of overloads for
   * the same reason, and the second one is the one the block calls.
   *
   * No new machinery is involved: an MLWE ciphertext is a rank-k `a_` plus a
   * single `b_`, both laid out with all limb data contiguous, so each is one
   * more plaintext product of the same shape -- the b-part over vectors of
   * `degree` words per limb and the a-part over `rank * degree`. That is
   * exactly the substitution `PcmmHandler::Multiply(MLWE)` makes when it
   * passes `a_stride` to `PcmmAccum` in place of the degree.
   *
   * @param res output, resized to u.rows
   * @param u split plaintext matrix, u.cols must equal cts.size()
   * @param cts input MLWE ciphertexts, sharing one NP, rank and degree
   */
  void Multiply(std::vector<MlweCiphertext<word>> &res, const SplitMatrix &u,
                const std::vector<MlweCiphertext<word>> &cts) const;

  /**
   * @brief Split the source ciphertexts once, for a whole tile.
   *
   * `u` is read only for the piece count and the NP, so any of the tile's
   * plaintext matrices will do -- they all share both.
   */
  void PrepareSource(SplitSource &res, const SplitMatrix &u,
                     const std::vector<MlweCiphertext<word>> &cts) const;

  /**
   * @brief The RLWE form of `PrepareSource`: whole ciphertexts split once,
   * for a product taken against several row tiles of one weight.
   *
   * This is the [KANG] Algorithm 1 form of the projection (`CiBatchProjection`):
   * the contracted channel sits on the CIPHERTEXT axis, so the source is the
   * ciphertexts themselves -- rank 1, vector length the ring degree -- and a
   * weight wider than one tile's worth of output rows is applied tile by tile
   * against the same split. `rows` is the LARGEST tile the split will meet
   * (0 = `u.rows`); the chunk is cut for it, and `Multiply(res, u, src)` then
   * accepts any `u` with `u.rows <= rows` and the same columns, pieces and NP.
   */
  void PrepareSource(SplitSource &res, const SplitMatrix &u,
                     const std::vector<Ct> &cts, int rows = 0) const;

  /**
   * @brief The RLWE split built one column at a time, so that the
   * ciphertexts need never all exist at once: `PrepareSourceBegin` sizes the
   * split for `cols` inputs at `level` (cut for `rows`), `SplitSourceColumn`
   * writes column `col`'s pieces from one ciphertext, which may then be
   * dropped. At the model's width the source of a projection is as many
   * bytes as its split, and the batched layer produces its columns one by
   * one (`CiBatchLayer::NormTurn`), so this halves the peak.
   *
   * @param scale the scale every column will carry (checked per column)
   */
  void PrepareSourceBegin(SplitSource &res, int level, int cols, int rows,
                          double scale, int num_slots) const;
  void SplitSourceColumn(SplitSource &res, int col, const Ct &ct) const;

  /**
   * @brief The RLWE product against a source split by the overload above.
   * Same contract as `Multiply(res, u, cts)`: no rescaling, the result at
   * `u.scale * cts[0].scale`. Word for word what that overload computes,
   * which now runs through here.
   */
  void Multiply(std::vector<Ct> &res, const SplitMatrix &u,
                const SplitSource &src) const;

  /**
   * @brief The same RLWE product into ONE buffer: row `i`'s b-part at
   * `dst + i * dst_ct_stride`, its a-part `q_words` further, `q_words =
   * np.GetNumTotal() * degree`. What a batched rescale over the tile wants
   * (`ModSwitchHandler::RescaleBatch` reads polynomials at one stride), and
   * the same words `Multiply` would have put in `u.rows` ciphertexts.
   */
  void MultiplyInto(word *dst, int dst_ct_stride, const SplitMatrix &u,
                    const SplitSource &src) const;

  /**
   * @brief The MLWE product against a source that is already split.
   *
   * Identical in result to the overload above; that one is this one with a
   * `PrepareSource` in front, kept for callers with a single product to do.
   */
  void Multiply(std::vector<MlweCiphertext<word>> &res, const SplitMatrix &u,
                const SplitSource &src) const;

 private:
  /**
   * @brief How many words of the vector axis one pass may hold.
   *
   * Two things bound it. `DeviceVector` is indexed by `int`, so neither the
   * `pieces * cols * chunk` split nor the `pieces^2 * rows * chunk`
   * accumulator may reach 2^31. And the accumulator is real memory that the
   * product does not otherwise need, so `CHEDDAR_PCMM_SCRATCH_LOG2` caps it.
   */
  int ChunkFor(int cols, int rows, int pieces, int vec_len) const;

  /**
   * @brief Split one ciphertext component into `chunk`-sized int8 buffers.
   *
   * `vec_len` is the number of words a single RNS limb of the component holds:
   * the ring degree for an RLWE component, the small degree for an MLWE b-part
   * and `rank * small_degree` for an MLWE a-part. Everything else about the
   * product -- the split, the `pieces^2` GEMMs and the recombination -- is
   * identical, which is why every path is two calls to this pair.
   */
  void SplitComponent(std::vector<DeviceVector<int8_t>> &res,
                      const word *const *src_ptrs, int cols, int pieces,
                      const NPInfo &np, int vec_len, int chunk) const;

  /** @brief The GEMMs and the recombination, against an already-split source. */
  void ProductComponent(word *const *dst_ptrs,
                        const std::vector<DeviceVector<int8_t>> &split,
                        const SplitMatrix &u, const NPInfo &np, int vec_len,
                        int chunk) const;
};

}  // namespace cheddar

#endif  // USE_CUBLAS
