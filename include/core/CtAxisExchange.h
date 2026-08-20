#pragma once

#include <vector>

#include "core/Container.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"

namespace cheddar {

/**
 * @brief Exchange a bit field of the slot index with the index of an array of
 * ciphertexts.
 *
 * ## Why this exists: the score product's right operand
 *
 * `SwitchedCcmmLayout` puts entry `(row, column, lane)` of an operand in big
 * ciphertext `column / rank`, so **three bits of the column index live on the
 * ciphertext axis**. For Q, the left operand, the column index is the channel,
 * and which channel lands in which ciphertext is a permutation of the
 * projection's weight columns -- free, and `AttentionTransportTest` spends it
 * on keeping RoPE's pair inside one ciphertext. For K the column index is the
 * **key token**, and no weight matrix can move a token. The block delivers the
 * token in the low slot bits (`slot = channel_in_ct * T + token`), so three of
 * them have to cross from the slot index to the ciphertext index, and no
 * `SlotPermute` can do that: a linear transform is a map inside one
 * ciphertext.
 *
 * ## What it computes
 *
 * Write `F(p) = (p >> field_offset) mod 2^w` for the field and
 * `S_X = src[X >> (w - w_src)]`. Then
 *
 *     res[Y][p] = S_{F(p)}[ p with its field set to Y ]
 *
 * -- the value that sat in ciphertext `X` at a slot whose field reads `Y`
 * moves to ciphertext `Y`, at the same slot with its field reading `X`. The
 * two axes trade places, so one call does any number of bits at once; it is a
 * transpose of the `(ciphertext, field)` pair, not a sequence of one-bit
 * swaps, and it costs one level however wide the field is.
 *
 * ## Why it is 2(2^w - 1) rotations and not 2^(2w)
 *
 * Written out, `res[Y] = sum_X M_X . rot_{(Y-X)*step}(S_X)` with `M_X` the
 * mask of the slots whose field reads `X` and `step = 2^field_offset`: an
 * all-to-all crossbar, 2^(2w) rotations. But the rotation amount depends only
 * on `Y - X`, so it factors:
 *
 *     W_X   = rot_{-X*step}(S_X)
 *     res[Y] = rot_{Y*step}( sum_X M_{(X+Y) mod 2^w} . W_X )
 *
 * The step that makes this work is that rotating a field mask by `-Y*step`
 * renames it: `rot_{-Y*step}(M_X) = M_{(X+Y) mod 2^w}`, because subtracting a
 * multiple of `2^field_offset` leaves the bits below the field alone and so
 * never borrows into it. What is left is `2^w - 1` rotations on the way in and
 * `2^w - 1` on the way out, with the crossbar itself carried entirely by
 * `2^w` mask plaintexts. For the score product's shape that is **14 key
 * switches instead of 64**.
 *
 * The mask multiplications do not go away -- there are `2^(2w)` of them, one
 * per (output ciphertext, source) pair, which is inherent to a multiplexer
 * built out of masks. They are the cheapest operation in the library and they
 * all sit at one level; the key switches are what would have hurt.
 *
 * ## Replication, which GQA needs and which is free here
 *
 * Llama-3-8B has 8 KV heads against 32 query heads, so a call group of 16
 * lanes needs each of its 4 KV heads four times over. `src` is therefore
 * allowed to be **shorter than the output**: `S_X = src[X >> (w - w_src)]`, so
 * the low `w - w_src` bits of the exchanged field are a copy index. After the
 * exchange those bits are the low bits of the *slot* field, which is exactly
 * where the lane index wants its replication -- the four lanes sharing a KV
 * head are adjacent. Two source ciphertexts become eight and nothing is added
 * to the rotation count, because the `2^w - 1` inward rotations are needed
 * whether or not the sources they rotate are distinct.
 *
 * ## One level, and the scale
 *
 * The mask multiply is the only level this spends. The masks are encoded at
 * the level's own rescale prime product, so the result comes back on the
 * canonical scale of `level - 1` -- the contract `LinearTransform` keeps, and
 * the one `Context::LevelDown` downstream assumes.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CtAxisExchange {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;

 public:
  /**
   * @brief Build the exchange at `level`; it returns one level below.
   *
   * @param context the ring the ciphertexts live in
   * @param log_num_cts `w`: the field is `w` bits wide and the output is
   *        `2^w` ciphertexts
   * @param field_offset how many slot bits below the field are left untouched
   * @param level the level the masks are encoded at
   * @param log_num_src_cts `w_src`: the input is `2^w_src` ciphertexts, each
   *        used `2^(w - w_src)` times. Defaults to `w`, i.e. no replication.
   */
  CtAxisExchange(ConstContextPtr<word> context, int log_num_cts,
                 int field_offset, int level, int log_num_src_cts = -1);

  // disable copying (or moving also)
  CtAxisExchange(const CtAxisExchange &) = delete;
  CtAxisExchange &operator=(const CtAxisExchange &) = delete;

  int GetNumCts() const { return num_cts_; }
  int GetNumSrcCts() const { return num_src_cts_; }
  int GetLevel() const { return level_; }
  /// Plaintext multiplications one Evaluate performs, which is the cost this
  /// trades the key switches for.
  int GetNumMults() const { return num_cts_ * num_cts_; }

  /// The `2 * (2^w - 1)` distinct rotation distances, inward ones first.
  std::vector<int> RotationIndices() const;

  /// The inward rotations run at `level` and the outward ones at `level - 1`.
  void AddRequiredRotations(EvkRequest &req) const;

  /**
   * @brief `res[Y][p] = src[F(p) >> (w - w_src)][p with field := Y]`.
   *
   * @param res output, `2^w` ciphertexts at `level - 1`
   * @param src input, `2^w_src` ciphertexts at `level`
   * @param evk_map rotation keys covering RotationIndices()
   */
  void Evaluate(std::vector<Ct> &res, const std::vector<Ct> &src,
                const EvkMap<word> &evk_map) const;

 private:
  ConstContextPtr<word> context_;
  int num_slots_;
  int log_num_cts_;
  int num_cts_;
  int log_num_src_cts_;
  int num_src_cts_;
  int field_offset_;
  int step_;
  int level_;
  std::vector<Pt> mask_;
};

}  // namespace cheddar
