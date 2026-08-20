#pragma once

#include <array>
#include <vector>

#include "core/Type.h"

namespace cheddar {

/**
 * @brief The attention packing layout of [SYLPH] section 3.2, and the
 * coefficient/slot index correspondence Cheddar's CtS and StC actually
 * implement.
 *
 * ## Why a layout is a piece of code at all
 *
 * The pipeline alternates between two encodings: the non-linear operators run
 * on slots at degree 2^16, the matrix products run on coefficients at a small
 * ring. HalfBoot and SlotToCoeff move between them, and both are *transports*
 * -- the values come out unchanged. What does not come out unchanged is
 * **where** they sit, because the homomorphic transform omits a bit reversal
 * that the host encoder performs (see SlotOfCoeff below).
 *
 * [SYLPH] 3.2 answers that by choosing the layout so the permutation lands the
 * data where the next algorithm wants it: the layout is "designed to
 * accommodate the bit-reversal operation that occurs when moving to
 * coefficient encoding, and the partial bit-reversal operation which occurs
 * when moving to SinC encoding", the net effect being "effectively removing
 * the need for most conversions throughout the entire process". So the layout
 * is not bookkeeping around the conversions -- it is what makes them free.
 *
 * ## The map
 *
 * The tensor is t[i][j][k] with i the head, j the channel within a head and k
 * the token. [SYLPH] gives, for 0 <= l < nheads/4,
 *
 *     ct_l[s] = t[ floor(s/2^7) mod nheads,
 *                  l*2^9/nheads + floor(s/(2^7*nheads)),
 *                  s mod 2^7 ]
 *
 * for nheads = 32, head_dim = 128, num_tokens = 2^7 and degree 2^16. Two
 * things are worth naming because they are not obvious from the formula.
 *
 * **s runs over the ring degree, not over the slot count.** The three fields
 * are 7 + 5 + 4 = 16 bits, and 8 * 2^16 = 2^19 = 32*128*128 accounts for every
 * tensor entry exactly once, whereas 8 * 2^15 would account for half. s is
 * therefore a *coefficient* index; a complex slot holds two of them (s and
 * s + degree/2) as its real and imaginary parts.
 *
 * **The constants generalise.** 2^9/nheads is degree/(num_tokens*nheads),
 * which is also the number of channels one ciphertext carries, so the whole
 * map is
 *
 *     k = s mod num_tokens
 *     i = (s / num_tokens) mod num_heads
 *     j = l * channels_per_ct + s / (num_tokens * num_heads)
 *
 * with channels_per_ct = degree / (num_tokens * num_heads) and
 * num_ciphertexts = head_dim / channels_per_ct. For [SYLPH]'s shape that is 16
 * and 8, and 8 = nheads/4 is a coincidence of those numbers rather than a
 * rule -- this class computes it.
 *
 * As a bit field the coefficient index reads, most significant first,
 *
 *     [ channel-within-ciphertext | head | token ]
 *
 * so the token index varies fastest and the channel index slowest, which is
 * the "lexicographic by (j, i, k)" ordering [SYLPH] describes.
 *
 * ## The permutation, and where it comes from
 *
 * Encoder::SpecialFFT begins with BitReverseVector and Encoder::SpecialIFFT
 * ends with one (Encode.cpp:252, 263), so the host encode/decode pair is
 * bit-reversal free end to end. EvalSpecialFFT::PopulatePlainMatrices builds
 * the *butterfly stages only* (EvalSpecialFFT.cpp:64-167) -- there is no
 * bit-reversal matrix anywhere in that file, and there could not cheaply be
 * one, since a bit reversal is not a striped matrix. So the homomorphic
 * transform differs from the host one by exactly one bit reversal, which is
 * the reversal [SYLPH] 3.2 says the layout exists to absorb.
 *
 * Writing n = degree/2 for the slot count, that gives
 *
 *     coefficient p  ->  real part of slot BitReverse(p, log2 n)          p < n
 *                        imaginary part of slot BitReverse(p - n, log2 n) p >= n
 *
 * up to a scalar. **This is a derivation, and AttentionPackingTest measures
 * it** on the GPU rather than trusting it: HalfBoot runs whose coefficient
 * probes are +-A according to one bit of the coefficient index recover the
 * whole map with no assumption about its shape, and the test compares that
 * measurement against SlotOfCoeff.
 *
 * ## What the permutation does to the axes
 *
 * A bit reversal on the low log2 n bits reverses the *order of the fields* as
 * well as the bits inside each, so in the slot index the three fields read
 *
 *     [ token | head | channel-within-ciphertext ]
 *
 * that is, the token and channel axes exchange the fastest- and
 * slowest-varying roles and the head axis stays between them. For the [SYLPH]
 * shape the channel axis owns slot bits 0-2, the head axis bits 3-7 and the
 * token axis bits 8-14.
 *
 * The channel field's own top bit does not take part in the reversal: it is
 * the coefficient index's bit 15, which selects the real or the imaginary part
 * of the slot. So the channel axis is the one axis split across the two parts
 * of a slot -- for the [SYLPH] shape, channels 0-7 of a ciphertext sit in the
 * real parts and channels 8-15 in the imaginary parts. An operator that treats
 * a slot as a single number rather than as two must not cross that axis.
 * ImaginaryFlagAxis reports it, and SlotAxisOrder covers the slot index
 * proper; all three are computed from the map rather than asserted.
 */
class AttentionPacking {
 public:
  /// The three axes of the attention tensor.
  enum class Axis { kHead, kChannel, kToken };

  /// One tensor position. head is [SYLPH]'s i, channel its j (absolute, not
  /// within a ciphertext) and token its k.
  struct TensorIndex {
    int head;
    int channel;
    int token;
  };

  /// Where one coefficient of one ciphertext ends up after the conversion.
  struct SlotPosition {
    int slot;        ///< complex slot index, in [0, degree/2)
    bool imaginary;  ///< whether the value is that slot's imaginary part
  };

  /**
   * @brief Build the layout for one attention tensor.
   *
   * @param num_heads nheads, a power of two
   * @param head_dim channels per head, a power of two
   * @param num_tokens sequence length, a power of two
   * @param degree ring degree of the ciphertexts holding the tensor
   */
  AttentionPacking(int num_heads, int head_dim, int num_tokens, int degree);

  int GetNumHeads() const { return num_heads_; }
  int GetHeadDim() const { return head_dim_; }
  int GetNumTokens() const { return num_tokens_; }
  int GetDegree() const { return degree_; }
  int GetNumSlots() const { return degree_ / 2; }
  int GetNumCiphertexts() const { return num_ciphertexts_; }
  int GetChannelsPerCiphertext() const { return channels_per_ct_; }
  /// Total tensor entries, equal to num_ciphertexts * degree.
  int GetTensorSize() const { return num_heads_ * head_dim_ * num_tokens_; }

  /// Offset of a tensor position in the flat host tensor, which is stored
  /// (head, channel, token) row-major.
  int TensorOffset(const TensorIndex &t) const;

  /// [SYLPH] 3.2's map: which tensor entry coefficient `coeff` of ciphertext
  /// `ct` carries.
  TensorIndex CoeffPosition(int ct, int coeff) const;

  /// The inverse of CoeffPosition.
  void CoeffIndexOf(const TensorIndex &t, int &ct, int &coeff) const;

  /// Where the conversion sends a coefficient. Static because it is a property
  /// of the ring and of the transform, not of this layout.
  static SlotPosition SlotOfCoeff(int coeff, int degree);
  /// The inverse of SlotOfCoeff.
  static int CoeffOfSlot(const SlotPosition &pos, int degree);

  SlotPosition SlotOfCoeff(int coeff) const {
    return SlotOfCoeff(coeff, degree_);
  }
  int CoeffOfSlot(const SlotPosition &pos) const {
    return CoeffOfSlot(pos, degree_);
  }

  /// Which tensor entry sits at a slot position after the conversion, i.e.
  /// CoeffPosition composed with CoeffOfSlot.
  TensorIndex SlotToTensor(int ct, const SlotPosition &pos) const;

  /// The three axes ordered fastest-varying first, in the coefficient index
  /// and in the slot index. Computed from the map by locating the index bits
  /// each axis controls; the constructor checks that each axis owns a
  /// contiguous block of them in both domains, which is what makes the
  /// ordering well defined.
  ///
  /// The slot ordering is over the slot index proper. The real/imaginary flag
  /// is a further bit that is not part of it, and it is not contiguous with
  /// the rest of its axis -- for the [SYLPH] shape the channel axis owns slot
  /// bits 0-2 *and* the flag -- so it is reported separately by
  /// ImaginaryFlagAxis rather than folded in.
  std::array<Axis, 3> CoeffAxisOrder() const { return coeff_axis_order_; }
  std::array<Axis, 3> SlotAxisOrder() const { return slot_axis_order_; }

  /// Which axis the real/imaginary flag of a slot splits. One value of this
  /// axis lives in the real parts and another in the imaginary parts, so an
  /// operator that treats a slot as one number rather than two must not cross
  /// it.
  Axis ImaginaryFlagAxis() const { return imaginary_flag_axis_; }

  /// Human-readable name of an axis, for reports.
  static const char *AxisName(Axis axis);

  /**
   * @brief Spread a flat tensor over the ciphertexts' coefficient vectors.
   *
   * @param res one length-degree coefficient vector per ciphertext
   * @param tensor flat tensor, GetTensorSize() entries, (head, channel, token)
   *               row-major
   */
  void PackCoeff(std::vector<std::vector<double>> &res,
                 const std::vector<double> &tensor) const;

  /// The inverse of PackCoeff.
  void UnpackCoeff(std::vector<double> &res,
                   const std::vector<std::vector<double>> &cts) const;

  /**
   * @brief Spread a flat tensor over the ciphertexts' *slot* vectors, so that
   * converting to coefficient encoding produces exactly PackCoeff's output.
   * This is the layout the non-linear operators must hold their data in for
   * the matrix product to find it where it expects.
   *
   * @param res one length-degree/2 complex slot vector per ciphertext
   * @param tensor flat tensor, GetTensorSize() entries
   */
  void PackSlot(std::vector<std::vector<Complex>> &res,
                const std::vector<double> &tensor) const;

  /// The inverse of PackSlot.
  void UnpackSlot(std::vector<double> &res,
                  const std::vector<std::vector<Complex>> &cts) const;

 private:
  int num_heads_;
  int head_dim_;
  int num_tokens_;
  int degree_;
  int channels_per_ct_;
  int num_ciphertexts_;

  std::array<Axis, 3> coeff_axis_order_;
  std::array<Axis, 3> slot_axis_order_;
  Axis imaginary_flag_axis_;

  // Order the axes by the index bits they control, checking on the way that
  // each axis owns a contiguous block of them.
  std::array<Axis, 3> DeriveAxisOrder(bool in_slot_domain) const;
  // Which axis the real/imaginary flag moves.
  Axis DeriveImaginaryFlagAxis() const;
};

}  // namespace cheddar
