#pragma once

#include <string>
#include <vector>

namespace cheddar {

/**
 * @brief [SYLPH] section 4.2: the depth-ONE plaintext-ciphertext matrix
 * product, and appendix E's lemma 2 that makes it one.
 *
 * ## The problem it solves
 *
 * PC-attention is wide -- section 4's footnote 2 says its input and output fit
 * in 8 ciphertexts while the computation itself passes through 256 -- so a
 * bootstrap inside it is unaffordable and the whole design goal is to spend as
 * few levels as possible. JKLS [26] spends THREE: one for `sigma` and `tau`,
 * one for the column rotations, one for the Hadamard products.
 *
 * ## How three become one
 *
 * JKLS is the identity
 *
 *     C = sum_k rot_C^k(sigma(A)) . rot_R^k(tau(B))                      (4)
 *
 * with, for all `0 <= i, j < d`,
 *
 *     (rot_C^k(A))_{i,j} = A_{i, j+k}      (sigma(A))_{i,j} = A_{i, i+j}
 *     (rot_R^k(A))_{i,j} = A_{i+k, j}      (tau(A))_{i,j}   = A_{i+j, j}
 *
 * all indices modulo `d`. Two of the three levels go for free:
 *
 *  1. `A` is the PLAINTEXT weight, so `sigma` and its column rotations are
 *     index permutations done on the host and cost nothing.
 *  2. The remaining `tau` on the ciphertext is not removed but MOVED. Feed the
 *     algorithm `tau^(l+1)(B)` and let it return `tau^l(C)`, and the `tau` it
 *     needed is already in its input -- so a chain of these products carries
 *     one `tau` from end to end instead of paying for one each.
 *
 * What is left is one plaintext multiply, which is the one level.
 *
 * ## Lemma 2, and where the plaintext's index comes from
 *
 * Appendix E:
 *
 *     tau(rot_R^k(M)) = rot_R^k(tau(M))
 *     tau(rot_C^k(M)) = rot_R^(-k) . rot_C^k (tau(M))
 *
 * Applying the second `l` times gives `tau^l(rot_C^k(M)) = rot_R^(-l k) .
 * rot_C^k (tau^l(M))`, and `tau` commutes with the Hadamard product because it
 * is a permutation of entries. So applying `tau^l` to (4):
 *
 *     tau^l(C) = sum_k [rot_R^(-l k) rot_C^k (tau^l sigma A)] . rot_R^k(tau^(l+1) B)
 *
 * and splitting `k = i + j b` in baby-step giant-step form, so that the
 * ciphertext's rotations factor as `rot_R^(i + j b) = rot_R^(j b) . rot_R^i`:
 *
 *     tau^l(C) = sum_j rot_R^(j b) ( sum_i pt_{A,i,j,l} . rot_R^i(tau^(l+1) B) )   (5)
 *     pt_{A,i,j,l} = rot_R^(-l(i + j b) - j b) . rot_C^(i + j b) (tau^l sigma A)
 *
 * `O(sqrt d)` ciphertext rotations, one level. Section 4.2 also refuses to
 * store the `d` rearranged plaintexts -- "impractical for large-scale models
 * due to the massive memory overhead" -- and keeps only `tau^l sigma(A)`,
 * computing each `pt_{A,i,j,l}` from it at runtime with two plaintext
 * rotations. `SylphPcmmPlan` is that single stored form.
 *
 * ## Row rotations are slot rotations, column rotations are not
 *
 * With the matrix laid out row by row, entry `(i, j)` at slot `i d + j`:
 * `rot_R^k` is a cyclic slot rotation by `k d` and is exactly what CKKS gives.
 * `rot_C^k` is a rotation WITHIN each row and would straddle row boundaries as
 * a slot rotation -- which is the level JKLS spends on masks. Here it only
 * ever acts on the plaintext, where it is an index permutation, so the
 * distinction costs nothing and is why `A` must be the plaintext operand and
 * `B` the ciphertext one.
 *
 * Everything in this header is host arithmetic on `d*d` doubles and can be
 * checked without a GPU; `SylphPcmm.h` is the encrypted half.
 */
namespace sylph_pcmm {

/** @brief A `d x d` matrix, row by row, exactly as it lies in the slots. */
using Mat = std::vector<double>;

/** @brief `(sigma(A))_{i,j} = A_{i, i+j}`. */
Mat Sigma(const Mat &a, int d);

/** @brief `(tau(A))_{i,j} = A_{i+j, j}`. */
Mat Tau(const Mat &a, int d);

/** @brief `tau` applied `n` times; `n` may be any non-negative integer. */
Mat TauPow(const Mat &a, int d, int n);

/** @brief `(rot_R^k(A))_{i,j} = A_{i+k, j}` -- a slot rotation by `k d`. */
Mat RotR(const Mat &a, int d, int k);

/** @brief `(rot_C^k(A))_{i,j} = A_{i, j+k}` -- NOT a slot rotation. */
Mat RotC(const Mat &a, int d, int k);

/** @brief The ordinary product `C = A B`, for checking against. */
Mat MatMul(const Mat &a, const Mat &b, int d);

/**
 * @brief What section 4.2 stores, and the shape of the loop that reads it.
 *
 * `g * b = d`; `b` is the baby-step count and `g` the giant-step count, so the
 * ciphertext pays `b - 1` inner rotations and `g - 1` outer ones.
 */
struct SylphPcmmPlan {
  int d = 0;
  int b = 0;  //!< baby steps
  int g = 0;  //!< giant steps
  int tau_power = 0;  //!< `l`: the input is `tau^(l+1)(B)`, the output `tau^l(C)`
  //! `tau^l . sigma(A)` -- the ONE plaintext kept, section 4.2's memory point.
  Mat base;
  bool ok = false;
  std::string why;

  /** @brief Ciphertext rotations Eq. (5) performs: `(b - 1) + (g - 1)`. */
  int NumRotations() const { return (b - 1) + (g - 1); }
  /** @brief Slot distances those rotations use, all multiples of `d`. */
  std::vector<int> RotationDistances() const;
};

/**
 * @brief Build the plan for `A` at `tau_power = l`.
 *
 * `b` defaults to the square root of `d` rounded to a divisor, which is the
 * BSGS optimum; any divisor of `d` is accepted so a caller can trade the two
 * rotation counts.
 */
SylphPcmmPlan BuildPlan(const Mat &a, int d, int tau_power, int b = 0);

/**
 * @brief `pt_{A,i,j,l}` of Eq. (5), rebuilt from `plan.base` at runtime.
 *
 * Two plaintext rotations, as section 4.2 says -- one column, one row.
 */
Mat PlaintextFor(const SylphPcmmPlan &plan, int i, int j);

/**
 * @brief The slot permutation that applies `tau^n`, for `SlotPermute`.
 *
 * Section 4.2 applies `tau^2` ONCE, "right after RoPE, before the computation
 * becomes wide", because each PCMM call consumes one power. `tau` is a
 * permutation of the slot index and nothing else, so it is a `SlotPermute` and
 * costs one level -- and a cheaper one than the estimate: output slot
 * `i d + j` reads input slot `((i + n j) mod d) d + j`, so every offset
 * `s - perm[s]` is a multiple of `d`, and the count is however many values
 * `n j mod d` takes. MEASURED at `d = 128` in 16384 slots: 128 distinct
 * offsets at `n = 1` and **64 at `n = 2`**, because an even `n` halves the
 * orbit. So the `tau^2` section 4.2 wants is about 64 diagonals and a BSGS
 * grid of 8 x 8 -- well inside the square-transpose case `SlotPermute`'s
 * header prices at 255 and 31 rotations, and nowhere near the field-swap case
 * that costs 2048 and 95.
 *
 * Slots at or beyond `d * d` map to themselves, so the result is a bijection
 * on the whole slot vector as `SlotPermute` requires.
 *
 * @param perm convention: output slot `perm[s]` receives input slot `s`
 */
std::vector<int> TauPermutation(int d, int n, int num_slots);

/**
 * @brief Eq. (5) in the clear: `tau^l(C)` from `A`'s plan and `tau^(l+1)(B)`.
 *
 * This is the reference the encrypted path is checked against, and it is also
 * the proof that the index algebra above is right -- it reproduces `A B` up to
 * the `tau^l` the caller asked for, and `SylphPcmmMath`'s tests say so.
 */
Mat PlainApply(const SylphPcmmPlan &plan, const Mat &tau_b);

}  // namespace sylph_pcmm
}  // namespace cheddar
