#pragma once

#include <utility>
#include <vector>

#include "core/NPInfo.h"
#include "core/Type.h"

namespace cheddar {

// THESE ARE THE PARAMETERS FOR SENSITIVITY STUDIES
// Best performance should be achieved with all true;
constexpr bool kMergePMult = true;
constexpr bool kMergeCMult = true;
constexpr bool kOptimizeAutomorphism = true;
constexpr bool kFuseMontgomery = true;
constexpr bool kFuseModDownEpilogue = true;
constexpr bool kFuseGSPAccum = true;
constexpr bool kFuseBSKeyMult = true;
constexpr bool kExtendedOT = true;

template <typename word>
class Parameter {
 private:
  using signed_word = make_signed_t<word>;
  constexpr static const std::pair<int, int> kZeroPair{0, 0};

 public:
  /**
   * @brief Construct a new Parameter object for holding CKKS parameters.
   *
   * @param log_degree log2(polynomial degree), currently only 16 is supported
   * @param base_scale scale at level 0
   * @param default_encryption_level default encryption level (maximum level
   * after bootstrapping)
   * @param level_config level configuration
   * @param main_primes list of main primes
   * @param aux_primes list of auxiliary primes
   * @param ter_primes list of terminal primes (optional)
   * @param additional_base additional base primes for level 0 (optional)
   * @param conjugate_invariant work in the conjugate-invariant ring (optional)
   */
  Parameter(int log_degree, double base_scale, int default_encryption_level,
            std::vector<std::pair<int, int>> level_config,
            const std::vector<word> &main_primes,
            const std::vector<word> &aux_primes,
            const std::vector<word> &ter_primes = std::vector<word>{},
            const std::pair<int, int> &additional_base = kZeroPair,
            bool conjugate_invariant = false);

  static constexpr const int word_size_ = sizeof(word);
  // For 32-bit word, we allow up to 2^31 for the primes
  // For 64-bit word, we allow up to 2^63 for the primes
  static constexpr const int extra_bits_ = 1;
  static constexpr const int galois_number_ = 5;

  const int log_degree_;
  const int degree_;
  // Conjugate-invariant CKKS (Kim and Song, ISISC 2018). The ring is the
  // maximal real subring Z[Y + Y^-1] of the 4N-th cyclotomic, of rank
  // `degree_` -- the same rank, and so the same lattice problem, as the
  // ordinary Z[X]/(X^N + 1). What changes is the canonical embedding: it is
  // real, so a message is `degree_` real slots rather than `degree_ / 2`
  // complex ones. Slot-wise multiplication no longer mixes a real part with
  // an imaginary one, which is the whole reason to want it.
  //
  // Concretely, three things move: the NTT root becomes a 4N-th root of unity
  // (so every prime must be 1 mod 4N), the transform gains a fold and an
  // unfold around the existing butterfly network, and Galois exponents live
  // mod 4N with N of them rather than N/2.
  const bool conjugate_invariant_;
  const int dnum_;
  const int L_;
  const int alpha_;

  const double base_scale_;
  const int default_encryption_level_;
  const int max_level_;

  const std::vector<word> main_primes_;
  const std::vector<word> ter_primes_;
  const std::vector<word> aux_primes_;

  const std::vector<std::pair<int, int>> level_config_;
  const std::pair<int, int> additional_base_;

  // Movable, but not copyable
  Parameter(Parameter &&) = default;
  Parameter &operator=(Parameter &&) = default;

  ~Parameter();

  /**
   * @brief Get Galois factor (galois_number_^i) % CyclotomicIndex()
   *
   * @param i index in range [0, MaxNumSlots()]
   * @return int Galois factor for index i
   */
  int GetGaloisFactor(int i) const;

  /**
   * @brief Number of slots a full-width message occupies: `degree_` real slots
   * in the conjugate-invariant ring, `degree_ / 2` complex ones otherwise.
   *
   * @return int maximum number of slots
   */
  int MaxNumSlots() const;

  /**
   * @brief The cyclotomic index M. Galois exponents and the NTT root both live
   * mod this: 4 * degree_ for the conjugate-invariant ring, 2 * degree_
   * otherwise. It is also the modulus every prime must be 1 modulo.
   *
   * @return int cyclotomic index
   */
  int CyclotomicIndex() const;

  /**
   * @brief Get default scale for a given level
   *
   * @param level level in range [0, default_encryption_level_]
   * @return double the scale
   */
  double GetScale(int level) const;

  /**
   * @brief Get rescale prime product for a given level
   *
   * @param level level in range [1, max_level_]
   * @return double the rescale prime product
   */
  double GetRescalePrimeProd(int level) const;

  /**
   * @brief Get Dense Hamming weight
   *
   * @return int dense_h_
   */
  int GetDenseHammingWeight() const;

  /**
   * @brief Get Sparse Hamming weight
   *
   * @return int sparse_h_
   */
  int GetSparseHammingWeight() const;

  /**
   * @brief Set Dense Hamming Weight object
   *
   * @param h new dense_h_
   */
  void SetDenseHammingWeight(int h);

  /**
   * @brief Set Sparse Hamming Weight object
   *
   * @param h new sparse_h_
   */
  void SetSparseHammingWeight(int h);

  /**
   * @brief Check whether sparse secret encapsulation (SSE) is used.
   *
   * @return true if using sparse secret encapsulation
   * @return false if not using sparse secret encapsulation
   */
  bool IsUsingSparseSecretEncapsulation() const;

  /**
   * @brief Get the maximum number of terminal primes. Refer to our paper.
   *
   * @return int maximum number of terminal primes
   */
  int GetMaxNumTer() const;

  /**
   * @brief Get the maximum number of main primes. Refer to our paper.
   *
   * @return int maximum number of main primes
   */
  int GetMaxNumMain() const;

  /**
   * @brief Get the maximum number of q primes = main + terminal primes.
   *
   * @return int maximum number of q primes
   */
  int GetMaxNumQ() const;

  /**
   * @brief Get the maximum number of auxiliary primes. Refer to our paper.
   *
   * @return int maximum number of auxiliary primes
   */
  int GetMaxNumAux() const;

  /**
   * @brief Get the number of auxiliary primes for sparse secret encapsulation
   * (SSE).
   *
   * @return int maximum number of auxiliary primes for SSE
   */
  int GetSSENumAux() const;

  /**
   * @brief Validate that the given NPInfo is valid.
   *
   * @param np NPInfo object
   */
  void AssertValidNP(const NPInfo &np) const;

  /**
   * @brief Convert a level to NPInfo.
   *
   * @param level level in range [-1, max_level_]
   * @param num_aux number of auxiliary primes in range [0, alpha_]
   * @return NPInfo conversion result
   */
  NPInfo LevelToNP(int level, int num_aux = 0) const;

  /**
   * @brief Convert NPInfo to a level.
   *
   * @param np NPInfo object
   * @return int level in range [-1, max_level_]
   */
  int NPToLevel(const NPInfo &np) const;

  /**
   * @brief Get a concatenated vector of primes for a given level. The
   * concatenated list is composed as follows: ter_primes_[num_ter - 1 ... 0],
   * main_primes_[0 ... num_main - 1], aux_primes_[0 ... num_aux - 1]
   *
   * @param np NPInfo object
   * @return std::vector<word>  Concatenated vector of primes
   */
  std::vector<word> GetPrimeVector(const NPInfo &np) const;

  /**
   * @brief Get a GPU-memory pointer to the list of primes for a given NPInfo
   *
   * @param np NPInfo object
   * @return const word* GPU-memory pointer to the list of primes
   */
  const word *GetPrimesPtr(const NPInfo &np) const;

  /**
   * @brief Get a GPU-memory pointer to the list of inverse primes for a given
   * NPInfo
   *
   * @param np NPInfo object
   * @return const signed_word* GPU-memory pointer to the list of inverse primes
   */
  const signed_word *GetInvPrimesPtr(const NPInfo &np) const;

  // Some special-use functions
  const word *__GetPrimesPtrModDownWithTerPrimes(
      const NPInfo &np_src, const NPInfo &np_non_intt) const;
  const signed_word *__GetInvPrimesPtrModDownWithTerPrimes(
      const NPInfo &np_src, const NPInfo &np_non_intt) const;

 private:
  std::vector<int> galois_factors_;

  // Refer to Bossuat, Jean-Philippe, Juan Troncoso-Pastoriza, and Jean-Pierre
  // Hubaux. "Bootstrapping for approximate homomorphic encryption with
  // negligible failure-probability by using sparse-secret encapsulation."
  // International Conference on Applied Cryptography and Network Security.
  // Cham: Springer International Publishing, 2022.
  int dense_h_;   // = degree_ / 2 (default)
  int sparse_h_;  // = 32 (default)

  std::vector<word> q_primes_;
  std::vector<signed_word> inv_q_primes_;
  std::vector<signed_word> inv_aux_primes_;

  std::vector<word *> primes_dv_;
  std::vector<signed_word *> inv_primes_dv_;

  NPInfo short_base_np_;

  std::vector<double> scale_;
  std::vector<double> rescale_prime_prod_;
};

}  // namespace cheddar
