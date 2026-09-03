#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/Context.h"

namespace cheddar {

enum class EvalPolyType {
  kOdd,    // Odd polynomial
  kEven,   // Even polynomial
  kNormal  // Neither odd nor even
};

constexpr double kZeroCoeffThreshold = 1e-9;

/**
 * @brief A batch of ciphertexts at ONE (level, scale), in one strided device
 * buffer: ciphertext b's polynomials (bx, ax and, when carried, rx,
 * `np_.GetNumTotal() * degree` words each) lie back to back at
 * `data_ + b * CtStride()`. The batched EvalMod keeps its whole state in
 * these, so every key switch and every elementwise pass runs once over the
 * group. EvalMod's ciphertexts carry no auxiliary primes, and neither do
 * these.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class CtBatch {
 public:
  DeviceVector<word> data_;
  NPInfo np_;
  double scale_ = 1.0;
  int batch_ = 0;
  int num_slots_ = 0;
  bool has_rx_ = false;

  CtBatch() = default;
  CtBatch(CtBatch &&) = default;
  CtBatch &operator=(CtBatch &&) = default;
  CtBatch(const CtBatch &) = delete;
  CtBatch &operator=(const CtBatch &) = delete;

  size_t PolyWords() const {
    return static_cast<size_t>(np_.GetNumTotal()) * np_.degree_;
  }
  size_t CtStride() const { return (has_rx_ ? 3 : 2) * PolyWords(); }

  void Allocate(const NPInfo &np, int batch, bool has_rx);

  word *CtData(int b) { return data_.data() + b * CtStride(); }
  const word *CtData(int b) const { return data_.data() + b * CtStride(); }

  /**
   * @brief Views of ciphertext 0's polynomials, exactly as
   * `Ciphertext::ViewVector` would give them; a batched kernel reaches
   * ciphertext b through the batch stride.
   */
  std::vector<DvView<word>> ViewVector(int np_front_ignore = 0,
                                       bool ignore_rx = false);
  std::vector<DvConstView<word>> ConstViewVector(int np_front_ignore = 0,
                                                 bool ignore_rx = false) const;
};

/**
 * @brief The batched counterpart of MultiLevelCiphertext: one CtBatch per
 * level, all at one scale.
 */
template <typename word>
using MLCtBatch = std::map<int, CtBatch<word>>;

/**
 * @brief A class for the evaluation of a * x * y + b * z where
 * a is a small integer, b is a real number, x, y, z are ciphertexts.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class AXYPBZ {
 private:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;

  Constant<word> a_;
  Constant<word> b_;

  bool has_a_;
  bool has_b_;
  bool has_z_;

  void AssertSameLevelAndScale(ConstContextPtr<word> context, const Ct &ct,
                               int level, double scale) const;

 public:
  int x_level_;
  double x_scale_;
  int y_level_;
  double y_scale_;
  int z_level_;
  double z_scale_;
  double final_scale_;
  double final_level_;

  AXYPBZ(ConstContextPtr<word> context, int a, double b, int x_level,
         double x_scale, int y_level, double y_scale, int z_level,
         double z_scale);
  AXYPBZ(ConstContextPtr<word> context, int a, double b, int x_level,
         double x_scale, int y_level, double y_scale);

  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &x,
                const Ct &y, const Ct &z, const Evk &mult_key) const;
  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &x,
                const Ct &y, const Evk &mult_key) const;

  // The same evaluations over a batch, word for word the per-ciphertext loop.
  void EvaluateBatch(ConstContextPtr<word> context, CtBatch<word> &res,
                     const CtBatch<word> &x, const CtBatch<word> &y,
                     const CtBatch<word> &z, const Evk &mult_key) const;
  void EvaluateBatch(ConstContextPtr<word> context, CtBatch<word> &res,
                     const CtBatch<word> &x, const CtBatch<word> &y,
                     const Evk &mult_key) const;
};

/**
 * @brief A map for the evaluation of basis for an arbitrary polynomial
 * evaluation.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class BasisMap {
 private:
  std::map<int, AXYPBZ<word>> basis_eval_;

  using Ct = Ciphertext<word>;
  using MLCt = MultiLevelCiphertext<word>;
  using Evk = EvaluationKey<word>;

  int input_level_;
  double input_scale_;
  EvalPolyType type_;
  bool chebyshev_;

  int SplitBaseDegree(int base_degree) const;

 public:
  BasisMap(int input_level, double input_scale, EvalPolyType type,
           bool chebyshev);

  BasisMap(const BasisMap &) = delete;
  BasisMap &operator=(const BasisMap &) = delete;
  BasisMap(BasisMap &&) = default;

  std::pair<int, double> GetBaseLevelAndScale(int base_degree) const;

  bool Exists(int base_degree) const;

  void AddBase(ConstContextPtr<word> context, int base_degree);

  void Evaluate(ConstContextPtr<word> context, std::map<int, MLCt> &res,
                const Evk &mult_key) const;
  void EvaluateBatch(ConstContextPtr<word> context,
                     std::map<int, MLCtBatch<word>> &res,
                     const Evk &mult_key) const;
  void PlainEvaluate(std::map<int, double> &res) const;
};

/**
 * @brief A node in the evaluation tree for arbitrary polynomial evaluation.
 * First need to Compile() before Evaluate().
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class EvalPolyNode {
 private:
  using Ptr = std::shared_ptr<EvalPolyNode<word>>;

  Ptr low_ = nullptr;
  Ptr high_ = nullptr;

  using Ct = Ciphertext<word>;
  using MLCt = MultiLevelCiphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Dv = DeviceVector<word>;
  using Const = Constant<word>;

  bool is_low_zero_ = false;
  bool is_low_constant_ = false;
  bool is_high_constant_ = false;
  Const low_constant_;
  Const high_constant_;

  int split_degree_ = 0;

  std::vector<double> coefficients_;
  std::map<int, Const> leaf_constants_;

  bool IsLeaf() const;
  void EvaluateMiddleNode(ConstContextPtr<word> context, Ct &res,
                          std::map<int, MLCt> &basis,
                          const Evk &mult_key) const;
  void EvaluateLeaf(ConstContextPtr<word> context, Ct &res,
                    std::map<int, MLCt> &basis, const Evk &mult_key,
                    bool inplace) const;
  void EvaluateMiddleNodeBatch(ConstContextPtr<word> context,
                               CtBatch<word> &res,
                               std::map<int, MLCtBatch<word>> &basis,
                               const Evk &mult_key) const;
  void EvaluateLeafBatch(ConstContextPtr<word> context, CtBatch<word> &res,
                         std::map<int, MLCtBatch<word>> &basis,
                         const Evk &mult_key, bool inplace) const;

  int target_level_;
  bool do_rescale_ = true;

 public:
  EvalPolyNode(const std::vector<double> &coefficients, int level_margin,
               int baby_threshold, bool chebyshev);

  EvalPolyNode(const EvalPolyNode &) = delete;
  EvalPolyNode &operator=(const EvalPolyNode &) = delete;
  EvalPolyNode(EvalPolyNode &&) = default;

  void CheckRequiredBasis(std::set<int> &required_bases) const;

  void Compile(ConstContextPtr<word> context, const BasisMap<word> &basis_eval,
               int target_level, double target_scale, bool do_rescale = true);

  void Evaluate(ConstContextPtr<word> context, Ct &res,
                std::map<int, MLCt> &basis, const Evk &mult_key) const;
  void EvaluateBatch(ConstContextPtr<word> context, CtBatch<word> &res,
                     std::map<int, MLCtBatch<word>> &basis,
                     const Evk &mult_key) const;
  double PlainEvaluate(std::map<int, double> &res) const;
};

/**
 * @brief A class for arbitrary polynomial evaluation. First need to Compile()
 * before Evaluate().
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class EvalPoly {
 private:
  using Ct = Ciphertext<word>;
  using MLCt = MultiLevelCiphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Dv = DeviceVector<word>;

  // Chebyshev polynomials expressed in the normal basis
  static inline std::vector<std::vector<double>> plain_chebyshev_basis_{};

  std::vector<double> coefficients_;
  EvalPolyType type_;
  bool chebyshev_;

  int input_level_;
  double input_scale_;
  double target_scale_;

  BasisMap<word> basis_map_;
  std::shared_ptr<EvalPolyNode<word>> tree_root_ = nullptr;

  // Preparation & compile methods
  void PreparePlainChebyshevBasis();
  EvalPolyType DetermineType();

 public:
  EvalPoly(const std::vector<double> &coefficients, int input_level,
           double input_scale, double target_scale, bool chebyshev = false);

  // disable copying (or moving also)
  EvalPoly(const EvalPoly &) = delete;
  EvalPoly &operator=(const EvalPoly &) = delete;

  // Just for forwarding cases
  EvalPoly(EvalPoly &&) = default;

  int GetPolyDegree() const;

  // Optionally change the basis type
  void ConvertToChebyshevBasis();
  void ConvertToNormalBasis();

  void Compile(ConstContextPtr<word> context);

  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const Evk &mult_key) const;

  /**
   * @brief `Evaluate` over a batch of ciphertexts at one (level, scale), in
   * about as many launches as ONE evaluation takes: the compiled tree is
   * walked once, every key switch is `MultKeyBatch`-shaped and every
   * elementwise pass runs with the batch on `gridDim.z`. Word for word the
   * per-ciphertext loop.
   */
  void EvaluateBatch(ConstContextPtr<word> context, CtBatch<word> &res,
                     const CtBatch<word> &input, const Evk &mult_key) const;

  double PlainEvaluate(double input) const;
};

}  // namespace cheddar