// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// A dense row-major float64 tensor. Deliberately minimal: this is a reference
// oracle, not a numerics library, and every operation on it is written out in
// index order so the answer does not depend on a BLAS implementation.

#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace oracle {

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<int64_t> shape, double fill = 0.0);
  Tensor(std::initializer_list<int64_t> shape, double fill = 0.0)
      : Tensor(std::vector<int64_t>(shape), fill) {}

  const std::vector<int64_t>& Shape() const { return shape_; }
  int64_t Rank() const { return static_cast<int64_t>(shape_.size()); }
  int64_t Dim(int64_t i) const { return shape_[static_cast<size_t>(i)]; }
  int64_t Size() const { return size_; }
  bool Empty() const { return size_ == 0; }

  double* Data() { return data_.data(); }
  const double* Data() const { return data_.data(); }

  double& operator[](int64_t i) { return data_[static_cast<size_t>(i)]; }
  double operator[](int64_t i) const { return data_[static_cast<size_t>(i)]; }

  /// Row pointer for a rank-2 tensor.
  double* Row(int64_t r) { return data_.data() + r * shape_[1]; }
  const double* Row(int64_t r) const { return data_.data() + r * shape_[1]; }

  double& At(int64_t i, int64_t j) {
    return data_[static_cast<size_t>(i * shape_[1] + j)];
  }
  double At(int64_t i, int64_t j) const {
    return data_[static_cast<size_t>(i * shape_[1] + j)];
  }
  double& At(int64_t i, int64_t j, int64_t k) {
    return data_[static_cast<size_t>((i * shape_[1] + j) * shape_[2] + k)];
  }
  double At(int64_t i, int64_t j, int64_t k) const {
    return data_[static_cast<size_t>((i * shape_[1] + j) * shape_[2] + k)];
  }

  void Fill(double v);
  bool SameShape(const Tensor& other) const { return shape_ == other.shape_; }

  /// Shape as "128x4096". Used in manifests and error messages.
  std::string ShapeString() const;

  /// A view of rows [begin, end) of a rank-2 tensor, copied out.
  Tensor RowSlice(int64_t begin, int64_t end) const;

 private:
  std::vector<int64_t> shape_;
  std::vector<double> data_;
  int64_t size_ = 0;
};

/// Elementwise a += b. Shapes must match exactly.
void AddInPlace(Tensor* a, const Tensor& b);

/// Elementwise a * b into out. Shapes must match exactly.
Tensor Multiply(const Tensor& a, const Tensor& b);

/// Scalar multiply, in place.
void ScaleInPlace(Tensor* a, double s);

/// FNV-1a over the raw little-endian float64 payload. Two oracle runs that
/// agree bit for bit produce the same value; it is a drift detector, not a
/// tolerance check.
uint64_t Checksum(const Tensor& t);

}  // namespace oracle
