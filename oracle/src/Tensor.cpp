// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Tensor.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace oracle {

Tensor::Tensor(std::vector<int64_t> shape, double fill) : shape_(std::move(shape)) {
  size_ = 1;
  for (int64_t d : shape_) {
    if (d < 0) throw std::invalid_argument("Tensor: negative dimension");
    size_ *= d;
  }
  data_.assign(static_cast<size_t>(size_), fill);
}

void Tensor::Fill(double v) {
  for (auto& x : data_) x = v;
}

std::string Tensor::ShapeString() const {
  std::ostringstream os;
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i) os << "x";
    os << shape_[i];
  }
  if (shape_.empty()) os << "scalar";
  return os.str();
}

Tensor Tensor::RowSlice(int64_t begin, int64_t end) const {
  if (Rank() != 2) throw std::invalid_argument("RowSlice: rank must be 2");
  if (begin < 0 || end > shape_[0] || begin > end)
    throw std::invalid_argument("RowSlice: bad range");
  Tensor out({end - begin, shape_[1]});
  std::memcpy(out.Data(), Data() + begin * shape_[1],
              static_cast<size_t>((end - begin) * shape_[1]) * sizeof(double));
  return out;
}

void AddInPlace(Tensor* a, const Tensor& b) {
  if (!a->SameShape(b))
    throw std::invalid_argument("AddInPlace: shape mismatch " +
                                a->ShapeString() + " vs " + b.ShapeString());
  double* pa = a->Data();
  const double* pb = b.Data();
  for (int64_t i = 0; i < a->Size(); ++i) pa[i] += pb[i];
}

Tensor Multiply(const Tensor& a, const Tensor& b) {
  if (!a.SameShape(b))
    throw std::invalid_argument("Multiply: shape mismatch " + a.ShapeString() +
                                " vs " + b.ShapeString());
  Tensor out(a.Shape());
  for (int64_t i = 0; i < a.Size(); ++i) out[i] = a[i] * b[i];
  return out;
}

void ScaleInPlace(Tensor* a, double s) {
  double* p = a->Data();
  for (int64_t i = 0; i < a->Size(); ++i) p[i] *= s;
}

uint64_t Checksum(const Tensor& t) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
  const unsigned char* p = reinterpret_cast<const unsigned char*>(t.Data());
  const size_t n = static_cast<size_t>(t.Size()) * sizeof(double);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace oracle
