#include "extension/StripedMatrix.h"

#include "common/Assert.h"
#include "common/ParallelFor.h"

namespace cheddar {

StripedMatrix::StripedMatrix(int height /*= 0*/, int width /*= 0*/)
    : Base(), height_(height), width_(width) {}

int StripedMatrix::GetHeight() const { return height_; }

int StripedMatrix::GetWidth() const { return width_; }

int StripedMatrix::GetNumDiag() const { return this->size(); }

StripedMatrix StripedMatrix::Mult(const StripedMatrix &a,
                                  const StripedMatrix &b) {
  AssertTrue(a.width_ == a.height_ && b.width_ == b.height_,
             "StripedMatrix must be square to perform multiplication");
  AssertTrue(a.GetHeight() == b.GetWidth(), "StripedMatrix dimension mismatch");
  int width = a.width_;

  StripedMatrix c(width, width);

  // Every (i, j) pair adds into diagonal (i + j) % width at every entry k, so
  // the entry index is the axis the work splits on: a thread owning a range
  // of k writes entries no other thread reads or writes, and each entry still
  // takes its terms in the (i, j) order the serial loop took them.
  struct Term {
    const Complex *a;
    const Complex *b;
    Complex *c;
    int shift;
  };
  std::vector<Term> terms;
  for (const auto &[i, diag_a] : a) {
    for (const auto &[j, diag_b] : b) {
      const int dest_idx = (i + j) % width;
      c.try_emplace(dest_idx, std::vector<Complex>(width));
      terms.push_back({diag_a.data(), diag_b.data(), c[dest_idx].data(), i});
    }
  }
  ParallelFor(width, [&](int k0, int k1) {
    for (const auto &t : terms) {
      for (int k = k0; k < k1; k++) {
        t.c[k] += t.a[k] * t.b[(k + t.shift) % width];
      }
    }
  });
  return c;
}

StripedMatrix StripedMatrix::Mult(const StripedMatrix &a, const Complex b) {
  StripedMatrix c(a.GetHeight(), a.GetWidth());
  for (const auto &[i, diag] : a) {
    int diag_size = diag.size();
    c.try_emplace(i, diag_size);
    for (int j = 0; j < diag_size; j++) {
      c[i][j] = diag[j] * b;
    }
  }
  return c;
}

}  // namespace cheddar