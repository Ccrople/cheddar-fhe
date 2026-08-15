#include "core/NPInfo.h"

#include "common/Assert.h"

namespace cheddar {

int NPInfo::GetNumQ() const { return num_main_ + num_ter_; }
int NPInfo::GetNumTotal() const { return num_main_ + num_ter_ + num_aux_; }

NPInfo::NPInfo(int num_main /*= 0*/, int num_ter /*= 0*/, int num_aux /*= 0*/,
               int degree /*= 0*/)
    : num_main_{num_main},
      num_ter_{num_ter},
      num_aux_{num_aux},
      degree_{degree} {
  AssertTrue(num_main >= 0, "Negative num_main given");
  AssertTrue(num_ter >= 0, "Negative num_ter given");
  AssertTrue(num_aux >= 0, "Negative num_aux given");
  AssertTrue(degree >= 0, "Negative degree given");
}

NPInfo::NPInfo(const NPInfo &other) {
  AssertTrue(other.num_main_ >= 0, "Negative num_main given");
  AssertTrue(other.num_ter_ >= 0, "Negative num_ter given");
  AssertTrue(other.num_aux_ >= 0, "Negative num_aux given");
  num_main_ = other.num_main_;
  num_ter_ = other.num_ter_;
  num_aux_ = other.num_aux_;
  degree_ = other.degree_;
}

NPInfo &NPInfo::operator=(const NPInfo &other) {
  AssertTrue(other.num_main_ >= 0, "Negative num_main given");
  AssertTrue(other.num_ter_ >= 0, "Negative num_ter given");
  AssertTrue(other.num_aux_ >= 0, "Negative num_aux given");
  num_main_ = other.num_main_;
  num_ter_ = other.num_ter_;
  num_aux_ = other.num_aux_;
  degree_ = other.degree_;
  return *this;
}

// The degree participates, so an operand from another ring fails the NP
// assertions the evaluation API already makes instead of being read at the
// wrong stride.
bool NPInfo::operator==(const NPInfo &other) const {
  return num_main_ == other.num_main_ && num_ter_ == other.num_ter_ &&
         num_aux_ == other.num_aux_ && degree_ == other.degree_;
}

// Subset and superset are about prime counts within one ring, so they require
// the ring to match rather than ordering by it.
bool NPInfo::IsSubsetOf(const NPInfo &other) const {
  return degree_ == other.degree_ && num_main_ <= other.num_main_ &&
         num_ter_ <= other.num_ter_ && num_aux_ <= other.num_aux_;
}

bool NPInfo::IsSupersetOf(const NPInfo &other) const {
  return degree_ == other.degree_ && num_main_ >= other.num_main_ &&
         num_ter_ >= other.num_ter_ && num_aux_ >= other.num_aux_;
}
}  // namespace cheddar