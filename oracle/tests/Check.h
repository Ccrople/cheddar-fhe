// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// A ~70-line test framework.
//
// Cheddar's own unit tests use googletest, which its `unittest/CMakeLists.txt`
// fetches from GitHub at configure time. The oracle must build with no network
// access, so it carries its own. If S0 later wires the oracle into the shared
// build, these tests can be re-expressed as gtest cases mechanically -- the
// macro names are deliberately the same shape.

#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tlite {

struct TestCase {
  std::string name;
  std::function<void()> fn;
};

std::vector<TestCase>& Registry();

struct Registrar {
  Registrar(const char* name, std::function<void()> fn);
};

/// Runs every registered case whose name contains `filter` (null = all).
/// Returns the number of failures.
int RunAll(const char* filter);

class Failure : public std::runtime_error {
 public:
  explicit Failure(const std::string& what) : std::runtime_error(what) {}
};

[[noreturn]] void Fail(const char* file, int line, const std::string& msg);

template <typename T>
std::string Show(const T& v) {
  std::ostringstream os;
  os.precision(17);
  os << v;
  return os.str();
}

}  // namespace tlite

#define TEST(suite, name)                                                    \
  static void suite##_##name();                                              \
  static ::tlite::Registrar reg_##suite##_##name(#suite "." #name,           \
                                                 suite##_##name);            \
  static void suite##_##name()

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) ::tlite::Fail(__FILE__, __LINE__, "CHECK failed: " #cond);  \
  } while (0)

#define CHECK_MSG(cond, msg)                                                 \
  do {                                                                       \
    if (!(cond))                                                             \
      ::tlite::Fail(__FILE__, __LINE__,                                      \
                    std::string("CHECK failed: " #cond " -- ") + (msg));     \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    const auto va_ = (a);                                                    \
    const auto vb_ = (b);                                                    \
    if (!(va_ == vb_))                                                       \
      ::tlite::Fail(__FILE__, __LINE__,                                      \
                    "CHECK_EQ failed: " #a " == " #b "\n    lhs = " +        \
                        ::tlite::Show(va_) + "\n    rhs = " +                \
                        ::tlite::Show(vb_));                                 \
  } while (0)

#define CHECK_LE(a, b)                                                       \
  do {                                                                       \
    const auto va_ = (a);                                                    \
    const auto vb_ = (b);                                                    \
    if (!(va_ <= vb_))                                                       \
      ::tlite::Fail(__FILE__, __LINE__,                                      \
                    "CHECK_LE failed: " #a " <= " #b "\n    lhs = " +        \
                        ::tlite::Show(va_) + "\n    rhs = " +                \
                        ::tlite::Show(vb_));                                 \
  } while (0)

#define CHECK_LT(a, b)                                                       \
  do {                                                                       \
    const auto va_ = (a);                                                    \
    const auto vb_ = (b);                                                    \
    if (!(va_ < vb_))                                                        \
      ::tlite::Fail(__FILE__, __LINE__,                                      \
                    "CHECK_LT failed: " #a " < " #b "\n    lhs = " +         \
                        ::tlite::Show(va_) + "\n    rhs = " +                \
                        ::tlite::Show(vb_));                                 \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
  do {                                                                       \
    const double va_ = (a);                                                  \
    const double vb_ = (b);                                                  \
    const double t_ = (tol);                                                 \
    if (!(std::fabs(va_ - vb_) <= t_))                                       \
      ::tlite::Fail(__FILE__, __LINE__,                                      \
                    "CHECK_NEAR failed: |" #a " - " #b "| <= " #tol          \
                    "\n    lhs = " + ::tlite::Show(va_) + "\n    rhs = " +   \
                        ::tlite::Show(vb_) + "\n    diff = " +               \
                        ::tlite::Show(std::fabs(va_ - vb_)));                \
  } while (0)

/// Relative comparison against the magnitude of `b`, with an absolute floor so
/// a zero reference is not an automatic failure.
#define CHECK_REL(a, b, rel)                                                 \
  do {                                                                       \
    const double va_ = (a);                                                  \
    const double vb_ = (b);                                                  \
    const double r_ = (rel);                                                 \
    const double den_ = std::fabs(vb_) > 1e-300 ? std::fabs(vb_) : 1.0;      \
    if (!(std::fabs(va_ - vb_) / den_ <= r_))                                \
      ::tlite::Fail(__FILE__, __LINE__,                                      \
                    "CHECK_REL failed: |" #a " - " #b "|/|" #b "| <= " #rel  \
                    "\n    lhs = " + ::tlite::Show(va_) + "\n    rhs = " +   \
                        ::tlite::Show(vb_) + "\n    rel = " +                \
                        ::tlite::Show(std::fabs(va_ - vb_) / den_));         \
  } while (0)

#define CHECK_THROWS(expr)                                                   \
  do {                                                                       \
    bool threw_ = false;                                                     \
    try {                                                                    \
      (void)(expr);                                                          \
    } catch (const std::exception&) {                                        \
      threw_ = true;                                                         \
    }                                                                        \
    if (!threw_)                                                             \
      ::tlite::Fail(__FILE__, __LINE__,                                      \
                    "CHECK_THROWS failed: " #expr " did not throw");         \
  } while (0)
