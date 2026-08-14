// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "Check.h"

#include <chrono>
#include <cstring>
#include <iostream>

namespace tlite {

std::vector<TestCase>& Registry() {
  static std::vector<TestCase> registry;
  return registry;
}

Registrar::Registrar(const char* name, std::function<void()> fn) {
  Registry().push_back({name, std::move(fn)});
}

void Fail(const char* file, int line, const std::string& msg) {
  std::ostringstream os;
  const char* base = file;
  for (const char* p = file; *p; ++p)
    if (*p == '/' || *p == '\\') base = p + 1;
  os << base << ":" << line << "\n    " << msg;
  throw Failure(os.str());
}

int RunAll(const char* filter) {
  int passed = 0, failed = 0, skipped = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const TestCase& tc : Registry()) {
    if (filter && *filter && tc.name.find(filter) == std::string::npos) {
      ++skipped;
      continue;
    }
    try {
      tc.fn();
      std::cout << "[  ok  ] " << tc.name << "\n";
      ++passed;
    } catch (const Failure& f) {
      std::cout << "[ FAIL ] " << tc.name << "\n  " << f.what() << "\n";
      ++failed;
    } catch (const std::exception& e) {
      std::cout << "[ FAIL ] " << tc.name
                << "\n    unexpected exception: " << e.what() << "\n";
      ++failed;
    }
  }
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  std::cout << "\n" << passed << " passed, " << failed << " failed";
  if (skipped) std::cout << ", " << skipped << " filtered out";
  std::cout << ", " << ms << " ms\n";
  return failed;
}

}  // namespace tlite

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  return tlite::RunAll(filter) == 0 ? 0 : 1;
}
