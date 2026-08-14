// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// A ~100-line JSON writer. The oracle must build with no network access and
// no third-party dependency, so nlohmann/json (which Cheddar's unittest tree
// fetches at configure time) is not available here.
//
// Write-only by design. Nothing in the oracle parses JSON; the report files
// are consumed by humans, by numpy, and by the C++ comparison utility, which
// reads the binary tensor files rather than the manifest.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace oracle {

class Json {
 public:
  static Json Object() { return Json(Kind::kObject); }
  static Json Array() { return Json(Kind::kArray); }
  static Json Null() { return Json(Kind::kNull); }
  static Json Of(double v);
  static Json Of(int64_t v);
  static Json Of(int v) { return Of(static_cast<int64_t>(v)); }
  static Json Of(uint64_t v);
  static Json Of(bool v);
  static Json Of(const std::string& v);
  static Json Of(const char* v) { return Of(std::string(v)); }

  /// Object insertion. Insertion order is preserved so diffs stay readable.
  Json& Set(const std::string& key, Json value);
  Json& Set(const std::string& key, double v) { return Set(key, Of(v)); }
  Json& Set(const std::string& key, int64_t v) { return Set(key, Of(v)); }
  Json& Set(const std::string& key, int v) { return Set(key, Of(v)); }
  Json& Set(const std::string& key, uint64_t v) { return Set(key, Of(v)); }
  Json& Set(const std::string& key, bool v) { return Set(key, Of(v)); }
  Json& Set(const std::string& key, const std::string& v) { return Set(key, Of(v)); }
  Json& Set(const std::string& key, const char* v) { return Set(key, Of(v)); }

  /// Array append.
  Json& Push(Json value);

  std::string Dump(int indent = 2) const;

 private:
  enum class Kind { kNull, kBool, kNumber, kString, kObject, kArray };
  explicit Json(Kind k) : kind_(k) {}

  void DumpTo(std::string* out, int indent, int depth) const;

  Kind kind_ = Kind::kNull;
  bool bool_ = false;
  std::string scalar_;  ///< pre-rendered number, or raw string content
  std::vector<std::pair<std::string, Json>> members_;
  std::vector<Json> elements_;
};

/// Escape a string as a JSON string literal, quotes included.
std::string JsonQuote(const std::string& s);

/// `%.17g`, with NaN and +-Inf rendered as JSON `null` (JSON has no other
/// spelling for them). Callers that care record a separate boolean.
std::string JsonNumber(double v);

}  // namespace oracle
