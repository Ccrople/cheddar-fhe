// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Json.h"

#include <cmath>
#include <cstdio>

namespace oracle {

std::string JsonNumber(double v) {
  if (std::isnan(v) || std::isinf(v)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

std::string JsonQuote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += "\"";
  return out;
}

Json Json::Of(double v) {
  Json j(Kind::kNumber);
  j.scalar_ = JsonNumber(v);
  return j;
}

Json Json::Of(int64_t v) {
  Json j(Kind::kNumber);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
  j.scalar_ = buf;
  return j;
}

Json Json::Of(uint64_t v) {
  Json j(Kind::kNumber);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
  j.scalar_ = buf;
  return j;
}

Json Json::Of(bool v) {
  Json j(Kind::kBool);
  j.bool_ = v;
  return j;
}

Json Json::Of(const std::string& v) {
  Json j(Kind::kString);
  j.scalar_ = v;
  return j;
}

Json& Json::Set(const std::string& key, Json value) {
  for (auto& kv : members_) {
    if (kv.first == key) {
      kv.second = std::move(value);
      return *this;
    }
  }
  members_.emplace_back(key, std::move(value));
  return *this;
}

Json& Json::Push(Json value) {
  elements_.push_back(std::move(value));
  return *this;
}

std::string Json::Dump(int indent) const {
  std::string out;
  DumpTo(&out, indent, 0);
  out += "\n";
  return out;
}

void Json::DumpTo(std::string* out, int indent, int depth) const {
  const std::string pad(static_cast<size_t>(indent * depth), ' ');
  const std::string pad_in(static_cast<size_t>(indent * (depth + 1)), ' ');
  const char* nl = indent > 0 ? "\n" : "";
  switch (kind_) {
    case Kind::kNull: *out += "null"; return;
    case Kind::kBool: *out += bool_ ? "true" : "false"; return;
    case Kind::kNumber: *out += scalar_; return;
    case Kind::kString: *out += JsonQuote(scalar_); return;
    case Kind::kObject: {
      if (members_.empty()) { *out += "{}"; return; }
      *out += "{";
      *out += nl;
      for (size_t i = 0; i < members_.size(); ++i) {
        *out += pad_in;
        *out += JsonQuote(members_[i].first);
        *out += indent > 0 ? ": " : ":";
        members_[i].second.DumpTo(out, indent, depth + 1);
        if (i + 1 < members_.size()) *out += ",";
        *out += nl;
      }
      *out += pad;
      *out += "}";
      return;
    }
    case Kind::kArray: {
      if (elements_.empty()) { *out += "[]"; return; }
      *out += "[";
      *out += nl;
      for (size_t i = 0; i < elements_.size(); ++i) {
        *out += pad_in;
        elements_[i].DumpTo(out, indent, depth + 1);
        if (i + 1 < elements_.size()) *out += ",";
        *out += nl;
      }
      *out += pad;
      *out += "]";
      return;
    }
  }
}

}  // namespace oracle
