// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0

#include "oracle/Serialize.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace oracle {
namespace {

const char kMagic[8] = {'C', 'H', 'D', 'O', 'R', 'C', '1', '\0'};
constexpr uint32_t kVersion = 1;

void PutU32(std::string* b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b->push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void PutU64(std::string* b, uint64_t v) {
  for (int i = 0; i < 8; ++i) b->push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
bool GetU32(const std::string& b, size_t* off, uint32_t* v) {
  if (*off + 4 > b.size()) return false;
  *v = 0;
  for (int i = 0; i < 4; ++i)
    *v |= static_cast<uint32_t>(static_cast<unsigned char>(b[*off + static_cast<size_t>(i)]))
          << (8 * i);
  *off += 4;
  return true;
}
bool GetU64(const std::string& b, size_t* off, uint64_t* v) {
  if (*off + 8 > b.size()) return false;
  *v = 0;
  for (int i = 0; i < 8; ++i)
    *v |= static_cast<uint64_t>(static_cast<unsigned char>(b[*off + static_cast<size_t>(i)]))
          << (8 * i);
  *off += 8;
  return true;
}

uint64_t Fnv1a(const char* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

Json MetaToJson(const TensorMeta& m) {
  Json j = Json::Object();
  j.Set("dims", m.dims);
  j.Set("description", m.description);
  j.Set("consumer", m.consumer);
  j.Set("fit_role", m.fit_role);
  j.Set("layer", m.layer);
  return j;
}

}  // namespace

std::string TensorFileName(const std::string& name) {
  std::string out;
  out.reserve(name.size() + 8);
  for (char c : name) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    out.push_back(ok ? c : '_');
  }
  out += ".tensor";
  return out;
}

bool EnsureDirectory(const std::string& dir, std::string* error) {
  std::error_code ec;
  if (std::filesystem::exists(dir, ec)) {
    if (std::filesystem::is_directory(dir, ec)) return true;
    if (error) *error = dir + " exists and is not a directory";
    return false;
  }
  if (!std::filesystem::create_directories(dir, ec)) {
    if (error) *error = "cannot create directory " + dir + ": " + ec.message();
    return false;
  }
  return true;
}

bool WriteTensorFile(const std::string& path, const std::string& name,
                     const Tensor& t, const TensorMeta& meta,
                     TensorDType dtype, std::string* error) {
  std::string header;
  header.append(kMagic, sizeof(kMagic));
  PutU32(&header, kVersion);
  PutU32(&header, static_cast<uint32_t>(dtype));
  PutU32(&header, static_cast<uint32_t>(t.Rank()));
  PutU32(&header, 0);
  for (int64_t d : t.Shape()) PutU64(&header, static_cast<uint64_t>(d));
  PutU32(&header, static_cast<uint32_t>(name.size()));
  header += name;
  const std::string meta_json = MetaToJson(meta).Dump(0);
  PutU32(&header, static_cast<uint32_t>(meta_json.size()));
  header += meta_json;

  std::string payload;
  if (dtype == TensorDType::kFloat64) {
    payload.resize(static_cast<size_t>(t.Size()) * sizeof(double));
    std::memcpy(&payload[0], t.Data(), payload.size());
  } else {
    payload.resize(static_cast<size_t>(t.Size()) * sizeof(float));
    for (int64_t i = 0; i < t.Size(); ++i) {
      const float f = static_cast<float>(t[i]);
      std::memcpy(&payload[static_cast<size_t>(i) * sizeof(float)], &f, sizeof(float));
    }
  }
  PutU64(&header, static_cast<uint64_t>(payload.size()));

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    if (error) *error = "cannot open for writing: " + path;
    return false;
  }
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  std::string tail;
  PutU64(&tail, Fnv1a(payload.data(), payload.size()));
  f.write(tail.data(), static_cast<std::streamsize>(tail.size()));
  if (!f) {
    if (error) *error = "write failed: " + path;
    return false;
  }
  return true;
}

bool ReadTensorFile(const std::string& path, Tensor* out, std::string* name,
                    std::string* meta_json, std::string* error) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  const std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::string b;
  b.resize(static_cast<size_t>(n));
  if (n > 0 && !f.read(&b[0], n)) {
    if (error) *error = "short read on " + path;
    return false;
  }

  size_t off = 0;
  if (b.size() < sizeof(kMagic) || std::memcmp(b.data(), kMagic, sizeof(kMagic)) != 0) {
    if (error) *error = path + ": bad magic (not a CHDORC1 tensor file)";
    return false;
  }
  off += sizeof(kMagic);
  uint32_t version = 0, dtype = 0, rank = 0, flags = 0;
  if (!GetU32(b, &off, &version) || !GetU32(b, &off, &dtype) ||
      !GetU32(b, &off, &rank) || !GetU32(b, &off, &flags)) {
    if (error) *error = path + ": truncated header";
    return false;
  }
  if (version != kVersion) {
    if (error) *error = path + ": version " + std::to_string(version) +
                        ", this build reads version " + std::to_string(kVersion);
    return false;
  }
  std::vector<int64_t> shape;
  for (uint32_t i = 0; i < rank; ++i) {
    uint64_t d = 0;
    if (!GetU64(b, &off, &d)) {
      if (error) *error = path + ": truncated dims";
      return false;
    }
    shape.push_back(static_cast<int64_t>(d));
  }
  uint32_t name_len = 0;
  if (!GetU32(b, &off, &name_len) || off + name_len > b.size()) {
    if (error) *error = path + ": truncated name";
    return false;
  }
  if (name) *name = b.substr(off, name_len);
  off += name_len;
  uint32_t meta_len = 0;
  if (!GetU32(b, &off, &meta_len) || off + meta_len > b.size()) {
    if (error) *error = path + ": truncated metadata";
    return false;
  }
  if (meta_json) *meta_json = b.substr(off, meta_len);
  off += meta_len;
  uint64_t payload_bytes = 0;
  if (!GetU64(b, &off, &payload_bytes)) {
    if (error) *error = path + ": truncated payload length";
    return false;
  }
  if (off + payload_bytes + 8 > b.size()) {
    if (error) *error = path + ": file is shorter than its declared payload";
    return false;
  }

  int64_t count = 1;
  for (int64_t d : shape) count *= d;
  const size_t elem = dtype == 0 ? sizeof(double) : sizeof(float);
  if (payload_bytes != static_cast<uint64_t>(count) * elem) {
    if (error) *error = path + ": payload size does not match the shape";
    return false;
  }

  const uint64_t want = Fnv1a(b.data() + off, static_cast<size_t>(payload_bytes));
  size_t tail_off = off + static_cast<size_t>(payload_bytes);
  uint64_t got = 0;
  GetU64(b, &tail_off, &got);
  if (got != want) {
    if (error) *error = path + ": payload checksum mismatch (file is corrupt)";
    return false;
  }

  *out = Tensor(shape);
  if (dtype == 0) {
    std::memcpy(out->Data(), b.data() + off, static_cast<size_t>(payload_bytes));
  } else {
    for (int64_t i = 0; i < count; ++i) {
      float v;
      std::memcpy(&v, b.data() + off + static_cast<size_t>(i) * sizeof(float),
                  sizeof(float));
      (*out)[i] = static_cast<double>(v);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------

FileSink::FileSink(std::string dir, TensorDType dtype)
    : dir_(std::move(dir)), dtype_(dtype) {}

bool FileSink::Wants(const std::string& name) const {
  if (filter_.empty()) return true;
  for (const std::string& s : filter_)
    if (name.find(s) != std::string::npos) return true;
  return false;
}

void FileSink::Emit(const std::string& name, const Tensor& value,
                    const TensorMeta& meta) {
  TensorRecord r;
  r.name = name;
  r.file = TensorFileName(name);
  r.shape = value.Shape();
  r.meta = meta;
  r.dtype = dtype_;
  r.count = value.Size();

  double sum = 0.0, sum_sq = 0.0;
  for (int64_t i = 0; i < value.Size(); ++i) {
    const double v = value[i];
    if (i == 0 || v < r.min) r.min = v;
    if (i == 0 || v > r.max) r.max = v;
    sum += v;
    sum_sq += v * v;
  }
  if (value.Size() > 0) {
    r.mean = sum / static_cast<double>(value.Size());
    r.rms = std::sqrt(sum_sq / static_cast<double>(value.Size()));
  }
  r.checksum = Checksum(value);

  std::string err;
  if (!WriteTensorFile(dir_ + "/" + r.file, name, value, meta, dtype_, &err))
    errors_.push_back(err);
  records_.push_back(std::move(r));
}

Json TensorRecord::ToJson() const {
  Json j = Json::Object();
  j.Set("name", name);
  j.Set("file", file);
  Json dims = Json::Array();
  for (int64_t d : shape) dims.Push(Json::Of(d));
  j.Set("shape", dims);
  j.Set("dims", meta.dims);
  j.Set("description", meta.description);
  j.Set("consumer", meta.consumer);
  j.Set("fit_role", meta.fit_role);
  j.Set("layer", meta.layer);
  j.Set("dtype", dtype == TensorDType::kFloat64 ? "float64" : "float32");
  j.Set("count", count);
  j.Set("checksum_fnv1a64", checksum);
  j.Set("min", min);
  j.Set("max", max);
  j.Set("mean", mean);
  j.Set("rms", rms);
  return j;
}

bool FileSink::WriteManifest(const Json& extra, std::string* error) const {
  Json m = extra;
  m.Set("format", "CHDORC1");
  m.Set("format_version", static_cast<int64_t>(1));
  m.Set("tensor_count", static_cast<int64_t>(records_.size()));
  Json arr = Json::Array();
  for (const TensorRecord& r : records_) arr.Push(r.ToJson());
  m.Set("tensors", arr);
  if (!errors_.empty()) {
    Json e = Json::Array();
    for (const std::string& s : errors_) e.Push(Json::Of(s));
    m.Set("write_errors", e);
  }

  const std::string path = dir_ + "/manifest.json";
  std::ofstream f(path, std::ios::trunc);
  if (!f) {
    if (error) *error = "cannot write " + path;
    return false;
  }
  const std::string text = m.Dump(2);
  f.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!f) {
    if (error) *error = "write failed: " + path;
    return false;
  }
  return true;
}

std::vector<std::string> DumpSet(const std::string& preset) {
  if (preset == "minimal") return {"x_in", "x_mid", "x_out"};
  if (preset == "full") return {};
  // "boundaries" -- everything except the three score-shaped tensors.
  std::vector<std::string> out;
  for (const BoundaryDecl& d : LayerBoundaries()) {
    if (d.suffix == "attn.scores" || d.suffix == "attn.exp_input" ||
        d.suffix == "attn.exp" || d.suffix == "attn.probs")
      continue;
    out.push_back(d.suffix);
  }
  return out;
}

std::string SerializationFormatDoc() {
  return
R"(Intermediate-tensor serialization format
=======================================
One tensor per file, magic "CHDORC1", version 1. All integers little-endian
unsigned, no padding anywhere.

  off  size           field
  0    8              magic "CHDORC1\0"
  8    4              uint32 version           = 1
  12   4              uint32 dtype             0 = float64, 1 = float32
  16   4              uint32 rank
  20   4              uint32 flags             = 0 (reserved, must be 0)
  24   8*rank         uint64 dims[], outermost first, row-major
  ..   4              uint32 name_len
  ..   name_len       tensor name, UTF-8
  ..   4              uint32 meta_len
  ..   meta_len       metadata, a UTF-8 JSON object
  ..   8              uint64 payload_bytes
  ..   payload_bytes  payload, row-major, little-endian
  ..   8              uint64 FNV-1a checksum of the payload bytes

The metadata object carries:
  dims          comma-separated axis names, e.g. "token,channel"
  description   what the tensor is
  consumer      which encrypted module is expected to produce or consume it
  fit_role      ""|"rsqrt"|"exp"|"reciprocal"|"silu" -- set when the tensor is
                the INPUT of a polynomial approximation
  layer         layer index, or -1

A directory of tensors also holds manifest.json: the configuration, the weight
source, the numerical conventions, and one record per tensor with its shape,
axis names, metadata, checksum and summary statistics (min/max/mean/rms). The
manifest is the contract; the .tensor files are the data.

Design notes, so the next session does not have to ask:
  - float64 is the default because the oracle IS the reference. float32 exists
    for bulk dumps where a factor of two in disk matters and the consumer is
    comparing against a ~1e-3 tolerance anyway.
  - The checksum detects corruption and detects drift between two oracle runs
    on the SAME host. It is not a cross-host equality test -- see
    NumericalConventions() point 8.
  - No compression, no chunking, no dependency. A reader is ~15 lines in any
    language.
)";
}

std::string PythonReaderSnippet() {
  return
R"(# Reading a .tensor file. No dependencies beyond numpy.
import json, struct
import numpy as np

def read_tensor(path):
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:8] == b"CHDORC1\x00", "not a CHDORC1 file"
    version, dtype, rank, flags = struct.unpack_from("<4I", blob, 8)
    assert version == 1, version
    off = 24
    dims = struct.unpack_from("<%dQ" % rank, blob, off)
    off += 8 * rank
    (name_len,) = struct.unpack_from("<I", blob, off); off += 4
    name = blob[off:off + name_len].decode(); off += name_len
    (meta_len,) = struct.unpack_from("<I", blob, off); off += 4
    meta = json.loads(blob[off:off + meta_len].decode() or "{}"); off += meta_len
    (payload_bytes,) = struct.unpack_from("<Q", blob, off); off += 8
    np_dtype = "<f8" if dtype == 0 else "<f4"
    array = np.frombuffer(blob, dtype=np_dtype, count=payload_bytes // (8 if dtype == 0 else 4),
                          offset=off).reshape(dims)
    return name, array, meta

# The manifest lists every tensor with its shape, metadata and checksum.
def read_manifest(directory):
    with open(directory + "/manifest.json") as f:
        return json.load(f)
)";
}

}  // namespace oracle
