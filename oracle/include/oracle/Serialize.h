// Copyright 2026
// Licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
//
// The intermediate-tensor serialization format.
//
// ON-DISK FORMAT -- "CHDORC1", version 1
// --------------------------------------
// One tensor per file. All integers are little-endian unsigned; all offsets
// are byte offsets from the start of the file; there is no padding anywhere.
//
//   off  size          field
//   0    8             magic, the ASCII bytes "CHDORC1\0"
//   8    4             uint32 version               = 1
//   12   4             uint32 dtype                 0 = float64, 1 = float32
//   16   4             uint32 rank
//   20   4             uint32 flags                 = 0 (reserved, must be 0)
//   24   8*rank        uint64 dims[], outermost first, row-major
//   ..   4             uint32 name_len
//   ..   name_len      name, UTF-8, no terminator
//   ..   4             uint32 meta_len
//   ..   meta_len      metadata, a UTF-8 JSON object
//   ..   8             uint64 payload_bytes
//   ..   payload_bytes payload, row-major, little-endian
//   ..   8             uint64 checksum (FNV-1a over the payload bytes)
//
// Reading it from Python is three lines of struct.unpack plus
// numpy.frombuffer; reading it from C++ is ReadTensorFile below. There is
// deliberately no compression, no chunking and no dependency.
//
// A DIRECTORY of tensors additionally carries `manifest.json`, which lists
// every tensor with its shape, axis names, metadata, checksum and summary
// statistics. The manifest is the contract; the files are the data.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "oracle/Config.h"
#include "oracle/Json.h"
#include "oracle/Ops.h"
#include "oracle/Tensor.h"

namespace oracle {

enum class TensorDType : uint32_t { kFloat64 = 0, kFloat32 = 1 };

/// Everything the manifest records about one exported tensor.
struct TensorRecord {
  std::string name;
  std::string file;  ///< path relative to the manifest
  std::vector<int64_t> shape;
  TensorMeta meta;
  TensorDType dtype = TensorDType::kFloat64;
  uint64_t checksum = 0;
  double min = 0.0, max = 0.0, mean = 0.0, rms = 0.0;
  int64_t count = 0;

  Json ToJson() const;
};

/// Writes one tensor file. Returns false and fills `error` on any I/O failure.
bool WriteTensorFile(const std::string& path, const std::string& name,
                     const Tensor& t, const TensorMeta& meta,
                     TensorDType dtype, std::string* error);

/// Reads one tensor file written by WriteTensorFile. Both dtypes are promoted
/// to float64 on the way in. Verifies the magic, the version, the declared
/// payload size and the checksum, and names which of those failed.
bool ReadTensorFile(const std::string& path, Tensor* out, std::string* name,
                    std::string* meta_json, std::string* error);

/// Turns a tensor name into a filesystem-safe file name:
/// "L0.attn.exp_input" -> "L0.attn.exp_input.tensor". Characters outside
/// [A-Za-z0-9._-] become '_'.
std::string TensorFileName(const std::string& tensor_name);

/// A TensorSink that writes every accepted tensor to `dir` and accumulates a
/// manifest.
///
/// The name filter is what keeps a full-model dump from being hundreds of
/// gigabytes. `SetNameFilter` takes a list of substrings; a tensor is written
/// when its name contains any of them. An empty list accepts everything.
class FileSink : public TensorSink {
 public:
  FileSink(std::string dir, TensorDType dtype);

  void SetNameFilter(std::vector<std::string> substrings) {
    filter_ = std::move(substrings);
  }
  bool Wants(const std::string& name) const override;
  void Emit(const std::string& name, const Tensor& value,
            const TensorMeta& meta) override;

  const std::vector<TensorRecord>& Records() const { return records_; }
  const std::vector<std::string>& Errors() const { return errors_; }

  /// Writes `manifest.json` into the directory. `extra` is merged into the
  /// top level so the caller can record the configuration, the weight source
  /// and the numerical conventions alongside the data.
  bool WriteManifest(const Json& extra, std::string* error) const;

 private:
  std::string dir_;
  TensorDType dtype_;
  std::vector<std::string> filter_;
  std::vector<TensorRecord> records_;
  std::vector<std::string> errors_;
};

/// Named dump presets, so a caller does not have to know the boundary list.
///   "minimal"    x_in, x_mid, x_out only -- the block-level seams
///   "boundaries" every module boundary except the [heads,tokens,tokens]
///                score-shaped tensors (the default; ~1 MB per layer at the
///                8B width and 128 tokens, versus ~50 MB with them)
///   "full"       everything, including the score tensors
/// Returns the substring filter for FileSink::SetNameFilter.
std::vector<std::string> DumpSet(const std::string& preset);

/// Creates a directory if it does not exist. Returns false with a message on
/// failure. Uses std::filesystem.
bool EnsureDirectory(const std::string& dir, std::string* error);

/// The format specification as text, for the report and for `--help`.
std::string SerializationFormatDoc();

/// A ready-to-paste Python reader for the format above.
std::string PythonReaderSnippet();

}  // namespace oracle
