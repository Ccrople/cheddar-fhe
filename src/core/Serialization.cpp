#include "core/Serialization.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>

#include "common/Assert.h"

namespace cheddar {

namespace {

// 'CHDRARC' plus a format generation. Bump the last byte when a record layout
// changes: an old cache then fails to open instead of being read as the new
// layout, which is the whole point of having a magic at all.
constexpr char kMagic[8] = {'C', 'H', 'D', 'R', 'A', 'R', 'C', '1'};

// FNV-1a. Not a cryptographic hash and not asked to be one: it separates
// parameter sets that differ, which is what the identity check needs.
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void DigestBytes(uint64_t &h, const void *data, size_t bytes) {
  const auto *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < bytes; i++) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= kFnvPrime;
  }
}

template <typename T>
void DigestPod(uint64_t &h, const T &value) {
  DigestBytes(h, &value, sizeof(T));
}

// Every archive read and write goes through host staging, and the copies are
// issued on the legacy stream. Nothing here may look at host bytes the device
// has not finished writing, nor free a staging buffer the device is still
// reading, so both directions synchronise.
void SyncLegacyStream() {
  const cudaError_t err = cudaStreamSynchronize(cudaStreamLegacy);
  AssertTrue(err == cudaSuccess,
             std::string("Serialization: stream synchronise failed: ") +
                 cudaGetErrorString(err));
}

}  // namespace

// -- identity ----------------------------------------------------------------

bool ArchiveIdentity::operator==(const ArchiveIdentity &other) const {
  return word_bytes == other.word_bytes && log_degree == other.log_degree &&
         conjugate_invariant == other.conjugate_invariant &&
         param_digest == other.param_digest;
}

bool ArchiveIdentity::operator!=(const ArchiveIdentity &other) const {
  return !(*this == other);
}

std::string ArchiveIdentity::Describe() const {
  std::ostringstream os;
  os << "word " << (word_bytes * 8) << "-bit, logN " << log_degree
     << (conjugate_invariant != 0 ? ", conjugate-invariant" : ", ordinary")
     << ", digest 0x" << std::hex << param_digest;
  return os.str();
}

template <typename word>
ArchiveIdentity IdentityOf(const Parameter<word> &param) {
  ArchiveIdentity id;
  id.word_bytes = static_cast<uint32_t>(sizeof(word));
  id.log_degree = static_cast<uint32_t>(param.log_degree_);
  id.conjugate_invariant = param.conjugate_invariant_ ? 1u : 0u;

  uint64_t h = kFnvOffset;
  DigestPod(h, id.word_bytes);
  DigestPod(h, id.log_degree);
  DigestPod(h, id.conjugate_invariant);
  // The prime lists and the level configuration are what determine the limb
  // layout of every buffer in the system, so they are what the digest is over.
  for (const auto &p : param.main_primes_) DigestPod(h, p);
  DigestBytes(h, "|main", 5);
  for (const auto &p : param.ter_primes_) DigestPod(h, p);
  DigestBytes(h, "|ter", 4);
  for (const auto &p : param.aux_primes_) DigestPod(h, p);
  DigestBytes(h, "|aux", 4);
  for (const auto &lc : param.level_config_) {
    DigestPod(h, lc.first);
    DigestPod(h, lc.second);
  }
  id.param_digest = h;
  return id;
}

// -- writer ------------------------------------------------------------------

ArchiveWriter::ArchiveWriter(const std::string &path, const ArchiveIdentity &id)
    : path_(path), os_(path, std::ios::binary | std::ios::trunc) {
  AssertTrue(os_.is_open(), "ArchiveWriter: cannot open " + path);
  Bytes(kMagic, sizeof(kMagic));
  Pod(id.word_bytes);
  Pod(id.log_degree);
  Pod(id.conjugate_invariant);
  Pod(id.param_digest);
}

ArchiveWriter::~ArchiveWriter() {
  if (os_.is_open()) os_.close();
}

void ArchiveWriter::Tag(const std::string &tag) {
  AssertTrue(tag.size() <= 255, "ArchiveWriter: tag too long: " + tag);
  Pod<uint8_t>(static_cast<uint8_t>(tag.size()));
  if (!tag.empty()) Bytes(tag.data(), tag.size());
}

void ArchiveWriter::Bytes(const void *data, size_t bytes) {
  if (bytes == 0) return;
  os_.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  AssertTrue(os_.good(), "ArchiveWriter: write failed on " + path_ +
                             " (disk full, or the path is not writable)");
  written_ += static_cast<int64_t>(bytes);
}

void ArchiveWriter::IntSet(const std::set<int> &values) {
  Pod<uint64_t>(static_cast<uint64_t>(values.size()));
  for (int v : values) Pod<int32_t>(static_cast<int32_t>(v));
}

int64_t ArchiveWriter::Written() const { return written_; }

void ArchiveWriter::Close() {
  if (!os_.is_open()) return;
  os_.flush();
  AssertTrue(os_.good(), "ArchiveWriter: flush failed on " + path_);
  os_.close();
}

// -- reader ------------------------------------------------------------------

ArchiveReader::ArchiveReader(const std::string &path,
                             const ArchiveIdentity &expected)
    : path_(path), is_(path, std::ios::binary) {
  AssertTrue(is_.is_open(), "ArchiveReader: cannot open " + path);

  char magic[sizeof(kMagic)];
  Bytes(magic, sizeof(magic));
  AssertTrue(std::memcmp(magic, kMagic, sizeof(kMagic)) == 0,
             "ArchiveReader: " + path +
                 " is not a Cheddar archive of this format generation");

  ArchiveIdentity found;
  found.word_bytes = Pod<uint32_t>();
  found.log_degree = Pod<uint32_t>();
  found.conjugate_invariant = Pod<uint32_t>();
  found.param_digest = Pod<uint64_t>();

  // The limbs themselves carry no parameter set, so this is the only place a
  // cross-preset load can be caught. Refusing here is the difference between a
  // diagnostic and a plausible wrong answer.
  AssertTrue(found == expected,
             "ArchiveReader: " + path +
                 " was written for a different parameter set.\n  file: " +
                 found.Describe() + "\n  want: " + expected.Describe());
}

ArchiveReader::~ArchiveReader() {
  if (is_.is_open()) is_.close();
}

void ArchiveReader::Tag(const std::string &tag) {
  const auto len = Pod<uint8_t>();
  std::string found(len, '\0');
  if (len != 0) Bytes(&found[0], len);
  AssertTrue(found == tag, "ArchiveReader: " + path_ + " expected record '" +
                               tag + "' but found '" + found +
                               "'; the write order and the read order differ");
}

void ArchiveReader::Bytes(void *data, size_t bytes) {
  if (bytes == 0) return;
  is_.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes));
  AssertTrue(is_.good(), "ArchiveReader: read failed on " + path_ +
                             " (truncated file?)");
  read_ += static_cast<int64_t>(bytes);
}

std::set<int> ArchiveReader::IntSet() {
  const auto size = Pod<uint64_t>();
  std::set<int> values;
  for (uint64_t i = 0; i < size; i++) values.insert(Pod<int32_t>());
  return values;
}

int64_t ArchiveReader::Read() const { return read_; }

bool ArchiveReader::Exists(const std::string &path) {
  std::ifstream probe(path, std::ios::binary);
  return probe.is_open();
}

ArchiveIdentity ArchiveReader::PeekIdentity(const std::string &path) {
  ArchiveIdentity id;
  std::ifstream probe(path, std::ios::binary);
  if (!probe.is_open()) return id;

  char magic[sizeof(kMagic)];
  probe.read(magic, sizeof(magic));
  if (!probe.good() || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return ArchiveIdentity{};
  }
  probe.read(reinterpret_cast<char *>(&id.word_bytes), sizeof(id.word_bytes));
  probe.read(reinterpret_cast<char *>(&id.log_degree), sizeof(id.log_degree));
  probe.read(reinterpret_cast<char *>(&id.conjugate_invariant),
             sizeof(id.conjugate_invariant));
  probe.read(reinterpret_cast<char *>(&id.param_digest),
             sizeof(id.param_digest));
  if (!probe.good()) return ArchiveIdentity{};
  return id;
}

int64_t ArchiveReader::FileSize(const std::string &path) {
  std::ifstream probe(path, std::ios::binary | std::ios::ate);
  if (!probe.is_open()) return -1;
  return static_cast<int64_t>(probe.tellg());
}

// -- device buffers ----------------------------------------------------------

template <typename word>
void SaveDeviceVector(ArchiveWriter &ar, const DeviceVector<word> &dv) {
  const uint64_t size = static_cast<uint64_t>(dv.size());
  ar.Pod(size);
  if (size == 0) return;
  HostVector<word> host;
  CopyDeviceToHost(host, dv);
  SyncLegacyStream();
  ar.Bytes(host.data(), size * sizeof(word));
}

template <typename word>
void LoadDeviceVector(ArchiveReader &ar, DeviceVector<word> &dv) {
  const auto size = ar.Pod<uint64_t>();
  dv.resize(static_cast<int>(size));
  if (size == 0) return;
  HostVector<word> host(size);
  ar.Bytes(host.data(), size * sizeof(word));
  CopyHostToDevice(dv, host);
  // The copy is asynchronous and `host` dies at the end of this scope.
  SyncLegacyStream();
}

// -- containers --------------------------------------------------------------

namespace {

void SaveNP(ArchiveWriter &ar, const NPInfo &np) {
  ar.Pod<int32_t>(np.num_main_);
  ar.Pod<int32_t>(np.num_ter_);
  ar.Pod<int32_t>(np.num_aux_);
  ar.Pod<int32_t>(np.degree_);
}

NPInfo LoadNP(ArchiveReader &ar) {
  const int num_main = ar.Pod<int32_t>();
  const int num_ter = ar.Pod<int32_t>();
  const int num_aux = ar.Pod<int32_t>();
  const int degree = ar.Pod<int32_t>();
  return NPInfo(num_main, num_ter, num_aux, degree);
}

}  // namespace

template <typename word>
void SaveContainer(ArchiveWriter &ar, const Plaintext<word> &pt) {
  ar.Tag("pt");
  SaveNP(ar, pt.GetNP());
  ar.Pod<double>(pt.GetScale());
  ar.Pod<int32_t>(pt.GetNumSlots());
  SaveDeviceVector(ar, pt.mx_);
}

template <typename word>
void LoadContainer(ArchiveReader &ar, Plaintext<word> &pt) {
  ar.Tag("pt");
  const NPInfo np = LoadNP(ar);
  const double scale = ar.Pod<double>();
  const int num_slots = ar.Pod<int32_t>();
  pt.ModifyNP(np);
  pt.SetScale(scale);
  // ModifyNP defaults a zero slot count to degree / 2; only override it when
  // the file actually recorded one, since SetNumSlots requires a power of two.
  if (num_slots > 0) pt.SetNumSlots(num_slots);
  LoadDeviceVector(ar, pt.mx_);
}

template <typename word>
void SaveContainer(ArchiveWriter &ar, const Ciphertext<word> &ct) {
  ar.Tag("ct");
  SaveNP(ar, ct.GetNP());
  ar.Pod<double>(ct.GetScale());
  ar.Pod<int32_t>(ct.GetNumSlots());
  ar.Pod<uint8_t>(ct.HasRx() ? 1 : 0);
  SaveDeviceVector(ar, ct.bx_);
  SaveDeviceVector(ar, ct.ax_);
  if (ct.HasRx()) SaveDeviceVector(ar, ct.rx_);
}

template <typename word>
void LoadContainer(ArchiveReader &ar, Ciphertext<word> &ct) {
  ar.Tag("ct");
  const NPInfo np = LoadNP(ar);
  const double scale = ar.Pod<double>();
  const int num_slots = ar.Pod<int32_t>();
  const bool has_rx = ar.Pod<uint8_t>() != 0;
  if (has_rx) {
    ct.PrepareRx();
  } else {
    ct.RemoveRx();
  }
  ct.ModifyNP(np);
  ct.SetScale(scale);
  if (num_slots > 0) ct.SetNumSlots(num_slots);
  LoadDeviceVector(ar, ct.bx_);
  LoadDeviceVector(ar, ct.ax_);
  if (has_rx) LoadDeviceVector(ar, ct.rx_);
}

template <typename word>
void SaveContainer(ArchiveWriter &ar, const Constant<word> &c) {
  ar.Tag("const");
  SaveNP(ar, c.GetNP());
  ar.Pod<double>(c.GetScale());
  SaveDeviceVector(ar, c.cx_);
}

template <typename word>
void LoadContainer(ArchiveReader &ar, Constant<word> &c) {
  ar.Tag("const");
  const NPInfo np = LoadNP(ar);
  const double scale = ar.Pod<double>();
  c.ModifyNP(np);
  c.SetScale(scale);
  LoadDeviceVector(ar, c.cx_);
}

template <typename word>
void SaveContainer(ArchiveWriter &ar, const EvaluationKey<word> &evk) {
  ar.Tag("evk");
  SaveNP(ar, evk.GetNP());
  const int beta = evk.GetBeta();
  ar.Pod<int32_t>(beta);
  for (int i = 0; i < beta; i++) SaveDeviceVector(ar, evk.bx_[i]);
  for (int i = 0; i < beta; i++) SaveDeviceVector(ar, evk.ax_[i]);
}

template <typename word>
void LoadContainer(ArchiveReader &ar, EvaluationKey<word> &evk) {
  ar.Tag("evk");
  const NPInfo np = LoadNP(ar);
  const int beta = ar.Pod<int32_t>();
  // A key's beta is fixed by its constructor, so rebuild rather than resize:
  // ModifyNP walks the existing beta and would silently keep a stale one.
  evk = EvaluationKey<word>(np, beta);
  for (int i = 0; i < beta; i++) LoadDeviceVector(ar, evk.bx_[i]);
  for (int i = 0; i < beta; i++) LoadDeviceVector(ar, evk.ax_[i]);
}

// -- whole key maps ----------------------------------------------------------

template <typename word>
void SaveEvkMap(ArchiveWriter &ar, const EvkMap<word> &map) {
  ar.Tag("evkmap");
  // std::unordered_map iterates in an unspecified order, so sort the indices:
  // two runs that requested the same keys then write identical files, which is
  // what makes a saved key set checkable by hash.
  std::vector<int> indices;
  indices.reserve(map.size());
  for (const auto &entry : map) indices.push_back(entry.first);
  std::sort(indices.begin(), indices.end());

  ar.Pod<uint64_t>(static_cast<uint64_t>(indices.size()));
  for (int idx : indices) {
    ar.Pod<int32_t>(idx);
    SaveContainer(ar, map.at(idx));
  }
}

template <typename word>
void LoadEvkMap(ArchiveReader &ar, EvkMap<word> &map) {
  ar.Tag("evkmap");
  const auto count = ar.Pod<uint64_t>();
  for (uint64_t i = 0; i < count; i++) {
    const int idx = ar.Pod<int32_t>();
    EvaluationKey<word> evk;
    LoadContainer(ar, evk);
    // Overwriting is deliberate: a caller assembling one map out of several
    // archives is the normal case here, and a repeated index means the later
    // archive owns that key.
    map.insert_or_assign(idx, std::move(evk));
  }
}

// Explicit instantiation of the templates
template ArchiveIdentity IdentityOf(const Parameter<uint32_t> &param);
template ArchiveIdentity IdentityOf(const Parameter<uint64_t> &param);

#define CHEDDAR_INSTANTIATE_SERIALIZATION(word)                              \
  template void SaveDeviceVector(ArchiveWriter &, const DeviceVector<word> &); \
  template void LoadDeviceVector(ArchiveReader &, DeviceVector<word> &);     \
  template void SaveContainer(ArchiveWriter &, const Plaintext<word> &);     \
  template void LoadContainer(ArchiveReader &, Plaintext<word> &);           \
  template void SaveContainer(ArchiveWriter &, const Ciphertext<word> &);    \
  template void LoadContainer(ArchiveReader &, Ciphertext<word> &);          \
  template void SaveContainer(ArchiveWriter &, const Constant<word> &);      \
  template void LoadContainer(ArchiveReader &, Constant<word> &);            \
  template void SaveContainer(ArchiveWriter &, const EvaluationKey<word> &); \
  template void LoadContainer(ArchiveReader &, EvaluationKey<word> &);       \
  template void SaveEvkMap(ArchiveWriter &, const EvkMap<word> &);           \
  template void LoadEvkMap(ArchiveReader &, EvkMap<word> &);

CHEDDAR_INSTANTIATE_SERIALIZATION(uint32_t)
CHEDDAR_INSTANTIATE_SERIALIZATION(uint64_t)

#undef CHEDDAR_INSTANTIATE_SERIALIZATION

}  // namespace cheddar
