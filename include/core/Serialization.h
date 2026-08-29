#pragma once

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "core/Container.h"
#include "core/EvkMap.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Reading and writing CKKS key material and ciphertexts as bytes.
 *
 * ## Why this exists
 *
 * The library's own account of what it lacks put it plainly: keys and
 * ciphertexts cannot be saved, so every process regenerates them. For a unit
 * test that is a few seconds. For the conjugate-invariant Llama-3 layer it is
 * the whole of the run -- of the ~840 s a layer takes end to end, ~823 s is
 * one-time preparation, and the attention leg's three `CiSinCConverter`s alone
 * are 728-803 s of host-side matrix construction and encoding -- against ~10 s
 * of GPU-online arithmetic. A 32-layer model reuses every one of those objects
 * unchanged, so the preparation is not per layer; but it is per PROCESS, which
 * is what makes iterating on the model cost a quarter of an hour a turn and
 * what makes a 32-layer run impossible to develop against.
 *
 * ## The one thing an archive must not do
 *
 * A CKKS plaintext, ciphertext or evaluation key is a device buffer of RNS
 * limbs and records NOTHING about the parameter set it was built against.
 * Handed limbs from another preset it will not fail: it will decode as noise,
 * or -- worse -- as a plausible wrong answer. So every archive carries an
 * `ArchiveIdentity` in its header and refuses to open against a mismatched
 * one. The digest covers the prime lists and the level configuration, which is
 * what actually determines limb layout; the degree, the word width and the
 * ring's conjugate-invariance are stored beside it because a mismatch there
 * has a legible diagnostic worth printing.
 *
 * This is the rule `BootContext::PrepareEvalSpecialFFT`'s CtS donation already
 * follows, for the same reason: check the INPUTS that determine the bytes,
 * because the bytes themselves cannot be checked.
 *
 * ## Format
 *
 * A header, then records in the order the caller asks for them. There is no
 * index and no random access: an archive is read back by the same code path
 * that wrote it, in the same order, and a per-record tag says so. That keeps
 * the format small enough to be obviously correct and costs nothing, because
 * every consumer here loads a whole key set at once.
 *
 * Words are written in the machine's own byte order. These files are a cache
 * beside a build directory, not an interchange format; a cross-endian reader
 * would be untestable here and is not pretended at.
 */

/**
 * @brief What a saved file must agree with before anything is read out of it.
 */
struct ArchiveIdentity {
  uint32_t word_bytes = 0;
  uint32_t log_degree = 0;
  uint32_t conjugate_invariant = 0;
  // Over the main / terminal / auxiliary prime lists and the level config.
  uint64_t param_digest = 0;

  bool operator==(const ArchiveIdentity &other) const;
  bool operator!=(const ArchiveIdentity &other) const;
  std::string Describe() const;
};

/**
 * @brief The identity of the parameter set `param` describes.
 */
template <typename word>
ArchiveIdentity IdentityOf(const Parameter<word> &param);

/**
 * @brief Sequential binary writer. Creates or truncates `path`.
 */
class ArchiveWriter {
 public:
  ArchiveWriter(const std::string &path, const ArchiveIdentity &id);
  ~ArchiveWriter();

  ArchiveWriter(const ArchiveWriter &) = delete;
  ArchiveWriter &operator=(const ArchiveWriter &) = delete;

  /**
   * @brief Write a record tag. The reader's `Tag` must see the same string.
   *
   * This is what turns a caller whose write order and read order have drifted
   * apart from silent corruption into an assertion naming the record it went
   * wrong at.
   */
  void Tag(const std::string &tag);

  void Bytes(const void *data, size_t bytes);

  template <typename T>
  void Pod(const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Pod: only trivially copyable types");
    Bytes(&value, sizeof(T));
  }

  template <typename T>
  void Vec(const std::vector<T> &values) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Vec: only trivially copyable element types");
    Pod<uint64_t>(static_cast<uint64_t>(values.size()));
    if (!values.empty()) Bytes(values.data(), values.size() * sizeof(T));
  }

  void IntSet(const std::set<int> &values);

  /** @brief Bytes written so far, header included. */
  int64_t Written() const;

  /** @brief Flush and close. The destructor does it; explicit for timing. */
  void Close();

 private:
  std::string path_;
  std::ofstream os_;
  int64_t written_ = 0;
};

/**
 * @brief Sequential binary reader. Exits with a diagnostic on any mismatch.
 */
class ArchiveReader {
 public:
  ArchiveReader(const std::string &path, const ArchiveIdentity &expected);
  ~ArchiveReader();

  ArchiveReader(const ArchiveReader &) = delete;
  ArchiveReader &operator=(const ArchiveReader &) = delete;

  void Tag(const std::string &tag);

  void Bytes(void *data, size_t bytes);

  template <typename T>
  T Pod() {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Pod: only trivially copyable types");
    T value{};
    Bytes(&value, sizeof(T));
    return value;
  }

  template <typename T>
  std::vector<T> Vec() {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Vec: only trivially copyable element types");
    const auto size = Pod<uint64_t>();
    std::vector<T> values(size);
    if (size != 0) Bytes(values.data(), size * sizeof(T));
    return values;
  }

  std::set<int> IntSet();

  int64_t Read() const;

  /** @brief Whether `path` exists and can be opened for reading. */
  static bool Exists(const std::string &path);

  /**
   * @brief The identity recorded in `path`, without committing to reading it.
   *
   * Returns a zeroed identity when the file is missing, too short, or not an
   * archive of this format generation. This exists so that the refusal above
   * is *testable*: opening a mismatched archive exits the process, which a
   * test cannot observe, and a safety property nothing can check is a claim
   * rather than a guarantee. It is also how a caller decides whether a cache
   * is usable before paying for it.
   */
  static ArchiveIdentity PeekIdentity(const std::string &path);

  /** @brief Size of `path` in bytes, or -1 if it cannot be read. */
  static int64_t FileSize(const std::string &path);

 private:
  std::string path_;
  std::ifstream is_;
  int64_t read_ = 0;
};

// -- device buffers ----------------------------------------------------------

/**
 * @brief Write a device buffer, staged through host memory one buffer at a
 * time.
 *
 * The staging buffer is the size of the vector being written, which for every
 * container here is one polynomial's limbs -- a few MiB -- so saving a 16 GiB
 * key set never needs 16 GiB of host memory.
 */
template <typename word>
void SaveDeviceVector(ArchiveWriter &ar, const DeviceVector<word> &dv);

template <typename word>
void LoadDeviceVector(ArchiveReader &ar, DeviceVector<word> &dv);

// -- containers --------------------------------------------------------------

template <typename word>
void SaveContainer(ArchiveWriter &ar, const Plaintext<word> &pt);
template <typename word>
void LoadContainer(ArchiveReader &ar, Plaintext<word> &pt);

template <typename word>
void SaveContainer(ArchiveWriter &ar, const Ciphertext<word> &ct);
template <typename word>
void LoadContainer(ArchiveReader &ar, Ciphertext<word> &ct);

template <typename word>
void SaveContainer(ArchiveWriter &ar, const Constant<word> &c);
template <typename word>
void LoadContainer(ArchiveReader &ar, Constant<word> &c);

template <typename word>
void SaveContainer(ArchiveWriter &ar, const EvaluationKey<word> &evk);
template <typename word>
void LoadContainer(ArchiveReader &ar, EvaluationKey<word> &evk);

// -- whole key maps ----------------------------------------------------------

/**
 * @brief Write every key in `map`, smallest index first.
 *
 * Ordered so that two runs which requested the same keys produce identical
 * files, which is what makes a saved key set checkable by hash.
 */
template <typename word>
void SaveEvkMap(ArchiveWriter &ar, const EvkMap<word> &map);

/**
 * @brief Read keys into `map`, ADDING to whatever it already holds.
 *
 * Adding rather than replacing is what the Llama layer needs: its four rings
 * and its two BootContexts share one map, so a key set is assembled out of
 * several archives.
 */
template <typename word>
void LoadEvkMap(ArchiveReader &ar, EvkMap<word> &map);

}  // namespace cheddar
