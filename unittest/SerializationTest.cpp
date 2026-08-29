// Keys, ciphertexts and compiled transforms, written to disk and read back.
//
// ## What this is for
//
// The library's own account of what it lacks said it plainly: keys and
// ciphertexts cannot be saved, so every process regenerates them. On the
// conjugate-invariant Llama-3 layer that is not a nuisance, it is the run --
// ~823 s of the ~840 s the layer takes is one-time preparation, and the
// attention leg's three `CiSinCConverter`s are 728-803 s of it, against ~10 s
// of GPU-online arithmetic. Those objects are identical for all 32 layers of
// the model, so the cost is per PROCESS, which is exactly what a cache fixes.
//
// ## What is checked, and against what
//
// Not "save it, load it, and compare the two" -- that would compare a run of
// this code against another run of the same code, and would pass just as well
// if both were wrong. Every case here decrypts through to a message and
// compares it with a value computed on the HOST:
//
//   - a ciphertext, against the plaintext vector it was encrypted from;
//   - a rotation key, by rotating with the LOADED map only, against the host's
//     own rotation of that vector;
//   - a compiled `LinearTransform`, against the host matrix-vector product,
//     with the freshly built transform measured beside it so that "the loaded
//     one is as good as the built one" is a measurement rather than a hope.
//
// The rotation case is the load-bearing one: an evaluation key that survived
// the round trip byte for byte but was mis-indexed would still decrypt to
// noise, and only using the loaded map in isolation can show it did not.
//
// ## The refusal is tested too, and needs a non-fatal probe to be
//
// An archive written for another parameter set must not open, because the
// limbs carry no parameter set of their own and a wrong one decodes as
// plausible noise rather than failing. `ArchiveReader`'s constructor enforces
// that by exiting, which a test cannot observe -- so `PeekIdentity` reads the
// header without committing, and `TheIdentitySeparatesParameterSets` uses it.
// A safety property nothing can check is a claim, not a guarantee.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/Serialization.h"
#include "extension/LinearTransform.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using Complex = std::complex<double>;
using cheddar::ArchiveIdentity;
using cheddar::ArchiveReader;
using cheddar::ArchiveWriter;
using cheddar::Ciphertext;
using cheddar::EvkMap;
using cheddar::EvkRequest;
using cheddar::LinearTransform;
using cheddar::Plaintext;

namespace {

// Degree 4096 and four levels: everything here is about bytes, not about the
// ring, so the ring should be the cheapest one that has a rotation key and a
// level to run a transform at.
constexpr const char *kParam = "ringdegree12_35.json";
// The same data on the real subring, to exercise the CI containers too.
constexpr const char *kCiParam = "ci12_35.json";

// gtest gives no scratch directory, so archives go beside the binary and are
// removed at the end of the test that wrote them.
std::string Scratch(const std::string &name) { return "./" + name; }

void Remove(const std::string &path) { std::remove(path.c_str()); }

double MaxAbsDiff(const std::vector<Complex> &a, const std::vector<Complex> &b,
                  int count) {
  double worst = 0.0;
  for (int i = 0; i < count; i++) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

double MaxAbs(const std::vector<Complex> &a, int count) {
  double mx = 0.0;
  for (int i = 0; i < count; i++) mx = std::max(mx, std::abs(a[i]));
  return mx;
}

std::vector<Complex> RandomMessage(int num_slots, bool real_only,
                                   uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<Complex> msg(num_slots);
  for (auto &v : msg) {
    v = real_only ? Complex(dist(rng), 0.0) : Complex(dist(rng), dist(rng));
  }
  return msg;
}

double MiB(int64_t bytes) { return static_cast<double>(bytes) / (1 << 20); }

}  // namespace

// ---------------------------------------------------------------------------
// A ciphertext, against the vector it was encrypted from.
// ---------------------------------------------------------------------------
TEST(Serialization, ACiphertextCarriesItsMessageThroughAFile) {
  for (const char *param : {kParam, kCiParam}) {
    Ring ring(param);
    const auto id = cheddar::IdentityOf(*ring.param);
    const int num_slots = ring.param->MaxNumSlots();
    const int level = ring.enc_level;
    const bool real_only = ring.param->conjugate_invariant_;

    const auto msg = RandomMessage(num_slots, real_only, 20260829u);
    Plaintext<word> pt;
    ring.context->encoder_.Encode(pt, level, ring.param->GetScale(level), msg);
    Ciphertext<word> ct;
    ring.ui->Encrypt(ct, pt);

    const std::string path = Scratch(std::string("ser_ct_") + param + ".bin");
    {
      ArchiveWriter ar(path, id);
      SaveContainer(ar, ct);
      ar.Close();
    }

    // A ciphertext the writer never touched, so nothing can be carried over in
    // memory: everything it holds has to have come out of the file.
    Ciphertext<word> loaded;
    {
      ArchiveReader ar(path, id);
      LoadContainer(ar, loaded);
    }

    EXPECT_EQ(loaded.GetNP(), ct.GetNP());
    EXPECT_EQ(loaded.GetNumSlots(), ct.GetNumSlots());
    EXPECT_DOUBLE_EQ(loaded.GetScale(), ct.GetScale());

    Plaintext<word> out_pt;
    ring.ui->Decrypt(out_pt, loaded);
    std::vector<Complex> got;
    ring.context->encoder_.Decode(got, out_pt);

    const double err = MaxAbsDiff(got, msg, num_slots);
    std::cout << "  [" << param << "] ciphertext round trip: " << err
              << " against |msg| <= " << MaxAbs(msg, num_slots) << ", "
              << MiB(ArchiveReader::FileSize(path)) << " MiB" << std::endl;
    // The reference is the host vector, so this is the encryption's own noise
    // and nothing else; a byte lost in transit would be orders above it.
    EXPECT_LT(err, 1e-5);
    Remove(path);
  }
}

// ---------------------------------------------------------------------------
// A rotation key, used in isolation, against the host's own rotation.
// ---------------------------------------------------------------------------
TEST(Serialization, ALoadedKeyRotatesWithoutTheOneThatWroteIt) {
  Ring ring(kParam);
  const auto id = cheddar::IdentityOf(*ring.param);
  const int num_slots = ring.param->MaxNumSlots();
  const int level = ring.enc_level;
  const int rot = 5;

  {
    EvkRequest req;
    req.AddRequest(rot, level);
    ring.ui->PrepareRotationKey(req);
  }

  const std::string path = Scratch("ser_evk.bin");
  {
    ArchiveWriter ar(path, id);
    SaveEvkMap(ar, ring.ui->GetEvkMap());
    ar.Close();
  }

  // A map of its own, so the rotation below cannot reach the generated key.
  EvkMap<word> loaded_map;
  {
    ArchiveReader ar(path, id);
    LoadEvkMap(ar, loaded_map);
  }
  EXPECT_EQ(loaded_map.size(), ring.ui->GetEvkMap().size());

  const auto msg = RandomMessage(num_slots, false, 20260830u);
  Plaintext<word> pt;
  ring.context->encoder_.Encode(pt, level, ring.param->GetScale(level), msg);
  Ciphertext<word> ct, out;
  ring.ui->Encrypt(ct, pt);
  ring.context->HRot(out, ct, rot, loaded_map);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  Plaintext<word> out_pt;
  ring.ui->Decrypt(out_pt, out);
  std::vector<Complex> got;
  ring.context->encoder_.Decode(got, out_pt);

  // HRot by `rot` reads slot j from slot j + rot; the host does the same.
  std::vector<Complex> want(num_slots);
  for (int j = 0; j < num_slots; j++) want[j] = msg[(j + rot) % num_slots];

  const double err = MaxAbsDiff(got, want, num_slots);
  std::cout << "  rotation through the LOADED key only: " << err
            << " against |msg| <= " << MaxAbs(msg, num_slots) << ", "
            << MiB(ArchiveReader::FileSize(path)) << " MiB for "
            << loaded_map.size() << " keys" << std::endl;
  // A key switch adds its own noise, which is what this tolerance is; a
  // mis-indexed or truncated key would decrypt to noise of order |msg|.
  EXPECT_LT(err, 1e-4);
  Remove(path);
}

// ---------------------------------------------------------------------------
// A compiled transform, against the host matrix-vector product -- and beside
// the freshly built one, so "as good as built" is measured.
// ---------------------------------------------------------------------------
TEST(Serialization, ALoadedTransformComputesTheSameProductAsABuiltOne) {
  Ring ring(kParam);
  const auto id = cheddar::IdentityOf(*ring.param);
  const int num_slots = ring.param->MaxNumSlots();
  const int level = ring.enc_level;
  const int height = num_slots;

  // Two diagonals with distinct constants, so a swapped or dropped diagonal
  // cannot cancel out of the answer.
  cheddar::StripedMatrix m(height, height);
  m.try_emplace(0, height, Complex(0.0, 0.0));
  m.try_emplace(8, height, Complex(0.0, 0.0));
  for (int j = 0; j < height; j++) {
    m[0][j] = Complex(0.25 + 0.5 * (j % 3), 0.0);
    m[8][j] = Complex(-0.75 + 0.25 * (j % 5), 0.0);
  }

  const double pt_scale = ring.param->GetScale(level - 1) *
                          ring.param->GetRescalePrimeProd(level) /
                          ring.param->GetScale(level);
  LinearTransform<word> built(ring.context, m, level, pt_scale, 2, 2);
  {
    EvkRequest req;
    built.AddRequiredRotations(req);
    ring.ui->PrepareRotationKey(req);
  }

  const std::string path = Scratch("ser_lt.bin");
  {
    ArchiveWriter ar(path, id);
    built.Save(ar);
    ar.Close();
  }
  LinearTransform<word> loaded = [&] {
    ArchiveReader ar(path, id);
    return LinearTransform<word>::Load(ar);
  }();

  EXPECT_EQ(loaded.GetBS(), built.GetBS());
  EXPECT_EQ(loaded.GetGS(), built.GetGS());
  EXPECT_EQ(loaded.GetDiagonalOffsets(), built.GetDiagonalOffsets());

  const auto msg = RandomMessage(num_slots, false, 20260831u);
  Plaintext<word> pt;
  ring.context->encoder_.Encode(pt, level, ring.param->GetScale(level), msg);
  Ciphertext<word> ct;
  ring.ui->Encrypt(ct, pt);

  // The host reference. `TheStripedMatrixOffsetConventionIsPinned` fixed the
  // reading: key `i` means `out[j] = m[i][j] * in[j + i]`.
  std::vector<Complex> want(num_slots, Complex(0.0, 0.0));
  for (const auto &[offset, diag] : m) {
    for (int j = 0; j < num_slots; j++) {
      want[j] += diag[j] * msg[(j + offset) % num_slots];
    }
  }

  auto evaluate = [&](const LinearTransform<word> &t) {
    Ciphertext<word> out;
    t.Evaluate(ring.context, out, ct, ring.ui->GetEvkMap());
    cudaDeviceSynchronize();
    Plaintext<word> out_pt;
    ring.ui->Decrypt(out_pt, out);
    std::vector<Complex> got;
    ring.context->encoder_.Decode(got, out_pt);
    return got;
  };

  const auto got_built = evaluate(built);
  const auto got_loaded = evaluate(loaded);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  const double err_built = MaxAbsDiff(got_built, want, num_slots);
  const double err_loaded = MaxAbsDiff(got_loaded, want, num_slots);
  std::cout << "  transform against the host product: built " << err_built
            << ", loaded " << err_loaded << " (|want| <= "
            << MaxAbs(want, num_slots) << "), "
            << MiB(ArchiveReader::FileSize(path)) << " MiB" << std::endl;

  ASSERT_GT(MaxAbs(want, num_slots), 1e-3)
      << "the reference product is zero, so neither comparison can fail";
  EXPECT_LT(err_built, 1e-4);
  EXPECT_LT(err_loaded, 1e-4);
  // Same plaintexts, so this should be bit-identical, not merely close.
  EXPECT_LT(MaxAbsDiff(got_loaded, got_built, num_slots), 1e-12)
      << "the loaded transform is not the built one";
  Remove(path);
}

// ---------------------------------------------------------------------------
// The refusal. Two different parameter sets must not produce the same
// identity, and `PeekIdentity` must read back what was written.
// ---------------------------------------------------------------------------
TEST(Serialization, TheIdentitySeparatesParameterSets) {
  Ring ordinary(kParam);
  Ring ci(kCiParam);

  const auto id_ordinary = cheddar::IdentityOf(*ordinary.param);
  const auto id_ci = cheddar::IdentityOf(*ci.param);

  std::cout << "  " << kParam << ": " << id_ordinary.Describe() << std::endl;
  std::cout << "  " << kCiParam << ": " << id_ci.Describe() << std::endl;
  EXPECT_NE(id_ordinary, id_ci);

  const std::string path = Scratch("ser_id.bin");
  {
    ArchiveWriter ar(path, id_ordinary);
    ar.Tag("nothing");
    ar.Close();
  }

  // What the file says, read without committing to it -- so that the refusal
  // in the constructor, which exits, can be reasoned about here instead.
  const auto peeked = ArchiveReader::PeekIdentity(path);
  EXPECT_EQ(peeked, id_ordinary);
  EXPECT_NE(peeked, id_ci)
      << "an archive of one parameter set would be accepted by another";

  // A file that is not an archive peeks as a zeroed identity rather than as
  // anything a live parameter set could match.
  const std::string junk = Scratch("ser_junk.bin");
  {
    std::ofstream os(junk, std::ios::binary);
    os << "not a cheddar archive at all, not even the right length";
  }
  const auto junk_id = ArchiveReader::PeekIdentity(junk);
  EXPECT_EQ(junk_id, ArchiveIdentity{});
  EXPECT_NE(junk_id, id_ordinary);
  EXPECT_EQ(ArchiveReader::PeekIdentity(Scratch("no_such_file.bin")),
            ArchiveIdentity{});

  Remove(path);
  Remove(junk);
}
