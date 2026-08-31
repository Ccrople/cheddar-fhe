// The projection leg fed the model's own tensor on the device, against the
// same leg fed the declared-width double matrix it always took.
//
// `CoeffLinearLeg::Project(const std::vector<double> &)` takes the weight at
// the DECLARED widths and converts it on the host: a gather per operand and a
// `BigInt` reduction per (value, prime). At the model's width that host loop
// is a layer's whole `pcmm: convert weights` row -- 86.88 s against 41.5 s of
// encrypted arithmetic (Doing.md 1.5ei). `Project(const DeviceWeights &)`
// reads the f32 tensor the exporter wrote, at its own size, through
// `GpuEncoder`'s gathered encode with the declared-to-live map composed into
// the kernel's two index vectors.
//
// THE BAR IS IDENTITY, NOT AGREEMENT. Both routes round the same product in
// the same order (`round((w * w_scale) * scale)`), widening f32 to double is
// exact, and the RNS reduction is exact on both sides, so the operands are
// the same limbs and the projections are the same WORDS. The test compares
// the output ciphertexts' `bx_`/`ax_` limb by limb, on both product paths:
// the cuBLAS int8 split (the layer's) and `PcmmAccum`'s Montgomery matrix.
// The host path is the reference: every earlier test validated it against
// a host product.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/BootContext.h"
#include "extension/LlamaLinear.h"

using word = uint32_t;
using cheddar::Ciphertext;
using cheddar::Plaintext;
using ringfixture::Ring;

namespace {

constexpr int kTokens = 128;
constexpr int kRank = 512;         // degree / (2 * tokens) on ci16_35
constexpr int kParents = 2;        // input ciphertexts
constexpr int kGroups = 2;         // output ciphertexts
constexpr int kInDeclared = kParents * kRank;
constexpr int kOutDeclared = kGroups * kRank;
constexpr int kProductLevel = 1;

// The leg is abstract on the attention side; this test projects only.
class ProjectOnlyLeg : public cheddar::CoeffLinearLeg<word> {
 public:
  using cheddar::CoeffLinearLeg<word>::CoeffLinearLeg;
  void Scores(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double,
              const std::vector<double> &) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: no Scores here");
  }
  void Values(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: no Values here");
  }
  void LocateScore(int, int, int, int &, int &) const override {
    cheddar::AssertTrue(false, "ProjectOnlyLeg: no score layout here");
  }
};

std::string Param() {
  const char *e = std::getenv("CHEDDAR_DEVICE_WEIGHTS_PARAM");
  return e ? e : "ci16_35.json";
}

// The layer's half-density maps (CiLlamaLayer): a model ciphertext carries
// `rank/2 - 1` live channels at the even declared indices 2..rank-2 --
// component zero has no partner (Doing.md 1.5cu) -- and a hidden one carries
// `rank/2` at 0, 2, ..., rank-2. `-1` marks a dead declared channel.
void ModelMap(std::vector<int> &slot, int declared, int &live) {
  slot.assign(declared, -1);
  live = 0;
  for (int d = 0; d < declared; d++) {
    if (d % 2 == 0 && d % kRank != 0) slot[d] = live++;
  }
}
void HiddenMap(std::vector<int> &slot, int declared, int &live) {
  slot.assign(declared, -1);
  live = 0;
  for (int d = 0; d < declared; d++) {
    if (d % 2 == 0) slot[d] = live++;
  }
}

struct Words {
  cheddar::HostVector<word> bx, ax;
};
Words Read(const Ciphertext<word> &ct) {
  Words w;
  cheddar::CopyDeviceToHost(w.bx, ct.bx_);
  cheddar::CopyDeviceToHost(w.ax, ct.ax_);
  return w;
}

void RunBothAndCompare(const char *label) {
  Ring boot(Param());
  const int rank = boot.Degree() / (2 * kTokens);
  ASSERT_EQ(rank, kRank);

  boot.ui->PrepareModPackKeys(kTokens, kProductLevel);
  std::vector<const cheddar::EvaluationKey<word> *> pack_keys(kRank);
  for (int j = 0; j < kRank; j++) {
    pack_keys[j] = &boot.ui->GetModPackKey(kRank, j);
  }
  typename cheddar::CoeffLinearLeg<word>::Config lcfg;
  lcfg.num_tokens = kTokens;
  lcfg.product_level = kProductLevel;
  lcfg.parents_per_tile = 0;
  lcfg.input_density = 2;
  lcfg.output_density = 2;
  ProjectOnlyLeg leg(boot.context, lcfg, pack_keys);
  ASSERT_EQ(leg.GetRank(), kRank);

  // ---- the tensor, at its own size, and the two maps -------------------
  std::vector<int> in_slot, out_slot;
  int in_live = 0, out_live = 0;
  ModelMap(in_slot, kInDeclared, in_live);
  HiddenMap(out_slot, kOutDeclared, out_live);
  ASSERT_EQ(in_live, kParents * (kRank / 2 - 1));
  ASSERT_EQ(out_live, kGroups * (kRank / 2));

  std::mt19937_64 gen(0xD3A1CE);
  std::normal_distribution<double> nd(0.0, 0.02);
  cheddar::HostVector<float> w32(static_cast<size_t>(in_live) * out_live);
  for (auto &v : w32) v = static_cast<float>(nd(gen));
  const double w_scale = 0.37;

  // The declared double matrix the host route reads, built from the SAME
  // tensor and maps -- what every caller before this test wrote by hand.
  std::vector<double> w_dec(static_cast<size_t>(kInDeclared) * kOutDeclared,
                            0.0);
  for (int di = 0; di < kInDeclared; di++) {
    if (in_slot[di] < 0) continue;
    for (int od = 0; od < kOutDeclared; od++) {
      if (out_slot[od] < 0) continue;
      w_dec[static_cast<size_t>(di) * kOutDeclared + od] =
          w32[static_cast<size_t>(in_slot[di]) * out_live + out_slot[od]];
    }
  }

  cheddar::DeviceVector<float> d_w;
  d_w.resize(static_cast<int>(w32.size()));
  cheddar::CopyHostToDevice(d_w, w32);
  typename cheddar::CoeffLinearLeg<word>::DeviceWeights dw;
  dw.data = &d_w;
  dw.in_live = in_live;
  dw.out_live = out_live;
  dw.in_slot = &in_slot;
  dw.out_slot = &out_slot;
  dw.fingerprint = cheddar::CoeffLinearLeg<word>::Fingerprint(
      w32.data(), w32.size(), w_scale);

  // ---- the parents: a coefficient payload at the product level ---------
  std::vector<Ciphertext<word>> x(kParents);
  std::uniform_real_distribution<double> ud(-0.5, 0.5);
  for (int p = 0; p < kParents; p++) {
    std::vector<double> coeffs(boot.Degree());
    for (auto &v : coeffs) v = ud(gen);
    Plaintext<word> pt;
    boot.context->encoder_.EncodeCoeff(pt, kProductLevel,
                                       boot.param->GetScale(kProductLevel),
                                       coeffs);
    boot.ui->Encrypt(x[p], pt);
  }

  // ---- both projections ---------------------------------------------------
  std::vector<Ciphertext<word>> from_host, from_device;
  leg.Project(from_host, x, kInDeclared, kOutDeclared, w_dec, w_scale,
              (std::string(label) + ".host").c_str());
  leg.Project(from_device, x, kInDeclared, kOutDeclared, dw, w_scale,
              (std::string(label) + ".device").c_str());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(from_host.size(), static_cast<size_t>(kGroups));
  ASSERT_EQ(from_device.size(), static_cast<size_t>(kGroups));

  size_t compared = 0, differ = 0;
  for (int g = 0; g < kGroups; g++) {
    const Words a = Read(from_host[g]);
    const Words b = Read(from_device[g]);
    ASSERT_EQ(a.bx.size(), b.bx.size());
    ASSERT_EQ(a.ax.size(), b.ax.size());
    for (size_t i = 0; i < a.bx.size(); i++) differ += (a.bx[i] != b.bx[i]);
    for (size_t i = 0; i < a.ax.size(); i++) differ += (a.ax[i] != b.ax[i]);
    compared += a.bx.size() + a.ax.size();
  }
  std::cout << label << ": " << differ << " of " << compared
            << " output words differ between the host-converted and the "
               "device-encoded projection (convert "
            << leg.GetConvertSeconds() << " s over both)" << std::endl;
  EXPECT_EQ(differ, 0u) << "the device-encoded operand is not the host's";
  ASSERT_GT(compared, 0u);

  // And a value sanity check on the shared answer: decrypting the two gives
  // the same coefficients, and they are not all zero.
  Plaintext<word> pt;
  boot.ui->Decrypt(pt, from_device[0]);
  std::vector<double> got;
  boot.context->encoder_.DecodeCoeff(got, pt);
  double mx = 0.0;
  for (double v : got) mx = std::max(mx, std::abs(v));
  std::cout << label << ": |output| <= " << mx << std::endl;
  EXPECT_GT(mx, 0.0);
}

}  // namespace

// The layer's own product path: the cuBLAS int8 split, when the build has it.
TEST(DeviceWeights, TheDeviceEncodedProjectionIsTheHostsWordForWord) {
  RunBothAndCompare("blas");
}

// `PcmmAccum`'s Montgomery matrix, which `GpuEncoder::EncodeMatrixGathered`
// writes directly. `CHEDDAR_PCMM_BLAS` is read when the leg is built.
TEST(DeviceWeights, TheSameOnThePlainMatrixProduct) {
  setenv("CHEDDAR_PCMM_BLAS", "0", 1);
  RunBothAndCompare("accum");
  unsetenv("CHEDDAR_PCMM_BLAS");
}
