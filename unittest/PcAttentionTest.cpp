// The PUBLIC-CONTEXT attention: the first consumer of the plain slot map.
//
// Everything on this branch so far has been a part. `PcPremapTest` showed
// that the chain addressing folds into both converters for free, that the
// subring lane IS the plain map's instance, that a subring element stores in
// `k` words and encodes in one device transform. `CiBatchAttention::Config::
// plain_map` wired the encrypted half onto the map. None of it computed a
// public-context attention.
//
// This does. `CiPcAttention` is [SYLPH] section 4's PC-attention on the
// batched layout: the encrypted query tokens against a public KV cache that
// differs per instance, which under the plain map is [KANG] Algorithm 1 twice
// with one exp between -- depth 1 each, no rotation, no automorphism key, no
// relinearization key, no encoding conversion. What has to be checked is not
// the algebra (`SubringMatrix.h` argues that, and PcPremapTest measures it)
// but that a whole head of it lands where a host loop over the same numbers
// says it should, and that streaming a long public context does not grow the
// residency.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "RingFixture.h"
#include "core/DeviceVector.h"
#include "core/MemoryPool.h"
#include "extension/CiPcAttention.h"

using word = uint32_t;
using Ring = ringfixture::Ring<word>;
using cheddar::CiBatchLayout;
using cheddar::CiPcAttention;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::Plaintext;
using cheddar::CopyDeviceToHost;
using cheddar::DeviceVector;
using cheddar::HostVector;
using cheddar::SubringWeights;

namespace {

// The shipped batched shape: `ci16_35`'s 65536 real slots as 128 encrypted
// query tokens of 512 instances.
constexpr int kTokens = 128;
constexpr int kInstances = 512;
constexpr int kHeadDim = 128;
constexpr int kQLevel = 16;

double Seconds(std::chrono::steady_clock::time_point a,
               std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

size_t FreeMiB() {
  size_t f = 0, t = 0;
  cudaMemGetInfo(&f, &t);
  return f >> 20;
}

// The host side of one head, in the same units the class works in.
struct PublicContext {
  int pub_tokens = 0;
  //! k[(b * pub_tokens + p) * kHeadDim + c], v likewise.
  std::vector<double> k, v;

  void Fill(int ptok, double bound, uint64_t seed) {
    pub_tokens = ptok;
    const size_t n = static_cast<size_t>(kInstances) * ptok * kHeadDim;
    k.assign(n, 0.0);
    v.assign(n, 0.0);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-bound, bound);
    for (size_t i = 0; i < n; i++) {
      k[i] = dist(gen);
      v[i] = dist(gen);
    }
  }

  //! The chunk source `CiPcAttention::Head` drives, in the entry-major,
  //! lane-major order `EncodeWeightsReal` reads.
  void Chunk(int start, int width, std::vector<double> &kb,
             std::vector<double> &vb) const {
    for (int b = 0; b < kInstances; b++) {
      for (int p = 0; p < width; p++) {
        const double *src =
            k.data() + (static_cast<size_t>(b) * pub_tokens + start + p) *
                           kHeadDim;
        const double *srv =
            v.data() + (static_cast<size_t>(b) * pub_tokens + start + p) *
                           kHeadDim;
        for (int c = 0; c < kHeadDim; c++) {
          kb[(static_cast<size_t>(c) * width + p) * kInstances + b] = src[c];
          vb[(static_cast<size_t>(p) * kHeadDim + c) * kInstances + b] = srv[c];
        }
      }
    }
  }
};

// The queries, on the host and encrypted: `q[c][slot]`, a slot message a
// channel, packed through the layout rather than through the formula.
struct Queries {
  std::vector<std::vector<double>> host;  // [channel][slot]
  std::vector<Ciphertext<word>> ct;

  void Build(const Ring &ring, const CiBatchLayout &layout, double bound,
             uint64_t seed) {
    const int degree = ring.Degree();
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-bound, bound);
    host.assign(kHeadDim, std::vector<double>(degree, 0.0));
    ct.resize(kHeadDim);
    std::vector<Complex> msg(degree);
    for (int c = 0; c < kHeadDim; c++) {
      for (int t = 0; t < kTokens; t++) {
        for (int b = 0; b < kInstances; b++) {
          host[c][layout.Slot(t, b)] = dist(gen);
        }
      }
      for (int s = 0; s < degree; s++) msg[s] = Complex(host[c][s], 0.0);
      Plaintext<word> pt;
      ring.context->gpu_encoder_.Encode(pt, kQLevel,
                                        ring.param->GetScale(kQLevel), msg);
      ring.ui->Encrypt(ct[c], pt);
    }
  }
};

// The reference scores of one chunk, with the affine's multiply already in
// them exactly as `EncodeKeys` folds it: `a1 * sum_c K[b][p][c] q[c][slot]`.
void HostScores(std::vector<std::vector<double>> &s, const Queries &q,
                const PublicContext &pc, const CiBatchLayout &layout,
                int start, int width, double a1) {
  const int degree = layout.num_slots;
  s.assign(width, std::vector<double>(degree, 0.0));
  for (int t = 0; t < kTokens; t++) {
    for (int b = 0; b < kInstances; b++) {
      const int slot = layout.Slot(t, b);
      for (int p = 0; p < width; p++) {
        const double *kk =
            pc.k.data() + (static_cast<size_t>(b) * pc.pub_tokens + start + p) *
                              kHeadDim;
        double acc = 0.0;
        for (int c = 0; c < kHeadDim; c++) acc += kk[c] * q.host[c][slot];
        s[p][slot] = a1 * acc;
      }
    }
  }
}

// max |got - want| and the magnitude of `want`, read at the plain map's slots.
std::pair<double, double> Compare(const std::vector<Complex> &got,
                                  const std::vector<double> &want,
                                  const CiBatchLayout &layout) {
  double worst = 0.0, mag = 0.0;
  for (int t = 0; t < kTokens; t++) {
    for (int b = 0; b < kInstances; b++) {
      const int s = layout.Slot(t, b);
      mag = std::max(mag, std::abs(want[s]));
      worst = std::max(worst, std::abs(got[s].real() - want[s]));
    }
  }
  return {worst, mag};
}

CiPcAttention<word>::Config MakeConfig(int chunk) {
  CiPcAttention<word>::Config cfg;
  cfg.num_tokens = kTokens;
  cfg.num_instances = kInstances;
  cfg.head_dim = kHeadDim;
  cfg.chunk = chunk;
  cfg.q_level = kQLevel;
  cfg.verbose = true;
  return cfg;
}

// A calibration that puts the affine's output squarely inside the exp's fit
// domain. `shift` is the score MAXIMUM and `span` the RANGE, exactly as
// `CiBatchAttention::SoftMaxCalibration` means them, so `span = 2`,
// `shift = 1` and `carried = 1` say "the scores live in [-1, 1]": a1 = 1 and
// a0 = 1 - 2 * 1 / 2 = 0, hence u = S and the data's own bound is what keeps
// |u| <= 1. The tests check that rather than assume it, because a polynomial
// evaluated outside its interval does not fail, it lies -- `shift = 0` here
// centred u on 1 instead of 0 and the guard is what said so.
CiPcAttention<word>::Calibration MakeCalibration(bool with_fold) {
  CiPcAttention<word>::Calibration cal;
  cal.m_eff = 8.0;
  cal.span = 2.0;
  cal.carried = 1.0;
  cal.shift = 1.0;
  if (with_fold) {
    cal.row_fold.resize(kTokens);
    for (int t = 0; t < kTokens; t++) {
      cal.row_fold[t] = 0.5 + 0.5 * (t + 1) / double(kTokens);
    }
  }
  return cal;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. THE SCORE PRODUCT. One chunk of the public context against the encrypted
//    queries: `res[p] = sum_c u[c][p] q[c]`, [KANG] Algorithm 1 with the
//    contraction over the CHANNELS and one output ciphertext per public key
//    token. The plaintext operand is per instance and constant over the query
//    tokens, which is the whole reason it is a subring element and the whole
//    reason the map had to be the plain one.
//
//    Also the two things the class does inside this call that a host loop
//    would not: the affine's multiply rides the encode (so the returned
//    scores are `a1 * S`, not `S`), and the product rescales, so the output
//    is one level below the queries at the canonical scale.
// ---------------------------------------------------------------------------
TEST(PcAttention, TheScoreProductIsTheHostsPublicContextScores) {
  Ring ring("ci16_35.json");
  ASSERT_EQ(ring.Degree(), kTokens * kInstances);
  constexpr int kWidth = 8;

  auto bctx = std::dynamic_pointer_cast<cheddar::BootContext<word>>(
      ring.context);
  ASSERT_NE(bctx, nullptr);
  auto pc_att = std::make_unique<CiPcAttention<word>>(bctx, MakeConfig(kWidth));
  const auto cal = MakeCalibration(false);
  pc_att->Prepare(cal);
  const CiBatchLayout &layout = pc_att->GetLayout();

  Queries q;
  q.Build(ring, layout, 0.2, 0xB0071E5ULL);
  PublicContext pc;
  pc.Fill(kWidth, 0.2, 0x5EEDC0DEULL);

  const double a1 = 2.0 / (cal.span * cal.carried);
  std::vector<std::vector<double>> want;
  HostScores(want, q, pc, layout, 0, kWidth, a1);

  std::vector<double> kb(static_cast<size_t>(kHeadDim) * kWidth * kInstances);
  std::vector<double> vb(static_cast<size_t>(kWidth) * kHeadDim * kInstances);
  pc.Chunk(0, kWidth, kb, vb);

  cheddar::SubringWeights<word> keys;
  cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  pc_att->EncodeKeys(keys, kb, kWidth);
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<Ciphertext<word>> scores;
  pc_att->Scores(scores, q.ct, keys);
  cudaDeviceSynchronize();
  const auto t2 = std::chrono::steady_clock::now();

  ASSERT_EQ(static_cast<int>(scores.size()), kWidth);
  EXPECT_EQ(ring.param->NPToLevel(scores[0].GetNP()), kQLevel - 1)
      << "Algorithm 1 rescales, so the scores are one level below the queries";
  const double canonical = ring.param->GetScale(kQLevel - 1);
  EXPECT_NEAR(scores[0].GetScale(), canonical, 1e-6 * canonical)
      << "the folded a1 must not leak into the declared scale";

  double worst = 0.0, mag = 0.0;
  for (int p = 0; p < kWidth; p++) {
    Plaintext<word> back;
    ring.ui->Decrypt(back, scores[p]);
    std::vector<Complex> got;
    ring.context->encoder_.Decode(got, back);
    const auto d = Compare(got, want[p], layout);
    worst = std::max(worst, d.first);
    mag = std::max(mag, d.second);
  }

  const double entries = static_cast<double>(kHeadDim) * kWidth;
  std::cout << std::fixed << std::setprecision(3)
            << "  chunk " << kWidth << " public tokens, contraction "
            << kHeadDim << " channels" << std::endl
            << "  key weights                      : " << entries
            << " subring entries, "
            << (keys.data_.size() * sizeof(word) / 1048576.0) << " MiB"
            << std::endl
            << "  encode                           : " << Seconds(t0, t1)
            << " s   (" << (1e6 * Seconds(t0, t1) / entries) << " us an entry)"
            << std::endl
            << "  Algorithm 1                      : " << Seconds(t1, t2)
            << " s" << std::endl
            << std::scientific << std::setprecision(3)
            << "  |scores - host|                  : " << worst << " of "
            << mag << "   (relative 2^" << std::fixed << std::setprecision(2)
            << std::log2(worst / mag) << ")" << std::endl;

  EXPECT_GT(mag, 1e-3) << "the reference is ~zero: nothing was contracted";
  EXPECT_LT(worst, 1e-4 * mag);
}

// ---------------------------------------------------------------------------
// 2. A WHOLE HEAD, streamed. The public context arrives in chunks and what
//    survives one is `sq` (the Euclidean-norm accumulator) and `acc` (the
//    value accumulator), so the answer must be the host's sum over ALL public
//    tokens even though nothing ever held them.
//
//    The quantity accumulated is `w = y^2`, which is what
//    `CiBatchAttention::SoftMax` reduces (Cho, k = 1: `r = invsqrt(sum y^2)`,
//    `P = (y r)^2`) -- so `sq` and `acc` are exactly the two things the
//    encrypted half hands the join, unnormalised, in the same map, and the
//    join is two adds. That is the claim this measures the numbers of.
// ---------------------------------------------------------------------------
TEST(PcAttention, TheStreamedHeadMatchesTheHost) {
  Ring ring("ci16_35.json");
  constexpr int kPubTokens = 32;
  constexpr int kChunk = 8;

  auto bctx = std::dynamic_pointer_cast<cheddar::BootContext<word>>(
      ring.context);
  ASSERT_NE(bctx, nullptr);
  auto pc_att = std::make_unique<CiPcAttention<word>>(bctx, MakeConfig(kChunk));
  const auto cal = MakeCalibration(true);
  pc_att->Prepare(cal);
  const CiBatchLayout &layout = pc_att->GetLayout();

  Queries q;
  q.Build(ring, layout, 0.2, 0x9E3779B9ULL);
  PublicContext pc;
  pc.Fill(kPubTokens, 0.2, 0xA5A5C0FFEEULL);

  // --- the host reference, and the guard on the fit domain ---------------
  const double a1 = 2.0 / (cal.span * cal.carried);
  const double hb = cal.m_eff / 2.0;
  const int degree = ring.Degree();
  std::vector<double> sq_want(degree, 0.0);
  std::vector<std::vector<double>> acc_want(kHeadDim,
                                            std::vector<double>(degree, 0.0));
  double worst_u = 0.0;
  {
    std::vector<std::vector<double>> s;
    HostScores(s, q, pc, layout, 0, kPubTokens, a1);
    const double a0 = 1.0 - 2.0 * cal.shift / cal.span;
    for (int t = 0; t < kTokens; t++) {
      for (int b = 0; b < kInstances; b++) {
        const int slot = layout.Slot(t, b);
        for (int p = 0; p < kPubTokens; p++) {
          const double u = s[p][slot] + a0;
          worst_u = std::max(worst_u, std::abs(u));
          const double w = std::exp(hb * (u - 1.0)) * cal.row_fold[t];
          sq_want[slot] += w;
          const double *vv =
              pc.v.data() +
              (static_cast<size_t>(b) * kPubTokens + p) * kHeadDim;
          for (int c = 0; c < kHeadDim; c++) acc_want[c][slot] += w * vv[c];
        }
      }
    }
  }
  ASSERT_LT(worst_u, 1.0)
      << "the affine leaves the exp's fit domain, so the reference and the "
         "evaluation are approximating different functions";

  // --- the encrypted head -------------------------------------------------
  std::vector<Ciphertext<word>> acc;
  Ciphertext<word> sq;
  const size_t free_before = FreeMiB();
  size_t min_free = free_before;
  int chunks = 0;
  const auto src = [&](int start, int width, std::vector<double> &kb,
                       std::vector<double> &vb) {
    pc.Chunk(start, width, kb, vb);
    min_free = std::min(min_free, FreeMiB());
    chunks++;
  };

  cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  pc_att->Head(acc, sq, q.ct, kPubTokens, src, ring.ui->GetEvkMap());
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  min_free = std::min(min_free, FreeMiB());

  EXPECT_EQ(chunks, kPubTokens / kChunk);
  ASSERT_EQ(static_cast<int>(acc.size()), kHeadDim);
  const int out_level = pc_att->GetOutputLevel();
  EXPECT_EQ(ring.param->NPToLevel(sq.GetNP()), out_level);
  EXPECT_EQ(ring.param->NPToLevel(acc[0].GetNP()), out_level)
      << "the accumulator and the norm must land together, or the join has "
         "to level them";

  Plaintext<word> back;
  std::vector<Complex> got;
  ring.ui->Decrypt(back, sq);
  ring.context->encoder_.Decode(got, back);
  const auto d_sq = Compare(got, sq_want, layout);

  double worst_acc = 0.0, mag_acc = 0.0;
  for (int c = 0; c < kHeadDim; c++) {
    ring.ui->Decrypt(back, acc[c]);
    ring.context->encoder_.Decode(got, back);
    const auto d = Compare(got, acc_want[c], layout);
    worst_acc = std::max(worst_acc, d.first);
    mag_acc = std::max(mag_acc, d.second);
  }

  std::cout << std::fixed << std::setprecision(2)
            << "  public tokens " << kPubTokens << " in " << chunks
            << " chunks of " << kChunk << ", head_dim " << kHeadDim
            << std::endl
            << "  plaintext multiplies             : "
            << (2.0 * kPubTokens * kHeadDim) << " (both products)"
            << std::endl
            << "  head                             : " << Seconds(t0, t1)
            << " s" << std::endl
            // The driver's figure, which is the POOL's reservation and not
            // live demand -- it is here as a ceiling check, and the
            // residency claim is measured properly in the next test.
            << "  driver free, before / lowest     : " << free_before
            << " / " << min_free << " MiB" << std::endl
            << "  output level                     : " << out_level
            << std::endl
            << std::scientific << std::setprecision(3)
            << "  |sq  - host|                     : " << d_sq.first << " of "
            << d_sq.second << "   (relative 2^" << std::fixed
            << std::setprecision(2) << std::log2(d_sq.first / d_sq.second)
            << ")" << std::endl
            << std::scientific << std::setprecision(3)
            << "  |acc - host|                     : " << worst_acc << " of "
            << mag_acc << "   (relative 2^" << std::fixed
            << std::setprecision(2) << std::log2(worst_acc / mag_acc) << ")"
            << std::endl;

  EXPECT_GT(d_sq.second, 1e-3) << "the reference norm is ~zero";
  EXPECT_GT(mag_acc, 1e-4) << "the reference accumulator is ~zero";
  EXPECT_LT(d_sq.first, 1e-3 * d_sq.second);
  EXPECT_LT(worst_acc, 1e-3 * mag_acc);
}

// ---------------------------------------------------------------------------
// 3. THE RESIDENCY CLAIM, which is the reason the interface is a stream and
//    not a matrix. Sylph's public context is 3968 tokens and its KV is PER
//    INSTANCE: one head's K alone is 512 x 3968 x 128 reals. The products are
//    sums over the public token, so the context can arrive in chunks and what
//    outlives a chunk is `head_dim + 1` ciphertexts whatever the length.
//
//    Measured, not asserted from the source, and NOT with `cudaMemGetInfo`:
//    the pool is a binning pool over `cuda_async_memory_resource`, so the
//    driver's free figure reports what the pool has RESERVED and does not
//    move while the working set fits inside it -- it read 40339 MiB before
//    and after a whole head, which would have "passed" a residency test that
//    held every chunk. `MemoryPool`'s statistics adaptor reports live demand
//    instead, and that is what a stream has to keep flat.
//
//    The structural half needs no ledger at all: the ciphertexts that OUTLIVE
//    the loop are `head_dim + 1`, and their bytes are countable. That number
//    must be identical at both lengths whatever the allocator is doing.
// ---------------------------------------------------------------------------
TEST(PcAttention, TheResidencyDoesNotMoveWithTheContextLength) {
  // Before the first Context, or it is ignored -- so this test measures the
  // ledger only when it runs first (or under CHEDDAR_MEM_STATS=1). The
  // structural count below does not depend on it.
  const bool ledger = cheddar::MemoryPool::SetStatisticsEnabled(true) ||
                      cheddar::MemoryPool::StatisticsEnabled();

  Ring ring("ci16_35.json");
  constexpr int kChunk = 8;

  auto bctx = std::dynamic_pointer_cast<cheddar::BootContext<word>>(
      ring.context);
  ASSERT_NE(bctx, nullptr);
  auto pc_att = std::make_unique<CiPcAttention<word>>(bctx, MakeConfig(kChunk));
  pc_att->Prepare(MakeCalibration(false));
  const CiBatchLayout &layout = pc_att->GetLayout();

  Queries q;
  q.Build(ring, layout, 0.2, 0x1234ABCDULL);

  struct Run {
    size_t survivors = 0;   //!< bytes of `acc` + `sq` after the loop
    int64_t live_peak = 0;  //!< the ledger's high water inside the loop
    double seconds = 0.0;
  };

  const auto run = [&](int ptok) {
    PublicContext pc;
    pc.Fill(ptok, 0.2, 0xFEEDFACEULL);
    std::vector<Ciphertext<word>> acc;
    Ciphertext<word> sq;
    Run r;
    const auto src = [&](int start, int width, std::vector<double> &kb,
                         std::vector<double> &vb) {
      pc.Chunk(start, width, kb, vb);
      if (ledger) {
        r.live_peak = std::max(r.live_peak,
                               cheddar::MemoryPool::GetUsage().current_bytes);
      }
    };
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    pc_att->Head(acc, sq, q.ct, ptok, src, ring.ui->GetEvkMap());
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    r.seconds = Seconds(t0, t1);
    if (ledger) {
      r.live_peak = std::max(r.live_peak,
                             cheddar::MemoryPool::GetUsage().current_bytes);
    }
    r.survivors = (sq.bx_.size() + sq.ax_.size()) * sizeof(word);
    for (const auto &c : acc) {
      r.survivors += (c.bx_.size() + c.ax_.size()) * sizeof(word);
    }
    return r;
  };

  const Run shrt = run(kChunk * 2);
  const Run lng = run(kChunk * 8);

  const double per_token_s = (lng.seconds - shrt.seconds) / (kChunk * 6);
  const double drift_mib =
      (lng.live_peak - shrt.live_peak) / 1048576.0;
  std::cout << std::fixed << std::setprecision(2)
            << "  survivors after the loop         : "
            << (shrt.survivors / 1048576.0) << " MiB at "
            << (kChunk * 2) << " public tokens, "
            << (lng.survivors / 1048576.0) << " MiB at " << (kChunk * 8)
            << "   (head_dim + 1 ciphertexts)" << std::endl;
  if (ledger) {
    std::cout << "  live demand, high water          : "
              << (shrt.live_peak / 1048576.0) << " -> "
              << (lng.live_peak / 1048576.0) << " MiB   (drift "
              << drift_mib << ")" << std::endl;
  } else {
    std::cout << "  live demand                      : not measured -- the "
                 "ledger has to be on before the first Context "
                 "(CHEDDAR_MEM_STATS=1)"
              << std::endl;
  }
  std::cout << "  " << (kChunk * 2) << " -> " << (kChunk * 8)
            << " public tokens          : " << shrt.seconds << " -> "
            << lng.seconds << " s" << std::endl
            << "  marginal cost a public token     : "
            << (1000.0 * per_token_s) << " ms   (one head, " << kHeadDim
            << " channels)" << std::endl
            << "  Sylph's 3968 would be            : "
            << (3968.0 * per_token_s) << " s a head" << std::endl;

  EXPECT_EQ(shrt.survivors, lng.survivors)
      << "what outlives the loop moved with the context length";
  EXPECT_GT(shrt.survivors, 0u);
  if (ledger) {
    // A non-streaming loop would hold every chunk's weight stores: ~70 MiB a
    // chunk at this shape, so six extra chunks is ~420 MiB. 256 catches that
    // and leaves room for the allocator's own slack.
    EXPECT_LT(drift_mib, 256.0)
        << "live demand grew with the context: the stream is holding on to "
           "something it should have dropped";
  }
  EXPECT_GT(lng.seconds, shrt.seconds)
      << "the longer context did no more work, so nothing was measured";
}

// ---------------------------------------------------------------------------
// 4. THE GQA SHARE IS THE SEPARATE HEADS. `HeadGroup` encodes one chunk of
//    the public keys and values ONCE and drives the group's query heads
//    against it, where `Head` -- one query head a call -- encodes the same
//    numbers again for every one of them. The encode is a plaintext
//    operation on numbers that do not depend on the query, so sharing it
//    cannot change an output word, and this says it does not: the library's
//    standing rule for every batched path (`CHEDDAR_CMT_SERIAL` and the
//    rest) is that the shared form stays word for word equal to the loop.
//
//    `Head` is `HeadGroup` with a group of one, so what is really under test
//    is that a group of three is three groups of one.
// ---------------------------------------------------------------------------
TEST(PcAttention, TheGroupShareIsTheSeparateHeadsWordForWord) {
  Ring ring("ci16_35.json");
  constexpr int kChunk = 8;
  constexpr int kPtok = 24;   // three chunks
  constexpr int kGroup = 3;

  auto bctx = std::dynamic_pointer_cast<cheddar::BootContext<word>>(
      ring.context);
  ASSERT_NE(bctx, nullptr);
  auto att = std::make_unique<CiPcAttention<word>>(bctx, MakeConfig(kChunk));
  att->Prepare(MakeCalibration(true));

  // Different queries a head: a group that shared its queries would pass
  // even if `HeadGroup` read the wrong one.
  std::vector<Queries> q(kGroup);
  for (int h = 0; h < kGroup; h++) {
    q[h].Build(ring, att->GetLayout(), 0.2, 0x51A7C0DEULL + h);
  }
  PublicContext pc;
  pc.Fill(kPtok, 0.2, 0xC0FFEE11ULL);
  const auto src = [&](int start, int width, std::vector<double> &kb,
                       std::vector<double> &vb) {
    pc.Chunk(start, width, kb, vb);
  };

  std::vector<std::vector<Ciphertext<word>>> acc_s(kGroup), acc_g(kGroup);
  std::vector<Ciphertext<word>> sq_s(kGroup), sq_g(kGroup);
  for (int h = 0; h < kGroup; h++) {
    att->Head(acc_s[h], sq_s[h], q[h].ct, kPtok, src, ring.ui->GetEvkMap());
  }

  std::vector<std::vector<Ciphertext<word>> *> ap(kGroup);
  std::vector<Ciphertext<word> *> sp(kGroup);
  std::vector<const std::vector<Ciphertext<word>> *> qp(kGroup);
  for (int h = 0; h < kGroup; h++) {
    ap[h] = &acc_g[h];
    sp[h] = &sq_g[h];
    qp[h] = &q[h].ct;
  }
  att->HeadGroup(ap, sp, qp, kPtok, src, ring.ui->GetEvkMap());

  size_t differ = 0, total = 0;
  const auto compare = [&](const Ciphertext<word> &got,
                           const Ciphertext<word> &want) {
    const DeviceVector<word> *g[2] = {&got.bx_, &got.ax_};
    const DeviceVector<word> *w[2] = {&want.bx_, &want.ax_};
    for (int p = 0; p < 2; p++) {
      HostVector<word> a, b;
      CopyDeviceToHost(a, *g[p]);
      CopyDeviceToHost(b, *w[p]);
      ASSERT_EQ(a.size(), b.size());
      for (size_t i = 0; i < a.size(); i++) differ += (a[i] != b[i]);
      total += a.size();
    }
    ASSERT_EQ(got.GetScale(), want.GetScale());
    ASSERT_EQ(got.GetNumSlots(), want.GetNumSlots());
  };

  for (int h = 0; h < kGroup; h++) {
    ASSERT_EQ(acc_g[h].size(), acc_s[h].size());
    ASSERT_FALSE(acc_g[h].empty());
    compare(sq_g[h], sq_s[h]);
    for (size_t c = 0; c < acc_g[h].size(); c++) {
      compare(acc_g[h][c], acc_s[h][c]);
    }
  }
  std::cout << "  group of " << kGroup << " vs " << kGroup
            << " separate heads   : " << differ << " of " << total
            << " words differ" << std::endl;
  ASSERT_EQ(differ, 0u);
  ASSERT_GT(total, 0u);
}

// ---------------------------------------------------------------------------
// 5. WHAT THE CONTEXT COSTS, and the lever the interface leaves on the table.
//
//    Test 3 prices a public token at one chunk size. This one prices it
//    across chunk sizes AND splits the price in two, because the halves do
//    not scale with the same head count:
//
//      ENCODE   `EncodeKeys` + `EncodeValues`. Kpub and Vpub are per KV
//               head, and Llama-3-8B is GQA 32/8 -- the same pair serves
//               FOUR query heads. `Head` encodes inside its own chunk loop
//               (CiPcAttention.cu), so four calls encode the same numbers
//               four times.
//      PRODUCT  `Scores` + `Weights` + `Accumulate`, per QUERY head, 32.
//
//    A layer is therefore `8 * encode + 32 * product` if a group shares its
//    encode and `32 * (encode + product)` if it does not. Which of those two
//    is the real figure decides whether the public context is affordable, so
//    it is measured rather than argued, and driven step by step rather than
//    through `Head` for exactly that reason.
//
//    A bench, not a check: skipped unless PC_SWEEP is set. PC_SWEEP_TOKENS
//    is the context it walks (default 256 -- long enough to average the
//    chunks, short enough to run in a minute), PC_SWEEP_CHUNKS the sizes,
//    PC_SWEEP_GROUP the query heads a KV head serves.
// ---------------------------------------------------------------------------
namespace {

int EnvInt(const char *name, int fallback) {
  const char *e = std::getenv(name);
  if (e == nullptr || e[0] == 0) return fallback;
  const int v = std::atoi(e);
  return v > 0 ? v : fallback;
}

std::vector<int> EnvChunks(const char *name, std::vector<int> fallback) {
  const char *e = std::getenv(name);
  if (e == nullptr || e[0] == 0) return fallback;
  std::vector<int> out;
  std::string s(e), tok;
  for (size_t i = 0; i <= s.size(); i++) {
    if (i == s.size() || s[i] == ',') {
      if (!tok.empty()) {
        const int v = std::atoi(tok.c_str());
        if (v > 0) out.push_back(v);
        tok.clear();
      }
    } else {
      tok.push_back(s[i]);
    }
  }
  return out.empty() ? fallback : out;
}

}  // namespace

TEST(PcAttention, TheContextPriceSplitsByHeadCount) {
  if (std::getenv("PC_SWEEP") == nullptr) {
    GTEST_SKIP() << "a bench, not a check -- set PC_SWEEP=1 to run it";
  }
  const bool ledger = cheddar::MemoryPool::SetStatisticsEnabled(true) ||
                      cheddar::MemoryPool::StatisticsEnabled();

  Ring ring("ci16_35.json");
  auto bctx = std::dynamic_pointer_cast<cheddar::BootContext<word>>(
      ring.context);
  ASSERT_NE(bctx, nullptr);

  const int ptok = EnvInt("PC_SWEEP_TOKENS", 256);
  const int group = EnvInt("PC_SWEEP_GROUP", 4);
  const std::vector<int> chunks =
      EnvChunks("PC_SWEEP_CHUNKS", {8, 16, 32, 64, 128});
  // Sylph's split of a 4096-token context, and Llama-3-8B's head counts.
  constexpr double kSylphPublic = 3968.0;
  constexpr int kQueryHeads = 32, kKvHeads = 8;

  // The layout does not depend on the chunk, so the queries are built once.
  const auto probe =
      std::make_unique<CiPcAttention<word>>(bctx, MakeConfig(chunks[0]));
  Queries q;
  q.Build(ring, probe->GetLayout(), 0.2, 0x1234ABCDULL);
  PublicContext pc;
  pc.Fill(ptok, 0.2, 0xFEEDFACEULL);

  std::cout << std::endl
            << "  " << ptok << " public tokens, " << group
            << " query heads a KV head, " << kHeadDim << " channels"
            << std::endl
            << "  chunk   encode/tok   product/tok/head   live      "
               "layer shared   layer per head"
            << std::endl;

  double best_shared = 0.0;
  for (int chunk : chunks) {
    auto att = std::make_unique<CiPcAttention<word>>(bctx, MakeConfig(chunk));
    att->Prepare(MakeCalibration(false));

    // What survives the whole context: one accumulator set a query head.
    std::vector<std::vector<Ciphertext<word>>> acc(group);
    std::vector<Ciphertext<word>> sq(group);
    std::vector<double> kbuf, vbuf;
    SubringWeights<word> kw, vw;
    double t_enc = 0.0, t_prod = 0.0;
    int64_t live = 0;
    const size_t lanes = static_cast<size_t>(kInstances);

    cudaDeviceSynchronize();
    for (int start = 0; start < ptok; start += chunk) {
      const int width = std::min(chunk, ptok - start);
      kbuf.assign(static_cast<size_t>(kHeadDim) * width * lanes, 0.0);
      vbuf.assign(static_cast<size_t>(width) * kHeadDim * lanes, 0.0);
      pc.Chunk(start, width, kbuf, vbuf);

      const auto t0 = std::chrono::steady_clock::now();
      att->EncodeKeys(kw, kbuf, width);
      att->EncodeValues(vw, vbuf, width);
      cudaDeviceSynchronize();
      const auto t1 = std::chrono::steady_clock::now();
      // The group's query heads read the SAME encoded chunk. That sharing is
      // the whole point of the split, so the bench does it here and `Head`
      // (one query head a call) cannot.
      for (int h = 0; h < group; h++) {
        std::vector<Ciphertext<word>> s;
        att->Scores(s, q.ct, kw);
        std::vector<Ciphertext<word>> w;
        att->Weights(w, s, ring.ui->GetEvkMap());
        att->Accumulate(acc[h], sq[h], w, vw);
      }
      cudaDeviceSynchronize();
      const auto t2 = std::chrono::steady_clock::now();
      t_enc += Seconds(t0, t1);
      t_prod += Seconds(t1, t2);
      if (ledger) {
        live = std::max(live, cheddar::MemoryPool::GetUsage().current_bytes);
      }
      kw = SubringWeights<word>();
      vw = SubringWeights<word>();
    }
    for (int h = 0; h < group; h++) att->Finish(acc[h], sq[h]);

    const double enc_ms = 1000.0 * t_enc / ptok;
    const double prod_ms = 1000.0 * t_prod / (static_cast<double>(ptok) * group);
    const double shared =
        kSylphPublic * (kKvHeads * enc_ms + kQueryHeads * prod_ms) / 1000.0;
    const double per_head =
        kSylphPublic * kQueryHeads * (enc_ms + prod_ms) / 1000.0;
    best_shared = (best_shared == 0.0) ? shared : std::min(best_shared, shared);

    std::cout << std::fixed << std::setprecision(3) << "  " << std::setw(5)
              << chunk << "   " << std::setw(10) << enc_ms << "   "
              << std::setw(16) << prod_ms << "   " << std::setw(7)
              << std::setprecision(0) << (live / 1048576.0) << " MiB"
              << std::setprecision(1) << std::setw(12) << shared << " s"
              << std::setw(14) << per_head << " s" << std::endl;

    EXPECT_GT(t_prod, 0.0) << "the product half was not measured";
    EXPECT_GT(t_enc, 0.0) << "the encode half was not measured";
  }
  std::cout << "  (layer = " << static_cast<int>(kSylphPublic)
            << " public tokens; shared = " << kKvHeads
            << " encodes + " << kQueryHeads
            << " products, per head = what `Head` does today)" << std::endl;
  EXPECT_GT(best_shared, 0.0);
}
