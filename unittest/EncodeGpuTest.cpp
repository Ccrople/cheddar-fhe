// The encoding unit on the GPU, against the host encoder it replaces.
//
// Cheddar encodes on the host: `Encoder` says so in its own class comment, and
// the RNS decomposition inside every one of its three encodings is a GMP
// `mpz_mod` per (value, prime) pair driven by one thread. Two of those three
// are on this pipeline's critical path at the model's width -- a message
// ciphertext at T = 128 tokens, and the plaintext weight matrices, whose
// conversion is the layer's `pcmm: convert weights` row -- so this file is the
// same encodings on the device, checked limb by limb against the host and then
// measured.
//
// The measurement answers three questions per stage, which is what a kernel
// has to be judged on: how long it takes, how much of the machine's memory
// bandwidth it moves, and how much of its arithmetic it issues. The first is
// CUDA events; the second is exact byte counts over that time, against a
// device-to-device copy measured in the same process as the achievable
// reference; the third is the occupancy calculator's own answer for the launch
// configuration the stage actually uses, printed beside the register and
// shared-memory counts the build produced.

#undef ENABLE_EXTENSION

#include "Testbed.h"

#include <cstdlib>
#include <cstring>

#include "common/CommonUtils.h"
#include "core/EncodeGpu.h"
#include "core/Pcmm.h"

using word = uint32_t;

namespace {

// A CUDA-event stopwatch. Wall clock cannot see a kernel; `Testbed`'s own
// __Profile macros measure a synchronised region, which is the right tool for
// a whole operation and the wrong one for a stage of 20 us.
class EventTimer {
 public:
  EventTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~EventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void Start() { cudaEventRecord(start_, cudaStreamLegacy); }
  // Milliseconds for the region just closed.
  double Stop() {
    cudaEventRecord(stop_, cudaStreamLegacy);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

// Median of repeated timings. A minimum flatters a kernel that is occasionally
// starved and a mean is dragged by the first launch, and the run-to-run spread
// on this box is the thing being measured against.
double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <typename F>
double TimeGpu(int reps, F &&body) {
  EventTimer timer;
  body();  // warm up: first launch carries module load and page faults
  cudaDeviceSynchronize();
  std::vector<double> samples;
  samples.reserve(reps);
  for (int r = 0; r < reps; r++) {
    timer.Start();
    body();
    samples.push_back(timer.Stop());
  }
  return Median(std::move(samples));
}

// The same, with a per-rep setup that is NOT inside the events. The special
// FFT is in place and its 1/S normalisation lives in the next stage, so
// running it a hundred times on its own output multiplies the message by S
// each pass -- and what that measures is the wide-value path of a later stage,
// not the transform. Everything is on one stream, so the setup has completed
// before the start event is reached.
template <typename S, typename F>
double TimeGpuWithSetup(int reps, S &&setup, F &&body) {
  EventTimer timer;
  setup();
  body();
  cudaDeviceSynchronize();
  std::vector<double> samples;
  samples.reserve(reps);
  for (int r = 0; r < reps; r++) {
    setup();
    timer.Start();
    body();
    samples.push_back(timer.Stop());
  }
  return Median(std::move(samples));
}

template <typename F>
double TimeHostMs(int reps, F &&body) {
  std::vector<double> samples;
  for (int r = 0; r < reps; r++) {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return Median(std::move(samples));
}

void Row(const std::string &name, double ms, double bytes, double ops,
         double reference_gbps) {
  const double gbps = bytes / (ms * 1e-3) / 1e9;
  std::cout << std::left << std::setw(30) << name << std::right << std::fixed
            << std::setprecision(3) << std::setw(11) << ms << " ms";
  if (bytes > 0.0) {
    std::cout << std::setw(10) << std::setprecision(1) << gbps << " GB/s";
    if (reference_gbps > 0.0) {
      std::cout << std::setw(8) << std::setprecision(1)
                << (100.0 * gbps / reference_gbps) << "% BW";
    } else {
      std::cout << std::setw(13) << " ";
    }
  } else {
    std::cout << std::setw(29) << " ";
  }
  if (ops > 0.0) {
    std::cout << std::setw(12) << std::setprecision(2) << (ops / (ms * 1e-3) / 1e9)
              << " G/s";
  }
  std::cout << std::endl;
}

// The shape of `CoeffLinearLeg::GatherWeights`, written out here rather than
// linked: that function lives in the extension, which this file compiles
// without, and what matters about it is the loop, not its two index maps --
//
//   values[r][c] = w[col_map[c]][row_map[r]] * w_scale
//
// -- which is separable, and which is the whole of why the gather can move
// into the encoding kernel. A negative map entry is a dead channel and
// contributes a zero, which is how the declared half-density layout is read
// off the live tensor without ever building it.
std::vector<double> HostGather(const std::vector<double> &w, int out_channels,
                               const std::vector<int> &row_map,
                               const std::vector<int> &col_map,
                               double w_scale) {
  const int rows = static_cast<int>(row_map.size());
  const int cols = static_cast<int>(col_map.size());
  std::vector<double> values(static_cast<size_t>(rows) * cols, 0.0);
  for (int r = 0; r < rows; r++) {
    if (row_map[r] < 0) continue;
    for (int c = 0; c < cols; c++) {
      if (col_map[c] < 0) continue;
      values[static_cast<size_t>(r) * cols + c] =
          w[static_cast<size_t>(col_map[c]) * out_channels + row_map[r]] *
          w_scale;
    }
  }
  return values;
}

// The maps the leg builds: an output channel is its group's base plus the bit
// reversal of the module component, and an input channel is its parent's base
// plus the same reversal. Reproduced so the benchmark reads a realistic --
// that is, scattered -- address pattern rather than a contiguous one.
std::vector<int> RowMap(int rows, int rank, int group) {
  const int log_rank = Log2Ceil(rank);
  std::vector<int> map(rows);
  for (int r = 0; r < rows; r++) {
    map[r] = group * rank + static_cast<int>(BitReverseInt(r, log_rank));
  }
  return map;
}

std::vector<int> ColMap(int cols, int rank, int live, int first_parent) {
  const int log_rank = Log2Ceil(rank);
  std::vector<int> map(cols);
  for (int c = 0; c < cols; c++) {
    map[c] = (first_parent + c / live) * rank +
             static_cast<int>(BitReverseInt(c % live, log_rank));
  }
  return map;
}

}  // namespace

class EncodeGpuTest : public Testbed<word> {
 protected:
  std::unique_ptr<GpuEncoder<word>> gpu_;

  void SetUp() override {
    Testbed<word>::SetUp();
    gpu_ = std::make_unique<GpuEncoder<word>>(*param_, context_->ntt_handler_);
  }

  // The encoder holds device buffers allocated out of the Context's pool, so
  // it has to go before the Context does. MemoryPool's class comment is about
  // exactly this ordering.
  void TearDown() override {
    gpu_.reset();
    Testbed<word>::TearDown();
  }

  // Every limb of a plaintext, on the host.
  std::vector<word> Limbs(const Plaintext<word> &ptxt) const {
    HostVector<word> host;
    CopyDeviceToHost(host, const_cast<Plaintext<word> &>(ptxt).mx_);
    cudaDeviceSynchronize();
    return std::vector<word>(host.begin(), host.end());
  }

  std::vector<word> Limbs(const PlainMatrix<word> &m) const {
    HostVector<word> host;
    CopyDeviceToHost(host, const_cast<PlainMatrix<word> &>(m).data_);
    cudaDeviceSynchronize();
    return std::vector<word>(host.begin(), host.end());
  }

  std::vector<double> RandomReals(int n, double range) const {
    std::vector<Complex> tmp;
    const_cast<EncodeGpuTest *>(this)->GenerateRandomMessage(tmp, n, -range,
                                                             range, false);
    std::vector<double> out(n);
    for (int i = 0; i < n; i++) out[i] = tmp[i].real();
    return out;
  }
};

// The coefficient encoding is the one both routes round the same way, so it is
// the one that can be asked for bit-identity rather than for agreement.
TEST_P(EncodeGpuTest, TheCoefficientEncodingIsBitIdenticalToTheHost) {
  const int degree = param_->degree_;
  for (int level : LevelsToSweep()) {
    const double scale = DetermineScale(level);
    const std::vector<double> coeffs = RandomReals(degree, 8.0);

    Plaintext<word> host_pt;
    Plaintext<word> gpu_pt;
    context_->encoder_.EncodeCoeff(host_pt, level, scale, coeffs);
    gpu_->EncodeCoeff(gpu_pt, level, scale, coeffs);
    cudaDeviceSynchronize();

    const auto a = Limbs(host_pt);
    const auto b = Limbs(gpu_pt);
    ASSERT_EQ(a.size(), b.size()) << "level " << level;
    int differing = 0;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) differing++;
    }
    EXPECT_EQ(differing, 0) << differing << " of " << a.size()
                            << " limbs differ at level " << level;
  }
}

// The slot encoding runs the special FFT first, so the two routes differ in
// floating-point rounding and in the half-ulp `Encoder::EncodeCoeff`'s comment
// already records (this rounds; ComplexVectorToPlaintext truncates). The
// comparison is therefore on the decoded values, and on the coefficients the
// transform produced -- which is what isolates the FFT from the RNS stage.
TEST_P(EncodeGpuTest, TheSlotEncodingMatchesTheHost) {
  const int num_slots = param_->MaxNumSlots();
  const bool ci = param_->conjugate_invariant_;

  // The coefficients the transform produced, at every level: `DecodeCoeff`
  // hoists its CRT weights out of the coefficient loop, so it is linear in the
  // prime count and can be afforded on a sweep.
  for (int level : LevelsToSweep()) {
    const double scale = DetermineScale(level);
    std::vector<Complex> message;
    GenerateRandomMessage(message, num_slots, -1.0, 1.0, !ci);

    Plaintext<word> host_pt;
    Plaintext<word> gpu_pt;
    context_->encoder_.Encode(host_pt, level, scale, message);
    gpu_->Encode(gpu_pt, level, scale, message);
    cudaDeviceSynchronize();

    std::vector<double> host_coeff;
    std::vector<double> gpu_coeff;
    context_->encoder_.DecodeCoeff(host_coeff, host_pt);
    context_->encoder_.DecodeCoeff(gpu_coeff, gpu_pt);
    double worst = 0.0;
    double magnitude = 0.0;
    for (size_t i = 0; i < host_coeff.size(); i++) {
      worst = std::max(worst, std::abs(host_coeff[i] - gpu_coeff[i]));
      magnitude = std::max(magnitude, std::abs(host_coeff[i]));
    }
    std::cout << "  level " << level << ": coeff |host - gpu| = "
              << std::scientific << worst << " against |coeff| <= " << magnitude
              << std::fixed << std::endl;
    // Same butterflies in the same order, so the two differ only in the last
    // places of a double -- far below the half-ulp of `scale` the RNS stage is
    // about to round to anyway.
    EXPECT_LT(worst, 1e-9 * std::max(magnitude, 1.0));
  }

  // And the slots themselves, once. `Decode` rebuilds its CRT weights inside
  // the slot loop, so it is quadratic in the prime count and belongs at the
  // bottom of the chain rather than on a sweep.
  {
    const int level = 0;
    const double scale = DetermineScale(level);
    std::vector<Complex> message;
    GenerateRandomMessage(message, num_slots, -1.0, 1.0, !ci);

    Plaintext<word> host_pt;
    Plaintext<word> gpu_pt;
    context_->encoder_.Encode(host_pt, level, scale, message);
    gpu_->Encode(gpu_pt, level, scale, message);
    cudaDeviceSynchronize();

    std::vector<Complex> host_msg;
    std::vector<Complex> gpu_msg;
    context_->encoder_.Decode(host_msg, host_pt);
    context_->encoder_.Decode(gpu_msg, gpu_pt);
    double worst_msg = 0.0;
    double gpu_round_trip = 0.0;
    double host_round_trip = 0.0;
    for (size_t i = 0; i < host_msg.size(); i++) {
      worst_msg = std::max(worst_msg, std::abs(host_msg[i] - gpu_msg[i]));
      // Against the message itself, so that "the two agree" cannot be two
      // wrong transforms agreeing -- and separately for each route, because
      // they do not round the same way.
      gpu_round_trip =
          std::max(gpu_round_trip, std::abs(message[i] - gpu_msg[i]));
      host_round_trip =
          std::max(host_round_trip, std::abs(message[i] - host_msg[i]));
    }
    std::cout << "  level 0: slots |host - gpu| = " << std::scientific
              << worst_msg << ", round trip: gpu " << gpu_round_trip
              << ", host " << host_round_trip << std::fixed << std::endl;

    // THE TWO ROUTES DO NOT ROUND THE SAME WAY, AND ONLY ONE OF THEM ROUNDS.
    // `Encoder::EncodeCoeff`'s own comment records it: the slot path reaches
    // the limbs through `BigInt(double)`, which TRUNCATES, while the
    // coefficient path calls `std::round` first. So off the conjugate-
    // invariant ring -- where `Encode` goes through `RealVectorToPlaintext`
    // and both routes round -- every coefficient here differs by up to one
    // unit of `scale`, a bias rather than a jitter, and the inverse transform
    // sums 2^15 of them into each slot. Demanding agreement below that would
    // be demanding that this reproduce a half-ulp the library documents as a
    // defect. What is asserted instead is the thing that matters: the GPU's
    // own round trip is no worse than the host's.
    const double quantum = 1.0 / DetermineScale(level);
    EXPECT_LT(worst_msg, quantum * num_slots);
    EXPECT_LT(gpu_round_trip, 1e-6);
    EXPECT_LE(gpu_round_trip, host_round_trip * 1.5 + quantum);
  }
}

// The real-message path, which sends half as many bytes because R+ has no
// imaginary axis for them to describe. It must be the SAME encoding, not a
// nearby one: the imaginary halves it does not transfer are written on the
// device as exact zeros, so the transform sees the identical input and every
// limb has to match what the complex path produces.
TEST_P(EncodeGpuTest, TheRealMessagePathIsTheSameEncoding) {
  const int num_slots = param_->MaxNumSlots();
  for (int level : LevelsToSweep()) {
    const double scale = DetermineScale(level);
    std::vector<Complex> message;
    GenerateRandomMessage(message, num_slots, -1.0, 1.0, false);
    std::vector<double> real_message(num_slots);
    for (int i = 0; i < num_slots; i++) real_message[i] = message[i].real();

    Plaintext<word> complex_pt;
    Plaintext<word> real_pt;
    gpu_->Encode(complex_pt, level, scale, message);
    gpu_->EncodeReal(real_pt, level, scale, real_message);
    cudaDeviceSynchronize();

    const auto a = Limbs(complex_pt);
    const auto b = Limbs(real_pt);
    ASSERT_EQ(a.size(), b.size()) << "level " << level;
    int differing = 0;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) differing++;
    }
    EXPECT_EQ(differing, 0) << differing << " of " << a.size()
                            << " limbs differ at level " << level;

    // And the zero-copy route, composed out of the public stages: the message
    // written straight into the pinned buffer, so no host copy happens at all.
    // It has to reach the same limbs, or the buffer is not the same buffer.
    const NPInfo np = param_->LevelToNP(level);
    Plaintext<word> staged_pt;
    double *buffer = gpu_->StagingBuffer(num_slots);
    for (int i = 0; i < num_slots; i++) buffer[i] = real_message[i];
    gpu_->StageFromBuffer(num_slots, true);
    gpu_->SpecialIFFT(gpu_->FftScratch(num_slots), num_slots);
    gpu_->FftToCoeff(gpu_->CoeffScratch(), gpu_->FftScratch(num_slots),
                     num_slots);
    staged_pt.ModifyNP(np);
    staged_pt.SetNumSlots(num_slots);
    staged_pt.SetScale(scale);
    gpu_->RnsDecompose(staged_pt.mx_.data(), gpu_->CoeffScratch(),
                       param_->degree_, np, scale, false);
    {
      auto view = staged_pt.View();
      context_->ntt_handler_.NTT(view, np, staged_pt.ConstView(), true);
    }
    cudaDeviceSynchronize();
    const auto c = Limbs(staged_pt);
    ASSERT_EQ(a.size(), c.size());
    int staged_differing = 0;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != c[i]) staged_differing++;
    }
    EXPECT_EQ(staged_differing, 0)
        << staged_differing << " of " << a.size()
        << " limbs differ on the zero-copy route at level " << level;
  }
}

// The plaintext matrix encoding, which is the layer's model-conversion row.
// Both routes round to nearest and convert to Montgomery form, so this one is
// bit-identical too -- and it has to be, because a limb that is off by one is
// a weight that is off by 2^-35 in a place no accuracy ledger would attribute.
TEST_P(EncodeGpuTest, TheMatrixEncodingIsBitIdenticalToTheHost) {
  PcmmHandler<word> pcmm(*param_);
  const int rows = 64;
  const int cols = 256;
  for (int level : LevelsToSweep()) {
    const double scale = DetermineScale(level);
    const std::vector<double> values = RandomReals(rows * cols, 1.0);

    PlainMatrix<word> host_u;
    PlainMatrix<word> gpu_u;
    pcmm.EncodeMatrix(host_u, level, scale, values, rows, cols);
    gpu_->EncodeMatrix(gpu_u, level, scale, values, rows, cols);
    cudaDeviceSynchronize();

    const auto a = Limbs(host_u);
    const auto b = Limbs(gpu_u);
    ASSERT_EQ(a.size(), b.size());
    int differing = 0;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) differing++;
    }
    EXPECT_EQ(differing, 0) << differing << " of " << a.size()
                            << " limbs differ at level " << level;
  }
}

// A value no message in this tree reaches, and the one place the fast path
// does not apply: |round(v * scale)| above 2^64 leaves the single Barrett
// reduction for the mantissa-and-doublings route. The host's BigInt is exact
// for any magnitude and this has to be too, so it is checked rather than
// asserted away.
TEST_P(EncodeGpuTest, TheWideValuePathIsExact) {
  const int level = 0;
  const double scale = DetermineScale(level);
  const int degree = param_->degree_;
  std::vector<double> coeffs(degree, 0.0);
  // scale is ~2^35 here, so 2^40 puts round(v * scale) at ~2^75.
  double v = std::ldexp(1.0, 40);
  for (int i = 0; i < 16; i++) {
    coeffs[i] = (i % 2 == 0) ? v : -v;
    v *= 1.5;
  }

  Plaintext<word> host_pt;
  Plaintext<word> gpu_pt;
  context_->encoder_.EncodeCoeff(host_pt, level, scale, coeffs);
  gpu_->EncodeCoeff(gpu_pt, level, scale, coeffs);
  cudaDeviceSynchronize();

  const auto a = Limbs(host_pt);
  const auto b = Limbs(gpu_pt);
  ASSERT_EQ(a.size(), b.size());
  int differing = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) differing++;
  }
  EXPECT_EQ(differing, 0) << differing << " of " << a.size() << " limbs differ";
}

// The gather fused into the encoding, against the host doing it in two steps.
// Bit-identical is the bar, not agreement: a limb off by one is a weight off
// by 2^-35 in a place no accuracy ledger would attribute.
TEST_P(EncodeGpuTest, TheGatheredEncodingIsBitIdenticalToTheHost) {
  const int rank = 64;
  const int live = rank / 2;
  const int rows = live;
  const int cols = 4 * live;
  const int in_channels = 4 * rank;
  const int out_channels = 3 * rank;
  const double w_scale = 0.375;

  std::vector<double> w = RandomReals(in_channels * out_channels, 1.0);
  // The model ships as f32, so the reference is the f32 values widened back --
  // otherwise this would be testing double-versus-float, which is a different
  // claim from the one being made.
  std::vector<float> w32(w.begin(), w.end());
  for (size_t i = 0; i < w.size(); i++) w[i] = static_cast<double>(w32[i]);
  // A dead entry in either map, which is what reading the real model through
  // the declared layout produces, and which the host reference has to agree
  // with by writing a zero.
  std::vector<int> row_map = RowMap(rows, rank, 1);
  std::vector<int> col_map = ColMap(cols, rank, live, 0);
  row_map[7] = -1;
  col_map[13] = -1;

  const std::vector<double> values =
      HostGather(w, out_channels, row_map, col_map, w_scale);

  DeviceVector<int> d_row(rows);
  DeviceVector<int> d_col(cols);
  CopyHostToDevice(d_row, HostVector<int>(row_map.begin(), row_map.end()));
  CopyHostToDevice(d_col, HostVector<int>(col_map.begin(), col_map.end()));
  DeviceVector<int32_t> d_w32(static_cast<int>(w32.size()));
  cudaMemcpyAsync(d_w32.data(), w32.data(), w32.size() * sizeof(float),
                  cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaDeviceSynchronize();

  PcmmHandler<word> pcmm(*param_);
  for (int level : LevelsToSweep()) {
    const double scale = DetermineScale(level);
    PlainMatrix<word> host_u;
    PlainMatrix<word> gpu_u;
    pcmm.EncodeMatrix(host_u, level, scale, values, rows, cols);
    gpu_->template EncodeMatrixGathered<float>(
        gpu_u, level, scale, reinterpret_cast<const float *>(d_w32.data()),
        out_channels, d_row.data(), d_col.data(), rows, cols, w_scale);
    cudaDeviceSynchronize();

    const auto a = Limbs(host_u);
    const auto b = Limbs(gpu_u);
    ASSERT_EQ(a.size(), b.size());
    int differing = 0;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) differing++;
    }
    // The host gathers in double from a double weight; the device gathers from
    // the f32 the model actually ships. Widening f32 to double is exact, and
    // `w` here came from f32, so the two are the same number -- which is the
    // reason the model can be left in its own precision on the device.
    EXPECT_EQ(differing, 0) << differing << " of " << a.size()
                            << " limbs differ at level " << level;
  }
}

// ---------------------------------------------------------------------------
// The real Llama-3-8B `gate` weight, at the real width, through both routes.
//
// Everything above runs on shapes; this runs on the tensor. `wgate.f32` is
// 4096 x 14336 f32 in this tree's [in, out] convention, and the layer turns it
// into 56 plaintext operands of 256 x 4352 -- one per output group, every
// model ciphertext as columns.
//
// What the pipeline does today is three host steps: marshal the real tensor
// into the DECLARED half-density layout (8704 x 28672 doubles, 1.86 GiB, four
// times the tensor and mostly zeros), gather a group out of it, encode. The
// two index maps compose, so the middle matrix is not information -- and the
// fused kernel reads the exporter's own f32 blob instead.
// ---------------------------------------------------------------------------
TEST_P(EncodeGpuTest, TheGatheredEncodingRunsOnTheRealLlamaWeights) {
  const char *dir_env = std::getenv("LLAMA3_REAL_DIR");
  const std::string dir =
      dir_env != nullptr && dir_env[0] != '\0' ? dir_env : "";
  if (dir.empty()) {
    GTEST_SKIP() << "LLAMA3_REAL_DIR is not set; this needs the real model";
  }
  std::ifstream blob(dir + "/wgate.f32", std::ios::binary);
  if (!blob.is_open()) {
    GTEST_SKIP() << "no wgate.f32 under " << dir;
  }

  // The real tensor, and the declared layout the leg indexes it through.
  constexpr int kModel = 4096;      // live model channels
  constexpr int kHidden = 14336;    // live hidden channels
  constexpr int kRank = 512;        // the ModPack group, and the module rank
  constexpr int kPerModel = 255;    // rank/2 - 1: component zero has no partner
  constexpr int kPerHidden = 256;   // rank/2
  const int model_cts = (kModel + kPerModel - 1) / kPerModel;    // 17
  const int hidden_groups = (kHidden + kPerHidden - 1) / kPerHidden;  // 56
  const int declared_hidden = hidden_groups * kRank;             // 28672

  std::vector<float> w32(static_cast<size_t>(kModel) * kHidden);
  blob.read(reinterpret_cast<char *>(w32.data()),
            static_cast<std::streamsize>(w32.size() * sizeof(float)));
  ASSERT_EQ(blob.gcount(),
            static_cast<std::streamsize>(w32.size() * sizeof(float)))
      << "wgate.f32 is not " << kModel << " x " << kHidden << " f32";
  blob.close();

  // The declared-to-live inverses. Model channel m sits at declared index
  // 2 * (m % 255 + 1) of ciphertext m / 255; hidden channel j at 2 * (j % 256)
  // of group j / 256. Everything else -- the odd declared indices, component
  // zero, and the tail past the live width -- is dead, and a dead map entry is
  // what the kernel writes a zero for.
  auto model_live = [&](int declared) {
    const int k = declared / kRank;
    const int e = declared % kRank;
    if ((e & 1) != 0 || e == 0) return -1;
    const int m = k * kPerModel + (e / 2 - 1);
    return m < kModel ? m : -1;
  };
  auto hidden_live = [&](int declared) {
    const int k = declared / kRank;
    const int e = declared % kRank;
    if ((e & 1) != 0) return -1;
    const int j = k * kPerHidden + e / 2;
    return j < kHidden ? j : -1;
  };

  const int group = 3;  // one of the 56 hidden groups
  const int rows = kRank / 2;              // 256 live output rows
  const int cols = model_cts * kRank / 2;  // 4352, every model ciphertext
  const double w_scale = 0.0423;           // the leg's own cq/ck

  // Composed straight to live indices: the declared matrix never exists.
  std::vector<int> row_map(rows);
  for (int r = 0; r < rows; r++) {
    row_map[r] = hidden_live(group * kRank +
                             static_cast<int>(BitReverseInt(r, 9)));
  }
  std::vector<int> col_map(cols);
  for (int c = 0; c < cols; c++) {
    col_map[c] = model_live((c / (kRank / 2)) * kRank +
                            static_cast<int>(BitReverseInt(c % (kRank / 2), 9)));
  }
  int dead_rows = 0, dead_cols = 0;
  for (int v : row_map) dead_rows += (v < 0);
  for (int v : col_map) dead_cols += (v < 0);
  std::cout << "  real gate: " << kModel << " x " << kHidden << " f32, group "
            << group << " -> " << rows << " x " << cols << " (" << dead_rows
            << " dead rows, " << dead_cols << " dead columns of " << cols << ")"
            << std::endl;
  // Only the last model ciphertext is short, and only in its tail.
  ASSERT_EQ(dead_rows, 0);
  ASSERT_EQ(dead_cols, model_cts * kPerHidden - kModel);

  const int level = 0;
  const double scale = DetermineScale(level);
  const NPInfo np = param_->LevelToNP(level);

  // ---- the route the pipeline runs today ---------------------------------
  const double marshal_ms = TimeHostMs(1, [&] {
    std::vector<double> declared(static_cast<size_t>(kRank) * declared_hidden,
                                 0.0);
    // One model ciphertext's worth of the declared matrix, so the timing is
    // per-parent rather than 1.86 GiB in one allocation; the full marshal is
    // this times `model_cts`.
    for (int m = 0; m < kPerModel; m++) {
      const size_t dc = 2 * (m + 1);
      for (int j = 0; j < kHidden; j++) {
        declared[dc * declared_hidden +
                 ((j / kPerHidden) * kRank + 2 * (j % kPerHidden))] =
            w32[static_cast<size_t>(m) * kHidden + j];
      }
    }
  });

  std::vector<double> values;
  const double gather_ms = TimeHostMs(1, [&] {
    values.assign(static_cast<size_t>(rows) * cols, 0.0);
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (row_map[r] < 0 || col_map[c] < 0) continue;
        values[static_cast<size_t>(r) * cols + c] =
            static_cast<double>(
                w32[static_cast<size_t>(col_map[c]) * kHidden + row_map[r]]) *
            w_scale;
      }
    }
  });

  PcmmHandler<word> pcmm(*param_);
  PlainMatrix<word> host_u;
  const double encode_ms = TimeHostMs(1, [&] {
    pcmm.EncodeMatrix(host_u, level, scale, values, rows, cols);
  });

  // ---- the fused route ----------------------------------------------------
  DeviceVector<int> d_row(rows);
  DeviceVector<int> d_col(cols);
  CopyHostToDevice(d_row, HostVector<int>(row_map.begin(), row_map.end()));
  CopyHostToDevice(d_col, HostVector<int>(col_map.begin(), col_map.end()));
  // f32 on the device, in the model's own precision. DeviceVector is not
  // instantiated for float, and the bytes are what matter here.
  DeviceVector<int32_t> d_w(static_cast<int>(w32.size()));
  const double upload_ms = TimeGpu(3, [&] {
    cudaMemcpyAsync(d_w.data(), w32.data(), w32.size() * sizeof(float),
                    cudaMemcpyHostToDevice, cudaStreamLegacy);
  });
  const float *dw = reinterpret_cast<const float *>(d_w.data());

  PlainMatrix<word> gpu_u;
  const double fused_ms = TimeGpu(20, [&] {
    gpu_->template EncodeMatrixGathered<float>(gpu_u, level, scale, dw, kHidden,
                                               d_row.data(), d_col.data(), rows,
                                               cols, w_scale);
  });
  cudaDeviceSynchronize();

  const auto a = Limbs(host_u);
  const auto b = Limbs(gpu_u);
  ASSERT_EQ(a.size(), b.size());
  size_t differing = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) differing++;
  }
  EXPECT_EQ(differing, 0u) << differing << " of " << a.size()
                           << " limbs differ on the real gate weight";

  const double entries = static_cast<double>(rows) * cols;
  const double limbs = entries * np.GetNumTotal();
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  host marshal (one of " << model_cts
            << " parents) " << marshal_ms << " ms, gather " << gather_ms
            << " ms, EncodeMatrix " << encode_ms << " ms" << std::endl;
  std::cout << "  gpu weight upload (" << (w32.size() * sizeof(float) >> 20)
            << " MiB f32, one-time) " << upload_ms << " ms, fused gather+encode "
            << fused_ms << " ms" << std::endl;
  const double host_group = marshal_ms * model_cts / hidden_groups + gather_ms +
                            encode_ms;
  std::cout << "  per group: host " << host_group << " ms (marshal amortised "
            << "over the " << hidden_groups << " groups), gpu " << fused_ms
            << " ms -- " << std::setprecision(1) << (host_group / fused_ms)
            << "x" << std::setprecision(3) << std::endl;
  std::cout << "  the whole gate tensor (" << hidden_groups << " groups): host "
            << std::setprecision(2) << (host_group * hidden_groups / 1000.0)
            << " s, gpu " << (upload_ms + fused_ms * hidden_groups) / 1000.0
            << " s including the upload" << std::endl;
  std::cout << "  fused throughput " << std::setprecision(1)
            << (limbs / (fused_ms * 1e-3) / 1e9) << " G limb/s" << std::endl;
}

// ---------------------------------------------------------------------------
// The measurement.
// ---------------------------------------------------------------------------
TEST_P(EncodeGpuTest, TheEncodingUnitIsMeasured) {
  const int degree = param_->degree_;
  const int num_slots = param_->MaxNumSlots();

  cudaDeviceProp prop;
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  // The clock fields left `cudaDeviceProp` in CUDA 13; the attributes did not.
  int memory_clock_khz = 0, core_clock_khz = 0;
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device);
  cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, device);
  const double peak_gbps =
      2.0 * memory_clock_khz * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

  std::cout << std::endl;
  std::cout << "=== " << GetParam() << " on " << prop.name << " ===" << std::endl;
  std::cout << "  ring: degree " << degree << ", " << num_slots
            << (param_->conjugate_invariant_ ? " real" : " complex")
            << " slots" << std::endl;
  std::cout << "  device: " << prop.multiProcessorCount << " SMs, "
            << prop.maxThreadsPerMultiProcessor << " threads/SM, "
            << core_clock_khz / 1000 << " MHz core, " << std::fixed
            << std::setprecision(1) << peak_gbps << " GB/s theoretical peak"
            << std::endl;
  GpuEncoder<word>::ReportKernelAttributes(
      std::cout, param_->LevelToNP(param_->max_level_ / 2).GetNumTotal(),
      (Log2Ceil(num_slots) + 1) / 2);

  // The achievable-bandwidth reference, measured in this process on this card:
  // a device-to-device copy of 256 MiB, which moves two bytes per byte copied.
  double copy_gbps = 0.0;
  {
    const size_t copy_bytes = size_t{256} << 20;
    DeviceVector<word> src(static_cast<int>(copy_bytes / sizeof(word)));
    DeviceVector<word> dst(static_cast<int>(copy_bytes / sizeof(word)));
    const double copy_ms = TimeGpu(20, [&] {
      cudaMemcpyAsync(dst.data(), src.data(), copy_bytes,
                      cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    });
    copy_gbps = 2.0 * copy_bytes / (copy_ms * 1e-3) / 1e9;
    std::cout << "  achievable D2D copy bandwidth: " << std::setprecision(1)
              << copy_gbps << " GB/s (" << (100.0 * copy_gbps / peak_gbps)
              << "% of peak)" << std::endl;
  }

  // The host link, both ways a caller can hand over a message.
  {
    const size_t bytes = size_t{16} << 20;
    HostVector<double> pageable(bytes / sizeof(double));
    double *pinned = nullptr;
    cudaMallocHost(reinterpret_cast<void **>(&pinned), bytes);
    DeviceVector<word> sink(static_cast<int>(bytes / sizeof(word)));
    const double page_ms = TimeGpu(10, [&] {
      cudaMemcpyAsync(sink.data(), pageable.data(), bytes,
                      cudaMemcpyHostToDevice, cudaStreamLegacy);
    });
    const double pin_ms = TimeGpu(10, [&] {
      cudaMemcpyAsync(sink.data(), pinned, bytes, cudaMemcpyHostToDevice,
                      cudaStreamLegacy);
    });
    std::cout << "  H2D 16 MiB: pageable " << std::setprecision(2) << page_ms
              << " ms (" << std::setprecision(1)
              << (bytes / (page_ms * 1e-3) / 1e9) << " GB/s), pinned "
              << std::setprecision(2) << pin_ms << " ms ("
              << std::setprecision(1) << (bytes / (pin_ms * 1e-3) / 1e9)
              << " GB/s)" << std::endl;
    cudaFreeHost(pinned);
  }

  // Two levels, because the prime count is what both the host cost and the
  // device's output traffic are linear in: the bottom of the chain, where the
  // projections encode their weights, and the middle, where a message crossing
  // the boundary is encoded.
  for (int level : {0, param_->max_level_ / 2}) {
    const double scale = DetermineScale(level);
    const NPInfo np = param_->LevelToNP(level);
    const int num_primes = np.GetNumTotal();

    std::cout << std::endl
              << "-- level " << level << ", " << num_primes << " limbs --"
              << std::endl;
    std::cout << std::left << std::setw(30) << "stage" << std::right
              << std::setw(14) << "time" << std::setw(15) << "bandwidth"
              << std::setw(8) << "of D2D" << std::setw(16) << "throughput"
              << std::endl;

    // ---- one ciphertext at T = 128 tokens: the whole ring ----
    {
      std::vector<Complex> message;
      GenerateRandomMessage(message, num_slots, -1.0, 1.0,
                            !param_->conjugate_invariant_);
      const std::vector<double> coeffs = RandomReals(degree, 8.0);

      Plaintext<word> pt;
      const double host_coeff_ms = TimeHostMs(1, [&] {
        context_->encoder_.EncodeCoeff(pt, level, scale, coeffs);
      });
      const double host_slot_ms = TimeHostMs(1, [&] {
        context_->encoder_.Encode(pt, level, scale, message);
      });

      double *fft = gpu_->FftScratch(num_slots);
      double *coeff_dev = gpu_->CoeffScratch();
      Plaintext<word> gpu_pt;
      gpu_->EncodeCoeff(gpu_pt, level, scale, coeffs);
      cudaDeviceSynchronize();

      // The RNS stage FIRST, while `coeff_dev` still holds the coefficients
      // EncodeCoeff put there: the two transform stages below overwrite it,
      // and a stage measured on the wrong input measures the wrong branch.
      const double rns_ms = TimeGpu(100, [&] {
        gpu_->RnsDecompose(gpu_pt.mx_.data(), coeff_dev, degree, np, scale,
                           false);
      });
      // The same stage with the Montgomery conversion switched on. It adds a
      // wide multiply and a reduction per limb and moves not one extra byte,
      // so the difference between these two rows is the whole of the evidence
      // for whether the stage is bound by memory or by arithmetic -- and it
      // needs no profiler to read.
      const double rns_mont_ms = TimeGpu(100, [&] {
        gpu_->RnsDecompose(gpu_pt.mx_.data(), coeff_dev, degree, np, scale,
                           true);
      });
      std::vector<double> real_message(num_slots);
      for (int i = 0; i < num_slots; i++) real_message[i] = message[i].real();
      const double stage_ms =
          TimeGpu(100, [&] { gpu_->StageMessage(message, num_slots); });

      // THE STAGE SPLIT, MEASURED RATHER THAN SUBTRACTED. The first reading of
      // this row assumed the transfer ran at the 16 MiB reference's 24 GB/s
      // and attributed the rest to the host copy. These two rows are the
      // transfer alone, at exactly the size the stage uses, and the host copy
      // alone -- so the attribution stops being an inference.
      const size_t stage_bytes =
          static_cast<size_t>(2) * num_slots * sizeof(double);
      double *pinned = nullptr;
      cudaMallocHost(reinterpret_cast<void **>(&pinned), stage_bytes);
      DeviceVector<word> stage_sink(
          static_cast<int>(stage_bytes / sizeof(word)));
      const double h2d_only_ms = TimeGpu(100, [&] {
        cudaMemcpyAsync(stage_sink.data(), pinned, stage_bytes,
                        cudaMemcpyHostToDevice, cudaStreamLegacy);
      });
      const double memcpy_only_ms = TimeHostMs(100, [&] {
        std::memcpy(pinned, message.data(), stage_bytes);
      });
      cudaFreeHost(pinned);

      const double stage_real_ms =
          TimeGpu(100, [&] { gpu_->StageRealMessage(real_message, num_slots); });
      // The floor: the message already in the pinned buffer, so the stage is
      // the transfer and nothing else.
      const double stage_buf_ms =
          TimeGpu(100, [&] { gpu_->StageFromBuffer(num_slots, false); });
      const double stage_buf_real_ms =
          TimeGpu(100, [&] { gpu_->StageFromBuffer(num_slots, true); });
      const double whole_real_ms = TimeGpu(
          50, [&] { gpu_->EncodeReal(gpu_pt, level, scale, real_message); });
      const double fft_ms = TimeGpuWithSetup(
          100, [&] { gpu_->StageMessage(message, num_slots); },
          [&] { gpu_->SpecialIFFT(fft, num_slots); });
      const double fold_ms =
          TimeGpu(100, [&] { gpu_->FftToCoeff(coeff_dev, fft, num_slots); });
      const double ntt_ms = TimeGpu(100, [&] {
        auto view = gpu_pt.View();
        context_->ntt_handler_.NTT(view, np, gpu_pt.ConstView(), true);
      });
      const double whole_slot_ms =
          TimeGpu(50, [&] { gpu_->Encode(gpu_pt, level, scale, message); });
      const double whole_coeff_ms =
          TimeGpu(50, [&] { gpu_->EncodeCoeff(gpu_pt, level, scale, coeffs); });

      // Bytes each stage moves, counted rather than estimated. The transform
      // is two passes, each reading and writing the whole complex vector.
      const double fft_bytes = 2.0 * 2.0 * 2.0 * num_slots * sizeof(double);
      const double fold_bytes =
          2.0 * num_slots * sizeof(double) + 1.0 * degree * sizeof(double);
      const double rns_bytes = 1.0 * degree * sizeof(double) +
                               1.0 * degree * num_primes * sizeof(word);
      // A radix-2 butterfly is two complex adds and one complex multiply, so
      // ten flops, and there are (S/2) log2(S) of them.
      const double fft_flops = 10.0 * (num_slots / 2.0) * Log2Ceil(num_slots);

      Row("host EncodeCoeff (GMP)", host_coeff_ms, 0, 0, 0);
      Row("host Encode, slots (GMP)", host_slot_ms, 0, 0, 0);
      Row("gpu 0. StageMessage (H2D)", stage_ms,
          2.0 * num_slots * sizeof(double), 0, 0);
      Row("gpu 0.  H2D alone, same size", h2d_only_ms,
          static_cast<double>(stage_bytes), 0, 0);
      Row("gpu 0.  host memcpy alone", memcpy_only_ms,
          static_cast<double>(stage_bytes), 0, 0);
      Row("gpu 0. StageRealMessage", stage_real_ms,
          1.0 * num_slots * sizeof(double), 0, 0);
      Row("gpu 0. StageFromBuffer", stage_buf_ms,
          static_cast<double>(stage_bytes), 0, 0);
      Row("gpu 0. StageFromBuffer, real", stage_buf_real_ms,
          1.0 * num_slots * sizeof(double), 0, 0);
      Row("gpu 1. SpecialIFFT", fft_ms, fft_bytes, fft_flops, copy_gbps);
      Row("gpu 2. FftToCoeff", fold_ms, fold_bytes, 0, copy_gbps);
      Row("gpu 3. RnsDecompose", rns_ms, rns_bytes,
          static_cast<double>(degree) * num_primes, copy_gbps);
      Row("gpu 3. RnsDecompose + Mont", rns_mont_ms, rns_bytes,
          static_cast<double>(degree) * num_primes, copy_gbps);
      Row("gpu 4. NTT (existing)", ntt_ms, 0, 0, 0);
      Row("gpu EncodeCoeff, total", whole_coeff_ms, 0, 0, 0);
      Row("gpu Encode slots, total", whole_slot_ms, 0, 0, 0);
      Row("gpu EncodeReal slots, total", whole_real_ms, 0, 0, 0);
      std::cout << "  speedup: EncodeCoeff " << std::setprecision(1)
                << (host_coeff_ms / whole_coeff_ms) << "x, Encode "
                << (host_slot_ms / whole_slot_ms) << "x" << std::endl;
    }

    // ---- one weight-matrix group ----
    // The shape is the layer's own: a projection group is `rank/2` live output
    // rows against `parents * rank/2` input columns, which at rank 512 and 16
    // parents is 256 x 4096.
    {
      const int rows = 256;
      const int cols = 4096;
      const int entries = rows * cols;
      const std::vector<double> values = RandomReals(entries, 1.0);
      PcmmHandler<word> pcmm(*param_);
      PlainMatrix<word> host_u;
      const double host_ms = TimeHostMs(1, [&] {
        pcmm.EncodeMatrix(host_u, level, scale, values, rows, cols);
      });

      PlainMatrix<word> gpu_u;
      const double gpu_ms = TimeGpu(20, [&] {
        gpu_->EncodeMatrix(gpu_u, level, scale, values, rows, cols);
      });

      // And with the values already on the device, which is what a caller that
      // gathers its weight rows on the GPU hands over: it separates the PCIe
      // transfer from the encoding.
      DeviceVector<word> device_values(2 * entries);
      cudaMemcpyAsync(device_values.data(), values.data(),
                      static_cast<size_t>(entries) * sizeof(double),
                      cudaMemcpyHostToDevice, cudaStreamLegacy);
      cudaDeviceSynchronize();
      const double *dv = reinterpret_cast<const double *>(device_values.data());
      const double gpu_dev_ms = TimeGpu(20, [&] {
        gpu_->EncodeMatrixFromDevice(gpu_u, level, scale, dv, rows, cols);
      });

      const double in_bytes = static_cast<double>(entries) * sizeof(double);
      const double out_bytes =
          static_cast<double>(entries) * num_primes * sizeof(word);
      const double limbs = static_cast<double>(entries) * num_primes;
      std::cout << "  weight group " << rows << " x " << cols << " = "
                << (entries >> 10) << " Ki entries" << std::endl;
      Row("host EncodeMatrix (GMP)", host_ms, 0, limbs, 0);
      Row("gpu EncodeMatrix (from host)", gpu_ms, in_bytes + out_bytes, limbs,
          copy_gbps);
      Row("gpu EncodeMatrix (on device)", gpu_dev_ms, in_bytes + out_bytes,
          limbs, copy_gbps);
      std::cout << "  speedup: " << std::setprecision(1) << (host_ms / gpu_ms)
                << "x from host memory, " << (host_ms / gpu_dev_ms)
                << "x from device memory" << std::endl;

      // What that is worth at the layer's own size. A Llama-3-8B layer's seven
      // converted projection operands come to 3354 MiB of 4-byte limbs at the
      // model's width (CLAUDE.md's own table: gate/up/down 892 MiB each, q/o
      // 271, k/v 68), and a limb is one (entry, prime) pair -- so the layer's
      // conversion is that many of exactly the operation measured above.
      const double layer_limbs = 3354.0 * 1024 * 1024 / sizeof(word);
      std::cout << "  a layer's 3354 MiB of operands: host "
                << std::setprecision(1)
                << (layer_limbs / (limbs / (host_ms * 1e-3))) << " s, gpu "
                << std::setprecision(3)
                << (layer_limbs / (limbs / (gpu_ms * 1e-3)))
                << " s from host memory, "
                << (layer_limbs / (limbs / (gpu_dev_ms * 1e-3)))
                << " s on device" << std::endl;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Cheddar, EncodeGpuTest,
                         testing::Values("ci16_35.json", "bootparam_35.json"),
                         [](const testing::TestParamInfo<const char *> &info) {
                           std::string name(info.param);
                           std::replace(name.begin(), name.end(), '.', '_');
                           return name;
                         });
