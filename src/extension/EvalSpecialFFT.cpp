#include "extension/EvalSpecialFFT.h"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

namespace {

// ===========================================================================
// The inverse SinC conversion's row correction on the real subring, solved on
// a small reference ring (Doing.md 1.5bn).
//
// The FORWARD identity is clean: the suffix of the StC stage product, with
// every input column outside block 0 doubled, IS `Encoder::EncodeSinC`
// composed with the block bit reversal -- the exact analogue of the full StC's
// own phase-0 correction, measured exact at every size.
//
// The INVERSE prefix needs a complex row scaling lambda on top of its 1/d.
// Reading the banded SinC basis back is the ill-conditioned direction -- the
// same ~2k/pi amplification Doing.md 1.5bj measured reading SinC lanes
// through the ring switch -- and lambda carries it. Three measured facts make
// lambda computable in milliseconds:
//
//   * it depends only on the row's block field A = s / k and lane r = s % k;
//   * block 0 is exact (lambda = 1), block 1 has its own table f1(r), and
//     EVERY block A >= 2 shares one table f2(r);
//   * none of that depends on the ring degree.
//
// So the tables are solved on the smallest ring holding all classes,
// n_ref = 4k, by a per-row least-squares fit of Re(lambda * Z) against the
// target permutation -- and every fit's residual is asserted (< 1e-6) rather
// than trusted, so a convention drift fails loudly here instead of decoding
// garbage later.
// ===========================================================================

int SinCBitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

// Encoder::SpecialFFT / SpecialIFFT and the conjugate-invariant encoder,
// mirrored at an arbitrary power-of-two size with conductor 4 * size. The
// twiddle of a butterfly is exp(2 pi i (5^j mod 8s) / 8s), so these agree
// with the Encoder's transforms at every size regardless of the conductor
// the table is indexed by.
struct CiSinCReferenceRing {
  int n;
  int M;
  std::vector<Complex> tw;
  std::vector<int> gal;

  explicit CiSinCReferenceRing(int n_in) : n{n_in}, M{4 * n_in} {
    const double pi = std::acos(-1.0);
    tw.resize(M);
    gal.resize(M);
    for (int i = 0; i < M; i++) {
      tw[i] = std::polar(1.0, 2.0 * pi * i / M);
    }
    gal[0] = 1;
    for (int i = 1; i < M; i++) gal[i] = gal[i - 1] * 5 % M;
  }

  static void BitReverse(std::vector<Complex> &v) {
    const int size = static_cast<int>(v.size());
    const int bits = Log2Ceil(size);
    std::vector<Complex> out(size);
    for (int i = 0; i < size; i++) out[SinCBitRev(i, bits)] = v[i];
    v = std::move(out);
  }

  void Fft(std::vector<Complex> &data) const {
    const int size = static_cast<int>(data.size());
    BitReverse(data);
    for (int stride = 1; stride < size; stride *= 2) {
      const int st8 = stride << 3;
      const int gap = M / st8;
      for (int base = 0; base < size; base += 2 * stride) {
        for (int j = 0; j < stride; j++) {
          const Complex t = tw[(gal[j] % st8) * gap];
          const Complex x = data[base + j];
          const Complex y = data[base + j + stride] * t;
          data[base + j] = x + y;
          data[base + j + stride] = x - y;
        }
      }
    }
  }

  void Ifft(std::vector<Complex> &data) const {
    const int size = static_cast<int>(data.size());
    for (int stride = size / 2; stride >= 1; stride /= 2) {
      const int st8 = stride << 3;
      const int gap = M / st8;
      for (int base = 0; base < size; base += 2 * stride) {
        for (int j = 0; j < stride; j++) {
          const Complex t = tw[(st8 - gal[j] % st8) * gap];
          const Complex x = data[base + j] + data[base + j + stride];
          const Complex y = (data[base + j] - data[base + j + stride]) * t;
          data[base + j] = x;
          data[base + j + stride] = y;
        }
      }
    }
    BitReverse(data);
    for (auto &v : data) v /= static_cast<double>(size);
  }

  // Encoder::EncodeSinC's conjugate-invariant branch: per block the CI
  // encoder at size k (SpecialIFFT + real part), the banded two-term
  // recomposition between blocks.
  std::vector<double> EncodeSinC(const std::vector<double> &m, int k) const {
    const int d = n / k;
    std::vector<std::vector<double>> comp(d, std::vector<double>(k));
    std::vector<Complex> block(k);
    for (int i = 0; i < d; i++) {
      for (int t = 0; t < k; t++) block[t] = Complex(m[i * k + t], 0.0);
      Ifft(block);
      for (int t = 0; t < k; t++) comp[i][t] = block[t].real();
    }
    std::vector<double> out(n, 0.0);
    for (int t = 0; t < k; t++) {
      for (int i = 0; i < d; i++) {
        double v = comp[i][t];
        if (i != 0 && t + 1 < k) v += comp[d - i][t + 1];
        out[static_cast<size_t>(t) * d + i] = v;
      }
    }
    return out;
  }

  // The CI decode at full size: the conjugate-symmetric fold, then the FFT.
  std::vector<Complex> Decode(const std::vector<double> &a) const {
    std::vector<Complex> z(n);
    z[0] = Complex(a[0], 0.0);
    for (int t = 1; t < n; t++) z[t] = Complex(a[t], -a[n - t]);
    Fft(z);
    return z;
  }

  // One plain_ifft_stages_ matrix at the given stride -- the butterfly WITHOUT
  // any bit reversal, exactly as EvalSpecialFFT composes them.
  void ApplyIfftStage(std::vector<Complex> &data, int stride) const {
    const int size = static_cast<int>(data.size());
    const int st8 = stride << 3;
    const int gap = M / st8;
    for (int base = 0; base < size; base += 2 * stride) {
      for (int j = 0; j < stride; j++) {
        const Complex t = tw[(st8 - gal[j] % st8) * gap];
        const Complex x = data[base + j] + data[base + j + stride];
        const Complex y = (data[base + j] - data[base + j + stride]) * t;
        data[base + j] = x;
        data[base + j + stride] = y;
      }
    }
  }
};

// The dense reference solve is O(k^2 log k) time and O(k^2) memory; 256 keeps
// it in the low milliseconds. Every inverse the pipeline needs is at or below
// it -- and the one above it (sub_degree = degree / rank, the return leg of
// the ring-switch arrangement) is one the layer should not build anyway: its
// lambda reaches ~2k/pi ~ 2600, eleven bits of noise, where the bootstrap's
// full CtS does the same job well-conditioned.
constexpr int kCiSinCInverseLambdaCap = 256;

void SolveCiSinCInverseLambda(int k, std::vector<Complex> &f1,
                              std::vector<Complex> &f2) {
  const int nr = 4 * k;
  const int d_ref = 4;
  CiSinCReferenceRing ring(nr);

  // Z rows k..4k-1 (blocks A = 1, 2, 3), columns over the message index:
  // the prefix product (two stages, top strides first) over the decode of
  // the SinC encoding, divided by d_ref.
  std::vector<std::vector<Complex>> z(3 * k, std::vector<Complex>(nr));
  std::vector<double> m(nr, 0.0);
  for (int c = 0; c < nr; c++) {
    m.assign(nr, 0.0);
    m[c] = 1.0;
    auto u = ring.Decode(ring.EncodeSinC(m, k));
    for (auto &v : u) v = Complex(v.real(), 0.0);
    ring.ApplyIfftStage(u, nr / 2);
    ring.ApplyIfftStage(u, nr / 4);
    for (int s = 0; s < 3 * k; s++) {
      z[s][c] = u[k + s] / static_cast<double>(d_ref);
    }
  }

  f1.assign(k, Complex(0.0, 0.0));
  f2.assign(k, Complex(0.0, 0.0));
  for (int s = k; s < 4 * k; s++) {
    const int a_field = s / k;
    const int r = s % k;
    const int target_col = SinCBitRev(a_field, 2) * k + r;
    const auto &row = z[s - k];
    double srr = 0.0, sii = 0.0, sri = 0.0, br = 0.0, bi = 0.0;
    for (int c = 0; c < nr; c++) {
      const double zr = row[c].real();
      const double zi = row[c].imag();
      srr += zr * zr;
      sii += zi * zi;
      sri += zr * zi;
      const double t = (c == target_col) ? 1.0 : 0.0;
      br += zr * t;
      bi += zi * t;
    }
    // The 2x2 normal equations can be rank-one: a row whose imaginary part
    // vanishes (or is all there is) is still exactly solvable, just with the
    // other unknown pinned to zero. The residual below is the correctness
    // gate either way.
    double x = 0.0, y = 0.0;
    const double det = srr * sii - sri * sri;
    if (srr > 0.0 && det > 1e-14 * srr * sii) {
      x = (sii * br - sri * bi) / det;
      y = (sri * br - srr * bi) / det;
    } else if (srr >= sii && srr > 0.0) {
      // Rank one: the row's imaginary part is absent or parallel to the real
      // part, so a real lambda already spans everything reachable.
      x = br / srr;
    } else if (sii > 0.0) {
      y = -bi / sii;
    }
    double resid = 0.0;
    for (int c = 0; c < nr; c++) {
      const double t = (c == target_col) ? 1.0 : 0.0;
      resid = std::max(resid,
                       std::abs(x * row[c].real() - y * row[c].imag() - t));
    }
    AssertTrue(resid < 1e-6,
               "SolveCiSinCInverseLambda: the lambda fit does not close at "
               "row " + std::to_string(s) + " -- a transform convention "
               "drifted");
    const Complex lam(x, y);
    if (a_field == 1) {
      f1[r] = lam;
    } else if (a_field == 2) {
      f2[r] = lam;
    } else {
      AssertTrue(std::abs(lam - f2[r]) < 1e-6,
                 "SolveCiSinCInverseLambda: block 3 disagrees with the "
                 "shared table at lane " + std::to_string(r));
    }
  }
}

}  // namespace

template <typename word>
EvalSpecialFFT<word>::EvalSpecialFFT(ConstContextPtr<word> context,
                                     const BootParameter &boot_param,
                                     int num_slots, double cts_const,
                                     double stc_const)
    : num_slots_{num_slots},
      boot_param_{boot_param},
      cts_const_{cts_const},
      stc_const_{stc_const},
      full_slot_{num_slots == context->param_.MaxNumSlots()},
      conjugate_invariant_{context->param_.conjugate_invariant_} {
  AssertTrue(num_slots >= 256,
             "Currently only high number of slots are supported");
  AssertTrue(IsPowOfTwo(num_slots), "Number of slots must be a power of 2");
  AssertTrue(num_slots <= context->param_.MaxNumSlots(),
             "Number of slots exceeds the maximum possible");
  PopulatePlainMatrices(context);
  PreparePlaintexts(context);
}

template <typename word>
std::pair<int, int> EvalSpecialFFT<word>::BSGSSplit(int num_diag) const {
  AssertTrue(IsPowOfTwo(num_diag) || IsPowOfTwo(num_diag + 1),
             "Invalid number of diagonals for EvalSpecialFFT");
  // this is somewhat heuristic
  int bs, gs;
  if (num_diag <= 4 && !conjugate_invariant_) {
    // The conjugate-invariant path must fall through: its fused complex
    // giant step cannot evaluate a gs == 1 (swapped) layout, and a small
    // phase -- the SinC suffix's last one has 4 diagonals -- really does
    // land here. The CI split below never returns gs == 1.
    return {num_diag, 1};
  }

  // The conjugate-invariant transforms want a different split, and the reason
  // is a measured cost ratio, not a preference. A baby-step rotation rides the
  // double hoisting -- one ModUp shared by the whole step, so each extra
  // rotation is one fused key multiply, 0.11 ms at the CtS levels on an A100.
  // A giant-step rotation pays its own ModDown + ModUp + key multiply, 0.77 ms
  // there. That is a 7x ratio, and the balanced split is bs ~ sqrt(7 D)
  // rather than sqrt(D). It matters twice as much here as on the ordinary
  // path because the pair transform runs its giant step once per output half.
  //
  // The cap at 16 is the fused kernels: GSFusedComplexKernel carries four
  // register arrays of the baby-step count, and 4 x 32 words spills where
  // 4 x 16 holds (~104 registers, measured resident). The ordinary path is
  // deliberately left on the split below -- its baselines and evk footprints
  // are measured, and retuning it is its own change.
  if (conjugate_invariant_) {
    bs = Min(16, 1 << DivCeil(Log2Ceil(7 * num_diag), 2));
    // Never let the split collapse to a single giant step. At gs == 1 the
    // HoistHandler constructor swaps baby for giant, and the swapped layout
    // answers EvaluateBabyStep with a bare no-aux copy -- a shape the fused
    // complex giant step does not speak (it read num_aux_ == 0 and divided by
    // it). The top-stride CtS phase really does land here: its offsets are
    // multiples of n/16, so it has at most 16 diagonals.
    while (bs > 2 && DivCeil(num_diag, bs) < 2) bs /= 2;
    gs = DivCeil(num_diag, bs);
    AssertTrue(gs >= 2, "BSGSSplit: a conjugate-invariant phase with a "
                        "single giant step is not supported");
    return {bs, gs};
  }

  switch (num_diag) {
    case 7:
    case 8:
    case 15:
      // consider using bs = 5;
    case 16:
      bs = 4;
      break;
    case 31:
    case 32:
    case 63:
      // consider using bs = 9, 11;
    case 64:
      // consider using bs = 11;
      bs = 8;
      break;
    default:  // over 127, don't care actually
      bs = 1 << DivCeil(Log2Ceil(num_diag), 2);
      break;
  }
  gs = DivCeil(num_diag, bs);

  return {bs, gs};
}

template <typename word>
void EvalSpecialFFT<word>::PopulatePlainMatrices(
    ConstContextPtr<word> context) {
  // The twiddle a stage wants is exp(2 pi i (5^j mod st8) / st8), and it is
  // looked up as GetTwiddleFactor((5^j mod st8) * (M / st8)) against a table of
  // M-th roots -- so M cancels and the stage matrices depend on the slot count
  // alone. That is why the real subring needs no new plain matrices at all;
  // what it does need is for M here to be the ring's own cyclotomic index (4N,
  // not 2N), or the division M / st8 truncates and the lookup lands elsewhere.
  int M = context->param_.CyclotomicIndex();
  const auto &encoder = context->encoder_;

  int num_stages = Log2Ceil(num_slots_);
  plain_fft_stages_.resize(num_stages);
  plain_ifft_stages_.resize(num_stages);

  for (int i = 0; i < num_stages; i++) {
    int stride = 1 << i;
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M / st8;

    // Multiplication order left (0) --> right (num_stages - 1)
    auto &fft_target = plain_fft_stages_[i];
    auto &ifft_target = plain_ifft_stages_[num_stages - i - 1];

    fft_target = StripedMatrix(num_slots_, num_slots_);
    ifft_target = StripedMatrix(num_slots_, num_slots_);

    fft_target.try_emplace(0, num_slots_, Complex(0));
    fft_target.try_emplace(stride, num_slots_, Complex(0));
    if (i != num_stages - 1) {
      fft_target.try_emplace(num_slots_ - stride, num_slots_, Complex(0));
    }
    auto &fft_diag_0 = fft_target[0];
    auto &fft_diag_plus = fft_target[stride];
    auto &fft_diag_minus = fft_target[num_slots_ - stride];

    ifft_target.try_emplace(0, num_slots_, Complex(0));
    ifft_target.try_emplace(stride, num_slots_, Complex(0));
    if (i != num_stages - 1) {
      ifft_target.try_emplace(num_slots_ - stride, num_slots_, Complex(0));
    }
    auto &ifft_diag_0 = ifft_target[0];
    auto &ifft_diag_plus = ifft_target[stride];
    auto &ifft_diag_minus = ifft_target[num_slots_ - stride];

    for (int j = 0; j < stride; j++) {
      int fft_twiddle_index = (context->param_.GetGaloisFactor(j) % st8) * gap;
      int ifft_twiddle_index =
          (st8 - (context->param_.GetGaloisFactor(j) % st8)) * gap;
      Complex fft_twiddle = encoder.GetTwiddleFactor(fft_twiddle_index);
      Complex ifft_twiddle = encoder.GetTwiddleFactor(ifft_twiddle_index);

      // FFT
      // (x, y) = (x + y * twiddle, x - y * twiddle)
      fft_diag_0[j] = 1;
      fft_diag_plus[j] = fft_twiddle;
      fft_diag_minus[j + stride] = 1;
      fft_diag_0[j + stride] = -fft_twiddle;
      /* Matrix form
      fft_target[j][j] = 1;
      fft_target[j][j + stride] = fft_twiddle;
      fft_target[j + stride][j] = 1;
      fft_target[j + stride][j + stride] = -fft_twiddle;
      */

      // IFFT
      // (x, y) = (x + y, (x - y) * twiddle)
      ifft_diag_0[j] = 1;
      ifft_diag_plus[j] = 1;
      ifft_diag_minus[j + stride] = ifft_twiddle;
      ifft_diag_0[j + stride] = -ifft_twiddle;

      /* Matrix form
      ifft_target[j][j] = 1;
      ifft_target[j][j + stride] = 1;
      ifft_target[j + stride][j] = ifft_twiddle;
      ifft_target[j + stride][j + stride] = -ifft_twiddle;
      */
    }

    // For the rest, we can simply copy the values
    int num_double = Log2Ceil(num_slots_ / stride_group_size);

    for (int r = 0; r < num_double; r++) {
      std::copy(fft_diag_0.begin(),
                fft_diag_0.begin() + stride_group_size * (1 << r),
                fft_diag_0.begin() + stride_group_size * (1 << r));
      std::copy(fft_diag_plus.begin(),
                fft_diag_plus.begin() + stride_group_size * (1 << r),
                fft_diag_plus.begin() + stride_group_size * (1 << r));
      if (i != num_stages - 1) {
        std::copy(fft_diag_minus.begin(),
                  fft_diag_minus.begin() + stride_group_size * (1 << r),
                  fft_diag_minus.begin() + stride_group_size * (1 << r));
      }
      std::copy(ifft_diag_0.begin(),
                ifft_diag_0.begin() + stride_group_size * (1 << r),
                ifft_diag_0.begin() + stride_group_size * (1 << r));
      std::copy(ifft_diag_plus.begin(),
                ifft_diag_plus.begin() + stride_group_size * (1 << r),
                ifft_diag_plus.begin() + stride_group_size * (1 << r));
      if (i != num_stages - 1) {
        std::copy(ifft_diag_minus.begin(),
                  ifft_diag_minus.begin() + stride_group_size * (1 << r),
                  ifft_diag_minus.begin() + stride_group_size * (1 << r));
      }
    }
  }
}

template <typename word>
void EvalSpecialFFT<word>::PreparePlaintexts(ConstContextPtr<word> context) {
  int num_cts_phases = boot_param_.num_cts_levels_;
  int num_stc_phases = boot_param_.num_stc_levels_;
  int log_num_slots = Log2Ceil(num_slots_);

  int cts_level = boot_param_.GetCtSStartLevel();
  int stc_level = boot_param_.GetStCStartLevel();
  AssertTrue(num_cts_phases >= 2, "Use at least 2 levels for CtS");
  AssertTrue(num_stc_phases >= 2, "Use at least 2 levels for StC");

  int cts_stages_left = log_num_slots;
  int cts_stages_cumul = 0;
  double cts_const_div = std::pow(cts_const_, 1.0 / num_cts_phases);
  // std::cout << "cts_const_div: " << cts_const_div << std::endl;
  double stc_const_div = std::pow(stc_const_, 1.0 / num_stc_phases);
  // std::cout << "stc_const_div: " << stc_const_div << std::endl;

  // We will use different scaling methodology for CtS and StC
  double cts_scale = 1.0;
  for (int i = 0; i < cts_level; i++) {
    cts_scale *= context->param_.GetRescalePrimeProd(cts_level - i);
  }
  cts_scale = std::pow(cts_scale, 1.0 / num_cts_phases);

  for (int i = 0; i < num_cts_phases; i++) {
    std::cout << "CtS preparation phase " << i << std::endl;
    // CtS: high strides (num_slots / 2) --> low strides (1)
    int num_stages;
    if (i == 0) {
      num_stages = DivCeil(cts_stages_left, num_cts_phases);
    } else {
      num_stages = cts_stages_left / (num_cts_phases - i);
    }
    cts_stages_left -= num_stages;

    StripedMatrix phase_matrix = plain_ifft_stages_[cts_stages_cumul];
    for (int j = cts_stages_cumul + 1; j < cts_stages_cumul + num_stages; j++) {
      phase_matrix = StripedMatrix::Mult(plain_ifft_stages_[j], phase_matrix);
    }

    // Decomposing into Wx and -iWx part for later decomposition of real and
    // imag part for non-full-slot cases
    if (i == num_cts_phases - 1 && !full_slot_ && !conjugate_invariant_) {
      StripedMatrix extended(num_slots_ * 2, num_slots_ * 2);
      for (auto &[i, diag] : phase_matrix) {
        int dst_idx = i;
        if (i >= num_slots_ / 2) dst_idx += num_slots_;
        extended.try_emplace(dst_idx, num_slots_ * 2, Complex(0));
        for (int j = 0; j < num_slots_; j++) {
          extended.at(dst_idx)[j] = diag[j];
          extended.at(dst_idx)[j + num_slots_] = diag[j] * Complex(0, -1);
        }
      }
      phase_matrix = extended;
    }
    phase_matrix = StripedMatrix::Mult(phase_matrix, cts_const_div);

    int num_eff_diag = phase_matrix.GetNumDiag();
    if (i == num_cts_phases - 1) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    // std::cout << "CtS phase " << i << ": bs = " << bs << ", gs = " << gs
    //          << std::endl;

    // Min-KS adjustment (can be used also for hoisting)
    int pre_rotation;
    int additional_pt_rot = -(1 << cts_stages_left);
    if (i == 0) {
      pre_rotation = (1 << cts_stages_left);
    } else if (i == num_cts_phases - 1) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << cts_stages_left);
    }
    // std::cout << "Pre rotation: " << pre_rotation << std::endl;
    // std::cout << "Additional pt rot: " << additional_pt_rot << std::endl;

    const double cts_pt_scale =
        context->param_.GetRescalePrimeProd(cts_level - i);
    if (conjugate_invariant_) {
      cts_ci_phases_.emplace_back(context, phase_matrix, cts_level - i,
                                  cts_pt_scale, bs, gs, pre_rotation,
                                  additional_pt_rot);
    } else {
      cts_phases_.emplace_back(context, phase_matrix, cts_level - i,
                               cts_pt_scale, bs, gs, pre_rotation,
                               additional_pt_rot);
    }
    cts_stages_cumul += num_stages;
  }

  // 2. StC initialization
  int stc_stages_left = log_num_slots;
  int stc_stages_cumul = 0;
  for (int i = 0; i < num_stc_phases; i++) {
    std::cout << "StC preparation phase " << i << std::endl;
    // StC: low strides (1) --> high strides (num_slots / 2)
    int num_stages = stc_stages_left / (num_stc_phases - i);
    stc_stages_left -= num_stages;

    StripedMatrix phase_matrix = plain_fft_stages_[stc_stages_cumul];
    for (int j = stc_stages_cumul + 1; j < stc_stages_cumul + num_stages; j++) {
      phase_matrix = StripedMatrix::Mult(plain_fft_stages_[j], phase_matrix);
    }

    if (conjugate_invariant_ && i == 0) {
      // WHERE THE FACTOR OF TWO COMES FROM. The composed StC stage product is
      // the adjoint of the CtS one -- plain_fft_stages_[i] is exactly
      // plain_ifft_stages_[L-1-i] conjugate-transposed -- so taking the real
      // part after it gives Re(A^T), which is the transpose of what CtS
      // computes. On the real subring E^T E is diag(n, 2n, ..., 2n) rather than
      // a multiple of the identity, because the basis vector 1 has no conjugate
      // partner where every c_j has one, so the transpose is E^-1 only up to
      // that diagonal. Undoing it is a column scaling of the first phase, which
      // is free: the plaintexts were being built anyway.
      //
      // Index 0 is the odd one out in the CtS-output (bit-reversed) order too,
      // since BitReverse fixes it. Verified against the host encoder for every
      // degree and sparse slot count.
      for (auto &[idx, diag] : phase_matrix) {
        for (int j = 0; j < num_slots_; j++) {
          int col = ((j + idx) % num_slots_ + num_slots_) % num_slots_;
          if (col != 0) diag[j] *= 2.0;
        }
      }
    }

    if (i == 0 && !full_slot_ && !conjugate_invariant_) {
      StripedMatrix extended(num_slots_ * 2, num_slots_ * 2);
      for (auto &[i, diag] : phase_matrix) {
        int dst_idx = i;
        if (i >= num_slots_ / 2) dst_idx += num_slots_;
        extended.try_emplace(dst_idx, num_slots_ * 2, Complex(0));
        for (int j = 0; j < num_slots_; j++) {
          extended.at(dst_idx)[j] = diag[j];
          extended.at(dst_idx)[j + num_slots_] = diag[j] * Complex(0, 1);
        }
      }
      phase_matrix = extended;
    }
    phase_matrix = StripedMatrix::Mult(phase_matrix, stc_const_div);

    int num_eff_diag = phase_matrix.GetNumDiag();
    if (i == 0) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    // std::cout << "StC phase " << i << ": bs = " << bs << ", gs = " << gs
    //          << std::endl;

    // Min-KS adjustment (can be used also for hoisting)
    int pre_rotation, additional_pt_rot;
    if (i == 0) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = (1 << num_stages);
    } else if (i == num_stc_phases - 1) {
      pre_rotation = (1 << stc_stages_cumul);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << stc_stages_cumul);
      additional_pt_rot = (1 << (num_stages + stc_stages_cumul));
    }
    // std::cout << "Pre rotation: " << pre_rotation << std::endl;
    // std::cout << "Additional pt rot: " << additional_pt_rot << std::endl;

    // double stc_scale = context->param_.GetScale(stc_level - i);
    double stc_scale = context->param_.GetRescalePrimeProd(stc_level - i);
    if (conjugate_invariant_) {
      stc_ci_phases_.emplace_back(context, phase_matrix, stc_level - i,
                                  stc_scale, bs, gs, pre_rotation,
                                  additional_pt_rot);
    } else {
      stc_phases_.emplace_back(context, phase_matrix, stc_level - i, stc_scale,
                               bs, gs, pre_rotation, additional_pt_rot);
    }
    stc_stages_cumul += num_stages;
  }
}

template <typename word>
void EvalSpecialFFT<word>::AddRequiredRotations(EvkRequest &req,
                                                bool min_ks) const {
  for (const auto &cts_phase : cts_phases_) {
    cts_phase.AddRequiredRotations(req, min_ks);
  }
  for (const auto &stc_phase : stc_phases_) {
    stc_phase.AddRequiredRotations(req, min_ks);
  }
  for (const auto &cts_phase : cts_ci_phases_) {
    cts_phase.AddRequiredRotations(req, min_ks);
  }
  for (const auto &stc_phase : stc_ci_phases_) {
    stc_phase.AddRequiredRotations(req, min_ks);
  }
  if (!full_slot_ && !conjugate_invariant_) {
    // The sparse-packing path folds Wx and -iWx into a transform of twice the
    // width and then merges the halves with a rotation by num_slots_. The real
    // subring has no such extension: the intermediate is a pair of ciphertexts
    // whatever the slot count, so nothing extra is rotated here.
    req.AddRequest(num_slots_, boot_param_.GetEndLevel());
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateCtS(ConstContextPtr<word> context, Ct &res,
                                       const Ct &input,
                                       const EvkMap<word> &evk_map,
                                       bool min_ks) const {
  if (conjugate_invariant_) {
    // Real in, complex through the middle, real out. The first phase lifts the
    // input to a pair, the last drops the imaginary half it does not need, and
    // what comes back is the coefficient vector in bit-reversed order -- the
    // same thing the ordinary path returns, only real.
    int num_phases = static_cast<int>(cts_ci_phases_.size());
    Ct im;
    cts_ci_phases_.at(0).EvaluateFromReal(context, res, im, input, evk_map,
                                          min_ks);
    for (int i = 1; i < num_phases - 1; i++) {
      cts_ci_phases_.at(i).EvaluatePair(context, res, im, res, im, evk_map,
                                        min_ks);
    }
    cts_ci_phases_.at(num_phases - 1)
        .EvaluateToReal(context, res, res, im, evk_map, min_ks);
    res.SetNumSlots(num_slots_);
    return;
  }
  int num_cts_phases = cts_phases_.size();
  cts_phases_.at(0).Evaluate(context, res, input, evk_map, min_ks);
  for (int i = 1; i < num_cts_phases; i++) {
    cts_phases_.at(i).Evaluate(context, res, res, evk_map, min_ks);
  }
  if (!full_slot_) {
    res.SetNumSlots(num_slots_ * 2);
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateStC(ConstContextPtr<word> context, Ct &res,
                                       const Ct &input,
                                       const EvkMap<word> &evk_map,
                                       bool min_ks) const {
  if (conjugate_invariant_) {
    int num_phases = static_cast<int>(stc_ci_phases_.size());
    Ct im;
    stc_ci_phases_.at(0).EvaluateFromReal(context, res, im, input, evk_map,
                                          min_ks);
    for (int i = 1; i < num_phases - 1; i++) {
      stc_ci_phases_.at(i).EvaluatePair(context, res, im, res, im, evk_map,
                                        min_ks);
    }
    stc_ci_phases_.at(num_phases - 1)
        .EvaluateToReal(context, res, res, im, evk_map, min_ks);
    res.SetNumSlots(num_slots_);
    return;
  }
  int num_stc_phases = stc_phases_.size();
  stc_phases_.at(0).Evaluate(context, res, input, evk_map, min_ks);
  for (int i = 1; i < num_stc_phases; i++) {
    stc_phases_.at(i).Evaluate(context, res, res, evk_map, min_ks);
  }

  Ct tmp;
  if (!full_slot_) {
    res.SetNumSlots(num_slots_ * 2);
    // res += HRot(res, num_slots_)
    context->HRotAdd(res, res, res, evk_map.GetRotationKey(num_slots_),
                     num_slots_);
  }
  res.SetNumSlots(num_slots_);
}

template <typename word>
void EvalSpecialFFT<word>::PrepareSinC(ConstContextPtr<word> context,
                                       int sub_degree, int stc_level,
                                       int cts_level, int num_phases) {
  const int degree = context->param_.degree_;
  AssertTrue(!conjugate_invariant_ || num_phases >= 2,
             "PrepareSinC: the conjugate-invariant conversions carry a "
             "complex intermediate as a pair of real ciphertexts, so they "
             "need at least two phases -- one to lift, one to drop");
  AssertTrue(full_slot_,
             "PrepareSinC: the SinC conversions are defined on the full slot "
             "count; the sparse-packing path is a different transform");
  AssertTrue(IsPowOfTwo(sub_degree) && sub_degree >= 2 && sub_degree < degree,
             "PrepareSinC: sub_degree must be a power of two in [2, degree); "
             "SinC(degree) is the ordinary slot encoding and needs no "
             "conversion");
  AssertTrue(degree % sub_degree == 0, "PrepareSinC: sub_degree must divide "
                                       "the ring degree");

  const int num_stages = Log2Ceil(num_slots_);
  const int d = degree / sub_degree;
  const int p = Log2Ceil(d);
  AssertTrue(p <= num_stages,
             "PrepareSinC: sub_degree is smaller than the transform allows");

  AssertTrue(p >= 1, "PrepareSinC: nothing to do");
  AssertTrue(num_phases >= 1 && num_phases <= p,
             "PrepareSinC: the phase count must be between one and the stage "
             "count");
  AssertTrue(stc_level - num_phases >= 0 && cts_level - num_phases >= 0,
             "PrepareSinC: the transform spends one level per phase and there "
             "are not that many below the level it starts at");
  sinc_sub_degree_ = sub_degree;
  sinc_stc_.clear();
  sinc_cts_.clear();
  sinc_stc_ci_.clear();
  sinc_cts_ci_.clear();

  // The inverse row correction, solved on the 4k reference ring; see the
  // anonymous namespace above. Beyond the cap the inverse is not built at
  // all -- the forward still is -- and EvaluateSinCToSlot says so.
  std::vector<Complex> lambda_f1, lambda_f2;
  const bool build_inverse =
      !conjugate_invariant_ || sub_degree <= kCiSinCInverseLambdaCap;
  if (conjugate_invariant_ && build_inverse) {
    SolveCiSinCInverseLambda(sub_degree, lambda_f1, lambda_f2);
  }

  // HOW THE p STAGES ARE SPLIT, AND WHY THAT IS THE WHOLE COST QUESTION.
  //
  // A product of `q` butterfly stages has 2^q diagonals, and a diagonal is a
  // full plaintext at the transform's own limb count. So one phase carrying
  // all p stages is 2^p plaintexts -- 2048 for the sub_degree = 32 the
  // attention product wants, which is gigabytes -- while three phases of
  // 4 + 4 + 3 are 16 + 16 + 8 = 40. That is the same trade `PreparePlaintexts`
  // makes for StC itself (`num_stc_levels_`, three on every logN=16 preset),
  // and it is level-neutral in the pipeline: a tensor bound for the product
  // pays SlotToSinC *instead of* SlotToCoeff, not on top of it.
  //
  // The stages are split as evenly as p allows, largest group first, matching
  // how StC's own phases are apportioned.
  auto split = [&](int total, int phases) {
    std::vector<int> counts;
    int left = total;
    for (int i = 0; i < phases; i++) {
      const int take = (i == 0) ? DivCeil(left, phases) : left / (phases - i);
      counts.push_back(take);
      left -= take;
    }
    return counts;
  };
  const std::vector<int> counts = split(p, num_phases);

  // THE SHIFT BETWEEN PHASES, AND WHY A SPLIT TRANSFORM CANNOT DO WITHOUT ONE.
  //
  // A group of `q` consecutive butterfly stages starting at stride 2^c has its
  // offsets spread over multiples of 2^c in +-(2^(c+q) - 2^c) -- so they
  // STRADDLE ZERO, and reduced mod the slot count the negative ones land near
  // the top. `LinearTransform::DetermineStride` then sees a spread of nearly
  // the whole ring and demands `bs * gs >= num_slots / 2^c`, which for the
  // first phase of the sub_degree = 32 suffix is 2048 against 31 diagonals. It
  // does not merely cost keys; it refuses to build.
  //
  // A transform whose stages are ALL of them does not have this problem -- the
  // offsets wrap around and cover every residue at their common stride, which
  // is why the single-phase form works and why nothing needed this until now.
  //
  // The fix is the one `PreparePlaintexts` already uses for StC's own phases,
  // and it is free: `LinearTransform` computes
  // `rot(M . rot(x, -(a + p)), a)` for `p = pre_rotation` and
  // `a = additional_pt_rot`, so a chain of phases is exact iff
  //
  //     p_0 + a_0 = 0        the first phase takes an unrotated input
  //     p_{i+1} = a_i - a_{i+1}   each phase undoes the last one's rotation
  //     a_{last} = 0         and the last one leaves it unrotated
  //
  // Choosing `a_i = 2^(cumul_{i+1})` -- the stride the NEXT phase starts at --
  // satisfies all three and puts every phase's reduced offsets in
  // `[0, 2 (2^q - 1) 2^c]`, i.e. `bs * gs >= 2^(q+1) - 1`, which is exactly the
  // diagonal count. It is the same rule StC uses; the only difference here is
  // that the suffix starts at stride `2^(num_stages - p)` rather than at 1, and
  // for `num_phases == 1` every shift is zero and this reduces to the plain
  // transform the single-phase form was.
  int cursor = num_stages - p;
  int prev_a = 0;
  for (int phase = 0; phase < num_phases; phase++) {
    StripedMatrix forward = plain_fft_stages_[cursor];
    for (int j = cursor + 1; j < cursor + counts[phase]; j++) {
      forward = StripedMatrix::Mult(plain_fft_stages_[j], forward);
    }
    cursor += counts[phase];
    const bool last = (phase == num_phases - 1);
    const int a = last ? 0 : (1 << cursor);
    const int pre_rotation = (phase == 0) ? -a : (prev_a - a);
    prev_a = a;

    if (conjugate_invariant_ && phase == 0) {
      // The forward correction (Doing.md 1.5bn): double every input column
      // outside block 0. Same mechanism as the full StC's phase-0 column
      // scaling, same reason -- on the real subring the stage product's
      // real part is the transpose of the encode only up to the diagonal
      // the fold degeneracy leaves -- and phase 0 is the one phase whose
      // input is unrotated, so its columns ARE the slot index.
      for (auto &[idx, diag] : forward) {
        for (int j = 0; j < num_slots_; j++) {
          const int col = ((j + idx) % num_slots_ + num_slots_) % num_slots_;
          if (col >= sub_degree) diag[j] *= 2.0;
        }
      }
    }

    const int level = stc_level - phase;
    auto [fbs, fgs] = BSGSSplit(forward.GetNumDiag());
    std::cout << "SinC forward phase " << phase << ": " << counts[phase]
              << " stages, " << forward.GetNumDiag() << " diagonals, level "
              << level << ", BSGS " << fbs << "x" << fgs << ", pre_rotation "
              << pre_rotation << ", pt_rot " << a << std::endl;
    if (conjugate_invariant_) {
      sinc_stc_ci_.emplace_back(context, forward, level,
                                context->param_.GetRescalePrimeProd(level),
                                fbs, fgs, pre_rotation, a);
    } else {
      sinc_stc_.emplace_back(context, forward, level,
                             context->param_.GetRescalePrimeProd(level), fbs,
                             fgs, pre_rotation, a);
    }
  }

  if (!build_inverse) {
    std::cout << "SinC inverse: not built (conjugate-invariant sub_degree "
              << sub_degree << " above the lambda cap "
              << kCiSinCInverseLambdaCap << ")" << std::endl;
    return;
  }

  // SinC -> slots: the PREFIX of CtS, which is the same set of strides in the
  // opposite order, times 1/d. `plain_ifft_stages_[num_stages-1-i]` holds
  // stride 2^i, so index 0 is the highest stride -- the one the forward
  // applied last, and therefore the one the inverse applies first. The 1/d
  // rides the first phase; each stage pair composes to 2I, so p of them
  // compose to 2^p = d.
  // The same chain, mirrored. `plain_ifft_stages_[j]` holds stride
  // 2^(num_stages-1-j), so a phase ending at index `cursor` has its LOWEST
  // stride at 2^(num_stages - cursor); that exponent is what `a` is built from,
  // negated because CtS descends where StC ascends.
  cursor = 0;
  prev_a = 0;
  for (int phase = 0; phase < num_phases; phase++) {
    StripedMatrix inverse = plain_ifft_stages_[cursor];
    for (int j = cursor + 1; j < cursor + counts[phase]; j++) {
      inverse = StripedMatrix::Mult(plain_ifft_stages_[j], inverse);
    }
    cursor += counts[phase];
    if (phase == 0) {
      inverse = StripedMatrix::Mult(inverse, 1.0 / static_cast<double>(d));
    }
    const bool last = (phase == num_phases - 1);
    const int a = last ? 0 : -(1 << (num_stages - cursor));
    const int pre_rotation = (phase == 0) ? -a : (prev_a - a);
    prev_a = a;

    if (conjugate_invariant_ && last) {
      // The inverse row correction (Doing.md 1.5bn): the complex lambda per
      // output row, folded into the last phase -- the one phase whose output
      // is unrotated, so its rows ARE the slot index. Block 0 is exact,
      // block 1 has its own table, every other block shares one.
      for (auto &[idx, diag] : inverse) {
        for (int j = 0; j < num_slots_; j++) {
          const int block = j / sub_degree;
          if (block == 1) {
            diag[j] *= lambda_f1[j % sub_degree];
          } else if (block >= 2) {
            diag[j] *= lambda_f2[j % sub_degree];
          }
        }
      }
    }

    const int level = cts_level - phase;
    auto [ibs, igs] = BSGSSplit(inverse.GetNumDiag());
    std::cout << "SinC inverse phase " << phase << ": " << counts[phase]
              << " stages, " << inverse.GetNumDiag() << " diagonals, level "
              << level << ", BSGS " << ibs << "x" << igs << ", pre_rotation "
              << pre_rotation << ", pt_rot " << a << std::endl;
    if (conjugate_invariant_) {
      sinc_cts_ci_.emplace_back(context, inverse, level,
                                context->param_.GetRescalePrimeProd(level),
                                ibs, igs, pre_rotation, a);
    } else {
      sinc_cts_.emplace_back(context, inverse, level,
                             context->param_.GetRescalePrimeProd(level), ibs,
                             igs, pre_rotation, a);
    }
  }
}

template <typename word>
StripedMatrix EvalSpecialFFT<word>::SinCPrefixMatrix(int sub_degree,
                                                     int &window) const {
  AssertTrue(!conjugate_invariant_,
             "SinCPrefixMatrix: no conjugate-invariant form yet. The "
             "SlotToSinC / SinCToSlot pair has one (Doing.md 1.5bn); the "
             "prefix is HalfBoot's counterpart and waits for the CI "
             "schedule");
  AssertTrue(full_slot_,
             "SinCPrefixMatrix: the SinC conversions are defined on the full "
             "slot count");
  const int degree = num_slots_ * 2;
  AssertTrue(IsPowOfTwo(sub_degree) && sub_degree >= 2 && sub_degree < degree,
             "SinCPrefixMatrix: sub_degree must be a power of two in "
             "[2, degree)");
  const int num_stages = Log2Ceil(num_slots_);
  const int q = num_stages - Log2Ceil(degree / sub_degree);
  AssertTrue(q >= 1,
             "SinCPrefixMatrix: SlotToSinC is the whole of SlotToCoeff at this "
             "sub_degree, so HalfBoot leaves nothing undone");
  StripedMatrix prefix = plain_fft_stages_[0];
  for (int j = 1; j < q; j++) {
    prefix = StripedMatrix::Mult(plain_fft_stages_[j], prefix);
  }
  window = 1 << q;
  return prefix;
}

template <typename word>
void EvalSpecialFFT<word>::PrepareSinCPrefix(ConstContextPtr<word> context,
                                             int sub_degree, int level,
                                             int num_phases, double constant,
                                             double pt_scale) {
  const int degree = context->param_.degree_;
  AssertTrue(!conjugate_invariant_,
             "PrepareSinCPrefix: no conjugate-invariant form yet. The "
             "SlotToSinC / SinCToSlot pair has one (Doing.md 1.5bn); the "
             "prefix is HalfBoot's counterpart and waits for the CI "
             "schedule");
  AssertTrue(full_slot_,
             "PrepareSinCPrefix: the SinC conversions are defined on the full "
             "slot count");
  AssertTrue(IsPowOfTwo(sub_degree) && sub_degree >= 2 && sub_degree < degree,
             "PrepareSinCPrefix: sub_degree must be a power of two in "
             "[2, degree)");
  AssertTrue(degree % sub_degree == 0,
             "PrepareSinCPrefix: sub_degree must divide the ring degree");

  const int num_stages = Log2Ceil(num_slots_);
  const int d = degree / sub_degree;
  const int p = Log2Ceil(d);
  const int q = num_stages - p;
  AssertTrue(q >= 1,
             "PrepareSinCPrefix: SlotToSinC is the whole of SlotToCoeff at "
             "this sub_degree, so HalfBoot leaves nothing undone");
  AssertTrue(num_phases >= 1 && num_phases <= q,
             "PrepareSinCPrefix: the phase count must be between one and the "
             "stage count");
  AssertTrue(level - num_phases >= 0,
             "PrepareSinCPrefix: the transform spends one level per phase and "
             "there are not that many below the level it starts at");
  sinc_prefix_.clear();
  sinc_prefix_level_ = level;

  auto split = [&](int total, int phases) {
    std::vector<int> counts;
    int left = total;
    for (int i = 0; i < phases; i++) {
      const int take = (i == 0) ? DivCeil(left, phases) : left / (phases - i);
      counts.push_back(take);
      left -= take;
    }
    return counts;
  };
  const std::vector<int> counts = split(q, num_phases);

  // THE WINDOW, AND WHY THIS TRANSFORM CANNOT DO WITHOUT ONE EITHER.
  //
  // The prefix starts at stride 1, so `q` stages give offsets straddling zero
  // over +-(2^q - 1) -- reduced mod the slot count they cover both ends of the
  // ring, `DetermineStride` finds gcd 1 and a spread of `num_slots - 1`, and
  // it demands `bs * gs >= num_slots`. Exactly the wall `PrepareSinC` hit, and
  // the same fix: the chain rule is
  //
  //     p_0 + a_0 = 0,   p_{i+1} = a_i - a_{i+1}
  //
  // with `a_i = 2^(cumul_{i+1})` the stride the next group starts at. The one
  // difference is the LAST phase. In a chain that continues, `a_last` must be
  // zero so the next transform sees an unrotated input; here nothing follows,
  // so `a_last` keeps the same rule and comes out `2^q` -- which is precisely
  // the stride SlotToSinC's first stage would have used. The result is
  // therefore `rot(P x, 2^q)`, and one HRot by `-2^q` finishes it.
  //
  // That HRot is the whole extra cost over a chained phase, and it is one key
  // switch against a bootstrap.
  int cursor = 0;
  int prev_a = 0;
  for (int phase = 0; phase < num_phases; phase++) {
    StripedMatrix prefix = plain_fft_stages_[cursor];
    for (int j = cursor + 1; j < cursor + counts[phase]; j++) {
      prefix = StripedMatrix::Mult(plain_fft_stages_[j], prefix);
    }
    cursor += counts[phase];
    // Canonicalise rides the first phase: it is one constant multiply and one
    // rescale at exactly this level, so folding it into the diagonals costs
    // nothing and saves the level the schedule would have spent on it.
    if (phase == 0 && constant != 1.0) {
      prefix = StripedMatrix::Mult(prefix, constant);
    }
    const int a = 1 << cursor;
    const int pre_rotation = (phase == 0) ? -a : (prev_a - a);
    prev_a = a;

    const int phase_level = level - phase;
    const double scale =
        (phase == 0 && pt_scale > 0.0)
            ? pt_scale
            : context->param_.GetRescalePrimeProd(phase_level);
    auto [bs, gs] = BSGSSplit(prefix.GetNumDiag());
    std::cout << "SinC prefix phase " << phase << ": " << counts[phase]
              << " stages, " << prefix.GetNumDiag() << " diagonals, level "
              << phase_level << ", BSGS " << bs << "x" << gs
              << ", pre_rotation " << pre_rotation << ", pt_rot " << a
              << std::endl;
    sinc_prefix_.emplace_back(context, prefix, phase_level, scale, bs, gs,
                              pre_rotation, a);
  }
  sinc_prefix_shift_ = prev_a;
}

template <typename word>
void EvalSpecialFFT<word>::AddRequiredSinCPrefixRotations(
    EvkRequest &req) const {
  if (sinc_prefix_.empty()) return;
  for (const auto &lt : sinc_prefix_) lt.AddRequiredRotations(req);
  // The window is undone on the output, which is one level below the last
  // phase.
  const int back = num_slots_ - sinc_prefix_shift_;
  req.AddRequest(back, sinc_prefix_level_ - static_cast<int>(sinc_prefix_.size()));
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateSinCPrefix(ConstContextPtr<word> context,
                                              Ct &res, const Ct &input,
                                              const EvkMap<word> &evk_map)
    const {
  AssertTrue(!sinc_prefix_.empty(),
             "EvaluateSinCPrefix: call PrepareSinCPrefix first");
  Ct shifted;
  sinc_prefix_.front().Evaluate(context, shifted, input, evk_map);
  for (size_t i = 1; i < sinc_prefix_.size(); i++) {
    Ct next;
    sinc_prefix_[i].Evaluate(context, next, shifted, evk_map);
    shifted = std::move(next);
  }
  const int back = num_slots_ - sinc_prefix_shift_;
  context->HRot(res, shifted, evk_map.GetRotationKey(back), back);
}

template <typename word>
void EvalSpecialFFT<word>::AddRequiredSinCRotations(EvkRequest &req) const {
  for (const auto &lt : sinc_stc_) lt.AddRequiredRotations(req);
  for (const auto &lt : sinc_cts_) lt.AddRequiredRotations(req);
  for (const auto &lt : sinc_stc_ci_) lt.AddRequiredRotations(req);
  for (const auto &lt : sinc_cts_ci_) lt.AddRequiredRotations(req);
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateSlotToSinC(
    ConstContextPtr<word> context, Ct &res, const Ct &input,
    const EvkMap<word> &evk_map) const {
  if (conjugate_invariant_) {
    AssertTrue(!sinc_stc_ci_.empty(),
               "EvaluateSlotToSinC: call PrepareSinC first");
    const int num_phases = static_cast<int>(sinc_stc_ci_.size());
    Ct im;
    sinc_stc_ci_.at(0).EvaluateFromReal(context, res, im, input, evk_map);
    for (int i = 1; i < num_phases - 1; i++) {
      sinc_stc_ci_.at(i).EvaluatePair(context, res, im, res, im, evk_map);
    }
    sinc_stc_ci_.at(num_phases - 1)
        .EvaluateToReal(context, res, res, im, evk_map);
    res.SetNumSlots(num_slots_);
    return;
  }
  AssertTrue(!sinc_stc_.empty(), "EvaluateSlotToSinC: call PrepareSinC first");
  sinc_stc_.front().Evaluate(context, res, input, evk_map);
  for (size_t i = 1; i < sinc_stc_.size(); i++) {
    Ct next;
    sinc_stc_[i].Evaluate(context, next, res, evk_map);
    res = std::move(next);
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateSinCToSlot(
    ConstContextPtr<word> context, Ct &res, const Ct &input,
    const EvkMap<word> &evk_map) const {
  if (conjugate_invariant_) {
    AssertTrue(!sinc_cts_ci_.empty(),
               "EvaluateSinCToSlot: call PrepareSinC first -- and note the "
               "conjugate-invariant inverse is only built for sub_degree <= "
               "256, the lambda cap of Doing.md 1.5bn");
    const int num_phases = static_cast<int>(sinc_cts_ci_.size());
    Ct im;
    sinc_cts_ci_.at(0).EvaluateFromReal(context, res, im, input, evk_map);
    for (int i = 1; i < num_phases - 1; i++) {
      sinc_cts_ci_.at(i).EvaluatePair(context, res, im, res, im, evk_map);
    }
    sinc_cts_ci_.at(num_phases - 1)
        .EvaluateToReal(context, res, res, im, evk_map);
    res.SetNumSlots(num_slots_);
    return;
  }
  AssertTrue(!sinc_cts_.empty(), "EvaluateSinCToSlot: call PrepareSinC first");
  sinc_cts_.front().Evaluate(context, res, input, evk_map);
  for (size_t i = 1; i < sinc_cts_.size(); i++) {
    Ct next;
    sinc_cts_[i].Evaluate(context, next, res, evk_map);
    res = std::move(next);
  }
}

template class EvalSpecialFFT<uint32_t>;
template class EvalSpecialFFT<uint64_t>;

}  // namespace cheddar
