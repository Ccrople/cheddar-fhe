#include "extension/Profile.h"
#include "extension/CiLlamaLayer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/Serialization.h"

namespace cheddar {

namespace {

int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

}  // namespace

// The projection leg with its ciphertext-ciphertext half stubbed out. On R+
// those are `CiSinCAttention`'s, by a different road entirely -- SinC operands,
// the ring switch, the lifted product -- so a leg reached from here can only
// ever be asked for `Project`, and saying so is better than a silent fallback.
template <typename word>
class CiProjectionLeg : public CoeffLinearLeg<word> {
 public:
  using CoeffLinearLeg<word>::CoeffLinearLeg;
  void Scores(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double,
              const std::vector<double> &) const override {
    Fail("CiProjectionLeg: the CI score product is CiSinCAttention's");
  }
  void Values(std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &,
              const std::vector<Ciphertext<word>> &, double) const override {
    Fail("CiProjectionLeg: the CI value product is CiSinCAttention's");
  }
  void LocateScore(int, int, int, int &, int &) const override {
    Fail("CiProjectionLeg: the CI score layout is CiSinCAttention's");
  }
};

template <typename word>
CiLlamaLayer<word>::CiLlamaLayer(
    std::shared_ptr<const BootContext<word>> boot,
    const CiSwitchedCcmmLayout &layout,
    std::vector<const EvaluationKey<word> *> modpack_keys, const Config &cfg)
    : boot_{std::move(boot)},
      cfg_{cfg},
      sched_{boot_, boot_->param_.MaxNumSlots()} {
  num_slots_ = boot_->param_.MaxNumSlots();
  // The seam's images: two half-density ones per booted ciphertext on the
  // banded image, one dense one on the module basis.
  attn_channels_ =
      (cfg_.module_basis ? 1 : 2) * layout.num_cts * cfg_.proj_rank;

  AssertTrue(cfg_.model_declared % cfg_.proj_rank == 0 &&
                 cfg_.hidden_declared % cfg_.proj_rank == 0,
             "CiLlamaLayer: both declared widths must be a whole number of "
             "half-density ciphertexts");
  num_model_cts_ = cfg_.model_declared / cfg_.proj_rank;
  num_hidden_cts_ = cfg_.hidden_declared / cfg_.proj_rank;
  // A half-density ciphertext carries `rank/2 - 1` live model channels --
  // component zero has no partner -- and `rank/2` hidden ones, because
  // nothing reduces over the hidden axis. Seventeen of the first hold Llama's
  // 4096; sixteen do not. On the module basis every component is a channel
  // and eight do.
  channel_stride_ = cfg_.module_basis ? 1 : 2;
  const int per_model =
      cfg_.module_basis ? cfg_.proj_rank : cfg_.proj_rank / 2 - 1;
  AssertTrue(num_model_cts_ * per_model >= cfg_.model_live,
             "CiLlamaLayer: " + std::to_string(num_model_cts_) +
                 (cfg_.module_basis ? " dense" : " half-density") +
                 " ciphertexts carry " +
                 std::to_string(num_model_cts_ * per_model) +
                 " live model channels, short of " +
                 std::to_string(cfg_.model_live) +
                 (cfg_.module_basis ? "" : " -- component zero has no partner"));
  if (cfg_.module_basis) {
    AssertTrue(boot_->param_.conjugate_invariant_,
               "CiLlamaLayer: the module basis is a conjugate-invariant "
               "object");
    AssertTrue(!cfg_.keep_component_zero,
               "CiLlamaLayer: keep_component_zero reproduces the banded "
               "contract, which the module basis does not have");
  }

  // THE CROSSING CONSTANT IS DERIVED, NOT FITTED. `HalfBoot` multiplies the
  // message by `level_zero_scale / q0` and `BootContext` computes both halves
  // already; the nominal `2^-log_message_ratio` is what the design ASKS for
  // and differs from what it gets by the rounding in `log_scaleup_`
  // (Doing.md 1.5dk). A constant fitted on one preset and carried to another
  // is wrong by percents.
  crossing_ = boot_->GetMessageRatio();
  AssertTrue(crossing_ > 0.0,
             "CiLlamaLayer: PrepareEvalMod must run before construction");
  // What a FULL turn carries: `ToCoeff` undoes the crossing by the NOMINAL
  // ratio, deliberately, because that is what makes `Boot` message
  // preserving. Every linear stage absorbs the difference and RMSNorm is
  // scale invariant, so it reaches SiLU unchallenged -- and
  // `SiLU(kx)/k - SiLU(x)` is not noise, it is a different function.
  kappa_ =
      std::pow(2.0, -boot_->GetBootParameter().GetLogMessageRatio()) / crossing_;

  slot_level_ = sched_.GetSlotLevel();
  op_level_ = slot_level_ - 1;

  if (cfg_.module_basis) {
    // The module transforms (Doing.md 3.5/3.6): StC at the schedule's StC
    // level in two real phases, with the native constant restated for the
    // level it lands on; CtS at the boot's CtS start in as many levels as the
    // BootParameter gives CoeffToSlot, with `n * cts_const` folded, which is
    // what `HalfBootModule` requires. `T` is the small degree, which on R+ is
    // the token count.
    const int cts_levels = boot_->GetBootParameter().num_cts_levels_;
    typename CiModuleBasis<word>::Phases phases;
    phases.stc_small = {7};
    phases.stc_twist = {9};
    if (cts_levels == 2) {
      phases.cts_twist = {9};
      phases.cts_small = {7};
    } else if (cts_levels == 3) {
      phases.cts_twist = {9};
      phases.cts_small = {4, 3};
    } else {
      AssertTrue(cts_levels == 4,
                 "CiLlamaLayer: the module CtS is grouped for 2, 3 or 4 "
                 "CoeffToSlot levels");
      phases.cts_twist = {5, 4};
      phases.cts_small = {4, 3};
    }
    const int stc_levels = static_cast<int>(phases.stc_small.size()) +
                           static_cast<int>(phases.stc_twist.size());
    const double stc_const = sched_.ModuleStCConst(stc_levels);
    const double cts_const = num_slots_ * boot_->GetCtSConst();
    const int cts_level = boot_->GetBootParameter().GetCtSStartLevel();
    // THE COMPILE, CACHED (Doing.md 3.20/3.21): 37.7 s of the 76 s setup is
    // this constructor, and every input to it is in the recipe below or in
    // the parameter set, which `ArchiveIdentity` guards. Written aside and
    // renamed, as the converter cache is, so the file is whole or absent.
    std::string cache_path;
    if (const char *dir = std::getenv("CHEDDAR_MODULE_BASIS_CACHE");
        dir != nullptr && *dir != 0) {
      auto bits = [](double v) {
        uint64_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        std::ostringstream s;
        s << std::hex << u;
        return s.str();
      };
      std::ostringstream os;
      os << dir << "/ci_module_basis_T" << cfg_.num_tokens << "_stc"
         << sched_.GetStCLevel() << "_cts" << cts_level << "_p";
      for (const auto *g : {&phases.stc_small, &phases.stc_twist,
                            &phases.cts_twist, &phases.cts_small}) {
        for (int c : *g) os << c;
        os << "-";
      }
      os << "_s" << bits(stc_const) << "_c" << bits(cts_const) << "_n"
         << num_slots_ << ".bin";
      cache_path = os.str();
    }
    const auto id = IdentityOf(boot_->param_);
    if (!cache_path.empty() &&
        ArchiveReader::PeekIdentity(cache_path) == id) {
      ArchiveReader ar(cache_path, id);
      basis_ = CiModuleBasis<word>::Load(ar);
      std::cout << "CiLlamaLayer: module basis read, "
                << (ArchiveReader::FileSize(cache_path) >> 20) << " MiB from "
                << cache_path << std::endl;
    } else {
      basis_ = std::make_unique<CiModuleBasis<word>>(
          boot_, cfg_.num_tokens, sched_.GetStCLevel(), cts_level, phases,
          stc_const, cts_const);
      if (!cache_path.empty()) {
        const std::string tmp = cache_path + ".tmp";
        int64_t written = 0;
        {
          ArchiveWriter ar(tmp, id);
          basis_->Save(ar);
          ar.Close();
          written = ar.Written();
        }
        AssertTrue(std::rename(tmp.c_str(), cache_path.c_str()) == 0,
                   "CiLlamaLayer: could not move the module basis cache into "
                   "place at " + cache_path);
        std::cout << "CiLlamaLayer: module basis written, " << (written >> 20)
                  << " MiB to " << cache_path << std::endl;
      }
    }
    sched_.SetModuleBasis(basis_.get());
    MemoryPool::Report("layer ctor: + the module basis (StC'/CtS' plaintexts)");
  }

  typename CiLlamaSeam<word>::Config scfg;
  scfg.proj_rank = cfg_.proj_rank;
  scfg.module_basis = cfg_.module_basis;
  scfg.verbose = cfg_.verbose;
  seam_ = std::make_unique<CiLlamaSeam<word>>(boot_, layout,
                                              sched_.GetStCLevel(), scfg);
  MemoryPool::Report("layer ctor: + the seam (T2/rev stages)");

  typename CoeffLinearLeg<word>::Config lcfg;
  lcfg.num_tokens = cfg_.num_tokens;
  lcfg.product_level = cfg_.product_level;
  lcfg.parents_per_tile = cfg_.parents_per_tile;
  // HALF DENSITY ON BOTH AXES, which is a statement about these operands and
  // not a tuning choice. Every parent here is a banded half-density image, so
  // its live module components are a contiguous prefix and the descent stops
  // there; every output is one too, so half of `GatherWeights`'s rows are
  // exact zeros that the operand would store, the GEMM would multiply and
  // `ModPack` would recompose. Measured exact at both settings and 1.9991x /
  // 1.69x apart in work (Doing.md 1.5db/1.5dh). On the module basis every
  // component is live at both ends, and the one projection whose ends differ
  // (O: the seam's banded images in, the dense stream out) states its own
  // densities per call.
  lcfg.input_density = GetDensity();
  lcfg.output_density = GetDensity();
  leg_ = std::make_unique<CiProjectionLeg<word>>(boot_, lcfg,
                                                 std::move(modpack_keys));
  MemoryPool::Report("layer ctor: + the projection leg (PC-MM handlers)");

  if (cfg_.verbose) {
    std::cout << "CiLlamaLayer: slot " << sched_.GetSlotLevel() << ", StC "
              << sched_.GetStCLevel() << ", coeff " << sched_.GetCoeffLevel()
              << ", seam input at " << seam_->GetInputLevel()
              << ", crossing " << crossing_ << " (2^"
              << std::log2(std::abs(crossing_)) << "), kappa " << kappa_
              << (cfg_.module_basis ? ", MODULE basis" : ", native basis")
              << ", stream " << num_model_cts_ << " cts, hidden "
              << num_hidden_cts_ << " cts" << std::endl;
    if (basis_) {
      std::cout << "CiLlamaLayer: module StC " << basis_->GetStCNumLevels()
                << " levels from " << basis_->GetStCLevel() << ", module CtS "
                << basis_->GetCtSNumLevels() << " levels from "
                << basis_->GetCtSLevel() << std::endl;
    }
  }
}

template <typename word>
void CiLlamaLayer<word>::AddRequiredRotations(EvkRequest &req) const {
  seam_->AddRequiredRotations(req);
  // RMSNorm's distances are a property of the SHAPE -- the reduction tree over
  // `num_channels` at `channel_stride` -- while its `alpha` and window are
  // per-layer calibration that does not outlive one `FeedForward`. So a
  // handler is built here only to be asked what it will rotate by.
  // A concrete degree, not `cfg_.rms_degree`: zero there means "derive it from
  // the window", and the window is per-layer calibration that does not exist
  // yet. The rotation distances do not depend on either.
  RmsNormHandler<word> probe(boot_, cfg_.num_tokens, cfg_.model_declared, 1.0,
                             op_level_, 1e-5, 2.0, NormDegree(2.0),
                             channel_stride_);
  for (int d : probe.GetRotationDistances()) req.AddRequest(d, op_level_);
  if (basis_) basis_->AddRequiredRotations(req, cfg_.min_ks);
}

template <typename word>
void CiLlamaLayer<word>::PrepareSeamHalf(int half) {
  seam_->PrepareHalf(half);
}

template <typename word>
void CiLlamaLayer<word>::AddSeamHalfRotations(EvkRequest &req) const {
  seam_->AddHalfRotations(req);
}

template <typename word>
void CiLlamaLayer<word>::DropSeamHalf() {
  seam_->DropHalf();
}

template <typename word>
void CiLlamaLayer<word>::Seam(Ct &res, const Ct &booted,
                              const EvkMap<word> &evk) {
  NvtxScope _nv("layer: Seam");
  seam_->Apply(res, booted, sched_, evk, cfg_.min_ks);
}

template <typename word>
int CiLlamaLayer<word>::SiLuDegree(double range) const {
  if (cfg_.silu_degree > 0) return cfg_.silu_degree;
  // Chebyshev's error on an analytic function falls as rho^-degree, where rho
  // is the Bernstein ellipse the nearest singularity sits on. SiLU is
  // x*sigmoid(x), whose poles are at x = i*pi*(2k+1), so on [-r, r]
  //     rho = (pi + sqrt(pi^2 + r^2)) / r
  // and the degree that reaches a given number of bits is linear in 1/log rho.
  // Sixteen bits is the target because that is where 1.5cv measured SiLU's
  // CIRCUIT (2^-15.2 to 2^-16.0 on a fresh encryption at every level), and a
  // fit below the circuit is a fit that costs nothing. Checked against four
  // fits computed in double: range 4.46 gives 31 (3.0e-9) and range 18.30
  // gives 63 (7.2e-5), where the shipped 31 gave 1.69e-2.
  const double r = std::max(range, 1e-9);
  const double rho = (M_PI + std::sqrt(M_PI * M_PI + r * r)) / r;
  const double want = 16.0 * std::log(2.0) / std::log(rho);
  // The evaluator's tree is a power of two deep, so only 2^k - 1 is free.
  for (int d : {15, 31, 63}) {
    if (d >= want) return d;
  }
  return 63;
}

template <typename word>
int CiLlamaLayer<word>::NormDegree(double window) const {
  if (cfg_.rms_degree > 0) return cfg_.rms_degree;
  // A Chebyshev fit's error is uniform over its interval, so the degree has to
  // follow the window rather than be typed beside it -- and for THIS function
  // the relation is closed form. `1/sqrt(a v + b)` on [-1, 1] with
  // `lo = 1/sqrt(W)`, `hi = sqrt(W)` has its singularity at `v = -b/a`, so the
  // Bernstein parameter is
  //
  //     rho = b/a + sqrt((b/a)^2 - 1) = (sqrt(W) + 1) / (sqrt(W) - 1)
  //
  // and the error falls as `rho^-degree`. Sixteen bits is the target because
  // that is where 1.5cv measured SiLU's CIRCUIT, and there is no point fitting
  // below the circuit that evaluates the fit. The three-way table this
  // replaces was right at the bottom of its range and two degrees short at the
  // top: it returned 15 at a window of 12, where `rho` is 1.81 and 15 buys
  // only 12.5 bits.
  //
  // AND IT IS CAPPED AT 15, WHICH IS A LEVEL BUDGET AND NOT A FIT. The
  // polynomial lands at `poly_level - ceil(log2(degree+1))` and the weight
  // multiply one below that with NO rescale, so degree 15 leaves the norm's
  // output at `stc_level + 1` and degree 31 at `stc_level` -- where
  // `SylphSchedule::ToCoeff` refuses outright, because the pending rescale it
  // has to settle needs a level and there is none. At the widest window the
  // real 32 layers reach (12.7, layer 1) degree 15 still buys 12.5 bits,
  // which is below this operator's own circuit, so the cap costs nothing
  // measurable here -- but it is the reason a window wider than ~13 would
  // need the schedule changed, not just the degree.
  const double w = std::max(window, 1.0 + 1e-9);
  const double rho = (std::sqrt(w) + 1.0) / (std::sqrt(w) - 1.0);
  const double want = 16.0 * std::log(2.0) / std::log(rho);
  for (int d : {9, 15}) {
    if (d >= want) return d;
  }
  return 15;
}

template <typename word>
std::vector<std::vector<Complex>> CiLlamaLayer<word>::NormWeights(
    const std::vector<double> &gain, double alpha) const {
  const int rank = cfg_.proj_rank;
  const int log_rank = Log2Ceil(rank);
  const int log_t = Log2Ceil(cfg_.num_tokens);
  (void)log_t;
  const double root_alpha = std::sqrt(alpha);

  std::vector<std::vector<Complex>> wts(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    wts[k].assign(num_slots_, Complex(0.0, 0.0));
    for (int c = 0; c < rank; c++) {
      const int I = Rev(c, log_rank);
      // AT AN ODD DECLARED CHANNEL THE WEIGHT IS THE PARTNER'S. The duplicate
      // band at channel `c` (odd) holds component `rank - I`, whose live
      // address is `Rev(rank - I)`; the reduction at `channel_stride = 2` sums
      // the two parities apart, so it must see the same gain at both. The
      // module basis has no band: every channel carries its own gain.
      const int src =
          (cfg_.module_basis || c % 2 == 0) ? c : Rev(rank - I, log_rank);
      const double v = gain[static_cast<size_t>(k) * rank + src];
      for (int t = 0; t < cfg_.num_tokens; t++) {
        wts[k][static_cast<size_t>(c) * cfg_.num_tokens + t] =
            Complex(v * root_alpha, 0.0);
      }
    }
  }
  return wts;
}

template <typename word>
Plaintext<word> CiLlamaLayer<word>::CrossingPlaintext(
    double factor, const std::vector<double> &sink, double at_scale) const {
  AssertTrue(static_cast<int>(sink.size()) == cfg_.num_tokens,
             "CiLlamaLayer: the sink rescale needs one factor per token");
  // The stream's slot address is `channel * num_tokens + rev(token)` (1.5du),
  // so a per-token factor is a stride-`num_tokens` pattern -- but it is NOT
  // the same pattern on both bands. The banded convention is
  // `rec[p*rank + I] = comp_I[p] + [I!=0] comp_{rank-I}[p+1]`, so a DEAD
  // component (declared channel odd, the duplicate half) at position `p`
  // carries its partner's value at position `p + 1`: a token's duplicate sits
  // one position BACK from its live copy. Applied uniformly, this multiply
  // scales token `t`'s duplicate by `sink[t-1]` -- which leaves the live band
  // right and corrupts the duplicate of the first user token by the last
  // sink's factor, measured as live 2^-5.99 against duplicate 2^-1.06 and a
  // layer at relative 265. Token 0 has no duplicate (position -1 is off the
  // image), exactly as the last position has no partner.
  const int log_t = Log2Ceil(cfg_.num_tokens);
  std::vector<Complex> vals(num_slots_, Complex(factor, 0.0));
  // On the module basis there is no duplicate band: a token's factor sits at
  // its own slot address in every channel, and that is all.
  const int live_step = cfg_.module_basis ? 1 : 2;
  for (int t = 0; t < cfg_.num_tokens; t++) {
    if (sink[t] == 1.0) continue;
    const int p = static_cast<int>(BitReverseInt(t, log_t));
    for (int c = 0; p + c * cfg_.num_tokens < num_slots_; c += live_step) {
      vals[p + c * cfg_.num_tokens] = Complex(factor * sink[t], 0.0);
    }
    if (t == 0 || cfg_.module_basis) continue;
    const int q = static_cast<int>(BitReverseInt(t - 1, log_t));
    for (int c = 1; q + c * cfg_.num_tokens < num_slots_; c += 2) {
      vals[q + c * cfg_.num_tokens] = Complex(factor * sink[t], 0.0);
    }
  }
  Plaintext<word> pt;
  boot_->encoder_.Encode(pt, slot_level_,
                         boot_->param_.GetScale(op_level_) *
                             boot_->param_.GetRescalePrimeProd(slot_level_) /
                             at_scale,
                         vals);
  return pt;
}

template <typename word>
void CiLlamaLayer<word>::Canonicalise(Ct &ct, const Plaintext<word> &pt) const {
  boot_->Mult(ct, ct, pt);
  boot_->Rescale(ct, ct);
}

template <typename word>
void CiLlamaLayer<word>::Canonicalise(Ct &ct, double factor) const {
  Constant<word> k;
  boot_->encoder_.EncodeConstant(
      k, slot_level_,
      boot_->param_.GetScale(op_level_) *
          boot_->param_.GetRescalePrimeProd(slot_level_) / ct.GetScale(),
      factor);
  boot_->Mult(ct, ct, k);
  boot_->Rescale(ct, ct);
}

template <typename word>
void CiLlamaLayer<word>::NormTurn(std::vector<Ct> &res,
                                  const std::vector<Ct> &stream,
                                  const std::vector<double> &gain,
                                  double alpha, double window,
                                  double stream_scale,
                                  const std::vector<double> &sink,
                                  const EvkMap<word> &evk, bool ffn) {
  NvtxScope _nv("layer: NormTurn");
  AssertTrue(static_cast<int>(stream.size()) == num_model_cts_,
             "CiLlamaLayer: the residual stream is " +
                 std::to_string(num_model_cts_) + " ciphertexts");
  // THE OPERATOR'S INPUT RIDES AT THE MODEL'S OWN MAGNITUDE, AND THAT IS
  // WHERE ITS PRECISION GOES. RMSNorm is scale invariant, so feeding it
  // `beta * x` with `alpha / beta^2` in place of `alpha` and `beta^2 * eps` in
  // place of `eps` computes exactly the same function -- `RmsNormTest` has
  // said so since it was written ("beta = sqrt(alpha) puts |x| near one") and
  // the layer never adopted it. What changes is not the answer but the
  // magnitude every noisy step carries it at: a ciphertext's added error is
  // ABSOLUTE, so a message riding at the model's own 0.0076 rms takes the
  // square, the eight-rotation reduction and the closing multiply at
  // `1 / alpha` of the precision the same circuit gets at rms one.
  // `beta = sqrt(alpha)` is exactly the constant that puts it there, and it
  // folds into the crossing's own multiply -- no level, no operation, no key.
  const double beta = std::sqrt(alpha);
  std::vector<Ct> slots;
  {
    // The crossings as ONE group: the per-ciphertext CtS then a single
    // batched EvalMod (Doing.md 3.23's lever).
    std::vector<const Ct *> xs(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      xs[k] = &stream[k];
    }
    sched_.ToSlotBatch(slots, xs, evk, cfg_.min_ks);
  }
  for (int k = 0; k < num_model_cts_; k++) {
    // What is divided out here is the CROSSING ALONE. A residual carries the
    // O projection's own factor at both ends -- the stream was encrypted with
    // it and the O output already has it -- so a fit measured on this
    // ciphertext would be right here and wrong at the gate's crossing, which
    // carries no such factor (1.5cu). And RMSNorm is scale invariant, which is
    // exactly what hid that mistake for a whole increment.
    // THE STREAM FACTOR GOES OUT HERE, WITH THE CROSSING. The residual
    // carries a global factor so that its crossing rides at `Config::ride`,
    // and `RmsNormHandler` wants `alpha * mean(x^2)` near 1 for the `x` it is
    // handed -- so either this multiply takes the factor out and the
    // calibration is the MODEL's, or it does not and every alpha and epsilon
    // downstream has to carry `stream_scale^2`. The first is what the
    // full-width FFN test does (`1 / (boundary * beta)`) and it is far less
    // to get wrong: measured, leaving the factor in put the invsqrt's
    // argument at 0.0038 where its window is [0.77, 1.3], and outside its
    // interval a Chebyshev fit does whatever it likes -- relative 1.1 at the
    // norm, with a fitted factor that moved between identical runs.
    //
    // Everything downstream of RMSNorm is then in MODEL units, because
    // RMSNorm is scale invariant. The two projections that write the stream
    // back -- O and down -- put the factor on again through their weights.
    // THE SINK RESCALE RIDES THIS MULTIPLY. [SYLPH] 3.1.1's prefix is
    // prompt-independent and so public, which is what makes a per-token
    // factor legal here; and it has to happen at EVERY norm, because the
    // stream's sink rows do not stay in range on their own -- layer 1's
    // output carries them at 74327x the user rows' mean square. Folding the
    // factors into the constant this crossing already pays costs no level and
    // no operation. The plaintext is built once, off the first ciphertext's
    // scale, because `ToSlot` leaves all of them at the same one.
    if (sink.empty()) {
      Canonicalise(slots[k], beta / (crossing_ * stream_scale));
    } else {
      if (k == 0) {
        crossing_pt_ = CrossingPlaintext(beta / (crossing_ * stream_scale),
                                         sink, slots[0].GetScale());
      }
      Canonicalise(slots[k], crossing_pt_);
    }
  }

  // RMSNorm DIVIDES BY THE WIDTH IT IS TOLD, and that is the DECLARED one.
  // Llama divides by `model_live`, so the two scalings below cancel exactly:
  // `eps * live / declared` makes the bracket `(live/declared)(S/live + eps)`
  // and `alpha * declared / live` puts its geometric mean back at 1. The
  // leftover `sqrt(declared/live)` is taken out by the weight, which already
  // carries `sqrt(alpha)`. Left alone this is not a scale error a fit absorbs:
  // it puts the polynomial's argument at 0.47 instead of 1 and the bottom
  // sixth of the data outside the fitted window (1.5dd, measured 2^-5.09).
  const double ratio =
      static_cast<double>(cfg_.model_declared) / cfg_.model_live;
  // The handler compiles its polynomial here and encodes its weight
  // plaintexts in `Prepare`; both are per-layer preparation, so they are
  // timed apart from the arithmetic below. `Apply` would do the encode on its
  // own at first use, which is what hid it inside the online row.
  // An event pair rather than a host clock behind `cudaDeviceSynchronize`:
  // the encode is on the device now and the drain cost the arithmetic
  // behind it an idle card.
  prepare_timer_.Begin();
  // `beta` above scaled the input, so the bracket has to be told: the handler
  // sees `S = beta^2 * sum(x^2)`, and `u = L * (S/n + e)` is the same `u` as
  // before exactly when `L = alpha * ratio / beta^2` and `e = beta^2 * eps /
  // ratio`. The weight then carries `sqrt(L)` by the handler's own contract,
  // which at `beta = sqrt(alpha)` is `sqrt(ratio)` -- the declared-width
  // leftover the comment above names, and nothing else.
  const double b2 = beta * beta;
  // The handler `PrepareNormAhead` built for exactly these inputs, if there
  // is one (the previous layer built it in one of its windows); otherwise
  // the same handler built here, as before.
  NormAhead &ahead = norm_ahead_[ffn ? 1 : 0];
  std::unique_ptr<RmsNormHandler<word>> own;
  std::vector<std::vector<Complex>> wts;
  RmsNormHandler<word> *rms_ptr = nullptr;
  if (ahead.valid && ahead.alpha == alpha && ahead.window == window &&
      ahead.gain == gain) {
    own = std::move(ahead.rms);
    wts = std::move(ahead.wts);
    ahead.valid = false;
    rms_ptr = own.get();
  } else {
    own = std::make_unique<RmsNormHandler<word>>(
        boot_, cfg_.num_tokens, cfg_.model_declared, alpha * ratio / b2,
        op_level_, b2 * cfg_.eps / ratio, window, NormDegree(window),
        channel_stride_);
    wts = NormWeights(gain, alpha / b2);
    own->Prepare(wts);
    rms_ptr = own.get();
  }
  RmsNormHandler<word> &rms = *rms_ptr;
  AssertTrue(rms.GetNumCiphertexts() == num_model_cts_,
             "CiLlamaLayer: RmsNormHandler disagrees about the stream width");
  prepare_timer_.End();
  // The reduction on its own, before anything fitted touches it. `Apply`
  // computes the same thing internally; recomputing it here keeps the
  // measured path untouched and costs one extra reduction, which only runs
  // when the diagnostic is on.
  if (cfg_.keep_norm_slots) rms.SumOfSquares(norm_acc_, slots, evk);
  std::vector<Ct> outv;
  rms.Apply(outv, slots, wts, evk);
  if (cfg_.keep_norm_slots) {
    // `Ciphertext` is non-copyable on purpose -- it owns device buffers -- so
    // the diagnostic copy has to go through the library's own.
    norm_slots_.resize(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->Copy(norm_slots_[k], outv[k]);
    }
  }
  res.resize(num_model_cts_);
  for (int k = 0; k < num_model_cts_; k++) {
    sched_.ToCoeff(res[k], outv[k], evk, cfg_.min_ks);
  }
}

template <typename word>
void CiLlamaLayer<word>::PrepareNormAhead(bool ffn,
                                          const std::vector<double> &gain,
                                          double alpha, double window) const {
  NvtxScope _nv("prep: RMSNorm ahead");
  // The same arithmetic as `NormTurn`'s, so that its match is exact.
  const double ratio =
      static_cast<double>(cfg_.model_declared) / cfg_.model_live;
  const double beta = std::sqrt(alpha);
  const double b2 = beta * beta;
  NormAhead &ahead = norm_ahead_[ffn ? 1 : 0];
  ahead.rms = std::make_unique<RmsNormHandler<word>>(
      boot_, cfg_.num_tokens, cfg_.model_declared, alpha * ratio / b2,
      op_level_, b2 * cfg_.eps / ratio, window, NormDegree(window),
      channel_stride_);
  ahead.wts = NormWeights(gain, alpha / b2);
  ahead.rms->Prepare(ahead.wts);
  ahead.gain = gain;
  ahead.alpha = alpha;
  ahead.window = window;
  ahead.valid = true;
}

template <typename word>
std::vector<double> CiLlamaLayer<word>::PlainNormInvSqrt(
    double alpha, double window,
    const std::vector<double> &mean_square) const {
  const double ratio =
      static_cast<double>(cfg_.model_declared) / cfg_.model_live;
  // The same constants the circuit runs on, `beta` included: `u` is identical
  // either way, but a probe that does not reproduce the shipped handler is
  // measuring itself.
  const double b2 = alpha;
  RmsNormHandler<word> probe(boot_, cfg_.num_tokens, cfg_.model_declared,
                             alpha * ratio / b2, op_level_,
                             b2 * cfg_.eps / ratio, window, NormDegree(window),
                             channel_stride_);
  // The bracket the circuit evaluates: the DECLARED width and the scaled
  // epsilon, whose two corrections cancel (see `NormTurn`).
  const double root = std::sqrt(alpha);
  std::vector<double> res(mean_square.size());
  for (size_t i = 0; i < mean_square.size(); i++) {
    const double u =
        (alpha * ratio / b2) *
        (b2 * mean_square[i] * cfg_.model_live / cfg_.model_declared +
         b2 * cfg_.eps / ratio);
    res[i] = root * probe.PlainInvSqrt(u);
  }
  return res;
}

template <typename word>
void CiLlamaLayer<word>::Project(std::vector<Ct> &res,
                                 const std::vector<Ct> &x, int in_declared,
                                 int out_declared,
                                 const std::vector<double> &w, double w_scale,
                                 const char *tag) const {
  leg_->SetDensity(GetDensity(), GetDensity());
  leg_->Project(res, x, in_declared, out_declared, w, w_scale, tag);
}

template <typename word>
void CiLlamaLayer<word>::Project(std::vector<Ct> &res,
                                 const std::vector<Ct> &x, int in_declared,
                                 int out_declared, const DeviceWeights &w,
                                 double w_scale, const char *tag) const {
  leg_->SetDensity(GetDensity(), GetDensity());
  leg_->Project(res, x, in_declared, out_declared, w, w_scale, tag);
}

template <typename word>
void CiLlamaLayer<word>::Project(std::vector<Ct> &res,
                                 const std::vector<Ct> &x, int in_declared,
                                 int out_declared, const ProjectionWeight &w,
                                 double w_scale, const char *tag) const {
  Project(res, x, in_declared, out_declared, w, w_scale, tag, GetDensity(),
          GetDensity());
}

template <typename word>
void CiLlamaLayer<word>::Project(std::vector<Ct> &res,
                                 const std::vector<Ct> &x, int in_declared,
                                 int out_declared, const ProjectionWeight &w,
                                 double w_scale, const char *tag,
                                 int in_density, int out_density) const {
  AssertTrue(w.Given(), std::string("CiLlamaLayer::Project(") + tag +
                            "): the weight must be given in exactly one form");
  leg_->SetDensity(in_density, out_density);
  if (w.device != nullptr) {
    leg_->Project(res, x, in_declared, out_declared, *w.device, w_scale, tag);
  } else {
    leg_->Project(res, x, in_declared, out_declared, *w.host, w_scale, tag);
  }
}

template <typename word>
void CiLlamaLayer<word>::AttentionNorm(std::vector<Ct> &res,
                                       const std::vector<Ct> &stream,
                                       const std::vector<double> &gain,
                                       const Calibration &c,
                                       const EvkMap<word> &evk) {
  NvtxScope _nv("layer: AttentionNorm");
  NormTurn(res, stream, gain, c.attn_alpha, c.attn_norm_window,
           c.stream_scale, c.attn_sink, evk, /*ffn=*/false);
}

template <typename word>
void CiLlamaLayer<word>::FeedForward(std::vector<Ct> &res,
                                     const std::vector<Ct> &h_cts,
                                     const std::vector<Ct> &stream,
                                     const Weights &w, const Calibration &c,
                                     const EvkMap<word> &evk) {
  NvtxScope _nv("layer: FeedForward");
  AssertTrue(w.o.Given() && w.gate.Given() && w.up.Given() &&
                 w.down.Given() && w.ffn_norm != nullptr,
             "CiLlamaLayer: every weight must be given, in exactly one form");
  AssertTrue(!w.tag.empty(),
             "CiLlamaLayer: a layer's weights need a tag -- the projection "
             "leg caches by name and a repeated name with different weights "
             "is a wrong layer that still decrypts");
  AssertTrue(static_cast<int>(h_cts.size()) * cfg_.proj_rank == attn_channels_,
             "CiLlamaLayer: the seam did not hand over the whole attention "
             "output");
  AssertTrue(static_cast<int>(stream.size()) == num_model_cts_,
             "CiLlamaLayer: the residual stream is " +
                 std::to_string(num_model_cts_) + " ciphertexts");

  const auto &p = boot_->param_;

  // ---- the O projection, then the residual -------------------------------
  std::vector<Ct> h_ct(num_model_cts_);
  {
    std::vector<Ct> ins(h_cts.size());
    for (size_t k = 0; k < h_cts.size(); k++) {
      boot_->LevelDown(ins[k], h_cts[k], cfg_.product_level);
    }
    std::vector<Ct> out;
    // The seam's images follow the layer's basis (dense on the module basis,
    // banded otherwise), and so does O's input density.
    Project(out, ins, attn_channels_, cfg_.model_declared, w.o, c.res_scale,
            (w.tag + ".o").c_str(), GetDensity(), GetDensity());
    AssertTrue(static_cast<int>(out.size()) == num_model_cts_,
               "CiLlamaLayer: the O projection did not land in " +
                   std::to_string(num_model_cts_) + " ciphertexts");
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->Add(h_ct[k], stream[k], out[k]);
    }
  }
  MemoryPool::Report("ffn: after the O projection and the residual");

  // ---- the crossing, RMSNorm, and back to coefficients --------------------
  std::vector<Ct> normed;
  NormTurn(normed, h_ct, *w.ffn_norm, c.alpha, c.norm_window,
           c.stream_scale, c.ffn_sink, evk, /*ffn=*/true);
  MemoryPool::Report("ffn: after the norm turn (crossings, RMSNorm, StC)");

  // ---- gate and up -------------------------------------------------------
  std::vector<Ct> gate, upv;
  {
    std::vector<Ct> ins(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->LevelDown(ins[k], normed[k], cfg_.product_level);
    }
    Project(gate, ins, cfg_.model_declared, cfg_.hidden_declared, w.gate,
            c.gate_scale, (w.tag + ".gate").c_str());
    Project(upv, ins, cfg_.model_declared, cfg_.hidden_declared, w.up,
            c.gate_scale, (w.tag + ".up").c_str());
  }
  AssertTrue(static_cast<int>(gate.size()) == num_hidden_cts_ &&
                 static_cast<int>(upv.size()) == num_hidden_cts_,
             "CiLlamaLayer: the gate and up projections did not land in " +
                 std::to_string(num_hidden_cts_) + " ciphertexts");
  MemoryPool::Report("ffn: after gate and up (2 x 28 coefficient images)");

  // ---- SiLU and the gate multiply ----------------------------------------
  std::vector<Ct> prod(num_hidden_cts_);
  {
    SiLuHandler<word> silu(boot_, c.silu_range, op_level_,
                           SiLuDegree(c.silu_range));
    // The 2 x 28 crossings as groups: the per-ciphertext CtS then batched
    // EvalMods, instead of 56 serial launch-bound reductions.
    std::vector<Ct> g_ups, u_ups;
    {
      std::vector<const Ct *> xs(num_hidden_cts_);
      for (int i = 0; i < num_hidden_cts_; i++) xs[i] = &gate[i];
      sched_.ToSlotBatch(g_ups, xs, evk, cfg_.min_ks);
      gate.clear();
      for (int i = 0; i < num_hidden_cts_; i++) xs[i] = &upv[i];
      sched_.ToSlotBatch(u_ups, xs, evk, cfg_.min_ks);
      upv.clear();
    }
    for (int i = 0; i < num_hidden_cts_; i++) {
      Ct sv, u_low;
      Ct &g_up = g_ups[i];
      Ct &u_up = u_ups[i];
      // `crossing_`, NOT a fit taken on the residual: these carry no O factor.
      // And `kappa_` beside it, which is the same mistake one turn further out
      // -- see the class comment.
      Canonicalise(g_up,
                   1.0 / (kappa_ * crossing_ * c.gate_scale * c.silu_range));
      if (c.up_sink.empty()) {
        Canonicalise(u_up, 1.0 / (kappa_ * crossing_ * c.gate_scale));
      } else {
        if (i == 0) {
          up_pt_ = CrossingPlaintext(1.0 / (kappa_ * crossing_ * c.gate_scale),
                                     c.up_sink, u_up.GetScale());
        }
        Canonicalise(u_up, up_pt_);
      }
      silu.Apply(sv, g_up, evk);
      boot_->LevelDown(u_low, u_up, p.NPToLevel(sv.GetNP()));
      boot_->HMult(prod[i], sv, u_low, evk.GetMultiplicationKey());
      g_ups[i] = Ct{};
      u_ups[i] = Ct{};
    }
  }
  MemoryPool::Report("ffn: after SiLU and the gate multiply (28 products)");

  // ---- the down projection ------------------------------------------------
  {
    std::vector<Ct> ins(num_hidden_cts_);
    for (int i = 0; i < num_hidden_cts_; i++) {
      Ct c2;
      sched_.ToCoeff(c2, prod[i], evk, cfg_.min_ks);
      boot_->LevelDown(ins[i], c2, cfg_.product_level);
    }
    std::vector<Ct> y;
    // `stream_scale`, not 1: RMSNorm is scale invariant, so `y` comes back in
    // the model's own units while `h_ct` carries the stream's factor, and the
    // two cannot be added until they agree. The weight is a plaintext, so
    // putting it back costs nothing.
    Project(y, ins, cfg_.hidden_declared, cfg_.model_declared, w.down,
            c.stream_scale, (w.tag + ".down").c_str());
    AssertTrue(static_cast<int>(y.size()) == num_model_cts_,
               "CiLlamaLayer: the down projection did not land in " +
                   std::to_string(num_model_cts_) + " ciphertexts");
    // THE SECOND RESIDUAL. The correctness-width layer test stopped at the
    // down projection and compared against the down projection, so it never
    // needed this; a layer that feeds its successor does.
    res.resize(num_model_cts_);
    for (int k = 0; k < num_model_cts_; k++) {
      boot_->Add(res[k], h_ct[k], y[k]);
    }
  }
  MemoryPool::Report("ffn: after the down projection and the residual");
}

template class CiProjectionLeg<uint32_t>;
template class CiProjectionLeg<uint64_t>;
template class CiLlamaLayer<uint32_t>;
template class CiLlamaLayer<uint64_t>;

}  // namespace cheddar
