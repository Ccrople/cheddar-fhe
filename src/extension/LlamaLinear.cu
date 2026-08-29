#include "extension/LlamaLinear.h"

#include "extension/Profile.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

namespace {

// Both switches default on and exist to be turned off: the cache is what makes
// the layer's measured pass free of model conversion, and the comparison that
// establishes what it bought needs the old path in the same binary.
bool EnvOn(const char *name, bool fallback) {
  const char *v = std::getenv(name);
  if (v == nullptr) return fallback;
  return !(v[0] == '0' && v[1] == '\0');
}

}  // namespace

template <typename word>
CoeffLinearLeg<word>::CoeffLinearLeg(
    ConstContextPtr<word> context, const Config &cfg,
    std::vector<const EvaluationKey<word> *> modpack_keys, Descent descent)
    : context_{context},
      cfg_{cfg},
      descent_{std::move(descent)},
      small_degree_{SmallDegreeFor(cfg.num_tokens, context->param_)},
      rank_{context->param_.degree_ /
            SmallDegreeFor(cfg.num_tokens, context->param_)},
      ring_rank_{1},
      sub_rank_{context->param_.degree_ /
                SmallDegreeFor(cfg.num_tokens, context->param_)},
      conjugate_invariant_{context->param_.conjugate_invariant_},
      modpack_keys_{std::move(modpack_keys)},
      mlwe_{context->param_, context->ntt_handler_},
      pcmm_{context->param_},
      product_param_{&context->param_},
      product_mlwe_{&mlwe_},
      product_pcmm_{&pcmm_},
      product_context_{context},
      pack_keys_{&modpack_keys_},
      cache_weights_{EnvOn("CHEDDAR_WEIGHT_CACHE", true)},
      use_blas_{false} {
  const int degree = context_->param_.degree_;
  const int num_slots = context_->param_.MaxNumSlots();

  // `Mlwe.cu:158` used to launch one 256-thread block per 256 positions and
  // so computed a zero-block grid below 256, which ruled out small_degree
  // 128 -- and hence T = 64 at degree 65536, and hence R+ at T = 128, where
  // small_degree IS 128. Doing.md 1.5bi relaxed all three such launches to
  // `min(length, 256)`; both quantities are powers of two, so the smaller
  // divides the larger and the grid shrinks instead of refusing, bit
  // identically at 256 and above. What is left is the power-of-two shape the
  // index arithmetic actually needs.
  AssertTrue(small_degree_ >= 128 && IsPowOfTwo(small_degree_),
             "CoeffLinearLeg: small_degree must be a power of two and at "
             "least 128");
  AssertTrue(small_degree_ < degree && degree % small_degree_ == 0,
             "CoeffLinearLeg: the small degree must properly divide the ring "
             "degree");

  // The alignment itself, stated as a check rather than trusted. The block
  // puts `num_slots / T` channels in one ciphertext and ModDecomp splits a
  // parent into `rank` module components; one channel lands in each only when
  // the two agree.
  AssertTrue(num_slots / cfg_.num_tokens == rank_,
             "CoeffLinearLeg: the block's channels-per-ciphertext (" +
                 std::to_string(num_slots / cfg_.num_tokens) +
                 ") must equal the ModDecomp rank (" + std::to_string(rank_) +
                 "), or a module component would carry more than one channel "
                 "and no plaintext matrix could separate them");

  AssertTrue(cfg_.product_level >= 1,
             "CoeffLinearLeg: the product rescales, so it cannot run at level "
             "0");
  AssertTrue(cfg_.parents_per_tile >= 0,
             "CoeffLinearLeg: parents_per_tile cannot be negative");

  // ---- [SYLPH]'s descent, when there is one ------------------------------
  //
  // Everything above is about the 256 module components and none of it moves:
  // the rank, the channel alignment and the operand shape are the same by
  // either route. What the descent changes is how those 256 are reached and
  // how they are put back, and the whole of the difference is bounded by this
  // block and by `Component`.
  if (descent_.Enabled()) {
    const Parameter<word> &sw_param = descent_.switch_context->param_;
    const Parameter<word> &sm_param = descent_.small_context->param_;
    AssertTrue(sw_param.degree_ == degree,
               "CoeffLinearLeg: the switching context must be at the block's "
               "own ring degree -- a ciphertext crosses to it unchanged");
    AssertTrue(sm_param.degree_ < degree && degree % sm_param.degree_ == 0,
               "CoeffLinearLeg: the product ring's degree must properly "
               "divide the block's");
    AssertTrue(sm_param.degree_ % small_degree_ == 0,
               "CoeffLinearLeg: the small degree must divide the product "
               "ring's degree, or ModDecomp has no rank to run at");
    AssertTrue(sm_param.conjugate_invariant_ ==
                   context_->param_.conjugate_invariant_,
               "CoeffLinearLeg: the product ring must be the same KIND of "
               "ring as the block's -- the coefficient packing differs");
    ring_rank_ = degree / sm_param.degree_;
    sub_rank_ = sm_param.degree_ / small_degree_;
    // The one identity the index permutation rests on. It is arithmetic, not
    // an assumption, but a parameter pair that broke it would produce a layer
    // that still decrypts and is wrong.
    AssertTrue(ring_rank_ * sub_rank_ == rank_,
               "CoeffLinearLeg: the two strides must compose to the rank the "
               "channel map is stated against");
    AssertTrue(descent_.forward != nullptr && descent_.inverse != nullptr,
               "CoeffLinearLeg: the descent needs both ring-switching keys");
    AssertTrue(static_cast<int>(descent_.modpack_keys.size()) == sub_rank_,
               "CoeffLinearLeg: the descent's ModPack runs on the product "
               "ring, so it needs " +
                   std::to_string(sub_rank_) +
                   " keys from that context, not the big ring's " +
                   std::to_string(rank_));
    // The operand is encoded against the product ring and the result is
    // rescaled there, but the block reads the ciphertext back on its own
    // ladder -- so `GetScale(L)^2 / GetRescalePrimeProd(L) = GetScale(L-1)`
    // has to hold across the two. It does when they share their primes at
    // these levels, which is what the pair is for; a set that did not would
    // land the product off the canonical scale and abort inside a later
    // `EvalPoly`, three layers from here and with nothing pointing back.
    AssertTrue(std::abs(sm_param.GetScale(cfg_.product_level) /
                            context_->param_.GetScale(cfg_.product_level) -
                        1.0) < 1e-9,
               "CoeffLinearLeg: the product ring's scale at the product level "
               "differs from the block's, so the rescaled result would not be "
               "canonical on the block's ladder");
    switcher_.reset(new RingSwitchHandler<word>(descent_.switch_context,
                                                descent_.small_context));
    AssertTrue(switcher_->GetRank() == ring_rank_,
               "CoeffLinearLeg: the ring switch disagrees about its own rank");
    small_mlwe_.reset(
        new MlweHandler<word>(sm_param, descent_.small_context->ntt_handler_));
    small_pcmm_.reset(new PcmmHandler<word>(sm_param));
    product_param_ = &sm_param;
    product_mlwe_ = small_mlwe_.get();
    product_pcmm_ = small_pcmm_.get();
    product_context_ = descent_.small_context;
    pack_keys_ = &descent_.modpack_keys;
  } else {
    AssertTrue(static_cast<int>(modpack_keys_.size()) == rank_,
               "CoeffLinearLeg: ModPack needs exactly one switching key per "
               "module component, i.e. " +
                   std::to_string(rank_));
  }

  // Half density is a claim about the CALLER's ciphertexts that nothing here
  // can verify, so the only thing checkable is that the shape admits it.
  AssertTrue(cfg_.input_density == 1 || cfg_.input_density == 2,
             "CoeffLinearLeg: input_density is 1 (dense) or 2 (half density)");
  AssertTrue(cfg_.output_density == 1 || cfg_.output_density == 2,
             "CoeffLinearLeg: output_density is 1 (dense) or 2 (half "
             "density)");
  if (cfg_.output_density == 2) {
    AssertTrue(conjugate_invariant_,
               "CoeffLinearLeg: a half-density OUTPUT is a "
               "conjugate-invariant contract; on the ordinary ring every "
               "output component is live");
    AssertTrue(rank_ % 2 == 0,
               "CoeffLinearLeg: a half-density output needs an even rank");
    // Under the descent a live flat index `j * sub_rank + n < rank/2` is
    // exactly `j < ring_rank/2`, so the dead half is whole PARTS: their
    // `ModPack` disappears rather than shrinking, and the inverse switch is
    // handed the zeros they stand for. That is why this needs an even part
    // count where the direct route needs only an even rank.
    AssertTrue(!descent_.Enabled() || ring_rank_ % 2 == 0,
               "CoeffLinearLeg: a half-density output needs an even number of "
               "product-ring parts");
  }
  if (cfg_.input_density == 2) {
    AssertTrue(conjugate_invariant_,
               "CoeffLinearLeg: half density is a conjugate-invariant "
               "contract -- on the ordinary ring a coefficient image has no "
               "banded partner and every module component is live");
    // The direct route has `ring_rank_ == 1` -- the parts are the components
    // -- so only the descent needs an even part count. Requiring it of both
    // refused the direct route outright, which is what the first run of
    // `CiFfn.TheFullWidthLayerRowsAreMeasured` at density 2 hit.
    AssertTrue(rank_ % 2 == 0,
               "CoeffLinearLeg: half density needs an even rank");
    AssertTrue(!descent_.Enabled() || ring_rank_ % 2 == 0,
               "CoeffLinearLeg: half density needs an even number of "
               "product-ring parts, because a live flat index is "
               "i * sub_rank + n < rank/2 exactly when i < ring_rank/2");
  }

#ifdef USE_CUBLAS
  use_blas_ = EnvOn("CHEDDAR_PCMM_BLAS", true);
  if (use_blas_) {
    blas_.reset(new PcmmBlasHandler<word>(context_->param_));
    product_blas_ = blas_.get();
    if (descent_.Enabled()) {
      // The split is encoded against the primes of the ring the MLWE
      // ciphertexts live in, so it follows the product and not the block.
      small_blas_.reset(new PcmmBlasHandler<word>(*product_param_));
      product_blas_ = small_blas_.get();
    }
  }
#endif
  std::cout << "CoeffLinearLeg: product on "
            << (use_blas_ ? "cuBLAS int8 GEMM" : "PcmmAccum")
            << ", converted weights "
            << (cache_weights_ ? "cached on the GPU" : "rebuilt every call")
            << ", descent ";
  if (descent_.Enabled()) {
    std::cout << "ring-switched " << degree << " -> "
              << product_param_->degree_ << " -> " << small_degree_
              << " (rank " << ring_rank_ << " x " << sub_rank_ << ")";
  } else {
    std::cout << "direct " << degree << " -> " << small_degree_ << " (rank "
              << rank_ << ")";
  }
  std::cout << std::endl;
}

template <typename word>
uint64_t CoeffLinearLeg<word>::Fingerprint(const std::vector<double> &w,
                                           double w_scale) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&h](uint64_t v) {
    for (int b = 0; b < 8; b++) {
      h ^= (v >> (8 * b)) & 0xffull;
      h *= 1099511628211ull;
    }
  };
  mix(static_cast<uint64_t>(w.size()));
  uint64_t scale_bits = 0;
  std::memcpy(&scale_bits, &w_scale, sizeof(scale_bits));
  mix(scale_bits);
  const size_t step = std::max<size_t>(1, w.size() / 4096);
  for (size_t i = 0; i < w.size(); i += step) {
    uint64_t bits = 0;
    std::memcpy(&bits, &w[i], sizeof(bits));
    mix(bits);
  }
  return h;
}

template <typename word>
void CoeffLinearLeg<word>::GatherWeights(std::vector<double> &values,
                                         const std::vector<double> &w,
                                         int in_channels, int out_channels,
                                         int group, double w_scale,
                                         int first_parent,
                                         int num_parents) const {
  const int log_rank = Log2Ceil(rank_);
  // The dead half of a half-density input is not a column of zeros here -- it
  // is not a column at all. See `Config::input_density`.
  const int live = LiveColumns();
  const int rows = LiveRows();
  const int cols = num_parents * live;
  values.assign(static_cast<size_t>(rows) * cols, 0.0);
  // `Component` is the identity on the direct route and the two-stride
  // reindexing on the ring-switched one; the channel map itself is the same
  // sentence either way, which is the point of putting the difference here.
  for (int r = 0; r < rows; r++) {
    const int out_channel =
        group * rank_ +
        static_cast<int>(BitReverseInt(Component(r), log_rank));
    for (int p = 0; p < num_parents; p++) {
      for (int i = 0; i < live; i++) {
        const int in_channel =
            (first_parent + p) * rank_ +
            static_cast<int>(BitReverseInt(Component(i), log_rank));
        values[static_cast<size_t>(r) * cols + p * live + i] =
            w[static_cast<size_t>(in_channel) * out_channels + out_channel] *
            w_scale;
      }
    }
  }
}

template <typename word>
void CoeffLinearLeg<word>::BuildOperands(Operands &res,
                                         const std::vector<double> &w,
                                         int in_channels, int out_channels,
                                         double w_scale, int first_parent,
                                         int num_parents, int groups) const {
  // The one scale that leaves the rescaled product canonical at the level
  // below; see the class comment.
  const int level = cfg_.product_level;
  // The ring the product runs in is the ring the operand is encoded against.
  // With the descent that is the small one, whose primes are the block's at
  // this level and whose scale ladder is therefore the same -- but taking it
  // from the wrong Parameter would be a silent scale error rather than a
  // mismatch, so it is read from the product's own.
  const double scale = product_param_->GetScale(level);
  const int cols = num_parents * LiveColumns();
  const int rows = LiveRows();
  std::vector<double> values;
  for (int g = 0; g < groups; g++) {
    GatherWeights(values, w, in_channels, out_channels, g, w_scale,
                  first_parent, num_parents);
    if (use_blas_) {
#ifdef USE_CUBLAS
      typename PcmmBlasHandler<word>::SplitMatrix s;
      product_blas_->SplitMatrixFrom(s, level, scale, values, rows, cols);
      res.bytes += PcmmBlasHandler<word>::SplitBytes(s);
      res.split.push_back(std::move(s));
#endif
    } else {
      PlainMatrix<word> u;
      product_pcmm_->EncodeMatrix(u, level, scale, values, rows, cols);
      res.bytes += static_cast<size_t>(u.data_.size()) * sizeof(word);
      res.u.push_back(std::move(u));
    }
  }
}

template <typename word>
const typename CoeffLinearLeg<word>::Operands &
CoeffLinearLeg<word>::GetOperands(const char *name,
                                  const std::vector<double> &w,
                                  int in_channels, int out_channels,
                                  double w_scale, int parents, int groups,
                                  int tile) const {
  const uint64_t fp = Fingerprint(w, w_scale);
  const std::string key(name);
  auto it = operands_.find(key);
  if (it != operands_.end()) {
    const Operands &held = it->second;
    // A different tensor under a name already converted would otherwise be
    // answered with the first one's operands and produce a wrong layer that
    // still decrypts cleanly. The check is what makes the cache safe to leave
    // on by default.
    AssertTrue(held.fingerprint == fp && held.in_channels == in_channels &&
                   held.out_channels == out_channels &&
                   held.w_scale == w_scale && held.groups == groups &&
                   held.tile == tile,
               std::string("CoeffLinearLeg: the converted weights held for \"") +
                   name +
                   "\" were built from a different tensor, scale or tiling. "
                   "Set CHEDDAR_WEIGHT_CACHE=0 if the weights really do change "
                   "between calls");
    return held;
  }

  Operands built;
  built.groups = groups;
  built.tile = tile;
  built.in_channels = in_channels;
  built.out_channels = out_channels;
  built.w_scale = w_scale;
  built.fingerprint = fp;
  for (int base = 0; base < parents; base += tile) {
    const int span = std::min(tile, parents - base);
    BuildOperands(built, w, in_channels, out_channels, w_scale, base, span,
                  groups);
    built.tiles++;
  }
  std::cout << "weight cache: converted " << name << " (" << in_channels << " x "
            << out_channels << ") into " << built.tiles << " x " << groups
            << " operands, " << built.bytes / 1048576 << " MiB on the device"
            << std::endl;
  auto ins = operands_.emplace(key, std::move(built));
  return ins.first->second;
}

template <typename word>
void CoeffLinearLeg<word>::Decompose(
    std::vector<MlweCiphertext<word>> &columns, const std::vector<Ct> &x,
    int base, int span) const {
  const int level = cfg_.product_level;
  columns.clear();
  columns.reserve(static_cast<size_t>(span) * rank_);
  for (int p = base; p < base + span; p++) {
    const Ct &parent = x[p];
    Ct lowered;
    const Ct *source = &parent;
    if (context_->param_.NPToLevel(parent.GetNP()) != level) {
      context_->LevelDown(lowered, parent, level);
      source = &lowered;
    }

    if (!descent_.Enabled()) {
      // The direct route: one stride, rank 256, and the components are as
      // large as the parent's own a-part.
      std::vector<MlweCiphertext<word>> decomposed;
      mlwe_.ModDecomp(decomposed, *source, small_degree_);
      AssertTrue(static_cast<int>(decomposed.size()) == rank_,
                 "CoeffLinearLeg: ModDecomp did not yield one module component "
                 "per input channel");
      // At half density the tail is identically zero and is dropped rather
      // than contracted over; the scan produces it either way, but nothing
      // downstream holds it or multiplies by it.
      const int live = LiveColumns();
      for (int c = 0; c < live; c++) columns.push_back(std::move(decomposed[c]));
      continue;
    }

    // [SYLPH]'s route. One key switch at the block's degree fans the parent
    // out into `ring_rank_` ciphertexts of the product ring, and each of those
    // decomposes at rank `sub_rank_` instead of `rank_`. The components come
    // out in `Component`'s order because the loops are nested that way.
    std::vector<Ct> parts;
    {
      NvtxScope _n("pcmm: RingSwitch");
      switcher_->Switch(parts, *source, *descent_.forward);
    }
    AssertTrue(static_cast<int>(parts.size()) == ring_rank_,
               "CoeffLinearLeg: the ring switch returned the wrong number of "
               "parts");
    NvtxScope _n("pcmm: ModDecomp");
    // A live flat index is `i * sub_rank_ + n < rank_ / input_density`, so at
    // half density the live components are exactly the FIRST HALF OF THE
    // PARTS: their `ModDecomp` is skipped outright rather than done and
    // discarded, and the part is freed unread.
    const int live_parts = ring_rank_ / cfg_.input_density;
    for (int i = 0; i < ring_rank_; i++) {
      if (i >= live_parts) {
        parts[i] = Ct();
        continue;
      }
      std::vector<MlweCiphertext<word>> decomposed;
      small_mlwe_->ModDecomp(decomposed, parts[i], small_degree_);
      AssertTrue(static_cast<int>(decomposed.size()) == sub_rank_,
                 "CoeffLinearLeg: ModDecomp on the product ring did not yield "
                 "sub_rank components");
      for (auto &c : decomposed) columns.push_back(std::move(c));
      // Released as we go: the part has been consumed and the decomposition
      // of the next one is about to be allocated.
      parts[i] = Ct();
    }
  }
  AssertTrue(static_cast<int>(columns.size()) == span * LiveColumns(),
             "CoeffLinearLeg: the descent did not yield one module component "
             "per live input channel of the tile");
}

template <typename word>
void CoeffLinearLeg<word>::Project(std::vector<Ct> &res,
                                   const std::vector<Ct> &x, int in_channels,
                                   int out_channels,
                                   const std::vector<double> &w, double w_scale,
                                   const char *name) const {
  RunProjection(res, x, in_channels, out_channels, w, w_scale, name, false);
}

template <typename word>
void CoeffLinearLeg<word>::ProjectMerged(
    std::vector<Ct> &res, const std::vector<Ct> &x, int in_channels,
    int out_channels, const std::vector<double> &w, double w_scale,
    const char *name, const Context<word> &) const {
  RunProjection(res, x, in_channels, out_channels, w, w_scale, name, true);
}

template <typename word>
void CoeffLinearLeg<word>::RunProjection(
    std::vector<Ct> &res, const std::vector<Ct> &x, int in_channels,
    int out_channels, const std::vector<double> &w, double w_scale,
    const char *name, bool merge) const {
  AssertTrue(static_cast<int>(x.size()) * rank_ == in_channels,
             std::string("CoeffLinearLeg::Project(") + name +
                 "): the input ciphertext count times the rank must be the "
                 "inner dimension");
  AssertTrue(out_channels % rank_ == 0,
             std::string("CoeffLinearLeg::Project(") + name +
                 "): the output width must be a whole number of ModPack "
                 "groups");
  AssertTrue(w.size() == static_cast<size_t>(in_channels) * out_channels,
             std::string("CoeffLinearLeg::Project(") + name +
                 "): the weight matrix is not [in_channels][out_channels]");

  const int parents = static_cast<int>(x.size());
  const int groups = out_channels / rank_;
  AssertTrue(!merge || groups % 2 == 0,
             std::string("CoeffLinearLeg::ProjectMerged(") + name +
                 "): an odd number of output ciphertexts has no pairing");
  // WHAT MERGING BUYS, AND WHERE IT HAS TO HAPPEN. `ModPack` is `rank` key
  // switches per output ciphertext -- 81% of the block's seven projections
  // against 6% for the product itself -- so putting two outputs in one
  // ciphertext before the pack halves the pack, while doing it after would buy
  // nothing. The product runs twice either way; it is the cheap half.
  const int out_groups = merge ? groups / 2 : groups;
  const int tile = (cfg_.parents_per_tile > 0 && cfg_.parents_per_tile < parents)
                       ? cfg_.parents_per_tile
                       : parents;

  // Model conversion, once. Without the cache the operands are rebuilt tile by
  // tile inside the loop, which is what the layer used to do on every call.
  const Operands *cached = nullptr;
  if (cache_weights_) {
    NvtxScope _n("pcmm: convert weights (first call only)");
    cached = &GetOperands(name, w, in_channels, out_channels, w_scale, parents,
                          groups, tile);
  }

  // The partial sums, one per output group and -- with the descent -- one per
  // ciphertext of the product ring within it. They are ordinary RLWE
  // ciphertexts because they have already been ModPacked; there is no MLWE
  // addition to accumulate with instead. Accumulating here rather than after
  // the return trip is what makes a tile cost one ModPack and not one inverse
  // ring switch as well.
  std::vector<std::vector<Ct>> partial(out_groups);
  for (auto &v : partial) v.resize(ring_rank_);
  bool started = false;
  int tile_index = 0;

  for (int base = 0; base < parents; base += tile, tile_index++) {
    const int span = std::min(tile, parents - base);

    // 1. Descend to the product level and split the channel axis onto the
    //    ciphertext axis. Direct, that is ModDecomp alone and costs nothing
    //    but memory -- `rank` module components per parent, each the size of
    //    the parent's own a-part, which is what the tile bounds. Through the
    //    ring switch it is one key switch per parent and a sixteenth of the
    //    memory; see `Decompose`.
    std::vector<MlweCiphertext<word>> columns;
    Decompose(columns, x, base, span);

    // Uncached: this tile's operands, discarded when the tile is done.
    Operands scratch;
    if (cached == nullptr) {
      NvtxScope _n("pcmm: EncodeWeights");
      BuildOperands(scratch, w, in_channels, out_channels, w_scale, base, span,
                    groups);
    }
    const Operands &ops = cached != nullptr ? *cached : scratch;
    const int first = cached != nullptr ? tile_index * groups : 0;

#ifdef USE_CUBLAS
    // 1b. Put the decomposition into the form the GEMM consumes, once for the
    //     tile. Which ModPack group is being computed does not enter the
    //     split, so doing it inside the product did it `groups` times over --
    //     fifty-six for gate and up. It replaces `columns` rather than joining
    //     it: four 7-bit pieces of a 32-bit word are the same four bytes, so
    //     the tile's footprint does not change.
    typename PcmmBlasHandler<word>::SplitSource prepared;
    if (use_blas_) {
      NvtxScope _n("pcmm: split source (once per tile)");
      product_blas_->PrepareSource(prepared, ops.split[first], columns);
      columns.clear();
    }
#endif

    // 2. One ModPack group of output channels at a time. The decomposition is
    //    shared by every group, which is why it is hoisted out of this loop --
    //    it is the expensive part, and the product itself is two plaintext
    //    matrix products with no key material at all.
    for (int g = 0; g < out_groups; g++) {
      const int lo_group = merge ? 2 * g : g;
      auto mix = [&](std::vector<MlweCiphertext<word>> &out, int which) {
        NvtxScope _n("pcmm: Multiply");
        if (use_blas_) {
#ifdef USE_CUBLAS
          product_blas_->Multiply(out, ops.split[first + which], prepared);
#endif
        } else {
          product_pcmm_->Multiply(out, ops.u[first + which], columns);
        }
      };
      std::vector<MlweCiphertext<word>> product;
      mix(product, lo_group);
      if (merge) {
        // The second output's payload into the coefficients the first leaves
        // empty. `AddShiftedHalf` is `Y^(N'/2)` on every module component,
        // which packing turns into `X^(N/2)` on the big ring -- exactly the
        // merge `HalfBootSplit` expects, arrived at one pack earlier.
        NvtxScope _n("pcmm: MergeHalves");
        std::vector<MlweCiphertext<word>> upper;
        mix(upper, lo_group + 1);
        // In place, because otherwise this allocates two fresh device buffers
        // per module component and a projection merges `rank` of them per
        // output group -- 7168 allocations for gate alone.
        for (size_t j = 0; j < product.size(); j++) {
          product_mlwe_->AddShiftedHalf(product[j], product[j], upper[j]);
        }
      }

      // ModPack takes the components of ONE ciphertext of the product ring,
      // so the `rank_` rows come back out in `ring_rank_` runs of
      // `sub_rank_`. Without the descent there is one run and it is all of
      // them, which is the old code exactly.
      NvtxScope _n("pcmm: ModPack");
      // At half density the product produced `rank_/2` components; whether
      // that shrinks each pack or deletes half of them is the one thing the
      // two routes read differently, and `LivePacks`/`PackSources` is that
      // sentence.
      const int packs = LivePacks();
      const int sources = PackSources();
      for (int i = 0; i < packs; i++) {
        std::vector<MlweCiphertext<word>> component;
        component.reserve(sources);
        for (int n = 0; n < sources; n++) {
          component.push_back(std::move(product[i * sources + n]));
        }
        Ct repacked;
        product_mlwe_->ModPack(product_context_, repacked, component,
                               *pack_keys_);
        if (!started) {
          partial[g][i] = std::move(repacked);
        } else {
          Ct sum;
          product_context_->Add(sum, partial[g][i], repacked);
          partial[g][i] = std::move(sum);
        }
      }
    }
    started = true;
  }

  // 3. One rescale, after the whole contraction. The product carries scale
  //    GetScale(level)^2; this brings it to GetScale(level - 1), and it is the
  //    single level the product spends however many tiles it took. It happens
  //    on the product ring, which with the descent is where the ciphertext
  //    still is -- a rescale there is a sixteenth of the work, and the inverse
  //    switch then carries a level-0 ciphertext home.
  res.resize(out_groups);
  {
    NvtxScope _n("pcmm: Rescale");
    for (int g = 0; g < out_groups; g++) {
      if (!descent_.Enabled()) {
        product_context_->Rescale(res[g], partial[g][0]);
        continue;
      }
      // The inverse switch composes all `ring_rank_` parts, so a half-density
      // output has to hand it the zeros its dead parts stand for. They are
      // built by subtracting a live part from itself rather than default
      // constructed: the composition needs a ciphertext at the right level and
      // prime set, and an empty one has neither.
      std::vector<Ct> rescaled(ring_rank_);
      for (int i = 0; i < LivePacks(); i++) {
        product_context_->Rescale(rescaled[i], partial[g][i]);
      }
      for (int i = LivePacks(); i < ring_rank_; i++) {
        product_context_->Sub(rescaled[i], rescaled[0], rescaled[0]);
      }
      NvtxScope _n2("pcmm: SwitchBack");
      switcher_->SwitchBack(res[g], rescaled, *descent_.inverse);
    }
  }
}

template class CoeffLinearLeg<uint32_t>;
template class CoeffLinearLeg<uint64_t>;

}  // namespace cheddar
