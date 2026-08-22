#include "extension/LlamaLinear.h"

#include "extension/Profile.h"

#include <algorithm>
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
    std::vector<const EvaluationKey<word> *> modpack_keys)
    : context_{context},
      cfg_{cfg},
      small_degree_{SmallDegreeFor(cfg.num_tokens)},
      rank_{context->param_.degree_ / SmallDegreeFor(cfg.num_tokens)},
      modpack_keys_{std::move(modpack_keys)},
      mlwe_{context->param_, context->ntt_handler_},
      pcmm_{context->param_},
      cache_weights_{EnvOn("CHEDDAR_WEIGHT_CACHE", true)},
      use_blas_{false} {
  const int degree = context_->param_.degree_;
  const int num_slots = degree / 2;

  // `Mlwe.cu:158`. This is what rules out T = 64 at degree 65536: it would
  // want rank 512, hence small_degree 128, which the kernel cannot launch.
  AssertTrue(small_degree_ % 256 == 0,
             "CoeffLinearLeg: ModDecomp needs small_degree (= 2 * num_tokens) "
             "to be a multiple of 256, so num_tokens must be at least 128");
  AssertTrue(small_degree_ < degree && degree % small_degree_ == 0,
             "CoeffLinearLeg: 2 * num_tokens must properly divide the ring "
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

  AssertTrue(static_cast<int>(modpack_keys_.size()) == rank_,
             "CoeffLinearLeg: ModPack needs exactly one switching key per "
             "module component, i.e. " +
                 std::to_string(rank_));
  AssertTrue(cfg_.product_level >= 1,
             "CoeffLinearLeg: the product rescales, so it cannot run at level "
             "0");
  AssertTrue(cfg_.parents_per_tile >= 0,
             "CoeffLinearLeg: parents_per_tile cannot be negative");

#ifdef USE_CUBLAS
  use_blas_ = EnvOn("CHEDDAR_PCMM_BLAS", true);
  if (use_blas_) blas_.reset(new PcmmBlasHandler<word>(context_->param_));
#endif
  std::cout << "CoeffLinearLeg: product on "
            << (use_blas_ ? "cuBLAS int8 GEMM" : "PcmmAccum")
            << ", converted weights "
            << (cache_weights_ ? "cached on the GPU" : "rebuilt every call")
            << std::endl;
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
  const int cols = num_parents * rank_;
  values.assign(static_cast<size_t>(rank_) * cols, 0.0);
  for (int r = 0; r < rank_; r++) {
    const int out_channel =
        group * rank_ + static_cast<int>(BitReverseInt(r, log_rank));
    for (int p = 0; p < num_parents; p++) {
      for (int i = 0; i < rank_; i++) {
        const int in_channel = (first_parent + p) * rank_ +
                               static_cast<int>(BitReverseInt(i, log_rank));
        values[static_cast<size_t>(r) * cols + p * rank_ + i] =
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
  const double scale = context_->param_.GetScale(level);
  const int cols = num_parents * rank_;
  std::vector<double> values;
  for (int g = 0; g < groups; g++) {
    GatherWeights(values, w, in_channels, out_channels, g, w_scale,
                  first_parent, num_parents);
    if (use_blas_) {
#ifdef USE_CUBLAS
      typename PcmmBlasHandler<word>::SplitMatrix s;
      blas_->SplitMatrixFrom(s, level, scale, values, rank_, cols);
      res.bytes += PcmmBlasHandler<word>::SplitBytes(s);
      res.split.push_back(std::move(s));
#endif
    } else {
      PlainMatrix<word> u;
      pcmm_.EncodeMatrix(u, level, scale, values, rank_, cols);
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
void CoeffLinearLeg<word>::Project(std::vector<Ct> &res,
                                   const std::vector<Ct> &x, int in_channels,
                                   int out_channels,
                                   const std::vector<double> &w, double w_scale,
                                   const char *name) const {
  const int level = cfg_.product_level;
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

  // The partial sums, one per output group, held at the product's level and
  // scale. They are ordinary RLWE ciphertexts because they have already been
  // ModPacked -- there is no MLWE addition to accumulate with instead.
  std::vector<Ct> partial(groups);
  bool started = false;
  int tile_index = 0;

  for (int base = 0; base < parents; base += tile, tile_index++) {
    const int span = std::min(tile, parents - base);

    // 1. Descend to the product level and split the channel axis onto the
    //    ciphertext axis. ModDecomp is the whole of that step and costs
    //    nothing but memory: `rank` module components per parent, each the
    //    size of the parent's own a-part. The tile bounds exactly that.
    std::vector<MlweCiphertext<word>> columns;
    columns.reserve(static_cast<size_t>(span) * rank_);
    {
    NvtxScope _n("pcmm: ModDecomp");
    for (int p = base; p < base + span; p++) {
      const Ct &parent = x[p];
      Ct lowered;
      const Ct *source = &parent;
      if (context_->param_.NPToLevel(parent.GetNP()) != level) {
        context_->LevelDown(lowered, parent, level);
        source = &lowered;
      }
      std::vector<MlweCiphertext<word>> decomposed;
      mlwe_.ModDecomp(decomposed, *source, small_degree_);
      for (auto &c : decomposed) columns.push_back(std::move(c));
    }
    }
    AssertTrue(static_cast<int>(columns.size()) == span * rank_,
               "CoeffLinearLeg::Project: ModDecomp did not yield one module "
               "component per input channel");

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
      blas_->PrepareSource(prepared, ops.split[first], columns);
      columns.clear();
    }
#endif

    // 2. One ModPack group of output channels at a time. The decomposition is
    //    shared by every group, which is why it is hoisted out of this loop --
    //    it is the expensive part, and the product itself is two plaintext
    //    matrix products with no key material at all.
    for (int g = 0; g < groups; g++) {
      std::vector<MlweCiphertext<word>> product;
      {
        NvtxScope _n("pcmm: Multiply");
        if (use_blas_) {
#ifdef USE_CUBLAS
          blas_->Multiply(product, ops.split[first + g], prepared);
#endif
        } else {
          pcmm_.Multiply(product, ops.u[first + g], columns);
        }
      }

      Ct repacked;
      {
        NvtxScope _n("pcmm: ModPack");
        mlwe_.ModPack(context_, repacked, product, modpack_keys_);
      }
      if (!started) {
        partial[g] = std::move(repacked);
      } else {
        Ct sum;
        context_->Add(sum, partial[g], repacked);
        partial[g] = std::move(sum);
      }
    }
    started = true;
  }

  // 3. One rescale, after the whole contraction. The product carries scale
  //    GetScale(level)^2; this brings it to GetScale(level - 1), and it is the
  //    single level the product spends however many tiles it took.
  res.resize(groups);
  {
    NvtxScope _n("pcmm: Rescale");
    for (int g = 0; g < groups; g++) context_->Rescale(res[g], partial[g]);
  }
}

template class CoeffLinearLeg<uint32_t>;
template class CoeffLinearLeg<uint64_t>;

}  // namespace cheddar
