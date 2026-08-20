#include "extension/LlamaLinear.h"

#include <algorithm>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

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
      pcmm_{context->param_} {
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
}

template <typename word>
void CoeffLinearLeg<word>::EncodeWeights(PlainMatrix<word> &res,
                                         const std::vector<double> &w,
                                         int in_channels, int out_channels,
                                         int group, double w_scale,
                                         int first_parent,
                                         int num_parents) const {
  const int log_rank = Log2Ceil(rank_);
  const int cols = num_parents * rank_;
  std::vector<double> values(static_cast<size_t>(rank_) * cols);
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
  // The one scale that leaves the rescaled product canonical at the level
  // below; see the class comment.
  pcmm_.EncodeMatrix(res, cfg_.product_level,
                     context_->param_.GetScale(cfg_.product_level), values,
                     rank_, cols);
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

  // The partial sums, one per output group, held at the product's level and
  // scale. They are ordinary RLWE ciphertexts because they have already been
  // ModPacked -- there is no MLWE addition to accumulate with instead.
  std::vector<Ct> partial(groups);
  bool started = false;

  for (int base = 0; base < parents; base += tile) {
    const int span = std::min(tile, parents - base);

    // 1. Descend to the product level and split the channel axis onto the
    //    ciphertext axis. ModDecomp is the whole of that step and costs
    //    nothing but memory: `rank` module components per parent, each the
    //    size of the parent's own a-part. The tile bounds exactly that.
    std::vector<MlweCiphertext<word>> columns;
    columns.reserve(static_cast<size_t>(span) * rank_);
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
    AssertTrue(static_cast<int>(columns.size()) == span * rank_,
               "CoeffLinearLeg::Project: ModDecomp did not yield one module "
               "component per input channel");

    // 2. One ModPack group of output channels at a time. The decomposition is
    //    shared by every group, which is why it is hoisted out of this loop --
    //    it is the expensive part, and the product itself is two plaintext
    //    matrix products with no key material at all.
    for (int g = 0; g < groups; g++) {
      PlainMatrix<word> u;
      EncodeWeights(u, w, in_channels, out_channels, g, w_scale, base, span);

      std::vector<MlweCiphertext<word>> product;
      pcmm_.Multiply(product, u, columns);

      Ct repacked;
      mlwe_.ModPack(context_, repacked, product, modpack_keys_);
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
  for (int g = 0; g < groups; g++) context_->Rescale(res[g], partial[g]);
}

template class CoeffLinearLeg<uint32_t>;
template class CoeffLinearLeg<uint64_t>;

}  // namespace cheddar
