#include "core/CtAxisExchange.h"

#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
CtAxisExchange<word>::CtAxisExchange(ConstContextPtr<word> context,
                                     int log_num_cts, int field_offset,
                                     int level,
                                     int log_num_src_cts /*= -1*/)
    : context_{context},
      num_slots_{context->param_.degree_ / 2},
      log_num_cts_{log_num_cts},
      log_num_src_cts_{log_num_src_cts < 0 ? log_num_cts : log_num_src_cts},
      field_offset_{field_offset},
      level_{level} {
  AssertTrue(log_num_cts_ >= 1, "CtAxisExchange: the field must be non-empty");
  AssertTrue(log_num_src_cts_ >= 0 && log_num_src_cts_ <= log_num_cts_,
             "CtAxisExchange: the source cannot be wider than the output");
  AssertTrue(field_offset_ >= 0 &&
                 (1 << (field_offset_ + log_num_cts_)) <= num_slots_,
             "CtAxisExchange: the field does not fit the slot index");
  AssertTrue(level_ >= 1, "CtAxisExchange: the mask multiply spends a level");
  num_cts_ = 1 << log_num_cts_;
  num_src_cts_ = 1 << log_num_src_cts_;
  step_ = 1 << field_offset_;

  // One mask per field value. They are the whole crossbar: every plaintext
  // multiplication below is by one of these, so there are 2^w of them however
  // many products they take part in.
  const double scale = context->param_.GetRescalePrimeProd(level_);
  mask_.resize(num_cts_);
  std::vector<Complex> m(num_slots_);
  for (int z = 0; z < num_cts_; z++) {
    for (int p = 0; p < num_slots_; p++) {
      m[p] = Complex(((p >> field_offset_) & (num_cts_ - 1)) == z ? 1.0 : 0.0,
                     0.0);
    }
    context->encoder_.Encode(mask_[z], level_, scale, m);
  }
}

template <typename word>
std::vector<int> CtAxisExchange<word>::RotationIndices() const {
  std::vector<int> idx;
  idx.reserve(2 * (num_cts_ - 1));
  for (int x = 1; x < num_cts_; x++) idx.push_back(num_slots_ - x * step_);
  for (int y = 1; y < num_cts_; y++) idx.push_back(y * step_);
  return idx;
}

template <typename word>
void CtAxisExchange<word>::AddRequiredRotations(EvkRequest &req) const {
  // The inward rotations act on the operands, the outward ones on the masked
  // sums -- which have already been rescaled, so they are one level lower.
  for (int x = 1; x < num_cts_; x++) {
    req.AddRequest(num_slots_ - x * step_, level_);
  }
  for (int y = 1; y < num_cts_; y++) req.AddRequest(y * step_, level_ - 1);
}

template <typename word>
void CtAxisExchange<word>::Evaluate(std::vector<Ct> &res,
                                    const std::vector<Ct> &src,
                                    const EvkMap<word> &evk_map) const {
  AssertTrue(static_cast<int>(src.size()) == num_src_cts_,
             "CtAxisExchange: expected " + std::to_string(num_src_cts_) +
                 " source ciphertexts, got " + std::to_string(src.size()));
  const int replication = log_num_cts_ - log_num_src_cts_;

  // W_X = rot_{-X*step}(S_X). W_0 is S_0 itself and is read in place.
  std::vector<Ct> rotated(num_cts_);
  for (int x = 1; x < num_cts_; x++) {
    const int dist = num_slots_ - x * step_;
    context_->HRot(rotated[x], src[x >> replication],
                   evk_map.GetRotationKey(dist), dist);
  }

  res.clear();
  res.resize(num_cts_);
  for (int y = 0; y < num_cts_; y++) {
    Ct accum, term;
    for (int x = 0; x < num_cts_; x++) {
      const Ct &w = (x == 0) ? src[0] : rotated[x];
      const Pt &mask = mask_[(x + y) & (num_cts_ - 1)];
      if (x == 0) {
        context_->Mult(accum, w, mask);
      } else {
        context_->Mult(term, w, mask);
        context_->Add(accum, accum, term);
      }
    }
    Ct scaled;
    context_->Rescale(scaled, accum);
    if (y == 0) {
      res[0] = std::move(scaled);
    } else {
      context_->HRot(res[y], scaled, evk_map.GetRotationKey(y * step_),
                     y * step_);
    }
  }
}

template class CtAxisExchange<uint32_t>;
template class CtAxisExchange<uint64_t>;

}  // namespace cheddar
