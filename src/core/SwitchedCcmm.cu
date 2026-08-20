#include "core/SwitchedCcmm.h"

#include <string>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

namespace {

int BitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

}  // namespace

SwitchedCcmmLayout::SwitchedCcmmLayout(int big_degree, int small_degree,
                                       int sub_degree)
    : big_degree{big_degree},
      small_degree{small_degree},
      sub_degree{sub_degree} {
  AssertTrue(IsPowOfTwo(big_degree) && IsPowOfTwo(small_degree) &&
                 IsPowOfTwo(sub_degree),
             "SwitchedCcmmLayout: every degree must be a power of two");
  AssertTrue(small_degree < big_degree,
             "SwitchedCcmmLayout: the product ring must be the smaller one");
  AssertTrue(sub_degree >= 2 && sub_degree < small_degree,
             "SwitchedCcmmLayout: sub_degree must leave at least two blocks "
             "and at least one lane");
  rank = big_degree / small_degree;
  dim = small_degree / sub_degree;
  lanes = sub_degree / 2;
  AssertTrue(dim % rank == 0,
             "SwitchedCcmmLayout: the matrix width must be a whole number of "
             "ring switches -- d = " +
                 std::to_string(dim) + " against rank " + std::to_string(rank));
  num_cts = dim / rank;
  // The three fields are exactly the big ring's slot index. This is an
  // identity, not a constraint, but it is the identity the whole layout rests
  // on and it costs nothing to check.
  AssertTrue(rank * dim * lanes == big_degree / 2,
             "SwitchedCcmmLayout: the three fields do not tile the slot index");
}

void SwitchedCcmmLayout::Locate(int row, int column, int lane, int &ct,
                                int &slot) const {
  AssertTrue(row >= 0 && row < dim && column >= 0 && column < dim &&
                 lane >= 0 && lane < lanes,
             "SwitchedCcmmLayout::Locate: index out of range");
  const int log_rank = Log2Ceil(rank);
  const int log_dim = Log2Ceil(dim);
  const int log_lanes = Log2Ceil(lanes);
  ct = column / rank;
  const int j = column % rank;
  slot = (BitRev(j, log_rank) << (log_dim + log_lanes)) |
         (BitRev(row, log_dim) << log_lanes) | lane;
}

void SwitchedCcmmLayout::LocateSinC(int row, int column, int lane, int &ct,
                                    int &index) const {
  AssertTrue(row >= 0 && row < dim && column >= 0 && column < dim &&
                 lane >= 0 && lane < lanes,
             "SwitchedCcmmLayout::LocateSinC: index out of range");
  ct = column / rank;
  const int j = column % rank;
  // [ row | j | lane ], which is the big block index `j + rank * row` times
  // the lane count plus the lane -- exactly what the ring switch's SinC
  // identity says lands at block `row` of small ciphertext `j`.
  index = (row * rank + j) * lanes + lane;
}

bool SwitchedCcmmLayout::Position(int ct, int slot, int &row, int &column,
                                  int &lane) const {
  if (ct < 0 || ct >= num_cts || slot < 0 || slot >= big_degree / 2) {
    return false;
  }
  const int log_rank = Log2Ceil(rank);
  const int log_dim = Log2Ceil(dim);
  const int log_lanes = Log2Ceil(lanes);
  lane = slot & (lanes - 1);
  row = BitRev((slot >> log_lanes) & (dim - 1), log_dim);
  column = ct * rank + BitRev(slot >> (log_dim + log_lanes), log_rank);
  return true;
}

template <typename word>
SwitchedCcmmHandler<word>::SwitchedCcmmHandler(
    ConstContextPtr<word> switch_ctx, ConstContextPtr<word> small_ctx,
    int sub_degree)
    : switch_ctx_{switch_ctx},
      small_ctx_{small_ctx},
      layout_{switch_ctx->param_.degree_, small_ctx->param_.degree_,
              sub_degree},
      switcher_{switch_ctx, small_ctx},
      ccmm_{small_ctx->param_, small_ctx->ntt_handler_} {}

template <typename word>
void SwitchedCcmmHandler<word>::Descend(std::vector<Ct> &res,
                                        const std::vector<Ct> &operand,
                                        const Evk &swk) const {
  AssertTrue(static_cast<int>(operand.size()) == layout_.num_cts,
             "SwitchedCcmm: an operand is d / rank big ciphertexts");
  res.clear();
  res.resize(layout_.dim);
  for (int b = 0; b < layout_.num_cts; b++) {
    std::vector<Ct> parts;
    switcher_.Switch(parts, operand[b], swk);
    AssertTrue(static_cast<int>(parts.size()) == layout_.rank,
               "SwitchedCcmm: the switch returned the wrong number of parts");
    // column = b * rank + j, which is the assignment SwitchedCcmmLayout::
    // Locate is written against; SwitchBack undoes it below by grouping the
    // results the same way.
    for (int j = 0; j < layout_.rank; j++) {
      res[b * layout_.rank + j] = std::move(parts[j]);
    }
  }
}

template <typename word>
void SwitchedCcmmHandler<word>::Multiply(std::vector<Ct> &res,
                                         const std::vector<Ct> &lhs,
                                         const std::vector<Ct> &rhs,
                                         const Evk &swk, const Evk &inv_swk,
                                         const EvkMap<word> &small_evk) const {
  std::vector<Ct> small_lhs, small_rhs;
  Descend(small_lhs, lhs, swk);
  Descend(small_rhs, rhs, swk);

  std::vector<Ct> small_res;
  ccmm_.Multiply(small_ctx_, small_res, small_lhs, small_rhs,
                 layout_.sub_degree, small_evk);
  AssertTrue(static_cast<int>(small_res.size()) == layout_.dim,
             "SwitchedCcmm: the product returned the wrong number of columns");
  // Freeing the operands before the return trip matters: each is d small
  // ciphertexts and the pool is sized from free memory at context creation.
  small_lhs.clear();
  small_rhs.clear();

  res.clear();
  res.resize(layout_.num_cts);
  for (int b = 0; b < layout_.num_cts; b++) {
    std::vector<Ct> parts(layout_.rank);
    for (int j = 0; j < layout_.rank; j++) {
      parts[j] = std::move(small_res[b * layout_.rank + j]);
    }
    switcher_.SwitchBack(res[b], parts, inv_swk);
  }
}

template class SwitchedCcmmHandler<uint32_t>;
template class SwitchedCcmmHandler<uint64_t>;

}  // namespace cheddar
