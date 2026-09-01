#include "core/CiSwitchedCcmm.h"

#include <string>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/Profile.h"

namespace cheddar {

CiSwitchedCcmmLayout::CiSwitchedCcmmLayout(int big_degree, int small_degree,
                                           int sub_degree)
    : big_degree{big_degree},
      small_degree{small_degree},
      sub_degree{sub_degree} {
  AssertTrue(IsPowOfTwo(big_degree) && IsPowOfTwo(small_degree) &&
                 IsPowOfTwo(sub_degree),
             "CiSwitchedCcmmLayout: every degree must be a power of two");
  AssertTrue(small_degree < big_degree,
             "CiSwitchedCcmmLayout: the product ring must be the smaller one");
  AssertTrue(sub_degree >= 2 && sub_degree < small_degree,
             "CiSwitchedCcmmLayout: sub_degree must leave at least two "
             "blocks");
  rank = big_degree / small_degree;
  dim = small_degree / sub_degree;
  lanes = sub_degree;  // k real lanes: the block subring is totally real
  AssertTrue(dim % rank == 0,
             "CiSwitchedCcmmLayout: the matrix width must be a whole number "
             "of ring switches -- d = " +
                 std::to_string(dim) + " against rank " +
                 std::to_string(rank));
  num_cts = dim / rank;
  AssertTrue(num_cts % 2 == 0,
             "CiSwitchedCcmmLayout: the half-contraction contract needs the "
             "live columns to be whole big ciphertexts, so d / rank must be "
             "even -- got " +
                 std::to_string(num_cts));
  contraction = dim / 2;
  // The three fields tile the big ring's real coefficients exactly. An
  // identity, not a constraint, but the one the part addressing rests on.
  AssertTrue(rank * dim * lanes == big_degree,
             "CiSwitchedCcmmLayout: the fields do not tile the big ring");
}

void CiSwitchedCcmmLayout::LocatePart(int row, int column, int lane,
                                      int &part, int &index) const {
  AssertTrue(row >= 0 && row < dim && column >= 0 && column < dim &&
                 lane >= 0 && lane < lanes,
             "CiSwitchedCcmmLayout::LocatePart: index out of range");
  part = column;
  index = row * sub_degree + lane;
}

bool CiSwitchedCcmmLayout::PositionPart(int part, int index, int &row,
                                        int &column, int &lane) const {
  if (part < 0 || part >= dim || index < 0 || index >= small_degree) {
    return false;
  }
  column = part;
  row = index / sub_degree;
  lane = index % sub_degree;
  return true;
}

namespace {

int BitReverse(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

}  // namespace

int CiSwitchedCcmmLayout::LocateSlot(int row, int column, int lane, int &ct,
                                     int &slot, int &copy_slot) const {
  AssertTrue(row >= 0 && row < dim && column >= 0 && column < dim &&
                 lane >= 0 && lane < lanes,
             "CiSwitchedCcmmLayout::LocateSlot: index out of range");
  ct = column / rank;
  const int cls = column % rank;
  // The flat block index runs over ALL of one big ciphertext's blocks at
  // this sub-degree; its bit reversal is the CI SlotToSinC's own block map
  // (EvalSpecialFFT: slot group A -> SinC block BitRev(A), lanes untouched).
  const int num_flat_blocks = big_degree / sub_degree;  // rank * dim
  const int log_blocks = Log2Ceil(num_flat_blocks);
  slot = BitReverse(row * rank + cls, log_blocks) * sub_degree + lane;
  copy_slot = -1;
  if (cls != 0 && row >= 1) {
    copy_slot =
        BitReverse((row - 1) * rank + (rank - cls), log_blocks) * sub_degree +
        lane;
  }
  return copy_slot < 0 ? 1 : 2;
}

template <typename word>
CiSwitchedCcmmHandler<word>::CiSwitchedCcmmHandler(
    ConstContextPtr<word> switch_ctx, ConstContextPtr<word> small_ctx,
    ConstContextPtr<word> lifted_ctx, int sub_degree)
    : switch_ctx_{switch_ctx},
      small_ctx_{small_ctx},
      lifted_ctx_{lifted_ctx},
      layout_{switch_ctx_->param_.degree_, small_ctx_->param_.degree_,
              sub_degree},
      switcher_{switch_ctx_, small_ctx_},
      lift_{small_ctx_, lifted_ctx_},
      ccmm_{lifted_ctx_->param_, lifted_ctx_->ntt_handler_} {
  // The switch pair must be conjugate-invariant; the lift handler already
  // enforces that lifted_ctx is the matching ordinary ring.
  AssertTrue(switch_ctx_->param_.conjugate_invariant_ &&
                 small_ctx_->param_.conjugate_invariant_,
             "CiSwitchedCcmm: the switch pair must be conjugate-invariant "
             "rings -- the ordinary chain has its own handler");
}

template <typename word>
void CiSwitchedCcmmHandler<word>::DescendAndLift(std::vector<Ct> &res,
                                                 const std::vector<Ct> &operand,
                                                 const Evk &swk) const {
  const int num_live = static_cast<int>(operand.size());
  res.clear();
  res.resize(layout_.dim);
  for (int b = 0; b < num_live; b++) {
    std::vector<Ct> parts;
    switcher_.Switch(parts, operand[b], swk);
    AssertTrue(static_cast<int>(parts.size()) == layout_.rank,
               "CiSwitchedCcmm: the switch returned the wrong number of "
               "parts");
    // column = b * rank + j, the assignment LocatePart is written against;
    // SwitchBack regroups the results the same way below.
    for (int j = 0; j < layout_.rank; j++) {
      lift_.Lift(res[b * layout_.rank + j], parts[j]);
    }
  }
  if (num_live == layout_.dim / layout_.rank) return;

  // The contract's dead columns: exact zeros at the lifted ring, which is
  // what makes the flip term of Doing.md 1.5bl vanish identically rather
  // than approximately -- and skips their switches and lifts entirely.
  AssertTrue(num_live > 0, "CiSwitchedCcmm: an operand has no ciphertexts");
  const NPInfo np = res[0].GetNP();
  const size_t component_bytes = static_cast<size_t>(np.GetNumTotal()) *
                                 lifted_ctx_->param_.degree_ * sizeof(word);
  for (int x = num_live * layout_.rank; x < layout_.dim; x++) {
    Ct &zero = res[x];
    zero.RemoveRx();
    zero.ModifyNP(np);
    zero.SetScale(res[0].GetScale());
    zero.SetNumSlots(res[0].GetNumSlots());
    cudaMemsetAsync(zero.bx_.data(), 0, component_bytes, cudaStreamLegacy);
    cudaMemsetAsync(zero.ax_.data(), 0, component_bytes, cudaStreamLegacy);
  }
}

template <typename word>
void CiSwitchedCcmmHandler<word>::Multiply(std::vector<Ct> &res,
                                           const std::vector<Ct> &lhs,
                                           const std::vector<Ct> &rhs,
                                           const Evk &swk, const Evk &inv_swk,
                                           const EvkMap<word> &lifted_evk)
    const {
  AssertTrue(static_cast<int>(lhs.size()) == layout_.num_cts / 2,
             "CiSwitchedCcmm: the lhs operand is num_cts / 2 big "
             "ciphertexts -- the contract's live columns and nothing else");
  AssertTrue(static_cast<int>(rhs.size()) == layout_.num_cts,
             "CiSwitchedCcmm: the rhs operand is num_cts big ciphertexts");

  NvtxScope *_d = new NvtxScope("ccmm: DescendAndLift (ring switch + lift)");
  std::vector<Ct> lifted_lhs, lifted_rhs;
  DescendAndLift(lifted_lhs, lhs, swk);
  DescendAndLift(lifted_rhs, rhs, swk);
  delete _d;

  std::vector<Ct> lifted_res;
  ccmm_.Multiply(lifted_ctx_, lifted_res, lifted_lhs, lifted_rhs,
                 2 * layout_.sub_degree, lifted_evk);
  AssertTrue(static_cast<int>(lifted_res.size()) == layout_.dim,
             "CiSwitchedCcmm: the product returned the wrong number of "
             "columns");
  // Each operand is d lifted ciphertexts and the pool is sized from free
  // memory at context creation; release them before the return trip.
  lifted_lhs.clear();
  lifted_rhs.clear();

  res.clear();
  res.resize(layout_.num_cts);
  NvtxScope _b("ccmm: Descend + SwitchBack");
  std::vector<Ct> parts(layout_.rank);
  for (int b = 0; b < layout_.num_cts; b++) {
    for (int j = 0; j < layout_.rank; j++) {
      lift_.Descend(parts[j], lifted_res[b * layout_.rank + j]);
    }
    switcher_.SwitchBack(res[b], parts, inv_swk);
  }
}

template class CiSwitchedCcmmHandler<uint32_t>;
template class CiSwitchedCcmmHandler<uint64_t>;

}  // namespace cheddar
