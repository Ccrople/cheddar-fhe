#include "extension/CiLlamaSeam.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

namespace {

// Complex is cheddar::Complex, from core/Type.h via StripedMatrix.h. A local
// alias would be ambiguous with it at every use site inside this namespace.
int Rev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

int SwapBits(int x, int i, int j) {
  if (((x >> i) & 1) != ((x >> j) & 1)) x ^= (1 << i) | (1 << j);
  return x;
}

}  // namespace

template <typename word>
CiLlamaSeam<word>::CiLlamaSeam(ConstContextPtr<word> context,
                               const CiSwitchedCcmmLayout &layout,
                               int stc_level, const Config &cfg)
    : context_{std::move(context)}, layout_{layout}, cfg_{cfg} {
  degree_ = context_->param_.degree_;

  AssertTrue(layout_.dim == 128 && layout_.lanes == 32 && layout_.num_cts == 8,
             "CiLlamaSeam: the address arithmetic is stated at the Llama "
             "alignment -- dim 128, 32 lanes, 8 ciphertexts");
  // TWO RANKS, KEPT APART. `layout_.rank` is the chain's columns per
  // ciphertext (16); `cfg_.proj_rank` is the module rank the banded
  // convention is stated against (512). The channel index is bit-reversed
  // over the second and the column index over the first.
  AssertTrue(IsPowOfTwo(layout_.rank) && IsPowOfTwo(cfg_.proj_rank) &&
                 IsPowOfTwo(layout_.dim),
             "CiLlamaSeam: the chain rank, the projection rank and the token "
             "count must all be powers of two");
  AssertTrue(layout_.rank * layout_.lanes == cfg_.proj_rank,
             "CiLlamaSeam: the chain's columns times its lanes must be the "
             "projection rank, or `chan_of` does not cover the channels");
  log_cols_ = Log2Ceil(layout_.rank);
  log_proj_rank_ = Log2Ceil(cfg_.proj_rank);
  log_dim_ = Log2Ceil(layout_.dim);
  AssertTrue(!cfg_.t1_stages.empty(),
             "CiLlamaSeam: T1 needs at least one stage");

  // THE LADDER IS DERIVED FROM StC, NOT WRITTEN DOWN. See the class comment:
  // the same constants were once correct and then silently wrong when the
  // slack moved, and the symptom was coefficients at 4.99e+47.
  t2_level_ = stc_level + 2;
  tokmap_level_ = t2_level_ + 1;
  t1_top_ = tokmap_level_ + static_cast<int>(cfg_.t1_stages.size());

  AssertTrue(t1_top_ <= context_->param_.max_level_,
             "CiLlamaSeam: the seam does not fit above SlotToCoeff at this "
             "slack (needs level " +
                 std::to_string(t1_top_) + " of " +
                 std::to_string(context_->param_.max_level_) + ")");
  AssertTrue(t2_level_ - 1 > stc_level,
             "CiLlamaSeam: the seam has to leave the ciphertext above StC's "
             "level with a rescale to spare");
  // 1.5bt's zone, asserted rather than remembered: a LinearTransform is a
  // hoisted transform and ci16_35 returns 1e25..1e47 below level 7.
  AssertTrue(t2_level_ > 7,
             "CiLlamaSeam: the seam runs inside the num_accum == 1 zone");

  const int dim = layout_.dim;
  const int cols = layout_.rank;
  const int lanes_per_half = layout_.lanes / 2;
  const int log_lanes = Log2Ceil(layout_.lanes);
  auto slot_block = [dim](int token, int chan) { return token + dim * chan; };
  auto chan_of = [this, log_lanes](int col, int lh) {
    return Rev(col, log_cols_) * layout_.lanes + Rev(lh, log_lanes);
  };

  // ---- T2: the duplicates ------------------------------------------------
  {
    StripedMatrix m2(degree_, degree_);
    for (int col = 0; col < cols; col++) {
      for (int lh = 0; lh < lanes_per_half; lh++) {
        const int c = chan_of(col, lh);
        const int I = Rev(c, log_proj_rank_);
        // COMPONENT ZERO HAS NO PARTNER: `rank - 0` wraps to channel 0, which
        // is even and so live, and the banded recomposition excludes i == 0.
        // Taking the formula literally writes a duplicate over a live value.
        if (I == 0) continue;
        const int cd = Rev(cfg_.proj_rank - I, log_proj_rank_);
        const int off = ((dim * (c - cd)) % degree_ + degree_) % degree_;
        m2.try_emplace(off, degree_, Complex(0.0, 0.0));
        for (int row = 1; row < dim; row++) {
          m2[off][slot_block(row - 1, cd)] = Complex(1.0, 0.0);
        }
      }
    }
    t2_ = Compile(m2, t2_level_);
    if (cfg_.verbose) {
      std::cout << "  seam T2: " << m2.GetNumDiag() << " diagonals at level "
                << t2_level_ << std::endl;
    }
  }

  // ---- the token map: a bit-reversed decrement ---------------------------
  {
    StripedMatrix mt(degree_, degree_);
    for (int col = 0; col < cols; col++) {
      for (int lh = 0; lh < lanes_per_half; lh++) {
        const int c = chan_of(col, lh);
        for (int t = 0; t < dim; t++) {
          const int pos = Rev(t, log_dim_);
          if (pos == 0) continue;  // nothing sits one position below it
          const int td = Rev(pos - 1, log_dim_);
          const int dst = slot_block(td, c);
          const int off = ((slot_block(t, c) - dst) % degree_ + degree_) %
                          degree_;
          mt.try_emplace(off, degree_, Complex(0.0, 0.0));
          mt[off][dst] = Complex(1.0, 0.0);
        }
      }
    }
    tokmap_ = Compile(mt, tokmap_level_);
    if (cfg_.verbose) {
      std::cout << "  seam token map: " << mt.GetNumDiag()
                << " diagonals at level " << tokmap_level_ << std::endl;
    }
  }
}

template <typename word>
double CiLlamaSeam<word>::PtScale(int level) const {
  const auto &p = context_->param_;
  return p.GetScale(level - 1) * p.GetRescalePrimeProd(level) /
         p.GetScale(level);
}

template <typename word>
int CiLlamaSeam<word>::BestWindow(const StripedMatrix &m, int *need) const {
  std::vector<int> offs;
  for (const auto &kv : m) offs.push_back(kv.first);
  int best_w = 0;
  long long best_n = -1;
  for (int w : offs) {
    long long g = 0, mx = 0;
    for (int o : offs) {
      const long long r = ((o - w) % degree_ + degree_) % degree_;
      mx = std::max(mx, r);
      long long a = g, b = r;
      while (b) {
        const long long t = a % b;
        a = b;
        b = t;
      }
      g = a;
    }
    if (g == 0) continue;
    const long long q = mx / g + 1;
    if (best_n < 0 || q < best_n) {
      best_n = q;
      best_w = w;
    }
  }
  AssertTrue(best_n > 0, "CiLlamaSeam: no window makes the offsets strided");
  *need = static_cast<int>(best_n);
  return best_w;
}

template <typename word>
typename CiLlamaSeam<word>::Stage CiLlamaSeam<word>::Compile(
    const StripedMatrix &m, int level) const {
  int need = 0;
  const int w = BestWindow(m, &need);
  int bs = 1;
  while (bs * bs < need) bs *= 2;
  int gs = 1;
  while (bs * gs < need) gs *= 2;

  Stage st;
  st.level = level;
  // THE WINDOW CONVENTION, AND ITS SIGN. `DetermineStride` reduces every
  // offset as `(i - pre_rotation) mod degree`, which is exactly what
  // `BestWindow` minimises over, so `pre_rotation` is `+w` and the plaintext
  // rotation that undoes it is `-w`. Compiling with the signs the other way
  // round does not produce a wrong answer -- it refuses, with "Incompatible
  // matrix and LinearTransform parameters", because the offsets are then
  // spread over the whole ring instead of a window.
  st.transform = std::make_unique<LinearTransform<word>>(
      context_, m, level, PtScale(level), bs, gs, /*pre_rotation=*/w,
      /*additional_pt_rot=*/-w);
  st.back = ((w % degree_) + degree_) % degree_;
  return st;
}

template <typename word>
void CiLlamaSeam<word>::AddRequiredRotations(EvkRequest &req) const {
  tokmap_.transform->AddRequiredRotations(req);
  req.AddRequest(tokmap_.back, tokmap_level_ - 1);
  t2_.transform->AddRequiredRotations(req);
  req.AddRequest(t2_.back, t2_level_ - 1);
  // The live half is brought onto the duplicates' level with a constant
  // multiply, and `SylphSchedule::ToCoeff` follows; the rotation by 1 at
  // `t2_level_` is what the recorded key set asks for at this joint.
  req.AddRequest(1, t2_level_);
}

template <typename word>
void CiLlamaSeam<word>::PrepareHalf(int half) {
  AssertTrue(half == 0 || half == 1, "CiLlamaSeam: half must be 0 or 1");
  DropHalf();

  const int dim = layout_.dim;
  const int cols = layout_.rank;
  const int lanes = layout_.lanes;
  const int lanes_per_half = lanes / 2;
  const int log_lanes = Log2Ceil(lanes);
  auto slot_chain = [&](int row, int col, int lane) {
    return Rev(col, log_cols_) * (dim * lanes) + Rev(row, log_dim_) * lanes +
           lane;
  };
  auto slot_block = [dim](int token, int chan) { return token + dim * chan; };
  auto chan_of = [&](int col, int lh) {
    return Rev(col, log_cols_) * lanes + Rev(lh, log_lanes);
  };

  // The live source addresses, walked through the stages: a stage's matrix is
  // built from where its inputs ARE and where it sends them, so the
  // composition is checked by construction rather than re-derived.
  std::vector<int> cur;
  cur.reserve(static_cast<size_t>(cols) * lanes_per_half * dim);
  for (int col = 0; col < cols; col++) {
    for (int lh = 0; lh < lanes_per_half; lh++) {
      for (int row = 0; row < dim; row++) {
        cur.push_back(slot_chain(row, col, half * lanes_per_half + lh));
      }
    }
  }

  for (size_t st = 0; st < cfg_.t1_stages.size(); st++) {
    const bool last = (st + 1 == cfg_.t1_stages.size());
    std::vector<int> mid(cur.size());
    StripedMatrix ms(degree_, degree_);
    for (size_t e = 0; e < cur.size(); e++) {
      int y = cur[e];
      for (const auto &sw : cfg_.t1_stages[st]) y = SwapBits(y, sw.first, sw.second);
      // The last transposition puts `half` at the destination's token field,
      // where the packing wants a zero.
      if (last) y -= dim * half;
      mid[e] = y;
      const int off = ((cur[e] - y) % degree_ + degree_) % degree_;
      ms.try_emplace(off, degree_, Complex(0.0, 0.0));
      ms[off][y] = Complex(1.0, 0.0);
    }
    const int lvl = t1_top_ - static_cast<int>(st);
    t1_.push_back(Compile(ms, lvl));
    if (cfg_.verbose) {
      std::cout << "  seam T1[" << half << "] stage " << st << ": "
                << ms.GetNumDiag() << " diagonals at level " << lvl
                << std::endl;
    }
    cur.swap(mid);
  }

  // The composition must land exactly on the block's live addresses. This is
  // the check the staging is worth having: a wrong split is a wrong layer that
  // still decrypts, and it costs a quarter of an hour to find on the device.
  for (int col = 0; col < cols; col++) {
    for (int lh = 0; lh < lanes_per_half; lh++) {
      for (int row = 0; row < dim; row++) {
        const size_t e =
            (static_cast<size_t>(col) * lanes_per_half + lh) * dim + row;
        AssertTrue(cur[e] == slot_block(row, chan_of(col, lh)),
                   "CiLlamaSeam: the staged bit permutation is not T1");
      }
    }
  }
  prepared_half_ = half;
}

template <typename word>
void CiLlamaSeam<word>::AddHalfRotations(EvkRequest &req) const {
  AssertTrue(prepared_half_ >= 0,
             "CiLlamaSeam: PrepareHalf must run before its rotations are "
             "requested -- T1's offsets are only known once it is built");
  for (const auto &st : t1_) {
    st.transform->AddRequiredRotations(req);
    req.AddRequest(st.back, st.level - 1);
  }
}

template <typename word>
void CiLlamaSeam<word>::DropHalf() {
  t1_.clear();
  prepared_half_ = -1;
}

template <typename word>
void CiLlamaSeam<word>::RunStage(Ct &res, const Ct &in, const Stage &st,
                                 const EvkMap<word> &evk) const {
  st.transform->Evaluate(context_, res, in, evk);
  if (st.back == 0) return;
  Ct rotated;
  context_->HRot(rotated, res, evk.GetRotationKey(st.back), st.back);
  res = std::move(rotated);
}

template <typename word>
void CiLlamaSeam<word>::Apply(Ct &res, const Ct &booted,
                              SylphSchedule<word> &sched,
                              const EvkMap<word> &evk, bool min_ks) const {
  AssertTrue(prepared_half_ >= 0,
             "CiLlamaSeam: PrepareHalf must run before Apply");

  Ct live;
  context_->LevelDown(live, booted, t1_top_);
  for (const auto &st : t1_) {
    Ct next;
    RunStage(next, live, st, evk);
    live = std::move(next);
  }

  // The duplicates come off the TOKEN-SHIFTED image, and the live half is T1's
  // own output -- the shift belongs to the duplicate band alone.
  Ct shifted, dup;
  RunStage(shifted, live, tokmap_, evk);
  RunStage(dup, shifted, t2_, evk);

  const auto &p = context_->param_;
  const int dl = p.NPToLevel(dup.GetNP());
  // The live half sits two levels above `dup`, because the token map took one;
  // bring it to `dl + 1` so the constant multiply below lands both at the same
  // level AND the same scale.
  if (p.NPToLevel(live.GetNP()) != dl + 1) {
    Ct down;
    context_->LevelDown(down, live, dl + 1);
    live = std::move(down);
  }
  Constant<word> one;
  context_->encoder_.EncodeConstant(
      one, dl + 1,
      p.GetScale(dl) * p.GetRescalePrimeProd(dl + 1) / live.GetScale(), 1.0);
  context_->Mult(live, live, one);
  context_->Rescale(live, live);

  Ct sum;
  context_->Add(sum, live, dup);
  sched.ToCoeff(res, sum, evk, min_ks);
}

template class CiLlamaSeam<uint32_t>;
template class CiLlamaSeam<uint64_t>;

}  // namespace cheddar
