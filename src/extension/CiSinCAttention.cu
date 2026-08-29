#include "extension/CiSinCAttention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "extension/ChebyshevFit.h"
#include "extension/StripedMatrix.h"

namespace cheddar {

namespace {

int BitRev(int v, int bits) {
  int r = 0;
  for (int i = 0; i < bits; i++) r |= ((v >> i) & 1) << (bits - 1 - i);
  return r;
}

// The token/head field swap of Doing.md 1.5by: slot bits [11..7] <-> [4..0].
int Exch(int s) {
  const int a = (s >> 7) & 31, b = s & 31;
  return (s & ~((31 << 7) | 31)) | (b << 7) | a;
}

// The doorstep (1.5bx): entry (token t, channel c, head i) of a projection
// sits at slot rev7(t) * 512 + rev4(c mod 16) * 32 + i of ciphertext c / 16.
int Door0(int t, int c, int i) {
  return (BitRev(c % 16, 4) << 12) | (BitRev(i, 5) << 7) | BitRev(t, 7);
}
int Door1(int t, int c, int i) { return Exch(Door0(t, c, i)); }
// K after its forced cross: the moved token field replaced by the deposit
// field (c / 16) mod 4 -- the cross rotations put the token's top bits into
// the ciphertext index, and this is where the remainder lands (1.5bx).
int Door1K(int t, int c, int i) {
  return (Door1(t, c, i) & ~(7 << 7)) | (((c / 16) % 4) << 7);
}

}  // namespace

template <typename word>
CiSinCAttention<word>::CiSinCAttention(
    std::shared_ptr<const BootContext<word>> boot,
    ConstContextPtr<word> switch_ctx, ConstContextPtr<word> small_ctx,
    ConstContextPtr<word> lifted_ctx, const Config &cfg)
    : boot_{std::move(boot)},
      switch_ctx_{std::move(switch_ctx)},
      cfg_{cfg},
      ccmm_{switch_ctx_, small_ctx, lifted_ctx, cfg.sub_degree} {
  degree_ = boot_->param_.degree_;
  num_slots_ = boot_->param_.MaxNumSlots();
  const auto &layout = ccmm_.GetLayout();
  AssertTrue(layout.dim == 128 && layout.lanes == 32 && layout.num_cts == 8 &&
                 cfg_.sub_degree == 32,
             "CiSinCAttention: the transport (doorstep, premaps, exchange, "
             "cross) is stated at the Llama alignment -- sub_degree 32, "
             "dim 128, 32 lanes, 8 ciphertexts");
  // The transport's height is a DIAL, not a fact: RoPE leaves the halves
  // at land_level - 1, but nothing between there and the descent needs
  // that height, and every key switch pays for the limbs it carries.
  // `exchange_level` may sit anywhere between the descent and the
  // landing; `Merge` drops the halves onto it. The two hard constraints
  // are that the exchange's OUTPUT is the cross's input (its selector
  // plaintexts are encoded at cross_level), and that a hoisted transform
  // must stay above the alpha-12 num_accum == 1 zone -- levels 0..6 on
  // ci16_35 (Doing.md 1.5bt), which the exchange's own BSGS is.
  AssertTrue(cfg_.exchange_level >= 7,
             "CiSinCAttention: the exchange is a hoisted transform and "
             "levels 0..6 are ci16_35's num_accum == 1 zone");
  AssertTrue(cfg_.cross_level == cfg_.exchange_level - 1,
             "CiSinCAttention: the exchange rescales onto the cross's "
             "level, so cross_level must be exchange_level - 1");
  AssertTrue(cfg_.land_level > cfg_.exchange_level &&
                 cfg_.exchange_level > cfg_.cross_level &&
                 cfg_.cross_level > cfg_.forward_level &&
                 cfg_.forward_level == cfg_.chain_level + 1 &&
                 cfg_.chain_level == cfg_.inverse_level + 1,
             "CiSinCAttention: the level ladder is land > exchange > cross "
             "and forward = chain + 1 = inverse + 2");

  // The canonicalising fold of Doing.md 1.5cb: HalfBoot declares its output
  // at GetStCInputScale() (~2^58), and riding that into the Boot boundary
  // breaks EvalMod. gamma folds into BOTH the RoPE/restore masks and the
  // exchange's plaintexts, so each stays an integer ~2^28.
  const double stc = boot_->GetStCInputScale();
  AssertTrue(stc > 0.0,
             "CiSinCAttention: PrepareEvalMod must run before construction; "
             "the canonicalising gamma reads GetStCInputScale()");
  gamma_ = std::sqrt(boot_->param_.GetScale(cfg_.cross_level) / stc);

  // THE CACHE DIRECTORY IS RESOLVED HERE, not at the three call sites, so that
  // every caller -- the layer test, the leg test, a 32-layer driver -- gets it
  // from one place and none of them can forget. The environment wins over the
  // Config, as elsewhere in this tree.
  if (cfg_.converter_cache_dir.empty()) {
    const char *dir = std::getenv("CHEDDAR_CONVERTER_CACHE");
    if (dir != nullptr && dir[0] != 0) cfg_.converter_cache_dir = dir;
  }

  BuildPremaps();

  conv_q_ = MakeConverter("q", /*inverse_level=*/-1, layout, &pre_q_);
  conv_k_ = MakeConverter("k", /*inverse_level=*/-1, layout, &pre_k_);
  conv_pv_ = MakeConverter("pv", cfg_.inverse_level, layout,
                           /*premap=*/nullptr);

  // The 63-diagonal token/head exchange (1.5by), window convention: the
  // negative offsets 127 * [-31, 31] pass DetermineStride's wrap wall as
  // pre_rotation = -window, and one closing rotation restores the frame.
  window_ = 31 * 127;
  StripedMatrix em(degree_, degree_);
  for (int r = 0; r < degree_; r++) {
    const int in = Exch(r);
    const int off = ((in - r) % degree_ + degree_) % degree_;
    em.try_emplace(off, degree_, Complex(0.0, 0.0));
    em[off][r] = Complex(1.0, 0.0);
  }
  exchange_.emplace_back(
      boot_, em, cfg_.exchange_level,
      boot_->param_.GetRescalePrimeProd(cfg_.exchange_level) * gamma_, 8, 8,
      /*pre_rotation=*/-window_, /*additional_pt_rot=*/window_);

  BuildTransportPlaintexts();

  reduce_dist_.clear();
  for (int t = 0; t < 4; t++) {
    reduce_dist_.push_back((num_slots_ / layout.rank) << t);
  }

  if (cfg_.verbose) {
    std::cout << "CiSinCAttention: gamma 2^" << std::log2(gamma_)
              << ", exchange " << em.GetNumDiag() << " diagonals @"
              << cfg_.exchange_level << std::endl;
  }
}

template <typename word>
std::string CiSinCAttention<word>::ConverterCachePath(
    const char *which) const {
  if (cfg_.converter_cache_dir.empty()) return std::string();
  // THE RECIPE GOES IN THE NAME. An archive's identity covers the parameter
  // set, which is necessary and not sufficient: two converters over the same
  // ring differ by the sub-degree, the three levels, the baby-step split and
  // the premap, and a hit on the wrong one would be silent because the
  // plaintexts decode perfectly well -- into the wrong transform. Everything
  // that shapes a diagonal is therefore either in this name or in the header.
  const auto &layout = ccmm_.GetLayout();
  std::ostringstream os;
  os << cfg_.converter_cache_dir << "/ci_conv_" << which << "_k"
     << cfg_.sub_degree << "_f" << cfg_.forward_level << "_i"
     << cfg_.inverse_level << "_bs" << cfg_.converter_baby_steps << "_r"
     << layout.rank << "_n" << num_slots_ << "_d" << degree_ << ".bin";
  return os.str();
}

template <typename word>
std::unique_ptr<CiSinCConverter<word>> CiSinCAttention<word>::MakeConverter(
    const char *which, int inverse_level, const CiSwitchedCcmmLayout &layout,
    const std::vector<int> *premap) const {
  const std::string path = ConverterCachePath(which);
  const auto id = IdentityOf(switch_ctx_->param_);

  if (!path.empty() && ArchiveReader::PeekIdentity(path) == id) {
    ArchiveReader ar(path, id);
    auto conv = CiSinCConverter<word>::Load(ar);
    if (cfg_.verbose) {
      std::cout << "  converter " << which << ": read "
                << (ArchiveReader::FileSize(path) >> 20) << " MiB from "
                << path << std::endl;
    }
    return conv;
  }

  auto conv = std::make_unique<CiSinCConverter<word>>(
      switch_ctx_, cfg_.sub_degree, cfg_.forward_level, inverse_level, &layout,
      premap, cfg_.converter_baby_steps);

  if (!path.empty()) {
    // WRITTEN ASIDE AND RENAMED. Writing after the build already means a run
    // that dies mid-build leaves no file; the remaining hole is a run killed
    // mid-WRITE, which leaves a valid header over a truncated body -- and a
    // valid header is exactly what the next run checks before committing to
    // read. `rename` is atomic within a filesystem, so the cache only ever
    // holds whole converters.
    const std::string tmp = path + ".tmp";
    int64_t written = 0;
    {
      ArchiveWriter ar(tmp, id);
      conv->Save(ar);
      ar.Close();
      written = ar.Written();
    }
    AssertTrue(std::rename(tmp.c_str(), path.c_str()) == 0,
               "CiSinCAttention: could not move the converter cache into "
               "place at " + path);
    if (cfg_.verbose) {
      std::cout << "  converter " << which << ": wrote " << (written >> 20)
                << " MiB to " << path << std::endl;
    }
  }
  return conv;
}

template <typename word>
void CiSinCAttention<word>::BuildPremaps() {
  const auto &layout = ccmm_.GetLayout();
  const int num_blocks = degree_ / cfg_.sub_degree;
  pre_q_.assign(num_blocks, -1);
  pre_k_.assign(num_blocks, -1);
  // Enumerated against LocateSlot, never hand-derived (the 1.5bx lesson).
  for (int t = 0; t < layout.dim; t++) {
    for (int c = 0; c < layout.dim; c++) {
      int ct_idx, slot, copy_slot;
      layout.LocateSlot(t, c, 0, ct_idx, slot, copy_slot);
      AssertTrue(ct_idx == c / 16,
                 "CiSinCAttention: the layout does not put the channel group "
                 "in the ciphertext index");
      int db = Door1(t, c, 0) >> 5;
      if (pre_q_[db] == -1) pre_q_[db] = slot >> 5;
      AssertTrue(pre_q_[db] == (slot >> 5),
                 "CiSinCAttention: Q's transport is not a block permutation");
      layout.LocateSlot(c % layout.contraction, t, 0, ct_idx, slot, copy_slot);
      AssertTrue(ct_idx == t / 16,
                 "CiSinCAttention: K's token bits do not land in the "
                 "ciphertext index");
      db = Door1K(t, c, 0) >> 5;
      if (pre_k_[db] == -1) pre_k_[db] = slot >> 5;
      AssertTrue(pre_k_[db] == (slot >> 5),
                 "CiSinCAttention: K's interior swap is not a block "
                 "permutation");
    }
  }
  // K's dead deposit fields complete the bijection onto exactly the chain
  // blocks (rows >= contraction) the half-contraction contract wants zero.
  std::vector<char> used(num_blocks, 0);
  for (int b = 0; b < num_blocks; b++) {
    if (pre_k_[b] != -1) used[pre_k_[b]] = 1;
  }
  std::vector<int> free_out;
  for (int b = 0; b < num_blocks; b++) {
    if (!used[b]) free_out.push_back(b);
  }
  size_t fo = 0;
  for (int b = 0; b < num_blocks; b++) {
    if (pre_k_[b] == -1) pre_k_[b] = free_out[fo++];
  }
  AssertTrue(fo == free_out.size(),
             "CiSinCAttention: K's premap did not complete to a bijection");
}

template <typename word>
void CiSinCAttention<word>::BuildTransportPlaintexts() {
  const auto &layout = ccmm_.GetLayout();
  const int half = layout.contraction;
  const double pt_scale =
      boot_->param_.GetRescalePrimeProd(cfg_.land_level) * gamma_;
  std::vector<double> theta(half);
  for (int m = 0; m < half; m++) {
    theta[m] = std::pow(cfg_.rope_base, -2.0 * m / layout.dim);
  }
  // RoPE + restore over the live doorstep addresses, shared by Q and K; the
  // masks also kill the half-density duplicates for free (1.5by). Encoding
  // only live addresses is the kill.
  rope_cos_.resize(4);
  rope_sin_.resize(4);
  rope_neg_sin_.resize(4);
  for (int lo = 0; lo < 4; lo++) {
    std::vector<Complex> cm(degree_, Complex(0.0, 0.0));
    std::vector<Complex> sm(degree_, Complex(0.0, 0.0));
    std::vector<Complex> nm(degree_, Complex(0.0, 0.0));
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        const double ang = t * theta[lo * 16 + cp];
        for (int hh = 0; hh < 16; hh++) {
          const int slot = Door0(t, lo * 16 + cp, hh);
          cm[slot] = Complex(std::cos(ang) * cfg_.restore, 0.0);
          sm[slot] = Complex(std::sin(ang) * cfg_.restore, 0.0);
          nm[slot] = Complex(-std::sin(ang) * cfg_.restore, 0.0);
        }
      }
    }
    boot_->encoder_.Encode(rope_cos_[lo], cfg_.land_level, pt_scale, cm);
    boot_->encoder_.Encode(rope_sin_[lo], cfg_.land_level, pt_scale, sm);
    boot_->encoder_.Encode(rope_neg_sin_[lo], cfg_.land_level, pt_scale, nm);
  }
  // V has no angles: its whole transport multiply is restore over the live
  // addresses, one plaintext for every ciphertext (the live-address SET does
  // not depend on the channel group).
  {
    std::vector<Complex> km(degree_, Complex(0.0, 0.0));
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        for (int hh = 0; hh < 16; hh++) {
          km[Door0(t, cp, hh)] = Complex(cfg_.restore, 0.0);
        }
      }
    }
    boot_->encoder_.Encode(kill_, cfg_.land_level, pt_scale, km);
  }
  // K's cross selectors (the token's top three bits, post-exchange slot bits
  // [9..7]) and V's call selector (bit 7 alone).
  const double cr_scale = boot_->param_.GetRescalePrimeProd(cfg_.cross_level);
  cross_sel_.resize(8);
  for (int v = 0; v < 8; v++) {
    std::vector<Complex> msg(degree_, Complex(0.0, 0.0));
    for (int s = 0; s < degree_; s++) {
      if (((s >> 7) & 7) == v) msg[s] = Complex(1.0, 0.0);
    }
    boot_->encoder_.Encode(cross_sel_[v], cfg_.cross_level, cr_scale, msg);
  }
  call_sel_.resize(2);
  for (int call = 0; call < 2; call++) {
    std::vector<Complex> msg(degree_, Complex(0.0, 0.0));
    for (int s = 0; s < degree_; s++) {
      if (((s >> 7) & 1) == call) msg[s] = Complex(1.0, 0.0);
    }
    boot_->encoder_.Encode(call_sel_[call], cfg_.cross_level, cr_scale, msg);
  }
}

template <typename word>
void CiSinCAttention<word>::AddRequiredRotations(EvkRequest &req) const {
  exchange_[0].AddRequiredRotations(req);
  req.AddRequest(degree_ - window_, cfg_.exchange_level - 1);
  req.AddRequest(degree_ - 128, cfg_.exchange_level);  // the merge
  std::set<int> idxs;
  for (int u = 0; u < 4; u++) {
    for (int v = 0; v < 8; v++) {
      const int rot = (v - u) * 128;
      if (rot != 0) idxs.insert((rot % degree_ + degree_) % degree_);
    }
  }
  idxs.insert(128);  // V's odd-call alignment
  for (int idx : idxs) req.AddRequest(idx, cfg_.cross_level);
  // The softmax reduction tree. It runs on the BOOTED scores, a few levels
  // below GetTopLevel() and far above the transport -- a rotation key
  // serves the levels below the one it was made at, so these must be asked
  // for at the top, NOT at the transport's level (which is a dial and may
  // sit at 8). The bootstrap's own CtS/StC keys already hold these
  // indices, so this is usually a no-op.
  const auto &layout = ccmm_.GetLayout();
  for (int t = 0; t < 4; t++) {
    req.AddRequest((num_slots_ / layout.rank) << t, GetTopLevel());
  }
}

template <typename word>
void CiSinCAttention<word>::AddSwitchRotations(EvkRequest &req) const {
  conv_q_->AddRequiredRotations(req);
  conv_k_->AddRequiredRotations(req);
  conv_pv_->AddRequiredRotations(req);
}

template <typename word>
void CiSinCAttention<word>::RopeAndKill(std::vector<Ct> &a_cts,
                                        std::vector<Ct> &b_cts,
                                        bool with_angles) const {
  if (!with_angles) {
    // V: restore + kill alone -- the sin terms are zero, so the pair
    // arithmetic collapses to one mask multiply per ciphertext.
    for (auto *cts : {&a_cts, &b_cts}) {
      for (auto &ct : *cts) {
        Ct t;
        boot_->Mult(t, ct, kill_);
        boot_->Rescale(ct, t);
      }
    }
    return;
  }
  for (int lo = 0; lo < 4; lo++) {
    for (int fam = 0; fam < 2; fam++) {
      std::vector<Ct> &cts = (fam == 0) ? a_cts : b_cts;
      Ct &lo_ct = cts[lo];
      Ct &hi_ct = cts[lo + 4];
      Ct aa, bb, dd;
      boot_->Mult(aa, lo_ct, rope_cos_[lo]);
      boot_->Mult(bb, hi_ct, rope_neg_sin_[lo]);
      boot_->Add(aa, aa, bb);
      boot_->Mult(bb, hi_ct, rope_cos_[lo]);
      boot_->Mult(dd, lo_ct, rope_sin_[lo]);
      boot_->Add(bb, bb, dd);
      boot_->Rescale(lo_ct, aa);
      boot_->Rescale(hi_ct, bb);
    }
  }
}

template <typename word>
void CiSinCAttention<word>::Merge(std::vector<Ct> &a_cts,
                                  std::vector<Ct> &b_cts,
                                  const EvkMap<word> &evk) const {
  const int merge_idx = degree_ - 128;
  for (size_t l = 0; l < a_cts.size(); l++) {
    // Onto the dial's level first (see the constructor): the drop is free
    // and everything after it -- this rotation, the exchange's whole
    // BSGS, K's cross -- is a key switch that no longer carries the
    // landing's limbs. The scale is untouched by LevelDown, and gamma's
    // fold is stated against cross_level, so the arithmetic is unchanged.
    const int here = boot_->param_.NPToLevel(a_cts[l].GetNP());
    if (here > cfg_.exchange_level) {
      Ct da, db;
      boot_->LevelDown(da, a_cts[l], cfg_.exchange_level);
      boot_->LevelDown(db, b_cts[l], cfg_.exchange_level);
      a_cts[l] = std::move(da);
      b_cts[l] = std::move(db);
    }
    Ct moved;
    boot_->HRot(moved, b_cts[l], evk.GetRotationKey(merge_idx), merge_idx);
    boot_->Add(a_cts[l], a_cts[l], moved);
  }
}

template <typename word>
void CiSinCAttention<word>::ExchangeAll(std::vector<Ct> &cts,
                                        const EvkMap<word> &evk) const {
  const int window_back = degree_ - window_;
  for (auto &ct : cts) {
    Ct shifted, swapped;
    exchange_[0].Evaluate(boot_, shifted, ct, evk);
    boot_->HRot(swapped, shifted, evk.GetRotationKey(window_back),
                window_back);
    ct = std::move(swapped);
  }
}

template <typename word>
std::vector<Ciphertext<word>> CiSinCAttention<word>::Cross(
    const std::vector<Ct> &k_cts, int call, const EvkMap<word> &evk) const {
  const auto &layout = ccmm_.GetLayout();
  std::vector<Ct> out(layout.num_cts);
  for (int t_hi = 0; t_hi < layout.num_cts; t_hi++) {
    const int v = BitRev(t_hi, 3);
    Ct acc;
    bool first = true;
    for (int l = call * 4; l < call * 4 + 4; l++) {
      Ct piece;
      boot_->Mult(piece, k_cts[l], cross_sel_[v]);
      const int rot = (v - l % 4) * 128;
      if (rot != 0) {
        const int idx = (rot % degree_ + degree_) % degree_;
        Ct moved;
        boot_->HRot(moved, piece, evk.GetRotationKey(idx), idx);
        piece = std::move(moved);
      }
      if (first) {
        acc = std::move(piece);
        first = false;
      } else {
        boot_->Add(acc, acc, piece);
      }
    }
    boot_->Rescale(out[t_hi], acc);
  }
  return out;
}

template <typename word>
std::vector<Ciphertext<word>> CiSinCAttention<word>::VCall(
    const std::vector<Ct> &v_cts, int call, const EvkMap<word> &evk) const {
  const auto &layout = ccmm_.GetLayout();
  std::vector<Ct> out(layout.num_cts);
  for (int l = 0; l < layout.num_cts; l++) {
    Ct piece;
    boot_->Mult(piece, v_cts[l], call_sel_[call]);
    if (call == 1) {
      Ct moved;
      boot_->HRot(moved, piece, evk.GetRotationKey(128), 128);
      piece = std::move(moved);
    }
    boot_->Rescale(out[l], piece);
  }
  return out;
}

template <typename word>
void CiSinCAttention<word>::Convert(const CiSinCConverter<word> &conv,
                                    std::vector<Ct> &cts,
                                    const EvkMap<word> &evk) const {
  for (auto &ct : cts) {
    Ct at_level;
    boot_->LevelDown(at_level, ct, cfg_.forward_level);
    Ct sinc;
    conv.SlotToSinC(switch_ctx_, sinc, at_level, evk);
    ct = std::move(sinc);
  }
}

template <typename word>
void CiSinCAttention<word>::ChainAndReturn(std::vector<Ct> &res,
                                           std::vector<Ct> &lhs_sinc,
                                           const std::vector<Ct> &rhs_source,
                                           bool rhs_is_k,
                                           const Keys &keys) const {
  const auto &layout = ccmm_.GetLayout();
  std::vector<Ct> acc;
  for (int call = 0; call < 2; call++) {
    std::vector<Ct> rhs = rhs_is_k ? Cross(rhs_source, call, *keys.boot)
                                   : VCall(rhs_source, call, *keys.boot);
    // V rides Q's converter (1.5cb): same block function of (token, channel).
    Convert(rhs_is_k ? *conv_k_ : *conv_q_, rhs, *keys.swtch);
    std::vector<Ct> lhs;
    for (int i = 0; i < layout.num_cts / 2; i++) {
      lhs.push_back(std::move(lhs_sinc[call * layout.num_cts / 2 + i]));
    }
    std::vector<Ct> part;
    ccmm_.Multiply(part, lhs, rhs, *keys.ring_switch,
                   *keys.inverse_ring_switch, *keys.lifted);
    if (call == 0) {
      acc = std::move(part);
    } else {
      for (int bi = 0; bi < layout.num_cts; bi++) {
        switch_ctx_->Add(acc[bi], acc[bi], part[bi]);
      }
    }
  }
  res.clear();
  res.resize(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    conv_pv_->SinCToSlot(switch_ctx_, res[bi], acc[bi], *keys.swtch);
  }
}

template <typename word>
void CiSinCAttention<word>::Scores(std::vector<Ct> &res, std::vector<Ct> &q_a,
                                   std::vector<Ct> &q_b, std::vector<Ct> &k_a,
                                   std::vector<Ct> &k_b,
                                   const Keys &keys) const {
  RopeAndKill(q_a, q_b, /*with_angles=*/true);
  RopeAndKill(k_a, k_b, /*with_angles=*/true);
  Merge(q_a, q_b, *keys.boot);
  Merge(k_a, k_b, *keys.boot);
  ExchangeAll(q_a, *keys.boot);
  ExchangeAll(k_a, *keys.boot);
  Convert(*conv_q_, q_a, *keys.swtch);
  ChainAndReturn(res, q_a, k_a, /*rhs_is_k=*/true, keys);
  if (cfg_.verbose) {
    std::cout << "CiSinCAttention::Scores: out @"
              << boot_->param_.NPToLevel(res[0].GetNP()) << ", carried "
              << res[0].GetScale() / boot_->param_.base_scale_ << std::endl;
  }
}

template <typename word>
void CiSinCAttention<word>::PrepareSoftMax(const SoftMaxCalibration &calib) {
  const auto &layout = ccmm_.GetLayout();
  calib_ = calib;
  const int top = GetTopLevel();
  exp_in_ = top - 1;
  const int exp_degree = (calib_.exp_degree > 0)
                             ? calib_.exp_degree
                             : (calib_.causal ? 7 : 9);
  // k = 1 (Cho): y = exp(m_eff (u - 1) / 4), squared later by the norm.
  const double hb = calib_.m_eff / 4.0;
  auto exp_coeffs = chebfit::Interpolate(
      [hb](double v) { return std::exp(hb * (v - 1.0)); }, exp_degree);
  const int exp_used =
      EvalPoly<word>(exp_coeffs, exp_in_, boot_->param_.GetScale(exp_in_),
                     boot_->param_.GetScale(exp_in_), true)
          .GetPolyDegree();
  exp_out_ = exp_in_ - Log2Ceil(exp_used + 1);
  polys_.clear();
  polys_.push_back(std::make_unique<EvalPoly<word>>(
      exp_coeffs, exp_in_, boot_->param_.GetScale(exp_in_),
      boot_->param_.GetScale(exp_out_), true));
  polys_[0]->Compile(boot_);

  // The causal mask spends one level on y (1.5cc); exp's shorter fit is
  // what freed it, so the downstream walk is unchanged either way.
  sq_level_ = (calib_.causal ? exp_out_ - 1 : exp_out_) - 1;
  poly_in_ = sq_level_ - 1;
  const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
  const double aff_b = 0.5 * (calib_.norm_hi + calib_.norm_lo);
  auto inv_coeffs = chebfit::Interpolate(
      [aff_a, aff_b](double v) { return 1.0 / std::sqrt(aff_a * v + aff_b); },
      calib_.inv_degree);
  const int inv_used =
      EvalPoly<word>(inv_coeffs, poly_in_, boot_->param_.GetScale(poly_in_),
                     boot_->param_.GetScale(poly_in_), true)
          .GetPolyDegree();
  const int inv_out = poly_in_ - Log2Ceil(inv_used + 1);
  AssertTrue(inv_out - 2 >= cfg_.forward_level,
             "CiSinCAttention: the softmax walk overspends its levels; P "
             "would land below forward_level");
  polys_.push_back(std::make_unique<EvalPoly<word>>(
      inv_coeffs, poly_in_, boot_->param_.GetScale(poly_in_),
      boot_->param_.GetScale(inv_out), true));
  polys_[1]->Compile(boot_);

  causal_a0_.clear();
  causal_mask_.clear();
  if (calib_.causal) {
    AssertTrue(static_cast<int>(calib_.row_shift.size()) == layout.lanes &&
                   static_cast<int>(calib_.row_shift[0].size()) == layout.dim,
               "CiSinCAttention: causal calibration needs a [lanes][dim] "
               "row_shift table");
    const bool use_norm = !calib_.row_norm.empty();
    if (use_norm) {
      AssertTrue(static_cast<int>(calib_.row_norm.size()) == layout.lanes &&
                     static_cast<int>(calib_.row_norm[0].size()) ==
                         layout.dim,
                 "CiSinCAttention: row_norm must be a [lanes][dim] table");
    }
    // The per-row shift is the interval answer (1.5cc): each row's largest
    // live y is exactly 1, so the norm interval is [1, live row sum] by
    // construction. Masked slots keep the global shift so u stays in the
    // fit domain; the mask kills their y anyway.
    const double u_scale = boot_->param_.GetScale(top) *
                           boot_->param_.GetScale(top) /
                           boot_->param_.GetRescalePrimeProd(top);
    causal_a0_.resize(layout.num_cts);
    causal_mask_.resize(layout.num_cts);
    for (int bi = 0; bi < layout.num_cts; bi++) {
      std::vector<Complex> a0_msg(degree_, Complex(0.0, 0.0));
      std::vector<Complex> mask_msg(degree_, Complex(0.0, 0.0));
      for (int row = 0; row < layout.dim; row++) {
        for (int j = 0; j < layout.rank; j++) {
          const int column = bi * layout.rank + j;
          for (int lane = 0; lane < layout.lanes; lane++) {
            int ct_idx, slot, copy_slot;
            layout.LocateSlot(row, column, lane, ct_idx, slot, copy_slot);
            const bool live = column <= row;
            const double shift =
                live ? calib_.row_shift[lane][row] : calib_.shift;
            a0_msg[slot] =
                Complex(1.0 - 2.0 * shift / calib_.span, 0.0);
            double mval = 0.0;
            if (live) {
              if (use_norm) {
                const double est = calib_.row_norm[lane][row];
                AssertTrue(est > 1e-9,
                           "CiSinCAttention: row_norm estimates must be "
                           "positive");
                mval = 1.0 / std::sqrt(est);
              } else {
                mval = 1.0;
              }
            }
            mask_msg[slot] = Complex(mval, 0.0);
          }
        }
      }
      boot_->encoder_.Encode(causal_a0_[bi], exp_in_, u_scale, a0_msg);
      boot_->encoder_.Encode(causal_mask_[bi], exp_out_,
                             boot_->param_.GetScale(exp_out_), mask_msg);
    }
  }
  softmax_ready_ = true;
  if (cfg_.verbose) {
    std::cout << "CiSinCAttention::PrepareSoftMax: exp deg " << exp_used
              << " @" << exp_in_ << ".." << exp_out_
              << (calib_.causal ? ", causal mask @" : ", no mask, sq @")
              << (calib_.causal ? exp_out_ - 1 : sq_level_) << ", invsqrt deg "
              << inv_used << " @" << poly_in_ << ".." << inv_out << std::endl;
  }
}

template <typename word>
void CiSinCAttention<word>::SoftMax(std::vector<Ct> &P,
                                    const std::vector<Ct> &scores,
                                    double carried,
                                    const EvkMap<word> &evk) const {
  AssertTrue(softmax_ready_, "CiSinCAttention: call PrepareSoftMax first");
  const auto &layout = ccmm_.GetLayout();
  const int top = GetTopLevel();
  AssertTrue(boot_->param_.NPToLevel(scores[0].GetNP()) == top,
             "CiSinCAttention: SoftMax expects its input at GetTopLevel()");
  const auto &mult_key = evk.GetMultiplicationKey();

  // Affine onto the fit domain; carried divides out here (1.5bu).
  const double a1 = 2.0 / (calib_.span * carried);
  std::vector<Ct> y(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Constant<word> c1;
    boot_->encoder_.EncodeConstant(c1, top, boot_->param_.GetScale(top), a1);
    Ct t1, u_ct;
    boot_->Mult(t1, scores[bi], c1);
    boot_->Rescale(u_ct, t1);
    if (calib_.causal) {
      boot_->Add(u_ct, u_ct, causal_a0_[bi]);
    } else {
      Constant<word> c0;
      boot_->encoder_.EncodeConstant(
          c0, exp_in_, u_ct.GetScale(),
          1.0 - 2.0 * calib_.shift / calib_.span);
      boot_->Add(u_ct, u_ct, c0);
    }
    Ct y_full;
    polys_[0]->Evaluate(boot_, y_full, u_ct, mult_key);
    if (calib_.causal) {
      Ct t2;
      boot_->Mult(t2, y_full, causal_mask_[bi]);
      boot_->Rescale(y[bi], t2);
    } else {
      y[bi] = std::move(y_full);
    }
  }

  // The Euclidean norm: the squared ciphertexts summed, then the top-field
  // rotate-and-add tree -- exact at every slot, no mask (1.5bv).
  Ct sq, term, rotated;
  boot_->HMult(sq, y[0], y[0], mult_key);
  for (int bi = 1; bi < layout.num_cts; bi++) {
    boot_->HMult(term, y[bi], y[bi], mult_key);
    boot_->Add(sq, sq, term);
  }
  for (int d : reduce_dist_) {
    boot_->HRotAdd(rotated, sq, sq, evk.GetRotationKey(d), d);
    boot_->Copy(sq, rotated);
  }
  {
    const double aff_a = 0.5 * (calib_.norm_hi - calib_.norm_lo);
    const double aff_b = 0.5 * (calib_.norm_hi + calib_.norm_lo);
    Constant<word> inv_a;
    boot_->encoder_.EncodeConstant(inv_a, sq_level_,
                                   boot_->param_.GetScale(sq_level_),
                                   1.0 / aff_a);
    Ct scaled;
    boot_->Mult(scaled, sq, inv_a);
    boot_->Rescale(sq, scaled);
    Constant<word> shift;
    boot_->encoder_.EncodeConstant(shift, poly_in_, sq.GetScale(),
                                   -aff_b / aff_a);
    boot_->Add(sq, sq, shift);
  }
  Ct r;
  polys_[1]->Evaluate(boot_, r, sq, mult_key);
  const int meet = boot_->param_.NPToLevel(r.GetNP());
  P.clear();
  P.resize(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    Ct levelled, prod;
    boot_->LevelDown(levelled, y[bi], meet);
    boot_->HMult(prod, levelled, r, mult_key);
    boot_->HMult(P[bi], prod, prod, mult_key);
  }
  const int p_level = boot_->param_.NPToLevel(P[0].GetNP());
  AssertTrue(p_level >= cfg_.forward_level,
             "CiSinCAttention: the softmax left P below forward_level");
  if (p_level > cfg_.forward_level) {
    for (int bi = 0; bi < layout.num_cts; bi++) {
      Ct down;
      boot_->LevelDown(down, P[bi], cfg_.forward_level);
      P[bi] = std::move(down);
    }
  }
}

template <typename word>
void CiSinCAttention<word>::Values(std::vector<Ct> &res, std::vector<Ct> &p,
                                   std::vector<Ct> &v_a, std::vector<Ct> &v_b,
                                   const Keys &keys) const {
  const auto &layout = ccmm_.GetLayout();
  AssertTrue(boot_->param_.NPToLevel(p[0].GetNP()) == cfg_.forward_level,
             "CiSinCAttention: Values expects P at forward_level");
  RopeAndKill(v_a, v_b, /*with_angles=*/false);
  Merge(v_a, v_b, *keys.boot);
  ExchangeAll(v_a, *keys.boot);
  // P's own ciphertexts are the two calls' lhs halves verbatim (1.5bw).
  std::vector<Ct> p_sinc(layout.num_cts);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    conv_pv_->SlotToSinC(switch_ctx_, p_sinc[bi], p[bi], *keys.swtch);
  }
  ChainAndReturn(res, p_sinc, v_a, /*rhs_is_k=*/false, keys);
}

template class CiSinCAttention<uint32_t>;
template class CiSinCAttention<uint64_t>;

}  // namespace cheddar
