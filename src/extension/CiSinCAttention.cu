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
#include "extension/Profile.h"
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
    ConstContextPtr<word> lifted_ctx, const Config &cfg,
    std::shared_ptr<const BootContext<word>> tower)
    : boot_{std::move(boot)},
      tower_{std::move(tower)},
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
  // The descent spends one level (a converter) or two (the tower forward's
  // inner and outer phases); K's standalone premap sits one above the
  // forward, still below the cross's output.
  const int descent_levels = cfg_.fused ? 2 : 1;
  AssertTrue(cfg_.land_level > cfg_.exchange_level &&
                 cfg_.exchange_level > cfg_.cross_level &&
                 cfg_.cross_level - 1 >= cfg_.forward_level + (cfg_.fused ? 1 : 0) &&
                 cfg_.forward_level == cfg_.chain_level + descent_levels &&
                 cfg_.chain_level == cfg_.inverse_level + 1,
             "CiSinCAttention: the level ladder is land > exchange > cross "
             "and forward = chain + descent = inverse + descent + 1");

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

  if (cfg_.fused) {
    // THE FUSED CONVERSIONS (Doing.md 3.16). The three descents are tower
    // forwards on the switching ring -- Q's premap folded into its first
    // phase (441 diagonals at the Llama shape), K's as a phase of its own
    // (479; folded it would fill the lattice), P's plain (255) -- and the
    // return is the tower ring's HalfBoot with its CtS' plus the lane
    // prefix. The prefix carries two folds, as the ordinary track's
    // (`SinCAttention::BuildPrefixes`): the inverse of the tower's message
    // ratio, so the scores come back as a `Boot` would have left them
    // (`carried * m`, the softmax dividing `carried` out), and the
    // plaintext scale that lands them CANONICAL at GetTopLevel() from the
    // tower's own EvalMod end scale.
    AssertTrue(tower_ != nullptr,
               "CiSinCAttention: the fused return needs the tower ring's "
               "BootContext");
    AssertTrue(tower_->GetStCInputScale() > 0.0 &&
                   tower_->GetMessageRatio() != 0.0,
               "CiSinCAttention: PrepareEvalMod must run on the tower ring "
               "before construction");
    // What crosses between the rings: the chain's output at level 0 into
    // the tower's HalfBoot, and the prefix's output at the tower's landing
    // less one back onto the leg ring for the softmax -- so the two must
    // agree on every level up to there.
    const int prefix_out = tower_->GetBootParameter().GetEvalModEndLevel() - 1;
    for (int L = 0; L <= prefix_out; L++) {
      AssertTrue(boot_->param_.GetPrimeVector(boot_->param_.LevelToNP(L)) ==
                     tower_->param_.GetPrimeVector(tower_->param_.LevelToNP(L)),
                 "CiSinCAttention: the tower ring must share the leg ring's "
                 "levels up to the prefix's landing (" + std::to_string(L) +
                     ")");
    }
    basis_ = std::make_unique<CiSinCBasis<word>>(degree_, layout.small_degree,
                                                 cfg_.sub_degree);
    basis_->PrepareForward(switch_ctx_, "q", cfg_.forward_level, &pre_q_);
    MemoryPool::Report("attn ctor: + the tower forward q (premap folded)");
    basis_->PrepareForward(switch_ctx_, "k", cfg_.forward_level + 1, &pre_k_,
                           typename CiSinCBasis<word>::Phases(),
                           /*fold_premap=*/false);
    MemoryPool::Report("attn ctor: + the tower forward k (premap standalone)");
    basis_->PrepareForward(switch_ctx_, "p", cfg_.forward_level);
    MemoryPool::Report("attn ctor: + the tower forward p");
    const auto &tp = tower_->GetBootParameter();
    basis_->PrepareCtS(tower_, tp.GetCtSStartLevel(),
                       num_slots_ * tower_->GetCtSConst());
    MemoryPool::Report("attn ctor: + the tower CtS' (outer/inner/lane)");
    const int prefix_level = tp.GetEvalModEndLevel();
    const double target = boot_->param_.GetScale(prefix_level - 1);
    const double pt_scale = target *
                            boot_->param_.GetRescalePrimeProd(prefix_level) /
                            tower_->GetStCInputScale();
    basis_->PreparePrefix(tower_, prefix_level,
                          /*constant=*/1.0 / tower_->GetMessageRatio(),
                          pt_scale);
    MemoryPool::Report("attn ctor: + the lane prefix");
    if (cfg_.verbose) {
      std::cout << "CiSinCAttention: fused conversions -- forwards at "
                << cfg_.forward_level << " (q), " << cfg_.forward_level + 1
                << " (k), " << cfg_.forward_level << " (p); the tower CtS' at "
                << tp.GetCtSStartLevel() << ", the prefix at " << prefix_level
                << " landing " << GetTopLevel() << " at scale 2^"
                << std::log2(target) << " from the tower's 2^"
                << std::log2(tower_->GetStCInputScale()) << ", ratio "
                << tower_->GetMessageRatio() << std::endl;
    }
  } else {
    conv_q_ = MakeConverter("q", /*inverse_level=*/-1, layout, &pre_q_);
    conv_k_ = MakeConverter("k", /*inverse_level=*/-1, layout, &pre_k_);
    conv_pv_ = MakeConverter("pv", cfg_.inverse_level, layout,
                             /*premap=*/nullptr);
  }

  // THE CONVERTERS' RESIDENCY (Doing.md 3.9). Each is read once a layer (pv
  // twice) and is the largest object the leg owns, so `host` takes q and k
  // off the device between uses and `host_all` pv too; the plaintexts come
  // back through `HoldConverter` for the use and go again after it.
  if (!cfg_.fused) {
    const char *env = std::getenv("CHEDDAR_CONVERTER_RESIDENCY");
    const std::string mode = (env != nullptr) ? env : "";
    converter_residency_ = (mode == "host") ? 1 : (mode == "host_all") ? 2 : 0;
    if (converter_residency_ != 0) {
      size_t bytes = 0;
      for (const auto *c : {conv_q_.get(), conv_k_.get(), conv_pv_.get()}) {
        if (!Staged(*c)) continue;
        for (const auto *t : {c->GetForward(), c->GetInverse()}) {
          if (t != nullptr) bytes += t->PlaintextBytes();
        }
        HoldConverter(*c, /*on_device=*/false);
      }
      if (cfg_.verbose) {
        std::cout << "CiSinCAttention: converter residency " << mode << ", "
                  << (bytes >> 20) << " MiB of plaintexts staged to host memory"
                  << std::endl;
      }
    }
  }

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
  // `landing_scale`: the images' declared scale on arrival against the one
  // this leg's constants were stated for (see Config). The ratio is a power
  // of two between two EvalMod ladders of this family (exactly 4 for the
  // K = 32 landing ring), so the fold is exact.
  const double stc_here = boot_->GetStCInputScale();
  const double arrival =
      cfg_.landing_scale > 0.0 ? cfg_.landing_scale : stc_here;
  const double pt_scale = boot_->param_.GetRescalePrimeProd(cfg_.land_level) *
                          gamma_ * (stc_here / arrival);
  if (cfg_.verbose && arrival != stc_here) {
    std::cout << "CiSinCAttention: images arrive at scale 2^"
              << std::log2(arrival) << " against the leg's 2^"
              << std::log2(stc_here) << "; the RoPE masks carry the ratio "
              << (stc_here / arrival) << std::endl;
  }
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
  // The heads a mask covers: the a family's 16 (the b family arrives by the
  // merge), or all 32 of a dense image.
  const int mask_heads = cfg_.dense_images ? layout.lanes : 16;
  for (int lo = 0; lo < 4; lo++) {
    std::vector<Complex> cm(degree_, Complex(0.0, 0.0));
    std::vector<Complex> sm(degree_, Complex(0.0, 0.0));
    std::vector<Complex> nm(degree_, Complex(0.0, 0.0));
    for (int t = 0; t < layout.dim; t++) {
      for (int cp = 0; cp < 16; cp++) {
        const double ang = t * theta[lo * 16 + cp];
        for (int hh = 0; hh < mask_heads; hh++) {
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
        for (int hh = 0; hh < mask_heads; hh++) {
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
  if (cfg_.fused) {
    basis_->AddForwardRotations(req);
    return;
  }
  conv_q_->AddRequiredRotations(req);
  conv_k_->AddRequiredRotations(req);
  conv_pv_->AddRequiredRotations(req);
}

template <typename word>
void CiSinCAttention<word>::AddTowerRotations(EvkRequest &req) const {
  AssertTrue(cfg_.fused, "CiSinCAttention: no tower ring without fused");
  basis_->AddCtSRotations(req);
  basis_->AddPrefixRotations(req);
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
void CiSinCAttention<word>::Rope(std::vector<Ct> &cts, bool with_angles) const {
  AssertTrue(cfg_.dense_images && static_cast<int>(cts.size()) == 8,
             "CiSinCAttention::Rope: eight dense images");
  if (!with_angles) {
    // V: the restore alone (the mask is `restore` at every doorstep slot).
    for (auto &ct : cts) {
      Ct t;
      boot_->Mult(t, ct, kill_);
      boot_->Rescale(ct, t);
    }
  } else {
    for (int lo = 0; lo < 4; lo++) {
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
  // Onto the exchange's level, as `Merge` does for the banded images -- for
  // V as much as for Q and K: the first dense run returned from the V branch
  // above this loop and handed the exchange level-12 images ("Hoist: input
  // level mismatch -- 14 main + 2 terminal against level 8").
  for (size_t i = 0; i < cts.size(); i++) {
    Ct &ct = cts[i];
    const NPInfo before = ct.GetNP();
    const int here = boot_->param_.NPToLevel(before);
    if (here > cfg_.exchange_level) {
      Ct down;
      boot_->LevelDown(down, ct, cfg_.exchange_level);
      ct = std::move(down);
    }
    const NPInfo after = ct.GetNP();
    AssertTrue(boot_->param_.NPToLevel(after) == cfg_.exchange_level,
               "CiSinCAttention::Rope: image " + std::to_string(i) +
                   " left at (" + std::to_string(after.num_main_) + " main + " +
                   std::to_string(after.num_ter_) + " terminal), level " +
                   std::to_string(boot_->param_.NPToLevel(after)) +
                   ", from (" + std::to_string(before.num_main_) + " + " +
                   std::to_string(before.num_ter_) + "), level " +
                   std::to_string(here) + "; the exchange is at " +
                   std::to_string(cfg_.exchange_level));
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
    {
      const NPInfo np = ct.GetNP();
      AssertTrue(boot_->param_.NPToLevel(np) == cfg_.exchange_level,
                 "CiSinCAttention::ExchangeAll: an image arrives at (" +
                     std::to_string(np.num_main_) + " main + " +
                     std::to_string(np.num_ter_) + " terminal), level " +
                     std::to_string(boot_->param_.NPToLevel(np)) +
                     ", not the exchange's " +
                     std::to_string(cfg_.exchange_level));
    }
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
bool CiSinCAttention<word>::Staged(const CiSinCConverter<word> &conv) const {
  if (converter_residency_ == 0) return false;
  if (&conv == conv_pv_.get()) return converter_residency_ == 2;
  return true;
}

template <typename word>
void CiSinCAttention<word>::HoldConverter(const CiSinCConverter<word> &conv,
                                          bool on_device) const {
  if (!Staged(conv)) return;
  NvtxScope _n(on_device ? "attn: stage a converter" : "attn: unstage a converter");
  for (const auto *t : {conv.GetForward(), conv.GetInverse()}) {
    if (t == nullptr) continue;
    if (on_device) {
      t->Stage();
    } else {
      t->Unstage();
    }
  }
}

template <typename word>
void CiSinCAttention<word>::Convert(const std::string &which,
                                    std::vector<Ct> &cts,
                                    const EvkMap<word> &evk) const {
  if (cfg_.fused) {
    const std::string name = (which == "pv") ? "p" : which;
    const int level = basis_->GetForwardLevel(name);
    for (auto &ct : cts) {
      if (boot_->param_.NPToLevel(ct.GetNP()) > level) {
        Ct at_level;
        boot_->LevelDown(at_level, ct, level);
        ct = std::move(at_level);
      }
      Ct sinc;
      basis_->Forward(name, sinc, ct, evk);
      ct = std::move(sinc);
    }
    return;
  }
  const CiSinCConverter<word> &conv =
      (which == "q") ? *conv_q_ : (which == "k") ? *conv_k_ : *conv_pv_;
  HoldConverter(conv, /*on_device=*/true);
  for (auto &ct : cts) {
    if (boot_->param_.NPToLevel(ct.GetNP()) > cfg_.forward_level) {
      Ct at_level;
      boot_->LevelDown(at_level, ct, cfg_.forward_level);
      ct = std::move(at_level);
    }
    Ct sinc;
    conv.SlotToSinC(switch_ctx_, sinc, ct, evk);
    ct = std::move(sinc);
  }
  HoldConverter(conv, /*on_device=*/false);
}

template <typename word>
void CiSinCAttention<word>::ChainAndReturn(std::vector<Ct> &res,
                                           std::vector<Ct> &lhs_sinc,
                                           const std::vector<Ct> &rhs_source,
                                           bool rhs_is_k, const Keys &keys,
                                           double *carried) const {
  const auto &layout = ccmm_.GetLayout();
  std::vector<Ct> acc;
  for (int call = 0; call < 2; call++) {
    std::vector<Ct> rhs = rhs_is_k ? Cross(rhs_source, call, *keys.boot)
                                   : VCall(rhs_source, call, *keys.boot);
    // V rides Q's converter (1.5cb): same block function of (token, channel).
    Convert(rhs_is_k ? "k" : "q", rhs, *keys.swtch);
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
  // The chain's message factor (1.5bu): its output scale over the base
  // scale, which the bootstrap reads its input against.
  if (carried != nullptr) {
    *carried = acc[0].GetScale() / boot_->param_.base_scale_;
  }
  if (cfg_.fused) {
    AssertTrue(keys.tower != nullptr,
               "CiSinCAttention: the fused return needs Keys::tower");
    for (int bi = 0; bi < layout.num_cts; bi++) {
      Ct half;
      tower_->HalfBootTower(half, acc[bi], *keys.tower, *basis_);
      acc[bi] = Ct{};
      basis_->Prefix(res[bi], half, *keys.tower);
    }
    return;
  }
  HoldConverter(*conv_pv_, /*on_device=*/true);
  for (int bi = 0; bi < layout.num_cts; bi++) {
    conv_pv_->SinCToSlot(switch_ctx_, res[bi], acc[bi], *keys.swtch);
  }
  HoldConverter(*conv_pv_, /*on_device=*/false);
}

template <typename word>
void CiSinCAttention<word>::Scores(std::vector<Ct> &res, std::vector<Ct> &q,
                                   std::vector<Ct> &k, const Keys &keys,
                                   double *carried) const {
  Rope(q, /*with_angles=*/true);
  Rope(k, /*with_angles=*/true);
  ExchangeAll(q, *keys.boot);
  ExchangeAll(k, *keys.boot);
  Convert("q", q, *keys.swtch);
  double f = 0.0;
  ChainAndReturn(res, q, k, /*rhs_is_k=*/true, keys, &f);
  if (carried != nullptr) *carried = f;
  if (cfg_.verbose) {
    std::cout << "CiSinCAttention::Scores (dense): out @"
              << boot_->param_.NPToLevel(res[0].GetNP()) << ", carried " << f
              << std::endl;
  }
}

template <typename word>
void CiSinCAttention<word>::Values(std::vector<Ct> &res, std::vector<Ct> &p,
                                   std::vector<Ct> &v, const Keys &keys,
                                   double *carried) const {
  AssertTrue(boot_->param_.NPToLevel(p[0].GetNP()) == cfg_.forward_level,
             "CiSinCAttention: Values expects P at forward_level");
  Rope(v, /*with_angles=*/false);
  ExchangeAll(v, *keys.boot);
  // P is read, not consumed: its descent works on copies.
  std::vector<Ct> p_sinc(p.size());
  for (size_t bi = 0; bi < p.size(); bi++) boot_->Copy(p_sinc[bi], p[bi]);
  Convert("pv", p_sinc, *keys.swtch);
  ChainAndReturn(res, p_sinc, v, /*rhs_is_k=*/false, keys, carried);
}

template <typename word>
void CiSinCAttention<word>::Scores(std::vector<Ct> &res, std::vector<Ct> &q_a,
                                   std::vector<Ct> &q_b, std::vector<Ct> &k_a,
                                   std::vector<Ct> &k_b, const Keys &keys,
                                   double *carried) const {
  AssertTrue(!cfg_.dense_images,
             "CiSinCAttention::Scores: this leg was built for dense images; "
             "call the 8-ciphertext form");
  RopeAndKill(q_a, q_b, /*with_angles=*/true);
  RopeAndKill(k_a, k_b, /*with_angles=*/true);
  Merge(q_a, q_b, *keys.boot);
  Merge(k_a, k_b, *keys.boot);
  ExchangeAll(q_a, *keys.boot);
  ExchangeAll(k_a, *keys.boot);
  Convert("q", q_a, *keys.swtch);
  double f = 0.0;
  ChainAndReturn(res, q_a, k_a, /*rhs_is_k=*/true, keys, &f);
  if (carried != nullptr) *carried = f;
  if (cfg_.verbose) {
    std::cout << "CiSinCAttention::Scores: out @"
              << boot_->param_.NPToLevel(res[0].GetNP()) << ", carried " << f
              << std::endl;
  }
}

namespace {

//! Max error of the degree-`deg` Chebyshev interpolant of `exp(hb (v-1))` on
//! [-1, 1], on a dense grid. Cheap enough to call a few times at setup.
double ExpFitError(double hb, int deg) {
  const auto c = chebfit::Interpolate(
      [hb](double v) { return std::exp(hb * (v - 1.0)); }, deg);
  double worst = 0.0;
  const int kGrid = 4001;
  for (int i = 0; i < kGrid; i++) {
    const double v = -1.0 + 2.0 * i / (kGrid - 1);
    double b0 = 0.0, b1 = 0.0;
    for (size_t j = c.size() - 1; j > 0; j--) {
      const double t = 2.0 * v * b0 - b1 + c[j];
      b1 = b0;
      b0 = t;
    }
    worst = std::max(worst, std::abs(v * b0 - b1 + c[0] -
                                     std::exp(hb * (v - 1.0))));
  }
  return worst;
}

//! The degree that reaches sixteen bits, CAPPED AT 15 -- and the cap is the
//! level budget, not the fit. Counting the walk: `exp_in = top - 1`, the
//! polynomial spends `ceil(log2(deg+1))`, the causal mask one, the square one,
//! the affine one and the inverse square root `ceil(log2(inv_deg+1))`, and
//! `PrepareSoftMax` then requires `inv_out - 2 >= forward_level`. At the
//! layer's own numbers (`top` 16, `forward_level` 3, `inv_degree` 7) degree 15
//! lands that at exactly 3 and degree 31 at 2, so 31 does not fit and would
//! have to be funded by dropping the inverse square root to degree 3 -- which
//! is 2^-13 on a window of [0.9, 1.1], against an exp fit at 15 that is
//! 2^-8.49 on the ONE layer of 32 that wants more. Neither is near stage 2's
//! 2^-5.70, so the cap costs nothing measurable and the alternative would.
int ExpDegree(double m_eff) {
  const double hb = std::max(m_eff, 0.0) / 4.0;
  for (int d : {7, 9, 15}) {
    if (ExpFitError(hb, d) < std::pow(2.0, -16.0)) return d;
  }
  return 15;
}

}  // namespace

template <typename word>
void CiSinCAttention<word>::PrepareSoftMax(const SoftMaxCalibration &calib) {
  const auto &layout = ccmm_.GetLayout();
  calib_ = calib;
  const int top = GetTopLevel();
  exp_in_ = top - 1;
  // THE DEGREE FOLLOWS m_eff, as SiLU's follows its range (1.5ea) and the
  // norm's its window (1.5ec). `exp(hb (v-1))` on [-1, 1] is entire, so there
  // is no Bernstein ellipse to read a rate off -- the Chebyshev coefficients
  // are `2 I_k(hb) e^{-hb}` and only start falling once `k > hb` -- so the
  // rule is measured rather than derived: interpolate at each candidate and
  // take the first that reaches sixteen bits, which is where 1.5cv measured
  // the circuits these fits are evaluated by. The old default (7 when causal,
  // 9 when not) was set at a correctness-width `m_eff`; 1.5cf then showed 7
  // is 3.2e-03 at a span of 24.36 and every caller has passed 15 by hand ever
  // since. At the REAL model's spans, which run 17.1 to 97.3 over the 32
  // layers, 15 is right for 31 of them and buys only 8.5 bits at layer 31.
  const int exp_degree =
      (calib_.exp_degree > 0) ? calib_.exp_degree : ExpDegree(calib_.m_eff);
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
                                   const Keys &keys, double *carried) const {
  AssertTrue(boot_->param_.NPToLevel(p[0].GetNP()) == cfg_.forward_level,
             "CiSinCAttention: Values expects P at forward_level");
  RopeAndKill(v_a, v_b, /*with_angles=*/false);
  Merge(v_a, v_b, *keys.boot);
  ExchangeAll(v_a, *keys.boot);
  // P's own ciphertexts are the two calls' lhs halves verbatim (1.5bw); P
  // is read, not consumed, so its descent works on copies.
  std::vector<Ct> p_sinc(p.size());
  for (size_t bi = 0; bi < p.size(); bi++) boot_->Copy(p_sinc[bi], p[bi]);
  Convert("pv", p_sinc, *keys.swtch);
  ChainAndReturn(res, p_sinc, v_a, /*rhs_is_k=*/false, keys, carried);
}

template class CiSinCAttention<uint32_t>;
template class CiSinCAttention<uint64_t>;

}  // namespace cheddar
