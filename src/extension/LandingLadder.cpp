#include "extension/LandingLadder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>

#include "common/Assert.h"
#include "extension/BootParameter.h"

namespace cheddar {

template <typename word>
LandingLadder<word>::LandingLadder(const LadderSpec<word> &pool) : pool_(pool) {
  AssertTrue(pool_.boot, "LandingLadder: the pool is not a boot preset");
  AssertTrue(pool_.num_cts_levels >= 2 && pool_.num_stc_levels >= 1,
             "LandingLadder: the pool needs CtS and StC levels");
  // EvalMod's level count is BootParameter's business (the polynomial's
  // depth plus the double angles); ask it rather than restate the rule.
  nem_ = BootParameter(pool_.MaxLevel(), pool_.num_cts_levels,
                       pool_.num_stc_levels, pool_.log_message_ratio, 0,
                       pool_.num_double_angle, pool_.initial_K)
             .GetNumEvalModLevels();
  const int dec = pool_.default_encryption_level;
  AssertTrue(pool_.MaxLevel() == dec + nem_ + pool_.num_cts_levels,
             "LandingLadder: the pool's top is not dec + EvalMod + CtS");
  AssertTrue(pool_.Landing() >= 0, "LandingLadder: the pool lands below 0");
  const auto [m, t] = pool_.level_config.at(dec);
  band_lo_ = m;
  for (int k = 1; k <= nem_; k++) {
    AssertTrue(pool_.level_config.at(dec + k) == std::make_pair(m + 2 * k, t),
               "LandingLadder: the pool's band is not +2 main a level at " +
                   std::to_string(dec + k));
  }
  // Every pair is the band width, or the pairs are not interchangeable and
  // the cut-and-paste below is not a valid ladder.
  width_bits_ = static_cast<int>(std::lround(pool_.RescaleBits(dec + 1)));
  for (int k = 1; k <= nem_; k++) {
    const double w = pool_.RescaleBits(dec + k);
    AssertTrue(std::abs(w - width_bits_) < 0.05,
               "LandingLadder: band level " + std::to_string(dec + k) +
                   " rescales by 2^" + std::to_string(w) +
                   ", not the band's 2^" + std::to_string(width_bits_) +
                   " -- this pool's band is not stationary");
  }
}

template <typename word>
int LandingLadder<word>::Peak(int upto) const {
  int peak = 0;
  for (int i = 0; i <= upto; i++)
    peak = std::max(peak, pool_.level_config.at(i).first);
  return peak;
}

template <typename word>
int LandingLadder<word>::NeedT(int dec) const {
  int need = 0;
  for (int i = 0; i <= dec; i++)
    need = std::max(need, pool_.level_config.at(i).second);
  return need;
}

template <typename word>
std::string LandingLadder<word>::CtSPlan(int t, int spares, int need_t) const {
  const int num_ter = static_cast<int>(pool_.ter_primes.size());
  const int num_cts = pool_.num_cts_levels;
  auto ter_pair_bits = [&](int tt) {
    return std::log2(double(pool_.ter_primes.at(tt))) +
           std::log2(double(pool_.ter_primes.at(tt + 1)));
  };
  std::string plan;
  // The 2^42 rule's order -- a terminal triple (the pool's own CtS, ~2^72),
  // a pair of unused compute mains (2^60), a terminal pair -- and a
  // lookahead: a choice is kept only if the levels after it can still be
  // formed and the terminals still reach `need_t`. On the 2^42 pools the
  // greedy never had to backtrack, so their ladders are what they were.
  std::function<bool(int, int, int)> go = [&](int tt, int sp, int c) -> bool {
    if (c == num_cts) return tt >= need_t;
    const int left = num_ter - tt;
    const bool last = (c + 1 == num_cts);
    if (left >= 3) {
      plan.push_back('3');
      if (go(tt + 3, sp, c + 1)) return true;
      plan.pop_back();
    }
    if (sp >= 2) {
      plan.push_back('M');
      if (go(tt, sp - 2, c + 1)) return true;
      plan.pop_back();
    }
    // A terminal pair: on the last level with nothing else left (the 2^42
    // rule, the k64 pool's own top), or wherever the compute prefix still
    // needs terminals declared -- and only where it carries a transform
    // (2^49; EvalSpecialFFT's thin rule is 2^30, param_audit's starved band
    // ends at 2^49).
    if (left >= 2 && (last || tt < need_t) && ter_pair_bits(tt) >= 49.0) {
      plan.push_back('2');
      if (go(tt + 2, sp, c + 1)) return true;
      plan.pop_back();
    }
    return false;
  };
  return go(t, spares, 0) ? plan : std::string();
}

template <typename word>
bool LandingLadder<word>::IsClean(int landing) const {
  if (landing < 0 || landing > PoolLanding()) return false;
  if (landing == PoolLanding()) return true;  // the pool is its own ladder
  const int dec = landing + pool_.num_stc_levels;
  const int peak = Peak(dec);
  const auto [m, t] = pool_.level_config.at(dec);
  return m == peak && peak <= band_lo_ &&
         !CtSPlan(t, band_lo_ - peak, NeedT(dec)).empty();
}

template <typename word>
bool LandingLadder<word>::IsJunction(int landing) const {
  if (landing < 0 || landing >= PoolLanding() || IsClean(landing)) return false;
  const int dec = landing + pool_.num_stc_levels;
  const int peak = Peak(dec);
  const auto [m, t] = pool_.level_config.at(dec);
  // The fill takes the pinned prime plus one spare into the band and hands
  // the pool's top pair to the spares: `band_lo_ - peak + 2 - 1` remain.
  return m + 1 == peak && peak <= band_lo_ &&
         !CtSPlan(t, band_lo_ - peak + 1, NeedT(dec)).empty();
}

template <typename word>
std::vector<int> LandingLadder<word>::CleanLandings() const {
  std::vector<int> out;
  for (int l = 0; l <= PoolLanding(); l++)
    if (IsClean(l)) out.push_back(l);
  return out;
}

template <typename word>
typename LandingLadder<word>::Result LandingLadder<word>::ForLanding(
    int landing, JunctionPolicy policy) const {
  AssertTrue(landing >= 0 && landing <= PoolLanding(),
             "LandingLadder: landing " + std::to_string(landing) +
                 " is outside [0, " + std::to_string(PoolLanding()) +
                 "]; above the pool's own landing there are no primes to "
                 "put a band on");
  int ladder = landing;
  bool fill = false;
  if (policy == JunctionPolicy::kFillJunction && IsJunction(landing)) {
    fill = true;
  } else {
    while (!IsClean(ladder)) ladder++;
  }
  Result r = Cut(ladder, fill);
  r.landing = landing;
  r.ladder_landing = ladder;
  r.slack = ladder - landing;
  return r;
}

template <typename word>
typename LandingLadder<word>::Result LandingLadder<word>::Cut(
    int ladder_landing, bool fill_junction) const {
  Result r;
  const int ns = pool_.num_stc_levels;
  if (ladder_landing == PoolLanding()) {
    r.spec = pool_;
  } else {
    const int dec = ladder_landing + ns;
    const auto [m, t] = pool_.level_config.at(dec);
    const int peak = Peak(dec);
    const auto &main = pool_.main_primes;
    LadderSpec<word> s = pool_;
    s.level_config.assign(pool_.level_config.begin(),
                          pool_.level_config.begin() + dec + 1);
    s.default_encryption_level = dec;

    // The band's primes, bottom pair first, and the spares CtS may draw on.
    std::vector<word> band;
    std::vector<word> spares(main.begin() + peak, main.begin() + band_lo_);
    const std::vector<word> pool_band(main.begin() + band_lo_,
                                      main.begin() + band_lo_ + 2 * nem_);
    if (!fill_junction) {
      AssertTrue(m == peak && peak <= band_lo_,
                 "LandingLadder::Cut: landing " +
                     std::to_string(ladder_landing) + " is not clean");
      band = pool_band;
    } else {
      // THE JUNCTION FILL. new_main[0..peak) is pinned, and new_main[m] ==
      // main[peak - 1] is the compute prime the graft step left out; the
      // band's bottom level (m + 2, t) pairs it with the free prime that
      // brings the pair closest to the band width. The pool's top pair is
      // set aside so the count comes out, its primes joining the spares.
      AssertTrue(m + 1 == peak && peak <= band_lo_,
                 "LandingLadder::Cut: landing " +
                     std::to_string(ladder_landing) + " is not a junction");
      std::vector<word> pool_pairs(pool_band.begin(), pool_band.end() - 2);
      std::vector<word> candidates = spares;
      candidates.push_back(pool_band[pool_band.size() - 2]);
      candidates.push_back(pool_band[pool_band.size() - 1]);
      const double pinned_bits = std::log2(double(main[peak - 1]));
      size_t best = 0;
      double best_dev = 1e9;
      for (size_t i = 0; i < candidates.size(); i++) {
        const double dev = std::abs(pinned_bits +
                                    std::log2(double(candidates[i])) -
                                    width_bits_);
        if (dev < best_dev) best_dev = dev, best = i;
      }
      band.push_back(candidates[best]);
      band.insert(band.end(), pool_pairs.begin(), pool_pairs.end());
      candidates.erase(candidates.begin() + best);
      spares = candidates;
    }
    // new_main[0..peak) verbatim -- the compute prefix and, at a junction,
    // the pinned prime that opens the band -- then the band.
    s.main_primes.assign(main.begin(), main.begin() + peak);
    s.main_primes.insert(s.main_primes.end(), band.begin(), band.end());
    AssertTrue(static_cast<int>(s.main_primes.size()) == m + 2 * nem_,
               "LandingLadder::Cut: band count");
    for (int k = 1; k <= nem_; k++) s.level_config.emplace_back(m + 2 * k, t);

    // CoeffToSlot: the planned levels -- terminal triples (the pool's own
    // CtS levels, ~2^72), pairs of unused compute mains (2^60), terminal
    // pairs where the compute prefix needs the terminals declared or on the
    // last level with nothing else left (the k64 pool's own top).
    int mm = m + 2 * nem_, tt = t;
    const int need_t = NeedT(dec);
    const std::string plan =
        CtSPlan(t, static_cast<int>(spares.size()), need_t);
    AssertTrue(!plan.empty(),
               "LandingLadder::Cut: landing " + std::to_string(ladder_landing) +
                   " cannot form " + std::to_string(pool_.num_cts_levels) +
                   " CtS levels from what the cut leaves");
    for (const char c : plan) {
      if (c == '3') {
        tt += 3;
      } else if (c == 'M') {
        s.main_primes.push_back(spares[0]);
        s.main_primes.push_back(spares[1]);
        spares.erase(spares.begin(), spares.begin() + 2);
        mm += 2;
      } else {
        tt += 2;
      }
      s.level_config.emplace_back(mm, tt);
    }
    AssertTrue(tt >= need_t, "LandingLadder::Cut: a compute level needs a "
                             "terminal the ladder dropped");
    s.ter_primes.assign(pool_.ter_primes.begin(),
                        pool_.ter_primes.begin() + tt);
    r.spec = std::move(s);
  }
  // The band, measured off the ladder's own primes.
  const int dec = r.spec.default_encryption_level;
  double dev = 0.0;
  for (int k = 1; k <= nem_; k++)
    dev = std::max(dev, std::abs(r.spec.RescaleBits(dec + k) - width_bits_));
  r.band_deviation_bits = dev;
  const auto top = r.spec.level_config.back();
  r.primes_at_top = top.first + top.second;
  return r;
}

template <typename word>
std::string LandingLadder<word>::Describe(const Result &r) {
  const auto &s = r.spec;
  const int dec = s.default_encryption_level;
  const int top = s.MaxLevel();
  const int dnum = (r.primes_at_top + static_cast<int>(s.aux_primes.size()) - 1) /
                   static_cast<int>(s.aux_primes.size());
  std::ostringstream o;
  o.precision(2);
  o << std::fixed;
  o << "landing " << r.landing << " (ladder " << r.ladder_landing << ", slack "
    << r.slack << "): climb " << top << ", top NP (" << s.level_config.back().first
    << ", " << s.level_config.back().second << "), " << r.primes_at_top
    << " primes, dnum " << dnum << "; band deviation " << r.band_deviation_bits
    << " b; CtS [";
  const int cts_lo = top - s.num_cts_levels + 1;
  for (int l = cts_lo; l <= top; l++)
    o << (l > cts_lo ? ", " : "") << s.RescaleBits(l);
  o << "], StC [";
  for (int l = r.ladder_landing + 1; l <= dec; l++)
    o << (l > r.ladder_landing + 1 ? ", " : "") << s.RescaleBits(l);
  o << "]";
  return o.str();
}

template class LandingLadder<uint32_t>;
template class LandingLadder<uint64_t>;

}  // namespace cheddar
