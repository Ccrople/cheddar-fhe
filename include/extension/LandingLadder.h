#pragma once

#include <string>
#include <vector>

#include "core/LadderSpec.h"

namespace cheddar {

/**
 * @brief ONE PARAMETER SET, ANY LANDING, AND THE SHORT CLIMB THAT GOES WITH IT.
 *
 * ## The problem
 *
 * A bootstrap's cost is set by how far it climbs: every limb operation in
 * CoeffToSlot, EvalMod and SlotToCoeff runs on the primes of the level it is
 * at, so a boot that only needs to land at level 5 and climbs to 26 pays for
 * twenty levels of primes it then throws away ([Grafting] appendix D.2). The
 * library already climbs only to `BootParameter::max_level_`, and every
 * boot level derives from it -- but a shorter climb on a fixed ladder drags
 * EvalMod's band onto the compute levels below it, and EvalMod's recursion
 * `s <- s^2 / prod` leaves its fixed point (measured: p = -518 bits, and now
 * refused by `EvalMod`). The band has to MOVE with the climb.
 *
 * It cannot move inside one `Parameter`: Cheddar's modulus at NP = (m, t) is
 * always the PREFIX `main[0..m) ++ ter[0..t)`, so a band at a different
 * height is a different main ORDER. That is why the `ci16_35_land*` presets
 * exist -- every one of them is the same prime pool with its mains reordered
 * by `gen_landing.py` -- and what the user does not want: a file per landing.
 *
 * ## What this does
 *
 * It is `gen_landing.py` inside the library, for the 2^42 family, whose
 * bands make it a cut-and-paste. A pool (`ci16_42_k{16,32,64}_w60`) is
 *
 *     compute[0..land] | StC: 3 graft steps | band: nem pairs of 2^60 | CtS
 *
 * and every band pair is stationary (2^60.00 to the second decimal), so a
 * pair is a valid EvalMod level WHEREVER it sits. The ladder for landing L
 * keeps the pool's levels 0..L+3 verbatim (the compute prefix, and the
 * pool's own graft steps above L serve as StC), puts the pool's band pairs
 * on top of them, then CoeffToSlot's levels from what is left: terminal
 * triples first (the pool's own CtS), then pairs of the pool's now-unused
 * compute mains. It declares only the primes it uses. No prime is mined and
 * no prime is new: the ladder is a subset of the pool in a different order,
 * which is exactly what the keyless crossing needs -- levels 0..L are the
 * pool's own primes in the pool's own order, so a ciphertext at any level
 * up to L is the same RNS words on both rings.
 *
 * ## What it refuses, and why slack covers it
 *
 * The prefix rule has one more consequence. The graft cycle's `[-1m, +3t]`
 * step leaves a main OUT that the level below still needs, so at such a
 * level (a JUNCTION: `lc[L+3].main + 1 == peak`) the band's first pair would
 * have to start with that pinned compute prime. `JunctionPolicy::kFillJunction`
 * does exactly that -- the pinned prime plus the best-matching unused
 * compute main, a pair within a few hundredths of a bit of 2^60 at the
 * BOTTOM of the band, where the recursion weights it by 2^0 -- and reports
 * the deviation. The default, `kStationaryOnly`, keeps the band the pool's
 * own pairs and nothing else: a junction landing is served by the nearest
 * exact ladder above it plus that many levels of SLACK, the knob that
 * moves only StC (`BootBack` LevelDowns across the gap at no cost in
 * precision). One level on every junction of the three pools, except k64's
 * landing 9, which is two, because k64's own band shares its bottom prime
 * with the graft step under it and its landing 10 is one prime short.
 *
 * The pool's own landing is the pool, byte for byte.
 *
 * ## The 2^35 family (2026-09-11)
 *
 * `ci16_35_k{16,32,64}_w58` (`reference/scripts/ci35_family.py`) are the
 * same construction on ci16_35's own compute prefix -- 30-bit mains, 25-bit
 * terminals, the graft cycle `[+2m, -1t] x5, [-3m, +5t]` -- with the band
 * mined to 2^58 pair by pair. Two things about that prefix the 2^42 rule
 * did not meet. Its `[-3m, +5t]` steps put ALL FIVE terminals in the
 * modulus at levels 3, 9 and 15, so every cut above those must declare all
 * five: CoeffToSlot's levels have to bring the terminal count back up to
 * what the compute prefix needs (`NeedT`), and a triple-first greedy that
 * left one terminal over could not. And its terminals are 25 bits, so a
 * terminal PAIR (2^50) is a transform level there, where the 2^42 family's
 * 23.5-bit pairs (2^47) fall in `param_audit`'s starved band. So the CtS
 * levels are now PLANNED (`CtSPlan`): the same preference order as before
 * -- a terminal triple, then a pair of unused compute mains, then a
 * terminal pair -- with a lookahead that refuses any choice from which the
 * remaining levels cannot both be formed and reach the needed terminal
 * count, and with a terminal pair admitted wherever it is at least 2^49.
 * On the three 2^42 pools every recorded ladder is unchanged (the old
 * greedy never had to backtrack there); a landing whose CtS cannot be
 * formed is not clean and is served, like a junction, by the exact ladder
 * above it plus slack. The `[-3m, +5t]` step also leaves THREE mains out
 * below its peak, so the levels right after it are neither clean nor
 * junctions (`peak - m` is 3, then 1): slack from above serves them too.
 *
 * ## The second 2^42 cut (2026-09-11)
 *
 * `ci16_42_k{16,32,64}` (`reference/scripts/ci42_family.py`) share ONE
 * compute prefix and encrypt at 18 / 15 / 17, and their CoeffToSlot levels
 * are 2^60 MAIN pairs where the terminal inventory allows (a pool re-adds in
 * triples only the terminals its prefix declared). Those CtS mains sit above
 * the band, outside every cut ladder's prefix and band, so `Cut` counts them
 * among the spares a cut's CtS may pair up (`NumCtSMains`); on the first
 * cut's pools there are none and nothing changes. And `kFillJunction` is
 * now conditional: the fill is taken only when its pair is within 0.15 bits
 * of the band (`ForLanding`), because this family's B-compensated compute
 * primes (2^29.3-29.6) have no partner under the hoist cap that comes
 * closer than 0.25-0.54 bits, and a landing scale that far off is worse
 * than the one level of slack the stationary route costs.
 *
 * The host mirror is `reference/scripts/landing_ladder.py`; every ladder it
 * writes passes `param_audit.py --strict`, and `landing_ladder_test` diffs
 * the two implementations and runs the crossing on the device.
 */
enum class JunctionPolicy { kStationaryOnly, kFillJunction };

template <typename word>
class LandingLadder {
 public:
  struct Result {
    LadderSpec<word> spec;
    int landing = 0;         //!< the level asked for
    int ladder_landing = 0;  //!< where the ladder itself lands (no slack)
    int slack = 0;           //!< StC slack the BootParameter must add
    //! worst |log2 prod - width| over the band; 0 unless a junction was filled
    double band_deviation_bits = 0.0;
    //! the top NP's prime count, for the climb's cost
    int primes_at_top = 0;
  };

  //! Validates the pool: a boot preset whose band is +2 main a level and
  //! whose pairs are all the same width to within 0.05 bits.
  explicit LandingLadder(const LadderSpec<word> &pool);

  const LadderSpec<word> &Pool() const { return pool_; }
  int PoolLanding() const { return pool_.Landing(); }
  int NumEvalModLevels() const { return nem_; }
  int BandWidthBits() const { return width_bits_; }

  //! An exact ladder exists: the pool's own landing, or a level whose
  //! compute prefix pins every main the band's first level holds and leaves
  //! the pool's band pairs untouched.
  bool IsClean(int landing) const;
  //! Not clean, but one pinned prime short of it (a `[-1m, +3t]` step).
  bool IsJunction(int landing) const;
  std::vector<int> CleanLandings() const;

  Result ForLanding(int landing,
                    JunctionPolicy policy = JunctionPolicy::kStationaryOnly) const;

  //! One line: landing, ladder, slack, climb, primes, dnum, band, CtS, StC.
  static std::string Describe(const Result &r);

 private:
  int Peak(int upto) const;
  //! The largest terminal count any level 0..dec holds: a ladder cut at dec
  //! must declare at least that many, so its CtS levels have to reach it.
  int NeedT(int dec) const;
  //! The pool's own CtS main pairs above the band (0 on a terminal-triple
  //! CtS pool); a cut ladder's CtS may draw on them like on unused compute
  //! mains.
  int NumCtSMains() const;
  //! CoeffToSlot's levels from what a cut leaves: one char a level, '3' a
  //! terminal triple, 'M' a pair of unused compute mains, '2' a terminal
  //! pair. Empty when `num_cts_levels` levels cannot be formed. See the
  //! header comment: the order is the 2^42 rule's and the lookahead is what
  //! the 2^35 family adds.
  std::string CtSPlan(int t, int spares, int need_t) const;
  Result Cut(int ladder_landing, bool fill_junction) const;

  LadderSpec<word> pool_;
  int nem_ = 0;        //!< EvalMod levels
  int band_lo_ = 0;    //!< index of the pool's first band prime
  int width_bits_ = 0; //!< the band pair width, 60 on this family
};

}  // namespace cheddar
