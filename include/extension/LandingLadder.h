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
  Result Cut(int ladder_landing, bool fill_junction) const;

  LadderSpec<word> pool_;
  int nem_ = 0;        //!< EvalMod levels
  int band_lo_ = 0;    //!< index of the pool's first band prime
  int width_bits_ = 0; //!< the band pair width, 60 on this family
};

}  // namespace cheddar
