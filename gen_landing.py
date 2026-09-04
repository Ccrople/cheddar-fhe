# Emit a conjugate-invariant "landing" boot preset for ANY landing level L in
# [5, 19], as a strict sub-ladder of ci16_35 sharing its bottom 0..L verbatim
# (keyless crossing). [Grafting] D.2's flexible-output bootstrap: climb only to
# the level the schedule needs, so ModUp/CoeffToSlot/EvalMod act on fewer limbs.
#
#   levels 0..L : ci16_35's own (byte-identical, keyless crossing up to L)
#   8 EvalMod   : ci16_35's 29-bit primes (2/level, 58-bit); at a JUNCTION
#                 landing (grafting has swapped mains out below the peak) the
#                 bottom two ride the swapped-out 30-bit mains at 59/60-bit,
#                 which EvalMod tolerates (measured L9, L15)
#   CtS         : saturate main + bring terminals up to 5.
#
# THE TERMINAL PARITY. Reaching the top's 5 terminals from t_L is a net-odd
# terminal change when t_L is EVEN, and each rescale is pure-main OR pure-term
# (ModSwitch forbids dropping both), so ONE CtS level must be a single 25-bit
# terminal. Device-measured: folding that thin level into a transform phase
# injects large coefficient noise that HalfBoot's slot comparison hides but
# SlotToCoeff exposes -- so an even landing's Full Boot corrupted (residual ~5).
# The fix (EvalSpecialFFT): a thin single-terminal level is consumed by a PURE
# RESCALE, not a transform phase. This generator therefore places that terminal
# at the VERY TOP of an even landing (num_cts = 5: 1 thin + 4 thick) so the
# EvaluateCtS prologue rescales it away before the transform. Odd landings have
# no thin level (num_cts = 4, byte-identical to the earlier v1, and L=19 == the
# ci16_35 ladder itself).
#
# WHAT WORKS (device-measured, A6000):
#   * HalfBoot -- the LAYER's crossing op -- lands at EVERY L, residual ~1.1e-4
#     at both parities (the thin level costs the even landing nothing now).
#   * Full Boot (adds StC) is clean at EVERY L whose landing clears ci16_35's
#     num_accum==1 zone -- odd L >= 11 and even L >= 10 (GetEndLevel() >= 7) --
#     residual ~3e-05 at both parities.
import json, math, sys
SRC, LAND, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# Optional 4th argument: CoeffToSlot levels -- default 4 (odd landing) or 5 (even:
# 1 thin + 4 thick); 2 is the module-basis CtS in its two-level real form on an
# odd landing (Doing.md 3.6/3.7). Optional 5th argument: EvalMod levels -- default
# 8 (the degree-30 polynomial's 5 + 3 double angles, K = 16); 9 is
# CHEDDAR_BOOT_DOUBLE_ANGLE=4, K = 32, which the module-sparse secret's
# wrap-around wants (Doing.md 3.9), and the preset then pins `num_double_angle`.
NUM_CTS_ARG = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NE_ARG = int(sys.argv[5]) if len(sys.argv) > 5 else 8
# Optional 6th argument: "v3" SOLVES EvalMod's scale recursion instead of
# tolerating it. The recursion e <- e^2 / prod, unrolled over the band's NE
# levels, leaves end = 58 + sum_L 2^(L-LAND-1) (58 - w(L)) bits, so v2's
# 30-bit extra pairs (w = 60) at the bottom hand every e9/e10 ladder a
# DIFFERENT landing scale (land17c3e10: 2^51.8; the deviation map,
# Doing 7.49). v3 mines the free levels' pair products to targets chosen by
# back-substitution so end == 2^58.000 exactly (sub-millibit): the
# half/fused landing scale becomes ONE number across every ladder. A
# junction landing's swapped-main levels are pinned by the keyless bottom;
# their contribution is absorbed by the free levels above them (converting
# inherited 29-bit pairs to mined ones when the extras' range is short).
# v3 also emits a "scale_manifest" into the JSON (per-level rescale and
# scale bits, the EvalMod start/end) that param_audit.py checks.
V3 = len(sys.argv) > 6 and sys.argv[6] == "v3"
j = json.load(open(SRC, encoding="utf-8"))
main, term, aux = j["main_primes"], j["terminal_primes"], j["auxiliary_primes"]
lc = [tuple(x) for x in j["level_config"]]
bits = lambda p: math.log2(p)
sizes = [round(bits(p)) for p in main]
b29 = [i for i, s in enumerate(sizes) if s == 29]
lo30 = [i for i, s in enumerate(sizes) if s == 30]
NE, NT = NE_ARG, len(term)
first29 = b29[0]
assert len(b29) >= 16
assert 5 <= LAND <= 19, "landing must be in [5,19] (below 5 no room to compute; " \
                        "above 19 the bottom consumes EvalMod's 29-bit primes)"
m_land, t_land = lc[LAND]
peak = max(lc[i][0] for i in range(LAND + 1))
assert peak <= first29
even = (t_land % 2 == 0)
NUM_CTS = NUM_CTS_ARG if NUM_CTS_ARG else (5 if even else 4)  # even: 1 thin (top) + thick;  odd: thick
assert not even or NUM_CTS >= 2, "an even landing needs its thin top level plus at least one thick"

# CtS level layout (going UP from the EvalMod top [m_land+16, t_land]).
if even:
    # Reach NT-1 terminals in pairs (net even, since t_land is even), saturate
    # the mains, then the single top terminal is the THIN level.
    term_pair_levels = (NT - 1 - t_land) // 2
    main_pair_levels = (NUM_CTS - 1) - term_pair_levels
    total_main = m_land + 2 * NE + 2 * main_pair_levels
else:
    ter_to_add = NT - t_land
    ter_levels = (ter_to_add + 1) // 2
    main_pair_levels = NUM_CTS - ter_levels
    total_main = m_land + 2 * NE + 2 * main_pair_levels

spare30 = [i for i in lo30 if i >= peak and not (first29 <= i < first29 + 16)]
spare_needed = total_main - (peak + 16)


def mine_primes(count, taken, log_degree):
    """Fresh ~30-bit primes 1 mod 4N, descending from 2^30, for a ladder
    whose upper reaches outgrow the mother preset's spare mains. Levels above
    the landing never touch the keyless crossing (only 0..L must match), so a
    new prime up there is as good as an inherited one."""
    mod = 4 << log_degree
    found = []
    p = ((1 << 30) // mod) * mod + 1
    import random
    def is_prime(n):
        if n < 2: return False
        for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % sp == 0: return n == sp
        d, r = n - 1, 0
        while d % 2 == 0: d //= 2; r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = pow(a, d, n)
            if x in (1, n - 1): continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1: break
            else:
                return False
        return True
    while len(found) < count:
        if p not in taken and is_prime(p): found.append(p)
        p -= mod
        assert p > (1 << 29), "ran out of 30-bit candidates"
    return found


mined = []
if len(spare30) < spare_needed:
    short = spare_needed - len(spare30)
    taken = set(main) | set(term) | set(aux)
    mined = mine_primes(short, taken, j["log_degree"])
    print(f"  (mined {short} fresh 30-bit primes for the ladder's top: "
          f"{mined})")
# The EvalMod ladder must rescale by the SAME width at every level it runs
# its polynomial on: EvalMod's scale recursion s <- s^2 / prod has the
# rescale width as its fixed point, and a 60-bit level at the TOP with
# 58-bit levels below sends it 2^60 -> 2^62 -> 2^66 -> 2^74 -> 2^90 and
# annihilates the output (measured on the first land13c2e9). So the extra
# EvalMod primes (NE > 8, two 30-bit spares a level) go at the BOTTOM of
# the ladder, where the K = 16 junction landings already ride 59/60-bit
# levels, and the 16 29-bit primes sit above them under CoeffToSlot.
extra = 2 * (NE - 8)
spares = [main[i] for i in spare30[:spare_needed]] + mined
new_main = main[:peak] + spares[:extra] + main[first29:first29 + 16] + spares[extra:]
assert len(new_main) == total_main
new_term = term[:]

new_lc = [list(lc[i]) for i in range(LAND + 1)]
m, t = m_land, t_land
for _ in range(NE):
    m += 2; new_lc.append([m, t])          # EvalMod: +2 main each, 58-bit
if even:
    for _ in range(term_pair_levels):
        t += 2; new_lc.append([m, t])      # thick terminal pairs (bottom)
    for _ in range(main_pair_levels):
        m += 2; new_lc.append([m, t])      # thick main pairs
    assert t == NT - 1 and m == total_main
    t += 1; new_lc.append([m, t])          # THE THIN single-terminal top level
    assert t == NT
else:
    for _ in range(main_pair_levels):
        m += 2; new_lc.append([m, t])
    assert m == total_main
    rem = NT - t
    while rem > 0:
        s = min(2, rem); t += s; rem -= s; new_lc.append([m, t])
    assert t == NT
max_level = len(new_lc) - 1

if V3:
    # SOLVE the recursion. end - 58 = sum_L 2^(L-LAND-1) * (58 - w(L)) over
    # the band (start = 2^58 exactly: the band's top is an inherited 29-bit
    # pair), so re-mining one free level L to width w + S/2^(L-LAND-1)
    # zeroes S; low-weight levels give millibit granularity. Free mains are
    # the indices >= peak (extras and inherited 29-bit primes both -- only
    # 0..LAND is pinned by the keyless crossing); a junction's swapped
    # mains below peak stay, and the free levels above absorb them.
    emS3 = max_level - NUM_CTS
    band = list(range(LAND + 1, emS3 + 1))  # bottom -> top

    def level_mains(L):
        return list(range(new_lc[L - 1][0], new_lc[L][0]))

    def width(L):
        return sum(bits(new_main[i]) for i in level_mains(L))

    def mine_pool(log_degree, lo_bits, hi_bits):
        mod = 4 << log_degree
        out = []
        p = (int(2 ** hi_bits) // mod) * mod + 1
        floor = int(2 ** lo_bits)
        # reuse mine_primes' Miller-Rabin via a tiny local
        def isp(n):
            if n < 2: return False
            for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
                if n % sp == 0: return n == sp
            d, r = n - 1, 0
            while d % 2 == 0: d //= 2; r += 1
            for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
                x = pow(a, d, n)
                if x in (1, n - 1): continue
                for _ in range(r - 1):
                    x = x * x % n
                    if x == n - 1: break
                else:
                    return False
            return True
        while p > floor:
            if isp(p): out.append(p)
            p -= mod
        return out

    # HOIST_MAIN_CAP = 2^30.2 (design_sylphflow40.py); stay under it.
    pool = mine_pool(j["log_degree"], 27.8, 30.19)

    slots = []  # (level, replaceable main indices), bottom-up, low weight
    for L in band:
        free_idx = [i for i in level_mains(L) if i >= peak]
        if free_idx:
            slots.append((L, free_idx))
        if len(slots) == 3:
            break
    assert slots, "v3: no free EvalMod level to solve with"

    for _ in range(6):
        S = sum((1 << (L - LAND - 1)) * (58.0 - width(L)) for L in band)
        if abs(S) < 5e-4:
            break
        taken3 = set(new_main) | set(new_term) | set(aux)
        for si, (L, free_idx) in enumerate(slots):
            wgt = 1 << (L - LAND - 1)
            tgt = width(L) + S / wgt
            last = si == len(slots) - 1
            if 56.2 <= tgt <= 60.5 or last:
                tgt = min(60.5, max(56.2, tgt))
                for i in free_idx:
                    taken3.discard(new_main[i])
                need = tgt - sum(bits(new_main[i]) for i in level_mains(L)
                                 if i not in free_idx)
                if len(free_idx) == 2:
                    cands = [p for p in pool if p not in taken3]
                    best, bd = None, 1e18
                    for ai, a in enumerate(cands):
                        rem = need - bits(a)
                        for b in cands[ai + 1:]:
                            d = abs(bits(b) - rem)
                            if d < bd: best, bd = (a, b), d
                    new_main[free_idx[0]], new_main[free_idx[1]] = best
                else:
                    best, bd = None, 1e18
                    for p in pool:
                        if p in taken3: continue
                        d = abs(bits(p) - need)
                        if d < bd: best, bd = p, d
                    new_main[free_idx[0]] = best
                break
    S = sum((1 << (L - LAND - 1)) * (58.0 - width(L)) for L in band)
    print("  (v3 solved the EvalMod recursion: end = 2^%.6f, |end - 58| = "
          "%.3f millibits; band widths bottom->top %s)"
          % (58.0 + S, abs(S) * 1000,
             ["%.3f" % width(L) for L in band]))
    assert abs(S) < 2e-3, "v3: the recursion did not converge"
    assert len(set(new_main)) == len(new_main), "v3: duplicate main"
    assert not (set(new_main) & set(new_term)), "v3: main/ter collision"

def resc(hi, lo):
    dm, dt = hi[0]-lo[0], hi[1]-lo[1]; b = 0.0
    if dm > 0: b += sum(bits(new_main[lo[0]+k]) for k in range(dm))
    if dm < 0: b -= sum(bits(new_main[hi[0]+k]) for k in range(-dm))  # grafts
    if dt > 0: b += sum(bits(new_term[lo[1]+k]) for k in range(dt))
    if dt < 0: b -= sum(bits(new_term[hi[1]+k]) for k in range(-dt))
    return b
ok = True
for i in range(1, len(new_lc)):
    a, b = new_lc[i-1], new_lc[i]
    ta, tb = a[0]+a[1], b[0]+b[1]
    if not (ta < tb or (ta == tb and a[0] < b[0])): print(f" order L{i}"); ok = False
    if b[0]-a[0] > 0 and b[1]-a[1] > 0: print(f" MIXED drop L{i}"); ok = False
if new_lc[-1] != [total_main, NT]: print(" last"); ok = False
emS, emE = max_level - NUM_CTS, max_level - NUM_CTS - NE
em = [resc(new_lc[L], new_lc[L-1]) for L in range(emS, emE, -1)]
ct = [resc(new_lc[L], new_lc[L-1]) for L in range(max_level, emS, -1)]  # top..bottom
# v3's compensator levels may sit a shade under 57 by design (the solve
# clamps to [56.2, 60.5]); EvalMod's own tolerance is the 2^62 assert.
ok = ok and emE == LAND and all((56 if V3 else 57) <= x <= 61 for x in em)
# the only sub-30-bit CtS rescale allowed is the even landing's single thin top
thin = [round(x) for x in ct if round(x) < 49]
ok = ok and (thin == [25] if even else thin == [])
logQP = sum(bits(p) for p in new_main+new_term+aux)
kind = "JUNCTION " if m_land < peak else ""
kind += "even (thin-top, full Boot)" if even else "odd (full Boot)"
print(f"L{LAND} t={t_land} [{kind}]: max={max_level} num_cts={NUM_CTS} #Q={total_main+NT} "
      f"logQP={logQP:.0f} EvalMod={[round(x) for x in em]} CtS(top..bot)={[round(x) for x in ct]} "
      f"{'OK' if ok else 'BAD'}")
if not ok: sys.exit(1)
out = {"log_degree": j["log_degree"], "log_default_scale": j["log_default_scale"],
           "boot": True, "dense_hamming_weight": j["dense_hamming_weight"],
           "sparse_hamming_weight": j["sparse_hamming_weight"], "num_cts_levels": NUM_CTS,
           "num_stc_levels": j["num_stc_levels"], "terminal_primes": new_term,
           "main_primes": new_main, "auxiliary_primes": aux,
           "default_encryption_level": LAND, "level_config": new_lc,
           "additional_base": j.get("additional_base", [0, 0]), "conjugate_invariant": True}
# EvalMod's double-angle count is pinned in the preset when it is not the
# process default: NE = 5 (the degree-30 polynomial) + r.
if NE != 8:
    out["num_double_angle"] = NE - 5
if V3:
    # The scale manifest: what every level's rescale and canonical scale
    # ARE, stated by the generator and checked by param_audit.py (and by
    # param_robust_test against the library). scale_bits runs 0..LAND (the
    # sqrt recursion only exists up to default_encryption_level).
    prods = [round(resc(new_lc[i], new_lc[i - 1]), 6)
             for i in range(1, len(new_lc))]
    sb = [float(j["log_default_scale"])]
    for i in range(1, LAND + 1):
        sb.append(round(0.5 * (sb[i - 1] + prods[i - 1]), 6))
    out["scale_manifest"] = {
        "prod_bits": prods,
        "scale_bits": [round(x, 6) for x in sb],
        "evalmod": {"start_level": max_level - NUM_CTS,
                    "num_levels": NE,
                    "start_bits": 58,
                    "end_bits": round(58.0 + S, 6)},
    }
json.dump(out, open(OUT, "w"), indent=2)
print(f"LANDING={LAND}")
