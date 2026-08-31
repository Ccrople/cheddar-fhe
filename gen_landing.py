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
#   4 CtS       : saturate main + bring terminals up to 5. At an EVEN landing
#                 (5 - t_L odd) one CtS level is a single 25-bit terminal.
#
# WHAT WORKS (measured on A6000, all L in [5,19]):
#   * HalfBoot -- the LAYER's crossing op -- lands at EVERY L. Odd L: residual
#     ~1.1e-04. Even L: ~3.8e-04 (the 25-bit CtS costs ~2 bits).
#   * Full Boot (adds StC) is clean only at ODD L >= 11 (residual ~3e-05). At
#     an even L, StC reads the scale the 25-bit CtS left slightly off and
#     corrupts; below 11 its StC hits ci16_35's num_accum==1 zone.
# So for the layer, any L >= 5 is usable; odd L is preferred (full precision,
# and full Boot available).
import json, math, sys
SRC, LAND, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# Optional 4th argument: CoeffToSlot levels (default 4, the native CtS; 2 is the
# module-basis CtS in its two-level real form, Doing.md 3.6/3.7).
NUM_CTS_ARG = int(sys.argv[4]) if len(sys.argv) > 4 else 4
# Optional 5th argument: EvalMod levels (default 8 = 5 for the degree-30
# polynomial + 3 double angles, K = 16; 9 is CHEDDAR_BOOT_DOUBLE_ANGLE=4,
# K = 32, which the module-sparse secret's wrap-around wants, Doing.md 3.9).
NE_ARG = int(sys.argv[5]) if len(sys.argv) > 5 else 8
j = json.load(open(SRC, encoding="utf-8"))
main, term, aux = j["main_primes"], j["terminal_primes"], j["auxiliary_primes"]
lc = [tuple(x) for x in j["level_config"]]
bits = lambda p: math.log2(p)
sizes = [round(bits(p)) for p in main]
b29 = [i for i, s in enumerate(sizes) if s == 29]
lo30 = [i for i, s in enumerate(sizes) if s == 30]
NE, NUM_CTS, NT = NE_ARG, NUM_CTS_ARG, len(term)
first29 = b29[0]
assert len(b29) >= 16
assert 5 <= LAND <= 19, "landing must be in [5,19] (below 5 no room to compute; " \
                        "above 19 the bottom consumes EvalMod's 29-bit primes)"
m_land, t_land = lc[LAND]
peak = max(lc[i][0] for i in range(LAND + 1))
assert peak <= first29

ter_to_add = NT - t_land
ter_levels = (ter_to_add + 1) // 2
main_levels = NUM_CTS - ter_levels
total_main = m_land + 2 * NE + 2 * main_levels
spare30 = [i for i in lo30 if i >= peak and not (first29 <= i < first29 + 16)]
spare_needed = total_main - (peak + 16)
assert len(spare30) >= spare_needed, "not enough spare 30-bit primes"
new_main = main[:peak] + main[first29:first29 + 16] + [main[i] for i in spare30[:spare_needed]]
assert len(new_main) == total_main
new_term = term[:]

new_lc = [list(lc[i]) for i in range(LAND + 1)]
m = m_land
for _ in range(NE):
    m += 2; new_lc.append([m, t_land])
for _ in range(main_levels):
    m += 2; new_lc.append([m, t_land])
assert m == total_main
t = t_land; rem = ter_to_add
while rem > 0:
    s = min(2, rem); t += s; rem -= s; new_lc.append([m, t])
assert t == NT
max_level = len(new_lc) - 1

def resc(hi, lo):
    dm, dt = hi[0]-lo[0], hi[1]-lo[1]; b = 0.0
    if dm > 0: b += sum(bits(new_main[lo[0]+k]) for k in range(dm))
    if dt > 0: b += sum(bits(new_term[lo[1]+k]) for k in range(dt))
    if dt < 0: b -= sum(bits(new_term[hi[1]+k]) for k in range(-dt))
    return b
ok = True
for i in range(1, len(new_lc)):
    a, b = new_lc[i-1], new_lc[i]
    ta, tb = a[0]+a[1], b[0]+b[1]
    if not (ta < tb or (ta == tb and a[0] < b[0])): print(f" order L{i}"); ok = False
if new_lc[-1] != [total_main, NT]: print(" last"); ok = False
emS, emE = max_level - NUM_CTS, max_level - NUM_CTS - NE
em = [resc(new_lc[L], new_lc[L-1]) for L in range(emS, emE, -1)]
ct = [resc(new_lc[L], new_lc[L-1]) for L in range(max_level, emS, -1)]
ok = ok and emE == LAND and all(57 <= x <= 61 for x in em)
logQP = sum(bits(p) for p in new_main+new_term+aux)
kind = "JUNCTION " if m_land < peak else ""
kind += "even (HalfBoot-only)" if t_land % 2 == 0 else "odd (HalfBoot+Boot)"
print(f"L{LAND} t={t_land} [{kind}]: max={max_level} climb#Q={total_main+NT} "
      f"logQP={logQP:.0f} EvalMod={[round(x) for x in em]} CtS={[round(x) for x in ct]} "
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
json.dump(out, open(OUT, "w"), indent=2)
print(f"LANDING={LAND}")
