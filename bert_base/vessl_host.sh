#!/bin/bash
# The host half on vessl (96 cores): every float64 reference and every
# calibration BERT-Base needs, concurrently, because the overlay is wiped
# whenever the laptop's session drops.
#   bash vessl_host.sh [T]      (T = 128 | 256 | 512)
# Writes, under /root/bert_base:
#   ref<T>/        the RECORDED prompt through all 12 layers + its oracle
#                  calibration (calib.json) -- the B = 1 milestone
#   held_ref<T>/   the 1000-prompt POPULATION calibration (calib.json) and
#                  the float64 output of the B = N/T prompts the card serves
set -u
T="${1:-128}"
B=/root/bert_base
STAMP=$B/host_done_$T
rm -f "$STAMP"
cd /root/work/cheddar-bb/bert_base || exit 1
mkdir -p "$B/ref$T" "$B/held_ref$T"

# ---- the oracle: one prompt, twelve layers, its own calibration ----------
( python3.12 -u reference.py "$B/all$T" "$B/ref$T" > "$B/ref$T/host.log" 2>&1 \
    || echo REF_FAIL >> "$B/ref$T/host.log"
  python3.12 -u sim.py "$B/all$T" "$B/ref$T/calib.json" --no-chain \
    > "$B/ref$T/sim.log" 2>&1 || echo SIM_FAIL >> "$B/ref$T/sim.log" ) &

# ---- the population calibration (1000 prompts) --------------------------
( python3.12 -u sim.py "$B/all$T" "$B/held_ref$T/calib.json" \
    --inputs "$B/all$T/prompts/inputs.f32" --no-chain \
    > "$B/held_ref$T/sim.log" 2>&1 || echo SIM_FAIL >> "$B/held_ref$T/sim.log" ) &

# ---- the held-out prompts the card serves, through float64 --------------
( python3.12 -u reference.py "$B/all$T" "$B/held_ref$T" \
    --inputs "$B/held$T/serve/inputs.f32" --mask "$B/held$T/serve/mask.u8" \
    > "$B/held_ref$T/host.log" 2>&1 || echo REF_FAIL >> "$B/held_ref$T/host.log" ) &
wait
date -u > "$STAMP"
echo "==== $(date -u) BB_HOST_DONE T=$T"
grep -l FAIL "$B/ref$T"/*.log "$B/held_ref$T"/*.log 2>/dev/null
