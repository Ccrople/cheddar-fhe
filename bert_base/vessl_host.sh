#!/bin/bash
# The host half on vessl (96 cores): the POPULATION calibration and the
# float64 reference the card is checked against, for one shape.
#   bash vessl_host.sh [T]      (T = 128 | 256 | 512)
# Writes, under /root/bert_base/held_ref<T>:
#   calib.json    the 1000-prompt population calibration (WikiText-2 TRAIN)
#   h_L*.f64      the float64 output of the B held-out prompts the card
#                 serves (WikiText-2 TEST), and cls_logits.f64
#
# THE MARGINS. `--margin 5.0 --gelu-margin 1.8 --exp-margin 2.0
# --exp-margin-hi 2.0` with the Cho inverse square roots capped at degree
# 255 is the set that survives twelve layers on prompts the calibration
# never saw; the defaults (1.3 / 1.2 / 1.0) are BERT-Tiny's and leak seven
# slots in two hundred million at layer 0, which is a batch-wide event.
# `--inv-max-degree 255` is not a taste: the LAST Cho pass's degree is what
# caps the level plan.
set -u
T="${1:-128}"
B=/root/bert_base
R=$B/held_ref$T
S=/root/work/cheddar-bb/bert_base
MARGINS="--margin 5.0 --gelu-margin 1.8 --exp-margin 2.0 --exp-margin-hi 2.0
         --inv-max-degree 255"
# A window is a maximum over TOKEN POSITIONS, so the population is sized in
# tokens and not in prompts: 1000 x 128 = 250 x 512. A calibration costs
# quadratically more in T through the scores, and this is what keeps T = 512
# from taking twelve hours to say the same thing.
NCAL=$((128000 / T))
rm -f "$B/host_done_$T"
mkdir -p "$R"
cd "$S" || exit 1

# ---- the population calibration (1000 prompts of the TRAIN split) --------
( python3.12 -u sim.py "$B/all$T" "$R/calib.json" \
    --inputs "$B/all$T/prompts/inputs.f32" --no-chain --limit "$NCAL" $MARGINS \
    > "$R/sim.log" 2>&1 || echo SIM_FAIL >> "$R/sim.log" ) &

# ---- the held-out prompts the card serves, through float64 --------------
( python3.12 -u reference.py "$B/all$T" "$R" \
    --inputs "$B/held$T/serve/prompts/inputs.f32" \
    --mask "$B/held$T/serve/prompts/mask.u8" \
    > "$R/host.log" 2>&1 || echo REF_FAIL >> "$R/host.log" ) &
wait
date -u > "$B/host_done_$T"
echo "==== $(date -u) BB_HOST_DONE T=$T"
grep -l FAIL "$R"/*.log 2>/dev/null
tail -3 "$R/sim.log"
