#!/bin/bash
# The host half on vessl (96 cores): the float64 reference for a prompt set
# and the calibrations sim.py writes, with the Cho fold and without it.
#   bash vessl_host.sh <pop_dir> <held_dir> <out_dir>
# e.g. bash vessl_host.sh /root/bert_tiny/pop3 /root/bert_tiny/held3 /root/bert_tiny/f3
set -u
POP=$1; HELD=$2; OUT=$3
cd /root/work/cheddar-bt/bert_tiny || exit 1
mkdir -p "$OUT"
LOG=$OUT/host.log
: > "$LOG"
ALL=/root/bert_tiny/all

echo "==== $(date -u) reference for the held-out prompts" >> "$LOG"
if [ ! -f "$OUT/h_L00.f64" ]; then
  python3.12 reference.py "$ALL" "$OUT" --inputs "$HELD/prompts/inputs.f32" \
      --mask "$HELD/prompts/mask.u8" >> "$LOG" 2>&1 || { echo HOST_FAIL_REF >> "$LOG"; exit 1; }
fi

echo "==== $(date -u) sim.py WITH the fold" >> "$LOG"
python3.12 sim.py "$ALL" "$OUT/calib_fold.json" \
    --inputs "$POP/prompts/inputs.f32" --held-out "$HELD/prompts/inputs.f32" \
    >> "$LOG" 2>&1 || { echo HOST_FAIL_FOLD >> "$LOG"; exit 1; }

echo "==== $(date -u) sim.py WITHOUT the fold (the A/B baseline)" >> "$LOG"
python3.12 sim.py "$ALL" "$OUT/calib_nofold.json" --no-fold \
    --inputs "$POP/prompts/inputs.f32" --held-out "$HELD/prompts/inputs.f32" \
    >> "$LOG" 2>&1 || { echo HOST_FAIL_NOFOLD >> "$LOG"; exit 1; }

echo "==== $(date -u) sim.py WITH the fold, k = 1 forced (no main-path boot)" >> "$LOG"
python3.12 sim.py "$ALL" "$OUT/calib_fold1.json" --k 1     --inputs "$POP/prompts/inputs.f32" --held-out "$HELD/prompts/inputs.f32"     >> "$LOG" 2>&1 || echo NOTE_FOLD1_FAILED >> "$LOG"

echo "==== $(date -u) BT_HOST_DONE" >> "$LOG"
grep -aE "^layer|window|Cho|inv|exp |gelu|ln1|ln2|tanh|chain rms|head:|prompt\(s\)|calib ->|certified" "$LOG" | tail -60
