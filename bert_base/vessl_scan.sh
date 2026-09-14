#!/bin/bash
# The held-out FAILURE RATE for one shape: 50,000 WikiText-2 test prompts
# through the twelve-layer chain on the HOST, against the calibration that
# ships, every escape attributed to its prompt.
#   bash vessl_scan.sh <T> [workers] [limit] [extra failure.py args]
#
# The 50,000 prompts are RANDOM windows of the test split, which holds only
# ~2,200 disjoint ones at T = 128: the draws overlap and are correlated.
#
# `W` processes each take prompts `i::W`. A degree-255 Clenshaw is one
# numpy pass per degree and is not threaded, so the split is what uses the
# box; OMP_NUM_THREADS keeps each worker's BLAS from oversubscribing.
set -u
T="${1:-128}"
W="${2:-12}"
LIMIT="${3:-0}"
shift 3 2>/dev/null || shift $#
B=/root/bert_base
O=$B/scan$T/out
S=/root/work/cheddar-bb/bert_base
mkdir -p "$O"
cd "$S" || exit 1
export OMP_NUM_THREADS=$((96 / W))
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
LIM=""
[ "$LIMIT" != "0" ] && LIM="--limit $LIMIT"
echo "==== $(date -u) scan T=$T, $W workers x $OMP_NUM_THREADS threads $LIM $*"
for i in $(seq 0 $((W - 1))); do
  python3.12 -u failure.py "$B/all$T" "$B/held_ref$T/calib.json" \
    --ids "$B/scan$T/prompts/ids.u32" --slice "$i/$W" --f32 --resume \
    --out "$O/part_$i.json" $LIM "$@" > "$O/part_$i.log" 2>&1 &
done
wait
echo "==== $(date -u) BB_SCAN_DONE T=$T"
python3.12 -u failure.py --merge "$O/part_*.json" "$T"
