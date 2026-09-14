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
W="${2:-64}"
LIMIT="${3:-0}"
shift 3 2>/dev/null || shift $#
B=/root/bert_base
O=$B/scan$T/out
S=/root/work/cheddar-bb/bert_base
mkdir -p "$O"
cd "$S" || exit 1
# One scan at a time. Three shapes' scans would be 72 numpy processes on 96
# cores beside a crypto run that needs the host for its launches and
# staging -- and the first measurement of that cost was a layer at 310 s
# against the 137-270 s the same shape used to take. Serialising costs no
# throughput (the total work is fixed) and gives the card its cores back.
exec 9> /root/bert_base/.scan.lock
flock 9
# ONE BLAS THREAD PER WORKER, and a worker per core. Measured: one prompt's
# twelve-layer chain is 1.35 core-seconds single-threaded at T = 128, so
# 50,000 prompts is 19 core-hours = 20 minutes on this box IF the cores are
# actually used. The first attempt (24 workers x 4 threads) got 8.9 of 96
# cores and would have taken two hours: OpenBLAS spin-waits on its barriers,
# so 96 BLAS threads across 24 processes thrash rather than share (vmstat
# showed 260k context switches a second against 81 % idle). The work here is
# elementwise-dominated anyway -- Clenshaw and exp, not GEMM -- so threads
# were buying little even when they worked.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
LIM=""
[ "$LIMIT" != "0" ] && LIM="--limit $LIMIT"
echo "==== $(date -u) scan T=$T, $W workers x $OMP_NUM_THREADS thread $LIM $*"
for i in $(seq 0 $((W - 1))); do
  nice -n 19 python3.12 -u failure.py "$B/all$T" "$B/held_ref$T/calib.json" \
    --ids "$B/scan$T/prompts/ids.u32" --slice "$i/$W" --f32 --resume \
    --out "$O/part_$i.json" $LIM "$@" > "$O/part_$i.log" 2>&1 &
done
wait
echo "==== $(date -u) BB_SCAN_DONE T=$T"
python3.12 -u failure.py --merge "$O/part_*.json" "$T"
