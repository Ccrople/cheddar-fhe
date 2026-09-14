#!/bin/bash
# Everything that follows a calibration, in the order the box allows.
#   bash vessl_pipeline.sh [T ...]      (default: 128 256 512)
#
# For each shape, as soon as its calibration lands:
#   * the 50,000-prompt host scan starts in the BACKGROUND (CPU), and
#   * the held-out crypto run goes on the GPU -- one at a time, because the
#     standing rule is one memory-heavy job per card.
# The two halves do not contend: the scan is BLAS on 96 cores, the run is
# the A100.
set -u
Ts="${*:-128 256 512}"
B=/root/bert_base
S=/root/work/cheddar-bb/bert_base
exec > >(tee -a /root/bb_pipeline.log) 2>&1
for T in $Ts; do
  echo "==== $(date -u) waiting for calib T=$T"
  while [ ! -f "$B/host_done_$T" ]; do sleep 20; done
  if [ ! -s "$B/held_ref$T/calib.json" ]; then
    echo "==== $(date -u) T=$T HAS NO calib.json -- skipping"; continue
  fi
  echo "==== $(date -u) T=$T scan (background) + crypto run"
  setsid bash -c "bash $S/vessl_scan.sh $T 24 > /root/bb_scan$T.log 2>&1" \
    < /dev/null > /dev/null 2>&1 &
  sleep 5
  bash "$S/vessl_held.sh" "$T"
  echo "==== $(date -u) T=$T crypto run finished"
done
echo "==== $(date -u) BB_PIPELINE_DONE"
