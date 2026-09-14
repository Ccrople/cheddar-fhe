#!/bin/bash
# Pull everything small and irreplaceable off vessl, onto the laptop.
#   bash bert_base/pull.sh [dest]      (default: bert_base/results)
#
# The overlay is wiped whenever the laptop's session drops, and the last
# session lost every BERT-Base calibration that way. A calibration is a few
# MB and takes an hour of 96 cores to make; the float64 references and the
# weights are big and cheap to remake, so they stay on the box.
set -u
D="${1:-$(dirname "$0")/results}"
mkdir -p "$D"
for T in 128 256 512; do
  mkdir -p "$D/T$T"
  scp -q "vessl:/root/bert_base/held_ref$T/calib.json" "$D/T$T/" 2>/dev/null \
    && echo "calib$T"
  scp -q "vessl:/root/bert_base/held_ref$T/sim.log" "$D/T$T/" 2>/dev/null
  scp -q "vessl:/root/bert_base/scan$T/out/part_*.json" "$D/T$T/" 2>/dev/null \
    && echo "scan$T parts"
  scp -q "vessl:/root/bert_base/run_held$T.log" "$D/T$T/" 2>/dev/null \
    && echo "run$T log"
done
du -sh "$D"/* 2>/dev/null
