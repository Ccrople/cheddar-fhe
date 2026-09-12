#!/bin/bash
# The A/B matrix on one prompt set: the baseline, each lever alone, and all
# of them. Every run is the same binary and the same prompts; only the
# calibration file and the three knobs change.
#
#   bash vessl_matrix.sh <tag> <prompts_dir> <ref_dir> [runs...]
#     prompts_dir/prompts/{inputs.f32,mask.u8}, ref_dir/{h_L*.f64,calib_*.json}
#     runs: base hoist fold all pb8 pb32 foldonly   (default: base hoist fold all)
set -u
TAG=$1; PROMPTS=$2; REF=$3; shift 3
RUNS=${*:-"base hoist fold all"}
OUT=/root/bert_tiny/matrix_$TAG.log
: > "$OUT"
for r in $RUNS; do
  case $r in
    base)     CAL=calib_nofold; H=0; PB=1 ;;
    hoist)    CAL=calib_nofold; H=1; PB=1 ;;
    foldonly) CAL=calib_fold;   H=0; PB=1 ;;
    fold)     CAL=calib_fold;   H=1; PB=1 ;;
    pb8)      CAL=calib_fold;   H=1; PB=8 ;;
    pb32)     CAL=calib_fold;   H=1; PB=32 ;;
    all)      CAL=calib_fold;   H=1; PB=32 ;;
    *) echo "unknown run $r" >> "$OUT"; continue ;;
  esac
  echo "======== $TAG/$r  calib=$CAL hoist=$H poly_batch=$PB  $(date -u +%H:%M:%S)" >> "$OUT"
  bash /root/vessl_run.sh "${TAG}_$r" \
      BERT_TINY_REF="$REF" \
      BERT_TINY_CALIB="$REF/$CAL.json" \
      BERT_TINY_INPUTS="$PROMPTS/prompts/inputs.f32" \
      BERT_TINY_HEAD=1 BERT_TINY_HOIST=$H BERT_TINY_POLY_BATCH=$PB \
      >> "$OUT" 2>&1
done
echo "BT_MATRIX_DONE $TAG" >> "$OUT"
grep -aE "^========|plan:|^LAYER|CHAIN|^HEAD|WARNING|Failure|FAILED|BT_RUN_DONE|^real" "$OUT" | cut -c1-200
