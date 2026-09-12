#!/bin/bash
# The host half for the other service shapes.
#   bash vessl_shapes.sh b1     -- one prompt served with the population
#                                  calibration (the batch is filled with its
#                                  copies: B = 1 is 512 instances of it)
#   bash vessl_shapes.sh t256 | t512
set -u
cd /root/work/cheddar-bt/bert_tiny || exit 1
WHAT=$1
ALL=/root/bert_tiny/all

case $WHAT in
b1)
  OUT=/root/bert_tiny/b1
  LOG=$OUT/host.log
  mkdir -p "$OUT/prompts"
  : > "$LOG"
  # the first held-out prompt alone
  python3.12 - "$OUT" >> "$LOG" 2>&1 <<'PY'
import sys, numpy as np, json, os
out = sys.argv[1]
meta = json.load(open('/root/bert_tiny/all/meta.json'))
T, H = meta['tokens'], meta['channels']
x = np.fromfile('/root/bert_tiny/held3/prompts/inputs.f32', dtype=np.float32).reshape(-1, T, H)
v = np.fromfile('/root/bert_tiny/held3/prompts/mask.u8', dtype=np.uint8).reshape(-1, T)
x[:1].tofile(os.path.join(out, 'prompts/inputs.f32'))
v[:1].tofile(os.path.join(out, 'prompts/mask.u8'))
print("one prompt, %d real tokens of %d" % (int(v[0].sum()), T))
PY
  python3.12 reference.py "$ALL" "$OUT" --inputs "$OUT/prompts/inputs.f32" \
      --mask "$OUT/prompts/mask.u8" >> "$LOG" 2>&1 || { echo B1_FAIL >> "$LOG"; exit 1; }
  cp /root/bert_tiny/f3/calib_fold.json /root/bert_tiny/f3/calib_nofold.json "$OUT/"
  echo "==== BT_SHAPE_DONE b1" >> "$LOG"
  ;;
t256|t512)
  SRC=/root/bert_tiny/$WHAT
  OUT=/root/bert_tiny/${WHAT}_ref
  LOG=$OUT/host.log
  mkdir -p "$OUT"
  : > "$LOG"
  # these bundles are unpadded (no mask.u8) and carry B prompts, which is
  # exactly the batch the layout holds at that T
  if [ ! -f "$OUT/h_L00.f64" ]; then
    python3.12 reference.py "$SRC" "$OUT" --inputs "$SRC/prompts/inputs.f32" \
        >> "$LOG" 2>&1 || { echo SHAPE_FAIL_REF >> "$LOG"; exit 1; }
  fi
  python3.12 sim.py "$SRC" "$OUT/calib_fold.json" \
      --inputs "$SRC/prompts/inputs.f32" >> "$LOG" 2>&1 || { echo SHAPE_FAIL_FOLD >> "$LOG"; exit 1; }
  python3.12 sim.py "$SRC" "$OUT/calib_nofold.json" --no-fold \
      --inputs "$SRC/prompts/inputs.f32" >> "$LOG" 2>&1 || { echo SHAPE_FAIL_NOFOLD >> "$LOG"; exit 1; }
  echo "==== BT_SHAPE_DONE $WHAT" >> "$LOG"
  ;;
*) echo "usage: vessl_shapes.sh b1|t256|t512"; exit 1 ;;
esac
grep -aE "prompt|layer |window|Cho|chain rms|head:|calib ->|DONE|FAIL" "$LOG" | tail -40
