#!/bin/bash
# The host half on vessl (96 cores). The vessl overlay is wiped whenever the
# laptop's session drops, so this does everything CONCURRENTLY and the caller
# pulls the calibrations back the moment it says BT_HOST_DONE.
#   bash vessl_host.sh
# Writes /root/bert_tiny/{f3,t256_ref,t512_ref,b1}/{calib_fold,calib_nofold,
# calib_fold1}.json + the float64 references the runs compare against.
set -u
cd /root/work/cheddar-bt/bert_tiny || exit 1
ALL=/root/bert_tiny/all
STAMP=/root/bert_tiny/host_done
rm -f "$STAMP"

# ---- the held-out reference and the B = 1 slice (cheap, first) -----------
mkdir -p /root/bert_tiny/f3 /root/bert_tiny/b1/prompts \
         /root/bert_tiny/t256_ref /root/bert_tiny/t512_ref
python3.12 - <<'PY' > /root/bert_tiny/b1/host.log 2>&1
import numpy as np, json
meta = json.load(open('/root/bert_tiny/all/meta.json'))
T, H = meta['tokens'], meta['channels']
x = np.fromfile('/root/bert_tiny/held3/prompts/inputs.f32', dtype=np.float32).reshape(-1, T, H)
v = np.fromfile('/root/bert_tiny/held3/prompts/mask.u8', dtype=np.uint8).reshape(-1, T)
x[:1].tofile('/root/bert_tiny/b1/prompts/inputs.f32')
v[:1].tofile('/root/bert_tiny/b1/prompts/mask.u8')
print("one prompt, %d real tokens of %d" % (int(v[0].sum()), T))
PY

ref() {  # ref <out> <inputs> [mask]
  local out=$1 inp=$2 msk=${3:-}
  [ -f "$out/h_L00.f64" ] && return 0
  if [ -n "$msk" ]; then
    python3.12 reference.py "$ALL" "$out" --inputs "$inp" --mask "$msk" >> "$out/host.log" 2>&1
  else
    python3.12 reference.py "$ALL" "$out" --inputs "$inp" >> "$out/host.log" 2>&1
  fi
}
ref /root/bert_tiny/f3 /root/bert_tiny/held3/prompts/inputs.f32 /root/bert_tiny/held3/prompts/mask.u8 &
ref /root/bert_tiny/b1 /root/bert_tiny/b1/prompts/inputs.f32 /root/bert_tiny/b1/prompts/mask.u8 &
ref /root/bert_tiny/t256_ref /root/bert_tiny/t256/prompts/inputs.f32 &
ref /root/bert_tiny/t512_ref /root/bert_tiny/t512/prompts/inputs.f32 &
wait

# ---- every calibration at once ------------------------------------------
POP=/root/bert_tiny/pop3/prompts/inputs.f32
HELD=/root/bert_tiny/held3/prompts/inputs.f32
sim() {  # sim <src_dir> <out.json> <inputs> [extra...]
  local src=$1 out=$2 inp=$3; shift 3
  python3.12 sim.py "$src" "$out" --inputs "$inp" --no-chain "$@" >> "${out%.json}.log" 2>&1 \
    || echo "SIM_FAIL $out" >> "${out%.json}.log"
}
sim /root/bert_tiny/all  /root/bert_tiny/f3/calib_fold.json   "$POP" --held-out "$HELD" &
sim /root/bert_tiny/all  /root/bert_tiny/f3/calib_nofold.json "$POP" --held-out "$HELD" --no-fold &
sim /root/bert_tiny/all  /root/bert_tiny/f3/calib_fold1.json  "$POP" --held-out "$HELD" --k 1 &
sim /root/bert_tiny/t256 /root/bert_tiny/t256_ref/calib_fold.json   /root/bert_tiny/t256/prompts/inputs.f32 &
sim /root/bert_tiny/t256 /root/bert_tiny/t256_ref/calib_nofold.json /root/bert_tiny/t256/prompts/inputs.f32 --no-fold &
sim /root/bert_tiny/t512 /root/bert_tiny/t512_ref/calib_fold.json   /root/bert_tiny/t512/prompts/inputs.f32 &
sim /root/bert_tiny/t512 /root/bert_tiny/t512_ref/calib_nofold.json /root/bert_tiny/t512/prompts/inputs.f32 --no-fold &
wait
cp /root/bert_tiny/f3/calib_fold.json /root/bert_tiny/f3/calib_nofold.json \
   /root/bert_tiny/f3/calib_fold1.json /root/bert_tiny/b1/ 2>/dev/null
date -u > "$STAMP"
echo "==== $(date -u) BT_HOST_DONE"
