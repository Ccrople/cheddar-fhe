#!/bin/bash
# Text file -> encrypt -> the 2 encoder layers -> pooler + classifier -> labels,
# on the A100 (vessl), with the POPULATION calibration:
#
#   bash serve.sh <text.txt> [tag]
#
# One prompt per line. The client's step (tokenize, embed, embedding LayerNorm)
# is encode.py; the float64 reference (reference.py) runs beside it so every
# encrypted logit and label is checked, not just printed. Requires the data
# layout of vessl_run.sh: /root/bert_tiny/all (export.py, with the embedding
# tables) and a calibration (default: the padded population's).
TEXT=${1:?text file}; TAG=${2:-serve}
CALIB=${BERT_TINY_CALIB:-/root/bert_tiny/held2_ref/calib.json}
OUT=/root/bert_tiny/serve_$TAG
cd /root/work/cheddar-bt/bert_tiny || exit 1
python3.12 encode.py /root/bert_tiny/all "$TEXT" "$OUT" || exit 1
python3.12 reference.py /root/bert_tiny/all "$OUT/ref" --inputs "$OUT/inputs.f32" | tail -1
cp "$CALIB" "$OUT/ref/calib.json"
bash /root/work/cheddar-bt/bert_tiny/vessl_run.sh "$TAG" BERT_TINY_LAYERS=2 BERT_TINY_HEAD=1 \
  BERT_TINY_INPUTS="$OUT/inputs.f32" BERT_TINY_REF="$OUT/ref" BERT_TINY_PRINT_LABELS=1 \
  BERT_TINY_LABELS_OUT="$OUT/labels.txt"
echo "labels -> $OUT/labels.txt"
