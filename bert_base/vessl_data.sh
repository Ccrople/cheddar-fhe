#!/bin/bash
# Every dataset BERT-Base needs on vessl, from nothing.  ~15 min.
#   bash vessl_data.sh
#
#   corpus/           WikiText-2 raw, train (CALIBRATION) and test (HELD OUT)
#   all<T>/           the weights + 1000 calibration prompts (train)
#   held<T>/serve/    the B prompts the card serves (test, DISJOINT windows)
#   scan<T>/prompts/  50,000 held-out prompts as IDS (test, random starts)
#
# The calibration and the held-out sets come from the SAME corpus's own
# train / test split, so a rate measured on the held-out set is about unseen
# prompts and not about a change of domain.
set -u
B=/root/bert_base
S=/root/work/cheddar-bb/bert_base
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$B/corpus"
cd "$S" || exit 1

# ---- numpy on all 96 cores (the image ships the reference BLAS: 2 GFLOPS) --
if ! update-alternatives --query libblas.so.3-x86_64-linux-gnu 2>/dev/null |
     grep -q openblas; then
  apt-get install -y -qq libopenblas-dev
  update-alternatives --set libblas.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3
  update-alternatives --set liblapack.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3
fi
python3.12 -c "import pyarrow" 2>/dev/null || \
  python3.12 -m pip install --break-system-packages -q pyarrow

# ---- the corpus ----------------------------------------------------------
[ -s "$B/corpus/wikitext2_test.txt" ] || python3.12 -u wikitext.py "$B/corpus" || exit 1
TRAIN=$B/corpus/wikitext2_train.txt
TEST=$B/corpus/wikitext2_test.txt

for T in 128 256 512; do
  NB=$((65536 / T))                      # the prompts one batch holds
  echo "==== T=$T, batch $NB"
  # the weights and the CALIBRATION population (train split)
  [ -s "$B/all$T/meta.json" ] || \
    python3.12 -u export.py "$B/all$T" --tokens "$T" --prompts 1000 \
      --corpus "$TRAIN" --min-len 3 --seed 1 || exit 1
  # what the card serves: B disjoint held-out windows (test split)
  [ -s "$B/held$T/serve/prompts/inputs.f32" ] || \
    python3.12 -u export.py "$B/held$T/serve" --tokens "$T" --prompts "$NB" \
      --corpus "$TEST" --min-len 3 --seed 7 --prompts-only || exit 1
  # the 50,000-prompt failure scan: ids only (inputs would be 78 GB at T=512)
  [ -s "$B/scan$T/prompts/ids.u32" ] || \
    python3.12 -u export.py "$B/scan$T" --tokens "$T" --prompts 50000 \
      --corpus "$TEST" --min-len 3 --seed 11 --prompts-only --ids-only \
      --overlap || exit 1
done
echo "==== $(date -u) BB_DATA_DONE"
du -sh "$B"/* 2>/dev/null
