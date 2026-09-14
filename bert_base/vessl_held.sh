#!/bin/bash
# ONE held-out crypto run: the B prompts WikiText-2's test split gives for
# this T, on the population calibration, twelve layers and the head.
#   bash vessl_held.sh <T> [tag] [ENV=.. ...]
#
# T = 128 | 256 | 512 and B = 65536 / T, so every shape fills the same
# ciphertext: 512 x 128, 256 x 256, 128 x 512.
set -u
T="${1:-128}"
TAG="${2:-held$T}"
shift 2 2>/dev/null || shift $#
B=/root/bert_base
LOG=$B/run_$TAG.log
cd /root/work/cheddar-bb/build_bb/unittest || exit 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > "$LOG"
( time env PATH=/usr/local/cuda-13.0/bin:$PATH \
    CHEDDAR_POOL_MIB=0 CHEDDAR_POOL_RELEASE_MIB=0 CHEDDAR_POOL_MAX_BIN_MIB=0 \
    BERT_BASE_ALL="$B/all$T" \
    BERT_BASE_REF="$B/held_ref$T" \
    BERT_BASE_CALIB="$B/held_ref$T/calib.json" \
    BERT_BASE_INPUTS="$B/held$T/serve/prompts/inputs.f32" \
    BERT_BASE_MASK="$B/held$T/serve/prompts/mask.u8" \
    BERT_BASE_LAYERS=12 BERT_BASE_HEAD=1 \
    "$@" ./ci_bert_base_test ) >> "$LOG" 2>&1
echo "BB_RUN_DONE rc=$?" >> "$LOG"
grep -aE "BERT-Base:|setup|plan:|input at|starting at|mask |WARNING|prompt\(s\)|^  prompt|^LAYER|CHAIN|^HEAD|Failure|what\(\)|Assert|FAILED|PASSED|BB_RUN_DONE|^real|bad_alloc|out of memory" "$LOG" | cut -c1-240
