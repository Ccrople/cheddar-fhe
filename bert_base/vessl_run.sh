#!/bin/bash
# The BERT-Base chain on the A100 (vessl): bash vessl_run.sh <tag> [ENV=.. ...]
#   data:  /root/bert_base/all128 (export.py), /root/bert_base/ref128
#          (reference.py + sim.py's calib.json); tree /root/work/cheddar-bb/build_bb
#   log:   /root/bert_base/run_<tag>.log, ends with BB_RUN_DONE rc=<n>
TAG=${1:-run}; shift
LOG=/root/bert_base/run_$TAG.log
cd /root/work/cheddar-bb/build_bb/unittest || exit 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > "$LOG"
( time env PATH=/usr/local/cuda-13.0/bin:$PATH \
    CHEDDAR_POOL_MIB=0 CHEDDAR_POOL_RELEASE_MIB=0 CHEDDAR_POOL_MAX_BIN_MIB=0 \
    BERT_BASE_ALL=/root/bert_base/all128 BERT_BASE_REF=/root/bert_base/ref128 \
    "$@" ./ci_bert_base_test ) >> "$LOG" 2>&1
echo "BB_RUN_DONE rc=$?" >> "$LOG"
grep -aE "BERT-Base:|setup|plan:|input at|starting at|mask |WARNING|prompt\(s\)|^  prompt|^LAYER|CHAIN|^HEAD|Failure|what\(\)|Assert|FAILED|PASSED|BB_RUN_DONE|^real|bad_alloc|out of memory" "$LOG" | cut -c1-240
