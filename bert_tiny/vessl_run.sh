#!/bin/bash
# The BERT-Tiny chain on the A100 (vessl): bash vessl_run.sh <tag> [ENV=.. ...]
#   data:  /root/bert_tiny/all (export.py), /root/bert_tiny/ref (reference.py
#          + sim.py's calib.json); tree /root/work/cheddar-bt/build_bt
#   log:   /root/bert_tiny/run_<tag>.log, ends with BT_RUN_DONE rc=<n>
TAG=${1:-run}; shift
LOG=/root/bert_tiny/run_$TAG.log
cd /root/work/cheddar-bt/build_bt/unittest || exit 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > "$LOG"
( time env PATH=/usr/local/cuda-13.0/bin:$PATH \
    CHEDDAR_POOL_MIB=0 CHEDDAR_POOL_RELEASE_MIB=0 CHEDDAR_POOL_MAX_BIN_MIB=0 \
    BERT_TINY_ALL=/root/bert_tiny/all BERT_TINY_REF=/root/bert_tiny/ref \
    "$@" ./ci_bert_tiny_test ) >> "$LOG" 2>&1
echo "BT_RUN_DONE rc=$?" >> "$LOG"
grep -aE "BERT-Tiny:|setup|plan:|input at|mask |WARNING|prompt\(s\)|^  prompt|^LAYER|CHAIN|^HEAD|Failure|what\(\)|Assert|FAILED|PASSED|BT_RUN_DONE|^real" "$LOG" | cut -c1-240
