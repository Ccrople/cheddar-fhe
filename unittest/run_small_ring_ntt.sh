#!/bin/bash
# Run the small-ring NTT probe, one log_degree per process.
#
# This separation is not stylistic. __cm_log_degree lives in constant memory and
# every handler guards its upload with a class-level `static inline bool
# cm_populated_`, so the first handler constructed in a process wins and any
# later one at a different degree is silently ignored. Running the binary
# unfiltered would therefore evaluate every case at whichever degree happened to
# go first. SmallRingNttTest.cpp asserts against that, but the fix is here.
#
# Usage: bash run_small_ring_ntt.sh [path-to-binary]

set -u

BIN="${1:-./small_ring_ntt_test}"

if [ ! -x "$BIN" ]; then
  echo "no such binary: $BIN" >&2
  exit 1
fi

rc=0
for d in 16 15 14 13 12; do
  echo "==================== log_degree $d ===================="
  if "$BIN" --gtest_filter="SmallRingNtt.RoundTripLogDegree${d}"; then
    echo "log_degree $d: PASS"
  else
    echo "log_degree $d: FAIL (exit $?)"
    rc=1
  fi
  echo
done

echo "======================================================="
if [ "$rc" -eq 0 ]; then
  echo "all degrees passed"
else
  echo "at least one degree failed"
fi
exit "$rc"
