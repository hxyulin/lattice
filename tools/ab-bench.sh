#!/usr/bin/env bash

# Interleaved A/B speed comparison of two engine binaries.
#
# Speed readings on a loaded laptop are worth very little taken one at a time:
# this machine's bench nps has been seen to swing 3.0M-5.0M between runs while
# nothing changed. Three things here make the comparison survive that:
#
#   - Interleaving. A and B alternate, so a load spike lands on both rather
#     than on whichever ran during it.
#   - The median, not the mean. A background build inflates a few samples
#     without moving the middle one.
#   - A QoS clamp. macOS exposes no CPU affinity (hwloc reports errno 78 on
#     Darwin), but it does let a process ask for performance cores. Without
#     it the scheduler is free to park the run on an E-core, which is a 3x
#     reading on the same binary.
#
# It still cannot make a busy machine idle. The spread of each side is printed
# next to its median: when they overlap, the honest answer is "no difference
# resolvable", not the direction the medians happen to point.
#
# Usage: tools/ab-bench.sh <binA> <binB> [runs] [bench|tactics] [depth]
#   tools/ab-bench.sh /tmp/base/lattice ./target/release/lattice
#   tools/ab-bench.sh a b 9 tactics 8
set -euo pipefail

BIN_A="${1:?usage: ab-bench.sh <binA> <binB> [runs] [mode] [depth]}"
BIN_B="${2:?usage: ab-bench.sh <binA> <binB> [runs] [mode] [depth]}"
RUNS="${3:-7}"
MODE="${4:-bench}"
DEPTH="${5:-8}"
CLAMP="${CLAMP:-utility}"

if command -v taskpolicy >/dev/null; then
  RUN=(taskpolicy -c "$CLAMP")
  CLAMP_NOTE="$CLAMP"
else
  RUN=()
  CLAMP_NOTE="none (taskpolicy not found)"
fi

# One sample: nps for bench (higher is better), seconds for tactics (lower).
sample() {
  local bin="$1"
  if [ "$MODE" = bench ]; then
    "${RUN[@]}" "$bin" bench </dev/null 2>&1 |
      sed -n 's/^Nodes\/second:[[:space:]]*//p' | tail -1
  else
    { /usr/bin/time -p "${RUN[@]}" "$bin" tactics --depth "$DEPTH" >/dev/null; } 2>&1 |
      awk '/^real/{print $2}'
  fi
}

# Which build is which. A stale binary left in /tmp is the failure this
# catches: it reads as a valid baseline and silently answers the wrong
# question. The bench node count identifies a build better than a git SHA
# would - it is a property of the binary itself, so it cannot be stamped
# wrongly or go stale against the source it was built from. Equal counts mean
# the two paths are the same build, whatever the paths suggest.
identify() {
  local bin="$1" nodes
  nodes=$("$bin" bench </dev/null 2>&1 | sed -n 's/^Nodes searched:[[:space:]]*//p' | tail -1)
  printf '%s (bench %s, %s)\n' "$bin" "${nodes:-unknown}" \
    "$(date -r "$bin" '+%Y-%m-%d %H:%M' 2>/dev/null || echo 'mtime unknown')"
}
A_ID=$(identify "$BIN_A")
B_ID=$(identify "$BIN_B")
echo "A: $A_ID" >&2
echo "B: $B_ID" >&2
if [ "${A_ID#* }" = "${B_ID#* }" ]; then
  echo "warning: A and B bench identically - comparing a build against itself?" >&2
fi

a_samples=()
b_samples=()
echo "interleaving $RUNS runs of $MODE (qos clamp: $CLAMP_NOTE)" >&2
for _ in $(seq "$RUNS"); do
  a_samples+=("$(sample "$BIN_A")")
  b_samples+=("$(sample "$BIN_B")")
  printf '.' >&2
done
echo >&2

[ "$MODE" = bench ] && fmt=%.0f || fmt=%.2f

stats() {
  printf '%s\n' "$@" | sort -g | awk -v f="$fmt" '
    {v[NR] = $1}
    END {
      mid = (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
      printf f "  (min " f ", max " f ")", mid, v[1], v[NR]
    }'
}

echo
if [ "$MODE" = bench ]; then
  echo "A nps median: $(stats "${a_samples[@]}")"
  echo "B nps median: $(stats "${b_samples[@]}")"
  echo "(higher is better)"
else
  echo "A seconds median: $(stats "${a_samples[@]}")"
  echo "B seconds median: $(stats "${b_samples[@]}")"
  echo "(lower is better)"
fi
