#!/usr/bin/env bash

# Sampling profiler for the search, with the results attributed to their
# callers rather than left as a flat self-time list.
#
# The flat list is what misleads: `Board::make` reading 18% self time says
# nothing about whether the search or the legality filter is paying for it.
# This aggregates the call tree, so the report answers "who is calling this,
# and is the result used" instead of just "what is hot".
#
# Usage: tools/profile.sh [position] [seconds]
#   tools/profile.sh                       # startpos, 12s
#   tools/profile.sh kiwipete              # a named position from the table below
#   tools/profile.sh "8/2p5/3p4/KP5r/..." 20
#
# Env knobs: SAMPLE_MS (sampling interval, default 1), OUT (raw profile path),
# WARMUP (seconds before sampling starts, default 2).
#
# macOS only: it uses /usr/bin/sample, which ships with the OS. On Linux use
# `perf record -g` against the same binary; the aggregation below reads
# sample(1) output specifically.
set -euo pipefail
cd "$(dirname "$0")/.."

POSITION="${1:-startpos}"
SECONDS_TO_SAMPLE="${2:-12}"
SAMPLE_MS="${SAMPLE_MS:-1}"
WARMUP="${WARMUP:-2}"
OUT="${OUT:-/tmp/lattice-profile.txt}"

command -v /usr/bin/sample >/dev/null || {
  echo "/usr/bin/sample not found - this script is macOS only" >&2
  exit 1
}

# Named positions, so a profile can be reproduced by name in a commit message.
case "$POSITION" in
startpos) FEN="startpos" ;;
kiwipete) FEN="fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" ;;
endgame) FEN="fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" ;;
*) FEN="fen $POSITION" ;;
esac

# Debuginfo in release: symbol names only, no codegen change, so the profile
# describes the binary that actually ships.
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release >/dev/null 2>&1
BIN="$PWD/target/release/lattice"
PROF_BIN="/tmp/lattice-prof-$$"
cp "$BIN" "$PROF_BIN"
trap 'rm -f "$PROF_BIN"' EXIT

MOVETIME=$(((WARMUP + SECONDS_TO_SAMPLE + 3) * 1000))
{
  echo "uci"
  echo "isready"
  echo "position $FEN"
  echo "go movetime $MOVETIME"
  sleep $((WARMUP + SECONDS_TO_SAMPLE + 5))
  echo "quit"
} | "$PROF_BIN" >/tmp/lattice-profile-search.log 2>&1 &
ENGINE=$!

sleep "$WARMUP"
# The basename is what pgrep matches, and it must be an exact name match: a
# -f pattern would match unrelated processes through their argv.
PID=$(pgrep -x "$(basename "$PROF_BIN")" | head -1)
[ -n "$PID" ] || {
  echo "engine did not start; see /tmp/lattice-profile-search.log" >&2
  exit 1
}
echo "sampling pid $PID for ${SECONDS_TO_SAMPLE}s at ${SAMPLE_MS}ms ($POSITION)" >&2
/usr/bin/sample "$PID" "$SECONDS_TO_SAMPLE" "$SAMPLE_MS" -f "$OUT" >/dev/null 2>&1
wait "$ENGINE" 2>/dev/null || true

python3 tools/profile-report.py "$OUT"
