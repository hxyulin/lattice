#!/usr/bin/env bash

# Cross-version tactics comparison: build two git revisions and run the same
# suite through both, so a search change can be read before it is SPRTed.
#
# The engine's `tactics` subcommand runs one revision. This runs two, back to
# back on the same machine, and diffs the reports - which is the form the
# question usually takes ("is my branch better than main?").
#
# Usage: tools/tactics.sh <refA> <refB> [suite] [depth]
#   tools/tactics.sh HEAD main               # wac at depth 10
#   tools/tactics.sh HEAD main arasan2024 8
#
# Env knobs: SUITES_DIR (default ~/dev/chess-data/test-suites), OUT (work dir).
#
# Read the solve count as the headline and the node columns as the reason: a
# change that solves the same positions in fewer nodes is faster, one that
# solves more but needs more depth may still lose Elo at a real time control.
# Neither replaces an SPRT.
set -euo pipefail
cd "$(dirname "$0")/.."

REF_A="${1:?usage: tactics.sh <refA> <refB> [suite] [depth]}"
REF_B="${2:?usage: tactics.sh <refA> <refB> [suite] [depth]}"
SUITE="${3:-wac}"
DEPTH="${4:-10}"
SUITES_DIR="${SUITES_DIR:-$HOME/dev/chess-data/test-suites}"
OUT="${OUT:-/tmp/lattice-tactics}"

[ -f "$SUITE" ] || [ -f "$SUITES_DIR/$SUITE.epd" ] || {
  echo "suite not found: $SUITE (looked in $SUITES_DIR)" >&2
  exit 2
}

mkdir -p "$OUT"

# On any exit, including the early ones below: a leaked worktree makes the
# next run's `git worktree add` fail on a path that already exists.
cleanup() {
  git worktree remove --force "$OUT/build-a" 2>/dev/null || true
  git worktree remove --force "$OUT/build-b" 2>/dev/null || true
}
trap cleanup EXIT

# Each revision is built in its own worktree so the working tree is left
# alone: this is usually run with uncommitted changes present.
build() {
  local ref="$1" dir="$OUT/build-$2"
  rm -rf "$dir"
  git worktree remove --force "$dir" 2>/dev/null || true
  git worktree add --detach "$dir" "$ref" >/dev/null 2>&1
  (cd "$dir" && cargo build --release --bin lattice -q)
  echo "$dir/target/release/lattice"
}

echo "building $REF_A ..." >&2
BIN_A=$(build "$REF_A" a)
echo "building $REF_B ..." >&2
BIN_B=$(build "$REF_B" b)

echo "running $SUITE at depth $DEPTH ..." >&2
SUITES_DIR="$SUITES_DIR" "$BIN_A" tactics "$SUITE" --depth "$DEPTH" > "$OUT/a.txt"
SUITES_DIR="$SUITES_DIR" "$BIN_B" tactics "$SUITE" --depth "$DEPTH" > "$OUT/b.txt"

# A revision from before the `tactics` subcommand existed reports nothing, and
# an empty side would otherwise diff as "every position regressed".
for side in a b; do
  [ -s "$OUT/$side.txt" ] || {
    ref=$([ "$side" = a ] && echo "$REF_A" || echo "$REF_B")
    echo "$ref produced no report: does that revision have the tactics subcommand?" >&2
    exit 2
  }
done

# The bench signature of each build, which says whether the two searches are
# even different trees. Equal counts plus a different solve rate means the
# suite is reading noise, not a change.
bench_of() { "$1" bench </dev/null 2>&1 | sed -n 's/^Nodes searched:[[:space:]]*//p' | tail -1; }

echo
echo "=== $REF_A (bench $(bench_of "$BIN_A")) ==="
cat "$OUT/a.txt"
echo "=== $REF_B (bench $(bench_of "$BIN_B")) ==="
cat "$OUT/b.txt"

echo "=== diff ==="
if diff -u "$OUT/a.txt" "$OUT/b.txt" > "$OUT/diff.txt"; then
  echo "identical reports"
else
  # The header lines carry the totals; the id lists are the detail behind them.
  grep -E '^[-+](solved|nodes|depth|unstable)' "$OUT/diff.txt" || true
  echo
  echo "full diff: $OUT/diff.txt"
fi
