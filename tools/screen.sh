#!/usr/bin/env bash

# Cross-version search screen: how deep each revision gets in a fixed wall
# clock, and how many nodes it spends reaching a fixed depth.
#
# This is the matched-time counterpart to tools/tactics.sh. A change that
# trades nodes for accuracy cannot be read at fixed depth - it reaches a
# different depth in the time a game actually gives it - so the headline here
# is the depth column and the node column is the reason behind it.
#
# Usage: tools/screen.sh <ref> [ref ...]
#   tools/screen.sh main HEAD
#   MOVETIME=5000 DEPTH=16 tools/screen.sh main feat/foo feat/bar
#
# Env knobs: MOVETIME (ms per position, default 3000), DEPTH (fixed-depth node
# count, default 14), OUT (work dir).
#
# A one-ply gain in one position out of four is noise: the budget lands
# mid-iteration and which side finishes first is chance. Look for a gain in
# most positions, backed by a node count moving the same way. Neither replaces
# an SPRT.
set -euo pipefail
cd "$(dirname "$0")/.."

[ $# -ge 1 ] || {
  echo "usage: screen.sh <ref> [ref ...]" >&2
  exit 2
}

MOVETIME="${MOVETIME:-3000}"
DEPTH="${DEPTH:-14}"
OUT="${OUT:-/tmp/lattice-screen}"

# Spread over opening, middlegame, endgame: a change that helps only one phase
# shows up as a split rather than as a small average.
FENS=(
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
  "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1"
)
# Index into FENS of the position the fixed-depth node count is taken from.
KIWIPETE=1

mkdir -p "$OUT"

cleanup() {
  local i=0
  for _ in "$@"; do
    git worktree remove --force "$OUT/build-$i" 2>/dev/null || true
    i=$((i + 1))
  done
}
trap 'cleanup "$@"' EXIT

# Each revision builds in its own worktree so the working tree is left alone:
# this is usually run with uncommitted changes present.
build() {
  local ref="$1" dir="$OUT/build-$2"
  rm -rf "$dir"
  git worktree remove --force "$dir" 2>/dev/null || true
  git worktree add --detach "$dir" "$ref" >/dev/null 2>&1 || return 1
  (cd "$dir" && cargo build --release --bin lattice -q) || return 1
  echo "$dir/target/release/lattice"
}

# stdin is held open past the search rather than closed after `go`. The search
# runs on the main thread, so a `quit` queued behind `go` is read the moment
# the search first yields and stops it at depth 1 - which still prints a
# well-formed `info depth 1` line and a `bestmove`, so the failure looks like a
# result rather than an error. An earlier version of this screen did exactly
# that and reported depth 1 across every revision.
uci_run() {
  local bin="$1" fen="$2" go="$3" hold="$4"
  {
    printf 'uci\nisready\nposition fen %s\n%s\n' "$fen" "$go"
    sleep "$hold"
  } | "$bin" 2>/dev/null
}

# Guards the hold-open contract above: a revision that reports depth 1 for a
# multi-second search is being cut off, not searching badly.
MIN_DEPTH=4

for ref in "$@"; do
  i=0
  for seen in "$@"; do
    [ "$seen" = "$ref" ] && break
    i=$((i + 1))
  done

  echo "building $ref ..." >&2
  bin=$(build "$ref" "$i") || {
    printf '%-24s | build failed\n' "$ref"
    continue
  }

  depths=""
  for fen in "${FENS[@]}"; do
    depth=$(uci_run "$bin" "$fen" "go movetime $MOVETIME" "$((MOVETIME / 1000 + 2))" |
      sed -n 's/^info depth \([0-9]*\) .*/\1/p' | tail -1)
    depth="${depth:-0}"
    if [ "$depth" -lt "$MIN_DEPTH" ]; then
      echo "$ref: depth $depth in ${MOVETIME}ms - the search is being cut off, not measured" >&2
      exit 3
    fi
    depths="$depths $depth"
  done

  nodes=$(uci_run "$bin" "${FENS[$KIWIPETE]}" "go depth $DEPTH" 300 |
    sed -n 's/.* nodes \([0-9]*\) .*/\1/p' | tail -1)

  printf '%-24s | depth in %sms:%s | d%s nodes: %s\n' \
    "$ref" "$MOVETIME" "$depths" "$DEPTH" "${nodes:-?}"
done
