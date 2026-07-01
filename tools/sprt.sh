#!/usr/bin/env bash

# Cross-version SPRT: build two git revisions of the engine and play them
# against each other under fastchess + SPRT to measure the Elo delta.
#
# Usage: tools/sprt.sh <refA> <refB> [rounds] [book.epd]
#   tools/sprt.sh HEAD main-backup          # current vs the old baseline
#   tools/sprt.sh HEAD HEAD~1               # this commit vs its parent
#   tools/sprt.sh v0.2.0 v0.1.0 800
#
# Env knobs: LIMIT=depth=3 (any -each limit), CONCURRENCY, BENCH_DEPTH,
# BOOKS_DIR, PGN (output path), ENGINE_BIN (binary name to pick from a build).
set -euo pipefail
cd "$(dirname "$0")/.."

REF_A="${1:?usage: sprt.sh <refA> <refB> [rounds] [book.epd]}"
REF_B="${2:?usage: sprt.sh <refA> <refB> [rounds] [book.epd]}"
ROUNDS="${3:-400}"
LIMIT="${LIMIT:-depth=3}"
BENCH_DEPTH="${BENCH_DEPTH:-}"
BOOKS_DIR="${BOOKS_DIR:-$HOME/dev/chess-data/books}"
# Default book: Fishtest standard if the data dir is present, else the tiny
# in-repo fallback (enough to smoke-test the tooling, too small for real Elo).
DEFAULT_BOOK="$BOOKS_DIR/UHO_4060_v3.epd"
[ -f "$DEFAULT_BOOK" ] || DEFAULT_BOOK="tools/openings.epd"
BOOK="${4:-$DEFAULT_BOOK}"
# Leave the system a couple of cores; perf cores only would under-utilise on a
# depth-limited (result-deterministic) run where thermal NPS drift is harmless.
CONCURRENCY="${CONCURRENCY:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc || echo 4) - 3 ))}"
[ "$CONCURRENCY" -ge 1 ] || CONCURRENCY=1

command -v fastchess >/dev/null || { echo "fastchess not found in PATH" >&2; exit 1; }
[ -f "$BOOK" ] || { echo "opening book not found: $BOOK" >&2; exit 1; }

ROOT="$PWD"
WORKTREES=()
cleanup() { for wt in "${WORKTREES[@]:-}"; do [ -n "$wt" ] && git worktree remove -f "$wt" 2>/dev/null || true; done; }
trap cleanup EXIT

# Resolve a ref to a binary path, building (and caching) if needed.
# Echoes the cached binary path on stdout; all logging goes to stderr.
build_ref() {
  local ref="$1"
  local sha; sha="$(git rev-parse --verify "${ref}^{commit}")" \
    || { echo "bad ref: $ref" >&2; exit 1; }
  local cache="$ROOT/.worktrees/cache/$sha"
  if [ -x "$cache" ]; then
    echo "cache hit: $ref ($sha) -> $cache" >&2
    echo "$cache"; return
  fi

  echo "building $ref ($sha)..." >&2
  local wt="$ROOT/.worktrees/wt-$sha"
  local tgt="$ROOT/.worktrees/build-$sha"
  git worktree add -f --detach "$wt" "$sha" >&2
  WORKTREES+=("$wt")
  # --workspace so the engine bin builds regardless of that commit's
  # default-members; isolated target dir avoids cross-version contamination.
  ( cd "$wt" && CARGO_TARGET_DIR="$tgt" cargo build --release --workspace >&2 )

  local found=""
  for cand in "${ENGINE_BIN:-}" lattice; do
    [ -n "$cand" ] && [ -x "$tgt/release/$cand" ] && { found="$tgt/release/$cand"; break; }
  done
  [ -n "$found" ] || { echo "no engine binary found in $tgt/release (set ENGINE_BIN)" >&2; exit 1; }

  mkdir -p "$(dirname "$cache")"
  cp "$found" "$cache"
  git worktree remove -f "$wt"; rm -rf "$tgt"
  echo "$cache"
}

BIN_A="$(build_ref "$REF_A")"
BIN_B="$(build_ref "$REF_B")"
SHA_A="$(git rev-parse --short "${REF_A}^{commit}")"
SHA_B="$(git rev-parse --short "${REF_B}^{commit}")"

# Bench both builds first: node/qnode counts are deterministic, so this is a
# fingerprint of each binary's search. Identical signatures => the two refs
# search the same (a no-op comparison); differing => records what changed.
# Goes to stderr so it never mixes into fastchess's stdout. Runs while cores
# are idle (before the match), so the NPS reading isn't thermally skewed.
echo "== bench signatures (the two builds should differ) ==" >&2
printf '\n-- A  %s@%s --\n' "$REF_A" "$SHA_A" >&2
# shellcheck disable=SC2086  # BENCH_DEPTH is an optional bare arg (empty = default)
"$BIN_A" bench $BENCH_DEPTH >&2
printf '\n-- B  %s@%s --\n' "$REF_B" "$SHA_B" >&2
# shellcheck disable=SC2086
"$BIN_B" bench $BENCH_DEPTH >&2
echo >&2

PGN="${PGN:-$ROOT/.worktrees/sprt-$SHA_A-vs-$SHA_B.pgn}"

# -repeat plays each opening twice with colours swapped (fair).
# SPRT H0: no Elo gain (elo0=0) vs H1: +10 Elo (elo1=10).
# Adjudication (Fishtest-style) ends decided games early for throughput:
#   -resign: 3 consecutive plies at >=|400|cp for the same side -> resign.
#   -draw:   from move 40, 8 plies within +/-10cp -> draw.
# -recover keeps the match alive if one engine crashes; -pgnout saves games.
fastchess \
  -engine "cmd=$BIN_A" "name=A:$REF_A@$SHA_A" \
  -engine "cmd=$BIN_B" "name=B:$REF_B@$SHA_B" \
  -openings "file=$BOOK" format=epd order=random \
  -each "$LIMIT" \
  -rounds "$ROUNDS" -repeat -concurrency "$CONCURRENCY" -recover \
  -resign movecount=3 score=400 \
  -draw movenumber=40 movecount=8 score=10 \
  -pgnout "file=$PGN" \
  -ratinginterval 10 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05
