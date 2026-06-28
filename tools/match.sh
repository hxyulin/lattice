#!/usr/bin/env bash

# Calibration gauntlet: play the engine against Stockfish pinned to a target
# UCI_Elo, under a real time control, and read off our engine's absolute Elo.
#
# Usage: tools/match.sh [target_elo] [games] [tc] [book.epd]
#   tools/match.sh 1400 100 8+0.08

set -euo pipefail
cd "$(dirname "$0")/.."

ELO="${1:-1400}"           # Stockfish UCI_Elo anchor (min 1320)
GAMES="${2:-100}"          # total games; each opening played twice, colours swapped
TC="${3:-8+0.08}"          # time control both engines obey, seconds+increment
BOOK="${4:-tools/openings.epd}"

command -v fastchess >/dev/null || { echo "fastchess not found in PATH" >&2; exit 1; }
command -v stockfish >/dev/null || { echo "stockfish not found in PATH" >&2; exit 1; }
[ -f "$BOOK" ] || { echo "opening book not found: $BOOK" >&2; exit 1; }

cargo build --release -p lattice-bin
BIN="$PWD/target/release/lattice"

ROUNDS=$(( (GAMES + 1) / 2 ))   # 2 games per round (-repeat), colours swapped

# -repeat plays each opening twice with colours swapped (fair). No -sprt: we want
# the point estimate + error bars fastchess prints, not an accept/reject verdict.
fastchess \
  -engine "cmd=$BIN" "name=lattice" \
  -engine cmd=stockfish "name=SF-$ELO" \
      "option.UCI_LimitStrength=true" "option.UCI_Elo=$ELO" \
  -openings "file=$BOOK" format=epd order=random \
  -each "tc=$TC" \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency 4
