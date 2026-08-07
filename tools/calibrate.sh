#!/bin/bash
# Estimate Lattice's rating by playing a gauntlet against Stockfish pinned to
# several UCI_Elo levels. Each level is a separate opponent, so one run brackets
# the rating instead of guessing a single level and re-running.
#
#   tools/calibrate.sh [games_per_level] [tc]
#
# Stockfish's UCI_Elo is its own scale and is not CCRL; treat the output as a
# ballpark and a progress marker between features, not an absolute rating.
set -e
cd "$(git rev-parse --show-toplevel)"

games=${1:-100}
tc=${2:-10+0.1}
book=/Users/hxyulin/dev/chess-data/books/UHO_4060_v3.epd
out=/tmp/lattice-calibrate

mkdir -p "$out"
cargo build --release

engines=()
for elo in 2000 2300 2600 2900; do
  engines+=(-engine "cmd=$(command -v stockfish)" "name=SF-$elo" proto=uci
            "option.UCI_LimitStrength=true" "option.UCI_Elo=$elo"
            option.Threads=1 option.Hash=64)
done

# gauntlet: the seeded engine (the first one, Lattice) plays every other engine
# and the SF levels do not play each other. Concurrency is left at 4 of 11 cores
# so the machine stays usable and the time control is not distorted by
# oversubscription.
fastchess \
  -engine cmd=./target/release/lattice name=Lattice proto=uci option.Hash=64 \
  "${engines[@]}" \
  -tournament gauntlet \
  -each tc="$tc" \
  -openings file="$book" format=epd order=random \
  -games 2 -rounds $((games / 2)) \
  -repeat -concurrency 4 \
  -pgnout "file=$out/games.pgn" \
  -log "file=$out/fastchess.log" level=warn
