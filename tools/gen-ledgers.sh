#!/bin/bash
# Regenerate the ledgers from commit trailers - the single source of truth.
#
#   bench.csv                 one row per engine commit, from `Bench:` trailers
#   docs/src/sprt-results.md  the strength table, from `SPRT:` trailers
#
# Both are committed, diffable snapshots of the trailers; re-run after adding or
# amending a trailer (and at the tip before a release):
#
#   tools/gen-ledgers.sh
#
# The `Bench:` trailer is stamped automatically (tools/stamp-bench.sh). The
# `SPRT:` trailer is written by hand from the OpenBench verdict, in one of:
#
#   SPRT: STC 10+0.1 [0,5] LLR 2.94 (368g) +37 Elo pass
#   SPRT: STC 10+0.1 [0,5] LLR 3.32 (38g) +348.7 +- 93.4 Elo pass
#   SPRT: pending (OpenBench rerun)
#
# A commit may carry several `SPRT:` trailers (e.g. an STC and an LTC run); each
# becomes its own row. The Elo may be a decimal and may be followed by error
# bars (`+- 93.4`).
set -e
cd "$(git rev-parse --show-toplevel)"
root=$(git rev-list --max-parents=0 HEAD)
results=docs/src/sprt-results.md

# bench.csv: hash, date, subject, depth-suite node count.
{
  echo "commit,date,subject,bench_nodes"
  git log --reverse --date=short \
    --format='%h%x09%ad%x09%s%x09%(trailers:key=Bench,valueonly,separator=)' \
    "$root"..HEAD |
    awk -F'\t' 'BEGIN { OFS = "," }
      { gsub(/"/, "\"\"", $3); gsub(/[ \t\r]/, "", $4); print $1, $2, "\"" $3 "\"", $4 }'
} >bench.csv

# sprt-results.md: rewrite the generated table between the markers, keeping the
# hand-written prose above and below untouched.
rows=$(git log --reverse \
  --format='%s%x09%(trailers:key=SPRT,valueonly,separator=%x1f)' "$root"..HEAD |
  awk -F'\t' '
    $2 == "" { next }                                       # no SPRT trailer -> not a tested feature
    { subject = $1
      sub(/^[a-z]+(\([^)]*\))?: /, "", subject)             # strip the conventional-commit scope
      nr = split($2, runs, "\037")                          # one commit may carry several SPRT runs
      for (r = 1; r <= nr; r++) {
        sprt = runs[r]
        if (sprt ~ /^pending/) { print "| " subject " | - | - | - | pending |"; continue }
        n = split(sprt, a, " "); tc = a[1]; verdict = a[n]; games = "-"; elo = "-"
        for (i = 1; i <= n; i++) {
          if (a[i] ~ /^\([0-9]+g\)$/) { g = a[i]; gsub(/[()g]/, "", g); games = g }
          # `(N pairs)` splits into two tokens on space. A pair is two games -
          # same opening played both ways - so it doubles into the games column.
          if (a[i] ~ /^\([0-9]+$/ && a[i + 1] == "pairs)") {
            g = a[i]; gsub(/[()]/, "", g); games = g * 2
          }
          if (a[i] == "Elo" && i > 1) { elo = (a[i - 2] == "+-") ? a[i - 3] : a[i - 1] }
        }
        print "| " subject " | " tc " | " games " | " elo " | " verdict " |"
      }
    }')

# Rows go through a temp file, not `-v rows=...`: BSD awk rejects a multi-line
# value passed with -v ("newline in string"), so read them with getline instead.
rowsfile=$(mktemp)
printf '%s' "$rows" >"$rowsfile"
awk -v rowsfile="$rowsfile" '
  /<!-- BEGIN generated:/ {
    print
    print "| Feature | TC | Games | Elo | Verdict |"
    print "|---------|----|------:|----:|---------|"
    while ((getline line < rowsfile) > 0) print line
    close(rowsfile)
    skip = 1
    next
  }
  /<!-- END generated -->/ { skip = 0 }
  !skip
' "$results" >"$results.tmp" && mv "$results.tmp" "$results"
rm -f "$rowsfile"
