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
#   SPRT: STC 8+0.08 [0,5] LLR 2.94 (368g) +37 Elo pass
#   SPRT: pending (OpenBench rerun)
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
  --format='%s%x09%(trailers:key=SPRT,valueonly,separator=)' "$root"..HEAD |
  awk -F'\t' '
    $2 == "" { next }                                       # no SPRT trailer -> not a tested feature
    { subject = $1; sprt = $2
      sub(/^[a-z]+(\([^)]*\))?: /, "", subject)             # strip the conventional-commit scope
      if (sprt ~ /^pending/) { print "| " subject " | - | - | - | pending |"; next }
      n = split(sprt, a, " "); tc = a[1]; verdict = a[n]; games = "-"; elo = "-"
      for (i = 1; i <= n; i++) {
        if (a[i] ~ /^\([0-9]+g\)$/) { g = a[i]; gsub(/[()g]/, "", g); games = g }
        if (a[i] == "Elo" && i > 1) { elo = a[i - 1] }
      }
      print "| " subject " | " tc " | " games " | " elo " | " verdict " |"
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
