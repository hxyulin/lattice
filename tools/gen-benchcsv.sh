#!/bin/bash
# Regenerate bench.csv from the `Bench:` trailers stamped on each commit.
# One row per engine commit (the empty root is excluded): short hash, date,
# subject, depth-4 suite node count. The trailer is the source of truth; the
# CSV is a committed, diffable snapshot of it.
#
#   tools/gen-benchcsv.sh > bench.csv
set -e
root=$(git rev-list --max-parents=0 HEAD)
git log --reverse --date=short \
  --format='%h%x09%ad%x09%s%x09%(trailers:key=Bench,valueonly,separator=)' \
  "$root"..HEAD |
  awk -F'\t' 'BEGIN { OFS = ","; print "commit", "date", "subject", "bench_nodes" }
    { gsub(/"/, "\"\"", $3); gsub(/[ \t\r]/, "", $4); print $1, $2, "\"" $3 "\"", $4 }'
