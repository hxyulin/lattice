#!/bin/bash

# pre-commit prepare-commit-msg hook: stamp a `Bench: <nodes>` trailer on commits
# that change engine source, so each carries its node-count signature (the same
# number ChessEval reads from `lattice bench`). Docs/tooling commits skip it.
#
# Wired via .pre-commit-config.yaml (installed by `pre-commit install`).
# Bypass for one commit:  SKIP_BENCH=1 git commit ...
set -e
msg="$1" # pre-commit passes the commit-message file path here
[ -n "$SKIP_BENCH" ] && exit 0
case "$PRE_COMMIT_COMMIT_MSG_SOURCE" in merge | squash) exit 0 ;; esac
# An existing trailer is not trusted: cherry-pick and rebase carry the number
# forward from a tree that no longer exists, so it is re-derived and replaced.
# Only Rust/Cargo changes move the node count; everything else skips the rebuild.
git diff --cached --name-only | grep -qE '\.rs$|Cargo\.(toml|lock)$' || exit 0
# Dropped before the build, not after: if the build or bench below fails this
# bails out, and an inherited number left behind would be worse than none.
grep -vi '^Bench:' "$msg" > "$msg.stamp" && mv "$msg.stamp" "$msg"
cargo build --release --bin lattice -q 2>/dev/null || exit 0
n=$(./target/release/lattice bench </dev/null 2>&1 \
      | sed -n 's/^Nodes searched:[[:space:]]*\([0-9]*\).*/\1/p' | tail -1)
[ -z "$n" ] && exit 0
git interpret-trailers --if-exists replace --trailer "Bench: $n" --in-place "$msg"
