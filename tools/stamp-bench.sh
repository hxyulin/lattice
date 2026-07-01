#!/bin/bash

# pre-commit prepare-commit-msg hook: stamp a `Bench: <nodes>` trailer on commits
# that change engine source, so each carries its node-count signature (the same
# number OpenBench reads from `lattice bench`). Docs/tooling commits skip it.
#
# Wired via .pre-commit-config.yaml (installed by `pre-commit install`).
# Bypass for one commit:  SKIP_BENCH=1 git commit ...
set -e
msg="$1" # pre-commit passes the commit-message file path here
[ -n "$SKIP_BENCH" ] && exit 0
case "$PRE_COMMIT_COMMIT_MSG_SOURCE" in merge | squash) exit 0 ;; esac
grep -qi '^Bench:' "$msg" && exit 0
# Only Rust/Cargo changes move the node count; everything else skips the rebuild.
git diff --cached --name-only | grep -qE '\.rs$|Cargo\.(toml|lock)$' || exit 0
cargo build --release --bin lattice -q 2>/dev/null || exit 0
n=$(./target/release/lattice bench </dev/null 2>/dev/null | tail -1 | grep -oE '^[0-9]+')
[ -z "$n" ] && exit 0
git interpret-trailers --trailer "Bench: $n" --in-place "$msg"
