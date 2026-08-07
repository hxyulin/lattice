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
# The env var alone is not enough. `git merge --no-ff` leaves
# PRE_COMMIT_COMMIT_MSG_SOURCE empty, so the case above does not fire, and the
# merge's staged files are the branch's Rust files - which stamped a merge
# commit with a `Bench:` trailer it must not carry. gen-ledgers.sh turns every
# trailer into a bench.csv row, so that is a duplicate row for a feature
# already counted at its own commit.
#
# MERGE_HEAD is git's own record that a merge is in progress, present for
# every form of it, so it is asked directly rather than inferred.
if [ -f "$(git rev-parse --git-dir 2>/dev/null)/MERGE_HEAD" ]; then exit 0; fi
# An existing trailer is not trusted: cherry-pick and rebase carry the number
# forward from a tree that no longer exists, so it is re-derived and replaced.
#
# Only Rust/Cargo changes move the node count; everything else skips the
# rebuild. What counts as "changed" has to be the content the commit will
# carry, not the index: `git commit --amend` with nothing staged leaves
# `git diff --cached` empty while still producing a commit full of Rust, and
# the hook then skipped a commit that needed stamping.
#
# An amend cannot be detected from the hook's arguments: git reports the
# message source as `commit` only for the editor form, while
# `commit --amend -m` and an ordinary `-m` commit both report `message`, and
# no state file separates them. What does separate them is the index. A
# commit with nothing staged would be rejected as empty, so an empty index
# here means the content is coming from the commit being rewritten - and
# `HEAD` is that commit.
#
# Falling back rather than unioning matters: `HEAD` is the *previous* commit
# during an ordinary commit, and including it there stamped a tools-only
# commit that happened to follow a Rust one. gen-ledgers.sh turns every
# trailer into a bench.csv row, so that is a wrong row, not a wasted rebuild.
# ponytail: amending only non-Rust files onto a Rust commit is not stamped -
# that state is indistinguishable from an ordinary docs commit following a
# Rust one. Stage the source too, or amend with an empty index.
changed=$(git diff --cached --name-only)
[ -z "$changed" ] && changed=$(git show --pretty= --name-only HEAD 2>/dev/null)
printf '%s\n' "$changed" | grep -qE '\.rs$|Cargo\.(toml|lock)$' || exit 0
# Dropped before the build, not after: if the build or bench below fails this
# bails out, and an inherited number left behind would be worse than none.
#
# The subject line is never a trailer, however it is spelled. `grep -vi
# '^Bench:'` deleted the subject of a commit titled `bench: ...` along with
# the trailer, which silently promoted the body's first line to subject.
# Skipping line 1 is enough: a trailer cannot be there.
{ head -1 "$msg"; tail -n +2 "$msg" | grep -vi '^Bench:'; } > "$msg.stamp" &&
  mv "$msg.stamp" "$msg"
cargo build --release --bin lattice -q 2>/dev/null || exit 0
n=$(./target/release/lattice bench </dev/null 2>&1 \
      | sed -n 's/^Nodes searched:[[:space:]]*\([0-9]*\).*/\1/p' | tail -1)
[ -z "$n" ] && exit 0
git interpret-trailers --if-exists replace --trailer "Bench: $n" --in-place "$msg"
