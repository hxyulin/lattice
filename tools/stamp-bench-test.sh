#!/bin/bash

# Self-check for stamp-bench.sh's trailer handling, which is what cherry-pick
# and rebase get wrong: they carry a `Bench:` number forward from a tree that
# no longer exists, and a stale signature in bench.csv is invisible once it is
# committed.
#
# The build/bench half is not exercised - that would cost a release build per
# case. Everything here is the trailer logic around it, with the bench stubbed.
#
#   tools/stamp-bench-test.sh
set -eu
cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0

check() {
  local name="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    echo "ok   $name"
  else
    echo "FAIL $name"
    echo "     expected: $(echo "$expected" | tr '\n' '/')"
    echo "     actual:   $(echo "$actual" | tr '\n' '/')"
    fail=1
  fi
}

mkdir -p "$tmp/bin" "$tmp/repo/tools" "$tmp/repo/target/release"

# A fake cargo that succeeds, and a fake engine binary that prints whatever
# the current case put in $tmp/nodes.
cat > "$tmp/bin/cargo" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$tmp/bin/cargo"

setup_repo() {
  rm -rf "$tmp/repo"
  mkdir -p "$tmp/repo/target/release"
  cd "$tmp/repo"
  git init -q .
  git config user.email t@t
  git config user.name t
  echo "fn main() {}" > src.rs
  git add src.rs
  cat > target/release/lattice <<EOF
#!/bin/sh
echo "Nodes searched: \$(cat "$tmp/nodes")"
echo "Nodes/second: 1"
EOF
  chmod +x target/release/lattice
  cd - >/dev/null
}

run_hook() {
  local msgfile="$1" nodes="$2"
  printf '%s' "$nodes" > "$tmp/nodes"
  (cd "$tmp/repo" && PATH="$tmp/bin:$PATH" \
    bash "$OLDPWD/tools/stamp-bench.sh" "$msgfile" >/dev/null 2>&1)
}

# Commits whatever setup_repo staged, so the next hook run sees the state an
# amend sees: a clean index over a commit that already holds the source.
commit_staged() {
  (cd "$tmp/repo" && git commit -q -m "${1:-a commit}")
}

# The trailer, not the subject: a commit titled `bench: ...` also matches a
# case-insensitive prefix, and conflating the two is what let the strip bug
# hide.
trailers() { tail -n +2 "$1" | grep -i '^Bench:' || true; }

# The cherry-pick case: an inherited trailer names a tree that no longer
# exists, so it must be replaced, and exactly once.
setup_repo
printf 'perf: something\n\nBench: 1274707\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "stale inherited trailer is replaced" "Bench: 1274778" "$(trailers "$tmp/msg")"
check "trailer is not duplicated" "1" "$(trailers "$tmp/msg" | wc -l | tr -d ' ')"

# The ordinary case: no trailer yet, one gets added.
setup_repo
printf 'feat: something\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "fresh commit is stamped" "Bench: 1274778" "$(trailers "$tmp/msg")"

# A build failure must leave no trailer rather than an inherited one: a
# missing signature is recoverable, a wrong one silently poisons bench.csv.
setup_repo
cat > "$tmp/bin/cargo" <<'EOF'
#!/bin/sh
exit 1
EOF
chmod +x "$tmp/bin/cargo"
printf 'perf: something\n\nBench: 1274707\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "failed build drops the stale trailer" "" "$(trailers "$tmp/msg")"
cat > "$tmp/bin/cargo" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$tmp/bin/cargo"

# Other trailers must survive the rewrite.
setup_repo
printf 'perf: x\n\nBench: 1\nCo-Authored-By: a <a@a>\n' > "$tmp/msg"
run_hook "$tmp/msg" 42
check "unrelated trailers survive" "Co-Authored-By: a <a@a>" \
  "$(grep '^Co-Authored-By:' "$tmp/msg" || true)"

# Docs-only commits skip the rebuild entirely and stay unstamped.
setup_repo
(cd "$tmp/repo" && git rm -q --cached src.rs && echo hi > README.md && git add README.md)
printf 'docs: something\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "docs-only commit is not stamped" "" "$(trailers "$tmp/msg")"

# `git commit --amend` with nothing staged: the index is empty but the commit
# being rewritten is full of Rust, so the guard has to look past the index or
# it silently skips a commit that needs a signature. This is the case that
# shipped unstamped.
setup_repo
commit_staged "perf: a rust change"
printf 'perf: a rust change, reworded\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "amend with an empty index is stamped" "Bench: 1274778" "$(trailers "$tmp/msg")"

# The same amend, but the commit being rewritten carried a stale number: it
# must be replaced rather than kept, exactly as on the cherry-pick path.
setup_repo
commit_staged "perf: a rust change"
printf 'perf: reworded\n\nBench: 1274707\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "amend replaces a stale trailer" "Bench: 1274778" "$(trailers "$tmp/msg")"

# Amending a docs file onto a Rust commit is the one case this gets wrong,
# and deliberately so: with a non-empty index it is indistinguishable from an
# ordinary docs commit following a Rust one, since git offers no reliable
# amend signal. The two want opposite answers, so one has to lose.
#
# The stamp loses. A missing trailer is a visible gap in bench.csv that
# re-running the hook fixes; a spurious one is a wrong row that reads as
# real. Stage the source alongside the docs, or amend with an empty index,
# and it stamps.
setup_repo
commit_staged "perf: a rust change"
(cd "$tmp/repo" && echo hi > README.md && git add README.md)
printf 'perf: plus a readme\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "amend adding only docs to a rust commit skips (known limit)" "" \
  "$(trailers "$tmp/msg")"

# catches: a case-insensitive strip eating the subject. This repo titles
# commits `bench: ...`, and `grep -vi '^Bench:'` deleted that line along with
# the trailer, silently promoting the body's first line to subject. It
# happened twice to a real commit before it was noticed.
setup_repo
printf 'bench: measure something\n\nA body paragraph.\n\nBench: 1274707\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "a bench: subject survives the strip" "bench: measure something" \
  "$(head -1 "$tmp/msg")"
check "and the trailer is still replaced" "Bench: 1274778" "$(trailers "$tmp/msg")"

# catches: reading HEAD when the index already answered. An ordinary
# docs-only commit landing right after a Rust commit must not inherit that
# commit's answer - gen-ledgers.sh turns every trailer into a bench.csv row,
# so a spurious stamp is a wrong row rather than a wasted rebuild. This is
# the regression the first version of the amend fix introduced.
setup_repo
commit_staged "perf: a rust change"
(cd "$tmp/repo" && echo hi > README.md && git add README.md)
printf 'docs: after a rust commit\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "docs commit after a rust commit is not stamped" "" "$(trailers "$tmp/msg")"

# A docs-only commit amended on top of another docs-only commit must still
# skip: neither the index nor the rewritten commit holds any Rust.
setup_repo
(cd "$tmp/repo" && git rm -q --cached src.rs && echo hi > README.md && git add README.md)
commit_staged "docs: something"
printf 'docs: reworded\n' > "$tmp/msg"
run_hook "$tmp/msg" 1274778
check "amend of a docs-only commit is not stamped" "" "$(trailers "$tmp/msg")"

# catches: a `--no-ff` merge being stamped. `git merge` sets the message
# source to `merge` only for the fast-forward-refusing default path; with
# `--no-ff` the variable arrives empty, so the env check does not fire, and
# the merge's staged files are the branch's Rust files. The result was a
# `Bench:` trailer on a merge commit, which gen-ledgers.sh turns into a
# duplicate bench.csv row for a feature already counted at its own commit.
setup_repo
commit_staged "feat: a rust change"
(cd "$tmp/repo" &&
  git checkout -q -b side &&
  echo "fn main() { let _ = 1; }" > src.rs &&
  git add src.rs && git commit -q -m "feat: side change" &&
  git checkout -q - &&
  git merge --no-ff --no-commit side >/dev/null 2>&1 || true)
printf 'Merge branch side\n' > "$tmp/msg"
# Explicitly empty, which is what --no-ff actually passes.
PRE_COMMIT_COMMIT_MSG_SOURCE="" run_hook "$tmp/msg" 1274778
check "a --no-ff merge is not stamped" "" "$(trailers "$tmp/msg")"

exit $fail
