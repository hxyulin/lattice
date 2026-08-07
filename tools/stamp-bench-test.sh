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

trailers() { grep -i '^Bench:' "$1" || true; }

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

exit $fail
