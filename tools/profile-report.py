#!/usr/bin/env python3
"""Aggregate a macOS sample(1) profile into a report about the search.

Three views, because the flat one alone is misleading:

  1. Flat self time, with Rust symbols demangled to readable names.
  2. Inclusive cost of the functions that own a strategy (generate_legal,
     qsearch, order_moves), which is what a design decision actually costs.
  3. Caller attribution for make/unmake and is_attacked, which is where a
     flat profile hides the difference between work the search needs and
     work a filter throws away.

Usage: tools/profile-report.py [profile.txt]
"""

import collections
import re
import sys

# Substring -> readable name. Rust's v0 mangling is regular enough that
# matching on the path fragment is more robust than a full demangler, and it
# avoids depending on rustfilt being installed.
NAMES = [
    ("Board4make", "Board::make"),
    ("Board6unmake", "Board::unmake"),
    ("6search7qsearch", "search::qsearch"),
    ("6search7negamax", "search::negamax"),
    ("6search12negamax_root", "search::negamax_root"),
    ("6search11order_moves", "search::order_moves"),
    ("6search9order_key", "search::order_key"),
    ("6search8is_legal", "search::is_legal"),
    ("11is_attacked", "movegen::is_attacked"),
    ("12rook_attacks", "magic::rook_attacks"),
    ("14bishop_attacks", "magic::bishop_attacks"),
    ("13queen_attacks", "magic::queen_attacks"),
    ("14push_pawn_move", "pawn::push_pawn_move"),
    ("4pawn17generate_captures", "pawn::generate_captures"),
    ("4pawn8generate", "pawn::generate"),
    ("7movegen14generate_legal", "movegen::generate_legal"),
    ("7movegen15generate_pseudo", "movegen::generate_pseudo"),
    ("7movegen17generate_captures", "movegen::generate_captures"),
    ("7movegen8generate", "movegen::generate"),
    ("4eval8evaluate", "eval::evaluate"),
    ("4eval4scan", "eval::scan"),
    ("small_sort_general", "core::sort small_sort_general"),
    ("sort4_stable", "core::sort sort4_stable"),
    ("sort8_stable", "core::sort sort8_stable"),
    ("9quicksort", "core::sort quicksort"),
    ("insertion_sort_shift_left", "core::sort insertion_sort"),
    ("7ipnsort", "core::sort ipnsort"),
    ("spec_from_iter", "Vec::from_iter (sort_by_cached_key)"),
]


def short(sym):
    for pat, name in NAMES:
        if pat in sym:
            return name
    if any(x in sym for x in ("malloc", "free", "dealloc", "rdl_alloc")):
        return "(allocator)"
    if "memset" in sym or "bzero" in sym:
        return "(memset)"
    if "mach_absolute_time" in sym:
        return "(clock)"
    return sym[:60]


def parse_tree(text):
    """Yield (indent, samples, symbol) for each call-tree line.

    Stops at the trailing summary sections. Those are flat lists that look
    like tree lines to a regex but carry no ancestors, so reading them as
    tree would double-count every symbol into an "unknown caller" bucket.
    """
    tree = text.split("Total number in stack")[0]
    if "Call graph:" in tree:
        tree = tree.split("Call graph:", 1)[1]
    for line in tree.split("\n"):
        m = re.match(r"^([ +!:|]*)(\d+)\s+(\S+)", line)
        if m:
            yield len(m.group(1)), int(m.group(2)), m.group(3)


def flat(text):
    """The collapsed self-time table at the end of the profile."""
    if "same collapsed (when >= 5):" not in text:
        return []
    seg = text.split("same collapsed (when >= 5):")[1].split("Binary Images:")[0]
    rows = []
    for line in seg.strip().split("\n"):
        m = re.match(r"\s*(\S+)\s+\(in (\S+)\)\s+(\d+)", line)
        if m:
            rows.append((m.group(1), int(m.group(3))))
    # `read` is the main thread blocked on stdin, not search work.
    return [r for r in rows if r[0] != "read"]


def inclusive(rows, needle):
    """Total samples under `needle`, counting each outermost frame once."""
    total, depth = 0, None
    for indent, n, sym in rows:
        if depth is not None and indent <= depth:
            depth = None
        if needle in sym and depth is None:
            total += n
            depth = indent
    return total


def attribute(rows, targets, callers):
    """Split each target's samples by its nearest enclosing caller."""
    out = collections.Counter()
    stack = []
    for indent, n, sym in rows:
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if any(t in sym for t in targets):
            parent = "(other)"
            for _, s in reversed(stack):
                hit = next((c for c in callers if c in s), None)
                if hit:
                    parent = hit
                    break
            out[(short(sym), parent)] += n
        stack.append((indent, sym))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lattice-profile.txt"
    text = open(path).read()
    rows = list(parse_tree(text))

    self_time = flat(text)
    total = sum(n for _, n in self_time)
    if not total:
        print(f"no samples found in {path}", file=sys.stderr)
        return 1

    agg = collections.Counter()
    for sym, n in self_time:
        agg[short(sym)] += n

    print(f"\nsearch-thread samples: {total}  (stdin `read` excluded)\n")
    print("== self time ==")
    for name, n in agg.most_common(18):
        print(f"  {100 * n / total:5.1f}%  {n:6d}  {name}")

    print("\n== inclusive cost of a strategy ==")
    for needle, label in [
        ("generate_legal", "movegen::generate_legal"),
        ("generate_pseudo", "movegen::generate_pseudo"),
        ("6search7qsearch", "search::qsearch"),
        ("order_moves", "search::order_moves"),
    ]:
        n = inclusive(rows, needle)
        if n:
            print(f"  {100 * n / total:5.1f}%  {n:6d}  {label}")

    def report(title, targets, callers):
        print(f"\n== {title} ==")
        calls = attribute(rows, targets, callers)
        total_calls = sum(calls.values()) or 1
        for (fn, parent), n in calls.most_common():
            print(f"  {100 * n / total_calls:5.1f}%  {n:6d}  {fn:20s} <- {parent}")
        # A large unattributed share means the caller list above is missing
        # something real, so the split below it cannot be trusted. Say so
        # rather than letting it read as a small rounding bucket.
        other = sum(n for (_, p), n in calls.items() if p == "(other)")
        if other > total_calls // 10:
            print(
                f"  WARNING: {100 * other / total_calls:.0f}% unattributed - "
                "the caller list is incomplete, treat this split as unreliable"
            )

    report(
        "who pays for make/unmake",
        ["Board4make", "Board6unmake"],
        ["generate_legal", "qsearch", "negamax_root", "negamax"],
    )
    report(
        "who pays for is_attacked",
        ["11is_attacked"],
        ["generate_legal", "is_legal", "qsearch", "negamax_root", "negamax"],
    )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
