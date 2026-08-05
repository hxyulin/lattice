# Contributing to Lattice

Lattice is developed test-first in the chess-engine sense: a change ships only
once a Sequential Probability Ratio Test (SPRT) shows it does not lose Elo. This
note covers that loop. For code style and layout, see [CLAUDE.md](CLAUDE.md).

## The loop

```
1. branch from main          git checkout -b feat/<thing>
2. implement ONE change
3. cargo test                (perft + unit tests stay green)
4. commit                    Conventional Commits; Bench trailer auto-stamped
5. push, SPRT vs main        on OpenBench (or tools/sprt.sh locally)
6. pass -> fast-forward merge to main, delete the branch
   fail -> delete the branch; the idea is dead, move on
```

`main` holds proven gains only: every commit on it is a change that passed its
SPRT. Failed experiments never land - they are just a deleted branch.

## One change, one SPRT

Test exactly one idea per branch. Bundling two changes into a single test makes a
result unattributable: a net-neutral SPRT could be a win plus a loss cancelling,
and you cannot tell. With only a few test machines this discipline matters more,
not less - wasted games are expensive.

Typical OpenBench bounds:

- **Gaining patch** (new pruning, new eval term, an extension): `SPRT[0, 5]` -
  H0 "no gain" vs H1 "+5 Elo". This is the default for most work here.
- **Simplification / cleanup** meant to be Elo-neutral: a non-regression bound
  like `SPRT[-5, 0]` or `SPRT[-3, 3]` - prove it does not *lose* Elo, then keep
  the smaller code.

Run STC (short time control) first; promote a passed patch to LTC only if the
change is depth-sensitive (extensions, reductions) and you want the confirmation.

## Bench signature

`lattice bench` searches a fixed position suite to a fixed depth and prints, as
its final line, the OpenBench-format signature:

```
<nodes> nodes <nps> nps
```

The **node count is deterministic and machine-independent** - it is a fingerprint
of the search, not a speed measurement (that is the `nps`, which varies by
machine). OpenBench reads this line to confirm a worker built the exact commit
under test.

Commits that touch engine source get a `Bench: <nodes>` trailer stamped
automatically (the `tools/stamp-bench.sh` pre-commit hook; docs/tooling commits
skip it). Reading the trailer:

- **The number changes** on a search change - expected. More nodes is not "worse":
  an extension *grows* the tree on purpose, pruning *shrinks* it. The number is an
  identity, not a score.
- **Two builds with identical signatures search identically** - useful as a
  no-op check (e.g. a refactor that must not change behaviour should keep the
  bench unchanged).

Default depth is 4; override with `lattice bench <depth>`. Bypass the stamp for
one commit with `SKIP_BENCH=1 git commit ...`.

## SPRT trailer

Once a test finishes, record its verdict as an `SPRT:` trailer on the feature
commit (`git commit --amend`, or `git notes`-free rebase if it has already been
pushed to a branch). Unlike `Bench:`, this one is written by hand from the
OpenBench result - nothing stamps it.

```
SPRT: <TC> <tc-spec> [<elo0>,<elo1>] LLR <llr> (<games>g) <elo> [+- <err>] Elo <verdict>
SPRT: pending (<note>)
```

Example trailers:

```
SPRT: STC 10+0.1 [0,5] LLR 2.94 (368g) +37 Elo pass
SPRT: LTC 60+0.6 [0,5] LLR 3.32 (38g) +348.7 +- 93.4 Elo pass
SPRT: STC 10+0.1 [-5,0] LLR -2.95 (1204g) -1.2 Elo fail
SPRT: pending (OpenBench rerun)
```

What `tools/gen-ledgers.sh` actually reads, field by field:

| Column in the table | Taken from | Rule |
|---|---|---|
| Feature | the commit subject | conventional-commit prefix and scope stripped (`feat(engine): x` -> `x`) |
| TC | **first whitespace-separated word** | free-form; `STC`/`LTC` by convention |
| Games | the token matching `(<digits>g)` | anywhere in the trailer; `-` if absent |
| Elo | the token **before** the literal word `Elo` | if that token is `+-`, the value three back is used instead, so error bars are skipped |
| Verdict | the **last** whitespace-separated word | free-form; `pass`/`fail` by convention |

So the parser is positional at only two points - the first word is the TC and
the last is the verdict - and keyed on the literal `Elo` and the `(Ng)` shape
for the middle two. Everything else in the line (the bounds, `LLR <n>`) is
documentation for a human reader and is not parsed. A trailer beginning with
`pending` short-circuits to a `pending` row with `-` in every other column.

Two rules that are easy to trip over:

- **Write `Elo` after the number, not before.** `+37 Elo` parses; `Elo +37`
  yields `-`, because the lookup walks backwards from the word `Elo`.
- **One commit may carry several `SPRT:` trailers** (an STC run and its LTC
  confirmation). Each becomes its own row, in trailer order.

After adding or amending a trailer, regenerate the committed ledgers:

```
tools/gen-ledgers.sh
```

It rewrites `bench.csv` in full and replaces only the table between the
`<!-- BEGIN generated: -->` / `<!-- END generated -->` markers in
[`docs/src/sprt-results.md`](docs/src/sprt-results.md); prose outside those
markers is left alone. Commit the regenerated files alongside the amendment.

## Tests and correctness

- `cargo test` - unit tests plus **perft**, the exhaustive move-generation
  correctness check (leaf-node counts against known references). Any change near
  move generation, make/unmake, or board state must keep perft green.
- Pre-commit hooks enforce `cargo fmt --check`, `cargo test`, spelling
  (`typos`), and the bench trailer. Enable them once per clone:

  ```
  pre-commit install
  ```

A green `cargo test` is the gate to *running* an SPRT; a passed SPRT is the gate
to *merging*. They check different things - correctness vs strength - and neither
substitutes for the other.

## Local SPRT (when you need it)

OpenBench is the primary test bed. Run a test locally when you want to sanity-check
a risky change before spending queue time, or to investigate a suspected serious
regression in isolation. Both scripts need [`fastchess`](https://github.com/Disservin/fastchess)
on `PATH` and an opening book.

- **`tools/sprt.sh <refA> <refB> [rounds] [book]`** - builds two git revisions in
  isolated worktrees, prints both bench signatures (they should differ), then runs
  fastchess under SPRT. Examples:

  ```
  tools/sprt.sh HEAD main          # this branch vs the baseline
  tools/sprt.sh HEAD HEAD~1        # this commit vs its parent
  LIMIT=depth=6 tools/sprt.sh HEAD main 800
  ```

  Knobs via env: `LIMIT` (any fastchess `-each` limit, default `depth=3`),
  `CONCURRENCY`, `BENCH_DEPTH`, `BOOKS_DIR`, `ENGINE_BIN`, `PGN`.

- **`tools/match.sh [elo] [games] [tc] [book]`** - a calibration gauntlet against
  Stockfish pinned to a target `UCI_Elo` under a real time control, to read off an
  *absolute* Elo (not an A/B verdict). Needs `stockfish` on `PATH` too.

A local depth-limited SPRT is result-deterministic, so it is a fast, reliable
filter - but it is not a substitute for an OpenBench run at a real time control
before merging anything non-trivial.

## Commits and branches

- **Conventional Commits**: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`,
  `test:`. Branch names follow the same prefixes (`feat/check-extensions`).
- Merge a passed branch **fast-forward** (`git merge --ff-only feat/<thing>`) to
  keep history linear, then delete it.
- Source is ASCII-only; keep inline comments minimal and doc comments on public
  items complete (`missing_docs` is a warning). See [CLAUDE.md](CLAUDE.md).
