# Lattice

A UCI chess engine written in Rust. The public engine name is **Lattice**;
the workspace crates are named `lattice-*`.

## Collaboration

1. Ask for clarification rather than assuming intent. If a request is vague,
   ambiguous, or looks wrong, stop and ask before acting.
2. Consider multiple approaches and surface the trade-offs instead of silently
   picking one. Chess engines have deep trade-offs (board representation, move
   generation, search pruning, evaluation) - make them explicit.
3. Plan before you build. Discuss the design, edge cases, and integration points
   first; begin implementing only once the plan is agreed upon.

## Crate

A single crate (package `lattice`, output name `lattice`), organized into module
groups so protocol/IO stays out of the engine core. The grouping is a
navigational convention, not a compile-enforced wall:

- `src/board/` - board representation, bitboards, magic sliders, move generation,
  Zobrist hashing, the incremental tapered-eval accumulator, and the rules. The
  pure, perft-tested core.
- `src/engine/` - search, evaluation, transposition table, and the `bench`
  suite. Built on `board`.
- `src/uci.rs` - UCI protocol parsing and types.
- `src/main.rs` - the runnable binary: the composition root, wiring the modules
  to stdin/stdout. `src/lib.rs` re-exports the three groups flat.

Keep protocol/IO in `src/main.rs` and pure engine logic in the modules.
Documentation lives in `docs/` (mdbook) and is written alongside the code.

The engine is OpenBench-compatible: a root `Makefile` (`make EXE=...` =>
`target/release/lattice`) and a `bench` subcommand whose final line is
`<nodes> nodes <nps> nps`, for distributed SPRT testing.

## Code Style

- Limit the amount of comments to a strict minimum. Almost never add comments,
  except sometimes on non-trivial code, on function definitions whose arguments
  are not self-explanatory, and on type definitions and their members.
- Do not use emoji.
- Source code is ASCII only: no smart quotes, em/en dashes, arrows, or other
  non-ASCII characters. Spell them out (`-`, `->`, `=>`, `...`).
- Public items carry doc comments (`missing_docs` is a warning). Doc comments on
  the public API are required; inline comments stay minimal.

## Conventions

- Edition 2024.
- Lints: clippy `all = "warn"` (workspace); `missing_docs = "warn"` per crate.
- Logging: never `println!`/`eprintln!` in library code. In the UCI binary or
  example, stdout is the protocol channel - emit only valid UCI there.
- Tests: non-trivial logic carries at least one test. Perft tests are the
  standard correctness check for move generation.
- Commits: Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`,
  `refactor:`, `test:`).
- Formatting and spelling are enforced via [pre-commit](https://pre-commit.com)
  (`cargo fmt --check` plus `typos`), configured in `.pre-commit-config.yaml`.
  Enable it once per clone with `pre-commit install`.
