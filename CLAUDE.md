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

## Workspace

A Cargo workspace under `crates/`, split so protocol/IO stays out of the engine
core and the libraries stay testable without going through stdin:

- `lattice-board` (`lattice_board`) - board representation, bitboards, magic
  sliders, move generation, Zobrist hashing, and the rules. The pure,
  perft-tested core.
- `lattice-engine` (`lattice_engine`) - search, evaluation, transposition table,
  and the `bench` suite. Built on `lattice-board`.
- `lattice-uci` (`lattice_uci`) - UCI protocol parsing and types (library).
- `lattice-bin` - the runnable binary (package `lattice-bin`, output name
  `lattice`): the composition root, wiring the three libraries to stdin/stdout.

Keep protocol/IO in `lattice-bin` and pure engine logic in the libraries.
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
