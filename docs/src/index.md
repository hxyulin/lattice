# Lattice

Lattice is a [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface)
chess engine written in Rust.

This book is written alongside the engine: each feature is documented as it
lands, so the explanation and the code grow together. It is a draft and will
fill out commit by commit.

## Workspace layout

Lattice is a Cargo workspace of three crates:

- **`lattice-board`** - board representation, move generation, and the rules.
- **`lattice-engine`** - search and evaluation (the "brain").
- **`lattice-uci`** - the UCI protocol front end.

The split keeps the rules independently testable (via perft) while the engine
and protocol layers evolve on top.
