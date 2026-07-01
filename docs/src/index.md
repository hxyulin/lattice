# Lattice

Lattice is a [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface)
chess engine written in Rust.

This book is written alongside the engine: each feature is documented as it
lands, so the explanation and the code grow together. It is a draft and will
fill out commit by commit.

## Crate layout

Lattice is a single crate, organized into module groups:

- **`board`** - board representation, move generation, and the rules.
- **`engine`** - search and evaluation (the "brain").
- **`uci`** - the UCI protocol front end.

The grouping keeps the rules independently testable (via perft) while the engine
and protocol layers evolve on top. It is a navigational convention, not a
compile-enforced boundary: the binary in `src/main.rs` wires the modules to
stdin/stdout.
