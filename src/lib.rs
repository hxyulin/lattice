//! Lattice, a UCI chess engine.
//!
//! One crate, three module groups, each the former workspace crate it replaces:
//! - `board` - board representation, bitboards, magic sliders, move generation,
//!   Zobrist hashing, and the incremental tapered-eval accumulator: the pure,
//!   perft-tested core.
//! - `engine` - search, evaluation, the transposition table, and the `bench`
//!   suite, built on `board`.
//! - `uci` - UCI protocol parsing and types.
//!
//! The runnable binary (the composition root wiring these to stdin/stdout) lives
//! in `src/main.rs`. The module split is a navigational convention, not a
//! compile-enforced wall: keep protocol/IO in the binary and pure logic here.

mod board;
mod engine;
mod uci;

pub use board::*;
pub use engine::*;
pub use uci::*;
