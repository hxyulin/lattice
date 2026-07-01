//! Lattice, a UCI chess engine.
//!
//! One crate, three module groups:
//! - `board` - board representation, bitboards, magic sliders, move generation,
//!   and the pure, perft-tested core.
//! - `engine` - search and evaluation, built on `board`.
//! - `uci` - UCI protocol parsing and types.

mod board;
mod engine;
mod uci;

pub use board::*;
pub use engine::*;
pub use uci::*;
