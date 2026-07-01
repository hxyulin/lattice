//! Lattice-board contains the core chess logic and primitives for lattice-engine.
//! This includes:
//!  - Board representation
//!  - Pieces and squares
//!  - Moves and move generation
//!  - Perft testing

mod bitboard;
mod board;
mod magic;
mod r#move;
mod movegen;
mod movelist;
#[cfg(feature = "nnue")]
mod nnue;
mod pesto;
mod piece;
mod square;
mod zobrist;

pub use magic::init_tables;
pub use zobrist::ZobristHash;
pub use {bitboard::*, board::*, r#move::*, movelist::*, piece::*, square::*};
