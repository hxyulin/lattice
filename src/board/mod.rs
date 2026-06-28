//! Lattice-board contains the core chess logic and primitives for lattice-engine.
//! This includes:
//!  - Board representation
//!  - Pieces and squares
//!  - Moves and move generation
//!  - Perft testing

mod bitboard;
mod board;
mod r#move;
mod movegen;
mod movelist;
mod piece;
mod square;

pub use {bitboard::*, board::*, r#move::*, movelist::*, piece::*, square::*};
