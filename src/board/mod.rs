//! The core chess logic and primitives the engine is built on:
//!  - Board representation
//!  - Pieces and squares
//!  - Moves and move generation
//!  - Perft testing

mod bitboard;
// The board representation lives in `board.rs` inside this `board` module; the
// repeated name is intentional, so silence module-inception here.
#[allow(clippy::module_inception)]
mod board;
mod magic;
mod r#move;
mod movegen;
mod movelist;
mod pesto;
mod piece;
mod square;
mod zobrist;

pub use magic::init_tables;
pub use zobrist::ZobristHash;
pub use {bitboard::*, board::*, r#move::*, movelist::*, piece::*, square::*};
