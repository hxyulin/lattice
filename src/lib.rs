//! Lattice, a UCI chess engine.

/// Fixed search benchmark.
pub mod bench;
/// Core chess board value types.
pub mod board;
/// Static position evaluation.
pub mod eval;
/// Chess attack and move generation.
pub mod movegen;
/// Position search and time management.
pub mod search;
/// Transposition table.
pub mod tt;
/// UCI command parsing and move notation.
pub mod uci;

pub use board::{
    Bitboard, Board, CastlingRights, Color, FenError, Move, MoveType, Piece, PieceType, Square,
    State, Undo,
};
