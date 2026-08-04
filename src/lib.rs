//! Lattice, a UCI chess engine.

/// Core chess board value types.
pub mod board;
/// Chess attack and move generation.
pub mod movegen;

pub use board::{
    Bitboard, Board, CastlingRights, Color, FenError, Move, MoveType, Piece, PieceType, Square,
    State, Undo,
};
