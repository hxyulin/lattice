//! Core chess board value types.

mod bitboard;
#[allow(clippy::module_inception)]
mod board;
mod fen;
/// Packed chess move types.
pub mod r#move;
mod piece;
mod square;
mod state;

pub use bitboard::Bitboard;
pub use board::Board;
pub use fen::FenError;
pub use r#move::{Move, MoveType};
pub use piece::{Color, Piece, PieceType};
pub use square::Square;
pub use state::{CastlingRights, State, Undo};

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::{Move, Piece, Square};

    #[test]
    fn niches_are_free() {
        assert_eq!(size_of::<Square>(), 1);
        assert_eq!(size_of::<Option<Square>>(), 1);
        assert_eq!(size_of::<Piece>(), 1);
        assert_eq!(size_of::<Option<Piece>>(), 1);
        assert_eq!(size_of::<Move>(), 2);
        assert_eq!(size_of::<Option<Move>>(), 2);
    }
}
