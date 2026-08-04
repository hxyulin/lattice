use super::{Color, Square};

/// Castling availability stored as four bits.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CastlingRights(pub(crate) u8);

impl CastlingRights {
    /// White may castle kingside.
    pub const WHITE_KING: Self = Self(1);
    /// White may castle queenside.
    pub const WHITE_QUEEN: Self = Self(2);
    /// Black may castle kingside.
    pub const BLACK_KING: Self = Self(4);
    /// Black may castle queenside.
    pub const BLACK_QUEEN: Self = Self(8);
    /// Returns no castling rights.
    pub const fn none() -> Self {
        Self(0)
    }
    /// Returns whether all supplied rights are present.
    pub const fn contains(self, rights: Self) -> bool {
        self.0 & rights.0 == rights.0
    }
}

/// Reversible and irreversible game state associated with a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    pub(crate) side_to_move: Color,
    pub(crate) castling: CastlingRights,
    pub(crate) ep: Option<Square>,
    pub(crate) halfmove: u8,
    pub(crate) fullmove: u16,
    pub(crate) zobrist: u64,
}

impl State {
    /// Returns the side to move.
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }
    /// Returns the castling rights.
    pub const fn castling(&self) -> CastlingRights {
        self.castling
    }
    /// Returns the en passant target square.
    pub const fn en_passant(&self) -> Option<Square> {
        self.ep
    }
    /// Returns the halfmove clock.
    pub const fn halfmove_clock(&self) -> u8 {
        self.halfmove
    }
    /// Returns the fullmove number.
    pub const fn fullmove_number(&self) -> u16 {
        self.fullmove
    }
    /// Returns the position's Zobrist key.
    pub const fn zobrist(&self) -> u64 {
        self.zobrist
    }
}

/// Information saved before a move for exact restoration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    pub(crate) state: State,
    pub(crate) captured: Option<super::Piece>,
}
