/// One of the two players / piece colors.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Color {
    /// The side moving up the board (ranks 1->8).
    White = 0,
    /// The side moving down the board (ranks 8->1).
    Black = 1,
}

impl Color {
    /// Build a color from its raw discriminant (`0 = White`, `1 = Black`).
    ///
    /// # Notes
    /// Out-of-range values debug-assert and otherwise map to `Black`.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        debug_assert!(value < 2, "invalid color value >= 2");
        match value {
            0 => Color::White,
            _ => Color::Black,
        }
    }

    /// The raw discriminant (`White = 0`, `Black = 1`).
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The opposing color.
    #[inline]
    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl From<u8> for Color {
    #[inline]
    fn from(value: u8) -> Self {
        Self::from_u8(value)
    }
}

/// A piece kind, independent of color.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum PieceType {
    /// Pawn.
    Pawn = 0,
    /// Knight.
    Knight = 1,
    /// Bishop.
    Bishop = 2,
    /// Rook.
    Rook = 3,
    /// Queen.
    Queen = 4,
    /// King.
    King = 5,
}

impl PieceType {
    /// Build a piece type from its raw discriminant (`0 = Pawn` ... `5 = King`).
    ///
    /// # Notes
    /// Out-of-range values debug-assert and otherwise map to `King`.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        debug_assert!(value < 6, "invalid piece type >= 6");
        match value {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            _ => PieceType::King,
        }
    }

    /// The raw discriminant (`Pawn = 0` ... `King = 5`).
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

impl From<u8> for PieceType {
    #[inline]
    fn from(value: u8) -> Self {
        Self::from_u8(value)
    }
}

use nonmax::NonMaxU8;

/// A colored piece packed into one byte as `(piece_type << 1) | color`.
///
/// # Notes
/// White pieces are even, black odd. The `NonMaxU8` wrapper keeps
/// `Option<Piece>` one byte via the `0xFF` niche.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Piece(NonMaxU8);

impl Piece {
    /// White pawn.
    pub const WHITE_PAWN: Self = Self::new(Color::White, PieceType::Pawn);
    /// Black pawn.
    pub const BLACK_PAWN: Self = Self::new(Color::Black, PieceType::Pawn);
    /// White knight.
    pub const WHITE_KNIGHT: Self = Self::new(Color::White, PieceType::Knight);
    /// Black knight.
    pub const BLACK_KNIGHT: Self = Self::new(Color::Black, PieceType::Knight);
    /// White bishop.
    pub const WHITE_BISHOP: Self = Self::new(Color::White, PieceType::Bishop);
    /// Black bishop.
    pub const BLACK_BISHOP: Self = Self::new(Color::Black, PieceType::Bishop);
    /// White rook.
    pub const WHITE_ROOK: Self = Self::new(Color::White, PieceType::Rook);
    /// Black rook.
    pub const BLACK_ROOK: Self = Self::new(Color::Black, PieceType::Rook);
    /// White queen.
    pub const WHITE_QUEEN: Self = Self::new(Color::White, PieceType::Queen);
    /// Black queen.
    pub const BLACK_QUEEN: Self = Self::new(Color::Black, PieceType::Queen);
    /// White king.
    pub const WHITE_KING: Self = Self::new(Color::White, PieceType::King);
    /// Black king.
    pub const BLACK_KING: Self = Self::new(Color::Black, PieceType::King);

    /// A piece of `color` and `piece` type.
    #[inline]
    #[must_use]
    pub const fn new(color: Color, piece: PieceType) -> Self {
        // SAFETY: piece < 6, so (piece << 1) | color < 12 - never the 0xFF niche.
        Self(unsafe { NonMaxU8::new_unchecked((color as u8) | ((piece as u8) << 1)) })
    }

    /// Is this a white piece?
    #[inline]
    #[must_use]
    pub const fn is_white(self) -> bool {
        matches!(self.color(), Color::White)
    }

    /// Is this a black piece?
    #[inline]
    #[must_use]
    pub const fn is_black(self) -> bool {
        matches!(self.color(), Color::Black)
    }

    /// Is this a pawn (of either color)?
    #[inline]
    #[must_use]
    pub const fn is_pawn(self) -> bool {
        matches!(self.piece(), PieceType::Pawn)
    }

    /// Is this a knight (of either color)?
    #[inline]
    #[must_use]
    pub const fn is_knight(self) -> bool {
        matches!(self.piece(), PieceType::Knight)
    }

    /// Is this a bishop (of either color)?
    #[inline]
    #[must_use]
    pub const fn is_bishop(self) -> bool {
        matches!(self.piece(), PieceType::Bishop)
    }

    /// Is this a rook (of either color)?
    #[inline]
    #[must_use]
    pub const fn is_rook(self) -> bool {
        matches!(self.piece(), PieceType::Rook)
    }

    /// Is this a queen (of either color)?
    #[inline]
    #[must_use]
    pub const fn is_queen(self) -> bool {
        matches!(self.piece(), PieceType::Queen)
    }

    /// Is this a king (of either color)?
    #[inline]
    #[must_use]
    pub const fn is_king(self) -> bool {
        matches!(self.piece(), PieceType::King)
    }

    /// Build a piece from its packed `(piece_type << 1) | color` byte (`0..12`).
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        debug_assert!(value < 12, "invalid piece value >= 12");
        // SAFETY: a valid piece value is < 12 - never the 0xFF niche.
        Self(unsafe { NonMaxU8::new_unchecked(value) })
    }

    /// This piece's color.
    #[inline]
    #[must_use]
    pub const fn color(self) -> Color {
        Color::from_u8(self.as_u8() & 1)
    }

    /// This piece's type, ignoring color.
    #[inline]
    #[must_use]
    pub const fn piece(self) -> PieceType {
        PieceType::from_u8(self.as_u8() >> 1)
    }

    /// The packed `(piece_type << 1) | color` byte (`0..12`).
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0.get()
    }
}

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Piece")
            .field("color", &self.color())
            .field("piece", &self.piece())
            .finish()
    }
}

// `Option<Piece>` stays one byte via the 0xFF niche on `NonMaxU8`.
const _: () = assert!(std::mem::size_of::<Option<Piece>>() == 1);

// Proof the `from_u8` conversions and accessors are `const`-callable.
const _: () = {
    let p = Piece::from_u8(7); // (Rook << 1) | Black
    assert!(matches!(p.color(), Color::Black));
    assert!(matches!(p.piece(), PieceType::Rook));
    assert!(matches!(Color::White.flip(), Color::Black));
};

#[cfg(test)]
mod tests {
    use super::*;

    const PIECE_INDICES: &[(Color, PieceType)] = &[
        (Color::White, PieceType::Pawn),
        (Color::Black, PieceType::Pawn),
        (Color::White, PieceType::Knight),
        (Color::Black, PieceType::Knight),
        (Color::White, PieceType::Bishop),
        (Color::Black, PieceType::Bishop),
        (Color::White, PieceType::Rook),
        (Color::Black, PieceType::Rook),
        (Color::White, PieceType::Queen),
        (Color::Black, PieceType::Queen),
        (Color::White, PieceType::King),
        (Color::Black, PieceType::King),
    ];

    #[test]
    fn test_piece_from_u8() {
        for (i, (color, piece)) in PIECE_INDICES.iter().enumerate() {
            let piece_from_u8 = Piece::from_u8(i as u8);
            assert_eq!(piece_from_u8.color(), *color);
            assert_eq!(piece_from_u8.piece(), *piece);
        }
    }

    #[test]
    fn test_piece_constructor() {
        for (i, (color, piece)) in PIECE_INDICES.iter().enumerate() {
            let piece_from_constructor = Piece::new(*color, *piece);
            assert_eq!(piece_from_constructor.color(), *color);
            assert_eq!(piece_from_constructor.piece(), *piece);
            assert_eq!(piece_from_constructor.as_u8(), i as u8);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_piece_out_of_bounds() {
        let _ = Piece::from_u8(12);
    }
}
