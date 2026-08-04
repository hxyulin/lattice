use nonmax::NonMaxU8;

/// A chess side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    /// White.
    White = 0,
    /// Black.
    Black = 1,
}

impl Color {
    /// Returns the opposing side.
    pub fn flip(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

/// A piece type independent of color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceType {
    /// A pawn.
    Pawn = 0,
    /// A knight.
    Knight = 1,
    /// A bishop.
    Bishop = 2,
    /// A rook.
    Rook = 3,
    /// A queen.
    Queen = 4,
    /// A king.
    King = 5,
}

impl PieceType {
    pub(crate) fn from_index(index: u8) -> Self {
        match index {
            0 => Self::Pawn,
            1 => Self::Knight,
            2 => Self::Bishop,
            3 => Self::Rook,
            4 => Self::Queen,
            5 => Self::King,
            _ => unreachable!(),
        }
    }
}

/// A color and piece type packed into one byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece(NonMaxU8);

impl Piece {
    /// Builds a piece from its color and type.
    pub fn new(color: Color, piece_type: PieceType) -> Self {
        Self(NonMaxU8::new((color as u8 & 1) << 3 | (piece_type as u8 & 7)).unwrap())
    }

    /// Returns the piece color.
    pub fn color(self) -> Color {
        if self.index() >> 3 == 0 {
            Color::White
        } else {
            Color::Black
        }
    }

    /// Returns the piece type.
    pub fn piece_type(self) -> PieceType {
        PieceType::from_index(self.index() & 7)
    }

    /// Returns the same piece with the opposing color.
    pub fn flip_color(self) -> Self {
        Self(NonMaxU8::new((self.index() ^ 8) & 15).unwrap())
    }

    /// Returns the packed index for a 16-entry piece table.
    pub fn index(self) -> u8 {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::{Color, Piece, PieceType};

    const COLORS: [Color; 2] = [Color::White, Color::Black];
    const PIECE_TYPES: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[test]
    fn pieces_roundtrip_exhaustively() {
        for color in COLORS {
            for piece_type in PIECE_TYPES {
                let piece = Piece::new(color, piece_type);
                assert_eq!(piece.color(), color);
                assert_eq!(piece.piece_type(), piece_type);
            }
        }
    }

    #[test]
    fn flips_are_involutions() {
        for color in COLORS {
            assert_eq!(color.flip().flip(), color);
            for piece_type in PIECE_TYPES {
                let piece = Piece::new(color, piece_type);
                assert_eq!(piece.flip_color().flip_color(), piece);
            }
        }
    }
}
