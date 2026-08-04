//! Lattice, a UCI chess engine.

/// The side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    /// White.
    White,
    /// Black.
    Black,
}

impl Color {
    /// The opposing side.
    pub fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// A piece kind, independent of color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    /// Pawn.
    Pawn,
    /// Knight.
    Knight,
    /// Bishop.
    Bishop,
    /// Rook.
    Rook,
    /// Queen.
    Queen,
    /// King.
    King,
}

/// A board square, indexed 0..64 in LERF order: 0 is a1, 7 is h1, 63 is h8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Square(u8);

impl Square {
    /// Builds a square from a file and rank, both 0..8. Returns `None` if
    /// either is out of range.
    pub fn new(file: u8, rank: u8) -> Option<Self> {
        (file < 8 && rank < 8).then(|| Square(rank * 8 + file))
    }

    /// The square's file, 0..8 (0 is the a-file).
    pub fn file(self) -> u8 {
        self.0 % 8
    }

    /// The square's rank, 0..8 (0 is rank 1).
    pub fn rank(self) -> u8 {
        self.0 / 8
    }

    /// The square's LERF index, 0..64.
    pub fn index(self) -> u8 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_roundtrips_file_and_rank() {
        for rank in 0..8 {
            for file in 0..8 {
                let sq = Square::new(file, rank).unwrap();
                assert_eq!((sq.file(), sq.rank()), (file, rank));
            }
        }
        assert_eq!(Square::new(0, 0).unwrap().index(), 0);
        assert_eq!(Square::new(7, 7).unwrap().index(), 63);
        assert!(Square::new(8, 0).is_none());
    }

    #[test]
    fn color_flips() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }
}
