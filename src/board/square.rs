use nonmax::NonMaxU8;

/// A board square indexed in little-endian rank-file order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Square(NonMaxU8);

impl Square {
    /// Builds a square from a zero-based file and rank.
    pub fn new(file: u8, rank: u8) -> Option<Self> {
        (file < 8 && rank < 8).then(|| Self::new_unchecked(rank * 8 + file))
    }

    /// Builds a square from its index if it is in the range `0..64`.
    pub fn from_index(index: u8) -> Option<Self> {
        (index < 64).then(|| Self::new_unchecked(index))
    }

    /// Builds a square from an index known to be in the range `0..64`.
    ///
    /// The index is masked to six bits, so the `NonMaxU8` invariant holds
    /// without a runtime check. A debug build asserts the caller's range.
    pub fn new_unchecked(index: u8) -> Self {
        debug_assert!(index < 64);
        Self(NonMaxU8::new(index & 63).unwrap())
    }

    /// Returns the zero-based file.
    pub fn file(self) -> u8 {
        self.index() & 7
    }

    /// Returns the zero-based rank.
    pub fn rank(self) -> u8 {
        self.index() >> 3
    }

    /// Returns the little-endian rank-file index.
    pub fn index(self) -> u8 {
        self.0.get()
    }

    /// Returns the square mirrored vertically across the board.
    pub fn flip_rank(self) -> Self {
        Self::new_unchecked(self.index() ^ 56)
    }
}

#[cfg(test)]
mod tests {
    use super::Square;

    #[test]
    fn squares_roundtrip_exhaustively() {
        for index in 0..64 {
            let square = Square::from_index(index).unwrap();
            assert_eq!(square.index(), index);
            assert_eq!(Square::new(square.file(), square.rank()), Some(square));
        }
    }

    #[test]
    fn rank_flip_is_an_involution() {
        for index in 0..64 {
            let square = Square::from_index(index).unwrap();
            assert_eq!(square.flip_rank().flip_rank(), square);
        }
    }

    #[test]
    fn rejects_out_of_range_coordinates() {
        assert_eq!(Square::new(8, 0), None);
        assert_eq!(Square::from_index(64), None);
    }
}
