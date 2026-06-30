use std::{fmt::Write, str::FromStr};

use nonmax::NonMaxU8;

/// Representation of a square on a chess board
///
/// Follows the convention 0 = a1, 7 = h1, 63 = h8
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Square(NonMaxU8);

impl Square {
    /// File letters indexed by file number (`0 = 'a'` ... `7 = 'h'`).
    pub const FILES: [u8; 8] = *b"abcdefgh";
    /// Rank digits indexed by rank number (`0 = '1'` ... `7 = '8'`).
    pub const RANKS: [u8; 8] = *b"12345678";

    /// The square at `rank`/`file` (both `0..8`). Debug-asserts the bounds.
    #[inline]
    #[must_use]
    pub const fn new(rank: u8, file: u8) -> Self {
        debug_assert!(rank < 8, "invalid rank >= 8");
        debug_assert!(file < 8, "invalid file>= 8");
        // SAFETY: rank < 8 and file < 8, so the index is < 64 - never the 0xFF niche.
        Self(unsafe { NonMaxU8::new_unchecked((rank << 3) + file) })
    }

    /// The square with the given `0..64` LERF index. Debug-asserts the bound.
    #[inline]
    #[must_use]
    pub const fn from_index(square: u8) -> Self {
        debug_assert!(square < 64, "invalid square value >= 64");
        // SAFETY: a valid square index is < 64 - never the 0xFF niche.
        Self(unsafe { NonMaxU8::new_unchecked(square) })
    }

    /// The rank (`0..8`, rank 1 = 0).
    #[inline]
    #[must_use]
    pub const fn rank(self) -> u8 {
        self.index() >> 3
    }

    /// The file (`0..8`, a-file = 0).
    #[inline]
    #[must_use]
    pub const fn file(self) -> u8 {
        self.index() & 7
    }

    /// Parse a square from two ASCII bytes like `b"e4"` (file letter then rank
    /// digit). Returns [`InvalidSquareError`] on any malformed input.
    pub const fn from_ascii(bytes: &[u8]) -> Result<Self, InvalidSquareError> {
        if bytes.len() != 2 {
            return Err(InvalidSquareError);
        }
        let c1 = bytes[0].to_ascii_lowercase();
        let c2 = bytes[1];
        if c1 < b'a' || c1 > b'h' || c2 < b'1' || c2 > b'8' {
            return Err(InvalidSquareError);
        }

        let rank = c2 - b'1';
        let file = c1 - b'a';

        Ok(Self::new(rank, file))
    }

    /// The raw `0..64` LERF index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0.get()
    }
}

/// Error returned when a byte/string pair does not name a valid square.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InvalidSquareError;

impl std::fmt::Display for InvalidSquareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid square")
    }
}

impl std::error::Error for InvalidSquareError {}

impl FromStr for Square {
    type Err = InvalidSquareError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if !s.is_ascii() {
            return Err(InvalidSquareError);
        }
        Self::from_ascii(s.as_bytes())
    }
}

impl std::fmt::Debug for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Square")
            .field("value", &self.index())
            .field("rank", &self.rank())
            .field("file", &self.file())
            .finish()
    }
}

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char(Self::FILES[self.file() as usize] as char)?;
        f.write_char(Self::RANKS[self.rank() as usize] as char)?;
        Ok(())
    }
}

// `Option<Square>` stays one byte due to `NonMaxU8`
const _: () = assert!(std::mem::size_of::<Option<Square>>() == 1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_creation_and_decomposition() {
        let a1 = Square::new(0, 0);
        assert_eq!(a1.index(), 0);
        assert_eq!(a1.rank(), 0);
        assert_eq!(a1.file(), 0);

        let h1 = Square::new(0, 7);
        assert_eq!(h1.index(), 7);
        assert_eq!(h1.rank(), 0);
        assert_eq!(h1.file(), 7);

        let a8 = Square::new(7, 0);
        assert_eq!(a8.index(), 56);
        assert_eq!(a8.rank(), 7);
        assert_eq!(a8.file(), 0);

        let h8 = Square::new(7, 7);
        assert_eq!(h8.index(), 63);
        assert_eq!(h8.rank(), 7);
        assert_eq!(h8.file(), 7);
    }

    #[test]
    fn test_square_from_u8() {
        for file in 0..8 {
            for rank in 0..8 {
                let idx = (rank * 8 + file) as u8;
                let square = Square::new(rank as u8, file as u8);
                assert_eq!(square.index(), idx);

                let square = Square::from_index(idx);
                assert_eq!(square.rank(), rank as u8);
                assert_eq!(square.file(), file as u8);
            }
        }
    }

    #[test]
    fn test_square_from_ascii() {
        for (file, c1) in Square::FILES.iter().enumerate() {
            for (rank, c2) in Square::RANKS.iter().enumerate() {
                let c1 = *c1 as char;
                let c2 = *c2 as char;
                let idx = (rank * 8 + file) as u8;
                let string = format!("{c1}{c2}");
                let square = Square::from_ascii(string.as_bytes());
                eprintln!("testing assertion conditions for {string}");
                assert!(square.is_ok());
                let square = square.unwrap();
                assert_eq!(square.file(), file as u8);
                assert_eq!(square.rank(), rank as u8);
                assert_eq!(square.index(), idx);
            }
        }
    }

    #[test]
    fn test_square_display() {
        for (file, c1) in Square::FILES.iter().enumerate() {
            for (rank, c2) in Square::RANKS.iter().enumerate() {
                let c1 = *c1 as char;
                let c2 = *c2 as char;
                let idx = (rank * 8 + file) as u8;
                let string = format!("{c1}{c2}");
                let square = Square::from_index(idx);
                assert_eq!(square.to_string(), string);
            }
        }
    }

    #[test]
    fn test_square_ascii_error_handling() {
        assert!(Square::from_ascii(b"a").is_err());
        assert!(Square::from_ascii(b"a12").is_err());
        assert!(Square::from_ascii(b"i1").is_err());
        assert!(Square::from_ascii(b"a9").is_err());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn square_out_of_bounds_should_panic() {
        let _ = Square::from_index(64);
    }
}
