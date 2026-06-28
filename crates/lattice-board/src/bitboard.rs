use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::Square;

/// 64-bit bitboard over the board's 64 squares, indexed like [`Square`] (a1=0, h8=63, LERF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(u64);

impl Bitboard {
    /// A bitboard with no squares set.
    pub const EMPTY: Bitboard = Bitboard(0);

    /// A bitboard with the leftmost file (a-file, file 0) set.
    pub const FILE_A: Bitboard = Self::file_mask(0);

    /// A bitboard with the rightmost file (h-file, file 7) set.
    pub const FILE_H: Bitboard = Self::file_mask(7);

    /// Bitboard of every square on a certain `file` (`0..8`, files a-h).
    #[inline]
    #[must_use]
    pub const fn file_mask(file: u8) -> Bitboard {
        debug_assert!(file < 8, "file must be in 0..8");
        Bitboard(0x0101_0101_0101_0101 << file)
    }

    /// Bitboard of every square on a certain `rank` (`0..8`, ranks 1-8).
    #[inline]
    #[must_use]
    pub const fn rank_mask(rank: u8) -> Bitboard {
        debug_assert!(rank < 8, "rank must be in 0..8");
        Bitboard(0xff << (rank * 8))
    }

    /// A bitboard with only a single bit set at square `sq`.
    #[inline]
    #[must_use]
    pub const fn from_square(sq: Square) -> Bitboard {
        Bitboard(1u64 << sq.index())
    }

    /// Wrap a raw 64-bit value.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u64) -> Bitboard {
        Bitboard(bits)
    }

    /// The raw 64-bit value.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Return true if `sq` is set in the bitboard.
    #[inline]
    #[must_use]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.index()) != 0
    }

    /// Sets the bit corresponding to `sq` in the bitboard.
    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.index();
    }

    /// Clears the bit corresponding to `sq` in the bitboard.
    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.index());
    }

    /// Returns true if the bitboard has no squares set.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the number of squares set in the bitboard.
    #[inline]
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns an iterator over the set squares of the bitboard, yielded a1-first.
    #[inline]
    #[must_use]
    pub const fn iter(self) -> Squares {
        Squares(self.0)
    }
}

impl BitAnd for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl Not for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

/// Iterator over the set squares of a [`Bitboard`], yielded a1-first.
#[derive(Debug, Clone)]
pub struct Squares(u64);

impl Iterator for Squares {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let index = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1; // clear the lowest set bit
        Some(Square::from_index(index))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Squares {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = Squares;
    #[inline]
    fn into_iter(self) -> Squares {
        Squares(self.0)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    fn sq(s: &str) -> Square {
        Square::from_str(s).unwrap()
    }

    #[test]
    fn set_clear_contains() {
        let mut bb = Bitboard::EMPTY;
        assert!(bb.is_empty());
        bb.set(sq("e4"));
        assert!(bb.contains(sq("e4")));
        assert!(!bb.contains(sq("e5")));
        assert_eq!(bb.count(), 1);
        bb.clear(sq("e4"));
        assert!(bb.is_empty());
    }

    #[test]
    fn from_square_matches_set() {
        for i in 0..64u8 {
            let s = Square::from_index(i);
            let mut bb = Bitboard::EMPTY;
            bb.set(s);
            assert_eq!(Bitboard::from_square(s), bb);
        }
    }

    #[test]
    fn bit_ops() {
        let a = Bitboard::from_square(sq("a1")) | Bitboard::from_square(sq("b1"));
        let b = Bitboard::from_square(sq("b1")) | Bitboard::from_square(sq("c1"));
        assert_eq!((a & b).count(), 1); // b1
        assert_eq!((a | b).count(), 3); // a1 b1 c1
        assert_eq!((a ^ b).count(), 2); // a1 c1
        assert_eq!((!Bitboard::EMPTY).count(), 64);
    }

    #[test]
    fn iterates_set_squares_in_order() {
        let squares = [sq("a1"), sq("e4"), sq("h8")];
        let mut bb = Bitboard::EMPTY;
        for &s in &squares {
            bb.set(s);
        }
        let collected: Vec<Square> = bb.into_iter().collect();
        assert_eq!(collected, squares);
        assert_eq!(bb.iter().len(), 3);
    }

    #[test]
    fn file_and_rank_masks() {
        assert_eq!(Bitboard::file_mask(0), Bitboard::FILE_A);
        assert_eq!(Bitboard::file_mask(7), Bitboard::FILE_H);
        assert_eq!(Bitboard::FILE_A.count(), 8);
        assert_eq!(Bitboard::rank_mask(0).count(), 8);

        let a1 = Square::from_str("a1").unwrap();
        assert!(Bitboard::FILE_A.contains(a1));
        assert!(Bitboard::rank_mask(0).contains(a1));
        assert!(!Bitboard::FILE_H.contains(a1));
        assert!(!Bitboard::rank_mask(7).contains(a1));

        // a file and a rank intersect in exactly one square
        let e4 = Square::from_str("e4").unwrap();
        let inter = Bitboard::file_mask(e4.file()) & Bitboard::rank_mask(e4.rank());
        assert_eq!(inter.count(), 1);
        assert!(inter.contains(e4));
    }
}
