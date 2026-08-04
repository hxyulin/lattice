use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use super::Square;

/// A set of squares stored in little-endian rank-file order.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Bitboard(u64);

impl Bitboard {
    /// Builds a bitboard from its raw bits.
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }
    /// Returns the raw bits.
    pub const fn bits(self) -> u64 {
        self.0
    }
    /// Returns an empty bitboard.
    pub const fn empty() -> Self {
        Self(0)
    }
    /// Returns a bitboard containing one square.
    pub fn from_square(square: Square) -> Self {
        Self(1 << square.index())
    }
    /// Returns whether the square is present.
    pub fn contains(self, square: Square) -> bool {
        self.0 & (1 << square.index()) != 0
    }
    /// Inserts a square.
    pub fn set(&mut self, square: Square) {
        self.0 |= 1 << square.index();
    }
    /// Removes a square.
    pub fn clear(&mut self, square: Square) {
        self.0 &= !(1 << square.index());
    }
    /// Returns whether no squares are present.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// Returns the number of squares present.
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
    /// Returns the least significant square.
    pub fn lsb(self) -> Option<Square> {
        (!self.is_empty()).then(|| Square::new_unchecked(self.0.trailing_zeros() as u8))
    }
}

impl Iterator for Bitboard {
    type Item = Square;
    fn next(&mut self) -> Option<Self::Item> {
        let square = self.lsb()?;
        self.0 &= self.0 - 1;
        Some(square)
    }
}

macro_rules! bit_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Bitboard { type Output = Self; fn $method(self, rhs: Self) -> Self { Self(self.0 $op rhs.0) } }
    };
}
macro_rules! bit_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Bitboard { fn $method(&mut self, rhs: Self) { self.0 $op rhs.0; } }
    };
}
bit_op!(BitAnd, bitand, &);
bit_op!(BitOr, bitor, |);
bit_op!(BitXor, bitxor, ^);
bit_assign!(BitAndAssign, bitand_assign, &=);
bit_assign!(BitOrAssign, bitor_assign, |=);
bit_assign!(BitXorAssign, bitxor_assign, ^=);
impl Not for Bitboard {
    type Output = Self;
    fn not(self) -> Self {
        Self(!self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operations_cover_all_squares() {
        let mut board = Bitboard::empty();
        assert!(board.is_empty());
        assert_eq!(board.lsb(), None);
        for i in 0..64 {
            let s = Square::new_unchecked(i);
            board.set(s);
            assert!(board.contains(s));
        }
        assert_eq!(board.count(), 64);
        assert_eq!(
            board.count() as usize,
            board.into_iter().collect::<Vec<_>>().len()
        );
        assert_eq!(
            board.into_iter().map(Square::index).collect::<Vec<_>>(),
            (0..64).collect::<Vec<_>>()
        );
        for i in 0..64 {
            let s = Square::new_unchecked(i);
            board.clear(s);
            assert!(!board.contains(s));
        }
        assert!(board.is_empty());
        assert_eq!(board.lsb(), None);
    }
}
