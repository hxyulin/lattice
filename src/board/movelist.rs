//! A fixed-capacity, stack-allocated [`MoveList`] - the allocation-free
//! replacement for `Vec<Move>` on the movegen and search hot paths.

use std::mem::MaybeUninit;
use std::ops::Deref;

use crate::Move;

/// A stack list of moves with fixed capacity and explicit length.
///
/// # Performance
///
/// Sized to a 512-byte tile via `#[repr(C, align(128))]`, which pins `len`
/// first (offset 0). `len` is touched on every `push`, so co-locating it with
/// the move array keeps it hot in cache for the early moves, which are visited
/// more often than the last few.
#[repr(C, align(128))]
pub struct MoveList {
    len: usize,
    moves: [MaybeUninit<Move>; Self::CAPACITY],
}

impl MoveList {
    /// Maximum number of moves the list can hold: the 218-move legal maximum of
    /// any position.
    pub const CAPACITY: usize = 218;

    /// An empty list with no initialization cost.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            len: 0,
            moves: [MaybeUninit::uninit(); Self::CAPACITY],
        }
    }

    /// Append a move. Debug-asserts the 218-move capacity is not exceeded, which
    /// movegen can never reach.
    #[inline]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < Self::CAPACITY, "MoveList capacity exceeded");
        self.moves[self.len].write(mv);
        self.len += 1;
    }

    /// Number of moves stored.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Is the list empty?
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drop all moves without touching the backing array.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// The stored moves as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: `push` initializes `moves[0..len]` in order and `len` never
        // exceeds that prefix. `MaybeUninit<Move>` shares `Move`'s layout, and
        // `Move` is `Copy` (no `Drop`), so reinterpreting the prefix is sound.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr().cast::<Move>(), self.len) }
    }
}

impl Default for MoveList {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for MoveList {
    type Target = [Move];
    #[inline]
    fn deref(&self) -> &[Move] {
        self.as_slice()
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// Owned iterator over a [`MoveList`], yielding [`Move`]s by value.
pub struct IntoIter {
    list: MoveList,
    pos: usize,
}

impl Iterator for IntoIter {
    type Item = Move;
    #[inline]
    fn next(&mut self) -> Option<Move> {
        if self.pos < self.list.len {
            // SAFETY: `pos < len`, and `moves[0..len]` were all initialized by
            // `push`; `Move` is `Copy`, so reading the slot doesn't move it out.
            let mv = unsafe { self.list.moves[self.pos].assume_init() };
            self.pos += 1;
            Some(mv)
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.list.len - self.pos;
        (n, Some(n))
    }
}

impl ExactSizeIterator for IntoIter {}

impl IntoIterator for MoveList {
    type Item = Move;
    type IntoIter = IntoIter;
    #[inline]
    fn into_iter(self) -> IntoIter {
        IntoIter { list: self, pos: 0 }
    }
}

// Compile-time invariants: capacity covers the legal maximum, the struct is an
// exact 512-byte cache-line tile (no padding), and the layout we documented
// actually holds (`len` first, 128-aligned).
const _: () = assert!(MoveList::CAPACITY >= 218);
const _: () = assert!(std::mem::size_of::<MoveList>() == 512);
const _: () = assert!(std::mem::align_of::<MoveList>() == 128);
const _: () = assert!(std::mem::offset_of!(MoveList, len) == 0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MoveFlag, Square};

    fn m(i: u8) -> Move {
        Move::new(
            Square::from_index(i),
            Square::from_index(i + 1),
            MoveFlag::Quiet,
        )
    }

    #[test]
    fn push_len_and_slice() {
        let mut l = MoveList::new();
        assert!(l.is_empty());
        l.push(m(0));
        l.push(m(2));
        assert_eq!(l.len(), 2);
        assert_eq!(l[0], m(0)); // Deref indexing
        assert_eq!(l.as_slice(), &[m(0), m(2)]);
        let via_ref: Vec<Move> = (&l).into_iter().copied().collect();
        assert_eq!(via_ref, vec![m(0), m(2)]);
        l.clear();
        assert!(l.is_empty());
    }

    #[test]
    fn holds_the_legal_maximum() {
        let mut l = MoveList::new();
        for _ in 0..218 {
            l.push(m(0));
        }
        assert_eq!(l.len(), 218);
    }
}
