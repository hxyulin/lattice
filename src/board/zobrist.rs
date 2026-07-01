//! Zobrist hashing: a 64-bit fingerprint of a position, maintained incrementally.
//!
//! # Notes
//!
//! Each `(piece, square)`, castling-rights combination, en-passant file, and
//! the side-to-move gets a fixed key; a position's hash is the XOR of every key
//! whose feature is present. XOR being its own inverse means a move only toggles
//! the keys it changes (see [`Board`](crate::Board)). The table is generated at
//! compile time by a [SplitMix64] PRNG seeded with a fixed constant, so it is
//! byte-for-byte identical on every build (a test pins the startpos hash).
//!
//! [SplitMix64]: https://prng.di.unimi.it/splitmix64.c

use std::fmt;

/// A 64-bit Zobrist hash of a board position.
///
/// # Notes
///
/// The key for a transposition table, repetition detection, or an opening book.
/// Move counters are *not* hashed, so two positions differing only in clock
/// values share a hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ZobristHash(pub(crate) u64);

impl ZobristHash {
    /// The raw 64-bit value, e.g. to index a transposition table.
    #[inline]
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for ZobristHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZobristHash({:#018x})", self.0)
    }
}

impl fmt::Display for ZobristHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// The full key table, built once at compile time into [`ZOBRIST`].
pub(crate) struct Zobrist {
    /// `[piece index 0..12][square 0..64]` - see [`crate::Piece::as_u8`].
    piece_keys: [[u64; 64]; 12],
    /// Indexed by the castling-rights bitset (0..16).
    castling_keys: [u64; 16],
    /// Indexed by the en-passant target file (0..8).
    ep_keys: [u64; 8],
    /// XORed in when it is Black's turn (White contributes nothing).
    side_key: u64,
}

impl Zobrist {
    /// Key for `piece` (its [`crate::Piece::as_u8`] index) on square `sq_idx`.
    #[inline]
    pub(crate) const fn piece(&self, piece_idx: u8, sq_idx: u8) -> u64 {
        self.piece_keys[piece_idx as usize][sq_idx as usize]
    }

    /// Key for the castling-rights bitset `bits` (0..16).
    #[inline]
    pub(crate) const fn castling(&self, bits: u8) -> u64 {
        self.castling_keys[bits as usize]
    }

    /// Key for an en-passant target on file `file` (0..8).
    #[inline]
    pub(crate) const fn en_passant(&self, file: u8) -> u64 {
        self.ep_keys[file as usize]
    }

    /// Key XORed in when Black is to move.
    #[inline]
    pub(crate) const fn side(&self) -> u64 {
        self.side_key
    }
}

/// Fixed seed for the key PRNG; changing it reshuffles every key (and breaks
/// the pinned startpos-hash test).
const SEED: u64 = 0x1A2B_3C4D_5E6F_7081;

/// One step of SplitMix64, a `const`-evaluable PRNG so the table is built at
/// compile time with no runtime init and no `rand` dependency.
const fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fill the whole key table from [`SEED`]; the fixed draw order makes it
/// deterministic.
const fn build() -> Zobrist {
    let mut state = SEED;

    let mut piece_keys = [[0u64; 64]; 12];
    let mut p = 0;
    while p < 12 {
        let mut s = 0;
        while s < 64 {
            piece_keys[p][s] = next(&mut state);
            s += 1;
        }
        p += 1;
    }

    let mut castling_keys = [0u64; 16];
    let mut c = 0;
    while c < 16 {
        castling_keys[c] = next(&mut state);
        c += 1;
    }

    let mut ep_keys = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        ep_keys[f] = next(&mut state);
        f += 1;
    }

    let side_key = next(&mut state);

    Zobrist {
        piece_keys,
        castling_keys,
        ep_keys,
        side_key,
    }
}

/// The process-wide key table, baked into the binary at compile time.
pub(crate) static ZOBRIST: Zobrist = build();
