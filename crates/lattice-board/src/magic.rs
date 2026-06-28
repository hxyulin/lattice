//! Magic bitboards: O(1) sliding-piece attacks via perfect-hashed occupancy
//! lookup.
//!
//! # Notes
//!
//! Only squares between a slider and the edge can block a ray (the `mask`). A
//! magic multiplier maps each mask occupancy to a dense index
//! `(occ & mask).wrapping_mul(magic) >> shift`, indexing a table precomputed
//! with a slow reference slider; a lookup is one multiply, one shift, one load.
//!
//! The magics are found, not hard-coded: a fixed-seed PRNG proposes sparse
//! candidates until one hashes a square's occupancies without a destructive
//! collision. Fixed seed plus fixed build order gives the same tables every run.
//! The search runs once inside a [`LazyLock`], paid up front on the `uci`
//! command via [`init_tables`] rather than mid-search.

use std::sync::LazyLock;

use crate::{Bitboard, Square};

/// `(file, rank)` steps for a rook (orthogonal) and bishop (diagonal).
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Fixed PRNG seed: pins the (deterministic) magic search to one outcome.
const SEED: u64 = 0xDEAD_BEEF_1234_5678;

/// Reference slider: attack set from `sq` over `occ`, stepping each direction to
/// the edge or first blocker (included - a capture target). The slow version
/// used to build the table.
fn slide(sq: u8, occ: u64, dirs: &[(i8, i8)]) -> u64 {
    let mut attacks = 0u64;
    let (sf, sr) = ((sq % 8) as i8, (sq / 8) as i8);
    for &(df, dr) in dirs {
        let (mut f, mut r) = (sf + df, sr + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            attacks |= 1u64 << (r * 8 + f);
            if occ & (1u64 << (r * 8 + f)) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    attacks
}

/// Relevant-occupancy mask for `sq`: ray squares a blocker could sit on,
/// excluding the edge square in each direction (a blocker there changes
/// nothing).
fn occupancy_mask(sq: u8, dirs: &[(i8, i8)]) -> u64 {
    let mut mask = 0u64;
    let (sf, sr) = ((sq % 8) as i8, (sq / 8) as i8);
    for &(df, dr) in dirs {
        let (mut f, mut r) = (sf + df, sr + dr);
        while (0..8).contains(&(f + df)) && (0..8).contains(&(r + dr)) {
            mask |= 1u64 << (r * 8 + f);
            f += df;
            r += dr;
        }
    }
    mask
}

/// xorshift64: a tiny deterministic PRNG for the magic search. `sparse` ANDs
/// three draws so candidates have few set bits - sparse magics hash cleaner and
/// are found faster.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

/// A solved magic for one square: mask, multiplier, shift
/// (`64 - mask.count_ones()`), and the attack table it indexes.
struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    attacks: Box<[Bitboard]>,
}

impl Magic {
    /// Solve the magic for `sq` over `dirs`, drawing candidates from `rng` until
    /// one hashes every mask occupancy without a destructive collision.
    fn build(sq: u8, dirs: &[(i8, i8)], rng: &mut Rng) -> Magic {
        let mask = occupancy_mask(sq, dirs);
        let shift = 64 - mask.count_ones();
        let size = 1usize << mask.count_ones();

        // Every occupancy subset of the mask (carry-rippler), with its reference
        // attack set. `slide` reads the full subset, so blockers stop the ray.
        let mut occs = Vec::with_capacity(size);
        let mut refs = Vec::with_capacity(size);
        let mut sub = 0u64;
        loop {
            occs.push(sub);
            refs.push(slide(sq, sub, dirs));
            sub = sub.wrapping_sub(mask) & mask;
            if sub == 0 {
                break;
            }
        }

        loop {
            let magic = rng.sparse();
            // Cheap reject: a good magic spreads the mask's high bits across the
            // top byte. Skips most hopeless candidates before the full fill.
            if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
                continue;
            }
            let mut table = vec![Bitboard::EMPTY; size];
            let mut filled = vec![false; size];
            let mut ok = true;
            for (i, &occ) in occs.iter().enumerate() {
                let idx = (occ.wrapping_mul(magic) >> shift) as usize;
                let attack = Bitboard::from_bits(refs[i]);
                if !filled[idx] {
                    filled[idx] = true;
                    table[idx] = attack;
                } else if table[idx] != attack {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Magic {
                    mask,
                    magic,
                    shift,
                    attacks: table.into_boxed_slice(),
                };
            }
        }
    }

    #[inline]
    fn attacks(&self, occ: u64) -> Bitboard {
        let idx = ((occ & self.mask).wrapping_mul(self.magic) >> self.shift) as usize;
        self.attacks[idx]
    }
}

/// The solved rook and bishop magics for all 64 squares.
struct SliderTables {
    rook: [Magic; 64],
    bishop: [Magic; 64],
}

impl SliderTables {
    fn build() -> Self {
        // One PRNG threaded through a fixed square order => deterministic tables.
        let mut rng = Rng(SEED);
        let rook = std::array::from_fn(|sq| Magic::build(sq as u8, &ROOK_DIRS, &mut rng));
        let bishop = std::array::from_fn(|sq| Magic::build(sq as u8, &BISHOP_DIRS, &mut rng));
        Self { rook, bishop }
    }
}

static SLIDERS: LazyLock<SliderTables> = LazyLock::new(SliderTables::build);

/// Rook attack set from `from` over occupancy `occ`.
#[inline]
pub(crate) fn rook_attacks(from: Square, occ: Bitboard) -> Bitboard {
    SLIDERS.rook[from.index() as usize].attacks(occ.bits())
}

/// Bishop attack set from `from` over occupancy `occ`.
#[inline]
pub(crate) fn bishop_attacks(from: Square, occ: Bitboard) -> Bitboard {
    SLIDERS.bishop[from.index() as usize].attacks(occ.bits())
}

/// Force the slider tables to build now; cheap if already built.
///
/// # Performance
///
/// The UCI layer calls this on the `uci` command so the one-time table build is
/// paid during engine init rather than on the first search.
pub fn init_tables() {
    LazyLock::force(&SLIDERS);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Magic lookup must equal the reference slider for every occupancy;
    // spot-checks a corner, center, and edge square. Perft is the exhaustive
    // guard.
    #[test]
    fn magic_matches_reference_slider() {
        let mut rng = Rng(1);
        for &sq in &[0u8, 27, 35, 7, 63, 9] {
            for _ in 0..2000 {
                let occ = rng.next() & rng.next(); // varied density
                let r = rook_attacks(Square::from_index(sq), Bitboard::from_bits(occ));
                assert_eq!(
                    r.bits(),
                    slide(sq, occ, &ROOK_DIRS),
                    "rook sq {sq} occ {occ:#x}"
                );
                let b = bishop_attacks(Square::from_index(sq), Bitboard::from_bits(occ));
                assert_eq!(
                    b.bits(),
                    slide(sq, occ, &BISHOP_DIRS),
                    "bishop sq {sq} occ {occ:#x}"
                );
            }
        }
    }
}
