use std::sync::LazyLock;

use crate::{Bitboard, Square};

use super::tables::{bishop_mask, ray_bishop, ray_rook, rook_mask};

#[derive(Clone, Copy, Default)]
struct Magic {
    mask: Bitboard,
    magic: u64,
    offset: u32,
    shift: u8,
}

struct MagicTables {
    rook: [Magic; 64],
    bishop: [Magic; 64],
    pool: Vec<Bitboard>,
}

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
}

// ponytail: runtime search takes about 0.2s; build.rs is the fallback if that becomes costly.
static TABLES: LazyLock<MagicTables> = LazyLock::new(build);

fn try_fill(
    entries: &[(u64, Bitboard)],
    candidate: u64,
    shift: u8,
    slots: &mut [(Bitboard, u32)],
    epoch: u32,
) -> bool {
    for &(sub, attack) in entries {
        let index = (sub.wrapping_mul(candidate) >> shift) as usize;
        let (old, stamp) = &mut slots[index];
        if *stamp == epoch {
            if *old != attack {
                return false;
            }
        } else {
            *old = attack;
            *stamp = epoch;
        }
    }
    true
}

fn find_magic(
    sq: Square,
    mask: Bitboard,
    ray: fn(Square, Bitboard) -> Bitboard,
    rng: &mut Rng,
) -> (Magic, Vec<Bitboard>) {
    let shift = 64 - mask.count() as u8;
    let mut entries = Vec::with_capacity(1usize << mask.count());
    let mut sub = 0u64;
    loop {
        entries.push((sub, ray(sq, Bitboard::new(sub))));
        sub = sub.wrapping_sub(mask.bits()) & mask.bits();
        if sub == 0 {
            break;
        }
    }
    let mut slots = vec![(Bitboard::empty(), 0); 1usize << mask.count()];
    let mut epoch = 0u32;
    loop {
        epoch = epoch.checked_add(1).expect("magic search epoch overflow");
        let candidate = rng.next() & rng.next() & rng.next();
        if (mask.bits().wrapping_mul(candidate) >> 56).count_ones() < 6 {
            continue;
        }
        if try_fill(&entries, candidate, shift, &mut slots, epoch) {
            let mut table = vec![Bitboard::empty(); slots.len()];
            for &(sub, attack) in &entries {
                let index = (sub.wrapping_mul(candidate) >> shift) as usize;
                table[index] = attack;
            }
            return (
                Magic {
                    mask,
                    magic: candidate,
                    offset: 0,
                    shift,
                },
                table,
            );
        }
    }
}

fn build() -> MagicTables {
    let mut rook = [Magic::default(); 64];
    let mut bishop = [Magic::default(); 64];
    let mut pool = Vec::new();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (i, slot) in rook.iter_mut().enumerate() {
        let sq = Square::new_unchecked(i as u8);
        let (mut magic, table) = find_magic(sq, rook_mask(sq), ray_rook, &mut rng);
        magic.offset = pool.len() as u32;
        pool.extend(table);
        *slot = magic;
    }
    for (i, slot) in bishop.iter_mut().enumerate() {
        let sq = Square::new_unchecked(i as u8);
        let (mut magic, table) = find_magic(sq, bishop_mask(sq), ray_bishop, &mut rng);
        magic.offset = pool.len() as u32;
        pool.extend(table);
        *slot = magic;
    }
    MagicTables { rook, bishop, pool }
}

fn lookup(magic: &Magic, occ: Bitboard) -> Bitboard {
    let index = ((occ & magic.mask).bits().wrapping_mul(magic.magic) >> magic.shift) as usize;
    TABLES.pool[magic.offset as usize + index]
}

pub(crate) fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    lookup(&TABLES.rook[sq.index() as usize], occ)
}

pub(crate) fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    lookup(&TABLES.bishop[sq.index() as usize], occ)
}

pub(crate) fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_matches_ray_walker_exhaustively() {
        for i in 0..64 {
            let sq = Square::new_unchecked(i);
            for (mask, magic, ray) in [
                (
                    rook_mask(sq),
                    rook_attacks as fn(_, _) -> _,
                    ray_rook as fn(_, _) -> _,
                ),
                (bishop_mask(sq), bishop_attacks, ray_bishop),
            ] {
                let mut sub = 0u64;
                loop {
                    let blockers = Bitboard::new(sub);
                    assert_eq!(magic(sq, blockers), ray(sq, blockers));
                    sub = sub.wrapping_sub(mask.bits()) & mask.bits();
                    if sub == 0 {
                        break;
                    }
                }
            }
        }
    }
}
