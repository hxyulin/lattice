use crate::{Bitboard, Square};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_B: u64 = FILE_A << 1;
const FILE_G: u64 = FILE_A << 6;
const FILE_H: u64 = FILE_A << 7;

pub(crate) static KNIGHT: [Bitboard; 64] = knight_table();
pub(crate) static KING: [Bitboard; 64] = king_table();
pub(crate) static PAWN: [[Bitboard; 64]; 2] = pawn_table();
/// The squares strictly between two aligned squares, empty when they do not
/// share a rank, file or diagonal. Blocking a check means moving onto one of
/// `BETWEEN[king][checker]`.
///
/// 32 KB of static data, which buys a single load where the alternative is
/// two magic lookups and an intersection per checker.
pub(crate) static BETWEEN: [[Bitboard; 64]; 64] = between_table();

const fn between_table() -> [[Bitboard; 64]; 64] {
    let mut out = [[Bitboard::empty(); 64]; 64];
    let mut from = 0usize;
    while from < 64 {
        let mut to = 0usize;
        while to < 64 {
            out[from][to] = Bitboard::new(between_bits(from, to));
            to += 1;
        }
        from += 1;
    }
    out
}

/// Walks square by square from `from` towards `to` along whichever of the
/// eight directions lines them up, accumulating what it passes over.
const fn between_bits(from: usize, to: usize) -> u64 {
    if from == to {
        return 0;
    }
    let (ff, fr) = ((from % 8) as i32, (from / 8) as i32);
    let (tf, tr) = ((to % 8) as i32, (to / 8) as i32);
    let (df, dr) = (tf - ff, tr - fr);
    // Aligned means equal files, equal ranks, or equal absolute deltas.
    let step = if df == 0 {
        (0, if dr > 0 { 1 } else { -1 })
    } else if dr == 0 {
        (if df > 0 { 1 } else { -1 }, 0)
    } else if abs(df) == abs(dr) {
        (if df > 0 { 1 } else { -1 }, if dr > 0 { 1 } else { -1 })
    } else {
        return 0;
    };
    let mut bits = 0u64;
    let (mut file, mut rank) = (ff + step.0, fr + step.1);
    while file != tf || rank != tr {
        bits |= 1u64 << (rank * 8 + file) as usize;
        file += step.0;
        rank += step.1;
    }
    bits
}

const fn abs(value: i32) -> i32 {
    if value < 0 { -value } else { value }
}

const fn knight_table() -> [Bitboard; 64] {
    let mut out = [Bitboard::empty(); 64];
    let mut i = 0;
    while i < 64 {
        let b = 1u64 << i;
        out[i] = Bitboard::new(
            ((b & !(FILE_A | FILE_B)) << 6)
                | ((b & !FILE_A) << 15)
                | ((b & !FILE_H) << 17)
                | ((b & !(FILE_G | FILE_H)) << 10)
                | ((b & !(FILE_G | FILE_H)) >> 6)
                | ((b & !FILE_H) >> 15)
                | ((b & !FILE_A) >> 17)
                | ((b & !(FILE_A | FILE_B)) >> 10),
        );
        i += 1;
    }
    out
}

const fn king_table() -> [Bitboard; 64] {
    let mut out = [Bitboard::empty(); 64];
    let mut i = 0;
    while i < 64 {
        let b = 1u64 << i;
        out[i] = Bitboard::new(
            (b << 8)
                | (b >> 8)
                | ((b & !FILE_H) << 1)
                | ((b & !FILE_A) >> 1)
                | ((b & !FILE_H) << 9)
                | ((b & !FILE_A) << 7)
                | ((b & !FILE_H) >> 7)
                | ((b & !FILE_A) >> 9),
        );
        i += 1;
    }
    out
}

const fn pawn_table() -> [[Bitboard; 64]; 2] {
    let mut out = [[Bitboard::empty(); 64]; 2];
    let mut i = 0;
    while i < 64 {
        let b = 1u64 << i;
        out[0][i] = Bitboard::new(((b & !FILE_A) << 7) | ((b & !FILE_H) << 9));
        out[1][i] = Bitboard::new(((b & !FILE_A) >> 9) | ((b & !FILE_H) >> 7));
        i += 1;
    }
    out
}

const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn walk(sq: Square, blockers: Bitboard, dirs: &[(i8, i8); 4]) -> Bitboard {
    let mut attacks = Bitboard::empty();
    for &(df, dr) in dirs {
        let mut file = sq.file() as i8 + df;
        let mut rank = sq.rank() as i8 + dr;
        while (0..8).contains(&file) && (0..8).contains(&rank) {
            let target = Square::new(file as u8, rank as u8).unwrap();
            attacks.set(target);
            if blockers.contains(target) {
                break;
            }
            file += df;
            rank += dr;
        }
    }
    attacks
}

pub(crate) fn ray_rook(sq: Square, blockers: Bitboard) -> Bitboard {
    walk(sq, blockers, &ROOK_DIRS)
}

pub(crate) fn ray_bishop(sq: Square, blockers: Bitboard) -> Bitboard {
    walk(sq, blockers, &BISHOP_DIRS)
}

fn mask(sq: Square, dirs: &[(i8, i8); 4]) -> Bitboard {
    let mut result = Bitboard::empty();
    for &(df, dr) in dirs {
        let mut file = sq.file() as i8 + df;
        let mut rank = sq.rank() as i8 + dr;
        while (1..7).contains(&file) && (1..7).contains(&rank) {
            result.set(Square::new(file as u8, rank as u8).unwrap());
            file += df;
            rank += dr;
        }
    }
    result
}

pub(crate) fn rook_mask(sq: Square) -> Bitboard {
    let mut result = Bitboard::empty();
    for &(df, dr) in &ROOK_DIRS {
        let mut file = sq.file() as i8 + df;
        let mut rank = sq.rank() as i8 + dr;
        while (0..8).contains(&file) && (0..8).contains(&rank) {
            let next_file = file + df;
            let next_rank = rank + dr;
            if !(0..8).contains(&next_file) || !(0..8).contains(&next_rank) {
                break;
            }
            result.set(Square::new(file as u8, rank as u8).unwrap());
            file = next_file;
            rank = next_rank;
        }
    }
    result
}

pub(crate) fn bishop_mask(sq: Square) -> Bitboard {
    mask(sq, &BISHOP_DIRS)
}

#[cfg(test)]
mod tests {
    use super::BETWEEN;
    use crate::movegen::magic::{bishop_attacks, rook_attacks};
    use crate::{Bitboard, Square};

    // catches: a table that disagrees with the ray arithmetic the rest of
    // movegen uses. `pinned_pieces` derives the same set from two magic
    // lookups, so that is the oracle; if the two ever diverge, evasions and
    // pins would answer differently about the same ray.
    #[test]
    fn between_agrees_with_the_magic_derivation() {
        crate::movegen::init();
        for from in 0..64u8 {
            for to in 0..64u8 {
                let (a, b) = (Square::new_unchecked(from), Square::new_unchecked(to));
                let empty = Bitboard::empty();
                let (at_a, at_b) = (Bitboard::from_square(a), Bitboard::from_square(b));
                let expected = if from == to {
                    empty
                } else if rook_attacks(a, empty).contains(b) {
                    rook_attacks(a, at_b) & rook_attacks(b, at_a)
                } else if bishop_attacks(a, empty).contains(b) {
                    bishop_attacks(a, at_b) & bishop_attacks(b, at_a)
                } else {
                    empty
                };
                assert_eq!(
                    BETWEEN[from as usize][to as usize], expected,
                    "between {from} {to}"
                );
            }
        }
    }

    #[test]
    fn between_is_symmetric_and_excludes_the_endpoints() {
        for (from, row) in BETWEEN.iter().enumerate() {
            for (to, &squares) in row.iter().enumerate() {
                assert_eq!(squares, BETWEEN[to][from], "asymmetric {from} {to}");
                assert!(!squares.contains(Square::new_unchecked(from as u8)));
                assert!(!squares.contains(Square::new_unchecked(to as u8)));
            }
        }
        // Adjacent squares have nothing between them; a1-a8 has six.
        assert!(BETWEEN[0][8].is_empty());
        assert_eq!(BETWEEN[0][56].count(), 6);
        assert_eq!(BETWEEN[0][63].count(), 6);
        // Unaligned squares are empty: a1 and b3 are a knight's move apart.
        assert!(BETWEEN[0][17].is_empty());
    }
}
