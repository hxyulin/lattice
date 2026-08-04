use crate::{Bitboard, Square};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_B: u64 = FILE_A << 1;
const FILE_G: u64 = FILE_A << 6;
const FILE_H: u64 = FILE_A << 7;

pub(crate) static KNIGHT: [Bitboard; 64] = knight_table();
pub(crate) static KING: [Bitboard; 64] = king_table();
pub(crate) static PAWN: [[Bitboard; 64]; 2] = pawn_table();

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
