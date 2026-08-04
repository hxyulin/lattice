use crate::{Bitboard, Board, Color, Move, MoveType, Square};

use super::MoveList;

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;
const RANK_3: u64 = 0x0000_0000_00ff_0000;
const RANK_6: u64 = 0x0000_ff00_0000_0000;

fn push_moves(list: &mut MoveList, targets: Bitboard, delta: i8, capture: bool) {
    for to in targets {
        let from = Square::new_unchecked((to.index() as i8 - delta) as u8);
        let promotion = if to.rank() == 0 || to.rank() == 7 {
            if capture {
                [
                    MoveType::KnightPromoCap,
                    MoveType::BishopPromoCap,
                    MoveType::RookPromoCap,
                    MoveType::QueenPromoCap,
                ]
            } else {
                [
                    MoveType::KnightPromo,
                    MoveType::BishopPromo,
                    MoveType::RookPromo,
                    MoveType::QueenPromo,
                ]
            }
        } else {
            let kind = if capture {
                MoveType::Capture
            } else {
                MoveType::Quiet
            };
            [kind; 4]
        };
        let count = if to.rank() == 0 || to.rank() == 7 {
            4
        } else {
            1
        };
        for kind in promotion.into_iter().take(count) {
            list.push(Move::new(from, to, kind));
        }
    }
}

pub(crate) fn generate(board: &Board, list: &mut MoveList, pawns: Bitboard, them: Bitboard) {
    let occ = board.occupied().bits();
    let ep = board
        .state()
        .en_passant()
        .map_or(0, |square| 1u64 << square.index());
    match board.state().side_to_move() {
        Color::White => {
            let single = (pawns.bits() << 8) & !occ;
            let double = ((single & RANK_3) << 8) & !occ;
            push_moves(list, Bitboard::new(single), 8, false);
            for to in Bitboard::new(double) {
                list.push(Move::new(
                    Square::new_unchecked(to.index() - 16),
                    to,
                    MoveType::DoublePawnPush,
                ));
            }
            let left = ((pawns.bits() & !FILE_A) << 7) & (them.bits() | ep);
            let right = ((pawns.bits() & !FILE_H) << 9) & (them.bits() | ep);
            push_captures(list, Bitboard::new(left), 7, ep);
            push_captures(list, Bitboard::new(right), 9, ep);
        }
        Color::Black => {
            let single = (pawns.bits() >> 8) & !occ;
            let double = ((single & RANK_6) >> 8) & !occ;
            push_moves(list, Bitboard::new(single), -8, false);
            for to in Bitboard::new(double) {
                list.push(Move::new(
                    Square::new_unchecked(to.index() + 16),
                    to,
                    MoveType::DoublePawnPush,
                ));
            }
            let left = ((pawns.bits() & !FILE_A) >> 9) & (them.bits() | ep);
            let right = ((pawns.bits() & !FILE_H) >> 7) & (them.bits() | ep);
            push_captures(list, Bitboard::new(left), -9, ep);
            push_captures(list, Bitboard::new(right), -7, ep);
        }
    }
}

fn push_captures(list: &mut MoveList, targets: Bitboard, delta: i8, ep: u64) {
    for to in targets {
        let from = Square::new_unchecked((to.index() as i8 - delta) as u8);
        if ep & (1u64 << to.index()) != 0 {
            list.push(Move::new(from, to, MoveType::EnPassant));
        } else {
            let promotion = to.rank() == 0 || to.rank() == 7;
            if promotion {
                for kind in [
                    MoveType::KnightPromoCap,
                    MoveType::BishopPromoCap,
                    MoveType::RookPromoCap,
                    MoveType::QueenPromoCap,
                ] {
                    list.push(Move::new(from, to, kind));
                }
            } else {
                list.push(Move::new(from, to, MoveType::Capture));
            }
        }
    }
}
