//! Chess attack and move generation.

mod magic;
mod pawn;
/// Perft correctness and diagnostics.
pub mod perft;
mod tables;

use core::ops::Index;

use crate::{Bitboard, Board, CastlingRights, Color, Move, MoveType, PieceType, Square};
use magic::{bishop_attacks, queen_attacks, rook_attacks};
use tables::{KING, KNIGHT, PAWN};

/// A fixed-capacity list of chess moves.
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl MoveList {
    /// Returns an empty move list.
    pub fn new() -> Self {
        let dummy = Move::new(
            Square::new_unchecked(0),
            Square::new_unchecked(1),
            MoveType::Quiet,
        );
        Self {
            moves: [dummy; 256],
            len: 0,
        }
    }
    /// Appends a move.
    ///
    /// Capacity is 256 against a measured pseudo-legal maximum of 218.
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < 256, "move list overflow");
        self.moves[self.len] = mv;
        self.len += 1;
    }
    /// Returns the number of moves.
    pub const fn len(&self) -> usize {
        self.len
    }
    /// Returns whether the list is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Iterates over the moves.
    pub fn iter(&self) -> core::slice::Iter<'_, Move> {
        self.moves[..self.len].iter()
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<usize> for MoveList {
    type Output = Move;
    fn index(&self, index: usize) -> &Self::Output {
        &self.moves[..self.len][index]
    }
}

fn add_targets(
    list: &mut MoveList,
    from: Square,
    attacks: Bitboard,
    them: Bitboard,
    occ: Bitboard,
) {
    for to in attacks & them {
        list.push(Move::new(from, to, MoveType::Capture));
    }
    for to in attacks & !occ {
        list.push(Move::new(from, to, MoveType::Quiet));
    }
}

/// Generates pseudo-legal moves for the side to move.
pub fn generate_pseudo(board: &Board, list: &mut MoveList) {
    let us_color = board.state().side_to_move();
    let them_color = us_color.flip();
    let us = board.color(us_color);
    let them = board.color(them_color);
    let occ = board.occupied();
    pawn::generate(board, list, board.pieces(PieceType::Pawn) & us, them);
    for from in board.pieces(PieceType::Knight) & us {
        add_targets(list, from, KNIGHT[from.index() as usize], them, occ);
    }
    for from in board.pieces(PieceType::Bishop) & us {
        add_targets(list, from, bishop_attacks(from, occ), them, occ);
    }
    for from in board.pieces(PieceType::Rook) & us {
        add_targets(list, from, rook_attacks(from, occ), them, occ);
    }
    for from in board.pieces(PieceType::Queen) & us {
        add_targets(list, from, queen_attacks(from, occ), them, occ);
    }
    for from in board.pieces(PieceType::King) & us {
        add_targets(list, from, KING[from.index() as usize], them, occ);
    }
    generate_castling(board, list, us_color);
}

fn generate_castling(board: &Board, list: &mut MoveList, us: Color) {
    let (origin, king_right, queen_right, king_transit, king_to, queen_transit, queen_to, rook_gap) =
        match us {
            Color::White => (
                4,
                CastlingRights::WHITE_KING,
                CastlingRights::WHITE_QUEEN,
                5,
                6,
                3,
                2,
                1,
            ),
            Color::Black => (
                60,
                CastlingRights::BLACK_KING,
                CastlingRights::BLACK_QUEEN,
                61,
                62,
                59,
                58,
                57,
            ),
        };
    let sq = Square::new_unchecked;
    let attacked = |i| is_attacked(board, sq(i), us.flip());
    let rights = board.state().castling();
    if rights.contains(king_right)
        && !board.occupied().contains(sq(king_transit))
        && !board.occupied().contains(sq(king_to))
        && !attacked(origin)
        && !attacked(king_transit)
        && !attacked(king_to)
    {
        list.push(Move::new(sq(origin), sq(king_to), MoveType::KingCastle));
    }
    if rights.contains(queen_right)
        && !board.occupied().contains(sq(queen_transit))
        && !board.occupied().contains(sq(queen_to))
        && !board.occupied().contains(sq(rook_gap))
        && !attacked(origin)
        && !attacked(queen_transit)
        && !attacked(queen_to)
    {
        list.push(Move::new(sq(origin), sq(queen_to), MoveType::QueenCastle));
    }
}

/// Returns whether a square is attacked by a given side.
pub fn is_attacked(board: &Board, sq: Square, by: Color) -> bool {
    let their = board.color(by);
    let piece = |kind| board.pieces(kind) & their;
    !(KNIGHT[sq.index() as usize] & piece(PieceType::Knight)).is_empty()
        || !(PAWN[by.flip() as usize][sq.index() as usize] & piece(PieceType::Pawn)).is_empty()
        || !(KING[sq.index() as usize] & piece(PieceType::King)).is_empty()
        || !(rook_attacks(sq, board.occupied())
            & (piece(PieceType::Rook) | piece(PieceType::Queen)))
        .is_empty()
        || !(bishop_attacks(sq, board.occupied())
            & (piece(PieceType::Bishop) | piece(PieceType::Queen)))
        .is_empty()
}

/// Generates legal moves for the side to move.
pub fn generate_legal(board: &mut Board, list: &mut MoveList) {
    let mut pseudo = MoveList::new();
    generate_pseudo(board, &mut pseudo);
    let us = board.state().side_to_move();
    for &mv in pseudo.iter() {
        board.make(mv);
        if !is_attacked(board, board.king_square(us), us.flip()) {
            list.push(mv);
        }
        board.unmake(mv);
    }
}

#[cfg(test)]
mod tests {
    use super::perft::perft;
    use crate::Board;

    #[test]
    fn perft_reference_positions() {
        // Manual deep counts: 4865609, 4085603, 674624, 422333, 422333, 2103487, 3894594.
        let cases = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                4,
                197_281,
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                3,
                97_862,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2_812),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                3,
                9_467,
            ),
            (
                "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
                3,
                9_467,
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
                3,
                62_379,
            ),
            (
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
                3,
                89_890,
            ),
        ];
        for (fen, depth, expected) in cases {
            let mut board: Board = fen.parse().unwrap();
            assert_eq!(perft(&mut board, depth), expected, "{fen}");
        }
    }
}
