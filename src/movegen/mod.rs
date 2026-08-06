//! Chess attack and move generation.

mod magic;
mod pawn;
/// Perft correctness and diagnostics.
pub mod perft;
mod tables;

use core::ops::Index;

use crate::{Bitboard, Board, CastlingRights, Color, Move, MoveType, PieceType, Square};
pub use magic::init;
use magic::{bishop_attacks, queen_attacks, rook_attacks};
use tables::{KING, KNIGHT, PAWN};

/// A fixed-capacity list of chess moves.
pub struct MoveList {
    moves: [Move; Self::CAPACITY],
    len: usize,
}

impl MoveList {
    /// Slots available, against a measured pseudo-legal maximum of 218.
    const CAPACITY: usize = 256;

    /// Returns an empty move list.
    pub fn new() -> Self {
        let dummy = Move::new(
            Square::new_unchecked(0),
            Square::new_unchecked(1),
            MoveType::Quiet,
        );
        Self {
            moves: [dummy; Self::CAPACITY],
            len: 0,
        }
    }
    /// Appends a move.
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < Self::CAPACITY, "move list overflow");
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
    /// Returns the moves as a mutable slice, for reordering in place.
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        &mut self.moves[..self.len]
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

/// Calls `f` with every non-pawn piece of `us` and the squares it attacks.
fn for_each_attack(
    board: &Board,
    us: Bitboard,
    occ: Bitboard,
    mut f: impl FnMut(Square, Bitboard),
) {
    for from in board.pieces(PieceType::Knight) & us {
        f(from, KNIGHT[from.index() as usize]);
    }
    for from in board.pieces(PieceType::Bishop) & us {
        f(from, bishop_attacks(from, occ));
    }
    for from in board.pieces(PieceType::Rook) & us {
        f(from, rook_attacks(from, occ));
    }
    for from in board.pieces(PieceType::Queen) & us {
        f(from, queen_attacks(from, occ));
    }
    for from in board.pieces(PieceType::King) & us {
        f(from, KING[from.index() as usize]);
    }
}

/// Generates pseudo-legal moves for the side to move.
pub fn generate_pseudo(board: &Board, list: &mut MoveList) {
    generate(board, list, true);
}

/// Generates pseudo-legal captures and promotion captures for the side to move.
///
/// This is the quiescence-search move set. Quiet promotions are deliberately
/// excluded.
// ponytail: captures only. Add quiet promotions if they SPRT positive.
pub fn generate_captures(board: &Board, list: &mut MoveList) {
    generate(board, list, false);
}

/// The body both generators share: captures always, quiets and castling only
/// when `quiets` is set. Kept as one function so the capture set cannot drift
/// out of agreement with the full set.
fn generate(board: &Board, list: &mut MoveList, quiets: bool) {
    let us_color = board.state().side_to_move();
    let us = board.color(us_color);
    let them = board.color(us_color.flip());
    let occ = board.occupied();
    let pawns = board.pieces(PieceType::Pawn) & us;
    if quiets {
        pawn::generate(board, list, pawns, them);
    } else {
        pawn::generate_captures(board, list, pawns, them);
    }
    for_each_attack(board, us, occ, |from, attacks| {
        for to in attacks & them {
            list.push(Move::new(from, to, MoveType::Capture));
        }
        if quiets {
            for to in attacks & !occ {
                list.push(Move::new(from, to, MoveType::Quiet));
            }
        }
    });
    if quiets {
        generate_castling(board, list, us_color);
    }
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
    use super::{MoveList, generate_captures, generate_legal, is_attacked};
    use crate::{Board, Color, Move, MoveType, Square};

    fn sorted(list: &MoveList) -> Vec<Move> {
        let mut moves: Vec<_> = list.iter().copied().collect();
        moves.sort_unstable_by_key(|mv| {
            (mv.move_type() as u16) << 12 | (mv.from().index() as u16) << 6 | mv.to().index() as u16
        });
        moves
    }

    fn legal(fen: &str) -> Vec<Move> {
        super::init();
        let mut board: Board = fen.parse().unwrap();
        let mut list = MoveList::new();
        generate_legal(&mut board, &mut list);
        sorted(&list)
    }

    fn has(fen: &str, from: &str, to: &str, kind: MoveType) -> bool {
        let sq = |s: &str| {
            let b = s.as_bytes();
            Square::new(b[0] - b'a', b[1] - b'1').unwrap()
        };
        legal(fen)
            .iter()
            .any(|mv| mv.from() == sq(from) && mv.to() == sq(to) && mv.move_type() == kind)
    }

    /// catches: pawn direction and blocker defects - a single push that ignores
    /// occupancy, a double push that ignores a blocker on either the transit or
    /// the destination square, and a double push offered from the wrong rank.
    #[test]
    fn pawn_pushes_respect_blockers() {
        assert!(has(
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            "e2",
            "e4",
            MoveType::DoublePawnPush
        ));
        assert!(!has(
            "4k3/8/8/8/4n3/8/4P3/4K3 w - - 0 1",
            "e2",
            "e4",
            MoveType::DoublePawnPush
        ));
        assert!(!has(
            "4k3/8/8/8/8/4n3/4P3/4K3 w - - 0 1",
            "e2",
            "e3",
            MoveType::Quiet
        ));
        assert!(!has(
            "4k3/8/8/8/8/4n3/4P3/4K3 w - - 0 1",
            "e2",
            "e4",
            MoveType::DoublePawnPush
        ));
        assert!(!has(
            "4k3/8/8/8/8/4P3/8/4K3 w - - 0 1",
            "e3",
            "e5",
            MoveType::DoublePawnPush
        ));
    }

    /// catches: en passant dropped from the capture target set, or emitted as an
    /// ordinary capture rather than `MoveType::EnPassant`.
    #[test]
    fn en_passant_is_generated_with_its_own_move_type() {
        assert!(has(
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "e5",
            "d6",
            MoveType::EnPassant
        ));
        assert!(!has(
            "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1",
            "e5",
            "d6",
            MoveType::EnPassant
        ));
        assert!(has(
            "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
            "d4",
            "e3",
            MoveType::EnPassant
        ));
    }

    /// catches: a pawn capture generator missing either file wrap guard. An
    /// unguarded a-file pawn shifts left onto the h-file of the same rank, and
    /// an unguarded h-file pawn shifts right onto the a-file of the next.
    #[test]
    fn pawn_captures_do_not_wrap_around_the_board() {
        assert!(
            !legal("4k3/8/8/8/8/8/P6n/4K3 w - - 0 1")
                .iter()
                .any(|mv| mv.is_capture())
        );
        assert!(
            !legal("4k3/8/8/8/8/n7/7P/4K3 w - - 0 1")
                .iter()
                .any(|mv| mv.is_capture())
        );
        assert!(
            !legal("4k3/8/8/8/8/8/n6p/4K3 b - - 0 1")
                .iter()
                .any(|mv| mv.is_capture())
        );
        assert!(has(
            "4k3/8/8/8/8/1n6/P7/4K3 w - - 0 1",
            "a2",
            "b3",
            MoveType::Capture
        ));
    }

    /// catches: promotion rank detection that only handles rank 7 (white) and
    /// not rank 0, and under-promotion omitted from either the quiet or the
    /// capture path.
    #[test]
    fn promotions_generate_all_four_pieces_on_both_ranks() {
        let kinds = |fen: &str, from: &str, to: &str| {
            let sq = |s: &str| {
                let b = s.as_bytes();
                Square::new(b[0] - b'a', b[1] - b'1').unwrap()
            };
            let mut k: Vec<_> = legal(fen)
                .iter()
                .filter(|mv| mv.from() == sq(from) && mv.to() == sq(to))
                .map(|mv| mv.move_type())
                .collect();
            k.sort_unstable_by_key(|t| *t as u8);
            k
        };
        assert_eq!(
            kinds("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7", "a8"),
            [
                MoveType::KnightPromo,
                MoveType::BishopPromo,
                MoveType::RookPromo,
                MoveType::QueenPromo
            ]
        );
        assert_eq!(
            kinds("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2", "a1"),
            [
                MoveType::KnightPromo,
                MoveType::BishopPromo,
                MoveType::RookPromo,
                MoveType::QueenPromo
            ]
        );
        assert_eq!(
            kinds("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7", "b8"),
            [
                MoveType::KnightPromoCap,
                MoveType::BishopPromoCap,
                MoveType::RookPromoCap,
                MoveType::QueenPromoCap
            ]
        );
    }

    /// catches: every castling legality guard - occupancy on the king transit or
    /// destination square, the queen-side b-file rook gap, and castling out of,
    /// through, or into check.
    #[test]
    fn castling_legality_guards() {
        let ok = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        assert!(has(ok, "e1", "g1", MoveType::KingCastle));
        assert!(has(ok, "e1", "c1", MoveType::QueenCastle));
        assert!(!has(
            "r3k2r/8/8/8/8/8/8/R3K1NR w KQkq - 0 1",
            "e1",
            "g1",
            MoveType::KingCastle
        ));
        assert!(!has(
            "r3k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1",
            "e1",
            "c1",
            MoveType::QueenCastle
        ));
        assert!(!has(
            "r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1",
            "e1",
            "g1",
            MoveType::KingCastle
        ));
        assert!(!has(
            "r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1",
            "e1",
            "g1",
            MoveType::KingCastle
        ));
        assert!(!has(
            "r3k2r/8/8/8/8/8/6r1/R3K2R w KQkq - 0 1",
            "e1",
            "g1",
            MoveType::KingCastle
        ));
        assert!(!has(ok, "e1", "g1", MoveType::Quiet));
        assert!(!has(
            "r3k2r/8/8/8/8/8/8/R3K2R w kq - 0 1",
            "e1",
            "g1",
            MoveType::KingCastle
        ));
    }

    /// catches: a missing attacker class in `is_attacked`. Deleting the king
    /// branch fails no other test in the crate: perft only consults it for
    /// squares that a slider or knight already covers.
    #[test]
    fn is_attacked_covers_every_piece_type() {
        super::init();
        let sq = |s: &str| {
            let b = s.as_bytes();
            Square::new(b[0] - b'a', b[1] - b'1').unwrap()
        };
        for (fen, square, by) in [
            ("4k3/8/8/8/8/5n2/8/4K3 w - - 0 1", "e1", Color::Black),
            ("4k3/8/8/8/8/8/3p4/4K3 w - - 0 1", "e1", Color::Black),
            ("4k3/8/8/8/8/8/8/r3K3 w - - 0 1", "e1", Color::Black),
            ("4k3/8/8/8/1b6/8/8/4K3 w - - 0 1", "e1", Color::Black),
            ("4k3/8/8/8/8/8/8/q3K3 w - - 0 1", "e1", Color::Black),
            ("4k3/8/8/8/1q6/8/8/4K3 w - - 0 1", "e1", Color::Black),
            // A king guards its neighbours; d2 is empty, so the position stays
            // legal while still exercising the king branch.
            ("8/8/8/8/8/8/4k3/K7 w - - 0 1", "d2", Color::Black),
        ] {
            let board: Board = fen.parse().unwrap();
            assert!(is_attacked(&board, sq(square), by), "{fen}");
        }
        let quiet: Board = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        assert!(!is_attacked(&quiet, sq("e1"), Color::Black));
        assert!(!is_attacked(&quiet, sq("a1"), Color::Black));
    }

    /// catches: a capture generator that omits non-pawn captures or leaks quiet
    /// moves into the quiescence set. `tests/movegen.rs` walks a tree for the
    /// same property; this pins it in one position.
    #[test]
    fn captures_are_exactly_the_capturing_legal_moves() {
        super::init();
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut board: Board = fen.parse().unwrap();
        let us = board.state().side_to_move();

        let mut pseudo = MoveList::new();
        generate_captures(&board, &mut pseudo);
        let mut got = MoveList::new();
        for &mv in pseudo.iter() {
            board.make(mv);
            let ok = !is_attacked(&board, board.king_square(us), us.flip());
            board.unmake(mv);
            if ok {
                got.push(mv);
            }
        }

        let mut all = MoveList::new();
        generate_legal(&mut board, &mut all);
        let mut expected = MoveList::new();
        for &mv in all.iter().filter(|mv| mv.is_capture()) {
            expected.push(mv);
        }
        assert!(!expected.is_empty());
        assert_eq!(sorted(&got), sorted(&expected));
    }

    /// catches: move-set defects in general. Shallow counts over the same
    /// positions that `tests/movegen.rs` searches deeply, so a broken generator
    /// fails here in a fraction of a second.
    #[test]
    fn shallow_perft_reference_positions() {
        super::init();
        let cases = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                3,
                8_902,
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                2,
                2_039,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2_812),
            (
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                2,
                264,
            ),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
                2,
                1_486,
            ),
        ];
        for (fen, depth, expected) in cases {
            let mut board: Board = fen.parse().unwrap();
            assert_eq!(perft(&mut board, depth), expected, "{fen}");
        }
    }
}
