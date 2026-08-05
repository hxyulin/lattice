//! Slow movegen correctness tests.
//!
//! Deep perft is the definitive move generation check. These run against the
//! public API only; fast variants that pin the same defects live beside the
//! code in `src/movegen`.

use lattice::movegen::perft::{perft, perft_divide};
use lattice::movegen::{MoveList, generate_captures, generate_legal, is_attacked};
use lattice::{Board, Move};

const POSITIONS: [&str; 7] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
];

fn sorted(list: &MoveList) -> Vec<Move> {
    let mut moves: Vec<_> = list.iter().copied().collect();
    moves.sort_unstable_by_key(|mv| {
        (mv.move_type() as u16) << 12 | (mv.from().index() as u16) << 6 | mv.to().index() as u16
    });
    moves
}

/// catches: any defect that changes the generated move set - wrong slider
/// blocking, a missing knight or king direction, pawn file wrap, en passant
/// treated as a normal capture, promotion under-generation, and every castling
/// legality guard (transit occupancy, the b-file rook gap, castling out of or
/// through check).
#[test]
fn perft_reference_positions() {
    lattice::movegen::init();
    let cases = [
        (POSITIONS[0], 5, 4_865_609),
        (POSITIONS[1], 4, 4_085_603),
        (POSITIONS[2], 5, 674_624),
        (POSITIONS[3], 4, 422_333),
        (POSITIONS[4], 4, 422_333),
        (POSITIONS[5], 4, 2_103_487),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
            4,
            3_894_594,
        ),
    ];
    for (fen, depth, expected) in cases {
        let mut board: Board = fen.parse().unwrap();
        assert_eq!(perft(&mut board, depth), expected, "{fen}");
    }
}

/// catches: `perft_divide` recursing at the wrong depth or failing to unmake
/// between root moves. Neither shows up in a plain `perft` total.
#[test]
fn divide_sums_to_perft() {
    lattice::movegen::init();
    let mut board = Board::startpos();
    let total: u64 = perft_divide(&mut board, 5).iter().map(|(_, n)| n).sum();
    assert_eq!(total, 4_865_609);
}

/// catches: a capture generator that omits non-pawn captures or leaks quiet
/// moves into the quiescence move set. A leaf count cannot see either, since
/// `generate_captures` is not on the perft path.
#[test]
fn captures_match_the_full_generator() {
    lattice::movegen::init();
    for fen in POSITIONS {
        let mut board: Board = fen.parse().unwrap();
        check_captures(&mut board, 3);
    }
}

fn check_captures(board: &mut Board, depth: u32) {
    let mut legal = MoveList::new();
    generate_legal(board, &mut legal);

    let us = board.state().side_to_move();
    let mut pseudo_captures = MoveList::new();
    generate_captures(board, &mut pseudo_captures);
    let mut legal_captures = MoveList::new();
    for &mv in pseudo_captures.iter() {
        board.make(mv);
        let ok = !is_attacked(board, board.king_square(us), us.flip());
        board.unmake(mv);
        if ok {
            legal_captures.push(mv);
        }
    }

    let mut expected = MoveList::new();
    for &mv in legal.iter().filter(|mv| mv.is_capture()) {
        expected.push(mv);
    }
    assert_eq!(
        sorted(&legal_captures),
        sorted(&expected),
        "capture generator disagrees at {board}"
    );

    if depth > 1 {
        for &mv in legal.iter() {
            board.make(mv);
            check_captures(board, depth - 1);
            board.unmake(mv);
        }
    }
}
