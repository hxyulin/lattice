//! Slow movegen correctness tests.
//!
//! Deep perft is the definitive move generation check. These run against the
//! public API only; fast variants that pin the same defects live beside the
//! code in `src/movegen`.
//!
//! Perft comes in two depths. The shallow table runs in every profile, because
//! in a debug build the walk doubles as the broadest check of `debug_check`'s
//! zobrist and eval-accumulator invariants. The deep table is release-only,
//! where the same trees cost seconds instead of minutes.

use lattice::movegen::perft::{perft, perft_divide};
use lattice::movegen::{
    MoveList, generate_captures, generate_legal, generate_pseudo, generate_quiets, is_attacked,
};
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

/// The same positions at a depth every profile can afford. One ply shallower
/// is 20-50x fewer nodes, which still visits every structural case below -
/// depth buys repetition of them, not new kinds.
const SHALLOW: [(&str, u32, u64); 7] = [
    (POSITIONS[0], 4, 197_281),
    (POSITIONS[1], 3, 97_862),
    (POSITIONS[2], 4, 43_238),
    (POSITIONS[3], 3, 9_467),
    (POSITIONS[4], 3, 9_467),
    (POSITIONS[5], 3, 62_379),
    (KIWIPETE_LIKE, 3, 89_890),
];

const KIWIPETE_LIKE: &str =
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1";

/// catches: any defect that changes the generated move set - wrong slider
/// blocking, a missing knight or king direction, pawn file wrap, en passant
/// treated as a normal capture, promotion under-generation, and every castling
/// legality guard (transit occupancy, the b-file rook gap, castling out of or
/// through check).
///
/// Runs in every profile, unlike its deep counterpart, and that is the point:
/// in a debug build `Board::debug_check` re-derives the zobrist key and the
/// eval accumulator on every make/unmake, so walking these trees is also the
/// broadest test that the incremental updates stay correct. A release-only
/// perft would check leaf counts and leave both invariants unexercised.
#[test]
fn perft_reference_positions() {
    lattice::movegen::init();
    for (fen, depth, expected) in SHALLOW {
        let mut board: Board = fen.parse().unwrap();
        assert_eq!(perft(&mut board, depth), expected, "{fen}");
    }
}

/// The deep counterpart, one ply further on each position. Release-only: these
/// are ~50x the nodes and `debug_check` makes a debug build roughly 60x slower
/// again, which is minutes rather than seconds.
///
/// catches: what only depth can reach - a defect needing a longer move sequence
/// to set up than the shallow table plays out.
#[test]
#[cfg_attr(debug_assertions, ignore = "too slow in debug; run with --release")]
fn perft_reference_positions_deep() {
    lattice::movegen::init();
    let cases = [
        (POSITIONS[0], 5, 4_865_609),
        (POSITIONS[1], 4, 4_085_603),
        (POSITIONS[2], 5, 674_624),
        (POSITIONS[3], 4, 422_333),
        (POSITIONS[4], 4, 422_333),
        (POSITIONS[5], 4, 2_103_487),
        (KIWIPETE_LIKE, 4, 3_894_594),
    ];
    for (fen, depth, expected) in cases {
        let mut board: Board = fen.parse().unwrap();
        assert_eq!(perft(&mut board, depth), expected, "{fen}");
    }
}

/// catches: `perft_divide` recursing at the wrong depth or failing to unmake
/// between root moves. Neither shows up in a plain `perft` total.
///
/// Depth 4 rather than 5: the property is that the divide sums to the total,
/// and one ply less proves it just as well for 25x fewer nodes.
#[test]
fn divide_sums_to_perft() {
    lattice::movegen::init();
    let mut board = Board::startpos();
    let total: u64 = perft_divide(&mut board, 4).iter().map(|(_, n)| n).sum();
    assert_eq!(total, 197_281);
    assert_eq!(total, perft(&mut board, 4), "divide must sum to perft");
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

/// catches: the staged generator dropping or duplicating a move - a quiet
/// promotion counted as a capture, castling emitted in both halves, a pawn push
/// lost when the two halves were split apart. The staged search relies on
/// `Captures` and `Quiets` partitioning `All` exactly; if they do not, the
/// search silently never considers some moves.
#[test]
fn captures_and_quiets_partition_the_full_move_set() {
    lattice::movegen::init();
    for fen in POSITIONS {
        let mut board: Board = fen.parse().unwrap();
        check_partition(&mut board, 3);
    }
}

fn check_partition(board: &mut Board, depth: u32) {
    let mut full = MoveList::new();
    generate_pseudo(board, &mut full);

    let mut split = MoveList::new();
    generate_captures(board, &mut split);
    let captures = split.len();
    generate_quiets(board, &mut split);

    assert_eq!(
        sorted(&full),
        sorted(&split),
        "staged generation disagrees with generate_pseudo at {board}"
    );
    // A move landing in both halves would still pass the set comparison above
    // if another were missing, and would be searched twice.
    assert!(
        split.iter().take(captures).all(|mv| mv.is_capture()),
        "non-capture in the capture half at {board}"
    );
    assert!(
        split.iter().skip(captures).all(|mv| !mv.is_capture()),
        "capture in the quiet half at {board}"
    );

    if depth > 1 {
        let mut legal = MoveList::new();
        generate_legal(board, &mut legal);
        for &mv in legal.iter() {
            board.make(mv);
            check_partition(board, depth - 1);
            board.unmake(mv);
        }
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
