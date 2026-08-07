//! Microbenchmarks for the paths the profile says the search spends its time
//! in.
//!
//! These exist to attribute a change to a component. The full-search benchmark
//! in `search.rs` answers whether a change is worth having; these answer why,
//! and they resolve effects the full search buries under everything else it
//! does.
//!
//! Read the numbers as ratios between variants of the same benchmark, never as
//! absolutes: a microbenchmark keeps its working set in L1 and calls the same
//! function in a loop, so it flatters anything whose real cost is a cache miss
//! or a mispredicted branch in a cold search tree.

// `criterion_group!` and `criterion_main!` generate undocumented items.
#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::atomic::AtomicBool;

use lattice::Board;
use lattice::eval::evaluate;
use lattice::movegen::{
    MoveList, attackers_to, generate_captures, generate_legal, generate_legal_in_check,
    generate_pseudo, is_attacked,
};
use lattice::search::{Limits, search};
use lattice::tt::{Bound, TranspositionTable};

/// Shallow enough that one sample is milliseconds rather than seconds, which
/// is what lets Criterion collect enough of them to resolve a few percent.
const SEARCH_DEPTH: u32 = 5;

/// The start position, a tactical middlegame (kiwipete), and a pawn endgame.
/// Move counts and piece mix differ enough between them that a change helping
/// only one shows up as a split rather than as a small average.
const POSITIONS: [(&str, &str); 3] = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
];

/// Positions where the side to move is in check, which `POSITIONS` has none
/// of - and that omission is why these exist.
///
/// `generate_legal` takes a different path per case, and the three differ by
/// more than a constant: out of check a move is admitted on the pinned set
/// alone, under a single check it must first land on the checker or the
/// squares between, and under double check only the king may move at all.
/// Averaging them into one number is what hid a 21% search win behind a
/// microbenchmark that moved barely at all - the suite measured the branch
/// nobody was optimizing. The pseudo-to-legal ratio is the thing to watch:
/// it is the work the filter has to throw away.
///
/// Counts as of this commit, from `generate_pseudo` and `generate_legal`:
///
///     case                  pseudo  legal
///     single_check_midgame      59      5
///     single_check_endgame      35      4
///     double_check              28      3
///
/// What the suite is worth is what it reports across the commit that
/// introduced evasion generation, `legal_in_check` before and after:
///
///     single_check_midgame   913ns -> 195ns   -79%
///     single_check_endgame   592ns -> 176ns   -70%
///     double_check           438ns -> 154ns   -65%
const CHECK_POSITIONS: [(&str, &str); 3] = [
    // Reached by walking a game from kiwipete: a crowded board where 59
    // pseudo-legal moves collapse to 5, the widest discard in the sweep.
    (
        "single_check_midgame",
        "r4k2/pp2qp1p/n1b3N1/P1p5/B5P1/2bPr2P/8/RN3KR1 b - - 1 27",
    ),
    // The same shape with few pieces, where the fixed costs - the checker
    // scan, the between lookup - are a larger share of the whole.
    (
        "single_check_endgame",
        "8/1Kp1r3/8/1P6/4p1Pk/7R/8/5q2 b - - 1 9",
    ),
    // Rook on the e-file and knight on f3 both checking e1, so the target
    // mask is empty and every non-king move is discarded without a trial.
    ("double_check", "4rk2/8/8/8/8/5n2/PPP5/2Q1K2R w K - 0 1"),
];

fn boards() -> Vec<(&'static str, Board)> {
    POSITIONS
        .iter()
        .map(|(name, fen)| (*name, fen.parse().unwrap()))
        .collect()
}

fn check_boards() -> Vec<(&'static str, Board)> {
    CHECK_POSITIONS
        .iter()
        .map(|(name, fen)| (*name, fen.parse().unwrap()))
        .collect()
}

fn movegen(c: &mut Criterion) {
    lattice::movegen::init();
    let mut group = c.benchmark_group("movegen");
    for (name, board) in boards() {
        group.bench_function(format!("pseudo/{name}"), |b| {
            b.iter(|| {
                let mut list = MoveList::new();
                generate_pseudo(black_box(&board), &mut list);
                black_box(list.len())
            })
        });
        group.bench_function(format!("captures/{name}"), |b| {
            b.iter(|| {
                let mut list = MoveList::new();
                generate_captures(black_box(&board), &mut list);
                black_box(list.len())
            })
        });
        // The legality filter make/unmakes every pseudo-legal move, so this is
        // several times the cost of `pseudo` and is why the main search filters
        // after `make` instead of calling this.
        group.bench_function(format!("legal/{name}"), |b| {
            b.iter_batched_ref(
                || board.clone(),
                |board| {
                    let mut list = MoveList::new();
                    generate_legal(board, &mut list);
                    black_box(list.len())
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// The in-check half of `generate_legal`, which the main `movegen` group
/// cannot see because none of its positions are in check.
///
/// `pseudo` is here beside `legal` on the same board deliberately: the ratio
/// between them is the number that moves when evasion generation changes, and
/// reading `legal` alone cannot distinguish "the filter got cheaper" from
/// "the position simply has fewer moves".
fn evasions(c: &mut Criterion) {
    lattice::movegen::init();
    let mut group = c.benchmark_group("evasions");
    for (name, board) in check_boards() {
        group.bench_function(format!("pseudo/{name}"), |b| {
            b.iter(|| {
                let mut list = MoveList::new();
                generate_pseudo(black_box(&board), &mut list);
                black_box(list.len())
            })
        });
        group.bench_function(format!("legal/{name}"), |b| {
            b.iter_batched_ref(
                || board.clone(),
                |board| {
                    let mut list = MoveList::new();
                    generate_legal(board, &mut list);
                    black_box(list.len())
                },
                criterion::BatchSize::SmallInput,
            )
        });
        // What quiescence actually calls: it has already tested for check to
        // decide against stand-pat, so it passes the answer in rather than
        // paying for it twice.
        group.bench_function(format!("legal_in_check/{name}"), |b| {
            b.iter_batched_ref(
                || board.clone(),
                |board| {
                    let mut list = MoveList::new();
                    generate_legal_in_check(board, &mut list, true);
                    black_box(list.len())
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

/// Whole searches from single positions, which is where a movegen change has
/// to show up to be worth anything.
///
/// `benches/search.rs` times the fixed 12-position suite as one number. This
/// splits it per position and includes in-check ones, so a change that helps
/// only the evasion path is visible instead of being averaged against eleven
/// positions that never enter it. Depth is low enough that a sample is a few
/// milliseconds; the tree shape is what matters, not its size.
fn search_positions(c: &mut Criterion) {
    lattice::movegen::init();
    let mut group = c.benchmark_group("search_position");
    group.sample_size(30);
    for (name, board) in boards().into_iter().chain(check_boards()) {
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || (board.clone(), TranspositionTable::with_size_mb(1)),
                |(board, tt)| {
                    black_box(search(
                        board,
                        Limits {
                            depth: Some(SEARCH_DEPTH),
                            infinite: true,
                            ..Limits::default()
                        },
                        &AtomicBool::new(false),
                        tt,
                        &mut (),
                    ))
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn make_unmake(c: &mut Criterion) {
    lattice::movegen::init();
    let mut group = c.benchmark_group("make_unmake");
    for (name, board) in boards() {
        let mut list = MoveList::new();
        generate_pseudo(&board, &mut list);
        let moves: Vec<_> = list.iter().copied().collect();
        group.bench_function(format!("roundtrip/{name}"), |b| {
            b.iter_batched_ref(
                || board.clone(),
                |board| {
                    for &mv in &moves {
                        board.make(mv);
                        board.unmake(mv);
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn attacks(c: &mut Criterion) {
    lattice::movegen::init();
    let mut group = c.benchmark_group("attacks");
    for (name, board) in boards() {
        let us = board.state().side_to_move();
        let king = board.king_square(us);
        // Called once per node in the main search and again per node in
        // quiescence, so its cost is multiplied by the whole tree.
        group.bench_function(format!("is_attacked/{name}"), |b| {
            b.iter(|| is_attacked(black_box(&board), black_box(king), us.flip()))
        });
        // The inner loop of SEE, which recomputes it per exchange step against
        // a shrinking occupancy.
        group.bench_function(format!("attackers_to/{name}"), |b| {
            b.iter(|| attackers_to(black_box(&board), black_box(king), board.occupied()))
        });
    }
    group.finish();
}

fn eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval");
    for (name, board) in boards() {
        // Three loads from the incrementally maintained accumulator. Here to
        // show it is already O(1), which is why caching a static eval in the
        // transposition table would buy nothing.
        group.bench_function(name, |b| b.iter(|| evaluate(black_box(&board))));
    }
    group.finish();
}

fn tt(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt");
    let table = TranspositionTable::with_size_mb(16);
    // Keys spread across the table, so probes miss in cache the way they do in
    // a real search rather than hitting one resident line.
    let keys: Vec<u64> = (0..4096u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    for &key in &keys {
        table.store(key, 42, None, 5, Bound::Exact);
    }
    group.bench_function("probe_hit", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % keys.len();
            black_box(table.probe(black_box(keys[i])))
        })
    });
    group.bench_function("probe_miss", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % keys.len();
            black_box(table.probe(black_box(!keys[i])))
        })
    });
    group.bench_function("store", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % keys.len();
            table.store(black_box(keys[i]), 42, None, 5, Bound::Exact)
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    movegen,
    evasions,
    search_positions,
    make_unmake,
    attacks,
    eval,
    tt
);
criterion_main!(benches);
