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

use lattice::Board;
use lattice::eval::evaluate;
use lattice::movegen::{
    MoveList, attackers_to, generate_captures, generate_legal, generate_pseudo, is_attacked,
};
use lattice::tt::{Bound, TranspositionTable};

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

fn boards() -> Vec<(&'static str, Board)> {
    POSITIONS
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

criterion_group!(benches, movegen, make_unmake, attacks, eval, tt);
criterion_main!(benches);
