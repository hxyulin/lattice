#![allow(missing_docs)]

//! Criterion micro-benchmarks for hot chess primitives.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lattice_board::{Bitboard, Board, Move, MoveFlag, Square};

/// A single benchmark case.
struct Case {
    name: &'static str,
    fen: &'static str,
}

fn bench_fen_parse(c: &mut Criterion) {
    let cases: &[Case] = &[
        Case {
            name: "startpos",
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        },
        // dense middlegame, every field present.
        Case {
            name: "kiwipete",
            fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        },
        // empty board, partial FEN (missing trailing fields).
        Case {
            name: "empty_partial",
            fen: "8/8/8/8/8/8/8/8 b",
        },
    ];

    let mut group = c.benchmark_group("fen_parse");
    for Case { name, fen } in cases {
        let bytes = fen.as_bytes();
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| Board::from_fen(black_box(bytes)).unwrap())
        });
    }
    group.finish();
}

/// Benchmark move generation (pseudo-legal moves only, no legality check).
fn bench_movegen(c: &mut Criterion) {
    // Positions spanning branching factors: opening, dense tactical middlegame,
    // and a sparse endgame. Boards are parsed once; we bench only generation.
    let cases: &[Case] = &[
        Case {
            name: "startpos",
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        },
        Case {
            name: "kiwipete",
            fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        },
        Case {
            name: "endgame",
            fen: "8/2k5/3p4/p2P1p2/P2P1P2/8/8/2K5 w - - 0 1",
        },
    ];

    let mut group = c.benchmark_group("movegen");
    for Case { name, fen } in cases {
        let board = Board::from_fen(fen.as_bytes()).unwrap();
        group.bench_function(*name, |b| {
            b.iter(|| {
                let moves = black_box(&board).pseudo_legal_moves();
                // must use reference to avoid full memory read of `MoveList`
                black_box(&moves);
            })
        });
    }
    group.finish();
}

fn bench_make_unmake(c: &mut Criterion) {
    struct MoveCase {
        name: &'static str,
        fen: &'static str,
        mv: Move,
    }

    fn mv(src: &str, dst: &str, flag: MoveFlag) -> Move {
        Move::new(
            Square::from_ascii(src.as_bytes()).unwrap(),
            Square::from_ascii(dst.as_bytes()).unwrap(),
            flag,
        )
    }

    let cases: &[MoveCase] = &[
        MoveCase {
            name: "quiet",
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            mv: mv("b1", "c3", MoveFlag::Quiet),
        },
        MoveCase {
            name: "capture",
            fen: "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
            mv: mv("e4", "d5", MoveFlag::Capture),
        },
        MoveCase {
            name: "castle",
            fen: "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
            mv: mv("e1", "g1", MoveFlag::KingCastle),
        },
    ];

    let mut group = c.benchmark_group("make_unmake");
    for MoveCase { name, fen, mv } in cases {
        let mut board = Board::from_fen(fen.as_bytes()).unwrap();
        let mv = *mv;
        group.bench_function(*name, |b| {
            b.iter(|| {
                let undo = board.make_move(black_box(mv));
                board.unmake_move(black_box(mv), undo);
            })
        });
    }
    group.finish();
}

/// Benchmark perft (movegen + make/unmake) throughput.
fn bench_perft(c: &mut Criterion) {
    struct PerftCase {
        name: &'static str,
        fen: &'static str,
        depth: u32,
        nodes: u64,
    }

    let cases: &[PerftCase] = &[
        PerftCase {
            name: "startpos_d4",
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            depth: 4,
            nodes: 197_281,
        },
        PerftCase {
            name: "kiwipete_d3",
            fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            depth: 3,
            nodes: 97_862,
        },
    ];

    let mut group = c.benchmark_group("perft");

    for PerftCase {
        name,
        fen,
        depth,
        nodes,
    } in cases
    {
        let mut board = Board::from_fen(fen.as_bytes()).unwrap();
        group.throughput(Throughput::Elements(*nodes));
        group.bench_function(*name, |b| b.iter(|| board.perft(black_box(*depth))));
    }
    group.finish();
}

fn bench_bitboard_iter(c: &mut Criterion) {
    // A half-full board's worth of set bits
    let bb = Bitboard::from_bits(0xFFFF_0000_00FF_FF00);

    c.bench_function("bitboard_iter", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for sq in black_box(bb) {
                acc += sq.index() as u32;
            }
            acc
        })
    });
}

fn bench_square_from_ascii(c: &mut Criterion) {
    c.bench_function("square_from_ascii", |b| {
        b.iter(|| Square::from_ascii(black_box(b"e4")).unwrap())
    });
}

criterion_group!(
    benches,
    bench_fen_parse,
    bench_movegen,
    bench_make_unmake,
    bench_perft,
    bench_bitboard_iter,
    bench_square_from_ascii
);
criterion_main!(benches);
