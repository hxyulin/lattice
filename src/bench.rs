//! Fixed search benchmark.

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use crate::Board;
use crate::search::{Limits, search_inner};

const DEPTH: u32 = 4;
const POSITIONS: [&str; 12] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
    "8/5pk1/6p1/3p4/3P4/6P1/5PK1/8 w - - 0 1",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
    "8/5pk1/7p/8/8/7P/5PP1/4R1K1 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
];

/// Runs the fixed-depth benchmark and returns its deterministic node count.
pub fn run(output: &mut dyn Write) -> u64 {
    run_at(output, DEPTH)
}

fn run_at(output: &mut dyn Write, depth: u32) -> u64 {
    let start = Instant::now();
    let stop = AtomicBool::new(false);
    let mut nodes = 0;
    for fen in POSITIONS {
        let mut board: Board = fen.parse().expect("bench FEN must be valid");
        nodes += search_inner(
            &mut board,
            Limits {
                depth: Some(depth),
                infinite: true,
                ..Limits::default()
            },
            &stop,
            &mut std::io::sink(),
            false,
        )
        .total();
    }
    let millis = start.elapsed().as_millis().max(1);
    let nps = u128::from(nodes) * 1000 / millis;
    let _ = writeln!(output, "Nodes searched: {nodes}");
    let _ = writeln!(output, "Nodes/second: {nps}");
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shallower than the real bench: this covers the output format and
    // determinism, both of which are depth-independent. `bench_is_stable`
    // exercises the shipped depth.
    const TEST_DEPTH: u32 = 2;

    #[test]
    fn output_has_verifier_line_and_count_is_stable() {
        let mut first = Vec::new();
        let mut second = Vec::new();
        let first_nodes = run_at(&mut first, TEST_DEPTH);
        let second_nodes = run_at(&mut second, TEST_DEPTH);
        assert_eq!(first_nodes, second_nodes);
        let text = String::from_utf8(first).unwrap();
        assert!(
            text.lines()
                .any(|line| line == format!("Nodes searched: {first_nodes}"))
        );
    }

    #[test]
    #[ignore = "runs the full depth-4 suite; minutes in debug"]
    fn bench_is_stable() {
        let mut first = Vec::new();
        let mut second = Vec::new();
        assert_eq!(run(&mut first), run(&mut second));
    }
}
