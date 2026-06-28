//! A fixed-suite search benchmark for documenting engine development.
//!
//! # Notes
//! Node counts are deterministic (a build signature); NPS is wall-clock speed
//! and hardware-dependent.

use std::time::{Duration, Instant};

use lattice_board::Board;

use crate::{Limits, TranspositionTable, search};

/// `(label, FEN)` for each benchmark position: the six standard perft positions.
const SUITE: &[(&str, &str)] = &[
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    (
        "position4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ),
    (
        "position5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ),
    (
        "position6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
    ),
];

/// One position's benchmark result.
pub struct BenchEntry {
    /// Human label for the position.
    pub name: &'static str,
    /// Nodes the search visited - deterministic for a given position and depth.
    /// Includes [`Self::qnodes`].
    pub nodes: u64,
    /// Subset of [`Self::nodes`] spent in quiescence.
    pub qnodes: u64,
    /// Wall-clock time the search took.
    pub elapsed: Duration,
}

/// The full benchmark result: one [`BenchEntry`] per suite position.
pub struct BenchReport {
    /// Depth every position was searched to.
    pub depth: u32,
    /// Per-position results, in suite order.
    pub entries: Vec<BenchEntry>,
}

impl BenchReport {
    /// Total nodes across the suite - the build's bench signature.
    #[must_use]
    pub fn total_nodes(&self) -> u64 {
        self.entries.iter().map(|e| e.nodes).sum()
    }

    /// Total quiescence nodes across the suite (a subset of [`Self::total_nodes`]).
    #[must_use]
    pub fn total_qnodes(&self) -> u64 {
        self.entries.iter().map(|e| e.qnodes).sum()
    }

    /// Quiescence nodes per second over the whole suite (machine-dependent).
    #[must_use]
    pub fn qnps(&self) -> u64 {
        nps(self.total_qnodes(), self.total_time())
    }

    /// Total wall-clock time across the suite.
    #[must_use]
    pub fn total_time(&self) -> Duration {
        self.entries.iter().map(|e| e.elapsed).sum()
    }

    /// Nodes per second over the whole suite (machine-dependent).
    #[must_use]
    pub fn nps(&self) -> u64 {
        nps(self.total_nodes(), self.total_time())
    }
}

/// Nodes per second, clamping time up to 1us so a sub-microsecond search can't
/// divide by zero.
#[must_use]
pub fn nps(nodes: u64, elapsed: Duration) -> u64 {
    let micros = elapsed.as_micros().max(1);
    (u128::from(nodes) * 1_000_000 / micros) as u64
}

/// Run the search benchmark: search every suite position to `depth`, collecting
/// node counts and timings.
///
/// # Panics
/// Panics if a hard-coded suite FEN fails to parse - that is a bug in the suite,
/// not a runtime condition.
#[must_use]
pub fn bench(depth: u32) -> BenchReport {
    // Build the magic slider tables before timing
    lattice_board::init_tables();
    let entries = SUITE
        .iter()
        .map(|&(name, fen)| {
            let mut board = Board::from_fen(fen.as_bytes()).expect("suite FEN must parse");
            // Fresh table per position: keeps node counts independent and
            // deterministic, no cross-position carryover.
            let mut tt = TranspositionTable::new(16);
            let start = Instant::now();
            let result = search(&mut board, &Limits::to_depth(depth), &mut tt);
            BenchEntry {
                name,
                nodes: result.nodes,
                qnodes: result.qnodes,
                elapsed: start.elapsed(),
            }
        })
        .collect();
    BenchReport { depth, entries }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Determinism is what makes total_nodes a usable build signature.
    #[test]
    fn node_counts_are_deterministic() {
        let a = bench(2);
        let b = bench(2);
        assert_eq!(a.total_nodes(), b.total_nodes());
        assert!(a.total_nodes() > 0);
    }
}
