//! Fixed search benchmark.

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use crate::Board;
#[cfg(feature = "profiling")]
use crate::search::SearchProfile;
use crate::search::{Limits, search_inner};
use crate::tt::TranspositionTable;

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
    // One table for the run, built fresh: shared across positions the way a
    // game shares it, and empty at the start so the count stays reproducible.
    let tt = TranspositionTable::new();
    let mut nodes = 0;
    let mut qnodes = 0;
    #[cfg(feature = "profiling")]
    let mut profile = SearchProfile::default();
    for fen in POSITIONS {
        let mut board: Board = fen.parse().expect("bench FEN must be valid");
        let result = search_inner(
            &mut board,
            Limits {
                depth: Some(depth),
                infinite: true,
                ..Limits::default()
            },
            &stop,
            &tt,
            &mut std::io::sink(),
            false,
        );
        nodes += result.nodes;
        qnodes += result.qnodes;
        #[cfg(feature = "profiling")]
        {
            profile += result.profile;
        }
    }
    let total = nodes + qnodes;
    let millis = start.elapsed().as_millis().max(1);
    let nps = u128::from(total) * 1000 / millis;
    // ChessEval anchors its bench signature to a line *starting* with `Nodes
    // searched`, so this one must stay unprefixed. `bench` is a one-shot
    // subcommand rather than part of a UCI session, so the protocol channel
    // is not in use here.
    let _ = writeln!(output, "Nodes searched: {total}");
    let _ = writeln!(output, "Nodes/second: {nps}");
    let _ = writeln!(
        output,
        "Qnodes: {qnodes} ({}% of total), main nodes: {nodes}",
        qnodes * 100 / total.max(1)
    );
    #[cfg(feature = "profiling")]
    write_profile(output, profile, qnodes);
    total
}

#[cfg(feature = "profiling")]
fn write_profile(output: &mut dyn Write, profile: SearchProfile, qnodes: u64) {
    write_distribution(
        output,
        "main cutoff index 1st/2nd/3rd/4th-8th/>8th",
        &profile.main_cutoffs,
    );
    write_distribution(
        output,
        "qsearch cutoff index 1st/2nd/3rd/4th-8th/>8th",
        &profile.q_cutoffs,
    );
    write_distribution(output, "main bounds high/low/exact", &profile.main_bounds);
    write_distribution(output, "qsearch bounds high/low/exact", &profile.q_bounds);
    let _ = writeln!(
        output,
        "info string profile TT probes {} hits {} ({:.2}%) collisions {} ({:.2}%) empty {} ({:.2}%) usable cutoffs {} ({:.2}% of hits) stores {} ({:.2}% of probes)",
        profile.tt_probes,
        profile.tt_hits,
        percent(profile.tt_hits, profile.tt_probes),
        profile.tt_collisions,
        percent(profile.tt_collisions, profile.tt_probes),
        profile.tt_probes - profile.tt_hits - profile.tt_collisions,
        percent(
            profile.tt_probes - profile.tt_hits - profile.tt_collisions,
            profile.tt_probes
        ),
        profile.tt_cutoffs,
        percent(profile.tt_cutoffs, profile.tt_hits),
        profile.tt_stores,
        percent(profile.tt_stores, profile.tt_probes),
    );
    let _ = writeln!(
        output,
        "info string profile qsearch in check {} ({:.2}%) stand-pat cutoffs {} / {} ({:.2}%)",
        profile.q_in_check,
        percent(profile.q_in_check, qnodes),
        profile.stand_pat_cutoffs,
        profile.stand_pat_nodes,
        percent(profile.stand_pat_cutoffs, profile.stand_pat_nodes),
    );
    let _ = writeln!(
        output,
        "info string profile PVS re-searches {} / {} ({:.2}%)",
        profile.pvs_researches,
        profile.pvs_probes,
        percent(profile.pvs_researches, profile.pvs_probes),
    );
    let qply_total = profile.qply.iter().sum();
    let values = profile
        .qply
        .iter()
        .enumerate()
        .map(|(qply, &count)| format!("{qply}:{count} ({:.2}%)", percent(count, qply_total)))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(output, "info string profile qply {values}");
}

#[cfg(feature = "profiling")]
fn write_distribution(output: &mut dyn Write, name: &str, counts: &[u64]) {
    let total = counts.iter().sum();
    let values = counts
        .iter()
        .map(|&count| format!("{count} ({:.2}%)", percent(count, total)))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(output, "info string profile {name}: {values}");
}

#[cfg(feature = "profiling")]
fn percent(count: u64, total: u64) -> f64 {
    count as f64 * 100.0 / total.max(1) as f64
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
        // Every line but the nps one, which is a speed reading and varies.
        let counts = |text: &str| {
            text.lines()
                .filter(|line| !line.starts_with("Nodes/second:"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            counts(&String::from_utf8(second).unwrap()),
            counts(&text),
            "the qnode split must be deterministic too"
        );
        let qnodes = text
            .lines()
            .find_map(|line| line.strip_prefix("Qnodes: "))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|n| n.parse::<u64>().ok())
            .expect("bench should report a qnode count");
        assert!(qnodes > 0, "every leaf runs quiescence");
        assert!(qnodes < first_nodes, "qnodes are part of the total");
    }

    // catches: the loop searching only some of POSITIONS. The count is the
    // whole point of the bench - it is the signature ChessEval compares
    // across commits - and nothing else notices if positions go missing.
    #[test]
    fn every_position_contributes_to_the_count() {
        let all = run_at(&mut std::io::sink(), TEST_DEPTH);
        let stop = AtomicBool::new(false);
        let mut summed = 0;
        for fen in POSITIONS {
            let tt = TranspositionTable::new();
            let mut board: Board = fen.parse().expect("bench FEN must be valid");
            let result = search_inner(
                &mut board,
                Limits {
                    depth: Some(TEST_DEPTH),
                    infinite: true,
                    ..Limits::default()
                },
                &stop,
                &tt,
                &mut std::io::sink(),
                false,
            );
            summed += result.nodes + result.qnodes;
        }
        assert_eq!(all, summed, "the bench must search every position once");
        assert_eq!(POSITIONS.len(), 12);
    }
}
