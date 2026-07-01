//! A runnable UCI engine wrapping `lattice-engine` and `lattice-board`

use std::io::{self, BufReader};
use std::time::{Duration, Instant};

use lattice::{Board, Color, Move};
use lattice::{Go, StartPos, UciCommand, UciInterface, UciMove};
use lattice::{Limits, MATE, Score, SearchInfo, bench, budget, nps, search_with_info};

/// Depth used for a bare `go`, no search limits exist, small enough to be fast
const DEFAULT_DEPTH: u32 = 4;

/// Depth for `lattice bench [depth]` when none is given.
const DEFAULT_BENCH_DEPTH: u32 = 4;

/// Wall-clock reserved per move before the think-time deadline.
///
/// # Notes
/// The search polls the clock only every few thousand nodes and must still
/// unwind and emit `bestmove`, so without a margin it overshoots and forfeits
/// on time. A fixed 10ms margin is uniform across builds, so it does not skew
/// equal-time comparisons the way a raw overrun (which scales with each build's
/// NPS) would.
const MOVE_OVERHEAD: Duration = Duration::from_millis(10);

fn main() -> io::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("bench") {
        let depth = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_BENCH_DEPTH);
        print_bench(depth);
        return Ok(());
    }

    let stdin = io::stdin();
    let mut uci = UciInterface::new(BufReader::new(stdin.lock()), io::stdout().lock());
    let mut board = Board::starting_position();

    while let Some(cmd) = uci.poll().map_err(io_err)? {
        match cmd {
            UciCommand::Uci => {
                uci.send("id name Lattice").map_err(io_err)?;
                uci.send("id author hxyulin").map_err(io_err)?;
                uci.send("uciok").map_err(io_err)?;
            }
            UciCommand::IsReady => uci.send("readyok").map_err(io_err)?,
            UciCommand::NewGame => board = Board::starting_position(),
            UciCommand::Position { start, moves } => {
                let base = match start {
                    StartPos::Startpos => Some(Board::starting_position()),
                    StartPos::Fen(fen) => Board::from_fen(&fen).ok(),
                };
                // Ignore a malformed FEN rather than corrupting the current board.
                if let Some(mut b) = base {
                    for um in &moves {
                        if let Some(mv) = resolve(&b, *um) {
                            let _ = b.make_move(mv); // discard Undo: GUI moves aren't unmade
                        }
                    }
                    board = b;
                }
            }
            UciCommand::Go(go) => {
                if let Some(depth) = go.perft {
                    let mut total = 0;
                    for (mv, n) in board.perft_divide(depth) {
                        uci.send(&format!("{mv}: {n}")).map_err(io_err)?;
                        total += n;
                    }
                    uci.send(&format!("\nNodes searched: {total}"))
                        .map_err(io_err)?;
                } else {
                    let limits = limits_from_go(&go, board.side_to_move());
                    let start = Instant::now();
                    // Emit an `info` line as each iterative-deepening depth
                    // completes, so a GUI/Lichess sees the search progress live.
                    let mut on_iter = |info: SearchInfo| {
                        let pv = info
                            .best_move
                            .map_or_else(|| "0000".to_string(), |m| m.to_string());
                        let _ = uci.send(&format!(
                            "info depth {} score {} nodes {} nps {} pv {pv}",
                            info.depth,
                            format_score(info.score),
                            info.nodes,
                            nps(info.nodes, start.elapsed()),
                        ));
                    };
                    let result = search_with_info(&mut board, &limits, &mut on_iter);
                    drop(on_iter);

                    let pv = result
                        .best_move
                        .map_or_else(|| "0000".to_string(), |m| m.to_string());
                    uci.send(&format!("bestmove {pv}")).map_err(io_err)?;
                }
            }
            UciCommand::Stop => {} // nothing to stop while single-threaded
            UciCommand::Quit => break,
            UciCommand::Unknown(_) => {} // UCI: ignore unrecognized input
        }
    }
    Ok(())
}

/// Reserve [`MOVE_OVERHEAD`] from an allotted think time, flooring at 1ms.
///
/// # Notes
/// A tiny budget still yields a move (depth 1 always completes via the search's
/// `armed` guard) rather than getting a zero-length deadline.
fn reserve_overhead(t: Duration) -> Duration {
    t.saturating_sub(MOVE_OVERHEAD)
        .max(Duration::from_millis(1))
}

/// Translate a parsed UCI `go` into engine [`Limits`].
///
/// `movetime` wins outright; otherwise the side-to-move's clock and increment
/// give a per-move budget via [`budget`], less a [`reserve_overhead`] margin. A
/// bare `go` with no limit at all falls back to [`DEFAULT_DEPTH`] so the engine
/// never searches forever.
fn limits_from_go(go: &Go, stm: Color) -> Limits {
    let move_time = go.movetime.map(Duration::from_millis).or_else(|| {
        let (remaining, inc) = match stm {
            Color::White => (go.wtime, go.winc),
            Color::Black => (go.btime, go.binc),
        };
        remaining.map(|r| {
            budget(
                Duration::from_millis(r),
                Duration::from_millis(inc.unwrap_or(0)),
            )
        })
    });
    // Reserve the fixed move-overhead margin from any wall-clock budget so the
    // search's poll-and-unwind overshoot stays inside the allotted time.
    let move_time = move_time.map(reserve_overhead);
    let mut limits = Limits {
        depth: go.depth,
        nodes: go.nodes,
        move_time,
        stop: None, // set by the caller when a worker thread runs the search
    };

    if !go.infinite
        && limits.depth.is_none()
        && limits.nodes.is_none()
        && limits.move_time.is_none()
    {
        limits.depth = Some(DEFAULT_DEPTH);
    }
    limits
}

/// Run the fixed-suite search benchmark to `depth` and print a human-readable
/// table plus the conventional `Nodes searched` / `Nodes/second` summary lines.
///
/// Node counts are deterministic (a build signature); NPS is machine-dependent.
fn print_bench(depth: u32) {
    let report = bench(depth);
    println!(
        "{:<11} {:>5} {:>12} {:>12}",
        "position", "depth", "nodes", "nps"
    );
    for e in &report.entries {
        println!(
            "{:<11} {:>5} {:>12} {:>12}",
            e.name,
            depth,
            e.nodes,
            nps(e.nodes, e.elapsed),
        );
    }
    println!();
    println!("Nodes searched: {}", report.total_nodes());
    println!("Nodes/second:   {}", report.nps());
    println!("{} nodes {} nps", report.total_nodes(), report.nps());
}

/// Format a score as a UCI `score` field: `cp <centipawns>` for a normal score,
/// or `mate <n>` (in full moves, signed) when it's a mate score.
fn format_score(score: Score) -> String {
    // Any score within plausible mate distance of MATE is a mate score; 1000 is
    // far more ply than the search reaches.
    const MATE_THRESHOLD: Score = MATE - 1000;
    if score >= MATE_THRESHOLD {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_THRESHOLD {
        format!("mate {}", -((MATE + score + 1) / 2))
    } else {
        format!("cp {score}")
    }
}

/// Turn a UCI from/to(/promo) into a flagged engine [`Move`]
fn resolve(board: &Board, um: UciMove) -> Option<Move> {
    board
        .pseudo_legal_moves()
        .into_iter()
        .find(|m| m.from() == um.from && m.to() == um.to && m.flag().promoted_piece() == um.promo)
}

fn io_err(e: lattice::UciError) -> io::Error {
    match e {
        lattice::UciError::Io(e) => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_overhead_subtracts_and_floors() {
        // A normal budget loses exactly the overhead.
        assert_eq!(
            reserve_overhead(Duration::from_millis(100)),
            Duration::from_millis(90)
        );
        // A budget at or below the overhead floors at 1ms - never zero - so the
        // engine still produces a move instead of getting a dead deadline.
        assert_eq!(
            reserve_overhead(Duration::from_millis(5)),
            Duration::from_millis(1)
        );
    }
}
