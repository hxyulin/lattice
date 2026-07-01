//! A runnable UCI engine wrapping `lattice-engine` and `lattice-board`

use std::io::{self, BufReader};
use std::time::Instant;

use lattice::{Board, Move};
use lattice::{MATE, Score, SearchInfo, bench, nps, search_with_info};
use lattice::{StartPos, UciCommand, UciInterface, UciMove};

/// Depth used for a bare `go`, no search limits exist, small enough to be fast
const DEFAULT_DEPTH: u32 = 4;

/// Depth for `lattice bench [depth]` when none is given.
const DEFAULT_BENCH_DEPTH: u32 = 4;

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
                    let depth = go.depth.unwrap_or(DEFAULT_DEPTH);
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
                    let result = search_with_info(&mut board, depth, &mut on_iter);
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
