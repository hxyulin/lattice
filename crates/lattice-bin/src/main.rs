//! A runnable UCI engine wrapping `lattice-engine` and `lattice-board`

use std::io::{self, BufReader};
use std::time::{Duration, Instant};

use lattice_board::{Board, Color, Move};
use lattice_engine::{Limits, MATE, Score, bench, budget, nps, search};
use lattice_uci::{Go, StartPos, UciCommand, UciInterface, UciMove};

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
                // Spec: do engine init on `uci`. Builds the magic slider tables
                // now (a few ms) so the first search isn't charged for it.
                lattice_board::init_tables();
                uci.send("id name lattice-engine").map_err(io_err)?;
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
                    let result = search(&mut board, &limits);
                    let elapsed = start.elapsed();

                    // Integer nps; clamp elapsed up to 1us so a sub-microsecond
                    // search doesn't divide by zero.
                    let micros = elapsed.as_micros().max(1);
                    let nps = (u128::from(result.nodes) * 1_000_000 / micros) as u64;
                    let pv = result
                        .best_move
                        .map_or_else(|| "0000".to_string(), |m| m.to_string());

                    let nodes = result.nodes;
                    uci.send(&format!(
                        "info depth {} score {} nodes {nodes} nps {nps} pv {pv}",
                        result.depth,
                        format_score(result.score),
                    ))
                    .map_err(io_err)?;
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

/// Translate a parsed UCI `go` into engine [`Limits`].
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
    let mut limits = Limits {
        depth: go.depth,
        nodes: go.nodes,
        move_time,
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
        "{:<11} {:>5} {:>12} {:>12} {:>12} {:>12}",
        "position", "depth", "nodes", "qnodes", "nps", "qnps"
    );
    for e in &report.entries {
        println!(
            "{:<11} {:>5} {:>12} {:>12} {:>12} {:>12}",
            e.name,
            depth,
            e.nodes,
            e.qnodes,
            nps(e.nodes, e.elapsed),
            nps(e.qnodes, e.elapsed),
        );
    }
    println!();
    println!("Nodes searched:   {}", report.total_nodes());
    println!("Q-nodes searched: {}", report.total_qnodes());
    println!("Nodes/second:     {}", report.nps());
    println!("Q-nodes/second:   {}", report.qnps());
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

fn io_err(e: lattice_uci::UciError) -> io::Error {
    match e {
        lattice_uci::UciError::Io(e) => e,
    }
}
