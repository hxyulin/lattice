//! A runnable UCI engine wrapping `lattice-engine` and `lattice-board`

use std::io::{self, BufReader, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use lattice::{Board, Color, Move};
use lattice::{Go, StartPos, UciCommand, UciInterface, UciMove};
use lattice::{
    Limits, MATE, Score, SearchInfo, TranspositionTable, bench, budget, nps, search_with_info,
};

/// Depth used for a bare `go`, no search limits exist, small enough to be fast
const DEFAULT_DEPTH: u32 = 4;

/// Depth for `lattice bench [depth]` when none is given.
///
/// # Notes
/// Bumped from 4 to 6 once SEE pruning shrank the depth-4 suite to a sub-100ms
/// blip: depth 6 is a steadier NPS signal and a larger build fingerprint, and it
/// trends down (not up) as further pruning lands. The `Bench:` trailer /
/// `bench.csv` switch from depth-4 to depth-6 signatures at that commit.
const DEFAULT_BENCH_DEPTH: u32 = 6;

/// Default transposition-table size in MB.
///
/// # Notes
/// Used until a GUI overrides it via `setoption name Hash`; matches the
/// advertised `Hash` default.
const DEFAULT_HASH_MB: usize = 16;
/// Smallest accepted `Hash` value (MB) - one bucket-array still allocates.
const MIN_HASH_MB: usize = 1;
/// Largest accepted `Hash` value (MB). A sanity cap, not a hardware limit.
const MAX_HASH_MB: usize = 1024;

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
    // Output goes through emit() (a brief per-message stdout lock), not the
    // interface's writer, so give it a sink; it is only used to read commands.
    let mut uci = UciInterface::new(BufReader::new(stdin.lock()), io::sink());
    // The board and transposition table live in exactly one place at a time:
    // `idle`/`idle_tt` while nothing runs; on `go` they move into the worker
    // thread together and `running` holds the handle until `reap` joins the
    // worker and brings them both back. The table persists across moves, so each
    // `go` reuses what the previous search learned.
    let mut idle: Option<Board> = Some(Board::starting_position());
    let mut idle_tt: Option<TranspositionTable> = Some(TranspositionTable::new(DEFAULT_HASH_MB));
    let mut running: Option<RunningSearch> = None;

    while let Some(cmd) = uci.poll().map_err(io_err)? {
        match cmd {
            UciCommand::Uci => {
                emit("id name Lattice");
                emit("id author hxyulin");
                emit(&format!(
                    "option name Hash type spin default {DEFAULT_HASH_MB} min {MIN_HASH_MB} max {MAX_HASH_MB}"
                ));
                emit("uciok");
            }
            UciCommand::IsReady => emit("readyok"),
            UciCommand::NewGame => {
                reap(&mut idle, &mut idle_tt, &mut running);
                idle = Some(Board::starting_position());
                if let Some(tt) = idle_tt.as_mut() {
                    tt.clear(); // a new game shares nothing with the last
                }
            }
            UciCommand::SetOption { name, value } => {
                // Only `Hash` (in MB) is supported; ignore anything else, per UCI.
                // Reap first so the table is back in `idle_tt` to resize.
                if name.eq_ignore_ascii_case(b"Hash")
                    && let Some(mb) = value
                        .as_ref()
                        .and_then(|v| std::str::from_utf8(v).ok())
                        .and_then(|s| s.trim().parse::<usize>().ok())
                {
                    reap(&mut idle, &mut idle_tt, &mut running);
                    if let Some(tt) = idle_tt.as_mut() {
                        tt.resize(mb.clamp(MIN_HASH_MB, MAX_HASH_MB));
                    }
                }
            }
            UciCommand::Position { start, moves } => {
                reap(&mut idle, &mut idle_tt, &mut running);
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
                    idle = Some(b);
                }
            }
            UciCommand::Go(go) => {
                reap(&mut idle, &mut idle_tt, &mut running);
                if let Some(depth) = go.perft {
                    let board = idle.as_mut().expect("idle after reap");
                    let mut total = 0;
                    for (mv, n) in board.perft_divide(depth) {
                        emit(&format!("{mv}: {n}"));
                        total += n;
                    }
                    emit(&format!("\nNodes searched: {total}"));
                } else {
                    // Hand the board and table to a worker so the read loop stays
                    // free for `stop`/`isready`/`quit`. The worker prints its own
                    // `info`/`bestmove` (a channel back would not print until the
                    // main thread returned from its blocking stdin read) and
                    // returns both through the handle so the next search reuses the
                    // table.
                    let mut board = idle.take().expect("idle after reap");
                    let mut tt = idle_tt.take().expect("idle after reap");
                    let stop = Arc::new(AtomicBool::new(false));
                    let mut limits = limits_from_go(&go, board.side_to_move());
                    limits.stop = Some(stop.clone());
                    let handle = thread::spawn(move || {
                        let start = Instant::now();
                        // One `info` line per completed iterative-deepening depth.
                        let mut on_iter = |info: SearchInfo| {
                            let pv = info
                                .best_move
                                .map_or_else(|| "0000".to_string(), |m| m.to_string());
                            emit(&format!(
                                "info depth {} score {} nodes {} nps {} pv {pv}",
                                info.depth,
                                format_score(info.score),
                                info.nodes,
                                nps(info.nodes, start.elapsed()),
                            ));
                        };
                        let result = search_with_info(&mut board, &limits, &mut tt, &mut on_iter);
                        let pv = result
                            .best_move
                            .map_or_else(|| "0000".to_string(), |m| m.to_string());
                        emit(&format!("bestmove {pv}"));
                        (board, tt)
                    });
                    running = Some(RunningSearch { stop, handle });
                }
            }
            UciCommand::Stop => {
                // Signal the worker; it unwinds and prints `bestmove`. The board
                // is reclaimed by the next command's `reap`. No-op if idle.
                if let Some(r) = &running {
                    r.stop.store(true, Ordering::Relaxed);
                }
            }
            UciCommand::Quit => {
                reap(&mut idle, &mut idle_tt, &mut running); // stop + join the worker cleanly
                break;
            }
            UciCommand::Unknown(_) => {} // UCI: ignore unrecognized input
        }
    }
    Ok(())
}

/// A search running on a worker thread.
///
/// # Notes
/// The worker owns the `Board` and [`TranspositionTable`] for the search and
/// returns them through `handle`; `stop` is the flag the main thread raises to
/// abort it.
struct RunningSearch {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<(Board, TranspositionTable)>,
}

/// Bring the engine back to idle, joining any running worker and reclaiming its
/// `Board` and [`TranspositionTable`]. No-op if already idle.
///
/// # Notes
/// Invariant: on return, `idle.is_some()`, `idle_tt.is_some()`, and
/// `running.is_none()`.
fn reap(
    idle: &mut Option<Board>,
    idle_tt: &mut Option<TranspositionTable>,
    running: &mut Option<RunningSearch>,
) {
    if let Some(r) = running.take() {
        r.stop.store(true, Ordering::Relaxed);
        let (board, tt) = r.handle.join().expect("search thread panicked");
        *idle = Some(board);
        *idle_tt = Some(tt);
    }
}

/// Write one UCI line to stdout, newline-terminated and flushed, under a brief
/// per-message lock.
///
/// # Notes
/// Locking per call rather than holding a session-long `StdoutLock` lets a
/// search worker emit `info`/`bestmove` while the main thread emits `readyok`
/// without interleaving mid-line or deadlocking. Write errors are ignored: if
/// the GUI pipe is gone the read side hits EOF and the loop exits cleanly.
fn emit(line: &str) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
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
