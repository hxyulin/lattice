//! Recursive search of game state using the following techniques to speed up and rank moves:
//!  - Negamax (minimax)
//!  - Alpha-beta pruning
//!  - Iterative deepening

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::{Board, Move};

use crate::{MATE, Score, evaluate};

/// Upper bound on iterative-deepening depth when no explicit depth cap is given;
/// bounds the loop so the search always terminates.
pub const MAX_PLY: u32 = 64;

/// Stopping conditions for a [`search`].
///
/// The search halts as soon as *any* configured limit is hit.
#[derive(Debug, Clone, Default)]
pub struct Limits {
    /// Hard cap on iterative-deepening depth.
    pub depth: Option<u32>,
    /// Node budget.
    pub nodes: Option<u64>,
    /// Wall-clock budget for the whole search.
    pub move_time: Option<Duration>,
    /// External stop signal raised by another thread; aborts at the next
    /// `should_stop()` check (after depth 1). `None` (the default) means none.
    pub stop: Option<Arc<AtomicBool>>,
}

impl Limits {
    /// A pure depth limit - search exactly `depth` plies.
    #[must_use]
    pub fn to_depth(depth: u32) -> Self {
        Self {
            depth: Some(depth),
            ..Self::default()
        }
    }

    /// A pure node budget.
    #[must_use]
    pub fn to_nodes(nodes: u64) -> Self {
        Self {
            nodes: Some(nodes),
            ..Self::default()
        }
    }

    /// A pure wall-clock budget.
    #[must_use]
    pub fn to_move_time(move_time: Duration) -> Self {
        Self {
            move_time: Some(move_time),
            ..Self::default()
        }
    }

    /// Depth at which to stop iterative deepening: the explicit cap, or
    /// [`MAX_PLY`] when only a node/time budget bounds the search.
    #[must_use]
    pub fn max_depth(&self) -> u32 {
        self.depth.unwrap_or(MAX_PLY)
    }
}

/// Time to allot one move given the clock left and the per-move increment:
/// `remaining / 20 + increment / 2`.
#[must_use]
pub fn budget(remaining: Duration, increment: Duration) -> Duration {
    remaining / 20 + increment / 2
}

/// The outcome of a [`search`].
pub struct SearchResult {
    /// The best move, or `None` only when the side to move has no legal moves
    pub best_move: Option<Move>,
    /// Score of the position from the side-to-move's perspective
    pub score: Score,
    /// Nodes visited during the search
    pub nodes: u64,
    /// The deepest iterative-deepening iteration that completed. Less than the
    /// requested depth when a node or time budget aborted the search.
    pub depth: u32,
}

/// A snapshot of one completed iterative-deepening iteration, passed to
/// [`search_with_info`]'s callback so a UCI front end can emit an `info` line
/// per depth as the search deepens.
#[derive(Clone, Copy)]
pub struct SearchInfo {
    /// Depth, in plies, of the iteration that just completed.
    pub depth: u32,
    /// Best move from the root at this depth (`None` only with no legal move).
    pub best_move: Option<Move>,
    /// Score of `best_move`, from the side-to-move's perspective.
    pub score: Score,
    /// Cumulative nodes searched so far.
    pub nodes: u64,
}

/// Search `board` under `limits`, reporting each completed depth to `on_iter`.
///
/// Iterative deepening drives the search; it stops at whichever of the depth,
/// node, and time limits fires first. `on_iter` fires once per *completed*
/// iteration, in increasing-depth order, so a front end can show the search
/// deepening live; a partial iteration aborted by a budget is discarded and
/// never reported.
#[must_use]
pub fn search_with_info(
    board: &mut Board,
    limits: &Limits,
    on_iter: &mut dyn FnMut(SearchInfo),
) -> SearchResult {
    let mut searcher = Searcher {
        nodes: 0,
        node_limit: limits.nodes,
        deadline: limits.move_time.map(|t| Instant::now() + t),
        stopped: false,
        armed: false,
        stop: limits.stop.clone(),
    };
    let mut best_move = None;
    let mut score = -MATE;
    let mut completed = 0;

    // Iterative deepening: re-search from depth 1 upward. Each completed
    // iteration yields a usable best move (so the search is abortable and
    // time-manageable) and seeds the next one with its best move for root
    // ordering.
    for d in 1..=limits.max_depth() {
        let (bm, sc) = searcher.search_root(board, d, best_move);
        if searcher.stopped {
            break; // partial iteration: discard it, keep the last complete depth
        }
        (best_move, score, completed) = (bm, sc, d);
        searcher.armed = true;
        on_iter(SearchInfo {
            depth: d,
            best_move,
            score,
            nodes: searcher.nodes,
        });
    }

    SearchResult {
        best_move,
        score,
        nodes: searcher.nodes,
        depth: completed,
    }
}

/// Search `board` under `limits` and return the best move with its score.
///
/// The plain entry point for callers (bench, tests) that only want the final
/// result; see [`search_with_info`] for per-depth reporting.
#[must_use]
pub fn search(board: &mut Board, limits: &Limits) -> SearchResult {
    search_with_info(board, limits, &mut |_| {})
}

/// Mutable search state threaded through the recursion.
struct Searcher {
    nodes: u64,
    /// Stop once `nodes` reaches this. `None` = unlimited.
    node_limit: Option<u64>,
    /// Wall-clock stop time. `None` = unlimited.
    deadline: Option<Instant>,
    /// Set once a budget fires; makes the whole search unwind fast.
    stopped: bool,
    /// False during depth 1 so the first iteration always completes and yields a
    /// legal move even under a tiny budget. True after depth 1.
    armed: bool,
    /// External stop flag from [`Limits::stop`], polled in
    /// [`Searcher::should_stop`]. `None` = no external stop.
    stop: Option<Arc<AtomicBool>>,
}

impl Searcher {
    /// Whether a node/time budget has been hit.
    ///
    /// # Performance
    /// Node compare runs every call (cheap); `Instant::now()` (a syscall) only
    /// every 2048 nodes.
    fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if !self.armed {
            return false;
        }
        if let Some(flag) = &self.stop
            && flag.load(Ordering::Relaxed)
        {
            self.stopped = true;
            return true;
        }
        if let Some(limit) = self.node_limit
            && self.nodes >= limit
        {
            self.stopped = true;
            return true;
        }
        if let Some(deadline) = self.deadline
            && self.nodes & 2047 == 0
            && Instant::now() >= deadline
        {
            self.stopped = true;
            return true;
        }
        false
    }

    /// Search the root to `depth` plies, returning the best move and its score.
    ///
    /// `hint` is the best move from the previous iterative-deepening iteration;
    /// it is searched first (root PV-move ordering). With no pruning yet this
    /// does not change the result, but sets up the ordering that alpha-beta will
    /// later exploit.
    fn search_root(
        &mut self,
        board: &mut Board,
        depth: u32,
        hint: Option<Move>,
    ) -> (Option<Move>, Score) {
        let mut best_move = None;
        let mut best = -MATE;
        // The root is a PV node: open the window fully, then raise alpha as
        // better moves are found so later root moves search a narrower window.
        let mut alpha = -MATE;
        let beta = MATE;

        let moves = board.pseudo_legal_moves();
        let ordered = hint
            .into_iter()
            .chain(moves.iter().copied().filter(|&m| Some(m) != hint));
        for mv in ordered {
            if self.should_stop() {
                break; // budget hit mid-iteration; `search` discards this depth
            }
            let undo = board.make_move(mv);
            if board.is_legal() {
                let score = -self.negamax(board, depth - 1, 1, -beta, -alpha);
                if score > best {
                    best = score;
                    best_move = Some(mv);
                    alpha = alpha.max(score);
                }
            }
            board.unmake_move(mv, undo);
        }

        // No legal move at the root: checkmate (in check) or stalemate (draw).
        let score = if best_move.is_some() {
            best
        } else if board.in_check(board.side_to_move()) {
            -MATE
        } else {
            0
        };
        (best_move, score)
    }

    /// Negamax score of `board` searched to `depth` plies, bounded by the
    /// `[alpha, beta]` window. `ply` is the distance from the root, used only to
    /// make mate scores prefer shorter mates. Fail-soft: the returned score may
    /// fall outside the window.
    fn negamax(
        &mut self,
        board: &mut Board,
        depth: u32,
        ply: u32,
        mut alpha: Score,
        beta: Score,
    ) -> Score {
        self.nodes += 1;
        if self.should_stop() {
            return 0; // abort: `search` discards the whole partial iteration
        }

        if depth == 0 {
            return evaluate(board);
        }

        let mut best = -MATE;
        let mut legal = 0u32;

        for mv in &board.pseudo_legal_moves() {
            let undo = board.make_move(*mv);
            if board.is_legal() {
                legal += 1;
                let score = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha);
                if score >= beta {
                    board.unmake_move(*mv, undo);
                    return score; // beta cutoff
                }
                alpha = alpha.max(score);
                best = best.max(score);
            }
            board.unmake_move(*mv, undo);
        }

        if legal == 0 {
            // Terminal node:
            //  - checkmate is `MATE` discounted by distance from the root
            //  - if not in check, then it is stalemate (draw)
            return if board.in_check(board.side_to_move()) {
                -(MATE - ply as Score)
            } else {
                0
            };
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Square;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen.as_bytes()).unwrap()
    }

    fn sq(s: &str) -> Square {
        Square::from_ascii(s.as_bytes()).unwrap()
    }

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn preset_stop_flag_returns_a_legal_move_after_depth_one() {
        // Flag already raised before the search starts. Depth 1 still completes
        // (the `armed` gate ignores every stop until the first iteration banks a
        // move), so we always get a legal move; depth 2+ is aborted immediately.
        let mut b = board(STARTPOS);
        let limits = Limits {
            stop: Some(Arc::new(AtomicBool::new(true))),
            ..Limits::default()
        };
        let r = search(&mut b, &limits);
        assert!(r.best_move.is_some(), "depth 1 must yield a legal move");
        assert!(r.depth >= 1);
    }

    #[test]
    fn stop_flag_set_from_another_thread_halts_an_infinite_search() {
        // `Limits::default()` has no depth/node/time cap -> runs to MAX_PLY (an
        // "infinite" search). A second thread raises the flag; the search must
        // observe it (cross-thread) and return a legal move rather than running
        // to depth 64.
        let flag = Arc::new(AtomicBool::new(false));
        let worker_flag = flag.clone();
        let handle = std::thread::spawn(move || {
            let mut b = board(STARTPOS);
            let limits = Limits {
                stop: Some(worker_flag),
                ..Limits::default()
            };
            search(&mut b, &limits)
        });
        // The `armed` gate guarantees depth 1 finishes regardless of timing, so
        // raising the flag immediately is safe and non-flaky.
        flag.store(true, Ordering::Relaxed);
        let r = handle.join().expect("search thread panicked");
        assert!(r.best_move.is_some());
        assert!(r.depth >= 1);
    }

    #[test]
    fn grabs_a_hanging_queen() {
        // White pawn e2 can capture an undefended Black queen on d3.
        let mut b = board("4k3/8/8/8/8/3q4/4P3/4K3 w - - 0 1");
        let r = search(&mut b, &Limits::to_depth(1));
        let mv = r.best_move.expect("a legal move exists");
        assert_eq!(mv.from(), sq("e2"));
        assert_eq!(mv.to(), sq("d3"));
        // After the grab White is up a queen, so the score is clearly positive.
        // (Exact centipawns belong to the eval's own tests, not here.)
        assert!(r.score > 0, "winning after the grab: {}", r.score);
    }

    #[test]
    fn finds_mate_in_one() {
        // Ra8 is back-rank mate;
        // Needs depth 2: the mated node must be expanded (depth >= 1 there) to
        // discover it has no legal replies
        let mut b = board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        let r = search(&mut b, &Limits::to_depth(2));
        assert_eq!(
            r.best_move.map(|m| (m.from(), m.to())),
            Some((sq("a1"), sq("a8")))
        );
        assert_eq!(r.score, MATE - 1); // mate delivered one ply from the root
    }

    #[test]
    fn stalemate_scores_zero() {
        // Classic stalemate: Black to move, not in check, no legal move.
        let mut b = board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        let r = search(&mut b, &Limits::to_depth(1));
        assert_eq!(r.best_move, None);
        assert_eq!(r.score, 0);
    }

    #[test]
    fn node_budget_stops_early_but_returns_a_move() {
        // A tiny node budget must still yield a legal move - depth 1 always
        // completes (see `armed`) - while keeping the search short.
        let mut b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let r = search(&mut b, &Limits::to_nodes(5_000));
        assert!(r.best_move.is_some(), "must return a move under any budget");
        assert!(
            r.nodes < 50_000,
            "node budget should cap the search: {}",
            r.nodes
        );
        assert!(r.depth >= 1);
    }
}
