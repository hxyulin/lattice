//! Recursive search of game state using the following techniques to speed up and rank moves:
//!  - Negamax (minimax)
//!  - Alpha-beta pruning
//!  - MVV-LVA capture move ordering
//!  - Iterative-deepening
//!  - Quiescence search

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use lattice_board::{Board, Move, MoveFlag, MoveList, PieceType};

use crate::{Bound, MATE, Score, TranspositionTable, evaluate};

/// Upper bound on iterative-deepening depth when no explicit depth cap is given;
/// bounds the loop so the search always terminates.
pub const MAX_PLY: u32 = 64;

/// Hard cap on quiescence recursion depth
const MAX_QPLY: u32 = 32;

/// Stopping conditions for a [`search`].
///
/// The search halts as soon as *any* configured limit is hit.
#[derive(Debug, Clone, Default)]
pub struct Limits {
    /// Hard cap on iterative-deepening depth.
    pub depth: Option<u32>,
    /// Node budget
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
    /// Nodes visited during the search (including quiescence nodes)
    pub nodes: u64,
    /// Subset of [`Self::nodes`] spent in quiescence. Split out for debugging -
    /// a runaway ratio flags a quiescence explosion.
    pub qnodes: u64,
    /// The deepest iterative-deepening iteration that completed. Less than the
    /// requested depth when a node or time budget aborted the search.
    pub depth: u32,
}

/// Search `board` under `limits` and return the best move with its score.
///
/// Iterative deepening drives the search; it stops at whichever of the depth,
/// node, and time limits fires first. A bare [`Limits::default`] runs to
/// [`MAX_PLY`].
#[must_use]
pub fn search(board: &mut Board, limits: &Limits, tt: &mut TranspositionTable) -> SearchResult {
    tt.new_search(); // age the previous move's entries before this search reuses them
    let mut searcher = Searcher {
        nodes: 0,
        qnodes: 0,
        node_limit: limits.nodes,
        deadline: limits.move_time.map(|t| Instant::now() + t),
        stopped: false,
        armed: false,
        stop: limits.stop.clone(),
        tt,
    };
    let mut best_move: Option<Move> = None;
    let mut score = -MATE;
    let mut completed = 0;

    // iterative-deepening require at least depth 1
    for d in 1..=limits.max_depth() {
        let (bm, sc) = searcher.search_root(board, d, best_move);
        if searcher.stopped {
            break; // partial iteration: discard it, keep the last complete depth
        }
        (best_move, score, completed) = (bm, sc, d);
        searcher.armed = true;
    }

    SearchResult {
        best_move,
        score,
        nodes: searcher.nodes,
        qnodes: searcher.qnodes,
        depth: completed,
    }
}

/// Mutable search state threaded through the recursion
struct Searcher<'a> {
    nodes: u64,
    /// Subset of `nodes` spent in [`Searcher::quiescence`]. Surfaced in
    /// [`SearchResult`] for debugging the quiescence/main-search split.
    qnodes: u64,
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
    /// Transposition table, owned by the caller and reused across moves. Probed
    /// for cutoffs and a move-ordering hint, and written after each node.
    tt: &'a mut TranspositionTable,
}

impl Searcher<'_> {
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
    /// `hint` - the best move from the previous iterative-deepening iteration
    fn search_root(
        &mut self,
        board: &mut Board,
        depth: u32,
        hint: Option<Move>,
    ) -> (Option<Move>, Score) {
        let mut best_move = None;
        let mut best = -MATE;

        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        moves.sort_by_key(|&m| -order_score(board, m, hint));
        for mv in &moves {
            if self.should_stop() {
                break; // budget hit mid-iteration; `search` discards this depth
            }
            let undo = board.make_move(*mv);
            if board.is_legal() {
                let score = -self.negamax(board, depth - 1, 1, -MATE, MATE);
                if score > best {
                    best = score;
                    best_move = Some(*mv);
                }
            }
            board.unmake_move(*mv, undo);
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

    /// Negamax score of `board` searched to `depth` plies. `ply` is the distance
    /// from the root, used only to make mate scores prefer shorter mates.
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
            // Resolve pending captures before evaluating, so the static eval
            // only ever sees a quiet position (no piece hanging mid-exchange).
            return self.quiescence(board, 0, alpha, beta);
        }

        // Transposition probe. A stored entry whose search was at least as deep
        // (`entry.depth() >= depth`) can cut this node outright, subject to its
        // bound; either way its best move is the ordering hint (see below).
        let orig_alpha = alpha;
        let hash = board.zobrist();
        let mut tt_move = None;
        if let Some(e) = self.tt.probe(hash) {
            tt_move = e.best();
            if u32::from(e.depth()) >= depth {
                let s = e.score(ply);
                match e.bound() {
                    Bound::Exact => return s,
                    Bound::Lower if s >= beta => return s,
                    Bound::Upper if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let mut best = -MATE;
        let mut best_move = None;
        let mut legal = 0u32;

        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        if depth >= ORDER_MIN_DEPTH {
            // The TT move (this position's best from a previous, possibly
            // shallower, search) is ordered first - the strongest hint there is.
            moves.sort_by_key(|&m| -order_score(board, m, tt_move));
        }
        for mv in &moves {
            let undo = board.make_move(*mv);
            if board.is_legal() {
                legal += 1;
                let score = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha);
                if score > best {
                    best = score;
                    best_move = Some(*mv);
                }
                if score >= beta {
                    board.unmake_move(*mv, undo);
                    // Fail-high: `best` is a lower bound on the true score.
                    self.store(hash, best_move, best, depth, Bound::Lower, ply);
                    return best;
                }
                alpha = alpha.max(score);
            }
            board.unmake_move(*mv, undo);
        }

        if legal == 0 {
            // Terminal node:
            //  - checkmate is `MATE` discounted by distance from the root
            //  - if not in check, then it is stalemate (draw)
            let score = if board.in_check(board.side_to_move()) {
                -(MATE - ply as Score)
            } else {
                0
            };
            self.store(hash, None, score, depth, Bound::Exact, ply);
            return score;
        }

        // A score that beat `orig_alpha` was searched inside the window -> exact;
        // otherwise every move failed low and `best` is only an upper bound.
        let bound = if best > orig_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.store(hash, best_move, best, depth, bound, ply);
        best
    }

    /// Write a node's result to the transposition table, but only while the
    /// search is running: an aborted iteration's children return the sentinel `0`,
    /// so storing its garbage scores would poison the table.
    fn store(
        &mut self,
        hash: lattice_board::ZobristHash,
        best: Option<Move>,
        score: Score,
        depth: u32,
        bound: Bound,
        ply: u32,
    ) {
        if !self.stopped {
            self.tt.store(hash, best, score, depth as u8, bound, ply);
        }
    }

    /// Quiescence search: resolve captures and promotions from a leaf until the
    /// position is quiet, then evaluate. Fixes the horizon problem.
    ///
    /// # Notes
    /// Bounded by the stand-pat cutoff (`evaluate` is a lower bound, since the
    /// side can decline to capture), captures/promotions only (every ply removes
    /// material), and the [`MAX_QPLY`] cap; `should_stop` bounds it by time/nodes.
    fn quiescence(&mut self, board: &mut Board, qply: u32, mut alpha: Score, beta: Score) -> Score {
        self.nodes += 1;
        self.qnodes += 1;
        if self.should_stop() {
            return 0; // abort: the whole partial iteration is discarded
        }

        // Quiescence is not probed. A probe+store (git b51739e) measured
        // strength-neutral at this engine's ~3M nps: the per-qnode probe cost
        // (~-18% NPS) cancels the ~-30% node savings while the eval stays cheap.
        // Restore with `git cherry-pick b51739e`.

        // Stand-pat: the score if the side to move makes no capture at all. It's
        // a lower bound on this node (a quiet move is always available in a real
        // game), so it seeds `best` and gates the alpha-beta window.
        let stand_pat = evaluate(board);
        let mut best = stand_pat;
        if best >= beta {
            return best; // already too good; the opponent won't enter this line
        }
        if qply >= MAX_QPLY {
            return best; // hard depth backstop
        }
        alpha = alpha.max(best);

        // No capture-only generator: generate all moves and filter to
        // captures/promotions below.
        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        moves.sort_by_key(|&m| -order_score(board, m, None));
        for mv in &moves {
            if !(mv.flag().is_capture() || mv.flag().is_promotion()) {
                continue; // quiet move: not searched in quiescence
            }
            // Delta pruning is disabled: it cut ~23% of qnodes but its per-node
            // `board.in_check()` probe dropped NPS 4.49M -> 3.50M (equal-time
            // SPRT -77 +/- 28 Elo). Re-enable with a SEE primitive for in_check.
            //
            // let in_check = board.in_check(board.side_to_move());
            // if !in_check
            //     && !mv.flag().is_promotion()
            //     && stand_pat + captured_value(board, *mv) + DELTA_MARGIN < alpha
            // { continue; }

            let undo = board.make_move(*mv);
            if board.is_legal() {
                let score = -self.quiescence(board, qply + 1, -beta, -alpha);
                if score >= beta {
                    board.unmake_move(*mv, undo);
                    return score; // fail-soft beta cutoff
                }
                best = best.max(score);
                alpha = alpha.max(best);
            }
            board.unmake_move(*mv, undo);
        }

        best
    }
}

/// Skip MVV-LVA ordering at remaining depth below this.
///
/// # Performance
/// The depth-1 frontier is the largest node layer but its children are leaves
/// (static eval), so sorting there costs more than the sibling evals it saves.
const ORDER_MIN_DEPTH: u32 = 2;

const VAL: [i32; 6] = [100, 320, 330, 500, 900, 0];

// Delta-pruning helpers, disabled with the pruning itself (see `quiescence`).
//
// /// Safety margin for quiescence delta pruning (~two pawns). A capture is only
// /// pruned when the static eval plus the victim's value plus this margin still
// /// falls short of alpha, so the margin is the slack that keeps a capture which
// /// sets up a *further* gain (a discovered threat, a follow-up win) from being
// /// pruned on its immediate material alone.
// const DELTA_MARGIN: i32 = 200;
//
// /// Centipawn value of the piece a capture removes - the gain estimate delta
// /// pruning tests against alpha. En passant takes a pawn (the destination is
// /// empty); every other capture takes whatever sits on the destination square.
// fn captured_value(board: &Board, mv: Move) -> i32 {
//     let victim = if mv.flag() == MoveFlag::EnPassant {
//         PieceType::Pawn
//     } else {
//         board.piece_at(mv.to()).unwrap().kind()
//     };
//     VAL[victim as usize]
// }

/// Ordering bonus that floats the previous iteration's best move ahead of every
/// capture.
///
/// # Notes
/// Well above the ~90k MVV-LVA ceiling and nowhere near i32 overflow, so ordering
/// scores need no bit-packing.
const PV_BONUS: i32 = 1_000_000;

fn order_score(board: &Board, mv: Move, hint: Option<Move>) -> i32 {
    if Some(mv) == hint {
        return PV_BONUS; // last iteration's best move: searched first
    }
    if !mv.flag().is_capture() {
        return 0; // quiet moves last - killers/history will slot in here later
    }
    let attacker = board.piece_at(mv.from()).unwrap().kind();
    let victim = if mv.flag() == MoveFlag::EnPassant {
        PieceType::Pawn // captured pawn sits beside dest, not on it
    } else {
        board.piece_at(mv.to()).unwrap().kind()
    };
    VAL[victim as usize] * 100 - VAL[attacker as usize] // MVV dominates, LVA breaks ties
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_board::Square;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen.as_bytes()).unwrap()
    }

    fn sq(s: &str) -> Square {
        Square::from_ascii(s.as_bytes()).unwrap()
    }

    /// Run a search with a throwaway 1 MB table - these tests probe behaviour,
    /// not cross-move reuse.
    fn go(b: &mut Board, limits: &Limits) -> SearchResult {
        search(b, limits, &mut TranspositionTable::new(1))
    }

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn preset_stop_flag_returns_a_legal_move_after_depth_one() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        // Flag already raised before the search starts. Depth 1 still completes
        // (the `armed` gate ignores every stop until the first iteration banks a
        // move), so we always get a legal move; depth 2+ is aborted immediately.
        let mut b = board(STARTPOS);
        let limits = Limits {
            stop: Some(Arc::new(AtomicBool::new(true))),
            ..Limits::default()
        };
        let r = go(&mut b, &limits);
        assert!(r.best_move.is_some(), "depth 1 must yield a legal move");
        assert!(r.depth >= 1);
    }

    #[test]
    fn stop_flag_set_from_another_thread_halts_an_infinite_search() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        // `TranspositionTable` is already in scope via `use super::*`.

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
            search(&mut b, &limits, &mut TranspositionTable::new(1))
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
        let r = go(&mut b, &Limits::to_depth(1));
        let mv = r.best_move.expect("a legal move exists");
        assert_eq!(mv.from(), sq("e2"));
        assert_eq!(mv.to(), sq("d3"));
        assert_eq!(r.score, 100);
    }

    #[test]
    fn finds_mate_in_one() {
        // Ra8 is back-rank mate;
        // Needs depth 2: the mated node must be expanded (depth >= 1 there) to
        // discover it has no legal replies
        let mut b = board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        let r = go(&mut b, &Limits::to_depth(2));
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
        let r = go(&mut b, &Limits::to_depth(1));
        assert_eq!(r.best_move, None);
        assert_eq!(r.score, 0);
    }

    #[test]
    fn quiescence_avoids_a_defended_capture() {
        // White is up a queen for two pawns. Qxd5 grabs a pawn but the d5 pawn
        // is defended by c6 - c6xd5 wins the queen back. A depth-1 search WITHOUT
        // quiescence stops right after Qxd5 and scores it +800 (a "free" pawn on
        // top of the queen), so it plays the blunder. With quiescence the
        // recapture is resolved, Qxd5 scores as losing the queen, and the engine
        // keeps its material instead.
        let mut b = board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1");
        let r = go(&mut b, &Limits::to_depth(1));
        let mv = r.best_move.expect("a legal move exists");
        assert_ne!(mv.to(), sq("d5"), "must not grab the defended pawn");
        assert!(r.score < 800, "no phantom won pawn: {}", r.score);
        assert!(r.qnodes > 0, "leaves should reach quiescence");
    }

    #[test]
    fn node_budget_stops_early_but_returns_a_move() {
        // A tiny node budget must still yield a legal move - depth 1 always
        // completes (see `armed`) - while keeping the search short.
        let mut b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let r = go(&mut b, &Limits::to_nodes(5_000));
        assert!(r.best_move.is_some(), "must return a move under any budget");
        assert!(
            r.nodes < 50_000,
            "node budget should cap the search: {}",
            r.nodes
        );
        assert!(r.depth >= 1);
    }
}
