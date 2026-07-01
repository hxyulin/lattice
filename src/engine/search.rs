//! Recursive search of game state using the following techniques to speed up and rank moves:
//!  - Negamax (minimax)
//!  - Alpha-beta pruning
//!  - MVV-LVA capture move ordering
//!  - Killer-move ordering
//!  - Iterative deepening
//!  - Quiescence search

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::{Board, Color, Move, MoveFlag, MoveList, PieceType};

use crate::{MATE, Score, evaluate};

/// Upper bound on iterative-deepening depth when no explicit depth cap is given;
/// bounds the loop so the search always terminates.
pub const MAX_PLY: u32 = 64;

/// Hard cap on quiescence recursion depth.
const MAX_QPLY: u32 = 32;

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
    /// Nodes visited during the search (including quiescence nodes)
    pub nodes: u64,
    /// Subset of [`Self::nodes`] spent in quiescence. Split out for debugging -
    /// a runaway ratio flags a quiescence explosion.
    pub qnodes: u64,
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
        qnodes: 0,
        node_limit: limits.nodes,
        deadline: limits.move_time.map(|t| Instant::now() + t),
        stopped: false,
        armed: false,
        stop: limits.stop.clone(),
        killers: [[None; 2]; MAX_PLY as usize],
        history: [[[0; 64]; 64]; 2],
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
        qnodes: searcher.qnodes,
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
    /// Killer moves: up to two quiet beta-cutoff moves per ply, ordered ahead of
    /// other quiets at sibling nodes (see [`order_score`]). Persists across the
    /// search; slot 0 is the most recent.
    killers: [[Option<Move>; 2]; MAX_PLY as usize],
    /// Butterfly history (`[side][from][to]`): cumulative quiet-cutoff score that
    /// orders non-killer quiets. Persists across the search; saturates at
    /// [`HISTORY_MAX`].
    history: HistoryTable,
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

        let mut moves = board.pseudo_legal_moves();
        let killers = self.killers[0];
        moves.sort_by_key(|&m| -order_score(board, m, killers, &self.history));
        // The PV-move hint still leads; captures then killers then quiets follow.
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
            // Resolve pending captures before evaluating, so the static eval
            // only ever sees a quiet position (no piece hanging mid-exchange).
            return self.quiescence(board, 0, alpha, beta);
        }

        let mut best = -MATE;
        let mut legal = 0u32;
        // Quiets searched at this node that did not cut; charged a history malus
        // if a later quiet causes the cutoff.
        let mut tried_quiets = MoveList::new();

        let mut moves = board.pseudo_legal_moves();
        if depth >= ORDER_MIN_DEPTH {
            let killers = self.killers[ply as usize];
            moves.sort_by_key(|&m| -order_score(board, m, killers, &self.history));
        }
        for mv in &moves {
            let undo = board.make_move(*mv);
            if board.is_legal() {
                legal += 1;
                let score = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha);
                if score >= beta {
                    board.unmake_move(*mv, undo);
                    // A quiet move that fails high becomes a killer and earns a
                    // history bonus; the quiets searched before it earn a malus.
                    // Captures and promotions are excluded (MVV-LVA orders them).
                    // After the unmake, `side_to_move` is the mover again.
                    if !mv.flag().is_capture() && !mv.flag().is_promotion() {
                        let side = board.side_to_move();
                        let bonus = (depth * depth) as i32;
                        self.store_killer(ply, *mv);
                        self.update_history(side, *mv, bonus);
                        for q in tried_quiets.as_slice() {
                            self.update_history(side, *q, -bonus);
                        }
                    }
                    return score; // beta cutoff
                }
                alpha = alpha.max(score);
                best = best.max(score);
                if !mv.flag().is_capture() && !mv.flag().is_promotion() {
                    tried_quiets.push(*mv);
                }
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

    /// Record a quiet cutoff move as a killer for `ply`, most-recent in slot 0.
    /// A move already in slot 0 is a no-op, so it cannot evict slot 1 with a copy.
    fn store_killer(&mut self, ply: u32, mv: Move) {
        let slot = &mut self.killers[ply as usize];
        if slot[0] != Some(mv) {
            slot[1] = slot[0];
            slot[0] = Some(mv);
        }
    }

    /// Nudge a quiet move's butterfly-history entry by a signed `delta` (positive
    /// for the cutoff move, negative for quiets that were searched but did not
    /// cut). Uses the history-gravity update: the entry is pulled toward the new
    /// evidence in proportion to how saturated it already is, so it behaves as an
    /// exponential moving average bounded by [`HISTORY_MAX`] (recent evidence
    /// outweighs stale evidence, and the entry can go negative to mark a quiet as
    /// bad) rather than a saturating sum. `side` is the move's mover.
    fn update_history(&mut self, side: Color, mv: Move, delta: i32) {
        let cap = HISTORY_MAX;
        let cell = &mut self.history[side.as_u8() as usize][mv.from().index() as usize]
            [mv.to().index() as usize];
        let d = delta.clamp(-cap, cap);
        *cell += d - *cell * d.abs() / cap;
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
        // captures/promotions below. Killers and history are quiet-only, so pass
        // empty killer slots.
        let mut moves = board.pseudo_legal_moves();
        moves.sort_by_key(|&m| -order_score(board, m, [None, None], &self.history));
        for mv in &moves {
            if !(mv.flag().is_capture() || mv.flag().is_promotion()) {
                continue; // quiet move: not searched in quiescence
            }
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

/// Piece values for move ordering (not evaluation), indexed by `PieceType`.
const VAL: [i32; 6] = [100, 320, 330, 500, 900, 0];

/// Ordering bonuses for killer moves.
///
/// # Notes
/// Below every capture (the smallest MVV-LVA score is 9100) and above ordinary
/// quiets (0), so a proven refutation is searched right after captures. Slot 0
/// (more recent) outranks slot 1.
const KILLER_1_BONUS: i32 = 9_000;
const KILLER_2_BONUS: i32 = 8_000;

/// Butterfly history table: a running `depth^2` count per `[side][from][to]` of
/// quiet beta-cutoff moves, used to order non-killer quiets (see [`order_score`]).
type HistoryTable = [[[i32; 64]; 64]; 2];

/// Saturation cap on a history entry.
///
/// # Notes
/// Held one band below [`KILLER_2_BONUS`] so a history-ranked quiet never
/// outranks a killer, keeping the ordering tiers fixed.
const HISTORY_MAX: i32 = 7_000;

/// MVV-LVA capture key with killer and history quiets slotted in: captures first
/// (most-valuable-victim / least-valuable-attacker), then this ply's two killers,
/// then the remaining quiets ranked by history (0 if never a cutoff). Higher
/// sorts earlier. History is capped below the killer band, so the tiers never
/// cross.
fn order_score(board: &Board, mv: Move, killers: [Option<Move>; 2], history: &HistoryTable) -> i32 {
    if !mv.flag().is_capture() {
        if Some(mv) == killers[0] {
            return KILLER_1_BONUS;
        }
        if Some(mv) == killers[1] {
            return KILLER_2_BONUS;
        }
        let side = board.side_to_move().as_u8() as usize;
        return history[side][mv.from().index() as usize][mv.to().index() as usize];
    }
    let attacker = board.piece_at(mv.from()).unwrap().kind();
    let victim = if mv.flag() == MoveFlag::EnPassant {
        PieceType::Pawn // the captured pawn sits beside the destination, not on it
    } else {
        board.piece_at(mv.to()).unwrap().kind()
    };
    VAL[victim as usize] * 100 - VAL[attacker as usize] // MVV dominates, LVA breaks ties
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
    fn quiescence_avoids_a_defended_capture() {
        // White is up a queen for two pawns. Qxd5 grabs a pawn but the d5 pawn
        // is defended by c6 - c6xd5 wins the queen back. A depth-1 search WITHOUT
        // quiescence stops right after Qxd5 and scores it as a "free" pawn on top
        // of the queen, so it plays the blunder. With quiescence the recapture is
        // resolved, Qxd5 scores as losing the queen, and the engine keeps its
        // material instead.
        let mut b = board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1");
        let r = search(&mut b, &Limits::to_depth(1));
        let mv = r.best_move.expect("a legal move exists");
        assert_ne!(mv.to(), sq("d5"), "must not grab the defended pawn");
        // Had it blundered Qxd5 the score would crater (a queen down for a pawn);
        // staying clearly ahead proves quiescence resolved the recapture.
        assert!(r.score > 500, "kept its material: {}", r.score);
        assert!(r.qnodes > 0, "leaves should reach quiescence");
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

    #[test]
    fn killer_sorts_after_captures_and_before_other_quiets() {
        // White Pe4 has a capture (e4xd5) and several quiets. Make one quiet the
        // slot-0 killer and assert the ordering contract: capture > killer > quiet.
        let b = board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        let moves = b.pseudo_legal_moves();

        let mut capture = None;
        let mut quiets = Vec::new();
        for &m in &moves {
            if m.flag().is_capture() {
                capture = Some(m);
            } else if !m.flag().is_promotion() {
                quiets.push(m);
            }
        }
        let capture = capture.expect("e4xd5 is a capture");
        assert!(quiets.len() >= 2, "need a killer and another quiet");

        let killer = quiets[0];
        let other = quiets[1];
        let killers = [Some(killer), None];
        let empty = [[[0; 64]; 64]; 2];

        let cap_score = order_score(&b, capture, killers, &empty);
        let killer_score = order_score(&b, killer, killers, &empty);
        let other_score = order_score(&b, other, killers, &empty);

        assert!(cap_score > killer_score, "captures outrank killers");
        assert!(killer_score > other_score, "killer outranks plain quiets");
        assert_eq!(other_score, 0, "a non-killer quiet scores 0");
    }

    #[test]
    fn history_ranks_quiets_below_killers_and_above_cold_quiets() {
        // Same position. A quiet with history credit sorts above a never-tried
        // quiet (0) but still below a killer and a capture - the tier order is
        // capture > killer > history-quiet > cold-quiet.
        let b = board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        let moves = b.pseudo_legal_moves();

        let mut capture = None;
        let mut quiets = Vec::new();
        for &m in &moves {
            if m.flag().is_capture() {
                capture = Some(m);
            } else if !m.flag().is_promotion() {
                quiets.push(m);
            }
        }
        let capture = capture.expect("e4xd5 is a capture");
        assert!(
            quiets.len() >= 3,
            "need a killer, a hot quiet, and a cold one"
        );

        let killer = quiets[0];
        let hot = quiets[1]; // earns history credit
        let cold = quiets[2]; // never a cutoff
        let killers = [Some(killer), None];

        // Credit `hot` in the history table for White (side to move).
        let mut history = [[[0; 64]; 64]; 2];
        let white = b.side_to_move().as_u8() as usize;
        history[white][hot.from().index() as usize][hot.to().index() as usize] = 500;

        let cap_s = order_score(&b, capture, killers, &history);
        let killer_s = order_score(&b, killer, killers, &history);
        let hot_s = order_score(&b, hot, killers, &history);
        let cold_s = order_score(&b, cold, killers, &history);

        assert_eq!(hot_s, 500, "history quiet carries its score");
        assert!(cap_s > killer_s, "capture > killer");
        assert!(killer_s > hot_s, "killer > history quiet");
        assert!(hot_s > cold_s, "history quiet > cold quiet");
        assert_eq!(cold_s, 0, "a cold quiet scores 0");
    }
}
