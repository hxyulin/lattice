//! Recursive search of game state using the following techniques to speed up and rank moves:
//!  - Negamax (minimax)
//!  - Alpha-beta pruning
//!  - MVV-LVA capture move ordering
//!  - Iterative-deepening
//!  - Quiescence search

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::{Board, Move, MoveFlag, MoveList, PieceType};

use crate::{Bound, MATE, Score, TranspositionTable, evaluate};

/// Upper bound on iterative-deepening depth when no explicit depth cap is given;
/// bounds the loop so the search always terminates.
pub const MAX_PLY: u32 = 64;

/// Hard cap on quiescence recursion depth
const MAX_QPLY: u32 = 32;

/// One search parameter exposed as an integer UCI spin option for SPSA tuning.
pub struct TunableSpec {
    /// UCI option name (e.g. `RfpMargin`).
    pub name: &'static str,
    /// Default value - reproduces the engine's built-in behavior.
    pub default: i32,
    /// Inclusive minimum the option accepts.
    pub min: i32,
    /// Inclusive maximum the option accepts.
    pub max: i32,
}

/// The SPSA-tunable search parameters, in UCI-option form.
///
/// Continuous knobs only (margins, reductions, ordering weights); discrete depth
/// gates stay constants, where SPSA has little to work with. Fractional knobs are
/// integer-only UCI options stored times 100 (the `x100` fields on [`Tunables`])
/// and divided at use.
pub const TUNABLES: &[TunableSpec] = &[
    TunableSpec {
        name: "RfpMargin",
        default: 80,
        min: 20,
        max: 200,
    },
    TunableSpec {
        name: "LmrBase",
        default: 75,
        min: 0,
        max: 200,
    },
    TunableSpec {
        name: "LmrDivisor",
        default: 225,
        min: 100,
        max: 400,
    },
    TunableSpec {
        name: "LmpBase",
        default: 3,
        min: 1,
        max: 10,
    },
    TunableSpec {
        name: "HistoryMax",
        default: 7000,
        min: 1000,
        max: 32000,
    },
    TunableSpec {
        name: "HistoryBonus",
        default: 100,
        min: 10,
        max: 400,
    },
    TunableSpec {
        name: "Killer1Bonus",
        default: 9000,
        min: 1000,
        max: 20000,
    },
    TunableSpec {
        name: "Killer2Bonus",
        default: 8000,
        min: 1000,
        max: 20000,
    },
];

/// Live values of the [`TUNABLES`] search parameters, threaded through the search.
///
/// [`Default`] reproduces the engine's built-in behavior exactly, so a default
/// search is byte-for-byte the untuned search (the bench signature is unchanged).
/// Set options by UCI name with [`Tunables::set`].
#[derive(Clone)]
pub struct Tunables {
    /// Reverse futility pruning margin, centipawns per ply.
    pub rfp_margin: i32,
    /// LMR base offset, times 100.
    pub lmr_base_x100: i32,
    /// LMR divisor, times 100.
    pub lmr_divisor_x100: i32,
    /// Late move pruning base count (the constant in `base + depth * depth`).
    pub lmp_base: i32,
    /// History entry saturation cap.
    pub history_max: i32,
    /// History cutoff-bonus scale, times 100 (`bonus = scale * depth * depth / 100`).
    pub history_bonus_x100: i32,
    /// Move-ordering bonus for the most-recent killer.
    pub killer1_bonus: i32,
    /// Move-ordering bonus for the older killer.
    pub killer2_bonus: i32,
    /// Precomputed LMR reductions, rebuilt when the base or divisor changes.
    lmr_table: [[u8; LMR_DIM]; LMR_DIM],
}

impl Default for Tunables {
    fn default() -> Self {
        let mut t = Tunables {
            rfp_margin: 0,
            lmr_base_x100: 0,
            lmr_divisor_x100: 0,
            lmp_base: 0,
            history_max: 0,
            history_bonus_x100: 0,
            killer1_bonus: 0,
            killer2_bonus: 0,
            lmr_table: [[0; LMR_DIM]; LMR_DIM],
        };
        // Apply each option's default; the final LMR rebuild (once the divisor is
        // set) yields the correct table, so any transient rebuild is harmless.
        for spec in TUNABLES {
            t.set(spec.name, spec.default);
        }
        t
    }
}

impl Tunables {
    /// Set one tunable by its UCI option name (case-insensitive), clamping to the
    /// option's range. Returns `false` for an unknown name. The LMR table is
    /// rebuilt when a value that feeds it changes.
    pub fn set(&mut self, name: &str, value: i32) -> bool {
        let Some(spec) = TUNABLES.iter().find(|s| s.name.eq_ignore_ascii_case(name)) else {
            return false;
        };
        let v = value.clamp(spec.min, spec.max);
        match spec.name {
            "RfpMargin" => self.rfp_margin = v,
            "LmrBase" => self.lmr_base_x100 = v,
            "LmrDivisor" => self.lmr_divisor_x100 = v,
            "LmpBase" => self.lmp_base = v,
            "HistoryMax" => self.history_max = v,
            "HistoryBonus" => self.history_bonus_x100 = v,
            "Killer1Bonus" => self.killer1_bonus = v,
            "Killer2Bonus" => self.killer2_bonus = v,
            _ => return false,
        }
        if matches!(spec.name, "LmrBase" | "LmrDivisor") {
            self.rebuild_lmr_table();
        }
        true
    }

    /// Recompute the LMR table from the current base and divisor.
    fn rebuild_lmr_table(&mut self) {
        let base = self.lmr_base_x100 as f64 / 100.0;
        let divisor = self.lmr_divisor_x100 as f64 / 100.0;
        for depth in 1..LMR_DIM {
            for moves in 1..LMR_DIM {
                let r = base + (depth as f64).ln() * (moves as f64).ln() / divisor;
                self.lmr_table[depth][moves] = r as u8;
            }
        }
    }

    /// Late-move reduction (plies) for a quiet move at `depth` with `move_number`
    /// legal moves already searched. Both indices are clamped to the table bounds.
    fn lmr_reduction(&self, depth: u32, move_number: u32) -> u32 {
        let d = (depth as usize).min(LMR_DIM - 1);
        let m = (move_number as usize).min(LMR_DIM - 1);
        u32::from(self.lmr_table[d][m])
    }

    /// Number of legal moves searched at `depth` before late move pruning skips
    /// the remaining quiets. Quadratic in depth.
    fn lmp_threshold(&self, depth: u32) -> u32 {
        self.lmp_base as u32 + depth * depth
    }

    /// History cutoff bonus credited to a quiet move that caused a cutoff at
    /// `depth`.
    fn history_bonus(&self, depth: u32) -> i32 {
        self.history_bonus_x100 * (depth * depth) as i32 / 100
    }
}

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
    /// Cumulative nodes searched so far, including quiescence.
    pub nodes: u64,
}

/// Search `board` under `limits` with the search parameters `tun`, returning the
/// best move with its score and reporting each completed depth to `on_iter`.
///
/// Iterative deepening drives the search; it stops at whichever of the depth,
/// node, and time limits fires first. A bare [`Limits::default`] runs to
/// [`MAX_PLY`]; [`Tunables::default`] runs the untuned search. `on_iter` fires
/// once per completed iteration, in increasing-depth order; a partial iteration
/// aborted by a limit is discarded and never reported.
#[must_use]
pub fn search_with_info(
    board: &mut Board,
    limits: &Limits,
    tt: &mut TranspositionTable,
    tun: &Tunables,
    on_iter: &mut dyn FnMut(SearchInfo),
) -> SearchResult {
    tt.new_search(); // age the previous move's entries before this search reuses them
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
        tt,
        tun: tun.clone(),
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

/// Search `board` under `limits`, returning the best move and score without
/// per-iteration reporting. The plain entry point for callers (bench, tests)
/// that only want the final result; see [`search_with_info`] for the callback.
#[must_use]
pub fn search(
    board: &mut Board,
    limits: &Limits,
    tt: &mut TranspositionTable,
    tun: &Tunables,
) -> SearchResult {
    search_with_info(board, limits, tt, tun, &mut |_| {})
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
    /// Killer moves: up to two quiet beta-cutoff moves per ply, tried first among
    /// quiets at sibling nodes (see [`order_score`]). Persists across the search;
    /// slot 0 is the most recent.
    killers: [[Option<Move>; 2]; MAX_PLY as usize],
    /// Butterfly history (`[side][from][to]`): cumulative quiet-cutoff score that
    /// orders non-killer quiets. Reset per search; saturates at [`HISTORY_MAX`].
    history: HistoryTable,
    /// Transposition table, owned by the caller and reused across moves. Probed
    /// for cutoffs and a move-ordering hint, and written after each node.
    tt: &'a mut TranspositionTable,
    /// SPSA-tunable search parameters; [`Tunables::default`] reproduces the
    /// untuned search.
    tun: Tunables,
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
        let killers = self.killers[0];
        moves.sort_by_key(|&m| -order_score(board, m, hint, killers, &self.history, &self.tun));
        // PVS at the root: `alpha` tracks the best score found so far. The first
        // legal move is searched with the full window for an exact baseline; later
        // moves are scouted with a null window and only re-searched if they beat it.
        let mut alpha = -MATE;
        for mv in &moves {
            if self.should_stop() {
                break; // budget hit mid-iteration; `search` discards this depth
            }
            let undo = board.make_move(*mv);
            if board.is_legal() {
                let score = if best_move.is_none() {
                    -self.negamax(board, depth - 1, 1, -MATE, MATE, true)
                } else {
                    let s = -self.negamax(board, depth - 1, 1, -alpha - 1, -alpha, true);
                    if s > alpha {
                        -self.negamax(board, depth - 1, 1, -MATE, -alpha, true)
                    } else {
                        s
                    }
                };
                if score > best {
                    best = score;
                    best_move = Some(*mv);
                }
                alpha = alpha.max(best);
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
        can_null: bool,
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

        // Whether this node is in check - loop-invariant, so computed once and
        // shared by null-move pruning and the LMR guard in the move loop.
        let in_check = board.in_check(board.side_to_move());

        // Reverse futility pruning (static null-move pruning). Near the leaves,
        // if the static eval already clears beta by a depth-scaled margin, the
        // side to move is far enough ahead that searching is almost certainly
        // wasted, so fail high on the static score directly. Confined to non-PV
        // nodes (where no exact PV is owed) and away from mate scores (which the
        // static eval cannot see); meaningless in check, so skipped there too.
        if !in_check
            && depth <= RFP_MAX_DEPTH
            && beta - alpha == 1
            && beta.abs() < MATE - MAX_PLY as Score
        {
            let eval = evaluate(board);
            if eval - self.tun.rfp_margin * depth as Score >= beta {
                return eval;
            }
        }

        // Null-move pruning. Pass to the opponent; if a reduced zero-window search
        // still beats beta, the real move would too, so prune. Skipped without
        // non-pawn material: pawn-only endgames are where zugzwang makes a "pass"
        // misleadingly good.
        if can_null
            && depth >= NMP_MIN_DEPTH
            && !in_check
            && has_non_pawn_material(board, board.side_to_move())
        {
            let undo = board.make_null_move();
            let score = -self.negamax(board, depth - 1 - NMP_R, ply + 1, -beta, -beta + 1, false);
            board.unmake_null_move(undo);
            if self.stopped {
                return 0; // the reduced search was aborted; discard this node
            }
            if score >= beta {
                // Don't propagate a mate score proved only by a reduced null
                // search - cap it to a plain fail-high so a false mate can't leak.
                return if score >= MATE - MAX_PLY as Score {
                    beta
                } else {
                    score
                };
            }
        }

        let mut best = -MATE;
        let mut best_move = None;
        let mut legal = 0u32;

        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        let killers = self.killers[ply as usize];
        if depth >= ORDER_MIN_DEPTH {
            // The TT move (this position's best from a previous, possibly
            // shallower, search) is ordered first - the strongest hint there is;
            // then captures, then this ply's killers, then the rest.
            moves.sort_by_key(|&m| {
                -order_score(board, m, tt_move, killers, &self.history, &self.tun)
            });
        }
        for mv in &moves {
            // Late move pruning. At shallow non-PV nodes, once enough legal moves
            // have been searched the remaining quiets (ordered worst) are so
            // unlikely to matter that they are skipped without being made at all.
            // Captures and promotions are exempt (handled per move), check
            // evasions are never pruned (not in check), and the mate guard keeps
            // us searching for an escape when the best score so far is a loss.
            if !in_check
                && depth <= LMP_MAX_DEPTH
                && beta - alpha == 1
                && legal >= self.tun.lmp_threshold(depth)
                && best > -MATE + MAX_PLY as Score
                && !mv.flag().is_capture()
                && !mv.flag().is_promotion()
            {
                continue;
            }
            let undo = board.make_move(*mv);
            if board.is_legal() {
                legal += 1;
                let full_depth = depth - 1;
                // Principal variation search. The first (best-ordered) move is
                // searched with the full window for an exact PV score; every later
                // move is scouted with a null window that only has to prove it fails
                // low (cheaper, since a 1-wide window cuts off far more often).
                let score = if legal == 1 {
                    -self.negamax(board, full_depth, ply + 1, -beta, -alpha, true)
                } else {
                    // Late move reductions. Past the first few well-ordered moves a
                    // quiet non-killer rarely is best, so scout it shallower.
                    let reduce = depth >= LMR_MIN_DEPTH
                        && legal > LMR_MIN_MOVES
                        && !mv.flag().is_capture()
                        && !mv.flag().is_promotion()
                        && Some(*mv) != killers[0]
                        && Some(*mv) != killers[1]
                        && !in_check;
                    let scout_depth = if reduce {
                        // Reduce by the table amount but keep at least one ply of
                        // real search, so the scout never collapses straight to
                        // quiescence.
                        full_depth
                            .saturating_sub(self.tun.lmr_reduction(depth, legal))
                            .max(1)
                    } else {
                        full_depth
                    };
                    let mut s =
                        -self.negamax(board, scout_depth, ply + 1, -alpha - 1, -alpha, true);
                    // A reduced scout that beat alpha may have under-searched: verify
                    // it at full depth (still the null window) before trusting it.
                    if reduce && s > alpha {
                        s = -self.negamax(board, full_depth, ply + 1, -alpha - 1, -alpha, true);
                    }
                    // The null-window scout can only fail high or low. If it landed
                    // above alpha and inside the real window the move may be a new PV,
                    // so re-search wide for its exact value. At null-window nodes
                    // beta == alpha + 1, so this never fires - PVS only ever
                    // re-searches along the principal variation.
                    if s > alpha && s < beta {
                        s = -self.negamax(board, full_depth, ply + 1, -beta, -alpha, true);
                    }
                    s
                };
                if score > best {
                    best = score;
                    best_move = Some(*mv);
                }
                if score >= beta {
                    board.unmake_move(*mv, undo);
                    // A quiet move that fails high becomes a killer and earns
                    // history credit; captures are excluded (MVV-LVA orders them).
                    // After the unmake, `side_to_move` is the mover again.
                    if !mv.flag().is_capture() && !mv.flag().is_promotion() {
                        self.store_killer(ply, *mv);
                        self.update_history(board.side_to_move(), *mv, depth);
                    }
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
        hash: crate::ZobristHash,
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

    /// Record a quiet cutoff move as a killer for `ply`, most-recent in slot 0.
    /// A move already in slot 0 is a no-op, so it can't evict slot 1 with a copy.
    fn store_killer(&mut self, ply: u32, mv: Move) {
        let slot = &mut self.killers[ply as usize];
        if slot[0] != Some(mv) {
            slot[1] = slot[0];
            slot[0] = Some(mv);
        }
    }

    /// Credit a quiet cutoff move in the butterfly history. The bonus scales with
    /// `depth^2` (a cutoff found deeper in the tree was more expensive to
    /// discover, so it weighs more), saturating at [`Tunables::history_max`].
    /// `side` is the move's mover.
    fn update_history(&mut self, side: crate::Color, mv: Move, depth: u32) {
        let bonus = self.tun.history_bonus(depth);
        let cap = self.tun.history_max;
        let cell = &mut self.history[side.as_u8() as usize][mv.from().index() as usize]
            [mv.to().index() as usize];
        *cell = (*cell + bonus).min(cap);
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
        // Quiescence searches only captures/promotions, so killers (quiet by
        // definition) and history never apply here - pass empty slots.
        moves
            .sort_by_key(|&m| -order_score(board, m, None, [None, None], &self.history, &self.tun));
        // When in check, every capture is a candidate escape, so SEE pruning is
        // suppressed (we may *have* to make a losing capture). Probed once per
        // node rather than per move.
        let in_check = board.in_check(board.side_to_move());
        for mv in &moves {
            if !(mv.flag().is_capture() || mv.flag().is_promotion()) {
                continue; // quiet move: not searched in quiescence
            }
            // SEE pruning: a capture that loses material by static exchange is
            // almost never the best move, so skip it and its recapture subtree.
            // Promotions are exempt - SEE scores only the captured piece, missing
            // the queening gain - and so are all captures while in check.
            if !in_check && !mv.flag().is_promotion() && board.see(*mv) < 0 {
                continue;
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

const VAL: [i32; 6] = [100, 320, 330, 500, 900, 0];

/// Maximum remaining depth at which reverse futility pruning is attempted.
///
/// # Notes
/// RFP trusts the static eval in place of a search, so it is confined to the
/// shallow frontier where the eval is closest to the truth. Deeper than this
/// the opponent has too much room to recover for a static cutoff to be safe.
const RFP_MAX_DEPTH: u32 = 6;

/// Maximum remaining depth at which late move pruning is attempted.
///
/// # Notes
/// Confined to the shallow frontier: deeper than this, skipping quiets unsearched
/// risks missing a move whose merit only shows below the horizon.
const LMP_MAX_DEPTH: u32 = 4;

/// Minimum remaining depth to attempt null-move pruning.
///
/// # Notes
/// Below this the reduced search (`depth - 1 - NMP_R`) is too shallow to trust.
/// With `NMP_R = 2` the null move bottoms out at depth >= 0 (quiescence).
const NMP_MIN_DEPTH: u32 = 3;

/// Null-move depth reduction.
///
/// # Notes
/// The free move is searched `1 + NMP_R` plies shallower - deep enough to expose
/// a refutation, cheap enough to be worth it. `R = 2` is the conservative classic.
const NMP_R: u32 = 2;

/// Minimum remaining depth to attempt a late-move reduction.
///
/// # Notes
/// Below this there is too little depth to give back, and the risk of
/// under-searching a good move outweighs the saving.
const LMR_MIN_DEPTH: u32 = 3;

/// Number of leading (well-ordered) legal moves searched at full depth before
/// reductions start.
///
/// # Notes
/// The TT move, captures, and killers occupy this prefix; the 4th legal move
/// onward is eligible for reduction.
const LMR_MIN_MOVES: u32 = 3;

/// Side length of the LMR table ([`Tunables::lmr_reduction`]) - depth and move
/// number are each clamped to this before indexing.
const LMR_DIM: usize = MAX_PLY as usize;

/// Whether `side` has any piece beyond pawns and the king. Null-move pruning is
/// gated on this because zugzwang - a free "pass" being misleadingly good -
/// essentially only occurs in pawn-and-king endgames.
fn has_non_pawn_material(board: &Board, side: crate::Color) -> bool {
    use PieceType::{Bishop, Knight, Queen, Rook};
    !(board.pieces(side, Knight)
        | board.pieces(side, Bishop)
        | board.pieces(side, Rook)
        | board.pieces(side, Queen))
    .is_empty()
}

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

/// Butterfly history table: a running `depth^2` count per `[side][from][to]` of
/// quiet beta-cutoff moves, used to order non-killer quiets (see [`order_score`]).
///
/// # Notes
/// Default tunables hold the saturation cap ([`Tunables::history_max`]) one band
/// below the killer bonuses, so a history-ranked quiet never outranks a killer;
/// SPSA may move that boundary.
type HistoryTable = [[[i32; 64]; 64]; 2];

fn order_score(
    board: &Board,
    mv: Move,
    hint: Option<Move>,
    killers: [Option<Move>; 2],
    history: &HistoryTable,
    tun: &Tunables,
) -> i32 {
    if Some(mv) == hint {
        return PV_BONUS; // last iteration's best move: searched first
    }
    if !mv.flag().is_capture() {
        // Quiet move: a killer for this ply jumps ahead of the other quiets;
        // the rest are ranked by history (0 if never a cutoff). With default
        // tunables history is capped below the killer band, so the tiers never
        // cross.
        if Some(mv) == killers[0] {
            return tun.killer1_bonus;
        }
        if Some(mv) == killers[1] {
            return tun.killer2_bonus;
        }
        let side = board.side_to_move().as_u8() as usize;
        return history[side][mv.from().index() as usize][mv.to().index() as usize];
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
    use crate::Square;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen.as_bytes()).unwrap()
    }

    fn sq(s: &str) -> Square {
        Square::from_ascii(s.as_bytes()).unwrap()
    }

    /// Run a search with a throwaway 1 MB table - these tests probe behaviour,
    /// not cross-move reuse.
    fn go(b: &mut Board, limits: &Limits) -> SearchResult {
        search(
            b,
            limits,
            &mut TranspositionTable::new(1),
            &Tunables::default(),
        )
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
            search(
                &mut b,
                &limits,
                &mut TranspositionTable::new(1),
                &Tunables::default(),
            )
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
        let r = go(&mut b, &Limits::to_depth(2));
        assert_eq!(
            r.best_move.map(|m| (m.from(), m.to())),
            Some((sq("a1"), sq("a8")))
        );
        assert_eq!(r.score, MATE - 1); // mate delivered one ply from the root
    }

    #[test]
    fn nmp_preserves_a_mate_at_active_depth() {
        // The same back-rank mate, but searched to depth 5 - deep enough that
        // null-move pruning is active in the tree (NMP_MIN_DEPTH = 3). NMP must
        // not prune the mating move, and the mate-score cap on a null cutoff must
        // not corrupt the reported mate (still MATE - 1, one ply from the root).
        let mut b = board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        let r = go(&mut b, &Limits::to_depth(5));
        assert_eq!(
            r.best_move.map(|m| (m.from(), m.to())),
            Some((sq("a1"), sq("a8")))
        );
        assert_eq!(r.score, MATE - 1);
    }

    #[test]
    fn lmr_preserves_a_mate_at_active_depth() {
        // The back-rank mate searched to depth 6 - deep enough that late-move
        // reductions are active (LMR_MIN_DEPTH = 3). Even if the mating move is
        // sorted late and gets reduced, its reduced search returns a mate score,
        // which beats alpha and forces the full-depth re-search - so LMR must
        // still surface Ra8 with the correct mate distance, not drop it.
        let mut b = board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        let r = go(&mut b, &Limits::to_depth(6));
        assert_eq!(
            r.best_move.map(|m| (m.from(), m.to())),
            Some((sq("a1"), sq("a8")))
        );
        assert_eq!(r.score, MATE - 1);
    }

    #[test]
    fn pvs_resurfaces_a_late_winning_move() {
        // Ra8 is a quiet rook move that mates, but MVV-LVA orders captures and
        // other quiets ahead of it, so PVS scouts it with a null window first. The
        // scout fails high (it is a mate), which must trigger the full-window
        // re-search that recovers the exact mate distance. A wrong scout bound
        // would either drop the move or report an inexact score.
        let mut b = board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        let r = go(&mut b, &Limits::to_depth(4));
        assert_eq!(
            r.best_move.map(|m| (m.from(), m.to())),
            Some((sq("a1"), sq("a8")))
        );
        assert_eq!(r.score, MATE - 1);
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
        let r = go(&mut b, &Limits::to_nodes(5_000));
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
        // White Pe4 has a capture (e4xd5), several quiets (e4e5, king moves).
        // Make one quiet the slot-0 killer and assert the ordering contract:
        // capture > killer > ordinary quiet.
        let b = board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        let mut moves = MoveList::new();
        b.generate_moves(&mut moves);

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
        let tun = Tunables::default();

        let cap_score = order_score(&b, capture, None, killers, &empty, &tun);
        let killer_score = order_score(&b, killer, None, killers, &empty, &tun);
        let other_score = order_score(&b, other, None, killers, &empty, &tun);

        assert!(cap_score > killer_score, "captures outrank killers");
        assert!(killer_score > other_score, "killer outranks plain quiets");
        assert_eq!(other_score, 0, "a non-killer quiet scores 0");
    }

    #[test]
    fn history_ranks_quiets_below_killers_and_above_cold_quiets() {
        // Same position. A quiet with history credit sorts above a never-tried
        // quiet (0) but still below a killer and a capture - the tier order is
        // PV > capture > killer > history-quiet > cold-quiet.
        let b = board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        let mut moves = MoveList::new();
        b.generate_moves(&mut moves);

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
        let tun = Tunables::default();

        let cap_s = order_score(&b, capture, None, killers, &history, &tun);
        let killer_s = order_score(&b, killer, None, killers, &history, &tun);
        let hot_s = order_score(&b, hot, None, killers, &history, &tun);
        let cold_s = order_score(&b, cold, None, killers, &history, &tun);

        assert_eq!(hot_s, 500, "history quiet carries its score");
        assert!(cap_s > killer_s, "capture > killer");
        assert!(killer_s > hot_s, "killer > history quiet");
        assert!(hot_s > cold_s, "history quiet > cold quiet");
        assert_eq!(cold_s, 0, "a cold quiet scores 0");
    }
}
