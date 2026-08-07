//! Iterative-deepening negamax search.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::eval::{TEMPO, evaluate};
use crate::movegen::{
    MoveList, attackers_to, generate_captures, generate_legal_in_check, generate_pseudo,
    generate_quiets, is_attacked,
};
#[cfg(feature = "profiling")]
use crate::tt::Miss;
use crate::tt::{Bound, TranspositionTable};
use crate::{Bitboard, Board, Color, Move, MoveType, PieceType, Square};

/// Nodes between clock checks. Low enough that a small tree still checks
/// several times: depth 2 from the start position is only 440 nodes, and at
/// 2048 it never checked at all.
const CHECK_INTERVAL: u64 = 256;
const MATE: i32 = 30_000;
const INFINITY: i32 = 31_000;
/// A drawn position, from the point of view of the side to move.
const DRAW: i32 = 0;
/// Halfmove clock at which the fifty-move rule applies: 50 full moves each.
const FIFTY_MOVE_PLIES: u8 = 100;
/// Scores above this in absolute value encode a distance to mate.
const MATE_BOUND: i32 = MATE - 1000;
/// Quiescence ply ceiling. Deep enough for any real exchange sequence, and a
/// hard bound keeps the bench node count finite and deterministic.
const MAX_QPLY: u32 = 8;
/// Null-move floor. Below this the null search reduces to a bare quiescence
/// verdict on a position that cannot legally arise, which is too thin to trust.
/// Also the depth at which `depth - 1 - R` bottoms out at exactly 0.
const MIN_NULL_DEPTH: u32 = 3;
/// Per-ply reverse futility margin, in centipawns. Read as: one ply is assumed
/// to be worth at most this much to the side to move.
///
/// Wide, because this evaluation is material and placement only and has a
/// measured positional blind spot: a margin that merely bounds real play would
/// here prune against a known eval error. Swept against the wac suite, where
/// the solve count falls off a cliff below 300 - 40/60/80/120/200 solve
/// 256/258/262/266/275 of 300 against main's 280 - while 300 and above all
/// hold 280. Worth re-sweeping downward once the evaluation gains mobility
/// and pawn structure and the blind spot shrinks.
const RFP_MARGIN: i32 = 300;
/// Half-width of the aspiration window, in centipawns. Narrow enough that most
/// nodes inside it fail high or low rather than computing an exact score, wide
/// enough that an ordinary iteration-to-iteration drift stays inside it.
///
/// Swept over 8/16/25/30/35/40/45/50/60/80 against the bench: flat between 40
/// and 60, rising sharply below 40 as re-searches start to dominate. 45 is the
/// measured minimum inside that plateau rather than a distinct optimum.
const ASPIRATION_DELTA: i32 = 45;
/// Depth at which aspiration starts. Below it an iteration is cheap enough
/// that a re-search costs more than the narrow window saves, and the score is
/// still moving too much between iterations for the last one to predict it.
const ASPIRATION_MIN_DEPTH: u32 = 4;
const LMR_MIN_DEPTH: u32 = 3;
const LMR_MIN_INDEX: usize = 4;
/// Scale on the `ln(depth) * ln(move number)` reduction, and the constant
/// subtracted from it. See [`LMR_TABLE`].
const LMR_SCALE: f64 = 0.40;
const LMR_BASE: f64 = 0.10;
/// Reduction depth by `[depth][move number]`, both clamped to the table.
///
/// A flat reduction is the wrong shape: it says a late move at depth 4 and a
/// late move at depth 20 deserve the same treatment, when the second is far
/// more likely to be refuted cheaply and far more expensive to search in full.
/// Growing the reduction with both terms is what bends the effective branching
/// factor down as depth rises, rather than shifting it by a constant.
///
/// Logarithms because both terms have diminishing returns: the 30th move is
/// barely less promising than the 20th, while the 5th is much less promising
/// than the 3rd. Tabulated at startup so the search does no float work.
static LMR_TABLE: LazyLock<[[u8; 64]; 64]> = LazyLock::new(|| {
    let mut table = [[0u8; 64]; 64];
    for (depth, row) in table.iter_mut().enumerate().skip(1) {
        for (index, slot) in row.iter_mut().enumerate().skip(1) {
            let reduction = LMR_BASE + (depth as f64).ln() * (index as f64).ln() * LMR_SCALE;
            *slot = reduction.max(0.0) as u8;
        }
    }
    table
});
/// Killer ply ceiling. Search depth is bounded by the clock long before this;
/// past it the killer slots are simply skipped.
const MAX_PLY: usize = 64;

/// Cutoff counts per side and from/to square, the ordering score for quiets
/// that are neither captures nor killers.
type History = [[[i32; 64]; 64]; 2];

/// Scratch keys for one `order_moves` call, one slot per possible move.
///
/// Lives in `Ctx` rather than on the stack of `order_moves`: at 2KiB it was
/// zeroed at every node, and a node sorts about 35 moves, so all but the first
/// 35 slots were written and discarded untouched. `order_moves` does not
/// recurse - it sorts and returns before the caller plays anything - so a
/// single buffer serves the whole search.
type OrderKeys = [(i32, i32); MoveList::CAPACITY];

/// Constraints on a single search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Limits {
    /// Stop after completing this depth.
    pub depth: Option<u32>,
    /// Stop once this many nodes have been searched.
    pub nodes: Option<u64>,
    /// Fixed time for this move, in milliseconds.
    pub movetime: Option<u64>,
    /// White's remaining clock, in milliseconds.
    pub wtime: Option<u64>,
    /// Black's remaining clock, in milliseconds.
    pub btime: Option<u64>,
    /// White's increment, in milliseconds.
    pub winc: u64,
    /// Black's increment, in milliseconds.
    pub binc: u64,
    /// Search until told to stop.
    pub infinite: bool,
}

/// What one iterative-deepening iteration found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Iteration {
    /// Depth this iteration completed.
    pub depth: u32,
    /// Score from the side to move's point of view, in centipawns, unless
    /// `mate_in` reads it as a distance to mate.
    pub score: i32,
    /// Nodes searched so far in this search, main plus quiescence.
    pub nodes: u64,
    /// Time since the search started.
    pub elapsed: Duration,
    /// Best move as of this iteration, absent only if the position has none.
    pub best_move: Option<Move>,
}

impl Iteration {
    /// Moves to mate if the score encodes one, negative when getting mated.
    pub fn mate_in(&self) -> Option<i32> {
        mate_in(self.score)
    }

    /// Nodes per second, averaged over the search so far.
    pub fn nps(&self) -> u128 {
        u128::from(self.nodes) * 1000 / self.elapsed.as_millis().max(1)
    }
}

/// Observes a search as it runs.
///
/// The search reports what it found; rendering it is the caller's business.
/// That keeps UCI text out of the library and lets a caller that wants the
/// data rather than the protocol - the tactics runner reading which iteration
/// first found the expected move - take it without parsing anything.
pub trait SearchListener {
    /// Called once per completed iterative-deepening iteration.
    fn iteration(&mut self, _iteration: &Iteration) {}
    /// Called once when the search stops, with its final answer.
    fn finished(&mut self, _best_move: Option<Move>) {}
}

/// Discards everything, for searches whose result is read from the return
/// value alone.
impl SearchListener for () {}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchResult {
    pub(crate) best_move: Option<Move>,
    /// Main-search nodes, excluding quiescence.
    pub(crate) nodes: u64,
    /// Quiescence nodes.
    pub(crate) qnodes: u64,
    #[cfg(feature = "profiling")]
    pub(crate) profile: SearchProfile,
}

#[cfg(feature = "profiling")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SearchProfile {
    pub(crate) main_cutoffs: [u64; 5],
    pub(crate) q_cutoffs: [u64; 5],
    pub(crate) main_bounds: [u64; 3],
    pub(crate) q_bounds: [u64; 3],
    pub(crate) tt_probes: u64,
    pub(crate) tt_hits: u64,
    /// Misses where the slot held a different position, so this one evicts it.
    /// The rest of the misses found an empty slot.
    pub(crate) tt_collisions: u64,
    pub(crate) tt_cutoffs: u64,
    pub(crate) tt_stores: u64,
    pub(crate) qply: [u64; MAX_QPLY as usize + 1],
    pub(crate) q_in_check: u64,
    pub(crate) stand_pat_nodes: u64,
    pub(crate) stand_pat_cutoffs: u64,
    pub(crate) pvs_probes: u64,
    pub(crate) pvs_researches: u64,
    pub(crate) rfp_cutoffs: u64,
    /// Checking moves searched at undiminished depth.
    pub(crate) check_extensions: u64,
    pub(crate) null_attempts: u64,
    pub(crate) null_cutoffs: u64,
    pub(crate) lmr_reductions: u64,
    pub(crate) lmr_researches: u64,
}

#[cfg(feature = "profiling")]
impl std::ops::AddAssign for SearchProfile {
    fn add_assign(&mut self, rhs: Self) {
        for (left, right) in self.main_cutoffs.iter_mut().zip(rhs.main_cutoffs) {
            *left += right;
        }
        for (left, right) in self.q_cutoffs.iter_mut().zip(rhs.q_cutoffs) {
            *left += right;
        }
        for (left, right) in self.main_bounds.iter_mut().zip(rhs.main_bounds) {
            *left += right;
        }
        for (left, right) in self.q_bounds.iter_mut().zip(rhs.q_bounds) {
            *left += right;
        }
        for (left, right) in self.qply.iter_mut().zip(rhs.qply) {
            *left += right;
        }
        self.tt_probes += rhs.tt_probes;
        self.tt_hits += rhs.tt_hits;
        self.tt_collisions += rhs.tt_collisions;
        self.tt_cutoffs += rhs.tt_cutoffs;
        self.tt_stores += rhs.tt_stores;
        self.q_in_check += rhs.q_in_check;
        self.stand_pat_nodes += rhs.stand_pat_nodes;
        self.stand_pat_cutoffs += rhs.stand_pat_cutoffs;
        self.pvs_probes += rhs.pvs_probes;
        self.pvs_researches += rhs.pvs_researches;
        self.rfp_cutoffs += rhs.rfp_cutoffs;
        self.check_extensions += rhs.check_extensions;
        self.null_attempts += rhs.null_attempts;
        self.null_cutoffs += rhs.null_cutoffs;
        self.lmr_reductions += rhs.lmr_reductions;
        self.lmr_researches += rhs.lmr_researches;
    }
}

#[cfg(feature = "profiling")]
impl SearchProfile {
    fn record_cutoff(&mut self, quiescence: bool, move_number: usize) {
        let bucket = match move_number {
            1 => 0,
            2 => 1,
            3 => 2,
            4..=8 => 3,
            _ => 4,
        };
        if quiescence {
            self.q_cutoffs[bucket] += 1;
        } else {
            self.main_cutoffs[bucket] += 1;
        }
    }

    fn record_bound(&mut self, quiescence: bool, bound: Bound) {
        let index = match bound {
            Bound::Lower => 0,
            Bound::Upper => 1,
            Bound::Exact => 2,
        };
        if quiescence {
            self.q_bounds[index] += 1;
        } else {
            self.main_bounds[index] += 1;
        }
    }

    fn record_main_bound(&mut self, score: i32, alpha: i32, beta: i32) {
        self.record_bound(false, score_bound(score, alpha, beta));
    }

    fn record_q_bound(&mut self, score: i32, alpha: i32, beta: i32) {
        self.record_bound(true, score_bound(score, alpha, beta));
    }

    fn record_main_exact(&mut self) {
        self.record_bound(false, Bound::Exact);
    }

    fn record_q_exact(&mut self) {
        self.record_bound(true, Bound::Exact);
    }

    fn record_tt_probe(&mut self, miss: Miss) {
        self.tt_probes += 1;
        match miss {
            Miss::Hit => self.tt_hits += 1,
            Miss::Collision => self.tt_collisions += 1,
            Miss::Empty => {}
        }
    }

    fn record_tt_cutoff(&mut self, bound: Bound) {
        self.tt_cutoffs += 1;
        self.record_bound(false, bound);
    }

    fn record_tt_store(&mut self) {
        self.tt_stores += 1;
    }

    fn record_qply(&mut self, qply: u32) {
        self.qply[qply.min(MAX_QPLY) as usize] += 1;
    }

    fn record_q_in_check(&mut self, in_check: bool) {
        self.q_in_check += u64::from(in_check);
    }

    fn record_stand_pat_node(&mut self) {
        self.stand_pat_nodes += 1;
    }

    fn record_stand_pat_cutoff(&mut self) {
        self.stand_pat_cutoffs += 1;
        self.record_bound(true, Bound::Lower);
    }

    fn record_pvs_probe(&mut self) {
        self.pvs_probes += 1;
    }

    fn record_pvs_research(&mut self) {
        self.pvs_researches += 1;
    }

    fn record_rfp_cutoff(&mut self) {
        self.rfp_cutoffs += 1;
    }

    fn record_check_extension(&mut self) {
        self.check_extensions += 1;
    }

    fn record_null_attempt(&mut self) {
        self.null_attempts += 1;
    }

    fn record_null_cutoff(&mut self) {
        self.null_cutoffs += 1;
    }

    fn record_lmr_reduction(&mut self) {
        self.lmr_reductions += 1;
    }

    fn record_lmr_research(&mut self) {
        self.lmr_researches += 1;
    }
}

#[cfg(feature = "profiling")]
fn score_bound(score: i32, alpha: i32, beta: i32) -> Bound {
    if score >= beta {
        Bound::Lower
    } else if score <= alpha {
        Bound::Upper
    } else {
        Bound::Exact
    }
}

struct Ctx<'a> {
    limits: Limits,
    stop: &'a AtomicBool,
    deadline: Option<Instant>,
    nodes: u64,
    qnodes: u64,
    iteration_depth: u32,
    /// Two quiet moves per ply that last caused a beta cutoff there.
    killers: [[Option<Move>; 2]; MAX_PLY],
    /// Cutoff counts per side and from/to square.
    history: History,
    /// Reused across nodes; see [`OrderKeys`].
    order_keys: OrderKeys,
    /// Shared across searches, so one move's tree seeds the next.
    tt: &'a TranspositionTable,
    #[cfg(feature = "profiling")]
    profile: SearchProfile,
    /// Whether leaves run quiescence. Always true in play; the alpha-beta
    /// equivalence tests turn it off so their unpruned oracle stays cheap.
    #[cfg(test)]
    quiesce_leaves: bool,
    #[cfg(test)]
    pvs: bool,
    #[cfg(test)]
    aspiration: bool,
    #[cfg(test)]
    nmp: bool,
    #[cfg(test)]
    rfp: bool,
    #[cfg(test)]
    check_extension: bool,
    #[cfg(test)]
    lmr: bool,
    #[cfg(test)]
    lmr_reductions: u64,
}

#[derive(Debug, Clone, Copy)]
struct Aborted;

/// Searches the position and returns the best move found, reporting each
/// iteration to `listener`.
pub fn search(
    board: &mut Board,
    limits: Limits,
    stop: &AtomicBool,
    tt: &TranspositionTable,
    listener: &mut dyn SearchListener,
) -> Option<Move> {
    search_inner(board, limits, stop, tt, listener).best_move
}

pub(crate) fn search_inner(
    board: &mut Board,
    limits: Limits,
    stop: &AtomicBool,
    tt: &TranspositionTable,
    listener: &mut dyn SearchListener,
) -> SearchResult {
    let start = Instant::now();
    let deadline =
        time_budget(board, limits).and_then(|ms| start.checked_add(Duration::from_millis(ms)));
    tt.new_search();
    let mut ctx = Ctx {
        limits,
        stop,
        deadline,
        nodes: 0,
        qnodes: 0,
        iteration_depth: 1,
        killers: [[None; 2]; MAX_PLY],
        history: [[[0; 64]; 64]; 2],
        order_keys: [(0, 0); MoveList::CAPACITY],
        tt,
        #[cfg(feature = "profiling")]
        profile: SearchProfile::default(),
        #[cfg(test)]
        quiesce_leaves: true,
        #[cfg(test)]
        pvs: true,
        #[cfg(test)]
        aspiration: true,
        #[cfg(test)]
        nmp: true,
        #[cfg(test)]
        rfp: true,
        #[cfg(test)]
        check_extension: true,
        #[cfg(test)]
        lmr: true,
        #[cfg(test)]
        lmr_reductions: 0,
    };
    let max_depth = limits.depth.unwrap_or(u32::MAX).max(1);
    let mut best_move = None;
    let mut prev_score = None;

    for depth in 1..=max_depth {
        // An unpruned tree grows about 35x per depth, so starting an iteration
        // with the budget already spent overshoots it by that factor.
        if depth > 1 && ctx.out_of_time() {
            break;
        }
        ctx.iteration_depth = depth;
        let result = aspirate(board, depth, prev_score, &mut ctx);
        let Ok((candidate, score)) = result else {
            break;
        };
        prev_score = Some(score);
        best_move = candidate;
        listener.iteration(&Iteration {
            depth,
            score,
            nodes: ctx.total(),
            elapsed: start.elapsed(),
            best_move: candidate,
        });
        if candidate.is_none() {
            break;
        }
    }
    listener.finished(best_move);

    SearchResult {
        best_move,
        nodes: ctx.nodes,
        qnodes: ctx.qnodes,
        #[cfg(feature = "profiling")]
        profile: ctx.profile,
    }
}

fn time_budget(board: &Board, limits: Limits) -> Option<u64> {
    if limits.infinite {
        return None;
    }
    if let Some(movetime) = limits.movetime {
        return Some(movetime.max(1));
    }
    let (remaining, increment) = match board.state().side_to_move() {
        crate::Color::White => (limits.wtime?, limits.winc),
        crate::Color::Black => (limits.btime?, limits.binc),
    };
    let budget = remaining / 20 + increment / 2;
    Some(budget.min(remaining.saturating_sub(50)).max(1))
}

/// Searches one iteration, narrowly around the last one's score where that is
/// worth trying, and at full width otherwise.
///
/// A narrow window makes every node cheaper, because more of them fail high or
/// low without an exact score being computed. The bet is that the next
/// iteration lands near the last, which holds often enough to pay for the
/// re-searches when it does not.
///
/// Widening straight to infinity rather than in steps keeps the returned score
/// exact at every depth. The driver assigns `best_move` unconditionally, and an
/// iteration abandoned partway is discarded whole, so a merely-bounded score
/// must never reach it.
///
/// Skipped below `ASPIRATION_MIN_DEPTH`, where iterations are too cheap for the
/// re-search to pay for itself and scores are still moving, and skipped for
/// mate scores, where the next iteration's score is not near the last one but a
/// mate distance away from it.
fn aspirate(
    board: &mut Board,
    depth: u32,
    prev_score: Option<i32>,
    ctx: &mut Ctx<'_>,
) -> Result<(Option<Move>, i32), Aborted> {
    let narrow = prev_score.filter(|score| {
        ctx.aspiration_enabled() && depth >= ASPIRATION_MIN_DEPTH && score.abs() <= MATE_BOUND
    });
    if let Some(score) = narrow {
        let (alpha, beta) = (
            score.saturating_sub(ASPIRATION_DELTA),
            score.saturating_add(ASPIRATION_DELTA),
        );
        let (best_move, score) = negamax_root(board, depth, alpha, beta, ctx)?;
        // Strictly inside: a score sitting on either bound was only proved to
        // reach it, not to be it.
        if score > alpha && score < beta {
            return Ok((best_move, score));
        }
    }
    negamax_root(board, depth, -INFINITY, INFINITY, ctx)
}

fn negamax_root(
    board: &mut Board,
    depth: u32,
    alpha: i32,
    beta: i32,
    ctx: &mut Ctx<'_>,
) -> Result<(Option<Move>, i32), Aborted> {
    let mut moves = MoveList::new();
    generate_pseudo(board, &mut moves);
    let key = board.state().zobrist();
    // Ordering only: the root must return a move, so it never cuts on a
    // stored score however deep that score was searched.
    #[cfg(not(feature = "profiling"))]
    let entry = ctx.tt.probe(key);
    #[cfg(feature = "profiling")]
    let entry = {
        let (entry, miss) = ctx.tt.probe_kind(key);
        ctx.profile.record_tt_probe(miss);
        entry
    };
    let tt_move = entry.and_then(|entry| entry.best_move);
    let killers = ctx.killers_at(0);
    order_moves(
        board,
        moves.as_mut_slice(),
        tt_move,
        killers,
        &ctx.history,
        &mut ctx.order_keys,
    );
    let us = board.state().side_to_move();
    let mut best_move = None;
    let mut best_score = -INFINITY;
    let mut legal = 0;
    for &mv in moves.iter() {
        board.make(mv);
        if !is_legal(board, us) {
            board.unmake(mv);
            continue;
        }
        // `alpha` is the floor the caller set; `best_score` raises it as moves
        // come in. Under a full window the two coincide and this reduces to
        // what the root did before aspiration existed.
        let cut = best_score.max(alpha);
        let result = search_move(board, depth, 1, cut, beta, legal, false, ctx);
        board.unmake(mv);
        legal += 1;
        let score = result?;
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
    }
    if legal == 0 {
        return Ok((None, terminal_score(board, 0)));
    }
    // The unconditional `Exact` this used to store was justified by `alpha`
    // being `-INFINITY`, so that every score which replaced `best_score` came
    // from an exact-window search. Aspiration breaks that: a search that fails
    // low proved only a ceiling and one that fails high only a floor, and
    // storing either as `Exact` would hand a later probe a number the search
    // never established.
    let bound = if best_score <= alpha {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    #[cfg(feature = "profiling")]
    ctx.profile.record_tt_store();
    ctx.tt
        .store(key, score_to_tt(best_score, 0), best_move, depth, bound);
    Ok((best_move, best_score))
}

/// Searches one already-made child and returns its score from the parent's
/// point of view.
///
/// The first move gets the full window; the rest are probed on a null window
/// first, on the assumption that ordering put the best move first and the
/// others only need refuting. A probe that beats `alpha` is re-searched
/// properly, because a null window can prove a move is better but not by how
/// much. `child_ply` is the ply of the position now on the board.
#[allow(clippy::too_many_arguments)]
fn search_move(
    board: &mut Board,
    depth: u32,
    child_ply: u32,
    alpha: i32,
    beta: i32,
    index: usize,
    reduce: bool,
    ctx: &mut Ctx<'_>,
) -> Result<i32, Aborted> {
    if index > 0 && ctx.pvs_enabled() {
        let lmr_reduced = reduce && ctx.lmr_enabled();
        if lmr_reduced {
            #[cfg(feature = "profiling")]
            ctx.profile.record_lmr_reduction();
            #[cfg(test)]
            {
                ctx.lmr_reductions += 1;
            }
            let reduced = -negamax(
                board,
                depth.saturating_sub(1 + lmr_reduction(depth, index)),
                child_ply,
                -alpha - 1,
                -alpha,
                true,
                ctx,
            )?;
            if reduced <= alpha {
                return Ok(reduced);
            }
            #[cfg(feature = "profiling")]
            ctx.profile.record_lmr_research();
        }
        #[cfg(feature = "profiling")]
        ctx.profile.record_pvs_probe();
        let narrow = -negamax(board, depth - 1, child_ply, -alpha - 1, -alpha, true, ctx)?;
        if !(narrow > alpha && narrow < beta) {
            return Ok(narrow);
        }
        #[cfg(feature = "profiling")]
        ctx.profile.record_pvs_research();
    }
    negamax(board, depth - 1, child_ply, -beta, -alpha, true, ctx).map(|score| -score)
}

fn negamax(
    board: &mut Board,
    depth: u32,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    can_null: bool,
    ctx: &mut Ctx<'_>,
) -> Result<i32, Aborted> {
    ctx.nodes += 1;
    ctx.check_abort()?;
    // Before the TT probe, and never stored: a draw by repetition or by the
    // clock is a property of the path taken to this position, not of the
    // position, so a score of 0 here would be wrong for another path that
    // reaches the same key. `ply > 0` because the root must return a move.
    if ply > 0 && is_draw(board) {
        return Ok(DRAW);
    }
    if depth == 0 {
        #[cfg(test)]
        if !ctx.quiesce_leaves {
            return Ok(evaluate(board));
        }
        let result = qsearch(board, ply, 0, alpha, beta, ctx);
        #[cfg(feature = "profiling")]
        if let Ok(score) = result {
            ctx.profile.record_main_bound(score, alpha, beta);
        }
        return result;
    }
    let key = board.state().zobrist();
    let alpha_original = alpha;
    #[cfg(not(feature = "profiling"))]
    let entry = ctx.tt.probe(key);
    #[cfg(feature = "profiling")]
    let entry = {
        let (entry, miss) = ctx.tt.probe_kind(key);
        ctx.profile.record_tt_probe(miss);
        entry
    };
    // A cut at ply 1 would return the previous search's score for the root's
    // own move without searching, so the iteration reports a depth it never
    // reached and the score freezes until the iteration passes the stored
    // depth. Deeper plies have a real parent to return to and may cut freely.
    if let Some(entry) = entry
        && entry.depth >= depth
        && ply > 1
    {
        let score = score_from_tt(entry.score, ply);
        let usable = match entry.bound {
            Bound::Exact => true,
            Bound::Lower => score >= beta,
            Bound::Upper => score <= alpha,
        };
        if usable {
            #[cfg(feature = "profiling")]
            ctx.profile.record_tt_cutoff(entry.bound);
            return Ok(score);
        }
    }
    let us = board.state().side_to_move();
    // Computed once for both gates below. `evaluate` reads the incremental
    // accumulator, so this is a few adds rather than a board scan.
    let static_eval = evaluate(board);

    // Reverse futility pruning. A node whose static score already clears beta
    // by more than the search could plausibly claw back is assumed to fail
    // high without searching it.
    //
    // Unlike the null-move gate below, both sides of this comparison are the
    // same node's score, so the tempo bonus cancels and is not withheld.
    //
    // No depth ceiling. The usual one exists because a linear margin stops
    // bounding anything once the search can recover more than `margin * depth`,
    // but this margin is wide enough that the product outruns any static score
    // long before that: capping at 4, 5, 6, 8 or 10 plies leaves the bench at
    // 886840 in every case, so a ceiling here would be a knob that does nothing.
    //
    // The in-check guard is last because `is_attacked` generates attacks and is
    // the expensive term here, while the margin test is a subtract and a
    // compare that rejects most nodes. Testing it first cost 25% of NPS: every
    // node paid for it, and only the few that clear beta ever needed it.
    //
    // The guard is defensive rather than load-bearing at this margin:
    // instrumented over kiwipete to depth 6, no in-check node ever cleared beta
    // by 300 per ply, because being in check is exactly when the static score
    // overstates the position. It costs nothing there, and the margin is meant
    // to come down once the evaluation improves, which is when it would fire.
    if ctx.rfp_enabled()
        // A mate bound is not a material claim, so a margin in centipawns
        // cannot say anything about the distance to it.
        && beta.abs() <= MATE_BOUND
        && static_eval - RFP_MARGIN * depth as i32 >= beta
        && !is_attacked(board, board.king_square(us), us.flip())
    {
        #[cfg(feature = "profiling")]
        ctx.profile.record_rfp_cutoff();
        return Ok(static_eval);
    }

    let has_non_pawn_material =
        !(board.color(us) & !board.pieces(PieceType::Pawn) & !board.pieces(PieceType::King))
            .is_empty();
    if depth >= MIN_NULL_DEPTH
        && can_null
        && ctx.nmp_enabled()
        && has_non_pawn_material
        // The tempo bonus does not survive a null move: it is added to
        // whoever is on move, and a null changes only that, so the score this
        // gate reads and the score the null search returns are inflated
        // `2 * TEMPO` apart. Withholding it here compares like with like.
        && static_eval - TEMPO >= beta
        && !is_attacked(board, board.king_square(us), us.flip())
    {
        #[cfg(feature = "profiling")]
        ctx.profile.record_null_attempt();
        let reduction = 2 + depth / 6;
        board.make_null();
        let result = negamax(
            board,
            depth.saturating_sub(1 + reduction),
            ply + 1,
            -beta,
            -beta + 1,
            false,
            ctx,
        );
        board.unmake_null();
        let score = -result?;
        if score >= beta {
            #[cfg(feature = "profiling")]
            ctx.profile.record_null_cutoff();
            return Ok(beta);
        }
    }
    // The stored move is only a hint: a key collision or an entry from an
    // earlier game can decode to a move that is nonsense here, so it is
    // screened by `is_pseudo_legal` before being played.
    let tt_move = entry
        .and_then(|entry| entry.best_move)
        .filter(|&mv| is_pseudo_legal(board, mv));
    let mut moves = MoveList::new();
    let mut best = -INFINITY;
    let mut best_move = None;
    let mut legal = 0;
    let mut cutoff = None;
    // Staged: the TT move alone, then captures, then quiets. Ordering is
    // unchanged - the TT move already sorted ahead of everything and captures
    // ahead of the killers - but a cutoff in an earlier stage means the later
    // stages are never generated, and 89.73% of cutoffs land on move one.
    'stages: for stage in [Stage::TtMove, Stage::Captures, Stage::Quiets] {
        let start = moves.len();
        match stage {
            Stage::TtMove => match tt_move {
                Some(mv) => {
                    // `is_pseudo_legal` is a screen, not a re-derivation, so
                    // this is what catches a structurally impossible move that
                    // it admits before such a move can reach a release build.
                    debug_assert!(
                        {
                            let mut all = MoveList::new();
                            generate_pseudo(board, &mut all);
                            all.iter().any(|&generated| generated == mv)
                        },
                        "TT move {mv:?} passed is_pseudo_legal but is not generated at {board}"
                    );
                    moves.push(mv);
                }
                None => continue,
            },
            Stage::Captures => {
                generate_captures(board, &mut moves);
                order_range(board, &mut moves, start, ply, ctx);
            }
            Stage::Quiets => {
                generate_quiets(board, &mut moves);
                order_range(board, &mut moves, start, ply, ctx);
            }
        }
        for index in start..moves.len() {
            let mv = moves[index];
            // The TT move was searched in its own stage; it is regenerated here
            // as an ordinary capture or quiet.
            if stage != Stage::TtMove && Some(mv) == tt_move {
                continue;
            }
            let losing = see(board, mv) < 0;
            board.make(mv);
            if !is_legal(board, us) {
                board.unmake(mv);
                continue;
            }
            // One attack generation, read two ways: the side to move is now
            // the side this move was played against, so "is it attacked" is
            // "does this move give check". LMR declines to reduce such a move
            // and the extension below lengthens it.
            let them = board.state().side_to_move();
            let gives_check = is_attacked(board, board.king_square(them), them.flip());
            let extension = check_extension(gives_check, losing, ply, ctx);
            #[cfg(feature = "profiling")]
            if extension > 0 {
                ctx.profile.record_check_extension();
            }
            let reduce = lmr_eligible(depth, legal, mv, ply, gives_check, ctx);
            let result = search_move(
                board,
                depth + extension,
                ply + 1,
                alpha,
                beta,
                legal,
                reduce,
                ctx,
            );
            board.unmake(mv);
            legal += 1;
            let score = result?;
            if score > best {
                best = score;
                best_move = Some(mv);
            }
            alpha = alpha.max(best);
            if alpha >= beta {
                cutoff = Some(mv);
                break 'stages;
            }
        }
    }
    if let Some(mv) = cutoff {
        #[cfg(feature = "profiling")]
        ctx.profile.record_cutoff(false, legal);
        if !mv.is_capture() && !mv.is_promotion() {
            ctx.store_killer(ply, mv);
            // ponytail: no decay - history is per-`go` and dies with Ctx.
            // Add halve-on-cap or the gravity update if it goes stale
            // within one search.
            let slot =
                &mut ctx.history[us as usize][mv.from().index() as usize][mv.to().index() as usize];
            *slot = slot.saturating_add((depth * depth) as i32);
        }
    }
    // Checkmate or stalemate. Known only after the loop now that legality is
    // decided per move, and returned without a store to match what the
    // pre-filtered version did.
    if legal == 0 {
        #[cfg(feature = "profiling")]
        ctx.profile.record_main_exact();
        return Ok(terminal_score(board, ply));
    }
    let bound = if best >= beta {
        Bound::Lower
    } else if best > alpha_original {
        Bound::Exact
    } else {
        Bound::Upper
    };
    #[cfg(feature = "profiling")]
    ctx.profile.record_bound(false, bound);
    #[cfg(feature = "profiling")]
    ctx.profile.record_tt_store();
    ctx.tt
        .store(key, score_to_tt(best, ply), best_move, depth, bound);
    Ok(best)
}

/// Plies to reduce a late quiet move by, from [`LMR_TABLE`].
///
/// Bounded above by `depth - 2` so a reduced search still has at least one ply
/// of main search left: reducing straight into quiescence would answer a quiet
/// move with a capture-only verdict, which is the one thing it cannot say
/// anything about.
///
/// At the current scale the bound never binds - the formula does not exceed
/// `depth - 2` anywhere in the table - so it is a guard against a future scale
/// rather than something the search relies on today. It is kept, unlike the
/// depth ceiling a flat reduction would need, because raising the scale is an
/// expected next step and the failure it prevents is silent.
fn lmr_reduction(depth: u32, index: usize) -> u32 {
    let reduction = u32::from(LMR_TABLE[(depth as usize).min(63)][index.min(63)]);
    reduction.clamp(1, depth.saturating_sub(2).max(1))
}

fn lmr_eligible(
    depth: u32,
    legal: usize,
    mv: Move,
    ply: u32,
    gives_check: bool,
    ctx: &Ctx<'_>,
) -> bool {
    depth >= LMR_MIN_DEPTH
        && legal >= LMR_MIN_INDEX
        && !mv.is_capture()
        && !mv.is_promotion()
        && !gives_check
        && ctx.killers_at(ply).iter().all(|killer| *killer != Some(mv))
}

/// Plies to add to a checking move's search, so a forcing line is not cut off
/// mid-sequence by the horizon.
///
/// A check is the one move type whose replies are nearly forced, so the
/// subtree is narrow and the usual reason to trust the horizon - that the
/// position is quiet enough for a static score to mean something - does not
/// hold. Extending trades a small number of nodes for seeing the end of the
/// sequence.
///
/// A check that hangs the checking piece is declined: it is a tempo rather
/// than a threat, and extending it costs a ply of depth in most positions
/// while gaining nothing. `see` scores a quiet move 0, so this only ever
/// declines a capture; a quiet check is extended regardless.
///
/// The `ply` bound is a stack guard, not a termination proof. An extension
/// holds depth constant, so `depth == 0` no longer bounds the recursion the
/// way it does for every other change here - but the fifty-move rule still
/// does, since a non-repeating check sequence needs captures or pawn moves to
/// reset the clock and both are finite. What is left is a sequence that
/// terminates but far too deep, and removing the bound leaves the bench at
/// 1060717 unchanged, so it does not fire at any depth reached today. It is
/// kept because the failure it prevents is a stack overflow rather than a
/// wrong score.
fn check_extension(gives_check: bool, losing: bool, ply: u32, ctx: &Ctx<'_>) -> u32 {
    u32::from(gives_check && !losing && ply + 1 < MAX_PLY as u32 && ctx.check_extension_enabled())
}

/// Searches the capture sequences leaving a leaf position until it is quiet,
/// so the score reflects settled material rather than a position caught
/// mid-exchange.
///
/// `qply` counts plies since the leaf and is bounded by `MAX_QPLY`; `ply`
/// continues from the main search so mate scores keep their distance.
fn qsearch(
    board: &mut Board,
    ply: u32,
    qply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &mut Ctx<'_>,
) -> Result<i32, Aborted> {
    #[cfg(feature = "profiling")]
    let alpha_original = alpha;
    ctx.qnodes += 1;
    ctx.check_abort()?;
    #[cfg(feature = "profiling")]
    ctx.profile.record_qply(qply);
    if qply >= MAX_QPLY {
        let score = evaluate(board);
        #[cfg(feature = "profiling")]
        {
            let us = board.state().side_to_move();
            ctx.profile
                .record_q_in_check(is_attacked(board, board.king_square(us), us.flip()));
            ctx.profile.record_q_bound(score, alpha, beta);
        }
        return Ok(score);
    }
    let us = board.state().side_to_move();
    let in_check = is_attacked(board, board.king_square(us), us.flip());
    #[cfg(feature = "profiling")]
    ctx.profile.record_q_in_check(in_check);

    // Stand-pat first, and the list built only once this node is known to
    // need one: 78% of quiescence nodes cut here, and constructing the list
    // above the test spent 512 bytes on each of them for nothing.
    let mut best = -INFINITY;
    if !in_check {
        // Declining to capture is always an option, so the static eval is a
        // lower bound on this node.
        best = evaluate(board);
        #[cfg(feature = "profiling")]
        ctx.profile.record_stand_pat_node();
        if best >= beta {
            #[cfg(feature = "profiling")]
            ctx.profile.record_stand_pat_cutoff();
            return Ok(best);
        }
        alpha = alpha.max(best);
    }
    let mut moves = MoveList::new();
    if in_check {
        // No stand-pat in check: the side to move must answer it, and a quiet
        // king move may be the only answer, so this needs every legal move.
        // `in_check` is already known here, so it is passed rather than
        // recomputed - this is the hottest `generate_legal` caller.
        generate_legal_in_check(board, &mut moves, in_check);
        if moves.is_empty() {
            #[cfg(feature = "profiling")]
            ctx.profile.record_q_exact();
            return Ok(terminal_score(board, ply));
        }
    } else {
        generate_captures(board, &mut moves);
    }
    order_moves(
        board,
        moves.as_mut_slice(),
        None,
        [None; 2],
        &ctx.history,
        &mut ctx.order_keys,
    );

    #[cfg(feature = "profiling")]
    let mut legal_moves = 0;
    for &mv in moves.iter() {
        // Losing captures cannot raise alpha once the recapture lands, so
        // searching them only grows the tree. Promotions are exempt: the
        // swap loop values the pawn, not the piece it becomes.
        if !in_check && !mv.is_promotion() && see(board, mv) < 0 {
            continue;
        }
        board.make(mv);
        // Captures come back pseudo-legal. Filtering after `make` rather than
        // up front means a beta cutoff skips the checks it never needed.
        let legal = in_check || is_legal(board, us);
        let result = legal.then(|| qsearch(board, ply + 1, qply + 1, -beta, -alpha, ctx));
        board.unmake(mv);
        let Some(result) = result else {
            continue;
        };
        #[cfg(feature = "profiling")]
        {
            legal_moves += 1;
        }
        best = best.max(-result?);
        alpha = alpha.max(best);
        if alpha >= beta {
            #[cfg(feature = "profiling")]
            ctx.profile.record_cutoff(true, legal_moves);
            break;
        }
    }
    #[cfg(feature = "profiling")]
    ctx.profile.record_q_bound(best, alpha_original, beta);
    Ok(best)
}

/// Ordering values, deliberately separate from the eval's material values:
/// retuning the evaluation should not silently reshape the search tree. The
/// king is a victim value only, for pseudo-legal safety; it is never actually
/// captured.
const ORDER_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 10_000];

/// Below every capture key, the largest of which is a queen-capture-promotion
/// at `-(900 + 900)`.
const TT_MOVE_KEY: i32 = -100_000;

/// One step of staged generation, searched in this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// The transposition table's move, alone and generated by nothing.
    TtMove,
    /// Captures and promotion captures.
    Captures,
    /// Everything else, generated only if the captures did not cut off.
    Quiets,
}

/// Whether `mv` is safe to play on this position.
///
/// A move from the transposition table survived a 48-bit key check, not a full
/// one, so it can decode to something impossible here - and `Board::make`
/// panics when the origin square is empty. This screens for that: the mover
/// must be ours, and a capture must have something of theirs to take.
///
/// Deliberately cheap rather than a full re-derivation of the generator. A
/// structurally impossible move that slips through (a knight moving like a
/// rook) corrupts one subtree's score rather than crashing, and the debug
/// assertion in `negamax` is what stops that reaching a release build.
fn is_pseudo_legal(board: &Board, mv: Move) -> bool {
    let us = board.state().side_to_move();
    if board.piece_on(mv.from()).is_none_or(|p| p.color() != us) {
        return false;
    }
    match mv.move_type() {
        // The victim stands beside the capturer, and only if the state's en
        // passant square agrees; anything else is a stale entry.
        MoveType::EnPassant => board.state().en_passant() == Some(mv.to()),
        // Castling encodes its rook implicitly, so a stale one would index
        // `castle_rook_squares` for a rook that is not there.
        MoveType::KingCastle | MoveType::QueenCastle => false,
        _ if mv.is_capture() => board
            .piece_on(mv.to())
            .is_some_and(|piece| piece.color() != us),
        _ => board.piece_on(mv.to()).is_none(),
    }
}

/// Sorts the moves added by the current stage, leaving earlier stages alone.
fn order_range(board: &Board, moves: &mut MoveList, start: usize, ply: u32, ctx: &mut Ctx<'_>) {
    // No `tt_move`: it has its own stage, so it never needs to sort first.
    let killers = ctx.killers_at(ply);
    order_moves(
        board,
        &mut moves.as_mut_slice()[start..],
        None,
        killers,
        &ctx.history,
        &mut ctx.order_keys,
    );
}

/// Sorts captures and promotions ahead of quiets, captures by MVV-LVA (most
/// valuable victim, least valuable attacker), and the `killers` for this ply
/// ahead of the remaining quiets, which sort by `history`. `tt_move`, when it
/// is present in the list at all, sorts ahead of everything.
fn order_moves(
    board: &Board,
    moves: &mut [Move],
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
    history: &History,
    keys: &mut OrderKeys,
) {
    // Keys computed once, not per comparison: `order_key` is two mailbox
    // lookups plus a probe into a 32KiB history table, and paying for it
    // O(n log n) times rather than O(n) made move ordering the hottest thing
    // in the whole search.
    //
    // Insertion sort over the caller's buffer rather than `sort_by_cached_key`,
    // which heap-allocates its key buffer on every node. Both are stable, so
    // ties keep their generated order and the node count is unchanged.
    for (slot, &mv) in keys.iter_mut().zip(moves.iter()) {
        *slot = order_key(board, mv, tt_move, killers, history);
    }
    for i in 1..moves.len() {
        let (key, mv) = (keys[i], moves[i]);
        let mut j = i;
        while j > 0 && keys[j - 1] > key {
            keys[j] = keys[j - 1];
            moves[j] = moves[j - 1];
            j -= 1;
        }
        keys[j] = key;
        moves[j] = mv;
    }
}

/// Sort key for one move, ascending: gain descending, then attacker ascending
/// to break ties within a victim. Every capture keys at or below `-100`, which
/// leaves the interval `(-100, 0)` free for the killers. Other quiets key to
/// `(0, -history)` and sort last, an unseen quiet at `(0, 0)` behind them all.
/// The TT move keys below every capture.
fn order_key(
    board: &Board,
    mv: Move,
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
    history: &History,
) -> (i32, i32) {
    if tt_move == Some(mv) {
        return (TT_MOVE_KEY, 0);
    }
    let victim = victim_square(mv)
        .and_then(|square| board.piece_on(square))
        .map_or(0, |piece| ORDER_VALUES[piece.piece_type() as usize]);
    let promoted = mv
        .promoted_piece()
        .map_or(0, |kind| ORDER_VALUES[kind as usize]);
    if victim == 0 && promoted == 0 {
        if killers[0] == Some(mv) {
            return (-2, 0);
        }
        if killers[1] == Some(mv) {
            return (-1, 0);
        }
        let side = board.state().side_to_move() as usize;
        return (
            0,
            -history[side][mv.from().index() as usize][mv.to().index() as usize],
        );
    }
    let attacker = board
        .piece_on(mv.from())
        .map_or(0, |piece| ORDER_VALUES[piece.piece_type() as usize]);
    (-(victim + promoted), attacker)
}

/// The square the captured piece stands on: `to` for every capture except en
/// passant, where the victim sits beside the mover rather than on `to`.
fn victim_square(mv: Move) -> Option<Square> {
    match mv.move_type() {
        MoveType::EnPassant => Square::new(mv.to().file(), mv.from().rank()),
        _ if mv.is_capture() => Some(mv.to()),
        _ => None,
    }
}

/// Returns the material a capture wins or loses once both sides have finished
/// recapturing on the destination square, in centipawns.
///
/// Static exchange evaluation: it plays out the capture sequence with the
/// cheapest available attacker each time, which is what makes it static - no
/// search, no move generation, just the attacker sets.
fn see(board: &Board, mv: Move) -> i32 {
    let target = mv.to();
    let Some(victim) = victim_square(mv).and_then(|square| board.piece_on(square)) else {
        return 0;
    };
    let Some(attacker) = board.piece_on(mv.from()) else {
        return 0;
    };

    let mut occ = board.occupied();
    occ.clear(mv.from());
    if let Some(square) = victim_square(mv) {
        occ.clear(square);
    }

    // gain[i] is the material the side to move at depth i stands to win if
    // every capture from here on is played.
    let mut gain = [0i32; 32];
    gain[0] = ORDER_VALUES[victim.piece_type() as usize];
    let mut on_square = ORDER_VALUES[attacker.piece_type() as usize];
    let mut side = board.state().side_to_move().flip();
    let mut depth = 1;

    loop {
        // Recomputed each iteration against the shrinking occupancy: that is
        // what uncovers x-ray attackers standing behind a piece just captured.
        let attackers = attackers_to(board, target, occ) & occ;
        let Some((from, piece)) = cheapest_attacker(board, attackers, side) else {
            break;
        };
        gain[depth] = on_square - gain[depth - 1];
        on_square = ORDER_VALUES[piece as usize];
        occ.clear(from);
        side = side.flip();
        depth += 1;
        if depth >= gain.len() {
            break;
        }
    }

    // Walk back down: at every point a side may stop capturing rather than
    // continue into a losing exchange, so a gain is only realised if it beats
    // declining.
    while depth > 1 {
        depth -= 1;
        gain[depth - 1] = -(-gain[depth - 1]).max(gain[depth]);
    }
    gain[0]
}

/// The least valuable of `attackers` belonging to `side`, with its square.
fn cheapest_attacker(
    board: &Board,
    attackers: Bitboard,
    side: Color,
) -> Option<(Square, PieceType)> {
    let ours = attackers & board.color(side);
    for kind in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ] {
        if let Some(square) = (ours & board.pieces(kind)).lsb() {
            return Some((square, kind));
        }
    }
    None
}

impl Ctx<'_> {
    fn nmp_enabled(&self) -> bool {
        #[cfg(test)]
        {
            self.nmp
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    fn rfp_enabled(&self) -> bool {
        #[cfg(test)]
        {
            self.rfp
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    fn check_extension_enabled(&self) -> bool {
        #[cfg(test)]
        {
            self.check_extension
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    fn pvs_enabled(&self) -> bool {
        #[cfg(test)]
        {
            self.pvs
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    fn aspiration_enabled(&self) -> bool {
        #[cfg(test)]
        {
            self.aspiration
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    fn lmr_enabled(&self) -> bool {
        #[cfg(test)]
        {
            self.lmr
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    /// Every node visited. Clock and node-limit checks use this rather than
    /// `nodes` alone: quiescence is most of the tree, so counting main nodes
    /// only would leave thousands of nodes between two clock checks.
    fn total(&self) -> u64 {
        self.nodes + self.qnodes
    }

    fn check_abort(&self) -> Result<(), Aborted> {
        if self.iteration_depth == 1 || !self.total().is_multiple_of(CHECK_INTERVAL) {
            return Ok(());
        }
        if self.out_of_time() {
            Err(Aborted)
        } else {
            Ok(())
        }
    }

    /// The killer slots for `ply`, empty past `MAX_PLY`.
    fn killers_at(&self, ply: u32) -> [Option<Move>; 2] {
        *self.killers.get(ply as usize).unwrap_or(&[None; 2])
    }

    fn store_killer(&mut self, ply: u32, mv: Move) {
        let Some(slot) = self.killers.get_mut(ply as usize) else {
            return;
        };
        if slot[0] == Some(mv) {
            return;
        }
        slot[1] = slot[0];
        slot[0] = Some(mv);
    }

    fn out_of_time(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self.limits.nodes.is_some_and(|limit| self.total() >= limit)
            || self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }
}

/// Moves a mate score `distance` plies further from the mate, leaving ordinary
/// scores alone. A positive score means we mate, so it shifts the other way.
fn shift_mate(score: i32, distance: i32) -> i32 {
    if score > MATE_BOUND {
        score + distance
    } else if score < -MATE_BOUND {
        score - distance
    } else {
        score
    }
}

/// Rewrites a mate score to be relative to the node it is stored at rather
/// than the root, so probing it at another ply reports the right distance.
fn score_to_tt(score: i32, ply: u32) -> i32 {
    shift_mate(score, ply as i32)
}

/// Inverse of `score_to_tt`.
fn score_from_tt(score: i32, ply: u32) -> i32 {
    shift_mate(score, -(ply as i32))
}

/// Whether the move just made was legal, i.e. left `us` without its king in
/// check. Call with the board after `make` and the side that moved.
fn is_legal(board: &Board, us: Color) -> bool {
    !is_attacked(board, board.king_square(us), us.flip())
}

/// Whether the position is drawn by repetition or by the fifty-move rule.
///
/// Both are properties of the path to this position rather than of the
/// position itself, which is why the result must never reach the
/// transposition table.
///
/// The fifty-move test deliberately ignores the case where the hundredth
/// halfmove delivers mate, which is a win rather than a draw. Detecting that
/// costs a legal move generation at every node to answer a question that
/// decides a handful of games ever.
fn is_draw(board: &Board) -> bool {
    board.state().halfmove_clock() >= FIFTY_MOVE_PLIES || board.is_repetition()
}

fn terminal_score(board: &Board, ply: u32) -> i32 {
    let side = board.state().side_to_move();
    if is_attacked(board, board.king_square(side), side.flip()) {
        -MATE + ply as i32
    } else {
        DRAW
    }
}

fn mate_in(score: i32) -> Option<i32> {
    (score.abs() > MATE_BOUND).then(|| {
        let moves = (MATE - score.abs() + 1) / 2;
        if score < 0 { -moves } else { moves }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::{MoveList, generate_legal};
    use crate::uci::{UciListener, move_text};

    fn run(fen: &str, depth: u32) -> (SearchResult, String) {
        let mut board: Board = fen.parse().unwrap();
        let mut listener = UciListener::new(Vec::new());
        let result = search_inner(
            &mut board,
            Limits {
                depth: Some(depth),
                ..Limits::default()
            },
            &AtomicBool::new(false),
            &TranspositionTable::new(),
            &mut listener,
        );
        (result, String::from_utf8(listener.into_inner()).unwrap())
    }

    fn test_ctx(iteration_depth: u32) -> Ctx<'static> {
        // Leaked so the oracle and the real search can share a `Ctx` shape
        // without threading a lifetime through every test helper.
        let stop: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
        let tt: &'static TranspositionTable = Box::leak(Box::new(TranspositionTable::new()));
        Ctx {
            limits: Limits::default(),
            stop,
            deadline: None,
            nodes: 0,
            qnodes: 0,
            iteration_depth,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            order_keys: [(0, 0); MoveList::CAPACITY],
            tt,
            #[cfg(feature = "profiling")]
            profile: SearchProfile::default(),
            quiesce_leaves: true,
            pvs: true,
            aspiration: true,
            nmp: true,
            rfp: true,
            #[cfg(test)]
            check_extension: true,
            lmr: true,
            lmr_reductions: 0,
        }
    }

    /// A `Ctx` whose leaves are the static eval, matching `plain_negamax`.
    fn static_leaf_ctx(iteration_depth: u32) -> Ctx<'static> {
        Ctx {
            quiesce_leaves: false,
            nmp: false,
            rfp: false,
            #[cfg(test)]
            check_extension: false,
            lmr: false,
            ..test_ctx(iteration_depth)
        }
    }

    /// The clock, not the position, decides: the same board is a forced mate
    /// with a fresh clock and a draw with an expired one. Asserting both halves
    /// is what makes this a test of the rule rather than of the eval.
    #[test]
    fn fifty_move_rule_draws_a_won_position() {
        let winning = "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1";
        let expired = "7k/8/5K2/8/8/8/8/6Q1 w - - 100 1";

        let mut board: Board = winning.parse().unwrap();
        let score = negamax_root(&mut board, 4, -INFINITY, INFINITY, &mut test_ctx(4))
            .unwrap()
            .1;
        assert!(score > MATE_BOUND, "expected a mate score, got {score}");

        let mut board: Board = expired.parse().unwrap();
        let score = negamax_root(&mut board, 4, -INFINITY, INFINITY, &mut test_ctx(4))
            .unwrap()
            .1;
        assert_eq!(score, DRAW, "an expired clock must draw a won position");
    }

    /// White is a queen up, so every honest evaluation of this position is a
    /// large positive score - but the position has already occurred, so the
    /// search must return a draw regardless of the material on the board.
    /// Material and verdict disagreeing is what makes this a test of
    /// repetition rather than of the eval.
    #[test]
    fn repetition_outranks_material() {
        let fen = "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1";
        let mut board: Board = fen.parse().unwrap();
        let fresh = negamax_root(&mut board.clone(), 4, -INFINITY, INFINITY, &mut test_ctx(4))
            .unwrap()
            .1;
        assert!(fresh > 500, "expected a winning score, got {fresh}");

        // Three plies of shuffling, leaving black to move one ply short of
        // repeating the initial position. The root itself must not be a draw -
        // the search has to find the repetition among the moves it tries, and
        // black, being a queen down, will take it.
        for mv in ["a1b1", "e8d8", "b1a1"] {
            let mv = find_move(&mut board, mv);
            board.make(mv);
        }
        assert!(
            !board.is_repetition(),
            "the root must not already be a repetition"
        );
        let score = negamax_root(&mut board, 4, -INFINITY, INFINITY, &mut test_ctx(4))
            .unwrap()
            .1;
        assert_eq!(
            score, DRAW,
            "black is a queen down and must take the repetition"
        );
    }

    /// A null move pushes an `Undo` and advances the halfmove clock without
    /// changing the piece placement, so a scan that ignored the side to move
    /// could match the pre-null position and report a repetition that never
    /// happened. Two nulls return to the original side to move, which is the
    /// case that would collide.
    #[test]
    fn null_moves_do_not_fabricate_a_repetition() {
        let mut board: Board = "4k3/8/8/3n4/8/8/4P3/4K3 w - - 4 1".parse().unwrap();
        assert!(!board.is_repetition());
        board.make_null();
        assert!(!board.is_repetition(), "one null move repeated nothing");
        board.make_null();
        assert!(
            !board.is_repetition(),
            "two null moves must not look like a repetition"
        );
        board.unmake_null();
        board.unmake_null();
        assert!(!board.is_repetition());
    }

    /// catches: the tempo bonus leaking into the null-move gate.
    ///
    /// `evaluate` adds `TEMPO` to whoever is on move, and a null move changes
    /// only that, so a position and its null differ by `2 * TEMPO` where they
    /// should be exact negations. The gate reads a pre-null score and compares
    /// it against one returned through a null, so the bonus has to come off or
    /// null-move pruning cuts on an advantage it invented.
    ///
    /// Written against the raw difference rather than a node count: a node
    /// count would also move for a dozen unrelated reasons.
    #[test]
    fn the_tempo_bonus_does_not_survive_a_null_move() {
        crate::movegen::init();
        for fen in [
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ] {
            let mut board: Board = fen.parse().unwrap();
            let before = evaluate(&board);
            board.make_null();
            let after = evaluate(&board);
            board.unmake_null();
            assert_eq!(
                before + after,
                2 * TEMPO,
                "a null move should leave the position worth its negation \
                 plus the bonus both sides collected: {fen}"
            );
            assert_eq!(
                (before - TEMPO) + (after - TEMPO),
                0,
                "stripping the bonus must make the two exact negations: {fen}"
            );
        }
    }

    fn find_move(board: &mut Board, text: &str) -> Move {
        let mut moves = MoveList::new();
        generate_legal(board, &mut moves);
        *moves
            .iter()
            .find(|mv| crate::uci::move_text(**mv) == text)
            .unwrap_or_else(|| panic!("{text} is not legal here"))
    }

    #[test]
    fn bare_pawn_zugzwang_suppresses_null_move_pruning() {
        let fen = "8/5pk1/6p1/3p4/3P4/6P1/5PK1/8 w - - 0 1";
        let mut enabled_board: Board = fen.parse().unwrap();
        let mut enabled = test_ctx(5);
        let enabled_score = negamax_root(&mut enabled_board, 5, -INFINITY, INFINITY, &mut enabled)
            .unwrap()
            .1;

        let mut disabled_board: Board = fen.parse().unwrap();
        let mut disabled = test_ctx(5);
        disabled.nmp = false;
        let disabled_score =
            negamax_root(&mut disabled_board, 5, -INFINITY, INFINITY, &mut disabled)
                .unwrap()
                .1;

        assert_eq!(enabled_score, disabled_score);
        assert_eq!(enabled.nodes, disabled.nodes);
    }

    /// Depth 8, because the margin at this position decays as other pruning
    /// lands. Measured from the start position: depth 6 saves 317 nodes (30774
    /// vs 31091, 1.0%), depth 7 saves 19069 (18.3%), depth 8 saves 53565
    /// (205632 vs 259197, 20.7%) and depth 9 saves 277021 (36.9%). Depth 6 was
    /// chosen when it still saved thousands, and this comment previously
    /// rejected depth 5 for saving "under 1% of the tree" - which is what depth
    /// 6 now saves, so it had decayed onto the same noise floor and unrelated
    /// search changes flipped its sign either way.
    ///
    /// The saving is still growing at depth 9, so 8 is a floor rather than a
    /// plateau, chosen for a 20% margin at a quarter of a second. Re-measure
    /// rather than nudge the depth if this starts failing again: each new
    /// pruning term takes candidate nodes from the ones already here, so a
    /// shrinking margin is the search changing shape, not the property dying.
    #[test]
    fn null_move_pruning_reduces_nodes() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mut enabled_board: Board = fen.parse().unwrap();
        let mut enabled = test_ctx(8);
        negamax_root(&mut enabled_board, 8, -INFINITY, INFINITY, &mut enabled).unwrap();

        let mut disabled_board: Board = fen.parse().unwrap();
        let mut disabled = test_ctx(8);
        disabled.nmp = false;
        negamax_root(&mut disabled_board, 8, -INFINITY, INFINITY, &mut disabled).unwrap();

        assert!(
            enabled.nodes < disabled.nodes,
            "NMP did not reduce nodes: {} vs {}",
            enabled.nodes,
            disabled.nodes
        );
    }

    /// The point of the table over a constant is that reduction grows with
    /// both terms, so this asserts monotonicity in each rather than a set of
    /// tabulated values, which would only restate the formula.
    #[test]
    fn the_reduction_grows_with_depth_and_move_number() {
        for index in [4, 8, 16, 32] {
            for depth in 4..40u32 {
                assert!(
                    lmr_reduction(depth + 1, index) >= lmr_reduction(depth, index),
                    "depth {depth} -> {} but {} -> {} at index {index}",
                    lmr_reduction(depth, index),
                    depth + 1,
                    lmr_reduction(depth + 1, index),
                );
            }
        }
        for depth in [4, 8, 16, 32] {
            for index in 4..40usize {
                assert!(
                    lmr_reduction(depth, index + 1) >= lmr_reduction(depth, index),
                    "index {index} -> {} but {} -> {} at depth {depth}",
                    lmr_reduction(depth, index),
                    index + 1,
                    lmr_reduction(depth, index + 1),
                );
            }
        }
        // Strict somewhere, or a constant reduction would satisfy every
        // >= above and the table would be pointless.
        assert!(
            lmr_reduction(32, 32) > lmr_reduction(4, 4),
            "a late move at high depth must reduce more than an early one at low depth"
        );
        assert!(
            lmr_reduction(32, 8) > lmr_reduction(4, 8),
            "reduction must grow with depth at a fixed move number"
        );
        assert!(
            lmr_reduction(8, 32) > lmr_reduction(8, 4),
            "reduction must grow with move number at a fixed depth"
        );
    }

    /// A reduction that swallows the whole remaining depth answers a quiet move
    /// with quiescence, which searches captures only and so can say nothing
    /// about it. The bound leaves at least one ply of main search.
    #[test]
    fn a_reduction_never_reaches_quiescence() {
        for depth in LMR_MIN_DEPTH..64 {
            for index in LMR_MIN_INDEX..64 {
                let reduction = lmr_reduction(depth, index);
                assert!(
                    reduction >= 1,
                    "an eligible move must be reduced: depth {depth} index {index}"
                );
                assert!(
                    depth.saturating_sub(1 + reduction) >= 1,
                    "depth {depth} index {index} reduced by {reduction} lands in quiescence"
                );
            }
        }
    }

    #[test]
    fn late_move_reductions_reduce_nodes() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let mut enabled_board: Board = fen.parse().unwrap();
        let mut enabled = test_ctx(5);
        negamax_root(&mut enabled_board, 5, -INFINITY, INFINITY, &mut enabled).unwrap();

        let mut disabled_board: Board = fen.parse().unwrap();
        let mut disabled = test_ctx(5);
        disabled.lmr = false;
        negamax_root(&mut disabled_board, 5, -INFINITY, INFINITY, &mut disabled).unwrap();

        assert!(enabled.lmr_reductions > 0, "LMR never fired");
        assert!(
            enabled.nodes < disabled.nodes,
            "LMR did not reduce nodes: {} vs {}",
            enabled.nodes,
            disabled.nodes
        );
    }

    #[test]
    fn root_moves_are_not_reduced() {
        let mut board: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse()
            .unwrap();
        let mut ctx = test_ctx(LMR_MIN_DEPTH);
        negamax_root(&mut board, LMR_MIN_DEPTH, -INFINITY, INFINITY, &mut ctx).unwrap();
        assert_eq!(ctx.lmr_reductions, 0, "LMR reached a root move");
    }

    #[test]
    fn killer_moves_are_not_reduced() {
        let mut board: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse()
            .unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let mv = moves
            .iter()
            .copied()
            .find(|&mv| move_text(mv) == "e2e3")
            .unwrap();
        let mut ctx = test_ctx(LMR_MIN_DEPTH);
        ctx.killers[0][0] = Some(mv);
        board.make(mv);
        assert!(!lmr_eligible(
            LMR_MIN_DEPTH,
            LMR_MIN_INDEX,
            mv,
            0,
            false,
            &ctx
        ));
    }

    #[test]
    fn moves_that_leave_the_child_in_check_are_not_reduced() {
        let mut board: Board = "4k3/8/8/8/8/8/R7/K7 w - - 0 1".parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let mv = moves
            .iter()
            .copied()
            .find(|&mv| move_text(mv) == "a2e2")
            .unwrap();
        let ctx = test_ctx(LMR_MIN_DEPTH);
        board.make(mv);
        let them = board.state().side_to_move();
        let gives_check = is_attacked(&board, board.king_square(them), them.flip());
        assert!(gives_check);
        assert!(!lmr_eligible(
            LMR_MIN_DEPTH,
            LMR_MIN_INDEX,
            mv,
            0,
            gives_check,
            &ctx
        ));
    }

    #[test]
    fn pvs_preserves_scores_and_reduces_nodes() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        ];
        let mut alpha_beta_nodes = 0;
        let mut pvs_nodes = 0;
        for fen in fens {
            let mut board: Board = fen.parse().unwrap();
            let mut alpha_beta_ctx = static_leaf_ctx(3);
            alpha_beta_ctx.pvs = false;
            let alpha_beta = negamax_root(&mut board, 3, -INFINITY, INFINITY, &mut alpha_beta_ctx)
                .unwrap()
                .1;
            alpha_beta_nodes += alpha_beta_ctx.nodes;

            let mut pvs_ctx = static_leaf_ctx(3);
            let pvs = negamax_root(&mut board, 3, -INFINITY, INFINITY, &mut pvs_ctx)
                .unwrap()
                .1;
            pvs_nodes += pvs_ctx.nodes;

            assert_eq!(pvs, alpha_beta, "PVS changed the score for {fen}");
        }
        assert!(
            pvs_nodes < alpha_beta_nodes,
            "PVS did not reduce nodes: {pvs_nodes} vs {alpha_beta_nodes}"
        );
    }

    // catches: `search_move` re-searching on the wrong window now that the
    // root and the interior share it. The root passes beta = INFINITY, so a
    // null-window probe that beats alpha must re-search there too - a
    // condition that held for the interior but silently skipped the root's
    // re-search would leave the root scoring off a null window, which is a
    // bound rather than a score. Compared against the unpruned oracle, since
    // that is what a wrong window would disagree with.
    #[test]
    fn root_and_interior_agree_with_the_oracle_on_every_pvs_path() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            // Mate is in range here, so the re-search runs on scores that
            // `shift_mate` rewrites on the way through the table.
            "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        ];
        for fen in fens {
            for depth in 1..=3 {
                let mut board: Board = fen.parse().unwrap();
                let mut nodes = 0;
                let expected = plain_negamax(&mut board, depth, 0, &mut nodes);
                let mut ctx = static_leaf_ctx(depth);
                let (_, score) =
                    negamax_root(&mut board, depth, -INFINITY, INFINITY, &mut ctx).unwrap();
                assert_eq!(score, expected, "{fen} at depth {depth}");
            }
        }
    }

    #[test]
    fn mate_score_shifts_are_inverses_and_spare_ordinary_scores() {
        for distance in [0, 1, 5, 30] {
            for score in [-MATE + 3, -MATE + 40, 0, 250, -700, MATE - 40, MATE - 3] {
                assert_eq!(shift_mate(shift_mate(score, distance), -distance), score);
            }
        }
        // A score just inside the mate band must not shift; just outside must.
        assert_eq!(shift_mate(MATE_BOUND, 7), MATE_BOUND);
        assert_eq!(shift_mate(-MATE_BOUND, 7), -MATE_BOUND);
        assert_eq!(shift_mate(MATE_BOUND + 1, 7), MATE_BOUND + 8);
        assert_eq!(shift_mate(-MATE_BOUND - 1, 7), -MATE_BOUND - 8);
    }

    /// Unpruned negamax with a static-eval leaf, the oracle for the alpha-beta
    /// equivalence tests.
    ///
    /// Deliberately does not run quiescence: pruning equivalence is a property
    /// of the main search, and quiescence is a subroutine both sides would call
    /// identically. Running it here would multiply an already-unpruned tree by
    /// the whole quiescence subtree for no extra coverage - the qsearch tests
    /// below check that directly.
    fn plain_negamax(board: &mut Board, depth: u32, ply: u32, nodes: &mut u64) -> i32 {
        *nodes += 1;
        if depth == 0 {
            return evaluate(board);
        }
        let mut moves = MoveList::new();
        generate_legal(board, &mut moves);
        if moves.is_empty() {
            return terminal_score(board, ply);
        }
        let mut best = -INFINITY;
        for &mv in moves.iter() {
            board.make(mv);
            let score = -plain_negamax(board, depth - 1, ply + 1, nodes);
            board.unmake(mv);
            best = best.max(score);
        }
        best
    }

    #[test]
    fn pruning_preserves_scores() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        ];
        for fen in fens {
            let mut board: Board = fen.parse().unwrap();
            let mut plain_nodes = 0;
            let expected = plain_negamax(&mut board, 3, 0, &mut plain_nodes);

            let mut ctx = static_leaf_ctx(3);
            let (best_move, score) =
                negamax_root(&mut board, 3, -INFINITY, INFINITY, &mut ctx).unwrap();
            assert_eq!(score, expected, "score changed for {fen}");
            assert!(best_move.is_some());
            assert!(
                ctx.nodes < plain_nodes,
                "no pruning for {fen}: {} vs {plain_nodes}",
                ctx.nodes
            );
        }
    }

    // catches: the search filtering legality itself rather than taking a
    // pre-filtered list. `plain_negamax` still calls `generate_legal`, so
    // these compare the two strategies against each other on the positions
    // where they can disagree: a pinned piece whose pseudo-legal moves are
    // all illegal, an en-passant capture that exposes the king along a rank,
    // castling out of and through check, and a double check where only king
    // moves answer. A search that forgot to filter would return a score off
    // an illegal move; one that miscounted legal moves would report mate or
    // stalemate wrongly.
    #[test]
    fn search_filters_legality_exactly_as_generate_legal_does() {
        let fens = [
            // En passant discovering check along the fifth rank: the classic
            // case a legality filter gets wrong.
            "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",
            // Absolutely pinned knight - every one of its moves is illegal.
            "4k3/8/8/8/8/8/3n4/3KR3 b - - 0 1",
            // In check with castling rights still set.
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            // Double check: only king moves are legal.
            "4k3/8/8/8/8/2n5/3PPP2/r3K3 w - - 0 1",
            // Stalemate, so the legal count must be exactly zero.
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
            // Checkmate, likewise zero but scored as mate rather than draw.
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        ];
        for fen in fens {
            for depth in 1..=3 {
                let mut board: Board = fen.parse().unwrap();
                let mut nodes = 0;
                let expected = plain_negamax(&mut board, depth, 0, &mut nodes);
                let mut ctx = static_leaf_ctx(depth);
                let (_, score) =
                    negamax_root(&mut board, depth, -INFINITY, INFINITY, &mut ctx).unwrap();
                assert_eq!(score, expected, "{fen} at depth {depth}");
            }
        }
    }

    fn ordered(fen: &str) -> (Board, MoveList) {
        let mut board: Board = fen.parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        order_moves(
            &board,
            moves.as_mut_slice(),
            None,
            [None; 2],
            &NO_HISTORY,
            &mut no_keys(),
        );
        (board, moves)
    }

    static NO_HISTORY: History = [[[0; 64]; 64]; 2];

    fn no_keys() -> OrderKeys {
        [(0, 0); MoveList::CAPACITY]
    }

    fn key(board: &Board, mv: Move) -> (i32, i32) {
        order_key(board, mv, None, [None; 2], &NO_HISTORY)
    }

    #[test]
    fn captures_sort_before_quiets_by_mvv_lva() {
        // The e4 pawn and the c3 knight both attack the d5 queen; the knight
        // also attacks the cheaper b5 knight.
        let (board, moves) = ordered("4k3/8/8/1n1q4/4P3/2N5/8/4K3 w - - 0 1");
        let keys: Vec<_> = moves.iter().map(|&mv| key(&board, mv)).collect();
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "not sorted: {keys:?}"
        );

        let first = *moves.iter().next().unwrap();
        assert!(first.is_capture());
        assert!(
            move_text(first).ends_with("d5"),
            "should take the queen, got {}",
            move_text(first)
        );
        assert_eq!(
            board.piece_on(first.from()).unwrap().piece_type(),
            PieceType::Pawn,
            "pawn is the cheaper attacker of the queen"
        );

        let quiets = moves.iter().filter(|mv| key(&board, **mv) == (0, 0));
        let captures = moves.iter().filter(|mv| mv.is_capture()).count();
        assert!(captures >= 2);
        assert!(quiets.count() > 0, "position should have quiet moves too");
    }

    #[test]
    fn killers_sort_between_captures_and_other_quiets() {
        let mut board: Board = "4k3/8/8/1n1q4/4P3/2N5/8/4K3 w - - 0 1".parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let killer = *moves
            .iter()
            .find(|mv| !mv.is_capture() && !mv.is_promotion())
            .unwrap();
        let killers = [Some(killer), None];
        order_moves(
            &board,
            moves.as_mut_slice(),
            None,
            killers,
            &NO_HISTORY,
            &mut no_keys(),
        );

        let keys: Vec<_> = moves
            .iter()
            .map(|&mv| order_key(&board, mv, None, killers, &NO_HISTORY))
            .collect();
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "not sorted: {keys:?}"
        );

        let capture = *moves.iter().find(|mv| mv.is_capture()).unwrap();
        let quiet = *moves
            .iter()
            .find(|mv| !mv.is_capture() && **mv != killer)
            .unwrap();
        assert!(
            order_key(&board, capture, None, killers, &NO_HISTORY)
                < order_key(&board, killer, None, killers, &NO_HISTORY)
        );
        assert!(
            order_key(&board, killer, None, killers, &NO_HISTORY)
                < order_key(&board, quiet, None, killers, &NO_HISTORY)
        );
    }

    #[test]
    fn killer_slots_rotate_and_dedupe() {
        let mut board = Board::startpos();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let a = *moves.iter().next().unwrap();
        let b = *moves.iter().nth(1).unwrap();

        let mut ctx = test_ctx(1);
        ctx.store_killer(3, a);
        assert_eq!(ctx.killers_at(3), [Some(a), None]);
        ctx.store_killer(3, a);
        assert_eq!(
            ctx.killers_at(3),
            [Some(a), None],
            "duplicate should not shift"
        );
        ctx.store_killer(3, b);
        assert_eq!(ctx.killers_at(3), [Some(b), Some(a)]);

        // Out of range: no panic, no write, and reads come back empty.
        ctx.store_killer(MAX_PLY as u32, a);
        ctx.store_killer(u32::MAX, a);
        assert_eq!(ctx.killers_at(MAX_PLY as u32), [None; 2]);
        assert_eq!(ctx.killers_at(u32::MAX), [None; 2]);
    }

    #[test]
    fn history_orders_the_remaining_quiets() {
        let mut board: Board = "4k3/8/8/1n1q4/4P3/2N5/8/4K3 w - - 0 1".parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let quiets: Vec<_> = moves
            .iter()
            .filter(|mv| !mv.is_capture() && !mv.is_promotion())
            .copied()
            .collect();
        let (killer, scored, unseen) = (quiets[0], quiets[1], quiets[2]);
        let killers = [Some(killer), None];

        let mut history: History = [[[0; 64]; 64]; 2];
        history[0][scored.from().index() as usize][scored.to().index() as usize] = 42;

        let capture = *moves.iter().find(|mv| mv.is_capture()).unwrap();
        assert!(
            order_key(&board, capture, None, killers, &history)
                < order_key(&board, killer, None, killers, &history)
        );
        assert!(
            order_key(&board, killer, None, killers, &history)
                < order_key(&board, scored, None, killers, &history)
        );
        assert!(
            order_key(&board, scored, None, killers, &history)
                < order_key(&board, unseen, None, killers, &history)
        );
        assert_eq!(order_key(&board, unseen, None, killers, &history), (0, 0));
    }

    #[test]
    fn history_bonus_scales_with_depth_and_saturates() {
        let mut board: Board = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let shallow = *moves.iter().next().unwrap();
        let deep = *moves.iter().nth(1).unwrap();

        let mut ctx = test_ctx(1);
        bump(&mut ctx, shallow, 2);
        bump(&mut ctx, deep, 5);
        assert_eq!(score_of(&ctx, shallow), 4);
        assert_eq!(score_of(&ctx, deep), 25);

        ctx.history[0][deep.from().index() as usize][deep.to().index() as usize] = i32::MAX - 1;
        bump(&mut ctx, deep, 8);
        assert_eq!(score_of(&ctx, deep), i32::MAX, "should saturate, not wrap");
    }

    fn bump(ctx: &mut Ctx<'_>, mv: Move, depth: u32) {
        let slot = &mut ctx.history[0][mv.from().index() as usize][mv.to().index() as usize];
        *slot = slot.saturating_add((depth * depth) as i32);
    }

    fn score_of(ctx: &Ctx<'_>, mv: Move) -> i32 {
        ctx.history[0][mv.from().index() as usize][mv.to().index() as usize]
    }

    #[test]
    fn en_passant_is_scored_as_a_capture() {
        // The ep victim is not on `to`, so a naive piece_on(to) lookup scores
        // this as a quiet and sorts it last.
        let (board, moves) = ordered("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        let ep = *moves
            .iter()
            .find(|mv| mv.move_type() == MoveType::EnPassant)
            .expect("d6 en passant should be legal");
        assert_ne!(key(&board, ep), (0, 0), "ep scored as a quiet");
        assert_eq!(key(&board, ep).0, -ORDER_VALUES[PieceType::Pawn as usize]);
    }

    #[test]
    fn promotions_outrank_quiets_and_stack_with_capture_value() {
        // The black king stays off f8: an e7 pawn attacks it there, so a
        // position with White to move would have Black already in check and
        // `generate_legal` would offer e7xf8 capturing the king.
        let (board, moves) = ordered("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1");
        let quiet_promo = *moves
            .iter()
            .find(|mv| {
                mv.is_promotion()
                    && !mv.is_capture()
                    && mv.promoted_piece() == Some(PieceType::Queen)
            })
            .unwrap();
        let promo_capture = *moves
            .iter()
            .find(|mv| {
                mv.is_promotion()
                    && mv.is_capture()
                    && mv.promoted_piece() == Some(PieceType::Queen)
            })
            .unwrap();
        let king_move = *moves.iter().find(|mv| !mv.is_promotion()).unwrap();

        assert!(key(&board, promo_capture) < key(&board, quiet_promo));
        assert!(key(&board, quiet_promo) < key(&board, king_move));
        assert_eq!(key(&board, king_move), (0, 0));
    }

    #[test]
    fn ordering_reduces_the_tree() {
        // Ordering must pay for itself in nodes; the score equality that
        // guards correctness is covered by `pruning_preserves_scores`. Every
        // cutoff here is a capture, so this measures MVV-LVA only; the killer
        // and history tables are exercised by `cutoffs_populate_the_tables`.
        //
        // The baseline is this position's known perft(4), asserted by
        // `movegen::tests::perft_reference_positions`. Leaves alone are a lower
        // bound on the unpruned node count, so the comparison is if anything
        // stricter than searching an unpruned tree here would be - and it costs
        // nothing to state a number another test already verifies.
        const UNPRUNED: u64 = 4_085_603;
        let mut board: Board =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                .parse()
                .unwrap();
        let mut ctx = static_leaf_ctx(4);
        negamax_root(&mut board, 4, -INFINITY, INFINITY, &mut ctx).unwrap();
        assert!(
            ctx.nodes * 64 < UNPRUNED,
            "ordering should cut the tree hard: {} vs {UNPRUNED}",
            ctx.nodes
        );
    }

    #[test]
    fn cutoffs_populate_the_tables() {
        // A real search must actually store killers and history, not just
        // read empty tables. Quiet cutoffs need a quiet position, so this
        // uses an endgame rather than the tactical suite above.
        crate::movegen::init();
        let mut board: Board = "8/5pk1/6p1/3p4/3P4/6P1/5PK1/8 w - - 0 1".parse().unwrap();
        let mut ctx = test_ctx(5);
        negamax_root(&mut board, 5, -INFINITY, INFINITY, &mut ctx).unwrap();

        let killers = ctx.killers.iter().flatten().filter(|k| k.is_some()).count();
        assert!(killers > 0, "no killer was stored by any cutoff");
        let history: i32 = ctx
            .history
            .iter()
            .flatten()
            .flatten()
            .filter(|&&h| h > 0)
            .count() as i32;
        assert!(history > 0, "no history bonus was recorded");
    }

    fn quiesce(fen: &str) -> (i32, u64) {
        crate::movegen::init();
        let mut board: Board = fen.parse().unwrap();
        let mut ctx = test_ctx(1);
        let score = qsearch(&mut board, 0, 0, -INFINITY, INFINITY, &mut ctx).unwrap();
        (score, ctx.qnodes)
    }

    #[test]
    fn resolves_a_hanging_capture_that_static_eval_misreads() {
        // The white queen can take a pawn on c5 that the b6 pawn defends.
        // A leaf evaluated right after Qxc5 reads +800; quiescence plays the
        // recapture out, sees the queen lost, and keeps the stand-pat instead.
        let fen = "4k3/8/1p6/2p5/3Q4/8/8/4K3 w - - 0 1";
        let mut board: Board = fen.parse().unwrap();
        let stand_pat = evaluate(&board);
        assert!(stand_pat > 0, "white is up a queen for two pawns");

        let (score, _) = quiesce(fen);
        assert_eq!(
            score,
            stand_pat,
            "qsearch should decline the capture, not score it at {}",
            stand_pat + 100
        );

        // The main search must inherit that refutation.
        crate::movegen::init();
        let mut ctx = test_ctx(2);
        let (best, _) = negamax_root(&mut board, 2, -INFINITY, INFINITY, &mut ctx).unwrap();
        assert_ne!(
            best.map(move_text).as_deref(),
            Some("d4c5"),
            "should not grab a defended pawn with the queen"
        );
    }

    #[test]
    fn stands_pat_in_a_quiet_position() {
        // No captures available: qsearch is just the static eval, one node.
        let fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";
        let board: Board = fen.parse().unwrap();
        let (score, qnodes) = quiesce(fen);
        assert_eq!(
            score,
            evaluate(&board),
            "no captures, so score is stand-pat"
        );
        assert_eq!(qnodes, 1);
    }

    /// Reverse futility answers a node with a static score instead of a
    /// search, so it must not change a mate verdict.
    ///
    /// The gate cannot fire for a large positive `beta` - the static score is
    /// material and placement, so it never reaches 29000 - but it clears a
    /// large negative one trivially, which is where the bound on `beta` earns
    /// its place. Black is mated in four here, verified against a depth-11
    /// search, so both directions have a mate answer to preserve. Comparing
    /// the gate against itself disabled is what makes this a test of the
    /// pruning rather than of the eval.
    #[test]
    fn reverse_futility_does_not_change_a_mate_verdict() {
        let fen = "7k/7p/5R1R/8/8/8/8/6K1 b - - 0 1";
        let score = |rfp: bool| {
            let mut board: Board = fen.parse().unwrap();
            let mut ctx = Ctx {
                rfp,
                ..test_ctx(10)
            };
            negamax_root(&mut board, 10, -INFINITY, INFINITY, &mut ctx)
                .unwrap()
                .1
        };
        let (with, without) = (score(true), score(false));
        assert!(
            mate_in(without).is_some(),
            "the position must be a mate for the test to say anything: {without}"
        );
        assert_eq!(
            with, without,
            "reverse futility changed a mate score it must have left alone"
        );
    }

    /// A node in check has no static score worth pruning on: the side to move
    /// must answer the check, and the reply may be forced and losing however
    /// good the material looks.
    ///
    /// Removing the guard does not currently fail this - at a margin of 300 no
    /// in-check node was observed clearing beta at all, instrumented over
    /// kiwipete to depth 6 - so this pins the intended behaviour ahead of the
    /// margin coming down rather than catching a live defect.
    #[test]
    fn reverse_futility_does_not_prune_in_check() {
        let fen = "4k3/8/8/8/8/8/4r3/4K2Q w - - 0 1";
        let board: Board = fen.parse().unwrap();
        let us = board.state().side_to_move();
        assert!(
            is_attacked(&board, board.king_square(us), us.flip()),
            "the position must be in check for the test to say anything"
        );
        let score = |rfp: bool| {
            let mut board: Board = fen.parse().unwrap();
            let mut ctx = Ctx { rfp, ..test_ctx(4) };
            negamax_root(&mut board, 4, -INFINITY, INFINITY, &mut ctx)
                .unwrap()
                .1
        };
        assert_eq!(
            score(true),
            score(false),
            "reverse futility pruned a node in check"
        );
    }

    /// The gate is load-bearing rather than decorative: turning it off has to
    /// move the node count, or the margin is so wide it never fires and the
    /// feature is dead code.
    #[test]
    fn reverse_futility_prunes() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let nodes = |rfp: bool| {
            let mut board: Board = fen.parse().unwrap();
            let mut ctx = Ctx { rfp, ..test_ctx(6) };
            negamax_root(&mut board, 6, -INFINITY, INFINITY, &mut ctx).unwrap();
            ctx.total()
        };
        let (on, off) = (nodes(true), nodes(false));
        assert!(on < off, "reverse futility did not prune: {on} vs {off}");
    }

    /// The extension holds depth constant, so unlike every other change here
    /// the recursion is not bounded by `depth == 0` the way every other change
    /// here is. This does not run away today - see `check_extension` - so what
    /// is asserted is that it still returns, and cheaply.
    #[test]
    fn the_ply_bound_terminates_a_check_sequence() {
        let mut board: Board = "7k/8/8/8/8/8/5RR1/6K1 w - - 0 1".parse().expect("fen");
        let mut ctx = test_ctx(6);
        let nodes = negamax_root(&mut board, 6, &mut ctx).map(|_| ctx.total());
        assert!(
            nodes.is_ok_and(|nodes| nodes < 5_000_000),
            "the ply bound did not contain the check sequence"
        );
    }

    /// Extending has to cost nodes, or the gate never fires and the feature is
    /// dead code. This is the mirror of `reverse_futility_prunes`: every other
    /// change here is checked for pruning, this one for spending.
    #[test]
    fn checks_are_extended() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let nodes = |check_extension: bool| {
            let mut board: Board = fen.parse().unwrap();
            let mut ctx = Ctx {
                check_extension,
                ..test_ctx(6)
            };
            negamax_root(&mut board, 6, &mut ctx).unwrap();
            ctx.total()
        };
        let (on, off) = (nodes(true), nodes(false));
        assert!(on > off, "checks were not extended: {on} vs {off}");
    }

    /// A check that hangs the checking piece is a tempo, not a threat, and
    /// extending it is what made the ungated version lose a ply of depth.
    ///
    /// Asserted on the gate rather than on a node count: checks arise all
    /// over a subtree, so a whole-search count cannot say which of them the
    /// gate declined.
    #[test]
    fn a_losing_check_is_not_extended() {
        crate::movegen::init();
        // The knight's only checks are Nxc7+ and Nxd6+, both onto a defended
        // square, so both lose the knight outright.
        let mut board: Board = "1b2k3/2p1p3/3p4/1N6/8/8/8/4K3 w - - 0 1".parse().unwrap();
        let ctx = test_ctx(4);
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let mut checks = 0;
        for &mv in moves.iter() {
            let losing = see(&board, mv) < 0;
            board.make(mv);
            let them = board.state().side_to_move();
            let gives_check = is_attacked(&board, board.king_square(them), them.flip());
            board.unmake(mv);
            if !gives_check {
                continue;
            }
            checks += 1;
            assert!(losing, "test position must have only losing checks: {mv:?}");
            assert_eq!(
                check_extension(gives_check, losing, 0, &ctx),
                0,
                "a check losing the piece outright was extended: {mv:?}"
            );
        }
        assert!(checks > 0, "position has no checks to decline");
    }

    #[test]
    fn qply_bound_terminates_a_long_exchange() {
        // A stack of mutual captures deeper than MAX_QPLY: the bound is what
        // stops this, so it must return rather than run away.
        let (_, qnodes) = quiesce("3rr1k1/8/8/3pp3/3PP3/8/8/3RR1K1 w - - 0 1");
        assert!(qnodes > 1, "position has captures to search");
        assert!(qnodes < 100_000, "qply bound should contain it: {qnodes}");
    }

    #[test]
    fn searches_evasions_and_finds_mate_in_check() {
        // Back-rank mate: king boxed in by its own pawns. In check with no
        // legal move, qsearch must return a mate score - it cannot find the
        // escape from captures alone, and cannot stand pat in check.
        let (score, _) = quiesce("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1");
        assert_eq!(score, -MATE, "checkmate should score as mate at ply 0");

        // Same check, but h7 is open: the king walks out, so this is no mate.
        let (score, _) = quiesce("R5k1/5pp1/8/8/8/8/8/6K1 b - - 0 1");
        assert!(score > -MATE + 1000, "king has an escape, got {score}");
    }

    /// SEE of the capture from `from` to `to` in `fen`.
    fn see_of(fen: &str, from: &str, to: &str) -> i32 {
        crate::movegen::init();
        let mut board: Board = fen.parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        let mv = *moves
            .iter()
            .find(|mv| move_text(**mv) == format!("{from}{to}"))
            .unwrap_or_else(|| panic!("{from}{to} should be legal in {fen}"));
        see(&board, mv)
    }

    #[test]
    fn see_values_a_simple_exchange() {
        // Rook takes an undefended pawn: wins the pawn outright.
        assert_eq!(see_of("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1", "d5"), 100);

        // Rook takes a pawn defended by a pawn: wins 100, loses the rook.
        assert_eq!(
            see_of("4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1", "d1", "d5"),
            100 - ORDER_VALUES[PieceType::Rook as usize]
        );

        // Pawn takes a pawn defended by a pawn: even trade.
        assert_eq!(see_of("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4", "d5"), 0);
    }

    #[test]
    fn see_stops_a_losing_exchange_early() {
        // Two white attackers (pawn, rook) against two black defenders on d5.
        // White should not run the full sequence: the pawn trade is where it
        // stops, so this is a clean pawn win, not a rook-for-pawn loss.
        let value = see_of("3rk3/8/2p5/3p4/4P3/8/8/3RK3 w - - 0 1", "e4", "d5");
        assert_eq!(value, 0, "pxp then rxp is a trade, not a loss: {value}");
    }

    #[test]
    fn see_sees_through_an_x_ray_battery() {
        // Rooks stacked on d1/d2 behind each other. Taking the defended d5
        // pawn is only sound because the second rook backs the first up, and
        // that is invisible until the front rook clears the file.
        let doubled = see_of("3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2", "d5");
        // Single rook against the same defence loses material.
        let single = see_of("3rk3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1", "d5");
        assert!(
            doubled > single,
            "the backup rook must count: doubled {doubled} vs single {single}"
        );
        assert_eq!(doubled, 100, "RxP, RxR, RxR wins the pawn");
    }

    #[test]
    fn see_prunes_a_losing_capture_from_quiescence() {
        // Queen can take a pawn defended by a pawn. SEE rejects it, so
        // quiescence never searches it and returns the stand-pat.
        let fen = "4k3/8/1p6/2p5/3Q4/8/8/4K3 w - - 0 1";
        assert!(see_of(fen, "d4", "c5") < 0, "QxP is a losing capture");
        let (score, qnodes) = quiesce(fen);
        assert_eq!(score, evaluate(&fen.parse::<Board>().unwrap()));
        assert_eq!(qnodes, 1, "the losing capture should never be searched");
    }

    #[test]
    fn finds_mate_and_scores_stalemate() {
        let (mate, info) = run("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", 2);
        assert!(mate.best_move.is_some());
        assert!(info.contains("score mate 1"));
        let (stalemate, info) = run("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 2);
        assert_eq!(stalemate.best_move, None);
        assert!(info.contains("score cp 0"));
    }

    #[test]
    fn mate_distance_prefers_shorter_mates() {
        let checkmate: Board = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1".parse().unwrap();
        let mate_in_two = -terminal_score(&checkmate, 3);
        let mate_in_four = -terminal_score(&checkmate, 7);
        assert!(mate_in_two > mate_in_four);
        assert_eq!(mate_in(mate_in_two), Some(2));
        assert_eq!(mate_in(mate_in_four), Some(4));
    }

    #[test]
    fn tt_scores_survive_a_ply_shift() {
        // A mate score is stored relative to its own node, so writing it at
        // one ply and reading it at another must not move the mate nearer or
        // further. Ordinary scores must pass through untouched.
        for ply in [0, 1, 5, 30] {
            for score in [-MATE + 3, -MATE + 40, 0, 250, -700, MATE - 40, MATE - 3] {
                assert_eq!(
                    score_from_tt(score_to_tt(score, ply), ply),
                    score,
                    "score {score} at ply {ply}"
                );
            }
        }
        // Only mate scores shift.
        assert_eq!(score_to_tt(250, 7), 250);
        assert_eq!(score_to_tt(MATE - 3, 7), MATE + 4);
        assert_eq!(score_to_tt(-MATE + 3, 7), -MATE - 4);
    }

    #[test]
    fn tt_does_not_change_scores() {
        // The strongest correctness check available: a TT may reshape the tree
        // and must not reshape the result. Compares against the unpruned
        // oracle with the table warmed by a prior identical search, which is
        // when a wrongly signed bound or an unadjusted mate score shows up.
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        ];
        for fen in fens {
            let mut board: Board = fen.parse().unwrap();
            let mut plain_nodes = 0;
            let expected = plain_negamax(&mut board, 3, 0, &mut plain_nodes);

            let mut ctx = static_leaf_ctx(3);
            let cold = negamax_root(&mut board, 3, -INFINITY, INFINITY, &mut ctx)
                .unwrap()
                .1;
            assert_eq!(cold, expected, "cold TT changed the score for {fen}");

            // Same table, now populated: every node can hit.
            let warm = negamax_root(&mut board, 3, -INFINITY, INFINITY, &mut ctx)
                .unwrap()
                .1;
            assert_eq!(warm, expected, "warm TT changed the score for {fen}");
        }
    }

    #[test]
    fn tt_reports_a_stable_mate_distance_across_depths() {
        // Ra8 is mate: f7, g7 and h7 are blocked by Black's own pawns. A mate
        // score stored at one depth and re-read at another must report the
        // same distance - an unadjusted score drifts by the ply it is probed
        // at, which is exactly what iterative deepening does here.
        let mut board: Board = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1".parse().unwrap();
        let mut ctx = test_ctx(1);
        for depth in 1..=5 {
            ctx.iteration_depth = depth;
            let (best, score) =
                negamax_root(&mut board, depth, -INFINITY, INFINITY, &mut ctx).unwrap();
            assert_eq!(mate_in(score), Some(1), "depth {depth} scored {score}");
            assert_eq!(
                best.map(move_text).as_deref(),
                Some("a1a8"),
                "depth {depth} missed the mate"
            );
        }
    }

    #[test]
    fn tt_move_sorts_first() {
        let (board, mut moves) =
            ordered("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1");
        // A quiet move, which without the TT would sort behind every capture.
        let quiet = *moves
            .iter()
            .find(|mv| !mv.is_capture() && !mv.is_promotion())
            .expect("position has a quiet move");
        order_moves(
            &board,
            moves.as_mut_slice(),
            Some(quiet),
            [None; 2],
            &NO_HISTORY,
            &mut no_keys(),
        );
        assert_eq!(
            moves.iter().next(),
            Some(&quiet),
            "the TT move must sort ahead of captures"
        );
        assert!(
            order_key(&board, quiet, Some(quiet), [None; 2], &NO_HISTORY)
                < order_key(&board, quiet, None, [None; 2], &NO_HISTORY)
        );
    }

    #[test]
    fn a_warm_table_still_searches_every_iteration() {
        // Regression: a table warmed by a previous `go` used to let the root's
        // own children cut, so every iteration returned the stored score after
        // ~1 node per root move and the reported depth was fiction. Each
        // iteration must do work that grows with depth.
        crate::movegen::init();
        let tt = TranspositionTable::new();
        let stop = AtomicBool::new(false);
        let mut first = Board::startpos();
        search_inner(
            &mut first,
            Limits {
                depth: Some(7),
                ..Limits::default()
            },
            &stop,
            &tt,
            &mut (),
        );

        // Search the same position again: the root store left a deep entry for
        // it, which is exactly the case that used to freeze the score.
        let mut second = Board::startpos();
        let warm = tt.probe(second.state().zobrist());
        assert!(
            warm.is_some_and(|entry| entry.depth >= 5),
            "test needs a deep stored entry to be meaningful, got {warm:?}"
        );

        let mut counts = Vec::new();
        for depth in 1..=5 {
            let result = search_inner(
                &mut second,
                Limits {
                    depth: Some(depth),
                    ..Limits::default()
                },
                &stop,
                &tt,
                &mut (),
            );
            counts.push(result.nodes + result.qnodes);
        }
        // The bug signature was a flat count: every iteration returned after
        // roughly one node per root move, so depth 5 cost no more than depth
        // 1. A TT that is working still prunes hard, so the growth is not
        // monotonic - but the deepest iteration must clearly outwork the
        // shallowest.
        let (first_count, last_count) = (counts[0], counts[counts.len() - 1]);
        assert!(
            last_count > first_count * 4,
            "depth 5 did not outwork depth 1, got {counts:?}"
        );
    }

    #[test]
    fn tt_reduces_the_tree_across_iterations() {
        // The point of the table: a second search of the same position reuses
        // the first one's work.
        let mut board: Board =
            "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8"
                .parse()
                .unwrap();
        let mut ctx = test_ctx(4);
        negamax_root(&mut board, 4, -INFINITY, INFINITY, &mut ctx).unwrap();
        let cold = ctx.total();
        let before = ctx.total();
        negamax_root(&mut board, 4, -INFINITY, INFINITY, &mut ctx).unwrap();
        let warm = ctx.total() - before;
        assert!(
            warm < cold,
            "warm search {warm} was not cheaper than {cold}"
        );
    }

    #[test]
    fn depth_search_is_legal_and_deterministic() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let (first, _) = run(fen, 2);
        let (second, _) = run(fen, 2);
        assert_eq!(first.nodes, second.nodes);
        assert_eq!(first.qnodes, second.qnodes);
        assert_eq!(first.best_move, second.best_move);
        let mut board: Board = fen.parse().unwrap();
        let mut legal = MoveList::new();
        generate_legal(&mut board, &mut legal);
        assert!(legal.iter().any(|&mv| Some(mv) == first.best_move));
    }

    #[test]
    fn clock_budget_honors_tolerance_and_low_time() {
        let board = Board::startpos();
        assert!(
            time_budget(
                &board,
                Limits {
                    wtime: Some(1000),
                    ..Limits::default()
                }
            )
            .unwrap()
                < 950
        );
        assert_eq!(
            time_budget(
                &board,
                Limits {
                    wtime: Some(60),
                    ..Limits::default()
                }
            ),
            Some(3)
        );
    }

    #[test]
    fn respects_a_short_budget() {
        crate::movegen::init();
        let mut board = Board::startpos();
        let start = Instant::now();
        let result = search_inner(
            &mut board,
            Limits {
                wtime: Some(100),
                btime: Some(100),
                ..Limits::default()
            },
            &AtomicBool::new(false),
            &TranspositionTable::new(),
            &mut (),
        );
        let elapsed = start.elapsed();
        assert!(result.best_move.is_some());
        // Budget is 100/20 = 5ms. ChessEval flags at remaining + 50ms, so the
        // whole move must land inside 150ms even in a debug build.
        assert!(elapsed < Duration::from_millis(150), "took {elapsed:?}");
    }

    #[test]
    fn depth_one_completes_despite_stop() {
        let mut board = Board::startpos();
        let result = search_inner(
            &mut board,
            Limits {
                depth: Some(1),
                movetime: Some(1),
                ..Limits::default()
            },
            &AtomicBool::new(true),
            &TranspositionTable::new(),
            &mut (),
        );
        assert!(result.best_move.is_some());
        // 20 root moves plus one re-search each for the four that beat the
        // running best. Exact rather than a lower bound: this test exists to
        // pin deterministic behaviour, and the re-search count is the part
        // most likely to move unnoticed.
        assert_eq!(result.nodes, 24);
        assert!(result.qnodes > 0, "each leaf runs quiescence");
    }

    #[cfg(feature = "profiling")]
    #[test]
    fn profile_classifies_cutoffs_and_bounds() {
        let mut profile = SearchProfile::default();
        for move_number in [1, 2, 3, 4, 8, 9] {
            profile.record_cutoff(false, move_number);
        }
        assert_eq!(profile.main_cutoffs, [1, 1, 1, 2, 1]);

        profile.record_main_bound(20, -10, 20);
        profile.record_main_bound(-10, -10, 20);
        profile.record_main_bound(0, -10, 20);
        assert_eq!(profile.main_bounds, [1, 1, 1]);

        profile.record_qply(0);
        profile.record_qply(MAX_QPLY);
        assert_eq!(profile.qply[0], 1);
        assert_eq!(profile.qply[MAX_QPLY as usize], 1);

        profile.record_null_attempt();
        profile.record_null_cutoff();
        assert_eq!(profile.null_attempts, 1);
        assert_eq!(profile.null_cutoffs, 1);

        profile.record_lmr_reduction();
        profile.record_lmr_research();
        assert_eq!(profile.lmr_reductions, 1);
        assert_eq!(profile.lmr_researches, 1);
    }
}
