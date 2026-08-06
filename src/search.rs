//! Iterative-deepening negamax search.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::eval::evaluate;
use crate::movegen::{MoveList, generate_captures, generate_legal, generate_pseudo, is_attacked};
#[cfg(feature = "profiling")]
use crate::tt::Miss;
use crate::tt::{Bound, TranspositionTable};
use crate::uci::move_text;
use crate::{Board, Color, Move, MoveType, Square};

/// Nodes between clock checks. Low enough that a small tree still checks
/// several times: depth 2 from the start position is only 440 nodes, and at
/// 2048 it never checked at all.
const CHECK_INTERVAL: u64 = 256;
const MATE: i32 = 30_000;
const INFINITY: i32 = 31_000;
/// Scores above this in absolute value encode a distance to mate.
const MATE_BOUND: i32 = MATE - 1000;
/// Quiescence ply ceiling. Deep enough for any real exchange sequence, and a
/// hard bound keeps the bench node count finite and deterministic.
const MAX_QPLY: u32 = 8;
/// Killer ply ceiling. Search depth is bounded by the clock long before this;
/// past it the killer slots are simply skipped.
const MAX_PLY: usize = 64;

/// Cutoff counts per side and from/to square, the ordering score for quiets
/// that are neither captures nor killers.
type History = [[[i32; 64]; 64]; 2];

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
}

#[derive(Debug, Clone, Copy)]
struct Aborted;

/// Searches the position and returns the best move found.
pub fn search(
    board: &mut Board,
    limits: Limits,
    stop: &AtomicBool,
    tt: &TranspositionTable,
    output: &mut dyn Write,
) -> Option<Move> {
    let result = search_inner(board, limits, stop, tt, output, true);
    let best = result
        .best_move
        .map_or_else(|| "0000".to_owned(), move_text);
    let _ = writeln!(output, "bestmove {best}");
    result.best_move
}

pub(crate) fn search_inner(
    board: &mut Board,
    limits: Limits,
    stop: &AtomicBool,
    tt: &TranspositionTable,
    output: &mut dyn Write,
    emit_info: bool,
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
        tt,
        #[cfg(feature = "profiling")]
        profile: SearchProfile::default(),
        #[cfg(test)]
        quiesce_leaves: true,
        #[cfg(test)]
        pvs: true,
    };
    let max_depth = limits.depth.unwrap_or(u32::MAX).max(1);
    let mut best_move = None;

    for depth in 1..=max_depth {
        // An unpruned tree grows about 35x per depth, so starting an iteration
        // with the budget already spent overshoots it by that factor.
        if depth > 1 && ctx.out_of_time() {
            break;
        }
        ctx.iteration_depth = depth;
        let result = negamax_root(board, depth, &mut ctx);
        let Ok((candidate, score)) = result else {
            break;
        };
        best_move = candidate;
        if emit_info {
            write_info(
                output,
                depth,
                score,
                ctx.total(),
                start.elapsed(),
                candidate,
            );
        }
        if candidate.is_none() {
            break;
        }
    }

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

fn negamax_root(
    board: &mut Board,
    depth: u32,
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
    order_moves(board, &mut moves, tt_move, ctx.killers_at(0), &ctx.history);
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
        // The root always searches on a full window, so `beta` is INFINITY and
        // the re-search condition reduces to `narrow > best_score`.
        let result = search_move(board, depth, 1, best_score, INFINITY, legal, ctx);
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
    // Exact: rejected null-window results never replace `best_score`, while
    // every result that does replace it came from an exact-window search.
    #[cfg(feature = "profiling")]
    ctx.profile.record_tt_store();
    ctx.tt.store(
        key,
        score_to_tt(best_score, 0),
        best_move,
        depth,
        Bound::Exact,
    );
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
fn search_move(
    board: &mut Board,
    depth: u32,
    child_ply: u32,
    alpha: i32,
    beta: i32,
    index: usize,
    ctx: &mut Ctx<'_>,
) -> Result<i32, Aborted> {
    if index > 0 && ctx.pvs_enabled() {
        #[cfg(feature = "profiling")]
        ctx.profile.record_pvs_probe();
        let narrow = -negamax(board, depth - 1, child_ply, -alpha - 1, -alpha, ctx)?;
        if !(narrow > alpha && narrow < beta) {
            return Ok(narrow);
        }
        #[cfg(feature = "profiling")]
        ctx.profile.record_pvs_research();
    }
    negamax(board, depth - 1, child_ply, -beta, -alpha, ctx).map(|score| -score)
}

fn negamax(
    board: &mut Board,
    depth: u32,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &mut Ctx<'_>,
) -> Result<i32, Aborted> {
    ctx.nodes += 1;
    ctx.check_abort()?;
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
    let mut moves = MoveList::new();
    generate_pseudo(board, &mut moves);
    // The stored move is only a hint: a key collision or an entry from an
    // earlier game can decode to a move that is illegal here, so it orders the
    // list rather than being searched directly.
    let tt_move = entry.and_then(|entry| entry.best_move);
    order_moves(
        board,
        &mut moves,
        tt_move,
        ctx.killers_at(ply),
        &ctx.history,
    );
    let us = board.state().side_to_move();
    let mut best = -INFINITY;
    let mut best_move = None;
    let mut legal = 0;
    for &mv in moves.iter() {
        board.make(mv);
        if !is_legal(board, us) {
            board.unmake(mv);
            continue;
        }
        let result = search_move(board, depth, ply + 1, alpha, beta, legal, ctx);
        board.unmake(mv);
        legal += 1;
        let score = result?;
        if score > best {
            best = score;
            best_move = Some(mv);
        }
        alpha = alpha.max(best);
        if alpha >= beta {
            #[cfg(feature = "profiling")]
            ctx.profile.record_cutoff(false, legal);
            if !mv.is_capture() && !mv.is_promotion() {
                ctx.store_killer(ply, mv);
                // ponytail: no decay - history is per-`go` and dies with Ctx.
                // Add halve-on-cap or the gravity update if it goes stale
                // within one search.
                let slot = &mut ctx.history[us as usize][mv.from().index() as usize]
                    [mv.to().index() as usize];
                *slot = slot.saturating_add((depth * depth) as i32);
            }
            break;
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

    let mut moves = MoveList::new();
    let mut best = -INFINITY;
    if in_check {
        // No stand-pat in check: the side to move must answer it, and a quiet
        // king move may be the only answer, so this needs every legal move.
        generate_legal(board, &mut moves);
        if moves.is_empty() {
            #[cfg(feature = "profiling")]
            ctx.profile.record_q_exact();
            return Ok(terminal_score(board, ply));
        }
    } else {
        // Stand-pat: declining to capture is always an option, so the static
        // eval is a lower bound on this node.
        best = evaluate(board);
        #[cfg(feature = "profiling")]
        ctx.profile.record_stand_pat_node();
        if best >= beta {
            #[cfg(feature = "profiling")]
            ctx.profile.record_stand_pat_cutoff();
            return Ok(best);
        }
        alpha = alpha.max(best);
        generate_captures(board, &mut moves);
    }
    order_moves(board, &mut moves, None, [None; 2], &ctx.history);

    #[cfg(feature = "profiling")]
    let mut legal_moves = 0;
    for &mv in moves.iter() {
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

/// Sorts captures and promotions ahead of quiets, captures by MVV-LVA (most
/// valuable victim, least valuable attacker), and the `killers` for this ply
/// ahead of the remaining quiets, which sort by `history`. `tt_move`, when it
/// is present in the list at all, sorts ahead of everything.
fn order_moves(
    board: &Board,
    moves: &mut MoveList,
    tt_move: Option<Move>,
    killers: [Option<Move>; 2],
    history: &History,
) {
    // Cached, not `sort_unstable_by_key`: that re-evaluates its closure on
    // every comparison, and `order_key` is two mailbox lookups plus a probe
    // into a 32KiB history table. Paying for it O(n log n) times rather than
    // O(n) made move ordering the hottest thing in the whole search - this one
    // word is worth roughly 1.9x throughput. Measured against an
    // allocation-free array sort, which came out both slower and longer.
    moves
        .as_mut_slice()
        .sort_by_cached_key(|&mv| order_key(board, mv, tt_move, killers, history));
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

impl Ctx<'_> {
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

fn terminal_score(board: &Board, ply: u32) -> i32 {
    let side = board.state().side_to_move();
    if is_attacked(board, board.king_square(side), side.flip()) {
        -MATE + ply as i32
    } else {
        0
    }
}

fn mate_in(score: i32) -> Option<i32> {
    (score.abs() > MATE_BOUND).then(|| {
        let moves = (MATE - score.abs() + 1) / 2;
        if score < 0 { -moves } else { moves }
    })
}

fn write_info(
    output: &mut dyn Write,
    depth: u32,
    score: i32,
    nodes: u64,
    elapsed: Duration,
    best_move: Option<Move>,
) {
    let millis = elapsed.as_millis();
    let nps = u128::from(nodes) * 1000 / millis.max(1);
    let score_text = mate_in(score).map_or_else(
        || format!("score cp {score}"),
        |moves| format!("score mate {moves}"),
    );
    let pv = best_move.map_or_else(String::new, |mv| format!(" pv {}", move_text(mv)));
    let _ = writeln!(
        output,
        "info depth {depth} {score_text} nodes {nodes} nps {nps} time {millis}{pv}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PieceType;
    use crate::movegen::{MoveList, generate_legal};

    fn run(fen: &str, depth: u32) -> (SearchResult, String) {
        let mut board: Board = fen.parse().unwrap();
        let mut output = Vec::new();
        let result = search_inner(
            &mut board,
            Limits {
                depth: Some(depth),
                ..Limits::default()
            },
            &AtomicBool::new(false),
            &TranspositionTable::new(),
            &mut output,
            true,
        );
        (result, String::from_utf8(output).unwrap())
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
            tt,
            #[cfg(feature = "profiling")]
            profile: SearchProfile::default(),
            quiesce_leaves: true,
            pvs: true,
        }
    }

    /// A `Ctx` whose leaves are the static eval, matching `plain_negamax`.
    fn static_leaf_ctx(iteration_depth: u32) -> Ctx<'static> {
        Ctx {
            quiesce_leaves: false,
            ..test_ctx(iteration_depth)
        }
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
            let alpha_beta = negamax_root(&mut board, 3, &mut alpha_beta_ctx).unwrap().1;
            alpha_beta_nodes += alpha_beta_ctx.nodes;

            let mut pvs_ctx = static_leaf_ctx(3);
            let pvs = negamax_root(&mut board, 3, &mut pvs_ctx).unwrap().1;
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
                let (_, score) = negamax_root(&mut board, depth, &mut ctx).unwrap();
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
            let (best_move, score) = negamax_root(&mut board, 3, &mut ctx).unwrap();
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
                let (_, score) = negamax_root(&mut board, depth, &mut ctx).unwrap();
                assert_eq!(score, expected, "{fen} at depth {depth}");
            }
        }
    }

    fn ordered(fen: &str) -> (Board, MoveList) {
        let mut board: Board = fen.parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        order_moves(&board, &mut moves, None, [None; 2], &NO_HISTORY);
        (board, moves)
    }

    static NO_HISTORY: History = [[[0; 64]; 64]; 2];

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
        order_moves(&board, &mut moves, None, killers, &NO_HISTORY);

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
        negamax_root(&mut board, 4, &mut ctx).unwrap();
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
        negamax_root(&mut board, 5, &mut ctx).unwrap();

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
        let (best, _) = negamax_root(&mut board, 2, &mut ctx).unwrap();
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
            let cold = negamax_root(&mut board, 3, &mut ctx).unwrap().1;
            assert_eq!(cold, expected, "cold TT changed the score for {fen}");

            // Same table, now populated: every node can hit.
            let warm = negamax_root(&mut board, 3, &mut ctx).unwrap().1;
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
            let (best, score) = negamax_root(&mut board, depth, &mut ctx).unwrap();
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
        order_moves(&board, &mut moves, Some(quiet), [None; 2], &NO_HISTORY);
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
            &mut std::io::sink(),
            false,
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
                &mut std::io::sink(),
                false,
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
        negamax_root(&mut board, 4, &mut ctx).unwrap();
        let cold = ctx.total();
        let before = ctx.total();
        negamax_root(&mut board, 4, &mut ctx).unwrap();
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
            &mut std::io::sink(),
            false,
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
        let mut output = Vec::new();
        let result = search_inner(
            &mut board,
            Limits {
                depth: Some(1),
                movetime: Some(1),
                ..Limits::default()
            },
            &AtomicBool::new(true),
            &TranspositionTable::new(),
            &mut output,
            false,
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
    }
}
