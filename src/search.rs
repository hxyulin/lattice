//! Iterative-deepening negamax search.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::eval::evaluate;
use crate::movegen::{MoveList, generate_captures, generate_legal, is_attacked};
use crate::uci::move_text;
use crate::{Board, Move, MoveType, Square};

/// Nodes between clock checks. Low enough that a small tree still checks
/// several times: depth 2 from the start position is only 440 nodes, and at
/// 2048 it never checked at all.
const CHECK_INTERVAL: u64 = 256;
const MATE: i32 = 30_000;
const INFINITY: i32 = 31_000;
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
    /// Whether leaves run quiescence. Always true in play; the alpha-beta
    /// equivalence tests turn it off so their unpruned oracle stays cheap.
    #[cfg(test)]
    quiesce_leaves: bool,
}

#[derive(Debug, Clone, Copy)]
struct Aborted;

/// Searches the position and returns the best move found.
pub fn search(
    board: &mut Board,
    limits: Limits,
    stop: &AtomicBool,
    output: &mut dyn Write,
) -> Option<Move> {
    let result = search_inner(board, limits, stop, output, true);
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
    output: &mut dyn Write,
    emit_info: bool,
) -> SearchResult {
    let start = Instant::now();
    let deadline =
        time_budget(board, limits).and_then(|ms| start.checked_add(Duration::from_millis(ms)));
    let mut ctx = Ctx {
        limits,
        stop,
        deadline,
        nodes: 0,
        qnodes: 0,
        iteration_depth: 1,
        killers: [[None; 2]; MAX_PLY],
        history: [[[0; 64]; 64]; 2],
        #[cfg(test)]
        quiesce_leaves: true,
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
    generate_legal(board, &mut moves);
    if moves.is_empty() {
        return Ok((None, terminal_score(board, 0)));
    }
    order_moves(board, &mut moves, ctx.killers_at(0), &ctx.history);
    let mut best_move = None;
    let mut best_score = -INFINITY;
    for &mv in moves.iter() {
        board.make(mv);
        // Beta stays at INFINITY so no root move can fail low: a score above
        // `best_score` is exact, and one below is a safe rejection.
        let result = negamax(board, depth - 1, 1, -INFINITY, -best_score, ctx);
        board.unmake(mv);
        let score = -result?;
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
    }
    Ok((best_move, best_score))
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
        return qsearch(board, ply, 0, alpha, beta, ctx);
    }
    let mut moves = MoveList::new();
    generate_legal(board, &mut moves);
    if moves.is_empty() {
        return Ok(terminal_score(board, ply));
    }
    order_moves(board, &mut moves, ctx.killers_at(ply), &ctx.history);
    let us = board.state().side_to_move();
    let mut best = -INFINITY;
    for &mv in moves.iter() {
        board.make(mv);
        let result = negamax(board, depth - 1, ply + 1, -beta, -alpha, ctx);
        board.unmake(mv);
        best = best.max(-result?);
        alpha = alpha.max(best);
        if alpha >= beta {
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
    ctx.qnodes += 1;
    ctx.check_abort()?;
    if qply >= MAX_QPLY {
        return Ok(evaluate(board));
    }
    let us = board.state().side_to_move();
    let in_check = is_attacked(board, board.king_square(us), us.flip());

    let mut moves = MoveList::new();
    let mut best = -INFINITY;
    if in_check {
        // No stand-pat in check: the side to move must answer it, and a quiet
        // king move may be the only answer, so this needs every legal move.
        generate_legal(board, &mut moves);
        if moves.is_empty() {
            return Ok(terminal_score(board, ply));
        }
    } else {
        // Stand-pat: declining to capture is always an option, so the static
        // eval is a lower bound on this node.
        best = evaluate(board);
        if best >= beta {
            return Ok(best);
        }
        alpha = alpha.max(best);
        generate_captures(board, &mut moves);
    }
    order_moves(board, &mut moves, [None; 2], &ctx.history);

    for &mv in moves.iter() {
        board.make(mv);
        // Captures come back pseudo-legal. Filtering after `make` rather than
        // up front means a beta cutoff skips the checks it never needed.
        let legal = in_check || !is_attacked(board, board.king_square(us), us.flip());
        let result = legal.then(|| qsearch(board, ply + 1, qply + 1, -beta, -alpha, ctx));
        board.unmake(mv);
        let Some(result) = result else {
            continue;
        };
        best = best.max(-result?);
        alpha = alpha.max(best);
        if alpha >= beta {
            break;
        }
    }
    Ok(best)
}

/// Ordering values, deliberately separate from the eval's material values:
/// retuning the evaluation should not silently reshape the search tree. The
/// king is a victim value only, for pseudo-legal safety; it is never actually
/// captured.
const ORDER_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 10_000];

/// Sorts captures and promotions ahead of quiets, captures by MVV-LVA (most
/// valuable victim, least valuable attacker), and the `killers` for this ply
/// ahead of the remaining quiets, which sort by `history`.
fn order_moves(board: &Board, moves: &mut MoveList, killers: [Option<Move>; 2], history: &History) {
    moves
        .as_mut_slice()
        .sort_unstable_by_key(|&mv| order_key(board, mv, killers, history));
}

/// Sort key for one move, ascending: gain descending, then attacker ascending
/// to break ties within a victim. Every capture keys at or below `-100`, which
/// leaves the interval `(-100, 0)` free for the killers. Other quiets key to
/// `(0, -history)` and sort last, an unseen quiet at `(0, 0)` behind them all.
fn order_key(board: &Board, mv: Move, killers: [Option<Move>; 2], history: &History) -> (i32, i32) {
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

fn terminal_score(board: &Board, ply: u32) -> i32 {
    let side = board.state().side_to_move();
    if is_attacked(board, board.king_square(side), side.flip()) {
        -MATE + ply as i32
    } else {
        0
    }
}

fn mate_in(score: i32) -> Option<i32> {
    (score.abs() > MATE - 1000).then(|| {
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
            &mut output,
            true,
        );
        (result, String::from_utf8(output).unwrap())
    }

    fn test_ctx(iteration_depth: u32) -> Ctx<'static> {
        // Leaked so the oracle and the real search can share a `Ctx` shape
        // without threading a lifetime through every test helper.
        let stop: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
        Ctx {
            limits: Limits::default(),
            stop,
            deadline: None,
            nodes: 0,
            qnodes: 0,
            iteration_depth,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            quiesce_leaves: true,
        }
    }

    /// A `Ctx` whose leaves are the static eval, matching `plain_negamax`.
    fn static_leaf_ctx(iteration_depth: u32) -> Ctx<'static> {
        Ctx {
            quiesce_leaves: false,
            ..test_ctx(iteration_depth)
        }
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

    fn ordered(fen: &str) -> (Board, MoveList) {
        let mut board: Board = fen.parse().unwrap();
        let mut moves = MoveList::new();
        generate_legal(&mut board, &mut moves);
        order_moves(&board, &mut moves, [None; 2], &NO_HISTORY);
        (board, moves)
    }

    static NO_HISTORY: History = [[[0; 64]; 64]; 2];

    fn key(board: &Board, mv: Move) -> (i32, i32) {
        order_key(board, mv, [None; 2], &NO_HISTORY)
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
        order_moves(&board, &mut moves, killers, &NO_HISTORY);

        let keys: Vec<_> = moves
            .iter()
            .map(|&mv| order_key(&board, mv, killers, &NO_HISTORY))
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
            order_key(&board, capture, killers, &NO_HISTORY)
                < order_key(&board, killer, killers, &NO_HISTORY)
        );
        assert!(
            order_key(&board, killer, killers, &NO_HISTORY)
                < order_key(&board, quiet, killers, &NO_HISTORY)
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
            order_key(&board, capture, killers, &history)
                < order_key(&board, killer, killers, &history)
        );
        assert!(
            order_key(&board, killer, killers, &history)
                < order_key(&board, scored, killers, &history)
        );
        assert!(
            order_key(&board, scored, killers, &history)
                < order_key(&board, unseen, killers, &history)
        );
        assert_eq!(order_key(&board, unseen, killers, &history), (0, 0));
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
        let (board, moves) = ordered("3r1k2/4P3/8/8/8/8/8/4K3 w - - 0 1");
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
        let mut board: Board =
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                .parse()
                .unwrap();
        let mut ctx = static_leaf_ctx(4);
        let (_, ordered_score) = negamax_root(&mut board, 4, &mut ctx).unwrap();
        let mut plain_nodes = 0;
        let expected = plain_negamax(&mut board, 4, 0, &mut plain_nodes);
        assert_eq!(ordered_score, expected);
        assert!(
            ctx.nodes * 64 < plain_nodes,
            "ordering should cut the tree hard: {} vs {plain_nodes}",
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
            &mut output,
            false,
        );
        assert!(result.best_move.is_some());
        assert_eq!(result.nodes, 20);
        assert!(result.qnodes > 0, "each leaf runs quiescence");
    }
}
