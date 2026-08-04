//! Iterative-deepening negamax search.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::eval::evaluate;
use crate::movegen::{MoveList, generate_legal, is_attacked};
use crate::uci::move_text;
use crate::{Board, Move};

/// Nodes between clock checks. Low enough that a small tree still checks
/// several times: depth 2 from the start position is only 440 nodes, and at
/// 2048 it never checked at all.
const CHECK_INTERVAL: u64 = 256;
const MATE: i32 = 30_000;
const INFINITY: i32 = 31_000;

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
    pub(crate) nodes: u64,
}

struct Ctx<'a> {
    limits: Limits,
    stop: &'a AtomicBool,
    deadline: Option<Instant>,
    nodes: u64,
    iteration_depth: u32,
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
        iteration_depth: 1,
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
            write_info(output, depth, score, ctx.nodes, start.elapsed(), candidate);
        }
        if candidate.is_none() {
            break;
        }
    }

    SearchResult {
        best_move,
        nodes: ctx.nodes,
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
        return Ok(evaluate(board));
    }
    let mut moves = MoveList::new();
    generate_legal(board, &mut moves);
    if moves.is_empty() {
        return Ok(terminal_score(board, ply));
    }
    let mut best = -INFINITY;
    for &mv in moves.iter() {
        board.make(mv);
        let result = negamax(board, depth - 1, ply + 1, -beta, -alpha, ctx);
        board.unmake(mv);
        best = best.max(-result?);
        alpha = alpha.max(best);
        if alpha >= beta {
            break;
        }
    }
    Ok(best)
}

impl Ctx<'_> {
    fn check_abort(&self) -> Result<(), Aborted> {
        if self.iteration_depth == 1 || !self.nodes.is_multiple_of(CHECK_INTERVAL) {
            return Ok(());
        }
        if self.out_of_time() {
            Err(Aborted)
        } else {
            Ok(())
        }
    }

    fn out_of_time(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self.limits.nodes.is_some_and(|limit| self.nodes >= limit)
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

    /// Unpruned negamax, kept as the oracle for `pruning_preserves_scores`.
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

            let mut ctx = Ctx {
                limits: Limits::default(),
                stop: &AtomicBool::new(false),
                deadline: None,
                nodes: 0,
                iteration_depth: 3,
            };
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
    }
}
