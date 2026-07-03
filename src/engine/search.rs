//! Recursive search of game state using the following techniques to speed up and rank moves:
//!  - Negamax (minimax)
//!  - Iterative deepening with a wall-clock deadline

use std::time::Instant;

use crate::{Board, Move};

use crate::{MATE, Score, evaluate};

/// Any score at least this close to `MATE` is a mate score; the search never
/// reaches this many plies, so it cannot be a real centipawn evaluation.
const MATE_THRESHOLD: Score = MATE - 1000;

/// How often (in nodes) the clock is polled. A power of two so the check is a
/// cheap bit mask; small enough that overshoot past the deadline is bounded.
const CHECK_INTERVAL: u64 = 2048;

/// The outcome of a [`search`].
pub struct SearchResult {
    /// The best move, or `None` only when the side to move has no legal moves
    pub best_move: Option<Move>,
    /// Score of the position from the side-to-move's perspective
    pub score: Score,
    /// Nodes visited during the search
    pub nodes: u64,
    /// Deepest iteration fully completed (0 for a bare static evaluation).
    pub depth: u32,
}

/// Search `board` to a fixed `depth` and return the best move with its score.
///
/// `depth` is the number of plies to look ahead; `depth == 0` just statically
/// evaluates the position and returns no move. This is the deterministic entry
/// point used by the benchmark and tests; timed play uses [`search_deadline`].
#[must_use]
pub fn search(board: &mut Board, depth: u32) -> SearchResult {
    if depth == 0 {
        return SearchResult {
            best_move: None,
            score: evaluate(board),
            nodes: 1,
            depth: 0,
        };
    }

    let mut searcher = Searcher::new(None);
    let (best_move, score) = searcher.search_root(board, depth);
    SearchResult {
        best_move,
        score,
        nodes: searcher.nodes,
        depth,
    }
}

/// Iterative deepening: search depth 1, 2, 3, ... keeping the best move from the
/// deepest iteration that completed before `deadline`. A partially searched
/// depth is discarded, so the returned move is always from a full search.
///
/// `max_depth` caps the iteration. `deadline == None` searches every depth up to
/// `max_depth` with no time limit.
#[must_use]
pub fn search_deadline(
    board: &mut Board,
    max_depth: u32,
    deadline: Option<Instant>,
) -> SearchResult {
    let mut searcher = Searcher::new(deadline);
    let mut result = SearchResult {
        best_move: None,
        score: 0,
        nodes: 0,
        depth: 0,
    };

    for depth in 1..=max_depth {
        let (best_move, score) = searcher.search_root(board, depth);
        // Ran out of time mid-iteration: drop it and keep the last full depth.
        if searcher.stopped {
            break;
        }
        result.best_move = best_move;
        result.score = score;
        result.depth = depth;
        result.nodes = searcher.nodes;
        // A forced mate is exact at any depth; searching deeper cannot improve it.
        if score.abs() >= MATE_THRESHOLD {
            break;
        }
    }

    result
}

/// Mutable search state threaded through the recursion.
struct Searcher {
    nodes: u64,
    /// Wall-clock stop, or `None` for an untimed (fixed-depth) search.
    deadline: Option<Instant>,
    /// Set once the deadline passes; every node unwinds without expanding.
    stopped: bool,
}

impl Searcher {
    fn new(deadline: Option<Instant>) -> Self {
        Self {
            nodes: 0,
            deadline,
            stopped: false,
        }
    }

    /// Poll the clock every `CHECK_INTERVAL` nodes and latch `stopped`.
    fn check_time(&mut self) {
        if self.nodes.is_multiple_of(CHECK_INTERVAL)
            && let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            self.stopped = true;
        }
    }

    /// Root search: like [`Self::negamax`] but records which move scored best.
    /// Returns `(best_move, score)`; `best_move` is `None` only at a terminal
    /// (mate/stalemate) position or if the search stops before a legal move.
    fn search_root(&mut self, board: &mut Board, depth: u32) -> (Option<Move>, Score) {
        let mut best_move = None;
        let mut best = -MATE;

        let mut alpha = -MATE;
        let beta = MATE;

        for mv in &board.pseudo_legal_moves() {
            let undo = board.make_move(*mv);
            if board.is_legal() {
                let score = -self.negamax(board, depth - 1, 1, -beta, -alpha);
                if score > best {
                    best = score;
                    best_move = Some(*mv);
                }

                alpha = alpha.max(score);
                if alpha >= beta {
                    // Beta cutoff: the opponent has a better option, so this
                    // branch will never be reached.
                    board.unmake_move(*mv, undo);
                    break;
                }
            }
            board.unmake_move(*mv, undo);
            if self.stopped {
                break;
            }
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
        self.check_time();
        if self.stopped {
            return 0;
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

                // The alpha-beta window is [alpha, beta), from the opponents point of view it is:
                // [-beta, -alpha), therefore we negate the score and swap alpha and beta.
                let score = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha);
                best = best.max(score);
                alpha = alpha.max(score);

                if alpha >= beta {
                    // Beta cutoff: the opponent has a better option, so this
                    // branch will never be reached.
                    board.unmake_move(*mv, undo);
                    break;
                }
            }
            board.unmake_move(*mv, undo);
            if self.stopped {
                return best;
            }
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

    #[test]
    fn grabs_a_hanging_queen() {
        // White pawn e2 can capture an undefended Black queen on d3.
        let mut b = board("4k3/8/8/8/8/3q4/4P3/4K3 w - - 0 1");
        let r = search(&mut b, 1);
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
        let r = search(&mut b, 2);
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
        let r = search(&mut b, 1);
        assert_eq!(r.best_move, None);
        assert_eq!(r.score, 0);
    }

    #[test]
    fn iterative_deepening_reaches_max_depth() {
        // With no deadline, ID must complete every depth up to the cap.
        let mut b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let r = search_deadline(&mut b, 4, None);
        assert_eq!(r.depth, 4);
        assert!(r.best_move.is_some());
    }

    #[test]
    fn iterative_deepening_finds_mate_in_one() {
        let mut b = board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        let r = search_deadline(&mut b, 8, None);
        assert_eq!(
            r.best_move.map(|m| (m.from(), m.to())),
            Some((sq("a1"), sq("a8")))
        );
        assert_eq!(r.score, MATE - 1);
    }

    #[test]
    fn expired_deadline_still_returns_a_move() {
        // A deadline already in the past must not yield a null move: depth 1 is
        // tiny enough to finish before the clock is ever polled.
        let mut b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let r = search_deadline(&mut b, 64, Some(Instant::now()));
        assert!(r.best_move.is_some());
        assert!(r.depth >= 1);
    }
}
