//! Recursive search of game state using the following techniques to speed up and rank moves:
//!  - Negamax (minimax)
//!  - Alpha-beta pruning
//!  - MVV-LVA capture move ordering
//!  - Iterative-deepening

use std::time::{Duration, Instant};

use lattice_board::{Board, Move, MoveFlag, MoveList, PieceType};

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
    /// Node budget
    pub nodes: Option<u64>,
    /// Wall-clock budget for the whole search.
    pub move_time: Option<Duration>,
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

/// Search `board` under `limits` and return the best move with its score.
///
/// Iterative deepening drives the search; it stops at whichever of the depth,
/// node, and time limits fires first. A bare [`Limits::default`] runs to
/// [`MAX_PLY`].
#[must_use]
pub fn search(board: &mut Board, limits: &Limits) -> SearchResult {
    let mut searcher = Searcher {
        nodes: 0,
        node_limit: limits.nodes,
        deadline: limits.move_time.map(|t| Instant::now() + t),
        stopped: false,
        armed: false,
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
        depth: completed,
    }
}

/// Mutable search state threaded through the recursion
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
            return evaluate(board);
        }

        let mut best = -MATE;
        let mut legal = 0u32;

        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        if depth >= ORDER_MIN_DEPTH {
            // No hint at interior nodes yet - per-node best moves need a
            // transposition table, which is the next step after this.
            moves.sort_by_key(|&m| -(order_score(board, m, None)));
        }
        for mv in &moves {
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

/// Skip MVV-LVA ordering at remaining depth below this.
///
/// # Performance
/// The depth-1 frontier is the largest node layer but its children are leaves
/// (static eval), so sorting there costs more than the sibling evals it saves.
const ORDER_MIN_DEPTH: u32 = 2;

const VAL: [i32; 6] = [100, 320, 330, 500, 900, 0];

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

    #[test]
    fn grabs_a_hanging_queen() {
        // White pawn e2 can capture an undefended Black queen on d3.
        let mut b = board("4k3/8/8/8/8/3q4/4P3/4K3 w - - 0 1");
        let r = search(&mut b, &Limits::to_depth(1));
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
