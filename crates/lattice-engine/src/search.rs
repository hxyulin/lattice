//! Recursive search of game state using the following techniques to speed up and rank moves:
//!  - Negamax (minimax)
//!  - Alpha-beta pruning

use lattice_board::{Board, Move};

use crate::{MATE, Score, evaluate};

/// The outcome of a [`search`].
pub struct SearchResult {
    /// The best move, or `None` only when the side to move has no legal moves
    pub best_move: Option<Move>,
    /// Score of the position from the side-to-move's perspective
    pub score: Score,
    /// Nodes visited during the search
    pub nodes: u64,
}

/// Search `board` to `depth` plies and return the best move with its score.
///
/// `depth` is the number of plies to look ahead; `depth == 0` just statically
/// evaluates the position and returns no move.
#[must_use]
pub fn search(board: &mut Board, depth: u32) -> SearchResult {
    if depth == 0 {
        return SearchResult {
            best_move: None,
            score: evaluate(board),
            nodes: 1,
        };
    }

    let mut searcher = Searcher { nodes: 0 };
    let mut best_move = None;
    let mut best = -MATE;

    // same as negamax loop but records the best move
    for mv in &board.pseudo_legal_moves() {
        let undo = board.make_move(*mv);
        if board.is_current_state_legal() {
            let score = -searcher.negamax(board, depth - 1, 1, -MATE, MATE);
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

    SearchResult {
        best_move,
        score,
        nodes: searcher.nodes,
    }
}

/// Mutable search state threaded through the recursion.
struct Searcher {
    nodes: u64,
}

impl Searcher {
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

        if depth == 0 {
            return evaluate(board);
        }

        let mut best = -MATE;
        let mut legal = 0u32;

        for mv in &board.pseudo_legal_moves() {
            let undo = board.make_move(*mv);
            if board.is_current_state_legal() {
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
        let r = search(&mut b, 1);
        let mv = r.best_move.expect("a legal move exists");
        assert_eq!(mv.src(), sq("e2"));
        assert_eq!(mv.dest(), sq("d3"));
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
            r.best_move.map(|m| (m.src(), m.dest())),
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
}
