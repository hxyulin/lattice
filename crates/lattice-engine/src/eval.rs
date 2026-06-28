//! Static evaluation
//!
//! Evaluate the position based on several factors and techniques, currently including:
//!  - material value

use lattice_board::Board;

use crate::Score;

/// Centipawn values indexed by PieceType
/// King is `0`: it cannot be captured, so it adds nothing to the material balance.
const PIECE_VALUES: [Score; 6] = [100, 300, 300, 500, 900, 0];

/// Static evaluation of `board`, from the side-to-move's perspective
#[must_use]
pub fn evaluate(board: &Board) -> Score {
    let us = board.side_to_move();
    let mut score = 0;

    for (_square, piece) in board.piece_iter() {
        let value = PIECE_VALUES[piece.piece() as usize];
        // Branchless sign: +1 for our pieces, -1 for theirs.
        let sign = (piece.color() == us) as Score * 2 - 1;
        score += value * sign;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen.as_bytes()).unwrap()
    }

    #[test]
    fn startpos_is_balanced() {
        assert_eq!(evaluate(&board(STARTPOS)), 0);
    }

    #[test]
    fn a_missing_pawn_is_worth_100() {
        // White to move, Black is down its e-pawn -> +100 for White.
        let white_up = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(evaluate(&board(white_up)), 100);
        // Same position with Black to move: the score flips sign.
        let black_to_move = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        assert_eq!(evaluate(&board(black_to_move)), -100);
    }

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}
