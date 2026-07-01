//! Static evaluation
//!
//! Evaluate the position based on several factors and techniques, currently including:
//!  - material value

use crate::{Board, Color, PieceType};

use crate::Score;

/// Centipawn values indexed by PieceType
/// King is `0`: it cannot be captured, so it adds nothing to the material balance.
const SCORED: [(PieceType, Score); 5] = [
    (PieceType::Pawn, 100),
    (PieceType::Knight, 300),
    (PieceType::Bishop, 300),
    (PieceType::Rook, 500),
    (PieceType::Queen, 900),
];

/// Static evaluation of `board`, from the side-to-move's perspective
#[must_use]
pub fn evaluate(board: &Board) -> Score {
    let mut score = 0; // from White's perspective
    for (pt, value) in SCORED {
        let white = board.pieces(Color::White, pt).count() as Score;
        let black = board.pieces(Color::Black, pt).count() as Score;
        score += value * (white - black);
    }

    // Flip into the side-to-move's frame, the convention negamax negates across.
    if board.side_to_move() == Color::Black {
        -score
    } else {
        score
    }
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
