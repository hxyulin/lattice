//! Static position evaluation.

use crate::{Board, Color, PieceType};

const VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

/// Returns the static evaluation in centipawns, relative to the side to move.
pub fn evaluate(board: &Board) -> i32 {
    let mut score = 0;
    for piece_type in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ] {
        let value = VALUES[piece_type as usize];
        score += value * (board.pieces(piece_type) & board.color(Color::White)).count() as i32;
        score -= value * (board.pieces(piece_type) & board.color(Color::Black)).count() as i32;
    }
    if board.state().side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_positions_are_zero_for_either_side() {
        assert_eq!(evaluate(&Board::startpos()), 0);
        let white: Board = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        let black: Board = "4k3/8/8/8/8/8/8/4K3 b - - 0 1".parse().unwrap();
        assert_eq!(evaluate(&white), 0);
        assert_eq!(evaluate(&black), 0);
    }

    #[test]
    fn material_delta_is_relative_to_side_to_move() {
        let white: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
            .parse()
            .unwrap();
        let black: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
            .parse()
            .unwrap();
        assert_eq!(evaluate(&white), -900);
        assert_eq!(evaluate(&black), 900);
    }
}
