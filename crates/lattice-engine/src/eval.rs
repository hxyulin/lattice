//! Static evaluation: material value and piece square tables.

use lattice_board::{Board, Color, PieceType};

use crate::Score;

/// Centipawn material value per piece type, indexed by `PieceType`
const MATERIAL: [Score; 6] = [100, 300, 300, 500, 900, 0];

/// Piece types in discriminant order, indexed in lockstep with `MATERIAL` and `PIECE_SQUARE_TABLES`.
const PIECES: [PieceType; 6] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::King,
];

/// Positional bonus per (piece type, square), in centipawns.
#[rustfmt::skip]
const PIECE_SQUARE_TABLES: [[Score; 64]; 6] = [
    // Pawn
    [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // Knight
    [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    // Bishop
    [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10, 0, 0, 0, 0, 0, 0,-10,
        -10, 0, 5, 10, 10, 5, 0,-10,
        -10, 5, 5, 10, 10, 5, 5,-10,
        -10, 0, 10, 10, 10, 10, 0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10, 5, 0, 0, 0, 0, 5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    // Rook
    [
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // Queen
    [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10, 0, 0, 0, 0, 0, 0,-10,
        -10, 0, 5, 5, 5, 5, 0,-10,
        -5, 0, 5, 5, 5, 5, 0,-5,
        0, 0, 5, 5, 5, 5, 0,-5,
        -10, 5, 5, 5, 5, 5, 0,-10,
        -10, 0, 5, 0, 0, 0, 0,-10,
        -20,-10,-10,-5,-5,-10,-10,-20,
    ],
    // King
    [
         20, 30, 10, 0, 0, 10, 30, 20,
         20, 20, 0, 0, 0, 0, 20, 20,
        -10, 20, 20, 20, 20, 20, 20, -10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
    ],
];

/// Static evaluation of `board`, from the side-to-move's perspective
#[must_use]
pub fn evaluate(board: &Board) -> Score {
    let mut score = 0; // from White's perspective
    for pt in PIECES {
        let i = pt as usize;
        let (value, table) = (MATERIAL[i], &PIECE_SQUARE_TABLES[i]);

        for sq in board.pieces(Color::White, pt) {
            score += value + table[sq.index() as usize];
        }
        for sq in board.pieces(Color::Black, pt) {
            score -= value + table[(sq.index() ^ 56) as usize];
        }
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
    fn a_missing_pawn_is_worth_a_pawn_plus_its_square() {
        // White to move, Black is down its e-pawn. Material is +100, but the
        // missing pawn also sat on e7 (a `-20` PST square from Black's frame),
        // so removing it nets +100 - 20 = +80 for White.
        let white_up = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(evaluate(&board(white_up)), 80);
        // Same position with Black to move: the score flips sign.
        let black_to_move = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        assert_eq!(evaluate(&board(black_to_move)), -80);
    }

    #[test]
    fn advanced_pawn_beats_home_pawn() {
        // Orientation guard: a white pawn one step from promotion must outscore
        // one on its starting square.
        let advanced = evaluate(&board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"));
        let home = evaluate(&board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"));
        assert!(advanced > home, "advanced={advanced} home={home}");
    }

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}
