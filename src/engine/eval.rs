//! Static evaluation
//!
//! Evaluate the position based on several factors and techniques, currently including:
//!  - material value
//!  - piece-square tables

use crate::{Board, Color, PieceType};

use crate::Score;

/// Piece-square tables indexed by `PieceType` discriminant.
///
/// Each table is written visually from White's perspective (rank 8 on the
/// first line, a-file on the left), so lookups mirror the LERF square for
/// White (`sq ^ 56`) and use it directly for Black.
#[rustfmt::skip]
const PIECE_SQUARE_TABLES: [[Score; 64]; 6] = [
    [
          0,   0,   0,   0,   0,   0,   0,   0,
         28,  30,  32,  34,  34,  32,  30,  28,
         22,  22,  25,  30,  30,  25,  22,  22,
         15,  15,  18,  20,  20,  18,  15,  15,
         10,  12,  14,  18,  18,  14,  12,  10,
          5,   8,  10,  10,  10,  10,   8,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    [
        -40, -30, -20, -20, -20, -20, -30, -40,
        -30, -10,   0,   5,   5,   0, -10, -30,
        -20,   5,  10,  15,  15,  10,   5, -20,
        -20,   0,  15,  20,  20,  15,   0, -20,
        -20,   5,  15,  20,  20,  15,   5, -20,
        -20,   0,  10,  15,  15,  10,   0, -20,
        -30, -10,   0,   5,   5,   0, -10, -30,
        -40, -30, -20, -20, -20, -20, -30, -40,
    ],
    [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    [
          0,   0,   0,   0,   0,   0,   0,   0,
          5,  10,  10,  10,  10,  10,  10,   5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          0,   0,   0,   5,   5,   0,   0,   0,
    ],
    [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
         -5,   0,   5,   5,   5,   5,   0,  -5,
        -10,   0,   5,   5,   5,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ],
    [
        -30, -30, -30, -30, -30, -30, -30, -30,
        -30, -30, -30, -30, -30, -30, -30, -30,
        -25, -25, -30, -30, -30, -30, -25, -25,
        -20, -20, -25, -30, -30, -25, -20, -20,
        -15, -15, -20, -25, -25, -20, -15, -15,
         -5, -10, -15, -20, -20, -15, -10,  -5,
         10,  10,  -5, -10, -10,  -5,  10,  10,
         15,  20,  10,   0,   5,   0,  25,  15,
    ],
];

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

    for (idx, table) in PIECE_SQUARE_TABLES.iter().enumerate() {
        let pt = PieceType::from_u8(idx as u8);
        for sq in board.pieces(Color::White, pt).iter() {
            score += table[(sq.index() ^ 56) as usize];
        }
        for sq in board.pieces(Color::Black, pt).iter() {
            score -= table[sq.index() as usize];
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
    fn a_missing_pawn_is_worth_100() {
        // White to move, Black is down its e-pawn -> +100 for White.
        let white_up = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(evaluate(&board(white_up)), 100);
        // Same position with Black to move: the score flips sign.
        let black_to_move = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        assert_eq!(evaluate(&board(black_to_move)), -100);
    }

    #[test]
    fn centralized_knight_beats_rim_knight() {
        let center = "4k3/8/8/4N3/8/8/8/4K3 w - - 0 1";
        let rim = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1";
        assert!(evaluate(&board(center)) > evaluate(&board(rim)));
    }

    #[test]
    fn pst_is_color_symmetric() {
        // Mirrored knights: White on f3, Black on f6. PST must cancel out.
        let mirrored = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1";
        assert_eq!(evaluate(&board(mirrored)), 0);
    }

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}
