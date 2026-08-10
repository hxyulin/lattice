use crate::{Color, Piece, PieceType, Square};

/// Number of vanilla HalfKP features: 64 king squares, ten relative
/// non-king piece categories, and 64 piece squares.
pub const FEATURES: usize = 64 * 10 * 64;
/// Number of absolute piece-square inputs in the compact Chess768 ABI.
pub const CHESS768_FEATURES: usize = 64 * 6 * 2;

/// Returns the HalfKP feature for a piece from one king's perspective.
/// Kings are anchors rather than active features and return `None`.
pub fn feature_index(
    perspective: Color,
    king: Square,
    piece: Piece,
    square: Square,
) -> Option<usize> {
    if piece.piece_type() == PieceType::King {
        return None;
    }
    let orient = |sq: Square| match perspective {
        Color::White => sq.index() as usize,
        Color::Black => sq.flip_rank().index() as usize,
    };
    let relative_color = usize::from(piece.color() != perspective);
    let piece_bucket = piece.piece_type() as usize + 5 * relative_color;
    Some(orient(square) + 64 * (piece_bucket + 10 * orient(king)))
}

/// Returns Chess768's white- and black-perspective feature indices.
pub(crate) fn chess768_indices(piece: Piece, square: Square) -> [usize; 2] {
    let color = piece.color() as usize;
    let kind = piece.piece_type() as usize;
    let white = color * 384 + kind * 64 + square.index() as usize;
    let black = (color ^ 1) * 384 + kind * 64 + square.flip_rank().index() as usize;
    [white, black]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_indices_fix_the_feature_abi() {
        let e1 = Square::new(4, 0).unwrap();
        let e8 = Square::new(4, 7).unwrap();
        let a2 = Square::new(0, 1).unwrap();
        let pawn = Piece::new(Color::White, PieceType::Pawn);

        assert_eq!(feature_index(Color::White, e1, pawn, a2), Some(2_568));
        assert_eq!(feature_index(Color::Black, e8, pawn, a2), Some(2_928));
        assert!(
            feature_index(
                Color::White,
                e1,
                Piece::new(Color::White, PieceType::King),
                e1
            )
            .is_none()
        );
    }

    #[test]
    fn every_legal_category_stays_in_range() {
        for perspective in [Color::White, Color::Black] {
            for king_index in 0..64 {
                let king = Square::new_unchecked(king_index);
                for color in [Color::White, Color::Black] {
                    for kind in [
                        PieceType::Pawn,
                        PieceType::Knight,
                        PieceType::Bishop,
                        PieceType::Rook,
                        PieceType::Queen,
                    ] {
                        for square_index in 0..64 {
                            let index = feature_index(
                                perspective,
                                king,
                                Piece::new(color, kind),
                                Square::new_unchecked(square_index),
                            )
                            .unwrap();
                            assert!(index < FEATURES);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chess768_indices_match_bullets_absolute_perspectives() {
        let a2 = Square::new(0, 1).unwrap();
        let white_pawn = Piece::new(Color::White, PieceType::Pawn);
        assert_eq!(chess768_indices(white_pawn, a2), [8, 432]);
        let black_queen = Piece::new(Color::Black, PieceType::Queen);
        assert_eq!(chess768_indices(black_queen, a2), [648, 304]);
    }
}
