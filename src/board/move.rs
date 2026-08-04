use core::num::NonZeroU16;

use super::{PieceType, Square};

/// The encoded kind of a chess move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
    /// A non-capturing move.
    Quiet = 0b0000,
    /// A two-square pawn advance.
    DoublePawnPush = 0b0001,
    /// Kingside castling.
    KingCastle = 0b0010,
    /// Queenside castling.
    QueenCastle = 0b0011,
    /// A capture.
    Capture = 0b0100,
    /// An en passant capture.
    EnPassant = 0b0101,
    /// Reserved move type 6.
    Reserved6 = 0b0110,
    /// Reserved move type 7.
    Reserved7 = 0b0111,
    /// A knight promotion.
    KnightPromo = 0b1000,
    /// A bishop promotion.
    BishopPromo = 0b1001,
    /// A rook promotion.
    RookPromo = 0b1010,
    /// A queen promotion.
    QueenPromo = 0b1011,
    /// A knight promotion capture.
    KnightPromoCap = 0b1100,
    /// A bishop promotion capture.
    BishopPromoCap = 0b1101,
    /// A rook promotion capture.
    RookPromoCap = 0b1110,
    /// A queen promotion capture.
    QueenPromoCap = 0b1111,
}

impl MoveType {
    /// Builds a move type from the low four bits of a byte.
    pub fn from_nibble(nibble: u8) -> Self {
        match nibble & 0x0f {
            0 => Self::Quiet,
            1 => Self::DoublePawnPush,
            2 => Self::KingCastle,
            3 => Self::QueenCastle,
            4 => Self::Capture,
            5 => Self::EnPassant,
            6 => Self::Reserved6,
            7 => Self::Reserved7,
            8 => Self::KnightPromo,
            9 => Self::BishopPromo,
            10 => Self::RookPromo,
            11 => Self::QueenPromo,
            12 => Self::KnightPromoCap,
            13 => Self::BishopPromoCap,
            14 => Self::RookPromoCap,
            15 => Self::QueenPromoCap,
            _ => unreachable!(),
        }
    }

    /// Returns whether the move type captures a piece.
    pub fn is_capture(self) -> bool {
        self as u8 & 0b0100 != 0
    }

    /// Returns whether the move type promotes a pawn.
    pub fn is_promotion(self) -> bool {
        self as u8 & 0b1000 != 0
    }

    /// Returns the promoted piece type, if this is a promotion.
    pub fn promoted_piece(self) -> Option<PieceType> {
        self.is_promotion()
            .then(|| PieceType::from_index(PieceType::Knight as u8 + (self as u8 & 0b0011)))
    }
}

/// A chess move packed into two bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move(NonZeroU16);

impl Move {
    /// Builds a move from its origin, destination, and type.
    ///
    /// `from` and `to` must differ, and `move_type` must not be a reserved
    /// variant; both are debug-asserted. The packed value is never zero
    /// because that would require `from == to == a1`.
    pub fn new(from: Square, to: Square, move_type: MoveType) -> Self {
        debug_assert!(from != to);
        debug_assert!(!matches!(
            move_type,
            MoveType::Reserved6 | MoveType::Reserved7
        ));
        let value = (move_type as u16 & 0xf) << 12
            | (from.index() as u16 & 0x3f) << 6
            | (to.index() as u16 & 0x3f);
        Self(NonZeroU16::new(value).expect("from != to guarantees a nonzero move"))
    }

    /// Returns the origin square.
    pub fn from(self) -> Square {
        Square::new_unchecked(((self.0.get() >> 6) & 0x3f) as u8)
    }

    /// Returns the destination square.
    pub fn to(self) -> Square {
        Square::new_unchecked((self.0.get() & 0x3f) as u8)
    }

    /// Returns the encoded move type.
    pub fn move_type(self) -> MoveType {
        MoveType::from_nibble((self.0.get() >> 12) as u8)
    }

    /// Returns whether the move captures a piece.
    pub fn is_capture(self) -> bool {
        self.move_type().is_capture()
    }

    /// Returns whether the move promotes a pawn.
    pub fn is_promotion(self) -> bool {
        self.move_type().is_promotion()
    }

    /// Returns the promoted piece type, if this is a promotion.
    pub fn promoted_piece(self) -> Option<PieceType> {
        self.move_type().promoted_piece()
    }
}

#[cfg(test)]
mod tests {
    use super::{Move, MoveType};
    use crate::{PieceType, Square};

    const MOVE_TYPES: [MoveType; 16] = [
        MoveType::Quiet,
        MoveType::DoublePawnPush,
        MoveType::KingCastle,
        MoveType::QueenCastle,
        MoveType::Capture,
        MoveType::EnPassant,
        MoveType::Reserved6,
        MoveType::Reserved7,
        MoveType::KnightPromo,
        MoveType::BishopPromo,
        MoveType::RookPromo,
        MoveType::QueenPromo,
        MoveType::KnightPromoCap,
        MoveType::BishopPromoCap,
        MoveType::RookPromoCap,
        MoveType::QueenPromoCap,
    ];
    const VALID_MOVE_TYPES: [MoveType; 14] = [
        MoveType::Quiet,
        MoveType::DoublePawnPush,
        MoveType::KingCastle,
        MoveType::QueenCastle,
        MoveType::Capture,
        MoveType::EnPassant,
        MoveType::KnightPromo,
        MoveType::BishopPromo,
        MoveType::RookPromo,
        MoveType::QueenPromo,
        MoveType::KnightPromoCap,
        MoveType::BishopPromoCap,
        MoveType::RookPromoCap,
        MoveType::QueenPromoCap,
    ];

    #[test]
    fn move_types_roundtrip_all_nibbles() {
        for (nibble, move_type) in MOVE_TYPES.into_iter().enumerate() {
            assert_eq!(MoveType::from_nibble(nibble as u8), move_type);
        }
    }

    #[test]
    fn moves_roundtrip_valid_domain_exhaustively() {
        for from_index in 0..64 {
            let from = Square::from_index(from_index).unwrap();
            for to_index in 0..64 {
                let to = Square::from_index(to_index).unwrap();
                if from == to {
                    continue;
                }
                for move_type in VALID_MOVE_TYPES {
                    let chess_move = Move::new(from, to, move_type);
                    assert_eq!(chess_move.from(), from);
                    assert_eq!(chess_move.to(), to);
                    assert_eq!(chess_move.move_type(), move_type);
                    assert_eq!(chess_move.is_capture(), move_type.is_capture());
                    assert_eq!(chess_move.is_promotion(), move_type.is_promotion());
                    assert_eq!(chess_move.promoted_piece(), move_type.promoted_piece());
                }
            }
        }
    }

    #[test]
    fn move_type_bit_predicates_agree_with_variants() {
        for move_type in MOVE_TYPES {
            let nibble = move_type as u8;
            assert_eq!(move_type.is_capture(), nibble & 0b0100 != 0);
            assert_eq!(move_type.is_promotion(), nibble & 0b1000 != 0);
            assert_eq!(move_type.promoted_piece().is_some(), nibble >= 8);
            if nibble >= 8 {
                assert_eq!(
                    move_type.promoted_piece(),
                    Some(PieceType::from_index(
                        PieceType::Knight as u8 + (nibble & 0b0011)
                    ))
                );
            }
        }
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn rejects_equal_squares() {
        let square = Square::from_index(1).unwrap();
        Move::new(square, square, MoveType::Quiet);
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn rejects_reserved_6() {
        Move::new(
            Square::from_index(1).unwrap(),
            Square::from_index(2).unwrap(),
            MoveType::Reserved6,
        );
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn rejects_reserved_7() {
        Move::new(
            Square::from_index(1).unwrap(),
            Square::from_index(2).unwrap(),
            MoveType::Reserved7,
        );
    }
}
