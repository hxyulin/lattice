use std::fmt::Write;

use crate::{PieceType, Square};
use nonmax::NonMaxU16;

/// The kind of move, stored in the high 4 bits of a [`Move`].
///
/// # Notes
/// The encoding is bit-structured: `& 4` marks a capture (including en passant
/// and promotion-captures) and `& 8` marks a promotion, with the promoted piece
/// in the low 2 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MoveFlag {
    /// A non-capturing, non-special move.
    Quiet = 0,
    /// A pawn's two-square advance (sets the en-passant target).
    DoublePawnPush = 1,
    /// Kingside castle (O-O).
    KingCastle = 2,
    /// Queenside castle (O-O-O).
    QueenCastle = 3,
    /// A capture.
    Capture = 4,
    /// An en-passant capture.
    EnPassant = 5,
    /// Promotion to a knight.
    PromoKnight = 8,
    /// Promotion to a bishop.
    PromoBishop = 9,
    /// Promotion to a rook.
    PromoRook = 10,
    /// Promotion to a queen.
    PromoQueen = 11,
    /// Capture with promotion to a knight.
    PromoKnightCapture = 12,
    /// Capture with promotion to a bishop.
    PromoBishopCapture = 13,
    /// Capture with promotion to a rook.
    PromoRookCapture = 14,
    /// Capture with promotion to a queen.
    PromoQueenCapture = 15,
}

impl MoveFlag {
    /// The raw 4-bit discriminant.
    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Does this move capture (ordinary, en passant, or promotion-capture)?
    #[inline]
    #[must_use]
    pub const fn is_capture(self) -> bool {
        (self.as_u8() & 4) != 0
    }

    /// Is this a promotion (capturing or not)?
    #[inline]
    #[must_use]
    pub const fn is_promotion(self) -> bool {
        (self.as_u8() & 8) != 0
    }

    /// The piece a promotion promotes to, or `None` for a non-promotion.
    ///
    /// # Notes
    /// The promoted type lives in the low two bits: `0..4` => knight, bishop,
    /// rook, queen - i.e. `PieceType` discriminants `1..5`.
    #[inline]
    #[must_use]
    pub const fn promoted_piece(self) -> Option<crate::PieceType> {
        if self.is_promotion() {
            Some(crate::PieceType::from_u8((self.as_u8() & 3) + 1))
        } else {
            None
        }
    }

    /// Build a flag from its raw discriminant.
    ///
    /// # Notes
    /// The `const` core that [`From<u8>`] delegates to; `6` and `7` are invalid
    /// encodings.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        debug_assert!(value < 16 && value != 6 && value != 7, "invalid move flag");
        match value {
            0 => MoveFlag::Quiet,
            1 => MoveFlag::DoublePawnPush,
            2 => MoveFlag::KingCastle,
            3 => MoveFlag::QueenCastle,
            4 => MoveFlag::Capture,
            5 => MoveFlag::EnPassant,
            8 => MoveFlag::PromoKnight,
            9 => MoveFlag::PromoBishop,
            10 => MoveFlag::PromoRook,
            11 => MoveFlag::PromoQueen,
            12 => MoveFlag::PromoKnightCapture,
            13 => MoveFlag::PromoBishopCapture,
            14 => MoveFlag::PromoRookCapture,
            _ => MoveFlag::PromoQueenCapture,
        }
    }
}

impl From<u8> for MoveFlag {
    #[inline]
    fn from(value: u8) -> Self {
        Self::from_u8(value)
    }
}

/// A move packed into 16 bits: src in 0..6, dst in 6..12, [`MoveFlag`] in 12..16.
///
/// # Performance
/// `NonMaxU16` is usable as the niche: `0xFFFF` decodes to h8h8, which is
/// neither a legal move nor the null move (0x0000 = a1a1).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(NonMaxU16);

impl Move {
    /// The NULL move, encoded as a1->a1 (value 0x0000).
    ///
    /// Used by null-move pruning in search.
    pub const NULL: Move = Move::new(
        Square::from_index(0),
        Square::from_index(0),
        MoveFlag::Quiet,
    );

    /// Pack a move from its source square, destination square, and kind.
    #[must_use]
    pub const fn new(src: Square, dst: Square, flag: MoveFlag) -> Self {
        let bits = (src.index() as u16) | ((dst.index() as u16) << 6) | ((flag as u16) << 12);
        debug_assert!(
            bits != u16::MAX,
            "0xFFFF (h8h8) is reserved as the Option<Move> niche"
        );
        // SAFETY: bits == 0xFFFF only when src == dst == 63; a move never has
        // equal src and dst, so the niche value is unreachable here.
        Self(unsafe { NonMaxU16::new_unchecked(bits) })
    }

    /// The square the moving piece starts on.
    #[inline]
    #[must_use]
    pub const fn from(&self) -> Square {
        Square::from_index((self.0.get() & 63) as u8)
    }

    /// The square the moving piece ends on.
    #[inline]
    #[must_use]
    pub const fn to(&self) -> Square {
        Square::from_index(((self.0.get() >> 6) & 63) as u8)
    }

    /// Is this the null move (source == destination)? True for [`Move::NULL`].
    #[inline]
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.from().index() == self.to().index()
    }

    /// The move's kind (capture, promotion, castle, ...).
    #[inline]
    #[must_use]
    pub const fn flag(&self) -> MoveFlag {
        MoveFlag::from_u8((self.0.get() >> 12) as u8)
    }
}

// `Option<Move>` stays two bytes via the 0xFFFF niche on `NonMaxU16`.
const _: () = assert!(std::mem::size_of::<Option<Move>>() == 2);

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(pt) = self.flag().promoted_piece() {
            f.write_char(match pt {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => unreachable!(),
            })?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Move")
            .field("src", &self.from())
            .field("dest", &self.to())
            .field("flag", &self.flag())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_move_is_recognized() {
        assert!(Move::NULL.is_null());
        assert_eq!(Move::NULL.from(), Move::NULL.to());
        let real = Move::new(
            Square::from_index(12),
            Square::from_index(28),
            MoveFlag::Quiet,
        );
        assert!(!real.is_null());
        // NULL is distinct from the Option niche - Option<Move> still fits in 2 bytes.
        assert!(Some(Move::NULL).is_some());
    }

    #[test]
    fn test_move_flag_is_capture() {
        assert!(!MoveFlag::Quiet.is_capture());
        assert!(MoveFlag::Capture.is_capture());
        assert!(MoveFlag::from(4).is_capture());
        assert!(MoveFlag::from(5).is_capture()); // EnPassant
    }

    #[test]
    fn test_move_flag_is_promotion() {
        assert!(!MoveFlag::Quiet.is_promotion());
        assert!(MoveFlag::PromoKnight.is_promotion());
        assert!(MoveFlag::from(8).is_promotion());
        assert!(MoveFlag::from(9).is_promotion());
    }

    #[test]
    fn test_promo_capture_flags() {
        let c_prom_kn = MoveFlag::from(12);
        assert!(c_prom_kn.is_capture());
        assert!(c_prom_kn.is_promotion());

        let c_prom_q = MoveFlag::from(15);
        assert!(c_prom_q.is_capture());
        assert!(c_prom_q.is_promotion());

        let ep_capture = MoveFlag::from(5);
        assert!(ep_capture.is_capture());
        assert!(!ep_capture.is_promotion());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_moveflag_should_panic() {
        let _ = MoveFlag::from(100);
    }

    #[test]
    fn test_move_creation_and_decompression() {
        let src_s = Square::new(1, 4); // e2
        let dst_s = Square::new(3, 4); // e4
        let quiet_move = Move::new(src_s, dst_s, MoveFlag::Quiet);

        assert_eq!(quiet_move.from(), src_s);
        assert_eq!(quiet_move.to(), dst_s);
        assert_eq!(quiet_move.flag(), MoveFlag::Quiet);

        let expected_u16 = (src_s.index() as u16)
            | ((dst_s.index() as u16) << 6)
            | (MoveFlag::Quiet.as_u8() as u16) << 12;
        assert_eq!(quiet_move.0.get(), expected_u16);
    }

    #[test]
    fn test_move_capture_and_flagging() {
        let src = Square::new(6, 0); // a7
        let dst = Square::new(7, 7); // h8
        let capture_move = Move::new(src, dst, MoveFlag::Capture);
        assert_eq!(capture_move.from(), src);
        assert_eq!(capture_move.to(), dst);
        assert!(capture_move.flag().is_capture());

        let src_promo = Square::new(6, 1); // b7
        let dst_promo = Square::new(7, 1); // b8
        let promo_move = Move::new(src_promo, dst_promo, MoveFlag::PromoQueenCapture);
        assert!(promo_move.flag().is_promotion());
        assert!(promo_move.flag().is_capture());
    }

    #[test]
    fn test_move_full_range_sanity_check() {
        let src_h8 = Square::new(7, 7);
        let dst_a1 = Square::new(0, 0);

        let quiet_move = Move::new(src_h8, dst_a1, MoveFlag::Quiet);
        assert_eq!(quiet_move.from(), src_h8);
        assert_eq!(quiet_move.to(), dst_a1);

        let capture_move = Move::new(src_h8, dst_a1, MoveFlag::Capture);
        assert!(capture_move.flag().is_capture());
    }
}
