use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

use crate::{Bitboard, Color, Move, MoveFlag, Piece, PieceType, Square};

/// The four castling rights as a bitset (one bit each: white/black x
/// king/queen side). Combine and test with the bitwise operators.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CastlingRights(u8);

impl CastlingRights {
    /// No castling rights.
    pub const NONE: CastlingRights = Self::new();
    /// White may castle kingside (O-O), bit 0.
    pub const WHITE_KINGSIDE: CastlingRights = Self::from_u8(1 << 0);
    /// White may castle queenside (O-O-O), bit 1.
    pub const WHITE_QUEENSIDE: CastlingRights = Self::from_u8(1 << 1);
    /// Black may castle kingside (O-O), bit 2.
    pub const BLACK_KINGSIDE: CastlingRights = Self::from_u8(1 << 2);
    /// Black may castle queenside (O-O-O), bit 3.
    pub const BLACK_QUEENSIDE: CastlingRights = Self::from_u8(1 << 3);
    /// All four rights set.
    pub const ALL: CastlingRights = Self::from_u8(0b1111);

    /// An empty set of rights (same as [`NONE`](Self::NONE)).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Wrap a raw 4-bit mask. Asserts no bits above the low four are set.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        assert!(value <= 0xF);
        Self(value)
    }

    /// Does this set contain every right in `right`?
    #[inline]
    #[must_use]
    pub fn contains(self, right: CastlingRights) -> bool {
        (self & right) != Self::NONE
    }
}

impl std::fmt::Debug for CastlingRights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CastlingRights(")?;
        const RIGHTS: &[(CastlingRights, &str)] = &[
            (CastlingRights::WHITE_KINGSIDE, "WhiteKingSide"),
            (CastlingRights::WHITE_QUEENSIDE, "WhiteQueenSide"),
            (CastlingRights::BLACK_KINGSIDE, "BlackKingSide"),
            (CastlingRights::BLACK_QUEENSIDE, "BlackQueenSide"),
        ];
        let mut written = false;
        for (r, s) in RIGHTS {
            if self.contains(*r) {
                if written {
                    write!(f, "|")?;
                }
                write!(f, "{}", s)?;
                written = true;
            }
        }
        if !written {
            write!(f, "None")?;
        }
        write!(f, ")")
    }
}

impl BitOr for CastlingRights {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for CastlingRights {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0
    }
}

impl BitAnd for CastlingRights {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for CastlingRights {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0
    }
}

impl Not for CastlingRights {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0 & 0xF)
    }
}

/// The chessboard state, including both bitboards and a mailbox for O(1) piece lookup.
///
/// The bitboards are the authoritative representation; the mailbox is updated on piece updates.
///
/// # Performance
/// Board is set to an 128-byte alignment to allow for cache-line alignment (64-128 bytes on most
/// machines).
#[repr(align(128))]
#[derive(Clone, PartialEq, Eq)]
pub struct Board {
    /// Piece bitboards, index corresponds with [`Piece::as_u8']
    bitboards: [Bitboard; 12],
    en_passent: Option<Square>,
    castling_rights: CastlingRights,
    side_to_move: Color,
    half_move_clock: u8,
    full_moves: u16,
    /// Piece mailbox to allow for O(1) piece lookup
    pieces: [Option<Piece>; 64],
}

/// The state needed to revert a [`make_move`](Board::make_move).
pub struct Undo {
    captured_piece: Option<Piece>,
    castling_rights: CastlingRights,
    en_passent: Option<Square>,
    half_move_clock: u8,
}

impl Board {
    /// Creates a new board with the starting position.
    ///
    /// # Notes
    /// Built directly from precomputed bitboards, so it is `const`.
    #[must_use]
    pub const fn starting_position() -> Self {
        let bitboards = [
            Bitboard::from_bits(0x0000_0000_0000_FF00), //  0 white pawns   (rank 2)
            Bitboard::from_bits(0x00FF_0000_0000_0000), //  1 black pawns   (rank 7)
            Bitboard::from_bits(0x0000_0000_0000_0042), //  2 white knights (b1,g1)
            Bitboard::from_bits(0x4200_0000_0000_0000), //  3 black knights (b8,g8)
            Bitboard::from_bits(0x0000_0000_0000_0024), //  4 white bishops (c1,f1)
            Bitboard::from_bits(0x2400_0000_0000_0000), //  5 black bishops (c8,f8)
            Bitboard::from_bits(0x0000_0000_0000_0081), //  6 white rooks   (a1,h1)
            Bitboard::from_bits(0x8100_0000_0000_0000), //  7 black rooks   (a8,h8)
            Bitboard::from_bits(0x0000_0000_0000_0008), //  8 white queen   (d1)
            Bitboard::from_bits(0x0800_0000_0000_0000), //  9 black queen   (d8)
            Bitboard::from_bits(0x0000_0000_0000_0010), // 10 white king    (e1)
            Bitboard::from_bits(0x1000_0000_0000_0000), // 11 black king    (e8)
        ];

        let mut pieces = [None; 64];
        let mut sq = 0u8;
        while sq < 64 {
            let mut idx = 0u8;
            while idx < 12 {
                if bitboards[idx as usize].contains(Square::from_index(sq)) {
                    pieces[sq as usize] = Some(Piece::from_u8(idx));
                    break;
                }
                idx += 1;
            }
            sq += 1;
        }

        Self {
            bitboards,
            en_passent: None,
            castling_rights: CastlingRights::ALL,
            side_to_move: Color::White,
            half_move_clock: 0,
            full_moves: 1,
            pieces,
        }
    }

    /// A board with no pieces, white to move, full castling rights, and the
    /// move counters at their initial values.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            bitboards: [Bitboard::EMPTY; 12],
            en_passent: None,
            castling_rights: CastlingRights::ALL,
            side_to_move: Color::White,
            half_move_clock: 0,
            full_moves: 1,
            pieces: [None; 64],
        }
    }

    /// The bitboard of all squares holding `piece`.
    #[inline]
    #[must_use]
    pub fn bitboard_for(&self, piece: Piece) -> &Bitboard {
        &self.bitboards[piece.as_u8() as usize]
    }

    /// Mutable access to the bitboard of all squares holding `piece`.
    #[inline]
    pub fn bitboard_for_mut(&mut self, piece: Piece) -> &mut Bitboard {
        &mut self.bitboards[piece.as_u8() as usize]
    }

    /// Every occupied square, regardless of color.
    #[must_use]
    pub fn occupied(&self) -> Bitboard {
        self.bitboards
            .iter()
            .fold(Bitboard::EMPTY, |acc, &bb| acc | bb)
    }

    /// Every square occupied by a piece of `color`.
    #[must_use]
    pub fn occupied_by(&self, color: Color) -> Bitboard {
        (0..6u8).fold(Bitboard::EMPTY, |acc, pt| {
            acc | self.bitboards[((pt << 1) | color as u8) as usize]
        })
    }

    /// The piece on `sq`, if any.
    ///
    /// # Performance
    /// O(1) mailbox lookup.
    #[inline]
    #[must_use]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.pieces[sq.index() as usize]
    }

    /// The side whose turn it is to move.
    #[inline]
    #[must_use]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// The en passant target square, if the last move was a double pawn push.
    #[inline]
    #[must_use]
    pub fn en_passant(&self) -> Option<Square> {
        self.en_passent
    }

    /// The castling rights still available to either side.
    #[inline]
    #[must_use]
    pub fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    /// Place a piece on `sq`, overwriting any existing piece.
    ///
    /// Updates the bitboard and the `piece` array.
    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        self.bitboard_for_mut(piece).set(sq);
        self.pieces[sq.index() as usize] = Some(piece);
    }

    /// Remove whatever piece is on `sq` (if any).
    pub fn remove_piece(&mut self, sq: Square) {
        if let Some(piece) = self.piece_at(sq) {
            self.bitboard_for_mut(piece).clear(sq);
            self.pieces[sq.index() as usize] = None;
        }
    }

    /// Is `color`'s king attacked by the opposing side?
    ///
    /// # Notes
    /// `in_check(side_to_move)` answers "am I in check"; `in_check(side_to_move.flip())`
    /// answers "did the side that just moved leave its own king in check".
    #[must_use]
    pub fn in_check(&self, color: Color) -> bool {
        let king_sq = self
            .bitboard_for(Piece::new(color, PieceType::King))
            .first_square()
            .expect("king must be on the board");
        self.is_attacked(king_sq, color.flip())
    }

    /// Is the position legal - i.e. is the side that *just moved* out of check?
    ///
    /// # Notes
    /// Call right after [`Board::make_move`]: it has already flipped `side_to_move`,
    /// so the mover is now the side not to move.
    #[must_use]
    pub fn is_current_state_legal(&self) -> bool {
        !self.in_check(self.side_to_move.flip())
    }

    /// Apply `mv` in place, returning the [`Undo`] needed to revert it.
    ///
    /// # Notes
    /// The move is assumed pseudo-legal.
    #[must_use]
    pub fn make_move(&mut self, mv: Move) -> Undo {
        let us = self.side_to_move;
        let flag = mv.flag();
        // For en passant the captured pawn is *not* on `dest`, so this is `None`;
        // it is reconstructed in `unmake_move` instead.
        let captured_piece = self.piece_at(mv.dest());
        let undo = Undo {
            captured_piece,
            castling_rights: self.castling_rights,
            en_passent: self.en_passent,
            half_move_clock: self.half_move_clock,
        };

        let mut piece = self
            .piece_at(mv.src())
            .expect("move source must be occupied");

        // Halfmove clock resets on a pawn move or any capture, else ticks up.
        if piece.is_pawn() || flag.is_capture() {
            self.half_move_clock = 0;
        } else {
            self.half_move_clock += 1;
        }

        // The en-passant target lives for exactly one ply: clear it, re-arm on double pawn push
        self.en_passent = None;

        match flag {
            MoveFlag::DoublePawnPush => {
                // En passent target rank is 3 for White, 6 for Black
                //
                // Branchless rank calculation:
                // White: 2 + (3 * 0) = 2 (Rank 3)
                // Black: 2 + (3 * 1) = 5 (Rank 6)
                let ep_rank = 2 + (3 * us.as_u8());
                self.en_passent = Some(Square::new(ep_rank, mv.src().file()));
            }
            MoveFlag::KingCastle | MoveFlag::QueenCastle => {
                let (rook_src, rook_dest) = castle_rook_squares(us, flag);
                self.remove_piece(rook_src);
                self.put_piece(rook_dest, Piece::new(us, PieceType::Rook));
            }
            MoveFlag::EnPassant => {
                // Captured pawn sits beside `dest`, on the mover's 5th/4th rank.
                // Branchless: White 4-0=4 (rank 5), Black 4-1=3 (rank 4).
                let cap_rank = 4 - us.as_u8();
                self.remove_piece(Square::new(cap_rank, mv.dest().file()));
            }
            f if f.is_promotion() => {
                piece = Piece::new(us, f.promoted_piece().expect("promotion flag"));
            }
            _ => {}
        }

        // Clear any castling right invalidated by leaving `from` or landing on `to`.
        self.castling_rights &=
            CASTLE_MASK[mv.src().index() as usize] & CASTLE_MASK[mv.dest().index() as usize];

        self.remove_piece(mv.src());
        // clears a captured piece (no-op if empty)
        self.remove_piece(mv.dest());
        self.put_piece(mv.dest(), piece);

        if us == Color::Black {
            self.full_moves += 1;
        }
        self.side_to_move = us.flip();

        undo
    }

    /// Revert a move applied by [`make_move`](Self::make_move), given its
    /// [`Undo`]. Restores the board to its exact prior state.
    pub fn unmake_move(&mut self, mv: Move, undo: Undo) {
        // Flip the side back first, so us is the correct color at the time of move
        self.side_to_move = self.side_to_move.flip();
        let us = self.side_to_move;
        if us == Color::Black {
            self.full_moves -= 1;
        }

        let flag = mv.flag();

        let moved = if flag.is_promotion() {
            Piece::new(us, PieceType::Pawn)
        } else {
            self.piece_at(mv.dest())
                .expect("move destination must be occupied")
        };

        if let MoveFlag::KingCastle | MoveFlag::QueenCastle = flag {
            let (rook_src, rook_dest) = castle_rook_squares(us, flag);
            self.remove_piece(rook_dest);
            self.put_piece(rook_src, Piece::new(us, PieceType::Rook));
        }

        self.remove_piece(mv.dest());
        self.put_piece(mv.src(), moved);

        if flag == MoveFlag::EnPassant {
            let cap_rank = if us == Color::White { 4 } else { 3 };
            self.put_piece(
                Square::new(cap_rank, mv.dest().file()),
                Piece::new(us.flip(), PieceType::Pawn),
            );
        } else if let Some(captured) = undo.captured_piece {
            self.put_piece(mv.dest(), captured);
        }

        self.castling_rights = undo.castling_rights;
        self.en_passent = undo.en_passent;
        self.half_move_clock = undo.half_move_clock;
    }

    /// An iterator over all occupied squares and their pieces, in square-index
    /// (LERF) order: a1, b1, ... h8.
    pub fn piece_iter(&self) -> impl Iterator<Item = (Square, Piece)> + '_ {
        self.pieces
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| p.map(|piece| (Square::from_index(i as u8), piece)))
    }
}

/// Per-square castling-rights mask. `castling_rights &= CASTLE_MASK[sq]` clears
/// exactly the rights invalidated when a piece leaves or lands on `sq`: the king
/// homes (e1/e8) drop both of a side's rights, each rook home drops its own. All
/// other squares are `ALL` (clear nothing). Indices are LERF: a1=0 ... h8=63.
const CASTLE_MASK: [CastlingRights; 64] = {
    let mut mask = [CastlingRights::ALL; 64];
    // 0xF & !right - "all rights except the one this square invalidates".
    mask[0] = CastlingRights::from_u8(0xF & !2); // a1: white queenside rook
    mask[7] = CastlingRights::from_u8(0xF & !1); // h1: white kingside rook
    mask[4] = CastlingRights::from_u8(0xF & !3); // e1: white king (both)
    mask[56] = CastlingRights::from_u8(0xF & !8); // a8: black queenside rook
    mask[63] = CastlingRights::from_u8(0xF & !4); // h8: black kingside rook
    mask[60] = CastlingRights::from_u8(0xF & !12); // e8: black king (both)
    mask
};

/// The rook's `(from, to)` squares for a castle by `color`. `flag` selects king-
/// or queen-side; any other flag is a caller bug.
fn castle_rook_squares(color: Color, flag: MoveFlag) -> (Square, Square) {
    let rank = if color == Color::White { 0 } else { 7 };
    match flag {
        MoveFlag::KingCastle => (Square::new(rank, 7), Square::new(rank, 5)), // h->f
        MoveFlag::QueenCastle => (Square::new(rank, 0), Square::new(rank, 3)), // a->d
        _ => unreachable!("castle_rook_squares called with non-castle flag"),
    }
}

/// Why a FEN string failed to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, thiserror::Error)]
pub enum ParseFenError {
    /// Input ended where a required field was expected.
    #[error("FEN ended before a required field")]
    IncompleteFen,
    /// An unexpected byte at the given offset.
    #[error("unexpected byte at offset {0}")]
    UnexpectedChar(usize),
    /// The castling field was malformed (empty, duplicate, or bad letter).
    #[error("malformed castling-rights field")]
    InvalidCastlingRights,
    /// The piece-placement field did not describe exactly 64 squares.
    #[error("piece placement did not describe exactly 64 squares")]
    InvalidPlacement,
    /// The en-passant field was not `-` or a square on rank 3 or 6.
    #[error("en passant target was not `-` or a square on rank 3 or 6")]
    InvalidEnpassentSquare,
    /// A numeric field overflowed or contained no digits.
    #[error("numeric field was empty or overflowed")]
    InvalidNumber,
    /// The half-move clock did not fit in a `u8`.
    #[error("half-move clock did not fit in a u8")]
    InvalidHalfMoveClock,
    /// The full-move counter was `0` or did not fit in a `u16`.
    #[error("full-move counter was 0 or out of range")]
    InvalidFullMove,
}

impl Board {
    /// Parse a board from a FEN byte string.
    ///
    /// # Notes
    /// Single-pass, no-allocation, lenient: parsing may stop after any complete
    /// field, and every absent field takes its default value.
    pub fn from_fen(fen: &[u8]) -> Result<Self, ParseFenError> {
        let mut cur = FenCursor { fen, pos: 0 };
        let mut board = Board::empty();
        // an absent castling field means no rights, not all.
        board.castling_rights = CastlingRights::NONE;

        cur.parse_placement(&mut board)?;
        if !cur.next_field()? {
            return Ok(board);
        }
        board.side_to_move = cur.parse_side_to_move()?;
        if !cur.next_field()? {
            return Ok(board);
        }
        board.castling_rights = cur.parse_castling_rights()?;
        if !cur.next_field()? {
            return Ok(board);
        }
        board.en_passent = cur.parse_en_passant()?;
        if !cur.next_field()? {
            return Ok(board);
        }
        board.half_move_clock = cur.parse_half_move_clock()?;
        if !cur.next_field()? {
            return Ok(board);
        }
        board.full_moves = cur.parse_full_move()?;
        if cur.peek().is_some() {
            return Err(ParseFenError::UnexpectedChar(cur.pos));
        }
        Ok(board)
    }
}

/// A forward-only byte cursor over a FEN string.
///
/// Each `parse_*` method consumes exactly one field, leaving the cursor on the
/// terminating space (or at end of input); [`next_field`](Self::next_field)
/// then consumes the separator and reports whether another field follows.
struct FenCursor<'a> {
    fen: &'a [u8],
    pos: usize,
}

impl FenCursor<'_> {
    #[inline]
    fn peek(&self) -> Option<u8> {
        self.fen.get(self.pos).copied()
    }

    #[inline]
    fn bump(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    /// Consume the single space separating two fields.
    ///
    /// `Ok(true)` - a field follows. `Ok(false)` - end of input (lenient stop).
    /// `Err` - a non-space byte sits where a separator should be (trailing
    /// garbage).
    fn next_field(&mut self) -> Result<bool, ParseFenError> {
        match self.peek() {
            None => Ok(false),
            Some(b' ') => {
                self.pos += 1;
                Ok(true)
            }
            Some(_) => Err(ParseFenError::UnexpectedChar(self.pos)),
        }
    }

    #[inline]
    fn parse_piece(b: u8) -> Option<Piece> {
        Some(match b {
            b'P' => Piece::WHITE_PAWN,
            b'p' => Piece::BLACK_PAWN,
            b'N' => Piece::WHITE_KNIGHT,
            b'n' => Piece::BLACK_KNIGHT,
            b'B' => Piece::WHITE_BISHOP,
            b'b' => Piece::BLACK_BISHOP,
            b'R' => Piece::WHITE_ROOK,
            b'r' => Piece::BLACK_ROOK,
            b'Q' => Piece::WHITE_QUEEN,
            b'q' => Piece::BLACK_QUEEN,
            b'K' => Piece::WHITE_KING,
            b'k' => Piece::BLACK_KING,
            _ => return None,
        })
    }

    /// Field 1: piece placement, written straight into the board's bitboards.
    /// FEN ranks run from rank 8 (index 7) down to rank 1 (index 0).
    fn parse_placement(&mut self, board: &mut Board) -> Result<(), ParseFenError> {
        let mut rank: u8 = 7;
        let mut file: u8 = 0;
        loop {
            let b = match self.peek() {
                None | Some(b' ') => break,
                Some(b) => b,
            };
            self.pos += 1;
            match b {
                b'/' => {
                    if file != 8 || rank == 0 {
                        return Err(ParseFenError::InvalidPlacement);
                    }
                    rank -= 1;
                    file = 0;
                }
                b'1'..=b'8' => {
                    file += b - b'0';
                    if file > 8 {
                        return Err(ParseFenError::InvalidPlacement);
                    }
                }
                _ => {
                    let piece =
                        Self::parse_piece(b).ok_or(ParseFenError::UnexpectedChar(self.pos - 1))?;
                    if file >= 8 {
                        return Err(ParseFenError::InvalidPlacement);
                    }
                    board.bitboard_for_mut(piece).set(Square::new(rank, file));
                    board.pieces[(rank << 3 | file) as usize] = Some(piece);
                    file += 1;
                }
            }
        }
        if rank != 0 || file != 8 {
            return Err(ParseFenError::InvalidPlacement);
        }
        Ok(())
    }

    /// Field 2: side to move.
    fn parse_side_to_move(&mut self) -> Result<Color, ParseFenError> {
        match self.bump() {
            Some(b'w') => Ok(Color::White),
            Some(b'b') => Ok(Color::Black),
            Some(_) => Err(ParseFenError::UnexpectedChar(self.pos - 1)),
            None => Err(ParseFenError::IncompleteFen),
        }
    }

    /// Field 3: castling rights (`KQkq`, a subset, or `-`).
    fn parse_castling_rights(&mut self) -> Result<CastlingRights, ParseFenError> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
            return Ok(CastlingRights::NONE);
        }
        let mut rights = CastlingRights::NONE;
        loop {
            let b = match self.peek() {
                None | Some(b' ') => break,
                Some(b) => b,
            };
            self.pos += 1;
            let right = match b {
                b'K' => CastlingRights::WHITE_KINGSIDE,
                b'Q' => CastlingRights::WHITE_QUEENSIDE,
                b'k' => CastlingRights::BLACK_KINGSIDE,
                b'q' => CastlingRights::BLACK_QUEENSIDE,
                _ => return Err(ParseFenError::UnexpectedChar(self.pos - 1)),
            };
            if rights.contains(right) {
                return Err(ParseFenError::InvalidCastlingRights);
            }
            rights |= right;
        }
        // An empty field (neither flags nor `-`) is malformed.
        if rights == CastlingRights::NONE {
            return Err(ParseFenError::InvalidCastlingRights);
        }
        Ok(rights)
    }

    /// Field 4: en passant target square (`-`, or a square on rank 3 or 6).
    fn parse_en_passant(&mut self) -> Result<Option<Square>, ParseFenError> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
            return Ok(None);
        }
        let file = self.bump().ok_or(ParseFenError::IncompleteFen)?;
        let rank = self.bump().ok_or(ParseFenError::IncompleteFen)?;
        let square =
            Square::from_ascii(&[file, rank]).map_err(|_| ParseFenError::InvalidEnpassentSquare)?;
        // A real en passant target is always on rank 3 (index 2) or 6 (index 5).
        if square.rank() != 2 && square.rank() != 5 {
            return Err(ParseFenError::InvalidEnpassentSquare);
        }
        Ok(Some(square))
    }

    /// Parse a run of ASCII digits into a `u32`, overflow-checked. No alloc,
    /// no UTF-8 validation - just fold the digits.
    fn parse_u32(&mut self) -> Result<u32, ParseFenError> {
        let start = self.pos;
        let mut value: u32 = 0;
        loop {
            match self.peek() {
                Some(b @ b'0'..=b'9') => {
                    self.pos += 1;
                    value = value
                        .checked_mul(10)
                        .and_then(|v| v.checked_add((b - b'0') as u32))
                        .ok_or(ParseFenError::InvalidNumber)?;
                }
                None | Some(b' ') => break,
                Some(_) => return Err(ParseFenError::UnexpectedChar(self.pos)),
            }
        }
        if self.pos == start {
            return Err(ParseFenError::InvalidNumber);
        }
        Ok(value)
    }

    /// Field 5: half-move clock.
    fn parse_half_move_clock(&mut self) -> Result<u8, ParseFenError> {
        u8::try_from(self.parse_u32()?).map_err(|_| ParseFenError::InvalidHalfMoveClock)
    }

    /// Field 6: full-move counter (1-based, so `0` is invalid).
    fn parse_full_move(&mut self) -> Result<u16, ParseFenError> {
        let value = self.parse_u32()?;
        if value == 0 || value > u16::MAX as u32 {
            return Err(ParseFenError::InvalidFullMove);
        }
        Ok(value as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(s: &str) -> Option<Square> {
        Some(Square::from_ascii(s.as_bytes()).unwrap())
    }

    fn sq(s: &str) -> Square {
        Square::from_ascii(s.as_bytes()).unwrap()
    }

    fn mv(src: &str, dst: &str, flag: MoveFlag) -> Move {
        Move::new(sq(src), sq(dst), flag)
    }

    #[test]
    fn const_startpos_matches_fen() {
        let from_fen =
            Board::from_fen(b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        // The hand-written const bitboards/mailbox must equal the parsed board.
        assert!(Board::starting_position() == from_fen);
    }

    #[test]
    fn make_unmake_round_trips() {
        // One move of every kind. The property: make then unmake restores the
        // board byte-for-byte. This is the foundation perft will stand on.
        let cases: &[(&str, Move)] = &[
            // quiet, double push, and a captures-then-back from the same startpos
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                mv("b1", "c3", MoveFlag::Quiet),
            ),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                mv("e2", "e4", MoveFlag::DoublePawnPush),
            ),
            // black to move: exercises the full-move decrement + side flip
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
                mv("b8", "c6", MoveFlag::Quiet),
            ),
            (
                "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
                mv("e4", "d5", MoveFlag::Capture),
            ),
            (
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                mv("e5", "d6", MoveFlag::EnPassant),
            ),
            (
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                mv("a7", "a8", MoveFlag::PromoQueen),
            ),
            (
                "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                mv("a7", "b8", MoveFlag::PromoQueenCapture),
            ),
            (
                "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
                mv("e1", "g1", MoveFlag::KingCastle),
            ),
            (
                "r3k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
                mv("e1", "c1", MoveFlag::QueenCastle),
            ),
        ];

        for (fen, m) in cases {
            let original = Board::from_fen(fen.as_bytes()).unwrap();
            let mut work = original.clone();
            let undo = work.make_move(*m);
            assert!(work != original, "make must change the board: {fen} {m:?}");
            work.unmake_move(*m, undo);
            assert!(work == original, "round trip must restore: {fen} {m:?}");
        }
    }

    #[test]
    fn capture_removes_victim() {
        // Directly catches a phantom captured bit: if `make` failed to clear the
        // victim's bitboard, occupancy would still read 4 after the capture.
        let mut b = Board::from_fen(b"4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(b.occupied().count(), 4);
        let m = mv("e4", "d5", MoveFlag::Capture);
        let undo = b.make_move(m);
        assert_eq!(b.occupied().count(), 3);
        assert_eq!(b.occupied_by(Color::Black).count(), 1); // lone king
        assert_eq!(b.piece_at(sq("d5")), Some(Piece::WHITE_PAWN));
        b.unmake_move(m, undo);
        assert_eq!(b.occupied().count(), 4);
        assert_eq!(b.piece_at(sq("d5")), Some(Piece::BLACK_PAWN));
    }

    #[test]
    fn make_move_updates_clocks_and_ep() {
        let mut b = Board::starting_position();
        let m1 = mv("e2", "e4", MoveFlag::DoublePawnPush);
        let u1 = b.make_move(m1);
        assert_eq!(b.side_to_move(), Color::Black);
        assert_eq!(b.en_passant(), ep("e3")); // White double push -> e3 target
        assert_eq!(b.piece_at(sq("e4")), Some(Piece::WHITE_PAWN));
        assert_eq!(b.piece_at(sq("e2")), None);

        let m2 = mv("e7", "e5", MoveFlag::DoublePawnPush);
        let u2 = b.make_move(m2);
        assert_eq!(b.full_moves, 2); // increments after Black's move
        assert_eq!(b.en_passant(), ep("e6")); // retargeted; old e3 cleared
        assert_eq!(b.side_to_move(), Color::White);

        b.unmake_move(m2, u2);
        assert_eq!(b.full_moves, 1);
        assert_eq!(b.en_passant(), ep("e3"));
        b.unmake_move(m1, u1);
        assert!(b == Board::starting_position());
    }

    #[test]
    fn castling_rights_clear_and_restore() {
        let start = Board::from_fen(b"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();

        // King move forfeits both of its side's rights; unmake restores them.
        let mut b = start.clone();
        let m = mv("e1", "e2", MoveFlag::Quiet);
        let u = b.make_move(m);
        assert_eq!(
            b.castling_rights,
            CastlingRights::BLACK_KINGSIDE | CastlingRights::BLACK_QUEENSIDE
        );
        b.unmake_move(m, u);
        assert_eq!(b.castling_rights, CastlingRights::ALL);

        // Rook move forfeits only its own corner.
        let mut b = start.clone();
        let _ = b.make_move(mv("a1", "a4", MoveFlag::Quiet));
        assert!(!b.castling_rights.contains(CastlingRights::WHITE_QUEENSIDE));
        assert!(b.castling_rights.contains(CastlingRights::WHITE_KINGSIDE));

        // Capturing a rook on its home square forfeits the *opponent's* right
        // there (via the move's destination), plus the moving rook's own.
        let mut b = start.clone();
        let _ = b.make_move(mv("a1", "a8", MoveFlag::Capture));
        assert_eq!(
            b.castling_rights,
            CastlingRights::WHITE_KINGSIDE | CastlingRights::BLACK_KINGSIDE
        );

        // The castle move itself drops both rights (king leaves e1).
        let mut b = start.clone();
        let _ = b.make_move(mv("e1", "g1", MoveFlag::KingCastle));
        assert_eq!(
            b.castling_rights,
            CastlingRights::BLACK_KINGSIDE | CastlingRights::BLACK_QUEENSIDE
        );
    }

    #[test]
    fn board_occupancy_and_piece_at() {
        let board = Board::starting_position();
        assert_eq!(board.occupied().count(), 32);
        assert_eq!(board.occupied_by(Color::White).count(), 16);
        assert_eq!(board.occupied_by(Color::Black).count(), 16);
        // White and black occupancy are disjoint and together cover everything.
        assert_eq!(
            board.occupied_by(Color::White) & board.occupied_by(Color::Black),
            Bitboard::EMPTY
        );
        assert_eq!(
            board.occupied_by(Color::White) | board.occupied_by(Color::Black),
            board.occupied()
        );

        let e1 = Square::from_ascii(b"e1").unwrap();
        let e4 = Square::from_ascii(b"e4").unwrap();
        assert_eq!(board.piece_at(e1), Some(Piece::WHITE_KING));
        assert_eq!(board.piece_at(e4), None);
    }

    #[test]
    fn parses_full_starting_position() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(fen.as_bytes()).expect("valid fen should parse");

        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.half_move_clock, 0);
        assert_eq!(board.full_moves, 1);
        assert_eq!(board.castling_rights, CastlingRights::ALL);
        assert_eq!(board.en_passent, None);
        // 32 pieces at the start, and the white king sits on e1.
        let occupied: u32 = board.bitboards.iter().map(|bb| bb.count()).sum();
        assert_eq!(occupied, 32);
        let e1 = Square::from_ascii(b"e1").unwrap();
        assert!(board.bitboard_for(Piece::WHITE_KING).contains(e1));
    }

    #[test]
    fn parses_en_passant_and_clocks() {
        let fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 1 2";
        let board = Board::from_fen(fen.as_bytes()).expect("valid fen should parse");
        assert_eq!(board.en_passent, ep("e6"));
        assert_eq!(board.half_move_clock, 1);
        assert_eq!(board.full_moves, 2);

        // A subset of castling rights round-trips.
        let fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b Kq - 1 2";
        let board = Board::from_fen(fen.as_bytes()).expect("valid fen should parse");
        assert_eq!(
            board.castling_rights,
            CastlingRights::WHITE_KINGSIDE | CastlingRights::BLACK_QUEENSIDE
        );
    }

    #[test]
    fn lenient_truncation_uses_defaults() {
        // Placement + side only: everything after defaults.
        let board = Board::from_fen(b"8/8/8/8/8/8/8/8 b").expect("partial fen should parse");
        assert_eq!(board.side_to_move, Color::Black);
        assert_eq!(board.castling_rights, CastlingRights::NONE);
        assert_eq!(board.en_passent, None);
        assert_eq!(board.half_move_clock, 0);
        assert_eq!(board.full_moves, 1);
    }

    #[test]
    fn error_variants_are_reachable() {
        use ParseFenError::*;
        // One representative trigger per error variant. We compare by
        // discriminant so the exact byte offset of `UnexpectedChar` stays an
        // implementation detail rather than a brittle assertion.
        let cases: &[(&[u8], ParseFenError)] = &[
            (b"8/8/8/8/8/8/8/8/8 w", InvalidPlacement), // nine ranks
            (b"8/8/8/8/8/8/8/8 x", UnexpectedChar(0)),  // bad side-to-move byte
            (b"8/8/8/8/8/8/8/8 ", IncompleteFen),       // separator, then nothing
            (b"8/8/8/8/8/8/8/8 w KK", InvalidCastlingRights), // duplicate right
            (b"8/8/8/8/8/8/8/8 w KQkq e4", InvalidEnpassentSquare), // ep off rank 3/6
            (b"8/8/8/8/8/8/8/8 w KQkq - 99999999999 1", InvalidNumber), // u32 overflow
            (b"8/8/8/8/8/8/8/8 w KQkq - 300 1", InvalidHalfMoveClock), // > u8::MAX
            (b"8/8/8/8/8/8/8/8 w KQkq - 0 0", InvalidFullMove), // full-move 0
        ];
        for (input, expected) in cases {
            let text = std::str::from_utf8(input).unwrap();
            match Board::from_fen(input) {
                Err(got) => assert_eq!(
                    std::mem::discriminant(&got),
                    std::mem::discriminant(expected),
                    "input {text:?}: got {got:?}, expected {expected:?}"
                ),
                Ok(_) => panic!("expected error for input {text:?}"),
            }
        }
    }

    #[test]
    fn parses_boundary_values() {
        // Max half-move (u8) and full-move (u16) counters, plus both legal
        // en passant ranks (3 and 6).
        let board = Board::from_fen(b"8/8/8/8/8/8/8/8 w KQkq e3 255 65535").expect("valid fen");
        assert_eq!(board.half_move_clock, 255);
        assert_eq!(board.full_moves, 65535);
        assert_eq!(board.en_passent, ep("e3"));

        let board = Board::from_fen(b"8/8/8/8/8/8/8/8 b - c6").expect("valid fen");
        assert_eq!(board.en_passent, ep("c6"));
    }
}
