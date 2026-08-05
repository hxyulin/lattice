use super::{
    Bitboard, CastlingRights, Color, Move, MoveType, Piece, PieceType, Square, State, Undo,
};

const CASTLE_MASK: [u8; 64] = castle_masks();
const fn castle_masks() -> [u8; 64] {
    let mut m = [15; 64];
    m[0] = 13;
    m[4] = 12;
    m[7] = 14;
    m[56] = 7;
    m[60] = 3;
    m[63] = 11;
    m
}
const fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn piece_key(piece: Piece, square: Square) -> u64 {
    let dense = (piece.color() as u64) * 6 + piece.piece_type() as u64;
    mix(dense * 64 + square.index() as u64)
}
fn castle_key(rights: CastlingRights) -> u64 {
    mix(768 + rights.0 as u64)
}
fn ep_key(square: Square) -> u64 {
    mix(784 + square.file() as u64)
}
fn side_key() -> u64 {
    mix(792)
}

/// A chess position with synchronized bitboards and mailbox storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    pieces: [Bitboard; 6],
    colors: [Bitboard; 2],
    occupied: Bitboard,
    mailbox: [Option<Piece>; 64],
    state: State,
    history: Vec<Undo>,
}

impl Board {
    /// Returns the standard chess starting position.
    pub fn startpos() -> Self {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse()
            .unwrap()
    }
    /// Returns the square occupied by a side's king.
    pub fn king_square(&self, color: Color) -> Square {
        (self.pieces(PieceType::King) & self.color(color))
            .lsb()
            .unwrap()
    }
    pub(crate) fn empty(state: State) -> Self {
        Self {
            pieces: [Bitboard::empty(); 6],
            colors: [Bitboard::empty(); 2],
            occupied: Bitboard::empty(),
            mailbox: [None; 64],
            state,
            history: Vec::with_capacity(256),
        }
    }
    /// Returns the piece occupying a square.
    pub fn piece_on(&self, square: Square) -> Option<Piece> {
        self.mailbox[square.index() as usize]
    }
    /// Returns all squares occupied by a piece type.
    pub fn pieces(&self, piece_type: PieceType) -> Bitboard {
        self.pieces[piece_type as usize]
    }
    /// Returns all squares occupied by a color.
    pub fn color(&self, color: Color) -> Bitboard {
        self.colors[color as usize]
    }
    /// Returns all occupied squares.
    pub fn occupied(&self) -> Bitboard {
        self.occupied
    }
    /// Returns the current game state.
    pub const fn state(&self) -> &State {
        &self.state
    }
    /// Applies a move and saves enough information to undo it.
    pub fn make(&mut self, mv: Move) {
        let from = mv.from();
        let to = mv.to();
        let moving = self.piece_on(from).expect("move origin must be occupied");
        let capture_square = if mv.move_type() == MoveType::EnPassant {
            Square::new_unchecked(if moving.color() == Color::White {
                to.index() - 8
            } else {
                to.index() + 8
            })
        } else {
            to
        };
        let captured = if mv.is_capture() {
            self.piece_on(capture_square)
        } else {
            None
        };
        self.history.push(Undo {
            state: self.state,
            captured,
        });
        self.set_ep(None);
        self.remove_piece(moving, from);
        if let Some(piece) = captured {
            self.remove_piece(piece, capture_square);
        }
        let placed = mv
            .promoted_piece()
            .map_or(moving, |kind| Piece::new(moving.color(), kind));
        self.add_piece(placed, to);
        match mv.move_type() {
            MoveType::DoublePawnPush => {
                let ep = Square::new_unchecked((from.index() + to.index()) / 2);
                self.set_ep(Some(ep)); /* ponytail: set ep only when an opposing pawn can legally capture. */
            }
            MoveType::KingCastle | MoveType::QueenCastle => {
                let (rf, rt) = castle_rook_squares(to);
                let rook = self.piece_on(rf).expect("castling rook must be present");
                self.remove_piece(rook, rf);
                self.add_piece(rook, rt);
            }
            _ => {}
        }
        let old_castle = self.state.castling;
        self.state.zobrist ^= castle_key(old_castle);
        self.state.castling.0 &=
            CASTLE_MASK[from.index() as usize] & CASTLE_MASK[to.index() as usize];
        self.state.zobrist ^= castle_key(self.state.castling);
        self.state.halfmove = if moving.piece_type() == PieceType::Pawn || captured.is_some() {
            0
        } else {
            self.state.halfmove.saturating_add(1)
        };
        if moving.color() == Color::Black {
            self.state.fullmove = self.state.fullmove.saturating_add(1);
        }
        self.state.side_to_move = self.state.side_to_move.flip();
        self.state.zobrist ^= side_key();
        self.debug_check();
    }
    /// Undoes the most recently applied move.
    pub fn unmake(&mut self, mv: Move) {
        let undo = self.history.pop().expect("unmake requires move history");
        let from = mv.from();
        let to = mv.to();
        let moved = self
            .piece_on(to)
            .expect("move destination must be occupied");
        self.remove_piece(moved, to);
        let original = if mv.is_promotion() {
            Piece::new(undo.state.side_to_move, PieceType::Pawn)
        } else {
            moved
        };
        self.add_piece(original, from);
        if matches!(mv.move_type(), MoveType::KingCastle | MoveType::QueenCastle) {
            let (rf, rt) = castle_rook_squares(to);
            let rook = self.piece_on(rt).expect("castled rook must be present");
            self.remove_piece(rook, rt);
            self.add_piece(rook, rf);
        }
        if let Some(piece) = undo.captured {
            let square = if mv.move_type() == MoveType::EnPassant {
                Square::new_unchecked(if undo.state.side_to_move == Color::White {
                    to.index() - 8
                } else {
                    to.index() + 8
                })
            } else {
                to
            };
            self.add_piece(piece, square);
        }
        self.state = undo.state;
        self.debug_check();
    }
    pub(crate) fn add_piece(&mut self, piece: Piece, square: Square) {
        self.mailbox[square.index() as usize] = Some(piece);
        self.pieces[piece.piece_type() as usize].set(square);
        self.colors[piece.color() as usize].set(square);
        self.occupied.set(square);
        self.state.zobrist ^= piece_key(piece, square);
    }
    fn remove_piece(&mut self, piece: Piece, square: Square) {
        self.mailbox[square.index() as usize] = None;
        self.pieces[piece.piece_type() as usize].clear(square);
        self.colors[piece.color() as usize].clear(square);
        self.occupied.clear(square);
        self.state.zobrist ^= piece_key(piece, square);
    }
    fn set_ep(&mut self, ep: Option<Square>) {
        if let Some(s) = self.state.ep {
            self.state.zobrist ^= ep_key(s)
        }
        self.state.ep = ep;
        if let Some(s) = ep {
            self.state.zobrist ^= ep_key(s)
        }
    }
    pub(crate) fn finish_setup(&mut self) {
        self.state.zobrist = self.recompute_zobrist();
        self.debug_check();
    }
    fn recompute_zobrist(&self) -> u64 {
        let mut z = castle_key(self.state.castling);
        if self.state.side_to_move == Color::Black {
            z ^= side_key()
        }
        if let Some(s) = self.state.ep {
            z ^= ep_key(s)
        }
        for i in 0..64 {
            let s = Square::new_unchecked(i);
            if let Some(p) = self.piece_on(s) {
                z ^= piece_key(p, s)
            }
        }
        z
    }
    #[cfg(debug_assertions)]
    fn debug_check(&self) {
        assert_eq!(self.occupied, self.colors[0] | self.colors[1]);
        assert!((self.colors[0] & self.colors[1]).is_empty());
        for i in 0..64 {
            let s = Square::new_unchecked(i);
            let found: Vec<_> = (0..6).filter(|&p| self.pieces[p].contains(s)).collect();
            assert!(found.len() <= 1);
            match self.piece_on(s) {
                Some(p) => {
                    assert_eq!(found, vec![p.piece_type() as usize]);
                    assert!(self.colors[p.color() as usize].contains(s));
                }
                None => assert!(found.is_empty()),
            }
        }
        assert_eq!(self.state.zobrist, self.recompute_zobrist());
    }
    #[cfg(not(debug_assertions))]
    fn debug_check(&self) {}
}
fn castle_rook_squares(king_to: Square) -> (Square, Square) {
    match king_to.index() {
        6 => (Square::new_unchecked(7), Square::new_unchecked(5)),
        2 => (Square::new_unchecked(0), Square::new_unchecked(3)),
        62 => (Square::new_unchecked(63), Square::new_unchecked(61)),
        58 => (Square::new_unchecked(56), Square::new_unchecked(59)),
        _ => panic!("invalid castling destination"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;
    fn sq(s: &str) -> Square {
        Square::new_unchecked((s.as_bytes()[1] - b'1') * 8 + s.as_bytes()[0] - b'a')
    }
    #[test]
    fn fen_roundtrips_corpus() {
        let mut fens: Vec<String> = vec![
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p2n2/8/8/6B1/PPP1NPPP/RN1QK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/8/2B1P3/2N1Q3/PPP2PPP/2KR3R w - - 0 10",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        for mask in 0..16 {
            let mut r = String::new();
            for (b, c) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] {
                if mask & b != 0 {
                    r.push(c)
                }
            }
            if r.is_empty() {
                r.push('-')
            }
            fens.push(format!("4k3/8/8/8/8/8/8/4K3 w {r} - 0 1"));
        }
        fens.push(String::from("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"));
        for fen in &fens {
            assert_eq!(&Board::from_str(fen).unwrap().to_string(), fen);
        }
    }
    #[test]
    fn all_move_types_roundtrip() {
        let cases = [
            (
                "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                "e2",
                "e3",
                MoveType::Quiet,
            ),
            (
                "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                "e2",
                "e4",
                MoveType::DoublePawnPush,
            ),
            (
                "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
                "e1",
                "g1",
                MoveType::KingCastle,
            ),
            (
                "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
                "e1",
                "c1",
                MoveType::QueenCastle,
            ),
            (
                "4k3/8/8/8/8/3p4/4P3/4K3 w - - 0 1",
                "e2",
                "d3",
                MoveType::Capture,
            ),
            (
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                "e5",
                "d6",
                MoveType::EnPassant,
            ),
            (
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "a8",
                MoveType::KnightPromo,
            ),
            (
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "a8",
                MoveType::BishopPromo,
            ),
            (
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "a8",
                MoveType::RookPromo,
            ),
            (
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "a8",
                MoveType::QueenPromo,
            ),
            (
                "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "b8",
                MoveType::KnightPromoCap,
            ),
            (
                "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "b8",
                MoveType::BishopPromoCap,
            ),
            (
                "1b2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "b8",
                MoveType::RookPromoCap,
            ),
            (
                "1q2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7",
                "b8",
                MoveType::QueenPromoCap,
            ),
        ];
        for (fen, a, b, t) in cases {
            let mut board = Board::from_str(fen).unwrap();
            let before = Board::from_str(fen).unwrap();
            let mv = Move::new(sq(a), sq(b), t);
            board.make(mv);
            assert_eq!(board.state.zobrist, board.recompute_zobrist());
            board.unmake(mv);
            assert_eq!(board.pieces, before.pieces, "pieces {t:?}");
            assert_eq!(board.colors, before.colors, "colors {t:?}");
            assert_eq!(board.occupied, before.occupied, "occupied {t:?}");
            assert_eq!(board.mailbox, before.mailbox, "mailbox {t:?}");
            assert_eq!(board.state, before.state, "state {t:?}");
            assert_eq!(board, before);
            assert_eq!(board.state.zobrist, board.recompute_zobrist());
        }
    }
}

#[cfg(test)]
mod invariants {
    use super::*;
    use core::str::FromStr;
    fn sq(s: &str) -> Square {
        Square::new_unchecked((s.as_bytes()[1] - b'1') * 8 + s.as_bytes()[0] - b'a')
    }
    // catches: the rank count check weakened to accept fewer than eight ranks,
    // a repeated castling letter accepted, and fullmove 0 accepted. The three
    // trailing cases are the ones no other test covers.
    #[test]
    fn fen_rejects_malformed_input() {
        for bad in [
            "not a fen",
            "",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0",
            "9/8/8/8/8/8/8/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
            "4k3/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w KK - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
        ] {
            assert!(Board::from_str(bad).is_err(), "accepted {bad:?}");
        }
    }
    #[test]
    fn multi_ply_sequence_unwinds() {
        let fen = "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1";
        let mut b = Board::from_str(fen).unwrap();
        let before = Board::from_str(fen).unwrap();
        let seq = [
            (sq("e1"), sq("g1"), MoveType::KingCastle),
            (sq("e8"), sq("c8"), MoveType::QueenCastle),
            (sq("g5"), sq("f6"), MoveType::Capture),
            (sq("g4"), sq("f3"), MoveType::Capture),
            (sq("g2"), sq("f3"), MoveType::Capture),
        ];
        let mvs: Vec<_> = seq.iter().map(|&(f, t, k)| Move::new(f, t, k)).collect();
        for m in &mvs {
            b.make(*m);
        }
        for m in mvs.iter().rev() {
            b.unmake(*m);
        }
        assert_eq!(b, before, "multi-ply unwind mismatch");
    }
    // catches: the halfmove clock not incrementing, not resetting on a pawn
    // move or on a capture, the fullmove number not advancing after Black or
    // advancing after White, a double push leaving no en passant square or
    // setting the wrong one, and a king move revoking only one of its two
    // castling rights. Asserting the whole FEN rather than one field is what
    // makes a single case cover all of them.
    #[test]
    fn make_updates_every_state_field() {
        let cases = [
            (
                "4k3/8/8/8/8/8/8/4K1N1 w - - 5 9",
                ("g1", "f3", MoveType::Quiet),
                "4k3/8/8/8/8/5N2/8/4K3 b - - 6 9",
            ),
            (
                "4k1n1/8/8/8/8/8/8/4K3 b - - 5 9",
                ("g8", "f6", MoveType::Quiet),
                "4k3/8/5n2/8/8/8/8/4K3 w - - 6 10",
            ),
            (
                "4k3/8/8/8/8/8/4P3/4K3 w - - 7 9",
                ("e2", "e3", MoveType::Quiet),
                "4k3/8/8/8/8/4P3/8/4K3 b - - 0 9",
            ),
            (
                "4k3/8/8/8/8/8/4P3/4K3 w - - 7 9",
                ("e2", "e4", MoveType::DoublePawnPush),
                "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 9",
            ),
            (
                "4k3/8/8/8/8/5n2/8/4K1N1 w - - 7 9",
                ("g1", "f3", MoveType::Capture),
                "4k3/8/8/8/8/5N2/8/4K3 b - - 0 9",
            ),
            (
                "4k3/8/8/8/8/8/8/R3K2R w KQ - 3 9",
                ("e1", "f1", MoveType::Quiet),
                "4k3/8/8/8/8/8/8/R4K1R b - - 4 9",
            ),
            (
                "4k3/8/8/8/8/8/8/R3K2R w KQ - 3 9",
                ("e1", "g1", MoveType::KingCastle),
                "4k3/8/8/8/8/8/8/R4RK1 b - - 4 9",
            ),
        ];
        for (fen, (from, to, kind), after) in cases {
            let mut b = Board::from_str(fen).unwrap();
            let m = Move::new(sq(from), sq(to), kind);
            b.make(m);
            assert_eq!(b.to_string(), after, "make {from}{to} from {fen}");
            b.unmake(m);
            assert_eq!(b.to_string(), fen, "unmake {from}{to} from {fen}");
        }
    }

    // catches: en passant removing the pawn on the destination square instead
    // of the one beside the capturer, for either color. `all_move_types_
    // roundtrip` only unwinds the move, so it passes as long as make and
    // unmake agree on the same wrong square.
    #[test]
    fn en_passant_captures_the_pawn_beside_the_capturer() {
        for (fen, from, to, after) in [
            (
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
                "e5",
                "d6",
                "4k3/8/3P4/8/8/8/8/4K3 b - - 0 2",
            ),
            (
                "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2",
                "e4",
                "d3",
                "4k3/8/8/8/8/3p4/8/4K3 w - - 0 3",
            ),
        ] {
            let mut b = Board::from_str(fen).unwrap();
            let m = Move::new(sq(from), sq(to), MoveType::EnPassant);
            b.make(m);
            assert_eq!(b.to_string(), after, "ep {from}{to}");
            b.unmake(m);
            assert_eq!(b.to_string(), fen, "ep unmake {from}{to}");
        }
    }

    #[test]
    fn ep_right_is_restored_after_unrelated_move() {
        let fen = "4k3/8/8/3pP3/8/8/6P1/4K3 w - d6 0 2";
        let mut b = Board::from_str(fen).unwrap();
        let before = Board::from_str(fen).unwrap();
        let m = Move::new(sq("g2"), sq("g3"), MoveType::Quiet);
        b.make(m);
        assert_eq!(b.state.ep, None, "ep must clear after a non-double-push");
        b.unmake(m);
        assert_eq!(b.state.ep, Some(sq("d6")), "ep must be restored");
        assert_eq!(b, before);
    }
    #[test]
    fn rook_capture_revokes_castling() {
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let mut b = Board::from_str(fen).unwrap();
        let before = Board::from_str(fen).unwrap();
        let m = Move::new(sq("a1"), sq("a8"), MoveType::Capture);
        b.make(m);
        assert!(
            !b.state.castling.contains(CastlingRights::BLACK_QUEEN),
            "captured rook keeps its right"
        );
        assert!(
            !b.state.castling.contains(CastlingRights::WHITE_QUEEN),
            "moved rook keeps its right"
        );
        assert!(b.state.castling.contains(CastlingRights::WHITE_KING));
        b.unmake(m);
        assert_eq!(b, before);
    }
}
