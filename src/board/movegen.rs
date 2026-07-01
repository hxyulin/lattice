//! Pseudo-legal move generation.
//!
//! # Notes
//! Pseudo-legal = obeys the piece's movement rules without checking whether the
//! move leaves the mover's own king in check; the legality filter is a separate
//! later stage. Castling is the exception, generated fully legally here via
//! [`Board::is_attacked`], because the king may not start in, pass through, or
//! land on an attacked square - a rule no destination-only filter can reconstruct.

use crate::{Bitboard, Board, CastlingRights, Color, Move, MoveFlag, MoveList, PieceType, Square};

const NOT_FILE_A: u64 = !0x0101_0101_0101_0101;
const NOT_FILE_H: u64 = !0x8080_8080_8080_8080;
const NOT_FILE_AB: u64 = !0x0303_0303_0303_0303; // knight +/-2 jumps
const NOT_FILE_GH: u64 = !0xC0C0_C0C0_C0C0_C0C0; // knight +/-2 jumps

/// Piece values (centipawns) for [`Board::see`], indexed by `PieceType::as_u8()`.
/// The king is near-infinite so it is never picked to recapture into defense.
const SEE_VALUE: [i32; 6] = [100, 320, 330, 500, 900, 10_000];

/// Knight attack set for every square.
const KNIGHT_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let b = 1u64 << sq;
        // File-changing shifts are masked so an edge knight can't wrap to the far side.
        let a = ((b << 17) & NOT_FILE_A)   // +1 file, +2 rank
              | ((b << 15) & NOT_FILE_H)   // -1 file, +2 rank
              | ((b << 10) & NOT_FILE_AB)  // +2 file, +1 rank
              | ((b << 6) & NOT_FILE_GH)   // -2 file, +1 rank
              | ((b >> 6) & NOT_FILE_AB)   // +2 file, -1 rank
              | ((b >> 10) & NOT_FILE_GH)  // -2 file, -1 rank
              | ((b >> 15) & NOT_FILE_A)   // +1 file, -2 rank
              | ((b >> 17) & NOT_FILE_H); // -1 file, -2 rank
        table[sq] = Bitboard::from_bits(a);
        sq += 1;
    }
    table
};

/// King attack set for every square.
const KING_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let b = 1u64 << sq;
        let a = (b << 8)                  // N
              | (b >> 8)                  // S
              | ((b << 1) & NOT_FILE_A)   // E
              | ((b >> 1) & NOT_FILE_H)   // W
              | ((b << 9) & NOT_FILE_A)   // NE
              | ((b << 7) & NOT_FILE_H)   // NW
              | ((b >> 7) & NOT_FILE_A)   // SE
              | ((b >> 9) & NOT_FILE_H); // SW
        table[sq] = Bitboard::from_bits(a);
        sq += 1;
    }
    table
};

// Sliding attacks via the magic-bitboard tables: one multiply-shift-load per query.
use super::magic::{bishop_attacks, rook_attacks};

/// Squares attacked by a set of `color` pawns, as raw bits.
///
/// # Notes
/// Wrap-masked: west-captures can't land on the h-file, east on the a-file.
#[inline]
fn pawn_attacks(pawns: u64, color: Color) -> u64 {
    match color {
        Color::White => ((pawns << 7) & NOT_FILE_H) | ((pawns << 9) & NOT_FILE_A),
        Color::Black => ((pawns >> 9) & NOT_FILE_H) | ((pawns >> 7) & NOT_FILE_A),
    }
}

/// Signed shift: positive shifts left (toward higher squares), negative right.
#[inline]
fn shift(bb: u64, by: i32) -> u64 {
    if by >= 0 { bb << by } else { bb >> -by }
}

/// Emit a quiet-or-capture move per destination (capture if it lands on `enemy`).
fn push_targets(from: Square, targets: Bitboard, enemy: Bitboard, out: &mut MoveList) {
    for to in targets {
        let flag = if enemy.contains(to) {
            MoveFlag::Capture
        } else {
            MoveFlag::Quiet
        };
        out.push(Move::new(from, to, flag));
    }
}

/// The four promotion moves for a pawn reaching the last rank.
fn push_promotions(from: Square, to: Square, capture: bool, out: &mut MoveList) {
    let flags = if capture {
        [
            MoveFlag::PromoKnightCapture,
            MoveFlag::PromoBishopCapture,
            MoveFlag::PromoRookCapture,
            MoveFlag::PromoQueenCapture,
        ]
    } else {
        [
            MoveFlag::PromoKnight,
            MoveFlag::PromoBishop,
            MoveFlag::PromoRook,
            MoveFlag::PromoQueen,
        ]
    };
    for flag in flags {
        out.push(Move::new(from, to, flag));
    }
}

impl Board {
    /// Generate every pseudo-legal move for the side to move
    #[must_use]
    pub fn pseudo_legal_moves(&self) -> MoveList {
        let mut moves = MoveList::new();
        self.generate_moves(&mut moves);
        moves
    }

    /// Fill `out` with every pseudo-legal move for the side to move.
    ///
    /// # Notes
    /// `out` is cleared first.
    pub fn generate_moves(&self, out: &mut MoveList) {
        out.clear();
        let us = self.side_to_move();
        let friendly = self.occupied_by(us);
        let enemy = self.occupied_by(us.flip());
        let occ = self.occupied();

        let moves = out;

        self.gen_pawns(us, enemy, occ, moves);

        for from in self.pieces(us, PieceType::Knight) {
            push_targets(
                from,
                KNIGHT_ATTACKS[from.index() as usize] & !friendly,
                enemy,
                moves,
            );
        }
        for from in self.pieces(us, PieceType::King) {
            push_targets(
                from,
                KING_ATTACKS[from.index() as usize] & !friendly,
                enemy,
                moves,
            );
        }
        for from in self.pieces(us, PieceType::Bishop) {
            push_targets(from, bishop_attacks(from, occ) & !friendly, enemy, moves);
        }
        for from in self.pieces(us, PieceType::Rook) {
            push_targets(from, rook_attacks(from, occ) & !friendly, enemy, moves);
        }
        for from in self.pieces(us, PieceType::Queen) {
            let attacks = (rook_attacks(from, occ) | bishop_attacks(from, occ)) & !friendly;
            push_targets(from, attacks, enemy, moves);
        }

        self.gen_castling(us, occ, moves);
    }

    /// Emit legal castling moves for `us` (standard chess, not Chess960).
    ///
    /// # Notes
    /// Legal when the right is held, the squares between king and rook are empty,
    /// and the king is not in check, does not pass through an attacked square, nor
    /// land on one. The queenside b-file square must be empty but is never
    /// attack-tested - the king never steps there.
    fn gen_castling(&self, us: Color, occ: Bitboard, out: &mut MoveList) {
        let rank = if matches!(us, Color::White) { 0 } else { 7 };
        let opp = us.flip();
        let e = Square::new(rank, 4); // king's home square

        // Can't castle out of check - also spares the path checks below.
        if self.is_attacked(e, opp) {
            return;
        }
        let (ks, qs) = match us {
            Color::White => (
                CastlingRights::WHITE_KINGSIDE,
                CastlingRights::WHITE_QUEENSIDE,
            ),
            Color::Black => (
                CastlingRights::BLACK_KINGSIDE,
                CastlingRights::BLACK_QUEENSIDE,
            ),
        };
        let rights = self.castling_rights();
        let empty = |file| !occ.contains(Square::new(rank, file));
        let safe = |file| !self.is_attacked(Square::new(rank, file), opp);

        // Kingside O-O: f,g empty and unattacked; king e->g.
        if rights.contains(ks) && empty(5) && empty(6) && safe(5) && safe(6) {
            out.push(Move::new(e, Square::new(rank, 6), MoveFlag::KingCastle));
        }
        // Queenside O-O-O: b,c,d empty; only c,d attack-tested (king passes them).
        if rights.contains(qs) && empty(1) && empty(2) && empty(3) && safe(2) && safe(3) {
            out.push(Move::new(e, Square::new(rank, 2), MoveFlag::QueenCastle));
        }
    }

    /// Count leaf nodes of the legal move tree at `depth` as `(root_move, count)` pairs.
    #[must_use]
    pub fn perft_divide(&mut self, depth: u32) -> Vec<(Move, u64)> {
        let us = self.side_to_move();
        let opp = us.flip();
        let mut out = Vec::new();
        let mut moves = MoveList::new();
        self.generate_moves(&mut moves);
        for &mv in &moves {
            let undo = self.make_move(mv);
            let king = self
                .pieces(us, PieceType::King)
                .iter()
                .next()
                .expect("side to move always has a king");
            if !self.is_attacked(king, opp) {
                // depth 1: each legal root move has exactly one child (itself).
                let count = if depth <= 1 { 1 } else { self.perft(depth - 1) };
                out.push((mv, count));
            }
            self.unmake_move(mv, undo);
        }
        out
    }

    /// Count leaf nodes of the legal move tree at `depth` - a *perft*, the
    /// standard movegen correctness check.
    ///
    /// # Performance
    /// The legality filter (make each move, reject if our king is left attacked)
    /// is fused with the recursion to save [`Board::make_move`]/[`Board::unmake_move`]
    /// and [`Board::is_attacked`] calls. At `depth == 1` legal moves are bulk-counted
    /// instead of descending to depth-0 leaves.
    #[must_use]
    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let us = self.side_to_move();
        let opp = us.flip();
        let mut nodes = 0;
        let mut moves = MoveList::new();
        self.generate_moves(&mut moves);
        for &mv in &moves {
            let undo = self.make_move(mv);
            let king = self
                .pieces(us, PieceType::King)
                .iter()
                .next()
                .expect("side to move always has a king");
            if !self.is_attacked(king, opp) {
                nodes += if depth == 1 { 1 } else { self.perft(depth - 1) };
            }
            self.unmake_move(mv, undo);
        }
        nodes
    }

    /// Is `sq` attacked by any piece of color `by`?
    ///
    /// # Notes
    /// Super-piece trick: from `sq`, a piece of each type hits an enemy of that
    /// type exactly when such an enemy attacks `sq`.
    #[must_use]
    pub fn is_attacked(&self, sq: Square, by: Color) -> bool {
        let i = sq.index() as usize;
        let knights = self.pieces(by, PieceType::Knight).bits();
        if KNIGHT_ATTACKS[i].bits() & knights != 0 {
            return true;
        }
        let king = self.pieces(by, PieceType::King).bits();
        if KING_ATTACKS[i].bits() & king != 0 {
            return true;
        }
        let pawns = self.pieces(by, PieceType::Pawn).bits();
        if pawn_attacks(1u64 << i, by.flip()) & pawns != 0 {
            return true;
        }
        let occ = self.occupied();
        let queens = self.pieces(by, PieceType::Queen).bits();
        let bishops = self.pieces(by, PieceType::Bishop).bits();
        if bishop_attacks(sq, occ).bits() & (bishops | queens) != 0 {
            return true;
        }
        let rooks = self.pieces(by, PieceType::Rook).bits();
        rook_attacks(sq, occ).bits() & (rooks | queens) != 0
    }

    /// All pieces of either color attacking `sq` under occupancy `occ`.
    ///
    /// # Notes
    /// Sliders are re-derived from `occ` rather than the board's real occupancy,
    /// so a square cleared mid-exchange exposes any x-ray attacker lined up
    /// behind the piece that just left. Pawns/knights/kings do not x-ray; pieces
    /// removed from `occ` are masked out of the result.
    fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let i = sq.index() as usize;
        let bit = 1u64 << i;
        let both =
            |kind| self.pieces(Color::White, kind).bits() | self.pieces(Color::Black, kind).bits();
        let mut bb = 0u64;
        bb |= KNIGHT_ATTACKS[i].bits() & both(PieceType::Knight);
        bb |= KING_ATTACKS[i].bits() & both(PieceType::King);
        // A `by` pawn attacks `sq` iff `sq` lies on the opposite color's capture
        // pattern read from `sq` (the is_attacked trick): white pawns are found
        // via the black attack mask and vice versa.
        bb |= pawn_attacks(bit, Color::White) & self.pieces(Color::Black, PieceType::Pawn).bits();
        bb |= pawn_attacks(bit, Color::Black) & self.pieces(Color::White, PieceType::Pawn).bits();
        let bishops_queens = both(PieceType::Bishop) | both(PieceType::Queen);
        bb |= bishop_attacks(sq, occ).bits() & bishops_queens;
        let rooks_queens = both(PieceType::Rook) | both(PieceType::Queen);
        bb |= rook_attacks(sq, occ).bits() & rooks_queens;
        Bitboard::from_bits(bb & occ.bits())
    }

    /// Static Exchange Evaluation: the net material (centipawns, from the moving
    /// side's point of view) of playing capture `mv` and then letting both sides
    /// recapture on the target square with their least valuable piece until
    /// neither can or wants to.
    ///
    /// `mv` must be a capture or en-passant capture; for any other move there is
    /// no victim and the result is `0`. No board state is mutated. The king is
    /// valued near-infinite so it is never chosen to recapture into a defended
    /// square.
    #[must_use]
    pub fn see(&self, mv: Move) -> i32 {
        let to = mv.to();
        let from = mv.from();
        let mut occ = self.occupied();

        // Victim value, and (for en passant) clear the captured pawn that sits
        // beside `to` so the exchange and any 5th-rank x-ray see it gone.
        let victim = if mv.flag() == MoveFlag::EnPassant {
            let cap_rank = 4 - self.side_to_move().as_u8();
            let cap = Square::new(cap_rank, to.file());
            occ = Bitboard::from_bits(occ.bits() & !(1u64 << cap.index()));
            SEE_VALUE[PieceType::Pawn.as_u8() as usize]
        } else {
            match self.piece_at(to) {
                Some(p) => SEE_VALUE[p.kind().as_u8() as usize],
                None => return 0, // not a capture
            }
        };

        // Value of the piece currently standing on `to`. After our capture that
        // is the moving piece; each recapture replaces it with the recapturer.
        let mut on_square = match self.piece_at(from) {
            Some(p) => SEE_VALUE[p.kind().as_u8() as usize],
            None => return 0,
        };

        // gain[d] is the running swap balance after the d-th capture; folded back
        // with negamax at the end (each side keeps the better of stand/continue).
        let mut gain = [0i32; 32];
        gain[0] = victim;

        // Remove the first attacker so a slider behind it x-rays through.
        occ = Bitboard::from_bits(occ.bits() & !(1u64 << from.index()));
        let mut side = self.side_to_move().flip();
        let mut attackers = self.attackers_to(to, occ);
        let mut top = 0usize;

        loop {
            // Least valuable attacker of the side now to recapture; if none, the
            // exchange ends.
            let mut chosen = 0u64;
            let mut chosen_val = 0;
            for kind in 0..6u8 {
                let bb = attackers.bits() & self.pieces(side, PieceType::from_u8(kind)).bits();
                if bb != 0 {
                    chosen = bb & bb.wrapping_neg(); // isolate one attacker
                    chosen_val = SEE_VALUE[kind as usize];
                    break;
                }
            }
            if chosen == 0 {
                break;
            }
            top += 1;
            gain[top] = on_square - gain[top - 1];
            // If even the optimistic balance is losing for the side that just
            // captured, it would have declined - deeper captures cannot help it.
            if (-gain[top - 1]).max(gain[top]) < 0 {
                break;
            }
            on_square = chosen_val;
            occ = Bitboard::from_bits(occ.bits() & !chosen);
            attackers = self.attackers_to(to, occ); // re-derive: handles x-ray
            side = side.flip();
        }

        // Negamax fold, deepest first: each side keeps max(stand, continue).
        for i in (0..top).rev() {
            gain[i] = -(-gain[i]).max(gain[i + 1]);
        }
        gain[0]
    }

    /// Generate pawn moves.
    fn gen_pawns(&self, us: Color, enemy: Bitboard, occ: Bitboard, out: &mut MoveList) {
        let pawns = self.pieces(us, PieceType::Pawn).bits();
        let empty = !occ.bits();
        let ep_bb = self.en_passant().map_or(0, |s| 1u64 << s.index());

        // Geometry is color-relative: `forward` is the push shift, captures straddle it.
        // West-captures wrap only onto the h-file, east onto the a-file (both colors).
        let (forward, double_rank, promo_rank) = match us {
            Color::White => (
                8i32,
                Bitboard::rank_mask(2).bits(),
                Bitboard::rank_mask(7).bits(),
            ),
            Color::Black => (
                -8i32,
                Bitboard::rank_mask(5).bits(),
                Bitboard::rank_mask(0).bits(),
            ),
        };
        let west = forward - 1;
        let east = forward + 1;

        // Single and double pushes land on empty squares only.
        let single = shift(pawns, forward) & empty;
        let double = shift(single & double_rank, forward) & empty;

        emit_from_targets(single & !promo_rank, forward, MoveFlag::Quiet, out);
        emit_from_targets(double, 2 * forward, MoveFlag::DoublePawnPush, out);
        emit_promotions(single & promo_rank, forward, false, out);

        // Captures: diagonal shift, mask the wrap, hit an enemy or the ep square.
        let targets = enemy.bits() | ep_bb;
        let west_caps = shift(pawns, west) & NOT_FILE_H & targets;
        let east_caps = shift(pawns, east) & NOT_FILE_A & targets;
        for (caps, by) in [(west_caps, west), (east_caps, east)] {
            emit_pawn_captures(caps & !promo_rank, by, ep_bb, out);
            emit_promotions(caps & promo_rank, by, true, out);
        }
    }
}

/// Emit one move per destination bit, with source = destination minus `by`.
fn emit_from_targets(targets: u64, by: i32, flag: MoveFlag, out: &mut MoveList) {
    for to in Bitboard::from_bits(targets) {
        let from = Square::from_index((to.index() as i32 - by) as u8);
        out.push(Move::new(from, to, flag));
    }
}

/// Pawn captures that aren't promotions: an `ep`-square hit is en passant,
/// everything else an ordinary capture.
fn emit_pawn_captures(targets: u64, by: i32, ep_bb: u64, out: &mut MoveList) {
    for to in Bitboard::from_bits(targets) {
        let from = Square::from_index((to.index() as i32 - by) as u8);
        let flag = if (1u64 << to.index()) == ep_bb {
            MoveFlag::EnPassant
        } else {
            MoveFlag::Capture
        };
        out.push(Move::new(from, to, flag));
    }
}

fn emit_promotions(targets: u64, by: i32, capture: bool, out: &mut MoveList) {
    for to in Bitboard::from_bits(targets) {
        let from = Square::from_index((to.index() as i32 - by) as u8);
        push_promotions(from, to, capture, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(fen: &str) -> usize {
        Board::from_fen(fen.as_bytes())
            .unwrap()
            .pseudo_legal_moves()
            .len()
    }

    /// SEE of the pseudo-legal capture `from`->`to` in `fen`. Resolves the flag
    /// (Capture vs EnPassant) by matching against generated moves.
    fn see_of(fen: &str, from: &str, to: &str) -> i32 {
        let board = Board::from_fen(fen.as_bytes()).unwrap();
        let from: Square = from.parse().unwrap();
        let to: Square = to.parse().unwrap();
        let mv = board
            .pseudo_legal_moves()
            .iter()
            .copied()
            .find(|m| m.from() == from && m.to() == to)
            .expect("capture must be pseudo-legal");
        board.see(mv)
    }

    #[test]
    fn see_wins_an_undefended_pawn() {
        // d4 pawn takes an undefended e5 pawn: +100, no recapture.
        assert_eq!(see_of("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1", "d4", "e5"), 100);
    }

    #[test]
    fn see_loses_queen_for_a_defended_pawn() {
        // Qh2xe5 grabs a pawn defended by the d6 pawn: +100 - 900 = -800.
        assert_eq!(
            see_of("4k3/8/3p4/4p3/8/8/7Q/4K3 w - - 0 1", "h2", "e5"),
            -800
        );
    }

    #[test]
    fn see_equal_trade_is_zero() {
        // d4xe5 pawn trade, recaptured by the f6 pawn: +100 - 100 = 0.
        assert_eq!(see_of("4k3/8/5p2/4p3/3P4/8/8/4K3 w - - 0 1", "d4", "e5"), 0);
    }

    #[test]
    fn see_wins_a_defended_queen_with_a_pawn() {
        // d4 pawn takes a queen defended by the d6 pawn: +900 - 100 = +800.
        assert_eq!(
            see_of("4k3/8/3p4/4q3/3P4/8/8/4K3 w - - 0 1", "d4", "e5"),
            800
        );
    }

    #[test]
    fn see_sees_through_an_xray_battery() {
        // Re2xe5 with Re1 stacked behind it, vs a single e8 rook defending the
        // pawn. The x-ray makes the back rook a real second attacker, so white
        // wins pawn + rook for one rook: +100. Without x-ray it would read -400.
        assert_eq!(
            see_of("4r1k1/8/8/4p3/8/8/4R3/4R1K1 w - - 0 1", "e2", "e5"),
            100
        );
    }

    #[test]
    fn see_handles_en_passant() {
        // e5xd6 e.p. removes the d5 pawn (undefended): +100.
        assert_eq!(see_of("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5", "d6"), 100);
    }

    #[test]
    fn see_resolves_a_bishop_queen_battery() {
        // Diagonal battery a1-h8: Bb2 in front, Qa1 behind it. Bxe5 wins the
        // pawn; Nc6 recaptures the bishop; the queen x-rays through and takes the
        // knight. White: +pawn +knight -bishop = +100 +320 -330 = +90.
        assert_eq!(
            see_of("6k1/8/2n5/4p3/8/8/1B6/Q5K1 w - - 0 1", "b2", "e5"),
            90
        );
    }

    #[test]
    fn see_resolves_a_triple_file_battery() {
        // e-file: white Re3/Re2/Qe1 stacked, black Re7/Re8 defending. Geometry
        // forces front-first on each side (a blocked piece is not yet an
        // attacker), so the queen only joins last. Five captures resolve to the
        // pawn plus an even rook-for-rook swap: +100.
        assert_eq!(
            see_of("4r2k/4r3/8/4p3/8/4R3/4R3/4Q2K w - - 0 1", "e3", "e5"),
            100
        );
    }

    #[test]
    fn see_ignores_a_pin_against_the_king() {
        // The d6 bishop is absolutely pinned to its king by Rd1, so it cannot
        // legally recapture on e5 - the pawn is really hanging (true value +100).
        // SEE has no legality model: it counts the pinned bishop as a defender
        // and reports Qxe5 as losing the queen for a pawn, -800. This documents
        // the accepted blind spot, it is not a defect to "fix".
        assert_eq!(
            see_of("3k4/8/3b4/4p3/8/8/7Q/3R3K w - - 0 1", "h2", "e5"),
            -800
        );
    }

    #[test]
    fn startpos_has_twenty_moves() {
        // 16 pawn moves + 4 knight moves; none of the 20 expose the king, so
        // pseudo-legal == legal here.
        assert_eq!(
            count("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            20
        );
    }

    #[test]
    fn lone_rook_sweeps_rank_and_file() {
        // Rook on d5: 7 along the rank + 7 along the file = 14. King on h1: 3.
        let board = Board::from_fen(b"7k/8/8/3R4/8/8/8/7K w - - 0 1").unwrap();
        let moves = board.pseudo_legal_moves();
        assert_eq!(moves.len(), 17);
        assert!(moves.iter().all(|m| !m.flag().is_capture()));
    }

    #[test]
    fn pawn_promotes_four_ways() {
        // White pawn on a7 can promote on a8 (empty) four ways, no capture.
        let board = Board::from_fen(b"4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let moves = board.pseudo_legal_moves();
        let promos = moves.iter().filter(|m| m.flag().is_promotion()).count();
        assert_eq!(promos, 4);
        assert!(moves.iter().all(|m| !m.flag().is_capture()));
    }

    #[test]
    fn en_passant_is_flagged() {
        // Black just played ...c7-c5; white pawn on d5 can take ep on c6.
        let board =
            Board::from_fen(b"rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2")
                .unwrap();
        let ep = board
            .pseudo_legal_moves()
            .into_iter()
            .filter(|m| m.flag() == MoveFlag::EnPassant)
            .collect::<Vec<_>>();
        assert_eq!(ep.len(), 1);
        assert_eq!(ep[0].to(), Square::from_ascii(b"c6").unwrap());
    }

    fn sq(s: &str) -> Square {
        Square::from_ascii(s.as_bytes()).unwrap()
    }

    #[test]
    fn attacks_by_each_piece_type() {
        // Knight d4 hits e6/c6 but not e5; king e1 hits d2 not d3; white pawn e4
        // hits d5/f5 not e5; queen d1 hits along rank/file/diagonal.
        let b = Board::from_fen(b"8/8/8/8/3NP3/8/8/3QK3 w - - 0 1").unwrap();
        assert!(b.is_attacked(sq("e6"), Color::White));
        assert!(b.is_attacked(sq("c6"), Color::White));
        assert!(!b.is_attacked(sq("e5"), Color::White)); // neither knight nor pawn
        assert!(b.is_attacked(sq("d5"), Color::White)); // pawn e4
        assert!(b.is_attacked(sq("f5"), Color::White)); // pawn e4
        assert!(b.is_attacked(sq("d2"), Color::White)); // king e1 (and queen d1)
        assert!(b.is_attacked(sq("a1"), Color::White)); // queen d1 along rank 1
        assert!(b.is_attacked(sq("a4"), Color::White)); // queen d1 diagonal
    }

    #[test]
    fn slider_attack_is_blocked_by_occupancy() {
        // White rook a1 sees up the a-file to a8 when empty...
        let open = Board::from_fen(b"7k/8/8/8/8/8/8/R6K w - - 0 1").unwrap();
        assert!(open.is_attacked(sq("a8"), Color::White));
        // ...but a black pawn on a4 blocks the ray; a8 is no longer attacked,
        // a4 (the blocker square itself) is.
        let blocked = Board::from_fen(b"7k/8/8/8/p7/8/8/R6K w - - 0 1").unwrap();
        assert!(!blocked.is_attacked(sq("a8"), Color::White));
        assert!(blocked.is_attacked(sq("a4"), Color::White));
    }

    #[test]
    fn attacker_color_is_respected() {
        // A lone white rook on d4 attacks d8 for White, never for Black.
        let b = Board::from_fen(b"7k/8/8/8/3R4/8/8/7K w - - 0 1").unwrap();
        assert!(b.is_attacked(sq("d8"), Color::White));
        assert!(!b.is_attacked(sq("d8"), Color::Black));
    }

    #[test]
    fn castling_is_generated_when_legal() {
        // White to move, both rooks home, nothing attacking the king's path.
        let b = Board::from_fen(b"4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        let castles: Vec<_> = b
            .pseudo_legal_moves()
            .into_iter()
            .filter(|m| matches!(m.flag(), MoveFlag::KingCastle | MoveFlag::QueenCastle))
            .collect();
        assert_eq!(castles.len(), 2);
    }

    #[test]
    fn castling_blocked_through_check_is_rejected() {
        // Black rook on f8 attacks f1: the king would pass through check
        // kingside (f1) -> no O-O. Queenside path (e1,d1,c1) is clear -> O-O-O ok.
        let b = Board::from_fen(b"4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        let flags: Vec<_> = b
            .pseudo_legal_moves()
            .into_iter()
            .map(|m| m.flag())
            .filter(|f| matches!(f, MoveFlag::KingCastle | MoveFlag::QueenCastle))
            .collect();
        assert_eq!(flags, vec![MoveFlag::QueenCastle]);
    }

    fn perft(fen: &str, depth: u32) -> u64 {
        Board::from_fen(fen.as_bytes()).unwrap().perft(depth)
    }

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // "Kiwipete": dense, every special move (castling, ep, promotions) in play.
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    // CPW position 3: sparse but heavy on checks and en passant.
    const POSITION3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

    #[test]
    fn perft_startpos() {
        assert_eq!(perft(STARTPOS, 1), 20);
        assert_eq!(perft(STARTPOS, 2), 400);
        assert_eq!(perft(STARTPOS, 3), 8902);
        assert_eq!(perft(STARTPOS, 4), 197_281);
    }

    #[test]
    fn perft_kiwipete() {
        assert_eq!(perft(KIWIPETE, 1), 48);
        assert_eq!(perft(KIWIPETE, 2), 2039);
        assert_eq!(perft(KIWIPETE, 3), 97_862);
    }

    #[test]
    fn perft_position3() {
        assert_eq!(perft(POSITION3, 1), 14);
        assert_eq!(perft(POSITION3, 2), 191);
        assert_eq!(perft(POSITION3, 3), 2812);
        assert_eq!(perft(POSITION3, 4), 43_238);
    }

    #[test]
    #[ignore = "slow: ~5M+ nodes, run with `cargo test -- --ignored`"]
    fn perft_deep() {
        assert_eq!(perft(STARTPOS, 5), 4_865_609);
        assert_eq!(perft(KIWIPETE, 4), 4_085_603);
    }
}
