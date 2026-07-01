//! Static evaluation: tapered material and piece square tables (PeSTO).
//!
//! Each term carries a middlegame (`mg`) and endgame (`eg`) value; the two are
//! blended by a `phase` scalar derived from the remaining non-pawn material, so
//! the evaluation slides continuously from opening to endgame (most visibly,
//! the king is driven to safety in the middlegame and to the centre in the
//! endgame). Values are Ronald Friederich's PeSTO tables.
//!
//! The per-(piece, square) terms are maintained incrementally by the board (its
//! accumulators are updated on every piece placement). This module only clamps
//! the phase and blends the two sums.

use crate::{Board, Color, PieceType};

use crate::Score;

/// Maximum game phase (full starting non-pawn material).
const PHASE_MAX: i32 = 24;

/// Per-piece mobility weights, centipawns per reachable square in the mobility
/// area, as `(kind, mg, eg)`. Deliberately small: a gentle nudge orders the
/// search without distorting the material balance (a coarse local fixed-depth
/// sweep preferred these over weights twice as large). Rooks value mobility more
/// in the endgame; the king and pawns are not mobility-scored.
const MOBILITY: [(PieceType, i32, i32); 4] = [
    (PieceType::Knight, 2, 2),
    (PieceType::Bishop, 2, 2),
    (PieceType::Rook, 1, 2),
    (PieceType::Queen, 1, 1),
];

/// White-relative `(mg, eg)` mobility differential: for each knight, bishop,
/// rook, and queen, the number of squares it reaches that are neither occupied
/// by a friendly piece nor attacked by an enemy pawn, weighted by piece and
/// game phase. Symmetric positions (e.g. the start) net to zero.
fn mobility(board: &Board) -> (i32, i32) {
    // A side's mobility area excludes its own pieces and any square an enemy pawn
    // covers (a square a piece nominally reaches but a pawn guards is not real
    // mobility).
    let white_area = !board.occupied_by(Color::White) & !board.pawn_attack_span(Color::Black);
    let black_area = !board.occupied_by(Color::Black) & !board.pawn_attack_span(Color::White);
    let mut mg = 0;
    let mut eg = 0;
    for (kind, wmg, weg) in MOBILITY {
        for sq in board.pieces(Color::White, kind) {
            let n = (board.attacks_from(sq, kind) & white_area).count() as i32;
            mg += n * wmg;
            eg += n * weg;
        }
        for sq in board.pieces(Color::Black, kind) {
            let n = (board.attacks_from(sq, kind) & black_area).count() as i32;
            mg -= n * wmg;
            eg -= n * weg;
        }
    }
    (mg, eg)
}

/// Static evaluation of `board`, from the side-to-move's perspective.
#[must_use]
pub fn evaluate(board: &Board) -> Score {
    // The board keeps White-relative middlegame/endgame/phase sums up to date;
    // promotions can push the phase past the starting maximum, so clamp it.
    let phase = board.eval_phase().min(PHASE_MAX);
    let (mob_mg, mob_eg) = mobility(board);
    let mg = board.eval_mg() + mob_mg;
    let eg = board.eval_eg() + mob_eg;
    let score = (mg * phase + eg * (PHASE_MAX - phase)) / PHASE_MAX;

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
    fn startpos_mobility_is_symmetric() {
        assert_eq!(mobility(&board(STARTPOS)), (0, 0));
    }

    #[test]
    fn central_knight_outmoves_cornered() {
        // White Ne4 reaches 8 squares; Black Na8 only 2. White mobility leads.
        let (mg, eg) = mobility(&board("n3k3/8/8/8/4N3/8/8/4K3 w - - 0 1"));
        assert!(
            mg > 0 && eg > 0,
            "white mobility should lead: mg={mg} eg={eg}"
        );
    }

    #[test]
    fn a_missing_pawn_favours_the_side_still_holding_it() {
        // White to move, Black is down its e-pawn: the score must be positive,
        // and flipping the side to move must flip the sign.
        let white_up = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let black_to_move = "rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
        assert!(evaluate(&board(white_up)) > 0);
        assert_eq!(evaluate(&board(black_to_move)), -evaluate(&board(white_up)));
    }

    #[test]
    fn advanced_pawn_beats_home_pawn() {
        // Orientation guard: a white pawn one step from promotion must outscore
        // one on its starting square.
        let advanced = evaluate(&board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"));
        let home = evaluate(&board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"));
        assert!(advanced > home, "advanced={advanced} home={home}");
    }

    #[test]
    fn endgame_king_wants_the_centre() {
        // Tapering tripwire: at phase 0 the endgame king table rewards a central
        // king. A non-tapered eval would read the middlegame table, which
        // punishes the centre, and score the cornered king higher instead.
        let central = evaluate(&board("4k3/8/8/8/4K3/8/8/8 w - - 0 1"));
        let cornered = evaluate(&board("4k3/8/8/8/8/8/8/K7 w - - 0 1"));
        assert!(central > cornered, "central={central} cornered={cornered}");
    }

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}
