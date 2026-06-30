//! Static evaluation: tapered material and piece square tables (PeSTO).
//!
//! Each term carries a middlegame (`mg`) and endgame (`eg`) value; the two are
//! blended by a `phase` scalar derived from the remaining non-pawn material, so
//! the evaluation slides continuously from opening to endgame (most visibly,
//! the king is driven to safety in the middlegame and to the centre in the
//! endgame). Values are Ronald Friederich's PeSTO tables.
//!
//! The per-(piece, square) terms are maintained incrementally by the board (see
//! the board's accumulators, updated on every piece placement beside the
//! Zobrist hash). This module only clamps the phase and blends the two sums.

use crate::{Board, Color};

use crate::Score;

/// Maximum game phase (full starting non-pawn material).
const PHASE_MAX: i32 = 24;

/// Static evaluation of `board`, from the side-to-move's perspective.
#[must_use]
pub fn evaluate(board: &Board) -> Score {
    // The board keeps White-relative middlegame/endgame/phase sums up to date;
    // promotions can push the phase past the starting maximum, so clamp it.
    let phase = board.eval_phase().min(PHASE_MAX);
    let score = (board.eval_mg() * phase + board.eval_eg() * (PHASE_MAX - phase)) / PHASE_MAX;

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
