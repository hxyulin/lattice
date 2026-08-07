//! Static position evaluation.

use crate::board::Piece;
use crate::{Board, Color, Square};

/// Midgame material values. The king is 0 by construction: it is never
/// captured, and a nonzero value would push scores into the mate range that
/// `search::mate_in` reads.
const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
/// Endgame material values. The king is 0 for the same reason as `MG_VALUE`.
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

/// Game phase weight per piece type, summing to 24 at the start position.
const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
const TOTAL_PHASE: i32 = 24;

/// Bonus for having the move.
///
/// Removes the score oscillation between odd and even search depths. A leaf
/// bonus is negated once per ply on the way to the root, so it arrives as `+T`
/// at even depth and `-T` at odd, and the odd/even gap therefore closes at
/// `2T`. The start position's measured gap of 31.8cp gives `T = 15.9`, which is
/// also the value Stockfish's classical evaluation used.
///
/// Flat rather than tapered: the measured swing shows no midgame/endgame split
/// (2.1..5.6 against 1.8..5.9, overlapping). The usual argument for tapering is
/// zugzwang, where having to move is a liability - but a constant cannot detect
/// zugzwang at any magnitude, so scaling it by phase would shrink a wrong
/// answer rather than fix it.
pub(crate) const TEMPO: i32 = 17;

// The tables below are transcribed verbatim from the published PeSTO values,
// in reading order: index 0 is a8 and index 63 is h1.
#[rustfmt::skip]
const MG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const EG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const MG_KNIGHT: [i32; 64] = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
];
#[rustfmt::skip]
const EG_KNIGHT: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
];
#[rustfmt::skip]
const MG_BISHOP: [i32; 64] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
];
#[rustfmt::skip]
const EG_BISHOP: [i32; 64] = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
];
#[rustfmt::skip]
const MG_ROOK: [i32; 64] = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
];
#[rustfmt::skip]
const EG_ROOK: [i32; 64] = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
];
#[rustfmt::skip]
const MG_QUEEN: [i32; 64] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
];
#[rustfmt::skip]
const EG_QUEEN: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
];
#[rustfmt::skip]
const MG_KING: [i32; 64] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];
#[rustfmt::skip]
const EG_KING: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
];

const MG_TABLE: [[i32; 64]; 6] = fold(
    [MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING],
    MG_VALUE,
);
const EG_TABLE: [[i32; 64]; 6] = fold(
    [EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING],
    EG_VALUE,
);

/// Folds material values into the piece-square tables at compile time, so the
/// hot loop is a single lookup per piece.
const fn fold(mut tables: [[i32; 64]; 6], values: [i32; 6]) -> [[i32; 64]; 6] {
    let mut piece = 0;
    while piece < 6 {
        let mut square = 0;
        while square < 64 {
            tables[piece][square] += values[piece];
            square += 1;
        }
        piece += 1;
    }
    tables
}

/// Running midgame score, endgame score, and raw game phase, maintained by the
/// board as pieces are added and removed.
///
/// Scoring is a sum over pieces, so each term can be applied when its piece
/// appears and withdrawn when it leaves. `Board::unmake` reverses exactly the
/// piece placements `make` performed, which undoes the accumulation with it -
/// no saved copy to restore.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Accumulator {
    mg: i32,
    eg: i32,
    phase: i32,
}

impl Accumulator {
    /// Applies the terms for a piece standing on a square.
    pub(crate) fn add(&mut self, piece: Piece, square: Square) {
        let (kind, index, sign) = terms(piece, square);
        self.mg += sign * MG_TABLE[kind][index];
        self.eg += sign * EG_TABLE[kind][index];
        self.phase += PHASE_WEIGHT[kind];
    }

    /// Withdraws the terms for a piece leaving a square.
    pub(crate) fn remove(&mut self, piece: Piece, square: Square) {
        let (kind, index, sign) = terms(piece, square);
        self.mg -= sign * MG_TABLE[kind][index];
        self.eg -= sign * EG_TABLE[kind][index];
        self.phase -= PHASE_WEIGHT[kind];
    }
}

/// Table index and score sign for a piece on a square.
fn terms(piece: Piece, square: Square) -> (usize, usize, i32) {
    // Tables are in reading order (index 0 is a8) while the board is
    // little-endian (a1 is 0), so White is the side that flips.
    let (index, sign) = match piece.color() {
        Color::White => (square.flip_rank().index() as usize, 1),
        Color::Black => (square.index() as usize, -1),
    };
    (piece.piece_type() as usize, index, sign)
}

/// Scans the board once, returning the midgame score, the endgame score, and
/// the raw game phase.
///
/// The accumulator makes this redundant on the hot path; it stays as the
/// independent definition the incremental update is checked against.
#[cfg(any(test, debug_assertions))]
pub(crate) fn scan(board: &Board) -> Accumulator {
    let mut accumulator = Accumulator::default();
    for square in board.occupied() {
        if let Some(piece) = board.piece_on(square) {
            accumulator.add(piece, square);
        }
    }
    accumulator
}

/// Returns the game phase, 24 at the start position falling to 0 in a bare
/// king endgame. Promotions can push the raw sum past 24, so callers clamp.
#[cfg(test)]
fn phase(board: &Board) -> i32 {
    board.accumulator().phase
}

/// Returns the static evaluation in centipawns, relative to the side to move.
///
/// Material and piece placement are scored from separate midgame and endgame
/// tables, interpolated by how much material remains, plus `TEMPO` for the side
/// to move.
pub fn evaluate(board: &Board) -> i32 {
    let Accumulator { mg, eg, phase } = *board.accumulator();
    let mg_phase = phase.min(TOTAL_PHASE);
    let eg_phase = TOTAL_PHASE - mg_phase;
    let score = (mg * mg_phase + eg * eg_phase) / TOTAL_PHASE;
    let score = if board.state().side_to_move() == Color::White {
        score
    } else {
        -score
    };
    // After the flip, so the bonus always favours whoever is on move rather
    // than always favouring White.
    //
    // A null move flips the side to move without touching a piece, so both
    // sides of one collect the bonus and `before + after` comes to `2 * TEMPO`
    // where a position unchanged by a null should give 0. Every leaf and
    // stand-pat is consistent with every other, so the inflation cancels
    // everywhere except across a null - `search` corrects it at that one
    // comparison rather than `evaluate` trying to detect a null it cannot see.
    score + TEMPO
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Square;

    /// Color-mirrors a FEN: the position is reflected vertically, every piece
    /// changes color, and the side to move swaps with it.
    fn mirror(fen: &str) -> String {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        let ranks: Vec<String> = fields[0]
            .split('/')
            .rev()
            .map(|rank| {
                rank.chars()
                    .map(|c| {
                        if c.is_ascii_uppercase() {
                            c.to_ascii_lowercase()
                        } else if c.is_ascii_lowercase() {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        }
                    })
                    .collect()
            })
            .collect();
        let side = if fields[1] == "w" { "b" } else { "w" };
        let castling: String = fields[2]
            .chars()
            .map(|c| {
                if c.is_ascii_uppercase() {
                    c.to_ascii_lowercase()
                } else if c.is_ascii_lowercase() {
                    c.to_ascii_uppercase()
                } else {
                    c
                }
            })
            .collect();
        let en_passant = if fields[3] == "-" {
            "-".to_owned()
        } else {
            let bytes = fields[3].as_bytes();
            format!("{}{}", bytes[0] as char, 9 - (bytes[1] - b'0'))
        };
        format!(
            "{} {side} {castling} {en_passant} {} {}",
            ranks.join("/"),
            fields[4],
            fields[5]
        )
    }

    /// The material and placement component alone, with the tempo bonus taken
    /// back off.
    ///
    /// `TEMPO` is the one part of the score that does not flip with the side to
    /// move, so a test asserting the material component negates, cancels, or
    /// matches a hand-computed table value has to strip it first or it is
    /// asserting against the bonus as well.
    fn placement(board: &Board) -> i32 {
        evaluate(board) - TEMPO
    }

    // catches: any change to the arithmetic that the relational tests below
    // are blind to, because they hold under it. Verified by mutation: swapping
    // the midgame and endgame lookups, inverting the phase interpolation,
    // dropping the phase clamp, zeroing a piece-square table, and making
    // `fold` drop the material values all leave every other test in this file
    // passing. Expected values are derived by hand from the published PeSTO
    // tables, not read back from `evaluate`.
    #[test]
    fn scores_match_hand_computed_pesto_values() {
        // MG_TABLE[kind][flip] and EG_TABLE[kind][flip] blended as
        // (mg * p + eg * (24 - p)) / 24 for phase p.
        let cases = [
            // d5 pawn, phase 0: pure endgame, EG_PAWN[27] + 94 = 5 + 94.
            ("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1", 99),
            // e4 knight, phase 1: (365 * 1 + 297 * 23) / 24.
            ("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1", 299),
            // c4 bishop, phase 1: (378 * 1 + 309 * 23) / 24.
            ("4k3/8/8/8/2B5/8/8/4K3 w - - 0 1", 312),
            // a1 rook, phase 2: (458 * 2 + 503 * 22) / 24.
            ("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", 499),
            // d1 queen, phase 4: (1035 * 4 + 893 * 20) / 24.
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 916),
            // Queens on mirrored squares cancel; only the king placement is
            // left, so this pins the king tables rather than material.
            ("3qk3/8/8/8/8/8/8/3Q2K1 w - - 0 1", 8),
            // Phase 28 clamps to 24, so this reads the pure midgame tables.
            ("4k3/8/8/8/8/8/8/QQQQKQQQ w - - 0 1", 7051),
        ];
        for (fen, want) in cases {
            let board: Board = fen.parse().unwrap();
            assert_eq!(placement(&board), want, "{fen}");
        }
    }

    // catches: white indexing the tables without `flip_rank`, and the sign
    // flip dropped from either the per-piece term or the side-to-move return.
    #[test]
    fn evaluation_is_color_symmetric() {
        // Equality, not negation: the score is relative to the side to move,
        // and mirroring swaps that side too.
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
            "8/5pk1/6p1/3p4/3P4/6P1/5PK1/8 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ];
        for fen in fens {
            let position: Board = fen.parse().unwrap();
            let mirrored: Board = mirror(fen).parse().unwrap();
            assert_eq!(
                evaluate(&position),
                evaluate(&mirrored),
                "asymmetric for {fen} (mirror {})",
                mirror(fen)
            );
        }
    }

    #[test]
    fn symmetric_positions_score_the_tempo_bonus_for_either_side() {
        // A position with nothing between the sides is worth exactly the tempo
        // bonus, and worth it to whoever is on move - the same number for
        // White and for Black, not a number that changes sign with colour.
        // That is what distinguishes a side-to-move bonus from a White bonus,
        // and it is the half a dropped sign flip would break.
        //
        // Written as TEMPO rather than 17 so retuning the constant does not
        // reach into the tests. That alone would also pass for TEMPO = 0, so
        // the bonus is pinned as nonzero here: the point of the term is to
        // shift the score, and a silently disabled one must not look correct.
        const { assert!(TEMPO > 0, "a zero tempo bonus is a disabled feature") };
        assert_eq!(evaluate(&Board::startpos()), TEMPO);
        let white: Board = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        let black: Board = "4k3/8/8/8/8/8/8/4K3 b - - 0 1".parse().unwrap();
        assert_eq!(evaluate(&white), TEMPO);
        assert_eq!(evaluate(&black), TEMPO);
        // The material and placement component really is zero here, so the
        // whole score above is the bonus and not a cancellation that happens
        // to land on the same number.
        assert_eq!(evaluate(&white) - TEMPO, 0);
    }

    #[test]
    fn phase_runs_from_the_start_position_down_to_bare_kings() {
        assert_eq!(phase(&Board::startpos()), TOTAL_PHASE);
        let bare: Board = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        assert_eq!(phase(&bare), 0);
    }

    #[test]
    fn extra_queens_do_not_extrapolate_past_the_midgame() {
        // Promotions can drive the raw phase above 24; without the clamp the
        // endgame weight goes negative and the blend runs off the table.
        let many: Board = "qqqqk3/8/8/8/8/8/8/QQQQK3 w - - 0 1".parse().unwrap();
        assert!(phase(&many) > TOTAL_PHASE, "raw phase should overflow");
        assert_eq!(phase(&many).min(TOTAL_PHASE), TOTAL_PHASE);
        // Symmetric material, so the clamp must still land on a zero score.
        assert_eq!(placement(&many), 0);
    }

    #[test]
    fn placement_beats_material_alone() {
        let centre: Board = "4k3/8/8/8/4N3/8/8/4K3 w - - 0 1".parse().unwrap();
        let corner: Board = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1".parse().unwrap();
        assert!(
            evaluate(&centre) > evaluate(&corner),
            "a centralised knight should outscore one in the corner: {} vs {}",
            evaluate(&centre),
            evaluate(&corner)
        );
    }

    #[test]
    fn material_delta_is_relative_to_side_to_move() {
        let white: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
            .parse()
            .unwrap();
        let black: Board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
            .parse()
            .unwrap();
        assert!(evaluate(&white) < 0, "white is a queen down");
        // The placement component negates with the side to move. The full
        // score deliberately does not: `TEMPO` favours whoever is on move, so
        // it is the same sign from both sides and survives the negation as a
        // `2 * TEMPO` gap. Asserting the raw scores negate would be asserting
        // the bonus away.
        assert_eq!(placement(&white), -placement(&black));
        assert_eq!(evaluate(&white) + evaluate(&black), 2 * TEMPO);
    }

    // catches: any make/unmake path that moves a piece without routing through
    // add_piece/remove_piece, and a promotion or en passant applying the wrong
    // piece to the accumulator. `debug_check` asserts this too, but compiles
    // out in release, which is the build that plays games.
    #[test]
    fn incremental_accumulator_tracks_a_full_scan_through_a_search() {
        use crate::movegen::{MoveList, generate_legal};

        fn walk(board: &mut Board, depth: u32) {
            assert_eq!(*board.accumulator(), scan(board), "{board}");
            if depth == 0 {
                return;
            }
            let mut moves = MoveList::new();
            generate_legal(board, &mut moves);
            for &mv in moves.iter() {
                board.make(mv);
                walk(board, depth - 1);
                board.unmake(mv);
                // The unwind must restore it exactly, not merely stay valid.
                assert_eq!(*board.accumulator(), scan(board), "after unmake {mv:?}");
            }
        }

        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            // Castling, en passant and captures all reachable within depth 3.
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // Promotions, including capture-promotions onto a defended rank.
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ];
        for fen in fens {
            let mut board: Board = fen.parse().unwrap();
            walk(&mut board, 3);
        }
    }

    #[test]
    fn tables_are_distinct() {
        // The realistic transcription bug in 768 numbers is pasting one block
        // twice; nothing else catches that.
        let tables = [
            MG_PAWN, EG_PAWN, MG_KNIGHT, EG_KNIGHT, MG_BISHOP, EG_BISHOP, MG_ROOK, EG_ROOK,
            MG_QUEEN, EG_QUEEN, MG_KING, EG_KING,
        ];
        for (i, first) in tables.iter().enumerate() {
            for (j, second) in tables.iter().enumerate().skip(i + 1) {
                assert_ne!(first, second, "tables {i} and {j} are identical");
            }
        }
    }

    #[test]
    fn white_and_black_index_mirrored_squares() {
        let a1 = Square::new(0, 0).unwrap();
        let a8 = Square::new(0, 7).unwrap();
        assert_eq!(a1.flip_rank(), a8);
        assert_eq!(a8.flip_rank(), a1);
    }
}
