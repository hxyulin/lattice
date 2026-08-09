//! Static position evaluation.

use crate::board::{Piece, PieceType};
use crate::movegen::{bishop_attacks, knight_attacks, queen_attacks, rook_attacks};
use crate::{Bitboard, Board, Color, Square};

#[path = "eval_weights.rs"]
mod weights;

/// Complete set of linear static-evaluation weights.
///
/// Piece-square entries already include material. Keeping this as one typed
/// value gives the tuner an unambiguous parameterization while the engine
/// still reads compile-time constants on its hot path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvalWeights {
    pub(crate) mg_table: [[i32; 64]; 6],
    pub(crate) eg_table: [[i32; 64]; 6],
    pub(crate) mobility_mg: [i32; 6],
    pub(crate) mobility_eg: [i32; 6],
    pub(crate) rook_open_mg: i32,
    pub(crate) rook_open_eg: i32,
    pub(crate) rook_semi_mg: i32,
    pub(crate) rook_semi_eg: i32,
    pub(crate) passed_mg: [i32; 7],
    pub(crate) passed_eg: [i32; 7],
    pub(crate) isolated_mg: i32,
    pub(crate) isolated_eg: i32,
    pub(crate) doubled_mg: i32,
    pub(crate) doubled_eg: i32,
    pub(crate) tempo: i32,
}

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
        self.mg += sign * weights::WEIGHTS.mg_table[kind][index];
        self.eg += sign * weights::WEIGHTS.eg_table[kind][index];
        self.phase += PHASE_WEIGHT[kind];
    }

    /// Withdraws the terms for a piece leaving a square.
    pub(crate) fn remove(&mut self, piece: Piece, square: Square) {
        let (kind, index, sign) = terms(piece, square);
        self.mg -= sign * weights::WEIGHTS.mg_table[kind][index];
        self.eg -= sign * weights::WEIGHTS.eg_table[kind][index];
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

/// Interpolates a midgame and an endgame score by how much material remains.
fn blend(mg: i32, eg: i32, phase: i32) -> i32 {
    let mg_phase = phase.min(TOTAL_PHASE);
    (mg * mg_phase + eg * (TOTAL_PHASE - mg_phase)) / TOTAL_PHASE
}

/// Midgame and endgame centipawns per mobility square, indexed by piece type.
///
/// Scalar rather than the per-count tables larger engines use: those encode a
/// diminishing return that a 300-position fit here could not resolve from the
/// noise, and a knob that cannot be measured is a knob that cannot be tuned.
/// Pawns and kings score nothing - a pawn's attacks are structure, not
/// freedom, and a king with many squares in the midgame is an exposed king,
/// which is the opposite sign.
const MOBILITY_MG: [i32; 6] = [0, 4, 4, 2, 1, 0];
const MOBILITY_EG: [i32; 6] = [0, 4, 5, 4, 6, 0];

/// Mobility for one side, in midgame and endgame centipawns.
///
/// Squares occupied by our own pieces are excluded (they are not available),
/// and so are squares an enemy pawn attacks: a knight standing where a pawn
/// can take it is not mobile there in any sense the search will believe.
/// Enemy pieces are counted, because attacking one is a real option.
fn mobility(board: &Board, us: Color) -> (i32, i32) {
    let occ = board.occupied();
    let ours = board.color(us);
    let their_pawns = board.pieces(PieceType::Pawn) & board.color(us.flip());
    let pawn_attacks = pawn_attack_span(their_pawns, us.flip());
    let available = !(ours | pawn_attacks);
    let (mut mg, mut eg) = (0, 0);
    for kind in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        let index = kind as usize;
        for square in board.pieces(kind) & ours {
            let attacks = match kind {
                PieceType::Knight => knight_attacks(square),
                PieceType::Bishop => bishop_attacks(square, occ),
                PieceType::Rook => rook_attacks(square, occ),
                _ => queen_attacks(square, occ),
            };
            let count = (attacks & available).count() as i32;
            mg += weights::WEIGHTS.mobility_mg[index] * count;
            eg += weights::WEIGHTS.mobility_eg[index] * count;
        }
    }
    (mg, eg)
}

/// Every square attacked by a set of pawns of one color.
fn pawn_attack_span(pawns: Bitboard, color: Color) -> Bitboard {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    const FILE_H: u64 = FILE_A << 7;
    let bits = pawns.bits();
    Bitboard::new(match color {
        Color::White => ((bits & !FILE_A) << 7) | ((bits & !FILE_H) << 9),
        Color::Black => ((bits & !FILE_A) >> 9) | ((bits & !FILE_H) >> 7),
    })
}

/// Bonus for a rook on a file with no pawns of either color: the rook sees the
/// whole board along it, which is most of what a rook is for.
const ROOK_OPEN_MG: i32 = 26;
const ROOK_OPEN_EG: i32 = 12;

/// Bonus for a rook on a file with no pawn of its own colour but one of the
/// enemy's. Less than open - the enemy pawn blocks the file - but the rook
/// still bears on a pawn that cannot be defended by another pawn on the file.
const ROOK_SEMI_MG: i32 = 11;
const ROOK_SEMI_EG: i32 = 6;

/// Rook file bonuses for one side, in midgame and endgame centipawns.
///
/// The endgame weight is the smaller one: with the board emptying, most files
/// are open and the distinction stops separating good rooks from bad ones.
/// `rook_files`, reading the cached open-file masks instead of scanning the
/// pawns again.
///
/// Same score by construction: "no pawn of ours on this file" and "no pawn of
/// theirs on this file" are exactly the two tests the scanning version makes,
/// and both are answered by a bit of the cached mask.
fn rook_files_cached(board: &Board, us: Color, pawns: &PawnEntry) -> (i32, i32) {
    let ours_empty = pawns.no_pawn[us as usize];
    let theirs_empty = pawns.no_pawn[us.flip() as usize];
    let (mut mg, mut eg) = (0, 0);
    for square in board.pieces(PieceType::Rook) & board.color(us) {
        let file = square.file();
        if ours_empty & (1 << file) == 0 {
            continue;
        }
        if theirs_empty & (1 << file) != 0 {
            mg += weights::WEIGHTS.rook_open_mg;
            eg += weights::WEIGHTS.rook_open_eg;
        } else {
            mg += weights::WEIGHTS.rook_semi_mg;
            eg += weights::WEIGHTS.rook_semi_eg;
        }
    }
    (mg, eg)
}

/// Rook file bonuses by scanning the pawns directly.
///
/// Superseded on the hot path by [`rook_files_cached`]; kept as the independent
/// definition that one is checked against.
#[cfg(test)]
fn rook_files(board: &Board, us: Color) -> (i32, i32) {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let ours = board.pieces(PieceType::Pawn) & board.color(us);
    let theirs = board.pieces(PieceType::Pawn) & board.color(us.flip());
    let (mut mg, mut eg) = (0, 0);
    for square in board.pieces(PieceType::Rook) & board.color(us) {
        let file = Bitboard::new(FILE_A << square.file());
        if !(ours & file).is_empty() {
            continue;
        }
        if (theirs & file).is_empty() {
            mg += weights::WEIGHTS.rook_open_mg;
            eg += weights::WEIGHTS.rook_open_eg;
        } else {
            mg += weights::WEIGHTS.rook_semi_mg;
            eg += weights::WEIGHTS.rook_semi_eg;
        }
    }
    (mg, eg)
}

/// Cached pawn-derived evaluation for one position's pawn structure.
///
/// Both halves are a function of pawn placement alone, which is what lets one
/// key cover them: `pawn_structure` reads only pawns, and the part of
/// `rook_files` that costs anything is deciding which files are open or
/// semi-open, which is also only about pawns. Where the rooks actually stand is
/// then a popcount against these masks.
#[derive(Clone, Copy, Default)]
struct PawnEntry {
    key: u64,
    /// Midgame and endgame pawn structure, indexed by [`Color`].
    structure: [(i32, i32); 2],
    /// Files with no pawn of that colour, indexed by [`Color`], as a mask of
    /// the eight file bits.
    no_pawn: [u8; 2],
}

/// Direct-mapped pawn cache.
///
/// Thread-local rather than shared: it is written on almost every miss, so a
/// shared table would be a contended cache line on the hottest path in the
/// evaluation. Losing it between searches costs nothing, since it refills
/// within a few thousand nodes.
///
/// 4096 entries is far more than the distinct pawn structures a search visits -
/// pawn moves are a small fraction of a tree - so collisions are rare and cost
/// only a recompute.
const PAWN_CACHE_SLOTS: usize = 4096;

thread_local! {
    static PAWN_CACHE: std::cell::RefCell<Vec<PawnEntry>> =
        std::cell::RefCell::new(vec![PawnEntry::default(); PAWN_CACHE_SLOTS]);
}

/// Pawn structure and open-file masks for both sides, from the cache when the
/// pawn key matches and recomputed into it when it does not.
///
/// A zero key is treated as a miss rather than as "no pawns": an empty slot is
/// also zero, and a pawnless position recomputes to nothing in no time at all.
fn pawn_entry(board: &Board) -> PawnEntry {
    let key = board.state().pawn_key();
    let slot = (key as usize) & (PAWN_CACHE_SLOTS - 1);
    PAWN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if key != 0 && cache[slot].key == key {
            return cache[slot];
        }
        let entry = PawnEntry {
            key,
            structure: [
                pawn_structure(board, Color::White),
                pawn_structure(board, Color::Black),
            ],
            no_pawn: [
                empty_files(board, Color::White),
                empty_files(board, Color::Black),
            ],
        };
        cache[slot] = entry;
        entry
    })
}

/// The eight file bits, set where `us` has no pawn.
fn empty_files(board: &Board, us: Color) -> u8 {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let ours = board.pieces(PieceType::Pawn) & board.color(us);
    let mut mask = 0;
    for file in 0..8 {
        if (ours & Bitboard::new(FILE_A << file)).is_empty() {
            mask |= 1 << file;
        }
    }
    mask
}

/// Every scored term other than the piece-square tables, for one side, in
/// midgame and endgame centipawns.
///
/// One place for `evaluate` to call, so adding a term is one line here rather
/// than another pair of bindings threaded through the blend.
fn term_sum(board: &Board, us: Color, pawns: &PawnEntry) -> (i32, i32) {
    let (mobility_mg, mobility_eg) = mobility(board, us);
    let (rook_mg, rook_eg) = rook_files_cached(board, us, pawns);
    let (pawn_mg, pawn_eg) = pawns.structure[us as usize];
    (
        mobility_mg + rook_mg + pawn_mg,
        mobility_eg + rook_eg + pawn_eg,
    )
}

/// A file, spread to the two files either side of it as well.
fn adjacent_files(file: u8) -> u64 {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let f = FILE_A << file;
    // The shifts cannot wrap onto the opposite edge because a file mask is
    // eight bits spaced eight apart, so a one-bit shift moves each into the
    // neighbouring file or off the board.
    ((f << 1) & !FILE_A) | ((f >> 1) & !(FILE_A << 7))
}

/// Passed pawn bonus by the number of ranks the pawn has advanced, from its
/// home rank (0, never scored) to the rank before promotion (6).
///
/// Steep and endgame-weighted: a passer on the sixth is close to a queen when
/// nothing is left to stop it, and close to irrelevant with a full board.
const PASSED_MG: [i32; 7] = [0, 2, 6, 14, 28, 50, 80];
const PASSED_EG: [i32; 7] = [0, 8, 16, 30, 55, 95, 145];

/// Penalty for a pawn with no friendly pawn on either adjacent file. It can
/// never be defended by a pawn, so it is a permanent target.
const ISOLATED_MG: i32 = -12;
const ISOLATED_EG: i32 = -18;

/// Penalty per pawn beyond the first on a file. They cannot defend each other
/// and together they cover fewer squares than they would spread out.
const DOUBLED_MG: i32 = -8;
const DOUBLED_EG: i32 = -18;

/// Pawn structure for one side, in midgame and endgame centipawns.
fn pawn_structure(board: &Board, us: Color) -> (i32, i32) {
    let ours = board.pieces(PieceType::Pawn) & board.color(us);
    let theirs = board.pieces(PieceType::Pawn) & board.color(us.flip());
    let (mut mg, mut eg) = (0, 0);
    for square in ours {
        let file = square.file();
        let rank = square.rank();
        // Ranks advanced, counted from the side's own home rank so that both
        // colors index the same table.
        let advanced = match us {
            Color::White => rank,
            Color::Black => 7 - rank,
        } as usize;
        let own_file = Bitboard::new(0x0101_0101_0101_0101 << file);
        let neighbours = Bitboard::new(adjacent_files(file));

        // Passed: no enemy pawn on this or an adjacent file ahead of it, which
        // is exactly the set that could either block it or capture it on the
        // way. `ahead` excludes the pawn's own square, so an enemy pawn beside
        // it on a neighbouring file does not count as stopping it.
        //
        // Our own pawn ahead on the file disqualifies it too: the rear pawn of
        // a doubled pair cannot run, and scoring both as passed would pay
        // twice for one passer.
        let ahead = ahead_of(square, us);
        let blocked = !(ours & ahead & own_file).is_empty();
        if !blocked && (theirs & ahead & (own_file | neighbours)).is_empty() {
            mg += weights::WEIGHTS.passed_mg[advanced];
            eg += weights::WEIGHTS.passed_eg[advanced];
        }
        // Isolated: no friendly pawn on either neighbouring file, anywhere.
        if (ours & neighbours).is_empty() {
            mg += weights::WEIGHTS.isolated_mg;
            eg += weights::WEIGHTS.isolated_eg;
        }
        // Doubled: charged once per pawn that has a friendly pawn ahead of it
        // on the same file, so a tripled file is charged twice.
        if blocked {
            mg += weights::WEIGHTS.doubled_mg;
            eg += weights::WEIGHTS.doubled_eg;
        }
    }
    (mg, eg)
}

/// Every square strictly ahead of a square from `color`'s point of view.
fn ahead_of(square: Square, color: Color) -> Bitboard {
    let rank = square.rank();
    Bitboard::new(match color {
        // All ranks above this one: shift the full board up past it.
        Color::White => u64::MAX << ((rank + 1) * 8),
        // All ranks below. `rank == 0` would shift by 64, which is undefined,
        // but a black pawn on rank 0 has already promoted and cannot exist.
        Color::Black => {
            if rank == 0 {
                0
            } else {
                u64::MAX >> ((8 - rank) * 8)
            }
        }
    })
}

/// Returns the static evaluation in centipawns, relative to the side to move.
///
/// Material and piece placement are scored from separate midgame and endgame
/// tables, interpolated by how much material remains, plus rook file control
/// and `TEMPO` for the side to move.
pub fn evaluate(board: &Board) -> i32 {
    let Accumulator { mg, eg, phase } = *board.accumulator();
    let pawns = pawn_entry(board);
    let (white_mg, white_eg) = term_sum(board, Color::White, &pawns);
    let (black_mg, black_eg) = term_sum(board, Color::Black, &pawns);
    let score = blend(mg + white_mg - black_mg, eg + white_eg - black_eg, phase);
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
    score + weights::WEIGHTS.tempo
}

/// Active side-to-move bonus, shared with null-move search correction.
pub(crate) const fn tempo() -> i32 {
    weights::WEIGHTS.tempo
}

/// Baseline weights in the tuner's stable parameter order.
pub(crate) fn tuning_parameters() -> [f64; crate::tuner::features::PARAMETER_COUNT] {
    crate::tuner::features::parameters_from_weights(&weights::WEIGHTS)
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
    ///
    /// Computed from the accumulator rather than by subtracting the other
    /// terms off `evaluate`: the blend truncates once, so peeling a separately
    /// blended term back off leaves a rounding difference of a centipawn.
    fn placement(board: &Board) -> i32 {
        let Accumulator { mg, eg, phase } = *board.accumulator();
        let score = blend(mg, eg, phase);
        if board.state().side_to_move() == Color::White {
            score
        } else {
            -score
        }
    }

    // The sparse tuner definition is independent of the incremental runtime
    // accumulator. Keeping them aligned is what makes generated weights safe
    // to install without silently training a different evaluation.
    #[test]
    fn sparse_definition_matches_incremental_evaluation() {
        crate::movegen::init();
        let parameters = tuning_parameters();
        for fen in [
            "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/4N3/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/2B5/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
            "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
            "3qk3/8/8/8/8/8/8/3Q2K1 w - - 0 1",
            "4k3/8/8/8/8/8/8/QQQQKQQQ w - - 0 1",
        ] {
            let board: Board = fen.parse().unwrap();
            let sparse = crate::tuner::features::score(
                &parameters,
                &crate::tuner::features::extract(&board),
            );
            assert!((sparse - f64::from(evaluate(&board))).abs() < 1.0, "{fen}");
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
        assert_eq!(evaluate(&Board::startpos()), tempo());
        let white: Board = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        let black: Board = "4k3/8/8/8/8/8/8/4K3 b - - 0 1".parse().unwrap();
        assert_eq!(evaluate(&white), tempo());
        assert_eq!(evaluate(&black), tempo());
        // The material and placement component really is zero here, so the
        // whole score above is the bonus and not a cancellation that happens
        // to land on the same number.
        assert_eq!(evaluate(&white) - tempo(), 0);
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
        assert_eq!(evaluate(&white) + evaluate(&black), 2 * tempo());
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

    // catches: counting our own pieces' squares as available, ignoring the
    // enemy pawn screen, scoring pawns or kings, and the whole term silently
    // reading zero. Each case isolates one of those.
    #[test]
    fn mobility_counts_only_squares_a_piece_can_actually_use() {
        let count = |fen: &str, color| {
            let board: Board = fen.parse().unwrap();
            mobility(&board, color)
        };
        // A lone knight on e4 reaches 8 squares.
        assert_eq!(
            count("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1", Color::White).0,
            8 * weights::WEIGHTS.mobility_mg[PieceType::Knight as usize]
        );
        // Own pawns on two of those squares remove them.
        assert_eq!(
            count("4k3/8/3P1P2/8/4N3/8/8/4K3 w - - 0 1", Color::White).0,
            6 * weights::WEIGHTS.mobility_mg[PieceType::Knight as usize]
        );
        // Enemy pawns on d6/f6 are targets, so those two squares still count -
        // this is what distinguishes "not occupied by us" from "empty". But the
        // same pawns attack c5, e5 and g5, and the knight reaches c5 and g5, so
        // two other squares go away: 8 targets, minus 2 screened.
        assert_eq!(
            count("4k3/8/3p1p2/8/4N3/8/8/4K3 w - - 0 1", Color::White).0,
            6 * weights::WEIGHTS.mobility_mg[PieceType::Knight as usize]
        );
        // A black pawn on d7 attacks c6 and e6; a white knight on d4 reaches
        // both c6 and e6, so two of its eight squares are screened off.
        assert_eq!(
            count("4k3/3p4/8/8/3N4/8/8/4K3 w - - 0 1", Color::White).0,
            6 * weights::WEIGHTS.mobility_mg[PieceType::Knight as usize]
        );
        // Pawns and kings contribute nothing, so a pawn-and-king position is
        // flat zero rather than a number that happens to cancel.
        assert_eq!(
            count("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1", Color::White),
            (0, 0)
        );
        // Nonzero somewhere, so a disabled term cannot pass the tests above.
        assert_ne!(weights::WEIGHTS.mobility_mg[PieceType::Knight as usize], 0);
    }

    // catches: the cached rook term drifting from the scanning one, a stale
    // cache entry surviving a pawn move, and the pawn key colliding across
    // structures that score differently.
    //
    // Every position is scored twice, the second time with the cache already
    // warm from the first, so a hit that returns another position's entry shows
    // up as a mismatch rather than as a plausible wrong number.
    #[test]
    fn the_pawn_cache_agrees_with_a_direct_scan() {
        let positions = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/3RK3 w - - 0 1",
            "4k3/3p4/8/8/8/8/3P4/3RK3 w - - 0 1",
            "3rk3/8/8/8/3R4/8/8/3R1K2 w - - 0 1",
            // No pawns at all: the pawn key is zero here, which is also what an
            // empty cache slot holds.
            "4k3/8/8/8/8/8/8/R3K2R w - - 0 1",
        ];
        for fen in positions {
            let board: Board = fen.parse().unwrap();
            for _ in 0..2 {
                let pawns = pawn_entry(&board);
                for us in [Color::White, Color::Black] {
                    assert_eq!(
                        rook_files_cached(&board, us, &pawns),
                        rook_files(&board, us),
                        "cached rook files disagree for {us:?} on {fen}"
                    );
                    assert_eq!(
                        pawns.structure[us as usize],
                        pawn_structure(&board, us),
                        "cached pawn structure disagrees for {us:?} on {fen}"
                    );
                }
            }
        }
    }

    // catches: a pawn key that does not change when the structure does, which
    // would serve the pre-move score for the post-move position. The push is
    // chosen to change the score: e2-e4 leaves the d-file rook's file alone but
    // moves the pawn two ranks up the passed-pawn table.
    #[test]
    fn a_pawn_move_changes_the_pawn_key() {
        crate::movegen::init();
        let mut board: Board = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".parse().unwrap();
        let before_key = board.state().pawn_key();
        let before = pawn_entry(&board).structure[Color::White as usize];

        let mut moves = crate::movegen::MoveList::new();
        crate::movegen::generate_legal(&mut board, &mut moves);
        let push = *moves
            .iter()
            .find(|mv| mv.from().index() == 12 && mv.to().index() == 28)
            .expect("e2-e4 must be legal here");
        board.make(push);

        assert_ne!(
            board.state().pawn_key(),
            before_key,
            "the pawn key survived a pawn move"
        );
        assert_ne!(
            pawn_entry(&board).structure[Color::White as usize],
            before,
            "the cache served the pre-move structure"
        );
    }

    // catches: open and semi-open swapped, an enemy pawn treated as blocking
    // like an own pawn, the bonus paid once per side rather than per rook, and
    // the term reading zero.
    #[test]
    fn rook_files_separates_open_from_semi_open_from_blocked() {
        let score = |fen: &str| {
            let board: Board = fen.parse().unwrap();
            rook_files(&board, Color::White).0
        };
        // No pawns at all on the d-file: open.
        assert_eq!(
            score("4k3/8/8/8/8/8/8/3RK3 w - - 0 1"),
            weights::WEIGHTS.rook_open_mg
        );
        // A black pawn on d7 makes it semi-open, not blocked.
        assert_eq!(
            score("4k3/3p4/8/8/8/8/8/3RK3 w - - 0 1"),
            weights::WEIGHTS.rook_semi_mg
        );
        // Our own pawn on d2 blocks it: no bonus, whatever else is on the file.
        assert_eq!(score("4k3/3p4/8/8/8/8/3P4/3RK3 w - - 0 1"), 0);
        // Two rooks on two open files are paid twice.
        assert_eq!(
            score("4k3/8/8/8/8/8/8/R2RK3 w - - 0 1"),
            2 * weights::WEIGHTS.rook_open_mg
        );
        // Two rooks doubled on one open file are also paid twice - the bonus is
        // per rook, and this is the case a per-file loop would score once.
        assert_eq!(
            score("3rk3/8/8/8/3R4/8/8/3R1K2 w - - 0 1"),
            2 * weights::WEIGHTS.rook_open_mg
        );
        // Only the rook's own file matters: an own pawn on a *different* file
        // leaves the bonus intact.
        assert_eq!(
            score("4k3/8/8/8/8/8/P7/3RK3 w - - 0 1"),
            weights::WEIGHTS.rook_open_mg
        );
    }

    // catches: a passer test that an enemy pawn beside it defeats, adjacent
    // files wrapping around the board edge, doubled charged per file rather
    // than per extra pawn, and the whole term reading zero.
    #[test]
    fn pawn_structure_scores_passers_isolanis_and_doubled_pawns() {
        let score = |fen: &str, color| {
            let board: Board = fen.parse().unwrap();
            pawn_structure(&board, color)
        };
        // A white pawn on e5 with no black pawn ahead of it on d/e/f is passed,
        // and isolated too - three ranks advanced, so PASSED[4] plus ISOLATED.
        assert_eq!(
            score("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1", Color::White),
            (
                weights::WEIGHTS.passed_mg[4] + weights::WEIGHTS.isolated_mg,
                weights::WEIGHTS.passed_eg[4] + weights::WEIGHTS.isolated_eg
            )
        );
        // A black pawn on d7 is ahead of it on an adjacent file, so not passed.
        assert_eq!(
            score("4k3/3p4/8/4P3/8/8/8/4K3 w - - 0 1", Color::White).0,
            weights::WEIGHTS.isolated_mg
        );
        // A black pawn on d5 is beside it, not ahead, so it is still passed.
        assert_eq!(
            score("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1", Color::White).0,
            weights::WEIGHTS.passed_mg[4] + weights::WEIGHTS.isolated_mg
        );
        // Doubled is per extra pawn, not per file: e4+e5 is charged once,
        // e3+e4+e5 twice. Both are isolated, and only the *front* pawn of each
        // stack is passed - the one behind is blocked by its own pawn, and
        // paying the passer bonus twice for one runner is the bug this pins.
        let two = score("4k3/8/8/4P3/4P3/8/8/4K3 w - - 0 1", Color::White).0;
        let three = score("4k3/8/8/4P3/4P3/4P3/8/4K3 w - - 0 1", Color::White).0;
        assert_eq!(
            two,
            weights::WEIGHTS.passed_mg[4]
                + 2 * weights::WEIGHTS.isolated_mg
                + weights::WEIGHTS.doubled_mg
        );
        assert_eq!(
            three,
            weights::WEIGHTS.passed_mg[4]
                + 3 * weights::WEIGHTS.isolated_mg
                + 2 * weights::WEIGHTS.doubled_mg
        );
        // An a-file pawn's neighbours are the b-file only. If the mask wrapped
        // to the h-file, the h-pawn here would stop it being isolated.
        assert_eq!(
            score("4k3/7p/8/8/8/8/P7/4K3 w - - 0 1", Color::White).0,
            weights::WEIGHTS.isolated_mg + weights::WEIGHTS.passed_mg[1]
        );
        // ...and symmetrically for the h-file, which is where a shift that
        // wraps the other way would show.
        assert_eq!(
            score("4k3/p7/8/8/8/8/7P/4K3 w - - 0 1", Color::White).0,
            weights::WEIGHTS.isolated_mg + weights::WEIGHTS.passed_mg[1]
        );
        // A pawn with a friend beside it is neither isolated nor doubled, and
        // with nothing ahead it is passed: this is the case where every
        // penalty must be absent rather than cancelling.
        assert_eq!(
            score("4k3/8/8/8/8/8/PP6/4K3 w - - 0 1", Color::White).0,
            2 * weights::WEIGHTS.passed_mg[1]
        );
        // Black is scored from its own side: a black pawn on d2 is one rank
        // from promoting, so it indexes the same slot as a white pawn on d7.
        assert_eq!(
            score("4k3/8/8/8/8/8/3p4/4K3 w - - 0 1", Color::Black).0,
            weights::WEIGHTS.passed_mg[6] + weights::WEIGHTS.isolated_mg
        );
        assert_ne!(weights::WEIGHTS.passed_eg[6], 0);
    }

    #[test]
    fn tables_are_distinct() {
        // The realistic transcription bug in 768 numbers is pasting one block
        // twice; nothing else catches that.
        let tables: Vec<&[i32; 64]> = weights::WEIGHTS
            .mg_table
            .iter()
            .chain(weights::WEIGHTS.eg_table.iter())
            .collect();
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
