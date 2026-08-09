//! Sparse linear features corresponding exactly to Lattice's static evaluator.

use crate::eval::EvalWeights;
use crate::movegen::{bishop_attacks, knight_attacks, queen_attacks, rook_attacks};
use crate::{Bitboard, Board, Color, PieceType};

const MG_PSQT: usize = 0;
const EG_PSQT: usize = 384;
const MOBILITY_MG: usize = 768;
const MOBILITY_EG: usize = 772;
const ROOK_OPEN_MG: usize = 776;
const ROOK_OPEN_EG: usize = 777;
const ROOK_SEMI_MG: usize = 778;
const ROOK_SEMI_EG: usize = 779;
const PASSED_MG: usize = 780;
const PASSED_EG: usize = 787;
const ISOLATED_MG: usize = 794;
const ISOLATED_EG: usize = 795;
const DOUBLED_MG: usize = 796;
const DOUBLED_EG: usize = 797;
const TEMPO: usize = 798;

/// Number of independently fitted evaluation weights.
pub(crate) const PARAMETER_COUNT: usize = 799;

/// One nonzero coefficient in a position's linear evaluation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FeatureTerm {
    pub(crate) index: u16,
    pub(crate) coefficient: f32,
}

/// Flattens engine weights into the stable optimizer order.
pub(crate) fn parameters_from_weights(weights: &EvalWeights) -> [f64; PARAMETER_COUNT] {
    let mut out = [0.0; PARAMETER_COUNT];
    for kind in 0..6 {
        for square in 0..64 {
            out[MG_PSQT + kind * 64 + square] = f64::from(weights.mg_table[kind][square]);
            out[EG_PSQT + kind * 64 + square] = f64::from(weights.eg_table[kind][square]);
        }
    }
    for (slot, kind) in [1usize, 2, 3, 4].into_iter().enumerate() {
        out[MOBILITY_MG + slot] = f64::from(weights.mobility_mg[kind]);
        out[MOBILITY_EG + slot] = f64::from(weights.mobility_eg[kind]);
    }
    out[ROOK_OPEN_MG] = f64::from(weights.rook_open_mg);
    out[ROOK_OPEN_EG] = f64::from(weights.rook_open_eg);
    out[ROOK_SEMI_MG] = f64::from(weights.rook_semi_mg);
    out[ROOK_SEMI_EG] = f64::from(weights.rook_semi_eg);
    for rank in 0..7 {
        out[PASSED_MG + rank] = f64::from(weights.passed_mg[rank]);
        out[PASSED_EG + rank] = f64::from(weights.passed_eg[rank]);
    }
    out[ISOLATED_MG] = f64::from(weights.isolated_mg);
    out[ISOLATED_EG] = f64::from(weights.isolated_eg);
    out[DOUBLED_MG] = f64::from(weights.doubled_mg);
    out[DOUBLED_EG] = f64::from(weights.doubled_eg);
    out[TEMPO] = f64::from(weights.tempo);
    out
}

/// Extracts the sparse coefficients whose dot product is static evaluation.
pub(crate) fn extract(board: &Board) -> Vec<FeatureTerm> {
    const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
    const TOTAL_PHASE: f32 = 24.0;

    let mut raw_phase = 0i32;
    for (kind, weight) in PHASE_WEIGHT.into_iter().enumerate() {
        raw_phase += weight * board.pieces(PieceType::from_index(kind as u8)).count() as i32;
    }
    let mg_phase = raw_phase.min(24) as f32 / TOTAL_PHASE;
    let eg_phase = 1.0 - mg_phase;
    let stm_sign = if board.state().side_to_move() == Color::White {
        1.0
    } else {
        -1.0
    };
    let mut terms = Vec::with_capacity(board.occupied().count() as usize * 2 + 32);

    for square in board.occupied() {
        let piece = board.piece_on(square).expect("occupied square has a piece");
        let (table_square, color_sign) = match piece.color() {
            Color::White => (square.flip_rank().index() as usize, 1.0),
            Color::Black => (square.index() as usize, -1.0),
        };
        let kind = piece.piece_type() as usize;
        push(
            &mut terms,
            MG_PSQT + kind * 64 + table_square,
            stm_sign * color_sign * mg_phase,
        );
        push(
            &mut terms,
            EG_PSQT + kind * 64 + table_square,
            stm_sign * color_sign * eg_phase,
        );
    }

    for (slot, kind) in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ]
    .into_iter()
    .enumerate()
    {
        let net =
            mobility_count(board, Color::White, kind) - mobility_count(board, Color::Black, kind);
        let net = stm_sign * net as f32;
        push(&mut terms, MOBILITY_MG + slot, net * mg_phase);
        push(&mut terms, MOBILITY_EG + slot, net * eg_phase);
    }

    let (white_open, white_semi) = rook_file_counts(board, Color::White);
    let (black_open, black_semi) = rook_file_counts(board, Color::Black);
    let open = stm_sign * (white_open - black_open) as f32;
    let semi = stm_sign * (white_semi - black_semi) as f32;
    push(&mut terms, ROOK_OPEN_MG, open * mg_phase);
    push(&mut terms, ROOK_OPEN_EG, open * eg_phase);
    push(&mut terms, ROOK_SEMI_MG, semi * mg_phase);
    push(&mut terms, ROOK_SEMI_EG, semi * eg_phase);

    let white = pawn_counts(board, Color::White);
    let black = pawn_counts(board, Color::Black);
    for rank in 0..7 {
        let net = stm_sign * (white.passed[rank] - black.passed[rank]) as f32;
        push(&mut terms, PASSED_MG + rank, net * mg_phase);
        push(&mut terms, PASSED_EG + rank, net * eg_phase);
    }
    let isolated = stm_sign * (white.isolated - black.isolated) as f32;
    let doubled = stm_sign * (white.doubled - black.doubled) as f32;
    push(&mut terms, ISOLATED_MG, isolated * mg_phase);
    push(&mut terms, ISOLATED_EG, isolated * eg_phase);
    push(&mut terms, DOUBLED_MG, doubled * mg_phase);
    push(&mut terms, DOUBLED_EG, doubled * eg_phase);
    push(&mut terms, TEMPO, 1.0);

    terms.sort_unstable_by_key(|term| term.index);
    let mut combined: Vec<FeatureTerm> = Vec::with_capacity(terms.len());
    for term in terms {
        if let Some(last) = combined.last_mut().filter(|last| last.index == term.index) {
            last.coefficient += term.coefficient;
        } else {
            combined.push(term);
        }
    }
    combined.retain(|term| term.coefficient != 0.0);
    combined
}

pub(crate) fn score(parameters: &[f64], terms: &[FeatureTerm]) -> f64 {
    terms
        .iter()
        .map(|term| parameters[term.index as usize] * f64::from(term.coefficient))
        .sum()
}

fn push(terms: &mut Vec<FeatureTerm>, index: usize, coefficient: f32) {
    if coefficient != 0.0 {
        terms.push(FeatureTerm {
            index: index as u16,
            coefficient,
        });
    }
}

fn mobility_count(board: &Board, us: Color, kind: PieceType) -> i32 {
    let occ = board.occupied();
    let ours = board.color(us);
    let their_pawns = board.pieces(PieceType::Pawn) & board.color(us.flip());
    let available = !(ours | pawn_attack_span(their_pawns, us.flip()));
    (board.pieces(kind) & ours)
        .into_iter()
        .map(|square| {
            let attacks = match kind {
                PieceType::Knight => knight_attacks(square),
                PieceType::Bishop => bishop_attacks(square, occ),
                PieceType::Rook => rook_attacks(square, occ),
                PieceType::Queen => queen_attacks(square, occ),
                _ => unreachable!("mobility is only defined for sliding pieces and knights"),
            };
            (attacks & available).count() as i32
        })
        .sum()
}

fn pawn_attack_span(pawns: Bitboard, color: Color) -> Bitboard {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    const FILE_H: u64 = FILE_A << 7;
    let bits = pawns.bits();
    Bitboard::new(match color {
        Color::White => ((bits & !FILE_A) << 7) | ((bits & !FILE_H) << 9),
        Color::Black => ((bits & !FILE_A) >> 9) | ((bits & !FILE_H) >> 7),
    })
}

fn rook_file_counts(board: &Board, us: Color) -> (i32, i32) {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let ours = board.pieces(PieceType::Pawn) & board.color(us);
    let theirs = board.pieces(PieceType::Pawn) & board.color(us.flip());
    let mut open = 0;
    let mut semi = 0;
    for square in board.pieces(PieceType::Rook) & board.color(us) {
        let file = Bitboard::new(FILE_A << square.file());
        if !(ours & file).is_empty() {
            continue;
        }
        if (theirs & file).is_empty() {
            open += 1;
        } else {
            semi += 1;
        }
    }
    (open, semi)
}

#[derive(Default)]
struct PawnCounts {
    passed: [i32; 7],
    isolated: i32,
    doubled: i32,
}

fn pawn_counts(board: &Board, us: Color) -> PawnCounts {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let ours = board.pieces(PieceType::Pawn) & board.color(us);
    let theirs = board.pieces(PieceType::Pawn) & board.color(us.flip());
    let mut counts = PawnCounts::default();
    for square in ours {
        let file = square.file();
        let rank = square.rank();
        let advanced = match us {
            Color::White => rank,
            Color::Black => 7 - rank,
        } as usize;
        let own_file = Bitboard::new(FILE_A << file);
        let neighbours = Bitboard::new(adjacent_files(file));
        let ahead = ahead_of(square.index(), us);
        let blocked = !(ours & ahead & own_file).is_empty();
        if advanced < counts.passed.len()
            && !blocked
            && (theirs & ahead & (own_file | neighbours)).is_empty()
        {
            counts.passed[advanced] += 1;
        }
        if (ours & neighbours).is_empty() {
            counts.isolated += 1;
        }
        if blocked {
            counts.doubled += 1;
        }
    }
    counts
}

fn adjacent_files(file: u8) -> u64 {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let f = FILE_A << file;
    ((f << 1) & !FILE_A) | ((f >> 1) & !(FILE_A << 7))
}

fn ahead_of(square: u8, color: Color) -> Bitboard {
    let rank = square / 8;
    Bitboard::new(match color {
        Color::White => u64::MAX << ((rank + 1) * 8),
        Color::Black if rank == 0 => 0,
        Color::Black => u64::MAX >> ((8 - rank) * 8),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_features_reproduce_runtime_evaluation() {
        crate::movegen::init();
        let parameters = crate::eval::tuning_parameters();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ] {
            let board: Board = fen.parse().unwrap();
            let sparse = score(&parameters, &extract(&board));
            let runtime = f64::from(crate::eval::evaluate(&board));
            assert!(
                (sparse - runtime).abs() < 1.0,
                "{fen}: sparse {sparse}, runtime {runtime}"
            );
        }
    }
}
