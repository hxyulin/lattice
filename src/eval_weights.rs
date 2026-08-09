//! Compile-time evaluation weights used by the engine.
//!
//! The baseline is assembled from the existing material and PeSTO tables so
//! this refactor is score-identical. The tuner emits a replacement module with
//! direct folded tables.

use super::{
    DOUBLED_EG, DOUBLED_MG, EG_TABLE, EvalWeights, ISOLATED_EG, ISOLATED_MG, MG_TABLE, MOBILITY_EG,
    MOBILITY_MG, PASSED_EG, PASSED_MG, ROOK_OPEN_EG, ROOK_OPEN_MG, ROOK_SEMI_EG, ROOK_SEMI_MG,
    TEMPO,
};

pub(crate) const WEIGHTS: EvalWeights = EvalWeights {
    mg_table: MG_TABLE,
    eg_table: EG_TABLE,
    mobility_mg: MOBILITY_MG,
    mobility_eg: MOBILITY_EG,
    rook_open_mg: ROOK_OPEN_MG,
    rook_open_eg: ROOK_OPEN_EG,
    rook_semi_mg: ROOK_SEMI_MG,
    rook_semi_eg: ROOK_SEMI_EG,
    passed_mg: PASSED_MG,
    passed_eg: PASSED_EG,
    isolated_mg: ISOLATED_MG,
    isolated_eg: ISOLATED_EG,
    doubled_mg: DOUBLED_MG,
    doubled_eg: DOUBLED_EG,
    tempo: TEMPO,
};
