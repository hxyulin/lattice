//! Texel tuning: fitting the evaluation weights to game outcomes.
//!
//! The evaluation is linear in its weights - every term is a weight times a
//! count, and `blend` is a weighted sum of the midgame and endgame halves - so
//! a position reduces to a sparse vector of coefficients that does not depend
//! on the weights at all. Extracting that vector once per position turns each
//! training step into a sparse dot product instead of a full re-evaluation.

pub mod data;
pub(crate) mod dataset;
pub(crate) mod features;
pub(crate) mod optimize;
pub(crate) mod output;

pub use optimize::{TuneConfig, TuneError, TuneSummary, run};
