//! The chess engine "brain": search and evaluation over [`lattice_board`].
//!
//! Modules:
//! - [`eval`] - evaluation
//! - [`search`] - negamax search with legality filter

mod eval;
mod search;

pub use eval::evaluate;
pub use search::{SearchResult, search};

/// A position score in centipawns,
pub type Score = i32;

/// Score of being checkmated *at this node*. A mate `n` plies away is stored as
/// `MATE - n`, so a shorter mate always outranks a longer one.
pub const MATE: Score = 30_000;
