//! The chess engine "brain": search and evaluation over the `board` module.
//!
//! Modules:
//! - [`eval`] - evaluation
//! - [`search`] - negamax search with legality filter

mod bench;
mod eval;
mod search;
mod tt;

pub use bench::{BenchEntry, BenchReport, bench, nps};
pub use eval::evaluate;
pub use search::{Limits, MAX_PLY, SearchResult, TUNABLES, TunableSpec, Tunables, budget, search};
pub use tt::{Bound, Entry, TranspositionTable};

/// A position score in centipawns,
pub type Score = i32;

/// Score of being checkmated *at this node*. A mate `n` plies away is stored as
/// `MATE - n`, so a shorter mate always outranks a longer one.
pub const MATE: Score = 30_000;
