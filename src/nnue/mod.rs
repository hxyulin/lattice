//! HalfKP feature extraction, incremental accumulation, and quantised inference.

mod accumulator;
mod features;
mod network;

pub(crate) use accumulator::Accumulator;
pub use features::{FEATURES, feature_index};
pub use network::{HIDDEN, Network, NetworkError};

use std::sync::OnceLock;

static NETWORK: OnceLock<Network> = OnceLock::new();

pub(crate) fn network() -> &'static Network {
    NETWORK.get_or_init(|| {
        Network::parse(include_bytes!(env!("LATTICE_NNUE_FILE")))
            .expect("the embedded NNUE must have a valid header and payload")
    })
}

/// Returns the embedded network's score relative to the side to move.
pub fn evaluate(board: &crate::Board) -> i32 {
    board
        .nnue_accumulator()
        .evaluate(board.state().side_to_move(), network())
}
