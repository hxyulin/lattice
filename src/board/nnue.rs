//! NNUE evaluation: a `(768 -> HIDDEN) x2 -> 1` perspective network trained by
//! `bullet` and embedded as a quantised network at compile time.
//!
//! This lives beside [`pesto`](super::pesto) in the board layer for the same
//! reason: the network is a pure function of the piece placement the board
//! owns, and `engine` cannot be a dependency of `board`. The engine's
//! `evaluate` calls into [`Board::nnue_eval`] under the `nnue` feature and reads
//! only the final centipawn score.
//!
//! Feature indexing mirrors bullet's `Chess768` exactly (verified against
//! `bullet/crates/bullet_lib/src/game/inputs/chess768.rs`): for a piece of
//! color `c` (White = 0), type `pt` (0..6), on square `sq` (LERF, a1 = 0),
//!
//! - white-perspective feature = `c*384 + pt*64 + sq`
//! - black-perspective feature = `(1^c)*384 + pt*64 + (sq^56)`
//!
//! Both perspectives are keyed on absolute color so a piece's contribution is
//! invariant to whose turn it is; at eval time the side to move's accumulator is
//! `us` and the other is `them`, matching bullet's `(stm, ntm)` ordering.
//!
//! The two accumulators are maintained incrementally by the board's
//! `put_piece`/`remove_piece` hooks (like the tapered-eval accumulator) and
//! seeded from the mailbox on setup; this module owns the network, the feature
//! math, and the forward pass. A from-scratch reseed is the oracle the
//! incremental path is tested against (`nnue_acc_incremental_matches_recompute`).

use std::sync::LazyLock;

use super::Piece;

/// Hidden-layer width per perspective.
pub(super) const HIDDEN: usize = 256;
/// Feature-transformer quantisation constant (bullet `QA`).
const QA: i32 = 255;
/// Output-layer quantisation constant (bullet `QB`).
const QB: i32 = 64;
/// Eval scale mapping network output to centipawns (bullet `eval_scale`).
const SCALE: i32 = 400;
/// Number of input features (64 squares * 6 piece types * 2 colors).
const FEATURES: usize = 768;

/// The quantised network, laid out as bullet writes `quantised.bin`:
/// little-endian `i16`, affine weights column-major (so each input feature owns
/// a contiguous `HIDDEN`-wide block of `l0w`), in save-format order
/// `l0w, l0b, l1w, l1b`.
struct Network {
    /// Feature transformer weights, `l0w[feature * HIDDEN + h]`.
    l0w: Box<[i16; FEATURES * HIDDEN]>,
    /// Feature transformer biases.
    l0b: [i16; HIDDEN],
    /// Output weights: `[0, HIDDEN)` apply to `us`, `[HIDDEN, 2*HIDDEN)` to `them`.
    l1w: [i16; 2 * HIDDEN],
    /// Output bias (quantised at `QA * QB`).
    l1b: i16,
}

impl Network {
    /// Parse the embedded `quantised.bin`. Reads the exact prefix the layout
    /// requires and ignores bullet's trailing 64-byte padding.
    fn load(bytes: &[u8]) -> Box<Self> {
        let need = (FEATURES * HIDDEN + HIDDEN + 2 * HIDDEN + 1) * 2;
        assert!(
            bytes.len() >= need,
            "embedded net too small: {} < {need} bytes",
            bytes.len()
        );
        let mut off = 0usize;
        let mut next = || {
            let v = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
            off += 2;
            v
        };
        let mut l0w = Box::new([0i16; FEATURES * HIDDEN]);
        for w in l0w.iter_mut() {
            *w = next();
        }
        let mut l0b = [0i16; HIDDEN];
        for b in &mut l0b {
            *b = next();
        }
        let mut l1w = [0i16; 2 * HIDDEN];
        for w in &mut l1w {
            *w = next();
        }
        let l1b = next();
        Box::new(Network { l0w, l0b, l1w, l1b })
    }
}

/// The embedded network, parsed once on first use.
static NET: LazyLock<Box<Network>> =
    LazyLock::new(|| Network::load(include_bytes!("lattice-768x256-v1-20260701.nnue")));

/// Squared clipped ReLU, bullet's `screlu`: clamp to `[0, QA]` then square.
/// Returns `i64` because the summed products overflow `i32` in the worst case.
#[inline]
fn screlu(x: i16) -> i64 {
    let c = i64::from(x).clamp(0, QA as i64);
    c * c
}

/// The feature-transformer bias: the accumulator value for an empty board, the
/// base a seed or reseed starts from.
pub(super) fn bias() -> [i16; HIDDEN] {
    NET.l0b
}

/// Base offsets into `l0w` for `piece` on `sq` (LERF), for the white- and
/// black-perspective accumulators respectively. Both perspectives are keyed on
/// absolute color, so this is invariant to whose turn it is.
#[inline]
fn offsets(piece: Piece, sq: usize) -> (usize, usize) {
    let c = piece.color().as_u8() as usize;
    let pt = (piece.as_u8() >> 1) as usize;
    let fw = (c * 384 + pt * 64 + sq) * HIDDEN;
    let fb = ((1 ^ c) * 384 + pt * 64 + (sq ^ 56)) * HIDDEN;
    (fw, fb)
}

/// Add `piece` on `sq` (LERF) into both perspective accumulators.
#[inline]
pub(super) fn add(white: &mut [i16; HIDDEN], black: &mut [i16; HIDDEN], piece: Piece, sq: usize) {
    let (fw, fb) = offsets(piece, sq);
    let net = &**NET;
    for h in 0..HIDDEN {
        white[h] = white[h].wrapping_add(net.l0w[fw + h]);
        black[h] = black[h].wrapping_add(net.l0w[fb + h]);
    }
}

/// Remove `piece` on `sq` (LERF) from both perspective accumulators.
#[inline]
pub(super) fn sub(white: &mut [i16; HIDDEN], black: &mut [i16; HIDDEN], piece: Piece, sq: usize) {
    let (fw, fb) = offsets(piece, sq);
    let net = &**NET;
    for h in 0..HIDDEN {
        white[h] = white[h].wrapping_sub(net.l0w[fw + h]);
        black[h] = black[h].wrapping_sub(net.l0w[fb + h]);
    }
}

/// Forward pass from the maintained accumulators, in centipawns from the side to
/// move's perspective. `white_to_move` selects which accumulator is `us`.
pub(super) fn forward(white: &[i16; HIDDEN], black: &[i16; HIDDEN], white_to_move: bool) -> i32 {
    let net = &**NET;
    let (us, them) = if white_to_move {
        (white, black)
    } else {
        (black, white)
    };
    let mut out: i64 = 0;
    for h in 0..HIDDEN {
        out += screlu(us[h]) * i64::from(net.l1w[h]);
        out += screlu(them[h]) * i64::from(net.l1w[HIDDEN + h]);
    }
    out = out / i64::from(QA) + i64::from(net.l1b);
    (out * i64::from(SCALE) / i64::from(QA * QB)) as i32
}
