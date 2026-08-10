//! Shared HalfKP input and network-file code for the Lattice trainer tools.

use bullet_lib::game::inputs::SparseInputType;
use bulletformat::ChessBoard;

/// Vanilla HalfKP feature count.
pub const FEATURES: usize = 40_960;
/// Width of one perspective accumulator.
pub const HIDDEN: usize = 256;
/// Feature-transformer quantisation.
pub const QA: i16 = 255;
/// Output-layer quantisation.
pub const QB: i16 = 64;
/// Centipawn scale used by training and inference.
pub const SCALE: i32 = 400;
/// Bytes emitted by Bullet before its optional 64-byte padding.
pub const RAW_NETWORK_BYTES: usize = 2 * (HIDDEN + FEATURES * HIDDEN + 1 + 2 * HIDDEN);
const NETWORK_VALUES: usize = HIDDEN + FEATURES * HIDDEN + 1 + 2 * HIDDEN;

/// Feature ABI written into a packed Lattice network header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkArchitecture {
    /// King-anchored 40,960-input HalfKP.
    HalfKp,
    /// Absolute 768-input piece-square features.
    Chess768,
}

impl NetworkArchitecture {
    fn header(self) -> (u32, usize) {
        match self {
            Self::HalfKp => (1, FEATURES),
            Self::Chess768 => (2, 768),
        }
    }

    fn raw_bytes(self) -> usize {
        let (_, features) = self.header();
        2 * (HIDDEN + features * HIDDEN + 1 + 2 * HIDDEN)
    }
}

/// HalfKP inputs in the mover-relative representation stored by bulletformat.
#[derive(Clone, Copy, Debug, Default)]
pub struct HalfKp;

/// The shared feature-index formula used by the trainer's mapper.
pub fn feature_index(king: u8, relative_colour: usize, piece_kind: usize, square: u8) -> usize {
    usize::from(square) + 64 * (piece_kind + 5 * relative_colour + 10 * usize::from(king))
}

impl SparseInputType for HalfKp {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        FEATURES
    }

    fn max_active(&self) -> usize {
        30
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        for (piece, square) in *pos {
            let piece_kind = usize::from(piece & 7);
            if piece_kind == 5 {
                continue;
            }
            let relative_colour = usize::from(piece & 8 != 0);
            let stm = feature_index(pos.our_ksq(), relative_colour, piece_kind, square);
            let ntm = feature_index(pos.opp_ksq(), relative_colour ^ 1, piece_kind, square ^ 56);
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        "HalfKP-40960".to_string()
    }

    fn description(&self) -> String {
        "Vanilla HalfKP with rank-flipped opposite perspective".to_string()
    }
}

/// Wraps Bullet's fixed raw layout in Lattice's strict network header.
pub fn pack_network(raw: &[u8]) -> Result<Vec<u8>, String> {
    pack_network_for(raw, NetworkArchitecture::HalfKp)
}

/// Wraps a Bullet checkpoint for the selected feature ABI.
pub fn pack_network_for(raw: &[u8], architecture: NetworkArchitecture) -> Result<Vec<u8>, String> {
    let raw_network_bytes = architecture.raw_bytes();
    let (feature_abi, features) = architecture.header();
    if raw.len() < raw_network_bytes {
        return Err(format!(
            "network is truncated: got {} bytes, need {raw_network_bytes}",
            raw.len(),
        ));
    }
    let padding = &raw[raw_network_bytes..];
    if padding.len() >= 64
        || padding
            .iter()
            .enumerate()
            .any(|(index, &byte)| byte != b"bullet"[index % 6])
    {
        return Err("Bullet padding is malformed".to_string());
    }
    let payload = &raw[..raw_network_bytes];
    let mut output = Vec::with_capacity(56 + payload.len());
    output.extend_from_slice(b"LTNNUE01");
    for value in [1_u32, feature_abi, features as u32, HIDDEN as u32] {
        output.extend_from_slice(&value.to_le_bytes());
    }
    for value in [i32::from(QA), i32::from(QB), SCALE, 0] {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    output.extend_from_slice(&fnv1a(payload).to_le_bytes());
    output.extend_from_slice(payload);
    Ok(output)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// Parsed integer network used to validate exported checkpoints independently
/// of Bullet's floating-point graph.
pub struct QuantizedNetwork {
    feature_bias: [i16; HIDDEN],
    feature_weights: Vec<i16>,
    output_bias: i16,
    output_weights: [i16; 2 * HIDDEN],
}

impl QuantizedNetwork {
    /// Parses a strict Lattice network produced by [`pack_network`].
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 56 + RAW_NETWORK_BYTES || bytes.get(..8) != Some(b"LTNNUE01") {
            return Err("invalid network size or magic".to_string());
        }
        let u32_at = |offset| {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed header"))
        };
        if [u32_at(8), u32_at(12), u32_at(16), u32_at(20)] != [1, 1, FEATURES as u32, HIDDEN as u32]
            || [u32_at(24), u32_at(28), u32_at(32), u32_at(36)]
                != [u32::from(QA as u16), u32::from(QB as u16), SCALE as u32, 0]
        {
            return Err("unsupported network layout".to_string());
        }
        let payload_len = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
        let expected_hash = u64::from_le_bytes(bytes[48..56].try_into().unwrap());
        let payload = &bytes[56..];
        if payload_len != payload.len() || fnv1a(payload) != expected_hash {
            return Err("invalid payload length or hash".to_string());
        }
        let mut cursor = 0;
        let mut next = || {
            let value = i16::from_le_bytes([payload[cursor], payload[cursor + 1]]);
            cursor += 2;
            value
        };
        let mut feature_bias = [0; HIDDEN];
        feature_bias.fill_with(&mut next);
        let feature_weights = (0..FEATURES * HIDDEN).map(|_| next()).collect();
        let output_bias = next();
        let mut output_weights = [0; 2 * HIDDEN];
        output_weights.fill_with(next);
        if cursor != payload.len() {
            return Err("network payload has trailing values".to_string());
        }
        Ok(Self {
            feature_bias,
            feature_weights,
            output_bias,
            output_weights,
        })
    }

    /// Evaluates one mover-relative Bullet record in centipawns.
    pub fn evaluate(&self, board: &ChessBoard) -> i32 {
        let (stm, ntm) = self.accumulators(board);
        let mut output = 0_i64;
        for (&value, &weight) in stm.iter().zip(&self.output_weights[..HIDDEN]) {
            let clipped = i64::from(value).clamp(0, i64::from(QA));
            output += clipped * clipped * i64::from(weight);
        }
        for (&value, &weight) in ntm.iter().zip(&self.output_weights[HIDDEN..]) {
            let clipped = i64::from(value).clamp(0, i64::from(QA));
            output += clipped * clipped * i64::from(weight);
        }
        output /= i64::from(QA);
        output += i64::from(self.output_bias);
        output *= i64::from(SCALE);
        output /= i64::from(QA) * i64::from(QB);
        output as i32
    }

    /// Returns the exact pre-activation accumulator bounds for one record.
    /// Values outside `i16` cannot be represented by Lattice's runtime.
    pub fn accumulator_bounds(&self, board: &ChessBoard) -> (i32, i32) {
        let (stm, ntm) = self.accumulators(board);
        stm.into_iter()
            .chain(ntm)
            .fold((i32::MAX, i32::MIN), |(lo, hi), value| {
                (lo.min(value), hi.max(value))
            })
    }

    fn accumulators(&self, board: &ChessBoard) -> ([i32; HIDDEN], [i32; HIDDEN]) {
        let mut stm = self.feature_bias.map(i32::from);
        let mut ntm = stm;
        HalfKp.map_features(board, |stm_feature, ntm_feature| {
            for neuron in 0..HIDDEN {
                stm[neuron] += i32::from(self.feature_weights[stm_feature * HIDDEN + neuron]);
                ntm[neuron] += i32::from(self.feature_weights[ntm_feature * HIDDEN + neuron]);
            }
        });
        (stm, ntm)
    }
}

/// Bullet's unquantised affine weights in the trainer's saved-format order.
/// This reconstructs the floating-point graph independently of the packed
/// engine evaluator.
pub struct FloatNetwork {
    feature_bias: [f32; HIDDEN],
    feature_weights: Vec<f32>,
    output_bias: f32,
    output_weights: [f32; 2 * HIDDEN],
}

impl FloatNetwork {
    /// Parses Bullet's `raw.bin` for the current `l0b,l0w,l1b,l1w` contract.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 4 * NETWORK_VALUES {
            return Err(format!(
                "invalid floating network size: got {}, need {}",
                bytes.len(),
                4 * NETWORK_VALUES
            ));
        }
        let mut cursor = 0;
        let mut next = || {
            let value = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            value
        };
        let mut feature_bias = [0.0; HIDDEN];
        feature_bias.fill_with(&mut next);
        let feature_weights = (0..FEATURES * HIDDEN).map(|_| next()).collect();
        let output_bias = next();
        let mut output_weights = [0.0; 2 * HIDDEN];
        output_weights.fill_with(next);
        debug_assert_eq!(cursor, bytes.len());
        Ok(Self {
            feature_bias,
            feature_weights,
            output_bias,
            output_weights,
        })
    }

    /// Evaluates Bullet's floating graph and converts its logit to centipawns.
    pub fn evaluate_cp(&self, board: &ChessBoard) -> f32 {
        let mut stm = self.feature_bias;
        let mut ntm = stm;
        HalfKp.map_features(board, |stm_feature, ntm_feature| {
            for neuron in 0..HIDDEN {
                stm[neuron] += self.feature_weights[stm_feature * HIDDEN + neuron];
                ntm[neuron] += self.feature_weights[ntm_feature * HIDDEN + neuron];
            }
        });
        let activate = |value: f32| value.clamp(0.0, 1.0).powi(2);
        let mut output = self.output_bias;
        for (&value, &weight) in stm.iter().zip(&self.output_weights[..HIDDEN]) {
            output += activate(value) * weight;
        }
        for (&value, &weight) in ntm.iter().zip(&self.output_weights[HIDDEN..]) {
            output += activate(value) * weight;
        }
        output * SCALE as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(black_to_move: bool) -> ChessBoard {
        let white = (1_u64 << 4) | (1_u64 << 8);
        let black = 1_u64 << 60;
        ChessBoard::from_raw(
            [
                white,
                black,
                1_u64 << 8,
                0,
                0,
                0,
                0,
                (1_u64 << 4) | (1_u64 << 60),
            ],
            usize::from(black_to_move),
            123,
            1.0,
        )
        .unwrap()
    }

    #[test]
    fn golden_white_to_move_features_match_the_engine_abi() {
        let board = position(false);
        let mut pairs = Vec::new();
        HalfKp.map_features(&board, |stm, ntm| pairs.push((stm, ntm)));
        assert_eq!(pairs, vec![(2_568, 2_928)]);
    }

    #[test]
    fn bullet_records_and_features_are_mover_relative() {
        let board = position(true);
        assert_eq!(board.score, -123);
        assert_eq!(board.result, 0);
        let mut pairs = Vec::new();
        HalfKp.map_features(&board, |stm, ntm| pairs.push((stm, ntm)));
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].0 < FEATURES && pairs[0].1 < FEATURES);
    }

    #[test]
    fn packer_rejects_nonzero_padding_and_adds_header() {
        let padding = 64 - RAW_NETWORK_BYTES % 64;
        let mut raw = vec![0; RAW_NETWORK_BYTES];
        raw.extend((0..padding).map(|index| b"bullet"[index % 6]));
        let packed = pack_network(&raw).unwrap();
        assert_eq!(&packed[..8], b"LTNNUE01");
        assert_eq!(packed.len(), 56 + RAW_NETWORK_BYTES);
        *raw.last_mut().unwrap() = 1;
        assert!(pack_network(&raw).is_err());
    }

    #[test]
    fn packed_zero_network_parses_and_evaluates_to_zero() {
        let padding = 64 - RAW_NETWORK_BYTES % 64;
        let mut raw = vec![0; RAW_NETWORK_BYTES];
        raw.extend((0..padding).map(|index| b"bullet"[index % 6]));
        let packed = pack_network(&raw).unwrap();
        let network = QuantizedNetwork::parse(&packed).unwrap();
        assert_eq!(network.evaluate(&position(false)), 0);
        assert_eq!(network.evaluate(&position(true)), 0);
    }

    #[test]
    fn floating_zero_network_parses_and_evaluates_to_zero() {
        let raw = vec![0; 4 * NETWORK_VALUES];
        let network = FloatNetwork::parse(&raw).unwrap();
        assert_eq!(network.evaluate_cp(&position(false)), 0.0);
        assert_eq!(network.evaluate_cp(&position(true)), 0.0);
    }
}
