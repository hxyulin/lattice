use core::fmt;

use super::features::{CHESS768_FEATURES, FEATURES};

/// Neurons in one HalfKP perspective accumulator.
pub const HIDDEN: usize = 256;
const MAGIC: &[u8; 8] = b"LTNNUE01";
const HEADER_LEN: usize = 56;
const VERSION: u32 = 1;
const HALFKP_ABI: u32 = 1;
const CHESS768_ABI: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FeatureAbi {
    HalfKp,
    Chess768,
}

impl FeatureAbi {
    fn parse(id: u32, features: usize) -> Option<Self> {
        match (id, features) {
            (HALFKP_ABI, FEATURES) => Some(Self::HalfKp),
            (CHESS768_ABI, CHESS768_FEATURES) => Some(Self::Chess768),
            _ => None,
        }
    }
}

#[derive(Clone)]
#[repr(C, align(64))]
pub(crate) struct FeatureColumn(pub(crate) [i16; HIDDEN]);

/// A parsed, quantised Lattice NNUE.
pub struct Network {
    pub(crate) feature_abi: FeatureAbi,
    pub(crate) feature_bias: [i16; HIDDEN],
    pub(crate) feature_weights: Box<[FeatureColumn]>,
    output_bias: i16,
    output_weights: [i16; 2 * HIDDEN],
    qa: i32,
    qb: i32,
    scale: i32,
}

/// Why an NNUE byte stream could not be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkError {
    /// Header is truncated or has the wrong magic.
    Header,
    /// Format, feature ABI, or dimensions are unsupported.
    Layout,
    /// Payload length does not match the header and fixed architecture.
    Length,
    /// Payload integrity hash does not match.
    Hash,
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Header => "invalid NNUE header",
            Self::Layout => "unsupported NNUE layout",
            Self::Length => "invalid NNUE payload length",
            Self::Hash => "NNUE payload hash mismatch",
        })
    }
}

impl std::error::Error for NetworkError {}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, NetworkError> {
    bytes
        .get(offset..offset + 4)
        .and_then(|x| x.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or(NetworkError::Header)
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, NetworkError> {
    read_u32(bytes, offset).map(|x| i32::from_le_bytes(x.to_le_bytes()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, NetworkError> {
    bytes
        .get(offset..offset + 8)
        .and_then(|x| x.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or(NetworkError::Header)
}

fn read_i16(bytes: &[u8], cursor: &mut usize) -> i16 {
    let value = i16::from_le_bytes([bytes[*cursor], bytes[*cursor + 1]]);
    *cursor += 2;
    value
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

impl Network {
    /// Parses the strict little-endian network format.
    pub fn parse(bytes: &[u8]) -> Result<Self, NetworkError> {
        if bytes.get(..8) != Some(MAGIC) {
            return Err(NetworkError::Header);
        }
        let features = read_u32(bytes, 16)? as usize;
        let Some(feature_abi) = FeatureAbi::parse(read_u32(bytes, 12)?, features) else {
            return Err(NetworkError::Layout);
        };
        if read_u32(bytes, 8)? != VERSION || read_u32(bytes, 20)? as usize != HIDDEN {
            return Err(NetworkError::Layout);
        }
        let qa = read_i32(bytes, 24)?;
        let qb = read_i32(bytes, 28)?;
        let scale = read_i32(bytes, 32)?;
        if qa <= 0 || qb <= 0 || scale <= 0 || read_u32(bytes, 36)? != 0 {
            return Err(NetworkError::Layout);
        }
        let expected_payload = 2 * (HIDDEN + features * HIDDEN + 1 + 2 * HIDDEN);
        let payload_len =
            usize::try_from(read_u64(bytes, 40)?).map_err(|_| NetworkError::Length)?;
        if payload_len != expected_payload || bytes.len() != HEADER_LEN + payload_len {
            return Err(NetworkError::Length);
        }
        let payload = &bytes[HEADER_LEN..];
        if fnv1a(payload) != read_u64(bytes, 48)? {
            return Err(NetworkError::Hash);
        }

        let mut cursor = 0;
        let mut feature_bias = [0; HIDDEN];
        for value in &mut feature_bias {
            *value = read_i16(payload, &mut cursor);
        }
        let mut columns = Vec::with_capacity(features);
        for _ in 0..features {
            let mut values = [0; HIDDEN];
            for value in &mut values {
                *value = read_i16(payload, &mut cursor);
            }
            columns.push(FeatureColumn(values));
        }
        let output_bias = read_i16(payload, &mut cursor);
        let mut output_weights = [0; 2 * HIDDEN];
        for value in &mut output_weights {
            *value = read_i16(payload, &mut cursor);
        }
        debug_assert_eq!(cursor, payload.len());

        Ok(Self {
            feature_abi,
            feature_bias,
            feature_weights: columns.into_boxed_slice(),
            output_bias,
            output_weights,
            qa,
            qb,
            scale,
        })
    }

    pub(crate) fn evaluate(&self, us: &[i16; HIDDEN], them: &[i16; HIDDEN]) -> i32 {
        let mut output = screlu_dot(us, &self.output_weights[..HIDDEN], self.qa)
            + screlu_dot(them, &self.output_weights[HIDDEN..], self.qa);
        output /= i64::from(self.qa);
        output += i64::from(self.output_bias);
        output *= i64::from(self.scale);
        output /= i64::from(self.qa) * i64::from(self.qb);
        output.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }
}

fn screlu_dot(values: &[i16; HIDDEN], weights: &[i16], qa: i32) -> i64 {
    #[cfg(all(not(debug_assertions), target_arch = "aarch64"))]
    // SAFETY: NEON is mandatory for AArch64 and the function uses unaligned,
    // bounds-checked chunks of the supplied slices.
    unsafe {
        return screlu_dot_neon(values, weights, qa);
    }

    #[cfg(all(not(debug_assertions), target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime AVX2 feature test.
        return unsafe { screlu_dot_avx2(values, weights, qa) };
    }

    #[cfg(not(all(not(debug_assertions), target_arch = "aarch64")))]
    {
        screlu_dot_scalar(values, weights, qa)
    }
}

fn screlu_dot_scalar(values: &[i16; HIDDEN], weights: &[i16], qa: i32) -> i64 {
    values
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| {
            let clipped = i64::from(value).clamp(0, i64::from(qa));
            clipped * clipped * i64::from(weight)
        })
        .sum()
}

#[cfg(all(not(debug_assertions), target_arch = "aarch64"))]
unsafe fn screlu_dot_neon(values: &[i16; HIDDEN], weights: &[i16], qa: i32) -> i64 {
    use std::arch::aarch64::*;

    // SAFETY: creating a vector from scalar values requires only NEON, which
    // AArch64 guarantees.
    let zero = unsafe { vdupq_n_s32(0) };
    // SAFETY: same as above.
    let upper = unsafe { vdupq_n_s32(qa) };
    // SAFETY: same as above.
    let mut sum0 = unsafe { vdupq_n_s64(0) };
    // SAFETY: same as above.
    let mut sum1 = unsafe { vdupq_n_s64(0) };
    for offset in (0..HIDDEN).step_by(8) {
        // SAFETY: each loop chunk is within both slices.
        let input16 = unsafe { vld1q_s16(values.as_ptr().add(offset)) };
        // SAFETY: each loop chunk is within both slices.
        let weight16 = unsafe { vld1q_s16(weights.as_ptr().add(offset)) };
        // SAFETY: all operations require only mandatory AArch64 NEON.
        let input_lo = unsafe { vmovl_s16(vget_low_s16(input16)) };
        // SAFETY: all operations require only mandatory AArch64 NEON.
        let input_hi = unsafe { vmovl_high_s16(input16) };
        // SAFETY: all operations require only mandatory AArch64 NEON.
        let weight_lo = unsafe { vmovl_s16(vget_low_s16(weight16)) };
        // SAFETY: all operations require only mandatory AArch64 NEON.
        let weight_hi = unsafe { vmovl_high_s16(weight16) };
        // SAFETY: all operations require only mandatory AArch64 NEON.
        let clipped_lo = unsafe { vminq_s32(vmaxq_s32(input_lo, zero), upper) };
        // SAFETY: all operations require only mandatory AArch64 NEON.
        let clipped_hi = unsafe { vminq_s32(vmaxq_s32(input_hi, zero), upper) };
        // SAFETY: each product fits i32: 255^2 * i16::MAX < i32::MAX.
        let product_lo = unsafe { vmulq_s32(vmulq_s32(clipped_lo, clipped_lo), weight_lo) };
        // SAFETY: each product fits i32: 255^2 * i16::MAX < i32::MAX.
        let product_hi = unsafe { vmulq_s32(vmulq_s32(clipped_hi, clipped_hi), weight_hi) };
        // SAFETY: widen before accumulation so the complete dot cannot wrap.
        sum0 = unsafe { vaddq_s64(sum0, vmovl_s32(vget_low_s32(product_lo))) };
        // SAFETY: widen before accumulation so the complete dot cannot wrap.
        sum1 = unsafe { vaddq_s64(sum1, vmovl_high_s32(product_lo)) };
        // SAFETY: widen before accumulation so the complete dot cannot wrap.
        sum0 = unsafe { vaddq_s64(sum0, vmovl_s32(vget_low_s32(product_hi))) };
        // SAFETY: widen before accumulation so the complete dot cannot wrap.
        sum1 = unsafe { vaddq_s64(sum1, vmovl_high_s32(product_hi)) };
    }
    let mut lanes = [0_i64; 2];
    // SAFETY: `lanes` has room for both i64 lanes.
    unsafe { vst1q_s64(lanes.as_mut_ptr(), vaddq_s64(sum0, sum1)) };
    lanes.into_iter().sum()
}

#[cfg(all(not(debug_assertions), target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn screlu_dot_avx2(values: &[i16; HIDDEN], weights: &[i16], qa: i32) -> i64 {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let upper = _mm256_set1_epi32(qa);
    let mut sum = _mm256_setzero_si256();
    for offset in (0..HIDDEN).step_by(8) {
        // SAFETY: each unaligned 8xi16 load stays within its slice.
        let input16 = unsafe { _mm_loadu_si128(values.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: each unaligned 8xi16 load stays within its slice.
        let weight16 = unsafe { _mm_loadu_si128(weights.as_ptr().add(offset).cast::<__m128i>()) };
        let input = _mm256_cvtepi16_epi32(input16);
        let weight = _mm256_cvtepi16_epi32(weight16);
        let clipped = _mm256_min_epi32(_mm256_max_epi32(input, zero), upper);
        let product = _mm256_mullo_epi32(_mm256_mullo_epi32(clipped, clipped), weight);
        let low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(product));
        let high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(product));
        sum = _mm256_add_epi64(sum, _mm256_add_epi64(low, high));
    }
    let mut lanes = [0_i64; 4];
    // SAFETY: `lanes` has room for the full unaligned vector store.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sum) };
    lanes.into_iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_network_parses_and_has_the_fixed_shape() {
        let network = Network::parse(include_bytes!(env!("LATTICE_NNUE_FILE"))).unwrap();
        assert!(matches!(
            (network.feature_abi, network.feature_weights.len()),
            (FeatureAbi::HalfKp, FEATURES) | (FeatureAbi::Chess768, CHESS768_FEATURES)
        ));
        assert_eq!(network.qa, 255);
        assert_eq!(network.qb, 64);
        assert_eq!(network.scale, 400);
    }

    #[test]
    fn parser_rejects_corruption_and_layout_drift() {
        let source = include_bytes!(env!("LATTICE_NNUE_FILE"));
        let mut bytes = source.to_vec();
        bytes[0] ^= 1;
        assert!(matches!(Network::parse(&bytes), Err(NetworkError::Header)));

        bytes.copy_from_slice(source);
        bytes[20..24].copy_from_slice(&128_u32.to_le_bytes());
        assert!(matches!(Network::parse(&bytes), Err(NetworkError::Layout)));

        bytes.copy_from_slice(source);
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert!(matches!(Network::parse(&bytes), Err(NetworkError::Hash)));
    }

    #[test]
    fn vector_dot_is_bit_exact_with_the_scalar_oracle() {
        let mut values = [0_i16; HIDDEN];
        let mut weights = [0_i16; HIDDEN];
        for index in 0..HIDDEN {
            values[index] = ((index * 37) as i16 % 700) - 300;
            weights[index] = ((index * 19) as i16 % 401) - 200;
        }
        assert_eq!(
            screlu_dot(&values, &weights, 255),
            screlu_dot_scalar(&values, &weights, 255)
        );
    }
}
