use crate::{Board, Color, Piece, PieceType, Square};

use super::{
    features::{chess768_indices, feature_index},
    network::{FeatureAbi, HIDDEN, Network},
};

/// Two king-perspective feature-transformer sums maintained by the board.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Accumulator {
    values: [[i16; HIDDEN]; 2],
    kings: [Option<Square>; 2],
    dirty: [bool; 2],
}

impl core::fmt::Debug for Accumulator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NnueAccumulator")
            .field("kings", &self.kings)
            .field("dirty", &self.dirty)
            .finish_non_exhaustive()
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            values: [[0; HIDDEN]; 2],
            kings: [None, None],
            dirty: [true, true],
        }
    }
}

fn apply(dst: &mut [i16; HIDDEN], src: &[i16; HIDDEN], add: bool) {
    #[cfg(debug_assertions)]
    for (to, &from) in dst.iter_mut().zip(src) {
        let value = if add {
            i32::from(*to) + i32::from(from)
        } else {
            i32::from(*to) - i32::from(from)
        };
        debug_assert!(i16::try_from(value).is_ok(), "NNUE accumulator overflow");
        *to = value as i16;
    }

    #[cfg(all(not(debug_assertions), target_arch = "aarch64"))]
    // SAFETY: NEON is mandatory in AArch64. Both arrays contain HIDDEN i16s;
    // the intrinsic loads are unaligned and the loop stays within the arrays.
    unsafe {
        apply_neon(dst, src, add);
    }

    #[cfg(all(not(debug_assertions), target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime feature test guards the target-feature body,
            // whose unaligned loads remain within the fixed-size arrays.
            unsafe { apply_avx2(dst, src, add) };
            return;
        }
        for (to, &from) in dst.iter_mut().zip(src) {
            *to = if add {
                to.wrapping_add(from)
            } else {
                to.wrapping_sub(from)
            };
        }
    }

    #[cfg(all(
        not(debug_assertions),
        not(any(target_arch = "aarch64", target_arch = "x86_64"))
    ))]
    for (to, &from) in dst.iter_mut().zip(src) {
        *to = if add {
            to.wrapping_add(from)
        } else {
            to.wrapping_sub(from)
        };
    }
}

#[cfg(all(not(debug_assertions), target_arch = "aarch64"))]
unsafe fn apply_neon(dst: &mut [i16; HIDDEN], src: &[i16; HIDDEN], add: bool) {
    use std::arch::aarch64::{vaddq_s16, vld1q_s16, vst1q_s16, vsubq_s16};

    for offset in (0..HIDDEN).step_by(8) {
        // SAFETY: offset..offset+8 is in both HIDDEN-element arrays.
        let (left, right) = unsafe {
            (
                vld1q_s16(dst.as_ptr().add(offset)),
                vld1q_s16(src.as_ptr().add(offset)),
            )
        };
        let value = if add {
            // SAFETY: AArch64 guarantees NEON for this target.
            unsafe { vaddq_s16(left, right) }
        } else {
            // SAFETY: AArch64 guarantees NEON for this target.
            unsafe { vsubq_s16(left, right) }
        };
        // SAFETY: the destination range is in the HIDDEN-element array.
        unsafe { vst1q_s16(dst.as_mut_ptr().add(offset), value) };
    }
}

#[cfg(all(not(debug_assertions), target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn apply_avx2(dst: &mut [i16; HIDDEN], src: &[i16; HIDDEN], add: bool) {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi16, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_sub_epi16,
    };

    for offset in (0..HIDDEN).step_by(16) {
        // SAFETY: offset..offset+16 is in both HIDDEN-element arrays and the
        // unaligned intrinsics impose no pointer-alignment precondition.
        let (left, right) = unsafe {
            (
                _mm256_loadu_si256(dst.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(src.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        let value = if add {
            _mm256_add_epi16(left, right)
        } else {
            _mm256_sub_epi16(left, right)
        };
        // SAFETY: the destination range is in the HIDDEN-element array.
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(offset).cast::<__m256i>(), value) };
    }
}

impl Accumulator {
    pub(crate) fn add_piece(&mut self, piece: Piece, square: Square, net: &Network) {
        if net.feature_abi == FeatureAbi::Chess768 {
            if piece.piece_type() == PieceType::King {
                self.kings[piece.color() as usize] = Some(square);
            }
            for (side, index) in chess768_indices(piece, square).into_iter().enumerate() {
                if !self.dirty[side] {
                    apply(&mut self.values[side], &net.feature_weights[index].0, true);
                }
            }
            return;
        }
        if piece.piece_type() == PieceType::King {
            let side = piece.color() as usize;
            self.kings[side] = Some(square);
            self.dirty[side] = true;
            return;
        }
        for perspective in [Color::White, Color::Black] {
            let side = perspective as usize;
            if !self.dirty[side] {
                let index =
                    feature_index(perspective, self.kings[side].unwrap(), piece, square).unwrap();
                apply(&mut self.values[side], &net.feature_weights[index].0, true);
            }
        }
    }

    pub(crate) fn remove_piece(&mut self, piece: Piece, square: Square, net: &Network) {
        if net.feature_abi == FeatureAbi::Chess768 {
            if piece.piece_type() == PieceType::King {
                self.kings[piece.color() as usize] = None;
            }
            for (side, index) in chess768_indices(piece, square).into_iter().enumerate() {
                if !self.dirty[side] {
                    apply(&mut self.values[side], &net.feature_weights[index].0, false);
                }
            }
            return;
        }
        if piece.piece_type() == PieceType::King {
            let side = piece.color() as usize;
            self.kings[side] = None;
            self.dirty[side] = true;
            return;
        }
        for perspective in [Color::White, Color::Black] {
            let side = perspective as usize;
            if !self.dirty[side] {
                let index =
                    feature_index(perspective, self.kings[side].unwrap(), piece, square).unwrap();
                apply(&mut self.values[side], &net.feature_weights[index].0, false);
            }
        }
    }

    pub(crate) fn refresh_all(&mut self, board: &Board, net: &Network) {
        self.dirty = [true, true];
        self.refresh_dirty(board, net);
    }

    pub(crate) fn refresh_dirty(&mut self, board: &Board, net: &Network) {
        for perspective in [Color::White, Color::Black] {
            let side = perspective as usize;
            if !self.dirty[side] {
                continue;
            }
            let king = board.king_square(perspective);
            self.kings[side] = Some(king);
            self.values[side] = net.feature_bias;
            for square in board.occupied() {
                let piece = board.piece_on(square).unwrap();
                let index = match net.feature_abi {
                    FeatureAbi::HalfKp => feature_index(perspective, king, piece, square),
                    FeatureAbi::Chess768 => Some(chess768_indices(piece, square)[side]),
                };
                if let Some(index) = index {
                    apply(&mut self.values[side], &net.feature_weights[index].0, true);
                }
            }
            self.dirty[side] = false;
        }
    }

    pub(crate) fn evaluate(&self, side_to_move: Color, net: &Network) -> i32 {
        debug_assert_eq!(self.dirty, [false, false]);
        let us = side_to_move as usize;
        net.evaluate(&self.values[us], &self.values[side_to_move.flip() as usize])
    }

    #[cfg(any(test, debug_assertions))]
    pub(crate) fn scanned(board: &Board, net: &Network) -> Self {
        let mut accumulator = Self::default();
        accumulator.refresh_all(board, net);
        accumulator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_updates_match_wrapping_scalar_arithmetic() {
        let original: [i16; HIDDEN] = core::array::from_fn(|index| (index as i16 - 128) * 50);
        let source: [i16; HIDDEN] = core::array::from_fn(|index| (index as i16 - 128) * -20);
        for add in [true, false] {
            let mut actual = original;
            apply(&mut actual, &source, add);
            let expected = core::array::from_fn(|index| {
                if add {
                    original[index].wrapping_add(source[index])
                } else {
                    original[index].wrapping_sub(source[index])
                }
            });
            assert_eq!(actual, expected);
        }
    }
}
