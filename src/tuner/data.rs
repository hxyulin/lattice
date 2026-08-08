//! Reading nyquist datagen shards.
//!
//! A shard is a flat run of 32-byte little-endian `bulletformat::ChessBoard`
//! records with no header and no delimiters, which is the raw in-memory layout
//! of that struct as written by bulletformat 1.8.0. The layout is reproduced
//! here rather than depended on: it is 32 bytes of fixed offsets, and a build
//! dependency for one struct is a poor trade.
//!
//! Records arrive canonicalized to the side to move. When Black was on move the
//! producer byte-swapped every bitboard, swapped the two colour boards, and
//! negated the score and result, so a decoded record always reads as White to
//! move. Lattice's own evaluation is side-to-move relative, so the two agree
//! without a correction - but the swap is a rank flip, so the squares are not
//! the ones the original game saw.

use crate::{Board, Color, Piece, PieceType, Square, board::State};

/// Bytes per record, asserted by bulletformat's own `_RIGHT_SIZE` check.
pub const RECORD_BYTES: usize = 32;

/// One training position: a board, and how the game it came from ended.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The position, always with White to move after canonicalization.
    pub board: Board,
    /// Game outcome for the side to move: 0.0 loss, 0.5 draw, 1.0 win.
    pub result: f64,
    /// Engine score in centipawns, relative to the side to move.
    ///
    /// Not used by a pure WDL fit. Treat with suspicion: the producer hardcodes
    /// `require_score: false`, so a position whose engine reported no score is
    /// retained with a fabricated zero and nothing marks it.
    pub score: i16,
}

/// Why a record could not be decoded.
#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The buffer is not a whole number of records.
    Ragged(usize),
    /// `result` was not one of the three defined values.
    BadResult(u8),
    /// A nibble decoded to a role above `King`.
    BadPiece(u8),
    /// The position has no king for one of the sides.
    NoKing,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ragged(len) => write!(f, "{len} bytes is not a whole number of 32-byte records"),
            Self::BadResult(value) => write!(f, "result {value} is not 0, 1 or 2"),
            Self::BadPiece(value) => write!(f, "piece nibble {value} has no role"),
            Self::NoKing => write!(f, "position is missing a king"),
        }
    }
}

/// Decodes one 32-byte record.
///
/// The piece nibbles are packed in the order the occupancy bits come out of
/// `trailing_zeros`, two per byte, low nibble first. Each is `colour << 3 |
/// role`, and the roles are in the same order as [`PieceType`].
fn decode(record: &[u8; RECORD_BYTES]) -> Result<Sample, DecodeError> {
    let mut occ = u64::from_le_bytes(record[0..8].try_into().expect("8 bytes"));
    let pcs = &record[8..24];
    let score = i16::from_le_bytes(record[24..26].try_into().expect("2 bytes"));
    let result = match record[26] {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        other => return Err(DecodeError::BadResult(other)),
    };

    let mut board = Board::empty(State {
        side_to_move: Color::White,
        castling: crate::board::CastlingRights(0),
        ep: None,
        halfmove: 0,
        fullmove: 1,
        zobrist: 0,
        pawn_key: 0,
    });

    let mut index = 0;
    while occ != 0 {
        let square = occ.trailing_zeros() as u8;
        occ &= occ - 1;
        let nibble = (pcs[index / 2] >> (4 * (index & 1))) & 0b1111;
        index += 1;

        let role = nibble & 0b111;
        if role > PieceType::King as u8 {
            return Err(DecodeError::BadPiece(nibble));
        }
        let color = if nibble & 0b1000 == 0 {
            Color::White
        } else {
            Color::Black
        };
        board.add_piece(
            Piece::new(color, PieceType::from_index(role)),
            Square::new_unchecked(square),
        );
    }
    // Castling rights and en passant are absent from the format, so the decoded
    // board is a placement rather than a full position. That is all the
    // evaluation reads, but `king_square` panics on a board without one, so the
    // gap is closed here rather than at the first eval.
    if (board.pieces(PieceType::King) & board.color(Color::White)).is_empty()
        || (board.pieces(PieceType::King) & board.color(Color::Black)).is_empty()
    {
        return Err(DecodeError::NoKing);
    }
    board.finish_setup();

    Ok(Sample {
        board,
        result,
        score,
    })
}

/// Decodes a whole shard.
///
/// Returns the samples and the count of records that failed to decode, rather
/// than failing the shard: a 256 MB shard is worth keeping for one bad record,
/// and a caller that wants strictness can check the count is zero.
pub fn read_shard(bytes: &[u8]) -> Result<(Vec<Sample>, usize), DecodeError> {
    if !bytes.len().is_multiple_of(RECORD_BYTES) {
        return Err(DecodeError::Ragged(bytes.len()));
    }
    let mut samples = Vec::with_capacity(bytes.len() / RECORD_BYTES);
    let mut rejected = 0;
    for chunk in bytes.chunks_exact(RECORD_BYTES) {
        let record: &[u8; RECORD_BYTES] = chunk.try_into().expect("chunks_exact");
        match decode(record) {
            Ok(sample) => samples.push(sample),
            Err(_) => rejected += 1,
        }
    }
    Ok((samples, rejected))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a record the way the producer does, so the test exercises the
    /// real packing rather than a restatement of the decoder.
    fn encode(pieces: &[(Color, PieceType, u8)], score: i16, result: u8) -> [u8; RECORD_BYTES] {
        let mut record = [0u8; RECORD_BYTES];
        let mut occ = 0u64;
        for &(_, _, square) in pieces {
            occ |= 1 << square;
        }
        record[0..8].copy_from_slice(&occ.to_le_bytes());

        // Nibbles follow occupancy order, not the caller's order.
        let mut sorted: Vec<_> = pieces.to_vec();
        sorted.sort_by_key(|&(_, _, square)| square);
        for (index, (color, kind, _)) in sorted.into_iter().enumerate() {
            let nibble = ((color as u8) << 3) | kind as u8;
            record[8 + index / 2] |= nibble << (4 * (index & 1));
        }
        record[24..26].copy_from_slice(&score.to_le_bytes());
        record[26] = result;
        record
    }

    // catches: nibble order reversed, colour bit misread, occupancy walked
    // high-to-low, and the two-per-byte packing losing the odd one.
    #[test]
    fn a_record_round_trips_to_the_placement_it_encodes() {
        let pieces = [
            (Color::White, PieceType::King, 4),
            (Color::White, PieceType::Rook, 0),
            (Color::Black, PieceType::King, 60),
            (Color::Black, PieceType::Pawn, 51),
            (Color::White, PieceType::Knight, 6),
        ];
        let sample = decode(&encode(&pieces, 42, 2)).expect("decodes");

        for (color, kind, square) in pieces {
            let square = Square::new_unchecked(square);
            assert_eq!(
                sample.board.piece_on(square),
                Some(Piece::new(color, kind)),
                "wrong piece on square {square:?}"
            );
        }
        assert_eq!(sample.board.occupied().count(), pieces.len() as u32);
        assert_eq!(sample.score, 42);
        assert_eq!(sample.result, 1.0);
    }

    // catches: the three result codes being mapped to the wrong scores, which
    // would train every position against an inverted or flattened target.
    #[test]
    fn the_result_codes_map_to_loss_draw_and_win() {
        let pieces = [
            (Color::White, PieceType::King, 4),
            (Color::Black, PieceType::King, 60),
        ];
        for (code, expected) in [(0u8, 0.0), (1, 0.5), (2, 1.0)] {
            let sample = decode(&encode(&pieces, 0, code)).expect("decodes");
            assert_eq!(sample.result, expected, "result code {code}");
        }
        assert_eq!(
            decode(&encode(&pieces, 0, 3)).unwrap_err(),
            DecodeError::BadResult(3)
        );
    }

    // catches: a shard whose length is not a multiple of the record size being
    // read as if the tail were a whole record.
    #[test]
    fn a_ragged_shard_is_rejected_rather_than_truncated() {
        assert_eq!(
            read_shard(&[0u8; RECORD_BYTES + 7]).unwrap_err(),
            DecodeError::Ragged(RECORD_BYTES + 7)
        );
        assert!(read_shard(&[]).is_ok());
    }

    // catches: a kingless record reaching the evaluation, where `king_square`
    // unwraps a missing king and panics. An all-zero buffer is the realistic
    // way this arrives - a short read, or a sparse file.
    #[test]
    fn a_kingless_record_is_rejected_rather_than_panicking() {
        let (samples, rejected) = read_shard(&[0u8; RECORD_BYTES * 3]).expect("well-formed");
        assert!(samples.is_empty());
        assert_eq!(rejected, 3);
    }
}
