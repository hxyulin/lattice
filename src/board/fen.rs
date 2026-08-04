use super::{Board, CastlingRights, Color, Piece, PieceType, Square, State};
use core::{fmt, str::FromStr};

/// An error returned when parsing an invalid FEN position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenError;
impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid FEN")
    }
}
impl std::error::Error for FenError {}

impl FromStr for Board {
    type Err = FenError;
    fn from_str(fen: &str) -> Result<Self, Self::Err> {
        let mut fields = fen.split_whitespace();
        let placement = fields.next().ok_or(FenError)?;
        let side = fields.next().ok_or(FenError)?;
        let rights = fields.next().ok_or(FenError)?;
        let ep = fields.next().ok_or(FenError)?;
        let halfmove = fields
            .next()
            .ok_or(FenError)?
            .parse::<u8>()
            .map_err(|_| FenError)?;
        let fullmove = fields
            .next()
            .ok_or(FenError)?
            .parse::<u16>()
            .map_err(|_| FenError)?;
        if fields.next().is_some() || fullmove == 0 {
            return Err(FenError);
        }
        let mut board = Board::empty(State {
            side_to_move: match side {
                "w" => Color::White,
                "b" => Color::Black,
                _ => return Err(FenError),
            },
            castling: CastlingRights(parse_rights(rights)?),
            ep: parse_ep(ep)?,
            halfmove,
            fullmove,
            zobrist: 0,
        });
        let ranks: Vec<_> = placement.split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError);
        }
        for (ri, text) in ranks.into_iter().enumerate() {
            let rank = 7 - ri as u8;
            let mut file = 0u8;
            for ch in text.chars() {
                if let Some(n) = ch.to_digit(10) {
                    if n == 0 || n > 8 {
                        return Err(FenError);
                    }
                    file = file.checked_add(n as u8).ok_or(FenError)?;
                } else {
                    if file >= 8 {
                        return Err(FenError);
                    }
                    let (color, lower) = if ch.is_ascii_uppercase() {
                        (Color::White, ch.to_ascii_lowercase())
                    } else {
                        (Color::Black, ch)
                    };
                    let kind = match lower {
                        'p' => PieceType::Pawn,
                        'n' => PieceType::Knight,
                        'b' => PieceType::Bishop,
                        'r' => PieceType::Rook,
                        'q' => PieceType::Queen,
                        'k' => PieceType::King,
                        _ => return Err(FenError),
                    };
                    board.add_piece(
                        Piece::new(color, kind),
                        Square::new_unchecked(rank * 8 + file),
                    );
                    file += 1;
                }
            }
            if file != 8 {
                return Err(FenError);
            }
        }
        board.finish_setup();
        Ok(board)
    }
}

fn parse_rights(s: &str) -> Result<u8, FenError> {
    if s == "-" {
        return Ok(0);
    }
    let mut v = 0;
    for c in s.chars() {
        let b = match c {
            'K' => 1,
            'Q' => 2,
            'k' => 4,
            'q' => 8,
            _ => return Err(FenError),
        };
        if v & b != 0 {
            return Err(FenError);
        }
        v |= b;
    }
    Ok(v)
}
fn parse_ep(s: &str) -> Result<Option<Square>, FenError> {
    if s == "-" {
        return Ok(None);
    }
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return Err(FenError);
    }
    Ok(Some(Square::new_unchecked((b[1] - b'1') * 8 + b[0] - b'a')))
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            if rank != 7 {
                f.write_str("/")?;
            }
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::new_unchecked(rank * 8 + file);
                if let Some(p) = self.piece_on(sq) {
                    if empty > 0 {
                        write!(f, "{empty}")?;
                        empty = 0;
                    }
                    let c = match p.piece_type() {
                        PieceType::Pawn => 'p',
                        PieceType::Knight => 'n',
                        PieceType::Bishop => 'b',
                        PieceType::Rook => 'r',
                        PieceType::Queen => 'q',
                        PieceType::King => 'k',
                    };
                    write!(
                        f,
                        "{}",
                        if p.color() == Color::White {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        }
                    )?;
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                write!(f, "{empty}")?;
            }
        }
        write!(
            f,
            " {} ",
            if self.state().side_to_move() == Color::White {
                "w"
            } else {
                "b"
            }
        )?;
        let r = self.state().castling().0;
        if r == 0 {
            f.write_str("-")?;
        } else {
            for (b, c) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] {
                if r & b != 0 {
                    write!(f, "{c}")?;
                }
            }
        }
        f.write_str(" ")?;
        if let Some(s) = self.state().en_passant() {
            write!(
                f,
                "{}{}",
                (b'a' + s.file()) as char,
                (b'1' + s.rank()) as char
            )?;
        } else {
            f.write_str("-")?;
        }
        write!(
            f,
            " {} {}",
            self.state().halfmove_clock(),
            self.state().fullmove_number()
        )
    }
}
