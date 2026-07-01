//! A minimal UCI (Universal Chess Interface) front end.
//!
//! # Notes
//!
//! A parser + formatter only; the engine still owns the logic.

use std::io::{BufRead, Write};

use crate::{PieceType, Square};
use bstr::{BString, ByteSlice};

/// An error from the UCI IO layer.
#[derive(Debug, thiserror::Error)]
pub enum UciError {
    /// An underlying read/write failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A move in UCI long algebraic notation (`e2e4`, `e7e8q`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UciMove {
    /// Origin square.
    pub from: Square,
    /// Destination square.
    pub to: Square,
    /// Promotion target, if the move is a promotion.
    pub promo: Option<PieceType>,
}

/// The starting point of a `position` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartPos {
    /// The standard initial position.
    Startpos,
    /// A position given as a FEN string (the six fields, space-joined).
    Fen(BString),
}

/// Parsed parameters of a `go` command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Go {
    /// `go perft <n>` - run a perft to depth `n` instead of searching.
    pub perft: Option<u32>,
    /// Fixed search depth.
    pub depth: Option<u32>,
    /// Fixed time per move, in milliseconds.
    pub movetime: Option<u64>,
    /// Node budget for the search.
    pub nodes: Option<u64>,
    /// White's clock remaining, in milliseconds.
    pub wtime: Option<u64>,
    /// Black's clock remaining, in milliseconds.
    pub btime: Option<u64>,
    /// White's per-move increment, in milliseconds.
    pub winc: Option<u64>,
    /// Black's per-move increment, in milliseconds.
    pub binc: Option<u64>,
    /// Moves until the next time control. Parsed but not yet used - the time
    /// budget assumes sudden death.
    pub movestogo: Option<u32>,
    /// Search until `stop`.
    pub infinite: bool,
}

/// A parsed UCI command from the GUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UciCommand {
    /// `uci` - identify and list options, ending with `uciok`.
    Uci,
    /// `isready` - reply `readyok` once ready.
    IsReady,
    /// `ucinewgame` - a new game is starting.
    NewGame,
    /// `position ...` - set up the board, then optionally apply moves.
    Position {
        /// Where to start from.
        start: StartPos,
        /// Moves to apply from `start`, in order.
        moves: Vec<UciMove>,
    },
    /// `go ...` - start calculating.
    Go(Go),
    /// `stop` - stop calculating and report `bestmove`.
    Stop,
    /// `quit` - terminate.
    Quit,
    /// An unrecognized line; surfaced (not dropped) so the application can log it.
    Unknown(BString),
}

/// Parse a single input line into a [`UciCommand`].
///
/// # Notes
///
/// Tokenizes on ASCII whitespace via [`bstr::ByteSlice::fields`]. Unrecognized
/// lines fall through to [`UciCommand::Unknown`] per UCI.
#[must_use]
pub fn parse_command(line: &[u8]) -> UciCommand {
    let mut toks = line.fields();
    let Some(head) = toks.next() else {
        return UciCommand::Unknown(BString::from(line));
    };
    match head {
        b"uci" => UciCommand::Uci,
        b"isready" => UciCommand::IsReady,
        b"ucinewgame" => UciCommand::NewGame,
        b"stop" => UciCommand::Stop,
        b"quit" => UciCommand::Quit,
        b"position" => parse_position(toks),
        b"go" => UciCommand::Go(parse_go(toks)),
        _ => UciCommand::Unknown(BString::from(line)),
    }
}

fn parse_position<'a>(mut toks: impl Iterator<Item = &'a [u8]>) -> UciCommand {
    let start = match toks.next() {
        Some(b"startpos") => StartPos::Startpos,
        Some(b"fen") => {
            // FEN is up to six space-separated fields; collect until `moves` or end.
            let mut fen = BString::default();
            let mut hit_moves = false;
            for t in toks.by_ref() {
                if t == b"moves" {
                    hit_moves = true;
                    break;
                }
                if !fen.is_empty() {
                    fen.push(b' ');
                }
                fen.extend_from_slice(t);
            }
            let moves = if hit_moves {
                parse_moves(toks)
            } else {
                Vec::new()
            };
            return UciCommand::Position {
                start: StartPos::Fen(fen),
                moves,
            };
        }
        _ => return UciCommand::Unknown(BString::from(&b"position"[..])),
    };
    let moves = match toks.next() {
        Some(b"moves") => parse_moves(toks),
        _ => Vec::new(),
    };
    UciCommand::Position { start, moves }
}

fn parse_moves<'a>(toks: impl Iterator<Item = &'a [u8]>) -> Vec<UciMove> {
    toks.filter_map(parse_uci_move).collect()
}

fn parse_uci_move(tok: &[u8]) -> Option<UciMove> {
    if tok.len() < 4 {
        return None;
    }
    let from = Square::from_ascii(&tok[0..2]).ok()?;
    let to = Square::from_ascii(&tok[2..4]).ok()?;
    let promo = match tok.get(4) {
        None => None,
        Some(b'n') => Some(PieceType::Knight),
        Some(b'b') => Some(PieceType::Bishop),
        Some(b'r') => Some(PieceType::Rook),
        Some(b'q') => Some(PieceType::Queen),
        Some(_) => return None,
    };
    Some(UciMove { from, to, promo })
}

fn parse_go<'a>(mut toks: impl Iterator<Item = &'a [u8]>) -> Go {
    let mut go = Go::default();
    while let Some(t) = toks.next() {
        match t {
            b"perft" => go.perft = toks.next().and_then(parse_num),
            b"depth" => go.depth = toks.next().and_then(parse_num),
            b"movetime" => go.movetime = toks.next().and_then(parse_num),
            b"nodes" => go.nodes = toks.next().and_then(parse_num),
            b"wtime" => go.wtime = toks.next().and_then(parse_num),
            b"btime" => go.btime = toks.next().and_then(parse_num),
            b"winc" => go.winc = toks.next().and_then(parse_num),
            b"binc" => go.binc = toks.next().and_then(parse_num),
            b"movestogo" => go.movestogo = toks.next().and_then(parse_num),
            b"infinite" => go.infinite = true,
            _ => {} // ignore params we don't model
        }
    }
    go
}

fn parse_num<T: std::str::FromStr>(tok: &[u8]) -> Option<T> {
    tok.to_str().ok()?.parse().ok()
}

/// Line-oriented UCI IO over any reader/writer.
///
/// # Notes
///
/// Reads raw bytes (UCI is ASCII) and flushes after each response.
pub struct UciInterface<R, W> {
    input: R,
    output: W,
    buf: Vec<u8>,
}

impl<R: BufRead, W: Write> UciInterface<R, W> {
    /// Wrap a reader and writer.
    pub fn new(input: R, output: W) -> Self {
        Self {
            input,
            output,
            buf: Vec::with_capacity(256),
        }
    }

    /// Read and parse the next command. `Ok(None)` signals EOF (stdin closed).
    pub fn poll(&mut self) -> Result<Option<UciCommand>, UciError> {
        self.buf.clear();
        if self.input.read_until(b'\n', &mut self.buf)? == 0 {
            return Ok(None);
        }
        Ok(Some(parse_command(&self.buf)))
    }

    /// Write one response line (newline appended) and flush.
    pub fn send(&mut self, line: &str) -> Result<(), UciError> {
        self.output.write_all(line.as_bytes())?;
        self.output.write_all(b"\n")?;
        self.output.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_commands() {
        assert_eq!(parse_command(b"uci\n"), UciCommand::Uci);
        assert_eq!(parse_command(b"  isready  "), UciCommand::IsReady);
        assert_eq!(parse_command(b"quit"), UciCommand::Quit);
    }

    #[test]
    fn parses_startpos_with_moves() {
        let cmd = parse_command(b"position startpos moves e2e4 e7e5 e1g1");
        let UciCommand::Position { start, moves } = cmd else {
            panic!()
        };
        assert_eq!(start, StartPos::Startpos);
        assert_eq!(moves.len(), 3);
        assert_eq!(moves[0].from, Square::from_ascii(b"e2").unwrap());
        assert_eq!(moves[2].to, Square::from_ascii(b"g1").unwrap());
    }

    #[test]
    fn parses_fen_then_moves() {
        let cmd = parse_command(
            b"position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves a2a4",
        );
        let UciCommand::Position { start, moves } = cmd else {
            panic!()
        };
        assert_eq!(
            start,
            StartPos::Fen(BString::from(
                &b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"[..]
            ))
        );
        assert_eq!(moves.len(), 1);
    }

    #[test]
    fn parses_promotion_move() {
        let cmd = parse_command(b"position startpos moves a7a8q");
        let UciCommand::Position { moves, .. } = cmd else {
            panic!()
        };
        assert_eq!(moves[0].promo, Some(PieceType::Queen));
    }

    #[test]
    fn parses_go_perft_and_params() {
        assert_eq!(
            parse_command(b"go perft 5"),
            UciCommand::Go(Go {
                perft: Some(5),
                ..Go::default()
            })
        );
        let UciCommand::Go(go) = parse_command(b"go depth 8 movetime 1000") else {
            panic!()
        };
        assert_eq!((go.depth, go.movetime), (Some(8), Some(1000)));
    }
}
