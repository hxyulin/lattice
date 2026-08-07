//! UCI command parsing and notation.

use std::io::Write;

use crate::movegen::{MoveList, generate_legal};
use crate::search::{Iteration, Limits, SearchListener};
use crate::{Board, Move, PieceType, Square};

/// A parsed UCI command.
#[derive(Debug)]
pub enum Command {
    /// Identify the engine and enter UCI mode.
    Uci,
    /// Confirm that the engine is ready.
    IsReady,
    /// Reset state for a new game.
    NewGame,
    /// Replace the current position.
    Position(Board),
    /// Start a search.
    Go(Limits),
    /// Stop the active search.
    Stop,
    /// Stop the active search and exit.
    Quit,
    /// Run a perft divide to the supplied depth.
    Perft(u32),
    /// Run the fixed search benchmark.
    Bench,
    /// Set an engine option. `value` is absent for button-type options.
    SetOption {
        /// Option name, which may contain spaces.
        name: String,
        /// Option value, which may contain spaces.
        value: Option<String>,
    },
}

/// Parses one line of UCI input. Returns `None` for unrecognised input.
pub fn parse(line: &str) -> Option<Command> {
    let mut words = line.split_whitespace();
    match words.next()? {
        "uci" => Some(Command::Uci),
        "isready" => Some(Command::IsReady),
        "ucinewgame" => Some(Command::NewGame),
        "position" => parse_position(words.collect()).map(Command::Position),
        "go" => parse_go(words).map(Command::Go),
        "stop" => Some(Command::Stop),
        "quit" => Some(Command::Quit),
        "perft" => words.next()?.parse().ok().map(Command::Perft),
        "bench" => Some(Command::Bench),
        "setoption" => parse_setoption(words),
        _ => None,
    }
}

/// `setoption name <name...> [value <value...>]`; both parts may span several
/// tokens, so they are delimited by the keywords rather than token counts.
fn parse_setoption<'a>(mut words: impl Iterator<Item = &'a str>) -> Option<Command> {
    if words.next()? != "name" {
        return None;
    }
    let mut name: Vec<&str> = Vec::new();
    let mut saw_value = false;
    for word in words.by_ref() {
        if word == "value" {
            saw_value = true;
            break;
        }
        name.push(word);
    }
    if name.is_empty() {
        return None;
    }
    Some(Command::SetOption {
        name: name.join(" "),
        value: saw_value.then(|| words.collect::<Vec<_>>().join(" ")),
    })
}

fn parse_go<'a>(mut words: impl Iterator<Item = &'a str>) -> Option<Limits> {
    let mut limits = Limits::default();
    while let Some(word) = words.next() {
        match word {
            "depth" => limits.depth = Some(words.next()?.parse().ok()?),
            "nodes" => limits.nodes = Some(words.next()?.parse().ok()?),
            "movetime" => limits.movetime = Some(words.next()?.parse().ok()?),
            "wtime" => limits.wtime = Some(words.next()?.parse().ok()?),
            "btime" => limits.btime = Some(words.next()?.parse().ok()?),
            "winc" => limits.winc = words.next()?.parse().ok()?,
            "binc" => limits.binc = words.next()?.parse().ok()?,
            "infinite" => limits.infinite = true,
            _ => {}
        }
    }
    Some(limits)
}

fn parse_position(words: Vec<&str>) -> Option<Board> {
    let (mut board, mut index) = match words.first().copied()? {
        "startpos" => (Board::startpos(), 1),
        "fen" if words.len() >= 7 => (words[1..7].join(" ").parse().ok()?, 7),
        _ => return None,
    };
    if words.get(index) == Some(&"moves") {
        index += 1;
        for text in &words[index..] {
            let mv = find_move(&mut board, text)?;
            board.make(mv);
        }
    } else if index != words.len() {
        return None;
    }
    Some(board)
}

fn find_move(board: &mut Board, text: &str) -> Option<Move> {
    let bytes = text.as_bytes();
    if !matches!(bytes.len(), 4 | 5) {
        return None;
    }
    let square = |file: u8, rank: u8| {
        (b'a'..=b'h')
            .contains(&file)
            .then_some(())
            .and_then(|()| (b'1'..=b'8').contains(&rank).then_some(()))
            .and_then(|()| Square::new(file - b'a', rank - b'1'))
    };
    let from = square(bytes[0], bytes[1])?;
    let to = square(bytes[2], bytes[3])?;
    let promotion = bytes.get(4).copied();
    let mut list = MoveList::new();
    generate_legal(board, &mut list);
    list.iter().copied().find(|mv| {
        mv.from() == from
            && mv.to() == to
            && matches!(
                (mv.promoted_piece(), promotion),
                (None, None)
                    | (Some(PieceType::Knight), Some(b'n'))
                    | (Some(PieceType::Bishop), Some(b'b'))
                    | (Some(PieceType::Rook), Some(b'r'))
                    | (Some(PieceType::Queen), Some(b'q'))
            )
    })
}

/// Renders a search as UCI `info` and `bestmove` lines.
///
/// This is the protocol side of [`SearchListener`]: the search reports what it
/// found, and this turns it into the text a GUI reads.
pub struct UciListener<W> {
    output: W,
}

impl<W: Write> UciListener<W> {
    /// Wraps a writer, normally stdout.
    pub fn new(output: W) -> Self {
        Self { output }
    }

    /// Returns the wrapped writer, for callers that captured into a buffer.
    pub fn into_inner(self) -> W {
        self.output
    }
}

impl<W: Write> SearchListener for UciListener<W> {
    fn iteration(&mut self, iteration: &Iteration) {
        let score = iteration.mate_in().map_or_else(
            || format!("score cp {}", iteration.score),
            |moves| format!("score mate {moves}"),
        );
        let pv = iteration
            .best_move
            .map_or_else(String::new, |mv| format!(" pv {}", move_text(mv)));
        let _ = writeln!(
            self.output,
            "info depth {} {score} nodes {} nps {} time {}{pv}",
            iteration.depth,
            iteration.nodes,
            iteration.nps(),
            iteration.elapsed.as_millis(),
        );
    }

    fn finished(&mut self, best_move: Option<Move>) {
        let best = best_move.map_or_else(|| "0000".to_owned(), move_text);
        let _ = writeln!(self.output, "bestmove {best}");
    }
}

/// Returns a move in UCI long algebraic notation.
pub fn move_text(mv: Move) -> String {
    let square = |sq: Square| [(b'a' + sq.file()) as char, (b'1' + sq.rank()) as char];
    let mut text: String = square(mv.from())
        .into_iter()
        .chain(square(mv.to()))
        .collect();
    if let Some(piece) = mv.promoted_piece() {
        text.push(match piece {
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::Pawn | PieceType::King => unreachable!(),
        });
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(line: &str) -> Limits {
        let Some(Command::Go(limits)) = parse(line) else {
            panic!("expected go command");
        };
        limits
    }

    // catches: wtime swapped with btime, winc swapped with binc, and either
    // increment dropped. The values are all distinct for that reason; equal
    // ones let a swap through.
    #[test]
    fn parses_tournament_clock() {
        assert_eq!(
            limits("go wtime 300000 btime 250000 winc 3000 binc 2000"),
            Limits {
                wtime: Some(300_000),
                btime: Some(250_000),
                winc: 3_000,
                binc: 2_000,
                ..Limits::default()
            }
        );
    }

    #[test]
    fn parses_individual_limits() {
        assert_eq!(limits("go depth 5").depth, Some(5));
        assert_eq!(limits("go nodes 10000").nodes, Some(10_000));
        assert_eq!(limits("go movetime 500").movetime, Some(500));
        assert!(limits("go infinite").infinite);
    }

    #[test]
    fn ignores_unrecognised_commands() {
        assert!(parse("vendor extension").is_none());
    }

    fn option(line: &str) -> Option<(String, Option<String>)> {
        match parse(line)? {
            Command::SetOption { name, value } => Some((name, value)),
            _ => None,
        }
    }

    // catches: name or value truncated at the first token, and `value` being
    // treated as part of the name.
    #[test]
    fn parses_setoption_with_multi_word_name_and_value() {
        assert_eq!(
            option("setoption name Hash value 16"),
            Some(("Hash".into(), Some("16".into())))
        );
        assert_eq!(
            option("setoption name Clear Hash"),
            Some(("Clear Hash".into(), None))
        );
        assert_eq!(
            option("setoption name Foo Bar value baz qux"),
            Some(("Foo Bar".into(), Some("baz qux".into())))
        );
    }

    #[test]
    fn rejects_malformed_setoption() {
        assert!(parse("setoption Hash 64").is_none());
        assert!(parse("setoption").is_none());
        assert!(parse("setoption name").is_none());
        assert!(parse("setoption name value 16").is_none());
    }

    // catches: any keyword mapped to the wrong variant or dropped entirely.
    // Nothing else covers isready, ucinewgame, stop, quit or bench, so
    // `"bench" => Command::Stop` was silently a passing change.
    #[test]
    fn every_keyword_maps_to_its_own_command() {
        assert!(matches!(parse("uci"), Some(Command::Uci)));
        assert!(matches!(parse("isready"), Some(Command::IsReady)));
        assert!(matches!(parse("ucinewgame"), Some(Command::NewGame)));
        assert!(matches!(parse("stop"), Some(Command::Stop)));
        assert!(matches!(parse("quit"), Some(Command::Quit)));
        assert!(matches!(parse("bench"), Some(Command::Bench)));
        assert!(matches!(parse("perft 3"), Some(Command::Perft(3))));
        assert!(parse("perft").is_none());
        assert!(parse("perft xyz").is_none());
    }

    // catches: `move_text` swapping from with to, swapping file with rank,
    // omitting the promotion suffix, or collapsing the four promotion pieces
    // onto one letter; and `find_move` ignoring the origin square or the
    // promotion suffix, which made underpromotion unreachable over the wire.
    #[test]
    fn move_text_and_find_move_roundtrip_including_underpromotion() {
        for (fen, text) in [
            ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e4"),
            ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e1d1"),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1"),
            ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1c1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
            ("1q2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8n"),
        ] {
            let mut board: Board = fen.parse().unwrap();
            let mv = find_move(&mut board, text).unwrap_or_else(|| panic!("{text} in {fen}"));
            assert_eq!(move_text(mv), text, "{fen}");
        }
    }

    // catches: the length check widened so a 3-character or 6-character token
    // is accepted, and `find_move` returning a move that is merely legal
    // rather than the one named.
    #[test]
    fn find_move_rejects_malformed_and_illegal_moves() {
        let mut board = Board::startpos();
        for bad in ["e2e", "e2e4e5", "", "z2z4", "e9e4", "e7e5", "e2e5", "e2e4q"] {
            assert!(
                find_move(&mut board, bad).is_none(),
                "accepted {bad:?} from the start position"
            );
        }
        // Only an exact-length check rejects this one: its first five bytes
        // are a legal promotion, so a `len() < 4` bound accepts the trailing
        // garbage silently.
        let mut promo: Board = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
        assert!(find_move(&mut promo, "a7a8qq").is_none());
    }

    #[test]
    fn parses_startpos_and_fen_moves() {
        let Some(Command::Position(startpos)) = parse("position startpos moves e2e4 e7e5") else {
            panic!("expected position command");
        };
        assert_eq!(
            startpos.to_string(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        );
        let Some(Command::Position(fen)) =
            parse("position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 moves e2e4")
        else {
            panic!("expected position command");
        };
        assert_eq!(fen.to_string(), "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1");
    }
}
