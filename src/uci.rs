//! UCI command parsing and notation.

use std::io::Write;

use crate::movegen::{MoveList, generate_legal};
use crate::search::{Iteration, Limits, SearchListener};
use crate::{Board, Move, MoveType, PieceType, Square};

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
    Position(Box<Board>),
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
        "position" => parse_position(words.collect())
            .map(Box::new)
            .map(Command::Position),
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

/// Returns a move in standard algebraic notation, as EPD and PGN write it.
///
/// The check and mate suffixes are omitted: producing them means making the
/// move to see whether the reply is check, and every consumer here compares
/// against [`parse_san`], which ignores them on both sides.
pub fn san_text(board: &Board, mv: Move) -> String {
    match mv.move_type() {
        MoveType::KingCastle => return "O-O".to_owned(),
        MoveType::QueenCastle => return "O-O-O".to_owned(),
        _ => {}
    }
    let file_char = |sq: Square| (b'a' + sq.file()) as char;
    let rank_char = |sq: Square| (b'1' + sq.rank()) as char;
    let Some(piece) = board.piece_on(mv.from()) else {
        return move_text(mv);
    };
    let piece_type = piece.piece_type();
    let mut text = String::new();
    if piece_type == PieceType::Pawn {
        // A capturing pawn names its file; a pushing pawn names nothing.
        if mv.is_capture() {
            text.push(file_char(mv.from()));
        }
    } else {
        text.push(piece_letter(piece_type));
        text.push_str(&disambiguator(board, mv, piece_type));
    }
    if mv.is_capture() {
        text.push('x');
    }
    text.push(file_char(mv.to()));
    text.push(rank_char(mv.to()));
    if let Some(promoted) = mv.promoted_piece() {
        text.push('=');
        text.push(piece_letter(promoted));
    }
    text
}

fn piece_letter(piece_type: PieceType) -> char {
    match piece_type {
        PieceType::Pawn => 'P',
        PieceType::Knight => 'N',
        PieceType::Bishop => 'B',
        PieceType::Rook => 'R',
        PieceType::Queen => 'Q',
        PieceType::King => 'K',
    }
}

/// The shortest origin hint that separates `mv` from the other legal moves of
/// the same piece type to the same square: nothing, the file, the rank, or the
/// whole square.
fn disambiguator(board: &Board, mv: Move, piece_type: PieceType) -> String {
    let mut board = board.clone();
    let mut moves = MoveList::new();
    generate_legal(&mut board, &mut moves);
    let rivals: Vec<Move> = moves
        .iter()
        .copied()
        .filter(|&other| {
            other.to() == mv.to()
                && other.from() != mv.from()
                && board
                    .piece_on(other.from())
                    .is_some_and(|piece| piece.piece_type() == piece_type)
        })
        .collect();
    if rivals.is_empty() {
        return String::new();
    }
    let file = (b'a' + mv.from().file()) as char;
    let rank = (b'1' + mv.from().rank()) as char;
    if !rivals.iter().any(|r| r.from().file() == mv.from().file()) {
        return file.to_string();
    }
    if !rivals.iter().any(|r| r.from().rank() == mv.from().rank()) {
        return rank.to_string();
    }
    format!("{file}{rank}")
}

/// Resolves a move written in standard algebraic notation against a position.
///
/// Rather than parsing SAN's disambiguation grammar, this renders every legal
/// move and compares: the grammar is only well defined relative to a position
/// anyway, so generating the candidates answers it exactly and leaves nothing
/// to get subtly wrong. Check, mate, and annotation suffixes are ignored, as
/// are the `e.p.` marker and the `0-0` castling spelling.
pub fn parse_san(board: &Board, san: &str) -> Option<Move> {
    let wanted = normalize_san(san);
    if wanted.is_empty() {
        return None;
    }
    let mut board = board.clone();
    let mut moves = MoveList::new();
    generate_legal(&mut board, &mut moves);
    moves
        .iter()
        .copied()
        .find(|&mv| normalize_san(&san_text(&board, mv)) == wanted)
}

/// Strips what does not identify the move: check, mate and annotation marks,
/// the `e.p.` suffix, and the separators some suites put inside a move.
fn normalize_san(san: &str) -> String {
    let san = san.trim();
    let san = san.strip_suffix("e.p.").unwrap_or(san);
    let mut text: String = san
        .chars()
        .filter(|c| !matches!(c, '+' | '#' | '!' | '?' | '=' | '-' | ' '))
        .collect();
    // `0-0` is the same castling move as `O-O`; the dashes are already gone.
    if text.chars().all(|c| c == '0') && !text.is_empty() {
        text = "O".repeat(text.len());
    }
    text
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

    fn san_of(fen: &str, uci: &str) -> String {
        let mut board: Board = fen.parse().unwrap();
        let mv = find_move(&mut board, uci).expect("test move must be legal");
        san_text(&board, mv)
    }

    #[test]
    fn renders_the_san_cases_that_differ_from_uci() {
        // Piece letter, and a pawn push carrying none.
        assert_eq!(san_of(START, "g1f3"), "Nf3");
        assert_eq!(san_of(START, "e2e4"), "e4");
        // A capturing pawn names its origin file, other pieces use `x` alone.
        let caps = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2";
        assert_eq!(san_of(caps, "e4d5"), "exd5");
        assert_eq!(san_of(KIWIPETE, "e5g6"), "Nxg6");
        // Castling either way, from a position that allows both.
        assert_eq!(san_of(KIWIPETE, "e1g1"), "O-O");
        assert_eq!(san_of(KIWIPETE, "e1c1"), "O-O-O");
        // Promotion, quiet and capturing.
        let promo = "6n1/5P2/8/8/8/8/8/K6k w - - 0 1";
        assert_eq!(san_of(promo, "f7f8q"), "f8=Q");
        assert_eq!(san_of(promo, "f7g8n"), "fxg8=N");
        // En passant: a pawn capture like any other, named by its file.
        let ep = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
        assert_eq!(san_of(ep, "e5d6"), "exd6");
    }

    #[test]
    fn disambiguates_by_the_shortest_hint_that_works() {
        // Two knights reach b3 from different files, so the file suffices.
        let files = "4k3/8/8/8/8/8/8/N1N1K3 w - - 0 1";
        assert_eq!(san_of(files, "a1b3"), "Nab3");
        // Same file, different ranks: the rank is the distinguishing part.
        let ranks = "4k3/8/8/N7/8/8/8/N3K3 w - - 0 1";
        assert_eq!(san_of(ranks, "a1b3"), "N1b3");
        // Three queens bear on d4: one shares the a-file with a1 and another
        // shares rank 1, so neither half alone identifies it.
        let both = "1k6/8/8/8/Q7/8/8/Q2QK3 w - - 0 1";
        assert_eq!(san_of(both, "a1d4"), "Qa1d4");
        // A lone piece never takes a hint.
        assert_eq!(san_of(START, "b1c3"), "Nc3");
    }

    #[test]
    fn parse_san_accepts_what_the_suites_write() {
        let mut board: Board = KIWIPETE.parse().unwrap();
        let expect = |san: &str, uci: &str| {
            let mut board: Board = KIWIPETE.parse().unwrap();
            let want = find_move(&mut board, uci).unwrap();
            assert_eq!(parse_san(&board, san), Some(want), "parsing {san}");
        };
        expect("Nxg6", "e5g6");
        expect("O-O", "e1g1");
        expect("0-0", "e1g1");
        expect("O-O-O", "e1c1");
        // Check, mate and annotation marks carry no information about which
        // move was meant, and the suites are inconsistent about them.
        expect("Nxg6+", "e5g6");
        expect("Nxg6!!", "e5g6");
        expect("Nxg6?!", "e5g6");
        // Whitespace around the token, as `bm Nxg6 ;` leaves behind.
        expect("  Nxg6  ", "e5g6");
        // An illegal or unparsable move resolves to nothing rather than to
        // some other move that happens to share a prefix.
        assert_eq!(parse_san(&board, "Nxg7"), None);
        assert_eq!(parse_san(&board, "zzz"), None);
        assert_eq!(parse_san(&board, ""), None);
        let _ = &mut board;
    }

    // catches: a disambiguation rule that renders a move the same way as one
    // of its rivals. Any such collision makes parse_san return whichever came
    // first, silently scoring the wrong move as solved. Rendering every legal
    // move in a position and requiring the strings to be distinct is the
    // property that rules it out, and it covers far more shapes than the
    // handwritten cases above.
    #[test]
    fn san_is_unique_and_roundtrips_for_every_legal_move() {
        crate::movegen::init();
        for fen in [START, KIWIPETE, POSITION_3, POSITION_4, POSITION_5] {
            let mut board: Board = fen.parse().unwrap();
            let mut moves = MoveList::new();
            generate_legal(&mut board, &mut moves);
            let mut seen: Vec<String> = Vec::new();
            for &mv in moves.iter() {
                let san = san_text(&board, mv);
                assert!(
                    !seen.contains(&san),
                    "{san} rendered twice in {fen}, so it cannot be resolved"
                );
                seen.push(san.clone());
                assert_eq!(
                    parse_san(&board, &san),
                    Some(mv),
                    "{san} did not resolve back to itself in {fen}"
                );
            }
            assert!(!seen.is_empty());
        }
    }

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    const POSITION_3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    const POSITION_4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    const POSITION_5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1";
}
