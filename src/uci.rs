//! UCI command parsing and notation.

use crate::movegen::{MoveList, generate_legal};
use crate::search::Limits;
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
        _ => None,
    }
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

    #[test]
    fn parses_tournament_clock() {
        assert_eq!(
            limits("go wtime 300000 btime 300000 winc 0 binc 0"),
            Limits {
                wtime: Some(300_000),
                btime: Some(300_000),
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
        assert!(parse("setoption name Hash value 16").is_none());
        assert!(parse("vendor extension").is_none());
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
