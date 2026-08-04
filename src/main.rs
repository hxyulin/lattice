//! Lattice UCI binary.

use std::io::{self, BufRead};

use lattice::movegen::perft::perft_divide;
use lattice::movegen::{MoveList, generate_legal};
use lattice::{Board, Move, PieceType};

fn main() {
    let stdin = io::stdin();
    let mut board = Board::startpos();
    for line in stdin.lock().lines().map_while(Result::ok) {
        let mut words = line.split_whitespace();
        // ponytail: keep UCI dispatch flat while the protocol surface is small.
        match words.next() {
            Some("uci") => {
                println!("id name Lattice");
                println!("id author hxyulin");
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("ucinewgame") => board = Board::startpos(),
            Some("position") => {
                if let Some(position) = parse_position(words.collect()) {
                    board = position;
                }
            }
            Some("perft") => {
                if let Some(depth) = words.next().and_then(|word| word.parse().ok()) {
                    let divide = perft_divide(&mut board, depth);
                    let total: u64 = divide.iter().map(|(_, nodes)| nodes).sum();
                    for (mv, nodes) in divide {
                        println!("{}: {nodes}", move_text(mv));
                    }
                    println!();
                    println!("Nodes searched: {total}");
                }
            }
            Some("quit") => break,
            _ => {}
        }
    }
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
            .and_then(|()| lattice::Square::new(file - b'a', rank - b'1'))
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

fn move_text(mv: Move) -> String {
    let square = |sq: lattice::Square| [(b'a' + sq.file()) as char, (b'1' + sq.rank()) as char];
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
