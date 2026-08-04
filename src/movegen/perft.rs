use crate::{Board, Move};

use super::{MoveList, generate_legal};

/// Counts legal leaf nodes at a given depth.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate_legal(board, &mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    let mut nodes = 0;
    for &mv in list.iter() {
        board.make(mv);
        nodes += perft(board, depth - 1);
        board.unmake(mv);
    }
    nodes
}

/// Counts each legal root move's leaf nodes in stable move encoding order.
pub fn perft_divide(board: &mut Board, depth: u32) -> Vec<(Move, u64)> {
    if depth == 0 {
        return Vec::new();
    }
    let mut list = MoveList::new();
    generate_legal(board, &mut list);
    let mut result = Vec::with_capacity(list.len());
    for &mv in list.iter() {
        board.make(mv);
        let nodes = perft(board, depth - 1);
        board.unmake(mv);
        result.push((mv, nodes));
    }
    result.sort_by_key(|(mv, _)| {
        (mv.move_type() as u16) << 12 | (mv.from().index() as u16) << 6 | mv.to().index() as u16
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_restores_the_board_exactly() {
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        for fen in fens {
            let mut board: Board = fen.parse().unwrap();
            let before = format!("{board}");
            let zobrist = board.state().zobrist();
            perft(&mut board, 3);
            assert_eq!(format!("{board}"), before, "{fen}");
            assert_eq!(board.state().zobrist(), zobrist, "{fen}");
        }
    }

    #[test]
    fn divide_sums_to_perft() {
        let mut board = Board::startpos();
        let total: u64 = perft_divide(&mut board, 4).iter().map(|(_, n)| n).sum();
        assert_eq!(total, perft(&mut Board::startpos(), 4));
    }

    #[test]
    fn detects_mate_and_stalemate() {
        let mut mate: Board = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1"
            .parse()
            .unwrap();
        assert_eq!(perft(&mut mate, 1), 0);
        let mut stalemate: Board = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1".parse().unwrap();
        assert_eq!(perft(&mut stalemate, 1), 0);
    }
}
