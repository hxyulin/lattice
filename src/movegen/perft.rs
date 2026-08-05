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

    /// catches: make/unmake that fails to restore state a leaf count cannot
    /// see - the halfmove clock, castling rights, the en passant square, or the
    /// zobrist key. Perft totals stay correct while the board silently drifts.
    #[test]
    fn generation_restores_the_board_exactly() {
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ];
        for fen in fens {
            let mut board: Board = fen.parse().unwrap();
            let before = format!("{board}");
            let zobrist = board.state().zobrist();
            perft(&mut board, 2);
            assert_eq!(format!("{board}"), before, "{fen}");
            assert_eq!(board.state().zobrist(), zobrist, "{fen}");
        }
    }

    /// catches: `perft_divide` recursing at the wrong depth, or failing to
    /// unmake between root moves. `tests/movegen.rs` pins the same property at
    /// depth 5; the reference total here keeps it cheap.
    #[test]
    fn divide_sums_to_perft() {
        let mut board = Board::startpos();
        let split = perft_divide(&mut board, 3);
        assert_eq!(split.len(), 20);
        assert_eq!(split.iter().map(|(_, n)| n).sum::<u64>(), 8_902);
    }

    /// catches: a sort key that drops the move type, or that orders by
    /// destination before origin. Both positions generate moves in an order
    /// that the mutated keys reorder differently from the real one.
    #[test]
    fn divide_orders_by_type_then_origin_then_destination() {
        for fen in [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            let mut board: Board = fen.parse().unwrap();
            let got: Vec<_> = perft_divide(&mut board, 1)
                .iter()
                .map(|(mv, _)| (mv.move_type() as u16, mv.from().index(), mv.to().index()))
                .collect();
            let mut want = got.clone();
            want.sort_unstable();
            assert_eq!(got, want, "{fen}");
        }
    }

    /// catches: a depth-0 base case that does not count the node itself, and a
    /// depth-1 shortcut that miscounts the move list.
    #[test]
    fn base_cases_count_the_node_and_the_move_list() {
        let mut board = Board::startpos();
        assert_eq!(perft(&mut board, 0), 1);
        assert_eq!(perft(&mut board, 1), 20);
    }

    /// catches: a legality filter that keeps moves leaving the king in check.
    /// Both positions have pseudo-legal moves available and zero legal ones.
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
