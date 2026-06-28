//! A minimal UCI engine loop: read commands, keep a board, answer `go perft`.

use std::io::{self, BufReader};

use lattice_board::{Board, Move};
use lattice_uci::{StartPos, UciCommand, UciInterface, UciMove};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut uci = UciInterface::new(BufReader::new(stdin.lock()), io::stdout().lock());
    let mut board = Board::starting_position();

    while let Some(cmd) = uci.poll().map_err(io_err)? {
        match cmd {
            UciCommand::Uci => {
                uci.send("id name lattice-engine").map_err(io_err)?;
                uci.send("id author hxyulin").map_err(io_err)?;
                uci.send("uciok").map_err(io_err)?;
            }
            UciCommand::IsReady => uci.send("readyok").map_err(io_err)?,
            UciCommand::NewGame => board = Board::starting_position(),
            UciCommand::Position { start, moves } => {
                let base = match start {
                    StartPos::Startpos => Some(Board::starting_position()),
                    StartPos::Fen(fen) => Board::from_fen(&fen).ok(),
                };
                // Ignore a malformed FEN rather than corrupting the current board.
                if let Some(mut b) = base {
                    for um in &moves {
                        if let Some(mv) = resolve(&b, *um) {
                            let _ = b.make_move(mv); // discard Undo: GUI moves aren't unmade
                        }
                    }
                    board = b;
                }
            }
            UciCommand::Go(go) => {
                if let Some(depth) = go.perft {
                    let mut total = 0;
                    for (mv, n) in board.perft_divide(depth) {
                        uci.send(&format!("{mv}: {n}")).map_err(io_err)?;
                        total += n;
                    }
                    uci.send(&format!("\nNodes searched: {total}"))
                        .map_err(io_err)?;
                } else {
                    // No search yet - emit a null move so a GUI doesn't hang.
                    uci.send("bestmove 0000").map_err(io_err)?;
                }
            }
            UciCommand::Stop => {} // nothing to stop while single-threaded
            UciCommand::Quit => break,
            UciCommand::Unknown(_) => {} // UCI: ignore unrecognized input
        }
    }
    Ok(())
}

/// Turn a UCI from/to(/promo) into a flagged engine [`Move`] by matching it
/// against generated moves, assuming UCI sends legal moves, as legality is
/// not checked
fn resolve(board: &Board, um: UciMove) -> Option<Move> {
    board
        .pseudo_legal_moves()
        .into_iter()
        .find(|m| m.src() == um.from && m.dest() == um.to && m.flag().promoted_piece() == um.promo)
}

fn io_err(e: lattice_uci::UciError) -> io::Error {
    match e {
        lattice_uci::UciError::Io(e) => e,
    }
}
