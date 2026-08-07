//! Lattice UCI binary.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use lattice::Board;
use lattice::bench;
use lattice::movegen::perft::perft_divide;
use lattice::search::search;
use lattice::tt::TranspositionTable;
use lattice::uci::{Command, move_text, parse};

fn main() {
    if std::env::args().nth(1).as_deref() == Some("bench") {
        bench::run(&mut io::stdout().lock());
        return;
    }

    let stdin = io::stdin();
    let mut board = Board::startpos();
    let stop = Arc::new(AtomicBool::new(false));
    // Outlives each `go` so one move's tree seeds the next.
    let mut tt = Arc::new(TranspositionTable::new());
    let default_hash_mb = tt.size_mb();
    let mut search_thread: Option<JoinHandle<()>> = None;
    for line in stdin.lock().lines().map_while(Result::ok) {
        reap_finished(&mut search_thread);
        let Some(command) = parse(&line) else {
            continue;
        };
        // ponytail: keep UCI dispatch flat while the protocol surface is small.
        match command {
            Command::Uci => {
                // Off the clock: building these lazily inside the first search
                // charges roughly 150ms to that move. The handshake has seconds.
                lattice::movegen::init();
                write_protocol(&format!(
                    "id name Lattice\nid author hxyulin\n\
                     option name Hash type spin default {default_hash_mb} min 1 max 4096\n\
                     option name Threads type spin default 1 min 1 max 1\n\
                     uciok"
                ));
            }
            Command::IsReady => write_protocol("readyok"),
            Command::NewGame => {
                board = Board::startpos();
                // Scores from the previous game describe positions this one
                // may reach by another path; carrying them over is wrong.
                tt.clear();
            }
            Command::Position(position) => board = position,
            Command::Go(limits) if search_thread.is_none() => {
                stop.store(false, Ordering::Relaxed);
                let mut search_board = board.clone();
                let search_stop = Arc::clone(&stop);
                let search_tt = Arc::clone(&tt);
                search_thread = Some(thread::spawn(move || {
                    let stdout = io::stdout();
                    let mut output = stdout.lock();
                    search(
                        &mut search_board,
                        limits,
                        &search_stop,
                        &search_tt,
                        &mut output,
                    );
                    let _ = output.flush();
                }));
            }
            Command::Go(_) => {}
            Command::Stop => stop.store(true, Ordering::Relaxed),
            Command::Quit => {
                stop.store(true, Ordering::Relaxed);
                join_search(&mut search_thread);
                break;
            }
            Command::Perft(depth) => {
                let divide = perft_divide(&mut board, depth);
                let total: u64 = divide.iter().map(|(_, nodes)| nodes).sum();
                let stdout = io::stdout();
                let mut output = stdout.lock();
                for (mv, nodes) in divide {
                    let _ = writeln!(output, "{}: {nodes}", move_text(mv));
                }
                let _ = writeln!(output, "\nNodes searched: {total}");
                let _ = output.flush();
            }
            Command::Bench if search_thread.is_none() => {
                let stdout = io::stdout();
                let mut output = stdout.lock();
                bench::run(&mut output);
                let _ = output.flush();
            }
            Command::Bench => {}
            // Unknown options are ignored: GUIs send options engines lack.
            Command::SetOption { name, value } => match name.to_ascii_lowercase().as_str() {
                // Replacing the Arc under a running search would leave that
                // search on the old table; UCI only sends options when idle.
                "hash" if search_thread.is_none() => {
                    if let Some(mb) = value.and_then(|v| v.trim().parse::<usize>().ok()) {
                        tt = Arc::new(TranspositionTable::with_size_mb(mb.clamp(1, 4096)));
                    }
                }
                "threads"
                    if value
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .is_some_and(|n| n > 1) =>
                {
                    write_protocol(
                        "info string Threads > 1 is not supported; searching with 1 thread",
                    );
                }
                _ => {}
            },
        }
    }
    stop.store(true, Ordering::Relaxed);
    join_search(&mut search_thread);
}

fn write_protocol(text: &str) {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let _ = writeln!(output, "{text}");
    let _ = output.flush();
}

fn reap_finished(handle: &mut Option<JoinHandle<()>>) {
    if handle.as_ref().is_some_and(JoinHandle::is_finished) {
        join_search(handle);
    }
}

fn join_search(handle: &mut Option<JoinHandle<()>>) {
    if let Some(thread) = handle.take() {
        let _ = thread.join();
    }
}
