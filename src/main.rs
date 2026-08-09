//! Lattice UCI binary.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use lattice::Board;
use lattice::bench;
use lattice::movegen::perft::perft_divide;
use lattice::search::{SearchOptions, UCI_SPIN_OPTIONS, search_with_options};
use lattice::tactics;
use lattice::tt::TranspositionTable;
use lattice::uci::{Command, UciListener, move_text, parse};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => {
            bench::run(&mut io::stdout().lock());
            return;
        }
        Some("tactics") => {
            std::process::exit(run_tactics(&args[1..]));
        }
        _ => {}
    }

    let stdin = io::stdin();
    let mut board = Board::startpos();
    let stop = Arc::new(AtomicBool::new(false));
    // Outlives each `go` so one move's tree seeds the next.
    let mut tt = Arc::new(TranspositionTable::new());
    let default_hash_mb = tt.size_mb();
    let mut search_options = SearchOptions::default();
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
                let tuning_options = UCI_SPIN_OPTIONS
                    .iter()
                    .map(|option| {
                        format!(
                            "option name {} type spin default {} min {} max {}",
                            option.name, option.default, option.min, option.max
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                write_protocol(&format!(
                    "id name Lattice\nid author hxyulin\n\
                     option name Hash type spin default {default_hash_mb} min 1 max 4096\n\
                     option name Threads type spin default 1 min 1 max 1\n\
                     {tuning_options}\n\
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
            Command::Position(position) => board = *position,
            Command::Go(limits) if search_thread.is_none() => {
                stop.store(false, Ordering::Relaxed);
                let mut search_board = board.clone();
                let search_stop = Arc::clone(&stop);
                let search_tt = Arc::clone(&tt);
                let search_options = search_options.clone();
                search_thread = Some(thread::spawn(move || {
                    let stdout = io::stdout();
                    let mut listener = UciListener::new(stdout.lock());
                    search_with_options(
                        &mut search_board,
                        limits,
                        &search_stop,
                        &search_tt,
                        &search_options,
                        &mut listener,
                    );
                    let _ = io::stdout().flush();
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
                    if let Some(mb) = value
                        .as_deref()
                        .and_then(|v| v.trim().parse::<usize>().ok())
                    {
                        tt = Arc::new(TranspositionTable::with_size_mb(mb.clamp(1, 4096)));
                    }
                }
                "threads"
                    if value
                        .as_deref()
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .is_some_and(|n| n > 1) =>
                {
                    write_protocol(
                        "info string Threads > 1 is not supported; searching with 1 thread",
                    );
                }
                _ if search_thread.is_none() => {
                    match search_options.set_spin(&name, value.as_deref().unwrap_or("")) {
                        Ok(true) => tt.clear(),
                        Ok(false) => {}
                        Err(error) => write_protocol(&format!("info string ignored {error}")),
                    }
                }
                _ => {}
            },
        }
    }
    stop.store(true, Ordering::Relaxed);
    join_search(&mut search_thread);
}

/// `tactics [suite] [--depth N]`. The suite is a path, or a bare name looked
/// up in `$SUITES_DIR` (default `~/dev/chess-data/test-suites`), so the large
/// EPD files stay out of the repository. Returns the process exit code.
fn run_tactics(args: &[String]) -> i32 {
    let mut suite = "wac".to_owned();
    let mut depth = 10;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--depth" => match rest.next().and_then(|d| d.parse().ok()) {
                Some(value) => depth = value,
                None => {
                    eprintln!("tactics: --depth needs a number");
                    return 2;
                }
            },
            other if other.starts_with('-') => {
                eprintln!("tactics: unknown option {other}");
                return 2;
            }
            other => suite = other.to_owned(),
        }
    }
    let path = resolve_suite(&suite);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("tactics: {}: {error}", path.display());
            return 2;
        }
    };
    let name = path
        .file_stem()
        .map_or_else(|| suite.clone(), |stem| stem.to_string_lossy().into_owned());
    let outcomes = tactics::run(&mut io::stdout().lock(), &name, &text, depth);
    // Reporting is the job; a regression is judged by diffing two runs, not
    // by this exit code. Only a suite that yielded nothing is an error.
    i32::from(outcomes.is_empty()) * 2
}

fn resolve_suite(suite: &str) -> std::path::PathBuf {
    let direct = std::path::Path::new(suite);
    if direct.is_file() {
        return direct.to_path_buf();
    }
    let dir = std::env::var("SUITES_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/dev/chess-data/test-suites")
    });
    std::path::Path::new(&dir).join(format!("{suite}.epd"))
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
