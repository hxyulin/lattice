//! Lattice UCI binary.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use lattice::Board;
use lattice::bench;
use lattice::movegen::perft::perft_divide;
use lattice::search::search;
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
        Some("tune") => {
            std::process::exit(run_tune(&args[1..]));
        }
        _ => {}
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
                    let mut listener = UciListener::new(stdout.lock());
                    search(
                        &mut search_board,
                        limits,
                        &search_stop,
                        &search_tt,
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

fn run_tune(args: &[String]) -> i32 {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        eprintln!(
            "usage: lattice tune SHARD... --output DIR [--epochs N] \
             [--learning-rate X] [--validation X] [--patience N] \
             [--regularization X] [--max-delta CP] [--seed N] [--threads N]"
        );
        return 0;
    }
    let mut shards = Vec::new();
    let mut output = None;
    let mut epochs = 200usize;
    let mut learning_rate = 1.0f64;
    let mut validation_fraction = 0.10f64;
    let mut patience = 15usize;
    let mut regularization = 0.01f64;
    let mut max_delta = 32.0f64;
    let mut seed = 1u64;
    let mut threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        let value = |name: &str, index: &mut usize| -> Result<&str, String> {
            *index += 1;
            args.get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("tune: {name} needs a value"))
        };
        let parsed = match arg.as_str() {
            "--output" => value(arg, &mut index).map(|v| output = Some(v.into())),
            "--epochs" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| epochs = n)
                    .map_err(|_| format!("tune: invalid epochs `{v}`"))
            }),
            "--learning-rate" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| learning_rate = n)
                    .map_err(|_| format!("tune: invalid learning rate `{v}`"))
            }),
            "--validation" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| validation_fraction = n)
                    .map_err(|_| format!("tune: invalid validation fraction `{v}`"))
            }),
            "--patience" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| patience = n)
                    .map_err(|_| format!("tune: invalid patience `{v}`"))
            }),
            "--regularization" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| regularization = n)
                    .map_err(|_| format!("tune: invalid regularization `{v}`"))
            }),
            "--max-delta" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| max_delta = n)
                    .map_err(|_| format!("tune: invalid max delta `{v}`"))
            }),
            "--seed" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| seed = n)
                    .map_err(|_| format!("tune: invalid seed `{v}`"))
            }),
            "--threads" => value(arg, &mut index).and_then(|v| {
                v.parse()
                    .map(|n| threads = n)
                    .map_err(|_| format!("tune: invalid threads `{v}`"))
            }),
            unknown if unknown.starts_with('-') => Err(format!("tune: unknown option {unknown}")),
            path => {
                shards.push(path.into());
                Ok(())
            }
        };
        if let Err(error) = parsed {
            eprintln!("{error}");
            return 2;
        }
        index += 1;
    }
    let Some(output) = output else {
        eprintln!("tune: --output DIR is required");
        return 2;
    };
    let config = lattice::tuner::TuneConfig {
        shards,
        output,
        epochs,
        learning_rate,
        validation_fraction,
        patience,
        regularization,
        max_delta,
        seed,
        threads,
    };
    match lattice::tuner::run(&config) {
        Ok(summary) => {
            println!(
                "tuned {} records / {} placements: K={:.8}, epoch {}, validation {:.10} -> {:.10} (continuous {:.10}); output {}",
                summary.records,
                summary.unique_placements,
                summary.k,
                summary.best_epoch,
                summary.baseline_validation_loss,
                summary.rounded_validation_loss,
                summary.best_validation_loss,
                summary.output.display()
            );
            0
        }
        Err(error) => {
            eprintln!("tune: {error}");
            2
        }
    }
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
