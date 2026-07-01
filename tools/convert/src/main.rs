//! Convert Lattice datagen text into bullet's `ChessBoard` binary format.
//!
//! Input: one position per line, `<FEN> | <score> | <result>`, score in
//! centipawns (white-relative), result `1.0`/`0.5`/`0.0` (white POV). This is
//! exactly bullet's text format, so `ChessBoard`'s own `FromStr` does the FEN
//! parsing and side-to-move normalisation for us.
//!
//! Usage: `convert <input.txt> <output.data> [--limit N]`
//!
//! Output records are the raw 32-byte `ChessBoard` structs back to back, which
//! is what bullet's `DirectSequentialDataLoader` reads.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::ExitCode;

use bulletformat::ChessBoard;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: convert <input.txt> <output.data> [--limit N]");
        return ExitCode::FAILURE;
    }
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(usize::MAX);

    let reader = BufReader::new(File::open(&args[1]).expect("open input"));
    let mut writer = BufWriter::new(File::create(&args[2]).expect("create output"));

    let (mut ok, mut bad) = (0usize, 0usize);
    for line in reader.lines().take(limit) {
        let line = line.expect("read line");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line.parse::<ChessBoard>() {
            Ok(board) => {
                // ChessBoard is repr(C), 32 bytes, POD: write its raw bytes.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        std::ptr::from_ref(&board).cast::<u8>(),
                        std::mem::size_of::<ChessBoard>(),
                    )
                };
                writer.write_all(bytes).expect("write record");
                ok += 1;
            }
            Err(_) => bad += 1,
        }
        if ok % 1_000_000 == 0 && ok > 0 {
            eprintln!("converted {ok} positions ...");
        }
    }
    writer.flush().expect("flush");
    eprintln!("done: {ok} positions written, {bad} skipped");
    ExitCode::SUCCESS
}
