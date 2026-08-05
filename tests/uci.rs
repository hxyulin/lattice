//! End-to-end UCI protocol sessions against the real binary.
//!
//! The unit tests in `src/uci.rs` cover parsing in isolation; nothing there
//! runs `main`, so the dispatch in `src/main.rs` - which command produces
//! which output, and whether the engine terminates - was entirely unverified.
//! These drive the shipped executable over stdin and stdout the way a GUI
//! does.

use std::io::Write;
use std::process::{Command, Output, Stdio};

/// Feeds `input` to the engine binary and returns its completed output.
///
/// The caller's script must end in `quit`, or the child never exits.
fn session(input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lattice"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("engine binary should start");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(input.as_bytes())
        .expect("engine should accept input");
    child.wait_with_output().expect("engine should exit")
}

fn stdout(input: &str) -> String {
    let out = session(input);
    assert!(out.status.success(), "engine exited with {}", out.status);
    assert!(
        out.stderr.is_empty(),
        "stdout is the protocol channel, but stderr had {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("protocol output is UTF-8")
}

/// catches: `uci` not answering `uciok`, `isready` not answering `readyok`,
/// the id lines dropped, and anything written to stderr instead of stdout. The
/// parse-level tests accept all of these because they never run `main`.
///
/// It does not catch `quit` failing to break the loop: `session` closes stdin,
/// so the engine reaches EOF and exits either way. `quit_exits_before_reading_
/// further_input` covers that separately.
#[test]
fn handshake_answers_uciok_and_readyok() {
    let text = stdout("uci\nisready\nquit\n");
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines.contains(&"uciok"), "no uciok in {lines:?}");
    assert!(lines.contains(&"readyok"), "no readyok in {lines:?}");
    assert!(
        lines.iter().any(|l| l.starts_with("id name ")),
        "no id name in {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.starts_with("id author ")),
        "no id author in {lines:?}"
    );
    // uciok must arrive before readyok, or a GUI that waits for the handshake
    // in order hangs.
    let uciok = lines.iter().position(|l| *l == "uciok").unwrap();
    let readyok = lines.iter().position(|l| *l == "readyok").unwrap();
    assert!(uciok < readyok, "uciok must precede readyok");
}

/// catches: a search producing no `bestmove`, producing one that is not legal
/// in the position it was given, and `position` being ignored so the engine
/// searches the start position instead. Legality is checked by replaying the
/// move through the parser, which only accepts a legal move, rather than by
/// hand-analysing the position. Every legal move here starts from e8, which no
/// move from the start position does, so a dropped `position` cannot pass.
#[test]
fn search_returns_a_legal_bestmove_for_the_position_given() {
    const FEN: &str = "4k3/8/8/8/8/8/8/4K2R b K - 0 1";
    let text = stdout(&format!("uci\nposition fen {FEN}\ngo depth 3\nquit\n"));
    let best = text
        .lines()
        .find_map(|l| l.strip_prefix("bestmove "))
        .map(|rest| rest.split_whitespace().next().unwrap_or("").to_owned())
        .unwrap_or_else(|| panic!("no bestmove in {text:?}"));
    assert!(
        best.starts_with("e8"),
        "bestmove {best} is not from the position sent"
    );
    assert!(
        lattice::uci::parse(&format!("position fen {FEN} moves {best}")).is_some(),
        "bestmove {best} is not legal in {FEN}"
    );
}

/// catches: `go depth` ignored so the engine searches forever, and `info`
/// lines never emitted. A depth-limited search must terminate on its own
/// without a `stop`.
#[test]
fn depth_limited_search_terminates_and_reports_info() {
    let text = stdout("uci\nposition startpos\ngo depth 4\nquit\n");
    assert!(
        text.lines().any(|l| l.starts_with("info ")),
        "no info lines in {text:?}"
    );
    assert!(
        text.lines().any(|l| l.starts_with("bestmove ")),
        "no bestmove in {text:?}"
    );
    let depths: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("info "))
        .filter_map(|l| {
            l.split_whitespace()
                .nth(1)
                .filter(|w| *w == "depth")
                .and(Some(l))
        })
        .collect();
    assert!(!depths.is_empty(), "info lines carry no depth: {text:?}");
}

/// catches: `position startpos moves ...` not applying the moves, which makes
/// the engine analyse the wrong position for the whole game. Perft after two
/// moves differs from perft at the start position, so the count detects it.
#[test]
fn position_moves_are_applied_before_the_next_command() {
    let text = stdout("uci\nposition startpos moves e2e4 e7e5\nperft 1\nquit\n");
    let total: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("Nodes searched: "))
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| panic!("no perft total in {text:?}"));
    // 29 legal moves after 1.e4 e5, against 20 at the start position.
    assert_eq!(total, 29, "moves were not applied: {text:?}");
}

/// catches: `bench` producing no output, or the verifier line ChessEval parses
/// being renamed. Invoked as an argv subcommand, which is the form the
/// tooling uses and which no other test exercises.
#[test]
fn bench_subcommand_prints_the_verifier_line() {
    let out = Command::new(env!("CARGO_BIN_EXE_lattice"))
        .arg("bench")
        .output()
        .expect("engine binary should start");
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).expect("bench output is UTF-8");
    let nodes: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("Nodes searched: "))
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| panic!("no verifier line in {text:?}"));
    assert!(nodes > 0);
}

/// catches: `quit` not breaking the command loop. Every other test here closes
/// stdin, so the engine exits at EOF whether or not `quit` is handled; only
/// input that continues past `quit` can tell the difference.
#[test]
fn quit_exits_before_reading_further_input() {
    let text = stdout("uci\nquit\nisready\n");
    assert!(
        !text.lines().any(|l| l == "readyok"),
        "engine kept reading after quit: {text:?}"
    );
}

/// catches: an unrecognised or malformed command killing the session. A GUI
/// sends `setoption` and vendor extensions the engine does not implement, and
/// the protocol requires ignoring them rather than exiting.
#[test]
fn unknown_and_malformed_input_does_not_end_the_session() {
    let text = stdout(concat!(
        "uci\n",
        "setoption name Hash value 64\n",
        "vendor nonsense here\n",
        "position fen garbage\n",
        "position\n",
        "go depth\n",
        "\n",
        "isready\n",
        "quit\n",
    ));
    assert!(
        text.lines().any(|l| l == "readyok"),
        "session died before isready: {text:?}"
    );
}
