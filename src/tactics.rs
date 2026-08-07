//! Tactical test-suite runner.
//!
//! Searches each position of an EPD suite to a fixed depth and reports how
//! many the engine solves, plus how much of the tree it took to get there.
//!
//! Both figures come from one search rather than two. Iterative deepening
//! already reports a best move per depth, so recording which iteration first
//! agreed with the expected move gives the solve verdict and a cost-to-solve
//! together, and the cost is a node count rather than a clock reading - it
//! compares across machines the way `bench` does.
//!
//! What this measures is not Elo. A suite is a fixed set of sharp positions
//! and rewards finding one move over playing well, so read it as a signal
//! about search and ordering before an SPRT, not as a substitute for one.

use std::fmt::Write as _;
use std::io::Write;
use std::sync::atomic::AtomicBool;

use crate::search::{Iteration, Limits, SearchListener, search_inner};
use crate::tt::TranspositionTable;
use crate::uci::parse_san;
use crate::{Board, Move};

/// One EPD position and the moves that count as solving it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    /// Board as given by the EPD's first four fields.
    pub board: Board,
    /// Moves that solve the position, from a `bm` operation.
    pub best: Vec<String>,
    /// Moves that fail it, from an `am` operation.
    pub avoid: Vec<String>,
    /// The `id` operation, or the line number when absent.
    pub id: String,
}

/// How one position turned out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// The position's `id`.
    pub id: String,
    /// Whether the final iteration answered correctly.
    pub solved: bool,
    /// First depth whose best move was correct, if any reached one.
    pub first_depth: Option<u32>,
    /// Nodes searched when that depth completed.
    pub first_nodes: Option<u64>,
    /// Total nodes spent on the position.
    pub nodes: u64,
    /// Solved at some depth, then lost again by the last one. A search that
    /// finds the move and talks itself out of it is a different failure from
    /// never seeing it, and only this column tells them apart.
    pub unstable: bool,
}

/// Parses an EPD line into a position, or `None` if it has no `bm`/`am` to
/// score against. `line_number` names the position when it carries no `id`.
///
/// EPD writes only the first four FEN fields, so the clocks are supplied here;
/// they do not affect a fixed-depth search.
pub fn parse_epd(line: &str, line_number: usize) -> Option<Position> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let fields: Vec<&str> = line.split_whitespace().take(4).collect();
    if fields.len() < 4 {
        return None;
    }
    let board: Board = format!("{} 0 1", fields.join(" ")).parse().ok()?;
    // The operations follow the board and are separated by `;`. Skipping the
    // board by token count rather than by searching for the first `;` keeps a
    // semicolon inside a quoted comment from splitting the line early.
    let rest = line
        .match_indices(char::is_whitespace)
        .nth(3)
        .map_or("", |(at, _)| &line[at..]);
    let mut best = Vec::new();
    let mut avoid = Vec::new();
    let mut id = None;
    for op in rest.split(';') {
        let op = op.trim();
        if let Some(moves) = op.strip_prefix("bm ") {
            best = moves.split_whitespace().map(str::to_owned).collect();
        } else if let Some(moves) = op.strip_prefix("am ") {
            avoid = moves.split_whitespace().map(str::to_owned).collect();
        } else if let Some(text) = op.strip_prefix("id ") {
            id = Some(text.trim().trim_matches('"').to_owned());
        }
    }
    if best.is_empty() && avoid.is_empty() {
        return None;
    }
    Some(Position {
        board,
        best,
        avoid,
        id: id.unwrap_or_else(|| format!("line {line_number}")),
    })
}

/// Records the first iteration whose best move solved the position, and
/// whether the last one still did.
struct Solve<'a> {
    position: &'a Position,
    board: Board,
    first_depth: Option<u32>,
    first_nodes: Option<u64>,
    last_correct: bool,
}

impl SearchListener for Solve<'_> {
    fn iteration(&mut self, iteration: &Iteration) {
        let correct = iteration
            .best_move
            .is_some_and(|mv| self.position.accepts(&self.board, mv));
        self.last_correct = correct;
        if correct && self.first_depth.is_none() {
            self.first_depth = Some(iteration.depth);
            self.first_nodes = Some(iteration.nodes);
        }
    }
}

impl Position {
    /// Whether `mv` solves this position: among the `bm` moves if it lists
    /// any, and not among the `am` moves.
    fn accepts(&self, board: &Board, mv: Move) -> bool {
        let matches = |list: &[String]| {
            list.iter()
                .any(|san| parse_san(board, san).is_some_and(|want| want == mv))
        };
        if !self.avoid.is_empty() && matches(&self.avoid) {
            return false;
        }
        self.best.is_empty() || matches(&self.best)
    }
}

/// Searches one position to `depth` and reports how it went.
pub fn run_position(position: &Position, depth: u32) -> Outcome {
    let mut board = position.board.clone();
    let mut solve = Solve {
        position,
        board: position.board.clone(),
        first_depth: None,
        first_nodes: None,
        last_correct: false,
    };
    let result = search_inner(
        &mut board,
        Limits {
            depth: Some(depth),
            infinite: true,
            ..Limits::default()
        },
        &AtomicBool::new(false),
        // A table per position: a shared one would let an earlier position's
        // tree change a later one's node count, which would make the report
        // depend on suite order.
        &TranspositionTable::new(),
        &mut solve,
    );
    Outcome {
        id: position.id.clone(),
        solved: solve.last_correct,
        first_depth: solve.first_depth,
        first_nodes: solve.first_nodes,
        nodes: result.nodes + result.qnodes,
        unstable: solve.first_depth.is_some() && !solve.last_correct,
    }
}

/// Runs every position of a suite and writes the report.
///
/// Returns the outcomes so a caller can compare two runs itself.
pub fn run(output: &mut dyn Write, suite: &str, text: &str, depth: u32) -> Vec<Outcome> {
    crate::movegen::init();
    let positions: Vec<Position> = text
        .lines()
        .enumerate()
        .filter_map(|(index, line)| parse_epd(line, index + 1))
        .collect();
    let outcomes: Vec<Outcome> = positions
        .iter()
        .map(|position| run_position(position, depth))
        .collect();
    let _ = write!(output, "{}", report(suite, depth, &outcomes));
    outcomes
}

/// Formats the report for a finished run.
pub fn report(suite: &str, depth: u32, outcomes: &[Outcome]) -> String {
    let total = outcomes.len();
    let solved: Vec<&Outcome> = outcomes.iter().filter(|o| o.solved).collect();
    let mut text = String::new();
    let _ = writeln!(text, "{suite}  {total} positions, depth {depth}");
    let _ = writeln!(text);
    let _ = writeln!(
        text,
        "solved {}/{total} ({:.1}%)",
        solved.len(),
        percent(solved.len(), total)
    );
    let nodes: u64 = outcomes.iter().map(|o| o.nodes).sum();
    let _ = writeln!(text, "nodes {nodes}");
    let mut to_solve: Vec<u64> = solved.iter().filter_map(|o| o.first_nodes).collect();
    to_solve.sort_unstable();
    if let Some(median) = median(&to_solve) {
        let _ = writeln!(
            text,
            "nodes to first solve: {} total, {median} median",
            to_solve.iter().sum::<u64>()
        );
    }
    let mut depths: Vec<u32> = solved.iter().filter_map(|o| o.first_depth).collect();
    depths.sort_unstable();
    if !depths.is_empty() {
        let _ = writeln!(
            text,
            "depth to first solve: {:.1} mean, {} max",
            depths.iter().map(|&d| f64::from(d)).sum::<f64>() / depths.len() as f64,
            depths.last().copied().unwrap_or(0),
        );
    }
    let unstable: Vec<&str> = outcomes
        .iter()
        .filter(|o| o.unstable)
        .map(|o| o.id.as_str())
        .collect();
    let _ = writeln!(text, "unstable (found then lost): {}", unstable.len());
    if !unstable.is_empty() {
        let _ = writeln!(text, "  {}", unstable.join(" "));
    }
    let failed: Vec<&str> = outcomes
        .iter()
        .filter(|o| !o.solved)
        .map(|o| o.id.as_str())
        .collect();
    if !failed.is_empty() {
        let _ = writeln!(text, "failed:");
        for chunk in failed.chunks(8) {
            let _ = writeln!(text, "  {}", chunk.join(" "));
        }
    }
    text
}

fn median(sorted: &[u64]) -> Option<u64> {
    match sorted.len() {
        0 => None,
        n if n % 2 == 1 => Some(sorted[n / 2]),
        n => Some(sorted[n / 2 - 1].midpoint(sorted[n / 2])),
    }
}

fn percent(count: usize, total: usize) -> f64 {
    count as f64 * 100.0 / total.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    const WAC_1: &str =
        "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; id \"WAC.001\";";

    #[test]
    fn parses_the_epd_shapes_the_suites_use() {
        let position = parse_epd(WAC_1, 1).expect("a bm line parses");
        assert_eq!(position.best, ["Qg6"]);
        assert_eq!(position.id, "WAC.001");
        assert!(position.avoid.is_empty());
        // EPD omits the clocks, so the board must still parse.
        assert!(position.board.to_string().starts_with("2rr3k/"));

        // Several accepted moves on one line, which wac and arasan both use.
        let multi = parse_epd(
            "r1bqk2r/ppp1nppp/4p3/n5N1/2BPp3/P1P5/2P2PPP/R1BQK2R w KQkq - bm Ba2 Nxf7; id \"WAC.022\";",
            1,
        )
        .unwrap();
        assert_eq!(multi.best, ["Ba2", "Nxf7"]);

        // `am` lines, with several moves to avoid.
        let avoid = parse_epd(
            "6k1/5p2/P3p1p1/2Qp4/5q2/2K5/8/8 b - - am Qc1+ Qe5+; id \"E_E_T 011\";",
            1,
        )
        .unwrap();
        assert_eq!(avoid.avoid, ["Qc1+", "Qe5+"]);
        assert!(avoid.best.is_empty());

        // A comment holding a semicolon must not cut the operations short.
        let commented = parse_epd(
            "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; c0 \"a; b\"; id \"X\";",
            1,
        )
        .unwrap();
        assert_eq!(commented.id, "X");
        assert_eq!(commented.best, ["Qg6"]);

        // Positions without a solution are skipped rather than scored as
        // failures, which would make a malformed suite look like a regression.
        assert!(parse_epd("", 1).is_none());
        assert!(parse_epd("# a comment", 1).is_none());
        assert!(parse_epd("8/8/8/8/8/8/8/K6k w - - ; id \"no bm\";", 1).is_none());
        // An id-less line falls back to its line number.
        let anonymous = parse_epd(
            "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6;",
            7,
        )
        .unwrap();
        assert_eq!(anonymous.id, "line 7");
    }

    // catches: `am` being scored as `bm`, which would invert every avoid-move
    // suite while still producing a plausible-looking solve count.
    #[test]
    fn avoid_moves_are_scored_the_other_way_round() {
        crate::movegen::init();
        let position = parse_epd(WAC_1, 1).unwrap();
        let board = position.board.clone();
        let qg6 = parse_san(&board, "Qg6").unwrap();
        let other = parse_san(&board, "Qg4").unwrap();
        assert!(position.accepts(&board, qg6));
        assert!(!position.accepts(&board, other));

        let avoiding = Position {
            best: Vec::new(),
            avoid: vec!["Qg6".to_owned()],
            ..position
        };
        assert!(!avoiding.accepts(&board, qg6));
        assert!(avoiding.accepts(&board, other));
    }

    #[test]
    fn solves_a_known_tactic_and_reports_where_it_first_saw_it() {
        crate::movegen::init();
        let position = parse_epd(WAC_1, 1).unwrap();
        let outcome = run_position(&position, 6);
        assert!(outcome.solved, "WAC.001 is a mate in 3 and must be found");
        assert!(outcome.first_depth.is_some());
        assert!(outcome.first_nodes.is_some_and(|n| n > 0));
        assert!(outcome.nodes >= outcome.first_nodes.unwrap());
        assert!(!outcome.unstable);
    }

    #[test]
    fn report_counts_what_it_says_it_counts() {
        let outcomes = vec![
            Outcome {
                id: "a".into(),
                solved: true,
                first_depth: Some(2),
                first_nodes: Some(100),
                nodes: 500,
                unstable: false,
            },
            Outcome {
                id: "b".into(),
                solved: false,
                first_depth: Some(3),
                first_nodes: Some(200),
                nodes: 700,
                unstable: true,
            },
            Outcome {
                id: "c".into(),
                solved: false,
                first_depth: None,
                first_nodes: None,
                nodes: 300,
                unstable: false,
            },
        ];
        let text = report("suite", 6, &outcomes);
        assert!(text.contains("solved 1/3 (33.3%)"), "{text}");
        assert!(text.contains("nodes 1500"), "{text}");
        assert!(text.contains("unstable (found then lost): 1"), "{text}");
        // Both unsolved positions are listed, the unstable one included.
        assert!(text.contains("b c"), "{text}");
    }

    #[test]
    fn median_handles_both_parities_and_empty() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[5]), Some(5));
        assert_eq!(median(&[1, 3]), Some(2));
        assert_eq!(median(&[1, 2, 3]), Some(2));
        // Large values must not overflow on the way to the midpoint.
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }
}
