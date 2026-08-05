//! End-to-end benchmark checks at the shipped depth.
//!
//! These run the real depth-4 suite, which is too slow for the unit tests.
//! The library-side tests in `src/bench.rs` cover the same output format at a
//! shallower depth; what is only checkable here is that the shipped `DEPTH`
//! constant is the one `run` uses, and that the count it prints is the count
//! it returns.

use lattice::bench;

/// catches: `run` searching at a depth other than the shipped one, and a
/// nondeterministic count. The move-ordering and transposition-table code is
/// shared state across the twelve positions, so an ordering bug that depends
/// on table contents shows up as a differing count between two runs.
#[test]
fn shipped_depth_is_reproducible() {
    let mut first = Vec::new();
    let mut second = Vec::new();
    let first_nodes = bench::run(&mut first);
    let second_nodes = bench::run(&mut second);
    assert_eq!(first_nodes, second_nodes);

    let text = String::from_utf8(first).unwrap();
    // Every line but the nps one, which is a speed reading and varies.
    let counts = |report: &str| {
        report
            .lines()
            .filter(|line| !line.starts_with("Nodes/second:"))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        counts(&String::from_utf8(second).unwrap()),
        counts(&text),
        "the node counts must be reproducible"
    );

    let reported: u64 = text
        .lines()
        .find_map(|line| line.strip_prefix("Nodes searched: "))
        .and_then(|n| n.parse().ok())
        .expect("bench must print the verifier line");
    assert_eq!(
        reported, first_nodes,
        "the printed count and the returned count must agree"
    );

    // Depth 4 over twelve positions is far more than a shallower search would
    // reach, so this fails if `run` stops using the shipped DEPTH.
    assert!(
        first_nodes > 100_000,
        "depth-4 suite searched only {first_nodes} nodes"
    );
}
