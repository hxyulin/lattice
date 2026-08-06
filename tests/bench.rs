//! End-to-end benchmark checks at the shipped depth.
//!
//! Release-only. The shipped suite is under a second optimized, but
//! `debug_check` re-derives the zobrist and the eval accumulator on every
//! make/unmake, which makes the same run roughly 60x slower in a debug build -
//! far too slow to sit in `cargo test`. The library-side tests in
//! `src/bench.rs` cover the output format and determinism at a shallow depth in
//! both profiles; what is only checkable here is that the shipped `DEPTH`
//! constant is the one `run` uses, and that the count it prints is the count it
//! returns.

/// catches: `run` searching at a depth other than the shipped one, and the
/// printed count drifting from the returned one. Determinism is depth
/// independent and covered by the depth-2 unit tests, so this runs the
/// expensive suite once.
#[test]
#[cfg_attr(debug_assertions, ignore = "too slow in debug; run with --release")]
fn shipped_depth_is_reproducible() {
    let mut report = Vec::new();
    let nodes = lattice::bench::run(&mut report);
    let text = String::from_utf8(report).unwrap();

    let reported: u64 = text
        .lines()
        .find_map(|line| line.strip_prefix("Nodes searched: "))
        .and_then(|n| n.parse().ok())
        .expect("bench must print the verifier line");
    assert_eq!(
        reported, nodes,
        "the printed count and the returned count must agree"
    );

    // The suite searches millions of nodes at the shipped depth and a small
    // fraction of that at any shallower one, so this fails if `run` stops
    // using the shipped DEPTH.
    assert!(
        nodes > 5_000_000,
        "shipped-depth suite searched only {nodes} nodes"
    );
}
