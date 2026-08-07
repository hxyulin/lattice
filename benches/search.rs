//! Wall-clock benchmarks for the fixed search.
//!
//! The `bench` subcommand already reports a deterministic node count, which is
//! what a search change is judged on. This measures the other half: how long
//! that same fixed tree takes to walk. Criterion is here because the changes
//! it is used for - data layout, allocation, instruction count - move
//! throughput by one or two percent, which is inside the run-to-run spread of
//! a single timed run on a machine doing anything else.

// `criterion_group!` and `criterion_main!` generate undocumented items.
#![allow(missing_docs)]

use std::io;

use criterion::{Criterion, criterion_group, criterion_main};

/// Shallower than `bench::DEPTH`, so a sample is a few tens of milliseconds
/// and Criterion can collect enough of them to say something. The tree shape
/// is the same one the full benchmark walks.
const DEPTH: u32 = 6;

fn fixed_search(c: &mut Criterion) {
    lattice::movegen::init();
    let mut group = c.benchmark_group("search");
    // Long enough to swamp the per-sample setup, short enough that the whole
    // suite stays under a minute.
    group.sample_size(100);
    group.bench_function("bench_positions", |b| {
        b.iter(|| lattice::bench::run_at(&mut io::sink(), DEPTH))
    });
    group.finish();
}

criterion_group!(benches, fixed_search);
criterion_main!(benches);
