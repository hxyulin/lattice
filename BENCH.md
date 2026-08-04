# Search benchmark

`lattice bench` searches a fixed 12-position suite to depth 4 and prints a
ChessEval-compatible signature:

```
Nodes searched: <nodes>
Nodes/second: <nps>
```

The node count is deterministic and machine-independent - a fingerprint of the
search, not a speed measurement (that is the `nps`). To compare versions, re-run
the same `bench <depth>` and diff the node counts.

Per-commit totals live in `bench.csv` (generated from the `Bench:` commit
trailers by `tools/gen-ledgers.sh`); per-feature Elo lives in
[`docs/src/sprt-results.md`](docs/src/sprt-results.md).

## Baselines

One row per notable search change: commit, machine, depth, per-position nodes
and nps, and the total.
