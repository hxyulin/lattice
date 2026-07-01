# Search benchmark

Run with `lattice bench [depth]` (default depth 4). The suite is the six
canonical perft positions.

To compare a new version, re-run the same `bench <depth>` and diff the node
counts: pruning (alpha-beta, etc.) should cut nodes hard for the same positions.

Per-commit totals live in `bench.csv` (generated from the `Bench:` commit
trailers by `tools/gen-ledgers.sh`); per-feature Elo lives in
[`docs/src/sprt-results.md`](docs/src/sprt-results.md).

## Baseline - material eval + fixed-depth negamax, no pruning

- Commit: `undetermined` (pre-alpha-beta)
- Machine: Apple M3 Pro, `rustc 1.96.0`, `--release`
- Depth: 4

| position  | nodes      | nps (M3 Pro) |
|-----------|-----------:|-------------:|
| startpos  |    206,603 |   13,008,626 |
| kiwipete  |  4,185,552 |   12,130,944 |
| endgame   |     46,255 |   13,059,006 |
| position4 |    432,070 |   11,999,944 |
| position5 |  2,167,396 |   12,546,721 |
| position6 |  3,986,609 |   12,705,837 |
| **total** | **11,024,485** | **12,429,363** |

Wall time: ~0.89s.
