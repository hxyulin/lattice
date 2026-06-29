# Search benchmark

Run with `lattice bench [depth]` (default depth 4). The suite is the six
canonical perft positions.

To compare a new version, re-run the same `bench <depth>` and diff the node
counts: pruning (alpha-beta, etc.) should cut nodes hard for the same positions.

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

## Alpha-beta pruning (no move ordering)

- Machine: Apple M3 Pro, `rustc 1.96.0`, `--release`
- Depth: 4

| position  | nodes      | vs baseline | nps (M3 Pro) |
|-----------|-----------:|------------:|-------------:|
| startpos  |     17,181 |       12.0x |   13,297,987 |
| kiwipete  |    319,587 |       13.1x |   11,625,572 |
| endgame   |      7,664 |        6.0x |   11,736,600 |
| position4 |     76,014 |        5.7x |   11,370,830 |
| position5 |    304,895 |        7.1x |   11,653,225 |
| position6 |    629,831 |        6.3x |   12,211,706 |
| **total** | **1,355,172** | **8.1x** | **11,901,985** |

8.1x fewer nodes at the same depth, NPS essentially flat (the small dip is the
alpha/beta window check per node). Next lever is move ordering (MVV-LVA): with the best
move searched first, cutoffs fire sooner and root-window narrowing starts to
pay.
