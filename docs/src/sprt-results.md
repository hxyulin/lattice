# SPRT results

Every feature in Lattice is tested for playing strength with an
[SPRT](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test)
(Sequential Probability Ratio Test) on [OpenBench]. This page records what each
one was worth in Elo. The node-count side of the story - what each feature did
to the search tree - lives in [`BENCH.md`](https://github.com/hxyulin/lattice/blob/main/BENCH.md)
and the per-commit `bench.csv`; this page is the strength side.

[OpenBench]: https://github.com/AndyGrant/OpenBench

## Method

Each feature is one commit on a linear history, so a test isolates it by setting
**Base = the feature's parent, Dev = the feature commit**. OpenBench reads the
`Bench: <nodes>` trailer on each commit to confirm both sides built the same
source.

- **Confidence:** `alpha = beta = 0.05`. The SPRT stops when the log-likelihood
  ratio (LLR) crosses `+2.94` (accept) or `-2.94` (reject).
- **Bounds:**
  - *Gainers* (expected Elo): `[0.00, 5.00]`.
  - *Efficiency* features that scale with depth are also re-run at long time
    control with `[0.00, 4.00]`.
  - *Non-regression* (behaviour-preserving or pure infrastructure): `[-5.00,
    0.00]` - accept if it is not a real regression.
- **Time controls:** STC `8.0+0.08`, LTC `40.0+0.40`.
- **Hardware:** Ryzen 9950X3D, 32 concurrent single-threaded games.

Elo below is the logistic estimate `-400 * log10(1 / score - 1)` from the final
W/L/D, rounded. It is a point estimate; the SPRT verdict (pass/fail) is the
authoritative result.

## Results

| Feature | TC | Games | Elo | Verdict |
|---------|----|------:|----:|---------|
| Principal variation search | STC | 368 | +286 | pass |
| Piece-square tables | STC | 4096 | +240 | pass |
| Quiescence search | STC | 648 | +256 | pass |
| Tapered PST (PeSTO) | STC | 540 | +126 | pass |
| Null-move pruning | STC / LTC | 680 / 594 | +84 / +100 | pass (scales) |
| Transposition table | STC / LTC | 1252 / 920 | +42 / +62 | pass (scales) |
| Late move reductions | STC / LTC | 1462 / 1240 | +37 / +42 | pass (scales) |
| Alpha-beta pruning | STC | 1112 | +35 * | pass |
| MVV-LVA ordering | STC | 3694 | +28 * | pass |
| Iterative deepening | STC | 4096 | +11 | pass (marginal) |
| Killer moves | STC | 7928 | +5 | pass |
| Magic bitboards | STC | 16384 | 0.0 | non-regression pass |
| Frontier ordering skip | STC | 15406 | 0.0 | non-regression pass |
| Bitboard material eval | STC | 12288 | 0.0 | non-regression (no regression) |
| Incremental Zobrist | STC | 12792 | -1.6 | fail (expected) |
| Butterfly history | STC | 564 | inconclusive | stopped early |

\* Inflated by time-forfeits, see "The fixed-depth caveat" below.

## What the numbers say

### Efficiency features scale with depth

Null-move pruning, the transposition table, and late move reductions all gained
**more at LTC than at STC** (NMP +84 -> +100, TT +42 -> +62, LMR +37 -> +42).
This is the whole reason these are tested at equal time rather than equal depth:
their benefit is searching deeper per second, so it grows when there is more
depth to reach. An equal-depth test would have scored them near zero.

### The biggest wins change either the move or the depth reached

A positional evaluation term (PST +240, tapered PeSTO +126), resolving captures
at the leaves (quiescence +256), and reaching far more depth per second
(principal variation search +286) dominate. The first three change which move
the engine picks, so they show up immediately and large. PVS is different: it is
behaviour-preserving (identical move at a fixed depth, verified score-exact), so
its number is pure speed converted to depth - which is why it is tested at time
control, not fixed depth. Its outsized +286 is partly a repair: the pre-PVS
search was ~99% quiescence nodes from an unordered frontier, and the null-window
scouts collapsed that (a 13.7x node drop at depth 4 - see `BENCH.md`), so it
bought ~2-3 extra plies at once rather than the textbook fraction of a ply.

### Behaviour-preserving features prove neutral

Magic bitboards, the bitboard material count, and skipping move ordering at the
frontier each returned an *exact* `0, 0, N, 0, 0` pentanomial - every game pair
scored precisely 1.0, i.e. byte-identical play. That is the signature of a
change that does not alter the move chosen, only the speed of finding it. Their
value is the NPS recorded in `bench.csv`, not an Elo line. (They cannot show
strength here at all: they predate the engine's time management - see below.)

### Zobrist and the transposition table are a pair

Incremental Zobrist hashing **failed** its non-regression test (-1.6 Elo). That
is correct: on its own it is pure overhead - a hash maintained on every move
with nothing yet reading it. The very next commit, the transposition table,
spends that hash for +42 (STC) / +62 (LTC). Read the two together: the
infrastructure commit pays a small cost that the feature commit more than earns
back.

### The fixed-depth caveat

Time management (parsing `go wtime/btime` and budgeting search time) only arrived
in commit `f7ac90b`. Every feature before it searches a fixed depth and ignores
the clock, which has two consequences:

- **Behaviour-preserving speedups read as 0 Elo** (magic, frontier skip,
  bitboard eval) - identical play at a fixed depth.
- **Alpha-beta (+35) and MVV-LVA (+28) are inflated by time-forfeits.** Both are
  exact at a fixed depth - they pick the same move as their base - so their only
  edge is that the faster (pruned) build avoids flagging on the clock. The Elo
  is real on the scoreboard but is not a playing-strength gain.

Piece-square tables (+240) is the exception among the early features: it changes
the evaluation, so it changes the move even at a fixed depth, and its number is
trustworthy.

## Pending

Butterfly history was stopped at 564 games (LLR -1.60), so it is **inconclusive**
- its point estimate is negative but the error bar is wide. Quiet-move history on
top of killer moves often gives little at STC. It should be re-run to completion,
and at LTC, before a verdict; a persistently negative result would point at a bug
in the history update rather than a weak heuristic.

## Summary

The confirmed, SPRT-verified gains - piece-square tables, quiescence, null-move
pruning, the transposition table, late move reductions, the tapered PeSTO eval,
and principal variation search - add up to roughly **+900 Elo** of real playing
strength (a loose sum: each is measured against the build before it, and Elo is
not strictly additive). The search-efficiency features each scale at long time
control, and PVS is the single biggest line so far. Everything behaviour-
preserving measured neutral, and Zobrist's expected dip is repaid by the table
that uses it.
