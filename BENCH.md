# Search benchmark

Run with `lattice bench [depth]` (default depth 4). The suite is the six
canonical perft positions.

To compare a new version, re-run the same `bench <depth>` and diff the node
counts: pruning (alpha-beta, etc.) should cut nodes hard for the same positions.

Per-commit totals live in `bench.csv` (generated from the `Bench:` commit
trailers by `tools/gen-benchcsv.sh`); the prose below walks the milestones.

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

## MVV-LVA capture ordering

- Machine: Apple M3 Pro, `rustc 1.96.0`, `--release`
- Depth: 4

| position  | nodes     | vs alpha-beta | vs baseline | nps (M3 Pro) |
|-----------|----------:|-------:|------------:|-------------:|
| startpos  |    12,476 |   1.4x |             |   12,576,612 |
| kiwipete  |    93,219 |   3.4x |             |    7,657,849 |
| endgame   |     3,819 |   2.0x |             |    8,902,097 |
| position4 |    12,506 |   6.1x |             |    7,174,985 |
| position5 |    61,410 |   5.0x |             |    8,573,223 |
| position6 |   100,567 |   6.3x |             |    9,463,348 |
| **total** | **283,997** | **4.8x** | **38.8x** | **8,572,200** |

4.8x fewer nodes than plain alpha-beta (38.8x under the unpruned baseline). NPS
drops ~28% (11.9M -> 8.6M) - the per-node cost of sorting (`sort_by_key` re-runs
the key during comparisons). Net wall-clock is still ~3.7x faster. If NPS
matters later: `sort_by_cached_key`, or score once and lazy-select the max per
iteration so an early cutoff skips the full sort.

## Frontier ordering guardrail

Ordering is skipped at the depth-1 frontier (`ORDER_MIN_DEPTH = 2`). That layer
holds the most nodes, but its children are leaves (static eval), so sorting it
costs more than the few sibling evals it saves. Sweep at depth 5:

| `ORDER_MIN_DEPTH` | what | nodes | nps | wall |
|------------------:|------|------:|----:|-----:|
| 1 | order everywhere       | 1,147,271 |  4.8M | 239ms |
| **2** | **skip depth-1 frontier** | 1,597,733 | 10.0M | **160ms** |
| 3 | skip depth <= 2         | 3,837,692 | 10.7M | 359ms |

Threshold 2 wins: ~2x NPS for a modest node increase -> ~1.5x faster wall-clock.
Threshold 3 explodes the node count - depth-2 subtrees are large enough that a
cutoff there saves an exponential amount, so ordering stays essential. The win
grows with depth (negligible at d4, clear at d5), so it matters more as search
deepens.

## Current baseline (depth 5 & 6)

The reference for future work, measured on the current engine
(alpha-beta + MVV-LVA + frontier guardrail). As the engine speeds up, depth 4
finishes too fast to be a useful signal - depth 5/6 are the comparison points
going forward. Node counts remain the deterministic signature; NPS is M3 Pro.

Depth 5 (~0.17s):

| position  | nodes     | nps (M3 Pro) |
|-----------|----------:|-------------:|
| startpos  |   236,809 |   12,227,448 |
| kiwipete  |   339,220 |    8,789,904 |
| endgame   |    22,502 |   10,886,308 |
| position4 |    39,265 |    8,788,048 |
| position5 |   287,237 |    9,408,660 |
| position6 |   672,700 |   10,683,373 |
| **total** | **1,597,733** | **10,112,746** |

Depth 6 (~1.6s):

| position  | nodes     | nps (M3 Pro) |
|-----------|----------:|-------------:|
| startpos  |   933,184 |   12,387,453 |
| kiwipete  | 3,905,909 |    9,325,272 |
| endgame   |    82,908 |   11,386,897 |
| position4 |   559,457 |   10,128,485 |
| position5 | 1,785,429 |    8,753,819 |
| position6 | 9,269,531 |   10,653,065 |
| **total** | **16,536,418** | **10,140,108** |

## NPS optimizations

Pure-speed work: each of these keeps the node counts above **byte-identical**
(the deterministic signature is the proof the change is behaviour-preserving)
and only moves NPS. Node totals stay D5 1,597,733 / D6 16,536,418 throughout.
NPS is M3 Pro, `--release`, and noisy run-to-run - read the trend, not the digit.

| change | D5 nps | D6 nps |
|--------|-------:|-------:|
| baseline (mailbox eval, ray-step sliders) | 10.1M | 10.1M |
| bitboard eval (10 popcounts, no mailbox walk) | 19.0M | 24.9M |
| in-place `generate_moves` (no return copy) | 19.0M | 24.5M (noise) |
| magic bitboards (O(1) sliders) | 28.4M | 32.6M |

The in-place move buffer was a **no-op on NPS** - the optimizer already elided
the 512-byte return copy (NRVO). 

## Quiescence search (depth 4)

Quiescence extends the search past the fixed horizon, resolving captures so the
static eval is only taken in quiet positions. It *raises* the node count - the
leaves now spawn a capture search - from 441,085 (iterative deepening) to
13,800,981 at depth 4. The payoff is tactical (no more horizon-effect blunders),
so it shows up as Elo, not as fewer nodes.

### Delta pruning (tried, then parked)

Delta pruning skips captures that cannot raise alpha even with a generous
margin. At depth 4 it cut the suite to 10,666,210 nodes:

| position  | nodes      | qnodes     | nps        | qnps       |
|-----------|-----------:|-----------:|-----------:|-----------:|
| startpos  |    115,960 |     57,628 | 26,420,596 | 13,130,097 |
| kiwipete  |  7,625,308 |  7,498,048 |  3,124,242 |  3,072,101 |
| endgame   |     21,296 |     12,752 | 17,023,181 | 10,193,445 |
| position4 |    383,061 |    366,403 |  3,757,857 |  3,594,441 |
| position5 |    548,382 |    450,045 |  4,836,587 |  3,969,280 |
| position6 |  1,972,203 |  1,832,558 |  4,229,590 |  3,930,107 |
| **total** | **10,666,210** | **10,217,434** | **3,409,982** | **3,266,508** |

It was then disabled (back to 13,800,981 nodes): at 3M+ nps the per-node delta
check cost more wall-clock than the nodes it saved. It is parked, not deleted -
re-enable it once a richer (slower) eval lowers NPS enough to flip the trade.
The parked quiescence TT carries the same per-node-overhead-vs-NPS lesson.

## Tapered PeSTO eval (depth 4)

Replacing the single untapered piece-square table with PeSTO's middlegame/
endgame tables (blended by a phase scalar) *raises* the depth-4 suite from
**10,034,609** (the LMR baseline) to **11,869,924** nodes - about +18%. That is
expected: a knowledge feature changes the scores, which reshuffles move ordering
and shifts the tree; it does not prune. The payoff is strength, not nodes
(+126 Elo STC - see `sprt-results.md`), and it is what makes the deeper search
PVS then buys actually worth having.

## Principal variation search (depth 4)

PVS searches only the first (best-ordered) move with the full window and scouts
the rest with a null window, re-searching wide only when a scout beats alpha.
Unlike the eval, this is **behaviour-preserving**: at every fixed depth the score
and PV are byte-identical to the pre-PVS engine (verified across the suite), so
the node drop is pure efficiency, not a different search.

- Machine: Apple M3 Pro, `--release`
- Depth: 4
- Base = the PeSTO-eval commit; Dev = PVS.

| position  | nodes (PeSTO) | nodes (PVS) | reduction |
|-----------|--------------:|------------:|----------:|
| startpos  |        14,430 |       4,654 |      3.1x |
| kiwipete  |     9,377,692 |     627,243 |     15.0x |
| endgame   |         9,410 |       1,721 |      5.5x |
| position4 |       380,661 |      54,694 |      7.0x |
| position5 |       504,532 |      71,743 |      7.0x |
| position6 |     1,583,199 |     104,265 |     15.2x |
| **total** | **11,869,924** | **864,320** | **13.7x** |

13.7x fewer nodes at the same depth - far above the textbook ~1.3-2x for PVS over
alpha-beta. The per-position spread tells why: the tactical positions (kiwipete,
position6) collapse ~15x while quiet startpos only 3x. The pre-PVS suite was ~99%
quiescence nodes, a frontier explosion - moves at the depth-1 frontier are
unordered (`ORDER_MIN_DEPTH = 2`), so full-window alpha-beta there fanned out
into a huge capture search, and the null-window scouts cut those subtrees off
cheaply. So PVS here is half the textbook feature and half a repair of pruning
the engine was leaving on the table; at equal time it now searches ~2-3 plies
deeper, worth +299 Elo STC. NPS is roughly flat (3.28M -> 3.57M) - the
scout/re-search bookkeeping is negligible per node.

## Per-commit node signatures

Node counts are deterministic - a build's fingerprint. Every engine commit
stamps its depth-4 suite total as a `Bench: <nodes>` trailer (the same number
OpenBench reads), and `bench.csv` is the committed snapshot of all of them:

    tools/gen-benchcsv.sh > bench.csv

A node-count change across a commit is a behaviour change; an unchanged count
across a perf/refactor commit is the proof it was behaviour-preserving (the
magic-bitboard and in-place-buffer commits both hold 318,497 at depth 4).

## Measuring per-feature Elo (OpenBench SPRT)

Bench nodes show *what* a feature did to the tree; they are not Elo. For that,
SPRT each feature on OpenBench:

- Base = the feature commit's parent, Dev = the feature commit. The linear
  history makes each commit exactly one feature, so the diff is clean.
- Node-reducing features (alpha-beta, MVV-LVA, ordering): a fixed-depth or
  equal-nodes test already shows the gain.
- Efficiency features (NPS perf, TT, null-move, LMR): test at equal *time*
  (e.g. `tc=8+0.08`), not equal depth/nodes - the win is searching deeper in
  the same clock, which an equal-depth test hides.

`make EXE=...` builds `target/release/lattice`; the `bench` final line
(`<nodes> nodes <nps> nps`) is the OpenBench entrypoint. Push to a GitHub remote
and tag each feature commit so the base/dev dropdowns read by name.
