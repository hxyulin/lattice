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
  - *Gainers* (expected Elo): `[0, 5]`.
  - *Efficiency* features that scale with depth are also re-run at long time
    control.
  - *Non-regression* (behaviour-preserving or pure infrastructure): `[-5, 0]` -
    accept if it is not a real regression.
- **Time controls:** STC `10+0.1`, LTC `60+0.6`. Both hold the increment at
  1/100 of the base, so LTC changes only the depth reached, not the
  time-management regime.

Each result is recorded as an `SPRT:` trailer on the feature commit; this table
is generated from those trailers by `tools/gen-ledgers.sh`. Elo is the logistic
point estimate from the final W/L/D; the SPRT verdict (pass/fail) is the
authoritative result.

## Results

<!-- BEGIN generated: tools/gen-ledgers.sh - do not edit this table by hand -->
| Feature | TC | Games | Elo | Verdict |
|---------|----|------:|----:|---------|
| alpha-beta pruning | STC | 128 | +339.5 | pass |
| MVV-LVA move ordering | STC | 160 | +186.7 | pass |
| quiescence search | STC | 216 | +135.4 | pass |
<!-- END generated -->
