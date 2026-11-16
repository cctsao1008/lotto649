# Issue #8 — Matched covariance walk-forward

The primary endpoint is the difference in the number of holdout draws
with at least one eight-tier award, Exact minus Matched MC. Those arms
share exact standardization; the legacy MC arm retains its original scaling.
The history-blind disjoint strategy from the v0.26 A/B comparison is also
scored as a descriptive baseline. All portfolios must match the exact
official-prize coverage of four disjoint tickets in every fold.

The chronology, settings and seeds reuse the v0.26 A/B walk-forward.
Use the 2,497-draw source through draw 115000103. The first training prefix
has 1,800 draws. For each of eight folds, select using only the expanding
history before the next 25 draws, then freeze that selection for the block.
The common candidate pool has 300 tickets, seed 649, ridge 0.25 and the
existing bounded search. The Monte Carlo covariance uses 8,192 independent
tickets, seed 649, and the 163-feature population covariance is pinned by
the v0.25 census. Use the six mains and special ball for retrospective
award scoring. No historical outcome contributes to the covariance matrices.

Generate 99 independently seeded fair-IID histories (2650–2748) with
six main balls and one distinct special ball, and repeat all eight folds
on each. Each replicate uses the same synthetic history across all arms.
Use the add-one upper-tail p-value of the integer primary contrast.
The positive direction requires a strictly positive gain and p <= 0.05.
Exploratory effect sizes and the central null range are reported even when
that criterion is not met. The finite sample and prior use of the historical
windows limit inference to this retrospective replay.

```bash
OPENBLAS_NUM_THREADS=1 python -m lotto7x7.matched_covariance_walkforward_v025
```

## Results — 2026-11-16

The 200 historical holdout draws ran from 2020-11-13 (109000099) through
2022-08-05 (111000072). The source cutoff includes 2,497 draws; the first
2,000 contain all training prefixes and holdouts used here.

| Arm | Draws with at least one award | Rate |
| --- | ---: | ---: |
| History-blind disjoint coverage | 22 | 11.0% |
| Legacy MC | 25 | 12.5% |
| Matched MC | 25 | 12.5% |
| Exact | 22 | 11.0% |

The four arms have the same theoretical any-award rate of 12.2731% under
fair IID. Exact and Matched MC selected identical tickets in seven of the
eight historical folds. Their paired counts were 2 Exact-only, 5 Matched
MC-only, 20 both, and 173 neither. All seven discordances came from the
final fold, which started on 2022-05-13.

The primary observed effect was -3 draws (-1.5 percentage points).
Of the 99 replays, 90 had an effect greater than or equal to -3,
giving an exploratory add-one upper-tail p of 0.91. The empirical
2.5th–97.5th percentiles of the IID effect were -7.55 and +8 draws.
For context, 34 replays had an absolute effect of at least 3 draws
(exploratory add-one two-sided p=0.35). The primary decision is
`NO_EVIDENCE` of an Exact advantage under this retrospective design.

See the [archived report and all 99 replay summaries](../data/research/v025_exact_covariance/v025_matched_covariance_walkforward_20261116.json).
