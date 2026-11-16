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
