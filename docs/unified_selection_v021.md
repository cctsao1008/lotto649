# Lotto649 v0.21 — Unified Number Selection Engine

## Purpose

Unify the previously separate v0.6 candidate pool and exact fair threshold
coverage methods with v0.15/v0.16 fixed graph-category theory and v0.17 frozen
one-step class estimates. The current v0.19 study freeze is **read-only**;
**no new outcomes, training or prospective e-process updates** occur in this
selection command.

## Dependencies and numerical definitions

- `data`: P1 catalog (`draw_ids`, `numbers`, class labels), exact P1 universe
  (`13,983,816` legal six-number sets), and sealed 2026-11-08 prospective
  freeze. Reuse `check_freeze()` to reject changed bytes, hidden history edits,
  changed rule schema or inconsistent class counts.
- `pool`: for each candidate, sample six labels without replacement uniformly
  from 1..49; reject duplicate *tickets* and resample. The result is a truly
  uniform fixed-size pool of distinct combinations, not a historically weighted
  shape pool. `seed` fixes the ordered result. `candidate_pool_sha256` commits
  to the exact ordered candidate list.
- `shape_score`: for each candidate L3 class `c`, use the **frozen** empirical
  smoothed categorical estimate

  `q(c) = (N_historical(c) + 384 * p_fair(c)) / (2497 + 384)`;
  score `q(c)/p_fair(c)` (same formula as v0.17). Ranking is a descriptive
  experiment: it is **not a conditional probability of this ticket winning**.
  Class priors are derived from exhaustive 6/49 legal combination counts.
- `coverage`: use the existing v0.6 `exact_coverage()` dynamic programming
  oracle to calculate **max-hits among all selected tickets** distribution
  for *every one* of the `13,983,816` valid future draws, algebraically without
  Monte Carlo. By the inclusion logic, all portfolios of N distinct six-number
  tickets cover exactly N possible 6/6 jackpots.
- `optimization`: greedily maximize exact >=3 counts over a finite random pool
  (or >=4, >=5 per `--objective`), tie-break lexicographically by the other
  thresholds, then accept improving one-swap moves in candidate order. Compare
  to the same pool's first-N baseline and never return a worse objective.
  **This is a bounded local heuristic, not an all-space optimal portfolio.**
- `strategies`: `uniform`, `coverage`, `shape_score` each select the same N
  distinct candidates **from the identical uniform candidate pool**. The
  reports show exactly computed probabilities of at least one ticket matching
  >=3/4/5/6, and optionally show differences, but **do not claim expected
  profit or validated predictive uplift**.

## One-command Windows invocation

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-select-unified `
  --catalog results/v015_v016_p1_shape_catalog.npz `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --freeze docs/prospective_freeze_20261108.json `
  --pool-size 100 `
  --tickets 5 `
  --objective 3plus `
  --seed 649 `
  --out results/v021_unified_selection.json

Start-Process .\results\v021_unified_selection.html
```

Outputs protected from overwrite: `v021_unified_selection.json` (complete
settings, exact probabilities and bets), `_pool.csv` (100 candidate numbers and
frozen shape class scores), `_bets.csv` (actual numbers for three strategies),
`.html` (three side-by-side visualizations and numeric fair coverage).
The report deliberately includes no *winning* recommendation for a date;
its numbers are research tickets only.

## Validation, limits and decision rules

The integration test must use the real frozen P1 files (when available):
`check_freeze` must accept them and reject mismatched old P1 outputs, the
pool must contain legal unique combinations, each selected group must have
exactly N distinct tickets and 6/6 coverage exactly N/13,983,816.

The v0.17 historical prequential scores were all negative compared with exact
fair class probabilities. Therefore this project's strongest **evidence-backed
position** remains that the `shape_score` arm is exploratory. The maximized
threshold coverage may improve the chance of >=3/4 matches *over the baseline
portfolio*, which is a different goal than predicting likely next six labels.
This work is not a wager, not an API to place bets, and no financial return or
winning odds beyond the fair combinatorial identity is promised.
