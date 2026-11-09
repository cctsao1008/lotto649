# v0.26 P3 — Full-pipeline IID calibration for Coverage A vs HDDC B

## Research question and null

In the **exploratory retrospective 200-draw** walk-forward fixed earlier,
history-blind Coverage Strategy A had **22** official-award winning draws,
while historical-deficit Strategy B had **25**. The difference was **+3
winning draws**, and the paired conditional exact discordance p was
`0.7708669945730264`. Both portfolios have exactly equal, globally maximal,
fair official eight-tier any-award probability of `0.1227314589619088` per
draw due to the four-ticket disjointness constraint.

Question: Under **independent fair 6/49 draws including a distinct special
ball**, how frequently does the *entire same A/B training-and-selection
pipeline* generate B-A differences at least as large as +3, by chance?

Null hypothesis: draws are IID legal uniform six-main-plus-special outcomes.
This test is an IID **pipeline-matched** Monte Carlo empirical calibration,
not an in-sample compensation-gain test, not a shuffle of realized winning
flags, and not a claim of forecast advantage.

## Exact protocol

1. Re-read validated annual CSV through historical cutoff 2026-11-06,
   including real special numbers, and the exact historical Shape universe.
2. Fix model parameters before evaluating: independent IID covariance sample
   count 8192, covariance seed 649, ridge 0.25; selector seed 649,
   pool size 300; A seed 649; expanding-train start 1800, holdout block 25,
   stride 25, maximum 8 non-overlapping folds.
3. Replay the *observed* P2 historical A/B walk-forward, requiring exact
   eight-tier + main coverage parity in each fold.
4. For each null replicate `r`, create an entirely independent sequence of
   **2497** fair legal draws with fixed seed `2649 + r + 1`. The *seven*
   unique numbers are drawn in one uniform without-replacement choice:
   first six are the mains (unordered), seventh is the special. Synthetic
   histories retain the original `(date, draw_id)` **only** for time/fold
   boundaries, never the historical numbers.
5. In each IID null replicate, rerun *all eight* B selection steps using
   **only that synthetic replicate's training prefix**, then score A and B
   on the same held-out 25 draws, including the special. History-blind A
   uses identical fixed seeds as in the observed protocol. Covariance model
   is fitted once to an external independent IID population, then held
   constant across observed and null runs.
6. Compare the observed +3 against the null differences `delta_r =
   B_wins_r - A_wins_r`. Report one-sided and two-sided empirical p values:

```text
p_upper = (1 + number of null deltas >= observed_delta) / (R + 1)
p_lower = (1 + number of null deltas <= observed_delta) / (R + 1)
p_two_sided = (1 + number of abs(null_delta) >= abs(observed_delta)) / (R + 1)
```

7. Store per-replicate A/B wins and paired `A_only, B_only, both, neither`,
   model settings and all null seeds. Check exact equality of real-vs-null
   fold indices and count. A failed coverage parity is an immediate error.
8. Retain `null_replicates=99` as the planned primary calibration (empirical
   resolution **0.01**). A quick `--null-replicates 3` is only a smoke test
   and cannot support useful inferential conclusions.

## Why this is preferable to a binomial test alone

- Correlation of A/B events on the same draw is respected.
- A/B selection depends on historical training sequences, including random
  candidate-pool constraints. Each null replicate **reruns selection**.
- Exact fair IID theoretical coverage parity is retained each fold.
- P-value resolution and search protocol are explicit.

Limits: the strategy, its features, hyperparameters and 200-draw evaluation
window were developed after previous research inspected history. All reported
p values are **exploratory**, and their interpretation should not be upgraded
to prospective confirmatory significance. IID calibration does not validate
an alternative non-IID or hidden physical-bias hypothesis. Do not re-tune
after inspecting null outputs to seek favorable p-values.

## Windows reproduction

```powershell
git switch feature/v025-deficit-compensation-foundation
git pull --ff-only

python -m pytest -q tests/test_strategy_ab_v026.py tests/test_strategy_ab_iid_calibration_v026.py

# Fast functional smoke: three independently generated histories.
python -m lotto7x7 strategy-ab-iid-calibration `
  --csv-dir data/csv `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --cutoff 2026-11-06 `
  --iid-samples 8192 `
  --covariance-seed 649 `
  --ridge 0.25 `
  --pool-size 300 `
  --selector-seed 649 `
  --a-seed 649 `
  --train-min 1800 `
  --horizon 25 `
  --stride 25 `
  --max-folds 8 `
  --null-replicates 3 `
  --null-seed 2649 `
  --out results/v026_strategy_ab_iid_smoke.json

# Planned complete exploratory run:
python -m lotto7x7 strategy-ab-iid-calibration `
  --csv-dir data/csv `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --cutoff 2026-11-06 `
  --iid-samples 8192 `
  --covariance-seed 649 `
  --ridge 0.25 `
  --pool-size 300 `
  --selector-seed 649 `
  --a-seed 649 `
  --train-min 1800 `
  --horizon 25 `
  --stride 25 `
  --max-folds 8 `
  --null-replicates 99 `
  --null-seed 2649 `
  --out results/v026_strategy_ab_iid_calibration.json

$r = Get-Content results/v026_strategy_ab_iid_calibration.json -Raw |
    ConvertFrom-Json
$r.observed
$r.null_calibration
$r.null_replay_summaries | Select-Object -First 10
```

## Acceptance

- Exact A/B eight-tier parity confirmed in all real and synthetic folds.
- Full legal 7-number IID outcomes, independent of the observed numerical
  draws, and distinct bonus per draw.
- Fixed, forward chronological folds; the synthetic training prefix cannot
  include its own future test draws.
- Complete **re-selection of B** for each null replicate and fold.
- Observed counts reproduce `A=22, B=25` under fixed parameters.
- Report empirical IID upper/two-sided p and *per-replicate* A/B counts.
- Do not mark prospective predictive advantage based solely on this P3 run.

Original purchased A–D 10-draw sealed ticket and issue #1 remain unchanged.
