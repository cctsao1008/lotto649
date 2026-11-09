# v0.26 P2 — Strategy A vs Strategy B under IDENTICAL exact prize coverage

## Pre-specified question

Does a historic multivariate HDDC selection method (Strategy B)
outperform a history-blind four-ticket pairwise-disjoint baseline
(Strategy A) on **later, unseen draws** when all theoretical lottery
coverage probabilities are held mathematically identical?

The v0.26 P1 exhaustive certificate established that four disjoint
six-number bets maximize the fair-outcome probability of at least one
official eight-tier award. Therefore both A and B must choose four
unique, mutually disjoint six-number bets. They have:

- identical exact probability of any official award;
- identical distribution of number of official prize-awarded tickets;
- identical exact union probabilities for all eight official tiers;
- identical main-number exact `>=3/4/5/6` coverage;
- identical expected number of winning tickets per draw.

Any difference in realized win counts is from the specific numbers
drawn, not a difference in unconditional IID winning probability.

## Strategy definitions

**A — Coverage**: shuffle labels 1..49 with a pre-specified seed
(independent of the draw history), partition 24 unique numbers into four
tickets of six. This realizes an exact optimal coverage portfolio
without fitting past outcomes.

**B — Coverage + HDDC**: reuse the v0.25 selector on only the training
history. Force `coverage_constraint="reference-3plus"`, which in
current implementation enforces **24 distinct numbers**. Do not
evaluate or accept the B portfolio unless its full official prize
coverage and main-number coverage exactly match A.

The multivariate historical metric is reported for descriptive
comparison but **is not an out-of-sample prediction metric**.

Covariance fitting uses only an independent IID synthetic cohort with
fixed `iid_samples=8192`, `covariance_seed=649`, `ridge=0.25`.
This independent stochastic covariance is a fixed model component,
not trained on future historical labels.

## Chronological pseudo-out-of-sample test

- Source: annual CSV through explicit cutoff 2026-11-06; no later
  outcomes are loaded for this analysis.
- Holdout: expanding training prefix; default first 1,800 historical
  draws, then next 25 drawn as holdout; stride 25 (no overlapping
  holdout blocks); maximum 8 folds (200 tested draws by default).
- At each fold, history-blind A is deterministically re-randomized
  with seed `a_seed + fold_index`. B is selected from *only* the
  preceding training prefix, with identical fixed selector settings.
- Each portfolio remains fixed throughout that fold's next 25 draws.
- Bonus ball is read independently from the validated CSV and checked
  against the six main numbers. Actual official-tier award events
  are computed using **both** six mains and the special.
- Paired `a_only`, `b_only`, `both`, `neither` count the same
  holdout draws; a two-sided exact discordance (binomial/McNemar
  conditional) p-value is reported as **exploratory**, not a
  preregistered confirmatory result.
- Per-fold trained-through date/id and evaluated date/id are saved
  as an explicit no-lookahead audit trail. Fold windows are nonoverlap.
- Both strategies' IID theoretical award probability is the same
  `0.1227314589619088` for all folds.

**Important methodological limitation:** the approach and parameter
choices were developed after examining historical data; a historical
walk-forward does not erase earlier researcher degrees of freedom.
Its inference is exploratory and must be validated against the
already sealed prospective four-ticket, ten-draw purchase (which was
not selected by these A/B rules) or a new *prospectively sealed*
portfolio-generation protocol. Historical walk-forward must not be
called a confirmed future-draw forecasting edge.

## Real data: reproduce on Windows

```powershell
git switch feature/v025-deficit-compensation-foundation
git pull --ff-only

python -m lotto7x7 strategy-ab-walkforward `
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
  --out results/v026_strategy_ab_walkforward.json

$r = Get-Content results/v026_strategy_ab_walkforward.json -Raw | ConvertFrom-Json
$r.comparison.parity
$r.comparison.difference_a_minus_b
$r.walkforward.aggregate
$r.walkforward.folds |
  Select-Object fold,train_end_id,test_begin_id,test_end_id,a_wins,b_wins
```

## Acceptance and follow-up

- Exact parity assertions must pass on every generated A/B pair.
- CSV main and bonus integrity must pass; no mismatched draw ids/dates.
- No training fold may include its next holdout.
- Expected awarded-ticket count and per-tier theoretical distribution
  must be exactly identical between A and B.
- Compare paired *realized* official prize events in untouched holdouts,
  not the historical objective decrease.
- Require prospective validation before claiming that HDDC offers an
  actual prediction advantage. Also report multiple-testing and
  retroactive rule-design limitations.
- Do not alter already-purchased A–D or issue #1 forecast freeze.

The selector is a bounded greedy/one-swap heuristic over 300
random candidate tickets, not globally optimal in the historic
covariance metric. The complete 6/49 objective coverage proof does
**not** imply that any particular HDDC ticket has increased odds.
