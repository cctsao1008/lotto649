# Lotto649 v0.22 — Timed Historical Exact-Hit Benchmark

## Research purpose

Search for **past** winning six-number combinations by repeatedly generating
fresh legal 6/49 candidates. Unlike a prospective prediction, all historical
winning sets are already known to the evaluator. A "hit" means **all six main
numbers match exactly** one or more historical records; bonus numbers are
excluded. The program prints every hit and records its original draw ID.

This experiment is an educational/retrieval and scoring-order test. It must not
be interpreted as a chance to win a future lottery draw. Every six-number
combination has the same probability 1/13,983,816 under a fair independent draw.

## Method

Each independent trial generates exactly `--per-round` distinct legal tickets
using the **same uniform 49-key six-of-49 sampling algorithm used by v0.21**.
`--per-round` may be 1–10,000, default 10,000. Each trial is an independent
replication: across different trials a previously sampled ticket can recur;
there is **no claim of global uniqueness across 20 minutes**. Candidates are
sampled without consulting history. Their order is compared in two arms:

- `uniform`: the original uniform candidate sampling order.
- `shape_score`: the **same** sampled candidate pool sorted by the P1 L3
  graph-category score, `q(c)/p_fair(c)`, with tau=384 and stable tie ordering.
  No exact-match outcomes are inspected until **both** orders are fixed.

Each hit has two counterfactual guess ranks: `uniform_rank` and `shape_rank`.
The underlying set of historical matches for a *completed trial is exactly
identical* across both arms: a ranking can find hits earlier, but **cannot
create additional hits** when both examine all 10,000 tickets. Do not sum
both strategies' hits as separate wins.

The next trial starts automatically until the 20-minute wall-clock budget is
reached. Finishing an in-progress trial may go slightly beyond the deadline.
`Ctrl+C` writes the final summary. Results are flushed to the hit and round
CSVs after each trial; JSON summary is checkpointed periodically.

## Two historical comparison modes

- `--target all`: all 2,497 existing history draws are eligible matches.
  **In-sample information leakage by design:** the class-score counts were
  learned from those same 2,497 drawings, including targets. A rare historical
  class can be artificially ranked high; this is *retrieval*, never evidence of
  predicting an unseen event.
- `--target holdout --holdout-count 500`: the last 500 historical draws are
  queried, while the class scorer is fitted **only to the first 1,997 draws**.
  This avoids direct target inclusion in this score fitting. However P0–P2
  and the choice of graph classes were already influenced by the complete
  historical archive: the comparison remains exploratory, **not** an
  independent prospective blind study.

Exact six-number targets are matched using an injective base-50 sorted-ticket
code. The historical records and the P1 exact-universe file are validated
before the experiment; neither is written or modified. For each trial, the
program reports the mathematically exact random-baseline probability of at
least one hit under without-replacement sampling (hypergeometric).

## Windows PowerShell — run for 20 minutes

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 historical-hit-benchmark `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --minutes 20 `
    --per-round 10000 `
    --target all `
    --seed 649 `
    --progress-every 100 `
    --out results/v022_historical_hit_20min.json
```

To check held-out historical retrieval instead:

```powershell
& $py -m lotto7x7 historical-hit-benchmark `
    --minutes 20 --per-round 10000 `
    --target holdout --holdout-count 500 `
    --seed 649 --progress-every 100 `
    --out results/v022_holdout_20min.json
```

For a **quick smoke test** without 20 minutes:

```powershell
& $py -m lotto7x7 historical-hit-benchmark `
    --minutes 20 --per-round 10000 --max-rounds 3 `
    --out results/v022_smoke.json
```

All output files must be new; the command refuses to overwrite them.

## Output and interpretation

Three files per experiment:

1. `<stem>.json`: input SHA256, settings, reference size, exact fair baseline,
   summary, in-sample cautions, first-hit ranks and mean first-hit rank.
2. `<stem>_hits.csv`: **one row per exact historical match**, immediately
   flushed. Columns `trial, pool_index, uniform_rank, shape_rank, numbers,
   historical_draw_ids, shape_ratio, edge_class` and timestamps.
3. `<stem>_rounds.csv`: candidate count, hit count, earliest ranked positions
   and which strategy found a hit earlier within each trial. Zero-hit trials
   are recorded as well.

`matching_candidates` counts hit **events** across independent trials;
`unique_hit_six_sets` and `unique_historical_draw_ids` distinguish unique
historical combinations from repeats across trials. Reports compare
`uniform_wins`, `shape_wins`, and `ties` for the first-hit rank **only on
trials with at least one matched candidate**. Win counts need paired null
calibration before making any statistical inference.

The final report is not a future lottery bet recommendation. Coverage-optimized
portfolio strategies target >=3/4/5 outcomes over **multiple tickets**, not
retrieval order for a historical exact 6/6 membership test; therefore the
meaningful arms here are fair random order versus historical L3 shape order.

## Validation

- Reproduced the previously found Seed 649 historical hits at uniform guesses
  4,278 and 9,627 (issue IDs 100000081, 93000103).
- Same first trial: L3 shape-score ranks 3,786 and **1** respectively.
  This is an **in-sample** finding; target 93000103 belongs to a historically
  discovered rare graph class, so its high rank is not forecasting evidence.
- Three independent 10,000-guess trials found 5 hit events in total.
- All methods require a properly validated P1 fair-universe baseline and
  strictly valid six-number tickets.
