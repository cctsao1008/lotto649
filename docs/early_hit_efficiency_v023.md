# Lotto649 v0.23 — Chronological Holdout Early-Hit Efficiency

## Research question

Does the historical 7×7 L3 graph-shape ranking (`shape_score`) retrieve **previously unseen (excluded from model training)** historical six-number sets earlier than an outcome-blind IID uniform candidate order? This is a **retrospective retrieval experiment**, not a lottery-prediction claim.

## Design, boundaries, and leakage prevention

- P1 input: historical draw IDs, legal six-main-number combinations, exact L3 fair-universe table, loaded using the existing strict `shape_prospective_v019.read_sources()` (including cached graph taxonomy validation).
- **Chronological holdout**: oldest 1,997 draws = shape-frequency model training, last 500 draws = target lookup. The target 500 draw numbers do **not** contribute to the model's L3 counts or ratio. The final 500 historical outcomes have already been explored during earlier project research, so this is **not a fully pristine prospective evaluation**.
- Model: for edge class `c`, `q(c) = (N_train(c) + tau*p_fair(c)) / (N_train + tau)`, where `tau=384`. Ranking score `q(c)/p_fair(c)`. Same v0.17 classification, v0.22 training-only reference builder, and v0.22.2 IID uniform candidate generator; **no new trained AI model** is introduced.
- Each seed uses NumPy `SeedSequence(root_seed).spawn(trials)`, and each offspring seed produces a new 10,000-ticket *candidate pool*. A candidate is sampled independently as six distinct uniformly chosen labels from 1..49. A six-number *combination* may repeat within/across pools. Both methods inspect **exactly the same candidate rows**; Shape Score sorts them by frozen training-only L3 ratio, Uniform keeps candidate-generation order.
- Early thresholds `K=100,500,1000,5000` are **predeclared**; 5000 is the primary endpoint. For each pool and K, count historical exact-six matches in the first K positions of **each strategy's order**. These are paired counts; no separate random pools or target-aware sorting are permitted.
- The two strategies must produce identical total match events when the **entire** 10,000-candidate pool is evaluated. The code asserts this. Never interpret full-pool equality as algorithm success or failure on the early-prefix ranking question.

## Statistical analysis

For cutoff K and random pool i, define `D_i(K) = shape_hits_i(K) - uniform_hits_i(K)`. The observed average advantage is `mean(D_i)`, with a paired percentile-bootstrap 95% confidence interval (2,000 resamples of **whole seed/pool pairs**). An *exploratory* two-sided one-sample t-test on `D_i` provides a p-value; apply **Holm familywise multiplicity adjustment** over the four declared K endpoints. Also report a paired sign test on which strategy has more early hits in each seed, and the number of ties. These measures describe **Monte Carlo / pool-sampling variation conditional on this fixed historical target archive**, **not** general uncertainty over possible future lottery draws. This is retrospective data analysis, not a causal inference or proof that a future draw is predictable.

Smaller early cutoffs have intrinsically low power. With 500 unique heldout six-sets and 1,000 seeds, the fair expected hit events are about 3.58 (K=100), 17.88 (K=500), 35.76 (K=1000) and 178.78 (K=5000). Treat sparse early hits carefully, particularly K=100 and K=500.

## Windows PowerShell

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

& $py -m lotto7x7 early-hit-backtest `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --trials 1000 `
    --pool-size 10000 `
    --holdout-count 500 `
    --cutoffs 100 500 1000 5000 `
    --primary-cutoff 5000 `
    --seed 210594425653336206127332942922757299723 `
    --out "results/v023_early_$stamp.json"
```

If `--seed` is omitted, the command uses a fresh OS CSPRNG 128-bit root seed and writes the exact integer in the JSON for later reproduction. This command **runs to completion**, not until Ctrl+C; `--trials` sets the number of new independently seeded pools. Its runtime scales with trials and pool size. `--progress-every 100` prints optional progress; by default it prints only the final summary and file locations.

Outputs use a fresh name and do not overwrite existing files:

1. `...json`: schema, settings, exact source SHA256s, training/target boundary, primary comparison, four cutoff comparisons, paired bootstrap CIs, exploratory p-values, Holm adjustment, full-pool count invariant and cautions.
2. `..._trials.csv`: one row per seeded pool, full-pool hit count, first hit ranks and the actual early-prefix event counts for each strategy/K. `seed_spawn_index` allows exact regeneration from root seed.
3. `..._cutoffs.csv`: per-K aggregate counts, expected IID count, differences, uncertainties, adjusted p-value and low-power warning.

## Reproducible initial P1 real-data result (2026-11-08)

Root seed `210594425653336206127332942922757299723`, 1,000 spawned seeds, 10,000 candidates per seed, train=1,997, holdout=500, exactly **10,000,000 candidates per strategy**, 5000 cutoff is 5,000,000 early guesses per strategy:

| K | Uniform hits | Shape hits | Shape minus Uniform | Holm-adjusted exploratory p |
|---:|---:|---:|---:|---:|
| 100 | 4 | 5 | +1 | 0.7057 |
| 500 | 17 | 26 | +9 | 0.4689 |
| 1,000 | 31 | 39 | +8 | 0.5875 |
| **5,000 (primary)** | **196** | **178** | **−18** | **0.5101** |

- The full 10,000 candidate pools had **356 exact target hits under BOTH strategies**; equal by construction.
- Nonempty pools=299; first hit came earlier under Shape in 144 pools and earlier under Uniform in 155 pools. All 4 Holm-adjusted p-values >0.05.
- Primary paired bootstrap delta interval crosses zero; no statistically supported improvement. The small-K positive differences do not constitute verified prediction value.
- This result should not be framed as a discovery of an anomaly in Taiwanese lottery draws, nor as a reason to alter fair-ticket jackpot probabilities.

## Implementation and tests

Modules: `lotto7x7/early_hit_v023.py`, `lotto7x7/early_hit_cli_v023.py` and their tests; CLI registered as `python -m lotto7x7 early-hit-backtest`. Read-only P1 sources; no modifications to historical files, model freeze, or prior benchmark results. Unit tests cover chronological exclusion of target from fit, paired full-pool invariant, exact match lookup, sparse/no-hit results, deterministic 128-bit seed, CLI flags, multi-comparison logic, output no-overwrite and input validation.
