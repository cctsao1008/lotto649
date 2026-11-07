# Lotto7x7 v0.7.3 — Independent CPU Audit and Full-History Multi-Seed Robustness

## Purpose and study design

Validate all **2,297 chronological target draws** in the v0.7.2 6/49 full-history
GPU result and compare multiple **separately generated full-history seed runs**.
This is a retrospective, exploratory research audit, not evidence of a
prospective lottery forecasting advantage.

The dataset has 2,497 historical draws (2004-01-05 through 2026-11-06) and
200 warmup-only draws. The evaluated targets are indices 200..2496, inclusive.
Each target uses exactly five distinct six-number tickets per arm in the
recommended run. Neither additional candidate combinations (M=100) nor
random-seed replications are purchased lottery tickets.

The current source manifest identifies **Biga** (third-party single source);
it does **not** establish independent validation against Taiwan Lottery's
official archives. The AUZO downloader was superseded and is not a dependency.

## Audit scope: `full-history-audit`

- Read binary data **without importing the project's image BIN decoder**:
  - `lotto649_images_v2.bin`: magic `L649IMG2`, header `<8sII`, rows `<I49B`.
  - `lotto649_history_v1.bin`: magic `L649HIS1`, header `<8sII`, rows `<II7B`.
- Confirm both BIN SHA256/byte lengths from `lotto649_source_manifest.json`,
  correct draw ordering, six-hot pixel masks, sorted and unique main numbers,
  special number validity, valid increasing dates and matching ROC year.
- Confirm the `archive_sha256` matches exactly the v0.7.2 input image bytes
  followed by little-endian signed 64-bit draw IDs. Reject changed source.
- Require the **complete** time-ordered eligible range by default; validate
  each target ID, preceding cutoff, winning numbers, 6 arm names, N tickets,
  ticket uniqueness, and B2 full disjointness.
- Recalculate every ticket's intersection with the actual winning numbers
  independently in CPU Python (no CUDA and no v0.7.2 GPU `score_batch`).
  Verify full max-hit histograms, >=3 hit counts/rates, mean maximum hits,
  per-year counts, and the two reported S1-B1 matched comparisons.
- Check all reported exact fair coverage thresholds 3/4/5/6:
  - **Independent** hypergeometric closed-form equation for disjoint portfolios.
  - Existing v0.6 **shared exact-DP routine** for overlapping portfolios.
  - Six-hit anchor for every fixed N-ticket portfolio: `N/C(49,6)`.
  **This is not an independent implementation of the overlapping-set DP.**
- Recalculate Exact McNemar conditional p-values with Python integer
  binomial coefficients, Holm correction, and draw-level bootstrap intervals.
- Save an audit receipt containing the canonical input report JSON SHA256,
  provenance digests, exact counts, year checks and limitations. The audit
  output is created **exclusively**; it does not rewrite the input report.

For N fully disjoint tickets and at least three matched numbers:

\[
P(\max_i H_i\ge 3)
=\frac{N\sum_{h=3}^6\binom{6}{h}\binom{43}{6-h}-\binom{N}{2}\binom{6}{3}^2}{\binom{49}{6}}.
\]

For thresholds 4, 5 and 6, the pair intersection term is zero. These are
analytical under **fair independently drawn 6/49 outcomes**, not empirical
forecasts.

### Run on Windows PowerShell

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -v

& $py -m lotto7x7 full-history-audit `
    --report results/v072_biga_2497_full.json `
    --data data/bin/lotto649_images_v2.bin `
    --manifest data/bin/lotto649_source_manifest.json `
    --out results/v073_biga_2497_seed649_audit.json
```

If you must audit a five-target smoke report, add `--allow-partial` and use
a different output name. A partial audit cannot enter the multi-seed full-
history aggregation.

## Independent full-history seeds: 649, 650, 651

The user already has the entire 2,297-target GPU run for **seed 649**.
Do **not** rerun it. Compute two additional matched runs with identical
`--warmup`, `--count`, `--size`, `--families`, `--scenarios`, `--device` and
input BIN, changing **only** seed, checkpoint and output path.

```powershell
& $py -m lotto7x7 full-history-gpu `
    --device cuda --warmup 200 --count 5 --size 100 --families 20 `
    --scenarios 8192 --seed 650 `
    --checkpoint results/v072_biga_2497_seed650_checkpoint.jsonl `
    --out results/v072_biga_2497_seed650_full.json

& $py -m lotto7x7 full-history-gpu `
    --device cuda --warmup 200 --count 5 --size 100 --families 20 `
    --scenarios 8192 --seed 651 `
    --checkpoint results/v072_biga_2497_seed651_checkpoint.jsonl `
    --out results/v072_biga_2497_seed651_full.json
```

In case of interruption, **rerun the same command** with `--resume`, and keep
all other parameters and its checkpoint path unchanged. A seed-specific
checkpoint must **never** be reused with a different seed.

Audit each new report independently:

```powershell
& $py -m lotto7x7 full-history-audit `
    --report results/v072_biga_2497_seed650_full.json `
    --out results/v073_biga_2497_seed650_audit.json

& $py -m lotto7x7 full-history-audit `
    --report results/v072_biga_2497_seed651_full.json `
    --out results/v073_biga_2497_seed651_audit.json
```

Aggregate **only after each separate CPU audit passes**:

```powershell
& $py -m lotto7x7 full-history-multiseed `
    --reports results/v072_biga_2497_full.json results/v072_biga_2497_seed650_full.json results/v072_biga_2497_seed651_full.json `
    --audits results/v073_biga_2497_seed649_audit.json results/v073_biga_2497_seed650_audit.json results/v073_biga_2497_seed651_audit.json `
    --out results/v073_biga_2497_multiseed.json
```

## Interpretation and statistical safeguards

For each target draw `t`, for seed `s`, and for each matched selector:

- `d[t,s] = 1{S1 hits >=3} - 1{B1 hits >=3}`;
- `f[t,s] = exact_fair_P_S1(>=3) - exact_fair_P_B1(>=3)`;
- `r[t] = mean_over_seeds(d[t,s] - f[t,s])`.

**One target draw contributes exactly one value `r[t]`**, no matter how many
seed runs are used. Confidence intervals are percentile bootstrap of **draw
indices**, not 2,297×3 independent rows. The report includes overall
observed/fair-adjusted differences, differences by calendar year, and
strategy-specific fairness residuals (particularly B2). For the two strategy
contrasts in each seed, it also reports Exact McNemar and Holm-adjusted
p-values across all seeds in that particular comparison invocation.

The pooled cross-seed result deliberately has **no artificial pooled McNemar
p-value**, because it would violate independence. Bootstrap percentile
intervals are **exploratory**, do not control for previously tried seeds,
strategies or periods, and should not be presented as confirmatory evidence.

## Explicit limitations

1. BIN cross-format consistency and SHA256 verify repository artifacts, but
   both derive from the same Biga source, not Taiwan Lottery official records.
2. This verifies scoring and reported math, not an independently recreated
   or timestamped proof of the original ticket-generation procedure.
3. Overlapping-ticket fair probabilities reuse the original exact DP; only
   disjoint formulas are mathematically independent.
4. The original v0.7.2 GPU selector is **scenario-coverage**, not the original
   v0.6 exact `coverage-optimized` selector. Do not merge their labels.
5. Different random seeds do not generate new independent official draws.
6. Inference remains retrospective even if the full-history sample is large;
   independent prospective shadow snapshots are still necessary.
