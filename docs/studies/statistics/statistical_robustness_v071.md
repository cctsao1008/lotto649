# Lotto7x7 v0.7.1 — Statistical Robustness Audit

## Scope

This release **does not change** the 7×7 image layout, v0.6 candidate generator,
selection policies, original BIN/CSV archive, v0.7 baseline definitions, or
N=1–5 ticket cap. It adds reproducible, descriptive **paired statistical
analysis** of the S1 history-shape candidate pool against the B1 IID-random
candidate pool using a fixed selection strategy and identical draw targets.

The results are **retrospective exploratory research**. Neither geometrical
shape diversity nor superior exact partial-hit coverage establishes predictive
skill. The 6/6 fair probability remains `N / C(49,6)` for distinct fixed bets.

## Stage A: Audit an existing v0.7 report (no rerun)

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -v

& $py -m lotto7x7 robustness-audit `
  --input results/baseline_study_v07_100.json `
  --seed 649 `
  --out results/robustness_audit_v071_100.json
```

For each of the three selectors (`random`, `shape-diverse`,
`coverage-optimized`), the audit reports:

- **Paired 2×2 table**: both hit >=3, S1 only, B1 only, neither.
- **Exact two-sided McNemar** conditional binomial p-value, with discordant
  outcomes as its sample size (not 100 independent success comparisons).
- **Holm family-wise p-value correction** across three selectors in the audit.
- Observed S1−B1 >=3 difference, mean difference in **exact fair >=3
  coverage**, and observed-minus-fair difference.
- Draw-level paired non-parametric 95% bootstrap intervals (4,000 replicates),
  with deterministic resampling seed.
- Grouped calendar-year descriptive counts, observed differences and fair
  coverage differences (the calendar year is `draw_id // 1,000,000 + 1911`).

Ticket sets, recorded hit counts, >=3/4/5/6 flags and exact fair probabilities
are independently revalidated against the stored winning six-number main draw
before inference. This protects against incorrectly scored or partially edited
reports. The original JSON remains unmodified.

**Specific anchor from the 100-draw seed-649 sample:** McNemar two-sided p
values for S1-only vs B1-only pairs `(9,5)`, `(10,5)`, `(8,6)` are approximately
`0.423950`, `0.301758`, `0.790527`; the three Holm-adjusted values are all
approximately `0.905273`. This is **not statistically significant**.

## Stage B: Multi-seed and ticket-count sensitivity (computationally expensive)

Start with a small smoke run:

```powershell
& $py -m lotto7x7 robustness-sweep `
  --seeds 649,650 `
  --counts 1,3,5 `
  --warmup 200 --eval-draws 12 --stride 5 `
  --size 100 --families 20 `
  --out results/robustness_sweep_v071_smoke.json
```

For **all N=1..5**, with the same set of target draw IDs in every run:

```powershell
& $py -m lotto7x7 robustness-sweep `
  --seeds 649,650,651 `
  --counts 1,2,3,4,5 `
  --warmup 200 --eval-draws 30 --stride 5 `
  --size 100 --families 20 `
  --out results/robustness_sweep_v071_30.json
```

The second command entails **3 seeds × 5 ticket counts × 30 target draws =
450 full baseline comparisons**, each including B0, B1 with three selectors,
S1 with three selectors and B2. Computational effort can be considerable.
Change `--eval-draws` to 100 only after the smaller runs succeed; **each
additional seed does not add independent draw outcomes**.

Each run records direct S1/B1 paired counts, McNemar p, Holm-adjusted p across
**all seed × N × selector tests in that sweep**, exact fair >=3 differences,
and compact target-wise score series. The cross-seed aggregate:

1. Computes S1−B1 per target draw for each seed.
2. Averages these values **across seeds within the target**.
3. Averages and bootstraps these **draw-level** values (never the
   `seed × draw` matrix as if those were distinct independent draws).
4. Reports per-seed sensitivity, per-year breakdown, fair-coverage delta, and
   excess observed-versus-fair difference.

A bootstrap interval excluding 0 is exploratory evidence only: repeated
historical testing, target selection, multiple alternative metrics, and
potential adaptation to previous results remain unaccounted for.

## Interpretation and reproducibility

- **McNemar test null**: conditional on disagreeing paired outcomes, the
  S1-only and B1-only directions are equally probable. It does **not** test
  whether a historical image encodes physical randomness; minor differences in
  theoretical coverage also affect the observed success probabilities.
- **Fair coverage delta**: `mean(p_S1_exact_ge3 − p_B1_exact_ge3)` for the
  already frozen portfolios, with **no unseen targets used for selection**.
- **Excess-over-fair difference**: mean of
  `(S1_hit − B1_hit) − (p_S1 − p_B1)`; descriptive only.
- **Bootstrap unit**: each selected historical winning draw; seed replications
  are correlated because they reuse exactly the same winning numbers.
- **Holm family**: the tests in the *current* invocation. No correction can
  retroactively account for earlier runs, alternate seeds viewed informally,
  historical periods, or other unreported experiments.
- **Year analysis** is diagnostic, not a claim of seasonality; annual subsamples
  can be very small.
- **N=1..5** arms are *separate alternative portfolios*, not 15 or 25 actual
  purchased tickets. There is no real-money transaction.
- The long-term evidence standard remains **prospective**, immutable pre-draw
  snapshots with independently timestamped commitments and later post-draw
  scoring, without tuning after outcomes are observed.
