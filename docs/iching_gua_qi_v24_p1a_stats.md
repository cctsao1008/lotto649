# Lotto649 v2.4 P1A — twelve Gua Qi × draw outcomes: exploratory association screen

## Research question

Given the frozen **USNO-only 12-state solar Gua Qi** mapping, does the historical
membership of a Lotto649 draw in one of the twelve seasons carry evidence of a
statistical association with (a) the total of six main balls, (b) the number of
odd main balls, or (c) the 7×7 L1 graph shape of those balls?

This is an **exploratory retrospective statistical test**, not a forecast, a
traditional divination algorithm or an intervention. The lottery's intended
fair IID design predicts no such association. An observed numerical difference
between seasons is *expected even under a fair IID lottery*.

## Inputs, source lock and coverage

- Original: `data/research/v24_iching/v24_p1_gua_qi_draw_context_2498.csv`;
  SHA-256 `53f35eb6fb0f6f5d6bc36e4b6a78aa63586263c924ba65940cc86c588c4fb1f1`.
- P1 astronomical mapping: `usno_quarter_anchor_centered_12state_half_up_v1`.
- 2,498 frozen draws from 2004-01-05 through 2026-11-10, 2,476 season-mapped;
  22 early-2004 draws remain unmatched because 2003's preceding USNO season
  anchor is absent. No synthesized 2003 anchor or inserted lottery rows.
- Pre-draw predictor is **date-based solar state only**. The 6 main numbers,
  `same_draw_hexagram_code_OUTCOME_ONLY`, sum, odd count and graph class are
  all **post-draw outcomes**, used only for retrospective tests. USNO events
  are already permanently archived. No CWA data are imported.
- Solar-state time is derived for local **12:00 UTC+08**, a date-level
  convention, not a confirmed historical lottery drawing clock time.

## Exactly defined endpoints (four families)

Let `s=0..11` be the solar Gua Qi state and `n_s` the number of archived draws
in that state. The theoretical uniform 6-from-49 null is over
`C(49,6)=13,983,816` legal unordered tickets.

1. **Sum mean** `S=sum_{j=1}^6 x_j`. Exactly
   `E[S]=150`, `Var(S)=1075` (sample without replacement).
2. **Odd count mean** `K=sum_j (x_j mod 2)`. Exactly
   `K ~ Hypergeometric(N=49, K=25, n=6)`.
3. **Shape L1 component-count mean** `C`: number of connected components of
   the six occupied cells under the *pre-existing* frozen 7×7 sequential
   raster and 8-neighbor graph definition. The exact population component
   frequencies for `C=1,2,3,4,5,6` are:
   `[58620,669462,2775454,5093678,4166304,1220298]`, taken from the
   exhaustive, versioned v0.16 enumeration, not a fitted historic model.
   Its first two moments are computed from that exact six-category PMF.
4. **Shape L1 complete category distribution**: a 12×6 observed contingency
   table. Its observed Pearson statistic is conditioned on the full historical
   six-class margin. No asymptotic chi-square p-value is used; rare `C=1`
   expected cells can be below 5.

For each of the first three responses `Y`, the observed state heterogeneity
statistic is

```
T(Y) = sum_s n_s * (mean(Y|s) - mean(Y))^2 / Var_fair(Y)
```

Using `mean(Y)` *from history* is intentional: this tests association with
**season**, not a historical all-season mean offset from the fair baseline.
Descriptive state-level scores `z_s=(mean(Y|s)-E_fair(Y)) /
sqrt(Var_fair(Y)/n_s)` are included for context, but **no** per-season
p-values or causal claims are drawn from them.

## Calibration and multiple testing

The null hypothesis for the primary tests is *no conditional association with
Gua Qi state*. Condition on the recorded dates, twelve season labels and
empirical full outcome vectors, and perform **3,999 seeded random permutations
within each `(calendar year, ISO weekday)` stratum**. This preserves:

- Different historical years' empirical ball-outcome distributions;
- Draw weekday / changing schedule and season counts;
- Correlations between sum, parity and shape for the same original ticket;
- Exact empirical overall class counts and all ticket values.

For each of the four **global** endpoints, calculate the one-sided
permutation upper-tail p-value with add-one correction:
`p=(1 + #{T_perm >= T_observed})/(3999 + 1)`.
Report Bonferroni-adjusted p-values over the **four predeclared families**
(`min(1,4p)`). Thus the familywise exploratory decision uses all twelve
states jointly for each endpoint, without cherry-picking the most exciting
state or category. The 3,999 permutations imply minimum raw p=0.00025.

This is a **conditional permutation association test**, not a complete fair-IID
joint-sum/shape simulator. The *theory baselines* are exact fair 6/49 results;
permutation inference conditions on the observed empirical distribution.
The interpretation assumes exchangeability of outcomes within year/weekday
under the no-season-association null; finer temporal changes may remain.

## Reproduce without modifying inputs

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -p 'test_iching_gua_qi_stats_v24_p1a.py' -v
& $py -m lotto7x7.iching_gua_qi_stats_v24_p1a `
    --source data/research/v24_iching/v24_p1_gua_qi_draw_context_2498.csv `
    --out-dir results/iching_v24_p1a_20261113 `
    --reps 3999 --seed 649
```

The two outputs are exclusive-create, and the user should choose a new
`--out-dir` to repeat an experiment:

- `v24_p1a_gua_qi_association_2498.json`: complete exact baseline, four
  permutation tests, Bonferroni corrections, effect sizes, provenance,
  interpretation and limitations.
- `v24_p1a_gua_qi_association_2498_states.csv`: 12-state sample sizes, fair
  theoretical mean comparison, descriptive z-scores, full L1 category counts
  and theoretical expected counts.

The one-time GitHub Actions workflow performs the 16 module tests, runs the
3,999-permutation experiment on the actual pinned archived history, checks
counts/hashes and commits only these two new artifacts. It is **not** a
recurring draw downloader or prospective monitoring pipeline.

## Research decision boundary

- An adjusted p-value ≥ 0.05 is exploratory **NO_EVIDENCE** of this particular
  seasonal association, not a proof that all possible signals are absent.
- An adjusted p-value < 0.05 only triggers **independent future review**, never
  a forecast GO decision, a re-ranked ticket, or an I Ching causation claim.
- The frozen hypotheses were chosen after prior historical discussions;
  **no** retrospective result is truly prospective preregistration.
- No modifications to original `data/csv`, `data/bin`, previously purchased
  tickets, pre-draw seals, v2.3 research, P0 v2.4 or P1 v2.4 archives.