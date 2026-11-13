# Lotto649 v2.4 P2A — Astronomical 64-Hexagram × Outcome Association

Tracking: [Issue #36](https://github.com/cctsao1008/lotto649/issues/36).

## Study design

**Goal:** Determine whether the pre-draw, USNO-derived **solar-8 × lunar-8**
64-state hexagram coding shows a retrospective statistical association with
Lotto 6/49 outcomes. It does not make predictions or select tickets. This is a
**researcher-defined Gray-code encoding**, not a historically authenticated
traditional Yi Jing divination procedure.

Inputs:

- Frozen 2,498-draw P2 archive
  `data/research/v24_iching_p2/v24_p2_solar_lunar_hexagram_draw_context_2498.csv`
  SHA-256 `a46f08269fc8021369f1a045116051e4f9591ba83801d5cdbe8c1a08873edf39`.
- P2 archive audit JSON of the same stem; source provenance additionally
  checks the 46-source USNO event manifest pointer and P1 archive checksum.
- 22 early-2004 rows have no complete astronomy phase due to the 2003 boundary;
  they remain in the audited source but are excluded from the test. **2,476**
  rows / 64 astronomical states enter the analysis.
- All post-draw outcome parity codes are recomputed independently from the six
  strictly ascending main numbers; special numbers are verified but excluded
  from the 6-number shape and parity tests. The 7×7 L1 component count is
  independently re-derived by BFS. No reliance on post-draw columns as
  astronomical predictors.

## Six prespecified global endpoints

| Key | Statistics | Direction |
|---|---|---|
| `outcome_hexagram_64x64` | 64×64 conditional categorical Pearson discrepancy | Large |
| `hamming_distance_mean_two_sided` | Absolute deviation of mean XOR bit distance from *stratum-conditional expected* distance | Large |
| `sum_by_sky_64` | Between-state total residual sum of squares scaled by exact fair sample-sum variance 1075 | Large |
| `odd_count_by_sky_64` | Between-state odd-count residual score scaled by exact hypergeometric variance | Large |
| `shape_L1_mean_by_sky_64` | Between-state 7×7 component count residual score scaled by prior exact 6/49 variance | Large |
| `shape_L1_64x6` | 64×6 conditional categorical Pearson discrepancy | Large |

**Conditioning:** In each `(calendar year, draw weekday)` stratum, fix the
astronomical sequence and the multiset of entire observed lottery outcomes,
then **permute outcomes within that stratum**. Every source target is moved as
one complete six-number result, never one ball independently. The conditional
expected cell count for sky code `a`, outcome category `b` is

`E_ab = Σ_stratum [n(stratum,a) · n(stratum,b) / n(stratum)]`.

For continuous features, the expected total within sky class `a` is

`E_a = Σ_stratum n(stratum,a) · mean(feature | stratum)`.

All test statistics use these **fixed** conditional expectations. Categorical
sparsity invalidates naive asymptotic chi-square p-values, so **all six
statistics are calibrated by the same 3,999 conditional permutations**, fixed
`random.Random(240649)` sequence. Upper-tail p-values are
`(1 + #permuted_stat >= observed_stat)/(3999+1)`. Hamming is **two-sided**
by taking the absolute deviation from its exact conditional permutation
mean prior to comparing the statistic. The six global p-values receive
**Bonferroni** familywise correction, `min(1,6*p)`; no per-hexagram hypothesis
or rank-optimized subgroup is claimed.

The P0 existing exact finite universe, `C(49,6)=13,983,816`, provides the
fair sums/odd-count/L1 moments and the **exact nonuniform 64-state** distribution
of six-bit sorted-parity codes (dynamic programming, not 1/64). This exact
reference is diagnostic; the conditional permutation comparison is the
inferential null against *astronomy association*.

## Important limitations

- This is **not independent confirmatory evidence**: predecessor P1A has
  already examined overlapping historic outcomes and other astral mappings.
- The 64×64 table is sparse (2,476 observations and 4,096 cells) and each sky
  state contains only ~39 observations on average. A non-significant result
  cannot establish the absence of every signal.
- Exchangeability within `(year,weekday)` is expected for independent fair
  lottery draws and controls some calendar composition effects, but not all
  potential procedural trends. Conditioning on the empirical outcome
  distributions is **not** a test that the overall lottery itself is fair.
- A past statistical association, even if found, would not demonstrate
  causation or practical future predictive utility; a separate genuinely
  future-only frozen protocol is needed to make any forecasting claim.
- Sampling uses **12:00 Asia/Taipei on each draw date**, not verified historic
  draw time. New-moon interval fractions are time proxies, not measured lunar
  illumination or actual traditional lunar-calendar dates.
- Keep original USNO JSON, 2,498 historic result CSV/BIN/manifest, v2.3
  prospective seals, P0/P1/P1A/P2 archives and actual paid ticket records
  untouched.

## Reproduce on Windows PowerShell

Run from your existing local repository (stdlib only):

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p test_iching_hexagram_association_v24_p2a.py -v
& $py -m lotto7x7.iching_hexagram_association_v24_p2a `
    --p2-source data/research/v24_iching_p2/v24_p2_solar_lunar_hexagram_draw_context_2498.csv `
    --p2-audit data/research/v24_iching_p2/v24_p2_solar_lunar_hexagram_audit_2498.json `
    --out-dir results/v24_iching_p2a_local
```

The CLI rejects overwriting existing research artifacts. To regenerate, use a
new output folder. GitHub's one-time workflow runs the same 3,999 null
permutations against the frozen repository archive, verifies all source
hashes and output metadata, then commits exactly the two additive P2A results
under `data/research/v24_iching_p2a/`:

- `v24_p2a_hexagram_association_2498.json`: statistics, Monte Carlo p-values,
  Bonferroni correction, exact theory, SHA-256 and source-coverage audit.
- `v24_p2a_hexagram_association_2498_states.csv`: 64-row *descriptive* state
  counts and observed mean outcome characteristics (no per-cell significance
  claims, no selected tickets).

## Decision rule

- No global adjusted p below 0.05: `EXPLORATORY_ASSOCIATION_NO_EVIDENCE`.
- At least one adjusted p below 0.05:
  `EXPLORATORY_ASSOCIATION_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION`.

Either decision is purely retrospective/exploratory. A prediction claim is
**never** allowed from P2A alone.
