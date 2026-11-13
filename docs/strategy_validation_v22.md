# Strategy Validation v2.2 — Historical Walk-forward Falsification

## Research question and evidence restrictions

The v2.1 exhaustive study proved that the **retrospectively frozen** M0 Sum/Log/Joint/Shape rank objective has an attainable global optimum among all C(49,6) legal six-number tickets. It **did not** demonstrate predictive value for future Lotto649 results.

v2.2 is a new, **historically exploratory**, leakage-controlled assessment of the underlying four-block *historical deficit repair* rule. It does **not** claim that recomputing one model on past dates turns those dates into truly independent prospective observations. The feature selection, mathematical formulation, and portions of the time series were examined by previous research; therefore any favorable retrospective p-value is hypothesis-generating only.

### Sources and timing

- Read-only original 2,497-draw history through 2026-11-06 / #115000103. Must pass the original prefix SHA-256 and verified archive/BIN consistency gate. Although newer input may contain subsequent draws, **no post-cutoff draw is used as training or evaluation in v2.2**.
- Original, independently regenerated exact fair Sum×Log grid and complete L1/L2/L3 shape population counts; these are mathematical constants, not outcomes leaked from the test period.
- Expanding training prefixes: 1,850, 1,900, ... 2,450 draws. Thirteen 25-draw, **nonoverlapping** holdout blocks, 325 historical target draws in total. Dates/IDs and special balls verified through independent CSV loaders.
- At each fold fit all four gain tables, fair-standardization and empirical candidate ranks from the prefix only. Never refit from later training or holdout observations. The B portfolio remains fixed for the fold's 25 later results.

### The history-based four-ticket arm (B)

For each fold, generate 40,000 independent uniform candidate tickets with seed `704011 + fold*101`. Compute the same per-prefix sum, log-product, joint and taxonomy shape hypothetical-append gains as the frozen M0 family. Require each candidate's four gains to be strictly positive. **Important deviation from v2.1 global optimization:** re-rank the qualified candidates *using only the current fold's candidate sample and training prefix*, select a shape-round-robin subset of 1,000 and apply six-start bounded greedy plus coordinate ascent. It selects **four pairwise number-disjoint tickets of different Shape partitions**. This is an **M0-style fold-specific bounded implementation**, not the original v2.1 complete-universe exact optimizer or a prospective version of its final fixed winners.

The model and every candidate pool have deterministic inputs and SHAs. No previous tickets or purchased holdings are research controls.

### Coverage parity control (A) and official award outcome

Generate another independently seeded, history-blind set of four disjoint tickets each fold. Validate with the exact official prize outcome engine that all tiers' per-portfolio probability profiles, number-of-winning-tickets profiles and any-award probability match B *exactly*. Under fair IID, the chance of at least one official tier among four disjoint tickets is **0.1227314589619088** for both arms, regardless of number choice.

Score 25 later draws, including the **actual special ball**, to obtain a paired table: A-only, B-only, both, neither. The exact conditional two-sided binomial/McNemar discordance p-value is descriptive. The holdout event is not judged by whether the retrospective repair score improved; it is judged by actual realized award events. Return no monetary expected-value or ROI claim because upper-tier payoffs are not fixed and neither betting costs nor winnings imply an ex ante selection edge.

### Predeclared independent feature-preference diagnostic

To directly measure the score's alleged future signal, for each held-out main six-tuple and each trained prefix generate **16 independent fair six-number controls** (fixed seed `202200000 + fold`). Apply the same four standardized repair gains to real and control tickets, *without requiring the realized ticket to have positive repair gain*. Use the pre-specified scalar `min(Sum, Log, Joint, Shape)`. For each actual ticket, the score is the fraction of 16 control scores it strictly beats plus **half** the equal-score fraction.

Under an **ideal randomized fair-control design**, a real draw and its independently randomized controls are exchangeable given the past, so the unconditional per-target fair null expectation is **0.5**. In this reproducible implementation the controls come from a **fixed-seed pseudorandom stream**; after conditioning on its fixed realized values, exact conditional exchangeability is not assured. Therefore the reported `min(1, 2*exp(-2*n*(mean-0.5)^2))` is a **nominal bounded-Monte-Carlo sensitivity diagnostic**, not a confirmatory p-value or a fully conditional exact guarantee. This comparison does not estimate a normalized joint probability over 13,983,816 draws. Targets are chronological; folds do not overlap in evaluation but their expanding training sets do. Never apply an ordinary independent Gaussian t test across overlapping training windows.

### Limitations and pass criteria

Engineering PASS requires (a) identical source/draw prefix and theory hashes, (b) independent special/main CSV consistency, (c) 13 complete nonoverlapping holds with training ending strictly before each hold, (d) identical exact fair official award probabilities at all folds, (e) reproducible seeded candidate streams and winner validity, and (f) all score controls in [0,1].

Predictive research is reported **NO_GO** unless a positive effect passes its respective exploratory p<0.05 threshold; even if both pass, do **not** declare confirmed predictive value due to multiple tested models/outcomes and historical strategy exploration. A later *new future-only pre-registered protocol* with independent unobserved events is mandatory before any confirmation. Do not overwrite the existing prospectively sealed v0.19, P3f, P5A or research artifacts. Under IID fair 6/49, higher repair ranks cannot increase a specific ticket's chance of being drawn.

### Running

```powershell
python -m unittest discover -s tests -p 'test_strategy_validation_v22.py' -v
python -m lotto7x7.strategy_validation_v22 --csv-root data/csv --bin-root data/bin --out results/v22_walkforward_exploratory.json
```

Output JSON is write-once. `results/` is a generated artifact directory, not input training data. The CI archives the JSON and fails closed if any source gate or coverage parity assertion fails.

## Observed historical exploratory result — 2026-11-13

**Pipeline:** 13 nonoverlapping, 25-draw pseudo-out-of-sample folds; **325** scored historical target draws within the immutable **2,497**-draw training/evaluation archive through **2026-11-06**. Training prefixes 1,850/1,900/.../2,450; each fold uses a distinct deterministic 40,000 raw candidate sample, an internally reranked eligible subset screened to 1,000, six-start greedy/local four-ticket selection, and an independent history-blind four-ticket portfolio. **Every fold** strictly trains before its holdout, holds both portfolios fixed within the 25 target dates, checks actual special balls, verifies 24 distinct numbers, and verifies exact official-prize probability parity between A/B.

[Initial successful GitHub Actions run 43353037196](https://github.com/cctsao1008/lotto649/actions/runs/43353037196/job/131648410933) **PASS**, **12/12 unit/adversarial tests** and source/coverage/chronology verification. Initial artifact `strategy-v22-expanding-walkforward` ID **14005230608**, report SHA256 `3de6bf28f1ee2ceb0c39135ed8a3a56d47ce320bf53629166ac265137c7a6c95`. A follow-up correction explicitly labels the pseudorandom control score p-value as nominal and exploratory, without changing scores/choice procedures or fold results.

| Endpoint, descriptive historical holdout | History-blind fair four-ticket A | Prefix-only M0-style four-ticket B |
| --- | ---: | ---: |
| Tested dates | 325 | 325 |
| Four-ticket exposures | 1,300 | 1,300 |
| Exact per-draw fair chance of any official prize | **0.1227314589619088** | **0.1227314589619088** |
| Fair expected prize-bearing draws over 325 | **39.89** | **39.89** |
| Actual draws with at least one award | **37** | **47** |
| Actual observed any-award rate | **11.38%** | **14.46%** |
| Draws where only this arm received an award | **32** | **42** |

The **+10 observed prize-bearing draws** for B correspond to +3.08 percentage points within these particular 325 historical holdout draws. There were **5 both-awarded** and **246 neither-awarded** draws. The paired **two-sided exact discordance p = 0.29541328065567013**; this is **not statistically significant** at 0.05. Crucially, the exact **theoretical IID probability is identical** by the matched disjointness constraint, so an observed gap cannot by itself imply an increased chance of winning.

**Independent four-block feature-signal diagnostic:** For each target, compare the exact prefix-scored real six-number draw with 16 fixed-seed pseudorandom fair six-number controls using the minimum of the four standardized hypothetical-repair gains. The observed averaged pairwise win/tie fraction was **0.5041346153846153** compared with the nominal fair reference **0.5** (difference **+0.004134615384615348**). Nominal conservative bounded-Hoeffding diagnostic is **1.0** (not significant); the fixed PRNG qualification above applies. This is *not* a calibrated forecast probability or a test of higher-order full-joint proper-score predictive quality.

**Decision: `EXPLORATORY_PREDICTIVE_NO_GO`.** The actual holdout results are compatible with random fluctuation; neither the strategy-level prize comparison nor the underlying four-feature preference test provides persuasive evidence of a predictive edge. The finding is also consistent with earlier v0.26 and v0.31 descriptive NO_GO results, while evaluating a different M0-inspired selector and endpoint.

**Scope limits:** This is not a strict independent prospective confirmation because the development team previously inspected historical outcomes and explored related models. This fold-specific heuristic is not the exact fully optimized v2.1 ticket selector; a v2.1 global-optimum study and a v2.2 historical-outcome study answer different questions. No source files, established future-only seals, ticket records, or existing reports were changed. Any further claim needs genuinely unobserved future outcomes, a preregistered rule, independent source verification, adequate statistical power and explicit multiplicity controls. Do not choose a new rule in hindsight based solely on the 47-versus-37 realization.
