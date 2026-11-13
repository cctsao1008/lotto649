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

Under the fair IID hypothesis, a real draw and its independent controls are conditionally exchangeable given the past. Accordingly the one-target fair null expectation is **0.5**. This comparison does not claim to estimate a normalized joint probability over 13,983,816 possible draws. The source-gated aggregate reports its observed mean and a **conservative bounded-martingale Hoeffding two-sided p-value upper bound**: `min(1, 2*exp(-2*n*(mean-0.5)^2))`. Targets are chronological; folds are disjoint in evaluation but training expands and overlaps. Never use an ordinary i.i.d. Gaussian t test on overlapping observations.

### Limitations and pass criteria

Engineering PASS requires (a) identical source/draw prefix and theory hashes, (b) independent special/main CSV consistency, (c) 13 complete nonoverlapping holds with training ending strictly before each hold, (d) identical exact fair official award probabilities at all folds, (e) reproducible seeded candidate streams and winner validity, and (f) all score controls in [0,1].

Predictive research is reported **NO_GO** unless a positive effect passes its respective exploratory p<0.05 threshold; even if both pass, do **not** declare confirmed predictive value due to multiple tested models/outcomes and historical strategy exploration. A later *new future-only pre-registered protocol* with independent unobserved events is mandatory before any confirmation. Do not overwrite the existing prospectively sealed v0.19, P3f, P5A or research artifacts. Under IID fair 6/49, higher repair ranks cannot increase a specific ticket's chance of being drawn.

### Running

```powershell
python -m unittest discover -s tests -p 'test_strategy_validation_v22.py' -v
python -m lotto7x7.strategy_validation_v22 --csv-root data/csv --bin-root data/bin --out results/v22_walkforward_exploratory.json
```

Output JSON is write-once. `results/` is a generated artifact directory, not input training data. The CI archives the JSON and fails closed if any source gate or coverage parity assertion fails.
