# Lotto649 v0.25 — Historical Number-Deficit Compensation Foundation

Related issues: [#8](https://github.com/cctsao1008/lotto649/issues/8), [#9](https://github.com/cctsao1008/lotto649/issues/9).

## Status

**P0/P1 first increment only.** This implementation handles *49 number membership indicators*, source CSV integrity, a covariance-aware number-only discrepancy, four-ticket joint greedy selection, and comparison with the existing four-ticket reference. It **does not implement yet** full joint Shape/Sum/Gap theory, exact portfolio hit coverage, IID Monte Carlo calibration, walk-forward validation, or prospective assessment. Do **not** close #8/#9 based on this increment alone.

## Mathematics

For each legal 6/49 draw, let `f_i(x)=1` if number `i` appears, otherwise 0. The exact fair marginal probability is `p=6/49`; the exact two-number joint probability is `p2=(6*5)/(49*48)`. The 49-dimensional covariance has one zero eigenvalue because every draw contains exactly six numbers; on the 48-dimensional sum-zero subspace its eigenvalue is `a=p-p2`.

For historical `N` draws with number counts `n_i`, define an augmented history after hypothetical ticket set `T`:

```text
mu_i(T) = (n_i + count of number i among T) / (N + |T|)
D(T) = sum_i (mu_i(T) - 6/49)^2 / (a + ridge)
Delta(T) = D(empty) - D(T)
```

For fixed-size tickets, the residuals sum to zero, so the above equals the covariance-aware quadratic form with the Moore–Penrose inverse when `ridge=0` (or the ridge-regularized inverse with `ridge>0`).

A positive `Delta` means that treating the proposed tickets **as hypothetical additions to the recorded sample** would move the number marginal distribution closer to the exact mathematical baseline. This is **not** a claim the tickets are more likely to occur or win. Plain IID sample-size dilution also changes the discrepancy; selection performance must eventually be compared to uniformly sampled portfolios.

The bounded optimizer generates a seeded pool of distinct uniformly sampled legal tickets, chooses the best sequential four-ticket additions, then performs up to three improving single-ticket swap passes. It is **not** globally optimal over the 13,983,816-combination universe. It permits shared numbers across tickets because overlap is not automatically bad or good for every prize objective.

## Commands

From the repository root:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 compensation-audit `
  --csv-dir data/csv `
  --cutoff 2026-11-06 `
  --pool-size 1000 `
  --seed 649 `
  --out results/v025_deficit_foundation.json

& $py -m pytest tests/test_deficit_compensation_v025.py -q
```

Alternatively the foundation can run directly: `python -m lotto7x7.deficit_compensation_v025 --csv-dir data/csv`.

The JSON includes the input CSV fingerprint, cutoff and chronological range, each number's observed/expected/deficit/z-marginal, selected four-ticket portfolio and its before/after discrepancy, reference portfolio before/after discrepancy, and an explicit scientific limitation.

## Fixed comparison reference — no retroactive change

```text
A: 04 09 17 24 29 42
B: 10 13 14 27 28 34
C: 06 12 19 20 44 48
D: 03 07 18 33 40 47
```

This reference is evaluated, never modified. Historical data, v0.20 prospective freeze/seals and v0.24 outputs must remain unchanged.

## Next implementation dependencies

1. Extend feature registry and exact universe moments to Sum, Parity, Gap, L1/L2/L3 classes and selected joint bins (#8).
2. Calibrate historical deviation with IID simulations and finite-sample uncertainty; multiple-testing correction (#8).
3. Add exact multi-ticket threshold-hit coverage and a separate bonus-ball official award engine (#8).
4. Integrate compensation-only, coverage-only and hybrid multiobjective optimization with transparent trade-offs (#9).
5. Add no-leak chronological backtest and new prospective seals without touching existing frozen evidence (#8/#9).

**Interpretation guard:** an observed historical deficit is not a future probability adjustment, and no fair-draw jackpot advantage is implied.

## Second increment: exact L1/L2/L3 shape-deficit research

The **`shape-deficit-audit`** command consumes an already-generated, **complete** v0.16 exact universe JSON. It rejects partial universes and mismatched taxonomy rule/schema. It uses the original 8-neighbor L1 component / L2 partition / L3 cardinal+diagonal-edge classes, not a newly invented shape encoding.

```powershell
& $py -m lotto7x7 shape-deficit-audit `
  --csv-dir data/csv `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --cutoff 2026-11-06 `
  --pool-size 300 `
  --seed 649 `
  --alpha 4 `
  --out results/v025_shape_deficit.json

& $py -m pytest tests/test_deficit_compensation_v025.py tests/test_shape_deficit_v025.py -q
```

The named `--universe` artifact is produced by the existing v0.16/P1 exact enumeration workflow; it is **not** checked into Git by this work. Generate it using existing repository tooling if absent. Never substitute a partial or synthetic catalog as the scientific baseline.

For level `l` and its classes `c`, let `p_lc = exact_universe_count_lc / 13,983,816`, `H_lc` be the historical occurrence count, and `T_lc` be the number of candidate tickets in that class. The descriptive, ridge-like regularized categorical objective is:

```text
D_l(T) = sum_c [ ((H_lc + T_lc)/(N+|T|) - p_lc)^2 / (p_lc + alpha/N) ]
D_shape(T) = sum_l weight_l * D_l(T)
```

`alpha` is fixed from the historical count `N` for before/after comparability. The default family weights are `(L1,L2,L3)=(1,1,1)`, deliberately **exposed** and not claimed statistically independent. These nested features are highly redundant: this is a tunable descriptive *objective*, **not** a covariance-inverted omnibus significance test. A full joint model and calibrated p-values remain open in #8.

Candidate generation draws legal six-number combinations uniformly from the ticket universe, rejecting duplicate tickets; a finite 300-ticket pool is optimized using greedy additions and three bounded local one-swaps. The search is a **local heuristic** and is only reproducible given source files, seed, weights and exact universe version, not globally optimal. It returns four distinct tickets and does not require number-level disjointness.

Its JSON includes:
- Verified historical cutoff, SHA-256 of original CSV files and exact universe JSON, total sample count.
- Historical observed/expected/deficit for **every** L1/L2/L3 class, selected class counts, before/after class contributions.
- Selected portfolio and the fixed A–D reference, with separately calculated exact **main-ball** 3+/4+/5+/6-hit coverage from the existing `coverage_v06.exact_coverage` DP.
- Explicit notice that neither the compensation objective nor official prize-rank probabilities are established by main-ball coverage alone.

**Still pending:** sum/gap/parity joint feature registry, cross-family covariance, multiple-test IID null calibration, comprehensive Pareto optimization, historical walk-forward and prospective validation. Existing A–D bets and sealed predictions are immutable.

## Third increment: 3+ reference-coverage constrained mode

User's measured original A–D exact fair main-ball 3+ coverage:

```text
A–D reference >=3: 0.07437855303588091
Shape-only candidate >=3: 0.07358477828941686
Shape-only baseline score: 0.02215376
Reference shape score: 0.02170049
Unconstrained shape candidate score: 0.02116221
```

These fixed values are historical observations from an existing 2497-draw cutoff and are recorded as **regression/reference facts**, not predictions or an optimized future result.

Run **two separate** modes, keeping the default no-constraint mode for backward compatibility:

```powershell
# Original unconstrained Shape selection; default remains unchanged
python -m lotto7x7 shape-deficit-audit `
  --csv-dir data/csv `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --cutoff 2026-11-06 --pool-size 300 --seed 649 `
  --out results/v025_shape_deficit.json

# 3+ main-ball portfolio event must be at least the existing A–D reference
python -m lotto7x7 shape-deficit-audit `
  --csv-dir data/csv `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --cutoff 2026-11-06 --pool-size 300 --seed 649 `
  --coverage-constraint reference-3plus `
  --out results/v025_shape_deficit_coverage_constrained.json
```

The constrained selector currently uses the **sufficient** condition that all four six-number tickets are pairwise disjoint. All four such tickets have the same 3+ main-ball coverage as the A–D reference by label-exchange symmetry. The implementation verifies the actual result using the exact combinatorial DP, and raises if the floor is violated.

**Limitations:** This is **not** a complete constrained optimizer over overlapping tickets, nor a proof of globally optimal coverage/compensation. It may exclude feasible overlapping portfolios, and does not yet implement a full Pareto frontier. A 3+ floor does not constrain every other prize objective; those are reported independently. Four disjoint tickets have 24 distinct numbers. The baseline score is a descriptive historical metric, not predictive skill.

The selected JSON includes `selected.coverage_constraint`, `selected.meets_reference_3plus_floor`, `selected.reference_3plus_count_floor` and exact coverage `at_least` and `at_least_counts`. Original A–D remains unchanged.

Research checklist: #8/#9 remain OPEN pending complete covariance, multivariate selection, prospective validation, and more general coverage constraints.
