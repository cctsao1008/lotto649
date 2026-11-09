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
