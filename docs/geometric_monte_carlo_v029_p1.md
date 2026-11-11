# v0.29 P1 — Exact Geometric Probabilities and Monte Carlo Cross-Checks

Tracking: [Issue #31](https://github.com/cctsao1008/lotto649/issues/31). Requires the fixed [P0 geometry feature catalog](geometric_monte_carlo_v029_p0.md). This milestone is **strictly outcome-blind** and does not read historical draw files, modify purchased tickets, fit models, select next-draw numbers or claim an edge over fair IID.

## Exact fair 6/49 probability

The experiment draws six **distinct** cells from the 7×7 grid, uniformly over the exact `C(49,6)=13,983,816` legal subsets.

For each predeclared region of `m` lattice cells, the count of occupied cells `K` is hypergeometric:

```text
P(K=k) = C(m,k) * C(49-m,6-k) / C(49,6), for k = 0..6.
E[K] = 6*m/49.
```

These probabilities are **exact integer rational counts**, not Monte Carlo estimates. A continuous radius-three circle's Euclidean area is not the null reference. For P0's centered inclusive radius-three lattice disk, `m=29`, `E[K]=3.55102040816`, and `P(K>=4)=0.52652366135`.

## Exact pairwise joint distribution

For any two regions A and B, partition the 49 cells into `A intersection B`, `A only`, `B only`, and `neither`, with counts `a,b,c,d`. Then:

```text
P(K_A=i, K_B=j)
  = sum_{h=0..6} C(a,h) C(b,i-h) C(c,j-h) C(d,6-i-j+h)
    / C(49,6).
```

Invalid binomial arguments contribute zero. The implementation computes all 45 distinct region-pair distributions as exact 7×7 **integer numerator** tables with denominator 13,983,816. Every row and column marginal must equal its matching exact single-region hypergeometric law. Exact within-draw occupancy covariance is also reported:

```text
Cov(K_A,K_B) =
    |A intersect B| * (6/49)
  + (|A|*|B| - |A intersect B|) * (6*5)/(49*48)
  - (6*|A|/49)*(6*|B|/49).
```

These features are correlated. It is incorrect to multiply the marginal region probabilities to obtain a joint probability.

## Seeded independent Monte Carlo control

The secondary control generates each six-number ticket with `random.Random(seed).sample(range(49),6)`, using distinct cells within each draw and independent sampling across draws. Default: 20,000 trials, seed 649. Compare each of the ten exact region histograms and the **predeclared** pair `disk_r3 x border` against synthetic occupancy counts.

The printed maximum standardized error across 70 correlated single-region bins is exploratory and **not** a multiplicity-adjusted p-value. The purpose is numerical cross-checking; exact theory takes precedence when available. The planned P2 whole-pipeline Digital Twin is a separate experiment that must recalibrate any post-selected feature, mapping, parameter or portfolio-search procedure.

## CLI and Windows verification

```powershell
cd E:\lotto649
$py = ".\.venv\Scripts\python.exe"

# Print a concise theory + Monte Carlo summary
& $py -m lotto7x7.geometry_theory_v029 --trials 20000 --seed 649

# Create the full non-overwriting JSON report (45 exact pair tables)
& $py -m lotto7x7.geometry_theory_v029 `
    --trials 20000 --seed 649 --out results/v029_p1_geometry_theory.json

# Exact theory only, no simulation
& $py -m lotto7x7.geometry_theory_v029 --theory-only

# Regressions
& $py -m unittest discover -s tests -p "test_geometry_theory_v029_p1.py" -v
```

The full JSON report references the **exact SHA256 of the fixed P0 feature catalog**, making it possible to reject mismatched geometry definitions. Output paths are opened in exclusive-create mode, never overwriting previous reports.

## P1 acceptance

- [x] All ten exact discrete-region hypergeometric laws normalize to 13,983,816.
- [x] All 45 exact pairwise joint occupancy tables normalize and return the correct marginals.
- [x] Analytic covariance agrees with joint-table expectations.
- [x] Fixed centered radius-three disk `m=29` and its exact `P(K>=4)` are regression fixtures.
- [x] Reproducible independent six-of-49 Monte Carlo checks with fixed seed and explicit uncertainty limitation.
- [x] Nine no-network regression tests and a separate main-branch CI verification.
- [ ] P2: Real historical convergence trajectories, whole-pipeline IID surrogate tests, layout sensitivity and statistical multiplicity handling.
- [ ] P3: Examine whether structural geometry adds nonredundant information over existing v0.28 Shape/Sum/Log/Joint repair.

**Scientific boundary:** historical deficits may be quantified or counterfactually minimized, but fair IID future draws do not preferentially "catch up."
