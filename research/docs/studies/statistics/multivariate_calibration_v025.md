# Issue #8 P1 — Finite-sample covariance and entropy calibration

Freeze the original 2,497-draw source through draw 115000103, exact
163-feature registry and population covariance, and standardized ridge
0.25. The primary upper-tail statistic is N times the ridge-whitened
squared distance of the feature mean from exact fair theory. Auxiliary
statistics are the maximum absolute marginal standardized residual and
Shannon entropy (bits) for the pooled label, unordered label-pair,
adjacent sorted gap, and L1/L2/L3 class distributions.

The number and unordered-pair probabilities are uniform by exchangeability.
Each draw contributes 6 labels and 15 distinct unordered pairs, with their
within-draw dependence preserved in the replay. The exact fair probability
for each adjacent rank gap g is C(49-g,5) / C(49,6), 1 <= g <= 44.
Collapse the pair with fixed gap at the specified rank into one value
and shift later values down by g. This bijects the constrained six-number
draw with five ordered values in 1..49-g. Every draw contributes five gaps.
L1/L2/L3 use the checked v0.16 tables.

Use full history and four consecutive blocks with index boundaries
floor(k*N/4). Simulate 1,999 independent fair 6/49 histories of length N
(seed 7649, one seeded NumPy generator). Recompute all endpoints over
the same segments. Use add-one upper tails for distance and maximum
absolute residual, and doubled smaller add-one tail (capped at one)
for entropy. Adjust the combined 40 exploratory tests by Bonferroni.
The simulation budget resolves the 0.05 familywise threshold for both
one-sided and two-sided endpoints. A 399-replay pilot used the same seed
and identified the resolution issue before inference.
The null mean entropy estimates finite-sample plug-in bias relative to
population entropy. Report each observed-minus-null-mean difference,
null 2.5% and 97.5% quantiles and Wilson simulation uncertainty for the
one-sided tail fractions. The primary descriptive decision is scoped
to this previously inspected archive.

Use a disjoint seed (8649) and 199 independent fair histories as a
calibration check of the full-history primary distance at nominal 5%.
Report the rejection frequency and Wilson interval as a diagnostic.
The reference null is shared and finite, so the interval is conditional
on that reference. Do not apply a chi-square approximation for rare or
nested features.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python -m lotto7x7.multivariate_calibration_v025
```

## Results — 2026-11-16

The full-history primary distance is 143.660163, with exploratory upper
tail p=0.089 before the combined correction. The maximum absolute
standardized residual has raw p=0.032. The smallest adjusted p among all
40 tests is 0.44 for full-history L1 component entropy (raw p=0.011).
None meets familywise 0.05. In the independent size diagnostic, 9 of
199 fair histories are rejected at nominal 5% (observed 4.52%, Wilson
95% interval 2.40%–8.37%, conditional on the simulated reference).

This is a finite-sample descriptive audit of a previously inspected
history. The output estimates plug-in bias by matching the sample size
and within-draw dependence. See the [archived calibration results](../../../artifacts/v025_exact_covariance/v025_multivariate_calibration_20261116.json)
for all time blocks, 40 corrected endpoints and 163 ranked residuals.
