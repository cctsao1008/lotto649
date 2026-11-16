# Issue #8 — Matched v0.25 covariance benchmark

This retrospective benchmark freezes the settings used by the original v0.25
calibration. The history ends at draw 115000103 on 2026-11-06 (2,497 draws).
All arms share the exact v0.16 registry and means, the legacy 300-ticket pool
(selector seed 649), the same four-ticket greedy and bounded swap search, and
the A–D exact main-ball 3+ coverage floor enforced by pairwise disjointness.
The standardized ridge is 0.25 throughout.

| Arm | Covariance | Standardization |
| --- | --- | --- |
| Legacy MC | 8,192 IID legal tickets, seed 649, sample covariance | Legacy sampled scale with theoretical Shape variance floors |
| Matched MC | Same MC covariance as Legacy MC | Exact population standard deviations |
| Exact | Exact 163-feature population covariance | Exact population standard deviations |

The primary comparison is the reduction in exact-model post-augmentation
distance when selecting with Exact instead of Matched MC. Evaluate both
portfolios in the exact metric, using their four tickets and the same history.
A positive value favors Exact on this descriptive objective. The comparison
against Legacy MC measures the combined effect of changing both covariance and
standardization. Each arm's reference-minus-selected gain is also evaluated in
its own metric to reproduce the prior legacy calibration.

Run the entire selection on 99 independent fair 6/49 histories of the same
length. Within each replicate all three arms share the same history. Draw
seeds are 1650–1748, matching the old calibration, and pool and covariance
stay fixed. Record each arm's choices and report the primary observed effect,
the paired null distribution and the exploratory upper-tail proportion
with add-one correction. Wilson intervals describe simulation uncertainty
in the underlying tail proportion conditional on this benchmark configuration.
The two additional comparisons are descriptive diagnostics. No settings are
chosen by maximizing historical results. The estimand is in-sample
distribution repair. The benchmark includes no later draw outcomes or test of
future prediction.

Reproduce from the repository root with Python and NumPy:

```bash
OPENBLAS_NUM_THREADS=1 python -m research.lotto7x7.matched_covariance_v025
```

## Results — 2026-11-16

| Arm | Exact metric after augmentation | Reference gain in own metric | Mean IID reference gain in own metric | Exploratory p for reference gain |
| --- | ---: | ---: | ---: | ---: |
| Legacy MC | 0.0564414493743 | 0.0001342891091 | 0.0009564059157 | 1.00 |
| Matched MC | 0.0564414493743 | 0.0001335511473 | 0.0009594205841 | 1.00 |
| Exact | 0.0564414493743 | 0.0000896528933 | 0.0009610243994 | 1.00 |

All arms chose the same historical four-ticket portfolio. The observed
primary gain is zero. Across paired IID histories, Exact scored lower
30 times, tied 59 times, and scored higher 10 times. The mean paired gain
was +0.0000205452 and the empirical 2.5th–97.5th percentiles were
−0.0000574082 to +0.000258540. For the observed zero gain,
89/99 replicates were at least as large, giving add-one p=0.90
(Wilson 95% interval for the underlying tail fraction: 0.824–0.944).
All three observed within-arm reference gains were below all 99 IID gains
in their corresponding arm. The Legacy MC result reproduces the previously
recorded portfolio, objective, and IID mean gain.

The exact covariance has numerical rank 142 of 163 at tolerance 1e-10
before ridge in standardized coordinates. The 8,192 MC draws had zero
variance in 18 rare L3 indicators. With the exact scaling, the relative
Frobenius error of the MC covariance was 0.530. These diagnostics show
finite-sample differences in the estimated reference. This benchmark has
one fixed MC covariance sample and one cutoff, and the bounded optimizer
does not certify a global minimum. The evidence decision is
`NO_EVIDENCE` for an unusually large descriptive gain in this configuration.

Every real and simulated portfolio matched the reference exact main-ball
3+/4+/5+/6 outcome counts: 1,040,096 / 55,216 / 1,036 / 4 of 13,983,816.
The machine-readable [results with all paired replicates](../../../artifacts/v025_exact_covariance/v025_matched_covariance_20261116.json)
also include hashes of the exact moments, the legacy pool, and the
historical rows used by the optimizer. The hash of the entire annual CSV
inputs is separate and may change when new rows are appended.
