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
OPENBLAS_NUM_THREADS=1 python -m lotto7x7.matched_covariance_v025
```
