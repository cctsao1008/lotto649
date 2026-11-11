# Lotto649 v0.29 P2A — Geometric Convergence Signature

Tracking: [Issue #31](https://github.com/cctsao1008/lotto649/issues/31). Depends on [P0 geometry definitions](geometric_monte_carlo_v029_p0.md) and [P1 exact fair probabilities](geometric_monte_carlo_v029_p1.md).

**Scope:** audit whether the predeclared 7×7 **region-occupancy** distributions in the validated historical archive show more divergence from exact fair 6/49 than an equally long independent synthetic history. This is exploratory research, not a predictive forecaster, a new ticket selector or prospective evidence.

## Read-only source gate

The CLI validates original history BIN, image BIN, their SHA256 entries in `data/bin/lotto649_source_manifest.json`, all flat annual CSV rows, the matching six-pixel images, chronological dates, and first/last draw IDs using the existing exporters and temporal dataset parser. Any mismatch aborts. The project current historical record has 2,498 draws through 2026-11-10/#115000104; the method automatically reads the current validated manifest cutoff.

The historical archive has third-party origin and later corroborated additions; matching files and hashes are **not** independent authentication from the lottery operator.

Nothing writes to the data directory, preregistration seal, M0 candidates or private paid-ticket ledger.

## Fixed search family (P2A protocol v1)

- **10** original P0 fixed geometric region occupancy features; regions are six-without-replacement lattice-cell counts `K=0..6`.
- **4** fixed layout mappings: canonical `sequential`, `snake`, and seeded random bijections `random:649` and `random:650`.
- Nested chronological prefix lengths: `100, 250, 500, 1000, 1500, 2000, N`, retaining only values `<=N`.
- **Primary endpoint:** the global maximum of `abs(z)` across every region, mapping and prefix, with `z = sqrt(n)*(observed_mean - fair_mean)/sqrt(fair_single_draw_variance)`.
- **Secondary endpoint:** the global maximum single-region histogram total-variation distance `TV = 0.5 * sum_{k=0..6} abs(observed_k/n - fair_P_k)`, across the exact same search family.
- The theoretically exact `fair_P_k`, means and variances are inherited from P1 without fitting historical data.

The prefix trajectories and complete seven-bin histograms for all maps/regions are recorded. The maximum statistics, not a hand-picked attractive class or chart, determine the predefined tests.

## Monte Carlo Digital Twin — whole-search null calibration

For each of `R` Monte Carlo replicates:

1. Generate **N independent** uniformly random six-number combinations without replacement within each draw. This is performed with an independent NumPy row-wise random permutation of 1..49 followed by the first six numbers.
2. Apply every **same** predeclared mapping and fixed region used in the historical analysis.
3. Recompute every prefix, all 10 region histograms, the region mean residuals, and the two global maxima.
4. Compare each actual global maximum to the replicate maxima: `p = (1+#(null_max >= actual_max))/(R+1)`.

Each raw p-value therefore accounts for searching the **40 region×layout views across all prefix lengths**, including highly dependent snapshots, rather than reporting a naive isolated-bin z p-value. Two endpoints are further Bonferroni-corrected with `min(1,2*p)`.

Default `R=199`, seed `649`: smallest resolvable raw p-value `1/200=0.005`. Increase replicates using a **new output file name**, and explicitly document a change in precision rather than treating seed mining as a scientific gain.

**Limitations:** earlier research already examined this historical dataset; selection across historical research versions is not accounted for. Any small p-value remains **exploratory** until independently frozen future draws replicate the effect. Neither the sample mean convergence nor an observed shortfall implies the next lottery draw has compensatory forces.

## Windows PowerShell

```powershell
cd E:\lotto649
$py = ".\.venv\Scripts\python.exe"

# No-network synthetic unit suite
& $py -m unittest discover -s tests -p "test_geometry_convergence_v029_p2a.py" -v

# Validate all historical records and emit a new, non-overwriting report
& $py -m lotto7x7.geometry_convergence_v029 `
    --csv-root data/csv `
    --bin-root data/bin `
    --iid-reps 199 `
    --seed 649 `
    --out results/v029_p2a_convergence_20261111.json

# Print the two familywise-calibrated endpoints and their adjustment
& $py -c 'import json; j=json.load(open("results/v029_p2a_convergence_20261111.json",encoding="utf-8")); x=j["whole_pipeline_iid_control"]; print(j["source_gate"]["status"],j["number_of_draws"]); print("primary",x["primary_empirical_p"],"adjusted",x["primary_bonferroni_two_endpoint_p"]); print("secondary",x["secondary_empirical_p"],"adjusted",x["secondary_bonferroni_two_endpoint_p"])'
```

The report must go beneath the Git-ignored `results/` directory, using exclusive-create mode. An existing report is never overwritten. All source identities, mapping IDs, seed, statistical endpoints and exact P0 schema SHA256 are preserved.

## Milestone acceptance

- [x] Separate exact-IID and historical views with no pseudo-label forecasts.
- [x] Fail-closed source audit of annual CSV, history+image BIN and source manifest.
- [x] Fixed region/mapping/prefix search specification with reproducible source fingerprint.
- [x] Whole-search IID recalibration of maximum standardized mean residual and histogram TV.
- [x] Bonferroni handling of two predeclared endpoints and explicit minimum p resolution.
- [x] Deterministic synthetic unit tests including future-row perturbation independence.
- [ ] **P2B:** extend the audited family to graph topology/pair-distance structure; quantify overlap and conditional redundancy against P0/P1 and existing v0.28 Shape.
- [ ] **P3:** design a multivariate counterfactual repair score only if it retains nonredundant geometry; any predictive claim requires an independent prospective seal.

This protocol explicitly reports `EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE` regardless of retrospective p-values.
