# v0.31 H2 — Causal Incremental Geometry vs Compressed M0 Features

Research: [#31](https://github.com/cctsao1008/lotto649/issues/31). Evaluation: [#32](https://github.com/cctsao1008/lotto649/issues/32).

## Exact scientific boundary

Under fair IID 6-of-49 draws, the conditional probability that each fixed number occurs next is exactly `6/49`, regardless of historical Sum, Shape, Log Product or geometry. A past-dependent predictive model can be validly tested for departures, but an exploratory low p-value does not automatically justify a physical compensation mechanism or ticket-selection edge.

The v0.29 P3A collision certificate proves that pair geometry is not determined by the *coarse* original `(L3, Sum, LogBin)` **for the same ticket**. A more recent exact proof also shows historical pair geometry cannot generally be recovered from 49 accumulated individual-number frequencies. Both facts are about feature structure, not next-draw prediction. Full raw six-hot draw histories already determine all engineered features exactly.

## Frozen four-arm ablation

**Every arm predicts the exact same target:** the 49 per-number inclusion probabilities for the next main-six draw. There is **one** previous-draw feature lag; expanding ridge training uses `(feature(draw[i-1]), binary-49(draw[i]))` pairs only through `i <= t-1` when predicting actual draw `t` (zero-based). No target draw contributes to its own features, training outcomes or hyperparameter selection.

| Arm | Predictors | Model | Notes |
| --- | --- | --- | --- |
| `iid` | no history | analytical constant 6/49 | Exact fair reference. |
| `m0_compressed` | original L1 component count, exact L2 component sizes, L3 cardinal/diagonal edges; exact Sum; v0.27 fixed integer LogBin=0.5; Sum×LogBin | fixed 13-dimensional regularized linear map, including intercept | **Compressed M0-derived representation**, *not* original M0 four-block counterfactual-maximin selector; injective on original L3 classes, but not an arbitrary one-hot lookup over all cells. |
| `geometry` | original sequential 7×7 three non-L3-determined pair descriptors, exact fair mean/SD and ridge=0.05 whitener | fixed four-dimensional regularized linear map, including intercept | Reuses frozen v0.29 structure without recomputing features from future data. |
| `combined` | compressed M0 features plus three geometric descriptors | fixed 16-dimensional regularized linear map | Tests marginal incremental value of geometric predictors over the compressed category baseline. |

Fixed linear ridge regularization `alpha=64` is shared for every fitted arm; no hyperparameter search, feature choice, or historical model tuning is performed. Sum uses its exact fair finite-population mean `150` and SD `sqrt(1075)`; LogBin uses the original integer-threshold bins and v0.27 analytically calculated log-product mean and SD. Graph partition dimensions and edge counts use fixed numerical scaling. No history-fitted standardization or vocabulary selection.

The raw 49 predicted marginals are projected onto the capped simplex `{p in [0,1]^49, sum(p)=6}` (numerical epsilon for logs). This hypersimplex is the convex hull of six-of-49 indicator vectors: the output is consistent with *some* legal-six distribution but **does not identify a normalized joint law over 13,983,816 tickets**.

## Primary and uncertainty

- **Primary endpoint:** `mean_Brier(m0_compressed) - mean_Brier(combined)`; positive means the combined model performs better than the compressed-category baseline. A per-number Brier is a **strictly proper score for individual number inclusion probabilities**, not joint ticket probabilities.
- **Secondary contrasts:** `IID - M0`, `IID - Geometry`, `IID - Combined`, where positive means an improvement over the exact fair marginal benchmark. Report Bernoulli log losses descriptively as a second proper marginal score.
- For observed data and every matched synthetic full 6/49 IID history, rerun **all** features, expanding fit, prediction origin and final scoring, rather than calibrating scores with a fixed model. Compute one-sided upper-tail empirical Monte Carlo p-values `(1 + null>=observed)/(R+1)` for each improvement and a familywise maximum test across four predeclared null-centered, null-SD-standardized improvements.
- With `R=19`, Monte Carlo p-value resolution is `0.05`; with `R=39`, it is `0.025`. Null-replicate SD estimation and model-capacity differences remain important limitations. Report effects, not p-value shopping.
- The 2,498-draw historical archive through 2026-11-10/#115000104 is already explored. This **must be labeled `EXPLORATORY_ONLY_NO_PROSPECTIVE_PREDICTIVE_GO`** regardless of observed p-values. A prospective inference requires a genuinely pre-outcome independent registered experiment and time-separated observations.

## Execution

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'

# Read-only regression
& $py -m unittest discover -s tests -p 'test_incremental_geometry_v031.py' -v
if ($LASTEXITCODE -ne 0) { throw 'v0.31 H2 regression failed' }

# Reproducible audit report, under the ignored results tree
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
& $py -m lotto7x7.incremental_geometry_v031 `
    --csv-root data/csv --bin-root data/bin `
    --start-draw 500 --step 10 --iid-reps 39 --seed 649 --ridge 64 `
    --out "results/v031_h2_incremental_$stamp.json"
if ($LASTEXITCODE -ne 0) { throw 'v0.31 H2 audit failed' }
```

CI checks source audit, full history score report, strict output no-overwrite, frozen shape/grid/geometry mappings, future mutation invariance, IID repeatability, fairness baselines and six-marginal conservation. The report includes the fixed configuration SHA256, all per-target proper scores, synthetic null contrasts, exact source archive fingerprints and explicit scientific limitations. It leaves purchased tickets, M0 candidate manifest, original P3f/P5A seals and history inputs untouched.

## Interpretation and next action

If a significant increase is seen retrospectively, do **not** tune or add new shapes against the same historical targets. Check training sample sparsity and complexity, compare consistent log-loss direction, inspect the IID null, and require future unseen and properly registered evidence. An exploratory `NO_GO` supports keeping the fair baseline rather than spending more real-money tickets on an unproven next-draw signal.
