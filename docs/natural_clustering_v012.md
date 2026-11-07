# Lotto7x7 v0.12 — Image Similarity & Natural Clustering

## Objective

Stop looking for periodicity. Test whether all historical legal six-hot `7×7` lottery images form interpretable, stable groups **without presupposing exactly twelve classes**. Count members of each discovered class, inspect the actual historical members instead of only one representative, and determine whether apparent grouping also arises with fair independent 6/49 draws.

This is an unsupervised descriptive image study, **not a lottery forecasting, ticket ranking, or selection model**. There is no CNN retraining, GRU retraining, temporal period scan, or rerun of v0.8/v0.9/v0.10 backtests.

## Dependencies

`requirements.txt` adds `scikit-learn>=1.7,<2` for the maintained scikit-learn HDBSCAN and adjusted Rand/silhouette functions. `scipy` and `torch` are existing requirements. Installation once in the Windows Python venv:

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m pip install -r requirements.txt
```

## Run on historical data

```powershell
# Geometric, translation/rotation/reflection-invariant features.
& $py -m lotto7x7 natural-cluster-gpu `
    --device cuda --features geometry `
    --k-values 4 6 8 12 16 24 32 `
    --min-cluster-size 30 --min-samples 10 `
    --stability-repeats 2 --iid-replicates 9 `
    --iid-samples 10000 --gallery-count 10 `
    --seed 649 --out results/v012_geometry.json

# Frozen CNN Encoder from v0.10 + geometry; no network training.
& $py -m lotto7x7 natural-cluster-gpu `
    --device cuda --features fused `
    --model results/v010_deep_feature_cuda.pt `
    --k-values 4 6 8 12 16 24 32 `
    --min-cluster-size 30 --min-samples 10 `
    --stability-repeats 2 --iid-replicates 9 `
    --iid-samples 10000 --gallery-count 10 `
    --seed 649 --out results/v012_fused.json

# Optional: only 15 newly introduced test cases.
& $py -m unittest discover -s tests -p 'test_natural_clustering_v012.py' -v
```

Each command writes its own **new** `.json`, `.csv`, and standalone `.html` gallery; existing reports are never silently overwritten. To browse locally: `Start-Process .\results\v012_geometry.html` or `Start-Process .\results\v012_fused.html`.

## Distance and image representation

- `geometry`: 40-dimensional features inherited from v0.11: pairwise distances among six points; bounding box; 8/4-neighbor graph connectivity; row/column occupancies; component sizes. They are normalized in blocks and invariant to image translations, quarter-turns, and reflections. This is an intentionally **shape-based**, not number-specific, similarity.
- `cnn`: frozen v0.10 CNN per-image embedding; train weights are *not changed*. CNN raster features need not share D4 invariance.
- `fused`: concatenate the normalized geometry and frozen CNN embedding. Normalization is fitted once on historical images; fair IID fixed-region comparisons reuse it, whereas full-refit IID controls normalize their own synthetic images in the same protocol.
- `pairwise_gpu`: chunked Euclidean pairwise distances using `torch.cdist` on selected GPU; only the downstream clustering/quality estimation is delegated to SciPy/scikit-learn CPU algorithms. The resulting N×N matrix is symmetrized with a zero diagonal.

Even if visually meaningful, image clusters under the arbitrary 1..49 raster may be nothing more than ordinary random spatial arrangements.

## Three independently reported clustering methods

| Method | Determines number of classes? | How it handles difficult members | Representative |
|---|---|---|---|
| **HDBSCAN** (`sklearn.cluster.HDBSCAN`) | Yes, density-based extraction; it may find zero classes | Keeps uncertain draws as **NOISE**, never assigns them automatically | Exact within-cluster medoid, a historical image |
| **Ward agglomerative hierarchy** | Compares explicit K grid `4,6,8,12,16,24,32` | Every draw belongs to a class | Exact within-cluster medoid |
| **K-medoids** (alternating PAM-style implementation) | Uses Ward's best tested K for a like-for-like prototype comparison | Every draw belongs to a class | Medoid is the actual image defining that class |

**Ward's selected K is the best sampled silhouette in the specified candidate grid**, not evidence that the universe intrinsically contains that exact many classes. HDBSCAN classes are the closest to the user-requested **unknown and potentially nonuniform natural cluster count**, though they depend on `min_cluster_size`, `min_samples`, and representation.

**No giant transitive chain group** is used as a class; the old `>=5-of-6` similarity graph was a descriptive network, not a coherent partition.

## Quantitative reports

For each clustering method, JSON contains:

- `class_count`, `noise_count`, `coverage`, silhouette (computed on sampled classes only, excluding HDBSCAN noise), and `class_size_histogram`;
- a subsample **Adjusted Rand Index** (ARI) comparing the complete-fit labels and independent 80% refits. High ARI means *reproducible grouping under the chosen metric*, not an unexpected lottery pattern; fair IID can also achieve high ARI;
- each class's size/share and an actual **medoid** draw ID and six numbers, mean/p90/max distance from its medoid, ten nearest actual members and ten **most distant boundary members**;
- unassigned HDBSCAN noise count and representative noise examples;
- one assignment per historical drawing, with explicit `NOISE` when appropriate.

The CSV joins HDBSCAN, Ward and K-medoids assignments on `draw_id`, with the six original numbers, no translated field names. The offline HTML gallery displays the medoid, nearest historical members, boundary members, and class counts for all methods. Opening a class reveals members; it is not merely a 12-card prototype list.

## Two distinct fair IID controls

1. **FULL REFIT IID CONTROL** (`--iid-replicates`, default 9). Each replication generates exactly N independently drawn six-of-49 images, recomputes feature standardization, HDBSCAN class count, and Ward's best silhouette after **searching the same K grid**. It estimates empirical null p-values using `(1+#null statistics >= observed)/(B+1)`. For nine replications, the lowest possible p-value is 0.1, so **this is a screening comparison only**. Increase to 99 or more later if a striking effect survives initial review. The frozen CNN is **not** retrained on each null, so CNN/fused controls are conditional on the historical encoder.
2. **FROZEN MEDOID IID REFERENCE** (`--iid-samples`). Apply the frozen historical normalizer and nearest-observed-medoid rule to independent fair IID combinations. Compare `fixed_medoid_historical_share` **only** with `fixed_medoid_iid_share` within these same Voronoi regions. **Do not compare the Ward original group count directly to the nearest-medoid synthetic group count**; those are different assignment rules. No HDBSCAN out-of-sample prediction is claimed.

For HDBSCAN, report `hdbscan_class_count_empirical_p`, `hdbscan_coverage_empirical_p`, and `hdbscan_silhouette_empirical_p`. These are **different exploratory measures** and are not adjusted for choosing among all of them; do not interpret an isolated one as a confirmed discovery. HDBSCAN's silhouette excludes noise and may be inflated by discarding ambiguous images. Cluster stability is not a test against IID; compare like with like.

## Acceptance criteria and limitations

- Look for interpretable actual images within each class, **tight classes with clearly identifiable boundary examples**, and cluster assignments stable across D4-invariant geometry and frozen CNN representations.
- Ask whether historical clustering quality exceeds the **identical whole-pipeline** IID control distribution, not whether a clustering algorithm merely returns clusters.
- Change `min_cluster_size`, K search grid, feature representation or random seed only as new exploratory analyses; if doing so, expand the null selection process accordingly. This study uses already inspected historical data and is not a blind test.
- Fair 6/49 gives every exact six-number combination equal probability at each future draw. Classification/quantification is scientifically interesting but does not change that probability without genuine evidence of a non-IID drawing process.
