# Lotto7x7 v0.14 — Gaussian Shape Distribution & Quantification

## Objective

We no longer search for historical periodicity. Given the **unchanged** v0.13 pairwise distances among historical 7×7 images, study whether the image cloud has density peaks, whether a Gaussian mixture is a useful *descriptive* representation, and how many images belong to the resulting classes. Compare the same statistics against independently sampled **legal uniform 6-of-49** lottery images.

v0.14 *does not* attempt to forecast the next draw, infer lottery machine behavior, or assign higher odds to a particular ticket. Under IID 6/49, the probability of any specific ticket remains exactly 1/13,983,816 regardless of any image classification.

## Inputs and identity audit

Read **existing files only**:

- `results/v013_pairwise_similarity.npz`: 2,497 historical `draw_ids`, symmetric 2,497×2,497 `shape` distances, originally computed with exact 6! matching, D4 transform and continuous centroid translation.
- `results/v013_pairwise_similarity.csv`: same per-draw IDs, six main numbers and v0.13 threshold-group labels.

Join by `draw_id` and fail closed on missing or duplicate IDs, invalid six-number draws, asymmetric distances, nonzero diagonals, or nonfinite values. The historical 3,116,256 distance pairs **are not recalculated**. This eliminates historical matching cost.

## Method 1 — Gaussian radial KDE similarity, without assuming Euclidean embedding

For each cached distance `D[i,j]` and bandwidth σ:

`q_sigma(i) = (sum_{j != i} exp[-D[i,j]^2 / (2 sigma^2)]) / (N - 1)`.

The diagonal is excluded. Because D is calculated by minimization across rotations, reflections and translations, the quotient space need not admit a global Euclidean embedding, and this kernel need not be positive semidefinite. Therefore these quantities are **leave-one-out radial similarity scores**, not normalized Gaussian probability-density functions. Their scale changes with bandwidth. Compare across images only under the same σ and sample size.

Default σ values: 0.35, 0.55, 0.80, 1.10 shape-distance grid cells. The 0.55 score is used to rank proposed density modes. Peaks must be separated by `peak_radius=0.6`, and at most 24 are *shown*; that display cap is **not** the true count of natural modes. Samples are assigned to the nearest selected peak only if distance ≤ `assignment_radius=0.85`, otherwise `NOISE`. Mode members can be farther apart than 0.85; these **are not** complete-linkage clusters, so the HTML and JSON must not claim that every two group members are near.

The peak cap being hit is exported as `peak_selection_hit_cap`; the unassigned count is retained.

## Method 2 — Approximate shape-space embedding and Gaussian Mixture Model

Classical multidimensional scaling on `B = -0.5 J D^2 J` produces the top 8 *positive* eigenvector coordinates. Distances are generally distorted; report sample relative stress from observed vs reconstructed random pairs. A GMM with diagonal covariance and regularization (`reg_covar=1e-4`) is then fitted at fixed embedding dimension with K in `{1,2,3,4,6,8,12,16,24}`. BIC selects among only **these tested options**; this does not prove K is the natural number of shape categories.

Output per-class image count, mixing weight, mean posterior confidence, class medoid (an actual historic draw), and distances to that medoid. Posterior `P(class | embedded shape)` is **not** probability of a future winning ticket. All class labels and embeddings are stored in a separate small NPZ for downstream experiments without recalculation.

Sensitivity to MDS dimensions, Gaussian covariance choice, arbitrary 7×7 coordinate mapping, and random initialization limits the interpretation. Report `selected_at_search_ceiling` when K is the largest tested value.

## IID controls: refit rather than only reassign

For each replicate, independently:

1. Select a historical subset of `control_draws` draws from cached D.
2. Generate an equally sized legal fair 6/49 IID subset, **without conditioning on historical draws**.
3. Only for these **new IID samples**, compute shape distances using the existing v0.13 exact matching routine (CUDA if chosen).
4. Compare same-bandwidth leave-one-out KDE score mean, heterogeneity `cv`, and mean logarithmic score with equal N.
5. Fit fresh small-sample MDS and GMM models on both subsets and compare `BIC(K=1)-min_K BIC(K)` at the same embedding dimensions and K grid.

These comparisons are **descriptive paired contrasts**. Overlapping historical subsets and the small replicate count invalidate treating all image pairs or contrasts as independent; the output does **not** compute p-values. The GMM control uses K in `{1,2,4,8}` by default. To reduce runtime, the default uses 9 replicates of 128 images (a first-stage benchmark); increase `--control-draws 256` or `--control-replicates` for more detailed sampling. `--skip-control-gmm` allows a faster KDE-only IID pilot, but leaves GMM claims weaker.

## Run with existing results (GPU-first)

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-gaussian-gpu `
    --device cuda `
    --matrix results/v013_pairwise_similarity.npz `
    --assignments results/v013_pairwise_similarity.csv `
    --bandwidths 0.35 0.55 0.8 1.1 `
    --mode-bandwidth 0.55 `
    --peak-radius 0.6 --assignment-radius 0.85 --max-peaks 24 `
    --mds-dimensions 8 --gmm-components 1 2 3 4 6 8 12 16 24 `
    --control-draws 128 --control-replicates 9 `
    --batch-size 512 --gallery-count 8 --seed 649 `
    --out results/v014_shape_gaussian_cuda.json

Start-Process .\results\v014_shape_gaussian_cuda.html
```

### Outputs

- JSON: bandwidth metrics, KDE peaks/members, MDS stress, GMM BIC and class counts, paired fair-IID contrasts, known limitations.
- CSV: each draw's six numbers, KDE density score, nearest density mode or NOISE, approximate GMM class and posterior confidence.
- HTML: actual 7×7 medoid and example historical images per mode/GMM class.
- NPZ: compact embedding, bandwidths, density scores and posterior arrays for future analysis. **Not a copy of 27-MB v0.13 distance matrices.**

All four outputs must be distinct and not already exist; no old research outputs are overwritten. This is a **descriptive image classification study**, not a selection-edge claim. Only new v0.14 tests should be run, not all old suites or old GPU training.
