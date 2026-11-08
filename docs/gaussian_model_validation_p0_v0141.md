# Lotto7x7 Batch 1 / P0 — Gaussian Shape Model Validation (v0.14.1)

## Scope and deliverables

This **one P0 batch** implements and checks all four previously agreed high-priority issues in the v0.14 Gaussian-shape research. It uses `results/v013_pairwise_similarity.npz` and its aligned CSV, preserving original v0.13/v0.14 reports and CNN checkpoint. It does not retrain any model, download data again, or perform historical pairwise matching again.

1. **Equal-spec, independent legal 6/49 IID comparison.** Each matched trial randomly draws `--matched-size` distinct historical rows, while independently generating exactly the same number of fair 6/49 six-hot images. Historical and IID each get their **own classical MDS** embedding with the same dimension, **identical complete** GMM candidate-K grid and identical diagonal covariance, `reg_covar=1e-4`, `n_init`, `max_iter=200`, `init_params=kmeans` and trial seed. The IID pairwise `shape` distances are independently measured; they cannot be copied from historical distances. Nine 384-image trials by default. Optionally the same command produces one or more **full 2,497-vs-2,497 IID** model fits for direct full-N comparison (`--full-iid-replicates`, default 1). A full-size IID replicate costs similar to one v0.13 pairwise run.
2. **GMM membership stability.** Multiple specified seeds each create *two independent overlapping subsets* of historical data and fit GMMs with the full-reference selected component count; compute ARI and NMI only on the **intersection** of their sample indices, invariant to component-label permutations. The MDS axes are intentionally fixed from the full historical sample to hold one factor constant. Thus this is **transductive conditional stability**; it is not independently refitted embedding stability or prospective forecasting.
3. **MDS dimension sensitivity.** Once compute enough classical-MDS eigenvectors (default maximum 16). Evaluate prefixes 4, 8, 12, 16 using identical sampled pair indices when calculating relative stress. For each dimension, run full GMM BIC K-grid search with identical fitting settings, record selected K, silhouette, confidence and BIC gain over K=1. A D4-plus-translation shape distance need not be Euclidean; stress is an approximation diagnostic.
4. **Gaussian KDE bandwidth sensitivity.** Compute self-excluded radial similarity at sigma 0.35, 0.55, 0.8, 1.1, with mean/CV, Spearman ranking correlation and top-ten-percent set Jaccard against the smallest bandwidth. Also select separated high-density representatives under the *same* peak-radius, assignment-radius and cap across bandwidths. Reaching the cap does **not** imply that the selected number of prototypes is the natural number of density modes. Nearest-peak assignments do **not** guarantee low pairwise group diameter.

## One-shot Windows PowerShell CUDA run

Run once after pulling the merged version:

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-gaussian-validate `
    --device cuda `
    --matrix results/v013_pairwise_similarity.npz `
    --assignments results/v013_pairwise_similarity.csv `
    --dimensions 4 8 12 16 `
    --comparison-dimension 8 `
    --gmm-components 1 2 3 4 6 8 12 16 24 `
    --gmm-n-init 2 `
    --bandwidths 0.35 0.55 0.8 1.1 `
    --matched-size 384 `
    --matched-replicates 9 `
    --full-iid-replicates 1 `
    --stability-seeds 649 650 651 652 `
    --stability-fraction 0.8 `
    --batch-size 512 `
    --seed 649 `
    --out results/v0141_p0_gaussian_validation.json

Start-Process .\results\v0141_p0_gaussian_validation.html
```

The CLI produces JSON, long-form metric CSV, offline HTML and small auxiliary NPZ. It refuses to overwrite any pre-existing output. The historical `shape` matrix remains unchanged.

**Resource trade-off:** The 9 matched trials with 384 images each and the full-size legal IID run are additional exact six-point distance computations on *new IID images*, not repeat calculations on historical images. Run them on the existing RTX 4060. If a development smoke test is required on a CPU machine, set `--matched-size 24 --matched-replicates 1 --full-iid-replicates 0 --dimensions 2 3 --comparison-dimension 3 --gmm-components 1 2 3` and use a new output path. The main production command should not silently fall back to the smoke settings.

## Interpretation and research limits

- BIC is comparable for K inside each fixed sample and coordinate system. **Do not treat absolute BICs from different sample sizes or embeddings as directly comparable.** BIC gain relative to K=1 and selected K can be descriptive comparisons, not formal tests of lottery mechanism.
- Nine repeated historical subsamples overlap. Their differences are **not nine independent p-values**. A single full-size IID replicate cannot establish statistical significance; add more IID cohorts if the effect is large and scientifically worth validating.
- A mean GMM posterior of 0.75 is the model's membership confidence, **not classification accuracy or winning probability**. Stronger IID behavior under GMM need not reflect statistical anomalies.
- KDE outputs are dimensionless leave-one-out radial similarity scores, **not normalized Gaussian PDFs** on a quotient-shape space.
- Fixed MDS axes for stability intentionally isolate the GMM assignment step. Additional independent-MDS stability can be implemented only if this controlled test shows meaningful robust structure. There is no leak-free predictive claim.
- The 7×7 location of each integer is a **researcher-chosen mapping**. Changes of grid mapping can alter shape distributions without changing real draw probabilities.
- In an IID fair lottery, all legal six-number tickets have equal chance regardless of shape, history, Gaussian density, BIC or KDE category.

## Validation / acceptance

- New P0 focused tests validate input safety, equal-size and grid matched controls, independently generated legal IID, representative rank statistics, stability ARI/NMI, MDS diagnostics, CLI wiring, integration outputs and file alignment.
- Offline integration uses real 7×7 images and the cached v0.13 matrix, CPU for local environment. The **full production CUDA run is reserved for the user's GPU**.
- Development acceptance: all four P0 experiments implemented, new tests pass, docs and CLI merged, output files do not overwrite v0.13 or v0.14.
- Research acceptance: complete one-shot CUDA run produces and validates all outputs; subsequent interpretation distinguishes matched-small-N and full-N IID findings.
