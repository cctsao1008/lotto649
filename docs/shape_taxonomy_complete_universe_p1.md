# Lotto649 — P1 Batch 2: v0.15 Shape Taxonomy + v0.16 Exact 6/49 Universe

## Goal and separation of scientific claims

This batch **does not presume GMM has exactly 12 natural groups**. P0 showed
that 8-dimensional GMM selected `K=12` for both the 2,497 historical samples
and one independently drawn fair-IID cohort, and that selected K changed with
MDS dimensions. P1 therefore separates:

1. **Exploratory, observed-sample KDE local-density modes** using the original
   v0.13 D4+centroid-translation optimal matching distances.
2. **Frozen, exact, deterministic graph shape taxonomy** defined on the original
   7×7 board, with mathematically exact combinatorial probabilities under
   uniform legal 6/49. This taxonomy is separate from KDE/GMM and need not
   reproduce their assignments.
3. **Exact D4+translation canonical families**: all historical shapes are
   assigned immutable canonical keys. Each observed canonical family's
   theoretical number of distinct legal placements is calculated by orbit
   counting. All *unseen* canonical families are included together as the
   exact remaining mass; P1 does **not** individually enumerate every unseen
   canonical family into a giant file.

All 6/49 combinations are equiprobable under a fair independent draw, even if
some **shape classes** are intrinsically more common because they contain more
combinations. A graph taxonomy distribution is not a probability forecast.

## Fixed hierarchy

The six selected cells are vertices on a 7×7 board. Two distinct vertices
share an eight-neighbor edge if their row/column coordinate differences are
both at most one.

- **L1 `Ck`**: number of connected components in the six-vertex graph (`k=1..6`).
- **L2 `Pa-b-...`**: decreasing list of component sizes, an integer partition
  of six (e.g. `P3-1-1-1`). This is D4-invariant.
- **L3 `Pa-b-..._Oo_Dd`**: the L2 partition plus the number of horizontal/vertical
  adjacency edges (`O`) and diagonal adjacency edges (`D`). All are D4-invariant.
- **L4 `S<16-hex>`**: SHA256 prefix of an exact D4 + integer-translation canonical
  six-point shape key. A family contains all possible translations, rotations
  and mirror images that fit the finite original 7×7 board. The exact orbit
  size is a sum of valid placements for distinct D4 normalized orientations.

L1-L3 constitute the frozen **exhaustively enumerated** taxonomy; L4 orbit
probabilities are exact for *historically observed* families, with a single
remainder probability for every unobserved family.

`shape8-graph-partition-cardinal-diagonal-v1` is the fixed classifier version;
its SHA256 is saved in results. This representation is designed for stable
counting and explainability, not learned from lottery outcomes. The relative
2D grid itself remains a researcher-defined number mapping.

## v0.15 observed sample KDE maxima

Use the cached 2,497×2,497 v0.13 `shape` distance matrix (not MDS). For each
bandwidth σ∈{0.35,0.55,0.8,1.1}, compute the leave-one-out dimensionless
Gaussian radial similarity score:

`score_i(σ) = Σ_{j≠i} exp(-d(i,j)^2/(2σ²))/(N-1)`.

For each neighborhood radius r∈{0.70,0.75,0.80}, **a local maximum among the
observed points** has no within-r neighbor with greater score (deterministic
lower-index tie break). These are **discrete empirical local maxima**; neither
analytical continuous-density stationary points nor normalized Gaussian PDFs.
There is **no `max-peaks` cap**. Record all 12 parameter combinations and
report class coverage for a separately fixed nearest-mode assignment radius
(0.85). A nearest-mode basin does **not** constrain all within-class pairwise
distances; report actual diameter, mean pairwise distance and mode persistence
across the 12 scales. Different radius thresholds may produce very different
mode counts, which is itself a crucial research result.

## v0.16 exact combinatorial fair baseline

Every legal combination is generated once from `itertools.combinations(
range(1,50),6)` and classified by the **same** vectorized NumPy fixed graph
classifier used for historical draws. The enumeration must process exactly

`C(49,6) = 13,983,816`.

Memory is bounded by `--universe-batch-size` (default 120000). No C++ compiler,
Numba, JIT, internet or CUDA is required for enumeration. CUDA is optional for
Gaussian-KDE evaluation of the **historic cached** distance matrix.

For each L1/L2/L3 class, exact theoretical count `M_k` and probability
`p_k=M_k/13,983,816` give expected history count `N*p_k`. No simulated
combination is used to estimate `M_k`.

For each frozen level, the Pearson statistic is calibrated against a simulated
multinomial null with **predeclared exact p_k** and `N` historical draws:

`X² = Σ_k (observed_k - N p_k)² / (N p_k)`.

`empirical p = (1+#null_X²>=observed_X²)/(1+R)` with default `R=5000`.
This remains correctly calibrated in sparse expected-count categories where
an asymptotic chi-square approximation is inappropriate. The three global
nested-level p-values also receive Bonferroni adjustment. Per-class exact
binomial p-values are reported with BH-FDR across all levels; their dependence
and the methodology's exploratory post-selection history are explicitly
stated. No category frequency is called a genuine anomaly merely because one
post-hoc p-value is small.

## Outputs

A **single command** writes eight files with a common stem; all are protected
from overwrite. The input v0.13 NPZ and CSV are never changed.

| File | Content |
|---|---|
| `<stem>.json` | Complete reproducible audit, modes, fixed categories, tests |
| `<stem>.html` | Offline 7×7 shape gallery and analytical tables |
| `<stem>.npz` | Frozen per-draw graph labels, KDE scores and mode membership |
| `<stem>_draws.csv` | Each historical draw, six numbers and all four levels |
| `<stem>_catalog.csv` | Every observed canonical shape, orbit mass and history |
| `<stem>_theory.csv` | All exact 6+11+94 L1-L3 class counts and comparison stats |
| `<stem>_modes.csv` | 12 bandwidth × radius mode-sensitivity comparisons |
| `<stem>_universe.json` | Independently reusable exhaustive fixed-code baseline |

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-catalog-exact `
    --device cuda `
    --matrix results/v013_pairwise_similarity.npz `
    --assignments results/v013_pairwise_similarity.csv `
    --bandwidths 0.35 0.55 0.8 1.1 `
    --radii 0.70 0.75 0.80 `
    --assignment-radius 0.85 `
    --universe-batch-size 120000 `
    --mc-replicates 5000 `
    --seed 649 `
    --out results/v015_v016_p1_shape_catalog.json

Start-Process .\results\v015_v016_p1_shape_catalog.html
```

To rerun experimental bandwidth/radius settings while avoiding another exact
universe pass, choose a **new output stem** and add:

`--baseline-cache results/v015_v016_p1_shape_catalog_universe.json`

The cache is validated for version and 13,983,816 count. The output is still
an explicitly new file to preserve reproducibility.

## Checks and limitations

- New v0.15/v0.16 unit tests include a **completely independent** six-vertex
  BFS validator on 1,000 legal draws, D4/translation invariance, exact-orbit
  equivalence, invalid input, discrete local mode logic and null machinery.
- Full 13,983,816 enumeration was executed and its L1 counts matched the
  previously independently computed C++ enumeration: `[58620,669462,
  2775454,5093678,4166304,1220298]`.
- Full historical `shape` NPZ + CSV were used for a real-data integration
  analysis. No CNN retraining or old GPU backtests were repeated.
- The library deliberately **does not** claim a natural unique mode count,
  because it depends on the choice of shape metric, σ and neighborhood radius.
- Full theoretical L1-L3 classes have exact probabilities; KDE mode basins
  do **not** currently have exact full-universe mode probabilities.
- The historical list of observed exact families is **post-selected**. Its
  theoretical mass and complement are exact descriptions, not a valid new
  hypothesis test of special recurrence.
