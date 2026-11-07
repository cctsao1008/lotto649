# Lotto7x7 v0.11 — 7×7 Image Classification and Quantification

## Research question

**Ignore periodicity for this experiment.** Given every historical legal six-hot 7×7 bitmap, can we group identical or similar drawings into meaningful shape classes and count each class? Optionally use **the already trained v0.10 CNN encoder**, without running CNN/GRU training again. The aim is a reproducible taxonomy, per-class image gallery, and fair-IID reference distribution; this stage **does not select winning numbers or claim predictive power**.

## Run on the Windows research machine

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

# First: interpretable geometry-based classes (does not load the old .pt model).
& $py -m lotto7x7 image-classify `
    --device cuda --clusters 12 --features geometry `
    --iid-samples 10000 --seed 649 `
    --out results/v011_geometry_12.json

# Second: combine geometry with your pre-trained v0.10 CNN latent image features.
# This LOADS the checkpoint: no training or v0.10 period/motif search.
& $py -m lotto7x7 image-classify `
    --device cuda --clusters 12 --features fused `
    --model results/v010_deep_feature_cuda.pt `
    --iid-samples 10000 --seed 649 `
    --out results/v011_fused_12.json
```

Only if you need to verify *newly changed* code, run focused tests:

```powershell
& $py -m unittest discover -s tests -p 'test_image_classification_v011.py' -v
```

The code never silently replaces an existing output file. Each run creates three distinct outputs sharing the JSON stem:

- `results/v011_geometry_12.json`: source fingerprint, exact shape classes, topology counts, 5-of-6 similarity network, twelve learned prototype classes, IID reference fractions, representative six-number images, all draw assignments;
- `results/v011_geometry_12.csv`: one row per historical drawing with its class ID and six numbers, all header and values in English;
- `results/v011_geometry_12.html`: fully offline image gallery of the twelve class representatives, counts and fair-IID frequencies.

The fused run produces analogous files under its own stem.

## Three different definitions of image similarity

1. **Exact D4 plus non-wrapping translation equivalence.** Transform the six active pixels using rotations, reflections and translations, then canonicalize the relative six-point geometry. This is a true, nonoverlapping mathematical partition. The legal orbit size differs by geometry; its exact probability under fair 6/49 is `orbit_size / 13,983,816`.
2. **8-neighbor connected components.** Treat horizontally, vertically and diagonally adjacent pixels as connected. Count 1–6 components, neighboring pairs, bounding box dimensions. This is a very interpretable coarse morphological taxonomy, not a next-draw prediction.
3. **5-of-6 image-similarity graph.** Two draws share an edge if a common five-pixel relative pattern can be matched after D4 and translation. Connected graph components are *not* valid mutually similar classes: graph chains can merge thousands of unrelated-looking images.

These three notions must remain distinct. In particular, the graph's giant component should never be displayed as an intrinsically coherent image class.

### First descriptive counts for the 2,497-draw historical BIN

The historical `lotto649_history_v1.bin` was read directly from the connected GitHub repository for an initial independent count (draw IDs range `93000001` to `115000103`). The new CLI reads the verified image dataset and recomputes the same definitions on Windows.

- Exact six-number combinations: **2,497 unique**, none duplicated.
- Exact D4+translation six-point shape families: **2,490**; 2,484 singleton classes, five doublets and one triplet.
- 8-neighbor components 1–6: **15, 133, 492, 887, 730, 240** draws, respectively.
- 5-of-6 graph: **2,442 distinct pairs**, 1,939 draws with at least one neighbor; a large transitive component of 1,523 draws, hence **the graph is not the right final classifier**.

These are descriptive empirical counts, not a significance test or evidence of a winning advantage. The source is a shared third-party dataset, **not independent authentication against Taiwan Lottery originals**.

## GPU K-means prototype grouping

To make every image belong to a nonoverlapping **similarity class** with an actual representative, construct a 40-dimensional D4-invariant geometry descriptor:

- 15 sorted squared pairwise distances between six pixels;
- normalized bounding-box dimensions (2);
- 8-neighbor edge count, 4-neighbor edge count, 8-neighbor components (3);
- sorted row- and column-occupancy patterns (14);
- sorted connected-component pixel sizes, padded to six entries (6).

Normalize features using the observed images, balance the five groups so distance features do not dominate, then run seeded k-means++/k-means on CUDA. `--clusters 12` is a *research choice*, not the discovered true number of underlying lottery categories. If the classes are too broad or too fine, compare a few values under otherwise frozen settings; do not select the best class count based on winning-number performance.

Alternative `--features cnn` uses the 16-dimensional encoder output from `results/v010_deep_feature_cuda.pt` (CNN image representation **only**, no GRU temporal context). `--features fused` concatenates geometry with CNN image embeddings. The 16 learned CNN features are **not D4-invariant by construction**, and model weights were fitted on the original number-to-raster mapping. That is why the geometry and fused runs should be compared rather than assuming the latent representation is more meaningful.

For each prototype class, output:

- historical count and percentage;
- the nearest actual 7×7 drawing to that class centroid (medoid-like representative, not a synthetic picture);
- draw ID, actual representative numbers, 7×7 bitmap and sample draw IDs;
- average distance to centroid and morphology statistics;
- IID baseline share and descriptive observed/IID ratio.

### IID baseline limitations

Generate `--iid-samples 10000` independent, legal six-of-49 draws (exactly six different numbers in each); keep the **same fitted centroids and normalization** and measure how often random images belong to each class. This is a useful denominator for class-size comparisons, because a visually common geometry can be common *even in a fair lottery*. However, centroids were fitted to real historical images, so **the observed/IID ratios are descriptive and cannot be treated as valid discovery-adjusted p-values**. Classes are also correlated with the arbitrary sequential 1..49-to-7×7 mapping.

## Next question only after class counting

Do coherent prototype classes recur across separate subsets of known history? Is a class proportion different from fair IID after accounting for learned class boundaries and multiple testing? Only if such a departure replicates can we consider a feature-based ticket selector with out-of-sample evaluation and equal-ticket fair coverage. Classification alone does not make one legal six-number combination more likely to win a fair IID lottery.
