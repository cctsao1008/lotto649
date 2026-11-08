# Lotto7x7 v0.13 — GPU Pairwise 7×7 Image Similarity

## Goal and scope

The previous v0.11/v0.12 shape categorization grouped actual 7×7 six-hot
lottery images, but both standard geometric features and a frozen CNN encoder
mostly sorted images by connectivity. In v0.13, **do not first decide how many
classes exist**. Instead, exhaustively calculate the geometric distance for
each unordered pair of historical images. For the current 2,497 draws there
are exactly **3,116,256 distinct pairs**. Derive bounded-diameter image groups
from those measured distances and compare their distributions with genuine
uniform six-of-49 random images.

No CNN/GRU training, no periodicity searching and no ticket selection. No
code touching the historical data/BIN or v0.8–v0.12 output files.

## Mathematical definition: four distances

Represent each image as six occupied positions `P={p1,...,p6}`, with integer
pixel row/column in `[0,6]`. Each pair uses **the exact 6! = 720 one-to-one
assignments** with squared Euclidean transport, not a Sinkhorn approximation.
For a transformation `T` on the second six-point image and any permutation
`π`, the squared assignment cost is:

`C(P,TQ) = min_π Σ_i ||p_i - T(q_{π(i)})||² / 6`.

Distances are the positive square roots (units: **grid cells per point RMS**):

1. **position**: `sqrt(C(P,Q))`. Retains original row and column locations;
   no reflection, rotation or translation. Same six numbers is distance zero.
2. **d4**: `min_{T in D4} sqrt(C(P,TQ))`. Allows precisely four quarter-turn
   rotations and their mirrored variants about the board center `(3,3)`.
   Still does **not** move images to match their centroids.
3. **shape**: `min_{T in D4,a in R²} sqrt(C(P,TQ+a))`. Continuous centroid
   alignment is the least-squares-optimal translation for **every** one-to-one
   assignment; the implementation uses `C(P,TQ) - ||mean(P)-mean(TQ)||²`
   before taking square roots (equivalent to `SSE - 6*mean_shift²` before
   dividing by six). This is **not** a claim that the translated image remains
   a legal integer-valued 7×7 drawing. It compares shapes independently of
   their location.
4. **shape_hausdorff**: symmetric max-nearest-neighbor distance after each D4
   transform and centroid alignment. Hausdorff does not enforce a one-to-one
   assignment, and the centroid translation is not necessarily the minimizer
   of Hausdorff itself. Treat this as an alternative shape descriptor.

For every image pair: `position >= d4 >= shape`, within floating precision.
Every distance matrix is symmetric with a zero diagonal. `shape_hausdorff` is
not required to preserve that ordering.

The lowest-cost transform and aligned **continuous** overlay are reported in
the HTML gallery for the closest `shape` pairs. The orange overlay points may
fall between or outside the original 7×7 lattice because they represent
centroid alignment, not actual lottery numbers.

## Real experiment on Windows / RTX 4060

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

# Only when changed functionality needs a focused verification:
& $py -m unittest discover -s tests -p 'test_pairwise_similarity_v013.py' -v

# All 3,116,256 actual historical image pairs; CUDA batched calculation.
& $py -m lotto7x7 pairwise-similarity-gpu `
    --device cuda `
    --batch-size 512 `
    --iid-pairs 30000 `
    --iid-quantiles 0.001 0.005 0.01 `
    --cluster-mode shape `
    --control-draws 384 `
    --control-replicates 9 `
    --top-pairs 25 `
    --gallery-count 8 `
    --seed 649 `
    --out results/v013_pairwise_similarity.json

Start-Process .\results\v013_pairwise_similarity.html
```

GPU-first implementation holds only `batch_size × 8 × 720 × 6` assignment
scores at a time; all four dense matrices are stored in host RAM, not the GPU.
The `torch`/NumPy code accepts `cuda`, `auto` and `cpu`. `cuda` fails fast when
unavailable. Batch size 512 is an initial Windows RTX 4060 configuration; if
CUDA reports memory pressure, try 256 instead. This is an engineering resource
setting, not a change in the distance definition or hypothesis.

### Outputs

The command creates **four new files**, rejecting any existing target path:

| Extension | Contents |
|---|---|
| JSON | 4 distance distributions, actual vs IID-pair quantiles, top actual image pairs, each complete-link threshold class summary, class membership, IID subset controls, source/checksum |
| CSV | Every historical draw with legal six numbers and its class for each tested distance threshold; singleton images remain `SINGLETON` |
| HTML | Offline side-by-side and aligned overlay gallery of the closest image pairs; representative images and neighboring members of diameter-bounded classes |
| NPZ | Unaltered 2497×2497 float32 distance matrices (`position`, `d4`, `shape`, `shape_hausdorff`) plus draw IDs, to support later analysis without recomputation |

### How thresholds and class counts are chosen

The threshold choices are **predeclared quantiles 0.1%, 0.5%, and 1% of IID
pairwise distances** in the selected `shape` metric. They are not chosen to
maximize how distinctive the real historical images appear. Given threshold
`t`, `scipy.cluster.hierarchy.linkage(..., method='complete')` creates a
partition in which **every pair within each class has distance ≤ t**.
Groups with ≥2 members are counted; singletons are retained, rather than
forced into a weakly related class. This eliminates the 5-of-6 connected-graph
chaining issue found previously. The class count depends on `t` and the
distance definition; there is no presupposed correct number of classes.

For transparency, the report also counts all actual image pairs with distance
≤ t; this pair count is not the same as the number of complete-link groups.
If the estimated IID distance distribution has ties, its observed `≤ t` share
can differ from the nominal target quantile.

### Independent fair-random controls and limitations

- Reference pair distribution: independent legal uniformly generated six-of-49
  pairs, not independent pixel Bernoulli values; 30,000 pairs by default.
- **Matched subset refits**: each replicate compares a uniformly chosen subset
  of 384 actual draws with a freshly generated 384-draw IID sample. Both use
  the **same predeclared cutoffs** and complete-link algorithm. The full
  historical pair matrix is reused to avoid the expensive real-data refit.
- At 9 replicates these are **exploratory descriptive controls**, not strong
  p-values; the actual sample draws across replicates can overlap. The 49
  number layout is arbitrary, and attractive-looking image classes are also
  expected from independent fair draws.
- The full historical 2,497-draw run must be executed on the user's RTX 4060;
  local verification uses only a small legal random fixture plus a 128-draw
  sample from the supplied v0.12 CSV. Do not confuse local test results with
  the unexecuted full-GPU historical analysis.

## Interpretation criteria

A useful *descriptive* classification needs groups with low within-class
maximum distance and clear gallery similarity. Compare near-pair rates and
non-singleton class coverage to IID at the same cutoffs. Only an independently
validated downstream selection study could establish forecasting value. Under
fair independent 6/49, all fixed six-number combinations retain equal chances
regardless of their visual class or proximity to old draws.
