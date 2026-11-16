# Lotto7x7 v0.5.1 — Visual Image Dynamics Audit

## Objective and scope

**Image understanding comes first.** The subject is an ordered history of 7×7
binary images, each containing *exactly six active pixels*. This patch adds
interpretable **visual analysis** without changing the archived v0.2–v0.5
CNN/GRU models, objective functions, splits or immutable data files. Historical
summaries and saved forecasts are **retrospective exploratory evidence only**.

The six image points encode six lottery number labels, not physical ball
positions. The 7×7 arrangement is arbitrary; a visual feature that depends on
neighboring raster cells may disappear under another numbering layout.

## Commands on Windows

Run from repository root, using project-local Python (not `python` from PATH).

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'

# Tests, including the new fixed-cardinality and gallery-consistency checks.
& $py -m unittest discover -s tests -v

# New descriptive visualization; no model fitting is performed.
& $py -m lotto7x7 image-dynamics `
    --null-images 4096 --mapping-seeds 2 `
    --transitions 12 --matrix-count 24 --null-shape-pairs 128 `
    --out results/image_dynamics_v051.json `
    --html-out results/image_dynamics_v051.html

# Optionally integrate the historical predictions already produced by v0.5.
# Fails closed if the prior report has a different source-data SHA256,
# mismatched draw ID, a non-six-pixel image or inconsistent stored hit counts.
& $py -m lotto7x7 image-dynamics `
    --null-images 4096 --mapping-seeds 2 `
    --compare-report results/image_compare_v05.json `
    --out results/image_dynamics_v051_with_models.json `
    --html-out results/image_dynamics_v051_with_models.html
```

All output is **local, offline JSON and HTML/SVG**. The optional saved-model
comparison only reads its previous JSON; it does not train models.

## Section 1: causal chronological *display*, not a next-image forecast

For every consecutive image pair `A -> B`, the three colored pixel states are:

- **Retained:** `A AND B` (the same exact numbered cell is active twice).
- **Removed:** `A AND NOT B`.
- **Added:** `B AND NOT A`.

Since each input has exactly six active pixels,
`retained + removed = retained + added = 6`. The exact-number image Jaccard is
`retained / (12 - retained)`.

A *separate* D4/translation-tolerant shape metric uses maximum matching under
rotations, reflections and whole-image translation. Thus, a translated six-pixel
pattern can have **shape similarity 1.0 and zero exact shared numbers**. The
metrics are deliberately shown side by side; one must not substitute for the
other when assessing a lottery forecast.

## Section 2: pixel-frequency and non-DC DCT heatmaps

The null is a uniformly drawn *six-subset without replacement* from 49 pixels.
Its one-pixel appearance probability is exactly `p = 6/49`, and its expected
shared active positions with an independent six-subset is `36/49`.

For each raster cell `(r,c)`, display its observed frequency, its deviation
from exact `6/49`, and the *descriptive, unadjusted* occupancy standardization

`z = (observed_frequency - 6/49) / sqrt((6/49)(43/49)/N)`.

This is not a multiple-testing-corrected significance claim; the 49 pixel
counts are dependent because each image has exactly six points.

For an **orthonormal 7×7 DCT**, DC is constant (`6/7`) and its squared power is
`36/49`. Every other DCT coefficient has the exact same expected squared power
under the fair six-subset null:

`E[c_k^2] = 6(49-6) / (49(49-1)) = 258/2352 ~= 0.109694` for `k != DC`.

The plots explicitly gray out DC and display (a) actual non-DC squared power,
(b) finite IID Monte Carlo control and (c) actual-minus-IID residuals.
Comparing colored images without shared color scales or considering sampling
variation can mislead; residuals are descriptive, not proof of structure.

## Section 3: shape-feature distributions versus fixed-six IID

Six of the previously implemented normalized image morphology descriptors are
shown as histogram comparisons: radial spread, mean pair distance, bounding-box
area, eight-neighbor edge pairs, eight-connected components and row occupancy
entropy. Both histograms are **probabilities**, with separate sample counts.
The displayed `MC z` accounts for two finite sample counts but is **unadjusted
for all inspected features and numbering layouts** and should not be treated
as a hypothesis-test p-value.

The same actual historical image sequence and identical IID random masks are
re-rasterized in sequential, snake and fixed permuted layouts. The table shows
how feature differences depend on mapping. A change with mapping indicates
representation sensitivity, not a new observation in the original draw data.

## Section 4: recent cross-draw similarity matrices

For a fixed chronological set of up to 80 recent draws:

1. The exact-position overlap matrix contains integers 0–6. It is invariant
   under any consistent number-to-cell bijection.
2. The shape matrix uses the pre-existing D4/translation-tolerant Jaccard
   score and depends on the artificial geometry of the chosen layout.

The diagonal is self-comparison and is always maximum, not a temporal signal.
A reproducible finite IID sample of independent image pairs provides context
for nontrivial shape similarities under chance.

## Optional archived v0.5 image-model gallery

If `--compare-report` points to the output of `image-compare`, the new report
renders the already-saved prediction examples alongside their original actual
images. **No model is retrained.** The file must match the same source-data hash
and each displayed example must have a valid draw ID, exact original outcome,
valid six-point predictions, and consistent stored hit counts.

The gallery explicitly distinguishes:

- Exact number/pixel hits: the true shared active cells.
- Shape resemblance: the maximum match after D4 transformations and shifts.

A model can score highly on the second and still hit none of the six numbers.

## Reproducibility and non-goals

- Source: original immutable BIN v2 dataset of main numbers, excluding bonus.
- Seeds, exact image count, SHA256, chosen fixed layouts and control sample
  counts are preserved in the JSON report.
- Outputs are offline and use SVG; no CDN or network runtime is required.
- Source analysis never modifies historical training, validation or test sets.
- Research on previously inspected draws is **not** a prospective blind test.
- This patch deliberately does not change `ImageSequenceLearner`, any loss,
  early stopping, CUDA behavior, archived v0.2/v0.3/v0.4 diagnostics, or v0.5
  image-compare outputs.
- No statistical anomaly alone proves a causal prediction advantage or
  positive financial expectation from gambling.
