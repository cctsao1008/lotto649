# Lotto7x7 v0.5 — Image Structure & Temporal Learning

## Research question

Lotto7x7 remains **image-first**: every draw is a 7×7 bitmap with exactly six
active pixels. We want to learn what image shapes, relative geometry, spatial
frequencies and transformations look like, whether their chronology carries
useful information, and whether image-only feature representations improve a
**causal forecast of the next six-pixel image**.

An image representation does not introduce new information. The mapping of
lottery number to raster position is arbitrary; therefore any geometric or
spectral association must survive the spatial-mapping negative control before
being interpreted as a robust property of the source. **Even a repeatable
pattern is not evidence of out-of-sample forecastability.**

## Commands

All commands require the project's Python environment; the native terminal
`python` may point at a different environment without SciPy.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'

& $py -m unittest discover -s tests -v

# Descriptive 7x7 image structure, mapping sensitivity, lagged chronology.
& $py -m lotto7x7 image-study `
    --permutations 499 --null-images 4096 --mapping-seeds 2 `
    --out results/image_structure_v05.json `
    --html-out results/image_structure_v05.html

# Causal next-image learning: four IMAGE-derived representations.
& $py -m lotto7x7 image-compare `
    --seeds 3 --epochs 45 --bootstrap 400 `
    --out results/image_compare_v05.json `
    --html-out results/image_compare_v05.html
```

Both HTML outputs are **offline** and include embedded 7×7 SVG images or
heatmaps. No CDN, login or Internet connection is needed to view them.

## Phase 1 — Image understanding

Input is verified BIN v2, in draw order, excluding the special number. Per-image
features have no access to other draws:

- **Morphology** (24 dimensions): normalized centroid, radial spread, pairwise
  distance moments, bounding box, 8-neighborhood adjacent edge pairs, number of
  8-connected components, row and column occupancy entropies, and seven row and
  seven column occupancy fractions.
- **Spectral** (13 dimensions): eight low-band orthonormal DCT coefficients,
  low-band DCT energy fraction, low-band FFT energy fraction, three radial FFT
  power bands. DC (the fixed six-pixel mass) is excluded from the energy ratio
  denominator to prevent this constant feature from dominating.
- Pixel occupancy and **mean squared DCT coefficient** heatmaps; recent real
  draw bitmaps and their geometric properties in HTML.

Baseline: reproducible independent **six-of-49** random images. Independent
Bernoulli pixels are deliberately *not* used because those do not preserve the
exactly-six constraint.

## Phase 2 — Temporal image dynamics

On the full historical sequence, compare lagged overlap count and low-band DCT
cosine for lags 1, 2, 3, 5 and 10. Pairs crossing calendar years are excluded.
The temporal null shuffles *image order only within the same year*, preserving
all observed yearly image and number compositions. The Monte Carlo two-sided
permutation tail includes the +1 correction; Bonferroni is applied over the
prespecified five lags × two metrics = ten tests **per mapping**.

A result from one of several mappings or other inspected exploratory metrics
still requires independent confirmation. The model does not assume each
post-shuffle draw to be independent of other draws, only a conditional
exchangeability null under temporal reordering.

The exact fair-draw expectation of shared pixels for two six-subsets is
`6 * 6 / 49 = 36 / 49 ≈ 0.734694`.

## Phase 3 — Spatial mapping negative control

Compare:

1. `sequential` (original mapping, number 1..49 in row-major order)
2. `snake` (every other row reversed)
3. Two different **fixed** random 49-number permutations

All maps are bijections, and the same historical draws are remapped to every
view. Pixel-overlap counts and actual six numbers are invariant; geometry and
DCT/FFT features can change. Do not select a mapping based on the previously
inspected historical test results. This module is an *exploratory diagnostic*;
future controlled forecasting with alternate mappings is separate work.

## Phase 4 — Image representation ablation

The `image-compare` command trains all models on the same *chronological*
60/20/20 train/validation/test targets, using the same seed list, same CNN
output shape, top-six decoder, balanced BCE v0.3 objective and AdamW protocol:

| Model | Input derived from 7×7 image | Temporal decoder |
| --- | --- | --- |
| `image_cnn_gru` | Raw 7×7 image, existing two-layer CNN | GRU |
| `morphology_gru` | Deterministic per-image 24-D spatial morphology | MLP + GRU |
| `spectral_gru` | Deterministic per-image 13-D DCT/FFT features | MLP + GRU |
| `image_fusion_gru` | CNN embedding plus morphology / spectral embedding | GRU |

All three feature-containing arms use standardization **fitted only on the
training-era images**, never on validation/test. The test image at target `t`
is never used in its own history window. A model's checkpoint is selected by
validation loss only (Epoch 0 included). Each experiment reports parameter
counts, training/validation histories, random seeds and train-only
normalization constants. Sizes differ between architectures; this is not a
strictly matched-parameter benchmark.

## Forecast scoring

Models emit 49 image logits. The six largest logits light exactly six pixels,
recovering six lottery numbers in the original mapping. Report:

- **Primary**: per-pixel Brier Score and skill relative to `p = 6/49`.
- **Primary**: top-six exact pixel hits and their 0..6 distribution.
- **Secondary**: raw pixel intersection-over-union (IoU), D4/translation
  tolerant image shape similarity, image diversity, predicted soft six-mass.
- **Paired exploratory**: moving-block intervals of the per-date difference
  from the existing `image_cnn_gru` arm after averaging across seeds.

The hypergeometric baseline of `6/49` for expected *hits* and the exactly
calculable fair-IID mean IoU are included. Shape similarity is reported raw
only and is NOT interchangeable with prediction of exact lottery numbers.

## Interpretation and limitations

- Historical evaluation includes outcomes previously studied in v0.2–v0.4;
  no model-comparison result from these dates is a prospective blind test.
- Pre-register and archive a future-image prediction **before** the drawing
  before using results to claim prospective predictive skill.
- A convolutional network's local spatial inductive bias assumes a numbering
  adjacency that the lottery drawing mechanism does not physically share.
- Adjacent pixels and DCT/FFT spatial patterns can be visually attractive even
  for purely independent random draws; IID and mapping controls are mandatory.
- Multiple mappings and many extracted descriptors create selection bias;
  adjusted per-mapping lag tests do not resolve all multiple-researcher-choice
  risks.
- New models are research controls, not suggested lottery wagers.

## Reproducibility / version integrity

Keep `data/`, `lotto7x7/learning.py`, `loss_study_v03.py`,
`predictability_v04.py` and older training/test code untouched. v0.5 adds
`image_structure_v05.py`, `image_compare_v05.py`, new tests and CLI entries.
Do not overwrite v0.2, v0.3 or v0.4 JSON reports.
