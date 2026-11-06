# Phase 7C — Temporal Pattern Signal and Dynamic Image Forecasting (v0.1)

**Branch:** `feature/phase7c-temporal-patterns`

**Research goal:** Determine whether the *time ordering* of historical 7×7 six-point lottery images carries reproducible information about the next image's **visual shape and spatial distribution**, beyond chance. Produce **dynamically changing 7×7 six-point forecasts**, not a pair of repeated static cluster representatives.

This is an exploratory retrospective analysis of draws already examined in prior phases. It cannot establish a prospective forecasting advantage, lottery prizes, or better odds of choosing a winning ticket.

## Background

Phase 7B's `cluster_transition` generated only two distinct six-pixel outputs across 437 retrospective draws:

- Pixel set `02 03 13 19 22 39`: 284 forecasts.
- Pixel set `12 21 25 29 42 48`: 153 forecasts.

Its raw shape similarity score was 0.485048, but its matched IID expected similarity was 0.484051, a **+0.000996** difference (small and not demonstrated statistically significant). Raw geometry scores are not a forecast success metric. The prior neural/logit/complete-six-match models remain retired from the active image-research workflow.

## Modules

| File | Purpose |
|---|---|
| `lotto_pattern_temporal_audit.py` | Within-calendar-year chronological order permutation, train-normalized image signature lag dependence |
| `lotto_pattern_dynamics.py` | Causal dynamic forecasts: centroid flow and multi-analog image fusion; grid/shape diversity |
| `lotto_pattern_dynamic_replay.py` | Validation-only model selection; 437-target replay, paired IID controls, block bootstrap, shuffled-context and shuffled-training controls |
| `lotto_pattern_dynamic_gallery.py` | Side-by-side SVG actual/forecast grids, image diversity, temporal audit |
| `lotto_image_repr.py` | Existing lossless six-of-49 ⇆ 7×7 reversible raster mapping |
| `lotto_pattern_similarity.py` | Existing six-component, predetermined image similarity with emphasis on D4 shapes |

No `lotto_neural`, MLP, CNN-logit classifier, `lotto_torch_decoder`, old six-subset probability model, or prior `lotto_image_experiment` is used in the Phase 7C execution path.

## Step 1 — Temporal order audit comes first

For each draw image, extract the **fixed 27-dimensional visual signature** from the original Phase 7B: low-frequency DCT (without constant term), row and column occupancies, centroid and adjacency counts.

Standardize features using `0..train_end-1` only. Calculate cosine similarity between signature `z_t` and lagged signature `z_{t-l}`. The **predeclared primary** is lag 1; lags **2, 3, 4, 5, 10** are exploratory secondary diagnostics.

Each null permutation independently *shuffles the chronological order within the same calendar year and segment*, preserving the actual six-point masks, per-year frequencies, sample counts and history segment. Cross-year lag pairs are excluded from this audit to prevent season/year boundary comparisons. Null distribution and empirical **two-sided** tail are reported for negative or positive lag dependence; the method does **not** assume all temporal structures increase adjacent-frame similarity.

Segments are reported independently:

- Train prefix: indices 0–1308 (no future data to standardize).
- Validation: indices 1309–1745.
- Retrospective evaluation: indices 1746–end (437 original test draws).

Primary time-order test statistic: **observed lag-1 cosine minus within-year shuffled expectation**, with a descriptive empirical two-sided tail. Multiple lags and retrospective inspection prevent using it as a definitive discovery p-value. Within-year time exchangeability is itself a modeling assumption; trends inside years can violate it.

## Step 2 — Dynamic forecast methods

All models output **one seven-row, six-pixel binary image** using `t-1` or earlier observed images, no labels from target t:

- **`last_pattern` (control):** Copy the immediately previous known image. One output per target, with no fitted parameters.
- **`centroid_flow`:** Estimate centroid translation between the two latest images and move the most recent six-point pattern accordingly. The translation is *clamped to the 7×7 boundaries without clipping or wraparound*, preserving six distinct pixels and their relative shape.
- **`analog_fusion`:** Compare the last five 27-feature image-signature frames and their successive changes with **training-only** analog windows. Select the eight closest windows under predeclared 70% level and 30% change distances, transport the following training image by centroid alignment, blend those six-point masks with a 20% recency weighting on the latest known image, and select the six highest-support pixels to form a **new, dynamically synthesized** image.

A frozen bank consists of training source images **indices <1309 only**. Windows and their following outcomes are both within the training segment; no validation or test outcome enters the reference bank. The current context may include previously observed validation/test draws, as in a real one-step system. Exact six-number match is retained only as a secondary audit field, not trained or optimized for.

**Key qualification:** There is no guarantee that these methods generate novel or diverse patterns on every target. The diversity report measures that result instead of asserting it.

## Step 3 — Score shape relative to identical-target IID controls

Retain the original Phase 7B predeclared six-component image metric:

| Component | Weight |
|---|---:|
| D4 rigid shape similarity after rotation/reflection and translation | 0.35 |
| Pixel Jaccard overlap | 0.15 |
| Row/column occupancy | 0.15 |
| Centroid proximity | 0.10 |
| Adjacency counts | 0.10 |
| Low-frequency DCT proximity | 0.15 |

Compute

`Delta_t = S(predicted_image_t, actual_image_t) - E_IID[S(predicted_image_t, random_six_pixel_outcome)]`.

Crucially, the IID expectation is conditional on that **exact predicted shape**: a geometrically generic image can have high raw similarity to random targets yet no predictiveness. Use independent, reproducible random-six-image outcomes for each target, reused across forecast methods to produce paired comparisons.

For each forecast method report:

- Mean raw image similarity, expected chance similarity, **chance-adjusted composite** (primary).
- Chance-adjusted individual components, especially rigid shape and low-frequency DCT.
- **Unique 7×7 output images** and unique D4-plus-translation-normalized shape classes.
- Top forecast image share, pixel-pattern entropy and effective number of distinct patterns.
- A circular moving-block bootstrap 95% interval for chance-adjusted similarity and paired difference against `last_pattern` (block length 8, 400 resamples by default).

The bootstrap captures some time dependence but does *not* fully propagate finite Monte Carlo baseline error, multiple comparisons, or the prior inspection of test data.

## Step 4 — Matched temporal-order negative controls

Two separate controls test whether temporal ordering contributes:

1. **Past-frame scramble:** For each prediction date t, shuffle only the already observed `t-window .. t-1` frames. Feed those *same masks* in a different order to `centroid_flow` and `analog_fusion`, with all model settings fixed. No future image can enter this input control.
2. **Training-transition shuffle:** Within each calendar year of the original training prefix, permute the training image sequence to destroy original transitions while preserving year-specific image counts and patterns. Refit only the analog reference bank and compare the resulting `analog_fusion` outputs against the **same unmodified evaluation outcomes** and the same conditional IID controls. This is a model-level negative control, distinct from the lag-correlation audit.

The default four training-bank shuffles provide only a **diagnostic** comparison; they cannot deliver a small or credible permutation p-value. Increase with `--bank-shuffles 24` or more *before* viewing test results if evaluating a genuinely fresh dataset. Do not select a favorable shuffle seed or mapping after seeing scores.

## Data split, defaults and research validity

Frozen Phase 5 boundaries stay at 1309 (train end exclusive) and 1746 (validation end exclusive); historical test has **437 targets**, including 2023–2026 draws in the original archived source.

| Parameter | Default |
|---|---:|
| 7×7 mapping | sequential |
| History window | 5 images |
| Nearest training analogs | 8 |
| Recency image weight in fusion | 0.20 |
| Target-specific IID six-point controls | 32 |
| IID controls per validation target | 16 |
| Within-year time permutations | 99 |
| Within-year shuffled training banks | 4 |
| Circular block bootstrap | 400 samples, blocks of 8 |
| Random seed | 649 |

All settings are fixed before Phase 7C test viewing. Earlier 2023–2026 data have already been examined, so this is **not** the start of a new blind prospective experiment.

## Windows / PowerShell commands

Run from your repo and **check `git status` before switching**; do not discard local work.

```powershell
cd D:\my-github\lotto649
git status --short
git fetch origin
git switch feature/phase7c-temporal-patterns
git pull --ff-only origin feature/phase7c-temporal-patterns

# Previous 139 + Phase 7C's 30 = 169 expected tests.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Step 1: time-order signal compared with 99 within-year shuffles.
.\.venv\Scripts\python.exe .\lotto_pattern_temporal_audit.py `
    --source data/normalized/lotto649_base.npy `
    --permutations 99

# Steps 2–4: dynamic-image test, IID conditioning, diversity,
# validation selection, scrambled-history, bank permutation, bootstrap.
.\.venv\Scripts\python.exe .\lotto_pattern_dynamic_replay.py `
    --source data/normalized/lotto649_base.npy `
    --year all `
    --null-samples 32 `
    --bank-shuffles 4

# Visual inspection of the changing actual and predicted images.
.\.venv\Scripts\python.exe .\lotto_pattern_dynamic_gallery.py `
    --report data/neural/phase7c_dynamic_replay.json `
    --out data/neural/phase7c_dynamic_gallery.html `
    --max-draws 40
Start-Process .\data\neural\phase7c_dynamic_gallery.html
```

The replay automatically includes its temporal permutation audit in the JSON even if the independent audit command has already been run; the independent command helps inspect time-order evidence **before** the new model output.

A convenient output summary is the program's printed table with **raw**, **IID expectation**, **adjusted**, **unique output patterns**, and **top share**, followed by lag-1 permutation results and scrambled-input comparisons.

### Interpretation gate

1. **Engineering gate:** New 169-test expected suite must pass. A deterministic synthetic four-pattern cycle must be recovered by one-neighbor image analog fusion on all 24 artificial held-out targets; no inference about genuine lottery draws follows.
2. **Temporal gate:** If lag-1 signature dependence is consistent with within-year shuffled sequences, there is **no established** evidence of an exploitable short-term time signal. A nonzero lag need not imply next-image predictability; its absence also does not disprove every nonlinear multi-lag relationship.
3. **Diversity gate:** A dynamic model outputting more unique shapes than the former two-prototype collapse satisfies only the *representation/expressiveness* goal, not the predictive goal. Always report unique **pixel configurations and rigid-shape classes** separately.
4. **Prediction gate:** Demand positive, repeated **chance-adjusted** similarity relative to IID, paired differences vs simple `last_pattern`, and scrutiny of uncertainty intervals. Historical test optimization invalidates future claims.
5. **Immutability:** Neither the original Phase 6 sealed prospective 2026-11-10 forecast nor its protected payload, source, or scoring policy may be altered for this experiment. A future *pattern-similarity* forecast requires a new explicit seal and separate preregistered evaluation.

These files generate reproducible algorithmic *experiments*, not a recommended ticket or winning-number promise.

All source comments, docstrings, terminal output and logs are English-only.
