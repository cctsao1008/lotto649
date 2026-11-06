> **SUPERSEDED FOR ACTIVE RESEARCH:** Phase 7B remains as a reproducible baseline. The current time-order, dynamic-shape and diversity research protocol is [Phase 7C](phase7c_temporal_patterns.md), run with `lotto_pattern_temporal_audit.py` and `lotto_pattern_dynamic_replay.py`. Keep Phase 7B historical reports unchanged.

# Phase 7B — Pattern Similarity Framework v0.1

**Active research goal:** Predict a 7×7 six-pixel *image pattern that resembles* the upcoming draw, **not** maximize the chance of exactly matching six numbered pixels. Exact 6/6 is optional secondary bookkeeping. No prizes or near-jackpot claims follow from visual similarity.

**Status:** Code added in `feature/phase7b-pattern-similarity`; full Windows suite and real-history evaluation must be executed and independently reviewed. Historical 2023–2026 results were inspected previously, so any further retrospective replay is exploratory and cannot create blind predictive evidence.

## What is active and what is retired?

The active Phase 7B pipeline uses:

- `lotto_image_repr.py` for reversible 6-of-49 ⇆ 7×7 binary masks; existing snake, sequential and fixed-permutation layouts remain available.
- `lotto_pattern_similarity.py` for **position, shape and frequency-domain** similarity.
- `lotto_pattern_models.py` for image-only past-pattern analogs and train-only image families; no logits or machine-learning probability decoder.
- `lotto_pattern_replay.py` for strict chronological and **conditional IID-calibrated image-similarity** evaluation.
- `lotto_pattern_gallery.py` for a standalone HTML report with SVG drawings of actual and predicted 7×7 patterns.

Earlier `lotto_neural.py`, `lotto_image_experiment.py`, `lotto_replay.py`, the static/logit/MLP/old CNN candidates, and their old 6/6-centric reports are **retired from the new active workflow**. They remain in Git history and the repository for reproducibility, but the Phase 7B scripts import none of them. The IID references used here are **independent six-point random image controls**, not the previous predictive IID model. The existing sealed Phase 6 Forecast #001 is unaffected; its commit/payload must not be edited.

## Primary similarity definition

For predicted mask `P` and observed mask `O`:

| Component | Weight | Definition |
|---|---:|---|
| `pixel_jaccard` | 0.15 | Exact filled-pixel Jaccard, `hits / (12 - hits)`; 0 if no identical positions |
| `shape_d4` | **0.35** | Maximum Jaccard after all 8 square rotations/reflections and integer translations; shape independent of location |
| `row_column` | 0.15 | One minus normalized L1 distance between each image's row- and column-occupancy histograms |
| `centroid` | 0.10 | One minus normalized distance between centers of six points |
| `adjacency` | 0.10 | One minus normalized L1 distance between horizontal/vertical/diagonal neighbor-pair counts |
| `low_dct` | 0.15 | One minus normalized L2 distance between the 3×3 *low-frequency* DCT coefficients, excluding the fixed DC term |

All six components are normalized into `[0,1]` and the weights add to one. This is a **declared research metric**, not an intrinsic property of the lottery. It emphasizes geometric resemblance over identical numbers, so visually matching patterns can score highly with zero shared numbers. D4 invariance considers shape on an unbounded integer grid to avoid cropping or wraparound artifacts.

The full DCT and FFT remain reversible in the representation module. The **low-frequency DCT** included in the similarity score is *intentionally a reduced, lossy descriptor*, with the original binary mask always preserved. Full DCT Euclidean distance is deliberately excluded because orthonormal energy preservation makes it equivalent to spatial Euclidean distance, not an independent notion of shape.

Define the raw similarity
`S(P,O) = Σ_m w_m S_m(P,O)`.

## Preventing misleading shape-score improvements

A predictor could output a single compact, aesthetically central shape and score well against many arbitrary masks without predicting any draw. Therefore raw similarity **cannot** be taken alone as evidence of forecasting skill.

For each prediction `P_t`, generate `B` independent random IID six-point **outcomes** `R_{t,1}..R_{t,B}`; calculate a matched baseline

`E_IID[S | P_t] ≈ (1/B) Σ_b S(P_t, R_{t,b})`.

The primary comparison is

`Δ_t = S(P_t, O_t) - E_IID[S | P_t]`.

The reported score is the draw-averaged `Δ_t`. This adjusts for the geometry of **that particular predicted image**. The same random reference outcomes are reused across the three methods at a given draw for a paired, reproducible comparison. Control samples, mapping seeds and metric weights are fixed by explicit CLI configuration. Small B has Monte Carlo uncertainty; increase `--null-samples` to refine estimates, not to tune models or choose winning periods.

**Important limitation:** This conditional expected-score correction is a descriptive baseline, not a calibrated p-value or a blind guarantee. A proper inferential claim needs a much longer independently precommitted prospective experiment and treatment of temporal/multiple-testing dependence.

## Three image-only prediction approaches

| Method | Forecast mechanism | Train-only learned information |
|---|---|---|
| `last_pattern` | Use previous draw's full 7×7 image as the predicted next image | None; transparent visual persistence control |
| `temporal_analog` | Match the previous five image signatures to train-only historical windows, then use the *next observed image following the best past analog* | Training-period image signature context/target pairs |
| `cluster_transition` | Cluster training images into **8 image families**; from the previous image's family, choose the most frequent next family within the training period and its representative six-pixel prototype | Training-period centroids, representatives and family transitions |

The fixed 27-feature image signatures consist of 8 low-band DCT values (no DC), 7 row occupancies, 7 column occupancies, 2 centroid coordinates and 3 adjacency counts. The temporal analog uses last-five image signatures, weighted toward recent images.

These are **pattern approaches**, not supervised old-model probability estimators. No new neural model is required to execute the first Phase 7B experiment. Train-only clusters are genuine image families, but any observed sequence regularity still requires comparison with a fair-IID control.

## Data isolation

Use the preexisting absolute chronological boundaries, not re-splitting based on the latest archive:

- Train targets from absolute index `window=5` to `1308` (inclusive); train-only historical image bank is indices 0..1308.
- Validation targets `1309..1745` (inclusive), used to select one of the three methods **by mean chance-adjusted similarity**.
- Test targets `1746..` (2023–2026 in the original base archive). Test output can be restricted by calendar year, or report all years together.

Target `t` is never included in its input context. The previously observed draw `t-1` may be used to forecast t, including in the replay test period. The bank, clusters and prototype transitions are fit/frozen from training only.

## Run on Windows / PowerShell

```powershell
cd D:\my-github\lotto649
git status --short
git fetch origin
git switch feature/phase7b-pattern-similarity
git pull --ff-only origin feature/phase7b-pattern-similarity

# Expect 139 total tests if prior 118-test Phase 7 branch remained intact.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Replay the complete historical evaluation range in one run
.\.venv\Scripts\python.exe .\lotto_pattern_replay.py `
    --source data/normalized/lotto649_base.npy `
    --year all `
    --null-samples 32 `
    --report data/neural/phase7b_pattern_similarity.json `
    --csv data/neural/phase7b_pattern_similarity.csv

# Render real and predicted 7x7 images and image family prototypes
.\.venv\Scripts\python.exe .\lotto_pattern_gallery.py `
    --report data/neural/phase7b_pattern_similarity.json `
    --out data/neural/phase7b_pattern_gallery.html `
    --max-draws 40

Start-Process .\data\neural\phase7b_pattern_gallery.html
```

For closer Monte Carlo calibration, you may repeat a **predeclared** comparison with `--null-samples 128`; higher values increase CPU cost. Do not select a favorable seed/mapping based on observed test results.

## Gatekeeping and interpretation

- Run the full suite: original 96 + Phase 7's 22 + new Phase 7B's 21 = **139 expected tests**. This is only an expected count, not a claim that Windows tests passed.
- Positive control: a known artificial 4-pattern deterministic cycle must be perfectly recovered by temporal-analog similarity; does **not** prove real Lotto649 predictability.
- Causal isolation: perturbing actual target t must not alter a forecast for t.
- Visual score is primary. A zero-overlap image can achieve `shape_d4 = 1` after rigid transformation; this meets the revised research objective even though its complete-six hit count is zero.
- Raw similarity must be compared against **conditioned** IID expectation. If mean adjusted score is close to 0, there is no evidence that observed future images are systematically closer than chance would make them.
- Historical 2023–2026 is *not newly blind*. Do not use it to retune geometry weights, family count, mapping choices, or methods and then claim discovery.
- The previous sealed future forecast `d4a131ad6bc503bc7360eb680bab899318a84234` stays immutable; if the goal of future prospectively sealed research is formally changed, create a **separate new seal and protocol**, never rewrite that one.

## Deliverables

- JSON: metadata, train-only family prototypes, validation selection, evaluation summaries and per-draw 7x7 text masks.
- CSV: one row per method per draw, with raw, IID expected, adjusted similarity and all six components.
- HTML: actual/predicted pixel grids, representative train-only cluster masks and comparison scorecards.

All Python comments and terminal output remain English-only.
