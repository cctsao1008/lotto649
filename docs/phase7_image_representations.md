> **ARCHIVED / SUPERSEDED:** This exact-six and old neural-model Phase 7 experiment is retained only for reproducibility. The current research objective is graphical shape similarity rather than full six-number coincidence. Use [Phase 7B Pattern Similarity](phase7b_pattern_similarity.md) and `lotto_pattern_replay.py` instead. All older 2023–2026 neural results remain historical records, not active selection criteria.

# Phase 7 — Reversible 7×7 Image Representation and Pattern Learning

Status: **experimental branch implemented; full Windows validation and real-history run pending**.

Branch: `feature/phase7-image-representation`. It does not change the already sealed prospective [Forecast #001](https://github.com/cctsao1008/lotto649/blob/d4a131ad6bc503bc7360eb680bab899318a84234/data/shadow/seals/forecast_20261110.json). Historical 2025 results have been examined before, so this phase is exploratory **and not a pristine blind test**.

## Research question

Can a reversible image representation make a model more effective at learning from a historical sequence of observed six-number lottery draws than the equivalent canonical 49-bit input? The encoding is merely a change of representation; **it does not add information or imply that numbered positions have physical spatial relationships**.

The primary outcome is **one complete six-number set**, and a prediction is PASS only when the six regular main numbers exactly match the actual six main numbers, in any order. All 0–5 overlaps are FAIL, regardless of partial prizes. The seventh special number is excluded. Exact full-set log probability versus IID is retained as a secondary model diagnostic, particularly because true complete hits are extremely rare.

## Data representations

All spatial encodings preserve 49 unique labels and exactly six activated pixels. `lotto_image_repr.py` provides both encode and decode, strictly validating the six-of-49 constraint:

| Path | Spatial or spectral representation | Reversibility |
|---|---|---|
| Sequential | Numbers 1–49 fill raster positions, row-major 7×7 | Exact |
| Snake | Alternating row direction | Exact |
| Permuted | Fixed RNG permutation, seed 649 by default | Exact |
| DCT | Full orthonormal DCT-II 7×7 coefficients for every input image | Numerically reversible; no truncation |
| FFT | Full orthonormal 2D FFT, preserving **both real and imaginary planes** | Numerically reversible; no truncation |
| Temporal | Previous 5 spatial images stacked into a 5×7×7 sequence | All input frames retained |

DCT and FFT are performed in float64 for reconstruction checks; the features used in PyTorch are float32. Floating point recovery is approximate rather than bit-for-bit, but image reconstruction errors are checked and should be well below any threshold that could change a 0/1 pixel after rounding. Removing coefficients or FFT imaginary components is **not** allowed in the lossless comparison.

`temporal_windows` for target t uses **only** `t-window ... t-1`. Feature tensors have shape `[batch,5,7,7]` for image and DCT CNNs, `[batch,10,7,7]` for FFT real/imag pairs, or `[batch,1,5,7,7]` for the 3D temporal CNN. All models produce 49 canonical-number logits; the unchanged exact six-subset probability decoder gives `P(S_t)` without independently predicting six unnormalized binary pixels.

## Controlled models

`lotto_image_experiment.py` compares eight named models:

1. `iid` — exact uniform six-combination negative control.
2. `flat_mlp` — existing 49-bit lag-window SmallMLP, unchanged architecture.
3. `cnn_sequential` — 2D spatial CNN with row-major input.
4. `cnn_snake` — the **same 2D CNN architecture** on snake-layout pixels.
5. `cnn_permuted` — the **same 2D CNN architecture** on one fixed random permutation (mapping seed recorded).
6. `cnn_dct` — the **same 2D CNN architecture** on complete DCT coefficients.
7. `cnn_fft` — the **same 2D CNN architecture** on paired real/imag FFT coefficients.
8. `cnn_temporal` — temporal 3D convolution on spatial image sequences.

The image CNN architecture has 8-channel intermediate layers and a 16-unit hidden head (default), while the temporal encoder uses two 4-channel 3D convolutional layers and a 16-unit head. Architectural differences affect parameter counts; **equal training hyperparameters do not mean equal model capacity**.

All learned models share:

- Original Phase 5 chronological absolute train boundary `train_end=1309` and validation boundary `validation_end=1746` (not recomputed from a growing archive).
- Training targets 5..1308, validation targets 1309..1745; the held-out segment begins at target 1746, with year 2025 selected for the first replay.
- Seed 649, history window 5, hidden 16, 40 maximum epochs, early-stopping patience 6, batch 128, learning rate 0.005, weight decay 0.02, exact six-subset log likelihood, and validation-only early stopping.
- Their original, already trained validation-selected result is recorded. A selected winner on an explored test year **must not** become the recommended forecasting model.
- The mapping seed and all model names are recorded in the JSON. No label or future result goes into the feature input of the same target.

The `cnn_permuted` model is **one control**, not a statistically sufficient permutation-invariance test. Multiple *precommitted* mapping seeds and repeated negative controls would be required for a stronger claim.

## Geometry-only exploratory analysis

`lotto_image_analysis.py` computes the 7×7 mask's row and column occupancy, centroid, 4-neighbor and diagonal-neighbor pairs, and horizontal/vertical mirror symmetry. The dataset summary compares observed neighbor counts with a seeded IID image simulation and reports actual consecutive draw overlap against the exact IID expectation of `36/49`.

These are **descriptive features**. Total energy of a binary six-pixel image and its full normalized FFT/DCT is always six by Parseval's identity; it cannot itself be used as a discriminating predictive feature. Pixel geometry depends on the mapping, so observed structure must be compared with a similarly mapped IID control.

## Local Windows instructions

Run from your existing repository. Do **not** overwrite previously sealed forecasting files.

```powershell
cd D:\my-github\lotto649
git status --short

git fetch origin
git switch feature/phase7-image-representation
git pull --ff-only origin feature/phase7-image-representation

# 96 existing tests + 11 representation + 8 learning + 3 geometry = 118 expected tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Check reversible binary layout and full DCT/FFT reconstruction
.\.venv\Scripts\python.exe .\lotto_image_repr.py `
    --numbers 5 9 26 32 41 48

# Compare geometric patterns to independently generated IID controls
.\.venv\Scripts\python.exe .\lotto_image_analysis.py `
    --source data/normalized/lotto649_base.npy `
    --mapping sequential `
    --report data/neural/phase7_geometry.json

# Experimental 2025 full-six historical replay with all eight models
.\.venv\Scripts\python.exe .\lotto_image_experiment.py `
    --source data/normalized/lotto649_base.npy `
    --year 2025 `
    --report data/neural/phase7_2025.json `
    --csv data/neural/phase7_2025.csv
```

Optional spot-check of the complete predictions:

```powershell
Import-Csv .\data\neural\phase7_2025.csv |
    Where-Object model -eq 'cnn_sequential' |
    Select-Object -First 10 date,draw_id,predicted_main_six,actual_main_six,result |
    Format-Table -AutoSize
```

Each CSV result is a literal **PASS** or **FAIL**; no partial matches are counted as success.

## Gates and interpretation

1. All **118** tests pass. The synthetic constant-six positive control must recover **24/24 exact six-of-six matches** using an image CNN while the IID control does not; this verifies ability to learn an *artificial* easily predictable mechanism, not genuine lottery prediction.
2. The mask encode/decode and FFT/DCT round-trip should reconstruct every tested input. Check the target-mutation leakage tests: changing target t must not change input x[t].
3. The 2025 replay must evaluate **118** original real draws and report all eight models. Confirm that the `flat_mlp` log-score result is close to the prior Phase 5 2025 lift (~`+0.003803` nats/draw) under identical training settings; large differences indicate a reproducibility or implementation error.
4. **Exact 6/6** is the principal experimental goal. A zero count for all models is fully consistent with a fair lottery; the expected exact match count for one fixed selection in 118 IID draws is `118/13,983,816 ≈ 0.00000844`.
5. Log score and any learned spatial statistics are **secondary** evidence and must be tested independently against IID before making predictive claims; post-hoc best mappings should not be promoted as prospective success.
6. The branch stays separate until correctness and method review. Forecast #001 stays sealed at `d4a131ad6bc503bc7360eb680bab899318a84234`, default model IID. New retrospective training never updates that seal.

All Python comments, docstrings and console messages are English-only; discussion with the user is in Traditional Chinese.
