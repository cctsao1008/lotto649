# Lotto7x7 — image-first research reboot

This repository begins **from scratch** with one problem: represent each Lotto649 draw as a **7×7 image containing exactly six active pixels**, study image shapes and their temporal order, and test whether a causal forecast resembles the next drawing more closely than an IID random outcome would.

There are **no inherited logits, MLP, legacy CNN classifiers, six-number jackpot likelihoods, legacy tests or phase-specific scripts** in the active tree. A new **CNN frame encoder + temporal GRU** is trained specifically for the 7×7 image-similarity objective. Old Git commits remain in ordinary Git history; they are not part of this rebuilt codebase.

## Preserved lottery records

The historical dataset files under `data/raw/`, `data/csv/`, `data/binary/` and `data/normalized/lotto649_*.{npy,parquet}` are preserved **byte for byte** from the original Git blobs. Non-draw reports and experimental artifacts are removed from the rebuilt working tree. The default model input is the versioned `data/bin/lotto649_images_v2.bin` containing **only Draw ID + 7×7 six-on bitmap** (53 bytes per record). The source CSV remains authoritative; the special number and exact date are not included in the BIN.

## Fresh implementation

- `lotto7x7/bin_dataset.py`: convert original Lotto649 yearly CSVs into BIN v2, with versioned header, 53-byte image records, strict validation, and byte-for-byte CSV verification.
- `lotto7x7/core.py`: reversible six-number ↔ 7×7 bitmap; sequential, snake and preseeded permutation mappings; full FFT and DCT roundtrip; geometric similarity with D4 rotations, translations, row/column occupancy and low-frequency DCT; conditional fair-IID baseline.
- `lotto7x7/learning.py`: **trainable image learning** with a 2D CNN per historical frame, recurrent GRU across frames, D4/translation-tolerant shape loss (image autocorrelation), auxiliary pixel reconstruction and six-active-point constraint; trained 7×7 image output with six brightest pixels selected. This does not import the retired six-of-49 logit decoder.
- `lotto7x7/models.py`: simple image-only causal controls (`previous`, `centroid_motion`, `temporal_analog`) and image-output diversity diagnostics; these are *non-neural baselines*, not the new learning model.
- `lotto7x7/research.py`: BIN v2 image importer (default; normalized NumPy files only as optional comparison); chronological train/validation/evaluation splits; conditional-IID-adjusted similarity; within-year time-order shuffling; validation-only method selection and per-draw JSON audit.
- `lotto7x7/diagnostics.py`: three-arm image-learning diagnostic (normal-order CNN+GRU, per-context time-shuffled CNN+GRU, and history-free trainable template); quantifies continuous logits, Top-6 collapse, latent variability, and paired IID-adjusted scores.
- `lotto7x7/__main__.py`: five commands: `check`, `learn`, `diagnose`, `replay`, and `gallery`.
- `tests/`: tests written **only for this newly designed 7×7 project**.

**Primary research outcome**: average `image_similarity(prediction, actual) - E_IID[image_similarity(prediction, random_draw)]`. A graphically similar forecast may have zero common numbers. The analysis is not a lottery ticket selector and contains no guarantee of an edge over random draws.

## Windows PowerShell

```powershell
cd D:\my-github\lotto649
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset --verify data/bin/lotto649_images_v2.bin
.\.venv\Scripts\python.exe -m lotto7x7 check
.\.venv\Scripts\python.exe -m lotto7x7 learn --year all --epochs 45 --baseline 32
.\.venv\Scripts\python.exe -m lotto7x7 gallery --report results/learned_replay.json --out results/learned_gallery.html
Start-Process .\results\learned_gallery.html

# Optional non-neural control study
.\.venv\Scripts\python.exe -m lotto7x7 replay --year all --baseline 32 --shuffles 99
.\.venv\Scripts\python.exe -m lotto7x7 gallery
Start-Process .\results\visual_gallery.html
```

Or create a clean venv using `py -3 -m venv .venv` and install requirements if needed. Generated JSON/HTML reports live under `results/` and are intentionally not tracked by Git.

## Optional NVIDIA GPU training for v0.3 loss-study

CUDA is **opt-in**. The original CPU `loss-study` still uses the exact frozen
v0.2 training function and v0.3 comparison path. GPU mode uses a separately
implemented device-aware copy of the **same** CNN/GRU, optimizer, loss variants,
chronological split, seed and early stopping. It does not modify v0.2 sources.

Use the **project venv interpreter**, not system `python`:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# Leave the existing historical report untouched; compare new files separately.
& $py -m lotto7x7 loss-study --device cpu --year all --epochs 45 --baseline 32 --bootstrap 400 --out results/loss_study_v03_cpu_check.json
& $py -m lotto7x7 loss-study --device cuda --year all --epochs 45 --baseline 32 --bootstrap 400 --out results/loss_study_v03_cuda.json
# --device auto selects CUDA if available; default remains CPU.
```

`--device cuda` fails explicitly if a CUDA-enabled PyTorch build or NVIDIA
GPU is unavailable; it does **not** silently run on CPU. Reports include
`execution.actual_device` and `execution.training_seconds` (the three model
training runs only; excludes data processing, scoring and HTML generation).
Compare runtime on the same PC under similar background load. CPU/CUDA
floating-point kernels may choose different checkpoints and Top-6 masks, so
the scores are **not** expected to match exactly. GPU is not guaranteed to
be faster for a small 7x7 model. No mixed precision is used.

The package's `requirements.txt` intentionally does not pin a particular
CUDA wheel: install the appropriate official CUDA-enabled PyTorch wheel
for your NVIDIA driver before running GPU studies.

## Research guardrails

The archive has already been explored in earlier work. A chronological test segment is useful for engineering checks but **not a new untouched blind test**. Do not tune shape weights, mappings, or analog windows on test scores and then claim improved future performance. The physical lottery machine has no demonstrated relationship to pixel geometry. An IID draw can resemble other IID draws; thus, high **raw** visual similarity by itself is not evidence of an advantage. Historical data remain unchanged. New prospective image-similarity research requires separately committed protocols and time-stamped pre-result predictions.

## Image learning vs. hand-built image statistics

The `learn` command **optimizes trainable CNN and GRU parameters** on the first chronological 60% of the 7×7 images. It chooses the epoch based on the next 20% (validation), and evaluates only the last 20% with a fixed image-similarity metric. The model consumes previous image frames and produces a 7×7 spatial output: there is no previously trained six-number model. For learning stability the differentiable training loss uses a 35% shape-autocorrelation term (invariant to translation and quarter-turn/reflection), 60% image-pixel reconstruction (weighted binary cross entropy), and 5% six-pixel mass penalty. The **evaluation objective**, however, is the external image-similarity score adjusted by the shape-specific IID expectation, never jackpot 6/6. This trainable path is distinct from the simple analog controls.

`results/learned_image_weights.pt` is a newly trained state dictionary. Only load weights from a trusted source using the corresponding architecture and safe framework loading settings. `results/learned_replay.json` contains per-draw output masks and adjusted similarity; the HTML gallery draws observed/predicted 7×7 images.

The new test suite includes **67 tests** (31 image-core tests, 15 BIN format tests,
7 BIN model-integration tests, and 14 collapse/temporal-diagnostic tests), including synthetic pattern learning, differentiable gradient flow, chronology and future leakage controls, DCT/FFT reversibility, visual similarity, and report generation. No old 169-case suite is retained. See [BIN v2 layout](docs/data_bin_format_v2.md).

## Canonical 7×7 image BIN v2

- **Source:** 20 annual `data/csv/<year>/大樂透_*.csv` files, 2007–2026.
- **Output:** `data/bin/lotto649_images_v2.bin`, currently **2,183 draws**, **115,715 bytes**.
- **Header:** `L649IMG2` (8 bytes) + record count (little-endian u32) + record width (little-endian u32 = 53).
- **Record:** draw ID (u32 little-endian) + a row-major 7×7 `uint8` bitmap (49 bytes, exactly six on pixels).
- **Absent:** special number, exact date and redundant six-number list. For year filtering, derive the calendar year from the ROC-year draw ID; do not fabricate a date.
- **Mapping:** sequential 01–49 raster mapping, fixed per file format. Other maps are research transformations derived from this canonical grid, not mixed inside the archive.
- **Verification:** regenerate from CSV, run `--verify`, and check `git diff --exit-code -- data/bin/lotto649_images_v2.bin` for byte identity.

To rebuild after a CSV update:

```powershell
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset --verify data/bin/lotto649_images_v2.bin
.\.venv\Scripts\python.exe -m lotto7x7 learn --year all --epochs 45 --baseline 32
```

The network reads raw 7×7 binary images from BIN v2 (converted to float32 for PyTorch). It does not train on special numbers. This data-format migration does not alter neural-network architecture or imply higher forecasting accuracy.

## Image Learning v0.2: collapse and temporal diagnostics

This is a diagnostic extension, not a model performance claim. It **does not
change the default `learn` command** or BIN v2. It adds a paired three-arm
comparison using identical 7x7 image targets, training/validation/test splits,
CNN+GRU hyperparameters, loss function, and IID scoring controls:

1. `ordered_cnn_gru`: normal chronological windows (the original v0.1 model).
2. `shuffled_cnn_gru`: independently permuted **past input frames per target**,
   at both training and inference; the target and the multiset of its earlier
   frames are unchanged. Training uses the same initialization, seeds,
   optimizer, and validation-loss selection as the ordered model.
3. `constant_template`: a learned 7x7 template, independent of past images,
   trained with the same loss, epochs, optimizer and validation criterion.

The diagnostic report additionally measures continuous logit variance,
Top-6/Top-7 margin, most-selected pixels, predicted-pattern concentration,
encoder/GRU latent variation, and a fixed-weight input-order sensitivity
control. Comparison uses shared per-target IID random masks, validation-only
early stopping and exploratory paired circular-block bootstrap intervals.

```powershell
cd D:\my-github\lotto649
.\.venv\Scripts\python.exe -m lotto7x7 diagnose --year all --epochs 45 --baseline 32
.\.venv\Scripts\python.exe -m lotto7x7 gallery --report results/image_diagnostics_v02.json --out results/image_diagnostics_v02.html
Start-Process .\results\image_diagnostics_v02.html
```

The output file is `results/image_diagnostics_v02.json`. The full report
contains each model's validation loss history, validation and retrospective
adjusted similarity, the 437 per-draw predictions and logit diagnostics.
**Do not tune on historical test scores and call it prospective performance**:
the dates have already been inspected. More varied images do not necessarily
imply predictive value. See `docs/image_learning_diagnostics_v02.md`.

## Image Loss Alignment v0.3 — frozen v0.2 comparator

The v0.2 diagnostics and original image-training functions are **unchanged** and anchored to Git commit
`0102c561530522dd0abae4fd4aa0b8fccef81a9e` (convenience branch
`baseline/image-learning-v02`). The new experiment isolates training-objective
changes under the same CNN, GRU, hard Top-6 decoder, BIN v2 data, split, and scoring.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m lotto7x7 loss-study `
    --year all --epochs 45 --baseline 32 --bootstrap 400
.\.venv\Scripts\python.exe -m lotto7x7 gallery `
    --report results/loss_study_v03.json `
    --out results/loss_study_v03.html
Start-Process .\results\loss_study_v03.html
.\.venv\Scripts\python.exe -m lotto7x7 loss-synthetic --epochs 45
```

The three arms are `original_v02` (unaltered v0.2 shape loss with positive-weighted BCE),
`balanced_bce` (same shape/mass loss, unweighted BCE), and `projected_six_mass`
(same CNN/GRU and six-logit decoder; differentiable per-sample soft mass exactly six).

New module: [`lotto7x7/loss_study_v03.py`](lotto7x7/loss_study_v03.py).
See the full preregistered control, loss and uncertainty details in
[`docs/image_loss_alignment_v03.md`](docs/image_loss_alignment_v03.md).
Historical evaluations are **exploratory**, since the targets were inspected
in prior versions. Increasing image variety is not evidence of lottery predictability.
