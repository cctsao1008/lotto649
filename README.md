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

## v0.4: multi-seed predictability audit

The `predictability` command records both training and validation loss
from **Epoch 0 (before any optimizer update)**, compares the same balanced-BCE
CNN+GRU with ordered and within-context-shuffled history, and adds a no-history
constant-template control. Five initialization seeds and a shared IID baseline
permit exploratory paired comparisons of image similarity and pixel Brier.
No loss selection or model changes are made on the historical test period.

A periodic positive control uses a cycle equal to the history-window length:
all historical windows have the **same unordered multiset** of images,
so a model must exploit ordering to identify the next phase. An independent
six-of-49 series is the negative control.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 predictability --year all --seeds 5 `
    --epochs 45 --control-seeds 2 --control-epochs 45 `
    --baseline 32 --bootstrap 400 `
    --out results/predictability_audit_v04.json
```

This audit is intentionally CPU-only: CPU was faster than RTX 4060 for the
current small network in one measured run. The 2023–2026 historical evaluation
data have already been inspected; this audit is exploratory rather than blind.
See [v0.4 experiment protocol](docs/predictability_audit_v04.md).

## v0.5: image structure, dynamics and image-only representation learning

The **research priority is 7x7 image geometry and image forecasting**, not
general numerical lottery models. Each draw remains a lossless six-pixel bitmap.

New commands:

- `image-study`: morphology / spatial geometry, FFT / DCT spectral features,
  lagged image relationships, IID six-of-49 nulls and fixed spatial-mapping
  controls (sequential, snake, randomized). Produces **offline HTML** with
  image and spectrum heatmaps, and a machine-readable JSON report.
- `image-compare`: same-history causal image prediction from a raw-image CNN,
  morphological image descriptors, spectral image descriptors and fused image
  features. All use GRU temporal sequence modeling and the v0.3 balanced BCE
  image loss; report Brier, exact six-point hits, geometry, parameter counts,
  paired uncertainty and an offline visual prediction gallery.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 image-study --permutations 499 --null-images 4096 --mapping-seeds 2
& $py -m lotto7x7 image-compare --seeds 3 --epochs 45 --bootstrap 400
```

The new code leaves archived v0.2 / v0.3 / v0.4 models and historical data
unchanged. Historical evaluation data have been inspected previously, so these
are **exploratory studies**, not prospective evidence of a lottery edge.
See [v0.5 experiment protocol](docs/image_structure_and_temporal_v05.md).

## v0.5.1: Visual Image Dynamics (image-first)

The `image-dynamics` command adds a stand-alone **offline SVG/HTML**
research view without retraining or altering any v0.2–v0.5 models:

- Per-draw image transitions: **retained / removed / added** pixels, exact
  position Jaccard and D4/translation-tolerant **shape similarity** separately.
- Pixel-frequency differences versus exact fair `6/49` expectation;
  six-pixel IID **morphology histograms** and spatial-mapping sensitivity.
- **Non-DC DCT power** (observed, IID control, residual); DC is explicitly
  excluded because every image contains exactly six filled pixels.
- Two pairwise similarity matrices across recent historical draws: exact
  shared-number positions versus transformed shape similarity.
- Optional **read-only** inclusion of already-saved v0.5 image forecast examples.
  Historical report hashes, draw identities, actual images and hit counts are
  checked before display; no model is trained.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 image-dynamics `
    --null-images 4096 --mapping-seeds 2 --transitions 12 `
    --matrix-count 24 --null-shape-pairs 128 `
    --out results/image_dynamics_v051.json `
    --html-out results/image_dynamics_v051.html

# Optional: include already-recorded v0.5 model output (no retraining).
& $py -m lotto7x7 image-dynamics `
    --compare-report results/image_compare_v05.json `
    --out results/image_dynamics_v051_with_models.json `
    --html-out results/image_dynamics_v051_with_models.html
```

These are **retrospective exploratory** plots, not a demonstration of an edge.
See [v0.5.1 methodology](docs/visual_image_dynamics_v051.md).

## v0.6 Phase 1: Image Shape Engine and 100-combination candidate pool

`shape-pool` establishes the first **shape-based number selection** workflow.
Six-point images are canonicalized under D4 rotations/reflections and
non-wrapping translations. A selected historical shape family generates
legal spatial placements; 20 families supply a default pool of 100 distinct
six-number combinations. Each result keeps the canonical bitmap, transformation
metadata and historical cutoff SHA256. The separate **1..5 random preview**
is not optimized and is not a purchase recommendation.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -v
& $py -m lotto7x7 shape-pool `
    --size 100 --families 20 --family-policy uniform-family `
    --sample-count 5 --seed 649 `
    --out results/shape_pool_v06.json `
    --html-out results/shape_pool_v06.html
Invoke-Item .\results\shape_pool_v06.html
```

Output: JSON candidate pool plus **offline HTML** showing the 20 canonical
shapes, 100 candidate six-pixel images and their numbers, and the five
separate random preview combinations. This version does not retrain AI or
implement the later Random / Shape-Diverse / Coverage-Optimized comparison.
The preserved historical files and v0.2–v0.5.1 models are unchanged.

See [v0.6 Phase 1 methodology](docs/shape_pool_v06_phase1.md).
Under an independent fair draw, even a graph-derived K-combination pool
has complete six-number coverage exactly K / C(49,6).


## v0.6 Phase 2–3 — Multi-Strategy Ticket Selection and Exact Coverage

From one frozen image-derived candidate pool (M=100), choose **N=1–5** distinct
combinations by random, shape-diverse, or exact-coverage greedy selection.
The optimizer prioritizes >=3 hit union coverage, with >=4/5 as lexicographic
tie breaks; it does not predict lottery draws. Audits compute **exact** fair
probabilities for 1–5 fixed tickets using combinatorial DP, and separately
estimate the 100-candidate research pool's partial-hit coverage using IID
Monte Carlo with uncertainty intervals.

```powershell
$py = '.\\.venv\\Scripts\\python.exe'
& $py -m lotto7x7 shape-pool --size 100 --seed 649 --out results/shape_pool_v06.json
& $py -m lotto7x7 bet-select --pool results/shape_pool_v06.json --count 5 --strategy all --out results/bet_selection_v06.json
& $py -m lotto7x7 coverage-audit --bets results/bet_selection_v06.json --out results/coverage_audit_v06.json
```

Read [the Phase 2–3 methodology](docs/shape_pool_v06_phase2.md).
A 100-combination research pool is not 100 purchased tickets; five distinct
bets have exactly **5 / 13,983,816** probability of all six main numbers in a
fair independent draw, regardless of selection strategy.


## v0.7 — Matched Baselines and Prospective-Ready Shadow Tracking

Compare **B0** global random tickets, **B1** uniformly random candidate pools,
**S1** historical image-derived shape pools, and **B2** disjoint portfolios.
B1/S1 use identical N=1..5, M=100, and Random / Shape-Diverse /
Coverage-Optimized selectors; all arms score on the same chronological targets.
Each historical target is excluded from the training prefix. Historical runs are
**exploratory**, not evidence of future prediction skill.

```powershell
$py = '.\\.venv\\Scripts\\python.exe'
& $py -m lotto7x7 baseline-study --warmup 200 --eval-draws 12 --size 100 --families 20 --count 5 --seed 649 --out results/baseline_study_v07.json
& $py -m lotto7x7 shadow-snapshot --size 100 --families 20 --count 5 --seed 649 --out results/shadow_pre_draw_v07.json
# Score only after actual next draw; replace ID and numbers with official results:
& $py -m lotto7x7 shadow-score --snapshot results/shadow_pre_draw_v07.json --draw-id 115000102 --numbers '01,02,03,04,05,06' --out results/shadow_post_draw_v07.json
```

The sample scored draw ID and numbers above are **placeholders**, not verified
results. Snapshot and scoring files use exclusive creation to avoid overwrites.
A local hash/timestamp cannot independently prove pre-draw existence; publish
the commitment externally before a draw to support blind evaluation.
See [v0.7 methods and safeguards](docs/baseline_blind_v07.md).


## v0.7.1 — Statistical Robustness Audit (S1 vs B1)

Re-analyze existing v0.7 matched baseline JSON **without rerunning candidate
pools**, using paired exact McNemar tests, three-strategy Holm family-wise
correction, draw-level paired bootstrap intervals, fair-coverage-adjusted
comparisons, and descriptive year groups. Score and exact fair coverage values
are recalculated from frozen tickets and outcomes.

```powershell
$py = '.\\.venv\\Scripts\\python.exe'
& $py -m lotto7x7 robustness-audit --input results/baseline_study_v07_100.json --out results/robustness_audit_v071_100.json
& $py -m lotto7x7 robustness-sweep --seeds 649,650 --counts 1,3,5 --warmup 200 --eval-draws 12 --stride 5 --size 100 --families 20 --out results/robustness_sweep_v071_smoke.json
```

The sweep can also use `--counts 1,2,3,4,5` and more seeds/targets, but each
configuration recomputes the full v0.7 baseline on identical target draw IDs.
**Seed replications are never treated as new independent lottery draws**;
S1-B1 differences are averaged across seeds *within each target draw*, then
bootstrapped over draws. Holm correction in a sweep covers all seed × N ×
strategy comparisons in that invocation; neither this nor the retrospective
analysis corrects for previous experiments or proves future predictive skill.

See [v0.7.1 statistical methodology](docs/statistical_robustness_v071.md).

## v0.7.2 — Full-History GPU Scenario-Coverage Replay

Analyze **all eligible historical targets**, instead of only the 12/100-target
functional samples. With 2,183 source drawings and a chronological warmup of
200, evaluate 1,983 separate target draws with an incrementally updated
past-only image-shape library, B0/B1/S1/B2 baselines, independent fair-draw
scenario optimization on CUDA via PyTorch, exact CPU postselection coverage
audits, and batched CUDA hit scoring.

The new six-portfolio `scenario-coverage` selector uses sampled fair scenarios
and **is not the same as** v0.6's exact-greedy `coverage-optimized` selector.
The old v0.7 and v0.7.1 methods are preserved. No GPU speed claim has been
benchmarked for the user's computer; CUDA is explicitly verified before use.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 gpu-check

# Smoke test: only five most recent target draws.
& $py -m lotto7x7 full-history-gpu --device cuda --warmup 200 --tail-targets 5 --count 5 --size 100 --families 20 --scenarios 8192 --seed 649 --checkpoint results/full_history_v072_smoke_checkpoint.jsonl --out results/full_history_v072_smoke.json

# Full 1,983 eligible targets (2,183 source images; first 200 are warmup-only).
& $py -m lotto7x7 full-history-gpu --device cuda --warmup 200 --count 5 --size 100 --families 20 --scenarios 8192 --seed 649 --checkpoint results/full_history_v072_checkpoint.jsonl --out results/full_history_v072_all.json

# After interruption, repeat the full command with --resume.
```

Documentation: [v0.7.2 full-history GPU methodology](docs/full_history_gpu_v072.md).
A CPU-only PyTorch build cannot use CUDA; select a compatible PyTorch build
using the official PyTorch Start Locally installer if the `gpu-check` command
reports CUDA unavailable. Historical replay remains exploratory, not prospective
prediction evidence.
