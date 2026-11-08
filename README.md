# Lotto7x7 — image-first research reboot

This repository begins **from scratch** with one problem: represent each Lotto649 draw as a **7×7 image containing exactly six active pixels**, study image shapes and their temporal order, and test whether a causal forecast resembles the next drawing more closely than an IID random outcome would.

There are **no inherited logits, MLP, legacy CNN classifiers, six-number jackpot likelihoods, legacy tests or phase-specific scripts** in the active tree. A new **CNN frame encoder + temporal GRU** is trained specifically for the 7×7 image-similarity objective. Old Git commits remain in ordinary Git history; they are not part of this rebuilt codebase.

## Preserved lottery records

The current 2004–2026 dataset contains **2,497** Lotto649 draws from one external Biga source, stored in `data/bin/lotto649_history_v1.bin` (including exact dates and bonus numbers), the model input `data/bin/lotto649_images_v2.bin` (Draw ID + 7×7 six-on bitmap), and 23 flat ASCII annual files `data/csv/lotto649_YYYY.csv`. The annual CSVs are deterministically exported from the verified history BIN. This is a single third-party data source, not independently verified against all original official drawings.

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

- **Source:** 23 flat annual `data/csv/lotto649_YYYY.csv` files, 2004–2026, with ASCII headers `date,draw_id,n1,n2,n3,n4,n5,n6,bonus`.
- **Output:** `data/bin/lotto649_images_v2.bin`, currently **2,497 draws**, **132,357 bytes**.
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
$py = '.\.venv\Scripts\python.exe'
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
$py = '.\.venv\Scripts\python.exe'
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
$py = '.\.venv\Scripts\python.exe'
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


## v0.7.3 — Independent Full-History Audit and Multi-Seed Robustness

Audit all **2,297 targets** against the canonical 2,497-draw Biga archive without
using the v0.7.2 CUDA scoring function. Independently read and compare the two
BIN wire formats against the source manifest, validate every chronological
cutoff and six-number portfolio, recompute the six strategies' match counts,
per-year summaries and paired S1-B1 statistics, and verify fair 6/49 coverage
(standalone analytic formula for disjoint tickets, existing exact DP for overlap).
Keep every existing source GPU report unchanged.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 full-history-audit --report results/v072_biga_2497_full.json --out results/v073_biga_2497_seed649_audit.json

# Compute GPU full-history runs with seeds 650 and 651 using unique checkpoint/
# output paths, then independently audit each. Read the detailed methodology.
& $py -m lotto7x7 full-history-multiseed --reports results/v072_biga_2497_full.json results/v072_biga_2497_seed650_full.json results/v072_biga_2497_seed651_full.json --audits results/v073_biga_2497_seed649_audit.json results/v073_biga_2497_seed650_audit.json results/v073_biga_2497_seed651_audit.json --out results/v073_biga_2497_multiseed.json
```

Each seed uses the **same historical targets**, so aggregation averages over
seeds *within each draw* and bootstrap intervals resample **draws only**.
Results are retrospective/exploratory, not an official source cross-check or
prospective evidence of a lottery prediction edge. Audit result files are
exclusive-create and tied to the input report SHA256.

[Full command examples and statistical limitations](docs/full_history_independent_audit_v073.md).


## v0.8 — GPU Causal Probability Research

This experiment shifts from shape-candidate *mathematical coverage* to
**individual next-draw number probability quality**. It compares fixed fair
6/49 IID marginals, smoothed historical-frequency marginals, recent
exponentially weighted marginals, and a GPU-trained 7×7 CNN+GRU model. All
models forecast the same chronological targets, using only prior published
draws. The network retrains every configurable block with an internal
past-only validation segment, and its 49 marginals are constrained to sum six.
Proper Brier Score / binary log loss and Top-6 hit distributions are reported.
A separate next-draw research forecast is produced using the latest known
historical prefix without access to an unseen winning result.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 predict-gpu --device cuda --warmup 200 --tail-targets 5 --epochs 12 --out results/probability_v08_smoke.json
& $py -m lotto7x7 predict-gpu --device cuda --warmup 200 --block-size 100 --epochs 12 --seed 649 --out results/probability_v08_full.json
```

With the current 2,497-draw Biga BIN, the complete experiment evaluates
2,297 chronological targets (one Top-6 research combination per model and
target). The previously reported v0.7.2 five-ticket union rates are **not**
comparable to this one-ticket forecast metric. A marginal sum of six does not
create a valid six-number joint probability model or demonstrate empirical
calibration. All historical results remain exploratory, not prospective proof.

See [v0.8 research protocol](docs/gpu_probability_research_v08.md).


## v0.8.1 — Audit existing GPU probability results without rerunning training

Re-score and statistically compare **saved v0.8 prediction JSON** for all 2,297
historical targets. Recompute Brier, binary log-loss and Top-6 hits for the
IID baseline and three predictors. Use paired McNemar with Holm correction,
per-draw and 100-target contiguous-block bootstrap sensitivity intervals,
plus year-level summaries. No retraining, CUDA setup, or old test suites.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 predict-audit --report results/probability_v08_full.json --out results/probability_v081_audit.json
```

Read [the v0.8.1 score-audit protocol](docs/probability_audit_v081.md).
This retrospective analysis cannot establish a predictive edge; source BIN
independent verification is a separate step.


## v0.9 — GPU Multi-Domain Feature Periodicity Discovery

This *discovery-only* experiment tests whether historical 7×7 legal six-number
lottery images contain recurring periodic structure in a multi-domain feature
bank: raw pixels, signed/second differences, 2D DCT, 2D FFT power, geometrical
and graph-pair motifs, and Sobel/Laplacian responses (254 channels). It scans
multiple recent horizons and finite-window Z-domain exponential radii, using
CUDA when selected. All tested periods, features, windows, and Z radii are
calibrated by **whole-search maximum** statistics on both fair legal 6/49 IID
surrogates and time-shuffled actual complete draw images.

```powershell
$py = '.\.venv\Scripts\python.exe'
# Optional: only 11 new tests, not the previously passed full suite:
& $py -m unittest discover -s tests -p 'test_feature_discovery_v09.py' -v
& $py -m lotto7x7 feature-discover-gpu --device cuda --holdout-draws 500 `
    --surrogates 199 --surrogate-batch 8 --recent-windows 256 512 1024 `
    --z-half-lives 0 64 256 --min-period 2 --max-period 512 --top 20 `
    --seed 649 --out results/v09_feature_discovery_cuda.json
```

Feature discovery is **not** a trained number-selection model and provides no
established forecasting advantage. The current 2,497 records allocate a 1,997
record discovery prefix and a 500 record held-out suffix for this experiment;
old v0.8 studies have already inspected that suffix, so genuinely prospective
validation remains essential. See [v0.9 method and controls](docs/multi_domain_feature_discovery_v09.md).


## v0.10 — GPU Deep Feature Discovery

Learn latent 7×7 six-hot image features using a CNN autoencoder and a causal
4/16/64-scale GRU next-image head, then **freeze the model** and search unseen
historical representations for latent spectral periods (finite Z-transform) and
repeated multivariate subsequence motifs (GPU matrix-profile-style nearest
neighbors). Calibrate each entire feature/period/motif-window search against
legitimate IID 6/49 images and shuffled historical draws. A separate historical
500-draw suffix is used for a predeclared Brier/Top-6 audit; it was previously
seen in v0.8 and is not genuinely prospectively blind.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 deep-feature-gpu `
    --device cuda --train-draws 1024 --holdout-draws 500 `
    --window 64 --validation 128 --epochs 12 --surrogates 99 `
    --surrogate-batch 4 --seed 649 `
    --out results/v010_deep_feature_cuda.json `
    --model-out results/v010_deep_feature_cuda.pt
```

The `.pt` checkpoint persists the fitted CNN/GRU weights for future
feature-conditioned ticket research **without rerunning this training**; it is
not a validated live selector. Only the 13 new tests were run in isolated
synthetic local verification, not the prior full suite or real RTX 4060 data.
See [v0.10 protocol](docs/gpu_deep_feature_discovery_v010.md).


## v0.11 — 7×7 Image Classification and Quantification (No Period Search)

Group all 2,497 historical images by **exact six-pixel D4+translation shape**,
**8-neighbor connectivity**, and **similar-image prototype classes**. The
5-of-6 shared-shape graph is counted but **not** misrepresented as a valid
mutually similar class system (chain effects form giant components). A CUDA
k-means prototype classifier uses 40 D4-invariant geometric descriptors,
or optionally the **frozen v0.10 CNN encoder** (`cnn` or `fused` feature modes).

All classes have member counts, an actual representative 7×7 drawing,
10000 legal fair-IID control draw proportions using the same frozen learned
centroids, and full assignment tables. Results are **descriptive**, not
winning-number probability claims. No retraining, no repeated historical
walk-forward/period scans, no previously completed regression suite.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 image-classify `
    --device cuda --clusters 12 --features geometry --iid-samples 10000 `
    --out results/v011_geometry_12.json
& $py -m lotto7x7 image-classify `
    --device cuda --clusters 12 --features fused `
    --model results/v010_deep_feature_cuda.pt --iid-samples 10000 `
    --out results/v011_fused_12.json
```

Each run creates JSON, CSV and an **offline HTML gallery** with colored 7×7
class representatives. Existing outputs are not overwritten. Read the
[v0.11 research and classification protocol](docs/image_classification_v011.md).

## v0.12 — Natural 7×7 Image Clustering and Similarity Galleries

Research **unknown image class counts**, without imposing the previous v0.11
twelve-cluster K-means assumption. The new CUDA-assisted
`natural-cluster-gpu` CLI compares *scikit-learn HDBSCAN* (density clusters
plus unclassified NOISE), *Ward hierarchical clustering* (silhouette-ranked
explicit K grid), and *K-medoids* (actual historic representative drawings).
Use D4-invariant 40-dimensional geometry features, or optionally fuse the
**frozen v0.10 CNN checkpoint** without retraining.

Silhouette, 80% subsample ARI stability, medoid distances, nearest and boundary
7×7 image galleries, full per-draw assignment CSV, and full-refit fair-IID
controls accompany the quantitative class counts. IID can produce stable,
apparently clean clusters too; **none of these outputs establishes a
winning-number probability advantage**.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m pip install -r requirements.txt

& $py -m lotto7x7 natural-cluster-gpu `
    --device cuda --features geometry `
    --k-values 4 6 8 12 16 24 32 `
    --min-cluster-size 30 --min-samples 10 `
    --iid-replicates 9 --iid-samples 10000 `
    --seed 649 --out results/v012_geometry.json

& $py -m lotto7x7 natural-cluster-gpu `
    --device cuda --features fused `
    --model results/v010_deep_feature_cuda.pt `
    --k-values 4 6 8 12 16 24 32 `
    --min-cluster-size 30 --min-samples 10 `
    --iid-replicates 9 --iid-samples 10000 `
    --seed 649 --out results/v012_fused.json
```

Each run creates JSON, CSV, and offline HTML files using the output stem;
previous reports are preserved. The 15 newly added tests passed locally;
previous test suites and GPU predictive backtests were not re-executed.
[Details, scientific limits, and controls](docs/natural_clustering_v012.md).

## v0.13 — GPU Pairwise Image Similarity (Exact Six-Point Matching)

Rather than assume an arbitrary number of natural image classes, compute the
**entire 2,497 × 2,497** pairwise distance matrix: 3,116,256 unique pairs.
Each pair solves all **720 one-to-one six-point assignments** for:

- `position`: optimal matching in the original board coordinates;
- `d4`: minimum after quarter-turn rotations/reflections;
- `shape`: minimum after D4 and **continuous centroid translation**;
- `shape_hausdorff`: alternative centered, D4-minimized Hausdorff distance.

The `shape` transform is **not limited to integer/non-wrapping board shifts**.
The offline HTML gallery shows actual images beside an aligned overlay. Only
groups whose **entire within-class pairwise diameter** fits a predefined
fair-IID distance threshold are reported as true complete-link classes; other
images remain singleton. Independent IID pairs and nine matched sample
refits are descriptive controls, not proof of any selection advantage.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 pairwise-similarity-gpu `
    --device cuda --batch-size 512 `
    --iid-pairs 30000 --iid-quantiles 0.001 0.005 0.01 `
    --cluster-mode shape --control-draws 384 --control-replicates 9 `
    --top-pairs 25 --gallery-count 8 --seed 649 `
    --out results/v013_pairwise_similarity.json

Start-Process .\results\v013_pairwise_similarity.html
```

Each run creates JSON, CSV, HTML and compressed **NPZ containing all four
full distance matrices**. Existing files are never intentionally overwritten.
The only locally run checks were **16 new v0.13 unit tests** and a
128-historical-draw integration analysis (CPU because the execution container
has no CUDA GPU); the full 2,497-draw CUDA run must be performed separately.
[Full method, controls and interpretation](docs/gpu_pairwise_image_similarity_v013.md).

## v0.14 — Gaussian Shape Distribution & Density-Based Classification

**Reuses the precomputed v0.13 `shape` distance NPZ** for all 2,497
historical images. No repeated 3.1-million-pair matching or CNN training.

- Multi-bandwidth Gaussian radial **leave-one-out similarity** scores computed
  on CUDA (these are not normalized PDFs in a non-Euclidean shape quotient).
- Density peak selection and nearest-mode assignment with explicit NOISE and
  a **maximum of 24 displayed** centers; a hit to that cap is reported, not
  misread as a natural number of modes.
- Approximate 8D classical MDS embedding with reported distance distortion,
  followed by diagonal-Gaussian mixture classification, BIC search and soft
  per-image membership probabilities.
- Legal uniform 6/49 **matched-size IID refits**, including both KDE and
  Gaussian-mixture BIC contrasts. No unjustified independent-pair p-values.
- JSON, per-draw CSV, HTML gallery with density histogram and BIC table, and
  lightweight embedding NPZ (all outputs protected from overwrite).

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-gaussian-gpu `
    --device cuda `
    --matrix results/v013_pairwise_similarity.npz `
    --assignments results/v013_pairwise_similarity.csv `
    --bandwidths 0.35 0.55 0.8 1.1 `
    --mode-bandwidth 0.55 --peak-radius 0.6 `
    --assignment-radius 0.85 --max-peaks 24 `
    --mds-dimensions 8 `
    --gmm-components 1 2 3 4 6 8 12 16 24 `
    --control-draws 128 --control-replicates 9 `
    --batch-size 512 --gallery-count 8 --seed 649 `
    --out results/v014_shape_gaussian_cuda.json

Start-Process .\results\v014_shape_gaussian_cuda.html
```

For an extended IID run, use `--control-draws 256` with a new output path.
The output is a **descriptive classification**, not a forecast. Only the
new v0.14 tests (14 PASS) and a cached-real-data CPU experiment were run;
old GPU backtests and old suites were not repeated.

Read [the Gaussian shape analysis protocol](docs/gaussian_shape_distribution_v014.md).

## v0.14.1 — P0 Gaussian Shape Validation (Batch 1 complete)

Single-batch implementation of all four P0 tasks:
(1) exact legal-6/49 matched-size historical vs independently refit IID with
**identical sample size, MDS dimension, candidate K and GMM settings**, including
one full-N IID refit; (2) GMM ARI/NMI stability over paired overlapping
subsamples; (3) MDS 4/8/12/16-dimensional stress and BIC sensitivity; and
(4) KDE bandwidth rank, prototype and density-CV sensitivity. Reuses **all
2,497 historical draws** in the existing v0.13 `shape` NPZ and makes **no**
new historical pairwise calculations or CNN retraining.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-gaussian-validate `
    --device cuda `
    --matrix results/v013_pairwise_similarity.npz `
    --assignments results/v013_pairwise_similarity.csv `
    --dimensions 4 8 12 16 `
    --comparison-dimension 8 `
    --gmm-components 1 2 3 4 6 8 12 16 24 `
    --gmm-n-init 2 `
    --bandwidths 0.35 0.55 0.8 1.1 `
    --matched-size 384 `
    --matched-replicates 9 `
    --full-iid-replicates 1 `
    --stability-seeds 649 650 651 652 `
    --stability-fraction 0.8 `
    --batch-size 512 `
    --seed 649 `
    --out results/v0141_p0_gaussian_validation.json

Start-Process .\results\v0141_p0_gaussian_validation.html
```

Creates JSON, long-form CSV, offline HTML and compact features NPZ without
overwriting prior v0.13–v0.14 outputs. Full-size IID generates **new**
random images and re-computes only those controls' distances. No statistical
significance claim is justified from just one full-size IID replicate.
**Development checks:** 20 new tests passed and a full historical-distance
CPU integration smoke test succeeded. **Research acceptance** requires
running the above complete CUDA experiment and reviewing its outputs.
[Detailed P0 protocol, interpretation limits and smoke settings](docs/gaussian_model_validation_p0_v0141.md).

## v0.15 + v0.16 — P1 Batch 2: Exact Shape Catalog

One command implements the entire P1 batch, without freezing the unstable
8D-GMM K=12 classification. Uncapped **empirical Gaussian KDE local modes**
are explored at 12 σ×radius combinations with coverage, persistence and
intra-/inter-class distance diagnostics. A separate **fixed D4-invariant
graph taxonomy** defines six component-count classes, eleven component-size
partitions, and 94 orthogonal/diagonal-edge profiles. Exact D4+translation
canonical families are also cataloged, including mathematically exact orbit
sizes for the historically observed families and an unseen-family complement.

The complete **13,983,816 legal 6/49 combinations are actually enumerated**
using bounded-memory NumPy streaming and classified with the **same graph
rules** as history; theoretical frequencies are exact. Exploratory
multinomial Monte Carlo p-values, Bonferroni across three nested levels and
BH-FDR per-class corrections compare history with the exact baseline.
All eight output files are protected from overwriting prior results.

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

The full enumeration and complete historical integration were executed locally
using the uploaded v0.13 distance cache: 13,983,816/13,983,816 processed;
**31/31 new P1 tests pass**; all eight outputs verified. CPU is sufficient
for full-universe counting, and CUDA remains an option for historical KDE.
No historical distances, CNN checkpoints or P0 results were changed.

**Caution:** Negative shape-distance silhouettes for graph taxonomy indicate
these are interpretable countable structural codes, not guaranteed natural
Euclidean clusters. KDE sample modes lack exact all-universe probabilities.
[Research methods and full output specification](docs/shape_taxonomy_complete_universe_p1.md).

## v0.17 + v0.18 — P2 Batch 3: Predictive Validation + Shape Explorer

**P2 implemented as one batch.** Directly reuses verified P1 shape taxonomy,
precomputed historical records and all 13,983,816 exact fair combination
counts. No v0.13 historical pairwise recomputation or CNN retraining.

- **v0.17**: per-draw *causal* one-step forecasts of L1/L2/L3 graph classes,
  with a frozen Dirichlet shrinkage rule `q_t=(past_count + 384*p_fair)/(t+384)`.
  Reports proper sequential log-gain over exact IID fair priors and
  **1,000 independent IID full-sequence refits** preserving the nesting of L3,
  L2 and L1 classes, with 3-level multiple-comparison correction.
- Equal-budget ticket backtest: generate the same fresh 64-ticket
  uniform legal candidate pool before each draw; compare 5 uniformly
  chosen tickets to 5 scored by **historical L3 q/p**. A *conditional*
  frozen-ticket IID future-draw simulation provides exploratory control
  without claiming it is a full adaptive strategy refit.
- **v0.18**: fully offline, interactive browser Shape Explorer. Input any
  six different numbers (1–49), see their 7×7 image, frozen topology, exact
  D4+legal-translation orbit size and fair category mass, and *all-history*
  nearest shapes computed using 8 D4 transforms and exact six-point 64-state
  minimum-cost assignment (not MDS approximate lookup).
- Rare-class `P6_O1_D5` included in an explicit **future-only frozen
  watchlist**; after-the-fact historical results never count as independent
  preregistered evidence. Never grants special forecast weight.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-p2-batch `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --min-train 500 `
    --tau 384 `
    --iid-replicates 1000 `
    --ticket-null-replicates 128 `
    --candidate-pool 64 `
    --tickets-per-draw 5 `
    --seed 649 `
    --out results/v017_v018_p2_batch.json

Start-Process .\results\v017_v018_p2_batch_report.html
Start-Process .\results\v017_v018_p2_batch_explorer.html
```

Seven new, nonoverwriting outputs: JSON audit, walk-forward CSV, last
backtested tickets CSV (NOT next picks), report HTML, interactive offline
Explorer HTML, NPZ auditable timeline and future-only watchlist JSON.

**Development PASS:** 24/24 new unit and integration tests; Node.js browser
implementation independently checked against six historical images and 36
v0.13 distance cells (maximum error under 6e-8); all seven outputs created
and inspected from real 2,497-draw data, 1,997 historical holdouts, 1,000
coupled IID sequences and 128 independent ticket-draw simulations.

**Research result:** under these frozen exploratory settings, all three
historical class-forecast log-gains against IID were negative. Some ticket
overlap differences arose in this one history but not reliably significant.
No verified advantage in future fair 6/49 outcomes has been established.
See [P2 research protocol and limitations](docs/shape_p2_predictive_validation_explorer.md).

## v0.19 — Future-only Prospective Validation (P3)

**No additional lottery draw is currently claimed.** After P0–P2's entirely
retrospective experiments, v0.19 freezes the verified 2,497-draw P1 baseline,
rule hash, exactly enumerated fair 6/49 masses, fixed `tau=384` predictor,
predeclared `P6_O1_D5` alternative, and the Taiwan-local freeze date
**2026-11-08** in
[`docs/prospective_freeze_20261108.json`](docs/prospective_freeze_20261108.json).
This manifest is intended to remain unchanged, with committed history.

The update command validates SHA-256 of both original P1 source files and
the sealed freeze, scans supplied annual CSV sources for conflicts with
frozen historical records, requires Gregorian draw dates **strictly after**
the freeze date for new observations, and rejects duplicate or skipped
within-year issue IDs. Every future class-probability forecast is scored
**before** that draw enters its online counts. No data-provider verification
is performed by the local program.

Four preregistered nonnegative martingale e-processes are monitored under
fair IID: L1, L2, L3 fair-class-normalized sequential forecasts, plus a fixed
rare-category alternative at 10× its exact fair frequency. Their equal
weighted mixture is an **anytime-valid e-process**, with a prespecified
research-review threshold `mixture_e >= 20` (0.05 familywise bound for this
specific mixture). A threshold crossing requests independent investigation:
it is not proof of bias or improved winning-number odds.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-prospective-update `
    --freeze docs/prospective_freeze_20261108.json `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --csv data/csv `
    --out results/v019_prospective_first_check.json

Start-Process .\results\v019_prospective_first_check.html
```

When a genuine new draw arrives, append the authoritative English-header
`date,draw_id,n1,n2,n3,n4,n5,n6,special` record to the annual CSV and rerun
using a **new** output name plus optional `--previous` for append-only
validation of already published prospective event scores. Never reset the
committed freeze to include newly observed events. A separate
`shape-prospective-freeze` CLI supports designing an unrelated new experiment
before new outcomes.

**Initial zero-data acceptance** on uploaded P1 files:
`NO_PROSPECTIVE_DRAWS`, `eligible_new_draws=0`, `mixture_e=1`,
`anytime_p_upper_bound=1`. **25 new tests PASS**; JSON, CSV and offline
HTML from the complete current-history scan verified. Only **future**
draws with externally verified sources can provide prospective evidence.

[Detailed study design, statistical justification and update protocol](docs/prospective_validation_v019.md).

## v0.20 — Pre-Draw Commitments and Post-Draw Source Audit

**First publicly committed pre-draw class-distribution snapshot:** Lotto649
issue `115000104`, standard draw date `2026-11-10`, is sealed in
[`docs/prospective_seals/115000104_2026-11-10.json`](docs/prospective_seals/115000104_2026-11-10.json)
before that drawing. It includes complete pre-result probability vectors
for L1's 6, L2's 11 and L3's 94 fixed shape classes, plus frozen
`P6_O1_D5` monitoring odds; **no six-number results or picks**. Seal:
`f962047a8a3404bd20763b03fb9abe0b0db651ff5497882ba42a103408c496b3`.

**v0.20** adds two subcommands, 17 new tests, and checks that any
pre-draw snapshot is for the next contiguous issue, a standard Tue/Fri date,
before the 20:30 Asia/Taipei drawing deadline. Post-draw auditing reconstructs
the entire prior probability vector and checks v0.19's event class,
fair probability, causal forecast, log gain and rare-class increment.
It rejects altered snapshots, rewritten prior events and inconsistent scoring.
An independently timestamped public Git push is necessary to prove
pre-draw commitment timing; a local creation timestamp is not sufficient.

Use the **already committed** seal for the 2026-11-10 target. Do not replace
it with a newly computed snapshot. Once the official result is available,
verify its six numbers/date/issue externally, append them to the appropriate
year CSV and run:

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-prospective-update `
    --freeze docs/prospective_freeze_20261108.json `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --csv data/csv `
    --previous results/v019_prospective_first_check.json `
    --out results/v019_after_115000104.json

& $py -m lotto7x7 shape-prospective-audit `
    --seal docs/prospective_seals/115000104_2026-11-10.json `
    --report results/v019_after_115000104.json `
    --out results/v020_115000104_audit.json

Start-Process .\results\v020_115000104_audit.html
```

For later dates, generate a new `shape-prospective-seal` using the
**previous**, already scored v0.19 report, and commit/push that new seal
**before the draw**. All outcomes still need verification against official
Taiwan Lottery or CTBC. The monthly annual-data download is not a same-night
feed; no unverified web scraper is installed. This is a **class-forecast
integrity audit**, not an endorsement of lottery prediction or betting.

**Validation:** 42/42 v0.19+v0.20 unit/CLI tests pass; real frozen
2,497-draw P1 inputs and user-submitted zero-event v0.19 report generated
the initial sealed snapshot. First post-draw audit currently correctly
returns `AWAITING_DRAW` and contains zero new outcomes.

[Full verification and evidence limitations](docs/pre_draw_commitment_v020.md).

## v0.21 — Unified Number Selection Engine

**Working one-command integration** of the existing P1 full-6/49 theoretical
category probabilities, v0.17 frozen L3 shape-class score, and v0.6 exact
fair portfolio coverage algebra. All three approaches choose the **same N
tickets** from the **same uniform, nonduplicate** candidate pool; there is
no historical-shape weighting during candidate generation.

Strategies:
- `uniform`: first N uniform candidates (benchmark).
- `coverage`: finite-pool **exact-oracle greedy plus one improving swap
  pass**, maximizing selected `3plus`/`4plus`/`5plus` union-coverage counts.
  It never returns a worse primary objective than the uniform benchmark in
  that same pool. This is a local heuristic **not** a global optimum.
- `shape_score`: highest v0.17 frozen Dirichlet-smoothed L3-class
  `q(c)/p_fair(c)` scores. This is a descriptive historical ratio,
  **not a validated AI neural selector or higher winning odds**.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-select-unified `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --freeze docs/prospective_freeze_20261108.json `
    --pool-size 100 `
    --tickets 5 `
    --objective 3plus `
    --seed 649 `
    --out results/v021_unified_selection.json

Start-Process .\results\v021_unified_selection.html
```

Four immutable result files: JSON, `_pool.csv` (all candidates, frozen
categories and q/p), `_bets.csv` (three distinct strategy portfolios), and
offline HTML with **7×7 image grids** and mathematically exact >=3/4/5/6
union probabilities over all 13,983,816 legal draws. The original
v0.19 P1 freeze, v0.20 first pre-draw seal and all historical inputs
remain **read-only**.

**Verified:** 32 new v0.21 tests PASS; 74/74 combined v0.19/v0.20/v0.21
tests PASS. Full integration with user's actual frozen 2,497-draw P1
artifacts passed: 100 uniform candidates, five bets, fixed seed 649.
Exact at-least-one-ticket >=3 coverage: uniform 0.0887766973;
coverage-local-optimized 0.0929016801; experimental shape score
0.0906839735. All three have the **same six-main-number jackpot
probability 5/13,983,816**. The displayed shape-score difference
does not establish predictive superiority; it is one seeded candidate
portfolio comparison, not a prospective test.

[Detailed design, exact probability definition, objective trade-offs and
validation](docs/unified_selection_v021.md).

## v0.22 — Timed Historical Exact-Hit Benchmark (20-minute local experiment)

This command **runs on the user's own Windows PC** and searches past known
drawings, using **up to 10,000 unique candidate tickets per independent trial**
for up to **20 minutes**. Each exact six-main-number match prints immediately
with the old draw ID, winning six numbers and the first-hit ranks in the
**uniform** order versus the frozen P1 **shape_score** (L3 q/p) order.

Two strategies examine the **same candidate pool** each trial, so they must
find the **same historical exact-six matches** upon completing all 10,000 guesses.
Only **how early** a hit is discovered can differ. Repeated trials may contain
the same previously drawn ticket; both total hit events and unique historical
draws are reported. The v0.21 3+/4+/5+ multi-ticket coverage optimizer is not
relevant to searching a known 6/6 historical match and is not included.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 historical-hit-benchmark `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --minutes 20 `
    --per-round 10000 `
    --target all `
    --seed 649 `
    --progress-every 100 `
    --out results/v022_historical_20min.json
```

The command writes three **non-overwriting** artifacts: JSON summary,
`_hits.csv` (flushed on every match) and `_rounds.csv` (including no-hit
trials). It respects Ctrl+C and records completed-trial results.
For a more disciplined retrospective research comparison, use
`--target holdout --holdout-count 500`: learn L3 shape frequencies on the
first 1,997 historical draws and search only the last 500.

**Critical:** `--target all` includes all 2,497 historical target draws in
its score training and **leaks target information**. Any apparently rapid
shape-score hit is an in-sample retrieval outcome, **not a prediction result**.
The last-500 holdout was inspected during prior project research, so that
alternative is also exploratory, not genuinely blind prospective evidence.

**Validation:** 14/14 v0.22 tests and 88/88 selected cross-version tests
PASS. Seed 649 produced two historic full-six hits in the first 10,000
uniform guesses at ranks 4,278 and 9,627. In the *same* candidate pool,
in-sample L3 shape ordering moved those ranks to 3,786 and 1. Three
10000-ticket trials had five historical hit events total; a separate
last-500-target check gave one hit in three trials.
[Full methodology and limitations](docs/historical_guess_benchmark_v022.md).
