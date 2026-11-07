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
- `lotto7x7/__main__.py`: four commands: `check`, `learn` (main training path), `replay` (heuristic controls), and `gallery` (supports either report).
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

## Research guardrails

The archive has already been explored in earlier work. A chronological test segment is useful for engineering checks but **not a new untouched blind test**. Do not tune shape weights, mappings, or analog windows on test scores and then claim improved future performance. The physical lottery machine has no demonstrated relationship to pixel geometry. An IID draw can resemble other IID draws; thus, high **raw** visual similarity by itself is not evidence of an advantage. Historical data remain unchanged. New prospective image-similarity research requires separately committed protocols and time-stamped pre-result predictions.

## Image learning vs. hand-built image statistics

The `learn` command **optimizes trainable CNN and GRU parameters** on the first chronological 60% of the 7×7 images. It chooses the epoch based on the next 20% (validation), and evaluates only the last 20% with a fixed image-similarity metric. The model consumes previous image frames and produces a 7×7 spatial output: there is no previously trained six-number model. For learning stability the differentiable training loss uses a 35% shape-autocorrelation term (invariant to translation and quarter-turn/reflection), 60% image-pixel reconstruction (weighted binary cross entropy), and 5% six-pixel mass penalty. The **evaluation objective**, however, is the external image-similarity score adjusted by the shape-specific IID expectation, never jackpot 6/6. This trainable path is distinct from the simple analog controls.

`results/learned_image_weights.pt` is a newly trained state dictionary. Only load weights from a trusted source using the corresponding architecture and safe framework loading settings. `results/learned_replay.json` contains per-draw output masks and adjusted similarity; the HTML gallery draws observed/predicted 7×7 images.

The new test suite includes **53 tests** (31 image-core tests, 15 BIN format tests and 7 BIN model-integration tests), including synthetic pattern learning, differentiable gradient flow, chronology and future leakage controls, DCT/FFT reversibility, visual similarity, and report generation. No old 169-case suite is retained. See [BIN v2 layout](docs/data_bin_format_v2.md).

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
