# Lotto7x7 — image-first research reboot

This repository begins **from scratch** with one problem: represent each Lotto649 draw as a **7×7 image containing exactly six active pixels**, study image shapes and their temporal order, and test whether a causal forecast resembles the next drawing more closely than an IID random outcome would.

There are **no inherited logits, MLP, legacy CNN classifiers, six-number jackpot likelihoods, legacy tests or phase-specific scripts** in the active tree. A new **CNN frame encoder + temporal GRU** is trained specifically for the 7×7 image-similarity objective. Old Git commits remain in ordinary Git history; they are not part of this rebuilt codebase.

## Preserved lottery records

The historical dataset files under `data/raw/`, `data/csv/`, `data/binary/` and `data/normalized/lotto649_*.{npy,parquet}` are preserved **byte for byte** from the original Git blobs. Non-draw reports and experimental artifacts are removed from the rebuilt working tree. The default analysis input is `data/normalized/lotto649_base.npy` (structured fields `date`, `draw_id`, `numbers`, `special`). The special number is validated but excluded from the six-pixel image.

## Fresh implementation

- `lotto7x7/core.py`: reversible six-number ↔ 7×7 bitmap; sequential, snake and preseeded permutation mappings; full FFT and DCT roundtrip; geometric similarity with D4 rotations, translations, row/column occupancy and low-frequency DCT; conditional fair-IID baseline.
- `lotto7x7/learning.py`: **trainable image learning** with a 2D CNN per historical frame, recurrent GRU across frames, D4/translation-tolerant shape loss (image autocorrelation), auxiliary pixel reconstruction and six-active-point constraint; trained 7×7 image output with six brightest pixels selected. This does not import the retired six-of-49 logit decoder.
- `lotto7x7/models.py`: simple image-only causal controls (`previous`, `centroid_motion`, `temporal_analog`) and image-output diversity diagnostics; these are *non-neural baselines*, not the new learning model.
- `lotto7x7/research.py`: read-only historical archive importer; chronological train/validation/evaluation splits; conditional-IID-adjusted similarity; within-year time-order shuffling; validation-only method selection and per-draw JSON audit.
- `lotto7x7/__main__.py`: four commands: `check`, `learn` (main training path), `replay` (heuristic controls), and `gallery` (supports either report).
- `tests/`: tests written **only for this newly designed 7×7 project**.

**Primary research outcome**: average `image_similarity(prediction, actual) - E_IID[image_similarity(prediction, random_draw)]`. A graphically similar forecast may have zero common numbers. The analysis is not a lottery ticket selector and contains no guarantee of an edge over random draws.

## Windows PowerShell

```powershell
cd D:\my-github\lotto649
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
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

The new test suite has **31 independent tests**, including synthetic pattern learning, differentiable gradient flow, chronology and future leakage controls, DCT/FFT reversibility, visual similarity, and report generation. No old 169-case suite is retained.

## Canonical binary draw dataset (v1)

The original annual CSV files remain the source of truth. A deterministic
**BIN v1** derived from the 20 annual **Lotto649** CSV files (2007–2026)
is available at [`data/bin/lotto649_draws_v1.bin`](data/bin/lotto649_draws_v1.bin).
It currently contains 2,183 consecutive chronological records (through
2026-10-30), totaling 24,029 bytes.

The binary format contains **only** the requested eight logical fields:
`draw_id:uint32`, `main_number[6]:uint8[6]`, `special:uint8`. A
16-byte versioned header precedes 11-byte packed little-endian records.
The format is documented in [`docs/data_bin_format_v1.md`](docs/data_bin_format_v1.md).
The special number is **not** part of the six-pixel image.

```powershell
# Regenerate from all 20 CSV source files and verify each parsed record.
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset

# Verify the generated binary without modifying it.
.\.venv\Scripts\python.exe -m lotto7x7.bin_dataset --verify data/bin/lotto649_draws_v1.bin

# Compare the regenerated BIN with the GitHub-tracked canonical file.
git diff --exit-code -- data/bin/lotto649_draws_v1.bin
```

CSV sources with an unpadded date (such as `2024/12/3`), UTF-8 BOMs,
extra unused columns, or full-range filenames (such as 2018 and 2020)
are explicitly supported. No existing CSV is rewritten. This BIN is a
data-storage artifact; the current `learn` command still reads the
normalized NumPy archive. Switching the trained-model input to BIN
should be a separate verified change.
