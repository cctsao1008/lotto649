# Lotto649 v0.22.1 — Quiet, paced historical exact-hit watch

## Fixed behavior and cause

v0.22 accidentally interpreted `--per-round 10000` as **ten thousand guesses in EVERY trial** and repeated trials as fast as the CPU could run until the 20-minute timer expired. The benchmark therefore produced hundreds of independent 10,000-ticket trials within seconds, printing every historical match and repeatedly matching previously seen historical combinations.

v0.22.1 changes the user-facing `historical-hit-benchmark` command:

- **10,000 is a TOTAL per execution**, not per trial. Accept `--max-guesses 10000` or legacy `--per-round 10000` as an alias for the same total cap.
- One deterministic (seeded) uniform candidate pool is sampled without replacement; no ticket repeats within one run.
- Candidates are searched in the chosen order: `--strategy shape_score` (default) uses the frozen P1 L3 `q/p` ranking; `--strategy uniform` uses generation order. Both orders operate on **the exact same pool**, so both find the same exact six-number combinations if allowed to exhaust all 10,000.
- **Only on an actual historical 6/6 match** write a terminal line with cumulative hit number (`[HIT #k]`), cumulative 1-based guess number (`guess #i/10,000`), the six numbers and historical issue. `uniform_rank` and `shape_rank` are supplied as counterfactual order comparisons.
- No trial-by-trial output and no `[PROGRESS]`. One `[DONE]` line summarizes the run. Every matched hit is flushed to `_hits.csv` immediately; JSON is updated after each hit and when the run ends or is interrupted. A single `_rounds.csv` summary record is written at the end for output-name compatibility.
- The whole **10,000 guesses are paced across 20 minutes** by sleeping between guess positions (first at time zero and final at 20 minutes). On Windows this is a throttled **display and experiment pacing schedule**, not 20 minutes of extra AI training. For faster results use a shorter `--minutes` duration, keeping `--max-guesses` the same.
- Ctrl+C records partial progress with `INTERRUPTED` status. New output basenames are required to avoid accidentally overwriting prior result files.

## Run on Windows

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 historical-hit-benchmark `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --minutes 20 `
    --max-guesses 10000 `
    --target all `
    --strategy shape_score `
    --seed 649 `
    --out results/v0221_historical_quiet_20min.json
```

**Expected terminal appearance** for the known fixed Seed 649 is only two `HIT` lines and a final `DONE` line:

```
[HIT #1] guess #1/10,000 numbers=01 08 16 22 24 30 historic=93000103 (uniform=#9,627; shape=#1)
[HIT #2] guess #3,786/10,000 numbers=02 06 18 33 34 43 historic=100000081 (uniform=#4,278; shape=#3,786)
[DONE] COMPLETED: guesses=10,000/10,000, hits=2, elapsed=1200.0s
```

The exact elapsed seconds can vary slightly due to OS sleep timing. `--seed 649 --target all` generates a known candidate pool; changing seed changes the actual candidates, not the expected fair distribution.

## Statistical interpretation

With 2,497 historically observed distinct six-number sets among 13,983,816 legal sets, 10,000 uniform distinct guesses have an expectation of approximately **1.786** historical matches. The selected order does **not** alter the total number of hits in the full 10,000 candidates, only *when* they appear. The v0.22.1 20-minute pacing likewise does not change how many there are. Because `--target all` estimates class scores on the same historic draws it is searching, finding a match at shape rank 1 is **in-sample retrieval**, not evidence that the method can predict future lottery draws. For a cleaner retrospective comparison use `--target holdout --holdout-count 500`, acknowledging the project's prior exploration of the suffix.

## Verification

- Existing v0.22 core helper functions are preserved for reproducibility, while the public CLI is routed through a new safe paced runner.
- Added tests for total 10,000 cap, exact cumulative attempt and hit numbers, silence without hits, unique candidate pool, simulated-clock 20-minute pacing, Ctrl+C partial audit, same-pool strategy equivalence, CLI aliases and output protection.
- 97 selected old and new tests passed locally.
- Actual uploaded 2,497-draw P1 NPZ/JSON integration: 10,000 guesses, fixed Seed 649, **two** exact historical matches; no progress spam. The brief smoke test compressed the display duration by setting very small `--minutes`; real Windows command above uses 20 minutes.
