# Lotto649 Historical Replay — Full-Six Prediction Diagnostics

Status: **implementation committed; Windows tests and official 2025 replay execution pending**.

## Purpose

Phase 5 already tested the full-six probability log score for the frozen static, lag and MLP candidates relative to IID. It did **not** print their actual diagnostic highest-probability six-number subsets for each historic draw. The user correctly requested a more direct, observable **predict-first, compare-after** test before considering any deployment.

`lotto_replay.py` adds a read-only, **exploratory** retrospective replay of all draw outcomes in a requested year, default **2025**, using the official `data/normalized/lotto649_base.npy` input (main six only; special excluded). This is not an executable ticket recommender and does not modify Forecast #001.

### Frozen chronological isolation

The code **does not use the final archive length to shift training boundaries**:

- Train target indices: `5..1308` (exclusive cutoff `1309`), taken from the original Phase 5 protocol.
- Validation indices: `1309..1745` (exclusive cutoff `1746`).
- Evaluation begins at index `1746` (2023-01-31 in the original archive).
- `--year 2025` restricts evaluation to the 2025 subset of that held-out period (**118 previously observed draws**, if the original 2,183-draw source is unchanged).
- Static and MLP retain Phase 5 fitting, identical seed 649, window=5, hidden=16, up to 40 epochs, patience=6, batch size 128, learning rate 0.005 and weight decay 0.02.
- Lag beta is fitted on train-only overlap counts, not the evaluation year.
- For target draw t, Static/MLP use features from draws `[t-5, ..., t-1]`. Lag uses `t-1`; IID uses none.
- Evaluated target `S_t` is used **only after the forecast vector and diagnostic top six have been formed**. Earlier *realized* test draws are permissible causal history for one-step evaluation.

The existing historical archive has been studied already. This replay is not a newly blind test. Repeating test-year searches or editing hyperparameters based on the outputs would increase retrospective selection bias.

### Four models, two complementary diagnostics

All probability models are exact fixed-cardinality distributions over **13,983,816 complete sets**:

`P(S) = exp(sum_{i in S} l_i) / sum_{|T|=6} exp(sum_{j in T} l_j)`

**Diagnostic top six:** select six numbers with the largest 49 output log weights (highest-probability complete set under this model family). A reproducible random secondary key breaks any tied logits **before reading the target**, including the fully tied IID case. This is a retrospective diagnostic, **not** a recommended future lottery combination.

**Score and realized-combination rank:** after revealing the target, compute the exact full-set log score `log P(S_t)` and compare to `-log(C(49,6))`; also estimate where that realized combination would rank according to each model's assigned probability. Lower `rank_fraction_lower_is_better` means more combinations are scored below the actual result.

- IID ranks are **exact midranks** at the middle of the universe because all 13,983,816 complete sets are equiprobable.
- Lag ranks are **exact combinatorial midranks** using its previous-draw overlap classes (0 through 6).
- Static and MLP ranks are **Monte Carlo estimates**, using 50,000 pre-seeded uniform combinations shared across test dates; 95% per-draw Hoeffding sampling half-width is about **0.0061 of the total rank range**. This quantifies sampling error only, not model correctness or statistical evidence across dates.
- A 1-indexed `observed_set_midrank` estimates the average rank of the realized six-subset, including ties.
- IID produces a diagnostic random six as a negative-control comparison; under IID all candidates are equally likely.
- Bonus activity results and the special seventh number are **never** targets.

### Fair baseline

For any fixed six-number set under independent fair drawings:

- Exact-six match probability = `1 / 13,983,816` per draw.
- Expected number of matched main numbers = `6*6/49 = 0.7346938776` per draw.
- Matched-number count follows the hypergeometric distribution `P(J=j)=C(6,j)C(43,6-j)/C(49,6)`.
- For 118 draws, expected six-of-six matches are **0.00000844**; observing zero complete matches is entirely expected and does not discriminate among models.

The key requirement for predictive value is reliable out-of-sample/prospective **distributional** advantage over IID, not a single high-overlap historic result. Even if top-six hit counts look better in one previously examined year, they require properly controlled replication.

## Windows instructions

```powershell
cd D:\my-github\lotto649
git pull origin main

# Expected: 96 tests, compared with the 83 previously passed.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 2025 retrospective diagnostic; keep the existing 60/20 Phase 5 boundaries
.\.venv\Scripts\python.exe .\lotto_replay.py `
    --source data/normalized/lotto649_base.npy `
    --year 2025 `
    --rank-samples 50000 `
    --show-model mlp `
    --show-draws
```

Expected output artifacts (not checked into Git):

- `data/neural/historical_replay_2025.json`: full reproducible metadata, all per-draw targets, four models' top-six sets, hit counts, complete-set log scores, rank diagnostics and summary.
- `data/neural/historical_replay_2025.csv`: one row per draw per model, suitable for inspection in Excel or PowerShell.

Quick PowerShell view:

```powershell
Import-Csv .\data\neural\historical_replay_2025.csv |
    Where-Object model -eq "mlp" |
    Select-Object -First 10 date,draw_id,actual_main_six,diagnostic_top_six,matched_main_numbers,log_score_lift_vs_iid_nats |
    Format-Table -AutoSize
```

This command lists *historical* diagnostic combinations only, not future-number recommendations.

### Consistency check against Phase 5

For the same original 2,183-draw archive and unchanged fitting settings, 2025 log-score lifts should be near the previously computed values:

| Candidate | Prior Phase 5 mean 2025 log-score lift (nats/draw) |
|---|---:|
| IID | 0 |
| Static | -0.0000623 |
| Lag | -0.0001084 |
| MLP | +0.0038034 |

Small floating-point differences may arise from PyTorch versus the exact float64 NumPy reference, but substantial disagreement would mean a protocol or implementation inconsistency and should be investigated **before** interpreting the new hit counts.

The MLP's positive 2025 result did **not** persist across 2023, 2024, and 2026 in Phase 5; it is not evidence of dependable lottery predictability. No model should be promoted based on this replay alone.

## Scope and gates

1. Confirm all 95 unit tests pass on the target Windows machine.
2. Run the 2025 replay against the original official archive, preserve both JSON and CSV.
3. Verify 2025 count, previous Phase 5 log-score parity, causal date boundaries and observed full-six match histogram.
4. Evaluate hit counts, model rank diagnostics and score stability relative to fair IID, with necessary uncertainty and multiple-testing caution.
5. Keep Forecast #001 `d4a131ad6bc503bc7360eb680bab899318a84234` unchanged. The first genuinely future result can only be scored after official publication and external timing checks.

All new code comments, docstrings and console output are English-only. Analysis and discussion with the user remain in Traditional Chinese.


## Primary decision metric: exact complete six-number prediction

The user requires **one whole six-number set exactly matching the drawn main six**. All 0–5 overlaps count as **FAIL**. In version 0.1.1, the CLI prints the four models' **exact 6/6 success counts and rates first**, with `--show-model mlp --show-draws` showing each historic full-six guess, actual result, and the literal PASS/FAIL result. Partial overlap and log-score outputs remain secondary diagnostics, never substitutes for exact complete-set prediction success.

Existing `historical_replay_2025.json` already contains `summary_by_model[model].exact_six_matches` and each date's `models[model].exact_six_match`, so its old results can be summarized without retraining. The new reporting format requires one additional test, bringing the expected suite to 96.
