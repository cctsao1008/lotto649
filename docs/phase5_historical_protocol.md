# Lotto649 Architecture v0.2 — Phase 5 Historical One-Step Diagnostic

Status: **implementation committed; historical Windows execution pending**.

## Why the next study is a historical diagnostic, not ticket selection

The synthetic experiments verified mathematical normalization, learning fixed bias, and learning an artificial temporal Markov signal. They have **not** detected such a signal in the actual Lotto649 history.

A proper historical study must distinguish four things:
1. Correct out-of-sample probability scoring on **complete six-number sets**.
2. Validation-based model selection without looking at the designated test period.
3. The model's score relative to the exact fair IID baseline.
4. The fact that this historic archive has been inspected in earlier research, so its chronological holdout is **not** an untouched prospective blind test.

This stage does not estimate 1/3/12-month cumulative hit probabilities or select two tickets; `lotto_select.py` is still deferred.

## Data adapter and scope

`lotto_historical.py` reads `data/normalized/lotto649_base.npy` directly. It requires the repository's structured `date`, `draw_id`, `numbers` and `special` fields; sorts six main numbers within each draw, verifies legal distinct numbers, verifies that the special number is legal but excluded, and requires strictly increasing dates with unique draw IDs.

It records the source SHA-256, first and last draw metadata, chronological split boundaries and their corresponding dates.

**No new data scraping, fetch, random filtering, bonus-activity mixing, or source file modification is performed.**

## Fixed comparison

- IID: exactly uniform conditional probability `1/C(49,6)`.
- Static: Phase 2 49-logit regularized model.
- Small MLP: Phase 2 window=5, hidden=16, validated early stopping.
- Lag: one-scalar beta estimated only from TRAIN draw overlaps via exact maximum likelihood. This is a *negative or positive* lag effect depending on training data; no synthetic oracle is supplied.

Training and validation are the first 60% and next 20% of the chronological record, with the final 20% used only as the evaluation period. The already-observed previous test draw is allowed as **one-step causal context**. All fitted model weights and coefficients are frozen throughout test.

**Model choice is a deterministic validation-only rule:** largest mean validation complete-six-set log score, ties favor IID, then Static, then Lag, then MLP. Test scores of *all* candidates are reported for auditing; a post-hoc test-set winner must not replace the validation-selected model.

For each model, report mean test log score and IID-relative lift, a descriptive moving-block bootstrap interval of the paired per-draw lift, and per-calendar-year lift to expose possible regime changes. The default bootstrap block is 13 draws. This interval is not corrected for the many earlier exploratory analyses or selection, and should **not** be interpreted as proof of profitable prediction.

### Interpretation boundaries

- **If IID wins on validation**, there is no reason to replace it with a nonuniform forecast based on the current models.
- **If a learned model wins on validation but loses on test**, the hypothesis did not generalize in this retrospective split. Avoid repeatedly tuning to that same test portion.
- **If a learned model also improves on test**, record the magnitude and uncertainty; prior research exposure means this remains an exploratory association requiring prospective precommitted forecasting.
- A positive mean Log Score lift does **not** establish that either of two fixed full-six selections hits earlier.
- Fair independent Lotto649 has no exploitable conditional pattern under the IID hypothesis.

## Phase 4.2 closeout

The user's **10-seed official Phase 4.2 synthetic report** showed on artificial Lag Echo:
- Legacy AdamW Echo: +1.83038 mean nats/draw.
- Train-only MLE Echo: +1.91900 mean nats/draw.
- Validation-gated staged MLP: +1.91760 mean nats/draw.
- On IID: +0.00056, +0.00071, +0.00038, respectively, without clear reliable advantage.

The residual was selected on validation in only 30% of Lag Echo datasets and 20% of IID datasets. It did not deliver repeatable holdout gains. For a rejected residual, Staged and MLE scores describe exactly the same model; **Phase 4.2 now directly reuses the MLE score** to prevent float32/float64 diagnostic noise that previously appeared as `-0.00000`.

These results identify a training convergence issue for the 1-scalar AdamW model under the artificial mechanism. They do not establish an advantage for real lottery forecasts.

## Windows PowerShell

```powershell
cd D:\my-github\lotto649
git pull origin main

# 43 previous tests + 5 Phase 4.2 tests + 1 precision fix test + 6 Phase 5 tests = 55 tests.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Phase 5 read-only real-history one-step diagnostic
.\.venv\Scripts\python.exe .\lotto_historical.py --report data/neural/phase5_historical.json

# Optional: inspect the machine-readable report
Get-Content .\data\neural\phase5_historical.json
```

All runtime source-code comments, docstrings and messages added here are **English-only**. Discussion with the user remains in Traditional Chinese. Do not change the historical data or use these diagnostics to generate bets.

## Next gates

1. Confirm the historical loader accepts the actual structured archive and verify the printed date boundaries.
2. Confirm all unit tests pass, especially the validation-only model selection and the exact OFF residual zero difference.
3. Examine validation-selected model, full candidate test scores, and year-by-year stability, **without changing the hyperparameters in response to held-out test results**.
4. Only consider prospective forecast sealing and 1/3/12-month horizon calibration after the distributional models pass the appropriate external validity checks.
