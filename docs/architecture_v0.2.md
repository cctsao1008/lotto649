# Lotto649 Neural Learning & Predictive Control — Architecture v0.2

Status: **Phase 6 seal/verify/score has passed 64/64 tests on the user's Windows host. Phase 6.1 adds a shadow-only bridge for recent official-site API draws; nine new unit tests and live API contract validation are pending. No real future forecast has been sealed or externally time-attested. IID remains the default.**

## Objective and non-goals

After draw t closes, estimate the probability that one of **two distinct six-number subsets** will be drawn at least once within **three calendar months**. One and twelve months are diagnostic horizons, not separate optimization targets. All forecasts must be sealed before evaluation draws occur.

The model cannot influence or control a lottery draw. Under fair IID 6/49, all candidate subsets have identical hitting-time distributions; no transformation or neural model can shorten them.

`lotto_select.py` remains out of scope until probability modeling and validation have passed review.

## Ground truth and constraints

- Official source: `data/normalized/lotto649_base.npy`; bonus activities are excluded from independent-draw modeling.
- Exactly six distinct numbers from 1 through 49 per draw; order within a subset has no probability meaning.
- Complete subset space: C(49,6) = 13,983,816.
- No train/validation/test leakage: use chronological, pre-registered forecast origins and never feed true intermediate future draws into a multi-step prediction.
- Historical data has already been examined during research. Retrospective results are exploratory; prospective frozen forecasts are needed for strong evidence.

## Phase 0: mathematical probability and synthetic controls

**`lotto_probability.py`**: auditable, NumPy-based, fixed-cardinality subset distribution.

For logits l_i and size k:

`P(S) = exp(sum(i in S, l_i)) / Z`

where `Z = sum(|T|=k, exp(sum(j in T, l_j)))`.

`log(Z)` uses log-space dynamic programming in O(n*k), rather than enumerating every combination. The sampler produces exactly weighted k-subsets. Marginal inclusion probabilities sum to k. All-zero logits recover fair 6/49 exactly. This is a **first-stage parametric family**, not a claim that true draws follow a product-weight mechanism.

**`lotto_synthetic.py`**: controlled experiments, reproducible by seed:

- `iid`: fair independent uniform draws (negative control).
- `fixed_bias`: weighted probabilities that do not change over time.
- `periodic_bias`: noisy weighted draws with a known phase cycle.
- `deterministic_cycle`: exact repeated subsets with a known period (positive control).
- `change_point`: change in distribution at the midpoint.

Synthetic files stay separate at `data/synthetic/*.npz` and include oracle `phases` and `logits` for evaluation ONLY. Never feed these ground-truth labels into a blind predictor.

**`tests/test_lotto_v02.py`**: exact small-space enumeration, normalization, shift invariance, sampler calibration, invalid-input rejection, reproducible generation, known-period repetition, and oracle-vs-IID diagnostic on biased synthetic samples.

## Phase 1: differentiable probability decoder

**`lotto_torch_decoder.py`** implements the same log-space O(batch*n*k) dynamic-programming distribution in PyTorch and exposes an autograd-compatible log-probability per subset. The decoder uses 1-based integer six-number sets and validates cardinality and uniqueness. Single/batched logits are supported. **`tests/test_lotto_torch_decoder.py`** compares the results to the NumPy reference, exact enumeration, analytic inclusion gradients and finite-difference autograd. Phase 1 self-check: `python lotto_torch_decoder.py` (CPU) or `python lotto_torch_decoder.py --device cuda` (CUDA enabled). Install PyTorch separately with `python -m pip install torch`. The base `requirements.txt` remains unchanged, so Phase 0 works without the optional ML dependency.

## Next gates, not implemented in Phase 0/1

1. **Differentiable decoder**: DONE (mathematical reference, parity, gradient tests; Windows verification pending).
2. **Learning models**: first static regularized logits, then a **small MLP** with past draw representations; TCN/GRU only if earlier models justify complexity.
3. **Scoring**: out-of-sample complete-subset log score relative to IID: `log P_model(S_t|history) + log(C(49,6))`. Compare full forecast paths, not training loss.
4. **Statistical control**: correct for model/search selection, multi-seed IID null simulations, no deployment from training-only wins; freeze parameters and candidates at every forecast origin.
5. **Decision horizon**: evaluate one / three / twelve calendar months. Three-month first-hit is the decision metric, but jackpot hits are too rare in 2,183 draws to measure small improvements with useful precision.

## Decision boundaries

- Better full-distribution log score **does not imply** the top two combinations hit sooner.
- A periodic feature (sum/product/modulo) **does not imply** a periodic exact six-number subset.
- A model that fails the IID baseline should be rejected or deweighted, never used to justify increasing the betting budget.
- A successful synthetic positive control proves performance under its synthetic assumptions, **not** that the actual lottery is predictable.

## Local commands

```powershell
cd D:\my-github\lotto649
python .\lotto_probability.py
python -m unittest discover -s tests -v
python .\lotto_synthetic.py --mode iid --count 2183 --seed 649
python .\lotto_synthetic.py --mode deterministic_cycle --count 2183 --period 52 --seed 649
python .\lotto_synthetic.py --mode periodic_bias --count 2183 --period 52 --strength 2.0 --seed 649
```

These modules do **not** modify existing downloader, normalizer, analyzer, or historical data files.


## Phase 2: static probability vs Small MLP (implemented; host verification pending)

**`lotto_neural.py`** provides the first neural-learning experiment. It intentionally reads **only** `numbers` from generated `.npz` files and cannot see synthetic oracle `logits` or `phases`. Two trainable models share the exact Phase 1 fixed-cardinality decoder:

- `static`: 49 trainable time-invariant logits; a simpler statistical model.
- `mlp`: causal sequence of the previous `window` six-number draws represented as flattened 49-bit vectors -> `Linear(window*49, hidden) -> Tanh -> Linear(hidden,49)`. Defaults: window=5, hidden=16. It starts at the IID distribution.

The loss is **negative exact joint six-number log-likelihood**, optimized with AdamW, gradient clipping and validation-based early stopping. Chronological draw-position splits are train 60%, validation 20%, test 20%. Every example's inputs end **before** its target draw; a test example may use previously observed test draws as context, as in operational one-step forecasting, but the frozen model cannot update its weights using them. Test scores are computed only after validation-based training finishes.

Scores are **mean test log probability per complete set**, and lift (nats/draw) vs IID is `mean_logp_model - (-log(13983816))`. The baseline `iid` has no trainable parameters. A positive single-test lift is **not** proof of advantage: Phase 3 requires many random seeds and search-corrected null comparisons.

The MLP is not guaranteed to outperform a static model on `fixed_bias` data: static bias is itself the truth. The `periodic_bias` generator has oracle periodic phases that must never be used as model inputs; the Phase 2 MLP intentionally has no phase feature. Phase 3 should evaluate explicit *past-observable* phase representations if justified, never use oracle phase labels.

Run in Windows PowerShell from the user's home PC repository:

```powershell
cd D:\my-github\lotto649
git pull origin main
.\.venv\Scripts\python.exe .\lotto_synthetic.py --mode fixed_bias --count 2183 --seed 649 --strength 2.0
.\.venv\Scripts\python.exe .\lotto_neural.py --data data/synthetic/fixed_bias_n2183_seed649.npz
.\.venv\Scripts\python.exe .\lotto_neural.py --data data/synthetic/iid_n2183_seed649.npz
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Generated experiment results are written to `data/neural/*.json`, excluded from Git. Preserve output and seed information for comparison. No model is yet used for real-money decisions or prospective 1/3/12-month forecasting. `lotto_select.py` remains last.


## Phase 3: multi-seed negative and positive controls

**`lotto_robustness.py`** reuses the Phase 2 trainer on independently seeded IID and fixed-bias synthetic datasets. It records *all* test lifts, including negative results; computes per-seed paired MLP vs Static differences; summarizes the distribution across seeds using descriptive bootstrap intervals; and reports a clearly labeled, optimistically biased best-of-two *test* diagnostic to illustrate why choosing models after seeing test outcomes is invalid. Synthetic oracle parameters are never written into the input file presented to the learner. **`tests/test_lotto_robustness.py`** checks statistical summaries, reproducibility and oracle isolation. See [Phase 3 experiment protocol](phase3_robustness_protocol.md) for full Windows commands and interpretation gates. **This is still one-step synthetic prediction, not prospective lottery outcome prediction or ticket selection.**


## Phase 4: time-dependent synthetic positive control (implemented; host validation pending)

**`lotto_temporal.py`** introduces exact Markov lag-echo sampling where `P(S_t|S_(t-1))` increases exponentially with the count of repeated numbers. Model sees previously observed draws only, never oracle parameters. On a separate seed range, compare IID, Static, Small MLP and a train-only estimated one-parameter lag model. MLP need not outperform this correctly specified lag baseline; whether it improves on time-independent Static demonstrates whether the neural architecture captures the synthetic temporal dependency. It uses the Phase 2 trainer unchanged and reports all seed-level outcomes. **`tests/test_lotto_temporal.py`** verifies exact overlap law, independence at zero coupling, causal fitting, reproducibility and oracle isolation. See [Phase 4 protocol](phase4_temporal_protocol.md). No 1/3/12-month forecasting and no final selection implementation.


## Phase 4.1: Inductive bias before increasing neural complexity

**`lotto_echo.py`** tests the hypothesis that a direct, interpretable neural skip connection from the previous observed draw to the 49 logits is more sample efficient than a generic MLP on a known synthetic lag-overlap distribution. The fully trainable `EchoOnlyModel` has one scalar; `EchoResidualMLP` adds a learned scalar echo path to the original MLP. Matched tests include IID, Static, MLP and the train-only fitted Lag Model, using a **fresh independent seed range**; results are saved in `data/neural/`. See [Phase 4.1 detailed experimental protocol](phase4_echo_inductive_bias.md). This does **not** assert an echo mechanism for the actual lottery. Neural state-space models, TCN, GRU and `lotto_select.py` remain deferred until specific validation criteria are met.


## Phase 4.2: optimizer audit before more architecture

**`lotto_echo_staged.py`** compares the original AdamW one-parameter echo training, an exact **train-only MLE** echo coefficient (with PyTorch parity check), and a **frozen-MLE echo plus a validation-gated neural residual**. At training epoch zero, the staged residual is identically zero, so the benchmark starts at the correctly estimated reference distribution. New network weights are selected only if they improve the chronological validation log score; this is not a test guarantee. Fresh seeds 30000..30009 are used for IID and Lag Echo controls. See [Phase 4.2 audit protocol](phase4_staged_residual.md). No new architecture is promoted from synthetic performance alone; TCN/GRU and `lotto_select.py` remain deferred.


## Phase 5: read-only historical distributional scoring

**`lotto_historical.py`** ingests the structured official `lotto649_base.npy` archive (date, draw ID, main six, special), validates and sorts the six main numbers, excludes special/bonus draws and records source SHA-256. It compares **IID, Static, Small MLP, and a train-only-fitted Lag model** under chronological 60/20/20 split. Model selection is fixed using **validation only**, and all candidates' untouched test-period scores are reported with descriptive moving-block bootstrap intervals and calendar-year breakdown. These are **one-step log-probability diagnostics**, not ticket advice or evidence of shorter full-set waiting time. Earlier historical exploration means the last 20% is not an uncontaminated prospective blind holdout. See [Phase 5 historical protocol](phase5_historical_protocol.md) and `tests/test_lotto_historical.py`.

The Phase 4.2 reporting fix ensures a residual rejected on validation has exactly zero staged-minus-MLE diagnostic difference. All newly added source code comments and docstrings are English-only. `lotto_select.py` remains the final stage.


## Phase 6: prospective shadow validation (infrastructure implemented, host validation pending)

**`lotto_shadow.py`** creates a once-only, content-hashed forecast envelope for one specified future draw, storing four full 49-logit conditional distributions (IID, Static, Lag, MLP), official archive SHA-256 and a canonical date/ID/six-main-number history-prefix fingerprint, Git code revision, training settings and explicit future target date / declared pre-draw deadline. The program refuses old data (>7 calendar days by default), future draws in the source, rewritten history, skipped target draws, overwriting existing seals, and attempts to score before the declared deadline. It can compare the exact seal file bytes to a specified Git commit and produces per-model complete-subset log scores after the updated official target draw becomes available.

**Trust boundary:** Local timestamps and Git commit timestamps are not independent evidence that the seal existed before the draw. A third-party timestamped publication/attestation must be independently checked; the first implementation does not auto-verify this and never reports a prospectively verified status. The existing historical archive ends on 2026-10-30 and must be refreshed before any actual future seal. The program does not guess official draw dates or times.

See [Phase 6 protocol](phase6_shadow_protocol.md), `tests/test_lotto_shadow.py`, and the explicit code/comment English-only convention. Model selection is not driven by historical test performance: IID is the default and other models are shadow comparators only. All forecasts are one-step, not a 1/3/12-month cumulative-hit estimator. `lotto_select.py` remains the final stage.


## Phase 6.1: closing the monthly-data publication gap

The official Taiwan Lottery annual-result download page says it updates on the fifth of each month through the **preceding month**. Consequently, a historical archive current to 2026-10-30 cannot automatically supply draws on 2026-11-03 and 2026-11-06 for a prospective forecast in early November. A community-documented, **unsupported internal API** at `api.taiwanlottery.com/TLCAPIWeB/Lottery/Lotto649Result` can supply recent monthly results, but its contract must be validated on the user's host.

**`lotto_live_data.py`** offers `fetch` (preserves exact monthly API JSON) and `build` (strictly merges new validated draws into **separate** `data/shadow/sources/*.npy`, creates SHA-256 provenance manifest, checks old-period conflicts and missing Tuesday/Friday draws, never overwrites the monthly official archive). Financial fields absent from the API are explicitly zero **placeholders**, not authentic sales or prize data, in shadow-only rows. The historical analyzer's official data source remains untouched. See [Phase 6.1 protocol](phase6_recent_data_bridge.md) and `tests/test_lotto_live_data.py`.

The new bridge does not prove an externally authenticated time of seal publication. No live forecast should be sealed until the latest official source, next actual drawing schedule, and independent timestamp method are verified. All added Python source comments and docstrings are English-only. `lotto_select.py` remains last.
