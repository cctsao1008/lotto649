# Lotto649 Neural Learning & Predictive Control — Architecture v0.2

Status: **Phase 6.1 passed 73/73 Windows tests; API bridge verified 2,183+2=2,185 records. First forecast for 2026-11-10 sealed in commit d4a131a and corroborated by server-created GitHub Issue #1 (2026-11-06T17:51:07Z); no future target results available. Phase 6.2 read-only readiness audit committed, Windows validation pending. IID remains default; no scores or bets produced.**

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


## Phase 6.2: pre-scoring readiness audit

A November 7 file named for November 10 contained only 2,185 draws through November 6; the original November 7 shadow source and this misleadingly named file shared exact SHA-256 `2e21fe66094ef256dc76fc77d5c324195071bc70d3642b68a675d35ee9e480c1`. Before official outcome publication, `lotto_shadow.py score` correctly refuses to score.

**`lotto_shadow_status.py`** is a **read-only inspection** command. It verifies sealed payload bytes against the original commit, the full historical draw prefix, candidate count and dates, the source SHA-256 and manifest provenance, and every staged record's date/period/main-six/special against the saved raw API snapshot. It reports `WAITING_FOR_TARGET_DRAW` for an unextended candidate or a distinct failure if missing dates, source conflicts, or deadline violations arise. A `READY_FOR_SCORING_PENDING_OFFICIAL_RESULT_CHECK` output is not independent confirmation that official draw results have been published. See [Phase 6.2 audit protocol](phase6_readiness_audit.md) and `tests/test_lotto_shadow_status.py`.

The immutable forecast remains [`d4a131a`](https://github.com/cctsao1008/lotto649/commit/d4a131ad6bc503bc7360eb680bab899318a84234), with identical payload SHA-256 in [GitHub Issue #1](https://github.com/cctsao1008/lotto649/issues/1) created at 2026-11-06T17:51:07Z. This third-party created time precedes the research cutoff, but issue edits and absence of a cryptographically protected timestamp still warrant cautious claims about immutability. IID remains the baseline. `lotto_select.py` remains deferred.

## Phase 5.1: retrospective full-six diagnostic replay (implemented; Windows validation pending)

The user requested a directly observable historical test of **what the models would have selected before each old drawing**, beyond the previous Phase 5 complete-six-set log-score results. `lotto_replay.py` reconstructs the identical frozen Phase 5 train/validation cutoffs **1309 / 1746** (rather than shifting them when an archive grows), uses pre-target historical inputs only, and compares retrospective top-six predictions to the actual main six in **2025** by default. It produces a detailed JSON and per-draw CSV containing individual predicted six, 0–6 matches, full-set log-score lift versus IID, and realized-outcome rank diagnostics. Other years within the original held-out period can be requested without refitting on their outcomes.

Under IID, all complete six-number sets tie; a reproducible per-period seed chooses a diagnostic subset for comparisons. The expected matched main numbers per draw is 36/49 and exact full-six match probability 1/13,983,816. Lag outcome midranks are combinatorially exact; Static/MLP use a uniformly sampled, explicitly uncertainty-marked combination pool (50,000 draws by default). These historical diagnostics do not constitute new prospective blindness or ticket recommendations. See [historical replay protocol](historical_replay_protocol.md), `lotto_replay.py`, and `tests/test_lotto_replay.py`.

**Research constraint:** Phase 5 already showed nonpersistent performance across 2023–2026, so a favorable 2025 match count must not be used to retune models or override the IID default. Prospective Forecast #001 (original commit `d4a131a`) is untouched. After the 83 previously passed Windows tests, twelve new replay tests are pending, bringing the expected suite to **95**.

## Phase 7: reversible image representation experiment (isolated branch)

**Branch:** `feature/phase7-image-representation`. [Protocol and verification commands](phase7_image_representations.md).

Each regular-main-six lottery result is represented as a strict six-pixel 7×7 binary mask. The new `lotto_image_repr.py` supports sequential/snake/permuted bijections, complete orthonormal 2D FFT (real and imaginary components) and DCT, and previous-only temporal windows. Encoding and decoding are reversible; the geometry is deliberately artificial and does **not** represent a physical proximity mechanism in the draw machine.

`lotto_image_experiment.py` trains and evaluates seven feature-learning model variants (flat original 49-bit MLP; sequential, snake, permuted, DCT, FFT spatial CNNs; and temporal 3D CNN), plus exact IID. All use fixed Phase 5 **train/validation boundaries 1309/1746**, exact 6-of-49 likelihood, and validation-only model selection; 2025 is the already explored retrospective diagnostic year. **Whole-six-set 6/6** match is the only PASS, with partial matches considered FAIL. The exact full-set log-score lift remains an explicitly secondary statistical diagnostic, not a surrogate jackpot hit.

`lotto_image_analysis.py` compares artificial-grid geometry to identically encoded IID control simulations. Added **22 Phase 7 tests** include reversible transforms, causal feature isolation, synthetic constant-six perfect-match recovery, reproducibility, and CSV exact-outcome verification. Previous 96 Windows tests passed, so the new expected suite is **118**, not yet executed on the user's Windows machine. Phase 6 Forecast #001 remains unchanged at original sealed Git commit `d4a131a`. No prospective advantage is claimed.
