# Lotto649 Neural Learning & Predictive Control — Architecture v0.2

Status: **Phase 0/1/2 tests confirmed; Phase 3 passed 29 tests and completed 10-seed robustness on home PC; Phase 4 synthetic temporal-learning benchmark implemented, awaiting Windows tests/results. No real-draw prediction advantage established.**

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
