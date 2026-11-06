# Lotto649 Neural Learning & Predictive Control — Architecture v0.2

Status: **Phase 0 verified by user's Windows machine (8/8); Phase 1 PyTorch decoder implemented and locally verified (8/8). Neural models not yet trained.**

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
