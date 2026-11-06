# Lotto649 v0.2 — Phase 4.2 Echo convergence / residual-gating audit

Status: **code committed; user Windows integration verification pending**.

## Grounded Phase 4.1 results (user report, 10 seeds, 2,183 draws each)

All **43 / 43** Phase 4.1 tests passed.

| Artificial source | Static | Plain MLP | Echo Only (AdamW) | Echo + MLP (joint fit) | Train-only Lag MLE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fair IID | -0.00121 | -0.00296 | -0.00034 | -0.00390 | -0.00065 |
| Lag Echo beta=2 | -0.00010 | +0.17013 | +1.85191 | +0.65037 | +1.94827 |

Units: mean *held-out* log-score lift relative to uniform 6/49 (nats/draw).

On Lag Echo, the original Echo Only retains approx **95.05%** of the properly specified one-parameter Lag Model's log-score gain. However, fitted echo alpha is about 1.57–1.60 (less than synthetic beta=2), while the train-only analytic MLE from the prior Phase 4 study recovered values close to 2. This strongly motivates an **optimization/convergence audit**, not an immediate increase in model capacity.

Joint Echo + MLP is about **1.2015 nats/draw WORSE** than Echo Only on Lag Echo, for all 10 seeds. Suspected causes include joint optimization interference, insufficient convergence and learned residual distortions, but **the available logs do not isolate these causes**.

Phase 4.1 used the seed range 20000–20009. Phase 4.2 uses fresh seeds **30000–30009** to avoid simply reusing the same observations. Model design remains exploratory, having been informed by earlier synthetic results.

## Phase 4.2 experiment

Use `lotto_echo_staged.py`, preserving the same exact six-number subset decoder, 60/20/20 chronological split, `window=5`, `hidden=16`, `epochs=40`, `patience=6`, `batch=128` and learning rate 0.005.

Three methods:

1. **Legacy Echo**: existing Phase 4.1 one-scalar model trained by AdamW. It is the direct optimizer-comparison baseline.
2. **Train-only MLE Echo**: estimate beta by matching the TRAINING-DATA overlap moment to its exact expectation; never read the validation or test labels when fitting beta. The fitted coefficient is mathematically identical to the `Lag Model` statistical baseline, and PyTorch log probability must agree numerically.
3. **Frozen MLE Echo + Neural Residual**: use training-only fitted beta, initialize residual MLP to exactly zero, **freeze the echo coefficient**, then optimize only residual weights. Compare all candidate residual checkpoints with **epoch 0** (the original MLE model) using VALIDATION data only. Keep residual only if mean validation log score beats the best prior checkpoint by at least `min_delta=0.0001`; otherwise revert.

Important: the validation gating mechanism cannot **guarantee** an improvement on unseen test draws. It protects model selection from test-set information, but validation noise can still select a harmful residual.

Metrics: `legacy_echo`, `mle_echo`, `staged_residual`, `mle_minus_legacy`, `staged_minus_mle`, and per-seed whether any residual was selected. Include IID and Lag Echo controls and descriptive seed-level bootstrap intervals.

## Windows PowerShell

```powershell
cd D:\my-github\lotto649
git pull origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Short smoke test of the entire actual training code
.\.venv\Scripts\python.exe .\lotto_echo_staged.py --seeds 1 --draws 120 --epochs 2 --report data/neural/phase4_staged_smoke.json

# Pre-specified 10-seed study using different synthetic data than Phase 4.1
.\.venv\Scripts\python.exe .\lotto_echo_staged.py --seeds 10 --draws 2183 --report data/neural/phase4_staged_residual.json
```

The report lives under the already-ignored `data/neural/` directory. **Expected test count: 48** (previously 43 plus five new tests). This is an anticipated count, not an assertion that Windows verification already passed.

## Decision gates

- If MLE Echo substantially outperforms legacy AdamW, the discrepancy is attributable primarily to the legacy scalar fitting procedure (40-epoch optimization, weight decay and validation selection), not to an intrinsic need for larger neural networks.
- If residual is rarely accepted, the synthetic process is already adequately explained by a single parameter. **No extra parameters are justified**.
- If residual is often accepted on validation but harms holdout tests, tighten the validation strategy/regularization; do **not** promote a test-chosen winner.
- IID controls must show no repeatable positive out-of-sample log-score advantage.
- A model cannot affect the lottery mechanism; this does not establish actual lottery predictability or shorter three-month jackpot waiting times.

**Out of scope:** Lotto649 real-data inference, 1/3/12-month cumulative-hit modeling, TCN, GRU, `lotto_select.py`, and wagering recommendations.
