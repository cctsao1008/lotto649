# Lotto649 Architecture v0.2 — Phase 4 Temporal Dependence Control

Status: **implementation committed; Windows tests and experimental results pending**.

## Phase 3 result gate (reported by user's home PC)

All **29/29 unit tests passed**. Smoke with two seeds and 120 synthetic draws completed. Formal Phase 3 fixed settings: 10 seeds (649..658), 2,183 draws per seed, chronological 60/20/20, 40-epoch cap.

| Synthetic truth | Static mean lift | Small MLP mean lift | MLP minus Static |
|---|---:|---:|---:|
| Fair IID | -0.001025 | -0.003256 | -0.002231 |
| Fixed bias (strength=2.0) | +2.495460 | +2.490813 | -0.004647 |

Units: mean complete-set test log-score lift in nats per draw versus exactly uniform fair 6/49.

**Interpretation:** IID shows no reproducible positive advantage; a large artificial fixed bias is learned by both models. Static and MLP are nearly tied on fixed bias, so temporal learning is **not established**. Ten seed-level bootstrap CIs are descriptive pilots, not formal proof of prediction. The `best_of_two_test_diagnostic` is test-set cherry picking and is not used for model selection.

## Why Phase 4 is needed

Static fixed bias has no temporal information. To test whether the MLP learns time structure, inject a true, explicitly known Markov dependency whose conditional distribution depends *only* on a previously observable draw.

The process is called `lag_echo`:

`P(S_t | S_(t-1)) = exp(beta * overlap(S_t,S_(t-1))) / Z(beta)`

where `overlap` is the number of common elements of two complete six-number sets. Its normalized partition function does not depend on which particular six numbers occurred before:

`Z(beta) = sum(j=0..6) C(6,j) C(43,6-j) exp(beta*j)`

At `beta=0`, this is fair IID. At `beta>0`, subsets overlapping the previous draw are favored. The synthetic generator samples the exact number of overlapping numbers first, then chooses them uniformly from the previous draw and its complement.

This is **not a claim** that real lottery drawings echo recent results. The sample generator artificially makes them do so.

## Benchmarks

Compare under identical chronological train/validation/test partitions:

1. **IID**: exact fair 6/49, lift 0.
2. **Static**: existing 49-parameter non-temporal fitted model.
3. **MLP**: existing 5-draw window, 16 hidden neurons, Phase 2 trainer; no modifications to the neural architecture.
4. **Lag Model**: a 1-parameter, mathematically correct reference model that estimates beta only from train-period overlap statistics via maximum likelihood. The generator's true beta is NEVER passed into the model.

The Lag Model is intentionally structurally aligned with the artificial generator, so an MLP *need not* beat it. Our goal is to measure **whether MLP offers any generalizable temporal value beyond Static**, while an explicit model gives an identifiable reference and reveals underfitting.

At time t, one-step prediction may use the already-observed previous draw S_(t-1) even if it belongs to the test period; test targets and trained model parameters remain unknown at prediction time.

## Safeguards

- Independent seed range **10000..10009** by default. Keep network architecture and hyperparameters fixed.
- Fresh `.npz` temporary file includes **only** the `numbers` array; no oracle probabilities or phase labels.
- The Markov reference estimates beta from chronological *training* draws only.
- All test lift results, including failures, are preserved in a versioned JSON report.
- Across-seed bootstrap intervals are descriptive, not adjusted for exploratory research choices.
- This research still concerns **one-step complete-set probability forecasting**, NOT three-month first-hit probabilities.
- No `lotto_select.py`, betting, or real draw predictions.

## Windows PowerShell commands

```powershell
cd D:\my-github\lotto649
git pull origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Short smoke test:
.\.venv\Scripts\python.exe .\lotto_temporal.py --seeds 2 --draws 120 --epochs 2 --report data/neural/phase4_smoke.json

# Phase 4 full experiment:
.\.venv\Scripts\python.exe .\lotto_temporal.py --seeds 10 --draws 2183 --report data/neural/phase4_temporal.json
```

Review each control separately, especially `mlp_minus_static` on `lag_echo`. If the MLP learns some temporal dependence but falls short of the explicit lag model, consider a smaller neural residual conditioned on recent overlap before increasing hidden size or adding GRU/TCN. If the MLP fails even on this strongly controlled synthetic dependency, diagnose input representation and inductive bias rather than attempting real lottery inference.
