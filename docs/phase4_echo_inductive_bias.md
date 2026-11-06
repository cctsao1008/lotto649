# Lotto649 v0.2 — Phase 4.1 Neural Inductive Bias Review

Status: **implementation committed; host Windows validation pending**.

## Evidence from Phase 4 (user's Windows execution)

- Unit tests: **37/37 pass**.
- Multi-seed: 10 independent synthetic datasets per control, each containing 2,183 draws.
- Fair IID: Static mean test lift -0.000798, Small MLP -0.003108, Lag Model -0.000222 nats/draw.
- Lag Echo (synthetic beta=2): Static +0.000130, Small MLP +0.177369, Lag Model +1.894964 nats/draw.
- `MLP - Static` on Lag Echo is +0.177238 mean nats; positive in all 10 seeds.
- Lag Model estimated beta from training history at approximately 1.905 to 2.039, consistent with synthetic truth beta=2.
- Small MLP captured approximately **9.36%** of the Lag Model's positive log-score lift. This is an information-gain ratio, NOT a hit-rate or jackpot-probability multiplier.

This indicates that the current MLP can detect synthetic temporal dependence, but its generic hidden-layer mapping uses the relevant signal inefficiently compared to a one-parameter, structurally correct benchmark. A bigger network is not automatically the right next step.

## Phase 4.1 hypothesis

**Inductive bias beats parameter count when the structural assumption is correct.**

Given a 49-bit encoding of the immediately previous draw `b_(t-1)`, the synthetic generator satisfies:

`P(S_t|S_(t-1)) ∝ exp(beta * overlap(S_t,S_(t-1)))`.

Under the exact fixed-cardinality decoder, the logits

`l_t = beta * b_(t-1)`

recover this entire conditional distribution *exactly*. Thus, a learnable one-scalar echo gain is a sufficient benchmark on this synthetic family.

Investigate three forms:

1. **EchoOnlyModel**: `logits = alpha * last_draw_binary`; one trainable scalar alpha. The oracle beta is NOT provided.
2. **EchoResidualMLP**: `logits = alpha * last_draw_binary + SmallMLP(history)`. It may improve representation but risks overfitting.
3. Previously fixed **Small MLP and Static** plus the analytical train-only fitted Lag Model for controls.

This is deliberately a favorable synthetic mechanism for an echo-aware inductive bias. The experiment checks engineering assumptions; it cannot show that real Lotto649 behaves similarly.

## Experimental protocol

- Fresh independent seed range 20000...20009 to avoid reusing the Phase 4 synthetic outcomes for adaptive selection.
- Each seed is evaluated for **IID and Lag Echo**.
- Chronological 60/20/20 train/validation/test, identical to Phase 2/3/4.
- Inputs to training include **numbers only**; oracle beta and generated states remain unavailable.
- Training epochs are selected exclusively on validation; test outcomes only produce scores.
- Fixed hyperparameters (no test-driven tuning): window=5, hidden=16, 40 epochs max, patience=6, batch=128, lr=.005, weight decay=.02.
- Report per-seed holdout test lifts and model-minus-model deltas. Percentile bootstrap intervals across independent seeds are descriptive.
- Comparing alternatives on the same tests should not be mistaken for a confirmed deployment choice.
- Success criterion for this *synthetic* positive control: EchoOnly meaningfully approaches Lag Model, EchoResidual is evaluated for incremental gain or overfitting, and on fair IID neither exhibits a repeatable positive advantage.

## Windows commands

```powershell
cd D:\my-github\lotto649
git pull origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Small integration smoke
.\.venv\Scripts\python.exe .\lotto_echo.py --seeds 1 --draws 120 --epochs 2 --report data/neural/phase4_echo_smoke.json

# Ten-seed independent full-scale validation
.\.venv\Scripts\python.exe .\lotto_echo.py --seeds 10 --draws 2183 --report data/neural/phase4_echo.json
```

Generated JSON reports remain ignored under `data/neural/`. No real lottery prediction, ticket selection, 1/3/12-month hit-time claim or monetary recommendation is made. `lotto_select.py` remains last.
