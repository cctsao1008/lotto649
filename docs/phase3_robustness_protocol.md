# Lotto649 Architecture v0.2 — Phase 3 Robustness Protocol

Status: code committed; **multi-seed results require execution** on the user's home PC.

## What we learned from Phase 2 (single seed 649, 2,183 draws)

| Synthetic source | IID lift (nats/draw) | Static | MLP |
| --- | ---: | ---: | ---: |
| Fair IID | 0 | +0.002414 | +0.003624 |
| Known fixed bias (strength 2.0) | 0 | +2.566448 | +2.570105 |

Positive test lifts of a few thousandths on *one* fair IID dataset can be pure sampling noise. In the fixed-bias control, both trained models detect a deliberately introduced distributional bias. MLP minus Static is only about +0.003657 nats/draw there, so this does **not** establish sequence-specific neural learning. Positive log-score lift is not a winning-ticket advantage or a change in real-draw lottery probability.

## Research hypothesis and controls

- **H0 / negative control:** independent fair uniform 6/49. The true expected out-of-sample log-score lift relative to IID cannot be positive for any other complete-set probability distribution.
- **Positive control:** time-invariant artificial unequal number weights. Since the simulator follows the same fixed-cardinality family used by the models, it is deliberately an *easy, well-specified* learning check, not evidence of real-world predictability.
- Pre-specify model candidates: `static` and `mlp`; same fixed architecture and training hyperparameters as Phase 2.
- Re-run full train/validation/test pipeline for independently generated seeds. **Never discard negative or failed-scoring runs.**
- Select epoch using validation only. Test outcomes must not influence epoch selection or hyperparameter tuning.

## Algorithm and data safeguards

`lotto_robustness.py` uses `lotto_synthetic.generate`, and writes a temporary file containing **only** observed `numbers`. Synthetic oracle `phases` and `logits` are discarded before calling `lotto_neural.evaluate`. The existing Phase 2 chronological 60/20/20 split and one-step complete-set log score are preserved. No true future target is included in each model's historical input.

Reports contain per-seed test lifts, best epochs, SHA-256 of the generated draw file, seed and split boundaries. The Monte Carlo samples are **entire independently seeded experiments**, not individual overlapping test draws.

The `best_of_two_test_diagnostic` column is the maximum test lift for Static/MLP on a seed. It is **intentionally optimistic and never eligible as an unbiased deployable score**: it quantifies why selecting a winner after examining the test set is invalid.

Summary: mean, median, sample SD, fraction with test lift greater than zero, min/max, and descriptive percentile bootstrap 95% interval for the **across-seed mean** (fixed bootstrap RNG). Ten independent replicates are a diagnostic pilot, not a definitive confidence claim, and a single positive-lift seed does not reject IID. No real-draw model comparison or multiple-model search-corrected p-value has yet been performed.

## Windows execution

```powershell
cd D:\my-github\lotto649
git pull origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Start with a short integration check of the real Phase 2 trainer
.\.venv\Scripts\python.exe .\lotto_robustness.py --seeds 2 --draws 120 --epochs 2 --report data/neural/phase3_smoke.json

# Reproducible full-length multi-seed IID and fixed-bias controls
.\.venv\Scripts\python.exe .\lotto_robustness.py --seeds 10 --draws 2183 --report data/neural/phase3_robustness.json
```

For additional replication later, lock all model settings first and use distinct seed ranges. Do not repeatedly change configurations in response to test outcomes and then describe the same test data as unseen.

## Phase 3 review gates

1. All previous and new unit tests pass. The short integration check produces finite test lifts with two models in both controls.
2. Fair IID: summarize the full distribution of lifts, including negative outcomes; **do not require each individual lift to be <= 0**.
3. Known fixed bias: both simple and neural models should typically improve on IID under sufficiently long training; compare them against each other. Do not expect an MLP to outperform the correctly specified static model.
4. Review all replicates, variance, and the selection-optimism diagnostic before considering more neural architecture.
5. Only later add time-dependent synthetic positive controls and independently validated TCN/GRU; do not infer their necessity from this static-bias experiment.

**Out of scope:** live draw forecasting, 1/3/12-month survival estimates, `lotto_select.py`, betting advice. A model of a known synthetic bias has no demonstrated predictive advantage on fair lottery draws.
