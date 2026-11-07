# v0.8.1 — Frozen probability replay score audit

Use **existing** `results/probability_v08_full.json`, with no model retraining or
recalculation of previously published next-draw forecasts. The audit checks
saved score integrity for every 49-number marginal forecast, recomputes
Brier/log loss and legal Top-6 hits, then compares three challenger models
against the same IID forecast on the same dated target draws.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m lotto7x7 predict-audit `
    --report results/probability_v08_full.json `
    --out results/probability_v081_audit.json
```

No repeated unit test suite is required. Only when source code changes need
focused verification, run the tests for the changed component.

- Primary metric: mean Brier skill = IID Brier minus model Brier; positive is good.
- Secondary diagnostics: binary log-loss skill, mean Top-6 hits, >=3 hit rate.
- Paired threshold inference: exact two-sided McNemar, Holm adjustment across
  the **three** challenger models. This does not correct prior exploratory work.
- Confidence intervals: a draw-level bootstrap plus an additional contiguous
  100-draw block resampling sensitivity analysis; do not mistake shared
  retraining blocks for independent replicates.
- Year summaries derive ROC year from the target's draw ID.
- A 49-number marginal probability vector sums to six; it is not a six-number
  joint distribution and does not guarantee empirical calibration.
- A threshold outcome can improve despite Brier Score degradation: describe
  the metrics separately, and prefer strictly proper scoring rules when judging
  **probability forecasting**.
- If source file is a tail smoke report, pass `--allow-partial`, which is marked
  as exploratory. The default rejects incomplete historical replays.

This audit does not re-open the source BIN and cannot independently verify
historical records. Dataset provenance and no-target-leak controls belong to
previous archived audits. The evaluations are retrospective and inspected.
