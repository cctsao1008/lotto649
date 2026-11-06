# Phase 7 — Historical Replay Result, 2023–2026

**Record type:** Operator-supplied Windows terminal logs, analyzed without direct access to the generated local JSON/CSV artifacts. This is exploratory historical reuse, not pristine prospectively blinded evidence.

## Protocol and observations

Command: `lotto_image_experiment.py --source data/normalized/lotto649_base.npy --year YEAR` on the Phase 7 experimental branch, using the default fixed Phase 5 train/validation boundaries, 49-logit six-subset decoder, and complete six-of-six exact-match criterion.

| Year | Historical test draws | Validation-selected model |
|---|---:|---|
| 2023 (portion of year available after test start) | 100 | cnn_permuted |
| 2024 | 118 | cnn_permuted |
| 2025 | 118 | cnn_permuted |
| 2026 (through archive cutoff) | 101 | cnn_permuted |
| **Total** | **437** | — |

**Primary outcome:** **0 complete 6/6 successes in 437 draws for every one of the eight models.** Six-of-six is the only success; 0–5 matches are failures. One model's IID single-set probability is exactly `1 / C(49,6) = 1 / 13,983,816`, so zero hits in 437 trials has probability about **99.996875%** under independent fair draws. This demonstrates no statistically discriminating evidence of full-set predictive performance. The models' successes are *correlated*, since they are scored on the same 437 real outcomes; do not treat the eight model streams as independent experiments.

## Per-year mean full-set log probability lift relative to IID

All values in nats per historical target draw. Aggregates below are **weighted by reported draw counts**, using the displayed six-decimal year means. Rounding at the last decimal is expected.

| Model | 2023 n=100 | 2024 n=118 | 2025 n=118 | 2026 n=101 | Weighted 437-draw mean |
|---|---:|---:|---:|---:|---:|
| iid | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| flat_mlp | -0.009795 | -0.012165 | +0.003803 | -0.009161 | **-0.006617** |
| cnn_sequential | -0.003838 | -0.005369 | -0.000770 | -0.005914 | **-0.003903** |
| cnn_snake | -0.001239 | -0.008629 | -0.003128 | -0.003496 | **-0.004266** |
| cnn_permuted | -0.004252 | -0.005643 | -0.000695 | -0.005124 | **-0.003869** |
| cnn_dct | -0.002426 | -0.004976 | +0.003865 | -0.003627 | **-0.001693** |
| cnn_fft | -0.005136 | -0.010877 | -0.002499 | -0.006528 | **-0.006296** |
| cnn_temporal | -0.017184 | -0.008836 | +0.003329 | -0.006627 | **-0.006951** |

The DCT CNN is the least-negative of these retrospective challengers but does **not** outperform IID on the combined held-out segment. The validation-selected permuted CNN also loses to IID in all four reported years. A positive 2025 lift for DCT, MLP, and temporal CNN does not generalize across the other years.

## Interpretation / decision gate

- **Software capability:** The supplied logs show that all eight methods completed historical replay and produced exact-6/6 and proper-log-score diagnostics. The logs do **not** establish that the full 118-test Phase 7 Windows suite passed; that validation remains to be confirmed separately.
- **Full-six predictive capability:** No positive full-set outcome is observed. Because of the enormous combination space, 437 targets cannot determine jackpot prediction power by itself.
- **Probability quality:** On this previously inspected, retrospectively evaluated segment, all non-IID models have negative average log-score lift. No image representation has demonstrated stable distributional advantage over fair IID.
- **Selection hygiene:** Do not promote `cnn_dct`, search for a lucky mapping seed, or retune on the 2023–2026 test outcomes. The 2025 advantage is post-hoc and not evidence of future performance.
- **Maintain original baseline:** Leave prospective Forecast #001 sealed at commit `d4a131ad6bc503bc7360eb680bab899318a84234` with original IID default and unchanged payload SHA-256 `99eff658789131708533638014c1e7c646b098b4d473a3099dc1d62ac359c3d6`.
- **Useful future checks:** Full 118-test Windows run; repeated synthetic IID negative controls and synthetic known-pattern positive controls; audit that all representations remain exactly reversible and that target t is absent from its features. Any assertion of real-draw prediction benefit must await independently time-attested prospective samples.

### Evidence and limitations

The above year-level numbers are transcribed from terminal logs posted in conversation. Raw local files `data/neural/phase7_YEAR.json` and `data/neural/phase7_YEAR.csv` were not downloaded here and have not been independently verified. The 437 draws are part of the historical archive that had been explored in earlier research; this is **exploratory replay**, not a fresh blind test. There is no claim of a wagering edge or increased full-set hit probability.
