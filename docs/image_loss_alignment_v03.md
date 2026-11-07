# Lotto7x7 v0.3: image loss vs. six-pixel decoder consistency

## Immutable v0.2 reference

The v0.2 code and research results are a **fixed comparator**, not an editable starting point for model optimization.

- Frozen Git commit: `0102c561530522dd0abae4fd4aa0b8fccef81a9e`
- Convenience branch: `baseline/image-learning-v02`. Use the commit SHA, not the movable branch pointer, for historical identity.
- `lotto7x7/learning.py` normalized-newline SHA256: `41718499b9e8095b501099a8c009eaa903922d6d05e7a38edafd1e66d9b169e2`
- `lotto7x7/diagnostics.py` normalized-newline SHA256: `9ad0048ec7575679e13d4a425330996c4ad09c1bee6eed2218bc85c0960a3811`
- Previously observed v0.2 historical evaluation (`437` targets, settings: `--year all --epochs 45 --baseline 32 --bootstrap 400`):
  - `ordered_cnn_gru`: adjusted `-0.006472`, 5 unique hard images, unprojected sigmoid soft mass `15.0797987`.
  - `shuffled_cnn_gru`: adjusted `-0.004962`, 4 unique hard images.
  - `constant_template`: adjusted `-0.003419`, one unique hard image.
- The existing `diagnose` command, v0.2 training functions, CNN/GRU architecture, six-top-logit decoder, similarity metric, chronological 60/20/20 split, and BIN v2 dataset must remain unchanged throughout v0.3.
- `tests/test_loss_study_v03.py` asserts the original normalized source hashes and exact original-loss **value and gradient** parity.

## Hypotheses

1. `H_loss`: sevenfold positive-weighted BCE plus only a weak mass penalty can push the sigmoid soft mass well above six. Replacing the loss may improve calibration of the *soft* 49-pixel field.
2. `H_decoder`: the hard Top-6 mapping discards small continuous-output differences even if soft predictions vary. Soft-mass calibration is not guaranteed to increase hard-image diversity.
3. `H_data`: historical independent fair draws carry no usable temporal information; even a technically well-calibrated loss need not improve next-draw image resemblance over fair IID.

## Exact experiment arms

The same `ImageSequenceLearner`, `AdamW` settings, initialization seed, training targets, validation targets, test targets, and `decode_image` are used in all arms:

| Arm | Shape term | Spatial objective | Six-pixel constraint |
|---|---|---|---|
| `original_v02` | v0.2 translation/D4-aware autocorrelation, weight 0.35 | weighted BCE (`pos_weight=7`), weight 0.60 | weak quadratic soft-mass penalty, weight 0.05 |
| `balanced_bce` | same autocorrelation, weight 0.35 | unweighted BCE, weight 0.60 | same weak mass penalty, weight 0.05 |
| `projected_six_mass` | same autocorrelation on projected soft field, weight 0.35 | unweighted BCE on projected field, weight 0.60 | **exact sum-six soft projection**, with 0.05 MSE replacing the now-redundant mass penalty |

**Projection detail:** for each [7,7] logit tensor `z`, solve a sample-specific scalar bias `b` such that

`sum(sigmoid(z + b)) = 6`.

The solution uses fixed-iteration bisection plus an implicit gradient correction. It preserves the ranking of the 49 original logits: the six largest logits remain the six largest projected probabilities. The existing hard decoder is unchanged; its output has exactly six active pixels. Consequently, a mass of six is a **calibration guarantee**, not proof of six-point generation diversity or predictive skill.

**Selection caveat:** each arm uses its own validation image loss for early stopping because losses differ. Their validation-loss numbers therefore should **not** be compared directly across arms. Compare external, prespecified image similarity (conditioned on each forecast shape's expected IID resemblance) and paired uncertainty estimates instead.

## Reproduce on Windows

```powershell
cd D:\my-github\lotto649
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Compare all three losses using the same already-inspected historical segment.
.\.venv\Scripts\python.exe -m lotto7x7 loss-study `
    --year all --epochs 45 --baseline 32 --bootstrap 400

# Visual audit (same six-pixel image gallery renderer).
.\.venv\Scripts\python.exe -m lotto7x7 gallery `
    --report results/loss_study_v03.json `
    --out results/loss_study_v03.html
Start-Process .\results\loss_study_v03.html

# Independently assess ability to learn a known cycle versus a fair-IID sequence.
.\.venv\Scripts\python.exe -m lotto7x7 loss-synthetic --epochs 45
```

Files created locally under `results/` are excluded from Git; no baseline report is overwritten. The new historical report defaults to `results/loss_study_v03.json`, and synthetic controls to `results/loss_synthetic_v03.json`.

## Measurements and interpretation

- `mean_iid_adjusted_similarity`: **primary external outcome**, same fixed metric for all arms. Improvement should exceed uncertainty and reproduce across prospectively held-out future draws before any predictive claim.
- `paired_intervals`: paired moving-block (length 8) exploratory intervals for each treatment **minus original v0.2**, using matched target dates and IID-control draws. An interval including zero does not support a clear treatment advantage.
- `logit_diagnostics.method.mean_pixel_logit_std`: mean temporal variance of each raw output logit. It does not directly measure information content.
- `logit_diagnostics.method.effective_soft_mass`: sum of **the probabilities actually fed to each loss**. The original and balanced arms use regular sigmoid; projected arm uses a constrained sigmoid.
- `logit_diagnostics.method.mean_sigmoid_mass`: raw unprojected sigmoid mass for all arms; **not the projected training soft mass** for the projected arm. Do not confuse these two measurements.
- `unique_patterns`, `top_fraction`, `mean_top6_top7_margin`: monitor decoder collapse and boundary instability; increased diversity alone is not evidence of skill.
- `loss-synthetic` runs *two separate seeded studies*, one a known 4-pattern cycle and one fair IID six-point masks. A finite IID sample may appear to show accidental positive or negative adjusted resemblance. Synthetic periodic learning validates capacity only, not the physical drawing process.

## Decision gates

1. **Calibration:** projected mass must remain at six to numerical precision, and gradients must pass finite checks.
2. **Decoder:** all arms must output exactly six binary pixels, with projection preserving Top-6 ordering.
3. **Causality:** all models must use only previous frames, with training and validation boundaries fixed.
4. **Learning sanity:** compare the same model on explicit synthetic learnable periodic sequences and independent IID sequences; do not fabricate positive conclusions from the IID case.
5. **Predictive value:** do not claim forecasting superiority based on the previously explored 437 historical targets. Use timestamped, precommitted prospective experiments for any future claim.
