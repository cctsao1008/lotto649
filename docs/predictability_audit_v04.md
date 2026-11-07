# Lotto7x7 v0.4 — Predictability Audit

## Purpose

Determine whether preceding 7×7 lottery-image context carries useful
out-of-sample information **beyond** what is learned by ignoring order or
ignoring history. Do not infer a lottery advantage from retrospective results.
This experiment does not change the archived v0.2 or v0.3 code paths, BIN v2,
CNN/GRU dimensions, or deterministic hard Top-6 decoder.

## Predeclared design

- **Data:** `data/bin/lotto649_images_v2.bin`; main numbers only, each 7×7
  image has exactly six active pixels. No bonus numbers or invented dates.
- **Split:** first 60% targets for training, next 20% for validation, final
  20% for exploratory evaluation. A target is never in its own input window.
- **Loss:** `balanced_bce` exactly as implemented in v0.3 for all three arms.
  No test-driven choice among losses; weights do not change.
- **Architecture:** same v0.3 CNN+GRU for two sequence arms; an explicitly
  smaller trainable constant-logit control is a no-history reference.
- **Arms:** (1) ordered contexts, (2) identically initialized CNN+GRU with
  each historical context's frame order permuted, (3) no-history constant
  template. Shuffling preserves the target and the multiset of past frames.
- **Seeds:** consecutive integers starting from `649`; 5 by default.
  All arms use paired train/validation/test boundaries and matching seeds.
- **Checkpointing:** Epoch 0 is the **untrained** initialized model. Record
  full training and validation loss after each epoch, including the untrained
  Epoch-0 evaluation score, and restore only the
  validation-best checkpoint (which may be Epoch 0). Never select on test.
- **Scoring:** legal six-pixel Top-6 images use the existing composite image
  similarity minus a **shared, target-index-seeded, conditional IID image**
  expectation. A separate pixel Brier score uses independent probability
  outputs and the theoretical no-history `6/49` baseline.
- **Intervals:** paired circular moving-block bootstrap over evaluation
  dates **after averaging** seed-level per-date differences; draws share the
  same date and seeds are not independent future experiments. Results remain
  exploratory and cannot yield a blind-significance claim.
- **Synthetic controls:** a periodic sequence whose cycle length equals the context window (positive
  control) and independent six-of-49 masks (negative control). These check
  whether an architecture can learn supplied structure. Each window holds
  the same unordered mask multiset in different orders, so the next target
  cannot be inferred from the multiset alone; this does not show whether any
  actual lottery draw is predictable. Both controls use the same methods and loss.

## CLI

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -v
& $py -m lotto7x7 predictability --year all --seeds 5 `
    --epochs 45 --control-seeds 2 --control-epochs 45 `
    --baseline 32 --bootstrap 400 `
    --out results/predictability_audit_v04.json
```

Use `--no-controls` for an initial, quicker real-history audit and rerun the
full command for both synthetic controls. This version intentionally uses CPU
because a measured RTX 4060 CUDA run was slower than CPU for this small model.

## Interpretation and limitations

Report per seed, per method: all train/validation curves (including Epoch 0),
chosen epoch, Epoch-0 Brier and IID-adjusted image score, selected-checkpoint
Brier and IID-adjusted image score, paired training gain over initialization,
similarity, predicted unique images and top-pattern share. Aggregate all seeds.
Compare the **ordered-minus-shuffled** and **ordered-minus-constant** adjusted
scores, both with intervals. Pixel Brier below the constant `6/49` baseline
is a separate check of useful probability forecasts.

It is possible for all arms to produce plausible six-point images and for
none to outperform random expectation. More unique predictions, a lower
training loss, GPU support, or synthetic periodic success are **not** proof
of real-world predictive skill. The 2023–2026 historical holdout has been
previously inspected; a genuinely fresh prospective protocol is needed for
claims about forecasting future drawings.

**Out of scope for v0.4:** mapping-permutation experiments, nested temporal
cross-validation, hyperparameter search, architecture scaling and Transformer
comparisons. Do not re-label prior inspected draws as an untouched test set.
