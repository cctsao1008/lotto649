# Image Learning v0.2 — collapse diagnosis and time-order ablation

## Purpose

An earlier 437-draw retrospective CNN+GRU run predicted only five distinct
binary shapes. Two shapes accounted for 431 of the 437 forecasts. The question
is whether the 49 continuous logits themselves vary, whether top-six decoding
amplifies similar logits into repeated output, and whether image time order is
predictive rather than a model-complexity artifact.

This study is diagnostic, not a search for a winning lottery configuration.
The previously examined 2023–2026 records are **not blind prospective data**.

## Controls and interpretation

- **Ordered CNN+GRU:** exactly the original v0.1 loss and model training path.
- **Permuted-history CNN+GRU:** same previous-image multiset, separately
  permuted for each target, with absolute-index-seeded permutations consistent
  across fitting and prediction. It destroys within-window chronology but does
  not shuffle targets, cross data boundaries, or access target/future images.
  A model may still learn position-invariant information about the previous
  five images. This is an ablation, not an exact permutation significance test.
- **Constant trainable template:** a single learned 49-logit field reused on
  every date. Same image-shape loss and validation-loss early stopping. It has
  fewer free parameters, so model capacity differs by design. It asks whether
  past-image input is needed to match the trained model's output behavior.
- **Fixed-weights inference shuffle:** take the *ordered fitted model*, keep
  all weights unchanged, permute each target's prior frames, and check how
  much its continuous logits and decoded masks change. Sensitivity is not
  evidence of true predictive information.

All model arms use only the first 60% of images for optimization, next 20%
for early stopping (validation shape loss), and last 20% for retrospective
comparison. Identical six-pixel IID masks are reused across arms at each
validation or test date. The primary comparison is mean chance-adjusted
geometric image similarity. Validation adjusted scores are **diagnostic**;
model selection remains validation shape loss, not test outcome.

## Output guide

- `training`: separately selected epochs, validation shape-loss curves,
  exact chronological boundary indices.
- `validation`: per-arm validation adjusted similarity, fixed before the
  retrospective evaluation segment is scored.
- `summary`: per-arm retrospective adjusted similarity, raw similarity,
  unique six-pixel images, top-pattern share and per-draw paired differences.
- `logit_diagnostics`: mean/max across-time pixel logit std, 6th-to-7th
  margin, mean sigmoid mass, always-selected numbers and top-ten masks.
- `latent_diagnostics`: variation in last input-frame CNN embeddings and
  final GRU states; not a proof of learned predictability.
- `fixed_weights_history_order_control`: sensitivity to shuffling only the
  previously observed image order using the same trained network.
- `paired_intervals`: descriptive 95% circular block bootstrap intervals on
  paired adjusted score differences; not blind p-values.
- `rows`: actual/predicted six-pixel masks, image similarity and conditional
  IID baseline for every retrospective draw.

## Run

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m lotto7x7 diagnose --year all --epochs 45 --baseline 32
.\.venv\Scripts\python.exe -m lotto7x7 gallery --report results/image_diagnostics_v02.json --out results/image_diagnostics_v02.html
Start-Process .\results\image_diagnostics_v02.html
```

The canonical draw source remains `data/bin/lotto649_images_v2.bin`.
No source CSV, BIN, or older model checkpoint is modified.

### Decision checklist

1. **Low logit variance and low diversity:** continuous model output is nearly
   constant. The issue is not exclusively the Top-6 conversion.
2. **Nontrivial logit variance but near-zero Top-6 changes:** Top-6 discretization
   is amplifying a similar set of highest-ranking pixels into fixed masks.
3. **Ordered and shuffled perform alike, and both resemble the template:** no
   evidence from this diagnostic that sequence order helps beyond a template.
4. **Ordered improves over controls on validation, not only test:** a more
   useful hypothesis, still requiring future strictly prospective validation.

The 7×7 layout is an arbitrary raster mapping of lottery numbers; CNN spatial
adjacency does not imply an underlying physical dependence in the lottery.
