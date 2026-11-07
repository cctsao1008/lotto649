# Lotto7x7 v0.8 — GPU probability research

## Goal and scope

Question: can **49 individual next-draw number marginal probabilities**, learned
from chronologically available Lotto649 images, improve proper scoring rules
beyond the fair 6/49 benchmark? This is a research tool; no expected
predictive advantage is assumed. The six numbers are drawn without replacement
and their **joint** distribution is not learned by independent Bernoulli logits.
The bonus number and money payout are excluded.

### Models

| Label | Method | Causal information used |
|---|---|---|
| `iid-uniform` | Fixed probability `6/49` for each number | None |
| `prefix-frequency` | Pseudocount-smoothed number occurrence frequency | Only preceding draws |
| `recent-ewma` | Exponentially weighted historical frequency, half-life 32, pseudocount 40 | Only preceding draws |
| `image-cnn-gru` | CNN encoding each 7×7 six-pixel frame, GRU over most recent `window` frames, 49-logit head | Only preceding draws and prefix-only training |

The CNN-GRU optimizes binary cross-entropy (unweighted, no look-ahead class
weights). A **shared logit intercept** ensures its 49 marginal probabilities
sum to six, without changing their rank. This is **not** empirical calibration
and does not specify valid six-number joint probabilities. The statistical
baselines naturally have a sum of six as well.

## Full chronological evaluation

Default canonical source: `data/bin/lotto649_images_v2.bin`. At 2,497 draws,
`--warmup 200` yields **2,297** historically eligible targets.

At target block `[t, t+B)`:

1. Fit CNN-GRU only on images strictly before `t`. Train targets are
   `window .. t-validation-1`; early stopping uses
   `t-validation .. t-1`. Neither split contains a block target.
2. Freeze weights for the entire block. Each prediction for `u` within the
   block uses image history `[u-window, u)`, which may contain *previously
   revealed* earlier outcomes from that same block. It never includes the
   unseen image at `u` or a later result.
3. Compute IID and frequency baselines with **only** `u` prior images.
4. Score the per-number marginal 49-label Brier and binary log loss,
   plus Top-6 exact number hits. Select six largest marginal probabilities,
   breaking ties by label index. IID Top-6 has no ranking and uses
   deterministic seed-only uniform six-subset sampling.

**Important:** This block-expanding scheme is a sequence of frozen-weight
models refreshed every B draws, not one fit at the beginning of 2004 and not
2,297 fully re-trained models. The history-to-target feature window remains
strictly causal for each draw. This is retrospective exploration (prior
research has seen the data), not blind proof.

The `next_forecast` field is a distinct latest-prefix fit with no unknown
next draw supplied; it contains all four models' marginal arrays and Top-6
research combinations **as of the last available draw ID**, not a verified
forecast for an already-known outcome. The report also contains all 49
probabilities, targets, history cutoffs, training block boundaries, and
source-image fingerprint. `retrospective_best_brier_skill_name` is descriptive
only and must not be used to choose the next model without independent
prospective validation.

### PowerShell / Windows / NVIDIA RTX 4060

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -v

# GPU smoke test: latest five known targets.
& $py -m lotto7x7 predict-gpu `
    --device cuda --warmup 200 --tail-targets 5 `
    --block-size 100 --window 5 --validation 32 `
    --epochs 12 --patience 3 --seed 649 `
    --out results/probability_v08_smoke.json

# All 2,297 eligible historical draws, re-fit every 100 targets.
& $py -m lotto7x7 predict-gpu `
    --device cuda --warmup 200 `
    --block-size 100 --window 5 --validation 32 `
    --epochs 12 --patience 3 --seed 649 `
    --out results/probability_v08_full.json
```

Each training block reports validation Brier and frozen historical cutoff.
The command uses CUDA if requested and available and fails closed otherwise;
it never silently claims GPU when using CPU.

### Statistical interpretation

- IID fair benchmark for one ticket: exact expected hits `36/49` and
  P(at least 3 main-number hits) =
  `sum(comb(6,k)*comb(43,6-k) for k=3..6)/comb(49,6)`.
- Better historical Brier/log loss is a **preliminary** result; evaluate
  drawn target differences with paired draw-level uncertainty. CIs in the JSON
  are raw exploratory bootstrap intervals (not adjusted for multiplicity,
  data snooping, adjacent model-training correlations, or old model selection).
- Models use all available historical data up to each cutoff but are NOT
  continuously self-updating at every target inside a block.
- Top-6 is **one** research combination per model. The earlier v0.7.2
  `N=5` 6/49 portfolio coverage numbers must not be compared as though
  they were the same metric.
- If results do not improve proper scores beyond IID or stable statistical
  controls, there is no evidence that image structure predicts the lottery.
- Data source is one third-party Biga archive, not official independent
  verification; source integrity is a separate problem.
- Prospective SHA256 snapshots with external timestamps are still needed to
  validate any apparent historical improvement.

### Next step, not shipped in v0.8 initial implementation

Determine any 1–5-ticket allocation only **after** assessing out-of-sample
probability quality and deriving a principled six-number joint approximation
or a conservative diversity baseline. Do not interpret marginals multiplied
independently as exact jackpot probabilities.
