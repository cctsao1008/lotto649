# Lotto7x7 v0.10 — GPU Deep Feature Discovery

## Research question

Can a learned representation of **historically completed legal six-hot 7×7 lottery images** reveal recurrent/periodic latent structure beyond what would appear for fair, independent 6-of-49 drawings? Does the same model outperform the exact fair IID marginal on a predeclared suffix?

This is **feature research**, not evidence of a winning-number probability advantage. The map from 1..49 to the raster is a human-defined display choice. Some observed recurrent representations arise in random data. No lottery draw after a forecast target is allowed as an input to that target.

## CLI

Run in the Windows repository after updating GitHub main:

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

# Optional: only 13 NEW v0.10 tests; do not rerun 216+ old tests.
& $py -m unittest discover -s tests -p 'test_deep_feature_v010.py' -v

& $py -m lotto7x7 deep-feature-gpu `
    --device cuda `
    --train-draws 1024 `
    --holdout-draws 500 `
    --window 64 `
    --validation 128 `
    --embedding 16 `
    --hidden 24 `
    --epochs 12 `
    --patience 3 `
    --batch-size 64 `
    --surrogates 99 `
    --surrogate-batch 4 `
    --recent-windows 256 512 `
    --z-half-lives 0 64 256 `
    --motif-windows 8 16 32 `
    --min-period 2 `
    --max-period 128 `
    --seed 649 `
    --out results/v010_deep_feature_cuda.json `
    --model-out results/v010_deep_feature_cuda.pt
```

Two results are exclusively created and never overwritten:

- `results/v010_deep_feature_cuda.json`: data identity, settings, training trace, latent discovery, max-null and Bonferroni p-values, held-out Brier/Top-6 metrics, model checkpoint SHA256.
- `results/v010_deep_feature_cuda.pt`: CPU-portable `torch.save` state dictionary plus architecture parameters, training prefix length, dataset fingerprint. **It is a research checkpoint trained only on the initial 1,024 draws**, not a validated up-to-date number selector. When reloading a personally generated checkpoint use `torch.load(path, map_location='cpu', weights_only=True)`.

No need to regenerate v0.8 or v0.9 outputs. CUDA is required by `--device cuda`; explicit failure if PyTorch CUDA is unavailable.

## Split: one training fit, independent feature search, reserved evaluation

For the current 2,497 ordered draws:

| Segment | Draw indices (zero-based) | Count | Use |
|---|---:|---:|---|
| Early representation prefix | 0..1023 | 1,024 | CNN reconstruction + causal GRU next-image training, internal last-128-draw validation |
| Latent discovery segment | 1024..1996 | 973 | Frozen-model embeddings; period and nontrivial motif search only |
| Historical reserved suffix | 1997..2496 | 500 | Predeclared one-ticket Top-6 and probability Brier score; never used to fit/select discovered features |

The internal training target indices are `window..(train_draws-validation-1)`; internal validation targets are `train_draws-validation..train_draws-1`. Input for target `t` is strictly `I[t-window:t]`. At each completed discovery draw `t`, the deep feature uses `I[t-window+1:t+1]`, known after that draw; it could only be used to forecast **after** `t`. The historical suffix of 500 draws was already inspected in preceding v0.8 work, therefore **it is not an untouched prospective blind test**.

## Deep model and learned representations

```text
Known 7×7 images (causal 64-draw windows)
    -> CNN Conv2d 1→8→12 (GELU) -> 16-dimensional embedding z[t]
    -> autoencoder reconstruction decoder [16→49] (BCE)
    -> concatenate z[t], z[t]-z[t-1], trailing means 4, 16, 64 (all causal)
    -> GRU hidden 24 -> prediction head 49 logits (BCE)
    -> constrained sum-six marginal probabilities at evaluation
```

Training loss:

`next_frame_binary_cross_entropy + 0.7 * within_window_reconstruction_binary_cross_entropy`.

An AdamW optimizer (learning rate 0.002, weight decay 0.001) trains only the early prefix. The best checkpoint is selected solely using **past-only internal validation Brier** with early stopping. The model is refitted once on the initial prefix, *not* for every null surrogate and *not* on the future discovery or suffix blocks. Frozen features comprise **40 channels** = 16 CNN latents + 24 GRU states. They are not necessarily physically interpretable or useful for next-draw selection.

## Two deep-feature discovery detectors

1. **Latent spectral periodicity:** apply the existing v0.9 finite-window Z-domain/FFT scan to every frozen latent/GRU component. Search recent windows (full 973, 256, 512), exponential half-lives (unweighted, 64, 256), all periods in 2..128 draws. The observed statistic is the maximum normalized periodogram peak over the entire search. This is a *draw-index* period, not calendar-time periodicity.
2. **Matrix-profile-style repeated motifs:** for subsequence lengths 8, 16, 32 draws, find the strongest z-normalized nonoverlapping self-match in each entire CNN-embedding or GRU-state vector sequence. GPU batches form the subsequence vectors and compute pairwise cosine similarity via matrix multiplication; pairs offset less than the motif length are explicitly excluded to avoid trivial self matches. This is a nearest-neighbor subsequence motif detector, **not** the full published STOMP algorithm.

The two detector types can detect different recurring structures. Finding a latent motif is **not** finding a future next-draw probability bias.

## Controls and multiplicity

Both detectors use **99 Monte Carlo maxima per null family** (empirical p resolution 0.01). Each null replicate reruns the same spectral and motif scan, including all features and all hyperparameters searched:

- `iid_legal_6of49`: fresh independent draws of exactly 6 distinct numbers from 1..49.
- `shuffled_historical_discovery`: permute whole historical 7×7 frames within the 973-draw discovery segment, preserving actual images and frequencies but breaking their order.

For both, preserve the same final 63 prefix images as fixed causal context. Pass null sequences through the **same frozen CNN+GRU encoder**, not a retrained network. Because model weights were fitted before and independently of the discovery segment, these are valid **conditional fixed-encoder discovery controls**; they are not a full re-training pipeline null model. A retrained-IID negative control remains a next-stage experiment if a candidate survives.

For a detector and one null family, calibration is based on the replicate maxima:

`p_detector = (1 + number of surrogate-maxima >= observed_score) / (99+1)`.

For the two separate detector families, Bonferroni-correct `p_both = min(1, 2*p_detector)`. Two null families are provided separately, rather than multiplying their p-values. High observed strength alone is not a discovery.

## Historical suffix score

After the feature search, score the final 500 historical targets using the frozen model and **only their 64 previous published draws**. Report:

- constrained-marginal Brier score for model and exact IID `(6/49)*(43/49)`;
- mean paired Brier skill (`IID - model`) and descriptive draw-wise bootstrap CI;
- Top-6 overlap counts and >=3 hit rate.

This is predeclared evaluation of a representation-learning model and is **not yet model-based six-number joint ticket scoring**. It should not be compared to v0.7.2 five-ticket union hit rates. Existing v0.8 Brier results should be treated as exploratory context, not a clean blind benchmark.

## Decision rule

- If neither detector passes both null families after multiplicity control, report no periodic/recurrent feature evidence. Do **not** tune a new model on the suffix to force significance.
- If a latent feature is significant, preserve the model checkpoint and candidate feature definition. Test whether it persists on later untouched data and supports causal candidate ranking; validate against IID and randomized-layout controls before using it for selection.
- A fair IID lottery assigns equal probability to each exact six-number ticket, regardless of its image score. A feature score is **not** a mathematically justified ticket-win probability without empirical evidence of non-IID behavior.

## Limitations

CUDA PyTorch stochastic details can vary across drivers; the seed is recorded but bit-for-bit GPU deterministic training is not guaranteed. Scan p-values are conditional on one frozen model; this is an exploratory search, and the last 500 past results were previously studied. The model's 7×7 CNN may learn raster artifacts, not lottery mechanics. Historical information and any mathematical transformation cannot create an information edge when actual draws are independent and fair.
