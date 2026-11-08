# Lotto649 v0.24 — Temporal Image & Signal Research (P0–P5)

This implementation is linked to GitHub Issues [#2](https://github.com/cctsao1008/lotto649/issues/2) through [#7](https://github.com/cctsao1008/lotto649/issues/7). The research question is **whether 7×7 six-pixel images and causal time-domain differences contain predictive information for the *next unseen* 6/49 draw**, beyond the fair IID 6-of-49 baseline. The answer must come from chronological held-out proper scores, not beautiful analog-looking interpolations or finding already-known historical tickets.

## Run on the actual repository's CSV/BIN sources (required to complete P0 verification)

The main command validates **23 flat annual date-bearing CSVs**, the immutable six-pixel `data/bin/lotto649_images_v2.bin`, and the original frozen P1 NPZ catalog, requiring row-for-row agreement (ID and six main numbers), chronological IDs and dates, valid bonus numbers and six-on pixels. All inputs are **read-only**. Only P1 universe *reference* probabilities are used; they do not encode future winning labels.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

& $py -m lotto7x7 temporal-research `
    --csv-root data/csv `
    --image-bin data/bin/lotto649_images_v2.bin `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --iid-reps 200 `
    --bootstrap-reps 1000 `
    --seed 649 `
    --out "results/v024_temporal_$stamp.json"
```

For *research only*, with catalog NPZ but without date-bearing CSVs/BIN, supply `--catalog-only` to explicitly waive the date check and report `P0.status=PARTIAL_NO_DATE_VERIFICATION`. This mode was used to perform the preliminary real-catalog integration reported below. Never describe that as CSV/BIN or independent official-source verification. Local raw data retain their original third-party provenance; data parity does **not** confirm the official lottery operator's source.

Four output files are written without overwriting existing reports:

1. `...json`: source hashes, P0–P5 metadata, IID comparisons, causal proper scores, bootstrap intervals, `NO_GO` decision;
2. `..._transitions.csv`: every adjacent pair's kept/removed/added/XOR, exact Jaccard, D4+translation best IoU, centroid/edge changes;
3. `..._features.csv`: each draw's **17 causal** image features, with missing early lags explicitly blank;
4. `..._forecast.csv`: strictly held-out next-draw target IDs and dates, observed numbers, predicted 49-number probabilities and proper per-draw Brier/Log scores.

## P0: source integrity, order and frozen splits

- `temporal_dataset_v024.py`: no implicit sorting of source dates, ID/year consistency, no duplicated IDs/dates, distinct 1..49 main numbers and special number exclusion, unique six-on binary image; catalog and BIN parity checked when present.
- Sequential 60% train / 20% validation / 20% test; on the 2,497-draw frozen catalog, `train_end=1498`, `validation_end=1997`, `test_end=2497`. A minimum 21-draw warmup prevents undefined lagged features entering the model.
- All feature calculations for forecast of draw `t` use images with index at most `t-1`.
- **Not independently verified** against the Taiwan Lottery official download. Single upstream Biga archive provenance retained.

## P1: adjacent image transitions

- `temporal_features_v024.py`: exact pixel commonality, 6-kept conservation, XOR, IoU, spatial centroid displacement, L3 graph-edge change, and maximum rotated/flipped/translational IoU over eight D4 transforms.
- The fair IID number-overlap law is **hypergeometric**: `P(K=k)=C(6,k) C(43,6-k)/C(49,6)`, average overlap `36/49≈0.734694`. Labels are arbitrary map locations.

## P2: multi-lag temporal features and IID controls

- Lagged common-number counts for 1,2,3,5,10,20 draws; 17 features total include centroid/spread, graph adjacency/components, low DCT energy, lagged overlap, centroid velocity, graph-edge change and rolling overlap means.
- `p2_iid_acf_control` regenerates entire IID six-of-49 draw streams and **recomputes the *same* overlapping rolling windows**. This is critical: a rolling five-transition average produces lag-1 ACF of roughly 0.8 *even when the underlying lottery draws are IID*.
- The simulated maximum absolute ACF across 5 preselected descriptors × 20 lags reports a familywise adjusted comparison across that particular set of tests; it is still an exploratory retrospective screen, not forecasting evidence.

## P3: spatial and temporal DSP

- `temporal_dsp_v024.py`: 2D orthonormal FFT/DCT power with Parseval checks; total energy exactly 6 and constant squared DC `36/49` for every legal six-pixel image. **Neither is evidence of an oscillatory lottery.**
- Periodogram peaks for centroid, spread, adjacent overlap, DCT energy; original sequential map, Snake, and a seeded random bijection. Max-peak test compares all 12 feature×layout searches to an IID simulation ensemble.
- The red interpolated sketch is not treated as measured time-domain analog voltage. No arbitrary continuous line interpolation appears in the model.

## P4/P5: forecast target and strict causal evaluation

- Simple regularized 49-output ridge predicts *next-draw* number weights. Initial-train-only standardization; choose alpha from `{1,10,100,1000}` based **only** on chronological validation Brier; expanding sufficient-statistics refits as *past labels become known*. No test label influences its own forecast.
- Outputs are projected to `[0,1]` and sum exactly to 6. They are evaluated as **per-number marginal Brier and Bernoulli log-loss scores**.
- Separately, a mathematically normalized six-number candidate distribution is formed as `P(S)=prod_{i in S} w_i / e_6(w)` using element weights `w_i=p_i/(1-p_i)` and an exact elementary-symmetric-polynomial DP. This provides **exact-ticket log score**, not a misleading sum of independent 49-way probabilities. Projected `p_i` are model weights and are not necessarily the normalized conditional-six distribution's exact marginals.
- Fair IID (`6/49` marginal and `1/C(49,6)` ticket) and time-only historical marginal frequencies are baselines. Frozen P1 L3 category counts and **pre-draw-only** Dirichlet class frequencies separately produce the normalized shape-only *ticket* log-score. Do not mix class probabilities with marginal Brier.
- Chronological test is **500 draws** on the frozen 2,497-row P1. Paired circular moving-block bootstrap (block=10, repetitions configurable) summarizes score difference; results are exploratory retrospective evidence. Any future integration requires independently preregistered, genuinely later draws, not more seed-mining or a changed historical split.
- The decision is `NO_GO` until independent preregistered future confirmation exists; *even if exploratory tests improve*. Results are not automatically connected to cash betting.

## Frozen P1 catalog-only preliminary integration (2,497 draws)

Configuration: catalog SHA256 `79f0c6477f8a5b6c4bde30b84658f0880abe181f255dca226ba9bc0f9ca5657f`, seeds=649/650/651, 200 IID surrogates per P2/P3, 1000 block bootstrap samples, 500 chronological test draws. **Actual CSV dates/BIN have not been validated in this runtime.**

| Diagnostic | Result | Interpretation |
|---|---|---|
| P0 | 2,497 legal chronological catalog IDs, dates not available | Provisional, full source gate still required |
| P1 | 2,496 transitions, mean common number **0.743590** | IID theoretical **0.734694**; small descriptive deviation |
| P2 | exploratory 5-series ×20-lag max-ACF IID p ≈ **0.0398** | Investigate but not evidence of predictability; window-induced correlation and exploratory selection concern |
| P3 | 12-test max-spectrum IID p ≈ **0.2935** | No demonstrated temporal spectral departure |
| P4 | selected ridge `alpha=1000`, static marginal Brier **0.107546** | Validation-only alpha selection |
| P5 | fair Brier **0.107455**, temporal Brier **0.107961** | Temporal performs **worse** |
| P5 CI | fair-minus-temporal Brier 95% block bootstrap **[-0.000692,-0.000339]** | Negative interval; lower Brier is better |
| P5 exact ticket | fair logp **-16.453411**, temporal **-16.576142**, train-only shape **-16.460968** | All observed model means worse than fair in this split |
| Decision | **NO_GO** | No predictive improvement confirmed |

The P2 exploratory borderline p-value must **not** be turned into a predictive claim: it was computed on already-inspected history and is not supported by proper next-draw test scores. It warrants investigation of lag stability and mapping robustness only.

## Regression coverage

`python -m pytest -q tests/test_temporal_v024.py` tests schema and ID/date failure cases, source catalog parity, exact hypergeometric law, P1 D4 conservation, causal future perturbation invariance, honest rolling-window IID controls, Parseval/DC invariants, mapping permutations, correctly normalized 6-ticket exact probabilities, train/val/test boundaries, no-go verdict and all four report files. In this implementation environment: **22/22 v0.24 tests PASS**; 43/43 combined v0.23+v0.24 tests PASS. The frozen catalog-only end-to-end research computation completed and was cross-checked.
