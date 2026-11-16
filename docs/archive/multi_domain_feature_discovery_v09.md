# Lotto7x7 v0.9 — GPU Multi-Domain Periodicity Candidate Discovery

## Scope

Feature discovery first, **not a next-draw number prediction and not a betting recommendation**. This executable experiment searches mathematical representations of historical *legal six-hot 7×7 images* for unexpected recurrent periods. It intentionally **does not re-train** the already tested v0.8 CNN-GRU or re-run the 216 old unit tests. Run only the 11 new focused tests if desired.

Existing v0.8 2,297-target score audit remains the selection-performance anchor. Its Brier score was worse than the IID baseline; v0.9 should not be characterized as an improvement without new out-of-sample results.

## Installed new CLI

```powershell
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

# Optional v0.9 focused tests only:
& $py -m unittest discover -s tests -p 'test_feature_discovery_v09.py' -v

# The real discovery run; GPU preferred; 199 replications *each* null type:
& $py -m lotto7x7 feature-discover-gpu `
    --device cuda `
    --holdout-draws 500 `
    --surrogates 199 `
    --surrogate-batch 8 `
    --recent-windows 256 512 1024 `
    --z-half-lives 0 64 256 `
    --min-period 2 `
    --max-period 512 `
    --top 20 `
    --seed 649 `
    --out results/v09_feature_discovery_cuda.json
```

The output is exclusive-create JSON, does not modify the historical BIN files, and avoids previously generated v0.8 results.

## Implemented deterministic, GPU-vectorized feature bank

Each row corresponds to the latest *known* completed triplet of draws `I[t-2], I[t-1], I[t]`.

| Feature family | Channels | Mathematics / analogy |
|:--|--:|:--|
| raw | 49 | six-hot 7×7 pixel incidence |
| signed_diff | 49 | `I[t]-I[t-1]` |
| central_second_diff | 49 | `I[t]-2 I[t-1]+I[t-2]`, centered on the already known `t-1` |
| spatial_dct | 48 | non-DC 2D DCT-II coefficients |
| spatial_fft_power | 27 | log-power non-DC real 2D FFT bins |
| geometry_graph | 13 | centroid, variance, covariance, row/col entropy and occupancy concentration, adjacency counts |
| pair_motif | 15 | co-occurring pixel pairs at fixed spatial offsets |
| spatial_filter_energy | 4 | Sobel and Laplacian differential filter energies |
| **Total** | **254** | Feature channels are exploratory and correlated; some duplicates are expected. |

The number-to-grid mapping is human-defined and does **not** represent physical lottery machinery; discoveries in spatial families require number-permutation/layout controls as future work.

## Period and Z-plane search

For each feature `f`, subtract its *within-window* mean and standardize. For each candidate recent window and exponential half-life, evaluate the finite-window transform:

\[
X_{f,t}(z)=\sum_{k=0}^{L-1}\tilde f_{t-k}z^{-k},\qquad
z=e^{\ln(2)/h} e^{j\omega}.
\]

`h=0` is a sentinel for no exponential weighting, i.e. the ordinary unit-circle DFT. Thus the search includes windows 256, 512, 1024, and full available discovery history; weighted/non-weighted finite Z spectra; and Fourier frequencies whose periods fall in `[2,512]` draws. This is discrete **draw-index** periodicity, not calendar-time periodicity.

The strength is `peak periodogram power / total selected periodogram power`. A score near 1 can be caused by true planted periodicity **or by purely mechanical differencing, window leakage, chance or an almost-degenerate feature**. No standalone peak is evidence of future predictive advantage.

## Anti-data-snooping and null controls

Search **only the prefix**: with the current 2,497 draws, the default 500-draw suffix is held out, leaving 1,997 historical draws for discovery. Prior v0.8 work has already looked at that historical suffix, so it cannot be represented as a genuinely never-examined blind test overall. All candidates selected here are exploratory.

The global discovery statistic is the **maximum** over every feature, spectral bin, window length, and Z radius. Compute the same maximum for each of two null ensembles:

1. `iid_legal_6of49`: every surrogate independently draws six unique uniform numbers out of 49 for each historical period, preserving exactly the lottery's without-replacement within-draw constraint;
2. `shuffled_historical_draws`: each surrogate randomly permutes the *whole 7×7 images* in the actual discovery prefix, preserving every observed six-number image and number frequency, but destroying chronological order.

For candidate strength `s`, family-wise conservative empirical p-value for each null is

\[
p_{\mathrm{max}}(s) = \frac{1+\#\{M_b\ge s\}}{B+1},
\]

where `M_b` is the *whole-search* maximum for null replicate `b`, and `B=199` by default. This accounts for choosing the best-looking period, feature, window, or transform among the scanned space for each null type. The resolution floor is 0.005 for 199 surrogates; finding a p-value near 0.005 would motivate more surrogates **and separate untouched-data validation**, not a claim of a forecast.

The two controls answer different questions. IID compares everything with genuinely fair 6/49 draws. Shuffle compares chronological order while retaining the observed nonuniformity and image distribution. Both must be considered; no independence between their p-values is presumed.

## What this stage does *not* do

- No CNN-GRU training, no pretrained model artifacts or Top-6 suggestions.
- No ranking of candidate six-number tickets by period peak.
- No test on the last 500 draws; they are held back for model development/validation, with aforementioned prior-exploration caveat.
- No claim that discovering a strong period makes any specific six-number combination more likely in a fair independent lottery.
- No Matrix Profile, persistent homology, Koopman/DMD, graph neural network or wavelet *yet*. These are literature-backed next-module candidates and should be added as independent feature families, with the exact same max-null and forward validation standards.

## Next stage if an interesting candidate survives both null families

1. Build `FeatureEncoder` for the selected *entire feature family*, not cherry-picked highest single feature from the suffix.
2. Compare v0.8 frozen baseline against CNN-GRU + feature fusion under the **same** chronological cutoffs; train on prefix only.
3. Run feature ablations, candidate-layout permutations, and IID synthetic re-training as a null model control.
4. Connect to `bet_selection_v06.py` only if marginal Brier and out-of-sample hit metrics show credible improvement; then evaluate 1–5 ticket legal portfolios with equal ticket counts and exact fair coverage.

## Scholarly inspirations (design inspirations, not evidence about lottery patterns)

- Fisher g / permutation search for periodicity: https://pmc.ncbi.nlm.nih.gov/articles/PMC1683571/
- Matrix Profile motif discovery in noisy time series: https://pmc.ncbi.nlm.nih.gov/articles/PMC10270338/
- Koopman DMD dynamics: https://www.mdpi.com/2076-3417/16/7/3392
- Topological Data Analysis representations: https://www.mdpi.com/2227-7390/14/12/2205
- Benjamini–Hochberg multiple testing: https://academic.oup.com/jrsssb/article/57/1/289/7035855
- Data leakage from feature selection: https://github.com/scikit-learn/scikit-learn/blob/main/doc/common_pitfalls.rst
