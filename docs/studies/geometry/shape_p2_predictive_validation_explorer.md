# Lotto649 P2 Batch 3 — v0.17 Predictive Validation + v0.18 Shape Explorer

## Scope and requirements

Implement **the entire P2 batch in one invocation**, using the existing P1
frozen graph classification and exhaustive fair C(49,6) category distribution.
Historical P0/P1 files are treated as immutable inputs. This release adds
no neural-network retraining and no historical pairwise distance recomputation.

* `shape_predictive_v017.py`: no-future-information sequential validation.
* `shape_explorer_v018.py`: one-file, offline, interactive six-number queries,
  exact D4+centroid-translation optimal assignment (same as v0.13), live
  7×7 bitmaps, deterministic topology and exact fair category probabilities.
* `shape_p2_batch_cli.py`: one command; JSON, two CSVs, HTML statistical
  report, offline HTML Explorer, NPZ audit and future-only hypothesis freeze.

## Fair-class models and online protocol

Each chronological target draw `t` follows strictly after `draws[:t]`.
For each `level` in `L1 (6), L2 (11), L3 (94)`, prior weights
`p_k = exact_count_k / 13983816` are fixed and never fitted on the history.
Empirical counts `n_{t,k}` use the first `t` outcomes. With fixed
`tau=384` the online one-step predictive class probability is

`q_{t,k}=(n_{t,k}+tau*p_k)/(t+tau)`.

The target's **proper** log-score improvement against fair iid is
`log(q_{t,y_t}/p_{y_t})`, measured in nats and bits. These are **class**
probabilities and *not* calibrated six-ticket odds. The full cumulative
ratio is a prequential likelihood ratio for this one fixed rule.
Its value is meaningful as an exploratory diagnostic, but because P0/P1
features and rules were chosen after examining these same data, any
retrospective significance is *not prospective independent evidence*.

The simulation null independently generates IID sequences of 2,497
**nested graph category labels** according to the **exact L3 fair masses**,
then maps each L3 label deterministically to L2 and L1. On each independent
simulation, all three online models are refitted and evaluated with precisely
matching `min_train`, `tau` and candidate levels. The report provides
simulation-tail p values and a simple three-test Bonferroni correction.

`min_train=500` yields `1997` chronological historical evaluations. Changing
a target value must not change that target's forecast. This property is unit
tested and no full-history KDE or exact-shape family labels enter forecasts.

## Candidate-ticket experiment (NOT a recommended strategy)

For each target draw, generate a fresh independent legal 6/49 pool of
`candidate_pool=64` *distinct* six-number combinations. Compare exactly
`5` tickets per draw for both:

1. Uniform: first 5 unbiased tickets in the shared pool.
2. Shape-ranked: highest `q_L3(class)/p_L3(class)` among tickets in the **same**
   unbiased pool. The online q uses only earlier draws; no future draw is
   read when selecting tickets. Ties retain random pool order.

Metrics: the 0–6 exact overlap histogram, count of 3+, 4+, 5+, and 6
matched main numbers, best-match averages and expected exact full wins.
The 6/49 fair theoretical per-ticket overlap is
`P(hit=j)=choose(6,j)*choose(43,6-j)/choose(49,6)`.

Independent fair future 6/49 draws are simulated and compared to *fixed*
selected ticket sets. This is **conditional-on-chosen-tickets calibration**;
it is not a full refit of the adaptive strategy on independent simulated past
histories. Do not interpret an isolated tail probability as evidence of
profitable or reliable next-draw advantage, especially across multiple
exploratory strategies.

## Future-only P6_O1_D5

`P6_O1_D5` was discovered by analyzing the same full history used here.
For this reason, merely holding out later records from that history after
its discovery does **not** make them independent prospective verification.
The `*_future_watchlist.json` file freezes the exact geometry rule and the
last historical draw ID. Only draws published **after** that last observed
ID can supply future-confirmation evidence; nothing is automated here.
No special weight is applied to the P6_O1_D5 class.

## Interactive browser Explorer

Open `*_explorer.html` locally; no external CDN, account, backend or web
server. Enter six different integers from 1 to 49, separated by commas or
spaces. The browser immediately computes:

* Original 7×7 six-hot bitmap.
* Immutable L1/L2/L3 topology codes with **exact** `C(49,6)` fair class counts.
* Canonical shape key under 8 D4 transforms plus integer legal translations.
* Exact orbit size under D4 and allowed placements on a 7×7 board.
* An exhaustive nearest-neighbor scan over the historical 2,497 images:
  every pair is optimized over 8 transforms and six-point one-to-one
  assignments via an exact bitmask dynamic program (64 DP states).
* Local class filter with historical examples and descriptive KDE scores.

This is not approximate embedding-based search. Cross-check with 36 distances
from v0.13 stored `shape` matrix gave maximum absolute difference under
`6e-8` in the local Node.js JS harness. It uses the centered RMS
assignment definition; its optimal translation may be fractional, even though
orbit size uses legal integer cell placements.

**No proposed bet has an advantage under a fair IID lottery.** The output
`*_tickets.csv` lists only the *last historical backtest's pre-draw
candidates and their observed hits*. It does not contain next-draw picks.

## Command

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-p2-batch `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --min-train 500 `
    --tau 384 `
    --iid-replicates 1000 `
    --ticket-null-replicates 128 `
    --candidate-pool 64 `
    --tickets-per-draw 5 `
    --seed 649 `
    --out results/v017_v018_p2_batch.json

Start-Process .\results\v017_v018_p2_batch_report.html
Start-Process .\results\v017_v018_p2_batch_explorer.html
```

## Acceptance

Development PASS requires: new P2 unit/integration tests, chronological
leakage checks, exact baseline checksum and class-mapping verification,
Node.js parity of the browser distance engine with v0.13 reference matrix,
and full 2,497 draw CPU integration producing seven new files.

Research execution PASS requires actual complete backtest results and a
reasoned conclusion versus the IID baseline, not an automatic 'predictive'
claim. Only prospectively collected draws after this research's freeze can
be independent evidence for the post-selected rare-class anomaly.
