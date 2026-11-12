# v0.31 H2 — Observed Incremental-Geometry Ablation Results (2026-11-12)

Tracking [#31](https://github.com/cctsao1008/lotto649/issues/31) and [#32](https://github.com/cctsao1008/lotto649/issues/32).

**Implementation:** [5e54d1d](https://github.com/cctsao1008/lotto649/commit/5e54d1d0cfcdbd9af8e377bdab01bc1240dea80e); minimal-dependency CI adjustment [982afb8](https://github.com/cctsao1008/lotto649/commit/982afb83a144c3140ee54c37822914e140e375c4). Regression/real-history GitHub Actions run [43148547133](https://github.com/cctsao1008/lotto649/actions/runs/43148547133) **PASS**, 10/10 H2 tests, verified 2,498-draw source gate, 200 targets and 39 complete IID histories. [Optimized-dependency rerun 43148753703](https://github.com/cctsao1008/lotto649/actions/runs/43148753703) also **PASS**. Original detailed report: artifact `v031-h2-incremental-geometry-audit` (ID `13913205331`, 30-day retention); stored filename `v031_h2_ci_pilot.json`.

## Predeclared primary endpoint

Every arm forecasts the same **49 next-draw main-number inclusion marginals**, projected into the legal six-of-49 hypersimplex. The primary improvement is

```text
Delta = mean Brier(M0_compressed) - mean Brier(Combined).
```

A positive Delta supports incremental geometry in this **specific fixed ridge-lag model**, not necessarily the original M0 four-ticket selector or improved lottery odds. Historical endpoints were already explored, so every result is `EXPLORATORY_ONLY`. The M0-derived input is a 13-dimensional low-rank linear encoding of frozen L1/L2/L3, exact Sum, exact integer LogBin, and Sum*LogBin—not the original v0.28 four-block maximin optimizer.

### Observed scores

| Model | 49-number mean Brier (lower is better) | Mean per-number Bernoulli log loss (lower is better) | Observed mean Brier improvement over analytical IID |
| --- | ---: | ---: | ---: |
| Analytical Fair IID | **0.107455227** | **0.371776180** | 0 |
| M0 compressed (13 features) | 0.107638153 | 0.372725064 | -0.000182926 |
| Geometry (4 features incl. intercept) | 0.107687909 | 0.372921510 | -0.000232682 |
| Combined (16 features) | 0.107852217 | 0.373732320 | -0.000396990 |

`Delta_observed = -0.00021406446688421177`: the combined model was **worse**, not better, than the M0-derived compressed feature model at the same predeclared regression strength.

### IID whole-pipeline Monte Carlo calibration (R=39)

| Predeclared improvement (positive is favorable) | Observed improvement | IID null mean improvement | Empirical one-sided p |
| --- | ---: | ---: | ---: |
| M0 - Fair IID | -0.000182926 | -0.000368267 | 0.100 |
| Geometry - Fair IID | -0.000232682 | -0.000313655 | 0.225 |
| Combined - Fair IID | -0.000396990 | -0.000565027 | 0.150 |
| Combined - M0 (primary) | **-0.000214064** | **-0.000196760** | **0.575** |

The fully matched familywise maximum of null-centered, null-SD-standardized contrasts has empirical `p=0.200`. With 39 histories, the smallest resolvable p-value is `1/40=0.025`. These null means are negative because ridge fitting and feature complexity can degrade proper scores even on completely fair histories; comparing **observed score** only to zero ignores training/selection penalties.

### Time stability (descriptive only)

Across the 200 fixed chronological target origins (`500,510,...,2490` zero-based), the combined model improved on M0 at only **77/200** origins. Nonoverlapping groups of 50 **evaluated target origins** (not independent data splits) gave these mean combined-minus-M0 advantages:

| Target-origin group | Mean `Brier(M0)-Brier(Combined)` |
| --- | ---: |
| First 50 | +0.000117696 |
| Second 50 | -0.000433009 |
| Third 50 | -0.000133799 |
| Last 50 | -0.000407146 |

The historical model was already selected and explored, and fitting windows overlap. **Do not** treat the one positive group as a separately preregistered discovery, or treat 77/200 as an independent binomial trial.

## Exact fair-IID conditional proper-score identity

For each fixed main-number label `i`, let `Y_i` be 1 when the number is present in the next draw, `q=6/49`, and `p_i(H)` be any numerical forecast measurable from past history `H`. Under *independent fair draws*, `E[Y_i | H] = q`, so

```text
E[(Y_i - p_i(H))^2 | H] = q(1-q) + (p_i(H) - q)^2.
```

Averaging over 49 labels yields the strictly proper marginal-Brier benchmark. Therefore, **under the fair IID hypothesis** no history-only predictor can have better *expected* Brier score than the constant `q`. The nonnegative squared-distance penalty is exact, not a claim that the real drawing machine has been physically proven fair. This distinction is why fitted historical models can have lower descriptive scores in isolated samples but still lack a genuine prospective mechanism.

## Audit / interpretation boundaries

- Audited archive: 2,498 draws through **2026-11-10/#115000104**, 23 annual CSVs, history/image BINs and manifest consistent. Manifest SHA256 `003ed03eda889952654b679859f2d465ffb7349c7f005a6df3aa4436be3d54a3`. History BIN SHA256 `57de2ead78147f998b7cac22dfbd94461a6af38584ee106d6a1a26c1dd97f3b8`. This is not an independent official certification of every historical draw.
- Fixed config SHA256: `e9c05b351eb191b7a40377bd621c46c8f96ae85ca93b9262617fe536e23c4a33`.
- All fitted arms use a **lag-1, expanding ridge alpha=64, no-post-hoc-tuning** model. A larger combined input has more degrees of freedom than each small model, and may incur an overfitting penalty.
- This test's **per-number marginal** proper score is not the same as a proper joint probability score on the entire 13,983,816-ticket space; no profit or actual lottery-bet recommendation follows.
- Earlier fixed v0.28 M0, P3f and P5A seals, user's purchased tickets, and historical data remain untouched. No new seal, paid ticket, or forecast winner is generated.

**Scientific decision: `RESEARCH_ENGINEERING_PASS / INCREMENTAL_GEOMETRY_PREDICTIVE_NO_GO_EXPLORATORY`.** Do not tune more features against these inspected endpoints. Next independent evaluation should use the original #32 source-verified P3f/P5A future-only gates as their predeclared draw dates actually occur; use new pre-outcome preregistration before claiming an independent H2 predictive signal.
