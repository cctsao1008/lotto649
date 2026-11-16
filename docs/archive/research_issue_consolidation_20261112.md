# Lotto649 Research Consolidation and Evidence-Gated Roadmap (2026-11-12)

## Purpose

Make [#31](https://github.com/cctsao1008/lotto649/issues/31) the canonical **structural/geometric information** workstream and [#32](https://github.com/cctsao1008/lotto649/issues/32) the canonical **dynamic, predictive, and portfolio evaluation** workstream. Preserve earlier issues as traceable scientific evidence instead of re-implementing finished experiments or changing old hypotheses after outcomes.

This document is an **administrative and research audit**, not a new predictive model, a live-data update, a new ticket-selection seal, or proof of a lottery advantage. Do not change any paid-ticket record, historical source CSV/BIN/manifest, v0.28 M0 manifest, original P3f/P5A seals, or existing source hashes to conform to this roadmap.

## Data and interpretation boundaries

- Current repository-audited research dataset: **2,498 draws**, through **2026-11-10 / #115000104**; the source gate checks 23 annual CSV files, history and 7x7 image BINs and the manifest. This is **repository consistency and recorded provenance**, not independent official-source certification of every historical result.
- Older v0.25-v0.28 studies and their original immutable seals used **2,497 draws**, through **2026-11-06 / #115000103**. Do not silently substitute the newer 2,498-draw sample in a frozen older experiment.
- Fair IID for one draw means six distinct numbers sampled uniformly without replacement from 1..49; independent draws do not experience a corrective force caused by past shortages.
- Distinguish **(a) an exact mathematical or geometric property**, **(b) a historically optimizable counterfactual**, **(c) an exploratory retrospective predictive comparison**, and **(d) an independently preregistered future result**. None implies the next category automatically.
- Thousands of combinations or simulations are not thousands of independent future lottery draws. A 19-path IID p-value has 0.05 resolution; a 39-path p-value has 0.025 resolution.

## Open issue audit

| Issue | Scope and verified evidence | Disposition / remaining gate |
| --- | --- | --- |
| [#1](https://github.com/cctsao1008/lotto649/issues/1) | Original 2026-11-10 forecast attestation | **Preserve as an immutable evidence record**; do not edit its original seal or retroactively claim a pre-draw result. |
| [#8](https://github.com/cctsao1008/lotto649/issues/8) | v0.25 shape covariance, distribution-repair and four-ticket coverage; multiple 99-path IID controls reported no anomalous repair gain | **Legacy backlog**; exact high-dimensional shape/joint covariance and full Pareto comparisons remain broader than the implemented three-feature P3B covariance. No blanket "superseded/completed" claim. |
| [#9](https://github.com/cctsao1008/lotto649/issues/9) | v0.25-v0.26 HDDC, exact prize and coverage, eight chronological A/B folds on 200 draws. Observed A=22, B=25, delta=+3; 99-path matched IID one-sided p=0.36 | **Retrospective NO_GO**. Keep historical findings, but route future predictive claims to #32 and remaining portfolio/coverage constraints to #31/#32. |
| [#15](https://github.com/cctsao1008/lotto649/issues/15) | v0.27 exact Log Product, joint Sum/Log, original P2, full 35-checkpoint historical replay and originally sealed P3f ten-draw shadow; #16-#24 constituent tasks closed | **Remain open** until its original P3f schedule is fully source-verified and evaluated. First historical reported outcome was 1/10 with 9 pending; do not present this as completed prospective inference. |
| [#26](https://github.com/cctsao1008/lotto649/issues/26) | v0.28 M0 independent four-block Shape/Sum/Log/Joint candidate experiment; frozen 13,983,816 exact-universe reference; four 24-disjoint candidates, six tests and original CI PASS | **Descriptive deliverable complete; eligible for issue closure** with explicit `DESCRIPTIVE_MULTI_OBJECTIVE_HEURISTIC` / predictive NO_GO. The candidate search was finite-pool, not a certified global optimum; future predictive questions belong to #32. |
| [#31](https://github.com/cctsao1008/lotto649/issues/31) | v0.29 P0/P1 exact geometry and nulls, P2A/P2B geometric convergence, P3A strict category-collision certificate, P3B counterfactual repair, preregistered P5A nine-draw shadow | **Primary geometry / incremental-information workstream**. Proving two tickets share the same L3/Sum/Log-bin but differ in pair geometry establishes *algebraic nonidentity*, **not** nonredundant next-draw predictive information. |
| [#32](https://github.com/cctsao1008/lotto649/issues/32) | v0.30 P0 exact martingale, P1 exact Sum-conditional MC, P2A chronological score-percentile IID test, P2B exact four-ticket main/special-ball probability engine | **Primary predictive and portfolio evaluation workstream**, with prospective conclusions gated on independent future data. No evidence of higher next-draw probability to date. |

## Cross-model evidence matrix

| Question | Existing implementation / reproducible evidence | Conclusion |
| --- | --- | --- |
| Do historical residuals cause corrective next draws? | `lotto7x7.distribution_dynamics_v030`; matched 199-path fair IID: alignment p=0.450, energy p=0.255 | **Not demonstrated**. The exact martingale conditional mean of raw alignment is zero under fair IID. |
| Can exact Sum/L3/Log-bin conditioning retain geometric variation? | `lotto7x7.conditional_monte_carlo_v030`; 40,000 exact-Sum proposals, 765 fully accepted, 662 distinct | **Yes, descriptively**. Conditioning on unknown outcome features cannot be a next-draw forecast. |
| Is pair geometry determined by the existing v0.28 M0 category? | `lotto7x7.geometry_redundancy_v029`; two certified Sum=132/L3=76087329/LogBin=20 tickets have pair squared distance sums 172 vs 258 | **No (algebraic nonidentity)**, but conditional increment in predictive information remains unproved. |
| Does the previously selected compensation ranking predict actual next draws? | `lotto7x7.distribution_prediction_v030`; 400 chronological origins, 39 matched IID paths, mean percentile=0.505215, primary p=0.325, eight-arm familywise p=1.000; duplicated across Windows PCs | **Exploratory predictive NO_GO**. Retrospective data are not a fresh independent holdout. |
| Is original HDDC four-ticket model better than fair structural coverage on historical outcomes? | #9; 8-fold / 200-draw matched A/B, 99 re-trained IID histories | **No reliable advantage**; +3 observed draws, upper-tail p=0.36. |
| Can four-ticket overlap change prize-event frequencies? | `lotto7x7.portfolio_benchmark_v030` and `lotto7x7.portfolio_prize_v030`; 512-pool / 256 random controls / 19 IID histories; exact main-and-special-ball DP; [CI](https://github.com/cctsao1008/lotto649/actions/runs/43138201642) 11 tests PASS | **Yes, exact combinatorial effect**, not a predictive advantage or a demonstrated increase in full expected monetary payout. Four different tickets have jackpot probability `4/C(49,6)` regardless of overlap. |

The v0.30 P2B four comparison arms are **random, coverage, three-feature repair, and a 22-unique-number hybrid**. **They do not include the historical frozen v0.28 M0 four-block portfolio as a fifth arm.** Do not describe these distinct baselines as a direct completed M0-vs-geometry ablation.

The complete v0.30 P2B pilot reported for 512 candidates: union sizes random 19, coverage 24, repair 19, hybrid 22; repair gains -0.28449364, 0.00875997, 0.59166552 and 0.56475476 respectively. Exact fair probabilities of at least one *regular base award* (including special ball) were 0.1167288921, 0.1227314590, 0.1171221141 and 0.1200976701. These are **selection-specific** probabilities, not rates measured from future draws. Different overlap changes the probability of any win, while each four-ticket arm has the same fair expected number of prize-winning lines.

## #31 formal information boundary: a completed algebraic check

Let `x_i(C)` be the 49-dimensional six-hot indicator of a **single** legal draw, and let `w_ij` be any frozen sequential-raster pair-distance weight. A fixed region's occupancy is

```text
O_R(C) = sum_i 1(i is in R) * x_i(C).
```

This is a **linear sum of number marginals**; rearranging regional occupancies cannot supply information absent from the full 49-dimensional single-draw indicator.

A pair-distance feature is instead

```text
G_w(C) = sum_{i<j} w_ij * x_i(C) * x_j(C).
```

This is a **quadratic interaction**. For a history of `n` draws, `sum_t G_w(C_t) = sum_{i<j} w_ij * N_ij`, where `N_ij` are historical pair co-occurrence counts. The 49 *aggregate marginal counts* `N_i` do not generally determine these `N_ij`.

**Constructive counterexample under the original sequential 7x7 map:** compare two two-draw histories, each legal and using every number `1..12` exactly once:

| History | Draw 1 | Draw 2 | Sum of pairwise squared-distance features across both draws |
| --- | --- | --- | ---: |
| A | `01 02 03 04 05 06` | `07 08 09 10 11 12` | `105 + 145 = 250` |
| B | `01 03 05 07 09 11` | `02 04 06 08 10 12` | `148 + 114 = 262` |

The 49 aggregate marginal counts are **identical** (each of 1..12 occurs once, every other number zero), but the second-order geometric summary differs. This is a mathematical proof that aggregate univariate frequency summaries cannot reconstruct all pair-geometry summaries. It is **not** evidence of time dependence or predictive power.

Conversely, when conditioned on the **full original ordered six-hot draw history**, these engineered pair features are deterministic functions of that input, so their conditional information-theoretic novelty is zero: `H(geometry | complete six-hot history) = 0`. They may still help *restricted, regularized models* extract useful second-order structure from finite data. Whether they improve independent future forecasts is precisely the outstanding empirical test, with equal-model-budget and full-pipeline IID controls.

This resolves an otherwise ambiguous phrase: **feature nonredundancy relative to historical marginal summaries** is true, while **new information beyond the entire raw 6/49 history** is false. Neither claim establishes a correction force or a forecast edge.

## Consolidated next research questions

### A. #31 — Incremental geometric information, not another feature search

1. Freeze the historical v0.28 M0 categorical block definitions and v0.29 pair-geometry definitions, mapping IDs and any added transformations **before** comparing new results.
2. Distinguish deterministic algebraic nonidentity from **conditional predictive information**. A fair-draw joint distribution of classes/geometry is a structural reference; predicting unknown next-draw categories requires a separate **past-only** model.
3. Compare explicitly matched reference models: **exact fair IID**, categorical Shape/Sum/Log/Joint block, frozen three-feature geometry, and **baseline-plus-geometry**. Preserve equivalent forecast targets, calibrated six-number support, chronological cutoffs and training budgets. Report redundancy and increment in proper predictive scoring, not just the ability to separate two tickets in one category.
4. Measure model complexity and conditional sample sparsity; use transparent shrinkage/regularization rather than unbounded dimensions. Randomized candidate searches or map/weight selection must be rerun in identical whole-pipeline IID synthetic controls.
5. Record null-adjusted results as `EXPLORATORY_ONLY`; do not turn a new retrospective best arm into a sealed prospective claim.

**Acceptance:** reproducible feature-and-model comparison, source fingerprint gate, leakage tests, negative controls, multiplicity-aware uncertainty and explicit GO/NO_GO *only for the exploratory question*. No automatic CNN/GNN/GRU escalation.

### B. #32 — Source-verified prospective evaluation

1. **Reuse existing immutable seals, do not re-seal or alter them.** The original P3f ten-draw shadow is in `prospective/v027_p3f_static10_20261110_seal.json` (Git blob SHA1 `b7dec534a44b1ec08d575a4f2e48e5c3580e76d1`). The original P5A nine-draw geometry shadow begins **2026-11-13 / #115000105** and is in `prospective/v029_p5a_static9_20261113_seal.json` (Git blob SHA1 `546c7352fd0a98a42959b333e51498ef7c56cf9d`).
2. Existing `lotto7x7.prospective_evaluation_m4` enforces source archive plus independent recorded publisher evidence. A future draw must not be scored from unverified numbers, an unreviewed append preview or an already examined retrospective report.
3. Keep the source-ingestion transaction separate from evaluation; append only after the exact draw and both publisher records have been independently reviewed. The public seals themselves remain unchanged.
4. The original nine-draw P5A has predeclared fixed arms and endpoints and is **small**; do not convert a single positive realization or a post-hoc subset into a general predictive GO. Report sequentially as **PENDING** until each audited result exists, and apply only the preregistered final statistic.
5. Do not attach v0.30 P2B's purely descriptive ticket designs to the frozen P5A/P3f tests after seeing them.

**Acceptance:** explicit original-seal hash validation, source evidence, chronology, append-only outcome log, exact official prize evaluation where applicable, and appropriately limited inference. No purchased-ticket modification.

### C. Evidence housekeeping

- Keep #8 and #9 **open** for their genuinely outstanding high-dimensional and historical backlog items, but do not duplicate completed v0.30 analysis; link their new research work to #31/#32.
- Keep #15 **open** while original P3f prospective draw observations are incomplete; its closed P0-P3f subissues remain historical traceability.
- Close #26 **only** for its finished descriptive M0 deliverable and explain in the closure note that no predictive GO or global-optimum certificate is implied. The M0 manifest remains immutable.
- Maintain #1 as an unchanged public attestation record. Do not edit its body to synchronize later studies.
- Register future experimental questions as a *narrow subsection* in #31 or #32 first; create a separate issue only when there is a genuinely independent deliverable.
- Test and commit new code independently of documentation. No code is changed by this consolidation report.

## Immediate work order

1. Publish this matrix and cross-link the two active master Issues (#31, #32) plus legacy Issues (#8, #9, #15).
2. Close #26 with a strictly descriptive completion rationale and a reference to its original archived candidate manifest and CI.
3. Implement #31's **baseline-plus-geometry proper-score ablation**, including a fair-IID negative control and leakage regression, before adding more shapes or optimizing an additional ticket portfolio.
4. Continue #32's **source-verified, originally sealed P3f/P5A** evaluation as target dates actually arrive. **No draw dated after 2026-11-10 is present in the current audited 2,498-draw archive** as of this consolidation.

**Current high-level decision:** `RESEARCH_IMPLEMENTATION_PASS / PROSPECTIVE_PREDICTIVE_GO_NOT_ESTABLISHED`.
