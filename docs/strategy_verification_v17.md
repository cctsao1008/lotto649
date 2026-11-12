# Lotto649 Strategy Verification v1.7 — Candidate Generation Robustness

## Research objective

Investigate whether M0's good score within the frozen 2,000-candidate pool is robust when we **replace the raw random sample**, enlarge sample budgets, and apply alternate *screening* methods. This does **not** estimate the real next-draw probability or evaluate historical purchased tickets. All findings are research about the declared objective.

## Design and comparability contract

The entire original 6-of-49 fair universe is unchanged. **Original source cutoff: 2,497 draws, 2026-11-06/#115000103.** The existing v1.2 152-check independent source/replay gate must pass first. Calculate Sum, integer-binned Log Product, Sum×Log and regularized Shape counterfactual gains from the frozen *historical* prefix exactly once.

**Do NOT re-normalize ranks on each independently sampled candidate set.** Freeze the original reference axes: the four sorted gain arrays of the original 13,704 strictly four-positive-qualified candidates from the M0 seed 20261110 80k sample. Each new candidate is rescored using the same precomputed gains and standard deviations, and ranked by its insertion position via `bisect_left(frozen_axis, new_gain)` for every block. Out-of-range (greater-than-maximum) candidates have rank 13,704. These units are cross-arm comparable **frozen empirical positions**, not probability forecasts or predictions of winning.

As an independent implementation regression, before testing any new candidate streams, v1.7 rescores **all** original 13,704 qualified candidates and requires exactly matching all four original rank vectors and Shape partitions. No source files or seals may be changed.

### Predeclared candidate generation and screening matrix

- **Three alternate deterministic uniform sampling seeds:** 649, 20261111, 20261112 (the original frozen baseline seed 20261110 is not used as a new sampling seed).
- **Three nested raw unique sample sizes per seed:** 80,000, 160,000, 320,000. Each checkpoint is based on *first encountered unique ticket*, not sorted slicing of the largest sample; the historical reference never includes the new samples.
- **Three distinct deterministic candidate-screening policies at a fixed budget of 2,000 qualified tickets:** `ranked` (descending worst-block rank, then total and Shape ranks), `shape_round_robin` (approximately equal allocation by frozen Shape partition), and `uniform_screen` (independent seeded uniform sample of the eligible rows). This yields 27 arms. These are **screening methods on a common uniform raw generator**, not three independent physical draw mechanisms.
- **Same deterministic bounded optimization** applied to each screened pool: greedy multi-start from the top six individual candidates (or configured number), followed by exact integer-vector one-ticket coordinate improvements. No past or future outcome is used to select a model. Four legal tickets must have 24 distinct numbers and four distinct Shape partitions.

## Independent acceptance gates

1. Historical source fingerprint, v1.2 replay and original reference rank table parity: mandatory PASS or abort.
2. Each seed unique, raw prefix checkpoint hash reproducible, source sampling with no replacement, raw/qualified count conservation and no post-cutoff data leakage.
3. Screened candidate pools exactly 2,000 unique tickets; scores and Shape partitions preserve immutable frozen reference semantics.
4. Any reported selection must have four 6-of-49 tickets, no duplicate numbers or Shape Partition codes; objective vector integer-recomputed after selection.
5. Each arm must report legal feasible lexicographic **lower** bound, and independent Shape-partition mathematical **upper** bound on *primary* maximin. Equality alone certifies primary optimality within that arm's screened pool. Lexicographic secondary optimality is **never certified by this bounded heuristic**.
6. Every sample and pool has SHA-256, seed, cutoff, rejection counts, reproducible budget and performance metadata; no modifications to prior experiments.

## Limitations and interpretation

A comparison of only three pseudo-random seeds cannot establish statistical robustness across all possible seeds, even if each 80k–320k sample is large. Different seeds can reuse some of the same 13,983,816 legal tickets; overlap is measured. The strategy's ranks are a constructed, frozen design metric; a higher rank says nothing about the true probability of next-draw selection. The candidate pool is screened with a fixed budget, so changing the raw sampling amount does **not** imply nested selected pools: the bound can increase or decrease across raw sizes. Also, maximizing a strategy objective does not mean its underlying statistical hypothesis is true.

The v1.7 mandate is a falsifiable, reproducible **engineering sensitivity analysis**, not a declaration of a new strategy, a purchase suggestion, or a reason to rewrite any existing experimental seal.

## Reproduce

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_strategy_verification_v17.py' -v
& $py -m lotto7x7.strategy_verification_v17 `
    --seeds 649 20261111 20261112 --raw-sizes 80000 160000 320000 `
    --pool-cap 2000 --restarts 6 `
    --out results/strategy_v17_candidate_robustness.json
```

Detailed observed CI results belong below only after the GitHub Actions workflow completes and the artifact JSON is read back. If any reference ranking is inconsistent, the experiment must report **FAIL**, not silently repair the original M0 input.

## Observed CI results — 2026-11-12

**Full source-gated CI:** [GitHub Actions run 43192449439](https://github.com/cctsao1008/lotto649/actions/runs/43192449439/job/131094917641) **PASS**. Ten of ten tests; **27 of 27** new independent-seed/screening arms passed all strategy legality and source checks. Artifact **`strategy-v17-candidate-generation-robustness`**, ID **13931068131** (30-day retention), JSON SHA256 `0925cd8a17e72d7f464519d4f779294b35e7e13301020cb52eaab250189dadf1`. File `strategy_v17_candidate_robustness.json`. Runtime for entire source audit/replay and 27 arms was **20.94 seconds** on GitHub runner.

**Integrity:** original v1.2 **152/152** checks passed; historical cutoff unchanged at 2,497 draws / 2026-11-06. New scorer exactly reconstructed all four frozen per-ticket rank vectors for all 13,704 original qualified candidates; baseline rank-axis SHA256 `18000dcc2d84b08fbcd3acf98cc56c4047fb86ad953ea108d116d02593c9034b`. No rank refit was allowed. Each new independent 80k raw sample produced between **13,478 and 13,639** qualified candidates (about 16.85%–17.05%), and each 320k sample about **54,179–54,651** qualified candidates (about 16.93%–17.08%). All test pools were fixed at 2,000 qualified candidates.

### Quality scores for each deterministic seed and raw sample size

These are **feasible bounded-search witness scores**, not full lexicographic optimum certificates. Quality tuple = `(Maximin, Total Rank, Shape Rank, Joint Rank)`.

| Seed | Raw uniform unique draws | Ranked screening: Total Rank | Shape round-robin: Total Rank | Uniform-screen: Total Rank |
| --- | ---: | ---: | ---: | ---: |
| 649 | 80,000 | **195,617** | **195,617** | 169,854 |
| 649 | 160,000 | **195,617** | **195,617** | 175,155 |
| 649 | 320,000 | **195,617** | **195,617** | 176,472 |
| 20261111 | 80,000 | 191,191 | 191,191 | 176,430 |
| 20261111 | 160,000 | **195,617** | **195,617** | 190,001 |
| 20261111 | 320,000 | **195,617** | **195,617** | 180,953 |
| 20261112 | 80,000 | 192,719 | 192,719 | 189,817 |
| 20261112 | 160,000 | **195,617** | **195,617** | 181,349 |
| 20261112 | 320,000 | **195,617** | **195,617** | 186,903 |

**Primary maximin:** all nine `ranked` and all nine `shape_round_robin` arms reached **43,549**, equaling their independently calculated valid primary Shape upper bound and thus certifying *primary optimality within those particular screened 2,000 pools*. Only one of nine `uniform_screen` arms reached primary 43,549; other uniform results were between 41,679 and 43,109, leaving a strictly positive certified primary gap.

**Secondary improvement:** frozen original 2,000-pool exact optimum was `(43,549, 192,715, 43,549, 52,895)` (v1.5–v1.6). The best independently sampled witnessed tuple is **`(43,549, 195,617, 43,549, 54,184)`**, i.e. Total Rank **+2,902**, Joint Rank **+1,289**, same primary. For ranked and shape-round-robin, **8/9** seed×budget arms had a higher feasible Total Rank than the original exact optimum. The two screeners selected *different* 2,000-candidate pools in **all nine** tests, but each independently reached the same best quality tuple within its seed×budget. This observation is about this particular deterministic search; it does not demonstrate that both screeners are always equivalent.

**Important interpretation:** The fixed-rank original sample was now shown to be a **candidate-coverage limitation**, not a cap on the underlying mathematical score achievable from different random samples. The original v1.6 result — 2,000 candidates were enough within **that specific 13,704-qualified source sample** — remains true. The new finding is complementary: sourcing an independent batch can expose candidate combinations absent from the original source, which can improve the frozen strategy score.

**No secondary/global claim:** v1.7 search is bounded greedy + coordinate ascent, not exhaustive among the new 2,000 candidates or their full raw pools. Therefore the new Total 195,617 and Joint 54,184 are **certified legal feasible lower bounds on each maximize objective**, not mathematically certified full lexicographic maxima. Observed scores happen to equal two previous non-tight LP upper bounds from a *different* fixed original pool; those previous pool-specific bounds do **not** apply to the new independently sampled pools and must **not** be reused as a global certificate.

### Engineering disposition and next actions

- **Strategy Compliance:** PASS, with all original source/rank consistency checks and new-sample conservation.
- **Primary objective:** finite **screened 2,000-pool** optimum certified in 18/18 ranked and shape-screened arms; uniform-screen often not certified.
- **Full lexicographic objective:** `FEASIBLE_BETTER_THAN_FROZEN_BASELINE / EXACT_OPTIMUM_UNVERIFIED`.
- **Candidate-generation robustness:** `EXPLORATORY_SENSITIVITY_OBSERVED`, with 3 independent seeds, not a general population significance statement.
- **Predicted lottery advantage:** not tested and not inferred.

Next work should separate **new-sample candidate coverage** from **optimizer completeness**, preferably using mathematically certified secondary upper bounds or controlled exhaustive optimality in appropriately reduced candidate pools, and extend the seeds beyond 3 if statistical stability is the goal. Any additional source/reference change needs a new versioned acceptance contract.

### Final regression-locked execution

The follow-up regression-locked [GitHub Actions run 43193017636](https://github.com/cctsao1008/lotto649/actions/runs/43193017636/job/131096830943) also **PASS**. It repeats all 10 unit tests, the full 27-arm study, the 152-check original gate, **exact rank parity for all 13,704 reference candidates**, and the predeclared deterministic outcome checks (ranked + shape-round-robin primary-optimal in all nine scenarios; ranked 8/9 secondary improvements, uniform 1/9 primary-optimal; best observed Total=195,617 and Joint=54,184). Final artifact `strategy-v17-candidate-generation-robustness` ID **13930854386**, with report `strategy_v17_candidate_robustness.json` SHA256 **`97807a1cc7909427d4d2c6869b74443110c3cd2c7caa67b427fd0506ed197580`**. The difference from the initial report hash is expected because wall-clock measurements are included in the JSON; discrete rankings, source fingerprints, pool hashes, and observed objective values are stable.
