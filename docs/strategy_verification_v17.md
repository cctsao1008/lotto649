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
