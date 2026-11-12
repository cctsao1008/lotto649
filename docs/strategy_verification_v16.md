# Strategy Verification v1.6 — Candidate Pool Scaling & Optimization Quality

## Research question

The original v0.28 M0 four-objective strategy was **executed correctly**, and v1.5 proved the exact full lexicographic optimum **within a fixed 2,000-ticket pool**. This experiment asks what happens as the candidate pool grows, **without changing the underlying scoring rule or making predictions about future draws**.

The comparison fixes the **historical cutoff (2,497 draws through 2026-11-06)**, the original 80,000 sampled six-number tickets, the 13,704 qualified positive-score candidates, the exact Shape/Sum/Log/Joint score tables and their **13,704-candidate percentile ranks**.

## Predeclared nested candidate budgets

| Arm | Candidate size | Construction |
| --- | ---: | --- |
| A | 2,000 | Original SHA-pinned frozen M0 search pool |
| B | 4,000 | Original 2,000 + 2,000 ranked additions |
| C | 8,000 | Arm B + 4,000 ranked additions |
| D | 13,704 | All original qualified positive-score candidates |

Candidates not in the original 2,000 are ordered deterministically by descending minimum of the 4 original percentile ranks, descending rank total, descending Shape rank, then ascending six-number ticket tuple. **The fixed 2,000 candidates remain the first 2,000 of every larger pool, regardless of score.** This is one deliberately defined nested expansion policy, not evidence that the expansion order is universally best. Other pool designs require separately registered comparisons.

Never re-rank, normalize or re-select the theoretical bins as the pool grows. Otherwise a numerical improvement could be caused by a shifting unit of measurement instead of candidate opportunity.

## What is verifiable?

All arms share four hard requirements: four distinct legal tickets, 24 different numbers, four different exact Shape partitions, and the original four-dimensional rank quality tuple:

```
(min(SumRank, LogRank, JointRank, ShapeRank),
 SumRank + LogRank + JointRank + ShapeRank,
 ShapeRank, JointRank)
```

For each pool:

1. Source and original selection replay pass the preexisting v1.2 **152-check independent gate**, including historical prefix hash and the original 80,000-candidate SHA.
2. The original 2,000-pool SHA matches `366af606a178173d159198fef784a779b8ed430d52babc89d526f1df575f216e`. The original **v1.5 exactly certified optimum** (`Maximin=43,549; Total=192,715; Joint=52,895`) must still be legal in this unchanged prefix.
3. Deterministic one-ticket-swap coordinate ascent and optional time-bounded numerical MILP supply **legal integer feasible lower bounds** for the primary objective. Every MILP witness is independently checked against hard constraints; numerical solver optimality is labeled separately from exact integer certification.
4. `shape_partition_bound` gives an integer-safe primary upper bound; a second independently checked exact integer Lagrangian dual (v1.3) may narrow the upper bound. **If upper == witnessed lower**, primary optimality is certified. If upper > lower, report `BOUNDED_GAP_UNRESOLVED`, not a FAIL of the original selection rule.
5. The complete lexicographic optimum for 2,000 is already certified by v1.5. When the primary gap is zero for larger pools, the lossless four-partition reduction (v1.4) first restricts secondary optimization to **only** candidates capable of reaching the proven primary maximum. A bounded MILP then maximizes **Total Rank**, followed by **Joint Rank** at the best certified Total. Both numerical solver statuses and independent integer Lagrangian upper bounds are recorded. A solver-reported zero Gap is **numerical-only**, not an independent exact integer proof. All found tickets and rank totals are independently integer-verified.
6. All outputs write only to a fresh ignored `results/` JSON file; no tickets are sold, no original sealed files or source archives are changed.

**Interpretation:** A growing *nested* candidate pool can only increase the true optimum value. However, a bounded search's found incumbent may remain unchanged or improve non-monotonically *unless the algorithm carries its previous incumbent forward*. This experiment carries the previous best full lexicographic incumbent forward and independently checks monotonicity, including the secondary objective. A larger pool alone does not increase any ticket's true fair-draw probability.

## Commands

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_strategy_verification_v16.py' -v
if ($LASTEXITCODE -ne 0) { throw 'v1.6 tests failed' }
& $py -m lotto7x7.strategy_verification_v16 `
    --csv-root data/csv --bin-root data/bin `
    --mip-seconds 6 `
    --out results/strategy_v16_nested_pool_scaling.json
if ($LASTEXITCODE -ne 0) { throw 'v1.6 audit failed' }
```

## Decision rules and limits

- **Engineering PASS** requires the original 152-check source gate, nested pool SHA reproducibility, stable original rank vectors, lawful integer witnesses, monotone best-found quality, and valid upper bounds.
- **Exact primary PASS** for an arm is only when `primary_rigorous_upper == best_feasible_witness.min_rank`.
- **No exact claim** follows from a time-limited search, a positive scoring gain, or lack of a better local swap.
- Arm D spans all 13,704 **original qualified** candidates, not all 80,000 raw sampled tickets, nor the full 13,983,816 possible single tickets. Original qualification rules and historic sampling remain fixed.
- This is strategy-quality/compute-budget research, **not** a prospective predictive test or new purchase recommendation.
