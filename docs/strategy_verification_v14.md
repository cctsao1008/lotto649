# Lotto649 Strategy Verification v1.4 — M0 Lexicographic Tie-Break Audit

## Research question and frozen baseline

The archived v0.28 M0 experiment selects **four disjoint six-number tickets with four different exact Shape Partition classes**, optimizing the **balanced** lexicographic quality tuple

```text
(min(SumRank, LogRank, JointRank, ShapeRank),
 SumRank+LogRank+JointRank+ShapeRank,
 ShapeRank,
 JointRank).
```

All ranks are empirical integer positions within the original 13,704 four-positive-qualified sampled candidates; they are **not calibrated draw probabilities**. The bounded pool is the original 2,000 tickets, drawn from the original 80,000 unique random sample with seed `20261110`, using only history through 2026-11-06 / issue `115000103` / 2,497 draws. No ticket choices, objectives, history, purchased portfolio, or original seals may be changed to suit the audit.

[The v1.2 independent replay](strategy_verification_v12.md) reproduced the candidate population, scores and original three search results. [v1.3](strategy_verification_v13.md) established that **43,549** is the mathematically exact *primary min-block-rank* optimum in the fixed candidate pool; it did **not** prove the original Primary was the best under the complete lexicographic tie-break.

## v1.4 exact primary-optimum reduction

Let `R_s(P)` denote the sum of four Shape ranks in portfolio `P`. With at most one ticket from each Shape Partition, the four highest per-partition maximum Shape ranks of the original 2,000-pool are `13,504`, `13,128`, `9,553`, and `7,364`. The fifth is only `6,624`, so the best four partition classes are **unique**:

```text
min(SumRank, LogRank, JointRank, ShapeRank)
     <= ShapeRank
     <= 13,504 + 13,128 + 9,553 + 7,364 = 43,549.
```

For *any* portfolio meeting primary `43,549`, **all inequalities must be equalities**. Consequently it must choose exactly one candidate from each of those four Shape partitions, **each having the maximum Shape rank for its partition**. Every other candidate is safely excluded from the *lexicographic-optimum search*, with no heuristic pruning. The verifier independently reruns the v1.2 gate and checks its SHA-pinned 2,000-candidate pool before applying this reduction.

The Shape rank sum is then constant `43,549`. Remaining conditions are four disjoint tickets and each of the Sum/Log/Joint rank sums at least `43,549`. We then maximize, in order:

1. **Total Rank Sum** (second original lexicographic criterion).
2. **Joint Rank Sum**, *subject to the proven optimum Total Rank Sum* (fourth original criterion; third Shape criterion is already constant).

A different candidate with an identical quality tuple does not by itself disprove the original objective's optimality; there can be multiple ties, and the original heuristic's deterministic ticket-order tie break is distinct from the rank-based quality objective.

## Independent certificates and finite-sample scope

For each stage, v1.4 emits two independently checkable items:

- A **feasible integer witness** from SciPy/HiGHS MIP, rechecked in integer arithmetic against each six-number ticket, four distinct Shape partitions, number-disjointness, primary threshold and optional secondary Total Rank threshold. A different ticket selection is **research-only**, not a new prediction or paid-ticket recommendation.
- An **exact scaled-integer Lagrangian upper bound** computed from LP-suggested but rounded integer multipliers; the resulting inequality is evaluated using only Python arbitrary-precision integers, not the LP solver's floating-point claim. A zero integer gap between this bound and a feasible integer witness is a mathematical certificate. In addition, report MIP solver status, best bound and numerical gap, carefully distinguishing a complete solver result from a time-limited search.

If neither an exact bound nor a numerically converged MIP proves a stage, mark it **UNRESOLVED**. If any fully legal alternative **strictly improves** the original second or fourth lexicographic criterion, report a concrete, auditable **bounded-pool counterexample** even if the optimum itself is unresolved. Original primary rank `43,549`, Total Rank `189,817`, Joint Rank `49,829`.

The v1.4 conclusion does not extend to the 80,000 raw sample beyond the final 2,000-pool, the entire `C(49,6)` universe, independently sourced real-world lottery fairness, the next draw, or monetary superiority. A correctly implemented bounded heuristic can be strictly suboptimal under a different, stronger *global bounded-pool objective* without violating its documented search procedure.

## Run the preexisting CI and archival evaluation

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_strategy_verification_v14.py' -v
if ($LASTEXITCODE -ne 0) { throw 'v1.4 regression failed' }
& $py -m lotto7x7.strategy_verification_v14 `
    --csv-root data/csv --bin-root data/bin `
    --mip-seconds 90 --out results/strategy_v14_m0_lexicographic.json
if ($LASTEXITCODE -ne 0) { throw 'v1.4 research audit failed' }
```

The report is exclusive-create under ignored `results/`; original source CSV/BIN/manifest, M0/P3f/P5A attestations, user-purchased tickets and prior research files remain unchanged. All numerical audit claims should cite a completed GitHub Actions run, not only the documented implementation.
