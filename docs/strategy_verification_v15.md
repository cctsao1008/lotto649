# Lotto649 Strategy Verification v1.5 — Exhaustive Integer Lexicographic Certificate

## Acceptance objective

Given the unchanged v0.28 M0 data cutoff (2,497 draws / 2026-11-06), unchanged seeded 80,000-candidate sample, and unchanged 2,000-ticket bounded final pool, independently certify the **entire original lexicographic quality tuple**:

```
(min(SumRank, LogRank, JointRank, ShapeRank),
 SumRank+LogRank+JointRank+ShapeRank,
 ShapeRank,
 JointRank)
```

The original M0 selected a portfolio with ranks `(47199,49240,49829,43549)`, giving primary min `43549`, secondary sum `189817`, and last Joint tie-break `49829`. v1.4 found a legal counterexample with primary `43549`, total `192715`, and Joint `52895`. That already *proves the original heuristic was secondary-suboptimal*. This new verifier seeks to independently certify the **maximum attainable values**, instead of relying on a numerical mixed-integer solver gap.

## Lossless reduction and exact scope

v1.3's independent integer Shape upper bound proves `Maximin <= ShapeRankSum <= 43549` in the original 2,000-ticket pool. All four highest Shape partitions are distinct and strictly higher than the fifth. The original M0 attains 43,549. Any portfolio that can tie this primary value must use exactly one maximum-Shape-rank candidate from each of the four top Shape partitions. v1.4 established that these **893** candidates are a **necessary and sufficient search space** for any primary-optimal portfolio when combined with the remaining disjointness and minimum-rank constraints.

The v1.5 C++17 enumerator traverses every four-way Cartesian candidate combination, pruning only a partially overlapping pair/triple that cannot possibly be extended to a 24-unique-number portfolio. Every complete nonoverlapping set is checked using native integer arithmetic for all three remaining block rank floors. The maximum `(TotalRank, JointRank)` is retained lexicographically. There are no stochastic restarts, unverified MIP optimality certificates or floating-point tolerance decisions in the exhaustive search.

The enumerator is intentionally independent of SciPy's `milp`, `linprog` and their numerical certificates. Its results are checked by Python against the SHA-audited exact candidate source, legal 6/49 rules and original rank sums. Small adversarial pools are exhaustively enumerated again in independent Python regression tests. The certified result is a *reproducible computationally exhaustive integer search*, contingent on correct code, input fingerprints, compiler and tests—not a formal proof in a theorem prover.

## Engineering and research boundaries

- `Strategy Compliance PASS` means original selection followed its documented **bounded heuristic** contract. It does not require optimal search output.
- `Primary Optimality PASS` means the original primary maximin 43,549 is rigorously optimal within the fixed 2,000-pool.
- `Full Lexicographic Optimality` asks whether an integer-feasible alternative in the same pool reaches a larger secondary or final Joint objective. An improving witness proves original suboptimality regardless of future winning outcomes.
- No scope claims about the original 80,000 sampled tickets outside the 2,000 pool or all 13,983,816 tickets. No future draw can be inferred from repair score optimization.
- The 2026-11-09 purchased 4×10 lines, existing M0 candidate manifest, historical CSV/BIN/source manifest, P3f and P5A frozen seals remain strictly read-only.

## Execute

On GitHub Actions, run the dedicated `.github/workflows/strategy_verification_v15.yml` workflow. With a C++17 compiler available locally:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_strategy_verification_v15.py' -v
if ($LASTEXITCODE -ne 0) { throw 'v1.5 exact tests failed' }
& $py -m lotto7x7.strategy_verification_v15 --out results/strategy_v15_m0_exhaustive.json
if ($LASTEXITCODE -ne 0) { throw 'v1.5 exact exhaustive audit failed' }
```

`--cxx` may be set to a compatible C++17 compiler on Windows. The JSON is write-once in the ignored `results/` folder and binds all raw source/reduced pool/C++ fingerprints.

The CI results should be recorded here with exact candidate counts, unique optimum/tie counts, and explicit checks of the earlier v1.4 numerical result **only after CI actually completes**.
