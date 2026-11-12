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

## Observed result — November 12, 2026

The independently compiled exact integer enumerator and the eight small-pool Python-oracle regression tests **PASS** in [original GitHub Actions 43185643386](https://github.com/cctsao1008/lotto649/actions/runs/43185643386/job/131071991507). After freezing the real-history counts as CI regressions, see [final run 43185875376](https://github.com/cctsao1008/lotto649/actions/runs/43185875376/job/131072765444) for the final verdict.

### Exactly enumerated finite-pool opportunity set

| Constraint gate | Exact count |
| --- | ---: |
| Primary-optimal slice | 893 distinct candidates in 4 Shape partitions (485/123/85/200) |
| All one-per-Shape-partition four-candidate tuples | **1,014,135,000** |
| Four tickets with 24 pairwise distinct numbers | **3,853,084** |
| Also satisfying all three Sum/Log/Joint rank totals >= 43,549 | **2,059** |
| Among them, tied at highest (TotalRank, JointRank) | **1** |

The exact-integer search enumerates every disjoint candidate set after pruning only impossible prefixes with overlapping ball numbers. Each feasible quadruple is evaluated with all original rank constraints. No LP, MIP, floating-point tolerances, random restarts, or iterative heuristics are used in this enumeration.

### Final objective comparison

| Rank total | Original M0 Primary | Exhaustively optimal portfolio |
| --- | ---: | ---: |
| Sum | 47,199 | 47,031 |
| Log | 49,240 | 49,240 |
| Joint | 49,829 | 52,895 |
| Shape | 43,549 | 43,549 |
| Primary Maximin | **43,549** | **43,549** |
| Secondary Total Rank | **189,817** | **192,715** |
| Final Joint Tie Break | **49,829** | **52,895** |

Only one feasible four-ticket set reaches the best full lexicographic quality tuple. In canonical ascending ticket order (research-only):

```text
11 26 29 32 42 46
14 20 25 33 45 49
18 21 28 35 40 44
13 19 31 37 43 47
```

The result agrees with the previous v1.4 numerical MIP optimum (`Total=192715; Joint=52895`) and upgrades the conclusion to an **exact-integer exhaustive finite-pool certificate**.

The original candidate pool SHA256 is `366af606a178173d159198fef784a779b8ed430d52babc89d526f1df575f216e`; the losslessly reduced 893-row pool SHA256 is `9fad6e315f6889b87f7c966e3e019b6924820cab4e2a414341d8957397d2cac6`; canonical input stream SHA256 is `75e42d2e06356b8d79c2a3ec63a8e5e33347f4bdc459d0f7ecbb74776708e0eb`; C++ enumerator source SHA256 is `838c46c85d0a3b519bda7b3bf4bc531d957a704bc44b2cb5b091bb1850ed7786`. Archive name: `strategy-v15-exact-m0-lexicographic` (original artifact ID `13927019882`, 30-day retention); original report file `strategy_v15_m0_exhaustive.json`.

**Final research decisions:**

- `STRATEGY_EXECUTION_PASS`: original M0 deterministically followed its documented bounded heuristic.
- `PRIMARY_OPTIMALITY_PASS`: original M0 Primary Maximin 43,549 was mathematically optimal for the frozen 2,000-pool.
- `FULL_LEXICOGRAPHIC_ORIGINAL_FAIL`: original M0 Primary did not maximize its secondary rank total. A legal witness achieves +2,898 more.
- `FULL_LEXICOGRAPHIC_ALTERNATIVE_EXHAUSTIVE_PASS`: the unique alternate portfolio maximizes the entire declared integer rank tuple for the **same fixed 2,000-pool**.

This is not evidence of future-draw predictability or financial advantage. The old M0 sealed experiment, original P3f/P5A seals and user's paid four-line subscription remain unchanged.
